from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import json
import pandas as pd

import unicodedata
import re
import difflib

def _canon(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _canon_team(s: str) -> str:
    s = _canon(s)
    # remove common tokens/sponsors that cause mismatches
    for tok in ["f1 team", "formula 1 team", "team", "scuderia", "gp", "grand prix"]:
        s = s.replace(tok, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _best_fuzzy(target: str, candidates: list[str], cutoff: float = 0.6) -> str | None:
    if not candidates:
        return None
    m = difflib.get_close_matches(target, candidates, n=1, cutoff=cutoff)
    return m[0] if m else None


PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from f1fantasy.ergast import fetch_all_supporting, fetch_schedule
from f1fantasy.fantasy_api import fetch_players, fetch_teams
from f1fantasy.model import (
    compute_weekend_points,
    expected_scores_horizon,
    apply_no_negative_expectation,
    _horizon_weights,
)
from f1fantasy.optimize import optimize_top_k
from f1fantasy.transfers import best_two_transfer_move


def _upcoming_circuits(schedule: pd.DataFrame, today: str, n: int = 5) -> list[str]:
    # schedule has date yyyy-mm-dd
    sch = schedule.copy()
    sch["date"] = sch["date"].astype(str)
    upcoming = sch[sch["date"] >= today].sort_values("round").head(n)
    # Use circuitName keywords; keep short tokens to match varying naming
    return [c.split(" Circuit")[0].strip() for c in upcoming["circuitName"].astype(str).tolist()]

def _load_current_team(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def main():

    current_season = datetime.utcnow().year
    today = datetime.utcnow().date().isoformat()

    # Fetch history window + current season for live updating as races happen
    start_year = current_season - 9
    end_year = current_season

    # Preload per-year supporting data
    all_results = []
    all_quali = []
    all_sprint = []
    for y in range(start_year, end_year + 1):
        d = fetch_all_supporting(y)
        all_results.append(d["results"])
        all_quali.append(d["qualifying"])
        all_sprint.append(d["sprint"])

    results = pd.concat(all_results, ignore_index=True)
    qualifying = pd.concat(all_quali, ignore_index=True) if any(len(x) for x in all_quali) else pd.DataFrame()
    sprint = pd.concat(all_sprint, ignore_index=True) if any(len(x) for x in all_sprint) else pd.DataFrame()


    # --- Enforce the intended season window even if cache contains older seasons ---
    results = results[(results["season"] >= start_year) & (results["season"] <= end_year)].copy()
    if not qualifying.empty:
        qualifying = qualifying[(qualifying["season"] >= start_year) & (qualifying["season"] <= end_year)].copy()
    if not sprint.empty:
        sprint = sprint[(sprint["season"] >= start_year) & (sprint["season"] <= end_year)].copy()
    # Live fantasy roster/prices
    players = fetch_players()
    teams = fetch_teams()

    # Horizon: next 5 races (weighted), using Ergast schedule
    schedule = fetch_schedule(current_season)
    upcoming = _upcoming_circuits(schedule, today=today, n=5)
    h_w = _horizon_weights(len(upcoming), w1=1.0, w_next=0.7)

    print("\nUpcoming circuits (horizon):", upcoming)
    print("Horizon weights:", h_w, "\n")

    weekend_points = compute_weekend_points(
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        current_season=current_season,
        last_season_weight=0.95,
        older_decay=0.75,
        race_dnf_penalty=20,
        sprint_dnf_penalty=10,
    )

    # Expected horizon scores (drivers + constructors) using circuit-aware history
    drv_exp, con_exp = expected_scores_horizon(weekend_points, upcoming, h_w)

    # No Negative expected for drivers (approx): floor each weekend score at 0 and re-take horizon expectation
    nn_driver = apply_no_negative_expectation(weekend_points, upcoming, h_w)
    drv_exp = drv_exp.merge(nn_driver.rename("nn_exp_score"), on="driverId", how="left")
    drv_exp["nn_exp_score"] = drv_exp["nn_exp_score"].fillna(drv_exp["exp_score"])


    # Join with live fantasy roster/prices using robust keys (avoid name-only merge)
    # Baselines for unseen rookies / new teams:
    drv_baseline = float(drv_exp["exp_score"].min()) if len(drv_exp) else 0.0
    drv_dnf_baseline = float(drv_exp["dnf_rate"].max()) if "dnf_rate" in drv_exp.columns and len(drv_exp) else 0.25
    ctor_baseline = float(con_exp["exp_score"].min()) if len(con_exp) else 0.0
    ctor_dnf_baseline = float(con_exp["dnf_rate"].max()) if "dnf_rate" in con_exp.columns and len(con_exp) else 0.25

    # ---- Drivers mapping: try exact name (accent-insensitive), then last-name unique, then fuzzy ----
    fp = players.copy()
    # Build canonical names from fantasy feed. Prefer FirstName/LastName if present.
    if "FirstName" in fp.columns and "LastName" in fp.columns:
        fp["canon_name"] = (fp["FirstName"].astype(str) + " " + fp["LastName"].astype(str)).map(_canon)
    else:
        fp["canon_name"] = fp["name"].astype(str).map(_canon)

    # Ergast model names
    drv_exp = drv_exp.copy()
    drv_exp["canon_name"] = drv_exp["driver"].astype(str).map(_canon)

    # last-name index for fallback
    drv_exp["canon_last"] = drv_exp["canon_name"].str.split(" ").str[-1]
    last_to_rows = drv_exp.groupby("canon_last")["canon_name"].apply(list).to_dict()
    canon_to_row = drv_exp.set_index("canon_name")[["driverId","exp_score","dnf_rate","volatility","nn_exp_score"]].to_dict("index")

    def map_driver_row(cname: str):
        # 1) exact
        if cname in canon_to_row:
            return canon_to_row[cname]
        # 2) substring containment (handles 'kimi antonelli' vs 'andrea kimi antonelli')
        for k in canon_to_row.keys():
            if cname and (cname in k or k in cname):
                return canon_to_row[k]
        # 3) last name unique
        last = cname.split(" ")[-1] if cname else ""
        cands = last_to_rows.get(last, [])
        if len(cands) == 1:
            return canon_to_row[cands[0]]
        # 4) fuzzy
        best = _best_fuzzy(cname, list(canon_to_row.keys()), cutoff=0.72)
        if best:
            return canon_to_row[best]
        return None

    mapped = fp["canon_name"].apply(map_driver_row)
    drivers = fp.copy()
    drivers["driverId"] = mapped.apply(lambda x: x["driverId"] if isinstance(x, dict) else None)
    drivers["exp_score"] = mapped.apply(lambda x: x["exp_score"] if isinstance(x, dict) else None)
    drivers["dnf_rate"] = mapped.apply(lambda x: x["dnf_rate"] if isinstance(x, dict) else None)
    drivers["volatility"] = mapped.apply(lambda x: x["volatility"] if isinstance(x, dict) else None)
    drivers["nn_exp_score"] = mapped.apply(lambda x: x.get("nn_exp_score", x["exp_score"]) if isinstance(x, dict) else None)

    # fill rookies / missing with baseline (bottom driver from last season)
    drivers["exp_score"] = pd.to_numeric(drivers["exp_score"], errors="coerce").fillna(drv_baseline)
    drivers["dnf_rate"] = pd.to_numeric(drivers["dnf_rate"], errors="coerce").fillna(drv_dnf_baseline)
    drivers["volatility"] = pd.to_numeric(drivers["volatility"], errors="coerce").fillna(pd.to_numeric(drivers["volatility"], errors="coerce").median())
    drivers["nn_exp_score"] = pd.to_numeric(drivers["nn_exp_score"], errors="coerce").fillna(drivers["exp_score"])

    drivers.rename(columns={"playerId": "id"}, inplace=True)

    # ---- Constructors mapping: alias current fantasy names to Ergast/Jolpica historical constructor names ----
    ft = teams.copy()
    ft["canon_team"] = ft["name"].astype(str).map(_canon_team)

    con_exp = con_exp.copy()
    con_exp["canon_team"] = con_exp["constructor"].astype(str).map(_canon_team)

    # Aliases based on 2025 Jolpica constructor names (e.g., 'RB F1 Team') and 2026 team branding.
    # Sauber becomes Audi factory team in 2026. citeturn0news19
    TEAM_ALIAS = {
        "red bull": "red bull",
        "red bull racing": "red bull",
        "mclaren": "mclaren",
        "mercedes": "mercedes",
        "ferrari": "ferrari",
        "williams": "williams",
        "aston martin": "aston martin",
        "haas": "haas",
        "haas f1": "haas",
        "haas f1 team": "haas",
        "alpine": "alpine",  # maps to 'alpine f1'
        "alpine f1": "alpine",
        "alpine f1 team": "alpine",
        "racing bulls": "rb",      # fantasy name -> 2025 constructorId 'rb' / 'RB F1 Team'
        "rb": "rb",
        "rb f1": "rb",
        "rb f1 team": "rb",
        "audi": "sauber",          # treat as Sauber history in 2025
        "sauber": "sauber",
        "kick sauber": "sauber",
        "cadillac": None,          # new team, no history -> baseline
    }

    # Build candidate mapping dict for con_exp
    con_keys = con_exp.set_index("canon_team")[["constructorId","exp_score","dnf_rate","volatility"]].to_dict("index")

    def map_constructor_row(cteam: str):
        base = TEAM_ALIAS.get(cteam, None)
        if base is None:
            return None
        # try exact match on canon_team of con_exp (which may be 'rb f1' simplified to 'rb')
        # also allow matching by containment
        for k, row in con_keys.items():
            if base == k or (base and (base in k or k in base)):
                return row
        # fuzzy fallback
        best = _best_fuzzy(base, list(con_keys.keys()), cutoff=0.65)
        return con_keys.get(best) if best else None

    mapped_c = ft["canon_team"].apply(map_constructor_row)
    constructors = ft.copy()
    constructors["constructorId"] = mapped_c.apply(lambda x: x["constructorId"] if isinstance(x, dict) else None)
    constructors["exp_score"] = mapped_c.apply(lambda x: x["exp_score"] if isinstance(x, dict) else None)
    constructors["dnf_rate"] = mapped_c.apply(lambda x: x["dnf_rate"] if isinstance(x, dict) else None)
    constructors["volatility"] = mapped_c.apply(lambda x: x["volatility"] if isinstance(x, dict) else None)

    # Fill new teams (Cadillac) as bottom constructor from last season
    constructors["exp_score"] = pd.to_numeric(constructors["exp_score"], errors="coerce").fillna(ctor_baseline)
    constructors["dnf_rate"] = pd.to_numeric(constructors["dnf_rate"], errors="coerce").fillna(ctor_dnf_baseline)
    constructors["volatility"] = pd.to_numeric(constructors["volatility"], errors="coerce").fillna(pd.to_numeric(constructors["volatility"], errors="coerce").median())

    constructors.rename(columns={"teamId": "id"}, inplace=True)

    # ================== TEAM-STRENGTH ADJUSTMENT (DRIVERS) ==================
    # Driver historical EV is team-context dependent. Adjust driver exp_score using current fantasy team strength
    # inferred from constructor expected scores. This prevents unrealistic cases like a top driver priced as a bottom-team asset.
    ctor_exp_by_name = constructors.set_index("name")["exp_score"].to_dict()
    drivers["team_exp"] = drivers["team"].map(ctor_exp_by_name)

    team_exps = constructors["exp_score"].astype(float)
    p10 = float(team_exps.quantile(0.10))
    p90 = float(team_exps.quantile(0.90))

    def _team_factor(te: float) -> float:
        # neutral if missing
        if te is None or pd.isna(te) or p90 <= p10:
            return 1.0
        # bottom teams get heavy penalty, top teams get mild boost
        if te <= p10:
            return 0.35
        if te >= p90:
            return 1.15
        return 0.35 + (te - p10) * (1.15 - 0.35) / (p90 - p10)

    drivers["team_factor"] = drivers["team_exp"].apply(_team_factor).astype(float)

    # keep raw for inspection
    drivers["exp_score_raw"] = drivers["exp_score"].astype(float)
    drivers["nn_exp_score_raw"] = drivers["nn_exp_score"].astype(float)

    drivers["exp_score"] = drivers["exp_score_raw"] * drivers["team_factor"]
    drivers["nn_exp_score"] = drivers["nn_exp_score_raw"] * drivers["team_factor"]
    # ========================================================================
# === OUTPUT: Top teams for different chip settings ===
    def print_solutions(title: str, sols):
        print(f"\n{title}\n")
        for i, sol in enumerate(sols, start=1):
            dd = sol.drivers.sort_values('price', ascending=False)[['name','price','exp_score','dnf_rate']]
            cc = sol.constructors.sort_values('price', ascending=False)[['name','price','exp_score','dnf_rate']]
            print(f"--- Team #{i} ---")
            print(f"Cost: {sol.total_cost:.1f}/100 | Expected: {sol.expected_score:.2f} | Boosted: {sol.boosted_driver} | NoNegative: {sol.no_negative} | Limitless: {sol.limitless}")
            print("Drivers:\n", dd.to_string(index=False))
            print("Constructors:\n", cc.to_string(index=False))
            print()

    # Standard (2x boost)
    sols_std = optimize_top_k(drivers, constructors, budget=100.0, k=5, drs_multiplier=2.0, allow_no_negative=False)
    print_solutions("Top 5 teams (2x Boost)", sols_std)

    # 3x DRS variant (some seasons call this 'Extra DRS') – we show it as a scenario
    sols_3x = optimize_top_k(drivers, constructors, budget=100.0, k=3, drs_multiplier=3.0, allow_no_negative=False)
    print_solutions("Top teams if 3x Boost is active", sols_3x)

    # No Negative chip scenario
    sols_nn = optimize_top_k(drivers, constructors.assign(nn_exp_score=constructors["exp_score"]), budget=100.0, k=3, drs_multiplier=2.0, allow_no_negative=True)
    print_solutions("Top teams with No Negative chip", sols_nn)

    # Limitless chip scenario (no budget)
    sols_lim = optimize_top_k(drivers, constructors, budget=None, k=3, drs_multiplier=2.0, allow_no_negative=False)
    print_solutions("Top teams with Limitless (no budget)", sols_lim)

    # === Transfer-aware suggestion ===
    cfg = _load_current_team(PKG_ROOT / "data" / "current_team.json")
    if cfg:
        cur_drv = [str(x) for x in cfg.get("drivers", [])]
        cur_con = [str(x) for x in cfg.get("constructors", [])]
        free = int(cfg.get("free_transfers", 2))
        recs = best_two_transfer_move(cur_drv, cur_con, drivers, constructors, budget=100.0, free_transfers=free)
        print("\nBest transfer recommendations from your current team (net changes; -10 per extra transfer):\n")
        for r in recs[:5]:
            print(f"Transfers: {r.num_transfers} (penalty {r.penalty_points}) | New cost {r.new_cost:.1f} | New expected {r.new_expected:.2f} | Δ after penalty {r.delta_expected_after_penalty:.2f}")
            print("Outgoing IDs:", r.outgoing)
            print("Incoming IDs:", r.incoming)
            print()
    else:
        print("\nTo get transfer suggestions, create data/current_team.json like:\n"
              "{\n  \"drivers\": [131, 117, 1982, 18, 11031],\n  \"constructors\": [27, 28],\n  \"free_transfers\": 2\n}\n")

if __name__ == "__main__":
    main()