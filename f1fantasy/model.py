from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import re

# === Scoring tables (2026 rules excerpt provided by user) ===
QUALI_POINTS = {1:10,2:9,3:8,4:7,5:6,6:5,7:4,8:3,9:2,10:1}
SPRINT_POINTS = {1:8,2:7,3:6,4:5,5:4,6:3,7:2,8:1}
RACE_POINTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}

OVERTAKE_CAP_RACE = 10
OVERTAKE_CAP_SPRINT = 5
FASTEST_LAP_RACE_POINTS = 10
FASTEST_LAP_SPRINT_POINTS = 5
CTOR_DNF_PENALTY_FACTOR_RACE = 0.70
CTOR_DNF_PENALTY_FACTOR_SPRINT = 0.70

def _has_time(x: str) -> bool:
    return isinstance(x, str) and x.strip() != ""

def driver_quali_points(pos: int, q1: str) -> int:
    # NC/DSQ/No time set: -5 (we detect as no Q1 time)
    if not _has_time(q1):
        return -5
    return int(QUALI_POINTS.get(int(pos), 0))

def driver_sprint_points(pos: int, grid: int, is_dnf: int, is_dsq: int = 0, has_fastest_lap: int = 0, dnf_penalty: int = 10) -> int:
    # 2026 Sprint DNF/DSQ/NC = -10
    if int(is_dsq) == 1:
        return -dnf_penalty
    if int(is_dnf) == 1:
        return -dnf_penalty
    finish = int(SPRINT_POINTS.get(int(pos), 0))
    delta = int(grid) - int(pos)  # positions gained/lost
    # Overtake proxy: cap the upside from positions gained
    overtake_proxy = max(0, delta)
    if overtake_proxy > OVERTAKE_CAP_SPRINT:
        overtake_proxy = OVERTAKE_CAP_SPRINT
    fl = FASTEST_LAP_SPRINT_POINTS if int(has_fastest_lap) == 1 else 0
    return finish + delta + overtake_proxy + fl


def driver_race_points(pos: int, grid: int, is_dnf: int, is_dsq: int = 0, has_fastest_lap: int = 0, dnf_penalty: int = 20) -> int:
    # Race DNF/DSQ/NC = -20
    if int(is_dsq) == 1:
        return -dnf_penalty
    if int(is_dnf) == 1:
        return -dnf_penalty
    finish = int(RACE_POINTS.get(int(pos), 0))
    delta = int(grid) - int(pos)  # positions gained/lost
    # Overtake proxy: cap the upside from positions gained
    overtake_proxy = max(0, delta)
    if overtake_proxy > OVERTAKE_CAP_RACE:
        overtake_proxy = OVERTAKE_CAP_RACE
    fl = FASTEST_LAP_RACE_POINTS if int(has_fastest_lap) == 1 else 0
    # DOTD not modelled (no data)
    return finish + delta + overtake_proxy + fl


def constructor_quali_progression_bonus(q2_reached: int, q3_reached: int) -> int:
    """Applies the single highest applicable bonus/penalty:
    - Both Q3: +10
    - One Q3: +5
    - Both Q2: +3
    - One Q2: +1
    - Neither Q2: -1
    """
    if q3_reached >= 2:
        return 10
    if q3_reached == 1:
        return 5
    if q2_reached >= 2:
        return 3
    if q2_reached == 1:
        return 1
    return -1

def _season_weight(season: int, current_season: int, last_season_weight: float = 0.95, older_decay: float = 0.75) -> float:
    if season == current_season:
        return 1.0
    if season == current_season - 1:
        return float(last_season_weight)
    gap = (current_season - 1) - season
    if gap <= 0:
        return float(last_season_weight)
    return float(last_season_weight) * (older_decay ** gap)

def _horizon_weights(n: int = 5, w1: float = 1.0, w_next: float = 0.7) -> List[float]:
    if n <= 0:
        return []
    return [w1] + [w_next] * (n - 1)


# === CHANGE: dynamic current-season vs historical blend helpers ===
def _current_season_share(completed_races: int, min_share: float = 0.50, max_share: float = 0.75, cap_races: int = 10) -> float:
    if completed_races <= 0:
        return 0.0
    if completed_races >= cap_races:
        return float(max_share)
    return float(min_share + (completed_races - 1) * (max_share - min_share) / max(1, cap_races - 1))


# === CHANGE: within-current-season recency weighting ===
def _current_round_weight(round_no: int, latest_round: int, decay: float = 0.95) -> float:
    return float(decay ** max(0, int(latest_round) - int(round_no)))


# === CHANGE: historical-only season scaling with 0.75^x ===
def _historical_season_weight_hist_only(season: int, current_season: int, decay: float = 0.75) -> float:
    seasons_back = int(current_season) - int(season)
    x = max(0, seasons_back - 1)
    return float(decay ** x)


# === CHANGE: helper for safe weighted means ===
def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna()
    if mask.sum() == 0:
        return float("nan")
    vv = v[mask].astype(float)
    ww = w[mask].astype(float)
    if float(ww.sum()) == 0.0:
        return float(vv.mean())
    return float(np.average(vv, weights=ww))


# === CHANGE: blend current and historical estimates with sensible fallbacks ===
def _blend_series(current: pd.Series, historical: pd.Series, current_share: float) -> pd.Series:
    out = current.copy()
    both = current.notna() & historical.notna()
    only_current = current.notna() & historical.isna()
    only_hist = current.isna() & historical.notna()

    out[:] = np.nan
    out.loc[both] = current_share * current.loc[both] + (1.0 - current_share) * historical.loc[both]
    out.loc[only_current] = current.loc[only_current]
    out.loc[only_hist] = historical.loc[only_hist]
    return out


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# Export for recommend.py imports
__all__ = [
    "compute_weekend_points",
    "expected_scores_horizon",
    "apply_no_negative_expectation",
    "_horizon_weights",
]

def compute_weekend_points(
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    sprint: pd.DataFrame,
    current_season: int,
    last_season_weight: float = 0.95,
    older_decay: float = 0.75,
    race_dnf_penalty: int = 20,
    sprint_dnf_penalty: int = 10,
) -> pd.DataFrame:
    """Per-driver per-round fantasy proxy points (qualifying + sprint + race) with season weights."""
    r = results.copy()

    # Split DSQ from DNF: DSQ handled separately
    status = r["status"].fillna("").astype(str)
    is_dsq = status.str.contains("Disqualified", case=False, na=False)
    classified = status.eq("Finished") | status.eq("Lapped") | status.str.match(r"^\+\d+\s*Lap(s)?$", na=False)
    r["is_dsq"] = is_dsq.astype(int)
    r["is_dnf"] = (~classified & ~is_dsq).astype(int)

    # Fastest lap (race): from Ergast/Jolpica results field FastestLap.rank==1 if available
    if "fastestLapRank" in r.columns:
        r["has_fastest_lap"] = (pd.to_numeric(r["fastestLapRank"], errors="coerce").fillna(0).astype(int) == 1).astype(int)
    else:
        r["has_fastest_lap"] = 0

    r["race_points"] = r.apply(lambda x: driver_race_points(int(x["position"]), int(x["grid"]), int(x["is_dnf"]), int(x.get("is_dsq",0)), int(x.get("has_fastest_lap",0)), dnf_penalty=race_dnf_penalty), axis=1)

    q = qualifying.copy()
    if len(q):
        q["quali_points"] = q.apply(lambda x: driver_quali_points(int(x["position"]), str(x.get("q1",""))), axis=1)
        q["q2_reached"] = q["q2"].apply(lambda s: 1 if _has_time(str(s)) else 0)
        q["q3_reached"] = q["q3"].apply(lambda s: 1 if _has_time(str(s)) else 0)
    else:
        q = pd.DataFrame(columns=["season","round","driverId","quali_points","q2_reached","q3_reached"])

    s = sprint.copy()
    if len(s):
        # Sprint DSQ/DNF split if status exists
        if "status" in s.columns:
            s_status = s["status"].fillna("").astype(str)
            s_is_dsq = s_status.str.contains("Disqualified", case=False, na=False).astype(int)
            s_classified = s_status.eq("Finished") | s_status.eq("Lapped") | s_status.str.match(r"^\+\d+\s*Lap(s)?$", na=False)
            s_is_dnf = (~s_classified & (s_is_dsq == 0)).astype(int)
            s["sprint_is_dsq"] = s_is_dsq
            s["sprint_is_dnf"] = s_is_dnf
        else:
            s["sprint_is_dsq"] = 0
            s["sprint_is_dnf"] = s.get("is_dnf", 0)

        # Fastest lap in sprint not available in our current feeds
        s["has_fastest_lap"] = 0
        s["sprint_points"] = s.apply(lambda x: driver_sprint_points(int(x["position"]), int(x["grid"]), int(x.get("sprint_is_dnf",0)), int(x.get("sprint_is_dsq",0)), int(x.get("has_fastest_lap",0)), dnf_penalty=sprint_dnf_penalty), axis=1)
    else:
        s = pd.DataFrame(columns=["season","round","driverId","sprint_points"])

    # Bug fix - Data handling without sprint data  
    # out = r[["season","round","circuitName","driverId","driver","constructorId","constructor","race_points","is_dnf","is_dsq","has_fastest_lap","grid","position","status"]].copy()
    # out = out.merge(q[["season","round","driverId","quali_points","q2_reached","q3_reached"]], on=["season","round","driverId"], how="left")
    # out = out.merge(s[["season","round","driverId","sprint_points","sprint_is_dnf","sprint_is_dsq"]], on=["season","round","driverId"], how="left")

    out = r[[
        "season", "round", "circuitName", "driverId", "driver",
        "constructorId", "constructor", "race_points", "is_dnf",
        "is_dsq", "has_fastest_lap", "grid", "position", "status"
    ]].copy()
    
    out = out.merge(
        q[["season", "round", "driverId", "quali_points", "q2_reached", "q3_reached"]],
        on=["season", "round", "driverId"],
        how="left"
    )
    
    # Handle seasons / rounds with no sprint data yet
    if not s.empty and "sprint_points" in s.columns:
        for col in ["sprint_is_dnf", "sprint_is_dsq"]:
            if col not in s.columns:
                s[col] = 0
    
        out = out.merge(
            s[["season", "round", "driverId", "sprint_points", "sprint_is_dnf", "sprint_is_dsq"]],
            on=["season", "round", "driverId"],
            how="left"
        )
    else:
        out["sprint_points"] = 0
        out["sprint_is_dnf"] = 0
        out["sprint_is_dsq"] = 0

    
    # Missing qualifying row usually means no time / did not participate / not classified.
    # Under the fantasy rules this should be -5 rather than 0.
    missing_quali = out["quali_points"].isna()
    
    out.loc[missing_quali, "quali_points"] = -5
    out.loc[missing_quali, "q2_reached"] = 0
    out.loc[missing_quali, "q3_reached"] = 0
    
    out["quali_points"] = out["quali_points"].astype(float)
    out["q2_reached"] = out["q2_reached"].fillna(0).astype(int)
    out["q3_reached"] = out["q3_reached"].fillna(0).astype(int)
    
    out["sprint_points"] = out["sprint_points"].fillna(0)
    out["sprint_is_dnf"] = out["sprint_is_dnf"].fillna(0).astype(int)
    out["sprint_is_dsq"] = out["sprint_is_dsq"].fillna(0).astype(int)
    

    out["weekend_points"] = out["quali_points"] + out["sprint_points"] + out["race_points"]
    out["season_w"] = out["season"].astype(int).apply(lambda yr: _season_weight(yr, current_season, last_season_weight, older_decay))
    return out

def _constructor_round_points(wp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate driver-level weekend points into constructor-level round points using 2026 rules.
    Implements:
    - Combined total of two drivers' qualifying points
    - Quali progression bonus/penalty (Q2/Q3 reach)
    - -5 per disqualified driver (constructor qualifying section)
    - Combined sprint & race points as sum of both drivers (DOTD excluded but not modelled)
    Note: Overtakes/FL/DOTD are omitted due to missing data.
    """
    d = wp.copy()

    # First compute constructor-level pieces per round
    # Sum driver points by phase
    agg = d.groupby(["season","round","circuitName","constructorId","constructor"], as_index=False).agg(
        quali_sum=("quali_points","sum"),
        sprint_sum=("sprint_points","sum"),
        race_sum=("race_points","sum"),
        q2_reached=("q2_reached","sum"),
        q3_reached=("q3_reached","sum"),
        dsq_drivers=("is_dsq","sum"),
        dnf_drivers=("is_dnf","sum"),
        sprint_dnf_drivers=("sprint_is_dnf","sum"),
        dnf_rate=("is_dnf","mean"),
    )

    agg["quali_bonus"] = agg.apply(lambda x: constructor_quali_progression_bonus(int(x["q2_reached"]), int(x["q3_reached"])), axis=1)
    # Constructor DSQ penalty in qualifying: -5 per DSQ driver (in addition to driver quali points)
    agg["quali_dsq_penalty"] = -5 * agg["dsq_drivers"].astype(int)

    agg["ctor_race_relief"] = (1.0 - CTOR_DNF_PENALTY_FACTOR_RACE) * float(20) * agg["dnf_drivers"].astype(float)
    agg["ctor_sprint_relief"] = (1.0 - CTOR_DNF_PENALTY_FACTOR_SPRINT) * float(10) * agg["sprint_dnf_drivers"].astype(float)

    agg["constructor_weekend_points"] = (
        agg["quali_sum"] + agg["quali_bonus"] + agg["quali_dsq_penalty"]
        + agg["sprint_sum"] + agg["race_sum"] + agg["ctor_race_relief"] + agg["ctor_sprint_relief"]
    )
    return agg


def expected_scores_horizon(
    weekend_points: pd.DataFrame,
    upcoming_circuits: List[str],
    horizon_weights: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expected scores for next N races.

    Current-season block:
    - uses ONLY completed races from the current season
    - uses within-season recency decay (latest = 1, previous = 0.95, ...)

    Historical block:
    - uses ONLY previous seasons
    - keeps circuit-specific historical behaviour

    Final EV for each future circuit:
    - current_share * overall_current
    - + (1-current_share) * historical_circuit_or_overall_hist
    """
    wp = weekend_points.copy()
    current_season = int(wp["season"].max())
    current_mask = wp["season"].astype(int) == current_season
    latest_round = int(wp.loc[current_mask, "round"].max()) if current_mask.any() else 0
    current_share = _current_season_share(latest_round)

    wp["w_current_component"] = np.where(
        current_mask,
        wp["round"].astype(int).apply(lambda r: _current_round_weight(r, latest_round, decay=0.95)),
        0.0,
    )
    wp["w_hist_component"] = np.where(
        current_mask,
        0.0,
        wp["season"].astype(int).apply(lambda yr: _historical_season_weight_hist_only(yr, current_season, decay=0.75)),
    )

    # Driver overall summaries
    base_driver = wp[["driverId", "driver"]].drop_duplicates().copy()

    current_driver = wp.loc[current_mask].groupby(["driverId", "driver"], as_index=False).apply(
        lambda d: pd.Series({
            "overall_current": _weighted_mean(d["weekend_points"], d["w_current_component"]),
            "dnf_current": float((d["is_dnf"].sum() + 1.0) / (len(d) + 15.0)) if len(d) else np.nan,
            "vol_current": float(np.std(d["weekend_points"])) if len(d) else np.nan,
        })
    ).reset_index(drop=True)

    if (~current_mask).any():
        hist_driver = wp.loc[~current_mask].groupby(["driverId", "driver"], as_index=False).apply(
            lambda d: pd.Series({
                "overall_hist": _weighted_mean(d["weekend_points"], d["w_hist_component"]),
                "dnf_hist": float((d["is_dnf"].sum() + 1.0) / (len(d) + 15.0)) if len(d) else np.nan,
                "vol_hist": float(np.std(d["weekend_points"])) if len(d) else np.nan,
            })
        ).reset_index(drop=True)

        hist_circ_driver = wp.loc[~current_mask].groupby(["circuitName", "driverId", "driver"], as_index=False).apply(
            lambda d: pd.Series({
                "circ_hist": _weighted_mean(d["weekend_points"], d["w_hist_component"]),
                "n_hist": int(len(d)),
            })
        ).reset_index(drop=True)
    else:
        hist_driver = pd.DataFrame(columns=["driverId", "driver", "overall_hist", "dnf_hist", "vol_hist"])
        hist_circ_driver = pd.DataFrame(columns=["circuitName", "driverId", "driver", "circ_hist", "n_hist"])

    hist_driver = _ensure_columns(hist_driver, ["driverId", "driver", "overall_hist", "dnf_hist", "vol_hist"])
    hist_circ_driver = _ensure_columns(hist_circ_driver, ["circuitName", "driverId", "driver", "circ_hist", "n_hist"])

    overall_driver = (
        base_driver
        .merge(current_driver, on=["driverId", "driver"], how="left")
        .merge(hist_driver, on=["driverId", "driver"], how="left")
    )
    overall_driver["overall_mean"] = _blend_series(overall_driver["overall_current"], overall_driver["overall_hist"], current_share)
    overall_driver["dnf_rate"] = _blend_series(overall_driver["dnf_current"], overall_driver["dnf_hist"], current_share).fillna(1.0 / 15.0)
    overall_driver["volatility"] = _blend_series(overall_driver["vol_current"], overall_driver["vol_hist"], current_share).fillna(0.0)

    # Constructor overall summaries
    ctor_round = _constructor_round_points(wp)
    ctor_current_mask = ctor_round["season"].astype(int) == current_season
    ctor_round["w_current_component"] = np.where(
        ctor_current_mask,
        ctor_round["round"].astype(int).apply(lambda r: _current_round_weight(r, latest_round, decay=0.95)),
        0.0,
    )
    ctor_round["w_hist_component"] = np.where(
        ctor_current_mask,
        0.0,
        ctor_round["season"].astype(int).apply(lambda yr: _historical_season_weight_hist_only(yr, current_season, decay=0.75)),
    )

    base_ctor = ctor_round[["constructorId", "constructor"]].drop_duplicates().copy()

    current_ctor = ctor_round.loc[ctor_current_mask].groupby(["constructorId", "constructor"], as_index=False).apply(
        lambda d: pd.Series({
            "overall_current": _weighted_mean(d["constructor_weekend_points"], d["w_current_component"]),
            "dnf_current": float((d["dnf_drivers"].sum() + 1.0) / (2.0 * len(d) + 15.0)) if len(d) else np.nan,
            "vol_current": float(np.std(d["constructor_weekend_points"])) if len(d) else np.nan,
        })
    ).reset_index(drop=True)

    if (~ctor_current_mask).any():
        hist_ctor = ctor_round.loc[~ctor_current_mask].groupby(["constructorId", "constructor"], as_index=False).apply(
            lambda d: pd.Series({
                "overall_hist": _weighted_mean(d["constructor_weekend_points"], d["w_hist_component"]),
                "dnf_hist": float((d["dnf_drivers"].sum() + 1.0) / (2.0 * len(d) + 15.0)) if len(d) else np.nan,
                "vol_hist": float(np.std(d["constructor_weekend_points"])) if len(d) else np.nan,
            })
        ).reset_index(drop=True)

        hist_circ_ctor = ctor_round.loc[~ctor_current_mask].groupby(["circuitName", "constructorId", "constructor"], as_index=False).apply(
            lambda d: pd.Series({
                "circ_hist": _weighted_mean(d["constructor_weekend_points"], d["w_hist_component"]),
                "n_hist": int(len(d)),
            })
        ).reset_index(drop=True)
    else:
        hist_ctor = pd.DataFrame(columns=["constructorId", "constructor", "overall_hist", "dnf_hist", "vol_hist"])
        hist_circ_ctor = pd.DataFrame(columns=["circuitName", "constructorId", "constructor", "circ_hist", "n_hist"])

    hist_ctor = _ensure_columns(hist_ctor, ["constructorId", "constructor", "overall_hist", "dnf_hist", "vol_hist"])
    hist_circ_ctor = _ensure_columns(hist_circ_ctor, ["circuitName", "constructorId", "constructor", "circ_hist", "n_hist"])

    overall_ctor = (
        base_ctor
        .merge(current_ctor, on=["constructorId", "constructor"], how="left")
        .merge(hist_ctor, on=["constructorId", "constructor"], how="left")
    )
    overall_ctor["overall_mean"] = _blend_series(overall_ctor["overall_current"], overall_ctor["overall_hist"], current_share)
    overall_ctor["dnf_rate"] = _blend_series(overall_ctor["dnf_current"], overall_ctor["dnf_hist"], current_share).fillna(1.0 / 15.0)
    overall_ctor["volatility"] = _blend_series(overall_ctor["vol_current"], overall_ctor["vol_hist"], current_share).fillna(0.0)

    # Horizon expectation helper
    def horizon_driver():
        base = overall_driver.copy()
        base["exp_score"] = 0.0
        for circuit, w in zip(upcoming_circuits, horizon_weights):
            sub = hist_circ_driver[hist_circ_driver["circuitName"].str.contains(circuit, case=False, na=False)].copy()
            tmp = base.merge(sub[["driverId", "circ_hist"]], on="driverId", how="left")
            hist_value = tmp["circ_hist"].fillna(tmp["overall_hist"])
            current_value = tmp["overall_current"]
            use = _blend_series(current_value, hist_value, current_share)
            use = use.fillna(current_value).fillna(hist_value).fillna(tmp["overall_mean"])
            base["exp_score"] += float(w) * use
        return base[["driverId", "driver", "exp_score", "dnf_rate", "volatility", "overall_current", "overall_hist", "overall_mean"]]

    def horizon_ctor():
        base = overall_ctor.copy()
        base["exp_score"] = 0.0
        for circuit, w in zip(upcoming_circuits, horizon_weights):
            sub = hist_circ_ctor[hist_circ_ctor["circuitName"].str.contains(circuit, case=False, na=False)].copy()
            tmp = base.merge(sub[["constructorId", "circ_hist"]], on="constructorId", how="left")
            hist_value = tmp["circ_hist"].fillna(tmp["overall_hist"])
            current_value = tmp["overall_current"]
            use = _blend_series(current_value, hist_value, current_share)
            use = use.fillna(current_value).fillna(hist_value).fillna(tmp["overall_mean"])
            base["exp_score"] += float(w) * use
        return base[["constructorId", "constructor", "exp_score", "dnf_rate", "volatility", "overall_current", "overall_hist", "overall_mean"]]

    return horizon_driver(), horizon_ctor()

def apply_no_negative_expectation(
    weekend_points: pd.DataFrame,
    upcoming_circuits: List[str],
    horizon_weights: List[float],
) -> pd.Series:
    """Approx EV under No Negative for drivers using the same current-vs-historical split."""
    wp = weekend_points.copy()
    wp["nn_points"] = wp["weekend_points"].clip(lower=0)

    current_season = int(wp["season"].max())
    current_mask = wp["season"].astype(int) == current_season
    latest_round = int(wp.loc[current_mask, "round"].max()) if current_mask.any() else 0
    current_share = _current_season_share(latest_round)

    wp["w_current_component"] = np.where(
        current_mask,
        wp["round"].astype(int).apply(lambda r: _current_round_weight(r, latest_round, decay=0.95)),
        0.0,
    )
    wp["w_hist_component"] = np.where(
        current_mask,
        0.0,
        wp["season"].astype(int).apply(lambda yr: _historical_season_weight_hist_only(yr, current_season, decay=0.75)),
    )

    base = wp[["driverId"]].drop_duplicates().copy()

    current_overall = wp.loc[current_mask].groupby(["driverId"], as_index=False).apply(
        lambda d: pd.Series({"overall_current": _weighted_mean(d["nn_points"], d["w_current_component"])})
    ).reset_index(drop=True)

    if (~current_mask).any():
        hist_overall = wp.loc[~current_mask].groupby(["driverId"], as_index=False).apply(
            lambda d: pd.Series({"overall_hist": _weighted_mean(d["nn_points"], d["w_hist_component"])})
        ).reset_index(drop=True)
        hist_circ = wp.loc[~current_mask].groupby(["circuitName", "driverId"], as_index=False).apply(
            lambda d: pd.Series({"circ_hist": _weighted_mean(d["nn_points"], d["w_hist_component"])})
        ).reset_index(drop=True)
    else:
        hist_overall = pd.DataFrame(columns=["driverId", "overall_hist"])
        hist_circ = pd.DataFrame(columns=["circuitName", "driverId", "circ_hist"])

    hist_overall = _ensure_columns(hist_overall, ["driverId", "overall_hist"])
    hist_circ = _ensure_columns(hist_circ, ["circuitName", "driverId", "circ_hist"])

    base = base.merge(current_overall, on="driverId", how="left").merge(hist_overall, on="driverId", how="left")
    base["nn_exp_score"] = 0.0

    for circuit, w in zip(upcoming_circuits, horizon_weights):
        sub = hist_circ[hist_circ["circuitName"].str.contains(circuit, case=False, na=False)].copy()
        tmp = base.merge(sub[["driverId", "circ_hist"]], on="driverId", how="left")
        hist_value = tmp["circ_hist"].fillna(tmp["overall_hist"])
        current_value = tmp["overall_current"]
        use = _blend_series(current_value, hist_value, current_share)
        use = use.fillna(current_value).fillna(hist_value).fillna(0.0)
        base["nn_exp_score"] += float(w) * use

    return base.set_index("driverId")["nn_exp_score"]
