from __future__ import annotations

"""
Quick debugging/sanity script (kept separate from recommend.py).

Run:
    python -m f1fantasy.debug_checks

Edits:
- WINDOW_MODE: "last2" or "last5"
- CTOR_TO_CHECK: constructorId string (e.g., "mclaren")
"""

from datetime import datetime
import pandas as pd

from f1fantasy.ergast import fetch_all_supporting, fetch_schedule
from f1fantasy.fantasy_api import fetch_players, fetch_teams
from f1fantasy.model import (
    compute_weekend_points, expected_scores_horizon, _horizon_weights,
    _current_season_share, _current_round_weight, _historical_season_weight_hist_only,
    _weighted_mean, _blend_series
)

WINDOW_MODE = "last2"  # options: "currentonly", "last2", "last5"
CTOR_TO_CHECK = "mclaren"
N_HORIZON = 5

def upcoming_circuits(schedule: pd.DataFrame, today: str, n: int = 5) -> list[str]:
    sch = schedule.copy()
    sch["date"] = sch["date"].astype(str)
    upcoming = sch[sch["date"] >= today].sort_values("round").head(n)
    return [c.split(" Circuit")[0].strip() for c in upcoming["circuitName"].astype(str).tolist()]



def _print_component_debug(weekend_points: pd.DataFrame, current_season: int) -> None:
    """Print current/historical/blended components to diagnose weighting issues."""
    wp = weekend_points.copy()
    current_mask = wp["season"].astype(int) == current_season
    latest_round = int(wp.loc[current_mask, "round"].max()) if current_mask.any() else 0
    current_share = _current_season_share(latest_round)

    wp["w_current_component"] = 0.0
    wp.loc[current_mask, "w_current_component"] = wp.loc[current_mask, "round"].astype(int).apply(
        lambda r: _current_round_weight(r, latest_round, decay=0.95)
    )
    wp["w_hist_component"] = 0.0
    wp.loc[~current_mask, "w_hist_component"] = wp.loc[~current_mask, "season"].astype(int).apply(
        lambda yr: _historical_season_weight_hist_only(yr, current_season, decay=0.75)
    )

    print(f"\nCurrent season = {current_season}, latest completed round = {latest_round}, current_share = {current_share:.3f}")

    # driver components
    base_driver = wp[["driverId","driver"]].drop_duplicates().copy()
    if current_mask.any():
        cur_drv = wp.loc[current_mask].groupby(["driverId","driver"], as_index=False).apply(
            lambda d: pd.Series({"overall_current": _weighted_mean(d["weekend_points"], d["w_current_component"])})
        ).reset_index(drop=True)
    else:
        cur_drv = pd.DataFrame(columns=["driverId","driver","overall_current"])
    if (~current_mask).any():
        hist_drv = wp.loc[~current_mask].groupby(["driverId","driver"], as_index=False).apply(
            lambda d: pd.Series({"overall_hist": _weighted_mean(d["weekend_points"], d["w_hist_component"])})
        ).reset_index(drop=True)
    else:
        hist_drv = pd.DataFrame(columns=["driverId","driver","overall_hist"])
    drv = base_driver.merge(cur_drv, on=["driverId","driver"], how="left").merge(hist_drv, on=["driverId","driver"], how="left")
    drv["overall_mean"] = _blend_series(drv.get("overall_current"), drv.get("overall_hist"), current_share)
    print("\n=== DRIVER CURRENT VS HISTORICAL (top 12 by blended) ===")
    cols=[c for c in ["driver","driverId","overall_current","overall_hist","overall_mean"] if c in drv.columns]
    print(drv[cols].sort_values("overall_mean", ascending=False).head(12).to_string(index=False))

    # constructor components
    from f1fantasy.model import _constructor_round_points
    ctor_round = _constructor_round_points(wp)
    c_mask = ctor_round["season"].astype(int) == current_season
    ctor_round["w_current_component"] = 0.0
    ctor_round.loc[c_mask, "w_current_component"] = ctor_round.loc[c_mask, "round"].astype(int).apply(
        lambda r: _current_round_weight(r, latest_round, decay=0.95)
    )
    ctor_round["w_hist_component"] = 0.0
    ctor_round.loc[~c_mask, "w_hist_component"] = ctor_round.loc[~c_mask, "season"].astype(int).apply(
        lambda yr: _historical_season_weight_hist_only(yr, current_season, decay=0.75)
    )
    base_ctor = ctor_round[["constructorId","constructor"]].drop_duplicates().copy()
    if c_mask.any():
        cur_ctor = ctor_round.loc[c_mask].groupby(["constructorId","constructor"], as_index=False).apply(
            lambda d: pd.Series({"overall_current": _weighted_mean(d["constructor_weekend_points"], d["w_current_component"])})
        ).reset_index(drop=True)
    else:
        cur_ctor = pd.DataFrame(columns=["constructorId","constructor","overall_current"])
    if (~c_mask).any():
        hist_ctor = ctor_round.loc[~c_mask].groupby(["constructorId","constructor"], as_index=False).apply(
            lambda d: pd.Series({"overall_hist": _weighted_mean(d["constructor_weekend_points"], d["w_hist_component"])})
        ).reset_index(drop=True)
    else:
        hist_ctor = pd.DataFrame(columns=["constructorId","constructor","overall_hist"])
    ctor = base_ctor.merge(cur_ctor, on=["constructorId","constructor"], how="left").merge(hist_ctor, on=["constructorId","constructor"], how="left")
    ctor["overall_mean"] = _blend_series(ctor.get("overall_current"), ctor.get("overall_hist"), current_share)
    print("\n=== CONSTRUCTOR CURRENT VS HISTORICAL ===")
    cols=[c for c in ["constructor","constructorId","overall_current","overall_hist","overall_mean"] if c in ctor.columns]
    print(ctor[cols].sort_values("overall_mean", ascending=False).to_string(index=False))

    print("\n=== CURRENT-SEASON CONSTRUCTOR ROUND POINTS ===")
    cols=[c for c in ["season","round","constructor","constructor_weekend_points","dnf_drivers"] if c in ctor_round.columns]
    print(ctor_round.loc[c_mask, cols].sort_values(["round","constructor"]).to_string(index=False))

def main():
    current_season = datetime.utcnow().year
    today = datetime.utcnow().date().isoformat()

    if WINDOW_MODE == "currentonly":
        start_year = current_season
        end_year = current_season
    elif WINDOW_MODE == "last5":
        start_year = current_season - 5
        end_year = current_season
    else:
        start_year = current_season - 1
        end_year = current_season

    all_results, all_quali, all_sprint = [], [], []
    for y in range(start_year, end_year + 1):
        d = fetch_all_supporting(y)
        all_results.append(d["results"])
        all_quali.append(d["qualifying"])
        all_sprint.append(d["sprint"])

    results = pd.concat(all_results, ignore_index=True)
    qualifying = pd.concat(all_quali, ignore_index=True) if any(len(x) for x in all_quali) else pd.DataFrame()
    sprint = pd.concat(all_sprint, ignore_index=True) if any(len(x) for x in all_sprint) else pd.DataFrame()

    # enforce window even if cache contains older seasons
    results = results[(results["season"] >= start_year) & (results["season"] <= end_year)].copy()
    if not qualifying.empty:
        qualifying = qualifying[(qualifying["season"] >= start_year) & (qualifying["season"] <= end_year)].copy()
    if not sprint.empty:
        sprint = sprint[(sprint["season"] >= start_year) & (sprint["season"] <= end_year)].copy()

    players = fetch_players()
    teams = fetch_teams()

    schedule = fetch_schedule(current_season)
    upcoming = upcoming_circuits(schedule, today=today, n=N_HORIZON)
    h_w = _horizon_weights(len(upcoming), w1=1.0, w_next=0.7)

    print("Window:", start_year, "to", end_year)
    print("Upcoming circuits:", upcoming)
    print("Horizon weights:", h_w)

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

    
    # --- New checks: fastest lap + DSQ split + DNF split ---
    if "has_fastest_lap" in weekend_points.columns:
        print("\nFastest-lap flags (race) value counts:")
        print(weekend_points["has_fastest_lap"].value_counts(dropna=False))
    if "is_dsq" in weekend_points.columns:
        print("\nDSQ counts:")
        print(weekend_points["is_dsq"].value_counts(dropna=False))
    if "is_dnf" in weekend_points.columns:
        print("\nDNF counts (excluding DSQ):")
        print(weekend_points["is_dnf"].value_counts(dropna=False))
    drv_exp, con_exp = expected_scores_horizon(weekend_points, upcoming, h_w)
    _print_component_debug(weekend_points, current_season)

    print("\nTop 10 driver exp_score (model):")
    print(drv_exp.sort_values("exp_score", ascending=False)[["driver","driverId","exp_score","dnf_rate"]].head(10).to_string(index=False))

    print("\nTop 10 constructor exp_score (model):")
    print(con_exp.sort_values("exp_score", ascending=False)[["constructor","constructorId","exp_score","dnf_rate"]].head(10).to_string(index=False))

    # Quick value check from fantasy feed only (no mapping)
    print("\nFantasy roster (players) sample:")
    print(players[["name","team","price"]].head(10).to_string(index=False))

    # Constructor per-round sanity
    crows = weekend_points[weekend_points["constructorId"] == CTOR_TO_CHECK].copy()
    if len(crows):
        per_round = (crows.groupby(["season","round","circuitName"], as_index=False)
                     .agg(driver_weekend_sum=("weekend_points","sum"),
                          n_drivers=("driverId","nunique")))
        print(f"\nConstructor per-round sums for {CTOR_TO_CHECK}:")
        print(per_round.sort_values(["season","round"]).tail(15).to_string(index=False))
        print("n_drivers counts:", per_round["n_drivers"].value_counts().to_dict())
    else:
        print(f"\nNo rows found for constructorId={CTOR_TO_CHECK}. Available constructorIds:")
        print(sorted(weekend_points["constructorId"].unique().tolist()))

if __name__ == "__main__":
    main()
