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

    out = r[["season","round","circuitName","driverId","driver","constructorId","constructor","race_points","is_dnf","is_dsq","has_fastest_lap","grid","position","status"]].copy()
    out = out.merge(q[["season","round","driverId","quali_points","q2_reached","q3_reached"]], on=["season","round","driverId"], how="left")
    out = out.merge(s[["season","round","driverId","sprint_points","sprint_is_dnf","sprint_is_dsq"]], on=["season","round","driverId"], how="left")

    out["quali_points"] = out["quali_points"].fillna(0)
    out["sprint_points"] = out["sprint_points"].fillna(0)
    out["sprint_is_dnf"] = out["sprint_is_dnf"].fillna(0).astype(int)
    out["sprint_is_dsq"] = out["sprint_is_dsq"].fillna(0).astype(int)
    out["q2_reached"] = out["q2_reached"].fillna(0).astype(int)
    out["q3_reached"] = out["q3_reached"].fillna(0).astype(int)

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
    """Expected scores for next N races.

    Drivers: circuit-aware season-weighted mean of weekend_points.
    Constructors: circuit-aware season-weighted mean of *constructor_weekend_points* (sum of both drivers + quali bonus + dsq penalty).
    """
    wp = weekend_points.copy()

    # === Driver-level overall means (fallback) ===
    overall_driver = wp.groupby(["driverId","driver"], as_index=False).apply(
        lambda d: pd.Series({
            "overall_mean": float(np.average(d["weekend_points"], weights=d["season_w"])) if len(d) else 0.0,
            "dnf_rate": float((d["is_dnf"].sum() + 1.0) / (len(d) + 2.0)) if len(d) else 0.0,
            "volatility": float(np.std(d["weekend_points"])) if len(d) else 0.0,
        })
    ).reset_index(drop=True)

    circ_driver = wp.groupby(["circuitName","driverId","driver"], as_index=False).apply(
        lambda d: pd.Series({
            "circ_mean": float(np.average(d["weekend_points"], weights=d["season_w"])) if len(d) else np.nan,
            "n": int(len(d)),
        })
    ).reset_index(drop=True)

    # === Constructor-level points per round (correct aggregation) ===
    ctor_round = _constructor_round_points(wp)
    ctor_round["season_w"] = ctor_round["season"].astype(int).apply(lambda yr: _season_weight(yr, int(wp["season"].max()), 0.95, 0.75))

    overall_ctor = ctor_round.groupby(["constructorId","constructor"], as_index=False).apply(
        lambda d: pd.Series({
            "overall_mean": float(np.average(d["constructor_weekend_points"], weights=d["season_w"])) if len(d) else 0.0,
            "dnf_rate": float((d["dnf_rate"].sum() + 1.0) / (len(d) + 2.0)) if len(d) else 0.0,
            "volatility": float(np.std(d["constructor_weekend_points"])) if len(d) else 0.0,
        })
    ).reset_index(drop=True)

    circ_ctor = ctor_round.groupby(["circuitName","constructorId","constructor"], as_index=False).apply(
        lambda d: pd.Series({
            "circ_mean": float(np.average(d["constructor_weekend_points"], weights=d["season_w"])) if len(d) else np.nan,
            "n": int(len(d)),
        })
    ).reset_index(drop=True)

    # === Horizon expectation helper ===
    def horizon_driver():
        base = overall_driver.copy()
        base["exp_score"] = 0.0
        for circuit, w in zip(upcoming_circuits, horizon_weights):
            sub = circ_driver[circ_driver["circuitName"].str.contains(circuit, case=False, na=False)].copy()
            tmp = base.merge(sub[["driverId","circ_mean"]], on="driverId", how="left")
            use = tmp["circ_mean"].fillna(tmp["overall_mean"])
            base["exp_score"] += float(w) * use
        return base

    def horizon_ctor():
        base = overall_ctor.copy()
        base["exp_score"] = 0.0
        for circuit, w in zip(upcoming_circuits, horizon_weights):
            sub = circ_ctor[circ_ctor["circuitName"].str.contains(circuit, case=False, na=False)].copy()
            tmp = base.merge(sub[["constructorId","circ_mean"]], on="constructorId", how="left")
            use = tmp["circ_mean"].fillna(tmp["overall_mean"])
            base["exp_score"] += float(w) * use
        return base

    return horizon_driver(), horizon_ctor()

def apply_no_negative_expectation(
    weekend_points: pd.DataFrame,
    upcoming_circuits: List[str],
    horizon_weights: List[float],
) -> pd.Series:
    """Approx EV under No Negative for drivers: floor weekend_points at 0 then recompute circuit-aware horizon EV."""
    wp = weekend_points.copy()
    wp["nn_points"] = wp["weekend_points"].clip(lower=0)

    overall = wp.groupby(["driverId"], as_index=False).apply(
        lambda d: pd.Series({"overall_nn": float(np.average(d["nn_points"], weights=d["season_w"])) if len(d) else 0.0})
    ).reset_index(drop=True)

    circ = wp.groupby(["circuitName","driverId"], as_index=False).apply(
        lambda d: pd.Series({"circ_nn": float(np.average(d["nn_points"], weights=d["season_w"])) if len(d) else np.nan})
    ).reset_index(drop=True)

    base = overall.copy()
    base["nn_exp_score"] = 0.0
    for circuit, w in zip(upcoming_circuits, horizon_weights):
        sub = circ[circ["circuitName"].str.contains(circuit, case=False, na=False)].copy()
        tmp = base.merge(sub[["driverId","circ_nn"]], on="driverId", how="left")
        use = tmp["circ_nn"].fillna(tmp["overall_nn"])
        base["nn_exp_score"] += float(w) * use
    return base.set_index("driverId")["nn_exp_score"]
