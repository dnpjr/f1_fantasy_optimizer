from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import re

from f1fantasy.race_selection import (
    RaceKey,
    available_races,
    canonical_race_key,
    recency_weights,
    resolve_selected_races,
)
from f1fantasy.weekend_state import EventKey, UpcomingEvent, WeekendFormat

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
    if both.any():
        out.loc[both] = current_share * current.loc[both] + (1.0 - current_share) * historical.loc[both]
    if only_current.any():
        out.loc[only_current] = current.loc[only_current]
    if only_hist.any():
        out.loc[only_hist] = historical.loc[only_hist]
    return out


def _adjust_current_share(base_share: float, current_weight: float = 1.0, past_weight: float = 1.0) -> float:
    current_part = float(base_share) * float(current_weight)
    past_part = (1.0 - float(base_share)) * float(past_weight)
    denom = current_part + past_part
    if denom <= 0:
        return float(base_share)
    return float(current_part / denom)


def _relative_current_share(current_weight: float, past_weight: float) -> float:
    """Normalize the two configured blend weights; zero/zero means equal shares."""
    current = max(0.0, float(current_weight))
    historical = max(0.0, float(past_weight))
    denominator = current + historical
    if denominator == 0.0:
        return 0.5
    return current / denominator


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# Export for recommend.py imports
__all__ = [
    "compute_weekend_points",
    "expected_scores_horizon",
    "expected_scores_horizon_by_component",
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
    completed_event_keys: set[EventKey] | tuple[EventKey, ...] | None = None,
    complete_qualifying_keys: set[EventKey] | tuple[EventKey, ...] | None = None,
    complete_sprint_keys: set[EventKey] | tuple[EventKey, ...] | None = None,
) -> pd.DataFrame:
    """Per-driver per-round fantasy proxy points (qualifying + sprint + race) with season weights."""
    r = results.copy()
    q = qualifying.copy()
    s = sprint.copy()

    def _keys(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [EventKey(int(season), int(round_no)) for season, round_no in zip(frame["season"], frame["round"])],
            index=frame.index,
            dtype=object,
        )

    if completed_event_keys is not None:
        completed = set(completed_event_keys)
        for name, frame in (("results", r), ("qualifying", q), ("sprint", s)):
            if frame.empty or not {"season", "round"}.issubset(frame.columns):
                continue
            seasons = pd.to_numeric(frame["season"], errors="coerce")
            keep = seasons != int(current_season)
            keep |= _keys(frame).isin(completed)
            if name == "results":
                r = frame[keep].copy()
            elif name == "qualifying":
                q = frame[keep].copy()
            else:
                s = frame[keep].copy()

    # Never convert a live/provisional classification into DNF scoring, even
    # when this helper is called without the higher-level weekend gate.
    if not r.empty and {"season", "round", "status"}.issubset(r.columns):
        non_final = r["status"].fillna("").astype(str).str.contains(
            r"\b(?:running|live|provisional|pending|in progress|not started|under investigation)\b",
            case=False,
            regex=True,
        )
        provisional_keys = set(_keys(r[non_final]).tolist())
        if provisional_keys:
            r = r[~_keys(r).isin(provisional_keys)].copy()

    if r.empty:
        return pd.DataFrame(
            columns=[
                "season", "round", "circuitName", "driverId", "driver",
                "constructorId", "constructor", "race_points", "is_dnf",
                "is_dsq", "has_fastest_lap", "grid", "position", "status",
                "quali_points", "q2_reached", "q3_reached", "sprint_points",
                "qualifying_points", "sprint_is_dnf", "sprint_is_dsq",
                "sprint_applicable", "sprint_observed", "weekend_points", "season_w",
            ]
        )

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

    if len(q):
        q["quali_points"] = q.apply(lambda x: driver_quali_points(int(x["position"]), str(x.get("q1",""))), axis=1)
        q["q2_reached"] = q["q2"].apply(lambda s: 1 if _has_time(str(s)) else 0)
        q["q3_reached"] = q["q3"].apply(lambda s: 1 if _has_time(str(s)) else 0)
    else:
        q = pd.DataFrame(columns=["season","round","driverId","quali_points","q2_reached","q3_reached"])

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

    sprint_event_keys = (
        set(_keys(s).tolist())
        if not s.empty and {"season", "round"}.issubset(s.columns)
        else set()
    )

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

    out["sprint_observed"] = out["sprint_points"].notna()
    out["sprint_applicable"] = _keys(out).isin(sprint_event_keys)

    
    # Missing qualifying row usually means no time / did not participate / not classified.
    # Under the fantasy rules this should be -5 rather than 0.
    missing_quali = out["quali_points"].isna()
    if complete_qualifying_keys is not None:
        complete_quali = set(complete_qualifying_keys)
        out_keys = _keys(out)
        missing_quali &= (
            pd.to_numeric(out["season"], errors="coerce") != int(current_season)
        ) | out_keys.isin(complete_quali)
    
    out.loc[missing_quali, "quali_points"] = -5
    out.loc[missing_quali, "q2_reached"] = 0
    out.loc[missing_quali, "q3_reached"] = 0

    # Rows excluded by the completion gate should not normally reach this
    # point. If a caller supplies independent incomplete qualifying data,
    # retain missingness instead of silently creating a -5 observation.
    unresolved_quali = out["quali_points"].isna()
    if unresolved_quali.any():
        out = out[~unresolved_quali].copy()
    
    out["quali_points"] = out["quali_points"].astype(float)
    out["q2_reached"] = out["q2_reached"].fillna(0).astype(int)
    out["q3_reached"] = out["q3_reached"].fillna(0).astype(int)
    
    if complete_sprint_keys is not None:
        complete_sprint = set(complete_sprint_keys)
        unresolved_sprint = (
            out["sprint_points"].isna()
            & (pd.to_numeric(out["season"], errors="coerce") == int(current_season))
            & ~_keys(out).isin(complete_sprint)
        )
        if unresolved_sprint.any():
            out = out[~unresolved_sprint].copy()
    out["sprint_points"] = out["sprint_points"].fillna(0)
    out["sprint_is_dnf"] = out["sprint_is_dnf"].fillna(0).astype(int)
    out["sprint_is_dsq"] = out["sprint_is_dsq"].fillna(0).astype(int)
    

    out["qualifying_points"] = out["quali_points"]
    out["weekend_points"] = out["qualifying_points"] + out["sprint_points"] + out["race_points"]
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
    if "qualifying_points" not in d.columns:
        d["qualifying_points"] = d["quali_points"]
    if "sprint_applicable" not in d.columns:
        sprint_values = pd.to_numeric(d.get("sprint_points"), errors="coerce")
        d["sprint_applicable"] = sprint_values.notna() & sprint_values.ne(0)
    if "sprint_observed" not in d.columns:
        d["sprint_observed"] = d["sprint_applicable"]

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
        sprint_applicable=("sprint_applicable", "max"),
        sprint_observed=("sprint_observed", "min"),
    )

    agg["quali_bonus"] = agg.apply(lambda x: constructor_quali_progression_bonus(int(x["q2_reached"]), int(x["q3_reached"])), axis=1)
    # Constructor DSQ penalty in qualifying: -5 per DSQ driver (in addition to driver quali points)
    agg["quali_dsq_penalty"] = -5 * agg["dsq_drivers"].astype(int)

    agg["ctor_race_relief"] = (1.0 - CTOR_DNF_PENALTY_FACTOR_RACE) * float(20) * agg["dnf_drivers"].astype(float)
    agg["ctor_sprint_relief"] = (1.0 - CTOR_DNF_PENALTY_FACTOR_SPRINT) * float(10) * agg["sprint_dnf_drivers"].astype(float)

    agg["qualifying_points"] = agg["quali_sum"] + agg["quali_bonus"] + agg["quali_dsq_penalty"]
    agg["sprint_points"] = agg["sprint_sum"] + agg["ctor_sprint_relief"]
    agg["race_points"] = agg["race_sum"] + agg["ctor_race_relief"]
    agg["constructor_weekend_points"] = (
        agg["qualifying_points"] + agg["sprint_points"] + agg["race_points"]
    )
    agg["weekend_points"] = agg["constructor_weekend_points"]
    return agg


def normalise_sprint_baseline_inputs(
    rows: pd.DataFrame,
    sprint_keys: set[tuple[int, int]],
) -> pd.DataFrame:
    """Remove historical Sprint points before adding a fitted future Sprint bonus.

    Recorded totals without a session split cannot be normalised safely, so omit
    those Sprint observations. Ordinary observations are retained exactly.
    """
    if rows.empty:
        return rows.copy(deep=True)
    result = rows.copy(deep=True)
    is_sprint = pd.Series(
        [(int(s), int(r)) in sprint_keys for s, r in zip(result["season"], result["round"])],
        index=result.index,
    )
    components = result.reindex(columns=["sprint_points", "sprint_qualifying_points"]).apply(
        pd.to_numeric, errors="coerce"
    ).sum(axis=1, min_count=1)
    for column in ("weekend_points", "constructor_weekend_points", "fantasy_points"):
        if column in result:
            result.loc[is_sprint, column] = (
                pd.to_numeric(result.loc[is_sprint, column], errors="coerce")
                - components.loc[is_sprint]
            )
    return result.loc[~is_sprint | components.notna()].copy()


def expected_scores_horizon(
    weekend_points: pd.DataFrame,
    upcoming_circuits: List[str],
    horizon_weights: List[float],
    current_season_weight: float = 1.0,
    past_season_weight: float = 1.0,
    recency_decay: float = 0.95,
    historical_season_decay: float = 0.75,
    selected_race_keys: Iterable[RaceKey | tuple[int, int]] | None = None,
    selected_race_weights: Mapping[RaceKey | tuple[int, int], float] | None = None,
    current_season: int | None = None,
    constructor_weekend_points: pd.DataFrame | None = None,
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
    current_season = int(current_season or wp["season"].max())
    current_rows = wp[wp["season"].astype(int) == current_season].copy()
    catalogue_input = current_rows.rename(
        columns={"weekend_points": "fantasy_points", "circuitName": "race_name"}
    )
    catalogue_input["is_played"] = 1
    proxy_available = available_races(catalogue_input, season=current_season)
    if selected_race_keys is None:
        selected = resolve_selected_races(proxy_available, "All")
    else:
        selected = resolve_selected_races(
            proxy_available,
            "Custom",
            custom_keys=selected_race_keys,
        )
    raw_race_weights = selected_race_weights or recency_weights(selected, recency_decay)
    race_weights = {
        canonical_race_key(
            key.season if isinstance(key, RaceKey) else key[0],
            key.round if isinstance(key, RaceKey) else key[1],
        ): float(weight)
        for key, weight in raw_race_weights.items()
    }
    selected_key_set = set(selected.included)
    row_keys = [RaceKey(int(season), int(round_no)) for season, round_no in zip(wp["season"], wp["round"])]
    current_mask = pd.Series(
        [key in selected_key_set for key in row_keys],
        index=wp.index,
        dtype=bool,
    )
    historical_mask = wp["season"].astype(int) != current_season
    current_share = _relative_current_share(current_season_weight, past_season_weight)

    wp["w_current_component"] = [float(race_weights.get(key, 0.0)) for key in row_keys]
    wp["w_hist_component"] = np.where(
        wp["season"].astype(int) == current_season,
        0.0,
        wp["season"].astype(int).apply(lambda yr: _historical_season_weight_hist_only(yr, current_season, decay=historical_season_decay)),
    )

    # Driver overall summaries
    base_driver = wp[["driverId", "driver"]].drop_duplicates().copy()

    if current_mask.any():
        current_driver = wp.loc[current_mask].groupby(["driverId", "driver"], as_index=False).apply(
            lambda d: pd.Series({
                "overall_current": _weighted_mean(d["weekend_points"], d["w_current_component"]),
                "dnf_current": float((d["is_dnf"].sum() + 1.0) / (len(d) + 15.0)) if len(d) else np.nan,
                "vol_current": float(np.std(d["weekend_points"])) if len(d) else np.nan,
            })
        ).reset_index(drop=True)
    else:
        current_driver = pd.DataFrame(columns=["driverId", "driver", "overall_current", "dnf_current", "vol_current"])

    if historical_mask.any():
        hist_driver = wp.loc[historical_mask].groupby(["driverId", "driver"], as_index=False).apply(
            lambda d: pd.Series({
                "overall_hist": _weighted_mean(d["weekend_points"], d["w_hist_component"]),
                "dnf_hist": float((d["is_dnf"].sum() + 1.0) / (len(d) + 15.0)) if len(d) else np.nan,
                "vol_hist": float(np.std(d["weekend_points"])) if len(d) else np.nan,
            })
        ).reset_index(drop=True)

        hist_circ_driver = wp.loc[historical_mask].groupby(["circuitName", "driverId", "driver"], as_index=False).apply(
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
    ctor_round = (
        constructor_weekend_points.copy(deep=True)
        if constructor_weekend_points is not None and not constructor_weekend_points.empty
        else _constructor_round_points(wp)
    )
    ctor_row_keys = [
        RaceKey(int(season), int(round_no))
        for season, round_no in zip(ctor_round["season"], ctor_round["round"])
    ]
    ctor_current_mask = pd.Series(
        [key in selected_key_set for key in ctor_row_keys],
        index=ctor_round.index,
        dtype=bool,
    )
    ctor_historical_mask = ctor_round["season"].astype(int) != current_season
    ctor_round["w_current_component"] = [float(race_weights.get(key, 0.0)) for key in ctor_row_keys]
    ctor_round["w_hist_component"] = np.where(
        ctor_current_mask,
        0.0,
        ctor_round["season"].astype(int).apply(lambda yr: _historical_season_weight_hist_only(yr, current_season, decay=historical_season_decay)),
    )

    base_ctor = ctor_round[["constructorId", "constructor"]].drop_duplicates().copy()

    if ctor_current_mask.any():
        current_ctor = ctor_round.loc[ctor_current_mask].groupby(["constructorId", "constructor"], as_index=False).apply(
            lambda d: pd.Series({
                "overall_current": _weighted_mean(d["constructor_weekend_points"], d["w_current_component"]),
                "dnf_current": float((d["dnf_drivers"].sum() + 1.0) / (2.0 * len(d) + 15.0)) if len(d) else np.nan,
                "vol_current": float(np.std(d["constructor_weekend_points"])) if len(d) else np.nan,
            })
        ).reset_index(drop=True)
    else:
        current_ctor = pd.DataFrame(columns=["constructorId", "constructor", "overall_current", "dnf_current", "vol_current"])

    if ctor_historical_mask.any():
        hist_ctor = ctor_round.loc[ctor_historical_mask].groupby(["constructorId", "constructor"], as_index=False).apply(
            lambda d: pd.Series({
                "overall_hist": _weighted_mean(d["constructor_weekend_points"], d["w_hist_component"]),
                "dnf_hist": float((d["dnf_drivers"].sum() + 1.0) / (2.0 * len(d) + 15.0)) if len(d) else np.nan,
                "vol_hist": float(np.std(d["constructor_weekend_points"])) if len(d) else np.nan,
            })
        ).reset_index(drop=True)

        hist_circ_ctor = ctor_round.loc[ctor_historical_mask].groupby(["circuitName", "constructorId", "constructor"], as_index=False).apply(
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
        base["historical_horizon_expected_points"] = 0.0
        base["historical_next_race_expected_points"] = np.nan
        for index, (circuit, w) in enumerate(zip(upcoming_circuits, horizon_weights)):
            sub = hist_circ_driver[hist_circ_driver["circuitName"].str.contains(circuit, case=False, na=False)].copy()
            tmp = base.merge(sub[["driverId", "circ_hist"]], on="driverId", how="left")
            hist_value = tmp["circ_hist"].fillna(tmp["overall_hist"])
            current_value = tmp["overall_current"]
            if index == 0:
                base["historical_next_race_expected_points"] = hist_value.to_numpy()
            base["historical_horizon_expected_points"] += float(w) * hist_value.fillna(0.0).to_numpy()
            use = _blend_series(current_value, hist_value, current_share)
            use = use.fillna(current_value).fillna(hist_value).fillna(tmp["overall_mean"])
            base["exp_score"] += float(w) * use
        historical_available = base["overall_hist"].notna()
        base.loc[~historical_available, "historical_horizon_expected_points"] = np.nan
        horizon_multiplier = float(sum(float(weight) for weight in horizon_weights))
        base["current_proxy_next_race_expected_points"] = base["overall_current"]
        base["current_proxy_horizon_expected_points"] = base["overall_current"] * horizon_multiplier
        base["next_race_exp_score"] = _blend_series(
            base["current_proxy_next_race_expected_points"],
            base["historical_next_race_expected_points"],
            current_share,
        )
        base["horizon_expected_points"] = base["exp_score"]
        base["current_proxy_volatility"] = base["vol_current"]
        base["historical_volatility"] = base["vol_hist"]
        return base[
            [
                "driverId",
                "driver",
                "exp_score",
                "next_race_exp_score",
                "horizon_expected_points",
                "dnf_rate",
                "volatility",
                "overall_current",
                "overall_hist",
                "overall_mean",
                "current_proxy_next_race_expected_points",
                "current_proxy_horizon_expected_points",
                "historical_next_race_expected_points",
                "historical_horizon_expected_points",
                "current_proxy_volatility",
                "historical_volatility",
            ]
        ]

    def horizon_ctor():
        base = overall_ctor.copy()
        base["exp_score"] = 0.0
        base["historical_horizon_expected_points"] = 0.0
        base["historical_next_race_expected_points"] = np.nan
        for index, (circuit, w) in enumerate(zip(upcoming_circuits, horizon_weights)):
            sub = hist_circ_ctor[hist_circ_ctor["circuitName"].str.contains(circuit, case=False, na=False)].copy()
            tmp = base.merge(sub[["constructorId", "circ_hist"]], on="constructorId", how="left")
            hist_value = tmp["circ_hist"].fillna(tmp["overall_hist"])
            current_value = tmp["overall_current"]
            if index == 0:
                base["historical_next_race_expected_points"] = hist_value.to_numpy()
            base["historical_horizon_expected_points"] += float(w) * hist_value.fillna(0.0).to_numpy()
            use = _blend_series(current_value, hist_value, current_share)
            use = use.fillna(current_value).fillna(hist_value).fillna(tmp["overall_mean"])
            base["exp_score"] += float(w) * use
        historical_available = base["overall_hist"].notna()
        base.loc[~historical_available, "historical_horizon_expected_points"] = np.nan
        horizon_multiplier = float(sum(float(weight) for weight in horizon_weights))
        base["current_proxy_next_race_expected_points"] = base["overall_current"]
        base["current_proxy_horizon_expected_points"] = base["overall_current"] * horizon_multiplier
        base["next_race_exp_score"] = _blend_series(
            base["current_proxy_next_race_expected_points"],
            base["historical_next_race_expected_points"],
            current_share,
        )
        base["horizon_expected_points"] = base["exp_score"]
        base["current_proxy_volatility"] = base["vol_current"]
        base["historical_volatility"] = base["vol_hist"]
        return base[
            [
                "constructorId",
                "constructor",
                "exp_score",
                "next_race_exp_score",
                "horizon_expected_points",
                "dnf_rate",
                "volatility",
                "overall_current",
                "overall_hist",
                "overall_mean",
                "current_proxy_next_race_expected_points",
                "current_proxy_horizon_expected_points",
                "historical_next_race_expected_points",
                "historical_horizon_expected_points",
                "current_proxy_volatility",
                "historical_volatility",
            ]
        ]

    return horizon_driver(), horizon_ctor()


# Sprint fallback depends on both non-Sprint components being resolved first.
_SHADOW_COMPONENTS = ("qualifying", "race", "sprint")


def _blend_component_values(
    current_value: float,
    historical_value: float,
    current_weight: float,
    historical_weight: float,
) -> tuple[float, str]:
    current_ok = pd.notna(current_value)
    historical_ok = pd.notna(historical_value)
    if current_ok and historical_ok:
        current_configured = max(0.0, float(current_weight))
        historical_configured = max(0.0, float(historical_weight))
        denominator = current_configured + historical_configured
        if denominator <= 0:
            return (float(current_value) + float(historical_value)) / 2.0, "blended_equal_weights"
        return (
            float(current_value) * current_configured
            + float(historical_value) * historical_configured
        ) / denominator, "blended_current_historical"
    if current_ok:
        return float(current_value), "current_only"
    if historical_ok:
        return float(historical_value), "historical_only"
    return float("nan"), "unavailable"


def _shadow_component_forecast(
    observations: pd.DataFrame,
    upcoming_events: Iterable[UpcomingEvent],
    *,
    asset_id_col: str,
    asset_name_col: str,
    current_season: int,
    current_season_weight: float,
    past_season_weight: float,
    recency_decay: float,
    historical_season_decay: float,
    selected_race_keys: Iterable[RaceKey | tuple[int, int]] | None,
    selected_race_weights: Mapping[RaceKey | tuple[int, int], float] | None,
) -> pd.DataFrame:
    """Forecast qualifying/Sprint/race components without changing legacy EV fields."""
    data = observations.copy(deep=True)
    events = tuple(upcoming_events)
    output_columns = [
        asset_id_col,
        asset_name_col,
        *[f"shadow_next_{component}_ev" for component in _SHADOW_COMPONENTS],
        "shadow_next_total_ev",
        *[f"shadow_horizon_{component}_ev" for component in _SHADOW_COMPONENTS],
        "shadow_horizon_total_ev",
    ]
    if data.empty or not events:
        return pd.DataFrame(columns=output_columns)
    if "qualifying_points" not in data.columns:
        data["qualifying_points"] = data.get("quali_points")
    if "sprint_applicable" not in data.columns:
        sprint_values = pd.to_numeric(data.get("sprint_points"), errors="coerce")
        data["sprint_applicable"] = sprint_values.notna() & sprint_values.ne(0)
    if "sprint_observed" not in data.columns:
        data["sprint_observed"] = data["sprint_applicable"]
    data["_event_key"] = [
        RaceKey(int(season), int(round_no))
        for season, round_no in zip(data["season"], data["round"])
    ]
    data["_asset_id"] = data[asset_id_col].astype(str)
    current_rows = data[data["season"].astype(int) == int(current_season)].copy()
    catalogue_input = current_rows.rename(
        columns={"weekend_points": "fantasy_points", "circuitName": "race_name"}
    )
    catalogue_input["is_played"] = 1
    available = available_races(catalogue_input, season=int(current_season))
    selected = (
        resolve_selected_races(available, "All")
        if selected_race_keys is None
        else resolve_selected_races(available, "Custom", custom_keys=selected_race_keys)
    )
    selected_set = set(selected.included)
    raw_weights = selected_race_weights or recency_weights(selected, recency_decay)
    current_weights = {
        canonical_race_key(
            key.season if isinstance(key, RaceKey) else key[0],
            key.round if isinstance(key, RaceKey) else key[1],
        ): float(weight)
        for key, weight in raw_weights.items()
    }
    sprint_keys = sorted(
        {
            key
            for key, applicable in zip(current_rows["_event_key"], current_rows["sprint_applicable"])
            if key in selected_set and bool(applicable)
        }
    )
    sprint_weights = recency_weights(sprint_keys, recency_decay)
    data["_historical_weight"] = data["season"].astype(int).apply(
        lambda season: _historical_season_weight_hist_only(
            season, int(current_season), decay=historical_season_decay
        )
        if int(season) != int(current_season)
        else 0.0
    )
    component_columns = {
        "qualifying": "qualifying_points",
        "sprint": "sprint_points",
        "race": "race_points",
    }

    sprint_valid = (
        data["sprint_applicable"].fillna(False).astype(bool)
        & data["sprint_observed"].fillna(False).astype(bool)
        & pd.to_numeric(data["sprint_points"], errors="coerce").notna()
    )
    base_points = (
        pd.to_numeric(data["qualifying_points"], errors="coerce")
        + pd.to_numeric(data["race_points"], errors="coerce")
    )
    ratio_rows = data[sprint_valid & base_points.gt(0)].copy()
    if ratio_rows.empty:
        field_sprint_ratio = float("nan")
    else:
        ratio_base = (
            pd.to_numeric(ratio_rows["qualifying_points"], errors="coerce")
            + pd.to_numeric(ratio_rows["race_points"], errors="coerce")
        )
        field_sprint_ratio = float(
            (pd.to_numeric(ratio_rows["sprint_points"], errors="coerce") / ratio_base)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .median()
        )

    def weighted_value(rows: pd.DataFrame, component: str, weights: Mapping[RaceKey, float]) -> tuple[float, int]:
        if rows.empty:
            return float("nan"), 0
        values = pd.to_numeric(rows[component_columns[component]], errors="coerce")
        valid = values.notna()
        if component == "sprint":
            valid &= rows["sprint_applicable"].fillna(False).astype(bool)
            valid &= rows["sprint_observed"].fillna(False).astype(bool)
        rows = rows[valid].copy()
        values = values[valid]
        if rows.empty:
            return float("nan"), 0
        row_weights = pd.Series(
            [float(weights.get(key, 0.0)) for key in rows["_event_key"]],
            index=rows.index,
            dtype=float,
        )
        positive = row_weights > 0
        if not positive.any():
            return float("nan"), 0
        return float(np.average(values[positive], weights=row_weights[positive])), int(positive.sum())

    rows_out: list[dict[str, object]] = []
    historical = data[data["season"].astype(int) != int(current_season)].copy()
    for asset_id, asset_rows in data.groupby("_asset_id", sort=True):
        name_values = asset_rows[asset_name_col].dropna().astype(str)
        result: dict[str, object] = {
            asset_id_col: asset_rows.iloc[0][asset_id_col],
            asset_name_col: name_values.iloc[0] if not name_values.empty else str(asset_id),
            "shadow_sprint_field_ratio": field_sprint_ratio,
        }
        current_asset = current_rows[
            (current_rows["_asset_id"] == asset_id)
            & current_rows["_event_key"].isin(selected_set)
        ]
        historical_asset = historical[historical["_asset_id"] == asset_id]
        current_estimates: dict[str, float] = {}
        historical_overall: dict[str, float] = {}
        for component in _SHADOW_COMPONENTS:
            weights = sprint_weights if component == "sprint" else current_weights
            current_value, current_count = weighted_value(current_asset, component, weights)
            historical_value, historical_count = weighted_value(
                historical_asset,
                component,
                {
                    key: float(weight)
                    for key, weight in zip(
                        historical_asset["_event_key"],
                        historical_asset["_historical_weight"],
                    )
                },
            )
            current_estimates[component] = current_value
            historical_overall[component] = historical_value
            result[f"shadow_{component}_current_estimate"] = current_value
            result[f"shadow_{component}_historical_overall_estimate"] = historical_value
            result[f"shadow_{component}_current_valid_count"] = current_count
            result[f"shadow_{component}_historical_valid_count"] = historical_count

        event_values: list[dict[str, float]] = []
        event_sources: list[dict[str, str]] = []
        for event in events:
            component_values: dict[str, float] = {}
            component_sources: dict[str, str] = {}
            circuit_needle = event.circuit.split(" Circuit")[0].strip()
            circuit_rows = historical_asset[
                historical_asset["circuitName"].astype(str).str.contains(
                    circuit_needle, case=False, na=False, regex=False
                )
            ]
            for component in _SHADOW_COMPONENTS:
                historical_circuit, _count = weighted_value(
                    circuit_rows,
                    component,
                    {
                        key: float(weight)
                        for key, weight in zip(
                            circuit_rows["_event_key"], circuit_rows["_historical_weight"]
                        )
                    },
                )
                historical_value = (
                    historical_circuit
                    if pd.notna(historical_circuit)
                    else historical_overall[component]
                )
                historical_source = (
                    "historical_circuit"
                    if pd.notna(historical_circuit)
                    else "historical_overall"
                )
                value, source = _blend_component_values(
                    current_estimates[component],
                    historical_value,
                    current_season_weight,
                    past_season_weight,
                )
                if source == "historical_only":
                    source = historical_source
                elif source == "blended_current_historical":
                    source = f"blended_current_{historical_source}"
                if component == "sprint" and event.format == WeekendFormat.NORMAL:
                    value = 0.0
                    source = "not_applicable_normal_weekend"
                elif component == "sprint" and pd.isna(value):
                    base_value = sum(
                        component_values.get(key, float("nan"))
                        for key in ("qualifying", "race")
                    )
                    if pd.notna(field_sprint_ratio) and pd.notna(base_value):
                        value = float(base_value) * float(field_sprint_ratio)
                        source = "field_sprint_to_non_sprint_ratio_fallback"
                component_values[component] = value
                component_sources[component] = source
            event_values.append(component_values)
            event_sources.append(component_sources)

        first_values = event_values[0]
        first_sources = event_sources[0]
        for component in _SHADOW_COMPONENTS:
            result[f"shadow_next_{component}_ev"] = first_values[component]
            result[f"shadow_next_{component}_source"] = first_sources[component]
        next_parts = [first_values[component] for component in _SHADOW_COMPONENTS]
        result["shadow_next_total_ev"] = (
            float(sum(next_parts)) if all(pd.notna(value) for value in next_parts) else float("nan")
        )
        for component in _SHADOW_COMPONENTS:
            required_values = [
                values[component]
                for event, values in zip(events, event_values)
                if component != "sprint" or event.format == WeekendFormat.SPRINT
            ]
            if not required_values:
                horizon_value = 0.0
            elif all(pd.notna(value) for value in required_values):
                horizon_value = float(
                    sum(
                        event.horizon_weight * values[component]
                        for event, values in zip(events, event_values)
                        if component != "sprint" or event.format == WeekendFormat.SPRINT
                    )
                )
            else:
                horizon_value = float("nan")
            result[f"shadow_horizon_{component}_ev"] = horizon_value
        horizon_parts = [result[f"shadow_horizon_{component}_ev"] for component in _SHADOW_COMPONENTS]
        result["shadow_horizon_total_ev"] = (
            float(sum(horizon_parts))
            if all(pd.notna(value) for value in horizon_parts)
            else float("nan")
        )
        result["shadow_component_status"] = (
            "complete" if pd.notna(result["shadow_next_total_ev"]) else "component_unavailable"
        )
        rows_out.append(result)
    return pd.DataFrame(rows_out)


def expected_scores_horizon_by_component(
    weekend_points: pd.DataFrame,
    upcoming_events: Iterable[UpcomingEvent],
    *,
    current_season_weight: float = 1.0,
    past_season_weight: float = 1.0,
    recency_decay: float = 0.95,
    historical_season_decay: float = 0.75,
    selected_race_keys: Iterable[RaceKey | tuple[int, int]] | None = None,
    selected_race_weights: Mapping[RaceKey | tuple[int, int], float] | None = None,
    current_season: int | None = None,
    constructor_weekend_points: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return format-aware driver and constructor forecasts in shadow-only columns."""
    wp = weekend_points.copy(deep=True)
    if wp.empty:
        return pd.DataFrame(), pd.DataFrame()
    season = int(current_season or wp["season"].max())
    driver_shadow = _shadow_component_forecast(
        wp,
        upcoming_events,
        asset_id_col="driverId",
        asset_name_col="driver",
        current_season=season,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        recency_decay=recency_decay,
        historical_season_decay=historical_season_decay,
        selected_race_keys=selected_race_keys,
        selected_race_weights=selected_race_weights,
    )
    constructor_rounds = (
        constructor_weekend_points.copy(deep=True)
        if constructor_weekend_points is not None and not constructor_weekend_points.empty
        else _constructor_round_points(wp)
    )
    constructor_shadow = _shadow_component_forecast(
        constructor_rounds,
        upcoming_events,
        asset_id_col="constructorId",
        asset_name_col="constructor",
        current_season=season,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        recency_decay=recency_decay,
        historical_season_decay=historical_season_decay,
        selected_race_keys=selected_race_keys,
        selected_race_weights=selected_race_weights,
    )
    return driver_shadow, constructor_shadow

def apply_no_negative_expectation(
    weekend_points: pd.DataFrame,
    upcoming_circuits: List[str],
    horizon_weights: List[float],
    current_season_weight: float = 1.0,
    past_season_weight: float = 1.0,
    recency_decay: float = 0.95,
    historical_season_decay: float = 0.75,
) -> pd.Series:
    """Approx EV under No Negative for drivers using the same current-vs-historical split."""
    wp = weekend_points.copy()
    wp["nn_points"] = wp["weekend_points"].clip(lower=0)

    current_season = int(wp["season"].max())
    current_mask = wp["season"].astype(int) == current_season
    latest_round = int(wp.loc[current_mask, "round"].max()) if current_mask.any() else 0
    current_share = _adjust_current_share(
        _current_season_share(latest_round),
        current_weight=current_season_weight,
        past_weight=past_season_weight,
    )

    wp["w_current_component"] = np.where(
        current_mask,
        wp["round"].astype(int).apply(lambda r: _current_round_weight(r, latest_round, decay=recency_decay)),
        0.0,
    )
    wp["w_hist_component"] = np.where(
        current_mask,
        0.0,
        wp["season"].astype(int).apply(lambda yr: _historical_season_weight_hist_only(yr, current_season, decay=historical_season_decay)),
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
