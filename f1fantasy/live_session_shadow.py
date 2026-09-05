from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping

import pandas as pd

from f1fantasy.weekend_state import (
    EventKey,
    SessionKind,
    SessionState,
    SessionStatus,
    WeekendFormat,
)


SESSION_SCORE_COLUMNS: dict[SessionKind, str] = {
    SessionKind.PRACTICE_1: "FP1_score",
    SessionKind.PRACTICE_2: "FP2_score",
    SessionKind.PRACTICE_3: "FP3_score",
    SessionKind.SPRINT_QUALIFYING: "SQ_score",
}

SESSION_DISPLAY_LABELS: dict[SessionKind, str] = {
    SessionKind.PRACTICE_1: "FP1",
    SessionKind.PRACTICE_2: "FP2",
    SessionKind.PRACTICE_3: "FP3",
    SessionKind.SPRINT_QUALIFYING: "SQ",
}

SESSION_WEIGHTS: dict[WeekendFormat, dict[SessionKind, float]] = {
    WeekendFormat.NORMAL: {
        SessionKind.PRACTICE_1: 1.0,
        SessionKind.PRACTICE_2: 2.0,
        SessionKind.PRACTICE_3: 3.0,
    },
    WeekendFormat.SPRINT: {
        SessionKind.PRACTICE_1: 1.0,
        SessionKind.SPRINT_QUALIFYING: 3.0,
    },
}

DRIVER_LIVE_SHADOW_COLUMNS = [
    *SESSION_SCORE_COLUMNS.values(),
    "sessions_used",
    "session_count",
    "weight_sum",
    "baseline_ev",
    "live_session_score",
    "live_session_rank",
    "live_only_ev",
]

CONSTRUCTOR_LIVE_SHADOW_COLUMNS = [
    "baseline_ev",
    "driver_coverage",
    "valid_driver_count",
    "expected_driver_count",
    "live_session_score",
    "live_session_rank",
    "live_only_ev",
    "constructor_live_session_score",
    "constructor_live_session_rank",
    "constructor_live_only_ev",
]

LIVE_SESSION_PRODUCTION_COLUMNS = [
    "baseline_next_race_expected_points",
    "baseline_horizon_expected_points",
    "live_session_emphasis",
    "adjusted_ev",
    "live_session_ev_difference",
]


@dataclass(frozen=True)
class WeightedLiveScore:
    live_session_score: float | None
    sessions_used: tuple[str, ...]
    session_count: int
    weight_sum: float


@dataclass(frozen=True)
class LiveSessionShadow:
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    session_scores: pd.DataFrame
    diagnostics: dict[str, Any]


def validate_live_session_emphasis(value: Any) -> float:
    """Return a finite production blend weight in the inclusive 0-to-1 range."""
    if isinstance(value, bool):
        raise ValueError("Live session emphasis must be a number from 0 to 1.")
    try:
        emphasis = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Live session emphasis must be a number from 0 to 1.") from exc
    if not math.isfinite(emphasis) or not 0.0 <= emphasis <= 1.0:
        raise ValueError("Live session emphasis must be a number from 0 to 1.")
    return emphasis


def completed_live_session_labels(
    session_states: Iterable[SessionState],
    *,
    forecast_event: EventKey | None = None,
) -> tuple[str, ...]:
    """Return compact ordered labels for completed prediction-eligible sessions."""
    completed = {
        state.kind
        for state in session_states
        if state.status == SessionStatus.COMPLETE
        and state.kind in SESSION_DISPLAY_LABELS
        and (forecast_event is None or state.event == forecast_event)
    }
    return tuple(
        label for kind, label in SESSION_DISPLAY_LABELS.items() if kind in completed
    )


def blend_live_session_ev(
    baseline_ev: Any,
    live_only_ev: Any,
    emphasis: Any,
) -> float:
    """Blend one baseline/live EV pair without inventing missing forecasts."""
    weight = validate_live_session_emphasis(emphasis)
    baseline = pd.to_numeric(pd.Series([baseline_ev]), errors="coerce").iloc[0]
    if pd.isna(baseline) or not math.isfinite(float(baseline)):
        return float("nan")
    baseline = float(baseline)
    if weight == 0.0:
        return baseline
    live = pd.to_numeric(pd.Series([live_only_ev]), errors="coerce").iloc[0]
    if pd.isna(live) or not math.isfinite(float(live)):
        return baseline
    return float((1.0 - weight) * baseline + weight * float(live))


def apply_live_session_emphasis(
    assets: pd.DataFrame,
    emphasis: Any,
    *,
    baseline_column: str = "baseline_ev",
    live_column: str = "live_only_ev",
) -> pd.DataFrame:
    """Apply the live blend to next-event production fields on a defensive copy."""
    weight = validate_live_session_emphasis(emphasis)
    out = assets.copy(deep=True)
    if baseline_column in out.columns:
        baseline = pd.to_numeric(out[baseline_column], errors="coerce")
    elif "next_race_expected_points" in out.columns:
        baseline = pd.to_numeric(out["next_race_expected_points"], errors="coerce")
    else:
        baseline = pd.Series(float("nan"), index=out.index, dtype=float)
    live = pd.to_numeric(
        out.get(live_column, pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )

    # The explicit branch is intentional: the default must reproduce the
    # production baseline exactly, including when live classifications exist.
    if weight == 0.0:
        adjusted = baseline.copy()
    else:
        adjusted = pd.Series(
            [
                blend_live_session_ev(base_value, live_value, weight)
                for base_value, live_value in zip(baseline, live)
            ],
            index=out.index,
            dtype=float,
        )
    difference = adjusted - baseline

    out["baseline_next_race_expected_points"] = baseline
    out["live_session_emphasis"] = weight
    out["adjusted_ev"] = adjusted
    out["live_session_ev_difference"] = difference

    if "horizon_expected_points" in out.columns:
        horizon = pd.to_numeric(out["horizon_expected_points"], errors="coerce")
        out["baseline_horizon_expected_points"] = horizon
        out["horizon_expected_points"] = horizon.where(
            difference.isna(), horizon + difference
        )

    for column in ("next_race_expected_points", "next_race_exp_score", "exp_score"):
        if column in out.columns:
            out[column] = adjusted
    if "nn_exp_score" in out.columns:
        no_negative = pd.to_numeric(out["nn_exp_score"], errors="coerce")
        out["nn_exp_score"] = no_negative.where(
            difference.isna(), no_negative + difference
        )
    return out


def _coerce_weekend_format(value: WeekendFormat | str) -> WeekendFormat:
    if isinstance(value, WeekendFormat):
        return value
    return WeekendFormat(str(value).strip().casefold())


def _coerce_session_kind(value: SessionKind | str) -> SessionKind | None:
    if isinstance(value, SessionKind):
        return value
    text = str(value or "").strip().casefold()
    aliases = {
        "fp1": SessionKind.PRACTICE_1,
        "fp2": SessionKind.PRACTICE_2,
        "fp3": SessionKind.PRACTICE_3,
        "sq": SessionKind.SPRINT_QUALIFYING,
        "sprint qualifying": SessionKind.SPRINT_QUALIFYING,
        "sprint shootout": SessionKind.SPRINT_QUALIFYING,
    }
    if text in aliases:
        return aliases[text]
    try:
        return SessionKind(text)
    except ValueError:
        return None


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _active_mask(frame: pd.DataFrame) -> pd.Series:
    for column in ("is_active", "IsActive", "active", "selectable"):
        if column not in frame.columns:
            continue
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
        text = values.astype(str).str.strip().str.casefold()
        return numeric.eq(1) | text.isin({"true", "yes", "active", "selectable"})
    return pd.Series(True, index=frame.index, dtype=bool)


def position_to_live_score(position: Any, field_size: Any) -> float:
    """Map a valid classified position onto the inclusive 1-to-0 scale."""
    parsed_position = pd.to_numeric(position, errors="coerce")
    parsed_size = pd.to_numeric(field_size, errors="coerce")
    if pd.isna(parsed_position) or pd.isna(parsed_size):
        raise ValueError("Position and classified field size must be numeric.")
    if not float(parsed_position).is_integer() or not float(parsed_size).is_integer():
        raise ValueError("Position and classified field size must be integers.")
    position_value = int(parsed_position)
    size_value = int(parsed_size)
    if size_value <= 1:
        raise ValueError("A classified field must contain at least two drivers.")
    if position_value < 1 or position_value > size_value:
        raise ValueError("Position must be within the classified field.")
    return float((size_value - position_value) / (size_value - 1))


def normalise_session_positions(results: pd.DataFrame) -> pd.DataFrame:
    """Return one validated classification with an added position_score column."""
    frame = results.copy(deep=True) if isinstance(results, pd.DataFrame) else pd.DataFrame()
    if frame.empty or "position" not in frame.columns:
        raise ValueError("Session classification has no position rows.")
    if "is_classified" in frame.columns:
        classified = frame["is_classified"]
        numeric = pd.to_numeric(classified, errors="coerce")
        text = classified.astype(str).str.strip().str.casefold()
        frame = frame[numeric.eq(1) | text.isin({"true", "yes", "classified"})].copy()
    if frame.empty:
        raise ValueError("Session classification has no classified drivers.")

    positions = pd.to_numeric(frame["position"], errors="coerce")
    if positions.isna().any() or not positions.map(lambda value: float(value).is_integer()).all():
        raise ValueError("Session positions must be complete integers.")
    positions = positions.astype(int)
    field_size = int(len(frame))
    if field_size <= 1:
        raise ValueError("A classified session must contain at least two drivers.")
    if positions.duplicated().any():
        raise ValueError("Session positions must be unique.")
    if set(positions.tolist()) != set(range(1, field_size + 1)):
        raise ValueError("Session positions must form a complete 1..N classification.")
    if "human_driver_id" in frame.columns:
        identities = frame["human_driver_id"].dropna().astype(str)
        if identities.duplicated().any():
            raise ValueError("A human driver appears more than once in the session classification.")

    frame["position"] = positions
    frame["classified_field_size"] = field_size
    frame["position_score"] = positions.map(
        lambda value: position_to_live_score(value, field_size)
    )
    return frame


def weighted_live_session_score(
    session_scores: Mapping[SessionKind | str, Any],
    weekend_format: WeekendFormat | str,
) -> WeightedLiveScore:
    """Combine only available eligible sessions, renormalising their fixed weights."""
    format_value = _coerce_weekend_format(weekend_format)
    available: dict[SessionKind, float] = {}
    for raw_kind, raw_score in session_scores.items():
        kind = _coerce_session_kind(raw_kind)
        score = pd.to_numeric(raw_score, errors="coerce")
        if kind is None or kind not in SESSION_WEIGHTS[format_value] or pd.isna(score):
            continue
        score_value = float(score)
        if not math.isfinite(score_value):
            continue
        available[kind] = score_value
    sessions_used = tuple(
        kind.value for kind in SESSION_WEIGHTS[format_value] if kind in available
    )
    weight_sum = float(
        sum(SESSION_WEIGHTS[format_value][kind] for kind in available)
    )
    if not available or weight_sum <= 0:
        return WeightedLiveScore(None, (), 0, 0.0)
    score = sum(
        SESSION_WEIGHTS[format_value][kind] * value
        for kind, value in available.items()
    ) / weight_sum
    return WeightedLiveScore(float(score), sessions_used, len(available), weight_sum)


def assign_ev_ladder(
    assets: pd.DataFrame,
    *,
    score_column: str = "live_session_score",
    baseline_column: str = "baseline_ev",
    stable_id_column: str,
) -> pd.DataFrame:
    """Allocate the eligible baseline EV multiset by deterministic live ranking."""
    out = assets.copy(deep=True)
    out["live_session_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["live_only_ev"] = pd.Series(float("nan"), index=out.index, dtype=float)
    scores = pd.to_numeric(
        out.get(score_column, pd.Series(index=out.index, dtype=float)), errors="coerce"
    )
    baselines = pd.to_numeric(
        out.get(baseline_column, pd.Series(index=out.index, dtype=float)), errors="coerce"
    )
    stable_ids = out.get(
        stable_id_column, pd.Series("", index=out.index, dtype=object)
    ).fillna("").astype(str)
    eligible = scores.notna() & baselines.notna() & scores.map(
        lambda value: math.isfinite(float(value)) if pd.notna(value) else False
    )
    if not eligible.any():
        return out
    ranked = pd.DataFrame(
        {
            "_score": scores[eligible],
            "_baseline": baselines[eligible],
            "_stable_id": stable_ids[eligible],
            "_input_order": range(int(eligible.sum())),
        },
        index=out.index[eligible],
    ).sort_values(
        ["_score", "_baseline", "_stable_id", "_input_order"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    ladder = sorted((float(value) for value in baselines[eligible]), reverse=True)
    for rank, (row_index, live_ev) in enumerate(zip(ranked.index, ladder), start=1):
        out.at[row_index, "live_session_rank"] = rank
        out.at[row_index, "live_only_ev"] = live_ev
    return out


def _empty_driver_shadow(drivers: pd.DataFrame, baseline_column: str) -> pd.DataFrame:
    out = drivers.copy(deep=True)
    for column in SESSION_SCORE_COLUMNS.values():
        out[column] = pd.Series(float("nan"), index=out.index, dtype=float)
    out["sessions_used"] = pd.Series([()] * len(out), index=out.index, dtype=object)
    out["session_count"] = 0
    out["weight_sum"] = 0.0
    out["baseline_ev"] = pd.to_numeric(
        out.get(baseline_column, pd.Series(index=out.index, dtype=float)), errors="coerce"
    )
    out["live_session_score"] = pd.Series(float("nan"), index=out.index, dtype=float)
    out["live_session_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["live_only_ev"] = pd.Series(float("nan"), index=out.index, dtype=float)
    return out


def _complete_session_scores(
    session_results: pd.DataFrame,
    session_states: Iterable[SessionState],
    weekend_format: WeekendFormat,
    forecast_event: EventKey | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], tuple[int, int] | None]:
    results = session_results.copy(deep=True) if isinstance(session_results, pd.DataFrame) else pd.DataFrame()
    eligible_kinds = set(SESSION_WEIGHTS[weekend_format])
    complete_states = [
        state
        for state in tuple(session_states or ())
        if state.status == SessionStatus.COMPLETE
        and state.supported
        and state.kind in eligible_kinds
        and (forecast_event is None or state.event == forecast_event)
    ]
    if not complete_states:
        event = (
            (forecast_event.season, forecast_event.round)
            if forecast_event is not None
            else None
        )
        return pd.DataFrame(), [], event
    active_event = (
        (forecast_event.season, forecast_event.round)
        if forecast_event is not None
        else max((state.event.season, state.event.round) for state in complete_states)
    )
    complete_states = [
        state
        for state in complete_states
        if (state.event.season, state.event.round) == active_event
    ]
    diagnostics: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for state in sorted(complete_states, key=lambda item: item.kind.value):
        if results.empty or not {"season", "round", "session_kind"}.issubset(results.columns):
            diagnostics.append(
                {
                    "session_kind": state.kind.value,
                    "status": "unavailable",
                    "diagnostic": "Complete session state has no matching normalized result table.",
                }
            )
            continue
        rows = results[
            pd.to_numeric(results["season"], errors="coerce").eq(state.event.season)
            & pd.to_numeric(results["round"], errors="coerce").eq(state.event.round)
            & results["session_kind"].astype(str).eq(state.kind.value)
        ].copy()
        try:
            normalised = normalise_session_positions(rows)
        except ValueError as exc:
            diagnostics.append(
                {
                    "session_kind": state.kind.value,
                    "status": "unavailable",
                    "rows": int(len(rows)),
                    "diagnostic": str(exc),
                }
            )
            continue
        normalised["session_kind"] = state.kind.value
        normalised["session_weight"] = SESSION_WEIGHTS[weekend_format][state.kind]
        identity_available = normalised.get(
            "human_driver_id", pd.Series(index=normalised.index, dtype=object)
        ).notna()
        if "identity_match_status" in normalised.columns:
            identity_available &= (
                normalised["identity_match_status"].isna()
                | normalised["identity_match_status"].astype(str).str.casefold().eq("matched")
            )
        frames.append(normalised)
        diagnostics.append(
            {
                "session_kind": state.kind.value,
                "status": "used",
                "rows": int(len(normalised)),
                "classified_field_size": int(len(normalised)),
                "identity_mapped_rows": int(identity_available.sum()),
                "identity_unavailable_rows": int((~identity_available).sum()),
                "weight": float(SESSION_WEIGHTS[weekend_format][state.kind]),
            }
        )
    session_scores = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return session_scores, diagnostics, active_event


def build_driver_live_shadow(
    drivers: pd.DataFrame,
    session_results: pd.DataFrame,
    session_states: Iterable[SessionState],
    weekend_format: WeekendFormat | str,
    *,
    baseline_column: str = "next_race_expected_points",
    forecast_event: EventKey | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach human-driver live scores to active current Fantasy driver assets only."""
    format_value = _coerce_weekend_format(weekend_format)
    out = _empty_driver_shadow(drivers, baseline_column)
    session_scores, session_diagnostics, active_event = _complete_session_scores(
        session_results, session_states, format_value, forecast_event
    )
    diagnostics: dict[str, Any] = {
        "status": "unavailable" if session_scores.empty else "available",
        "active_event": active_event,
        "weekend_format": format_value.value,
        "session_weights": {
            kind.value: weight for kind, weight in SESSION_WEIGHTS[format_value].items()
        },
        "sessions": session_diagnostics,
        "ambiguous_active_human_ids": [],
        "scored_active_driver_count": 0,
    }
    if out.empty or session_scores.empty or "human_driver_id" not in out.columns:
        return out, session_scores, diagnostics

    active = _active_mask(out)
    human_ids = out["human_driver_id"].fillna("").astype(str)
    active_human_counts = human_ids[active & human_ids.ne("")].value_counts()
    ambiguous_ids = sorted(active_human_counts[active_human_counts.gt(1)].index.tolist())
    diagnostics["ambiguous_active_human_ids"] = ambiguous_ids

    def identity_is_usable(row: Mapping[str, Any]) -> bool:
        human_id = row.get("human_driver_id")
        identity_status = row.get("identity_match_status")
        return bool(
            human_id is not None
            and pd.notna(human_id)
            and pd.notna(row.get("position_score"))
            and (
                identity_status is None
                or pd.isna(identity_status)
                or str(identity_status).casefold() == "matched"
            )
        )

    score_lookup = {
        (str(row["human_driver_id"]), str(row["session_kind"])): float(row["position_score"])
        for row in session_scores.to_dict("records")
        if identity_is_usable(row)
    }
    for row_index in out.index[active]:
        human_id = human_ids.at[row_index]
        if not human_id or human_id in ambiguous_ids:
            continue
        available_scores: dict[SessionKind, float] = {}
        for kind, column in SESSION_SCORE_COLUMNS.items():
            score = score_lookup.get((human_id, kind.value))
            if score is None:
                continue
            out.at[row_index, column] = score
            available_scores[kind] = score
        weighted = weighted_live_session_score(available_scores, format_value)
        out.at[row_index, "sessions_used"] = weighted.sessions_used
        out.at[row_index, "session_count"] = weighted.session_count
        out.at[row_index, "weight_sum"] = weighted.weight_sum
        if weighted.live_session_score is not None:
            out.at[row_index, "live_session_score"] = weighted.live_session_score

    out = assign_ev_ladder(
        out,
        score_column="live_session_score",
        baseline_column="baseline_ev",
        stable_id_column="human_driver_id",
    )
    diagnostics["scored_active_driver_count"] = int(
        pd.to_numeric(out["live_session_score"], errors="coerce").notna().sum()
    )
    diagnostics["driver_ladder_multiset_preserved"] = sorted(
        pd.to_numeric(
            out.loc[out["live_session_score"].notna(), "baseline_ev"], errors="coerce"
        ).dropna().tolist()
    ) == sorted(pd.to_numeric(out["live_only_ev"], errors="coerce").dropna().tolist())
    return out, session_scores, diagnostics


def _constructor_team_key(row: Mapping[str, Any]) -> str:
    for column in ("team_reference",):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return f"reference:{_normalise_text(value)}"
    return f"name:{_normalise_text(row.get('name', row.get('team')))}"


def _driver_team_key(row: Mapping[str, Any]) -> str:
    for column in ("team_reference",):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return f"reference:{_normalise_text(value)}"
    return f"name:{_normalise_text(row.get('team'))}"


def build_constructor_live_shadow(
    constructors: pd.DataFrame,
    scored_drivers: pd.DataFrame,
    *,
    baseline_column: str = "next_race_expected_points",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build current-seat constructor means and assign the constructor EV ladder."""
    out = constructors.copy(deep=True)
    out["baseline_ev"] = pd.to_numeric(
        out.get(baseline_column, pd.Series(index=out.index, dtype=float)), errors="coerce"
    )
    out["driver_coverage"] = "0/2"
    out["valid_driver_count"] = 0
    out["expected_driver_count"] = 2
    out["live_session_score"] = pd.Series(float("nan"), index=out.index, dtype=float)
    out["live_session_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["live_only_ev"] = pd.Series(float("nan"), index=out.index, dtype=float)
    drivers = scored_drivers.copy(deep=True)
    if not drivers.empty:
        drivers = drivers[_active_mask(drivers)].copy()
        drivers["_team_key"] = [
            _driver_team_key(row) for row in drivers.to_dict("records")
        ]
    team_diagnostics: list[dict[str, Any]] = []
    for row_index, row in out.iterrows():
        team_key = _constructor_team_key(row.to_dict())
        current = drivers[drivers.get("_team_key", pd.Series(index=drivers.index, dtype=object)).eq(team_key)].copy()
        valid_scores = pd.to_numeric(
            current.get("live_session_score", pd.Series(index=current.index, dtype=float)),
            errors="coerce",
        ).dropna()
        valid_count = int(len(valid_scores))
        out.at[row_index, "valid_driver_count"] = valid_count
        out.at[row_index, "driver_coverage"] = f"{valid_count}/2"
        if valid_count:
            out.at[row_index, "live_session_score"] = float(valid_scores.mean())
        team_diagnostics.append(
            {
                "constructor": str(row.get("name", row.get("id", ""))),
                "current_driver_count": int(len(current)),
                "valid_driver_count": valid_count,
                "expected_driver_count": 2,
                "coverage": f"{valid_count}/2",
            }
        )
    out = assign_ev_ladder(
        out,
        score_column="live_session_score",
        baseline_column="baseline_ev",
        stable_id_column="name" if "name" in out.columns else "id",
    )
    out["constructor_live_session_score"] = out["live_session_score"]
    out["constructor_live_session_rank"] = out["live_session_rank"]
    out["constructor_live_only_ev"] = out["live_only_ev"]
    diagnostics = {
        "status": "available" if out["live_session_score"].notna().any() else "unavailable",
        "teams": team_diagnostics,
        "scored_constructor_count": int(out["live_session_score"].notna().sum()),
        "constructor_ladder_multiset_preserved": sorted(
            pd.to_numeric(
                out.loc[out["live_session_score"].notna(), "baseline_ev"], errors="coerce"
            ).dropna().tolist()
        ) == sorted(pd.to_numeric(out["live_only_ev"], errors="coerce").dropna().tolist()),
    }
    return out, diagnostics


def build_live_session_shadow(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    session_results: pd.DataFrame,
    session_states: Iterable[SessionState],
    weekend_format: WeekendFormat | str,
    *,
    baseline_column: str = "next_race_expected_points",
    forecast_event: EventKey | None = None,
) -> LiveSessionShadow:
    """Build live-only driver and constructor forecasts ready for production blending."""
    driver_rows, session_scores, driver_diagnostics = build_driver_live_shadow(
        drivers,
        session_results,
        session_states,
        weekend_format,
        baseline_column=baseline_column,
        forecast_event=forecast_event,
    )
    constructor_rows, constructor_diagnostics = build_constructor_live_shadow(
        constructors,
        driver_rows,
        baseline_column=baseline_column,
    )
    diagnostics = {
        "label": "Live-session forecast ready for production blending",
        "status": (
            "available"
            if driver_diagnostics.get("status") == "available"
            else "unavailable"
        ),
        "weekend_format": _coerce_weekend_format(weekend_format).value,
        "drivers": driver_diagnostics,
        "constructors": constructor_diagnostics,
        "blend_semantics": (
            "This helper derives baseline and live-only values without choosing a production "
            "weight; the caller applies the user-selected emphasis."
        ),
    }
    return LiveSessionShadow(
        drivers=driver_rows,
        constructors=constructor_rows,
        session_scores=session_scores.copy(deep=True),
        diagnostics=diagnostics,
    )
