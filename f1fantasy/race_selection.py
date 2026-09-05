from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


RACE_PRESETS = ("Last 1", "Last 3", "Last 5", "All", "Custom")


@dataclass(frozen=True, order=True)
class RaceKey:
    """Canonical Grand Prix identity; names and feed-specific IDs are metadata only."""

    season: int
    round: int


@dataclass(frozen=True)
class RaceOption:
    key: RaceKey
    race_name: str


@dataclass(frozen=True)
class RaceSelection:
    """Resolved race keys in deterministic chronological (oldest-first) order."""

    preset: str
    included: tuple[RaceKey, ...]
    excluded: tuple[RaceKey, ...]


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        raise ValueError(f"{field_name} must be a positive integer.")
    numeric_float = float(numeric)
    if not math.isfinite(numeric_float) or not numeric_float.is_integer() or numeric_float <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(numeric_float)


def canonical_race_key(season: Any, round: Any) -> RaceKey:
    """Build a canonical race key from season and championship round."""
    return RaceKey(
        season=_positive_integer(season, "season"),
        round=_positive_integer(round, "round"),
    )


def _coerce_race_key(value: Any) -> RaceKey:
    if isinstance(value, RaceOption):
        return value.key
    if isinstance(value, RaceKey):
        return canonical_race_key(value.season, value.round)
    if isinstance(value, Mapping) and "season" in value and "round" in value:
        return canonical_race_key(value["season"], value["round"])
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return canonical_race_key(value[0], value[1])
    if hasattr(value, "season") and hasattr(value, "round"):
        return canonical_race_key(value.season, value.round)
    raise ValueError("Race keys must provide season and round.")


def _ordered_unique_race_keys(values: Iterable[Any]) -> tuple[RaceKey, ...]:
    return tuple(sorted({_coerce_race_key(value) for value in values}))


def _display_race_name(values: pd.Series, key: RaceKey) -> str:
    names: set[str] = set()
    for value in values.tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            names.add(text)
    if not names:
        return f"Round {key.round}"
    return sorted(names, key=lambda value: (value.casefold(), value))[0]


def available_races(
    observations: pd.DataFrame,
    season: int | None = None,
) -> tuple[RaceOption, ...]:
    """Return completed observed races, ordered chronologically.

    A row is completed when ``is_played == 1`` when that column exists. A
    non-missing ``fantasy_points`` value is also required when the points
    column exists, so genuine zeroes remain valid while incomplete rows do not
    create catalogue entries. If ``is_played`` is absent, a valid points value
    is the completion evidence. Conflicting display names are resolved
    deterministically and never affect race identity.
    """
    if observations is None or observations.empty:
        return ()
    if "season" not in observations.columns or "round" not in observations.columns:
        return ()
    if "is_played" not in observations.columns and "fantasy_points" not in observations.columns:
        return ()

    data = observations.copy(deep=True)
    data["_race_season"] = pd.to_numeric(data["season"], errors="coerce")
    data["_race_round"] = pd.to_numeric(data["round"], errors="coerce")
    integral = (
        data["_race_season"].notna()
        & data["_race_round"].notna()
        & (data["_race_season"] > 0)
        & (data["_race_round"] > 0)
        & (data["_race_season"] % 1 == 0)
        & (data["_race_round"] % 1 == 0)
    )
    data = data[integral].copy()
    if season is not None:
        selected_season = _positive_integer(season, "season")
        data = data[data["_race_season"].astype(int) == selected_season].copy()
    if "is_played" in data.columns:
        played = pd.to_numeric(data["is_played"], errors="coerce").fillna(0).astype(int)
        data = data[played == 1].copy()
    if "fantasy_points" in data.columns:
        points = pd.to_numeric(data["fantasy_points"], errors="coerce")
        data = data[points.notna()].copy()
    if data.empty:
        return ()

    data["_race_season"] = data["_race_season"].astype(int)
    data["_race_round"] = data["_race_round"].astype(int)
    if "race_name" not in data.columns:
        data["race_name"] = pd.NA

    options: list[RaceOption] = []
    for (race_season, race_round), group in data.groupby(
        ["_race_season", "_race_round"],
        sort=True,
    ):
        key = canonical_race_key(race_season, race_round)
        options.append(RaceOption(key=key, race_name=_display_race_name(group["race_name"], key)))
    return tuple(options)


def _canonical_preset(preset: str) -> str:
    normalized = str(preset).strip().casefold()
    by_name = {name.casefold(): name for name in RACE_PRESETS}
    if normalized not in by_name:
        supported = ", ".join(RACE_PRESETS)
        raise ValueError(f"Unsupported race preset {preset!r}. Expected one of: {supported}.")
    return by_name[normalized]


def resolve_selected_races(
    available: Iterable[RaceOption | RaceKey | tuple[int, int]],
    preset: str,
    custom_keys: Iterable[RaceKey | tuple[int, int]] | None = None,
    excluded_keys: Iterable[RaceKey | tuple[int, int]] | None = None,
) -> RaceSelection:
    """Resolve a preset and exclusions into chronological included race keys.

    Unknown but well-formed custom keys are ignored because only currently
    available completed races can be selected. Exclusions are applied after
    preset/custom resolution and duplicate inputs are harmless.
    """
    preset_name = _canonical_preset(preset)
    available_keys = _ordered_unique_race_keys(available)
    available_set = set(available_keys)

    if preset_name == "Custom":
        requested = _ordered_unique_race_keys(custom_keys or ())
        initially_included = tuple(key for key in requested if key in available_set)
    elif preset_name == "All":
        initially_included = available_keys
    else:
        count = int(preset_name.split()[-1])
        initially_included = available_keys[-count:]

    requested_exclusions = set(_ordered_unique_race_keys(excluded_keys or ()))
    applied_exclusions = tuple(key for key in initially_included if key in requested_exclusions)
    included = tuple(key for key in initially_included if key not in requested_exclusions)
    return RaceSelection(
        preset=preset_name,
        included=included,
        excluded=applied_exclusions,
    )


def recency_weights(
    selected_races: RaceSelection | Iterable[RaceKey | RaceOption | tuple[int, int]],
    p: float,
) -> dict[RaceKey, float]:
    """Weight chronological included races using contiguous newest-first positions."""
    numeric_p = pd.to_numeric(p, errors="coerce")
    if pd.isna(numeric_p) or not math.isfinite(float(numeric_p)):
        raise ValueError("Recency p must be between 0 and 1 inclusive.")
    decay = float(numeric_p)
    if decay < 0.0 or decay > 1.0:
        raise ValueError("Recency p must be between 0 and 1 inclusive.")

    values = selected_races.included if isinstance(selected_races, RaceSelection) else selected_races
    keys = _ordered_unique_race_keys(values)
    if not keys:
        return {}
    if decay == 0.0:
        return {key: 1.0 if index == len(keys) - 1 else 0.0 for index, key in enumerate(keys)}
    if decay == 1.0:
        return {key: 1.0 for key in keys}
    return {
        key: decay ** (len(keys) - index - 1)
        for index, key in enumerate(keys)
    }


_WEIGHTED_RESULT_COLUMNS = [
    "asset_id",
    "asset_type",
    "weighted_points",
    "selected_race_count",
    "valid_race_count",
    "missing_race_count",
    "coverage_fraction",
    "weight_sum",
    "selected_race_keys",
    "valid_race_keys",
    "missing_race_keys",
    "has_source_failure",
    "status",
]


def _empty_weighted_results() -> pd.DataFrame:
    return pd.DataFrame(columns=_WEIGHTED_RESULT_COLUMNS)


def _failure_flag(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "none", "ok"}
    return bool(value)


def _failure_identities(
    source_failures: Iterable[Any] | Mapping[Any, Any] | None,
    default_asset_type: str,
) -> set[tuple[str, str]]:
    if source_failures is None:
        return set()
    values: Iterable[Any]
    if isinstance(source_failures, Mapping):
        values = [key for key, value in source_failures.items() if _failure_flag(value)]
    elif isinstance(source_failures, (str, bytes)):
        values = [source_failures]
    else:
        values = source_failures

    failures: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            failures.add((str(value[0]), str(value[1])))
        else:
            failures.add((default_asset_type, str(value)))
    return failures


def weighted_asset_points(
    observations: pd.DataFrame,
    selected_races: RaceSelection | Iterable[RaceKey | RaceOption | tuple[int, int]],
    weights: Mapping[RaceKey | tuple[int, int], float],
    *,
    asset_id_col: str = "PlayerId",
    asset_type: str | None = None,
    asset_type_col: str = "asset_type",
    points_col: str = "fantasy_points",
    played_col: str = "is_played",
    source_failure_col: str = "source_failed",
    source_failures: Iterable[Any] | Mapping[Any, Any] | None = None,
) -> pd.DataFrame:
    """Calculate normalized per-asset means over valid selected observations.

    Missing observations are excluded from the numerator and denominator;
    genuine zero-point observations remain valid. Duplicate rows for the same
    asset and race are collapsed to their mean so a race cannot receive its
    weight more than once. Explicit source failures may be supplied as asset
    IDs, ``(asset_type, asset_id)`` pairs, or a mapping of either to a truthy
    failure value.
    """
    values = selected_races.included if isinstance(selected_races, RaceSelection) else selected_races
    selected_keys = _ordered_unique_race_keys(values)
    selected_set = set(selected_keys)

    normalized_weights: dict[RaceKey, float] = {}
    for raw_key, raw_weight in weights.items():
        key = _coerce_race_key(raw_key)
        numeric_weight = pd.to_numeric(raw_weight, errors="coerce")
        if pd.isna(numeric_weight) or not math.isfinite(float(numeric_weight)) or float(numeric_weight) < 0:
            raise ValueError("Race weights must be finite and non-negative.")
        normalized_weights[key] = float(numeric_weight)
    missing_weight_keys = [key for key in selected_keys if key not in normalized_weights]
    if missing_weight_keys:
        raise ValueError(f"Missing weights for selected races: {missing_weight_keys}.")

    default_asset_type = str(asset_type or "unknown")
    failures = _failure_identities(source_failures, default_asset_type)
    data = observations.copy(deep=True) if observations is not None else pd.DataFrame()
    if data.empty and not failures:
        return _empty_weighted_results()
    if not data.empty:
        resolved_id_col = asset_id_col
        if resolved_id_col not in data.columns and asset_id_col == "PlayerId" and "asset_id" in data.columns:
            resolved_id_col = "asset_id"
        required = [resolved_id_col, "season", "round", points_col]
        missing_columns = [column for column in required if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Observations are missing required columns: {missing_columns}.")
        data["_asset_id"] = data[resolved_id_col].astype("string")
        data = data[data["_asset_id"].notna() & (data["_asset_id"].str.strip() != "")].copy()
        data["_asset_id"] = data["_asset_id"].astype(str)
        if asset_type is not None:
            data["_asset_type"] = str(asset_type)
        elif asset_type_col in data.columns:
            data["_asset_type"] = data[asset_type_col].fillna("unknown").astype(str)
        else:
            data["_asset_type"] = "unknown"
    else:
        data = pd.DataFrame(columns=["_asset_id", "_asset_type"])

    identities = set(zip(data.get("_asset_type", ()), data.get("_asset_id", ()))) | failures
    if not identities:
        return _empty_weighted_results()

    row_failures: set[tuple[str, str]] = set()
    if source_failure_col in data.columns:
        for row in data[["_asset_type", "_asset_id", source_failure_col]].itertuples(index=False, name=None):
            if _failure_flag(row[2]):
                row_failures.add((str(row[0]), str(row[1])))
    failures |= row_failures

    observations_by_race: dict[tuple[str, str], dict[RaceKey, list[float]]] = {}
    for row in data.to_dict("records"):
        identity = (str(row["_asset_type"]), str(row["_asset_id"]))
        try:
            race_key = canonical_race_key(row.get("season"), row.get("round"))
        except ValueError:
            continue
        if race_key not in selected_set:
            continue
        if played_col in data.columns:
            played = pd.to_numeric(row.get(played_col), errors="coerce")
            if pd.isna(played) or int(played) != 1:
                continue
        points = pd.to_numeric(row.get(points_col), errors="coerce")
        if pd.isna(points):
            continue
        observations_by_race.setdefault(identity, {}).setdefault(race_key, []).append(float(points))

    rows: list[dict[str, Any]] = []
    selected_count = len(selected_keys)
    for identity in sorted(identities):
        asset_type_value, asset_id = identity
        race_values = observations_by_race.get(identity, {})
        collapsed = {
            key: float(sum(points) / len(points))
            for key, points in race_values.items()
            if points
        }
        valid_keys = tuple(key for key in selected_keys if key in collapsed)
        missing_keys = tuple(key for key in selected_keys if key not in collapsed)
        valid_count = len(valid_keys)
        weight_sum = float(sum(normalized_weights[key] for key in valid_keys))
        if weight_sum > 0:
            weighted_points: float | Any = float(
                sum(collapsed[key] * normalized_weights[key] for key in valid_keys) / weight_sum
            )
        else:
            weighted_points = pd.NA
        has_source_failure = identity in failures
        if selected_count == 0:
            status = "no_races_selected"
        elif has_source_failure:
            status = "source_failure"
        elif valid_count == 0:
            status = "no_valid_observations"
        elif weight_sum == 0:
            status = "no_effective_weight"
        elif valid_count < selected_count:
            status = "incomplete"
        else:
            status = "complete"
        rows.append(
            {
                "asset_id": asset_id,
                "asset_type": asset_type_value,
                "weighted_points": weighted_points,
                "selected_race_count": selected_count,
                "valid_race_count": valid_count,
                "missing_race_count": selected_count - valid_count,
                "coverage_fraction": float(valid_count / selected_count) if selected_count else 0.0,
                "weight_sum": weight_sum,
                "selected_race_keys": selected_keys,
                "valid_race_keys": valid_keys,
                "missing_race_keys": missing_keys,
                "has_source_failure": bool(has_source_failure),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=_WEIGHTED_RESULT_COLUMNS)
