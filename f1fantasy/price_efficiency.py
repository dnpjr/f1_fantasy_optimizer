from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from f1fantasy.race_selection import (
    RaceKey,
    RaceOption,
    RaceSelection,
    canonical_race_key,
    weighted_asset_points,
)


PRICE_EFFICIENCY_COLUMNS = [
    "asset_id",
    "asset_type",
    "abbreviation",
    "full_name",
    "team_name",
    "team_colour",
    "current_price",
    "selected_points_total",
    "average_points_per_race",
    "price_efficiency",
    "selected_race_count",
    "valid_race_count",
    "missing_race_count",
    "coverage_fraction",
    "has_source_failure",
    "status",
    "valid_race_keys",
]


def _first_present(frame: pd.DataFrame, columns: list[str], default: Any = pd.NA) -> pd.Series:
    result = pd.Series(default, index=frame.index, dtype="object")
    for column in reversed(columns):
        if column in frame.columns:
            result = frame[column].combine_first(result)
    return result


def _source_failure_ids(roster: pd.DataFrame, asset_type: str) -> list[tuple[str, str]]:
    if "recent_points_source" not in roster.columns or "id" not in roster.columns:
        return []
    source = roster["recent_points_source"].fillna("").astype(str).str.casefold()
    failed = source.str.contains("failed|failure|skipped|timeout|error", regex=True)
    return [(asset_type, str(asset_id)) for asset_id in roster.loc[failed, "id"]]


def _selected_keys(
    selected_races: RaceSelection | Iterable[RaceKey | RaceOption | tuple[int, int]],
) -> tuple[RaceKey, ...]:
    values = selected_races.included if isinstance(selected_races, RaceSelection) else selected_races
    keys: set[RaceKey] = set()
    for value in values:
        if isinstance(value, RaceOption):
            value = value.key
        if isinstance(value, RaceKey):
            keys.add(canonical_race_key(value.season, value.round))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            keys.add(canonical_race_key(value[0], value[1]))
        else:
            raise ValueError("Race keys must provide season and round.")
    return tuple(sorted(keys))


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


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def build_price_efficiency_table(
    roster: pd.DataFrame,
    observations: pd.DataFrame,
    selected_races: RaceSelection | Iterable[RaceKey | RaceOption | tuple[int, int]],
    weights: Mapping[RaceKey | tuple[int, int], float] | None = None,
    asset_type: str | None = None,
    source_failures: Iterable[Any] | Mapping[Any, Any] | None = None,
) -> pd.DataFrame:
    """Build a current-roster Price Efficiency table.

    The primary metric is the ordinary official-points mean over valid
    selected races divided by current roster price. Historical per-race prices
    are deliberately ignored. When recency weights are supplied, separately
    named weighted metrics are added without changing the primary metric.
    """
    roster_data = roster.copy(deep=True) if roster is not None else pd.DataFrame()
    observation_data = observations.copy(deep=True) if observations is not None else pd.DataFrame()
    if roster_data.empty:
        return pd.DataFrame(columns=PRICE_EFFICIENCY_COLUMNS)
    if "id" not in roster_data.columns:
        raise ValueError("Roster is missing required column: id.")

    selected_keys = _selected_keys(selected_races)
    type_name = str(asset_type or "unknown")
    if asset_type is not None and "asset_type" in observation_data.columns:
        observation_data = observation_data[
            observation_data["asset_type"].astype(str) == type_name
        ].copy()
    inferred_failures = _source_failure_ids(roster_data, type_name)
    if source_failures is None:
        combined_failures: Iterable[Any] | Mapping[Any, Any] | None = inferred_failures
    elif isinstance(source_failures, Mapping):
        combined_failures = [
            *inferred_failures,
            *[key for key, value in source_failures.items() if _failure_flag(value)],
        ]
    else:
        combined_failures = [*inferred_failures, *list(source_failures)]

    equal_weights = {key: 1.0 for key in selected_keys}
    coverage = weighted_asset_points(
        observation_data,
        selected_keys,
        equal_weights,
        asset_type=type_name,
        source_failures=combined_failures,
    ).rename(columns={"weighted_points": "average_points_per_race"})
    coverage_columns = [
        "asset_id",
        "average_points_per_race",
        "selected_race_count",
        "valid_race_count",
        "missing_race_count",
        "coverage_fraction",
        "has_source_failure",
        "status",
        "valid_race_keys",
    ]
    coverage = coverage[[column for column in coverage_columns if column in coverage.columns]].copy()

    roster_data["asset_id"] = roster_data["id"].astype(str)
    roster_data["asset_type"] = type_name
    roster_data["abbreviation"] = _first_present(
        roster_data,
        ["abbreviation", "tla", "TLA", "short_name", "name"],
    )
    roster_data["full_name"] = _first_present(roster_data, ["full_name", "name"])
    if type_name == "constructor":
        roster_data["team_name"] = _first_present(roster_data, ["name", "team"])
    else:
        roster_data["team_name"] = _first_present(roster_data, ["team", "name"])
    roster_data["team_colour"] = _first_present(roster_data, ["team_colour"], default="")
    roster_data["current_price"] = pd.to_numeric(roster_data.get("price"), errors="coerce")

    result = roster_data[
        [
            "asset_id",
            "asset_type",
            "abbreviation",
            "full_name",
            "team_name",
            "team_colour",
            "current_price",
        ]
    ].merge(coverage, on="asset_id", how="left")

    selected_count = len(selected_keys)
    result["selected_race_count"] = pd.to_numeric(
        result.get("selected_race_count"), errors="coerce"
    ).fillna(selected_count).astype(int)
    result["valid_race_count"] = pd.to_numeric(
        result.get("valid_race_count"), errors="coerce"
    ).fillna(0).astype(int)
    result["missing_race_count"] = (
        result["selected_race_count"] - result["valid_race_count"]
    ).clip(lower=0).astype(int)
    result["coverage_fraction"] = (
        result["valid_race_count"] / result["selected_race_count"].replace(0, pd.NA)
    ).fillna(0.0).astype(float)
    result["has_source_failure"] = result.get(
        "has_source_failure", pd.Series(False, index=result.index)
    ).fillna(False).astype(bool)
    result["valid_race_keys"] = result.get(
        "valid_race_keys", pd.Series([()] * len(result), index=result.index)
    ).apply(lambda value: value if isinstance(value, tuple) else ())
    result["average_points_per_race"] = pd.to_numeric(
        result.get("average_points_per_race"), errors="coerce"
    )
    result["selected_points_total"] = (
        result["average_points_per_race"] * result["valid_race_count"]
    )
    valid_price = result["current_price"].notna() & (result["current_price"] > 0)
    result["price_efficiency"] = (
        result["average_points_per_race"] / result["current_price"]
    ).where(valid_price)

    status = pd.Series("complete", index=result.index, dtype="object")
    status = status.where(result["missing_race_count"] == 0, "incomplete")
    status = status.where(result["valid_race_count"] > 0, "unavailable")
    status = status.where(valid_price | result["average_points_per_race"].isna(), "invalid_price")
    status = status.where(~result["has_source_failure"], "source_failure")
    if selected_count == 0:
        status[:] = "no_races_selected"
    result["status"] = status

    if weights is not None:
        weighted = weighted_asset_points(
            observation_data,
            selected_keys,
            weights,
            asset_type=type_name,
            source_failures=combined_failures,
        )[["asset_id", "weighted_points"]].rename(
            columns={"weighted_points": "weighted_average_points_per_race"}
        )
        result = result.merge(weighted, on="asset_id", how="left")
        result["weighted_price_efficiency"] = (
            pd.to_numeric(result["weighted_average_points_per_race"], errors="coerce")
            / result["current_price"]
        ).where(valid_price)

    ordered_columns = list(PRICE_EFFICIENCY_COLUMNS)
    for optional in ["weighted_average_points_per_race", "weighted_price_efficiency"]:
        if optional in result.columns:
            ordered_columns.append(optional)
    return result[ordered_columns].sort_values(
        ["price_efficiency", "full_name"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def summarize_price_efficiency_team(
    efficiency_table: pd.DataFrame,
    selected_driver_ids: Iterable[str],
    selected_constructor_ids: Iterable[str],
    budget: float,
) -> dict:
    """Validate and summarize a manually selected five-driver/two-team squad.

    ``average_points_per_valid_asset_race`` divides all known selected points
    by the count of valid asset-race observations. For complete, like-for-like
    coverage, ``average_team_points_per_selected_race`` divides total team
    points by the shared selected-race count, and ``team_price_efficiency``
    divides that team race average by total cost. The sum of individual asset
    ratios is exposed separately and is never presented as the team ratio.
    """
    table = efficiency_table.copy(deep=True) if efficiency_table is not None else pd.DataFrame()
    driver_ids = [str(asset_id) for asset_id in selected_driver_ids]
    constructor_ids = [str(asset_id) for asset_id in selected_constructor_ids]
    unique_driver_ids = list(dict.fromkeys(driver_ids))
    unique_constructor_ids = list(dict.fromkeys(constructor_ids))
    messages: list[str] = []

    if len(driver_ids) != len(unique_driver_ids):
        messages.append("Driver selection contains duplicates.")
    if len(constructor_ids) != len(unique_constructor_ids):
        messages.append("Constructor selection contains duplicates.")
    if len(unique_driver_ids) != 5:
        messages.append("Select exactly five unique drivers.")
    if len(unique_constructor_ids) != 2:
        messages.append("Select exactly two unique constructors.")

    if table.empty or not {"asset_id", "asset_type"}.issubset(table.columns):
        selected = pd.DataFrame(columns=table.columns)
        missing_driver_ids = unique_driver_ids
        missing_constructor_ids = unique_constructor_ids
    else:
        normalized = table.copy()
        normalized["asset_id"] = normalized["asset_id"].astype(str)
        normalized["asset_type"] = normalized["asset_type"].astype(str)
        driver_rows = normalized[
            (normalized["asset_type"] == "driver")
            & normalized["asset_id"].isin(unique_driver_ids)
        ]
        constructor_rows = normalized[
            (normalized["asset_type"] == "constructor")
            & normalized["asset_id"].isin(unique_constructor_ids)
        ]
        selected = pd.concat([driver_rows, constructor_rows], ignore_index=True)
        missing_driver_ids = sorted(set(unique_driver_ids) - set(driver_rows["asset_id"]))
        missing_constructor_ids = sorted(set(unique_constructor_ids) - set(constructor_rows["asset_id"]))
    if missing_driver_ids:
        messages.append(f"Unknown driver ids: {missing_driver_ids}.")
    if missing_constructor_ids:
        messages.append(f"Unknown constructor ids: {missing_constructor_ids}.")

    current_price = _numeric_column(selected, "current_price")
    total_cost = float(current_price.sum(min_count=1)) if current_price.notna().any() else 0.0
    numeric_budget = float(budget)
    remaining_budget = numeric_budget - total_cost
    if total_cost > numeric_budget:
        messages.append(f"Team is over budget by {total_cost - numeric_budget:.2f}.")

    selected_points = _numeric_column(selected, "selected_points_total")
    total_selected_points = (
        float(selected_points.sum(min_count=1)) if selected_points.notna().any() else float("nan")
    )
    valid_counts = _numeric_column(selected, "valid_race_count").fillna(0)
    selected_counts = _numeric_column(selected, "selected_race_count").fillna(0)
    total_valid_asset_races = int(valid_counts.sum())
    total_selected_asset_races = int(selected_counts.sum())
    component_coverage = (
        float(total_valid_asset_races / total_selected_asset_races)
        if total_selected_asset_races
        else 0.0
    )
    average_points_per_valid_asset_race = (
        float(total_selected_points / total_valid_asset_races)
        if total_valid_asset_races and pd.notna(total_selected_points)
        else float("nan")
    )
    efficiencies = _numeric_column(selected, "price_efficiency")
    sum_individual_asset_efficiencies = (
        float(efficiencies.sum(min_count=1)) if efficiencies.notna().any() else float("nan")
    )
    metric_complete = bool(
        len(selected) == 7
        and selected_counts.nunique() == 1
        and (valid_counts == selected_counts).all()
        and selected_points.notna().all()
        and current_price.notna().all()
        and (current_price > 0).all()
    )
    shared_race_count = int(selected_counts.iloc[0]) if metric_complete and len(selected_counts) else 0
    average_team_points_per_selected_race = (
        float(total_selected_points / shared_race_count)
        if metric_complete and shared_race_count > 0
        else float("nan")
    )
    team_price_efficiency = (
        float(average_team_points_per_selected_race / total_cost)
        if metric_complete and total_cost > 0 and pd.notna(average_team_points_per_selected_race)
        else float("nan")
    )
    if len(selected) and not metric_complete:
        messages.append("Team efficiency is incomplete because one or more assets lack complete comparable data.")

    return {
        "valid": not messages,
        "messages": messages,
        "selected_driver_count": len(unique_driver_ids),
        "selected_constructor_count": len(unique_constructor_ids),
        "total_cost": total_cost,
        "remaining_budget": remaining_budget,
        "total_selected_official_points": total_selected_points,
        "total_valid_asset_races": total_valid_asset_races,
        "total_selected_asset_races": total_selected_asset_races,
        "average_points_per_valid_asset_race": average_points_per_valid_asset_race,
        "average_team_points_per_selected_race": average_team_points_per_selected_race,
        "sum_individual_asset_efficiencies": sum_individual_asset_efficiencies,
        "team_price_efficiency": team_price_efficiency,
        "team_price_efficiency_race_denominator": shared_race_count,
        "component_coverage": component_coverage,
        "assets_with_complete_efficiency": int((selected.get("status", pd.Series(dtype=object)) == "complete").sum()),
        "assets_selected": int(len(selected)),
    }
