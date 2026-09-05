"""Frozen 2026 Sprint EV calibration and calculations.

This module is deliberately isolated from live fetching and research fitting.
It reads reviewed local artefacts and calculates either the retained shadow
view or the approved additive production adjustment without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Iterable, Mapping

import pandas as pd

from f1fantasy.race_selection import RaceKey, canonical_race_key, recency_weights
from f1fantasy.weekend_state import EventKey, WeekendFormat, weekend_format


SHADOW_HISTORY_SEASON = 2026
DEFAULT_SPRINT_SHADOW_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/generated/sprint_ev_shadow/sprint_ev_shadow_2026_v1.json"
)
DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/generated/sprint_ev_calibration/sprint_ev_2026_v1.json"
)


@dataclass(frozen=True)
class SprintShadowCalibration:
    model_version: str
    source_research_model: str
    source_data_version: str
    generated_at: str
    research_only: bool
    calibration_form_mean: float
    calibration_form_sd: float
    driver_group_intercept: float
    driver_group_slope: float
    driver_within_variance: float
    driver_tau_squared: float
    constructor_intercept: float
    constructor_slope: float
    constructor_form_weight: float
    constructor_price_weight: float
    future_event_effect: float
    driver_personal_history: tuple[dict[str, Any], ...]
    calibration_season: int = SHADOW_HISTORY_SEASON
    calibration_status: str = "research_only"


@dataclass(frozen=True)
class SprintShadowResult:
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SprintProductionResult:
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    diagnostics: dict[str, Any]


_SHADOW_COLUMNS = [
    "id",
    "shadow_canonical_entity_id",
    "shadow_normal_ev",
    "shadow_sprint_bonus",
    "shadow_sprint_ev",
    "shadow_weekend_format",
    "shadow_model_version",
    "shadow_history_season",
    "shadow_personal_weight",
    "shadow_group_bonus",
    "shadow_personal_mean_bonus",
    "shadow_personal_observation_count",
    "shadow_form_percentile",
    "shadow_price_percentile",
    "shadow_strength",
    "shadow_selected_race_count",
    "shadow_valid_race_count",
    "shadow_missing_race_count",
    "shadow_coverage_fraction",
    "shadow_status",
]


def _finite_float(value: Any, field: str) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not math.isfinite(float(numeric)):
        raise ValueError(f"Frozen Sprint calibration field {field!r} must be finite.")
    return float(numeric)


def load_sprint_shadow_calibration(
    path: str | Path = DEFAULT_SPRINT_SHADOW_CALIBRATION_PATH,
) -> SprintShadowCalibration:
    """Load and validate the reviewed local calibration without network access."""
    artifact_path = Path(path)
    with artifact_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    required_text = ("model_version", "source_research_model", "source_data_version", "generated_at")
    missing = [field for field in required_text if not str(raw.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Frozen Sprint calibration is missing: {', '.join(missing)}")
    if raw.get("research_only") is not True:
        raise ValueError("Frozen Sprint calibration must remain marked research_only.")
    numeric_fields = (
        "calibration_form_mean",
        "calibration_form_sd",
        "driver_group_intercept",
        "driver_group_slope",
        "driver_within_variance",
        "driver_tau_squared",
        "constructor_intercept",
        "constructor_slope",
        "constructor_form_weight",
        "constructor_price_weight",
        "future_event_effect",
    )
    numeric = {field: _finite_float(raw.get(field), field) for field in numeric_fields}
    if numeric["calibration_form_sd"] <= 0:
        raise ValueError("Frozen Sprint calibration form SD must be positive.")
    if numeric["driver_within_variance"] < 0 or numeric["driver_tau_squared"] < 0:
        raise ValueError("Frozen Sprint calibration variances must be non-negative.")
    personal: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw.get("driver_personal_history", []):
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id or entity_id in seen:
            raise ValueError("Frozen Sprint driver identities must be non-empty and unique.")
        count = pd.to_numeric(row.get("observation_count"), errors="coerce")
        if pd.isna(count) or int(count) != float(count) or int(count) < 0:
            raise ValueError(f"Invalid Sprint observation count for {entity_id!r}.")
        mean = row.get("personal_mean_bonus")
        personal.append(
            {
                "entity_id": entity_id,
                "name": str(row.get("name", "")).strip(),
                "personal_mean_bonus": (
                    _finite_float(mean, f"driver_personal_history.{entity_id}.personal_mean_bonus")
                    if int(count) > 0
                    else math.nan
                ),
                "observation_count": int(count),
            }
        )
        seen.add(entity_id)
    return SprintShadowCalibration(
        model_version=str(raw["model_version"]),
        source_research_model=str(raw["source_research_model"]),
        source_data_version=str(raw["source_data_version"]),
        generated_at=str(raw["generated_at"]),
        research_only=True,
        driver_personal_history=tuple(personal),
        **numeric,
    )


def parse_sprint_production_calibration(
    raw: Mapping[str, Any],
    *,
    allowed_statuses: Iterable[str] = ("approved_production",),
) -> SprintShadowCalibration:
    """Validate one nested runtime calibration payload without I/O."""
    allowed = {str(value) for value in allowed_statuses}
    required_text = (
        "model_version",
        "generated_at",
        "source_data_version",
        "calibration_status",
    )
    missing = [field for field in required_text if not str(raw.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Production Sprint calibration is missing: {', '.join(missing)}")
    status = str(raw["calibration_status"])
    if status not in allowed:
        raise ValueError(
            f"Production Sprint calibration status {status!r} is not one of {sorted(allowed)}."
        )
    season = pd.to_numeric(raw.get("calibration_season"), errors="coerce")
    if pd.isna(season) or int(season) != SHADOW_HISTORY_SEASON:
        raise ValueError(f"Production Sprint calibration season must be {SHADOW_HISTORY_SEASON}.")
    driver = raw.get("driver")
    constructor = raw.get("constructor")
    if not isinstance(driver, Mapping) or not isinstance(constructor, Mapping):
        raise ValueError("Production Sprint calibration needs driver and constructor sections.")
    numeric = {
        "calibration_form_mean": _finite_float(driver.get("form_mean"), "driver.form_mean"),
        "calibration_form_sd": _finite_float(driver.get("form_sd"), "driver.form_sd"),
        "driver_group_intercept": _finite_float(
            driver.get("group_intercept"), "driver.group_intercept"
        ),
        "driver_group_slope": _finite_float(driver.get("group_slope"), "driver.group_slope"),
        "driver_within_variance": _finite_float(
            driver.get("within_variance"), "driver.within_variance"
        ),
        "driver_tau_squared": _finite_float(driver.get("tau_squared"), "driver.tau_squared"),
        "constructor_intercept": _finite_float(
            constructor.get("intercept"), "constructor.intercept"
        ),
        "constructor_slope": _finite_float(constructor.get("slope"), "constructor.slope"),
        "constructor_form_weight": _finite_float(
            constructor.get("form_weight"), "constructor.form_weight"
        ),
        "constructor_price_weight": _finite_float(
            constructor.get("price_weight"), "constructor.price_weight"
        ),
        "future_event_effect": _finite_float(
            constructor.get("future_event_effect", 0.0), "constructor.future_event_effect"
        ),
    }
    if numeric["calibration_form_sd"] <= 0:
        raise ValueError("Production Sprint calibration form SD must be positive.")
    if numeric["driver_within_variance"] < 0 or numeric["driver_tau_squared"] < 0:
        raise ValueError("Production Sprint calibration variances must be non-negative.")
    if not math.isclose(
        numeric["constructor_form_weight"] + numeric["constructor_price_weight"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Production Sprint constructor strength weights must sum to one.")
    if not math.isclose(numeric["future_event_effect"], 0.0, abs_tol=1e-12):
        raise ValueError("Production Sprint future-event effect must remain zero.")

    personal: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in driver.get("personal_history", []):
        entity_id = str(
            row.get("canonical_entity_id", row.get("entity_id", ""))
        ).strip()
        if not entity_id or entity_id in seen:
            raise ValueError("Production Sprint driver identities must be non-empty and unique.")
        count = pd.to_numeric(row.get("observation_count"), errors="coerce")
        if pd.isna(count) or int(count) != float(count) or int(count) < 0:
            raise ValueError(f"Invalid Sprint observation count for {entity_id!r}.")
        mean = row.get("personal_mean_bonus")
        personal.append(
            {
                "entity_id": entity_id,
                "name": str(row.get("name", "")).strip(),
                "personal_mean_bonus": (
                    _finite_float(mean, f"driver.personal_history.{entity_id}.personal_mean_bonus")
                    if int(count) > 0
                    else math.nan
                ),
                "observation_count": int(count),
            }
        )
        seen.add(entity_id)
    return SprintShadowCalibration(
        model_version=str(raw["model_version"]),
        source_research_model=str(raw.get("source_research_model", "approved-2026-sprint-model")),
        source_data_version=str(raw["source_data_version"]),
        generated_at=str(raw["generated_at"]),
        research_only=False,
        driver_personal_history=tuple(personal),
        calibration_season=int(season),
        calibration_status=status,
        **numeric,
    )


def load_sprint_production_calibration(
    path: str | Path = DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH,
) -> SprintShadowCalibration:
    """Load the approved runtime-only calibration without fitting or networking."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return parse_sprint_production_calibration(raw)


def active_sprint_calibration_version(
    path: str | Path = DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH,
) -> str:
    """Return the validated active version used in prediction identities."""
    return load_sprint_production_calibration(path).model_version


def driver_personal_weight(observation_count: int, calibration: SprintShadowCalibration) -> float:
    """Return the approved empirical-Bayes personal-history weight."""
    count = int(observation_count)
    if count <= 0:
        return 0.0
    noise = calibration.driver_within_variance / count
    denominator = calibration.driver_tau_squared + noise
    return calibration.driver_tau_squared / denominator if denominator > 0 else 0.0


def _identity_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value).strip()


def _normalised_name(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in text).split())


def _coerce_race_keys(values: Iterable[RaceKey | EventKey | tuple[int, int]]) -> tuple[RaceKey, ...]:
    return tuple(
        sorted(
            {
                canonical_race_key(
                    value.season if hasattr(value, "season") else value[0],
                    value.round if hasattr(value, "round") else value[1],
                )
                for value in values
            }
        )
    )


def _schedule_formats(schedule: pd.DataFrame, season: int) -> dict[RaceKey, WeekendFormat]:
    if schedule is None or schedule.empty or not {"season", "round"}.issubset(schedule.columns):
        return {}
    data = schedule.copy(deep=True)
    data["_season"] = pd.to_numeric(data["season"], errors="coerce")
    data["_round"] = pd.to_numeric(data["round"], errors="coerce")
    data = data[(data["_season"] == season) & data["_round"].notna()].copy()
    data.sort_values(["_season", "_round"], kind="stable", inplace=True)
    data.drop_duplicates(["_season", "_round"], keep="first", inplace=True)
    return {
        canonical_race_key(row["_season"], row["_round"]): weekend_format(row)
        for _, row in data.iterrows()
    }


def normal_equivalent_history(
    recorded_scores: pd.DataFrame,
    schedule: pd.DataFrame,
    selected_race_keys: Iterable[RaceKey | EventKey | tuple[int, int]],
    *,
    upcoming_event: EventKey | RaceKey | tuple[int, int] | None = None,
    season: int = SHADOW_HISTORY_SEASON,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return selected 2026 recorded totals with Sprint-only points removed."""
    selected = tuple(key for key in _coerce_race_keys(selected_race_keys) if key.season == season)
    if upcoming_event is not None:
        upcoming = _coerce_race_keys([upcoming_event])[0]
        selected = tuple(key for key in selected if key != upcoming)
    formats = _schedule_formats(schedule, season)
    columns = [
        "season", "round", "entity_type", "canonical_entity_id", "source_entity_id", "name",
        "fantasy_points_total", "sprint_points", "sprint_qualifying_points",
        "shadow_weekend_format", "shadow_sprint_only_points", "normal_equivalent_score",
        "normal_equivalent_status",
    ]
    if recorded_scores is None or recorded_scores.empty or not selected:
        return pd.DataFrame(columns=columns), {
            "selected_race_keys": [(key.season, key.round) for key in selected],
            "schedule_missing_race_keys": [
                (key.season, key.round) for key in selected if key not in formats
            ],
            "missing_sprint_component_observations": 0,
        }
    required = {"season", "round", "entity_type", "canonical_entity_id", "fantasy_points_total"}
    missing_columns = sorted(required - set(recorded_scores.columns))
    if missing_columns:
        raise ValueError(f"Recorded Sprint-shadow history is missing columns: {', '.join(missing_columns)}")
    data = recorded_scores.copy(deep=True)
    data["_season"] = pd.to_numeric(data["season"], errors="coerce")
    data["_round"] = pd.to_numeric(data["round"], errors="coerce")
    key_pairs = {(key.season, key.round) for key in selected}
    data = data[
        data.apply(lambda row: (row["_season"], row["_round"]) in key_pairs, axis=1)
    ].copy()
    if "source_entity_id" not in data:
        data["source_entity_id"] = pd.NA
    if "name" not in data:
        data["name"] = ""
    for component in ("sprint_points", "sprint_qualifying_points"):
        if component not in data:
            data[component] = pd.NA
    duplicate = data.duplicated(["_season", "_round", "entity_type", "canonical_entity_id"], keep=False)
    if duplicate.any():
        raise ValueError("Recorded Sprint-shadow history contains duplicate canonical asset/event rows.")
    data["fantasy_points_total"] = pd.to_numeric(data["fantasy_points_total"], errors="coerce")
    data["sprint_points"] = pd.to_numeric(data["sprint_points"], errors="coerce")
    data["sprint_qualifying_points"] = pd.to_numeric(
        data["sprint_qualifying_points"], errors="coerce"
    )
    data["shadow_weekend_format"] = data.apply(
        lambda row: formats.get(canonical_race_key(row["_season"], row["_round"])),
        axis=1,
    )
    sprint_component = data[["sprint_points", "sprint_qualifying_points"]].sum(axis=1, min_count=1)
    is_sprint = data["shadow_weekend_format"].eq(WeekendFormat.SPRINT)
    schedule_known = data["shadow_weekend_format"].notna()
    data["shadow_sprint_only_points"] = sprint_component.where(is_sprint, 0.0)
    data["normal_equivalent_score"] = data["fantasy_points_total"]
    data.loc[is_sprint, "normal_equivalent_score"] = (
        data.loc[is_sprint, "fantasy_points_total"]
        - data.loc[is_sprint, "shadow_sprint_only_points"]
    )
    data.loc[~schedule_known, "normal_equivalent_score"] = math.nan
    status = pd.Series("valid", index=data.index, dtype=object)
    status.loc[~schedule_known] = "schedule_missing"
    status.loc[schedule_known & data["fantasy_points_total"].isna()] = "recorded_total_missing"
    status.loc[
        schedule_known & is_sprint & sprint_component.isna() & data["fantasy_points_total"].notna()
    ] = "sprint_component_missing"
    data["normal_equivalent_status"] = status
    data["shadow_weekend_format"] = data["shadow_weekend_format"].map(
        lambda value: value.value if isinstance(value, WeekendFormat) else "unknown"
    )
    data["season"] = data["_season"].astype(int)
    data["round"] = data["_round"].astype(int)
    data.sort_values(["season", "round", "entity_type", "canonical_entity_id"], kind="stable", inplace=True)
    result = data[columns].reset_index(drop=True)
    return result, {
        "selected_race_keys": [(key.season, key.round) for key in selected],
        "schedule_missing_race_keys": [
            (key.season, key.round) for key in selected if key not in formats
        ],
        "missing_sprint_component_observations": int(
            result["normal_equivalent_status"].eq("sprint_component_missing").sum()
        ),
    }


def _weighted_forms(
    history: pd.DataFrame,
    selected_keys: tuple[RaceKey, ...],
    decay: float,
) -> pd.DataFrame:
    weights = recency_weights(selected_keys, decay)
    identities = (
        history[["entity_type", "canonical_entity_id", "source_entity_id", "name"]]
        .drop_duplicates(["entity_type", "canonical_entity_id"], keep="last")
        if not history.empty
        else pd.DataFrame(columns=["entity_type", "canonical_entity_id", "source_entity_id", "name"])
    )
    rows: list[dict[str, Any]] = []
    for identity in identities.to_dict("records"):
        subset = history[
            history["entity_type"].eq(identity["entity_type"])
            & history["canonical_entity_id"].astype(str).eq(str(identity["canonical_entity_id"]))
        ].copy()
        values = {
            canonical_race_key(row.season, row.round): row.normal_equivalent_score
            for row in subset.itertuples(index=False)
        }
        valid_keys = tuple(key for key in selected_keys if pd.notna(values.get(key)))
        missing_keys = tuple(key for key in selected_keys if key not in valid_keys)
        denominator = sum(weights[key] for key in valid_keys)
        weighted = (
            sum(float(values[key]) * weights[key] for key in valid_keys) / denominator
            if denominator > 0
            else math.nan
        )
        rows.append(
            {
                **identity,
                "shadow_normal_ev": weighted,
                "shadow_selected_race_count": len(selected_keys),
                "shadow_valid_race_count": len(valid_keys),
                "shadow_missing_race_count": len(missing_keys),
                "shadow_coverage_fraction": len(valid_keys) / len(selected_keys) if selected_keys else 0.0,
                "shadow_valid_race_keys": valid_keys,
                "shadow_missing_race_keys": missing_keys,
            }
        )
    columns = [
        "entity_type",
        "canonical_entity_id",
        "source_entity_id",
        "name",
        "shadow_normal_ev",
        "shadow_selected_race_count",
        "shadow_valid_race_count",
        "shadow_missing_race_count",
        "shadow_coverage_fraction",
        "shadow_valid_race_keys",
        "shadow_missing_race_keys",
    ]
    return pd.DataFrame(rows, columns=columns)


def _empty_asset_result(assets: pd.DataFrame) -> pd.DataFrame:
    if assets is None or assets.empty:
        return pd.DataFrame(columns=_SHADOW_COLUMNS)
    result = pd.DataFrame({"id": assets["id"].copy()})
    for column in _SHADOW_COLUMNS[1:]:
        result[column] = pd.NA
    return result


def _map_current_assets(assets: pd.DataFrame, forms: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    if assets is None or assets.empty:
        return pd.DataFrame(columns=["id", "name", "price", "shadow_canonical_entity_id", "shadow_normal_ev"])
    current = assets.copy(deep=True)
    for column in ("id", "name", "price"):
        if column not in current:
            current[column] = pd.NA
    typed = forms[forms.get("entity_type", pd.Series(dtype=object)).eq(asset_type)].copy()
    source_map = {
        _identity_text(row.source_entity_id): row
        for row in typed.itertuples(index=False)
        if _identity_text(row.source_entity_id)
    }
    name_map = {
        _normalised_name(row.name): row
        for row in typed.itertuples(index=False)
        if _normalised_name(row.name)
    }
    mapped: list[dict[str, Any]] = []
    for row in current.itertuples(index=False):
        row_data = row._asdict()
        match = source_map.get(_identity_text(row_data.get("id")))
        if match is None:
            match = name_map.get(_normalised_name(row_data.get("name")))
        mapped.append(
            {
                "id": row_data.get("id"),
                "name": row_data.get("name"),
                "price": row_data.get("price"),
                "shadow_canonical_entity_id": getattr(match, "canonical_entity_id", pd.NA),
                "shadow_normal_ev": getattr(match, "shadow_normal_ev", math.nan),
                "shadow_selected_race_count": getattr(match, "shadow_selected_race_count", 0),
                "shadow_valid_race_count": getattr(match, "shadow_valid_race_count", 0),
                "shadow_missing_race_count": getattr(match, "shadow_missing_race_count", 0),
                "shadow_coverage_fraction": getattr(match, "shadow_coverage_fraction", 0.0),
            }
        )
    return pd.DataFrame(mapped)


def _apply_driver_formula(
    current: pd.DataFrame,
    calibration: SprintShadowCalibration,
    upcoming_format: str,
) -> pd.DataFrame:
    if current.empty:
        return _empty_asset_result(current)
    personal_by_id = {row["entity_id"]: row for row in calibration.driver_personal_history}
    personal_by_name = {_normalised_name(row["name"]): row for row in calibration.driver_personal_history}
    rows: list[dict[str, Any]] = []
    for row in current.to_dict("records"):
        normal_ev = pd.to_numeric(row.get("shadow_normal_ev"), errors="coerce")
        personal = personal_by_id.get(str(row.get("shadow_canonical_entity_id", "")))
        if personal is None:
            personal = personal_by_name.get(_normalised_name(row.get("name")))
        count = int(personal["observation_count"]) if personal else 0
        personal_mean = float(personal["personal_mean_bonus"]) if personal and count > 0 else math.nan
        weight = driver_personal_weight(count, calibration)
        group_bonus = (
            calibration.driver_group_intercept
            + calibration.driver_group_slope
            * ((float(normal_ev) - calibration.calibration_form_mean) / calibration.calibration_form_sd)
            if pd.notna(normal_ev)
            else math.nan
        )
        if pd.isna(normal_ev) or upcoming_format not in {"normal", "sprint"}:
            bonus = math.nan
            sprint_ev = math.nan
            status = "unavailable"
        elif upcoming_format == "normal":
            bonus = 0.0
            sprint_ev = float(normal_ev)
            status = "available"
        else:
            bonus = (
                weight * personal_mean + (1.0 - weight) * group_bonus
                if count > 0
                else group_bonus
            )
            sprint_ev = float(normal_ev) + float(bonus)
            status = "available"
        rows.append(
            {
                **row,
                "shadow_sprint_bonus": bonus,
                "shadow_sprint_ev": sprint_ev,
                "shadow_personal_weight": weight,
                "shadow_group_bonus": group_bonus,
                "shadow_personal_mean_bonus": personal_mean,
                "shadow_personal_observation_count": count,
                "shadow_form_percentile": math.nan,
                "shadow_price_percentile": math.nan,
                "shadow_strength": math.nan,
                "shadow_status": status,
            }
        )
    return pd.DataFrame(rows)


def _apply_constructor_formula(
    current: pd.DataFrame,
    calibration: SprintShadowCalibration,
    upcoming_format: str,
) -> pd.DataFrame:
    if current.empty:
        return _empty_asset_result(current)
    result = current.copy(deep=True)
    result["shadow_normal_ev"] = pd.to_numeric(result["shadow_normal_ev"], errors="coerce")
    result["price"] = pd.to_numeric(result["price"], errors="coerce")
    result["shadow_form_percentile"] = result["shadow_normal_ev"].rank(method="average", pct=True)
    result["shadow_price_percentile"] = result["price"].rank(method="average", pct=True)
    result["shadow_strength"] = (
        calibration.constructor_form_weight * result["shadow_form_percentile"]
        + calibration.constructor_price_weight * result["shadow_price_percentile"]
    )
    formula_bonus = (
        calibration.constructor_intercept
        + calibration.constructor_slope * result["shadow_strength"]
    )
    form_available = result["shadow_normal_ev"].notna()
    sprint_available = form_available & result["shadow_strength"].notna()
    if upcoming_format == "normal":
        result["shadow_sprint_bonus"] = 0.0
        result.loc[~form_available, "shadow_sprint_bonus"] = math.nan
        available = form_available
    elif upcoming_format == "sprint":
        result["shadow_sprint_bonus"] = formula_bonus.where(sprint_available)
        available = sprint_available
    else:
        result["shadow_sprint_bonus"] = math.nan
        available = pd.Series(False, index=result.index)
    result["shadow_sprint_ev"] = result["shadow_normal_ev"] + result["shadow_sprint_bonus"]
    result["shadow_personal_weight"] = math.nan
    result["shadow_group_bonus"] = formula_bonus
    result["shadow_personal_mean_bonus"] = math.nan
    result["shadow_personal_observation_count"] = math.nan
    result["shadow_status"] = available.map({True: "available", False: "unavailable"})
    if upcoming_format not in {"normal", "sprint"}:
        result["shadow_status"] = "unavailable"
    return result


def _finalise(frame: pd.DataFrame, calibration: SprintShadowCalibration, upcoming_format: str) -> pd.DataFrame:
    result = frame.copy(deep=True)
    result["shadow_weekend_format"] = upcoming_format
    result["shadow_model_version"] = calibration.model_version
    result["shadow_history_season"] = SHADOW_HISTORY_SEASON
    for column in _SHADOW_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    return result[_SHADOW_COLUMNS].copy()


def _calculate_with_calibration(
    recorded_scores: pd.DataFrame,
    schedule: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    selected_race_keys: Iterable[RaceKey | EventKey | tuple[int, int]],
    recency_decay: float,
    upcoming_event: EventKey | RaceKey | tuple[int, int] | None,
    *,
    production_history_mode: str,
    calibration: SprintShadowCalibration,
    label: str,
    production_isolation: str,
) -> SprintShadowResult:
    selected_2026 = tuple(
        key for key in _coerce_race_keys(selected_race_keys) if key.season == SHADOW_HISTORY_SEASON
    )
    upcoming_key = _coerce_race_keys([upcoming_event])[0] if upcoming_event is not None else None
    formats = _schedule_formats(schedule, SHADOW_HISTORY_SEASON)
    if upcoming_key is None or upcoming_key.season != SHADOW_HISTORY_SEASON:
        upcoming_format = "unknown"
    else:
        format_value = formats.get(upcoming_key)
        upcoming_format = format_value.value if format_value is not None else "unknown"
    history, history_diagnostics = normal_equivalent_history(
        recorded_scores,
        schedule,
        selected_2026,
        upcoming_event=upcoming_key,
        season=SHADOW_HISTORY_SEASON,
    )
    eligible_keys = tuple(
        key for key in selected_2026 if upcoming_key is None or key != upcoming_key
    )
    forms = _weighted_forms(history, eligible_keys, recency_decay)
    driver_current = _map_current_assets(drivers, forms, "driver")
    constructor_current = _map_current_assets(constructors, forms, "constructor")
    driver_result = _finalise(
        _apply_driver_formula(driver_current, calibration, upcoming_format),
        calibration,
        upcoming_format,
    )
    constructor_result = _finalise(
        _apply_constructor_formula(constructor_current, calibration, upcoming_format),
        calibration,
        upcoming_format,
    )
    available_asset_count = int(driver_result["shadow_normal_ev"].notna().sum()) + int(
        constructor_result["shadow_normal_ev"].notna().sum()
    )
    diagnostics = {
        "label": label,
        "status": (
            "available"
            if upcoming_format in {"normal", "sprint"} and available_asset_count > 0
            else "unavailable"
        ),
        "model_version": calibration.model_version,
        "source_research_model": calibration.source_research_model,
        "source_data_version": calibration.source_data_version,
        "calibration_generated_at": calibration.generated_at,
        "research_only": calibration.research_only,
        "future_event_effect": calibration.future_event_effect,
        "production_history_mode": str(production_history_mode),
        "sprint_shadow_history": "2026_only",
        "history_season": SHADOW_HISTORY_SEASON,
        "upcoming_event": (
            {"season": upcoming_key.season, "round": upcoming_key.round}
            if upcoming_key is not None
            else None
        ),
        "upcoming_weekend_format": upcoming_format,
        "selected_2026_race_keys": [(key.season, key.round) for key in eligible_keys],
        "selected_2026_race_weights": {
            f"{key.season}:{key.round}": weight
            for key, weight in recency_weights(eligible_keys, recency_decay).items()
        },
        "recency_decay": float(recency_decay),
        "missing_sprint_component_observations": history_diagnostics[
            "missing_sprint_component_observations"
        ],
        "schedule_missing_race_keys": history_diagnostics["schedule_missing_race_keys"],
        "driver_available_count": int(driver_result["shadow_normal_ev"].notna().sum()),
        "constructor_available_count": int(constructor_result["shadow_normal_ev"].notna().sum()),
        "calibration_season": calibration.calibration_season,
        "calibration_status": calibration.calibration_status,
        "production_isolation": production_isolation,
    }
    return SprintShadowResult(driver_result, constructor_result, diagnostics)


def calculate_sprint_shadow(
    recorded_scores: pd.DataFrame,
    schedule: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    selected_race_keys: Iterable[RaceKey | EventKey | tuple[int, int]],
    recency_decay: float,
    upcoming_event: EventKey | RaceKey | tuple[int, int] | None,
    *,
    production_history_mode: str,
    calibration_path: str | Path = DEFAULT_SPRINT_SHADOW_CALIBRATION_PATH,
) -> SprintShadowResult:
    """Calculate the retained parallel research view without side effects."""
    calibration = load_sprint_shadow_calibration(calibration_path)
    return _calculate_with_calibration(
        recorded_scores,
        schedule,
        drivers,
        constructors,
        selected_race_keys,
        recency_decay,
        upcoming_event,
        production_history_mode=production_history_mode,
        calibration=calibration,
        label="Shadow / experimental — retained for audit",
        production_isolation=(
            "Only shadow_* columns are added by this audit calculation; production promotion "
            "is performed separately from the frozen approved calibration."
        ),
    )


_PRODUCTION_ADJUSTMENT_COLUMNS = [
    "id",
    "baseline_expected_points",
    "sprint_bonus",
    "sprint_adjusted_expected_points",
    "sprint_calibration_version",
    "sprint_calibration_season",
    "sprint_weekend_format",
    "sprint_bonus_applied",
    "sprint_bonus_status",
    "sprint_bonus_driver_group_component",
    "sprint_bonus_driver_personal_component",
    "sprint_bonus_driver_weight",
    "sprint_constructor_strength",
]


def _production_adjustment_rows(
    assets: pd.DataFrame,
    calibrated: pd.DataFrame,
    *,
    entity_type: str,
    calibration: SprintShadowCalibration,
    upcoming_format: str,
) -> pd.DataFrame:
    if assets is None or assets.empty:
        return pd.DataFrame(columns=_PRODUCTION_ADJUSTMENT_COLUMNS)
    if "next_race_expected_points" not in assets.columns:
        raise ValueError("Production Sprint adjustment requires next_race_expected_points.")
    base = assets[["id", "next_race_expected_points"]].copy(deep=True).rename(
        columns={"next_race_expected_points": "baseline_expected_points"}
    )
    details = calibrated.copy(deep=True)
    detail_columns = [
        "id",
        "shadow_sprint_bonus",
        "shadow_personal_weight",
        "shadow_group_bonus",
        "shadow_personal_mean_bonus",
        "shadow_personal_observation_count",
        "shadow_strength",
        "shadow_status",
    ]
    details = details[[column for column in detail_columns if column in details.columns]]
    result = base.merge(details, on="id", how="left", validate="one_to_one")
    baseline = pd.to_numeric(result["baseline_expected_points"], errors="coerce")
    calibrated_bonus = pd.to_numeric(result.get("shadow_sprint_bonus"), errors="coerce")
    is_sprint = upcoming_format == WeekendFormat.SPRINT.value
    result["sprint_bonus"] = calibrated_bonus.fillna(0.0) if is_sprint else 0.0
    result["sprint_adjusted_expected_points"] = baseline
    if is_sprint:
        result["sprint_adjusted_expected_points"] = baseline + result["sprint_bonus"]
    result["sprint_calibration_version"] = calibration.model_version
    result["sprint_calibration_season"] = calibration.calibration_season
    result["sprint_weekend_format"] = upcoming_format
    result["sprint_bonus_applied"] = bool(is_sprint)

    available = calibrated_bonus.notna()
    result["sprint_bonus_status"] = "not_sprint"
    if is_sprint:
        result["sprint_bonus_status"] = available.map(
            {True: "applied", False: "missing_2026_form_baseline_only"}
        )
    if entity_type == "driver":
        weight = pd.to_numeric(result.get("shadow_personal_weight"), errors="coerce").fillna(0.0)
        personal_mean = pd.to_numeric(result.get("shadow_personal_mean_bonus"), errors="coerce")
        group_bonus = pd.to_numeric(result.get("shadow_group_bonus"), errors="coerce")
        counts = pd.to_numeric(
            result.get("shadow_personal_observation_count"), errors="coerce"
        ).fillna(0)
        result["sprint_bonus_driver_weight"] = weight
        result["sprint_bonus_driver_personal_component"] = (
            weight * personal_mean.fillna(0.0) if is_sprint else 0.0
        )
        result["sprint_bonus_driver_group_component"] = (
            (1.0 - weight) * group_bonus if is_sprint else 0.0
        )
        result["sprint_constructor_strength"] = math.nan
        if is_sprint:
            group_only = available & counts.eq(0)
            result.loc[group_only, "sprint_bonus_status"] = "group_only_no_personal_history"
    else:
        result["sprint_bonus_driver_weight"] = math.nan
        result["sprint_bonus_driver_personal_component"] = math.nan
        result["sprint_bonus_driver_group_component"] = math.nan
        result["sprint_constructor_strength"] = pd.to_numeric(
            result.get("shadow_strength"), errors="coerce"
        )
    return result[_PRODUCTION_ADJUSTMENT_COLUMNS].copy()


def calculate_sprint_production_adjustment(
    recorded_scores: pd.DataFrame,
    schedule: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    selected_race_keys: Iterable[RaceKey | EventKey | tuple[int, int]],
    recency_decay: float,
    upcoming_event: EventKey | RaceKey | tuple[int, int] | None,
    *,
    production_history_mode: str,
    calibration_path: str | Path = DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH,
) -> SprintProductionResult:
    """Calculate the approved bonus and pair it with the untouched production baseline."""
    calibration = load_sprint_production_calibration(calibration_path)
    calibrated = _calculate_with_calibration(
        recorded_scores,
        schedule,
        drivers,
        constructors,
        selected_race_keys,
        recency_decay,
        upcoming_event,
        production_history_mode=production_history_mode,
        calibration=calibration,
        label="Approved production Sprint adjustment",
        production_isolation=(
            "The 2026 calibration determines only sprint_bonus; baseline_expected_points "
            "is copied from the completed legacy production calculation."
        ),
    )
    upcoming_format = str(calibrated.diagnostics["upcoming_weekend_format"])
    driver_rows = _production_adjustment_rows(
        drivers,
        calibrated.drivers,
        entity_type="driver",
        calibration=calibration,
        upcoming_format=upcoming_format,
    )
    constructor_rows = _production_adjustment_rows(
        constructors,
        calibrated.constructors,
        entity_type="constructor",
        calibration=calibration,
        upcoming_format=upcoming_format,
    )
    diagnostics = {
        **calibrated.diagnostics,
        "label": "Approved production Sprint adjustment",
        "research_only": False,
        "bonus_applied": upcoming_format == WeekendFormat.SPRINT.value,
        "driver_group_only_fallback_count": int(
            driver_rows["sprint_bonus_status"].eq("group_only_no_personal_history").sum()
        ),
        "driver_missing_form_count": int(
            driver_rows["sprint_bonus_status"].eq("missing_2026_form_baseline_only").sum()
        ),
        "constructor_missing_form_count": int(
            constructor_rows["sprint_bonus_status"].eq("missing_2026_form_baseline_only").sum()
        ),
        "production_semantics": (
            "final next-race EV = unchanged production baseline + approved Sprint bonus "
            "when and only when canonical upcoming weekend_format is sprint"
        ),
    }
    return SprintProductionResult(driver_rows, constructor_rows, diagnostics)


def apply_sprint_production_adjustment(
    assets: pd.DataFrame,
    adjustment: pd.DataFrame,
) -> pd.DataFrame:
    """Apply one precomputed adjustment exactly once to a copied asset table."""
    if any(column in assets.columns for column in _PRODUCTION_ADJUSTMENT_COLUMNS[1:]):
        raise ValueError("Sprint production adjustment has already been applied.")
    original = assets.copy(deep=True)
    if original.empty:
        return original
    merged = original.merge(adjustment.copy(deep=True), on="id", how="left", validate="one_to_one")
    before = pd.to_numeric(original["next_race_expected_points"], errors="coerce").reset_index(drop=True)
    captured = pd.to_numeric(merged["baseline_expected_points"], errors="coerce").reset_index(drop=True)
    pd.testing.assert_series_equal(before, captured, check_names=False, check_dtype=False)
    final = pd.to_numeric(merged["sprint_adjusted_expected_points"], errors="coerce")
    merged["next_race_expected_points"] = final
    merged["next_race_exp_score"] = final
    merged["exp_score"] = final
    return merged
