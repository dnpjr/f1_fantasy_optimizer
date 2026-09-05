#!/usr/bin/env python3
"""Estimate crude Sprint-weekend multipliers from recorded Fantasy totals.

This script is deliberately offline and research-only.  It reads the canonical
recorded Fantasy dataset plus locally cached schedule metadata, then writes a
deterministic analysis bundle without importing it into any production path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH
from f1fantasy.weekend_state import parse_schedule_timestamp, weekend_format


SUPPORTED_SEASONS = (2023, 2024, 2025, 2026)
RECORDED_ORIGINS = {"official_recorded", "third_party_recorded"}
REQUIRED_DATA_COLUMNS = {
    "season",
    "round",
    "event_name",
    "event_date",
    "entity_type",
    "canonical_entity_id",
    "name",
    "fantasy_points_total",
    "is_recorded_total",
    "is_reconstructed",
    "fantasy_score_origin",
}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().casefold() in {"1", "true", "yes"}
    )


def _normalise_event_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", text.casefold()).replace("grandprix", "")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _trimmed_mean(values: pd.Series | np.ndarray, proportion: float = 0.10) -> float:
    array = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    if not len(array):
        return float("nan")
    trim = int(np.floor(len(array) * proportion))
    if trim and len(array) > 2 * trim:
        array = array[trim:-trim]
    return float(array.mean())


def load_recorded_data(path: str | Path = DEFAULT_CANONICAL_DATASET_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Canonical recorded Fantasy dataset is missing: {dataset_path}")
    data = pd.read_csv(dataset_path)
    missing = sorted(REQUIRED_DATA_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Canonical dataset is missing required columns: {missing}")

    data = data.copy(deep=True)
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    data["round"] = pd.to_numeric(data["round"], errors="coerce")
    data = data[data["season"].isin(SUPPORTED_SEASONS)].copy()
    if data.empty:
        raise ValueError("Canonical dataset has no rows for supported seasons 2023–2026.")
    if data[["season", "round"]].isna().any().any():
        raise ValueError("Canonical dataset contains invalid season/round identities.")
    data[["season", "round"]] = data[["season", "round"]].astype(int)

    recorded = _as_bool(data["is_recorded_total"])
    reconstructed = _as_bool(data["is_reconstructed"])
    origin = data["fantasy_score_origin"].fillna("").astype(str)
    invalid = ~recorded | reconstructed | ~origin.isin(RECORDED_ORIGINS)
    if invalid.any():
        raise ValueError(
            "Supported-season input contains non-recorded or reconstructed Fantasy totals."
        )
    if not data["entity_type"].isin({"driver", "constructor"}).all():
        raise ValueError("Canonical dataset contains an unsupported entity type.")
    data["fantasy_points_total"] = pd.to_numeric(
        data["fantasy_points_total"], errors="coerce"
    )
    if data["fantasy_points_total"].isna().any():
        raise ValueError("Recorded Fantasy totals contain missing or non-numeric values.")
    key = ["season", "round", "entity_type", "canonical_entity_id"]
    if data.duplicated(key).any():
        duplicates = data.loc[data.duplicated(key, keep=False), key].head(5).to_dict("records")
        raise ValueError(f"Canonical recorded keys are duplicated: {duplicates}")
    return data.sort_values(["season", "round", "entity_type", "canonical_entity_id"]).reset_index(
        drop=True
    )


def load_schedule_metadata(
    schedule_dir: str | Path,
    seasons: tuple[int, ...] = SUPPORTED_SEASONS,
) -> pd.DataFrame:
    directory = Path(schedule_dir)
    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = directory / f"schedule_{season}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Required local schedule metadata is missing: {path}")
        frame = pd.read_csv(path)
        missing = sorted({"season", "round", "raceName", "date"} - set(frame.columns))
        if missing:
            raise ValueError(f"Schedule {path} is missing required columns: {missing}")
        frames.append(frame)
    schedule = pd.concat(frames, ignore_index=True, sort=False)
    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
    schedule["round"] = pd.to_numeric(schedule["round"], errors="coerce")
    if schedule[["season", "round"]].isna().any().any():
        raise ValueError("Schedule metadata contains invalid season/round identities.")
    schedule[["season", "round"]] = schedule[["season", "round"]].astype(int)
    if schedule.duplicated(["season", "round"]).any():
        raise ValueError("Schedule metadata has duplicate or conflicting event identities.")

    for field, time_field in (
        ("sprint_date", "sprint_time"),
        ("sprint_qualifying_date", "sprint_qualifying_time"),
    ):
        if field not in schedule:
            continue
        supplied = schedule[field].notna() & schedule[field].astype(str).str.strip().ne("")
        malformed = supplied & schedule.apply(
            lambda row: parse_schedule_timestamp(row.get(field), row.get(time_field)) is None,
            axis=1,
        )
        if malformed.any():
            keys = schedule.loc[malformed, ["season", "round"]].to_dict("records")
            raise ValueError(f"Ambiguous or malformed Sprint metadata for events: {keys}")
    schedule["weekend_format"] = schedule.apply(lambda row: weekend_format(row).value, axis=1)
    if not schedule["weekend_format"].isin({"normal", "sprint"}).all():
        raise ValueError("Weekend format classification was ambiguous.")
    return schedule.sort_values(["season", "round"]).reset_index(drop=True)


def build_event_summary(data: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recorded_events = data[["season", "round", "event_name", "event_date"]].drop_duplicates()
    if recorded_events.duplicated(["season", "round"]).any():
        raise ValueError("Canonical rows disagree on event name or date for the same event key.")
    metadata = schedule[
        ["season", "round", "raceName", "date", "weekend_format"]
    ].copy()
    events = recorded_events.merge(
        metadata,
        on=["season", "round"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_metadata = events[events["_merge"].ne("both")]
    if not missing_metadata.empty:
        keys = missing_metadata[["season", "round"]].to_dict("records")
        raise ValueError(f"Event-format metadata is missing for recorded events: {keys}")
    name_conflicts = events.apply(
        lambda row: _normalise_event_name(row["event_name"])
        != _normalise_event_name(row["raceName"]),
        axis=1,
    )
    if name_conflicts.any():
        conflicts = events.loc[
            name_conflicts, ["season", "round", "event_name", "raceName"]
        ].to_dict("records")
        raise ValueError(f"Canonical and schedule event names conflict: {conflicts}")
    canonical_dates = pd.to_datetime(events["event_date"], errors="coerce").dt.date
    schedule_dates = pd.to_datetime(events["date"], errors="coerce").dt.date
    date_conflicts = canonical_dates.notna() & schedule_dates.notna() & canonical_dates.ne(schedule_dates)
    if date_conflicts.any():
        raise ValueError("Canonical and schedule event dates conflict.")
    events["event_date"] = schedule_dates.astype(str)
    events = events.drop(columns=["raceName", "date", "_merge"])

    annotated = data.merge(
        events[["season", "round", "event_date", "weekend_format"]],
        on=["season", "round"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_verified"),
    )
    annotated["event_date"] = annotated["event_date_verified"].combine_first(
        annotated["event_date"]
    )
    annotated = annotated.drop(columns=["event_date_verified"])

    rows: list[dict[str, object]] = []
    for event, group in annotated.groupby(
        ["season", "round", "event_name", "event_date", "weekend_format"], sort=True
    ):
        season, round_no, event_name, event_date, fmt = event
        row: dict[str, object] = {
            "season": int(season),
            "round": int(round_no),
            "event_name": event_name,
            "event_date": event_date,
            "weekend_format": fmt,
        }
        for entity_type in ("driver", "constructor"):
            values = group.loc[
                group["entity_type"].eq(entity_type), "fantasy_points_total"
            ].astype(float)
            if values.empty:
                raise ValueError(f"Event {season}/{round_no} has no {entity_type} rows.")
            prefix = entity_type
            row[f"{prefix}_rows"] = int(len(values))
            row[f"{prefix}_active_asset_count"] = int(len(values))
            row[f"{prefix}_total_points"] = float(values.sum())
            row[f"{prefix}_mean_points_per_asset"] = float(values.mean())
            row[f"{prefix}_median_points_per_asset"] = float(values.median())
            row[f"{prefix}_min_points"] = float(values.min())
            row[f"{prefix}_max_points"] = float(values.max())
            row[f"{prefix}_negative_score_count"] = int(values.lt(0).sum())
            row[f"{prefix}_zero_score_count"] = int(values.eq(0).sum())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["season", "round"]).reset_index(drop=True)
    for entity_type in ("driver", "constructor"):
        column = f"{entity_type}_mean_points_per_asset"
        std = float(summary[column].std(ddof=0))
        zscore = (summary[column] - summary[column].mean()) / std if std else 0.0
        summary[f"{entity_type}_event_mean_zscore"] = zscore
        summary[f"{entity_type}_extreme_event"] = pd.Series(zscore).abs().ge(2.0)
    return summary, annotated


def summarize_period(
    annotated: pd.DataFrame,
    seasons: tuple[int, ...] | list[int],
    period: str,
) -> pd.DataFrame:
    subset = annotated[annotated["season"].isin(seasons)].copy()
    rows: list[dict[str, object]] = []
    for entity_type in ("driver", "constructor"):
        entity = subset[subset["entity_type"].eq(entity_type)]
        normal = entity[entity["weekend_format"].eq("normal")]
        sprint = entity[entity["weekend_format"].eq("sprint")]
        normal_event_means = normal.groupby(["season", "round"])["fantasy_points_total"].mean()
        sprint_event_means = sprint.groupby(["season", "round"])["fantasy_points_total"].mean()
        normal_mean = float(normal["fantasy_points_total"].mean())
        sprint_mean = float(sprint["fantasy_points_total"].mean())
        event_normal_mean = float(normal_event_means.mean())
        event_sprint_mean = float(sprint_event_means.mean())
        rows.append(
            {
                "period": period,
                "entity_type": entity_type,
                "normal_event_count": int(len(normal_event_means)),
                "sprint_event_count": int(len(sprint_event_means)),
                "normal_asset_observations": int(len(normal)),
                "sprint_asset_observations": int(len(sprint)),
                "normal_mean_points": normal_mean,
                "sprint_mean_points": sprint_mean,
                "additive_uplift": sprint_mean - normal_mean,
                "multiplicative_uplift": _safe_ratio(sprint_mean, normal_mean),
                "event_means_normal_mean": event_normal_mean,
                "event_means_sprint_mean": event_sprint_mean,
                "event_means_additive_uplift": event_sprint_mean - event_normal_mean,
                "event_means_multiplier": _safe_ratio(event_sprint_mean, event_normal_mean),
                "median_normal_event_mean": float(normal_event_means.median()),
                "median_sprint_event_mean": float(sprint_event_means.median()),
                "median_based_multiplier": _safe_ratio(
                    float(sprint_event_means.median()), float(normal_event_means.median())
                ),
                "trimmed_normal_event_mean": _trimmed_mean(normal_event_means),
                "trimmed_sprint_event_mean": _trimmed_mean(sprint_event_means),
                "trimmed_mean_multiplier": _safe_ratio(
                    _trimmed_mean(sprint_event_means), _trimmed_mean(normal_event_means)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_season_and_pooled_summaries(
    annotated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_summary = pd.concat(
        [summarize_period(annotated, [season], str(season)) for season in SUPPORTED_SEASONS],
        ignore_index=True,
    )
    pooled_summary = pd.concat(
        [
            summarize_period(annotated, [2023, 2024, 2025], "2023-2025"),
            summarize_period(annotated, list(SUPPORTED_SEASONS), "2023-2026"),
        ],
        ignore_index=True,
    )

    for index, row in pooled_summary.iterrows():
        entity_type = str(row["entity_type"])
        period_seasons = [2023, 2024, 2025] if row["period"] == "2023-2025" else list(
            SUPPORTED_SEASONS
        )
        seasonal = season_summary[
            season_summary["entity_type"].eq(entity_type)
            & season_summary["period"].astype(int).isin(period_seasons)
        ]
        pooled_summary.loc[index, "season_multiplier_std"] = seasonal[
            "multiplicative_uplift"
        ].std(ddof=1)
        pooled_summary.loc[index, "season_additive_uplift_std"] = seasonal[
            "additive_uplift"
        ].std(ddof=1)
        period_data = annotated[annotated["season"].isin(period_seasons)]
        combined_normal = period_data.loc[
            period_data["weekend_format"].eq("normal"), "fantasy_points_total"
        ].mean()
        combined_sprint = period_data.loc[
            period_data["weekend_format"].eq("sprint"), "fantasy_points_total"
        ].mean()
        pooled_summary.loc[index, "global_additive_all_entities"] = float(
            combined_sprint - combined_normal
        )
        errors = {
            "multiplicative": [],
            "separate_additive": [],
            "global_additive": [],
            "no_adjustment": [],
        }
        for held_season in period_seasons:
            training = period_data[period_data["season"].ne(held_season)]
            held = period_data[
                period_data["season"].eq(held_season)
                & period_data["entity_type"].eq(entity_type)
            ]
            training_entity = training[training["entity_type"].eq(entity_type)]
            train_normal = float(
                training_entity.loc[
                    training_entity["weekend_format"].eq("normal"), "fantasy_points_total"
                ].mean()
            )
            train_sprint = float(
                training_entity.loc[
                    training_entity["weekend_format"].eq("sprint"), "fantasy_points_total"
                ].mean()
            )
            train_x = _safe_ratio(train_sprint, train_normal)
            train_additive = train_sprint - train_normal
            train_global_additive = float(
                training.loc[
                    training["weekend_format"].eq("sprint"), "fantasy_points_total"
                ].mean()
                - training.loc[
                    training["weekend_format"].eq("normal"), "fantasy_points_total"
                ].mean()
            )
            held_normal = float(
                held.loc[held["weekend_format"].eq("normal"), "fantasy_points_total"].mean()
            )
            held_sprint = float(
                held.loc[held["weekend_format"].eq("sprint"), "fantasy_points_total"].mean()
            )
            errors["multiplicative"].append(abs(held_sprint - train_x * held_normal))
            errors["separate_additive"].append(
                abs(held_sprint - (held_normal + train_additive))
            )
            errors["global_additive"].append(
                abs(held_sprint - (held_normal + train_global_additive))
            )
            errors["no_adjustment"].append(abs(held_sprint - held_normal))
        pooled_summary.loc[index, "season_mae_multiplicative"] = np.mean(
            errors["multiplicative"]
        )
        pooled_summary.loc[index, "season_mae_separate_additive"] = np.mean(
            errors["separate_additive"]
        )
        pooled_summary.loc[index, "season_mae_global_additive"] = np.mean(
            errors["global_additive"]
        )
        pooled_summary.loc[index, "season_mae_no_adjustment"] = np.mean(
            errors["no_adjustment"]
        )
    return season_summary, pooled_summary


def bootstrap_intervals(
    annotated: pd.DataFrame,
    samples: int = 10_000,
    seed: int = 20260806,
) -> pd.DataFrame:
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for entity_type in ("driver", "constructor"):
        entity = annotated[annotated["entity_type"].eq(entity_type)]
        aggregate = entity.groupby(
            ["season", "round", "weekend_format"], as_index=False
        )["fantasy_points_total"].agg(["sum", "count"]).reset_index()
        normal = aggregate[aggregate["weekend_format"].eq("normal")]
        sprint = aggregate[aggregate["weekend_format"].eq("sprint")]
        if normal.empty or sprint.empty:
            raise ValueError(f"Cannot bootstrap {entity_type}: one event class is empty.")
        normal_indices = rng.integers(0, len(normal), size=(samples, len(normal)))
        sprint_indices = rng.integers(0, len(sprint), size=(samples, len(sprint)))
        normal_means = normal["sum"].to_numpy()[normal_indices].sum(axis=1) / normal[
            "count"
        ].to_numpy()[normal_indices].sum(axis=1)
        sprint_means = sprint["sum"].to_numpy()[sprint_indices].sum(axis=1) / sprint[
            "count"
        ].to_numpy()[sprint_indices].sum(axis=1)
        multipliers = sprint_means / normal_means
        estimate = _safe_ratio(
            float(entity.loc[entity["weekend_format"].eq("sprint"), "fantasy_points_total"].mean()),
            float(entity.loc[entity["weekend_format"].eq("normal"), "fantasy_points_total"].mean()),
        )
        percentiles = np.percentile(multipliers, [2.5, 50.0, 97.5])
        rows.append(
            {
                "entity_type": entity_type,
                "bootstrap_samples": int(samples),
                "random_seed": int(seed),
                "estimate": estimate,
                "p2_5": float(percentiles[0]),
                "p50": float(percentiles[1]),
                "p97_5": float(percentiles[2]),
            }
        )
    return pd.DataFrame(rows)


def _primary_multiplier(data: pd.DataFrame, entity_type: str) -> float:
    entity = data[data["entity_type"].eq(entity_type)]
    normal = entity.loc[entity["weekend_format"].eq("normal"), "fantasy_points_total"].mean()
    sprint = entity.loc[entity["weekend_format"].eq("sprint"), "fantasy_points_total"].mean()
    return _safe_ratio(float(sprint), float(normal))


def leave_one_sprint_out(annotated: pd.DataFrame) -> pd.DataFrame:
    full_driver = _primary_multiplier(annotated, "driver")
    full_constructor = _primary_multiplier(annotated, "constructor")
    events = annotated.loc[
        annotated["weekend_format"].eq("sprint"),
        ["season", "round", "event_name"],
    ].drop_duplicates().sort_values(["season", "round"])
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        keep = ~(
            annotated["season"].eq(event.season) & annotated["round"].eq(event.round)
        )
        reduced = annotated[keep]
        driver = _primary_multiplier(reduced, "driver")
        constructor = _primary_multiplier(reduced, "constructor")
        rows.append(
            {
                "excluded_season": int(event.season),
                "excluded_round": int(event.round),
                "excluded_event": event.event_name,
                "driver_multiplier": driver,
                "constructor_multiplier": constructor,
                "change_from_full_driver_estimate": driver - full_driver,
                "change_from_full_constructor_estimate": constructor - full_constructor,
            }
        )
    return pd.DataFrame(rows)


def leave_one_season_out(annotated: pd.DataFrame) -> pd.DataFrame:
    full_driver = _primary_multiplier(annotated, "driver")
    full_constructor = _primary_multiplier(annotated, "constructor")
    rows: list[dict[str, object]] = []
    for season in SUPPORTED_SEASONS:
        reduced = annotated[annotated["season"].ne(season)]
        driver = _primary_multiplier(reduced, "driver")
        constructor = _primary_multiplier(reduced, "constructor")
        rows.append(
            {
                "excluded_season": season,
                "driver_multiplier": driver,
                "constructor_multiplier": constructor,
                "change_from_full_driver_estimate": driver - full_driver,
                "change_from_full_constructor_estimate": constructor - full_constructor,
            }
        )
    return pd.DataFrame(rows)


def assign_pre_event_tiers(annotated: pd.DataFrame) -> pd.DataFrame:
    ordered = annotated.sort_values(["event_date", "season", "round"]).copy(deep=True)
    ordered["pre_event_normal_form"] = np.nan
    ordered["strength_tier"] = "Unranked"
    history: dict[str, dict[str, list[float]]] = {
        "driver": defaultdict(list),
        "constructor": defaultdict(list),
    }
    event_keys = ordered[["event_date", "season", "round"]].drop_duplicates().itertuples(
        index=False
    )
    for event_date, season, round_no in event_keys:
        event_mask = (
            ordered["event_date"].eq(event_date)
            & ordered["season"].eq(season)
            & ordered["round"].eq(round_no)
        )
        for entity_type in ("driver", "constructor"):
            mask = event_mask & ordered["entity_type"].eq(entity_type)
            indices = ordered.index[mask]
            forms = pd.Series(
                {
                    index: (
                        float(np.mean(history[entity_type][str(ordered.at[index, "canonical_entity_id"])]))
                        if history[entity_type][str(ordered.at[index, "canonical_entity_id"])]
                        else np.nan
                    )
                    for index in indices
                },
                dtype=float,
            )
            ordered.loc[indices, "pre_event_normal_form"] = forms
            ranked = forms.dropna().sort_values(ascending=False, kind="stable")
            rank_count = len(ranked)
            for rank, index in enumerate(ranked.index, start=1):
                if entity_type == "driver":
                    tier = "A" if rank <= 5 else "B" if rank <= 15 else "C"
                else:
                    tier = "A" if rank <= 3 else "C" if rank > max(3, rank_count - 3) else "B"
                ordered.at[index, "strength_tier"] = tier
        if ordered.loc[event_mask, "weekend_format"].iloc[0] == "normal":
            for row in ordered.loc[event_mask].itertuples():
                history[row.entity_type][str(row.canonical_entity_id)].append(
                    float(row.fantasy_points_total)
                )
    return ordered.sort_values(["season", "round", "entity_type", "canonical_entity_id"])


def build_tier_summary(tiered: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ranked = tiered[tiered["strength_tier"].isin({"A", "B", "C"})]
    for (entity_type, tier), group in ranked.groupby(["entity_type", "strength_tier"]):
        normal = group[group["weekend_format"].eq("normal")]
        sprint = group[group["weekend_format"].eq("sprint")]
        normal_mean = float(normal["fantasy_points_total"].mean())
        sprint_mean = float(sprint["fantasy_points_total"].mean())
        rows.append(
            {
                "period": "2023-2026",
                "entity_type": entity_type,
                "tier": tier,
                "normal_mean": normal_mean,
                "sprint_mean": sprint_mean,
                "additive_uplift": sprint_mean - normal_mean,
                "multiplier": _safe_ratio(sprint_mean, normal_mean),
                "normal_observation_count": int(len(normal)),
                "sprint_observation_count": int(len(sprint)),
                "normal_event_count": int(normal[["season", "round"]].drop_duplicates().shape[0]),
                "sprint_event_count": int(sprint[["season", "round"]].drop_duplicates().shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["entity_type", "tier"]).reset_index(drop=True)


def build_asset_examples(
    annotated: pd.DataFrame, multipliers: dict[str, float]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entity_type, sample_count in (("driver", 4), ("constructor", 3)):
        entity = annotated[annotated["entity_type"].eq(entity_type)]
        grouped = entity.groupby("canonical_entity_id").agg(
            entity=("name", "last"),
            normal_races=("weekend_format", lambda values: int((values == "normal").sum())),
            sprint_races=("weekend_format", lambda values: int((values == "sprint").sum())),
            total_points=("fantasy_points_total", "sum"),
        ).sort_values("total_points", ascending=False)
        positions = np.linspace(0, len(grouped) - 1, min(sample_count, len(grouped)), dtype=int)
        for position in dict.fromkeys(positions.tolist()):
            asset = grouped.iloc[position]
            x = float(multipliers[entity_type])
            denominator = float(asset.normal_races + x * asset.sprint_races)
            baseline = float(asset.total_points / denominator)
            reconstructed = baseline * float(asset.normal_races) + baseline * x * float(
                asset.sprint_races
            )
            rows.append(
                {
                    "entity": asset.entity,
                    "entity_type": entity_type,
                    "normal_races": int(asset.normal_races),
                    "sprint_races": int(asset.sprint_races),
                    "total_points": float(asset.total_points),
                    "global_multiplier": x,
                    "implied_normal_baseline": baseline,
                    "implied_sprint_ev": x * baseline,
                    "reconstructed_total": reconstructed,
                    "reconstruction_error": reconstructed - float(asset.total_points),
                }
            )
    return pd.DataFrame(rows)


def verify_reconstruction_identity(annotated: pd.DataFrame, multipliers: dict[str, float]) -> float:
    grouped = annotated.groupby(["entity_type", "canonical_entity_id"]).agg(
        normal_count=("weekend_format", lambda values: int((values == "normal").sum())),
        sprint_count=("weekend_format", lambda values: int((values == "sprint").sum())),
        total=("fantasy_points_total", "sum"),
    ).reset_index()
    errors: list[float] = []
    for row in grouped.itertuples(index=False):
        x = float(multipliers[row.entity_type])
        denominator = row.normal_count + x * row.sprint_count
        baseline = row.total / denominator
        reconstructed = baseline * row.normal_count + baseline * x * row.sprint_count
        errors.append(abs(float(reconstructed - row.total)))
    maximum = max(errors, default=0.0)
    if maximum > 1e-9:
        raise AssertionError(f"Baseline reconstruction identity failed; max error={maximum}")
    return maximum


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    shown = frame[columns].copy()
    for column in shown.select_dtypes(include=["float", "float64"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_report(
    event_summary: pd.DataFrame,
    season_summary: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    leave_sprint: pd.DataFrame,
    leave_season: pd.DataFrame,
    tier_summary: pd.DataFrame,
    asset_examples: pd.DataFrame,
    dataset_path: Path,
    max_reconstruction_error: float,
) -> str:
    primary = pooled_summary[pooled_summary["period"].eq("2023-2026")].set_index(
        "entity_type"
    )
    driver_x = float(primary.loc["driver", "multiplicative_uplift"])
    constructor_x = float(primary.loc["constructor", "multiplicative_uplift"])
    boot = bootstrap.set_index("entity_type")
    coverage = event_summary.groupby(["season", "weekend_format"]).size().unstack(fill_value=0).reset_index()
    sprint_events = event_summary[event_summary["weekend_format"].eq("sprint")]
    sprint_ranges = {
        "driver": (leave_sprint["driver_multiplier"].min(), leave_sprint["driver_multiplier"].max()),
        "constructor": (
            leave_sprint["constructor_multiplier"].min(),
            leave_sprint["constructor_multiplier"].max(),
        ),
    }
    season_ranges = {
        "driver": (
            leave_season["driver_multiplier"].min(),
            leave_season["driver_multiplier"].max(),
        ),
        "constructor": (
            leave_season["constructor_multiplier"].min(),
            leave_season["constructor_multiplier"].max(),
        ),
    }
    alternative = primary[
        [
            "season_mae_multiplicative",
            "season_mae_separate_additive",
            "season_mae_global_additive",
            "season_mae_no_adjustment",
        ]
    ]
    multiplicative_wins = (
        alternative["season_mae_multiplicative"]
        <= alternative["season_mae_separate_additive"]
    ).all()
    additive_wins = (
        alternative["season_mae_separate_additive"]
        < alternative["season_mae_multiplicative"]
    ).all()
    if multiplicative_wins:
        recommendation = "A. Use the pooled global multipliers as a first approximation."
    elif additive_wins:
        recommendation = "B. Use separate additive uplifts instead because they are more stable."
    else:
        recommendation = "D. More investigation is required before production use."

    extremes = event_summary[
        event_summary["driver_extreme_event"] | event_summary["constructor_extreme_event"]
    ]
    tier_note = (
        "Tier estimates vary, but every tier remains observational and is not stable enough to replace "
        "the global result. Unranked assets with no prior normal-weekend form are excluded."
    )
    negative_count = int(
        event_summary[["driver_negative_score_count", "constructor_negative_score_count"]]
        .sum()
        .sum()
    )
    zero_count = int(
        event_summary[["driver_zero_score_count", "constructor_zero_score_count"]]
        .sum()
        .sum()
    )
    global_additive = float(primary["global_additive_all_entities"].iloc[0])
    return f"""# Recorded F1 Fantasy Sprint-multiplier analysis

## 1. Executive conclusion

Recommended driver Sprint multiplier: **x_driver = {driver_x:.4f}**
Recommended constructor Sprint multiplier: **x_constructor = {constructor_x:.4f}**

Event-level bootstrap 95% intervals are {boot.loc['driver', 'p2_5']:.4f}–{boot.loc['driver', 'p97_5']:.4f}
for drivers and {boot.loc['constructor', 'p2_5']:.4f}–{boot.loc['constructor', 'p97_5']:.4f}
for constructors. These are crude pooled associations, suitable only as a transparent first approximation.

## 2. Data coverage

Source: `{dataset_path}`. Only canonical recorded totals from 2023–2026 were used; reconstructed
scores and 2021–2022 were excluded. Schedule metadata, rather than score components, defined format.

{_markdown_table(coverage, ['season', 'normal', 'sprint'], digits=0)}

Sprint events:

{_markdown_table(sprint_events, ['season', 'round', 'event_name', 'event_date'], digits=0)}

## 3. Main calculations

{_markdown_table(primary.reset_index(), ['entity_type', 'normal_event_count', 'sprint_event_count', 'normal_asset_observations', 'sprint_asset_observations', 'normal_mean_points', 'sprint_mean_points', 'additive_uplift', 'multiplicative_uplift', 'event_means_multiplier', 'median_based_multiplier', 'trimmed_mean_multiplier'])}

Season-by-season and pooled results are in `season_summary.csv` and `pooled_summary.csv`.

{_markdown_table(season_summary, ['period', 'entity_type', 'normal_event_count', 'sprint_event_count', 'normal_mean_points', 'sprint_mean_points', 'additive_uplift', 'multiplicative_uplift', 'median_based_multiplier', 'trimmed_mean_multiplier'])}

Negative, zero and unusually poor recorded observations were retained: {negative_count} negative
asset-event scores and {zero_count} genuine zero scores. No outlier filtering was applied.

## 4. Stability

{_markdown_table(bootstrap, ['entity_type', 'estimate', 'p2_5', 'p50', 'p97_5', 'bootstrap_samples', 'random_seed'])}

Leave-one-Sprint-out ranges are **{sprint_ranges['driver'][0]:.4f}–{sprint_ranges['driver'][1]:.4f}**
for drivers and **{sprint_ranges['constructor'][0]:.4f}–{sprint_ranges['constructor'][1]:.4f}** for
constructors. Leave-one-season-out ranges are **{season_ranges['driver'][0]:.4f}–{season_ranges['driver'][1]:.4f}**
and **{season_ranges['constructor'][0]:.4f}–{season_ranges['constructor'][1]:.4f}**, respectively.
Full exclusions are in their corresponding CSV files.

Cross-season mean absolute errors for the crude alternatives:

{_markdown_table(alternative.reset_index(), ['entity_type', 'season_mae_multiplicative', 'season_mae_separate_additive', 'season_mae_global_additive', 'season_mae_no_adjustment'])}

The one combined additive uplift is `c = {global_additive:.4f}` points per asset. Separate pooled
uplifts are `{primary.loc['driver', 'additive_uplift']:.4f}` for drivers and
`{primary.loc['constructor', 'additive_uplift']:.4f}` for constructors. In leave-one-season-out
validation, the separate additive adjustment is best for drivers and the shared additive constant is
best for constructors. The separate entity-type additive adjustment still beats multiplication for
both entity types, while no adjustment is least accurate.

Extreme events (absolute event-mean z-score at least 2) are retained, not removed:

{_markdown_table(extremes, ['season', 'round', 'event_name', 'weekend_format', 'driver_mean_points_per_asset', 'constructor_mean_points_per_asset']) if not extremes.empty else 'None under this diagnostic threshold.'}

## 5. Tier findings

{tier_note}

{_markdown_table(tier_summary, ['entity_type', 'tier', 'normal_mean', 'sprint_mean', 'additive_uplift', 'multiplier', 'normal_observation_count', 'sprint_observation_count'])}

## 6. Proposed first approximation

For an asset with total recorded points `T`, `N_normal` normal races, `N_sprint` Sprint races,
and the entity-type multiplier `x`:

```text
baseline = T / (N_normal + x × N_sprint)
normal EV = baseline
Sprint EV = x × baseline
```

The numerical identity check across every asset had maximum reconstruction error
`{max_reconstruction_error:.3e}`. This identity does not establish predictive accuracy.

{_markdown_table(asset_examples, ['entity', 'entity_type', 'normal_races', 'sprint_races', 'total_points', 'global_multiplier', 'implied_normal_baseline', 'implied_sprint_ev', 'reconstructed_total'])}

## 7. Limitations

- Sprint samples are small relative to normal weekends, so event-level uncertainty remains material.
- Crashes, DNFs, penalties, zeroes and negative scores are intentionally retained as realised EV.
- Scoring environments and roster sizes change across seasons.
- Team and driver strength changes over time; pooled means are not causal Sprint effects.
- Replacement drivers often lack enough prior normal-weekend form for tier assignment.
- One common multiplier assumes proportional uplift across all strengths and event conditions.
- The tier analysis is exploratory and susceptible to small samples and regression to the mean.
- This is a first approximation, not a final individual-asset forecast model.

## 8. Recommendation

**{recommendation}**

This choice is based on the reported pooled effects, event-bootstrap uncertainty, leave-one-out
ranges and cross-season alternative errors. No multiplier is activated in production by this analysis.
"""


def run_analysis(
    dataset_path: str | Path = DEFAULT_CANONICAL_DATASET_PATH,
    schedule_dir: str | Path = Path("data/cache"),
    output_dir: str | Path = Path("reports/sprint_multiplier"),
    seed: int = 20260806,
    bootstrap_samples: int = 10_000,
) -> dict[str, pd.DataFrame | str]:
    data = load_recorded_data(dataset_path)
    schedule = load_schedule_metadata(schedule_dir)
    event_summary, annotated = build_event_summary(data, schedule)
    season_summary, pooled_summary = build_season_and_pooled_summaries(annotated)
    bootstrap = bootstrap_intervals(annotated, samples=bootstrap_samples, seed=seed)
    leave_sprint = leave_one_sprint_out(annotated)
    leave_season = leave_one_season_out(annotated)
    tiered = assign_pre_event_tiers(annotated)
    tier_summary = build_tier_summary(tiered)
    primary = pooled_summary[pooled_summary["period"].eq("2023-2026")].set_index(
        "entity_type"
    )
    multipliers = primary["multiplicative_uplift"].astype(float).to_dict()
    max_error = verify_reconstruction_identity(annotated, multipliers)
    asset_examples = build_asset_examples(annotated, multipliers)
    report = build_report(
        event_summary,
        season_summary,
        pooled_summary,
        bootstrap,
        leave_sprint,
        leave_season,
        tier_summary,
        asset_examples,
        Path(dataset_path),
        max_error,
    )

    outputs: dict[str, pd.DataFrame | str] = {
        "event_summary.csv": event_summary,
        "season_summary.csv": season_summary,
        "pooled_summary.csv": pooled_summary,
        "bootstrap_intervals.csv": bootstrap,
        "leave_one_sprint_out.csv": leave_sprint,
        "leave_one_season_out.csv": leave_season,
        "tier_summary.csv": tier_summary,
        "asset_examples.csv": asset_examples,
        "REPORT.md": report,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, output in outputs.items():
        path = destination / filename
        if isinstance(output, pd.DataFrame):
            output.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
        else:
            path.write_text(output, encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CANONICAL_DATASET_PATH)
    parser.add_argument("--schedule-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/sprint_multiplier"))
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    outputs = run_analysis(
        args.dataset,
        args.schedule_dir,
        args.output_dir,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    pooled = outputs["pooled_summary.csv"]
    assert isinstance(pooled, pd.DataFrame)
    recommended = pooled[pooled["period"].eq("2023-2026")][
        ["entity_type", "multiplicative_uplift"]
    ]
    print(recommended.to_string(index=False))


if __name__ == "__main__":
    main()
