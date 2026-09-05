#!/usr/bin/env python3
"""Research-only 2026 Sprint bonus model with partially pooled asset effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH
from scripts.analyse_2026_sprint_bonus import _markdown_table
from scripts.analyse_sprint_multiplier import (
    build_event_summary,
    load_recorded_data,
    load_schedule_metadata,
)
from scripts.calibrate_asset_sprint_adjustments import (
    CALIBRATION_MAX_ROUND,
    DEFAULT_RECENCY_DECAY,
    build_baselines,
    build_normalised_history,
    build_sprint_observations,
    prepare_calibration_data,
)


SEASON = 2026
SCRIPT_VERSION = "2026-sprint-partial-pooling-v1"
RANDOM_SEED = 20260810
PENALTY_GRID = (1.0, 4.0, 16.0)
STRENGTH_BLEND_WEIGHTS = (1.00, 0.75, 0.50, 0.25, 0.00)
CASE_STUDY_NAMES = (
    "Nico Hulkenberg", "Valtteri Bottas", "Lance Stroll", "Liam Lawson",
    "Oliver Bearman", "Kimi Antonelli", "George Russell", "Lando Norris",
    "Mercedes", "Ferrari", "McLaren", "Red Bull Racing", "Williams", "Audi",
    "Aston Martin", "Cadillac",
)


def _finite(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _rank_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    left = pd.Series(actual).rank(method="average")
    right = pd.Series(predicted).rank(method="average")
    value = left.corr(right, method="pearson")
    return float(value) if pd.notna(value) else np.nan


def _metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int]:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    actual_array, predicted_array = actual_array[valid], predicted_array[valid]
    residual = predicted_array - actual_array
    return {
        "observations": int(len(actual_array)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": _rank_correlation(actual_array, predicted_array),
    }


def build_strength_definitions(
    baselines: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate form and price strength independently within each asset class."""
    assets = baselines.copy(deep=True).rename(columns={"current_normal_baseline": "normal_ev"})
    assets["form_percentile"] = assets.groupby("entity_type")["normal_ev"].rank(
        pct=True, method="average"
    )
    assets["price_percentile"] = assets.groupby("entity_type")["current_price"].rank(
        pct=True, method="average"
    )
    assets["z_form"] = assets.groupby("entity_type")["normal_ev"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    definition_columns: dict[str, pd.Series] = {
        "z_form": assets["z_form"],
        "form_percentile": assets["form_percentile"],
    }
    for weight in STRENGTH_BLEND_WEIGHTS:
        name = f"blend_form_{weight:.2f}_price_{1 - weight:.2f}"
        definition_columns[name] = (
            weight * assets["form_percentile"]
            + (1 - weight) * assets["price_percentile"]
        )
    rows = []
    for name, values in definition_columns.items():
        for index, asset in assets.iterrows():
            rows.append(
                {
                    "entity_type": asset["entity_type"],
                    "entity_id": asset["entity_id"],
                    "entity_name": asset["entity_name"],
                    "normal_ev": asset["normal_ev"],
                    "normal_weekend_only_mean": asset["normal_weekend_only_mean"],
                    "median_normalised_score": asset["median_normalised_baseline"],
                    "recency_weighted_normal_ev": asset["recency_weighted_normal_baseline"],
                    "current_price": asset["current_price"],
                    "form_percentile": asset["form_percentile"],
                    "price_percentile": asset["price_percentile"],
                    "strength_definition": name,
                    "strength": float(values.loc[index]),
                }
            )
    diagnostics = []
    for entity_type, group in assets.groupby("entity_type", sort=True):
        correlation = float(group["normal_ev"].corr(group["current_price"]))
        percentile_correlation = float(
            group["form_percentile"].corr(group["price_percentile"])
        )
        diagnostics.append(
            {
                "entity_type": entity_type,
                "form_price_correlation": correlation,
                "form_price_percentile_correlation": percentile_correlation,
                "form_price_vif": 1 / (1 - percentile_correlation**2)
                if abs(percentile_correlation) < 1 else np.inf,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def complete_observation_grid(
    observations: pd.DataFrame,
    events: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """Expose every current asset × Sprint event, including wholly absent source rows."""
    sprint_events = events[events["weekend_format"].eq("sprint")][
        ["season", "round", "event_name", "event_date"]
    ]
    grid_rows = []
    for event in sprint_events.itertuples(index=False):
        for asset in baselines.itertuples(index=False):
            grid_rows.append(
                {
                    "season": int(event.season),
                    "round": int(event.round),
                    "event_name_grid": event.event_name,
                    "event_date_grid": event.event_date,
                    "entity_type": asset.entity_type,
                    "entity_id": asset.entity_id,
                    "entity_name_grid": asset.entity_name,
                    "current_price_grid": asset.current_price,
                }
            )
    grid = pd.DataFrame(grid_rows)
    merged = grid.merge(
        observations,
        on=["season", "round", "entity_type", "entity_id"],
        how="left",
        validate="one_to_one",
    )
    for column in ("event_name", "event_date", "entity_name", "current_price"):
        merged[column] = merged[column].combine_first(merged[f"{column}_grid"])
        merged = merged.drop(columns=[f"{column}_grid"])
    absent = merged["total_fantasy_points"].isna()
    merged.loc[absent, "observation_valid"] = False
    merged.loc[absent, "exclusion_reason"] = "missing_canonical_event_asset_row"
    merged.loc[absent, "data_source"] = "missing"
    merged["observation_valid"] = merged["observation_valid"].fillna(False).astype(bool)
    return merged[observations.columns].sort_values(
        ["entity_type", "entity_id", "round"], kind="stable"
    ).reset_index(drop=True)


def attach_strength(
    observations: pd.DataFrame,
    definitions: pd.DataFrame,
    strength_definition: str,
) -> pd.DataFrame:
    selected = definitions[definitions["strength_definition"].eq(strength_definition)][
        ["entity_type", "entity_id", "strength", "normal_ev", "form_percentile", "price_percentile"]
    ]
    return observations.merge(
        selected,
        on=["entity_type", "entity_id"],
        how="inner",
        validate="many_to_one",
    )


def _centred_indicator(values: pd.Series, levels: list[str]) -> np.ndarray:
    if not levels:
        return np.empty((len(values), 0))
    positions = {level: index for index, level in enumerate(levels)}
    matrix = np.zeros((len(values), len(levels)), dtype=float)
    for row_index, value in enumerate(values.astype(str)):
        if value in positions:
            matrix[row_index, positions[value]] = 1.0
    return matrix - 1.0 / len(levels)


def fit_penalised_model(
    observations: pd.DataFrame,
    *,
    include_strength: bool,
    include_asset: bool,
    include_event: bool,
    constrain_strength: bool = True,
    asset_penalty: float = 0.0,
    event_penalty: float = 0.0,
    asset_levels: list[str] | None = None,
    event_levels: list[str] | None = None,
) -> dict[str, object]:
    """Fit centred ridge effects with an optional non-negative strength slope."""
    data = observations[
        observations["observation_valid"].astype(bool)
    ].dropna(subset=["extra_sprint_points", "strength"]).copy()
    if data.empty:
        raise ValueError("No valid Sprint observations are available for the model.")
    asset_levels = sorted(asset_levels or data["entity_id"].astype(str).unique().tolist())
    event_levels = sorted(event_levels or data["round"].astype(int).astype(str).unique().tolist())
    parts = [np.ones((len(data), 1), dtype=float)]
    names = ["mu"]
    penalties = [0.0]
    if include_strength:
        parts.append(data[["strength"]].to_numpy(float))
        names.append("lambda")
        penalties.append(0.0)
    if include_asset:
        parts.append(_centred_indicator(data["entity_id"], asset_levels))
        names.extend([f"asset::{value}" for value in asset_levels])
        penalties.extend([float(asset_penalty)] * len(asset_levels))
    if include_event:
        event_values = data["round"].astype(int).astype(str)
        parts.append(_centred_indicator(event_values, event_levels))
        names.extend([f"event::{value}" for value in event_levels])
        penalties.extend([float(event_penalty)] * len(event_levels))
    design = np.column_stack(parts)
    y = data["extra_sprint_points"].to_numpy(float)
    penalty_matrix = np.diag(penalties)
    gram = design.T @ design + penalty_matrix
    coefficients = np.linalg.solve(gram, design.T @ y)
    if include_strength and constrain_strength and coefficients[1] < 0:
        return fit_penalised_model(
            observations,
            include_strength=False,
            include_asset=include_asset,
            include_event=include_event,
            constrain_strength=False,
            asset_penalty=asset_penalty,
            event_penalty=event_penalty,
            asset_levels=asset_levels,
            event_levels=event_levels,
        ) | {"lambda_forced_zero": True, "requested_include_strength": True}
    fitted = design @ coefficients
    residual = y - fitted
    effective_parameters = 1 + int(include_strength) + int(include_asset) + int(include_event)
    residual_variance = float(
        np.sum(residual**2) / max(len(y) - effective_parameters, 1)
    )
    covariance = residual_variance * np.linalg.pinv(gram)
    coefficient_map = dict(zip(names, coefficients))
    asset_effects = {
        level: float(coefficient_map.get(f"asset::{level}", 0.0))
        for level in asset_levels
    }
    event_effects = {
        int(level): float(coefficient_map.get(f"event::{level}", 0.0))
        for level in event_levels
    }
    if asset_effects:
        centre = float(np.mean(list(asset_effects.values())))
        asset_effects = {key: value - centre for key, value in asset_effects.items()}
    if event_effects:
        centre = float(np.mean(list(event_effects.values())))
        event_effects = {key: value - centre for key, value in event_effects.items()}
    return {
        "mu": float(coefficient_map["mu"]),
        "lambda": float(coefficient_map.get("lambda", 0.0)),
        "asset_effects": asset_effects,
        "event_effects": event_effects,
        "asset_penalty": float(asset_penalty),
        "event_penalty": float(event_penalty),
        "include_strength": include_strength,
        "include_asset": include_asset,
        "include_event": include_event,
        "lambda_forced_zero": False,
        "asset_levels": asset_levels,
        "event_levels": event_levels,
        "coefficient_names": names,
        "coefficients": coefficients,
        "covariance": covariance,
        "residual_variance": residual_variance,
        "observation_count": len(data),
    }


def predict_model(
    fit: dict[str, object],
    frame: pd.DataFrame,
    *,
    include_event_effect: bool,
) -> np.ndarray:
    prediction = (
        float(fit["mu"])
        + float(fit["lambda"]) * frame["strength"].to_numpy(float)
    )
    if fit["include_asset"]:
        prediction = prediction + frame["entity_id"].astype(str).map(
            fit["asset_effects"]
        ).fillna(0.0).to_numpy(float)
    if fit["include_event"] and include_event_effect:
        prediction = prediction + frame["round"].astype(int).map(
            fit["event_effects"]
        ).fillna(0.0).to_numpy(float)
    return np.asarray(prediction, dtype=float)


def prediction_standard_error(
    fit: dict[str, object],
    *,
    strength: float,
    entity_id: str,
) -> float:
    values = [1.0]
    if fit["include_strength"]:
        values.append(float(strength))
    if fit["include_asset"]:
        levels = list(fit["asset_levels"])
        indicator = np.full(len(levels), -1 / len(levels), dtype=float)
        if entity_id in levels:
            indicator[levels.index(entity_id)] += 1.0
        values.extend(indicator.tolist())
    if fit["include_event"]:
        values.extend([0.0] * len(fit["event_levels"]))
    vector = np.asarray(values, dtype=float)
    variance = float(vector @ np.asarray(fit["covariance"]) @ vector)
    return float(np.sqrt(max(variance, 0.0)))


def empirical_bayes_shrunk_means(
    training: pd.DataFrame,
    group_fit: dict[str, object],
    assets: pd.DataFrame,
) -> pd.DataFrame:
    """Shrink each personal residual mean toward its strength-based group bonus."""
    data = training[training["observation_valid"]].copy()
    data["group_bonus"] = float(group_fit["mu"]) + float(group_fit["lambda"]) * data["strength"]
    data["residual_bonus"] = data["extra_sprint_points"] - data["group_bonus"]
    grouped = data.groupby("entity_id")["residual_bonus"].agg(["mean", "count", "var"])
    within = float(data.groupby("entity_id")["residual_bonus"].var().mean())
    if not np.isfinite(within):
        within = float(data["residual_bonus"].var())
    between = float(grouped["mean"].var()) if len(grouped) > 1 else 0.0
    average_noise = float((grouped["var"].fillna(within) / grouped["count"]).mean())
    tau_squared = max(0.0, between - average_noise)
    rows = []
    for asset in assets.itertuples(index=False):
        if asset.entity_id in grouped.index:
            item = grouped.loc[asset.entity_id]
            noise = within / max(float(item["count"]), 1.0)
            weight = tau_squared / (tau_squared + noise) if tau_squared + noise > 0 else 0.0
            residual_mean = float(item["mean"])
            count = int(item["count"])
        else:
            weight, residual_mean, count = 0.0, 0.0, 0
        group_bonus = float(group_fit["mu"]) + float(group_fit["lambda"]) * float(asset.strength)
        rows.append(
            {
                "entity_type": asset.entity_type,
                "entity_id": asset.entity_id,
                "entity_name": asset.entity_name,
                "observation_count": count,
                "group_bonus": group_bonus,
                "personal_residual_mean": residual_mean,
                "empirical_bayes_weight": weight,
                "shrunk_personal_mean_bonus": group_bonus + weight * residual_mean,
                "tau_asset_squared": tau_squared,
                "within_residual_variance": within,
            }
        )
    return pd.DataFrame(rows)


def _personal_mean_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_fit: dict[str, object],
    *,
    shrink: bool,
) -> np.ndarray:
    means = train[train["observation_valid"]].groupby("entity_id")["extra_sprint_points"].mean()
    if not shrink:
        fallback = float(train.loc[train["observation_valid"], "extra_sprint_points"].mean())
        return test["entity_id"].map(means).fillna(fallback).to_numpy(float)
    assets = test[["entity_type", "entity_id", "entity_name", "strength"]].drop_duplicates()
    shrunk = empirical_bayes_shrunk_means(train, group_fit, assets).set_index("entity_id")
    return test["entity_id"].map(shrunk["shrunk_personal_mean_bonus"]).to_numpy(float)


def leave_one_sprint_out(
    observations: pd.DataFrame,
    *,
    model_name: str,
    asset_penalty: float = 4.0,
    event_penalty: float = 4.0,
) -> pd.DataFrame:
    """Remove each complete Sprint event and predict it with v_next fixed at zero."""
    rows = []
    all_assets = sorted(observations["entity_id"].astype(str).unique())
    rounds = sorted(observations["round"].astype(int).unique())
    for held_round in rounds:
        train = observations[observations["round"].ne(held_round)].copy()
        test = observations[
            observations["round"].eq(held_round) & observations["observation_valid"]
        ].copy()
        training_rounds = sorted(train["round"].astype(int).unique())
        if model_name == "constant":
            fit = fit_penalised_model(
                train, include_strength=False, include_asset=False, include_event=False
            )
            predicted = predict_model(fit, test, include_event_effect=False)
        elif model_name == "strength_only":
            fit = fit_penalised_model(
                train, include_strength=True, include_asset=False, include_event=False
            )
            predicted = predict_model(fit, test, include_event_effect=False)
        elif model_name == "strength_event":
            fit = fit_penalised_model(
                train, include_strength=True, include_asset=False, include_event=True,
                event_penalty=event_penalty, event_levels=[str(value) for value in training_rounds],
            )
            predicted = predict_model(fit, test, include_event_effect=False)
        elif model_name == "partial_asset":
            fit = fit_penalised_model(
                train, include_strength=True, include_asset=True, include_event=False,
                asset_penalty=asset_penalty, asset_levels=all_assets,
            )
            predicted = predict_model(fit, test, include_event_effect=False)
        elif model_name == "full_partial_pooling":
            fit = fit_penalised_model(
                train, include_strength=True, include_asset=True, include_event=True,
                asset_penalty=asset_penalty, event_penalty=event_penalty,
                asset_levels=all_assets, event_levels=[str(value) for value in training_rounds],
            )
            predicted = predict_model(fit, test, include_event_effect=False)
        elif model_name == "personal_mean":
            fit = fit_penalised_model(
                train, include_strength=True, include_asset=False, include_event=False
            )
            predicted = _personal_mean_predictions(train, test, fit, shrink=False)
        elif model_name == "shrunk_personal_mean":
            fit = fit_penalised_model(
                train, include_strength=True, include_asset=False, include_event=False
            )
            predicted = _personal_mean_predictions(train, test, fit, shrink=True)
        else:
            raise ValueError(f"Unsupported validation model: {model_name}")
        for index, (_, item) in enumerate(test.iterrows()):
            rows.append(
                {
                    "entity_type": item["entity_type"],
                    "model": model_name,
                    "excluded_round": int(held_round),
                    "training_rounds": ",".join(map(str, training_rounds)),
                    "entity_id": item["entity_id"],
                    "entity_name": item["entity_name"],
                    "form_tier": item["form_tier"],
                    "actual_bonus": float(item["extra_sprint_points"]),
                    "predicted_bonus": float(predicted[index]),
                    "prediction_event_effect": 0.0,
                }
            )
    result = pd.DataFrame(rows)
    result["error"] = result["predicted_bonus"] - result["actual_bonus"]
    return result


def summarise_validation(rows: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = _metrics(rows["actual_bonus"], rows["predicted_bonus"])
    for tier in ("low", "middle", "high"):
        subset = rows[rows["form_tier"].eq(tier)]
        summary[f"mae_{tier}_form_third"] = float(subset["error"].abs().mean())
    return summary


def compare_strength_definitions(
    observations: pd.DataFrame,
    definitions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows, selected = [], {}
    definition_names = definitions["strength_definition"].drop_duplicates().tolist()
    for entity_type in ("driver", "constructor"):
        source = observations[observations["entity_type"].eq(entity_type)]
        entity_rows = []
        for definition in definition_names:
            attached = attach_strength(source, definitions, definition)
            attached["form_tier"] = pd.cut(
                attached["form_percentile"],
                bins=[0.0, 1 / 3, 2 / 3, 1.0],
                labels=["low", "middle", "high"],
                include_lowest=True,
            ).astype(str)
            validation = leave_one_sprint_out(attached, model_name="strength_only")
            metrics = summarise_validation(validation)
            full_fit = fit_penalised_model(
                attached, include_strength=True, include_asset=False, include_event=False
            )
            row = {
                "entity_type": entity_type,
                "comparison_scope": "strength_definition",
                "model": "strength_only",
                "strength_definition": definition,
                "mu": full_fit["mu"],
                "lambda": full_fit["lambda"],
                "asset_penalty": np.nan,
                "event_penalty": np.nan,
                **metrics,
            }
            rows.append(row)
            entity_rows.append(row)
        ranked = pd.DataFrame(entity_rows).sort_values(["mae", "strength_definition"])
        best_mae = float(ranked.iloc[0]["mae"])
        near = ranked[ranked["mae"].le(best_mae * 1.02 + 1e-12)]
        preference = [
            "z_form", "form_percentile", "blend_form_1.00_price_0.00",
            "blend_form_0.75_price_0.25", "blend_form_0.50_price_0.50",
            "blend_form_0.25_price_0.75", "blend_form_0.00_price_1.00",
        ]
        selected[entity_type] = min(
            near["strength_definition"], key=lambda value: preference.index(value)
        )
    return pd.DataFrame(rows), selected


def build_model_comparison(
    observations: pd.DataFrame,
    definitions: pd.DataFrame,
    selected_strength: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    comparison_rows, validation_frames = [], []
    chosen_penalties: dict[str, dict[str, float]] = {}
    for entity_type in ("driver", "constructor"):
        source = observations[observations["entity_type"].eq(entity_type)]
        attached = attach_strength(source, definitions, selected_strength[entity_type])
        attached["form_tier"] = pd.cut(
            attached["form_percentile"],
            bins=[0.0, 1 / 3, 2 / 3, 1.0], labels=["low", "middle", "high"],
            include_lowest=True,
        ).astype(str)
        base_models = (
            "constant", "strength_only", "strength_event", "personal_mean",
            "shrunk_personal_mean",
        )
        for model in base_models:
            validation = leave_one_sprint_out(attached, model_name=model)
            validation_frames.append(validation)
            comparison_rows.append(
                {
                    "entity_type": entity_type,
                    "comparison_scope": "candidate_model",
                    "model": model,
                    "strength_definition": selected_strength[entity_type],
                    "asset_penalty": np.nan,
                    "event_penalty": 4.0 if model == "strength_event" else np.nan,
                    **summarise_validation(validation),
                }
            )
        penalty_results = []
        for asset_penalty in PENALTY_GRID:
            partial_validation = leave_one_sprint_out(
                attached, model_name="partial_asset", asset_penalty=asset_penalty
            )
            validation_frames.append(partial_validation)
            row = {
                "entity_type": entity_type,
                "comparison_scope": "asset_penalty_tuning",
                "model": "partial_asset",
                "strength_definition": selected_strength[entity_type],
                "asset_penalty": asset_penalty,
                "event_penalty": np.nan,
                **summarise_validation(partial_validation),
            }
            comparison_rows.append(row)
            penalty_results.append(row)
        best_asset = float(pd.DataFrame(penalty_results).sort_values(["mae", "asset_penalty"]).iloc[0]["asset_penalty"])
        full_results = []
        for asset_penalty in PENALTY_GRID:
            for event_penalty in PENALTY_GRID:
                full_validation = leave_one_sprint_out(
                    attached,
                    model_name="full_partial_pooling",
                    asset_penalty=asset_penalty,
                    event_penalty=event_penalty,
                )
                validation_frames.append(full_validation)
                row = {
                    "entity_type": entity_type,
                    "comparison_scope": "full_penalty_tuning",
                    "model": "full_partial_pooling",
                    "strength_definition": selected_strength[entity_type],
                    "asset_penalty": asset_penalty,
                    "event_penalty": event_penalty,
                    **summarise_validation(full_validation),
                }
                comparison_rows.append(row)
                full_results.append(row)
        best_full = pd.DataFrame(full_results).sort_values(
            ["mae", "asset_penalty", "event_penalty"]
        ).iloc[0]
        chosen_penalties[entity_type] = {
            "partial_asset_penalty": best_asset,
            "full_asset_penalty": float(best_full["asset_penalty"]),
            "full_event_penalty": float(best_full["event_penalty"]),
        }
    comparison = pd.DataFrame(comparison_rows)
    comparison["selected_penalties"] = False
    for entity_type, choices in chosen_penalties.items():
        comparison.loc[
            comparison["entity_type"].eq(entity_type)
            & comparison["model"].eq("partial_asset")
            & comparison["asset_penalty"].eq(choices["partial_asset_penalty"]),
            "selected_penalties",
        ] = True
        comparison.loc[
            comparison["entity_type"].eq(entity_type)
            & comparison["model"].eq("full_partial_pooling")
            & comparison["asset_penalty"].eq(choices["full_asset_penalty"])
            & comparison["event_penalty"].eq(choices["full_event_penalty"]),
            "selected_penalties",
        ] = True
    validation = pd.concat(validation_frames, ignore_index=True)
    return comparison, validation, chosen_penalties


def fit_final_models(
    observations: pd.DataFrame,
    definitions: pd.DataFrame,
    selected_strength: dict[str, str],
    penalties: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    fits, coefficient_rows = {}, []
    for entity_type in ("driver", "constructor"):
        source = observations[observations["entity_type"].eq(entity_type)]
        attached = attach_strength(source, definitions, selected_strength[entity_type])
        fit = fit_penalised_model(
            attached,
            include_strength=True,
            include_asset=True,
            include_event=True,
            asset_penalty=penalties[entity_type]["full_asset_penalty"],
            event_penalty=penalties[entity_type]["full_event_penalty"],
        )
        fits[entity_type] = fit
        coefficient_rows.append(
            {
                "entity_type": entity_type,
                "model": "full_partial_pooling",
                "strength_definition": selected_strength[entity_type],
                "mu": fit["mu"],
                "lambda": fit["lambda"],
                "lambda_constrained_nonnegative": True,
                "lambda_forced_zero": fit["lambda_forced_zero"],
                "asset_penalty": fit["asset_penalty"],
                "event_penalty": fit["event_penalty"],
                "residual_variance": fit["residual_variance"],
                "observations": fit["observation_count"],
            }
        )
        unconstrained = fit_penalised_model(
            attached,
            include_strength=True,
            include_asset=True,
            include_event=True,
            constrain_strength=False,
            asset_penalty=penalties[entity_type]["full_asset_penalty"],
            event_penalty=penalties[entity_type]["full_event_penalty"],
        )
        coefficient_rows.append(
            {
                "entity_type": entity_type,
                "model": "full_partial_pooling_unconstrained_diagnostic",
                "strength_definition": selected_strength[entity_type],
                "mu": unconstrained["mu"],
                "lambda": unconstrained["lambda"],
                "lambda_constrained_nonnegative": False,
                "lambda_forced_zero": False,
                "asset_penalty": unconstrained["asset_penalty"],
                "event_penalty": unconstrained["event_penalty"],
                "residual_variance": unconstrained["residual_variance"],
                "observations": unconstrained["observation_count"],
            }
        )
    return fits, pd.DataFrame(coefficient_rows)


def build_asset_outputs(
    observations: pd.DataFrame,
    definitions: pd.DataFrame,
    selected_strength: dict[str, str],
    fits: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows, effect_rows, shrinkage_frames, uncertainty_rows = [], [], [], []
    for entity_type in ("driver", "constructor"):
        asset_definitions = definitions[
            definitions["entity_type"].eq(entity_type)
            & definitions["strength_definition"].eq(selected_strength[entity_type])
        ].copy()
        source = attach_strength(
            observations[observations["entity_type"].eq(entity_type)],
            definitions,
            selected_strength[entity_type],
        )
        fit = fits[entity_type]
        shrunk_means = empirical_bayes_shrunk_means(source, fit, asset_definitions)
        shrinkage_frames.append(shrunk_means)
        observed = source[source["observation_valid"]].groupby("entity_id")["extra_sprint_points"].agg(
            sprint_observation_count="count",
            observed_mean_bonus="mean",
            observed_median_bonus="median",
            observed_bonus_sd="std",
        )
        for asset in asset_definitions.itertuples(index=False):
            stats = observed.loc[asset.entity_id] if asset.entity_id in observed.index else None
            group_bonus = float(fit["mu"]) + float(fit["lambda"]) * float(asset.strength)
            asset_effect = float(fit["asset_effects"].get(asset.entity_id, 0.0))
            personal_bonus = group_bonus + asset_effect
            normal_ev = float(asset.normal_ev)
            sprint_ev = normal_ev + personal_bonus
            standard_error = prediction_standard_error(
                fit, strength=float(asset.strength), entity_id=asset.entity_id
            )
            lower, upper = personal_bonus - 1.96 * standard_error, personal_bonus + 1.96 * standard_error
            observation_count = int(stats["sprint_observation_count"]) if stats is not None else 0
            reliability = (
                "insufficient_observations_partial_pooling_only"
                if observation_count < 3
                else "high_uncertainty"
                if upper - lower > (14.0 if entity_type == "driver" else 28.0)
                else "partial_pooling_research_only"
            )
            multiplier = sprint_ev / normal_ev if abs(normal_ev) >= 1.0 else np.nan
            shrink_row = shrunk_means[shrunk_means["entity_id"].eq(asset.entity_id)].iloc[0]
            prediction_rows.append(
                {
                    "entity": asset.entity_name,
                    "entity_type": entity_type,
                    "entity_id": asset.entity_id,
                    "current_price": asset.current_price,
                    "normal_ev": normal_ev,
                    "form_percentile": asset.form_percentile,
                    "price_percentile": asset.price_percentile,
                    "selected_strength_definition": selected_strength[entity_type],
                    "selected_strength": asset.strength,
                    "sprint_observation_count": observation_count,
                    "observed_mean_bonus": stats["observed_mean_bonus"] if stats is not None else np.nan,
                    "observed_median_bonus": stats["observed_median_bonus"] if stats is not None else np.nan,
                    "observed_bonus_sd": stats["observed_bonus_sd"] if stats is not None else np.nan,
                    "group_bonus": group_bonus,
                    "asset_effect_u": asset_effect,
                    "predicted_next_sprint_bonus": personal_bonus,
                    "candidate_normal_ev": normal_ev,
                    "candidate_sprint_ev": sprint_ev,
                    "effective_sprint_multiplier": multiplier,
                    "uncertainty_lower": lower,
                    "uncertainty_upper": upper,
                    "reliability_flag": reliability,
                    "raw_personal_mean_bonus": stats["observed_mean_bonus"] if stats is not None else np.nan,
                    "shrunk_personal_mean_bonus": shrink_row["shrunk_personal_mean_bonus"],
                    "empirical_bayes_mean_weight": shrink_row["empirical_bayes_weight"],
                }
            )
            effect_rows.append(
                {
                    "entity_type": entity_type,
                    "entity_id": asset.entity_id,
                    "entity": asset.entity_name,
                    "strength": asset.strength,
                    "group_bonus": group_bonus,
                    "asset_effect_u": asset_effect,
                    "personalised_bonus": personal_bonus,
                    "asset_effects_centred": True,
                    "asset_penalty": fit["asset_penalty"],
                }
            )
            uncertainty_rows.append(
                {
                    "entity_type": entity_type,
                    "entity_id": asset.entity_id,
                    "entity": asset.entity_name,
                    "prediction_standard_error": standard_error,
                    "uncertainty_lower": lower,
                    "uncertainty_upper": upper,
                    "interval_type": "approximate_95_percent_expected_bonus",
                }
            )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["entity_type", "candidate_sprint_ev", "entity"], ascending=[True, False, True]
    ).reset_index(drop=True)
    effects = pd.DataFrame(effect_rows).sort_values(
        ["entity_type", "asset_effect_u"], ascending=[True, False]
    ).reset_index(drop=True)
    shrinkage = pd.concat(shrinkage_frames, ignore_index=True)
    fixed_rows = []
    for row in predictions.itertuples(index=False):
        for weight in (0.25, 0.50, 0.75):
            fixed_rows.append(
                {
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "entity": row.entity,
                    "method": f"fixed_weight_{weight:.2f}",
                    "weight": weight,
                    "group_bonus": row.group_bonus,
                    "personal_mean_bonus": row.raw_personal_mean_bonus,
                    "shrunk_mean_bonus": weight * row.raw_personal_mean_bonus + (1 - weight) * row.group_bonus,
                }
            )
        fixed_rows.append(
            {
                "entity_type": row.entity_type,
                "entity_id": row.entity_id,
                "entity": row.entity,
                "method": "empirical_bayes",
                "weight": row.empirical_bayes_mean_weight,
                "group_bonus": row.group_bonus,
                "personal_mean_bonus": row.raw_personal_mean_bonus,
                "shrunk_mean_bonus": row.shrunk_personal_mean_bonus,
            }
        )
    return predictions, effects, pd.DataFrame(fixed_rows), pd.DataFrame(uncertainty_rows)


def build_event_effects(
    events: pd.DataFrame,
    fits: dict[str, dict[str, object]],
) -> pd.DataFrame:
    sprint_events = events[events["weekend_format"].eq("sprint")][
        ["season", "round", "event_name"]
    ].copy()
    sprint_events["driver_event_effect"] = sprint_events["round"].map(
        fits["driver"]["event_effects"]
    )
    sprint_events["constructor_event_effect"] = sprint_events["round"].map(
        fits["constructor"]["event_effects"]
    )
    sprint_events["driver_effects_centred"] = True
    sprint_events["constructor_effects_centred"] = True
    return sprint_events.sort_values("round").reset_index(drop=True)


def build_monotonicity(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions[[
        "entity_type", "entity_id", "entity", "selected_strength", "group_bonus",
        "asset_effect_u", "predicted_next_sprint_bonus",
    ]].copy()
    output = output.rename(columns={"predicted_next_sprint_bonus": "personalised_bonus"})
    output = output.sort_values(
        ["entity_type", "selected_strength", "entity"], kind="stable"
    ).reset_index(drop=True)
    output["group_bonus_monotonic_non_decreasing"] = output.groupby("entity_type")[
        "group_bonus"
    ].transform(lambda values: bool((values.diff().dropna() >= -1e-10).all()))
    return output


def build_case_studies(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    event_effects: pd.DataFrame,
) -> pd.DataFrame:
    selected = predictions[predictions["entity"].isin(CASE_STUDY_NAMES)]
    rows = []
    event_map = {
        "driver": event_effects.set_index("round")["driver_event_effect"].to_dict(),
        "constructor": event_effects.set_index("round")["constructor_event_effect"].to_dict(),
    }
    for asset in selected.itertuples(index=False):
        history = observations[
            observations["entity_type"].eq(asset.entity_type)
            & observations["entity_id"].eq(asset.entity_id)
        ]
        for observation in history.itertuples(index=False):
            rows.append(
                {
                    "entity_type": asset.entity_type,
                    "entity_id": asset.entity_id,
                    "entity": asset.entity,
                    "round": observation.round,
                    "event_name": observation.event_name,
                    "observed_sprint_bonus": observation.extra_sprint_points,
                    "observation_valid": observation.observation_valid,
                    "group_prediction": asset.group_bonus,
                    "raw_personal_mean": asset.raw_personal_mean_bonus,
                    "shrunk_personal_mean": asset.shrunk_personal_mean_bonus,
                    "common_event_effect": event_map[asset.entity_type].get(observation.round, 0.0),
                    "asset_effect_u": asset.asset_effect_u,
                    "final_candidate_bonus": asset.predicted_next_sprint_bonus,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["entity_type", "entity", "round"]
    ).reset_index(drop=True)


def historical_shape_check(
    canonical_path: str | Path,
    schedule_path: str | Path,
) -> tuple[pd.DataFrame, str]:
    """Qualitatively test normalised 2023–2025 Sprint-uplift shape without refitting 2026."""
    data = load_recorded_data(canonical_path)
    data = data[data["season"].isin([2023, 2024, 2025])].copy()
    schedule_arg = Path(schedule_path)
    schedule = load_schedule_metadata(
        schedule_arg if schedule_arg.is_dir() else schedule_arg.parent,
        seasons=(2023, 2024, 2025),
    )
    _events, annotated = build_event_summary(data, schedule)
    normal_means = annotated[annotated["weekend_format"].eq("normal")].groupby(
        ["season", "entity_type", "canonical_entity_id"]
    )["fantasy_points_total"].mean().rename("asset_normal_mean")
    sprint = annotated[annotated["weekend_format"].eq("sprint")].merge(
        normal_means.reset_index(),
        on=["season", "entity_type", "canonical_entity_id"],
        how="inner",
        validate="many_to_one",
    )
    sprint["sprint_uplift"] = sprint["fantasy_points_total"] - sprint["asset_normal_mean"]
    sprint["z_strength"] = sprint.groupby(["season", "entity_type"])["asset_normal_mean"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    sprint["z_uplift"] = sprint.groupby(["season", "entity_type"])["sprint_uplift"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    rows = []
    for (season, entity_type), group in sprint.groupby(["season", "entity_type"], sort=True):
        x, y = group["z_strength"].to_numpy(float), group["z_uplift"].to_numpy(float)
        slope = float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1])
        rows.append(
            {
                "season": int(season),
                "entity_type": entity_type,
                "observations": len(group),
                "standardised_strength_uplift_slope": slope,
                "pearson": float(group["z_strength"].corr(group["z_uplift"])),
                "spearman": _rank_correlation(x, y),
                "raw_values_used_in_2026_fit": False,
            }
        )
    output = pd.DataFrame(rows)
    positive = int((output["standardised_strength_uplift_slope"] > 0).sum())
    if positive >= 5 and output["standardised_strength_uplift_slope"].mean() > 0:
        conclusion = "supports_positive_strength_relationship"
    elif positive <= 1 and output["standardised_strength_uplift_slope"].mean() < 0:
        conclusion = "contradicts_positive_strength_relationship"
    else:
        conclusion = "inconclusive"
    output["historical_shape_support"] = conclusion
    return output, conclusion


def _recommended_model_rows(
    comparison: pd.DataFrame,
    penalties: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for entity_type in ("driver", "constructor"):
        group = comparison[
            comparison["entity_type"].eq(entity_type)
            & comparison["comparison_scope"].eq("candidate_model")
        ].copy()
        selected_full = comparison[
            comparison["entity_type"].eq(entity_type)
            & comparison["model"].eq("full_partial_pooling")
            & comparison["asset_penalty"].eq(penalties[entity_type]["full_asset_penalty"])
            & comparison["event_penalty"].eq(penalties[entity_type]["full_event_penalty"])
        ].copy()
        selected_partial = comparison[
            comparison["entity_type"].eq(entity_type)
            & comparison["model"].eq("partial_asset")
            & comparison["asset_penalty"].eq(penalties[entity_type]["partial_asset_penalty"])
        ].copy()
        candidates = pd.concat([group, selected_partial, selected_full], ignore_index=True)
        best_mae = float(candidates["mae"].min())
        near = candidates[candidates["mae"].le(best_mae * 1.02 + 1e-12)]
        complexity = {
            "constant": 0,
            "strength_only": 1,
            "personal_mean": 2,
            "shrunk_personal_mean": 3,
            "strength_event": 4,
            "partial_asset": 5,
            "full_partial_pooling": 6,
        }
        selected = near.sort_values(
            "model", key=lambda values: values.map(complexity)
        ).iloc[0]
        rows.append(selected.to_dict())
    return pd.DataFrame(rows)


def _report_case_summary(case_studies: pd.DataFrame) -> str:
    lines = []
    for entity in CASE_STUDY_NAMES:
        rows = case_studies[case_studies["entity"].eq(entity)]
        if rows.empty:
            continue
        first = rows.iloc[0]
        bonuses = ", ".join(
            "missing" if pd.isna(value) else f"{value:.1f}"
            for value in rows["observed_sprint_bonus"]
        )
        lines.append(
            f"- **{entity}:** observations [{bonuses}]; group {first['group_prediction']:.2f}; "
            f"raw mean {first['raw_personal_mean']:.2f}; shrunk mean {first['shrunk_personal_mean']:.2f}; "
            f"u={first['asset_effect_u']:.2f}; final bonus {first['final_candidate_bonus']:.2f}."
        )
    return "\n".join(lines)


def write_report(
    output: Path,
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    coefficients: pd.DataFrame,
    effects: pd.DataFrame,
    event_effects: pd.DataFrame,
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    case_studies: pd.DataFrame,
    selected_strength: dict[str, str],
    recommended: pd.DataFrame,
    historical_support: str,
) -> None:
    selected_comparison = []
    for row in recommended.itertuples(index=False):
        selected_comparison.append(
            f"{row.entity_type}: `{row.model}` MAE {row.mae:.3f}, RMSE {row.rmse:.3f}, "
            f"bias {row.bias:.3f}, Spearman {row.spearman:.3f}"
        )
    lines = [
        "# 2026 Sprint partial-pooling research model",
        "",
        "## 1. Executive conclusion",
        "",
        "The full proposed model was fitted separately for drivers and constructors with centred ridge asset and event effects. It remains research-only.",
        "",
        _markdown_table(coefficients[coefficients["model"].eq("full_partial_pooling")][[
            "entity_type", "strength_definition", "mu", "lambda", "asset_penalty",
            "event_penalty", "residual_variance",
        ]].round(4)),
        "",
        "Leave-one-Sprint-out preferred candidates: " + "; ".join(selected_comparison) + ". The validation table explicitly shows whether full partial pooling improves on constant, strength-only, and shrunk personal mean.",
        "",
        "## 2. Current form definition",
        "",
        "Normal weekends use recorded total Fantasy points. Sprint weekends use `total - sprint_points - sprint_qualifying_points`. The equal-weight mean across valid completed rounds 1–11 is `normal_ev`; round 12 is excluded. Missing Sprint components remain missing.",
        "",
        "## 3. Strength comparison",
        "",
        f"Selected driver strength: `{selected_strength['driver']}`. Selected constructor strength: `{selected_strength['constructor']}`.",
        "",
        _markdown_table(diagnostics.round(4)),
        "",
        _markdown_table(comparison[comparison["comparison_scope"].eq("strength_definition")][[
            "entity_type", "strength_definition", "mae", "rmse", "bias", "spearman"
        ]].round(4)),
        "",
        "Price is considered useful only when a blend materially lowers whole-event validation error; high VIF values indicate that form and price cannot be interpreted independently.",
        "",
        "## 4. Driver model",
        "",
        _markdown_table(effects[effects["entity_type"].eq("driver")][[
            "entity", "strength", "group_bonus", "asset_effect_u", "personalised_bonus"
        ]].round(4)),
        "",
        "## 5. Constructor model",
        "",
        _markdown_table(effects[effects["entity_type"].eq("constructor")][[
            "entity", "strength", "group_bonus", "asset_effect_u", "personalised_bonus"
        ]].round(4)),
        "",
        "## 6. Sprint event effects",
        "",
        _markdown_table(event_effects.round(4)),
        "",
        "The effects are centred to zero. Prediction for the next unknown Sprint always sets the event effect to zero.",
        "",
        "## 7. Validation",
        "",
        _markdown_table(comparison[
            comparison["comparison_scope"].eq("candidate_model")
            | comparison["selected_penalties"].fillna(False)
        ][[
            "entity_type", "model", "mae", "rmse", "bias", "spearman",
            "mae_low_form_third", "mae_middle_form_third", "mae_high_form_third",
        ]].round(4)),
        "",
        "## 8. Case studies",
        "",
        _report_case_summary(case_studies),
        "",
        "## 9. Historical shape check",
        "",
        f"The season-normalised 2023–2025 qualitative result is `{historical_support}`. Older raw values never enter the 2026 fit.",
        "",
        "## 10. Recommended shadow candidate",
        "",
        "```text\nnormal_ev = mean completed 2026 normal-equivalent scores\nstrength = selected within-class strength definition\ngroup_bonus = mu + lambda * strength\npersonalised_bonus = group_bonus + shrunk asset effect u_i\nSprint EV = normal_ev + personalised_bonus\nfuture unknown event effect = 0\n```",
        "",
        "A shadow implementation is justified only if the selected partial-pooling candidate is competitive with the simpler shrunk personal mean under leave-one-Sprint-out validation. Nothing is activated by this analysis.",
        "",
        "## 11. Limitations",
        "",
        "Only four 2026 Sprints exist, asset outcomes are correlated within events, and constructors provide only 11 cross-sectional units. Reliability effects remain difficult to distinguish from randomness. Form and price are collinear. This is current-state descriptive calibration and must be recalibrated after future Sprint events.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    canonical_path: str | Path,
    schedule_path: str | Path,
    current_prices_path: str | Path,
    output_path: str | Path,
    *,
    recency_decay: float = DEFAULT_RECENCY_DECAY,
) -> dict[str, object]:
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    events, annotated, prices, market_metadata = prepare_calibration_data(
        canonical_path, schedule_path, current_prices_path
    )
    observations = build_sprint_observations(annotated, prices).rename(
        columns={"included_in_regression": "observation_valid"}
    )
    history = build_normalised_history(annotated, prices)
    baselines = build_baselines(history, recency_decay)
    observations = complete_observation_grid(observations, events, baselines)
    definitions, strength_diagnostics = build_strength_definitions(baselines)
    strength_comparison, selected_strength = compare_strength_definitions(
        observations, definitions
    )
    model_comparison, validation, penalties = build_model_comparison(
        observations, definitions, selected_strength
    )
    comparison = pd.concat([strength_comparison, model_comparison], ignore_index=True, sort=False)
    fits, coefficients = fit_final_models(
        observations, definitions, selected_strength, penalties
    )
    predictions, effects, shrunk_means, uncertainty = build_asset_outputs(
        observations, definitions, selected_strength, fits
    )
    event_effects = build_event_effects(events, fits)
    monotonicity = build_monotonicity(predictions)
    case_studies = build_case_studies(observations, predictions, event_effects)
    historical, historical_support = historical_shape_check(canonical_path, schedule_path)
    recommended = _recommended_model_rows(model_comparison, penalties)

    selected_validation_frames = []
    for entity_type, choices in penalties.items():
        mask = validation["entity_type"].eq(entity_type) & (
            validation["model"].isin([
                "constant", "strength_only", "strength_event", "personal_mean",
                "shrunk_personal_mean",
            ])
            | validation["model"].eq("partial_asset")
            | validation["model"].eq("full_partial_pooling")
        )
        selected_validation_frames.append(validation[mask])
    selected_validation = pd.concat(selected_validation_frames, ignore_index=True)

    source_versions = sorted(annotated["data_version"].dropna().astype(str).unique())
    if len(source_versions) != 1:
        raise ValueError(f"Expected one 2026 source data version, found {source_versions}")
    research_model = {
        "model_name": "2026_sprint_partial_pooling",
        "research_only": True,
        "generated_at": "2026-07-05T00:00:00Z",
        "generated_at_policy": "deterministic latest completed round-11 event date",
        "script_version": SCRIPT_VERSION,
        "random_seed": RANDOM_SEED,
        "source_data_version": source_versions[0],
        "completed_rounds": sorted(int(value) for value in events["round"].unique()),
        "sprint_rounds": sorted(
            int(value) for value in events.loc[events["weekend_format"].eq("sprint"), "round"].unique()
        ),
        "current_market_feed_round": market_metadata["feed_round"],
        "future_event_effect": 0.0,
        "historical_shape_support": historical_support,
    }
    for entity_type in ("driver", "constructor"):
        fit = fits[entity_type]
        entity_predictions = predictions[predictions["entity_type"].eq(entity_type)]
        entity_uncertainty = uncertainty[uncertainty["entity_type"].eq(entity_type)]
        recommendation = recommended[recommended["entity_type"].eq(entity_type)].iloc[0]
        research_model[f"{entity_type}_model"] = {
            "strength_definition": selected_strength[entity_type],
            "mu": fit["mu"],
            "lambda": fit["lambda"],
            "lambda_constrained_nonnegative": True,
            "asset_effects": dict(zip(entity_predictions["entity_id"], entity_predictions["asset_effect_u"])),
            "event_effects": {str(key): value for key, value in fit["event_effects"].items()},
            "shrinkage_method": "centred penalised least squares ridge random intercept",
            "asset_penalty": fit["asset_penalty"],
            "event_penalty": fit["event_penalty"],
            "future_event_effect": 0.0,
            "recommended_by_leave_one_sprint_out": recommendation["model"],
            "recommended_validation": {
                key: recommendation[key] for key in ("mae", "rmse", "bias", "spearman")
            },
            "uncertainty": {
                row.entity_id: {
                    "lower": row.uncertainty_lower,
                    "upper": row.uncertainty_upper,
                }
                for row in entity_uncertainty.itertuples(index=False)
            },
        }
    (output / "research_model.json").write_text(
        json.dumps(_finite(research_model), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = {
        "observation_dataset.csv": observations,
        "strength_definitions.csv": pd.concat(
            [definitions, strength_diagnostics.assign(strength_definition="diagnostic")],
            ignore_index=True,
            sort=False,
        ),
        "model_comparison.csv": comparison,
        "model_coefficients.csv": coefficients,
        "asset_effects.csv": effects,
        "event_effects.csv": event_effects,
        "asset_predictions.csv": predictions,
        "leave_one_sprint_out.csv": selected_validation,
        "shrunk_mean_comparison.csv": shrunk_means,
        "strength_monotonicity.csv": monotonicity,
        "case_studies.csv": case_studies,
        "uncertainty.csv": uncertainty,
        "historical_shape_check.csv": historical,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False, float_format="%.10g")
    write_report(
        output,
        strength_diagnostics,
        comparison,
        coefficients,
        effects,
        event_effects,
        predictions,
        selected_validation,
        case_studies,
        selected_strength,
        recommended,
        historical_support,
    )
    return {
        "events": events,
        "observations": observations,
        "baselines": baselines,
        "definitions": definitions,
        "selected_strength": selected_strength,
        "comparison": comparison,
        "validation": selected_validation,
        "penalties": penalties,
        "fits": fits,
        "coefficients": coefficients,
        "predictions": predictions,
        "effects": effects,
        "event_effects": event_effects,
        "historical": historical,
        "historical_support": historical_support,
        "recommended": recommended,
        "research_model": research_model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_DATASET_PATH)
    parser.add_argument("--schedule", type=Path, default=PROJECT_ROOT / "data/cache/schedule_2026.csv")
    parser.add_argument(
        "--current-prices", type=Path,
        default=PROJECT_ROOT / "data/cache/verified_fantasy_market.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "reports/2026_sprint_partial_pooling",
    )
    parser.add_argument("--recency-decay", type=float, default=DEFAULT_RECENCY_DECAY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_analysis(
        args.canonical,
        args.schedule,
        args.current_prices,
        args.output,
        recency_decay=args.recency_decay,
    )
    print(f"Saved research-only partial-pooling analysis to {args.output}")
    print(f"Selected strengths: {result['selected_strength']}")
    for entity_type in ("driver", "constructor"):
        fit = result["fits"][entity_type]
        recommendation = result["recommended"]
        model = recommendation[recommendation["entity_type"].eq(entity_type)].iloc[0]
        print(
            f"{entity_type.title()}: mu={fit['mu']:.4f}, lambda={fit['lambda']:.4f}, "
            f"recommended={model['model']}, LO-Sprint MAE={model['mae']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
