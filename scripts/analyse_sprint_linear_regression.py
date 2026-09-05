#!/usr/bin/env python3
"""Offline Sprint EV regression research using recorded canonical totals only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH
from scripts.analyse_sprint_multiplier import (
    SUPPORTED_SEASONS,
    build_event_summary,
    load_recorded_data,
    load_schedule_metadata,
)


PERIODS: dict[str, tuple[int, ...]] = {
    "2023": (2023,),
    "2024": (2024,),
    "2025": (2025,),
    "2026": (2026,),
    "2023-2025": (2023, 2024, 2025),
    "2023-2026": SUPPORTED_SEASONS,
}
DRIVER_LEVELS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
CONSTRUCTOR_LEVELS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
RIDGE_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
WEIGHT_GRID = (0.25, 0.35, 0.50, 0.65, 0.80)


def _safe_float(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else float("nan")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and np.isfinite(denominator) else float("nan")


def _weighted_prior_normal(
    history: pd.DataFrame,
    decay: float,
) -> float:
    values = history.sort_values(["event_date", "season", "round"], ascending=False)[
        "fantasy_points_total"
    ].to_numpy(float)
    if not len(values):
        return float("nan")
    weights = np.power(float(decay), np.arange(len(values), dtype=float))
    return float(np.average(values, weights=weights))


def build_observation_dataset(
    data: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    recency_decay: float = 0.80,
    shrinkage_prior_races: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe, one-row-per-asset Sprint observations."""
    if not 0 < recency_decay <= 1:
        raise ValueError("Recency decay must be within (0, 1].")
    event_summary, annotated = build_event_summary(data, schedule)
    annotated = annotated.copy(deep=True)
    annotated["event_date"] = pd.to_datetime(annotated["event_date"], errors="raise")
    annotated = annotated.sort_values(
        ["event_date", "season", "round", "entity_type", "canonical_entity_id"]
    ).reset_index(drop=True)

    full_normal = (
        annotated[annotated["weekend_format"].eq("normal")]
        .groupby(["season", "entity_type", "canonical_entity_id"])["fantasy_points_total"]
        .mean()
        .to_dict()
    )
    season_normal_scale = (
        annotated[annotated["weekend_format"].eq("normal")]
        .groupby(["season", "entity_type"])["fantasy_points_total"]
        .mean()
        .to_dict()
    )
    prior_sprint_pairs: dict[str, list[tuple[float, float]]] = {
        "driver": [],
        "constructor": [],
    }
    rows: list[dict[str, object]] = []
    sprint_events = event_summary[event_summary["weekend_format"].eq("sprint")].sort_values(
        ["event_date", "season", "round"]
    )
    for event in sprint_events.itertuples(index=False):
        event_date = pd.Timestamp(event.event_date)
        event_rows = annotated[
            annotated["season"].eq(event.season)
            & annotated["round"].eq(event.round)
            & annotated["weekend_format"].eq("sprint")
        ]
        event_outputs: list[dict[str, object]] = []
        for asset in event_rows.itertuples(index=False):
            asset_id = str(asset.canonical_entity_id)
            entity_type = str(asset.entity_type)
            prior = annotated[
                annotated["event_date"].lt(event_date)
                & annotated["entity_type"].eq(entity_type)
                & annotated["canonical_entity_id"].astype(str).eq(asset_id)
            ].copy()
            same_season_normal = prior[
                prior["season"].eq(event.season) & prior["weekend_format"].eq("normal")
            ]
            strict = float(same_season_normal["fantasy_points_total"].mean())
            recency = _weighted_prior_normal(same_season_normal, recency_decay)

            pairs = prior_sprint_pairs[entity_type]
            prior_sprint_factor = (
                _safe_divide(sum(y for _x, y in pairs), sum(x for x, _y in pairs))
                if pairs
                else 1.0
            )
            adjusted_values = prior["fantasy_points_total"].astype(float).copy()
            adjusted_values.loc[prior["weekend_format"].eq("sprint")] /= prior_sprint_factor
            all_prior_adjusted = float(adjusted_values.mean()) if len(adjusted_values) else float("nan")

            previous_season = full_normal.get((event.season - 1, entity_type, asset_id), np.nan)
            earlier_group = annotated[
                annotated["event_date"].lt(event_date)
                & annotated["entity_type"].eq(entity_type)
                & annotated["weekend_format"].eq("normal")
            ]["fantasy_points_total"].mean()
            prior_mean = float(previous_season) if pd.notna(previous_season) else float(earlier_group)
            same_count = int(len(same_season_normal))
            same_sum = float(same_season_normal["fantasy_points_total"].sum())
            if pd.notna(prior_mean):
                shrunk = (same_sum + shrinkage_prior_races * prior_mean) / (
                    same_count + shrinkage_prior_races
                )
                shrunk_source = (
                    "same_season_plus_previous_season_prior"
                    if pd.notna(previous_season)
                    else "same_season_plus_prior_group_prior"
                )
            else:
                shrunk = strict
                shrunk_source = "same_season_only_no_prior_available"
            descriptive = full_normal.get((event.season, entity_type, asset_id), np.nan)
            scale = season_normal_scale[(event.season, entity_type)]
            event_outputs.append(
                {
                    "season": int(event.season),
                    "round": int(event.round),
                    "event_name": event.event_name,
                    "event_date": str(event.event_date),
                    "event_cluster": f"{event.season}-{event.round}",
                    "entity_type": entity_type,
                    "canonical_entity_id": asset_id,
                    "entity": asset.name,
                    "price": _safe_float(asset.price),
                    "sprint_score": float(asset.fantasy_points_total),
                    "x_strict_prior_normal": strict,
                    "x_recency_prior_normal": recency,
                    "x_all_prior_adjusted": all_prior_adjusted,
                    "x_shrunk_prior": float(shrunk),
                    "x_descriptive_full_season": float(descriptive),
                    "strict_prior_normal_count": same_count,
                    "strict_excluded": bool(pd.isna(strict)),
                    "shrunk_prior_imputed": bool(same_count < shrinkage_prior_races),
                    "baseline_method_strict": "same_season_prior_normal_mean_walk_forward",
                    "baseline_method_recency": f"same_season_prior_normal_recency_decay_{recency_decay:g}",
                    "baseline_method_all_prior": (
                        "all_prior_weekends_sprint_scores_divided_by_prior_global_factor"
                    ),
                    "baseline_method_shrunk": shrunk_source,
                    "baseline_method_descriptive": "full_season_normal_mean_non_predictive",
                    "prior_global_sprint_factor": float(prior_sprint_factor),
                    "season_normal_mean": float(scale),
                }
            )
        rows.extend(event_outputs)
        for output in event_outputs:
            if pd.notna(output["x_strict_prior_normal"]):
                prior_sprint_pairs[str(output["entity_type"])].append(
                    (float(output["x_strict_prior_normal"]), float(output["sprint_score"]))
                )

    observations = pd.DataFrame(rows).sort_values(
        ["season", "round", "entity_type", "canonical_entity_id"]
    ).reset_index(drop=True)
    observations["x_strict_normalised"] = (
        observations["x_strict_prior_normal"] / observations["season_normal_mean"]
    )
    observations["y_normalised"] = observations["sprint_score"] / observations[
        "season_normal_mean"
    ]
    observations["price_percentile"] = observations.groupby(
        ["season", "round", "entity_type"]
    )["price"].rank(pct=True, method="average")
    observations["form_rank"] = observations.groupby(
        ["season", "round", "entity_type"]
    )["x_strict_prior_normal"].rank(ascending=False, method="first")
    observations["form_tier"] = "lower"
    driver_mask = observations["entity_type"].eq("driver")
    constructor_mask = observations["entity_type"].eq("constructor")
    observations.loc[driver_mask & observations["form_rank"].le(5), "form_tier"] = "top"
    observations.loc[
        driver_mask & observations["form_rank"].between(6, 15), "form_tier"
    ] = "middle"
    observations.loc[
        constructor_mask & observations["form_rank"].le(3), "form_tier"
    ] = "top"
    constructor_counts = observations.groupby(["season", "round", "entity_type"])[
        "canonical_entity_id"
    ].transform("size")
    observations.loc[
        constructor_mask
        & observations["form_rank"].gt(3)
        & observations["form_rank"].le(constructor_counts - 3),
        "form_tier",
    ] = "middle"
    observations["price_tier"] = pd.cut(
        observations["price_percentile"],
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return observations, event_summary


def _fit_design(
    design: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    coefficient_names: tuple[str, ...] = ("alpha", "beta"),
) -> dict[str, object]:
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(design).all(axis=1) & np.isfinite(y)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        valid &= np.isfinite(weights) & (weights > 0)
    design = design[valid]
    y = y[valid]
    weights = np.ones(len(y), dtype=float) if weights is None else weights[valid]
    if len(y) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Regression design is not mathematically identifiable.")
    root_w = np.sqrt(weights)
    weighted_design = design * root_w[:, None]
    weighted_y = y * root_w
    coefficients = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    predictions = design @ coefficients
    residuals = y - predictions
    weighted_sse = float(np.sum(weights * residuals**2))
    dof = len(y) - design.shape[1]
    xtwx_inverse = np.linalg.pinv(design.T @ (weights[:, None] * design))
    covariance = xtwx_inverse * (weighted_sse / dof)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / total if total else float("nan")
    adjusted = (
        1.0 - (1.0 - r_squared) * (len(y) - 1) / dof if np.isfinite(r_squared) else float("nan")
    )
    result: dict[str, object] = {
        "coefficients": coefficients,
        "covariance": covariance,
        "predictions": predictions,
        "residuals": residuals,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted,
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "bias": float(np.mean(predictions - y)),
        "observation_count": int(len(y)),
        "condition_number": float(np.linalg.cond(design)),
    }
    for index, name in enumerate(coefficient_names):
        result[name] = float(coefficients[index])
        result[f"standard_error_{name}"] = float(standard_errors[index])
        result[f"{name}_ci_lower"] = float(coefficients[index] - 1.96 * standard_errors[index])
        result[f"{name}_ci_upper"] = float(coefficients[index] + 1.96 * standard_errors[index])
    return result


def fit_linear(
    x: Iterable[float],
    y: Iterable[float],
    *,
    weights: Iterable[float] | None = None,
) -> dict[str, object]:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    design = np.column_stack([np.ones(len(x_array)), x_array])
    weight_array = None if weights is None else np.asarray(list(weights), dtype=float)
    return _fit_design(design, y_array, weights=weight_array)


def fit_huber(
    x: Iterable[float],
    y: Iterable[float],
    *,
    delta: float = 1.345,
    max_iterations: int = 100,
) -> dict[str, object]:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    weights = np.ones(len(y_array), dtype=float)
    previous = None
    for _ in range(max_iterations):
        result = fit_linear(x_array, y_array, weights=weights)
        coefficients = np.asarray(result["coefficients"])
        residuals = y_array - (coefficients[0] + coefficients[1] * x_array)
        median = np.median(residuals)
        scale = 1.4826 * np.median(np.abs(residuals - median))
        if not np.isfinite(scale) or scale < 1e-9:
            break
        scaled = np.abs(residuals) / scale
        weights = np.where(scaled <= delta, 1.0, delta / np.maximum(scaled, 1e-12))
        if previous is not None and np.max(np.abs(coefficients - previous)) < 1e-10:
            break
        previous = coefficients
    final = fit_linear(x_array, y_array, weights=weights)
    final["huber_weight_min"] = float(weights.min())
    final["huber_downweighted_observations"] = int((weights < 1.0).sum())
    return final


def _coefficient_row(
    fit: dict[str, object],
    *,
    period: str,
    entity_type: str,
    baseline_method: str,
    scale: str,
    estimator: str,
    sprint_event_count: int,
    excluded_count: int = 0,
    imputed_count: int = 0,
    model_variant: str = "hybrid",
) -> dict[str, object]:
    alpha = float(fit["alpha"])
    beta = float(fit["beta"])
    return {
        "period": period,
        "entity_type": entity_type,
        "baseline_method": baseline_method,
        "scale": scale,
        "estimator": estimator,
        "model_variant": model_variant,
        "alpha": alpha,
        "beta": beta,
        "standard_error_alpha": fit.get("standard_error_alpha"),
        "standard_error_beta": fit.get("standard_error_beta"),
        "alpha_ci_lower": fit.get("alpha_ci_lower"),
        "alpha_ci_upper": fit.get("alpha_ci_upper"),
        "beta_ci_lower": fit.get("beta_ci_lower"),
        "beta_ci_upper": fit.get("beta_ci_upper"),
        "confidence_interval_alpha": f"[{float(fit['alpha_ci_lower']):.6g}, {float(fit['alpha_ci_upper']):.6g}]",
        "confidence_interval_beta": f"[{float(fit['beta_ci_lower']):.6g}, {float(fit['beta_ci_upper']):.6g}]",
        "r_squared": fit["r_squared"],
        "adjusted_r_squared": fit["adjusted_r_squared"],
        "mae": fit["mae"],
        "rmse": fit["rmse"],
        "bias": fit["bias"],
        "observation_count": fit["observation_count"],
        "sprint_event_count": int(sprint_event_count),
        "excluded_observation_count": int(excluded_count),
        "imputed_observation_count": int(imputed_count),
        "condition_number": fit["condition_number"],
    }


def _season_weights(observations: pd.DataFrame, target_2026_share: float) -> np.ndarray:
    seasons = sorted(observations["season"].unique())
    if 2026 not in seasons or len(seasons) == 1:
        shares = {season: 1.0 / len(seasons) for season in seasons}
    else:
        historical_share = (1.0 - target_2026_share) / (len(seasons) - 1)
        shares = {season: historical_share for season in seasons}
        shares[2026] = target_2026_share
    counts = observations["season"].value_counts().to_dict()
    return observations["season"].map(lambda season: shares[int(season)] / counts[int(season)]).to_numpy(float)


def fit_season_fixed_effects(observations: pd.DataFrame, target_season: int = 2026) -> dict[str, object]:
    data = observations.dropna(subset=["x_strict_prior_normal", "sprint_score"]).copy()
    seasons = sorted(data["season"].unique())
    base = seasons[0]
    dummy_seasons = seasons[1:]
    design = np.column_stack(
        [
            np.ones(len(data)),
            data["x_strict_prior_normal"].to_numpy(float),
            *[data["season"].eq(season).astype(float).to_numpy() for season in dummy_seasons],
        ]
    )
    names = ("intercept", "beta", *[f"season_{season}" for season in dummy_seasons])
    fit = _fit_design(design, data["sprint_score"].to_numpy(float), coefficient_names=names)
    coefficients = dict(zip(names, np.asarray(fit["coefficients"], dtype=float)))
    alpha = coefficients["intercept"] + coefficients.get(f"season_{target_season}", 0.0)
    covariance = np.asarray(fit["covariance"])
    contrast = np.zeros(len(names))
    contrast[0] = 1.0
    if f"season_{target_season}" in names:
        contrast[names.index(f"season_{target_season}")] = 1.0
    alpha_se = float(np.sqrt(max(0.0, contrast @ covariance @ contrast)))
    fit.update(
        {
            "alpha": float(alpha),
            "standard_error_alpha": alpha_se,
            "alpha_ci_lower": float(alpha - 1.96 * alpha_se),
            "alpha_ci_upper": float(alpha + 1.96 * alpha_se),
            "standard_error_beta": fit["standard_error_beta"],
            "beta_ci_lower": fit["beta_ci_lower"],
            "beta_ci_upper": fit["beta_ci_upper"],
            "season_effects": json.dumps(coefficients, sort_keys=True),
            "base_season": int(base),
        }
    )
    return fit


def select_2026_weight(observations: pd.DataFrame, entity_type: str) -> tuple[float, pd.DataFrame]:
    entity = observations[
        observations["entity_type"].eq(entity_type)
        & observations["x_strict_prior_normal"].notna()
    ].copy()
    events = sorted(entity.loc[entity["season"].eq(2026), "event_cluster"].unique())
    rows: list[dict[str, float]] = []
    for share in WEIGHT_GRID:
        errors: list[float] = []
        for event in events:
            training = entity[~(entity["season"].eq(2026) & entity["event_cluster"].eq(event))]
            test = entity[entity["season"].eq(2026) & entity["event_cluster"].eq(event)]
            fit = fit_linear(
                training["x_strict_prior_normal"],
                training["sprint_score"],
                weights=_season_weights(training, share),
            )
            predictions = float(fit["alpha"]) + float(fit["beta"]) * test[
                "x_strict_prior_normal"
            ].to_numpy(float)
            errors.extend(np.abs(test["sprint_score"].to_numpy(float) - predictions))
        rows.append({"target_2026_weight": share, "leave_event_out_mae": float(np.mean(errors))})
    validation = pd.DataFrame(rows)
    best = validation.sort_values(["leave_event_out_mae", "target_2026_weight"]).iloc[0]
    return float(best["target_2026_weight"]), validation


def build_coefficient_tables(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    raw_rows: list[dict[str, object]] = []
    normalized_rows: list[dict[str, object]] = []
    for period, seasons in PERIODS.items():
        for entity_type in ("driver", "constructor"):
            subset = observations[
                observations["season"].isin(seasons)
                & observations["entity_type"].eq(entity_type)
            ]
            baseline_specs = (
                (
                    "x_strict_prior_normal",
                    "strict_same_season_prior_normal_walk_forward",
                    "hybrid",
                ),
                (
                    "x_recency_prior_normal",
                    "recency_weighted_same_season_prior_normal_walk_forward",
                    "hybrid_baseline_sensitivity",
                ),
                (
                    "x_all_prior_adjusted",
                    "all_prior_weekends_adjusted_by_prior_global_sprint_factor",
                    "hybrid_baseline_sensitivity",
                ),
                (
                    "x_shrunk_prior",
                    "same_season_prior_normal_plus_previous_season_or_group_prior",
                    "hybrid_shrunk_prior_sample",
                ),
            )
            for baseline_column, baseline_name, model_variant in baseline_specs:
                sample = subset.dropna(subset=[baseline_column, "sprint_score"])
                fit = fit_linear(sample[baseline_column], sample["sprint_score"])
                raw_rows.append(
                    _coefficient_row(
                        fit,
                        period=period,
                        entity_type=entity_type,
                        baseline_method=baseline_name,
                        scale="raw",
                        estimator="OLS",
                        sprint_event_count=sample["event_cluster"].nunique(),
                        excluded_count=len(subset) - len(sample),
                        imputed_count=(
                            int(sample["shrunk_prior_imputed"].sum())
                            if baseline_column == "x_shrunk_prior"
                            else 0
                        ),
                        model_variant=model_variant,
                    )
                )
            descriptive = subset.dropna(subset=["x_descriptive_full_season", "sprint_score"])
            descriptive_fit = fit_linear(
                descriptive["x_descriptive_full_season"], descriptive["sprint_score"]
            )
            raw_rows.append(
                _coefficient_row(
                    descriptive_fit,
                    period=period,
                    entity_type=entity_type,
                    baseline_method="descriptive_full_season_normal_non_predictive",
                    scale="raw",
                    estimator="OLS",
                    sprint_event_count=descriptive["event_cluster"].nunique(),
                    excluded_count=len(subset) - len(descriptive),
                    model_variant="descriptive_hybrid_non_predictive",
                )
            )
            normalized = subset.dropna(subset=["x_strict_normalised", "y_normalised"])
            normalized_fit = fit_linear(
                normalized["x_strict_normalised"], normalized["y_normalised"]
            )
            normalized_rows.append(
                _coefficient_row(
                    normalized_fit,
                    period=period,
                    entity_type=entity_type,
                    baseline_method="strict_walk_forward_divided_by_season_normal_mean",
                    scale="season_normalised",
                    estimator="OLS",
                    sprint_event_count=normalized["event_cluster"].nunique(),
                    excluded_count=len(subset) - len(normalized),
                )
            )

    season_coefficients = pd.DataFrame(raw_rows)
    normalised_coefficients = pd.DataFrame(normalized_rows)
    overall_rows: list[dict[str, object]] = []
    selected_weights: dict[str, float] = {}
    for entity_type in ("driver", "constructor"):
        current = observations[
            observations["season"].eq(2026)
            & observations["entity_type"].eq(entity_type)
        ].dropna(subset=["x_strict_prior_normal", "sprint_score"])
        ols = fit_linear(current["x_strict_prior_normal"], current["sprint_score"])
        huber = fit_huber(current["x_strict_prior_normal"], current["sprint_score"])
        overall_rows.extend(
            [
                _coefficient_row(
                    ols,
                    period="2026",
                    entity_type=entity_type,
                    baseline_method="strict_same_season_prior_normal_walk_forward",
                    scale="raw",
                    estimator="OLS",
                    sprint_event_count=current["event_cluster"].nunique(),
                    model_variant="2026_overall_hybrid",
                ),
                _coefficient_row(
                    huber,
                    period="2026",
                    entity_type=entity_type,
                    baseline_method="strict_same_season_prior_normal_walk_forward",
                    scale="raw",
                    estimator="Huber_IRLS",
                    sprint_event_count=current["event_cluster"].nunique(),
                    model_variant="2026_overall_robust_hybrid",
                ),
            ]
        )
        pooled = observations[
            observations["entity_type"].eq(entity_type)
        ].dropna(subset=["x_strict_prior_normal", "sprint_score"])
        naive = fit_linear(pooled["x_strict_prior_normal"], pooled["sprint_score"])
        overall_rows.append(
            _coefficient_row(
                naive,
                period="2023-2026",
                entity_type=entity_type,
                baseline_method="strict_same_season_prior_normal_walk_forward",
                scale="raw",
                estimator="OLS",
                sprint_event_count=pooled["event_cluster"].nunique(),
                model_variant="naive_raw_pooling",
            )
        )
        fixed = fit_season_fixed_effects(pooled)
        fixed_row = _coefficient_row(
            fixed,
            period="2023-2026",
            entity_type=entity_type,
            baseline_method="strict_same_season_prior_normal_walk_forward",
            scale="raw",
            estimator="OLS",
            sprint_event_count=pooled["event_cluster"].nunique(),
            model_variant="raw_with_season_fixed_effects_2026_intercept",
        )
        fixed_row["season_effects"] = fixed["season_effects"]
        overall_rows.append(fixed_row)
        for share, variant in ((0.25, "equal_season_weight"), (0.50, "2026_50_percent_weight")):
            weighted = fit_linear(
                pooled["x_strict_prior_normal"],
                pooled["sprint_score"],
                weights=_season_weights(pooled, share),
            )
            row = _coefficient_row(
                weighted,
                period="2023-2026",
                entity_type=entity_type,
                baseline_method="strict_same_season_prior_normal_walk_forward",
                scale="raw",
                estimator="weighted_OLS",
                sprint_event_count=pooled["event_cluster"].nunique(),
                model_variant=variant,
            )
            row["target_2026_weight"] = share
            overall_rows.append(row)
        selected, validation = select_2026_weight(observations, entity_type)
        selected_weights[entity_type] = selected
        selected_fit = fit_linear(
            pooled["x_strict_prior_normal"],
            pooled["sprint_score"],
            weights=_season_weights(pooled, selected),
        )
        selected_row = _coefficient_row(
            selected_fit,
            period="2023-2026",
            entity_type=entity_type,
            baseline_method="strict_same_season_prior_normal_walk_forward",
            scale="raw",
            estimator="weighted_OLS",
            sprint_event_count=pooled["event_cluster"].nunique(),
            model_variant="validation_selected_2026_weight",
        )
        selected_row["target_2026_weight"] = selected
        selected_row["weight_validation_mae"] = float(
            validation.loc[validation["target_2026_weight"].eq(selected), "leave_event_out_mae"].iloc[0]
        )
        overall_rows.append(selected_row)
    return (
        pd.DataFrame(overall_rows),
        season_coefficients,
        normalised_coefficients,
        selected_weights,
    )


def cluster_bootstrap_2026(
    observations: pd.DataFrame,
    *,
    samples: int = 10_000,
    seed: int = 20260806,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for entity_type in ("driver", "constructor"):
        entity = observations[
            observations["season"].eq(2026)
            & observations["entity_type"].eq(entity_type)
        ].dropna(subset=["x_strict_prior_normal", "sprint_score"])
        clusters = sorted(entity["event_cluster"].unique())
        coefficients: list[tuple[float, float]] = []
        attempts = 0
        while len(coefficients) < samples and attempts < samples * 3:
            attempts += 1
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            frame = pd.concat(
                [entity[entity["event_cluster"].eq(cluster)] for cluster in sampled],
                ignore_index=True,
            )
            try:
                fit = fit_linear(frame["x_strict_prior_normal"], frame["sprint_score"])
            except ValueError:
                continue
            coefficients.append((float(fit["alpha"]), float(fit["beta"])))
        if len(coefficients) < samples:
            raise RuntimeError(f"Could not obtain {samples} identifiable bootstrap fits for {entity_type}.")
        array = np.asarray(coefficients)
        for coefficient, index in (("alpha", 0), ("beta", 1)):
            percentiles = np.percentile(array[:, index], [2.5, 50.0, 97.5])
            rows.append(
                {
                    "period": "2026",
                    "entity_type": entity_type,
                    "coefficient": coefficient,
                    "bootstrap_method": "Sprint_event_cluster_resampling",
                    "bootstrap_samples": int(samples),
                    "random_seed": int(seed),
                    "p2_5": float(percentiles[0]),
                    "p50": float(percentiles[1]),
                    "p97_5": float(percentiles[2]),
                }
            )
    return pd.DataFrame(rows)


def leave_one_sprint_out_2026(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entity_type in ("driver", "constructor"):
        entity = observations[
            observations["season"].eq(2026)
            & observations["entity_type"].eq(entity_type)
        ].dropna(subset=["x_strict_prior_normal", "sprint_score"])
        full = fit_linear(entity["x_strict_prior_normal"], entity["sprint_score"])
        events = entity[["event_cluster", "round", "event_name"]].drop_duplicates()
        for event in events.itertuples(index=False):
            reduced = entity[entity["event_cluster"].ne(event.event_cluster)]
            fit = fit_linear(reduced["x_strict_prior_normal"], reduced["sprint_score"])
            rows.append(
                {
                    "entity_type": entity_type,
                    "excluded_season": 2026,
                    "excluded_round": int(event.round),
                    "excluded_event": event.event_name,
                    "remaining_sprint_events": reduced["event_cluster"].nunique(),
                    "alpha": fit["alpha"],
                    "beta": fit["beta"],
                    "change_from_full_alpha": float(fit["alpha"]) - float(full["alpha"]),
                    "change_from_full_beta": float(fit["beta"]) - float(full["beta"]),
                }
            )
    return pd.DataFrame(rows)


def _ridge_asset_coefficients(
    asset: pd.DataFrame,
    group_alpha: float,
    group_beta: float,
    penalty: float,
    center: float,
    scale: float,
) -> tuple[float, float]:
    if asset.empty:
        return float(group_alpha), float(group_beta)
    scale = scale if np.isfinite(scale) and scale > 1e-9 else 1.0
    z = (asset["x_strict_prior_normal"].to_numpy(float) - center) / scale
    design = np.column_stack([np.ones(len(z)), z])
    y = asset["sprint_score"].to_numpy(float)
    group_z = np.array([group_alpha + group_beta * center, group_beta * scale])
    matrix = design.T @ design + float(penalty) * np.eye(2)
    target = design.T @ y + float(penalty) * group_z
    fitted_z = np.linalg.solve(matrix, target)
    beta = float(fitted_z[1] / scale)
    alpha = float(fitted_z[0] - beta * center)
    return alpha, beta


def select_ridge_penalty(observations: pd.DataFrame, entity_type: str) -> tuple[float, pd.DataFrame]:
    entity = observations[
        observations["season"].eq(2026)
        & observations["entity_type"].eq(entity_type)
    ].dropna(subset=["x_strict_prior_normal", "sprint_score"])
    events = sorted(entity["event_cluster"].unique())
    rows: list[dict[str, float]] = []
    for penalty in RIDGE_GRID:
        errors: list[float] = []
        for event in events:
            training = entity[entity["event_cluster"].ne(event)]
            test = entity[entity["event_cluster"].eq(event)]
            group = fit_linear(training["x_strict_prior_normal"], training["sprint_score"])
            center = float(training["x_strict_prior_normal"].mean())
            scale = float(training["x_strict_prior_normal"].std(ddof=0))
            for row in test.itertuples(index=False):
                asset_training = training[
                    training["canonical_entity_id"].eq(row.canonical_entity_id)
                ]
                alpha, beta = _ridge_asset_coefficients(
                    asset_training,
                    float(group["alpha"]),
                    float(group["beta"]),
                    penalty,
                    center,
                    scale,
                )
                errors.append(abs(float(row.sprint_score) - (alpha + beta * row.x_strict_prior_normal)))
        rows.append({"ridge_penalty": penalty, "leave_event_out_mae": float(np.mean(errors))})
    validation = pd.DataFrame(rows)
    best = validation.sort_values(["leave_event_out_mae", "ridge_penalty"]).iloc[0]
    return float(best["ridge_penalty"]), validation


def individual_2026_coefficients(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    raw_rows: list[dict[str, object]] = []
    shrunk_rows: list[dict[str, object]] = []
    selected_penalties: dict[str, float] = {}
    for entity_type in ("driver", "constructor"):
        entity = observations[
            observations["season"].eq(2026)
            & observations["entity_type"].eq(entity_type)
        ].dropna(subset=["x_strict_prior_normal", "sprint_score"])
        group = fit_linear(entity["x_strict_prior_normal"], entity["sprint_score"])
        penalty, validation = select_ridge_penalty(observations, entity_type)
        selected_penalties[entity_type] = penalty
        center = float(entity["x_strict_prior_normal"].mean())
        scale = float(entity["x_strict_prior_normal"].std(ddof=0))
        for asset_id, asset in entity.groupby("canonical_entity_id", sort=True):
            name = str(asset["entity"].iloc[-1])
            x = asset["x_strict_prior_normal"].to_numpy(float)
            y = asset["sprint_score"].to_numpy(float)
            design = np.column_stack([np.ones(len(x)), x])
            rank = np.linalg.matrix_rank(design)
            condition = float(np.linalg.cond(design)) if rank == 2 else float("inf")
            x_range = float(np.ptp(x)) if len(x) else 0.0
            variation_identifiable = bool(rank == 2 and x_range >= 0.5 and condition < 1e4)
            reliably_identifiable = bool(len(x) >= 8 and variation_identifiable)
            if rank == 2:
                fitted = np.linalg.lstsq(design, y, rcond=None)[0]
                predictions = design @ fitted
                total = float(np.sum((y - y.mean()) ** 2))
                r_squared = 1.0 - float(np.sum((y - predictions) ** 2)) / total if total else np.nan
                raw_alpha, raw_beta = map(float, fitted)
            else:
                raw_alpha = raw_beta = r_squared = float("nan")
            raw_rows.append(
                {
                    "entity": name,
                    "canonical_entity_id": asset_id,
                    "entity_type": entity_type,
                    "sprint_observations": int(len(asset)),
                    "raw_alpha": raw_alpha,
                    "raw_beta": raw_beta,
                    "raw_r_squared": r_squared,
                    "condition_number": condition,
                    "x_range": x_range,
                    "mathematically_identifiable": bool(rank == 2),
                    "slope_variation_identifiable": variation_identifiable,
                    "reliably_identifiable": reliably_identifiable,
                    "identifiability_warning": (
                        "low x variation, sparse observations, or ill-conditioned design"
                        if not variation_identifiable
                        else "only four-or-fewer observations; descriptive slope remains unstable"
                    ),
                    "replacement_or_sparse_asset": bool(len(asset) < 4),
                }
            )
            shrunk_alpha, shrunk_beta = _ridge_asset_coefficients(
                asset,
                float(group["alpha"]),
                float(group["beta"]),
                penalty,
                center,
                scale,
            )
            shrunk_rows.append(
                {
                    "entity": name,
                    "canonical_entity_id": asset_id,
                    "entity_type": entity_type,
                    "sprint_observations": int(len(asset)),
                    "ridge_penalty": penalty,
                    "penalty_validation_mae": float(
                        validation.loc[
                            validation["ridge_penalty"].eq(penalty), "leave_event_out_mae"
                        ].iloc[0]
                    ),
                    "group_alpha": group["alpha"],
                    "group_beta": group["beta"],
                    "shrunk_alpha": shrunk_alpha,
                    "shrunk_beta": shrunk_beta,
                    "alpha_deviation_from_group": shrunk_alpha - float(group["alpha"]),
                    "beta_deviation_from_group": shrunk_beta - float(group["beta"]),
                    "replacement_or_sparse_asset": bool(len(asset) < 4),
                }
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(shrunk_rows), selected_penalties


def _rank_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(pd.Series(actual).rank().corr(pd.Series(predicted).rank()))


def _calibration_slope(actual: np.ndarray, predicted: np.ndarray) -> float:
    if np.ptp(predicted) < 1e-12:
        return float("nan")
    return float(fit_linear(predicted, actual)["beta"])


def _comparison_metrics(predictions: pd.DataFrame, model: str, entity_type: str) -> dict[str, object]:
    residual = predictions["prediction"] - predictions["sprint_score"]
    overlaps: list[float] = []
    top_n = 5 if entity_type == "driver" else 3
    for _event, group in predictions.groupby("event_cluster"):
        actual = set(group.nlargest(top_n, "sprint_score")["canonical_entity_id"])
        predicted = set(group.nlargest(top_n, "prediction")["canonical_entity_id"])
        overlaps.append(len(actual & predicted) / top_n)
    return {
        "entity_type": entity_type,
        "model": model,
        "mae": float(residual.abs().mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(residual.mean()),
        "rank_correlation": _rank_correlation(
            predictions["sprint_score"].to_numpy(float), predictions["prediction"].to_numpy(float)
        ),
        "top_asset_overlap": float(np.mean(overlaps)),
        "calibration_slope": _calibration_slope(
            predictions["sprint_score"].to_numpy(float), predictions["prediction"].to_numpy(float)
        ),
        "observation_count": int(len(predictions)),
        "sprint_event_count": int(predictions["event_cluster"].nunique()),
    }


def model_comparison_2026(
    observations: pd.DataFrame,
    selected_weights: dict[str, float],
    selected_penalties: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entity_type in ("driver", "constructor"):
        current = observations[
            observations["season"].eq(2026)
            & observations["entity_type"].eq(entity_type)
        ].dropna(subset=["x_strict_prior_normal", "sprint_score"])
        pooled = observations[
            observations["entity_type"].eq(entity_type)
        ].dropna(subset=["x_strict_prior_normal", "sprint_score"])
        model_frames: dict[str, list[pd.DataFrame]] = {
            "0_no_adjustment": [],
            "1_global_multiplier": [],
            "2_global_additive": [],
            "3_hybrid_linear": [],
            "4_2026_weighted_hybrid": [],
            "5_shrunk_asset_hybrid": [],
        }
        for event in sorted(current["event_cluster"].unique()):
            test = current[current["event_cluster"].eq(event)].copy()
            training = current[current["event_cluster"].ne(event)]
            x_train = training["x_strict_prior_normal"].to_numpy(float)
            y_train = training["sprint_score"].to_numpy(float)
            multiplier = float(np.dot(x_train, y_train) / np.dot(x_train, x_train))
            additive = float(np.mean(y_train - x_train))
            hybrid = fit_linear(x_train, y_train)
            pooled_training = pooled[
                ~(pooled["season"].eq(2026) & pooled["event_cluster"].eq(event))
            ]
            weighted = fit_linear(
                pooled_training["x_strict_prior_normal"],
                pooled_training["sprint_score"],
                weights=_season_weights(pooled_training, selected_weights[entity_type]),
            )
            group_center = float(training["x_strict_prior_normal"].mean())
            group_scale = float(training["x_strict_prior_normal"].std(ddof=0))
            formulas = {
                "0_no_adjustment": test["x_strict_prior_normal"].to_numpy(float),
                "1_global_multiplier": multiplier * test["x_strict_prior_normal"].to_numpy(float),
                "2_global_additive": test["x_strict_prior_normal"].to_numpy(float) + additive,
                "3_hybrid_linear": float(hybrid["alpha"]) + float(hybrid["beta"]) * test[
                    "x_strict_prior_normal"
                ].to_numpy(float),
                "4_2026_weighted_hybrid": float(weighted["alpha"]) + float(weighted["beta"]) * test[
                    "x_strict_prior_normal"
                ].to_numpy(float),
            }
            for model, predicted in formulas.items():
                frame = test.copy()
                frame["prediction"] = predicted
                model_frames[model].append(frame)
            shrunk_predictions: list[float] = []
            for asset in test.itertuples(index=False):
                asset_training = training[
                    training["canonical_entity_id"].eq(asset.canonical_entity_id)
                ]
                alpha, beta = _ridge_asset_coefficients(
                    asset_training,
                    float(hybrid["alpha"]),
                    float(hybrid["beta"]),
                    selected_penalties[entity_type],
                    group_center,
                    group_scale,
                )
                shrunk_predictions.append(alpha + beta * float(asset.x_strict_prior_normal))
            shrunk_frame = test.copy()
            shrunk_frame["prediction"] = shrunk_predictions
            model_frames["5_shrunk_asset_hybrid"].append(shrunk_frame)
        for model, frames in model_frames.items():
            rows.append(_comparison_metrics(pd.concat(frames, ignore_index=True), model, entity_type))
    return pd.DataFrame(rows)


def residual_strength_summary(
    observations: pd.DataFrame,
    overall_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entity_type in ("driver", "constructor"):
        coefficient = overall_coefficients[
            overall_coefficients["period"].eq("2026")
            & overall_coefficients["entity_type"].eq(entity_type)
            & overall_coefficients["estimator"].eq("OLS")
        ].iloc[0]
        entity = observations[
            observations["season"].eq(2026)
            & observations["entity_type"].eq(entity_type)
        ].copy()
        entity["prediction"] = coefficient.alpha + coefficient.beta * entity[
            "x_strict_prior_normal"
        ]
        entity["residual"] = entity["prediction"] - entity["sprint_score"]
        for dimension, column in (("price", "price_tier"), ("pre_event_form", "form_tier")):
            for tier, group in entity.groupby(column):
                rows.append(
                    {
                        "entity_type": entity_type,
                        "dimension": dimension,
                        "tier": tier,
                        "mean_normal_ev": group["x_strict_prior_normal"].mean(),
                        "mean_actual_sprint_score": group["sprint_score"].mean(),
                        "mean_predicted_sprint_score": group["prediction"].mean(),
                        "mean_residual_prediction_minus_actual": group["residual"].mean(),
                        "mae": group["residual"].abs().mean(),
                        "observation_count": int(len(group)),
                    }
                )
    return pd.DataFrame(rows)


def build_prediction_examples(overall_coefficients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = overall_coefficients[
        (
            overall_coefficients["period"].eq("2026")
            & overall_coefficients["estimator"].isin(["OLS", "Huber_IRLS"])
        )
        | overall_coefficients["model_variant"].eq("validation_selected_2026_weight")
    ]
    for coefficient in candidates.itertuples(index=False):
        levels = DRIVER_LEVELS if coefficient.entity_type == "driver" else CONSTRUCTOR_LEVELS
        for normal_ev in levels:
            prediction = float(coefficient.alpha + coefficient.beta * normal_ev)
            rows.append(
                {
                    "entity_type": coefficient.entity_type,
                    "candidate": coefficient.model_variant + "_" + coefficient.estimator,
                    "period": coefficient.period,
                    "alpha": coefficient.alpha,
                    "beta": coefficient.beta,
                    "normal_ev": normal_ev,
                    "predicted_sprint_ev": prediction,
                    "absolute_uplift": prediction - normal_ev,
                    "effective_multiplier": _safe_divide(prediction, normal_ev),
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    shown = frame[columns].copy()
    for column in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def build_report(
    observations: pd.DataFrame,
    overall: pd.DataFrame,
    season: pd.DataFrame,
    normalised: pd.DataFrame,
    comparison: pd.DataFrame,
    leave_out: pd.DataFrame,
    bootstrap: pd.DataFrame,
    raw_individual: pd.DataFrame,
    shrunk_individual: pd.DataFrame,
    examples: pd.DataFrame,
    residuals: pd.DataFrame,
    selected_weights: dict[str, float],
    selected_penalties: dict[str, float],
    dataset_path: Path,
) -> str:
    current = overall[
        overall["period"].eq("2026") & overall["estimator"].eq("OLS")
    ].set_index("entity_type")
    blended = overall[overall["model_variant"].eq("validation_selected_2026_weight")].set_index(
        "entity_type"
    )
    current_boot = bootstrap.pivot(index="entity_type", columns="coefficient", values=["p2_5", "p97_5"])
    strict_season = season[
        season["baseline_method"].eq("strict_same_season_prior_normal_walk_forward")
    ]
    strict_normalised = normalised[
        normalised["baseline_method"].eq("strict_walk_forward_divided_by_season_normal_mean")
    ]
    best_models = comparison.sort_values(["entity_type", "mae"]).groupby("entity_type").head(1)
    extreme_shrunk = shrunk_individual.assign(
        deviation_magnitude=lambda frame: frame["alpha_deviation_from_group"].abs()
        + frame["beta_deviation_from_group"].abs()
    ).sort_values(["entity_type", "deviation_magnitude"], ascending=[True, False]).groupby(
        "entity_type"
    ).head(5)
    constructor = current.loc["constructor"]
    constructor_levels = examples[
        examples["entity_type"].eq("constructor")
        & examples["candidate"].str.startswith("2026_overall_hybrid_OLS")
    ]
    sparse_count = int((~raw_individual["reliably_identifiable"]).sum())
    excluded = observations.groupby("entity_type")["strict_excluded"].sum().to_dict()
    imputed = observations.groupby("entity_type")["shrunk_prior_imputed"].sum().to_dict()
    recommendation = best_models.set_index("entity_type")["model"].to_dict()
    return f"""# Sprint EV linear-regression research

## 1. Executive conclusion

Leakage-safe 2026 OLS candidates:

- Driver: **alpha = {current.loc['driver', 'alpha']:.4f}, beta = {current.loc['driver', 'beta']:.4f}**
- Constructor: **alpha = {constructor.alpha:.4f}, beta = {constructor.beta:.4f}**

Validation-selected historical/2026 blends:

- Driver: **alpha = {blended.loc['driver', 'alpha']:.4f}, beta = {blended.loc['driver', 'beta']:.4f}**
  with 2026 weight {selected_weights['driver']:.2f}.
- Constructor: **alpha = {blended.loc['constructor', 'alpha']:.4f}, beta = {blended.loc['constructor', 'beta']:.4f}**
  with 2026 weight {selected_weights['constructor']:.2f}.

Leave-one-Sprint-event-out validation selects `{recommendation.get('driver')}` for drivers and
`{recommendation.get('constructor')}` for constructors. These are research candidates only; no
coefficient is activated in production.

## 2. 2026 results

{_markdown_table(overall[overall['period'].eq('2026')], ['entity_type', 'estimator', 'alpha', 'beta', 'standard_error_alpha', 'standard_error_beta', 'r_squared', 'mae', 'rmse', 'observation_count'])}

Event-cluster bootstrap intervals:

{_markdown_table(bootstrap, ['entity_type', 'coefficient', 'p2_5', 'p50', 'p97_5', 'bootstrap_samples'])}

Leave-one-Sprint-event coefficient sensitivity is in `leave_one_sprint_out.csv`. Because all assets
within a weekend share conditions, the event-cluster intervals—not row-level OLS standard errors—are
the preferred uncertainty statement.

## 3. Individual 2026 assets

Every driver has four Sprint observations; one constructor has only three. With two fitted parameters,
raw individual fits have at most two residual degrees of freedom. {sparse_count} fits additionally fail
the conservative variation/conditioning reliability screen. Raw coefficients are descriptive and
unstable.

Ridge penalties selected by leave-one-event-out validation are {selected_penalties['driver']:.3g} for
drivers and {selected_penalties['constructor']:.3g} for constructors. Largest shrunk deviations:

{_markdown_table(extreme_shrunk, ['entity', 'entity_type', 'sprint_observations', 'shrunk_alpha', 'shrunk_beta', 'alpha_deviation_from_group', 'beta_deviation_from_group'])}

The group coefficients are more reliable than any individual pair. No mixed-effects package was
available, so ridge partial pooling was used without adding a dependency.

## 4. Season-by-season results

Strict predictive walk-forward regressions:

{_markdown_table(strict_season[strict_season['period'].isin(['2023','2024','2025','2026'])], ['period', 'entity_type', 'alpha', 'beta', 'r_squared', 'mae', 'observation_count', 'excluded_observation_count'])}

Strict baselines excluded {int(excluded.get('driver', 0))} driver and
{int(excluded.get('constructor', 0))} constructor observations. The shrunk sample explicitly imputed
or materially shrank {int(imputed.get('driver', 0))} and {int(imputed.get('constructor', 0))}, respectively.

## 5. Pooled results

{_markdown_table(strict_season[strict_season['period'].isin(['2023-2025','2023-2026'])], ['period', 'entity_type', 'alpha', 'beta', 'r_squared', 'mae', 'rmse'])}

Season-normalised pooling, raw fixed-effects estimates, equal-season weighting, 50% 2026 weighting,
and validation-selected weighting are recorded in `normalised_coefficients.csv` and
`overall_coefficients.csv`.

## 6. Model comparison

{_markdown_table(comparison, ['entity_type', 'model', 'mae', 'rmse', 'bias', 'rank_correlation', 'top_asset_overlap', 'calibration_slope'])}

Metrics are out-of-event predictions from leaving each completed 2026 Sprint out in turn. Historical
prices were retained for residual strength diagnostics, but the production optimiser was deliberately
not invoked or changed.

## 7. Strength behaviour

Representative 2026 OLS predictions:

{_markdown_table(examples[examples['candidate'].str.startswith('2026_overall_hybrid_OLS')], ['entity_type', 'normal_ev', 'predicted_sprint_ev', 'absolute_uplift', 'effective_multiplier'])}

Residual summaries by within-event price and pre-event-form tiers are in
`residual_by_strength.csv`. They diagnose—not prescribe—possible non-linearity.

Constructor-specific interpretation: the recorded 2026 constructor hybrid has alpha
`{constructor.alpha:.4f}` and beta `{constructor.beta:.4f}`. Its implied uplifts across the requested
strength grid are:

{_markdown_table(constructor_levels, ['normal_ev', 'predicted_sprint_ev', 'absolute_uplift', 'effective_multiplier'])}

This directly uses recorded constructor totals. The observation file records 11 constructors in the
first three completed 2026 Sprints and 10 in round 9, compared with 10 in earlier seasons.
The positive intercept with beta below 1 explains why a large fixed Sprint opportunity can coexist
with a lower pooled percentage multiplier. For the 2026 OLS line, absolute uplift is
`alpha + (beta - 1) × normal_ev`, declining from `{constructor.alpha:.4f}` at zero normal EV to
`{constructor.alpha + (constructor.beta - 1) * 50:.4f}` at normal EV 50. The current sample therefore
does **not** show stronger constructors receiving larger absolute Sprint uplifts. Per-asset scoring
prevents the additional 2026 constructor from mechanically inflating the regression.

## 8. Scoring-rule sensitivity

{_markdown_table(strict_normalised[strict_normalised['period'].isin(['2023-2025','2023-2026'])], ['period', 'entity_type', 'alpha', 'beta', 'r_squared', 'mae'])}

Season normalisation divides both baseline and Sprint score by that season/entity normal-event mean.
It is the main cross-season comparability view; raw coefficients remain the directly interpretable
point-scale candidates.

## 9. Recommendation

Use the validation-selected historical/2026 hybrid as the specific first shadow-production candidate:
`Sprint EV = {blended.loc['driver', 'alpha']:.4f} + {blended.loc['driver', 'beta']:.4f} × normal EV`
for drivers and
`Sprint EV = {blended.loc['constructor', 'alpha']:.4f} + {blended.loc['constructor', 'beta']:.4f} × normal EV`
for constructors. The direct 2026 OLS lines remain the contemporary sensitivity, while individual
ridge estimates remain diagnostic. Do not activate any coefficient until shadow forecasts confirm
calibration on new Sprint events.

## 10. Limitations

- Only four 2026 Sprint weekends are complete.
- Asset outcomes within a Sprint event are correlated; only four event clusters drive uncertainty.
- Crashes, DNFs, penalties, negative scores and zeroes are retained as realised outcomes.
- Individual two-parameter regressions are extremely sparse and often ill-conditioned.
- Fantasy scoring rules and point scales change across seasons.
- Replacement drivers and the one sparse constructor have less history.
- A single line may miss threshold effects or other non-linearity.
- Prices and strength evolve; price tiers are within-event diagnostics, not causal controls.
- Descriptive full-season baselines use future normal events and are explicitly non-predictive.

Source: `{dataset_path}`. Only recorded, non-reconstructed 2023–2026 canonical totals and verified
local Sprint schedule metadata were used. No network call or production-model change was made.
"""


def run_analysis(
    dataset_path: str | Path = DEFAULT_CANONICAL_DATASET_PATH,
    schedule_dir: str | Path = Path("data/cache"),
    output_dir: str | Path = Path("reports/sprint_linear_regression"),
    *,
    seed: int = 20260806,
    bootstrap_samples: int = 10_000,
) -> dict[str, pd.DataFrame | str]:
    data = load_recorded_data(dataset_path)
    schedule = load_schedule_metadata(schedule_dir)
    observations, _events = build_observation_dataset(data, schedule)
    overall, season, normalised, selected_weights = build_coefficient_tables(observations)
    bootstrap = cluster_bootstrap_2026(observations, samples=bootstrap_samples, seed=seed)
    leave_out = leave_one_sprint_out_2026(observations)
    raw_individual, shrunk_individual, selected_penalties = individual_2026_coefficients(observations)
    comparison = model_comparison_2026(observations, selected_weights, selected_penalties)
    examples = build_prediction_examples(overall)
    residuals = residual_strength_summary(observations, overall)
    report = build_report(
        observations,
        overall,
        season,
        normalised,
        comparison,
        leave_out,
        bootstrap,
        raw_individual,
        shrunk_individual,
        examples,
        residuals,
        selected_weights,
        selected_penalties,
        Path(dataset_path),
    )
    outputs: dict[str, pd.DataFrame | str] = {
        "overall_coefficients.csv": overall,
        "season_coefficients.csv": season,
        "normalised_coefficients.csv": normalised,
        "model_comparison.csv": comparison,
        "leave_one_sprint_out.csv": leave_out,
        "bootstrap_intervals.csv": bootstrap,
        "individual_2026_raw_coefficients.csv": raw_individual,
        "individual_2026_shrunk_coefficients.csv": shrunk_individual,
        "prediction_examples.csv": examples,
        "residual_by_strength.csv": residuals,
        "observation_dataset.csv": observations,
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
    parser.add_argument("--output-dir", type=Path, default=Path("reports/sprint_linear_regression"))
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
    overall = outputs["overall_coefficients.csv"]
    assert isinstance(overall, pd.DataFrame)
    print(
        overall[
            overall["period"].eq("2026") & overall["estimator"].eq("OLS")
        ][["entity_type", "alpha", "beta"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
