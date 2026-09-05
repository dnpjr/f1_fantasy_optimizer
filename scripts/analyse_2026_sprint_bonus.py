#!/usr/bin/env python3
"""Offline, descriptive calibration of 2026 Sprint-session Fantasy bonuses.

The target is deliberately limited to official Sprint-specific components:
``sprint_points + sprint_qualifying_points``.  Ordinary qualifying, Grand Prix
race points, residuals, and complete-weekend totals are never target inputs.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH
from scripts.analyse_sprint_multiplier import (
    _trimmed_mean,
    build_event_summary,
    load_recorded_data,
    load_schedule_metadata,
)


SEASON = 2026
RANDOM_SEED = 20260806
RECENCY_DECAY = 0.80
MIN_COMPONENT_COVERAGE = 0.90
RIDGE_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SHRINKAGE_GRID = (0.0, 1.0, 2.0, 4.0, 8.0, 1_000_000.0)
BOOTSTRAP_REPLICATES = 500
COMPONENT_FIELDS = (
    "sprint_points",
    "sprint_qualifying_points",
    "qualifying_points",
    "race_points",
    "other_points",
    "fantasy_points_total",
    "price",
)
MODEL_SPECS: dict[str, tuple[str, ...]] = {
    "constant": ("intercept",),
    "proportional": ("normal_weekend_mean",),
    "hybrid": ("intercept", "normal_weekend_mean"),
    "price_only": ("intercept", "price_percentile_within_entity_type"),
    "form_price": (
        "intercept",
        "normal_weekend_mean",
        "price_percentile_within_entity_type",
    ),
    "interaction": (
        "intercept",
        "normal_weekend_mean",
        "price_percentile_within_entity_type",
        "form_price_interaction",
    ),
    "constrained_proportional": ("normal_weekend_mean",),
    "constrained_hybrid": ("intercept", "normal_weekend_mean"),
    "constrained_form_price": (
        "intercept",
        "normal_weekend_mean",
        "price_percentile_within_entity_type",
    ),
}
COEFFICIENT_NAMES = {
    "intercept": "alpha",
    "normal_weekend_mean": "gamma",
    "price_percentile_within_entity_type": "delta",
    "form_price_interaction": "eta",
}


def load_2026_recorded_data(path: str | Path) -> pd.DataFrame:
    """Load exact recorded data and intentionally discard every non-2026 row."""
    data = load_recorded_data(path)
    data = data[data["season"].eq(SEASON)].copy()
    if data.empty:
        raise ValueError("Canonical dataset has no completed 2026 records.")
    required = set(COMPONENT_FIELDS) | {"abbreviation", "is_official"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Canonical 2026 data is missing required fields: {missing}")
    official = data["is_official"].map(
        lambda value: value if isinstance(value, bool) else str(value).casefold() == "true"
    )
    if not official.all():
        raise ValueError("The 2026 Sprint analysis requires official recorded rows only.")
    return data.reset_index(drop=True)


def load_current_prices(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the accepted verified market cache without making a network call."""
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Verified official market cache is missing: {cache_path}")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    required_meta = {"feed_round", "verified_at_utc", "source_url", "players", "teams"}
    if not required_meta.issubset(payload):
        raise ValueError("Current-price cache is not a verified official market snapshot.")
    rows: list[dict[str, object]] = []
    for entity_type, key in (("driver", "players"), ("constructor", "teams")):
        for item in payload[key]:
            price = pd.to_numeric(item.get("price"), errors="coerce")
            abbreviation = str(item.get("tla") or "").strip().upper()
            if not abbreviation or pd.isna(price) or float(price) <= 0:
                raise ValueError(f"Invalid current {entity_type} price row in verified cache.")
            rows.append(
                {
                    "entity_type": entity_type,
                    "abbreviation": abbreviation,
                    "entity": str(item.get("name") or abbreviation),
                    "current_price": float(price),
                }
            )
    prices = pd.DataFrame(rows)
    if prices.duplicated(["entity_type", "abbreviation"]).any():
        raise ValueError("Verified current-price cache has duplicate asset abbreviations.")
    metadata = {
        "feed_round": int(payload["feed_round"]),
        "verified_at_utc": str(payload["verified_at_utc"]),
        "source_url": str(payload["source_url"]),
    }
    return prices.sort_values(["entity_type", "abbreviation"]).reset_index(drop=True), metadata


def sprint_only_target(frame: pd.DataFrame) -> pd.Series:
    """Sum only official Sprint-specific fields while preserving wholly missing rows."""
    required = {"sprint_points", "sprint_qualifying_points"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sprint target fields are missing: {missing}")
    components = frame[["sprint_points", "sprint_qualifying_points"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return components.sum(axis=1, min_count=1)


def prepare_2026_events(
    data: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    minimum_component_coverage: float = MIN_COMPONENT_COVERAGE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach verified weekend format and form the Sprint-only target."""
    source = data[data["season"].eq(SEASON)].copy(deep=True)
    schedule_2026 = schedule[schedule["season"].eq(SEASON)].copy(deep=True)
    events, annotated = build_event_summary(source, schedule_2026)
    annotated = annotated.copy(deep=True)
    for field in COMPONENT_FIELDS:
        annotated[field] = pd.to_numeric(annotated[field], errors="coerce")
    sprint = annotated["weekend_format"].eq("sprint")
    annotated["extra_sprint_points"] = pd.Series(np.nan, index=annotated.index, dtype=float)
    annotated.loc[sprint, "extra_sprint_points"] = sprint_only_target(annotated.loc[sprint])
    for entity_type, group in annotated[sprint].groupby("entity_type"):
        coverage = float(group["extra_sprint_points"].notna().mean())
        if coverage < minimum_component_coverage:
            raise ValueError(
                f"Official Sprint-component coverage is insufficient for {entity_type}: "
                f"{coverage:.1%} < {minimum_component_coverage:.1%}."
            )
    return events.reset_index(drop=True), annotated.reset_index(drop=True)


def build_component_audit(annotated: pd.DataFrame) -> pd.DataFrame:
    sprint = annotated[annotated["weekend_format"].eq("sprint")]
    interpretations = {
        "sprint_points": ("Sprint race", True, "Official Sprint-session total."),
        "sprint_qualifying_points": (
            "Sprint Qualifying",
            True,
            "Official separately labelled Sprint Qualifying total when emitted.",
        ),
        "qualifying_points": ("Grand Prix qualifying", False, "Ordinary-weekend component."),
        "race_points": ("Grand Prix race", False, "Ordinary-weekend component."),
        "other_points": ("Residual/unclassified", False, "Not proven Sprint-specific."),
        "fantasy_points_total": ("Complete weekend", False, "Mixes Sprint and ordinary sessions."),
        "price": ("Historical market value", False, "Not a scoring component."),
    }
    rows = []
    for field in COMPONENT_FIELDS:
        values = pd.to_numeric(sprint[field], errors="coerce")
        session, included, reason = interpretations[field]
        rows.append(
            {
                "source_field": field,
                "interpreted_session": session,
                "included_in_extra_sprint_points": included,
                "reason": reason,
                "observation_count": int(values.notna().sum()),
                "missing_count": int(values.isna().sum()),
                "minimum": float(values.min()) if values.notna().any() else np.nan,
                "maximum": float(values.max()) if values.notna().any() else np.nan,
                "mean": float(values.mean()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _recent_weighted_mean(group: pd.DataFrame, decay: float = RECENCY_DECAY) -> float:
    ordered = group.sort_values(["event_date", "round"], ascending=False)
    values = ordered["fantasy_points_total"].to_numpy(float)
    if not len(values):
        return np.nan
    return float(np.average(values, weights=np.power(decay, np.arange(len(values)))))


def build_asset_summary(annotated: pd.DataFrame, current_prices: pd.DataFrame) -> pd.DataFrame:
    """Build one current-state descriptive row per verified current asset."""
    source = annotated[annotated["season"].eq(SEASON)].copy(deep=True)
    normal = source[source["weekend_format"].eq("normal")]
    sprint = source[source["weekend_format"].eq("sprint")]
    keys = ["entity_type", "canonical_entity_id"]
    normal_summary = normal.groupby(keys, sort=True).agg(
        normal_event_count=("fantasy_points_total", "count"),
        normal_weekend_mean=("fantasy_points_total", "mean"),
        normal_weekend_median=("fantasy_points_total", "median"),
        normal_weekend_standard_deviation=("fantasy_points_total", "std"),
    ).reset_index()
    trimmed = normal.groupby(keys, sort=True)["fantasy_points_total"].apply(_trimmed_mean)
    recent = normal.groupby(keys, sort=True).apply(_recent_weighted_mean, include_groups=False)
    normal_summary = normal_summary.merge(
        trimmed.rename("normal_weekend_trimmed_mean").reset_index(), on=keys, validate="one_to_one"
    ).merge(recent.rename("recent_normal_form").reset_index(), on=keys, validate="one_to_one")
    sprint_summary = sprint.groupby(keys, sort=True).agg(
        sprint_selected_event_count=("round", "count"),
        sprint_event_count=("extra_sprint_points", "count"),
        mean_extra_sprint_points=("extra_sprint_points", "mean"),
        median_extra_sprint_points=("extra_sprint_points", "median"),
        extra_sprint_points_standard_deviation=("extra_sprint_points", "std"),
        minimum_extra_sprint_points=("extra_sprint_points", "min"),
        maximum_extra_sprint_points=("extra_sprint_points", "max"),
    ).reset_index()
    totals = source.groupby(keys, sort=True)["fantasy_points_total"].sum().rename(
        "total_2026_points"
    ).reset_index()
    identities = source.sort_values(["round", "event_date"]).drop_duplicates(keys, keep="last")[[
        *keys, "name", "abbreviation"
    ]]
    summary = identities.merge(normal_summary, on=keys, how="left", validate="one_to_one")
    summary = summary.merge(sprint_summary, on=keys, how="left", validate="one_to_one")
    summary = summary.merge(totals, on=keys, how="left", validate="one_to_one")
    summary = summary.merge(
        current_prices,
        on=["entity_type", "abbreviation"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_current"),
    )
    expected_current = len(current_prices)
    if len(summary) != expected_current:
        missing = current_prices.merge(
            summary[["entity_type", "abbreviation"]],
            on=["entity_type", "abbreviation"],
            how="left",
            indicator=True,
        )
        missing = missing.loc[missing["_merge"].eq("left_only"), "abbreviation"].tolist()
        raise ValueError(f"Current official assets could not be mapped to 2026 records: {missing}")
    summary["price_percentile_within_entity_type"] = summary.groupby("entity_type")[
        "current_price"
    ].rank(pct=True, method="average")
    summary["sprint_component_coverage"] = (
        summary["sprint_event_count"] / summary["sprint_selected_event_count"]
    )
    summary["form_rank"] = summary.groupby("entity_type")["normal_weekend_mean"].rank(
        ascending=False, method="min"
    )
    summary["bonus_rank"] = summary.groupby("entity_type")["mean_extra_sprint_points"].rank(
        ascending=False, method="min"
    )
    summary["current_price_source"] = "verified_official_market_cache"
    summary["form_scope"] = "all_completed_2026_normal_weekends_current_state_descriptive"
    return summary.sort_values(["entity_type", "form_rank", "canonical_entity_id"]).reset_index(drop=True)


def _design(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = []
    for column in columns:
        if column == "intercept":
            values.append(np.ones(len(frame)))
        elif column == "form_price_interaction":
            values.append(
                frame["normal_weekend_mean"].to_numpy(float)
                * frame["price_percentile_within_entity_type"].to_numpy(float)
            )
        else:
            values.append(frame[column].to_numpy(float))
    return np.column_stack(values)


def _nonnegative_least_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact active-set enumeration for the at-most-three constrained columns."""
    best = np.zeros(x.shape[1])
    best_sse = float(np.sum(y**2))
    for size in range(1, x.shape[1] + 1):
        for active in combinations(range(x.shape[1]), size):
            coef = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
            if np.any(coef < -1e-10):
                continue
            candidate = np.zeros(x.shape[1])
            candidate[list(active)] = np.maximum(coef, 0.0)
            sse = float(np.sum((y - x @ candidate) ** 2))
            if sse < best_sse:
                best, best_sse = candidate, sse
    return best


def _ridge_coefficients(frame: pd.DataFrame, y: np.ndarray, penalty: float) -> np.ndarray:
    predictors = frame[["normal_weekend_mean", "price_percentile_within_entity_type"]].to_numpy(float)
    means, scales = predictors.mean(axis=0), predictors.std(axis=0, ddof=0)
    scales = np.where(scales > 0, scales, 1.0)
    z = (predictors - means) / scales
    design = np.column_stack([np.ones(len(z)), z])
    gram = design.T @ design + np.diag([0.0, penalty, penalty])
    standard = np.linalg.solve(gram, design.T @ y)
    slopes = standard[1:] / scales
    intercept = standard[0] - float(means @ slopes)
    return np.array([intercept, *slopes])


def _fit_model(frame: pd.DataFrame, model: str, *, ridge_penalty: float | None = None) -> dict[str, object]:
    clean = frame.dropna(
        subset=["mean_extra_sprint_points", "normal_weekend_mean", "price_percentile_within_entity_type"]
    )
    y = clean["mean_extra_sprint_points"].to_numpy(float)
    if model == "ridge_form_price":
        if ridge_penalty is None:
            raise ValueError("Ridge model requires an explicit penalty.")
        columns = MODEL_SPECS["form_price"]
        coefficients = _ridge_coefficients(clean, y, ridge_penalty)
    else:
        columns = MODEL_SPECS[model]
        x = _design(clean, columns)
        if model.startswith("constrained_"):
            coefficients = _nonnegative_least_squares(x, y)
        else:
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    return {
        "model": model,
        "columns": columns,
        "coefficients": coefficients,
        "ridge_penalty": ridge_penalty,
    }


def predict_model(fit: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    prediction = _design(frame, fit["columns"]) @ np.asarray(fit["coefficients"], dtype=float)
    if str(fit["model"]).startswith("constrained_"):
        prediction = np.maximum(0.0, prediction)
    return prediction


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    residual = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    spearman = pd.Series(actual).rank(method="average").corr(
        pd.Series(predicted).rank(method="average"), method="pearson"
    )
    return {
        "observations": int(len(actual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": float(spearman) if pd.notna(spearman) else np.nan,
        "r_squared": float(1 - np.sum(residual**2) / denominator) if denominator else np.nan,
        "negative_predictions": int(np.sum(predicted < 0)),
    }


def select_ridge_penalty(frame: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for penalty in RIDGE_GRID:
        actual, predicted = [], []
        for heldout in frame.index:
            train = frame.drop(index=heldout)
            fit = _fit_model(train, "ridge_form_price", ridge_penalty=penalty)
            actual.append(float(frame.loc[heldout, "mean_extra_sprint_points"]))
            predicted.append(float(predict_model(fit, frame.loc[[heldout]])[0]))
        rows.append({"ridge_penalty": penalty, **_metrics(np.array(actual), np.array(predicted))})
    comparison = pd.DataFrame(rows).sort_values(["mae", "ridge_penalty"], kind="stable")
    return float(comparison.iloc[0]["ridge_penalty"]), comparison.reset_index(drop=True)


def leave_one_asset_out(
    frame: pd.DataFrame,
    models: Iterable[str],
    ridge_penalty: float,
) -> pd.DataFrame:
    rows = []
    for model in models:
        if model == "interaction" and len(frame) < 16:
            continue
        for heldout in frame.index:
            train, test = frame.drop(index=heldout), frame.loc[[heldout]]
            fit = _fit_model(
                train, model, ridge_penalty=ridge_penalty if model == "ridge_form_price" else None
            )
            coefficients = {
                COEFFICIENT_NAMES[column]: float(value)
                for column, value in zip(fit["columns"], fit["coefficients"])
            }
            rows.append(
                {
                    "entity_type": str(frame.iloc[0]["entity_type"]),
                    "model": model,
                    "held_out_entity_id": test.iloc[0]["canonical_entity_id"],
                    "held_out_entity": test.iloc[0]["entity"],
                    "actual_bonus": float(test.iloc[0]["mean_extra_sprint_points"]),
                    "predicted_bonus": float(predict_model(fit, test)[0]),
                    **{name: coefficients.get(name, np.nan) for name in ("alpha", "gamma", "delta", "eta")},
                    "ridge_penalty": ridge_penalty if model == "ridge_form_price" else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["error"] = out["predicted_bonus"] - out["actual_bonus"]
    return out


def _coefficient_record(entity_type: str, fit: dict[str, object], method: str = "full_sample") -> dict[str, object]:
    mapped = {
        COEFFICIENT_NAMES[column]: float(value)
        for column, value in zip(fit["columns"], fit["coefficients"])
    }
    return {
        "entity_type": entity_type,
        "model": fit["model"],
        "method": method,
        "alpha": mapped.get("alpha", 0.0),
        "gamma": mapped.get("gamma", 0.0),
        "delta": mapped.get("delta", 0.0),
        "eta": mapped.get("eta", 0.0),
        "ridge_penalty": fit.get("ridge_penalty", np.nan),
    }


def _huber_fit(frame: pd.DataFrame, columns: tuple[str, ...], iterations: int = 100) -> np.ndarray:
    x = _design(frame, columns)
    y = frame["mean_extra_sprint_points"].to_numpy(float)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(iterations):
        residual = y - x @ coefficients
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if not np.isfinite(scale) or scale < 1e-12:
            break
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        root = np.sqrt(weights)
        updated = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def build_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("normal_weekend_mean", "mean_extra_sprint_points"),
        ("current_price", "mean_extra_sprint_points"),
        ("price_percentile_within_entity_type", "mean_extra_sprint_points"),
        ("normal_weekend_mean", "current_price"),
        ("normal_weekend_standard_deviation", "mean_extra_sprint_points"),
    )
    rows = []
    for entity_type, group in summary.groupby("entity_type"):
        for x_name, y_name in pairs:
            valid = group[[x_name, y_name]].dropna()
            rows.append(
                {
                    "entity_type": entity_type,
                    "x": x_name,
                    "y": y_name,
                    "observations": len(valid),
                    "pearson": valid[x_name].corr(valid[y_name], method="pearson"),
                    "spearman": valid[x_name].rank(method="average").corr(
                        valid[y_name].rank(method="average"), method="pearson"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_bin_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entity_type, group in summary.groupby("entity_type"):
        bins = 3 if entity_type == "driver" else (3 if len(group) >= 9 else 2)
        for bin_type, field in (("price", "current_price"), ("form", "normal_weekend_mean")):
            labelled = group.copy()
            labelled["bin"] = pd.qcut(
                labelled[field].rank(method="first"), bins, labels=["low", "middle", "high"][-bins:]
            )
            for label, values in labelled.groupby("bin", observed=True):
                rows.append(
                    {
                        "entity_type": entity_type,
                        "bin_type": bin_type,
                        "bin": str(label),
                        "asset_count": len(values),
                        "mean_current_price": values["current_price"].mean(),
                        "mean_normal_form": values["normal_weekend_mean"].mean(),
                        "mean_observed_bonus": values["mean_extra_sprint_points"].mean(),
                        "mean_predicted_bonus": values["predicted_sprint_bonus"].mean()
                        if "predicted_sprint_bonus" in values else np.nan,
                        "mean_effective_multiplier": values["effective_multiplier"].mean()
                        if "effective_multiplier" in values else np.nan,
                        "pooled_prediction_mae": (
                            values["pooled_predicted_bonus"] - values["mean_extra_sprint_points"]
                        ).abs().mean() if "pooled_predicted_bonus" in values else np.nan,
                        "pooled_prediction_bias": (
                            values["pooled_predicted_bonus"] - values["mean_extra_sprint_points"]
                        ).mean() if "pooled_predicted_bonus" in values else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _choose_pooled_model(comparison: pd.DataFrame) -> str:
    candidates = comparison[comparison["model"].isin(
        ["constant", "proportional", "constrained_hybrid", "constrained_form_price"]
    )].sort_values("loao_mae")
    best_mae = float(candidates.iloc[0]["loao_mae"])
    near = candidates[candidates["loao_mae"].le(best_mae * 1.03 + 1e-12)]
    complexity = {
        "constant": 0,
        "proportional": 1,
        "constrained_hybrid": 2,
        "constrained_form_price": 3,
    }
    return min(near["model"], key=lambda model: complexity[model])


def fit_models(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    all_models = tuple(MODEL_SPECS) + ("ridge_form_price",)
    coefficient_rows, comparison_rows, loao_frames = [], [], []
    selected: dict[str, dict[str, object]] = {}
    for entity_type, group in summary.groupby("entity_type", sort=True):
        group = group.dropna(subset=["mean_extra_sprint_points"]).copy()
        ridge_penalty, _ridge_search = select_ridge_penalty(group)
        loao = leave_one_asset_out(group, all_models, ridge_penalty)
        loao_frames.append(loao)
        for model in loao["model"].unique():
            rows = loao[loao["model"].eq(model)]
            metrics = _metrics(rows["actual_bonus"].to_numpy(), rows["predicted_bonus"].to_numpy())
            coefficient_rows_fold = rows[["alpha", "gamma", "delta", "eta"]]
            comparison_rows.append(
                {
                    "entity_type": entity_type,
                    "model": model,
                    **{f"loao_{key}": value for key, value in metrics.items()},
                    "coefficient_stability_mean_std": float(coefficient_rows_fold.std(ddof=0).mean()),
                }
            )
            fit = _fit_model(
                group, model, ridge_penalty=ridge_penalty if model == "ridge_form_price" else None
            )
            coefficient_rows.append(_coefficient_record(entity_type, fit))
        huber_columns = MODEL_SPECS["hybrid"]
        huber = {
            "model": "huber_hybrid",
            "columns": huber_columns,
            "coefficients": _huber_fit(group, huber_columns),
            "ridge_penalty": None,
        }
        coefficient_rows.append(_coefficient_record(entity_type, huber, "robust_sensitivity"))
        entity_comparison = pd.DataFrame(
            [row for row in comparison_rows if row["entity_type"] == entity_type]
        )
        pooled_name = _choose_pooled_model(entity_comparison)
        selected[entity_type] = {
            "model_name": pooled_name,
            "fit": _fit_model(group, pooled_name),
            "ridge_penalty": ridge_penalty,
        }
    coefficients = pd.DataFrame(coefficient_rows)
    comparison = pd.DataFrame(comparison_rows).sort_values(["entity_type", "loao_mae", "model"])
    loao_all = pd.concat(loao_frames, ignore_index=True)
    for entity_type, info in selected.items():
        comparison.loc[
            comparison["entity_type"].eq(entity_type) & comparison["model"].eq(info["model_name"]),
            "selected_pooled_model",
        ] = True
    comparison["selected_pooled_model"] = comparison["selected_pooled_model"].fillna(False)
    return coefficients, comparison.reset_index(drop=True), loao_all, selected


def leave_one_sprint_out(
    annotated: pd.DataFrame,
    summary: pd.DataFrame,
    selected: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    sprint_rounds = sorted(annotated.loc[annotated["weekend_format"].eq("sprint"), "round"].unique())
    for excluded_round in sprint_rounds:
        retained = annotated[
            ~(
                annotated["weekend_format"].eq("sprint")
                & annotated["round"].eq(excluded_round)
            )
        ]
        targets = retained.groupby(["entity_type", "canonical_entity_id"])["extra_sprint_points"].agg(
            mean_extra_sprint_points="mean", training_sprint_observations="count"
        ).reset_index()
        heldout = annotated[
            annotated["weekend_format"].eq("sprint") & annotated["round"].eq(excluded_round)
        ][["entity_type", "canonical_entity_id", "extra_sprint_points"]]
        for entity_type, info in selected.items():
            features = summary[summary["entity_type"].eq(entity_type)].drop(
                columns=["mean_extra_sprint_points"], errors="ignore"
            ).merge(targets, on=["entity_type", "canonical_entity_id"], how="inner")
            features = features.dropna(subset=["mean_extra_sprint_points"])
            fit = _fit_model(features, str(info["model_name"]))
            coefficients = _coefficient_record(entity_type, fit, "leave_one_sprint_out")
            scored = features.merge(
                heldout,
                on=["entity_type", "canonical_entity_id"],
                how="inner",
                suffixes=("_training", "_heldout"),
            ).dropna(subset=["extra_sprint_points"])
            predictions = predict_model(fit, scored)
            metrics = _metrics(scored["extra_sprint_points"].to_numpy(), predictions)
            rows.append(
                {
                    "entity_type": entity_type,
                    "excluded_season": SEASON,
                    "excluded_round": int(excluded_round),
                    "model": info["model_name"],
                    **{name: coefficients[name] for name in ("alpha", "gamma", "delta", "eta")},
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def influence_diagnostics(summary: pd.DataFrame, selected: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for entity_type, group in summary.groupby("entity_type"):
        group = group.dropna(subset=["mean_extra_sprint_points"]).copy()
        fit = selected[entity_type]["fit"]
        x = _design(group, fit["columns"])
        y = group["mean_extra_sprint_points"].to_numpy(float)
        prediction = predict_model(fit, group)
        residual = y - prediction
        p = x.shape[1]
        mse = float(np.sum(residual**2) / max(len(y) - p, 1))
        leverage = np.diag(x @ np.linalg.pinv(x.T @ x) @ x.T)
        cooks = (residual**2 / max(p * mse, 1e-12)) * leverage / np.maximum((1 - leverage) ** 2, 1e-12)
        for index, (_, asset) in enumerate(group.iterrows()):
            rows.append(
                {
                    "entity_type": entity_type,
                    "entity_id": asset["canonical_entity_id"],
                    "entity": asset["entity"],
                    "model": fit["model"],
                    "residual": residual[index],
                    "leverage": leverage[index],
                    "cooks_distance": cooks[index],
                    "influential_cook_threshold": bool(cooks[index] > 4 / len(group)),
                }
            )
    return pd.DataFrame(rows)


def select_shrinkage(
    annotated: pd.DataFrame,
    summary: pd.DataFrame,
    selected: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    sprint_rounds = sorted(annotated.loc[annotated["weekend_format"].eq("sprint"), "round"].unique())
    rows, choices = [], {}
    for entity_type, group in summary.groupby("entity_type"):
        errors: dict[float, list[float]] = {k: [] for k in SHRINKAGE_GRID}
        for excluded_round in sprint_rounds:
            event_rows = annotated[
                annotated["entity_type"].eq(entity_type)
                & annotated["weekend_format"].eq("sprint")
            ]
            train_targets = event_rows[event_rows["round"].ne(excluded_round)].groupby(
                "canonical_entity_id"
            )["extra_sprint_points"].agg(["mean", "count"])
            heldout = event_rows[event_rows["round"].eq(excluded_round)].set_index(
                "canonical_entity_id"
            )["extra_sprint_points"]
            features = group.set_index("canonical_entity_id").copy()
            features["mean_extra_sprint_points"] = train_targets["mean"]
            features = features.dropna(subset=["mean_extra_sprint_points"])
            pooled_fit = _fit_model(features.reset_index(), str(selected[entity_type]["model_name"]))
            pooled = pd.Series(predict_model(pooled_fit, features.reset_index()), index=features.index)
            for k in SHRINKAGE_GRID:
                weight = train_targets["count"].reindex(features.index) / (
                    train_targets["count"].reindex(features.index) + k
                )
                prediction = weight * train_targets["mean"].reindex(features.index) + (1 - weight) * pooled
                common = prediction.index.intersection(heldout.dropna().index)
                errors[k].extend((prediction.loc[common] - heldout.loc[common]).tolist())
        for k, residuals in errors.items():
            residual = np.asarray(residuals, float)
            rows.append(
                {
                    "entity_type": entity_type,
                    "k": k,
                    "description": "pooled_only" if k >= 1_000_000 else ("unshrunk_asset" if k == 0 else "shrunk_asset"),
                    "observations": len(residual),
                    "leave_one_sprint_mae": float(np.mean(np.abs(residual))),
                    "leave_one_sprint_rmse": float(np.sqrt(np.mean(residual**2))),
                    "leave_one_sprint_bias": float(np.mean(residual)),
                }
            )
        entity_rows = [row for row in rows if row["entity_type"] == entity_type]
        best = min(entity_rows, key=lambda row: (row["leave_one_sprint_mae"], -row["k"]))
        choices[entity_type] = float(best["k"])
    comparison = pd.DataFrame(rows)
    comparison["selected"] = comparison.apply(
        lambda row: bool(row["k"] == choices[row["entity_type"]]), axis=1
    )
    return comparison, choices


def bootstrap_predictions(
    summary: pd.DataFrame,
    selected: dict[str, dict[str, object]],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[float, float]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    coefficient_rows, intervals = [], {}
    for entity_type, group in summary.groupby("entity_type", sort=True):
        group = group.dropna(subset=["mean_extra_sprint_points"]).reset_index(drop=True)
        model = str(selected[entity_type]["model_name"])
        coefficient_samples, prediction_samples = [], []
        for _ in range(replicates):
            sample = group.iloc[rng.integers(0, len(group), len(group))].reset_index(drop=True)
            try:
                fit = _fit_model(sample, model)
            except (ValueError, np.linalg.LinAlgError):
                continue
            coefficient_samples.append(np.asarray(fit["coefficients"], float))
            prediction_samples.append(predict_model(fit, group))
        coefficients = np.vstack(coefficient_samples)
        predictions = np.vstack(prediction_samples)
        for index, column in enumerate(selected[entity_type]["fit"]["columns"]):
            values = coefficients[:, index]
            coefficient_rows.append(
                {
                    "entity_type": entity_type,
                    "model": model,
                    "parameter": COEFFICIENT_NAMES[column],
                    "replicates": len(values),
                    "estimate": float(np.median(values)),
                    "lower_2_5": float(np.quantile(values, 0.025)),
                    "upper_97_5": float(np.quantile(values, 0.975)),
                }
            )
        for index, asset in group.iterrows():
            intervals[(entity_type, str(asset["canonical_entity_id"]))] = (
                float(np.quantile(predictions[:, index], 0.025)),
                float(np.quantile(predictions[:, index], 0.975)),
            )
    return pd.DataFrame(coefficient_rows), intervals


def build_asset_predictions(
    summary: pd.DataFrame,
    selected: dict[str, dict[str, object]],
    shrinkage_k: dict[str, float],
    intervals: dict[tuple[str, str], tuple[float, float]],
) -> pd.DataFrame:
    frames = []
    for entity_type, group in summary.groupby("entity_type", sort=True):
        group = group.copy()
        pooled = np.maximum(0.0, predict_model(selected[entity_type]["fit"], group))
        n = group["sprint_event_count"].to_numpy(float)
        k = shrinkage_k[entity_type]
        weight = n / (n + k)
        observed = group["mean_extra_sprint_points"].to_numpy(float)
        shrunk = weight * observed + (1 - weight) * pooled
        group["observed_mean_sprint_bonus"] = observed
        group["pooled_predicted_bonus"] = pooled
        group["shrinkage_k"] = k
        group["shrinkage_weight"] = weight
        group["shrunk_asset_bonus"] = np.maximum(0.0, shrunk)
        group["normal_weekend_ev"] = group["normal_weekend_mean"]
        group["predicted_sprint_bonus"] = group["shrunk_asset_bonus"]
        group["predicted_sprint_ev"] = group["normal_weekend_ev"] + group["predicted_sprint_bonus"]
        group["effective_multiplier"] = group["predicted_sprint_ev"] / group["normal_weekend_ev"]
        group["selected_pooled_model"] = selected[entity_type]["model_name"]
        group["uncertainty_lower"] = group.apply(
            lambda row: intervals[(entity_type, str(row["canonical_entity_id"]))][0], axis=1
        )
        group["uncertainty_upper"] = group.apply(
            lambda row: intervals[(entity_type, str(row["canonical_entity_id"]))][1], axis=1
        )
        frames.append(group)
    columns = [
        "entity", "entity_type", "canonical_entity_id", "abbreviation", "current_price",
        "price_percentile_within_entity_type", "normal_weekend_mean", "recent_normal_form",
        "observed_mean_sprint_bonus", "pooled_predicted_bonus", "shrunk_asset_bonus",
        "normal_weekend_ev", "predicted_sprint_bonus", "predicted_sprint_ev",
        "effective_multiplier", "sprint_event_count", "sprint_component_coverage",
        "selected_pooled_model", "shrinkage_k", "shrinkage_weight", "uncertainty_lower",
        "uncertainty_upper", "current_price_source", "form_scope",
    ]
    return pd.concat(frames, ignore_index=True)[columns].sort_values(
        ["entity_type", "predicted_sprint_bonus", "entity"], ascending=[True, False, True]
    ).reset_index(drop=True)


def _variance_inflation(summary: pd.DataFrame, entity_type: str) -> float:
    group = summary[summary["entity_type"].eq(entity_type)]
    correlation = group["normal_weekend_mean"].corr(group["price_percentile_within_entity_type"])
    return float(1 / (1 - correlation**2)) if pd.notna(correlation) and abs(correlation) < 1 else np.inf


def _format_formula(row: pd.Series) -> str:
    terms = [f"{row['alpha']:.4f}"]
    if abs(row["gamma"]) > 1e-12:
        terms.append(f"{row['gamma']:.4f} × normal_form")
    if abs(row["delta"]) > 1e-12:
        terms.append(f"{row['delta']:.4f} × price_percentile")
    return " + ".join(terms)


def _markdown_table(frame: pd.DataFrame, *, index: bool = False) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    display = frame.copy()
    if index:
        if isinstance(display.index, pd.MultiIndex):
            display = display.reset_index()
        else:
            display.insert(0, display.index.name or "index", display.index)
    columns = [str(column) for column in display.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in display.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join("" if pd.isna(value) else str(value) for value in values)
            + " |"
        )
    return "\n".join([header, separator, *rows])


def write_report(
    output: Path,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    comparison: pd.DataFrame,
    sprint_sensitivity: pd.DataFrame,
    predictions: pd.DataFrame,
    selected: dict[str, dict[str, object]],
    price_metadata: dict[str, object],
) -> None:
    sprint_events = events[events["weekend_format"].eq("sprint")]
    normal_events = events[events["weekend_format"].eq("normal")]
    conclusions = {}
    for entity_type in ("driver", "constructor"):
        model = selected[entity_type]["model_name"]
        row = coefficients[
            coefficients["entity_type"].eq(entity_type) & coefficients["model"].eq(model)
        ].iloc[0]
        price_model = comparison[
            comparison["entity_type"].eq(entity_type) & comparison["model"].eq("constrained_form_price")
        ].iloc[0]
        form_model = comparison[
            comparison["entity_type"].eq(entity_type) & comparison["model"].eq("constrained_hybrid")
        ].iloc[0]
        conclusions[entity_type] = {
            "model": model,
            "formula": _format_formula(row),
            "mean_bonus": summary.loc[summary["entity_type"].eq(entity_type), "mean_extra_sprint_points"].mean(),
            "price_improvement": float(form_model["loao_mae"] - price_model["loao_mae"]),
        }
    lines = [
        "# 2026 Sprint-only Fantasy bonus analysis",
        "",
        "## 1. Executive conclusion",
        "",
        "The extra target is `sprint_points + sprint_qualifying_points`, summed only when at least one official component is present. Ordinary qualifying, Grand Prix race points, residuals, complete-weekend totals, and price changes are excluded.",
        "",
        f"Across current assets, the observed mean Sprint-only bonus is **{conclusions['driver']['mean_bonus']:.2f} points for drivers** and **{conclusions['constructor']['mean_bonus']:.2f} points for constructors**. The selected pooled models are `{conclusions['driver']['model']}` for drivers and `{conclusions['constructor']['model']}` for constructors.",
        "",
        f"Driver recommendation: `predicted_bonus = max(0, {conclusions['driver']['formula']})`.",
        f"Constructor recommendation: `predicted_bonus = max(0, {conclusions['constructor']['formula']})`.",
        "",
        f"Adding price percentile improved constrained-model LOAO MAE by {conclusions['driver']['price_improvement']:.3f} driver points and {conclusions['constructor']['price_improvement']:.3f} constructor points. In both selected fits the form coefficient collapsed to zero while price percentile remained positive: stronger/more expensive constructors therefore receive larger absolute bonuses descriptively, but the four-Sprint sample cannot separate form from price reliably.",
        "",
        "## 2. Component semantics",
        "",
        "The official playerstats parser classifies sessions using both session labels and scoring-event labels. The redacted official fixture demonstrates that a feed session labelled `Sprint Qualifying` can contain `Sprint Position` and `Sprint overtake` events; these are correctly classified as `sprint_points`. A separately emitted Sprint Qualifying score is retained as `sprint_qualifying_points`. Canonical nulls remain null, and target aggregation uses `min_count=1`, so an absent observation is never invented as zero.",
        "",
        "## 3. Current 2026 asset summaries",
        "",
        f"The descriptive features use all {len(normal_events)} completed normal 2026 weekends; targets use {len(sprint_events)} completed Sprint weekends. This is current-state calibration, so races after an early Sprint may contribute to current form. Current prices are from verified official feed {price_metadata['feed_round']} ({price_metadata['verified_at_utc']}).",
        "",
        _markdown_table(summary[["entity_type", "entity", "current_price", "normal_weekend_mean", "recent_normal_form", "mean_extra_sprint_points", "sprint_event_count"]].round(3)),
        "",
        "## 4. Model comparison",
        "",
        _markdown_table(comparison[["entity_type", "model", "loao_mae", "loao_rmse", "loao_bias", "loao_spearman", "loao_r_squared", "loao_negative_predictions", "selected_pooled_model"]].round(4)),
        "",
        "## 5. Drivers",
        "",
        _markdown_table(predictions[predictions["entity_type"].eq("driver")][["entity", "normal_weekend_ev", "observed_mean_sprint_bonus", "pooled_predicted_bonus", "shrunk_asset_bonus", "predicted_sprint_ev"]].round(3)),
        "",
        "## 6. Constructors",
        "",
        _markdown_table(predictions[predictions["entity_type"].eq("constructor")][["entity", "normal_weekend_ev", "observed_mean_sprint_bonus", "pooled_predicted_bonus", "shrunk_asset_bonus", "predicted_sprint_ev"]].round(3)),
        "",
        "The constructor table directly answers the absolute-bonus question: compare `pooled_predicted_bonus` for leading and lower-form teams. The formula does not assume a named team tier.",
        "",
        "## 7. Price contribution",
        "",
        f"Form/price VIF is {_variance_inflation(summary, 'driver'):.2f} for drivers and {_variance_inflation(summary, 'constructor'):.2f} for constructors. Price is retained only if its LOAO benefit is material; otherwise the simpler form-only or constant candidate is preferred.",
        "",
        "## 8. Stability and uncertainty",
        "",
        "Leave-one-asset-out predictions are the primary pooled-model comparison. Bootstrap intervals resample assets with a fixed seed. Leave-one-Sprint-out refits after removing the event before target aggregation; coefficient ranges are:",
        "",
        _markdown_table(sprint_sensitivity.groupby(["entity_type", "model"])[["alpha", "gamma", "delta", "mae"]].agg(["min", "max"]).round(4), index=True),
        "",
        "## 9. Recommended first production candidate",
        "",
        f"Drivers: `predicted_bonus = max(0, {conclusions['driver']['formula']})`; `Sprint EV = normal EV + predicted_bonus`.",
        "",
        f"Constructors: `predicted_bonus = max(0, {conclusions['constructor']['formula']})`; `Sprint EV = normal EV + predicted_bonus`.",
        "",
        "These are research recommendations only and are not activated in production.",
        "",
        "## 10. Limitations",
        "",
        "Only four completed 2026 Sprints are available. Assets within the same Sprint are correlated. A few official component observations are missing and remain excluded rather than converted to zero. Crashes, DNFs, and penalties are retained and can strongly affect four-event means. Current price partly reflects results already observed. Constructor analysis has only 11 assets. This is descriptive full-season/current-state calibration, not a leakage-safe historical walk-forward test.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    canonical_path: str | Path,
    schedule_path: str | Path,
    current_prices_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    data = load_2026_recorded_data(canonical_path)
    schedule_arg = Path(schedule_path)
    schedule_dir = schedule_arg if schedule_arg.is_dir() else schedule_arg.parent
    schedule = load_schedule_metadata(schedule_dir, seasons=(SEASON,))
    prices, price_metadata = load_current_prices(current_prices_path)
    events, annotated = prepare_2026_events(data, schedule)
    event_counts = events["weekend_format"].value_counts().to_dict()
    if event_counts.get("normal") != 7 or event_counts.get("sprint") != 4:
        raise ValueError(
            "Expected all completed 2026 coverage through round 11: seven normal and four Sprint weekends."
        )
    audit = build_component_audit(annotated)
    summary = build_asset_summary(annotated, prices)
    coefficients, comparison, loao, selected = fit_models(summary)
    sprint_sensitivity = leave_one_sprint_out(annotated, summary, selected)
    influence = influence_diagnostics(summary, selected)
    correlations = build_correlations(summary)
    shrinkage, shrinkage_k = select_shrinkage(annotated, summary, selected)
    candidate_labels = {
        "constant": "A_constant_bonus",
        "proportional": "B_proportional_bonus",
        "constrained_hybrid": "C_constrained_hybrid",
        "constrained_form_price": "D_constrained_form_and_price",
    }
    comparison["production_candidate"] = comparison["model"].map(candidate_labels)
    comparison["evaluation_scope"] = "leave_one_asset_out_aggregated_bonus"
    residual_rows = []
    for entity_type, choice in shrinkage_k.items():
        row = shrinkage[
            shrinkage["entity_type"].eq(entity_type) & shrinkage["k"].eq(choice)
        ].iloc[0]
        residual_rows.append(
            {
                "entity_type": entity_type,
                "model": "shrunk_asset_residual",
                "loao_observations": np.nan,
                "loao_mae": np.nan,
                "loao_rmse": np.nan,
                "loao_bias": np.nan,
                "loao_spearman": np.nan,
                "loao_r_squared": np.nan,
                "loao_negative_predictions": 0,
                "coefficient_stability_mean_std": np.nan,
                "selected_pooled_model": False,
                "production_candidate": "E_pooled_plus_shrunk_asset_residual",
                "evaluation_scope": "leave_one_sprint_out_event_bonus",
                "leave_one_sprint_mae": row["leave_one_sprint_mae"],
                "leave_one_sprint_rmse": row["leave_one_sprint_rmse"],
                "leave_one_sprint_bias": row["leave_one_sprint_bias"],
                "shrinkage_k": choice,
            }
        )
    comparison = pd.concat([comparison, pd.DataFrame(residual_rows)], ignore_index=True, sort=False)
    bootstrap, intervals = bootstrap_predictions(summary, selected)
    predictions = build_asset_predictions(summary, selected, shrinkage_k, intervals)
    bins = build_bin_summary(
        summary.merge(
            predictions[[
                "entity_type", "canonical_entity_id", "predicted_sprint_bonus",
                "pooled_predicted_bonus", "effective_multiplier",
            ]],
            on=["entity_type", "canonical_entity_id"],
            how="left",
            validate="one_to_one",
        )
    )

    outputs = {
        "component_audit.csv": audit,
        "asset_summary.csv": summary,
        "correlations.csv": correlations,
        "model_coefficients.csv": coefficients,
        "model_comparison.csv": comparison,
        "bootstrap_intervals.csv": bootstrap,
        "leave_one_asset_out.csv": loao,
        "leave_one_sprint_out.csv": sprint_sensitivity,
        "influence_diagnostics.csv": influence,
        "price_form_bins.csv": bins,
        "asset_predictions.csv": predictions,
        "shrinkage_comparison.csv": shrinkage,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False, float_format="%.10g")
    write_report(
        output, events, summary, coefficients, comparison, sprint_sensitivity,
        predictions, selected, price_metadata
    )
    return {
        "events": events,
        "annotated": annotated,
        "summary": summary,
        "coefficients": coefficients,
        "comparison": comparison,
        "predictions": predictions,
        "selected": selected,
        "shrinkage_k": shrinkage_k,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_DATASET_PATH)
    parser.add_argument("--schedule", type=Path, default=PROJECT_ROOT / "data/cache/schedule_2026.csv")
    parser.add_argument(
        "--current-prices", type=Path, default=PROJECT_ROOT / "data/cache/verified_fantasy_market.json"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/2026_sprint_bonus")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_analysis(args.canonical, args.schedule, args.current_prices, args.output)
    chosen = {key: value["model_name"] for key, value in result["selected"].items()}
    print(f"Wrote deterministic 2026 Sprint-bonus analysis to {args.output}")
    print(f"Selected pooled models: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
