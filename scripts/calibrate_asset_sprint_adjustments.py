#!/usr/bin/env python3
"""Calibrate research-only personalised 2026 Sprint adjustments offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH
from scripts.analyse_2026_sprint_bonus import (
    MIN_COMPONENT_COVERAGE,
    _markdown_table,
    load_2026_recorded_data,
    load_current_prices,
    prepare_2026_events,
)
from scripts.analyse_sprint_multiplier import load_schedule_metadata


SEASON = 2026
CALIBRATION_MAX_ROUND = 11
SCRIPT_VERSION = "2026-personalised-sprint-adjustments-v1"
DEFAULT_RECENCY_DECAY = 0.80
RANDOM_SEED = 20260806
SHRINKAGE_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
NEAR_ZERO_VARIANCE = 1e-8
ILL_CONDITIONED_THRESHOLD = 1_000.0
EXTREME_SLOPE_THRESHOLD = 1.0
HIGH_RESIDUAL_ERROR_THRESHOLD = {"driver": 8.0, "constructor": 16.0}


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def _weighted_mean(values: pd.DataFrame, value_column: str, decay: float) -> float:
    if not 0 <= decay <= 1:
        raise ValueError("Recency decay must be within [0, 1].")
    ordering = [column for column in ("event_date", "round") if column in values]
    ordered = values.dropna(subset=[value_column])
    if ordering:
        ordered = ordered.sort_values(ordering, ascending=False)
    if ordered.empty:
        return np.nan
    weights = np.power(decay, np.arange(len(ordered), dtype=float))
    if weights.sum() == 0:
        weights[0] = 1.0
    return float(np.average(ordered[value_column].to_numpy(float), weights=weights))


def prepare_calibration_data(
    canonical_path: str | Path,
    schedule_path: str | Path,
    current_prices_path: str | Path,
    *,
    maximum_round: int | None = CALIBRATION_MAX_ROUND,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load verified history, optionally retaining the original research cutoff."""
    data = load_2026_recorded_data(canonical_path)
    if maximum_round is None:
        maximum_round = int(data["round"].max())
    data = data[data["round"].le(maximum_round)].copy()
    if data.empty or int(data["round"].max()) != maximum_round:
        raise ValueError(f"Completed official 2026 coverage through round {maximum_round} is required.")
    schedule_arg = Path(schedule_path)
    schedule = load_schedule_metadata(
        schedule_arg if schedule_arg.is_dir() else schedule_arg.parent,
        seasons=(SEASON,),
    )
    if maximum_round > CALIBRATION_MAX_ROUND:
        expected_counts = data[data["round"].le(CALIBRATION_MAX_ROUND)].groupby(
            ["round", "entity_type"]
        ).size().groupby("entity_type").max()
        for round_no in range(CALIBRATION_MAX_ROUND + 1, maximum_round + 1):
            latest = data[data["round"].eq(round_no)]
            for kind, count in expected_counts.items():
                group = latest[latest["entity_type"].eq(kind)]
                if len(group) < count or group["race_points"].notna().mean() < MIN_COMPONENT_COVERAGE:
                    raise ValueError(f"Incomplete completed-weekend coverage for {round_no} {kind}.")
    prices, market_metadata = load_current_prices(current_prices_path)
    events, annotated = prepare_2026_events(
        data,
        schedule,
        minimum_component_coverage=MIN_COMPONENT_COVERAGE,
    )
    if set(events["round"]) != set(range(1, maximum_round + 1)):
        raise ValueError(f"Calibration requires completed canonical rounds 1–{maximum_round} with no gaps.")
    # Inactive drivers still carry valid historical observations. A current-market
    # inner join must not silently discard them after a roster change.
    missing = annotated.sort_values("round").drop_duplicates(
        ["entity_type", "abbreviation"], keep="last"
    )
    missing = missing.merge(prices[["entity_type", "abbreviation"]],
                            on=["entity_type", "abbreviation"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")]
    if not missing.empty:
        retained = missing.rename(columns={"name": "entity", "price": "current_price"})
        if retained["current_price"].isna().any():
            raise ValueError("Historical assets missing from the market require a recorded price.")
        prices = pd.concat([prices, retained[prices.columns]], ignore_index=True)
    dates = pd.to_datetime(events["event_date"], utc=True, errors="coerce")
    if dates.isna().any() or dates.ge(pd.Timestamp.now(tz="UTC").normalize()).any():
        raise ValueError("Calibration requires past completed event dates.")
    return events, annotated, prices, market_metadata


def build_sprint_observations(
    annotated: pd.DataFrame,
    current_prices: pd.DataFrame,
    *, maximum_round: int | None = CALIBRATION_MAX_ROUND,
) -> pd.DataFrame:
    """Create the transparent event-level regression table."""
    sprint = annotated[
        annotated["season"].eq(SEASON)
        & annotated["round"].le(maximum_round if maximum_round is not None else float("inf"))
        & annotated["weekend_format"].eq("sprint")
    ].copy(deep=True)
    sprint["base_weekend_points"] = (
        sprint["fantasy_points_total"] - sprint["extra_sprint_points"]
    )
    sprint = sprint.merge(
        current_prices[["entity_type", "abbreviation", "current_price", "entity"]],
        on=["entity_type", "abbreviation"],
        how="inner",
        validate="many_to_one",
    )
    sprint["included_in_regression"] = (
        sprint["extra_sprint_points"].notna() & sprint["base_weekend_points"].notna()
    )
    sprint["exclusion_reason"] = ""
    sprint.loc[sprint["extra_sprint_points"].isna(), "exclusion_reason"] = (
        "both_sprint_components_missing"
    )
    sprint.loc[
        sprint["extra_sprint_points"].notna() & sprint["base_weekend_points"].isna(),
        "exclusion_reason",
    ] = "total_or_base_missing"
    output = sprint.rename(
        columns={
            "canonical_entity_id": "entity_id",
            "entity": "entity_name",
            "fantasy_points_total": "total_fantasy_points",
            "source_name": "data_source",
        }
    )
    columns = [
        "season", "round", "event_name", "event_date", "entity_type", "entity_id", "entity_name",
        "total_fantasy_points", "sprint_points", "sprint_qualifying_points",
        "extra_sprint_points", "base_weekend_points", "current_price", "data_source",
        "included_in_regression", "exclusion_reason",
    ]
    return output[columns].sort_values(
        ["entity_type", "entity_id", "round"], kind="stable"
    ).reset_index(drop=True)


def build_normalised_history(
    annotated: pd.DataFrame,
    current_prices: pd.DataFrame,
    *, maximum_round: int | None = CALIBRATION_MAX_ROUND,
) -> pd.DataFrame:
    """Remove Sprint-only components from Sprint weekends; retain normal totals."""
    history = annotated[
        annotated["season"].eq(SEASON) & annotated["round"].le(maximum_round if maximum_round is not None else float("inf"))
    ].copy(deep=True)
    history["normalised_score"] = history["fantasy_points_total"]
    sprint = history["weekend_format"].eq("sprint")
    history.loc[sprint, "normalised_score"] = (
        history.loc[sprint, "fantasy_points_total"]
        - history.loc[sprint, "extra_sprint_points"]
    )
    history = history.merge(
        current_prices[["entity_type", "abbreviation", "current_price", "entity"]],
        on=["entity_type", "abbreviation"],
        how="inner",
        validate="many_to_one",
    )
    return history.sort_values(
        ["entity_type", "canonical_entity_id", "round"], kind="stable"
    ).reset_index(drop=True)


def build_baselines(history: pd.DataFrame, recency_decay: float) -> pd.DataFrame:
    rows = []
    keys = ["entity_type", "canonical_entity_id"]
    for (entity_type, entity_id), group in history.groupby(keys, sort=True):
        normal = group[group["weekend_format"].eq("normal")]
        all_valid = group.dropna(subset=["normalised_score"])
        rows.append(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_name": group.iloc[-1]["entity"],
                "abbreviation": group.iloc[-1]["abbreviation"],
                "current_price": float(group.iloc[-1]["current_price"]),
                "completed_2026_events": int(group["fantasy_points_total"].notna().sum()),
                "normalised_event_count": int(len(all_valid)),
                "current_normal_baseline": float(all_valid["normalised_score"].mean()),
                "normal_weekend_only_mean": float(normal["fantasy_points_total"].mean()),
                "median_normalised_baseline": float(all_valid["normalised_score"].median()),
                "recency_weighted_normal_baseline": _weighted_mean(
                    all_valid, "normalised_score", recency_decay
                ),
                "recency_weighted_normal_weekend_only_mean": _weighted_mean(
                    normal, "fantasy_points_total", recency_decay
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["entity_type", "entity_id"]).reset_index(drop=True)


def fit_linear_regression(
    observations: pd.DataFrame,
    *,
    entity_type: str,
) -> dict[str, object]:
    """Fit y=alpha+beta*x with transparent small-sample diagnostics."""
    valid = observations[
        observations["included_in_regression"].astype(bool)
    ].dropna(subset=["base_weekend_points", "extra_sprint_points"])
    n = int(len(valid))
    x = valid["base_weekend_points"].to_numpy(float)
    y = valid["extra_sprint_points"].to_numpy(float)
    variance = float(np.var(x, ddof=1)) if n > 1 else np.nan
    result: dict[str, object] = {
        "sprint_observation_count": n,
        "base_points_variance": variance,
        "alpha": np.nan,
        "beta": np.nan,
        "alpha_standard_error": np.nan,
        "beta_standard_error": np.nan,
        "r_squared": np.nan,
        "adjusted_r_squared": np.nan,
        "condition_number": np.nan,
        "residual_standard_error": np.nan,
        "identifiable": False,
        "reliability_flag": "insufficient_observations",
    }
    if n < 3:
        return result
    if not np.isfinite(variance) or variance <= NEAR_ZERO_VARIANCE:
        result["reliability_flag"] = "near_zero_predictor_variance"
        return result
    design = np.column_stack([np.ones(n), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    residual = y - predicted
    degrees = n - 2
    sse = float(residual @ residual)
    residual_variance = sse / degrees if degrees > 0 else np.nan
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - sse / total if total > 0 else np.nan
    adjusted = 1 - (1 - r_squared) * (n - 1) / degrees if degrees > 0 and np.isfinite(r_squared) else np.nan
    condition = float(np.linalg.cond(design))
    residual_error = float(np.sqrt(residual_variance)) if np.isfinite(residual_variance) else np.nan
    alpha, beta = map(float, coefficients)
    flag = "usable_but_uncertain"
    if condition > ILL_CONDITIONED_THRESHOLD:
        flag = "ill_conditioned"
    elif abs(beta) > EXTREME_SLOPE_THRESHOLD:
        flag = "extreme_slope"
    elif residual_error > HIGH_RESIDUAL_ERROR_THRESHOLD[entity_type]:
        flag = "high_residual_error"
    result.update(
        {
            "alpha": alpha,
            "beta": beta,
            "alpha_standard_error": float(standard_errors[0]),
            "beta_standard_error": float(standard_errors[1]),
            "r_squared": float(r_squared) if np.isfinite(r_squared) else np.nan,
            "adjusted_r_squared": float(adjusted) if np.isfinite(adjusted) else np.nan,
            "condition_number": condition,
            "residual_standard_error": residual_error,
            "identifiable": True,
            "reliability_flag": flag,
        }
    )
    return result


def fit_group_regressions(observations: pd.DataFrame) -> dict[str, dict[str, object]]:
    fits = {}
    for entity_type, group in observations.groupby("entity_type", sort=True):
        fits[entity_type] = fit_linear_regression(group, entity_type=entity_type)
    return fits


def calibrate_assets(
    observations: pd.DataFrame,
    baselines: pd.DataFrame,
    group_fits: dict[str, dict[str, object]],
    *,
    recency_decay: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows, sensitivity_rows, shrinkage_rows = [], [], []
    baseline_columns = (
        "current_normal_baseline",
        "normal_weekend_only_mean",
        "median_normalised_baseline",
        "recency_weighted_normal_baseline",
        "recency_weighted_normal_weekend_only_mean",
    )
    for asset in baselines.itertuples(index=False):
        asset_observations = observations[
            observations["entity_type"].eq(asset.entity_type)
            & observations["entity_id"].eq(asset.entity_id)
        ]
        fit = fit_linear_regression(asset_observations, entity_type=asset.entity_type)
        valid_bonus = asset_observations.loc[
            asset_observations["included_in_regression"], "extra_sprint_points"
        ].dropna()
        bonus_history = asset_observations[
            asset_observations["included_in_regression"]
        ].rename(columns={"extra_sprint_points": "bonus"})
        mean_bonus = float(valid_bonus.mean()) if len(valid_bonus) else np.nan
        median_bonus = float(valid_bonus.median()) if len(valid_bonus) else np.nan
        recent_bonus = _weighted_mean(bonus_history, "bonus", recency_decay)
        group = group_fits[asset.entity_type]
        baseline = float(asset.current_normal_baseline)
        raw_adjustment = (
            float(fit["alpha"]) + float(fit["beta"]) * baseline
            if fit["identifiable"] else np.nan
        )
        group_adjustment = float(group["alpha"]) + float(group["beta"]) * baseline
        row = {
            **asset._asdict(),
            "valid_sprint_observations": int(fit["sprint_observation_count"]),
            "mean_observed_sprint_bonus": mean_bonus,
            "median_observed_sprint_bonus": median_bonus,
            "recency_weighted_observed_sprint_bonus": recent_bonus,
            "minimum_observed_sprint_bonus": float(valid_bonus.min()) if len(valid_bonus) else np.nan,
            "maximum_observed_sprint_bonus": float(valid_bonus.max()) if len(valid_bonus) else np.nan,
            "sprint_bonus_standard_deviation": float(valid_bonus.std(ddof=1)) if len(valid_bonus) > 1 else np.nan,
            "raw_alpha": fit["alpha"],
            "raw_beta": fit["beta"],
            "alpha_standard_error": fit["alpha_standard_error"],
            "beta_standard_error": fit["beta_standard_error"],
            "raw_regression_adjustment": raw_adjustment,
            "raw_candidate_normal_ev": baseline,
            "raw_candidate_sprint_ev": baseline + raw_adjustment if np.isfinite(raw_adjustment) else np.nan,
            "group_alpha": group["alpha"],
            "group_beta": group["beta"],
            "group_adjustment": group_adjustment,
            "group_candidate_sprint_ev": baseline + group_adjustment,
            "mean_bonus_candidate_sprint_ev": baseline + mean_bonus,
            "median_bonus_candidate_sprint_ev": baseline + median_bonus,
            "recency_bonus_candidate_sprint_ev": baseline + recent_bonus,
            "base_points_variance": fit["base_points_variance"],
            "condition_number": fit["condition_number"],
            "r_squared": fit["r_squared"],
            "adjusted_r_squared": fit["adjusted_r_squared"],
            "residual_standard_error": fit["residual_standard_error"],
            "identifiable": fit["identifiable"],
            "reliability_flag": fit["reliability_flag"],
        }
        for weight in SHRINKAGE_WEIGHTS:
            if fit["identifiable"]:
                alpha = weight * float(fit["alpha"]) + (1 - weight) * float(group["alpha"])
                beta = weight * float(fit["beta"]) + (1 - weight) * float(group["beta"])
                adjustment = alpha + beta * baseline
            elif weight < 1:
                alpha, beta = float(group["alpha"]), float(group["beta"])
                adjustment = alpha + beta * baseline
            else:
                alpha = beta = adjustment = np.nan
            shrinkage_rows.append(
                {
                    "entity_type": asset.entity_type,
                    "entity_id": asset.entity_id,
                    "entity_name": asset.entity_name,
                    "weight": weight,
                    "shrunk_alpha": alpha,
                    "shrunk_beta": beta,
                    "baseline": baseline,
                    "shrunk_adjustment": adjustment,
                    "shrunk_candidate_sprint_ev": baseline + adjustment if np.isfinite(adjustment) else np.nan,
                }
            )
            if weight in (0.25, 0.50, 0.75):
                row[f"shrunk_adjustment_w{int(weight * 100)}"] = adjustment
        coefficient_rows.append(row)
        for baseline_name in baseline_columns:
            baseline_value = float(getattr(asset, baseline_name))
            adjustment = (
                float(fit["alpha"]) + float(fit["beta"]) * baseline_value
                if fit["identifiable"] else np.nan
            )
            sensitivity_rows.append(
                {
                    "entity_type": asset.entity_type,
                    "entity_id": asset.entity_id,
                    "entity_name": asset.entity_name,
                    "baseline_method": baseline_name,
                    "baseline_value": baseline_value,
                    "raw_personalised_adjustment": adjustment,
                    "raw_candidate_sprint_ev": baseline_value + adjustment if np.isfinite(adjustment) else np.nan,
                }
            )
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients["raw_candidate_ev_rank"] = coefficients.groupby("entity_type")[
        "raw_candidate_sprint_ev"
    ].rank(ascending=False, method="min", na_option="bottom")
    coefficients["raw_adjustment_rank"] = coefficients.groupby("entity_type")[
        "raw_regression_adjustment"
    ].rank(ascending=False, method="min", na_option="bottom")
    coefficients = coefficients.sort_values(
        ["entity_type", "raw_candidate_sprint_ev", "entity_name"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return coefficients, pd.DataFrame(sensitivity_rows), pd.DataFrame(shrinkage_rows)


def reliability_diagnostics(coefficients: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "entity_type", "entity_id", "entity_name", "valid_sprint_observations",
        "base_points_variance", "raw_alpha", "raw_beta", "alpha_standard_error",
        "beta_standard_error", "r_squared", "adjusted_r_squared", "condition_number",
        "residual_standard_error", "identifiable", "reliability_flag",
    ]
    return coefficients[columns].sort_values(
        ["entity_type", "reliability_flag", "entity_name"]
    ).reset_index(drop=True)


def _json_asset_records(
    coefficients: pd.DataFrame,
    shrinkage: pd.DataFrame,
    source_version: str,
) -> list[dict[str, object]]:
    rows = []
    for asset in coefficients.itertuples(index=False):
        candidates = shrinkage[
            shrinkage["entity_type"].eq(asset.entity_type)
            & shrinkage["entity_id"].eq(asset.entity_id)
        ]
        rows.append(
            {
                "entity_type": asset.entity_type,
                "canonical_entity_id": asset.entity_id,
                "entity_name": asset.entity_name,
                "alpha": asset.raw_alpha,
                "beta": asset.raw_beta,
                "observation_count": asset.valid_sprint_observations,
                "reliability_flag": asset.reliability_flag,
                "current_normal_baseline": asset.current_normal_baseline,
                "raw_personalised_adjustment": asset.raw_regression_adjustment,
                "raw_candidate_sprint_ev": asset.raw_candidate_sprint_ev,
                "mean_bonus": asset.mean_observed_sprint_bonus,
                "median_bonus": asset.median_observed_sprint_bonus,
                "group_alpha": asset.group_alpha,
                "group_beta": asset.group_beta,
                "shrinkage_candidates": [
                    {
                        "weight": row.weight,
                        "alpha": row.shrunk_alpha,
                        "beta": row.shrunk_beta,
                        "adjustment": row.shrunk_adjustment,
                        "candidate_sprint_ev": row.shrunk_candidate_sprint_ev,
                    }
                    for row in candidates.itertuples(index=False)
                ],
                "source_data_version": source_version,
                "research_only": True,
            }
        )
    return _finite_or_none(rows)


def _case_study_text(
    coefficients: pd.DataFrame,
    observations: pd.DataFrame,
    names: list[str],
) -> str:
    blocks = []
    for name in names:
        match = coefficients[coefficients["entity_name"].str.contains(name, case=False, regex=False)]
        if match.empty:
            continue
        asset = match.iloc[0]
        history = observations[
            observations["entity_type"].eq(asset["entity_type"])
            & observations["entity_id"].eq(asset["entity_id"])
        ][["round", "event_name", "base_weekend_points", "extra_sprint_points", "included_in_regression"]]
        blocks.extend(
            [
                f"#### {asset['entity_name']}",
                "",
                f"Baseline {asset['current_normal_baseline']:.2f}; alpha {_display(asset['raw_alpha'])}; beta {_display(asset['raw_beta'])}; adjustment {_display(asset['raw_regression_adjustment'])}; reliability `{asset['reliability_flag']}`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.",
                "",
                _markdown_table(history.round(4)),
                "",
            ]
        )
    return "\n".join(blocks)


def _display(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{float(numeric):.4f}" if pd.notna(numeric) else "unavailable"


def write_report(
    output: Path,
    coefficients: pd.DataFrame,
    observations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    group_fits: dict[str, dict[str, object]],
) -> None:
    report_columns = [
        "entity_name", "current_normal_baseline", "mean_observed_sprint_bonus",
        "raw_alpha", "raw_beta", "raw_regression_adjustment", "raw_candidate_sprint_ev",
        "reliability_flag",
    ]
    driver_table = coefficients[coefficients["entity_type"].eq("driver")][report_columns].round(4)
    constructor_table = coefficients[coefficients["entity_type"].eq("constructor")][report_columns].round(4)
    unstable = coefficients[~coefficients["reliability_flag"].eq("usable_but_uncertain")][
        ["entity_type", "entity_name", "valid_sprint_observations", "raw_beta", "condition_number", "reliability_flag"]
    ]
    sensitivity_summary = sensitivity.groupby("baseline_method").agg(
        assets=("entity_id", "count"),
        mean_baseline=("baseline_value", "mean"),
        mean_adjustment=("raw_personalised_adjustment", "mean"),
        min_adjustment=("raw_personalised_adjustment", "min"),
        max_adjustment=("raw_personalised_adjustment", "max"),
    ).reset_index()
    cases = _case_study_text(
        coefficients,
        observations,
        [
            "Hulkenberg", "Bottas", "Stroll", "Lawson", "Bearman", "Antonelli",
            "Mercedes", "McLaren", "Aston Martin", "Cadillac",
        ],
    )
    lines = [
        "# Personalised 2026 Sprint adjustments",
        "",
        "## 1. Executive summary",
        "",
        "This offline study fits a separate Sprint relationship for every current 2026 asset. All coefficients are research-only; no production forecast, optimiser, cache, or UI reads these files.",
        "",
        f"The pooled reference is `bonus = {group_fits['driver']['alpha']:.4f} + {group_fits['driver']['beta']:.4f} × base` for drivers and `bonus = {group_fits['constructor']['alpha']:.4f} + {group_fits['constructor']['beta']:.4f} × base` for constructors.",
        "",
        "## 2. Method",
        "",
        "```text\nextra Sprint points = Sprint points + Sprint Qualifying points\nbase weekend points = total points - extra Sprint points\npersonalised bonus = alpha_asset + beta_asset × current normal baseline\ncandidate Sprint EV = current normal baseline + personalised bonus\n```",
        "",
        "The primary baseline averages ordinary-weekend content across every valid completed round 1–11: normal weekends use total points, while Sprint weekends use total minus their official Sprint-specific components. Round 12 is explicitly excluded.",
        "",
        "## 3. Drivers",
        "",
        _markdown_table(driver_table),
        "",
        "## 4. Constructors",
        "",
        _markdown_table(constructor_table),
        "",
        "## 5. Asset case studies",
        "",
        cases,
        "## 6. Group versus personalised estimates",
        "",
        "`asset_predictions.csv` includes raw, group, mean, median, recency, and 25/50/75% shrunk alternatives. Raw personal fits can be dominated by four points; group shrinkage deliberately exposes a continuum rather than selecting a production weight.",
        "",
        "## 7. Baseline sensitivity",
        "",
        _markdown_table(sensitivity_summary.round(4)),
        "",
        "## 8. Reliability",
        "",
        "Every ordinary two-parameter asset fit needs at least three valid Sprint observations and non-trivial base variance. Extreme slopes and high residual error are retained and flagged rather than hidden.",
        "",
        _markdown_table(unstable.round(4)),
        "",
        "## 9. Recommendation for later implementation",
        "",
        "A lightly shrunk personal regression (for example the explicit 25% asset / 75% group candidate) is the most defensible future shadow model among these outputs. It retains directionally personal evidence without treating four Sprints as a stable asset law. Mean personal bonus is a useful simpler benchmark. Raw personal coefficients should not be activated without more Sprint events and a walk-forward evaluation.",
        "",
        "## 10. Limitations",
        "",
        "There are only four completed Sprints, and outcomes within each event are correlated. Crashes, DNFs, penalties, and positions lost are genuine but can dominate individual slopes. Replacement drivers can have missing component observations. This current-season-only analysis uses all completed rounds descriptively, not a historical walk-forward design. Coefficients can change materially after every future Sprint.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_calibration(
    canonical_path: str | Path,
    schedule_path: str | Path,
    current_prices_path: str | Path,
    output_path: str | Path,
    *,
    recency_decay: float = DEFAULT_RECENCY_DECAY,
) -> dict[str, object]:
    if not 0 <= recency_decay <= 1:
        raise ValueError("Recency decay must be within [0, 1].")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    events, annotated, prices, market_metadata = prepare_calibration_data(
        canonical_path, schedule_path, current_prices_path
    )
    observations = build_sprint_observations(annotated, prices)
    history = build_normalised_history(annotated, prices)
    baselines = build_baselines(history, recency_decay)
    group_fits = fit_group_regressions(observations)
    coefficients, sensitivity, shrinkage = calibrate_assets(
        observations, baselines, group_fits, recency_decay=recency_decay
    )
    diagnostics = reliability_diagnostics(coefficients)
    source_versions = sorted(annotated["data_version"].dropna().astype(str).unique())
    if len(source_versions) != 1:
        raise ValueError(f"Expected one canonical source version, found: {source_versions}")
    source_version = source_versions[0]
    completed_rounds = sorted(int(value) for value in events["round"].unique())
    sprint_rounds = sorted(
        int(value) for value in events.loc[events["weekend_format"].eq("sprint"), "round"].unique()
    )
    manifest = {
        "model_name": "personalised_2026_sprint_adjustment",
        "formula": "extra_sprint_points = alpha_asset + beta_asset * base_weekend_points",
        "target_definition": "sprint_points + sprint_qualifying_points, min_count=1",
        "predictor_definition": "fantasy_points_total - extra_sprint_points on completed Sprint weekends",
        "baseline_definition": "mean normalised ordinary-session score across valid completed rounds 1-11",
        "source_dataset_version": source_version,
        "completed_rounds": completed_rounds,
        "sprint_rounds": sprint_rounds,
        "generated_at": "2026-07-05T00:00:00Z",
        "generated_at_policy": "deterministic latest completed round-11 event date",
        "script_version": SCRIPT_VERSION,
        "random_seed": RANDOM_SEED,
        "recency_decay": recency_decay,
        "missing_data_policy": "both Sprint fields absent means missing; never converted to zero",
        "negative_adjustment_policy": "retained without clipping",
        "coefficient_status": "research_only",
        "current_market_feed_round": market_metadata["feed_round"],
    }
    group_json = {
        entity_type: {
            **_finite_or_none(fit),
            "formula": "extra_sprint_points = alpha + beta * base_weekend_points",
            "research_only": True,
        }
        for entity_type, fit in group_fits.items()
    }
    asset_json = {
        "model_name": manifest["model_name"],
        "source_data_version": source_version,
        "research_only": True,
        "assets": _json_asset_records(coefficients, shrinkage, source_version),
    }
    (output / "method_manifest.json").write_text(
        json.dumps(_finite_or_none(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "group_coefficients.json").write_text(
        json.dumps(group_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "asset_coefficients.json").write_text(
        json.dumps(asset_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    coefficient_columns = [
        "entity_type", "entity_id", "entity_name", "current_price",
        "valid_sprint_observations", "base_points_variance", "raw_alpha", "raw_beta",
        "alpha_standard_error", "beta_standard_error", "r_squared", "adjusted_r_squared",
        "condition_number", "residual_standard_error", "identifiable", "reliability_flag",
        "current_normal_baseline", "raw_regression_adjustment", "raw_candidate_sprint_ev",
        "group_alpha", "group_beta",
    ]
    coefficients.sort_values(
        ["entity_type", "raw_regression_adjustment"], ascending=[True, False], na_position="last"
    )[coefficient_columns].to_csv(output / "asset_coefficients.csv", index=False, float_format="%.10g")
    coefficients.to_csv(output / "asset_predictions.csv", index=False, float_format="%.10g")
    observations.to_csv(output / "sprint_observations.csv", index=False, float_format="%.10g")
    sensitivity.to_csv(output / "baseline_sensitivity.csv", index=False, float_format="%.10g")
    shrinkage.to_csv(output / "shrinkage_comparison.csv", index=False, float_format="%.10g")
    diagnostics.to_csv(output / "reliability_diagnostics.csv", index=False, float_format="%.10g")
    write_report(output, coefficients, observations, sensitivity, group_fits)
    return {
        "events": events,
        "annotated": annotated,
        "observations": observations,
        "history": history,
        "baselines": baselines,
        "group_fits": group_fits,
        "coefficients": coefficients,
        "sensitivity": sensitivity,
        "shrinkage": shrinkage,
        "manifest": manifest,
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
        default=PROJECT_ROOT / "reports/2026_personalised_sprint_adjustments",
    )
    parser.add_argument("--recency-decay", type=float, default=DEFAULT_RECENCY_DECAY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_calibration(
        args.canonical,
        args.schedule,
        args.current_prices,
        args.output,
        recency_decay=args.recency_decay,
    )
    print(f"Saved research-only personalised Sprint calibration to {args.output}")
    for entity_type in ("driver", "constructor"):
        fit = result["group_fits"][entity_type]
        print(
            f"{entity_type.title()} group: alpha={fit['alpha']:.4f}, "
            f"beta={fit['beta']:.4f}, n={fit['sprint_observation_count']}"
        )
    flags = result["coefficients"]["reliability_flag"].value_counts().to_dict()
    print(f"Reliability flags: {flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
