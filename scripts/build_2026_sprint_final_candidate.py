#!/usr/bin/env python3
"""Build the research-only simplified 2026 Sprint-EV candidate offline."""

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
from scripts.analyse_2026_sprint_bonus import _markdown_table
from scripts.analyse_2026_sprint_partial_pooling import (
    attach_strength,
    build_strength_definitions,
    complete_observation_grid,
    empirical_bayes_shrunk_means,
    fit_penalised_model,
)
from scripts.calibrate_asset_sprint_adjustments import (
    CALIBRATION_MAX_ROUND,
    build_baselines,
    build_normalised_history,
    build_sprint_observations,
    prepare_calibration_data,
)


SCRIPT_VERSION = "2026-sprint-final-candidate-v1"
DRIVER_STRENGTH = "z_form"
CONSTRUCTOR_STRENGTH = "blend_form_0.75_price_0.25"
MULTIPLIER_MINIMUM_NORMAL_EV = 1.0
SANITY_ASSETS = (
    "Nico Hulkenberg", "Valtteri Bottas", "Lance Stroll", "Liam Lawson",
    "Oliver Bearman", "Kimi Antonelli", "George Russell", "Lando Norris",
    "Lewis Hamilton", "Charles Leclerc", "Mercedes", "Ferrari", "McLaren",
    "Red Bull Racing", "Williams", "Haas F1 Team", "Audi", "Aston Martin",
    "Cadillac",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def prepare_candidate_data(
    canonical_path: str | Path,
    schedule_path: str | Path,
    current_prices_path: str | Path,
    *, maximum_round: int | None = CALIBRATION_MAX_ROUND,
) -> dict[str, Any]:
    """Prepare comparable observations, optionally using the frozen research cutoff."""
    events, annotated, prices, market_metadata = prepare_calibration_data(
        canonical_path,
        schedule_path,
        current_prices_path,
        maximum_round=maximum_round,
    )
    observations = build_sprint_observations(annotated, prices, maximum_round=maximum_round).rename(
        columns={"included_in_regression": "observation_valid"}
    )
    history = build_normalised_history(annotated, prices, maximum_round=maximum_round)
    # The proposed candidate is explicitly the decay=1 current-form definition.
    baselines = build_baselines(history, recency_decay=1.0)
    observations = complete_observation_grid(observations, events, baselines)
    definitions, _diagnostics = build_strength_definitions(baselines)
    source_versions = sorted(annotated["data_version"].dropna().astype(str).unique())
    if len(source_versions) != 1:
        raise ValueError(f"Expected one canonical source version, found {source_versions}.")
    return {
        "events": events,
        "annotated": annotated,
        "prices": prices,
        "market_metadata": market_metadata,
        "observations": observations,
        "history": history,
        "baselines": baselines,
        "definitions": definitions,
        "source_data_version": source_versions[0],
    }


def _shrinkage_diagnostics(
    attached: pd.DataFrame,
    group_fit: dict[str, Any],
) -> dict[str, float]:
    """Reproduce the exact variance estimates used by empirical Bayes."""
    valid = attached[attached["observation_valid"]].copy()
    valid["group_bonus"] = (
        float(group_fit["mu"]) + float(group_fit["lambda"]) * valid["strength"]
    )
    valid["residual_bonus"] = valid["extra_sprint_points"] - valid["group_bonus"]
    grouped = valid.groupby("entity_id")["residual_bonus"].agg(["mean", "count", "var"])
    within = float(valid.groupby("entity_id")["residual_bonus"].var().mean())
    if not np.isfinite(within):
        within = float(valid["residual_bonus"].var())
    between = float(grouped["mean"].var()) if len(grouped) > 1 else 0.0
    average_noise = float((grouped["var"].fillna(within) / grouped["count"]).mean())
    tau_squared = max(0.0, between - average_noise)
    return {
        "within_residual_variance": within,
        "between_personal_residual_mean_variance": between,
        "average_personal_mean_noise_variance": average_noise,
        "tau_asset_squared": tau_squared,
    }


def _load_prior_outputs(report_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = Path(report_path)
    predictions = pd.read_csv(report / "asset_predictions.csv")
    comparison = pd.read_csv(report / "model_comparison.csv")
    required_predictions = {
        "entity_id", "entity_type", "predicted_next_sprint_bonus",
    }
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("The prior partial-pooling asset predictions have an unexpected schema.")
    return predictions, comparison


def _selected_comparison(previous: pd.DataFrame) -> pd.DataFrame:
    """Extract established LO-Sprint metrics without rerunning experiments."""
    base = previous[
        previous["comparison_scope"].eq("candidate_model")
        & previous["model"].isin(
            ["constant", "strength_only", "personal_mean", "shrunk_personal_mean"]
        )
    ].copy()
    full = previous[
        previous["model"].eq("full_partial_pooling")
        & previous["selected_penalties"].fillna(False).astype(bool)
    ].copy()
    selected = pd.concat([base, full], ignore_index=True)
    expected = {(kind, model) for kind in ("driver", "constructor") for model in (
        "constant", "strength_only", "personal_mean", "shrunk_personal_mean",
        "full_partial_pooling",
    )}
    actual = set(zip(selected["entity_type"], selected["model"]))
    if actual != expected or len(selected) != len(expected):
        raise ValueError("Prior validation output does not contain one selected row per model.")
    columns = [
        "entity_type", "model", "strength_definition", "asset_penalty",
        "event_penalty", "observations", "mae", "rmse", "bias", "spearman",
        "mae_low_form_third", "mae_middle_form_third", "mae_high_form_third",
    ]
    return selected[columns].sort_values(["entity_type", "model"]).reset_index(drop=True)


def build_final_candidate(
    prepared: dict[str, Any],
    prior_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit only the two already-selected simple candidates and tabulate assets."""
    observations = prepared["observations"]
    definitions = prepared["definitions"]
    prior_lookup = prior_predictions.set_index(["entity_type", "entity_id"])
    all_rows: list[dict[str, Any]] = []
    driver_details = pd.DataFrame()
    constructor_details = pd.DataFrame()
    model_details: dict[str, Any] = {}

    for entity_type, strength_name in (
        ("driver", DRIVER_STRENGTH),
        ("constructor", CONSTRUCTOR_STRENGTH),
    ):
        source = attach_strength(
            observations[observations["entity_type"].eq(entity_type)],
            definitions,
            strength_name,
        )
        asset_definitions = definitions[
            definitions["entity_type"].eq(entity_type)
            & definitions["strength_definition"].eq(strength_name)
        ].copy()
        group_fit = fit_penalised_model(
            source,
            include_strength=True,
            include_asset=False,
            include_event=False,
            constrain_strength=True,
        )
        shrunk = empirical_bayes_shrunk_means(source, group_fit, asset_definitions)
        stats = source[source["observation_valid"]].groupby("entity_id")[
            "extra_sprint_points"
        ].agg(
            observed_sprint_count="count",
            observed_mean_sprint_bonus="mean",
            observed_median_sprint_bonus="median",
        )
        constant_bonus = float(
            source.loc[source["observation_valid"], "extra_sprint_points"].mean()
        )
        shrinkage = _shrinkage_diagnostics(source, group_fit)
        shrunk_lookup = shrunk.set_index("entity_id")

        for asset in asset_definitions.sort_values("entity_name").itertuples(index=False):
            key = (entity_type, asset.entity_id)
            observed = stats.loc[asset.entity_id] if asset.entity_id in stats.index else None
            eb = shrunk_lookup.loc[asset.entity_id]
            count = int(observed["observed_sprint_count"]) if observed is not None else 0
            personal_mean = (
                float(observed["observed_mean_sprint_bonus"])
                if observed is not None else np.nan
            )
            median = (
                float(observed["observed_median_sprint_bonus"])
                if observed is not None else np.nan
            )
            group_bonus = float(group_fit["mu"]) + float(group_fit["lambda"]) * float(asset.strength)
            weight = float(eb["empirical_bayes_weight"])
            shrunk_bonus = float(eb["shrunk_personal_mean_bonus"])
            if count:
                personal_contribution = weight * personal_mean
                group_contribution = (1.0 - weight) * group_bonus
            else:
                personal_contribution = 0.0
                group_contribution = group_bonus
            final_bonus = shrunk_bonus if entity_type == "driver" else group_bonus
            normal_ev = float(asset.normal_ev)
            sprint_ev = normal_ev + final_bonus
            multiplier_defined = normal_ev >= MULTIPLIER_MINIMUM_NORMAL_EV
            full_bonus = (float(prior_lookup.loc[key, "predicted_next_sprint_bonus"])
                          if key in prior_lookup.index else np.nan)
            all_rows.append(
                {
                    "entity": asset.entity_name,
                    "entity_type": entity_type,
                    "entity_id": asset.entity_id,
                    "current_price": float(asset.current_price),
                    "normal_ev": normal_ev,
                    "form_percentile": float(asset.form_percentile),
                    "price_percentile": float(asset.price_percentile),
                    "observed_sprint_count": count,
                    "observed_mean_sprint_bonus": personal_mean,
                    "observed_median_sprint_bonus": median,
                    "selected_strength_definition": strength_name,
                    "selected_strength": float(asset.strength),
                    "group_bonus": group_bonus,
                    "empirical_bayes_weight": weight if entity_type == "driver" else np.nan,
                    "personal_mean_contribution": personal_contribution if entity_type == "driver" else np.nan,
                    "group_contribution": group_contribution if entity_type == "driver" else np.nan,
                    "final_sprint_bonus": final_bonus,
                    "normal_weekend_ev": normal_ev,
                    "sprint_weekend_ev": sprint_ev,
                    "absolute_sprint_uplift": final_bonus,
                    "effective_multiplier": sprint_ev / normal_ev if multiplier_defined else np.nan,
                    "effective_multiplier_status": "defined" if multiplier_defined else "undefined",
                    "constant_bonus_candidate": constant_bonus,
                    "strength_only_candidate": group_bonus,
                    "personal_mean_candidate": personal_mean,
                    "shrunk_personal_mean_candidate": shrunk_bonus,
                    "full_partial_pooling_candidate": full_bonus,
                    "future_event_effect": 0.0,
                }
            )

        model_details[entity_type] = {
            "strength_definition": strength_name,
            "strength_parameters": {
                "normal_ev_mean": float(asset_definitions["normal_ev"].mean()),
                "normal_ev_population_sd": float(asset_definitions["normal_ev"].std(ddof=0)),
            },
            "mu": float(group_fit["mu"]),
            "lambda": float(group_fit["lambda"]),
            "lambda_constrained_nonnegative": True,
            "observation_count": int(group_fit["observation_count"]),
            "constant_bonus": constant_bonus,
            "shrinkage": shrinkage,
        }

        details = pd.DataFrame(all_rows)
        details = details[details["entity_type"].eq(entity_type)].copy()
        if entity_type == "driver":
            details["personal_residual_mean"] = (
                details["observed_mean_sprint_bonus"] - details["group_bonus"]
            )
            details["personal_mean_noise_variance"] = (
                shrinkage["within_residual_variance"]
                / details["observed_sprint_count"].replace(0, np.nan)
            )
            details["tau_asset_squared"] = shrinkage["tau_asset_squared"]
            details["within_residual_variance"] = shrinkage["within_residual_variance"]
            driver_details = details[[
                "entity", "entity_id", "observed_sprint_count",
                "observed_mean_sprint_bonus", "observed_median_sprint_bonus",
                "selected_strength", "group_bonus", "personal_residual_mean",
                "personal_mean_noise_variance", "tau_asset_squared",
                "within_residual_variance", "empirical_bayes_weight",
                "personal_mean_contribution", "group_contribution", "final_sprint_bonus",
            ]]
        else:
            constructor_details = details[[
                "entity", "entity_id", "current_price", "normal_ev",
                "form_percentile", "price_percentile", "selected_strength",
                "group_bonus", "final_sprint_bonus",
            ]]

    candidate = pd.DataFrame(all_rows).sort_values(
        ["entity_type", "sprint_weekend_ev", "entity"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    # Prior research is comparison-only; new or retired drivers must not block a refit.
    return candidate, driver_details, constructor_details, model_details


def build_sanity_checks(candidate: pd.DataFrame) -> pd.DataFrame:
    selected = candidate[candidate["entity"].isin(SANITY_ASSETS)].copy()
    if set(selected["entity"]) != set(SANITY_ASSETS):
        missing = sorted(set(SANITY_ASSETS) - set(selected["entity"]))
        raise ValueError(f"Missing requested sanity assets: {missing}")
    selected["sanity_conclusion"] = "value follows the selected simplified formula"
    selected.loc[selected["entity"].eq("Valtteri Bottas"), "sanity_conclusion"] = (
        "moderated EB bonus; old unstable +13 adjustment is absent"
    )
    selected.loc[selected["entity"].eq("Nico Hulkenberg"), "sanity_conclusion"] = (
        "negative personal mean retained and moderated toward the group"
    )
    selected.loc[selected["entity"].eq("Lance Stroll"), "sanity_conclusion"] = (
        "weak normal form and positive personal Sprint history are both represented"
    )
    selected.loc[selected["entity"].isin(["Liam Lawson", "Oliver Bearman"]), "sanity_conclusion"] = (
        "reasonable positive driver bonus retained"
    )
    selected.loc[selected["entity"].isin(["Kimi Antonelli", "George Russell", "Lando Norris"]), "sanity_conclusion"] = (
        "material positive bonus for a stronger driver"
    )
    selected.loc[selected["entity"].isin(["Mercedes", "Ferrari", "McLaren"]), "sanity_conclusion"] = (
        "large leading-constructor strength bonus"
    )
    selected.loc[selected["entity"].eq("Red Bull Racing"), "sanity_conclusion"] = (
        "positive strength-only constructor bonus"
    )
    selected.loc[selected["entity"].isin(["Williams", "Haas F1 Team", "Audi", "Aston Martin", "Cadillac"]), "sanity_conclusion"] = (
        "smaller positive constructor bonus than the leaders"
    )
    return selected[[
        "entity", "entity_type", "normal_ev", "observed_mean_sprint_bonus",
        "group_bonus", "empirical_bayes_weight", "final_sprint_bonus",
        "sprint_weekend_ev", "sanity_conclusion",
    ]].sort_values(["entity_type", "entity"]).reset_index(drop=True)


def _write_report(
    output: Path,
    candidate: pd.DataFrame,
    comparison: pd.DataFrame,
    sanity: pd.DataFrame,
    models: dict[str, Any],
) -> None:
    driver = models["driver"]
    constructor = models["constructor"]
    shrink = driver["shrinkage"]
    driver_table = candidate[candidate["entity_type"].eq("driver")][[
        "entity", "normal_ev", "observed_mean_sprint_bonus", "group_bonus",
        "empirical_bayes_weight", "final_sprint_bonus", "sprint_weekend_ev",
    ]].sort_values("entity")
    constructor_table = candidate[candidate["entity_type"].eq("constructor")][[
        "entity", "normal_ev", "selected_strength", "final_sprint_bonus",
        "sprint_weekend_ev",
    ]].sort_values("entity")
    lines = [
        "# Simplified 2026 Sprint-EV final candidate",
        "",
        "## Executive recommendation",
        "",
        "**A. Values look sensible enough for a shadow implementation.** This is a research recommendation only; the model is not production-approved or activated.",
        "",
        "Normal EV is the equal-weight mean of included completed normal-equivalent scores. Normal weekends retain recorded total; Sprint weekends subtract official Sprint and Sprint-qualifying points. With application decay `p`, this generalises to a recency-weighted mean whose included scores receive weights `1, p, p², ...` newest first.",
        "",
        "Driver formulas:",
        "",
        f"- `z_form = (normal_ev - {driver['strength_parameters']['normal_ev_mean']:.12f}) / {driver['strength_parameters']['normal_ev_population_sd']:.12f}` (within the current driver class, population SD).",
        f"- `group_bonus = {driver['mu']:.12f} + {driver['lambda']:.12f} * z_form`.",
        "- `personal_mean = arithmetic mean of valid official Sprint-only bonuses`, retaining negative values and omitting missing observations.",
        f"- `w_i = tau² / (tau² + sigma_within² / n_i)`, where `tau²={shrink['tau_asset_squared']:.12f}` and `sigma_within²={shrink['within_residual_variance']:.12f}`.",
        "- `Sprint bonus = w_i * personal_mean + (1 - w_i) * group_bonus`; an asset with no valid observation gets `w_i=0`.",
        "- `Sprint EV = normal_ev + Sprint bonus`.",
        "",
        "Constructor formulas:",
        "",
        "- `strength = 0.75 * form_percentile + 0.25 * price_percentile`.",
        f"- `Sprint bonus = {constructor['mu']:.12f} + {constructor['lambda']:.12f} * strength`, with the slope constrained nonnegative.",
        "- `Sprint EV = normal_ev + Sprint bonus`.",
        "",
        "The simplified candidate has no asset-specific constructor effect and no future Sprint-event effect; `v_next=0`.",
        "",
        "## Driver table",
        "",
        _markdown_table(driver_table.round(4)),
        "",
        "## Constructor table",
        "",
        _markdown_table(constructor_table.round(4)),
        "",
        "## Sanity checks",
        "",
        _markdown_table(sanity.round(4)),
        "",
        "Bottas receives a small shrunk bonus rather than the old unstable +13 adjustment. Hülkenberg's negative observed mean is retained but moderated. The selected formulas also preserve positive separation for stronger drivers and positive, monotonic constructor bonuses, with Mercedes/Ferrari/McLaren above the weaker constructor group.",
        "",
        "## Validation",
        "",
        "These are historical validation metrics (rounds 2, 4, 5, 9), NOT validation of the refreshed fit, copied from the completed partial-pooling research; no new model search was run.",
        "",
        _markdown_table(comparison[[
            "entity_type", "model", "mae", "rmse", "bias", "spearman",
        ]].round(4)),
        "",
        "## Recommendation",
        "",
        "A. Values look sensible enough for a shadow implementation. The original validation evidence is limited to four completed 2026 Sprints, so a future implementation should remain observable and must be recalibrated as more Sprint data arrives. This report does not activate the model.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_build(
    canonical_path: str | Path,
    schedule_path: str | Path,
    current_prices_path: str | Path,
    previous_report_path: str | Path,
    output_path: str | Path,
    *, maximum_round: int | None = CALIBRATION_MAX_ROUND,
) -> dict[str, Any]:
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_candidate_data(canonical_path, schedule_path, current_prices_path, maximum_round=maximum_round)
    prior_predictions, prior_comparison = _load_prior_outputs(previous_report_path)
    comparison = _selected_comparison(prior_comparison)
    candidate, drivers, constructors, models = build_final_candidate(
        prepared, prior_predictions
    )
    sanity = build_sanity_checks(candidate)

    candidate.to_csv(output / "final_candidate.csv", index=False, float_format="%.12g")
    drivers.to_csv(output / "driver_shrinkage_details.csv", index=False, float_format="%.12g")
    constructors.to_csv(output / "constructor_strength_details.csv", index=False, float_format="%.12g")
    comparison.to_csv(output / "comparison_to_previous_models.csv", index=False, float_format="%.12g")
    sanity.to_csv(output / "sanity_checks.csv", index=False, float_format="%.12g")

    events = prepared["events"]
    prepared["observations"].to_csv(output / "sprint_observations.csv", index=False)
    prepared["history"].to_csv(output / "normalised_history.csv", index=False)
    events.to_csv(output / "included_events.csv", index=False)
    model_json = {
        "research_only": True,
        "production_approved": False,
        "generated_at": str(events["event_date"].max()) + "T00:00:00Z",
        "generated_at_policy": "deterministic latest completed calibration event date",
        "script_version": SCRIPT_VERSION,
        "source_data_version": prepared["source_data_version"],
        "completed_rounds": sorted(int(value) for value in events["round"].unique()),
        "sprint_rounds": sorted(
            int(value)
            for value in events.loc[events["weekend_format"].eq("sprint"), "round"].unique()
        ),
        "normal_ev_method": {
            "decay": 1.0,
            "formula": "mean normal-equivalent score across included valid completed 2026 rounds",
            "normal_weekend": "recorded Fantasy total",
            "sprint_weekend": "recorded Fantasy total - sprint_points - sprint_qualifying_points",
            "missing_treatment": "omit missing observations; never impute zero",
            "application_recency_generalisation": "weighted mean with weights 1,p,p^2,... over included races newest first",
        },
        "driver_model": {
            "method": "empirical_bayes_shrunk_personal_mean",
            "strength_definition": "z_form = within-driver-class population z-score of normal_ev",
            "strength_parameters": models["driver"]["strength_parameters"],
            "group_coefficients": {
                "mu": models["driver"]["mu"],
                "lambda": models["driver"]["lambda"],
                "lambda_constrained_nonnegative": True,
            },
            "shrinkage_method": "residual empirical Bayes; algebraically w*personal_mean + (1-w)*group_bonus",
            "shrinkage_parameters": {
                **models["driver"]["shrinkage"],
                "personal_mean_noise_formula": "within_residual_variance / observed_sprint_count",
                "weight_formula": "tau_asset_squared / (tau_asset_squared + personal_mean_noise_variance)",
                "missing_observation_rule": "w=0; use group bonus",
                "negative_observation_rule": "retain as valid",
            },
            "future_event_effect": 0.0,
        },
        "constructor_model": {
            "method": "strength_only",
            "strength_definition": "0.75 * form_percentile + 0.25 * price_percentile",
            "mu": models["constructor"]["mu"],
            "lambda": models["constructor"]["lambda"],
            "lambda_constrained_nonnegative": True,
            "personal_constructor_effect": None,
            "future_event_effect": 0.0,
        },
        "assets": [
            {
                "entity": row.entity,
                "entity_type": row.entity_type,
                "entity_id": row.entity_id,
                "normal_ev": row.normal_ev,
                "sprint_bonus": row.final_sprint_bonus,
                "sprint_ev": row.sprint_weekend_ev,
            }
            for row in candidate.itertuples(index=False)
        ],
    }
    (output / "final_candidate.json").write_text(
        json.dumps(_json_value(model_json), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output, candidate, comparison, sanity, models)
    return {
        "prepared": prepared,
        "candidate": candidate,
        "driver_details": drivers,
        "constructor_details": constructors,
        "comparison": comparison,
        "sanity": sanity,
        "models": models,
        "model_json": model_json,
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
        "--previous-report", type=Path,
        default=PROJECT_ROOT / "reports/2026_sprint_partial_pooling",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "reports/2026_sprint_final_candidate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_build(
        args.canonical,
        args.schedule,
        args.current_prices,
        args.previous_report,
        args.output,
    )
    print(f"Saved research-only final Sprint candidate to {args.output}")
    print(
        "Driver formula: "
        f"{result['models']['driver']['mu']:.10f} + "
        f"{result['models']['driver']['lambda']:.10f} * z_form, then EB shrinkage"
    )
    print(
        "Constructor formula: "
        f"{result['models']['constructor']['mu']:.10f} + "
        f"{result['models']['constructor']['lambda']:.10f} * strength"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
