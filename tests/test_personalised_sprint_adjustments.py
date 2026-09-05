import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyse_2026_sprint_bonus import sprint_only_target
from scripts.calibrate_asset_sprint_adjustments import (
    CALIBRATION_MAX_ROUND,
    build_baselines,
    build_normalised_history,
    build_sprint_observations,
    calibrate_assets,
    fit_group_regressions,
    fit_linear_regression,
    prepare_calibration_data,
    run_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "data/generated/historical_fantasy_scores_v3_recorded_2023_2026"
    / "historical_fantasy_scores_2023_2026.csv"
)
SCHEDULE = ROOT / "data/cache/schedule_2026.csv"
MARKET = ROOT / "data/cache/verified_fantasy_market.json"


@pytest.fixture(scope="module")
def prepared():
    return prepare_calibration_data(CANONICAL, SCHEDULE, MARKET)


def _regression_rows(entity_type="driver", entity_id="a", x=(0.0, 1.0, 2.0), y=(1.0, 3.0, 5.0)):
    return pd.DataFrame(
        {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "base_weekend_points": x,
            "extra_sprint_points": y,
            "included_in_regression": [True] * len(x),
        }
    )


def test_only_2026_completed_rounds_are_used(prepared):
    events, annotated, _prices, _metadata = prepared
    assert set(annotated["season"]) == {2026}
    assert set(events["round"]) == set(range(1, 12))
    assert annotated["round"].max() == CALIBRATION_MAX_ROUND == 11


def test_driver_and_constructor_group_regressions_are_separate():
    observations = pd.concat(
        [
            _regression_rows("driver", "d", y=(1.0, 3.0, 5.0)),
            _regression_rows("constructor", "c", y=(10.0, 10.0, 10.0)),
        ],
        ignore_index=True,
    )
    fits = fit_group_regressions(observations)
    assert fits["driver"]["beta"] == pytest.approx(2.0)
    assert fits["constructor"]["beta"] == pytest.approx(0.0)
    assert fits["driver"]["alpha"] != fits["constructor"]["alpha"]


def test_target_uses_only_sprint_specific_components_and_preserves_missing():
    frame = pd.DataFrame(
        {
            "sprint_points": [3.0, 0.0, -5.0, np.nan],
            "sprint_qualifying_points": [2.0, 0.0, 1.0, np.nan],
            "race_points": [100.0] * 4,
            "qualifying_points": [-100.0] * 4,
        }
    )
    target = sprint_only_target(frame)
    assert target.iloc[:3].tolist() == [5.0, 0.0, -4.0]
    assert pd.isna(target.iloc[3])


def test_explicit_zero_and_negative_sprint_components_remain_observations():
    target = sprint_only_target(
        pd.DataFrame(
            {
                "sprint_points": [0.0, -8.0],
                "sprint_qualifying_points": [0.0, -2.0],
            }
        )
    )
    assert target.notna().all()
    assert target.tolist() == [0.0, -10.0]


def test_base_weekend_points_are_total_minus_sprint_components(prepared):
    _events, annotated, prices, _metadata = prepared
    observations = build_sprint_observations(annotated, prices)
    valid = observations[observations["included_in_regression"]]
    assert np.allclose(
        valid["base_weekend_points"],
        valid["total_fantasy_points"] - valid["extra_sprint_points"],
    )


def test_missing_components_are_not_converted_to_zero(prepared):
    _events, annotated, prices, _metadata = prepared
    observations = build_sprint_observations(annotated, prices)
    missing = observations[observations["extra_sprint_points"].isna()]
    assert len(missing) == 3
    assert missing["base_weekend_points"].isna().all()
    assert not missing["included_in_regression"].any()
    assert set(missing["exclusion_reason"]) == {"both_sprint_components_missing"}


def test_current_normal_baseline_removes_sprint_points(prepared):
    _events, annotated, prices, _metadata = prepared
    history = build_normalised_history(annotated, prices)
    sprint = history[history["weekend_format"].eq("sprint") & history["extra_sprint_points"].notna()]
    assert np.allclose(
        sprint["normalised_score"],
        sprint["fantasy_points_total"] - sprint["extra_sprint_points"],
    )
    baselines = build_baselines(history, 0.8)
    asset = baselines.iloc[0]
    asset_rows = history[
        history["entity_type"].eq(asset["entity_type"])
        & history["canonical_entity_id"].eq(asset["entity_id"])
    ]
    assert asset["current_normal_baseline"] == pytest.approx(asset_rows["normalised_score"].mean())


def test_round_12_cannot_enter_normalised_history(prepared):
    _events, annotated, prices, _metadata = prepared
    injected = annotated.iloc[[0]].copy()
    injected["round"] = 12
    injected["fantasy_points_total"] = 1_000_000.0
    history = build_normalised_history(pd.concat([annotated, injected]), prices)
    assert history["round"].max() == 11
    assert history["normalised_score"].max() < 1_000_000.0


def test_individual_coefficients_use_only_that_assets_observations():
    first = _regression_rows("driver", "first", y=(1.0, 3.0, 5.0))
    second = _regression_rows("driver", "second", y=(100.0, -100.0, 100.0))
    alone = fit_linear_regression(first, entity_type="driver")
    mixed_but_filtered = fit_linear_regression(
        pd.concat([first, second]).query("entity_id == 'first'"), entity_type="driver"
    )
    assert mixed_but_filtered["alpha"] == pytest.approx(alone["alpha"])
    assert mixed_but_filtered["beta"] == pytest.approx(alone["beta"])


def test_near_zero_predictor_variance_is_flagged():
    rows = _regression_rows(x=(4.0, 4.0, 4.0, 4.0), y=(1.0, 2.0, 3.0, 4.0))
    fit = fit_linear_regression(rows, entity_type="driver")
    assert not fit["identifiable"]
    assert fit["reliability_flag"] == "near_zero_predictor_variance"
    assert pd.isna(fit["beta"])


def test_insufficient_observations_are_flagged():
    rows = _regression_rows(x=(1.0, 2.0), y=(3.0, 4.0))
    fit = fit_linear_regression(rows, entity_type="driver")
    assert not fit["identifiable"]
    assert fit["reliability_flag"] == "insufficient_observations"


def test_negative_personalised_adjustments_are_preserved():
    observations = _regression_rows(x=(0.0, 1.0, 2.0), y=(-2.0, -2.0, -2.0))
    observations["entity_name"] = "Negative"
    observations["event_date"] = ["2026-01-01", "2026-02-01", "2026-03-01"]
    baselines = pd.DataFrame(
        [
            {
                "entity_type": "driver", "entity_id": "a", "entity_name": "Negative",
                "abbreviation": "NEG", "current_price": 1.0, "completed_2026_events": 3,
                "normalised_event_count": 3, "current_normal_baseline": 10.0,
                "normal_weekend_only_mean": 10.0, "median_normalised_baseline": 10.0,
                "recency_weighted_normal_baseline": 10.0,
                "recency_weighted_normal_weekend_only_mean": 10.0,
            }
        ]
    )
    groups = fit_group_regressions(observations)
    coefficients, _sensitivity, _shrinkage = calibrate_assets(
        observations, baselines, groups, recency_decay=0.8
    )
    assert coefficients.iloc[0]["raw_regression_adjustment"] == pytest.approx(-2.0)
    assert coefficients.iloc[0]["raw_candidate_sprint_ev"] == pytest.approx(8.0)


def test_group_and_shrunk_coefficients_are_weighted_correctly():
    asset = _regression_rows(x=(0.0, 1.0, 2.0), y=(1.0, 3.0, 5.0))
    asset["entity_name"] = "Asset"
    asset["event_date"] = ["2026-01-01", "2026-02-01", "2026-03-01"]
    group_fit = fit_group_regressions(asset)
    group_fit["driver"] = {**group_fit["driver"], "alpha": 5.0, "beta": 4.0}
    baselines = pd.DataFrame(
        [{
            "entity_type": "driver", "entity_id": "a", "entity_name": "Asset",
            "abbreviation": "AST", "current_price": 1.0, "completed_2026_events": 3,
            "normalised_event_count": 3, "current_normal_baseline": 10.0,
            "normal_weekend_only_mean": 10.0, "median_normalised_baseline": 10.0,
            "recency_weighted_normal_baseline": 10.0,
            "recency_weighted_normal_weekend_only_mean": 10.0,
        }]
    )
    _coefficients, _sensitivity, shrinkage = calibrate_assets(
        asset, baselines, group_fit, recency_decay=0.8
    )
    half = shrinkage[shrinkage["weight"].eq(0.5)].iloc[0]
    assert half["shrunk_alpha"] == pytest.approx(3.0)
    assert half["shrunk_beta"] == pytest.approx(3.0)
    assert half["shrunk_adjustment"] == pytest.approx(33.0)


def test_json_outputs_are_research_only(tmp_path):
    run_calibration(CANONICAL, SCHEDULE, MARKET, tmp_path)
    manifest = json.loads((tmp_path / "method_manifest.json").read_text())
    assets = json.loads((tmp_path / "asset_coefficients.json").read_text())
    groups = json.loads((tmp_path / "group_coefficients.json").read_text())
    assert manifest["coefficient_status"] == "research_only"
    assert assets["research_only"] is True
    assert all(asset["research_only"] for asset in assets["assets"])
    assert all(group["research_only"] for group in groups.values())


def test_script_outputs_are_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run_calibration(CANONICAL, SCHEDULE, MARKET, first, recency_decay=0.7)
    run_calibration(CANONICAL, SCHEDULE, MARKET, second, recency_decay=0.7)
    assert sorted(path.name for path in first.iterdir()) == sorted(path.name for path in second.iterdir())
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_calibration_script_is_not_referenced_from_production_startup_paths():
    forbidden = "calibrate_asset_sprint_adjustments"
    paths = [ROOT / "streamlit_app.py", *sorted((ROOT / "f1fantasy").glob("*.py"))]
    assert all(forbidden not in path.read_text(encoding="utf-8") for path in paths)
