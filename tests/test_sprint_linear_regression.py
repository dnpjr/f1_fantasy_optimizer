from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH
from scripts.analyse_sprint_linear_regression import (
    _ridge_asset_coefficients,
    build_coefficient_tables,
    build_observation_dataset,
    cluster_bootstrap_2026,
    individual_2026_coefficients,
    leave_one_sprint_out_2026,
    run_analysis,
)
from scripts.analyse_sprint_multiplier import load_recorded_data, load_schedule_metadata


@pytest.fixture(scope="module")
def recorded_data() -> pd.DataFrame:
    return load_recorded_data("data/research/sprint_round_11/canonical.csv")


@pytest.fixture(scope="module")
def schedule() -> pd.DataFrame:
    return load_schedule_metadata(Path("data/cache"))


@pytest.fixture(scope="module")
def observations(recorded_data, schedule) -> pd.DataFrame:
    return build_observation_dataset(recorded_data, schedule)[0]


@pytest.fixture(scope="module")
def coefficients(observations):
    return build_coefficient_tables(observations)


def test_drivers_and_constructors_are_always_fitted_separately(coefficients):
    overall, season, normalised, _weights = coefficients
    for frame in (overall, season, normalised):
        assert set(frame["entity_type"]) == {"driver", "constructor"}
        assert not frame["entity_type"].isna().any()
    current = overall[overall["period"].eq("2026") & overall["estimator"].eq("OLS")]
    assert len(current) == 2
    assert current.groupby("entity_type").size().eq(1).all()


def test_only_recorded_canonical_totals_are_used(recorded_data):
    assert recorded_data["is_recorded_total"].astype(bool).all()
    assert not recorded_data["is_reconstructed"].astype(bool).any()
    assert set(recorded_data["fantasy_score_origin"]) <= {
        "official_recorded",
        "third_party_recorded",
    }


def test_weekend_format_comes_from_verified_schedule(observations):
    events = observations[["season", "round", "event_name"]].drop_duplicates()
    assert set(events[events["season"].eq(2026)]["round"]) == {2, 4, 5, 9}
    assert set(observations["event_cluster"]) == {
        "2023-4", "2023-9", "2023-12", "2023-17", "2023-18", "2023-20",
        "2024-5", "2024-6", "2024-11", "2024-19", "2024-21", "2024-23",
        "2025-2", "2025-6", "2025-13", "2025-19", "2025-21", "2025-23",
        "2026-2", "2026-4", "2026-5", "2026-9",
    }


def test_predictive_baseline_uses_no_future_events(recorded_data, schedule):
    original = build_observation_dataset(recorded_data, schedule)[0]
    changed = recorded_data.copy(deep=True)
    changed.loc[
        changed["season"].eq(2026)
        & changed["round"].gt(2)
        & changed["entity_type"].eq("driver"),
        "fantasy_points_total",
    ] += 10_000.0
    recalculated = build_observation_dataset(changed, schedule)[0]
    columns = ["canonical_entity_id", "x_strict_prior_normal"]
    pd.testing.assert_frame_equal(
        original[
            original["season"].eq(2026)
            & original["round"].eq(2)
            & original["entity_type"].eq("driver")
        ][columns].reset_index(drop=True),
        recalculated[
            recalculated["season"].eq(2026)
            & recalculated["round"].eq(2)
            & recalculated["entity_type"].eq("driver")
        ][columns].reset_index(drop=True),
    )


def test_descriptive_full_season_results_are_labelled_non_predictive(coefficients):
    _overall, season, _normalised, _weights = coefficients
    descriptive = season[season["baseline_method"].str.contains("descriptive")]
    assert not descriptive.empty
    assert descriptive["model_variant"].str.contains("non_predictive").all()


def test_season_normalisation_is_exact(observations):
    valid = observations.dropna(subset=["x_strict_prior_normal"])
    assert valid["x_strict_normalised"].to_numpy() == pytest.approx(
        (valid["x_strict_prior_normal"] / valid["season_normal_mean"]).to_numpy()
    )
    assert valid["y_normalised"].to_numpy() == pytest.approx(
        (valid["sprint_score"] / valid["season_normal_mean"]).to_numpy()
    )


def test_individual_four_event_fits_are_flagged_unreliable(observations):
    raw, _shrunk, _penalties = individual_2026_coefficients(observations)
    assert raw["sprint_observations"].max() == 4
    assert not raw["reliably_identifiable"].any()
    assert raw["identifiability_warning"].str.contains("unstable|variation|sparse").all()


def test_ridge_estimate_shrinks_toward_correct_entity_group():
    asset = pd.DataFrame(
        {"x_strict_prior_normal": [1.0, 2.0, 3.0], "sprint_score": [50.0, -20.0, 80.0]}
    )
    group_alpha, group_beta = 8.0, 0.75
    weak_alpha, weak_beta = _ridge_asset_coefficients(
        asset, group_alpha, group_beta, 0.1, center=2.0, scale=1.0
    )
    strong_alpha, strong_beta = _ridge_asset_coefficients(
        asset, group_alpha, group_beta, 10_000.0, center=2.0, scale=1.0
    )
    assert abs(strong_alpha - group_alpha) < abs(weak_alpha - group_alpha)
    assert abs(strong_beta - group_beta) < abs(weak_beta - group_beta)


def test_leave_one_sprint_out_removes_exact_2026_event(observations):
    sensitivity = leave_one_sprint_out_2026(observations)
    assert len(sensitivity) == 8
    assert sensitivity.groupby("entity_type")["excluded_round"].apply(set).to_dict() == {
        "driver": {2, 4, 5, 9},
        "constructor": {2, 4, 5, 9},
    }
    assert sensitivity["remaining_sprint_events"].eq(3).all()


def test_event_cluster_bootstrap_is_deterministic(observations):
    first = cluster_bootstrap_2026(observations, samples=100, seed=17)
    second = cluster_bootstrap_2026(observations, samples=100, seed=17)
    pd.testing.assert_frame_equal(first, second)


def test_2021_and_2022_are_absent(recorded_data, observations):
    assert recorded_data["season"].min() == 2023
    assert observations["season"].min() == 2023
    assert not observations["season"].isin({2021, 2022}).any()


def test_constructor_totals_are_not_rebuilt_from_driver_totals(recorded_data, observations):
    source = recorded_data[
        recorded_data["season"].eq(2026)
        & recorded_data["round"].eq(2)
        & recorded_data["entity_type"].eq("constructor")
    ].set_index("canonical_entity_id")["fantasy_points_total"]
    observed = observations[
        observations["season"].eq(2026)
        & observations["round"].eq(2)
        & observations["entity_type"].eq("constructor")
    ].set_index("canonical_entity_id")["sprint_score"]
    pd.testing.assert_series_equal(source.sort_index(), observed.sort_index(), check_names=False)
    driver_total = recorded_data[
        recorded_data["season"].eq(2026)
        & recorded_data["round"].eq(2)
        & recorded_data["entity_type"].eq("driver")
    ]["fantasy_points_total"].sum()
    assert observed.sum() != driver_total


def test_script_outputs_are_reproducible(tmp_path):
    output = tmp_path / "analysis"
    run_analysis(output_dir=output, seed=99, bootstrap_samples=50)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    run_analysis(output_dir=output, seed=99, bootstrap_samples=50)
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    assert first == second
