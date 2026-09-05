import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_2026_sprint_final_candidate import (
    CONSTRUCTOR_STRENGTH,
    DRIVER_STRENGTH,
    build_final_candidate,
    prepare_candidate_data,
    run_build,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "data/generated/historical_fantasy_scores_v3_recorded_2023_2026"
    / "historical_fantasy_scores_2023_2026.csv"
)
SCHEDULE = ROOT / "data/cache/schedule_2026.csv"
MARKET = ROOT / "data/cache/verified_fantasy_market.json"
PREVIOUS = ROOT / "reports/2026_sprint_partial_pooling"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run_build(
        CANONICAL,
        SCHEDULE,
        MARKET,
        PREVIOUS,
        tmp_path_factory.mktemp("sprint_final_candidate"),
    )


def _row(result, entity):
    return result["candidate"].set_index("entity").loc[entity]


def test_only_completed_2026_scoring_data_enters_calibration(result):
    prepared = result["prepared"]
    assert set(prepared["annotated"]["season"]) == {2026}
    assert set(prepared["events"]["round"]) == set(range(1, 12))
    assert set(prepared["observations"]["round"]) == {2, 4, 5, 9}


def test_round_12_scoring_cannot_enter(result):
    prepared = result["prepared"]
    assert prepared["history"]["round"].max() == 11
    assert result["model_json"]["completed_rounds"] == list(range(1, 12))
    assert 12 not in result["model_json"]["completed_rounds"]


def test_normal_ev_removes_sprint_specific_components(result):
    history = result["prepared"]["history"]
    sprint = history[
        history["weekend_format"].eq("sprint")
        & history["extra_sprint_points"].notna()
    ]
    assert np.allclose(
        sprint["normalised_score"],
        sprint["fantasy_points_total"] - sprint["extra_sprint_points"],
    )
    for row in result["candidate"].itertuples(index=False):
        source = history[
            history["entity_type"].eq(row.entity_type)
            & history["canonical_entity_id"].eq(row.entity_id)
        ]["normalised_score"].dropna()
        assert row.normal_ev == pytest.approx(source.mean())


def test_driver_model_is_shrunk_personal_mean_not_regression(result):
    model = result["model_json"]["driver_model"]
    assert model["method"] == "empirical_bayes_shrunk_personal_mean"
    assert DRIVER_STRENGTH in result["models"]["driver"]["strength_definition"]
    assert "alpha" not in model
    assert "beta" not in model
    drivers = result["candidate"][result["candidate"]["entity_type"].eq("driver")]
    assert np.allclose(
        drivers["final_sprint_bonus"],
        drivers["shrunk_personal_mean_candidate"],
    )


def test_constructor_model_has_no_personalised_effect(result):
    model = result["model_json"]["constructor_model"]
    assert model["method"] == "strength_only"
    assert model["personal_constructor_effect"] is None
    constructors = result["candidate"][
        result["candidate"]["entity_type"].eq("constructor")
    ]
    assert np.allclose(constructors["final_sprint_bonus"], constructors["group_bonus"])
    assert constructors["empirical_bayes_weight"].isna().all()


def test_driver_empirical_bayes_weights_reproduce_research(result):
    model = result["models"]["driver"]
    shrink = model["shrinkage"]
    assert model["mu"] == pytest.approx(5.167337315125289)
    assert model["lambda"] == pytest.approx(2.328882120967809)
    assert shrink["tau_asset_squared"] == pytest.approx(4.710348565855759)
    assert shrink["within_residual_variance"] == pytest.approx(26.378787878787882)
    drivers = result["candidate"].query("entity_type == 'driver'")
    four = drivers[drivers["observed_sprint_count"].eq(4)]
    assert np.allclose(four["empirical_bayes_weight"], 0.416658964445934)
    assert _row(result, "Pierre Gasly")["empirical_bayes_weight"] == pytest.approx(
        0.348830011135972
    )
    assert _row(result, "Carlos Sainz")["empirical_bayes_weight"] == pytest.approx(
        0.263151750059175
    )


def test_negative_sprint_bonuses_are_retained(result):
    observations = result["prepared"]["observations"]
    negative = observations[
        observations["observation_valid"] & observations["extra_sprint_points"].lt(0)
    ]
    assert not negative.empty
    assert _row(result, "Nico Hulkenberg")["observed_mean_sprint_bonus"] == pytest.approx(-2.75)
    assert _row(result, "Arvid Lindblad")["observed_mean_sprint_bonus"] == pytest.approx(-2.0)


def test_bottas_does_not_inherit_old_unstable_plus_thirteen(result):
    bottas = _row(result, "Valtteri Bottas")
    assert bottas["final_sprint_bonus"] == pytest.approx(1.273596724516509)
    assert bottas["final_sprint_bonus"] < 5.0
    assert bottas["final_sprint_bonus"] != pytest.approx(13.0)


def test_constructor_group_bonus_is_monotonic_in_selected_strength(result):
    constructors = result["candidate"].query("entity_type == 'constructor'").sort_values(
        ["selected_strength", "entity"]
    )
    assert CONSTRUCTOR_STRENGTH == "blend_form_0.75_price_0.25"
    expected_strength = (
        0.75 * constructors["form_percentile"]
        + 0.25 * constructors["price_percentile"]
    )
    assert np.allclose(constructors["selected_strength"], expected_strength)
    assert (constructors["group_bonus"].diff().dropna() >= -1e-12).all()
    assert result["models"]["constructor"]["lambda"] >= 0


def test_future_sprint_event_effect_is_zero_and_not_modelled(result):
    for model_name in ("driver_model", "constructor_model"):
        model = result["model_json"][model_name]
        assert model["future_event_effect"] == 0.0
        assert "event_effects" not in model
    assert (result["candidate"]["future_event_effect"] == 0.0).all()


def test_drivers_and_constructors_remain_separate(result):
    candidate = result["candidate"]
    assert set(candidate["entity_type"]) == {"driver", "constructor"}
    assert DRIVER_STRENGTH != CONSTRUCTOR_STRENGTH
    assert set(candidate.query("entity_type == 'driver'")["entity_id"]).isdisjoint(
        set(candidate.query("entity_type == 'constructor'")["entity_id"])
    )
    assert result["models"]["driver"]["observation_count"] == 85
    assert result["models"]["constructor"]["observation_count"] == 43


def test_json_is_explicitly_research_only(result):
    model = result["model_json"]
    assert model["research_only"] is True
    assert model["production_approved"] is False
    assert model["driver_model"]["method"] == "empirical_bayes_shrunk_personal_mean"
    assert model["constructor_model"]["method"] == "strength_only"


def test_script_output_is_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run_build(CANONICAL, SCHEDULE, MARKET, PREVIOUS, first)
    run_build(CANONICAL, SCHEDULE, MARKET, PREVIOUS, second)
    expected = {
        "REPORT.md", "final_candidate.csv", "final_candidate.json",
        "driver_shrinkage_details.csv", "constructor_strength_details.csv",
        "comparison_to_previous_models.csv", "sanity_checks.csv",
        "included_events.csv", "normalised_history.csv", "sprint_observations.csv",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert {path.name for path in second.iterdir()} == expected
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_production_modules_do_not_import_script_or_candidate_json():
    forbidden = (
        "build_2026_sprint_final_candidate",
        "2026_sprint_final_candidate",
        "final_candidate.json",
    )
    paths = [ROOT / "streamlit_app.py", *sorted((ROOT / "f1fantasy").glob("*.py"))]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert all(value not in text for value in forbidden), path
