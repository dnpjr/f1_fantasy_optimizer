import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyse_2026_sprint_bonus import sprint_only_target
from scripts.analyse_2026_sprint_partial_pooling import (
    attach_strength,
    build_strength_definitions,
    complete_observation_grid,
    fit_penalised_model,
    leave_one_sprint_out,
    predict_model,
    run_analysis,
)
from scripts.calibrate_asset_sprint_adjustments import (
    build_baselines,
    build_normalised_history,
    build_sprint_observations,
    prepare_calibration_data,
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
    events, annotated, prices, metadata = prepare_calibration_data(CANONICAL, SCHEDULE, MARKET)
    observations = build_sprint_observations(annotated, prices).rename(
        columns={"included_in_regression": "observation_valid"}
    )
    history = build_normalised_history(annotated, prices)
    baselines = build_baselines(history, 0.8)
    observations = complete_observation_grid(observations, events, baselines)
    definitions, diagnostics = build_strength_definitions(baselines)
    return events, annotated, prices, observations, history, baselines, definitions, diagnostics, metadata


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run_analysis(
        CANONICAL,
        SCHEDULE,
        MARKET,
        tmp_path_factory.mktemp("partial_pooling"),
    )


def _synthetic_observations(slope=-2.0):
    rows = []
    for asset_index, asset in enumerate(("a", "b", "c")):
        strength = float(asset_index)
        for round_number in (2, 4, 5):
            rows.append(
                {
                    "entity_type": "driver",
                    "entity_id": asset,
                    "entity_name": asset.upper(),
                    "round": round_number,
                    "strength": strength,
                    "form_percentile": (asset_index + 1) / 3,
                    "extra_sprint_points": 10 + slope * strength + (round_number - 4),
                    "observation_valid": True,
                }
            )
    return pd.DataFrame(rows)


def test_only_2026_enters_primary_fit(result):
    assert set(result["observations"]["season"]) == {2026}
    assert set(result["events"]["round"]) == set(range(1, 12))


def test_drivers_and_constructors_are_always_separate(result):
    assert set(result["fits"]) == {"driver", "constructor"}
    for entity_type, fit in result["fits"].items():
        expected = set(result["predictions"].loc[
            result["predictions"]["entity_type"].eq(entity_type), "entity_id"
        ])
        assert set(fit["asset_effects"]) == expected


def test_target_is_only_sprint_plus_sprint_qualifying():
    frame = pd.DataFrame(
        {
            "sprint_points": [4.0, -3.0, np.nan],
            "sprint_qualifying_points": [2.0, 0.0, np.nan],
            "race_points": [100.0, 100.0, 100.0],
            "qualifying_points": [-50.0, -50.0, -50.0],
        }
    )
    target = sprint_only_target(frame)
    assert target.iloc[:2].tolist() == [6.0, -3.0]
    assert pd.isna(target.iloc[2])


def test_sprint_normalised_form_is_calculated_correctly(prepared):
    _events, _annotated, _prices, _observations, history, baselines, _definitions, _diagnostics, _meta = prepared
    sprint = history[history["weekend_format"].eq("sprint") & history["extra_sprint_points"].notna()]
    assert np.allclose(
        sprint["normalised_score"],
        sprint["fantasy_points_total"] - sprint["extra_sprint_points"],
    )
    asset = baselines.iloc[0]
    source = history[
        history["entity_type"].eq(asset["entity_type"])
        & history["canonical_entity_id"].eq(asset["entity_id"])
    ]
    assert asset["current_normal_baseline"] == pytest.approx(source["normalised_score"].mean())


def test_round_12_cannot_enter_scoring_calibration(prepared):
    _events, annotated, prices, *_rest = prepared
    injected = annotated.iloc[[0]].copy()
    injected["round"] = 12
    injected["fantasy_points_total"] = 999999.0
    history = build_normalised_history(pd.concat([annotated, injected]), prices)
    assert history["round"].max() == 11


def test_negative_observed_bonuses_remain_valid(prepared):
    observations = prepared[3]
    negative = observations[observations["extra_sprint_points"].lt(0)]
    assert not negative.empty
    assert negative["observation_valid"].all()


def test_asset_effects_are_centred_and_shrunk():
    observations = _synthetic_observations(slope=1.0)
    low_penalty = fit_penalised_model(
        observations, include_strength=True, include_asset=True, include_event=False,
        asset_penalty=1.0,
    )
    high_penalty = fit_penalised_model(
        observations, include_strength=True, include_asset=True, include_event=False,
        asset_penalty=16.0,
    )
    assert sum(low_penalty["asset_effects"].values()) == pytest.approx(0.0, abs=1e-10)
    assert np.linalg.norm(list(high_penalty["asset_effects"].values())) <= np.linalg.norm(
        list(low_penalty["asset_effects"].values())
    )


def test_event_effects_are_centred(result):
    for fit in result["fits"].values():
        assert sum(fit["event_effects"].values()) == pytest.approx(0.0, abs=1e-10)


def test_group_strength_coefficient_is_constrained_nonnegative():
    observations = _synthetic_observations(slope=-3.0)
    fit = fit_penalised_model(
        observations, include_strength=True, include_asset=False, include_event=False,
        constrain_strength=True,
    )
    assert fit["lambda"] == 0.0
    assert fit["lambda_forced_zero"] is True


def test_personal_effects_may_be_negative(result):
    assert (result["effects"]["asset_effect_u"] < 0).any()


def test_leave_one_sprint_out_removes_complete_intended_event(prepared):
    observations, definitions = prepared[3], prepared[6]
    drivers = attach_strength(
        observations[observations["entity_type"].eq("driver")], definitions, "z_form"
    )
    drivers["form_tier"] = pd.cut(
        drivers["form_percentile"], [0, 1 / 3, 2 / 3, 1],
        labels=["low", "middle", "high"], include_lowest=True,
    ).astype(str)
    heldout = leave_one_sprint_out(drivers, model_name="full_partial_pooling")
    for row in heldout.itertuples(index=False):
        assert str(row.excluded_round) not in row.training_rounds.split(",")
    assert set(heldout["excluded_round"]) == {2, 4, 5, 9}


def test_price_form_blend_calculations_are_correct(prepared):
    definitions = prepared[6]
    blend = definitions[definitions["strength_definition"].eq("blend_form_0.75_price_0.25")]
    expected = 0.75 * blend["form_percentile"] + 0.25 * blend["price_percentile"]
    assert np.allclose(blend["strength"], expected)


def test_unknown_future_sprint_uses_zero_event_effect(prepared):
    observations, definitions = prepared[3], prepared[6]
    drivers = attach_strength(
        observations[observations["entity_type"].eq("driver")], definitions, "z_form"
    )
    fit = fit_penalised_model(
        drivers, include_strength=True, include_asset=True, include_event=True,
        asset_penalty=4.0, event_penalty=4.0,
    )
    row = drivers.iloc[[0]]
    future = predict_model(fit, row, include_event_effect=False)[0]
    expected = fit["mu"] + fit["lambda"] * row.iloc[0]["strength"] + fit["asset_effects"][row.iloc[0]["entity_id"]]
    assert future == pytest.approx(expected)
    assert future != pytest.approx(predict_model(fit, row, include_event_effect=True)[0])


def test_constructor_components_are_not_rebuilt_from_driver_values(prepared):
    observations = prepared[3]
    constructors = observations[observations["entity_type"].eq("constructor")]
    valid = constructors[constructors["observation_valid"]]
    assert set(valid["data_source"]) == {"Formula 1 Fantasy playerstats"}
    assert len(constructors) == 44
    assert constructors["entity_id"].nunique() == 11
    assert len(constructors[~constructors["observation_valid"]]) == 1


def test_research_model_json_is_marked_research_only(tmp_path):
    run_analysis(CANONICAL, SCHEDULE, MARKET, tmp_path)
    model = json.loads((tmp_path / "research_model.json").read_text())
    assert model["research_only"] is True
    assert model["future_event_effect"] == 0.0


def test_script_output_is_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run_analysis(CANONICAL, SCHEDULE, MARKET, first)
    run_analysis(CANONICAL, SCHEDULE, MARKET, second)
    assert sorted(path.name for path in first.iterdir()) == sorted(path.name for path in second.iterdir())
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_no_production_startup_module_references_script_or_research_model():
    forbidden = ("analyse_2026_sprint_partial_pooling", "2026_sprint_partial_pooling")
    paths = [ROOT / "streamlit_app.py", *sorted((ROOT / "f1fantasy").glob("*.py"))]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert all(value not in text for value in forbidden)
