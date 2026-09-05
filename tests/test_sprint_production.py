from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from f1fantasy import app_core
from f1fantasy.app_core import (
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    HISTORY_MODE_ALL_SUPPORTED,
    HISTORY_MODE_CURRENT_SEASON_ONLY,
    apply_probabilistic_price_change_model,
    run_optimizer,
)
from f1fantasy.race_selection import RaceKey
from f1fantasy.sprint_shadow import (
    DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH,
    active_sprint_calibration_version,
    apply_sprint_production_adjustment,
    calculate_sprint_production_adjustment,
    driver_personal_weight,
    load_sprint_production_calibration,
)
from f1fantasy.weekend_state import EventKey
from scripts import recalibrate_sprint_ev
from scripts.compare_sprint_shadow_to_production import (
    derive_scenario,
    deterministic_budget,
    load_offline_snapshot,
)


def _schedule(*, sprint: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 1,
                "raceName": "Completed",
                "circuitName": "Completed Circuit",
                "date": "2026-07-01",
                "time": "14:00:00Z",
                "sprint_date": None,
                "sprint_time": None,
                "sprint_qualifying_date": None,
                "sprint_qualifying_time": None,
            },
            {
                "season": 2026,
                "round": 2,
                "raceName": "Upcoming",
                "circuitName": "Upcoming Circuit",
                "date": "2026-08-20",
                "time": "14:00:00Z",
                "sprint_date": "2026-08-19" if sprint else None,
                "sprint_time": "10:00:00Z" if sprint else None,
                "sprint_qualifying_date": None,
                "sprint_qualifying_time": None,
            },
        ]
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 1,
                "entity_type": "driver",
                "canonical_entity_id": "hamilton",
                "source_entity_id": "44",
                "name": "Lewis Hamilton",
                "fantasy_points_total": 20.0,
                "sprint_points": pd.NA,
                "sprint_qualifying_points": pd.NA,
            },
            {
                "season": 2026,
                "round": 1,
                "entity_type": "constructor",
                "canonical_entity_id": "ferrari",
                "source_entity_id": "25",
                "name": "Ferrari",
                "fantasy_points_total": 40.0,
                "sprint_points": pd.NA,
                "sprint_qualifying_points": pd.NA,
            },
        ]
    )


def _assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            [{"id": "44", "name": "Lewis Hamilton", "price": 25.0, "next_race_expected_points": 100.0}]
        ),
        pd.DataFrame(
            [{"id": "25", "name": "Ferrari", "price": 26.6, "next_race_expected_points": 200.0}]
        ),
    )


def _calculate(*, sprint: bool, drivers=None, constructors=None):
    default_drivers, default_constructors = _assets()
    return calculate_sprint_production_adjustment(
        _history(),
        _schedule(sprint=sprint),
        default_drivers if drivers is None else drivers,
        default_constructors if constructors is None else constructors,
        [RaceKey(2026, 1)],
        1.0,
        EventKey(2026, 2),
        production_history_mode=HISTORY_MODE_ALL_SUPPORTED,
    )


def _frozen_round_11_snapshot():
    from unittest.mock import patch
    from scripts import compare_sprint_shadow_to_production as comparison
    recorded = pd.read_csv(comparison.DEFAULT_CANONICAL_DATASET_PATH, dtype={"source_entity_id": str})
    recorded = recorded[~(recorded["season"].eq(2026) & recorded["round"].gt(11))]
    current = recorded[recorded["season"].eq(2026)].sort_values("round").drop_duplicates(
        ["entity_type", "canonical_entity_id"], keep="last"
    )
    original_report = pd.read_csv(Path(__file__).resolve().parents[1] / "reports/2026_sprint_final_candidate/final_candidate.csv").set_index("entity_id")
    prices = original_report["current_price"]
    def roster(kind, id_column):
        rows = current[current["entity_type"].eq(kind)].copy()
        rows["price"] = rows["canonical_entity_id"].map(prices)
        rows["name"] = rows["canonical_entity_id"].map(original_report["entity"])
        return rows.rename(columns={"source_entity_id": id_column, "abbreviation": "tla", "constructor_name": "team"})
    market = {"feed_round": 12, "verified_at_utc": "2026-08-05T00:00:00Z",
              "players": roster("driver", "playerId"), "teams": roster("constructor", "teamId")}
    with patch.object(comparison, "load_verified_market_cache", return_value=market), patch.object(comparison, "load_canonical_scores", return_value=recorded):
        snapshot, metadata = comparison.load_offline_snapshot()
        for recent in (snapshot.driver_recent_points, snapshot.constructor_recent_points):
            recent["id"] = recent["id"].astype(str)
        return snapshot, metadata


@pytest.fixture(scope="module")
def offline_scenarios():
    snapshot, _ = _frozen_round_11_snapshot()
    return {
        (1.0, HISTORY_MODE_ALL_SUPPORTED): derive_scenario(
            snapshot, 1.0, HISTORY_MODE_ALL_SUPPORTED
        ),
        (0.85, HISTORY_MODE_ALL_SUPPORTED): derive_scenario(
            snapshot, 0.85, HISTORY_MODE_ALL_SUPPORTED
        ),
        (1.0, HISTORY_MODE_CURRENT_SEASON_ONLY): derive_scenario(
            snapshot, 1.0, HISTORY_MODE_CURRENT_SEASON_ONLY
        ),
        (0.85, HISTORY_MODE_CURRENT_SEASON_ONLY): derive_scenario(
            snapshot, 0.85, HISTORY_MODE_CURRENT_SEASON_ONLY
        ),
    }


@pytest.fixture(scope="module")
def normal_offline_model():
    snapshot, _ = _frozen_round_11_snapshot()
    snapshot = deepcopy(snapshot)
    upcoming = pd.to_numeric(snapshot.schedule["round"], errors="coerce").eq(12)
    for column in ("sprint_date", "sprint_time", "sprint_qualifying_date", "sprint_qualifying_time"):
        if column in snapshot.schedule.columns:
            snapshot.schedule.loc[upcoming, column] = None
    return app_core.derive_model_data(
        snapshot,
        today="2026-08-10",
        effective_time="2026-08-10T12:00:00Z",
        historical_seasons_back=3,
        horizon_races=5,
        current_season_weight=1.0,
        past_season_weight=0.7,
        recency_decay=0.85,
        selected_race_preset="All",
        history_mode=HISTORY_MODE_ALL_SUPPORTED,
    )


def test_active_calibration_loads_exact_approved_formulas():
    calibration = load_sprint_production_calibration()
    assert calibration.model_version == "sprint_ev_2026_v2"
    assert calibration.calibration_status == "approved_production"
    assert calibration.calibration_season == 2026
    assert calibration.calibration_form_mean == pytest.approx(9.46745718050066)
    assert calibration.calibration_form_sd == pytest.approx(10.915416414446662)
    assert calibration.driver_group_intercept == pytest.approx(4.706549290132376)
    assert calibration.driver_group_slope == pytest.approx(2.266533074786191)
    assert calibration.driver_within_variance == pytest.approx(24.174999999999997)
    assert calibration.driver_tau_squared == pytest.approx(5.27333829160053)
    assert calibration.constructor_intercept == pytest.approx(0.9925779581375448)
    assert calibration.constructor_slope == pytest.approx(14.409013242204193)
    assert calibration.constructor_form_weight == 0.75
    assert calibration.constructor_price_weight == 0.25


def test_runtime_loader_does_not_fit_import_research_or_use_network(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("network request attempted")

    monkeypatch.setattr("requests.get", fail)
    before = set(sys.modules)
    assert active_sprint_calibration_version() == "sprint_ev_2026_v2"
    imported = set(sys.modules) - before
    assert not any(name.startswith("scripts.") for name in imported)


def test_normal_weekend_is_exact_baseline_with_zero_bonus():
    drivers, _ = _assets()
    result = _calculate(sprint=False)
    row = result.drivers.iloc[0]
    adjusted = apply_sprint_production_adjustment(drivers, result.drivers)
    assert row["sprint_weekend_format"] == "normal"
    assert row["sprint_bonus"] == 0.0
    assert row["sprint_adjusted_expected_points"] == row["baseline_expected_points"] == 100.0
    assert adjusted.loc[0, "next_race_expected_points"] == 100.0
    assert adjusted.loc[0, "exp_score"] == 100.0


def test_real_normal_weekend_preserves_every_ev_and_optimizer_result(normal_offline_model):
    drivers = normal_offline_model.drivers
    constructors = normal_offline_model.constructors
    for frame in (drivers, constructors):
        pd.testing.assert_series_equal(
            frame["next_race_expected_points"],
            frame["baseline_expected_points"],
            check_names=False,
        )
        assert frame["sprint_bonus"].eq(0.0).all()
    budget, _ = deterministic_budget(drivers, constructors)
    final = run_optimizer(
        drivers, constructors, budget=budget, top_k=1,
        objective_col="exp_score", boost_col="exp_score"
    )[0]
    baseline_drivers = drivers.assign(exp_score=drivers["baseline_expected_points"])
    baseline_constructors = constructors.assign(exp_score=constructors["baseline_expected_points"])
    baseline = run_optimizer(
        baseline_drivers, baseline_constructors, budget=budget, top_k=1,
        objective_col="exp_score", boost_col="exp_score"
    )[0]
    assert final.drivers["id"].astype(str).tolist() == baseline.drivers["id"].astype(str).tolist()
    assert final.constructors["id"].astype(str).tolist() == baseline.constructors["id"].astype(str).tolist()
    assert final.expected_score == baseline.expected_score
    assert normal_offline_model.diagnostics["sprint_ev_production"]["upcoming_weekend_format"] == "normal"


def test_sprint_weekend_adds_bonus_to_production_baseline_not_calibration_form():
    drivers, _ = _assets()
    result = _calculate(sprint=True)
    calibration = load_sprint_production_calibration()
    group = calibration.driver_group_intercept + calibration.driver_group_slope * (
        (20.0 - calibration.calibration_form_mean) / calibration.calibration_form_sd
    )
    personal = next(row for row in calibration.driver_personal_history if row["entity_id"] == "hamilton")
    weight = driver_personal_weight(personal["observation_count"], calibration)
    bonus = weight * personal["personal_mean_bonus"] + (1.0 - weight) * group
    row = result.drivers.iloc[0]
    adjusted = apply_sprint_production_adjustment(drivers, result.drivers)
    assert row["baseline_expected_points"] == 100.0
    assert row["sprint_bonus"] == pytest.approx(bonus)
    assert row["sprint_bonus_driver_personal_component"] == pytest.approx(weight * personal["personal_mean_bonus"])
    assert row["sprint_bonus_driver_group_component"] == pytest.approx((1 - weight) * group)
    assert adjusted.loc[0, "next_race_expected_points"] == pytest.approx(100.0 + bonus)


def test_constructor_uses_2026_form_and_current_price_percentiles():
    history = pd.DataFrame(
        [
            {**_history().iloc[1].to_dict(), "canonical_entity_id": "ferrari", "source_entity_id": "25", "name": "Ferrari", "fantasy_points_total": 40.0},
            {**_history().iloc[1].to_dict(), "canonical_entity_id": "mercedes", "source_entity_id": "28", "name": "Mercedes", "fantasy_points_total": 20.0},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "25", "name": "Ferrari", "price": 20.0, "next_race_expected_points": 300.0},
            {"id": "28", "name": "Mercedes", "price": 40.0, "next_race_expected_points": 10.0},
        ]
    )
    result = calculate_sprint_production_adjustment(
        history,
        _schedule(sprint=True),
        pd.DataFrame(columns=["id", "name", "price", "next_race_expected_points"]),
        constructors,
        [RaceKey(2026, 1)],
        1.0,
        EventKey(2026, 2),
        production_history_mode=HISTORY_MODE_ALL_SUPPORTED,
    )
    ferrari = result.constructors.set_index("id").loc["25"]
    strength = 0.75 * 1.0 + 0.25 * 0.5
    calibration = load_sprint_production_calibration()
    expected = calibration.constructor_intercept + calibration.constructor_slope * strength
    assert ferrari["baseline_expected_points"] == 300.0
    assert ferrari["sprint_constructor_strength"] == strength
    assert ferrari["sprint_bonus"] == pytest.approx(expected)


def test_missing_driver_history_uses_group_only_bonus():
    unknown = pd.DataFrame(
        [{"id": "999", "name": "New Driver", "price": 5.0, "next_race_expected_points": 12.0}]
    )
    history = _history().copy()
    history.loc[history["entity_type"].eq("driver"), "canonical_entity_id"] = "new_driver"
    history.loc[history["entity_type"].eq("driver"), "source_entity_id"] = "999"
    history.loc[history["entity_type"].eq("driver"), "name"] = "New Driver"
    result = calculate_sprint_production_adjustment(
        history,
        _schedule(sprint=True),
        unknown,
        pd.DataFrame(columns=["id", "name", "price", "next_race_expected_points"]),
        [RaceKey(2026, 1)],
        1.0,
        EventKey(2026, 2),
        production_history_mode=HISTORY_MODE_ALL_SUPPORTED,
    )
    row = result.drivers.iloc[0]
    assert row["sprint_bonus_driver_weight"] == 0.0
    assert row["sprint_bonus"] == pytest.approx(row["sprint_bonus_driver_group_component"])
    assert row["sprint_bonus_status"] == "group_only_no_personal_history"


def test_adjustment_is_guarded_against_double_application():
    drivers, _ = _assets()
    result = _calculate(sprint=True)
    adjusted = apply_sprint_production_adjustment(drivers, result.drivers)
    with pytest.raises(ValueError, match="already been applied"):
        apply_sprint_production_adjustment(adjusted, result.drivers)


def test_calculation_and_application_do_not_mutate_inputs():
    history = _history()
    schedule = _schedule(sprint=True)
    drivers, constructors = _assets()
    originals = [frame.copy(deep=True) for frame in (history, schedule, drivers, constructors)]
    result = calculate_sprint_production_adjustment(
        history,
        schedule,
        drivers,
        constructors,
        [RaceKey(2026, 1)],
        1.0,
        EventKey(2026, 2),
        production_history_mode=HISTORY_MODE_ALL_SUPPORTED,
    )
    apply_sprint_production_adjustment(drivers, result.drivers)
    for actual, expected in zip((history, schedule, drivers, constructors), originals):
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("decay", "history_mode", "expected_baseline"),
    [
        (1.0, HISTORY_MODE_ALL_SUPPORTED, 29.564882430644325),
        (0.85, HISTORY_MODE_ALL_SUPPORTED, 27.060159067926048),
        (1.0, HISTORY_MODE_CURRENT_SEASON_ONLY, 39.0),
        (0.85, HISTORY_MODE_CURRENT_SEASON_ONLY, 34.81319426119082),
    ],
)
def test_real_baseline_history_mode_decay_and_race_selection_are_preserved(
    offline_scenarios, decay, history_mode, expected_baseline
):
    model = offline_scenarios[(decay, history_mode)].model
    row = model.drivers.set_index("name").loc["Kimi Antonelli"]
    assert row["baseline_expected_points"] < expected_baseline
    if history_mode == HISTORY_MODE_CURRENT_SEASON_ONLY:
        snapshot, _ = _frozen_round_11_snapshot()
        scores = snapshot.historical_fantasy_scores
        scores = scores[scores["season"].eq(2026) & scores["canonical_entity_id"].eq("antonelli")].sort_values("round", ascending=False)
        bonus = scores[["sprint_points", "sprint_qualifying_points"]].sum(axis=1, min_count=1)
        normal = scores["fantasy_points_total"].copy()
        sprint_rows = scores["round"].isin([2, 4, 5, 9])
        normal.loc[sprint_rows] -= bonus.loc[sprint_rows]
        weights = decay ** np.arange(len(normal))
        assert row["baseline_expected_points"] == pytest.approx(np.average(normal, weights=weights))
    assert row["next_race_expected_points"] == pytest.approx(
        row["baseline_expected_points"] + row["sprint_bonus"]
    )
    assert model.diagnostics["history_mode"] == history_mode
    assert model.diagnostics["sprint_ev_production"]["selected_2026_race_keys"] == [
        (2026, round_no) for round_no in range(1, 12)
    ]


def test_approved_p_one_bonus_sanity_values(offline_scenarios):
    model = offline_scenarios[(1.0, HISTORY_MODE_ALL_SUPPORTED)].model
    drivers = model.drivers.set_index("name")
    constructors = model.constructors.set_index("name")
    calibration = load_sprint_production_calibration()
    assert drivers.loc["Kimi Antonelli", "sprint_bonus"] > drivers.loc["Nico Hulkenberg", "sprint_bonus"]
    assert constructors.loc["Mercedes", "sprint_bonus"] == pytest.approx(
        calibration.constructor_intercept + calibration.constructor_slope
    )
    assert drivers["sprint_calibration_version"].eq("sprint_ev_2026_v2").all()


def test_sprint_optimizer_consumes_final_ev_once_with_same_prices_and_budget(offline_scenarios):
    scenario = offline_scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)]
    drivers = scenario.model.drivers.copy(deep=True)
    constructors = scenario.model.constructors.copy(deep=True)
    budget, _ = deterministic_budget(drivers, constructors)
    baseline_drivers = drivers.copy(deep=True)
    baseline_constructors = constructors.copy(deep=True)
    baseline_drivers["exp_score"] = baseline_drivers["baseline_expected_points"]
    baseline_constructors["exp_score"] = baseline_constructors["baseline_expected_points"]
    before = run_optimizer(
        baseline_drivers, baseline_constructors, budget=budget, top_k=1,
        objective_col="exp_score", boost_col="exp_score"
    )[0]
    after = run_optimizer(
        drivers, constructors, budget=budget, top_k=1,
        objective_col="exp_score", boost_col="exp_score"
    )[0]
    for solution in (before, after):
        assert len(solution.drivers) == 5
        assert len(solution.constructors) == 2
        assert solution.drivers["id"].is_unique
        assert solution.constructors["id"].is_unique
    assert before.total_cost <= budget and after.total_cost <= budget
    pd.testing.assert_series_equal(drivers["price"], scenario.model.drivers["price"])
    assert (drivers["exp_score"] == drivers["next_race_expected_points"]).all()
    assert not (
        drivers["exp_score"]
        == drivers["baseline_expected_points"] + 2.0 * drivers["sprint_bonus"]
    ).any()


def test_price_growth_keeps_its_existing_dependency_on_final_next_race_ev(offline_scenarios):
    drivers = offline_scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)].model.drivers
    projected = apply_probabilistic_price_change_model(
        drivers,
        DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        predicted_points_col="next_race_expected_points",
    )
    pd.testing.assert_series_equal(
        projected["price_change_predicted_next"],
        pd.to_numeric(drivers["next_race_expected_points"], errors="coerce"),
        check_names=False,
    )


def test_calibration_version_only_enters_sprint_prediction_identity(monkeypatch):
    drivers, constructors = _assets()
    base = app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2026,
        requested_seasons=(2026,),
        loaded_seasons=(2026,),
        season_load_failures={},
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=_schedule(sprint=True),
        players=drivers,
        teams=constructors,
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={},
    )
    monkeypatch.setattr(app_core, "active_sprint_calibration_version", lambda: "v1")
    sprint_v1 = app_core.model_settings_signature(base, 0, 1, 1, 0.7, 0.85, "2026-08-10")
    monkeypatch.setattr(app_core, "active_sprint_calibration_version", lambda: "v2")
    sprint_v2 = app_core.model_settings_signature(base, 0, 1, 1, 0.7, 0.85, "2026-08-10")
    assert sprint_v1 != sprint_v2

    normal = deepcopy(base)
    normal.schedule = _schedule(sprint=False)
    normal_v2 = app_core.model_settings_signature(normal, 0, 1, 1, 0.7, 0.85, "2026-08-10")
    monkeypatch.setattr(app_core, "active_sprint_calibration_version", lambda: "v3")
    normal_v3 = app_core.model_settings_signature(normal, 0, 1, 1, 0.7, 0.85, "2026-08-10")
    assert normal_v2 == normal_v3


def _fake_recalibration_result() -> dict:
    return {
        "prepared": {"source_data_version": "candidate-data"},
        "model_json": {"generated_at": "2026-08-10T00:00:00Z", "script_version": "candidate-fit", "completed_rounds": [1, 2], "sprint_rounds": [2]},
        "models": {
            "driver": {
                "strength_parameters": {"normal_ev_mean": 10.0, "normal_ev_population_sd": 2.0},
                "mu": 5.0,
                "lambda": 2.0,
                "shrinkage": {"within_residual_variance": 20.0, "tau_asset_squared": 4.0},
            },
            "constructor": {"mu": 1.0, "lambda": 16.0},
        },
        "candidate": pd.DataFrame(
            [{"entity_type": "driver", "entity_id": "new", "entity": "New Driver", "observed_mean_sprint_bonus": 3.0, "observed_sprint_count": 1}]
        ),
    }


def test_candidate_generation_does_not_alter_active_calibration(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    active.write_bytes(DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH.read_bytes())
    before = active.read_bytes()
    monkeypatch.setattr(recalibrate_sprint_ev, "run_build", lambda *_args, **_kwargs: _fake_recalibration_result())
    result = recalibrate_sprint_ev.generate_candidate(
        tmp_path / "candidate", active_path=active
    )
    assert active.read_bytes() == before
    assert result["candidate"]["calibration_status"] == "candidate"
    assert result["candidate"]["model_version"] == "sprint_ev_2026_v3"


def test_explicit_promotion_archives_and_atomically_versions(tmp_path):
    active = tmp_path / "active.json"
    active.write_bytes(DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH.read_bytes())
    candidate_raw = json.loads(active.read_text(encoding="utf-8"))
    candidate_raw["model_version"] = "sprint_ev_2026_v3"
    candidate_raw["calibration_status"] = "candidate"
    candidate_raw["driver"]["group_slope"] += 0.01
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(candidate_raw), encoding="utf-8")
    archive = tmp_path / "archive"

    changes = recalibrate_sprint_ev.promote_candidate(
        candidate, active_path=active, archive_dir=archive
    )

    promoted = load_sprint_production_calibration(active)
    assert promoted.model_version == "sprint_ev_2026_v3"
    assert (archive / "sprint_ev_2026_v2.json").exists()
    assert any(change["field"] == "driver.group_slope" for change in changes)


def test_production_startup_imports_no_recalibration_script():
    root = Path(__file__).resolve().parents[1]
    needle = "recalibrate_sprint_ev"
    production_paths = [root / "streamlit_app.py", *sorted((root / "f1fantasy").glob("*.py"))]
    assert all(needle not in path.read_text(encoding="utf-8") for path in production_paths)


def test_normal_weekend_never_enters_sprint_baseline_normalisation(monkeypatch, normal_offline_model):
    snapshot, _ = _frozen_round_11_snapshot()
    upcoming = snapshot.schedule['round'].eq(12)
    for column in ('sprint_date', 'sprint_time', 'sprint_qualifying_date', 'sprint_qualifying_time'):
        snapshot.schedule.loc[upcoming, column] = None
    def forbidden(*args, **kwargs):
        raise AssertionError('Normal weekend entered Sprint normalisation')
    monkeypatch.setattr(app_core, 'normalise_sprint_baseline_inputs', forbidden)
    model = app_core.derive_model_data(
        snapshot, today='2026-08-10', effective_time='2026-08-10T12:00:00Z',
        historical_seasons_back=3, horizon_races=5, current_season_weight=1.0,
        past_season_weight=0.7, recency_decay=0.85, selected_race_preset='All',
        history_mode=HISTORY_MODE_ALL_SUPPORTED,
    )
    for actual, expected in ((model.drivers, normal_offline_model.drivers),
                             (model.constructors, normal_offline_model.constructors)):
        for column in ('exp_score', 'next_race_expected_points', 'horizon_expected_points'):
            pd.testing.assert_series_equal(actual[column], expected[column])
