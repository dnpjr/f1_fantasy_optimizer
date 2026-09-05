from pathlib import Path

import pandas as pd
import pytest

from f1fantasy.app_core import (
    HISTORY_MODE_ALL_SUPPORTED,
    HISTORY_MODE_CURRENT_SEASON_ONLY,
    OBJECTIVE_COMBINED,
    OBJECTIVE_POINTS_ONLY,
)
from f1fantasy.optimize import TeamSolution
from scripts.compare_sprint_shadow_to_production import (
    COMPLETED_ROUNDS,
    CURRENT_SEASON,
    build_asset_comparison,
    derive_scenario,
    deterministic_budget,
    load_offline_snapshot,
    optimizer_input_copies,
    optimizer_point_copies,
    run_optimizer_comparison,
    score_solution,
)


@pytest.fixture(scope="module")
def comparison_cases():
    snapshot, metadata = load_offline_snapshot()
    scenarios = {
        (1.0, HISTORY_MODE_ALL_SUPPORTED): derive_scenario(
            snapshot, 1.0, HISTORY_MODE_ALL_SUPPORTED
        ),
        (0.85, HISTORY_MODE_ALL_SUPPORTED): derive_scenario(
            snapshot, 0.85, HISTORY_MODE_ALL_SUPPORTED
        ),
        (0.85, HISTORY_MODE_CURRENT_SEASON_ONLY): derive_scenario(
            snapshot, 0.85, HISTORY_MODE_CURRENT_SEASON_ONLY
        ),
    }
    return snapshot, metadata, scenarios


@pytest.fixture(scope="module")
def primary_run(comparison_cases):
    _, _, scenarios = comparison_cases
    scenario = scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)]
    budget, _ = deterministic_budget(scenario.model.drivers, scenario.model.constructors)
    teams, cross_scores, summary = run_optimizer_comparison(
        scenario, budget, OBJECTIVE_POINTS_ONLY
    )
    return scenario, budget, teams, cross_scores, summary


def test_asset_comparison_never_mutates_production_ev(comparison_cases):
    _, _, scenarios = comparison_cases
    model = scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)].model
    drivers_before = model.drivers.copy(deep=True)
    constructors_before = model.constructors.copy(deep=True)

    comparison = build_asset_comparison(model.drivers, model.constructors)

    pd.testing.assert_frame_equal(model.drivers, drivers_before)
    pd.testing.assert_frame_equal(model.constructors, constructors_before)
    pd.testing.assert_series_equal(
        comparison.set_index(["entity_type", "asset_id"])["production_ev"].sort_index(),
        pd.concat(
            [
                drivers_before.assign(entity_type="driver", asset_id=drivers_before["id"].astype(str)),
                constructors_before.assign(
                    entity_type="constructor", asset_id=constructors_before["id"].astype(str)
                ),
            ]
        ).set_index(["entity_type", "asset_id"])["baseline_expected_points"].sort_index(),
        check_names=False,
        check_dtype=False,
    )


def test_optimizer_preparation_leaves_production_input_unchanged(comparison_cases):
    _, _, scenarios = comparison_cases
    model = scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)].model
    drivers_before = model.drivers.copy(deep=True)
    constructors_before = model.constructors.copy(deep=True)

    optimizer_input_copies(model.drivers, model.constructors, OBJECTIVE_COMBINED)

    pd.testing.assert_frame_equal(model.drivers, drivers_before)
    pd.testing.assert_frame_equal(model.constructors, constructors_before)


def test_shadow_optimizer_copy_replaces_only_points_field(comparison_cases):
    _, _, scenarios = comparison_cases
    model = scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)].model
    production_d, production_c, shadow_d, shadow_c = optimizer_point_copies(
        model.drivers, model.constructors
    )

    for production, shadow in ((production_d, shadow_d), (production_c, shadow_c)):
        changed = [column for column in production if not production[column].equals(shadow[column])]
        assert changed == ["exp_score"]
        pd.testing.assert_series_equal(
            shadow["exp_score"], shadow["shadow_sprint_ev"], check_names=False
        )


def test_prices_and_optimizer_constraints_are_identical(primary_run):
    scenario, budget, teams, _, _ = primary_run
    production_d, production_c, shadow_d, shadow_c = optimizer_point_copies(
        scenario.model.drivers, scenario.model.constructors
    )
    for production, shadow in ((production_d, shadow_d), (production_c, shadow_c)):
        for column in ("id", "price", "expected_price_gain", "projected_price"):
            pd.testing.assert_series_equal(
                production[column], shadow[column], check_names=False, check_dtype=False
            )

    invariant_columns = [
        "budget",
        "driver_slots",
        "constructor_slots",
        "drs_multiplier",
        "chip",
        "locked_driver_count",
        "excluded_driver_count",
        "locked_constructor_count",
        "excluded_constructor_count",
        "transfer_constraint",
    ]
    assert teams[invariant_columns].drop_duplicates().to_dict("records") == [
        {
            "budget": budget,
            "driver_slots": 5,
            "constructor_slots": 2,
            "drs_multiplier": 2.0,
            "chip": "none",
            "locked_driver_count": 0,
            "excluded_driver_count": 0,
            "locked_constructor_count": 0,
            "excluded_constructor_count": 0,
            "transfer_constraint": "not_applicable_fresh_team_optimisation",
        }
    ]


def test_fixed_team_is_cross_scored_under_both_models():
    comparison = pd.DataFrame(
        [
            {"entity_type": "driver", "asset_id": "d1", "production_ev": 1.0, "shadow_sprint_ev": 3.0},
            {"entity_type": "driver", "asset_id": "d2", "production_ev": 2.0, "shadow_sprint_ev": 5.0},
            {"entity_type": "constructor", "asset_id": "c1", "production_ev": 10.0, "shadow_sprint_ev": 20.0},
        ]
    )
    solution = TeamSolution(
        drivers=pd.DataFrame([{"id": "d1", "name": "One"}, {"id": "d2", "name": "Two"}]),
        constructors=pd.DataFrame([{"id": "c1", "name": "Team"}]),
        boosted_driver="Two",
        no_negative=False,
        limitless=False,
        total_cost=0.0,
        expected_score=0.0,
    )

    assert score_solution(solution, comparison, "production_ev") == pytest.approx(15.0)
    assert score_solution(solution, comparison, "shadow_sprint_ev") == pytest.approx(33.0)


def test_round_12_scoring_cannot_enter_analysis(comparison_cases):
    _, metadata, scenarios = comparison_cases
    scenario = scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)]
    diagnostics = scenario.model.diagnostics["approved_sprint_ev_shadow"]

    assert metadata["recorded_rounds"] == list(COMPLETED_ROUNDS)
    assert diagnostics["selected_2026_race_keys"] == [
        (CURRENT_SEASON, round_no) for round_no in COMPLETED_ROUNDS
    ]
    assert (CURRENT_SEASON, 12) not in diagnostics["selected_2026_race_keys"]
    assert diagnostics["upcoming_event"] == {"season": CURRENT_SEASON, "round": 12}


def test_p_1_and_p_085_use_expected_shadow_values(comparison_cases):
    _, _, scenarios = comparison_cases
    expected = {
        1.0: (35.36363636363637, 10.272284057440586, 45.63592042107695),
        0.85: (31.3986736544852, 9.793855032963956, 41.192528687449155),
    }
    for decay, values in expected.items():
        row = scenarios[(decay, HISTORY_MODE_ALL_SUPPORTED)].comparison.query(
            "entity == 'Kimi Antonelli' and entity_type == 'driver'"
        ).iloc[0]
        assert row["shadow_normal_ev"] == pytest.approx(values[0])
        assert row["shadow_sprint_bonus"] == pytest.approx(values[1])
        assert row["shadow_sprint_ev"] == pytest.approx(values[2])


def test_driver_and_constructor_rankings_remain_separate(comparison_cases):
    _, metadata, scenarios = comparison_cases
    comparison = scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)].comparison
    drivers = comparison[comparison["entity_type"].eq("driver")]
    constructors = comparison[comparison["entity_type"].eq("constructor")]

    assert len(drivers) == metadata["driver_count"] == 22
    assert len(constructors) == metadata["constructor_count"] == 11
    assert sorted(drivers["production_rank"]) == list(range(1, 23))
    assert sorted(constructors["production_rank"]) == list(range(1, 12))
    assert drivers["asset_id"].is_unique
    assert constructors["asset_id"].is_unique


def test_history_modes_are_distinctly_labelled_with_same_2026_shadow(comparison_cases):
    _, _, scenarios = comparison_cases
    all_supported = scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)]
    current_only = scenarios[(0.85, HISTORY_MODE_CURRENT_SEASON_ONLY)]

    assert all_supported.history_mode == HISTORY_MODE_ALL_SUPPORTED
    assert current_only.history_mode == HISTORY_MODE_CURRENT_SEASON_ONLY
    pd.testing.assert_series_equal(
        all_supported.comparison.set_index(["entity_type", "asset_id"])["shadow_sprint_ev"].sort_index(),
        current_only.comparison.set_index(["entity_type", "asset_id"])["shadow_sprint_ev"].sort_index(),
    )
    assert not all_supported.comparison["production_ev"].equals(
        current_only.comparison["production_ev"]
    )


def test_comparison_script_is_not_imported_by_production_startup():
    root = Path(__file__).resolve().parents[1]
    production_paths = [root / "streamlit_app.py", *sorted((root / "f1fantasy").glob("*.py"))]
    needle = "compare_sprint_shadow_to_production"

    assert all(needle not in path.read_text(encoding="utf-8") for path in production_paths)
