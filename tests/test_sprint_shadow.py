from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd
import pytest

from f1fantasy.app_core import (
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    apply_probabilistic_price_change_model,
    build_transfer_recommendations,
    current_team_budget_from_selection,
    transfer_baseline,
)
from f1fantasy.optimize import optimize_top_k
from f1fantasy.race_selection import RaceKey
from f1fantasy.sprint_shadow import (
    DEFAULT_SPRINT_SHADOW_CALIBRATION_PATH,
    calculate_sprint_shadow,
    driver_personal_weight,
    load_sprint_shadow_calibration,
    normal_equivalent_history,
)
from f1fantasy.weekend_state import EventKey


CANONICAL_PATH = Path(
    "data/generated/historical_fantasy_scores_v3_recorded_2023_2026/"
    "historical_fantasy_scores_2023_2026.csv"
)
FINAL_CANDIDATE_PATH = Path("reports/2026_sprint_final_candidate/final_candidate.csv")


def _schedule(*, rounds=(1, 2, 3), sprint_rounds=(2,), season=2026):
    rows = []
    for round_no in rounds:
        sprint = round_no in sprint_rounds
        rows.append(
            {
                "season": season,
                "round": round_no,
                "raceName": f"Race {round_no}",
                "circuitName": f"Circuit {round_no}",
                "date": f"{season}-08-{round_no:02d}",
                "time": "14:00:00Z",
                "sprint_date": f"{season}-08-{round_no:02d}" if sprint else None,
                "sprint_time": "10:00:00Z" if sprint else None,
                "sprint_qualifying_date": None,
                "sprint_qualifying_time": None,
            }
        )
    return pd.DataFrame(rows)


def _history(rows):
    defaults = {
        "season": 2026,
        "entity_type": "driver",
        "canonical_entity_id": "hamilton",
        "source_entity_id": "44",
        "name": "Lewis Hamilton",
        "sprint_points": math.nan,
        "sprint_qualifying_points": math.nan,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _drivers(*rows):
    return pd.DataFrame(rows or ({"id": "44", "name": "Lewis Hamilton", "price": 25.0},))


def _constructors(*rows):
    return pd.DataFrame(rows or ({"id": "25", "name": "Ferrari", "price": 26.6},))


def _calculate(
    history,
    *,
    selected=(RaceKey(2026, 1), RaceKey(2026, 2)),
    decay=1.0,
    upcoming=EventKey(2026, 3),
    schedule=None,
    drivers=None,
    constructors=None,
    history_mode="all_supported",
):
    return calculate_sprint_shadow(
        history,
        schedule if schedule is not None else _schedule(),
        drivers if drivers is not None else _drivers(),
        constructors if constructors is not None else pd.DataFrame(columns=["id", "name", "price"]),
        selected,
        decay,
        upcoming,
        production_history_mode=history_mode,
    )


def test_frozen_calibration_loads_reviewed_version_and_22_driver_histories():
    calibration = load_sprint_shadow_calibration()
    assert calibration.model_version == "sprint_ev_shadow_2026_v1"
    assert calibration.source_research_model == "2026-sprint-final-candidate-v1"
    assert calibration.research_only is True
    assert len(calibration.driver_personal_history) == 22


def test_normal_weekend_recorded_total_is_unchanged():
    history = _history([{"round": 1, "fantasy_points_total": 12.0}])
    normalised, _ = normal_equivalent_history(history, _schedule(), [RaceKey(2026, 1)])
    assert normalised.loc[0, "normal_equivalent_score"] == 12.0


def test_sprint_weekend_removes_sprint_and_sprint_qualifying_points():
    history = _history(
        [{"round": 2, "fantasy_points_total": 30.0, "sprint_points": 6.0, "sprint_qualifying_points": 2.0}]
    )
    normalised, _ = normal_equivalent_history(history, _schedule(), [RaceKey(2026, 2)])
    assert normalised.loc[0, "shadow_sprint_only_points"] == 8.0
    assert normalised.loc[0, "normal_equivalent_score"] == 22.0


def test_available_sprint_component_is_used_when_other_component_is_not_applicable():
    history = _history(
        [{"round": 2, "fantasy_points_total": 30.0, "sprint_points": 6.0, "sprint_qualifying_points": math.nan}]
    )
    normalised, _ = normal_equivalent_history(history, _schedule(), [RaceKey(2026, 2)])
    assert normalised.loc[0, "normal_equivalent_score"] == 24.0
    assert normalised.loc[0, "normal_equivalent_status"] == "valid"


def test_genuine_zero_sprint_component_remains_a_valid_zero():
    history = _history(
        [{"round": 2, "fantasy_points_total": 0.0, "sprint_points": 0.0, "sprint_qualifying_points": math.nan}]
    )
    normalised, diagnostics = normal_equivalent_history(history, _schedule(), [RaceKey(2026, 2)])
    assert normalised.loc[0, "normal_equivalent_score"] == 0.0
    assert diagnostics["missing_sprint_component_observations"] == 0


def test_wholly_missing_sprint_component_is_not_silently_zero_filled():
    history = _history([{"round": 2, "fantasy_points_total": 10.0}])
    normalised, diagnostics = normal_equivalent_history(history, _schedule(), [RaceKey(2026, 2)])
    assert pd.isna(normalised.loc[0, "normal_equivalent_score"])
    assert normalised.loc[0, "normal_equivalent_status"] == "sprint_component_missing"
    assert diagnostics["missing_sprint_component_observations"] == 1


def test_p_one_shadow_normal_ev_is_arithmetic_mean_of_selected_normal_equivalents():
    history = _history(
        [
            {"round": 1, "fantasy_points_total": 10.0},
            {"round": 2, "fantasy_points_total": 30.0, "sprint_points": 10.0},
        ]
    )
    result = _calculate(history, decay=1.0)
    assert result.drivers.loc[0, "shadow_normal_ev"] == 15.0


def test_recency_decay_uses_contiguous_selected_positions():
    history = _history(
        [
            {"round": 1, "fantasy_points_total": 10.0},
            {"round": 3, "fantasy_points_total": 30.0},
        ]
    )
    result = _calculate(
        history,
        selected=(RaceKey(2026, 1), RaceKey(2026, 3)),
        decay=0.5,
        upcoming=EventKey(2026, 4),
        schedule=_schedule(rounds=(1, 3, 4), sprint_rounds=()),
    )
    assert result.drivers.loc[0, "shadow_normal_ev"] == pytest.approx((10 * 0.5 + 30) / 1.5)
    assert result.diagnostics["selected_2026_race_weights"] == {"2026:1": 0.5, "2026:3": 1.0}


def test_upcoming_event_scoring_is_excluded_even_if_erroneously_selected():
    history = _history(
        [
            {"round": 1, "fantasy_points_total": 10.0},
            {"round": 3, "fantasy_points_total": 999.0},
        ]
    )
    result = _calculate(
        history,
        selected=(RaceKey(2026, 1), RaceKey(2026, 3)),
        upcoming=EventKey(2026, 3),
    )
    assert result.drivers.loc[0, "shadow_normal_ev"] == 10.0
    assert result.diagnostics["selected_2026_race_keys"] == [(2026, 1)]


def test_normal_upcoming_weekend_has_zero_bonus_and_normal_shadow_total():
    history = _history([{"round": 1, "fantasy_points_total": 20.0}])
    result = _calculate(history, selected=(RaceKey(2026, 1),))
    row = result.drivers.iloc[0]
    assert row["shadow_weekend_format"] == "normal"
    assert row["shadow_sprint_bonus"] == 0.0
    assert row["shadow_sprint_ev"] == row["shadow_normal_ev"]


def test_normal_upcoming_constructor_needs_no_price_or_strength_for_zero_bonus():
    history = _history(
        [
            {
                "round": 1,
                "entity_type": "constructor",
                "canonical_entity_id": "ferrari",
                "source_entity_id": "25",
                "name": "Ferrari",
                "fantasy_points_total": 40.0,
            }
        ]
    )
    result = _calculate(
        history,
        selected=(RaceKey(2026, 1),),
        drivers=pd.DataFrame(columns=["id", "name", "price"]),
        constructors=_constructors({"id": "25", "name": "Ferrari", "price": math.nan}),
    )
    row = result.constructors.iloc[0]
    assert row["shadow_sprint_bonus"] == 0.0
    assert row["shadow_sprint_ev"] == 40.0
    assert row["shadow_status"] == "available"


def test_sprint_upcoming_driver_formula_matches_approved_coefficients():
    history = _history([{"round": 1, "fantasy_points_total": 20.0}])
    result = _calculate(
        history,
        selected=(RaceKey(2026, 1),),
        upcoming=EventKey(2026, 3),
        schedule=_schedule(sprint_rounds=(2, 3)),
    )
    calibration = load_sprint_shadow_calibration()
    group = calibration.driver_group_intercept + calibration.driver_group_slope * (
        (20.0 - calibration.calibration_form_mean) / calibration.calibration_form_sd
    )
    weight = driver_personal_weight(4, calibration)
    expected_bonus = weight * 7.0 + (1 - weight) * group
    row = result.drivers.iloc[0]
    assert row["shadow_group_bonus"] == pytest.approx(group)
    assert row["shadow_sprint_bonus"] == pytest.approx(expected_bonus)
    assert row["shadow_sprint_ev"] == pytest.approx(20.0 + expected_bonus)


@pytest.mark.parametrize(
    ("count", "expected"),
    [(4, 0.416658964446), (3, 0.348830011136), (2, 0.263151750059), (0, 0.0)],
)
def test_driver_empirical_bayes_weights(count, expected):
    assert driver_personal_weight(count, load_sprint_shadow_calibration()) == pytest.approx(expected)


def test_negative_personal_sprint_mean_remains_valid_and_unclipped():
    history = _history(
        [{"round": 1, "fantasy_points_total": 6.0, "canonical_entity_id": "arvid_lindblad", "source_entity_id": "7", "name": "Arvid Lindblad"}]
    )
    result = _calculate(
        history,
        selected=(RaceKey(2026, 1),),
        upcoming=EventKey(2026, 3),
        schedule=_schedule(sprint_rounds=(2, 3)),
        drivers=_drivers({"id": "7", "name": "Arvid Lindblad", "price": 8.0}),
    )
    assert result.drivers.loc[0, "shadow_personal_mean_bonus"] == -2.0
    assert result.drivers.loc[0, "shadow_personal_weight"] > 0


def test_constructor_strength_is_75_percent_form_and_25_percent_current_price():
    history = _history(
        [
            {"round": 1, "entity_type": "constructor", "canonical_entity_id": "ferrari", "source_entity_id": "25", "name": "Ferrari", "fantasy_points_total": 40.0},
            {"round": 1, "entity_type": "constructor", "canonical_entity_id": "mercedes", "source_entity_id": "28", "name": "Mercedes", "fantasy_points_total": 20.0},
        ]
    )
    constructors = _constructors(
        {"id": "25", "name": "Ferrari", "price": 20.0},
        {"id": "28", "name": "Mercedes", "price": 40.0},
    )
    result = _calculate(
        history,
        selected=(RaceKey(2026, 1),),
        upcoming=EventKey(2026, 3),
        schedule=_schedule(sprint_rounds=(2, 3)),
        drivers=pd.DataFrame(columns=["id", "name", "price"]),
        constructors=constructors,
    )
    ferrari = result.constructors.set_index("id").loc["25"]
    assert ferrari["shadow_form_percentile"] == 1.0
    assert ferrari["shadow_price_percentile"] == 0.5
    assert ferrari["shadow_strength"] == 0.75 * 1.0 + 0.25 * 0.5


def test_constructor_formula_matches_approved_coefficients_without_personal_effect():
    history = _history(
        [{"round": 1, "entity_type": "constructor", "canonical_entity_id": "ferrari", "source_entity_id": "25", "name": "Ferrari", "fantasy_points_total": 40.0}]
    )
    result = _calculate(
        history,
        selected=(RaceKey(2026, 1),),
        upcoming=EventKey(2026, 3),
        schedule=_schedule(sprint_rounds=(2, 3)),
        drivers=pd.DataFrame(columns=["id", "name", "price"]),
        constructors=_constructors(),
    )
    calibration = load_sprint_shadow_calibration()
    expected = calibration.constructor_intercept + calibration.constructor_slope
    row = result.constructors.iloc[0]
    assert row["shadow_sprint_bonus"] == pytest.approx(expected)
    assert pd.isna(row["shadow_personal_weight"])


def test_driver_and_constructor_identities_and_form_stay_separate():
    history = _history(
        [
            {"round": 1, "fantasy_points_total": 10.0},
            {"round": 1, "entity_type": "constructor", "canonical_entity_id": "ferrari", "source_entity_id": "25", "name": "Ferrari", "fantasy_points_total": 80.0},
        ]
    )
    result = _calculate(
        history,
        selected=(RaceKey(2026, 1),),
        constructors=_constructors(),
    )
    assert result.drivers.loc[0, "shadow_normal_ev"] == 10.0
    assert result.constructors.loc[0, "shadow_normal_ev"] == 80.0


def test_shadow_history_is_2026_only_when_production_mode_is_all_supported():
    history = pd.concat(
        [
            _history([{"round": 1, "fantasy_points_total": 10.0}]),
            _history([{"season": 2025, "round": 1, "fantasy_points_total": 999.0}]),
        ],
        ignore_index=True,
    )
    result = _calculate(
        history,
        selected=(RaceKey(2025, 1), RaceKey(2026, 1)),
        history_mode="all_supported",
    )
    assert result.drivers.loc[0, "shadow_normal_ev"] == 10.0
    assert result.diagnostics["production_history_mode"] == "all_supported"
    assert result.diagnostics["sprint_shadow_history"] == "2026_only"


def test_pre_2026_selected_races_cannot_enter_shadow():
    history = _history([{"season": 2025, "round": 1, "fantasy_points_total": 999.0}])
    result = _calculate(history, selected=(RaceKey(2025, 1),))
    assert pd.isna(result.drivers.loc[0, "shadow_normal_ev"])
    assert result.diagnostics["selected_2026_race_keys"] == []


def test_user_excluded_2026_round_is_absent_when_not_in_selected_keys():
    history = _history(
        [{"round": 1, "fantasy_points_total": 10.0}, {"round": 2, "fantasy_points_total": 100.0, "sprint_points": 0.0}]
    )
    result = _calculate(history, selected=(RaceKey(2026, 1),))
    assert result.drivers.loc[0, "shadow_normal_ev"] == 10.0
    assert result.drivers.loc[0, "shadow_selected_race_count"] == 1


def test_no_usable_2026_observations_marks_shadow_unavailable():
    history = _history([{"round": 2, "fantasy_points_total": 10.0}])
    result = _calculate(history, selected=(RaceKey(2026, 2),))
    assert pd.isna(result.drivers.loc[0, "shadow_normal_ev"])
    assert result.drivers.loc[0, "shadow_status"] == "unavailable"
    assert result.diagnostics["status"] == "unavailable"


def test_calculation_does_not_mutate_any_input_dataframe():
    history = _history([{"round": 1, "fantasy_points_total": 10.0}])
    schedule = _schedule()
    drivers = _drivers()
    constructors = _constructors()
    originals = [frame.copy(deep=True) for frame in (history, schedule, drivers, constructors)]
    _calculate(
        history,
        selected=(RaceKey(2026, 1),),
        schedule=schedule,
        drivers=drivers,
        constructors=constructors,
    )
    for actual, expected in zip((history, schedule, drivers, constructors), originals):
        pd.testing.assert_frame_equal(actual, expected)


def test_shadow_columns_do_not_change_production_optimiser_selection_or_objective():
    drivers = pd.DataFrame(
        [
            {"id": f"d{i}", "name": f"Driver {i}", "price": 10.0, "exp_score": float(i)}
            for i in range(1, 7)
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": f"c{i}", "name": f"Team {i}", "price": 10.0, "exp_score": float(i * 10)}
            for i in range(1, 4)
        ]
    )
    base = optimize_top_k(drivers, constructors, budget=70.0, k=1)[0]
    shadow_drivers = drivers.assign(shadow_normal_ev=-1000.0, shadow_sprint_ev=1000.0)
    shadow_constructors = constructors.assign(shadow_normal_ev=-1000.0, shadow_sprint_ev=1000.0)
    shadow = optimize_top_k(shadow_drivers, shadow_constructors, budget=70.0, k=1)[0]
    assert shadow.drivers["id"].tolist() == base.drivers["id"].tolist()
    assert shadow.constructors["id"].tolist() == base.constructors["id"].tolist()
    assert shadow.expected_score == base.expected_score


def test_shadow_columns_do_not_change_prices_budget_value_or_transfer_recommendations():
    drivers = pd.DataFrame(
        [
            {
                "id": f"d{i}",
                "name": f"Driver {i}",
                "price": 10.0,
                "exp_score": float(i),
                "expected_price_gain": i / 100,
                "volatility": 5.0,
                "dnf_rate": 0.05,
                "recent_points_2ago": float(i - 2),
                "recent_points_1ago": float(i - 1),
            }
            for i in range(1, 7)
        ]
    )
    constructors = pd.DataFrame(
        [
            {
                "id": f"c{i}",
                "name": f"Team {i}",
                "price": 10.0,
                "exp_score": float(i * 10),
                "expected_price_gain": i / 10,
                "volatility": 8.0,
            }
            for i in range(1, 4)
        ]
    )
    shadow_drivers = drivers.assign(
        shadow_normal_ev=-1000.0,
        shadow_sprint_bonus=2000.0,
        shadow_sprint_ev=1000.0,
    )
    shadow_constructors = constructors.assign(
        shadow_normal_ev=-1000.0,
        shadow_sprint_bonus=2000.0,
        shadow_sprint_ev=1000.0,
    )
    base_price = apply_probabilistic_price_change_model(drivers, DEFAULT_PRICE_CHANGE_CHEAP_RULES)
    shadow_price = apply_probabilistic_price_change_model(
        shadow_drivers, DEFAULT_PRICE_CHANGE_CHEAP_RULES
    )
    price_columns = [
        column
        for column in base_price
        if column.startswith("price_change_")
        or column.startswith("prob_")
        or column in {"expected_price_change", "projected_price"}
    ]
    pd.testing.assert_frame_equal(base_price[price_columns], shadow_price[price_columns])
    selected_driver_ids = [f"d{i}" for i in range(1, 6)]
    selected_constructor_ids = ["c1", "c2"]
    assert current_team_budget_from_selection(drivers, constructors, bank=2.5) == (
        current_team_budget_from_selection(shadow_drivers, shadow_constructors, bank=2.5)
    )
    base_value = transfer_baseline(
        selected_driver_ids, selected_constructor_ids, drivers, constructors, budget=72.5
    )
    shadow_value = transfer_baseline(
        selected_driver_ids,
        selected_constructor_ids,
        shadow_drivers,
        shadow_constructors,
        budget=72.5,
    )
    for field in (
        "team_cost",
        "remaining_budget",
        "expected_points",
        "expected_price_gain",
        "projected_team_value",
        "boosted_driver",
    ):
        assert shadow_value[field] == base_value[field]
    base_transfers = build_transfer_recommendations(
        selected_driver_ids,
        selected_constructor_ids,
        drivers,
        constructors,
        budget=72.5,
        max_transfers=1,
        top_n=5,
    )
    shadow_transfers = build_transfer_recommendations(
        selected_driver_ids,
        selected_constructor_ids,
        shadow_drivers,
        shadow_constructors,
        budget=72.5,
        max_transfers=1,
        top_n=5,
    )
    pd.testing.assert_frame_equal(base_transfers, shadow_transfers)


def test_loader_has_no_network_or_research_script_runtime_dependency(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr("requests.get", fail)
    before = set(sys.modules)
    calibration = load_sprint_shadow_calibration(DEFAULT_SPRINT_SHADOW_CALIBRATION_PATH)
    imported = set(sys.modules) - before
    assert calibration.model_version == "sprint_ev_shadow_2026_v1"
    assert not any(name.startswith("scripts.") for name in imported)


def test_p_one_full_2026_output_reproduces_frozen_research_candidate():
    canonical = pd.read_csv(CANONICAL_PATH)
    candidate = pd.read_csv(FINAL_CANDIDATE_PATH)
    current = canonical[canonical["season"].eq(2026)].sort_values("round", kind="stable")
    identities = current.drop_duplicates(["entity_type", "canonical_entity_id"], keep="last")
    price_by_key = candidate.set_index(["entity_type", "entity_id"])["current_price"].to_dict()

    def assets(asset_type):
        rows = []
        for row in identities[identities["entity_type"].eq(asset_type)].itertuples(index=False):
            key = (asset_type, row.canonical_entity_id)
            if key in price_by_key:
                rows.append({"id": _source_id(row.source_entity_id), "name": row.name, "price": price_by_key[key]})
        return pd.DataFrame(rows)

    schedule = _schedule(rounds=range(1, 13), sprint_rounds=(2, 4, 5, 9, 12))
    result = calculate_sprint_shadow(
        canonical,
        schedule,
        assets("driver"),
        assets("constructor"),
        [RaceKey(2026, round_no) for round_no in range(1, 12)],
        1.0,
        EventKey(2026, 12),
        production_history_mode="current_season_only",
    )
    combined = pd.concat(
        [result.drivers.assign(entity_type="driver"), result.constructors.assign(entity_type="constructor")],
        ignore_index=True,
    )
    actual = combined.set_index(["entity_type", "shadow_canonical_entity_id"])
    expected = candidate.set_index(["entity_type", "entity_id"])
    common = actual.index.intersection(expected.index)
    assert len(common) == 33
    for key in common:
        assert actual.loc[key, "shadow_normal_ev"] == pytest.approx(expected.loc[key, "normal_ev"], abs=1e-9)
        assert actual.loc[key, "shadow_sprint_bonus"] == pytest.approx(expected.loc[key, "final_sprint_bonus"], abs=1e-9)
        assert actual.loc[key, "shadow_sprint_ev"] == pytest.approx(expected.loc[key, "sprint_weekend_ev"], abs=1e-9)


def _source_id(value):
    numeric = pd.to_numeric(value, errors="coerce")
    return str(int(numeric)) if pd.notna(numeric) and float(numeric).is_integer() else str(value)
