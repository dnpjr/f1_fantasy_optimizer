from __future__ import annotations

from copy import deepcopy

import pandas as pd

from f1fantasy import app_core
from f1fantasy.asset_identity import build_player_identity_map
from f1fantasy.ui_helpers import optimiser_result_signature
from f1fantasy.weekend_state import EventKey, SessionKind, SessionState, SessionStatus
from scripts.compare_sprint_shadow_to_production import deterministic_budget, load_offline_snapshot


def _complete_fp1(snapshot, human_ids: list[str]) -> None:
    event = EventKey(2026, 12)
    snapshot.session_results = pd.DataFrame(
        [
            {
                "season": event.season,
                "round": event.round,
                "session_kind": SessionKind.PRACTICE_1.value,
                "human_driver_id": human_id,
                "position": position,
                "is_classified": True,
            }
            for position, human_id in enumerate(human_ids, start=1)
        ]
    )
    snapshot.session_states = (
        SessionState(
            event=event,
            kind=SessionKind.PRACTICE_1,
            scheduled_at=None,
            observed_row_count=len(human_ids),
            expected_participant_count=len(human_ids),
            status=SessionStatus.COMPLETE,
            source="test",
        ),
    )


def _settings(weight: float) -> dict:
    return {
        "today": "2026-08-21",
        "effective_time": "2026-08-21T12:00:00Z",
        "historical_seasons_back": 3,
        "horizon_races": 5,
        "current_season_weight": 1.0,
        "past_season_weight": 0.7,
        "recency_decay": 0.85,
        "selected_race_preset": "All",
        "history_mode": app_core.HISTORY_MODE_ALL_SUPPORTED,
        "live_session_emphasis": weight,
    }


def _snapshot_with_reverse_baseline_fp1():
    snapshot, _metadata = load_offline_snapshot()
    snapshot.player_identity_map = build_player_identity_map(
        snapshot.players, snapshot.results
    )
    baseline = app_core.derive_model_data(deepcopy(snapshot), **_settings(0.0))
    ordered_humans = (
        baseline.drivers.loc[
            baseline.drivers["human_driver_id"].notna()
        ]
        .sort_values("next_race_expected_points", ascending=True, kind="stable")
        ["human_driver_id"]
        .astype(str)
        .tolist()
    )
    _complete_fp1(snapshot, ordered_humans)
    return snapshot


def test_model_signature_versions_live_session_weight_without_changing_snapshot_identity():
    snapshot, _metadata = load_offline_snapshot()
    zero = app_core.model_settings_signature(
        snapshot, 3, 5, 1.0, 0.7, 0.85, "2026-08-21", live_session_emphasis=0.0
    )
    half = app_core.model_settings_signature(
        snapshot, 3, 5, 1.0, 0.7, 0.85, "2026-08-21", live_session_emphasis=0.5
    )

    assert zero != half
    assert app_core.live_data_snapshot_identity(snapshot) == app_core.live_data_snapshot_identity(snapshot)

    zero_result = app_core.model_data_version(snapshot, zero)
    half_result = app_core.model_data_version(snapshot, half)
    assert zero_result != half_result
    common_inputs = {
        "budget": 100.0,
        "chip_mode": app_core.CHIP_NONE,
        "price_growth_value": 50,
    }
    assert optimiser_result_signature(
        data_version=zero_result, **common_inputs
    ) != optimiser_result_signature(data_version=half_result, **common_inputs)


def test_default_zero_weight_is_bit_for_bit_compatible_with_explicit_zero():
    snapshot, _metadata = load_offline_snapshot()
    implicit = app_core.derive_model_data(snapshot, **{k: v for k, v in _settings(0.0).items() if k != "live_session_emphasis"})
    explicit = app_core.derive_model_data(snapshot, **_settings(0.0))

    pd.testing.assert_frame_equal(implicit.drivers, explicit.drivers)
    pd.testing.assert_frame_equal(implicit.constructors, explicit.constructors)
    pd.testing.assert_frame_equal(implicit.driver_price_efficiency, explicit.driver_price_efficiency)
    pd.testing.assert_frame_equal(implicit.constructor_price_efficiency, explicit.constructor_price_efficiency)


def test_production_driver_and_constructor_ev_respond_to_complete_live_session():
    snapshot = _snapshot_with_reverse_baseline_fp1()
    zero = app_core.derive_model_data(deepcopy(snapshot), **_settings(0.0))
    full = app_core.derive_model_data(deepcopy(snapshot), **_settings(1.0))

    pd.testing.assert_series_equal(
        zero.drivers["next_race_expected_points"], zero.drivers["baseline_ev"], check_names=False
    )
    pd.testing.assert_series_equal(
        full.drivers["next_race_expected_points"], full.drivers["adjusted_ev"], check_names=False
    )
    assert not full.drivers["next_race_expected_points"].equals(
        zero.drivers["next_race_expected_points"]
    )
    assert not full.constructors["next_race_expected_points"].equals(
        zero.constructors["next_race_expected_points"]
    )
    assert full.diagnostics["live_session_emphasis"] == 1.0


def test_no_complete_live_session_keeps_baseline_at_full_weight():
    snapshot, _metadata = load_offline_snapshot()
    derived = app_core.derive_model_data(snapshot, **_settings(1.0))

    pd.testing.assert_series_equal(
        derived.drivers["next_race_expected_points"],
        derived.drivers["baseline_ev"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        derived.constructors["next_race_expected_points"],
        derived.constructors["baseline_ev"],
        check_names=False,
    )


def test_price_efficiency_stays_historical_while_next_event_forecast_changes():
    snapshot = _snapshot_with_reverse_baseline_fp1()
    zero = app_core.derive_model_data(deepcopy(snapshot), **_settings(0.0))
    full = app_core.derive_model_data(deepcopy(snapshot), **_settings(1.0))

    pd.testing.assert_frame_equal(zero.driver_price_efficiency, full.driver_price_efficiency)
    pd.testing.assert_frame_equal(
        zero.constructor_price_efficiency, full.constructor_price_efficiency
    )
    assert not zero.drivers["next_race_expected_points"].equals(
        full.drivers["next_race_expected_points"]
    )


def test_price_projection_and_optimizer_consume_adjusted_next_event_ev():
    snapshot = _snapshot_with_reverse_baseline_fp1()
    zero = app_core.derive_model_data(deepcopy(snapshot), **_settings(0.0))
    full = app_core.derive_model_data(deepcopy(snapshot), **_settings(1.0))

    zero_prices = app_core.apply_probabilistic_price_change_model(
        zero.drivers,
        app_core.DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        expensive_rules=app_core.DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
        expensive_price_min=app_core.DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
        bounds=app_core.DEFAULT_PRICE_CHANGE_BOUNDS,
        predicted_points_col="next_race_expected_points",
    )
    full_prices = app_core.apply_probabilistic_price_change_model(
        full.drivers,
        app_core.DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        expensive_rules=app_core.DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
        expensive_price_min=app_core.DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
        bounds=app_core.DEFAULT_PRICE_CHANGE_BOUNDS,
        predicted_points_col="next_race_expected_points",
    )
    pd.testing.assert_series_equal(zero_prices["price"], full_prices["price"])
    assert not zero_prices["price_change_predicted_next"].equals(
        full_prices["price_change_predicted_next"]
    )
    assert not zero_prices["expected_price_gain"].equals(
        full_prices["expected_price_gain"]
    )

    budget, _metadata = deterministic_budget(zero.drivers, zero.constructors)
    zero_teams = app_core.run_optimizer(
        zero.drivers, zero.constructors, budget=budget, top_k=3
    )
    full_teams = app_core.run_optimizer(
        full.drivers, full.constructors, budget=budget, top_k=3
    )

    def team_assets(solution):
        return (
            tuple(sorted(solution.drivers["id"].astype(str))),
            tuple(sorted(solution.constructors["id"].astype(str))),
        )

    assert [team_assets(team) for team in zero_teams] != [
        team_assets(team) for team in full_teams
    ]
    assert zero_teams[0].boosted_driver != full_teams[0].boosted_driver
