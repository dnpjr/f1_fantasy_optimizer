from __future__ import annotations

import math

import pandas as pd
import pytest

from f1fantasy.live_session_shadow import (
    apply_live_session_emphasis,
    blend_live_session_ev,
    completed_live_session_labels,
    validate_live_session_emphasis,
)
from f1fantasy.weekend_state import EventKey, SessionKind, SessionState, SessionStatus


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), None, True])
def test_invalid_live_session_emphasis_is_rejected(value):
    with pytest.raises(ValueError, match="0 to 1"):
        validate_live_session_emphasis(value)


def test_zero_emphasis_is_exact_baseline_even_when_live_ev_exists():
    assert blend_live_session_ev(31.25, 99.0, 0.0) == 31.25


def test_full_emphasis_uses_live_only_ev():
    assert blend_live_session_ev(31.25, 44.5, 1.0) == 44.5


def test_intermediate_emphasis_uses_single_linear_formula():
    assert blend_live_session_ev(20.0, 40.0, 0.25) == 25.0


def test_missing_live_ev_falls_back_to_baseline_for_any_weight():
    assert blend_live_session_ev(20.0, None, 1.0) == 20.0
    assert blend_live_session_ev(20.0, float("inf"), 1.0) == 20.0


def test_missing_baseline_stays_missing_even_when_live_ev_exists():
    assert math.isnan(blend_live_session_ev(None, 40.0, 1.0))


def test_frame_blend_updates_only_first_event_delta_and_does_not_mutate_input():
    source = pd.DataFrame(
        [
            {
                "id": "driver-a",
                "baseline_ev": 20.0,
                "live_only_ev": 40.0,
                "next_race_expected_points": 20.0,
                "next_race_exp_score": 20.0,
                "exp_score": 20.0,
                "nn_exp_score": 23.0,
                "horizon_expected_points": 72.0,
            },
            {
                "id": "driver-b",
                "baseline_ev": 30.0,
                "live_only_ev": float("nan"),
                "next_race_expected_points": 30.0,
                "next_race_exp_score": 30.0,
                "exp_score": 30.0,
                "nn_exp_score": 35.0,
                "horizon_expected_points": 90.0,
            },
        ]
    )
    before = source.copy(deep=True)

    blended = apply_live_session_emphasis(source, 0.5)

    pd.testing.assert_frame_equal(source, before)
    first = blended.set_index("id").loc["driver-a"]
    assert first["adjusted_ev"] == 30.0
    assert first["live_session_ev_difference"] == 10.0
    assert first["next_race_expected_points"] == 30.0
    assert first["next_race_exp_score"] == 30.0
    assert first["exp_score"] == 30.0
    assert first["nn_exp_score"] == 33.0
    assert first["baseline_horizon_expected_points"] == 72.0
    assert first["horizon_expected_points"] == 82.0

    missing_live = blended.set_index("id").loc["driver-b"]
    assert missing_live["adjusted_ev"] == 30.0
    assert missing_live["live_session_ev_difference"] == 0.0
    assert missing_live["horizon_expected_points"] == 90.0


def test_zero_weight_preserves_all_existing_production_values_exactly():
    source = pd.DataFrame(
        [
            {
                "baseline_ev": 19.125,
                "live_only_ev": 40.0,
                "next_race_expected_points": 19.125,
                "next_race_exp_score": 19.125,
                "exp_score": 19.125,
                "nn_exp_score": 21.5,
                "horizon_expected_points": 73.75,
            }
        ]
    )

    blended = apply_live_session_emphasis(source, 0.0)

    for column in (
        "next_race_expected_points",
        "next_race_exp_score",
        "exp_score",
        "nn_exp_score",
        "horizon_expected_points",
    ):
        pd.testing.assert_series_equal(blended[column], source[column], check_names=False)


def test_compact_status_lists_only_complete_supported_sessions_in_weekend_order():
    def state(kind: SessionKind, status: SessionStatus) -> SessionState:
        return SessionState(
            event=EventKey(2026, 12),
            kind=kind,
            scheduled_at=None,
            observed_row_count=22,
            expected_participant_count=22,
            status=status,
            source="test",
        )

    states = (
        state(SessionKind.SPRINT_QUALIFYING, SessionStatus.COMPLETE),
        state(SessionKind.PRACTICE_2, SessionStatus.PARTIAL),
        state(SessionKind.PRACTICE_1, SessionStatus.COMPLETE),
        state(SessionKind.GRAND_PRIX_QUALIFYING, SessionStatus.COMPLETE),
    )

    assert completed_live_session_labels(states) == ("FP1", "SQ")
    assert completed_live_session_labels(
        (state(SessionKind.PRACTICE_1, SessionStatus.PROVISIONAL),)
    ) == ()
