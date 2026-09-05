from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from f1fantasy.live_session_shadow import (
    normalise_session_positions,
    position_to_live_score,
    weighted_live_session_score,
)
from f1fantasy.weekend_state import SessionKind, WeekendFormat


def _classification(size: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "human_driver_id": [f"driver_{index}" for index in range(1, size + 1)],
            "position": list(range(1, size + 1)),
            "is_classified": [True] * size,
        }
    )


def test_position_one_maps_to_one():
    assert position_to_live_score(1, 20) == 1.0


def test_last_position_maps_to_zero():
    assert position_to_live_score(20, 20) == 0.0


def test_intermediate_position_scores_are_monotonic():
    scores = [position_to_live_score(position, 20) for position in range(1, 21)]
    assert scores == sorted(scores, reverse=True)
    assert all(left > right for left, right in zip(scores, scores[1:]))


@pytest.mark.parametrize("field_size", [20, 22])
def test_actual_classified_field_size_is_used(field_size):
    scored = normalise_session_positions(_classification(field_size))

    assert scored["classified_field_size"].unique().tolist() == [field_size]
    assert scored.iloc[0]["position_score"] == 1.0
    assert scored.iloc[-1]["position_score"] == 0.0


def test_duplicate_positions_are_rejected():
    classification = _classification(3)
    classification.loc[2, "position"] = 2

    with pytest.raises(ValueError, match="unique"):
        normalise_session_positions(classification)


def test_gapped_or_out_of_range_positions_are_rejected():
    classification = _classification(3)
    classification.loc[2, "position"] = 4

    with pytest.raises(ValueError, match="1..N"):
        normalise_session_positions(classification)


def test_single_driver_field_is_unavailable():
    with pytest.raises(ValueError, match="at least two"):
        normalise_session_positions(_classification(1))


def test_position_normalisation_does_not_mutate_input():
    classification = _classification(20)
    original = deepcopy(classification)

    normalise_session_positions(classification)

    pd.testing.assert_frame_equal(classification, original)


def test_normal_fp1_only_renormalises_to_fp1():
    result = weighted_live_session_score(
        {SessionKind.PRACTICE_1: 0.4}, WeekendFormat.NORMAL
    )

    assert result.live_session_score == pytest.approx(0.4)
    assert result.sessions_used == (SessionKind.PRACTICE_1.value,)
    assert result.weight_sum == 1.0


def test_normal_fp1_fp2_uses_one_to_two_weights():
    result = weighted_live_session_score(
        {SessionKind.PRACTICE_1: 0.4, SessionKind.PRACTICE_2: 0.7},
        WeekendFormat.NORMAL,
    )

    assert result.live_session_score == pytest.approx((0.4 + 2 * 0.7) / 3)
    assert result.session_count == 2
    assert result.weight_sum == 3.0


def test_normal_all_practice_uses_one_two_three_weights():
    result = weighted_live_session_score(
        {
            SessionKind.PRACTICE_1: 0.2,
            SessionKind.PRACTICE_2: 0.5,
            SessionKind.PRACTICE_3: 0.8,
        },
        WeekendFormat.NORMAL,
    )

    assert result.live_session_score == pytest.approx((0.2 + 2 * 0.5 + 3 * 0.8) / 6)
    assert result.weight_sum == 6.0


def test_normal_fp2_only_renormalises_to_fp2():
    result = weighted_live_session_score(
        {SessionKind.PRACTICE_2: 0.65}, WeekendFormat.NORMAL
    )

    assert result.live_session_score == pytest.approx(0.65)
    assert result.sessions_used == (SessionKind.PRACTICE_2.value,)
    assert result.weight_sum == 2.0


def test_sprint_fp1_only_renormalises_to_fp1():
    result = weighted_live_session_score(
        {SessionKind.PRACTICE_1: 0.4}, WeekendFormat.SPRINT
    )

    assert result.live_session_score == pytest.approx(0.4)
    assert result.weight_sum == 1.0


def test_sprint_fp1_and_sq_use_one_to_three_weights():
    result = weighted_live_session_score(
        {SessionKind.PRACTICE_1: 0.4, SessionKind.SPRINT_QUALIFYING: 0.75},
        WeekendFormat.SPRINT,
    )

    assert result.live_session_score == pytest.approx(0.6625)
    assert result.sessions_used == (
        SessionKind.PRACTICE_1.value,
        SessionKind.SPRINT_QUALIFYING.value,
    )
    assert result.weight_sum == 4.0


def test_sprint_qualifying_has_triple_fp1_influence():
    high_fp1 = weighted_live_session_score(
        {SessionKind.PRACTICE_1: 1.0, SessionKind.SPRINT_QUALIFYING: 0.0},
        WeekendFormat.SPRINT,
    )
    high_sq = weighted_live_session_score(
        {SessionKind.PRACTICE_1: 0.0, SessionKind.SPRINT_QUALIFYING: 1.0},
        WeekendFormat.SPRINT,
    )

    assert high_fp1.live_session_score == pytest.approx(0.25)
    assert high_sq.live_session_score == pytest.approx(0.75)


def test_wrong_format_sessions_are_ignored():
    result = weighted_live_session_score(
        {
            SessionKind.PRACTICE_1: 0.5,
            SessionKind.SPRINT_QUALIFYING: 1.0,
            SessionKind.GRAND_PRIX_QUALIFYING: 1.0,
            SessionKind.SPRINT: 1.0,
        },
        WeekendFormat.NORMAL,
    )

    assert result.live_session_score == 0.5
    assert result.sessions_used == (SessionKind.PRACTICE_1.value,)
