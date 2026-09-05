from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from f1fantasy.weekend_state import (
    EventKey,
    SessionKind,
    SessionStatus,
    WeekendFormat,
    build_weekend_state,
    classify_deadline_payload,
    classify_playerstats_payload,
    classify_schedule_dataframe,
    classify_session_dataframe,
    select_active_event,
    validate_deadline_candidate,
    validate_weekend_snapshot,
)


def _normal_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 10,
                "raceName": "Safety Grand Prix",
                "circuitName": "Safety Circuit",
                "date": "2026-08-02",
                "time": "14:00:00Z",
                "practice_1_date": "2026-07-31",
                "practice_1_time": "10:00:00Z",
                "practice_2_date": "2026-07-31",
                "practice_2_time": "14:00:00Z",
                "practice_3_date": "2026-08-01",
                "practice_3_time": "10:00:00Z",
                "qualifying_date": "2026-08-01",
                "qualifying_time": "14:00:00Z",
                "sprint_date": "",
                "sprint_time": "",
            },
            {
                "season": 2026,
                "round": 11,
                "raceName": "Next Grand Prix",
                "circuitName": "Next Circuit",
                "date": "2026-08-09",
                "time": "14:00:00Z",
                "qualifying_date": "2026-08-08",
                "qualifying_time": "14:00:00Z",
                "sprint_date": "",
                "sprint_time": "",
            },
        ]
    )


def _sprint_schedule() -> pd.DataFrame:
    data = _normal_schedule()
    data.loc[0, "practice_2_date"] = ""
    data.loc[0, "practice_2_time"] = ""
    data.loc[0, "practice_3_date"] = ""
    data.loc[0, "practice_3_time"] = ""
    data.loc[0, "sprint_qualifying_date"] = "2026-07-31"
    data.loc[0, "sprint_qualifying_time"] = "15:00:00Z"
    data.loc[0, "sprint_date"] = "2026-08-01"
    data.loc[0, "sprint_time"] = "10:00:00Z"
    return data


def _classification(kind: str, count: int = 2, status: str = "Finished") -> pd.DataFrame:
    rows = []
    for index in range(count):
        base = {
            "season": 2026,
            "round": 10,
            "driverId": f"d{index + 1}",
            "driver": f"Driver {index + 1}",
            "constructorId": "c1",
            "constructor": "Team One",
            "position": index + 1,
        }
        if kind in {"race", "sprint"}:
            base.update({"grid": index + 1, "status": status, "fastestLapRank": 0})
        else:
            base.update({"q1": "1:20", "q2": "1:19", "q3": "1:18"})
        rows.append(base)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("checkpoint", "effective_time", "expected_status"),
    [
        ("N0", "2026-07-30T12:00:00Z", "upcoming"),
        ("N1", "2026-07-31T12:00:00Z", "live"),
        ("N2", "2026-07-31T16:00:00Z", "live"),
        ("N3", "2026-08-01T12:00:00Z", "live"),
    ],
)
def test_normal_pre_scoring_checkpoints_are_pending_not_failed(
    checkpoint, effective_time, expected_status
):
    state = build_weekend_state(
        _normal_schedule().iloc[0],
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        effective_time=effective_time,
        expected_participant_count=2,
    )

    assert checkpoint.startswith("N")
    assert state.format == WeekendFormat.NORMAL
    assert state.status == expected_status
    assert state.session(SessionKind.GRAND_PRIX).status == SessionStatus.PENDING
    assert state.session(SessionKind.SPRINT).status == SessionStatus.NOT_SCHEDULED
    assert not state.is_final


def test_n4_completed_qualifying_does_not_complete_normal_weekend():
    state = build_weekend_state(
        _normal_schedule().iloc[0],
        results=pd.DataFrame(),
        qualifying=_classification("qualifying"),
        sprint=pd.DataFrame(),
        effective_time="2026-08-01T16:00:00Z",
        expected_participant_count=2,
    )

    assert state.session(SessionKind.GRAND_PRIX_QUALIFYING).status == SessionStatus.COMPLETE
    assert state.session(SessionKind.GRAND_PRIX).status == SessionStatus.PENDING
    assert not state.is_final


def test_n5_running_partial_race_is_not_final_and_snapshot_is_unsafe_partial():
    running = _classification("race", count=1, status="Running")
    state = build_weekend_state(
        _normal_schedule().iloc[0],
        results=running,
        qualifying=_classification("qualifying"),
        sprint=pd.DataFrame(),
        effective_time="2026-08-02T14:30:00Z",
        expected_participant_count=2,
    )
    validation = validate_weekend_snapshot(
        _normal_schedule(),
        results=running,
        qualifying=_classification("qualifying"),
        sprint=pd.DataFrame(),
        effective_time="2026-08-02T14:30:00Z",
        expected_participant_count=2,
    )

    assert state.session(SessionKind.GRAND_PRIX).status == SessionStatus.IN_PROGRESS
    assert not state.is_final
    assert validation.status == "unsafe_partial"
    assert EventKey(2026, 10) in validation.excluded_partial_event_keys


def test_n6_final_classification_completes_weekend():
    state = build_weekend_state(
        _normal_schedule().iloc[0],
        results=_classification("race"),
        qualifying=_classification("qualifying"),
        sprint=pd.DataFrame(),
        effective_time="2026-08-02T18:00:00Z",
        expected_participant_count=2,
    )

    assert state.is_final
    assert state.status == "complete"


@pytest.mark.parametrize(
    ("checkpoint", "effective_time", "sprint_rows", "quali_rows", "race_rows", "expected_sprint"),
    [
        ("S0", "2026-07-30T12:00:00Z", 0, 0, 0, SessionStatus.PENDING),
        ("S1", "2026-07-31T12:00:00Z", 0, 0, 0, SessionStatus.PENDING),
        ("S2", "2026-07-31T16:00:00Z", 0, 0, 0, SessionStatus.PENDING),
        ("S3", "2026-08-01T10:30:00Z", 1, 0, 0, SessionStatus.IN_PROGRESS),
        ("S4", "2026-08-01T12:30:00Z", 2, 0, 0, SessionStatus.COMPLETE),
        ("S5", "2026-08-01T16:00:00Z", 2, 2, 0, SessionStatus.COMPLETE),
        ("S6", "2026-08-02T14:30:00Z", 2, 2, 1, SessionStatus.COMPLETE),
        ("S7", "2026-08-02T18:00:00Z", 2, 2, 2, SessionStatus.COMPLETE),
    ],
)
def test_sprint_weekend_checkpoint_matrix(
    checkpoint, effective_time, sprint_rows, quali_rows, race_rows, expected_sprint
):
    sprint = _classification(
        "sprint", sprint_rows, status="Running" if checkpoint == "S3" else "Finished"
    ) if sprint_rows else pd.DataFrame()
    race = _classification(
        "race", race_rows, status="Running" if checkpoint == "S6" else "Finished"
    ) if race_rows else pd.DataFrame()
    qualifying = _classification("qualifying", quali_rows) if quali_rows else pd.DataFrame()
    state = build_weekend_state(
        _sprint_schedule().iloc[0],
        results=race,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=effective_time,
        expected_participant_count=2,
    )

    assert checkpoint.startswith("S")
    assert state.format == WeekendFormat.SPRINT
    assert state.session(SessionKind.SPRINT).status == expected_sprint
    assert state.session(SessionKind.SPRINT_QUALIFYING).kind != SessionKind.GRAND_PRIX_QUALIFYING
    assert state.is_final is (checkpoint == "S7")


def test_empty_after_publication_is_partial_but_future_empty_is_pending():
    future = classify_session_dataframe(
        pd.DataFrame(),
        event=EventKey(2026, 10),
        kind=SessionKind.GRAND_PRIX,
        scheduled_at=datetime(2026, 8, 2, 14, tzinfo=UTC),
        effective_time="2026-08-02T12:00:00Z",
        expected_participant_count=2,
        source="test",
    )
    late = classify_session_dataframe(
        pd.DataFrame(),
        event=EventKey(2026, 10),
        kind=SessionKind.GRAND_PRIX,
        scheduled_at=datetime(2026, 8, 2, 14, tzinfo=UTC),
        effective_time="2026-08-02T20:00:00Z",
        expected_participant_count=2,
        source="test",
    )

    assert future.status == SessionStatus.PENDING
    assert late.status == SessionStatus.PARTIAL


def test_incomplete_required_values_are_partial_even_when_row_count_matches():
    rows = _classification("race")
    rows.loc[1, "position"] = pd.NA
    state = classify_session_dataframe(
        rows,
        event=EventKey(2026, 10),
        kind=SessionKind.GRAND_PRIX,
        scheduled_at=datetime(2026, 8, 2, 14, tzinfo=UTC),
        effective_time="2026-08-02T18:00:00Z",
        expected_participant_count=2,
        source="test",
    )

    assert state.status == SessionStatus.PARTIAL


def test_classifiers_distinguish_malformed_failed_and_valid_sources():
    malformed = classify_session_dataframe(
        pd.DataFrame([{"season": 2026, "round": 10, "driverId": "d1"}]),
        event=EventKey(2026, 10),
        kind=SessionKind.GRAND_PRIX,
        scheduled_at=datetime(2026, 8, 2, 14, tzinfo=UTC),
        effective_time="2026-08-02T18:00:00Z",
        expected_participant_count=1,
        source="test",
    )
    failed = classify_session_dataframe(
        pd.DataFrame(),
        event=EventKey(2026, 10),
        kind=SessionKind.GRAND_PRIX,
        scheduled_at=datetime(2026, 8, 2, 14, tzinfo=UTC),
        effective_time="2026-08-02T18:00:00Z",
        source="test",
        source_error="timeout",
    )

    assert malformed.status == SessionStatus.MALFORMED
    assert failed.status == SessionStatus.FAILED
    assert classify_schedule_dataframe(_normal_schedule(), season=2026)[0] == SessionStatus.COMPLETE
    assert classify_schedule_dataframe(pd.DataFrame(), season=2026)[0] == SessionStatus.PARTIAL
    assert classify_playerstats_payload({"Value": {}})[0] == SessionStatus.PENDING
    assert classify_playerstats_payload({"bad": True})[0] == SessionStatus.MALFORMED
    assert classify_deadline_payload({"team_lock_deadline_utc": None})[0] == SessionStatus.PENDING


def test_active_event_remains_after_midnight_until_final_then_advances():
    partial = _classification("race", count=1, status="Provisional")
    schedule = _normal_schedule()
    current = select_active_event(
        schedule,
        results=partial,
        qualifying=_classification("qualifying"),
        sprint=pd.DataFrame(),
        effective_time="2026-08-03T00:30:00Z",
        expected_participant_count=2,
    )
    complete = select_active_event(
        schedule,
        results=_classification("race"),
        qualifying=_classification("qualifying"),
        sprint=pd.DataFrame(),
        effective_time="2026-08-03T00:30:00Z",
        expected_participant_count=2,
    )

    assert current.event == EventKey(2026, 10)
    assert current.status == "awaiting_final_classification"
    assert complete.event == EventKey(2026, 11)


def test_timeout_advance_keeps_prior_event_explicitly_unresolved():
    validation = validate_weekend_snapshot(
        _normal_schedule(),
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        effective_time="2026-08-05T15:00:00Z",
        expected_participant_count=2,
    )

    assert validation.active_weekend.event == EventKey(2026, 11)
    assert EventKey(2026, 10) in validation.unresolved_event_keys
    assert any("explicitly unresolved" in warning for warning in validation.warnings)


def test_deadline_validation_matches_event_and_format_and_rejects_wrong_meaning():
    row = _normal_schedule().iloc[0]
    valid = validate_deadline_candidate(
        {
            "team_lock_deadline_utc": "2026-08-01T14:00:00Z",
            "team_lock_session_type": "Qualifying",
            "team_lock_meeting_name": "Safety Grand Prix",
        },
        active_event=EventKey(2026, 10),
        schedule_row=row,
        format=WeekendFormat.NORMAL,
    )
    wrong_event = validate_deadline_candidate(
        {
            "team_lock_deadline_utc": "2026-08-01T14:00:00Z",
            "team_lock_session_type": "Qualifying",
            "team_lock_meeting_name": "Different Grand Prix",
        },
        active_event=EventKey(2026, 10),
        schedule_row=row,
        format=WeekendFormat.NORMAL,
    )
    race_start = validate_deadline_candidate(
        {
            "team_lock_deadline_utc": "2026-08-02T14:00:00Z",
            "team_lock_session_type": "Race",
            "team_lock_meeting_name": "Safety Grand Prix",
        },
        active_event=EventKey(2026, 10),
        schedule_row=row,
        format=WeekendFormat.NORMAL,
    )
    sprint_fallback_candidate = validate_deadline_candidate(
        {
            "team_lock_deadline_utc": "2026-08-01T10:00:00Z",
            "team_lock_session_type": "Sprint",
            "team_lock_meeting_name": "Safety Grand Prix",
        },
        active_event=EventKey(2026, 10),
        schedule_row=_sprint_schedule().iloc[0],
        format=WeekendFormat.SPRINT,
    )

    assert valid["team_lock_deadline_valid"] is True
    assert wrong_event["team_lock_deadline_valid"] is False
    assert race_start["team_lock_deadline_valid"] is False
    assert sprint_fallback_candidate["team_lock_deadline_valid"] is True


def test_classification_does_not_mutate_inputs():
    source = _classification("race")
    original = source.copy(deep=True)
    build_weekend_state(
        _normal_schedule().iloc[0],
        results=source,
        qualifying=_classification("qualifying"),
        sprint=pd.DataFrame(),
        effective_time="2026-08-02T18:00:00Z",
        expected_participant_count=2,
    )
    pd.testing.assert_frame_equal(source, original)
