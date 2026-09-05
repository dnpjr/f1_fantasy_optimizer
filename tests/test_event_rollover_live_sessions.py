from __future__ import annotations

from copy import deepcopy

import pandas as pd

from f1fantasy import app_core
from f1fantasy.live_session_shadow import (
    apply_live_session_emphasis,
    build_live_session_shadow,
    completed_live_session_labels,
)
from f1fantasy.weekend_state import (
    EventKey,
    SessionKind,
    SessionState,
    SessionStatus,
    WeekendFormat,
    select_forecast_event,
    upcoming_event_records,
)


DUTCH = EventKey(2026, 12)
ITALY = EventKey(2026, 13)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 12,
                "raceName": "Dutch Grand Prix",
                "circuitName": "Circuit Park Zandvoort",
                "date": "2026-08-23",
                "time": "13:00:00Z",
                "practice_1_date": "2026-08-21",
                "practice_1_time": "10:00:00Z",
                "sprint_qualifying_date": "2026-08-21",
                "sprint_qualifying_time": "14:00:00Z",
                "sprint_date": "2026-08-22",
                "sprint_time": "10:00:00Z",
                "qualifying_date": "2026-08-22",
                "qualifying_time": "14:00:00Z",
            },
            {
                "season": 2026,
                "round": 13,
                "raceName": "Italian Grand Prix",
                "circuitName": "Autodromo Nazionale di Monza",
                "date": "2026-09-06",
                "time": "13:00:00Z",
                "practice_1_date": "2026-09-04",
                "practice_1_time": "11:00:00Z",
                "practice_2_date": "2026-09-04",
                "practice_2_time": "15:00:00Z",
                "practice_3_date": "2026-09-05",
                "practice_3_time": "10:00:00Z",
                "qualifying_date": "2026-09-05",
                "qualifying_time": "14:00:00Z",
                "sprint_date": "",
                "sprint_time": "",
                "sprint_qualifying_date": "",
                "sprint_qualifying_time": "",
            },
        ]
    )


def _classification(kind: str, *, status: str = "Finished") -> pd.DataFrame:
    rows = []
    for position, driver_id in enumerate(("lawson", "tsunoda"), start=1):
        row = {
            "season": 2026,
            "round": 12,
            "driverId": driver_id,
            "driver": driver_id.title(),
            "constructorId": "team",
            "constructor": "Team",
            "position": position,
        }
        if kind == "race":
            row.update({"grid": position, "status": status})
        else:
            row.update({"q1": "1:20", "q2": "1:19", "q3": "1:18"})
        rows.append(row)
    return pd.DataFrame(rows)


def _session_state(event: EventKey, kind: SessionKind) -> SessionState:
    return SessionState(
        event=event,
        kind=kind,
        scheduled_at=None,
        observed_row_count=2,
        expected_participant_count=2,
        status=SessionStatus.COMPLETE,
        source="test",
    )


def _session_rows(event: EventKey, kind: SessionKind) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": event.season,
                "round": event.round,
                "session_kind": kind.value,
                "source_driver_id": human_id,
                "human_driver_id": human_id,
                "identity_match_status": "matched",
                "position": position,
            }
            for position, human_id in enumerate(("lawson", "tsunoda"), start=1)
        ]
    )


def _drivers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 116,
                "human_driver_id": "lawson",
                "identity_match_status": "matched",
                "is_active": 1,
                "team": "Red Bull Racing",
                "next_race_expected_points": 25.0,
            },
            {
                "id": 130,
                "human_driver_id": "tsunoda",
                "identity_match_status": "matched",
                "is_active": 1,
                "team": "Racing Bulls",
                "next_race_expected_points": 15.0,
            },
        ]
    )


def _empty_constructors() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "name", "next_race_expected_points"])


def _snapshot(event: EventKey, kind: SessionKind) -> app_core.LiveDataSnapshot:
    return app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2026,
        requested_seasons=(2026,),
        loaded_seasons=(2026,),
        season_load_failures={},
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=_schedule(),
        players=pd.DataFrame(),
        teams=pd.DataFrame(),
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={
            "forecast_target_event": {"season": event.season, "round": event.round}
        },
        session_results=_session_rows(event, kind),
        session_states=(_session_state(event, kind),),
    )


def test_dutch_stays_target_until_grand_prix_classification_is_complete():
    pending = select_forecast_event(
        _schedule(),
        results=_classification("race", status="Provisional"),
        qualifying=_classification("qualifying"),
        sprint=pd.DataFrame(),
        effective_time="2026-08-24T08:00:00Z",
        expected_participant_count=2,
    )

    assert pending is not None
    assert pending.event == DUTCH


def test_complete_grand_prix_advances_forecast_without_waiting_for_midnight_or_supporting_sessions():
    target = select_forecast_event(
        _schedule(),
        results=_classification("race"),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        effective_time="2026-08-23T15:00:00Z",
        expected_participant_count=2,
    )

    assert target is not None
    assert target.event == ITALY
    assert target.format == WeekendFormat.NORMAL


def test_verified_fantasy_gameday_advances_forecast_while_scoring_source_is_delayed():
    target = select_forecast_event(
        _schedule(),
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        effective_time="2026-08-24T08:00:00Z",
        expected_participant_count=2,
        verified_target_event=ITALY,
    )

    assert target is not None
    assert target.event == ITALY
    assert target.format == WeekendFormat.NORMAL


def test_runtime_feed_round_maps_legacy_snapshot_to_italy_target():
    assert app_core._market_forecast_event_hint(
        _schedule(),
        current_season=2026,
        market_resolution={"feed_round": 13, "live_data_snapshot_name": None},
    ) == ITALY


def test_stale_verified_hint_cannot_roll_forecast_back_from_a_later_active_event():
    target = select_forecast_event(
        _schedule(),
        results=_classification("race"),
        qualifying=_classification("qualifying"),
        sprint=_classification("race"),
        effective_time="2026-08-24T08:00:00Z",
        expected_participant_count=2,
        verified_target_event=DUTCH,
    )

    assert target is not None
    assert target.event == ITALY


def test_horizon_starts_with_italy_and_does_not_leak_dutch_sprint_format():
    horizon = upcoming_event_records(_schedule(), start_event=ITALY, limit=2)

    assert [event.event for event in horizon] == [ITALY]
    assert horizon[0].circuit == "Autodromo Nazionale di Monza"
    assert horizon[0].format == WeekendFormat.NORMAL


def test_italy_forecast_ignores_complete_dutch_sessions_even_at_full_emphasis():
    states = (
        _session_state(DUTCH, SessionKind.PRACTICE_1),
        _session_state(DUTCH, SessionKind.SPRINT_QUALIFYING),
    )
    rows = pd.concat(
        [
            _session_rows(DUTCH, SessionKind.PRACTICE_1),
            _session_rows(DUTCH, SessionKind.SPRINT_QUALIFYING),
        ],
        ignore_index=True,
    )
    shadow = build_live_session_shadow(
        _drivers(),
        _empty_constructors(),
        rows,
        states,
        WeekendFormat.NORMAL,
        forecast_event=ITALY,
    )
    blended = apply_live_session_emphasis(shadow.drivers, 1.0)

    assert shadow.diagnostics["status"] == "unavailable"
    assert shadow.diagnostics["drivers"]["active_event"] == (2026, 13)
    assert blended["live_session_score"].isna().all()
    assert blended["live_only_ev"].isna().all()
    pd.testing.assert_series_equal(
        blended["adjusted_ev"], blended["baseline_ev"], check_names=False
    )
    assert completed_live_session_labels(states, forecast_event=ITALY) == ()


def test_italy_fp1_activates_same_persisted_emphasis_and_maps_human_identity():
    shadow = build_live_session_shadow(
        _drivers(),
        _empty_constructors(),
        _session_rows(ITALY, SessionKind.PRACTICE_1),
        (_session_state(ITALY, SessionKind.PRACTICE_1),),
        WeekendFormat.NORMAL,
        forecast_event=ITALY,
    )
    blended = apply_live_session_emphasis(shadow.drivers, 0.6)

    assert shadow.diagnostics["status"] == "available"
    assert shadow.drivers["live_session_score"].notna().all()
    assert blended["live_session_emphasis"].eq(0.6).all()
    assert completed_live_session_labels(
        (_session_state(ITALY, SessionKind.PRACTICE_1),), forecast_event=ITALY
    ) == ("FP1",)


def test_refresh_rollover_retains_dutch_raw_rows_but_keeps_italy_states_current():
    current = _snapshot(DUTCH, SessionKind.PRACTICE_1)
    loaded = _snapshot(ITALY, SessionKind.PRACTICE_1)
    resolved = app_core.resolve_live_data_snapshot(current, True, lambda _force: loaded)
    snapshot = resolved["snapshot"]

    assert set(zip(snapshot.session_results["round"], snapshot.session_results["session_kind"])) == {
        (12, SessionKind.PRACTICE_1.value),
        (13, SessionKind.PRACTICE_1.value),
    }
    assert {(state.event.round, state.kind) for state in snapshot.session_states} == {
        (12, SessionKind.PRACTICE_1),
        (13, SessionKind.PRACTICE_1),
    }


def test_forecast_target_versions_model_and_marks_stored_optimiser_inputs_stale():
    dutch = _snapshot(DUTCH, SessionKind.PRACTICE_1)
    italy = deepcopy(dutch)
    italy.source_diagnostics["forecast_target_event"] = {"season": 2026, "round": 13}
    settings = (3, 5, 1.0, 0.7, 0.85, "2026-08-24")

    dutch_signature = app_core.model_settings_signature(dutch, *settings)
    italy_signature = app_core.model_settings_signature(italy, *settings)

    assert dutch_signature != italy_signature
    assert app_core.model_data_version(dutch, dutch_signature) != app_core.model_data_version(
        italy, italy_signature
    )
