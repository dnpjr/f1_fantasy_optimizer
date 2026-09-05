from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from f1fantasy import app_core
from f1fantasy.live_sessions import (
    JOLPICA_ALPHA_BASE,
    clear_live_session_cache,
    ingest_active_event_sessions,
    live_session_signature,
)
from f1fantasy.weekend_state import EventKey, SessionKind, SessionState, SessionStatus, WeekendFormat
from scripts.compare_sprint_shadow_to_production import deterministic_budget, load_offline_snapshot


EVENT = EventKey(2026, 12)


def _result(
    abbreviation: str,
    given_name: str,
    family_name: str,
    position: int,
    *,
    driver_id: str | None = None,
) -> dict:
    return {
        "driver": {
            "id": driver_id or f"driver_{abbreviation.casefold()}",
            "abbreviation": abbreviation,
            "given_name": given_name,
            "family_name": family_name,
        },
        "team": {"id": "team_example", "name": "Example Team"},
        "position": position,
        "position_text": str(position),
        "time": f"1:1{position}.000",
        "is_classified": True,
        "laps": 20 + position,
        "components": {},
    }


def session_payload(kind: SessionKind, results: list[dict] | None = None) -> dict:
    code = {
        SessionKind.PRACTICE_1: "FP1",
        SessionKind.PRACTICE_2: "FP2",
        SessionKind.PRACTICE_3: "FP3",
        SessionKind.SPRINT_QUALIFYING: "SQ",
    }[kind]
    return {
        "metadata": {"timestamp": "2026-08-21T16:00:00Z"},
        "data": {
            "code": code,
            "title": code,
            "timestamp": "2026-08-21T10:30:00Z",
            "season": {"year": EVENT.season},
            "round": {"number": EVENT.round, "name": "Dutch Grand Prix"},
            "results": results
            if results is not None
            else [
                _result("LAW", "Liam", "Lawson", 1),
                _result("TSU", "Yuki", "Tsunoda", 2),
                _result("LIN", "Arvid", "Lindblad", 3),
            ],
        },
    }


def _schedule_row(*, sprint: bool = False) -> dict:
    row = {
        "season": 2026,
        "round": 12,
        "raceName": "Dutch Grand Prix",
        "practice_1_date": "2026-08-21",
        "practice_1_time": "10:30:00Z",
        "practice_2_date": "2026-08-21" if not sprint else "",
        "practice_2_time": "14:00:00Z" if not sprint else "",
        "practice_3_date": "2026-08-22" if not sprint else "",
        "practice_3_time": "10:30:00Z" if not sprint else "",
        "sprint_qualifying_date": "2026-08-21" if sprint else "",
        "sprint_qualifying_time": "14:30:00Z" if sprint else "",
    }
    return row


def _identity(count: int = 3) -> pd.DataFrame:
    values = [
        (101, "lawson", "LAW", "Liam Lawson"),
        (102, "tsunoda", "TSU", "Yuki Tsunoda"),
        (103, "arvid_lindblad", "LIN", "Arvid Lindblad"),
    ][:count]
    return pd.DataFrame(
        [
            {
                "fantasy_asset_id": asset,
                "human_driver_id": human,
                "tla": tla,
                "display_name": name,
                "active": True,
                "match_status": "matched",
            }
            for asset, human, tla, name in values
        ]
    )


def _alpha_schedule(*, sprint: bool) -> dict:
    kinds = (
        (SessionKind.PRACTICE_1, SessionKind.SPRINT_QUALIFYING)
        if sprint
        else (SessionKind.PRACTICE_1, SessionKind.PRACTICE_2, SessionKind.PRACTICE_3)
    )
    codes = {
        SessionKind.PRACTICE_1: "FP1",
        SessionKind.PRACTICE_2: "FP2",
        SessionKind.PRACTICE_3: "FP3",
        SessionKind.SPRINT_QUALIFYING: "SQ",
    }
    return {
        "data": {
            "events": [
                {
                    "round": {"number": 12, "name": "Dutch Grand Prix"},
                    "schedule": [
                        {
                            "code": codes[kind],
                            "results_url": f"{JOLPICA_ALPHA_BASE}/results/round_test/{codes[kind]}/",
                        }
                        for kind in kinds
                    ],
                }
            ]
        }
    }


def _loader(payloads: dict[str, object], calls: list[str]):
    def load(url: str):
        calls.append(url)
        value = payloads[url]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    return load


def _payload_map(*, sprint: bool, results_by_kind: dict[SessionKind, list[dict]] | None = None):
    schedule_url = f"{JOLPICA_ALPHA_BASE}/schedules/2026/"
    payloads: dict[str, object] = {schedule_url: _alpha_schedule(sprint=sprint)}
    kinds = (
        (SessionKind.PRACTICE_1, SessionKind.SPRINT_QUALIFYING)
        if sprint
        else (SessionKind.PRACTICE_1, SessionKind.PRACTICE_2, SessionKind.PRACTICE_3)
    )
    codes = {
        SessionKind.PRACTICE_1: "FP1",
        SessionKind.PRACTICE_2: "FP2",
        SessionKind.PRACTICE_3: "FP3",
        SessionKind.SPRINT_QUALIFYING: "SQ",
    }
    for kind in kinds:
        results = None if results_by_kind is None else results_by_kind.get(kind, [])
        payloads[f"{JOLPICA_ALPHA_BASE}/results/round_test/{codes[kind]}/"] = session_payload(
            kind, results
        )
    return payloads


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_live_session_cache()
    yield
    clear_live_session_cache()


def test_normal_and_sprint_weekends_fetch_only_expected_distinct_sessions():
    normal_calls: list[str] = []
    normal = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.NORMAL,
        player_identity_map=_identity(),
        effective_time="2026-08-23T12:00:00Z",
        json_loader=_loader(_payload_map(sprint=False), normal_calls),
    )
    assert [state.kind for state in normal.states] == [
        SessionKind.PRACTICE_1,
        SessionKind.PRACTICE_2,
        SessionKind.PRACTICE_3,
    ]
    assert all(state.status == SessionStatus.COMPLETE for state in normal.states)
    assert set(normal.results["session_kind"]) == {"practice_1", "practice_2", "practice_3"}

    clear_live_session_cache()
    sprint_calls: list[str] = []
    sprint = ingest_active_event_sessions(
        _schedule_row(sprint=True),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        json_loader=_loader(_payload_map(sprint=True), sprint_calls),
    )
    assert [state.kind for state in sprint.states] == [
        SessionKind.PRACTICE_1,
        SessionKind.SPRINT_QUALIFYING,
    ]
    assert all(state.status == SessionStatus.COMPLETE for state in sprint.states)
    assert len(normal_calls) == 4
    assert len(sprint_calls) == 3


def test_empty_future_session_is_pending():
    empty = {kind: [] for kind in (SessionKind.PRACTICE_1, SessionKind.PRACTICE_2, SessionKind.PRACTICE_3)}
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.NORMAL,
        player_identity_map=_identity(),
        effective_time="2026-08-21T09:00:00Z",
        json_loader=_loader(_payload_map(sprint=False, results_by_kind=empty), []),
    )
    assert all(state.status == SessionStatus.PENDING for state in result.states)


def test_partial_nonempty_response_is_not_complete():
    partial_rows = [
        _result("LAW", "Liam", "Lawson", 1),
        _result("TSU", "Yuki", "Tsunoda", 2),
    ]
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.NORMAL,
        player_identity_map=_identity(),
        effective_time="2026-08-23T12:00:00Z",
        json_loader=_loader(
            _payload_map(
                sprint=False,
                results_by_kind={kind: partial_rows for kind in (SessionKind.PRACTICE_1, SessionKind.PRACTICE_2, SessionKind.PRACTICE_3)},
            ),
            [],
        ),
    )
    assert all(state.status == SessionStatus.PARTIAL for state in result.states)
    assert all(state.observed_row_count == 2 for state in result.states)
    assert all(state.expected_participant_count == 3 for state in result.states)


def test_full_coverage_during_scheduled_session_is_in_progress():
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.NORMAL,
        player_identity_map=_identity(),
        effective_time="2026-08-21T10:45:00Z",
        json_loader=_loader(_payload_map(sprint=False), []),
    )
    assert result.states[0].status == SessionStatus.IN_PROGRESS


def test_nonempty_rows_without_independent_expected_field_are_not_assumed_complete():
    result = ingest_active_event_sessions(
        _schedule_row(sprint=True),
        format=WeekendFormat.SPRINT,
        player_identity_map=pd.DataFrame(),
        effective_time="2026-08-22T12:00:00Z",
        json_loader=_loader(_payload_map(sprint=True), []),
    )
    assert all(state.status == SessionStatus.PARTIAL for state in result.states)


def test_source_provisional_metadata_stays_provisional():
    payloads = _payload_map(sprint=True)
    sq_url = f"{JOLPICA_ALPHA_BASE}/results/round_test/SQ/"
    payloads[sq_url]["data"]["status"] = "provisional"
    result = ingest_active_event_sessions(
        _schedule_row(sprint=True),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        json_loader=_loader(payloads, []),
    )
    assert result.states[1].status == SessionStatus.PROVISIONAL


def test_malformed_payload_and_source_exception_are_distinct():
    payloads = _payload_map(sprint=True)
    fp1_url = f"{JOLPICA_ALPHA_BASE}/results/round_test/FP1/"
    sq_url = f"{JOLPICA_ALPHA_BASE}/results/round_test/SQ/"
    payloads[fp1_url] = {"bad": True}
    payloads[sq_url] = RuntimeError("source unavailable")
    result = ingest_active_event_sessions(
        _schedule_row(sprint=True),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        json_loader=_loader(payloads, []),
    )
    assert result.states[0].status == SessionStatus.MALFORMED
    assert result.states[1].status == SessionStatus.FAILED


def test_current_human_field_supports_22_without_hard_coded_20():
    identities = pd.DataFrame(
        [
            {
                "fantasy_asset_id": index,
                "human_driver_id": f"driver_{index}",
                "tla": f"X{index}",
                "display_name": f"Driver {index}",
                "active": True,
                "match_status": "matched",
            }
            for index in range(22)
        ]
    )
    rows = [
        _result(f"X{index}", "Driver", str(index), index + 1, driver_id=f"driver_{index}")
        for index in range(22)
    ]
    payloads = _payload_map(
        sprint=True,
        results_by_kind={SessionKind.PRACTICE_1: rows, SessionKind.SPRINT_QUALIFYING: rows},
    )
    result = ingest_active_event_sessions(
        _schedule_row(sprint=True),
        format=WeekendFormat.SPRINT,
        player_identity_map=identities,
        effective_time="2026-08-22T12:00:00Z",
        json_loader=_loader(payloads, []),
    )
    assert all(state.expected_participant_count == 22 for state in result.states)


def test_pending_cache_is_bounded_complete_cache_is_reusable_and_refresh_bypasses():
    empty = {kind: [] for kind in (SessionKind.PRACTICE_1, SessionKind.PRACTICE_2, SessionKind.PRACTICE_3)}
    calls: list[str] = []
    payloads = _payload_map(sprint=False, results_by_kind=empty)
    loader = _loader(payloads, calls)
    kwargs = {
        "format": WeekendFormat.NORMAL,
        "player_identity_map": _identity(),
        "json_loader": loader,
    }
    ingest_active_event_sessions(_schedule_row(), effective_time="2026-08-21T09:00:00Z", **kwargs)
    ingest_active_event_sessions(_schedule_row(), effective_time="2026-08-21T09:01:00Z", **kwargs)
    assert len(calls) == 4
    ingest_active_event_sessions(_schedule_row(), effective_time="2026-08-21T09:03:01Z", **kwargs)
    assert len(calls) == 7

    clear_live_session_cache()
    calls.clear()
    final_loader = _loader(_payload_map(sprint=False), calls)
    kwargs["json_loader"] = final_loader
    ingest_active_event_sessions(_schedule_row(), effective_time="2026-08-23T12:00:00Z", **kwargs)
    ingest_active_event_sessions(_schedule_row(), effective_time="2026-08-23T12:10:00Z", **kwargs)
    assert len(calls) == 4
    ingest_active_event_sessions(
        _schedule_row(),
        effective_time="2026-08-23T12:11:00Z",
        force_refresh=True,
        **kwargs,
    )
    assert len(calls) == 8


def _snapshot(marker: str, *, session_status: SessionStatus = SessionStatus.COMPLETE):
    state = SessionState(
        event=EVENT,
        kind=SessionKind.PRACTICE_1,
        scheduled_at=None,
        observed_row_count=1 if session_status == SessionStatus.COMPLETE else 0,
        expected_participant_count=1,
        status=session_status,
        source="test",
    )
    return app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2026,
        requested_seasons=(2026,),
        loaded_seasons=(2026,),
        season_load_failures={},
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.DataFrame(),
        players=pd.DataFrame(),
        teams=pd.DataFrame(),
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={"raw_live_load_finished_utc": marker},
        session_results=pd.DataFrame(
            [
                {
                    "season": 2026,
                    "round": 12,
                    "session_kind": "practice_1",
                    "source_driver_id": "driver_law",
                    "human_driver_id": "lawson",
                    "position": 1,
                }
            ]
        )
        if session_status == SessionStatus.COMPLETE
        else pd.DataFrame(),
        session_states=(state,),
    )


def test_snapshot_copy_is_defensive_and_signature_is_deterministic():
    snapshot = _snapshot("one")
    copied = app_core.copy_live_data_snapshot(snapshot)
    copied.session_results.loc[0, "position"] = 99
    assert snapshot.session_results.loc[0, "position"] == 1
    assert live_session_signature(snapshot.session_results, snapshot.session_states) == live_session_signature(
        snapshot.session_results.sample(frac=1), snapshot.session_states
    )


def test_ordinary_rerun_and_model_setting_rerun_do_not_call_session_source():
    calls: list[bool] = []
    snapshot = _snapshot("one")
    reused = app_core.resolve_live_data_snapshot(
        snapshot,
        False,
        lambda force: calls.append(force) or _snapshot("unexpected"),
    )
    assert calls == []
    derives: list[str] = []
    model = app_core.ModelData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
    resolved = app_core.resolve_derived_model_data(
        reused["snapshot"], model, ("same",), ("same",), lambda _snapshot: derives.append("derive") or model
    )
    assert derives == []
    assert resolved["status"] == "reused"


def test_refresh_failure_retains_last_good_rows_but_exposes_failed_state():
    current = _snapshot("one")
    failed = _snapshot("two", session_status=SessionStatus.FAILED)
    resolved = app_core.resolve_live_data_snapshot(current, True, lambda _force: failed)

    assert resolved["result_accepted"] is True
    assert len(resolved["snapshot"].session_results) == 1
    assert resolved["snapshot"].session_states[0].status == SessionStatus.FAILED
    assert resolved["snapshot"].source_diagnostics["live_session_last_good_retained"] is True


def test_session_failure_remains_noncritical_to_snapshot_acceptance():
    failed = _snapshot("two", session_status=SessionStatus.FAILED)
    resolved = app_core.resolve_live_data_snapshot(None, False, lambda _force: failed)
    assert resolved["result_accepted"] is True
    assert resolved["snapshot"] is not None


def test_raw_loader_places_ingestion_results_and_states_on_snapshot(monkeypatch):
    schedule = pd.DataFrame(
        [
            {
                **_schedule_row(sprint=True),
                "date": "2026-08-23",
                "time": "13:00:00Z",
                "circuitName": "Circuit Park Zandvoort",
                "qualifying_date": "2026-08-22",
                "qualifying_time": "14:00:00Z",
                "sprint_date": "2026-08-22",
                "sprint_time": "10:00:00Z",
            }
        ]
    )
    monkeypatch.setattr(
        app_core,
        "fetch_all_supporting",
        lambda _year, force_refresh=False: {
            "results": pd.DataFrame(),
            "qualifying": pd.DataFrame(),
            "sprint": pd.DataFrame(),
            "schedule": schedule,
        },
    )
    players = pd.DataFrame(
        [{"playerId": 101, "name": "Liam Lawson", "tla": "LAW", "price": 20.0, "is_active": 1}]
    )
    teams = pd.DataFrame([{"teamId": 201, "name": "Red Bull", "price": 30.0, "is_active": 1}])
    monkeypatch.setattr(
        app_core,
        "resolve_market_data",
        lambda **_kwargs: {
            "feed_round": 12,
            "live_data_status": "fresh",
            "players": players,
            "teams": teams,
            "player_assets": players,
            "constructor_assets": teams,
            "asset_ledger_complete": True,
            "snapshot_round": 12,
        },
    )
    monkeypatch.setattr(app_core, "load_canonical_scores", lambda _path: pd.DataFrame())
    monkeypatch.setattr(app_core, "fetch_team_lock_deadline_from_playerstats", lambda _id: {})
    state = SessionState(
        event=EVENT,
        kind=SessionKind.PRACTICE_1,
        scheduled_at=None,
        observed_row_count=1,
        expected_participant_count=1,
        status=SessionStatus.COMPLETE,
        source="test",
    )
    ingested_rows = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 12,
                "session_kind": "practice_1",
                "human_driver_id": "lawson",
                "source_driver_id": "driver_law",
                "position": 1,
            }
        ]
    )
    session_calls: list[bool] = []
    monkeypatch.setattr(
        app_core,
        "ingest_active_event_sessions",
        lambda *_args, force_refresh=False, **_kwargs: session_calls.append(force_refresh)
        or app_core.LiveSessionIngestion(
            ingested_rows,
            (state,),
            {"source": "test", "sessions": {"practice_1": {"status": "complete"}}},
        ),
    )

    snapshot = app_core.load_live_data_snapshot(
        current_season=2026,
        historical_seasons_back=0,
        include_playerstats=False,
        effective_time="2026-08-21T12:00:00Z",
    )

    assert session_calls == [False]
    assert snapshot.session_results.iloc[0]["human_driver_id"] == "lawson"
    assert snapshot.session_states == (state,)
    assert snapshot.source_diagnostics["live_session_rows_observed"] == 1


def test_raw_session_data_does_not_change_any_production_model_table_or_ranking():
    snapshot, _ = load_offline_snapshot()
    baseline_snapshot = deepcopy(snapshot)
    session_snapshot = deepcopy(snapshot)
    session_snapshot.session_results = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 12,
                "session_kind": "practice_1",
                "human_driver_id": "lawson",
                "source_driver_id": "driver_law",
                "position": 1,
            }
        ]
    )
    session_snapshot.session_states = (
        SessionState(
            event=EVENT,
            kind=SessionKind.PRACTICE_1,
            scheduled_at=None,
            observed_row_count=1,
            expected_participant_count=1,
            status=SessionStatus.COMPLETE,
            source="test",
        ),
    )
    settings = {
        "today": "2026-08-10",
        "effective_time": "2026-08-10T12:00:00Z",
        "historical_seasons_back": 3,
        "horizon_races": 5,
        "current_season_weight": 1.0,
        "past_season_weight": 0.7,
        "recency_decay": 0.85,
        "selected_race_preset": "All",
        "history_mode": app_core.HISTORY_MODE_ALL_SUPPORTED,
    }
    baseline = app_core.derive_model_data(baseline_snapshot, **settings)
    with_sessions = app_core.derive_model_data(session_snapshot, **settings)
    failed_session_snapshot = deepcopy(snapshot)
    failed_session_snapshot.session_states = (
        SessionState(
            event=EVENT,
            kind=SessionKind.PRACTICE_1,
            scheduled_at=None,
            observed_row_count=0,
            expected_participant_count=22,
            status=SessionStatus.FAILED,
            source="test",
            diagnostic="source unavailable",
        ),
    )
    after_session_failure = app_core.derive_model_data(failed_session_snapshot, **settings)

    pd.testing.assert_frame_equal(baseline.drivers, with_sessions.drivers)
    pd.testing.assert_frame_equal(baseline.constructors, with_sessions.constructors)
    pd.testing.assert_frame_equal(baseline.trends, with_sessions.trends)
    pd.testing.assert_frame_equal(
        baseline.driver_price_efficiency, with_sessions.driver_price_efficiency
    )
    pd.testing.assert_frame_equal(
        baseline.constructor_price_efficiency, with_sessions.constructor_price_efficiency
    )
    pd.testing.assert_frame_equal(baseline.drivers, after_session_failure.drivers)
    pd.testing.assert_frame_equal(baseline.constructors, after_session_failure.constructors)

    budget, _ = deterministic_budget(baseline.drivers, baseline.constructors)
    baseline_ranked = app_core.run_optimizer(
        baseline.drivers, baseline.constructors, budget=budget, top_k=3
    )
    session_ranked = app_core.run_optimizer(
        with_sessions.drivers, with_sessions.constructors, budget=budget, top_k=3
    )
    def signatures(solutions):
        return [
            (
                tuple(sorted(solution.drivers["id"].astype(str))),
                tuple(sorted(solution.constructors["id"].astype(str))),
                solution.boosted_driver,
                solution.triple_driver,
                solution.expected_score,
                solution.total_cost,
            )
            for solution in solutions
        ]

    assert signatures(baseline_ranked) == signatures(session_ranked)
