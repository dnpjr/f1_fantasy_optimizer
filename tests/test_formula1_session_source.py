from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from f1fantasy.live_sessions import (
    FORMULA1_API_BASE,
    FORMULA1_SOURCE,
    JOLPICA_ALPHA_BASE,
    JOLPICA_FALLBACK_SOURCE,
    clear_live_session_cache,
    ingest_active_event_sessions,
    parse_formula1_session_payload,
)
from f1fantasy.weekend_state import EventKey, SessionKind, SessionStatus, WeekendFormat


EVENT = EventKey(2026, 12)
MEETING_ID = 1292


def _official_row(tla: str, first: str, last: str, position: int, *, team: str) -> dict:
    return {
        "positionValue": str(position),
        "positionNumber": str(position),
        "lapsCompleted": str(20 + position),
        "classifiedTime": f"1:1{position}.000",
        "gapToLeader": "0" if position == 1 else f"0.{position}",
        "driverId": 1000 + position,
        "driverFirstName": first,
        "driverLastName": last,
        "driverReference": f"{first[:3]}{last[:3]}01".upper(),
        "driverTLA": tla,
        "teamId": 200 + position,
        "teamName": team,
        "displayTeamName": team,
        "displayTime": f"+0.{position}s",
        "completionStatusCode": "OK",
    }


def _official_payload(
    kind: SessionKind,
    results: list[dict] | None = None,
    *,
    state: str = "completed",
) -> dict:
    key, code, number, description = {
        SessionKind.PRACTICE_1: ("raceResultsPractice1", "p1", 1, "Practice 1"),
        SessionKind.PRACTICE_2: ("raceResultsPractice2", "p2", 2, "Practice 2"),
        SessionKind.PRACTICE_3: ("raceResultsPractice3", "p3", 3, "Practice 3"),
        SessionKind.SPRINT_QUALIFYING: (
            "raceResultsSprintShootout",
            "ss",
            0,
            "Sprint Qualifying",
        ),
    }[kind]
    rows = results if results is not None else [
        _official_row("LAW", "Liam", "Lawson", 1, team="Red Bull"),
        _official_row("TSU", "Yuki", "Tsunoda", 2, team="Racing Bulls"),
        _official_row("LIN", "Arvid", "Lindblad", 3, team="Racing Bulls"),
    ]
    return {
        key: {
            "session": code,
            "description": description,
            "startTime": "2026-08-21T14:30:00",
            "endTime": "2026-08-21T15:14:00",
            "gmtOffset": "+02:00",
            "state": state,
            "sessionNumber": number,
            "results": rows,
        }
    }


def _identity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fantasy_asset_id": 116, "human_driver_id": "lawson", "tla": "LAW", "display_name": "Liam Lawson", "active": True, "match_status": "matched"},
            {"fantasy_asset_id": 117, "human_driver_id": "tsunoda", "tla": "TSU", "display_name": "Yuki Tsunoda", "active": True, "match_status": "matched"},
            {"fantasy_asset_id": 118, "human_driver_id": "arvid_lindblad", "tla": "LIN", "display_name": "Arvid Lindblad", "active": True, "match_status": "matched"},
        ]
    )


def _schedule_row(*, sprint: bool = True) -> dict:
    return {
        "season": EVENT.season,
        "round": EVENT.round,
        "raceName": "Dutch Grand Prix",
        "practice_1_date": "2026-08-21",
        "practice_1_time": "10:30:00Z",
        "practice_2_date": "" if sprint else "2026-08-21",
        "practice_2_time": "" if sprint else "14:00:00Z",
        "practice_3_date": "" if sprint else "2026-08-22",
        "practice_3_time": "" if sprint else "10:30:00Z",
        "sprint_qualifying_date": "2026-08-21" if sprint else "",
        "sprint_qualifying_time": "14:30:00Z" if sprint else "",
    }


def _meeting_payload() -> list[dict]:
    return [
        {"value": 1200 + round_no, "text": f"Round {round_no}"}
        for round_no in range(1, EVENT.round)
    ] + [{"value": MEETING_ID, "text": "Netherlands"}]


def _dataset_payload(*, sprint: bool = True) -> list[dict]:
    datasets = [
        {
            "text": "Practice 1",
            "value": f"practice?meeting={MEETING_ID}&session=1",
            "isSessionResult": True,
            "isAvailable": True,
            "editorialSessionType": "Practice1",
        }
    ]
    if sprint:
        datasets.append(
            {
                "text": "Sprint Qualifying",
                "value": f"sprint-shootout?meeting={MEETING_ID}",
                "isSessionResult": True,
                "isAvailable": True,
                "editorialSessionType": "Sprint Shootout",
            }
        )
    else:
        for session in (2, 3):
            datasets.append(
                {
                    "text": f"Practice {session}",
                    "value": f"practice?meeting={MEETING_ID}&session={session}",
                    "isSessionResult": True,
                    "isAvailable": True,
                    "editorialSessionType": f"Practice{session}",
                }
            )
    return datasets


def _formula_urls(*, sprint: bool = True) -> tuple[str, str, dict[SessionKind, str]]:
    meetings = f"{FORMULA1_API_BASE}/dropdown-meetings?season=2026"
    datasets = f"{FORMULA1_API_BASE}/dropdown-meeting-datasets?meeting={MEETING_ID}"
    results = {
        SessionKind.PRACTICE_1: f"{FORMULA1_API_BASE}/practice?meeting={MEETING_ID}&session=1",
    }
    if sprint:
        results[SessionKind.SPRINT_QUALIFYING] = (
            f"{FORMULA1_API_BASE}/sprint-shootout?meeting={MEETING_ID}"
        )
    else:
        results.update(
            {
                SessionKind.PRACTICE_2: f"{FORMULA1_API_BASE}/practice?meeting={MEETING_ID}&session=2",
                SessionKind.PRACTICE_3: f"{FORMULA1_API_BASE}/practice?meeting={MEETING_ID}&session=3",
            }
        )
    return meetings, datasets, results


def _loader(payloads: dict[str, object], calls: list[str]):
    def load(url: str):
        calls.append(url)
        value = payloads[url]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    return load


def _formula_payload_map(
    *,
    sprint: bool = True,
    results: dict[SessionKind, object] | None = None,
) -> dict[str, object]:
    meetings, datasets, urls = _formula_urls(sprint=sprint)
    payloads: dict[str, object] = {
        meetings: _meeting_payload(),
        datasets: _dataset_payload(sprint=sprint),
    }
    for kind, url in urls.items():
        value = results.get(kind) if results is not None and kind in results else _official_payload(kind)
        payloads[url] = value
    return payloads


def _jolpica_payload_map(results: list[dict] | None = None) -> dict[str, object]:
    schedule_url = f"{JOLPICA_ALPHA_BASE}/schedules/2026/"
    fp1_url = f"{JOLPICA_ALPHA_BASE}/results/round_test/FP1/"
    sq_url = f"{JOLPICA_ALPHA_BASE}/results/round_test/SQ/"

    def payload(code: str) -> dict:
        rows = results if results is not None else [
            {
                "driver": {"id": "lawson", "abbreviation": "LAW", "given_name": "Liam", "family_name": "Lawson"},
                "team": {"id": "red_bull", "name": "Red Bull"},
                "position": 1,
                "position_text": "1",
                "is_classified": True,
            },
            {
                "driver": {"id": "tsunoda", "abbreviation": "TSU", "given_name": "Yuki", "family_name": "Tsunoda"},
                "team": {"id": "racing_bulls", "name": "Racing Bulls"},
                "position": 2,
                "position_text": "2",
                "is_classified": True,
            },
            {
                "driver": {"id": "lindblad", "abbreviation": "LIN", "given_name": "Arvid", "family_name": "Lindblad"},
                "team": {"id": "racing_bulls", "name": "Racing Bulls"},
                "position": 3,
                "position_text": "3",
                "is_classified": True,
            },
        ]
        return {"data": {"code": code, "season": {"year": 2026}, "round": {"number": 12}, "timestamp": "2026-08-21T14:30:00Z", "results": rows}}

    return {
        schedule_url: {
            "data": {
                "events": [
                    {
                        "round": {"number": 12},
                        "schedule": [
                            {"code": "FP1", "results_url": fp1_url},
                            {"code": "SQ", "results_url": sq_url},
                        ],
                    }
                ]
            }
        },
        fp1_url: payload("FP1"),
        sq_url: payload("SQ"),
    }


@pytest.fixture(autouse=True)
def _clear_source_cache():
    clear_live_session_cache()
    yield
    clear_live_session_cache()


@pytest.mark.parametrize(
    "kind",
    [
        SessionKind.PRACTICE_1,
        SessionKind.PRACTICE_2,
        SessionKind.PRACTICE_3,
        SessionKind.SPRINT_QUALIFYING,
    ],
)
def test_official_formula1_payloads_parse_to_existing_contract(kind):
    payload = _official_payload(kind)
    original = deepcopy(payload)

    frame = parse_formula1_session_payload(
        payload,
        event=EVENT,
        kind=kind,
        meeting_id=MEETING_ID,
        player_identity_map=_identity(),
        source_fetched_at="2026-08-21T16:00:00Z",
    )

    assert payload == original
    assert frame["session_kind"].unique().tolist() == [kind.value]
    assert frame["position"].astype(int).tolist() == [1, 2, 3]
    assert frame["human_driver_id"].tolist() == ["lawson", "tsunoda", "arvid_lindblad"]
    assert frame["source"].unique().tolist() == [FORMULA1_SOURCE]
    assert frame.iloc[0]["driver_reference"] == "LIALAW01"
    assert frame.iloc[0]["source_driver_id"] == 1001


def test_official_identity_is_human_based_and_ambiguous_evidence_is_diagnostic():
    identity = _identity()
    identity.loc[identity["human_driver_id"].eq("lawson"), "fantasy_asset_id"] = 99999
    matched = parse_formula1_session_payload(
        _official_payload(SessionKind.PRACTICE_1),
        event=EVENT,
        kind=SessionKind.PRACTICE_1,
        meeting_id=MEETING_ID,
        player_identity_map=identity,
    )
    assert matched.iloc[0]["human_driver_id"] == "lawson"

    identity.loc[identity["human_driver_id"].eq("lawson"), "human_driver_id"] = "other_lawson"
    ambiguous = parse_formula1_session_payload(
        _official_payload(SessionKind.PRACTICE_1),
        event=EVENT,
        kind=SessionKind.PRACTICE_1,
        meeting_id=MEETING_ID,
        player_identity_map=identity,
    )
    assert pd.isna(ambiguous.iloc[0]["human_driver_id"])
    assert ambiguous.iloc[0]["identity_match_status"] == "ambiguous"


def test_complete_formula1_wins_without_calling_jolpica():
    formula_calls: list[str] = []
    jolpica_calls: list[str] = []
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        formula1_json_loader=_loader(_formula_payload_map(), formula_calls),
        json_loader=_loader({}, jolpica_calls),
    )

    assert all(state.status == SessionStatus.COMPLETE for state in result.states)
    assert all(state.source == FORMULA1_SOURCE for state in result.states)
    assert set(result.results["source"]) == {FORMULA1_SOURCE}
    assert len(formula_calls) == 4
    assert jolpica_calls == []


def test_normal_weekend_fetches_only_official_fp1_fp2_fp3():
    formula_calls: list[str] = []
    jolpica_calls: list[str] = []
    result = ingest_active_event_sessions(
        _schedule_row(sprint=False),
        format=WeekendFormat.NORMAL,
        player_identity_map=_identity(),
        effective_time="2026-08-23T12:00:00Z",
        formula1_json_loader=_loader(
            _formula_payload_map(sprint=False), formula_calls
        ),
        json_loader=_loader({}, jolpica_calls),
    )

    assert [state.kind for state in result.states] == [
        SessionKind.PRACTICE_1,
        SessionKind.PRACTICE_2,
        SessionKind.PRACTICE_3,
    ]
    assert all(state.status == SessionStatus.COMPLETE for state in result.states)
    assert len(formula_calls) == 5
    assert jolpica_calls == []


@pytest.mark.parametrize("official_value", [{"bad": True}, RuntimeError("blocked")])
def test_malformed_or_failed_formula1_falls_back_once(official_value):
    formula_payloads = _formula_payload_map(
        results={SessionKind.PRACTICE_1: official_value}
    )
    jolpica_calls: list[str] = []
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        formula1_json_loader=_loader(formula_payloads, []),
        json_loader=_loader(_jolpica_payload_map(), jolpica_calls),
    )

    fp1 = result.states[0]
    assert fp1.status == SessionStatus.COMPLETE
    assert fp1.source == JOLPICA_FALLBACK_SOURCE
    assert result.diagnostics["sessions"]["practice_1"]["fallback_used"] is True
    assert jolpica_calls == [
        f"{JOLPICA_ALPHA_BASE}/schedules/2026/",
        f"{JOLPICA_ALPHA_BASE}/results/round_test/FP1/",
    ]


def test_empty_formula1_falls_back_and_empty_jolpica_does_not_become_complete():
    formula_payloads = _formula_payload_map(
        results={
            SessionKind.PRACTICE_1: _official_payload(SessionKind.PRACTICE_1, []),
        }
    )
    empty_jolpica = _jolpica_payload_map(results=[])
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-21T09:00:00Z",
        formula1_json_loader=_loader(formula_payloads, []),
        json_loader=_loader(empty_jolpica, []),
    )

    assert result.states[0].status == SessionStatus.PENDING
    assert result.diagnostics["sessions"]["practice_1"]["fallback_attempted"] is True


def test_partial_formula1_uses_complete_jolpica_fallback():
    partial = _official_payload(
        SessionKind.PRACTICE_1,
        [_official_row("LAW", "Liam", "Lawson", 1, team="Red Bull")],
    )
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        formula1_json_loader=_loader(
            _formula_payload_map(results={SessionKind.PRACTICE_1: partial}), []
        ),
        json_loader=_loader(_jolpica_payload_map(), []),
    )

    assert result.states[0].status == SessionStatus.COMPLETE
    assert result.states[0].source == JOLPICA_FALLBACK_SOURCE


def test_provisional_formula1_remains_nonfinal_when_fallback_fails():
    provisional = _official_payload(
        SessionKind.PRACTICE_1, state="provisional"
    )
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        formula1_json_loader=_loader(
            _formula_payload_map(results={SessionKind.PRACTICE_1: provisional}), []
        ),
        json_loader=lambda _url: (_ for _ in ()).throw(RuntimeError("fallback unavailable")),
    )

    assert result.states[0].status == SessionStatus.PROVISIONAL
    assert result.states[0].source == FORMULA1_SOURCE


def test_partial_duplicate_and_insufficient_official_tables_are_never_complete():
    partial = _official_payload(
        SessionKind.PRACTICE_1,
        [_official_row("LAW", "Liam", "Lawson", 1, team="Red Bull")],
    )
    duplicate_rows = [
        _official_row("LAW", "Liam", "Lawson", 1, team="Red Bull"),
        _official_row("LAW", "Liam", "Lawson", 2, team="Red Bull"),
    ]
    for payload, expected_status in (
        (partial, SessionStatus.PARTIAL),
        (_official_payload(SessionKind.PRACTICE_1, duplicate_rows), SessionStatus.MALFORMED),
    ):
        clear_live_session_cache()
        formula_payloads = _formula_payload_map(results={SessionKind.PRACTICE_1: payload})
        result = ingest_active_event_sessions(
            _schedule_row(),
            format=WeekendFormat.SPRINT,
            player_identity_map=_identity(),
            effective_time="2026-08-22T12:00:00Z",
            formula1_json_loader=_loader(formula_payloads, []),
            json_loader=lambda _url: (_ for _ in ()).throw(RuntimeError("fallback unavailable")),
        )
        assert result.states[0].status == expected_status


def test_formula1_complete_jolpica_empty_and_disagreement_validation_keep_formula1():
    for jolpica_rows, expect_disagreement in (([], False), (None, True)):
        clear_live_session_cache()
        jolpica_payloads = _jolpica_payload_map(results=jolpica_rows)
        if expect_disagreement:
            jolpica_payloads[f"{JOLPICA_ALPHA_BASE}/results/round_test/FP1/"]["data"]["results"][0]["position"] = 2
            jolpica_payloads[f"{JOLPICA_ALPHA_BASE}/results/round_test/FP1/"]["data"]["results"][0]["position_text"] = "2"
            jolpica_payloads[f"{JOLPICA_ALPHA_BASE}/results/round_test/FP1/"]["data"]["results"][1]["position"] = 1
            jolpica_payloads[f"{JOLPICA_ALPHA_BASE}/results/round_test/FP1/"]["data"]["results"][1]["position_text"] = "1"
        result = ingest_active_event_sessions(
            _schedule_row(),
            format=WeekendFormat.SPRINT,
            player_identity_map=_identity(),
            effective_time="2026-08-22T12:00:00Z",
            formula1_json_loader=_loader(_formula_payload_map(), []),
            json_loader=_loader(jolpica_payloads, []),
            validate_sources=True,
        )
        assert result.states[0].status == SessionStatus.COMPLETE
        assert result.states[0].source == FORMULA1_SOURCE
        diagnostic = result.diagnostics["sessions"]["practice_1"]
        if expect_disagreement:
            assert diagnostic["cross_source_validation"]["disagrees"] is True
            assert "Formula 1 remains authoritative" in diagnostic["source_disagreement"]
        else:
            assert "cross_source_validation" not in diagnostic


def test_equal_complete_sources_validate_without_warning_or_state_churn():
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        formula1_json_loader=_loader(_formula_payload_map(), []),
        json_loader=_loader(_jolpica_payload_map(), []),
        validate_sources=True,
    )

    diagnostic = result.diagnostics["sessions"]["practice_1"]
    assert diagnostic["cross_source_validation"]["disagrees"] is False
    assert "source_disagreement" not in diagnostic
    assert result.states[0].source == FORMULA1_SOURCE


def test_complete_formula1_cache_reuses_results_and_refresh_revalidates_once_per_session():
    calls: list[str] = []
    loader = _loader(_formula_payload_map(), calls)
    kwargs = {
        "format": WeekendFormat.SPRINT,
        "player_identity_map": _identity(),
        "formula1_json_loader": loader,
        "json_loader": _loader({}, []),
    }
    ingest_active_event_sessions(
        _schedule_row(), effective_time="2026-08-22T12:00:00Z", **kwargs
    )
    ingest_active_event_sessions(
        _schedule_row(), effective_time="2026-08-22T12:01:00Z", **kwargs
    )
    assert len(calls) == 4
    ingest_active_event_sessions(
        _schedule_row(),
        effective_time="2026-08-22T12:02:00Z",
        force_refresh=True,
        **kwargs,
    )
    assert len(calls) == 8
    _, _, result_urls = _formula_urls()
    assert calls[-2:] == list(result_urls.values())


def test_both_sources_failed_remains_failed_without_fabricated_rows():
    result = ingest_active_event_sessions(
        _schedule_row(),
        format=WeekendFormat.SPRINT,
        player_identity_map=_identity(),
        effective_time="2026-08-22T12:00:00Z",
        formula1_json_loader=lambda _url: (_ for _ in ()).throw(RuntimeError("official down")),
        json_loader=lambda _url: (_ for _ in ()).throw(RuntimeError("fallback down")),
    )
    assert all(state.status == SessionStatus.FAILED for state in result.states)
    assert result.results.empty
