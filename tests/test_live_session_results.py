from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from f1fantasy.live_sessions import (
    SESSION_RESULT_COLUMNS,
    expected_live_session_kinds,
    parse_jolpica_session_payload,
)
from f1fantasy.weekend_state import EventKey, SessionKind, WeekendFormat


EVENT = EventKey(2026, 12)


def _result(
    abbreviation: str,
    given_name: str,
    family_name: str,
    position: int,
    *,
    driver_id: str | None = None,
    team: str = "Example Team",
) -> dict:
    return {
        "driver": {
            "id": driver_id or f"driver_{abbreviation.casefold()}",
            "abbreviation": abbreviation,
            "given_name": given_name,
            "family_name": family_name,
        },
        "team": {"id": f"team_{team.casefold().replace(' ', '_')}", "name": team},
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
                _result("LAW", "Liam", "Lawson", 1, team="Red Bull"),
                _result("TSU", "Yuki", "Tsunoda", 2, team="RB F1 Team"),
                _result("LIN", "Arvid", "Lindblad", 3, team="RB F1 Team"),
            ],
        },
    }


@pytest.mark.parametrize(
    "kind",
    [
        SessionKind.PRACTICE_1,
        SessionKind.PRACTICE_2,
        SessionKind.PRACTICE_3,
        SessionKind.SPRINT_QUALIFYING,
    ],
)
def test_supported_session_payloads_parse_distinctly(kind):
    frame = parse_jolpica_session_payload(session_payload(kind), event=EVENT, kind=kind)

    assert list(frame.columns) == SESSION_RESULT_COLUMNS
    assert frame["session_kind"].unique().tolist() == [kind.value]
    assert frame["position"].astype(int).tolist() == [1, 2, 3]


def test_parser_preserves_human_identity_and_source_driver_identity():
    frame = parse_jolpica_session_payload(
        session_payload(SessionKind.PRACTICE_1),
        event=EVENT,
        kind=SessionKind.PRACTICE_1,
    )

    lawson = frame.loc[frame["abbreviation"].eq("LAW")].iloc[0]
    assert lawson["human_driver_id"] == "lawson"
    assert lawson["driver_reference"] == "driver_law"
    assert lawson["display_name"] == "Liam Lawson"
    assert lawson["team"] == "Red Bull"
    assert lawson["identity_match_method"] == "tla"


def test_parser_does_not_mutate_input_payload():
    payload = session_payload(SessionKind.PRACTICE_2)
    original = deepcopy(payload)

    parse_jolpica_session_payload(payload, event=EVENT, kind=SessionKind.PRACTICE_2)

    assert payload == original


def test_same_human_new_fantasy_asset_does_not_duplicate_session_driver():
    identity = pd.DataFrame(
        [
            {
                "fantasy_asset_id": 101,
                "human_driver_id": "lawson",
                "tla": "LAW",
                "display_name": "Liam Lawson",
                "active": False,
                "match_status": "matched",
            },
            {
                "fantasy_asset_id": 202,
                "human_driver_id": "lawson",
                "tla": "LAW",
                "display_name": "Liam Lawson",
                "active": True,
                "match_status": "matched",
            },
        ]
    )
    payload = session_payload(
        SessionKind.PRACTICE_1,
        [_result("LAW", "Liam", "Lawson", 1, team="Red Bull")],
    )

    frame = parse_jolpica_session_payload(
        payload,
        event=EVENT,
        kind=SessionKind.PRACTICE_1,
        player_identity_map=identity,
    )

    assert len(frame) == 1
    assert frame.iloc[0]["human_driver_id"] == "lawson"


def test_conflicting_identity_evidence_is_diagnostic_not_arbitrary():
    identity = pd.DataFrame(
        [
            {
                "fantasy_asset_id": 999,
                "human_driver_id": "different_lawson",
                "tla": "LAW",
                "display_name": "Liam Lawson",
                "active": True,
                "match_status": "matched",
            }
        ]
    )
    payload = session_payload(
        SessionKind.PRACTICE_1,
        [_result("LAW", "Liam", "Lawson", 1)],
    )

    frame = parse_jolpica_session_payload(
        payload,
        event=EVENT,
        kind=SessionKind.PRACTICE_1,
        player_identity_map=identity,
    )

    assert pd.isna(frame.iloc[0]["human_driver_id"])
    assert frame.iloc[0]["identity_match_status"] == "ambiguous"
    assert "Conflicting" in frame.iloc[0]["identity_diagnostic"]


def test_weekend_formats_require_only_prediction_input_sessions():
    assert expected_live_session_kinds(WeekendFormat.NORMAL) == (
        SessionKind.PRACTICE_1,
        SessionKind.PRACTICE_2,
        SessionKind.PRACTICE_3,
    )
    assert SessionKind.SPRINT_QUALIFYING not in expected_live_session_kinds(WeekendFormat.NORMAL)
    assert expected_live_session_kinds(WeekendFormat.SPRINT) == (
        SessionKind.PRACTICE_1,
        SessionKind.SPRINT_QUALIFYING,
    )
    assert SessionKind.PRACTICE_2 not in expected_live_session_kinds(WeekendFormat.SPRINT)
    assert SessionKind.PRACTICE_3 not in expected_live_session_kinds(WeekendFormat.SPRINT)


def test_parser_rejects_wrong_event_or_session_kind():
    with pytest.raises(ValueError, match="does not match"):
        parse_jolpica_session_payload(
            session_payload(SessionKind.PRACTICE_1),
            event=EVENT,
            kind=SessionKind.PRACTICE_2,
        )
    with pytest.raises(ValueError, match="different season or round"):
        parse_jolpica_session_payload(
            session_payload(SessionKind.PRACTICE_1),
            event=EventKey(2026, 13),
            kind=SessionKind.PRACTICE_1,
        )
