import json
from pathlib import Path

import pandas as pd
import pytest

from f1fantasy import player_stats
from f1fantasy.app_core import (
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
    DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
    _add_playerstats_recent_points,
    apply_recent_point_overrides,
    apply_price_change_model,
    price_change_threshold_table,
)
from f1fantasy.player_stats import parse_player_race_points, fetch_recent_points_for_roster, latest_two_races


FIXTURE = Path(__file__).parent / "fixtures" / "playerstats_124_redacted.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _history_payload(player_id: int, records: list[tuple[int, float | None, int, str]]) -> dict:
    return {
        "Value": {
            "PlayerId": player_id,
            "PlayerSkill": "1",
            "GamedayWiseStats": [
                {
                    "GamedayId": round_no,
                    "IsPlayed": is_played,
                    "IsActive": is_played,
                    "StatsWise": [] if score is None else [{"Event": "Total", "Value": score}],
                }
                for round_no, score, is_played, _status in records
            ],
            "MatchWiseStats": [
                {
                    "GamedayId": round_no,
                    "RaceDayWise": [
                        {
                            "MeetingNumber": round_no,
                            "MeetingName": f"Round {round_no}",
                            "Season": "2026",
                            "MatchStatus": status,
                        }
                    ],
                }
                for round_no, _score, _is_played, status in records
            ],
        }
    }


def test_parser_extracts_race_by_race_fantasy_points():
    df = parse_player_race_points(_payload(), player_id=124)

    assert list(df["round"]) == [1, 2, 3]
    assert list(df["race_name"]) == ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]
    assert list(df["fantasy_points"]) == [39.0, 45.0, 27.0]
    assert df.loc[df["round"] == 2, "qualifying_points"].iloc[0] == 9.0
    assert df.loc[df["round"] == 2, "race_points"].iloc[0] == 24.0
    assert df.loc[df["round"] == 2, "sprint_points"].iloc[0] == 12.0
    assert df.loc[df["round"] == 2, "price_change"].iloc[0] == pytest.approx(0.3)


def test_parser_sums_complete_components_when_zero_total_item_is_omitted():
    payload = {
        "Value": {
            "PlayerId": 11051,
            "PlayerSkill": "1",
            "GamedayWiseStats": [
                {
                    "GamedayId": 14,
                    "PlayerValue": 8.6,
                    "OldPlayerValue": 8.0,
                    "IsPlayed": 1,
                    "IsActive": 1,
                    "StatsWise": [
                        {"Event": "Qualifying Position", "Value": 0},
                        {"Event": "Race Position lost", "Value": -1},
                        {"Event": "race overtake bonus", "Value": 1},
                    ],
                }
            ],
            "MatchWiseStats": [
                {
                    "GamedayId": 14,
                    "RaceDayWise": [
                        {
                            "MeetingNumber": 14,
                            "MeetingName": "Spanish Grand Prix",
                            "RaceDayId": 7383,
                            "Season": "2026",
                            "SessionType": "Race",
                            "StatsWise": [
                                {"Event": "Race Position lost", "Value": -1},
                                {"Event": "race overtake bonus", "Value": 1},
                            ],
                        }
                    ],
                }
            ],
        }
    }

    parsed = parse_player_race_points(payload, player_id=11051)

    assert parsed.loc[0, "fantasy_points"] == 0.0
    assert pd.isna(parsed.loc[0, "race_points"])


def test_parser_keeps_empty_or_partial_component_lists_missing():
    payload = {
        "Value": {
            "PlayerId": 1,
            "PlayerSkill": "1",
            "GamedayWiseStats": [
                {
                    "GamedayId": 1,
                    "IsPlayed": 1,
                    "StatsWise": [{"Event": "Race Position", "Value": None}],
                },
                {"GamedayId": 2, "IsPlayed": 1, "StatsWise": []},
            ],
            "MatchWiseStats": [],
        }
    }

    parsed = parse_player_race_points(payload, player_id=1)

    assert parsed["fantasy_points"].isna().all()


def test_recent_two_races_are_selected_from_playerstats(monkeypatch):
    monkeypatch.setattr(player_stats, "fetch_player_stats", lambda player_id: _payload())
    roster = pd.DataFrame([{"id": 124, "name": "George Russell"}])

    recent, race_points, diagnostics = fetch_recent_points_for_roster(roster, asset_type="driver")

    assert recent.loc[0, "recent_points_2ago"] == 45.0
    assert recent.loc[0, "recent_points_1ago"] == 27.0
    assert recent.loc[0, "recent_points_source"] == "playerstats"
    assert latest_two_races(race_points) == [
        {"round": 2, "race_name": "Chinese Grand Prix"},
        {"round": 3, "race_name": "Japanese Grand Prix"},
    ]
    assert diagnostics["playerstats_assets_loaded"] == 1


def test_active_asset_uses_latest_two_official_scores(monkeypatch):
    payload = _history_payload(6, [(10, 10.0, 1, "4"), (11, 8.0, 1, "4")])
    monkeypatch.setattr(player_stats, "fetch_player_stats", lambda _player_id: payload)

    recent, _races, _diagnostics = fetch_recent_points_for_roster(
        pd.DataFrame([{"id": 6, "name": "Active Driver"}]), asset_type="driver"
    )

    assert recent.loc[0, "recent_points_2ago"] == 10.0
    assert recent.loc[0, "recent_points_1ago"] == 8.0


def test_recent_history_preserves_official_zeroes_across_inactivity_and_team_change(monkeypatch):
    payload = _history_payload(
        7,
        [(10, 18.0, 1, "4"), (11, None, 0, "4"), (12, None, 0, "4"), (13, None, 0, "1")],
    )
    monkeypatch.setattr(player_stats, "fetch_player_stats", lambda _player_id: payload)

    recent, _races, _diagnostics = fetch_recent_points_for_roster(
        pd.DataFrame([{"id": 7, "name": "Returning Driver", "team": "New Team"}]),
        asset_type="driver",
    )

    assert recent.loc[0, "recent_points_2ago"] == 0.0
    assert recent.loc[0, "recent_points_1ago"] == 0.0
    assert recent.loc[0, "recent_points_available"] == 2

    parsed = parse_player_race_points(payload, player_id=7).set_index("round")
    assert pd.isna(parsed.loc[11, "fantasy_points_playerstats_total"])
    assert parsed.loc[11, "fantasy_points_official_ui"] == 0.0
    assert parsed.loc[11, "fantasy_points_source"] == "official_ui_inactive_zero"

    for price, expected, display in [
        (14.5, (26.1, 39.15, 52.2), ("≤ 26", "27 to 39", "40 to 52", "≥ 53")),
        (9.7, (17.46, 26.19, 34.92), ("≤ 17", "18 to 26", "27 to 34", "≥ 35")),
    ]:
        thresholds = price_change_threshold_table(
            recent.assign(name="Returning Driver", price=price, exp_score=0.0),
            DEFAULT_PRICE_CHANGE_CHEAP_RULES,
            expensive_rules=DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
            expensive_price_min=DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
        ).iloc[0]
        assert thresholds["price_history_mode"] == "established"
        assert thresholds["required_terrible_max"] == pytest.approx(expected[0])
        assert thresholds["required_good_min"] == pytest.approx(expected[1])
        assert thresholds["required_great_min"] == pytest.approx(expected[2])
        assert tuple(
            thresholds[column]
            for column in (
                "points_needed_terrible",
                "points_needed_poor",
                "points_needed_good",
                "points_needed_great",
            )
        ) == display


def test_completed_missing_scores_remain_missing_without_backfilling(monkeypatch):
    payload = _history_payload(
        8,
        [(10, 18.0, 1, "4"), (11, None, 0, "4"), (12, None, 0, "4"), (13, None, 1, "1")],
    )
    for gameday in payload["Value"]["GamedayWiseStats"]:
        if gameday["GamedayId"] in {11, 12}:
            gameday["StatsWise"] = [{"Event": "Race Position", "Value": None}]
    monkeypatch.setattr(player_stats, "fetch_player_stats", lambda _player_id: payload)

    recent, _races, _diagnostics = fetch_recent_points_for_roster(
        pd.DataFrame([{"id": 8, "name": "Returning Driver"}]),
        asset_type="driver",
    )

    assert pd.isna(recent.loc[0, "recent_points_2ago"])
    assert pd.isna(recent.loc[0, "recent_points_1ago"])
    assert recent.loc[0, "recent_points_available"] == 0
    assert recent.loc[0, "recent_points_source"] == "playerstats_incomplete"


def test_new_asset_without_completed_official_rounds_keeps_empty_history(monkeypatch):
    payload = _history_payload(9, [(13, None, 1, "1")])
    monkeypatch.setattr(player_stats, "fetch_player_stats", lambda _player_id: payload)

    recent, _races, _diagnostics = fetch_recent_points_for_roster(
        pd.DataFrame([{"id": 9, "name": "New Driver"}]),
        asset_type="driver",
    )

    assert pd.isna(recent.loc[0, "recent_points_2ago"])
    assert pd.isna(recent.loc[0, "recent_points_1ago"])
    assert recent.loc[0, "recent_points_available"] == 0


def test_missing_endpoint_does_not_become_zero(monkeypatch):
    def fail(_player_id):
        raise RuntimeError("not found")

    monkeypatch.setattr(player_stats, "fetch_player_stats", fail)
    roster = pd.DataFrame([{"id": 999999, "name": "Missing Asset"}])

    recent, _race_points, diagnostics = fetch_recent_points_for_roster(roster, asset_type="driver")

    assert pd.isna(recent.loc[0, "recent_points_2ago"])
    assert pd.isna(recent.loc[0, "recent_points_1ago"])
    assert recent.loc[0, "recent_points_available"] == 0
    assert recent.loc[0, "recent_points_source"] == "playerstats_failed"
    assert diagnostics["playerstats_assets_failed"] == 1


def test_manual_fallback_still_overrides_missing_playerstats():
    base = pd.DataFrame(
        [
            {
                "driverId": "a",
                "recent_points_2ago": pd.NA,
                "recent_points_1ago": pd.NA,
                "recent_points_available": 0,
                "recent_points_source": "playerstats_failed",
            }
        ]
    )
    manual = pd.DataFrame([{"driverId": "a", "recent_points_2ago": 11.0, "recent_points_1ago": 22.0}])

    out = apply_recent_point_overrides(base, manual, "driverId")

    assert out.loc[0, "recent_points_2ago"] == 11.0
    assert out.loc[0, "recent_points_1ago"] == 22.0
    assert out.loc[0, "recent_points_source"] == "manual"


def test_price_change_model_uses_true_playerstats_recent_points(monkeypatch):
    monkeypatch.setattr(player_stats, "fetch_player_stats", lambda player_id: _payload())
    roster = pd.DataFrame([{"id": 124, "name": "George Russell", "price": 28.0, "exp_score": 50.0}])

    assets, _race_points, _diag = _add_playerstats_recent_points(roster, "driver")
    priced = apply_price_change_model(assets, rules={
        "terrible_max": 0.5,
        "poor_min": 0.5,
        "poor_max": 1.0,
        "good_min": 1.0,
        "good_max": 2.0,
        "great_min": 2.0,
        "terrible_price_change": -0.6,
        "poor_price_change": -0.2,
        "good_price_change": 0.2,
        "great_price_change": 0.6,
    })

    assert priced.loc[0, "recent_points_2ago"] == 45.0
    assert priced.loc[0, "recent_points_1ago"] == 27.0
    assert priced.loc[0, "avg_ppm"] == (45.0 + 27.0 + 50.0) / 3.0 / 28.0


def test_playerstats_progress_reports_loaded_failed_skipped(monkeypatch):
    calls: list[dict] = []

    def _mock_fetch(player_id: int):
        if int(player_id) == 1:
            return _payload()
        raise RuntimeError("endpoint failure")

    monkeypatch.setattr(player_stats, "fetch_player_stats", _mock_fetch)
    roster = pd.DataFrame([{"id": 1, "name": "One"}, {"id": 2, "name": "Two"}])

    recent, _race_points, diagnostics = fetch_recent_points_for_roster(
        roster,
        asset_type="driver",
        progress_callback=lambda payload: calls.append(payload),
    )

    assert len(calls) >= 2
    assert calls[-1]["processed"] == 2
    assert calls[-1]["loaded"] >= 1
    assert calls[-1]["failed"] >= 1
    assert calls[-1]["skipped"] >= 0
    assert diagnostics["playerstats_assets_failed"] == 1
    assert recent.loc[recent["id"] == 2, "recent_points_source"].iloc[0] == "playerstats_failed"
