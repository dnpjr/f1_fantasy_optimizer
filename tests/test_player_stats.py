import json
from pathlib import Path

import pandas as pd
import pytest

from f1fantasy import player_stats
from f1fantasy.app_core import _add_playerstats_recent_points, apply_recent_point_overrides, apply_price_change_model
from f1fantasy.player_stats import parse_player_race_points, fetch_recent_points_for_roster, latest_two_races


FIXTURE = Path(__file__).parent / "fixtures" / "playerstats_124_redacted.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_parser_extracts_race_by_race_fantasy_points():
    df = parse_player_race_points(_payload(), player_id=124)

    assert list(df["round"]) == [1, 2, 3]
    assert list(df["race_name"]) == ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]
    assert list(df["fantasy_points"]) == [39.0, 45.0, 27.0]
    assert df.loc[df["round"] == 2, "qualifying_points"].iloc[0] == 9.0
    assert df.loc[df["round"] == 2, "race_points"].iloc[0] == 24.0
    assert df.loc[df["round"] == 2, "sprint_points"].iloc[0] == 12.0
    assert df.loc[df["round"] == 2, "price_change"].iloc[0] == pytest.approx(0.3)


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
