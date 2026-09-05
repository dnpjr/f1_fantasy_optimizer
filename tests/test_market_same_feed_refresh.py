from __future__ import annotations

import json

import pandas as pd
import pytest

from f1fantasy import app_core, fantasy_api


def _market_frames(price: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = pd.DataFrame(
        [
            {
                "playerId": index + 1,
                "name": f"Driver {index + 1}",
                "team": f"Team {(index % 10) + 1}",
                "price": price if index == 0 else 10.0 + index / 10,
                "previous_price": 9.5 if index == 0 else 9.8 + index / 10,
            }
            for index in range(20)
        ]
    )
    players["official_price_change"] = players["price"] - players["previous_price"]
    teams = pd.DataFrame(
        [
            {
                "teamId": 100 + index,
                "name": f"Team {index + 1}",
                "price": 20.0 + index,
                "previous_price": 19.5 + index,
                "official_price_change": 0.5,
            }
            for index in range(10)
        ]
    )
    return players, teams


def _snapshot(players: pd.DataFrame, teams: pd.DataFrame, *, fetched_at: str) -> app_core.LiveDataSnapshot:
    signature = fantasy_api.market_content_signature(players, teams)
    return app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2026,
        requested_seasons=(2026,),
        loaded_seasons=(2026,),
        season_load_failures={},
        results=pd.DataFrame([{"season": 2026, "round": 1, "driverId": "driver_1"}]),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.DataFrame([{"season": 2026, "round": 1}]),
        players=players,
        teams=teams,
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={
            "raw_live_load_finished_utc": fetched_at,
            "live_data_verified_at_utc": fetched_at,
            "market_content_signature": signature,
            "historical_fantasy_data_version": "test",
        },
        player_assets=players,
        constructor_assets=teams,
    )


def test_same_feed_refresh_accepts_changed_market_content_and_rewrites_verified_cache(tmp_path):
    cache_path = tmp_path / "market.json"
    old_players, teams = _market_frames(10.0)
    new_players, _ = _market_frames(11.2)
    old_cache = fantasy_api.save_verified_market_cache(12, old_players, teams, path=cache_path)

    refreshed = fantasy_api.resolve_market_data(
        latest_feed_loader=lambda: 12,
        player_loader=lambda **_kwargs: new_players,
        team_loader=lambda **_kwargs: teams,
        cache_path=cache_path,
    )

    assert refreshed["feed_round"] == 12
    assert refreshed["live_data_status"] == "fresh"
    assert refreshed["content_changed"] is True
    assert refreshed["content_signature"] != old_cache["content_signature"]
    assert refreshed["players"].loc[0, "price"] == pytest.approx(11.2)
    persisted = fantasy_api.load_verified_market_cache(path=cache_path)
    assert persisted["players"].loc[0, "price"] == pytest.approx(11.2)
    assert persisted["content_signature"] == refreshed["content_signature"]
    assert app_core.live_data_snapshot_identity(
        _snapshot(old_players, teams, fetched_at="2026-08-24T10:00:00+00:00")
    ) != app_core.live_data_snapshot_identity(
        _snapshot(new_players, teams, fetched_at="2026-08-24T10:00:00+00:00")
    )


def test_same_feed_refresh_with_identical_content_does_not_change_snapshot_identity(tmp_path):
    cache_path = tmp_path / "market.json"
    players, teams = _market_frames(11.2)
    fantasy_api.save_verified_market_cache(12, players, teams, path=cache_path)

    refreshed = fantasy_api.resolve_market_data(
        latest_feed_loader=lambda: 12,
        player_loader=lambda **_kwargs: players.copy(deep=True),
        team_loader=lambda **_kwargs: teams.copy(deep=True),
        cache_path=cache_path,
    )

    assert refreshed["content_changed"] is False
    first = _snapshot(players, teams, fetched_at="2026-08-24T10:00:00+00:00")
    second = _snapshot(players, teams, fetched_at="2026-08-24T11:00:00+00:00")
    assert app_core.live_data_snapshot_identity(first) == app_core.live_data_snapshot_identity(second)
    source_calls = 0

    def unexpected_loader(_force_refresh: bool) -> app_core.LiveDataSnapshot:
        nonlocal source_calls
        source_calls += 1
        return second

    reused = app_core.resolve_live_data_snapshot(second, False, unexpected_loader)
    assert reused["status"] == "reused"
    assert source_calls == 0


def test_legacy_verified_cache_is_loaded_and_given_a_content_signature(tmp_path):
    cache_path = tmp_path / "market.json"
    players, teams = _market_frames()
    fantasy_api.save_verified_market_cache(12, players, teams, path=cache_path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["cache_version"] = 1
    payload.pop("content_signature")
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = fantasy_api.load_verified_market_cache(path=cache_path)

    assert loaded["feed_round"] == 12
    assert len(loaded["content_signature"]) == 64
