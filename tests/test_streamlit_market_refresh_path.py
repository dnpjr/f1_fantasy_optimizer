from __future__ import annotations

import time

import pandas as pd

from f1fantasy import app_core, fantasy_api, player_stats


def _market(price: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = pd.DataFrame(
        [
            {
                "playerId": 116,
                "name": "Liam Lawson",
                "tla": "LAW",
                "team": "Red Bull Racing",
                "price": price,
                "previous_price": 14.5,
                "official_price_change": price - 14.5,
                "is_active": 1,
            }
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "teamId": 29,
                "name": "Red Bull Racing",
                "tla": "RBR",
                "price": 31.2,
                "previous_price": 30.9,
                "official_price_change": 0.3,
                "is_active": 1,
            }
        ]
    )
    return players, teams


def _playerstats(asset_id: int, *, constructor: bool) -> dict:
    return {
        "Value": {
            "PlayerId": asset_id,
            "PlayerSkill": 2 if constructor else 1,
            "GamedayWiseStats": [
                {
                    "GamedayId": 12,
                    "PlayerValue": 14.5 if not constructor else 30.9,
                    "OldPlayerValue": 14.5 if not constructor else 30.6,
                    "IsPlayed": 1,
                    "IsActive": 1,
                    "StatsWise": [],
                },
                {
                    "GamedayId": 13,
                    "PlayerValue": 14.3 if not constructor else 31.2,
                    "OldPlayerValue": 14.5 if not constructor else 30.9,
                    "IsPlayed": 0,
                    "IsActive": 1,
                    "StatsWise": [],
                },
            ],
            "MatchWiseStats": [
                {
                    "GamedayId": 12,
                    "RaceDayWise": [
                        {
                            "MeetingNumber": 12,
                            "MeetingName": "Dutch Grand Prix",
                            "Season": 2026,
                            "RaceDayId": 1200 + asset_id,
                        }
                    ],
                },
                {
                    "GamedayId": 13,
                    "RaceDayWise": [
                        {
                            "MeetingNumber": 13,
                            "MeetingName": "Italian Grand Prix",
                            "Season": 2026,
                            "RaceDayId": 1300 + asset_id,
                        }
                    ],
                },
            ],
        }
    }


def _supporting() -> dict[str, pd.DataFrame]:
    return {
        "results": pd.DataFrame(
            [
                {
                    "season": 2026,
                    "round": 12,
                    "driverId": "lawson",
                    "driver": "Liam Lawson",
                    "constructorId": "red_bull",
                    "constructor": "Red Bull Racing",
                    "grid": 1,
                    "position": 1,
                    "status": "Running",
                    "fastestLapRank": 0,
                }
            ]
        ),
        "qualifying": pd.DataFrame(
            [
                {
                    "season": 2026,
                    "round": 12,
                    "driverId": "lawson",
                    "position": 1,
                    "q1": "1:20",
                    "q2": "1:19",
                    "q3": "1:18",
                }
            ]
        ),
        "sprint": pd.DataFrame(),
        "schedule": pd.DataFrame(
            [
                {
                    "season": 2026,
                    "round": 12,
                    "raceName": "Dutch Grand Prix",
                    "circuitName": "Zandvoort",
                    "date": "2026-08-23",
                    "time": "13:00:00Z",
                    "qualifying_date": "2026-08-22",
                    "qualifying_time": "14:00:00Z",
                    "sprint_date": "",
                    "sprint_time": "",
                },
                {
                    "season": 2026,
                    "round": 13,
                    "raceName": "Italian Grand Prix",
                    "circuitName": "Monza",
                    "date": "2026-09-06",
                    "time": "13:00:00Z",
                    "qualifying_date": "2026-09-05",
                    "qualifying_time": "14:00:00Z",
                    "sprint_date": "",
                    "sprint_time": "",
                    "practice_1_date": "2026-09-04",
                    "practice_1_time": "11:00:00Z",
                    "practice_2_date": "2026-09-04",
                    "practice_2_time": "15:00:00Z",
                    "practice_3_date": "2026-09-05",
                    "practice_3_time": "10:00:00Z",
                },
            ]
        ),
    }


def _previous_snapshot(players: pd.DataFrame, teams: pd.DataFrame) -> app_core.LiveDataSnapshot:
    support = _supporting()
    previous_results = support["results"].copy(deep=True)
    previous_results["status"] = "Finished"
    return app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2026,
        requested_seasons=(2026,),
        loaded_seasons=(2026,),
        season_load_failures={},
        results=previous_results,
        qualifying=support["qualifying"],
        sprint=support["sprint"],
        schedule=support["schedule"],
        players=players,
        teams=teams,
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={
            "feed_round": 12,
            "live_data_status": "fresh",
            "market_content_signature": fantasy_api.market_content_signature(players, teams),
            "snapshot_validation_status": "valid",
            "snapshot_validation_safe_for_scoring": True,
            "weekend_state_validation": {},
        },
        player_assets=players,
        constructor_assets=teams,
    )


def _configure_runtime(monkeypatch, tmp_path, *, discovery_observations: list[dict]):
    old_players, old_teams = _market(14.5)
    new_players, new_teams = _market(14.3)
    cache_path = tmp_path / "verified-market.json"
    monkeypatch.setattr(fantasy_api, "MIN_CACHEABLE_DRIVER_ROWS", 1)
    monkeypatch.setattr(fantasy_api, "MIN_CACHEABLE_CONSTRUCTOR_ROWS", 1)
    monkeypatch.setattr(fantasy_api, "VERIFIED_MARKET_CACHE_PATH", cache_path)
    monkeypatch.setattr(app_core, "fetch_all_supporting", lambda *_args, **_kwargs: _supporting())
    monkeypatch.setattr(app_core, "load_canonical_scores", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(
        app_core,
        "canonical_market_snapshot",
        lambda *_args: {
            "players": old_players,
            "teams": old_teams,
            "round": 12,
            "event_name": "Dutch Grand Prix",
        },
    )

    def failed_latest():
        discovery_observations.append(
            {
                "latest_round_cache": fantasy_api._LATEST_FEED_CACHE["round"],
                "market_cache": tuple(sorted(fantasy_api._MARKET_CACHE)),
            }
        )
        raise RuntimeError("feed 20 returned 403 XML")

    monkeypatch.setattr(app_core, "_latest_feed_round", failed_latest)
    monkeypatch.setattr(
        app_core,
        "fetch_market_asset_ledgers",
        lambda feed_round: {
            "players": new_players,
            "teams": new_teams,
            "player_assets": new_players,
            "constructor_assets": new_teams,
        }
        if int(feed_round) == 13
        else (_ for _ in ()).throw(AssertionError(f"unexpected feed {feed_round}")),
    )
    monkeypatch.setattr(
        player_stats,
        "fetch_player_stats",
        lambda asset_id: _playerstats(int(asset_id), constructor=int(asset_id) == 29),
    )
    monkeypatch.setattr(app_core, "fetch_team_lock_deadline_from_playerstats", lambda _id: {})
    monkeypatch.setattr(
        app_core,
        "ingest_active_event_sessions",
        lambda *_args, **_kwargs: app_core.LiveSessionIngestion(
            results=app_core.empty_session_results(),
            states=(),
            diagnostics={"sessions": {}, "rows_observed": 0},
        ),
    )
    return old_players, old_teams, new_players, new_teams, cache_path


def test_clean_process_uses_advanced_official_gameday_13(monkeypatch, tmp_path):
    observations: list[dict] = []
    _old_players, _old_teams, _new_players, _new_teams, _cache_path = _configure_runtime(
        monkeypatch, tmp_path, discovery_observations=observations
    )

    snapshot = app_core.load_live_data_snapshot(
        current_season=2026,
        historical_seasons_back=0,
        include_playerstats=False,
        effective_time="2026-08-24T08:00:00Z",
    )

    assert snapshot.source_diagnostics["feed_round"] == 13
    assert snapshot.source_diagnostics["live_data_status"] == "fresh"
    assert snapshot.source_diagnostics["market_resolution_method"] == "active_gameday_verified"
    assert snapshot.source_diagnostics["market_expected_event_advanced"] is True
    assert snapshot.source_diagnostics["forecast_target_event"] == {
        "season": 2026,
        "round": 13,
    }
    assert snapshot.source_diagnostics["forecast_target_weekend_state"]["format"] == "normal"
    assert snapshot.players.loc[0, "price"] == 14.3


def test_running_process_refresh_keeps_feed_13_market_when_scoring_is_retained(
    monkeypatch, tmp_path
):
    observations: list[dict] = []
    old_players, old_teams, _new_players, _new_teams, cache_path = _configure_runtime(
        monkeypatch, tmp_path, discovery_observations=observations
    )
    fantasy_api.save_verified_market_cache(12, old_players, old_teams, path=cache_path)
    fantasy_api._LATEST_FEED_CACHE.update({"round": 12, "ts": time.time()})
    fantasy_api._MARKET_CACHE[12] = (time.time(), [{"PlayerId": 116}])
    previous = _previous_snapshot(old_players, old_teams)
    previous.source_diagnostics["live_data_status"] = "cached"

    resolved = app_core.resolve_live_data_snapshot(
        previous,
        True,
        lambda force_refresh: app_core.load_live_data_snapshot(
            current_season=2026,
            historical_seasons_back=0,
            include_playerstats=False,
            force_refresh=force_refresh,
            effective_time="2026-08-24T08:00:00Z",
            previous_snapshot=previous,
        ),
    )
    snapshot = resolved["snapshot"]

    assert observations == [{"latest_round_cache": 0, "market_cache": ()}]
    assert resolved["status"] == "market_refreshed_scoring_retained"
    assert snapshot.source_diagnostics["feed_round"] == 13
    assert snapshot.source_diagnostics["live_data_status"] == "fresh"
    assert snapshot.players.loc[0, "price"] == 14.3
    assert snapshot.player_assets.loc[0, "price"] == 14.3
    assert snapshot.results.loc[0, "status"] == "Finished"

    derived = app_core.resolve_derived_model_data(
        snapshot,
        None,
        None,
        ("feed-13",),
        lambda raw: app_core.ModelData(
            drivers=raw.players.rename(columns={"playerId": "id"}),
            constructors=raw.teams.rename(columns={"teamId": "id"}),
            trends=pd.DataFrame(),
            diagnostics={},
        ),
    )["data"]
    assert derived.drivers.loc[0, "price"] == 14.3
    assert derived.constructors.loc[0, "price"] == 31.2

    ui_status = app_core.market_runtime_status(snapshot)
    assert ui_status["state"] == "current"
    assert ui_status["feed_round"] == 13
    assert ui_status["show_stale_warning"] is False

    source_calls = 0

    def unexpected_loader(_force_refresh):
        nonlocal source_calls
        source_calls += 1
        return snapshot

    ordinary = app_core.resolve_live_data_snapshot(snapshot, False, unexpected_loader)
    assert ordinary["status"] == "reused"
    assert source_calls == 0


def test_cached_market_with_retained_scoring_never_reports_refresh_success():
    players, teams = _market(14.5)
    previous = _previous_snapshot(players, teams)
    cached = _previous_snapshot(players, teams)
    cached.source_diagnostics.update(
        {
            "live_data_status": "cached",
            "snapshot_validation_status": "unsafe_partial",
            "snapshot_validation_warnings": ["Dutch classification is partial."],
        }
    )

    resolved = app_core.resolve_live_data_snapshot(previous, True, lambda _force: cached)
    ui_status = app_core.market_runtime_status(resolved["snapshot"])

    assert resolved["status"] == "cached_market_scoring_retained"
    assert ui_status["state"] == "cached"
    assert ui_status["show_stale_warning"] is True
