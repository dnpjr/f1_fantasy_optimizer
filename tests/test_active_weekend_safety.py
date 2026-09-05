from __future__ import annotations

from datetime import UTC, datetime
import json

import pandas as pd
import pytest

from f1fantasy import app_core, ergast, fantasy_api
from f1fantasy.model import compute_weekend_points
from f1fantasy.weekend_state import EventKey


def _result_rows(*, status: str = "Finished", round_no: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": round_no,
                "circuitName": "Safety Circuit",
                "driverId": driver_id,
                "driver": name,
                "constructorId": "team",
                "constructor": "Safety Team",
                "grid": position,
                "position": position,
                "status": status if position == 1 else "Finished",
                "fastestLapRank": 0,
            }
            for position, (driver_id, name) in enumerate(
                [("d1", "Driver One"), ("d2", "Driver Two")], start=1
            )
        ]
    )


def _qualifying_rows(*, include_second: bool = True, round_no: int = 5) -> pd.DataFrame:
    drivers = [("d1", 1), ("d2", 2)] if include_second else [("d1", 1)]
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": round_no,
                "driverId": driver_id,
                "position": position,
                "q1": "1:20.000",
                "q2": "1:19.000",
                "q3": "1:18.000",
            }
            for driver_id, position in drivers
        ]
    )


def _sprint_rows(*, status: str = "Finished", include_second: bool = True) -> pd.DataFrame:
    drivers = [("d1", 1), ("d2", 2)] if include_second else [("d1", 1)]
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 5,
                "circuitName": "Safety Circuit",
                "driverId": driver_id,
                "driver": f"Driver {position}",
                "constructorId": "team",
                "grid": position,
                "position": position,
                "status": status if position == 1 else "Finished",
            }
            for driver_id, position in drivers
        ]
    )


def _snapshot(marker: str, validation_status: str = "valid") -> app_core.LiveDataSnapshot:
    return app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2026,
        requested_seasons=(2026,),
        loaded_seasons=(2026,),
        season_load_failures={},
        results=pd.DataFrame([{"marker": marker}]),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.DataFrame([{"marker": marker}]),
        players=pd.DataFrame([{"playerId": 1, "name": marker}]),
        teams=pd.DataFrame([{"teamId": 1, "name": marker}]),
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={
            "snapshot_validation_status": validation_status,
            "snapshot_validation_warnings": ["Live race is provisional."],
            "weekend_state_validation": {"marker": marker},
        },
    )


def test_running_result_is_excluded_instead_of_scored_as_dnf():
    results = _result_rows(status="Running")
    original = results.copy(deep=True)

    scored = compute_weekend_points(
        results,
        _qualifying_rows(),
        pd.DataFrame(),
        current_season=2026,
    )

    assert scored.empty
    pd.testing.assert_frame_equal(results, original)


def test_missing_qualifying_is_not_minus_five_until_source_is_complete():
    scored = compute_weekend_points(
        _result_rows(),
        _qualifying_rows(include_second=False),
        pd.DataFrame(),
        current_season=2026,
        completed_event_keys={EventKey(2026, 5)},
        complete_qualifying_keys=set(),
    )

    assert scored["driverId"].tolist() == ["d1"]
    assert -5 not in scored["quali_points"].tolist()


def test_missing_sprint_is_not_zero_until_sprint_source_is_complete():
    scored = compute_weekend_points(
        _result_rows(),
        _qualifying_rows(),
        _sprint_rows(include_second=False),
        current_season=2026,
        completed_event_keys={EventKey(2026, 5)},
        complete_qualifying_keys={EventKey(2026, 5)},
        complete_sprint_keys=set(),
    )

    assert scored["driverId"].tolist() == ["d1"]
    assert scored.loc[0, "sprint_points"] != 0


def test_complete_gate_preserves_existing_final_scoring_math():
    results = _result_rows()
    qualifying = _qualifying_rows()
    sprint = _sprint_rows()
    legacy = compute_weekend_points(results, qualifying, sprint, current_season=2026)
    gated = compute_weekend_points(
        results,
        qualifying,
        sprint,
        current_season=2026,
        completed_event_keys={EventKey(2026, 5)},
        complete_qualifying_keys={EventKey(2026, 5)},
        complete_sprint_keys={EventKey(2026, 5)},
    )

    pd.testing.assert_frame_equal(legacy, gated)


def test_unsafe_partial_refresh_accepts_market_but_retains_last_good_scoring():
    previous = _snapshot("previous")
    partial = app_core.resolve_live_data_snapshot(
        previous,
        True,
        lambda _force: _snapshot("partial", "unsafe_partial"),
    )

    assert partial["snapshot"].results.iloc[0]["marker"] == "previous"
    assert partial["snapshot"].players.iloc[0]["name"] == "partial"
    assert partial["result_accepted"] is True
    assert partial["status"] == "market_refreshed_scoring_retained"
    assert partial["snapshot"].source_diagnostics["scoring_data_status"] == "retained_last_good"
    assert partial["live_diagnostics"] == {"marker": "partial"}

    complete = app_core.resolve_live_data_snapshot(
        partial["snapshot"],
        True,
        lambda _force: _snapshot("complete", "valid"),
    )
    assert complete["snapshot"].results.iloc[0]["marker"] == "complete"
    assert complete["result_accepted"] is True
    assert complete["status"] == "refreshed"


def test_unsafe_partial_first_load_is_available_only_with_explicit_warning():
    loaded = app_core.resolve_live_data_snapshot(
        None,
        False,
        lambda _force: _snapshot("partial", "unsafe_partial"),
    )

    assert loaded["snapshot"].results.iloc[0]["marker"] == "partial"
    assert loaded["status"] == "loaded_with_partial_sessions"
    assert loaded["result_accepted"] is True


def test_partial_current_round_is_absent_from_completed_race_catalogue():
    snapshot = _snapshot("partial", "unsafe_partial")
    snapshot.results = _result_rows(status="Running")
    snapshot.driver_race_points = pd.DataFrame(
        [
            {
                "PlayerId": 1,
                "season": 2026,
                "round": 5,
                "race_name": "Safety Grand Prix",
                "fantasy_points": 0.0,
                "is_played": 1,
            }
        ]
    )
    snapshot.source_diagnostics["completed_current_event_keys"] = []

    catalogue, source = app_core.snapshot_race_catalogue(snapshot)

    assert catalogue == ()
    assert source == "unavailable"


def test_current_season_cache_expires_while_historical_cache_remains(tmp_path):
    cache = tmp_path / "results.csv"
    pd.DataFrame([{"season": 2026, "round": 1}]).to_csv(cache, index=False)
    metadata = {
        "fetched_at_utc": "2026-01-01T00:00:00+00:00",
        "season": 2026,
        "source_kind": "grand_prix",
        "status": "available_unverified",
        "event_keys": [[2026, 1]],
    }
    ergast._cache_metadata_file(cache).write_text(json.dumps(metadata), encoding="utf-8")
    now = datetime(2026, 8, 1, tzinfo=UTC)

    assert ergast._try_read_cache(cache, year=2026, now_utc=now) is None
    assert ergast._try_read_cache(cache, year=2025, now_utc=now) is not None


def test_transient_feed_probe_failure_does_not_confirm_an_older_feed(monkeypatch):
    fantasy_api.clear_market_cache()
    monkeypatch.setattr(fantasy_api, "_probe_feed_round", lambda _round: "failed")

    with pytest.raises(RuntimeError, match="latest feed was not changed"):
        fantasy_api._latest_feed_round(max_search=8)

    assert fantasy_api._LATEST_FEED_CACHE["round"] == 0


def test_feed_probe_records_non_json_http_failure_detail(monkeypatch):
    class Response:
        status_code = 403
        headers = {"content-type": "application/xml"}

    fantasy_api.clear_market_cache()
    monkeypatch.setattr(fantasy_api.requests, "get", lambda *_args, **_kwargs: Response())

    assert fantasy_api._probe_feed_round(20) == "failed"
    assert fantasy_api._FEED_PROBE_ERRORS[20] == (
        "HTTP 403 from https://fantasy.formula1.com/feeds/drivers/20_en.json (application/xml)"
    )


def test_deadline_failure_does_not_abort_explicit_refresh(monkeypatch, tmp_path):
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 5,
                "raceName": "Safety Grand Prix",
                "circuitName": "Safety Circuit",
                "date": "2026-08-09",
                "time": "14:00:00Z",
                "qualifying_date": "2026-08-08",
                "qualifying_time": "14:00:00Z",
                "sprint_date": "",
                "sprint_time": "",
            }
        ]
    )
    support = {
        "results": _result_rows(round_no=4).iloc[0:0].copy(),
        "qualifying": _qualifying_rows(round_no=4).iloc[0:0].copy(),
        "sprint": _sprint_rows().iloc[0:0].copy(),
        "schedule": schedule,
    }
    players = pd.DataFrame(
        [
            {
                "playerId": 1,
                "name": "One",
                "price": 10.0,
                "previous_price": 9.9,
                "official_price_change": 0.1,
            },
            {
                "playerId": 2,
                "name": "Two",
                "price": 11.0,
                "previous_price": 10.9,
                "official_price_change": 0.1,
            },
        ]
    )
    monkeypatch.setattr(app_core, "fetch_all_supporting", lambda *_args, **_kwargs: support)
    monkeypatch.setattr(app_core, "_latest_feed_round", lambda: 5)
    monkeypatch.setattr(
        fantasy_api,
        "VERIFIED_MARKET_CACHE_PATH",
        tmp_path / "verified-market.json",
    )
    teams = pd.DataFrame(
        [
            {
                "teamId": 1,
                "name": "Safety Team",
                "price": 20.0,
                "previous_price": 19.7,
                "official_price_change": 0.3,
            }
        ]
    )
    monkeypatch.setattr(
        app_core,
        "fetch_market_asset_ledgers",
        lambda **_kwargs: {
            "players": players,
            "teams": teams,
            "player_assets": players.assign(is_active=1),
            "constructor_assets": teams.assign(is_active=1),
        },
    )
    monkeypatch.setattr(
        app_core,
        "fetch_team_lock_deadline_from_playerstats",
        lambda _player_id: (_ for _ in ()).throw(RuntimeError("deadline unavailable")),
    )

    snapshot = app_core.load_live_data_snapshot(
        current_season=2026,
        historical_seasons_back=0,
        include_playerstats=False,
        force_refresh=True,
        effective_time="2026-08-07T09:00:00Z",
    )

    assert snapshot.team_lock_payload["team_lock_deadline_valid"] is False
    assert "deadline unavailable" in snapshot.source_diagnostics["team_lock_deadline_warning"]
    assert snapshot.source_diagnostics["snapshot_validation_status"] == "valid_with_pending_sessions"
