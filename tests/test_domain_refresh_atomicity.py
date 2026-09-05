from __future__ import annotations

import pandas as pd

from f1fantasy import app_core, fantasy_api
from f1fantasy.weekend_state import EventKey, SessionKind, SessionStatus, weekend_states


def _snapshot(*, price: float, scoring_marker: str, validation: str = "valid") -> app_core.LiveDataSnapshot:
    players = pd.DataFrame(
        [
            {
                "playerId": 116,
                "name": "Liam Lawson",
                "tla": "LAW",
                "team": "Red Bull Racing",
                "price": price,
                "previous_price": 14.0,
                "official_price_change": price - 14.0,
                "is_active": 1,
            }
        ]
    )
    teams = pd.DataFrame(
        [{"teamId": 1, "name": "Red Bull Racing", "price": 30.0, "previous_price": 29.5, "official_price_change": 0.5}]
    )
    results = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 13,
                "driverId": "lawson",
                "driver": "Liam Lawson",
                "marker": scoring_marker,
            }
        ]
    )
    snapshot = app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2026,
        requested_seasons=(2026,),
        loaded_seasons=(2026,),
        season_load_failures={},
        results=results,
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.DataFrame([{"season": 2026, "round": 14, "date": "2026-08-30"}]),
        players=players,
        teams=teams,
        driver_recent_points=pd.DataFrame([{"id": 116, "recent_points_1ago": 12.0 if scoring_marker == "new" else 8.0}]),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame([{"PlayerId": 116, "marker": scoring_marker}]),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={"team_lock_deadline_utc": f"{price}"},
        source_diagnostics={
            "snapshot_validation_status": validation,
            "snapshot_validation_safe_for_scoring": validation == "valid",
            "snapshot_validation_warnings": ["Current result is partial."],
            "weekend_state_validation": {"marker": scoring_marker},
            "market_content_signature": fantasy_api.market_content_signature(players, teams),
            "historical_fantasy_data_version": "test",
        },
        player_assets=players,
        constructor_assets=teams,
        session_results=pd.DataFrame(
            [
                {
                    "season": 2026,
                    "round": 14,
                    "session_kind": "practice_1",
                    "source_driver_id": "lawson",
                    "human_driver_id": "lawson",
                    "position": 1,
                }
            ]
        ),
    )
    snapshot.source_diagnostics["scoring_content_signature"] = app_core.scoring_content_signature(snapshot)
    return snapshot


def test_valid_refreshed_market_is_accepted_when_scoring_snapshot_is_unsafe():
    previous = _snapshot(price=14.5, scoring_marker="old")
    refreshed = _snapshot(price=15.1, scoring_marker="new", validation="unsafe_partial")

    resolved = app_core.resolve_live_data_snapshot(previous, True, lambda _force: refreshed)
    accepted = resolved["snapshot"]

    assert resolved["result_accepted"] is True
    assert resolved["status"] == "market_refreshed_scoring_retained"
    assert accepted.players.loc[0, "price"] == 15.1
    assert accepted.player_assets.loc[0, "price"] == 15.1
    assert accepted.results.loc[0, "marker"] == "old"
    assert accepted.driver_race_points.loc[0, "marker"] == "old"
    assert accepted.driver_recent_points.loc[0, "recent_points_1ago"] == 8.0
    assert accepted.team_lock_payload == refreshed.team_lock_payload
    assert accepted.session_results.loc[0, "position"] == 1
    assert accepted.source_diagnostics["market_scoring_freshness_mismatch"] is True
    assert accepted.source_diagnostics["playerstats_data_status"] == "retained_last_good"


def test_raw_refresh_keeps_market_when_supporting_scoring_source_fails(monkeypatch):
    previous = _snapshot(price=14.5, scoring_marker="old")
    previous.schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 14,
                "raceName": "Current GP",
                "circuitName": "Current Circuit",
                "date": "2026-08-30",
                "time": "14:00:00Z",
                "qualifying_date": "2026-08-29",
                "qualifying_time": "14:00:00Z",
            }
        ]
    )
    fresh_market = _snapshot(price=15.1, scoring_marker="unused")
    monkeypatch.setattr(app_core, "invalidate_live_data_caches", lambda: None)
    monkeypatch.setattr(
        app_core,
        "fetch_all_supporting",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Jolpica failed")),
    )
    monkeypatch.setattr(app_core, "load_canonical_scores", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(
        app_core,
        "resolve_market_data",
        lambda **_kwargs: {
            "live_data_status": "fresh",
            "market_resolution_method": "latest_verified_feed",
            "feed_round": 12,
            "snapshot_round": None,
            "snapshot_name": "Current GP",
            "verified_at_utc": "2026-08-24T12:00:00+00:00",
            "content_signature": fantasy_api.market_content_signature(
                fresh_market.players,
                fresh_market.teams,
                player_assets=fresh_market.player_assets,
                constructor_assets=fresh_market.constructor_assets,
            ),
            "content_changed": True,
            "players": fresh_market.players,
            "teams": fresh_market.teams,
            "player_assets": fresh_market.player_assets,
            "constructor_assets": fresh_market.constructor_assets,
            "asset_ledger_complete": True,
            "refresh_error": None,
            "fallback_failures": [],
            "latest_probe_error": None,
        },
    )
    monkeypatch.setattr(
        app_core,
        "fetch_team_lock_deadline_from_playerstats",
        lambda _player_id: {},
    )
    monkeypatch.setattr(
        app_core,
        "ingest_active_event_sessions",
        lambda *_args, **_kwargs: app_core.LiveSessionIngestion(
            results=app_core.empty_session_results(),
            states=(),
            diagnostics={"sessions": {}, "rows_observed": 0},
        ),
    )

    loaded = app_core.load_live_data_snapshot(
        current_season=2026,
        historical_seasons_back=0,
        include_playerstats=False,
        force_refresh=True,
        effective_time="2026-08-24T12:00:00Z",
        previous_snapshot=previous,
    )

    assert loaded.players.loc[0, "price"] == 15.1
    assert loaded.results.loc[0, "marker"] == "old"
    assert loaded.source_diagnostics["scoring_data_status"] == "retained_last_good"
    assert loaded.source_diagnostics["market_scoring_freshness_mismatch"] is True
    assert "Jolpica failed" in loaded.source_diagnostics["scoring_refresh_error"]


def _classification(round_no: int, count: int, *, race: bool) -> pd.DataFrame:
    rows = []
    for position in range(1, count + 1):
        row = {
            "season": 2026,
            "round": round_no,
            "driverId": f"d{position}",
            "position": position,
        }
        if race:
            row.update({"grid": position, "status": "Finished", "fastestLapRank": 0})
        else:
            row.update({"q1": "1:20", "q2": "1:19", "q3": "1:18"})
        rows.append(row)
    return pd.DataFrame(rows)


def test_historical_completeness_uses_each_events_participant_field():
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 1,
                "raceName": "Historical GP",
                "date": "2026-03-01",
                "time": "14:00:00Z",
                "qualifying_date": "2026-02-28",
                "qualifying_time": "14:00:00Z",
            },
            {
                "season": 2026,
                "round": 14,
                "raceName": "Current GP",
                "date": "2026-08-24",
                "time": "14:00:00Z",
                "qualifying_date": "2026-08-23",
                "qualifying_time": "14:00:00Z",
            },
        ]
    )
    results = pd.concat([_classification(1, 20, race=True), _classification(14, 20, race=True)])
    qualifying = pd.concat([_classification(1, 20, race=False), _classification(14, 22, race=False)])

    states = weekend_states(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=pd.DataFrame(),
        effective_time="2026-08-24T15:00:00Z",
        expected_participant_count=22,
    )
    by_event = {state.event: state for state in states}

    assert by_event[EventKey(2026, 1)].is_final is True
    assert by_event[EventKey(2026, 1)].session(SessionKind.GRAND_PRIX).expected_participant_count == 20
    assert by_event[EventKey(2026, 14)].is_final is False
    assert by_event[EventKey(2026, 14)].session(SessionKind.GRAND_PRIX).status == SessionStatus.PARTIAL
    assert by_event[EventKey(2026, 14)].session(SessionKind.GRAND_PRIX).expected_participant_count == 22
