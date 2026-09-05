import pandas as pd
import pytest

from f1fantasy import app_core, fantasy_api, player_stats


def _snapshot(marker: str) -> app_core.LiveDataSnapshot:
    return app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2021,
        requested_seasons=(2021, 2022, 2023, 2024, 2025, 2026),
        loaded_seasons=(2021, 2022, 2023, 2024, 2025, 2026),
        season_load_failures={},
        results=pd.DataFrame([{"marker": marker}]),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.DataFrame([{"marker": marker}]),
        players=pd.DataFrame([{"playerId": 1, "name": marker}]),
        teams=pd.DataFrame([{"teamId": 11, "name": marker}]),
        driver_recent_points=pd.DataFrame([{"id": 1, "recent_points_1ago": 10.0}]),
        constructor_recent_points=pd.DataFrame([{"id": 11, "recent_points_1ago": 20.0}]),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={"marker": marker},
        source_diagnostics={"raw_live_load_finished_utc": marker},
    )


def _model(marker: str) -> app_core.ModelData:
    return app_core.ModelData(
        drivers=pd.DataFrame([{"marker": marker}]),
        constructors=pd.DataFrame([{"marker": marker}]),
        trends=pd.DataFrame([{"marker": marker}]),
        diagnostics={"marker": marker},
    )


def _snapshot_with_completed_races(marker: str = "raw-v1") -> app_core.LiveDataSnapshot:
    snapshot = _snapshot(marker)
    snapshot.driver_race_points = pd.DataFrame(
        [
            {
                "PlayerId": 1,
                "asset_type": "driver",
                "season": 2026,
                "round": round_no,
                "race_name": f"Round {round_no}",
                "fantasy_points": points,
                "is_played": 1,
            }
            for round_no, points in [(1, 10.0), (3, 30.0), (5, 50.0)]
        ]
    )
    return snapshot


def _supporting_data(year: int) -> dict[str, pd.DataFrame]:
    return {
        "results": pd.DataFrame([{"season": year, "round": 1, "driverId": "d1"}]),
        "qualifying": pd.DataFrame(),
        "sprint": pd.DataFrame(),
        "schedule": pd.DataFrame(
            [
                {
                    "round": 1,
                    "date": "2026-08-01",
                    "raceName": "Test Grand Prix",
                    "circuitName": "Test Circuit",
                }
            ]
        ),
    }


def _market_frames(driver_count: int = 20, constructor_count: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = pd.DataFrame(
        [
            {
                "playerId": index + 1,
                "name": f"Driver {index + 1}",
                "price": 10.0 + index / 10,
                "previous_price": 9.9 + index / 10,
                "official_price_change": 0.1,
                "team": f"Team {(index % constructor_count) + 1}",
            }
            for index in range(driver_count)
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "teamId": 100 + index,
                "name": f"Team {index + 1}",
                "price": 20.0 + index / 10,
                "previous_price": 19.7 + index / 10,
                "official_price_change": 0.3,
            }
            for index in range(constructor_count)
        ]
    )
    return players, teams


def _active_playerstats_payload(
    player_id: int,
    *,
    asset_type: str,
    price: float,
    previous_price: float,
    gameday_id: int = 12,
    race_name: str = "Dutch Grand Prix",
) -> dict:
    return {
        "Value": {
            "PlayerId": player_id,
            "PlayerSkill": 2 if asset_type == "constructor" else 1,
            "GamedayWiseStats": [
                {
                    "GamedayId": gameday_id,
                    "PlayerValue": price,
                    "OldPlayerValue": previous_price,
                    "IsPlayed": 0,
                    "IsActive": 1,
                    "StatsWise": [],
                }
            ],
            "MatchWiseStats": [
                {
                    "GamedayId": gameday_id,
                    "RaceDayWise": [
                        {
                            "MeetingNumber": 14,
                            "MeetingName": race_name,
                            "Season": 2026,
                            "RaceDayId": 1200 + player_id,
                        }
                    ],
                }
            ],
        }
    }


def test_market_resolution_prefers_fresh_and_persists_verified_full_roster(tmp_path):
    cache_path = tmp_path / "verified-market.json"
    players, teams = _market_frames()
    original_players = players.copy(deep=True)
    original_teams = teams.copy(deep=True)

    resolved = fantasy_api.resolve_market_data(
        latest_feed_loader=lambda: 12,
        player_loader=lambda **_kwargs: players,
        team_loader=lambda **_kwargs: teams,
        cache_path=cache_path,
    )

    assert resolved["live_data_status"] == "fresh"
    assert resolved["feed_round"] == 12
    assert cache_path.exists()
    cached = fantasy_api.load_verified_market_cache(path=cache_path)
    assert cached["feed_round"] == 12
    pd.testing.assert_frame_equal(players, original_players)
    pd.testing.assert_frame_equal(teams, original_teams)


def test_market_resolution_uses_verified_cache_when_latest_probe_fails(tmp_path):
    cache_path = tmp_path / "verified-market.json"
    players, teams = _market_frames()
    fantasy_api.save_verified_market_cache(12, players, teams, path=cache_path)

    resolved = fantasy_api.resolve_market_data(
        latest_feed_loader=lambda: (_ for _ in ()).throw(RuntimeError("feed 20 returned 403 XML")),
        player_loader=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("not reached")),
        team_loader=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("not reached")),
        cache_path=cache_path,
    )

    assert resolved["live_data_status"] == "cached"
    assert resolved["feed_round"] == 12
    assert "403 XML" in resolved["refresh_error"]
    assert len(resolved["players"]) == 20


def test_market_resolution_rejects_malformed_cache_and_does_not_use_historical_prices(tmp_path):
    cache_path = tmp_path / "verified-market.json"
    cache_path.write_text('{"cache_version": 1, "feed_round": 12, "players": []}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="No safe current-season market data") as exc_info:
        fantasy_api.resolve_market_data(
            latest_feed_loader=lambda: (_ for _ in ()).throw(RuntimeError("latest discovery failed")),
            cache_path=cache_path,
        )

    assert "latest discovery failed" in str(exc_info.value)
    assert "verified cache" in str(exc_info.value)


def test_market_resolution_does_not_downgrade_newer_verified_cache(tmp_path):
    cache_path = tmp_path / "verified-market.json"
    cached_players, cached_teams = _market_frames()
    fresh_players, fresh_teams = _market_frames()
    fantasy_api.save_verified_market_cache(12, cached_players, cached_teams, path=cache_path)

    resolved = fantasy_api.resolve_market_data(
        latest_feed_loader=lambda: 11,
        player_loader=lambda **_kwargs: fresh_players,
        team_loader=lambda **_kwargs: fresh_teams,
        cache_path=cache_path,
    )

    assert resolved["live_data_status"] == "cached"
    assert resolved["feed_round"] == 12
    assert "older than verified cached feed 12" in resolved["refresh_error"]


def test_market_resolution_reports_unavailable_when_every_source_is_invalid(tmp_path):
    with pytest.raises(RuntimeError, match="No safe current-season market data") as exc_info:
        fantasy_api.resolve_market_data(
            latest_feed_loader=lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
            cache_path=tmp_path / "missing.json",
        )

    message = str(exc_info.value)
    assert "probe failed" in message
    assert "verified cache unavailable" in message


def test_official_market_normalisation_preserves_current_previous_and_change(monkeypatch):
    monkeypatch.setattr(
        fantasy_api,
        "_get_market",
        lambda feed_round=None: [
            {
                "PlayerId": 28,
                "FUllName": "Mercedes",
                "PositionName": "CONSTRUCTOR",
                "IsActive": 1,
                "Value": 32.6,
                "OldPlayerValue": 32.3,
            }
        ],
    )

    teams = fantasy_api.fetch_teams(feed_round=12)

    assert teams.loc[0, "price"] == pytest.approx(32.6)
    assert teams.loc[0, "previous_price"] == pytest.approx(32.3)
    assert teams.loc[0, "official_price_change"] == pytest.approx(0.3)


def test_failed_higher_probe_accepts_validated_current_gameday_market(tmp_path):
    players, teams = _market_frames()

    resolved = fantasy_api.resolve_market_data(
        latest_feed_loader=lambda: (_ for _ in ()).throw(RuntimeError("feed 20 returned 403 XML")),
        current_gameday_loader=lambda: {
            "feed_round": 12,
            "snapshot_name": "Dutch Grand Prix",
            "players": players,
            "teams": teams,
        },
        cache_path=tmp_path / "verified-market.json",
    )

    assert resolved["live_data_status"] == "fresh"
    assert resolved["market_resolution_method"] == "active_gameday_verified"
    assert resolved["feed_round"] == 12
    assert resolved["players"].loc[0, "price"] == pytest.approx(players.loc[0, "price"])
    assert "feed 20 returned 403 XML" in resolved["latest_probe_error"]


def test_active_gameday_market_is_cross_checked_against_playerstats(monkeypatch):
    monkeypatch.setattr(fantasy_api, "MIN_CACHEABLE_DRIVER_ROWS", 1)
    monkeypatch.setattr(fantasy_api, "MIN_CACHEABLE_CONSTRUCTOR_ROWS", 1)
    players = pd.DataFrame(
        [
            {
                "playerId": 18,
                "name": "Pierre Gasly",
                "price": 13.0,
                "previous_price": 12.8,
                "official_price_change": 0.2,
            }
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "teamId": 28,
                "name": "Mercedes",
                "price": 32.6,
                "previous_price": 32.3,
                "official_price_change": 0.3,
            }
        ]
    )
    payloads = {
        18: _active_playerstats_payload(
            18,
            asset_type="driver",
            price=13.0,
            previous_price=12.8,
        ),
        28: _active_playerstats_payload(
            28,
            asset_type="constructor",
            price=32.6,
            previous_price=32.3,
        ),
    }

    validated = fantasy_api.fetch_validated_current_gameday_market(
        [18],
        expected_event_name="Dutch Grand Prix",
        expected_season=2026,
        playerstats_loader=lambda player_id: payloads[player_id],
        player_loader=lambda **_kwargs: players,
        team_loader=lambda **_kwargs: teams,
    )

    assert validated["feed_round"] == 12
    assert validated["validated_asset_count"] == 2
    assert validated["players"].loc[0, "price"] == pytest.approx(13.0)
    assert validated["teams"].loc[0, "official_price_change"] == pytest.approx(0.3)


def test_first_load_then_ui_only_reruns_make_zero_additional_source_calls():
    calls: list[bool] = []

    def loader(force_refresh: bool) -> app_core.LiveDataSnapshot:
        calls.append(force_refresh)
        return _snapshot(f"load-{len(calls)}")

    first = app_core.resolve_live_data_snapshot(None, False, loader)
    snapshot = first["snapshot"]
    assert first["source_load_attempted"] is True
    assert calls == [False]

    ui_only_changes = [
        {"budget": 105.0},
        {"top_k": 3},
        {"objective": "combined", "coefficient": 0.4},
        {"chip": "triple"},
        {"locked": ["1"], "excluded": ["2"]},
        {"current_team": ["1", "2"]},
        {"display_sort": "price"},
    ]
    for _ui_state in ui_only_changes:
        resolved = app_core.resolve_live_data_snapshot(snapshot, False, loader)
        snapshot = resolved["snapshot"]
        assert resolved["status"] == "reused"
        assert resolved["source_load_attempted"] is False

    assert calls == [False]


@pytest.mark.parametrize(
    ("live_status", "expected_status"),
    [("cached", "loaded_cached"), ("generated_snapshot", "loaded_generated_snapshot")],
)
def test_snapshot_resolution_exposes_market_fallback_status(live_status: str, expected_status: str):
    snapshot = _snapshot("fallback")
    snapshot.source_diagnostics["live_data_status"] = live_status

    resolved = app_core.resolve_live_data_snapshot(None, False, lambda _force: snapshot)

    assert resolved["status"] == expected_status
    assert resolved["source_load_succeeded"] is True
    assert resolved["result_accepted"] is True


def test_model_setting_change_rederives_without_loading_sources():
    source_calls: list[bool] = []
    derive_calls: list[str] = []

    def loader(force_refresh: bool) -> app_core.LiveDataSnapshot:
        source_calls.append(force_refresh)
        return _snapshot("raw-v1")

    snapshot = app_core.resolve_live_data_snapshot(None, False, loader)["snapshot"]
    initial_signature = app_core.model_settings_signature(snapshot, 2, 5, 1.0, 0.7, 0.85, "2026-07-31")

    def derive(_snapshot_copy: app_core.LiveDataSnapshot) -> app_core.ModelData:
        derive_calls.append(f"derive-{len(derive_calls) + 1}")
        return _model(derive_calls[-1])

    first = app_core.resolve_derived_model_data(snapshot, None, None, initial_signature, derive)
    assert first["recomputed"] is True
    same = app_core.resolve_derived_model_data(
        snapshot,
        first["data"],
        initial_signature,
        initial_signature,
        derive,
    )
    assert same["recomputed"] is False

    reused_raw = app_core.resolve_live_data_snapshot(snapshot, False, loader)["snapshot"]
    changed_signature = app_core.model_settings_signature(reused_raw, 3, 5, 1.0, 0.7, 0.85, "2026-07-31")
    changed = app_core.resolve_derived_model_data(
        reused_raw,
        same["data"],
        initial_signature,
        changed_signature,
        derive,
    )

    assert changed["recomputed"] is True
    assert derive_calls == ["derive-1", "derive-2"]
    assert source_calls == [False]


def test_selected_races_and_recency_rederive_without_loading_raw_sources():
    snapshot = _snapshot_with_completed_races()
    source_calls: list[bool] = []
    derive_calls = 0

    def loader(force_refresh: bool) -> app_core.LiveDataSnapshot:
        source_calls.append(force_refresh)
        return _snapshot_with_completed_races("unexpected-load")

    def derive(_snapshot_copy: app_core.LiveDataSnapshot) -> app_core.ModelData:
        nonlocal derive_calls
        derive_calls += 1
        return _model(f"derive-{derive_calls}")

    all_signature = app_core.model_settings_signature(
        snapshot, 2, 5, 1.0, 0.7, 0.95, "2026-07-31"
    )
    last_one_signature = app_core.model_settings_signature(
        snapshot,
        2,
        5,
        1.0,
        0.7,
        0.95,
        "2026-07-31",
        selected_race_preset="Last 1",
    )
    excluded_signature = app_core.model_settings_signature(
        snapshot,
        2,
        5,
        1.0,
        0.7,
        0.95,
        "2026-07-31",
        excluded_race_keys=[(2026, 5)],
    )
    changed_p_signature = app_core.model_settings_signature(
        snapshot, 2, 5, 1.0, 0.7, 0.5, "2026-07-31"
    )

    assert len({all_signature, last_one_signature, excluded_signature, changed_p_signature}) == 4
    first = app_core.resolve_derived_model_data(
        snapshot, None, None, all_signature, derive
    )
    reused_snapshot = app_core.resolve_live_data_snapshot(snapshot, False, loader)["snapshot"]
    changed = app_core.resolve_derived_model_data(
        reused_snapshot,
        first["data"],
        all_signature,
        last_one_signature,
        derive,
    )

    assert changed["recomputed"] is True
    assert derive_calls == 2
    assert source_calls == []


def test_model_data_copy_defensively_copies_price_efficiency_payloads():
    model = _model("original")
    model.driver_price_efficiency = pd.DataFrame([{"asset_id": "d1", "price_efficiency": 1.0}])
    model.constructor_price_efficiency = pd.DataFrame([{"asset_id": "c1", "price_efficiency": 2.0}])

    copied = app_core.copy_model_data(model)
    copied.driver_price_efficiency.loc[0, "price_efficiency"] = 99.0
    copied.constructor_price_efficiency.loc[0, "price_efficiency"] = 88.0

    assert model.driver_price_efficiency.loc[0, "price_efficiency"] == 1.0
    assert model.constructor_price_efficiency.loc[0, "price_efficiency"] == 2.0


def test_price_table_and_team_builder_ui_state_do_not_rederive_or_reload():
    snapshot = _snapshot_with_completed_races()
    signature = app_core.model_settings_signature(
        snapshot, 2, 5, 1.0, 0.7, 0.95, "2026-07-31"
    )
    derive_calls = 0
    source_calls = 0

    def derive(_snapshot_copy: app_core.LiveDataSnapshot) -> app_core.ModelData:
        nonlocal derive_calls
        derive_calls += 1
        return _model(f"derive-{derive_calls}")

    def loader(_force_refresh: bool) -> app_core.LiveDataSnapshot:
        nonlocal source_calls
        source_calls += 1
        return _snapshot_with_completed_races("unexpected")

    first = app_core.resolve_derived_model_data(snapshot, None, None, signature, derive)
    _ui_only_state = {
        "efficiency_sort": "Coverage",
        "efficiency_asset_type": "Constructors",
        "efficiency_race_preset": "Custom",
        "efficiency_custom_race_keys": [(2026, 1), (2026, 5)],
        "efficiency_excluded_race_keys": [(2026, 1)],
        "efficiency_team_driver_ids": ["d1"],
        "efficiency_team_budget": 123.0,
        "active_tab": "Price Efficiency",
        "price_efficiency_image_layout": "Reddit landscape",
        "optimise_image_layout": "Portrait",
        "optimise_price_gain_weight_slider": 55,
    }
    reused_snapshot = app_core.resolve_live_data_snapshot(snapshot, False, loader)["snapshot"]
    reused = app_core.resolve_derived_model_data(
        reused_snapshot,
        first["data"],
        signature,
        signature,
        derive,
    )

    assert reused["recomputed"] is False
    assert derive_calls == 1
    assert source_calls == 0


def test_date_change_rederives_without_loading_sources():
    source_calls: list[bool] = []
    derive_calls = 0

    def loader(force_refresh: bool) -> app_core.LiveDataSnapshot:
        source_calls.append(force_refresh)
        return _snapshot("raw-v1")

    def derive(_snapshot_copy: app_core.LiveDataSnapshot) -> app_core.ModelData:
        nonlocal derive_calls
        derive_calls += 1
        return _model(f"derive-{derive_calls}")

    snapshot = app_core.resolve_live_data_snapshot(None, False, loader)["snapshot"]
    first_date = app_core.model_settings_signature(snapshot, 2, 5, 1.0, 0.7, 0.85, "2026-07-31")
    next_date = app_core.model_settings_signature(snapshot, 2, 5, 1.0, 0.7, 0.85, "2026-08-01")
    first = app_core.resolve_derived_model_data(snapshot, None, None, first_date, derive)
    same_date = app_core.resolve_derived_model_data(snapshot, first["data"], first_date, first_date, derive)
    reused_snapshot = app_core.resolve_live_data_snapshot(snapshot, False, loader)["snapshot"]
    next_day = app_core.resolve_derived_model_data(reused_snapshot, same_date["data"], first_date, next_date, derive)

    assert same_date["status"] == "reused"
    assert next_day["status"] == "derived"
    assert derive_calls == 2
    assert source_calls == [False]


def test_failed_derivation_is_suppressed_until_its_signature_changes():
    snapshot = _snapshot("raw-v1")
    successful_model = _model("last-good")
    successful_signature = ("successful",)
    failed_signature = ("requested-failure",)
    calls = 0

    def fail(_snapshot_copy: app_core.LiveDataSnapshot) -> app_core.ModelData:
        nonlocal calls
        calls += 1
        raise RuntimeError("bad model inputs")

    first_failure = app_core.resolve_derived_model_data(
        snapshot,
        successful_model,
        successful_signature,
        failed_signature,
        fail,
    )
    ordinary_rerun = app_core.resolve_derived_model_data(
        snapshot,
        successful_model,
        successful_signature,
        failed_signature,
        fail,
        failed_signature=first_failure["failed_signature"],
        failed_error=first_failure["error"],
    )

    assert first_failure["status"] == "failed"
    assert first_failure["data"].diagnostics["marker"] == "last-good"
    assert ordinary_rerun["status"] == "suppressed_failed_signature"
    assert ordinary_rerun["data"].diagnostics["marker"] == "last-good"
    assert calls == 1

    changed_signature = ("changed-model-setting",)
    successful_retry = app_core.resolve_derived_model_data(
        snapshot,
        successful_model,
        successful_signature,
        changed_signature,
        lambda _snapshot_copy: _model("recovered"),
        failed_signature=first_failure["failed_signature"],
        failed_error=first_failure["error"],
    )
    assert successful_retry["status"] == "derived"
    assert successful_retry["failed_signature"] is None
    assert successful_retry["error"] is None
    assert successful_retry["data"].diagnostics["marker"] == "recovered"


def test_changed_raw_snapshot_retries_a_previously_failed_derivation():
    old_snapshot = _snapshot("raw-v1")
    fresh_snapshot = _snapshot("raw-v2")
    old_requested = app_core.model_settings_signature(old_snapshot, 2, 5, 1.0, 0.7, 0.85, "2026-07-31")
    fresh_requested = app_core.model_settings_signature(fresh_snapshot, 2, 5, 1.0, 0.7, 0.85, "2026-07-31")
    calls = 0

    def derive(_snapshot_copy: app_core.LiveDataSnapshot) -> app_core.ModelData:
        nonlocal calls
        calls += 1
        return _model("fresh")

    retried = app_core.resolve_derived_model_data(
        fresh_snapshot,
        _model("old"),
        ("old-success",),
        fresh_requested,
        derive,
        failed_signature=old_requested,
        failed_error="old failure",
    )

    assert old_requested != fresh_requested
    assert retried["status"] == "derived"
    assert calls == 1


def test_transfer_result_signature_tracks_model_and_raw_versions_only():
    initial_snapshot = _snapshot("raw-v1")
    refreshed_snapshot = _snapshot("raw-v2")
    initial_model = app_core.model_settings_signature(initial_snapshot, 2, 5, 1.0, 0.7, 0.85, "2026-07-31")
    changed_model = app_core.model_settings_signature(initial_snapshot, 3, 5, 1.0, 0.7, 0.85, "2026-07-31")
    refreshed_model = app_core.model_settings_signature(refreshed_snapshot, 2, 5, 1.0, 0.7, 0.85, "2026-07-31")
    transfer_inputs = (("d1", "d2"), ("c1",), 100.0, "balanced")
    initial_local_efficiency_state = ("All", (), (), "Drivers")
    changed_local_efficiency_state = (
        "Custom",
        ((2026, 1), (2026, 5)),
        ((2026, 1),),
        "Constructors",
    )

    initial = app_core.build_transfer_result_signature(
        app_core.model_data_version(initial_snapshot, initial_model),
        transfer_inputs,
    )
    model_changed = app_core.build_transfer_result_signature(
        app_core.model_data_version(initial_snapshot, changed_model),
        transfer_inputs,
    )
    raw_refreshed = app_core.build_transfer_result_signature(
        app_core.model_data_version(refreshed_snapshot, refreshed_model),
        transfer_inputs,
    )
    unrelated_ui_rerun = app_core.build_transfer_result_signature(
        app_core.model_data_version(initial_snapshot, initial_model),
        transfer_inputs,
    )

    assert model_changed != initial
    assert raw_refreshed != initial
    assert changed_local_efficiency_state != initial_local_efficiency_state
    assert unrelated_ui_rerun == initial


def test_transfer_signature_invalidates_for_weighted_race_model_changes():
    snapshot = _snapshot_with_completed_races()
    transfer_inputs = (("d1", "d2"), ("c1",), 100.0, "balanced")
    all_signature = app_core.model_settings_signature(
        snapshot, 2, 5, 1.0, 0.7, 0.95, "2026-07-31"
    )
    selected_signature = app_core.model_settings_signature(
        snapshot,
        2,
        5,
        1.0,
        0.7,
        0.95,
        "2026-07-31",
        selected_race_preset="Custom",
        custom_race_keys=[(2026, 1), (2026, 5)],
    )
    decay_signature = app_core.model_settings_signature(
        snapshot, 2, 5, 1.0, 0.7, 0.5, "2026-07-31"
    )

    initial = app_core.build_transfer_result_signature(
        app_core.model_data_version(snapshot, all_signature), transfer_inputs
    )
    selected = app_core.build_transfer_result_signature(
        app_core.model_data_version(snapshot, selected_signature), transfer_inputs
    )
    decay = app_core.build_transfer_result_signature(
        app_core.model_data_version(snapshot, decay_signature), transfer_inputs
    )

    assert selected != initial
    assert decay != initial


def test_explicit_refresh_loads_once_and_replaces_snapshot():
    calls: list[bool] = []

    def loader(force_refresh: bool) -> app_core.LiveDataSnapshot:
        calls.append(force_refresh)
        return _snapshot("fresh" if force_refresh else "initial")

    initial = app_core.resolve_live_data_snapshot(None, False, loader)["snapshot"]
    refreshed = app_core.resolve_live_data_snapshot(initial, True, loader)

    assert calls == [False, True]
    assert refreshed["source_load_attempted"] is True
    assert refreshed["source_load_succeeded"] is True
    assert refreshed["status"] == "refreshed"
    assert refreshed["snapshot"].results.iloc[0]["marker"] == "fresh"


def test_failed_refresh_retains_last_good_and_does_not_retry_on_ui_rerun():
    calls = 0
    last_good = _snapshot("last-good")

    def failing_loader(_force_refresh: bool) -> app_core.LiveDataSnapshot:
        nonlocal calls
        calls += 1
        raise RuntimeError("source unavailable")

    failed = app_core.resolve_live_data_snapshot(last_good, True, failing_loader)
    assert calls == 1
    assert failed["status"] == "refresh_failed"
    assert failed["source_load_succeeded"] is False
    assert failed["snapshot"].results.iloc[0]["marker"] == "last-good"

    ordinary = app_core.resolve_live_data_snapshot(failed["snapshot"], False, failing_loader)
    assert calls == 1
    assert ordinary["status"] == "reused"
    assert ordinary["source_load_attempted"] is False


def test_refresh_failure_feedback_persists_until_a_successful_source_load():
    identity = app_core.live_data_snapshot_identity(_snapshot("last-good"))
    failed = app_core.refresh_status_transition(
        None,
        None,
        identity,
        refresh_requested=True,
        source_load_attempted=True,
        source_load_succeeded=False,
        result_accepted=False,
        error="source unavailable",
        successful_identity=None,
    )
    ordinary = app_core.refresh_status_transition(
        failed["status"],
        failed["error"],
        failed["successful_identity"],
        refresh_requested=False,
        source_load_attempted=False,
        source_load_succeeded=True,
        result_accepted=True,
        error=None,
        successful_identity=identity,
    )
    refreshed_identity = app_core.live_data_snapshot_identity(_snapshot("fresh"))
    recovered = app_core.refresh_status_transition(
        ordinary["status"],
        ordinary["error"],
        ordinary["successful_identity"],
        refresh_requested=True,
        source_load_attempted=True,
        source_load_succeeded=True,
        result_accepted=True,
        error=None,
        successful_identity=refreshed_identity,
    )

    assert failed["status"] == "failed"
    assert failed["error"] == "source unavailable"
    assert ordinary == failed
    assert recovered["status"] == "succeeded"
    assert recovered["error"] is None
    assert recovered["successful_identity"] == refreshed_identity


def test_snapshot_and_model_reuse_return_defensive_pandas_copies():
    raw = _snapshot("clean")
    reused_raw = app_core.resolve_live_data_snapshot(raw, False, lambda _force: _snapshot("unused"))["snapshot"]
    reused_raw.results.loc[0, "marker"] = "mutated"
    reused_raw.team_lock_payload["marker"] = "mutated"
    assert raw.results.loc[0, "marker"] == "clean"
    assert raw.team_lock_payload["marker"] == "clean"

    current_model = _model("clean")
    signature = ("same",)
    reused_model = app_core.resolve_derived_model_data(
        raw,
        current_model,
        signature,
        signature,
        lambda _snapshot_copy: _model("unused"),
    )["data"]
    reused_model.drivers.loc[0, "marker"] = "mutated"
    reused_model.diagnostics["marker"] = "mutated"
    assert current_model.drivers.loc[0, "marker"] == "clean"
    assert current_model.diagnostics["marker"] == "clean"


def test_market_and_playerstats_cache_clear_interfaces(monkeypatch):
    fantasy_api._LATEST_FEED_CACHE.update({"round": 7, "ts": 123.0})
    fantasy_api._MARKET_CACHE[7] = (123.0, [{"PlayerId": 1}])
    fantasy_api.clear_market_cache()
    assert fantasy_api._LATEST_FEED_CACHE == {"round": 0, "ts": 0.0}
    assert fantasy_api._MARKET_CACHE == {}

    monkeypatch.setattr(player_stats.fetch_player_stats, "_cache", {1: (123.0, {"Data": {}})}, raising=False)
    player_stats.clear_playerstats_cache()
    assert player_stats.fetch_player_stats._cache == {}


def test_raw_loader_calls_all_sources_and_refresh_invalidates_caches(monkeypatch, tmp_path):
    supporting_calls: list[tuple[int, bool]] = []
    market_calls: list[str] = []
    playerstats_calls: list[str] = []
    invalidations: list[str] = []

    def fetch_supporting(year: int, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        supporting_calls.append((year, force_refresh))
        return _supporting_data(year)

    monkeypatch.setattr(app_core, "fetch_all_supporting", fetch_supporting)
    monkeypatch.setattr(
        fantasy_api,
        "VERIFIED_MARKET_CACHE_PATH",
        tmp_path / "verified-market.json",
    )
    monkeypatch.setattr(app_core, "fetch_schedule", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")))
    monkeypatch.setattr(app_core, "_latest_feed_round", lambda: market_calls.append("round") or 3)
    def fetch_market(**_kwargs):
        market_calls.append("market")
        players = pd.DataFrame(
            [{"playerId": 1, "name": "Driver One", "team": "Ferrari", "price": 20.0}]
        )
        teams = pd.DataFrame([{"teamId": 11, "name": "Ferrari", "price": 30.0}])
        return {
            "players": players,
            "teams": teams,
            "player_assets": players.assign(is_active=1),
            "constructor_assets": teams.assign(is_active=1),
        }

    monkeypatch.setattr(app_core, "fetch_market_asset_ledgers", fetch_market)
    monkeypatch.setattr(
        app_core,
        "fetch_team_lock_deadline_from_playerstats",
        lambda _player_id: playerstats_calls.append("deadline") or {"team_lock_deadline_utc": "2026-08-01T12:00:00Z"},
    )

    def add_playerstats(df: pd.DataFrame, asset_type: str, progress_callback=None):
        playerstats_calls.append(asset_type)
        out = df.copy(deep=True)
        out["recent_points_2ago"] = 10.0
        out["recent_points_1ago"] = 12.0
        out["recent_points_available"] = 2
        out["recent_points_source"] = "playerstats"
        if progress_callback is not None:
            progress_callback({"processed": len(out), "total": len(out), "failed": 0})
        diagnostics = {
            "playerstats_assets_loaded": len(out),
            "playerstats_assets_failed": 0,
            "playerstats_timeout_failures": 0,
            "playerstats_skipped_after_failure_limit": 0,
            "playerstats_failures": [],
        }
        return out, pd.DataFrame([{"id": int(out.iloc[0]["id"])}]), diagnostics

    monkeypatch.setattr(app_core, "_add_playerstats_recent_points", add_playerstats)
    monkeypatch.setattr(app_core, "clear_market_cache", lambda: invalidations.append("market"))
    monkeypatch.setattr(app_core, "clear_playerstats_cache", lambda: invalidations.append("playerstats"))

    first = app_core.load_live_data_snapshot(
        current_season=2026,
        historical_seasons_back=1,
        include_playerstats=True,
    )
    assert supporting_calls == [(2025, False), (2026, False)]
    assert market_calls == ["round", "market"]
    assert playerstats_calls == ["deadline", "driver", "constructor"]
    assert invalidations == []
    assert first.requested_seasons == (2025, 2026)
    assert first.loaded_seasons == (2025, 2026)
    assert first.season_load_failures == {}
    assert first.team_lock_payload["team_lock_deadline_utc"] == "2026-08-01T12:00:00Z"
    assert not first.schedule.empty

    app_core.load_live_data_snapshot(
        current_season=2026,
        historical_seasons_back=1,
        include_playerstats=True,
        force_refresh=True,
    )
    assert invalidations == ["market", "playerstats"]
    assert supporting_calls[-2:] == [(2025, True), (2026, True)]
    assert market_calls == ["round", "market", "round", "market"]
    assert playerstats_calls == [
        "deadline",
        "driver",
        "constructor",
        "deadline",
        "driver",
        "constructor",
    ]


def test_partial_initial_history_records_exact_interior_gap(monkeypatch, tmp_path):
    def fetch_supporting(year: int, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        assert force_refresh is False
        if year == 2024:
            raise RuntimeError("historical endpoint unavailable")
        return _supporting_data(year)

    monkeypatch.setattr(app_core, "fetch_all_supporting", fetch_supporting)
    monkeypatch.setattr(
        fantasy_api,
        "VERIFIED_MARKET_CACHE_PATH",
        tmp_path / "verified-market.json",
    )
    monkeypatch.setattr(app_core, "_latest_feed_round", lambda: 3)
    def fetch_market(**_kwargs):
        players = pd.DataFrame(
            [{"playerId": 1, "name": "Driver One", "team": "Ferrari", "price": 20.0}]
        )
        teams = pd.DataFrame([{"teamId": 11, "name": "Ferrari", "price": 30.0}])
        return {
            "players": players,
            "teams": teams,
            "player_assets": players.assign(is_active=1),
            "constructor_assets": teams.assign(is_active=1),
        }

    monkeypatch.setattr(app_core, "fetch_market_asset_ledgers", fetch_market)
    monkeypatch.setattr(app_core, "fetch_team_lock_deadline_from_playerstats", lambda _player_id: {})

    snapshot = app_core.load_live_data_snapshot(
        current_season=2026,
        historical_seasons_back=3,
        include_playerstats=False,
    )
    coverage = app_core.season_coverage(snapshot, historical_seasons_back=3)

    assert snapshot.requested_seasons == (2023, 2024, 2025, 2026)
    assert snapshot.loaded_seasons == (2023, 2025, 2026)
    assert 2024 in snapshot.season_load_failures
    assert coverage["requested_seasons"] == (2023, 2024, 2025, 2026)
    assert coverage["used_seasons"] == (2023, 2025, 2026)
    assert coverage["missing_requested_seasons"] == (2024,)
    assert coverage["historical_seasons_requested"] == 3
    assert coverage["historical_seasons_used"] == 2
    assert coverage["historical_coverage_complete"] is False


def test_history_coverage_clamps_legacy_snapshot_to_production_start():
    snapshot = _snapshot("raw-v1")
    snapshot.loaded_seasons = (2022, 2023, 2024, 2025, 2026)
    snapshot.season_load_failures = {2021: "oldest season unavailable"}

    coverage = app_core.season_coverage(snapshot, historical_seasons_back=5)

    assert coverage["requested_seasons"] == (2023, 2024, 2025, 2026)
    assert coverage["available_seasons"] == (2023, 2024, 2025, 2026)
    assert coverage["used_seasons"] == (2023, 2024, 2025, 2026)
    assert coverage["missing_requested_seasons"] == ()
    assert coverage["historical_seasons_requested"] == 3
    assert coverage["historical_seasons_used"] == 3


def test_raw_refresh_propagates_partial_supporting_source_failure(monkeypatch):
    monkeypatch.setattr(app_core, "clear_market_cache", lambda: None)
    monkeypatch.setattr(app_core, "clear_playerstats_cache", lambda: None)
    monkeypatch.setattr(
        app_core,
        "fetch_all_supporting",
        lambda _year, force_refresh=False: (_ for _ in ()).throw(RuntimeError("Jolpica unavailable")),
    )

    with pytest.raises(RuntimeError, match="Could not refresh supporting race data"):
        app_core.load_live_data_snapshot(
            current_season=2026,
            historical_seasons_back=1,
            force_refresh=True,
        )
