import pandas as pd
import inspect

from f1fantasy import app_core


def _mock_supporting_data(year: int) -> dict[str, pd.DataFrame]:
    results = pd.DataFrame(
        [
            {
                "season": int(year),
                "round": 1,
                "circuitName": "Test Circuit",
                "driverId": "d1",
                "driver": "Driver One",
                "constructorId": "c1",
                "constructor": "Ferrari",
                "quali_points": 0.0,
                "sprint_points": 0.0,
                "race_points": 10.0,
                "weekend_points": 10.0,
                "q2_reached": 0,
                "q3_reached": 0,
                "is_dsq": 0,
                "is_dnf": 0,
                "sprint_is_dnf": 0,
            }
        ]
    )
    empty = pd.DataFrame()
    schedule = pd.DataFrame(
        [
            {
                "round": 1,
                "date": "2026-06-01",
                "raceName": "Test Grand Prix",
                "circuitName": "Test Circuit",
                "qualifying_date": "2026-05-31",
                "qualifying_time": "12:00:00Z",
                "sprint_date": "",
                "sprint_time": "",
            }
        ]
    )
    return {
        "results": results,
        "qualifying": empty,
        "sprint": empty,
        "schedule": schedule,
    }


def _mock_expected_scores(*_args, **_kwargs):
    drivers = pd.DataFrame(
        [
            {
                "driverId": "d1",
                "driver": "Driver One",
                "constructorId": "c1",
                "constructor": "Ferrari",
                "exp_score": 12.0,
                "dnf_rate": 0.1,
                "volatility": 6.0,
            }
        ]
    )
    constructors = pd.DataFrame(
        [
            {
                "constructorId": "c1",
                "constructor": "Ferrari",
                "exp_score": 24.0,
                "dnf_rate": 0.08,
                "volatility": 9.0,
            }
        ]
    )
    return drivers, constructors


def _mock_market_asset_ledgers(**_kwargs):
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


def _mock_market_resolution(**kwargs):
    feed_round = int(kwargs["latest_feed_loader"]())
    market = kwargs["market_loader"](feed_round=feed_round)
    return {
        "live_data_status": "fresh",
        "market_resolution_method": "test_fixture",
        "feed_round": feed_round,
        "snapshot_round": None,
        "snapshot_name": None,
        "verified_at_utc": None,
        "players": market["players"],
        "teams": market["teams"],
        "player_assets": market["player_assets"],
        "constructor_assets": market["constructor_assets"],
        "asset_ledger_complete": True,
        "refresh_error": None,
        "fallback_failures": [],
        "latest_probe_error": None,
    }


def test_load_model_data_include_playerstats_flag_controls_prefetch(monkeypatch):
    playerstats_calls: list[str] = []

    # This test isolates the prefetch flag; canonical-history integration has
    # dedicated coverage tests and would otherwise replace the one-row fixture.
    monkeypatch.setattr(app_core, "load_canonical_scores", lambda _path: pd.DataFrame())
    monkeypatch.setattr(app_core, "fetch_all_supporting", lambda year: _mock_supporting_data(year))
    monkeypatch.setattr(app_core, "fetch_schedule", lambda _year: _mock_supporting_data(2026)["schedule"])
    monkeypatch.setattr(app_core, "_latest_feed_round", lambda: 1)
    monkeypatch.setattr(app_core, "fetch_market_asset_ledgers", _mock_market_asset_ledgers)

    # This test targets playerstats prefetching, so keep the synthetic market
    # independent of any newer verified cache present in the developer checkout.
    monkeypatch.setattr(app_core, "resolve_market_data", _mock_market_resolution)
    monkeypatch.setattr(app_core, "compute_weekend_points", lambda **kwargs: kwargs["results"])
    monkeypatch.setattr(app_core, "expected_scores_horizon", _mock_expected_scores)
    def _mock_no_negative(*_args, **_kwargs):
        series = pd.Series({"d1": 12.0})
        series.index.name = "driverId"
        return series

    monkeypatch.setattr(app_core, "apply_no_negative_expectation", _mock_no_negative)
    monkeypatch.setattr(app_core, "fetch_team_lock_deadline_from_playerstats", lambda _player_id: {})
    monkeypatch.setattr(app_core, "build_trends_data", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(
        app_core,
        "playerstats_recent_points_diagnostics",
        lambda *_args, **_kwargs: {
            "recent_points_source": "mock",
            "recent_points_endpoint_pattern": "mock",
            "recent_points_driver_complete": 0,
            "recent_points_constructor_complete": 0,
            "recent_points_driver_manual": 0,
            "recent_points_constructor_manual": 0,
            "recent_points_driver_total": 1,
            "recent_points_constructor_total": 1,
            "recent_points_rounds": [],
            "recent_points_circuits": [],
            "recent_points_fallback_used": False,
            "playerstats_assets_loaded": 0,
            "playerstats_assets_failed": 0,
            "playerstats_driver_failures": [],
            "playerstats_constructor_failures": [],
        },
    )

    def _mock_add_playerstats(df: pd.DataFrame, asset_type: str, progress_callback=None):
        playerstats_calls.append(asset_type)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "playerstats",
                    "asset_type": asset_type,
                    "processed": 1,
                    "total": 1,
                    "loaded": 1,
                    "failed": 0,
                    "skipped": 0,
                    "timeouts": 0,
                }
            )
        out = df.copy()
        out["recent_points_2ago"] = 11.0
        out["recent_points_1ago"] = 13.0
        out["recent_points_available"] = 2
        out["recent_points_source"] = "playerstats"
        out["recent_points_fallback_used"] = False
        out["recent_points_missing"] = False
        race_points = pd.DataFrame(
            [
                {
                    "PlayerId": asset_id,
                    "asset_type": asset_type,
                    "season": 2026,
                    "round": 1,
                    "race_name": "Test Grand Prix",
                    "fantasy_points": 12.0 if asset_type == "driver" else 24.0,
                    "is_played": 1,
                }
                for asset_id in out["id"]
            ]
        )
        return out, race_points, {
            "playerstats_assets_loaded": len(out),
            "playerstats_assets_failed": 0,
            "playerstats_timeout_failures": 0,
            "playerstats_skipped_after_failure_limit": 0,
            "playerstats_failures": [],
        }

    monkeypatch.setattr(app_core, "_add_playerstats_recent_points", _mock_add_playerstats)

    fast = app_core.load_model_data(current_season=2026, today="2026-01-01", include_playerstats=False)
    assert fast.diagnostics["playerstats_prefetch_enabled"] is False
    assert playerstats_calls == []

    detailed = app_core.load_model_data(current_season=2026, today="2026-01-01", include_playerstats=True)
    assert detailed.diagnostics["playerstats_prefetch_enabled"] is True
    assert detailed.diagnostics["playerstats_load_duration_seconds"] >= 0.0
    assert detailed.diagnostics["recent_points_source"] == "mock"
    assert detailed.diagnostics["requested_seasons"] == [2023, 2024, 2025, 2026]
    assert detailed.diagnostics["used_seasons"] == [2023, 2024, 2025, 2026]
    assert detailed.diagnostics["missing_requested_seasons"] == []
    assert detailed.diagnostics["historical_coverage_complete"] is True
    assert detailed.diagnostics["current_season_race_catalogue"] == [
        {"season": 2026, "round": 1, "race_name": "Test Grand Prix"}
    ]
    assert detailed.diagnostics["current_season_completed_race_count"] == 1
    assert detailed.diagnostics["current_season_race_catalogue_has_source_failures"] is False
    assert detailed.diagnostics["selected_race_preset"] == "All"
    assert detailed.diagnostics["selected_race_keys"] == [(2026, 1)]
    assert detailed.diagnostics["excluded_race_keys"] == []
    assert detailed.diagnostics["selected_race_weights"] == {"2026:1": 1.0}
    assert detailed.diagnostics["blend_application_count"] == 1
    assert detailed.diagnostics["horizon_weight_sum"] == 1.0
    assert detailed.drivers.loc[0, "current_component_source"] == "official_current"
    assert detailed.drivers.loc[0, "next_race_expected_points"] == 12.0
    assert detailed.drivers.loc[0, "horizon_expected_points"] == 12.0
    driver_efficiency = detailed.driver_price_efficiency.set_index("full_name").loc["Driver One"]
    assert driver_efficiency["average_points_per_race"] == 12.0
    assert driver_efficiency["current_price"] == 20.0
    assert driver_efficiency["price_efficiency"] == 12.0 / 20.0
    constructor_efficiency = detailed.constructor_price_efficiency.set_index("full_name").loc["Ferrari"]
    assert constructor_efficiency["average_points_per_race"] == 24.0
    assert constructor_efficiency["current_price"] == 30.0
    assert constructor_efficiency["price_efficiency"] == 24.0 / 30.0
    assert playerstats_calls == ["driver", "constructor"]

    def _partial_supporting(year: int):
        if year == 2025:
            raise RuntimeError("historical season unavailable")
        return _mock_supporting_data(year)

    monkeypatch.setattr(app_core, "fetch_all_supporting", _partial_supporting)
    partial = app_core.load_model_data(current_season=2026, today="2026-01-01", include_playerstats=False)
    assert partial.diagnostics["requested_seasons"] == [2023, 2024, 2025, 2026]
    assert partial.diagnostics["used_seasons"] == [2023, 2024, 2026]
    assert partial.diagnostics["missing_requested_seasons"] == [2025]
    assert partial.diagnostics["historical_seasons_requested"] == 3
    assert partial.diagnostics["historical_seasons_used"] == 2
    assert partial.diagnostics["historical_coverage_complete"] is False


def test_load_model_data_progress_events_include_stage_names_and_elapsed(monkeypatch):
    events: list[dict] = []

    monkeypatch.setattr(app_core, "fetch_all_supporting", lambda year: _mock_supporting_data(year))
    monkeypatch.setattr(app_core, "fetch_schedule", lambda _year: _mock_supporting_data(2026)["schedule"])
    monkeypatch.setattr(app_core, "_latest_feed_round", lambda: 1)
    monkeypatch.setattr(app_core, "fetch_market_asset_ledgers", _mock_market_asset_ledgers)
    monkeypatch.setattr(app_core, "resolve_market_data", _mock_market_resolution)
    monkeypatch.setattr(app_core, "compute_weekend_points", lambda **kwargs: kwargs["results"])
    monkeypatch.setattr(app_core, "expected_scores_horizon", _mock_expected_scores)

    def _mock_no_negative(*_args, **_kwargs):
        series = pd.Series({"d1": 12.0})
        series.index.name = "driverId"
        return series

    monkeypatch.setattr(app_core, "apply_no_negative_expectation", _mock_no_negative)
    monkeypatch.setattr(app_core, "fetch_team_lock_deadline_from_playerstats", lambda _player_id: {})
    monkeypatch.setattr(app_core, "build_trends_data", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(
        app_core,
        "playerstats_recent_points_diagnostics",
        lambda *_args, **_kwargs: {
            "recent_points_source": "mock",
            "recent_points_endpoint_pattern": "mock",
            "recent_points_driver_complete": 0,
            "recent_points_constructor_complete": 0,
            "recent_points_driver_manual": 0,
            "recent_points_constructor_manual": 0,
            "recent_points_driver_total": 1,
            "recent_points_constructor_total": 1,
            "recent_points_rounds": [],
            "recent_points_circuits": [],
            "recent_points_fallback_used": False,
            "playerstats_assets_loaded": 0,
            "playerstats_assets_failed": 0,
            "playerstats_driver_failures": [],
            "playerstats_constructor_failures": [],
        },
    )

    def _mock_add_playerstats(df: pd.DataFrame, asset_type: str, progress_callback=None):
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "playerstats",
                    "asset_type": asset_type,
                    "processed": 1,
                    "total": 1,
                    "loaded": 1,
                    "failed": 0,
                    "skipped": 0,
                    "timeouts": 0,
                }
            )
        out = df.copy()
        out["recent_points_2ago"] = 11.0
        out["recent_points_1ago"] = 13.0
        out["recent_points_available"] = 2
        out["recent_points_source"] = "playerstats"
        out["recent_points_fallback_used"] = False
        out["recent_points_missing"] = False
        return out, pd.DataFrame(), {
            "playerstats_assets_loaded": len(out),
            "playerstats_assets_failed": 0,
            "playerstats_timeout_failures": 0,
            "playerstats_skipped_after_failure_limit": 0,
            "playerstats_failures": [],
        }

    monkeypatch.setattr(app_core, "_add_playerstats_recent_points", _mock_add_playerstats)

    data = app_core.load_model_data(
        current_season=2026,
        today="2026-01-01",
        include_playerstats=True,
        progress_callback=lambda payload: events.append(payload),
    )

    assert events
    stage_names = {event.get("stage_name") for event in events}
    assert "Loading market feed" in stage_names
    assert "Loading current prices" in stage_names
    assert "Loading supporting race/schedule data" in stage_names
    assert "Loading playerstats" in stage_names
    assert "Building model inputs" in stage_names
    assert "Computing expected points" in stage_names
    assert "Computing price-change probabilities" in stage_names
    assert "Ready" in stage_names
    playerstats_events = [event for event in events if event.get("stage_name") == "Loading playerstats"]
    assert len(playerstats_events) >= 2
    assert events[0].get("stage_name") == "Loading market feed"
    assert events[-1].get("stage_name") == "Ready"
    assert data.diagnostics["model_load_duration_seconds"] >= 0.0


def test_load_model_data_signature_supports_progress_callback():
    signature = inspect.signature(app_core.load_model_data)
    assert "progress_callback" in signature.parameters
    parameter = signature.parameters["progress_callback"]
    assert parameter.default is None
    assert signature.parameters["selected_race_preset"].default == "All"
    assert signature.parameters["custom_race_keys"].default is None
    assert signature.parameters["excluded_race_keys"].default is None
