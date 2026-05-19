import pandas as pd

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


def test_load_model_data_include_playerstats_flag_controls_prefetch(monkeypatch):
    playerstats_calls: list[str] = []

    monkeypatch.setattr(app_core, "fetch_all_supporting", lambda year: _mock_supporting_data(year))
    monkeypatch.setattr(app_core, "fetch_schedule", lambda _year: _mock_supporting_data(2026)["schedule"])
    monkeypatch.setattr(app_core, "_latest_feed_round", lambda: 1)
    monkeypatch.setattr(
        app_core,
        "fetch_players",
        lambda **_kwargs: pd.DataFrame([{"playerId": 1, "name": "Driver One", "team": "Ferrari", "price": 20.0}]),
    )
    monkeypatch.setattr(
        app_core,
        "fetch_teams",
        lambda **_kwargs: pd.DataFrame([{"teamId": 11, "name": "Ferrari", "price": 30.0}]),
    )
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
        return out, pd.DataFrame(), {
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
    assert playerstats_calls == ["driver", "constructor"]


def test_load_model_data_progress_events_include_stage_names_and_elapsed(monkeypatch):
    events: list[dict] = []

    monkeypatch.setattr(app_core, "fetch_all_supporting", lambda year: _mock_supporting_data(year))
    monkeypatch.setattr(app_core, "fetch_schedule", lambda _year: _mock_supporting_data(2026)["schedule"])
    monkeypatch.setattr(app_core, "_latest_feed_round", lambda: 1)
    monkeypatch.setattr(
        app_core,
        "fetch_players",
        lambda **_kwargs: pd.DataFrame([{"playerId": 1, "name": "Driver One", "team": "Ferrari", "price": 20.0}]),
    )
    monkeypatch.setattr(
        app_core,
        "fetch_teams",
        lambda **_kwargs: pd.DataFrame([{"teamId": 11, "name": "Ferrari", "price": 30.0}]),
    )
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
    assert data.diagnostics["model_load_duration_seconds"] >= 0.0
