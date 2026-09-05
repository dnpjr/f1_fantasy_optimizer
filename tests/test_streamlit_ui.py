from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from f1fantasy import app_core
from f1fantasy.ui_helpers import team_solution_key


def _asset_rows(asset_type: str, count: int) -> pd.DataFrame:
    rows = []
    for index in range(1, count + 1):
        is_driver = asset_type == "driver"
        rows.append(
            {
                "id": f"{asset_type[0]}{index}",
                "name": f"{'Driver' if is_driver else 'Team'} {index}",
                "abbreviation": f"{'D' if is_driver else 'T'}{index}",
                "team": f"Team {((index - 1) % 3) + 1}",
                "price": 8.0 + index,
                "exp_score": 10.0 + index,
                "next_race_exp_score": 10.0 + index,
                "next_race_expected_points": 10.0 + index,
                "horizon_expected_points": 30.0 + index,
                "nn_exp_score": 10.0 + index,
                "dnf_rate": 0.1,
                "volatility": 5.0,
                "recent_points_2ago": 8.0 + index,
                "recent_points_1ago": 9.0 + index,
                "recent_points_available": 2,
                "recent_points_source": "playerstats",
                "image_url": "",
                "team_colour": "#dc2626" if index % 2 else "#14b8a6",
            }
        )
    return pd.DataFrame(rows)


def _efficiency_table(assets: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    rows = []
    for _, asset in assets.iterrows():
        rows.append(
            {
                "asset_id": str(asset["id"]),
                "asset_type": asset_type,
                "abbreviation": asset["abbreviation"],
                "full_name": asset["name"],
                "team_name": asset["team"] if asset_type == "driver" else asset["name"],
                "team_colour": asset["team_colour"],
                "current_price": asset["price"],
                "selected_points_total": 20.0,
                "average_points_per_race": 10.0,
                "price_efficiency": 10.0 / float(asset["price"]),
                "selected_race_count": 2,
                "valid_race_count": 2,
                "missing_race_count": 0,
                "coverage_fraction": 1.0,
                "has_source_failure": False,
                "status": "complete",
                "valid_race_keys": (app_core.RaceKey(2026, 1), app_core.RaceKey(2026, 2)),
            }
        )
    return pd.DataFrame(rows)


def _snapshot() -> app_core.LiveDataSnapshot:
    race_points = pd.DataFrame(
        [
            {
                "PlayerId": "d1",
                "asset_type": "driver",
                "season": 2026,
                "round": round_no,
                "race_name": f"Race {round_no}",
                "fantasy_points": points,
                "is_played": 1,
            }
            for round_no, points in [(1, 10.0), (2, 20.0)]
        ]
    )
    return app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2024,
        requested_seasons=(2024, 2025, 2026),
        loaded_seasons=(2024, 2025, 2026),
        season_load_failures={},
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.DataFrame(),
        players=pd.DataFrame(),
        teams=pd.DataFrame(),
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=race_points,
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={"raw_live_load_finished_utc": "ui-fixture", "feed_round": 2},
    )


def _model() -> app_core.ModelData:
    drivers = _asset_rows("driver", 6)
    constructors = _asset_rows("constructor", 3)
    for frame in (drivers, constructors):
        frame["shadow_normal_ev"] = frame["next_race_expected_points"] - 1.0
        frame["shadow_sprint_bonus"] = 2.0
        frame["shadow_sprint_ev"] = frame["shadow_normal_ev"] + frame["shadow_sprint_bonus"]
        frame["shadow_valid_race_count"] = 2
        frame["shadow_selected_race_count"] = 2
        frame["baseline_expected_points"] = frame["next_race_expected_points"] - 2.0
        frame["sprint_bonus"] = 2.0
        frame["sprint_adjusted_expected_points"] = frame["next_race_expected_points"]
        frame["baseline_ev"] = frame["next_race_expected_points"]
        frame["live_session_score"] = pd.Series(
            [1.0 - (index / max(1, len(frame) - 1)) for index in range(len(frame))]
        )
        frame["live_session_rank"] = range(1, len(frame) + 1)
        frame["live_only_ev"] = sorted(
            frame["next_race_expected_points"].astype(float), reverse=True
        )
    drivers["FP1_score"] = drivers["live_session_score"]
    drivers["FP2_score"] = pd.NA
    drivers["FP3_score"] = pd.NA
    drivers["SQ_score"] = drivers["live_session_score"]
    constructors["driver_coverage"] = "2/2"
    constructors["constructor_live_session_score"] = constructors["live_session_score"]
    constructors["constructor_live_session_rank"] = constructors["live_session_rank"]
    constructors["constructor_live_only_ev"] = constructors["live_only_ev"]
    diagnostics = {
        "current_season": 2026,
        "feed_round": 2,
        "driver_count": len(drivers),
        "constructor_count": len(constructors),
        "current_season_weight": 1.0,
        "past_season_weight": 0.7,
        "recency_decay": 0.85,
        "upcoming_race_horizon": 5,
        "horizon_weights": [1.0, 0.7, 0.7, 0.7, 0.7],
        "upcoming_circuits": ["Test Circuit"],
        "start_year": 2024,
        "historical_seasons_used": 2,
        "historical_seasons_requested": 2,
        "approved_sprint_ev_shadow": {
            "status": "available",
            "model_version": "sprint_ev_shadow_2026_v1",
            "upcoming_weekend_format": "sprint",
            "production_history_mode": app_core.HISTORY_MODE_ALL_SUPPORTED,
            "sprint_shadow_history": "2026_only",
            "production_isolation": "Shadow columns are not optimiser inputs.",
        },
        "sprint_ev_production": {
            "status": "available",
            "model_version": "sprint_ev_2026_v1",
            "upcoming_weekend_format": "sprint",
            "production_history_mode": app_core.HISTORY_MODE_ALL_SUPPORTED,
            "bonus_applied": True,
        },
        "live_session_shadow": {
            "label": "Live-session shadow forecast — not yet used for prices or optimisation",
            "status": "available",
            "weekend_format": "sprint",
            "production_isolation": "Shadow fields are diagnostics only.",
        },
        "missing_requested_seasons": [],
        "used_seasons": [2024, 2025, 2026],
        "next_race_name": "Test Grand Prix",
        "next_race_date": "2026-08-02",
        "next_race_round": 13,
        "team_lock_deadline_utc": None,
        "team_lock_deadline_source": "unavailable",
    }
    return app_core.ModelData(
        drivers=drivers,
        constructors=constructors,
        trends=pd.DataFrame(),
        diagnostics=diagnostics,
        driver_price_efficiency=_efficiency_table(drivers, "driver"),
        constructor_price_efficiency=_efficiency_table(constructors, "constructor"),
    )


def test_live_session_blend_diagnostics_do_not_fetch_or_run_optimizer(monkeypatch):
    snapshot = _snapshot()
    model = _model()
    effective_date = datetime.now(UTC).date().isoformat()
    signature = app_core.model_settings_signature(
        snapshot, 2, 5, 1.0, 0.7, 0.85, effective_date
    )
    source_calls: list[bool] = []
    optimiser_calls: list[bool] = []
    derive_calls: list[float] = []

    def forbidden_source_call(*_args, **_kwargs):
        source_calls.append(True)
        raise AssertionError("Diagnostics rerun must not load raw live data.")

    def forbidden_optimiser_call(*_args, **_kwargs):
        optimiser_calls.append(True)
        raise AssertionError("Diagnostics rerun must not invoke the optimiser.")

    def controlled_derivation(_snapshot, **kwargs):
        derive_calls.append(float(kwargs["live_session_emphasis"]))
        return model

    monkeypatch.setattr(app_core, "load_live_data_snapshot", forbidden_source_call)
    monkeypatch.setattr(app_core, "run_optimizer", forbidden_optimiser_call)
    monkeypatch.setattr(app_core, "derive_model_data", controlled_derivation)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=20)
    app.session_state["live_data_snapshot"] = snapshot
    app.session_state["derived_model_data"] = model
    app.session_state["derived_model_signature"] = signature
    app.session_state["budget_defaults_initialised"] = True
    app.session_state["optimizer_budget"] = 100.0
    app.session_state["optimizer_budget_source"] = "manual"
    app.session_state["chip_mode_label"] = "3x chip"
    app.session_state["current_team_driver_ids"] = ["d1", "d2", "d3", "d4", "d5"]
    app.session_state["current_team_constructor_ids"] = ["c1", "c2"]
    app.session_state["locked_driver_ids"] = ["d1"]
    app.session_state["excluded_driver_ids"] = ["d6"]
    app.session_state["locked_constructor_ids"] = ["c1"]
    app.session_state["excluded_constructor_ids"] = ["c3"]

    app.run()

    assert not app.exception
    assert source_calls == []
    assert optimiser_calls == []
    assert any(
        "Live session emphasis: 0.00" in caption.value
        for caption in app.caption
    )
    rendered_columns = [set(table.value.columns) for table in app.dataframe]
    assert any({"Asset", "Baseline EV", "FP1", "SQ", "Live score", "Live rank", "Live-only EV"}.issubset(columns) for columns in rendered_columns)
    assert any({"Constructor", "Baseline EV", "Driver coverage", "Live score", "Live rank", "Live-only EV"}.issubset(columns) for columns in rendered_columns)
    assert any(slider.label == "Live session emphasis" for slider in app.slider)
    assert any(caption.value == "No completed sessions" for caption in app.caption)

    before_budget = app.session_state["optimizer_budget"]
    preserved_state = {
        key: app.session_state[key]
        for key in (
            "chip_mode_label",
            "current_team_driver_ids",
            "current_team_constructor_ids",
            "locked_driver_ids",
            "excluded_driver_ids",
            "locked_constructor_ids",
            "excluded_constructor_ids",
        )
    }
    derive_count_before_slider = len(derive_calls)
    desktop_slider = next(
        slider
        for slider in app.slider
        if slider.key == "model_live_session_emphasis"
    )
    desktop_slider.set_value(0.5).run()

    assert not app.exception
    assert source_calls == []
    assert optimiser_calls == []
    assert len(derive_calls) == derive_count_before_slider + 1
    assert derive_calls[-1] == 0.5
    assert app.session_state["optimizer_budget"] == before_budget
    assert app.session_state["model_live_session_emphasis"] == 0.5
    for key, value in preserved_state.items():
        assert app.session_state[key] == value


def test_streamlit_renders_new_controls_and_payload_without_live_calls():
    snapshot = _snapshot()
    effective_date = datetime.now(UTC).date().isoformat()
    signature = app_core.model_settings_signature(
        snapshot,
        historical_seasons_back=2,
        horizon_races=5,
        current_season_weight=1.0,
        past_season_weight=0.7,
        recency_decay=0.85,
        effective_date=effective_date,
    )
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=20)
    app.session_state["live_data_snapshot"] = snapshot
    app.session_state["derived_model_data"] = _model()
    app.session_state["derived_model_signature"] = signature
    app.session_state["budget_defaults_initialised"] = True
    app.session_state["optimizer_budget"] = 100.0
    app.session_state["optimizer_budget_source"] = "manual"

    app.run()

    assert not app.exception
    assert all("Could not load live data" not in error.value for error in app.error)
    tab_labels = [tab.label for tab in app.tabs]
    assert all(area in tab_labels for area in ("Optimise", "Market", "Team", "Settings"))
    assert [tab_labels.index(area) for area in ("Optimise", "Market", "Team", "Settings")] == sorted(
        tab_labels.index(area) for area in ("Optimise", "Market", "Team", "Settings")
    )
    assert "Projection & thresholds" in tab_labels
    assert "Efficiency" in tab_labels
    selectboxes = {widget.key: widget for widget in app.selectbox}
    sliders = {widget.label: widget for widget in app.slider}
    multiselects = {widget.label: widget for widget in app.multiselect}
    number_inputs = {widget.label: widget for widget in app.number_input}
    button_groups = {widget.label: widget for widget in app.get("button_group")}
    download_labels = {widget.label for widget in app.get("download_button")}
    toggles = {widget.label: widget for widget in app.toggle}
    assert all(widget.label != "PNG export format" for widget in app.selectbox)
    assert selectboxes["efficiency_race_preset"].value == "All"
    assert selectboxes["model_race_preset"].value == "All"
    assert selectboxes["price_efficiency_image_layout"].value == "Portrait"
    assert "optimise_image_layout" not in selectboxes
    assert all(widget.label != "Optimisation objective" for widget in app.selectbox)
    assert button_groups["Asset type"].value == "Drivers"
    assert toggles["Current season only"].value is False
    assert toggles["Show Sprint EV breakdown"].value is False
    assert all("Approved Sprint adjustment" not in element.value for element in app.markdown)
    assert sliders["Current-season recency decay p"].value == 0.85
    assert sliders["Price-growth value"].value == 50
    assert sliders["Price-growth value"].min == 0.0
    assert sliders["Price-growth value"].max == 100.0
    assert sliders["Price-growth value"].step == 5.0
    assert isinstance(sliders["Price-growth value"].value, int)
    assert "Exclude races" in multiselects
    assert "Drivers (5)" in multiselects
    assert "Constructors (2)" in multiselects
    assert number_inputs["Team-builder budget"].value == 100.0
    assert "Number of teams" not in number_inputs
    assert "Budget" in number_inputs
    assert any(button.label == "Run optimiser" for button in app.button)
    assert "Download drivers table PNG" in download_labels
    assert "Download constructors table PNG" not in download_labels
    assert "Download Price Efficiency team PNG" not in download_labels
    assert "Download driver projection PNG" in download_labels
    assert "Download constructor projection PNG" not in download_labels
    assert "Download driver targets PNG" not in download_labels
    assert all(expander.label != "Advanced objective settings" for expander in app.expander)
    assert all("Active objective" not in element.value for element in app.markdown)
    assert any(element.value == "Price Efficiency" for element in app.subheader)
    price_change_tables = [
        element.value
        for element in app.markdown
        if "f1-price-change-table" in element.value and "<table" in element.value
    ]
    assert len(price_change_tables) == 1
    assert all(">Asset</th>" in rendered for rendered in price_change_tables)
    assert all(">Abbrev</th>" not in rendered for rendered in price_change_tables)
    assert all(">Name</th>" not in rendered for rendered in price_change_tables)
    assert all(">Team</th>" not in rendered for rendered in price_change_tables)
    assert all('<span class="f1-asset-id"' in rendered for rendered in price_change_tables)
    assert any('title="Driver 1 — Team 1"' in rendered for rendered in price_change_tables)
    captions = "\n".join(element.value for element in app.caption)
    assert "R2: 1.00 · R1: 0.85" in captions
    assert "Objective = expected points + λ × expected price gain." not in captions

    button_groups["Asset type"].set_value("Constructors").run()

    assert not app.exception
    assert app.session_state["efficiency_asset_type"] == "Constructors"
    toggled_download_labels = {widget.label for widget in app.get("download_button")}
    assert "Download constructors table PNG" in toggled_download_labels
    assert "Download drivers table PNG" not in toggled_download_labels

    central_signature = app.session_state["derived_model_signature"]
    local_race_window = {
        widget.key: widget for widget in app.selectbox
    }["efficiency_race_preset"]
    local_race_window.set_value("Custom").run()

    assert not app.exception
    assert app.session_state["derived_model_signature"] == central_signature
    assert any("No races are selected" in warning.value for warning in app.warning)
    empty_download_labels = {widget.label for widget in app.get("download_button")}
    assert "Download drivers table PNG" not in empty_download_labels
    assert "Download constructors table PNG" not in empty_download_labels

    growth_slider = {
        widget.label: widget for widget in app.slider
    }["Price-growth value"]
    growth_slider.set_value(55).run()

    assert not app.exception
    assert app.session_state["optimise_price_gain_weight_slider"] == 55
    assert app.session_state["derived_model_signature"] == central_signature

    chip_control = {widget.key: widget for widget in app.selectbox}["chip_mode_label"]
    chip_control.set_value("Limitless").run()

    assert not app.exception
    limitless_slider = {
        widget.label: widget for widget in app.slider
    }["Price-growth value"]
    assert limitless_slider.disabled
    assert app.session_state["optimizer_objective_mode"] == app_core.OBJECTIVE_COMBINED
    assert all("Limitless uses points only" not in caption.value for caption in app.caption)


def test_current_season_toggle_uses_separate_identity_without_resetting_user_state():
    snapshot = _snapshot()
    effective_date = datetime.now(UTC).date().isoformat()
    signature = app_core.model_settings_signature(
        snapshot,
        historical_seasons_back=2,
        horizon_races=5,
        current_season_weight=1.0,
        past_season_weight=0.7,
        recency_decay=0.85,
        effective_date=effective_date,
        history_mode=app_core.HISTORY_MODE_CURRENT_SEASON_ONLY,
    )
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=20)
    app.session_state["current_season_only"] = True
    app.session_state["live_data_snapshot"] = snapshot
    app.session_state["derived_model_data"] = _model()
    app.session_state["derived_model_signature"] = signature
    app.session_state["budget_defaults_initialised"] = True
    app.session_state["optimizer_budget"] = 117.5
    app.session_state["optimizer_budget_source"] = "manual"
    app.session_state["current_team_budget"] = 117.5
    app.session_state["current_team_budget_user_overridden"] = True
    app.session_state["current_team_driver_ids"] = ["d1", "d2", "d3", "d4", "d5"]
    app.session_state["current_team_constructor_ids"] = ["c1", "c2"]
    app.session_state["uploaded_team_last_success_hash"] = "preserve-me"

    app.run()

    assert not app.exception
    toggle = {widget.label: widget for widget in app.toggle}["Current season only"]
    assert toggle.value is True
    assert app.session_state["derived_model_signature"] == signature
    assert app.session_state["optimizer_budget"] == 117.5
    assert app.session_state["current_team_budget"] == 117.5
    assert app.session_state["current_team_driver_ids"] == ["d1", "d2", "d3", "d4", "d5"]
    assert app.session_state["current_team_constructor_ids"] == ["c1", "c2"]
    assert app.session_state["uploaded_team_last_success_hash"] == "preserve-me"

    toggle.set_value(False).run()

    assert not app.exception
    assert app.session_state["optimizer_budget"] == 117.5
    assert app.session_state["current_team_budget"] == 117.5
    assert app.session_state["current_team_driver_ids"] == ["d1", "d2", "d3", "d4", "d5"]
    assert app.session_state["current_team_constructor_ids"] == ["c1", "c2"]
    assert app.session_state["uploaded_team_last_success_hash"] == "preserve-me"


def test_sprint_breakdown_toggle_only_changes_diagnostic_visibility_and_preserves_user_state():
    snapshot = _snapshot()
    effective_date = datetime.now(UTC).date().isoformat()
    signature = app_core.model_settings_signature(
        snapshot,
        historical_seasons_back=2,
        horizon_races=5,
        current_season_weight=1.0,
        past_season_weight=0.7,
        recency_decay=0.85,
        effective_date=effective_date,
        selected_race_preset="Last 1",
        history_mode=app_core.HISTORY_MODE_CURRENT_SEASON_ONLY,
    )
    model = _model()
    model.diagnostics["production_history_mode"] = app_core.HISTORY_MODE_CURRENT_SEASON_ONLY
    model.diagnostics["approved_sprint_ev_shadow"]["production_history_mode"] = (
        app_core.HISTORY_MODE_CURRENT_SEASON_ONLY
    )
    model.diagnostics["sprint_ev_production"]["production_history_mode"] = (
        app_core.HISTORY_MODE_CURRENT_SEASON_ONLY
    )
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=20)
    app.session_state["current_season_only"] = True
    app.session_state["model_race_preset"] = "Last 1"
    app.session_state["model_recency_decay"] = 0.85
    app.session_state["live_data_snapshot"] = snapshot
    app.session_state["derived_model_data"] = model
    app.session_state["derived_model_signature"] = signature
    app.session_state["budget_defaults_initialised"] = True
    app.session_state["optimizer_budget"] = 117.5
    app.session_state["optimizer_budget_source"] = "manual"
    app.session_state["current_team_budget"] = 116.0
    app.session_state["current_team_budget_user_overridden"] = True
    app.session_state["current_team_driver_ids"] = ["d1", "d2", "d3", "d4", "d5"]
    app.session_state["current_team_constructor_ids"] = ["c1", "c2"]
    app.session_state["optimizer_objective_mode"] = app_core.OBJECTIVE_COMBINED
    app.session_state["optimizer_price_growth_value"] = 65
    app.run()

    before = {
        key: app.session_state[key]
        for key in (
            "optimizer_budget",
            "current_team_budget",
            "current_team_driver_ids",
            "current_team_constructor_ids",
            "current_season_only",
            "model_race_preset",
            "model_recency_decay",
            "optimizer_objective_mode",
            "optimizer_price_growth_value",
            "derived_model_signature",
        )
    }
    breakdown_toggle = {
        widget.label: widget for widget in app.toggle
    }["Show Sprint EV breakdown"]
    breakdown_toggle.set_value(True).run()

    assert not app.exception
    assert any("Approved Sprint adjustment" in element.value for element in app.markdown)
    assert any("Base EV" in table.value.columns for table in app.dataframe)
    for key, value in before.items():
        assert app.session_state[key] == value

    {widget.label: widget for widget in app.toggle}["Show Sprint EV breakdown"].set_value(False).run()
    assert all("Approved Sprint adjustment" not in element.value for element in app.markdown)
    for key, value in before.items():
        assert app.session_state[key] == value


def test_normal_weekend_does_not_show_sprint_adjustment_controls():
    snapshot = _snapshot()
    effective_date = datetime.now(UTC).date().isoformat()
    signature = app_core.model_settings_signature(
        snapshot, 2, 5, 1.0, 0.7, 0.85, effective_date
    )
    model = _model()
    model.diagnostics["sprint_ev_production"]["upcoming_weekend_format"] = "normal"
    model.diagnostics["sprint_ev_production"]["bonus_applied"] = False
    for frame in (model.drivers, model.constructors):
        frame["sprint_bonus"] = 0.0
        frame["baseline_expected_points"] = frame["next_race_expected_points"]
        frame["sprint_adjusted_expected_points"] = frame["next_race_expected_points"]
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=20)
    app.session_state["live_data_snapshot"] = snapshot
    app.session_state["derived_model_data"] = model
    app.session_state["derived_model_signature"] = signature
    app.session_state["budget_defaults_initialised"] = True
    app.session_state["optimizer_budget"] = 100.0
    app.session_state["optimizer_budget_source"] = "manual"

    app.run()

    assert not app.exception
    assert all(widget.label != "Show Sprint EV breakdown" for widget in app.toggle)
    assert all("Includes Sprint adjustment" not in element.value for element in app.caption)


def test_streamlit_optimiser_persists_ten_uniform_results_and_appends_next_batch():
    snapshot = _snapshot()
    effective_date = datetime.now(UTC).date().isoformat()
    signature = app_core.model_settings_signature(
        snapshot,
        historical_seasons_back=2,
        horizon_races=5,
        current_season_weight=1.0,
        past_season_weight=0.7,
        recency_decay=0.85,
        effective_date=effective_date,
    )
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30)
    app.session_state["live_data_snapshot"] = snapshot
    app.session_state["derived_model_data"] = _model()
    app.session_state["derived_model_signature"] = signature
    app.session_state["budget_defaults_initialised"] = True
    app.session_state["optimizer_budget"] = 100.0
    app.session_state["optimizer_budget_source"] = "manual"
    app.session_state["current_team_bank"] = 4.2
    app.session_state["current_team_free_transfers"] = 3
    app.run()

    {button.label: button for button in app.button}["Run optimiser"].click().run()

    assert not app.exception
    first_ten = list(app.session_state["optimiser_solutions"])
    assert len(first_ten) == 10
    rendered_teams = [
        element.value
        for element in app.markdown
        if 'class="f1-ranked-team"' in element.value
    ]
    assert len(rendered_teams) == 10
    assert all("f1-team-summary" in rendered for rendered in rendered_teams)
    assert all(rendered.count('class="f1-driver-card"') == 7 for rendered in rendered_teams)
    assert all("Best Team" not in rendered for rendered in rendered_teams)
    assert {metric for metric in ("Value", "Left", "Gain", "Pts")} <= set(
        word
        for rendered in rendered_teams[:1]
        for word in ("Value", "Left", "Gain", "Pts")
        if f">{word}<" in rendered
    )
    selectboxes = {widget.key: widget for widget in app.selectbox}
    assert selectboxes["optimise_image_layout"].value == "Portrait"
    assert any(expander.label == "Export a ranked team" for expander in app.expander)

    first_team = first_ten[0]
    expected_driver_ids = first_team.drivers["id"].astype(str).tolist()
    expected_constructor_ids = first_team.constructors["id"].astype(str).tolist()
    budget_before = app.session_state["optimizer_budget"]
    solution_keys_before_copy = [
        team_solution_key(solution) for solution in app.session_state["optimiser_solutions"]
    ]
    {button.label: button for button in app.button}["Set as current team"].click().run()

    assert app.session_state["current_team_driver_ids"] == expected_driver_ids
    assert app.session_state["current_team_constructor_ids"] == expected_constructor_ids
    assert app.session_state["current_team_bank"] == 4.2
    assert app.session_state["current_team_free_transfers"] == 3
    assert app.session_state["optimizer_budget"] == budget_before
    assert [
        team_solution_key(solution) for solution in app.session_state["optimiser_solutions"]
    ] == solution_keys_before_copy
    assert any("Team 1 copied to Current Team" in success.value for success in app.success)

    {widget.key: widget for widget in app.selectbox}["optimise_image_layout"].set_value(
        "Reddit landscape"
    ).run()
    assert [
        team_solution_key(solution) for solution in app.session_state["optimiser_solutions"]
    ] == solution_keys_before_copy

    lock_control = {
        widget.key: widget for widget in app.checkbox
    }["optimiser_universe_driver_d1_lock"]
    lock_control.set_value(True).run()
    assert app.session_state["locked_driver_ids"] == ["d1"]
    assert "d1" not in app.session_state["excluded_driver_ids"]
    assert any("Inputs changed" in warning.value for warning in app.warning)
    assert {widget.label: widget for widget in app.multiselect}["Locked drivers"].value == ["d1"]

    exclude_control = {
        widget.key: widget for widget in app.checkbox
    }["optimiser_universe_driver_d1_exclude"]
    exclude_control.set_value(True).run()
    assert "d1" not in app.session_state["locked_driver_ids"]
    assert app.session_state["excluded_driver_ids"] == ["d1"]
    assert {widget.label: widget for widget in app.multiselect}["Excluded drivers"].value == ["d1"]

    {widget.label: widget for widget in app.multiselect}["Locked drivers"].set_value(["d1"]).run()
    assert app.session_state["locked_driver_ids"] == ["d1"]
    assert "d1" not in app.session_state["excluded_driver_ids"]

    market_view = {
        widget.key: widget for widget in app.get("button_group")
    }["market_price_view"]
    market_view.set_value("Thresholds").run()

    assert [team_solution_key(solution) for solution in app.session_state["optimiser_solutions"]] == [
        team_solution_key(solution) for solution in first_ten
    ]

    {button.label: button for button in app.button}["Run optimiser"].click().run()
    batched_base = list(app.session_state["optimiser_solutions"])
    {button.label: button for button in app.button}["Load teams 11–20"].click().run()

    accumulated = list(app.session_state["optimiser_solutions"])
    accumulated_keys = [team_solution_key(solution) for solution in accumulated]
    assert len(accumulated) > 10
    assert accumulated_keys[:10] == [team_solution_key(solution) for solution in batched_base]
    assert len(accumulated_keys) == len(set(accumulated_keys))

    {widget.key: widget for widget in app.slider}["optimise_price_gain_weight_slider"].set_value(55).run()
    assert any("Inputs changed" in warning.value for warning in app.warning)


def test_mobile_optimise_subviews_preserve_session_and_do_not_refresh_live_data():
    snapshot = _snapshot()
    effective_date = datetime.now(UTC).date().isoformat()
    signature = app_core.model_settings_signature(
        snapshot,
        historical_seasons_back=2,
        horizon_races=5,
        current_season_weight=1.0,
        past_season_weight=0.7,
        recency_decay=0.85,
        effective_date=effective_date,
    )
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30)
    app.session_state["live_data_snapshot"] = snapshot
    app.session_state["derived_model_data"] = _model()
    app.session_state["derived_model_signature"] = signature
    app.session_state["budget_defaults_initialised"] = True
    app.session_state["optimizer_budget"] = 117.5
    app.session_state["optimizer_budget_source"] = "manual"
    app.session_state["current_team_driver_ids"] = ["d1", "d2", "d3", "d4", "d5"]
    app.session_state["current_team_constructor_ids"] = ["c1", "c2"]
    app.session_state["locked_driver_ids"] = ["d1"]
    app.session_state["excluded_driver_ids"] = ["d2"]
    app.session_state["locked_constructor_ids"] = ["c1"]
    app.run()
    {button.label: button for button in app.button}["Run optimiser"].click().run()

    assert not app.exception
    navigation = {
        widget.label: widget for widget in app.get("button_group")
    }["Optimise view"]
    assert navigation.value == "Teams"
    assert "optimiser_universe_driver_d1_lock" in {
        widget.key for widget in app.checkbox
    }
    assert "optimiser_universe_constructor_c1_lock" in {
        widget.key for widget in app.checkbox
    }
    assert "optimise_mobile_model_race_preset" in {
        widget.key for widget in app.selectbox
    }

    preserved_keys = (
        "current_team_driver_ids",
        "current_team_constructor_ids",
        "optimizer_budget",
        "locked_driver_ids",
        "excluded_driver_ids",
        "locked_constructor_ids",
        "model_race_preset",
        "model_recency_decay",
        "current_season_only",
        "optimiser_result_signature",
    )
    before = {key: app.session_state[key] for key in preserved_keys}
    solution_keys = [
        team_solution_key(solution)
        for solution in app.session_state["optimiser_solutions"]
    ]
    refresh_marker = app.session_state["live_data_snapshot"].source_diagnostics[
        "raw_live_load_finished_utc"
    ]

    for view in ("Drivers", "Constructors", "Controls", "Teams"):
        {
            widget.label: widget for widget in app.get("button_group")
        }["Optimise view"].set_value(view).run()
        assert not app.exception
        assert app.session_state["optimise_mobile_subview"] == view
        for key, value in before.items():
            assert app.session_state[key] == value
        assert [
            team_solution_key(solution)
            for solution in app.session_state["optimiser_solutions"]
        ] == solution_keys
        assert app.session_state["live_data_snapshot"].source_diagnostics[
            "raw_live_load_finished_utc"
        ] == refresh_marker
