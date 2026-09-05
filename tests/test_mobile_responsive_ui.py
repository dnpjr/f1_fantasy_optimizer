from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _source() -> str:
    return APP_SOURCE.read_text(encoding="utf-8")


def test_contextual_navigation_is_optimise_only_and_desktop_hidden():
    source = _source()

    assert 'key="optimise_mobile_subview"' in source
    assert ".st-key-optimise_mobile_subview" in source
    assert ".st-key-optimise_mobile_subview," in source
    assert ".st-key-primary_navigation > [data-baseweb=\"tab-list\"]" not in source
    assert "bottom: calc(8px + env(safe-area-inset-bottom));" in source
    assert "padding-bottom: calc(68px + env(safe-area-inset-bottom) + 16px);" in source


def test_mobile_subviews_hide_unrelated_optimise_sections_without_unmounting_them():
    source = _source()

    for view in ("teams", "drivers", "constructors", "controls"):
        assert f"f1-optimise-view-{view}" in source
    for container in (
        "optimiser_teams_view",
        "optimiser_drivers_view",
        "optimiser_constructors_view",
        "optimiser_controls_view",
    ):
        assert f'key="{container}"' in source
        assert f".st-key-{container}" in source
    assert "body:has(.f1-optimise-view-teams) .st-key-optimiser_teams_view" in source
    assert "body:has(.f1-optimise-view-drivers) .st-key-optimiser_drivers_view" in source
    assert "body:has(.f1-optimise-view-constructors) .st-key-optimiser_constructors_view" in source
    assert "body:has(.f1-optimise-view-controls) .st-key-optimiser_controls_view" in source


def test_mobile_team_cards_use_three_plus_two_drivers_and_two_wide_constructors():
    source = _source()

    assert ".f1-ranked-team .f1-driver-grid .f1-driver-card:nth-child(-n+3)" in source
    assert ".f1-ranked-team .f1-driver-grid .f1-driver-card:nth-child(n+4)" in source
    assert "grid-column: span 2;" in source
    assert "grid-column: span 3;" in source
    assert ".f1-ranked-team .f1-constructor-grid" in source
    assert "width: 100%;" in source


def test_mobile_threshold_schema_and_driver_export_label_are_correct():
    source = _source()

    assert "<th>Asset</th><th>Price</th><th>Good</th><th>Great</th>" in source
    assert 'f"Download {active_asset_label} targets PNG"' in source
    assert 'active_asset_label = "constructor" if active_is_constructor else "driver"' in source
    assert '"Download constructor targets PNG"' not in source


def test_diagnostics_are_contained_and_mobile_sprint_table_has_room_for_final():
    source = _source()

    assert 'st.container(key="diagnostics_summary_metrics")' in source
    assert ".st-key-diagnostics_summary_metrics" in source
    assert 'st.expander("Technical diagnostics", expanded=False)' in source
    assert 'key=f"sprint_diagnostics_mobile_{label.casefold()}"' in source
    assert "sprint_diagnostic_table_html(frame)" in source
    assert ".f1-sprint-mobile th:first-child" in source
    assert "width: 31%;" in source


def test_race_window_copy_reports_eligible_and_selected_counts():
    source = _source()

    assert 'f"{len(race_catalogue)} eligible · "' in source
    assert 'f"{len(race_control.selection.included)} selected"' in source
