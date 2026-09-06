from pathlib import Path
import re

from f1fantasy.ui_styles import DASHBOARD_CSS


APP_SOURCE = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _source() -> str:
    return APP_SOURCE.read_text(encoding="utf-8")


def test_app_loads_the_shared_responsive_stylesheet():
    source = _source()
    assert "from f1fantasy.ui_styles import DASHBOARD_CSS" in source
    assert "st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)" in source
    assert "--f1-bg:" in DASHBOARD_CSS
    assert "@media" in DASHBOARD_CSS


def test_contextual_navigation_stays_inline_and_only_controls_optimise():
    source = _source()
    assert 'key="optimise_mobile_subview"' in source
    assert ".st-key-optimise_mobile_subview" in DASHBOARD_CSS
    # Navigation must not overlay results or compete with the iPhone safe area.
    nav_rules = re.findall(
        r"\.st-key-optimise_mobile_subview\s*\{([^}]+)\}", DASHBOARD_CSS
    )
    assert nav_rules
    assert not any("position: fixed" in rule for rule in nav_rules)
    equal_tabs = re.search(
        r'\.st-key-optimise_mobile_subview \[data-testid="stButtonGroup"\] > div\s*\{([^}]+)\}',
        DASHBOARD_CSS,
    )
    assert equal_tabs is not None
    assert "grid-template-columns: repeat(4, minmax(0, 1fr))" in equal_tabs.group(1)


def test_mobile_subviews_keep_widgets_mounted_while_showing_the_selected_area():
    source = _source()
    for view in ("teams", "drivers", "constructors", "controls"):
        assert f"f1-optimise-view-{view}" in DASHBOARD_CSS
    for container in (
        "optimiser_teams_view",
        "optimiser_drivers_view",
        "optimiser_constructors_view",
        "optimiser_controls_view",
    ):
        assert f'key="{container}"' in source
        assert f".st-key-{container}" in DASHBOARD_CSS
    for view in ("teams", "drivers", "constructors", "controls"):
        assert (
            f"body:has(.f1-optimise-view-{view}) .st-key-optimiser_{view}_view"
            in DASHBOARD_CSS
        )


def test_mobile_results_expand_the_streamlit_wrapper_to_keep_exports_in_flow():
    wrapper = '[data-testid="stLayoutWrapper"]:has(> .st-key-optimiser_results_scroll)'
    rule = re.search(re.escape(wrapper) + r"[^{}]*\{([^}]+)\}", DASHBOARD_CSS)
    assert rule is not None
    # Expanding only the inner container clips later teams under load/export controls.
    assert "height: auto !important" in rule.group(1)
    assert "overflow: visible !important" in rule.group(1)


def test_mobile_threshold_schema_and_driver_export_label_are_correct():
    source = _source()
    assert "f1-threshold-table" in source
    for label in ("Asset", "Price ($M)", "Terrible", "Poor", "Good", "Great"):
        assert f'<th scope="col">{label}</th>' in source
    assert 'f"Download {active_asset_label} targets PNG"' in source
    assert 'active_asset_label = "constructor" if active_is_constructor else "driver"' in source
    assert '"Download constructor targets PNG"' not in source


def test_diagnostics_and_full_tables_remain_available():
    source = _source()
    assert 'st.container(key="diagnostics_summary_metrics")' in source
    assert 'st.expander("Technical diagnostics", expanded=False)' in source
    assert 'key=f"sprint_diagnostics_mobile_{label.casefold()}"' in source
    assert "sprint_diagnostic_table_html(frame)" in source
    assert ".f1-mobile-schema" in DASHBOARD_CSS
    assert ".f1-desktop-table" in DASHBOARD_CSS
    assert ".f1-mobile-table" in DASHBOARD_CSS


def test_race_window_copy_reports_eligible_and_selected_counts():
    source = _source()
    assert 'f"{len(race_catalogue)} eligible · "' in source
    assert 'f"{len(race_control.selection.included)} selected"' in source
