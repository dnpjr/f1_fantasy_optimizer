from __future__ import annotations

import html
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from f1fantasy.app_core import (
    DEFAULT_TOP_K,
    DEFAULT_PRICE_CHANGE_BOUNDS,
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
    DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
    CHIP_LIMITLESS,
    CHIP_NO_NEGATIVE,
    CHIP_NONE,
    CHIP_TRIPLE,
    HISTORY_MODE_ALL_SUPPORTED,
    HISTORY_MODE_CURRENT_SEASON_ONLY,
    OBJECTIVE_COMBINED,
    OBJECTIVE_POINTS_ONLY,
    OBJECTIVE_PRICE_GROWTH_ONLY,
    OBJECTIVE_RISK_ADJUSTED_COMBINED,
    PRICE_BAND_STYLES,
    apply_no_negative_scores,
    annotate_card_expected_points,
    apply_objective_mode,
    apply_probabilistic_price_change_model,
    build_asset_option_labels,
    build_holding_asset_universe,
    build_price_change_asset_universe,
    build_transfer_result_signature,
    build_transfer_recommendations,
    chip_mode_from_label,
    current_team_json,
    current_team_budget_from_selection,
    current_team_option_labels,
    current_team_selection_signature,
    current_team_upload_transition,
    copy_live_data_snapshot,
    copy_model_data,
    derive_model_data,
    effective_current_race_points,
    auto_budget_from_team_cost,
    resolve_budget_value,
    fantasy_asset_card_html,
    fantasy_card_grid_html,
    ranked_team_component_html,
    format_countdown,
    parse_team_lock_deadline_timestamp,
    format_money,
    format_points,
    format_probability,
    format_signed_money,
    format_signed_points,
    format_transfer_recommendations_display,
    hide_user_internal_columns,
    format_next_race_header,
    format_selected_asset_display_table,
    live_data_snapshot_identity,
    load_live_data_snapshot,
    model_data_version,
    model_settings_signature,
    market_runtime_status,
    price_change_threshold_table,
    price_change_probability_matrix_table,
    price_change_projection_summary_table,
    price_change_target_summary_table,
    projected_team_value_from_budget,
    reconcile_imported_budget_suggestion,
    refresh_status_transition,
    resolve_derived_model_data,
    resolve_live_data_snapshot,
    optimizer_budget_state_updates,
    run_optimizer,
    selected_assets_price_gain,
    snapshot_race_catalogue,
    transfer_baseline,
    select_chip_boost_drivers,
    team_expected_points_with_chips,
    team_colour,
    validate_current_team,
)
from f1fantasy.price_efficiency import build_price_efficiency_table, summarize_price_efficiency_team
from f1fantasy.live_session_shadow import completed_live_session_labels
from f1fantasy.weekend_state import EventKey
from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH, load_canonical_scores
from f1fantasy.race_selection import recency_weights
from f1fantasy.exports import (
    render_price_change_table_png,
    render_price_efficiency_table_png,
    render_price_efficiency_team_png,
    render_projected_team_png,
    safe_export_filename,
)
from f1fantasy.ui_helpers import (
    OPTIMISE_MOBILE_SUBVIEWS,
    PRIMARY_NAVIGATION_AREAS,
    PRICE_EFFICIENCY_SORT_COLUMNS,
    asset_constraint_transition,
    compact_asset_table_html,
    compact_asset_universe_rows,
    effective_blend_percentages,
    gain_value_class,
    next_team_batch,
    normalize_price_growth_value,
    optimise_mobile_subview,
    optimiser_result_signature,
    prepare_price_efficiency_display,
    prepare_compact_asset_table,
    price_efficiency_race_summary,
    price_efficiency_table_html,
    race_option_label,
    race_weight_summary,
    ranked_solution_current_team_update,
    reconcile_constraint_pair,
    reconcile_price_efficiency_team_state,
    reconcile_race_control_state,
    resolve_price_efficiency_asset_type,
    sprint_diagnostic_table_html,
    team_summary_html,
    team_summary_payload,
    team_solution_key,
)


st.set_page_config(
    page_title="F1 Fantasy Optimiser",
    page_icon="F1",
    layout="wide",
)


def _inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --f1-bg: #07090f;
            --f1-panel: #11141c;
            --f1-panel-soft: #171b24;
            --f1-border: rgba(255, 255, 255, 0.10);
            --f1-text: #f8fafc;
            --f1-muted: #a8b0bd;
            --f1-red: #e10600;
            --f1-red-soft: rgba(225, 6, 0, 0.18);
            --f1-green: #22c55e;
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(225, 6, 0, 0.18), transparent 28rem),
                linear-gradient(135deg, #07090f 0%, #0d111a 48%, #07090f 100%);
            color: var(--f1-text);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        h1 {
            font-weight: 900;
            border-left: 5px solid var(--f1-red);
            padding-left: 0.9rem;
        }
        h2, h3 {
            font-weight: 800;
        }
        .f1-app-header h1 {
            margin: 0;
        }
        .f1-app-header p {
            margin: 0.25rem 0 0.45rem 1.25rem;
            color: var(--f1-muted);
            font-size: 0.88rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.025));
            border: 1px solid var(--f1-border);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        }
        div[data-testid="stMetric"] label {
            color: var(--f1-muted) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--f1-border);
            overflow-x: auto;
            overflow-y: hidden;
            flex-wrap: nowrap;
            scrollbar-width: thin;
            touch-action: pan-x;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 0.9rem;
            flex: 0 0 auto;
            min-height: 44px;
            white-space: nowrap;
        }
        .stTabs [aria-selected="true"] {
            border-color: rgba(225, 6, 0, 0.65);
            box-shadow: inset 0 -3px 0 var(--f1-red);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 19, 29, 0.72);
            border-color: var(--f1-border);
            border-radius: 8px;
        }
        .f1-race-card, .f1-panel-card {
            border: 1px solid var(--f1-border);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.072), rgba(255, 255, 255, 0.026));
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
        }
        .f1-race-card {
            padding: 0.65rem 0.85rem;
            margin: 0.4rem 0 0.65rem;
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 1rem;
            align-items: center;
            overflow: hidden;
            position: relative;
        }
        .f1-race-card:before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: var(--f1-red);
        }
        .f1-round {
            color: #fff;
            background: var(--f1-red);
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .f1-race-name {
            font-size: clamp(1.05rem, 1.8vw, 1.5rem);
            line-height: 1.05;
            font-weight: 900;
        }
        .f1-race-date, .f1-race-sub {
            color: var(--f1-muted);
            font-weight: 650;
        }
        .f1-race-countdown {
            font-size: 1.05rem;
            font-weight: 900;
            letter-spacing: 0.03em;
            color: #ffffff;
            margin-top: 0.25rem;
        }
        .f1-section-kicker {
            color: var(--f1-red);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 1rem 0 0.25rem;
        }
        .f1-card-grid {
            display: grid;
            --f1-asset-gap: 6px;
            gap: var(--f1-asset-gap);
            margin: 0;
        }
        .f1-driver-grid {
            grid-template-columns: repeat(5, minmax(64px, 1fr));
            overflow-x: auto;
            scrollbar-width: thin;
        }
        .f1-constructor-grid {
            grid-template-columns: repeat(2, minmax(64px, 1fr));
            width: calc(40% - 3.6px);
            min-width: 134px;
            margin-inline: auto;
        }
        .f1-driver-card {
            position: relative;
            width: 100%;
            min-width: 0;
            height: 78px;
            border: 1px solid var(--f1-border);
            border-top: 3px solid var(--team-color, #64748b);
            border-radius: 9px;
            background: #131722;
            overflow: hidden;
            padding: 6px 7px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 3px;
        }
        .f1-card-top {
            display: flex;
            justify-content: flex-start;
            gap: 4px;
            align-items: center;
        }
        .f1-card-identity {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            min-width: 0;
        }
        .f1-driver-card .f1-asset-id {
            min-width: 0;
            min-height: 0;
            padding: 0;
            border-radius: 0;
            background: transparent !important;
            color: #ffffff !important;
            font-size: 0.76rem;
            letter-spacing: 0.035em;
        }
        .f1-asset-id {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 3.15rem;
            min-height: 2rem;
            border-radius: 7px;
            padding: 0.28rem 0.55rem;
            font-size: 0.88rem;
            font-weight: 950;
            letter-spacing: 0.055em;
            white-space: nowrap;
        }
        .f1-card-price, .f1-card-points, .f1-card-gain {
            color: #ffffff;
            font-weight: 850;
            white-space: nowrap;
        }
        .f1-card-middle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 2px;
            min-width: 0;
        }
        .f1-card-price {
            font-size: 0.68rem;
        }
        .f1-card-points {
            font-size: 0.70rem;
        }
        .f1-card-gain {
            font-size: 0.68rem;
        }
        .f1-boost {
            background: var(--f1-red);
            color: #fff;
            border-radius: 999px;
            padding: 1px 4px;
            font-size: 0.58rem;
            font-weight: 900;
            white-space: nowrap;
        }
        .f1-availability-muted {
            background: #303744;
            color: #cbd5e1;
            border: 1px solid #4b5563;
            border-radius: 999px;
            padding: 1px 4px;
            font-size: 0.54rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .f1-gain-positive { color: #6ee7a3 !important; }
        .f1-gain-negative { color: #fb7185 !important; }
        .f1-gain-neutral { color: #cbd5e1 !important; }
        .f1-gain-missing { color: #7d8797 !important; }
        .f1-ranked-team {
            border: 1px solid var(--f1-border);
            border-radius: 11px;
            background: rgba(13, 17, 26, 0.92);
            padding: 7px 8px 8px;
            margin-bottom: 7px;
        }
        .f1-team-header {
            display: grid;
            grid-template-columns: 30px minmax(0, 1fr);
            align-items: center;
            gap: 6px;
            margin-bottom: 5px;
        }
        .f1-team-rank {
            display: grid;
            place-items: center;
            width: 29px;
            height: 29px;
            border-radius: 7px;
            background: var(--f1-red-soft);
            border: 1px solid rgba(225, 6, 0, 0.55);
            font-size: 0.92rem;
            font-weight: 900;
        }
        .f1-team-summary {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 4px;
        }
        .f1-team-stat {
            min-width: 0;
            text-align: right;
            line-height: 1.05;
        }
        .f1-team-stat span {
            display: block;
            color: var(--f1-muted);
            font-size: 0.67rem;
            text-transform: uppercase;
            letter-spacing: 0.035em;
        }
        .f1-team-stat strong {
            display: block;
            margin-top: 2px;
            color: #ffffff;
            font-size: 0.88rem;
            white-space: nowrap;
        }
        .f1-team-assets {
            display: grid;
            gap: 5px;
        }
        .f1-ranked-team .f1-driver-grid {
            grid-template-columns: repeat(5, minmax(64px, 142px));
            justify-content: center;
        }
        .f1-ranked-team .f1-constructor-grid {
            grid-template-columns: repeat(2, minmax(64px, 142px));
            width: auto;
            min-width: 0;
            justify-content: center;
        }
        .f1-ranked-team .f1-driver-card {
            height: 76px;
            padding: 5px 6px;
        }
        .f1-ranked-team .f1-driver-card .f1-asset-id { font-size: 0.86rem; }
        .f1-ranked-team .f1-card-price,
        .f1-ranked-team .f1-card-gain { font-size: 0.76rem; }
        .f1-ranked-team .f1-card-points { font-size: 0.79rem; }
        .f1-ranked-team .f1-boost { font-size: 0.65rem; }
        .st-key-optimiser_quick_setup [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.35rem 0.55rem 0.2rem;
        }
        .st-key-optimiser_quick_setup [data-testid="stWidgetLabel"] {
            margin-bottom: 0.1rem;
        }
        .f1-universe-heading {
            color: var(--f1-muted);
            font-size: 0.66rem;
            font-weight: 850;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .f1-universe-number {
            display: block;
            padding-top: 0.34rem;
            font-size: 0.76rem;
            font-weight: 750;
            white-space: nowrap;
        }
        [class*="st-key-optimiser_universe_scroll"] div[data-testid="stHorizontalBlock"] {
            min-height: 40px;
            gap: 2px;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.07);
        }
        [class*="st-key-optimiser_universe_scroll"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            min-width: 0;
        }
        [class*="st-key-optimiser_universe_scroll"] [data-testid="stMarkdownContainer"] p {
            margin: 0;
        }
        [class*="st-key-optimiser_universe_scroll"] [data-testid="stCheckbox"] {
            display: flex;
            justify-content: center;
            margin: 0;
        }
        .f1-results-scroll, .f1-universe-scroll {
            max-height: min(68vh, 650px);
            overflow-y: auto;
            overscroll-behavior: contain;
            padding-right: 3px;
        }
        .f1-universe-table {
            min-width: 0;
            table-layout: auto;
        }
        .f1-universe-table th, .f1-universe-table td {
            height: 46px;
            padding: 0.38rem 0.42rem;
        }
        .f1-table-note {
            color: var(--f1-muted);
            font-size: 0.72rem;
            padding: 0.4rem 0.55rem;
        }
        .f1-transfer-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            justify-content: center;
            gap: 0.7rem;
            align-items: center;
            margin-bottom: 0.75rem;
            width: min(100%, 620px);
            margin-inline: auto;
        }
        .f1-transfer-card-slot {
            display: flex;
            justify-content: center;
        }
        .f1-transfer-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            font-weight: 900;
            color: #ffffff;
            min-width: 2.25rem;
        }
        .f1-table-scroll {
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--f1-border);
            border-radius: 10px;
            margin: 0.55rem 0 0.9rem;
        }
        .f1-compact-table {
            width: 100%;
            min-width: 660px;
            border-collapse: collapse;
            font-size: 0.88rem;
        }
        .f1-compact-table th, .f1-compact-table td {
            padding: 0.68rem 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            text-align: right;
            vertical-align: middle;
        }
        .f1-compact-table th {
            color: var(--f1-muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .f1-compact-table th:first-child, .f1-compact-table td:first-child {
            text-align: left;
            position: sticky;
            left: 0;
            background: #11141c;
            z-index: 1;
        }
        .f1-compact-table small {
            color: var(--f1-muted);
            display: block;
            white-space: nowrap;
            margin-top: 0.12rem;
        }
        .f1-price-change-table {
            min-width: 720px;
        }
        .f1-price-change-table th, .f1-price-change-table td {
            white-space: nowrap;
        }
        .f1-price-change-table th:first-child,
        .f1-price-change-table td:first-child {
            width: 5.4rem;
            min-width: 5.4rem;
            max-width: 5.4rem;
        }
        .f1-empty-table {
            color: var(--f1-muted);
            padding: 0.8rem;
            border: 1px dashed var(--f1-border);
            border-radius: 10px;
        }
        .f1-mobile-table,
        .st-key-optimise_mobile_subview,
        .st-key-optimiser_teams_action,
        .st-key-optimiser_mobile_model_controls,
        [class*="st-key-sprint_diagnostics_mobile"] {
            display: none;
        }
        @media (min-width: 769px) {
            body:has(.f1-universe-desktop-drivers) .st-key-optimiser_constructors_view,
            body:has(.f1-universe-desktop-constructors) .st-key-optimiser_drivers_view {
                display: none;
            }
        }
        @media (max-width: 1024px) {
            .block-container {
                padding-left: 1.25rem;
                padding-right: 1.25rem;
            }
        }
        @media (max-width: 768px) {
            .block-container {
                padding-bottom: calc(68px + env(safe-area-inset-bottom) + 16px);
            }
            .f1-race-card {
                grid-template-columns: 1fr;
            }
            .f1-transfer-row {
                grid-template-columns: 1fr;
                width: 100%;
            }
            .f1-transfer-arrow {
                min-height: 1.5rem;
            }
            div[data-testid="stMetric"] {
                min-height: 72px;
            }
            button, input, [role="combobox"] {
                min-height: 44px;
            }
            .st-key-optimise_mobile_subview {
                display: block;
                position: fixed;
                z-index: 999;
                left: 8px;
                width: calc(100vw - 16px) !important;
                box-sizing: border-box;
                bottom: calc(8px + env(safe-area-inset-bottom));
                padding: 5px;
                border: 1px solid var(--f1-border);
                border-radius: 12px;
                background: rgba(7, 9, 15, 0.96);
                box-shadow: 0 12px 30px rgba(0,0,0,0.45);
                backdrop-filter: blur(12px);
            }
            .st-key-optimise_mobile_subview [data-testid="stButtonGroup"] {
                width: 100% !important;
            }
            .st-key-optimise_mobile_subview [data-testid="stButtonGroup"] > div {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                width: 100% !important;
                gap: 4px;
            }
            .st-key-optimise_mobile_subview button {
                min-width: 0;
                min-height: 48px;
                padding: 5px 3px;
                border-radius: 8px;
                font-size: 0.72rem;
                white-space: nowrap;
            }
            .st-key-optimise_mobile_subview button[aria-checked="true"] {
                background: rgba(255,255,255,0.10);
                box-shadow: inset 0 -2px 0 var(--f1-red);
            }
            .st-key-optimiser_controls_view,
            .st-key-optimiser_teams_action,
            .st-key-optimiser_teams_view,
            .st-key-optimiser_drivers_view,
            .st-key-optimiser_constructors_view {
                display: none;
            }
            body:has(.f1-optimise-view-teams) .st-key-optimiser_teams_action,
            body:has(.f1-optimise-view-teams) .st-key-optimiser_teams_view,
            body:has(.f1-optimise-view-drivers) .st-key-optimiser_drivers_view,
            body:has(.f1-optimise-view-constructors) .st-key-optimiser_constructors_view,
            body:has(.f1-optimise-view-controls) .st-key-optimiser_controls_view {
                display: block;
            }
            .st-key-optimiser_universe_selector {
                display: none;
            }
            .st-key-diagnostics_summary_metrics {
                display: none;
            }
            .st-key-optimiser_mobile_model_controls {
                display: block;
            }
            .f1-desktop-table,
            [class*="st-key-sprint_diagnostics_desktop"] {
                display: none;
            }
            .f1-mobile-table,
            [class*="st-key-sprint_diagnostics_mobile"] {
                display: block;
            }
            .f1-mobile-schema {
                width: 100%;
                min-width: 0;
                table-layout: fixed;
                font-size: 0.88rem;
                font-variant-numeric: tabular-nums;
            }
            .f1-mobile-schema th,
            .f1-mobile-schema td {
                padding: 0.58rem 0.42rem;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .f1-mobile-schema th:first-child,
            .f1-mobile-schema td:first-child {
                width: 29%;
            }
            .f1-efficiency-mobile th:first-child,
            .f1-efficiency-mobile td:first-child {
                width: 38%;
            }
            .f1-sprint-mobile th:first-child,
            .f1-sprint-mobile td:first-child {
                width: 31%;
            }
            .st-key-optimiser_quick_setup div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            .st-key-optimiser_quick_setup div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            .st-key-optimiser_quick_setup div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(n+2) {
                flex: 1 1 calc(33.333% - 6px) !important;
                min-width: 0 !important;
            }
            .f1-ranked-team .f1-driver-grid {
                grid-template-columns: repeat(6, minmax(0, 1fr));
                justify-content: stretch;
                overflow: visible;
            }
            .f1-ranked-team .f1-driver-grid .f1-driver-card:nth-child(-n+3) {
                grid-column: span 2;
            }
            .f1-ranked-team .f1-driver-grid .f1-driver-card:nth-child(n+4) {
                grid-column: span 3;
            }
            .f1-ranked-team .f1-constructor-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                width: 100%;
                min-width: 0;
            }
        }
        @media (max-width: 480px) {
            .block-container {
                padding: 0.65rem 0.55rem calc(68px + env(safe-area-inset-bottom) + 16px);
            }
            h1 {
                font-size: 1.25rem;
                padding-left: 0.65rem;
            }
            h2 { font-size: 1.35rem; }
            h3 { font-size: 1.08rem; }
            .f1-race-card {
                padding: 0.55rem 0.7rem;
                gap: 0.35rem;
                grid-template-columns: auto 1fr;
            }
            .f1-race-card > div:last-child {
                display: none;
            }
            .f1-app-header p {
                display: none;
            }
            .f1-driver-card {
                height: 76px;
                padding: 5px;
            }
            .f1-ranked-team .f1-card-price,
            .f1-ranked-team .f1-card-gain { font-size: 0.75rem; }
            .f1-ranked-team .f1-card-points { font-size: 0.80rem; }
            .f1-ranked-team .f1-driver-card .f1-asset-id { font-size: 0.86rem; }
            .f1-price-change-table th, .f1-price-change-table td {
                padding: 0.48rem 0.45rem;
                font-size: 0.78rem;
            }
            .f1-price-change-table .f1-asset-id {
                min-width: 2.8rem;
                padding-inline: 0.4rem;
                font-size: 0.78rem;
            }
            .stTabs [data-baseweb="tab"] {
                padding-inline: 0.72rem;
            }
            .f1-ranked-team {
                padding-inline: 7px;
            }
            .f1-team-header {
                grid-template-columns: 25px minmax(0, 1fr);
                gap: 4px;
            }
            .f1-team-stat strong { font-size: 0.70rem; }
            .f1-team-stat span { font-size: 0.55rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_dashboard_css()
LOGGER = logging.getLogger(__name__)
RAW_LIVE_HISTORY_SEASONS = 3


def _option_labels(df: pd.DataFrame) -> dict[str, str]:
    return build_asset_option_labels(df)


def _load_current_team_config() -> dict:
    path = Path("data/current_team.local.json")
    if not path.exists():
        return {"drivers": [], "constructors": [], "free_transfers": 2, "bank": 0.0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"drivers": [], "constructors": [], "free_transfers": 2, "bank": 0.0}


def _with_points_per_million(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price = pd.to_numeric(out["price"], errors="coerce")
    expected = pd.to_numeric(out["exp_score"], errors="coerce")
    out["points_per_million"] = (expected / price).where(price > 0)
    return out


def _styleable_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return hide_user_internal_columns(df.copy())


def _asset_table_with_images(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    config = {}
    if "image_url" in out.columns and out["image_url"].fillna("").astype(str).str.len().gt(0).any():
        out.rename(columns={"image_url": "Image"}, inplace=True)
        config["Image"] = st.column_config.ImageColumn("Image", width="small")
    elif "image_url" in out.columns:
        out.drop(columns=["image_url"], inplace=True)
    return out, config


def _display_table_with_optional_images(df: pd.DataFrame):
    out = df.copy()
    config = {}
    if "Image" in out.columns and out["Image"].fillna("").astype(str).str.len().gt(0).any():
        config["Image"] = st.column_config.ImageColumn("Image", width="small")
    elif "Image" in out.columns:
        out.drop(columns=["Image"], inplace=True)
    return out, config


def _safe_text(value: object, fallback: str = "") -> str:
    if value is None or pd.isna(value):
        return fallback
    return html.escape(str(value))


def _asset_card_grid(
    df: pd.DataFrame,
    boosted_driver: str | None = None,
    triple_driver: str | None = None,
    asset_label: str = "Driver",
) -> None:
    if df.empty:
        st.info("No assets selected yet.")
        return
    display_df = annotate_card_expected_points(df, boosted_driver=boosted_driver, triple_driver=triple_driver)
    st.markdown(
        fantasy_card_grid_html(display_df, boosted_driver=boosted_driver, triple_driver=triple_driver, asset_label=asset_label),
        unsafe_allow_html=True,
    )


def _ranked_team_component(solution, *, rank: int, budget: float, limitless: bool) -> None:
    summary = _team_summary(solution, budget)
    display_drivers = annotate_card_expected_points(
        solution.drivers,
        boosted_driver=solution.boosted_driver,
        triple_driver=solution.triple_driver,
    )
    st.markdown(
        ranked_team_component_html(
            rank=rank,
            summary=summary,
            drivers=display_drivers,
            constructors=solution.constructors,
            boosted_driver=solution.boosted_driver,
            triple_driver=solution.triple_driver,
            limitless=limitless,
        ),
        unsafe_allow_html=True,
    )


_CONSTRAINT_STATE_KEYS = (
    "locked_driver_ids",
    "excluded_driver_ids",
    "locked_constructor_ids",
    "excluded_constructor_ids",
)


def _on_bulk_constraint_change(primary_key: str, conflicting_key: str) -> None:
    primary, conflicting = reconcile_constraint_pair(
        st.session_state.get(primary_key, ()),
        st.session_state.get(conflicting_key, ()),
    )
    st.session_state[primary_key] = primary
    st.session_state[conflicting_key] = conflicting


def _on_inline_asset_constraint_change(
    asset_type: str,
    asset_id: str,
    action: str,
    widget_key: str,
) -> None:
    transition = asset_constraint_transition(
        {key: st.session_state.get(key, ()) for key in _CONSTRAINT_STATE_KEYS},
        asset_type=asset_type,
        asset_id=asset_id,
        action=action,
        active=bool(st.session_state.get(widget_key)),
    )
    for key, value in transition.items():
        st.session_state[key] = value
    opposite = "exclude" if action == "lock" else "lock"
    opposite_key = f"optimiser_universe_{asset_type}_{asset_id}_{opposite}"
    if bool(st.session_state.get(widget_key)) and opposite_key in st.session_state:
        st.session_state[opposite_key] = False


def _render_compact_asset_universe(
    assets: pd.DataFrame,
    *,
    asset_type: str,
    locked_ids: list[str],
    excluded_ids: list[str],
    container_key: str = "optimiser_universe_scroll",
) -> None:
    rows = compact_asset_universe_rows(
        assets,
        asset_type=asset_type,
        locked_ids=locked_ids,
        excluded_ids=excluded_ids,
    )
    if not rows:
        st.info("No assets available.")
        return
    with st.container(height=620, border=True, key=container_key):
        header = st.columns([1.34, 0.72, 0.78, 0.68, 0.55, 0.67], gap="small")
        for column, label in zip(header, ("Asset", "Price", "Gain", "EV", "Lock", "Out")):
            column.markdown(f'<span class="f1-universe-heading">{label}</span>', unsafe_allow_html=True)
        for row in rows:
            columns = st.columns([1.34, 0.72, 0.78, 0.68, 0.55, 0.67], gap="small")
            columns[0].markdown(row["asset"], unsafe_allow_html=True)
            columns[1].markdown(f'<span class="f1-universe-number">{row["price"]}</span>', unsafe_allow_html=True)
            columns[2].markdown(
                f'<span class="f1-universe-number {row["gain_class"]}">{row["gain"]}</span>',
                unsafe_allow_html=True,
            )
            columns[3].markdown(f'<span class="f1-universe-number">{row["points"]}</span>', unsafe_allow_html=True)
            lock_key = f'optimiser_universe_{asset_type}_{row["asset_id"]}_lock'
            exclude_key = f'optimiser_universe_{asset_type}_{row["asset_id"]}_exclude'
            st.session_state[lock_key] = bool(row["lock"])
            st.session_state[exclude_key] = bool(row["exclude"])
            with columns[4]:
                st.checkbox(
                    f'Lock {row["abbreviation"]}',
                    key=lock_key,
                    label_visibility="collapsed",
                    help=f'Lock {row["abbreviation"]} into optimiser teams',
                    on_change=_on_inline_asset_constraint_change,
                    args=(asset_type, row["asset_id"], "lock", lock_key),
                )
            with columns[5]:
                st.checkbox(
                    f'Exclude {row["abbreviation"]}',
                    key=exclude_key,
                    label_visibility="collapsed",
                    help=f'Exclude {row["abbreviation"]} from optimiser teams',
                    on_change=_on_inline_asset_constraint_change,
                    args=(asset_type, row["asset_id"], "exclude", exclude_key),
                )


def _copy_ranked_team_to_current(
    solution,
    valid_driver_ids: list[str],
    valid_constructor_ids: list[str],
    rank: int,
) -> None:
    transition = ranked_solution_current_team_update(
        solution,
        valid_driver_ids=valid_driver_ids,
        valid_constructor_ids=valid_constructor_ids,
    )
    if not transition["ok"]:
        st.session_state["ranked_team_copy_error"] = transition["error"]
        st.session_state["ranked_team_copy_notice"] = None
        return
    for key, value in transition["updates"].items():
        st.session_state[key] = value
    st.session_state["ranked_team_copy_error"] = None
    st.session_state["ranked_team_copy_notice"] = f"Team {int(rank)} copied to Current Team."


def _render_png_download(
    label: str,
    *,
    filename: str,
    key: str,
    renderer,
) -> None:
    try:
        payload = renderer()
    except Exception:
        LOGGER.exception("PNG export generation failed: %s", key)
        st.error(f"{label} is temporarily unavailable. The rest of the page is unaffected.")
        return
    st.download_button(
        label,
        data=payload,
        file_name=safe_export_filename(filename),
        mime="image/png",
        key=key,
        use_container_width=True,
        on_click="ignore",
    )


def _price_efficiency_option_labels(table: pd.DataFrame) -> dict[str, str]:
    if table is None or table.empty:
        return {}
    labels: dict[str, str] = {}
    for _, row in table.iterrows():
        asset_id = str(row.get("asset_id", ""))
        abbreviation_value = row.get("abbreviation")
        abbreviation = "?" if pd.isna(abbreviation_value) else str(abbreviation_value)
        full_name_value = row.get("full_name")
        full_name = abbreviation if pd.isna(full_name_value) else str(full_name_value)
        price = format_money(row.get("current_price"))
        labels[asset_id] = f"{abbreviation} · {full_name} · {price}"
    return labels


def _price_efficiency_source_failure_ids(
    table: pd.DataFrame,
    asset_type: str,
) -> list[tuple[str, str]]:
    if table is None or table.empty or not {"asset_id", "has_source_failure"}.issubset(table.columns):
        return []
    failed = table["has_source_failure"].fillna(False).astype(bool)
    return [(asset_type, str(asset_id)) for asset_id in table.loc[failed, "asset_id"]]


def _render_price_efficiency_section(
    table: pd.DataFrame,
    title: str,
    key_prefix: str,
    *,
    race_summary: str,
) -> pd.DataFrame:
    st.subheader(title)
    sort_col, order_col = st.columns([2, 1])
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            options=list(PRICE_EFFICIENCY_SORT_COLUMNS),
            key=f"{key_prefix}_efficiency_sort",
        )
    with order_col:
        ascending = st.checkbox("Ascending", value=False, key=f"{key_prefix}_efficiency_ascending")
    display = prepare_price_efficiency_display(table, sort_by=sort_by, ascending=ascending)
    st.markdown(price_efficiency_table_html(display), unsafe_allow_html=True)
    if display.empty:
        st.caption(f"No {title.casefold()} table is available to export.")
    else:
        sort_label = f"{sort_by} {'ascending' if ascending else 'descending'}"
        _render_png_download(
            f"Download {title.casefold()} table PNG",
            filename=f"f1_{key_prefix}_price_efficiency.png",
            key=f"download_{key_prefix}_price_efficiency_png",
            renderer=lambda: render_price_efficiency_table_png(
                display,
                asset_type=key_prefix,
                race_summary=race_summary,
                sort_label=sort_label,
            ),
        )
    with st.expander(f"{title} details", expanded=False):
        detail_columns = [
            "asset_id",
            "full_name",
            "team_name",
            "current_price",
            "selected_points_total",
            "average_points_per_race",
            "price_efficiency",
            "selected_race_count",
            "valid_race_count",
            "missing_race_count",
            "coverage_fraction",
            "has_source_failure",
            "status",
            "valid_race_keys",
        ]
        details = table[[column for column in detail_columns if column in table.columns]].copy()
        if "valid_race_keys" in details.columns:
            details["valid_race_keys"] = details["valid_race_keys"].apply(
                lambda keys: ", ".join(
                    f"{key.season} R{key.round}"
                    for key in keys
                    if hasattr(key, "season") and hasattr(key, "round")
                )
                if isinstance(keys, (tuple, list))
                else ""
            )
        st.dataframe(details, hide_index=True, width="stretch")
    return display


def _render_race_header(diagnostics: dict) -> None:
    race_name = diagnostics.get("next_race_name")
    race_date = diagnostics.get("next_race_date")
    round_no = diagnostics.get("next_race_round")
    lock_deadline_raw = diagnostics.get("team_lock_deadline_utc")
    lock_source = diagnostics.get("team_lock_deadline_source", "unavailable")
    lock_deadline = parse_team_lock_deadline_timestamp(lock_deadline_raw)
    deadline_note = "Team lock deadline" if lock_deadline is not None else "Team lock deadline unavailable"
    countdown_markup = (
        f'<div class="f1-race-countdown">{_safe_text(format_countdown(lock_deadline))}</div>'
        if lock_deadline is not None
        else ""
    )
    race_label = _safe_text(race_name, "Next race")
    date_label = format_next_race_header(None, race_date).replace("Next race: ", "").replace("Next race", "")
    round_label = f"Round {int(round_no)}" if round_no else "Next race"
    st.markdown(
        f"""
        <div class="f1-race-card">
            <div class="f1-round">{_safe_text(round_label)}</div>
            <div>
                <div class="f1-race-sub">Upcoming fantasy deadline focus</div>
                <div class="f1-race-name">{race_label}</div>
            </div>
            <div>
                <div class="f1-race-date">{_safe_text(date_label.strip(), "Date TBC")}</div>
                <div class="f1-race-sub">{_safe_text(deadline_note)}</div>
                {countdown_markup}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _asset_editor(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    sort_options = {
        "Expected Points": "exp_score",
        "Price": "price",
        "Points per million": "points_per_million",
        "Volatility / race": "volatility",
    }
    sort_col, ascending = st.columns([2, 1])
    with sort_col:
        selected_sort = st.selectbox(
            "Sort by",
            options=list(sort_options.keys()),
            index=0,
            key=f"{kind}_sort_by",
        )
    with ascending:
        sort_ascending = st.checkbox("Ascending", value=False, key=f"{kind}_sort_ascending")

    display = _with_points_per_million(df)
    display = display.sort_values(sort_options[selected_sort], ascending=sort_ascending, na_position="last")

    base_cols = [
        "id",
        "name",
        "price",
        "exp_score",
        "points_per_million",
        "dnf_rate",
        "volatility",
    ]
    if kind == "drivers" and "team" in df.columns:
        base_cols.insert(2, "team")

    cols = [col for col in base_cols if col in df.columns]
    if "points_per_million" not in cols:
        cols.insert(cols.index("dnf_rate"), "points_per_million")

    edited = st.data_editor(
        display[cols].copy(),
        hide_index=True,
        width="stretch",
        disabled=[
            col
            for col in [
                "id",
                "name",
                "team",
                "points_per_million",
            ]
            if col in cols
        ],
        column_config={
            "name": "Name",
            "team": "Team",
            "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.1, format="%.1f"),
            "exp_score": st.column_config.NumberColumn("Expected Points", step=0.1, format="%.2f"),
            "points_per_million": st.column_config.NumberColumn("Points / million", format="%.2f"),
            "dnf_rate": st.column_config.NumberColumn("DNF rate", min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
            "volatility": st.column_config.NumberColumn("Volatility / race", min_value=0.0, step=0.1, format="%.2f"),
        },
        key=f"{kind}_editor",
    )

    out = df.copy()
    editable_cols = [col for col in ["price", "exp_score", "dnf_rate", "volatility"] if col in edited.columns]
    edited_values = edited.copy()
    edited_values["id"] = edited_values["id"].astype(str)
    edited_values = edited_values.set_index("id")
    for col in editable_cols:
        mapped = out["id"].astype(str).map(pd.to_numeric(edited_values[col], errors="coerce"))
        out[col] = mapped.fillna(pd.to_numeric(out[col], errors="coerce"))

    advanced_cols = [
        col
        for col in [
            "image_url",
            "next_race_expected_points",
            "horizon_expected_points",
            "current_season_observed_avg_points_per_race",
            "historical_prior_expected_points_per_race",
            "normalised_historical_expected_points_per_race",
            "current_season_points_count",
            "current_season_avg_points",
            "current_season_volatility",
            "historical_volatility",
            "normalised_historical_volatility",
            "blended_volatility_before_floor",
            "volatility_source",
            "volatility_floor",
            "expected_points_source",
            "nn_exp_score",
            "exp_score_raw",
            "nn_exp_score_raw",
            "team_exp",
            "team_factor",
        ]
        if col in out.columns
    ]
    if advanced_cols:
        with st.expander("Advanced assumptions"):
            advanced_display = out[["id", "name", *advanced_cols]].copy()
            advanced_edited = st.data_editor(
                advanced_display,
                hide_index=True,
                width="stretch",
                disabled=[col for col in ["id", "name"] if col in advanced_display.columns],
                column_config={
                    "nn_exp_score": st.column_config.NumberColumn("No Negative EV", step=0.1, format="%.2f"),
                    "next_race_expected_points": st.column_config.NumberColumn("Next race EV", step=0.1, format="%.2f"),
                    "horizon_expected_points": st.column_config.NumberColumn("Horizon EV", step=0.1, format="%.2f"),
                    "current_season_observed_avg_points_per_race": st.column_config.NumberColumn("Current observed avg", step=0.1, format="%.2f"),
                    "historical_prior_expected_points_per_race": st.column_config.NumberColumn("Historical avg", step=0.1, format="%.2f"),
                    "normalised_historical_expected_points_per_race": st.column_config.NumberColumn("Normalised historical avg", step=0.1, format="%.2f"),
                    "current_season_points_count": st.column_config.NumberColumn("Current race count", step=1, format="%d"),
                    "current_season_avg_points": st.column_config.NumberColumn("Current observed avg", step=0.1, format="%.2f"),
                    "current_season_volatility": st.column_config.NumberColumn("Current volatility", step=0.1, format="%.2f"),
                    "historical_volatility": st.column_config.NumberColumn("Historical volatility", step=0.1, format="%.2f"),
                    "normalised_historical_volatility": st.column_config.NumberColumn("Normalised historical volatility", step=0.1, format="%.2f"),
                    "blended_volatility_before_floor": st.column_config.NumberColumn("Blended volatility before floor", step=0.1, format="%.2f"),
                    "volatility_floor": st.column_config.NumberColumn("Volatility floor", step=0.1, format="%.2f"),
                    "exp_score_raw": st.column_config.NumberColumn("Raw expected", step=0.1, format="%.2f"),
                    "nn_exp_score_raw": st.column_config.NumberColumn("Raw No Negative EV", step=0.1, format="%.2f"),
                    "team_exp": st.column_config.NumberColumn("Team expected", step=0.1, format="%.2f"),
                    "team_factor": st.column_config.NumberColumn("Team factor", step=0.01, format="%.3f"),
                    "image_url": st.column_config.TextColumn("Image URL"),
                },
                key=f"{kind}_advanced_editor",
            )
        advanced_values = advanced_edited.copy()
        advanced_values["id"] = advanced_values["id"].astype(str)
        advanced_values = advanced_values.set_index("id")
        for col in advanced_cols:
            if col in {"image_url", "expected_points_source"}:
                out[col] = out["id"].astype(str).map(advanced_values[col].fillna("").astype(str)).fillna("")
            else:
                mapped = out["id"].astype(str).map(pd.to_numeric(advanced_values[col], errors="coerce"))
                out[col] = mapped.fillna(pd.to_numeric(out[col], errors="coerce"))

    if "nn_exp_score" in out.columns:
        out["nn_exp_score"] = pd.to_numeric(out["nn_exp_score"], errors="coerce").fillna(out["exp_score"])
    return out


def _solutions_table(solutions, budget: float | None = None) -> pd.DataFrame:
    rows = []
    for idx, sol in enumerate(solutions, start=1):
        expected_points = _team_expected_points(sol)
        price_change = _team_price_change(sol)
        objective_score = float(sol.expected_score)
        rows.append(
            {
                "Rank": idx,
                "Drivers": ", ".join(sol.drivers.sort_values("price", ascending=False)["name"].astype(str)),
                "Constructors": ", ".join(sol.constructors.sort_values("price", ascending=False)["name"].astype(str)),
                "Total cost": round(sol.total_cost, 1),
                "Expected points": round(expected_points, 2),
                "Expected price gain": round(price_change, 2),
                "Projected team value": round(projected_team_value_from_budget(float(budget) if budget is not None else sol.total_cost, price_change), 2),
                "Objective score": round(objective_score, 2),
                "Boosted driver": sol.boosted_driver or "",
            }
        )
    return pd.DataFrame(rows)


def _selected_asset_table(df: pd.DataFrame) -> pd.DataFrame:
    return format_selected_asset_display_table(df)


def _team_colour_badge(name: object) -> str:
    return f"background-color: {team_colour(str(name))}; color: #ffffff; font-weight: 700;"


def _team_table_styler(df: pd.DataFrame):
    df = _styleable_dataframe(df)
    styler = df.style
    if "Team" in df.columns:
        styler = styler.map(_team_colour_badge, subset=["Team"])
    elif "Name" in df.columns:
        styler = styler.map(_team_colour_badge, subset=["Name"])
    styler = styler.format(
        {
            "Price": lambda v: format_money(v),
            "Team cost": lambda v: format_money(v),
            "Remaining": lambda v: format_money(v),
            "Remaining budget": lambda v: format_money(v),
            "Expected / race": lambda v: format_points(v),
            "Expected points": lambda v: format_points(v),
            "Expected points gain": lambda v: format_signed_points(v),
            "Net expected points gain": lambda v: format_signed_points(v),
            "Expected price gain": lambda v: format_signed_money(v),
            "Expected price gain delta": lambda v: format_signed_money(v),
            "Projected price": lambda v: format_money(v),
            "Projected team value": lambda v: format_money(v),
            "Projected team value delta": lambda v: format_signed_money(v),
            "DNF rate": "{:.3f}",
        },
        na_rep="-",
    )
    return styler


def _tier_key_styler(df: pd.DataFrame):
    styler = df.style
    for col, color in [
        ("Terrible", PRICE_BAND_STYLES["Terrible"]),
        ("Poor", PRICE_BAND_STYLES["Poor"]),
        ("Good", PRICE_BAND_STYLES["Good"]),
        ("Great", PRICE_BAND_STYLES["Great"]),
    ]:
        if col in df.columns:
            styler = styler.map(lambda _, c=color: c, subset=[col])
    return styler


def _team_expected_points(sol) -> float:
    chip = CHIP_TRIPLE if sol.triple_driver else CHIP_NONE
    return team_expected_points_with_chips(sol.drivers, sol.constructors, chip, sol.boosted_driver, sol.triple_driver)


def _team_price_change(sol) -> float:
    driver_change = pd.to_numeric(
        sol.drivers.get("expected_price_gain", sol.drivers.get("expected_price_change", pd.Series(dtype=float))),
        errors="coerce",
    ).fillna(0.0).sum()
    constructor_change = pd.to_numeric(
        sol.constructors.get("expected_price_gain", sol.constructors.get("expected_price_change", pd.Series(dtype=float))),
        errors="coerce",
    ).fillna(0.0).sum()
    return float(driver_change + constructor_change)


def _team_summary(sol, budget: float) -> dict:
    remaining_budget = float(budget) - float(sol.total_cost)
    expected_points = _team_expected_points(sol)
    price_change = _team_price_change(sol)
    return {
        "Expected points": expected_points,
        "Expected price gain": price_change,
        "Total cost": float(sol.total_cost),
        "Remaining budget": remaining_budget,
        "Projected team value": projected_team_value_from_budget(float(budget), price_change),
        "Objective score": float(sol.expected_score),
    }


def _team_header(sol, budget: float) -> str:
    remaining_budget = float(budget) - float(sol.total_cost)
    expected_points = _team_expected_points(sol)
    price_change = _team_price_change(sol)
    return (
        f"Cost {format_money(sol.total_cost)} | "
        f"Remaining budget {format_money(remaining_budget)} | "
        f"Points {expected_points:.2f} | "
        f"Expected price gain {format_signed_money(price_change)} | "
        f"Objective {sol.expected_score:.2f} | "
        f"2x {sol.boosted_driver or 'None'}"
    )


def _fmt_signed(value: object, suffix: str = "") -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric):+.2f}{suffix}"


def _transfer_move_card(row: pd.Series, is_top: bool = False) -> None:
    with st.container(border=True):
        move_rows = row.get("Move rows", [])
        if isinstance(move_rows, list) and move_rows:
            for move in move_rows:
                out_asset = move.get("out", {}) or {}
                in_asset = move.get("in", {}) or {}
                out_card = fantasy_asset_card_html(out_asset, asset_label=str(move.get("asset_type", "Asset")).title())
                in_card = fantasy_asset_card_html(in_asset, asset_label=str(move.get("asset_type", "Asset")).title())
                st.markdown(
                    (
                        "<div class='f1-transfer-row'>"
                        f"<div class='f1-transfer-card-slot'>{out_card}</div>"
                        "<div class='f1-transfer-arrow'>→</div>"
                        f"<div class='f1-transfer-card-slot'>{in_card}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(f"**OUT:** {html.escape(str(row.get('OUT') or '-'))}")
            st.markdown(f"**IN:** {html.escape(str(row.get('IN') or '-'))}")

        st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
        metric_cols = st.columns(5)
        metric_cols[0].metric("Transfers used", f"{int(pd.to_numeric(row.get('Transfers'), errors='coerce') or 0)}")
        metric_cols[1].metric("Transfer penalty", format_points(pd.to_numeric(row.get("Transfer penalty"), errors="coerce")))
        metric_cols[2].metric("Δ Expected points", format_signed_points(row.get("Net expected points gain")))
        metric_cols[3].metric("Δ Expected price gain", format_signed_money(row.get("Expected price gain delta")))
        metric_cols[4].metric("Remaining budget", format_money(row.get("Remaining budget")))


def _render_transfer_tradeoff_box(row: pd.Series) -> None:
    points_delta = float(pd.to_numeric(row.get("Expected points gain"), errors="coerce") or 0.0)
    price_delta = float(pd.to_numeric(row.get("Expected price gain delta"), errors="coerce") or 0.0)
    message = str(row.get("Explanation") or "").strip()
    if not message:
        message = "This is a trade-off move with mixed upside."
    if points_delta > 0 and price_delta < 0:
        st.warning(message)
    elif points_delta < 0 and price_delta > 0:
        st.warning(message)
    elif points_delta > 0 and price_delta > 0:
        st.success(message)
    else:
        st.info(message)


def _format_recent_point(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1f}"


def _signed_cell_style(value: object) -> str:
    state = gain_value_class(value)
    if state == "f1-gain-positive":
        return "background-color: rgba(46, 160, 67, 0.22);"
    if state == "f1-gain-negative":
        return "background-color: rgba(248, 81, 73, 0.22);"
    if state == "f1-gain-missing":
        return "color: rgba(148, 163, 184, 0.72);"
    return "background-color: rgba(148, 163, 184, 0.18);"


def _price_change_table_styler(df: pd.DataFrame):
    styler = df.style
    for col, color in PRICE_BAND_STYLES.items():
        if col in df.columns:
            styler = styler.map(lambda _, c=color: c, subset=[col])
        probability_col = f"P({col})"
        if probability_col in df.columns:
            styler = styler.map(lambda _, c=color: c, subset=[probability_col])
    if "Expected price gain" in df.columns:
        styler = styler.map(_signed_cell_style, subset=["Expected price gain"])
    if "Projected price gain" in df.columns:
        styler = styler.map(_signed_cell_style, subset=["Projected price gain"])
    if "Effective price change" in df.columns:
        styler = styler.map(_signed_cell_style, subset=["Effective price change"])
    if "Team" in df.columns:
        styler = styler.map(_team_colour_badge, subset=["Team"])
    elif "Name" in df.columns:
        styler = styler.map(_team_colour_badge, subset=["Name"])
    styler = styler.format(
        {
            "Price": lambda v: format_money(v),
            "Expected Points": "{:.2f}",
            "Predicted next": "{:.2f}",
            "Predicted next / Expected points": "{:.2f}",
            "Projected price": lambda v: format_money(v),
            "Projected price gain": "{:+.2f}",
            "Rise difficulty": "{:.3f}",
            "Projected avgPPM": "{:.3f}",
            "P(price fall)": "{:.1%}",
            "P(Terrible)": "{:.1%}",
            "P(Poor)": "{:.1%}",
            "P(Good)": "{:.1%}",
            "P(Great)": "{:.1%}",
            "Expected price gain": lambda v: format_signed_money(v),
            "Raw expected gain": "{:+.2f}",
            "Expected price gain / million": "{:.3f}",
            "Risk-adjusted price gain": "{:.3f}",
            "Expected points / million": "{:.2f}",
            "Expected points / volatility": "{:.2f}",
            "Volatility": "{:.2f}",
            "Volatility / race": "{:.2f}",
            "DNF rate used": "{:.1%}",
            "DNF score used": "{:.1f}",
            "Raw price change": "{:+.1f}",
            "Effective price change": "{:+.1f}",
            "Race -2": _format_recent_point,
            "Race -1": _format_recent_point,
        },
        na_rep="-",
    )
    return styler


def _price_change_table_html(df: pd.DataFrame) -> str:
    styler = _price_change_table_styler(df)
    styler = styler.hide(axis="index").set_table_attributes(
        'class="f1-compact-table f1-price-change-table"'
    )
    desktop = f'<div class="f1-table-scroll">{styler.to_html()}</div>'
    mobile_columns = {"Asset", "Price", "Good", "Great"}
    if not mobile_columns.issubset(df.columns):
        return desktop
    rows: list[str] = []
    for _, row in df.copy(deep=True).iterrows():
        price = pd.to_numeric(row.get("Price"), errors="coerce")
        price_text = "—" if pd.isna(price) else f"{float(price):.1f}"
        rows.append(
            "<tr>"
            f'<td class="f1-asset-cell">{row.get("Asset", "")}</td>'
            f"<td>{price_text}</td>"
            f"<td>{html.escape(str(row.get('Good', '—')))}</td>"
            f"<td>{html.escape(str(row.get('Great', '—')))}</td>"
            "</tr>"
        )
    mobile = (
        '<div class="f1-table-scroll"><table class="f1-compact-table '
        'f1-mobile-schema f1-threshold-mobile">'
        "<thead><tr><th>Asset</th><th>Price</th><th>Good</th><th>Great</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    return (
        f'<div class="f1-responsive-table f1-desktop-table">{desktop}</div>'
        f'<div class="f1-responsive-table f1-mobile-table">{mobile}</div>'
    )


def _price_change_display_table(
    df: pd.DataFrame,
    rules,
    kind: str,
    expensive_rules=None,
    expensive_price_min: float | None = None,
    bounds=None,
) -> pd.DataFrame:
    table = price_change_target_summary_table(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col="next_race_expected_points",
    )
    if kind == "constructors" and "Team" in table.columns:
        table = table.drop(columns=["Team"])
    return table


def _price_change_projection_table(
    df: pd.DataFrame,
    rules,
    kind: str,
    expensive_rules=None,
    expensive_price_min: float | None = None,
    bounds=None,
) -> pd.DataFrame:
    table = price_change_projection_summary_table(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col="next_race_expected_points",
    )
    if kind == "constructors" and "Team" in table.columns:
        table = table.drop(columns=["Team"])
    return table


def _price_change_probability_matrix(
    df: pd.DataFrame,
    rules,
    kind: str,
    expensive_rules=None,
    expensive_price_min: float | None = None,
    bounds=None,
) -> pd.DataFrame:
    table = price_change_probability_matrix_table(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col="next_race_expected_points",
    )
    if kind == "constructors" and "Team" in table.columns:
        table = table.drop(columns=["Team"])
    return table


def _price_change_probability_detail_table(
    df: pd.DataFrame,
    rules,
    kind: str,
    expensive_rules=None,
    expensive_price_min: float | None = None,
    bounds=None,
) -> pd.DataFrame:
    table = apply_probabilistic_price_change_model(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col="next_race_expected_points",
    )
    abbrev_col = "tla" if "tla" in table.columns else ("driver_reference" if "driver_reference" in table.columns else "id")
    cols = [
        abbrev_col,
        "name",
        "team",
        "price",
        "recent_points_2ago",
        "recent_points_1ago",
        "price_change_predicted_next",
        "volatility_used",
        "projected_avg_ppm",
        "p_terrible",
        "p_poor",
        "p_good",
        "p_great",
        "p_price_rise",
        "p_price_fall",
        "expected_price_gain",
        "expected_price_gain_per_million",
        "risk_adjusted_price_gain",
        "expected_points_per_million",
        "expected_points_per_volatility",
        "projected_price_after_expected_gain",
        "dnf_rate_used",
    ]
    out = table[[col for col in cols if col in table.columns]].copy()
    out.rename(
        columns={
            abbrev_col: "Abbrev",
            "name": "Name",
            "team": "Team",
            "price": "Price",
            "recent_points_2ago": "Race -2",
            "recent_points_1ago": "Race -1",
            "price_change_predicted_next": "Expected Points",
            "volatility_used": "Volatility / race",
            "projected_avg_ppm": "Projected avgPPM",
            "p_terrible": "P(Terrible)",
            "p_poor": "P(Poor)",
            "p_good": "P(Good)",
            "p_great": "P(Great)",
            "p_price_rise": "P(price rise)",
            "p_price_fall": "P(price fall)",
            "expected_price_gain": "Expected price gain",
            "expected_price_gain_per_million": "Expected price gain / million",
            "risk_adjusted_price_gain": "Risk-adjusted price gain",
            "expected_points_per_million": "Expected points / million",
            "expected_points_per_volatility": "Expected points / volatility",
            "projected_price_after_expected_gain": "Projected price",
            "dnf_rate_used": "DNF rate used",
        },
        inplace=True,
    )
    if kind == "constructors" and "Team" in out.columns:
        out = out.drop(columns=["Team"])
    return out.sort_values("Expected price gain", ascending=False, na_position="last") if "Expected price gain" in out.columns else out


st.markdown(
    '<div class="f1-app-header"><h1>F1 Fantasy Optimiser</h1>'
    '<p>Live projections, market movement and transfer planning.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    refresh_live_data_requested = st.button("Refresh live data")
    if refresh_live_data_requested:
        st.cache_data.clear()
    current_season_only = st.toggle(
        "Current season only",
        value=False,
        key="current_season_only",
        help=(
            "Ignore previous-season Fantasy history and calculate projections using "
            "only completed races from the current season."
        ),
    )

    st.caption("Use the tabs below to edit settings and run the optimiser.")

history_mode = (
    HISTORY_MODE_CURRENT_SEASON_ONLY
    if current_season_only
    else HISTORY_MODE_ALL_SUPPORTED
)

if "app_budget" not in st.session_state:
    st.session_state.app_budget = 100.0
if "current_team_budget" not in st.session_state:
    st.session_state.current_team_budget = st.session_state.app_budget
if "optimizer_budget" not in st.session_state:
    st.session_state.optimizer_budget = st.session_state.app_budget
if "budget_user_overridden" not in st.session_state:
    st.session_state.budget_user_overridden = False
if "budget_auto_signature" not in st.session_state:
    st.session_state.budget_auto_signature = None
if "optimizer_budget_source" not in st.session_state:
    st.session_state.optimizer_budget_source = "manual" if st.session_state.budget_user_overridden else "default"
if "current_team_budget_user_overridden" not in st.session_state:
    st.session_state.current_team_budget_user_overridden = False
if "current_team_budget_source" not in st.session_state:
    st.session_state.current_team_budget_source = "default"
if "budget_defaults_initialised" not in st.session_state:
    st.session_state.budget_defaults_initialised = False
if "uploaded_team_last_attempt_hash" not in st.session_state:
    st.session_state.uploaded_team_last_attempt_hash = None
if "uploaded_team_last_success_hash" not in st.session_state:
    st.session_state.uploaded_team_last_success_hash = None
if "uploaded_team_import_status" not in st.session_state:
    st.session_state.uploaded_team_import_status = None
if "uploaded_team_import_error" not in st.session_state:
    st.session_state.uploaded_team_import_error = None
st.session_state.optimizer_objective_mode = OBJECTIVE_COMBINED
st.session_state["optimise_price_gain_weight_slider"] = normalize_price_growth_value(
    st.session_state.get("optimise_price_gain_weight_slider"),
)
legacy_export_layout = st.session_state.get("png_export_format", "Portrait")
normalized_export_layout = (
    "Reddit landscape" if legacy_export_layout == "Reddit landscape" else "Portrait"
)
if "price_efficiency_image_layout" not in st.session_state:
    st.session_state["price_efficiency_image_layout"] = normalized_export_layout
if "optimise_image_layout" not in st.session_state:
    st.session_state["optimise_image_layout"] = normalized_export_layout
if "efficiency_asset_type" not in st.session_state:
    st.session_state["efficiency_asset_type"] = "Drivers"
else:
    st.session_state["efficiency_asset_type"] = resolve_price_efficiency_asset_type(
        st.session_state["efficiency_asset_type"]
    )
if "model_live_session_emphasis" not in st.session_state:
    st.session_state["model_live_session_emphasis"] = 0.0


def _mark_budget_manual_from_current_team():
    st.session_state.current_team_budget_user_overridden = True
    st.session_state.current_team_budget_source = "manual"


def _mark_budget_manual_from_optimizer():
    updates = optimizer_budget_state_updates(
        st.session_state.optimizer_budget,
        source="manual",
    )
    for key, value in updates.items():
        st.session_state[key] = value


def _sync_session_value(source_key: str, target_key: str) -> None:
    """Synchronise a responsive mirror without changing any value semantics."""
    st.session_state[target_key] = st.session_state.get(source_key)


optimise_tab, market_tab, team_tab, settings_tab = st.tabs(
    list(PRIMARY_NAVIGATION_AREAS),
    key="primary_navigation",
    default="Optimise",
)
with market_tab:
    price_changes_tab, price_efficiency_tab = st.tabs(
        ["Projection & thresholds", "Efficiency"],
        key="market_navigation",
    )
with team_tab:
    current_team_tab, transfers_tab = st.tabs(
        ["Current team", "Transfers"],
        key="team_navigation",
    )
with settings_tab:
    locks_tab, model_settings_tab, diagnostics_tab = st.tabs(
        ["Locks", "Model", "Diagnostics"],
        key="settings_navigation",
    )

with model_settings_tab:
    st.markdown('<div class="f1-section-kicker">MODEL SETTINGS</div>', unsafe_allow_html=True)
    st.subheader("Model Settings")
    st.caption("Choose completed races, recency and the relative current/history blend.")
    model_settings_container = st.empty()

previous_snapshot = st.session_state.get("live_data_snapshot")
previous_model_data = st.session_state.get("derived_model_data")
previous_model_signature = st.session_state.get("derived_model_signature")
failed_model_signature = st.session_state.get("failed_derived_model_signature")
failed_model_error = st.session_state.get("failed_derived_model_error")
effective_model_date = datetime.now(UTC).date().isoformat()
historical_market_session = bool(
    previous_snapshot is not None
    and str(previous_snapshot.source_diagnostics.get("live_data_status", "")) == "generated_snapshot"
)
source_load_needed = previous_snapshot is None or refresh_live_data_requested or historical_market_session
load_started = time.time()
load_ui = load_status = load_progress = load_line = None
if source_load_needed:
    load_ui = st.empty()
    with load_ui.container():
        load_status = st.status("Loading live data...", expanded=True)
        load_progress = st.progress(0, text="Loading market feed")
        load_line = st.empty()


def _on_live_load_progress(event: dict) -> None:
    if load_progress is None or load_line is None:
        return
    message = str(event.get("message", "Loading..."))
    stage_name = str(event.get("stage_name", "Loading"))
    progress_raw = event.get("progress")
    if progress_raw is None:
        stage_index = int(event.get("stage_index", 0) or 0)
        stage_total = max(int(event.get("stage_total", 8) or 8), 1)
        progress_raw = stage_index / stage_total
    progress = max(0.0, min(1.0, float(progress_raw)))
    load_progress.progress(int(progress * 100), text=stage_name)
    load_line.caption(f"{stage_name}: {message}")


source_resolution = resolve_live_data_snapshot(
    None if historical_market_session else previous_snapshot,
    refresh_live_data_requested,
    lambda force_refresh: load_live_data_snapshot(
        historical_seasons_back=RAW_LIVE_HISTORY_SEASONS,
        include_playerstats=True,
        force_refresh=force_refresh,
        progress_callback=_on_live_load_progress,
        previous_snapshot=previous_snapshot,
    ),
)
snapshot = source_resolution["snapshot"]
source_error = source_resolution.get("error")
if snapshot is None:
    if load_status is not None:
        load_status.update(label="Data loading failed", state="error", expanded=False)
    if load_ui is not None:
        load_ui.empty()
    st.session_state["last_load_error"] = source_error
    technical_reason = str(source_error or "No validated current-season market source is available.")
    st.error(f"Current live data is unavailable. {technical_reason}")
    try:
        historical_only = load_canonical_scores(DEFAULT_CANONICAL_DATASET_PATH)
    except Exception as historical_exc:
        historical_only = pd.DataFrame()
        technical_reason = f"{technical_reason} Historical data also failed validation: {historical_exc}"
    with price_changes_tab:
        st.subheader("Historical market data")
        if historical_only.empty:
            st.warning("No validated local historical market data is available.")
        else:
            seasons = sorted(
                pd.to_numeric(historical_only["season"], errors="coerce").dropna().astype(int).unique(),
                reverse=True,
            )
            historical_season = st.selectbox(
                "Historical season",
                seasons,
                key="unavailable_live_historical_season",
            )
            season_rows = historical_only[
                pd.to_numeric(historical_only["season"], errors="coerce").eq(historical_season)
            ].copy()
            rounds = sorted(
                pd.to_numeric(season_rows["round"], errors="coerce").dropna().astype(int).unique(),
                reverse=True,
            )
            historical_round = st.selectbox(
                "Historical round",
                rounds,
                key="unavailable_live_historical_round",
            )
            display = season_rows[
                pd.to_numeric(season_rows["round"], errors="coerce").eq(historical_round)
            ][["entity_type", "name", "abbreviation", "fantasy_points_total", "price"]].copy()
            display.rename(
                columns={
                    "entity_type": "Type",
                    "name": "Asset",
                    "abbreviation": "Abbrev",
                    "fantasy_points_total": "Points",
                    "price": "Price",
                },
                inplace=True,
            )
            st.dataframe(display, hide_index=True, use_container_width=True)
            st.caption("Historical scores and recorded prices remain available from the local canonical dataset.")
    with model_settings_tab:
        st.warning("Current-season model controls are disabled until validated market data is available.")
    with diagnostics_tab:
        st.write("Live data status:", "unavailable")
        st.write("Technical reason:", technical_reason)
    st.stop()

live_data_status = str(snapshot.source_diagnostics.get("live_data_status", "fresh"))
live_data_refresh_error = snapshot.source_diagnostics.get("live_data_refresh_error")
market_status = market_runtime_status(snapshot)
live_market_fallback_in_use = not market_status["is_current"]
scoring_fallback_in_use = (
    snapshot.source_diagnostics.get("scoring_data_status") == "retained_last_good"
)

race_catalogue, race_catalogue_source = snapshot_race_catalogue(snapshot)
driver_race_observations, constructor_race_observations = effective_current_race_points(
    snapshot
)
race_name_by_key = {option.key: option.race_name for option in race_catalogue}
stored_efficiency_race_state = reconcile_race_control_state(
    race_catalogue,
    st.session_state.get("efficiency_race_preset", "All"),
    st.session_state.get("efficiency_custom_race_keys", ()),
    st.session_state.get("efficiency_excluded_race_keys", ()),
)
for state_key, value in [
    ("efficiency_race_preset", stored_efficiency_race_state.preset),
    ("efficiency_custom_race_keys", list(stored_efficiency_race_state.custom_keys)),
    ("efficiency_excluded_race_keys", list(stored_efficiency_race_state.excluded_keys)),
]:
    if st.session_state.get(state_key) != value:
        st.session_state[state_key] = value
stored_race_state = reconcile_race_control_state(
    race_catalogue,
    st.session_state.get("model_race_preset", "All"),
    st.session_state.get("model_custom_race_keys", ()),
    st.session_state.get("model_excluded_race_keys", ()),
)
if st.session_state.get("model_race_preset") != stored_race_state.preset:
    st.session_state["model_race_preset"] = stored_race_state.preset
if list(st.session_state.get("model_custom_race_keys", ())) != list(stored_race_state.custom_keys):
    st.session_state["model_custom_race_keys"] = list(stored_race_state.custom_keys)
if list(st.session_state.get("model_excluded_race_keys", ())) != list(stored_race_state.excluded_keys):
    st.session_state["model_excluded_race_keys"] = list(stored_race_state.excluded_keys)

forecast_event_payload = snapshot.source_diagnostics.get("forecast_target_event")
try:
    forecast_event_key = EventKey(
        int(forecast_event_payload["season"]), int(forecast_event_payload["round"])
    )
except (KeyError, TypeError, ValueError):
    forecast_event_key = None
complete_live_session_labels = completed_live_session_labels(
    snapshot.session_states,
    forecast_event=forecast_event_key,
)
live_session_status_text = (
    " + ".join(complete_live_session_labels)
    if complete_live_session_labels
    else "No completed sessions"
)
live_session_emphasis = float(st.session_state["model_live_session_emphasis"])

with model_settings_container.container():
    st.markdown("### Current-season races")
    st.caption(
        f"{len(race_catalogue)} eligible completed races are available from "
        f"{snapshot.current_season}."
    )
    race_preset = st.selectbox(
        "Race window",
        options=["Last 1", "Last 3", "Last 5", "All", "Custom"],
        key="model_race_preset",
        format_func=lambda value: "All completed races" if value == "All" else value,
    )
    if race_preset == "Custom":
        custom_race_keys = st.multiselect(
            "Custom races",
            options=[option.key for option in race_catalogue],
            key="model_custom_race_keys",
            format_func=lambda key: race_option_label(key, race_name_by_key),
        )
    else:
        custom_race_keys = list(st.session_state.get("model_custom_race_keys", ()))

    before_exclusions = reconcile_race_control_state(
        race_catalogue,
        race_preset,
        custom_race_keys,
        st.session_state.get("model_excluded_race_keys", ()),
    )
    if list(st.session_state.get("model_excluded_race_keys", ())) != list(before_exclusions.excluded_keys):
        st.session_state["model_excluded_race_keys"] = list(before_exclusions.excluded_keys)
    excluded_race_keys = st.multiselect(
        "Exclude races",
        options=list(before_exclusions.exclusion_options),
        key="model_excluded_race_keys",
        format_func=lambda key: race_option_label(key, race_name_by_key),
    )
    race_control = reconcile_race_control_state(
        race_catalogue,
        race_preset,
        custom_race_keys,
        excluded_race_keys,
    )
    mode_label = "Current season only" if current_season_only else "All supported seasons"
    st.info(
        f"{mode_label} · {len(race_catalogue)} eligible · "
        f"{len(race_control.selection.included)} selected"
    )
    adjusted_keys = tuple(
        sorted(
            {
                *stored_race_state.removed_custom_keys,
                *stored_race_state.removed_excluded_keys,
                *before_exclusions.removed_excluded_keys,
            }
        )
    )
    if adjusted_keys:
        adjusted = ", ".join(f"R{key.round}" for key in adjusted_keys)
        st.info(f"Unavailable or out-of-window race choices were removed safely: {adjusted}.")
    if not race_control.selection.included:
        st.warning("No completed races are selected. Current official form will be unavailable until at least one race is included.")

    recency_decay = st.slider(
        "Current-season recency decay p",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        key="model_recency_decay",
    )
    st.caption(
        "Latest included race = 1 · previous = p · two included races ago = p² · "
        "p = 0 uses only the latest included race · p = 1 weights included races equally."
    )
    visible_race_weights = recency_weights(race_control.selection, float(recency_decay))
    weight_summary = race_weight_summary(race_control.selection, visible_race_weights)
    st.caption(f"Weights: {weight_summary}" if weight_summary else "Weights: no races selected")

    st.markdown("### Current and historical blend")
    blend_current_col, blend_history_col = st.columns(2)
    with blend_current_col:
        current_season_weight = st.slider(
            "Current-season relative weight",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key="model_current_season_weight",
        )
    with blend_history_col:
        past_season_weight = st.slider(
            "Historical relative weight",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.05,
            key="model_past_season_weight",
        )
    current_percent, historical_percent = effective_blend_percentages(
        current_season_weight,
        past_season_weight,
    )
    if current_season_only:
        st.caption(
            "Current-season mode bypasses 2023–2025 rows; historical-weight settings are "
            "retained for when all-supported mode is restored."
        )
    else:
        st.caption(f"Effective blend: {current_percent}% current season · {historical_percent}% historical")

    with st.expander("Advanced model range", expanded=False):
        historical_seasons_back = st.number_input(
            "Historical seasons back",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            key="model_historical_seasons_back",
        )
        upcoming_race_horizon = st.number_input(
            "Upcoming race horizon",
            min_value=1,
            max_value=5,
            value=5,
            step=1,
            key="model_upcoming_race_horizon",
        )
        st.caption(
            "Relative weights are normalized over available components. If both are zero, available current and historical components share equally."
        )

selected_race_preset = race_control.preset
selected_custom_race_keys = list(race_control.custom_keys) if race_control.preset == "Custom" else None
selected_excluded_race_keys = list(race_control.excluded_keys)

requested_model_signature = model_settings_signature(
    snapshot,
    historical_seasons_back=int(historical_seasons_back),
    horizon_races=int(upcoming_race_horizon),
    current_season_weight=float(current_season_weight),
    past_season_weight=float(past_season_weight),
    recency_decay=float(recency_decay),
    effective_date=effective_model_date,
    selected_race_preset=selected_race_preset,
    custom_race_keys=selected_custom_race_keys,
    excluded_race_keys=selected_excluded_race_keys,
    history_mode=history_mode,
    live_session_emphasis=float(live_session_emphasis),
)
if failed_model_signature is not None and failed_model_signature != requested_model_signature:
    failed_model_signature = None
    failed_model_error = None
    st.session_state["failed_derived_model_signature"] = None
    st.session_state["failed_derived_model_error"] = None
model_data = None


def _derive_requested_model(raw_snapshot):
    return derive_model_data(
        raw_snapshot,
        today=effective_model_date,
        effective_time=datetime.now(UTC),
        historical_seasons_back=int(historical_seasons_back),
        current_season_weight=float(current_season_weight),
        past_season_weight=float(past_season_weight),
        recency_decay=float(recency_decay),
        horizon_races=int(upcoming_race_horizon),
        progress_callback=_on_live_load_progress if source_load_needed else None,
        selected_race_preset=selected_race_preset,
        custom_race_keys=selected_custom_race_keys,
        excluded_race_keys=selected_excluded_race_keys,
        history_mode=history_mode,
        live_session_emphasis=float(live_session_emphasis),
    )


known_derived_resolution = previous_model_data is not None and (
    previous_model_signature == requested_model_signature
    or failed_model_signature == requested_model_signature
)
if source_load_needed or known_derived_resolution:
    derived_resolution = resolve_derived_model_data(
        snapshot,
        previous_model_data,
        previous_model_signature,
        requested_model_signature,
        _derive_requested_model,
        failed_signature=failed_model_signature,
        failed_error=failed_model_error,
    )
else:
    with st.spinner("Recalculating projections..."):
        derived_resolution = resolve_derived_model_data(
            snapshot,
            previous_model_data,
            previous_model_signature,
            requested_model_signature,
            _derive_requested_model,
            failed_signature=failed_model_signature,
            failed_error=failed_model_error,
        )
model_data = derived_resolution["data"]
derivation_status = derived_resolution["status"]
derivation_error = derived_resolution.get("error")
derivation_unavailable = derivation_status in {"failed", "suppressed_failed_signature"}

if derivation_status == "failed":
    LOGGER.error("Model derivation failed: %s", derivation_error)
    st.session_state["failed_derived_model_signature"] = requested_model_signature
    st.session_state["failed_derived_model_error"] = derivation_error
elif derivation_status == "derived":
    st.session_state["failed_derived_model_signature"] = None
    st.session_state["failed_derived_model_error"] = None

if derivation_unavailable:
    if previous_snapshot is not None and model_data is not None:
        snapshot = copy_live_data_snapshot(previous_snapshot)
    else:
        if load_status is not None:
            load_status.update(label="Data loading failed", state="error", expanded=False)
        if load_ui is not None:
            load_ui.empty()
        st.session_state["last_load_error"] = derivation_error
        st.error("Could not load live data. Try Refresh live data, or try again later.")
        st.stop()

if model_data is None:
    st.error("Could not load live data. Try Refresh live data, or try again later.")
    st.stop()

successful_state_accepted = (
    source_resolution["source_load_succeeded"]
    and source_resolution.get("result_accepted", True)
    and not derivation_unavailable
)
if successful_state_accepted:
    st.session_state["live_data_snapshot"] = copy_live_data_snapshot(snapshot)
    st.session_state["derived_model_data"] = copy_model_data(model_data)
    st.session_state["derived_model_signature"] = requested_model_signature
    st.session_state["last_good_model_payload"] = {
        "drivers": model_data.drivers.copy(deep=True),
        "constructors": model_data.constructors.copy(deep=True),
        "trends": model_data.trends.copy(deep=True),
        "diagnostics": dict(model_data.diagnostics),
    }

refresh_attempt_error = source_error
if refresh_live_data_requested and source_resolution["source_load_succeeded"] and derivation_unavailable:
    refresh_attempt_error = f"Fresh data loaded, but projection recalculation failed: {derivation_error}"
refresh_state = refresh_status_transition(
    st.session_state.get("last_refresh_status"),
    st.session_state.get("last_refresh_error"),
    st.session_state.get("last_successful_refresh_identity"),
    refresh_requested=refresh_live_data_requested,
    source_load_attempted=source_resolution["source_load_attempted"],
    source_load_succeeded=source_resolution["source_load_succeeded"],
    result_accepted=successful_state_accepted,
    error=refresh_attempt_error,
    successful_identity=live_data_snapshot_identity(snapshot) if successful_state_accepted else None,
)
st.session_state["last_refresh_status"] = refresh_state["status"]
st.session_state["last_refresh_error"] = refresh_state["error"]
st.session_state["last_successful_refresh_identity"] = refresh_state["successful_identity"]
st.session_state["last_load_error"] = (
    derivation_error
    if derivation_unavailable
    else live_data_refresh_error or refresh_state["error"]
)

if refresh_state["status"] == "failed":
    st.warning(f"Live refresh failed. Previous successful data remains in use. {refresh_state['error']}")
if derivation_unavailable:
    st.warning(
        "Projection recalculation failed for the selected model settings. "
        "The previous successful model remains in use and will not be retried until its inputs change."
    )

drivers = model_data.drivers.copy(deep=True)
constructors = model_data.constructors.copy(deep=True)
driver_price_efficiency = model_data.driver_price_efficiency.copy(deep=True)
constructor_price_efficiency = model_data.constructor_price_efficiency.copy(deep=True)
_trends = model_data.trends.copy(deep=True)
diagnostics = dict(model_data.diagnostics)
fallback_in_use = (
    refresh_state["status"] == "failed"
    or derivation_unavailable
    or not source_resolution.get("result_accepted", True)
    or live_market_fallback_in_use
    or scoring_fallback_in_use
)
diagnostics["last_load_error"] = st.session_state.get("last_load_error")
diagnostics["last_load_fallback_used"] = bool(fallback_in_use)
diagnostics["live_data_resolution_status"] = source_resolution["status"]
diagnostics["derived_model_resolution_status"] = derivation_status
diagnostics["effective_model_date"] = effective_model_date
diagnostics["latest_live_weekend_diagnostics"] = source_resolution.get("live_diagnostics")
active_model_data_version = model_data_version(snapshot, requested_model_signature)

if load_status is not None and load_progress is not None:
    elapsed = time.time() - load_started
    current_load_attempt_failed = (
        not source_resolution["source_load_succeeded"]
        or not source_resolution.get("result_accepted", True)
        or derivation_status == "failed"
    )
    if current_load_attempt_failed:
        load_progress.progress(100, text="Using last loaded data")
        load_status.update(label="Live refresh failed. Using last loaded data.", state="error", expanded=False)
    elif scoring_fallback_in_use:
        load_progress.progress(100, text="Market refreshed; previous scoring retained")
        load_status.update(
            label="Market refreshed; previous verified scoring data retained",
            state="complete",
            expanded=False,
        )
    elif live_market_fallback_in_use:
        load_progress.progress(100, text="Using validated fallback data")
        label = (
            "Using latest verified cached official data"
            if live_data_status == "cached"
            else "Using generated canonical official snapshot"
        )
        load_status.update(label=label, state="complete", expanded=False)
    else:
        load_progress.progress(100, text=f"Ready in {elapsed:.1f}s")
        load_status.update(label=f"Live data loaded in {elapsed:.1f}s", state="complete", expanded=False)
if load_ui is not None:
    load_ui.empty()

if drivers is None or constructors is None or diagnostics is None:
    st.error("Could not load live data. Try Refresh live data, or try again later.")
    st.stop()

driver_labels = _option_labels(drivers)
constructor_labels = _option_labels(constructors)
holding_market_drivers = build_holding_asset_universe(
    drivers,
    snapshot.player_assets,
    "driver",
)
holding_market_constructors = build_holding_asset_universe(
    constructors,
    snapshot.constructor_assets,
    "constructor",
)
valid_constraint_ids = {
    "locked_driver_ids": set(driver_labels),
    "excluded_driver_ids": set(driver_labels),
    "locked_constructor_ids": set(constructor_labels),
    "excluded_constructor_ids": set(constructor_labels),
}
for constraint_key, valid_ids in valid_constraint_ids.items():
    st.session_state[constraint_key] = [
        str(value)
        for value in st.session_state.get(constraint_key, ())
        if str(value) in valid_ids
    ]
st.session_state["locked_driver_ids"], st.session_state["excluded_driver_ids"] = (
    reconcile_constraint_pair(
        st.session_state["locked_driver_ids"],
        st.session_state["excluded_driver_ids"],
    )
)
st.session_state["locked_constructor_ids"], st.session_state["excluded_constructor_ids"] = (
    reconcile_constraint_pair(
        st.session_state["locked_constructor_ids"],
        st.session_state["excluded_constructor_ids"],
    )
)
current_team_config = _load_current_team_config()
if "chip_mode_label" not in st.session_state:
    st.session_state.chip_mode_label = "None"
chip_mode = chip_mode_from_label(st.session_state.chip_mode_label)

preloaded_driver_ids = [str(x) for x in current_team_config.get("drivers", [])]
preloaded_constructor_ids = [str(x) for x in current_team_config.get("constructors", [])]
preloaded_bank = float(current_team_config.get("bank", 0.0))
if "current_team_driver_ids" not in st.session_state:
    st.session_state.current_team_driver_ids = preloaded_driver_ids
if "current_team_constructor_ids" not in st.session_state:
    st.session_state.current_team_constructor_ids = preloaded_constructor_ids
if "current_team_free_transfers" not in st.session_state:
    st.session_state.current_team_free_transfers = int(current_team_config.get("free_transfers", 2))
if "current_team_bank" not in st.session_state:
    st.session_state.current_team_bank = preloaded_bank

if not st.session_state.budget_defaults_initialised:
    preload_driver_frame = holding_market_drivers[
        holding_market_drivers["id"].astype(str).isin(st.session_state.current_team_driver_ids)
    ]
    preload_constructor_frame = holding_market_constructors[
        holding_market_constructors["id"].astype(str).isin(
            st.session_state.current_team_constructor_ids
        )
    ]
    preload_budget = current_team_budget_from_selection(
        preload_driver_frame,
        preload_constructor_frame,
        bank=float(st.session_state.current_team_bank),
    )
    if preload_budget <= 0 and (len(st.session_state.current_team_driver_ids) + len(st.session_state.current_team_constructor_ids)) > 0:
        preload_budget = max(float(st.session_state.get("app_budget", 100.0) or 100.0), 100.0)
    if preload_budget > 0:
        if not st.session_state.current_team_budget_user_overridden:
            st.session_state.current_team_budget = float(preload_budget)
            st.session_state.current_team_budget_source = "default"
        if st.session_state.optimizer_budget_source == "default" and not st.session_state.budget_user_overridden:
            st.session_state.optimizer_budget = float(preload_budget)
        st.session_state.app_budget = float(preload_budget)
        st.session_state.budget_init_team_cost = float(
            pd.to_numeric(preload_driver_frame.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
            + pd.to_numeric(preload_constructor_frame.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        )
        st.session_state.budget_init_bank = float(st.session_state.current_team_bank)
        st.session_state.budget_init_mode = "auto_from_current_team"
        st.session_state.budget_auto_signature = current_team_selection_signature(
            st.session_state.current_team_driver_ids,
            st.session_state.current_team_constructor_ids,
            float(st.session_state.current_team_bank),
        )
    st.session_state.budget_defaults_initialised = True

_render_race_header(diagnostics)
market_signature = market_status["content_signature"]
session_domain_status = (
    "retained"
    if snapshot.source_diagnostics.get("live_session_last_good_retained")
    else "failed"
    if any(
        str(values.get("status")) == "failed"
        for values in snapshot.source_diagnostics.get("live_session_states", {}).values()
    )
    else "current"
)
deadline_domain_status = str(
    snapshot.source_diagnostics.get("deadline_data_status", "unavailable")
).replace("_", " ")
playerstats_domain_status = str(
    snapshot.source_diagnostics.get("playerstats_data_status", "unavailable")
).replace("_", " ")
st.caption(
    " · ".join(
        [
            f"Market: {market_status['state']} "
            f"(feed {market_status['feed_round']}, "
            f"{market_signature[:10] or 'no signature'})",
            f"Scoring: {'previous verified' if scoring_fallback_in_use else 'current'}",
            f"Live sessions: {session_domain_status}",
            f"Playerstats: {playerstats_domain_status}",
            f"Deadline: {deadline_domain_status}",
        ]
    )
)
if market_status["show_stale_warning"]:
    cached_feed = market_status["feed_round"]
    verified_at = market_status["verified_at_utc"]
    detail = f" (feed {cached_feed}" + (f", verified {verified_at}" if verified_at else "") + ")"
    st.warning(
        "Live data could not be refreshed. Showing the latest verified cached data"
        f"{detail}. Current prices and availability may be stale. Use Refresh live data to try again."
    )
active_weekend_warnings = []
if source_resolution.get("status") == "market_refreshed_scoring_retained":
    active_weekend_warnings.append(
        "The market refreshed successfully, but scoring data did not fully validate. "
        "Previous verified scoring inputs remain in use; current prices are not rolled back."
    )
elif diagnostics.get("weekend_status") == "awaiting_final_classification":
    active_weekend_warnings.append(
        "Awaiting final race classification; the live weekend is excluded from completed-race form."
    )
elif diagnostics.get("completed_form_excludes_live_weekend"):
    st.info(
        "The active weekend is pending or live. Optimiser projections use previously completed weekends only."
    )
if diagnostics.get("team_lock_deadline_warning"):
    active_weekend_warnings.append(
        "The official fantasy deadline could not be validated; the active-event schedule fallback is in use."
    )
if active_weekend_warnings:
    st.warning(" ".join(active_weekend_warnings))

historical_coverage_warnings = []
if diagnostics.get("missing_requested_seasons"):
    missing_seasons = ", ".join(str(year) for year in diagnostics["missing_requested_seasons"])
    used_seasons = ", ".join(str(year) for year in diagnostics.get("used_seasons", [])) or "none"
    historical_coverage_warnings.append(
        f"Season source data was unavailable for: {missing_seasons}. "
        f"The model used only these loaded seasons: {used_seasons}."
    )
if diagnostics.get("playerstats_timeout_failures", 0):
    historical_coverage_warnings.append(
        f"Playerstats timeouts detected: {int(diagnostics.get('playerstats_timeout_failures', 0))}. "
        "Recorded canonical history remains available where present."
    )
if diagnostics.get("playerstats_skipped_after_failure_limit", 0):
    historical_coverage_warnings.append(
        f"Playerstats requests skipped after repeated failures: "
        f"{int(diagnostics.get('playerstats_skipped_after_failure_limit', 0))}."
    )
if historical_coverage_warnings:
    st.warning(" ".join(historical_coverage_warnings))

edited_drivers = drivers.copy()
edited_constructors = constructors.copy()

with model_settings_tab:
    with st.expander("Advanced: Asset assumptions", expanded=False):
        st.caption("Optional manual overrides for prices, expected points, DNF rate and volatility.")
        driver_tab, constructor_tab = st.tabs(["Drivers", "Constructors"])
        with driver_tab:
            edited_drivers = _asset_editor(drivers, "drivers")
        with constructor_tab:
            edited_constructors = _asset_editor(constructors, "constructors")

price_change_bounds = DEFAULT_PRICE_CHANGE_BOUNDS
driver_expensive_min = DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF
constructor_expensive_min = DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF
cheap_driver_rules = DEFAULT_PRICE_CHANGE_CHEAP_RULES
cheap_constructor_rules = DEFAULT_PRICE_CHANGE_CHEAP_RULES
expensive_driver_rules = DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES
expensive_constructor_rules = DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES
price_change_drivers = edited_drivers.copy()
price_change_constructors = edited_constructors.copy()

with current_team_tab:
    st.markdown('<div class="f1-section-kicker">SQUAD BUILDER</div>', unsafe_allow_html=True)
    st.subheader("Current Team Builder")
    st.caption("Enter your current drivers, constructors, bank and free transfers to analyse your team.")
    current_team_input_drivers = apply_no_negative_scores(edited_drivers) if chip_mode == CHIP_NO_NEGATIVE else edited_drivers
    current_team_input_constructors = apply_no_negative_scores(edited_constructors) if chip_mode == CHIP_NO_NEGATIVE else edited_constructors
    selectable_current_team_drivers = apply_probabilistic_price_change_model(
        current_team_input_drivers,
        cheap_driver_rules,
        expensive_rules=expensive_driver_rules,
        expensive_price_min=driver_expensive_min,
        bounds=price_change_bounds,
        predicted_points_col="next_race_expected_points",
    )
    selectable_current_team_constructors = apply_probabilistic_price_change_model(
        current_team_input_constructors,
        cheap_constructor_rules,
        expensive_rules=expensive_constructor_rules,
        expensive_price_min=constructor_expensive_min,
        bounds=price_change_bounds,
        predicted_points_col="next_race_expected_points",
    )
    current_team_drivers = build_holding_asset_universe(
        selectable_current_team_drivers,
        snapshot.player_assets,
        "driver",
    )
    current_team_constructors = build_holding_asset_universe(
        selectable_current_team_constructors,
        snapshot.constructor_assets,
        "constructor",
    )

    st.caption("Build a current_team.json-style squad with a current-team budget that is independent from the optimiser budget.")
    with st.expander("Advanced: JSON import / export", expanded=False):
        st.write(
            "Use this to save or restore your F1 Fantasy squad. Importing a current_team.json file fills the selected drivers, constructors, bank, free transfers, and current-team budget. Its team value is offered separately as an optimiser-budget suggestion."
        )
        uploaded_team_file = st.file_uploader("Upload current_team.json", type=["json"], key="current_team_upload_file")
        json_tools_container = st.container()
    if uploaded_team_file is not None:
        uploaded_contents = uploaded_team_file.getvalue()
        upload_transition = current_team_upload_transition(
            uploaded_contents,
            st.session_state.uploaded_team_last_attempt_hash,
            current_team_drivers,
            current_team_constructors,
        )
        if upload_transition["attempted"]:
            st.session_state.uploaded_team_last_attempt_hash = upload_transition["upload_hash"]
            st.session_state.uploaded_team_import_status = upload_transition["status"]
            st.session_state.uploaded_team_import_error = upload_transition["error"]
            if upload_transition["status"] == "success":
                for key, value in upload_transition["state_updates"].items():
                    st.session_state[key] = value
                st.session_state.uploaded_team_last_success_hash = upload_transition["upload_hash"]
                st.session_state.imported_budget_suggestion_data_identity = None
        if st.session_state.uploaded_team_import_status == "success":
            st.success("Loaded current_team.json.")
        elif st.session_state.uploaded_team_import_status == "error":
            st.error(f"Could not import current_team.json: {st.session_state.uploaded_team_import_error}")

    imported_driver_ids = st.session_state.get("imported_budget_driver_ids")
    imported_constructor_ids = st.session_state.get("imported_budget_constructor_ids")
    imported_bank = st.session_state.get("imported_budget_bank")
    current_price_identity = live_data_snapshot_identity(snapshot)
    if (
        imported_driver_ids is not None
        and imported_constructor_ids is not None
        and imported_bank is not None
        and st.session_state.get("imported_budget_suggestion_data_identity") != current_price_identity
    ):
        suggestion_update = reconcile_imported_budget_suggestion(
            imported_driver_ids,
            imported_constructor_ids,
            float(imported_bank),
            current_team_drivers,
            current_team_constructors,
        )
        st.session_state.imported_budget_suggestion = suggestion_update["suggestion"]
        st.session_state.imported_budget_suggestion_status = suggestion_update["status"]
        st.session_state.imported_budget_missing_driver_ids = suggestion_update["missing_driver_ids"]
        st.session_state.imported_budget_missing_constructor_ids = suggestion_update["missing_constructor_ids"]
        st.session_state.imported_budget_missing_price_driver_ids = suggestion_update.get(
            "missing_price_driver_ids", []
        )
        st.session_state.imported_budget_missing_price_constructor_ids = suggestion_update.get(
            "missing_price_constructor_ids", []
        )
        st.session_state.imported_budget_suggestion_data_identity = current_price_identity

    imported_budget_suggestion = st.session_state.get("imported_budget_suggestion")
    imported_budget_status = st.session_state.get("imported_budget_suggestion_status")
    if imported_budget_status == "available" and imported_budget_suggestion is not None:
        st.info(
            f"Imported team value: {format_money(imported_budget_suggestion)}. "
            f"Your optimiser budget remains {format_money(st.session_state.optimizer_budget)} unless you accept this suggestion."
        )
        if st.button("Use imported team value as optimiser budget", key="accept_imported_optimizer_budget"):
            updates = optimizer_budget_state_updates(
                imported_budget_suggestion,
                source="imported_accepted",
                accepted_import_hash=st.session_state.uploaded_team_last_success_hash,
            )
            for key, value in updates.items():
                st.session_state[key] = value
            st.success(f"Optimiser budget set to {format_money(imported_budget_suggestion)}.")
    elif imported_budget_status == "incomplete":
        missing_imported_assets = (
            list(st.session_state.get("imported_budget_missing_driver_ids", []))
            + list(st.session_state.get("imported_budget_missing_constructor_ids", []))
            + list(st.session_state.get("imported_budget_missing_price_driver_ids", []))
            + list(st.session_state.get("imported_budget_missing_price_constructor_ids", []))
        )
        st.warning(
            "The imported team-value suggestion is unavailable because refreshed roster data is missing "
            f"these imported assets: {missing_imported_assets}. The optimiser budget was not changed."
        )

    selected_driver_frame = current_team_drivers[current_team_drivers["id"].astype(str).isin(st.session_state.current_team_driver_ids)]
    selected_constructor_frame = current_team_constructors[current_team_constructors["id"].astype(str).isin(st.session_state.current_team_constructor_ids)]
    selection_signature = current_team_selection_signature(
        st.session_state.current_team_driver_ids,
        st.session_state.current_team_constructor_ids,
        float(st.session_state.current_team_bank),
    )
    auto_budget_target = auto_budget_from_team_cost(
        float(pd.to_numeric(selected_driver_frame.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        + float(pd.to_numeric(selected_constructor_frame.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
        float(st.session_state.current_team_bank),
    )
    resolved_budget = resolve_budget_value(
        st.session_state.get("current_team_budget"),
        team_cost=float(auto_budget_target - float(st.session_state.current_team_bank)),
        bank=float(st.session_state.current_team_bank),
        user_overridden=bool(st.session_state.current_team_budget_user_overridden),
    )
    if (
        not st.session_state.current_team_budget_user_overridden
        and (
            st.session_state.budget_auto_signature != selection_signature
            or st.session_state.get("current_team_budget") != resolved_budget
        )
    ):
        st.session_state.current_team_budget = resolved_budget
        st.session_state.current_team_budget_source = "auto_from_selected_team"
        st.session_state.budget_init_team_cost = float(auto_budget_target - float(st.session_state.current_team_bank))
        st.session_state.budget_init_bank = float(st.session_state.current_team_bank)
        st.session_state.budget_auto_signature = selection_signature

    current_payload = current_team_json(
        st.session_state.current_team_driver_ids,
        st.session_state.current_team_constructor_ids,
        free_transfers=st.session_state.current_team_free_transfers,
        bank=st.session_state.current_team_bank,
    )
    with json_tools_container:
        if st.checkbox("Preview JSON", value=False, key="current_team_json_preview"):
            st.code(json.dumps(current_payload, indent=2), language="json")
        st.download_button(
            "Download current_team.json",
            data=json.dumps(current_payload, indent=2),
            file_name="current_team.json",
            mime="application/json",
        )

    with st.container(border=True):
        budget_col, bank_col, transfers_col = st.columns(3)
        with budget_col:
            current_budget = st.number_input(
                "Budget",
                min_value=0.0,
                step=0.1,
                key="current_team_budget",
                on_change=_mark_budget_manual_from_current_team,
            )
        with bank_col:
            bank = st.number_input(
                "Bank",
                min_value=0.0,
                step=0.1,
                key="current_team_bank",
            )
        with transfers_col:
            free_transfers = st.number_input(
                "Free transfers",
                min_value=0,
                max_value=10,
                step=1,
                key="current_team_free_transfers",
            )
        selector_col1, selector_col2 = st.columns(2)
        current_driver_labels = current_team_option_labels(
            current_team_drivers,
            st.session_state.current_team_driver_ids,
            "driver",
        )
        current_constructor_labels = current_team_option_labels(
            current_team_constructors,
            st.session_state.current_team_constructor_ids,
            "constructor",
        )
        with selector_col1:
            selected_current_driver_ids = st.multiselect(
                "Current drivers",
                options=list(current_driver_labels.keys()),
                format_func=current_driver_labels.get,
                key="current_team_driver_ids",
            )
        with selector_col2:
            selected_current_constructor_ids = st.multiselect(
                "Current constructors",
                options=list(current_constructor_labels.keys()),
                format_func=current_constructor_labels.get,
                key="current_team_constructor_ids",
            )
        effective_current_budget = float(current_budget)
    current_validation = validate_current_team(
        selected_current_driver_ids,
        selected_current_constructor_ids,
        current_team_drivers,
        current_team_constructors,
        budget=effective_current_budget,
    )
    current_boosted_driver, current_triple_driver = select_chip_boost_drivers(
        current_validation["selected_drivers"],
        chip_mode,
    )
    current_expected_points = team_expected_points_with_chips(
        current_validation["selected_drivers"],
        current_validation["selected_constructors"],
        chip_mode,
        current_boosted_driver,
        current_triple_driver,
    )
    expected_current_price_gain = selected_assets_price_gain(
        current_validation["selected_drivers"],
        current_validation["selected_constructors"],
    )
    current_summary = team_summary_payload(
        total_cost=current_validation["total_cost"],
        budget=effective_current_budget,
        expected_gain=expected_current_price_gain,
        expected_points=current_expected_points,
        limitless=chip_mode == CHIP_LIMITLESS,
    )
    st.markdown(team_summary_html(current_summary), unsafe_allow_html=True)

    for msg in current_validation["errors"]:
        st.error(msg)
    for msg in current_validation["warnings"]:
        st.warning(msg)
    if current_validation["valid"]:
        st.success("Current team shape is valid.")

    _asset_card_grid(
        current_validation["selected_drivers"],
        boosted_driver=current_boosted_driver,
        triple_driver=current_triple_driver,
        asset_label="Driver",
    )
    _asset_card_grid(current_validation["selected_constructors"], asset_label="Constructor")

with locks_tab:
    st.markdown('<div class="f1-section-kicker">SELECTION RULES</div>', unsafe_allow_html=True)
    st.subheader("Locks and Exclusions")
    st.caption(
        "Lock assets you want to keep and exclude assets you do not want the optimiser or transfer tool to use."
    )
    st.caption(
        "Locks force an asset into optimiser and transfer results. Exclusions prevent an asset from being selected or recommended."
    )
    col1, col2 = st.columns(2)
    with col1:
        locked_driver_ids = st.multiselect(
            "Locked drivers",
            options=list(driver_labels.keys()),
            format_func=driver_labels.get,
            key="locked_driver_ids",
            on_change=_on_bulk_constraint_change,
            args=("locked_driver_ids", "excluded_driver_ids"),
        )
        locked_constructor_ids = st.multiselect(
            "Locked constructors",
            options=list(constructor_labels.keys()),
            format_func=constructor_labels.get,
            key="locked_constructor_ids",
            on_change=_on_bulk_constraint_change,
            args=("locked_constructor_ids", "excluded_constructor_ids"),
        )
    with col2:
        excluded_driver_ids = st.multiselect(
            "Excluded drivers",
            options=list(driver_labels.keys()),
            format_func=driver_labels.get,
            key="excluded_driver_ids",
            on_change=_on_bulk_constraint_change,
            args=("excluded_driver_ids", "locked_driver_ids"),
        )
        excluded_constructor_ids = st.multiselect(
            "Excluded constructors",
            options=list(constructor_labels.keys()),
            format_func=constructor_labels.get,
            key="excluded_constructor_ids",
            on_change=_on_bulk_constraint_change,
            args=("excluded_constructor_ids", "locked_constructor_ids"),
        )
    summary_cols = st.columns(4)
    summary_cols[0].metric("Locked drivers", str(len(locked_driver_ids)))
    summary_cols[1].metric("Locked constructors", str(len(locked_constructor_ids)))
    summary_cols[2].metric("Excluded drivers", str(len(excluded_driver_ids)))
    summary_cols[3].metric("Excluded constructors", str(len(excluded_constructor_ids)))

with transfers_tab:
    st.markdown('<div class="f1-section-kicker">TRANSFER DESK</div>', unsafe_allow_html=True)
    st.subheader("Transfer Recommendations")
    st.caption(
        "Compare your current squad with transfer options ranked by points, budget growth or a combined objective."
    )
    if chip_mode == CHIP_LIMITLESS:
        st.info("Limitless is a temporary team chip, so permanent transfer recommendations are disabled.")
    elif not current_validation["valid"]:
        st.warning("Build a valid current team first before asking for transfer recommendations.")
    else:
        if "transfer_results" not in st.session_state:
            st.session_state.transfer_results = None
        if "transfer_run_signature" not in st.session_state:
            st.session_state.transfer_run_signature = None
        if "transfer_run_diagnostics" not in st.session_state:
            st.session_state.transfer_run_diagnostics = {}

        active_locks = len(locked_driver_ids) + len(locked_constructor_ids)
        active_exclusions = len(excluded_driver_ids) + len(excluded_constructor_ids)
        if active_locks or active_exclusions:
            st.caption(f"Using {active_locks} locked assets and {active_exclusions} excluded assets from Locks and exclusions.")

        transfer_col1, transfer_col2, transfer_col3, transfer_col4 = st.columns(4)
        with transfer_col1:
            max_transfers = st.number_input("Max transfers to consider", min_value=1, max_value=4, value=2, step=1)
        with transfer_col2:
            transfer_options = st.selectbox("Number of transfer options", options=[3, 5, 10, 20], index=1)
        with transfer_col3:
            transfer_penalty = st.number_input("Penalty per extra transfer", min_value=0.0, value=10.0, step=1.0)
        with transfer_col4:
            transfer_search_mode = st.selectbox(
                "Search mode",
                options=["Fast", "Balanced", "Exhaustive"],
                index=1,
                help="Balanced prunes unlikely moves before scoring. Exhaustive checks everything but can be slow.",
            )
        transfer_objective_col, transfer_weight_col = st.columns(2)
        with transfer_objective_col:
            transfer_objective = st.selectbox(
                "Recommendation objective",
                options=[OBJECTIVE_POINTS_ONLY, OBJECTIVE_PRICE_GROWTH_ONLY, OBJECTIVE_COMBINED, OBJECTIVE_RISK_ADJUSTED_COMBINED],
                index=2,
            )
        with transfer_weight_col:
            transfer_price_gain_weight = st.slider(
                "Price-growth value",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                disabled=transfer_objective not in {OBJECTIVE_COMBINED, OBJECTIVE_RISK_ADJUSTED_COMBINED},
                key="transfer_price_gain_weight_slider",
                help="Objective points assigned per +1.0M expected gain.",
            )
        st.caption("Combined transfer objective = expected points + λ × expected price gain.")

        baseline = transfer_baseline(
            selected_current_driver_ids,
            selected_current_constructor_ids,
            current_team_drivers,
            current_team_constructors,
            float(current_budget),
            chip_mode=chip_mode,
        )
        st.markdown("### Do nothing baseline")
        base_cols = st.columns(5)
        base_cols[0].metric("Team cost", format_money(baseline["team_cost"]))
        base_cols[1].metric("Remaining budget", format_money(baseline["remaining_budget"]))
        base_cols[2].metric("Expected points", format_points(baseline["expected_points"]))
        base_cols[3].metric("Expected price gain", format_signed_money(baseline["expected_price_gain"]))
        base_cols[4].metric("Projected team value", format_money(baseline["projected_team_value"]))
        st.markdown("<div style='height:0.45rem;'></div>", unsafe_allow_html=True)

        transfer_inputs = (
            tuple(sorted(str(x) for x in selected_current_driver_ids)),
            tuple(sorted(str(x) for x in selected_current_constructor_ids)),
            float(current_budget),
            float(free_transfers),
            int(max_transfers),
            int(transfer_options),
            float(transfer_penalty),
            str(transfer_objective),
            float(transfer_price_gain_weight),
            str(chip_mode),
            tuple(sorted(str(x) for x in locked_driver_ids)),
            tuple(sorted(str(x) for x in excluded_driver_ids)),
            tuple(sorted(str(x) for x in locked_constructor_ids)),
            tuple(sorted(str(x) for x in excluded_constructor_ids)),
            str(transfer_search_mode),
        )
        transfer_signature = build_transfer_result_signature(active_model_data_version, transfer_inputs)
        run_transfer_clicked = st.button("Run transfer recommendations", type="primary", use_container_width=True)
        signature_changed = st.session_state.transfer_run_signature != transfer_signature

        if signature_changed and st.session_state.transfer_results is not None and not run_transfer_clicked:
            st.info("Settings changed. Run transfer recommendations again.")
            st.session_state.transfer_results = None

        if run_transfer_clicked:
            transfer_progress_ui = st.empty()
            transfer_started = time.time()
            transfer_stage_order: list[str] = []
            transfer_meta: dict[str, object] = {}
            with transfer_progress_ui.container():
                transfer_status_line = st.empty()
                transfer_status_line.markdown("Running transfer recommendations...")
                transfer_progress = st.progress(0)

            def _on_transfer_progress(event: dict) -> None:
                stage = str(event.get("stage", "running"))
                message = str(event.get("message", "Running transfer recommendations..."))
                stage_progress_map = {
                    "read_current_team": 0.08,
                    "apply_locks_exclusions": 0.12,
                    "filter_candidates": 0.18,
                    "generate_candidates": 0.25,
                    "score_candidates": 0.75,
                    "rank_recommendations": 0.92,
                    "ready": 1.0,
                }
                progress = event.get("progress")
                if progress is None:
                    progress = stage_progress_map.get(stage, 0.5)
                progress = max(0.0, min(1.0, float(progress)))
                transfer_status_line.markdown(message)
                transfer_progress.progress(int(progress * 100))
                if not transfer_stage_order or transfer_stage_order[-1] != stage:
                    transfer_stage_order.append(stage)
                for key in [
                    "transfer_candidate_count",
                    "transfer_candidate_count_total",
                    "transfer_candidates_evaluated",
                    "transfer_candidates_scored",
                    "transfer_candidates_filtered",
                    "transfer_results_count",
                    "search_mode",
                    "candidate_pool_mode",
                    "incoming_driver_candidates",
                    "incoming_constructor_candidates",
                    "incoming_driver_candidates_kept",
                    "incoming_constructor_candidates_kept",
                    "outgoing_driver_candidates",
                    "outgoing_constructor_candidates",
                    "outgoing_driver_candidates_kept",
                    "outgoing_constructor_candidates_kept",
                    "exhaustive_candidate_count_before_filtering",
                    "candidate_count_after_filtering",
                    "valid_transfer_plans_generated",
                    "candidate_teams_scored",
                    "generated_partial_plans",
                    "evaluated_full_candidates",
                    "number_candidates_generated",
                    "number_pruned_by_filtering",
                    "duplicate_teams_skipped",
                    "pruned_by_budget",
                    "pruned_by_beam",
                    "candidate_filter_score_used_for_prefilter",
                    "exhaustive_scoring_used_after_prefilter",
                    "final_score_used_for_sorting",
                    "transfer_generation_duration_seconds",
                    "transfer_scoring_duration_seconds",
                    "transfer_total_duration_seconds",
                    "generated_candidates_by_depth",
                    "beam_kept_by_depth",
                    "fully_scored_by_depth",
                    "final_recommendations_by_transfer_count",
                ]:
                    if key in event:
                        transfer_meta[key] = event[key]

            transfer_drivers = (
                apply_no_negative_scores(selectable_current_team_drivers)
                if chip_mode == CHIP_NO_NEGATIVE
                else selectable_current_team_drivers
            )
            transfer_constructors = (
                apply_no_negative_scores(selectable_current_team_constructors)
                if chip_mode == CHIP_NO_NEGATIVE
                else selectable_current_team_constructors
            )
            transfer_holding_drivers = (
                apply_no_negative_scores(current_team_drivers)
                if chip_mode == CHIP_NO_NEGATIVE
                else current_team_drivers
            )
            transfer_holding_constructors = (
                apply_no_negative_scores(current_team_constructors)
                if chip_mode == CHIP_NO_NEGATIVE
                else current_team_constructors
            )
            try:
                recs = build_transfer_recommendations(
                    selected_current_driver_ids,
                    selected_current_constructor_ids,
                    transfer_drivers,
                    transfer_constructors,
                    budget=float(current_budget),
                    free_transfers=int(free_transfers),
                    max_transfers=int(max_transfers),
                    allow_extra_transfers=True,
                    transfer_penalty=float(transfer_penalty),
                    objective_mode=transfer_objective,
                    price_gain_weight=float(transfer_price_gain_weight),
                    locked_driver_ids=locked_driver_ids,
                    excluded_driver_ids=excluded_driver_ids,
                    locked_constructor_ids=locked_constructor_ids,
                    excluded_constructor_ids=excluded_constructor_ids,
                    chip_mode=chip_mode,
                    search_mode=str(transfer_search_mode).lower(),
                    top_n=int(transfer_options),
                    progress_callback=_on_transfer_progress,
                    holding_drivers=transfer_holding_drivers,
                    holding_constructors=transfer_holding_constructors,
                )
            except Exception as exc:
                LOGGER.exception("Transfer recommendation run failed")
                transfer_status_line.markdown("Transfer recommendation run failed.")
                st.warning("Transfer run failed. Keeping last successful recommendations.")
                st.session_state["transfer_last_error"] = str(exc)
            else:
                st.session_state.transfer_results = recs
                st.session_state.transfer_run_signature = transfer_signature
                st.session_state["transfer_last_error"] = None
                transfer_status_line.markdown("Transfer recommendations ready.")
            finally:
                elapsed = max(0.0, time.time() - transfer_started)
                st.session_state.transfer_run_diagnostics = {
                    "transfer_run_duration_seconds": float(elapsed),
                    "transfer_total_duration_seconds": float(transfer_meta.get("transfer_total_duration_seconds", elapsed) or elapsed),
                    "transfer_generation_duration_seconds": float(transfer_meta.get("transfer_generation_duration_seconds", 0.0) or 0.0),
                    "transfer_scoring_duration_seconds": float(transfer_meta.get("transfer_scoring_duration_seconds", 0.0) or 0.0),
                    "transfer_stage_order": transfer_stage_order,
                    "transfer_candidate_count_total": int(
                        transfer_meta.get(
                            "transfer_candidate_count_total",
                            transfer_meta.get("transfer_candidate_count", 0),
                        )
                        or 0
                    ),
                    "transfer_candidate_count": int(
                        transfer_meta.get(
                            "transfer_candidate_count_total",
                            transfer_meta.get("transfer_candidate_count", 0),
                        )
                        or 0
                    ),
                    "transfer_candidates_evaluated": int(transfer_meta.get("transfer_candidates_evaluated", 0) or 0),
                    "transfer_candidates_scored": int(transfer_meta.get("transfer_candidates_scored", 0) or 0),
                    "transfer_candidates_filtered": int(transfer_meta.get("transfer_candidates_filtered", 0) or 0),
                    "search_mode": str(transfer_meta.get("search_mode", "")),
                    "candidate_pool_mode": str(transfer_meta.get("candidate_pool_mode", "")),
                    "incoming_driver_candidates": int(transfer_meta.get("incoming_driver_candidates", 0) or 0),
                    "incoming_constructor_candidates": int(transfer_meta.get("incoming_constructor_candidates", 0) or 0),
                    "incoming_driver_candidates_kept": int(transfer_meta.get("incoming_driver_candidates_kept", 0) or 0),
                    "incoming_constructor_candidates_kept": int(transfer_meta.get("incoming_constructor_candidates_kept", 0) or 0),
                    "outgoing_driver_candidates": int(transfer_meta.get("outgoing_driver_candidates", 0) or 0),
                    "outgoing_constructor_candidates": int(transfer_meta.get("outgoing_constructor_candidates", 0) or 0),
                    "outgoing_driver_candidates_kept": int(transfer_meta.get("outgoing_driver_candidates_kept", 0) or 0),
                    "outgoing_constructor_candidates_kept": int(transfer_meta.get("outgoing_constructor_candidates_kept", 0) or 0),
                    "exhaustive_candidate_count_before_filtering": int(
                        transfer_meta.get("exhaustive_candidate_count_before_filtering", 0) or 0
                    ),
                    "candidate_count_after_filtering": int(transfer_meta.get("candidate_count_after_filtering", 0) or 0),
                    "valid_transfer_plans_generated": int(transfer_meta.get("valid_transfer_plans_generated", 0) or 0),
                    "candidate_teams_scored": int(transfer_meta.get("candidate_teams_scored", 0) or 0),
                    "generated_partial_plans": int(transfer_meta.get("generated_partial_plans", 0) or 0),
                    "evaluated_full_candidates": int(transfer_meta.get("evaluated_full_candidates", 0) or 0),
                    "number_candidates_generated": int(transfer_meta.get("number_candidates_generated", 0) or 0),
                    "number_pruned_by_filtering": int(transfer_meta.get("number_pruned_by_filtering", 0) or 0),
                    "duplicate_teams_skipped": int(transfer_meta.get("duplicate_teams_skipped", 0) or 0),
                    "pruned_by_budget": int(transfer_meta.get("pruned_by_budget", 0) or 0),
                    "pruned_by_beam": int(transfer_meta.get("pruned_by_beam", 0) or 0),
                    "candidate_filter_score_used_for_prefilter": bool(
                        transfer_meta.get("candidate_filter_score_used_for_prefilter", False)
                    ),
                    "exhaustive_scoring_used_after_prefilter": bool(
                        transfer_meta.get("exhaustive_scoring_used_after_prefilter", False)
                    ),
                    "final_score_used_for_sorting": bool(transfer_meta.get("final_score_used_for_sorting", False)),
                    "generated_candidates_by_depth": transfer_meta.get("generated_candidates_by_depth", {}),
                    "beam_kept_by_depth": transfer_meta.get("beam_kept_by_depth", {}),
                    "fully_scored_by_depth": transfer_meta.get("fully_scored_by_depth", {}),
                    "final_recommendations_by_transfer_count": transfer_meta.get("final_recommendations_by_transfer_count", {}),
                    "recommendations_by_transfer_count": transfer_meta.get("recommendations_by_transfer_count", {}),
                    "transfer_results_count": int(transfer_meta.get("transfer_results_count", 0) or 0),
                }
                transfer_progress_ui.empty()

        recs = st.session_state.transfer_results
        if recs is None:
            st.info("Set transfer options, then run transfer recommendations.")
        elif recs.empty:
            if active_locks or active_exclusions:
                st.warning("No valid transfer recommendations found with the current locks/exclusions and transfer constraints.")
            else:
                st.warning("No valid transfer recommendations found with the current constraints.")
        else:
            top = recs.iloc[0]
            st.markdown("### Top recommendation")
            _transfer_move_card(top, is_top=True)
            _render_transfer_tradeoff_box(top)
            st.subheader("Other options")
            for _, rec in recs.iloc[1:].iterrows():
                with st.expander(f"#{int(rec['Rank'])}: {rec['OUT']} → {rec['IN']}"):
                    st.markdown(f"#### Option #{int(rec['Rank'])}")
                    _transfer_move_card(rec)
                    _render_transfer_tradeoff_box(rec)

with price_changes_tab:
    st.markdown('<div class="f1-section-kicker">MARKET</div>', unsafe_allow_html=True)
    st.caption(
        "Asset prices are the latest accepted official market values. Green/red gain values are "
        "projected next-round gains from the model, not already-realised official movements."
    )
    st.caption(
        "Inactive official assets remain visible when priced. Their settlement eligibility is "
        "unconfirmed, so projected gain stays unavailable rather than being shown as zero."
    )
    market_price_view = st.segmented_control(
        "Market view",
        options=["Projection", "Thresholds"],
        default="Projection",
        key="market_price_view",
        selection_mode="single",
    ) or "Projection"
    market_price_asset_type = st.segmented_control(
        "Asset type",
        options=["Drivers", "Constructors"],
        default="Drivers",
        key="market_price_asset_type",
        selection_mode="single",
    ) or "Drivers"
    with st.container(border=True):
        st.caption(format_next_race_header(diagnostics.get("next_race_name"), diagnostics.get("next_race_date")))
        if market_price_view == "Thresholds":
            tier_key = pd.DataFrame(
                [
                    {"Price tier": "≤ 18.5M", "Terrible": "-0.6M", "Poor": "-0.2M", "Good": "+0.2M", "Great": "+0.6M"},
                    {"Price tier": "> 18.5M", "Terrible": "-0.3M", "Poor": "-0.1M", "Good": "+0.1M", "Great": "+0.3M"},
                ]
            )
            st.dataframe(_tier_key_styler(tier_key), hide_index=True, width="stretch")

    if diagnostics.get("recent_points_fallback_used"):
        st.warning(
            "Recent true fantasy points are missing for some assets. The app uses the official playerstats endpoint when available; missing values stay blank unless you provide manual fallback points below."
        )

    price_view_drivers = build_price_change_asset_universe(
        edited_drivers,
        snapshot.player_assets,
        "driver",
        race_observations=driver_race_observations,
        player_identity_map=snapshot.player_identity_map,
    )
    price_view_constructors = build_price_change_asset_universe(
        edited_constructors,
        snapshot.constructor_assets,
        "constructor",
        race_observations=constructor_race_observations,
    )
    driver_price_change_table = _price_change_display_table(
        price_view_drivers,
        cheap_driver_rules,
        "drivers",
        expensive_rules=expensive_driver_rules,
        expensive_price_min=driver_expensive_min,
        bounds=price_change_bounds,
    )
    driver_price_change_table = prepare_compact_asset_table(
        driver_price_change_table,
        price_view_drivers,
        asset_type="driver",
    )
    constructor_price_change_table = _price_change_display_table(
        price_view_constructors,
        cheap_constructor_rules,
        "constructors",
        expensive_rules=expensive_constructor_rules,
        expensive_price_min=constructor_expensive_min,
        bounds=price_change_bounds,
    )
    constructor_price_change_table = prepare_compact_asset_table(
        constructor_price_change_table,
        price_view_constructors,
        asset_type="constructor",
    )
    driver_probability_matrix = _price_change_probability_matrix(
        price_view_drivers,
        cheap_driver_rules,
        "drivers",
        expensive_rules=expensive_driver_rules,
        expensive_price_min=driver_expensive_min,
        bounds=price_change_bounds,
    )
    driver_probability_matrix = prepare_compact_asset_table(
        driver_probability_matrix,
        price_view_drivers,
        asset_type="driver",
    )
    constructor_probability_matrix = _price_change_probability_matrix(
        price_view_constructors,
        cheap_constructor_rules,
        "constructors",
        expensive_rules=expensive_constructor_rules,
        expensive_price_min=constructor_expensive_min,
        bounds=price_change_bounds,
    )
    constructor_probability_matrix = prepare_compact_asset_table(
        constructor_probability_matrix,
        price_view_constructors,
        asset_type="constructor",
    )

    active_is_constructor = market_price_asset_type == "Constructors"
    if market_price_view == "Thresholds":
        active_threshold_table = (
            constructor_price_change_table if active_is_constructor else driver_price_change_table
        )
        active_asset_label = "constructor" if active_is_constructor else "driver"
        st.markdown(_price_change_table_html(active_threshold_table), unsafe_allow_html=True)
        _render_png_download(
            f"Download {active_asset_label} targets PNG",
            filename=f"f1_{active_asset_label}_price_change_targets.png",
            key=f"download_{active_asset_label}_price_change_targets_png",
            renderer=lambda: render_price_change_table_png(
                active_threshold_table,
                asset_type=active_asset_label,
                table_type="threshold",
            ),
        )
    else:
        driver_projection_assets = apply_probabilistic_price_change_model(
            price_view_drivers,
            cheap_driver_rules,
            expensive_rules=expensive_driver_rules,
            expensive_price_min=driver_expensive_min,
            bounds=price_change_bounds,
            predicted_points_col="next_race_expected_points",
        )
        constructor_projection_assets = apply_probabilistic_price_change_model(
            price_view_constructors,
            cheap_constructor_rules,
            expensive_rules=expensive_constructor_rules,
            expensive_price_min=constructor_expensive_min,
            bounds=price_change_bounds,
            predicted_points_col="next_race_expected_points",
        )
        active_projection_assets = (
            constructor_projection_assets if active_is_constructor else driver_projection_assets
        )
        active_probability_matrix = (
            constructor_probability_matrix if active_is_constructor else driver_probability_matrix
        )
        active_asset_label = "constructor" if active_is_constructor else "driver"
        st.markdown(
            compact_asset_table_html(active_projection_assets, asset_type=active_asset_label),
            unsafe_allow_html=True,
        )
        _render_png_download(
            f"Download {active_asset_label} projection PNG",
            filename=f"f1_{active_asset_label}_price_change_projection.png",
            key=f"download_{active_asset_label}_price_change_projection_png",
            renderer=lambda: render_price_change_table_png(
                active_probability_matrix,
                asset_type=active_asset_label,
                table_type="projection",
            ),
        )
        with st.expander("Probability details", expanded=False):
            st.markdown(_price_change_table_html(active_probability_matrix), unsafe_allow_html=True)

with price_efficiency_tab:
    st.markdown('<div class="f1-section-kicker">VALUE FINDER</div>', unsafe_allow_html=True)
    st.subheader("Price Efficiency")
    market_feed_label = diagnostics.get("feed_round") or "unavailable"
    st.caption(
        "Official selected-race points per million of current asset price "
        f"(accepted market feed {market_feed_label})."
    )

    with st.container(border=True):
        st.markdown("### Price Efficiency races")
        efficiency_race_preset = st.selectbox(
            "Race window",
            options=["Last 1", "Last 3", "Last 5", "All", "Custom"],
            key="efficiency_race_preset",
            format_func=lambda value: "All completed races" if value == "All" else value,
        )
        if efficiency_race_preset == "Custom":
            efficiency_custom_race_keys = st.multiselect(
                "Included races",
                options=[option.key for option in race_catalogue],
                key="efficiency_custom_race_keys",
                format_func=lambda key: race_option_label(key, race_name_by_key),
            )
        else:
            efficiency_custom_race_keys = list(
                st.session_state.get("efficiency_custom_race_keys", ())
            )

        efficiency_before_exclusions = reconcile_race_control_state(
            race_catalogue,
            efficiency_race_preset,
            efficiency_custom_race_keys,
            st.session_state.get("efficiency_excluded_race_keys", ()),
        )
        if list(st.session_state.get("efficiency_excluded_race_keys", ())) != list(
            efficiency_before_exclusions.excluded_keys
        ):
            st.session_state["efficiency_excluded_race_keys"] = list(
                efficiency_before_exclusions.excluded_keys
            )
        efficiency_excluded_race_keys = st.multiselect(
            "Exclude races",
            options=list(efficiency_before_exclusions.exclusion_options),
            key="efficiency_excluded_race_keys",
            format_func=lambda key: race_option_label(key, race_name_by_key),
        )
        efficiency_race_control = reconcile_race_control_state(
            race_catalogue,
            efficiency_race_preset,
            efficiency_custom_race_keys,
            efficiency_excluded_race_keys,
        )

    price_efficiency_export_race_summary = price_efficiency_race_summary(
        efficiency_race_control.selection,
        race_name_by_key,
    )
    if efficiency_race_control.selection.included:
        st.info(price_efficiency_export_race_summary)
    else:
        st.warning("No races are selected. Choose at least one completed race to calculate Price Efficiency.")
    st.caption(
        "Price Efficiency = average official points per valid selected race ÷ current price. "
        "This local view uses an unweighted arithmetic average and does not change model projections."
    )

    driver_source_failures = _price_efficiency_source_failure_ids(
        driver_price_efficiency,
        "driver",
    )
    constructor_source_failures = _price_efficiency_source_failure_ids(
        constructor_price_efficiency,
        "constructor",
    )
    local_driver_price_efficiency = build_price_efficiency_table(
        drivers,
        driver_race_observations,
        efficiency_race_control.selection,
        asset_type="driver",
        source_failures=driver_source_failures,
    )
    local_constructor_price_efficiency = build_price_efficiency_table(
        constructors,
        constructor_race_observations,
        efficiency_race_control.selection,
        asset_type="constructor",
        source_failures=constructor_source_failures,
    )

    active_efficiency_asset_type = st.segmented_control(
        "Asset type",
        options=["Drivers", "Constructors"],
        key="efficiency_asset_type",
        selection_mode="single",
    )
    active_efficiency_asset_type = resolve_price_efficiency_asset_type(
        active_efficiency_asset_type
    )
    efficiency_export_layout_label = st.selectbox(
        "Image layout",
        options=["Portrait", "Reddit landscape"],
        key="price_efficiency_image_layout",
        help="Portrait: 1080 × 1350. Reddit landscape: 1600 × 900.",
    )
    efficiency_export_format = (
        "landscape" if efficiency_export_layout_label == "Reddit landscape" else "portrait"
    )
    if efficiency_race_control.selection.included:
        if active_efficiency_asset_type == "Constructors":
            _render_price_efficiency_section(
                local_constructor_price_efficiency,
                "Constructors",
                "constructor",
                race_summary=price_efficiency_export_race_summary,
            )
        else:
            _render_price_efficiency_section(
                local_driver_price_efficiency,
                "Drivers",
                "driver",
                race_summary=price_efficiency_export_race_summary,
            )

    st.markdown("### Manual team builder")
    st.caption("Select five drivers and two constructors. This builder does not optimise or change your optimiser budget.")
    driver_efficiency_labels = _price_efficiency_option_labels(local_driver_price_efficiency)
    constructor_efficiency_labels = _price_efficiency_option_labels(local_constructor_price_efficiency)
    reconciled_efficiency_team = reconcile_price_efficiency_team_state(
        {
            "driver_ids": st.session_state.get("efficiency_team_driver_ids", ()),
            "constructor_ids": st.session_state.get("efficiency_team_constructor_ids", ()),
            "budget": st.session_state.get("efficiency_team_budget"),
        },
        driver_efficiency_labels,
        constructor_efficiency_labels,
        float(st.session_state.optimizer_budget),
    )
    for state_key, value in [
        ("efficiency_team_driver_ids", reconciled_efficiency_team["driver_ids"]),
        ("efficiency_team_constructor_ids", reconciled_efficiency_team["constructor_ids"]),
        ("efficiency_team_budget", reconciled_efficiency_team["budget"]),
    ]:
        if st.session_state.get(state_key) != value:
            st.session_state[state_key] = value

    efficiency_driver_col, efficiency_constructor_col = st.columns(2)
    with efficiency_driver_col:
        efficiency_driver_ids = st.multiselect(
            "Drivers (5)",
            options=list(driver_efficiency_labels),
            format_func=driver_efficiency_labels.get,
            max_selections=5,
            key="efficiency_team_driver_ids",
        )
    with efficiency_constructor_col:
        efficiency_constructor_ids = st.multiselect(
            "Constructors (2)",
            options=list(constructor_efficiency_labels),
            format_func=constructor_efficiency_labels.get,
            max_selections=2,
            key="efficiency_team_constructor_ids",
        )
    efficiency_budget = st.number_input(
        "Team-builder budget",
        min_value=0.0,
        step=0.1,
        key="efficiency_team_budget",
        help="Defaults to the current optimiser budget once. Later changes remain local to this builder.",
    )
    combined_efficiency_table = pd.concat(
        [local_driver_price_efficiency, local_constructor_price_efficiency],
        ignore_index=True,
    )
    efficiency_team_summary = summarize_price_efficiency_team(
        combined_efficiency_table,
        efficiency_driver_ids,
        efficiency_constructor_ids,
        float(efficiency_budget),
    )
    efficiency_metrics = st.columns(4)
    efficiency_metrics[0].metric("Team cost", format_money(efficiency_team_summary["total_cost"]))
    efficiency_metrics[1].metric("Remaining budget", format_money(efficiency_team_summary["remaining_budget"]))
    efficiency_metrics[2].metric(
        "Selected official points",
        format_points(efficiency_team_summary["total_selected_official_points"]),
    )
    efficiency_metrics[3].metric(
        "Team points / selected race",
        format_points(efficiency_team_summary["average_team_points_per_selected_race"]),
    )
    efficiency_metrics_2 = st.columns(3)
    efficiency_metrics_2[0].metric(
        "Summed asset efficiencies",
        format_points(efficiency_team_summary["sum_individual_asset_efficiencies"]),
    )
    efficiency_metrics_2[1].metric(
        "Comparable team efficiency",
        format_points(efficiency_team_summary["team_price_efficiency"]),
    )
    efficiency_metrics_2[2].metric(
        "Component coverage",
        f"{float(efficiency_team_summary['component_coverage']):.0%}",
    )
    st.caption(
        "Team points / selected race is total official team points divided by the shared selected-race count and is shown only for complete comparable coverage. "
        "Comparable team efficiency divides that value by total current cost."
    )
    selected_efficiency_asset_count = len(efficiency_driver_ids) + len(efficiency_constructor_ids)
    if selected_efficiency_asset_count < 7:
        st.info("Choose exactly five drivers and two constructors to validate a complete team.")
    else:
        for message in efficiency_team_summary["messages"]:
            if "over budget" in message.casefold() or "unknown" in message.casefold():
                st.error(message)
            else:
                st.warning(message)
        if efficiency_team_summary["valid"]:
            st.success("Team composition, budget and efficiency coverage are valid.")

    if selected_efficiency_asset_count == 7 and efficiency_race_control.selection.included:
        selected_efficiency_ids = [
            *(str(asset_id) for asset_id in efficiency_driver_ids),
            *(str(asset_id) for asset_id in efficiency_constructor_ids),
        ]
        efficiency_by_id = {
            str(row.get("asset_id")): dict(row)
            for row in combined_efficiency_table.to_dict(orient="records")
        }
        selected_efficiency_assets = pd.DataFrame(
            [efficiency_by_id[asset_id] for asset_id in selected_efficiency_ids if asset_id in efficiency_by_id]
        )
        if len(selected_efficiency_assets) == 7:
            _render_png_download(
                "Download Price Efficiency team PNG",
                filename="f1_price_efficiency_team.png",
                key="download_price_efficiency_team_png",
                renderer=lambda: render_price_efficiency_team_png(
                    efficiency_team_summary,
                    selected_efficiency_assets,
                    price_efficiency_export_race_summary,
                    format=efficiency_export_format,
                ),
            )
        else:
            st.caption("The selected team cannot be exported until all seven assets are available.")
    elif selected_efficiency_asset_count == 7:
        st.caption("Select at least one race before exporting this team.")

    with st.expander("Coverage and data quality", expanded=False):
        st.write(
            "A zero-point race is valid. Missing races are not converted to zero. Replacement drivers are averaged only over races they contested, so their reduced coverage remains visible. Source failures and unknown missing observations are flagged in the table."
        )

model_selected_race_labels = [
    race_option_label(key, race_name_by_key)
    for key in race_control.selection.included
]
model_selected_summary = (
    " · ".join(model_selected_race_labels)
    if model_selected_race_labels
    else "No races selected"
)

with optimise_tab:
    st.markdown('<div class="f1-section-kicker">OPTIMISE</div>', unsafe_allow_html=True)
    mobile_subview = optimise_mobile_subview(
        st.segmented_control(
            "Optimise view",
            options=list(OPTIMISE_MOBILE_SUBVIEWS),
            default="Teams",
            key="optimise_mobile_subview",
            label_visibility="collapsed",
        )
    )
    st.markdown(
        f'<span class="f1-optimise-view-marker f1-optimise-view-{mobile_subview.casefold()}" '
        'aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )
    with st.container(key="optimiser_teams_action"):
        teams_run_clicked = st.button(
            "Run / rerun optimiser",
            type="primary",
            use_container_width=True,
            key="run_optimiser_mobile_teams",
        )
    with st.container(key="optimiser_dashboard"):
        with st.container(key="optimiser_controls_view"):
            st.markdown("#### Optimiser controls")
            with st.container(key="optimiser_quick_setup", border=True):
                setup_slider_col, setup_budget_col, setup_chip_col, setup_run_col = st.columns(
                    [4.4, 1.35, 1.8, 1.45],
                    gap="small",
                    vertical_alignment="bottom",
                )
                with setup_slider_col:
                    price_gain_weight = st.slider(
                        "Price-growth value",
                        min_value=0,
                        max_value=100,
                        step=5,
                        disabled=chip_mode == CHIP_LIMITLESS,
                        key="optimise_price_gain_weight_slider",
                        help="Objective points assigned per +1.0M expected gain.",
                    )
                with setup_budget_col:
                    st.number_input(
                        "Budget",
                        min_value=0.0,
                        step=0.1,
                        key="optimizer_budget",
                        on_change=_mark_budget_manual_from_optimizer,
                    )
                budget = float(st.session_state.optimizer_budget)
                with setup_chip_col:
                    st.selectbox(
                        "Chip",
                        options=["None", "3x chip", "Limitless", "No Negative chip"],
                        key="chip_mode_label",
                    )
                chip_mode = chip_mode_from_label(st.session_state.chip_mode_label)
                with setup_run_col:
                    controls_run_clicked = st.button(
                        "Run optimiser",
                        type="primary",
                        use_container_width=True,
                        key="run_optimiser_controls",
                    )
                st.slider(
                    "Live session emphasis",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key="model_live_session_emphasis",
                    help="0 = ignore practice/SQ; 1 = use live-session ranking entirely",
                )
                st.caption(live_session_status_text)
            controls_objective_mode = (
                OBJECTIVE_POINTS_ONLY if chip_mode == CHIP_LIMITLESS else OBJECTIVE_COMBINED
            )

            with st.container(key="optimiser_mobile_model_controls", border=True):
                st.markdown("##### Model window")
                mirror_values = {
                    "optimise_mobile_current_season_only": bool(current_season_only),
                    "optimise_mobile_model_race_preset": race_control.preset,
                    "optimise_mobile_model_excluded_races": list(race_control.excluded_keys),
                    "optimise_mobile_model_recency_decay": float(recency_decay),
                    "optimise_mobile_live_session_emphasis": float(live_session_emphasis),
                }
                if race_control.preset == "Custom":
                    mirror_values["optimise_mobile_model_custom_races"] = list(
                        race_control.custom_keys
                    )
                for mirror_key, mirror_value in mirror_values.items():
                    if st.session_state.get(mirror_key) != mirror_value:
                        st.session_state[mirror_key] = mirror_value
                st.toggle(
                    "Current season only",
                    key="optimise_mobile_current_season_only",
                    on_change=_sync_session_value,
                    args=("optimise_mobile_current_season_only", "current_season_only"),
                )
                mobile_race_preset = st.selectbox(
                    "Race window",
                    options=["Last 1", "Last 3", "Last 5", "All", "Custom"],
                    key="optimise_mobile_model_race_preset",
                    format_func=lambda value: "All completed races" if value == "All" else value,
                    on_change=_sync_session_value,
                    args=("optimise_mobile_model_race_preset", "model_race_preset"),
                )
                if mobile_race_preset == "Custom":
                    st.multiselect(
                        "Custom model races",
                        options=[option.key for option in race_catalogue],
                        key="optimise_mobile_model_custom_races",
                        format_func=lambda key: race_option_label(key, race_name_by_key),
                        on_change=_sync_session_value,
                        args=("optimise_mobile_model_custom_races", "model_custom_race_keys"),
                    )
                st.multiselect(
                    "Exclude model races",
                    options=list(before_exclusions.exclusion_options),
                    key="optimise_mobile_model_excluded_races",
                    format_func=lambda key: race_option_label(key, race_name_by_key),
                    on_change=_sync_session_value,
                    args=("optimise_mobile_model_excluded_races", "model_excluded_race_keys"),
                )
                st.slider(
                    "Current-season recency decay p",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key="optimise_mobile_model_recency_decay",
                    on_change=_sync_session_value,
                    args=("optimise_mobile_model_recency_decay", "model_recency_decay"),
                )
                st.slider(
                    "Live session emphasis",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key="optimise_mobile_live_session_emphasis",
                    help="0 = ignore practice/SQ; 1 = use live-session ranking entirely",
                    on_change=_sync_session_value,
                    args=("optimise_mobile_live_session_emphasis", "model_live_session_emphasis"),
                )
                st.caption(live_session_status_text)
                st.caption(
                    f"{len(race_catalogue)} eligible · "
                    f"{len(race_control.selection.included)} selected · "
                    f"history: {'current season only' if current_season_only else 'all supported'}"
                )
                with st.expander("Advanced optimiser and model settings", expanded=False):
                    st.write("Objective:", controls_objective_mode)
                    st.write("Current / historical blend:", f"{current_percent}% / {historical_percent}%")
                    st.write("Selected races:", model_selected_summary)
                    st.caption("Detailed blend and model-range controls remain available in Settings → Model.")

        run_clicked = bool(teams_run_clicked or controls_run_clicked)

        objective_mode = controls_objective_mode
        optimiser_input_drivers = (
            apply_no_negative_scores(price_change_drivers)
            if chip_mode == CHIP_NO_NEGATIVE
            else price_change_drivers
        )
        optimiser_input_constructors = (
            apply_no_negative_scores(price_change_constructors)
            if chip_mode == CHIP_NO_NEGATIVE
            else price_change_constructors
        )
        try:
            optimizer_drivers = apply_objective_mode(
                apply_probabilistic_price_change_model(
                    optimiser_input_drivers,
                    cheap_driver_rules,
                    expensive_rules=expensive_driver_rules,
                    expensive_price_min=driver_expensive_min,
                    bounds=price_change_bounds,
                    predicted_points_col="next_race_expected_points",
                ),
                objective_mode=objective_mode,
                price_gain_weight=price_gain_weight,
            )
            optimizer_constructors = apply_objective_mode(
                apply_probabilistic_price_change_model(
                    optimiser_input_constructors,
                    cheap_constructor_rules,
                    expensive_rules=expensive_constructor_rules,
                    expensive_price_min=constructor_expensive_min,
                    bounds=price_change_bounds,
                    predicted_points_col="next_race_expected_points",
                ),
                objective_mode=objective_mode,
                price_gain_weight=price_gain_weight,
            )
            objective_drs_multiplier = 1.0 if objective_mode == OBJECTIVE_PRICE_GROWTH_ONLY else 2.0
            triple_multiplier = 3.0 if chip_mode == CHIP_TRIPLE and objective_mode != OBJECTIVE_PRICE_GROWTH_ONLY else None
            optimiser_budget = None if chip_mode == CHIP_LIMITLESS else budget
            current_optimiser_signature = optimiser_result_signature(
                data_version=active_model_data_version,
                budget=optimiser_budget,
                chip_mode=chip_mode,
                price_growth_value=price_gain_weight,
                locked_driver_ids=locked_driver_ids,
                excluded_driver_ids=excluded_driver_ids,
                locked_constructor_ids=locked_constructor_ids,
                excluded_constructor_ids=excluded_constructor_ids,
            )
        except Exception as exc:
            LOGGER.exception("Optimiser inputs could not be prepared")
            st.session_state["last_optimiser_error"] = str(exc)
            optimizer_drivers = pd.DataFrame()
            optimizer_constructors = pd.DataFrame()
            current_optimiser_signature = None
        else:
            if run_clicked:
                try:
                    initial_solutions = run_optimizer(
                        optimizer_drivers,
                        optimizer_constructors,
                        budget=optimiser_budget,
                        top_k=10,
                        drs_multiplier=objective_drs_multiplier,
                        allow_no_negative=chip_mode == CHIP_NO_NEGATIVE,
                        locked_driver_ids=locked_driver_ids,
                        excluded_driver_ids=excluded_driver_ids,
                        locked_constructor_ids=locked_constructor_ids,
                        excluded_constructor_ids=excluded_constructor_ids,
                        objective_col="combined_objective_score",
                        boost_col="exp_score",
                        triple_multiplier=triple_multiplier,
                    )
                except Exception as exc:
                    LOGGER.exception("Optimiser run failed")
                    st.session_state["last_optimiser_error"] = str(exc)
                else:
                    st.session_state["last_optimiser_error"] = None
                    st.session_state["optimiser_solutions"] = initial_solutions
                    st.session_state["optimiser_result_signature"] = current_optimiser_signature
                    st.session_state["optimiser_results_exhausted"] = len(initial_solutions) < 10
                    st.session_state["optimiser_result_context"] = {
                        "budget": budget,
                        "chip_mode": chip_mode,
                    }

        results_col, universe_col = st.columns([73, 27], gap="small")

        with universe_col:
            with st.container(key="optimiser_universe_selector"):
                st.markdown("#### Asset universe")
                universe_type = st.segmented_control(
                    "Universe",
                    options=["Drivers", "Constructors"],
                    default="Drivers",
                    key="optimiser_asset_type",
                    label_visibility="collapsed",
                ) or "Drivers"
                st.markdown(
                    f'<span class="f1-universe-desktop-{universe_type.casefold()}" '
                    'aria-hidden="true"></span>',
                    unsafe_allow_html=True,
                )
            with st.container(key="optimiser_drivers_view"):
                st.markdown("#### Drivers")
                _render_compact_asset_universe(
                    optimizer_drivers,
                    asset_type="driver",
                    locked_ids=list(locked_driver_ids),
                    excluded_ids=list(excluded_driver_ids),
                    container_key="optimiser_universe_scroll_drivers",
                )
            with st.container(key="optimiser_constructors_view"):
                st.markdown("#### Constructors")
                _render_compact_asset_universe(
                    optimizer_constructors,
                    asset_type="constructor",
                    locked_ids=list(locked_constructor_ids),
                    excluded_ids=list(excluded_constructor_ids),
                    container_key="optimiser_universe_scroll_constructors",
                )

        with results_col:
            with st.container(key="optimiser_teams_view"):
                stored_solutions = list(st.session_state.get("optimiser_solutions", ()))
                stored_signature = st.session_state.get("optimiser_result_signature")
                stored_context = dict(st.session_state.get("optimiser_result_context", {}))
                display_budget = float(stored_context.get("budget", budget))
                display_chip_mode = str(stored_context.get("chip_mode", chip_mode))
                results_stale = bool(stored_solutions) and stored_signature != current_optimiser_signature
                if st.session_state.get("last_optimiser_error"):
                    st.error("The optimiser could not run with the selected inputs.")
                if results_stale:
                    st.warning("Inputs changed. Run optimiser to refresh these teams.")
                if not stored_solutions:
                    st.info("Run the optimiser to calculate Teams 1–10.")
                else:
                    st.markdown(f"#### Ranked teams · {len(stored_solutions)}")
                    with st.container(height=620, border=False, key="optimiser_results_scroll"):
                        for rank, solution in enumerate(stored_solutions, start=1):
                            _ranked_team_component(
                                solution,
                                rank=rank,
                                budget=display_budget,
                                limitless=display_chip_mode == CHIP_LIMITLESS,
                            )

                    next_start = len(stored_solutions) + 1
                    next_label = f"Load teams {next_start}–{next_start + 9}"
                    no_more = bool(st.session_state.get("optimiser_results_exhausted", False))
                    load_next_clicked = st.button(
                        next_label,
                        key="load_next_optimiser_batch",
                        disabled=results_stale or no_more,
                        use_container_width=True,
                    )
                    if load_next_clicked:
                        excluded_combinations = [
                            (list(team_solution_key(solution)[0]), list(team_solution_key(solution)[1]))
                            for solution in stored_solutions
                        ]
                        try:
                            candidates = run_optimizer(
                                optimizer_drivers,
                                optimizer_constructors,
                                budget=optimiser_budget,
                                top_k=10,
                                drs_multiplier=objective_drs_multiplier,
                                allow_no_negative=chip_mode == CHIP_NO_NEGATIVE,
                                locked_driver_ids=locked_driver_ids,
                                excluded_driver_ids=excluded_driver_ids,
                                locked_constructor_ids=locked_constructor_ids,
                                excluded_constructor_ids=excluded_constructor_ids,
                                objective_col="combined_objective_score",
                                boost_col="exp_score",
                                triple_multiplier=triple_multiplier,
                                excluded_team_combinations=excluded_combinations,
                            )
                        except Exception as exc:
                            LOGGER.exception("Next optimiser batch failed")
                            st.session_state["last_optimiser_error"] = str(exc)
                        else:
                            batch = next_team_batch(stored_solutions, candidates, batch_size=10)
                            st.session_state["optimiser_solutions"] = batch["solutions"]
                            st.session_state["optimiser_results_exhausted"] = batch["exhausted"]
                            st.rerun()
                    if no_more:
                        st.caption("No further unique teams are available for these inputs.")

                    if not results_stale:
                        with st.expander("Export a ranked team", expanded=False):
                            export_team_col, export_layout_col = st.columns(2, gap="small")
                            with export_team_col:
                                export_rank = st.selectbox(
                                    "Team",
                                    options=list(range(1, len(stored_solutions) + 1)),
                                    format_func=lambda value: f"Team {value}",
                                    key="optimise_export_team_rank",
                                )
                            with export_layout_col:
                                optimise_export_layout_label = st.selectbox(
                                    "Layout",
                                    options=["Portrait", "Reddit landscape"],
                                    key="optimise_image_layout",
                                )
                            export_solution = stored_solutions[int(export_rank) - 1]
                            export_summary = _team_summary(export_solution, display_budget)
                            optimise_export_format = (
                                "landscape"
                                if optimise_export_layout_label == "Reddit landscape"
                                else "portrait"
                            )
                            export_download_col, export_current_col = st.columns(2, gap="small")
                            with export_download_col:
                                _render_png_download(
                                    f"Download Team {export_rank} PNG",
                                    filename=f"f1_projected_team_{int(export_rank)}.png",
                                    key="download_ranked_projected_team_png",
                                    renderer=lambda: render_projected_team_png(
                                        export_solution,
                                        title=f"Projected Team {export_rank}",
                                        budget=None if display_chip_mode == CHIP_LIMITLESS else display_budget,
                                        expected_points=export_summary["Expected points"],
                                        format=optimise_export_format,
                                    ),
                                )
                            with export_current_col:
                                st.button(
                                    "Set as current team",
                                    key="set_ranked_team_as_current",
                                    use_container_width=True,
                                    on_click=_copy_ranked_team_to_current,
                                    args=(
                                        export_solution,
                                        list(driver_labels),
                                        list(constructor_labels),
                                        int(export_rank),
                                    ),
                                )
                        if st.session_state.get("ranked_team_copy_notice"):
                            st.success(st.session_state["ranked_team_copy_notice"])
                        if st.session_state.get("ranked_team_copy_error"):
                            st.error(st.session_state["ranked_team_copy_error"])

with diagnostics_tab:
    st.markdown('<div class="f1-section-kicker">DATA CHECKS</div>', unsafe_allow_html=True)
    st.subheader("Diagnostics")
    st.caption("Check feed status, playerstats coverage, deadline source and model assumptions.")
    with st.container(key="diagnostics_summary_metrics"):
        diag_cols = st.columns(4)
        diag_cols[0].metric("Season", diagnostics["current_season"])
        diag_cols[1].metric("Fantasy feed", diagnostics["feed_round"])
        diag_cols[2].metric("Drivers", diagnostics["driver_count"])
        diag_cols[3].metric("Constructors", diagnostics["constructor_count"])

        settings_cols = st.columns(5)
        settings_cols[0].metric(
            "Historical seasons used",
            f"{diagnostics.get('historical_seasons_used', 0)} / {diagnostics.get('historical_seasons_requested', 0)}",
        )
        settings_cols[1].metric("Current weight", f"{diagnostics['current_season_weight']:.2f}")
        settings_cols[2].metric("Past weight", f"{diagnostics['past_season_weight']:.2f}")
        settings_cols[3].metric("Recency decay", f"{diagnostics['recency_decay']:.2f}")
        settings_cols[4].metric("Race horizon", diagnostics["upcoming_race_horizon"])

    st.markdown("#### Current status")
    status_cols = st.columns(3)
    status_cols[0].metric("Live data", str(diagnostics.get("live_data_status", "fresh")).title())
    status_cols[1].metric("History", "Current only" if current_season_only else "All supported")
    status_cols[2].metric(
        "Selected races",
        diagnostics.get("selected_current_season_race_count", len(race_control.selection.included)),
    )
    st.caption(
        f"Decay p={float(diagnostics.get('recency_decay', recency_decay)):.2f} · "
        f"feed {diagnostics.get('feed_round') or 'unavailable'} · "
        f"last refresh {diagnostics.get('raw_live_load_finished_utc', 'unavailable')}"
    )
    if scoring_fallback_in_use:
        st.warning(
            "Market data is current; previous verified scoring inputs remain in use because "
            "the latest scoring refresh did not fully validate."
        )
    elif diagnostics.get("last_load_fallback_used"):
        st.warning("The latest refresh did not fully validate; verified fallback data remains in use.")
    last_load_error = diagnostics.get("last_load_error") or st.session_state.get("last_load_error")
    if last_load_error:
        st.warning(str(last_load_error))

    live_session_diagnostics = diagnostics.get("live_session_ingestion", {})
    live_session_states = live_session_diagnostics.get("sessions", {})
    if live_session_states:
        st.markdown("#### Live session classifications")
        st.caption(
            "Complete Free Practice and Sprint Qualifying results can blend into the "
            "next-event production forecast at the selected emphasis."
        )
        session_summary = pd.DataFrame(
            [
                {
                    "Session": values.get("session_kind", kind),
                    "Status": values.get("status", "unknown"),
                    "Rows": values.get("rows_observed", 0),
                    "Expected": values.get("expected_participants"),
                    "Source": values.get("source", live_session_diagnostics.get("source")),
                    "Fetched (UTC)": values.get("fetch_timestamp_utc")
                    or live_session_diagnostics.get("fetch_timestamp_utc"),
                }
                for kind, values in live_session_states.items()
            ]
        )
        st.dataframe(session_summary, hide_index=True, width="stretch")
        for kind, values in live_session_states.items():
            if values.get("source_disagreement"):
                st.warning(f"{kind}: {values['source_disagreement']}")
        complete_kinds = {
            kind
            for kind, values in live_session_states.items()
            if values.get("status") == "complete"
        }
        preview = snapshot.session_results[
            snapshot.session_results.get(
                "session_kind", pd.Series(index=snapshot.session_results.index, dtype=object)
            ).isin(complete_kinds)
        ].copy()
        if not preview.empty:
            with st.expander("Completed-session classification preview", expanded=False):
                st.dataframe(
                    preview[
                        [
                            "session_kind",
                            "position",
                            "abbreviation",
                            "display_name",
                            "team",
                            "human_driver_id",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )

    live_shadow_diagnostics = diagnostics.get("live_session_shadow", {})
    if live_shadow_diagnostics:
        st.markdown("#### Live-session production blend")
        st.caption(
            f"Live session emphasis: {float(live_session_emphasis):.2f} · "
            "only the next-event delta is added to the multi-race horizon."
        )
        if live_shadow_diagnostics.get("error"):
            st.warning(
                "The live-session layer is unavailable; baseline production forecasts remain in use. "
                f"{live_shadow_diagnostics['error']}"
            )
        else:
            driver_shadow_columns = [
                "name",
                "baseline_ev",
                "FP1_score",
                "FP2_score",
                "FP3_score",
                "SQ_score",
                "live_session_score",
                "live_session_rank",
                "live_only_ev",
                "sessions_used",
                "live_session_emphasis",
                "adjusted_ev",
                "live_session_ev_difference",
            ]
            driver_shadow_table = drivers[
                [column for column in driver_shadow_columns if column in drivers.columns]
            ].copy()
            driver_shadow_table.rename(
                columns={
                    "name": "Asset",
                    "baseline_ev": "Baseline EV",
                    "FP1_score": "FP1",
                    "FP2_score": "FP2",
                    "FP3_score": "FP3",
                    "SQ_score": "SQ",
                    "live_session_score": "Live score",
                    "live_session_rank": "Live rank",
                    "live_only_ev": "Live-only EV",
                    "sessions_used": "Sessions used",
                    "live_session_emphasis": "w",
                    "adjusted_ev": "Adjusted EV",
                    "live_session_ev_difference": "Difference",
                },
                inplace=True,
            )
            constructor_shadow_columns = [
                "name",
                "baseline_ev",
                "driver_coverage",
                "constructor_live_session_score",
                "constructor_live_session_rank",
                "constructor_live_only_ev",
                "live_session_emphasis",
                "adjusted_ev",
                "live_session_ev_difference",
            ]
            constructor_shadow_table = constructors[
                [
                    column
                    for column in constructor_shadow_columns
                    if column in constructors.columns
                ]
            ].copy()
            constructor_shadow_table.rename(
                columns={
                    "name": "Constructor",
                    "baseline_ev": "Baseline EV",
                    "driver_coverage": "Driver coverage",
                    "constructor_live_session_score": "Live score",
                    "constructor_live_session_rank": "Live rank",
                    "constructor_live_only_ev": "Live-only EV",
                    "live_session_emphasis": "w",
                    "adjusted_ev": "Adjusted EV",
                    "live_session_ev_difference": "Difference",
                },
                inplace=True,
            )
            if not driver_shadow_table.empty:
                st.markdown("**Drivers**")
                st.dataframe(driver_shadow_table, hide_index=True, width="stretch")
            if not constructor_shadow_table.empty:
                st.markdown("**Constructors**")
                st.dataframe(constructor_shadow_table, hide_index=True, width="stretch")

    st.write("Upcoming circuits:", ", ".join(diagnostics["upcoming_circuits"]))
    sprint_production = diagnostics.get("sprint_ev_production", {})
    if sprint_production.get("upcoming_weekend_format") == "sprint":
        st.caption("Includes Sprint adjustment")
        show_sprint_breakdown = st.toggle(
            "Show Sprint EV breakdown",
            value=False,
            key="show_sprint_ev_breakdown",
            help="Shows the unchanged production baseline, approved Sprint add-on and final EV.",
        )
        if show_sprint_breakdown:
            st.markdown("**Approved Sprint adjustment**")
            if sprint_production.get("status") != "available":
                st.warning(
                    "Sprint adjustment is unavailable; baseline EV remains in use. "
                    f"{sprint_production.get('error') or ''}".strip()
                )
            else:
                st.caption(
                    f"{sprint_production.get('model_version')} · 2026 calibration · "
                    f"production history: {sprint_production.get('production_history_mode')}"
                )
                for label, frame in (("Drivers", drivers), ("Constructors", constructors)):
                    columns = [
                        "name",
                        "baseline_expected_points",
                        "sprint_bonus",
                        "next_race_expected_points",
                    ]
                    if set(columns).issubset(frame.columns):
                        table = frame[columns].copy()
                        table.rename(
                            columns={
                                "name": "Asset",
                                "baseline_expected_points": "Base EV",
                                "sprint_bonus": "Sprint adjustment",
                                "next_race_expected_points": "Final EV",
                            },
                            inplace=True,
                        )
                        st.markdown(f"**{label}**")
                        with st.container(key=f"sprint_diagnostics_desktop_{label.casefold()}"):
                            st.dataframe(
                                table,
                                hide_index=True,
                                use_container_width=True,
                                column_config={
                                    "Base EV": st.column_config.NumberColumn(format="%.2f"),
                                    "Sprint adjustment": st.column_config.NumberColumn(format="%+.2f"),
                                    "Final EV": st.column_config.NumberColumn(format="%.2f"),
                                },
                            )
                        with st.container(key=f"sprint_diagnostics_mobile_{label.casefold()}"):
                            st.markdown(sprint_diagnostic_table_html(frame), unsafe_allow_html=True)
    technical_diagnostics = st.expander("Technical diagnostics", expanded=False)
    technical_diagnostics.__enter__()
    st.write("Model load started (UTC):", diagnostics.get("model_load_started_utc", "Unavailable"))
    st.write("Model load finished (UTC):", diagnostics.get("model_load_finished_utc", "Unavailable"))
    st.write(
        "Last successful data refresh (UTC):",
        diagnostics.get("raw_live_load_finished_utc", "Unavailable"),
    )
    st.write("Model load duration (s):", f"{float(diagnostics.get('model_load_duration_seconds', 0.0)):.2f}")
    st.write("Live data status:", diagnostics.get("live_data_status", "fresh"))
    st.write("Market resolution:", diagnostics.get("market_resolution_method") or "Unavailable")
    st.write("Market content signature:", diagnostics.get("market_content_signature") or "Unavailable")
    st.write("Market content changed:", bool(diagnostics.get("market_content_changed", False)))
    st.write("Scoring data status:", diagnostics.get("scoring_data_status", "current"))
    st.write("Scoring verified at (UTC):", diagnostics.get("scoring_verified_at_utc") or "Unavailable")
    st.write("Scoring refresh reason:", diagnostics.get("scoring_refresh_error") or "None")
    st.write("Playerstats data status:", diagnostics.get("playerstats_data_status", "unavailable"))
    st.write("Deadline data status:", diagnostics.get("deadline_data_status", "unavailable"))
    st.write(
        "Market/scoring freshness mismatch:",
        bool(diagnostics.get("market_scoring_freshness_mismatch", False)),
    )
    st.write("Higher-feed probe note:", diagnostics.get("market_latest_probe_error") or "None")
    st.write("History mode:", diagnostics.get("history_mode", HISTORY_MODE_ALL_SUPPORTED))
    st.write(
        "Eligible current-season races:",
        diagnostics.get("eligible_current_season_race_count", 0),
    )
    st.write("Official feed in use:", diagnostics.get("feed_round") or "None")
    st.write(
        "Fantasy driver asset ledger:",
        (
            f"{diagnostics.get('driver_asset_count', 0)} total, "
            f"{diagnostics.get('selectable_driver_asset_count', 0)} selectable, "
            f"{diagnostics.get('inactive_driver_asset_count', 0)} inactive"
        ),
    )
    st.write("Full asset ledger available:", bool(diagnostics.get("asset_ledger_complete", False)))
    st.write(
        "Duplicate human drivers represented by multiple assets:",
        int(diagnostics.get("duplicate_human_driver_count", 0) or 0),
    )
    duplicate_human_assets = diagnostics.get("duplicate_human_driver_assets", [])
    if duplicate_human_assets:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Driver": group.get("display_name"),
                        "Human ID": group.get("human_driver_id"),
                        "Fantasy asset": asset.get("fantasy_asset_id"),
                        "Team": asset.get("team_name"),
                        "Active": asset.get("active"),
                        "Identity status": asset.get("match_status"),
                    }
                    for group in duplicate_human_assets
                    for asset in group.get("assets", [])
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    player_asset_mappings = diagnostics.get("player_asset_identity_mappings", [])
    if player_asset_mappings:
        with st.expander("Fantasy asset → human driver mappings", expanded=False):
            st.dataframe(pd.DataFrame(player_asset_mappings), hide_index=True, width="stretch")
    st.write("Generated snapshot round:", diagnostics.get("live_data_snapshot_round") or "None")
    st.write("Live refresh technical reason:", diagnostics.get("live_data_refresh_error") or "None")
    st.write("Last load fallback used:", bool(diagnostics.get("last_load_fallback_used", False)))
    st.write("Last load error:", diagnostics.get("last_load_error") or st.session_state.get("last_load_error") or "None")
    st.write("Last optimiser error:", st.session_state.get("last_optimiser_error") or "None")
    st.write("Playerstats prefetch enabled:", diagnostics.get("playerstats_prefetch_enabled", False))
    st.write("Budget init mode:", st.session_state.get("budget_init_mode", "unknown"))
    st.write("Budget init team cost:", format_money(st.session_state.get("budget_init_team_cost")))
    st.write("Budget init bank:", format_money(st.session_state.get("budget_init_bank")))
    st.write("Optimiser budget source:", st.session_state.get("optimizer_budget_source", "default"))
    st.write("Current-team budget source:", st.session_state.get("current_team_budget_source", "default"))
    st.write("Optimiser budget fixed by user action:", bool(st.session_state.get("budget_user_overridden", False)))
    st.write("Team lock deadline (UTC):", diagnostics.get("team_lock_deadline_utc") or "Unavailable")
    st.write("Team lock source:", diagnostics.get("team_lock_deadline_source", "Unavailable"))
    st.write("Team lock raw field:", diagnostics.get("team_lock_deadline_raw_field") or "Unavailable")
    st.write("Team lock raw value:", diagnostics.get("team_lock_deadline_raw_value") or "Unavailable")
    st.write("Team lock timezone assumption:", diagnostics.get("team_lock_timezone_assumption", "Unavailable"))
    st.write("Team lock matched event:", diagnostics.get("team_lock_matched_event") or "Unavailable")
    st.write("Team lock validation:", diagnostics.get("team_lock_validation_reason") or "Unavailable")
    st.write("Active event:", diagnostics.get("active_event") or "Unavailable")
    st.write("Weekend format:", diagnostics.get("weekend_format", "Unavailable"))
    st.write("Weekend status:", diagnostics.get("weekend_status", "Unavailable"))
    st.write("Session states:", diagnostics.get("weekend_session_states", {}))
    st.write("Live weekend excluded from completed form:", diagnostics.get("completed_form_excludes_live_weekend", False))
    st.write("Partial/pending weekend warnings:", diagnostics.get("weekend_state_warnings", []))
    round_lineage = diagnostics.get("current_season_round_lineage", [])
    if round_lineage:
        st.markdown("**Current-season score lineage**")
        st.dataframe(pd.DataFrame(round_lineage), hide_index=True, width="stretch")
    st.write(
        "Latest rejected live-session diagnostics:",
        diagnostics.get("latest_live_weekend_diagnostics") or "None",
    )
    st.write("Horizon weights:", diagnostics["horizon_weights"])
    st.write("Data window:", f"{diagnostics['start_year']} to {diagnostics['current_season']}")
    st.write("Recent price-change point source:", diagnostics.get("recent_points_source", "Unavailable"))
    st.write("Playerstats endpoint pattern:", diagnostics.get("recent_points_endpoint_pattern", "Unavailable"))
    st.write(
        "Observed-current calibration:",
        (
            f"Current avg {diagnostics.get('observed_current_avg_points_per_race'):.2f}, "
            f"historical avg {diagnostics.get('historical_avg_points_per_race'):.2f}, "
            f"scale {diagnostics.get('historical_scale_factor'):.3f}, "
            f"clipped {diagnostics.get('historical_scale_factor_clipped')}"
        )
        if diagnostics.get("observed_current_avg_points_per_race") is not None
        and diagnostics.get("historical_avg_points_per_race") is not None
        else "Unavailable",
    )
    st.write(
        "Observed playerstats coverage:",
        (
            f"{diagnostics.get('observed_current_assets', 0)} assets, "
            f"{diagnostics.get('observed_current_race_rows', 0)} race rows"
        ),
    )
    st.write("Volatility source:", diagnostics.get("volatility_source", "Unavailable"))
    st.write(
        "Volatility floors:",
        (
            f"Drivers {diagnostics.get('driver_volatility_floor', 0):.1f}, "
            f"Constructors {diagnostics.get('constructor_volatility_floor', 0):.1f}"
        ),
    )
    st.write(
        "Volatility coverage:",
        (
            f"Current-season {diagnostics.get('current_season_volatility_assets', 0)} assets, "
            f"historical/model {diagnostics.get('historical_volatility_assets', 0)} assets, "
            f"blended {diagnostics.get('blended_current_historical_volatility_assets', 0)}, "
            f"current-only {diagnostics.get('current_only_volatility_assets', 0)}, "
            f"historical-only {diagnostics.get('historical_only_volatility_assets', 0)}, "
            f"fallback {diagnostics.get('fallback_volatility_assets', 0)}, "
            f"floor applied {diagnostics.get('volatility_floor_applied_assets', 0)}"
        ),
    )
    st.write("Volatility source counts:", diagnostics.get("volatility_source_counts", {}))
    st.write(
        "DNF mixture:",
        (
            f"price-gain DNF component score {diagnostics.get('dnf_price_gain_score', 0):.1f}; "
            f"race DNF {diagnostics.get('race_dnf_bad_score', 0):.1f}; "
            f"sprint DNF {diagnostics.get('sprint_dnf_bad_score', 0):.1f}; "
            f"missing DNF rates - drivers {diagnostics.get('driver_dnf_rate_missing', 0)}, "
            f"constructors {diagnostics.get('constructor_dnf_rate_missing', 0)}"
        ),
    )
    st.write("DNF score source:", diagnostics.get("dnf_price_gain_score_source", "Unavailable"))
    st.write("Race -2 / Race -1 rounds:", diagnostics.get("recent_points_rounds", []))
    st.write("Race -2 / Race -1 circuits:", diagnostics.get("recent_points_circuits", []))
    st.write(
        "Recent points coverage:",
        (
            f"Drivers {diagnostics.get('recent_points_driver_complete', 0)}/"
            f"{diagnostics.get('recent_points_driver_total', 0)}, "
            f"Constructors {diagnostics.get('recent_points_constructor_complete', 0)}/"
            f"{diagnostics.get('recent_points_constructor_total', 0)}"
        ),
    )
    st.write(
        "Playerstats endpoint load:",
        (
            f"Loaded {diagnostics.get('playerstats_assets_loaded', 0)} assets, "
            f"failed {diagnostics.get('playerstats_assets_failed', 0)}"
        ),
    )
    st.write("Playerstats timeout failures:", diagnostics.get("playerstats_timeout_failures", 0))
    st.write("Playerstats skipped after failure limit:", diagnostics.get("playerstats_skipped_after_failure_limit", 0))
    if diagnostics.get("model_load_events"):
        st.markdown("**Model load event log**")
        for event in diagnostics.get("model_load_events", []):
            st.code(str(event))
    st.write("Fallback recent-point values used:", diagnostics.get("recent_points_fallback_used", False))
    transfer_diag = st.session_state.get("transfer_run_diagnostics", {}) or {}
    st.write("Transfer run duration (s):", f"{float(transfer_diag.get('transfer_run_duration_seconds', 0.0)):.2f}")
    st.write("Transfer generation duration (s):", f"{float(transfer_diag.get('transfer_generation_duration_seconds', 0.0)):.2f}")
    st.write("Transfer scoring duration (s):", f"{float(transfer_diag.get('transfer_scoring_duration_seconds', 0.0)):.2f}")
    st.write("Transfer total duration (s):", f"{float(transfer_diag.get('transfer_total_duration_seconds', 0.0)):.2f}")
    st.write("Transfer candidate count total:", int(transfer_diag.get("transfer_candidate_count_total", transfer_diag.get("transfer_candidate_count", 0)) or 0))
    st.write("Transfer candidates evaluated:", int(transfer_diag.get("transfer_candidates_evaluated", 0) or 0))
    st.write("Transfer candidates scored:", int(transfer_diag.get("transfer_candidates_scored", 0) or 0))
    st.write("Transfer candidates filtered:", int(transfer_diag.get("transfer_candidates_filtered", 0) or 0))
    st.write("Transfer search mode:", transfer_diag.get("search_mode", "unknown"))
    st.write("Transfer candidate pool mode:", transfer_diag.get("candidate_pool_mode", "unknown"))
    st.write("Candidate filter score used for prefilter:", bool(transfer_diag.get("candidate_filter_score_used_for_prefilter", False)))
    st.write(
        "Exhaustive scoring used after prefilter:",
        bool(transfer_diag.get("exhaustive_scoring_used_after_prefilter", False)),
    )
    st.write("Final objective score used for sorting:", bool(transfer_diag.get("final_score_used_for_sorting", False)))
    st.write(
        "Transfer candidate pools:",
        (
            f"incoming drivers {int(transfer_diag.get('incoming_driver_candidates', 0) or 0)}, "
            f"incoming constructors {int(transfer_diag.get('incoming_constructor_candidates', 0) or 0)}, "
            f"outgoing drivers {int(transfer_diag.get('outgoing_driver_candidates', 0) or 0)}, "
            f"outgoing constructors {int(transfer_diag.get('outgoing_constructor_candidates', 0) or 0)}"
        ),
    )
    st.write(
        "Transfer candidate pools kept:",
        (
            f"incoming drivers {int(transfer_diag.get('incoming_driver_candidates_kept', 0) or 0)}, "
            f"incoming constructors {int(transfer_diag.get('incoming_constructor_candidates_kept', 0) or 0)}, "
            f"outgoing drivers {int(transfer_diag.get('outgoing_driver_candidates_kept', 0) or 0)}, "
            f"outgoing constructors {int(transfer_diag.get('outgoing_constructor_candidates_kept', 0) or 0)}"
        ),
    )
    st.write(
        "Transfer candidate counts:",
        (
            f"before filtering {int(transfer_diag.get('exhaustive_candidate_count_before_filtering', 0) or 0)}, "
            f"after filtering {int(transfer_diag.get('candidate_count_after_filtering', 0) or 0)}, "
            f"valid plans generated {int(transfer_diag.get('valid_transfer_plans_generated', 0) or 0)}, "
            f"teams scored {int(transfer_diag.get('candidate_teams_scored', 0) or 0)}"
        ),
    )
    st.write("Transfer generated partial plans:", int(transfer_diag.get("generated_partial_plans", 0) or 0))
    st.write("Transfer fully scored candidates:", int(transfer_diag.get("evaluated_full_candidates", 0) or 0))
    st.write("Transfer pruned by candidate filtering:", int(transfer_diag.get("number_pruned_by_filtering", 0) or 0))
    st.write("Transfer duplicate teams skipped:", int(transfer_diag.get("duplicate_teams_skipped", 0) or 0))
    st.write("Transfer pruned by budget:", int(transfer_diag.get("pruned_by_budget", 0) or 0))
    st.write("Transfer pruned by beam:", int(transfer_diag.get("pruned_by_beam", 0) or 0))
    st.write("Transfer generated candidates by depth:", transfer_diag.get("generated_candidates_by_depth", {}))
    st.write("Transfer beam kept by depth:", transfer_diag.get("beam_kept_by_depth", {}))
    st.write("Transfer fully scored by depth:", transfer_diag.get("fully_scored_by_depth", {}))
    st.write("Final recommendations by transfer count:", transfer_diag.get("final_recommendations_by_transfer_count", {}))
    st.write("Recommendations by transfer count:", transfer_diag.get("recommendations_by_transfer_count", {}))
    st.write("Transfer results count:", int(transfer_diag.get("transfer_results_count", 0) or 0))
    st.write("Transfer stage order:", transfer_diag.get("transfer_stage_order", []))
    st.write("Transfer last error:", st.session_state.get("transfer_last_error") or "None")
    st.write(
        "Cache:",
        "Raw live data is retained for this browser session. Model controls recalculate from that snapshot; Refresh live data replaces it.",
    )
    technical_diagnostics.__exit__(None, None, None)
