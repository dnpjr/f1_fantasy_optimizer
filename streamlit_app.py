from __future__ import annotations

import html
import json
import logging
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
    build_transfer_recommendations,
    chip_mode_from_label,
    current_team_json,
    current_team_budget_from_selection,
    auto_budget_from_team_cost,
    resolve_budget_value,
    load_current_team_json_text,
    current_team_upload_summary,
    fantasy_asset_card_html,
    fantasy_card_grid_html,
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
    load_model_data,
    price_change_threshold_table,
    price_change_probability_matrix_table,
    price_change_projection_summary_table,
    price_change_target_summary_table,
    projected_team_value_from_budget,
    run_optimizer,
    selected_assets_price_gain,
    transfer_baseline,
    select_chip_boost_drivers,
    team_expected_points_with_chips,
    team_colour,
    validate_current_team,
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
            padding-top: 1.4rem;
            padding-bottom: 3rem;
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
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 0.9rem;
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
            padding: 1.05rem 1.2rem;
            margin: 0.8rem 0 1.1rem;
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
            font-size: clamp(1.35rem, 2.5vw, 2.15rem);
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
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-items: stretch;
            gap: 0.75rem;
            margin: 0.7rem 0 1rem;
        }
        .f1-driver-card {
            position: relative;
            width: 252px;
            min-height: 148px;
            border: 1px solid var(--f1-border);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.025));
            overflow: hidden;
            padding: 0.9rem;
            display: flex;
            flex-direction: column;
        }
        .f1-driver-card:before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 6px;
            background: var(--team-color, #64748b);
        }
        .f1-card-top {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            align-items: flex-start;
        }
        .f1-initials {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 999px;
            background: var(--team-color, #64748b);
            color: #fff;
            font-weight: 900;
            flex: 0 0 auto;
        }
        .f1-card-name {
            font-size: 0.98rem;
            font-weight: 850;
            line-height: 1.16;
            padding-left: 0.35rem;
            overflow-wrap: anywhere;
        }
        .f1-card-team {
            color: var(--f1-muted);
            font-size: 0.80rem;
            font-weight: 650;
            padding-left: 0.35rem;
            margin-top: 0.2rem;
        }
        .f1-card-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.45rem;
            margin-top: auto;
            padding-top: 0.75rem;
        }
        .f1-stat {
            background: rgba(255, 255, 255, 0.055);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 8px;
            padding: 0.45rem;
        }
        .f1-stat-label {
            color: var(--f1-muted);
            font-size: 0.68rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .f1-stat-value {
            color: #fff;
            font-weight: 850;
            margin-top: 0.1rem;
        }
        .f1-boost {
            position: absolute;
            top: 0.65rem;
            right: 4.1rem;
            background: var(--f1-red);
            color: #fff;
            border-radius: 999px;
            padding: 0.12rem 0.38rem;
            font-size: 0.64rem;
            font-weight: 900;
        }
        .f1-transfer-row {
            display: grid;
            grid-template-columns: 252px auto 252px;
            justify-content: center;
            gap: 0.7rem;
            align-items: center;
            margin-bottom: 0.75rem;
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
        @media (max-width: 720px) {
            .f1-race-card {
                grid-template-columns: 1fr;
            }
            .f1-card-grid {
                justify-content: center;
            }
            .f1-transfer-row {
                grid-template-columns: 1fr;
            }
            .f1-transfer-arrow {
                min-height: 1.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_dashboard_css()
LOGGER = logging.getLogger(__name__)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_cached_model_data(
    historical_seasons_back: int,
    current_season_weight: float,
    past_season_weight: float,
    recency_decay: float,
    upcoming_race_horizon: int,
):
    data = load_model_data(
        historical_seasons_back=historical_seasons_back,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        recency_decay=recency_decay,
        horizon_races=upcoming_race_horizon,
        include_playerstats=True,
    )
    return data.drivers, data.constructors, data.trends, data.diagnostics


def _option_labels(df: pd.DataFrame) -> dict[str, str]:
    return build_asset_option_labels(df)


def _load_current_team_config() -> dict:
    path = Path("data/current_team.json")
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
    if pd.isna(value):
        return ""
    numeric = float(value)
    if numeric > 0:
        return "background-color: rgba(46, 160, 67, 0.22);"
    if numeric < 0:
        return "background-color: rgba(248, 81, 73, 0.22);"
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


st.title("F1 Fantasy Optimiser")
st.caption(
    "Optimise your F1 Fantasy team using live prices, race history, price-change probabilities and transfer recommendations."
)

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh live data"):
        st.cache_data.clear()
        st.rerun()

    st.caption("Use the tabs below to edit settings and run the optimiser.")

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


def _sync_budget_from_current_team():
    st.session_state.app_budget = float(st.session_state.current_team_budget)
    st.session_state.optimizer_budget = float(st.session_state.current_team_budget)


def _sync_budget_from_optimizer():
    st.session_state.app_budget = float(st.session_state.optimizer_budget)
    st.session_state.current_team_budget = float(st.session_state.optimizer_budget)


def _mark_budget_manual_from_current_team():
    st.session_state.budget_user_overridden = True
    st.session_state.budget_init_mode = "manual_override"
    _sync_budget_from_current_team()


def _mark_budget_manual_from_optimizer():
    st.session_state.budget_user_overridden = True
    st.session_state.budget_init_mode = "manual_override"
    _sync_budget_from_optimizer()


(
    optimise_tab,
    price_changes_tab,
    current_team_tab,
    transfers_tab,
    locks_tab,
    model_settings_tab,
    diagnostics_tab,
) = st.tabs(
    [
        "Optimise",
        "Price changes",
        "Current team",
        "Transfers",
        "Locks and exclusions",
        "Model settings",
        "Diagnostics",
    ]
)

with model_settings_tab:
    st.markdown('<div class="f1-section-kicker">MODEL SETTINGS</div>', unsafe_allow_html=True)
    st.subheader("Model Settings")
    st.caption("Tune how much the model trusts current-season form, historical data and recency.")

    historical_seasons_back = st.number_input(
        "Historical seasons back",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
    )
    st.caption("How many complete previous seasons to include alongside the current season.")

    current_season_weight = st.slider(
        "Current-season weight",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.05,
    )
    st.caption("Scales the influence of completed races from the current season.")

    past_season_weight = st.slider(
        "Past-season weight",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05,
    )
    st.caption("Scales the influence of previous-season historical data.")

    recency_decay = st.slider(
        "Recency decay",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
    )
    st.caption("Controls how quickly older current-season races fade. Lower values emphasize the latest race more.")

    upcoming_race_horizon = st.number_input(
        "Upcoming race horizon",
        min_value=1,
        max_value=5,
        value=5,
        step=1,
    )
    st.caption("How many upcoming races to include in the expected-points horizon.")

load_ui = st.empty()
load_started = time.time()
with load_ui.container():
    load_status = st.status("Loading live market and model data...", expanded=True)
    load_progress = st.progress(0, text="Loading market feed")
    load_line = st.empty()

drivers = constructors = _trends = diagnostics = None

def _on_startup_progress(event: dict) -> None:
    nonlocal_text = str(event.get("message", "Loading..."))
    stage_name = str(event.get("stage_name", "Loading"))
    progress_raw = event.get("progress")
    if progress_raw is None:
        stage_index = int(event.get("stage_index", 0) or 0)
        stage_total = max(int(event.get("stage_total", 8) or 8), 1)
        progress_raw = stage_index / stage_total
    progress = max(0.0, min(1.0, float(progress_raw)))
    load_progress.progress(int(progress * 100), text=f"{stage_name}")
    text = f"{stage_name}: {nonlocal_text}"
    # Keep one status line updated in place.
    load_line.caption(text)

try:
    model_data = load_model_data(
        historical_seasons_back=int(historical_seasons_back),
        current_season_weight=float(current_season_weight),
        past_season_weight=float(past_season_weight),
        recency_decay=float(recency_decay),
        horizon_races=int(upcoming_race_horizon),
        include_playerstats=True,
        progress_callback=_on_startup_progress,
    )
    drivers, constructors, _trends, diagnostics = (
        model_data.drivers,
        model_data.constructors,
        model_data.trends,
        model_data.diagnostics,
    )
    elapsed = time.time() - load_started
    load_progress.progress(100, text=f"Ready in {elapsed:.1f}s")
    load_status.update(label=f"Data loaded in {elapsed:.1f}s", state="complete", expanded=False)
    st.session_state["last_good_model_payload"] = {
        "drivers": drivers,
        "constructors": constructors,
        "trends": _trends,
        "diagnostics": diagnostics,
    }
    st.session_state["last_load_error"] = None
except Exception as exc:
    LOGGER.exception("Live model/data load failed")
    st.session_state["last_load_error"] = str(exc)
    cached_payload = st.session_state.get("last_good_model_payload")
    if isinstance(cached_payload, dict):
        drivers = cached_payload.get("drivers")
        constructors = cached_payload.get("constructors")
        _trends = cached_payload.get("trends")
        diagnostics = dict(cached_payload.get("diagnostics") or {})
        diagnostics["last_load_error"] = str(exc)
        diagnostics["last_load_fallback_used"] = True
        load_progress.progress(100, text="Using last loaded data")
        load_status.update(label="Live refresh failed. Using last loaded data.", state="running", expanded=False)
        st.warning("Live refresh failed. Using last loaded data.")
    else:
        load_status.update(label="Data loading failed", state="error", expanded=False)
        st.error("Could not load live data. Try Refresh live data, or try again later.")
        st.stop()
finally:
    load_ui.empty()

if drivers is None or constructors is None or diagnostics is None:
    st.error("Could not load live data. Try Refresh live data, or try again later.")
    st.stop()

driver_labels = _option_labels(drivers)
constructor_labels = _option_labels(constructors)
current_team_config = _load_current_team_config()
if "chip_mode_label" not in st.session_state:
    st.session_state.chip_mode_label = "None"
chip_mode = chip_mode_from_label(st.session_state.chip_mode_label)

preloaded_driver_ids = [str(x) for x in current_team_config.get("drivers", []) if str(x) in driver_labels]
preloaded_constructor_ids = [str(x) for x in current_team_config.get("constructors", []) if str(x) in constructor_labels]
preloaded_bank = float(current_team_config.get("bank", 0.0))
if "current_team_driver_ids" not in st.session_state:
    st.session_state.current_team_driver_ids = preloaded_driver_ids
if "current_team_constructor_ids" not in st.session_state:
    st.session_state.current_team_constructor_ids = preloaded_constructor_ids
if "current_team_free_transfers" not in st.session_state:
    st.session_state.current_team_free_transfers = int(current_team_config.get("free_transfers", 2))
if "current_team_bank" not in st.session_state:
    st.session_state.current_team_bank = preloaded_bank

if not st.session_state.budget_user_overridden:
    preload_driver_frame = drivers[drivers["id"].astype(str).isin(st.session_state.current_team_driver_ids)]
    preload_constructor_frame = constructors[constructors["id"].astype(str).isin(st.session_state.current_team_constructor_ids)]
    preload_budget = current_team_budget_from_selection(
        preload_driver_frame,
        preload_constructor_frame,
        bank=float(st.session_state.current_team_bank),
    )
    if preload_budget <= 0 and (len(st.session_state.current_team_driver_ids) + len(st.session_state.current_team_constructor_ids)) > 0:
        preload_budget = max(float(st.session_state.get("app_budget", 100.0) or 100.0), 100.0)
    if preload_budget > 0:
        st.session_state.current_team_budget = float(preload_budget)
        st.session_state.optimizer_budget = float(preload_budget)
        st.session_state.app_budget = float(preload_budget)
        st.session_state.budget_init_team_cost = float(
            pd.to_numeric(preload_driver_frame.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
            + pd.to_numeric(preload_constructor_frame.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        )
        st.session_state.budget_init_bank = float(st.session_state.current_team_bank)
        st.session_state.budget_init_mode = "auto_from_current_team"
        st.session_state.budget_auto_signature = (
            tuple(sorted(st.session_state.current_team_driver_ids)),
            tuple(sorted(st.session_state.current_team_constructor_ids)),
            round(float(st.session_state.current_team_bank), 4),
        )

_render_race_header(diagnostics)
if diagnostics.get("playerstats_timeout_failures", 0):
    st.warning(
        f"Playerstats timeouts detected: {int(diagnostics.get('playerstats_timeout_failures', 0))}. "
        "The app will continue with partial data where needed."
    )
if diagnostics.get("playerstats_skipped_after_failure_limit", 0):
    st.warning(
        f"Playerstats requests skipped after repeated failures: "
        f"{int(diagnostics.get('playerstats_skipped_after_failure_limit', 0))}."
    )

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
    current_team_drivers = apply_probabilistic_price_change_model(
        current_team_input_drivers,
        cheap_driver_rules,
        expensive_rules=expensive_driver_rules,
        expensive_price_min=driver_expensive_min,
        bounds=price_change_bounds,
        predicted_points_col="next_race_expected_points",
    )
    current_team_constructors = apply_probabilistic_price_change_model(
        current_team_input_constructors,
        cheap_constructor_rules,
        expensive_rules=expensive_constructor_rules,
        expensive_price_min=constructor_expensive_min,
        bounds=price_change_bounds,
        predicted_points_col="next_race_expected_points",
    )

    st.caption("Build a current_team.json-style squad and check it against the same budget used by the optimiser.")
    with st.expander("Advanced: JSON import / export", expanded=False):
        st.write(
            "Use this to save or restore your F1 Fantasy squad. Importing a current_team.json file fills the selected drivers, constructors, bank, free transfers, and budget. Exporting downloads the currently selected squad in the same format."
        )
        uploaded_team_file = st.file_uploader("Upload current_team.json", type=["json"], key="current_team_upload_file")
        json_tools_container = st.container()
    uploaded_team_summary = None
    if uploaded_team_file is not None:
        try:
            uploaded_payload = load_current_team_json_text(uploaded_team_file.getvalue().decode("utf-8"))
            uploaded_team_summary = current_team_upload_summary(uploaded_payload, driver_labels, constructor_labels)
        except Exception as exc:
            st.error(f"Could not read current_team.json: {exc}")
        else:
            if uploaded_team_summary["missing_drivers"]:
                st.warning(f"Missing driver ids not found in the current roster: {uploaded_team_summary['missing_drivers']}")
            if uploaded_team_summary["missing_constructors"]:
                st.warning(f"Missing constructor ids not found in the current roster: {uploaded_team_summary['missing_constructors']}")
            if uploaded_team_summary["drivers"] or uploaded_team_summary["constructors"]:
                st.success("Loaded current_team.json.")

    current_team_source = dict(current_team_config)
    if uploaded_team_summary is not None:
        current_team_source.update(
            {
                "drivers": uploaded_team_summary["drivers"],
                "constructors": uploaded_team_summary["constructors"],
                "bank": uploaded_team_summary["bank"],
                "free_transfers": uploaded_team_summary["free_transfers"],
            }
        )

    default_driver_ids = [str(x) for x in current_team_source.get("drivers", []) if str(x) in driver_labels]
    default_constructor_ids = [str(x) for x in current_team_source.get("constructors", []) if str(x) in constructor_labels]
    default_bank = float(current_team_source.get("bank", 0.0))
    default_free_transfers = int(current_team_source.get("free_transfers", 2))
    if "current_team_driver_ids" not in st.session_state or uploaded_team_summary is not None:
        st.session_state.current_team_driver_ids = default_driver_ids
    if "current_team_constructor_ids" not in st.session_state or uploaded_team_summary is not None:
        st.session_state.current_team_constructor_ids = default_constructor_ids
    if "current_team_free_transfers" not in st.session_state or uploaded_team_summary is not None:
        st.session_state.current_team_free_transfers = default_free_transfers
    if "current_team_bank" not in st.session_state or uploaded_team_summary is not None:
        st.session_state.current_team_bank = default_bank

    if uploaded_team_summary is not None:
        uploaded_driver_frame = current_team_drivers[current_team_drivers["id"].astype(str).isin(uploaded_team_summary["drivers"])]
        uploaded_constructor_frame = current_team_constructors[current_team_constructors["id"].astype(str).isin(uploaded_team_summary["constructors"])]
        uploaded_budget = current_team_budget_from_selection(uploaded_driver_frame, uploaded_constructor_frame, bank=uploaded_team_summary["bank"])
        st.session_state.current_team_budget = uploaded_budget
        st.session_state.optimizer_budget = uploaded_budget
        st.session_state.app_budget = uploaded_budget
        st.session_state.budget_init_team_cost = float(
            pd.to_numeric(uploaded_driver_frame.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
            + pd.to_numeric(uploaded_constructor_frame.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        )
        st.session_state.budget_init_bank = float(uploaded_team_summary["bank"])
        st.session_state.budget_init_mode = "auto_from_uploaded_current_team"
        st.session_state.budget_user_overridden = False
        st.session_state.budget_auto_signature = (
            tuple(sorted(st.session_state.current_team_driver_ids)),
            tuple(sorted(st.session_state.current_team_constructor_ids)),
            round(float(st.session_state.current_team_bank), 4),
        )

    selected_driver_frame = current_team_drivers[current_team_drivers["id"].astype(str).isin(st.session_state.current_team_driver_ids)]
    selected_constructor_frame = current_team_constructors[current_team_constructors["id"].astype(str).isin(st.session_state.current_team_constructor_ids)]
    selection_signature = (
        tuple(sorted(st.session_state.current_team_driver_ids)),
        tuple(sorted(st.session_state.current_team_constructor_ids)),
        round(float(st.session_state.current_team_bank), 4),
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
        user_overridden=bool(st.session_state.budget_user_overridden),
    )
    if (
        not st.session_state.budget_user_overridden
        and (
            st.session_state.budget_auto_signature != selection_signature
            or st.session_state.get("current_team_budget") != resolved_budget
        )
    ):
        st.session_state.current_team_budget = resolved_budget
        st.session_state.optimizer_budget = resolved_budget
        st.session_state.app_budget = resolved_budget
        st.session_state.budget_init_team_cost = float(auto_budget_target - float(st.session_state.current_team_bank))
        st.session_state.budget_init_bank = float(st.session_state.current_team_bank)
        st.session_state.budget_init_mode = "auto_from_selected_team"
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
        with selector_col1:
            selected_current_driver_ids = st.multiselect(
                "Current drivers",
                options=list(driver_labels.keys()),
                format_func=driver_labels.get,
                key="current_team_driver_ids",
            )
        with selector_col2:
            selected_current_constructor_ids = st.multiselect(
                "Current constructors",
                options=list(constructor_labels.keys()),
                format_func=constructor_labels.get,
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
    remaining_current_budget = effective_current_budget - current_validation["total_cost"]
    expected_current_price_gain = selected_assets_price_gain(
        current_validation["selected_drivers"],
        current_validation["selected_constructors"],
    )
    projected_current_team_value = projected_team_value_from_budget(float(current_budget), expected_current_price_gain)
    cur_cols = st.columns(5)
    cur_cols[0].metric("Team cost", format_money(current_validation["total_cost"]))
    cur_cols[1].metric("Remaining budget", format_money(remaining_current_budget))
    cur_cols[2].metric("Expected points", format_points(current_expected_points))
    cur_cols[3].metric("Expected price gain", format_signed_money(expected_current_price_gain))
    cur_cols[4].metric("Projected team value", format_money(projected_current_team_value))

    for msg in current_validation["errors"]:
        st.error(msg)
    for msg in current_validation["warnings"]:
        st.warning(msg)
    if current_validation["valid"]:
        st.success("Current team shape is valid.")

    st.markdown("**Fantasy cards**")
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
        )
        locked_constructor_ids = st.multiselect(
            "Locked constructors",
            options=list(constructor_labels.keys()),
            format_func=constructor_labels.get,
        )
    with col2:
        excluded_driver_ids = st.multiselect(
            "Excluded drivers",
            options=list(driver_labels.keys()),
            format_func=driver_labels.get,
        )
        excluded_constructor_ids = st.multiselect(
            "Excluded constructors",
            options=list(constructor_labels.keys()),
            format_func=constructor_labels.get,
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
                "Price gain weight",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                disabled=transfer_objective not in {OBJECTIVE_COMBINED, OBJECTIVE_RISK_ADJUSTED_COMBINED},
                key="transfer_price_gain_weight_slider",
            )

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

        transfer_signature = (
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

            transfer_drivers = apply_no_negative_scores(current_team_drivers) if chip_mode == CHIP_NO_NEGATIVE else current_team_drivers
            transfer_constructors = apply_no_negative_scores(current_team_constructors) if chip_mode == CHIP_NO_NEGATIVE else current_team_constructors
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
    st.markdown('<div class="f1-section-kicker">BUDGET BUILDER</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### Price Change Targets / Budget Builder")
        st.caption(
            "See what each asset needs to score to move price, then compare it with the model’s expected price gain."
        )
        header_cols = st.columns([2.2, 1.2])
        with header_cols[0]:
            st.caption(format_next_race_header(diagnostics.get("next_race_name"), diagnostics.get("next_race_date")))
            st.caption(
                "Terrible, Poor, Good, and Great are price-change tiers based on rolling 3-race points per million. "
                "The target table shows how many fantasy points each asset needs next race to fall into each tier."
            )
        with header_cols[1]:
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

    target_container = st.container()
    projection_container = st.container()

    price_change_drivers = edited_drivers.copy()
    price_change_constructors = edited_constructors.copy()
    driver_price_change_table = _price_change_display_table(
        price_change_drivers,
        cheap_driver_rules,
        "drivers",
        expensive_rules=expensive_driver_rules,
        expensive_price_min=driver_expensive_min,
        bounds=price_change_bounds,
    )
    constructor_price_change_table = _price_change_display_table(
        price_change_constructors,
        cheap_constructor_rules,
        "constructors",
        expensive_rules=expensive_constructor_rules,
        expensive_price_min=constructor_expensive_min,
        bounds=price_change_bounds,
    )
    driver_probability_matrix = _price_change_probability_matrix(
        price_change_drivers,
        cheap_driver_rules,
        "drivers",
        expensive_rules=expensive_driver_rules,
        expensive_price_min=driver_expensive_min,
        bounds=price_change_bounds,
    )
    constructor_probability_matrix = _price_change_probability_matrix(
        price_change_constructors,
        cheap_constructor_rules,
        "constructors",
        expensive_rules=expensive_constructor_rules,
        expensive_price_min=constructor_expensive_min,
        bounds=price_change_bounds,
    )

    with target_container:
        st.caption(
            "Rise difficulty: Lower is better. This is the number of next-race points needed for a strong price-rise outcome, divided by current price. Negative means the asset can still rise even with a low score."
        )
        price_driver_tab, price_constructor_tab = st.tabs(["Drivers", "Constructors"])
        with price_driver_tab:
            st.dataframe(_price_change_table_styler(driver_price_change_table), hide_index=True, width="stretch")
        with price_constructor_tab:
            st.dataframe(_price_change_table_styler(constructor_price_change_table), hide_index=True, width="stretch")

    with projection_container:
        st.subheader("Model Projection")
        st.caption(
            "This combines the one-race expected-points model with volatility and DNF risk to estimate each asset’s probability of landing in each price-change tier."
        )
        prob_driver_tab, prob_constructor_tab = st.tabs(["Drivers", "Constructors"])
        with prob_driver_tab:
            st.dataframe(_price_change_table_styler(driver_probability_matrix), hide_index=True, width="stretch")
        with prob_constructor_tab:
            st.dataframe(_price_change_table_styler(constructor_probability_matrix), hide_index=True, width="stretch")

        with st.expander("Advanced / probability details", expanded=False):
            detail_driver_tab, detail_constructor_tab = st.tabs(["Drivers", "Constructors"])
            with detail_driver_tab:
                detail = _price_change_probability_detail_table(
                    price_change_drivers,
                    cheap_driver_rules,
                    "drivers",
                    expensive_rules=expensive_driver_rules,
                    expensive_price_min=driver_expensive_min,
                    bounds=price_change_bounds,
                )
                st.dataframe(_price_change_table_styler(detail), hide_index=True, width="stretch")
            with detail_constructor_tab:
                detail = _price_change_probability_detail_table(
                    price_change_constructors,
                    cheap_constructor_rules,
                    "constructors",
                    expensive_rules=expensive_constructor_rules,
                    expensive_price_min=constructor_expensive_min,
                    bounds=price_change_bounds,
                )
                st.dataframe(_price_change_table_styler(detail), hide_index=True, width="stretch")

with optimise_tab:
    st.markdown('<div class="f1-section-kicker">OPTIMISER COCKPIT</div>', unsafe_allow_html=True)
    st.caption("Build the strongest team for your budget, chip choice and objective.")
    with st.container(border=True):
        st.subheader("Optimiser Settings")
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        with opt_col1:
            st.number_input(
                "Budget",
                min_value=0.0,
                step=0.1,
                key="optimizer_budget",
                on_change=_mark_budget_manual_from_optimizer,
            )
            budget = float(st.session_state.optimizer_budget)
        with opt_col2:
            top_k = st.number_input("Number of teams", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)
        with opt_col3:
            st.selectbox(
                "Chips applied",
                options=["None", "3x chip", "Limitless", "No Negative chip"],
                key="chip_mode_label",
            )
            chip_mode = chip_mode_from_label(st.session_state.chip_mode_label)
        if chip_mode == CHIP_LIMITLESS:
            st.info("Limitless is a temporary team chip, so budget growth and projected team value are not applied.")
        elif chip_mode == CHIP_NO_NEGATIVE:
            st.info("No Negative floors negative driver and constructor scores at 0.")

        st.subheader("Objective")
        objective_col1, objective_col2 = st.columns([2, 1])
        with objective_col1:
            objective_mode = st.selectbox(
                "Optimisation objective",
                options=[OBJECTIVE_POINTS_ONLY, OBJECTIVE_PRICE_GROWTH_ONLY, OBJECTIVE_COMBINED, OBJECTIVE_RISK_ADJUSTED_COMBINED],
                index=0,
                disabled=chip_mode == CHIP_LIMITLESS,
            )
            if chip_mode == CHIP_LIMITLESS:
                objective_mode = OBJECTIVE_POINTS_ONLY
        with objective_col2:
            price_gain_weight = st.slider(
                "Price gain weight",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                disabled=objective_mode not in {OBJECTIVE_COMBINED, OBJECTIVE_RISK_ADJUSTED_COMBINED} or chip_mode == CHIP_LIMITLESS,
                key="optimise_price_gain_weight_slider",
            )

        run_clicked = st.button("Run optimiser", type="primary", use_container_width=True)

    if run_clicked:
        try:
            optimiser_input_drivers = apply_no_negative_scores(price_change_drivers) if chip_mode == CHIP_NO_NEGATIVE else price_change_drivers
            optimiser_input_constructors = apply_no_negative_scores(price_change_constructors) if chip_mode == CHIP_NO_NEGATIVE else price_change_constructors
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
            solutions = run_optimizer(
                optimizer_drivers,
                optimizer_constructors,
                budget=optimiser_budget,
                top_k=top_k,
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
            st.error("The optimiser could not run with the selected inputs.")
            st.warning("Try relaxing locks/exclusions, changing chip/objective settings, or refreshing live data.")
            st.session_state["last_optimiser_error"] = str(exc)
        else:
            st.session_state["last_optimiser_error"] = None
            if not solutions:
                st.warning("No valid team found. Try relaxing locks/exclusions or increasing the budget.")
            else:
                best = solutions[0]
                best_summary = _team_summary(best, budget)

                st.subheader("Best Team")
                metric_cols = st.columns(5)
                metric_cols[0].metric("Team cost", format_money(best_summary["Total cost"]))
                metric_cols[1].metric("Remaining budget", "N/A" if chip_mode == CHIP_LIMITLESS else format_money(best_summary["Remaining budget"]))
                metric_cols[2].metric("Expected points", format_points(best_summary["Expected points"]))
                if chip_mode == CHIP_LIMITLESS:
                    metric_cols[3].metric("Expected price gain", "N/A")
                    metric_cols[4].metric("Projected team value", "N/A")
                else:
                    metric_cols[3].metric("Expected price gain", format_signed_money(best_summary["Expected price gain"]))
                    metric_cols[4].metric("Projected team value", format_money(best_summary["Projected team value"]))
                if chip_mode == CHIP_TRIPLE:
                    st.caption(f"2x driver: {best.boosted_driver or 'None'} | 3x driver: {best.triple_driver or 'None'}")
                else:
                    st.caption(f"2x driver: {best.boosted_driver or 'None'}")

                st.markdown("**Fantasy cards**")
                _asset_card_grid(best.drivers, boosted_driver=best.boosted_driver, triple_driver=best.triple_driver, asset_label="Driver")
                _asset_card_grid(best.constructors, asset_label="Constructor")

                if len(solutions) > 1:
                    st.subheader("Alternative Teams")
                    for idx, sol in enumerate(solutions[1:], start=2):
                        with st.expander(f"Team #{idx} - {_team_header(sol, budget)}"):
                            alt_metrics = st.columns(3)
                            alt_summary = _team_summary(sol, budget)
                            alt_metrics[0].metric("Expected points", format_points(alt_summary["Expected points"]))
                            alt_metrics[1].metric("Expected price gain", format_signed_money(alt_summary["Expected price gain"]))
                            alt_metrics[2].metric("Projected team value", format_money(alt_summary["Projected team value"]))
                            _asset_card_grid(sol.drivers, boosted_driver=sol.boosted_driver, triple_driver=sol.triple_driver, asset_label="Driver")
                            _asset_card_grid(sol.constructors, asset_label="Constructor")

with diagnostics_tab:
    st.markdown('<div class="f1-section-kicker">DATA CHECKS</div>', unsafe_allow_html=True)
    st.subheader("Diagnostics")
    st.caption("Check feed status, playerstats coverage, deadline source and model assumptions.")
    diag_cols = st.columns(4)
    diag_cols[0].metric("Season", diagnostics["current_season"])
    diag_cols[1].metric("Fantasy feed", diagnostics["feed_round"])
    diag_cols[2].metric("Drivers", diagnostics["driver_count"])
    diag_cols[3].metric("Constructors", diagnostics["constructor_count"])

    settings_cols = st.columns(5)
    settings_cols[0].metric("Historical seasons", diagnostics["historical_seasons_back"])
    settings_cols[1].metric("Current weight", f"{diagnostics['current_season_weight']:.2f}")
    settings_cols[2].metric("Past weight", f"{diagnostics['past_season_weight']:.2f}")
    settings_cols[3].metric("Recency decay", f"{diagnostics['recency_decay']:.2f}")
    settings_cols[4].metric("Race horizon", diagnostics["upcoming_race_horizon"])

    st.write("Upcoming circuits:", ", ".join(diagnostics["upcoming_circuits"]))
    st.write("Model load started (UTC):", diagnostics.get("model_load_started_utc", "Unavailable"))
    st.write("Model load finished (UTC):", diagnostics.get("model_load_finished_utc", "Unavailable"))
    st.write("Model load duration (s):", f"{float(diagnostics.get('model_load_duration_seconds', 0.0)):.2f}")
    st.write("Last load fallback used:", bool(diagnostics.get("last_load_fallback_used", False)))
    st.write("Last load error:", diagnostics.get("last_load_error") or st.session_state.get("last_load_error") or "None")
    st.write("Last optimiser error:", st.session_state.get("last_optimiser_error") or "None")
    st.write("Playerstats prefetch enabled:", diagnostics.get("playerstats_prefetch_enabled", False))
    st.write("Budget init mode:", st.session_state.get("budget_init_mode", "unknown"))
    st.write("Budget init team cost:", format_money(st.session_state.get("budget_init_team_cost")))
    st.write("Budget init bank:", format_money(st.session_state.get("budget_init_bank")))
    st.write("Budget manually overridden:", bool(st.session_state.get("budget_user_overridden", False)))
    st.write("Team lock deadline (UTC):", diagnostics.get("team_lock_deadline_utc") or "Unavailable")
    st.write("Team lock source:", diagnostics.get("team_lock_deadline_source", "Unavailable"))
    st.write("Team lock raw field:", diagnostics.get("team_lock_deadline_raw_field") or "Unavailable")
    st.write("Team lock raw value:", diagnostics.get("team_lock_deadline_raw_value") or "Unavailable")
    st.write("Team lock timezone assumption:", diagnostics.get("team_lock_timezone_assumption", "Unavailable"))
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
        with st.expander("Model load event log", expanded=False):
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
    st.write("Cache:", "Live/model data is cached for 1 hour. Use Refresh live data to clear it.")
