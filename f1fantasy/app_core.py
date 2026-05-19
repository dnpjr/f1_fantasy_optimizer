from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
import difflib
import html
import json
import math
import re
import unicodedata
from typing import Any, Callable

import pandas as pd

from f1fantasy.ergast import fetch_all_supporting, fetch_schedule
from f1fantasy.fantasy_api import _latest_feed_round, fetch_players, fetch_teams
from f1fantasy.model import (
    _horizon_weights,
    _constructor_round_points,
    apply_no_negative_expectation,
    compute_weekend_points,
    expected_scores_horizon,
)
from f1fantasy.optimize import TeamSolution, optimize_top_k
from f1fantasy.player_stats import (
    PLAYERSTATS_ENDPOINT_PATTERN,
    fetch_team_lock_deadline_from_playerstats,
    fetch_recent_points_for_roster,
    latest_two_races,
)


DEFAULT_HISTORICAL_SEASONS_BACK = 2
DEFAULT_UPCOMING_RACE_HORIZON = 5
DEFAULT_TOP_K = 1
OBJECTIVE_POINTS_ONLY = "Points only"
OBJECTIVE_PRICE_GROWTH_ONLY = "Price growth only"
OBJECTIVE_COMBINED = "Combined points + price growth"
OBJECTIVE_RISK_ADJUSTED_COMBINED = "Risk-adjusted combined"
CHIP_NONE = "none"
CHIP_TRIPLE = "triple"
CHIP_LIMITLESS = "limitless"
CHIP_NO_NEGATIVE = "no_negative"


@dataclass
class ModelData:
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    trends: pd.DataFrame
    diagnostics: dict


@dataclass(frozen=True)
class PriceChangeRules:
    terrible_max: float
    poor_min: float
    poor_max: float
    good_min: float
    good_max: float
    great_min: float
    terrible_price_change: float
    poor_price_change: float
    good_price_change: float
    great_price_change: float


@dataclass(frozen=True)
class PriceChangeBounds:
    min_asset_price: float = 3.0
    max_asset_price: float = 34.0


DEFAULT_PRICE_CHANGE_BOUNDS = PriceChangeBounds(min_asset_price=3.0, max_asset_price=34.0)
DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF = 18.5
DEFAULT_PRICE_GAIN_VOLATILITY_FALLBACK = 1.0
DEFAULT_RACE_DNF_BAD_SCORE = -20.0
DEFAULT_SPRINT_DNF_BAD_SCORE = -10.0
# Official/repo scoring uses -20 for race DNF and -10 for sprint DNF.
# The price-change model is a whole race-weekend projection, so this generic
# bad-outcome score is deliberately a little harsher than race DNF alone.
DEFAULT_DNF_PRICE_GAIN_SCORE = -30.0
DEFAULT_DRIVER_SCORE_VOLATILITY_FLOOR = 5.0
DEFAULT_CONSTRUCTOR_SCORE_VOLATILITY_FLOOR = 8.0
DEFAULT_PRICE_CHANGE_CHEAP_RULES = PriceChangeRules(
    # Community-calibrated avgPPM bands from Canada-style price-change tables.
    # These are editable in code, not official hidden F1 Fantasy thresholds.
    terrible_max=0.60,
    poor_min=0.60,
    poor_max=0.90,
    good_min=0.90,
    good_max=1.20,
    great_min=1.20,
    terrible_price_change=-0.6,
    poor_price_change=-0.2,
    good_price_change=0.2,
    great_price_change=0.6,
)
DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES = PriceChangeRules(
    # Same avgPPM bands as cheap assets; only price movements differ by tier.
    terrible_max=0.60,
    poor_min=0.60,
    poor_max=0.90,
    good_min=0.90,
    good_max=1.20,
    great_min=1.20,
    terrible_price_change=-0.3,
    poor_price_change=-0.1,
    good_price_change=0.1,
    great_price_change=0.3,
)

PRICE_BAND_STYLES = {
    "Terrible": "background-color: rgba(127, 29, 29, 0.72); color: #ffffff;",
    "Poor": "background-color: rgba(248, 113, 113, 0.42); color: #ffffff;",
    "Good": "background-color: rgba(132, 204, 22, 0.34); color: #ffffff;",
    "Great": "background-color: rgba(22, 163, 74, 0.48); color: #ffffff;",
}

TEAM_COLOURS = {
    "ferrari": "#dc2626",
    "mclaren": "#f97316",
    "mercedes": "#14b8a6",
    "red bull racing": "#1e3a8a",
    "red bull": "#1e3a8a",
    "williams": "#2563eb",
    "aston martin": "#15803d",
    "alpine": "#ec4899",
    "haas f1 team": "#6b7280",
    "haas": "#6b7280",
    "racing bulls": "#3b82f6",
    "rb": "#3b82f6",
    "audi": "#14532d",
    "sauber": "#14532d",
    "cadillac": "#64748b",
}
DEFAULT_TEAM_COLOUR = "#64748b"
USER_HIDDEN_COLUMNS = {"team_colour"}


def _normalize_display_zero(value: float | int | None, threshold: float = 0.005) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    return 0.0 if abs(number) < float(threshold) else number


def _canon(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    return re.sub(r"\s+", " ", s)


def _canon_team(s: str) -> str:
    s = _canon(s)
    for tok in ["f1 team", "formula 1 team", "team", "scuderia", "gp", "grand prix"]:
        s = s.replace(tok, " ")
    return re.sub(r"\s+", " ", s).strip()


def team_colour(team_name: str | None) -> str:
    key = _canon_team(team_name or "")
    if key in TEAM_COLOURS:
        return TEAM_COLOURS[key]
    for alias, colour in TEAM_COLOURS.items():
        if alias in key or key in alias:
            return colour
    return DEFAULT_TEAM_COLOUR


def hide_user_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop implementation-only columns before rendering user-facing tables."""
    return df.drop(columns=[col for col in USER_HIDDEN_COLUMNS if col in df.columns], errors="ignore")


def format_money(value: float | int | None) -> str:
    numeric = _normalize_display_zero(value)
    if numeric is None:
        return "-"
    return f"{float(numeric):.2f}M"


def format_signed_money(value: float | int | None) -> str:
    numeric = _normalize_display_zero(value)
    if numeric is None:
        return "-"
    if numeric == 0.0:
        return "0.00M"
    return f"{float(numeric):+.2f}M"


def format_points(value: float | int | None) -> str:
    numeric = _normalize_display_zero(value)
    if numeric is None:
        return "-"
    return f"{float(numeric):.2f}"


def format_signed_points(value: float | int | None) -> str:
    numeric = _normalize_display_zero(value)
    if numeric is None:
        return "-"
    if numeric == 0.0:
        return "0.00"
    return f"{float(numeric):+.2f}"


def format_probability(value: float | int | None) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric) * 100:.1f}%"


def adjust_money_value(value: float | int | None, delta: float, min_value: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    base = float(numeric) if pd.notna(numeric) else 0.0
    return max(float(min_value), base + float(delta))


def auto_budget_from_team_cost(team_cost: float, bank: float) -> float:
    return max(0.0, float(team_cost) + float(bank))


def resolve_budget_value(
    current_budget: float | int | None,
    team_cost: float,
    bank: float,
    user_overridden: bool,
) -> float:
    if user_overridden:
        numeric = pd.to_numeric(current_budget, errors="coerce")
        return float(numeric) if pd.notna(numeric) else auto_budget_from_team_cost(team_cost, bank)
    return auto_budget_from_team_cost(team_cost, bank)


def build_asset_option_labels(df: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in df[["id", "name", "price"]].itertuples(index=False):
        labels[str(row.id)] = f"{row.name} ({format_money(row.price)})"
    return labels


def _best_fuzzy(target: str, candidates: list[str], cutoff: float = 0.6) -> str | None:
    if not candidates:
        return None
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def upcoming_circuits(schedule: pd.DataFrame, today: str, n: int = 5) -> list[str]:
    sch = schedule.copy()
    sch["date"] = sch["date"].astype(str)
    upcoming = sch[sch["date"] >= today].sort_values("round").head(n)
    return [c.split(" Circuit")[0].strip() for c in upcoming["circuitName"].astype(str).tolist()]


def format_next_race_header(race_name: str | None = None, race_date: str | None = None) -> str:
    parts: list[str] = []
    if race_name:
        parts.append(str(race_name).strip())
    if race_date:
        date_text = str(race_date).strip()
        try:
            parsed = datetime.strptime(date_text, "%Y-%m-%d")
            date_text = parsed.strftime("%-d %b %Y")
        except Exception:
            pass
        if date_text:
            parts.append(date_text)
    if parts:
        return "Next race: " + ", ".join(parts)
    return "Next race"


def _parse_schedule_datetime(date_value: str | None, time_value: str | None = None) -> datetime | None:
    date_text = str(date_value or "").strip()
    if not date_text:
        return None
    time_text = str(time_value or "").strip()
    raw = f"{date_text}T{time_text}" if time_text else f"{date_text}T00:00:00Z"
    raw = raw.replace(" ", "")
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def derive_team_lock_deadline(next_race_row: dict | pd.Series | None) -> tuple[datetime | None, str]:
    """Choose team-lock timestamp from schedule fields (sprint first, then qualifying)."""
    if next_race_row is None:
        return None, "unavailable"
    row = next_race_row if isinstance(next_race_row, dict) else next_race_row.to_dict()
    sprint_dt = _parse_schedule_datetime(row.get("sprint_date"), row.get("sprint_time"))
    if sprint_dt is not None:
        return sprint_dt, "schedule_derived_sprint_start"
    qualifying_dt = _parse_schedule_datetime(row.get("qualifying_date"), row.get("qualifying_time"))
    if qualifying_dt is not None:
        return qualifying_dt, "schedule_derived_qualifying_start"
    return None, "unavailable"


def format_countdown(target_utc: datetime | None, now_utc: datetime | None = None) -> str:
    if target_utc is None:
        return "Team lock deadline unavailable"
    now = now_utc or datetime.now(UTC)
    delta_seconds = int((target_utc - now).total_seconds())
    if delta_seconds <= 0:
        return "LOCKED"
    days = delta_seconds // 86400
    hours = (delta_seconds % 86400) // 3600
    minutes = (delta_seconds % 3600) // 60
    return f"{days:02d}D : {hours:02d}H : {minutes:02d}M"


def parse_team_lock_deadline_timestamp(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def load_model_data(
    current_season: int | None = None,
    today: str | None = None,
    historical_seasons_back: int = DEFAULT_HISTORICAL_SEASONS_BACK,
    horizon_races: int = DEFAULT_UPCOMING_RACE_HORIZON,
    current_season_weight: float = 1.0,
    past_season_weight: float = 1.0,
    recency_decay: float = 0.95,
    include_playerstats: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ModelData:
    """Load live fantasy prices and model assumptions for the Streamlit app."""
    load_started = datetime.now(UTC)
    load_events: list[str] = []

    def _log(event: str) -> None:
        load_events.append(f"{datetime.now(UTC).isoformat()} - {event}")

    def _emit(
        stage_index: int,
        stage_name: str,
        message: str,
        progress: float | None = None,
        status: str = "running",
        details: dict[str, Any] | None = None,
    ) -> None:
        if progress_callback is None:
            return
        payload: dict[str, Any] = {
            "stage_index": int(stage_index),
            "stage_total": 8,
            "stage_name": stage_name,
            "message": message,
            "progress": float(progress) if progress is not None else None,
            "status": status,
        }
        if details:
            payload.update(details)
        try:
            progress_callback(payload)
        except Exception:
            pass

    _log("load_model_data_start")
    _emit(1, "Loading market feed", "Loading market feed...", progress=0.05)
    now = datetime.now(UTC)
    current_season = int(current_season or now.year)
    today = today or now.date().isoformat()
    start_year = current_season - int(historical_seasons_back)

    all_results = []
    all_quali = []
    all_sprint = []
    current_schedule_fallback = pd.DataFrame()
    for year in range(start_year, current_season + 1):
        _emit(3, "Loading supporting race/schedule data", f"Loading supporting race/schedule data ({year})...", progress=0.20)
        _log(f"fetch_supporting_start season={year}")
        try:
            data = fetch_all_supporting(year)
        except Exception as exc:
            _log(f"fetch_supporting_failed season={year} error={exc}")
            continue
        _log(f"fetch_supporting_done season={year}")
        all_results.append(data["results"])
        all_quali.append(data["qualifying"])
        all_sprint.append(data["sprint"])
        if year == current_season:
            current_schedule_fallback = data.get("schedule", pd.DataFrame())

    if not all_results:
        raise RuntimeError("Could not load race-result support data from the public endpoints.")

    results = pd.concat(all_results, ignore_index=True)
    qualifying = pd.concat(all_quali, ignore_index=True) if any(len(x) for x in all_quali) else pd.DataFrame()
    sprint = pd.concat(all_sprint, ignore_index=True) if any(len(x) for x in all_sprint) else pd.DataFrame()

    results = results[(results["season"] >= start_year) & (results["season"] <= current_season)].copy()
    if not qualifying.empty:
        qualifying = qualifying[(qualifying["season"] >= start_year) & (qualifying["season"] <= current_season)].copy()
    if not sprint.empty:
        sprint = sprint[(sprint["season"] >= start_year) & (sprint["season"] <= current_season)].copy()

    _log("market_feed_round_detect_start")
    _emit(1, "Loading market feed", "Detecting latest market feed...", progress=0.08)
    feed_round = _latest_feed_round()
    _log(f"market_feed_round_detect_done round={feed_round}")
    _log("market_players_fetch_start")
    _emit(2, "Loading current prices", "Loading current driver prices...", progress=0.12)
    players = fetch_players(feed_round=feed_round)
    _log(f"market_players_fetch_done count={len(players)}")
    _log("market_constructors_fetch_start")
    _emit(2, "Loading current prices", "Loading current constructor prices...", progress=0.16)
    teams = fetch_teams(feed_round=feed_round)
    _log(f"market_constructors_fetch_done count={len(teams)}")

    _log("schedule_fetch_start")
    _emit(3, "Loading supporting race/schedule data", "Loading current-season schedule...", progress=0.26)
    try:
        schedule = fetch_schedule(current_season)
        _log(f"schedule_fetch_done rows={len(schedule)}")
    except Exception as exc:
        if not current_schedule_fallback.empty:
            schedule = current_schedule_fallback.copy()
            _log(f"schedule_fetch_failed_using_fallback error={exc} rows={len(schedule)}")
            _emit(
                3,
                "Loading supporting race/schedule data",
                "Schedule endpoint failed; using fallback schedule data.",
                progress=0.30,
                status="warning",
            )
        else:
            _log(f"schedule_fetch_failed_no_fallback error={exc}")
            raise
    upcoming = upcoming_circuits(schedule, today=today, n=horizon_races)
    upcoming_rows = schedule[schedule["date"].astype(str) >= today].sort_values("round")
    if not upcoming:
        raise ValueError("No remaining races found in the current season schedule.")
    next_race_name = None
    next_race_date = None
    next_race_round = None
    team_lock_deadline_utc = None
    team_lock_deadline_source = "unavailable"
    team_lock_deadline_raw_field = None
    team_lock_deadline_raw_value = None
    team_lock_timezone_assumption = "SessionStartDate parsed as ISO-8601 when available."
    if not upcoming_rows.empty:
        next_row = upcoming_rows.iloc[0]
        next_race_name = next_row.get("raceName") or next_row.get("circuitName")
        next_race_date = next_row.get("date")
        next_race_round = next_row.get("round")
        schedule_deadline_utc, schedule_source = derive_team_lock_deadline(next_row)
        schedule_deadline_iso = schedule_deadline_utc.isoformat() if schedule_deadline_utc is not None else None
        team_lock_deadline_utc = schedule_deadline_iso
        team_lock_deadline_source = schedule_source
        team_lock_deadline_raw_field = "schedule.qualifying_date/sprint_date"
        team_lock_deadline_raw_value = schedule_deadline_iso

    if not players.empty and "playerId" in players.columns:
        try:
            _log("team_lock_playerstats_probe_start")
            lock_payload = fetch_team_lock_deadline_from_playerstats(int(players.iloc[0]["playerId"]))
            _log("team_lock_playerstats_probe_done")
        except Exception:
            lock_payload = {}
            _log("team_lock_playerstats_probe_failed")
        official_deadline = lock_payload.get("team_lock_deadline_utc")
        if official_deadline:
            team_lock_deadline_utc = official_deadline
            team_lock_deadline_source = lock_payload.get("team_lock_deadline_source", "official_feed_playerstats_session_start")
            team_lock_deadline_raw_field = lock_payload.get("team_lock_deadline_raw_field")
            team_lock_deadline_raw_value = lock_payload.get("team_lock_deadline_raw_value")
            team_lock_timezone_assumption = lock_payload.get("team_lock_timezone_assumption", team_lock_timezone_assumption)
        elif team_lock_deadline_source == "unavailable":
            team_lock_deadline_source = "unavailable"

    horizon_weights = _horizon_weights(len(upcoming), w1=1.0, w_next=0.7)
    _emit(5, "Building model inputs", "Building model inputs...", progress=0.40)
    weekend_points = compute_weekend_points(
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        current_season=current_season,
        last_season_weight=0.95,
        older_decay=0.75,
        race_dnf_penalty=20,
        sprint_dnf_penalty=10,
    )

    drv_exp, con_exp = expected_scores_horizon(
        weekend_points,
        upcoming,
        horizon_weights,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        recency_decay=recency_decay,
    )
    drv_next, con_next = expected_scores_horizon(
        weekend_points,
        upcoming[:1],
        [1.0],
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        recency_decay=recency_decay,
    )
    drv_exp = drv_exp.merge(
        drv_next[["driverId", "exp_score"]].rename(columns={"exp_score": "next_race_exp_score"}),
        on="driverId",
        how="left",
    )
    drv_exp["horizon_expected_points"] = drv_exp["exp_score"]
    drv_exp["exp_score"] = pd.to_numeric(drv_exp["next_race_exp_score"], errors="coerce").fillna(drv_exp["exp_score"])
    con_exp = con_exp.merge(
        con_next[["constructorId", "exp_score"]].rename(columns={"exp_score": "next_race_exp_score"}),
        on="constructorId",
        how="left",
    )
    con_exp["horizon_expected_points"] = con_exp["exp_score"]
    con_exp["exp_score"] = pd.to_numeric(con_exp["next_race_exp_score"], errors="coerce").fillna(con_exp["exp_score"])
    nn_driver = apply_no_negative_expectation(
        weekend_points,
        upcoming,
        horizon_weights,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        recency_decay=recency_decay,
    )
    drv_exp = drv_exp.merge(nn_driver.rename("nn_exp_score"), on="driverId", how="left")
    drv_exp["nn_exp_score"] = drv_exp["nn_exp_score"].fillna(drv_exp["exp_score"])

    _emit(6, "Computing expected points", "Computing expected points...", progress=0.55)
    drivers = _build_driver_table(players, drv_exp)
    constructors = _build_constructor_table(teams, con_exp)
    drivers = _apply_team_strength_adjustment(drivers, constructors)
    if include_playerstats:
        playerstats_started = datetime.now(UTC)
        _log("playerstats_fetch_start drivers")
        _emit(4, "Loading playerstats", "Loading playerstats for drivers (using cache where available)...", progress=0.62)
        def _driver_progress(payload: dict[str, Any]) -> None:
            processed = int(payload.get("processed", 0) or 0)
            total = max(int(payload.get("total", 0) or 0), 1)
            frac = min(1.0, processed / total)
            _emit(
                4,
                "Loading playerstats",
                (
                    f"Loading playerstats {processed}/{total} "
                    f"(failed/timeouts: {int(payload.get('failed', 0) or 0)}) "
                    "using cached stats where available."
                ),
                progress=0.62 + 0.08 * frac,
            )

        drivers, driver_race_points, driver_stats_diag = _add_playerstats_recent_points(
            drivers,
            "driver",
            progress_callback=_driver_progress,
        )
        _log(
            "playerstats_fetch_done drivers "
            f"loaded={driver_stats_diag.get('playerstats_assets_loaded', 0)} "
            f"failed={driver_stats_diag.get('playerstats_assets_failed', 0)}"
        )
        _emit(
            4,
            "Loading playerstats",
            (
                "Driver playerstats loaded "
                f"{driver_stats_diag.get('playerstats_assets_loaded', 0)}/{len(drivers)}; "
                f"failed/timeouts: {driver_stats_diag.get('playerstats_assets_failed', 0)}"
            ),
            progress=0.70,
        )
        _log("playerstats_fetch_start constructors")
        _emit(4, "Loading playerstats", "Loading playerstats for constructors (using cache where available)...", progress=0.72)
        def _constructor_progress(payload: dict[str, Any]) -> None:
            processed = int(payload.get("processed", 0) or 0)
            total = max(int(payload.get("total", 0) or 0), 1)
            frac = min(1.0, processed / total)
            _emit(
                4,
                "Loading playerstats",
                (
                    f"Loading playerstats {processed}/{total} "
                    f"(failed/timeouts: {int(payload.get('failed', 0) or 0)}) "
                    "using cached stats where available."
                ),
                progress=0.72 + 0.06 * frac,
            )

        constructors, constructor_race_points, constructor_stats_diag = _add_playerstats_recent_points(
            constructors,
            "constructor",
            progress_callback=_constructor_progress,
        )
        _log(
            "playerstats_fetch_done constructors "
            f"loaded={constructor_stats_diag.get('playerstats_assets_loaded', 0)} "
            f"failed={constructor_stats_diag.get('playerstats_assets_failed', 0)}"
        )
        _emit(
            4,
            "Loading playerstats",
            (
                "Constructor playerstats loaded "
                f"{constructor_stats_diag.get('playerstats_assets_loaded', 0)}/{len(constructors)}; "
                f"failed/timeouts: {constructor_stats_diag.get('playerstats_assets_failed', 0)}"
            ),
            progress=0.78,
        )
        playerstats_load_duration_seconds = max(0.0, (datetime.now(UTC) - playerstats_started).total_seconds())
    else:
        drivers = _fill_recent_point_columns(drivers)
        constructors = _fill_recent_point_columns(constructors)
        driver_race_points = pd.DataFrame()
        constructor_race_points = pd.DataFrame()
        driver_stats_diag = {
            "playerstats_assets_loaded": 0,
            "playerstats_assets_failed": 0,
            "playerstats_timeout_failures": 0,
            "playerstats_skipped_after_failure_limit": 0,
            "playerstats_failures": [],
        }
        constructor_stats_diag = {
            "playerstats_assets_loaded": 0,
            "playerstats_assets_failed": 0,
            "playerstats_timeout_failures": 0,
            "playerstats_skipped_after_failure_limit": 0,
            "playerstats_failures": [],
        }
        _log("playerstats_prefetch_skipped")
        _emit(4, "Loading playerstats", "Skipping detailed playerstats prefetch.", progress=0.78, status="warning")
        playerstats_load_duration_seconds = 0.0
    _emit(7, "Computing price-change probabilities", "Computing price-change probabilities...", progress=0.90)
    drivers, constructors, calibration_diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_race_points,
        constructor_race_points,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
    )
    recent_diag = playerstats_recent_points_diagnostics(
        drivers,
        constructors,
        driver_race_points,
        constructor_race_points,
        driver_stats_diag,
        constructor_stats_diag,
    )
    drivers = ensure_image_url_column(drivers)
    constructors = ensure_image_url_column(constructors)
    drivers["team_colour"] = drivers["team"].apply(team_colour) if "team" in drivers.columns else DEFAULT_TEAM_COLOUR
    constructors["team_colour"] = constructors["name"].apply(team_colour)
    trends = build_trends_data(drivers, constructors, driver_race_points, constructor_race_points)
    load_finished = datetime.now(UTC)
    load_seconds = max(0.0, (load_finished - load_started).total_seconds())

    diagnostics = {
        "current_season": current_season,
        "start_year": start_year,
        "historical_seasons_back": int(historical_seasons_back),
        "today": today,
        "feed_round": feed_round,
        "upcoming_circuits": upcoming,
        "next_race_name": next_race_name,
        "next_race_date": next_race_date,
        "next_race_round": int(next_race_round) if pd.notna(next_race_round) else None,
        "team_lock_deadline_utc": team_lock_deadline_utc,
        "team_lock_deadline_source": team_lock_deadline_source,
        "team_lock_deadline_raw_field": team_lock_deadline_raw_field,
        "team_lock_deadline_raw_value": team_lock_deadline_raw_value,
        "team_lock_timezone_assumption": team_lock_timezone_assumption,
        "upcoming_race_horizon": int(horizon_races),
        "horizon_weights": horizon_weights,
        "current_season_weight": float(current_season_weight),
        "past_season_weight": float(past_season_weight),
        "recency_decay": float(recency_decay),
        "driver_count": len(drivers),
        "constructor_count": len(constructors),
        "driver_dnf_rate_missing": int(pd.to_numeric(drivers.get("dnf_rate", pd.Series(dtype=float)), errors="coerce").isna().sum()),
        "constructor_dnf_rate_missing": int(pd.to_numeric(constructors.get("dnf_rate", pd.Series(dtype=float)), errors="coerce").isna().sum()),
        "dnf_price_gain_score": float(DEFAULT_DNF_PRICE_GAIN_SCORE),
        "race_dnf_bad_score": float(DEFAULT_RACE_DNF_BAD_SCORE),
        "sprint_dnf_bad_score": float(DEFAULT_SPRINT_DNF_BAD_SCORE),
        "dnf_price_gain_score_source": "Fixed generic race-weekend bad-outcome score; repo scoring uses -20 race DNF and -10 sprint DNF.",
        "playerstats_prefetch_enabled": bool(include_playerstats),
        "model_load_started_utc": load_started.isoformat(),
        "model_load_finished_utc": load_finished.isoformat(),
        "model_load_duration_seconds": float(load_seconds),
        "playerstats_load_duration_seconds": float(playerstats_load_duration_seconds),
        "model_load_events": load_events[-40:],
        **calibration_diag,
        **recent_diag,
    }
    if not include_playerstats:
        diagnostics["recent_points_source"] = "Playerstats prefetch skipped for faster startup; per-race fields may be incomplete until enrichment."
    diagnostics["playerstats_timeout_failures"] = int(driver_stats_diag.get("playerstats_timeout_failures", 0)) + int(
        constructor_stats_diag.get("playerstats_timeout_failures", 0)
    )
    diagnostics["playerstats_skipped_after_failure_limit"] = int(
        driver_stats_diag.get("playerstats_skipped_after_failure_limit", 0)
    ) + int(constructor_stats_diag.get("playerstats_skipped_after_failure_limit", 0))
    _emit(8, "Ready", f"Ready. Data loaded in {load_seconds:.1f}s.", progress=1.0, status="complete")
    return ModelData(drivers=drivers, constructors=constructors, trends=trends, diagnostics=diagnostics)


def _build_driver_table(players: pd.DataFrame, drv_exp: pd.DataFrame) -> pd.DataFrame:
    drv_baseline = float(drv_exp["exp_score"].min()) if len(drv_exp) else 0.0
    drv_dnf_baseline = float(drv_exp["dnf_rate"].max()) if "dnf_rate" in drv_exp.columns and len(drv_exp) else 0.25

    fp = players.copy()
    if "FirstName" in fp.columns and "LastName" in fp.columns:
        fp["canon_name"] = (fp["FirstName"].astype(str) + " " + fp["LastName"].astype(str)).map(_canon)
    else:
        fp["canon_name"] = fp["name"].astype(str).map(_canon)

    drv_exp = drv_exp.copy()
    drv_exp["canon_name"] = drv_exp["driver"].astype(str).map(_canon)
    drv_exp["canon_last"] = drv_exp["canon_name"].str.split(" ").str[-1]

    last_to_rows = drv_exp.groupby("canon_last")["canon_name"].apply(list).to_dict()
    canon_to_row = drv_exp.set_index("canon_name")[["driverId", "exp_score", "next_race_exp_score", "horizon_expected_points", "dnf_rate", "volatility", "nn_exp_score"]].to_dict("index")

    def map_driver_row(cname: str):
        if cname in canon_to_row:
            return canon_to_row[cname]
        for key in canon_to_row:
            if cname and (cname in key or key in cname):
                return canon_to_row[key]
        last = cname.split(" ")[-1] if cname else ""
        cands = last_to_rows.get(last, [])
        if len(cands) == 1:
            return canon_to_row[cands[0]]
        best = _best_fuzzy(cname, list(canon_to_row.keys()), cutoff=0.72)
        return canon_to_row[best] if best else None

    mapped = fp["canon_name"].apply(map_driver_row)
    drivers = fp.copy()
    drivers["driverId"] = mapped.apply(lambda x: x["driverId"] if isinstance(x, dict) else None)
    drivers["exp_score"] = mapped.apply(lambda x: x["exp_score"] if isinstance(x, dict) else None)
    drivers["next_race_exp_score"] = mapped.apply(lambda x: x.get("next_race_exp_score") if isinstance(x, dict) else None)
    drivers["horizon_expected_points"] = mapped.apply(lambda x: x.get("horizon_expected_points") if isinstance(x, dict) else None)
    drivers["dnf_rate"] = mapped.apply(lambda x: x["dnf_rate"] if isinstance(x, dict) else None)
    drivers["volatility"] = mapped.apply(lambda x: x["volatility"] if isinstance(x, dict) else None)
    drivers["nn_exp_score"] = mapped.apply(lambda x: x.get("nn_exp_score", x["exp_score"]) if isinstance(x, dict) else None)

    drivers["exp_score"] = pd.to_numeric(drivers["exp_score"], errors="coerce").fillna(drv_baseline)
    drivers["next_race_exp_score"] = pd.to_numeric(drivers["next_race_exp_score"], errors="coerce").fillna(drivers["exp_score"])
    drivers["horizon_expected_points"] = pd.to_numeric(drivers["horizon_expected_points"], errors="coerce").fillna(drivers["exp_score"])
    drivers["dnf_rate"] = pd.to_numeric(drivers["dnf_rate"], errors="coerce").fillna(drv_dnf_baseline)
    drivers["volatility"] = pd.to_numeric(drivers["volatility"], errors="coerce").fillna(
        pd.to_numeric(drivers["volatility"], errors="coerce").median()
    )
    drivers["nn_exp_score"] = pd.to_numeric(drivers["nn_exp_score"], errors="coerce").fillna(drivers["exp_score"])
    drivers.rename(columns={"playerId": "id"}, inplace=True)
    return drivers


def _build_constructor_table(teams: pd.DataFrame, con_exp: pd.DataFrame) -> pd.DataFrame:
    ctor_baseline = float(con_exp["exp_score"].min()) if len(con_exp) else 0.0
    ctor_dnf_baseline = float(con_exp["dnf_rate"].max()) if "dnf_rate" in con_exp.columns and len(con_exp) else 0.25

    team_alias = {
        "red bull": "red bull",
        "red bull racing": "red bull",
        "mclaren": "mclaren",
        "mercedes": "mercedes",
        "ferrari": "ferrari",
        "williams": "williams",
        "aston martin": "aston martin",
        "haas": "haas",
        "haas f1": "haas",
        "haas f1 team": "haas",
        "alpine": "alpine",
        "alpine f1": "alpine",
        "alpine f1 team": "alpine",
        "racing bulls": "rb",
        "rb": "rb",
        "rb f1": "rb",
        "rb f1 team": "rb",
        "audi": "sauber",
        "sauber": "sauber",
        "kick sauber": "sauber",
        "cadillac": None,
    }

    ft = teams.copy()
    ft["canon_team"] = ft["name"].astype(str).map(_canon_team)

    con_exp = con_exp.copy()
    con_exp["canon_team"] = con_exp["constructor"].astype(str).map(_canon_team)
    con_keys = con_exp.set_index("canon_team")[["constructorId", "exp_score", "next_race_exp_score", "horizon_expected_points", "dnf_rate", "volatility"]].to_dict("index")

    def map_constructor_row(cteam: str):
        base = team_alias.get(cteam)
        if base is None:
            return None
        for key, row in con_keys.items():
            if base == key or (base and (base in key or key in base)):
                return row
        best = _best_fuzzy(base, list(con_keys.keys()), cutoff=0.65)
        return con_keys.get(best) if best else None

    mapped = ft["canon_team"].apply(map_constructor_row)
    constructors = ft.copy()
    constructors["constructorId"] = mapped.apply(lambda x: x["constructorId"] if isinstance(x, dict) else None)
    constructors["exp_score"] = mapped.apply(lambda x: x["exp_score"] if isinstance(x, dict) else None)
    constructors["next_race_exp_score"] = mapped.apply(lambda x: x.get("next_race_exp_score") if isinstance(x, dict) else None)
    constructors["horizon_expected_points"] = mapped.apply(lambda x: x.get("horizon_expected_points") if isinstance(x, dict) else None)
    constructors["dnf_rate"] = mapped.apply(lambda x: x["dnf_rate"] if isinstance(x, dict) else None)
    constructors["volatility"] = mapped.apply(lambda x: x["volatility"] if isinstance(x, dict) else None)

    constructors["exp_score"] = pd.to_numeric(constructors["exp_score"], errors="coerce").fillna(ctor_baseline)
    constructors["next_race_exp_score"] = pd.to_numeric(constructors["next_race_exp_score"], errors="coerce").fillna(constructors["exp_score"])
    constructors["horizon_expected_points"] = pd.to_numeric(constructors["horizon_expected_points"], errors="coerce").fillna(constructors["exp_score"])
    constructors["dnf_rate"] = pd.to_numeric(constructors["dnf_rate"], errors="coerce").fillna(ctor_dnf_baseline)
    constructors["volatility"] = pd.to_numeric(constructors["volatility"], errors="coerce").fillna(
        pd.to_numeric(constructors["volatility"], errors="coerce").median()
    )
    constructors.rename(columns={"teamId": "id"}, inplace=True)
    return constructors


def _apply_team_strength_adjustment(drivers: pd.DataFrame, constructors: pd.DataFrame) -> pd.DataFrame:
    drivers = drivers.copy()
    ctor_exp_by_name = constructors.set_index("name")["exp_score"].to_dict()
    drivers["team_exp"] = drivers["team"].map(ctor_exp_by_name)

    team_exps = constructors["exp_score"].astype(float)
    p10 = float(team_exps.quantile(0.10))
    p90 = float(team_exps.quantile(0.90))

    def team_factor(team_exp: float) -> float:
        if team_exp is None or pd.isna(team_exp) or p90 <= p10:
            return 1.0
        if team_exp <= p10:
            return 0.35
        if team_exp >= p90:
            return 1.15
        return 0.35 + (team_exp - p10) * (1.15 - 0.35) / (p90 - p10)

    drivers["team_factor"] = drivers["team_exp"].apply(team_factor).astype(float)
    drivers["exp_score_raw"] = drivers["exp_score"].astype(float)
    drivers["nn_exp_score_raw"] = drivers["nn_exp_score"].astype(float)
    drivers["exp_score"] = drivers["exp_score_raw"] * drivers["team_factor"]
    drivers["nn_exp_score"] = drivers["nn_exp_score_raw"] * drivers["team_factor"]
    if "next_race_exp_score" in drivers.columns:
        drivers["next_race_exp_score"] = pd.to_numeric(drivers["next_race_exp_score"], errors="coerce") * drivers["team_factor"]
    if "horizon_expected_points" in drivers.columns:
        drivers["horizon_expected_points"] = pd.to_numeric(drivers["horizon_expected_points"], errors="coerce") * drivers["team_factor"]
    return drivers


def _recent_two_points(df: pd.DataFrame, id_col: str, points_col: str, current_season: int) -> pd.DataFrame:
    required_cols = [id_col, "season", "round", points_col]
    if df.empty or any(col not in df.columns for col in required_cols):
        return pd.DataFrame(columns=[id_col, "recent_points_2ago", "recent_points_1ago", "recent_points_available"])

    current = df[df["season"].astype(int) == int(current_season)].copy()
    if current.empty:
        return pd.DataFrame(columns=[id_col, "recent_points_2ago", "recent_points_1ago", "recent_points_available"])

    rows = []
    for asset_id, group in current.sort_values("round").groupby(id_col):
        points = pd.to_numeric(group[points_col], errors="coerce").dropna().tail(2).tolist()
        rows.append(
            {
                id_col: asset_id,
                "recent_points_2ago": float(points[-2]) if len(points) >= 2 else pd.NA,
                "recent_points_1ago": float(points[-1]) if len(points) >= 1 else pd.NA,
                "recent_points_available": int(len(points)),
                "recent_points_source": "actual",
            }
        )
    return pd.DataFrame(rows)


def _add_recent_driver_points(drivers: pd.DataFrame, weekend_points: pd.DataFrame, current_season: int) -> pd.DataFrame:
    recent = _recent_two_points(weekend_points, "driverId", "weekend_points", current_season)
    out = drivers.merge(recent, on="driverId", how="left")
    return _fill_recent_point_columns(out)


def _add_recent_constructor_points(constructors: pd.DataFrame, weekend_points: pd.DataFrame, current_season: int) -> pd.DataFrame:
    ctor_round = _constructor_round_points(weekend_points)
    recent = _recent_two_points(ctor_round, "constructorId", "constructor_weekend_points", current_season)
    out = constructors.merge(recent, on="constructorId", how="left")
    return _fill_recent_point_columns(out)


def _add_playerstats_recent_points(
    df: pd.DataFrame,
    asset_type: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    recent, race_points, diagnostics = fetch_recent_points_for_roster(
        df,
        asset_type=asset_type,
        progress_callback=progress_callback,
    )
    recent_cols = [
        "recent_points_2ago",
        "recent_points_1ago",
        "recent_points_available",
        "recent_points_source",
        "recent_points_races",
        "recent_points_fallback_used",
        "recent_points_missing",
    ]
    out = df.drop(columns=[col for col in recent_cols if col in df.columns]).merge(recent, on="id", how="left")
    return _fill_recent_point_columns(out), race_points, diagnostics


def _fill_recent_point_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["recent_points_2ago", "recent_points_1ago"]:
        if col not in out.columns:
            out[col] = pd.NA
    if "recent_points_available" not in out.columns:
        out["recent_points_available"] = 0
    if "recent_points_source" not in out.columns:
        out["recent_points_source"] = "missing"
    out["recent_points_2ago"] = pd.to_numeric(out["recent_points_2ago"], errors="coerce")
    out["recent_points_1ago"] = pd.to_numeric(out["recent_points_1ago"], errors="coerce")
    out["recent_points_available"] = pd.to_numeric(out["recent_points_available"], errors="coerce").fillna(0).astype(int)
    out["recent_points_source"] = out["recent_points_source"].fillna("missing").astype(str)
    out["recent_points_fallback_used"] = out["recent_points_available"] < 2
    out["recent_points_missing"] = out["recent_points_2ago"].isna() | out["recent_points_1ago"].isna()
    return out


def historical_scale_factor(
    current_avg_points_per_race: float | None,
    historical_avg_points_per_race: float | None,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
) -> tuple[float, bool]:
    current = pd.to_numeric(current_avg_points_per_race, errors="coerce")
    historical = pd.to_numeric(historical_avg_points_per_race, errors="coerce")
    if pd.isna(current) or pd.isna(historical) or float(historical) <= 0:
        return 1.0, False
    raw = float(current) / float(historical)
    clipped = min(max(raw, float(min_scale)), float(max_scale))
    return clipped, clipped != raw


def _observed_average_by_player(race_points: pd.DataFrame) -> pd.DataFrame:
    if race_points.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "observed_current_avg_points",
                "observed_current_races",
                "current_season_avg_points",
                "current_season_points_count",
                "current_season_volatility",
            ]
        )
    df = race_points.copy()
    df = df[(df.get("is_played", 0) == 1) & pd.to_numeric(df.get("fantasy_points"), errors="coerce").notna()]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "observed_current_avg_points",
                "observed_current_races",
                "current_season_avg_points",
                "current_season_points_count",
                "current_season_volatility",
            ]
        )
    df["fantasy_points"] = pd.to_numeric(df["fantasy_points"], errors="coerce")

    def observed_volatility(points: pd.Series):
        clean = pd.to_numeric(points, errors="coerce").dropna()
        if len(clean) < 2:
            return pd.NA
        return float(clean.std(ddof=0))

    grouped = df.groupby("PlayerId", as_index=False).agg(
        observed_current_avg_points=("fantasy_points", "mean"),
        observed_current_races=("fantasy_points", "count"),
        current_season_volatility=("fantasy_points", observed_volatility),
    )
    grouped.rename(columns={"PlayerId": "id"}, inplace=True)
    grouped["current_season_avg_points"] = grouped["observed_current_avg_points"]
    grouped["current_season_points_count"] = grouped["observed_current_races"]
    return grouped


def apply_observed_playerstats_projection(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    driver_race_points: pd.DataFrame,
    constructor_race_points: pd.DataFrame,
    current_season_weight: float,
    past_season_weight: float,
    driver_volatility_floor: float = DEFAULT_DRIVER_SCORE_VOLATILITY_FLOOR,
    constructor_volatility_floor: float = DEFAULT_CONSTRUCTOR_SCORE_VOLATILITY_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Blend true current-season playerstats form with calibrated model proxy values."""
    driver_obs = _observed_average_by_player(driver_race_points)
    constructor_obs = _observed_average_by_player(constructor_race_points)
    combined_observed = pd.concat([driver_obs, constructor_obs], ignore_index=True)
    current_avg = pd.to_numeric(combined_observed.get("observed_current_avg_points"), errors="coerce").mean()

    historical_values = pd.concat(
        [
            pd.to_numeric(drivers.get("exp_score"), errors="coerce"),
            pd.to_numeric(constructors.get("exp_score"), errors="coerce"),
        ],
        ignore_index=True,
    )
    historical_avg = historical_values.dropna().mean()
    scale, clipped = historical_scale_factor(current_avg, historical_avg)

    def apply_one(df: pd.DataFrame, obs: pd.DataFrame, volatility_floor: float) -> pd.DataFrame:
        out = df.copy()
        score_cols = [col for col in ["exp_score", "next_race_exp_score", "horizon_expected_points"] if col in out.columns]
        for col in score_cols:
            out[f"historical_proxy_{col}"] = pd.to_numeric(out[col], errors="coerce")
        historical_volatility = pd.to_numeric(
            out.get("volatility", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        )
        out["historical_volatility"] = historical_volatility
        out["normalised_historical_volatility"] = historical_volatility * scale
        out = out.merge(obs, on="id", how="left")
        for col in score_cols:
            out[f"scaled_historical_{col}"] = out[f"historical_proxy_{col}"] * scale
        cur = pd.to_numeric(out["observed_current_avg_points"], errors="coerce")
        cur_w = float(current_season_weight)
        hist_w = float(past_season_weight)
        for col in score_cols:
            hist = pd.to_numeric(out[f"scaled_historical_{col}"], errors="coerce")
            denom = (cur.notna().astype(float) * cur_w) + (hist.notna().astype(float) * hist_w)
            numerator = cur.fillna(0.0) * cur_w + hist.fillna(0.0) * hist_w
            blended = numerator.where(denom > 0, hist).div(denom.where(denom > 0, 1.0))
            out[col] = blended.fillna(out[f"historical_proxy_{col}"])
        if "nn_exp_score" in out.columns:
            out["nn_exp_score"] = out["exp_score"]
        out["current_season_observed_avg_points_per_race"] = out["observed_current_avg_points"]
        out["historical_prior_expected_points_per_race"] = out.get("historical_proxy_next_race_exp_score", out.get("historical_proxy_exp_score"))
        out["normalised_historical_expected_points_per_race"] = out.get("scaled_historical_next_race_exp_score", out.get("scaled_historical_exp_score"))
        out["next_race_expected_points"] = out["next_race_exp_score"] if "next_race_exp_score" in out.columns else out["exp_score"]
        out["exp_score"] = out["next_race_expected_points"]
        out["expected_points_source"] = out["observed_current_avg_points"].apply(
            lambda value: "playerstats_blended" if pd.notna(value) else "historical_proxy_scaled"
        )

        current_vol = pd.to_numeric(
            out.get("current_season_volatility", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        )
        hist_vol = pd.to_numeric(
            out.get("normalised_historical_volatility", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        )
        current_available = current_vol.notna()
        historical_available = hist_vol.notna()
        denom = current_available.astype(float) * cur_w + historical_available.astype(float) * hist_w
        numerator = current_vol.fillna(0.0) * cur_w + hist_vol.fillna(0.0) * hist_w
        blended_raw = numerator.where(denom > 0).div(denom.where(denom > 0, 1.0))
        blended_raw = blended_raw.combine_first(current_vol).combine_first(hist_vol)
        volatility_source = pd.Series("fallback_floor", index=out.index, dtype=object)
        volatility_source = volatility_source.where(~(current_available & historical_available), "blended_current_historical")
        volatility_source = volatility_source.where(~(current_available & ~historical_available), "current_playerstats")
        volatility_source = volatility_source.where(~(~current_available & historical_available), "historical_model_proxy")
        out["blended_volatility_before_floor"] = blended_raw
        out["volatility_floor"] = float(volatility_floor)
        out["volatility_floor_applied"] = pd.to_numeric(blended_raw, errors="coerce").notna() & (pd.to_numeric(blended_raw, errors="coerce") < float(volatility_floor))
        out["volatility_source"] = volatility_source
        out["volatility"] = pd.to_numeric(blended_raw, errors="coerce").clip(lower=float(volatility_floor)).fillna(float(volatility_floor))
        return out

    out_drivers = apply_one(drivers, driver_obs, driver_volatility_floor)
    out_constructors = apply_one(constructors, constructor_obs, constructor_volatility_floor)
    observed_races = int(pd.to_numeric(combined_observed.get("observed_current_races"), errors="coerce").sum()) if len(combined_observed) else 0
    volatility_sources = pd.concat(
        [
            out_drivers.get("volatility_source", pd.Series(dtype=object)),
            out_constructors.get("volatility_source", pd.Series(dtype=object)),
        ],
        ignore_index=True,
    )
    volatility_source_counts = {str(k): int(v) for k, v in volatility_sources.value_counts(dropna=False).to_dict().items()}
    current_vol_available = int(
        pd.concat(
            [
                pd.to_numeric(out_drivers.get("current_season_volatility", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(out_constructors.get("current_season_volatility", pd.Series(dtype=float)), errors="coerce"),
            ],
            ignore_index=True,
        ).notna().sum()
    )
    historical_vol_available = int(
        pd.concat(
            [
                pd.to_numeric(out_drivers.get("normalised_historical_volatility", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(out_constructors.get("normalised_historical_volatility", pd.Series(dtype=float)), errors="coerce"),
            ],
            ignore_index=True,
        ).notna().sum()
    )
    floor_applied = int(
        pd.concat(
            [
                out_drivers.get("volatility_floor_applied", pd.Series(dtype=bool)).fillna(False).astype(bool),
                out_constructors.get("volatility_floor_applied", pd.Series(dtype=bool)).fillna(False).astype(bool),
            ],
            ignore_index=True,
        ).sum()
    )
    diagnostics = {
        "observed_current_avg_points_per_race": float(current_avg) if pd.notna(current_avg) else None,
        "historical_avg_points_per_race": float(historical_avg) if pd.notna(historical_avg) else None,
        "historical_scale_factor": float(scale),
        "historical_scale_factor_clipped": bool(clipped),
        "observed_current_assets": int(len(combined_observed)),
        "observed_current_race_rows": observed_races,
        "volatility_source": "True playerstats current-season race scores blended with scaled historical/model proxy volatility.",
        "volatility_source_counts": volatility_source_counts,
        "current_season_volatility_assets": current_vol_available,
        "historical_volatility_assets": historical_vol_available,
        "fallback_volatility_assets": int(volatility_source_counts.get("fallback_floor", 0)),
        "blended_current_historical_volatility_assets": int(volatility_source_counts.get("blended_current_historical", 0)),
        "current_only_volatility_assets": int(volatility_source_counts.get("current_playerstats", 0)),
        "historical_only_volatility_assets": int(volatility_source_counts.get("historical_model_proxy", 0)),
        "volatility_floor_applied_assets": floor_applied,
        "driver_volatility_floor": float(driver_volatility_floor),
        "constructor_volatility_floor": float(constructor_volatility_floor),
    }
    return out_drivers, out_constructors, diagnostics


def apply_recent_point_overrides(
    df: pd.DataFrame,
    overrides: pd.DataFrame | None,
    id_col: str,
) -> pd.DataFrame:
    out = _fill_recent_point_columns(df)
    if overrides is None or overrides.empty or id_col not in out.columns or id_col not in overrides.columns:
        return out

    manual = overrides[[id_col, *[col for col in ["recent_points_2ago", "recent_points_1ago"] if col in overrides.columns]]].copy()
    manual.rename(
        columns={
            "recent_points_2ago": "recent_points_2ago_manual",
            "recent_points_1ago": "recent_points_1ago_manual",
        },
        inplace=True,
    )
    out = out.merge(manual, on=id_col, how="left")
    manual_mask = pd.Series(False, index=out.index)
    for col in ["recent_points_2ago", "recent_points_1ago"]:
        manual_col = f"{col}_manual"
        if manual_col in out.columns:
            manual_mask = manual_mask | pd.to_numeric(out[manual_col], errors="coerce").notna()
            out[col] = pd.to_numeric(out[manual_col], errors="coerce").combine_first(pd.to_numeric(out[col], errors="coerce"))
            out.drop(columns=[manual_col], inplace=True)

    out["recent_points_available"] = out[["recent_points_2ago", "recent_points_1ago"]].notna().sum(axis=1).astype(int)
    out["recent_points_source"] = out["recent_points_source"].where(~manual_mask, "manual")
    out.loc[out["recent_points_available"] == 0, "recent_points_source"] = "missing"
    return _fill_recent_point_columns(out)


def recent_points_diagnostics(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    weekend_points: pd.DataFrame,
    current_season: int,
) -> dict:
    current = weekend_points[weekend_points["season"].astype(int) == int(current_season)].copy()
    rounds = sorted(current["round"].dropna().astype(int).unique().tolist())[-2:]
    circuit_by_round = (
        current[["round", "circuitName"]]
        .drop_duplicates()
        .sort_values("round")
        .tail(2)
        .to_dict("records")
    )
    driver_complete = int((drivers.get("recent_points_available", pd.Series(dtype=int)) >= 2).sum())
    constructor_complete = int((constructors.get("recent_points_available", pd.Series(dtype=int)) >= 2).sum())
    driver_manual = int((drivers.get("recent_points_source", pd.Series(dtype=object)).astype(str) == "manual").sum())
    constructor_manual = int((constructors.get("recent_points_source", pd.Series(dtype=object)).astype(str) == "manual").sum())
    fallback_used = driver_complete < len(drivers) or constructor_complete < len(constructors)
    return {
        "recent_points_source": "Derived from Jolpica/Ergast race, qualifying and sprint results via the local fantasy scoring model.",
        "recent_points_driver_complete": driver_complete,
        "recent_points_constructor_complete": constructor_complete,
        "recent_points_driver_manual": driver_manual,
        "recent_points_constructor_manual": constructor_manual,
        "recent_points_driver_total": int(len(drivers)),
        "recent_points_constructor_total": int(len(constructors)),
        "recent_points_rounds": rounds,
        "recent_points_circuits": circuit_by_round,
        "recent_points_fallback_used": bool(fallback_used),
    }


def playerstats_recent_points_diagnostics(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    driver_race_points: pd.DataFrame,
    constructor_race_points: pd.DataFrame,
    driver_stats_diag: dict | None = None,
    constructor_stats_diag: dict | None = None,
) -> dict:
    driver_stats_diag = driver_stats_diag or {}
    constructor_stats_diag = constructor_stats_diag or {}
    all_races = pd.concat(
        [df for df in [driver_race_points, constructor_race_points] if not df.empty],
        ignore_index=True,
    ) if not driver_race_points.empty or not constructor_race_points.empty else pd.DataFrame()
    latest = latest_two_races(all_races)
    driver_complete = int((drivers.get("recent_points_available", pd.Series(dtype=int)) >= 2).sum())
    constructor_complete = int((constructors.get("recent_points_available", pd.Series(dtype=int)) >= 2).sum())
    driver_manual = int((drivers.get("recent_points_source", pd.Series(dtype=object)).astype(str) == "manual").sum())
    constructor_manual = int((constructors.get("recent_points_source", pd.Series(dtype=object)).astype(str) == "manual").sum())
    fallback_used = driver_complete < len(drivers) or constructor_complete < len(constructors)
    failed = int(driver_stats_diag.get("playerstats_assets_failed", 0)) + int(constructor_stats_diag.get("playerstats_assets_failed", 0))
    loaded = driver_complete + constructor_complete
    return {
        "recent_points_source": "Official F1 Fantasy playerstats popup endpoint.",
        "recent_points_endpoint_pattern": PLAYERSTATS_ENDPOINT_PATTERN,
        "recent_points_driver_complete": driver_complete,
        "recent_points_constructor_complete": constructor_complete,
        "recent_points_driver_manual": driver_manual,
        "recent_points_constructor_manual": constructor_manual,
        "recent_points_driver_total": int(len(drivers)),
        "recent_points_constructor_total": int(len(constructors)),
        "recent_points_rounds": [int(r["round"]) for r in latest if pd.notna(r.get("round"))],
        "recent_points_circuits": latest,
        "recent_points_fallback_used": bool(fallback_used),
        "playerstats_assets_loaded": loaded,
        "playerstats_assets_failed": failed,
        "playerstats_driver_failures": driver_stats_diag.get("playerstats_failures", []),
        "playerstats_constructor_failures": constructor_stats_diag.get("playerstats_failures", []),
    }


def clean_assumption_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["id"] = out["id"].astype(str)
    for col in [
        "price",
        "exp_score",
        "next_race_exp_score",
        "next_race_expected_points",
        "horizon_expected_points",
        "current_season_observed_avg_points_per_race",
        "historical_prior_expected_points_per_race",
        "normalised_historical_expected_points_per_race",
        "dnf_rate",
        "volatility_used",
        "nn_exp_score",
        "expected_price_change",
        "raw_price_change",
        "effective_price_change_after_floor_ceiling",
        "projected_price",
        "avg_ppm",
        "required_terrible_max",
        "required_poor_min",
        "required_good_min",
        "required_great_min",
        "points_objective",
        "price_growth_objective",
        "combined_objective_score",
        "recent_points_2ago",
        "recent_points_1ago",
        "recent_points_available",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "nn_exp_score" in out.columns:
        out["nn_exp_score"] = out["nn_exp_score"].fillna(out["exp_score"])
    return out


def ensure_image_url_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "image_url" not in out.columns:
        out["image_url"] = ""
    out["image_url"] = out["image_url"].fillna("").astype(str)
    return out


def _asset_metadata(roster: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    if roster.empty:
        return pd.DataFrame(columns=["asset_id", "asset_type", "name", "team", "current_price", "team_colour"])
    meta = roster.copy()
    meta["asset_id"] = meta["id"].astype(str)
    meta["asset_type"] = asset_type
    if "team" not in meta.columns:
        meta["team"] = meta["name"]
    if "team_colour" not in meta.columns:
        meta["team_colour"] = meta["team"].apply(team_colour)
    return meta[["asset_id", "asset_type", "name", "team", "price", "team_colour"]].rename(
        columns={"price": "current_price"}
    )


def build_trends_data(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    driver_race_points: pd.DataFrame,
    constructor_race_points: pd.DataFrame,
) -> pd.DataFrame:
    """Build a long race-by-race trend table from playerstats rows and roster metadata."""
    race_frames = []
    for frame, asset_type in [(driver_race_points, "driver"), (constructor_race_points, "constructor")]:
        if frame is None or frame.empty:
            continue
        data = frame.copy()
        data["asset_type"] = asset_type
        race_frames.append(data)

    if not race_frames:
        return pd.DataFrame(
            columns=[
                "asset_type",
                "asset_id",
                "name",
                "team",
                "round",
                "race_name",
                "fantasy_points",
                "cumulative_points",
                "rolling_3race_avg",
                "price_used",
                "points_per_million",
            ]
        )

    races = pd.concat(race_frames, ignore_index=True)
    races["asset_id"] = races["PlayerId"].astype(str)
    races["round"] = pd.to_numeric(races.get("round"), errors="coerce")
    races["fantasy_points"] = pd.to_numeric(races.get("fantasy_points"), errors="coerce")
    if "is_played" in races.columns:
        races = races[pd.to_numeric(races["is_played"], errors="coerce").fillna(0).astype(int) == 1]
    races = races[races["fantasy_points"].notna()].copy()

    metadata = pd.concat(
        [_asset_metadata(drivers, "driver"), _asset_metadata(constructors, "constructor")],
        ignore_index=True,
    )
    out = races.merge(metadata, on=["asset_id", "asset_type"], how="left", suffixes=("", "_roster"))
    if "name_roster" in out.columns:
        out["name"] = out["name_roster"].combine_first(out.get("name"))
        out.drop(columns=["name_roster"], inplace=True)
    elif "name" not in out.columns:
        out["name"] = out["asset_id"]
    out["team"] = out["team"].fillna(out["name"])
    out["team_colour"] = out["team_colour"].fillna(out["team"].apply(team_colour))

    race_price = pd.to_numeric(out.get("price"), errors="coerce")
    current_price = pd.to_numeric(out.get("current_price"), errors="coerce")
    out["price_used"] = race_price.combine_first(current_price)
    out["price_source"] = race_price.notna().map(lambda has_race_price: "playerstats race price" if has_race_price else "current price approximation")
    out["points_per_million"] = (out["fantasy_points"] / out["price_used"]).where(out["price_used"] > 0)

    out = out.sort_values(["asset_type", "name", "round"], na_position="last").reset_index(drop=True)
    grouped = out.groupby(["asset_type", "asset_id"], sort=False)["fantasy_points"]
    out["cumulative_points"] = grouped.cumsum()
    out["rolling_3race_avg"] = grouped.transform(lambda series: series.rolling(3, min_periods=1).mean())
    return out


def filter_trends_data(
    trends: pd.DataFrame,
    asset_type: str | None = None,
    selected_asset_ids: list[str] | None = None,
) -> pd.DataFrame:
    out = trends.copy()
    if asset_type:
        out = out[out["asset_type"].astype(str) == str(asset_type)]
    if selected_asset_ids:
        selected = {str(asset_id) for asset_id in selected_asset_ids}
        out = out[out["asset_id"].astype(str).isin(selected)]
    return out.reset_index(drop=True)


def selected_assets_price_gain(*asset_frames: pd.DataFrame) -> float:
    total = 0.0
    for frame in asset_frames:
        if frame is None or frame.empty:
            continue
        if "expected_price_gain" in frame.columns:
            total += pd.to_numeric(frame["expected_price_gain"], errors="coerce").fillna(0.0).sum()
        elif "expected_price_change" in frame.columns:
            total += pd.to_numeric(frame["expected_price_change"], errors="coerce").fillna(0.0).sum()
        elif "effective_price_change_after_floor_ceiling" in frame.columns:
            total += pd.to_numeric(frame["effective_price_change_after_floor_ceiling"], errors="coerce").fillna(0.0).sum()
    return float(total)


def projected_team_value_from_budget(budget: float, expected_price_gain: float) -> float:
    """Projected total squad value, including bank already represented in budget."""
    return float(budget) + float(expected_price_gain)


def select_chip_boost_drivers(drivers: pd.DataFrame, chip_mode: str = CHIP_NONE) -> tuple[str | None, str | None]:
    """Pick point-boost drivers for a selected team without changing asset EV columns."""
    if drivers.empty or "name" not in drivers.columns:
        return None, None
    scored = drivers.copy()
    scored["_points"] = pd.to_numeric(scored.get("exp_score", pd.Series(index=scored.index, dtype=float)), errors="coerce").fillna(0.0)
    scored = scored.sort_values(["_points", "price"], ascending=False, na_position="last")
    names = scored["name"].astype(str).tolist()
    if not names:
        return None, None
    if chip_mode == CHIP_TRIPLE:
        triple_driver = names[0]
        boosted_driver = names[1] if len(names) > 1 else None
        return boosted_driver, triple_driver
    return names[0], None


def team_expected_points_with_chips(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    chip_mode: str = CHIP_NONE,
    boosted_driver: str | None = None,
    triple_driver: str | None = None,
) -> float:
    """Team expected points with chips applied only to points, never price-gain fields."""
    driver_points = pd.to_numeric(drivers.get("exp_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    constructor_points = pd.to_numeric(constructors.get("exp_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    total = float(driver_points.sum() + constructor_points.sum())
    auto_boosted, auto_triple = select_chip_boost_drivers(drivers, chip_mode)
    boosted_driver = boosted_driver or auto_boosted
    triple_driver = triple_driver or auto_triple
    by_name = drivers.assign(_name=drivers["name"].astype(str)).set_index("_name")["exp_score"].to_dict() if "name" in drivers.columns else {}
    def point_for(name: str | None) -> float:
        if not name:
            return 0.0
        value = pd.to_numeric(by_name.get(str(name), 0.0), errors="coerce")
        return 0.0 if pd.isna(value) else float(value)

    if boosted_driver:
        total += point_for(boosted_driver)
    if chip_mode == CHIP_TRIPLE and triple_driver and str(triple_driver) != str(boosted_driver):
        total += 2.0 * point_for(triple_driver)
    return total


def annotate_card_expected_points(
    drivers: pd.DataFrame,
    boosted_driver: str | None = None,
    triple_driver: str | None = None,
) -> pd.DataFrame:
    """Add display-only boosted points for cards without mutating unboosted exp_score."""
    out = drivers.copy()
    out["display_exp_score"] = pd.to_numeric(out.get("exp_score", pd.Series(index=out.index, dtype=float)), errors="coerce")
    if "name" not in out.columns:
        return out
    if boosted_driver:
        out.loc[out["name"].astype(str) == str(boosted_driver), "display_exp_score"] = (
            pd.to_numeric(out.loc[out["name"].astype(str) == str(boosted_driver), "exp_score"], errors="coerce") * 2.0
        )
    if triple_driver:
        out.loc[out["name"].astype(str) == str(triple_driver), "display_exp_score"] = (
            pd.to_numeric(out.loc[out["name"].astype(str) == str(triple_driver), "exp_score"], errors="coerce") * 3.0
        )
    return out


def _asset_names_by_id(df: pd.DataFrame, ids: list[str]) -> list[str]:
    if df.empty or "id" not in df.columns:
        return [str(x) for x in ids]
    names = df.assign(_id=df["id"].astype(str)).set_index("_id")["name"].astype(str).to_dict()
    return [names.get(str(x), str(x)) for x in ids]


def _asset_summary_map(df: pd.DataFrame) -> dict[str, dict]:
    if df.empty:
        return {}
    out = {}
    for row in df.itertuples(index=False):
        asset_id = str(getattr(row, "_id", getattr(row, "id", "")))
        out[asset_id] = {
            "id": asset_id,
            "name": str(getattr(row, "name", asset_id)),
            "team": str(getattr(row, "team", getattr(row, "name", ""))),
            "price": float(pd.to_numeric(getattr(row, "price", 0.0), errors="coerce") or 0.0),
            "exp_score": float(pd.to_numeric(getattr(row, "exp_score", 0.0), errors="coerce") or 0.0),
            "expected_price_gain": float(pd.to_numeric(getattr(row, "expected_price_gain", 0.0), errors="coerce") or 0.0),
        }
    return out


def recommendation_badges(row: dict | pd.Series, risk_appetite: str = "Balanced") -> list[str]:
    record = row if isinstance(row, dict) else row.to_dict()
    badges: list[str] = []
    delta_pts = float(pd.to_numeric(record.get("Expected points gain"), errors="coerce") or 0.0)
    delta_gain = float(pd.to_numeric(record.get("Expected price gain delta"), errors="coerce") or 0.0)
    remaining = float(pd.to_numeric(record.get("Remaining budget"), errors="coerce") or 0.0)
    penalty = float(pd.to_numeric(record.get("Transfer penalty"), errors="coerce") or 0.0)
    extra = int(pd.to_numeric(record.get("Extra transfers"), errors="coerce") or 0)
    volatility = float(pd.to_numeric(record.get("Incoming volatility mean"), errors="coerce") or 0.0)

    if delta_pts > 0:
        badges.append("Points upgrade")
    if delta_gain > 0:
        badges.append("Budget builder")
    if remaining > 0.5:
        badges.append("Frees cash")
    if float(pd.to_numeric(record.get("Outgoing negative gain count"), errors="coerce") or 0.0) > 0:
        badges.append("Avoids price drop")
    if volatility >= 18.0:
        badges.append("Risky / high variance")
    if penalty <= 0:
        badges.append("No penalty")
    if extra > 0:
        badges.append("Paid hit")
    if risk_appetite == "Conservative":
        badges.append("Conservative")
    if risk_appetite == "Aggressive":
        badges.append("Aggressive")
    return badges


def transfer_baseline(
    driver_ids: list[str],
    constructor_ids: list[str],
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float,
    chip_mode: str = CHIP_NONE,
) -> dict:
    d_ids = {str(x) for x in driver_ids}
    c_ids = {str(x) for x in constructor_ids}
    selected_d = drivers[drivers["id"].astype(str).isin(d_ids)].copy()
    selected_c = constructors[constructors["id"].astype(str).isin(c_ids)].copy()
    boosted_driver, triple_driver = select_chip_boost_drivers(selected_d, chip_mode)
    points = team_expected_points_with_chips(selected_d, selected_c, chip_mode, boosted_driver, triple_driver)
    price_gain = selected_assets_price_gain(selected_d, selected_c)
    cost = float(pd.to_numeric(selected_d.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    cost += float(pd.to_numeric(selected_c.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    return {
        "selected_drivers": selected_d,
        "selected_constructors": selected_c,
        "team_cost": cost,
        "remaining_budget": float(budget) - cost,
        "expected_points": points,
        "expected_price_gain": price_gain,
        "projected_team_value": projected_team_value_from_budget(float(budget), price_gain),
        "boosted_driver": boosted_driver,
        "triple_driver": triple_driver,
    }


def transfer_asset_max_price_gain(price: float | int | None, expensive_cutoff: float = DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF) -> float:
    numeric = pd.to_numeric(price, errors="coerce")
    if pd.isna(numeric):
        return 0.6
    return 0.6 if float(numeric) <= float(expensive_cutoff) else 0.3


def transfer_candidate_filter_score(
    row: pd.Series | dict,
    objective_mode: str = OBJECTIVE_COMBINED,
    price_gain_weight: float = 10.0,
) -> float:
    """Cheap search score for transfer candidate pre-filtering and beam pruning."""
    data = row if isinstance(row, pd.Series) else pd.Series(row)
    points = float(pd.to_numeric(data.get("exp_score", 0.0), errors="coerce") or 0.0)
    price = float(pd.to_numeric(data.get("price", 0.0), errors="coerce") or 0.0)
    gain = float(pd.to_numeric(data.get("expected_price_gain", data.get("expected_price_change", 0.0)), errors="coerce") or 0.0)
    volatility = float(pd.to_numeric(data.get("volatility", 0.0), errors="coerce") or 0.0)
    normalised_points = points / price if price > 0 else 0.0
    normalised_price_gain = gain / transfer_asset_max_price_gain(price)
    # Slider range is 0..100. Bring it onto a comparable scale to points-per-price.
    scaled_price_weight = float(price_gain_weight) / 10.0
    if objective_mode == OBJECTIVE_POINTS_ONLY:
        return normalised_points
    if objective_mode == OBJECTIVE_PRICE_GROWTH_ONLY:
        return normalised_price_gain
    if objective_mode == OBJECTIVE_RISK_ADJUSTED_COMBINED:
        risk_component = points / volatility if volatility > 0 else normalised_points
        return risk_component + scaled_price_weight * normalised_price_gain
    return normalised_points + scaled_price_weight * normalised_price_gain


def build_transfer_recommendations(
    current_driver_ids: list[str],
    current_constructor_ids: list[str],
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float,
    free_transfers: int = 2,
    max_transfers: int = 2,
    allow_extra_transfers: bool = True,
    transfer_penalty: float = 10.0,
    objective_mode: str = OBJECTIVE_POINTS_ONLY,
    price_gain_weight: float = 10.0,
    locked_driver_ids: list[str] | None = None,
    excluded_driver_ids: list[str] | None = None,
    locked_constructor_ids: list[str] | None = None,
    excluded_constructor_ids: list[str] | None = None,
    limitless: bool = False,
    chip_mode: str = CHIP_NONE,
    search_mode: str = "balanced",
    top_n: int = 25,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Generate transfer recommendations with optional fast/balanced pruning."""
    def _emit(stage: str, message: str, progress: float | None = None, details: dict[str, Any] | None = None) -> None:
        if progress_callback is None:
            return
        payload: dict[str, Any] = {"stage": stage, "message": message, "progress": progress}
        if details:
            payload.update(details)
        try:
            progress_callback(payload)
        except Exception:
            pass

    def _score_from_deltas(net_points_gain: float, price_gain_delta: float, volatility_sum: float) -> float:
        if objective_mode == OBJECTIVE_PRICE_GROWTH_ONLY:
            return float(price_gain_delta)
        if objective_mode == OBJECTIVE_COMBINED:
            return float(net_points_gain + float(price_gain_weight) * price_gain_delta)
        if objective_mode == OBJECTIVE_RISK_ADJUSTED_COMBINED:
            return float(net_points_gain + float(price_gain_weight) * (price_gain_delta / volatility_sum if volatility_sum > 0 else 0.0))
        return float(net_points_gain)

    def _sum_from_map(ids: tuple[str, ...], value_map: dict[str, float]) -> float:
        return float(sum(float(value_map.get(str(asset_id), 0.0)) for asset_id in ids))

    def _numeric_map(frame: pd.DataFrame, column: str) -> dict[str, float]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
        else:
            values = pd.Series(0.0, index=frame.index, dtype=float)
        return dict(zip(frame["_id"].astype(str), values))

    _emit("read_current_team", "Reading current team...", 0.02)
    current_driver_ids = [str(x) for x in current_driver_ids]
    current_constructor_ids = [str(x) for x in current_constructor_ids]
    locked_driver_set = {str(x) for x in locked_driver_ids or []}
    excluded_driver_set = {str(x) for x in excluded_driver_ids or []}
    locked_constructor_set = {str(x) for x in locked_constructor_ids or []}
    excluded_constructor_set = {str(x) for x in excluded_constructor_ids or []}
    _emit(
        "apply_locks_exclusions",
        "Applying locks and exclusions...",
        0.06,
        {
            "locked_total": len(locked_driver_set) + len(locked_constructor_set),
            "excluded_total": len(excluded_driver_set) + len(excluded_constructor_set),
        },
    )

    if len(current_driver_ids) != 5 or len(current_constructor_ids) != 2:
        _emit("ready", "Current team shape is invalid for transfer recommendations.", 1.0)
        return pd.DataFrame()

    drivers = drivers.copy()
    constructors = constructors.copy()
    drivers["_id"] = drivers["id"].astype(str)
    constructors["_id"] = constructors["id"].astype(str)
    driver_summary = _asset_summary_map(drivers)
    constructor_summary = _asset_summary_map(constructors)
    baseline = transfer_baseline(current_driver_ids, current_constructor_ids, drivers, constructors, budget, chip_mode=chip_mode)
    base_points = baseline["expected_points"]
    base_gain = baseline["expected_price_gain"]

    current_driver_set = set(current_driver_ids)
    current_constructor_set = set(current_constructor_ids)
    search_mode_key = str(search_mode or "balanced").strip().lower()
    if search_mode_key not in {"fast", "balanced", "exhaustive"}:
        search_mode_key = "balanced"
    max_transfers = max(1, min(int(max_transfers), 4))

    mode_config = {
        "fast": {
            "driver_incoming_limit": 8,
            "constructor_drop_bottom": 3,
            "candidate_pool_mode": "fast_prefiltered",
        },
        "balanced": {
            "driver_incoming_limit": 15,
            "constructor_drop_bottom": 2,
            "candidate_pool_mode": "balanced_prefiltered",
        },
        "exhaustive": {
            "driver_incoming_limit": None,
            "constructor_drop_bottom": 0,
            "candidate_pool_mode": "full",
        },
    }[search_mode_key]

    drivers["candidate_filter_score"] = drivers.apply(
        lambda row: transfer_candidate_filter_score(
            row,
            objective_mode=objective_mode,
            price_gain_weight=price_gain_weight,
        ),
        axis=1,
    )
    constructors["candidate_filter_score"] = constructors.apply(
        lambda row: transfer_candidate_filter_score(
            row,
            objective_mode=objective_mode,
            price_gain_weight=price_gain_weight,
        ),
        axis=1,
    )

    driver_filter_score_map = _numeric_map(drivers, "candidate_filter_score")
    constructor_filter_score_map = _numeric_map(constructors, "candidate_filter_score")
    combined_assets = pd.concat([drivers, constructors], ignore_index=True)

    incoming_driver_df = drivers[
        ~drivers["_id"].isin(current_driver_set) & ~drivers["_id"].isin(excluded_driver_set)
    ].copy()
    incoming_constructor_df = constructors[
        ~constructors["_id"].isin(current_constructor_set) & ~constructors["_id"].isin(excluded_constructor_set)
    ].copy()
    incoming_driver_ids_all = incoming_driver_df["_id"].astype(str).tolist()
    incoming_constructor_ids_all = incoming_constructor_df["_id"].astype(str).tolist()
    required_locked_incoming_drivers = sorted((locked_driver_set - current_driver_set) - excluded_driver_set)
    required_locked_incoming_constructors = sorted((locked_constructor_set - current_constructor_set) - excluded_constructor_set)

    prefilter_pruned = 0
    _emit("filter_candidates", "Filtering candidate assets...", 0.10)
    if search_mode_key != "exhaustive":
        driver_limit = mode_config["driver_incoming_limit"]
        if driver_limit is not None:
            ranked_driver_ids = incoming_driver_df.sort_values("candidate_filter_score", ascending=False, na_position="last")["_id"].astype(str).tolist()
            kept_driver_ids = ranked_driver_ids[: int(driver_limit)]
            kept_driver_ids = sorted(set(kept_driver_ids) | set(required_locked_incoming_drivers))
            prefilter_pruned += max(0, len(incoming_driver_df) - len(kept_driver_ids))
            incoming_driver_df = incoming_driver_df[incoming_driver_df["_id"].isin(kept_driver_ids)].copy()
        drop_bottom = int(mode_config["constructor_drop_bottom"] or 0)
        if drop_bottom > 0 and len(incoming_constructor_df) > drop_bottom:
            ranked_constructor_ids = incoming_constructor_df.sort_values("candidate_filter_score", ascending=True, na_position="last")["_id"].astype(str).tolist()
            dropped = set(ranked_constructor_ids[:drop_bottom]) - set(required_locked_incoming_constructors)
            kept_constructor_ids = [cid for cid in incoming_constructor_df["_id"].astype(str).tolist() if cid not in dropped]
            prefilter_pruned += max(0, len(incoming_constructor_df) - len(kept_constructor_ids))
            incoming_constructor_df = incoming_constructor_df[incoming_constructor_df["_id"].isin(kept_constructor_ids)].copy()

    candidate_driver_ids = incoming_driver_df["_id"].astype(str).tolist()
    candidate_constructor_ids = incoming_constructor_df["_id"].astype(str).tolist()
    removable_drivers = [x for x in current_driver_ids if x not in locked_driver_set]
    removable_constructors = [x for x in current_constructor_ids if x not in locked_constructor_set]
    outgoing_driver_candidates = removable_drivers
    outgoing_constructor_candidates = removable_constructors

    generation_started = datetime.now(UTC)
    generated_partial_plans = 0
    duplicate_teams_skipped = 0
    pruned_by_budget = 0
    pruned_by_beam = 0

    generated_by_depth: dict[int, int] = {depth: 0 for depth in range(1, max_transfers + 1)}
    beam_kept_by_depth: dict[int, int] = {depth: 0 for depth in range(1, max_transfers + 1)}
    fully_scored_by_depth: dict[int, int] = {depth: 0 for depth in range(1, max_transfers + 1)}
    final_recommendations_by_transfer_count: dict[int, int] = {}
    finalist_specs: list[
        tuple[
            float,
            tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]],
            int,
        ]
    ] = []
    seen_team_keys: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    team_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}

    def _count_generation_iterations(driver_pool: list[str], constructor_pool: list[str]) -> int:
        total = 0
        for transfers in range(1, max_transfers + 1):
            if not allow_extra_transfers and transfers > int(free_transfers):
                continue
            for d_k in range(0, min(5, transfers) + 1):
                c_k = transfers - d_k
                if c_k < 0 or c_k > 2:
                    continue
                if d_k > len(outgoing_driver_candidates) or d_k > len(driver_pool):
                    continue
                if c_k > len(outgoing_constructor_candidates) or c_k > len(constructor_pool):
                    continue
                total += (
                    math.comb(len(outgoing_driver_candidates), d_k)
                    * math.comb(len(driver_pool), d_k)
                    * math.comb(len(outgoing_constructor_candidates), c_k)
                    * math.comb(len(constructor_pool), c_k)
                )
        return int(total)

    total_generation_iterations_before_filtering = _count_generation_iterations(incoming_driver_ids_all, incoming_constructor_ids_all)
    total_generation_iterations_after_filtering = max(1, _count_generation_iterations(candidate_driver_ids, candidate_constructor_ids))
    generation_iterations_processed = 0
    _emit("generate_candidates", "Generating valid transfer plans...", 0.18)

    for transfers in range(1, max_transfers + 1):
        if not allow_extra_transfers and transfers > int(free_transfers):
            continue
        _emit("generate_candidates", f"Generating {transfers}-transfer candidates...", 0.18)
        for d_k in range(0, min(5, transfers) + 1):
            c_k = transfers - d_k
            if c_k < 0 or c_k > 2:
                continue
            if d_k > len(outgoing_driver_candidates) or d_k > len(candidate_driver_ids):
                continue
            if c_k > len(outgoing_constructor_candidates) or c_k > len(candidate_constructor_ids):
                continue
            for d_out in combinations(outgoing_driver_candidates, d_k):
                remaining_drivers = [x for x in current_driver_ids if x not in set(d_out)]
                for d_in in combinations(candidate_driver_ids, d_k):
                    new_driver_ids = remaining_drivers + list(d_in)
                    if locked_driver_set and not locked_driver_set <= set(new_driver_ids):
                        generation_iterations_processed += math.comb(len(outgoing_constructor_candidates), c_k) * math.comb(len(candidate_constructor_ids), c_k)
                        continue
                    for c_out in combinations(outgoing_constructor_candidates, c_k):
                        remaining_constructors = [x for x in current_constructor_ids if x not in set(c_out)]
                        for c_in in combinations(candidate_constructor_ids, c_k):
                            generation_iterations_processed += 1
                            if generation_iterations_processed <= 10 or generation_iterations_processed % 500 == 0:
                                progress = 0.18 + 0.07 * (
                                    generation_iterations_processed / total_generation_iterations_after_filtering
                                )
                                _emit(
                                    "generate_candidates",
                                    (
                                        f"Generating {transfers}-transfer candidates... "
                                        f"{generation_iterations_processed:,} / {total_generation_iterations_after_filtering:,} checked"
                                    ),
                                    min(0.25, float(progress)),
                                )
                            new_constructor_ids = remaining_constructors + list(c_in)
                            if locked_constructor_set and not locked_constructor_set <= set(new_constructor_ids):
                                continue
                            team_key = (tuple(sorted(new_driver_ids)), tuple(sorted(new_constructor_ids)))
                            if team_key in seen_team_keys:
                                duplicate_teams_skipped += 1
                                continue
                            seen_team_keys.add(team_key)
                            approx_filter_in = _sum_from_map(tuple(d_in), driver_filter_score_map) + _sum_from_map(tuple(c_in), constructor_filter_score_map)
                            approx_filter_out = _sum_from_map(tuple(d_out), driver_filter_score_map) + _sum_from_map(tuple(c_out), constructor_filter_score_map)
                            candidate_filter_score_value = float(approx_filter_in - approx_filter_out)

                            finalist_specs.append(
                                (
                                    float(candidate_filter_score_value),
                                    (tuple(d_out), tuple(d_in), tuple(c_out), tuple(c_in)),
                                    transfers,
                                )
                            )
                            generated_partial_plans += 1
                            generated_by_depth[transfers] += 1

    beam_kept_by_depth = dict(generated_by_depth)
    candidate_count_total = len(finalist_specs)
    generation_elapsed = max(0.0, (datetime.now(UTC) - generation_started).total_seconds())

    common_diag = {
        "search_mode": search_mode_key,
        "candidate_pool_mode": str(mode_config["candidate_pool_mode"]),
        "max_transfers": int(max_transfers),
        "candidate_filter_score_used_for_prefilter": bool(search_mode_key != "exhaustive"),
        "exhaustive_scoring_used_after_prefilter": True,
        "final_score_used_for_sorting": True,
        "incoming_driver_candidates": int(len(candidate_driver_ids)),
        "incoming_constructor_candidates": int(len(candidate_constructor_ids)),
        "incoming_driver_candidates_kept": int(len(candidate_driver_ids)),
        "incoming_constructor_candidates_kept": int(len(candidate_constructor_ids)),
        "outgoing_driver_candidates": int(len(outgoing_driver_candidates)),
        "outgoing_constructor_candidates": int(len(outgoing_constructor_candidates)),
        "outgoing_driver_candidates_kept": int(len(outgoing_driver_candidates)),
        "outgoing_constructor_candidates_kept": int(len(outgoing_constructor_candidates)),
        "exhaustive_candidate_count_before_filtering": int(total_generation_iterations_before_filtering),
        "candidate_count_after_filtering": int(total_generation_iterations_after_filtering),
        "valid_transfer_plans_generated": int(candidate_count_total),
        "generated_partial_plans": int(generated_partial_plans),
        "generated_candidates_by_depth": dict(generated_by_depth),
        "beam_kept_by_depth": dict(beam_kept_by_depth),
        "number_candidates_generated": int(sum(generated_by_depth.values())),
        "total_candidates_generated": int(sum(generated_by_depth.values())),
        "number_pruned_by_filtering": int(prefilter_pruned),
        "duplicate_teams_skipped": int(duplicate_teams_skipped),
        "pruned_by_budget": int(pruned_by_budget),
        "pruned_by_beam": int(pruned_by_beam),
        "transfer_generation_duration_seconds": float(generation_elapsed),
        "transfer_scoring_duration_seconds": 0.0,
        "transfer_total_duration_seconds": float(generation_elapsed),
    }

    _emit(
        "generate_candidates",
        f"Generating candidate transfers... {candidate_count_total:,} finalists selected",
        0.25,
        {
            **common_diag,
            "transfer_candidate_count_total": int(candidate_count_total),
            "transfer_candidates_evaluated": 0,
            "transfer_candidates_scored": 0,
            "transfer_candidates_filtered": 0,
            "candidate_teams_scored": 0,
            "transfer_candidate_count": int(candidate_count_total),
            "evaluated_full_candidates": 0,
            "total_candidates_fully_scored": 0,
            "fully_scored_by_depth": dict(fully_scored_by_depth),
            "final_recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
            "recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
        },
    )

    if candidate_count_total == 0:
        _emit(
            "ready",
            "No valid transfer recommendations found.",
            1.0,
            {
                **common_diag,
                "transfer_candidate_count_total": 0,
                "transfer_candidates_evaluated": 0,
                "transfer_candidates_scored": 0,
                "transfer_candidates_filtered": 0,
                "candidate_teams_scored": 0,
                "transfer_results_count": 0,
                "transfer_candidate_count": 0,
                "evaluated_full_candidates": 0,
                "total_candidates_fully_scored": 0,
                "fully_scored_by_depth": dict(fully_scored_by_depth),
                "final_recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
                "recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
            },
        )
        return pd.DataFrame()

    rows: list[dict] = []
    candidate_evaluated = 0
    candidate_filtered = 0
    candidate_scored = 0
    scoring_started = datetime.now(UTC)

    for candidate_evaluated, (candidate_filter_score_value, (d_out, d_in, c_out, c_in), depth) in enumerate(
        finalist_specs,
        start=1,
    ):
        remaining_drivers = [x for x in current_driver_ids if x not in set(d_out)]
        new_driver_ids = remaining_drivers + list(d_in)
        remaining_constructors = [x for x in current_constructor_ids if x not in set(c_out)]
        new_constructor_ids = remaining_constructors + list(c_in)

        new_d = drivers[drivers["_id"].isin(new_driver_ids)].copy()
        new_c = constructors[constructors["_id"].isin(new_constructor_ids)].copy()
        score_progress = 0.25 + 0.70 * (float(candidate_evaluated) / float(max(candidate_count_total, 1)))
        if candidate_evaluated <= 10 or candidate_evaluated % 100 == 0 or candidate_evaluated == candidate_count_total:
            _emit(
                "score_candidates",
                f"Scoring candidate teams... {candidate_evaluated:,} / {candidate_count_total:,} evaluated",
                min(0.95, score_progress),
                {
                    **common_diag,
                    "transfer_candidate_count_total": int(candidate_count_total),
                    "transfer_candidates_evaluated": int(candidate_evaluated),
                    "transfer_candidates_scored": int(candidate_scored),
                    "transfer_candidates_filtered": int(candidate_filtered),
                    "candidate_teams_scored": int(candidate_scored),
                    "transfer_candidate_count": int(candidate_count_total),
                    "evaluated_full_candidates": int(candidate_scored),
                    "total_candidates_fully_scored": int(candidate_scored),
                    "fully_scored_by_depth": dict(fully_scored_by_depth),
                    "final_recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
                    "recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
                    "transfer_scoring_duration_seconds": float(max(0.0, (datetime.now(UTC) - scoring_started).total_seconds())),
                    "transfer_total_duration_seconds": float(max(0.0, (datetime.now(UTC) - generation_started).total_seconds())),
                },
            )

        transfers = len(d_out) + len(c_out)
        if len(new_d) != 5 or len(new_c) != 2:
            candidate_filtered += 1
            continue
        cost = float(
            pd.to_numeric(new_d["price"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(new_c["price"], errors="coerce").fillna(0).sum()
        )
        if not limitless and cost > float(budget):
            candidate_filtered += 1
            pruned_by_budget += 1
            continue

        cache_key = (tuple(sorted(new_driver_ids)), tuple(sorted(new_constructor_ids)))
        if cache_key in team_cache:
            cached = team_cache[cache_key]
            boosted_driver = cached["boosted_driver"]
            triple_driver = cached["triple_driver"]
            points = cached["points"]
            gain = cached["gain"]
            volatility_sum = cached["volatility_sum"]
        else:
            boosted_driver, triple_driver = select_chip_boost_drivers(new_d, chip_mode)
            points = team_expected_points_with_chips(new_d, new_c, chip_mode, boosted_driver, triple_driver)
            gain = selected_assets_price_gain(new_d, new_c)
            volatility_sum = float(
                pd.to_numeric(
                    pd.concat([new_d.get("volatility", pd.Series(dtype=float)), new_c.get("volatility", pd.Series(dtype=float))]),
                    errors="coerce",
                ).fillna(0.0).sum()
            )
            team_cache[cache_key] = {
                "boosted_driver": boosted_driver,
                "triple_driver": triple_driver,
                "points": points,
                "gain": gain,
                "volatility_sum": volatility_sum,
            }

        candidate_scored += 1
        fully_scored_by_depth[depth] = fully_scored_by_depth.get(depth, 0) + 1

        extra = max(0, transfers - int(free_transfers))
        penalty = float(transfer_penalty) * extra
        points_gain = float(points - base_points)
        net_points_gain = float(points_gain - penalty)
        price_gain_delta = float(gain - base_gain)
        objective_improvement = _score_from_deltas(net_points_gain, price_gain_delta, volatility_sum)

        if objective_mode == OBJECTIVE_POINTS_ONLY:
            final_recommendation_score = net_points_gain
        elif objective_mode == OBJECTIVE_PRICE_GROWTH_ONLY:
            final_recommendation_score = price_gain_delta
        else:
            final_recommendation_score = objective_improvement

        if net_points_gain > 0 and price_gain_delta < 0:
            explanation = f"This improves expected points but sacrifices {price_gain_delta:+.2f}M expected price gain."
        elif net_points_gain < 0 and price_gain_delta > 0:
            explanation = f"This improves expected price gain but costs {abs(net_points_gain):.2f} expected points."
        elif net_points_gain > 0 and price_gain_delta > 0:
            explanation = "This improves expected points and expected price gain."
        else:
            explanation = "This is a trade-off move with mixed upside."

        move_rows: list[dict] = []
        for out_id, in_id in zip(d_out, d_in):
            move_rows.append(
                {
                    "asset_type": "driver",
                    "out": driver_summary.get(str(out_id), {"id": str(out_id), "name": str(out_id)}),
                    "in": driver_summary.get(str(in_id), {"id": str(in_id), "name": str(in_id)}),
                }
            )
        for out_id, in_id in zip(c_out, c_in):
            move_rows.append(
                {
                    "asset_type": "constructor",
                    "out": constructor_summary.get(str(out_id), {"id": str(out_id), "name": str(out_id)}),
                    "in": constructor_summary.get(str(in_id), {"id": str(in_id), "name": str(in_id)}),
                }
            )

        incoming_vol = pd.to_numeric(
            pd.concat([new_d.get("volatility", pd.Series(dtype=float)), new_c.get("volatility", pd.Series(dtype=float))]),
            errors="coerce",
        )
        outgoing_assets = pd.concat(
            [
                drivers[drivers["_id"].isin([str(x) for x in d_out])],
                constructors[constructors["_id"].isin([str(x) for x in c_out])],
            ],
            ignore_index=True,
        )
        outgoing_negative_count = int(
            (
                pd.to_numeric(
                    outgoing_assets.get("expected_price_gain", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0.0)
                < 0
            ).sum()
        )
        projected_value = projected_team_value_from_budget(float(budget), gain)
        base_projected_value = projected_team_value_from_budget(float(budget), base_gain)

        rows.append(
            {
                "Transfers": transfers,
                "OUT": ", ".join(_asset_names_by_id(combined_assets, list(d_out) + list(c_out))),
                "IN": ", ".join(_asset_names_by_id(combined_assets, list(d_in) + list(c_in))),
                "Team cost": round(cost, 2),
                "Remaining budget": round(float(budget) - cost, 2),
                "Expected points": round(points, 2),
                "Expected points gain": round(points_gain, 2),
                "Transfer penalty": round(penalty, 2),
                "Net expected points gain": round(net_points_gain, 2),
                "Expected price gain": round(gain, 2),
                "Expected price gain delta": round(price_gain_delta, 2),
                "Projected team value": round(projected_value, 2),
                "Projected team value delta": round(projected_value - base_projected_value, 2),
                "Objective improvement": round(float(objective_improvement), 4),
                "Candidate filter score": round(float(candidate_filter_score_value), 4),
                "Final recommendation score": round(float(final_recommendation_score), 4),
                "Extra transfers": int(extra),
                "2x driver": boosted_driver or "",
                "3x driver": triple_driver or "",
                "Move rows": move_rows,
                "Incoming volatility mean": float(incoming_vol.mean()) if len(incoming_vol.dropna()) else 0.0,
                "Outgoing negative gain count": outgoing_negative_count,
                "Explanation": explanation,
            }
        )

    scoring_elapsed = float(max(0.0, (datetime.now(UTC) - scoring_started).total_seconds())) if scoring_started else 0.0
    total_elapsed = float(max(0.0, (datetime.now(UTC) - generation_started).total_seconds()))

    if not rows:
        _emit(
            "ready",
            "No valid transfer recommendations found.",
            1.0,
            {
                **common_diag,
                "transfer_candidate_count_total": int(candidate_count_total),
                "transfer_candidates_evaluated": int(candidate_evaluated),
                "transfer_candidates_scored": int(candidate_scored),
                "transfer_candidates_filtered": int(candidate_filtered),
                "candidate_teams_scored": int(candidate_scored),
                "transfer_results_count": 0,
                "transfer_candidate_count": int(candidate_count_total),
                "evaluated_full_candidates": int(candidate_scored),
                "total_candidates_fully_scored": int(candidate_scored),
                "fully_scored_by_depth": dict(fully_scored_by_depth),
                "final_recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
                "recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
                "pruned_by_budget": int(pruned_by_budget),
                "transfer_scoring_duration_seconds": float(scoring_elapsed),
                "transfer_total_duration_seconds": float(total_elapsed),
            },
        )
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    _emit(
        "rank_recommendations",
        "Ranking recommendations...",
        0.97,
        {
            **common_diag,
            "transfer_candidate_count_total": int(candidate_count_total),
            "transfer_candidates_evaluated": int(candidate_evaluated),
            "transfer_candidates_scored": int(candidate_scored),
            "transfer_candidates_filtered": int(candidate_filtered),
            "candidate_teams_scored": int(candidate_scored),
            "transfer_candidate_count": int(candidate_count_total),
            "evaluated_full_candidates": int(candidate_scored),
            "total_candidates_fully_scored": int(candidate_scored),
            "fully_scored_by_depth": dict(fully_scored_by_depth),
            "final_recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
            "recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
            "pruned_by_budget": int(pruned_by_budget),
            "transfer_scoring_duration_seconds": float(scoring_elapsed),
            "transfer_total_duration_seconds": float(total_elapsed),
        },
    )

    if objective_mode == OBJECTIVE_POINTS_ONLY:
        out = out.sort_values(
            ["Final recommendation score", "Expected price gain delta"],
            ascending=[False, False],
            na_position="last",
        )
    elif objective_mode == OBJECTIVE_PRICE_GROWTH_ONLY:
        out = out.sort_values(
            ["Final recommendation score", "Net expected points gain"],
            ascending=[False, False],
            na_position="last",
        )
    else:
        out = out.sort_values(
            ["Final recommendation score", "Net expected points gain", "Expected price gain delta"],
            ascending=[False, False, False],
            na_position="last",
        )

    out = out.reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    result = out.head(int(top_n)).copy()
    final_recommendations_by_transfer_count = {
        int(k): int(v) for k, v in result["Transfers"].value_counts().sort_index().to_dict().items()
    }

    _emit(
        "ready",
        f"Ready. {len(result)} recommendations generated.",
        1.0,
        {
            **common_diag,
            "transfer_candidate_count_total": int(candidate_count_total),
            "transfer_candidates_evaluated": int(candidate_evaluated),
            "transfer_candidates_scored": int(candidate_scored),
            "transfer_candidates_filtered": int(candidate_filtered),
            "candidate_teams_scored": int(candidate_scored),
            "transfer_results_count": int(len(result)),
            "transfer_candidate_count": int(candidate_count_total),
            "evaluated_full_candidates": int(candidate_scored),
            "total_candidates_fully_scored": int(candidate_scored),
            "fully_scored_by_depth": dict(fully_scored_by_depth),
            "final_recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
            "recommendations_by_transfer_count": dict(final_recommendations_by_transfer_count),
            "pruned_by_budget": int(pruned_by_budget),
            "transfer_scoring_duration_seconds": float(scoring_elapsed),
            "transfer_total_duration_seconds": float(total_elapsed),
        },
    )
    return result


def format_transfer_recommendations_display(recs: pd.DataFrame) -> pd.DataFrame:
    if recs.empty:
        return recs.copy()
    hidden = {"Explanation", "Move rows", "Incoming volatility mean", "Outgoing negative gain count"}
    out = recs.drop(columns=[col for col in hidden if col in recs.columns], errors="ignore").copy()
    numeric_formats = {
        "Team cost": 2,
        "Remaining budget": 2,
        "Expected points": 2,
        "Expected points gain": 2,
        "Transfer penalty": 2,
        "Net expected points gain": 2,
        "Expected price gain": 2,
        "Expected price gain delta": 2,
        "Projected team value": 2,
        "Projected team value delta": 2,
        "Objective improvement": 2,
    }
    for col, digits in numeric_formats.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(digits)
    return out


def predicted_three_race_average(
    points_race_minus_2: float,
    points_race_minus_1: float,
    predicted_next_points: float,
) -> float:
    if pd.isna(points_race_minus_2) or pd.isna(points_race_minus_1) or pd.isna(predicted_next_points):
        return pd.NA
    return (float(points_race_minus_2) + float(points_race_minus_1) + float(predicted_next_points)) / 3.0


def avg_ppm_from_points(avg_points: float, current_price: float) -> float:
    if pd.isna(avg_points) or pd.isna(current_price):
        return pd.NA
    price = float(current_price)
    if price <= 0:
        return pd.NA
    return float(avg_points) / price


def predicted_avg_ppm(
    points_race_minus_2: float,
    points_race_minus_1: float,
    predicted_next_points: float,
    current_price: float,
) -> float:
    avg_points = predicted_three_race_average(points_race_minus_2, points_race_minus_1, predicted_next_points)
    return avg_ppm_from_points(avg_points, current_price)


def required_next_points(
    current_price: float,
    target_avg_ppm: float,
    points_race_minus_2: float,
    points_race_minus_1: float,
) -> float:
    return 3.0 * float(current_price) * float(target_avg_ppm) - float(points_race_minus_2) - float(points_race_minus_1)


def price_change_tier(avg_ppm: float, rules: PriceChangeRules | dict) -> str:
    if isinstance(rules, dict):
        rules = PriceChangeRules(**rules)

    if pd.isna(avg_ppm):
        return "Missing"
    ppm = float(avg_ppm)
    if ppm <= float(rules.terrible_max):
        return "Terrible"
    if ppm < float(rules.poor_max):
        return "Poor"
    if ppm < float(rules.great_min):
        return "Good"
    return "Great"


def raw_price_change_for_tier(tier: str, rules: PriceChangeRules | dict) -> float:
    if isinstance(rules, dict):
        rules = PriceChangeRules(**rules)

    if tier == "Terrible":
        return float(rules.terrible_price_change)
    if tier == "Poor":
        return float(rules.poor_price_change)
    if tier == "Good":
        return float(rules.good_price_change)
    if tier == "Great":
        return float(rules.great_price_change)
    return 0.0


def expected_price_change(avg_ppm: float, rules: PriceChangeRules | dict) -> float:
    return raw_price_change_for_tier(price_change_tier(avg_ppm, rules), rules)


def _normal_cdf(x: float, mean: float, sd: float) -> float:
    if pd.isna(x) or pd.isna(mean) or pd.isna(sd):
        return float("nan")
    sd = abs(float(sd))
    if sd <= 0:
        sd = DEFAULT_PRICE_GAIN_VOLATILITY_FALLBACK
    z = (float(x) - float(mean)) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def band_probabilities_from_normal(
    mean: float,
    sd: float,
    thresholds: dict[str, float],
    dnf_rate: float | None = 0.0,
    dnf_score: float = DEFAULT_DNF_PRICE_GAIN_SCORE,
) -> dict[str, float]:
    """Return band probabilities for a Normal score distribution with optional DNF tail risk."""
    required = {"terrible_max", "poor_max", "great_min"}
    if not required <= set(thresholds):
        missing = sorted(required - set(thresholds))
        raise ValueError(f"Missing thresholds for band probabilities: {missing}")
    if pd.isna(mean):
        return {
            "p_terrible": pd.NA,
            "p_poor": pd.NA,
            "p_good": pd.NA,
            "p_great": pd.NA,
            "p_price_rise": pd.NA,
            "p_price_fall": pd.NA,
        }

    terrible_max = float(thresholds["terrible_max"])
    poor_max = float(thresholds["poor_max"])
    great_min = float(thresholds["great_min"])
    cdf_terrible = _normal_cdf(terrible_max, mean, sd)
    cdf_poor = _normal_cdf(poor_max, mean, sd)
    cdf_great = _normal_cdf(great_min, mean, sd)
    if any(pd.isna(value) for value in [cdf_terrible, cdf_poor, cdf_great]):
        return {
            "p_terrible": pd.NA,
            "p_poor": pd.NA,
            "p_good": pd.NA,
            "p_great": pd.NA,
            "p_price_rise": pd.NA,
            "p_price_fall": pd.NA,
        }
    normal_probs = {
        "p_terrible": float(cdf_terrible),
        "p_poor": float(max(0.0, cdf_poor - cdf_terrible)),
        "p_good": float(max(0.0, cdf_great - cdf_poor)),
        "p_great": float(max(0.0, 1.0 - cdf_great)),
    }
    dnf = pd.to_numeric(dnf_rate, errors="coerce")
    dnf = 0.0 if pd.isna(dnf) else min(max(float(dnf), 0.0), 1.0)
    dnf_band = {"p_terrible": 0.0, "p_poor": 0.0, "p_good": 0.0, "p_great": 0.0}
    dnf_score = pd.to_numeric(dnf_score, errors="coerce")
    if pd.isna(dnf_score):
        dnf_score = DEFAULT_DNF_PRICE_GAIN_SCORE
    if float(dnf_score) <= terrible_max:
        dnf_band["p_terrible"] = 1.0
    elif float(dnf_score) < poor_max:
        dnf_band["p_poor"] = 1.0
    elif float(dnf_score) < great_min:
        dnf_band["p_good"] = 1.0
    else:
        dnf_band["p_great"] = 1.0

    p_terrible = (1.0 - dnf) * normal_probs["p_terrible"] + dnf * dnf_band["p_terrible"]
    p_poor = (1.0 - dnf) * normal_probs["p_poor"] + dnf * dnf_band["p_poor"]
    p_good = (1.0 - dnf) * normal_probs["p_good"] + dnf * dnf_band["p_good"]
    p_great = (1.0 - dnf) * normal_probs["p_great"] + dnf * dnf_band["p_great"]
    p_price_fall = float(p_terrible + p_poor)
    p_price_rise = float(p_good + p_great)
    total = p_terrible + p_poor + p_good + p_great
    if total > 0:
        p_terrible /= total
        p_poor /= total
        p_good /= total
        p_great /= total
        p_price_fall = p_terrible + p_poor
        p_price_rise = p_good + p_great
    return {
        "p_terrible": p_terrible,
        "p_poor": p_poor,
        "p_good": p_good,
        "p_great": p_great,
        "p_price_rise": p_price_rise,
        "p_price_fall": p_price_fall,
    }


def expected_price_gain_from_probabilities(
    probabilities: dict[str, float],
    price: float,
    rules: PriceChangeRules | dict,
    bounds: PriceChangeBounds | dict | None = None,
) -> dict[str, float]:
    """Convert band probabilities to expected and effective price gain metrics."""
    if isinstance(rules, dict):
        rules = PriceChangeRules(**rules)
    if bounds is None:
        bounds = PriceChangeBounds()
    elif isinstance(bounds, dict):
        bounds = PriceChangeBounds(**bounds)

    p_terrible = pd.to_numeric(probabilities.get("p_terrible"), errors="coerce")
    p_poor = pd.to_numeric(probabilities.get("p_poor"), errors="coerce")
    p_good = pd.to_numeric(probabilities.get("p_good"), errors="coerce")
    p_great = pd.to_numeric(probabilities.get("p_great"), errors="coerce")
    if any(pd.isna(value) for value in [p_terrible, p_poor, p_good, p_great]):
        return {
            "raw_expected_price_gain": pd.NA,
            "expected_price_gain": pd.NA,
            "projected_price_after_expected_gain": pd.NA,
            "risk_adjusted_price_gain": pd.NA,
            "expected_price_gain_per_million": pd.NA,
            "volatility_used": pd.NA,
            "volatility_fallback_used": pd.NA,
        }

    price = pd.to_numeric(price, errors="coerce")
    if pd.isna(price) or float(price) <= 0:
        return {
            "raw_expected_price_gain": pd.NA,
            "expected_price_gain": pd.NA,
            "projected_price_after_expected_gain": pd.NA,
            "risk_adjusted_price_gain": pd.NA,
            "expected_price_gain_per_million": pd.NA,
            "volatility_used": pd.NA,
            "volatility_fallback_used": pd.NA,
        }

    raw_expected_gain = (
        float(p_terrible) * float(rules.terrible_price_change)
        + float(p_poor) * float(rules.poor_price_change)
        + float(p_good) * float(rules.good_price_change)
        + float(p_great) * float(rules.great_price_change)
    )
    projected_price, effective_gain = clamp_price_change(float(price), raw_expected_gain, bounds)
    return {
        "raw_expected_price_gain": float(raw_expected_gain),
        "expected_price_gain": float(effective_gain),
        "projected_price_after_expected_gain": float(projected_price),
        "risk_adjusted_price_gain": pd.NA,
        "expected_price_gain_per_million": float(effective_gain / float(price)) if float(price) > 0 else pd.NA,
        "volatility_used": pd.NA,
        "volatility_fallback_used": pd.NA,
    }


def probabilistic_price_projection(
    asset_row: pd.Series | dict,
    mean: float,
    volatility: float,
    recent_points: tuple[float | int | None, float | int | None] | list[float | int | None] | pd.Series | None,
    rules: PriceChangeRules | dict,
    expensive_rules: PriceChangeRules | dict | None = None,
    expensive_price_min: float | None = None,
    bounds: PriceChangeBounds | dict | None = None,
    dnf_rate: float | None = None,
    dnf_score: float = DEFAULT_DNF_PRICE_GAIN_SCORE,
) -> dict[str, object]:
    """Project price gain using a Normal(next-race score) model."""
    if isinstance(rules, dict):
        rules = PriceChangeRules(**rules)
    if bounds is None:
        bounds = PriceChangeBounds()
    elif isinstance(bounds, dict):
        bounds = PriceChangeBounds(**bounds)

    row = asset_row if isinstance(asset_row, pd.Series) else pd.Series(asset_row)
    price = pd.to_numeric(row.get("price"), errors="coerce")
    recent_2, recent_1 = (pd.NA, pd.NA)
    if recent_points is not None:
        if isinstance(recent_points, pd.Series):
            recent_2 = recent_points.iloc[0] if len(recent_points) > 0 else pd.NA
            recent_1 = recent_points.iloc[1] if len(recent_points) > 1 else pd.NA
        else:
            recent_points = list(recent_points)
            recent_2 = recent_points[0] if len(recent_points) > 0 else pd.NA
            recent_1 = recent_points[1] if len(recent_points) > 1 else pd.NA

    selected_rules = (
        choose_price_change_rules(price, rules, expensive_rules=expensive_rules, expensive_price_min=expensive_price_min)
        if pd.notna(price)
        else rules
    )
    mean = pd.to_numeric(mean, errors="coerce")
    volatility = pd.to_numeric(volatility, errors="coerce")
    volatility_fallback_used = pd.isna(volatility) or float(volatility) <= 0
    volatility_used = DEFAULT_PRICE_GAIN_VOLATILITY_FALLBACK if volatility_fallback_used else float(volatility)
    if dnf_rate is None:
        dnf_rate = row.get("dnf_rate", 0.0)
    dnf_rate = pd.to_numeric(dnf_rate, errors="coerce")
    dnf_rate_used = 0.0 if pd.isna(dnf_rate) else min(max(float(dnf_rate), 0.0), 1.0)

    thresholds = {}
    if pd.notna(price) and pd.notna(recent_2) and pd.notna(recent_1):
        thresholds = {
            "terrible_max": required_next_points(price, selected_rules.terrible_max, recent_2, recent_1),
            "poor_max": required_next_points(price, selected_rules.poor_max, recent_2, recent_1),
            "great_min": required_next_points(price, selected_rules.great_min, recent_2, recent_1),
        }

    probs = band_probabilities_from_normal(
        mean,
        volatility_used,
        thresholds,
        dnf_rate=dnf_rate_used,
        dnf_score=dnf_score,
    ) if thresholds else {
        "p_terrible": pd.NA,
        "p_poor": pd.NA,
        "p_good": pd.NA,
        "p_great": pd.NA,
        "p_price_rise": pd.NA,
        "p_price_fall": pd.NA,
    }
    expected_gain = expected_price_gain_from_probabilities(probs, price, selected_rules, bounds)
    projected_avg_ppm = pd.NA
    projected_tier = "Missing"
    if pd.notna(price) and price > 0 and pd.notna(recent_2) and pd.notna(recent_1) and pd.notna(mean):
        projected_avg_ppm = predicted_avg_ppm(recent_2, recent_1, mean, price)
        projected_tier = price_change_tier(projected_avg_ppm, selected_rules)

    expected_points_per_million = float(mean) / float(price) if pd.notna(mean) and pd.notna(price) and float(price) > 0 else pd.NA
    expected_points_per_volatility = (float(mean) / volatility_used) if pd.notna(mean) and volatility_used > 0 else pd.NA
    risk_adjusted_price_gain = (
        expected_gain["expected_price_gain"] / volatility_used
        if pd.notna(expected_gain["expected_price_gain"]) and volatility_used > 0
        else pd.NA
    )

    out = {
        "price_change_predicted_next": float(mean) if pd.notna(mean) else pd.NA,
        "p_terrible": probs["p_terrible"],
        "p_poor": probs["p_poor"],
        "p_good": probs["p_good"],
        "p_great": probs["p_great"],
        "p_good_plus": (
            float(probs["p_good"]) + float(probs["p_great"])
            if pd.notna(probs["p_good"]) and pd.notna(probs["p_great"])
            else pd.NA
        ),
        "p_price_rise": probs["p_price_rise"],
        "p_price_fall": probs["p_price_fall"],
        "raw_expected_price_gain": expected_gain["raw_expected_price_gain"],
        "expected_price_gain": expected_gain["expected_price_gain"],
        "projected_price_after_expected_gain": expected_gain["projected_price_after_expected_gain"],
        "projected_price": expected_gain["projected_price_after_expected_gain"],
        "projected_avg_ppm": projected_avg_ppm,
        "projected_tier": projected_tier,
        "expected_price_gain_per_million": expected_gain["expected_price_gain_per_million"],
        "risk_adjusted_price_gain": risk_adjusted_price_gain,
        "expected_points_per_million": expected_points_per_million,
        "expected_points_per_volatility": expected_points_per_volatility,
        "volatility_used": float(volatility_used),
        "volatility_fallback_used": bool(volatility_fallback_used),
        "dnf_rate_used": float(dnf_rate_used),
        "dnf_score_used": float(dnf_score),
        "dnf_score_source": "fixed_generic_race_weekend_bad_outcome",
    }
    return out


def apply_probabilistic_price_change_model(
    df: pd.DataFrame,
    rules: PriceChangeRules | dict,
    expensive_rules: PriceChangeRules | dict | None = None,
    expensive_price_min: float | None = None,
    bounds: PriceChangeBounds | dict | None = None,
    predicted_points_col: str = "exp_score",
    volatility_col: str = "volatility",
    dnf_score: float = DEFAULT_DNF_PRICE_GAIN_SCORE,
) -> pd.DataFrame:
    """Augment the deterministic price model with probabilistic gain fields."""
    out = apply_price_change_model(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col=predicted_points_col,
    )
    rows = []
    for _, row in out.iterrows():
        recent_points = (row.get("recent_points_2ago"), row.get("recent_points_1ago"))
        rows.append(
            probabilistic_price_projection(
                row,
                mean=row.get("price_change_predicted_next", row.get("exp_score")),
                volatility=row.get(volatility_col, row.get("volatility")),
                recent_points=recent_points,
                rules=rules,
                expensive_rules=expensive_rules,
                expensive_price_min=expensive_price_min,
                bounds=bounds,
                dnf_rate=row.get("dnf_rate", 0.0),
                dnf_score=dnf_score,
            )
        )
    if rows:
        prob_df = pd.DataFrame(rows, index=out.index)
        for col in prob_df.columns:
            out[col] = prob_df[col]
    return out


def clamp_price_change(current_price: float, raw_price_change: float, bounds: PriceChangeBounds | dict) -> tuple[float, float]:
    if isinstance(bounds, dict):
        bounds = PriceChangeBounds(**bounds)
    price = float(current_price)
    raw_change = float(raw_price_change)
    projected_price = min(max(price + raw_change, float(bounds.min_asset_price)), float(bounds.max_asset_price))
    effective_change = projected_price - price
    return float(projected_price), float(effective_change)


def choose_price_change_rules(
    current_price: float,
    cheap_rules: PriceChangeRules | dict,
    expensive_rules: PriceChangeRules | dict | None = None,
    expensive_price_min: float | None = None,
) -> PriceChangeRules:
    cheap = PriceChangeRules(**cheap_rules) if isinstance(cheap_rules, dict) else cheap_rules
    if expensive_rules is None or expensive_price_min is None:
        return cheap
    expensive = PriceChangeRules(**expensive_rules) if isinstance(expensive_rules, dict) else expensive_rules
    return expensive if float(current_price) > float(expensive_price_min) else cheap


def apply_price_change_model(
    df: pd.DataFrame,
    rules: PriceChangeRules | dict,
    expensive_rules: PriceChangeRules | dict | None = None,
    expensive_price_min: float | None = None,
    bounds: PriceChangeBounds | dict | None = None,
    predicted_points_col: str = "exp_score",
) -> pd.DataFrame:
    out = clean_assumption_table(df)
    if isinstance(rules, dict):
        rules = PriceChangeRules(**rules)
    if bounds is None:
        bounds = PriceChangeBounds()
    elif isinstance(bounds, dict):
        bounds = PriceChangeBounds(**bounds)

    out = _fill_recent_point_columns(out)
    if predicted_points_col in out.columns:
        out["price_change_predicted_next"] = pd.to_numeric(out[predicted_points_col], errors="coerce")
    elif "exp_score" in out.columns:
        out["price_change_predicted_next"] = pd.to_numeric(out["exp_score"], errors="coerce")
    else:
        out["price_change_predicted_next"] = pd.NA
    out["avg_ppm"] = out.apply(
        lambda row: predicted_avg_ppm(row["recent_points_2ago"], row["recent_points_1ago"], row["price_change_predicted_next"], row["price"]),
        axis=1,
    )
    out["price_change_rule_group"] = out["price"].apply(
        lambda price: "Expensive" if expensive_rules is not None and expensive_price_min is not None and float(price) > float(expensive_price_min) else "Cheap"
    )
    out["price_change_tier"] = out.apply(
        lambda row: price_change_tier(
            row["avg_ppm"],
            choose_price_change_rules(row["price"], rules, expensive_rules=expensive_rules, expensive_price_min=expensive_price_min),
        ),
        axis=1,
    )
    out["raw_price_change"] = out.apply(
        lambda row: raw_price_change_for_tier(
            row["price_change_tier"],
            choose_price_change_rules(row["price"], rules, expensive_rules=expensive_rules, expensive_price_min=expensive_price_min),
        ),
        axis=1,
    )
    projected_effective = out.apply(
        lambda row: clamp_price_change(row["price"], row["raw_price_change"], bounds),
        axis=1,
    )
    out["projected_price"] = projected_effective.apply(lambda x: x[0])
    out["effective_price_change_after_floor_ceiling"] = projected_effective.apply(lambda x: x[1])
    out["expected_price_change"] = out["effective_price_change_after_floor_ceiling"]
    price = pd.to_numeric(out["price"], errors="coerce")
    volatility = pd.to_numeric(out.get("volatility", pd.Series(index=out.index, dtype=float)), errors="coerce")
    out["expected_price_gain_per_million"] = (out["expected_price_change"] / price).where(price > 0)
    out["risk_adjusted_price_gain"] = (out["expected_price_change"] / volatility).where(volatility > 0)
    out["expected_points_per_million"] = (out["price_change_predicted_next"] / price).where(price > 0)
    out["expected_points_per_volatility"] = (out["price_change_predicted_next"] / volatility).where(volatility > 0)
    return out


def price_change_threshold_table(
    df: pd.DataFrame,
    rules: PriceChangeRules | dict,
    expensive_rules: PriceChangeRules | dict | None = None,
    expensive_price_min: float | None = None,
    bounds: PriceChangeBounds | dict | None = None,
    predicted_points_col: str = "exp_score",
) -> pd.DataFrame:
    if isinstance(rules, dict):
        rules = PriceChangeRules(**rules)

    out = apply_price_change_model(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col=predicted_points_col,
    )
    out["terrible_threshold"] = out["price"].apply(lambda price: choose_price_change_rules(price, rules, expensive_rules, expensive_price_min).terrible_max)
    out["poor_range"] = out["price"].apply(
        lambda price: (
            f"{choose_price_change_rules(price, rules, expensive_rules, expensive_price_min).poor_min:g} "
            f"to {choose_price_change_rules(price, rules, expensive_rules, expensive_price_min).poor_max:g}"
        )
    )
    out["good_range"] = out["price"].apply(
        lambda price: (
            f"{choose_price_change_rules(price, rules, expensive_rules, expensive_price_min).good_min:g} "
            f"to {choose_price_change_rules(price, rules, expensive_rules, expensive_price_min).good_max:g}"
        )
    )
    out["great_threshold"] = out["price"].apply(lambda price: choose_price_change_rules(price, rules, expensive_rules, expensive_price_min).great_min)
    out["required_terrible_max"] = out.apply(
        lambda row: required_next_points(row["price"], choose_price_change_rules(row["price"], rules, expensive_rules, expensive_price_min).terrible_max, row["recent_points_2ago"], row["recent_points_1ago"]),
        axis=1,
    )
    out["required_poor_min"] = out.apply(
        lambda row: required_next_points(row["price"], choose_price_change_rules(row["price"], rules, expensive_rules, expensive_price_min).poor_min, row["recent_points_2ago"], row["recent_points_1ago"]),
        axis=1,
    )
    out["required_good_min"] = out.apply(
        lambda row: required_next_points(row["price"], choose_price_change_rules(row["price"], rules, expensive_rules, expensive_price_min).good_min, row["recent_points_2ago"], row["recent_points_1ago"]),
        axis=1,
    )
    out["required_great_min"] = out.apply(
        lambda row: required_next_points(row["price"], choose_price_change_rules(row["price"], rules, expensive_rules, expensive_price_min).great_min, row["recent_points_2ago"], row["recent_points_1ago"]),
        axis=1,
    )

    def rounded_boundary(value: float):
        if pd.isna(value):
            return pd.NA
        return int(round(float(value)))

    def ceil_boundary(value: float):
        if pd.isna(value):
            return pd.NA
        return int(math.ceil(float(value)))

    def fmt_boundary(value) -> str:
        if pd.isna(value):
            return "-"
        return str(int(value))

    def poor_points(row) -> str:
        terrible_max = rounded_boundary(row["required_terrible_max"])
        good_min = rounded_boundary(row["required_good_min"])
        if pd.isna(terrible_max) or pd.isna(good_min):
            return "- to -"
        return f"{int(terrible_max) + 1} to {int(good_min) - 1}"

    def good_points(row) -> str:
        good_min = rounded_boundary(row["required_good_min"])
        great_min = ceil_boundary(row["required_great_min"])
        if pd.isna(good_min) or pd.isna(great_min):
            return "- to -"
        return f"{int(good_min)} to {int(great_min) - 1}"

    out["points_needed_terrible"] = out["required_terrible_max"].apply(lambda value: f"≤ {fmt_boundary(rounded_boundary(value))}")
    out["points_needed_poor"] = out.apply(poor_points, axis=1)
    out["points_needed_good"] = out.apply(good_points, axis=1)
    out["points_needed_great"] = out["required_great_min"].apply(lambda value: f"≥ {fmt_boundary(ceil_boundary(value))}")

    price = pd.to_numeric(out["price"], errors="coerce")
    out["price_change_efficiency"] = (pd.to_numeric(out["required_great_min"], errors="coerce") / price).where(price > 0)
    return out


def price_change_target_summary_table(
    df: pd.DataFrame,
    rules: PriceChangeRules | dict,
    expensive_rules: PriceChangeRules | dict | None = None,
    expensive_price_min: float | None = None,
    bounds: PriceChangeBounds | dict | None = None,
    predicted_points_col: str = "exp_score",
) -> pd.DataFrame:
    table = price_change_threshold_table(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col=predicted_points_col,
    )
    if "tla" in table.columns:
        abbrev_col = "tla"
    elif "driver_reference" in table.columns:
        abbrev_col = "driver_reference"
    else:
        abbrev_col = "id"
    cols = [
        abbrev_col,
        "name",
        "team",
        "price",
        "points_needed_terrible",
        "points_needed_poor",
        "points_needed_good",
        "points_needed_great",
        "price_change_efficiency",
    ]
    if "team" not in table.columns:
        cols = [col for col in cols if col != "team"]
    out = table[[col for col in cols if col in table.columns]].copy()
    out.rename(
        columns={
            abbrev_col: "Abbrev",
            "name": "Name",
            "team": "Team",
            "price": "Price",
            "points_needed_terrible": "Terrible",
            "points_needed_poor": "Poor",
            "points_needed_good": "Good",
            "points_needed_great": "Great",
            "price_change_efficiency": "Rise difficulty",
        },
        inplace=True,
    )
    if "Rise difficulty" in out.columns:
        out = out.sort_values("Rise difficulty", ascending=True, na_position="last")
    return out


def price_change_projection_summary_table(
    df: pd.DataFrame,
    rules: PriceChangeRules | dict,
    expensive_rules: PriceChangeRules | dict | None = None,
    expensive_price_min: float | None = None,
    bounds: PriceChangeBounds | dict | None = None,
    predicted_points_col: str = "exp_score",
) -> pd.DataFrame:
    table = apply_probabilistic_price_change_model(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col=predicted_points_col,
    )
    if "tla" in table.columns:
        abbrev_col = "tla"
    elif "driver_reference" in table.columns:
        abbrev_col = "driver_reference"
    else:
        abbrev_col = "id"

    cols = [
        abbrev_col,
        "name",
        "team",
        "price",
        "price_change_predicted_next",
        "expected_price_gain",
    ]
    out = table[[col for col in cols if col in table.columns]].copy()
    out.rename(
        columns={
            abbrev_col: "Abbrev",
            "name": "Name",
            "team": "Team",
            "price": "Price",
            "price_change_predicted_next": "Expected Points",
            "expected_price_gain": "Expected price gain",
        },
        inplace=True,
    )
    sort_cols = [col for col in ["Expected price gain", "Expected Points"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return out


def price_change_probability_matrix_table(
    df: pd.DataFrame,
    rules: PriceChangeRules | dict,
    expensive_rules: PriceChangeRules | dict | None = None,
    expensive_price_min: float | None = None,
    bounds: PriceChangeBounds | dict | None = None,
    predicted_points_col: str = "exp_score",
) -> pd.DataFrame:
    table = apply_probabilistic_price_change_model(
        df,
        rules,
        expensive_rules=expensive_rules,
        expensive_price_min=expensive_price_min,
        bounds=bounds,
        predicted_points_col=predicted_points_col,
    )
    if "tla" in table.columns:
        abbrev_col = "tla"
    elif "driver_reference" in table.columns:
        abbrev_col = "driver_reference"
    else:
        abbrev_col = "id"
    cols = [
        abbrev_col,
        "name",
        "team",
        "price",
        "p_terrible",
        "p_poor",
        "p_good",
        "p_great",
        "price_change_predicted_next",
        "expected_price_gain",
    ]
    out = table[[col for col in cols if col in table.columns]].copy()
    out.rename(
        columns={
            abbrev_col: "Abbrev",
            "name": "Name",
            "team": "Team",
            "price": "Price",
            "price_change_predicted_next": "Expected Points",
            "p_terrible": "P(Terrible)",
            "p_poor": "P(Poor)",
            "p_good": "P(Good)",
            "p_great": "P(Great)",
            "expected_price_gain": "Expected price gain",
        },
        inplace=True,
    )
    if "Expected price gain" in out.columns:
        out = out.sort_values("Expected price gain", ascending=False, na_position="last")
    return out


def apply_objective_mode(
    df: pd.DataFrame,
    objective_mode: str,
    price_gain_weight: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    if "expected_price_gain" not in out.columns and "expected_price_change" in out.columns:
        out["expected_price_gain"] = out["expected_price_change"]
    if "expected_price_gain" not in out.columns:
        out["expected_price_gain"] = 0.0
    if "expected_price_change" not in out.columns:
        out["expected_price_change"] = out["expected_price_gain"]

    out["points_objective"] = pd.to_numeric(out["exp_score"], errors="coerce").fillna(0.0)
    out["price_growth_objective"] = pd.to_numeric(out["expected_price_gain"], errors="coerce").fillna(0.0)

    if objective_mode == OBJECTIVE_PRICE_GROWTH_ONLY:
        out["combined_objective_score"] = out["price_growth_objective"]
    elif objective_mode == OBJECTIVE_COMBINED:
        out["combined_objective_score"] = out["points_objective"] + float(price_gain_weight) * out["price_growth_objective"]
    elif objective_mode == OBJECTIVE_RISK_ADJUSTED_COMBINED:
        risk_gain = pd.to_numeric(out.get("risk_adjusted_price_gain", out["price_growth_objective"]), errors="coerce").fillna(0.0)
        out["combined_objective_score"] = out["points_objective"] + float(price_gain_weight) * risk_gain
    else:
        out["combined_objective_score"] = out["points_objective"]

    return out


def apply_no_negative_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["exp_score", "next_race_expected_points", "nn_exp_score", "combined_objective_score", "points_objective"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0)
    return out


def chip_mode_from_label(label: str) -> str:
    mapping = {
        "None": CHIP_NONE,
        "3x chip": CHIP_TRIPLE,
        "Limitless": CHIP_LIMITLESS,
        "No Negative chip": CHIP_NO_NEGATIVE,
    }
    return mapping.get(str(label), CHIP_NONE)


def run_optimizer(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float | None,
    top_k: int = DEFAULT_TOP_K,
    drs_multiplier: float = 2.0,
    allow_no_negative: bool = False,
    locked_driver_ids: list[str] | None = None,
    excluded_driver_ids: list[str] | None = None,
    locked_constructor_ids: list[str] | None = None,
    excluded_constructor_ids: list[str] | None = None,
    objective_col: str = "exp_score",
    boost_col: str = "exp_score",
    triple_multiplier: float | None = None,
) -> list[TeamSolution]:
    solutions = optimize_top_k(
        clean_assumption_table(drivers),
        clean_assumption_table(constructors),
        budget=None if budget is None else float(budget),
        k=int(top_k),
        drs_multiplier=float(drs_multiplier),
        allow_no_negative=bool(allow_no_negative),
        locked_driver_ids=locked_driver_ids,
        excluded_driver_ids=excluded_driver_ids,
        locked_constructor_ids=locked_constructor_ids,
        excluded_constructor_ids=excluded_constructor_ids,
        objective_col=objective_col,
        boost_col=boost_col,
        triple_multiplier=triple_multiplier,
    )
    # Always place display chips by highest expected points in the selected team.
    # This keeps 2x/3x assignment deterministic even when objective/weights make
    # chip placement irrelevant for optimisation (e.g. price-growth-only mode).
    chip_mode = CHIP_TRIPLE if triple_multiplier is not None else CHIP_NONE
    normalized: list[TeamSolution] = []
    for sol in solutions:
        boosted_driver, triple_driver = select_chip_boost_drivers(sol.drivers, chip_mode=chip_mode)
        normalized.append(
            TeamSolution(
                drivers=sol.drivers,
                constructors=sol.constructors,
                boosted_driver=boosted_driver,
                no_negative=sol.no_negative,
                limitless=sol.limitless,
                total_cost=sol.total_cost,
                expected_score=sol.expected_score,
                triple_driver=triple_driver,
            )
        )
    return normalized


def validate_current_team(
    driver_ids: list[str],
    constructor_ids: list[str],
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    budget: float = 100.0,
) -> dict:
    driver_ids = [str(x) for x in driver_ids]
    constructor_ids = [str(x) for x in constructor_ids]
    available_driver_ids = set(drivers["id"].astype(str))
    available_constructor_ids = set(constructors["id"].astype(str))

    selected_drivers = drivers[drivers["id"].astype(str).isin(driver_ids)].copy()
    selected_constructors = constructors[constructors["id"].astype(str).isin(constructor_ids)].copy()
    total_cost = float(selected_drivers["price"].sum() + selected_constructors["price"].sum())
    projected_points = float(selected_drivers["exp_score"].sum() + selected_constructors["exp_score"].sum())

    errors: list[str] = []
    warnings: list[str] = []

    if len(driver_ids) != 5:
        errors.append("Select exactly 5 drivers.")
    if len(constructor_ids) != 2:
        errors.append("Select exactly 2 constructors.")
    if len(set(driver_ids)) != len(driver_ids):
        errors.append("Duplicate driver selections found.")
    if len(set(constructor_ids)) != len(constructor_ids):
        errors.append("Duplicate constructor selections found.")

    missing_drivers = sorted(set(driver_ids) - available_driver_ids)
    missing_constructors = sorted(set(constructor_ids) - available_constructor_ids)
    if missing_drivers:
        errors.append(f"Unknown driver IDs: {missing_drivers}")
    if missing_constructors:
        errors.append(f"Unknown constructor IDs: {missing_constructors}")
    if total_cost > float(budget):
        warnings.append(f"Current team is over budget by {total_cost - float(budget):.1f}.")

    return {
        "valid": not errors and total_cost <= float(budget),
        "errors": errors,
        "warnings": warnings,
        "total_cost": total_cost,
        "projected_points": projected_points,
        "selected_drivers": selected_drivers,
        "selected_constructors": selected_constructors,
    }


def current_team_json(driver_ids: list[str], constructor_ids: list[str], free_transfers: int = 2, bank: float = 0.0) -> dict:
    def clean_id(value: str):
        value = str(value)
        return int(value) if value.isdigit() else value

    return {
        "drivers": [clean_id(x) for x in driver_ids],
        "constructors": [clean_id(x) for x in constructor_ids],
        "free_transfers": int(free_transfers),
        "bank": round(float(bank), 1),
    }


def parse_current_team_json_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("current_team.json must contain a JSON object.")

    drivers = payload.get("drivers", [])
    constructors = payload.get("constructors", [])
    free_transfers = payload.get("free_transfers", 2)
    bank = payload.get("bank", 0.0)

    if not isinstance(drivers, list) or not isinstance(constructors, list):
        raise ValueError("current_team.json must contain list values for drivers and constructors.")

    return {
        "drivers": [str(x) for x in drivers],
        "constructors": [str(x) for x in constructors],
        "free_transfers": int(free_transfers),
        "bank": float(bank),
    }


def load_current_team_json_text(text: str) -> dict:
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError("Invalid current_team.json: could not parse JSON.") from exc
    return parse_current_team_json_payload(payload)


def current_team_budget_from_selection(drivers: pd.DataFrame, constructors: pd.DataFrame, bank: float = 0.0) -> float:
    total = pd.to_numeric(drivers.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    total += pd.to_numeric(constructors.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    return float(total) + float(bank)


def current_team_upload_summary(
    payload: dict,
    available_driver_ids: list[str] | set[str],
    available_constructor_ids: list[str] | set[str],
) -> dict:
    parsed = parse_current_team_json_payload(payload)
    driver_ids = [str(x) for x in parsed["drivers"]]
    constructor_ids = [str(x) for x in parsed["constructors"]]
    available_driver_ids = {str(x) for x in available_driver_ids}
    available_constructor_ids = {str(x) for x in available_constructor_ids}
    missing_drivers = [x for x in driver_ids if x not in available_driver_ids]
    missing_constructors = [x for x in constructor_ids if x not in available_constructor_ids]
    return {
        **parsed,
        "drivers": [x for x in driver_ids if x in available_driver_ids],
        "constructors": [x for x in constructor_ids if x in available_constructor_ids],
        "missing_drivers": missing_drivers,
        "missing_constructors": missing_constructors,
    }


def format_selected_asset_display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gain_col = None
    for candidate in ["expected_price_gain", "expected_price_change", "effective_price_change_after_floor_ceiling"]:
        if candidate in out.columns:
            gain_col = candidate
            break
    rename_map = {
        "image_url": "Image",
        "name": "Name",
        "team": "Team",
        "price": "Price",
        "exp_score": "Expected / race",
        "projected_price": "Projected price",
        "dnf_rate": "DNF rate",
    }
    if gain_col:
        rename_map[gain_col] = "Expected price gain"
    if "raw_price_change" in out.columns:
        out = out.drop(columns=["raw_price_change"])
    cols = [col for col in ["image_url", "name", "team", "price", "exp_score", gain_col, "projected_price", "dnf_rate"] if col and col in out.columns]
    out = out[cols].copy()
    out.rename(columns=rename_map, inplace=True)
    if "Price" in out.columns:
        out["Price"] = pd.to_numeric(out["Price"], errors="coerce").round(2)
    if "Expected / race" in out.columns:
        out["Expected / race"] = pd.to_numeric(out["Expected / race"], errors="coerce").round(2)
    if "Expected price gain" in out.columns:
        out["Expected price gain"] = pd.to_numeric(out["Expected price gain"], errors="coerce").round(2)
    if "Projected price" in out.columns:
        out["Projected price"] = pd.to_numeric(out["Projected price"], errors="coerce").round(2)
    if "DNF rate" in out.columns:
        out["DNF rate"] = pd.to_numeric(out["DNF rate"], errors="coerce").round(3)
    if "Price" in out.columns:
        out = out.sort_values("Price", ascending=False, na_position="last")
    return out


def _fmt_card_number(value: object, fmt: str, fallback: str = "-") -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return fallback
    return format(float(numeric), fmt)


def _asset_initials(name: object) -> str:
    words = [part for part in str(name or "").replace("-", " ").split() if part]
    if not words:
        return "?"
    if len(words) == 1:
        return html.escape(words[0][:3].upper())
    return html.escape((words[0][0] + words[-1][0]).upper())


def fantasy_asset_card_html(
    asset: dict | pd.Series,
    boosted_token: str | None = None,
    asset_label: str = "Asset",
) -> str:
    row = asset if isinstance(asset, pd.Series) else pd.Series(asset)
    name = str(row.get("name", ""))
    team_name = str(row.get("team", row.get("name", "")))
    colour = html.escape(team_colour(team_name))
    safe_name = html.escape(name)
    safe_team = html.escape(team_name)
    badge = ""
    if boosted_token:
        badge = f'<span class="f1-boost">{html.escape(boosted_token)}</span>'
    return (
        '<div class="f1-driver-card" style="--team-color:{colour}">'
        "{badge}"
        '<div class="f1-card-top">'
        "<div>"
        '<div class="f1-card-name">{name}</div>'
        '<div class="f1-card-team">{team}</div>'
        "</div>"
        '<div class="f1-initials">{initials}</div>'
        "</div>"
        '<div class="f1-card-stats">'
        '<div class="f1-stat"><div class="f1-stat-label">Price</div><div class="f1-stat-value">{price}</div></div>'
        '<div class="f1-stat"><div class="f1-stat-label">Exp Pts</div><div class="f1-stat-value">{expected}</div></div>'
        '<div class="f1-stat"><div class="f1-stat-label">Exp Gain</div><div class="f1-stat-value">{gain}</div></div>'
        "</div>"
        "</div>".format(
            colour=colour,
            badge=badge,
            name=safe_name,
            team=safe_team,
            initials=_asset_initials(name),
            price=f"{_fmt_card_number(row.get('price'), '.2f')}M",
            expected=_fmt_card_number(row.get("display_exp_score", row.get("exp_score")), ".2f"),
            gain=f"{_fmt_card_number(row.get('expected_price_gain', row.get('expected_price_change')), '+.2f')}M",
        )
    )


def fantasy_card_grid_html(
    df: pd.DataFrame,
    boosted_driver: str | None = None,
    triple_driver: str | None = None,
    asset_label: str = "Driver",
) -> str:
    """Return escaped, unindented HTML for fantasy summary cards."""
    if df.empty:
        return ""

    boosted = str(boosted_driver or "").strip().lower()
    tripled = str(triple_driver or "").strip().lower()
    cards: list[str] = []
    for _, row in df.sort_values("price", ascending=False).iterrows():
        boost_badge = ""
        name = str(row.get("name", ""))
        if tripled and tripled == name.lower():
            boost_badge = "3x"
        elif boosted and boosted == name.lower():
            boost_badge = "2x"
        cards.append(
            fantasy_asset_card_html(
                row,
                boosted_token=boost_badge or None,
                asset_label=asset_label,
            )
        )
    return '<div class="f1-card-grid">' + "".join(cards) + "</div>"
