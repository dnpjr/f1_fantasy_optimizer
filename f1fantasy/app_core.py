from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
import difflib
import hashlib
import html
import json
import math
import re
import unicodedata
from typing import Any, Callable, Mapping

import pandas as pd

from f1fantasy.asset_identity import asset_ledger_diagnostics, build_player_identity_map
from f1fantasy.ergast import fetch_all_supporting, fetch_schedule
from f1fantasy.fantasy_api import (
    _latest_feed_round,
    clear_market_cache,
    fetch_market_asset_ledgers,
    fetch_validated_current_gameday_market,
    holding_valid_constructor_assets,
    holding_valid_player_assets,
    market_content_signature,
    price_view_constructor_assets,
    price_view_player_assets,
    resolve_market_data,
)
from f1fantasy.historical_scores import (
    DATA_VERSION as HISTORICAL_FANTASY_DATA_VERSION,
    DEFAULT_CANONICAL_DATASET_PATH,
    EARLIEST_PRODUCTION_SEASON,
    apply_recorded_scores_to_model,
    canonical_market_snapshot,
    canonical_playerstats_observations,
    load_canonical_scores,
    normalise_official_playerstats,
    resolve_score_precedence,
)
from f1fantasy.live_sessions import (
    LiveSessionIngestion,
    clear_live_session_cache,
    current_human_driver_field,
    empty_session_results,
    expected_live_session_kinds,
    ingest_active_event_sessions,
    live_session_signature,
)
from f1fantasy.live_session_shadow import (
    apply_live_session_emphasis,
    build_live_session_shadow,
    validate_live_session_emphasis,
)
from f1fantasy.model import (
    _horizon_weights,
    _constructor_round_points,
    apply_no_negative_expectation,
    compute_weekend_points,
    expected_scores_horizon,
    normalise_sprint_baseline_inputs,
    expected_scores_horizon_by_component,
)
from f1fantasy.optimize import TeamSolution, optimize_top_k
from f1fantasy.player_stats import (
    PLAYERSTATS_ENDPOINT_PATTERN,
    clear_playerstats_cache,
    fetch_team_lock_deadline_from_playerstats,
    fetch_recent_points_for_roster,
    latest_two_races,
)
from f1fantasy.price_efficiency import build_price_efficiency_table
from f1fantasy.race_selection import (
    RaceKey,
    RaceSelection,
    available_races,
    recency_weights,
    resolve_selected_races,
    weighted_asset_points,
)
from f1fantasy.sprint_shadow import (
    active_sprint_calibration_version,
    apply_sprint_production_adjustment,
    calculate_sprint_production_adjustment,
    calculate_sprint_shadow,
)
from f1fantasy.ui_helpers import (
    compact_asset_identity_html,
    compact_asset_payload,
    team_summary_html,
    team_summary_payload,
)
from f1fantasy.weekend_state import (
    EventKey,
    SessionKind,
    SessionState,
    SessionStatus,
    UpcomingEvent,
    WeekendFormat,
    coerce_utc_datetime,
    select_active_event,
    select_forecast_event,
    upcoming_circuit_names,
    upcoming_event_records,
    validate_deadline_candidate,
    validate_weekend_snapshot,
    weekend_states,
    weekend_format,
)


DEFAULT_HISTORICAL_SEASONS_BACK = 3
DEFAULT_UPCOMING_RACE_HORIZON = 5
DEFAULT_TOP_K = 1
HISTORY_MODE_ALL_SUPPORTED = "all_supported"
HISTORY_MODE_CURRENT_SEASON_ONLY = "current_season_only"
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
    driver_price_efficiency: pd.DataFrame = field(default_factory=pd.DataFrame)
    constructor_price_efficiency: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class LiveDataSnapshot:
    current_season: int
    loaded_start_year: int
    requested_seasons: tuple[int, ...]
    loaded_seasons: tuple[int, ...]
    season_load_failures: dict[int, str]
    results: pd.DataFrame
    qualifying: pd.DataFrame
    sprint: pd.DataFrame
    schedule: pd.DataFrame
    players: pd.DataFrame
    teams: pd.DataFrame
    driver_recent_points: pd.DataFrame
    constructor_recent_points: pd.DataFrame
    driver_race_points: pd.DataFrame
    constructor_race_points: pd.DataFrame
    team_lock_payload: dict
    source_diagnostics: dict
    historical_fantasy_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_assets: pd.DataFrame = field(default_factory=pd.DataFrame)
    constructor_assets: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_identity_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    session_results: pd.DataFrame = field(default_factory=empty_session_results)
    session_states: tuple[SessionState, ...] = ()


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
    columns = ["id", "name", "price"] + (["holding_status"] if "holding_status" in df.columns else [])
    for row in df[columns].itertuples(index=False):
        status = ""
        if hasattr(row, "holding_status") and str(row.holding_status).casefold() == "inactive":
            status = " · Inactive"
        labels[str(row.id)] = f"{row.name} ({format_money(row.price)}){status}"
    return labels


def current_team_option_labels(
    holdings: pd.DataFrame,
    selected_ids: list[str] | tuple[str, ...],
    asset_label: str,
) -> dict[str, str]:
    """Keep unresolved exact IDs representable so validation can report them."""
    labels = build_asset_option_labels(holdings)
    for asset_id in (str(value) for value in selected_ids):
        labels.setdefault(asset_id, f"Unknown {asset_label} ({asset_id})")
    return labels


def build_holding_asset_universe(
    selectable_model_assets: pd.DataFrame,
    official_asset_ledger: pd.DataFrame,
    asset_type: str,
) -> pd.DataFrame:
    """Enrich exact official holdings with model fields without changing identity."""
    selectable = selectable_model_assets.copy(deep=True)
    ledger = (
        holding_valid_player_assets(official_asset_ledger)
        if str(asset_type).casefold() == "driver"
        else holding_valid_constructor_assets(official_asset_ledger)
    )
    if ledger.empty:
        out = selectable.copy(deep=True)
        out["is_active"] = 1
        out["selectable"] = True
        out["holding_status"] = "Active"
        return out.reset_index(drop=True)

    source_id = "playerId" if str(asset_type).casefold() == "driver" else "teamId"
    if source_id not in ledger.columns:
        if "id" not in ledger.columns:
            raise ValueError(f"Official {asset_type} ledger has no asset ID column.")
        ledger[source_id] = ledger["id"].copy()
    ledger["id"] = ledger[source_id].astype(str)
    if "name" not in ledger.columns:
        raise ValueError(f"Official {asset_type} ledger has no asset name column.")
    if str(asset_type).casefold() == "constructor" and "team" not in ledger.columns:
        ledger["team"] = ledger["name"]

    active_values = pd.to_numeric(
        ledger.get("is_active", ledger.get("IsActive", pd.Series(1, index=ledger.index))),
        errors="coerce",
    ).fillna(0).astype(int)
    ledger["is_active"] = active_values
    ledger["selectable"] = active_values.eq(1)
    ledger["holding_status"] = active_values.map({1: "Active"}).fillna("Inactive")
    ledger["asset_type"] = str(asset_type).casefold()

    if selectable.empty:
        return ledger.reset_index(drop=True)
    selectable["id"] = selectable["id"].astype(str)
    authoritative = {
        "id", source_id, "PlayerId", "name", "team", "price", "previous_price",
        "official_price_change", "is_active", "IsActive", "selectable", "holding_status",
        "asset_type", "team_id", "TeamId", "driver_reference", "DriverReference",
        "tla", "DriverTLA", "f1_player_id", "F1PlayerId", "status", "Status",
    }
    model_columns = [
        column for column in selectable.columns
        if column == "id" or column not in authoritative
    ]
    enriched = ledger.merge(
        selectable[model_columns],
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_model"),
    )
    return enriched.reset_index(drop=True)


def completed_asset_price_history(
    race_observations: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return the latest two completed scores keyed only by Fantasy asset ID."""
    columns = [
        "id",
        "recent_points_2ago",
        "recent_points_1ago",
        "recent_points_available",
        "recent_points_source",
    ]
    if race_observations is None or race_observations.empty:
        return pd.DataFrame(columns=columns)
    required = {"PlayerId", "fantasy_points"}
    if not required.issubset(race_observations.columns):
        return pd.DataFrame(columns=columns)

    data = race_observations.copy(deep=True)
    data["fantasy_points"] = pd.to_numeric(data["fantasy_points"], errors="coerce")
    if "is_played" in data.columns:
        played = pd.to_numeric(data["is_played"], errors="coerce").fillna(0).eq(1)
        data = data.loc[played].copy()
    data = data[data["fantasy_points"].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["id"] = data["PlayerId"].astype(str)
    sort_columns = [
        column for column in ("season", "round", "gameday_id") if column in data.columns
    ]
    if sort_columns:
        data = data.sort_values(sort_columns, kind="stable", na_position="last")

    rows: list[dict[str, Any]] = []
    for asset_id, group in data.groupby("id", sort=False):
        points = group["fantasy_points"].tail(2).astype(float).tolist()
        rows.append(
            {
                "id": str(asset_id),
                "recent_points_2ago": points[-2] if len(points) >= 2 else pd.NA,
                "recent_points_1ago": points[-1] if points else pd.NA,
                "recent_points_available": len(points),
                "recent_points_source": "asset_specific_completed",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_price_change_asset_universe(
    selectable_model_assets: pd.DataFrame,
    official_asset_ledger: pd.DataFrame,
    asset_type: str,
    race_observations: pd.DataFrame | None = None,
    player_identity_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a priced official universe without changing optimiser eligibility."""
    normalized_type = str(asset_type).casefold()
    priced_ledger = (
        price_view_player_assets(official_asset_ledger)
        if normalized_type == "driver"
        else price_view_constructor_assets(official_asset_ledger)
    )
    universe = build_holding_asset_universe(
        selectable_model_assets,
        priced_ledger,
        normalized_type,
    )

    exact_history = completed_asset_price_history(race_observations)
    if not exact_history.empty:
        universe = universe.merge(
            exact_history,
            on="id",
            how="left",
            suffixes=("", "_asset_history"),
            validate="one_to_one",
        )
        for column in (
            "recent_points_2ago",
            "recent_points_1ago",
            "recent_points_available",
            "recent_points_source",
        ):
            history_column = f"{column}_asset_history"
            if history_column not in universe.columns:
                continue
            if column in universe.columns:
                universe[column] = universe[column].combine_first(universe[history_column])
            else:
                universe[column] = universe[history_column]
            universe.drop(columns=[history_column], inplace=True)

    if normalized_type == "driver" and isinstance(player_identity_map, pd.DataFrame):
        if not player_identity_map.empty and {
            "fantasy_asset_id",
            "human_driver_id",
        }.issubset(player_identity_map.columns):
            identity = player_identity_map[
                ["fantasy_asset_id", "human_driver_id"]
            ].copy(deep=True)
            identity["id"] = identity.pop("fantasy_asset_id").astype(str)
            universe = universe.merge(
                identity,
                on="id",
                how="left",
                suffixes=("", "_identity"),
                validate="one_to_one",
            )
            if "human_driver_id_identity" in universe.columns:
                if "human_driver_id" in universe.columns:
                    universe["human_driver_id"] = universe[
                        "human_driver_id"
                    ].combine_first(universe["human_driver_id_identity"])
                else:
                    universe["human_driver_id"] = universe["human_driver_id_identity"]
                universe.drop(columns=["human_driver_id_identity"], inplace=True)

    return _fill_recent_point_columns(universe).reset_index(drop=True)


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


def _emit_load_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
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


def copy_live_data_snapshot(snapshot: LiveDataSnapshot) -> LiveDataSnapshot:
    return LiveDataSnapshot(
        current_season=int(snapshot.current_season),
        loaded_start_year=int(snapshot.loaded_start_year),
        requested_seasons=tuple(int(year) for year in snapshot.requested_seasons),
        loaded_seasons=tuple(int(year) for year in snapshot.loaded_seasons),
        season_load_failures=deepcopy(snapshot.season_load_failures),
        results=snapshot.results.copy(deep=True),
        qualifying=snapshot.qualifying.copy(deep=True),
        sprint=snapshot.sprint.copy(deep=True),
        schedule=snapshot.schedule.copy(deep=True),
        players=snapshot.players.copy(deep=True),
        teams=snapshot.teams.copy(deep=True),
        driver_recent_points=snapshot.driver_recent_points.copy(deep=True),
        constructor_recent_points=snapshot.constructor_recent_points.copy(deep=True),
        driver_race_points=snapshot.driver_race_points.copy(deep=True),
        constructor_race_points=snapshot.constructor_race_points.copy(deep=True),
        team_lock_payload=deepcopy(snapshot.team_lock_payload),
        source_diagnostics=deepcopy(snapshot.source_diagnostics),
        historical_fantasy_scores=snapshot.historical_fantasy_scores.copy(deep=True),
        player_assets=snapshot.player_assets.copy(deep=True),
        constructor_assets=snapshot.constructor_assets.copy(deep=True),
        player_identity_map=snapshot.player_identity_map.copy(deep=True),
        session_results=snapshot.session_results.copy(deep=True),
        session_states=tuple(deepcopy(snapshot.session_states)),
    )


def copy_model_data(data: ModelData) -> ModelData:
    return ModelData(
        drivers=data.drivers.copy(deep=True),
        constructors=data.constructors.copy(deep=True),
        trends=data.trends.copy(deep=True),
        diagnostics=deepcopy(data.diagnostics),
        driver_price_efficiency=data.driver_price_efficiency.copy(deep=True),
        constructor_price_efficiency=data.constructor_price_efficiency.copy(deep=True),
    )


def effective_current_race_points(
    snapshot: LiveDataSnapshot,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge live observations with canonical current-season recorded totals.

    Current market rows, supporting classifications and the schedule do not
    define score-history coverage.  Canonical recorded observations fill that
    role and win duplicate keys; newly fetched playerstats rows remain available
    for rounds not yet present in the generated canonical snapshot.
    """

    def merge(entity_type: str, live: pd.DataFrame) -> pd.DataFrame:
        canonical = canonical_playerstats_observations(
            snapshot.historical_fantasy_scores,
            int(snapshot.current_season),
            entity_type,
        )
        live_copy = live.copy(deep=True) if live is not None else pd.DataFrame()
        if live_copy.empty:
            return canonical
        if canonical.empty:
            return live_copy
        combined = pd.concat([live_copy, canonical], ignore_index=True, sort=False)
        combined["_player_key"] = combined["PlayerId"].astype("string")
        combined["_season_key"] = pd.to_numeric(combined["season"], errors="coerce")
        combined["_round_key"] = pd.to_numeric(combined["round"], errors="coerce")
        combined = combined.drop_duplicates(
            ["_player_key", "_season_key", "_round_key"], keep="last"
        )
        return combined.drop(
            columns=["_player_key", "_season_key", "_round_key"]
        ).sort_values(["season", "round", "PlayerId"], kind="stable").reset_index(drop=True)

    return (
        merge("driver", snapshot.driver_race_points),
        merge("constructor", snapshot.constructor_race_points),
    )


def invalidate_live_data_caches() -> None:
    clear_market_cache()
    clear_playerstats_cache()
    clear_live_session_cache()


def _retain_last_good_session_results(
    current_snapshot: LiveDataSnapshot,
    loaded_snapshot: LiveDataSnapshot,
) -> LiveDataSnapshot:
    """Retain prior raw sessions across event rollover and source failure."""
    loaded_keys = {
        (state.event.season, state.event.round, state.kind.value)
        for state in loaded_snapshot.session_states
    }
    failed_keys = {
        (state.event.season, state.event.round, state.kind.value)
        for state in loaded_snapshot.session_states
        if state.status == SessionStatus.FAILED
    }
    historical_keys = {
        (state.event.season, state.event.round, state.kind.value)
        for state in current_snapshot.session_states
        if (state.event.season, state.event.round, state.kind.value) not in loaded_keys
    }
    retain_keys = failed_keys | historical_keys
    if not retain_keys or current_snapshot.session_results.empty:
        return loaded_snapshot
    previous = current_snapshot.session_results.copy(deep=True)
    if not {"season", "round", "session_kind"}.issubset(previous.columns):
        return loaded_snapshot
    retain_mask = [
        (int(season), int(round_no), str(kind)) in retain_keys
        if pd.notna(season) and pd.notna(round_no)
        else False
        for season, round_no, kind in zip(
            pd.to_numeric(previous["season"], errors="coerce"),
            pd.to_numeric(previous["round"], errors="coerce"),
            previous["session_kind"],
        )
    ]
    retained = previous.loc[retain_mask].copy(deep=True)
    if retained.empty:
        return loaded_snapshot
    fresh = loaded_snapshot.session_results.copy(deep=True)
    loaded_snapshot.session_results = pd.concat(
        [fresh, retained], ignore_index=True, sort=False
    ).drop_duplicates(
        ["season", "round", "session_kind", "source_driver_id"], keep="first"
    ).reset_index(drop=True)
    retained_states = tuple(
        deepcopy(state)
        for state in current_snapshot.session_states
        if (state.event.season, state.event.round, state.kind.value) in historical_keys
    )
    loaded_snapshot.session_states = tuple(loaded_snapshot.session_states) + retained_states
    loaded_snapshot.source_diagnostics["live_session_last_good_retained"] = True
    loaded_snapshot.source_diagnostics["live_session_last_good_retained_rows"] = int(len(retained))
    loaded_snapshot.source_diagnostics["live_session_snapshot_signature"] = live_session_signature(
        loaded_snapshot.session_results,
        loaded_snapshot.session_states,
    )
    return loaded_snapshot


def _dataframe_content_signature(frame: pd.DataFrame | None) -> str:
    """Return a deterministic identity without depending on row or column order."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return hashlib.sha256(b"[]").hexdigest()
    data = frame.copy(deep=True).reindex(sorted(frame.columns), axis=1)
    records = data.astype(object).where(data.notna(), None).to_dict("records")
    records.sort(
        key=lambda row: json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
    )
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scoring_content_signature(snapshot: LiveDataSnapshot) -> str:
    """Return the stable identity of every raw input used as scoring history."""
    parts = {
        name: _dataframe_content_signature(getattr(snapshot, name))
        for name in (
            "results",
            "qualifying",
            "sprint",
            "schedule",
            "driver_recent_points",
            "constructor_recent_points",
            "driver_race_points",
            "constructor_race_points",
            "historical_fantasy_scores",
        )
    }
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _retain_last_good_scoring_domain(
    current_snapshot: LiveDataSnapshot,
    loaded_snapshot: LiveDataSnapshot,
    *,
    reason: str,
) -> LiveDataSnapshot:
    """Accept refreshed non-scoring domains while retaining verified scoring inputs."""
    retained = copy_live_data_snapshot(loaded_snapshot)
    for field_name in (
        "results",
        "qualifying",
        "sprint",
        "driver_recent_points",
        "constructor_recent_points",
        "driver_race_points",
        "constructor_race_points",
        "historical_fantasy_scores",
    ):
        setattr(retained, field_name, getattr(current_snapshot, field_name).copy(deep=True))
    retained.player_identity_map = build_player_identity_map(
        retained.player_assets,
        retained.results,
    )
    diagnostics = retained.source_diagnostics
    rejected_validation = deepcopy(diagnostics.get("weekend_state_validation"))
    previous_diagnostics = current_snapshot.source_diagnostics
    diagnostics["scoring_data_status"] = "retained_last_good"
    diagnostics["playerstats_data_status"] = "retained_last_good"
    diagnostics["scoring_refresh_error"] = str(reason)
    diagnostics["market_scoring_freshness_mismatch"] = True
    diagnostics["scoring_rejected_weekend_state_validation"] = rejected_validation
    diagnostics["scoring_content_signature"] = previous_diagnostics.get(
        "scoring_content_signature"
    ) or scoring_content_signature(current_snapshot)
    diagnostics["scoring_verified_at_utc"] = previous_diagnostics.get(
        "scoring_verified_at_utc"
    ) or previous_diagnostics.get("raw_live_load_finished_utc")
    for key in (
        "snapshot_validation_status",
        "snapshot_validation_safe_for_scoring",
        "weekend_state_validation",
        "completed_current_event_keys",
        "excluded_partial_current_event_keys",
    ):
        if key in previous_diagnostics:
            diagnostics[key] = deepcopy(previous_diagnostics[key])
    return retained


def resolve_live_data_snapshot(
    current_snapshot: LiveDataSnapshot | None,
    refresh_requested: bool,
    loader: Callable[[bool], LiveDataSnapshot],
) -> dict:
    """Resolve a raw snapshot for one rerun without retrying ordinary UI changes."""
    if current_snapshot is not None and not refresh_requested:
        return {
            "snapshot": copy_live_data_snapshot(current_snapshot),
            "source_load_attempted": False,
            "source_load_succeeded": True,
            "result_accepted": True,
            "status": "reused",
            "error": None,
            "live_diagnostics": None,
        }
    try:
        loaded = loader(bool(refresh_requested))
    except Exception as exc:
        return {
            "snapshot": copy_live_data_snapshot(current_snapshot) if current_snapshot is not None else None,
            "source_load_attempted": True,
            "source_load_succeeded": False,
            "result_accepted": False,
            "status": "refresh_failed" if current_snapshot is not None else "initial_load_failed",
            "error": str(exc),
            "live_diagnostics": None,
        }
    if current_snapshot is not None:
        loaded = _retain_last_good_session_results(current_snapshot, loaded)
    validation_status = str(
        loaded.source_diagnostics.get("snapshot_validation_status", "valid")
    )
    live_data_status = str(
        loaded.source_diagnostics.get("live_data_status", "fresh")
    )
    live_diagnostics = deepcopy(loaded.source_diagnostics.get("weekend_state_validation"))
    if validation_status == "unsafe_partial" and current_snapshot is not None:
        warnings = loaded.source_diagnostics.get("snapshot_validation_warnings", [])
        reason = "; ".join(str(item) for item in warnings) or (
            "Current-session data is incomplete; previous scoring data remains in use."
        )
        retained = _retain_last_good_scoring_domain(
            current_snapshot,
            loaded,
            reason=reason,
        )
        retained_market_status = str(
            retained.source_diagnostics.get("live_data_status", "fresh")
        )
        return {
            "snapshot": copy_live_data_snapshot(retained),
            "source_load_attempted": True,
            "source_load_succeeded": True,
            "result_accepted": True,
            "status": (
                "market_refreshed_scoring_retained"
                if retained_market_status == "fresh"
                else "cached_market_scoring_retained"
            ),
            "error": None,
            "live_diagnostics": live_diagnostics,
        }
    if validation_status == "unsafe_partial":
        resolution_status = "loaded_with_partial_sessions"
    elif loaded.source_diagnostics.get("scoring_data_status") == "retained_last_good":
        resolution_status = (
            "market_refreshed_scoring_retained"
            if live_data_status == "fresh"
            else "cached_market_scoring_retained"
        )
    elif live_data_status in {"cached", "generated_snapshot"}:
        prefix = "refreshed" if refresh_requested else "loaded"
        resolution_status = f"{prefix}_{live_data_status}"
    else:
        resolution_status = "refreshed" if refresh_requested else "loaded"
    return {
        "snapshot": copy_live_data_snapshot(loaded),
        "source_load_attempted": True,
        "source_load_succeeded": True,
        "result_accepted": True,
        "status": resolution_status,
        "error": None,
        "live_diagnostics": live_diagnostics,
    }


def refresh_status_transition(
    current_status: str | None,
    current_error: str | None,
    current_successful_identity: tuple | None,
    *,
    refresh_requested: bool,
    source_load_attempted: bool,
    source_load_succeeded: bool,
    result_accepted: bool,
    error: str | None,
    successful_identity: tuple | None,
) -> dict:
    """Persist explicit refresh feedback across ordinary snapshot reuse."""
    if source_load_attempted and source_load_succeeded and result_accepted:
        return {
            "status": "succeeded" if refresh_requested or current_status == "failed" else current_status,
            "error": None,
            "successful_identity": successful_identity,
        }
    if refresh_requested and source_load_attempted:
        return {
            "status": "failed",
            "error": str(error or "Live refresh failed; previous successful data remains in use."),
            "successful_identity": current_successful_identity,
        }
    return {
        "status": current_status,
        "error": current_error,
        "successful_identity": current_successful_identity,
    }


def snapshot_race_catalogue(snapshot: LiveDataSnapshot) -> tuple[tuple, str]:
    """Return the current-season race catalogue without mutating or fetching.

    Canonical recorded driver/constructor totals are authoritative. Fresh raw
    playerstats and proxy result rows remain subject to the active-weekend
    completion gate when no recorded observation covers their event.
    """
    driver_points, constructor_points = effective_current_race_points(snapshot)
    race_frames = [
        frame
        for frame in (driver_points, constructor_points)
        if frame is not None and not frame.empty
    ]
    official = pd.concat(race_frames, ignore_index=True) if race_frames else pd.DataFrame()
    if not official.empty:
        origins = official.get(
            "fantasy_score_origin", pd.Series(index=official.index, dtype=object)
        ).fillna("").astype(str)
        recorded = official[
            origins.isin({"official_recorded", "third_party_recorded"})
        ].copy()
        raw = official.loc[~official.index.isin(recorded.index)].copy()
        completed_keys_raw = snapshot.source_diagnostics.get("completed_current_event_keys")
        if completed_keys_raw is not None and not raw.empty:
            completed_keys = {
                (int(season), int(round_no)) for season, round_no in completed_keys_raw
            }
            raw_keys = zip(
                pd.to_numeric(raw["season"], errors="coerce"),
                pd.to_numeric(raw["round"], errors="coerce"),
            )
            raw = raw[
                [
                    (int(season), int(round_no)) in completed_keys
                    if pd.notna(season) and pd.notna(round_no)
                    else False
                    for season, round_no in raw_keys
                ]
            ].copy()
        trusted = pd.concat([recorded, raw], ignore_index=True, sort=False)
        options = available_races(trusted, season=int(snapshot.current_season))
        if options:
            source = (
                "canonical_recorded_playerstats_union"
                if not recorded.empty
                else "official_playerstats_union"
            )
            return options, source

    options: tuple = ()
    results = snapshot.results.copy(deep=True)
    if not results.empty and {"season", "round"}.issubset(results.columns):
        completed_keys_raw = snapshot.source_diagnostics.get("completed_current_event_keys")
        if completed_keys_raw is None:
            inferred = validate_weekend_snapshot(
                snapshot.schedule,
                results=snapshot.results,
                qualifying=snapshot.qualifying,
                sprint=snapshot.sprint,
                effective_time=snapshot.source_diagnostics.get("raw_live_load_finished_utc"),
                expected_participant_count=len(snapshot.players) or None,
            )
            if inferred.active_weekend is not None:
                completed_keys_raw = [
                    (key.season, key.round) for key in inferred.completed_event_keys
                ]
        if completed_keys_raw is not None:
            completed_keys = {
                (int(season), int(round_no)) for season, round_no in completed_keys_raw
            }
            row_keys = list(zip(
                pd.to_numeric(results["season"], errors="coerce"),
                pd.to_numeric(results["round"], errors="coerce"),
            ))
            results = results[
                [
                    (int(season), int(round_no)) in completed_keys
                    if pd.notna(season) and pd.notna(round_no)
                    else False
                    for season, round_no in row_keys
                ]
            ].copy()
        proxy = results.copy(deep=True)
        proxy["race_name"] = proxy.get("circuitName", pd.Series(index=proxy.index, dtype=object))
        proxy["fantasy_points"] = 0.0
        proxy["is_played"] = 1
        options = available_races(proxy, season=int(snapshot.current_season))
    return options, "current_proxy_results_fallback" if options else "unavailable"


def current_season_round_lineage(
    snapshot: LiveDataSnapshot,
    selected_race_keys: tuple[RaceKey, ...] | list[RaceKey] | None = None,
) -> pd.DataFrame:
    """Describe how each canonical current-season round reaches consumers."""
    columns = [
        "season",
        "round",
        "event_name",
        "canonical_driver_rows",
        "canonical_constructor_rows",
        "recorded_driver_totals",
        "recorded_constructor_totals",
        "loaded_into_history",
        "loaded_into_model",
        "available_to_race_selector",
        "selected_by_all_completed",
        "used_by_price_efficiency",
        "used_by_projection_model",
        "exclusion_reason",
    ]
    recorded = snapshot.historical_fantasy_scores.copy(deep=True)
    if recorded.empty:
        return pd.DataFrame(columns=columns)
    current = recorded[
        pd.to_numeric(recorded["season"], errors="coerce").eq(int(snapshot.current_season))
    ].copy()
    current = current[
        current.get(
            "fantasy_score_origin", pd.Series(index=current.index, dtype=object)
        ).isin({"official_recorded", "third_party_recorded"})
    ].copy()
    if current.empty:
        return pd.DataFrame(columns=columns)

    catalogue, _source = snapshot_race_catalogue(snapshot)
    catalogue_keys = {option.key for option in catalogue}
    selected_keys = set(selected_race_keys or tuple(catalogue_keys))
    driver_points, constructor_points = effective_current_race_points(snapshot)

    def observation_keys(frame: pd.DataFrame) -> set[RaceKey]:
        if frame.empty:
            return set()
        valid = frame[
            pd.to_numeric(
                frame.get(
                    "fantasy_points", pd.Series(index=frame.index, dtype=float)
                ),
                errors="coerce",
            ).notna()
        ]
        return {
            RaceKey(int(season), int(round_no))
            for season, round_no in zip(valid["season"], valid["round"])
            if pd.notna(season) and pd.notna(round_no)
        }

    efficiency_keys = observation_keys(driver_points) | observation_keys(constructor_points)
    rows: list[dict[str, Any]] = []
    for round_no, group in current.groupby("round", sort=True):
        key = RaceKey(int(snapshot.current_season), int(round_no))
        drivers = group[group["entity_type"].eq("driver")]
        constructors = group[group["entity_type"].eq("constructor")]
        driver_totals = int(
            pd.to_numeric(drivers["fantasy_points_total"], errors="coerce").notna().sum()
        )
        constructor_totals = int(
            pd.to_numeric(constructors["fantasy_points_total"], errors="coerce").notna().sum()
        )
        if driver_totals == 0 and constructor_totals == 0:
            reason = "no recorded driver or constructor totals"
        elif driver_totals == 0:
            reason = "no recorded driver totals"
        elif constructor_totals == 0:
            reason = "no recorded constructor totals"
        elif key not in catalogue_keys:
            reason = "recorded totals did not reach race catalogue"
        else:
            reason = ""
        event_names = group.get("event_name", pd.Series(dtype=object)).dropna().astype(str)
        rows.append(
            {
                "season": key.season,
                "round": key.round,
                "event_name": event_names.iloc[0] if not event_names.empty else f"Round {key.round}",
                "canonical_driver_rows": int(len(drivers)),
                "canonical_constructor_rows": int(len(constructors)),
                "recorded_driver_totals": driver_totals,
                "recorded_constructor_totals": constructor_totals,
                "loaded_into_history": bool(driver_totals or constructor_totals),
                "loaded_into_model": bool(driver_totals or constructor_totals),
                "available_to_race_selector": key in catalogue_keys,
                "selected_by_all_completed": key in catalogue_keys,
                "used_by_price_efficiency": key in efficiency_keys and key in selected_keys,
                "used_by_projection_model": key in selected_keys,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def resolve_snapshot_race_selection(
    snapshot: LiveDataSnapshot,
    preset: str = "All",
    custom_race_keys: list[RaceKey | tuple[int, int]] | tuple[RaceKey | tuple[int, int], ...] | None = None,
    excluded_race_keys: list[RaceKey | tuple[int, int]] | tuple[RaceKey | tuple[int, int], ...] | None = None,
) -> tuple[RaceSelection, str]:
    catalogue, source = snapshot_race_catalogue(snapshot)
    selection = resolve_selected_races(
        catalogue,
        preset,
        custom_keys=custom_race_keys,
        excluded_keys=excluded_race_keys,
    )
    return selection, source


def sprint_calibration_prediction_identity(
    snapshot: LiveDataSnapshot,
    effective_date: str | None = None,
) -> tuple[str, str | None]:
    """Version only predictions whose next canonical event is a 2026 Sprint."""
    events = upcoming_event_records(
        snapshot.schedule,
        effective_time=effective_date,
        limit=1,
    )
    if (
        not events
        or events[0].season != 2026
        or events[0].format != WeekendFormat.SPRINT
    ):
        return ("sprint_calibration", None)
    try:
        version = active_sprint_calibration_version()
    except Exception:
        version = "unavailable"
    return ("sprint_calibration", version)


def model_settings_signature(
    snapshot: LiveDataSnapshot,
    historical_seasons_back: int,
    horizon_races: int,
    current_season_weight: float,
    past_season_weight: float,
    recency_decay: float,
    effective_date: str | None = None,
    selected_race_preset: str = "All",
    custom_race_keys: list[RaceKey | tuple[int, int]] | tuple[RaceKey | tuple[int, int], ...] | None = None,
    excluded_race_keys: list[RaceKey | tuple[int, int]] | tuple[RaceKey | tuple[int, int], ...] | None = None,
    history_mode: str = HISTORY_MODE_ALL_SUPPORTED,
    live_session_emphasis: float = 0.0,
) -> tuple:
    """Return the raw-snapshot/model-controls key used for derived data."""
    effective_date = effective_date or datetime.now(UTC).date().isoformat()
    selection, _catalogue_source = resolve_snapshot_race_selection(
        snapshot,
        preset=selected_race_preset,
        custom_race_keys=custom_race_keys,
        excluded_race_keys=excluded_race_keys,
    )
    live_session_emphasis = validate_live_session_emphasis(live_session_emphasis)
    return (
        live_data_snapshot_identity(snapshot),
        str(effective_date),
        int(historical_seasons_back),
        int(horizon_races),
        float(current_season_weight),
        float(past_season_weight),
        float(recency_decay),
        live_session_emphasis,
        str(history_mode),
        selection.preset,
        tuple((key.season, key.round) for key in selection.included),
        tuple((key.season, key.round) for key in selection.excluded),
        sprint_calibration_prediction_identity(snapshot, effective_date),
    )


def live_data_snapshot_identity(snapshot: LiveDataSnapshot) -> tuple:
    """Return the stable identity of one successfully acquired raw snapshot."""
    market_signature = snapshot.source_diagnostics.get("market_content_signature")
    if not market_signature:
        market_parts = {
            name: _dataframe_content_signature(getattr(snapshot, name))
            for name in ("players", "teams", "player_assets", "constructor_assets")
        }
        market_signature = hashlib.sha256(
            json.dumps(market_parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    scoring_signature = snapshot.source_diagnostics.get(
        "scoring_content_signature"
    ) or scoring_content_signature(snapshot)
    deadline_signature = hashlib.sha256(
        json.dumps(
            snapshot.team_lock_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    forecast_event = _event_key_from_diagnostics(
        snapshot.source_diagnostics.get("forecast_target_event")
    )
    return (
        int(snapshot.current_season),
        tuple(int(year) for year in snapshot.loaded_seasons),
        str(market_signature),
        str(scoring_signature),
        deadline_signature,
        str(snapshot.source_diagnostics.get("historical_fantasy_data_version", HISTORICAL_FANTASY_DATA_VERSION)),
        (forecast_event.season, forecast_event.round) if forecast_event else None,
        live_session_signature(snapshot.session_results, snapshot.session_states),
    )


def market_runtime_status(snapshot: LiveDataSnapshot) -> dict[str, Any]:
    """Describe only the final market domain presented to downstream consumers."""
    live_status = str(snapshot.source_diagnostics.get("live_data_status", "fresh"))
    is_current = live_status == "fresh"
    return {
        "state": "current" if is_current else "cached",
        "is_current": is_current,
        "show_stale_warning": not is_current,
        "feed_round": int(snapshot.source_diagnostics.get("feed_round", 0) or 0),
        "content_signature": str(
            snapshot.source_diagnostics.get("market_content_signature") or ""
        ),
        "verified_at_utc": snapshot.source_diagnostics.get("live_data_verified_at_utc"),
        "source": snapshot.source_diagnostics.get("market_resolution_method"),
    }


def model_data_version(snapshot: LiveDataSnapshot, requested_model_signature: tuple) -> tuple:
    """Version persisted outputs that depend on both raw and model inputs."""
    return (live_data_snapshot_identity(snapshot), tuple(requested_model_signature))


def build_transfer_result_signature(data_version: tuple, transfer_inputs: tuple) -> tuple:
    """Build the signature for recommendations persisted across reruns."""
    return (tuple(data_version), tuple(transfer_inputs))


def resolve_derived_model_data(
    snapshot: LiveDataSnapshot,
    current_data: ModelData | None,
    current_signature: tuple | None,
    requested_signature: tuple,
    deriver: Callable[[LiveDataSnapshot], ModelData],
    failed_signature: tuple | None = None,
    failed_error: str | None = None,
) -> dict:
    """Reuse or derive model data, suppressing a repeated known failure."""
    if current_data is not None and current_signature == requested_signature:
        return {
            "data": copy_model_data(current_data),
            "derivation_attempted": False,
            "recomputed": False,
            "status": "reused",
            "error": None,
            "failed_signature": None,
        }
    if current_data is not None and failed_signature == requested_signature:
        return {
            "data": copy_model_data(current_data),
            "derivation_attempted": False,
            "recomputed": False,
            "status": "suppressed_failed_signature",
            "error": str(failed_error or "Model derivation previously failed for these settings."),
            "failed_signature": failed_signature,
        }
    try:
        derived = deriver(copy_live_data_snapshot(snapshot))
    except Exception as exc:
        return {
            "data": copy_model_data(current_data) if current_data is not None else None,
            "derivation_attempted": True,
            "recomputed": False,
            "status": "failed",
            "error": str(exc),
            "failed_signature": requested_signature,
        }
    return {
        "data": copy_model_data(derived),
        "derivation_attempted": True,
        "recomputed": True,
        "status": "derived",
        "error": None,
        "failed_signature": None,
    }


def _event_key_from_diagnostics(value: Any) -> EventKey | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return EventKey(int(value["season"]), int(value["round"]))
    except (KeyError, TypeError, ValueError):
        return None


def _market_forecast_event_hint(
    schedule: pd.DataFrame,
    *,
    current_season: int,
    market_resolution: Mapping[str, Any],
) -> EventKey | None:
    """Map a verified Fantasy market/gameday identity onto the race schedule."""
    if schedule is None or schedule.empty or not {"season", "round"}.issubset(schedule.columns):
        return None
    rows = schedule[
        pd.to_numeric(schedule["season"], errors="coerce").eq(int(current_season))
    ].copy()
    if rows.empty:
        return None
    snapshot_name = str(
        market_resolution.get("snapshot_name")
        or market_resolution.get("live_data_snapshot_name")
        or ""
    ).strip()
    if snapshot_name and "raceName" in rows.columns:
        normalised_name = re.sub(r"[^a-z0-9]+", "", snapshot_name.casefold()).replace(
            "grandprix", ""
        )
        names = rows["raceName"].fillna("").astype(str).map(
            lambda value: re.sub(r"[^a-z0-9]+", "", value.casefold()).replace(
                "grandprix", ""
            )
        )
        matching = rows[names.eq(normalised_name)]
        if len(matching) == 1:
            return EventKey(int(matching.iloc[0]["season"]), int(matching.iloc[0]["round"]))
    feed_round = pd.to_numeric(market_resolution.get("feed_round"), errors="coerce")
    if pd.notna(feed_round):
        matching = rows[pd.to_numeric(rows["round"], errors="coerce").eq(int(feed_round))]
        if len(matching) == 1:
            return EventKey(int(current_season), int(feed_round))
    return None


def load_live_data_snapshot(
    current_season: int | None = None,
    historical_seasons_back: int = DEFAULT_HISTORICAL_SEASONS_BACK,
    include_playerstats: bool = True,
    force_refresh: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    effective_time: datetime | str | None = None,
    previous_snapshot: LiveDataSnapshot | None = None,
) -> LiveDataSnapshot:
    """Fetch one raw live-data snapshot; model settings are intentionally absent."""
    source_started = datetime.now(UTC)
    events: list[str] = []

    def _log(event: str) -> None:
        events.append(f"{datetime.now(UTC).isoformat()} - {event}")

    if force_refresh:
        invalidate_live_data_caches()
        _log("live_source_caches_invalidated")

    current_season = int(current_season or source_started.year)
    state_time = coerce_utc_datetime(effective_time or source_started)
    requested_start_year = max(
        EARLIEST_PRODUCTION_SEASON,
        current_season - int(historical_seasons_back),
    )
    requested_seasons = tuple(range(requested_start_year, current_season + 1))
    _emit_load_progress(progress_callback, 1, "Loading market feed", "Loading market feed...", progress=0.05)

    all_results: list[pd.DataFrame] = []
    all_quali: list[pd.DataFrame] = []
    all_sprint: list[pd.DataFrame] = []
    loaded_seasons: list[int] = []
    season_load_failures: dict[int, str] = {}
    scoring_fallback_reasons: list[str] = []
    schedule = pd.DataFrame()

    def previous_season_rows(frame: pd.DataFrame, year: int) -> pd.DataFrame:
        if frame.empty:
            return frame.copy(deep=True)
        if "season" not in frame.columns:
            return frame.copy(deep=True) if year in previous_snapshot.loaded_seasons else pd.DataFrame()
        seasons = pd.to_numeric(frame["season"], errors="coerce")
        return frame.loc[seasons.eq(int(year))].copy(deep=True)

    for year in requested_seasons:
        _emit_load_progress(
            progress_callback,
            3,
            "Loading supporting race/schedule data",
            f"Loading supporting race/schedule data ({year})...",
            progress=0.20,
        )
        _log(f"fetch_supporting_start season={year} force_refresh={force_refresh}")
        try:
            data = (
                fetch_all_supporting(year, force_refresh=True)
                if force_refresh
                else fetch_all_supporting(year)
            )
            results_frame = data["results"].copy(deep=True)
            qualifying_frame = data["qualifying"].copy(deep=True)
            sprint_frame = data["sprint"].copy(deep=True)
            schedule_frame = data.get("schedule", pd.DataFrame()).copy(deep=True)
        except Exception as exc:
            _log(f"fetch_supporting_failed season={year} error={exc}")
            season_load_failures[int(year)] = str(exc)
            if force_refresh and previous_snapshot is not None:
                previous_results = previous_season_rows(previous_snapshot.results, year)
                reason = f"{year}: {exc}"
                scoring_fallback_reasons.append(reason)
                if not previous_results.empty:
                    loaded_seasons.append(int(year))
                    all_results.append(previous_results)
                    all_quali.append(previous_season_rows(previous_snapshot.qualifying, year))
                    all_sprint.append(previous_season_rows(previous_snapshot.sprint, year))
                    if year == current_season:
                        schedule = previous_snapshot.schedule.copy(deep=True)
                    _log(f"fetch_supporting_retained_previous season={year}")
                else:
                    _log(f"fetch_supporting_no_previous_rows season={year}")
                continue
            if force_refresh:
                raise RuntimeError(f"Could not refresh supporting race data for {year}.") from exc
            continue
        _log(f"fetch_supporting_done season={year}")
        loaded_seasons.append(int(year))
        all_results.append(results_frame)
        all_quali.append(qualifying_frame)
        all_sprint.append(sprint_frame)
        if year == current_season:
            schedule = schedule_frame

    if (
        not all_results
        and previous_snapshot is not None
        and not previous_snapshot.results.empty
    ):
        all_results = [previous_snapshot.results.copy(deep=True)]
        all_quali = [previous_snapshot.qualifying.copy(deep=True)]
        all_sprint = [previous_snapshot.sprint.copy(deep=True)]
        loaded_seasons = [
            int(year)
            for year in previous_snapshot.loaded_seasons
            if int(year) in requested_seasons
        ]
        schedule = previous_snapshot.schedule.copy(deep=True)
        _log("supporting_scoring_domain_retained_whole")
    if not all_results:
        raise RuntimeError("Could not load race-result support data from the public endpoints.")
    if schedule.empty:
        _log("schedule_fetch_fallback_start")
        schedule = (
            fetch_schedule(current_season, force_refresh=True)
            if force_refresh
            else fetch_schedule(current_season)
        ).copy(deep=True)
        _log(f"schedule_fetch_fallback_done rows={len(schedule)}")

    results = pd.concat(all_results, ignore_index=True)
    qualifying = pd.concat(all_quali, ignore_index=True) if any(len(frame) for frame in all_quali) else pd.DataFrame()
    sprint = pd.concat(all_sprint, ignore_index=True) if any(len(frame) for frame in all_sprint) else pd.DataFrame()

    historical_fantasy_scores = load_canonical_scores(DEFAULT_CANONICAL_DATASET_PATH)
    try:
        historical_reference = canonical_market_snapshot(historical_fantasy_scores, current_season)
        _log(
            "historical_market_reference_ready "
            f"season={current_season} round={historical_reference['round']}"
        )
    except Exception as exc:
        historical_reference = {
            "players": pd.DataFrame(),
            "teams": pd.DataFrame(),
            "round": None,
            "event_name": None,
        }
        _log(f"historical_market_reference_unavailable error={exc}")

    market_event = select_active_event(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=state_time,
        expected_participant_count=None,
    )
    seed_asset_ids = [
        *pd.to_numeric(
            historical_reference["players"].get("playerId", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().astype(int).tolist(),
        *pd.to_numeric(
            historical_reference["teams"].get("teamId", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().astype(int).tolist(),
    ]

    _log("market_feed_round_detect_start")
    _emit_load_progress(progress_callback, 1, "Loading market feed", "Detecting latest market feed...", progress=0.08)
    market_resolution = resolve_market_data(
        latest_feed_loader=_latest_feed_round,
        current_gameday_loader=lambda: fetch_validated_current_gameday_market(
            seed_asset_ids,
            expected_event_name=market_event.race_name if market_event is not None else None,
            expected_season=current_season,
            market_loader=fetch_market_asset_ledgers,
        ),
        market_loader=fetch_market_asset_ledgers,
    )
    feed_round = market_resolution.get("feed_round")
    live_data_status = str(market_resolution["live_data_status"])
    players = market_resolution["players"].copy(deep=True)
    teams = market_resolution["teams"].copy(deep=True)
    player_assets = market_resolution.get("player_assets", players).copy(deep=True)
    constructor_assets = market_resolution.get("constructor_assets", teams).copy(deep=True)
    player_identity_map = build_player_identity_map(player_assets, results)
    ledger_diagnostics = asset_ledger_diagnostics(player_assets, player_identity_map)
    _log(
        "market_resolution_done "
        f"status={live_data_status} feed_round={feed_round} "
        f"snapshot_round={market_resolution.get('snapshot_round')} "
        f"drivers={len(players)}/{len(player_assets)} constructors={len(teams)}/{len(constructor_assets)}"
    )
    if market_resolution.get("refresh_error"):
        _log(f"market_refresh_failed error={market_resolution['refresh_error']}")
    _emit_load_progress(
        progress_callback,
        2,
        "Loading current prices",
        (
            "Loaded fresh verified official prices."
            if live_data_status == "fresh"
            else "Loaded the newest safe fallback prices."
        ),
        progress=0.16,
        status="complete" if live_data_status == "fresh" else "warning",
        details={"live_data_status": live_data_status},
    )

    weekend_validation = validate_weekend_snapshot(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=state_time,
        expected_participant_count=len(players) or None,
    )
    scoring_active_weekend = weekend_validation.active_weekend
    market_target_hint = _market_forecast_event_hint(
        schedule,
        current_season=current_season,
        market_resolution=market_resolution,
    )
    forecast_weekend = select_forecast_event(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=state_time,
        expected_participant_count=len(players) or None,
        verified_target_event=market_target_hint,
    )
    forecast_schedule_row = None
    if forecast_weekend is not None and {"season", "round"}.issubset(schedule.columns):
        active_rows = schedule[
            (pd.to_numeric(schedule["season"], errors="coerce") == forecast_weekend.event.season)
            & (pd.to_numeric(schedule["round"], errors="coerce") == forecast_weekend.event.round)
        ]
        if not active_rows.empty:
            forecast_schedule_row = active_rows.iloc[0]

    session_ingestion = LiveSessionIngestion(
        results=empty_session_results(),
        states=(),
        diagnostics={
            "active_event": None,
            "weekend_format": "unavailable",
            "source": "jolpica_alpha_results",
            "sessions": {},
            "rows_observed": 0,
        },
    )
    if forecast_weekend is not None and forecast_schedule_row is not None:
        _log(
            "live_session_fetch_start "
            f"season={forecast_weekend.event.season} round={forecast_weekend.event.round} "
            f"format={forecast_weekend.format.value}"
        )
        try:
            session_ingestion = ingest_active_event_sessions(
                forecast_schedule_row,
                format=forecast_weekend.format,
                history=results,
                player_identity_map=player_identity_map,
                effective_time=state_time,
                force_refresh=force_refresh,
            )
            _log(
                "live_session_fetch_done "
                f"rows={len(session_ingestion.results)} "
                "states="
                + ",".join(
                    f"{state.kind.value}:{state.status.value}"
                    for state in session_ingestion.states
                )
            )
        except Exception as exc:
            expected_humans = current_human_driver_field(player_identity_map)
            failed_states = tuple(
                SessionState(
                    event=forecast_weekend.event,
                    kind=kind,
                    scheduled_at=forecast_weekend.session(kind).scheduled_at,
                    observed_row_count=0,
                    expected_participant_count=len(expected_humans) or None,
                    status=SessionStatus.FAILED,
                    source="jolpica_alpha_results",
                    diagnostic=f"Unexpected session ingestion failure: {exc}",
                    supported=True,
                )
                for kind in expected_live_session_kinds(forecast_weekend.format)
            )
            session_ingestion = LiveSessionIngestion(
                results=empty_session_results(),
                states=failed_states,
                diagnostics={
                    "active_event": {
                        "season": forecast_weekend.event.season,
                        "round": forecast_weekend.event.round,
                    },
                    "weekend_format": forecast_weekend.format.value,
                    "source": "jolpica_alpha_results",
                    "source_error": str(exc),
                    "rows_observed": 0,
                    "sessions": {
                        state.kind.value: {
                            "session_kind": state.kind.value,
                            "status": state.status.value,
                            "rows_observed": 0,
                            "expected_participants": state.expected_participant_count,
                            "source": state.source,
                            "diagnostic": state.diagnostic,
                        }
                        for state in failed_states
                    },
                },
            )
            _log(f"live_session_fetch_failed error={exc}")

    team_lock_payload: dict = {}
    deadline_warning = None
    if not players.empty and "playerId" in players.columns:
        try:
            _log("team_lock_playerstats_probe_start")
            team_lock_payload = fetch_team_lock_deadline_from_playerstats(int(players.iloc[0]["playerId"]))
            _log("team_lock_playerstats_probe_done")
            if forecast_weekend is not None and forecast_schedule_row is not None:
                team_lock_payload = validate_deadline_candidate(
                    team_lock_payload,
                    active_event=forecast_weekend.event,
                    schedule_row=forecast_schedule_row,
                    format=forecast_weekend.format,
                )
                if not team_lock_payload.get("team_lock_deadline_valid", False):
                    deadline_warning = team_lock_payload.get("team_lock_validation_reason")
                    _log(f"team_lock_official_rejected reason={deadline_warning}")
        except Exception as exc:
            _log(f"team_lock_playerstats_probe_failed error={exc}")
            deadline_warning = f"Official fantasy deadline unavailable: {exc}"
            team_lock_payload = {
                "team_lock_deadline_utc": None,
                "team_lock_deadline_valid": False,
                "team_lock_validation_reason": deadline_warning,
                "team_lock_matched_event": (
                    (forecast_weekend.event.season, forecast_weekend.event.round)
                    if forecast_weekend is not None else None
                ),
            }

    empty_stats_diag = {
        "playerstats_assets_loaded": 0,
        "playerstats_assets_failed": 0,
        "playerstats_timeout_failures": 0,
        "playerstats_skipped_after_failure_limit": 0,
        "playerstats_failures": [],
    }
    driver_recent = pd.DataFrame()
    constructor_recent = pd.DataFrame()
    driver_race_points = pd.DataFrame()
    constructor_race_points = pd.DataFrame()
    driver_stats_diag = dict(empty_stats_diag)
    constructor_stats_diag = dict(empty_stats_diag)
    playerstats_started = datetime.now(UTC)
    if include_playerstats:
        def _driver_progress(payload: dict[str, Any]) -> None:
            processed = int(payload.get("processed", 0) or 0)
            total = max(int(payload.get("total", 0) or 0), 1)
            _emit_load_progress(
                progress_callback,
                4,
                "Loading playerstats",
                f"Loading playerstats {processed}/{total} (failed/timeouts: {int(payload.get('failed', 0) or 0)}).",
                progress=0.62 + 0.08 * min(1.0, processed / total),
            )

        def _constructor_progress(payload: dict[str, Any]) -> None:
            processed = int(payload.get("processed", 0) or 0)
            total = max(int(payload.get("total", 0) or 0), 1)
            _emit_load_progress(
                progress_callback,
                4,
                "Loading playerstats",
                f"Loading playerstats {processed}/{total} (failed/timeouts: {int(payload.get('failed', 0) or 0)}).",
                progress=0.72 + 0.06 * min(1.0, processed / total),
            )

        _emit_load_progress(progress_callback, 4, "Loading playerstats", "Loading playerstats for drivers...", progress=0.62)
        driver_roster = players.rename(columns={"playerId": "id"})
        _log("playerstats_fetch_start drivers")
        driver_with_recent, driver_race_points, driver_stats_diag = _add_playerstats_recent_points(
            driver_roster,
            "driver",
            progress_callback=_driver_progress,
        )
        driver_recent = driver_with_recent[
            [col for col in driver_with_recent.columns if col == "id" or col.startswith("recent_points_")]
        ].copy(deep=True)
        _log(
            "playerstats_fetch_done drivers "
            f"loaded={driver_stats_diag.get('playerstats_assets_loaded', 0)} "
            f"failed={driver_stats_diag.get('playerstats_assets_failed', 0)}"
        )
        _emit_load_progress(progress_callback, 4, "Loading playerstats", "Loading playerstats for constructors...", progress=0.72)
        constructor_roster = teams.rename(columns={"teamId": "id"})
        _log("playerstats_fetch_start constructors")
        constructor_with_recent, constructor_race_points, constructor_stats_diag = _add_playerstats_recent_points(
            constructor_roster,
            "constructor",
            progress_callback=_constructor_progress,
        )
        constructor_recent = constructor_with_recent[
            [col for col in constructor_with_recent.columns if col == "id" or col.startswith("recent_points_")]
        ].copy(deep=True)
        _log(
            "playerstats_fetch_done constructors "
            f"loaded={constructor_stats_diag.get('playerstats_assets_loaded', 0)} "
            f"failed={constructor_stats_diag.get('playerstats_assets_failed', 0)}"
        )
        playerstats_load_seconds = max(0.0, (datetime.now(UTC) - playerstats_started).total_seconds())
    else:
        _log("playerstats_prefetch_skipped")
        _emit_load_progress(
            progress_callback,
            4,
            "Loading playerstats",
            "Skipping detailed playerstats prefetch.",
            progress=0.78,
            status="warning",
        )
        playerstats_load_seconds = 0.0
    if scoring_fallback_reasons and previous_snapshot is not None:
        driver_recent = previous_snapshot.driver_recent_points.copy(deep=True)
        constructor_recent = previous_snapshot.constructor_recent_points.copy(deep=True)
        driver_race_points = previous_snapshot.driver_race_points.copy(deep=True)
        constructor_race_points = previous_snapshot.constructor_race_points.copy(deep=True)
        _log("playerstats_scoring_retained_previous")
    source_finished = datetime.now(UTC)
    source_diagnostics = {
        "feed_round": int(feed_round or 0),
        "live_data_status": live_data_status,
        "market_resolution_method": market_resolution.get("market_resolution_method"),
        "market_latest_probe_error": market_resolution.get("latest_probe_error"),
        "live_data_refresh_error": market_resolution.get("refresh_error"),
        "live_data_fallback_failures": list(market_resolution.get("fallback_failures", [])),
        "live_data_verified_at_utc": market_resolution.get("verified_at_utc"),
        "market_content_signature": market_resolution.get("content_signature"),
        "market_content_changed": bool(market_resolution.get("content_changed", False)),
        "market_data_status": "current" if live_data_status == "fresh" else "cached",
        "live_data_snapshot_round": market_resolution.get("snapshot_round"),
        "live_data_snapshot_name": market_resolution.get("snapshot_name"),
        "market_requested_event_name": market_resolution.get("requested_event_name"),
        "market_expected_event_advanced": bool(
            market_resolution.get("expected_event_advanced", False)
        ),
        "asset_ledger_complete": bool(market_resolution.get("asset_ledger_complete", False)),
        **ledger_diagnostics,
        "playerstats_prefetch_enabled": bool(include_playerstats),
        "driver_stats_diag": deepcopy(driver_stats_diag),
        "constructor_stats_diag": deepcopy(constructor_stats_diag),
        "raw_live_load_started_utc": source_started.isoformat(),
        "raw_live_load_finished_utc": source_finished.isoformat(),
        "raw_live_load_duration_seconds": max(0.0, (source_finished - source_started).total_seconds()),
        "playerstats_load_duration_seconds": float(playerstats_load_seconds),
        "raw_live_force_refresh": bool(force_refresh),
        "raw_requested_seasons": list(requested_seasons),
        "raw_loaded_seasons": list(loaded_seasons),
        "raw_season_load_failures": deepcopy(season_load_failures),
        "raw_live_events": events[-40:],
        "scoring_data_status": (
            "retained_last_good" if scoring_fallback_reasons else "current"
        ),
        "scoring_refresh_error": (
            "; ".join(scoring_fallback_reasons) if scoring_fallback_reasons else None
        ),
        "scoring_verified_at_utc": (
            previous_snapshot.source_diagnostics.get("scoring_verified_at_utc")
            or previous_snapshot.source_diagnostics.get("raw_live_load_finished_utc")
            if scoring_fallback_reasons and previous_snapshot is not None
            else source_finished.isoformat()
        ),
        "market_scoring_freshness_mismatch": bool(scoring_fallback_reasons),
        "playerstats_data_status": (
            "retained_last_good"
            if scoring_fallback_reasons and previous_snapshot is not None
            else "disabled"
            if not include_playerstats
            else "partial"
            if int(driver_stats_diag.get("playerstats_assets_failed", 0) or 0)
            or int(constructor_stats_diag.get("playerstats_assets_failed", 0) or 0)
            else "current"
        ),
        "deadline_data_status": (
            "official"
            if team_lock_payload.get("team_lock_deadline_valid") is True
            else "schedule_fallback"
            if forecast_weekend is not None
            else "unavailable"
        ),
        "snapshot_validation_status": weekend_validation.status,
        "snapshot_validation_safe_for_scoring": weekend_validation.safe_for_scoring,
        "snapshot_validation_warnings": list(weekend_validation.warnings),
        "weekend_state_validation": weekend_validation.as_dict(),
        "active_weekend_state": (
            scoring_active_weekend.as_dict() if scoring_active_weekend else None
        ),
        "forecast_target_event": (
            {
                "season": forecast_weekend.event.season,
                "round": forecast_weekend.event.round,
            }
            if forecast_weekend else None
        ),
        "forecast_target_weekend_state": (
            forecast_weekend.as_dict() if forecast_weekend else None
        ),
        "forecast_target_source": (
            "verified_fantasy_gameday"
            if market_target_hint is not None and forecast_weekend is not None
            and forecast_weekend.event == market_target_hint
            else "grand_prix_finality_and_schedule"
        ),
        "live_session_ingestion": deepcopy(session_ingestion.diagnostics),
        "live_session_states": deepcopy(session_ingestion.diagnostics.get("sessions", {})),
        "live_session_rows_observed": int(len(session_ingestion.results)),
        "live_session_snapshot_signature": live_session_signature(
            session_ingestion.results,
            session_ingestion.states,
        ),
        "completed_current_event_keys": [
            (key.season, key.round) for key in weekend_validation.completed_event_keys
        ] if scoring_active_weekend is not None else None,
        "excluded_partial_current_event_keys": [
            (key.season, key.round) for key in weekend_validation.excluded_partial_event_keys
        ] if scoring_active_weekend is not None else None,
        "team_lock_deadline_warning": deadline_warning,
        "historical_fantasy_data_version": HISTORICAL_FANTASY_DATA_VERSION,
        "historical_fantasy_recorded_rows_loaded": int(len(historical_fantasy_scores)),
        "historical_fantasy_dataset_path": str(DEFAULT_CANONICAL_DATASET_PATH),
    }
    snapshot = LiveDataSnapshot(
        current_season=current_season,
        loaded_start_year=min(loaded_seasons),
        requested_seasons=requested_seasons,
        loaded_seasons=tuple(loaded_seasons),
        season_load_failures=deepcopy(season_load_failures),
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        schedule=schedule,
        players=players,
        teams=teams,
        driver_recent_points=driver_recent,
        constructor_recent_points=constructor_recent,
        driver_race_points=driver_race_points,
        constructor_race_points=constructor_race_points,
        team_lock_payload=deepcopy(team_lock_payload),
        source_diagnostics=source_diagnostics,
        historical_fantasy_scores=historical_fantasy_scores,
        player_assets=player_assets,
        constructor_assets=constructor_assets,
        player_identity_map=player_identity_map,
        session_results=session_ingestion.results.copy(deep=True),
        session_states=tuple(deepcopy(session_ingestion.states)),
    )
    source_diagnostics["scoring_content_signature"] = scoring_content_signature(snapshot)
    return snapshot


def _merge_snapshot_recent_points(df: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    if recent is None or recent.empty:
        return _fill_recent_point_columns(df)
    recent_cols = [
        "recent_points_2ago",
        "recent_points_1ago",
        "recent_points_available",
        "recent_points_source",
        "recent_points_races",
        "recent_points_fallback_used",
        "recent_points_missing",
    ]
    out = df.drop(columns=[col for col in recent_cols if col in df.columns]).merge(
        recent.copy(deep=True),
        on="id",
        how="left",
    )
    return _fill_recent_point_columns(out)


def season_coverage(
    snapshot: LiveDataSnapshot,
    historical_seasons_back: int,
    history_mode: str = HISTORY_MODE_ALL_SUPPORTED,
) -> dict:
    """Resolve exact requested and usable seasons without filling interior gaps."""
    current_season = int(snapshot.current_season)
    if history_mode not in {
        HISTORY_MODE_ALL_SUPPORTED,
        HISTORY_MODE_CURRENT_SEASON_ONLY,
    }:
        raise ValueError(f"Unsupported history mode: {history_mode}")
    requested_start_year = (
        current_season
        if history_mode == HISTORY_MODE_CURRENT_SEASON_ONLY
        else max(
            EARLIEST_PRODUCTION_SEASON,
            current_season - int(historical_seasons_back),
        )
    )
    requested_seasons = tuple(range(requested_start_year, current_season + 1))
    available_seasons = tuple(sorted({
        int(year) for year in snapshot.loaded_seasons
        if EARLIEST_PRODUCTION_SEASON <= int(year) <= current_season
    }))
    available_set = set(available_seasons)
    used_seasons = tuple(year for year in requested_seasons if year in available_set)
    missing_requested_seasons = tuple(year for year in requested_seasons if year not in available_set)
    return {
        "requested_start_year": requested_start_year,
        "requested_seasons": requested_seasons,
        "available_seasons": available_seasons,
        "used_seasons": used_seasons,
        "missing_requested_seasons": missing_requested_seasons,
        "historical_seasons_requested": max(current_season - requested_start_year, 0),
        "historical_seasons_used": len([year for year in used_seasons if year < current_season]),
        "historical_coverage_complete": not missing_requested_seasons,
        "history_mode": history_mode,
    }


def recorded_history_for_mode(
    recorded: pd.DataFrame,
    current_season: int,
    history_mode: str = HISTORY_MODE_ALL_SUPPORTED,
) -> pd.DataFrame:
    """Return a non-mutating recorded-score view for the selected runtime mode."""
    if history_mode not in {
        HISTORY_MODE_ALL_SUPPORTED,
        HISTORY_MODE_CURRENT_SEASON_ONLY,
    }:
        raise ValueError(f"Unsupported history mode: {history_mode}")
    out = recorded.copy(deep=True) if recorded is not None else pd.DataFrame()
    if out.empty or history_mode == HISTORY_MODE_ALL_SUPPORTED:
        return out
    return out[
        pd.to_numeric(out["season"], errors="coerce").eq(int(current_season))
    ].copy()


def _shadow_forecast_diagnostics(
    frame: pd.DataFrame,
    *,
    id_column: str,
    name_column: str,
) -> dict[str, Any]:
    """Summarise shadow EV without invoking any model, source or optimiser path."""
    component_columns = {
        "legacy_next_event_ev": "next_race_expected_points",
        "qualifying_ev": "shadow_next_qualifying_ev",
        "sprint_ev": "shadow_next_sprint_ev",
        "race_ev": "shadow_next_race_ev",
        "shadow_total_ev": "shadow_next_total_ev",
        "uplift_vs_legacy": "sprint_ev_uplift_vs_legacy",
    }
    if frame.empty or "shadow_next_total_ev" not in frame.columns:
        return {"asset_count": 0, "means": {}, "source_counts": {}, "assets": []}
    data = frame.copy(deep=True)
    means = {
        label: (
            float(values.mean()) if values.notna().any() else None
        )
        for label, column in component_columns.items()
        for values in [
            pd.to_numeric(
                data.get(column, pd.Series(index=data.index, dtype=float)),
                errors="coerce",
            )
        ]
    }
    source_columns = [
        column
        for column in (
            "shadow_next_qualifying_source",
            "shadow_next_sprint_source",
            "shadow_next_race_source",
            "shadow_component_status",
        )
        if column in data.columns
    ]
    source_counts = {
        column: {
            str(value): int(count)
            for value, count in data[column].fillna("unavailable").value_counts().items()
        }
        for column in source_columns
    }
    asset_columns = [
        column
        for column in [
            id_column,
            name_column,
            *component_columns.values(),
            *source_columns,
        ]
        if column in data.columns
    ]
    assets = data[asset_columns].replace({float("nan"): None}).to_dict("records")
    return {
        "asset_count": int(len(data)),
        "means": means,
        "source_counts": source_counts,
        "assets": assets,
    }


def derive_model_data(
    snapshot: LiveDataSnapshot,
    today: str | None = None,
    effective_time: datetime | str | None = None,
    historical_seasons_back: int = DEFAULT_HISTORICAL_SEASONS_BACK,
    horizon_races: int = DEFAULT_UPCOMING_RACE_HORIZON,
    current_season_weight: float = 1.0,
    past_season_weight: float = 1.0,
    recency_decay: float = 0.95,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    selected_race_preset: str = "All",
    custom_race_keys: list[RaceKey | tuple[int, int]] | tuple[RaceKey | tuple[int, int], ...] | None = None,
    excluded_race_keys: list[RaceKey | tuple[int, int]] | tuple[RaceKey | tuple[int, int], ...] | None = None,
    history_mode: str = HISTORY_MODE_ALL_SUPPORTED,
    live_session_emphasis: float = 0.0,
) -> ModelData:
    """Derive model tables from a raw snapshot without calling live sources."""
    derive_started = datetime.now(UTC)
    live_session_emphasis = validate_live_session_emphasis(live_session_emphasis)
    current_season = int(snapshot.current_season)
    today = today or derive_started.date().isoformat()
    state_time = coerce_utc_datetime(effective_time or today)
    race_catalogue, race_catalogue_source = snapshot_race_catalogue(snapshot)
    race_selection = resolve_selected_races(
        race_catalogue,
        selected_race_preset,
        custom_keys=custom_race_keys,
        excluded_keys=excluded_race_keys,
    )
    current_race_weights = recency_weights(race_selection, recency_decay)
    coverage = season_coverage(snapshot, historical_seasons_back, history_mode)
    requested_start_year = coverage["requested_start_year"]
    requested_seasons = coverage["requested_seasons"]
    available_seasons = coverage["available_seasons"]
    used_seasons = coverage["used_seasons"]
    missing_requested_seasons = coverage["missing_requested_seasons"]
    if not used_seasons:
        raise ValueError("No successfully loaded seasons are available for the requested model history.")
    start_year = min(used_seasons)

    results = snapshot.results.copy(deep=True)
    qualifying = snapshot.qualifying.copy(deep=True)
    sprint = snapshot.sprint.copy(deep=True)
    schedule = snapshot.schedule.copy(deep=True)
    players = snapshot.players.copy(deep=True)
    teams = snapshot.teams.copy(deep=True)
    driver_race_points, constructor_race_points = effective_current_race_points(snapshot)
    historical_fantasy_scores = recorded_history_for_mode(
        snapshot.historical_fantasy_scores,
        current_season,
        history_mode,
    )
    results = results[results["season"].isin(used_seasons)].copy()
    if not qualifying.empty:
        qualifying = qualifying[qualifying["season"].isin(used_seasons)].copy()
    if not sprint.empty:
        sprint = sprint[sprint["season"].isin(used_seasons)].copy()

    state_validation = validate_weekend_snapshot(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=state_time,
        expected_participant_count=len(players) or None,
    )
    scoring_active_weekend = state_validation.active_weekend
    verified_forecast_event = _event_key_from_diagnostics(
        snapshot.source_diagnostics.get("forecast_target_event")
    )
    if verified_forecast_event is None:
        verified_forecast_event = _market_forecast_event_hint(
            schedule,
            current_season=current_season,
            market_resolution=snapshot.source_diagnostics,
        )
    forecast_weekend = select_forecast_event(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=state_time,
        expected_participant_count=len(players) or None,
        verified_target_event=verified_forecast_event,
    )
    schedule_seasons = pd.to_numeric(
        schedule.get("season", pd.Series(current_season, index=schedule.index)),
        errors="coerce",
    )
    current_schedule = schedule[schedule_seasons == current_season].copy()
    upcoming_events = upcoming_event_records(
        current_schedule,
        start_event=forecast_weekend.event if forecast_weekend is not None else None,
        effective_time=None if forecast_weekend is not None else state_time,
        limit=horizon_races,
        first_weight=1.0,
        later_weight=0.7,
    )
    if upcoming_events:
        first_event = upcoming_events[0].event
        schedule_rounds = pd.to_numeric(current_schedule["round"], errors="coerce")
        schedule_years = pd.to_numeric(
            current_schedule.get("season", pd.Series(current_season, index=current_schedule.index)),
            errors="coerce",
        )
        upcoming_rows = current_schedule[
            (schedule_years > first_event.season)
            | ((schedule_years == first_event.season) & (schedule_rounds >= first_event.round))
        ].sort_values(["season", "round"], kind="stable").head(horizon_races)
        upcoming = upcoming_circuit_names(upcoming_events)
        horizon_weights = [event.horizon_weight for event in upcoming_events]
    else:
        # Compatibility for older fixtures/schedules without canonical season metadata.
        if forecast_weekend is not None:
            upcoming_rows = current_schedule[
                pd.to_numeric(current_schedule["round"], errors="coerce") >= forecast_weekend.event.round
            ].sort_values("round").head(horizon_races)
        else:
            upcoming_rows = current_schedule[
                current_schedule["date"].astype(str) >= today
            ].sort_values("round").head(horizon_races)
        upcoming = [
            circuit.split(" Circuit")[0].strip()
            for circuit in upcoming_rows.get("circuitName", pd.Series(dtype=object)).astype(str).tolist()
        ]
        horizon_weights = _horizon_weights(len(upcoming), w1=1.0, w_next=0.7)
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
    official_deadline = snapshot.team_lock_payload.get("team_lock_deadline_utc")
    matched_deadline_event = snapshot.team_lock_payload.get("team_lock_matched_event")
    try:
        matched_deadline_key = EventKey(
            int(matched_deadline_event[0]), int(matched_deadline_event[1])
        )
    except (IndexError, TypeError, ValueError):
        matched_deadline_key = None
    official_deadline_valid = bool(
        snapshot.team_lock_payload.get("team_lock_deadline_valid") is True
        and forecast_weekend is not None
        and matched_deadline_key == forecast_weekend.event
    )
    if official_deadline and official_deadline_valid:
        team_lock_deadline_utc = official_deadline
        team_lock_deadline_source = snapshot.team_lock_payload.get(
            "team_lock_deadline_source",
            "official_feed_playerstats_session_start",
        )
        team_lock_deadline_raw_field = snapshot.team_lock_payload.get("team_lock_deadline_raw_field")
        team_lock_deadline_raw_value = snapshot.team_lock_payload.get("team_lock_deadline_raw_value")
        team_lock_timezone_assumption = snapshot.team_lock_payload.get(
            "team_lock_timezone_assumption",
            team_lock_timezone_assumption,
        )
    team_lock_validation_reason = snapshot.team_lock_payload.get(
        "team_lock_validation_reason",
        "Schedule fallback used because no validated official deadline was available.",
    )
    if official_deadline and not official_deadline_valid:
        team_lock_validation_reason = (
            "Schedule fallback used because the official deadline did not match "
            "the forecast target event."
        )

    _emit_load_progress(progress_callback, 5, "Building model inputs", "Building model inputs...", progress=0.40)
    weekend_points = compute_weekend_points(
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        current_season=current_season,
        last_season_weight=0.95,
        older_decay=0.75,
        race_dnf_penalty=20,
        sprint_dnf_penalty=10,
        completed_event_keys=set(state_validation.completed_event_keys)
        if weekend_states(
            schedule,
            results=results,
            qualifying=qualifying,
            sprint=sprint,
            effective_time=state_time,
            expected_participant_count=len(players) or None,
        ) else None,
        complete_qualifying_keys=set(state_validation.completed_event_keys),
        complete_sprint_keys=set(state_validation.completed_event_keys),
    )
    official_scores, official_score_warnings = normalise_official_playerstats(
        snapshot.driver_race_points,
        snapshot.constructor_race_points,
        players,
        teams,
        results=results,
        schedule=schedule,
    )
    recorded_scores = resolve_score_precedence(historical_fantasy_scores, official_scores)
    recorded_scores = recorded_scores[recorded_scores["season"].isin(used_seasons)].copy()
    weekend_points, constructor_weekend_points, historical_score_diag = apply_recorded_scores_to_model(
        weekend_points,
        recorded_scores,
    )
    historical_score_diag["historical_fantasy_mapping_warnings"] = official_score_warnings
    baseline_driver_points = weekend_points
    baseline_constructor_points = constructor_weekend_points
    baseline_official_drivers = driver_race_points
    baseline_official_constructors = constructor_race_points
    if forecast_weekend is not None and forecast_weekend.format == WeekendFormat.SPRINT:
        sprint_keys = {
            (int(row.season), int(row.round))
            for row in sprint.itertuples(index=False)
        } if not sprint.empty else set()
        sprint_keys.update(
            (int(row["season"]), int(row["round"]))
            for _, row in schedule.iterrows()
            if weekend_format(row) == WeekendFormat.SPRINT
        )
        baseline_driver_points = normalise_sprint_baseline_inputs(weekend_points, sprint_keys)
        baseline_constructor_points = normalise_sprint_baseline_inputs(constructor_weekend_points, sprint_keys)
        baseline_official_drivers = normalise_sprint_baseline_inputs(driver_race_points, sprint_keys)
        baseline_official_constructors = normalise_sprint_baseline_inputs(constructor_race_points, sprint_keys)
    drv_exp, con_exp = expected_scores_horizon(
        baseline_driver_points,
        upcoming,
        horizon_weights,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        recency_decay=recency_decay,
        selected_race_keys=race_selection.included,
        selected_race_weights=current_race_weights,
        current_season=current_season,
        constructor_weekend_points=baseline_constructor_points,
    )
    if upcoming_events:
        driver_shadow, constructor_shadow = expected_scores_horizon_by_component(
            weekend_points,
            upcoming_events,
            current_season_weight=current_season_weight,
            past_season_weight=past_season_weight,
            recency_decay=recency_decay,
            selected_race_keys=race_selection.included,
            selected_race_weights=current_race_weights,
            current_season=current_season,
            constructor_weekend_points=constructor_weekend_points,
        )
    else:
        driver_shadow = pd.DataFrame()
        constructor_shadow = pd.DataFrame()
    if "next_race_exp_score" not in drv_exp.columns:
        drv_exp["next_race_exp_score"] = drv_exp["exp_score"]
    if "horizon_expected_points" not in drv_exp.columns:
        drv_exp["horizon_expected_points"] = drv_exp["exp_score"]
    if "next_race_exp_score" not in con_exp.columns:
        con_exp["next_race_exp_score"] = con_exp["exp_score"]
    if "horizon_expected_points" not in con_exp.columns:
        con_exp["horizon_expected_points"] = con_exp["exp_score"]
    drv_exp["exp_score"] = pd.to_numeric(drv_exp["next_race_exp_score"], errors="coerce").fillna(drv_exp["exp_score"])
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

    _emit_load_progress(progress_callback, 6, "Computing expected points", "Computing expected points...", progress=0.55)
    drivers = _build_driver_table(players, drv_exp, snapshot.player_identity_map)
    constructors = _build_constructor_table(teams, con_exp)
    if not driver_shadow.empty:
        drivers = drivers.merge(
            driver_shadow.drop(columns=["driver"], errors="ignore"),
            on="driverId",
            how="left",
            validate="many_to_one",
        )
    if not constructor_shadow.empty:
        constructors = constructors.merge(
            constructor_shadow.drop(columns=["constructor"], errors="ignore"),
            on="constructorId",
            how="left",
            validate="many_to_one",
        )
    drivers = _apply_team_strength_adjustment(drivers, constructors)
    drivers = _merge_snapshot_recent_points(drivers, snapshot.driver_recent_points)
    constructors = _merge_snapshot_recent_points(constructors, snapshot.constructor_recent_points)
    _emit_load_progress(
        progress_callback,
        7,
        "Computing price-change probabilities",
        "Computing price-change probabilities...",
        progress=0.90,
    )
    drivers, constructors, calibration_diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        baseline_official_drivers,
        baseline_official_constructors,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        selected_races=race_selection,
        race_weights=current_race_weights,
        driver_source_failures=_playerstats_failure_ids(snapshot.driver_recent_points, "driver"),
        constructor_source_failures=_playerstats_failure_ids(snapshot.constructor_recent_points, "constructor"),
        horizon_weight_sum=float(sum(horizon_weights)),
    )
    approved_sprint_shadow_diagnostics: dict[str, Any]
    try:
        approved_sprint_shadow = calculate_sprint_shadow(
            snapshot.historical_fantasy_scores,
            schedule,
            drivers,
            constructors,
            race_selection.included,
            recency_decay,
            upcoming_events[0].event if upcoming_events else None,
            production_history_mode=history_mode,
        )
        drivers = drivers.merge(
            approved_sprint_shadow.drivers,
            on="id",
            how="left",
            validate="one_to_one",
        )
        constructors = constructors.merge(
            approved_sprint_shadow.constructors,
            on="id",
            how="left",
            validate="one_to_one",
        )
        approved_sprint_shadow_diagnostics = approved_sprint_shadow.diagnostics
    except Exception as exc:
        approved_sprint_shadow_diagnostics = {
            "label": "Shadow / experimental — not used by optimiser",
            "status": "unavailable",
            "error": str(exc),
            "production_history_mode": history_mode,
            "sprint_shadow_history": "2026_only",
            "production_isolation": (
                "The optional shadow calculation failed closed; production expected points, "
                "prices and optimiser inputs remain unchanged."
            ),
        }
    for frame in (drivers, constructors):
        if "shadow_next_total_ev" in frame.columns:
            frame["sprint_aware_next_event_ev"] = pd.to_numeric(
                frame["shadow_next_total_ev"], errors="coerce"
            )
            frame["sprint_aware_horizon_ev"] = pd.to_numeric(
                frame["shadow_horizon_total_ev"], errors="coerce"
            )
            frame["sprint_ev_uplift_vs_legacy"] = (
                frame["sprint_aware_next_event_ev"]
                - pd.to_numeric(frame["next_race_expected_points"], errors="coerce")
            )
    sprint_production_diagnostics: dict[str, Any]
    try:
        sprint_production = calculate_sprint_production_adjustment(
            snapshot.historical_fantasy_scores,
            schedule,
            drivers,
            constructors,
            race_selection.included,
            recency_decay,
            upcoming_events[0].event if upcoming_events else None,
            production_history_mode=history_mode,
        )
        drivers = apply_sprint_production_adjustment(
            drivers, sprint_production.drivers
        )
        constructors = apply_sprint_production_adjustment(
            constructors, sprint_production.constructors
        )
        sprint_production_diagnostics = sprint_production.diagnostics
    except Exception as exc:
        upcoming_format = (
            upcoming_events[0].format.value if upcoming_events else "unknown"
        )
        for frame in (drivers, constructors):
            baseline = pd.to_numeric(frame["next_race_expected_points"], errors="coerce")
            frame["baseline_expected_points"] = baseline
            frame["sprint_bonus"] = 0.0
            frame["sprint_adjusted_expected_points"] = baseline
            frame["sprint_calibration_version"] = "unavailable"
            frame["sprint_calibration_season"] = 2026
            frame["sprint_weekend_format"] = upcoming_format
            frame["sprint_bonus_applied"] = False
            frame["sprint_bonus_status"] = "calibration_unavailable_baseline_only"
            frame["sprint_bonus_driver_group_component"] = pd.NA
            frame["sprint_bonus_driver_personal_component"] = pd.NA
            frame["sprint_bonus_driver_weight"] = pd.NA
            frame["sprint_constructor_strength"] = pd.NA
        sprint_production_diagnostics = {
            "label": "Approved production Sprint adjustment",
            "status": "unavailable",
            "error": str(exc),
            "model_version": "unavailable",
            "calibration_season": 2026,
            "calibration_status": "unavailable",
            "upcoming_weekend_format": upcoming_format,
            "bonus_applied": False,
            "production_semantics": (
                "Calibration failed closed; the existing production baseline remains unchanged."
            ),
        }
    driver_stats_diag = deepcopy(snapshot.source_diagnostics.get("driver_stats_diag", {}))
    constructor_stats_diag = deepcopy(snapshot.source_diagnostics.get("constructor_stats_diag", {}))
    recent_diag = playerstats_recent_points_diagnostics(
        drivers,
        constructors,
        driver_race_points,
        constructor_race_points,
        driver_stats_diag,
        constructor_stats_diag,
    )
    race_catalogue_source_failures = sum(
        int(stats.get("playerstats_assets_failed", 0) or 0)
        + int(stats.get("playerstats_skipped_after_failure_limit", 0) or 0)
        for stats in (driver_stats_diag, constructor_stats_diag)
    )
    round_lineage = current_season_round_lineage(
        snapshot,
        list(race_selection.included),
    )
    race_catalogue_diag = {
        "current_season_race_catalogue": [
            {
                "season": option.key.season,
                "round": option.key.round,
                "race_name": option.race_name,
            }
            for option in race_catalogue
        ],
        "current_season_completed_race_count": len(race_catalogue),
        "current_season_race_catalogue_source": race_catalogue_source,
        "current_season_race_catalogue_source_failure_count": race_catalogue_source_failures,
        "current_season_race_catalogue_has_source_failures": race_catalogue_source_failures > 0,
        "selected_race_preset": race_selection.preset,
        "selected_race_keys": [(key.season, key.round) for key in race_selection.included],
        "excluded_race_keys": [(key.season, key.round) for key in race_selection.excluded],
        "selected_race_weights": {
            f"{key.season}:{key.round}": float(weight)
            for key, weight in current_race_weights.items()
        },
        "current_season_round_lineage": round_lineage.to_dict("records"),
        "eligible_current_season_race_count": int(len(race_catalogue)),
    }
    drivers = ensure_image_url_column(drivers)
    constructors = ensure_image_url_column(constructors)
    drivers["team_colour"] = drivers["team"].apply(team_colour) if "team" in drivers.columns else DEFAULT_TEAM_COLOUR
    constructors["team_colour"] = constructors["name"].apply(team_colour)
    shadow_forecast_diagnostics = {
        "label": "Sprint-aware shadow forecast — not yet used for prices or optimisation",
        "active_event": upcoming_events[0].as_dict() if upcoming_events else None,
        "drivers": _shadow_forecast_diagnostics(
            drivers,
            id_column="driverId",
            name_column="name",
        ),
        "constructors": _shadow_forecast_diagnostics(
            constructors,
            id_column="constructorId",
            name_column="name",
        ),
        "production_isolation": (
            "Legacy next_race_expected_points remains the baseline for the separate "
            "Sprint-aware shadow comparison; the live-session production blend is "
            "applied independently afterwards."
        ),
    }
    live_session_shadow_diagnostics: dict[str, Any]
    try:
        live_shadow = build_live_session_shadow(
            drivers,
            constructors,
            snapshot.session_results,
            snapshot.session_states,
            forecast_weekend.format if forecast_weekend is not None else WeekendFormat.NORMAL,
            forecast_event=forecast_weekend.event if forecast_weekend is not None else None,
        )
        drivers = apply_live_session_emphasis(
            live_shadow.drivers, live_session_emphasis
        )
        constructors = apply_live_session_emphasis(
            live_shadow.constructors, live_session_emphasis
        )
        live_session_shadow_diagnostics = {
            **live_shadow.diagnostics,
            "label": "Live-session production blend",
            "live_session_emphasis": live_session_emphasis,
            "production_semantics": (
                "The next-event baseline is blended with live-only EV; only that "
                "first-event delta is added to the horizon total."
            ),
        }
    except Exception as exc:
        drivers = apply_live_session_emphasis(
            drivers,
            live_session_emphasis,
            baseline_column="next_race_expected_points",
        )
        constructors = apply_live_session_emphasis(
            constructors,
            live_session_emphasis,
            baseline_column="next_race_expected_points",
        )
        live_session_shadow_diagnostics = {
            "label": "Live-session production blend",
            "status": "unavailable",
            "error": str(exc),
            "live_session_emphasis": live_session_emphasis,
            "production_semantics": (
                "The live-session layer failed closed; the Sprint-adjusted baseline remains in use."
            ),
        }
    driver_price_efficiency = build_price_efficiency_table(
        drivers,
        driver_race_points,
        race_selection,
        weights=current_race_weights,
        asset_type="driver",
        source_failures=_playerstats_failure_ids(snapshot.driver_recent_points, "driver"),
    )
    constructor_price_efficiency = build_price_efficiency_table(
        constructors,
        constructor_race_points,
        race_selection,
        weights=current_race_weights,
        asset_type="constructor",
        source_failures=_playerstats_failure_ids(snapshot.constructor_recent_points, "constructor"),
    )
    trends = build_trends_data(drivers, constructors, driver_race_points, constructor_race_points)
    derive_finished = datetime.now(UTC)
    derive_seconds = max(0.0, (derive_finished - derive_started).total_seconds())
    include_playerstats = bool(snapshot.source_diagnostics.get("playerstats_prefetch_enabled", False))
    diagnostics = {
        "current_season": current_season,
        "start_year": start_year,
        "requested_start_year": requested_start_year,
        "historical_seasons_back": int(historical_seasons_back),
        "history_mode": history_mode,
        "current_season_only": history_mode == HISTORY_MODE_CURRENT_SEASON_ONLY,
        "historical_seasons_requested": coverage["historical_seasons_requested"],
        "historical_seasons_used": coverage["historical_seasons_used"],
        "requested_seasons": list(requested_seasons),
        "available_seasons": list(available_seasons),
        "used_seasons": list(used_seasons),
        "missing_requested_seasons": list(missing_requested_seasons),
        "historical_coverage_complete": coverage["historical_coverage_complete"],
        "season_load_failures": deepcopy(snapshot.season_load_failures),
        "today": today,
        "feed_round": int(snapshot.source_diagnostics.get("feed_round", 0)),
        "upcoming_circuits": upcoming,
        "upcoming_event_records": [event.as_dict() for event in upcoming_events],
        "sprint_aware_shadow_forecast": shadow_forecast_diagnostics,
        "live_session_shadow": live_session_shadow_diagnostics,
        "approved_sprint_ev_shadow": approved_sprint_shadow_diagnostics,
        "sprint_ev_production": sprint_production_diagnostics,
        "next_race_name": next_race_name,
        "next_race_date": next_race_date,
        "next_race_round": int(next_race_round) if pd.notna(next_race_round) else None,
        "active_event": (
            {"season": forecast_weekend.event.season, "round": forecast_weekend.event.round}
            if forecast_weekend else None
        ),
        "forecast_target_event": (
            {"season": forecast_weekend.event.season, "round": forecast_weekend.event.round}
            if forecast_weekend else None
        ),
        "scoring_active_event": (
            {
                "season": scoring_active_weekend.event.season,
                "round": scoring_active_weekend.event.round,
            }
            if scoring_active_weekend else None
        ),
        "weekend_format": forecast_weekend.format.value if forecast_weekend else "unknown",
        "weekend_status": forecast_weekend.status if forecast_weekend else "unavailable",
        "weekend_is_final": bool(forecast_weekend.is_final) if forecast_weekend else False,
        "weekend_session_states": (
            forecast_weekend.as_dict().get("sessions", {}) if forecast_weekend else {}
        ),
        "completed_scoring_event_keys": [
            (key.season, key.round) for key in state_validation.completed_event_keys
        ],
        "excluded_partial_scoring_event_keys": [
            (key.season, key.round) for key in state_validation.excluded_partial_event_keys
        ],
        "completed_form_excludes_live_weekend": bool(
            scoring_active_weekend is not None and not scoring_active_weekend.is_final
        ),
        "weekend_state_warnings": list(state_validation.warnings),
        "team_lock_deadline_utc": team_lock_deadline_utc,
        "team_lock_deadline_source": team_lock_deadline_source,
        "team_lock_deadline_raw_field": team_lock_deadline_raw_field,
        "team_lock_deadline_raw_value": team_lock_deadline_raw_value,
        "team_lock_timezone_assumption": team_lock_timezone_assumption,
        "team_lock_matched_event": snapshot.team_lock_payload.get("team_lock_matched_event"),
        "team_lock_selected_session_meaning": snapshot.team_lock_payload.get(
            "team_lock_selected_session_meaning"
        ),
        "team_lock_validation_reason": team_lock_validation_reason,
        "upcoming_race_horizon": int(horizon_races),
        "horizon_weights": horizon_weights,
        "horizon_weight_sum": float(sum(horizon_weights)),
        "horizon_race_count": len(upcoming),
        "current_season_weight": float(current_season_weight),
        "past_season_weight": float(past_season_weight),
        "recency_decay": float(recency_decay),
        "live_session_emphasis": live_session_emphasis,
        "driver_count": len(drivers),
        "constructor_count": len(constructors),
        "driver_dnf_rate_missing": int(pd.to_numeric(drivers.get("dnf_rate", pd.Series(dtype=float)), errors="coerce").isna().sum()),
        "constructor_dnf_rate_missing": int(pd.to_numeric(constructors.get("dnf_rate", pd.Series(dtype=float)), errors="coerce").isna().sum()),
        "dnf_price_gain_score": float(DEFAULT_DNF_PRICE_GAIN_SCORE),
        "race_dnf_bad_score": float(DEFAULT_RACE_DNF_BAD_SCORE),
        "sprint_dnf_bad_score": float(DEFAULT_SPRINT_DNF_BAD_SCORE),
        "dnf_price_gain_score_source": "Fixed generic race-weekend bad-outcome score; repo scoring uses -20 race DNF and -10 sprint DNF.",
        "playerstats_prefetch_enabled": include_playerstats,
        "model_load_started_utc": derive_started.isoformat(),
        "model_load_finished_utc": derive_finished.isoformat(),
        "model_load_duration_seconds": float(derive_seconds),
        "playerstats_load_duration_seconds": float(snapshot.source_diagnostics.get("playerstats_load_duration_seconds", 0.0)),
        "model_load_events": list(snapshot.source_diagnostics.get("raw_live_events", []))[-40:],
        **{key: deepcopy(value) for key, value in snapshot.source_diagnostics.items() if key not in {"driver_stats_diag", "constructor_stats_diag"}},
        **calibration_diag,
        **recent_diag,
        **race_catalogue_diag,
        **historical_score_diag,
    }
    if not include_playerstats:
        diagnostics["recent_points_source"] = "Playerstats prefetch skipped for faster startup; per-race fields may be incomplete until enrichment."
    diagnostics["playerstats_timeout_failures"] = int(driver_stats_diag.get("playerstats_timeout_failures", 0)) + int(
        constructor_stats_diag.get("playerstats_timeout_failures", 0)
    )
    diagnostics["playerstats_skipped_after_failure_limit"] = int(
        driver_stats_diag.get("playerstats_skipped_after_failure_limit", 0)
    ) + int(constructor_stats_diag.get("playerstats_skipped_after_failure_limit", 0))
    _emit_load_progress(progress_callback, 8, "Ready", f"Ready. Model derived in {derive_seconds:.1f}s.", progress=1.0, status="complete")
    return ModelData(
        drivers=drivers,
        constructors=constructors,
        trends=trends,
        diagnostics=diagnostics,
        driver_price_efficiency=driver_price_efficiency,
        constructor_price_efficiency=constructor_price_efficiency,
    )


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
    force_refresh: bool = False,
    selected_race_preset: str = "All",
    custom_race_keys: list[RaceKey | tuple[int, int]] | tuple[RaceKey | tuple[int, int], ...] | None = None,
    excluded_race_keys: list[RaceKey | tuple[int, int]] | tuple[RaceKey | tuple[int, int], ...] | None = None,
    effective_time: datetime | str | None = None,
    history_mode: str = HISTORY_MODE_ALL_SUPPORTED,
    live_session_emphasis: float = 0.0,
) -> ModelData:
    """Compatibility orchestration for callers that still need fetch + derive."""
    snapshot = load_live_data_snapshot(
        current_season=current_season,
        historical_seasons_back=historical_seasons_back,
        include_playerstats=include_playerstats,
        force_refresh=force_refresh,
        progress_callback=progress_callback,
        effective_time=effective_time,
    )
    return derive_model_data(
        snapshot,
        today=today,
        effective_time=effective_time,
        historical_seasons_back=historical_seasons_back,
        horizon_races=horizon_races,
        current_season_weight=current_season_weight,
        past_season_weight=past_season_weight,
        recency_decay=recency_decay,
        progress_callback=progress_callback,
        selected_race_preset=selected_race_preset,
        custom_race_keys=custom_race_keys,
        excluded_race_keys=excluded_race_keys,
        history_mode=history_mode,
        live_session_emphasis=live_session_emphasis,
    )


def _build_driver_table(
    players: pd.DataFrame,
    drv_exp: pd.DataFrame,
    player_identity_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
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
    component_columns = [
        "current_proxy_next_race_expected_points",
        "current_proxy_horizon_expected_points",
        "historical_next_race_expected_points",
        "historical_horizon_expected_points",
        "current_proxy_volatility",
        "historical_volatility",
    ]
    mapped_columns = [
        "driverId",
        "exp_score",
        "next_race_exp_score",
        "horizon_expected_points",
        "dnf_rate",
        "volatility",
        "nn_exp_score",
        *[column for column in component_columns if column in drv_exp.columns],
    ]
    canon_to_row = drv_exp.set_index("canon_name")[mapped_columns].to_dict("index")
    driver_id_to_row = (
        drv_exp.drop_duplicates("driverId", keep="last")
        .set_index("driverId", drop=False)[mapped_columns]
        .to_dict("index")
    )
    identity_by_asset: dict[int, dict[str, Any]] = {}
    if isinstance(player_identity_map, pd.DataFrame) and not player_identity_map.empty:
        identity_by_asset = {
            int(row["fantasy_asset_id"]): row
            for row in player_identity_map.to_dict("records")
            if pd.notna(row.get("fantasy_asset_id"))
        }

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

    def map_asset_row(row: pd.Series):
        asset_id = pd.to_numeric(row.get("playerId"), errors="coerce")
        identity = identity_by_asset.get(int(asset_id)) if pd.notna(asset_id) else None
        history_driver_id = identity.get("history_driver_id") if identity else None
        if history_driver_id is not None and not pd.isna(history_driver_id):
            explicit = driver_id_to_row.get(str(history_driver_id))
            if explicit is not None:
                return explicit
        return map_driver_row(str(row.get("canon_name") or ""))

    mapped = fp.apply(map_asset_row, axis=1)
    drivers = fp.copy()
    if identity_by_asset:
        drivers["human_driver_id"] = drivers["playerId"].map(
            lambda value: identity_by_asset.get(int(value), {}).get("human_driver_id")
        )
        drivers["identity_match_method"] = drivers["playerId"].map(
            lambda value: identity_by_asset.get(int(value), {}).get("match_method")
        )
        drivers["identity_match_status"] = drivers["playerId"].map(
            lambda value: identity_by_asset.get(int(value), {}).get("match_status")
        )
    drivers["driverId"] = mapped.apply(lambda x: x["driverId"] if isinstance(x, dict) else None)
    drivers["exp_score"] = mapped.apply(lambda x: x["exp_score"] if isinstance(x, dict) else None)
    drivers["next_race_exp_score"] = mapped.apply(lambda x: x.get("next_race_exp_score") if isinstance(x, dict) else None)
    drivers["horizon_expected_points"] = mapped.apply(lambda x: x.get("horizon_expected_points") if isinstance(x, dict) else None)
    drivers["dnf_rate"] = mapped.apply(lambda x: x["dnf_rate"] if isinstance(x, dict) else None)
    drivers["volatility"] = mapped.apply(lambda x: x["volatility"] if isinstance(x, dict) else None)
    drivers["nn_exp_score"] = mapped.apply(lambda x: x.get("nn_exp_score", x["exp_score"]) if isinstance(x, dict) else None)
    for column in component_columns:
        drivers[column] = mapped.apply(lambda row, key=column: row.get(key) if isinstance(row, dict) else None)

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
    component_columns = [
        "current_proxy_next_race_expected_points",
        "current_proxy_horizon_expected_points",
        "historical_next_race_expected_points",
        "historical_horizon_expected_points",
        "current_proxy_volatility",
        "historical_volatility",
    ]
    mapped_columns = [
        "constructorId",
        "exp_score",
        "next_race_exp_score",
        "horizon_expected_points",
        "dnf_rate",
        "volatility",
        *[column for column in component_columns if column in con_exp.columns],
    ]
    con_keys = con_exp.set_index("canon_team")[mapped_columns].to_dict("index")

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
    for column in component_columns:
        constructors[column] = mapped.apply(lambda row, key=column: row.get(key) if isinstance(row, dict) else None)

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
    for component_column in [
        "current_proxy_next_race_expected_points",
        "current_proxy_horizon_expected_points",
        "historical_next_race_expected_points",
        "historical_horizon_expected_points",
    ]:
        if component_column in drivers.columns:
            drivers[component_column] = (
                pd.to_numeric(drivers[component_column], errors="coerce")
                * drivers["team_factor"]
            )
    for shadow_column in [
        column
        for column in drivers.columns
        if column.startswith("shadow_") and column.endswith("_ev")
    ]:
        drivers[shadow_column] = (
            pd.to_numeric(drivers[shadow_column], errors="coerce")
            * drivers["team_factor"]
        )
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


def _playerstats_failure_ids(recent_points: pd.DataFrame, asset_type: str) -> list[tuple[str, str]]:
    if recent_points is None or recent_points.empty or "id" not in recent_points.columns:
        return []
    source = recent_points.get(
        "recent_points_source",
        pd.Series("", index=recent_points.index, dtype=object),
    ).fillna("").astype(str).str.casefold()
    failed = source.str.contains("failed|failure|skipped|timeout|error", regex=True)
    return [(str(asset_type), str(asset_id)) for asset_id in recent_points.loc[failed, "id"]]


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


def _observed_average_by_player(
    race_points: pd.DataFrame,
    selected_races: RaceSelection | None = None,
    race_weights: dict[RaceKey, float] | None = None,
    asset_type: str = "unknown",
    source_failures: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    columns = [
        "id",
        "observed_current_avg_points",
        "observed_current_races",
        "current_season_avg_points",
        "current_season_points_count",
        "current_season_volatility",
        "official_selected_race_count",
        "official_valid_race_count",
        "official_missing_race_count",
        "official_coverage_fraction",
        "official_has_source_failure",
        "official_observation_status",
        "official_valid_race_keys",
    ]
    if race_points is None or race_points.empty:
        if selected_races is None or not source_failures:
            return pd.DataFrame(columns=columns)
        weighted = weighted_asset_points(
            pd.DataFrame(),
            selected_races,
            race_weights or {},
            asset_type=asset_type,
            source_failures=source_failures,
        )
    elif selected_races is None or not {"season", "round"}.issubset(race_points.columns):
        data = race_points.copy(deep=True)
        played = pd.to_numeric(data.get("is_played", 0), errors="coerce")
        if not isinstance(played, pd.Series):
            played = pd.Series(played, index=data.index)
        played = played.fillna(0).astype(int)
        data["fantasy_points"] = pd.to_numeric(data.get("fantasy_points"), errors="coerce")
        data = data[(played == 1) & data["fantasy_points"].notna()].copy()
        if data.empty:
            return pd.DataFrame(columns=columns)

        def observed_volatility(points: pd.Series):
            clean = pd.to_numeric(points, errors="coerce").dropna()
            return float(clean.std(ddof=0)) if len(clean) >= 2 else pd.NA

        grouped = data.groupby("PlayerId", as_index=False).agg(
            observed_current_avg_points=("fantasy_points", "mean"),
            observed_current_races=("fantasy_points", "count"),
            current_season_volatility=("fantasy_points", observed_volatility),
        )
        grouped.rename(columns={"PlayerId": "id"}, inplace=True)
        grouped["current_season_avg_points"] = grouped["observed_current_avg_points"]
        grouped["current_season_points_count"] = grouped["observed_current_races"]
        grouped["official_selected_race_count"] = grouped["observed_current_races"]
        grouped["official_valid_race_count"] = grouped["observed_current_races"]
        grouped["official_missing_race_count"] = 0
        grouped["official_coverage_fraction"] = 1.0
        grouped["official_has_source_failure"] = False
        grouped["official_observation_status"] = "complete"
        grouped["official_valid_race_keys"] = [()] * len(grouped)
        return grouped[columns]
    else:
        weighted = weighted_asset_points(
            race_points,
            selected_races,
            race_weights or {},
            asset_type=asset_type,
            source_failures=source_failures,
        )

    if weighted.empty:
        return pd.DataFrame(columns=columns)
    observed = weighted.rename(
        columns={
            "asset_id": "id",
            "weighted_points": "observed_current_avg_points",
            "valid_race_count": "observed_current_races",
            "selected_race_count": "official_selected_race_count",
            "missing_race_count": "official_missing_race_count",
            "coverage_fraction": "official_coverage_fraction",
            "has_source_failure": "official_has_source_failure",
            "status": "official_observation_status",
            "valid_race_keys": "official_valid_race_keys",
        }
    )
    observed["official_valid_race_count"] = observed["observed_current_races"]
    observed["current_season_avg_points"] = observed["observed_current_avg_points"]
    observed["current_season_points_count"] = observed["observed_current_races"]

    volatility_by_id: dict[str, float | Any] = {}
    selected_set = set(selected_races.included if selected_races is not None else ())
    data = race_points.copy(deep=True) if race_points is not None else pd.DataFrame()
    if not data.empty and {"PlayerId", "season", "round", "fantasy_points"}.issubset(data.columns):
        data["fantasy_points"] = pd.to_numeric(data["fantasy_points"], errors="coerce")
        if "is_played" in data.columns:
            played = pd.to_numeric(data["is_played"], errors="coerce").fillna(0).astype(int)
            data = data[played == 1].copy()
        data = data[data["fantasy_points"].notna()].copy()
        data["_race_key"] = [
            RaceKey(int(season), int(round_no))
            for season, round_no in zip(data["season"], data["round"])
        ]
        data = data[data["_race_key"].isin(selected_set)].copy()
        collapsed = data.groupby(["PlayerId", "_race_key"], as_index=False)["fantasy_points"].mean()
        for player_id, group in collapsed.groupby("PlayerId"):
            volatility_by_id[str(player_id)] = (
                float(group["fantasy_points"].std(ddof=0)) if len(group) >= 2 else pd.NA
            )
    observed["current_season_volatility"] = observed["id"].astype(str).map(volatility_by_id)
    return observed[columns]


def _blend_available_components(
    current: pd.Series,
    historical: pd.Series,
    current_weight: float,
    historical_weight: float,
    fallback: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    current_values = pd.to_numeric(current, errors="coerce")
    historical_values = pd.to_numeric(historical, errors="coerce")
    current_available = current_values.notna()
    historical_available = historical_values.notna()
    current_configured = max(0.0, float(current_weight))
    historical_configured = max(0.0, float(historical_weight))

    current_share = pd.Series(0.0, index=current_values.index, dtype=float)
    historical_share = pd.Series(0.0, index=current_values.index, dtype=float)
    both = current_available & historical_available
    only_current = current_available & ~historical_available
    only_historical = ~current_available & historical_available
    denominator = current_configured + historical_configured
    if denominator > 0:
        current_share.loc[both] = current_configured / denominator
        historical_share.loc[both] = historical_configured / denominator
    else:
        current_share.loc[both] = 0.5
        historical_share.loc[both] = 0.5
    current_share.loc[only_current] = 1.0
    historical_share.loc[only_historical] = 1.0
    blended = current_values.fillna(0.0) * current_share + historical_values.fillna(0.0) * historical_share
    blended.loc[~current_available & ~historical_available] = pd.NA
    if fallback is not None:
        blended = blended.combine_first(pd.to_numeric(fallback, errors="coerce"))
    return blended, current_share, historical_share


def apply_observed_playerstats_projection(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    driver_race_points: pd.DataFrame,
    constructor_race_points: pd.DataFrame,
    current_season_weight: float,
    past_season_weight: float,
    driver_volatility_floor: float = DEFAULT_DRIVER_SCORE_VOLATILITY_FLOOR,
    constructor_volatility_floor: float = DEFAULT_CONSTRUCTOR_SCORE_VOLATILITY_FLOOR,
    selected_races: RaceSelection | None = None,
    race_weights: dict[RaceKey, float] | None = None,
    driver_source_failures: list[tuple[str, str]] | None = None,
    constructor_source_failures: list[tuple[str, str]] | None = None,
    horizon_weight_sum: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Blend selected official current form with historical projection exactly once."""
    driver_obs = _observed_average_by_player(
        driver_race_points,
        selected_races,
        race_weights,
        "driver",
        driver_source_failures,
    )
    constructor_obs = _observed_average_by_player(
        constructor_race_points,
        selected_races,
        race_weights,
        "constructor",
        constructor_source_failures,
    )
    combined_observed = pd.concat([driver_obs, constructor_obs], ignore_index=True)
    current_avg = pd.to_numeric(combined_observed.get("observed_current_avg_points"), errors="coerce").mean()

    historical_values = pd.concat(
        [
            pd.to_numeric(
                drivers.get("historical_next_race_expected_points", drivers.get("next_race_exp_score", drivers.get("exp_score"))),
                errors="coerce",
            ),
            pd.to_numeric(
                constructors.get("historical_next_race_expected_points", constructors.get("next_race_exp_score", constructors.get("exp_score"))),
                errors="coerce",
            ),
        ],
        ignore_index=True,
    )
    historical_avg = historical_values.dropna().mean()
    scale, clipped = historical_scale_factor(current_avg, historical_avg)

    def apply_one(df: pd.DataFrame, obs: pd.DataFrame, volatility_floor: float) -> pd.DataFrame:
        out = df.copy(deep=True)
        out["_asset_join_id"] = out["id"].astype(str)
        observations = obs.copy(deep=True)
        observations["_asset_join_id"] = observations.get(
            "id", pd.Series(dtype=object)
        ).astype(str)
        observations.drop(columns=["id"], inplace=True, errors="ignore")
        out = out.merge(observations, on="_asset_join_id", how="left")
        out.drop(columns=["_asset_join_id"], inplace=True)

        if selected_races is not None:
            selected_count = len(selected_races.included)
            out["official_selected_race_count"] = pd.to_numeric(
                out.get("official_selected_race_count"), errors="coerce"
            ).fillna(selected_count).astype(int)
            out["official_valid_race_count"] = pd.to_numeric(
                out.get("official_valid_race_count"), errors="coerce"
            ).fillna(0).astype(int)
            out["official_missing_race_count"] = (
                out["official_selected_race_count"] - out["official_valid_race_count"]
            ).clip(lower=0).astype(int)
            out["official_coverage_fraction"] = (
                out["official_valid_race_count"]
                / out["official_selected_race_count"].replace(0, pd.NA)
            ).fillna(0.0)
            out["official_has_source_failure"] = out.get(
                "official_has_source_failure", pd.Series(False, index=out.index)
            ).fillna(False).astype(bool)
            default_status = "no_races_selected" if selected_count == 0 else "no_valid_observations"
            out["official_observation_status"] = out.get(
                "official_observation_status", pd.Series(default_status, index=out.index)
            ).fillna(default_status)
            out["official_valid_race_keys"] = out.get(
                "official_valid_race_keys", pd.Series([()] * len(out), index=out.index)
            ).apply(lambda value: value if isinstance(value, tuple) else ())
            out["observed_current_races"] = pd.to_numeric(
                out.get("observed_current_races"), errors="coerce"
            ).fillna(0).astype(int)
            out["current_season_points_count"] = pd.to_numeric(
                out.get("current_season_points_count"), errors="coerce"
            ).fillna(0).astype(int)

        original_next = pd.to_numeric(
            out.get("next_race_exp_score", out.get("exp_score")), errors="coerce"
        )
        original_horizon = pd.to_numeric(
            out.get("horizon_expected_points", original_next), errors="coerce"
        )
        historical_next_raw = pd.to_numeric(
            out.get("historical_next_race_expected_points", original_next), errors="coerce"
        )
        historical_horizon_raw = pd.to_numeric(
            out.get("historical_horizon_expected_points", original_horizon), errors="coerce"
        )
        current_proxy_next_raw = pd.to_numeric(
            out.get(
                "current_proxy_next_race_expected_points",
                pd.Series(pd.NA, index=out.index, dtype="Float64"),
            ),
            errors="coerce",
        )
        out["historical_proxy_next_race_exp_score"] = historical_next_raw
        out["historical_proxy_horizon_expected_points"] = historical_horizon_raw
        out["normalised_historical_expected_points_per_race"] = historical_next_raw * scale
        out["normalised_historical_horizon_expected_points"] = historical_horizon_raw * scale
        out["normalised_current_proxy_expected_points_per_race"] = current_proxy_next_raw * scale

        official_current = pd.to_numeric(out.get("observed_current_avg_points"), errors="coerce")
        current_proxy = out["normalised_current_proxy_expected_points_per_race"]
        current_component = official_current.combine_first(current_proxy)
        out["current_component_expected_points_per_race"] = current_component
        out["current_component_source"] = "unavailable"
        out.loc[current_proxy.notna(), "current_component_source"] = "current_proxy_fallback"
        out.loc[official_current.notna(), "current_component_source"] = "official_current"
        historical_next = out["normalised_historical_expected_points_per_race"]
        final_next, current_share, historical_share = _blend_available_components(
            current_component,
            historical_next,
            current_season_weight,
            past_season_weight,
            fallback=original_next,
        )
        out["effective_current_share"] = current_share
        out["effective_historical_share"] = historical_share
        out["next_race_exp_score"] = final_next
        out["next_race_expected_points"] = final_next
        out["exp_score"] = final_next

        if horizon_weight_sum is None:
            inferred_multiplier = (
                historical_horizon_raw / historical_next_raw.replace(0, pd.NA)
            ).replace([float("inf"), float("-inf")], pd.NA)
            multiplier = inferred_multiplier.fillna(1.0)
        else:
            multiplier = pd.Series(float(horizon_weight_sum), index=out.index, dtype=float)
        out["horizon_weight_sum"] = pd.to_numeric(multiplier, errors="coerce").fillna(1.0)
        current_horizon = current_component * out["horizon_weight_sum"]
        historical_horizon = out["normalised_historical_horizon_expected_points"]
        final_horizon, _horizon_current_share, _horizon_historical_share = _blend_available_components(
            current_horizon,
            historical_horizon,
            current_season_weight,
            past_season_weight,
            fallback=original_horizon,
        )
        out["current_component_horizon_expected_points"] = current_horizon
        out["historical_component_horizon_expected_points"] = historical_horizon
        out["horizon_expected_points"] = final_horizon

        both_components = current_component.notna() & historical_next.notna()
        current_only = current_component.notna() & historical_next.isna()
        historical_only = current_component.isna() & historical_next.notna()
        out["expected_points_source"] = "safe_fallback"
        out.loc[both_components, "expected_points_source"] = "blended"
        out.loc[current_only, "expected_points_source"] = out.loc[
            current_only, "current_component_source"
        ]
        out.loc[historical_only, "expected_points_source"] = "historical_only"
        current_endpoint = both_components & (current_share == 1.0)
        out.loc[current_endpoint, "expected_points_source"] = out.loc[
            current_endpoint, "current_component_source"
        ]
        out.loc[both_components & (historical_share == 1.0), "expected_points_source"] = "historical_only"
        if "nn_exp_score" in out.columns:
            out["nn_exp_score"] = out["exp_score"]
        out["current_season_observed_avg_points_per_race"] = out["observed_current_avg_points"]
        out["historical_prior_expected_points_per_race"] = historical_next_raw

        current_vol = pd.to_numeric(
            out.get("current_season_volatility", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        )
        current_proxy_vol = pd.to_numeric(
            out.get("current_proxy_volatility", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        ) * scale
        current_vol = current_vol.combine_first(current_proxy_vol)
        historical_volatility_raw = pd.to_numeric(
            out.get(
                "historical_volatility",
                out.get("volatility", pd.Series(index=out.index, dtype=float)),
            ),
            errors="coerce",
        )
        out["normalised_historical_volatility"] = historical_volatility_raw * scale
        hist_vol = out["normalised_historical_volatility"]
        current_available = current_vol.notna()
        historical_available = hist_vol.notna()
        blended_raw, _vol_current_share, _vol_historical_share = _blend_available_components(
            current_vol,
            hist_vol,
            current_season_weight,
            past_season_weight,
        )
        volatility_source = pd.Series("fallback_floor", index=out.index, dtype=object)
        volatility_source = volatility_source.where(~(current_available & historical_available), "blended_current_historical")
        volatility_source = volatility_source.where(~(current_available & ~historical_available), "current_component")
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
    combined_outputs = pd.concat(
        [out_drivers.assign(_asset_type="driver"), out_constructors.assign(_asset_type="constructor")],
        ignore_index=True,
    )
    expected_source_counts = {
        str(key): int(value)
        for key, value in combined_outputs.get(
            "expected_points_source", pd.Series(dtype=object)
        ).value_counts(dropna=False).to_dict().items()
    }
    component_source_counts = {
        str(key): int(value)
        for key, value in combined_outputs.get(
            "current_component_source", pd.Series(dtype=object)
        ).value_counts(dropna=False).to_dict().items()
    }
    official_status_counts = {
        str(key): int(value)
        for key, value in combined_outputs.get(
            "official_observation_status", pd.Series(dtype=object)
        ).value_counts(dropna=False).to_dict().items()
    }
    diagnostics = {
        "observed_current_avg_points_per_race": float(current_avg) if pd.notna(current_avg) else None,
        "historical_avg_points_per_race": float(historical_avg) if pd.notna(historical_avg) else None,
        "historical_scale_factor": float(scale),
        "historical_scale_factor_clipped": bool(clipped),
        "observed_current_assets": int(
            pd.to_numeric(combined_observed.get("observed_current_avg_points"), errors="coerce").notna().sum()
        ),
        "observed_current_race_rows": observed_races,
        "official_selected_race_count": len(selected_races.included) if selected_races is not None else None,
        "official_valid_asset_race_count": int(
            pd.to_numeric(
                combined_outputs.get("official_valid_race_count", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "official_missing_asset_race_count": int(
            pd.to_numeric(
                combined_outputs.get("official_missing_race_count", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "official_source_failure_assets": int(
            combined_outputs.get(
                "official_has_source_failure", pd.Series(dtype=bool)
            ).fillna(False).astype(bool).sum()
        ),
        "official_observation_status_counts": official_status_counts,
        "current_component_source_counts": component_source_counts,
        "expected_points_source_counts": expected_source_counts,
        "safe_fallback_assets": int(expected_source_counts.get("safe_fallback", 0)),
        "volatility_source": "Selected official current-season volatility blended once with historical-only model volatility.",
        "volatility_source_counts": volatility_source_counts,
        "current_season_volatility_assets": current_vol_available,
        "historical_volatility_assets": historical_vol_available,
        "fallback_volatility_assets": int(volatility_source_counts.get("fallback_floor", 0)),
        "blended_current_historical_volatility_assets": int(volatility_source_counts.get("blended_current_historical", 0)),
        "current_only_volatility_assets": int(volatility_source_counts.get("current_component", 0)),
        "historical_only_volatility_assets": int(volatility_source_counts.get("historical_model_proxy", 0)),
        "volatility_floor_applied_assets": floor_applied,
        "driver_volatility_floor": float(driver_volatility_floor),
        "constructor_volatility_floor": float(constructor_volatility_floor),
        "configured_current_season_weight": float(current_season_weight),
        "configured_historical_weight": float(past_season_weight),
        "both_weights_zero_behavior": "equal shares when both components exist; sole available component otherwise",
        "effective_current_share_mean": float(
            pd.concat(
                [out_drivers["effective_current_share"], out_constructors["effective_current_share"]],
                ignore_index=True,
            ).mean()
        ),
        "effective_historical_share_mean": float(
            pd.concat(
                [out_drivers["effective_historical_share"], out_constructors["effective_historical_share"]],
                ignore_index=True,
            ).mean()
        ),
        "horizon_weight_sum": float(horizon_weight_sum) if horizon_weight_sum is not None else None,
        "blend_application_count": 1,
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
    holding_drivers: pd.DataFrame | None = None,
    holding_constructors: pd.DataFrame | None = None,
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

    drivers = drivers.copy(deep=True)
    constructors = constructors.copy(deep=True)
    holding_drivers = (
        holding_drivers.copy(deep=True)
        if isinstance(holding_drivers, pd.DataFrame)
        else drivers.copy(deep=True)
    )
    holding_constructors = (
        holding_constructors.copy(deep=True)
        if isinstance(holding_constructors, pd.DataFrame)
        else constructors.copy(deep=True)
    )
    drivers["_id"] = drivers["id"].astype(str)
    constructors["_id"] = constructors["id"].astype(str)
    holding_drivers["_id"] = holding_drivers["id"].astype(str)
    holding_constructors["_id"] = holding_constructors["id"].astype(str)
    holding_driver_ids = set(holding_drivers["_id"])
    holding_constructor_ids = set(holding_constructors["_id"])
    if not set(current_driver_ids) <= holding_driver_ids or not set(current_constructor_ids) <= holding_constructor_ids:
        _emit("ready", "Current team contains unresolved exact Fantasy asset IDs.", 1.0)
        return pd.DataFrame()
    combined_driver_assets = pd.concat(
        [holding_drivers, drivers], ignore_index=True, sort=False
    ).drop_duplicates("_id", keep="first")
    combined_constructor_assets = pd.concat(
        [holding_constructors, constructors], ignore_index=True, sort=False
    ).drop_duplicates("_id", keep="first")
    driver_summary = _asset_summary_map(combined_driver_assets)
    constructor_summary = _asset_summary_map(combined_constructor_assets)
    baseline = transfer_baseline(
        current_driver_ids,
        current_constructor_ids,
        holding_drivers,
        holding_constructors,
        budget,
        chip_mode=chip_mode,
    )
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
    for combined_frame in (combined_driver_assets, combined_constructor_assets):
        combined_frame["candidate_filter_score"] = combined_frame.apply(
            lambda row: transfer_candidate_filter_score(
                row,
                objective_mode=objective_mode,
                price_gain_weight=price_gain_weight,
            ),
            axis=1,
        )
    driver_filter_score_map.update(_numeric_map(combined_driver_assets, "candidate_filter_score"))
    constructor_filter_score_map.update(
        _numeric_map(combined_constructor_assets, "candidate_filter_score")
    )
    combined_assets = pd.concat(
        [combined_driver_assets, combined_constructor_assets], ignore_index=True, sort=False
    )

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

        new_d = combined_driver_assets[
            combined_driver_assets["_id"].isin(new_driver_ids)
        ].copy()
        new_c = combined_constructor_assets[
            combined_constructor_assets["_id"].isin(new_constructor_ids)
        ].copy()
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
                combined_driver_assets[
                    combined_driver_assets["_id"].isin([str(x) for x in d_out])
                ],
                combined_constructor_assets[
                    combined_constructor_assets["_id"].isin([str(x) for x in c_out])
                ],
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


def prospective_price_history(
    completed_observations: list[float | int | None]
    | tuple[float | int | None, ...]
    | pd.Series
    | None,
    predicted_next_points: float | int | None,
) -> dict[str, object]:
    """Build the exact-asset prospective rolling window for the next event."""
    values = list(completed_observations) if completed_observations is not None else []
    completed: list[float] = []
    for value in values:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            completed.append(float(numeric))
    prior = tuple(completed[-2:])
    predicted = pd.to_numeric(predicted_next_points, errors="coerce")
    prospective = prior + ((float(predicted),) if pd.notna(predicted) else ())
    projected_average = (
        float(sum(prospective) / len(prospective))
        if pd.notna(predicted) and prospective
        else pd.NA
    )
    return {
        "completed_observations_used": len(prior),
        "prior_observations": prior,
        "prospective_observations": prospective,
        "prospective_window_length": len(prior) + 1,
        "projected_rolling_average": projected_average,
        "price_history_mode": "established" if len(prior) >= 2 else "fresh",
    }


def required_next_points_for_history(
    current_price: float,
    target_avg_ppm: float,
    completed_observations: list[float | int | None]
    | tuple[float | int | None, ...]
    | pd.Series
    | None,
) -> float:
    """Invert the prospective rolling average for zero, one, or two priors."""
    history = prospective_price_history(completed_observations, 0.0)
    prior = history["prior_observations"]
    window_length = int(history["prospective_window_length"])
    return (
        window_length * float(current_price) * float(target_avg_ppm)
        - sum(float(value) for value in prior)
    )


def _row_price_history(row: pd.Series | dict, predicted_next_points: object) -> dict[str, object]:
    values = row if isinstance(row, (pd.Series, dict)) else {}
    return prospective_price_history(
        [values.get("recent_points_2ago"), values.get("recent_points_1ago")],
        predicted_next_points,
    )


def _price_eligibility(row: pd.Series | dict) -> tuple[bool, str]:
    values = row if isinstance(row, (pd.Series, dict)) else {}
    if "is_active" not in values:
        return True, "active"
    active = pd.to_numeric(values.get("is_active"), errors="coerce")
    is_active = pd.notna(active) and int(active) == 1
    return (
        (True, "active")
        if is_active
        else (False, "inactive_price_eligibility_unknown")
    )


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
    return required_next_points_for_history(
        current_price,
        target_avg_ppm,
        [points_race_minus_2, points_race_minus_1],
    )


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

    history = prospective_price_history([recent_2, recent_1], mean)
    prior_observations = history["prior_observations"]
    price_eligible, price_eligibility_status = _price_eligibility(row)
    thresholds = {}
    if pd.notna(price):
        thresholds = {
            "terrible_max": required_next_points_for_history(
                price, selected_rules.terrible_max, prior_observations
            ),
            "poor_max": required_next_points_for_history(
                price, selected_rules.poor_max, prior_observations
            ),
            "great_min": required_next_points_for_history(
                price, selected_rules.great_min, prior_observations
            ),
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
    if not price_eligible:
        probs = {key: pd.NA for key in probs}
    expected_gain = expected_price_gain_from_probabilities(probs, price, selected_rules, bounds)
    projected_rolling_average = history["projected_rolling_average"]
    projected_avg_ppm = avg_ppm_from_points(projected_rolling_average, price)
    projected_tier = "Missing"
    if price_eligible and pd.notna(projected_avg_ppm):
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
        "price_history_observations": int(history["completed_observations_used"]),
        "price_history_prior_observations": history["prior_observations"],
        "price_history_prospective_observations": history["prospective_observations"],
        "price_history_window_length": int(history["prospective_window_length"]),
        "projected_rolling_average": projected_rolling_average,
        "price_history_mode": (
            history["price_history_mode"] if price_eligible else "inactive_unknown"
        ),
        "price_eligibility_status": price_eligibility_status,
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
    history_rows = out.apply(
        lambda row: _row_price_history(row, row["price_change_predicted_next"]),
        axis=1,
    )
    history_frame = pd.DataFrame(
        history_rows.tolist(),
        index=out.index,
        columns=[
            "completed_observations_used",
            "prior_observations",
            "prospective_observations",
            "prospective_window_length",
            "projected_rolling_average",
            "price_history_mode",
        ],
    )
    out["price_history_observations"] = history_frame["completed_observations_used"].astype(int)
    out["price_history_prior_observations"] = history_frame["prior_observations"]
    out["price_history_prospective_observations"] = history_frame["prospective_observations"]
    out["price_history_window_length"] = history_frame["prospective_window_length"].astype(int)
    out["projected_rolling_average"] = history_frame["projected_rolling_average"]
    out["price_history_mode"] = history_frame["price_history_mode"]
    eligibility = out.apply(_price_eligibility, axis=1)
    out["price_eligible"] = eligibility.apply(lambda value: value[0])
    out["price_eligibility_status"] = eligibility.apply(lambda value: value[1])
    out.loc[~out["price_eligible"], "price_history_mode"] = "inactive_unknown"
    out["price_display_status"] = out["price_history_mode"].map(
        {
            "established": "Active",
            "fresh": "Fresh history",
            "inactive_unknown": "Inactive",
        }
    )
    out["avg_ppm"] = out.apply(
        lambda row: avg_ppm_from_points(row["projected_rolling_average"], row["price"]),
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
    unavailable = ~out["price_eligible"]
    for column in (
        "raw_price_change",
        "projected_price",
        "effective_price_change_after_floor_ceiling",
        "expected_price_change",
        "expected_price_gain_per_million",
        "risk_adjusted_price_gain",
    ):
        out.loc[unavailable, column] = pd.NA
    out.loc[unavailable, "price_change_tier"] = "Unknown"
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

    def required_for_row(row: pd.Series, target: float) -> float:
        return required_next_points_for_history(
            row["price"],
            target,
            row["price_history_prior_observations"],
        )

    out["required_terrible_max"] = out.apply(
        lambda row: required_for_row(
            row,
            choose_price_change_rules(
                row["price"], rules, expensive_rules, expensive_price_min
            ).terrible_max,
        ),
        axis=1,
    )
    out["required_poor_min"] = out.apply(
        lambda row: required_for_row(
            row,
            choose_price_change_rules(
                row["price"], rules, expensive_rules, expensive_price_min
            ).poor_min,
        ),
        axis=1,
    )
    out["required_good_min"] = out.apply(
        lambda row: required_for_row(
            row,
            choose_price_change_rules(
                row["price"], rules, expensive_rules, expensive_price_min
            ).good_min,
        ),
        axis=1,
    )
    out["required_great_min"] = out.apply(
        lambda row: required_for_row(
            row,
            choose_price_change_rules(
                row["price"], rules, expensive_rules, expensive_price_min
            ).great_min,
        ),
        axis=1,
    )
    threshold_columns = [
        "required_terrible_max",
        "required_poor_min",
        "required_good_min",
        "required_great_min",
    ]
    out.loc[~out["price_eligible"], threshold_columns] = pd.NA

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
    table["_price_abbrev"] = table[abbrev_col]
    cols = [
        "id",
        "_price_abbrev",
        "name",
        "team",
        "price",
        "price_display_status",
        "price_history_observations",
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
            "id": "Fantasy asset ID",
            "_price_abbrev": "Abbrev",
            "name": "Name",
            "team": "Team",
            "price": "Price",
            "price_display_status": "Status",
            "price_history_observations": "History used",
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
    table["_price_abbrev"] = table[abbrev_col]

    cols = [
        "id",
        "_price_abbrev",
        "name",
        "team",
        "price",
        "price_display_status",
        "price_history_observations",
        "price_change_predicted_next",
        "expected_price_gain",
    ]
    out = table[[col for col in cols if col in table.columns]].copy()
    out.rename(
        columns={
            "id": "Fantasy asset ID",
            "_price_abbrev": "Abbrev",
            "name": "Name",
            "team": "Team",
            "price": "Price",
            "price_display_status": "Status",
            "price_history_observations": "History used",
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
    table["_price_abbrev"] = table[abbrev_col]
    cols = [
        "id",
        "_price_abbrev",
        "name",
        "team",
        "price",
        "human_driver_id",
        "is_active",
        "price_history_observations",
        "price_history_prior_observations",
        "projected_rolling_average",
        "price_history_mode",
        "price_eligibility_status",
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
            "id": "Fantasy asset ID",
            "_price_abbrev": "Abbrev",
            "name": "Name",
            "team": "Team",
            "price": "Price",
            "human_driver_id": "Human identity",
            "is_active": "Active",
            "price_history_observations": "History used",
            "price_history_prior_observations": "Prior observations",
            "projected_rolling_average": "Projected rolling average",
            "price_history_mode": "History mode",
            "price_eligibility_status": "Price eligibility",
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
    excluded_team_combinations: list[tuple[list[str], list[str]]] | None = None,
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
        excluded_team_combinations=excluded_team_combinations,
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
    if not isinstance(driver_ids, (list, tuple)) or not isinstance(constructor_ids, (list, tuple)):
        return {
            "valid": False,
            "errors": ["Current team selections must be lists of exact Fantasy asset IDs."],
            "warnings": [],
            "total_cost": 0.0,
            "projected_points": 0.0,
            "selected_drivers": drivers.iloc[0:0].copy(deep=True),
            "selected_constructors": constructors.iloc[0:0].copy(deep=True),
            "unknown_driver_ids": [],
            "unknown_constructor_ids": [],
            "inactive_driver_ids": [],
            "inactive_constructor_ids": [],
            "missing_price_driver_ids": [],
            "missing_price_constructor_ids": [],
            "valuation_complete": False,
        }
    driver_ids = [str(x) for x in driver_ids]
    constructor_ids = [str(x) for x in constructor_ids]
    available_driver_ids = set(drivers["id"].astype(str))
    available_constructor_ids = set(constructors["id"].astype(str))

    selected_drivers = drivers[drivers["id"].astype(str).isin(driver_ids)].copy()
    selected_constructors = constructors[constructors["id"].astype(str).isin(constructor_ids)].copy()
    driver_prices = pd.to_numeric(
        selected_drivers.get("price", pd.Series(index=selected_drivers.index, dtype=float)),
        errors="coerce",
    )
    constructor_prices = pd.to_numeric(
        selected_constructors.get("price", pd.Series(index=selected_constructors.index, dtype=float)),
        errors="coerce",
    )
    total_cost = float(driver_prices.fillna(0.0).sum() + constructor_prices.fillna(0.0).sum())
    projected_points = float(
        pd.to_numeric(selected_drivers.get("exp_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        + pd.to_numeric(selected_constructors.get("exp_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    )

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
    selected_driver_ids = selected_drivers["id"].astype(str)
    selected_constructor_ids = selected_constructors["id"].astype(str)
    missing_price_drivers = sorted(selected_driver_ids[driver_prices.isna()].tolist())
    missing_price_constructors = sorted(selected_constructor_ids[constructor_prices.isna()].tolist())
    if missing_price_drivers:
        errors.append(f"Missing current prices for driver IDs: {missing_price_drivers}")
    if missing_price_constructors:
        errors.append(f"Missing current prices for constructor IDs: {missing_price_constructors}")
    driver_active = pd.to_numeric(
        selected_drivers.get("is_active", pd.Series(1, index=selected_drivers.index)),
        errors="coerce",
    ).fillna(0).eq(1)
    constructor_active = pd.to_numeric(
        selected_constructors.get("is_active", pd.Series(1, index=selected_constructors.index)),
        errors="coerce",
    ).fillna(0).eq(1)
    inactive_drivers = sorted(selected_driver_ids[~driver_active].tolist())
    inactive_constructors = sorted(selected_constructor_ids[~constructor_active].tolist())
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
        "unknown_driver_ids": missing_drivers,
        "unknown_constructor_ids": missing_constructors,
        "inactive_driver_ids": inactive_drivers,
        "inactive_constructor_ids": inactive_constructors,
        "missing_price_driver_ids": missing_price_drivers,
        "missing_price_constructor_ids": missing_price_constructors,
        "valuation_complete": not missing_drivers
        and not missing_constructors
        and not missing_price_drivers
        and not missing_price_constructors,
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


def uploaded_file_hash(contents: bytes) -> str:
    if not isinstance(contents, bytes):
        raise TypeError("Uploaded current-team contents must be bytes.")
    return hashlib.sha256(contents).hexdigest()


def should_process_upload(new_hash: str, previous_attempt_hash: str | None) -> bool:
    return bool(new_hash) and str(new_hash) != str(previous_attempt_hash or "")


def current_team_selection_signature(
    driver_ids: list[str],
    constructor_ids: list[str],
    bank: float,
) -> tuple[tuple[str, ...], tuple[str, ...], float]:
    return (
        tuple(sorted(str(asset_id) for asset_id in driver_ids)),
        tuple(sorted(str(asset_id) for asset_id in constructor_ids)),
        round(float(bank), 4),
    )


def prepare_uploaded_team_import(
    contents: bytes,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
) -> dict:
    """Parse and validate a complete upload without mutating application state."""
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid current_team.json: file must be UTF-8 encoded.") from exc

    payload = load_current_team_json_text(text)
    summary = current_team_upload_summary(
        payload,
        available_driver_ids=set(drivers["id"].astype(str)),
        available_constructor_ids=set(constructors["id"].astype(str)),
    )

    bank = pd.to_numeric(summary.get("bank"), errors="coerce")
    if pd.isna(bank) or not math.isfinite(float(bank)) or float(bank) < 0:
        raise ValueError("current_team.json bank must be a finite non-negative number.")
    free_transfers = int(summary.get("free_transfers", 2))
    if free_transfers < 0 or free_transfers > 10:
        raise ValueError("current_team.json free_transfers must be between 0 and 10.")

    selected_drivers = drivers[drivers["id"].astype(str).isin(summary["drivers"])].copy()
    selected_constructors = constructors[constructors["id"].astype(str).isin(summary["constructors"])].copy()
    team_cost = float(
        pd.to_numeric(selected_drivers.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        + pd.to_numeric(selected_constructors.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    )
    budget_suggestion = current_team_budget_from_selection(
        selected_drivers,
        selected_constructors,
        bank=float(bank),
    )
    validation = validate_current_team(
        summary["drivers"],
        summary["constructors"],
        drivers,
        constructors,
        budget=budget_suggestion,
    )
    errors = list(validation["errors"])
    if summary["missing_drivers"]:
        errors.append(f"Unknown driver IDs: {summary['missing_drivers']}")
    if summary["missing_constructors"]:
        errors.append(f"Unknown constructor IDs: {summary['missing_constructors']}")
    if errors:
        raise ValueError("Invalid current team: " + " ".join(errors))

    normalized_summary = {
        **summary,
        "bank": float(bank),
        "free_transfers": free_transfers,
    }
    return {
        "summary": normalized_summary,
        "team_cost": team_cost,
        "budget_suggestion": float(budget_suggestion),
        "selection_signature": current_team_selection_signature(
            normalized_summary["drivers"],
            normalized_summary["constructors"],
            float(bank),
        ),
    }


def build_import_state_updates(prepared_import: dict) -> dict:
    """Return only current-team state updates; optimiser budget ownership is separate."""
    summary = prepared_import["summary"]
    return {
        "current_team_driver_ids": list(summary["drivers"]),
        "current_team_constructor_ids": list(summary["constructors"]),
        "current_team_bank": float(summary["bank"]),
        "current_team_free_transfers": int(summary["free_transfers"]),
        "current_team_budget": float(prepared_import["budget_suggestion"]),
        "current_team_budget_user_overridden": False,
        "current_team_budget_source": "imported",
        "budget_auto_signature": prepared_import["selection_signature"],
        "budget_init_team_cost": float(prepared_import["team_cost"]),
        "budget_init_bank": float(summary["bank"]),
        "imported_budget_suggestion": float(prepared_import["budget_suggestion"]),
        "imported_budget_selection_signature": prepared_import["selection_signature"],
        "imported_budget_driver_ids": list(summary["drivers"]),
        "imported_budget_constructor_ids": list(summary["constructors"]),
        "imported_budget_bank": float(summary["bank"]),
        "imported_budget_suggestion_status": "available",
        "imported_budget_missing_driver_ids": [],
        "imported_budget_missing_constructor_ids": [],
        "imported_budget_missing_price_driver_ids": [],
        "imported_budget_missing_price_constructor_ids": [],
    }


def reconcile_imported_budget_suggestion(
    driver_ids: list[str],
    constructor_ids: list[str],
    bank: float,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
) -> dict:
    """Reprice a previously parsed imported team against the current roster."""
    normalized_driver_ids = [str(asset_id) for asset_id in driver_ids]
    normalized_constructor_ids = [str(asset_id) for asset_id in constructor_ids]
    available_driver_ids = set(drivers["id"].astype(str))
    available_constructor_ids = set(constructors["id"].astype(str))
    missing_driver_ids = [asset_id for asset_id in normalized_driver_ids if asset_id not in available_driver_ids]
    missing_constructor_ids = [
        asset_id for asset_id in normalized_constructor_ids if asset_id not in available_constructor_ids
    ]
    if missing_driver_ids or missing_constructor_ids:
        return {
            "status": "incomplete",
            "suggestion": None,
            "missing_driver_ids": missing_driver_ids,
            "missing_constructor_ids": missing_constructor_ids,
            "missing_price_driver_ids": [],
            "missing_price_constructor_ids": [],
        }
    selected_drivers = drivers[drivers["id"].astype(str).isin(normalized_driver_ids)]
    selected_constructors = constructors[constructors["id"].astype(str).isin(normalized_constructor_ids)]
    driver_prices = pd.to_numeric(
        selected_drivers.get("price", pd.Series(index=selected_drivers.index, dtype=float)),
        errors="coerce",
    )
    constructor_prices = pd.to_numeric(
        selected_constructors.get("price", pd.Series(index=selected_constructors.index, dtype=float)),
        errors="coerce",
    )
    missing_price_driver_ids = sorted(
        selected_drivers.loc[driver_prices.isna(), "id"].astype(str).tolist()
    )
    missing_price_constructor_ids = sorted(
        selected_constructors.loc[constructor_prices.isna(), "id"].astype(str).tolist()
    )
    if missing_price_driver_ids or missing_price_constructor_ids:
        return {
            "status": "incomplete",
            "suggestion": None,
            "missing_driver_ids": [],
            "missing_constructor_ids": [],
            "missing_price_driver_ids": missing_price_driver_ids,
            "missing_price_constructor_ids": missing_price_constructor_ids,
        }
    suggestion = current_team_budget_from_selection(
        selected_drivers,
        selected_constructors,
        bank=float(bank),
    )
    return {
        "status": "available",
        "suggestion": float(suggestion),
        "missing_driver_ids": [],
        "missing_constructor_ids": [],
        "missing_price_driver_ids": [],
        "missing_price_constructor_ids": [],
    }


def current_team_upload_transition(
    contents: bytes,
    previous_attempt_hash: str | None,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
) -> dict:
    """Describe one upload attempt without mutating the supplied application state."""
    digest = uploaded_file_hash(contents)
    if not should_process_upload(digest, previous_attempt_hash):
        return {
            "attempted": False,
            "upload_hash": digest,
            "status": None,
            "error": None,
            "state_updates": {},
        }
    try:
        prepared = prepare_uploaded_team_import(contents, drivers, constructors)
    except Exception as exc:
        return {
            "attempted": True,
            "upload_hash": digest,
            "status": "error",
            "error": str(exc),
            "state_updates": {},
        }
    return {
        "attempted": True,
        "upload_hash": digest,
        "status": "success",
        "error": None,
        "state_updates": build_import_state_updates(prepared),
    }


def optimizer_budget_state_updates(
    value: float,
    source: str,
    accepted_import_hash: str | None = None,
) -> dict:
    """Build the explicit user-owned optimiser-budget transition."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not math.isfinite(float(numeric)) or float(numeric) < 0:
        raise ValueError("Optimiser budget must be a finite non-negative number.")
    if source not in {"manual", "imported_accepted"}:
        raise ValueError("Optimiser budget source must be manual or imported_accepted.")
    return {
        "optimizer_budget": float(numeric),
        "optimizer_budget_source": source,
        "optimizer_budget_accepted_import_hash": accepted_import_hash if source == "imported_accepted" else None,
        "budget_user_overridden": True,
        "budget_init_mode": source,
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


def fantasy_asset_card_html(
    asset: dict | pd.Series,
    boosted_token: str | None = None,
    asset_label: str = "Asset",
) -> str:
    row = asset if isinstance(asset, pd.Series) else pd.Series(asset)
    name = str(row.get("name", ""))
    team_name = str(row.get("team", row.get("name", "")))
    colour_value = row.get("team_colour")
    colour = (
        team_colour(team_name)
        if colour_value is None or colour_value is pd.NA or pd.isna(colour_value)
        else str(colour_value)
    )
    identity_values = row.to_dict()
    identity_values.update(
        {
            "asset_type": str(asset_label).casefold(),
            "full_name": name,
            "team_name": team_name,
            "team_colour": colour,
        }
    )
    payload = compact_asset_payload(
        identity_values,
        asset_type=asset_label,
        marker=boosted_token,
    )
    identity_html = compact_asset_identity_html(identity_values)
    badge = (
        f'<span class="f1-boost" aria-label="{html.escape(payload["marker"])} points multiplier">'
        f'{html.escape(payload["marker"])}</span>'
        if payload["marker"]
        else ""
    )
    holding_status_value = row.get("holding_status")
    holding_status = (
        ""
        if holding_status_value is None or holding_status_value is pd.NA or pd.isna(holding_status_value)
        else str(holding_status_value).strip()
    )
    availability_badge = (
        '<span class="f1-availability-muted">Inactive</span>'
        if holding_status.casefold() == "inactive"
        else ""
    )
    return (
        '<div class="f1-driver-card" style="--team-color:{colour}" role="group" '
        'title="{title}" aria-label="{aria}">'
        '<div class="f1-card-top">'
        '<div class="f1-card-identity">{identity}{badge}{availability_badge}</div>'
        "</div>"
        '<div class="f1-card-middle">'
        '<span class="f1-card-price">{price}</span>'
        '<span class="f1-card-gain {gain_class}">{gain}</span>'
        "</div>"
        '<div class="f1-card-points">{expected}</div>'
        "</div>".format(
            colour=html.escape(colour, quote=True),
            title=html.escape(payload["identity"], quote=True),
            aria=html.escape(
                f'{payload["identity"]}: {payload["price"]}, {payload["gain"]} expected gain, {payload["points"]}',
                quote=True,
            ),
            identity=identity_html,
            badge=badge,
            availability_badge=availability_badge,
            price=html.escape(payload["price"]),
            expected=html.escape(payload["points"]),
            gain=html.escape(payload["gain"]),
            gain_class=payload["gain_class"],
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
    grid_class = "f1-constructor-grid" if str(asset_label).casefold().startswith("constructor") else "f1-driver-grid"
    return f'<div class="f1-card-grid {grid_class}">' + "".join(cards) + "</div>"


def ranked_team_component_html(
    *,
    rank: int,
    summary: Mapping[str, Any],
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    boosted_driver: str | None = None,
    triple_driver: str | None = None,
    limitless: bool = False,
) -> str:
    """Render every optimiser result with the same compact information contract."""
    stats = team_summary_payload(
        total_cost=summary.get("Total cost"),
        budget=None if limitless else (
            pd.to_numeric(summary.get("Total cost"), errors="coerce")
            + pd.to_numeric(summary.get("Remaining budget"), errors="coerce")
        ),
        expected_gain=summary.get("Expected price gain"),
        expected_points=summary.get("Expected points"),
        limitless=limitless,
    )
    stat_html = team_summary_html(stats)
    driver_cards = fantasy_card_grid_html(
        drivers,
        boosted_driver=boosted_driver,
        triple_driver=triple_driver,
        asset_label="Driver",
    )
    constructor_cards = fantasy_card_grid_html(constructors, asset_label="Constructor")
    return (
        '<article class="f1-ranked-team" aria-label="Ranked optimiser team {rank}">'
        '<div class="f1-team-header"><span class="f1-team-rank">{rank}</span>{stats}</div>'
        '<div class="f1-team-assets">{drivers}{constructors}</div>'
        "</article>"
    ).format(
        rank=max(1, int(rank)),
        stats=stat_html,
        drivers=driver_cards,
        constructors=constructor_cards,
    )
