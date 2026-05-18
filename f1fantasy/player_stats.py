from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
import time
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://fantasy.formula1.com/feeds/popup"
PLAYERSTATS_ENDPOINT_PATTERN = BASE_URL + "/playerstats_{player_id}.json"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _playerstats_url(player_id: int) -> str:
    return PLAYERSTATS_ENDPOINT_PATTERN.format(player_id=int(player_id))


def _stats_total(stats: list[dict[str, Any]] | None, event_name: str = "Total") -> float | None:
    for item in stats or []:
        if str(item.get("Event", "")).strip().lower() == event_name.lower():
            return pd.to_numeric(item.get("Value"), errors="coerce")
    return None


def _component_total(stats: list[dict[str, Any]] | None, contains: str) -> float:
    total = 0.0
    found = False
    needle = contains.lower()
    for item in stats or []:
        event = str(item.get("Event", "")).strip().lower()
        if needle in event:
            value = pd.to_numeric(item.get("Value"), errors="coerce")
            if pd.notna(value):
                total += float(value)
                found = True
    return total if found else 0.0


def _int_or_zero(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else 0


def _parse_session_start(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


@lru_cache(maxsize=256)
def fetch_player_stats(player_id: int) -> dict:
    """Fetch the public F1 Fantasy popup stats payload for a driver/constructor."""
    response = requests.get(
        _playerstats_url(player_id),
        timeout=20,
        headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"},
    )
    response.raise_for_status()
    return response.json()


def parse_player_race_points(payload: dict, player_id: int | None = None) -> pd.DataFrame:
    """Parse playerstats JSON into one row per completed race/gameday."""
    value = payload.get("Value", {}) if isinstance(payload, dict) else {}
    player_id = int(player_id if player_id is not None else value.get("PlayerId"))
    player_skill = value.get("PlayerSkill")
    asset_type = "constructor" if str(player_skill) == "2" else "driver"

    match_by_gameday = {
        item.get("GamedayId"): item
        for item in value.get("MatchWiseStats", []) or []
        if isinstance(item, dict)
    }

    rows: list[dict[str, Any]] = []
    for gameday in value.get("GamedayWiseStats", []) or []:
        if not isinstance(gameday, dict):
            continue
        gameday_id = gameday.get("GamedayId")
        total = _stats_total(gameday.get("StatsWise"))
        price = pd.to_numeric(gameday.get("PlayerValue"), errors="coerce")
        old_price = pd.to_numeric(gameday.get("OldPlayerValue"), errors="coerce")

        match = match_by_gameday.get(gameday_id, {})
        sessions = match.get("RaceDayWise", []) if isinstance(match, dict) else []
        first_session = sessions[0] if sessions else {}

        session_totals: dict[str, float] = {}
        overtake_points = 0.0
        for session in sessions:
            session_type = str(session.get("SessionType") or session.get("SessionName") or "").strip().lower()
            session_total = _stats_total(session.get("StatsWise"))
            if session_type and session_total is not None and pd.notna(session_total):
                if "sprint" in session_type:
                    session_totals["sprint"] = session_totals.get("sprint", 0.0) + float(session_total)
                elif "qualifying" in session_type:
                    session_totals["qualifying"] = session_totals.get("qualifying", 0.0) + float(session_total)
                elif "race" in session_type:
                    session_totals["race"] = session_totals.get("race", 0.0) + float(session_total)
            overtake_points += _component_total(session.get("StatsWise"), "overtake")

        row = {
            "PlayerId": player_id,
            "asset_type": asset_type,
            "gameday_id": pd.to_numeric(gameday_id, errors="coerce"),
            "round": pd.to_numeric(first_session.get("MeetingNumber"), errors="coerce"),
            "race_name": first_session.get("MeetingName"),
            "race_id": first_session.get("RaceDayId"),
            "season": pd.to_numeric(first_session.get("Season"), errors="coerce"),
            "fantasy_points": float(total) if total is not None and pd.notna(total) else pd.NA,
            "qualifying_points": session_totals.get("qualifying", pd.NA),
            "race_points": session_totals.get("race", pd.NA),
            "sprint_points": session_totals.get("sprint", pd.NA),
            "overtaking_points": overtake_points,
            "price": float(price) if pd.notna(price) else pd.NA,
            "old_price": float(old_price) if pd.notna(old_price) else pd.NA,
            "price_change": float(price - old_price) if pd.notna(price) and pd.notna(old_price) else pd.NA,
            "is_played": _int_or_zero(gameday.get("IsPlayed")),
            "is_active": _int_or_zero(gameday.get("IsActive")),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "PlayerId",
                "asset_type",
                "gameday_id",
                "round",
                "race_name",
                "fantasy_points",
                "qualifying_points",
                "race_points",
                "sprint_points",
                "overtaking_points",
                "price",
                "price_change",
                "is_played",
            ]
        )
    return out.sort_values(["round", "gameday_id"], na_position="last").reset_index(drop=True)


def fetch_recent_points_for_roster(roster_df: pd.DataFrame, asset_type: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fetch playerstats for roster rows and return recent Race -2/Race -1 values."""
    rows: list[dict[str, Any]] = []
    race_rows: list[pd.DataFrame] = []
    failures: list[str] = []
    id_col = "id" if "id" in roster_df.columns else "PlayerId"

    for row in roster_df.itertuples(index=False):
        player_id = getattr(row, id_col)
        name = getattr(row, "name", "")
        try:
            payload = fetch_player_stats(int(player_id))
            parsed = parse_player_race_points(payload, int(player_id))
        except Exception as exc:
            failures.append(f"{name or player_id}: {exc}")
            rows.append(
                {
                    id_col: player_id,
                    "recent_points_2ago": pd.NA,
                    "recent_points_1ago": pd.NA,
                    "recent_points_available": 0,
                    "recent_points_source": "playerstats_failed",
                }
            )
            continue

        if asset_type:
            parsed["asset_type"] = asset_type
        if "name" in roster_df.columns:
            parsed["name"] = name
        race_rows.append(parsed)

        completed = parsed[(parsed["is_played"] == 1) & pd.to_numeric(parsed["fantasy_points"], errors="coerce").notna()]
        completed = completed.sort_values(["round", "gameday_id"], na_position="last").tail(2)
        points = pd.to_numeric(completed["fantasy_points"], errors="coerce").tolist()
        races = completed[["round", "race_name"]].to_dict("records")
        rows.append(
            {
                id_col: player_id,
                "recent_points_2ago": float(points[-2]) if len(points) >= 2 else pd.NA,
                "recent_points_1ago": float(points[-1]) if len(points) >= 1 else pd.NA,
                "recent_points_available": int(len(points)),
                "recent_points_source": "playerstats" if len(points) >= 2 else "playerstats_incomplete",
                "recent_points_races": races,
            }
        )
        time.sleep(0.03)

    recent = pd.DataFrame(rows)
    all_races = pd.concat(race_rows, ignore_index=True) if race_rows else pd.DataFrame()
    complete_assets = int((recent.get("recent_points_available", pd.Series(dtype=int)) >= 2).sum()) if not recent.empty else 0
    diagnostics = {
        "playerstats_endpoint_pattern": PLAYERSTATS_ENDPOINT_PATTERN,
        "playerstats_assets_loaded": complete_assets,
        "playerstats_assets_failed": len(failures),
        "playerstats_failures": failures[:10],
    }
    return recent, all_races, diagnostics


def latest_two_races(race_points: pd.DataFrame) -> list[dict[str, Any]]:
    if race_points.empty or "round" not in race_points.columns:
        return []
    data = race_points.copy()
    if "is_played" in data.columns:
        data = data[data["is_played"] == 1]
    if "fantasy_points" in data.columns:
        data = data[pd.to_numeric(data["fantasy_points"], errors="coerce").notna()]
    rounds = (
        data[["round", "race_name"]]
        .dropna(subset=["round"])
        .drop_duplicates()
        .sort_values("round")
        .tail(2)
    )
    return rounds.to_dict("records")


def parse_team_lock_deadline_from_payload(payload: dict) -> dict[str, Any]:
    """Extract next team-lock style session from official playerstats payload."""
    value = payload.get("Value", {}) if isinstance(payload, dict) else {}
    now = datetime.now(UTC)
    sessions: list[dict[str, Any]] = []
    for container_key in ["FixtureWiseStats", "MatchWiseStats"]:
        for container in value.get(container_key, []) or []:
            if not isinstance(container, dict):
                continue
            gameday_id = container.get("GamedayId")
            for race_day in container.get("RaceDayWise", []) or []:
                if not isinstance(race_day, dict):
                    continue
                start_raw = race_day.get("SessionStartDate")
                start_utc = _parse_session_start(start_raw)
                if start_utc is None:
                    continue
                session_type = str(race_day.get("SessionType") or race_day.get("SessionName") or "").strip()
                sessions.append(
                    {
                        "container": container_key,
                        "gameday_id": gameday_id,
                        "meeting_name": race_day.get("MeetingName"),
                        "session_type": session_type,
                        "match_status": str(race_day.get("MatchStatus", "")),
                        "is_played": _int_or_zero(race_day.get("IsPlayed")),
                        "session_start_raw": start_raw,
                        "session_start_utc": start_utc,
                    }
                )
    if not sessions:
        return {
            "team_lock_deadline_utc": None,
            "team_lock_deadline_source": "unavailable",
            "team_lock_deadline_raw_field": None,
            "team_lock_deadline_raw_value": None,
            "team_lock_meeting_name": None,
            "team_lock_session_type": None,
            "team_lock_timezone_assumption": "SessionStartDate parsed as ISO-8601 when available.",
        }

    future = [row for row in sessions if row["session_start_utc"] >= now]
    pool = future if future else sessions
    status_one = [row for row in pool if row.get("match_status") == "1"]
    candidates = status_one if status_one else pool
    chosen = min(candidates, key=lambda row: row["session_start_utc"])
    return {
        "team_lock_deadline_utc": chosen["session_start_utc"].isoformat(),
        "team_lock_deadline_source": "official_feed_playerstats_session_start",
        "team_lock_deadline_raw_field": f"{chosen['container']}.RaceDayWise.SessionStartDate",
        "team_lock_deadline_raw_value": chosen.get("session_start_raw"),
        "team_lock_meeting_name": chosen.get("meeting_name"),
        "team_lock_session_type": chosen.get("session_type"),
        "team_lock_timezone_assumption": "SessionStartDate parsed as ISO-8601 when available.",
    }


def fetch_team_lock_deadline_from_playerstats(player_id: int) -> dict[str, Any]:
    payload = fetch_player_stats(int(player_id))
    return parse_team_lock_deadline_from_payload(payload)
