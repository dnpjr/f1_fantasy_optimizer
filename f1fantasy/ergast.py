from __future__ import annotations
from datetime import UTC, datetime
import json
from pathlib import Path
import pandas as pd
import requests

# Jolpica provides Ergast-compatible endpoints (Ergast has been deprecated).
ERGAST = "https://api.jolpi.ca/ergast/f1"
ERGAST_TIMEOUT_SECONDS = 15
CURRENT_SEASON_CACHE_TTL_SECONDS = 5 * 60
CURRENT_SEASON_SCHEDULE_CACHE_TTL_SECONDS = 6 * 60 * 60

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _is_dnf(status: str) -> int:
    """Heuristic: 'Finished' or '+X Laps' are classified finishes; everything else counts as DNF/NC."""
    if not isinstance(status, str):
        return 1
    s = status.strip()
    if s == "Finished":
        return 0
    if "Lap" in s and "+" in s:
        return 0
    return 1

def _get_json(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=ERGAST_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()

def _cache_metadata_file(cache_file: Path) -> Path:
    return cache_file.with_suffix(cache_file.suffix + ".meta.json")


def _try_read_cache(
    cache_file: Path,
    *,
    year: int | None = None,
    source_kind: str = "session",
    now_utc: datetime | None = None,
) -> pd.DataFrame | None:
    """Return a usable cache, applying a bounded TTL to the active season."""
    if not cache_file.exists():
        return None
    try:
        df = pd.read_csv(cache_file)
        if df.shape[1] == 0:
            raise ValueError("empty cache")
        current_year = (now_utc or datetime.now(UTC)).year
        if year is not None and int(year) >= int(current_year):
            ttl = (
                CURRENT_SEASON_SCHEDULE_CACHE_TTL_SECONDS
                if source_kind == "schedule"
                else CURRENT_SEASON_CACHE_TTL_SECONDS
            )
            metadata_file = _cache_metadata_file(cache_file)
            fetched_at: datetime | None = None
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                raw = str(metadata.get("fetched_at_utc") or "")
                if raw:
                    fetched_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    fetched_at = fetched_at.astimezone(UTC) if fetched_at.tzinfo else fetched_at.replace(tzinfo=UTC)
            if fetched_at is None:
                fetched_at = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC)
            if ((now_utc or datetime.now(UTC)) - fetched_at).total_seconds() > ttl:
                return None
        return df
    except Exception:
        try:
            cache_file.unlink()
        except Exception:
            pass
        return None


def _write_cache(
    frame: pd.DataFrame,
    cache_file: Path,
    *,
    year: int,
    source_kind: str,
) -> None:
    frame.to_csv(cache_file, index=False)
    event_keys: list[list[int]] = []
    event_statuses: list[dict[str, object]] = []
    if {"season", "round"}.issubset(frame.columns):
        keys = frame[["season", "round"]].dropna().drop_duplicates()
        event_keys = [[int(row.season), int(row.round)] for row in keys.itertuples(index=False)]
        for (season, round_no), rows in frame.groupby(["season", "round"], dropna=True):
            status = "available_unverified"
            if "status" in rows.columns and rows["status"].fillna("").astype(str).str.contains(
                r"\b(?:running|live|provisional|pending|in progress|not started|under investigation)\b",
                case=False,
                regex=True,
            ).any():
                status = "provisional"
            elif source_kind in {"grand_prix", "grand_prix_qualifying", "sprint"}:
                participants = (
                    int(rows["driverId"].nunique()) if "driverId" in rows.columns else len(rows)
                )
                status = "complete_candidate" if participants >= 20 else "partial_candidate"
            event_statuses.append(
                {
                    "season": int(season),
                    "round": int(round_no),
                    "observed_rows": int(len(rows)),
                    "status": status,
                }
            )
    metadata = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "season": int(year),
        "source_kind": source_kind,
        "status": "expected_empty" if frame.empty else "available_unverified",
        "event_keys": event_keys,
        "event_statuses": event_statuses,
        "participant_count_fallback": 20,
    }
    _cache_metadata_file(cache_file).write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )

def fetch_season_results(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Race results (one row per driver per race). Cached to data/cache/results_<year>.csv"""
    cache_file = CACHE_DIR / f"results_{year}.csv"
    if not force_refresh:
        cached = _try_read_cache(cache_file, year=year, source_kind="grand_prix")
        if cached is not None:
            return cached

    data = _get_json(f"{ERGAST}/{year}/results.json", params={"limit": 10000})
    races = data["MRData"]["RaceTable"]["Races"]
    rows: list[dict] = []
    for race in races:
        circuit = race["Circuit"]["circuitName"]
        round_no = int(race["round"])
        race_name = race.get("raceName", "")
        date = race.get("date", "")
        for res in race["Results"]:
            drv = res["Driver"]
            con = res["Constructor"]
            status = res.get("status", "")
            rows.append({
                "season": year,
                "round": round_no,
                "raceName": race_name,
                "date": date,
                "circuitName": circuit,
                "driverId": drv.get("driverId", ""),
                "driver": f'{drv.get("givenName","")} {drv.get("familyName","")}'.strip(),
                "constructorId": con.get("constructorId", ""),
                "constructor": con.get("name", ""),
                "grid": int(res.get("grid", 0) or 0),
                "position": int(res.get("position", 0) or 0),
                "status": status,
                "fastestLapRank": int(res.get("FastestLap", {}).get("rank", 0) or 0),
                "is_dnf": _is_dnf(status),
            })

    df = pd.DataFrame(rows, columns=[
        "season", "round", "raceName", "date", "circuitName", "driverId",
        "driver", "constructorId", "constructor", "grid", "position",
        "status", "fastestLapRank", "is_dnf",
    ])
    _write_cache(df, cache_file, year=year, source_kind="grand_prix")
    return df

def fetch_qualifying(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Qualifying results (one row per driver per round)."""
    cache_file = CACHE_DIR / f"qualifying_{year}.csv"
    if not force_refresh:
        cached = _try_read_cache(cache_file, year=year, source_kind="grand_prix_qualifying")
        if cached is not None:
            return cached

    data = _get_json(f"{ERGAST}/{year}/qualifying.json", params={"limit": 10000})
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        round_no = int(race["round"])
        circuit = race["Circuit"]["circuitName"]
        for res in race.get("QualifyingResults", []):
            drv = res["Driver"]
            con = res["Constructor"]
            rows.append({
                "season": year,
                "round": round_no,
                "circuitName": circuit,
                "driverId": drv.get("driverId", ""),
                "driver": f'{drv.get("givenName","")} {drv.get("familyName","")}'.strip(),
                "constructorId": con.get("constructorId", ""),
                "position": int(res.get("position", 0) or 0),
                "q1": res.get("Q1", ""),
                "q2": res.get("Q2", ""),
                "q3": res.get("Q3", ""),
            })
    df = pd.DataFrame(rows, columns=[
        "season", "round", "circuitName", "driverId", "driver",
        "constructorId", "position", "q1", "q2", "q3",
    ])
    _write_cache(df, cache_file, year=year, source_kind="grand_prix_qualifying")
    return df

def fetch_sprint(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Sprint results (one row per driver per sprint round). Some seasons have none."""
    cache_file = CACHE_DIR / f"sprint_{year}.csv"
    if not force_refresh:
        cached = _try_read_cache(cache_file, year=year, source_kind="sprint")
        if cached is not None:
            return cached

    data = _get_json(f"{ERGAST}/{year}/sprint.json", params={"limit": 10000})
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        round_no = int(race["round"])
        circuit = race["Circuit"]["circuitName"]
        for res in race.get("SprintResults", []):
            drv = res["Driver"]
            con = res["Constructor"]
            status = res.get("status", "")
            rows.append({
                "season": year,
                "round": round_no,
                "circuitName": circuit,
                "driverId": drv.get("driverId", ""),
                "driver": f'{drv.get("givenName","")} {drv.get("familyName","")}'.strip(),
                "constructorId": con.get("constructorId", ""),
                "grid": int(res.get("grid", 0) or 0),
                "position": int(res.get("position", 0) or 0),
                "status": status,
                "fastestLapRank": int(res.get("FastestLap", {}).get("rank", 0) or 0),
                "is_dnf": _is_dnf(status),
            })
    df = pd.DataFrame(rows, columns=[
        "season", "round", "circuitName", "driverId", "driver",
        "constructorId", "position", "grid", "status", "fastestLapRank", "is_dnf",
    ])
    _write_cache(df, cache_file, year=year, source_kind="sprint")
    return df

def fetch_schedule(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Race schedule (round, circuit) for a season."""
    cache_file = CACHE_DIR / f"schedule_{year}.csv"
    if not force_refresh:
        cached = _try_read_cache(cache_file, year=year, source_kind="schedule")
        if cached is not None:
            return cached

    data = _get_json(f"{ERGAST}/{year}.json", params={"limit": 1000})
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        qualifying = race.get("Qualifying", {}) or {}
        sprint = race.get("Sprint", {}) or {}
        sprint_qualifying = race.get("SprintQualifying", {}) or {}
        first_practice = race.get("FirstPractice", {}) or {}
        second_practice = race.get("SecondPractice", {}) or {}
        third_practice = race.get("ThirdPractice", {}) or {}
        rows.append({
            "season": int(race["season"]),
            "round": int(race["round"]),
            "raceName": race.get("raceName", ""),
            "date": race.get("date", ""),
            "time": race.get("time", ""),
            "circuitName": race["Circuit"]["circuitName"],
            "qualifying_date": qualifying.get("date", ""),
            "qualifying_time": qualifying.get("time", ""),
            "sprint_date": sprint.get("date", ""),
            "sprint_time": sprint.get("time", ""),
            "sprint_qualifying_date": sprint_qualifying.get("date", ""),
            "sprint_qualifying_time": sprint_qualifying.get("time", ""),
            "practice_1_date": first_practice.get("date", ""),
            "practice_1_time": first_practice.get("time", ""),
            "practice_2_date": second_practice.get("date", ""),
            "practice_2_time": second_practice.get("time", ""),
            "practice_3_date": third_practice.get("date", ""),
            "practice_3_time": third_practice.get("time", ""),
        })
    df = pd.DataFrame(rows, columns=[
        "season", "round", "raceName", "date", "time", "circuitName",
        "qualifying_date", "qualifying_time", "sprint_date", "sprint_time",
        "sprint_qualifying_date", "sprint_qualifying_time",
        "practice_1_date", "practice_1_time", "practice_2_date", "practice_2_time",
        "practice_3_date", "practice_3_time",
    ])
    _write_cache(df, cache_file, year=year, source_kind="schedule")
    return df

def fetch_results_range(start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    dfs = [fetch_season_results(y, force_refresh=force_refresh) for y in range(start_year, end_year + 1)]
    return pd.concat(dfs, ignore_index=True)

def fetch_all_supporting(year: int, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Convenience: fetch results + qualifying + sprint + schedule for a year."""
    return {
        "results": fetch_season_results(year, force_refresh=force_refresh),
        "qualifying": fetch_qualifying(year, force_refresh=force_refresh),
        "sprint": fetch_sprint(year, force_refresh=force_refresh),
        "schedule": fetch_schedule(year, force_refresh=force_refresh),
    }
