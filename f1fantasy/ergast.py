from __future__ import annotations
from pathlib import Path
import pandas as pd
import requests

# Jolpica provides Ergast-compatible endpoints (Ergast has been deprecated)
ERGAST = "http://api.jolpi.ca/ergast/f1"

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
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _try_read_cache(cache_file: Path) -> pd.DataFrame | None:
    """Return cached dataframe, or None if cache missing/corrupt/empty (deletes corrupt cache)."""
    if not cache_file.exists():
        return None
    try:
        df = pd.read_csv(cache_file)
        # EmptyDataError gets caught above; also guard empty frames
        if df.shape[1] == 0:
            raise ValueError("empty cache")
        return df
    except Exception:
        try:
            cache_file.unlink()
        except Exception:
            pass
        return None

def fetch_season_results(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Race results (one row per driver per race). Cached to data/cache/results_<year>.csv"""
    cache_file = CACHE_DIR / f"results_{year}.csv"
    if not force_refresh:
        cached = _try_read_cache(cache_file)
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

    df = pd.DataFrame(rows)
    df.to_csv(cache_file, index=False)
    return df

def fetch_qualifying(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Qualifying results (one row per driver per round)."""
    cache_file = CACHE_DIR / f"qualifying_{year}.csv"
    if not force_refresh:
        cached = _try_read_cache(cache_file)
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
    df = pd.DataFrame(rows)
    df.to_csv(cache_file, index=False)
    return df

def fetch_sprint(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Sprint results (one row per driver per sprint round). Some seasons have none."""
    cache_file = CACHE_DIR / f"sprint_{year}.csv"
    if not force_refresh:
        cached = _try_read_cache(cache_file)
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
    df = pd.DataFrame(rows)
    df.to_csv(cache_file, index=False)
    return df

def fetch_schedule(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Race schedule (round, circuit) for a season."""
    cache_file = CACHE_DIR / f"schedule_{year}.csv"
    if not force_refresh:
        cached = _try_read_cache(cache_file)
        if cached is not None:
            return cached

    data = _get_json(f"{ERGAST}/{year}.json", params={"limit": 1000})
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        rows.append({
            "season": int(race["season"]),
            "round": int(race["round"]),
            "raceName": race.get("raceName", ""),
            "date": race.get("date", ""),
            "circuitName": race["Circuit"]["circuitName"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(cache_file, index=False)
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
