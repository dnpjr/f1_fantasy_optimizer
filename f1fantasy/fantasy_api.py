from __future__ import annotations
import time
import requests
import pandas as pd

BASE = "https://fantasy.formula1.com/feeds/drivers"
REQUEST_TIMEOUT_SECONDS = 12
MARKET_CACHE_TTL_SECONDS = 60 * 15

_LATEST_FEED_CACHE: dict[str, float | int] = {"round": 0, "ts": 0.0}
_MARKET_CACHE: dict[int, tuple[float, list[dict]]] = {}

def _feed_url(feed_round: int) -> str:
    return f"{BASE}/{feed_round}_en.json"


def _valid_feed_round(feed_round: int) -> bool:
    try:
        response = requests.get(
            _feed_url(feed_round),
            params={"buster": str(int(time.time()))},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return False
        payload = response.json()
        return "Data" in payload and "Value" in payload["Data"]
    except Exception:
        return False

def _latest_feed_round(max_search: int = 40) -> int:
    """
    Find the latest available fantasy feed number by probing upward until a feed
    stops existing, then return the highest valid one.
    """
    now = time.time()
    cached_round = int(_LATEST_FEED_CACHE.get("round", 0) or 0)
    cached_ts = float(_LATEST_FEED_CACHE.get("ts", 0.0) or 0.0)
    if cached_round > 0 and (now - cached_ts) < MARKET_CACHE_TTL_SECONDS:
        return cached_round

    low, high = 1, max_search
    last_ok = 0
    # Feed validity is monotonic over round number, so binary search is safe and
    # significantly faster than linear probing on cold startup.
    while low <= high:
        mid = (low + high) // 2
        if _valid_feed_round(mid):
            last_ok = mid
            low = mid + 1
        else:
            high = mid - 1

    if last_ok <= 0:
        raise RuntimeError("Could not find any valid F1 Fantasy feed.")
    _LATEST_FEED_CACHE["round"] = int(last_ok)
    _LATEST_FEED_CACHE["ts"] = now
    return last_ok

def _get_market(feed_round: int | None = None) -> list[dict]:
    """
    Pull the latest market snapshot unless a specific feed_round is provided.
    """
    if feed_round is None:
        feed_round = _latest_feed_round()

    now = time.time()
    cached = _MARKET_CACHE.get(int(feed_round))
    if cached is not None and (now - cached[0]) < MARKET_CACHE_TTL_SECONDS:
        return cached[1]

    url = _feed_url(feed_round)
    params = {"buster": str(int(time.time()))}
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    j = r.json()
    value = j["Data"]["Value"]
    _MARKET_CACHE[int(feed_round)] = (now, value)
    return value

def fetch_players(year: int | None = None, feed_round: int | None = None) -> pd.DataFrame:
    rows = _get_market(feed_round=feed_round)
    df = pd.DataFrame(rows)

    df = df[(df["PositionName"] == "DRIVER") & (df["IsActive"].astype(str) == "1")].copy()

    # Handle possible schema spelling variants
    rename_map = {
        "PlayerId": "playerId",
        "Value": "price",
        "TeamName": "team",
        "SelectedPercentage": "selected_pct",
        "CaptainSelectedPercentage": "captain_selected_pct",
        "DriverReference": "driver_reference",
        "DriverTLA": "tla",
        "F1PlayerId": "f1_player_id",
    }
    if "FUllName" in df.columns:
        rename_map["FUllName"] = "name"
    elif "FullName" in df.columns:
        rename_map["FullName"] = "name"

    df.rename(columns=rename_map, inplace=True)

    df["playerId"] = df["playerId"].astype(int)
    df["price"] = df["price"].astype(float)

    cols = [
        "playerId", "name", "price", "team", "selected_pct",
        "captain_selected_pct", "driver_reference", "tla", "f1_player_id"
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]

def fetch_teams(year: int | None = None, feed_round: int | None = None) -> pd.DataFrame:
    rows = _get_market(feed_round=feed_round)
    df = pd.DataFrame(rows)

    df = df[(df["PositionName"] == "CONSTRUCTOR") & (df["IsActive"].astype(str) == "1")].copy()

    rename_map = {
        "PlayerId": "teamId",
        "Value": "price",
        "SelectedPercentage": "selected_pct",
        "DriverTLA": "tla",
        "F1PlayerId": "f1_team_id",
    }
    if "FUllName" in df.columns:
        rename_map["FUllName"] = "name"
    elif "FullName" in df.columns:
        rename_map["FullName"] = "name"

    df.rename(columns=rename_map, inplace=True)

    df["teamId"] = df["teamId"].astype(int)
    df["price"] = df["price"].astype(float)

    cols = ["teamId", "name", "price", "selected_pct", "tla", "f1_team_id"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]

def debug_feed_info() -> None:
    latest = _latest_feed_round()
    print(f"Latest available feed round: {latest}")
    print(f"Feed URL: {_feed_url(latest)}")
