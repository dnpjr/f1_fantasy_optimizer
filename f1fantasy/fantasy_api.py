from __future__ import annotations
import time
import requests
import pandas as pd

BASE = "https://fantasy.formula1.com/feeds/drivers"

def _feed_url(feed_round: int) -> str:
    return f"{BASE}/{feed_round}_en.json"

def _latest_feed_round(max_search: int = 40) -> int:
    """
    Find the latest available fantasy feed number by probing upward until a feed
    stops existing, then return the highest valid one.
    """
    last_ok = None

    for n in range(1, max_search + 1):
        url = _feed_url(n)
        try:
            r = requests.get(url, params={"buster": str(int(time.time()))}, timeout=15)
            if r.status_code == 200:
                # Check it is a valid fantasy payload
                j = r.json()
                if "Data" in j and "Value" in j["Data"]:
                    last_ok = n
                else:
                    break
            else:
                break
        except Exception:
            break

    if last_ok is None:
        raise RuntimeError("Could not find any valid F1 Fantasy feed.")
    return last_ok

def _get_market(feed_round: int | None = None) -> list[dict]:
    """
    Pull the latest market snapshot unless a specific feed_round is provided.
    """
    if feed_round is None:
        feed_round = _latest_feed_round()

    url = _feed_url(feed_round)
    params = {"buster": str(int(time.time()))}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["Data"]["Value"]

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