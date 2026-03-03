from __future__ import annotations
import time
import requests
import pandas as pd

# This feed contains *both* drivers and constructors including their current fantasy prices.
FEED = "https://fantasy.formula1.com/feeds/drivers/1_en.json"

def _get_market() -> list[dict]:
    # buster avoids CDN caching
    params = {"buster": str(int(time.time()))}
    r = requests.get(FEED, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["Data"]["Value"]

def fetch_players(year: int | None = None) -> pd.DataFrame:
    rows = _get_market()
    df = pd.DataFrame(rows)

    df = df[(df["PositionName"] == "DRIVER") & (df["IsActive"].astype(str) == "1")].copy()

    df.rename(columns={
        "PlayerId": "playerId",
        "FUllName": "name",
        "Value": "price",
        "TeamName": "team",
        "SelectedPercentage": "selected_pct",
        "CaptainSelectedPercentage": "captain_selected_pct",
        "DriverReference": "driver_reference",
        "DriverTLA": "tla",
        "F1PlayerId": "f1_player_id",
    }, inplace=True)

    df["playerId"] = df["playerId"].astype(int)
    df["price"] = df["price"].astype(float)

    # Keep extra fields if present; downstream code can ignore them safely.
    cols = ["playerId", "name", "price", "team", "selected_pct", "captain_selected_pct",
            "driver_reference", "tla", "f1_player_id"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]

def fetch_teams(year: int | None = None) -> pd.DataFrame:
    rows = _get_market()
    df = pd.DataFrame(rows)

    df = df[(df["PositionName"] == "CONSTRUCTOR") & (df["IsActive"].astype(str) == "1")].copy()

    df.rename(columns={
        "PlayerId": "teamId",
        "FUllName": "name",
        "Value": "price",
        "SelectedPercentage": "selected_pct",
        "DriverTLA": "tla",
        "F1PlayerId": "f1_team_id",
    }, inplace=True)

    df["teamId"] = df["teamId"].astype(int)
    df["price"] = df["price"].astype(float)

    cols = ["teamId", "name", "price", "selected_pct", "tla", "f1_team_id"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]
