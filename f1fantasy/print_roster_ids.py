"""
Print current Fantasy F1 roster IDs (drivers + constructors).

This is meant to make it easy to build:
  f1fantasy/data/current_team.json

Outputs:
- Driver IDs (playerId) with name, team, price, tla, driver_reference
- Constructor IDs (teamId) with name, price, tla
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Ensure local package imports work when running as a script
PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from f1fantasy.fantasy_api import fetch_players, fetch_teams  # noqa: E402


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def main() -> None:
    players = fetch_players()
    teams = fetch_teams()

    if players is None or len(players) == 0:
        print("No players returned from fantasy API.")
        return
    if teams is None or len(teams) == 0:
        print("No teams returned from fantasy API.")
        return

    # --- Drivers ---
    print("\n=== DRIVERS (playerId) ===")
    driver_cols = _safe_cols(
        players,
        ["playerId", "name", "team", "price", "tla", "driver_reference"],
    )
    d = players[driver_cols].copy()

    if "price" in d.columns:
        d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d.sort_values(["price", "name"], ascending=[False, True])

    print(d.to_string(index=False))

    # --- Constructors ---
    print("\n=== CONSTRUCTORS (teamId) ===")
    team_cols = _safe_cols(teams, ["teamId", "name", "price", "tla"])
    t = teams[team_cols].copy()

    if "price" in t.columns:
        t["price"] = pd.to_numeric(t["price"], errors="coerce")
    t = t.sort_values(["price", "name"], ascending=[False, True])

    print(t.to_string(index=False))

    print("\nTip: Use these IDs in f1fantasy/data/current_team.json")
    print("Drivers use playerId; constructors use teamId.\n")


if __name__ == "__main__":
    main()