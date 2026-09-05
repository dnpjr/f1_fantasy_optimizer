from __future__ import annotations

import pandas as pd
import pytest

from f1fantasy.app_core import (
    DEFAULT_PRICE_CHANGE_BOUNDS,
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    apply_probabilistic_price_change_model,
    current_team_budget_from_selection,
)
from f1fantasy.optimize import optimize_top_k
from f1fantasy.price_efficiency import build_price_efficiency_table
from f1fantasy.race_selection import RaceKey, RaceOption, resolve_selected_races
from f1fantasy.ui_helpers import compact_asset_payload


def _accepted_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    drivers = pd.DataFrame(
        [
            {
                "id": str(index),
                "name": "Pierre Gasly" if index == 18 else f"Driver {index}",
                "price": 13.0 if index == 18 else 8.0 + index / 10,
                "previous_price": 12.8 if index == 18 else 7.9 + index / 10,
                "official_price_change": 0.2 if index == 18 else 0.1,
                "exp_score": 10.0 + index,
                "next_race_expected_points": 10.0 + index,
                "recent_points_2ago": 8.0,
                "recent_points_1ago": 6.0,
                "volatility": 5.0,
                "dnf_rate": 0.1,
            }
            for index in [18, 2, 3, 4, 5]
        ]
    )
    constructors = pd.DataFrame(
        [
            {
                "id": str(index),
                "name": "Mercedes" if index == 28 else "Constructor 29",
                "price": 32.6 if index == 28 else 20.0,
                "previous_price": 32.3 if index == 28 else 19.7,
                "official_price_change": 0.3,
                "exp_score": 40.0 + index,
            }
            for index in [28, 29]
        ]
    )
    return drivers, constructors


def test_current_price_is_consistent_for_ui_optimizer_team_value_and_efficiency():
    drivers, constructors = _accepted_assets()
    solution = optimize_top_k(drivers, constructors, budget=200.0, k=1)[0]
    gasly = drivers[drivers["id"].eq("18")].iloc[0]
    ui_payload = compact_asset_payload(gasly, asset_type="driver")
    selection = resolve_selected_races(
        (RaceOption(RaceKey(2026, 11), "Hungarian Grand Prix"),),
        "All",
    )
    observations = pd.DataFrame(
        [
            {
                "PlayerId": "18",
                "asset_type": "driver",
                "season": 2026,
                "round": 11,
                "fantasy_points": 6.0,
                "is_played": 1,
            }
        ]
    )
    efficiency = build_price_efficiency_table(
        drivers[drivers["id"].eq("18")],
        observations,
        selection,
        asset_type="driver",
    )

    assert gasly["price"] == pytest.approx(13.0)
    assert ui_payload["price_value"] == pytest.approx(13.0)
    assert solution.drivers.loc[solution.drivers["id"].eq("18"), "price"].iloc[0] == pytest.approx(13.0)
    assert efficiency.loc[0, "current_price"] == pytest.approx(13.0)
    assert current_team_budget_from_selection(drivers, constructors) == pytest.approx(
        drivers["price"].sum() + constructors["price"].sum()
    )


def test_official_movement_and_projected_gain_remain_separate():
    drivers, _constructors = _accepted_assets()
    gasly = drivers[drivers["id"].eq("18")].copy()

    projected = apply_probabilistic_price_change_model(
        gasly,
        DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        bounds=DEFAULT_PRICE_CHANGE_BOUNDS,
        predicted_points_col="next_race_expected_points",
    )

    assert projected.loc[0, "price"] == pytest.approx(13.0)
    assert projected.loc[0, "previous_price"] == pytest.approx(12.8)
    assert projected.loc[0, "official_price_change"] == pytest.approx(0.2)
    assert "expected_price_gain" in projected.columns
    assert projected.loc[0, "expected_price_gain"] != pytest.approx(
        projected.loc[0, "official_price_change"]
    )
