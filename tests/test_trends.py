import pandas as pd
import pytest

from f1fantasy.app_core import (
    build_trends_data,
    filter_trends_data,
    hide_user_internal_columns,
    selected_assets_price_gain,
    team_colour,
)


def _drivers():
    return pd.DataFrame(
        [
            {"id": "1", "name": "Driver One", "team": "Ferrari", "price": 10.0, "team_colour": team_colour("Ferrari")},
            {"id": "2", "name": "Driver Two", "team": "McLaren", "price": 0.0, "team_colour": team_colour("McLaren")},
        ]
    )


def _constructors():
    return pd.DataFrame(
        [
            {"id": "10", "name": "Mercedes", "price": 30.0, "team_colour": team_colour("Mercedes")},
        ]
    )


def test_hidden_team_colour_column_is_removed_from_display_frames():
    frame = pd.DataFrame([{"Name": "Driver One", "Team": "Ferrari", "team_colour": "#dc2626"}])

    display = hide_user_internal_columns(frame)

    assert "team_colour" not in display.columns
    assert list(display.columns) == ["Name", "Team"]


def test_team_colour_helper_uses_known_colours_and_fallback():
    assert team_colour("Ferrari") == "#dc2626"
    assert team_colour("Unknown Racing Outfit") == "#64748b"


def test_selected_assets_price_gain_totals_effective_changes():
    drivers = pd.DataFrame([{"expected_price_change": 0.2}, {"expected_price_change": -0.1}])
    constructors = pd.DataFrame([{"effective_price_change_after_floor_ceiling": 0.3}])

    assert selected_assets_price_gain(drivers, constructors) == pytest.approx(0.4)


def test_build_trends_data_computes_long_format_cumulative_and_rolling_average():
    driver_points = pd.DataFrame(
        [
            {"PlayerId": 1, "round": 1, "race_name": "Race 1", "fantasy_points": 10.0, "price": 10.0, "is_played": 1},
            {"PlayerId": 1, "round": 2, "race_name": "Race 2", "fantasy_points": 20.0, "price": 10.0, "is_played": 1},
            {"PlayerId": 1, "round": 3, "race_name": "Race 3", "fantasy_points": 30.0, "price": 10.0, "is_played": 1},
            {"PlayerId": 1, "round": 4, "race_name": "Race 4", "fantasy_points": 40.0, "price": 10.0, "is_played": 1},
            {"PlayerId": 2, "round": 1, "race_name": "Race 1", "fantasy_points": 5.0, "price": 0.0, "is_played": 1},
        ]
    )
    constructor_points = pd.DataFrame(
        [
            {"PlayerId": 10, "round": 1, "race_name": "Race 1", "fantasy_points": 50.0, "price": 30.0, "is_played": 1},
        ]
    )

    trends = build_trends_data(_drivers(), _constructors(), driver_points, constructor_points)
    driver_one = trends[(trends["asset_type"] == "driver") & (trends["asset_id"] == "1")].sort_values("round")

    assert list(driver_one["cumulative_points"]) == [10.0, 30.0, 60.0, 100.0]
    assert list(driver_one["rolling_3race_avg"]) == [10.0, 15.0, 20.0, 30.0]
    assert driver_one.loc[driver_one["round"] == 2, "points_per_million"].iloc[0] == pytest.approx(2.0)

    driver_two = trends[(trends["asset_type"] == "driver") & (trends["asset_id"] == "2")]
    assert pd.isna(driver_two["points_per_million"].iloc[0])


def test_filter_trends_data_filters_asset_type_and_selected_assets():
    points = pd.DataFrame(
        [
            {"PlayerId": 1, "round": 1, "race_name": "Race 1", "fantasy_points": 10.0, "price": 10.0, "is_played": 1},
            {"PlayerId": 2, "round": 1, "race_name": "Race 1", "fantasy_points": 5.0, "price": 8.0, "is_played": 1},
        ]
    )
    trends = build_trends_data(_drivers(), _constructors(), points, pd.DataFrame())

    filtered = filter_trends_data(trends, asset_type="driver", selected_asset_ids=["2"])

    assert filtered["asset_id"].tolist() == ["2"]
    assert filtered["name"].tolist() == ["Driver Two"]
