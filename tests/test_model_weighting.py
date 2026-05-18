import math

import pandas as pd

from f1fantasy.app_core import (
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    apply_observed_playerstats_projection,
    apply_probabilistic_price_change_model,
    historical_scale_factor,
)
from f1fantasy.model import expected_scores_horizon


def _weekend_points(rows):
    base = {
        "quali_points": 0.0,
        "sprint_points": 0.0,
        "q2_reached": 0,
        "q3_reached": 0,
        "is_dsq": 0,
        "is_dnf": 0,
        "sprint_is_dnf": 0,
    }
    df = pd.DataFrame([{**base, **row} for row in rows])
    df["weekend_points"] = df["quali_points"] + df["sprint_points"] + df["race_points"]
    return df


def test_current_and_past_weights_change_driver_expectation():
    wp = _weekend_points(
        [
            {
                "season": 2025,
                "round": 1,
                "circuitName": "Example Track",
                "driverId": "d1",
                "driver": "Driver One",
                "constructorId": "c1",
                "constructor": "Constructor One",
                "race_points": 10.0,
            },
            {
                "season": 2026,
                "round": 1,
                "circuitName": "Current Track",
                "driverId": "d1",
                "driver": "Driver One",
                "constructorId": "c1",
                "constructor": "Constructor One",
                "race_points": 100.0,
            },
        ]
    )

    current_only, _ = expected_scores_horizon(
        wp,
        ["Example Track"],
        [1.0],
        current_season_weight=1.0,
        past_season_weight=0.0,
    )
    past_only, _ = expected_scores_horizon(
        wp,
        ["Example Track"],
        [1.0],
        current_season_weight=0.0,
        past_season_weight=1.0,
    )

    assert float(current_only.loc[0, "exp_score"]) == 100.0
    assert float(past_only.loc[0, "exp_score"]) == 10.0


def test_recency_decay_changes_current_season_expectation():
    wp = _weekend_points(
        [
            {
                "season": 2026,
                "round": 1,
                "circuitName": "Round One",
                "driverId": "d1",
                "driver": "Driver One",
                "constructorId": "c1",
                "constructor": "Constructor One",
                "race_points": 0.0,
            },
            {
                "season": 2026,
                "round": 2,
                "circuitName": "Round Two",
                "driverId": "d1",
                "driver": "Driver One",
                "constructorId": "c1",
                "constructor": "Constructor One",
                "race_points": 100.0,
            },
        ]
    )

    latest_only, _ = expected_scores_horizon(wp, ["Future Track"], [1.0], recency_decay=0.0)
    equal_weight, _ = expected_scores_horizon(wp, ["Future Track"], [1.0], recency_decay=1.0)

    assert float(latest_only.loc[0, "exp_score"]) == 100.0
    assert float(equal_weight.loc[0, "exp_score"]) == 50.0


def test_playerstats_observed_points_are_preferred_over_estimates():
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 10.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 20.0}])
    driver_races = pd.DataFrame(
        [
            {"PlayerId": 1, "fantasy_points": 100.0, "is_played": 1},
            {"PlayerId": 1, "fantasy_points": 120.0, "is_played": 1},
        ]
    )
    constructor_races = pd.DataFrame([{"PlayerId": 2, "fantasy_points": 50.0, "is_played": 1}])

    out_drivers, _out_constructors, diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=1.0,
        past_season_weight=0.0,
    )

    assert out_drivers.loc[0, "exp_score"] == 110.0
    assert out_drivers.loc[0, "expected_points_source"] == "playerstats_blended"
    assert diag["observed_current_assets"] == 2


def test_historical_estimates_are_scaled_to_current_observed_average():
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 10.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 10.0}])
    driver_races = pd.DataFrame([{"PlayerId": 1, "fantasy_points": 20.0, "is_played": 1}])
    constructor_races = pd.DataFrame([{"PlayerId": 2, "fantasy_points": 20.0, "is_played": 1}])

    out_drivers, _out_constructors, diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=0.0,
        past_season_weight=1.0,
    )

    assert diag["historical_scale_factor"] == 1.5
    assert diag["historical_scale_factor_clipped"] is True
    assert out_drivers.loc[0, "exp_score"] == 15.0


def test_scale_factor_handles_zero_and_clipping():
    assert historical_scale_factor(20.0, 0.0) == (1.0, False)
    assert historical_scale_factor(100.0, 10.0) == (1.5, True)
    assert historical_scale_factor(1.0, 10.0) == (0.5, True)


def test_app_expected_points_are_per_race_and_horizon_is_separate():
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 10.0, "next_race_exp_score": 10.0, "horizon_expected_points": 50.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 10.0, "next_race_exp_score": 10.0, "horizon_expected_points": 50.0}])
    driver_races = pd.DataFrame([{"PlayerId": 1, "fantasy_points": 20.0, "is_played": 1}])
    constructor_races = pd.DataFrame([{"PlayerId": 2, "fantasy_points": 20.0, "is_played": 1}])

    out_drivers, _out_constructors, _diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=2.0,
        past_season_weight=2.0,
    )

    assert out_drivers.loc[0, "exp_score"] == 17.5
    assert out_drivers.loc[0, "next_race_expected_points"] == 17.5
    assert out_drivers.loc[0, "normalised_historical_expected_points_per_race"] == 15.0
    assert out_drivers.loc[0, "horizon_expected_points"] == 47.5
    assert out_drivers.loc[0, "horizon_expected_points"] > out_drivers.loc[0, "exp_score"]


def test_current_season_volatility_comes_from_playerstats_points():
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 10.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 10.0}])
    driver_races = pd.DataFrame(
        [
            {"PlayerId": 1, "fantasy_points": 10.0, "is_played": 1},
            {"PlayerId": 1, "fantasy_points": 30.0, "is_played": 1},
        ]
    )
    constructor_races = pd.DataFrame([{"PlayerId": 2, "fantasy_points": 20.0, "is_played": 1}])

    out_drivers, _out_constructors, diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=1.0,
        past_season_weight=0.0,
    )

    assert out_drivers.loc[0, "current_season_points_count"] == 2
    assert out_drivers.loc[0, "current_season_avg_points"] == 20.0
    assert out_drivers.loc[0, "current_season_volatility"] == 10.0
    assert out_drivers.loc[0, "volatility"] == 10.0
    assert diag["current_season_volatility_assets"] == 1


def test_current_season_volatility_missing_with_fewer_than_two_races():
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 10.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 10.0}])
    driver_races = pd.DataFrame([{"PlayerId": 1, "fantasy_points": 10.0, "is_played": 1}])
    constructor_races = pd.DataFrame()

    out_drivers, _out_constructors, diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=1.0,
        past_season_weight=0.0,
    )

    assert pd.isna(out_drivers.loc[0, "current_season_volatility"])
    assert out_drivers.loc[0, "volatility_source"] == "fallback_floor"
    assert out_drivers.loc[0, "volatility"] == 5.0
    assert diag["fallback_volatility_assets"] >= 1


def test_historical_volatility_is_scaled_to_current_scoring_scale():
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 10.0, "volatility": 4.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 10.0, "volatility": 4.0}])
    driver_races = pd.DataFrame([{"PlayerId": 1, "fantasy_points": 20.0, "is_played": 1}])
    constructor_races = pd.DataFrame([{"PlayerId": 2, "fantasy_points": 20.0, "is_played": 1}])

    out_drivers, _out_constructors, diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=0.0,
        past_season_weight=1.0,
    )

    assert diag["historical_scale_factor"] == 1.5
    assert out_drivers.loc[0, "normalised_historical_volatility"] == 6.0
    assert out_drivers.loc[0, "volatility"] == 6.0
    assert out_drivers.loc[0, "volatility_source"] == "historical_model_proxy"


def test_blended_volatility_uses_normalised_active_weights():
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 10.0, "volatility": 20.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 10.0, "volatility": 20.0}])
    driver_races = pd.DataFrame(
        [
            {"PlayerId": 1, "fantasy_points": 10.0, "is_played": 1},
            {"PlayerId": 1, "fantasy_points": 30.0, "is_played": 1},
        ]
    )
    constructor_races = pd.DataFrame([{"PlayerId": 2, "fantasy_points": 20.0, "is_played": 1}])

    out_drivers, _out_constructors, _diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=1.0,
        past_season_weight=1.0,
    )

    assert out_drivers.loc[0, "current_season_volatility"] == 10.0
    assert out_drivers.loc[0, "normalised_historical_volatility"] == 30.0
    assert out_drivers.loc[0, "volatility"] == 20.0
    assert out_drivers.loc[0, "volatility_source"] == "blended_current_historical"


def test_volatility_floor_is_applied():
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 10.0, "volatility": 1.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 10.0, "volatility": 1.0}])
    driver_races = pd.DataFrame(
        [
            {"PlayerId": 1, "fantasy_points": 19.0, "is_played": 1},
            {"PlayerId": 1, "fantasy_points": 21.0, "is_played": 1},
        ]
    )
    constructor_races = pd.DataFrame()

    out_drivers, _out_constructors, diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=1.0,
        past_season_weight=0.0,
    )

    assert out_drivers.loc[0, "blended_volatility_before_floor"] == 1.0
    assert out_drivers.loc[0, "volatility"] == 5.0
    assert bool(out_drivers.loc[0, "volatility_floor_applied"]) is True
    assert diag["volatility_floor_applied_assets"] >= 1


def test_probabilistic_price_gain_uses_blended_volatility():
    drivers = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Driver",
                "price": 10.0,
                "exp_score": 10.0,
                "volatility": 20.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 10.0,
            }
        ]
    )
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 10.0, "volatility": 20.0}])
    driver_races = pd.DataFrame(
        [
            {"PlayerId": 1, "fantasy_points": 10.0, "is_played": 1},
            {"PlayerId": 1, "fantasy_points": 30.0, "is_played": 1},
        ]
    )
    constructor_races = pd.DataFrame([{"PlayerId": 2, "fantasy_points": 20.0, "is_played": 1}])

    out_drivers, _out_constructors, _diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        driver_races,
        constructor_races,
        current_season_weight=1.0,
        past_season_weight=1.0,
    )
    projected = apply_probabilistic_price_change_model(out_drivers, DEFAULT_PRICE_CHANGE_CHEAP_RULES)

    assert projected.loc[0, "volatility_used"] == out_drivers.loc[0, "volatility"]
    assert math.isclose(
        projected.loc[0, "p_terrible"]
        + projected.loc[0, "p_poor"]
        + projected.loc[0, "p_good"]
        + projected.loc[0, "p_great"],
        1.0,
        rel_tol=1e-9,
    )
