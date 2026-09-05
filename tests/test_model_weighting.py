import math

import pandas as pd
import pytest

from f1fantasy.app_core import (
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    apply_observed_playerstats_projection,
    apply_probabilistic_price_change_model,
    historical_scale_factor,
)
from f1fantasy.model import expected_scores_horizon
from f1fantasy.race_selection import RaceKey, RaceOption, recency_weights, resolve_selected_races


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


def _selection(rounds, p=1.0, excluded=()):
    available = [
        RaceOption(RaceKey(2026, round_no), f"Round {round_no}")
        for round_no in rounds
    ]
    selected = resolve_selected_races(
        available,
        "All",
        excluded_keys=[RaceKey(2026, round_no) for round_no in excluded],
    )
    return selected, recency_weights(selected, p)


def _official_rows(player_id, points_by_round, asset_type="driver"):
    return pd.DataFrame(
        [
            {
                "PlayerId": player_id,
                "asset_type": asset_type,
                "season": 2026,
                "round": round_no,
                "race_name": f"Round {round_no}",
                "fantasy_points": points,
                "is_played": 1,
            }
            for round_no, points in points_by_round.items()
        ]
    )


def _empty_assets():
    return pd.DataFrame(
        columns=[
            "id",
            "name",
            "exp_score",
            "next_race_exp_score",
            "horizon_expected_points",
            "historical_next_race_expected_points",
            "historical_horizon_expected_points",
        ]
    )


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
    assert float(current_only.loc[0, "current_proxy_next_race_expected_points"]) == 100.0
    assert float(current_only.loc[0, "historical_next_race_expected_points"]) == 10.0


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
    assert out_drivers.loc[0, "expected_points_source"] == "official_current"
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
    assert out_drivers.loc[0, "horizon_expected_points"] == 87.5
    assert out_drivers.loc[0, "horizon_expected_points"] > out_drivers.loc[0, "exp_score"]


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, 30.0),
        (1.0, 20.0),
        (0.5, (10.0 * 0.25 + 20.0 * 0.5 + 30.0) / 1.75),
    ],
)
def test_official_projection_uses_selected_race_recency_weights(p, expected):
    selected, weights = _selection([9, 11, 12], p=p)
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 5.0}])

    out, _constructors, _diag = apply_observed_playerstats_projection(
        drivers,
        _empty_assets(),
        _official_rows(1, {9: 10.0, 11: 20.0, 12: 30.0}),
        pd.DataFrame(),
        current_season_weight=1.0,
        past_season_weight=0.0,
        selected_races=selected,
        race_weights=weights,
    )

    assert out.loc[0, "observed_current_avg_points"] == pytest.approx(expected)
    assert out.loc[0, "next_race_expected_points"] == pytest.approx(expected)
    assert out.loc[0, "official_valid_race_count"] == 3


def test_excluding_latest_race_moves_weight_one_to_next_included_race():
    selected, weights = _selection([9, 11, 12], p=0.5, excluded=[12])
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 5.0}])

    out, _constructors, _diag = apply_observed_playerstats_projection(
        drivers,
        _empty_assets(),
        _official_rows(1, {9: 10.0, 11: 20.0, 12: 100.0}),
        pd.DataFrame(),
        current_season_weight=1.0,
        past_season_weight=0.0,
        selected_races=selected,
        race_weights=weights,
    )

    assert weights == {RaceKey(2026, 9): 0.5, RaceKey(2026, 11): 1.0}
    assert out.loc[0, "observed_current_avg_points"] == pytest.approx(50.0 / 3.0)


def test_official_zero_replacement_coverage_and_source_failure_are_preserved():
    selected, weights = _selection([9, 11, 12], p=1.0)
    drivers = pd.DataFrame(
        [
            {"id": 1, "name": "Replacement", "exp_score": 5.0},
            {"id": 2, "name": "Failed", "exp_score": 7.0},
        ]
    )

    out, _constructors, _diag = apply_observed_playerstats_projection(
        drivers,
        _empty_assets(),
        _official_rows(1, {11: 0.0, 12: 20.0}),
        pd.DataFrame(),
        current_season_weight=1.0,
        past_season_weight=0.0,
        selected_races=selected,
        race_weights=weights,
        driver_source_failures=[("driver", "2")],
    )
    replacement = out[out["id"] == 1].iloc[0]
    failed = out[out["id"] == 2].iloc[0]

    assert replacement["observed_current_avg_points"] == 10.0
    assert replacement["official_valid_race_count"] == 2
    assert replacement["official_missing_race_count"] == 1
    assert replacement["official_coverage_fraction"] == pytest.approx(2 / 3)
    assert replacement["official_observation_status"] == "incomplete"
    assert bool(failed["official_has_source_failure"]) is True
    assert failed["official_observation_status"] == "source_failure"


def test_driver_and_constructor_official_weighting_use_the_same_semantics():
    selected, weights = _selection([9, 11, 12], p=0.5)
    drivers = pd.DataFrame([{"id": 1, "name": "Driver", "exp_score": 5.0}])
    constructors = pd.DataFrame([{"id": 2, "name": "Team", "exp_score": 5.0}])

    out_drivers, out_constructors, _diag = apply_observed_playerstats_projection(
        drivers,
        constructors,
        _official_rows(1, {9: 10.0, 11: 20.0, 12: 30.0}),
        _official_rows(2, {9: 10.0, 11: 20.0, 12: 30.0}, "constructor"),
        current_season_weight=1.0,
        past_season_weight=0.0,
        selected_races=selected,
        race_weights=weights,
    )

    assert out_drivers.loc[0, "observed_current_avg_points"] == pytest.approx(
        out_constructors.loc[0, "observed_current_avg_points"]
    )


def test_components_are_blended_once_with_normalized_relative_weights():
    selected, weights = _selection([12], p=1.0)
    drivers = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Driver One",
                "exp_score": 999.0,
                "next_race_exp_score": 999.0,
                "horizon_expected_points": 999.0,
                "historical_next_race_expected_points": 10.0,
                "historical_horizon_expected_points": 30.0,
            },
            {
                "id": 2,
                "name": "Driver Two",
                "exp_score": 999.0,
                "next_race_exp_score": 999.0,
                "horizon_expected_points": 999.0,
                "historical_next_race_expected_points": 30.0,
                "historical_horizon_expected_points": 90.0,
            },
        ]
    )
    observations = pd.concat(
        [
            _official_rows(1, {12: 30.0}),
            _official_rows(2, {12: 10.0}),
        ],
        ignore_index=True,
    )

    out, _constructors, diag = apply_observed_playerstats_projection(
        drivers,
        _empty_assets(),
        observations,
        pd.DataFrame(),
        current_season_weight=1.0,
        past_season_weight=1.0,
        selected_races=selected,
        race_weights=weights,
        horizon_weight_sum=3.0,
    )

    first = out[out["id"] == 1].iloc[0]
    assert diag["historical_scale_factor"] == 1.0
    assert diag["blend_application_count"] == 1
    assert first["next_race_expected_points"] == 20.0
    assert first["horizon_expected_points"] == 60.0
    assert first["effective_current_share"] == 0.5
    assert first["effective_historical_share"] == 0.5
    assert diag["effective_current_share_mean"] == 0.5
    assert diag["effective_historical_share_mean"] == 0.5


def test_missing_components_and_both_zero_weights_have_deterministic_fallbacks():
    selected, weights = _selection([12], p=1.0)
    drivers = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Historical Only",
                "exp_score": 5.0,
                "historical_next_race_expected_points": 10.0,
                "historical_horizon_expected_points": 20.0,
            },
            {
                "id": 2,
                "name": "Current Only",
                "exp_score": 5.0,
                "historical_next_race_expected_points": math.nan,
                "historical_horizon_expected_points": math.nan,
            },
            {
                "id": 3,
                "name": "Both",
                "exp_score": 5.0,
                "historical_next_race_expected_points": 20.0,
                "historical_horizon_expected_points": 40.0,
            },
        ]
    )
    observations = pd.concat(
        [_official_rows(2, {12: 20.0}), _official_rows(3, {12: 10.0})],
        ignore_index=True,
    )

    out, _constructors, diag = apply_observed_playerstats_projection(
        drivers,
        _empty_assets(),
        observations,
        pd.DataFrame(),
        current_season_weight=0.0,
        past_season_weight=0.0,
        selected_races=selected,
        race_weights=weights,
        horizon_weight_sum=2.0,
    )
    by_id = out.set_index("id")

    assert by_id.loc[1, "next_race_expected_points"] == 10.0
    assert by_id.loc[1, "effective_historical_share"] == 1.0
    assert by_id.loc[2, "next_race_expected_points"] == 20.0
    assert by_id.loc[2, "effective_current_share"] == 1.0
    assert by_id.loc[3, "next_race_expected_points"] == 15.0
    assert by_id.loc[3, "effective_current_share"] == 0.5
    assert by_id.loc[3, "effective_historical_share"] == 0.5
    assert diag["both_weights_zero_behavior"].startswith("equal shares")


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
