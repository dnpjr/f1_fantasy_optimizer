from __future__ import annotations

import pandas as pd
import pytest

from f1fantasy import app_core, fantasy_api
from f1fantasy.asset_identity import build_player_identity_map
from f1fantasy.ui_helpers import compact_asset_table_html


def _rules() -> app_core.PriceChangeRules:
    return app_core.PriceChangeRules(
        terrible_max=0.5,
        poor_min=0.5,
        poor_max=1.0,
        good_min=1.0,
        good_max=2.0,
        great_min=2.0,
        terrible_price_change=-0.6,
        poor_price_change=-0.2,
        good_price_change=0.2,
        great_price_change=0.6,
    )


def _raw_market() -> list[dict]:
    return [
        {
            "PlayerId": 1,
            "FUllName": "Established Driver",
            "PositionName": "DRIVER",
            "IsActive": 1,
            "Value": 10.0,
            "OldPlayerValue": 9.9,
            "TeamId": 1,
            "TeamName": "Established Team",
            "DriverReference": "ESTABLISHED01",
            "DriverTLA": "EST",
            "F1PlayerId": 1,
        },
        {
            "PlayerId": 114,
            "FUllName": "Liam Lawson",
            "PositionName": "DRIVER",
            "IsActive": 0,
            "Value": 13.0,
            "OldPlayerValue": 12.9,
            "TeamId": 9,
            "TeamName": "Racing Bulls",
            "DriverReference": "LIALAW01",
            "DriverTLA": "LAW",
            "F1PlayerId": -8,
        },
        {
            "PlayerId": 116,
            "FUllName": "Liam Lawson",
            "PositionName": "DRIVER",
            "IsActive": 1,
            "Value": 13.0,
            "OldPlayerValue": 13.0,
            "TeamId": 5,
            "TeamName": "Red Bull",
            "DriverReference": "LIALAW01",
            "DriverTLA": "LAW",
            "F1PlayerId": 14,
        },
        {
            "PlayerId": 130,
            "FUllName": "Yuki Tsunoda",
            "PositionName": "DRIVER",
            "IsActive": 1,
            "Value": 10.2,
            "OldPlayerValue": 10.2,
            "TeamId": 9,
            "TeamName": "Racing Bulls",
            "DriverReference": "YUKTSU01",
            "DriverTLA": "TSU",
            "F1PlayerId": 22,
        },
        {
            "PlayerId": 11032,
            "FUllName": "Isack Hadjar",
            "PositionName": "DRIVER",
            "IsActive": 0,
            "Value": 13.0,
            "OldPlayerValue": 12.8,
            "TeamId": 5,
            "TeamName": "Red Bull",
            "DriverReference": "ISAHAD01",
            "DriverTLA": "HAD",
            "F1PlayerId": 47,
        },
    ]


def _identity_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"driverId": "established", "driver": "Established Driver"},
            {"driverId": "lawson", "driver": "Liam Lawson"},
            {"driverId": "tsunoda", "driver": "Yuki Tsunoda"},
            {"driverId": "hadjar", "driver": "Isack Hadjar"},
        ]
    )


def _selectable_model() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "1",
                "name": "Established Driver",
                "team": "Established Team",
                "price": 10.0,
                "exp_score": 30.0,
                "next_race_expected_points": 30.0,
                "volatility": 5.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 20.0,
                "recent_points_available": 2,
            },
            {
                "id": "116",
                "name": "Liam Lawson",
                "team": "Red Bull",
                "price": 13.0,
                "exp_score": 26.0,
                "next_race_expected_points": 26.0,
                "volatility": 6.0,
                "human_driver_id": "lawson",
                "recent_points_available": 0,
            },
            {
                "id": "130",
                "name": "Yuki Tsunoda",
                "team": "Racing Bulls",
                "price": 10.2,
                "exp_score": 18.0,
                "next_race_expected_points": 18.0,
                "volatility": 5.0,
                "human_driver_id": "tsunoda",
                "recent_points_available": 0,
            },
        ]
    )


def _race_observations() -> pd.DataFrame:
    rows = []
    for asset_id, points in {
        1: [10.0, 20.0],
        114: [40.0, 50.0],
        11032: [5.0, 6.0],
    }.items():
        for round_no, score in enumerate(points, start=1):
            rows.append(
                {
                    "PlayerId": asset_id,
                    "season": 2026,
                    "round": round_no,
                    "fantasy_points": score,
                    "is_played": 1,
                }
            )
    return pd.DataFrame(rows)


def _price_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    identity = build_player_identity_map(ledger, _identity_history())
    universe = app_core.build_price_change_asset_universe(
        _selectable_model(),
        ledger,
        "driver",
        race_observations=_race_observations(),
        player_identity_map=identity,
    )
    projected = app_core.apply_probabilistic_price_change_model(
        universe,
        _rules(),
        predicted_points_col="next_race_expected_points",
    )
    return ledger, universe, projected


@pytest.mark.parametrize(
    ("prior", "forecast", "expected_average", "count", "window"),
    [
        ([], 30.0, 30.0, 0, 1),
        ([10.0], 30.0, 20.0, 1, 2),
        ([10.0, 20.0], 30.0, 20.0, 2, 3),
        ([999.0, 10.0, 20.0], 30.0, 20.0, 2, 3),
    ],
)
def test_prospective_price_history_uses_zero_one_or_latest_two_priors(
    prior, forecast, expected_average, count, window
):
    result = app_core.prospective_price_history(prior, forecast)

    assert result["completed_observations_used"] == count
    assert result["prospective_window_length"] == window
    assert result["projected_rolling_average"] == pytest.approx(expected_average)
    assert 999.0 not in result["prior_observations"]


def test_threshold_inversion_uses_actual_prospective_window_length():
    assert app_core.required_next_points_for_history(10.0, 2.0, []) == 20.0
    assert app_core.required_next_points_for_history(10.0, 2.0, [10.0]) == 30.0
    assert app_core.required_next_points_for_history(10.0, 2.0, [10.0, 20.0]) == 30.0


def test_established_active_asset_retains_three_observation_price_math():
    frame = _selectable_model().iloc[[0]].copy()
    out = app_core.apply_probabilistic_price_change_model(
        frame,
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).iloc[0]

    assert out["projected_rolling_average"] == pytest.approx(20.0)
    assert out["avg_ppm"] == pytest.approx(2.0)
    assert out["price_change_tier"] == "Great"
    assert out["effective_price_change_after_floor_ceiling"] == pytest.approx(0.6)
    assert out["price_history_mode"] == "established"
    assert out["price_history_prior_observations"] == (10.0, 20.0)


def test_fresh_zero_history_asset_gets_projection_and_normalised_probabilities():
    frame = _selectable_model().query("id == '116'")
    out = app_core.apply_probabilistic_price_change_model(
        frame,
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).iloc[0]

    probability_sum = sum(out[column] for column in ("p_terrible", "p_poor", "p_good", "p_great"))
    assert out["price_history_observations"] == 0
    assert out["price_history_prior_observations"] == ()
    assert out["projected_rolling_average"] == pytest.approx(26.0)
    assert out["price_history_mode"] == "fresh"
    assert probability_sum == pytest.approx(1.0)
    assert pd.notna(out["expected_price_gain"])
    threshold = app_core.price_change_threshold_table(
        frame,
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).iloc[0]
    assert threshold["required_great_min"] == pytest.approx(26.0)


def test_one_completed_observation_uses_two_event_window_only():
    frame = _selectable_model().query("id == '116'").assign(
        recent_points_1ago=14.0,
        recent_points_available=1,
    )
    projected = app_core.apply_probabilistic_price_change_model(
        frame,
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).iloc[0]
    thresholds = app_core.price_change_threshold_table(
        frame,
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).iloc[0]

    assert projected["price_history_prior_observations"] == (14.0,)
    assert projected["projected_rolling_average"] == pytest.approx(20.0)
    assert thresholds["required_great_min"] == pytest.approx(38.0)


def test_same_human_new_asset_and_same_price_seat_predecessor_do_not_share_history():
    ledger, universe, projected = _price_universe()
    by_id = projected.set_index("id")

    assert by_id.loc["114", "human_driver_id"] == "lawson"
    assert by_id.loc["116", "human_driver_id"] == "lawson"
    assert by_id.loc["114", "price_history_prior_observations"] == (40.0, 50.0)
    assert by_id.loc["116", "price_history_prior_observations"] == ()
    assert by_id.loc["116", "price_history_observations"] == 0
    assert by_id.loc["116", "next_race_expected_points"] == pytest.approx(26.0)
    assert pd.notna(by_id.loc["116", "expected_price_gain"])
    assert by_id.loc["11032", "price"] == by_id.loc["116", "price"]
    assert by_id.loc["11032", "price_history_prior_observations"] == (5.0, 6.0)
    assert by_id.loc["116", "price_history_prior_observations"] == ()
    assert 116 not in set(
        app_core.completed_asset_price_history(_race_observations())["id"].astype(int)
    )
    assert len(ledger) == len(universe)


def test_established_active_rows_are_numerically_identical_through_price_view():
    _ledger, universe, _projected = _price_universe()
    direct = app_core.apply_probabilistic_price_change_model(
        _selectable_model(),
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).set_index("id")
    via_price_view = app_core.apply_probabilistic_price_change_model(
        universe[universe["id"] == "1"],
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).set_index("id")
    columns = [
        "avg_ppm",
        "price_change_tier",
        "effective_price_change_after_floor_ceiling",
        "p_terrible",
        "p_poor",
        "p_good",
        "p_great",
        "expected_price_gain",
    ]

    pd.testing.assert_series_equal(
        via_price_view.loc["1", columns],
        direct.loc["1", columns],
        check_names=False,
    )


def test_probability_diagnostics_expose_exact_asset_history_and_identity():
    _ledger, universe, _projected = _price_universe()
    details = app_core.price_change_probability_matrix_table(
        universe,
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).set_index("Fantasy asset ID")

    assert details.loc["116", "Human identity"] == "lawson"
    assert details.loc["116", "History used"] == 0
    assert details.loc["116", "Prior observations"] == ()
    assert details.loc["116", "Projected rolling average"] == pytest.approx(26.0)
    assert details.loc["116", "History mode"] == "fresh"
    assert details.loc["114", "History mode"] == "inactive_unknown"
    assert details.loc["114", "Price eligibility"] == "inactive_price_eligibility_unknown"


def test_tsunoda_style_fresh_asset_keeps_human_ev_without_fabricated_history():
    _ledger, _universe, projected = _price_universe()
    tsunoda = projected.set_index("id").loc["130"]

    assert tsunoda["human_driver_id"] == "tsunoda"
    assert tsunoda["next_race_expected_points"] == pytest.approx(18.0)
    assert tsunoda["price_history_observations"] == 0
    assert tsunoda["price_history_prior_observations"] == ()
    assert pd.notna(tsunoda["expected_price_gain"])


def test_inactive_priced_assets_are_visible_but_gain_and_eligibility_are_unknown():
    ledger, universe, projected = _price_universe()
    by_id = projected.set_index("id")
    selectable_ids = set(fantasy_api.selectable_player_assets(ledger)["playerId"])

    assert {"114", "11032"}.issubset(set(universe["id"]))
    assert by_id.loc["114", "price"] == pytest.approx(13.0)
    assert by_id.loc["11032", "price"] == pytest.approx(13.0)
    assert by_id.loc["114", "price_history_mode"] == "inactive_unknown"
    assert by_id.loc["114", "price_eligibility_status"] == "inactive_price_eligibility_unknown"
    assert pd.isna(by_id.loc["114", "expected_price_gain"])
    assert pd.isna(by_id.loc["11032", "expected_price_gain"])
    assert 114 not in selectable_ids and 11032 not in selectable_ids

    rendered = compact_asset_table_html(
        projected[projected["id"] == "114"],
        asset_type="driver",
    )
    assert "Inactive" in rendered
    assert "0.00" not in rendered
    assert "—" in rendered


def test_price_view_and_current_team_holding_share_exact_inactive_asset_identity():
    ledger, universe, _projected = _price_universe()
    holdings = app_core.build_holding_asset_universe(
        _selectable_model(), ledger, "driver"
    )

    assert holdings.set_index("id").loc["114", "price"] == pytest.approx(13.0)
    assert universe.set_index("id").loc["114", "price"] == pytest.approx(13.0)
    assert holdings.set_index("id").loc["114", "holding_status"] == "Inactive"
    assert universe.set_index("id").loc["114", "playerId"] == 114


def test_inactive_constructor_is_price_visible_without_becoming_selectable():
    raw = [
        {
            "PlayerId": 50,
            "FUllName": "Old Constructor",
            "PositionName": "CONSTRUCTOR",
            "IsActive": 0,
            "Value": 20.0,
            "OldPlayerValue": 19.8,
        },
        {
            "PlayerId": 51,
            "FUllName": "Active Constructor",
            "PositionName": "CONSTRUCTOR",
            "IsActive": 1,
            "Value": 21.0,
            "OldPlayerValue": 20.8,
        },
    ]
    ledger = fantasy_api.normalise_constructor_asset_ledger(raw, feed_round=12)
    selectable = fantasy_api.selectable_constructor_assets(ledger).rename(
        columns={"teamId": "id"}
    )
    selectable["id"] = selectable["id"].astype(str)
    selectable["exp_score"] = 20.0
    selectable["next_race_expected_points"] = 20.0
    selectable["volatility"] = 5.0
    universe = app_core.build_price_change_asset_universe(
        selectable, ledger, "constructor"
    )
    projected = app_core.apply_probabilistic_price_change_model(
        universe,
        _rules(),
        predicted_points_col="next_race_expected_points",
    ).set_index("id")

    assert set(fantasy_api.selectable_constructor_assets(ledger)["teamId"]) == {51}
    assert set(universe["id"]) == {"50", "51"}
    assert projected.loc["50", "price"] == pytest.approx(20.0)
    assert pd.isna(projected.loc["50", "expected_price_gain"])
