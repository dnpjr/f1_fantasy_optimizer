import pandas as pd
import pytest

from f1fantasy.app_core import team_expected_points_with_chips
from f1fantasy.optimize import optimize_top_k


def _drivers():
    return pd.DataFrame(
        [
            {"id": "d1", "name": "Driver 1", "price": 10.0, "exp_score": 50.0},
            {"id": "d2", "name": "Driver 2", "price": 10.0, "exp_score": 45.0},
            {"id": "d3", "name": "Driver 3", "price": 10.0, "exp_score": 40.0},
            {"id": "d4", "name": "Driver 4", "price": 10.0, "exp_score": 35.0},
            {"id": "d5", "name": "Driver 5", "price": 10.0, "exp_score": 30.0},
            {"id": "d6", "name": "Driver 6", "price": 10.0, "exp_score": 5.0},
        ]
    )


def _constructors():
    return pd.DataFrame(
        [
            {"id": "c1", "name": "Constructor 1", "price": 10.0, "exp_score": 40.0},
            {"id": "c2", "name": "Constructor 2", "price": 10.0, "exp_score": 35.0},
            {"id": "c3", "name": "Constructor 3", "price": 10.0, "exp_score": 1.0},
        ]
    )


def test_locked_assets_are_selected():
    sol = optimize_top_k(
        _drivers(),
        _constructors(),
        budget=100.0,
        k=1,
        locked_driver_ids=["d6"],
        locked_constructor_ids=["c3"],
    )[0]

    assert "d6" in set(sol.drivers["id"])
    assert "c3" in set(sol.constructors["id"])


def test_excluded_assets_are_not_selected():
    sol = optimize_top_k(
        _drivers(),
        _constructors(),
        budget=100.0,
        k=1,
        excluded_driver_ids=["d1"],
        excluded_constructor_ids=["c1"],
    )[0]

    assert "d1" not in set(sol.drivers["id"])
    assert "c1" not in set(sol.constructors["id"])


def test_conflicting_lock_and_exclude_raises():
    with pytest.raises(ValueError, match="both locked and excluded"):
        optimize_top_k(
            _drivers(),
            _constructors(),
            budget=100.0,
            k=1,
            locked_driver_ids=["d1"],
            excluded_driver_ids=["d1"],
        )


def test_too_many_locked_drivers_raises():
    with pytest.raises(ValueError, match="more than 5 drivers"):
        optimize_top_k(
            _drivers(),
            _constructors(),
            budget=100.0,
            k=1,
            locked_driver_ids=["d1", "d2", "d3", "d4", "d5", "d6"],
        )


def test_none_mode_applies_one_2x_boosted_driver():
    sol = optimize_top_k(_drivers(), _constructors(), budget=100.0, k=1, drs_multiplier=2.0)[0]

    assert sol.boosted_driver == "Driver 1"
    assert sol.triple_driver is None
    assert sol.expected_score == pytest.approx(50 + 45 + 40 + 35 + 30 + 40 + 35 + 50)
    assert team_expected_points_with_chips(sol.drivers, sol.constructors, "none", sol.boosted_driver) == pytest.approx(sol.expected_score)


def test_triple_chip_applies_distinct_2x_and_3x_drivers():
    sol = optimize_top_k(_drivers(), _constructors(), budget=100.0, k=1, drs_multiplier=2.0, triple_multiplier=3.0)[0]

    assert sol.boosted_driver == "Driver 2"
    assert sol.triple_driver == "Driver 1"
    assert sol.boosted_driver != sol.triple_driver
    base = 50 + 45 + 40 + 35 + 30 + 40 + 35
    assert sol.expected_score == pytest.approx(base + 45 + 2 * 50)
    assert team_expected_points_with_chips(sol.drivers, sol.constructors, "triple", sol.boosted_driver, sol.triple_driver) == pytest.approx(sol.expected_score)


def test_limitless_ignores_budget_constraint():
    sol = optimize_top_k(_drivers(), _constructors(), budget=60.0, k=1)
    limitless = optimize_top_k(_drivers(), _constructors(), budget=None, k=1)

    assert sol == []
    assert limitless[0].total_cost == pytest.approx(70.0)
    assert limitless[0].limitless is True
    assert team_expected_points_with_chips(limitless[0].drivers, limitless[0].constructors, "limitless", limitless[0].boosted_driver) == pytest.approx(limitless[0].expected_score)


def test_top_k_can_continue_after_existing_team_combinations_without_duplicates():
    initial = optimize_top_k(_drivers(), _constructors(), budget=100.0, k=2)
    excluded = [
        (
            solution.drivers["id"].astype(str).tolist(),
            solution.constructors["id"].astype(str).tolist(),
        )
        for solution in initial
    ]

    continued = optimize_top_k(
        _drivers(),
        _constructors(),
        budget=100.0,
        k=2,
        excluded_team_combinations=excluded,
    )

    initial_keys = {
        (
            tuple(sorted(solution.drivers["id"].astype(str))),
            tuple(sorted(solution.constructors["id"].astype(str))),
        )
        for solution in initial
    }
    continued_keys = {
        (
            tuple(sorted(solution.drivers["id"].astype(str))),
            tuple(sorted(solution.constructors["id"].astype(str))),
        )
        for solution in continued
    }
    assert continued_keys
    assert initial_keys.isdisjoint(continued_keys)
