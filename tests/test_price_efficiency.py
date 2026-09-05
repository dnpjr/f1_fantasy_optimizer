from __future__ import annotations

import math

import pandas as pd
import pytest

from f1fantasy.price_efficiency import (
    build_price_efficiency_table,
    summarize_price_efficiency_team,
)
from f1fantasy.race_selection import (
    RaceKey,
    RaceOption,
    recency_weights,
    resolve_selected_races,
)


def _key(round_number: int) -> RaceKey:
    return RaceKey(2026, round_number)


def _selection(preset="All", custom=None, excluded=None):
    available = tuple(RaceOption(_key(round_number), f"Race {round_number}") for round_number in range(1, 6))
    return resolve_selected_races(
        available,
        preset,
        custom_keys=custom,
        excluded_keys=excluded,
    )


def _driver_roster() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "d1",
                "name": "Driver One",
                "abbreviation": "ONE",
                "team": "Ferrari",
                "team_colour": "#f00",
                "price": 10.0,
                "recent_points_source": "playerstats",
            },
            {
                "id": "d2",
                "name": "Replacement Driver",
                "abbreviation": "REP",
                "team": "Cadillac",
                "team_colour": "#111",
                "price": 8.0,
                "recent_points_source": "playerstats",
            },
            {
                "id": "d3",
                "name": "Missing Driver",
                "abbreviation": "MIS",
                "team": "Audi",
                "team_colour": "#222",
                "price": 7.0,
                "recent_points_source": "playerstats_failed",
            },
        ]
    )


def _driver_observations() -> pd.DataFrame:
    rows = []
    points = [0.0, 10.0, 20.0, 30.0, 40.0]
    for round_number, fantasy_points in enumerate(points, start=1):
        rows.append(
            {
                "PlayerId": "d1",
                "asset_type": "driver",
                "season": 2026,
                "round": round_number,
                "fantasy_points": fantasy_points,
                "price": 999.0,
                "is_played": 1,
            }
        )
    rows.extend(
        [
            {
                "PlayerId": "d2",
                "asset_type": "driver",
                "season": 2026,
                "round": 4,
                "fantasy_points": 12.0,
                "price": 1.0,
                "is_played": 1,
            },
            {
                "PlayerId": "d2",
                "asset_type": "driver",
                "season": 2026,
                "round": 5,
                "fantasy_points": 20.0,
                "price": 1.0,
                "is_played": 1,
            },
        ]
    )
    return pd.DataFrame(rows)


def test_driver_efficiency_uses_current_price_and_ordinary_selected_mean():
    table = build_price_efficiency_table(
        _driver_roster(),
        _driver_observations(),
        _selection(),
        weights=recency_weights(_selection(), 0.5),
        asset_type="driver",
    )

    driver = table[table["asset_id"] == "d1"].iloc[0]
    assert driver["selected_points_total"] == 100.0
    assert driver["average_points_per_race"] == 20.0
    assert driver["price_efficiency"] == 2.0
    assert driver["current_price"] == 10.0
    assert driver["valid_race_count"] == 5
    assert driver["status"] == "complete"
    assert driver["weighted_average_points_per_race"] != driver["average_points_per_race"]


def test_zero_point_race_counts_and_historical_prices_are_ignored():
    observations = _driver_observations()
    observations.loc[observations["PlayerId"] == "d1", "price"] = [1, 2, 3, 4, 5]
    table = build_price_efficiency_table(
        _driver_roster(), observations, _selection("Last 1", custom=None), asset_type="driver"
    )

    driver = table[table["asset_id"] == "d1"].iloc[0]
    assert driver["average_points_per_race"] == 40.0
    assert driver["valid_race_count"] == 1
    assert driver["price_efficiency"] == 4.0


def test_replacement_driver_normalizes_valid_races_and_exposes_coverage():
    table = build_price_efficiency_table(
        _driver_roster(), _driver_observations(), _selection(), asset_type="driver"
    )

    replacement = table[table["asset_id"] == "d2"].iloc[0]
    assert replacement["selected_points_total"] == 32.0
    assert replacement["average_points_per_race"] == 16.0
    assert replacement["price_efficiency"] == 2.0
    assert replacement["valid_race_count"] == 2
    assert replacement["missing_race_count"] == 3
    assert replacement["coverage_fraction"] == pytest.approx(0.4)
    assert replacement["status"] == "incomplete"


def test_missing_active_asset_remains_visible_and_source_failure_is_flagged():
    table = build_price_efficiency_table(
        _driver_roster(), _driver_observations(), _selection(), asset_type="driver"
    )

    missing = table[table["asset_id"] == "d3"].iloc[0]
    assert pd.isna(missing["price_efficiency"])
    assert missing["valid_race_count"] == 0
    assert missing["missing_race_count"] == 5
    assert missing["has_source_failure"]
    assert missing["status"] == "source_failure"


@pytest.mark.parametrize(
    "preset,custom,excluded,expected_total",
    [
        ("Last 1", None, None, 40.0),
        ("Last 3", None, None, 90.0),
        ("Last 5", None, None, 100.0),
        ("Custom", [_key(1), _key(3)], None, 20.0),
        ("Last 5", None, [_key(5)], 60.0),
    ],
)
def test_presets_custom_and_exclusions_change_selected_totals(preset, custom, excluded, expected_total):
    selection = _selection(preset, custom=custom, excluded=excluded)
    table = build_price_efficiency_table(
        _driver_roster(), _driver_observations(), selection, asset_type="driver"
    )

    driver = table[table["asset_id"] == "d1"].iloc[0]
    assert driver["selected_points_total"] == expected_total
    assert driver["selected_race_count"] == len(selection.included)


def test_constructor_uses_identical_formula_and_metadata():
    roster = pd.DataFrame(
        [{"id": "c1", "name": "Ferrari", "price": 20.0, "team_colour": "#f00"}]
    )
    observations = pd.DataFrame(
        [
            {"PlayerId": "c1", "asset_type": "constructor", "season": 2026, "round": 1, "fantasy_points": 30, "is_played": 1},
            {"PlayerId": "c1", "asset_type": "constructor", "season": 2026, "round": 2, "fantasy_points": 50, "is_played": 1},
        ]
    )
    selection = resolve_selected_races(
        [RaceOption(_key(1), "One"), RaceOption(_key(2), "Two")], "All"
    )

    table = build_price_efficiency_table(roster, observations, selection, asset_type="constructor")

    assert table.loc[0, "asset_type"] == "constructor"
    assert table.loc[0, "team_name"] == "Ferrari"
    assert table.loc[0, "selected_points_total"] == 80.0
    assert table.loc[0, "average_points_per_race"] == 40.0
    assert table.loc[0, "price_efficiency"] == 2.0


def test_team_summary_recalculates_from_local_race_selection_and_current_prices():
    drivers = pd.DataFrame(
        [
            {"id": f"d{index}", "name": f"Driver {index}", "price": 10.0}
            for index in range(1, 6)
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": f"c{index}", "name": f"Constructor {index}", "price": 10.0}
            for index in range(1, 3)
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "PlayerId": asset_id,
                "asset_type": asset_type,
                "season": 2026,
                "round": round_number,
                "fantasy_points": points,
                "is_played": 1,
            }
            for asset_type, asset_ids in [
                ("driver", drivers["id"]),
                ("constructor", constructors["id"]),
            ]
            for asset_id in asset_ids
            for round_number, points in [(1, 10.0), (2, 20.0)]
        ]
    )
    available = [RaceOption(_key(1), "One"), RaceOption(_key(2), "Two")]

    def summary_for(preset):
        selection = resolve_selected_races(available, preset)
        table = pd.concat(
            [
                build_price_efficiency_table(
                    drivers,
                    observations,
                    selection,
                    asset_type="driver",
                ),
                build_price_efficiency_table(
                    constructors,
                    observations,
                    selection,
                    asset_type="constructor",
                ),
            ],
            ignore_index=True,
        )
        return summarize_price_efficiency_team(
            table,
            drivers["id"],
            constructors["id"],
            budget=100.0,
        )

    all_races = summary_for("All")
    last_race = summary_for("Last 1")

    assert all_races["total_selected_official_points"] == 210.0
    assert last_race["total_selected_official_points"] == 140.0
    assert all_races["average_team_points_per_selected_race"] == 105.0
    assert last_race["average_team_points_per_selected_race"] == 140.0
    assert all_races["total_cost"] == last_race["total_cost"] == 70.0
    assert all_races["team_price_efficiency"] == 1.5
    assert last_race["team_price_efficiency"] == 2.0


def test_price_efficiency_does_not_mutate_inputs():
    roster = _driver_roster()
    observations = _driver_observations()
    roster_original = roster.copy(deep=True)
    observations_original = observations.copy(deep=True)

    build_price_efficiency_table(roster, observations, _selection(), asset_type="driver")

    pd.testing.assert_frame_equal(roster, roster_original)
    pd.testing.assert_frame_equal(observations, observations_original)


def _team_table(incomplete=False) -> pd.DataFrame:
    rows = []
    for asset_type, ids in [("driver", [f"d{i}" for i in range(1, 6)]), ("constructor", ["c1", "c2"])]:
        for asset_id in ids:
            rows.append(
                {
                    "asset_id": asset_id,
                    "asset_type": asset_type,
                    "current_price": 10.0,
                    "selected_points_total": 20.0,
                    "average_points_per_race": 10.0,
                    "price_efficiency": 1.0,
                    "selected_race_count": 2,
                    "valid_race_count": 1 if incomplete and asset_id == "d1" else 2,
                    "missing_race_count": 1 if incomplete and asset_id == "d1" else 0,
                    "coverage_fraction": 0.5 if incomplete and asset_id == "d1" else 1.0,
                    "status": "incomplete" if incomplete and asset_id == "d1" else "complete",
                }
            )
    return pd.DataFrame(rows)


def test_valid_team_summary_reports_cost_budget_points_coverage_and_two_efficiency_forms():
    summary = summarize_price_efficiency_team(
        _team_table(),
        [f"d{i}" for i in range(1, 6)],
        ["c1", "c2"],
        budget=100.0,
    )

    assert summary["valid"]
    assert summary["messages"] == []
    assert summary["total_cost"] == 70.0
    assert summary["remaining_budget"] == 30.0
    assert summary["total_selected_official_points"] == 140.0
    assert summary["average_points_per_valid_asset_race"] == 10.0
    assert summary["average_team_points_per_selected_race"] == 70.0
    assert summary["sum_individual_asset_efficiencies"] == 7.0
    assert summary["team_price_efficiency"] == 1.0
    assert summary["team_price_efficiency_race_denominator"] == 2
    assert summary["component_coverage"] == 1.0


def test_team_summary_rejects_duplicates_wrong_composition_and_over_budget():
    duplicate = summarize_price_efficiency_team(
        _team_table(), ["d1", "d1", "d2", "d3", "d4"], ["c1", "c2"], 100.0
    )
    wrong = summarize_price_efficiency_team(
        _team_table(), ["d1", "d2"], ["c1"], 100.0
    )
    over = summarize_price_efficiency_team(
        _team_table(), [f"d{i}" for i in range(1, 6)], ["c1", "c2"], 60.0
    )

    assert not duplicate["valid"]
    assert any("duplicates" in message for message in duplicate["messages"])
    assert not wrong["valid"]
    assert any("exactly five" in message for message in wrong["messages"])
    assert any("exactly two" in message for message in wrong["messages"])
    assert not over["valid"]
    assert any("over budget" in message for message in over["messages"])


def test_incomplete_team_data_warns_and_does_not_report_false_complete_efficiency():
    summary = summarize_price_efficiency_team(
        _team_table(incomplete=True),
        [f"d{i}" for i in range(1, 6)],
        ["c1", "c2"],
        100.0,
    )

    assert not summary["valid"]
    assert summary["component_coverage"] == pytest.approx(13 / 14)
    assert math.isnan(summary["team_price_efficiency"])
    assert any("incomplete" in message for message in summary["messages"])
