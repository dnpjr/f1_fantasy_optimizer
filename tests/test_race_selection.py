from __future__ import annotations

import pandas as pd
import pytest

from f1fantasy.race_selection import (
    RaceKey,
    RaceOption,
    available_races,
    canonical_race_key,
    recency_weights,
    resolve_selected_races,
    weighted_asset_points,
)


def _key(round_number: int, season: int = 2026) -> RaceKey:
    return canonical_race_key(season, round_number)


def _options(rounds: list[int], season: int = 2026) -> tuple[RaceOption, ...]:
    return tuple(RaceOption(_key(round_number, season), f"Race {round_number}") for round_number in rounds)


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PlayerId": 1,
                "asset_type": "driver",
                "season": 2026,
                "round": 1,
                "race_name": "Australian Grand Prix",
                "fantasy_points": 10.0,
                "is_played": 1,
            },
            {
                "PlayerId": 1,
                "asset_type": "driver",
                "season": 2026,
                "round": 2,
                "race_name": "Chinese Grand Prix",
                "fantasy_points": 20.0,
                "is_played": 1,
            },
            {
                "PlayerId": 1,
                "asset_type": "driver",
                "season": 2026,
                "round": 3,
                "race_name": "Japanese Grand Prix",
                "fantasy_points": 30.0,
                "is_played": 1,
            },
        ]
    )


def test_canonical_identity_uses_season_and_round_not_display_metadata():
    first = canonical_race_key("2026", 3.0)
    same = RaceOption(first, "One display name")
    renamed = RaceOption(first, "A different display name")

    assert first == RaceKey(season=2026, round=3)
    assert same.key == renamed.key
    assert canonical_race_key(2025, 3) != first


@pytest.mark.parametrize("season,round_number", [(0, 1), (2026, 0), (2026, 1.5), ("bad", 1)])
def test_canonical_identity_rejects_invalid_values(season, round_number):
    with pytest.raises(ValueError, match="positive integer"):
        canonical_race_key(season, round_number)


def test_available_races_collapses_duplicates_and_resolves_names_deterministically():
    observations = pd.DataFrame(
        [
            {"season": 2026, "round": 2, "race_name": "Zulu Name", "fantasy_points": 5, "is_played": 1},
            {"season": 2026, "round": 2, "race_name": "Alpha Name", "fantasy_points": 8, "is_played": 1},
            {"season": 2025, "round": 2, "race_name": "Prior Season", "fantasy_points": 3, "is_played": 1},
        ]
    )

    races = available_races(observations)

    assert [option.key for option in races] == [_key(2, 2025), _key(2, 2026)]
    assert races[1].race_name == "Alpha Name"


def test_available_races_is_chronological_and_omits_unplayed_or_incomplete_rows():
    observations = pd.DataFrame(
        [
            {"season": 2026, "round": 4, "race_name": "Future", "fantasy_points": 15, "is_played": 0},
            {"season": 2026, "round": 3, "race_name": "Incomplete", "fantasy_points": pd.NA, "is_played": 1},
            {"season": 2026, "round": 2, "race_name": "Second", "fantasy_points": 0, "is_played": 1},
            {"season": 2026, "round": 1, "race_name": "First", "fantasy_points": 7, "is_played": 1},
        ]
    )
    original = observations.copy(deep=True)

    races = available_races(observations, season=2026)

    assert [option.key for option in races] == [_key(1), _key(2)]
    assert races[1].race_name == "Second"
    pd.testing.assert_frame_equal(observations, original)


def test_available_races_accepts_completed_driver_constructor_union():
    observations = pd.DataFrame(
        [
            {"asset_type": "driver", "season": 2026, "round": 1, "fantasy_points": 10, "is_played": 1},
            {"asset_type": "constructor", "season": 2026, "round": 2, "fantasy_points": 20, "is_played": 1},
            {"asset_type": "driver", "season": 2025, "round": 2, "fantasy_points": 30, "is_played": 1},
        ]
    )

    races = available_races(observations, season=2026)

    assert [option.key for option in races] == [_key(1), _key(2)]


@pytest.mark.parametrize(
    "preset,expected_rounds",
    [
        ("Last 1", [7]),
        ("Last 3", [5, 6, 7]),
        ("Last 5", [3, 4, 5, 6, 7]),
        ("All", [1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_presets_select_latest_available_races_in_chronological_order(preset, expected_rounds):
    selection = resolve_selected_races(_options([7, 2, 1, 6, 4, 3, 5]), preset)

    assert [key.round for key in selection.included] == expected_rounds


def test_custom_uses_only_available_keys_and_ignores_unknowns_and_duplicates():
    selection = resolve_selected_races(
        _options([1, 2, 4]),
        "Custom",
        custom_keys=[_key(4), _key(99), _key(1), _key(4)],
    )

    assert selection.included == (_key(1), _key(4))


def test_exclusions_apply_after_presets_and_are_duplicate_safe():
    last_three = resolve_selected_races(
        _options([1, 2, 3, 4, 5]),
        "Last 3",
        excluded_keys=[_key(4), _key(4), _key(99)],
    )
    last_five = resolve_selected_races(
        _options([1, 2, 3, 4, 5]),
        "Last 5",
        excluded_keys=[_key(3)],
    )

    assert last_three.included == (_key(3), _key(5))
    assert last_three.excluded == (_key(4),)
    assert last_five.included == (_key(1), _key(2), _key(4), _key(5))


def test_exclusions_apply_after_custom_resolution():
    selection = resolve_selected_races(
        _options([1, 2, 3]),
        "Custom",
        custom_keys=[_key(1), _key(3)],
        excluded_keys=[_key(1)],
    )

    assert selection.included == (_key(3),)
    assert selection.excluded == (_key(1),)


def test_empty_selection_and_unsupported_preset_are_deliberate():
    assert resolve_selected_races((), "Last 3").included == ()
    assert resolve_selected_races(_options([1]), "Custom", custom_keys=[]).included == ()
    with pytest.raises(ValueError, match="Unsupported race preset"):
        resolve_selected_races(_options([1]), "Latest")


def test_non_consecutive_rounds_receive_contiguous_recency_exponents():
    keys = [_key(12), _key(9), _key(11)]

    weights = recency_weights(keys, 0.5)

    assert weights == {_key(9): 0.25, _key(11): 0.5, _key(12): 1.0}


def test_excluding_middle_race_recloses_recency_positions():
    selection = resolve_selected_races(
        _options([8, 9, 10, 11, 12]),
        "Last 5",
        excluded_keys=[_key(10)],
    )

    weights = recency_weights(selection, 0.5)

    assert weights == {_key(8): 0.125, _key(9): 0.25, _key(11): 0.5, _key(12): 1.0}


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.0, [0.0, 0.0, 1.0]),
        (1.0, [1.0, 1.0, 1.0]),
        (0.5, [0.25, 0.5, 1.0]),
    ],
)
def test_recency_boundary_and_midpoint_values(p, expected):
    weights = recency_weights([_key(1), _key(2), _key(3)], p)

    assert list(weights.values()) == expected


@pytest.mark.parametrize("p", [-0.01, 1.01, float("nan")])
def test_recency_rejects_values_outside_closed_unit_interval(p):
    with pytest.raises(ValueError, match="between 0 and 1"):
        recency_weights([_key(1)], p)


def test_weighted_points_preserve_genuine_zero_and_normalize_valid_values():
    observations = _observations()
    observations.loc[observations["round"] == 1, "fantasy_points"] = 0.0
    selected = [_key(1), _key(2), _key(3)]
    weights = recency_weights(selected, 0.5)

    result = weighted_asset_points(observations, selected, weights).iloc[0]

    assert result["weighted_points"] == pytest.approx((0 * 0.25 + 20 * 0.5 + 30) / 1.75)
    assert result["valid_race_count"] == 3
    assert result["missing_race_count"] == 0
    assert result["coverage_fraction"] == 1.0
    assert result["status"] == "complete"


def test_missing_and_unplayed_observations_are_not_converted_to_zero():
    observations = _observations()
    observations.loc[observations["round"] == 1, "fantasy_points"] = pd.NA
    observations.loc[observations["round"] == 2, "is_played"] = 0
    selected = [_key(1), _key(2), _key(3)]

    result = weighted_asset_points(observations, selected, recency_weights(selected, 0.5)).iloc[0]

    assert result["weighted_points"] == 30.0
    assert result["valid_race_keys"] == (_key(3),)
    assert result["missing_race_keys"] == (_key(1), _key(2))
    assert result["coverage_fraction"] == pytest.approx(1 / 3)
    assert result["status"] == "incomplete"


def test_replacement_driver_uses_only_two_valid_races_and_reports_two_of_five_coverage():
    observations = pd.DataFrame(
        [
            {"PlayerId": 22, "season": 2026, "round": 4, "fantasy_points": 8.0, "is_played": 1},
            {"PlayerId": 22, "season": 2026, "round": 5, "fantasy_points": 12.0, "is_played": 1},
        ]
    )
    selected = [_key(round_number) for round_number in range(1, 6)]

    result = weighted_asset_points(
        observations,
        selected,
        recency_weights(selected, 0.5),
        asset_type="driver",
    ).iloc[0]

    assert result["weighted_points"] == pytest.approx((8 * 0.5 + 12) / 1.5)
    assert result["selected_race_count"] == 5
    assert result["valid_race_count"] == 2
    assert result["missing_race_count"] == 3
    assert result["coverage_fraction"] == pytest.approx(0.4)
    assert result["status"] == "incomplete"


def test_driver_and_constructor_observations_share_identity_and_weighting_semantics():
    observations = pd.DataFrame(
        [
            {"PlayerId": 1, "asset_type": "driver", "season": 2026, "round": 1, "fantasy_points": 5, "is_played": 1},
            {"PlayerId": 1, "asset_type": "driver", "season": 2026, "round": 2, "fantasy_points": 15, "is_played": 1},
            {"PlayerId": 1, "asset_type": "constructor", "season": 2026, "round": 1, "fantasy_points": 20, "is_played": 1},
            {"PlayerId": 1, "asset_type": "constructor", "season": 2026, "round": 2, "fantasy_points": 40, "is_played": 1},
        ]
    )
    selected = [_key(1), _key(2)]

    result = weighted_asset_points(observations, selected, recency_weights(selected, 0.5))

    driver = result[result["asset_type"] == "driver"].iloc[0]
    constructor = result[result["asset_type"] == "constructor"].iloc[0]
    assert driver["weighted_points"] == pytest.approx((5 * 0.5 + 15) / 1.5)
    assert constructor["weighted_points"] == pytest.approx((20 * 0.5 + 40) / 1.5)
    assert driver["valid_race_keys"] == constructor["valid_race_keys"] == (_key(1), _key(2))


def test_source_failures_are_explicit_even_when_an_asset_has_no_observation_rows():
    selected = [_key(1), _key(2)]
    observations = pd.DataFrame(
        [{"PlayerId": 1, "season": 2026, "round": 2, "fantasy_points": 10, "is_played": 1}]
    )

    result = weighted_asset_points(
        observations,
        selected,
        recency_weights(selected, 0.5),
        asset_type="driver",
        source_failures=[("driver", "99")],
    )

    failed = result[result["asset_id"] == "99"].iloc[0]
    assert failed["has_source_failure"]
    assert failed["status"] == "source_failure"
    assert failed["valid_race_count"] == 0
    assert failed["coverage_fraction"] == 0.0


def test_row_level_source_failure_is_not_silently_reported_as_ordinary_incomplete_coverage():
    observations = _observations()
    observations["source_failed"] = True
    selected = [_key(1), _key(2), _key(3)]

    result = weighted_asset_points(observations, selected, recency_weights(selected, 1.0)).iloc[0]

    assert result["weighted_points"] == 20.0
    assert result["has_source_failure"]
    assert result["status"] == "source_failure"


def test_empty_observations_and_empty_selections_are_predictable():
    empty = weighted_asset_points(pd.DataFrame(), [_key(1)], {_key(1): 1.0})
    no_selection = weighted_asset_points(_observations(), [], {})

    assert empty.empty
    assert list(empty.columns) == [
        "asset_id",
        "asset_type",
        "weighted_points",
        "selected_race_count",
        "valid_race_count",
        "missing_race_count",
        "coverage_fraction",
        "weight_sum",
        "selected_race_keys",
        "valid_race_keys",
        "missing_race_keys",
        "has_source_failure",
        "status",
    ]
    assert no_selection.iloc[0]["status"] == "no_races_selected"
    assert no_selection.iloc[0]["coverage_fraction"] == 0.0
    assert pd.isna(no_selection.iloc[0]["weighted_points"])


def test_weighted_calculation_and_selection_do_not_mutate_inputs():
    observations = _observations()
    original_observations = observations.copy(deep=True)
    available = list(_options([1, 2, 3]))
    custom = [_key(1), _key(3), _key(3)]
    excluded = [_key(1)]
    original_available = list(available)
    original_custom = list(custom)
    original_excluded = list(excluded)
    selection = resolve_selected_races(available, "Custom", custom, excluded)
    weights = recency_weights(selection, 0.5)
    original_weights = dict(weights)

    weighted_asset_points(observations, selection, weights)

    pd.testing.assert_frame_equal(observations, original_observations)
    assert available == original_available
    assert custom == original_custom
    assert excluded == original_excluded
    assert weights == original_weights


def test_duplicate_asset_race_rows_do_not_apply_the_same_race_weight_twice():
    observations = pd.DataFrame(
        [
            {"PlayerId": 1, "season": 2026, "round": 1, "fantasy_points": 10, "is_played": 1},
            {"PlayerId": 1, "season": 2026, "round": 1, "fantasy_points": 20, "is_played": 1},
            {"PlayerId": 1, "season": 2026, "round": 2, "fantasy_points": 30, "is_played": 1},
        ]
    )
    selected = [_key(1), _key(2)]

    result = weighted_asset_points(observations, selected, recency_weights(selected, 1.0)).iloc[0]

    assert result["weighted_points"] == pytest.approx((15 + 30) / 2)
    assert result["valid_race_count"] == 2
