from __future__ import annotations

import math

import pandas as pd

from f1fantasy.app_core import (
    OBJECTIVE_COMBINED,
    OBJECTIVE_POINTS_ONLY,
    OBJECTIVE_PRICE_GROWTH_ONLY,
    fantasy_asset_card_html,
    ranked_team_component_html,
)
from f1fantasy.race_selection import RaceKey, RaceOption, recency_weights
from f1fantasy.ui_helpers import (
    asset_constraint_transition,
    asset_abbreviation,
    compact_asset_identity_html,
    contrast_text_colour,
    effective_blend_percentages,
    compact_asset_payload,
    compact_asset_table_html,
    compact_asset_universe_rows,
    format_compact_gain,
    format_compact_price,
    gain_value_class,
    next_team_batch,
    normalize_price_growth_value,
    optimise_mobile_subview,
    optimiser_result_signature,
    primary_navigation_state,
    prepare_price_efficiency_display,
    price_efficiency_race_summary,
    price_efficiency_table_html,
    race_weight_summary,
    ranked_solution_current_team_update,
    reconcile_constraint_pair,
    reconcile_price_efficiency_team_state,
    reconcile_race_control_state,
    resolve_objective_mode,
    resolve_price_efficiency_asset_type,
    responsive_layout_mode,
    sort_projection_assets,
    sprint_diagnostic_table_html,
    team_summary_payload,
)


def _races(rounds=(1, 3, 5)):
    return tuple(RaceOption(RaceKey(2026, round_no), f"Race {round_no}") for round_no in rounds)


def test_race_ui_state_uses_production_presets_custom_and_exclusions():
    preset = reconcile_race_control_state(_races(), "Last 3", excluded_keys=[(2026, 3)])
    custom = reconcile_race_control_state(
        _races(),
        "Custom",
        custom_keys=[(2026, 1), (2026, 5)],
        excluded_keys=[(2026, 1)],
    )

    assert preset.selection.included == (RaceKey(2026, 1), RaceKey(2026, 5))
    assert preset.selection.excluded == (RaceKey(2026, 3),)
    assert custom.custom_keys == (RaceKey(2026, 1), RaceKey(2026, 5))
    assert custom.selection.included == (RaceKey(2026, 5),)


def test_race_ui_state_handles_empty_selection_and_refresh_removals():
    empty = reconcile_race_control_state(_races(), "Custom", custom_keys=[])
    refreshed = reconcile_race_control_state(
        _races((1, 5)),
        "Custom",
        custom_keys=[(2026, 1), (2026, 3), (2026, 5)],
        excluded_keys=[(2026, 3), (2026, 5)],
    )

    assert empty.selection.included == ()
    assert refreshed.custom_keys == (RaceKey(2026, 1), RaceKey(2026, 5))
    assert refreshed.excluded_keys == (RaceKey(2026, 5),)
    assert refreshed.removed_custom_keys == (RaceKey(2026, 3),)
    assert refreshed.removed_excluded_keys == (RaceKey(2026, 3),)


def test_race_weight_summary_is_newest_first_and_has_contiguous_weights():
    state = reconcile_race_control_state(_races((9, 11, 12)), "All")
    weights = recency_weights(state.selection, 0.5)

    assert race_weight_summary(state.selection, weights) == "R12: 1.00 · R11: 0.50 · R9: 0.25"


def test_price_efficiency_race_summary_reports_inclusions_exclusions_and_empty_state():
    names = {race.key: race.race_name for race in _races()}
    all_races = reconcile_race_control_state(_races(), "All")
    excluded = reconcile_race_control_state(_races(), "Last 3", excluded_keys=[(2026, 3)])
    empty = reconcile_race_control_state(_races(), "Custom", custom_keys=[])

    assert price_efficiency_race_summary(all_races.selection, names) == (
        "Using 3 races: Race 1, Race 3, Race 5"
    )
    assert price_efficiency_race_summary(excluded.selection, names) == (
        "Using 2 of the last 3 races: Race 1, Race 5 · Race 3 excluded"
    )
    assert price_efficiency_race_summary(empty.selection, names) == "No races selected"


def test_relative_blend_percentages_and_zero_case_are_clear():
    assert effective_blend_percentages(2.0, 1.0) == (67, 33)
    assert effective_blend_percentages(0.0, 0.0) == (50, 50)


def test_objective_state_defaults_combined_preserves_saved_and_forces_limitless_points():
    allowed = [OBJECTIVE_COMBINED, OBJECTIVE_POINTS_ONLY, OBJECTIVE_PRICE_GROWTH_ONLY]

    assert resolve_objective_mode(
        None, allowed=allowed, default=OBJECTIVE_COMBINED, points_only=OBJECTIVE_POINTS_ONLY
    ) == OBJECTIVE_COMBINED
    assert resolve_objective_mode(
        OBJECTIVE_PRICE_GROWTH_ONLY,
        allowed=allowed,
        default=OBJECTIVE_COMBINED,
        points_only=OBJECTIVE_POINTS_ONLY,
    ) == OBJECTIVE_PRICE_GROWTH_ONLY
    assert resolve_objective_mode(
        OBJECTIVE_COMBINED,
        allowed=allowed,
        default=OBJECTIVE_COMBINED,
        force_points_only=True,
        points_only=OBJECTIVE_POINTS_ONLY,
    ) == OBJECTIVE_POINTS_ONLY


def test_price_growth_slider_state_defaults_rounds_clamps_and_stays_integer():
    assert normalize_price_growth_value(None) == 50
    assert normalize_price_growth_value(float("nan")) == 50
    assert normalize_price_growth_value(52.4) == 50
    assert normalize_price_growth_value(52.5) == 55
    assert normalize_price_growth_value(-12) == 0
    assert normalize_price_growth_value(103) == 100
    assert normalize_price_growth_value("35") == 35
    assert isinstance(normalize_price_growth_value(50.0), int)


def test_price_efficiency_active_table_state_defaults_to_drivers():
    assert resolve_price_efficiency_asset_type(None) == "Drivers"
    assert resolve_price_efficiency_asset_type("Drivers") == "Drivers"
    assert resolve_price_efficiency_asset_type("Constructors") == "Constructors"
    assert resolve_price_efficiency_asset_type("retired") == "Drivers"


def test_abbreviation_colour_contrast_and_accessible_identity_fallbacks():
    driver = {
        "asset_type": "driver",
        "full_name": "Nico Hülkenberg",
        "team_name": "Audi",
        "team_colour": "#f8fafc",
    }
    constructor = {
        "asset_type": "constructor",
        "full_name": "Aston Martin",
        "team_name": "Aston Martin",
        "team_colour": "#15803d",
    }

    assert asset_abbreviation(driver) == "HÜL"
    assert asset_abbreviation(constructor) == "AM"
    assert contrast_text_colour("#ffffff") == "#111827"
    assert contrast_text_colour("#111111") == "#ffffff"
    identity = compact_asset_identity_html(driver)
    assert "Nico Hülkenberg — Audi" in identity
    assert 'aria-label="Driver: Nico Hülkenberg — Audi"' in identity
    assert 'role="img"' in identity
    assert "HÜL" in identity
    assert '<span class="f1-asset-name">Nico Hülkenberg</span>' in identity
    assert '<span class="f1-asset-team">Audi</span>' in identity


def _efficiency_rows():
    return pd.DataFrame(
        [
            {
                "asset_id": "d1",
                "asset_type": "driver",
                "abbreviation": "ONE",
                "full_name": "Driver One",
                "team_name": "Ferrari",
                "team_colour": "#dc2626",
                "current_price": 10.0,
                "selected_points_total": 30.0,
                "average_points_per_race": 15.0,
                "price_efficiency": 1.5,
                "selected_race_count": 2,
                "valid_race_count": 2,
                "coverage_fraction": 1.0,
                "has_source_failure": False,
                "status": "complete",
            },
            {
                "asset_id": "d2",
                "asset_type": "driver",
                "abbreviation": "TWO",
                "full_name": "Driver Two",
                "team_name": "Audi",
                "team_colour": "#14532d",
                "current_price": 8.0,
                "selected_points_total": 10.0,
                "average_points_per_race": 10.0,
                "price_efficiency": 1.25,
                "selected_race_count": 2,
                "valid_race_count": 1,
                "coverage_fraction": 0.5,
                "has_source_failure": False,
                "status": "incomplete",
            },
            {
                "asset_id": "d3",
                "asset_type": "driver",
                "abbreviation": "MIS",
                "full_name": "Missing Driver",
                "team_name": "Team",
                "team_colour": "#64748b",
                "current_price": 7.0,
                "selected_points_total": math.nan,
                "average_points_per_race": math.nan,
                "price_efficiency": math.nan,
                "selected_race_count": 2,
                "valid_race_count": 0,
                "coverage_fraction": 0.0,
                "has_source_failure": True,
                "status": "source_failure",
            },
        ]
    )


def test_efficiency_display_uses_payload_values_statuses_and_unavailable_last():
    display = prepare_price_efficiency_display(_efficiency_rows())

    assert display["asset_id"].tolist() == ["d1", "d2", "d3"]
    assert display.loc[0, "Points per million"] == 1.5
    assert display.loc[1, "Status"] == "◐ Limited coverage"
    assert display.loc[2, "Status"] == "⚠ Source failure"
    assert pd.isna(display.loc[2, "Points per million"])
    rendered = price_efficiency_table_html(display)
    assert "ONE" in rendered
    assert "⚠ Source failure" in rendered
    assert ">—<" in rendered
    assert ">0.00<" not in rendered.split("MIS", 1)[-1]


def test_efficiency_sorting_supports_price_and_coverage_without_lifting_missing_rows():
    by_price = prepare_price_efficiency_display(_efficiency_rows(), "Current price", ascending=True)
    by_coverage = prepare_price_efficiency_display(_efficiency_rows(), "Coverage", ascending=False)

    assert by_price["asset_id"].tolist() == ["d3", "d2", "d1"]
    assert by_coverage["asset_id"].tolist() == ["d1", "d2", "d3"]


def test_team_builder_state_is_isolated_from_optimizer_budget_and_reconciles_assets():
    existing = {
        "driver_ids": ["d1", "gone", "d1", "d2"],
        "constructor_ids": ["c1", "gone"],
        "budget": 110.0,
    }
    state = reconcile_price_efficiency_team_state(existing, ["d1", "d2"], ["c1"], 123.0)
    defaulted = reconcile_price_efficiency_team_state({}, ["d1"], ["c1"], 123.0)

    assert state == {"driver_ids": ["d1", "d2"], "constructor_ids": ["c1"], "budget": 110.0}
    assert defaulted["budget"] == 123.0
    assert "optimizer_budget" not in state


def test_compact_card_has_identity_price_points_gain_and_multiplier_without_redundancy():
    card = fantasy_asset_card_html(
        {
            "name": "Kimi Antonelli",
            "abbreviation": "ANT",
            "team": "Mercedes",
            "team_colour": "#14b8a6",
            "price": 25.7,
            "exp_score": 85.76,
            "expected_price_gain": 0.24,
        },
        boosted_token="3x",
        asset_label="Driver",
    )

    assert "ANT" in card
    assert "$25.7" in card
    assert "85.8 Pts" in card
    assert "+0.24" in card
    assert ">3x<" in card
    assert 'title="Kimi Antonelli — Mercedes"' in card
    assert '<span class="f1-asset-name">Kimi Antonelli</span>' in card
    assert '<span class="f1-asset-team">Mercedes</span>' in card
    assert '<span class="f1-card-value">$25.7M</span>' in card
    assert '<span class="f1-card-label">Price gain</span>' in card
    assert 'class="f1-initials"' not in card


def test_shared_compact_asset_payload_is_exact_equal_shape_and_immutable():
    driver = {
        "name": "Kimi Antonelli",
        "abbreviation": "ANT",
        "team": "Mercedes",
        "team_colour": "#14b8a6",
        "price": 25.7,
        "exp_score": 60.9,
        "expected_price_gain": 0.18,
    }
    constructor = {
        "name": "Ferrari",
        "abbreviation": "FER",
        "team_colour": "#dc2626",
        "price": 26.6,
        "exp_score": 61.3,
        "expected_price_gain": 0.30,
    }
    original_driver = dict(driver)
    original_constructor = dict(constructor)

    driver_payload = compact_asset_payload(driver, asset_type="driver", marker="2x")
    constructor_payload = compact_asset_payload(constructor, asset_type="constructor")

    assert set(driver_payload) == set(constructor_payload)
    assert driver_payload["abbreviation"] == "ANT"
    assert driver_payload["price"] == "$25.7"
    assert driver_payload["gain"] == "+0.18"
    assert driver_payload["points"] == "60.9 Pts"
    assert driver_payload["marker"] == "2x"
    assert constructor_payload["abbreviation"] == "FER"
    assert "Kimi Antonelli" not in " ".join(
        str(driver_payload[key]) for key in ("abbreviation", "price", "gain", "points", "marker")
    )
    assert driver == original_driver
    assert constructor == original_constructor


def test_dense_asset_table_has_only_asset_price_gain_and_points_with_shared_gain_state():
    assets = pd.DataFrame(
        [
            {
                "name": "Kimi Antonelli",
                "abbreviation": "ANT",
                "team": "Mercedes",
                "price": 25.7,
                "exp_score": 60.9,
                "expected_price_gain": -0.03,
            }
        ]
    )
    original = assets.copy(deep=True)

    rendered = compact_asset_table_html(assets, asset_type="driver")

    assert all(f">{heading}</th>" in rendered for heading in ("Asset", "Price ($M)", "Gain ($M)", "Points"))
    assert ">Name</th>" not in rendered
    assert ">Team</th>" not in rendered
    assert "ANT" in rendered
    assert "-0.03" in rendered
    assert "f1-gain-negative" in rendered
    pd.testing.assert_frame_equal(assets, original)


def test_projection_sort_uses_unrounded_numbers_with_stable_ties_and_missing_last():
    assets = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "price": ["2", "10", "2", None, "unavailable"],
            "expected_price_gain": [0.05, -0.1, 0.05, None, 0.2],
            "exp_score": [1.04, 1.03, 1.04, float("inf"), -1.0],
        },
        index=[7, 7, 2, 1, 1],
    )
    original = assets.copy(deep=True)
    cases = (
        ("Price", False, [1, 0, 2, 3, 4]),
        ("Price", True, [0, 2, 1, 3, 4]),
        ("Price gain", False, [4, 0, 2, 1, 3]),
        ("Price gain", True, [1, 0, 2, 4, 3]),
        ("Projected points", False, [0, 2, 1, 4, 3]),
        ("Projected points", True, [4, 1, 0, 2, 3]),
    )
    for metric, ascending, positions in cases:
        result = sort_projection_assets(assets, metric, ascending=ascending)
        pd.testing.assert_frame_equal(result, original.iloc[positions])
    pd.testing.assert_frame_equal(assets, original)


def test_projection_sort_matches_display_fallback_columns_and_returns_independent_copy():
    assets = pd.DataFrame(
        {
            "current_price": [10.0, 20.0],
            "expected_price_change": [0.2, -0.1],
            "Expected Points": [1.0, 20.0],
        }
    )
    assert sort_projection_assets(assets, "Price").index.tolist() == [1, 0]
    assert sort_projection_assets(assets, "Price gain").index.tolist() == [0, 1]
    assert sort_projection_assets(assets, "Projected points").index.tolist() == [1, 0]
    result = sort_projection_assets(assets, "outdated saved label")
    result.iloc[0, 0] = 99.0
    assert assets["current_price"].tolist() == [10.0, 20.0]
    assert sort_projection_assets(None).empty
    empty = assets.iloc[:0]
    pd.testing.assert_frame_equal(sort_projection_assets(empty), empty)


def test_optimiser_asset_universe_rows_have_six_compact_columns_and_exact_values():
    assets = pd.DataFrame(
        [
            {
                "id": "d1",
                "name": "Kimi Antonelli",
                "abbreviation": "ANT",
                "team": "Mercedes",
                "price": 25.7,
                "exp_score": 60.9,
                "expected_price_gain": -0.03,
            }
        ]
    )
    before = assets.copy(deep=True)

    rows = compact_asset_universe_rows(
        assets,
        asset_type="driver",
        locked_ids=["d1"],
        excluded_ids=["d1"],
    )

    assert len(rows) == 1
    assert set(rows[0]) >= {
        "asset", "price", "gain", "points", "lock", "exclude"
    }
    assert rows[0]["price_value"] == 25.7
    assert rows[0]["gain_value"] == -0.03
    assert rows[0]["points_value"] == 60.9
    assert rows[0]["lock"] is True
    assert rows[0]["exclude"] is False
    assert '<span class="f1-asset-name">Kimi Antonelli</span>' in rows[0]["asset"]
    pd.testing.assert_frame_equal(assets, before)


def test_inline_and_bulk_constraint_transitions_share_mutually_exclusive_state():
    locked = asset_constraint_transition(
        {}, asset_type="driver", asset_id="d1", action="lock", active=True
    )
    excluded = asset_constraint_transition(
        locked, asset_type="driver", asset_id="d1", action="exclude", active=True
    )
    constructors = asset_constraint_transition(
        excluded, asset_type="constructor", asset_id="c1", action="lock", active=True
    )
    primary, conflict = reconcile_constraint_pair(["d2", "d1"], ["d1", "d3"])

    assert locked["locked_driver_ids"] == ["d1"]
    assert locked["excluded_driver_ids"] == []
    assert excluded["locked_driver_ids"] == []
    assert excluded["excluded_driver_ids"] == ["d1"]
    assert constructors["locked_constructor_ids"] == ["c1"]
    assert (primary, conflict) == (["d2", "d1"], ["d3"])


def test_ranked_solution_current_team_update_is_atomic_and_stable_id_only():
    solution = _compact_solution(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
    )
    drivers_before = solution["drivers"].copy(deep=True)
    constructors_before = solution["constructors"].copy(deep=True)
    current_state = {
        "current_team_bank": 4.2,
        "current_team_free_transfers": 3,
        "optimizer_budget": 118.0,
        "transfer_history": ["kept"],
    }

    result = ranked_solution_current_team_update(
        solution,
        valid_driver_ids=["d1", "d2", "d3", "d4", "d5"],
        valid_constructor_ids=["c1", "c2"],
    )
    updated = {**current_state, **result["updates"]}
    missing = ranked_solution_current_team_update(
        solution,
        valid_driver_ids=["d1", "d2", "d3", "d4"],
        valid_constructor_ids=["c1", "c2"],
    )

    assert result == {
        "ok": True,
        "updates": {
            "current_team_driver_ids": ["d1", "d2", "d3", "d4", "d5"],
            "current_team_constructor_ids": ["c1", "c2"],
        },
        "error": None,
    }
    assert updated["current_team_bank"] == 4.2
    assert updated["current_team_free_transfers"] == 3
    assert updated["optimizer_budget"] == 118.0
    assert updated["transfer_history"] == ["kept"]
    assert missing["ok"] is False
    assert missing["updates"] == {}
    pd.testing.assert_frame_equal(solution["drivers"], drivers_before)
    pd.testing.assert_frame_equal(solution["constructors"], constructors_before)


def test_compact_values_and_gain_states_cover_positive_negative_zero_and_missing():
    assert format_compact_price(25.7) == "$25.7"
    assert format_compact_price(float("nan")) == "—"
    assert [format_compact_gain(value) for value in (0.18, -0.03, 0.0, float("nan"))] == [
        "+0.18",
        "-0.03",
        "0.00",
        "—",
    ]
    assert [gain_value_class(value) for value in (0.18, -0.03, 0.0, float("nan"))] == [
        "f1-gain-positive",
        "f1-gain-negative",
        "f1-gain-neutral",
        "f1-gain-missing",
    ]


def test_team_summary_is_exactly_four_compact_values_with_signed_gain():
    summary = team_summary_payload(
        total_cost=114.4,
        budget=117.0,
        expected_gain=1.69,
        expected_points=188.22,
    )

    assert [item["label"] for item in summary.values()] == ["Value", "Left", "Gain", "Pts"]
    assert [item["value"] for item in summary.values()] == ["114.4M", "2.6M", "+1.69M", "188.2"]
    assert summary["gain"]["class"] == "f1-gain-positive"


def _compact_solution(driver_ids, constructor_ids):
    return {
        "drivers": pd.DataFrame({"id": driver_ids}),
        "constructors": pd.DataFrame({"id": constructor_ids}),
    }


def test_next_team_batch_appends_unique_teams_without_replacing_existing():
    first = _compact_solution(["d1", "d2", "d3", "d4", "d5"], ["c1", "c2"])
    second = _compact_solution(["d1", "d2", "d3", "d4", "d6"], ["c1", "c2"])
    result = next_team_batch([first], [first, second], batch_size=10)

    assert result["solutions"][0] is first
    assert result["appended"] == [second]
    assert result["next_label"] == "Load teams 3–12"
    assert result["exhausted"] is True


def test_optimiser_signature_ignores_navigation_and_layout_state():
    base = optimiser_result_signature(
        data_version=("v1",),
        budget=117,
        chip_mode="none",
        price_growth_value=50,
        locked_driver_ids=["2", "1"],
    )
    after_display_only_change = optimiser_result_signature(
        data_version=("v1",),
        budget=117,
        chip_mode="none",
        price_growth_value=50,
        locked_driver_ids=["1", "2"],
    )

    assert base == after_display_only_change
    assert primary_navigation_state("market") == "Market"
    assert primary_navigation_state("unknown") == "Optimise"
    assert responsive_layout_mode("mobile") == "mobile"
    assert responsive_layout_mode("anything") == "auto"


def test_mobile_optimise_subview_defaults_and_normalises_without_model_state():
    assert optimise_mobile_subview(None) == "Teams"
    assert optimise_mobile_subview("drivers") == "Drivers"
    assert optimise_mobile_subview("CONSTRUCTORS") == "Constructors"
    assert optimise_mobile_subview("controls") == "Controls"
    assert optimise_mobile_subview("unknown") == "Teams"


def test_mobile_projection_and_efficiency_schemas_keep_critical_columns_visible():
    assets = pd.DataFrame(
        [
            {
                "id": "d1",
                "name": "Driver One",
                "abbreviation": "ONE",
                "team": "Ferrari",
                "price": 10.0,
                "expected_price_gain": 0.2,
                "exp_score": 12.5,
            }
        ]
    )
    projection = compact_asset_table_html(assets, asset_type="driver")
    efficiency = price_efficiency_table_html(
        pd.DataFrame(
            [
                {
                    "asset_identity_html": '<span class="f1-asset-id">ONE</span>',
                    "Current price": 10.0,
                    "Average/race": 11.0,
                    "Points per million": 1.1,
                    "Status": "Complete",
                }
            ]
        )
    )

    assert 'class="f1-compact-table f1-mobile-schema f1-projection-mobile"' in projection
    assert all(f">{heading}</th>" in projection for heading in ("Asset", "Price ($M)", "Gain ($M)", "Points"))
    assert 'title="Expected points"' in projection
    assert '>EV</th>' not in projection
    assert 'class="f1-compact-table f1-mobile-schema f1-efficiency-mobile"' in efficiency
    assert "<th>Asset</th><th>Price</th><th>Pts/M</th>" in efficiency
    assert "Avg/race" in efficiency  # retained in the desktop schema


def test_mobile_sprint_schema_exposes_base_bonus_and_final_without_mutation():
    frame = pd.DataFrame(
        [
            {
                "id": "d1",
                "name": "Driver One",
                "abbreviation": "ONE",
                "team": "Ferrari",
                "baseline_expected_points": 10.0,
                "sprint_bonus": 2.5,
                "next_race_expected_points": 12.5,
            }
        ]
    )
    before = frame.copy(deep=True)

    rendered = sprint_diagnostic_table_html(frame)

    pd.testing.assert_frame_equal(frame, before)
    assert "<th>Asset</th><th>Base</th><th>+Sprint</th><th>Final</th>" in rendered
    assert ">+2.50<" in rendered


def test_ranked_team_renderer_is_uniform_and_has_five_plus_two_grouped_assets():
    drivers = pd.DataFrame(
        [
            {
                "id": f"d{index}",
                "name": f"Driver {index}",
                "abbreviation": f"D{index}",
                "team": "Ferrari",
                "price": 20 - index,
                "exp_score": 10 + index,
                "expected_price_gain": -0.03 if index == 1 else 0.18,
            }
            for index in range(1, 6)
        ]
    )
    constructors = pd.DataFrame(
        [
            {
                "id": f"c{index}",
                "name": f"Constructor {index}",
                "abbreviation": f"C{index}",
                "price": 15 - index,
                "exp_score": 20 + index,
                "expected_price_gain": 0.30,
            }
            for index in range(1, 3)
        ]
    )
    summary = {
        "Total cost": 100.0,
        "Remaining budget": 17.0,
        "Expected price gain": 1.2,
        "Expected points": 188.2,
    }

    team_one = ranked_team_component_html(
        rank=1, summary=summary, drivers=drivers, constructors=constructors
    )
    team_two = ranked_team_component_html(
        rank=2, summary=summary, drivers=drivers, constructors=constructors
    )

    assert 'data-rank="1"' in team_one
    assert 'data-rank="2"' in team_two
    assert team_one.replace("team 1", "team 2").replace(">1<", ">2<").replace(
        'data-rank="1"', 'data-rank="2"'
    ) == team_two
    assert team_one.count('class="f1-driver-card"') == 7
    assert 'class="f1-card-grid f1-driver-grid"' in team_one
    assert 'class="f1-card-grid f1-constructor-grid"' in team_one
    assert ">Drivers<" in team_one
    assert ">Constructors<" in team_one
    assert ">Team cost<" in team_one
    assert ">Budget left<" in team_one
    assert ">Expected pts<" in team_one
    assert "f1-gain-negative" in team_one
