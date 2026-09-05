from __future__ import annotations

from copy import deepcopy
from io import BytesIO

import pandas as pd
from pandas.testing import assert_frame_equal
from PIL import Image, ImageDraw

import f1fantasy.exports as exports
from f1fantasy.exports import (
    LANDSCAPE_SIZE,
    PORTRAIT_SIZE,
    PRICE_CHANGE_TABLE_WIDTH,
    TABLE_MAX_HEIGHT,
    contrast_text_colour,
    fit_text,
    price_change_table_export_plan,
    price_efficiency_table_export_plan,
    price_efficiency_team_export_plan,
    projected_team_export_plan,
    projected_team_layout,
    render_price_change_table_png,
    render_price_efficiency_table_png,
    render_price_efficiency_team_png,
    render_projected_team_png,
    resolve_export_font,
    safe_export_filename,
)


def _assets(asset_type: str, count: int) -> pd.DataFrame:
    rows = []
    for index in range(1, count + 1):
        driver = asset_type == "driver"
        rows.append(
            {
                "id": f"{asset_type[0]}{index}",
                "asset_id": f"{asset_type[0]}{index}",
                "asset_type": asset_type,
                "name": f"{'Driver' if driver else 'Constructor'} {index}",
                "full_name": f"{'Driver' if driver else 'Constructor'} {index}",
                "team_name": f"Team {index}",
                "abbreviation": f"{'D' if driver else 'C'}{index}",
                "team_colour": "#facc15" if index % 2 else "#172554",
                "price": 7.0 + index,
                "current_price": 7.0 + index,
                "exp_score": 9.0 + index,
                "expected_price_gain": 0.1 * index if index % 2 else -0.1 * index,
                "selected_points_total": 20.0 + index,
                "average_points_per_race": 10.0 + index / 2,
                "price_efficiency": (10.0 + index / 2) / (7.0 + index),
                "selected_race_count": 2,
                "valid_race_count": 2,
                "coverage_fraction": 1.0,
                "has_source_failure": False,
                "status": "complete",
            }
        )
    return pd.DataFrame(rows)


def _team() -> dict:
    return {
        "drivers": _assets("driver", 5),
        "constructors": _assets("constructor", 2),
        "boosted_driver": "Driver 2",
        "triple_driver": "Driver 1",
        "total_cost": 80.0,
        "limitless": False,
        "no_negative": False,
    }


def _summary(*, complete: bool = True) -> dict:
    return {
        "valid": complete,
        "messages": [] if complete else ["One or more assets have incomplete official coverage."],
        "total_cost": 80.0,
        "remaining_budget": 20.0,
        "total_selected_official_points": 175.0,
        "average_team_points_per_selected_race": 87.5 if complete else float("nan"),
        "sum_individual_asset_efficiencies": 8.75,
        "team_price_efficiency": 1.09375 if complete else float("nan"),
        "component_coverage": 1.0 if complete else 0.86,
    }


def _open_png(payload: bytes) -> Image.Image:
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    image = Image.open(BytesIO(payload))
    image.load()
    return image


def test_projected_team_renderer_returns_valid_portrait_and_landscape_pngs():
    team = _team()

    portrait = render_projected_team_png(
        team,
        title="Best Projected Team",
        subtitle="Last 3 · p=0.70 · Combined",
        budget=100.0,
    )
    landscape = render_projected_team_png(
        team,
        title="Alternative Team 2",
        budget=100.0,
        format="landscape",
    )

    assert _open_png(portrait).size == PORTRAIT_SIZE
    assert _open_png(landscape).size == LANDSCAPE_SIZE


def test_projected_team_handles_both_asset_types_markers_gains_and_missing_values():
    team = _team()
    team["drivers"].loc[0, "expected_price_gain"] = float("nan")
    team["drivers"].loc[1, "team_colour"] = "not-a-colour"
    team["constructors"].loc[0, "name"] = "A Constructor Name That Is Deliberately Much Longer Than The Export Card"

    payload = render_projected_team_png(
        team,
        title="Team with 2x and 3x",
        budget=None,
        expected_gain=None,
    )

    assert _open_png(payload).size == PORTRAIT_SIZE
    assert exports._format_number(None, "M", signed=True) == "—"
    assert exports._format_number(0.0, "M", signed=True) == "0.00M"
    assert exports._format_number(0.25, "M", signed=True) == "+0.25M"
    assert exports._format_number(-0.25, "M", signed=True) == "-0.25M"


def test_fit_text_never_exceeds_requested_width_for_long_identity():
    image = Image.new("RGB", (300, 100))
    draw = ImageDraw.Draw(image)

    text, font = fit_text(
        draw,
        "An exceptionally long driver or constructor identity that cannot fit",
        120,
        font_size=40,
        min_size=12,
        bold=True,
    )

    bounds = draw.textbbox((0, 0), text, font=font)
    assert bounds[2] - bounds[0] <= 120


def test_regular_and_bold_export_fonts_use_the_same_family_and_support_required_symbols():
    regular = resolve_export_font(20)
    bold = resolve_export_font(20, bold=True)

    assert regular.getname()[0] == bold.getname()[0]
    assert regular.getlength("—") > regular.getlength("-")
    assert regular.getlength("≤") > 0
    assert regular.getlength("≥") > 0


def test_contrast_helper_uses_readable_light_and_dark_text():
    assert contrast_text_colour("#ffffff") == "#111827"
    assert contrast_text_colour("#000000") == "#ffffff"
    assert contrast_text_colour("invalid") == "#ffffff"


def test_price_efficiency_team_png_handles_complete_and_incomplete_coverage():
    selected = pd.concat([_assets("driver", 5), _assets("constructor", 2)], ignore_index=True)
    incomplete = selected.copy(deep=True)
    incomplete.loc[0, ["valid_race_count", "coverage_fraction"]] = [1, 0.5]
    incomplete.loc[0, "status"] = "incomplete"

    complete_png = render_price_efficiency_team_png(
        _summary(),
        selected,
        "Last 3 · R12, R11, R9",
    )
    incomplete_png = render_price_efficiency_team_png(
        _summary(complete=False),
        incomplete,
        "Custom · R12, R9",
    )

    assert _open_png(complete_png).size == PORTRAIT_SIZE
    assert _open_png(incomplete_png).size == PORTRAIT_SIZE
    assert complete_png != incomplete_png
    assert exports._price_efficiency_warning(_summary(), selected.to_dict("records")) == ""
    assert "incomplete" in exports._price_efficiency_warning(
        _summary(complete=False), incomplete.to_dict("records")
    ).casefold()


def test_driver_and_constructor_table_exports_are_bounded_and_keep_display_order():
    drivers = pd.concat([_assets("driver", 5)] * 8, ignore_index=True)
    drivers["asset_id"] = [f"d{index}" for index in range(len(drivers))]
    constructors = _assets("constructor", 4)

    driver_png = render_price_efficiency_table_png(
        drivers,
        asset_type="driver",
        race_summary="All completed races",
        sort_label="Price Efficiency descending",
    )
    constructor_png = render_price_efficiency_table_png(
        constructors,
        asset_type="constructor",
        race_summary="Last 5",
        sort_label="Current price ascending",
    )

    driver_image = _open_png(driver_png)
    constructor_image = _open_png(constructor_png)
    assert [row["asset_id"] for row in exports.dataframe_rows_for_export(constructors)] == constructors[
        "asset_id"
    ].tolist()
    assert driver_image.width == 1080
    assert driver_image.height <= TABLE_MAX_HEIGHT
    assert constructor_image.width == 1080
    assert 380 <= constructor_image.height <= TABLE_MAX_HEIGHT


def test_table_export_preserves_unavailable_metrics_instead_of_zeroing_them():
    table = _assets("driver", 2)
    table.loc[0, ["selected_points_total", "average_points_per_race", "price_efficiency"]] = float("nan")
    table.loc[0, ["valid_race_count", "coverage_fraction"]] = [0, 0.0]
    table.loc[0, ["status", "has_source_failure"]] = ["source_failure", True]

    normalized = exports._table_row(table.iloc[0].to_dict())
    payload = render_price_efficiency_table_png(
        table,
        asset_type="driver",
        race_summary="Last 1",
        sort_label="Price Efficiency descending",
    )

    assert normalized["selected"] is None
    assert normalized["average"] is None
    assert normalized["efficiency"] is None
    assert normalized["incomplete"] is True
    assert _open_png(payload).width == 1080


def test_projected_team_plan_contains_only_compact_asset_drawing_data_and_no_race_metadata():
    team = _team()
    plan = projected_team_export_plan(
        team,
        title="Best Projected Team",
        budget=100.0,
    )
    drawing_data = repr(plan)

    for forbidden in [
        "Driver 1",
        "Constructor 1",
        "Team 1",
        "Ferrari",
        "All: R1 Australian Grand Prix",
    ]:
        assert forbidden not in drawing_data
    assert [card["abbreviation"] for card in plan["drivers"]] == ["D1", "D2", "D3", "D4", "D5"]
    assert [card["abbreviation"] for card in plan["constructors"]] == ["C1", "C2"]
    assert plan["drivers"][0]["price"] == "$8.0"
    assert plan["drivers"][0]["points"] == "30.0 Pts"
    assert plan["drivers"][0]["gain"] == "+0.10"
    assert plan["drivers"][0]["marker"] == "3x"
    assert {"PRICE", "EXPECTED POINTS", "EXPECTED GAIN"}.isdisjoint(plan["drivers"][0].values())
    assert [metric["label"] for metric in plan["summary"]] == ["Value", "Left", "Gain", "Pts"]
    assert all("Projected team value" != metric["label"] for metric in plan["summary"])


def test_team_renderer_ignores_long_subtitle_and_uses_neutral_zero_gain(monkeypatch):
    team = _team()
    team["drivers"].loc[0, "expected_price_gain"] = 0.0
    captured_subtitles = []
    original_header = exports._draw_header

    def capture_header(draw, width, title, subtitle, *, compact):
        captured_subtitles.append(subtitle)
        return original_header(draw, width, title, subtitle, compact=compact)

    monkeypatch.setattr(exports, "_draw_header", capture_header)
    payload = render_projected_team_png(
        team,
        title="Best Projected Team",
        subtitle="All: R1 Australian Grand Prix · p=0.85 · model blend details",
        budget=100.0,
    )
    plan = projected_team_export_plan(team, title="Best Projected Team", budget=100.0)

    assert _open_png(payload).size == PORTRAIT_SIZE
    assert captured_subtitles == [None]
    assert plan["drivers"][0]["gain"] == "0.00"
    assert plan["drivers"][0]["gain_colour"] == exports._MUTED


def test_projected_card_and_summary_bounds_stay_inside_both_image_formats():
    plan = projected_team_export_plan(_team(), title="Bounds", budget=100.0)

    for export_format in ["portrait", "landscape"]:
        layout = projected_team_layout(plan, export_format)
        width, height = layout["size"]
        all_bounds = [
            *layout["driver_cards"],
            *layout["constructor_cards"],
            *layout["summary"],
        ]
        assert all(0 <= left < right <= width for left, _, right, _ in all_bounds)
        assert all(0 <= top < bottom <= height for _, top, _, bottom in all_bounds)
        driver_sizes = {(right - left, bottom - top) for left, top, right, bottom in layout["driver_cards"]}
        constructor_sizes = {
            (right - left, bottom - top) for left, top, right, bottom in layout["constructor_cards"]
        }
        assert len(layout["driver_cards"]) == 5
        assert len(layout["constructor_cards"]) == 2
        assert driver_sizes == constructor_sizes
        assert len({top for _, top, _, _ in layout["driver_cards"]}) == 1
        constructor_row = layout["constructor_cards"]
        assert constructor_row[0][0] + constructor_row[-1][2] == width


def test_price_efficiency_export_plans_have_five_columns_no_names_or_metadata():
    drivers = _assets("driver", 3)
    plan = price_efficiency_table_export_plan(drivers, asset_type="driver")
    team_plan = price_efficiency_team_export_plan(
        _summary(),
        pd.concat([_assets("driver", 5), _assets("constructor", 2)], ignore_index=True),
    )

    assert plan["headers"] == ["Asset", "Selected pts", "Avg/race", "Price", "Pts/M"]
    assert "Coverage" not in plan["headers"]
    assert "Status" not in plan["headers"]
    drawing_data = repr((plan, team_plan))
    assert "Driver 1" not in drawing_data
    assert "Constructor 1" not in drawing_data
    assert "Team 1" not in drawing_data
    assert "race_summary" not in drawing_data


def test_price_efficiency_incomplete_footer_is_conditional_and_missing_values_stay_missing():
    complete = _assets("driver", 2)
    incomplete = complete.copy(deep=True)
    incomplete.loc[0, ["selected_points_total", "average_points_per_race", "price_efficiency"]] = float("nan")
    incomplete.loc[0, ["valid_race_count", "coverage_fraction"]] = [0, 0.0]
    incomplete.loc[0, ["status", "has_source_failure"]] = ["source_failure", True]

    complete_plan = price_efficiency_table_export_plan(complete, asset_type="driver")
    incomplete_plan = price_efficiency_table_export_plan(incomplete, asset_type="driver")

    assert complete_plan["footer"] == ""
    assert incomplete_plan["footer"] == "* Based on incomplete official race data"
    assert incomplete_plan["rows"][0]["selected"] is None
    assert incomplete_plan["rows"][0]["average"] is None
    assert incomplete_plan["rows"][0]["efficiency"] is None
    assert exports._format_number(incomplete_plan["rows"][0]["selected"]) == "—"


def test_price_efficiency_table_header_receives_no_race_or_sort_subtitle(monkeypatch):
    captured = []
    original_header = exports._draw_header

    def capture_header(draw, width, title, subtitle, *, compact):
        captured.append((title, subtitle))
        return original_header(draw, width, title, subtitle, compact=compact)

    monkeypatch.setattr(exports, "_draw_header", capture_header)
    render_price_efficiency_table_png(
        _assets("driver", 2),
        asset_type="driver",
        race_summary="Using 5 races: Monaco, Spain, Canada, Austria, Britain",
        sort_label="Price Efficiency descending",
    )
    render_price_efficiency_team_png(
        _summary(),
        pd.concat([_assets("driver", 5), _assets("constructor", 2)], ignore_index=True),
        "Using 5 races: Monaco, Spain, Canada, Austria, Britain",
    )

    assert captured == [
        ("Drivers Price Efficiency", None),
        ("Price Efficiency Team", None),
    ]


def test_price_change_threshold_and_projection_exports_are_name_free_and_coloured():
    threshold = pd.DataFrame(
        [
            {
                "Asset": '<span style="background:#14532d" title="Nico Hülkenberg — Audi">HUL</span>',
                "Price": 8.2,
                "Terrible": "≤ 3",
                "Poor": "4 to 8",
                "Good": "9 to 13",
                "Great": "≥ 14",
                "Rise difficulty": 1.707,
            }
        ]
    )
    projection = pd.DataFrame(
        [
            {
                "Asset": '<span style="background:#1e3a8a" title="Red Bull Racing">RBR</span>',
                "Price": 27.5,
                "P(Terrible)": 0.1,
                "P(Poor)": 0.2,
                "P(Good)": 0.3,
                "P(Great)": 0.4,
                "Expected Points": 31.5,
                "Expected price gain": -0.2,
            }
        ]
    )

    threshold_plan = price_change_table_export_plan(
        threshold,
        asset_type="driver",
        table_type="threshold",
    )
    projection_plan = price_change_table_export_plan(
        projection,
        asset_type="constructor",
        table_type="projection",
    )
    threshold_png = render_price_change_table_png(
        threshold,
        asset_type="driver",
        table_type="threshold",
    )
    projection_png = render_price_change_table_png(
        projection,
        asset_type="constructor",
        table_type="projection",
    )

    drawing_data = repr((threshold_plan, projection_plan))
    assert "HUL" in drawing_data
    assert "RBR" in drawing_data
    assert "Nico Hülkenberg" not in drawing_data
    assert "Audi" not in drawing_data
    assert "Red Bull Racing" not in drawing_data
    assert threshold_plan["headers"] == [
        "Asset", "Price", "Terrible", "Poor", "Good", "Great", "Rise difficulty"
    ]
    assert projection_plan["headers"] == [
        "Asset", "Price", "P(Terrible)", "P(Poor)", "P(Good)", "P(Great)",
        "Expected Points", "Expected price gain",
    ]
    assert _open_png(threshold_png).width == PRICE_CHANGE_TABLE_WIDTH
    assert _open_png(projection_png).width == PRICE_CHANGE_TABLE_WIDTH


def test_renderers_are_deterministic_and_do_not_mutate_inputs():
    team = _team()
    drivers_before = team["drivers"].copy(deep=True)
    constructors_before = team["constructors"].copy(deep=True)
    summary = _summary()
    summary_before = deepcopy(summary)
    selected = pd.concat([team["drivers"], team["constructors"]], ignore_index=True)
    selected_before = selected.copy(deep=True)

    first = render_projected_team_png(team, title="Deterministic", budget=100.0)
    second = render_projected_team_png(team, title="Deterministic", budget=100.0)
    render_price_efficiency_team_png(summary, selected, "Last 3")
    render_price_efficiency_table_png(
        selected,
        asset_type="driver",
        race_summary="Last 3",
        sort_label="Displayed order",
    )

    assert first == second
    assert_frame_equal(team["drivers"], drivers_before)
    assert_frame_equal(team["constructors"], constructors_before)
    assert_frame_equal(selected, selected_before)
    assert summary == summary_before


def test_safe_export_filename_removes_paths_unicode_and_duplicate_extension():
    assert safe_export_filename("../../F1 Best Tëam!!.PNG") == "f1_best_team.png"
    assert safe_export_filename("") == "f1_export.png"
    assert "/" not in safe_export_filename("folder/name")


def test_export_module_has_no_streamlit_dependency():
    assert "streamlit" not in exports.__dict__
