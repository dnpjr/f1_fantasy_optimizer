from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
from io import BytesIO
import math
import re
import unicodedata
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from f1fantasy.ui_helpers import (
    asset_abbreviation,
    contrast_text_colour,
    format_compact_gain,
    format_compact_points,
    format_compact_price,
    gain_value_class,
    normalize_hex_colour,
)


PORTRAIT_SIZE = (1080, 1350)
LANDSCAPE_SIZE = (1600, 900)
TABLE_WIDTH = 1080
TABLE_MAX_HEIGHT = 2200
PRICE_CHANGE_TABLE_WIDTH = 1600

_BACKGROUND = "#090d16"
_PANEL = "#151b27"
_PANEL_ALT = "#111722"
_BORDER = "#30394a"
_TEXT = "#f8fafc"
_MUTED = "#aab4c4"
_ACCENT = "#e10600"
_POSITIVE = "#4ade80"
_NEGATIVE = "#fb7185"
_NEUTRAL_COLOUR = "#64748b"


def resolve_export_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Resolve a deployment-safe font without requiring a bundled asset."""
    size = max(8, int(size))
    preferred_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    fallback_name = "DejaVuSans.ttf" if bold else "DejaVuSans-Bold.ttf"
    candidates = [
        (f"/usr/share/fonts/truetype/dejavu/{preferred_name}", 0),
        (f"/usr/local/share/fonts/{preferred_name}", 0),
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
        (f"/usr/share/fonts/truetype/dejavu/{fallback_name}", 0),
        (preferred_name, 0),
        (fallback_name, 0),
    ]
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size=size, index=index)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None or value is pd.NA:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return fallback if text.casefold() in {"", "nan", "none", "<na>"} else text


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _record_value(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            value = record.get(key)
            if _number(value) is not None or _clean_text(value):
                return value
    return None


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bounds = draw.textbbox((0, 0), text, font=font)
    return max(0, int(bounds[2] - bounds[0]))


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: Any,
    max_width: int,
    *,
    font_size: int = 32,
    min_size: int = 14,
    bold: bool = False,
) -> tuple[str, ImageFont.ImageFont]:
    """Return fitted text and font, shrinking then truncating when necessary."""
    value = _clean_text(text, "—")
    max_width = max(1, int(max_width))
    for size in range(max(int(font_size), int(min_size)), int(min_size) - 1, -1):
        font = resolve_export_font(size, bold=bold)
        if _text_width(draw, value, font) <= max_width:
            return value, font
    font = resolve_export_font(min_size, bold=bold)
    suffix = "..."
    if _text_width(draw, suffix, font) > max_width:
        return "", font
    candidate = value
    while candidate and _text_width(draw, candidate + suffix, font) > max_width:
        candidate = candidate[:-1]
    return candidate.rstrip() + suffix, font


def draw_rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    *,
    radius: int = 20,
    fill: str = _PANEL,
    outline: str | None = _BORDER,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(
        tuple(int(value) for value in bounds),
        radius=max(0, int(radius)),
        fill=fill,
        outline=outline,
        width=max(1, int(width)),
    )


def draw_asset_identity(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    asset: Mapping[str, Any] | pd.Series,
    *,
    show_name: bool = False,
) -> tuple[int, int, int, int]:
    """Draw a compact coloured abbreviation badge.

    Exported images intentionally never draw full asset or team names. The
    retained ``show_name`` keyword is ignored for backward compatibility with
    callers of the earlier helper.
    """
    record = asset.to_dict() if isinstance(asset, pd.Series) else dict(asset)
    left, top, right, bottom = (int(value) for value in bounds)
    colour = normalize_hex_colour(record.get("team_colour"), _NEUTRAL_COLOUR)
    abbreviation = asset_abbreviation(record, _clean_text(record.get("asset_type")))
    pill_width = min(max(86, 34 + len(abbreviation) * 23), min(136, right - left))
    pill_bottom = min(bottom, top + 50)
    draw_rounded_rectangle(
        draw,
        (left, top, left + pill_width, pill_bottom),
        radius=13,
        fill=colour,
        outline=None,
    )
    label, label_font = fit_text(
        draw,
        abbreviation,
        pill_width - 20,
        font_size=28,
        min_size=17,
        bold=True,
    )
    label_bounds = draw.textbbox((0, 0), label, font=label_font)
    label_height = label_bounds[3] - label_bounds[1]
    draw.text(
        (left + pill_width / 2, top + (pill_bottom - top - label_height) / 2 - label_bounds[1]),
        label,
        font=label_font,
        fill=contrast_text_colour(colour),
        anchor="ma",
    )
    return left, top, left + pill_width, pill_bottom


def _format_number(value: Any, suffix: str = "", *, signed: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "—"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.2f}{suffix}"


def _asset_export_identity(
    record: Mapping[str, Any],
    asset_type: str | None = None,
) -> tuple[str, str]:
    """Return abbreviation and colour without retaining display names."""
    abbreviation = asset_abbreviation(record, asset_type or _clean_text(record.get("asset_type")))
    colour = normalize_hex_colour(record.get("team_colour"), _NEUTRAL_COLOUR)
    compact_markup = _clean_text(record.get("Asset"))
    if "<" in compact_markup and ">" in compact_markup:
        badge_match = re.search(r">\s*([^<>]+?)\s*</span>", compact_markup, flags=re.IGNORECASE)
        colour_match = re.search(
            r"background\s*:\s*(#[0-9a-fA-F]{3,8})",
            compact_markup,
            flags=re.IGNORECASE,
        )
        if badge_match:
            abbreviation = html.unescape(badge_match.group(1)).strip().upper()[:5]
        if colour_match:
            colour = normalize_hex_colour(colour_match.group(1), _NEUTRAL_COLOUR)
    elif compact_markup and abbreviation == "?":
        abbreviation = compact_markup.upper()[:5]
    return abbreviation, colour


def draw_metric(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    label: str,
    value: Any,
    *,
    value_colour: str = _TEXT,
) -> None:
    left, top, right, bottom = (int(item) for item in bounds)
    draw_rounded_rectangle(draw, (left, top, right, bottom), radius=16, fill=_PANEL_ALT)
    label_text, label_font = fit_text(
        draw,
        label.upper(),
        right - left - 28,
        font_size=17,
        min_size=12,
        bold=True,
    )
    value_text, value_font = fit_text(
        draw,
        value,
        right - left - 28,
        font_size=31,
        min_size=16,
        bold=True,
    )
    draw.text((left + 14, top + 14), label_text, font=label_font, fill=_MUTED)
    draw.text((left + 14, bottom - 18), value_text, font=value_font, fill=value_colour, anchor="ls")


def dataframe_rows_for_export(
    dataframe: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Copy export rows while preserving the caller's displayed order."""
    if dataframe is None:
        return []
    if isinstance(dataframe, pd.DataFrame):
        copied = dataframe.copy(deep=True)
        if max_rows is not None:
            copied = copied.head(max(0, int(max_rows)))
        return [dict(row) for row in copied.to_dict(orient="records")]
    rows = [dict(row) for row in dataframe]
    return rows if max_rows is None else rows[: max(0, int(max_rows))]


def image_to_png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def safe_export_filename(value: Any, default: str = "f1_export") -> str:
    text = unicodedata.normalize("NFKD", _clean_text(value, default))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\.png$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    stem = (text or default)[:88].rstrip("_")
    return f"{stem}.png"


def _draw_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    title: str,
    subtitle: str | None,
    *,
    compact: bool,
) -> int:
    margin = 52 if compact else 64
    draw.rounded_rectangle((margin, 44, margin + 12, 134 if compact else 156), radius=6, fill=_ACCENT)
    app_font = resolve_export_font(19 if compact else 22, bold=True)
    title_text, title_font = fit_text(
        draw,
        title,
        width - 2 * margin - 50,
        font_size=42 if compact else 50,
        min_size=24,
        bold=True,
    )
    draw.text((margin + 30, 45), "F1 FANTASY OPTIMISER", font=app_font, fill=_ACCENT)
    draw.text((margin + 30, 77), title_text, font=title_font, fill=_TEXT)
    next_y = 144 if compact else 166
    if subtitle:
        subtitle_text, subtitle_font = fit_text(
            draw,
            subtitle,
            width - 2 * margin,
            font_size=19 if compact else 22,
            min_size=14,
        )
        draw.text((margin, next_y), subtitle_text, font=subtitle_font, fill=_MUTED)
        next_y += 34
    return next_y + 16


def _records_from_team(team: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    drivers = dataframe_rows_for_export(_object_value(team, "drivers", []))
    constructors = dataframe_rows_for_export(_object_value(team, "constructors", []))
    for record in drivers:
        record.setdefault("asset_type", "driver")
    for record in constructors:
        record.setdefault("asset_type", "constructor")
    return drivers, constructors


def _row_price(record: Mapping[str, Any]) -> float | None:
    return _number(_record_value(record, "price", "current_price", "Current price"))


def _row_points(record: Mapping[str, Any]) -> float | None:
    return _number(
        _record_value(
            record,
            "display_exp_score",
            "next_race_expected_points",
            "next_race_exp_score",
            "exp_score",
            "Expected points",
        )
    )


def _row_gain(record: Mapping[str, Any]) -> float | None:
    return _number(
        _record_value(record, "expected_price_gain", "expected_price_change", "Expected price gain")
    )


def _matching_asset(record: Mapping[str, Any], identity: Any) -> bool:
    wanted = _clean_text(identity).casefold()
    if not wanted:
        return False
    return wanted in {
        _clean_text(record.get("name")).casefold(),
        _clean_text(record.get("full_name")).casefold(),
        _clean_text(record.get("id")).casefold(),
    }


def _team_totals(
    team: Any,
    drivers: Sequence[Mapping[str, Any]],
    constructors: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    records = [*drivers, *constructors]
    prices = [_row_price(record) for record in records]
    cost = _number(_object_value(team, "total_cost"))
    if cost is None and prices and all(value is not None for value in prices):
        cost = float(sum(value for value in prices if value is not None))

    boosted = _object_value(team, "boosted_driver")
    tripled = _object_value(team, "triple_driver")
    points: list[float] = []
    points_complete = bool(records)
    for record in records:
        value = _row_points(record)
        if value is None:
            points_complete = False
            continue
        if _clean_text(record.get("asset_type")).casefold() == "driver" and "display_exp_score" not in record:
            if _matching_asset(record, tripled):
                value *= 3.0
            elif _matching_asset(record, boosted):
                value *= 2.0
        points.append(value)
    expected_points = float(sum(points)) if points_complete else None

    gains = [_row_gain(record) for record in records]
    expected_gain = (
        float(sum(value for value in gains if value is not None))
        if gains and all(value is not None for value in gains)
        else None
    )
    return cost, expected_points, expected_gain


def _chip_summary(team: Any) -> str:
    parts: list[str] = []
    if _clean_text(_object_value(team, "triple_driver")):
        parts.append("3X CHIP")
    if bool(_object_value(team, "limitless", False)):
        parts.append("LIMITLESS")
    if bool(_object_value(team, "no_negative", False)):
        parts.append("NO NEGATIVE")
    return " · ".join(parts)


def _gain_colour(value: Any) -> str:
    return {
        "f1-gain-positive": _POSITIVE,
        "f1-gain-negative": _NEGATIVE,
    }.get(gain_value_class(value), _MUTED)


def projected_team_export_plan(
    team: Any,
    *,
    title: str,
    budget: float | None = None,
    expected_points: float | None = None,
    expected_gain: float | None = None,
) -> dict[str, Any]:
    """Build name-free drawing data for projected-team PNGs."""
    drivers, constructors = _records_from_team(team)
    cost, derived_points, derived_gain = _team_totals(team, drivers, constructors)
    total_points = _number(expected_points) if expected_points is not None else derived_points
    total_gain = _number(expected_gain) if expected_gain is not None else derived_gain
    limitless = bool(_object_value(team, "limitless", False))
    if limitless:
        total_gain = None
    numeric_budget = _number(budget)
    remaining = numeric_budget - cost if numeric_budget is not None and cost is not None else None
    boosted = _object_value(team, "boosted_driver")
    tripled = _object_value(team, "triple_driver")

    def cards(records: Sequence[Mapping[str, Any]], asset_type: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for record in records:
            abbreviation, colour = _asset_export_identity(record, asset_type)
            marker = ""
            if asset_type == "driver" and _matching_asset(record, tripled):
                marker = "3x"
            elif asset_type == "driver" and _matching_asset(record, boosted):
                marker = "2x"
            point_value = _row_points(record)
            if point_value is not None and "display_exp_score" not in record:
                if marker == "3x":
                    point_value *= 3.0
                elif marker == "2x":
                    point_value *= 2.0
            gain = _row_gain(record)
            output.append(
                {
                    "asset_type": asset_type,
                    "abbreviation": abbreviation,
                    "team_colour": colour,
                    "marker": marker,
                    "price": format_compact_price(_row_price(record)),
                    "points": format_compact_points(point_value),
                    "gain": format_compact_gain(gain),
                    "gain_colour": _gain_colour(gain),
                }
            )
        return output

    def compact_money(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f}M"

    def compact_points(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f}"

    return {
        "app_title": "F1 FANTASY OPTIMISER",
        "title": _clean_text(title, "Projected Team"),
        "chip_marker": _chip_summary(team),
        "drivers": cards(drivers[:5], "driver"),
        "constructors": cards(constructors[:2], "constructor"),
        "summary": [
            {"label": "Value", "value": compact_money(cost), "colour": _TEXT},
            {
                "label": "Left",
                "value": compact_money(remaining),
                "colour": _TEXT,
            },
            {
                "label": "Gain",
                "value": "—" if total_gain is None else f"{format_compact_gain(total_gain)}M",
                "colour": _gain_colour(total_gain),
            },
            {
                "label": "Pts",
                "value": compact_points(total_points),
                "colour": _TEXT,
            },
        ],
    }


def projected_team_layout(plan: Mapping[str, Any], export_format: str) -> dict[str, Any]:
    """Return explicit five-plus-two bubble layouts for both export canvases."""
    width, height, landscape = _format_dimensions(export_format)
    margin = 52 if landscape else 64
    if landscape:
        gap = 18
        card_width = 248
        card_height = 174
        summary_top, summary_bottom = 162, 268
        driver_top, constructor_top = 300, 500
    else:
        gap = 12
        card_width = 177
        card_height = 208
        summary_top, summary_bottom = 186, 318
        driver_top, constructor_top = 374, 610

    def centered_bounds(count: int, top: int) -> list[tuple[int, int, int, int]]:
        row_width = count * card_width + max(0, count - 1) * gap
        left_edge = int((width - row_width) / 2)
        return [
            (
                left_edge + index * (card_width + gap),
                top,
                left_edge + index * (card_width + gap) + card_width,
                top + card_height,
            )
            for index in range(count)
        ]

    metric_gap = 12
    metric_columns = 4
    metric_width = int((width - 2 * margin - metric_gap * (metric_columns - 1)) / metric_columns)
    metric_bounds = [
        (
            margin + index * (metric_width + metric_gap),
            summary_top,
            margin + index * (metric_width + metric_gap) + metric_width,
            summary_bottom,
        )
        for index in range(len(plan.get("summary", ())))
    ]
    return {
        "size": (width, height),
        "landscape": landscape,
        "driver_cards": centered_bounds(len(plan.get("drivers", ())), driver_top),
        "constructor_cards": centered_bounds(len(plan.get("constructors", ())), constructor_top),
        "summary": metric_bounds,
    }


def _draw_projected_asset_card(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    card: Mapping[str, Any],
) -> None:
    left, top, right, bottom = bounds
    draw_rounded_rectangle(draw, bounds, radius=18, fill=_PANEL)
    colour = normalize_hex_colour(card.get("team_colour"), _NEUTRAL_COLOUR)
    draw.rounded_rectangle((left, top, right, top + 7), radius=4, fill=colour)
    marker = _clean_text(card.get("marker"))
    badge_right = right - 62 if marker else right - 14
    draw_asset_identity(
        draw,
        (left + 14, top + 20, badge_right, top + 70),
        card,
        show_name=False,
    )
    if marker:
        marker_font = resolve_export_font(16, bold=True)
        marker_left = right - 54
        draw_rounded_rectangle(
            draw,
            (marker_left, top + 28, right - 12, top + 62),
            radius=10,
            fill=_ACCENT,
            outline=None,
        )
        draw.text(((marker_left + right - 12) / 2, top + 45), marker, font=marker_font, fill="#ffffff", anchor="mm")

    middle_y = top + int((bottom - top) * 0.62)
    price_text, price_font = fit_text(
        draw, card.get("price", "—"), (right - left) // 2 - 18, font_size=22, min_size=15, bold=True
    )
    gain_text, gain_font = fit_text(
        draw, card.get("gain", "—"), (right - left) // 2 - 18, font_size=22, min_size=15, bold=True
    )
    points_text, points_font = fit_text(
        draw, card.get("points", "—"), right - left - 28, font_size=26, min_size=17, bold=True
    )
    draw.text((left + 14, middle_y), price_text, font=price_font, fill=_TEXT, anchor="lm")
    draw.text(
        (right - 14, middle_y),
        gain_text,
        font=gain_font,
        fill=_clean_text(card.get("gain_colour"), _MUTED),
        anchor="rm",
    )
    draw.text((left + 14, bottom - 18), points_text, font=points_font, fill=_TEXT, anchor="ls")


def _format_dimensions(export_format: str) -> tuple[int, int, bool]:
    normalized = _clean_text(export_format, "portrait").casefold().replace("_", " ")
    if normalized in {"landscape", "reddit landscape", "reddit"}:
        return (*LANDSCAPE_SIZE, True)
    if normalized != "portrait":
        raise ValueError("format must be 'portrait' or 'landscape'")
    return (*PORTRAIT_SIZE, False)


def render_projected_team_png(
    team: Any,
    *,
    title: str,
    subtitle: str | None = None,
    budget: float | None = None,
    expected_points: float | None = None,
    expected_gain: float | None = None,
    format: str = "portrait",
) -> bytes:
    plan = projected_team_export_plan(
        team,
        title=title,
        budget=budget,
        expected_points=expected_points,
        expected_gain=expected_gain,
    )
    layout = projected_team_layout(plan, format)
    width, height = layout["size"]
    landscape = bool(layout["landscape"])
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_header(draw, width, plan["title"], None, compact=landscape)
    margin = 52 if landscape else 64
    if plan["chip_marker"]:
        chip_text, chip_font = fit_text(
            draw,
            plan["chip_marker"],
            260,
            font_size=16,
            min_size=12,
            bold=True,
        )
        chip_width = max(120, _text_width(draw, chip_text, chip_font) + 30)
        draw_rounded_rectangle(
            draw,
            (width - margin - chip_width, 48, width - margin, 84),
            radius=10,
            fill="#282f3d",
            outline=None,
        )
        draw.text(
            (width - margin - chip_width / 2, 66),
            chip_text,
            font=chip_font,
            fill=_TEXT,
            anchor="mm",
        )
    for metric, bounds in zip(plan["summary"], layout["summary"]):
        draw_metric(
            draw,
            bounds,
            metric["label"],
            metric["value"],
            value_colour=metric["colour"],
        )
    for card, bounds in zip(plan["drivers"], layout["driver_cards"]):
        _draw_projected_asset_card(draw, bounds, card)
    for card, bounds in zip(plan["constructors"], layout["constructor_cards"]):
        _draw_projected_asset_card(draw, bounds, card)
    if not plan["drivers"] and not plan["constructors"]:
        empty_font = resolve_export_font(28, bold=True)
        draw.text((width / 2, 540), "No team assets available", font=empty_font, fill=_MUTED, anchor="mm")
    return image_to_png_bytes(image)


def _efficiency_value(record: Mapping[str, Any], *keys: str) -> float | None:
    return _number(_record_value(record, *keys))


def _efficiency_status(record: Mapping[str, Any]) -> str:
    failure = record.get("has_source_failure", False)
    try:
        has_failure = False if failure is pd.NA or pd.isna(failure) else bool(failure)
    except (TypeError, ValueError):
        has_failure = bool(failure)
    status_value = record.get("status")
    if not _clean_text(status_value):
        status_value = record.get("Status")
    status = _clean_text(status_value).casefold()
    if has_failure or "source failure" in status or status == "source_failure":
        return "SOURCE FAILURE"
    if status == "complete":
        return "COMPLETE"
    if status in {"incomplete", "limited coverage"} or "limited" in status:
        return "LIMITED"
    if status in {"unavailable", "no_valid_observations"} or "no official" in status:
        return "NO DATA"
    return status.replace("_", " ").upper() if status else "NO DATA"


def _price_efficiency_warning(summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> str:
    coverage = _number(summary.get("component_coverage"))
    incomplete = coverage is None or coverage < 1.0
    incomplete = incomplete or any(_efficiency_status(record) != "COMPLETE" for record in records)
    return "* Based on incomplete official race data" if incomplete else ""


def _efficiency_incomplete(record: Mapping[str, Any]) -> bool:
    return _efficiency_status(record) != "COMPLETE"


def _efficiency_card_plan(record: Mapping[str, Any]) -> dict[str, Any]:
    abbreviation, colour = _asset_export_identity(record, _clean_text(record.get("asset_type")))
    return {
        "abbreviation": abbreviation,
        "team_colour": colour,
        "incomplete": _efficiency_incomplete(record),
        "metrics": [
            ("PRICE", _format_number(_efficiency_value(record, "current_price", "Current price"), "M")),
            ("POINTS", _format_number(_efficiency_value(record, "selected_points_total", "Selected points"))),
            ("AVG/RACE", _format_number(_efficiency_value(record, "average_points_per_race", "Average/race"))),
            ("PTS/M", _format_number(_efficiency_value(record, "price_efficiency", "Points per million"))),
        ],
    }


def price_efficiency_team_export_plan(
    team_summary: Mapping[str, Any],
    selected_assets: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build compact name-free Price Efficiency team drawing data."""
    records = dataframe_rows_for_export(selected_assets)
    cards = [_efficiency_card_plan(record) for record in records[:7]]
    warning = _price_efficiency_warning(team_summary, records)
    coverage = _number(team_summary.get("component_coverage"))
    return {
        "app_title": "F1 FANTASY OPTIMISER",
        "title": "Price Efficiency Team",
        "cards": cards,
        "footer": warning,
        "summary": [
            ("Total cost", _format_number(team_summary.get("total_cost"), "M")),
            ("Remaining", _format_number(team_summary.get("remaining_budget"), "M")),
            ("Official points", _format_number(team_summary.get("total_selected_official_points"))),
            ("Team points/race", _format_number(team_summary.get("average_team_points_per_selected_race"))),
            ("Summed efficiency", _format_number(team_summary.get("sum_individual_asset_efficiencies"))),
            ("Team efficiency", _format_number(team_summary.get("team_price_efficiency"))),
            (
                "Component coverage",
                _format_number(coverage * 100 if coverage is not None else None, "%"),
            ),
        ],
    }


def _draw_efficiency_asset_card(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    card: Mapping[str, Any],
) -> None:
    left, top, right, bottom = bounds
    draw_rounded_rectangle(draw, bounds, radius=20, fill=_PANEL)
    colour = normalize_hex_colour(card.get("team_colour"), _NEUTRAL_COLOUR)
    draw.rounded_rectangle((left, top, left + 9, bottom), radius=4, fill=colour)
    badge_bounds = draw_asset_identity(
        draw,
        (left + 24, top + 18, right - 22, top + 68),
        card,
        show_name=False,
    )
    if card.get("incomplete"):
        star_font = resolve_export_font(20, bold=True)
        draw.text((badge_bounds[2] + 8, top + 22), "*", font=star_font, fill=_MUTED)
    label_font = resolve_export_font(13, bold=True)
    metric_font = resolve_export_font(21, bold=True)
    columns = list(card.get("metrics", ()))
    cell_width = (right - left - 48) / len(columns)
    for index, (label, value) in enumerate(columns):
        x = left + 24 + index * cell_width
        draw.text((x, bottom - 62), label, font=label_font, fill=_MUTED)
        value_text, value_font = fit_text(draw, value, int(cell_width - 8), font_size=21, min_size=14, bold=True)
        draw.text((x, bottom - 28), value_text, font=value_font, fill=_TEXT)


def render_price_efficiency_team_png(
    team_summary: Mapping[str, Any],
    selected_assets: pd.DataFrame | Sequence[Mapping[str, Any]],
    race_summary: str,
    *,
    format: str = "portrait",
) -> bytes:
    width, height, landscape = _format_dimensions(format)
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    plan = price_efficiency_team_export_plan(team_summary, selected_assets)
    cards_top = _draw_header(draw, width, plan["title"], None, compact=landscape)
    margin = 52 if landscape else 64
    columns = 4 if landscape else 2
    gap = 18 if landscape else 22
    rows = max(1, math.ceil(max(len(plan["cards"]), 1) / columns))
    warning = plan["footer"]
    summary_height = 128 if landscape else 216
    warning_height = 54 if warning else 0
    summary_bottom_margin = 40 if landscape else 50
    cards_bottom = height - summary_height - summary_bottom_margin - warning_height - 26
    card_height = max(150, int((cards_bottom - cards_top - gap * (rows - 1)) / rows))
    card_width = int((width - 2 * margin - gap * (columns - 1)) / columns)
    for index, card in enumerate(plan["cards"]):
        row, column = divmod(index, columns)
        left = margin + column * (card_width + gap)
        top = cards_top + row * (card_height + gap)
        _draw_efficiency_asset_card(draw, (left, top, left + card_width, top + card_height), card)

    warning_top = height - summary_height - summary_bottom_margin - warning_height
    if warning:
        draw_rounded_rectangle(
            draw,
            (margin, warning_top, width - margin, warning_top + warning_height - 8),
            radius=14,
            fill="#3a1720",
            outline="#7f1d2d",
        )
        warning_text, warning_font = fit_text(
            draw,
            warning,
            width - 2 * margin - 30,
            font_size=17,
            min_size=12,
            bold=True,
        )
        draw.text((margin + 15, warning_top + 12), warning_text, font=warning_font, fill="#fecdd3")

    summary_top = height - summary_height - summary_bottom_margin
    metrics = plan["summary"]
    metric_columns = 7 if landscape else 4
    metric_rows = math.ceil(len(metrics) / metric_columns)
    metric_gap = 12
    metric_width = int((width - 2 * margin - metric_gap * (metric_columns - 1)) / metric_columns)
    metric_height = int((summary_height - metric_gap * (metric_rows - 1)) / metric_rows)
    for index, (label, value) in enumerate(metrics):
        row, column = divmod(index, metric_columns)
        left = margin + column * (metric_width + metric_gap)
        top = summary_top + row * (metric_height + metric_gap)
        draw_metric(draw, (left, top, left + metric_width, top + metric_height), label, value)
    return image_to_png_bytes(image)


def _table_row(record: Mapping[str, Any]) -> dict[str, Any]:
    abbreviation, colour = _asset_export_identity(record, _clean_text(record.get("asset_type")))
    return {
        "abbreviation": abbreviation,
        "team_colour": colour,
        "selected": _efficiency_value(record, "selected_points_total", "Selected points"),
        "average": _efficiency_value(record, "average_points_per_race", "Average/race"),
        "price": _efficiency_value(record, "current_price", "Current price"),
        "efficiency": _efficiency_value(record, "price_efficiency", "Points per million"),
        "incomplete": _efficiency_incomplete(record),
    }


def price_efficiency_table_export_plan(
    dataframe: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    asset_type: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Build the compact five-column, name-free Price Efficiency table plan."""
    all_rows = dataframe_rows_for_export(dataframe)
    maximum_rows = max(1, int((TABLE_MAX_HEIGHT - 280) / 76))
    requested_limit = maximum_rows if max_rows is None else min(maximum_rows, max(0, int(max_rows)))
    rows = [_table_row(row) for row in all_rows[:requested_limit]]
    title_asset = "Drivers" if _clean_text(asset_type).casefold().startswith("driver") else "Constructors"
    return {
        "app_title": "F1 FANTASY OPTIMISER",
        "title": f"{title_asset} Price Efficiency",
        "headers": ["Asset", "Selected pts", "Avg/race", "Price", "Pts/M"],
        "rows": rows,
        "footer": "* Based on incomplete official race data"
        if any(row["incomplete"] for row in rows)
        else "",
        "omitted_row_count": max(0, len(all_rows) - len(rows)),
    }


def render_price_efficiency_table_png(
    dataframe: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    asset_type: str,
    race_summary: str,
    sort_label: str,
    max_rows: int | None = None,
) -> bytes:
    plan = price_efficiency_table_export_plan(
        dataframe,
        asset_type=asset_type,
        max_rows=max_rows,
    )
    row_height = 76
    footer_height = 42 if plan["footer"] or plan["omitted_row_count"] else 0
    height = min(TABLE_MAX_HEIGHT, max(380, 260 + row_height * len(plan["rows"]) + footer_height))
    image = Image.new("RGB", (TABLE_WIDTH, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    table_top = _draw_header(draw, TABLE_WIDTH, plan["title"], None, compact=True)
    margin = 44
    column_widths = [250, 200, 180, 180, 182]
    header_height = 54
    draw_rounded_rectangle(
        draw,
        (margin, table_top, TABLE_WIDTH - margin, table_top + header_height),
        radius=13,
        fill="#232b3a",
    )
    header_font = resolve_export_font(14, bold=True)
    x = margin
    for label, column_width in zip(plan["headers"], column_widths):
        draw.text((x + 12, table_top + 18), label.upper(), font=header_font, fill=_MUTED)
        x += column_width
    body_top = table_top + header_height + 6
    for index, record in enumerate(plan["rows"]):
        top = body_top + index * row_height
        fill = _PANEL if index % 2 == 0 else _PANEL_ALT
        draw.rounded_rectangle((margin, top, TABLE_WIDTH - margin, top + row_height - 4), radius=10, fill=fill)
        badge_bounds = draw_asset_identity(
            draw,
            (margin + 10, top + 11, margin + column_widths[0] - 10, top + 61),
            record,
            show_name=False,
        )
        if record["incomplete"]:
            star_font = resolve_export_font(18, bold=True)
            draw.text((badge_bounds[2] + 8, top + 16), "*", font=star_font, fill=_MUTED)
        values = [
            _format_number(record["selected"]),
            _format_number(record["average"]),
            _format_number(record["price"], "M"),
            _format_number(record["efficiency"]),
        ]
        value_font = resolve_export_font(19, bold=True)
        x = margin + column_widths[0]
        for value, column_width in zip(values, column_widths[1:5]):
            value_text, fitted = fit_text(draw, value, column_width - 24, font_size=19, min_size=13, bold=True)
            draw.text((x + 12, top + 28), value_text, font=fitted, fill=_TEXT)
            x += column_width
    if not plan["rows"]:
        empty_font = resolve_export_font(25, bold=True)
        draw.text((TABLE_WIDTH / 2, body_top + 90), "No official observations available", font=empty_font, fill=_MUTED, anchor="mm")
    footer_parts = [part for part in [plan["footer"]] if part]
    if plan["omitted_row_count"]:
        footer_parts.append(f"{plan['omitted_row_count']} additional rows omitted")
    if footer_parts:
        footer_font = resolve_export_font(15, bold=True)
        draw.text((margin, height - 28), " · ".join(footer_parts), font=footer_font, fill=_MUTED)
    return image_to_png_bytes(image)


_PRICE_CHANGE_BAND_COLOURS = {
    "Terrible": "#5b1c25",
    "Poor": "#6b3542",
    "Good": "#405724",
    "Great": "#1d5a34",
}


def _price_change_columns(table_type: str) -> list[str]:
    normalized = _clean_text(table_type).casefold()
    if normalized in {"projection", "model projection", "probability"}:
        return [
            "Asset",
            "Price",
            "P(Terrible)",
            "P(Poor)",
            "P(Good)",
            "P(Great)",
            "Expected Points",
            "Expected price gain",
        ]
    if normalized not in {"threshold", "targets", "target"}:
        raise ValueError("table_type must be 'threshold' or 'projection'")
    return ["Asset", "Price", "Terrible", "Poor", "Good", "Great", "Rise difficulty"]


def _format_price_change_cell(column: str, value: Any) -> str:
    if column == "Price":
        return _format_number(value, "M")
    if column.startswith("P("):
        number = _number(value)
        return "—" if number is None else f"{number:.1%}"
    if column == "Expected Points":
        return _format_number(value)
    if column == "Expected price gain":
        return _format_number(value, "M", signed=True)
    if column == "Rise difficulty":
        number = _number(value)
        return "—" if number is None else f"{number:.3f}"
    return _clean_text(value, "—")


def price_change_table_export_plan(
    dataframe: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    asset_type: str,
    table_type: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Build name-free Price Changes table drawing data."""
    source_rows = dataframe_rows_for_export(dataframe)
    maximum_rows = max(1, int((TABLE_MAX_HEIGHT - 280) / 72))
    requested_limit = maximum_rows if max_rows is None else min(maximum_rows, max(0, int(max_rows)))
    columns = _price_change_columns(table_type)
    rows: list[dict[str, Any]] = []
    for source in source_rows[:requested_limit]:
        abbreviation, colour = _asset_export_identity(source, asset_type)
        rows.append(
            {
                "abbreviation": abbreviation,
                "team_colour": colour,
                "values": {
                    column: _format_price_change_cell(column, source.get(column))
                    for column in columns[1:]
                },
                "gain_colour": _gain_colour(source.get("Expected price gain")),
            }
        )
    title_asset = "Drivers" if _clean_text(asset_type).casefold().startswith("driver") else "Constructors"
    normalized_type = _clean_text(table_type).casefold()
    title_suffix = "Model Projection" if normalized_type in {"projection", "model projection", "probability"} else "Price Change Targets"
    return {
        "app_title": "F1 FANTASY OPTIMISER",
        "title": f"{title_asset} {title_suffix}",
        "headers": columns,
        "rows": rows,
        "omitted_row_count": max(0, len(source_rows) - len(rows)),
    }


def render_price_change_table_png(
    dataframe: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    asset_type: str,
    table_type: str,
    max_rows: int | None = None,
) -> bytes:
    """Render a compact Price Changes table with abbreviation-only identity."""
    plan = price_change_table_export_plan(
        dataframe,
        asset_type=asset_type,
        table_type=table_type,
        max_rows=max_rows,
    )
    row_height = 72
    footer_height = 42 if plan["omitted_row_count"] else 0
    height = min(TABLE_MAX_HEIGHT, max(380, 260 + row_height * len(plan["rows"]) + footer_height))
    image = Image.new("RGB", (PRICE_CHANGE_TABLE_WIDTH, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    table_top = _draw_header(draw, PRICE_CHANGE_TABLE_WIDTH, plan["title"], None, compact=True)
    margin = 44
    available_width = PRICE_CHANGE_TABLE_WIDTH - 2 * margin
    if len(plan["headers"]) == 7:
        column_widths = [190, 150, 190, 190, 190, 190, available_width - 1100]
    else:
        column_widths = [190, 140, 170, 170, 170, 170, 240, available_width - 1250]
    header_height = 58
    draw_rounded_rectangle(
        draw,
        (margin, table_top, PRICE_CHANGE_TABLE_WIDTH - margin, table_top + header_height),
        radius=13,
        fill="#232b3a",
    )
    header_font = resolve_export_font(14, bold=True)
    x = margin
    for label, column_width in zip(plan["headers"], column_widths):
        fitted_label, fitted_font = fit_text(
            draw,
            label.upper(),
            column_width - 20,
            font_size=14,
            min_size=10,
            bold=True,
        )
        draw.text((x + 10, table_top + 20), fitted_label, font=fitted_font, fill=_MUTED)
        x += column_width

    body_top = table_top + header_height + 6
    for index, row in enumerate(plan["rows"]):
        top = body_top + index * row_height
        draw.rounded_rectangle(
            (margin, top, PRICE_CHANGE_TABLE_WIDTH - margin, top + row_height - 4),
            radius=10,
            fill=_PANEL if index % 2 == 0 else _PANEL_ALT,
        )
        draw_asset_identity(
            draw,
            (margin + 10, top + 9, margin + column_widths[0] - 10, top + 59),
            row,
            show_name=False,
        )
        x = margin + column_widths[0]
        value_font = resolve_export_font(18, bold=True)
        for column, column_width in zip(plan["headers"][1:], column_widths[1:]):
            band_name = column[2:-1] if column.startswith("P(") else column
            cell_fill = _PRICE_CHANGE_BAND_COLOURS.get(band_name)
            if column == "Expected price gain":
                gain_colour = row["gain_colour"]
                if gain_colour == _POSITIVE:
                    cell_fill = "#183f2a"
                elif gain_colour == _NEGATIVE:
                    cell_fill = "#4a202b"
                else:
                    cell_fill = "#252d3a"
            if cell_fill:
                draw.rounded_rectangle(
                    (x + 4, top + 6, x + column_width - 4, top + row_height - 10),
                    radius=8,
                    fill=cell_fill,
                )
            value = row["values"].get(column, "—")
            fitted_value, fitted_font = fit_text(
                draw,
                value,
                column_width - 22,
                font_size=18,
                min_size=11,
                bold=True,
            )
            draw.text((x + 11, top + 25), fitted_value, font=fitted_font or value_font, fill=_TEXT)
            x += column_width
    if not plan["rows"]:
        empty_font = resolve_export_font(25, bold=True)
        draw.text(
            (PRICE_CHANGE_TABLE_WIDTH / 2, body_top + 90),
            "No price-change rows available",
            font=empty_font,
            fill=_MUTED,
            anchor="mm",
        )
    if plan["omitted_row_count"]:
        footer_font = resolve_export_font(15, bold=True)
        draw.text(
            (margin, height - 28),
            f"{plan['omitted_row_count']} additional rows omitted",
            font=footer_font,
            fill=_MUTED,
        )
    return image_to_png_bytes(image)


__all__ = [
    "LANDSCAPE_SIZE",
    "PORTRAIT_SIZE",
    "TABLE_MAX_HEIGHT",
    "PRICE_CHANGE_TABLE_WIDTH",
    "contrast_text_colour",
    "dataframe_rows_for_export",
    "draw_asset_identity",
    "draw_metric",
    "draw_rounded_rectangle",
    "fit_text",
    "image_to_png_bytes",
    "price_change_table_export_plan",
    "price_efficiency_table_export_plan",
    "price_efficiency_team_export_plan",
    "projected_team_export_plan",
    "projected_team_layout",
    "render_price_change_table_png",
    "render_price_efficiency_table_png",
    "render_price_efficiency_team_png",
    "render_projected_team_png",
    "resolve_export_font",
    "safe_export_filename",
]
