from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import html
import math
from typing import Any

import pandas as pd

from f1fantasy.race_selection import (
    RaceKey,
    RaceOption,
    RaceSelection,
    canonical_race_key,
    resolve_selected_races,
)


PRICE_EFFICIENCY_SORT_COLUMNS = {
    "Price Efficiency": "price_efficiency",
    "Average points": "average_points_per_race",
    "Total selected points": "selected_points_total",
    "Current price": "current_price",
    "Coverage": "coverage_fraction",
}
PROJECTION_SORT_OPTIONS = ("Price", "Price gain", "Projected points")

GAIN_TOLERANCE = 0.005
PRIMARY_NAVIGATION_AREAS = ("Optimise", "Market", "Team", "Settings")
OPTIMISE_MOBILE_SUBVIEWS = ("Teams", "Drivers", "Constructors", "Controls")


@dataclass(frozen=True)
class RaceControlResolution:
    preset: str
    custom_keys: tuple[RaceKey, ...]
    exclusion_options: tuple[RaceKey, ...]
    excluded_keys: tuple[RaceKey, ...]
    selection: RaceSelection
    removed_custom_keys: tuple[RaceKey, ...]
    removed_excluded_keys: tuple[RaceKey, ...]


def _canonical_keys(values: Iterable[Any] | None) -> tuple[RaceKey, ...]:
    keys: set[RaceKey] = set()
    for value in values or ():
        try:
            if isinstance(value, RaceOption):
                value = value.key
            if isinstance(value, RaceKey):
                key = canonical_race_key(value.season, value.round)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                key = canonical_race_key(value[0], value[1])
            else:
                continue
        except ValueError:
            continue
        keys.add(key)
    return tuple(sorted(keys))


def reconcile_race_control_state(
    available: Iterable[RaceOption | RaceKey | tuple[int, int]],
    preset: str,
    custom_keys: Iterable[RaceKey | tuple[int, int]] | None = None,
    excluded_keys: Iterable[RaceKey | tuple[int, int]] | None = None,
) -> RaceControlResolution:
    """Reconcile persisted UI keys through the production race selectors."""
    available_values = tuple(available)
    available_keys = resolve_selected_races(available_values, "All").included
    available_set = set(available_keys)
    requested_custom = _canonical_keys(custom_keys)
    requested_excluded = _canonical_keys(excluded_keys)
    valid_custom = resolve_selected_races(
        available_values,
        "Custom",
        custom_keys=requested_custom,
    ).included
    try:
        base = resolve_selected_races(
            available_values,
            preset,
            custom_keys=valid_custom,
        )
    except ValueError:
        preset = "All"
        base = resolve_selected_races(available_values, preset)
    selection = resolve_selected_races(
        available_values,
        base.preset,
        custom_keys=valid_custom,
        excluded_keys=requested_excluded,
    )
    base_set = set(base.included)
    return RaceControlResolution(
        preset=selection.preset,
        custom_keys=valid_custom,
        exclusion_options=base.included,
        excluded_keys=selection.excluded,
        selection=selection,
        removed_custom_keys=tuple(key for key in requested_custom if key not in available_set),
        removed_excluded_keys=tuple(key for key in requested_excluded if key not in base_set),
    )


def race_option_label(option: RaceOption | RaceKey, names: Mapping[RaceKey, str] | None = None) -> str:
    if isinstance(option, RaceOption):
        key = option.key
        name = option.race_name
    else:
        key = option
        name = (names or {}).get(key, "")
    suffix = str(name or "").strip()
    return f"R{key.round} · {suffix}" if suffix else f"R{key.round}"


def race_weight_summary(
    selection: RaceSelection,
    weights: Mapping[RaceKey, float],
) -> str:
    return " · ".join(
        f"R{key.round}: {float(weights.get(key, 0.0)):.2f}"
        for key in reversed(selection.included)
    )


def price_efficiency_race_summary(
    selection: RaceSelection,
    names: Mapping[RaceKey, str] | None = None,
) -> str:
    """Describe a local Price Efficiency race selection without model language."""
    race_names = names or {}
    included_names = [
        str(race_names.get(key) or f"R{key.round}").strip()
        for key in selection.included
    ]
    excluded_names = [
        str(race_names.get(key) or f"R{key.round}").strip()
        for key in selection.excluded
    ]
    if not included_names:
        return "No races selected"

    included = ", ".join(included_names)
    excluded = ", ".join(excluded_names)
    if selection.preset.startswith("Last ") and excluded_names:
        requested_count = selection.preset.removeprefix("Last ")
        return (
            f"Using {len(included_names)} of the last {requested_count} races: {included}"
            f" · {excluded} excluded"
        )
    if excluded_names:
        return f"Using {len(included_names)} races: {included} · {excluded} excluded"
    return f"Using {len(included_names)} races: {included}"


def effective_blend_percentages(current_weight: float, historical_weight: float) -> tuple[int, int]:
    def clean(value: Any) -> float:
        numeric = pd.to_numeric(value, errors="coerce")
        return 0.0 if pd.isna(numeric) else max(0.0, float(numeric))

    current = clean(current_weight)
    historical = clean(historical_weight)
    denominator = current + historical
    current_share = 0.5 if denominator == 0 else current / denominator
    current_percent = int(round(current_share * 100))
    return current_percent, 100 - current_percent


def resolve_objective_mode(
    saved_value: Any,
    *,
    allowed: Iterable[str],
    default: str,
    force_points_only: bool = False,
    points_only: str,
) -> str:
    allowed_values = tuple(allowed)
    resolved = str(saved_value) if saved_value in allowed_values else str(default)
    return str(points_only) if force_points_only else resolved


def normalize_price_growth_value(
    value: Any,
    *,
    default: int = 50,
    minimum: int = 0,
    maximum: int = 100,
    step: int = 5,
) -> int:
    """Clamp and round persisted objective coefficients to integer slider steps."""
    if step <= 0 or maximum < minimum:
        raise ValueError("Objective slider bounds and step must be valid.")
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not math.isfinite(float(numeric)):
        numeric = default
    clamped = max(float(minimum), min(float(maximum), float(numeric)))
    steps = math.floor(((clamped - minimum) / step) + 0.5)
    normalized = minimum + steps * step
    return int(max(minimum, min(maximum, normalized)))


def resolve_price_efficiency_asset_type(value: Any) -> str:
    """Normalize persisted active-table state, defaulting safely to Drivers."""
    return "Constructors" if str(value).casefold() == "constructors" else "Drivers"


def reconcile_asset_ids(
    selected_ids: Iterable[Any] | None,
    valid_ids: Iterable[Any],
    *,
    limit: int,
) -> tuple[str, ...]:
    valid = {str(value) for value in valid_ids}
    unique: list[str] = []
    for value in selected_ids or ():
        asset_id = str(value)
        if asset_id in valid and asset_id not in unique:
            unique.append(asset_id)
        if len(unique) == int(limit):
            break
    return tuple(unique)


def reconcile_price_efficiency_team_state(
    existing: Mapping[str, Any] | None,
    valid_driver_ids: Iterable[Any],
    valid_constructor_ids: Iterable[Any],
    optimiser_budget: float,
) -> dict[str, Any]:
    """Return team-builder-owned state without writing optimiser-owned keys."""
    state = dict(existing or {})
    budget = pd.to_numeric(state.get("budget"), errors="coerce")
    if pd.isna(budget) or not math.isfinite(float(budget)) or float(budget) < 0:
        budget = max(0.0, float(optimiser_budget))
    return {
        "driver_ids": list(
            reconcile_asset_ids(state.get("driver_ids"), valid_driver_ids, limit=5)
        ),
        "constructor_ids": list(
            reconcile_asset_ids(state.get("constructor_ids"), valid_constructor_ids, limit=2)
        ),
        "budget": float(budget),
    }


def _text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    value_text = str(value).strip()
    return "" if value_text.casefold() in {"nan", "none", "<na>"} else value_text


def _finite_number(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not math.isfinite(float(numeric)):
        return None
    return float(numeric)


def format_compact_price(value: Any) -> str:
    """Format a fantasy price in millions without repeating a wide unit label."""
    numeric = _finite_number(value)
    return "—" if numeric is None else f"${numeric:.1f}"


def gain_value_class(value: Any, *, tolerance: float = GAIN_TOLERANCE) -> str:
    """Return the single authoritative display state for a signed price gain."""
    numeric = _finite_number(value)
    if numeric is None:
        return "f1-gain-missing"
    if numeric > abs(float(tolerance)):
        return "f1-gain-positive"
    if numeric < -abs(float(tolerance)):
        return "f1-gain-negative"
    return "f1-gain-neutral"


def format_compact_gain(value: Any, *, tolerance: float = GAIN_TOLERANCE) -> str:
    """Format gain with stable precision and a neutral representation near zero."""
    numeric = _finite_number(value)
    if numeric is None:
        return "—"
    if abs(numeric) <= abs(float(tolerance)):
        numeric = 0.0
    return f"{numeric:+.2f}" if numeric else "0.00"


def format_compact_points(value: Any) -> str:
    numeric = _finite_number(value)
    return "—" if numeric is None else f"{numeric:.1f} Pts"


def compact_asset_payload(
    asset: Mapping[str, Any] | pd.Series,
    *,
    asset_type: str | None = None,
    marker: str | None = None,
) -> dict[str, Any]:
    """Build the shared, immutable compact asset display contract."""
    row = dict(asset) if isinstance(asset, Mapping) else asset.to_dict()
    normalized_type = (
        "constructor"
        if str(asset_type or row.get("asset_type", "")).casefold().startswith("constructor")
        else "driver"
    )
    abbreviation = asset_abbreviation(row, normalized_type)
    full_name = next(
        (candidate for candidate in (_text(row.get("full_name")), _text(row.get("name"))) if candidate),
        abbreviation,
    )
    team_name = next(
        (candidate for candidate in (_text(row.get("team_name")), _text(row.get("team"))) if candidate),
        "",
    )
    identity = full_name if not team_name or team_name == full_name else f"{full_name} — {team_name}"
    price_value = _finite_number(row.get("price", row.get("current_price")))
    gain_value = _finite_number(
        row.get(
            "expected_price_gain",
            row.get("expected_price_change", row.get("Expected price gain")),
        )
    )
    points_value = _finite_number(
        row.get(
            "display_exp_score",
            row.get("exp_score", row.get("expected_points", row.get("Expected Points"))),
        )
    )
    normalized_marker = str(marker or "").strip().lower()
    if normalized_marker not in {"2x", "3x"}:
        normalized_marker = ""
    return {
        "asset_type": normalized_type,
        "abbreviation": abbreviation,
        "identity": identity,
        "team_colour": normalize_hex_colour(row.get("team_colour")),
        "marker": normalized_marker,
        "price_value": price_value,
        "gain_value": gain_value,
        "points_value": points_value,
        "price": format_compact_price(price_value),
        "gain": format_compact_gain(gain_value),
        "points": format_compact_points(points_value),
        "gain_class": gain_value_class(gain_value),
    }


def compact_asset_universe_rows(
    assets: pd.DataFrame,
    *,
    asset_type: str,
    locked_ids: Iterable[Any] = (),
    excluded_ids: Iterable[Any] = (),
) -> tuple[dict[str, Any], ...]:
    """Return immutable display/action rows for the compact optimiser universe."""
    if assets is None or assets.empty:
        return ()
    locked = {str(value) for value in locked_ids}
    excluded = {str(value) for value in excluded_ids} - locked
    rows: list[dict[str, Any]] = []
    for _, source in assets.copy(deep=True).iterrows():
        row = source.to_dict()
        asset_id = _text(row.get("id")) or _text(row.get("asset_id"))
        if not asset_id:
            continue
        payload = compact_asset_payload(row, asset_type=asset_type)
        rows.append(
            {
                "asset_id": asset_id,
                "asset": compact_asset_identity_html(
                    {
                        **row,
                        "asset_type": payload["asset_type"],
                        "abbreviation": payload["abbreviation"],
                        "full_name": payload["identity"].split(" — ", 1)[0],
                        "team_colour": payload["team_colour"],
                    }
                ),
                "abbreviation": payload["abbreviation"],
                "price": payload["price"].removeprefix("$"),
                "price_value": payload["price_value"],
                "gain": payload["gain"],
                "gain_value": payload["gain_value"],
                "gain_class": payload["gain_class"],
                "points": "—" if payload["points_value"] is None else f'{payload["points_value"]:.1f}',
                "points_value": payload["points_value"],
                "lock": asset_id in locked,
                "exclude": asset_id in excluded,
            }
        )
    return tuple(rows)


def sort_projection_assets(
    assets: pd.DataFrame | None,
    sort_by: str = "Price",
    ascending: bool = False,
) -> pd.DataFrame:
    """Sort the displayed projection metric, preserving values, ties and source data."""
    data = assets.copy(deep=True) if assets is not None else pd.DataFrame()
    if data.empty:
        return data
    metric_key = {
        "Price": "price_value",
        "Price gain": "gain_value",
        "Projected points": "points_value",
    }.get(sort_by, "price_value")
    values = pd.Series(
        [compact_asset_payload(row)[metric_key] for _, row in data.iterrows()],
        dtype="float64",
    )
    positions = values.sort_values(
        ascending=bool(ascending), kind="stable", na_position="last"
    ).index
    return data.iloc[positions].copy(deep=True)


def asset_constraint_transition(
    state: Mapping[str, Iterable[Any]] | None,
    *,
    asset_type: str,
    asset_id: Any,
    action: str,
    active: bool,
) -> dict[str, list[str]]:
    """Apply one lock/exclude action while keeping both states mutually exclusive."""
    keys = (
        "locked_driver_ids",
        "excluded_driver_ids",
        "locked_constructor_ids",
        "excluded_constructor_ids",
    )

    def unique(values: Iterable[Any] | None) -> list[str]:
        output: list[str] = []
        for value in values or ():
            token = str(value)
            if token not in output:
                output.append(token)
        return output

    resolved = {key: unique((state or {}).get(key)) for key in keys}
    normalized_type = (
        "constructor" if str(asset_type).casefold().startswith("constructor") else "driver"
    )
    normalized_action = str(action).casefold()
    if normalized_action not in {"lock", "exclude"}:
        raise ValueError("action must be 'lock' or 'exclude'")
    token = str(asset_id)
    selected_key = f"{'locked' if normalized_action == 'lock' else 'excluded'}_{normalized_type}_ids"
    conflicting_key = f"{'excluded' if normalized_action == 'lock' else 'locked'}_{normalized_type}_ids"
    resolved[selected_key] = [value for value in resolved[selected_key] if value != token]
    if active:
        resolved[selected_key].append(token)
        resolved[conflicting_key] = [value for value in resolved[conflicting_key] if value != token]
    return resolved


def reconcile_constraint_pair(
    primary_ids: Iterable[Any] | None,
    conflicting_ids: Iterable[Any] | None,
) -> tuple[list[str], list[str]]:
    """Prefer the freshly edited selection and remove it from its conflicting peer."""
    primary = list(dict.fromkeys(str(value) for value in (primary_ids or ())))
    primary_set = set(primary)
    conflicting = [
        value
        for value in dict.fromkeys(str(value) for value in (conflicting_ids or ()))
        if value not in primary_set
    ]
    return primary, conflicting


def ranked_solution_current_team_update(
    solution: Any,
    *,
    valid_driver_ids: Iterable[Any],
    valid_constructor_ids: Iterable[Any],
) -> dict[str, Any]:
    """Validate a ranked solution and build an atomic Current Team selection update."""
    valid_drivers = {str(value) for value in valid_driver_ids}
    valid_constructors = {str(value) for value in valid_constructor_ids}

    def frame_ids(frame: Any, required: int, valid: set[str], label: str) -> tuple[list[str], str | None]:
        if not isinstance(frame, pd.DataFrame):
            return [], f"The ranked team has no valid {label} data."
        id_column = next((column for column in ("id", "asset_id") if column in frame.columns), None)
        if id_column is None:
            return [], f"The ranked team has no stable {label} IDs."
        ids = [str(value) for value in frame[id_column].tolist()]
        if len(ids) != required or len(set(ids)) != required:
            return [], f"The ranked team must contain exactly {required} unique {label}."
        missing = [value for value in ids if value not in valid]
        if missing:
            return [], f"The ranked team contains unavailable {label}: {', '.join(missing)}."
        return ids, None

    drivers = solution.get("drivers") if isinstance(solution, Mapping) else getattr(solution, "drivers", None)
    constructors = (
        solution.get("constructors")
        if isinstance(solution, Mapping)
        else getattr(solution, "constructors", None)
    )
    driver_ids, driver_error = frame_ids(drivers, 5, valid_drivers, "drivers")
    constructor_ids, constructor_error = frame_ids(
        constructors, 2, valid_constructors, "constructors"
    )
    error = driver_error or constructor_error
    if error:
        return {"ok": False, "updates": {}, "error": error}
    return {
        "ok": True,
        "updates": {
            "current_team_driver_ids": driver_ids,
            "current_team_constructor_ids": constructor_ids,
        },
        "error": None,
    }


def team_summary_payload(
    *,
    total_cost: Any,
    budget: Any,
    expected_gain: Any,
    expected_points: Any,
    limitless: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build the four equal-status values shown on every ranked team."""
    cost = _finite_number(total_cost)
    numeric_budget = _finite_number(budget)
    gain = None if limitless else _finite_number(expected_gain)
    points = _finite_number(expected_points)
    remaining = (
        numeric_budget - cost
        if not limitless and numeric_budget is not None and cost is not None
        else None
    )

    def money(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f}M"

    return {
        "value": {"label": "Value", "value": money(cost), "class": "f1-gain-neutral"},
        "left": {"label": "Left", "value": money(remaining), "class": "f1-gain-neutral"},
        "gain": {
            "label": "Gain",
            "value": "—" if gain is None else f"{format_compact_gain(gain)}M",
            "class": gain_value_class(gain),
        },
        "points": {
            "label": "Pts",
            "value": "—" if points is None else f"{points:.1f}",
            "class": "f1-gain-neutral",
        },
    }


def team_summary_html(summary: Mapping[str, Mapping[str, Any]]) -> str:
    """Render the shared four-value summary without large metric widgets."""
    labels = {
        "value": "Team cost",
        "left": "Budget left",
        "gain": "Price gain",
        "points": "Expected pts",
    }
    items = "".join(
        (
            '<div class="f1-team-stat">'
            f'<span>{html.escape(str(labels.get(key, item.get("label", ""))))}</span>'
            f'<strong class="{html.escape(str(item.get("class", "f1-gain-neutral")))}">'
            f'{html.escape(str(item.get("value", "—")))}</strong>'
            "</div>"
        )
        for key, item in summary.items()
    )
    return f'<div class="f1-team-summary">{items}</div>'


def optimiser_result_signature(
    *,
    data_version: Any,
    budget: Any,
    chip_mode: Any,
    price_growth_value: Any,
    locked_driver_ids: Iterable[Any] = (),
    excluded_driver_ids: Iterable[Any] = (),
    locked_constructor_ids: Iterable[Any] = (),
    excluded_constructor_ids: Iterable[Any] = (),
) -> tuple[Any, ...]:
    """Canonicalise only inputs which can change optimiser output."""
    numeric_budget = _finite_number(budget)
    return (
        data_version,
        numeric_budget,
        str(chip_mode),
        normalize_price_growth_value(price_growth_value),
        tuple(sorted(str(value) for value in locked_driver_ids)),
        tuple(sorted(str(value) for value in excluded_driver_ids)),
        tuple(sorted(str(value) for value in locked_constructor_ids)),
        tuple(sorted(str(value) for value in excluded_constructor_ids)),
    )


def team_solution_key(solution: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def asset_ids(frame: Any) -> tuple[str, ...]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return ()
        column = next((name for name in ("id", "asset_id", "name") if name in frame.columns), None)
        if column is None:
            return tuple(sorted(str(value) for value in frame.index))
        return tuple(sorted(frame[column].astype(str)))

    drivers = solution.get("drivers") if isinstance(solution, Mapping) else getattr(solution, "drivers", None)
    constructors = solution.get("constructors") if isinstance(solution, Mapping) else getattr(solution, "constructors", None)
    return asset_ids(drivers), asset_ids(constructors)


def next_team_batch(
    existing: Iterable[Any] | None,
    candidates: Iterable[Any] | None,
    *,
    batch_size: int = 10,
) -> dict[str, Any]:
    """Append up to one deterministic batch, rejecting duplicate combinations."""
    retained = list(existing or ())
    seen = {team_solution_key(solution) for solution in retained}
    appended: list[Any] = []
    for solution in candidates or ():
        key = team_solution_key(solution)
        if key in seen or not all(key):
            continue
        retained.append(solution)
        appended.append(solution)
        seen.add(key)
        if len(appended) >= max(1, int(batch_size)):
            break
    next_start = len(retained) + 1
    return {
        "solutions": retained,
        "appended": appended,
        "next_label": f"Load teams {next_start}–{next_start + max(1, int(batch_size)) - 1}",
        "exhausted": len(appended) < max(1, int(batch_size)),
    }


def primary_navigation_state(value: Any) -> str:
    candidate = str(value or "").strip().casefold()
    return next(
        (area for area in PRIMARY_NAVIGATION_AREAS if area.casefold() == candidate),
        PRIMARY_NAVIGATION_AREAS[0],
    )


def responsive_layout_mode(value: Any) -> str:
    """Resolve an optional testing override; CSS remains the automatic default."""
    candidate = str(value or "").strip().casefold()
    return candidate if candidate in {"mobile", "desktop"} else "auto"


def optimise_mobile_subview(value: Any) -> str:
    """Return a stable contextual Optimise view without touching model state."""
    candidate = str(value or "").strip().casefold()
    return next(
        (view for view in OPTIMISE_MOBILE_SUBVIEWS if view.casefold() == candidate),
        OPTIMISE_MOBILE_SUBVIEWS[0],
    )


def asset_abbreviation(asset: Mapping[str, Any] | pd.Series, asset_type: str | None = None) -> str:
    row = asset if isinstance(asset, Mapping) else asset.to_dict()
    for column in ("abbreviation", "tla", "TLA", "short_name"):
        candidate = _text(row.get(column))
        if candidate:
            return candidate.upper()[:5]
    name = next(
        (candidate for candidate in (_text(row.get(column)) for column in ("full_name", "name", "team_name")) if candidate),
        "",
    )
    words = [word for word in name.replace("-", " ").split() if word]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:3].upper()
    abbreviation = "".join(word[0] for word in words[:3]).upper()
    if str(asset_type or row.get("asset_type", "")).casefold() == "driver":
        last_name = words[-1][:3].upper()
        return last_name if len(last_name) == 3 else abbreviation
    return abbreviation


def normalize_hex_colour(value: Any, fallback: str = "#64748b") -> str:
    colour = _text(value).lower()
    if len(colour) == 4 and colour.startswith("#"):
        colour = "#" + "".join(character * 2 for character in colour[1:])
    if len(colour) != 7 or not colour.startswith("#"):
        return fallback
    try:
        int(colour[1:], 16)
    except ValueError:
        return fallback
    return colour


def contrast_text_colour(background: Any) -> str:
    colour = normalize_hex_colour(background)
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return "#111827" if luminance > 0.42 else "#ffffff"


def compact_asset_identity_html(asset: Mapping[str, Any] | pd.Series) -> str:
    """Keep identity accessible when responsive layouts show only the team badge."""
    row = asset if isinstance(asset, Mapping) else asset.to_dict()
    asset_type = _text(row.get("asset_type")) or "asset"
    abbreviation = asset_abbreviation(row, asset_type)
    full_name = next(
        (candidate for candidate in (_text(row.get("full_name")), _text(row.get("name"))) if candidate),
        abbreviation,
    )
    team_name = next(
        (candidate for candidate in (_text(row.get("team_name")), _text(row.get("team"))) if candidate),
        "",
    )
    identity = full_name if not team_name or team_name == full_name else f"{full_name} — {team_name}"
    background = normalize_hex_colour(row.get("team_colour"))
    foreground = contrast_text_colour(background)
    team_html = (
        f'<span class="f1-asset-team">{html.escape(team_name)}</span>'
        if team_name and team_name != full_name
        else ""
    )
    return (
        '<span class="f1-asset-identity" role="img" title="{title}" aria-label="{aria}">'
        '<span class="f1-asset-id" style="background:{background};color:{foreground}" '
        'aria-hidden="true">{abbreviation}</span>'
        '<span class="f1-asset-text"><span class="f1-asset-name">{name}</span>{team}</span>'
        '</span>'
    ).format(
        background=background,
        foreground=foreground,
        title=html.escape(identity, quote=True),
        aria=html.escape(f"{asset_type.title()}: {identity}", quote=True),
        abbreviation=html.escape(abbreviation),
        name=html.escape(full_name),
        team=team_html,
    )


def compact_asset_table_html(
    assets: pd.DataFrame,
    *,
    asset_type: str,
) -> str:
    """Render the dense Asset/Price/Gain/Pts universe without changing values."""
    if assets is None or assets.empty:
        return '<div class="f1-empty-table">No assets available.</div>'
    rows: list[str] = []
    for _, source in assets.iterrows():
        payload = compact_asset_payload(source, asset_type=asset_type)
        identity = compact_asset_identity_html(
            {
                **source.to_dict(),
                "asset_type": payload["asset_type"],
                "abbreviation": payload["abbreviation"],
                "full_name": payload["identity"].split(" — ", 1)[0],
                "team_colour": payload["team_colour"],
            }
        )
        points = "—" if payload["points_value"] is None else f'{payload["points_value"]:.1f}'
        history_mode = _text(source.get("price_history_mode")).casefold()
        status_badge = ""
        if history_mode == "inactive_unknown":
            status_badge = (
                ' <span class="f1-availability-muted" '
                'title="Price settlement eligibility is unknown">Inactive</span>'
            )
        elif history_mode == "fresh":
            status_badge = (
                ' <span class="f1-availability-muted" '
                'title="Fewer than two completed asset-specific observations">Fresh history</span>'
            )
        rows.append(
            "<tr>"
            f'<td class="f1-asset-cell">{identity}{status_badge}</td>'
            f'<td>{html.escape(payload["price"].removeprefix("$"))}</td>'
            f'<td class="{payload["gain_class"]}">{html.escape(payload["gain"])}</td>'
            f"<td>{html.escape(points)}</td>"
            "</tr>"
        )
    body = "".join(rows)
    return (
        '<div class="f1-responsive-table f1-desktop-table">'
        '<div class="f1-table-scroll f1-universe-scroll"><table class="f1-compact-table f1-universe-table">'
        '<thead><tr><th scope="col">Asset</th><th scope="col" title="Current price in millions">Price ($M)</th>'
        '<th scope="col" title="Expected price gain in millions">Gain ($M)</th>'
        '<th scope="col" title="Expected points">Points</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div></div>"
        '<div class="f1-responsive-table f1-mobile-table">'
        '<div class="f1-table-scroll f1-universe-scroll"><table class="f1-compact-table f1-mobile-schema f1-projection-mobile">'
        '<thead><tr><th scope="col">Asset</th><th scope="col" title="Current price in millions">Price ($M)</th>'
        '<th scope="col" title="Expected price gain in millions">Gain ($M)</th>'
        '<th scope="col" title="Expected points">Points</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div></div>"
    )


def prepare_compact_asset_table(
    table: pd.DataFrame,
    source_assets: pd.DataFrame | None = None,
    *,
    asset_type: str,
) -> pd.DataFrame:
    """Replace wide identity columns with the shared accessible Asset badge.

    The calculation table and source asset frame are defensively copied. Row
    order and every non-identity column are preserved exactly.
    """
    data = table.copy(deep=True) if table is not None else pd.DataFrame()
    assets = source_assets.copy(deep=True) if source_assets is not None else pd.DataFrame()
    normalized_type = "constructor" if str(asset_type).casefold().startswith("constructor") else "driver"

    source_by_token: dict[str, dict[str, Any]] = {}
    for _, source_row in assets.iterrows():
        source = source_row.to_dict()
        for column in (
            "id",
            "asset_id",
            "tla",
            "TLA",
            "abbreviation",
            "driver_reference",
            "name",
            "full_name",
            "team_name",
        ):
            token = _text(source.get(column)).casefold()
            if token:
                source_by_token.setdefault(token, source)

    identities: list[str] = []
    for _, display_row in data.iterrows():
        display = display_row.to_dict()
        source: dict[str, Any] = {}
        for value in (
            display.get("Fantasy asset ID"),
            display.get("Abbrev"),
            display.get("Name"),
        ):
            token = _text(value).casefold()
            if token and token in source_by_token:
                source = source_by_token[token]
                break

        source_abbreviation = asset_abbreviation(source, normalized_type) if source else ""
        abbreviation = (
            source_abbreviation
            if source_abbreviation and source_abbreviation != "?"
            else _text(display.get("Abbrev"))
        )
        full_name = _text(display.get("Name")) or _text(source.get("full_name")) or _text(source.get("name"))
        team_name = (
            _text(display.get("Team"))
            or _text(source.get("team_name"))
            or _text(source.get("team"))
        )
        identity = {
            **source,
            "asset_type": normalized_type,
            "abbreviation": abbreviation,
            "full_name": full_name,
            "team_name": team_name,
            "team_colour": source.get("team_colour", display.get("team_colour")),
        }
        identities.append(compact_asset_identity_html(identity))

    compact = data.drop(
        columns=[column for column in ("Asset", "Abbrev", "Name", "Team") if column in data.columns]
    ).copy(deep=True)
    compact.insert(0, "Asset", identities)
    return compact


def price_efficiency_status_label(row: Mapping[str, Any] | pd.Series) -> str:
    values = row if isinstance(row, Mapping) else row.to_dict()
    status = _text(values.get("status")).casefold()
    failure_value = values.get("has_source_failure", False)
    has_failure = False if failure_value is pd.NA or pd.isna(failure_value) else bool(failure_value)
    if has_failure or status == "source_failure":
        return "⚠ Source failure"
    if status == "complete":
        return "Complete"
    if status == "invalid_price":
        return "⚠ Invalid price"
    if status == "no_races_selected":
        return "No races selected"
    if status in {"unavailable", "no_valid_observations"}:
        return "— No official data"
    if status == "incomplete":
        return "◐ Limited coverage"
    return status.replace("_", " ").title() if status else "— No official data"


def prepare_price_efficiency_display(
    efficiency_table: pd.DataFrame,
    sort_by: str = "Price Efficiency",
    ascending: bool = False,
) -> pd.DataFrame:
    data = efficiency_table.copy(deep=True) if efficiency_table is not None else pd.DataFrame()
    output_columns = [
        "asset_identity_html",
        "Asset",
        "Selected points",
        "Average/race",
        "Current price",
        "Points per million",
        "Coverage",
        "Status",
        "asset_id",
        "full_name",
        "team_name",
        "asset_type",
        "abbreviation",
        "team_colour",
        "status",
        "has_source_failure",
        "selected_race_count",
        "valid_race_count",
    ]
    if data.empty:
        return pd.DataFrame(columns=output_columns)
    sort_column = PRICE_EFFICIENCY_SORT_COLUMNS.get(sort_by, "price_efficiency")
    if sort_column not in data.columns:
        sort_column = "price_efficiency"
    data = data.sort_values(sort_column, ascending=bool(ascending), na_position="last").reset_index(drop=True)
    for column in [
        "selected_points_total",
        "average_points_per_race",
        "current_price",
        "price_efficiency",
        "coverage_fraction",
        "valid_race_count",
        "selected_race_count",
    ]:
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data["asset_identity_html"] = data.apply(compact_asset_identity_html, axis=1)
    data["Asset"] = data.apply(
        lambda row: asset_abbreviation(row, _text(row.get("asset_type"))), axis=1
    )
    data["Selected points"] = data["selected_points_total"]
    data["Average/race"] = data["average_points_per_race"]
    data["Current price"] = data["current_price"]
    data["Points per million"] = data["price_efficiency"]
    data["Coverage"] = data.apply(
        lambda row: (
            f"{int(row['valid_race_count'])}/{int(row['selected_race_count'])}"
            if pd.notna(row["valid_race_count"]) and pd.notna(row["selected_race_count"])
            else "0/0"
        ),
        axis=1,
    )
    data["Status"] = data.apply(price_efficiency_status_label, axis=1)
    for column in [
        "asset_id",
        "full_name",
        "team_name",
        "asset_type",
        "abbreviation",
        "team_colour",
        "status",
        "has_source_failure",
    ]:
        if column not in data.columns:
            data[column] = ""
    return data[output_columns]


def price_efficiency_table_html(display: pd.DataFrame) -> str:
    if display is None or display.empty:
        return '<div class="f1-empty-table">No official observations available.</div>'

    def metric(value: Any, suffix: str = "") -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        return "—" if pd.isna(numeric) else f"{float(numeric):.2f}{suffix}"

    rows: list[str] = []
    notes: list[str] = []
    for _, row in display.iterrows():
        status = html.escape(str(row.get("Status", "")))
        incomplete = str(row.get("Status", "")) != "Complete"
        marker = f'<sup title="{status}">*</sup>' if incomplete else ""
        if incomplete and status and status not in notes:
            notes.append(status)
        rows.append(
            "<tr>"
            f'<td class="f1-asset-cell">{row.get("asset_identity_html", "")}{marker}</td>'
            f'<td>{metric(row.get("Current price"))}</td>'
            f'<td>{metric(row.get("Average/race"))}</td>'
            f'<td>{metric(row.get("Points per million"))}</td>'
            "</tr>"
        )
    note_html = (
        f'<div class="f1-table-note">* {" · ".join(notes)}</div>'
        if notes
        else ""
    )
    desktop_body = "".join(rows)
    mobile_rows: list[str] = []
    for _, row in display.iterrows():
        status = html.escape(str(row.get("Status", "")))
        incomplete = str(row.get("Status", "")) != "Complete"
        marker = f'<sup title="{status}">*</sup>' if incomplete else ""
        mobile_rows.append(
            "<tr>"
            f'<td class="f1-asset-cell">{row.get("asset_identity_html", "")}{marker}</td>'
            f'<td>{metric(row.get("Current price"))}</td>'
            f'<td>{metric(row.get("Points per million"))}</td>'
            "</tr>"
        )
    return (
        '<div class="f1-responsive-table f1-desktop-table"><div class="f1-table-scroll">'
        '<table class="f1-compact-table f1-efficiency-desktop">'
        "<thead><tr><th>Asset</th><th>Price</th><th>Avg/race</th><th>Pts/M</th></tr></thead>"
        f"<tbody>{desktop_body}</tbody></table>{note_html}</div></div>"
        '<div class="f1-responsive-table f1-mobile-table"><div class="f1-table-scroll">'
        '<table class="f1-compact-table f1-mobile-schema f1-efficiency-mobile">'
        "<thead><tr><th>Asset</th><th>Price</th><th>Pts/M</th></tr></thead>"
        f"<tbody>{''.join(mobile_rows)}</tbody></table>{note_html}</div></div>"
    )


def sprint_diagnostic_table_html(frame: pd.DataFrame) -> str:
    """Render the compact mobile Base/+Sprint/Final diagnostic schema."""
    if frame is None or frame.empty:
        return '<div class="f1-empty-table">No Sprint diagnostics available.</div>'
    required = {
        "name",
        "baseline_expected_points",
        "sprint_bonus",
        "next_race_expected_points",
    }
    if not required.issubset(frame.columns):
        return '<div class="f1-empty-table">Sprint diagnostics are incomplete.</div>'
    rows: list[str] = []
    for _, row in frame.copy(deep=True).iterrows():
        abbreviation = asset_abbreviation(row)
        identity = compact_asset_identity_html(
            {
                **row.to_dict(),
                "abbreviation": abbreviation,
                "full_name": row.get("name"),
                "asset_type": "constructor" if not _text(row.get("team")) else "driver",
            }
        )
        base = _finite_number(row.get("baseline_expected_points"))
        bonus = _finite_number(row.get("sprint_bonus"))
        final = _finite_number(row.get("next_race_expected_points"))
        rows.append(
            "<tr>"
            f'<td class="f1-asset-cell">{identity}</td>'
            f"<td>{'—' if base is None else f'{base:.2f}'}</td>"
            f"<td class=\"{gain_value_class(bonus)}\">{'—' if bonus is None else f'{bonus:+.2f}'}</td>"
            f"<td>{'—' if final is None else f'{final:.2f}'}</td>"
            "</tr>"
        )
    return (
        '<div class="f1-table-scroll"><table class="f1-compact-table f1-mobile-schema f1-sprint-mobile">'
        "<thead><tr><th>Asset</th><th>Base</th><th>+Sprint</th><th>Final</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
