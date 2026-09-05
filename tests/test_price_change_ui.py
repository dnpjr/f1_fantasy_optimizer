from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from f1fantasy.ui_helpers import prepare_compact_asset_table


def test_driver_price_change_table_uses_one_accessible_coloured_asset_identity():
    table = pd.DataFrame(
        [
            {
                "Abbrev": "HUL",
                "Name": "Nico Hülkenberg",
                "Team": "Audi",
                "Price": 8.2,
                "Terrible": "≤ 3",
                "Poor": "4 to 8",
                "Good": "9 to 13",
                "Great": "≥ 14",
                "Rise difficulty": 1.707,
            }
        ]
    )
    assets = pd.DataFrame(
        [
            {
                "id": "hulkenberg",
                "tla": "HUL",
                "name": "Nico Hülkenberg",
                "team": "Audi",
                "team_colour": "#14532d",
            }
        ]
    )

    compact = prepare_compact_asset_table(table, assets, asset_type="driver")

    assert compact.columns.tolist() == [
        "Asset",
        "Price",
        "Terrible",
        "Poor",
        "Good",
        "Great",
        "Rise difficulty",
    ]
    identity = compact.loc[0, "Asset"]
    assert ">HUL</span>" in identity
    assert "background:#14532d" in identity
    assert "color:#ffffff" in identity
    assert 'title="Nico Hülkenberg — Audi"' in identity
    assert 'aria-label="Driver: Nico Hülkenberg — Audi"' in identity


def test_constructor_price_change_table_prefers_stable_source_abbreviation():
    table = pd.DataFrame(
        [
            {
                "Abbrev": "constructor-1",
                "Name": "Red Bull Racing",
                "Price": 27.5,
                "P(Terrible)": 0.1,
                "P(Poor)": 0.2,
                "P(Good)": 0.3,
                "P(Great)": 0.4,
                "Expected Points": 31.5,
                "Expected price gain": 0.2,
            }
        ]
    )
    assets = pd.DataFrame(
        [
            {
                "id": "constructor-1",
                "name": "Red Bull Racing",
                "team_colour": "#1e3a8a",
            }
        ]
    )

    compact = prepare_compact_asset_table(table, assets, asset_type="constructor")

    assert ">RBR</span>" in compact.loc[0, "Asset"]
    assert 'title="Red Bull Racing"' in compact.loc[0, "Asset"]
    assert 'aria-label="Constructor: Red Bull Racing"' in compact.loc[0, "Asset"]
    assert "Abbrev" not in compact.columns
    assert "Name" not in compact.columns
    assert "Team" not in compact.columns


def test_compact_price_change_identity_uses_existing_fallbacks():
    table = pd.DataFrame(
        [{"Abbrev": "", "Name": "Alex Albon", "Team": "Williams", "Price": 12.0}]
    )
    assets = pd.DataFrame([{"id": "albon", "name": "Alex Albon", "team": "Williams"}])

    compact = prepare_compact_asset_table(table, assets, asset_type="driver")

    identity = compact.loc[0, "Asset"]
    assert ">ALB</span>" in identity
    assert "background:#64748b" in identity
    assert "color:#ffffff" in identity


def test_compact_price_change_table_preserves_order_values_and_inputs():
    table = pd.DataFrame(
        [
            {
                "Abbrev": "B",
                "Name": "Second",
                "Team": "Team B",
                "Price": 10.0,
                "P(Terrible)": 0.4,
                "P(Poor)": 0.3,
                "P(Good)": 0.2,
                "P(Great)": 0.1,
                "Expected Points": 12.25,
                "Expected price gain": -0.2,
            },
            {
                "Abbrev": "A",
                "Name": "First",
                "Team": "Team A",
                "Price": 11.0,
                "P(Terrible)": 0.1,
                "P(Poor)": 0.2,
                "P(Good)": 0.3,
                "P(Great)": 0.4,
                "Expected Points": 18.5,
                "Expected price gain": 0.3,
            },
        ]
    )
    assets = pd.DataFrame(
        [
            {"id": "B", "name": "Second", "team": "Team B", "team_colour": "#111827"},
            {"id": "A", "name": "First", "team": "Team A", "team_colour": "#facc15"},
        ]
    )
    table_before = table.copy(deep=True)
    assets_before = assets.copy(deep=True)

    compact = prepare_compact_asset_table(table, assets, asset_type="driver")

    for column in [
        "Price",
        "P(Terrible)",
        "P(Poor)",
        "P(Good)",
        "P(Great)",
        "Expected Points",
        "Expected price gain",
    ]:
        assert_series_equal(compact[column], table[column], check_names=True)
    assert ">SEC</span>" in compact.loc[0, "Asset"]
    assert ">FIR</span>" in compact.loc[1, "Asset"]
    assert "color:#111827" in compact.loc[1, "Asset"]
    assert_frame_equal(table, table_before)
    assert_frame_equal(assets, assets_before)


def test_compact_asset_markup_remains_raw_for_html_table_rendering():
    table = pd.DataFrame([{"Abbrev": "LEC", "Name": "Charles Leclerc", "Team": "Ferrari"}])
    assets = pd.DataFrame(
        [{"tla": "LEC", "name": "Charles Leclerc", "team": "Ferrari", "team_colour": "#dc2626"}]
    )

    compact = prepare_compact_asset_table(table, assets, asset_type="driver")
    rendered = compact.style.hide(axis="index").to_html()

    assert '<span class="f1-asset-id"' in rendered
    assert "&lt;span" not in rendered
