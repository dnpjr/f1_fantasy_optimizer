from __future__ import annotations

import json

import pandas as pd
import pytest

from f1fantasy import app_core, fantasy_api
from f1fantasy.asset_identity import build_player_identity_map


def _raw_driver_market(*, old_lawson_active: int = 0) -> list[dict]:
    rows = [
        {
            "PlayerId": 114,
            "FUllName": "Liam Lawson",
            "PositionName": "DRIVER",
            "IsActive": old_lawson_active,
            "Value": 12.0,
            "OldPlayerValue": 11.9,
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
            "Value": 14.5,
            "OldPlayerValue": 14.4,
            "TeamId": 5,
            "TeamName": "Red Bull",
            "DriverReference": "LIALAW01",
            "DriverTLA": "LAW",
            "F1PlayerId": 14,
        },
    ]
    for asset_id in range(2, 7):
        rows.append(
            {
                "PlayerId": asset_id,
                "FUllName": f"Driver {asset_id}",
                "PositionName": "DRIVER",
                "IsActive": 1,
                "Value": 10.0,
                "OldPlayerValue": 9.9,
                "TeamId": asset_id,
                "TeamName": f"Team {asset_id}",
                "DriverReference": f"DRIVER{asset_id}",
                "DriverTLA": f"D{asset_id}",
                "F1PlayerId": asset_id,
            }
        )
    return rows


def _raw_constructor_market() -> list[dict]:
    return [
        {
            "PlayerId": 1,
            "FUllName": "Constructor One",
            "PositionName": "CONSTRUCTOR",
            "IsActive": 0,
            "Value": 20.0,
            "OldPlayerValue": 19.8,
        },
        {
            "PlayerId": 2,
            "FUllName": "Constructor Two",
            "PositionName": "CONSTRUCTOR",
            "IsActive": 1,
            "Value": 20.0,
            "OldPlayerValue": 19.8,
        },
        {
            "PlayerId": 3,
            "FUllName": "Constructor Three",
            "PositionName": "CONSTRUCTOR",
            "IsActive": 1,
            "Value": 22.0,
            "OldPlayerValue": 21.8,
        },
    ]


def _universes(*, old_lawson_active: int = 0):
    player_ledger = fantasy_api.normalise_player_asset_ledger(
        _raw_driver_market(old_lawson_active=old_lawson_active),
        feed_round=12,
    )
    constructor_ledger = fantasy_api.normalise_constructor_asset_ledger(
        _raw_constructor_market(),
        feed_round=12,
    )
    selectable_drivers = fantasy_api.selectable_player_assets(player_ledger).rename(
        columns={"playerId": "id"}
    )
    selectable_constructors = fantasy_api.selectable_constructor_assets(
        constructor_ledger
    ).rename(columns={"teamId": "id"})
    selectable_drivers["id"] = selectable_drivers["id"].astype(str)
    selectable_constructors["id"] = selectable_constructors["id"].astype(str)
    selectable_drivers["exp_score"] = selectable_drivers["id"].map(
        {"116": 30.0, "2": 10.0, "3": 9.0, "4": 8.0, "5": 7.0, "6": 6.0}
    )
    selectable_drivers["expected_price_gain"] = 0.1
    selectable_drivers["volatility"] = 5.0
    selectable_constructors["exp_score"] = selectable_constructors["id"].map(
        {"2": 18.0, "3": 25.0}
    )
    selectable_constructors["expected_price_gain"] = 0.2
    selectable_constructors["volatility"] = 8.0
    holding_drivers = app_core.build_holding_asset_universe(
        selectable_drivers,
        player_ledger,
        "driver",
    )
    holding_constructors = app_core.build_holding_asset_universe(
        selectable_constructors,
        constructor_ledger,
        "constructor",
    )
    return (
        player_ledger,
        constructor_ledger,
        selectable_drivers,
        selectable_constructors,
        holding_drivers,
        holding_constructors,
    )


def _current_ids() -> tuple[list[str], list[str]]:
    return ["114", "2", "3", "4", "5"], ["1", "2"]


def _upload(driver_ids=None, constructor_ids=None, *, bank: float = 1.0) -> bytes:
    return json.dumps(
        {
            "drivers": driver_ids or _current_ids()[0],
            "constructors": constructor_ids or _current_ids()[1],
            "bank": bank,
            "free_transfers": 2,
        }
    ).encode("utf-8")


def test_inactive_official_holdings_validate_and_use_exact_prices():
    *_, holding_drivers, holding_constructors = _universes()
    driver_ids, constructor_ids = _current_ids()

    result = app_core.validate_current_team(
        driver_ids,
        constructor_ids,
        holding_drivers,
        holding_constructors,
        budget=100.0,
    )

    assert result["valid"] is True
    assert result["unknown_driver_ids"] == []
    assert result["unknown_constructor_ids"] == []
    assert result["inactive_driver_ids"] == ["114"]
    assert result["inactive_constructor_ids"] == ["1"]
    assert result["total_cost"] == pytest.approx(92.0)
    assert result["valuation_complete"] is True
    old_lawson = result["selected_drivers"].set_index("id").loc["114"]
    assert old_lawson["price"] == pytest.approx(12.0)
    assert old_lawson["holding_status"] == "Inactive"
    assert "116" not in set(result["selected_drivers"]["id"])


def test_unknown_exact_id_is_not_fuzzy_or_human_remapped():
    *_, holding_drivers, holding_constructors = _universes()
    result = app_core.validate_current_team(
        ["999999", "2", "3", "4", "5"],
        ["1", "2"],
        holding_drivers,
        holding_constructors,
        budget=100.0,
    )

    assert result["valid"] is False
    assert result["unknown_driver_ids"] == ["999999"]
    assert "114" not in set(result["selected_drivers"]["id"])
    assert "116" not in set(result["selected_drivers"]["id"])


def test_missing_exact_price_is_an_incomplete_valuation_not_an_identity_substitution():
    *_, holding_drivers, holding_constructors = _universes()
    holding_drivers.loc[holding_drivers["id"] == "114", "price"] = pd.NA
    result = app_core.validate_current_team(
        *_current_ids(),
        holding_drivers,
        holding_constructors,
        budget=100.0,
    )

    assert result["valid"] is False
    assert result["valuation_complete"] is False
    assert result["missing_price_driver_ids"] == ["114"]
    assert result["unknown_driver_ids"] == []


def test_inactive_status_is_compact_in_labels_and_cards():
    *_, holding_drivers, _holding_constructors = _universes()
    labels = app_core.current_team_option_labels(holding_drivers, ["114"], "driver")
    card = app_core.fantasy_asset_card_html(
        holding_drivers[holding_drivers["id"] == "114"].iloc[0],
        asset_label="Driver",
    )

    assert labels["114"].endswith("· Inactive")
    assert "f1-availability-muted" in card
    assert ">Inactive<" in card
    assert "Unknown" not in labels["114"]


def test_import_accepts_inactive_official_assets_and_preserves_budget_ownership():
    *_, holding_drivers, holding_constructors = _universes()
    transition = app_core.current_team_upload_transition(
        _upload(),
        None,
        holding_drivers,
        holding_constructors,
    )
    existing = {"optimizer_budget": 140.0, "optimizer_budget_source": "manual"}
    resulting = {**existing, **transition["state_updates"]}

    assert transition["status"] == "success"
    assert resulting["current_team_driver_ids"] == ["114", "2", "3", "4", "5"]
    assert resulting["current_team_constructor_ids"] == ["1", "2"]
    assert resulting["current_team_bank"] == pytest.approx(1.0)
    assert resulting["optimizer_budget"] == pytest.approx(140.0)
    assert resulting["optimizer_budget_source"] == "manual"


def test_unknown_import_fails_atomically_without_same_human_replacement():
    *_, holding_drivers, holding_constructors = _universes()
    existing = {
        "current_team_driver_ids": ["6"],
        "current_team_constructor_ids": ["3"],
        "optimizer_budget": 140.0,
    }
    transition = app_core.current_team_upload_transition(
        _upload(driver_ids=[999999, 2, 3, 4, 5]),
        None,
        holding_drivers,
        holding_constructors,
    )

    assert transition["status"] == "error"
    assert transition["state_updates"] == {}
    assert {**existing, **transition["state_updates"]} == existing
    assert "999999" in transition["error"]


def test_active_holding_becoming_inactive_survives_refresh_and_rerun():
    before = _universes(old_lawson_active=1)[4]
    after = _universes(old_lawson_active=0)[4]
    selected_ids = ["114", "2", "3", "4", "5"]

    assert before.set_index("id").loc["114", "holding_status"] == "Active"
    assert after.set_index("id").loc["114", "holding_status"] == "Inactive"
    assert "114" not in set(_universes(old_lawson_active=0)[2]["id"])
    labels = app_core.current_team_option_labels(after, selected_ids, "driver")
    rerun_labels = app_core.current_team_option_labels(after.copy(deep=True), selected_ids, "driver")

    assert "114" in labels
    assert labels == rerun_labels
    assert selected_ids == ["114", "2", "3", "4", "5"]
    assert after.set_index("id").loc["114", "price"] == pytest.approx(12.0)


def test_old_inactive_lawson_can_transfer_to_new_same_human_asset():
    (
        player_ledger,
        _constructor_ledger,
        selectable_drivers,
        selectable_constructors,
        holding_drivers,
        holding_constructors,
    ) = _universes()
    mapping = build_player_identity_map(
        player_ledger,
        pd.DataFrame([{"driverId": "lawson", "driver": "Liam Lawson"}]),
    ).set_index("fantasy_asset_id")

    recs = app_core.build_transfer_recommendations(
        *_current_ids(),
        selectable_drivers,
        selectable_constructors,
        budget=100.0,
        free_transfers=1,
        max_transfers=1,
        search_mode="exhaustive",
        top_n=20,
        holding_drivers=holding_drivers,
        holding_constructors=holding_constructors,
    )
    lawson_move = next(
        row
        for moves in recs["Move rows"]
        for row in moves
        if row["asset_type"] == "driver"
        and row["out"]["id"] == "114"
        and row["in"]["id"] == "116"
    )
    matching_result = recs[
        recs["Move rows"].map(
            lambda moves: any(
                move["out"]["id"] == "114" and move["in"]["id"] == "116"
                for move in moves
            )
        )
    ].iloc[0]

    assert mapping.loc[114, "human_driver_id"] == mapping.loc[116, "human_driver_id"] == "lawson"
    assert lawson_move["out"]["price"] == pytest.approx(12.0)
    assert lawson_move["in"]["price"] == pytest.approx(14.5)
    assert matching_result["Team cost"] == pytest.approx(94.5)
    assert matching_result["Transfer penalty"] == pytest.approx(0.0)


def test_inactive_asset_is_outgoing_only_and_active_transfer_behavior_is_unchanged():
    (
        _player_ledger,
        _constructor_ledger,
        selectable_drivers,
        selectable_constructors,
        holding_drivers,
        holding_constructors,
    ) = _universes()
    inactive_recs = app_core.build_transfer_recommendations(
        *_current_ids(),
        selectable_drivers,
        selectable_constructors,
        budget=100.0,
        max_transfers=1,
        search_mode="exhaustive",
        top_n=50,
        holding_drivers=holding_drivers,
        holding_constructors=holding_constructors,
    )
    assert any("114" == move["out"]["id"] for moves in inactive_recs["Move rows"] for move in moves)
    assert all("114" != move["in"]["id"] for moves in inactive_recs["Move rows"] for move in moves)

    active_ids = ["2", "3", "4", "5", "6"]
    baseline = app_core.build_transfer_recommendations(
        active_ids,
        ["2", "3"],
        selectable_drivers,
        selectable_constructors,
        budget=100.0,
        max_transfers=1,
        search_mode="exhaustive",
        top_n=20,
    )
    explicit_holdings = app_core.build_transfer_recommendations(
        active_ids,
        ["2", "3"],
        selectable_drivers,
        selectable_constructors,
        budget=100.0,
        max_transfers=1,
        search_mode="exhaustive",
        top_n=20,
        holding_drivers=holding_drivers,
        holding_constructors=holding_constructors,
    )
    pd.testing.assert_frame_equal(baseline, explicit_holdings)


def test_imported_budget_repricing_uses_inactive_exact_asset_price():
    *_, holding_drivers, holding_constructors = _universes()
    repriced = app_core.reconcile_imported_budget_suggestion(
        *_current_ids(),
        1.0,
        holding_drivers,
        holding_constructors,
    )

    assert repriced["status"] == "available"
    assert repriced["suggestion"] == pytest.approx(93.0)
    assert repriced["missing_driver_ids"] == []
    assert repriced["missing_constructor_ids"] == []
