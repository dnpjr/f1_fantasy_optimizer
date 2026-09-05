import json

import pandas as pd
import pytest

from f1fantasy import app_core
from f1fantasy.app_core import (
    current_team_upload_transition,
    optimizer_budget_state_updates,
    prepare_uploaded_team_import,
    reconcile_imported_budget_suggestion,
    should_process_upload,
    uploaded_file_hash,
)


def _drivers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": str(i), "name": f"Driver {i}", "price": 10.0, "exp_score": float(i)}
            for i in range(1, 7)
        ]
    )


def _constructors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": str(i), "name": f"Constructor {i}", "price": 10.0, "exp_score": float(i)}
            for i in range(1, 4)
        ]
    )


def _valid_upload(bank: float = 0.2) -> bytes:
    return json.dumps(
        {
            "drivers": [1, 2, 3, 4, 5],
            "constructors": [1, 2],
            "bank": bank,
            "free_transfers": 2,
        },
        sort_keys=True,
    ).encode("utf-8")


def test_uploaded_file_hash_is_deterministic_and_content_based():
    first = _valid_upload()
    same = bytes(first)
    different = _valid_upload(bank=0.3)

    assert uploaded_file_hash(first) == uploaded_file_hash(same)
    assert uploaded_file_hash(first) != uploaded_file_hash(different)


def test_should_process_upload_uses_the_previous_attempt_hash():
    digest = uploaded_file_hash(_valid_upload())

    assert should_process_upload(digest, None) is True
    assert should_process_upload(digest, digest) is False


def test_valid_upload_is_processed_once_and_new_content_is_processed():
    first = current_team_upload_transition(_valid_upload(), None, _drivers(), _constructors())
    retained = current_team_upload_transition(
        _valid_upload(),
        first["upload_hash"],
        _drivers(),
        _constructors(),
    )
    replacement = current_team_upload_transition(
        _valid_upload(bank=0.3),
        first["upload_hash"],
        _drivers(),
        _constructors(),
    )

    assert first["attempted"] is True
    assert first["status"] == "success"
    assert retained["attempted"] is False
    assert retained["state_updates"] == {}
    assert replacement["attempted"] is True
    assert replacement["status"] == "success"


def test_retained_invalid_upload_is_not_reparsed(monkeypatch):
    calls = []

    def fail_prepare(*_args, **_kwargs):
        calls.append("called")
        raise ValueError("bad upload")

    monkeypatch.setattr(app_core, "prepare_uploaded_team_import", fail_prepare)
    contents = b"{not valid json"
    first = current_team_upload_transition(contents, None, _drivers(), _constructors())
    retained = current_team_upload_transition(
        contents,
        first["upload_hash"],
        _drivers(),
        _constructors(),
    )

    assert first["attempted"] is True
    assert first["status"] == "error"
    assert retained["attempted"] is False
    assert calls == ["called"]


def test_invalid_json_has_no_import_state_updates():
    existing = {
        "current_team_driver_ids": ["6"],
        "current_team_constructor_ids": ["3"],
        "current_team_bank": 1.0,
        "optimizer_budget": 123.4,
        "optimizer_budget_source": "manual",
    }

    transition = current_team_upload_transition(
        b"{not valid json",
        None,
        _drivers(),
        _constructors(),
    )
    resulting = {**existing, **transition["state_updates"]}

    assert transition["status"] == "error"
    assert transition["state_updates"] == {}
    assert resulting == existing


def test_parsed_but_invalid_team_has_no_import_state_updates():
    invalid_team = json.dumps(
        {
            "drivers": [1, 2],
            "constructors": [1],
            "bank": 0.2,
            "free_transfers": 2,
        }
    ).encode("utf-8")

    transition = current_team_upload_transition(
        invalid_team,
        None,
        _drivers(),
        _constructors(),
    )

    assert transition["status"] == "error"
    assert "Select exactly 5 drivers" in transition["error"]
    assert transition["state_updates"] == {}


def test_missing_roster_ids_make_the_import_atomic():
    missing_driver = json.dumps(
        {
            "drivers": [1, 2, 3, 4, 999],
            "constructors": [1, 2],
            "bank": 0.2,
            "free_transfers": 2,
        }
    ).encode("utf-8")

    transition = current_team_upload_transition(
        missing_driver,
        None,
        _drivers(),
        _constructors(),
    )

    assert transition["status"] == "error"
    assert "Unknown driver IDs: ['999']" in transition["error"]
    assert transition["state_updates"] == {}


def test_successful_import_builds_current_team_state_and_budget_suggestion():
    prepared = prepare_uploaded_team_import(_valid_upload(), _drivers(), _constructors())
    transition = current_team_upload_transition(
        _valid_upload(),
        None,
        _drivers(),
        _constructors(),
    )
    updates = transition["state_updates"]

    assert prepared["team_cost"] == pytest.approx(70.0)
    assert prepared["budget_suggestion"] == pytest.approx(70.2)
    assert updates["current_team_driver_ids"] == ["1", "2", "3", "4", "5"]
    assert updates["current_team_constructor_ids"] == ["1", "2"]
    assert updates["current_team_bank"] == pytest.approx(0.2)
    assert updates["current_team_free_transfers"] == 2
    assert updates["current_team_budget"] == pytest.approx(70.2)
    assert updates["imported_budget_suggestion"] == pytest.approx(70.2)
    assert updates["imported_budget_driver_ids"] == ["1", "2", "3", "4", "5"]
    assert updates["imported_budget_constructor_ids"] == ["1", "2"]
    assert updates["imported_budget_bank"] == pytest.approx(0.2)


def test_successful_import_does_not_own_or_overwrite_optimizer_budget():
    state = {
        "optimizer_budget": 123.4,
        "optimizer_budget_source": "manual",
        "budget_user_overridden": True,
        "app_budget": 100.0,
    }
    transition = current_team_upload_transition(
        _valid_upload(),
        None,
        _drivers(),
        _constructors(),
    )
    resulting = {**state, **transition["state_updates"]}

    assert "optimizer_budget" not in transition["state_updates"]
    assert "optimizer_budget_source" not in transition["state_updates"]
    assert "budget_user_overridden" not in transition["state_updates"]
    assert "app_budget" not in transition["state_updates"]
    assert resulting["optimizer_budget"] == pytest.approx(123.4)
    assert resulting["optimizer_budget_source"] == "manual"
    assert resulting["budget_user_overridden"] is True


def test_manual_optimizer_budget_survives_unrelated_state_changes():
    state = {
        **optimizer_budget_state_updates(123.4, source="manual"),
        "current_team_budget": 70.2,
    }
    after_widgets = {
        **state,
        "top_k": 5,
        "objective": "Points only",
        "chip": "none",
        "locked_driver_ids": ["1"],
    }

    assert after_widgets["optimizer_budget"] == pytest.approx(123.4)
    assert after_widgets["optimizer_budget_source"] == "manual"
    assert after_widgets["current_team_budget"] == pytest.approx(70.2)


def test_explicit_acceptance_updates_only_optimizer_owned_state_and_persists():
    state = {
        "optimizer_budget": 123.4,
        "current_team_budget": 70.2,
        "app_budget": 100.0,
    }
    accepted = {
        **state,
        **optimizer_budget_state_updates(
            70.2,
            source="imported_accepted",
            accepted_import_hash="abc123",
        ),
    }
    after_rerun = {**accepted, "top_k": 3, "active_tab": "Price changes"}

    assert accepted["optimizer_budget"] == pytest.approx(70.2)
    assert accepted["optimizer_budget_source"] == "imported_accepted"
    assert accepted["optimizer_budget_accepted_import_hash"] == "abc123"
    assert accepted["budget_user_overridden"] is True
    assert accepted["current_team_budget"] == pytest.approx(70.2)
    assert accepted["app_budget"] == pytest.approx(100.0)
    assert after_rerun["optimizer_budget"] == pytest.approx(70.2)


def test_later_manual_optimizer_budget_takes_precedence_over_accepted_import():
    accepted = optimizer_budget_state_updates(
        70.2,
        source="imported_accepted",
        accepted_import_hash="abc123",
    )
    manual = {**accepted, **optimizer_budget_state_updates(125.0, source="manual")}

    assert manual["optimizer_budget"] == pytest.approx(125.0)
    assert manual["optimizer_budget_source"] == "manual"
    assert manual["optimizer_budget_accepted_import_hash"] is None


def test_current_team_and_optimizer_budgets_remain_independent():
    state = {
        "optimizer_budget": 140.0,
        "optimizer_budget_source": "manual",
        "current_team_budget": 100.0,
    }
    transition = current_team_upload_transition(
        _valid_upload(),
        None,
        _drivers(),
        _constructors(),
    )
    resulting = {**state, **transition["state_updates"]}

    assert resulting["current_team_budget"] == pytest.approx(70.2)
    assert resulting["optimizer_budget"] == pytest.approx(140.0)


def test_imported_budget_suggestion_reprices_without_reparsing_or_changing_optimizer_budget(monkeypatch):
    transition = current_team_upload_transition(
        _valid_upload(),
        None,
        _drivers(),
        _constructors(),
    )
    imported = transition["state_updates"]
    prepare_calls = []

    def unexpected_prepare(*_args, **_kwargs):
        prepare_calls.append("called")
        raise AssertionError("retained bytes must not be reparsed")

    monkeypatch.setattr(app_core, "prepare_uploaded_team_import", unexpected_prepare)
    retained = current_team_upload_transition(
        _valid_upload(),
        transition["upload_hash"],
        _drivers(),
        _constructors(),
    )
    refreshed_drivers = _drivers()
    refreshed_constructors = _constructors()
    refreshed_drivers["price"] = 11.0
    refreshed_constructors["price"] = 12.0
    state = {
        "optimizer_budget": 140.0,
        "optimizer_budget_source": "manual",
        **imported,
    }

    repriced = reconcile_imported_budget_suggestion(
        state["imported_budget_driver_ids"],
        state["imported_budget_constructor_ids"],
        state["imported_budget_bank"],
        refreshed_drivers,
        refreshed_constructors,
    )

    assert retained["attempted"] is False
    assert prepare_calls == []
    assert repriced["status"] == "available"
    assert repriced["suggestion"] == pytest.approx(79.2)
    assert state["optimizer_budget"] == pytest.approx(140.0)
    assert state["optimizer_budget_source"] == "manual"

    accepted = {
        **state,
        **optimizer_budget_state_updates(
            repriced["suggestion"],
            source="imported_accepted",
            accepted_import_hash="latest-import",
        ),
    }
    assert accepted["optimizer_budget"] == pytest.approx(79.2)
    assert accepted["optimizer_budget_source"] == "imported_accepted"


def test_missing_refreshed_asset_makes_imported_suggestion_unavailable():
    refreshed_drivers = _drivers()
    refreshed_drivers = refreshed_drivers[refreshed_drivers["id"] != "5"]
    state = {
        "optimizer_budget": 140.0,
        "optimizer_budget_source": "manual",
    }

    repriced = reconcile_imported_budget_suggestion(
        ["1", "2", "3", "4", "5"],
        ["1", "2"],
        0.2,
        refreshed_drivers,
        _constructors(),
    )

    assert repriced["status"] == "incomplete"
    assert repriced["suggestion"] is None
    assert repriced["missing_driver_ids"] == ["5"]
    assert state["optimizer_budget"] == pytest.approx(140.0)
