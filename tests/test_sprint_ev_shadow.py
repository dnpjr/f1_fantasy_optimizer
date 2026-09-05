from __future__ import annotations

import pandas as pd
import pytest

from f1fantasy import app_core
from f1fantasy.app_core import (
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    _shadow_forecast_diagnostics,
    apply_probabilistic_price_change_model,
)
from f1fantasy.model import expected_scores_horizon, expected_scores_horizon_by_component
from f1fantasy.optimize import optimize_top_k
from f1fantasy.race_selection import RaceKey
from f1fantasy.weekend_state import EventKey, UpcomingEvent, WeekendFormat


def _event(
    round_no: int,
    format_: WeekendFormat,
    *,
    weight: float = 1.0,
    circuit: str | None = None,
) -> UpcomingEvent:
    return UpcomingEvent(
        event=EventKey(2026, round_no),
        circuit=circuit or f"Future {round_no} Circuit",
        race_name=f"Future {round_no} Grand Prix",
        format=format_,
        scheduled_at=None,
        horizon_weight=weight,
    )


def _row(
    driver_id: str,
    constructor_id: str,
    round_no: int,
    *,
    qualifying: float,
    sprint: float | None,
    race: float,
    sprint_applicable: bool,
    sprint_observed: bool | None = None,
    circuit: str | None = None,
    season: int = 2026,
) -> dict:
    return {
        "season": season,
        "round": round_no,
        "circuitName": circuit or f"Past {round_no} Circuit",
        "driverId": driver_id,
        "driver": f"Driver {driver_id}",
        "constructorId": constructor_id,
        "constructor": f"Constructor {constructor_id}",
        "qualifying_points": qualifying,
        "quali_points": qualifying,
        "sprint_points": sprint,
        "race_points": race,
        "weekend_points": qualifying + (sprint or 0.0) + race,
        "q2_reached": 1,
        "q3_reached": 1,
        "is_dsq": 0,
        "is_dnf": 0,
        "sprint_is_dnf": 0,
        "sprint_applicable": sprint_applicable,
        "sprint_observed": sprint_observed if sprint_observed is not None else sprint is not None,
    }


def _single_asset_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("d1", "c1", 1, qualifying=5.0, sprint=3.0, race=10.0, sprint_applicable=True),
            _row("d1", "c1", 2, qualifying=5.0, sprint=None, race=10.0, sprint_applicable=False),
        ]
    )


def test_normal_and_sprint_shadow_totals_differ_only_by_sprint_component():
    points = _single_asset_history()
    original = points.copy(deep=True)

    normal, _ = expected_scores_horizon_by_component(
        points,
        [_event(3, WeekendFormat.NORMAL)],
        current_season_weight=1.0,
        past_season_weight=0.0,
        recency_decay=1.0,
        current_season=2026,
    )
    sprint, _ = expected_scores_horizon_by_component(
        points,
        [_event(3, WeekendFormat.SPRINT)],
        current_season_weight=1.0,
        past_season_weight=0.0,
        recency_decay=1.0,
        current_season=2026,
    )

    normal_row = normal.iloc[0]
    sprint_row = sprint.iloc[0]
    assert normal_row["shadow_next_sprint_ev"] == 0.0
    assert normal_row["shadow_next_total_ev"] == pytest.approx(
        normal_row["shadow_next_qualifying_ev"] + normal_row["shadow_next_race_ev"]
    )
    assert sprint_row["shadow_next_total_ev"] == pytest.approx(
        sprint_row["shadow_next_qualifying_ev"]
        + sprint_row["shadow_next_sprint_ev"]
        + sprint_row["shadow_next_race_ev"]
    )
    assert sprint_row["shadow_next_total_ev"] - normal_row["shadow_next_total_ev"] == pytest.approx(
        sprint_row["shadow_next_sprint_ev"]
    )
    pd.testing.assert_frame_equal(points, original)


def test_sprint_events_contribute_only_at_their_weighted_horizon_positions():
    points = _single_asset_history()
    normal_events = [
        _event(3, WeekendFormat.NORMAL, weight=1.0),
        _event(4, WeekendFormat.NORMAL, weight=0.7),
        _event(5, WeekendFormat.NORMAL, weight=0.7),
    ]
    middle_sprint = [
        normal_events[0],
        _event(4, WeekendFormat.SPRINT, weight=0.7),
        normal_events[2],
    ]
    two_sprints = [
        _event(3, WeekendFormat.SPRINT, weight=1.0),
        normal_events[1],
        _event(5, WeekendFormat.SPRINT, weight=0.7),
    ]

    normal, _ = expected_scores_horizon_by_component(points, normal_events, current_season_weight=1, past_season_weight=0)
    middle, _ = expected_scores_horizon_by_component(points, middle_sprint, current_season_weight=1, past_season_weight=0)
    double, _ = expected_scores_horizon_by_component(points, two_sprints, current_season_weight=1, past_season_weight=0)
    sprint_ev = float(double.loc[0, "shadow_next_sprint_ev"])
    assert sprint_ev == pytest.approx(3.0)

    assert middle.loc[0, "shadow_horizon_total_ev"] - normal.loc[0, "shadow_horizon_total_ev"] == pytest.approx(0.7 * sprint_ev)
    assert double.loc[0, "shadow_horizon_total_ev"] - normal.loc[0, "shadow_horizon_total_ev"] == pytest.approx(1.7 * sprint_ev)


def test_sprint_recency_is_contiguous_over_completed_sprint_events_only():
    points = pd.DataFrame(
        [
            _row("d1", "c1", 1, qualifying=5, sprint=10, race=10, sprint_applicable=True),
            _row("d1", "c1", 2, qualifying=5, sprint=None, race=10, sprint_applicable=False),
            _row("d1", "c1", 3, qualifying=5, sprint=30, race=10, sprint_applicable=True),
        ]
    )
    drivers, _ = expected_scores_horizon_by_component(
        points,
        [_event(4, WeekendFormat.SPRINT)],
        current_season_weight=1,
        past_season_weight=0,
        recency_decay=0.5,
        selected_race_keys=[RaceKey(2026, 1), RaceKey(2026, 2), RaceKey(2026, 3)],
        current_season=2026,
    )

    assert drivers.loc[0, "shadow_sprint_current_valid_count"] == 2
    assert drivers.loc[0, "shadow_sprint_current_estimate"] == pytest.approx(
        (10 * 0.5 + 30) / 1.5
    )


def test_missing_asset_sprint_history_uses_explicit_field_fallback_without_zero_fill():
    points = pd.DataFrame(
        [
            _row("d1", "c1", 1, qualifying=5, sprint=3, race=10, sprint_applicable=True),
            _row("d2", "c2", 1, qualifying=8, sprint=None, race=12, sprint_applicable=False),
        ]
    )
    drivers, constructors = expected_scores_horizon_by_component(
        points,
        [_event(2, WeekendFormat.SPRINT)],
        current_season_weight=1,
        past_season_weight=0,
        current_season=2026,
    )
    missing = drivers.set_index("driverId").loc["d2"]

    assert missing["shadow_sprint_current_valid_count"] == 0
    assert missing["shadow_next_sprint_source"] == "field_sprint_to_non_sprint_ratio_fallback"
    assert missing["shadow_next_sprint_ev"] == pytest.approx(4.0)
    assert missing["shadow_next_sprint_ev"] != 0
    assert constructors.set_index("constructorId").loc["c2", "shadow_next_sprint_source"] == (
        "field_sprint_to_non_sprint_ratio_fallback"
    )


def test_replacement_driver_uses_only_their_valid_sprint_observation():
    points = pd.DataFrame(
        [
            _row("regular", "c1", 1, qualifying=5, sprint=2, race=10, sprint_applicable=True),
            _row("replacement", "c2", 1, qualifying=6, sprint=None, race=9, sprint_applicable=True, sprint_observed=False),
            _row("replacement", "c2", 3, qualifying=7, sprint=8, race=11, sprint_applicable=True),
        ]
    )
    drivers, _ = expected_scores_horizon_by_component(
        points,
        [_event(4, WeekendFormat.SPRINT)],
        current_season_weight=1,
        past_season_weight=0,
        current_season=2026,
    )
    replacement = drivers.set_index("driverId").loc["replacement"]

    assert replacement["shadow_sprint_current_valid_count"] == 1
    assert replacement["shadow_sprint_current_estimate"] == pytest.approx(8.0)


def test_unavailable_sprint_field_history_stays_unavailable_on_sprint_event():
    points = pd.DataFrame(
        [_row("d1", "c1", 1, qualifying=5, sprint=None, race=10, sprint_applicable=False)]
    )
    drivers, constructors = expected_scores_horizon_by_component(
        points,
        [_event(2, WeekendFormat.SPRINT)],
        current_season_weight=1,
        past_season_weight=0,
        current_season=2026,
    )

    assert pd.isna(drivers.loc[0, "shadow_next_sprint_ev"])
    assert pd.isna(drivers.loc[0, "shadow_next_total_ev"])
    assert drivers.loc[0, "shadow_next_sprint_source"] == "unavailable"
    assert pd.isna(constructors.loc[0, "shadow_next_total_ev"])


def test_driver_and_constructor_component_estimates_are_independent():
    points = pd.DataFrame(
        [
            _row("d1", "c1", 1, qualifying=5, sprint=8, race=10, sprint_applicable=True),
            _row("d2", "c1", 1, qualifying=4, sprint=2, race=8, sprint_applicable=True),
        ]
    )
    drivers, constructors = expected_scores_horizon_by_component(
        points,
        [_event(2, WeekendFormat.SPRINT)],
        current_season_weight=1,
        past_season_weight=0,
        current_season=2026,
    )

    assert constructors.loc[0, "shadow_next_sprint_ev"] == pytest.approx(10.0)
    assert constructors.loc[0, "shadow_next_sprint_ev"] not in drivers["shadow_next_sprint_ev"].tolist()
    assert constructors.loc[0, "shadow_next_qualifying_ev"] > drivers["shadow_next_qualifying_ev"].max()


def test_shadow_calls_do_not_change_legacy_production_forecast():
    points = _single_asset_history()
    legacy_before, constructors_before = expected_scores_horizon(points, ["Future 3"], [1.0])

    expected_scores_horizon_by_component(points, [_event(3, WeekendFormat.NORMAL)])
    expected_scores_horizon_by_component(points, [_event(3, WeekendFormat.SPRINT)])
    legacy_after, constructors_after = expected_scores_horizon(points, ["Future 3"], [1.0])

    pd.testing.assert_frame_equal(legacy_after, legacy_before)
    pd.testing.assert_frame_equal(constructors_after, constructors_before)


def test_shadow_columns_cannot_change_price_outputs_ranked_team_or_captain():
    drivers = pd.DataFrame(
        [
            {
                "id": f"d{index}",
                "name": f"Driver {index}",
                "price": 10.0,
                "exp_score": score,
                "volatility": 5.0,
                "dnf_rate": 0.05,
                "recent_points_2ago": score - 2,
                "recent_points_1ago": score - 1,
            }
            for index, score in enumerate([50.0, 45.0, 40.0, 35.0, 30.0, 5.0], start=1)
        ]
    )
    constructors = pd.DataFrame(
        [
            {
                "id": f"c{index}",
                "name": f"Constructor {index}",
                "price": 10.0,
                "exp_score": score,
            }
            for index, score in enumerate([40.0, 35.0, 1.0], start=1)
        ]
    )
    shadow_drivers = drivers.assign(
        shadow_next_total_ev=[1, 2, 3, 4, 5, 1000],
        shadow_next_qualifying_ev=1.0,
        shadow_next_sprint_ev=1.0,
        shadow_next_race_ev=1.0,
        sprint_ev_uplift_vs_legacy=999.0,
    )
    shadow_constructors = constructors.assign(shadow_next_total_ev=[1, 2, 1000])

    base_price = apply_probabilistic_price_change_model(
        drivers,
        DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    )
    shadow_price = apply_probabilistic_price_change_model(
        shadow_drivers,
        DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    )
    price_columns = [
        column
        for column in base_price.columns
        if column.startswith("price_change_")
        or column.startswith("prob_")
        or column in {"expected_price_change", "projected_price"}
    ]
    pd.testing.assert_frame_equal(shadow_price[price_columns], base_price[price_columns])

    base_team = optimize_top_k(drivers, constructors, budget=100.0, k=1)[0]
    shadow_team = optimize_top_k(shadow_drivers, shadow_constructors, budget=100.0, k=1)[0]
    assert shadow_team.drivers["id"].tolist() == base_team.drivers["id"].tolist()
    assert shadow_team.constructors["id"].tolist() == base_team.constructors["id"].tolist()
    assert shadow_team.boosted_driver == base_team.boosted_driver
    assert shadow_team.expected_score == pytest.approx(base_team.expected_score)


def test_shadow_diagnostic_render_payload_is_a_pure_summary():
    frame = pd.DataFrame(
        [
            {
                "driverId": "d1",
                "name": "Driver One",
                "next_race_expected_points": 15.0,
                "shadow_next_qualifying_ev": 5.0,
                "shadow_next_sprint_ev": 3.0,
                "shadow_next_race_ev": 10.0,
                "shadow_next_total_ev": 18.0,
                "sprint_ev_uplift_vs_legacy": 3.0,
                "shadow_next_sprint_source": "current_only",
                "shadow_component_status": "complete",
            }
        ]
    )
    original = frame.copy(deep=True)

    summary = _shadow_forecast_diagnostics(frame, id_column="driverId", name_column="name")

    assert summary["means"]["legacy_next_event_ev"] == 15.0
    assert summary["means"]["shadow_total_ev"] == 18.0
    assert summary["source_counts"]["shadow_next_sprint_source"] == {"current_only": 1}
    pd.testing.assert_frame_equal(frame, original)


def test_derived_payload_exposes_event_records_and_shadow_fields_without_replacing_legacy_ev():
    results = pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "circuitName": "Historical Circuit",
                "driverId": "d1",
                "driver": "Driver One",
                "constructorId": "c1",
                "constructor": "Ferrari",
                "position": 1,
                "grid": 1,
                "status": "Finished",
            }
        ]
    )
    qualifying = pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "driverId": "d1",
                "position": 1,
                "q1": "1:20.000",
                "q2": "1:19.000",
                "q3": "1:18.000",
            }
        ]
    )
    sprint = pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "driverId": "d1",
                "position": 1,
                "grid": 1,
                "status": "Finished",
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 1,
                "raceName": "Upcoming Sprint Grand Prix",
                "circuitName": "Upcoming Circuit",
                "date": "2026-06-07",
                "time": "14:00:00Z",
                "qualifying_date": "2026-06-06",
                "qualifying_time": "14:00:00Z",
                "sprint_date": "2026-06-06",
                "sprint_time": "10:00:00Z",
                "sprint_qualifying_date": "2026-06-05",
                "sprint_qualifying_time": "15:00:00Z",
            }
        ]
    )
    snapshot = app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2025,
        requested_seasons=(2025, 2026),
        loaded_seasons=(2025, 2026),
        season_load_failures={},
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        schedule=schedule,
        players=pd.DataFrame(
            [{"playerId": 1, "name": "Driver One", "team": "Ferrari", "price": 20.0}]
        ),
        teams=pd.DataFrame([{"teamId": 11, "name": "Ferrari", "price": 30.0}]),
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={"feed_round": 1, "playerstats_prefetch_enabled": False},
    )

    derived = app_core.derive_model_data(
        snapshot,
        effective_time="2026-06-01T12:00:00Z",
        historical_seasons_back=1,
        horizon_races=1,
    )

    assert derived.diagnostics["upcoming_event_records"][0]["weekend_format"] == "sprint"
    assert derived.diagnostics["upcoming_event_records"][0]["season"] == 2026
    assert derived.diagnostics["sprint_aware_shadow_forecast"]["active_event"]["round"] == 1
    assert "shadow_next_total_ev" in derived.drivers.columns
    assert "sprint_aware_next_event_ev" in derived.drivers.columns
    assert "shadow_normal_ev" in derived.drivers.columns
    assert "shadow_sprint_bonus" in derived.drivers.columns
    assert "shadow_sprint_ev" in derived.drivers.columns
    assert derived.drivers.loc[0, "exp_score"] == pytest.approx(
        derived.drivers.loc[0, "next_race_expected_points"]
    )
    assert derived.diagnostics["sprint_aware_shadow_forecast"]["production_isolation"].startswith(
        "Legacy next_race_expected_points"
    )
    assert derived.diagnostics["approved_sprint_ev_shadow"]["sprint_shadow_history"] == "2026_only"
    assert "Only shadow_* columns" in derived.diagnostics[
        "approved_sprint_ev_shadow"
    ]["production_isolation"]
