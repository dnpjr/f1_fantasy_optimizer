from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from f1fantasy.model import _constructor_round_points, compute_weekend_points
from f1fantasy.weekend_state import (
    EventKey,
    WeekendFormat,
    upcoming_circuit_names,
    upcoming_event_records,
)


def _completed_source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": round_no,
                "circuitName": circuit,
                "driverId": driver_id,
                "driver": driver,
                "constructorId": "c1",
                "constructor": "Team One",
                "position": position,
                "grid": position,
                "status": "Finished",
            }
            for round_no, circuit in [(1, "Normal Circuit"), (2, "Sprint Circuit")]
            for driver_id, driver, position in [("d1", "Driver One", 1), ("d2", "Driver Two", 2)]
        ]
    )
    qualifying = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": round_no,
                "driverId": driver_id,
                "position": position,
                "q1": "1:20.000",
                "q2": "1:19.000",
                "q3": "1:18.000",
            }
            for round_no in [1, 2]
            for driver_id, position in [("d1", 1), ("d2", 2)]
        ]
    )
    sprint = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 2,
                "driverId": driver_id,
                "position": position,
                "grid": position,
                "status": "Finished",
            }
            for driver_id, position in [("d1", 1), ("d2", 2)]
        ]
    )
    return results, qualifying, sprint


def test_completed_driver_and_constructor_components_preserve_existing_totals():
    results, qualifying, sprint = _completed_source_frames()
    originals = tuple(frame.copy(deep=True) for frame in (results, qualifying, sprint))

    points = compute_weekend_points(
        results,
        qualifying,
        sprint,
        current_season=2026,
        completed_event_keys={EventKey(2026, 1), EventKey(2026, 2)},
        complete_qualifying_keys={EventKey(2026, 1), EventKey(2026, 2)},
        complete_sprint_keys={EventKey(2026, 2)},
    )

    assert points["weekend_points"].tolist() == pytest.approx(
        (points["qualifying_points"] + points["sprint_points"] + points["race_points"]).tolist()
    )
    assert points["weekend_points"].tolist() == pytest.approx(
        (points["quali_points"] + points["sprint_points"] + points["race_points"]).tolist()
    )
    normal = points[points["round"] == 1]
    assert normal["sprint_applicable"].eq(False).all()
    assert normal["sprint_points"].eq(0).all()

    constructors = _constructor_round_points(points)
    assert constructors["weekend_points"].tolist() == pytest.approx(
        (
            constructors["qualifying_points"]
            + constructors["sprint_points"]
            + constructors["race_points"]
        ).tolist()
    )
    sprint_constructor = constructors.loc[constructors["round"] == 2].iloc[0]
    assert sprint_constructor["qualifying_points"] > points.loc[points["round"] == 2, "qualifying_points"].sum()
    assert sprint_constructor["sprint_points"] == pytest.approx(
        points.loc[points["round"] == 2, "sprint_points"].sum()
    )

    for original, current in zip(originals, (results, qualifying, sprint)):
        pd.testing.assert_frame_equal(current, original)


def test_incomplete_or_running_sprint_weekend_never_creates_completed_component_rows():
    results, qualifying, sprint = _completed_source_frames()
    sprint_results = results[results["round"] == 2].copy()
    sprint_results.loc[sprint_results.index[0], "status"] = "Running"

    gated = compute_weekend_points(
        sprint_results,
        qualifying[qualifying["round"] == 2],
        sprint,
        current_season=2026,
        completed_event_keys=set(),
        complete_qualifying_keys=set(),
        complete_sprint_keys=set(),
    )
    assert gated.empty

    defensive = compute_weekend_points(
        sprint_results,
        qualifying[qualifying["round"] == 2],
        sprint,
        current_season=2026,
    )
    assert defensive.empty


def test_upcoming_event_records_are_canonical_ordered_immutable_and_format_aware():
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 3,
                "raceName": "Third Grand Prix",
                "circuitName": "Third Circuit",
                "date": "2026-03-22",
                "time": "14:00:00Z",
            },
            {
                "season": 2026,
                "round": 2,
                "raceName": "Sprint Grand Prix",
                "circuitName": "Sprint Circuit",
                "date": "2026-03-15",
                "time": "14:00:00Z",
                "sprint_date": "2026-03-14",
                "sprint_time": "11:00:00Z",
            },
            {
                "season": 2026,
                "round": 1,
                "raceName": "Opening Grand Prix",
                "circuitName": "Opening Circuit",
                "date": "2026-03-08",
                "time": "14:00:00Z",
            },
            {
                "season": 2026,
                "round": 2,
                "raceName": "Sprint Grand Prix duplicate metadata",
                "circuitName": "Sprint Circuit",
                "date": "2026-03-15",
                "time": "14:00:00Z",
                "sprint_date": "2026-03-14",
                "sprint_time": "11:00:00Z",
            },
        ]
    )
    original = schedule.copy(deep=True)

    events = upcoming_event_records(schedule, start_event=EventKey(2026, 1), limit=3)

    assert [(event.season, event.round) for event in events] == [(2026, 1), (2026, 2), (2026, 3)]
    assert [event.horizon_weight for event in events] == [1.0, 0.7, 0.7]
    assert [event.format for event in events] == [
        WeekendFormat.NORMAL,
        WeekendFormat.SPRINT,
        WeekendFormat.NORMAL,
    ]
    assert upcoming_circuit_names(events) == ["Opening", "Sprint", "Third"]
    assert events[1].race_name == "Sprint Grand Prix"
    with pytest.raises(AttributeError):
        events[0].horizon_weight = 2.0  # type: ignore[misc]
    pd.testing.assert_frame_equal(schedule, original)


def test_empty_upcoming_schedule_is_safe():
    assert upcoming_event_records(pd.DataFrame()) == ()
