from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from f1fantasy import app_core
from f1fantasy.asset_identity import build_player_identity_map
from f1fantasy.live_session_shadow import (
    CONSTRUCTOR_LIVE_SHADOW_COLUMNS,
    DRIVER_LIVE_SHADOW_COLUMNS,
    assign_ev_ladder,
    build_constructor_live_shadow,
    build_driver_live_shadow,
    build_live_session_shadow,
)
from scripts.compare_sprint_shadow_to_production import (
    deterministic_budget,
    load_offline_snapshot,
)
from f1fantasy.weekend_state import (
    EventKey,
    SessionKind,
    SessionState,
    SessionStatus,
    WeekendFormat,
)


EVENT = EventKey(2026, 12)


def _state(
    kind: SessionKind,
    status: SessionStatus = SessionStatus.COMPLETE,
    rows: int = 4,
) -> SessionState:
    return SessionState(
        event=EVENT,
        kind=kind,
        scheduled_at=None,
        observed_row_count=rows,
        expected_participant_count=rows,
        status=status,
        source="test",
    )


def _session(kind: SessionKind, ordered_humans: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": EVENT.season,
                "round": EVENT.round,
                "session_kind": kind.value,
                "human_driver_id": human_id,
                "position": position,
                "is_classified": True,
            }
            for position, human_id in enumerate(ordered_humans, start=1)
        ]
    )


def _drivers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": "1", "name": "Driver A", "human_driver_id": "a", "team": "Alpha", "next_race_expected_points": 40.0, "exp_score": 40.0, "price": 20.0, "is_active": 1},
            {"id": "2", "name": "Driver B", "human_driver_id": "b", "team": "Alpha", "next_race_expected_points": 35.0, "exp_score": 35.0, "price": 19.0, "is_active": 1},
            {"id": "3", "name": "Driver C", "human_driver_id": "c", "team": "Beta", "next_race_expected_points": 30.0, "exp_score": 30.0, "price": 18.0, "is_active": 1},
            {"id": "4", "name": "Driver D", "human_driver_id": "d", "team": "Beta", "next_race_expected_points": 25.0, "exp_score": 25.0, "price": 17.0, "is_active": 1},
        ]
    )


def _constructors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": "11", "name": "Alpha", "next_race_expected_points": 50.0, "exp_score": 50.0, "price": 30.0},
            {"id": "12", "name": "Beta", "next_race_expected_points": 45.0, "exp_score": 45.0, "price": 28.0},
            {"id": "13", "name": "Gamma", "next_race_expected_points": 35.0, "exp_score": 35.0, "price": 20.0},
        ]
    )


def test_driver_shadow_uses_only_complete_sessions_and_exposes_per_session_scores():
    results = pd.concat(
        [
            _session(SessionKind.PRACTICE_1, ["a", "b", "c", "d"]),
            _session(SessionKind.PRACTICE_2, ["d", "c", "b", "a"]),
        ],
        ignore_index=True,
    )
    shadow, normalised, diagnostics = build_driver_live_shadow(
        _drivers(),
        results,
        (
            _state(SessionKind.PRACTICE_1),
            _state(SessionKind.PRACTICE_2, SessionStatus.PARTIAL),
        ),
        WeekendFormat.NORMAL,
    )

    by_human = shadow.set_index("human_driver_id")
    assert by_human.loc["a", "FP1_score"] == 1.0
    assert pd.isna(by_human.loc["a", "FP2_score"])
    assert by_human.loc["a", "live_session_score"] == 1.0
    assert by_human.loc["a", "sessions_used"] == (SessionKind.PRACTICE_1.value,)
    assert set(normalised["session_kind"]) == {SessionKind.PRACTICE_1.value}
    assert diagnostics["scored_active_driver_count"] == 4


@pytest.mark.parametrize("status", [SessionStatus.PARTIAL, SessionStatus.PROVISIONAL, SessionStatus.IN_PROGRESS])
def test_noncomplete_session_is_ignored(status):
    shadow, normalised, diagnostics = build_driver_live_shadow(
        _drivers(),
        _session(SessionKind.PRACTICE_1, ["a", "b", "c", "d"]),
        (_state(SessionKind.PRACTICE_1, status),),
        WeekendFormat.NORMAL,
    )

    assert normalised.empty
    assert shadow["live_session_score"].isna().all()
    assert diagnostics["status"] == "unavailable"


def test_driver_ev_ladder_assigns_baseline_multiset_by_live_rank():
    shadow, _normalised, diagnostics = build_driver_live_shadow(
        _drivers(),
        _session(SessionKind.PRACTICE_1, ["d", "c", "b", "a"]),
        (_state(SessionKind.PRACTICE_1),),
        WeekendFormat.NORMAL,
    )
    by_human = shadow.set_index("human_driver_id")

    assert by_human.loc["d", "live_session_rank"] == 1
    assert by_human.loc["d", "live_only_ev"] == 40.0
    assert by_human.loc["a", "live_session_rank"] == 4
    assert by_human.loc["a", "live_only_ev"] == 25.0
    assert sorted(shadow["live_only_ev"].dropna()) == sorted(shadow["baseline_ev"])
    assert diagnostics["driver_ladder_multiset_preserved"] is True


def test_ev_ladder_ties_use_baseline_then_stable_human_id():
    assets = pd.DataFrame(
        [
            {"human_driver_id": "z", "live_session_score": 0.5, "baseline_ev": 20.0},
            {"human_driver_id": "a", "live_session_score": 0.5, "baseline_ev": 30.0},
            {"human_driver_id": "b", "live_session_score": 0.5, "baseline_ev": 30.0},
        ]
    )

    ranked = assign_ev_ladder(assets, stable_id_column="human_driver_id")

    assert ranked.set_index("human_driver_id")["live_session_rank"].to_dict() == {
        "z": 3,
        "a": 1,
        "b": 2,
    }
    assert ranked.set_index("human_driver_id")["live_only_ev"].to_dict() == {
        "z": 20.0,
        "a": 30.0,
        "b": 30.0,
    }


def test_missing_live_driver_stays_missing_and_does_not_receive_ladder_value():
    drivers = _drivers()
    results = _session(SessionKind.PRACTICE_1, ["a", "b", "c"])
    shadow, _normalised, _diagnostics = build_driver_live_shadow(
        drivers,
        results,
        (_state(SessionKind.PRACTICE_1, rows=3),),
        WeekendFormat.NORMAL,
    )
    missing = shadow.set_index("human_driver_id").loc["d"]

    assert pd.isna(missing["live_session_score"])
    assert pd.isna(missing["live_session_rank"])
    assert pd.isna(missing["live_only_ev"])
    assert sorted(shadow["live_only_ev"].dropna()) == [30.0, 35.0, 40.0]


def test_driver_missing_from_one_complete_session_renormalises_available_sessions():
    results = pd.concat(
        [
            _session(SessionKind.PRACTICE_1, ["a", "b", "c"]),
            _session(SessionKind.PRACTICE_2, ["a", "c"]),
        ],
        ignore_index=True,
    )
    shadow, _normalised, _diagnostics = build_driver_live_shadow(
        _drivers().iloc[:3],
        results,
        (
            _state(SessionKind.PRACTICE_1, rows=3),
            _state(SessionKind.PRACTICE_2, rows=2),
        ),
        WeekendFormat.NORMAL,
    )
    by_human = shadow.set_index("human_driver_id")

    assert by_human.loc["b", "FP1_score"] == 0.5
    assert pd.isna(by_human.loc["b", "FP2_score"])
    assert by_human.loc["b", "live_session_score"] == 0.5
    assert by_human.loc["b", "sessions_used"] == (SessionKind.PRACTICE_1.value,)
    assert by_human.loc["b", "weight_sum"] == 1.0


def test_same_human_new_asset_scores_only_active_current_asset():
    drivers = pd.DataFrame(
        [
            {"id": "old", "name": "Liam Lawson", "human_driver_id": "lawson", "team": "Racing Bulls", "next_race_expected_points": 18.0, "is_active": 0},
            {"id": "new", "name": "Liam Lawson", "human_driver_id": "lawson", "team": "Red Bull", "next_race_expected_points": 30.0, "is_active": 1},
            {"id": "other", "name": "Other Driver", "human_driver_id": "other", "team": "Red Bull", "next_race_expected_points": 25.0, "is_active": 1},
        ]
    )
    shadow, _normalised, diagnostics = build_driver_live_shadow(
        drivers,
        _session(SessionKind.PRACTICE_1, ["lawson", "other"]),
        (_state(SessionKind.PRACTICE_1, rows=2),),
        WeekendFormat.NORMAL,
    )
    by_id = shadow.set_index("id")

    assert pd.isna(by_id.loc["old", "live_session_score"])
    assert pd.isna(by_id.loc["old", "live_only_ev"])
    assert by_id.loc["new", "live_session_score"] == 1.0
    assert by_id.loc["new", "live_only_ev"] == 30.0
    assert diagnostics["ambiguous_active_human_ids"] == []


def test_same_human_new_asset_uses_new_current_constructor_assignment():
    drivers = pd.DataFrame(
        [
            {"id": "old", "name": "Liam Lawson", "human_driver_id": "lawson", "team": "Racing Bulls", "next_race_expected_points": 18.0, "is_active": 0},
            {"id": "new", "name": "Liam Lawson", "human_driver_id": "lawson", "team": "Red Bull Racing", "next_race_expected_points": 30.0, "is_active": 1},
            {"id": "mate", "name": "Team Mate", "human_driver_id": "mate", "team": "Red Bull Racing", "next_race_expected_points": 25.0, "is_active": 1},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "rb", "name": "Red Bull Racing", "next_race_expected_points": 50.0},
            {"id": "vcarb", "name": "Racing Bulls", "next_race_expected_points": 30.0},
        ]
    )
    driver_shadow, _normalised, _diagnostics = build_driver_live_shadow(
        drivers,
        _session(SessionKind.PRACTICE_1, ["lawson", "mate"]),
        (_state(SessionKind.PRACTICE_1, rows=2),),
        WeekendFormat.NORMAL,
    )
    constructor_shadow, _diagnostics = build_constructor_live_shadow(
        constructors, driver_shadow
    )
    by_name = constructor_shadow.set_index("name")

    assert by_name.loc["Red Bull Racing", "driver_coverage"] == "2/2"
    assert by_name.loc["Red Bull Racing", "live_session_score"] == 0.5
    assert by_name.loc["Racing Bulls", "driver_coverage"] == "0/2"
    assert pd.isna(by_name.loc["Racing Bulls", "live_session_score"])


def test_ambiguous_active_same_human_assets_are_diagnostic_not_duplicated():
    drivers = pd.DataFrame(
        [
            {"id": "one", "human_driver_id": "lawson", "next_race_expected_points": 30.0, "is_active": 1},
            {"id": "two", "human_driver_id": "lawson", "next_race_expected_points": 25.0, "is_active": 1},
        ]
    )
    shadow, _normalised, diagnostics = build_driver_live_shadow(
        drivers,
        _session(SessionKind.PRACTICE_1, ["lawson", "other"]),
        (_state(SessionKind.PRACTICE_1, rows=2),),
        WeekendFormat.NORMAL,
    )

    assert shadow["live_session_score"].isna().all()
    assert diagnostics["ambiguous_active_human_ids"] == ["lawson"]


def test_unresolved_session_identity_keeps_field_size_but_receives_no_score():
    drivers = _drivers().iloc[:2].copy()
    results = pd.DataFrame(
        [
            {"season": 2026, "round": 12, "session_kind": "practice_1", "human_driver_id": "a", "identity_match_status": "matched", "position": 1, "is_classified": True},
            {"season": 2026, "round": 12, "session_kind": "practice_1", "human_driver_id": pd.NA, "identity_match_status": "ambiguous", "position": 2, "is_classified": True},
            {"season": 2026, "round": 12, "session_kind": "practice_1", "human_driver_id": "b", "identity_match_status": "matched", "position": 3, "is_classified": True},
        ]
    )
    shadow, normalised, diagnostics = build_driver_live_shadow(
        drivers,
        results,
        (_state(SessionKind.PRACTICE_1, rows=3),),
        WeekendFormat.NORMAL,
    )
    by_human = shadow.set_index("human_driver_id")

    assert len(normalised) == 3
    assert by_human.loc["a", "live_session_score"] == 1.0
    assert by_human.loc["b", "live_session_score"] == 0.0
    assert diagnostics["sessions"][0]["identity_unavailable_rows"] == 1


def test_constructor_score_averages_two_current_driver_scores():
    driver_shadow, _normalised, _diagnostics = build_driver_live_shadow(
        _drivers(),
        _session(SessionKind.PRACTICE_1, ["a", "c", "b", "d"]),
        (_state(SessionKind.PRACTICE_1),),
        WeekendFormat.NORMAL,
    )
    constructors, diagnostics = build_constructor_live_shadow(_constructors(), driver_shadow)
    by_name = constructors.set_index("name")

    assert by_name.loc["Alpha", "live_session_score"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert by_name.loc["Alpha", "driver_coverage"] == "2/2"
    assert by_name.loc["Beta", "live_session_score"] == pytest.approx((2 / 3 + 0.0) / 2)
    assert pd.isna(by_name.loc["Gamma", "live_session_score"])
    assert by_name.loc["Gamma", "driver_coverage"] == "0/2"
    assert diagnostics["constructor_ladder_multiset_preserved"] is True


def test_constructor_with_one_valid_driver_uses_that_score_and_reports_one_of_two():
    drivers = _drivers()
    driver_shadow, _normalised, _diagnostics = build_driver_live_shadow(
        drivers,
        _session(SessionKind.PRACTICE_1, ["a", "c", "d"]),
        (_state(SessionKind.PRACTICE_1, rows=3),),
        WeekendFormat.NORMAL,
    )
    constructors, _diagnostics = build_constructor_live_shadow(_constructors(), driver_shadow)
    alpha = constructors.set_index("name").loc["Alpha"]

    assert alpha["driver_coverage"] == "1/2"
    assert alpha["live_session_score"] == 1.0


def test_constructor_ladder_preserves_scored_constructor_ev_multiset():
    shadow = build_live_session_shadow(
        _drivers(),
        _constructors(),
        _session(SessionKind.PRACTICE_1, ["c", "d", "a", "b"]),
        (_state(SessionKind.PRACTICE_1),),
        WeekendFormat.NORMAL,
    )

    scored = shadow.constructors[shadow.constructors["live_session_score"].notna()]
    assert sorted(scored["constructor_live_only_ev"]) == sorted(scored["baseline_ev"])
    assert shadow.constructors.set_index("name").loc["Beta", "constructor_live_session_rank"] == 1


def test_dutch_style_fp1_sq_weighting_and_human_identity_are_deterministic():
    humans = ["russell", "norris", "leclerc", "piastri", "antonelli", "lawson", "tsunoda", "hadjar"]
    drivers = pd.DataFrame(
        [
            {
                "id": str(index),
                "name": human.title(),
                "human_driver_id": human,
                "team": "Team A" if index % 2 else "Team B",
                "next_race_expected_points": float(50 - index),
                "exp_score": float(50 - index),
                "is_active": 1,
            }
            for index, human in enumerate(humans, start=1)
        ]
    )
    results = pd.concat(
        [
            _session(SessionKind.PRACTICE_1, ["antonelli", "lawson", "tsunoda", "russell", "norris", "leclerc", "piastri"]),
            _session(SessionKind.SPRINT_QUALIFYING, ["russell", "norris", "leclerc", "piastri", "antonelli", "lawson", "tsunoda"]),
        ],
        ignore_index=True,
    )
    shadow, _normalised, _diagnostics = build_driver_live_shadow(
        drivers,
        results,
        (
            _state(SessionKind.PRACTICE_1, rows=7),
            _state(SessionKind.SPRINT_QUALIFYING, rows=7),
        ),
        WeekendFormat.SPRINT,
    )
    by_human = shadow.set_index("human_driver_id")

    assert by_human.loc["russell", "live_session_score"] == pytest.approx(0.875)
    assert by_human.loc["norris", "live_session_score"] == pytest.approx(17 / 24)
    assert by_human.loc["leclerc", "live_session_score"] == pytest.approx(13 / 24)
    assert by_human.loc["piastri", "live_session_score"] == pytest.approx(3 / 8)
    assert by_human.loc["antonelli", "live_session_score"] == pytest.approx(0.5)
    assert by_human.loc["lawson", "live_session_score"] == pytest.approx(1 / 3)
    assert by_human.loc["tsunoda", "live_session_score"] == pytest.approx(1 / 6)
    assert by_human.loc["russell", "live_session_rank"] == 1
    assert by_human.loc["antonelli", "live_session_rank"] == 4
    assert by_human.loc["lawson", "live_session_rank"] == 6
    assert by_human.loc["tsunoda", "live_session_rank"] == 7
    assert pd.isna(by_human.loc["hadjar", "live_session_score"])
    assert pd.isna(by_human.loc["hadjar", "live_only_ev"])


def test_shadow_builder_does_not_mutate_inputs_or_production_columns():
    drivers = _drivers()
    constructors = _constructors()
    for frame in (drivers, constructors):
        frame["p_price_rise"] = 0.4
        frame["p_price_hold"] = 0.4
        frame["p_price_fall"] = 0.2
        frame["expected_price_gain"] = 0.1
    results = _session(SessionKind.PRACTICE_1, ["d", "c", "b", "a"])
    original_drivers = deepcopy(drivers)
    original_constructors = deepcopy(constructors)
    original_results = deepcopy(results)

    shadow = build_live_session_shadow(
        drivers,
        constructors,
        results,
        (_state(SessionKind.PRACTICE_1),),
        WeekendFormat.NORMAL,
    )

    pd.testing.assert_frame_equal(drivers, original_drivers)
    pd.testing.assert_frame_equal(constructors, original_constructors)
    pd.testing.assert_frame_equal(results, original_results)
    pd.testing.assert_frame_equal(shadow.drivers[original_drivers.columns], original_drivers)
    pd.testing.assert_frame_equal(
        shadow.constructors[original_constructors.columns], original_constructors
    )


def test_shadow_fields_do_not_change_optimizer_ranking_or_boost_selection():
    drivers = pd.DataFrame(
        [
            {
                "id": f"d{index}",
                "name": f"Driver {index}",
                "human_driver_id": f"driver_{index}",
                "team": ("Alpha", "Beta", "Gamma")[(index - 1) % 3],
                "next_race_expected_points": float(40 - index),
                "exp_score": float(40 - index),
                "price": float(8 + index),
                "is_active": 1,
                "p_price_rise": 0.4,
                "p_price_hold": 0.4,
                "p_price_fall": 0.2,
                "expected_price_gain": 0.1 * index,
            }
            for index in range(1, 7)
        ]
    )
    constructors = pd.DataFrame(
        [
            {
                "id": f"c{index}",
                "name": name,
                "next_race_expected_points": float(30 - index),
                "exp_score": float(30 - index),
                "price": float(18 + index),
                "p_price_rise": 0.4,
                "p_price_hold": 0.4,
                "p_price_fall": 0.2,
                "expected_price_gain": 0.2 * index,
            }
            for index, name in enumerate(("Alpha", "Beta", "Gamma"), start=1)
        ]
    )
    results = _session(
        SessionKind.PRACTICE_1,
        [f"driver_{index}" for index in (6, 5, 4, 3, 2, 1)],
    )
    state = _state(SessionKind.PRACTICE_1, rows=6)
    shadow = build_live_session_shadow(
        drivers, constructors, results, (state,), WeekendFormat.NORMAL
    )

    baseline_price_projection = app_core.apply_probabilistic_price_change_model(
        drivers,
        app_core.DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        expensive_rules=app_core.DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
        expensive_price_min=app_core.DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
        bounds=app_core.DEFAULT_PRICE_CHANGE_BOUNDS,
    )
    shadow_price_projection = app_core.apply_probabilistic_price_change_model(
        shadow.drivers,
        app_core.DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        expensive_rules=app_core.DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
        expensive_price_min=app_core.DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
        bounds=app_core.DEFAULT_PRICE_CHANGE_BOUNDS,
    )
    pd.testing.assert_frame_equal(
        shadow_price_projection[baseline_price_projection.columns],
        baseline_price_projection,
    )

    baseline_solutions = app_core.run_optimizer(
        drivers, constructors, budget=200.0, top_k=3, triple_multiplier=3.0
    )
    shadow_solutions = app_core.run_optimizer(
        shadow.drivers,
        shadow.constructors,
        budget=200.0,
        top_k=3,
        triple_multiplier=3.0,
    )

    def signatures(solutions):
        return [
            (
                tuple(sorted(solution.drivers["id"].astype(str))),
                tuple(sorted(solution.constructors["id"].astype(str))),
                solution.boosted_driver,
                solution.triple_driver,
                solution.expected_score,
                solution.total_cost,
            )
            for solution in solutions
        ]

    assert signatures(shadow_solutions) == signatures(baseline_solutions)


def test_derived_live_shadow_leaves_production_tables_and_ranked_teams_unchanged():
    snapshot, _metadata = load_offline_snapshot()
    snapshot.player_identity_map = build_player_identity_map(
        snapshot.players, snapshot.results
    )
    baseline_snapshot = deepcopy(snapshot)
    live_snapshot = deepcopy(snapshot)
    active_identities = snapshot.player_identity_map[
        snapshot.player_identity_map["active"].fillna(False).astype(bool)
        & snapshot.player_identity_map["human_driver_id"].notna()
    ].sort_values("human_driver_id", kind="stable")
    human_ids = active_identities["human_driver_id"].astype(str).tolist()
    assert len(human_ids) == 22
    live_snapshot.session_results = _session(
        SessionKind.PRACTICE_1, human_ids
    )
    live_snapshot.session_states = (
        _state(SessionKind.PRACTICE_1, rows=len(human_ids)),
    )
    settings = {
        "today": "2026-08-21",
        "effective_time": "2026-08-21T12:00:00Z",
        "historical_seasons_back": 3,
        "horizon_races": 5,
        "current_season_weight": 1.0,
        "past_season_weight": 0.7,
        "recency_decay": 0.85,
        "selected_race_preset": "All",
        "history_mode": app_core.HISTORY_MODE_ALL_SUPPORTED,
    }

    baseline = app_core.derive_model_data(baseline_snapshot, **settings)
    with_live = app_core.derive_model_data(live_snapshot, **settings)

    driver_production_columns = [
        column
        for column in baseline.drivers.columns
        if column not in DRIVER_LIVE_SHADOW_COLUMNS
    ]
    constructor_production_columns = [
        column
        for column in baseline.constructors.columns
        if column not in CONSTRUCTOR_LIVE_SHADOW_COLUMNS
        and column not in {"live_session_score", "live_session_rank", "live_only_ev"}
    ]
    pd.testing.assert_frame_equal(
        baseline.drivers[driver_production_columns],
        with_live.drivers[driver_production_columns],
    )
    pd.testing.assert_frame_equal(
        baseline.constructors[constructor_production_columns],
        with_live.constructors[constructor_production_columns],
    )
    pd.testing.assert_frame_equal(baseline.trends, with_live.trends)
    pd.testing.assert_frame_equal(
        baseline.driver_price_efficiency, with_live.driver_price_efficiency
    )
    pd.testing.assert_frame_equal(
        baseline.constructor_price_efficiency, with_live.constructor_price_efficiency
    )
    assert with_live.drivers["live_session_score"].notna().sum() == 22

    budget, _budget_metadata = deterministic_budget(
        baseline.drivers, baseline.constructors
    )
    baseline_solutions = app_core.run_optimizer(
        baseline.drivers,
        baseline.constructors,
        budget=budget,
        top_k=3,
        triple_multiplier=3.0,
    )
    live_solutions = app_core.run_optimizer(
        with_live.drivers,
        with_live.constructors,
        budget=budget,
        top_k=3,
        triple_multiplier=3.0,
    )

    def signatures(solutions):
        return [
            (
                tuple(sorted(solution.drivers["id"].astype(str))),
                tuple(sorted(solution.constructors["id"].astype(str))),
                solution.boosted_driver,
                solution.triple_driver,
                solution.expected_score,
                solution.total_cost,
            )
            for solution in solutions
        ]

    assert signatures(live_solutions) == signatures(baseline_solutions)
