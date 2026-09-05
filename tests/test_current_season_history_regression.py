from __future__ import annotations

import pandas as pd

from f1fantasy import app_core
from f1fantasy.historical_scores import (
    DEFAULT_CANONICAL_DATASET_PATH,
    canonical_market_snapshot,
    load_canonical_scores,
)
from f1fantasy.price_efficiency import build_price_efficiency_table
from f1fantasy.race_selection import RaceKey, resolve_selected_races
from f1fantasy.ui_helpers import reconcile_race_control_state


def _snapshot(*, raw_rounds: tuple[int, ...] = (2, 3)) -> app_core.LiveDataSnapshot:
    recorded = load_canonical_scores("data/research/sprint_round_11/canonical.csv")
    market = canonical_market_snapshot(recorded, 2026)
    race_names = {
        int(row.round): str(row.event_name)
        for row in recorded.loc[recorded["season"].eq(2026), ["round", "event_name"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    driver_id = int(market["players"].iloc[0]["playerId"])
    constructor_id = int(market["teams"].iloc[0]["teamId"])
    raw_rows = [
        {
            "PlayerId": asset_id,
            "asset_type": asset_type,
            "season": 2026,
            "round": round_no,
            "race_name": race_names[round_no],
            "fantasy_points": float(round_no),
            "is_played": 1,
        }
        for asset_type, asset_id in (
            ("driver", driver_id),
            ("constructor", constructor_id),
        )
        for round_no in raw_rounds
    ]
    raw = pd.DataFrame(
        raw_rows,
        columns=[
            "PlayerId",
            "asset_type",
            "season",
            "round",
            "race_name",
            "fantasy_points",
            "is_played",
        ],
    )
    return app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2023,
        requested_seasons=(2023, 2024, 2025, 2026),
        loaded_seasons=(2023, 2024, 2025, 2026),
        season_load_failures={},
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.read_csv("data/cache/schedule_2026.csv"),
        players=market["players"],
        teams=market["teams"],
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=raw[raw["asset_type"].eq("driver")].copy(),
        constructor_race_points=raw[raw["asset_type"].eq("constructor")].copy(),
        team_lock_payload={},
        source_diagnostics={
            "raw_live_load_finished_utc": "regression-fixture",
            "live_data_status": "generated_snapshot",
            "completed_current_event_keys": [(2026, 2), (2026, 3)],
        },
        historical_fantasy_scores=recorded,
    )


def test_canonical_rounds_one_to_eleven_reach_all_completed_race_selector():
    snapshot = _snapshot()

    catalogue, source = app_core.snapshot_race_catalogue(snapshot)
    selection = resolve_selected_races(catalogue, "All")
    lineage = app_core.current_season_round_lineage(snapshot, list(selection.included))

    assert source == "canonical_recorded_playerstats_union"
    assert [option.key.round for option in catalogue] == list(range(1, 12))
    assert [key.round for key in selection.included] == list(range(1, 12))
    assert lineage["available_to_race_selector"].all()
    assert lineage["used_by_projection_model"].all()
    assert lineage["exclusion_reason"].eq("").all()


def test_previous_seasons_and_colliding_names_or_source_ids_do_not_truncate_2026():
    snapshot = _snapshot()
    current = snapshot.historical_fantasy_scores[
        snapshot.historical_fantasy_scores["season"].eq(2026)
    ].iloc[[0]].copy()
    collision = current.copy()
    collision["season"] = 2025
    collision["round"] = 99
    snapshot.historical_fantasy_scores = pd.concat(
        [snapshot.historical_fantasy_scores, collision], ignore_index=True
    )

    catalogue, _source = app_core.snapshot_race_catalogue(snapshot)

    assert [option.key for option in catalogue] == [RaceKey(2026, round_no) for round_no in range(1, 12)]


def test_market_and_incomplete_classification_do_not_define_recorded_score_history():
    snapshot = _snapshot(raw_rounds=())
    snapshot.source_diagnostics["completed_current_event_keys"] = []

    catalogue, _source = app_core.snapshot_race_catalogue(snapshot)
    driver_points, constructor_points = app_core.effective_current_race_points(snapshot)

    assert len(snapshot.players) == 22
    assert len(snapshot.teams) == 11
    assert [option.key.round for option in catalogue] == list(range(1, 12))
    assert sorted(driver_points["round"].unique()) == list(range(1, 12))
    assert sorted(constructor_points["round"].unique()) == list(range(1, 12))


def test_valid_totals_survive_missing_components_and_drive_price_efficiency():
    snapshot = _snapshot(raw_rounds=())
    driver_points, _constructor_points = app_core.effective_current_race_points(snapshot)
    catalogue, _source = app_core.snapshot_race_catalogue(snapshot)
    selection = resolve_selected_races(catalogue, "All")
    counts = driver_points.groupby("PlayerId")["round"].nunique()
    asset_id = str(counts[counts.eq(11)].index[0])
    market_row = snapshot.players[
        snapshot.players["playerId"].astype(str).eq(asset_id)
    ].iloc[[0]].rename(columns={"playerId": "id"})

    table = build_price_efficiency_table(
        market_row,
        driver_points,
        selection,
        asset_type="driver",
    )

    assert driver_points["fantasy_points"].notna().all()
    assert driver_points["qualifying_points"].isna().any()
    assert int(table.iloc[0]["selected_race_count"]) == 11
    assert int(table.iloc[0]["valid_race_count"]) == 11


def test_stale_two_race_selection_reconciles_against_expanded_canonical_universe():
    snapshot = _snapshot()
    catalogue, _source = app_core.snapshot_race_catalogue(snapshot)

    state = reconcile_race_control_state(
        catalogue,
        "All",
        [RaceKey(2026, 2), RaceKey(2026, 3)],
        [],
    )

    assert [key.round for key in state.selection.included] == list(range(1, 12))


def test_current_season_only_filters_history_without_touching_market_or_observations():
    snapshot = _snapshot()
    original_players = snapshot.players.copy(deep=True)
    original_scores = snapshot.historical_fantasy_scores.copy(deep=True)

    current = app_core.recorded_history_for_mode(
        snapshot.historical_fantasy_scores,
        2026,
        app_core.HISTORY_MODE_CURRENT_SEASON_ONLY,
    )
    all_supported = app_core.recorded_history_for_mode(
        snapshot.historical_fantasy_scores,
        2026,
        app_core.HISTORY_MODE_ALL_SUPPORTED,
    )
    current_coverage = app_core.season_coverage(
        snapshot,
        3,
        app_core.HISTORY_MODE_CURRENT_SEASON_ONLY,
    )
    catalogue, _source = app_core.snapshot_race_catalogue(snapshot)

    assert set(current["season"]) == {2026}
    assert set(all_supported["season"]) == {2023, 2024, 2025, 2026}
    assert current_coverage["used_seasons"] == (2026,)
    assert [option.key.round for option in catalogue] == list(range(1, 12))
    pd.testing.assert_frame_equal(snapshot.players, original_players)
    pd.testing.assert_frame_equal(snapshot.historical_fantasy_scores, original_scores)


def test_history_modes_have_distinct_model_identities():
    snapshot = _snapshot()
    common = dict(
        historical_seasons_back=3,
        horizon_races=5,
        current_season_weight=1.0,
        past_season_weight=0.7,
        recency_decay=0.85,
        effective_date="2026-08-06",
    )

    all_supported = app_core.model_settings_signature(
        snapshot,
        **common,
        history_mode=app_core.HISTORY_MODE_ALL_SUPPORTED,
    )
    current_only = app_core.model_settings_signature(
        snapshot,
        **common,
        history_mode=app_core.HISTORY_MODE_CURRENT_SEASON_ONLY,
    )

    assert all_supported != current_only
    assert app_core.HISTORY_MODE_ALL_SUPPORTED in all_supported
    assert app_core.HISTORY_MODE_CURRENT_SEASON_ONLY in current_only


def test_production_derivation_uses_all_eleven_rounds_in_both_history_modes():
    snapshot = _snapshot(raw_rounds=())
    snapshot.results = pd.concat(
        [pd.read_csv(f"data/cache/results_{year}.csv") for year in range(2023, 2027)],
        ignore_index=True,
    )
    snapshot.qualifying = pd.concat(
        [pd.read_csv(f"data/cache/qualifying_{year}.csv") for year in range(2023, 2027)],
        ignore_index=True,
    )
    snapshot.sprint = pd.concat(
        [pd.read_csv(f"data/cache/sprint_{year}.csv") for year in range(2023, 2027)],
        ignore_index=True,
    )

    current_only = app_core.derive_model_data(
        snapshot,
        today="2026-08-06",
        effective_time="2026-08-06T12:00:00Z",
        history_mode=app_core.HISTORY_MODE_CURRENT_SEASON_ONLY,
    )
    all_supported = app_core.derive_model_data(
        snapshot,
        today="2026-08-06",
        effective_time="2026-08-06T12:00:00Z",
        history_mode=app_core.HISTORY_MODE_ALL_SUPPORTED,
    )

    expected_keys = [(2026, round_no) for round_no in range(1, 12)]
    assert current_only.diagnostics["selected_race_keys"] == expected_keys
    assert all_supported.diagnostics["selected_race_keys"] == expected_keys
    assert current_only.diagnostics["used_seasons"] == [2026]
    assert all_supported.diagnostics["used_seasons"] == [2023, 2024, 2025, 2026]
    assert set(current_only.driver_price_efficiency["selected_race_count"]) == {11}
    assert set(all_supported.driver_price_efficiency["selected_race_count"]) == {11}
    assert current_only.diagnostics["next_race_round"] == 12
    assert all_supported.diagnostics["next_race_round"] == 12
