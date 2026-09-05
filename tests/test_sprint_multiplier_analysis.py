from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.analyse_sprint_multiplier import (
    assign_pre_event_tiers,
    bootstrap_intervals,
    build_asset_examples,
    build_event_summary,
    leave_one_sprint_out,
    load_recorded_data,
    load_schedule_metadata,
    run_analysis,
    summarize_period,
    verify_reconstruction_identity,
)


def _recorded_data(include_old_season: bool = False) -> pd.DataFrame:
    rows = []
    for season, round_no, event_name, event_date in (
        (2023, 1, "Opening Grand Prix", "2023-03-01"),
        (2023, 2, "Sprint Grand Prix", "2023-03-08"),
        (2023, 3, "Closing Grand Prix", "2023-03-15"),
    ):
        for entity_type, assets in (
            ("driver", [("d1", "Driver One", -2.0), ("d2", "Driver Two", -5.0)]),
            ("constructor", [("c1", "Constructor One", 10.0), ("c2", "Constructor Two", 2.0)]),
        ):
            for asset_id, name, base in assets:
                points = base + round_no * (2 if entity_type == "driver" else 3)
                rows.append(
                    {
                        "season": season,
                        "round": round_no,
                        "event_name": event_name,
                        "event_date": event_date,
                        "entity_type": entity_type,
                        "canonical_entity_id": asset_id,
                        "name": name,
                        "fantasy_points_total": points,
                        "qualifying_points": 99.0 if round_no == 1 else pd.NA,
                        "sprint_points": 99.0 if round_no == 1 else pd.NA,
                        "is_recorded_total": True,
                        "is_reconstructed": False,
                        "fantasy_score_origin": "third_party_recorded",
                    }
                )
    if include_old_season:
        old = rows[0].copy()
        old.update(season=2022, round=1, event_name="Old Grand Prix", event_date="2022-01-01")
        rows.append(old)
    return pd.DataFrame(rows)


def _write_inputs(tmp_path: Path, include_old_season: bool = False) -> tuple[Path, Path]:
    dataset = tmp_path / "recorded.csv"
    _recorded_data(include_old_season).to_csv(dataset, index=False)
    schedule_dir = tmp_path / "schedule"
    schedule_dir.mkdir()
    base = pd.DataFrame(
        [
            {
                "season": 2023,
                "round": 1,
                "raceName": "Opening Grand Prix",
                "date": "2023-03-01",
                "sprint_date": pd.NA,
                "sprint_time": pd.NA,
                "sprint_qualifying_date": pd.NA,
                "sprint_qualifying_time": pd.NA,
            },
            {
                "season": 2023,
                "round": 2,
                "raceName": "Sprint Grand Prix",
                "date": "2023-03-08",
                "sprint_date": "2023-03-07",
                "sprint_time": "12:00:00Z",
                "sprint_qualifying_date": pd.NA,
                "sprint_qualifying_time": pd.NA,
            },
            {
                "season": 2023,
                "round": 3,
                "raceName": "Closing Grand Prix",
                "date": "2023-03-15",
                "sprint_date": pd.NA,
                "sprint_time": pd.NA,
                "sprint_qualifying_date": pd.NA,
                "sprint_qualifying_time": pd.NA,
            },
        ]
    )
    for season in (2023, 2024, 2025, 2026):
        frame = base.copy()
        frame["season"] = season
        frame.to_csv(schedule_dir / f"schedule_{season}.csv", index=False)
    return dataset, schedule_dir


def _annotated(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, schedule_dir = _write_inputs(tmp_path)
    data = load_recorded_data(dataset)
    schedule = load_schedule_metadata(schedule_dir)
    return build_event_summary(data, schedule)


def test_schedule_metadata_defines_normal_and_sprint_not_component_presence(tmp_path):
    events, _annotated_rows = _annotated(tmp_path)
    assert events.set_index("round")["weekend_format"].to_dict() == {
        1: "normal",
        2: "sprint",
        3: "normal",
    }


def test_only_recorded_totals_are_accepted(tmp_path):
    dataset, _schedule_dir = _write_inputs(tmp_path)
    data = pd.read_csv(dataset)
    data.loc[0, "is_reconstructed"] = True
    data.to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="non-recorded or reconstructed"):
        load_recorded_data(dataset)


def test_driver_and_constructor_estimators_remain_separate(tmp_path):
    _events, annotated = _annotated(tmp_path)
    summary = summarize_period(annotated, [2023], "2023").set_index("entity_type")
    assert set(summary.index) == {"driver", "constructor"}
    assert summary.loc["driver", "normal_mean_points"] != summary.loc[
        "constructor", "normal_mean_points"
    ]


def test_per_asset_normalisation_handles_differing_counts(tmp_path):
    _events, annotated = _annotated(tmp_path)
    extra = annotated[
        annotated["entity_type"].eq("constructor") & annotated["round"].eq(2)
    ].iloc[[0]].copy()
    extra["canonical_entity_id"] = "c3"
    extra["fantasy_points_total"] = 30.0
    expanded = pd.concat([annotated, extra], ignore_index=True)
    summary = summarize_period(expanded, [2023], "2023").set_index("entity_type")
    expected = expanded[
        expanded["entity_type"].eq("constructor")
        & expanded["weekend_format"].eq("sprint")
    ]["fantasy_points_total"].mean()
    assert summary.loc["constructor", "sprint_mean_points"] == pytest.approx(expected)


def test_negative_and_zero_recorded_scores_are_retained(tmp_path):
    events, annotated = _annotated(tmp_path)
    assert (annotated["fantasy_points_total"] < 0).any()
    assert (annotated["fantasy_points_total"] == 0).any()
    opening = events.set_index("round").loc[1]
    expected = annotated[
        annotated["round"].eq(1) & annotated["entity_type"].eq("driver")
    ]["fantasy_points_total"].sum()
    assert opening["driver_total_points"] == pytest.approx(expected)


def test_tiers_use_no_future_information(tmp_path):
    _events, annotated = _annotated(tmp_path)
    before = assign_pre_event_tiers(annotated)
    future = annotated[annotated["round"].eq(3)].copy()
    future["round"] = 4
    future["event_date"] = "2023-03-22"
    future["fantasy_points_total"] = 10_000.0
    after = assign_pre_event_tiers(pd.concat([annotated, future], ignore_index=True))
    columns = ["canonical_entity_id", "strength_tier", "pre_event_normal_form"]
    pd.testing.assert_frame_equal(
        before[before["round"].eq(3)][columns].reset_index(drop=True),
        after[after["round"].eq(3)][columns].reset_index(drop=True),
    )


def test_bootstrap_is_reproducible_with_fixed_seed(tmp_path):
    _events, annotated = _annotated(tmp_path)
    first = bootstrap_intervals(annotated, samples=250, seed=42)
    second = bootstrap_intervals(annotated, samples=250, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_leave_one_sprint_excludes_exact_event(tmp_path):
    _events, annotated = _annotated(tmp_path)
    sensitivity = leave_one_sprint_out(annotated)
    assert sensitivity[["excluded_season", "excluded_round"]].to_dict("records") == [
        {"excluded_season": 2023, "excluded_round": 2}
    ]


def test_baseline_reconstruction_identity(tmp_path):
    _events, annotated = _annotated(tmp_path)
    multipliers = {"driver": 1.2, "constructor": 1.4}
    assert verify_reconstruction_identity(annotated, multipliers) < 1e-9
    examples = build_asset_examples(annotated, multipliers)
    assert examples["reconstruction_error"].abs().max() < 1e-9


def test_2021_and_2022_are_absent(tmp_path):
    dataset, _schedule_dir = _write_inputs(tmp_path, include_old_season=True)
    loaded = load_recorded_data(dataset)
    assert loaded["season"].min() == 2023
    assert not loaded["season"].isin({2021, 2022}).any()


def test_running_analysis_twice_is_deterministic(tmp_path):
    dataset, schedule_dir = _write_inputs(tmp_path)
    output = tmp_path / "report"
    run_analysis(dataset, schedule_dir, output, seed=7, bootstrap_samples=100)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    run_analysis(dataset, schedule_dir, output, seed=7, bootstrap_samples=100)
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    assert first == second
