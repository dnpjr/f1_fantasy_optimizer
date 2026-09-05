import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyse_2026_sprint_bonus import (
    MODEL_SPECS,
    _fit_model,
    build_asset_predictions,
    build_asset_summary,
    leave_one_asset_out,
    leave_one_sprint_out,
    load_2026_recorded_data,
    load_current_prices,
    predict_model,
    prepare_2026_events,
    run_analysis,
    sprint_only_target,
)
from scripts.analyse_sprint_multiplier import load_schedule_metadata


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "data/research/sprint_round_11/canonical.csv"
)
SCHEDULE = ROOT / "data/cache/schedule_2026.csv"
MARKET = ROOT / "data/research/sprint_round_11/market.json"


@pytest.fixture(scope="module")
def analysis_data():
    data = load_2026_recorded_data(CANONICAL)
    schedule = load_schedule_metadata(SCHEDULE.parent, seasons=(2026,))
    prices, metadata = load_current_prices(MARKET)
    events, annotated = prepare_2026_events(data, schedule)
    summary = build_asset_summary(annotated, prices)
    return data, events, annotated, prices, metadata, summary


def test_loader_uses_only_2026_data():
    data = load_2026_recorded_data(CANONICAL)
    assert set(data["season"]) == {2026}
    assert data["round"].max() == 11


def test_drivers_and_constructors_remain_separate(analysis_data):
    _data, _events, _annotated, prices, _metadata, summary = analysis_data
    assert summary.groupby("entity_type").size().to_dict() == {"constructor": 11, "driver": 22}
    assert len(summary) == len(prices)
    assert not summary.duplicated(["entity_type", "canonical_entity_id"]).any()


def test_target_contains_only_sprint_specific_components():
    frame = pd.DataFrame(
        {
            "sprint_points": [3.0, 0.0, np.nan, -4.0],
            "sprint_qualifying_points": [2.0, 0.0, np.nan, 1.0],
            "qualifying_points": [100.0, 100.0, 100.0, 100.0],
            "race_points": [200.0, 200.0, 200.0, 200.0],
            "fantasy_points_total": [999.0, 999.0, 999.0, 999.0],
        }
    )
    target = sprint_only_target(frame)
    assert target.iloc[0] == 5.0
    assert target.iloc[1] == 0.0
    assert pd.isna(target.iloc[2])
    assert target.iloc[3] == -3.0


def test_ordinary_race_and_qualifying_do_not_change_target():
    frame = pd.DataFrame(
        {"sprint_points": [4.0], "sprint_qualifying_points": [-1.0], "race_points": [8.0], "qualifying_points": [2.0]}
    )
    before = sprint_only_target(frame).iloc[0]
    frame.loc[0, ["race_points", "qualifying_points"]] = [8000.0, -9000.0]
    assert sprint_only_target(frame).iloc[0] == before == 3.0


def test_full_season_normal_form_uses_only_normal_weekends(analysis_data):
    _data, _events, annotated, _prices, _metadata, summary = analysis_data
    asset = summary.iloc[0]
    source = annotated[
        annotated["entity_type"].eq(asset["entity_type"])
        & annotated["canonical_entity_id"].eq(asset["canonical_entity_id"])
        & annotated["weekend_format"].eq("normal")
    ]
    assert asset["normal_event_count"] == len(source)
    assert asset["normal_weekend_mean"] == pytest.approx(source["fantasy_points_total"].mean())
    assert asset["form_scope"] == "all_completed_2026_normal_weekends_current_state_descriptive"


def test_current_price_is_accepted_verified_official_price(analysis_data):
    _data, _events, _annotated, prices, metadata, summary = analysis_data
    raw = json.loads(MARKET.read_text(encoding="utf-8"))
    russell = next(row for row in raw["players"] if row["tla"] == "RUS")
    analysed = summary[summary["abbreviation"].eq("RUS")].iloc[0]
    assert analysed["current_price"] == russell["price"]
    assert analysed["current_price_source"] == "verified_official_market_cache"
    assert metadata["feed_round"] == raw["feed_round"] == 12
    assert len(prices) == 33


def test_negative_and_explicit_zero_sprint_points_are_retained():
    target = sprint_only_target(
        pd.DataFrame(
            {
                "sprint_points": [-10.0, 0.0, np.nan],
                "sprint_qualifying_points": [0.0, 0.0, np.nan],
            }
        )
    )
    assert target.iloc[:2].tolist() == [-10.0, 0.0]
    assert pd.isna(target.iloc[2])


def test_constructor_target_uses_official_recorded_constructor_components(analysis_data):
    data, _events, annotated, _prices, _metadata, summary = analysis_data
    mercedes = summary[
        summary["entity_type"].eq("constructor")
        & summary["canonical_entity_id"].eq("mercedes")
    ].iloc[0]
    official_rows = annotated[
        annotated["entity_type"].eq("constructor")
        & annotated["canonical_entity_id"].eq("mercedes")
        & annotated["weekend_format"].eq("sprint")
    ]
    assert mercedes["mean_extra_sprint_points"] == pytest.approx(
        official_rows["extra_sprint_points"].mean()
    )
    assert set(data[data["entity_type"].eq("constructor")]["source_name"]) == {
        "Formula 1 Fantasy playerstats"
    }


@pytest.mark.parametrize(
    "model",
    ["constrained_proportional", "constrained_hybrid", "constrained_form_price"],
)
def test_constrained_models_produce_nonnegative_observed_range_predictions(analysis_data, model):
    summary = analysis_data[-1]
    for _entity_type, group in summary.groupby("entity_type"):
        fit = _fit_model(group, model)
        assert np.all(np.asarray(fit["coefficients"]) >= -1e-10)
        assert np.all(predict_model(fit, group) >= 0)


def test_leave_one_asset_out_excludes_and_predicts_the_intended_asset(analysis_data):
    drivers = analysis_data[-1][analysis_data[-1]["entity_type"].eq("driver")].copy()
    result = leave_one_asset_out(drivers, ["constrained_hybrid"], ridge_penalty=1.0)
    held = result.iloc[0]
    assert len(result) == len(drivers)
    assert result["held_out_entity_id"].is_unique
    train = drivers[drivers["canonical_entity_id"].ne(held["held_out_entity_id"])]
    test = drivers[drivers["canonical_entity_id"].eq(held["held_out_entity_id"])]
    expected = predict_model(_fit_model(train, "constrained_hybrid"), test)[0]
    assert held["predicted_bonus"] == pytest.approx(expected)


def test_leave_one_sprint_out_removes_event_before_aggregation(analysis_data):
    _data, _events, annotated, _prices, _metadata, summary = analysis_data
    selected = {}
    for entity_type, group in summary.groupby("entity_type"):
        selected[entity_type] = {
            "model_name": "constrained_hybrid",
            "fit": _fit_model(group, "constrained_hybrid"),
        }
    result = leave_one_sprint_out(annotated, summary, selected)
    assert set(result["excluded_round"]) == {2, 4, 5, 9}
    assert (result.groupby("excluded_round")["entity_type"].nunique() == 2).all()
    row = result[(result["entity_type"].eq("driver")) & result["excluded_round"].eq(2)].iloc[0]
    retained = annotated[
        ~(annotated["weekend_format"].eq("sprint") & annotated["round"].eq(2))
    ]
    target = retained.groupby(["entity_type", "canonical_entity_id"])["extra_sprint_points"].mean().rename(
        "mean_extra_sprint_points"
    ).reset_index()
    features = summary[summary["entity_type"].eq("driver")].drop(
        columns="mean_extra_sprint_points"
    ).merge(target, on=["entity_type", "canonical_entity_id"]).dropna(subset=["mean_extra_sprint_points"])
    expected = _fit_model(features, "constrained_hybrid")
    assert row["alpha"] == pytest.approx(expected["coefficients"][0])
    assert row["gamma"] == pytest.approx(expected["coefficients"][1])


def test_shrinkage_moves_observed_bonus_toward_pooled_prediction():
    summary = pd.DataFrame(
        [
            {
                "entity": "Asset",
                "entity_type": "driver",
                "canonical_entity_id": "asset",
                "abbreviation": "AST",
                "current_price": 10.0,
                "price_percentile_within_entity_type": 0.5,
                "normal_weekend_mean": 10.0,
                "recent_normal_form": 9.0,
                "mean_extra_sprint_points": 10.0,
                "sprint_event_count": 4,
                "sprint_component_coverage": 1.0,
                "current_price_source": "verified_official_market_cache",
                "form_scope": "all_completed_2026_normal_weekends_current_state_descriptive",
            }
        ]
    )
    fit = {"model": "constant", "columns": MODEL_SPECS["constant"], "coefficients": np.array([2.0])}
    prediction = build_asset_predictions(
        summary,
        {"driver": {"model_name": "constant", "fit": fit}},
        {"driver": 4.0},
        {("driver", "asset"): (1.0, 3.0)},
    ).iloc[0]
    assert prediction["pooled_predicted_bonus"] == 2.0
    assert prediction["shrunk_asset_bonus"] == 6.0
    assert 2.0 < prediction["shrunk_asset_bonus"] < 10.0
    assert prediction["predicted_sprint_ev"] == 16.0


def test_insufficient_official_component_coverage_fails_clearly(analysis_data):
    data, _events, _annotated, _prices, _metadata, _summary = analysis_data
    damaged = data.copy()
    damaged.loc[:, ["sprint_points", "sprint_qualifying_points"]] = np.nan
    schedule = load_schedule_metadata(SCHEDULE.parent, seasons=(2026,))
    with pytest.raises(ValueError, match="coverage is insufficient"):
        prepare_2026_events(damaged, schedule)


def test_outputs_are_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run_analysis(CANONICAL, SCHEDULE, MARKET, first)
    run_analysis(CANONICAL, SCHEDULE, MARKET, second)
    assert sorted(path.name for path in first.iterdir()) == sorted(path.name for path in second.iterdir())
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
