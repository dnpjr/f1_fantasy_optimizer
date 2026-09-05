#!/usr/bin/env python3
"""Offline comparison of production EV and the frozen 2026 Sprint shadow.

This research script reads only validated local caches and generated canonical
data.  It never refreshes data, updates calibration, mutates production tables,
or runs during application startup.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1fantasy.app_core import (
    DEFAULT_PRICE_CHANGE_BOUNDS,
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
    DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
    HISTORY_MODE_ALL_SUPPORTED,
    HISTORY_MODE_CURRENT_SEASON_ONLY,
    LiveDataSnapshot,
    ModelData,
    OBJECTIVE_COMBINED,
    OBJECTIVE_POINTS_ONLY,
    apply_objective_mode,
    apply_probabilistic_price_change_model,
    current_team_budget_from_selection,
    derive_model_data,
    run_optimizer,
)
from f1fantasy.fantasy_api import load_verified_market_cache
from f1fantasy.historical_scores import (
    DEFAULT_CANONICAL_DATASET_PATH,
    canonical_playerstats_observations,
    load_canonical_scores,
)
from f1fantasy.optimize import TeamSolution


CACHE_DIR = PROJECT_ROOT / "data/cache"
RESEARCH_INPUTS = PROJECT_ROOT / "data/research/sprint_round_11"
CURRENT_TEAM_PATH = RESEARCH_INPUTS / "example_team.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/2026_sprint_shadow_comparison"
ANALYSIS_EFFECTIVE_TIME = "2026-08-10T12:00:00Z"
CURRENT_SEASON = 2026
UPCOMING_ROUND = 12
COMPLETED_ROUNDS = tuple(range(1, UPCOMING_ROUND))
DECAYS = (1.0, 0.85, 0.70)
HISTORY_MODES = (HISTORY_MODE_CURRENT_SEASON_ONLY, HISTORY_MODE_ALL_SUPPORTED)
DEFAULT_PRICE_GROWTH_WEIGHT = 50.0


@dataclass(frozen=True)
class Scenario:
    decay: float
    history_mode: str
    model: ModelData
    comparison: pd.DataFrame


def _read_cache_csv(stem: str, season: int) -> pd.DataFrame:
    path = CACHE_DIR / f"{stem}_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required offline cache is missing: {path}")
    return pd.read_csv(path)


def canonical_recent_points(recorded: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    """Build the app's two-race price-growth input from canonical official totals."""
    observations = canonical_playerstats_observations(recorded, CURRENT_SEASON, entity_type)
    rows: list[dict[str, Any]] = []
    for player_id, group in observations.groupby("PlayerId", sort=True):
        recent = group.sort_values(["season", "round"], kind="stable").tail(2)
        points = pd.to_numeric(recent["fantasy_points"], errors="coerce").dropna().tolist()
        race_keys = [
            (int(row.season), int(row.round))
            for row in recent.itertuples(index=False)
        ]
        rows.append(
            {
                "id": int(player_id),
                "recent_points_2ago": float(points[-2]) if len(points) >= 2 else pd.NA,
                "recent_points_1ago": float(points[-1]) if points else pd.NA,
                "recent_points_available": len(points),
                "recent_points_source": "canonical_recorded_offline",
                "recent_points_races": race_keys,
                "recent_points_fallback_used": len(points) < 2,
                "recent_points_missing": len(points) < 2,
            }
        )
    return pd.DataFrame(rows)


def load_offline_snapshot() -> tuple[LiveDataSnapshot, dict[str, Any]]:
    """Build the accepted snapshot entirely from validated local files."""
    recorded = load_canonical_scores(RESEARCH_INPUTS / "canonical.csv")
    market = load_verified_market_cache(path=RESEARCH_INPUTS / "market.json")
    if int(market["feed_round"]) != UPCOMING_ROUND:
        raise ValueError(
            f"Expected accepted feed {UPCOMING_ROUND}, found {market['feed_round']}."
        )
    current_rows = recorded[pd.to_numeric(recorded["season"], errors="coerce").eq(CURRENT_SEASON)]
    recorded_rounds = sorted(pd.to_numeric(current_rows["round"], errors="coerce").dropna().astype(int).unique())
    if recorded_rounds != list(COMPLETED_ROUNDS):
        raise ValueError(
            "Canonical 2026 history must contain exactly completed rounds 1-11; "
            f"found {recorded_rounds}."
        )
    seasons = tuple(range(2023, CURRENT_SEASON + 1))
    results = pd.concat([_read_cache_csv("results", season) for season in seasons], ignore_index=True)
    qualifying = pd.concat(
        [_read_cache_csv("qualifying", season) for season in seasons], ignore_index=True
    )
    sprint = pd.concat([_read_cache_csv("sprint", season) for season in seasons], ignore_index=True)
    schedule = _read_cache_csv("schedule", CURRENT_SEASON)
    driver_recent = canonical_recent_points(recorded, "driver")
    constructor_recent = canonical_recent_points(recorded, "constructor")
    snapshot = LiveDataSnapshot(
        current_season=CURRENT_SEASON,
        loaded_start_year=min(seasons),
        requested_seasons=seasons,
        loaded_seasons=seasons,
        season_load_failures={},
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        schedule=schedule,
        players=market["players"].copy(deep=True),
        teams=market["teams"].copy(deep=True),
        driver_recent_points=driver_recent,
        constructor_recent_points=constructor_recent,
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={
            "feed_round": int(market["feed_round"]),
            "live_data_status": "cached",
            "market_resolution_method": "verified_cache_offline_analysis",
            "live_data_verified_at_utc": market.get("verified_at_utc"),
            "raw_live_load_finished_utc": "offline-comparison-fixed-inputs",
            "playerstats_prefetch_enabled": False,
            "completed_current_event_keys": [
                (CURRENT_SEASON, round_no) for round_no in COMPLETED_ROUNDS
            ],
        },
        historical_fantasy_scores=recorded,
    )
    metadata = {
        "feed_round": int(market["feed_round"]),
        "verified_at_utc": market.get("verified_at_utc"),
        "driver_count": len(market["players"]),
        "constructor_count": len(market["teams"]),
        "recorded_rounds": recorded_rounds,
        "price_growth_recent_points_source": "canonical official totals, latest two completed observations per asset",
    }
    return snapshot, metadata


def deterministic_budget(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    current_team_path: Path = CURRENT_TEAM_PATH,
) -> tuple[float, dict[str, Any]]:
    payload = json.loads(current_team_path.read_text(encoding="utf-8"))
    driver_ids = {str(value) for value in payload.get("drivers", [])}
    constructor_ids = {str(value) for value in payload.get("constructors", [])}
    selected_drivers = drivers[drivers["id"].astype(str).isin(driver_ids)].copy()
    selected_constructors = constructors[constructors["id"].astype(str).isin(constructor_ids)].copy()
    if len(selected_drivers) != 5 or len(selected_constructors) != 2:
        raise ValueError("Persisted current team cannot be resolved against the accepted market.")
    bank = float(payload.get("bank", 0.0) or 0.0)
    budget = current_team_budget_from_selection(selected_drivers, selected_constructors, bank)
    return budget, {
        "source": str(current_team_path.relative_to(PROJECT_ROOT)),
        "team_cost": budget - bank,
        "bank": bank,
        "driver_ids": sorted(driver_ids),
        "constructor_ids": sorted(constructor_ids),
    }


def derive_scenario(snapshot: LiveDataSnapshot, decay: float, history_mode: str) -> Scenario:
    model = derive_model_data(
        snapshot,
        today="2026-08-10",
        effective_time=ANALYSIS_EFFECTIVE_TIME,
        historical_seasons_back=3,
        horizon_races=5,
        current_season_weight=1.0,
        past_season_weight=0.7,
        recency_decay=float(decay),
        selected_race_preset="All",
        history_mode=history_mode,
    )
    diagnostics = model.diagnostics.get("approved_sprint_ev_shadow", {})
    if diagnostics.get("upcoming_event") != {"season": CURRENT_SEASON, "round": UPCOMING_ROUND}:
        raise ValueError("Shadow comparison did not target 2026 round 12.")
    if diagnostics.get("upcoming_weekend_format") != "sprint":
        raise ValueError("Upcoming Dutch GP is not classified as Sprint in canonical schedule metadata.")
    if diagnostics.get("selected_2026_race_keys") != [
        (CURRENT_SEASON, round_no) for round_no in COMPLETED_ROUNDS
    ]:
        raise ValueError("Shadow history is not exactly completed 2026 rounds 1-11.")
    comparison = build_asset_comparison(model.drivers, model.constructors)
    return Scenario(float(decay), history_mode, model, comparison)


def _asset_rows(frame: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    required = {
        "id",
        "name",
        "price",
        "next_race_expected_points",
        "shadow_normal_ev",
        "shadow_sprint_bonus",
        "shadow_sprint_ev",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{entity_type} comparison input is missing columns: {missing}")
    production_column = (
        "baseline_expected_points"
        if "baseline_expected_points" in frame.columns
        else "next_race_expected_points"
    )
    rows = pd.DataFrame(
        {
            "asset_id": frame["id"].astype(str),
            "entity": frame["name"].astype(str),
            "entity_type": entity_type,
            "current_price": pd.to_numeric(frame["price"], errors="raise").astype(float),
            "production_ev": pd.to_numeric(
                frame[production_column], errors="raise"
            ).astype(float),
            "shadow_normal_ev": pd.to_numeric(frame["shadow_normal_ev"], errors="coerce"),
            "shadow_sprint_bonus": pd.to_numeric(frame["shadow_sprint_bonus"], errors="coerce"),
            "shadow_sprint_ev": pd.to_numeric(frame["shadow_sprint_ev"], errors="coerce"),
            "shadow_personal_mean_bonus": pd.to_numeric(
                frame.get("shadow_personal_mean_bonus"), errors="coerce"
            ),
            "shadow_personal_weight": pd.to_numeric(
                frame.get("shadow_personal_weight"), errors="coerce"
            ),
            "shadow_group_bonus": pd.to_numeric(frame.get("shadow_group_bonus"), errors="coerce"),
            "shadow_strength": pd.to_numeric(frame.get("shadow_strength"), errors="coerce"),
            "shadow_valid_race_count": pd.to_numeric(
                frame.get("shadow_valid_race_count"), errors="coerce"
            ),
        }
    )
    return rows


def build_asset_comparison(drivers: pd.DataFrame, constructors: pd.DataFrame) -> pd.DataFrame:
    """Build rankings without mutating either production table."""
    driver_before = drivers.copy(deep=True)
    constructor_before = constructors.copy(deep=True)
    comparison = pd.concat(
        [_asset_rows(drivers, "driver"), _asset_rows(constructors, "constructor")],
        ignore_index=True,
    )
    if comparison[["shadow_normal_ev", "shadow_sprint_bonus", "shadow_sprint_ev"]].isna().any().any():
        raise ValueError("Every active asset must have a complete Sprint shadow value.")
    comparison["shadow_minus_production"] = (
        comparison["shadow_sprint_ev"] - comparison["production_ev"]
    )
    comparison["baseline_difference"] = (
        comparison["shadow_normal_ev"] - comparison["production_ev"]
    )
    comparison["production_points_per_million"] = (
        comparison["production_ev"] / comparison["current_price"]
    )
    comparison["shadow_points_per_million"] = (
        comparison["shadow_sprint_ev"] / comparison["current_price"]
    )
    comparison["production_rank"] = comparison.groupby("entity_type")["production_ev"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["shadow_rank"] = comparison.groupby("entity_type")["shadow_sprint_ev"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["rank_change"] = comparison["production_rank"] - comparison["shadow_rank"]
    comparison["absolute_rank_change"] = comparison["rank_change"].abs()
    comparison.sort_values(
        ["entity_type", "production_rank", "entity"], kind="stable", inplace=True
    )
    comparison.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(drivers, driver_before)
    pd.testing.assert_frame_equal(constructors, constructor_before)
    return comparison


def ranking_views(comparison: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    rows = comparison[comparison["entity_type"].eq(entity_type)].copy(deep=True)
    specs = (
        ("production_ev", ["production_ev", "entity"], [False, True]),
        ("sprint_aware_ev", ["shadow_sprint_ev", "entity"], [False, True]),
        ("positive_ev_difference", ["shadow_minus_production", "entity"], [False, True]),
        ("absolute_rank_change", ["absolute_rank_change", "shadow_minus_production", "entity"], [False, False, True]),
    )
    views = []
    for view_name, columns, ascending in specs:
        view = rows.sort_values(columns, ascending=ascending, kind="stable").copy()
        view.insert(0, "view_position", range(1, len(view) + 1))
        view.insert(0, "view", view_name)
        views.append(view)
    return pd.concat(views, ignore_index=True)


def validate_market_prices(model: ModelData, snapshot: LiveDataSnapshot) -> None:
    pairs = (
        (model.drivers, snapshot.players, "id", "playerId"),
        (model.constructors, snapshot.teams, "id", "teamId"),
    )
    for derived, accepted, derived_id, accepted_id in pairs:
        left = derived[[derived_id, "price"]].copy()
        right = accepted[[accepted_id, "price"]].copy()
        left["_id"] = left[derived_id].astype(str)
        right["_id"] = right[accepted_id].astype(str)
        merged = left.merge(right, on="_id", suffixes=("_derived", "_accepted"), validate="one_to_one")
        if len(merged) != len(accepted):
            raise ValueError("Derived roster does not match the accepted official market.")
        if not (merged["price_derived"] - merged["price_accepted"]).abs().le(1e-12).all():
            raise ValueError("Derived prices disagree with the accepted official market.")


def _project_price_growth(frame: pd.DataFrame) -> pd.DataFrame:
    return apply_probabilistic_price_change_model(
        frame,
        DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        expensive_rules=DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
        expensive_price_min=DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
        bounds=DEFAULT_PRICE_CHANGE_BOUNDS,
        predicted_points_col="next_race_expected_points",
    )


def optimizer_input_copies(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    objective_mode: str,
    price_growth_weight: float = DEFAULT_PRICE_GROWTH_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return equal-price inputs where only the shadow points field differs."""
    production_drivers, production_constructors, shadow_drivers, shadow_constructors = (
        optimizer_point_copies(drivers, constructors)
    )
    return (
        apply_objective_mode(production_drivers, objective_mode, price_growth_weight),
        apply_objective_mode(production_constructors, objective_mode, price_growth_weight),
        apply_objective_mode(shadow_drivers, objective_mode, price_growth_weight),
        apply_objective_mode(shadow_constructors, objective_mode, price_growth_weight),
    )


def optimizer_point_copies(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build isolated optimiser inputs before applying the shared objective."""
    production_driver_source = drivers.copy(deep=True)
    production_constructor_source = constructors.copy(deep=True)
    for frame in (production_driver_source, production_constructor_source):
        if "baseline_expected_points" in frame.columns:
            frame["next_race_expected_points"] = pd.to_numeric(
                frame["baseline_expected_points"], errors="raise"
            )
    production_drivers = _project_price_growth(production_driver_source)
    production_constructors = _project_price_growth(production_constructor_source)
    production_drivers["exp_score"] = pd.to_numeric(
        production_drivers["next_race_expected_points"], errors="raise"
    )
    production_constructors["exp_score"] = pd.to_numeric(
        production_constructors["next_race_expected_points"], errors="raise"
    )
    shadow_drivers = production_drivers.copy(deep=True)
    shadow_constructors = production_constructors.copy(deep=True)
    shadow_drivers["exp_score"] = pd.to_numeric(shadow_drivers["shadow_sprint_ev"], errors="raise")
    shadow_constructors["exp_score"] = pd.to_numeric(
        shadow_constructors["shadow_sprint_ev"], errors="raise"
    )
    for production, shadow in (
        (production_drivers, shadow_drivers),
        (production_constructors, shadow_constructors),
    ):
        if not production["id"].astype(str).equals(shadow["id"].astype(str)):
            raise AssertionError("Optimiser asset identities changed between models.")
        for column in ("price", "expected_price_gain", "projected_price"):
            pd.testing.assert_series_equal(
                production[column], shadow[column], check_names=False, check_dtype=False
            )
        differing = [
            column
            for column in production.columns.intersection(shadow.columns)
            if not production[column].equals(shadow[column])
        ]
        if differing != ["exp_score"]:
            raise AssertionError(f"Shadow optimiser copy changed unexpected fields: {differing}")
    return production_drivers, production_constructors, shadow_drivers, shadow_constructors


def _solve(inputs: tuple[pd.DataFrame, pd.DataFrame], budget: float) -> TeamSolution:
    solutions = run_optimizer(
        inputs[0],
        inputs[1],
        budget=budget,
        top_k=1,
        drs_multiplier=2.0,
        allow_no_negative=False,
        objective_col=(
            "exp_score"
            if "comparison_objective" not in inputs[0].columns
            else "comparison_objective"
        ),
        boost_col="exp_score",
    )
    if not solutions:
        raise RuntimeError("Comparison optimiser did not find a feasible team.")
    return solutions[0]


def _solution_ids(solution: TeamSolution) -> tuple[frozenset[str], frozenset[str]]:
    return (
        frozenset(solution.drivers["id"].astype(str)),
        frozenset(solution.constructors["id"].astype(str)),
    )


def score_solution(
    solution: TeamSolution,
    comparison: pd.DataFrame,
    points_column: str,
) -> float:
    """Cross-score one fixed team, preserving its selected 2x driver."""
    values = comparison.set_index(["entity_type", "asset_id"])[points_column]
    driver_ids, constructor_ids = _solution_ids(solution)
    score = sum(float(values.loc[("driver", asset_id)]) for asset_id in driver_ids)
    score += sum(float(values.loc[("constructor", asset_id)]) for asset_id in constructor_ids)
    boosted = solution.drivers[solution.drivers["name"].astype(str).eq(str(solution.boosted_driver))]
    if len(boosted) != 1:
        raise ValueError("Selected team does not identify exactly one 2x driver.")
    score += float(values.loc[("driver", str(boosted.iloc[0]["id"]))])
    return float(score)


def _team_names(solution: TeamSolution) -> tuple[str, str]:
    drivers = ", ".join(sorted(solution.drivers["name"].astype(str)))
    constructors = ", ".join(sorted(solution.constructors["name"].astype(str)))
    return drivers, constructors


def _swap_summary(production: TeamSolution, shadow: TeamSolution) -> dict[str, Any]:
    prod_drivers, prod_constructors = _solution_ids(production)
    shadow_drivers, shadow_constructors = _solution_ids(shadow)
    return {
        "driver_ids_out": ", ".join(sorted(prod_drivers - shadow_drivers)),
        "driver_ids_in": ", ".join(sorted(shadow_drivers - prod_drivers)),
        "constructor_ids_out": ", ".join(sorted(prod_constructors - shadow_constructors)),
        "constructor_ids_in": ", ".join(sorted(shadow_constructors - prod_constructors)),
        "overlap_count": len(prod_drivers & shadow_drivers) + len(prod_constructors & shadow_constructors),
        "swap_count": len(prod_drivers - shadow_drivers) + len(prod_constructors - shadow_constructors),
    }


def run_optimizer_comparison(
    scenario: Scenario,
    budget: float,
    objective_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    original_drivers = scenario.model.drivers.copy(deep=True)
    original_constructors = scenario.model.constructors.copy(deep=True)
    prod_d, prod_c, shadow_d, shadow_c = optimizer_input_copies(
        scenario.model.drivers,
        scenario.model.constructors,
        objective_mode,
    )
    objective_column = "exp_score"
    if objective_mode == OBJECTIVE_COMBINED:
        for frame in (prod_d, prod_c, shadow_d, shadow_c):
            frame["comparison_objective"] = frame["combined_objective_score"]
        objective_column = "comparison_objective"
    production_solution = run_optimizer(
        prod_d,
        prod_c,
        budget=budget,
        top_k=1,
        drs_multiplier=2.0,
        objective_col=objective_column,
        boost_col="exp_score",
    )[0]
    shadow_solution = run_optimizer(
        shadow_d,
        shadow_c,
        budget=budget,
        top_k=1,
        drs_multiplier=2.0,
        objective_col=objective_column,
        boost_col="exp_score",
    )[0]
    prod_prod = score_solution(production_solution, scenario.comparison, "production_ev")
    prod_shadow = score_solution(production_solution, scenario.comparison, "shadow_sprint_ev")
    shadow_prod = score_solution(shadow_solution, scenario.comparison, "production_ev")
    shadow_shadow = score_solution(shadow_solution, scenario.comparison, "shadow_sprint_ev")
    swaps = _swap_summary(production_solution, shadow_solution)
    scenario_fields = {
        "decay": scenario.decay,
        "history_mode": scenario.history_mode,
        "objective": objective_mode,
        "budget": budget,
        "driver_slots": 5,
        "constructor_slots": 2,
        "drs_multiplier": 2.0,
        "chip": "none",
        "locked_driver_count": 0,
        "excluded_driver_count": 0,
        "locked_constructor_count": 0,
        "excluded_constructor_count": 0,
        "transfer_constraint": "not_applicable_fresh_team_optimisation",
        **swaps,
        "production_ev_difference_shadow_minus_production_team": shadow_prod - prod_prod,
        "shadow_ev_difference_shadow_minus_production_team": shadow_shadow - prod_shadow,
        "production_penalty_of_shadow_team": prod_prod - shadow_prod,
        "shadow_advantage_of_shadow_team": shadow_shadow - prod_shadow,
    }
    rows = []
    for selected_by, solution, production_score, shadow_score in (
        ("production_ev", production_solution, prod_prod, prod_shadow),
        ("sprint_shadow_ev", shadow_solution, shadow_prod, shadow_shadow),
    ):
        driver_names, constructor_names = _team_names(solution)
        rows.append(
            {
                **scenario_fields,
                "selected_by": selected_by,
                "drivers": driver_names,
                "constructors": constructor_names,
                "boosted_driver_2x": solution.boosted_driver,
                "total_price": solution.total_cost,
                "production_model_score": production_score,
                "sprint_shadow_score": shadow_score,
                "optimizer_objective_value": solution.expected_score,
            }
        )
    cross_rows = []
    for selected_by, production_score, shadow_score in (
        ("production_ev", prod_prod, prod_shadow),
        ("sprint_shadow_ev", shadow_prod, shadow_shadow),
    ):
        cross_rows.extend(
            [
                {
                    **scenario_fields,
                    "selected_by": selected_by,
                    "scored_by": "production_ev",
                    "predicted_score": production_score,
                },
                {
                    **scenario_fields,
                    "selected_by": selected_by,
                    "scored_by": "sprint_shadow_ev",
                    "predicted_score": shadow_score,
                },
            ]
        )
    pd.testing.assert_frame_equal(scenario.model.drivers, original_drivers)
    pd.testing.assert_frame_equal(scenario.model.constructors, original_constructors)
    return pd.DataFrame(rows), pd.DataFrame(cross_rows), scenario_fields


def sanity_cases(comparison: pd.DataFrame) -> pd.DataFrame:
    requested = (
        "Kimi Antonelli", "George Russell", "Lando Norris", "Lewis Hamilton",
        "Charles Leclerc", "Liam Lawson", "Oliver Bearman", "Nico Hulkenberg",
        "Valtteri Bottas", "Lance Stroll", "Mercedes", "Ferrari", "McLaren",
        "Red Bull Racing", "Haas F1 Team", "Williams", "Audi", "Aston Martin", "Cadillac",
    )
    rows = comparison[comparison["entity"].isin(requested)].copy()
    if len(rows) != len(requested):
        missing = sorted(set(requested) - set(rows["entity"]))
        raise ValueError(f"Sanity assets missing from comparison: {missing}")

    def review(row: pd.Series) -> tuple[str, str]:
        name = str(row["entity"])
        bonus = float(row["shadow_sprint_bonus"])
        if name == "Nico Hulkenberg":
            return (
                "sensible" if bonus < 2.0 else "manual_review",
                f"Conservative {bonus:.2f}-point bonus reflects the stored -2.75 personal Sprint mean.",
            )
        if name == "Valtteri Bottas":
            return (
                "sensible" if bonus < 4.0 else "manual_review",
                f"Bonus is {bonus:.2f}, avoiding the former unstable +13-style artefact.",
            )
        if row["entity_type"] == "constructor":
            return (
                "sensible" if 0.0 <= bonus <= 18.0 else "manual_review",
                f"Strength-ranked constructor bonus is {bonus:.2f}; no personal/event effect is used.",
            )
        return (
            "sensible" if -2.0 <= bonus <= 12.0 else "manual_review",
            f"Driver bonus {bonus:.2f} combines form-linked group expectation with frozen personal history.",
        )

    assessed = rows.apply(review, axis=1)
    rows["assessment"] = assessed.map(lambda value: value[0])
    rows["qualitative_note"] = assessed.map(lambda value: value[1])
    order = {name: index for index, name in enumerate(requested)}
    rows["_order"] = rows["entity"].map(order)
    return rows.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _names_for_ids(comparison: pd.DataFrame, entity_type: str, ids: str) -> str:
    if not ids:
        return "none"
    wanted = {value.strip() for value in ids.split(",") if value.strip()}
    names = comparison[
        comparison["entity_type"].eq(entity_type)
        & comparison["asset_id"].isin(wanted)
    ]["entity"].tolist()
    return ", ".join(sorted(names)) or "none"


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str], labels: dict[str, str]) -> str:
    selected = frame[list(columns)].copy()
    for column in selected:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
    headers = [labels.get(column, column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _top_movers(comparison: pd.DataFrame, entity_type: str, positive: bool, count: int = 5) -> pd.DataFrame:
    rows = comparison[comparison["entity_type"].eq(entity_type)].copy()
    rows = rows[rows["rank_change"].gt(0) if positive else rows["rank_change"].lt(0)]
    return rows.sort_values(
        ["rank_change", "shadow_minus_production"],
        ascending=[not positive, not positive],
        kind="stable",
    ).head(count)


def render_report(
    primary: Scenario,
    optimiser_rows: pd.DataFrame,
    cross_rows: pd.DataFrame,
    decay_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
    sanity: pd.DataFrame,
    metadata: dict[str, Any],
    budget_meta: dict[str, Any],
) -> str:
    points = optimiser_rows[
        optimiser_rows["decay"].eq(0.85)
        & optimiser_rows["history_mode"].eq(HISTORY_MODE_ALL_SUPPORTED)
        & optimiser_rows["objective"].eq(OBJECTIVE_POINTS_ONLY)
    ]
    prod_team = points[points["selected_by"].eq("production_ev")].iloc[0]
    shadow_team = points[points["selected_by"].eq("sprint_shadow_ev")].iloc[0]
    primary_cross = cross_rows[
        cross_rows["decay"].eq(0.85)
        & cross_rows["history_mode"].eq(HISTORY_MODE_ALL_SUPPORTED)
        & cross_rows["objective"].eq(OBJECTIVE_POINTS_ONLY)
    ]
    matrix = primary_cross.pivot(index="selected_by", columns="scored_by", values="predicted_score")
    combined = optimiser_rows[
        optimiser_rows["decay"].eq(0.85)
        & optimiser_rows["history_mode"].eq(HISTORY_MODE_ALL_SUPPORTED)
        & optimiser_rows["objective"].eq(OBJECTIVE_COMBINED)
    ]
    combined_prod_team = combined[combined["selected_by"].eq("production_ev")].iloc[0]
    combined_shadow_team = combined[combined["selected_by"].eq("sprint_shadow_ev")].iloc[0]
    combined_cross = cross_rows[
        cross_rows["decay"].eq(0.85)
        & cross_rows["history_mode"].eq(HISTORY_MODE_ALL_SUPPORTED)
        & cross_rows["objective"].eq(OBJECTIVE_COMBINED)
    ].pivot(index="selected_by", columns="scored_by", values="predicted_score")
    unique_shadow_teams = decay_rows["shadow_team_signature"].nunique()
    suspicious = sanity["assessment"].eq("manual_review").any()
    material_team_change = bool(
        int(prod_team["swap_count"]) > 0
        and float(prod_team["shadow_advantage_of_shadow_team"]) >= 1.0
    )
    if unique_shadow_teams > 1:
        recommendation = "C. Asset-level values look reasonable, but optimiser impact is unstable."
    elif suspicious:
        recommendation = "B. Sprint-aware shadow looks promising but needs one more adjustment."
    elif material_team_change:
        recommendation = "A. Sprint-aware shadow looks sensible enough to promote to production on Sprint weekends."
    else:
        recommendation = "B. Sprint-aware shadow looks promising but needs one more adjustment."
    comparison = primary.comparison
    driver_top = comparison[comparison["entity_type"].eq("driver")].nsmallest(10, "shadow_rank")
    constructor_top = comparison[comparison["entity_type"].eq("constructor")].nsmallest(10, "shadow_rank")
    driver_up = _top_movers(comparison, "driver", True)
    driver_down = _top_movers(comparison, "driver", False)
    constructor_up = _top_movers(comparison, "constructor", True)
    constructor_down = _top_movers(comparison, "constructor", False)
    largest_driver_gain = comparison[comparison["entity_type"].eq("driver")].nlargest(1, "shadow_minus_production").iloc[0]
    largest_constructor_gain = comparison[comparison["entity_type"].eq("constructor")].nlargest(1, "shadow_minus_production").iloc[0]
    baseline_examples = comparison.reindex(comparison["baseline_difference"].abs().sort_values(ascending=False).index).head(5)
    sprint_examples = comparison.nlargest(5, "shadow_sprint_bonus")
    columns = (
        "entity", "production_ev", "shadow_normal_ev", "shadow_sprint_bonus",
        "shadow_sprint_ev", "rank_change",
    )
    labels = {
        "entity": "Asset", "production_ev": "Production EV", "shadow_normal_ev": "Normal EV",
        "shadow_sprint_bonus": "Sprint bonus", "shadow_sprint_ev": "Sprint EV",
        "rank_change": "Rank change", "selected_by": "Team selected by",
        "production_model_score": "Production score", "sprint_shadow_score": "Sprint score",
    }
    production_top_driver_names = ", ".join(
        comparison[comparison["entity_type"].eq("driver")].nsmallest(10, "production_rank")["entity"]
    )
    shadow_top_driver_names = ", ".join(driver_top["entity"])
    production_top_constructor_names = ", ".join(
        comparison[comparison["entity_type"].eq("constructor")].nsmallest(10, "production_rank")["entity"]
    )
    shadow_top_constructor_names = ", ".join(constructor_top["entity"])
    return f"""# 2026 Sprint shadow versus production EV

Inputs are frozen local data: verified official feed {metadata['feed_round']} ({metadata['verified_at_utc']}), completed canonical rounds 1–11, and the canonical schedule identifying Dutch GP round 12 as Sprint. No network refresh or calibration was run.

## 1. Executive result

- Sprint-aware EV {'materially changes' if comparison['absolute_rank_change'].max() >= 3 else 'modestly changes'} asset rankings; the largest driver uplift is **{largest_driver_gain['entity']} {largest_driver_gain['shadow_minus_production']:+.2f}**, and the largest constructor uplift is **{largest_constructor_gain['entity']} {largest_constructor_gain['shadow_minus_production']:+.2f}**.
- At the primary p=0.85/all-supported/points-only setting, the optimiser changes **{int(prod_team['swap_count'])}** asset slot(s), with **{int(prod_team['overlap_count'])}/7** assets overlapping.
- Under Sprint EV, the shadow-selected team gains **{prod_team['shadow_advantage_of_shadow_team']:.2f}** points over the production-selected team. Under production EV it gives up **{prod_team['production_penalty_of_shadow_team']:.2f}** points.
- The shadow-selected team is {'stable' if unique_shadow_teams == 1 else 'not fully stable'} across p=1.00, 0.85 and 0.70 ({unique_shadow_teams} distinct shadow team composition(s)).
- All {metadata['driver_count']} driver and {metadata['constructor_count']} constructor prices exactly match the accepted verified official market.

## 2. Drivers

Top 10 by Sprint-aware EV:

{_markdown_table(driver_top, columns, labels)}

Top 10 production order: {production_top_driver_names}.

Top 10 Sprint order: {shadow_top_driver_names}.

Largest movers up:

{_markdown_table(driver_up, columns, labels) if not driver_up.empty else 'None.'}

Largest movers down:

{_markdown_table(driver_down, columns, labels) if not driver_down.empty else 'None.'}

## 3. Constructors

Top 10 by Sprint-aware EV:

{_markdown_table(constructor_top, columns, labels)}

Top 10 production order: {production_top_constructor_names}.

Top 10 Sprint order: {shadow_top_constructor_names}.

Largest movers up:

{_markdown_table(constructor_up, columns, labels) if not constructor_up.empty else 'None.'}

Largest movers down:

{_markdown_table(constructor_down, columns, labels) if not constructor_down.empty else 'None.'}

## 4. Optimiser comparison

Budget: **{float(prod_team['budget']):.2f}M**, derived from persisted current-team value {budget_meta['team_cost']:.2f}M plus {budget_meta['bank']:.2f}M bank. Standard 5-driver/2-constructor roster, one 2× driver, no chip, no locks or exclusions.

Production-selected team:

- Drivers: {prod_team['drivers']}
- Constructors: {prod_team['constructors']}
- 2×: {prod_team['boosted_driver_2x']}
- Cost: {prod_team['total_price']:.2f}M

Sprint-shadow-selected team:

- Drivers: {shadow_team['drivers']}
- Constructors: {shadow_team['constructors']}
- 2×: {shadow_team['boosted_driver_2x']}
- Cost: {shadow_team['total_price']:.2f}M

Swaps: drivers out **{_names_for_ids(comparison, 'driver', prod_team['driver_ids_out'])}**, drivers in **{_names_for_ids(comparison, 'driver', prod_team['driver_ids_in'])}**; constructors out **{_names_for_ids(comparison, 'constructor', prod_team['constructor_ids_out'])}**, constructors in **{_names_for_ids(comparison, 'constructor', prod_team['constructor_ids_in'])}**.

| Selected team | Scored by production EV | Scored by Sprint EV |
|---|---:|---:|
| Production-selected | {matrix.loc['production_ev', 'production_ev']:.2f} | {matrix.loc['production_ev', 'sprint_shadow_ev']:.2f} |
| Sprint-selected | {matrix.loc['sprint_shadow_ev', 'production_ev']:.2f} | {matrix.loc['sprint_shadow_ev', 'sprint_shadow_ev']:.2f} |

The separate combined-objective rows in `optimiser_comparison.csv` keep the same price-growth projection and weight 50; only the points field differs.

### Current default combined objective

Price-growth input is deterministic canonical official scoring from each asset's latest two completed observations. Production-selected: **{combined_prod_team['drivers']} | {combined_prod_team['constructors']} | 2× {combined_prod_team['boosted_driver_2x']}**. Sprint-selected: **{combined_shadow_team['drivers']} | {combined_shadow_team['constructors']} | 2× {combined_shadow_team['boosted_driver_2x']}**.

| Selected team | Scored by production EV | Scored by Sprint EV |
|---|---:|---:|
| Production-selected | {combined_cross.loc['production_ev', 'production_ev']:.2f} | {combined_cross.loc['production_ev', 'sprint_shadow_ev']:.2f} |
| Sprint-selected | {combined_cross.loc['sprint_shadow_ev', 'production_ev']:.2f} | {combined_cross.loc['sprint_shadow_ev', 'sprint_shadow_ev']:.2f} |

## 5. Decay sensitivity

{_markdown_table(decay_rows, ('decay', 'swap_count', 'overlap_count', 'shadow_team_score', 'shadow_advantage', 'production_penalty'), {'decay':'p','swap_count':'Swaps','overlap_count':'Overlap','shadow_team_score':'Shadow-team Sprint score','shadow_advantage':'Shadow advantage','production_penalty':'Production penalty'})}

The personal Sprint histories and calibration coefficients are identical in every row; only selected-race form weighting changes.

## 6. Current-season vs all-history production comparison

{_markdown_table(history_rows, ('history_mode', 'production_team_score', 'shadow_team_score', 'swap_count', 'shadow_advantage', 'production_penalty'), {'history_mode':'Production history','production_team_score':'Production team / production score','shadow_team_score':'Shadow team / Sprint score','swap_count':'Swaps','shadow_advantage':'Shadow advantage','production_penalty':'Production penalty'})}

The shadow side remains `2026_only` in both cases. Differences between these rows therefore isolate the effect of the production model's older-season prior.

## 7. Difference decomposition and sanity review

`Sprint EV = normal-equivalent EV + Sprint bonus`. `baseline_difference = normal-equivalent EV - production EV` separates normal-form disagreement from the actual Sprint adjustment.

Largest absolute baseline differences:

{_markdown_table(baseline_examples, ('entity','production_ev','shadow_normal_ev','baseline_difference','shadow_sprint_bonus'), {**labels,'baseline_difference':'Baseline difference'})}

Largest Sprint bonuses:

{_markdown_table(sprint_examples, ('entity','shadow_normal_ev','shadow_sprint_bonus','shadow_sprint_ev'), labels)}

Sanity cases:

{_markdown_table(sanity, ('entity','production_ev','shadow_normal_ev','shadow_sprint_bonus','shadow_sprint_ev','assessment','qualitative_note'), {**labels,'assessment':'Assessment','qualitative_note':'Review'})}

## 8. Recommendation

**{recommendation}**

This is an analysis recommendation only. Sprint-aware EV remains shadow-only and is not activated in production or optimisation.
"""


def generate_outputs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    snapshot, metadata = load_offline_snapshot()
    scenarios: dict[tuple[float, str], Scenario] = {}
    for decay in DECAYS:
        for history_mode in HISTORY_MODES:
            scenario = derive_scenario(snapshot, decay, history_mode)
            validate_market_prices(scenario.model, snapshot)
            scenarios[(decay, history_mode)] = scenario
    primary = scenarios[(0.85, HISTORY_MODE_ALL_SUPPORTED)]
    budget, budget_meta = deterministic_budget(primary.model.drivers, primary.model.constructors)

    optimiser_frames = []
    cross_frames = []
    summaries: dict[tuple[float, str, str], dict[str, Any]] = {}
    for (decay, history_mode), scenario in scenarios.items():
        for objective in (OBJECTIVE_POINTS_ONLY, OBJECTIVE_COMBINED):
            teams, cross, summary = run_optimizer_comparison(scenario, budget, objective)
            optimiser_frames.append(teams)
            cross_frames.append(cross)
            summaries[(decay, history_mode, objective)] = summary
    optimiser_rows = pd.concat(optimiser_frames, ignore_index=True)
    cross_rows = pd.concat(cross_frames, ignore_index=True)

    decay_rows = []
    for decay in DECAYS:
        scenario = scenarios[(decay, HISTORY_MODE_ALL_SUPPORTED)]
        summary = summaries[(decay, HISTORY_MODE_ALL_SUPPORTED, OBJECTIVE_POINTS_ONLY)]
        teams = optimiser_rows[
            optimiser_rows["decay"].eq(decay)
            & optimiser_rows["history_mode"].eq(HISTORY_MODE_ALL_SUPPORTED)
            & optimiser_rows["objective"].eq(OBJECTIVE_POINTS_ONLY)
        ]
        shadow_team = teams[teams["selected_by"].eq("sprint_shadow_ev")].iloc[0]
        top_driver_changes = scenario.comparison[scenario.comparison["entity_type"].eq("driver")].nlargest(
            3, "shadow_minus_production"
        )["entity"].str.cat(sep=", ")
        top_constructor_changes = scenario.comparison[
            scenario.comparison["entity_type"].eq("constructor")
        ].nlargest(3, "shadow_minus_production")["entity"].str.cat(sep=", ")
        decay_rows.append(
            {
                "decay": decay,
                "history_mode": HISTORY_MODE_ALL_SUPPORTED,
                "top_driver_changes": top_driver_changes,
                "top_constructor_changes": top_constructor_changes,
                "swap_count": summary["swap_count"],
                "overlap_count": summary["overlap_count"],
                "shadow_team_signature": f"{shadow_team['drivers']} | {shadow_team['constructors']} | 2x {shadow_team['boosted_driver_2x']}",
                "shadow_team_score": shadow_team["sprint_shadow_score"],
                "shadow_advantage": summary["shadow_advantage_of_shadow_team"],
                "production_penalty": summary["production_penalty_of_shadow_team"],
            }
        )
    decay_frame = pd.DataFrame(decay_rows)

    history_rows = []
    for history_mode in HISTORY_MODES:
        summary = summaries[(0.85, history_mode, OBJECTIVE_POINTS_ONLY)]
        teams = optimiser_rows[
            optimiser_rows["decay"].eq(0.85)
            & optimiser_rows["history_mode"].eq(history_mode)
            & optimiser_rows["objective"].eq(OBJECTIVE_POINTS_ONLY)
        ]
        production_team = teams[teams["selected_by"].eq("production_ev")].iloc[0]
        shadow_team = teams[teams["selected_by"].eq("sprint_shadow_ev")].iloc[0]
        history_rows.append(
            {
                "history_mode": history_mode,
                "sprint_shadow_history": "2026_only",
                "production_team": f"{production_team['drivers']} | {production_team['constructors']} | 2x {production_team['boosted_driver_2x']}",
                "shadow_team": f"{shadow_team['drivers']} | {shadow_team['constructors']} | 2x {shadow_team['boosted_driver_2x']}",
                "production_team_score": production_team["production_model_score"],
                "shadow_team_score": shadow_team["sprint_shadow_score"],
                "swap_count": summary["swap_count"],
                "overlap_count": summary["overlap_count"],
                "shadow_advantage": summary["shadow_advantage_of_shadow_team"],
                "production_penalty": summary["production_penalty_of_shadow_team"],
            }
        )
    history_frame = pd.DataFrame(history_rows)
    sanity = sanity_cases(primary.comparison)

    output_dir.mkdir(parents=True, exist_ok=True)
    primary.comparison.to_csv(output_dir / "asset_comparison.csv", index=False, float_format="%.12f")
    ranking_views(primary.comparison, "driver").to_csv(
        output_dir / "driver_rankings.csv", index=False, float_format="%.12f"
    )
    ranking_views(primary.comparison, "constructor").to_csv(
        output_dir / "constructor_rankings.csv", index=False, float_format="%.12f"
    )
    optimiser_rows.to_csv(output_dir / "optimiser_comparison.csv", index=False, float_format="%.12f")
    cross_rows.to_csv(output_dir / "cross_scored_teams.csv", index=False, float_format="%.12f")
    decay_frame.to_csv(output_dir / "decay_sensitivity.csv", index=False, float_format="%.12f")
    history_frame.to_csv(output_dir / "history_mode_comparison.csv", index=False, float_format="%.12f")
    sanity.to_csv(output_dir / "sanity_cases.csv", index=False, float_format="%.12f")
    report = render_report(
        primary,
        optimiser_rows,
        cross_rows,
        decay_frame,
        history_frame,
        sanity,
        metadata,
        budget_meta,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return {
        "output_dir": output_dir,
        "metadata": metadata,
        "budget": budget,
        "budget_metadata": budget_meta,
        "primary": primary,
        "optimiser_comparison": optimiser_rows,
        "cross_scored_teams": cross_rows,
        "decay_sensitivity": decay_frame,
        "history_mode_comparison": history_frame,
        "sanity_cases": sanity,
    }


def main() -> int:
    outputs = generate_outputs()
    primary = outputs["primary"]
    points = outputs["optimiser_comparison"]
    points = points[
        points["decay"].eq(0.85)
        & points["history_mode"].eq(HISTORY_MODE_ALL_SUPPORTED)
        & points["objective"].eq(OBJECTIVE_POINTS_ONLY)
    ]
    print(f"Wrote Sprint-shadow comparison to {outputs['output_dir']}")
    print(
        f"Accepted feed {outputs['metadata']['feed_round']}; budget {outputs['budget']:.2f}M; "
        f"assets {len(primary.comparison)}"
    )
    for row in points.itertuples(index=False):
        print(
            f"{row.selected_by}: {row.drivers} | {row.constructors} | "
            f"2x {row.boosted_driver_2x} | production {row.production_model_score:.2f} | "
            f"shadow {row.sprint_shadow_score:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
