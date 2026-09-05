#!/usr/bin/env python3
"""Build the canonical recorded F1 Fantasy score dataset for 2023–2026."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from f1fantasy.fantasy_api import fetch_players, fetch_teams
from f1fantasy.historical_scores import (
    CANONICAL_COLUMNS,
    DATA_VERSION,
    approximation_comparison,
    coverage_report,
    load_canonical_scores,
    normalise_official_playerstats,
    normalise_third_party_recorded,
    resolve_score_precedence,
    validate_canonical_scores,
)
from f1fantasy.model import _constructor_round_points, compute_weekend_points
from f1fantasy.player_stats import fetch_recent_points_for_roster, parse_player_race_points


DEFAULT_RAW_ROOT = REPO_ROOT / "scripts/api_probe/raw-data/github_jm1261_fantasy_f1_league"
DEFAULT_FIXTURE_ROOT = REPO_ROOT / "scripts/api_probe/fixtures"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/generated" / DATA_VERSION
EXPECTED_COMPLETED_COVERAGE = {
    2023: {"driver": (440, 22), "constructor": (220, 22)},
    2024: {"driver": (480, 24), "constructor": (240, 24)},
    2025: {"driver": (480, 24), "constructor": (240, 24)},
}


def _path_reference(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schedules() -> dict[int, pd.DataFrame]:
    schedules: dict[int, pd.DataFrame] = {}
    for season in (2023, 2024, 2025, 2026):
        path = REPO_ROOT / f"data/cache/schedule_{season}.csv"
        if path.exists():
            schedules[season] = pd.read_csv(path)
    return schedules


def _fixture_payloads(fixtures: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    driver_frames: list[pd.DataFrame] = []
    constructor_frames: list[pd.DataFrame] = []
    players: list[dict] = []
    teams: list[dict] = []
    for path in sorted(fixtures.glob("playerstats_*_2026_sanitised.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        payload = wrapper.get("payload", wrapper)
        value = payload.get("Value", {})
        source_id = int(value["PlayerId"])
        parsed = parse_player_race_points(payload, source_id)
        metadata = wrapper.get("_fixture_metadata", {})
        name = metadata.get("entity_name") or metadata.get("name") or str(source_id)
        tla = metadata.get("abbreviation") or metadata.get("tla") or ""
        if str(value.get("PlayerSkill")) == "2":
            constructor_frames.append(parsed)
            teams.append({"teamId": source_id, "name": name, "tla": tla})
        else:
            driver_frames.append(parsed)
            players.append({"playerId": source_id, "name": name, "tla": tla, "team": metadata.get("constructor_name", "")})
    return (
        pd.concat(driver_frames, ignore_index=True) if driver_frames else pd.DataFrame(),
        pd.concat(constructor_frames, ignore_index=True) if constructor_frames else pd.DataFrame(),
        pd.DataFrame(players),
        pd.DataFrame(teams),
    )


def _live_official(feed_round: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = fetch_players(feed_round=feed_round)
    teams = fetch_teams(feed_round=feed_round)
    _, driver_points, _ = fetch_recent_points_for_roster(players.rename(columns={"playerId": "id"}), "driver")
    _, constructor_points, _ = fetch_recent_points_for_roster(teams.rename(columns={"teamId": "id"}), "constructor")
    return driver_points, constructor_points, players, teams


def _snapshot_official(path: Path, schedules: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Replay a retained completed-weekend snapshot through the existing parser."""
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    season, round_no = int(snapshot["season"]), int(snapshot["round"])
    schedule = schedules[season]
    event = schedule[schedule["round"].eq(round_no)]
    if len(event) != 1 or pd.to_datetime(event.iloc[0]["date"], utc=True) >= pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("Snapshot must identify one past completed event.")
    parsed = []
    for item in snapshot["payloads"]:
        rows = parse_player_race_points(item["payload"], int(item["player_id"]))
        if rows.empty:
            continue
        rows = rows[rows["season"].eq(season) & rows["round"].eq(round_no)]
        parsed.append(rows)
    if not parsed:
        raise ValueError("Snapshot has no completed-weekend observations.")
    points = pd.concat(parsed, ignore_index=True)
    official, warnings = normalise_official_playerstats(
        points[points["asset_type"].eq("driver")],
        points[points["asset_type"].eq("constructor")],
        pd.DataFrame(snapshot["players"]), pd.DataFrame(snapshot["teams"]),
        schedule=schedule,
    )
    if warnings:
        raise ValueError(f"Unresolved snapshot identities: {warnings}")
    # Reject truncated imports; missing session scores within a played row remain
    # null and are omitted by the fit, never converted to zero.
    expected = snapshot["expected_played_counts"]
    counts = official.groupby("entity_type").size().to_dict()
    if counts != expected:
        raise ValueError(f"Incomplete snapshot roster: {counts}, expected {expected}")
    if set(official["round"]) != {round_no}:
        raise ValueError("Snapshot event name/round mismatch.")
    return official, warnings


def _approximation_rows() -> pd.DataFrame:
    result_frames: list[pd.DataFrame] = []
    qualifying_frames: list[pd.DataFrame] = []
    sprint_frames: list[pd.DataFrame] = []
    for season in (2023, 2024, 2025):
        for stem, target in (("results", result_frames), ("qualifying", qualifying_frames), ("sprint", sprint_frames)):
            path = REPO_ROOT / f"data/cache/{stem}_{season}.csv"
            if path.exists():
                target.append(pd.read_csv(path))
    if not result_frames:
        return pd.DataFrame()
    results = pd.concat(result_frames, ignore_index=True)
    qualifying = pd.concat(qualifying_frames, ignore_index=True) if qualifying_frames else pd.DataFrame()
    sprint = pd.concat(sprint_frames, ignore_index=True) if sprint_frames else pd.DataFrame()
    driver = compute_weekend_points(results, qualifying, sprint, current_season=2026)
    constructor = _constructor_round_points(driver)
    driver_out = driver.rename(columns={"driverId": "canonical_entity_id", "weekend_points": "fantasy_points_total"})
    driver_out["entity_type"] = "driver"
    constructor_out = constructor.rename(columns={"constructorId": "canonical_entity_id", "constructor_weekend_points": "fantasy_points_total"})
    constructor_out["entity_type"] = "constructor"
    return pd.concat(
        [driver_out[["season", "round", "entity_type", "canonical_entity_id", "fantasy_points_total"]],
         constructor_out[["season", "round", "entity_type", "canonical_entity_id", "fantasy_points_total"]]],
        ignore_index=True,
    )


def _validate_production_coverage(canonical: pd.DataFrame) -> dict:
    if canonical["season"].lt(2023).any():
        raise ValueError("Production canonical data must not contain a season before 2023")
    if canonical["is_reconstructed"].fillna(False).astype(bool).any():
        raise ValueError("Production canonical data must not contain reconstructed rows")
    coverage = coverage_report(canonical).set_index(["season", "entity_type"])
    for season, entity_types in EXPECTED_COMPLETED_COVERAGE.items():
        for entity_type, (expected_rows, expected_races) in entity_types.items():
            if (season, entity_type) not in coverage.index:
                raise ValueError(f"Missing {season} {entity_type} production coverage")
            actual = coverage.loc[(season, entity_type)]
            if [int(actual["rows"]), int(actual["races"])] != [expected_rows, expected_races]:
                raise ValueError(
                    f"Unexpected {season} {entity_type} coverage: "
                    f"{int(actual['rows'])} rows across {int(actual['races'])} races"
                )
    historic = canonical[canonical["season"].isin((2023, 2024, 2025))]
    if historic["price"].isna().any():
        raise ValueError("Every 2023–2025 production row must retain its recorded price")
    if not historic["authority_class"].eq("third_party_recorded").all():
        raise ValueError("Every 2023–2025 production row must be third-party recorded")
    official = canonical[canonical["season"] == 2026]
    if not official.empty and not official["fantasy_score_origin"].eq("official_recorded").all():
        raise ValueError("Every emitted 2026 production row must be official recorded")
    return {
        "earliest_season": int(canonical["season"].min()),
        "reconstructed_rows": 0,
        "completed_season_rows": int(len(historic)),
        "official_2026_rows": int(len(official)),
    }


def build(args: argparse.Namespace) -> dict:
    schedules = _schedules()
    third_party = normalise_third_party_recorded(args.raw_root, schedules)
    official = pd.DataFrame(columns=CANONICAL_COLUMNS)
    official_warnings: list[str] = []
    official_mode = "not_requested"
    if args.fetch_official:
        driver_points, constructor_points, players, teams = _live_official(args.feed_round)
        official_mode = "live_public_playerstats"
    elif args.include_fixtures:
        driver_points, constructor_points, players, teams = _fixture_payloads(args.fixture_root)
        official_mode = "sanitised_test_fixtures"
    else:
        driver_points = constructor_points = players = teams = pd.DataFrame()
    if official_mode != "not_requested":
        results_2026_path = REPO_ROOT / "data/cache/results_2026.csv"
        results_2026 = pd.read_csv(results_2026_path) if results_2026_path.exists() else pd.DataFrame()
        official, official_warnings = normalise_official_playerstats(
            driver_points, constructor_points, players, teams,
            results=results_2026, schedule=schedules.get(2026, pd.DataFrame()),
        )
    else:
        existing_path = args.output_root / "historical_fantasy_scores_2023_2026.csv"
        if existing_path.exists():
            cached = load_canonical_scores(existing_path)
            official = cached[
                (cached["season"] == 2026)
                & cached["is_official"].fillna(False).astype(bool)
                & (cached["data_version"] == DATA_VERSION)
            ].copy()
            if not official.empty:
                official_mode = "existing_valid_official_cache"

    snapshot_path = getattr(args, "official_snapshot", None)
    if snapshot_path is not None:
        added, official_warnings = _snapshot_official(snapshot_path, schedules)
        overlap = official.merge(added, on=["season", "round", "entity_type", "canonical_entity_id"])
        if not overlap.empty:
            # Idempotent replay is safe; conflicting records require explicit review.
            existing = official.set_index(["season", "round", "entity_type", "canonical_entity_id"])
            incoming = added.set_index(existing.index.names)
            shared = existing.index.intersection(incoming.index)
            if not existing.loc[shared].fillna("").eq(incoming.loc[shared].fillna("")).all().all():
                raise ValueError("Snapshot conflicts with existing official history.")
        official = pd.concat([official, added], ignore_index=True).drop_duplicates(
            ["season", "round", "entity_type", "canonical_entity_id"], keep="first"
        )
        official_mode = "existing_official_cache_plus_snapshot"

    canonical = resolve_score_precedence(third_party, official)
    return write_canonical_outputs(
        canonical, args.output_root, raw_root=args.raw_root,
        official_mode=official_mode, official_warnings=official_warnings,
        snapshot_path=snapshot_path,
    )


def write_canonical_outputs(
    canonical: pd.DataFrame,
    output: Path,
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    official_mode: str = "existing_official_cache_plus_snapshot",
    official_warnings: list[str] | None = None,
    snapshot_path: Path | None = None,
    published_root: Path | None = None,
) -> dict:
    """Write the existing canonical representations/reports, also usable in staging."""
    published_root = published_root or output
    official_warnings = official_warnings or []
    validation = validate_canonical_scores(canonical)
    validation["production_coverage"] = _validate_production_coverage(canonical)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "historical_fantasy_scores_2023_2026.csv"
    canonical.to_csv(csv_path, index=False)
    parquet_path = output / "historical_fantasy_scores_2023_2026.parquet"
    parquet_written = False
    parquet_error = None
    try:
        canonical.to_parquet(parquet_path, index=False)
        parquet_written = True
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        parquet_error = str(exc)

    coverage = coverage_report(canonical)
    coverage.to_csv(output / "source_coverage.csv", index=False)
    (output / "source_coverage.json").write_text(
        json.dumps(coverage.to_dict("records"), indent=2, default=str) + "\n", encoding="utf-8"
    )
    source_rows = canonical.copy(deep=True)
    source_rows["has_components"] = source_rows[
        ["qualifying_points", "sprint_qualifying_points", "sprint_points", "race_points", "other_points"]
    ].notna().any(axis=1)
    source_summary = source_rows.groupby(
        ["season", "source_name", "authority_class", "source_licence"],
        dropna=False,
        as_index=False,
    ).agg(
        rows=("canonical_entity_id", "size"),
        races=("round", "nunique"),
        price_rows=("price", lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum())),
        component_rows=("has_components", "sum"),
        reconstructed_rows=("is_reconstructed", "sum"),
    )
    source_summary.to_csv(output / "source_summary.csv", index=False)
    (output / "source_summary.json").write_text(
        json.dumps(source_summary.to_dict("records"), indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    approximation = _approximation_rows()
    if approximation.empty:
        comparison = pd.DataFrame()
        discrepancies = pd.DataFrame()
    else:
        comparison, discrepancies = approximation_comparison(
            canonical[canonical["season"].isin((2023, 2024, 2025))], approximation
        )
    comparison.to_csv(output / "approximation_vs_recorded_summary.csv", index=False)
    discrepancies.head(100).to_csv(output / "approximation_vs_recorded_largest_discrepancies.csv", index=False)

    raw_files = sorted(
        path for season in (2023, 2024, 2025)
        for path in Path(raw_root).glob(f"Data/{season}/Lineup/*_Results.json")
        if path.is_file()
    )
    manifest = {
        "data_version": DATA_VERSION,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "supported_seasons": [2023, 2024, 2025, 2026],
        "intentionally_excluded_seasons": [2021, 2022],
        "archived_research_seasons": [2022],
        "official_mode": official_mode,
        "official_snapshot": ({"path": _path_reference(snapshot_path),
                               "sha256": _sha256(snapshot_path)} if snapshot_path else None),
        "official_warnings": official_warnings,
        "validation": validation,
        "outputs": {
            "csv": {"path": _path_reference(published_root / csv_path.name), "sha256": _sha256(csv_path)},
            "parquet": {"path": _path_reference(published_root / parquet_path.name), "written": parquet_written, "error": parquet_error},
        },
        "sources": [
            {
                "seasons": [2023, 2024, 2025],
                "name": "jm1261/Fantasy-F1-League",
                "authority_class": "third_party_recorded",
                "licence": "MIT",
                "reference": "https://github.com/jm1261/Fantasy-F1-League",
                "files": [{"path": _path_reference(path), "sha256": _sha256(path)} for path in raw_files],
            },
            {
                "seasons": [2026],
                "name": "Formula 1 Fantasy public playerstats feed",
                "authority_class": "official",
                "reference": "https://fantasy.formula1.com/feeds/popup/playerstats_{player_id}.json",
                "retained_raw_payloads": False,
            },
        ],
        "limitations": [
            "The retained MIT source records exact totals and prices for 2023–2025 but no reliable Q/S/R component split; component fields remain null.",
            "Recorded 2022 totals remain archived for research but are intentionally outside the production build because exact historical prices and detailed components are unavailable.",
            "The approximation comparison is limited to rows present in the repository's pre-existing ordinary-results caches.",
        ],
    }
    (output / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--feed-round", type=int, default=12, help="verified current official market feed number")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-official", action="store_true", help="GET current official 2026 playerstats")
    mode.add_argument("--official-snapshot", type=Path, help="Append a retained complete official weekend snapshot to existing history")
    mode.add_argument("--include-fixtures", action="store_true", help="include sanitised official fixture samples")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
