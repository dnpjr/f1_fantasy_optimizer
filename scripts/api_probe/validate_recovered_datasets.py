"""Validate consistency of the licensed 2021–2022 recovered score files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
RAW_ROOT = HERE / "raw-data"
OUTPUT = HERE / "recovered_dataset_validation.json"
KAGGLE_ROOT = RAW_ROOT / "kaggle_formula_1_fantasy_2021"
GITHUB_ROOT = RAW_ROOT / "github_jm1261_fantasy_f1_league"

RACES_2021 = (
    "bahrain",
    "imola",
    "portugal",
    "spain",
    "monaco",
    "azerbaijan",
    "france",
    "stryria",
    "austria",
    "britain",
    "hungary",
    "belgium",
    "netherlands",
    "italy",
    "russia",
    "turkey",
    "usa",
    "mexico",
    "brazil",
    "qatar",
    "saudiarabia",
    "abudhabi",
)


def normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def surname(value: str) -> str:
    return normalise(value.split()[-1])


def load_json(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [float(item) for item in values] for key, values in payload.items()}


def kaggle_race_points(asset_type: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    key_field = "Driver Name" if asset_type == "driver" else "Constructor Name"
    for race in RACES_2021:
        # The published file uses a one-letter typo for the Netherlands constructor file.
        stem = race
        if race == "netherlands" and asset_type == "constructor":
            stem = "netherands"
        if race == "stryria" and asset_type == "constructor":
            stem = "styria"
        path = KAGGLE_ROOT / f"{stem}_{asset_type}_performance.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            key = surname(row[key_field]) if asset_type == "driver" else normalise(row[key_field])
            result.setdefault(key, []).append(float(row["Fantasy Points"]))
    return result


def compare_2021(asset_type: str) -> dict[str, Any]:
    github_name = (
        "Individual_Driver_Points.config"
        if asset_type == "driver"
        else "Individual_Team_Points.config"
    )
    github = load_json(GITHUB_ROOT / "Data/2021/Lineup" / github_name)
    github_normalised = {
        (surname(key) if asset_type == "driver" else normalise(key)): values
        for key, values in github.items()
    }
    kaggle = kaggle_race_points(asset_type)
    shared = sorted(set(github_normalised) & set(kaggle))
    mismatches: list[dict[str, Any]] = []
    observations = 0
    for entity in shared:
        for index, (left, right) in enumerate(zip(github_normalised[entity], kaggle[entity])):
            observations += 1
            if left != right:
                mismatches.append(
                    {
                        "entity": entity,
                        "round_index": index + 1,
                        "github": left,
                        "kaggle": right,
                    }
                )
    return {
        "asset_type": asset_type,
        "github_entity_count": len(github_normalised),
        "kaggle_entity_count": len(kaggle),
        "shared_entity_count": len(shared),
        "observations_compared": observations,
        "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:20],
        "github_only_entities": sorted(set(github_normalised) - set(kaggle)),
        "kaggle_only_entities": sorted(set(kaggle) - set(github_normalised)),
    }


def validate_cumulative(year: int, asset_type: str) -> dict[str, Any]:
    stem = "Driver" if asset_type == "driver" else "Team"
    cumulative = load_json(GITHUB_ROOT / f"Data/{year}/Lineup/{stem}_Points.config")
    individual = load_json(GITHUB_ROOT / f"Data/{year}/Lineup/Individual_{stem}_Points.config")
    mismatches: list[dict[str, Any]] = []
    observations = 0
    for entity in sorted(set(cumulative) & set(individual)):
        running = 0.0
        for round_index, value in enumerate(individual[entity], start=1):
            running += value
            observations += 1
            expected = cumulative[entity][round_index - 1]
            if running != expected:
                mismatches.append(
                    {
                        "entity": entity,
                        "round_index": round_index,
                        "cumulative": expected,
                        "sum_of_individual": running,
                    }
                )
    return {
        "season": year,
        "asset_type": asset_type,
        "entity_count": len(individual),
        "round_count_values": sorted({len(values) for values in individual.values()}),
        "observations_checked": observations,
        "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:20],
    }


def main() -> None:
    output = {
        "cross_source_2021": [compare_2021("driver"), compare_2021("constructor")],
        "github_internal_consistency": [
            validate_cumulative(year, asset_type)
            for year in (2021, 2022)
            for asset_type in ("driver", "constructor")
        ],
        "limitations": [
            "The checks establish agreement between third-party records, not official API provenance.",
            "The GitHub records contain total points only and do not identify scoring components.",
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
