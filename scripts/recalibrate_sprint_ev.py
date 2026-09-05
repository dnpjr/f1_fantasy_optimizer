#!/usr/bin/env python3
"""Build or explicitly promote an offline Sprint-EV calibration candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f1fantasy.historical_scores import DEFAULT_CANONICAL_DATASET_PATH
from f1fantasy.sprint_shadow import (
    DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH,
    load_sprint_production_calibration,
    parse_sprint_production_calibration,
)
from scripts.build_2026_sprint_final_candidate import run_build


DEFAULT_OUTPUT = PROJECT_ROOT / "reports/sprint_recalibration_candidate"
DEFAULT_SCHEDULE = PROJECT_ROOT / "data/cache/schedule_2026.csv"
DEFAULT_MARKET = PROJECT_ROOT / "data/cache/verified_fantasy_market.json"
DEFAULT_PREVIOUS_REPORT = PROJECT_ROOT / "reports/2026_sprint_partial_pooling"


def _next_version(current: str) -> str:
    match = re.fullmatch(r"sprint_ev_2026_v(\d+)", str(current))
    if match is None:
        raise ValueError(f"Active Sprint calibration version is not incrementable: {current!r}")
    return f"sprint_ev_2026_v{int(match.group(1)) + 1}"


def runtime_candidate_payload(result: Mapping[str, Any], active_version: str) -> dict[str, Any]:
    """Convert reviewed research output into the small runtime schema."""
    models = result["models"]
    candidate = result["candidate"]
    drivers = candidate[candidate["entity_type"].eq("driver")].sort_values("entity_id")
    driver_model = models["driver"]
    constructor_model = models["constructor"]
    shrinkage = driver_model["shrinkage"]
    source_version = str(result["prepared"]["source_data_version"])
    return {
        "model_version": _next_version(active_version),
        "generated_at": datetime.now(UTC).isoformat(),
        "source_data_version": source_version,
        "completed_rounds": result["model_json"]["completed_rounds"],
        "sprint_rounds": result["model_json"]["sprint_rounds"],
        "source_research_model": str(result["model_json"]["script_version"]),
        "calibration_season": 2026,
        "calibration_status": "candidate",
        "driver": {
            "form_mean": float(driver_model["strength_parameters"]["normal_ev_mean"]),
            "form_sd": float(
                driver_model["strength_parameters"]["normal_ev_population_sd"]
            ),
            "group_intercept": float(driver_model["mu"]),
            "group_slope": float(driver_model["lambda"]),
            "within_variance": float(shrinkage["within_residual_variance"]),
            "tau_squared": float(shrinkage["tau_asset_squared"]),
            "personal_history": [
                {
                    "canonical_entity_id": str(row.entity_id),
                    "name": str(row.entity),
                    "personal_mean_bonus": (
                        float(row.observed_mean_sprint_bonus)
                        if int(row.observed_sprint_count) > 0
                        else None
                    ),
                    "observation_count": int(row.observed_sprint_count),
                }
                for row in drivers.itertuples(index=False)
            ],
        },
        "constructor": {
            "intercept": float(constructor_model["mu"]),
            "slope": float(constructor_model["lambda"]),
            "form_weight": 0.75,
            "price_weight": 0.25,
            "future_event_effect": 0.0,
        },
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        rows: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_flatten(value[key], child))
        return rows
    if isinstance(value, list):
        rows: dict[str, Any] = {}
        for index, item in enumerate(value):
            key = item.get("canonical_entity_id", index) if isinstance(item, Mapping) else index
            rows.update(_flatten(item, f"{prefix}[{key}]"))
        return rows
    return {prefix: value}


def calibration_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    old = _flatten(before)
    new = _flatten(after)
    return [
        {"field": key, "before": old.get(key), "after": new.get(key)}
        for key in sorted(set(old) | set(new))
        if old.get(key) != new.get(key)
    ]


def generate_candidate(
    output: Path = DEFAULT_OUTPUT,
    *,
    active_path: Path = DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH,
    canonical_path: Path = DEFAULT_CANONICAL_DATASET_PATH,
    schedule_path: Path = DEFAULT_SCHEDULE,
    market_path: Path = DEFAULT_MARKET,
    previous_report: Path = DEFAULT_PREVIOUS_REPORT,
) -> dict[str, Any]:
    """Run the offline fit and write a candidate without touching production."""
    active_raw = json.loads(active_path.read_text(encoding="utf-8"))
    active = load_sprint_production_calibration(active_path)
    result = run_build(
        canonical_path,
        schedule_path,
        market_path,
        previous_report,
        output,
        maximum_round=None,
    )
    candidate = runtime_candidate_payload(result, active.model_version)
    candidate["input_sha256"] = {
        label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for label, path in (("canonical", canonical_path), ("schedule", schedule_path), ("market", market_path))
    }
    parse_sprint_production_calibration(candidate, allowed_statuses=("candidate",))
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "runtime_calibration_candidate.json"
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    changes = calibration_changes(active_raw, candidate)
    (output / "comparison_to_active.json").write_text(
        json.dumps(changes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "PROMOTION_REVIEW.md").write_text(
        "# Sprint calibration promotion review\n\n"
        f"Active: `{active.model_version}`\n\n"
        f"Candidate: `{candidate['model_version']}`\n\n"
        f"Changed fields: {len(changes)}\n\n"
        "Nothing in production was changed. After review, promotion requires the explicit command:\n\n"
        "```bash\npython scripts/recalibrate_sprint_ev.py --promote "
        f"{candidate_path}\n```\n",
        encoding="utf-8",
    )
    return {"candidate_path": candidate_path, "candidate": candidate, "changes": changes}


def promote_candidate(
    candidate_path: Path,
    *,
    active_path: Path = DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH,
    archive_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate, archive and atomically promote only after explicit invocation."""
    candidate_raw = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate = parse_sprint_production_calibration(
        candidate_raw, allowed_statuses=("candidate",)
    )
    active_raw = json.loads(active_path.read_text(encoding="utf-8"))
    active = load_sprint_production_calibration(active_path)
    expected_version = _next_version(active.model_version)
    if candidate.model_version != expected_version:
        raise ValueError(
            f"Candidate version must be {expected_version}, found {candidate.model_version}."
        )
    promoted = deepcopy(candidate_raw)
    promoted["calibration_status"] = "approved_production"
    parse_sprint_production_calibration(promoted)
    changes = calibration_changes(active_raw, promoted)

    archive = archive_dir or active_path.parent / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    previous_path = archive / f"{active.model_version}.json"
    if previous_path.exists():
        if previous_path.read_bytes() != active_path.read_bytes():
            raise FileExistsError(f"Calibration archive collision: {previous_path}")
    else:
        previous_path.write_bytes(active_path.read_bytes())

    active_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=active_path.parent,
            prefix=f".{active_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(promoted, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, active_path)
        temporary_name = None
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--active", type=Path, default=DEFAULT_SPRINT_PRODUCTION_CALIBRATION_PATH)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_DATASET_PATH)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--current-prices", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--previous-report", type=Path, default=DEFAULT_PREVIOUS_REPORT)
    parser.add_argument(
        "--promote",
        type=Path,
        help="Explicitly promote a reviewed runtime_calibration_candidate.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.promote is not None:
        changes = promote_candidate(args.promote, active_path=args.active)
        print(f"Promoted {args.promote} to {args.active}")
        print(json.dumps(changes, indent=2, sort_keys=True))
        return 0
    result = generate_candidate(
        args.output,
        active_path=args.active,
        canonical_path=args.canonical,
        schedule_path=args.schedule,
        market_path=args.current_prices,
        previous_report=args.previous_report,
    )
    print(f"Saved candidate to {result['candidate_path']}")
    print("Production calibration was not changed; review before using --promote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
