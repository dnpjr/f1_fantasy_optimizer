"""Probe historical query parameters on the confirmed public playerstats route.

Only the working ``fantasy.formula1.com/feeds/popup/playerstats_*.json`` route
family is contacted. Requests are GET-only and begin at least one second apart.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlencode

import requests


HERE = Path(__file__).resolve().parent
PRIOR_RESULTS = HERE / "results.json"
OUTPUT_JSON = HERE / "playerstats_historical_variants.json"
OUTPUT_CSV = HERE / "playerstats_historical_variants.csv"
FIXTURE_DIR = HERE / "fixtures"

BASE_PATTERN = "https://fantasy.formula1.com/feeds/popup/playerstats_{player_id}.json"
TIMEOUT_SECONDS = 15
MIN_INTERVAL_SECONDS = 1.0
HISTORICAL_SEASONS = (2025, 2024, 2023, 2022, 2021)
PARAMETER_FAMILIES = ("season", "season_name", "year")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
SENSITIVE_KEY = re.compile(
    r"authorization|cookie|email|guid|password|secret|session|token",
    re.IGNORECASE,
)


def sanitise(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SENSITIVE_KEY.search(str(key)) else sanitise(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitise(item) for item in value]
    return value


def without_feed_time(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    return {key: value for key, value in payload.items() if key != "FeedTime"}


def payload_signature(payload: Any) -> str:
    encoded = json.dumps(
        without_feed_time(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def observed_seasons(value: Any) -> list[int]:
    found: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in {"season", "season_name", "seasonname", "year"}:
                    try:
                        found.add(int(child))
                    except (TypeError, ValueError):
                        pass
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


def gameday_ids(payload: Any) -> list[int]:
    value = payload.get("Value", {}) if isinstance(payload, dict) else {}
    rows = value.get("GamedayWiseStats", []) if isinstance(value, dict) else []
    found: list[int] = []
    for row in rows:
        try:
            item = int(row.get("GamedayId"))
        except (AttributeError, TypeError, ValueError):
            continue
        if item not in found:
            found.append(item)
    return found


def stats_shape(payload: Any) -> dict[str, Any]:
    value = payload.get("Value", {}) if isinstance(payload, dict) else {}
    if not isinstance(value, dict):
        return {}
    shape: dict[str, Any] = {
        "top_level_keys": sorted(payload),
        "value_keys": sorted(value),
        "player_id": value.get("PlayerId"),
        "player_skill": value.get("PlayerSkill"),
        "gameday_count": len(value.get("GamedayWiseStats", []) or []),
        "match_count": len(value.get("MatchWiseStats", []) or []),
        "fixture_count": len(value.get("FixtureWiseStats", []) or []),
    }
    for collection in ("GamedayWiseStats", "MatchWiseStats", "FixtureWiseStats"):
        rows = value.get(collection, []) or []
        shape[f"{collection}_row_keys"] = sorted(rows[0]) if rows and isinstance(rows[0], dict) else []
    matches = value.get("MatchWiseStats", []) or []
    sessions = matches[0].get("RaceDayWise", []) if matches and isinstance(matches[0], dict) else []
    shape["RaceDayWise_row_keys"] = sorted(sessions[0]) if sessions and isinstance(sessions[0], dict) else []
    stats = sessions[0].get("StatsWise", []) if sessions and isinstance(sessions[0], dict) else []
    shape["StatsWise_row_keys"] = sorted(stats[0]) if stats and isinstance(stats[0], dict) else []
    return shape


class RateLimitedGetter:
    def __init__(self) -> None:
        self.last_started: float | None = None

    def get(self, url: str) -> requests.Response:
        now = time.monotonic()
        if self.last_started is not None:
            remaining = MIN_INTERVAL_SECONDS - (now - self.last_started)
            if remaining > 0:
                time.sleep(remaining)
        self.last_started = time.monotonic()
        return requests.get(
            url,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"},
        )


def discover_ids() -> dict[str, int]:
    payload = json.loads(PRIOR_RESULTS.read_text(encoding="utf-8"))
    found = payload.get("discoveries", {}).get("assets_by_season", {}).get("2026_public_feed", {})
    result = {key: int(value) for key, value in found.items() if key in {"driver", "constructor"}}
    if set(result) != {"driver", "constructor"}:
        raise RuntimeError("Could not discover both current asset IDs from results.json")
    return result


def main() -> None:
    getter = RateLimitedGetter()
    ids = discover_ids()
    rows: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for asset_type, player_id in ids.items():
        url = BASE_PATTERN.format(player_id=player_id)
        response = getter.get(url)
        response.raise_for_status()
        payload = response.json()
        baselines[asset_type] = payload
        fixture = {
            "_fixture_metadata": {
                "source": url,
                "fetched_at_utc": datetime.now(UTC).isoformat(),
                "asset_type": asset_type,
                "provenance": "Official public F1 Fantasy playerstats response.",
                "licence": "No separate reuse licence identified; retained as a sanitised schema fixture for research.",
                "credentials_used": False,
            },
            "payload": sanitise(payload),
        }
        (FIXTURE_DIR / f"playerstats_{asset_type}_2026_sanitised.json").write_text(
            json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "asset_type": asset_type,
                "player_id": player_id,
                "parameter": "baseline",
                "requested_season": "",
                "url": url,
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "response_length": len(response.content),
                "valid_json": True,
                "observed_seasons": observed_seasons(payload),
                "gameday_ids": gameday_ids(payload),
                "matches_baseline": True,
                "historical_season_verified": False,
                "diagnostic": "Baseline current-season response.",
            }
        )

    baseline = baselines["driver"]
    baseline_signature = payload_signature(baseline)
    driver_id = ids["driver"]
    base_url = BASE_PATTERN.format(player_id=driver_id)

    for parameter in PARAMETER_FAMILIES:
        for requested_season in HISTORICAL_SEASONS:
            url = f"{base_url}?{urlencode({parameter: requested_season})}"
            response = getter.get(url)
            valid_json = False
            payload: Any = None
            try:
                payload = response.json()
                valid_json = True
            except ValueError:
                pass
            seasons = observed_seasons(payload)
            matches = bool(valid_json and payload_signature(payload) == baseline_signature)
            verified = requested_season in seasons
            diagnostic = (
                f"Verified payload season {requested_season}."
                if verified
                else "Requested year is absent from payload."
            )
            if matches:
                diagnostic += " Payload Value matches the 2026 baseline; parameter is ignored."
            rows.append(
                {
                    "asset_type": "driver",
                    "player_id": driver_id,
                    "parameter": parameter,
                    "requested_season": requested_season,
                    "url": url,
                    "http_status": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "response_length": len(response.content),
                    "valid_json": valid_json,
                    "observed_seasons": seasons,
                    "gameday_ids": gameday_ids(payload),
                    "matches_baseline": matches,
                    "historical_season_verified": verified,
                    "diagnostic": diagnostic,
                }
            )
            print(response.status_code, parameter, requested_season, seasons, diagnostic)
            if matches and seasons == observed_seasons(baseline):
                break

    output = {
        "configuration": {
            "hostname": "fantasy.formula1.com",
            "path_template": "/feeds/popup/playerstats_{player_id}.json",
            "timeout_seconds": TIMEOUT_SECONDS,
            "minimum_request_interval_seconds": MIN_INTERVAL_SECONDS,
            "methods": ["GET"],
            "headers": {
                "Accept": "application/json,text/plain,*/*",
                "User-Agent": UA,
            },
            "credentials_used": False,
        },
        "baseline_schema": {
            asset_type: stats_shape(payload) for asset_type, payload in baselines.items()
        },
        "results": rows,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_row = dict(row)
        csv_row["observed_seasons"] = json.dumps(row["observed_seasons"])
        csv_row["gameday_ids"] = json.dumps(row["gameday_ids"])
        csv_rows.append(csv_row)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote schema fixtures under {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
