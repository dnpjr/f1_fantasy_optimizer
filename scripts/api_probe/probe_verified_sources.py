"""Inspect only endpoint templates discovered in genuine public frontends.

The script performs low-volume GET requests, starts requests at least one second
apart, and stores response metadata/schema summaries rather than third-party raw
datasets.  It does not use credentials or execute downloaded JavaScript.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import time
from typing import Any

import requests


HERE = Path(__file__).resolve().parent
OUTPUT_JSON = HERE / "verified_source_results.json"
OUTPUT_CSV = HERE / "verified_source_results.csv"

TIMEOUT_SECONDS = 15
MIN_INTERVAL_SECONDS = 1.0
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

OFFICIAL_ENDPOINTS = (
    (
        "official_web_config",
        "https://fantasy.formula1.com/feeds/v2/apps/web_config.json",
    ),
    (
        "official_current_statistics",
        "https://fantasy.formula1.com/feeds/v2/statistics/driverconstructors_4.json",
    ),
    (
        "official_current_market",
        "https://fantasy.formula1.com/feeds/drivers/12_en.json",
    ),
)
TOOLS_YEARS = (2023, 2024, 2025, 2026)
TOOLS_PATTERN = "https://f1fantasytools.com/api/statistics/{year}"

YEAR_KEYS = {
    "season",
    "season_name",
    "seasonname",
    "seasonyear",
    "year",
    "championshipyear",
}
SENSITIVE_KEY = re.compile(
    r"authorization|cookie|email|guid|password|secret|session|token",
    re.IGNORECASE,
)
POINT_KEY = re.compile(r"point", re.IGNORECASE)
PRICE_KEY = re.compile(r"price|value", re.IGNORECASE)


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
            headers={
                "User-Agent": UA,
                "Accept": "application/json,text/plain,*/*",
                "Referer": "https://f1fantasytools.com/statistics",
            },
        )


def sanitised_preview(payload: Any, limit: int = 900) -> str:
    def sanitise(value: Any, depth: int = 0) -> Any:
        if depth > 4:
            return "[TRUNCATED]"
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in list(value.items())[:12]:
                result[str(key)] = (
                    "[REDACTED]"
                    if SENSITIVE_KEY.search(str(key))
                    else sanitise(item, depth + 1)
                )
            return result
        if isinstance(value, list):
            return [sanitise(item, depth + 1) for item in value[:2]]
        return value

    return json.dumps(sanitise(payload), ensure_ascii=False, separators=(",", ":"))[:limit]


def observed_years(payload: Any) -> list[int]:
    years: set[int] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in YEAR_KEYS:
                    try:
                        parsed = int(item)
                    except (TypeError, ValueError):
                        pass
                    else:
                        if 2000 <= parsed <= 2100:
                            years.add(parsed)
                visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                visit(item, depth + 1)

    visit(payload)
    return sorted(years)


def collect_keys(payload: Any, *, max_depth: int = 9) -> set[str]:
    keys: set[str] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > max_depth:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(str(key))
                visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value[:4]:
                visit(item, depth + 1)

    visit(payload)
    return keys


def first_dict_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def tools_summary(payload: Any, requested_year: int) -> dict[str, Any]:
    races_value = payload.get("races", []) if isinstance(payload, dict) else []
    races = first_dict_rows(races_value)
    season_result = payload.get("seasonResult", {}) if isinstance(payload, dict) else {}
    race_results = (
        season_result.get("raceResults", {})
        if isinstance(season_result, dict)
        else {}
    )
    if not races and isinstance(race_results, dict):
        races = first_dict_rows(race_results)

    recorded_rounds: set[int] = set()
    populated_rounds: set[int] = set()
    recorded_driver_counts: list[int] = []
    recorded_constructor_counts: list[int] = []
    all_driver_ids: set[str] = set()
    all_constructor_ids: set[str] = set()
    session_types: set[str] = set()
    point_keys: set[str] = set()
    price_keys: set[str] = set()
    race_rounds: set[int] = set()

    def inspect_asset(asset: dict[str, Any], destination: set[str]) -> None:
        asset_id = asset.get("id") or asset.get("assetId") or asset.get("abbreviation")
        if asset_id is not None:
            destination.add(str(asset_id))
        for key in collect_keys(asset, max_depth=7):
            if POINT_KEY.search(key):
                point_keys.add(key)
            if PRICE_KEY.search(key):
                price_keys.add(key)
        result = asset.get("raceResult")
        if isinstance(result, dict):
            session_types.update(str(key) for key in result)

    if isinstance(race_results, dict):
        for round_key, result in race_results.items():
            if not isinstance(result, dict):
                continue
            try:
                recorded_rounds.add(int(round_key))
            except (TypeError, ValueError):
                pass
            drivers = first_dict_rows(result.get("drivers", []))
            constructors = first_dict_rows(result.get("constructors", []))
            if drivers or constructors:
                try:
                    populated_rounds.add(int(round_key))
                except (TypeError, ValueError):
                    pass
            recorded_driver_counts.append(len(drivers))
            recorded_constructor_counts.append(len(constructors))

    containers: list[dict[str, Any]] = []
    if isinstance(race_results, dict):
        containers.extend(first_dict_rows(race_results))
    containers.extend(races)
    for race in containers:
        round_value = race.get("roundNumber") or race.get("round") or race.get("raceNumber")
        try:
            race_rounds.add(int(round_value))
        except (TypeError, ValueError):
            pass
        for key in ("drivers", "driverResults"):
            for asset in first_dict_rows(race.get(key, [])):
                inspect_asset(asset, all_driver_ids)
        for key in ("constructors", "constructorResults"):
            for asset in first_dict_rows(race.get(key, [])):
                inspect_asset(asset, all_constructor_ids)

    all_keys = collect_keys(payload)
    years = observed_years(payload)
    # Some responses identify their season only through their race timestamps and
    # the requested route.  The route year is considered verified only when the
    # payload exposes a matching year or every race date falls within that year.
    date_years: set[int] = set()
    for race in races:
        for key in ("date", "raceDate", "eventDate", "startDate"):
            value = race.get(key)
            if isinstance(value, str):
                match = re.match(r"(20\d{2})-", value)
                if match:
                    date_years.add(int(match.group(1)))
    verified_year = requested_year in years or date_years == {requested_year}

    return {
        "observed_years": years,
        "race_date_years": sorted(date_years),
        "requested_year_verified": verified_year,
        "scheduled_race_count": len(races),
        "recorded_race_count": len(recorded_rounds),
        "recorded_rounds": sorted(recorded_rounds),
        "populated_race_count": len(populated_rounds),
        "populated_rounds": sorted(populated_rounds),
        "race_count": len(race_rounds) or len(races),
        "rounds": sorted(race_rounds),
        "drivers_per_recorded_race_min": min(recorded_driver_counts, default=0),
        "drivers_per_recorded_race_max": max(recorded_driver_counts, default=0),
        "constructors_per_recorded_race_min": min(recorded_constructor_counts, default=0),
        "constructors_per_recorded_race_max": max(recorded_constructor_counts, default=0),
        "driver_entity_count": len(all_driver_ids),
        "constructor_entity_count": len(all_constructor_ids),
        "session_types": sorted(session_types),
        "point_fields": sorted(point_keys),
        "price_fields": sorted(price_keys),
        "has_race_by_race": bool(populated_rounds),
        "has_score_breakdown": len(point_keys) > 2,
        "has_prices": bool(price_keys),
        "all_keys_sample": sorted(all_keys)[:160],
    }


def official_summary(payload: Any) -> dict[str, Any]:
    all_keys = collect_keys(payload)
    return {
        "observed_years": observed_years(payload),
        "point_fields": sorted(key for key in all_keys if POINT_KEY.search(key)),
        "identifier_fields": sorted(
            key
            for key in all_keys
            if re.search(r"(?:^|_)(?:id|tourid|seasonid|championshipid)$", key, re.I)
            or key.lower() in {"playerid", "gamedayid", "meetingnumber", "tourid"}
        ),
        "endpoint_values": sorted(
            {
                str(value)
                for value in walk_scalars(payload)
                if isinstance(value, str) and ("feeds/" in value or ".json" in value)
            }
        )[:120],
        "all_keys_sample": sorted(all_keys)[:200],
    }


def walk_scalars(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_scalars(item)
    else:
        yield value


def request_record(
    getter: RateLimitedGetter,
    *,
    source: str,
    url: str,
    requested_year: int | None = None,
) -> dict[str, Any]:
    fetched_at = datetime.now(UTC).isoformat()
    try:
        response = getter.get(url)
    except requests.RequestException as exc:
        return {
            "source": source,
            "requested_year": requested_year,
            "url": url,
            "fetched_at_utc": fetched_at,
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload: Any = None
    valid_json = False
    try:
        payload = response.json()
        valid_json = True
    except ValueError:
        pass
    summary: dict[str, Any] = {}
    if valid_json:
        summary = (
            tools_summary(payload, requested_year)
            if source == "F1 Fantasy Tools" and requested_year is not None
            else official_summary(payload)
        )
    return {
        "source": source,
        "requested_year": requested_year,
        "url": url,
        "fetched_at_utc": fetched_at,
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "response_length": len(response.content),
        "valid_json": valid_json,
        "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
        "authentication_required": response.status_code in {401, 403},
        "sanitised_preview": sanitised_preview(payload) if valid_json else response.text[:900],
        "summary": summary,
    }


def main() -> None:
    getter = RateLimitedGetter()
    records: list[dict[str, Any]] = []
    for name, url in OFFICIAL_ENDPOINTS:
        records.append(request_record(getter, source=name, url=url))
        print(name, records[-1].get("http_status"), records[-1].get("response_length"))
    for year in TOOLS_YEARS:
        url = TOOLS_PATTERN.format(year=year)
        records.append(
            request_record(getter, source="F1 Fantasy Tools", url=url, requested_year=year)
        )
        summary = records[-1].get("summary", {})
        print(
            "F1 Fantasy Tools",
            year,
            records[-1].get("http_status"),
            summary.get("requested_year_verified"),
            summary.get("race_count"),
        )

    output = {
        "configuration": {
            "methods": ["GET"],
            "timeout_seconds": TIMEOUT_SECONDS,
            "minimum_request_interval_seconds": MIN_INTERVAL_SECONDS,
            "credentials_used": False,
            "raw_third_party_payloads_saved": False,
            "third_party_licence_note": (
                "No reuse licence was identified. Only schema and coverage summaries are retained; "
                "the site's published usage restrictions apply."
            ),
        },
        "results": records,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    for record in records:
        summary = record.get("summary", {})
        csv_rows.append(
            {
                "source": record["source"],
                "requested_year": record.get("requested_year"),
                "url": record["url"],
                "http_status": record.get("http_status"),
                "content_type": record.get("content_type"),
                "response_length": record.get("response_length"),
                "valid_json": record.get("valid_json"),
                "authentication_required": record.get("authentication_required"),
                "observed_years": json.dumps(summary.get("observed_years", [])),
                "race_date_years": json.dumps(summary.get("race_date_years", [])),
                "requested_year_verified": summary.get("requested_year_verified"),
                "race_count": summary.get("race_count"),
                "scheduled_race_count": summary.get("scheduled_race_count"),
                "recorded_race_count": summary.get("recorded_race_count"),
                "populated_race_count": summary.get("populated_race_count"),
                "driver_entity_count": summary.get("driver_entity_count"),
                "constructor_entity_count": summary.get("constructor_entity_count"),
                "has_race_by_race": summary.get("has_race_by_race"),
                "has_score_breakdown": summary.get("has_score_breakdown"),
                "has_prices": summary.get("has_prices"),
                "error": record.get("error", ""),
            }
        )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
