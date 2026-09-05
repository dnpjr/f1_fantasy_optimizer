"""Search public repository, dataset, and web-archive indexes cautiously.

Only GET requests are issued, at least one second apart, with a 15-second
timeout.  Results are index metadata and sanitised summaries; no dataset is
downloaded automatically.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode

import requests


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "public_archive_search_results.json"
TIMEOUT_SECONDS = 15
MIN_INTERVAL_SECONDS = 1.0
UA = "f1-fantasy-historical-data-research/1.0 (GET-only; no authentication)"

GITHUB_QUERIES = (
    '"F1 fantasy" data',
    '"Formula 1 fantasy" data',
    "f1-fantasy-data",
)
KAGGLE_QUERIES = ("f1 fantasy", "formula 1 fantasy")
CDX_QUERIES = (
    {
        "label": "official_playerstats_json",
        "url": "fantasy.formula1.com/feeds/popup/playerstats_*.json",
        "filter": ["statuscode:200"],
        "from": "2021",
        "to": "2025",
    },
    {
        "label": "legacy_game_period_scores",
        "url": "fantasy-api.formula1.com/*game_periods_scores*",
        "filter": ["statuscode:200"],
        "from": "2021",
        "to": "2025",
    },
    {
        "label": "official_frontend_javascript",
        "url": "fantasy.formula1.com/static-assets/build/static/js/*",
        "filter": ["statuscode:200", "mimetype:application/javascript"],
        "from": "2021",
        "to": "2025",
    },
)


class Getter:
    def __init__(self) -> None:
        self.last_started: float | None = None

    def get(self, url: str, *, params: Any = None) -> requests.Response:
        now = time.monotonic()
        if self.last_started is not None:
            remaining = MIN_INTERVAL_SECONDS - (now - self.last_started)
            if remaining > 0:
                time.sleep(remaining)
        self.last_started = time.monotonic()
        return requests.get(
            url,
            params=params,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"},
        )


def response_record(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        valid_json = True
    except ValueError:
        payload = None
        valid_json = False
    return {
        "url": response.url,
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "response_length": len(response.content),
        "valid_json": valid_json,
        "payload": payload,
        "text_preview": "" if valid_json else response.text[:600],
    }


def github_search(getter: Getter) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query in GITHUB_QUERIES:
        try:
            response = getter.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": 20},
            )
        except requests.RequestException as exc:
            records.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
            continue
        base = response_record(response)
        payload = base.pop("payload", None)
        items = payload.get("items", []) if isinstance(payload, dict) else []
        base.update(
            {
                "query": query,
                "total_count": payload.get("total_count") if isinstance(payload, dict) else None,
                "items": [
                    {
                        "full_name": item.get("full_name"),
                        "html_url": item.get("html_url"),
                        "description": item.get("description"),
                        "default_branch": item.get("default_branch"),
                        "license": (item.get("license") or {}).get("spdx_id"),
                        "updated_at": item.get("updated_at"),
                    }
                    for item in items
                ],
            }
        )
        records.append(base)
    return records


def kaggle_search(getter: Getter) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query in KAGGLE_QUERIES:
        try:
            response = getter.get(
                "https://www.kaggle.com/api/v1/datasets/list",
                params={"search": query, "sortBy": "relevance", "page": 1},
            )
        except requests.RequestException as exc:
            records.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
            continue
        base = response_record(response)
        payload = base.pop("payload", None)
        items = payload if isinstance(payload, list) else []
        base.update(
            {
                "query": query,
                "items": [
                    {
                        "ref": item.get("ref"),
                        "title": item.get("title"),
                        "subtitle": item.get("subtitle"),
                        "url": item.get("url"),
                        "license_name": item.get("licenseName"),
                        "last_updated": item.get("lastUpdated"),
                    }
                    for item in items[:50]
                ],
            }
        )
        records.append(base)
    return records


def cdx_search(getter: Getter) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query in CDX_QUERIES:
        params: list[tuple[str, str]] = [
            ("url", query["url"]),
            ("output", "json"),
            ("fl", "timestamp,original,statuscode,mimetype,digest"),
            ("collapse", "digest"),
            ("from", query["from"]),
            ("to", query["to"]),
            ("limit", "200"),
        ]
        params.extend(("filter", value) for value in query["filter"])
        try:
            response = getter.get("https://web.archive.org/cdx/search/cdx", params=params)
        except requests.RequestException as exc:
            records.append(
                {"label": query["label"], "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        base = response_record(response)
        payload = base.pop("payload", None)
        captures: list[dict[str, Any]] = []
        if isinstance(payload, list) and payload:
            headers = payload[0]
            if isinstance(headers, list):
                for row in payload[1:]:
                    if isinstance(row, list):
                        captures.append(dict(zip(headers, row)))
        base.update(
            {
                "label": query["label"],
                "query_url_pattern": query["url"],
                "capture_count": len(captures),
                "captures": captures,
            }
        )
        records.append(base)
    return records


def main() -> None:
    getter = Getter()
    output = {
        "configuration": {
            "methods": ["GET"],
            "timeout_seconds": TIMEOUT_SECONDS,
            "minimum_request_interval_seconds": MIN_INTERVAL_SECONDS,
            "credentials_used": False,
            "datasets_downloaded": False,
            "run_at_utc": datetime.now(UTC).isoformat(),
        },
        "github": github_search(getter),
        "kaggle": kaggle_search(getter),
        "internet_archive_cdx": cdx_search(getter),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    for source in ("github", "kaggle", "internet_archive_cdx"):
        for record in output[source]:
            count = record.get("capture_count", len(record.get("items", [])))
            print(source, record.get("query", record.get("label")), record.get("http_status"), count)


if __name__ == "__main__":
    main()
