"""Inspect a small allow-list of public files identified by index searches."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import io
import json
from pathlib import Path
import time
from typing import Any

import requests


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "candidate_file_findings.json"
TIMEOUT_SECONDS = 15
MIN_INTERVAL_SECONDS = 1.0
MAX_BYTES = 3_000_000
UA = "f1-fantasy-historical-data-research/1.0 (GET-only; no authentication)"

TARGETS = (
    ("jm1261/Fantasy-F1-League", "README.md", "https://raw.githubusercontent.com/jm1261/Fantasy-F1-League/main/README.md"),
    ("jm1261/Fantasy-F1-League", "LICENSE", "https://raw.githubusercontent.com/jm1261/Fantasy-F1-League/main/LICENSE"),
    ("jm1261/Fantasy-F1-League", "2021 driver points", "https://raw.githubusercontent.com/jm1261/Fantasy-F1-League/main/Data/2021/Lineup/Driver_Points.config"),
    ("jm1261/Fantasy-F1-League", "2021 constructor points", "https://raw.githubusercontent.com/jm1261/Fantasy-F1-League/main/Data/2021/Lineup/Team_Points.config"),
    ("jm1261/Fantasy-F1-League", "2022 driver points", "https://raw.githubusercontent.com/jm1261/Fantasy-F1-League/main/Data/2022/Lineup/Driver_Points.config"),
    ("jm1261/Fantasy-F1-League", "2022 constructor points", "https://raw.githubusercontent.com/jm1261/Fantasy-F1-League/main/Data/2022/Lineup/Team_Points.config"),
    ("JoshCBruce/fantasy-data", "README.md", "https://raw.githubusercontent.com/JoshCBruce/fantasy-data/main/README.md"),
    ("JoshCBruce/fantasy-data", "2025 driver sample", "https://raw.githubusercontent.com/JoshCBruce/fantasy-data/main/22-United%20States/driver_data/VER.json"),
    ("JoshCBruce/fantasy-data", "2025 constructor sample", "https://raw.githubusercontent.com/JoshCBruce/fantasy-data/main/22-United%20States/constructor_data/FER.json"),
    ("sajal147x/Formula_1_fantasy_analysis", "results.csv", "https://raw.githubusercontent.com/sajal147x/Formula_1_fantasy_analysis/main/src/data/rawdata/results.csv"),
    ("EduardoFAFernandes/F1FantasyData", "README.md", "https://raw.githubusercontent.com/EduardoFAFernandes/F1FantasyData/main/README.md"),
)


class Getter:
    def __init__(self) -> None:
        self.last_started: float | None = None

    def get(self, url: str) -> requests.Response:
        now = time.monotonic()
        if self.last_started is not None:
            remaining = MIN_INTERVAL_SECONDS - (now - self.last_started)
            if remaining > 0:
                time.sleep(remaining)
        self.last_started = time.monotonic()
        response = requests.get(
            url,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"},
        )
        if len(response.content) > MAX_BYTES:
            raise RuntimeError(f"Refusing oversized response: {url}")
        return response


def summarise_body(response: requests.Response) -> dict[str, Any]:
    text = response.text
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if payload is not None:
        return {
            "format": "json",
            "top_level_type": type(payload).__name__,
            "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
            "preview": json.dumps(payload, ensure_ascii=False)[:1600],
        }
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        rows = []
    csv_like = bool(rows and len(rows) > 1 and len(rows[0]) > 1)
    return {
        "format": "csv" if csv_like else "text",
        "line_count": len(text.splitlines()),
        "csv_row_count": len(rows) if csv_like else None,
        "csv_header": rows[0] if csv_like else None,
        "csv_sample": rows[1:4] if csv_like else None,
        "preview": text[:1600],
    }


def main() -> None:
    getter = Getter()
    records: list[dict[str, Any]] = []
    for repository, label, url in TARGETS:
        try:
            response = getter.get(url)
            record = {
                "repository": repository,
                "label": label,
                "url": response.url,
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "response_length": len(response.content),
                "summary": summarise_body(response),
            }
        except (requests.RequestException, RuntimeError) as exc:
            record = {
                "repository": repository,
                "label": label,
                "url": url,
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        print(repository, label, record.get("http_status"), record.get("response_length"))
    output = {
        "configuration": {
            "methods": ["GET"],
            "timeout_seconds": TIMEOUT_SECONDS,
            "minimum_request_interval_seconds": MIN_INTERVAL_SECONDS,
            "credentials_used": False,
            "raw_candidate_files_saved": False,
            "run_at_utc": datetime.now(UTC).isoformat(),
        },
        "results": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
