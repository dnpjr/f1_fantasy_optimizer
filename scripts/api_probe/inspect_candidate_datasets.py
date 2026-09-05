"""Inspect metadata and file inventories for evidence-backed public candidates."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

import requests


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "candidate_dataset_inventories.json"
TIMEOUT_SECONDS = 15
MIN_INTERVAL_SECONDS = 1.0
UA = "f1-fantasy-historical-data-research/1.0 (GET-only; no authentication)"

GITHUB_CANDIDATES = (
    ("EduardoFAFernandes/F1FantasyData", "main"),
    ("JoshCBruce/fantasy-data", "main"),
    ("JoshCBruce/formula-fantasy", "main"),
    ("sajal147x/Formula_1_fantasy_analysis", "main"),
    ("jm1261/Fantasy-F1-League", "main"),
)
KAGGLE_CANDIDATES = ("prathamsharma123/formula-1-fantasy-2021",)
DATA_SUFFIXES = (
    ".csv",
    ".json",
    ".jsonl",
    ".parquet",
    ".xlsx",
    ".xls",
    ".zip",
    ".pickle",
    ".pkl",
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
        return requests.get(
            url,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"},
        )


def get_json(getter: Getter, url: str) -> tuple[dict[str, Any], Any]:
    try:
        response = getter.get(url)
    except requests.RequestException as exc:
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"}, None
    try:
        payload = response.json()
        valid_json = True
    except ValueError:
        payload = None
        valid_json = False
    return (
        {
            "url": response.url,
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "response_length": len(response.content),
            "valid_json": valid_json,
            "text_preview": "" if valid_json else response.text[:500],
        },
        payload,
    )


def github_inventories(getter: Getter) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for full_name, branch in GITHUB_CANDIDATES:
        url = f"https://api.github.com/repos/{full_name}/git/trees/{branch}?recursive=1"
        record, payload = get_json(getter, url)
        tree = payload.get("tree", []) if isinstance(payload, dict) else []
        data_files = [
            {
                "path": item.get("path"),
                "size": item.get("size"),
                "api_url": item.get("url"),
                "raw_url": f"https://raw.githubusercontent.com/{full_name}/{branch}/{item.get('path')}",
            }
            for item in tree
            if item.get("type") == "blob"
            and str(item.get("path", "")).lower().endswith(DATA_SUFFIXES)
        ]
        relevant_source_files = [
            item.get("path")
            for item in tree
            if item.get("type") == "blob"
            and any(
                term in str(item.get("path", "")).lower()
                for term in ("fantasy", "playerstats", "gameday", "point", "price", "score")
            )
        ][:150]
        record.update(
            {
                "repository": full_name,
                "branch": branch,
                "truncated": payload.get("truncated") if isinstance(payload, dict) else None,
                "data_files": data_files,
                "relevant_source_files": relevant_source_files,
            }
        )
        records.append(record)
        print("github", full_name, record.get("http_status"), len(data_files))
    return records


def kaggle_inventories(getter: Getter) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for reference in KAGGLE_CANDIDATES:
        url = f"https://www.kaggle.com/api/v1/datasets/view/{reference}"
        record, payload = get_json(getter, url)
        files = payload.get("resources", []) if isinstance(payload, dict) else []
        if not files and isinstance(payload, dict):
            files = payload.get("files", [])
        record.update(
            {
                "reference": reference,
                "title": payload.get("title") if isinstance(payload, dict) else None,
                "subtitle": payload.get("subtitle") if isinstance(payload, dict) else None,
                "description": payload.get("description") if isinstance(payload, dict) else None,
                "licence": (
                    payload.get("licenseName") or payload.get("license_name")
                    if isinstance(payload, dict)
                    else None
                ),
                "dataset_url": payload.get("url") if isinstance(payload, dict) else None,
                "files": [
                    (
                        {
                            "name": item.get("name") or item.get("ref"),
                            "size": item.get("totalBytes") or item.get("size"),
                            "url": item.get("url"),
                        }
                        if isinstance(item, dict)
                        else {"name": str(item), "size": None, "url": None}
                    )
                    for item in files
                ],
                "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
            }
        )
        records.append(record)
        print("kaggle", reference, record.get("http_status"), len(record["files"]))
    return records


def main() -> None:
    getter = Getter()
    output = {
        "configuration": {
            "methods": ["GET"],
            "timeout_seconds": TIMEOUT_SECONDS,
            "minimum_request_interval_seconds": MIN_INTERVAL_SECONDS,
            "credentials_used": False,
            "files_downloaded": False,
            "run_at_utc": datetime.now(UTC).isoformat(),
        },
        "github": github_inventories(getter),
        "kaggle": kaggle_inventories(getter),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
