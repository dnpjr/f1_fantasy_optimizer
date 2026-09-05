"""Download explicitly licensed public datasets selected during the audit.

The allow-list is intentionally small.  Downloads are GET-only, use a
15-second timeout, and are retained with provenance and licence metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import requests


HERE = Path(__file__).resolve().parent
RAW_ROOT = HERE / "raw-data"
TIMEOUT_SECONDS = 15
MAX_BYTES = 50_000_000
MIN_INTERVAL_SECONDS = 1.0
UA = "f1-fantasy-historical-data-research/1.0 (GET-only; no authentication)"

DATASET = {
    "source": "Kaggle",
    "reference": "prathamsharma123/formula-1-fantasy-2021",
    "title": "Formula1 Fantasy 2021",
    "page": "https://www.kaggle.com/datasets/prathamsharma123/formula-1-fantasy-2021",
    "download": "https://www.kaggle.com/api/v1/datasets/download/prathamsharma123/formula-1-fantasy-2021",
    "licence": "CC0: Public Domain",
    "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
}

GITHUB_REPOSITORY = "jm1261/Fantasy-F1-League"
GITHUB_BRANCH = "main"
GITHUB_LICENCE = "MIT"
GITHUB_EVENT_FILES = {
    2023: (
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami",
        "Monaco", "Spain", "Canada", "Austria", "Great Britain", "Hungary",
        "Belgium", "Netherlands", "Italy", "Singapore", "Japan", "Qatar",
        "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi",
    ),
    2024: (
        "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
        "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria",
        "Great Britain", "Hungary", "Belgium", "Netherlands", "Italy",
        "Azerbaijan", "Singapore", "United States", "Mexico", "Brazil",
        "Las Vegas", "Qatar", "Abu Dhabi",
    ),
    2025: (
        "Australia", "China", "Japan", "Bahrain", "Saudi Arabia", "Miami",
        "Emilia Romagna", "Monaco", "Spain", "Canada", "Austria",
        "Great Britain", "Belgium", "Hungary", "Netherlands", "Italy",
        "Azerbaijan", "Singapore", "United States", "Mexico", "Brazil",
        "Las Vegas", "Qatar", "Abu Dhabi",
    ),
}

GITHUB_FILES = (
    "LICENSE",
    "README.md",
    "Data/2022/Lineup/Driver_Points.config",
    "Data/2022/Lineup/Individual_Driver_Points.config",
    "Data/2022/Lineup/Team_Points.config",
    "Data/2022/Lineup/Individual_Team_Points.config",
) + tuple(
    f"Data/{season}/Lineup/{event}_Results.json"
    for season, events in GITHUB_EVENT_FILES.items()
    for event in events
)


class Getter:
    def __init__(self) -> None:
        self.last_started: float | None = None

    def get(self, url: str, *, accept: str) -> requests.Response:
        now = time.monotonic()
        if self.last_started is not None:
            remaining = MIN_INTERVAL_SECONDS - (now - self.last_started)
            if remaining > 0:
                time.sleep(remaining)
        self.last_started = time.monotonic()
        return requests.get(
            url,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": UA, "Accept": accept},
        )


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> list[str]:
    extracted: list[str] = []
    resolved_destination = destination.resolve()
    for info in archive.infolist():
        target = (destination / info.filename).resolve()
        if target != resolved_destination and resolved_destination not in target.parents:
            raise RuntimeError(f"Unsafe archive member: {info.filename}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        extracted.append(str(target.relative_to(destination)))
    return extracted


def download_kaggle(getter: Getter) -> None:
    destination = RAW_ROOT / "kaggle_formula_1_fantasy_2021"
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "dataset.zip"

    response = getter.get(DATASET["download"], accept="application/zip,*/*")
    response.raise_for_status()
    if len(response.content) > MAX_BYTES:
        raise RuntimeError(f"Dataset exceeds {MAX_BYTES} bytes")
    archive_path.write_bytes(response.content)
    with zipfile.ZipFile(archive_path) as archive:
        extracted = safe_extract(archive, destination)

    provenance = {
        **DATASET,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "download_bytes": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "credentials_used": False,
        "methods": ["GET"],
        "extracted_files": extracted,
        "note": "Retained because the public Kaggle metadata identifies the dataset licence as CC0.",
    }
    (destination / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(destination)
    print("bytes", len(response.content))
    print("files", extracted)


def download_github(getter: Getter) -> None:
    destination = RAW_ROOT / "github_jm1261_fantasy_f1_league"
    destination.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    for path in GITHUB_FILES:
        url = (
            f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/"
            f"{GITHUB_BRANCH}/{path}"
        )
        response = getter.get(url, accept="application/json,text/plain,*/*")
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            raise RuntimeError(f"File exceeds {MAX_BYTES} bytes: {path}")
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        inventory.append(
            {
                "path": path,
                "url": url,
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "bytes": len(response.content),
                "sha256": hashlib.sha256(response.content).hexdigest(),
            }
        )
    provenance = {
        "source": "GitHub",
        "repository": GITHUB_REPOSITORY,
        "repository_url": f"https://github.com/{GITHUB_REPOSITORY}",
        "branch": GITHUB_BRANCH,
        "licence": GITHUB_LICENCE,
        "licence_source": f"https://github.com/{GITHUB_REPOSITORY}/blob/{GITHUB_BRANCH}/LICENSE",
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "credentials_used": False,
        "methods": ["GET"],
        "files": inventory,
        "note": (
            "The retained point files are third-party cumulative and individual-race "
            "Fantasy score records. They are not official API responses."
        ),
    }
    (destination / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(destination)
    print("files", [item["path"] for item in inventory])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-scores-only",
        action="store_true",
        help="download only the allow-listed 2022–2025 MIT score source",
    )
    args = parser.parse_args()
    getter = Getter()
    if not args.production_scores_only:
        download_kaggle(getter)
    download_github(getter)


if __name__ == "__main__":
    main()
