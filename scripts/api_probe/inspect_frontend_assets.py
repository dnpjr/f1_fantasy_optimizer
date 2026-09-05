"""Download and search public frontend assets without executing JavaScript."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests


HERE = Path(__file__).resolve().parent
ASSET_ROOT = HERE / "frontend_assets"
FINDINGS_PATH = HERE / "frontend_asset_findings.json"
TIMEOUT_SECONDS = 15
MIN_INTERVAL_SECONDS = 1.0
MAX_ASSET_BYTES = 12_000_000
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

OFFICIAL_PAGE = "https://fantasy.formula1.com/en/statistics/details"
OFFICIAL_SCRIPTS = (
    "https://fantasy.formula1.com/static-assets/build/static/js/main.cdb70fa6.js",
    "https://fantasy.formula1.com/static-assets/build/static/js/app.a2733cda.chunk.js",
    "https://fantasy.formula1.com/static-assets/build/static/js/statistics.dfbce992.chunk.js",
)
TOOLS_PAGE = "https://f1fantasytools.com/statistics"
SEARCH_TERMS = (
    "playerstats",
    "StatsWise",
    "gamedayId",
    "season_name",
    "seasonId",
    "championshipId",
    "statistics",
    "historical",
    "archive",
    "fantasy points",
    "supabase",
)
URL_RE = re.compile(r"https?://[^\"'\\\s<>]{8,}")
SOURCE_MAP_RE = re.compile(r"(?:\/\/[#@]|\/\*[#@])\s*sourceMappingURL=([^\s*]+)")


class ScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        values = dict(attrs)
        source = values.get("src")
        if source:
            self.scripts.append(source)


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
            headers={"User-Agent": UA, "Accept": "text/html,application/javascript,text/javascript,*/*"},
        )


def safe_name(url: str, fallback: str) -> str:
    name = Path(urlsplit(url).path).name or fallback
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:180]


def contexts(text: str, term: str, radius: int = 260, limit: int = 12) -> list[str]:
    found: list[str] = []
    lower = text.lower()
    needle = term.lower()
    start = 0
    while len(found) < limit:
        index = lower.find(needle, start)
        if index < 0:
            break
        snippet = text[max(0, index - radius) : min(len(text), index + len(term) + radius)]
        found.append(re.sub(r"\s+", " ", snippet))
        start = index + len(term)
    return found


def download(
    getter: RateLimitedGetter,
    *,
    source: str,
    url: str,
    directory: Path,
    fallback: str,
) -> tuple[dict[str, Any], str]:
    response = getter.get(url)
    body = response.content
    if len(body) > MAX_ASSET_BYTES:
        raise RuntimeError(f"Refusing oversized asset ({len(body)} bytes): {url}")
    name = safe_name(url, fallback)
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    text = response.text
    record = {
        "source": source,
        "url": url,
        "path": str(path.relative_to(HERE)),
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "licence": "No separate dataset/reuse licence identified; retained for endpoint and schema investigation only.",
    }
    return record, text


def main() -> None:
    getter = RateLimitedGetter()
    inventory: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    downloaded_urls: set[str] = set()

    targets: list[tuple[str, str, Path, str]] = [
        ("Official F1 Fantasy", OFFICIAL_PAGE, ASSET_ROOT / "official", "statistics.html"),
        *(
            ("Official F1 Fantasy", url, ASSET_ROOT / "official", "bundle.js")
            for url in OFFICIAL_SCRIPTS
        ),
        ("F1 Fantasy Tools", TOOLS_PAGE, ASSET_ROOT / "f1fantasytools", "statistics.html"),
    ]

    downloaded_text: list[tuple[str, str, str, Path]] = []
    for source, url, directory, fallback in targets:
        record, text = download(getter, source=source, url=url, directory=directory, fallback=fallback)
        inventory.append(record)
        downloaded_urls.add(url)
        downloaded_text.append((source, url, text, directory))
        print(record["http_status"], url, record["bytes"])

        if url == TOOLS_PAGE:
            parser = ScriptParser()
            parser.feed(text)
            for script in parser.scripts:
                absolute = urljoin(url, script)
                if urlsplit(absolute).hostname == "f1fantasytools.com":
                    targets.append((source, absolute, directory, "chunk.js"))

    index = len(downloaded_text)
    while index < len(targets):
        source, url, directory, fallback = targets[index]
        index += 1
        if url in downloaded_urls:
            continue
        record, text = download(getter, source=source, url=url, directory=directory, fallback=fallback)
        inventory.append(record)
        downloaded_urls.add(url)
        downloaded_text.append((source, url, text, directory))
        print(record["http_status"], url, record["bytes"])

    # Source maps are fetched only when a downloaded script genuinely names one.
    map_targets: list[tuple[str, str, Path, str]] = []
    for source, url, text, directory in downloaded_text:
        for match in SOURCE_MAP_RE.finditer(text):
            map_url = urljoin(url, match.group(1).strip())
            if urlsplit(map_url).hostname == urlsplit(url).hostname and map_url not in downloaded_urls:
                map_targets.append((source, map_url, directory, "bundle.js.map"))
    for source, url, directory, fallback in map_targets:
        if url in downloaded_urls:
            continue
        try:
            record, text = download(getter, source=source, url=url, directory=directory, fallback=fallback)
        except requests.RequestException as exc:
            inventory.append({"source": source, "url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue
        inventory.append(record)
        downloaded_urls.add(url)
        downloaded_text.append((source, url, text, directory))

    for source, url, text, _directory in downloaded_text:
        term_hits = {term: contexts(text, term) for term in SEARCH_TERMS}
        term_hits = {term: hits for term, hits in term_hits.items() if hits}
        urls = sorted({match.rstrip(").,;}") for match in URL_RE.findall(text)})[:200]
        if term_hits or urls:
            findings.append(
                {
                    "source": source,
                    "asset_url": url,
                    "term_contexts": term_hits,
                    "absolute_urls": urls,
                }
            )

    output = {
        "configuration": {
            "methods": ["GET"],
            "timeout_seconds": TIMEOUT_SECONDS,
            "minimum_request_interval_seconds": MIN_INTERVAL_SECONDS,
            "javascript_executed": False,
            "official_script_urls_from_live_browser_inventory": list(OFFICIAL_SCRIPTS),
            "f1fantasytools_script_urls_from_statistics_html": True,
        },
        "inventory": inventory,
        "findings": findings,
    }
    FINDINGS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ASSET_ROOT / "PROVENANCE.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
