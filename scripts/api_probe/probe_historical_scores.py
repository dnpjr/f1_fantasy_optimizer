"""GET-only probe for documented official F1 Fantasy historical-score routes.

Run from the repository root with:

    .venv/bin/python scripts/api_probe/probe_historical_scores.py

An optional bearer token may be supplied through F1_FANTASY_TOKEN. The token is
used only as an in-memory request header and is never logged or persisted.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests


OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUTPUT_DIR / "results.csv"
RESULTS_JSON = OUTPUT_DIR / "results.json"

SEASONS = tuple(range(2021, 2027))
TIMEOUT_SECONDS = 15
MIN_REQUEST_INTERVAL_SECONDS = 1.0
MAX_ACCESS_FAILURES_PER_FAMILY = 2

CURRENT_MARKET_URL = "https://fantasy.formula1.com/feeds/drivers/1_en.json"
CURRENT_PLAYERSTATS_PATTERN = (
    "https://fantasy.formula1.com/feeds/popup/playerstats_{player_id}.json"
)
API_BASES = {
    "partner": "https://fantasy-api.formula1.com/partner_games/f1",
    "legacy": "https://fantasy-api.formula1.com/f1",
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
SENSITIVE_KEY = re.compile(
    r"token|secret|password|authorization|cookie|session|guid|email",
    re.IGNORECASE,
)
SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "key",
    "session",
    "token",
}
AUTH_TEXT = re.compile(
    r"auth|bearer|credential|forbidden|http_x_f1_cookie_data|login|token|unauthor",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProbeResult:
    request_number: int
    requested_at_utc: str
    family: str
    purpose: str
    season: int | None
    asset_type: str | None
    player_id: int | None
    player_id_source: str | None
    game_period_id: int | None
    game_period_id_source: str | None
    url: str
    http_status: int | None
    content_type: str
    response_length: int
    valid_json: bool
    top_level_json_keys: list[str]
    response_preview: str
    authentication_appears_necessary: bool
    contains_expected_season: bool
    contains_scoring_information: bool
    classification: str
    diagnostic: str


def sanitise_url(url: str) -> str:
    parts = urlsplit(url)
    safe_query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        safe_query.append(
            (key, "[REDACTED]" if key.lower() in SENSITIVE_QUERY_KEYS else value)
        )
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(safe_query), parts.fragment)
    )


def sanitise_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value[:240]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= 4:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 12:
                clean["..."] = "[TRUNCATED]"
                break
            clean[str(key)] = (
                "[REDACTED]"
                if SENSITIVE_KEY.search(str(key))
                else sanitise_value(item, depth=depth + 1)
            )
        return clean
    if isinstance(value, list):
        return [sanitise_value(item, depth=depth + 1) for item in value[:3]]
    return value


def response_preview(payload: Any, response_text: str) -> str:
    if payload is not None:
        rendered = json.dumps(sanitise_value(payload), ensure_ascii=False, sort_keys=True)
    else:
        rendered = re.sub(r"\s+", " ", response_text).strip()
    return rendered[:500]


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from iter_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_dicts(item)


def recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    for row in iter_dicts(value):
        keys.update(str(key).lower() for key in row)
    return keys


def payload_contains_season(payload: Any, season: int | None) -> bool:
    if season is None or payload is None:
        return False
    target = str(season)
    season_keys = {"season", "season_name", "seasonname", "year"}
    for row in iter_dicts(payload):
        for key, value in row.items():
            if str(key).lower() in season_keys and str(value) == target:
                return True
    return False


def payload_contains_scores(payload: Any) -> bool:
    if payload is None:
        return False
    keys = recursive_keys(payload)
    exact = {
        "fantasy_points",
        "game_periods_scores",
        "gamedaypoints",
        "points",
        "qualifyingpoints",
        "racepoints",
        "score",
        "scores",
        "sprintpoints",
        "statswise",
    }
    return bool(keys & exact) or any(
        key.endswith("_score") or key.endswith("_points") for key in keys
    )


def classify_response(
    *,
    purpose: str,
    status: int | None,
    payload: Any,
    preview: str,
    expected_season: int | None,
) -> tuple[str, bool, bool, bool, str]:
    season_match = payload_contains_season(payload, expected_season)
    has_scores = payload_contains_scores(payload)
    auth_needed = status in {401, 403} or bool(AUTH_TEXT.search(preview))

    if auth_needed:
        return "C_authentication_required", season_match, has_scores, True, (
            "The response status or body indicates authentication/access control."
        )
    if status in {404, 410}:
        return "D_endpoint_not_found", season_match, has_scores, False, (
            "The server reported that this route does not exist."
        )
    if status is None:
        return "D_transport_or_endpoint_failure", season_match, has_scores, False, (
            "No HTTP response was received."
        )
    if status == 429:
        return "C_rate_limited", season_match, has_scores, False, (
            "The server rate-limited the request."
        )
    if 200 <= status < 300 and has_scores and season_match:
        return "A_historical_scores_confirmed", True, True, False, (
            "Inspected JSON contains scoring fields and the requested season."
        )
    if 200 <= status < 300 and payload is not None:
        if purpose in {"players_metadata", "game_period_metadata", "season_metadata"}:
            return "B_metadata_only", season_match, has_scores, False, (
                "The endpoint returned JSON metadata, without confirmed requested-season scores."
            )
        return "E_response_not_confirmed_as_historical_scores", season_match, has_scores, False, (
            "The response does not prove both requested-season identity and scoring content."
        )
    return "D_endpoint_unusable", season_match, has_scores, False, (
        f"HTTP {status} did not provide usable historical score data."
    )


def list_records(payload: Any, candidate_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    lower_map = {str(key).lower(): value for key, value in payload.items()}
    for key in candidate_keys:
        value = lower_map.get(key.lower())
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = list_records(value, candidate_keys)
            if nested:
                return nested
    for value in payload.values():
        nested = list_records(value, candidate_keys)
        if nested:
            return nested
    return []


def integer_from_keys(row: dict[str, Any], candidates: tuple[str, ...]) -> int | None:
    lower = {str(key).lower(): value for key, value in row.items()}
    for key in candidates:
        value = lower.get(key.lower())
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def asset_type_from_row(row: dict[str, Any]) -> str | None:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("PositionName", "position_name", "PlayerSkill", "player_skill", "type")
    ).lower()
    if "constructor" in text or text.strip() == "2":
        return "constructor"
    if "driver" in text or text.strip() == "1":
        return "driver"
    return None


def discover_assets(payload: Any) -> dict[str, int]:
    rows = list_records(payload, ("players", "value"))
    found: dict[str, int] = {}
    for row in rows:
        player_id = integer_from_keys(row, ("PlayerId", "player_id", "id"))
        asset_type = asset_type_from_row(row)
        if player_id is not None and asset_type and asset_type not in found:
            found[asset_type] = player_id
    return found


def discover_game_period_ids(payload: Any) -> list[int]:
    rows = list_records(
        payload,
        ("game_periods", "gameperiods", "periods", "fixtures", "gamedays"),
    )
    result: list[int] = []
    for row in rows:
        period_id = integer_from_keys(
            row,
            ("game_period_id", "gameperiodid", "gamePeriodId", "id"),
        )
        if period_id is not None and period_id not in result:
            result.append(period_id)
    return result


class ProbeClient:
    def __init__(self, token: str | None) -> None:
        self._token = token
        self._last_request_started: float | None = None
        self._request_number = 0
        self._access_failures: dict[str, int] = {}
        self.results: list[ProbeResult] = []
        self.payloads: dict[str, Any] = {}

    def family_blocked(self, family: str) -> bool:
        return self._access_failures.get(family, 0) >= MAX_ACCESS_FAILURES_PER_FAMILY

    def get(
        self,
        *,
        family: str,
        purpose: str,
        url: str,
        season: int | None = None,
        asset_type: str | None = None,
        player_id: int | None = None,
        player_id_source: str | None = None,
        game_period_id: int | None = None,
        game_period_id_source: str | None = None,
    ) -> tuple[ProbeResult | None, Any]:
        if self.family_blocked(family):
            return None, None

        now = time.monotonic()
        if self._last_request_started is not None:
            remaining = MIN_REQUEST_INTERVAL_SECONDS - (now - self._last_request_started)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started = time.monotonic()
        self._request_number += 1

        headers = {
            "Accept": "application/json,text/plain,*/*",
            "User-Agent": USER_AGENT,
        }
        if self._token and urlsplit(url).hostname == "fantasy-api.formula1.com":
            headers["Authorization"] = f"Bearer {self._token}"

        requested_at = datetime.now(UTC).isoformat()
        status: int | None = None
        content_type = ""
        body = ""
        payload: Any = None
        valid_json = False
        error = ""
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
            status = response.status_code
            content_type = response.headers.get("Content-Type", "")
            body = response.text
            try:
                payload = response.json()
                valid_json = True
            except (requests.JSONDecodeError, ValueError):
                payload = None
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"

        preview = response_preview(payload, body or error)
        classification, season_match, has_scores, auth_needed, diagnostic = classify_response(
            purpose=purpose,
            status=status,
            payload=payload,
            preview=preview,
            expected_season=season,
        )
        if error:
            diagnostic = f"{diagnostic} {error}"
        if status in {401, 403, 429}:
            self._access_failures[family] = self._access_failures.get(family, 0) + 1

        top_keys = sorted(str(key) for key in payload) if isinstance(payload, dict) else []
        result = ProbeResult(
            request_number=self._request_number,
            requested_at_utc=requested_at,
            family=family,
            purpose=purpose,
            season=season,
            asset_type=asset_type,
            player_id=player_id,
            player_id_source=player_id_source,
            game_period_id=game_period_id,
            game_period_id_source=game_period_id_source,
            url=sanitise_url(url),
            http_status=status,
            content_type=content_type,
            response_length=len(body.encode("utf-8")),
            valid_json=valid_json,
            top_level_json_keys=top_keys,
            response_preview=preview,
            authentication_appears_necessary=auth_needed,
            contains_expected_season=season_match,
            contains_scoring_information=has_scores,
            classification=classification,
            diagnostic=diagnostic,
        )
        self.results.append(result)
        self.payloads[result.url] = payload
        print(
            f"[{result.request_number:02d}] {status or '-':>3} "
            f"{classification} {result.url}"
        )
        return result, payload


def probe() -> tuple[list[ProbeResult], dict[str, Any]]:
    token = os.environ.get("F1_FANTASY_TOKEN") or None
    client = ProbeClient(token)
    discoveries: dict[str, Any] = {
        "token_supplied": bool(token),
        "assets_by_season": {},
        "game_period_ids_by_season": {},
        "current_playerstats_gameday_ids": [],
        "notes": [],
    }

    _, market_payload = client.get(
        family="current_public_market",
        purpose="players_metadata",
        url=CURRENT_MARKET_URL,
        season=2026,
    )
    current_assets = discover_assets(market_payload)
    discoveries["assets_by_season"]["2026_public_feed"] = current_assets

    for asset_type in ("driver", "constructor"):
        player_id = current_assets.get(asset_type)
        if player_id is None:
            continue
        _, stats_payload = client.get(
            family="current_public_playerstats",
            purpose="historical_scores",
            url=CURRENT_PLAYERSTATS_PATTERN.format(player_id=player_id),
            season=2026,
            asset_type=asset_type,
            player_id=player_id,
            player_id_source="current public drivers feed",
        )
        if isinstance(stats_payload, dict):
            value = stats_payload.get("Value", {})
            for row in value.get("GamedayWiseStats", []) if isinstance(value, dict) else []:
                period_id = integer_from_keys(row, ("GamedayId",))
                if period_id is not None and period_id not in discoveries["current_playerstats_gameday_ids"]:
                    discoveries["current_playerstats_gameday_ids"].append(period_id)

    # Discovery routes come directly from the documented 2022 API and its
    # partner_games base-path successor. Query each season without guessing IDs.
    for base_name, base_url in API_BASES.items():
        for season in SEASONS:
            _, players_payload = client.get(
                family=f"{base_name}_players",
                purpose="players_metadata",
                url=f"{base_url}/{season}/players?v=1",
                season=season,
            )
            assets = discover_assets(players_payload)
            if assets:
                discoveries["assets_by_season"][f"{base_name}_{season}"] = assets

            _, season_payload = client.get(
                family=f"{base_name}_season_metadata",
                purpose="season_metadata",
                url=f"{base_url}/{season}",
                season=season,
            )
            period_ids = discover_game_period_ids(season_payload)

            _, periods_payload = client.get(
                family=f"{base_name}_game_periods",
                purpose="game_period_metadata",
                url=f"{base_url}/{season}/game_periods",
                season=season,
            )
            for period_id in discover_game_period_ids(periods_payload):
                if period_id not in period_ids:
                    period_ids.append(period_id)
            if period_ids:
                discoveries["game_period_ids_by_season"][f"{base_name}_{season}"] = period_ids

    def assets_for(base_name: str, season: int) -> tuple[dict[str, int], str]:
        exact_key = f"{base_name}_{season}"
        if exact_key in discoveries["assets_by_season"]:
            return discoveries["assets_by_season"][exact_key], f"{base_name} {season} players endpoint"
        return current_assets, "current public drivers feed (current-season ID; historical stability unproven)"

    # Required route variants: a season in the path and the documented fixed
    # 2022 game namespace with season_name selecting the requested season.
    for base_name, base_url in API_BASES.items():
        for route_mode in ("season_path", "fixed_2022_path"):
            for season in SEASONS:
                assets, source = assets_for(base_name, season)
                path_season = season if route_mode == "season_path" else 2022
                for asset_type in ("driver", "constructor"):
                    player_id = assets.get(asset_type)
                    if player_id is None:
                        continue
                    client.get(
                        family=f"{base_name}_game_period_scores_{route_mode}_{asset_type}",
                        purpose="historical_scores",
                        url=(
                            f"{base_url}/{path_season}/players/{player_id}/"
                            f"game_periods_scores?season_name={season}"
                        ),
                        season=season,
                        asset_type=asset_type,
                        player_id=player_id,
                        player_id_source=source,
                    )

    # live_stats is tested only with an ID positively discovered from a legacy
    # API game-period metadata response. Current playerstats GamedayId values
    # are deliberately not assumed to be interchangeable.
    for base_name, base_url in API_BASES.items():
        for season in SEASONS:
            ids = discoveries["game_period_ids_by_season"].get(f"{base_name}_{season}", [])
            if not ids:
                continue
            game_period_id = ids[0]
            client.get(
                family=f"{base_name}_live_stats",
                purpose="live_scores",
                url=f"{base_url}/{season}/live_stats?game_period_id={game_period_id}",
                season=season,
                game_period_id=game_period_id,
                game_period_id_source=f"{base_name} {season} game-period metadata",
            )

    if discoveries["current_playerstats_gameday_ids"]:
        discoveries["notes"].append(
            "Public playerstats exposed GamedayId values, but the probe did not treat them as legacy game_period_id values."
        )
    return client.results, discoveries


def write_results(results: list[ProbeResult], discoveries: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    serialised = [asdict(item) for item in results]
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "probe_configuration": {
                    "seasons": list(SEASONS),
                    "timeout_seconds": TIMEOUT_SECONDS,
                    "minimum_request_interval_seconds": MIN_REQUEST_INTERVAL_SECONDS,
                    "max_access_failures_per_family": MAX_ACCESS_FAILURES_PER_FAMILY,
                    "http_methods": ["GET"],
                    "token_supplied": discoveries.get("token_supplied", False),
                },
                "discoveries": discoveries,
                "results": serialised,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    fieldnames = list(ProbeResult.__dataclass_fields__)
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in serialised:
            row = dict(row)
            row["top_level_json_keys"] = json.dumps(row["top_level_json_keys"])
            writer.writerow(row)


def main() -> None:
    results, discoveries = probe()
    write_results(results, discoveries)
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
