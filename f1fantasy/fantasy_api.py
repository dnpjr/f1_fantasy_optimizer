from __future__ import annotations
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import requests
import pandas as pd

BASE = "https://fantasy.formula1.com/feeds/drivers"
REQUEST_TIMEOUT_SECONDS = 12
MARKET_CACHE_TTL_SECONDS = 60 * 15
VERIFIED_MARKET_CACHE_VERSION = 2
SUPPORTED_VERIFIED_MARKET_CACHE_VERSIONS = frozenset({1, VERIFIED_MARKET_CACHE_VERSION})
VERIFIED_MARKET_CACHE_PATH = (
    Path(__file__).resolve().parents[1] / "data/cache/verified_fantasy_market.json"
)
MIN_CACHEABLE_DRIVER_ROWS = 15
MIN_CACHEABLE_CONSTRUCTOR_ROWS = 8

_LATEST_FEED_CACHE: dict[str, float | int] = {"round": 0, "ts": 0.0}
_MARKET_CACHE: dict[int, tuple[float, list[dict]]] = {}
_FEED_PROBE_ERRORS: dict[int, str] = {}


def _market_cache_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else VERIFIED_MARKET_CACHE_PATH


def validate_market_frames(
    players: pd.DataFrame,
    teams: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and defensively copy one usable official-market snapshot."""
    if not isinstance(players, pd.DataFrame) or not isinstance(teams, pd.DataFrame):
        raise ValueError("Market players and constructors must be dataframes.")
    driver_data = players.copy(deep=True)
    constructor_data = teams.copy(deep=True)
    requirements = (
        ("driver", driver_data, "playerId"),
        ("constructor", constructor_data, "teamId"),
    )
    for label, frame, id_column in requirements:
        missing = [column for column in (id_column, "name", "price") if column not in frame.columns]
        if missing:
            raise ValueError(f"Official {label} market data is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"Official {label} market data is empty.")
        identifiers = pd.to_numeric(frame[id_column], errors="coerce")
        prices = pd.to_numeric(frame["price"], errors="coerce")
        names = frame["name"].fillna("").astype(str).str.strip()
        if identifiers.isna().any() or identifiers.duplicated().any():
            raise ValueError(f"Official {label} market IDs are missing or duplicated.")
        if prices.isna().any() or prices.map(lambda value: not math.isfinite(float(value))).any():
            raise ValueError(f"Official {label} prices contain missing or non-finite values.")
        if prices.le(0).any():
            raise ValueError(f"Official {label} prices must be positive.")
        if names.eq("").any():
            raise ValueError(f"Official {label} names contain missing values.")
        frame[id_column] = identifiers.astype(int)
        frame["price"] = prices.astype(float)
        frame["name"] = names
        previous_prices = pd.to_numeric(
            frame.get("previous_price", pd.Series(index=frame.index, dtype=float)),
            errors="coerce",
        )
        official_changes = pd.to_numeric(
            frame.get("official_price_change", pd.Series(index=frame.index, dtype=float)),
            errors="coerce",
        )
        calculated_changes = prices - previous_prices
        conflicting_changes = (
            official_changes.notna()
            & calculated_changes.notna()
            & (official_changes - calculated_changes).abs().gt(1e-9)
        )
        if conflicting_changes.any():
            raise ValueError(f"Official {label} price changes disagree with current and previous prices.")
        frame["previous_price"] = previous_prices.astype(float)
        frame["official_price_change"] = calculated_changes.combine_first(official_changes).astype(float)
    return driver_data, constructor_data


def _validate_asset_ledger_frame(
    frame: pd.DataFrame,
    *,
    id_column: str,
    label: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"Official {label} asset ledger is empty.")
    ledger = frame.copy(deep=True)
    missing = [column for column in (id_column, "name", "price") if column not in ledger.columns]
    if missing:
        raise ValueError(f"Official {label} asset ledger is missing columns: {missing}")
    identifiers = pd.to_numeric(ledger[id_column], errors="coerce")
    prices = pd.to_numeric(ledger["price"], errors="coerce")
    names = ledger["name"].fillna("").astype(str).str.strip()
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError(f"Official {label} asset ledger IDs are missing or duplicated.")
    if prices.isna().any() or prices.map(lambda value: not math.isfinite(float(value))).any():
        raise ValueError(f"Official {label} asset ledger prices are missing or non-finite.")
    if prices.le(0).any() or names.eq("").any():
        raise ValueError(f"Official {label} asset ledger contains invalid names or prices.")
    ledger[id_column] = identifiers.astype(int)
    ledger["price"] = prices.astype(float)
    ledger["name"] = names
    return ledger


def _normalise_event_name(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _event_names_match(left: Any, right: Any) -> bool:
    left_key = _normalise_event_name(left).replace("grandprix", "")
    right_key = _normalise_event_name(right).replace("grandprix", "")
    return bool(left_key and right_key and (left_key in right_key or right_key in left_key))


def fetch_validated_current_gameday_market(
    seed_asset_ids: Iterable[int],
    *,
    expected_event_name: str | None = None,
    expected_season: int | None = None,
    playerstats_loader: Callable[[int], dict[str, Any]] | None = None,
    player_loader: Callable[..., pd.DataFrame] | None = None,
    team_loader: Callable[..., pd.DataFrame] | None = None,
    market_loader: Callable[..., dict[str, pd.DataFrame]] | None = None,
) -> dict[str, Any]:
    """Resolve and cross-check the official market for the active Fantasy gameday.

    A seed ID is used only to discover the active unplayed gameday. The market
    feed for that gameday defines the complete active roster, so replacement
    assets are not limited to the historical seed roster.
    """
    from f1fantasy.player_stats import fetch_player_stats, parse_player_race_points

    playerstats_loader = playerstats_loader or fetch_player_stats
    split_loaders_supplied = player_loader is not None or team_loader is not None
    player_loader = player_loader or fetch_players
    team_loader = team_loader or fetch_teams
    if market_loader is None and not split_loaders_supplied:
        market_loader = fetch_market_asset_ledgers
    candidate_gameday: int | None = None
    candidate_event_name: str | None = None
    expected_event_advanced = False
    seed_failures: list[str] = []

    for raw_id in dict.fromkeys(int(value) for value in seed_asset_ids):
        try:
            parsed = parse_player_race_points(playerstats_loader(raw_id), raw_id)
        except Exception as exc:
            seed_failures.append(f"{raw_id}: {exc}")
            continue
        if parsed.empty:
            continue
        candidates = parsed[
            (pd.to_numeric(parsed.get("is_played"), errors="coerce") == 0)
            & (pd.to_numeric(parsed.get("is_active"), errors="coerce") == 1)
            & pd.to_numeric(parsed.get("gameday_id"), errors="coerce").notna()
        ].copy()
        if expected_season is not None:
            candidates = candidates[
                pd.to_numeric(candidates.get("season"), errors="coerce") == int(expected_season)
            ]
        if expected_event_name and not candidates.empty:
            matching_expected = candidates[
                candidates.get("race_name", pd.Series(index=candidates.index, dtype=object)).map(
                    lambda value: _event_names_match(value, expected_event_name)
                )
            ]
            if not matching_expected.empty:
                candidates = matching_expected
            else:
                played_expected = parsed[
                    (pd.to_numeric(parsed.get("is_played"), errors="coerce") == 1)
                    & parsed.get(
                        "race_name", pd.Series(index=parsed.index, dtype=object)
                    ).map(lambda value: _event_names_match(value, expected_event_name))
                ].copy()
                if expected_season is not None:
                    played_expected = played_expected[
                        pd.to_numeric(played_expected.get("season"), errors="coerce")
                        == int(expected_season)
                    ]
                played_gamedays = pd.to_numeric(
                    played_expected.get("gameday_id"), errors="coerce"
                ).dropna()
                if played_gamedays.empty:
                    candidates = candidates.iloc[0:0]
                else:
                    candidates = candidates[
                        pd.to_numeric(candidates.get("gameday_id"), errors="coerce")
                        > int(played_gamedays.max())
                    ]
                    expected_event_advanced = not candidates.empty
        if candidates.empty:
            continue
        chosen = candidates.sort_values(["gameday_id", "round"], na_position="last").iloc[0]
        candidate_gameday = int(chosen["gameday_id"])
        candidate_event_name = str(chosen.get("race_name") or expected_event_name or "")
        break

    if candidate_gameday is None:
        detail = f" Seed failures: {'; '.join(seed_failures[:3])}" if seed_failures else ""
        raise RuntimeError(f"Could not discover a coherent active Fantasy gameday.{detail}")

    if market_loader is not None:
        market = market_loader(feed_round=candidate_gameday)
        player_assets = market["player_assets"].copy(deep=True)
        constructor_assets = market["constructor_assets"].copy(deep=True)
        players, teams = validate_market_frames(market["players"], market["teams"])
        asset_ledger_complete = True
    else:
        players, teams = validate_market_frames(
            player_loader(feed_round=candidate_gameday),
            team_loader(feed_round=candidate_gameday),
        )
        player_assets = players.copy(deep=True)
        constructor_assets = teams.copy(deep=True)
        asset_ledger_complete = False
    if len(players) < MIN_CACHEABLE_DRIVER_ROWS or len(teams) < MIN_CACHEABLE_CONSTRUCTOR_ROWS:
        raise ValueError("Active-gameday official market contains an incomplete roster.")
    if players["previous_price"].isna().any() or teams["previous_price"].isna().any():
        raise ValueError("Active-gameday official market is missing previous prices.")

    checked_assets = 0
    for frame, id_column, expected_type in (
        (players, "playerId", "driver"),
        (teams, "teamId", "constructor"),
    ):
        for row in frame.itertuples(index=False):
            asset_id = int(getattr(row, id_column))
            parsed = parse_player_race_points(playerstats_loader(asset_id), asset_id)
            matches = parsed[
                (pd.to_numeric(parsed.get("gameday_id"), errors="coerce") == candidate_gameday)
                & (pd.to_numeric(parsed.get("is_played"), errors="coerce") == 0)
                & (pd.to_numeric(parsed.get("is_active"), errors="coerce") == 1)
                & parsed.get("asset_type", pd.Series(index=parsed.index, dtype=object)).eq(expected_type)
            ].copy()
            if expected_season is not None:
                matches = matches[
                    pd.to_numeric(matches.get("season"), errors="coerce") == int(expected_season)
                ]
            if len(matches) != 1:
                raise ValueError(
                    f"Official {expected_type} {asset_id} does not have exactly one active "
                    f"unplayed row for gameday {candidate_gameday}."
                )
            observed = matches.iloc[0]
            if candidate_event_name and not _event_names_match(
                observed.get("race_name"), candidate_event_name
            ):
                raise ValueError(
                    f"Official {expected_type} {asset_id} does not match upcoming event "
                    f"{candidate_event_name}."
                )
            if not math.isclose(float(observed["price"]), float(row.price), abs_tol=1e-9):
                raise ValueError(f"Official {expected_type} {asset_id} current price disagrees with playerstats.")
            if not math.isclose(
                float(observed["old_price"]), float(row.previous_price), abs_tol=1e-9
            ):
                raise ValueError(f"Official {expected_type} {asset_id} previous price disagrees with playerstats.")
            checked_assets += 1

    return {
        "feed_round": candidate_gameday,
        "snapshot_name": candidate_event_name or expected_event_name,
        "players": players,
        "teams": teams,
        "player_assets": player_assets,
        "constructor_assets": constructor_assets,
        "asset_ledger_complete": asset_ledger_complete,
        "validated_asset_count": checked_assets,
        "requested_event_name": expected_event_name,
        "expected_event_advanced": bool(expected_event_advanced),
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = frame.astype(object).where(frame.notna(), None)
    return serializable.to_dict("records")


def market_content_signature(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    player_assets: pd.DataFrame | None = None,
    constructor_assets: pd.DataFrame | None = None,
) -> str:
    """Return an order-independent identity for one official market payload."""
    driver_data, constructor_data = validate_market_frames(players, teams)
    driver_ledger = _validate_asset_ledger_frame(
        player_assets if player_assets is not None else driver_data,
        id_column="playerId",
        label="driver",
    )
    constructor_ledger = _validate_asset_ledger_frame(
        constructor_assets if constructor_assets is not None else constructor_data,
        id_column="teamId",
        label="constructor",
    )

    def records(frame: pd.DataFrame, id_column: str) -> list[dict[str, Any]]:
        ordered = frame.sort_values(id_column, kind="stable").reset_index(drop=True)
        return _json_records(ordered.reindex(sorted(ordered.columns), axis=1))

    canonical = {
        "players": records(driver_data, "playerId"),
        "teams": records(constructor_data, "teamId"),
        "player_assets": records(driver_ledger, "playerId"),
        "constructor_assets": records(constructor_ledger, "teamId"),
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_verified_market_cache(
    feed_round: int,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    player_assets: pd.DataFrame | None = None,
    constructor_assets: pd.DataFrame | None = None,
    asset_ledger_complete: bool | None = None,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Atomically persist a full roster that was fetched from a verified feed."""
    driver_data, constructor_data = validate_market_frames(players, teams)
    driver_ledger = _validate_asset_ledger_frame(
        player_assets if player_assets is not None else driver_data,
        id_column="playerId",
        label="driver",
    )
    constructor_ledger = _validate_asset_ledger_frame(
        constructor_assets if constructor_assets is not None else constructor_data,
        id_column="teamId",
        label="constructor",
    )
    if (
        len(driver_data) < MIN_CACHEABLE_DRIVER_ROWS
        or len(constructor_data) < MIN_CACHEABLE_CONSTRUCTOR_ROWS
    ):
        return None
    cache_path = _market_cache_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": VERIFIED_MARKET_CACHE_VERSION,
        "feed_round": int(feed_round),
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "source_url": _feed_url(int(feed_round)),
        "players": _json_records(driver_data),
        "teams": _json_records(constructor_data),
        "player_assets": _json_records(driver_ledger),
        "constructor_assets": _json_records(constructor_ledger),
        "asset_ledger_complete": bool(
            asset_ledger_complete
            if asset_ledger_complete is not None
            else player_assets is not None and constructor_assets is not None
        ),
        "content_signature": market_content_signature(
            driver_data,
            constructor_data,
            player_assets=driver_ledger,
            constructor_assets=constructor_ledger,
        ),
    }
    temporary = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(cache_path)
    return payload


def load_verified_market_cache(
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and revalidate the newest previously verified official market cache."""
    cache_path = _market_cache_path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"No verified official market cache exists at {cache_path}.")
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Verified official market cache is unreadable: {exc}") from exc
    if payload.get("cache_version") not in SUPPORTED_VERIFIED_MARKET_CACHE_VERSIONS:
        raise ValueError("Verified official market cache has an unsupported version.")
    try:
        feed_round = int(payload["feed_round"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Verified official market cache has no valid feed round.") from exc
    if feed_round <= 0:
        raise ValueError("Verified official market cache has no valid feed round.")
    players, teams = validate_market_frames(
        pd.DataFrame(payload.get("players", [])),
        pd.DataFrame(payload.get("teams", [])),
    )
    if len(players) < MIN_CACHEABLE_DRIVER_ROWS or len(teams) < MIN_CACHEABLE_CONSTRUCTOR_ROWS:
        raise ValueError("Verified official market cache contains an incomplete roster.")
    player_assets = _validate_asset_ledger_frame(
        pd.DataFrame(payload.get("player_assets", payload.get("players", []))),
        id_column="playerId",
        label="driver",
    )
    constructor_assets = _validate_asset_ledger_frame(
        pd.DataFrame(payload.get("constructor_assets", payload.get("teams", []))),
        id_column="teamId",
        label="constructor",
    )
    content_signature = market_content_signature(
        players,
        teams,
        player_assets=player_assets,
        constructor_assets=constructor_assets,
    )
    recorded_signature = payload.get("content_signature")
    if recorded_signature and str(recorded_signature) != content_signature:
        raise ValueError("Verified official market cache content signature is invalid.")
    return {
        "feed_round": feed_round,
        "verified_at_utc": payload.get("verified_at_utc"),
        "source_url": payload.get("source_url") or _feed_url(feed_round),
        "players": players,
        "teams": teams,
        "player_assets": player_assets.copy(deep=True),
        "constructor_assets": constructor_assets.copy(deep=True),
        "asset_ledger_complete": bool(payload.get("asset_ledger_complete", False)),
        "content_signature": content_signature,
    }


def resolve_market_data(
    *,
    latest_feed_loader: Callable[[], int] | None = None,
    current_gameday_loader: Callable[[], dict[str, Any]] | None = None,
    player_loader: Callable[..., pd.DataFrame] | None = None,
    team_loader: Callable[..., pd.DataFrame] | None = None,
    market_loader: Callable[..., dict[str, pd.DataFrame]] | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve current prices from a validated official payload or verified cache."""
    latest_feed_loader = latest_feed_loader or _latest_feed_round
    split_loaders_supplied = player_loader is not None or team_loader is not None
    player_loader = player_loader or fetch_players
    team_loader = team_loader or fetch_teams
    if market_loader is None and not split_loaders_supplied:
        market_loader = fetch_market_asset_ledgers
    failures: list[str] = []
    cached: dict[str, Any] | None = None
    try:
        cached = load_verified_market_cache(path=cache_path)
    except Exception as exc:
        failures.append(f"verified cache unavailable: {exc}")

    try:
        feed_round = int(latest_feed_loader())
        if market_loader is not None:
            market = market_loader(feed_round=feed_round)
            players, teams = validate_market_frames(market["players"], market["teams"])
            player_assets = market["player_assets"].copy(deep=True)
            constructor_assets = market["constructor_assets"].copy(deep=True)
            asset_ledger_complete = True
        else:
            players, teams = validate_market_frames(
                player_loader(feed_round=feed_round),
                team_loader(feed_round=feed_round),
            )
            player_assets = players.copy(deep=True)
            constructor_assets = teams.copy(deep=True)
            asset_ledger_complete = False
        if cached is not None and int(cached["feed_round"]) > feed_round:
            raise RuntimeError(
                f"Discovered feed {feed_round} is older than verified cached feed "
                f"{int(cached['feed_round'])}."
            )
        content_signature = market_content_signature(
            players,
            teams,
            player_assets=player_assets,
            constructor_assets=constructor_assets,
        )
        content_changed = cached is None or cached.get("content_signature") != content_signature
        try:
            written = save_verified_market_cache(
                feed_round,
                players,
                teams,
                player_assets=player_assets,
                constructor_assets=constructor_assets,
                asset_ledger_complete=asset_ledger_complete,
                path=cache_path,
            )
        except Exception as cache_write_exc:
            written = None
            failures.append(f"verified cache write failed: {cache_write_exc}")
        return {
            "live_data_status": "fresh",
            "market_resolution_method": "latest_verified_feed",
            "feed_round": feed_round,
            "snapshot_round": None,
            "snapshot_name": None,
            "requested_event_name": None,
            "expected_event_advanced": False,
            "verified_at_utc": written.get("verified_at_utc") if written else None,
            "content_signature": content_signature,
            "content_changed": bool(content_changed),
            "players": players,
            "teams": teams,
            "player_assets": player_assets,
            "constructor_assets": constructor_assets,
            "asset_ledger_complete": asset_ledger_complete,
            "refresh_error": None,
            "fallback_failures": failures,
            "latest_probe_error": None,
        }
    except Exception as exc:
        latest_probe_error = f"fresh official feed unavailable: {exc}"
        failures.insert(0, latest_probe_error)

    if current_gameday_loader is not None:
        try:
            active_market = current_gameday_loader()
            active_feed_round = int(active_market["feed_round"])
            players, teams = validate_market_frames(
                active_market["players"], active_market["teams"]
            )
            player_assets = active_market.get("player_assets", players).copy(deep=True)
            constructor_assets = active_market.get("constructor_assets", teams).copy(deep=True)
            asset_ledger_complete = bool(active_market.get("asset_ledger_complete", False))
            if cached is not None and int(cached["feed_round"]) > active_feed_round:
                raise RuntimeError(
                    f"Validated active-gameday feed {active_feed_round} is older than verified "
                    f"cached feed {int(cached['feed_round'])}."
                )
            content_signature = market_content_signature(
                players,
                teams,
                player_assets=player_assets,
                constructor_assets=constructor_assets,
            )
            content_changed = cached is None or cached.get("content_signature") != content_signature
            try:
                written = save_verified_market_cache(
                    active_feed_round,
                    players,
                    teams,
                    player_assets=player_assets,
                    constructor_assets=constructor_assets,
                    asset_ledger_complete=asset_ledger_complete,
                    path=cache_path,
                )
            except Exception as cache_write_exc:
                written = None
                failures.append(f"verified cache write failed: {cache_write_exc}")
            return {
                "live_data_status": "fresh",
                "market_resolution_method": "active_gameday_verified",
                "feed_round": active_feed_round,
                "snapshot_round": None,
                "snapshot_name": active_market.get("snapshot_name"),
                "requested_event_name": active_market.get("requested_event_name"),
                "expected_event_advanced": bool(
                    active_market.get("expected_event_advanced", False)
                ),
                "verified_at_utc": written.get("verified_at_utc") if written else None,
                "content_signature": content_signature,
                "content_changed": bool(content_changed),
                "players": players,
                "teams": teams,
                "player_assets": player_assets,
                "constructor_assets": constructor_assets,
                "asset_ledger_complete": asset_ledger_complete,
                "refresh_error": None,
                "fallback_failures": failures,
                "latest_probe_error": latest_probe_error,
            }
        except Exception as exc:
            failures.append(f"active-gameday official market unavailable: {exc}")

    if cached is not None:
        return {
            "live_data_status": "cached",
            "market_resolution_method": "verified_cache",
            "feed_round": int(cached["feed_round"]),
            "snapshot_round": None,
            "snapshot_name": None,
            "requested_event_name": None,
            "expected_event_advanced": False,
            "verified_at_utc": cached.get("verified_at_utc"),
            "content_signature": cached.get("content_signature"),
            "content_changed": False,
            "players": cached["players"].copy(deep=True),
            "teams": cached["teams"].copy(deep=True),
            "player_assets": cached["player_assets"].copy(deep=True),
            "constructor_assets": cached["constructor_assets"].copy(deep=True),
            "asset_ledger_complete": bool(cached.get("asset_ledger_complete", False)),
            "refresh_error": failures[0],
            "fallback_failures": failures,
            "latest_probe_error": latest_probe_error,
        }
    raise RuntimeError("No safe current-season market data is available. " + " | ".join(failures))


def clear_market_cache() -> None:
    """Invalidate the in-process latest-round and market snapshot caches."""
    _LATEST_FEED_CACHE["round"] = 0
    _LATEST_FEED_CACHE["ts"] = 0.0
    _MARKET_CACHE.clear()
    _FEED_PROBE_ERRORS.clear()

def _feed_url(feed_round: int) -> str:
    return f"{BASE}/{feed_round}_en.json"


def _probe_feed_round(feed_round: int) -> str:
    """Return valid, missing, or failed without conflating transport failure with 404."""
    try:
        response = requests.get(
            _feed_url(feed_round),
            params={"buster": str(int(time.time()))},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code == 404:
            _FEED_PROBE_ERRORS.pop(int(feed_round), None)
            return "missing"
        if response.status_code != 200:
            content_type = response.headers.get("content-type", "unknown content type").split(";", 1)[0]
            _FEED_PROBE_ERRORS[int(feed_round)] = (
                f"HTTP {response.status_code} from {_feed_url(feed_round)} ({content_type})"
            )
            return "failed"
        try:
            payload = response.json()
        except Exception as exc:
            _FEED_PROBE_ERRORS[int(feed_round)] = f"Malformed JSON from {_feed_url(feed_round)}: {exc}"
            return "failed"
        if "Data" in payload and "Value" in payload["Data"]:
            _FEED_PROBE_ERRORS.pop(int(feed_round), None)
            return "valid"
        _FEED_PROBE_ERRORS[int(feed_round)] = (
            f"Unexpected JSON schema from {_feed_url(feed_round)}: missing Data.Value"
        )
        return "failed"
    except Exception as exc:
        _FEED_PROBE_ERRORS[int(feed_round)] = (
            f"Request failed for {_feed_url(feed_round)}: {type(exc).__name__}: {exc}"
        )
        return "failed"


def _valid_feed_round(feed_round: int) -> bool:
    """Compatibility boolean for callers that only need confirmed validity."""
    return _probe_feed_round(feed_round) == "valid"

def _latest_feed_round(max_search: int = 40) -> int:
    """
    Find the latest available fantasy feed number by probing upward until a feed
    stops existing, then return the highest valid one.
    """
    now = time.time()
    cached_round = int(_LATEST_FEED_CACHE.get("round", 0) or 0)
    cached_ts = float(_LATEST_FEED_CACHE.get("ts", 0.0) or 0.0)
    if cached_round > 0 and (now - cached_ts) < MARKET_CACHE_TTL_SECONDS:
        return cached_round

    low, high = 1, max_search
    last_ok = 0
    # Feed validity is monotonic over round number, so binary search is safe and
    # significantly faster than linear probing on cold startup.
    while low <= high:
        mid = (low + high) // 2
        probe = _probe_feed_round(mid)
        if probe == "failed":
            retry = _probe_feed_round(mid)
            if retry == "failed":
                detail = _FEED_PROBE_ERRORS.get(int(mid))
                raise RuntimeError(
                    f"Could not confirm fantasy feed {mid}; latest feed was not changed."
                    + (f" {detail}." if detail else "")
                )
            probe = retry
        if probe == "valid":
            last_ok = mid
            low = mid + 1
        else:
            high = mid - 1

    if last_ok <= 0:
        raise RuntimeError("Could not find any valid F1 Fantasy feed.")
    _LATEST_FEED_CACHE["round"] = int(last_ok)
    _LATEST_FEED_CACHE["ts"] = now
    return last_ok

def _get_market(feed_round: int | None = None) -> list[dict]:
    """
    Pull the latest market snapshot unless a specific feed_round is provided.
    """
    if feed_round is None:
        feed_round = _latest_feed_round()

    now = time.time()
    cached = _MARKET_CACHE.get(int(feed_round))
    if cached is not None and (now - cached[0]) < MARKET_CACHE_TTL_SECONDS:
        return cached[1]

    url = _feed_url(feed_round)
    params = {"buster": str(int(time.time()))}
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    j = r.json()
    value = j["Data"]["Value"]
    _MARKET_CACHE[int(feed_round)] = (now, value)
    return value


def _copy_first_available(
    frame: pd.DataFrame,
    target: str,
    candidates: Iterable[str],
) -> None:
    for candidate in candidates:
        if candidate in frame.columns:
            frame[target] = frame[candidate].copy()
            return


def _normalise_asset_ledger(
    rows: Iterable[dict[str, Any]],
    *,
    position_name: str,
    feed_round: int | None,
) -> pd.DataFrame:
    """Return a defensive, lossless official-asset ledger for one position."""
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty or "PositionName" not in frame.columns:
        return frame.copy(deep=True)
    frame = frame[frame["PositionName"].astype(str).str.upper() == position_name].copy()
    if frame.empty:
        return frame.reset_index(drop=True)

    id_target = "playerId" if position_name == "DRIVER" else "teamId"
    _copy_first_available(frame, id_target, ("PlayerId", id_target, "id"))
    _copy_first_available(frame, "name", ("FUllName", "FullName", "name"))
    _copy_first_available(frame, "price", ("Value", "CurrentValue", "price"))
    _copy_first_available(
        frame,
        "previous_price",
        ("OldPlayerValue", "OldValue", "previous_price"),
    )
    _copy_first_available(frame, "selected_pct", ("SelectedPercentage", "selected_pct"))
    _copy_first_available(frame, "tla", ("DriverTLA", "TLA", "tla"))
    _copy_first_available(frame, "position_name", ("PositionName", "position_name"))
    _copy_first_available(frame, "is_active", ("IsActive", "is_active"))
    _copy_first_available(frame, "status", ("Status", "PlayerStatus", "status"))
    _copy_first_available(frame, "is_removed", ("IsRemoved", "Removed", "is_removed"))
    _copy_first_available(frame, "is_deleted", ("IsDeleted", "Deleted", "is_deleted"))
    if position_name == "DRIVER":
        _copy_first_available(frame, "team_id", ("TeamId", "team_id"))
        _copy_first_available(frame, "team", ("TeamName", "team"))
        _copy_first_available(
            frame,
            "captain_selected_pct",
            ("CaptainSelectedPercentage", "captain_selected_pct"),
        )
        _copy_first_available(
            frame,
            "driver_reference",
            ("DriverReference", "driver_reference"),
        )
        _copy_first_available(frame, "f1_player_id", ("F1PlayerId", "f1_player_id"))
    else:
        _copy_first_available(frame, "f1_team_id", ("F1PlayerId", "f1_team_id"))

    if id_target in frame.columns:
        frame[id_target] = pd.to_numeric(frame[id_target], errors="raise").astype(int)
    if "price" in frame.columns:
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce").astype(float)
    if "previous_price" in frame.columns:
        frame["previous_price"] = pd.to_numeric(frame["previous_price"], errors="coerce").astype(float)
        frame["official_price_change"] = frame["price"] - frame["previous_price"]
    else:
        frame["previous_price"] = pd.Series(float("nan"), index=frame.index, dtype=float)
        frame["official_price_change"] = pd.Series(float("nan"), index=frame.index, dtype=float)
    if "team_id" in frame.columns:
        frame["team_id"] = pd.to_numeric(frame["team_id"], errors="coerce").astype("Int64")
    if "f1_player_id" in frame.columns:
        frame["f1_player_id"] = pd.to_numeric(frame["f1_player_id"], errors="coerce").astype("Int64")
    if "f1_team_id" in frame.columns:
        frame["f1_team_id"] = pd.to_numeric(frame["f1_team_id"], errors="coerce").astype("Int64")

    active_values = pd.to_numeric(
        frame.get("is_active", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    frame["is_active"] = active_values
    frame["selectable"] = active_values.eq(1)
    frame["feed_round"] = int(feed_round) if feed_round is not None else pd.NA
    _copy_first_available(frame, "gameday_id", ("GamedayId", "FeedId", "feed_round"))
    return frame.reset_index(drop=True)


def normalise_player_asset_ledger(
    rows: Iterable[dict[str, Any]],
    *,
    feed_round: int | None = None,
) -> pd.DataFrame:
    """Normalise all official driver rows without discarding inactive assets."""
    return _normalise_asset_ledger(rows, position_name="DRIVER", feed_round=feed_round)


def normalise_constructor_asset_ledger(
    rows: Iterable[dict[str, Any]],
    *,
    feed_round: int | None = None,
) -> pd.DataFrame:
    """Normalise all official constructor rows without discarding inactive assets."""
    return _normalise_asset_ledger(rows, position_name="CONSTRUCTOR", feed_round=feed_round)


def selectable_player_assets(player_assets: pd.DataFrame) -> pd.DataFrame:
    """Derive the existing active-only driver market from a full ledger."""
    ledger = player_assets.copy(deep=True)
    active = pd.to_numeric(
        ledger.get("is_active", ledger.get("IsActive", pd.Series(1, index=ledger.index))),
        errors="coerce",
    ).fillna(0).eq(1)
    frame = ledger.loc[active].copy()
    columns = [
        "playerId", "name", "price", "previous_price", "official_price_change", "team",
        "selected_pct", "captain_selected_pct", "driver_reference", "tla", "f1_player_id",
    ]
    return frame[[column for column in columns if column in frame.columns]].reset_index(drop=True)


def holding_valid_player_assets(player_assets: pd.DataFrame) -> pd.DataFrame:
    """Expose the full official ledger without inferring future holding rules."""
    return player_assets.copy(deep=True).reset_index(drop=True)


def holding_valid_constructor_assets(constructor_assets: pd.DataFrame) -> pd.DataFrame:
    """Expose the full constructor ledger for exact-ID holding validation."""
    return constructor_assets.copy(deep=True).reset_index(drop=True)


def price_view_player_assets(player_assets: pd.DataFrame) -> pd.DataFrame:
    """Expose priced official assets without equating visibility to selectability.

    This interface intentionally makes no claim about future price-movement
    eligibility; that policy remains a later milestone.
    """
    ledger = player_assets.copy(deep=True)
    prices = pd.to_numeric(
        ledger.get("price", ledger.get("Value", pd.Series(index=ledger.index, dtype=float))),
        errors="coerce",
    )
    return ledger.loc[prices.notna()].copy(deep=True).reset_index(drop=True)


def price_view_constructor_assets(constructor_assets: pd.DataFrame) -> pd.DataFrame:
    """Expose priced official constructors independently of selectability."""
    ledger = constructor_assets.copy(deep=True)
    prices = pd.to_numeric(
        ledger.get("price", ledger.get("Value", pd.Series(index=ledger.index, dtype=float))),
        errors="coerce",
    )
    return ledger.loc[prices.notna()].copy(deep=True).reset_index(drop=True)


def selectable_constructor_assets(constructor_assets: pd.DataFrame) -> pd.DataFrame:
    ledger = constructor_assets.copy(deep=True)
    active = pd.to_numeric(
        ledger.get("is_active", ledger.get("IsActive", pd.Series(1, index=ledger.index))),
        errors="coerce",
    ).fillna(0).eq(1)
    frame = ledger.loc[active].copy()
    columns = [
        "teamId", "name", "price", "previous_price", "official_price_change",
        "selected_pct", "tla", "f1_team_id",
    ]
    return frame[[column for column in columns if column in frame.columns]].reset_index(drop=True)


def fetch_market_asset_ledgers(
    year: int | None = None,
    feed_round: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch one payload and derive both full ledgers and selectable markets."""
    del year
    resolved_round = int(feed_round) if feed_round is not None else _latest_feed_round()
    rows = _get_market(feed_round=resolved_round)
    player_assets = normalise_player_asset_ledger(rows, feed_round=resolved_round)
    constructor_assets = normalise_constructor_asset_ledger(rows, feed_round=resolved_round)
    return {
        "player_assets": player_assets,
        "constructor_assets": constructor_assets,
        "players": selectable_player_assets(player_assets),
        "teams": selectable_constructor_assets(constructor_assets),
    }


def fetch_player_asset_ledger(
    year: int | None = None,
    feed_round: int | None = None,
) -> pd.DataFrame:
    return fetch_market_asset_ledgers(year=year, feed_round=feed_round)["player_assets"]


def fetch_constructor_asset_ledger(
    year: int | None = None,
    feed_round: int | None = None,
) -> pd.DataFrame:
    return fetch_market_asset_ledgers(year=year, feed_round=feed_round)["constructor_assets"]


def fetch_players(year: int | None = None, feed_round: int | None = None) -> pd.DataFrame:
    return selectable_player_assets(fetch_player_asset_ledger(year=year, feed_round=feed_round))

def fetch_teams(year: int | None = None, feed_round: int | None = None) -> pd.DataFrame:
    return selectable_constructor_assets(
        fetch_constructor_asset_ledger(year=year, feed_round=feed_round)
    )

def debug_feed_info() -> None:
    latest = _latest_feed_round()
    print(f"Latest available feed round: {latest}")
    print(f"Feed URL: {_feed_url(latest)}")
