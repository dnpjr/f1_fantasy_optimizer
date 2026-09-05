from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
import json
import os
import re
import unicodedata
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import pandas as pd
import requests

from f1fantasy.historical_scores import DRIVER_TLA_TO_ID
from f1fantasy.weekend_state import (
    DEFAULT_EXPECTED_PARTICIPANTS,
    EventKey,
    SessionKind,
    SessionState,
    SessionStatus,
    WeekendFormat,
    classify_session_dataframe,
    coerce_utc_datetime,
    scheduled_session_timestamp,
)


FORMULA1_SOURCE = "formula1"
FORMULA1_API_BASE = "https://api.formula1.com/v2/fom-results"
# Formula1.com publishes this browser key in NEXT_PUBLIC_GLOBAL_EVENTTRACKER_APIKEY.
# It is not an account credential; an environment override permits key rotation.
FORMULA1_RESULTS_API_KEY = os.getenv(
    "FORMULA1_RESULTS_API_KEY", "v1JVGPgXlahatAqwhakbrGtFdxW5rQBz"
)
JOLPICA_ALPHA_SOURCE = "jolpica_alpha_results"
JOLPICA_FALLBACK_SOURCE = "jolpica_alpha_results_fallback"
JOLPICA_ALPHA_BASE = "https://api.jolpi.ca/f1/alpha"
SESSION_SOURCE_TIMEOUT_SECONDS = 15
CURRENT_EVENT_NONFINAL_TTL = timedelta(minutes=2)
CURRENT_SEASON_SCHEDULE_TTL = timedelta(hours=6)
MINIMUM_SESSION_AGE_FOR_FINAL = timedelta(minutes=60)

SESSION_RESULT_COLUMNS = [
    "season",
    "round",
    "session_kind",
    "human_driver_id",
    "driver_reference",
    "source_driver_id",
    "abbreviation",
    "display_name",
    "team",
    "team_reference",
    "position",
    "classification",
    "status",
    "source",
    "session_timestamp",
    "source_fetched_at",
    "laps",
    "time_gap",
    "source_finality",
    "identity_match_method",
    "identity_match_status",
    "identity_diagnostic",
    "is_classified",
    "session_components",
]

_SOURCE_CODE_BY_KIND = {
    SessionKind.PRACTICE_1: "FP1",
    SessionKind.PRACTICE_2: "FP2",
    SessionKind.PRACTICE_3: "FP3",
    SessionKind.SPRINT_QUALIFYING: "SQ",
}

_FORMULA1_PAYLOAD_KEY_BY_KIND = {
    SessionKind.PRACTICE_1: "raceResultsPractice1",
    SessionKind.PRACTICE_2: "raceResultsPractice2",
    SessionKind.PRACTICE_3: "raceResultsPractice3",
    SessionKind.SPRINT_QUALIFYING: "raceResultsSprintShootout",
}

_FORMULA1_SESSION_CODE_BY_KIND = {
    SessionKind.PRACTICE_1: "p1",
    SessionKind.PRACTICE_2: "p2",
    SessionKind.PRACTICE_3: "p3",
    SessionKind.SPRINT_QUALIFYING: "ss",
}

_FORMULA1_DATASET_TYPE_BY_KIND = {
    SessionKind.PRACTICE_1: "Practice1",
    SessionKind.PRACTICE_2: "Practice2",
    SessionKind.PRACTICE_3: "Practice3",
    SessionKind.SPRINT_QUALIFYING: "Sprint Shootout",
}


@dataclass(frozen=True)
class LiveSessionIngestion:
    results: pd.DataFrame
    states: tuple[SessionState, ...]
    diagnostics: dict[str, Any]


@dataclass
class _JsonCacheEntry:
    payload: Any
    fetched_at: datetime
    authoritative: bool = False


_JSON_CACHE: dict[str, _JsonCacheEntry] = {}


def empty_session_results() -> pd.DataFrame:
    return pd.DataFrame(columns=SESSION_RESULT_COLUMNS)


def expected_live_session_kinds(format: WeekendFormat) -> tuple[SessionKind, ...]:
    if format == WeekendFormat.SPRINT:
        return (SessionKind.PRACTICE_1, SessionKind.SPRINT_QUALIFYING)
    return (
        SessionKind.PRACTICE_1,
        SessionKind.PRACTICE_2,
        SessionKind.PRACTICE_3,
    )


def clear_live_session_cache() -> None:
    _JSON_CACHE.clear()


def _request_json(url: str) -> Any:
    response = requests.get(
        url,
        headers={"User-Agent": "F1FantasyOptimizer/1.0 requests"},
        timeout=SESSION_SOURCE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _request_formula1_json(url: str) -> Any:
    response = requests.get(
        url,
        headers={
            "User-Agent": "F1FantasyOptimizer/1.0 requests",
            "Accept": "application/json",
            "Origin": "https://www.formula1.com",
            "Referer": "https://www.formula1.com/",
            "apikey": FORMULA1_RESULTS_API_KEY,
        },
        timeout=SESSION_SOURCE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _cached_json(
    url: str,
    *,
    force_refresh: bool,
    effective_time: datetime,
    loader: Callable[[str], Any],
    ttl: timedelta,
) -> tuple[Any, datetime, bool]:
    cached = _JSON_CACHE.get(url)
    if cached is not None and not force_refresh:
        age = effective_time - cached.fetched_at
        if cached.authoritative or timedelta(0) <= age <= ttl:
            return deepcopy(cached.payload), cached.fetched_at, True
    payload = loader(url)
    fetched_at = effective_time
    _JSON_CACHE[url] = _JsonCacheEntry(deepcopy(payload), fetched_at)
    return deepcopy(payload), fetched_at, False


def _mark_authoritative(url: str) -> None:
    cached = _JSON_CACHE.get(url)
    if cached is not None:
        cached.authoritative = True


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _identity_candidates(
    driver: Mapping[str, Any],
    *,
    history: pd.DataFrame | None,
    player_identity_map: pd.DataFrame | None,
) -> tuple[str | None, str, str, str | None]:
    abbreviation = str(driver.get("abbreviation") or "").strip().upper()
    display_name = " ".join(
        part
        for part in (
            str(driver.get("given_name") or "").strip(),
            str(driver.get("family_name") or "").strip(),
        )
        if part
    )
    candidates: dict[str, set[str]] = {}
    mapped_tla = DRIVER_TLA_TO_ID.get(abbreviation)
    if mapped_tla:
        candidates.setdefault(mapped_tla, set()).add("tla")

    identity = (
        player_identity_map.copy(deep=True)
        if isinstance(player_identity_map, pd.DataFrame)
        else pd.DataFrame()
    )
    if not identity.empty and "human_driver_id" in identity.columns:
        tla_matches = identity[
            identity.get("tla", pd.Series(index=identity.index, dtype=object))
            .fillna("")
            .astype(str)
            .str.upper()
            .eq(abbreviation)
        ]
        for value in tla_matches["human_driver_id"].dropna().astype(str):
            candidates.setdefault(value, set()).add("fantasy_identity_tla")
        name_key = _normalise_text(display_name)
        if name_key and "display_name" in identity.columns:
            name_matches = identity[
                identity["display_name"].map(_normalise_text).eq(name_key)
            ]
            for value in name_matches["human_driver_id"].dropna().astype(str):
                candidates.setdefault(value, set()).add("fantasy_identity_name")

    history_frame = history.copy(deep=True) if isinstance(history, pd.DataFrame) else pd.DataFrame()
    if not history_frame.empty and "driverId" in history_frame.columns:
        for column in ("code", "tla", "abbreviation"):
            if column in history_frame.columns and abbreviation:
                matches = history_frame[
                    history_frame[column].fillna("").astype(str).str.upper().eq(abbreviation)
                ]
                for value in matches["driverId"].dropna().astype(str):
                    candidates.setdefault(value, set()).add("history_tla")
        name_key = _normalise_text(display_name)
        for column in ("driver", "name"):
            if column in history_frame.columns and name_key:
                matches = history_frame[history_frame[column].map(_normalise_text).eq(name_key)]
                for value in matches["driverId"].dropna().astype(str):
                    candidates.setdefault(value, set()).add("history_name")

    if len(candidates) == 1:
        human_id, methods = next(iter(candidates.items()))
        preferred = next(
            (
                method
                for method in ("tla", "fantasy_identity_tla", "history_tla", "fantasy_identity_name", "history_name")
                if method in methods
            ),
            sorted(methods)[0],
        )
        return human_id, preferred, "matched", None
    if len(candidates) > 1:
        return (
            None,
            "conflicting_identity_evidence",
            "ambiguous",
            "Conflicting human-driver candidates: " + ", ".join(sorted(candidates)) + ".",
        )
    return (
        None,
        "unresolved",
        "unresolved",
        f"No deterministic human identity matched {abbreviation or display_name or 'source driver'}.",
    )


def _source_finality(payload: Mapping[str, Any]) -> str:
    for key in ("finality", "status", "classification_status"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if payload.get("is_final") is True:
        return "final"
    if payload.get("is_final") is False:
        return "provisional"
    return "not_explicitly_provided"


def parse_jolpica_session_payload(
    payload: Any,
    *,
    event: EventKey,
    kind: SessionKind,
    history: pd.DataFrame | None = None,
    player_identity_map: pd.DataFrame | None = None,
    source: str = JOLPICA_ALPHA_SOURCE,
) -> pd.DataFrame:
    """Normalize one Jolpica alpha classification without mutating the payload."""
    value = deepcopy(payload)
    if not isinstance(value, Mapping) or not isinstance(value.get("data"), Mapping):
        raise ValueError("Session payload is missing a data object.")
    data = value["data"]
    source_code = str(data.get("code") or "").strip().upper()
    if source_code != _SOURCE_CODE_BY_KIND.get(kind):
        raise ValueError(
            f"Session payload code {source_code or 'missing'} does not match {kind.value}."
        )
    season = data.get("season")
    round_data = data.get("round")
    if not isinstance(season, Mapping) or not isinstance(round_data, Mapping):
        raise ValueError("Session payload is missing season or round identity.")
    if int(season.get("year", 0) or 0) != event.season or int(round_data.get("number", 0) or 0) != event.round:
        raise ValueError("Session payload belongs to a different season or round.")
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Session payload results are not a list.")

    metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}
    fetched_at = str(metadata.get("timestamp") or "").strip() or None
    finality = _source_finality(data)
    finality_status = (
        "provisional"
        if re.search(r"provisional|live|running|in progress", finality, re.IGNORECASE)
        else ""
    )
    rows: list[dict[str, Any]] = []
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            raise ValueError("Session result contains a non-object row.")
        driver = raw.get("driver")
        team = raw.get("team")
        if not isinstance(driver, Mapping) or not isinstance(team, Mapping):
            raise ValueError("Session result is missing driver or team identity.")
        human_id, match_method, match_status, identity_diagnostic = _identity_candidates(
            driver,
            history=history,
            player_identity_map=player_identity_map,
        )
        given_name = str(driver.get("given_name") or "").strip()
        family_name = str(driver.get("family_name") or "").strip()
        display_name = " ".join(part for part in (given_name, family_name) if part)
        position = pd.to_numeric(raw.get("position"), errors="coerce")
        is_classified = raw.get("is_classified")
        position_text = str(raw.get("position_text") or "").strip()
        classification = (
            "classified"
            if is_classified is True
            else position_text or "unclassified"
        )
        rows.append(
            {
                "season": event.season,
                "round": event.round,
                "session_kind": kind.value,
                "human_driver_id": human_id,
                "driver_reference": driver.get("id"),
                "source_driver_id": driver.get("id"),
                "abbreviation": str(driver.get("abbreviation") or "").strip().upper(),
                "display_name": display_name,
                "team": str(team.get("name") or "").strip(),
                "team_reference": team.get("id"),
                "position": None if pd.isna(position) else int(position),
                "classification": classification,
                "status": finality_status or classification,
                "source": source,
                "session_timestamp": data.get("timestamp"),
                "source_fetched_at": fetched_at,
                "laps": raw.get("laps"),
                "time_gap": raw.get("time", raw.get("gap_to_leader")),
                "source_finality": finality,
                "identity_match_method": match_method,
                "identity_match_status": match_status,
                "identity_diagnostic": identity_diagnostic,
                "is_classified": is_classified,
                "session_components": json.dumps(raw.get("components") or {}, sort_keys=True),
            }
        )
    frame = pd.DataFrame(rows, columns=SESSION_RESULT_COLUMNS)
    if not frame.empty:
        frame["position"] = pd.array(frame["position"], dtype="Int64")
    return frame


def _formula1_session_timestamp(session: Mapping[str, Any]) -> str | None:
    start = str(session.get("startTime") or "").strip()
    if not start:
        return None
    offset = str(session.get("gmtOffset") or "").strip()
    if re.fullmatch(r"[+-]\d{2}:\d{2}", offset) and not re.search(
        r"(?:Z|[+-]\d{2}:\d{2})$", start
    ):
        return f"{start}{offset}"
    return start


def parse_formula1_session_payload(
    payload: Any,
    *,
    event: EventKey,
    kind: SessionKind,
    meeting_id: int,
    history: pd.DataFrame | None = None,
    player_identity_map: pd.DataFrame | None = None,
    source_fetched_at: datetime | str | None = None,
) -> pd.DataFrame:
    """Normalize one official Formula1.com FOM classification without mutation."""
    value = deepcopy(payload)
    if not isinstance(value, Mapping):
        raise ValueError("Formula 1 session payload is not an object.")
    payload_key = _FORMULA1_PAYLOAD_KEY_BY_KIND.get(kind)
    session = value.get(payload_key) if payload_key else None
    if not isinstance(session, Mapping):
        raise ValueError(
            f"Formula 1 payload is missing the {payload_key or kind.value} session object."
        )
    session_code = str(session.get("session") or "").strip().casefold()
    expected_code = _FORMULA1_SESSION_CODE_BY_KIND.get(kind)
    if session_code != expected_code:
        raise ValueError(
            f"Formula 1 session code {session_code or 'missing'} does not match {kind.value}."
        )
    raw_results = session.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Formula 1 session results are not a list.")

    finality = str(session.get("state") or "not_explicitly_provided").strip()
    fetched_at = (
        coerce_utc_datetime(source_fetched_at).isoformat()
        if source_fetched_at is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            raise ValueError("Formula 1 session result contains a non-object row.")
        driver = {
            "id": raw.get("driverReference") or raw.get("driverId"),
            "abbreviation": raw.get("driverTLA"),
            "given_name": raw.get("driverFirstName"),
            "family_name": raw.get("driverLastName"),
        }
        human_id, match_method, match_status, identity_diagnostic = _identity_candidates(
            driver,
            history=history,
            player_identity_map=player_identity_map,
        )
        given_name = str(raw.get("driverFirstName") or "").strip()
        family_name = str(raw.get("driverLastName") or "").strip()
        display_name = " ".join(part for part in (given_name, family_name) if part)
        position = pd.to_numeric(
            raw.get("positionNumber", raw.get("positionValue")), errors="coerce"
        )
        completion = str(raw.get("completionStatusCode") or "").strip()
        classification = "classified" if not pd.isna(position) else completion or "unclassified"
        status = finality or completion or classification
        rows.append(
            {
                "season": event.season,
                "round": event.round,
                "session_kind": kind.value,
                "human_driver_id": human_id,
                "driver_reference": raw.get("driverReference"),
                "source_driver_id": raw.get("driverId"),
                "abbreviation": str(raw.get("driverTLA") or "").strip().upper(),
                "display_name": display_name,
                "team": str(
                    raw.get("displayTeamName")
                    or raw.get("teamName")
                    or raw.get("constructorSeasonName")
                    or ""
                ).strip(),
                "team_reference": raw.get("teamId", raw.get("teamKey")),
                "position": None if pd.isna(position) else int(position),
                "classification": classification,
                "status": status,
                "source": FORMULA1_SOURCE,
                "session_timestamp": _formula1_session_timestamp(session),
                "source_fetched_at": fetched_at,
                "laps": raw.get("lapsCompleted"),
                "time_gap": raw.get("displayTime", raw.get("gapToLeader")),
                "source_finality": finality,
                "identity_match_method": match_method,
                "identity_match_status": match_status,
                "identity_diagnostic": identity_diagnostic,
                "is_classified": not pd.isna(position),
                "session_components": json.dumps(
                    {
                        key: raw.get(key)
                        for key in ("q1", "q2", "q3")
                        if raw.get(key) is not None
                    },
                    sort_keys=True,
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=SESSION_RESULT_COLUMNS)
    if not frame.empty:
        frame["position"] = pd.array(frame["position"], dtype="Int64")
    frame.attrs["formula1_meeting_id"] = int(meeting_id)
    return frame


def fetch_formula1_session_results(
    url: str,
    *,
    event: EventKey,
    kind: SessionKind,
    meeting_id: int,
    history: pd.DataFrame | None = None,
    player_identity_map: pd.DataFrame | None = None,
    json_loader: Callable[[str], Any] | None = None,
) -> pd.DataFrame:
    """Fetch and normalize a discovered official Formula1.com session result URL."""
    loader = json_loader or _request_formula1_json
    fetched_at = datetime.now(UTC)
    return parse_formula1_session_payload(
        loader(url),
        event=event,
        kind=kind,
        meeting_id=meeting_id,
        history=history,
        player_identity_map=player_identity_map,
        source_fetched_at=fetched_at,
    )


def current_human_driver_field(player_identity_map: pd.DataFrame | None) -> frozenset[str]:
    mapping = (
        player_identity_map.copy(deep=True)
        if isinstance(player_identity_map, pd.DataFrame)
        else pd.DataFrame()
    )
    if mapping.empty or "human_driver_id" not in mapping.columns:
        return frozenset()
    active = mapping.get("active", pd.Series(True, index=mapping.index)).fillna(False).astype(bool)
    matched = ~mapping.get(
        "match_status", pd.Series("matched", index=mapping.index)
    ).isin({"ambiguous", "unresolved"})
    return frozenset(mapping.loc[active & matched, "human_driver_id"].dropna().astype(str))


def expected_participant_count(
    rows: pd.DataFrame,
    current_human_ids: frozenset[str] | set[str] | tuple[str, ...],
) -> tuple[int, str]:
    current = {str(value) for value in current_human_ids if str(value).strip()}
    observed = set()
    if isinstance(rows, pd.DataFrame) and not rows.empty:
        human = rows.get("human_driver_id", pd.Series(index=rows.index, dtype=object))
        source = rows.get("source_driver_id", pd.Series(index=rows.index, dtype=object))
        observed = {
            str(human_value) if pd.notna(human_value) and str(human_value).strip() else f"source:{source_value}"
            for human_value, source_value in zip(human, source)
            if pd.notna(human_value) or pd.notna(source_value)
        }
    if current:
        return max(len(current), len(observed)), "current_human_driver_field"
    if len(observed) >= DEFAULT_EXPECTED_PARTICIPANTS:
        return len(observed), "observed_session_participants"
    return DEFAULT_EXPECTED_PARTICIPANTS, "documented_fallback"


def _formula1_meeting_id(payload: Any, event: EventKey) -> int:
    if not isinstance(payload, list):
        raise ValueError("Formula 1 meeting discovery payload is not a list.")
    if event.round > len(payload):
        raise ValueError(
            f"Formula 1 meeting discovery has no entry for round {event.round}."
        )
    meeting = payload[event.round - 1]
    if not isinstance(meeting, Mapping):
        raise ValueError("Formula 1 meeting discovery contains a non-object entry.")
    meeting_id = pd.to_numeric(meeting.get("value"), errors="coerce")
    if pd.isna(meeting_id) or not float(meeting_id).is_integer() or int(meeting_id) <= 0:
        raise ValueError("Formula 1 meeting discovery has an invalid meeting ID.")
    return int(meeting_id)


def _formula1_result_url(
    payload: Any,
    *,
    kind: SessionKind,
    meeting_id: int,
) -> str:
    if not isinstance(payload, list):
        raise ValueError("Formula 1 meeting datasets payload is not a list.")
    expected_type = _FORMULA1_DATASET_TYPE_BY_KIND[kind].casefold()
    matches = [
        item
        for item in payload
        if isinstance(item, Mapping)
        and str(item.get("editorialSessionType") or "").strip().casefold()
        == expected_type
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Formula 1 meeting datasets has {len(matches)} matches for {kind.value}."
        )
    dataset = matches[0]
    if dataset.get("isSessionResult") is not True or dataset.get("isAvailable") is not True:
        raise ValueError(f"Formula 1 marks {kind.value} results as unavailable.")
    value = str(dataset.get("value") or "").strip()
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    discovered_meeting = pd.to_numeric(
        (query.get("meeting") or [None])[0], errors="coerce"
    )
    if pd.isna(discovered_meeting) or int(discovered_meeting) != meeting_id:
        raise ValueError("Formula 1 dataset belongs to a different meeting.")
    if kind == SessionKind.SPRINT_QUALIFYING:
        valid_path = parsed.path.strip("/") == "sprint-shootout"
    else:
        expected_number = str(
            {
                SessionKind.PRACTICE_1: 1,
                SessionKind.PRACTICE_2: 2,
                SessionKind.PRACTICE_3: 3,
            }[kind]
        )
        valid_path = (
            parsed.path.strip("/") == "practice"
            and (query.get("session") or [None])[0] == expected_number
        )
    if not valid_path:
        raise ValueError(f"Formula 1 dataset path does not match {kind.value}.")
    return urljoin(f"{FORMULA1_API_BASE}/", value)


def _frame_validation_issue(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    human_ids = frame.get("human_driver_id", pd.Series(index=frame.index, dtype=object))
    source_ids = frame.get("source_driver_id", pd.Series(index=frame.index, dtype=object))
    participant_ids = human_ids.where(human_ids.notna(), source_ids)
    if participant_ids.isna().any():
        return "Classification contains a row without any driver identity."
    if participant_ids.astype(str).duplicated().any():
        return "Classification contains duplicate driver rows."
    positions = pd.to_numeric(frame.get("position"), errors="coerce")
    if positions.isna().any() or (positions <= 0).any():
        return "Classification contains missing or implausible positions."
    if positions.astype(int).duplicated().any():
        return "Classification contains duplicate positions."
    if set(positions.astype(int)) != set(range(1, len(frame) + 1)):
        return "Classification positions are not a contiguous 1..N field."
    return None


def compare_session_classifications(
    formula1_rows: pd.DataFrame,
    jolpica_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Compare completed normalized classifications without merging either source."""
    columns = ["human_driver_id", "position", "team", "session_kind"]
    left = formula1_rows.copy(deep=True)
    right = jolpica_rows.copy(deep=True)
    for frame in (left, right):
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
    left_records = {
        str(row.human_driver_id): (
            int(row.position),
            _normalise_text(row.team),
            str(row.session_kind),
        )
        for row in left[columns].itertuples(index=False)
        if pd.notna(row.human_driver_id) and pd.notna(row.position)
    }
    right_records = {
        str(row.human_driver_id): (
            int(row.position),
            _normalise_text(row.team),
            str(row.session_kind),
        )
        for row in right[columns].itertuples(index=False)
        if pd.notna(row.human_driver_id) and pd.notna(row.position)
    }
    missing_from_jolpica = sorted(set(left_records) - set(right_records))
    missing_from_formula1 = sorted(set(right_records) - set(left_records))
    differences = {
        human_id: {"formula1": left_records[human_id], "jolpica": right_records[human_id]}
        for human_id in sorted(set(left_records) & set(right_records))
        if left_records[human_id] != right_records[human_id]
    }
    disagrees = bool(
        len(left) != len(right)
        or missing_from_jolpica
        or missing_from_formula1
        or differences
    )
    return {
        "disagrees": disagrees,
        "formula1_field_size": int(len(left)),
        "jolpica_field_size": int(len(right)),
        "missing_from_jolpica": missing_from_jolpica,
        "missing_from_formula1": missing_from_formula1,
        "driver_differences": differences,
    }


def _diagnostic_state(
    *,
    event: EventKey,
    kind: SessionKind,
    scheduled_at: datetime | None,
    expected: int | None,
    status: SessionStatus,
    diagnostic: str,
    source: str = JOLPICA_ALPHA_SOURCE,
) -> SessionState:
    return SessionState(
        event=event,
        kind=kind,
        scheduled_at=scheduled_at,
        observed_row_count=0,
        expected_participant_count=expected,
        status=status,
        source=source,
        diagnostic=diagnostic,
        supported=True,
    )


def _schedule_event(payload: Any, event: EventKey) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), Mapping):
        raise ValueError("Jolpica alpha schedule is missing a data object.")
    events = payload["data"].get("events")
    if not isinstance(events, list):
        raise ValueError("Jolpica alpha schedule events are not a list.")
    matches = [
        item
        for item in events
        if isinstance(item, Mapping)
        and isinstance(item.get("round"), Mapping)
        and int(item["round"].get("number", 0) or 0) == event.round
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Jolpica alpha schedule has {len(matches)} matches for round {event.round}."
        )
    return matches[0]


def _classify_normalized_session(
    frame: pd.DataFrame,
    *,
    event: EventKey,
    kind: SessionKind,
    scheduled_at: datetime | None,
    effective_time: datetime,
    current_field: frozenset[str],
    source: str,
    source_finality: str,
) -> tuple[SessionState, str]:
    expected, expected_source = expected_participant_count(frame, current_field)
    validation_issue = _frame_validation_issue(frame)
    if validation_issue is not None:
        return (
            SessionState(
                event=event,
                kind=kind,
                scheduled_at=scheduled_at,
                observed_row_count=int(len(frame)),
                expected_participant_count=expected,
                status=SessionStatus.MALFORMED,
                source=source,
                diagnostic=validation_issue,
            ),
            expected_source,
        )
    classifier_frame = frame.copy(deep=True)
    classifier_frame["driverId"] = classifier_frame.get("human_driver_id")
    state = classify_session_dataframe(
        classifier_frame,
        event=event,
        kind=kind,
        scheduled_at=scheduled_at,
        effective_time=effective_time,
        expected_participant_count=expected,
        source=source,
    )
    explicitly_provisional = bool(
        re.search(r"provisional|live|running|in progress", source_finality, re.IGNORECASE)
    )
    explicitly_final = bool(
        re.fullmatch(
            r"final|official|complete|completed", source_finality.strip(), re.IGNORECASE
        )
    )
    if explicitly_provisional:
        state = replace(
            state,
            status=SessionStatus.PROVISIONAL,
            diagnostic=f"Source finality is {source_finality}.",
        )
    if (
        state.status == SessionStatus.COMPLETE
        and expected_source != "current_human_driver_field"
        and not explicitly_final
    ):
        state = replace(
            state,
            status=SessionStatus.PROVISIONAL,
            diagnostic=(
                "Classification coverage is present, but no independent current-driver "
                "field or explicit source finality confirms completeness."
            ),
        )
    earliest_final = (
        scheduled_at + MINIMUM_SESSION_AGE_FOR_FINAL
        if scheduled_at is not None
        else None
    )
    if (
        state.status == SessionStatus.COMPLETE
        and not explicitly_final
        and earliest_final is not None
        and effective_time < earliest_final
    ):
        state = replace(
            state,
            status=SessionStatus.IN_PROGRESS,
            diagnostic="Full coverage is present, but the scheduled session may still be running.",
        )
    return state, expected_source


def ingest_active_event_sessions(
    schedule_row: Mapping[str, Any] | pd.Series,
    *,
    format: WeekendFormat,
    history: pd.DataFrame | None = None,
    player_identity_map: pd.DataFrame | None = None,
    effective_time: datetime | str | None = None,
    force_refresh: bool = False,
    json_loader: Callable[[str], Any] | None = None,
    formula1_json_loader: Callable[[str], Any] | None = None,
    validate_sources: bool = False,
) -> LiveSessionIngestion:
    """Fetch active-event sessions from Formula 1, falling back lazily to Jolpica."""
    row = schedule_row.to_dict() if isinstance(schedule_row, pd.Series) else dict(schedule_row)
    event = EventKey(int(row.get("season")), int(row.get("round")))
    now = coerce_utc_datetime(effective_time)
    jolpica_loader = json_loader or _request_json
    if formula1_json_loader is not None:
        formula1_loader = formula1_json_loader
    elif json_loader is None:
        formula1_loader = _request_formula1_json
    else:
        # Existing tests and callers may inject only the legacy Jolpica loader.
        # A separate injection point prevents accidental real network calls.
        def formula1_loader(_url: str) -> Any:
            raise RuntimeError("Formula 1 source loader was not injected.")

    kinds = expected_live_session_kinds(format)
    current_field = current_human_driver_field(player_identity_map)
    meetings_url = f"{FORMULA1_API_BASE}/dropdown-meetings?{urlencode({'season': event.season})}"
    jolpica_schedule_url = f"{JOLPICA_ALPHA_BASE}/schedules/{event.season}/"
    diagnostics: dict[str, Any] = {
        "active_event": {"season": event.season, "round": event.round},
        "weekend_format": format.value,
        "source": FORMULA1_SOURCE,
        "source_priority": [FORMULA1_SOURCE, JOLPICA_FALLBACK_SOURCE],
        "expected_session_kinds": [kind.value for kind in kinds],
        "current_human_driver_count": len(current_field),
        "formula1_meetings_url": meetings_url,
        "jolpica_schedule_url": jolpica_schedule_url,
        "cross_source_validation_enabled": bool(validate_sources),
        "sessions": {},
    }

    meeting_id: int | None = None
    datasets_payload: Any = None
    datasets_url: str | None = None
    try:
        meetings_payload, meetings_fetched_at, meetings_cache_hit = _cached_json(
            meetings_url,
            force_refresh=force_refresh,
            effective_time=now,
            loader=formula1_loader,
            ttl=CURRENT_SEASON_SCHEDULE_TTL,
        )
        meeting_id = _formula1_meeting_id(meetings_payload, event)
        datasets_url = (
            f"{FORMULA1_API_BASE}/dropdown-meeting-datasets?"
            f"{urlencode({'meeting': meeting_id})}"
        )
        datasets_payload, datasets_fetched_at, datasets_cache_hit = _cached_json(
            datasets_url,
            force_refresh=force_refresh,
            effective_time=now,
            loader=formula1_loader,
            ttl=CURRENT_EVENT_NONFINAL_TTL,
        )
        if all(
            _formula1_result_url(
                datasets_payload,
                kind=kind,
                meeting_id=meeting_id,
            )
            for kind in kinds
        ):
            _mark_authoritative(datasets_url)
        diagnostics.update(
            {
                "formula1_meeting_id": meeting_id,
                "formula1_meetings_fetch_timestamp_utc": meetings_fetched_at.isoformat(),
                "formula1_meetings_cache_hit": meetings_cache_hit,
                "formula1_datasets_url": datasets_url,
                "formula1_datasets_fetch_timestamp_utc": datasets_fetched_at.isoformat(),
                "formula1_datasets_cache_hit": datasets_cache_hit,
            }
        )
    except Exception as exc:
        diagnostics["formula1_discovery_error"] = str(exc)

    jolpica_schedule_loaded = False
    jolpica_schedule_payload: Any = None
    jolpica_schedule_fetched_at: datetime | None = None
    jolpica_schedule_cache_hit = False
    jolpica_schedule_error: Exception | None = None

    def jolpica_attempt(
        kind: SessionKind,
        scheduled_at: datetime | None,
    ) -> tuple[pd.DataFrame, SessionState, dict[str, Any], str | None]:
        nonlocal jolpica_schedule_loaded
        nonlocal jolpica_schedule_payload
        nonlocal jolpica_schedule_fetched_at
        nonlocal jolpica_schedule_cache_hit
        nonlocal jolpica_schedule_error
        try:
            if not jolpica_schedule_loaded:
                jolpica_schedule_loaded = True
                try:
                    (
                        jolpica_schedule_payload,
                        jolpica_schedule_fetched_at,
                        jolpica_schedule_cache_hit,
                    ) = _cached_json(
                        jolpica_schedule_url,
                        force_refresh=force_refresh,
                        effective_time=now,
                        loader=jolpica_loader,
                        ttl=CURRENT_SEASON_SCHEDULE_TTL,
                    )
                except Exception as exc:
                    jolpica_schedule_error = exc
            if jolpica_schedule_error is not None:
                raise jolpica_schedule_error
            alpha_event = _schedule_event(jolpica_schedule_payload, event)
            schedule_items = alpha_event.get("schedule")
            if not isinstance(schedule_items, list):
                raise ValueError("Jolpica alpha event schedule is not a list.")
            source_code = _SOURCE_CODE_BY_KIND[kind]
            matches = [
                item
                for item in schedule_items
                if isinstance(item, Mapping)
                and str(item.get("code") or "").upper() == source_code
            ]
            if len(matches) != 1 or not str(matches[0].get("results_url") or "").startswith(
                f"{JOLPICA_ALPHA_BASE}/results/"
            ):
                raise ValueError(f"Alpha schedule is missing one valid {source_code} URL.")
            result_url = str(matches[0]["results_url"])
            payload, fetched_at, cache_hit = _cached_json(
                result_url,
                force_refresh=force_refresh,
                effective_time=now,
                loader=jolpica_loader,
                ttl=CURRENT_EVENT_NONFINAL_TTL,
            )
            try:
                frame = parse_jolpica_session_payload(
                    payload,
                    event=event,
                    kind=kind,
                    history=history,
                    player_identity_map=player_identity_map,
                    source=JOLPICA_FALLBACK_SOURCE,
                )
            except Exception as exc:
                state = _diagnostic_state(
                    event=event,
                    kind=kind,
                    scheduled_at=scheduled_at,
                    expected=len(current_field) or DEFAULT_EXPECTED_PARTICIPANTS,
                    status=SessionStatus.MALFORMED,
                    diagnostic=f"Jolpica fallback payload is malformed: {exc}",
                    source=JOLPICA_FALLBACK_SOURCE,
                )
                return empty_session_results(), state, {
                    "source": JOLPICA_FALLBACK_SOURCE,
                    "results_url": result_url,
                    "fetch_timestamp_utc": fetched_at.isoformat(),
                    "cache_hit": cache_hit,
                    "error": str(exc),
                }, result_url
            payload_data = payload.get("data", {}) if isinstance(payload, Mapping) else {}
            finality = _source_finality(payload_data) if isinstance(payload_data, Mapping) else ""
            state, expected_source = _classify_normalized_session(
                frame,
                event=event,
                kind=kind,
                scheduled_at=scheduled_at,
                effective_time=now,
                current_field=current_field,
                source=JOLPICA_FALLBACK_SOURCE,
                source_finality=finality,
            )
            if state.status == SessionStatus.COMPLETE:
                _mark_authoritative(result_url)
            return frame, state, {
                "source": JOLPICA_FALLBACK_SOURCE,
                "results_url": result_url,
                "fetch_timestamp_utc": fetched_at.isoformat(),
                "cache_hit": cache_hit,
                "schedule_fetch_timestamp_utc": (
                    jolpica_schedule_fetched_at.isoformat()
                    if jolpica_schedule_fetched_at is not None
                    else None
                ),
                "schedule_cache_hit": jolpica_schedule_cache_hit,
                "expected_participant_source": expected_source,
            }, result_url
        except Exception as exc:
            state = _diagnostic_state(
                event=event,
                kind=kind,
                scheduled_at=scheduled_at,
                expected=len(current_field) or DEFAULT_EXPECTED_PARTICIPANTS,
                status=SessionStatus.FAILED,
                diagnostic=f"Jolpica fallback failed: {exc}",
                source=JOLPICA_FALLBACK_SOURCE,
            )
            return empty_session_results(), state, {
                "source": JOLPICA_FALLBACK_SOURCE,
                "error": str(exc),
            }, None

    frames: list[pd.DataFrame] = []
    states: list[SessionState] = []
    for kind in kinds:
        scheduled_at = scheduled_session_timestamp(row, kind)
        formula_frame = empty_session_results()
        formula_state: SessionState | None = None
        formula_diag: dict[str, Any] = {"source": FORMULA1_SOURCE}
        formula_url: str | None = None
        try:
            if meeting_id is None or datasets_payload is None:
                raise RuntimeError(
                    diagnostics.get("formula1_discovery_error", "Formula 1 discovery unavailable.")
                )
            formula_url = _formula1_result_url(
                datasets_payload, kind=kind, meeting_id=meeting_id
            )
            formula_payload, formula_fetched_at, formula_cache_hit = _cached_json(
                formula_url,
                force_refresh=force_refresh,
                effective_time=now,
                loader=formula1_loader,
                ttl=CURRENT_EVENT_NONFINAL_TTL,
            )
            formula_frame = parse_formula1_session_payload(
                formula_payload,
                event=event,
                kind=kind,
                meeting_id=meeting_id,
                history=history,
                player_identity_map=player_identity_map,
                source_fetched_at=formula_fetched_at,
            )
            session_key = _FORMULA1_PAYLOAD_KEY_BY_KIND[kind]
            session_payload = formula_payload.get(session_key, {})
            finality = (
                str(session_payload.get("state") or "")
                if isinstance(session_payload, Mapping)
                else ""
            )
            formula_state, expected_source = _classify_normalized_session(
                formula_frame,
                event=event,
                kind=kind,
                scheduled_at=scheduled_at,
                effective_time=now,
                current_field=current_field,
                source=FORMULA1_SOURCE,
                source_finality=finality,
            )
            if formula_state.status == SessionStatus.COMPLETE:
                _mark_authoritative(formula_url)
            formula_diag.update(
                {
                    "results_url": formula_url,
                    "fetch_timestamp_utc": formula_fetched_at.isoformat(),
                    "cache_hit": formula_cache_hit,
                    "expected_participant_source": expected_source,
                }
            )
        except ValueError as exc:
            formula_state = _diagnostic_state(
                event=event,
                kind=kind,
                scheduled_at=scheduled_at,
                expected=len(current_field) or DEFAULT_EXPECTED_PARTICIPANTS,
                status=SessionStatus.MALFORMED,
                diagnostic=f"Formula 1 source payload is malformed: {exc}",
                source=FORMULA1_SOURCE,
            )
            formula_diag["error"] = str(exc)
        except Exception as exc:
            formula_state = _diagnostic_state(
                event=event,
                kind=kind,
                scheduled_at=scheduled_at,
                expected=len(current_field) or DEFAULT_EXPECTED_PARTICIPANTS,
                status=SessionStatus.FAILED,
                diagnostic=f"Formula 1 source failed: {exc}",
                source=FORMULA1_SOURCE,
            )
            formula_diag["error"] = str(exc)

        selected_frame = formula_frame
        selected_state = formula_state
        session_diag: dict[str, Any] = {
            "formula1": formula_diag,
            "fallback_attempted": False,
        }
        should_fallback = formula_state.status != SessionStatus.COMPLETE
        if should_fallback or validate_sources:
            jolpica_frame, jolpica_state, jolpica_diag, _ = jolpica_attempt(
                kind, scheduled_at
            )
            session_diag["fallback_attempted"] = True
            session_diag["jolpica"] = jolpica_diag
            if formula_state.status == SessionStatus.COMPLETE:
                if jolpica_state.status == SessionStatus.COMPLETE:
                    comparison = compare_session_classifications(
                        formula_frame, jolpica_frame
                    )
                    session_diag["cross_source_validation"] = comparison
                    if comparison["disagrees"]:
                        session_diag["source_disagreement"] = (
                            "Jolpica classification differs; Formula 1 remains authoritative."
                        )
            elif jolpica_state.status == SessionStatus.COMPLETE:
                selected_frame = jolpica_frame
                selected_state = jolpica_state
                session_diag["fallback_used"] = True
            else:
                nonfailed = {
                    SessionStatus.PENDING,
                    SessionStatus.IN_PROGRESS,
                    SessionStatus.PROVISIONAL,
                    SessionStatus.PARTIAL,
                }
                if (
                    formula_state.status not in nonfailed
                    and jolpica_state.status in nonfailed
                ) or (
                    formula_state.status in nonfailed
                    and jolpica_state.status in nonfailed
                    and jolpica_state.observed_row_count
                    > formula_state.observed_row_count
                ):
                    selected_frame = jolpica_frame
                    selected_state = jolpica_state
                elif formula_state.status not in nonfailed and jolpica_state.status not in nonfailed:
                    if (
                        formula_state.status == SessionStatus.FAILED
                        and jolpica_state.status == SessionStatus.MALFORMED
                    ):
                        selected_frame = jolpica_frame
                        selected_state = jolpica_state
                    selected_state = replace(
                        selected_state,
                        diagnostic=(
                            f"{formula_state.diagnostic} {jolpica_state.diagnostic}"
                        ).strip(),
                    )

        if not selected_frame.empty:
            frames.append(selected_frame)
        states.append(selected_state)
        identity_issues = selected_frame[
            ~selected_frame.get(
                "identity_match_status",
                pd.Series(index=selected_frame.index, dtype=object),
            ).eq("matched")
        ]
        selected_source_diag = (
            formula_diag
            if selected_state.source == FORMULA1_SOURCE
            else session_diag.get("jolpica", {})
        )
        diagnostics["sessions"][kind.value] = {
            **_state_diagnostic(selected_state),
            **session_diag,
            "results_url": selected_source_diag.get("results_url"),
            "fetch_timestamp_utc": selected_source_diag.get("fetch_timestamp_utc"),
            "cache_hit": selected_source_diag.get("cache_hit", False),
            "identity_issue_count": int(len(identity_issues)),
            "identity_diagnostics": identity_issues[
                ["driver_reference", "abbreviation", "display_name", "identity_match_status", "identity_diagnostic"]
            ].to_dict("records")
            if not identity_issues.empty
            else [],
        }

    results = (
        pd.concat(frames, ignore_index=True, sort=False)[SESSION_RESULT_COLUMNS]
        if frames
        else empty_session_results()
    )
    diagnostics["rows_observed"] = int(len(results))
    diagnostics["fetch_timestamp_utc"] = now.isoformat()
    return LiveSessionIngestion(results.copy(deep=True), tuple(states), diagnostics)


def _state_diagnostic(state: SessionState) -> dict[str, Any]:
    return {
        "session_kind": state.kind.value,
        "status": state.status.value,
        "rows_observed": state.observed_row_count,
        "expected_participants": state.expected_participant_count,
        "scheduled_at": state.scheduled_at.isoformat() if state.scheduled_at else None,
        "source": state.source,
        "diagnostic": state.diagnostic,
    }


def live_session_signature(
    results: pd.DataFrame,
    states: tuple[SessionState, ...],
) -> tuple[Any, ...]:
    """Return a stable, order-independent signature for raw session data."""
    rows = results.copy(deep=True) if isinstance(results, pd.DataFrame) else empty_session_results()
    keys = [
        column
        for column in (
            "season",
            "round",
            "session_kind",
            "human_driver_id",
            "source_driver_id",
            "position",
            "classification",
            "team",
        )
        if column in rows.columns
    ]
    row_signature = tuple(
        tuple(None if pd.isna(value) else str(value) for value in row)
        for row in rows[keys].sort_values(keys, kind="stable", na_position="last").itertuples(index=False, name=None)
    ) if keys and not rows.empty else ()
    state_signature = tuple(
        (
            state.event.season,
            state.event.round,
            state.kind.value,
            state.status.value,
            state.observed_row_count,
            state.expected_participant_count,
        )
        for state in sorted(states, key=lambda item: (item.event, item.kind.value))
    )
    return state_signature, row_signature
