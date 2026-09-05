from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import re
from typing import Any, Iterable, Mapping

import pandas as pd


DEFAULT_EXPECTED_PARTICIPANTS = 20
SESSION_PUBLICATION_GRACE = timedelta(hours=3)
ACTIVE_EVENT_FALLBACK_TIMEOUT = timedelta(hours=48)


@dataclass(frozen=True, order=True)
class EventKey:
    season: int
    round: int


class WeekendFormat(str, Enum):
    NORMAL = "normal"
    SPRINT = "sprint"


class SessionKind(str, Enum):
    PRACTICE_1 = "practice_1"
    PRACTICE_2 = "practice_2"
    PRACTICE_3 = "practice_3"
    SPRINT_QUALIFYING = "sprint_qualifying"
    SPRINT = "sprint"
    GRAND_PRIX_QUALIFYING = "grand_prix_qualifying"
    GRAND_PRIX = "grand_prix"


class SessionStatus(str, Enum):
    NOT_SCHEDULED = "not_scheduled"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PROVISIONAL = "provisional"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class SessionState:
    event: EventKey
    kind: SessionKind
    scheduled_at: datetime | None
    observed_row_count: int
    expected_participant_count: int | None
    status: SessionStatus
    source: str
    diagnostic: str = ""
    supported: bool = True


@dataclass(frozen=True)
class WeekendState:
    event: EventKey
    race_name: str
    format: WeekendFormat
    sessions: tuple[SessionState, ...]
    active_session: SessionKind | None
    next_session: SessionKind | None
    status: str
    is_final: bool
    unresolved_after_timeout: bool
    diagnostics: tuple[str, ...] = ()

    def session(self, kind: SessionKind) -> SessionState:
        return next(item for item in self.sessions if item.kind == kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": {"season": self.event.season, "round": self.event.round},
            "race_name": self.race_name,
            "format": self.format.value,
            "status": self.status,
            "is_final": self.is_final,
            "unresolved_after_timeout": self.unresolved_after_timeout,
            "active_session": self.active_session.value if self.active_session else None,
            "next_session": self.next_session.value if self.next_session else None,
            "sessions": {
                item.kind.value: {
                    "status": item.status.value,
                    "scheduled_at": item.scheduled_at.isoformat() if item.scheduled_at else None,
                    "observed_row_count": item.observed_row_count,
                    "expected_participant_count": item.expected_participant_count,
                    "source": item.source,
                    "diagnostic": item.diagnostic,
                    "supported": item.supported,
                }
                for item in self.sessions
            },
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class UpcomingEvent:
    event: EventKey
    circuit: str
    race_name: str
    format: WeekendFormat
    scheduled_at: datetime | None
    horizon_weight: float

    @property
    def season(self) -> int:
        return self.event.season

    @property
    def round(self) -> int:
        return self.event.round

    def as_dict(self) -> dict[str, Any]:
        return {
            "season": self.event.season,
            "round": self.event.round,
            "circuit": self.circuit,
            "race_name": self.race_name,
            "weekend_format": self.format.value,
            "scheduled_race_timestamp": (
                self.scheduled_at.isoformat() if self.scheduled_at else None
            ),
            "horizon_weight": float(self.horizon_weight),
        }


@dataclass(frozen=True)
class SnapshotValidation:
    status: str
    safe_for_scoring: bool
    warnings: tuple[str, ...]
    active_weekend: WeekendState | None
    completed_event_keys: tuple[EventKey, ...]
    excluded_partial_event_keys: tuple[EventKey, ...]
    unresolved_event_keys: tuple[EventKey, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "safe_for_scoring": self.safe_for_scoring,
            "warnings": list(self.warnings),
            "completed_event_keys": [(key.season, key.round) for key in self.completed_event_keys],
            "excluded_partial_event_keys": [
                (key.season, key.round) for key in self.excluded_partial_event_keys
            ],
            "unresolved_event_keys": [
                (key.season, key.round) for key in self.unresolved_event_keys
            ],
            "active_weekend": self.active_weekend.as_dict() if self.active_weekend else None,
        }


_SCHEDULE_FIELDS: dict[SessionKind, tuple[str, str]] = {
    SessionKind.PRACTICE_1: ("practice_1_date", "practice_1_time"),
    SessionKind.PRACTICE_2: ("practice_2_date", "practice_2_time"),
    SessionKind.PRACTICE_3: ("practice_3_date", "practice_3_time"),
    SessionKind.SPRINT_QUALIFYING: ("sprint_qualifying_date", "sprint_qualifying_time"),
    SessionKind.SPRINT: ("sprint_date", "sprint_time"),
    SessionKind.GRAND_PRIX_QUALIFYING: ("qualifying_date", "qualifying_time"),
    SessionKind.GRAND_PRIX: ("date", "time"),
}

_REQUIRED_COLUMNS: dict[SessionKind, frozenset[str]] = {
    SessionKind.PRACTICE_1: frozenset(
        {"season", "round", "driverId", "position"}
    ),
    SessionKind.PRACTICE_2: frozenset(
        {"season", "round", "driverId", "position"}
    ),
    SessionKind.PRACTICE_3: frozenset(
        {"season", "round", "driverId", "position"}
    ),
    SessionKind.GRAND_PRIX: frozenset(
        {"season", "round", "driverId", "position", "grid", "status"}
    ),
    SessionKind.SPRINT: frozenset(
        {"season", "round", "driverId", "position", "grid", "status"}
    ),
    SessionKind.GRAND_PRIX_QUALIFYING: frozenset(
        {"season", "round", "driverId", "position"}
    ),
    SessionKind.SPRINT_QUALIFYING: frozenset(
        {"season", "round", "driverId", "position"}
    ),
}

_NON_FINAL_STATUS_WORDS = re.compile(
    r"\b(?:running|live|provisional|pending|in progress|not started|under investigation)\b",
    re.IGNORECASE,
)


def coerce_utc_datetime(value: Any | None) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return datetime.now(UTC)
    if len(text) == 10:
        text = f"{text}T12:00:00Z"
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def canonical_event_key(season: Any, round_no: Any) -> EventKey:
    season_value = pd.to_numeric(season, errors="coerce")
    round_value = pd.to_numeric(round_no, errors="coerce")
    if (
        pd.isna(season_value)
        or pd.isna(round_value)
        or float(season_value) <= 0
        or float(round_value) <= 0
        or not float(season_value).is_integer()
        or not float(round_value).is_integer()
    ):
        raise ValueError("Event identity requires positive integer season and round values.")
    return EventKey(int(season_value), int(round_value))


def parse_schedule_timestamp(date_value: Any, time_value: Any = None) -> datetime | None:
    date_text = str(date_value or "").strip()
    if not date_text or date_text.casefold() in {"nan", "nat", "none"}:
        return None
    time_text = str(time_value or "").strip()
    if time_text.casefold() in {"nan", "nat", "none"}:
        time_text = ""
    raw = f"{date_text}T{time_text}" if time_text else f"{date_text}T00:00:00Z"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def weekend_format(schedule_row: Mapping[str, Any] | pd.Series) -> WeekendFormat:
    row = schedule_row.to_dict() if isinstance(schedule_row, pd.Series) else dict(schedule_row)
    if parse_schedule_timestamp(row.get("sprint_date"), row.get("sprint_time")) is not None:
        return WeekendFormat.SPRINT
    if parse_schedule_timestamp(
        row.get("sprint_qualifying_date"), row.get("sprint_qualifying_time")
    ) is not None:
        return WeekendFormat.SPRINT
    return WeekendFormat.NORMAL


def scheduled_session_timestamp(
    schedule_row: Mapping[str, Any] | pd.Series,
    kind: SessionKind,
) -> datetime | None:
    row = schedule_row.to_dict() if isinstance(schedule_row, pd.Series) else dict(schedule_row)
    date_field, time_field = _SCHEDULE_FIELDS[kind]
    return parse_schedule_timestamp(row.get(date_field), row.get(time_field))


def upcoming_event_records(
    schedule: pd.DataFrame,
    *,
    start_event: EventKey | None = None,
    effective_time: datetime | str | None = None,
    limit: int = 5,
    first_weight: float = 1.0,
    later_weight: float = 0.7,
) -> tuple[UpcomingEvent, ...]:
    """Build an ordered, immutable event horizon without mutating the schedule."""
    if schedule is None or schedule.empty or not {"season", "round"}.issubset(schedule.columns):
        return ()
    if int(limit) <= 0:
        return ()
    data = schedule.copy(deep=True)
    data["_season"] = pd.to_numeric(data["season"], errors="coerce")
    data["_round"] = pd.to_numeric(data["round"], errors="coerce")
    data = data.dropna(subset=["_season", "_round"]).sort_values(
        ["_season", "_round"], kind="stable"
    )
    data = data.drop_duplicates(subset=["_season", "_round"], keep="first")
    if start_event is not None:
        data = data[
            (data["_season"] > start_event.season)
            | (
                (data["_season"] == start_event.season)
                & (data["_round"] >= start_event.round)
            )
        ]
    elif effective_time is not None:
        now = coerce_utc_datetime(effective_time)
        scheduled = data.apply(
            lambda row: scheduled_session_timestamp(row, SessionKind.GRAND_PRIX),
            axis=1,
        )
        data = data[
            [timestamp is None or timestamp >= now for timestamp in scheduled]
        ]
    data = data.head(int(limit))
    records: list[UpcomingEvent] = []
    for index, (_, row) in enumerate(data.iterrows()):
        event = canonical_event_key(row.get("season"), row.get("round"))
        records.append(
            UpcomingEvent(
                event=event,
                circuit=str(row.get("circuitName") or "").strip(),
                race_name=str(
                    row.get("raceName") or row.get("circuitName") or f"Round {event.round}"
                ).strip(),
                format=weekend_format(row),
                scheduled_at=scheduled_session_timestamp(row, SessionKind.GRAND_PRIX),
                horizon_weight=float(first_weight if index == 0 else later_weight),
            )
        )
    return tuple(records)


def upcoming_circuit_names(events: Iterable[UpcomingEvent]) -> list[str]:
    """Compatibility projection for legacy circuit-string forecast helpers."""
    return [event.circuit.split(" Circuit")[0].strip() for event in events]


def _event_rows(frame: pd.DataFrame | None, event: EventKey) -> pd.DataFrame:
    if frame is None or frame.empty or not {"season", "round"}.issubset(frame.columns):
        return pd.DataFrame()
    data = frame.copy(deep=True)
    seasons = pd.to_numeric(data["season"], errors="coerce")
    rounds = pd.to_numeric(data["round"], errors="coerce")
    return data[(seasons == event.season) & (rounds == event.round)].copy()


def classify_session_dataframe(
    frame: pd.DataFrame | None,
    *,
    event: EventKey,
    kind: SessionKind,
    scheduled_at: datetime | None,
    effective_time: datetime | str | None,
    expected_participant_count: int | None = None,
    source: str,
    source_error: Exception | str | None = None,
    supported: bool = True,
    publication_grace: timedelta = SESSION_PUBLICATION_GRACE,
) -> SessionState:
    """Classify one session without mutating its source dataframe."""
    now = coerce_utc_datetime(effective_time)
    expected = (
        int(expected_participant_count)
        if expected_participant_count is not None and int(expected_participant_count) > 0
        else DEFAULT_EXPECTED_PARTICIPANTS
    )
    if source_error is not None:
        return SessionState(
            event, kind, scheduled_at, 0, expected, SessionStatus.FAILED, source,
            f"Source failed: {source_error}", supported,
        )
    if not supported:
        status = SessionStatus.PENDING if scheduled_at is not None else SessionStatus.NOT_SCHEDULED
        return SessionState(
            event, kind, scheduled_at, 0, expected, status, source,
            "Session is scheduled but this source is not currently supported."
            if scheduled_at is not None
            else "Session is not scheduled for this weekend.",
            False,
        )
    if scheduled_at is None:
        return SessionState(
            event, kind, None, 0, None, SessionStatus.NOT_SCHEDULED, source,
            "Session is not scheduled for this weekend.", supported,
        )
    if frame is not None and not isinstance(frame, pd.DataFrame):
        return SessionState(
            event, kind, scheduled_at, 0, expected, SessionStatus.MALFORMED, source,
            "Source response is not a dataframe.", supported,
        )

    event_rows = _event_rows(frame, event)
    observed = int(event_rows["driverId"].nunique()) if "driverId" in event_rows.columns else len(event_rows)
    if event_rows.empty:
        if now < scheduled_at:
            status = SessionStatus.PENDING
            diagnostic = "Expected empty response before the scheduled session."
        elif now <= scheduled_at + publication_grace:
            status = SessionStatus.IN_PROGRESS
            diagnostic = "No classification published while the session may still be in progress."
        else:
            status = SessionStatus.PARTIAL
            diagnostic = "No classification published after the session publication grace period."
        return SessionState(event, kind, scheduled_at, 0, expected, status, source, diagnostic)

    required = _REQUIRED_COLUMNS.get(kind, frozenset())
    missing_columns = sorted(required - set(event_rows.columns))
    if missing_columns:
        return SessionState(
            event, kind, scheduled_at, observed, expected, SessionStatus.MALFORMED, source,
            f"Missing required columns: {', '.join(missing_columns)}.", supported,
        )

    required_values = [column for column in ("driverId", "position") if column in required]
    if required_values and event_rows[required_values].isna().any(axis=None):
        return SessionState(
            event, kind, scheduled_at, observed, expected, SessionStatus.PARTIAL, source,
            "Classification contains rows without participant identity or position.", supported,
        )

    if "status" in event_rows.columns:
        statuses = event_rows["status"].fillna("").astype(str)
        if kind in {SessionKind.GRAND_PRIX, SessionKind.SPRINT} and statuses.str.strip().eq("").any():
            return SessionState(
                event, kind, scheduled_at, observed, expected, SessionStatus.PARTIAL, source,
                "Classification contains rows without a final status.", supported,
            )
        non_final = statuses.str.contains(_NON_FINAL_STATUS_WORDS, na=False)
        if non_final.any():
            status = (
                SessionStatus.IN_PROGRESS
                if statuses[non_final].str.contains(r"running|live|in progress", case=False, regex=True).any()
                else SessionStatus.PROVISIONAL
            )
            return SessionState(
                event, kind, scheduled_at, observed, expected, status, source,
                "Classification contains running or provisional statuses.", supported,
            )

    if observed < expected:
        return SessionState(
            event, kind, scheduled_at, observed, expected, SessionStatus.PARTIAL, source,
            f"Participant coverage is {observed}/{expected}.", supported,
        )
    return SessionState(
        event, kind, scheduled_at, observed, expected, SessionStatus.COMPLETE, source,
        f"Final classification has {observed}/{expected} participants.", supported,
    )


def classify_schedule_dataframe(
    schedule: pd.DataFrame | None,
    *,
    season: int,
    source_error: Exception | str | None = None,
) -> tuple[SessionStatus, str]:
    if source_error is not None:
        return SessionStatus.FAILED, f"Schedule source failed: {source_error}"
    if schedule is None or not isinstance(schedule, pd.DataFrame):
        return SessionStatus.MALFORMED, "Schedule response is not a dataframe."
    if schedule.empty:
        return SessionStatus.PARTIAL, "Schedule response is empty."
    missing = {"season", "round", "date", "circuitName"} - set(schedule.columns)
    if missing:
        return SessionStatus.MALFORMED, f"Schedule is missing: {', '.join(sorted(missing))}."
    rows = schedule[pd.to_numeric(schedule["season"], errors="coerce") == int(season)]
    if rows.empty:
        return SessionStatus.PARTIAL, f"Schedule has no rows for season {season}."
    return SessionStatus.COMPLETE, f"Schedule has {len(rows)} events for season {season}."


def classify_playerstats_payload(
    payload: Any,
    *,
    source_error: Exception | str | None = None,
) -> tuple[SessionStatus, str]:
    if source_error is not None:
        return SessionStatus.FAILED, f"Playerstats source failed: {source_error}"
    if not isinstance(payload, dict) or not isinstance(payload.get("Value"), dict):
        return SessionStatus.MALFORMED, "Playerstats payload is missing a Value object."
    value = payload["Value"]
    if not value.get("GamedayWiseStats") and not value.get("MatchWiseStats"):
        return SessionStatus.PENDING, "Playerstats contains no published gameday observations."
    return SessionStatus.COMPLETE, "Playerstats payload has structured gameday observations."


def classify_deadline_payload(
    payload: Any,
    *,
    source_error: Exception | str | None = None,
) -> tuple[SessionStatus, str]:
    if source_error is not None:
        return SessionStatus.FAILED, f"Deadline source failed: {source_error}"
    if not isinstance(payload, dict):
        return SessionStatus.MALFORMED, "Deadline payload is not an object."
    value = payload.get("team_lock_deadline_utc")
    if not value:
        return SessionStatus.PENDING, "Official deadline is not currently published."
    try:
        coerce_utc_datetime(value)
    except (TypeError, ValueError):
        return SessionStatus.MALFORMED, "Official deadline timestamp is invalid."
    return SessionStatus.COMPLETE, "Official deadline timestamp is parseable."


def build_weekend_state(
    schedule_row: Mapping[str, Any] | pd.Series,
    *,
    results: pd.DataFrame | None,
    qualifying: pd.DataFrame | None,
    sprint: pd.DataFrame | None,
    sprint_qualifying: pd.DataFrame | None = None,
    effective_time: datetime | str | None = None,
    expected_participant_count: int | None = None,
    source_errors: Mapping[str, Exception | str] | None = None,
) -> WeekendState:
    row = schedule_row.to_dict() if isinstance(schedule_row, pd.Series) else dict(schedule_row)
    event = canonical_event_key(row.get("season"), row.get("round"))
    now = coerce_utc_datetime(effective_time)
    fmt = weekend_format(row)
    errors = dict(source_errors or {})
    frames = {
        SessionKind.GRAND_PRIX: (results, "jolpica_results", True),
        SessionKind.GRAND_PRIX_QUALIFYING: (qualifying, "jolpica_qualifying", True),
        SessionKind.SPRINT: (sprint, "jolpica_sprint", True),
        SessionKind.SPRINT_QUALIFYING: (
            sprint_qualifying,
            "unsupported_sprint_qualifying",
            sprint_qualifying is not None,
        ),
    }
    ordered_kinds = (
        SessionKind.PRACTICE_1,
        SessionKind.PRACTICE_2,
        SessionKind.PRACTICE_3,
        SessionKind.SPRINT_QUALIFYING,
        SessionKind.SPRINT,
        SessionKind.GRAND_PRIX_QUALIFYING,
        SessionKind.GRAND_PRIX,
    )
    sessions: list[SessionState] = []
    for kind in ordered_kinds:
        scheduled_at = scheduled_session_timestamp(row, kind)
        if kind in frames:
            frame, source, supported = frames[kind]
            if fmt == WeekendFormat.NORMAL and kind in {
                SessionKind.SPRINT,
                SessionKind.SPRINT_QUALIFYING,
            }:
                scheduled_at = None
            sessions.append(
                classify_session_dataframe(
                    frame,
                    event=event,
                    kind=kind,
                    scheduled_at=scheduled_at,
                    effective_time=now,
                    expected_participant_count=expected_participant_count,
                    source=source,
                    source_error=errors.get(kind.value),
                    supported=supported,
                )
            )
        else:
            sessions.append(
                SessionState(
                    event=event,
                    kind=kind,
                    scheduled_at=scheduled_at,
                    observed_row_count=0,
                    expected_participant_count=None,
                    status=(
                        SessionStatus.PENDING
                        if scheduled_at is not None and now < scheduled_at
                        else SessionStatus.IN_PROGRESS
                        if scheduled_at is not None and now <= scheduled_at + SESSION_PUBLICATION_GRACE
                        else SessionStatus.COMPLETE
                        if scheduled_at is not None
                        else SessionStatus.NOT_SCHEDULED
                    ),
                    source="schedule_only",
                    diagnostic="Schedule metadata only; practice results are not ingested.",
                    supported=False,
                )
            )

    by_kind = {item.kind: item for item in sessions}
    required = [SessionKind.GRAND_PRIX_QUALIFYING, SessionKind.GRAND_PRIX]
    if fmt == WeekendFormat.SPRINT:
        required.insert(0, SessionKind.SPRINT)
    is_final = all(by_kind[kind].status == SessionStatus.COMPLETE for kind in required)
    race_time = by_kind[SessionKind.GRAND_PRIX].scheduled_at
    unresolved_after_timeout = bool(
        not is_final and race_time is not None and now > race_time + ACTIVE_EVENT_FALLBACK_TIMEOUT
    )
    if is_final:
        status = "complete"
    elif race_time is not None and now >= race_time:
        status = "awaiting_final_classification"
    else:
        scheduled = [item.scheduled_at for item in sessions if item.scheduled_at is not None]
        status = "upcoming" if scheduled and now < min(scheduled) else "live"

    active_candidates = [
        item for item in sessions
        if item.status in {
            SessionStatus.IN_PROGRESS,
            SessionStatus.PROVISIONAL,
            SessionStatus.PARTIAL,
            SessionStatus.FAILED,
            SessionStatus.MALFORMED,
        }
        and item.scheduled_at is not None
        and item.scheduled_at <= now
    ]
    active_session = max(active_candidates, key=lambda item: item.scheduled_at).kind if active_candidates else None
    next_candidates = [
        item for item in sessions
        if item.scheduled_at is not None
        and item.scheduled_at > now
        and item.status in {SessionStatus.PENDING, SessionStatus.NOT_SCHEDULED}
    ]
    next_session = min(next_candidates, key=lambda item: item.scheduled_at).kind if next_candidates else None
    diagnostics = tuple(
        item.diagnostic
        for item in sessions
        if item.status in {
            SessionStatus.PARTIAL,
            SessionStatus.PROVISIONAL,
            SessionStatus.FAILED,
            SessionStatus.MALFORMED,
        }
    )
    return WeekendState(
        event=event,
        race_name=str(row.get("raceName") or row.get("circuitName") or f"Round {event.round}"),
        format=fmt,
        sessions=tuple(sessions),
        active_session=active_session,
        next_session=next_session,
        status=status,
        is_final=is_final,
        unresolved_after_timeout=unresolved_after_timeout,
        diagnostics=diagnostics,
    )


def weekend_states(
    schedule: pd.DataFrame,
    *,
    results: pd.DataFrame | None,
    qualifying: pd.DataFrame | None,
    sprint: pd.DataFrame | None,
    effective_time: datetime | str | None = None,
    expected_participant_count: int | None = None,
) -> tuple[WeekendState, ...]:
    if schedule is None or schedule.empty or not {"season", "round"}.issubset(schedule.columns):
        return ()
    data = schedule.copy(deep=True)
    data["_season"] = pd.to_numeric(data["season"], errors="coerce")
    data["_round"] = pd.to_numeric(data["round"], errors="coerce")
    data = data.dropna(subset=["_season", "_round"]).sort_values(["_season", "_round"])
    now = coerce_utc_datetime(effective_time)

    def event_expected_participants(row: pd.Series) -> int | None:
        race_time = scheduled_session_timestamp(row, SessionKind.GRAND_PRIX)
        if race_time is None or now <= race_time + ACTIVE_EVENT_FALLBACK_TIMEOUT:
            return expected_participant_count
        event = EventKey(int(row["_season"]), int(row["_round"]))
        observed_counts: list[int] = []
        for frame in (results, qualifying, sprint):
            rows = _event_rows(frame, event)
            if rows.empty:
                continue
            observed = (
                int(rows["driverId"].nunique())
                if "driverId" in rows.columns
                else int(len(rows))
            )
            if observed > 0:
                observed_counts.append(observed)
        return max(observed_counts) if observed_counts else expected_participant_count

    return tuple(
        build_weekend_state(
            row,
            results=results,
            qualifying=qualifying,
            sprint=sprint,
            effective_time=effective_time,
            expected_participant_count=event_expected_participants(row),
        )
        for _, row in data.iterrows()
    )


def select_active_event(
    schedule: pd.DataFrame,
    *,
    results: pd.DataFrame | None,
    qualifying: pd.DataFrame | None,
    sprint: pd.DataFrame | None,
    effective_time: datetime | str | None = None,
    expected_participant_count: int | None = None,
) -> WeekendState | None:
    now = coerce_utc_datetime(effective_time)
    states = weekend_states(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=now,
        expected_participant_count=expected_participant_count,
    )
    if not states:
        return None
    unresolved: WeekendState | None = None
    for state in states:
        race_time = state.session(SessionKind.GRAND_PRIX).scheduled_at
        if state.is_final:
            continue
        if race_time is None:
            return state
        if now <= race_time + ACTIVE_EVENT_FALLBACK_TIMEOUT:
            return state
        unresolved = state
    return unresolved or states[-1]


def select_forecast_event(
    schedule: pd.DataFrame,
    *,
    results: pd.DataFrame | None,
    qualifying: pd.DataFrame | None,
    sprint: pd.DataFrame | None,
    effective_time: datetime | str | None = None,
    expected_participant_count: int | None = None,
    verified_target_event: EventKey | None = None,
) -> WeekendState | None:
    """Select the event whose performance is being forecast.

    Scoring safety may conservatively retain an older weekend until every
    supporting classification is coherent. Forecasting has a narrower rollover
    rule: a complete Grand Prix classification advances immediately. A target
    independently verified by the active Fantasy gameday may also advance the
    forecast while the scoring domain retains its last-good inputs.
    """
    states = weekend_states(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=effective_time,
        expected_participant_count=expected_participant_count,
    )
    if not states:
        return None
    active = select_active_event(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=effective_time,
        expected_participant_count=expected_participant_count,
    )
    if verified_target_event is not None:
        verified = next(
            (state for state in states if state.event == verified_target_event),
            None,
        )
        if verified is not None and (active is None or verified.event >= active.event):
            return verified
    if active is None:
        return None
    grand_prix = active.session(SessionKind.GRAND_PRIX)
    if grand_prix.status != SessionStatus.COMPLETE:
        return active
    return next((state for state in states if state.event > active.event), active)


def validate_weekend_snapshot(
    schedule: pd.DataFrame,
    *,
    results: pd.DataFrame | None,
    qualifying: pd.DataFrame | None,
    sprint: pd.DataFrame | None,
    effective_time: datetime | str | None = None,
    expected_participant_count: int | None = None,
) -> SnapshotValidation:
    states = weekend_states(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=effective_time,
        expected_participant_count=expected_participant_count,
    )
    active = select_active_event(
        schedule,
        results=results,
        qualifying=qualifying,
        sprint=sprint,
        effective_time=effective_time,
        expected_participant_count=expected_participant_count,
    )
    completed = tuple(state.event for state in states if state.is_final)
    unresolved = tuple(state.event for state in states if state.unresolved_after_timeout)
    unsafe_statuses = {
        SessionStatus.IN_PROGRESS,
        SessionStatus.PROVISIONAL,
        SessionStatus.PARTIAL,
        SessionStatus.MALFORMED,
        SessionStatus.FAILED,
    }
    partial_keys: list[EventKey] = []
    warnings: list[str] = []
    pending_partial_warnings: list[str] = []
    if unresolved:
        warnings.append(
            "An earlier event passed the conservative final-classification timeout "
            "without a complete result; it remains explicitly unresolved."
        )
    for state in states:
        unsafe_observed = [
            session for session in state.sessions
            if session.kind in {
                SessionKind.GRAND_PRIX,
                SessionKind.GRAND_PRIX_QUALIFYING,
                SessionKind.SPRINT,
            }
            and session.observed_row_count > 0
            and session.status in unsafe_statuses
        ]
        if unsafe_observed:
            partial_keys.append(state.event)
            warnings.append(
                f"{state.race_name} has incomplete live classification data; "
                "the round is excluded from completed scoring."
            )
        empty_late = [
            session for session in state.sessions
            if session.kind in {
                SessionKind.GRAND_PRIX,
                SessionKind.GRAND_PRIX_QUALIFYING,
                SessionKind.SPRINT,
            }
            and session.observed_row_count == 0
            and session.status in {
                SessionStatus.PARTIAL,
                SessionStatus.FAILED,
                SessionStatus.MALFORMED,
            }
        ]
        if empty_late:
            pending_partial_warnings.append(
                f"{state.race_name} has an expected session classification that is not yet available."
            )
    if partial_keys:
        status = "unsafe_partial"
        safe = False
    elif pending_partial_warnings:
        status = "valid_with_partial_noncritical_sources"
        safe = True
        warnings.extend(pending_partial_warnings)
    elif active is not None and not active.is_final:
        status = "valid_with_pending_sessions"
        safe = True
    else:
        status = "valid"
        safe = True
    return SnapshotValidation(
        status=status,
        safe_for_scoring=safe,
        warnings=tuple(warnings),
        active_weekend=active,
        completed_event_keys=tuple(sorted(set(completed))),
        excluded_partial_event_keys=tuple(sorted(set(partial_keys))),
        unresolved_event_keys=tuple(sorted(set(unresolved))),
    )


def filter_frame_to_completed_events(
    frame: pd.DataFrame,
    *,
    current_season: int,
    completed_event_keys: Iterable[EventKey],
) -> pd.DataFrame:
    if frame is None or frame.empty or not {"season", "round"}.issubset(frame.columns):
        return frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    data = frame.copy(deep=True)
    completed = set(completed_event_keys)
    seasons = pd.to_numeric(data["season"], errors="coerce")
    rounds = pd.to_numeric(data["round"], errors="coerce")
    keep = seasons != int(current_season)
    keep |= pd.Series(
        [
            EventKey(int(season), int(round_no)) in completed
            if pd.notna(season) and pd.notna(round_no)
            else False
            for season, round_no in zip(seasons, rounds)
        ],
        index=data.index,
    )
    return data[keep].copy()


def validate_deadline_candidate(
    candidate: Mapping[str, Any] | None,
    *,
    active_event: EventKey,
    schedule_row: Mapping[str, Any] | pd.Series,
    format: WeekendFormat,
) -> dict[str, Any]:
    row = schedule_row.to_dict() if isinstance(schedule_row, pd.Series) else dict(schedule_row)
    allowed = {"sprint", "sprint race"} if format == WeekendFormat.SPRINT else {
        "qualifying", "grand prix qualifying", "gp qualifying"
    }
    expected = (
        scheduled_session_timestamp(row, SessionKind.SPRINT)
        if format == WeekendFormat.SPRINT
        else scheduled_session_timestamp(row, SessionKind.GRAND_PRIX_QUALIFYING)
    )
    expected_name = str(row.get("raceName") or row.get("circuitName") or "").strip().casefold()
    original = dict(candidate or {})
    options = [
        dict(item) for item in original.get("team_lock_candidates", [])
        if isinstance(item, Mapping)
    ]
    if original.get("team_lock_deadline_utc"):
        options.append(original)

    valid: list[tuple[timedelta, dict[str, Any]]] = []
    rejection_reasons: list[str] = []
    for option in options:
        value = option.get("team_lock_deadline_utc")
        try:
            candidate_time = coerce_utc_datetime(value)
        except (TypeError, ValueError):
            rejection_reasons.append("Official deadline timestamp is malformed.")
            continue
        meaning = str(option.get("team_lock_session_type") or "").strip().casefold()
        if meaning not in allowed:
            rejection_reasons.append(
                f"Official session meaning {meaning or 'unknown'} is not allowed for a {format.value} weekend."
            )
            continue
        candidate_season = option.get("team_lock_season", option.get("season"))
        candidate_round = option.get("team_lock_round", option.get("round"))
        if candidate_season is not None and pd.notna(candidate_season):
            try:
                season_matches = int(pd.to_numeric(candidate_season, errors="raise")) == active_event.season
            except (TypeError, ValueError):
                season_matches = False
            if not season_matches:
                rejection_reasons.append("Official deadline belongs to a different season.")
                continue
        if candidate_round is not None and pd.notna(candidate_round):
            try:
                round_matches = int(pd.to_numeric(candidate_round, errors="raise")) == active_event.round
            except (TypeError, ValueError):
                round_matches = False
            if not round_matches:
                rejection_reasons.append("Official deadline belongs to a different round.")
                continue
        schedule_gameday = row.get("gameday_id", row.get("GamedayId"))
        candidate_gameday = option.get("team_lock_gameday_id")
        if (
            schedule_gameday is not None
            and pd.notna(schedule_gameday)
            and candidate_gameday is not None
            and pd.notna(candidate_gameday)
            and str(candidate_gameday) != str(schedule_gameday)
        ):
            rejection_reasons.append("Official deadline has a different gameday identity.")
            continue
        if expected is None or abs(candidate_time - expected) > timedelta(hours=18):
            rejection_reasons.append("Official deadline is not close to the active event lock session.")
            continue
        meeting = str(option.get("team_lock_meeting_name") or "").strip().casefold()
        if meeting and expected_name and meeting not in expected_name and expected_name not in meeting:
            rejection_reasons.append("Official deadline belongs to a different event.")
            continue
        valid.append((abs(candidate_time - expected), option))

    result = dict(original)
    result["team_lock_matched_event"] = (active_event.season, active_event.round)
    if not valid:
        result["team_lock_deadline_valid"] = False
        result["team_lock_validation_reason"] = (
            rejection_reasons[0] if rejection_reasons else "Official deadline is unavailable."
        )
        return result

    _, selected = min(valid, key=lambda item: item[0])
    result.update(selected)
    meaning = str(selected.get("team_lock_session_type") or "").strip().casefold()
    result["team_lock_deadline_source"] = "official_feed_playerstats_session_start"
    result["team_lock_deadline_valid"] = True
    result["team_lock_selected_session_meaning"] = meaning
    result["team_lock_validation_reason"] = "Official deadline matched the active event and allowed session."
    return result
