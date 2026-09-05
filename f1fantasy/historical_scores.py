from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping

import pandas as pd


DATA_VERSION = "historical_fantasy_scores_v3_recorded_2023_2026"
EARLIEST_PRODUCTION_SEASON = 2023
DEFAULT_CANONICAL_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/generated"
    / DATA_VERSION
    / "historical_fantasy_scores_2023_2026.csv"
)
SUPPORTED_SEASONS = (2023, 2024, 2025, 2026)
THIRD_PARTY_SEASONS = (2023, 2024, 2025)
CANONICAL_KEY = ["season", "round", "entity_type", "canonical_entity_id"]
CANONICAL_COLUMNS = [
    "season", "round", "event_name", "event_date", "entity_type",
    "canonical_entity_id", "source_entity_id", "name", "abbreviation",
    "constructor_name", "fantasy_points_total", "qualifying_points",
    "sprint_qualifying_points", "sprint_points", "race_points",
    "other_points", "price", "source_name", "source_reference",
    "source_licence", "authority_class", "is_official",
    "is_recorded_total", "is_reconstructed", "fantasy_score_origin",
    "data_version",
]

THIRD_PARTY_SOURCE = "jm1261/Fantasy-F1-League"
THIRD_PARTY_REFERENCE = "https://github.com/jm1261/Fantasy-F1-League"
THIRD_PARTY_LICENCE = "MIT"


@dataclass(frozen=True)
class EntityMeta:
    canonical_id: str
    name: str
    abbreviation: str
    constructor_id: str | None = None
    constructor_name: str | None = None


def _driver(
    canonical_id: str,
    name: str,
    abbreviation: str,
    constructor_id: str,
    constructor_name: str,
) -> EntityMeta:
    return EntityMeta(canonical_id, name, abbreviation, constructor_id, constructor_name)


DRIVERS_2023 = {
    "Hamilton": _driver("hamilton", "Lewis Hamilton", "HAM", "mercedes", "Mercedes"),
    "Russell": _driver("russell", "George Russell", "RUS", "mercedes", "Mercedes"),
    "Verstappen": _driver("max_verstappen", "Max Verstappen", "VER", "red_bull", "Red Bull"),
    "Perez": _driver("perez", "Sergio Pérez", "PER", "red_bull", "Red Bull"),
    "Leclerc": _driver("leclerc", "Charles Leclerc", "LEC", "ferrari", "Ferrari"),
    "Sainz": _driver("sainz", "Carlos Sainz", "SAI", "ferrari", "Ferrari"),
    "Norris": _driver("norris", "Lando Norris", "NOR", "mclaren", "McLaren"),
    "Piastri": _driver("piastri", "Oscar Piastri", "PIA", "mclaren", "McLaren"),
    "Alonso": _driver("alonso", "Fernando Alonso", "ALO", "aston_martin", "Aston Martin"),
    "Stroll": _driver("stroll", "Lance Stroll", "STR", "aston_martin", "Aston Martin"),
    "Ocon": _driver("ocon", "Esteban Ocon", "OCO", "alpine", "Alpine"),
    "Gasly": _driver("gasly", "Pierre Gasly", "GAS", "alpine", "Alpine"),
    "Tsunoda": _driver("tsunoda", "Yuki Tsunoda", "TSU", "alphatauri", "AlphaTauri"),
    "De Vries": _driver("de_vries", "Nyck de Vries", "DEV", "alphatauri", "AlphaTauri"),
    "Ricciardo": _driver("ricciardo", "Daniel Ricciardo", "RIC", "alphatauri", "AlphaTauri"),
    "Lawson": _driver("lawson", "Liam Lawson", "LAW", "alphatauri", "AlphaTauri"),
    "Albon": _driver("albon", "Alexander Albon", "ALB", "williams", "Williams"),
    "Sargeant": _driver("sargeant", "Logan Sargeant", "SAR", "williams", "Williams"),
    "Bottas": _driver("bottas", "Valtteri Bottas", "BOT", "alfa", "Alfa Romeo"),
    "Guanyu": _driver("zhou", "Guanyu Zhou", "ZHO", "alfa", "Alfa Romeo"),
    "Magnussen": _driver("kevin_magnussen", "Kevin Magnussen", "MAG", "haas", "Haas"),
    "Hulkenberg": _driver("hulkenberg", "Nico Hülkenberg", "HUL", "haas", "Haas"),
}

DRIVERS_2024 = {
    "Ocon": _driver("ocon", "Esteban Ocon", "OCO", "alpine", "Alpine"),
    "Gasly": _driver("gasly", "Pierre Gasly", "GAS", "alpine", "Alpine"),
    "Doohan": _driver("doohan", "Jack Doohan", "DOO", "alpine", "Alpine"),
    "Stroll": _driver("stroll", "Lance Stroll", "STR", "aston_martin", "Aston Martin"),
    "Alonso": _driver("alonso", "Fernando Alonso", "ALO", "aston_martin", "Aston Martin"),
    "Leclerc": _driver("leclerc", "Charles Leclerc", "LEC", "ferrari", "Ferrari"),
    "Sainz": _driver("sainz", "Carlos Sainz", "SAI", "ferrari", "Ferrari"),
    "Bearman": _driver("bearman", "Oliver Bearman", "BEA", "ferrari", "Ferrari"),
    "Magnussen": _driver("kevin_magnussen", "Kevin Magnussen", "MAG", "haas", "Haas"),
    "Hulkenberg": _driver("hulkenberg", "Nico Hülkenberg", "HUL", "haas", "Haas"),
    "Bottas": _driver("bottas", "Valtteri Bottas", "BOT", "sauber", "Kick Sauber"),
    "Guanyu": _driver("zhou", "Guanyu Zhou", "ZHO", "sauber", "Kick Sauber"),
    "Norris": _driver("norris", "Lando Norris", "NOR", "mclaren", "McLaren"),
    "Piastri": _driver("piastri", "Oscar Piastri", "PIA", "mclaren", "McLaren"),
    "Hamilton": _driver("hamilton", "Lewis Hamilton", "HAM", "mercedes", "Mercedes"),
    "Russell": _driver("russell", "George Russell", "RUS", "mercedes", "Mercedes"),
    "Tsunoda": _driver("tsunoda", "Yuki Tsunoda", "TSU", "rb", "RB"),
    "Ricciardo": _driver("ricciardo", "Daniel Ricciardo", "RIC", "rb", "RB"),
    "Lawson": _driver("lawson", "Liam Lawson", "LAW", "rb", "RB"),
    "Verstappen": _driver("max_verstappen", "Max Verstappen", "VER", "red_bull", "Red Bull"),
    "Perez": _driver("perez", "Sergio Pérez", "PER", "red_bull", "Red Bull"),
    "Albon": _driver("albon", "Alexander Albon", "ALB", "williams", "Williams"),
    "Sargeant": _driver("sargeant", "Logan Sargeant", "SAR", "williams", "Williams"),
    "Colapinto": _driver("colapinto", "Franco Colapinto", "COL", "williams", "Williams"),
}

DRIVERS_2025 = {
    "Gasly": _driver("gasly", "Pierre Gasly", "GAS", "alpine", "Alpine"),
    "Doohan": _driver("doohan", "Jack Doohan", "DOO", "alpine", "Alpine"),
    "Colapinto": _driver("colapinto", "Franco Colapinto", "COL", "alpine", "Alpine"),
    "Stroll": _driver("stroll", "Lance Stroll", "STR", "aston_martin", "Aston Martin"),
    "Alonso": _driver("alonso", "Fernando Alonso", "ALO", "aston_martin", "Aston Martin"),
    "Leclerc": _driver("leclerc", "Charles Leclerc", "LEC", "ferrari", "Ferrari"),
    "Hamilton": _driver("hamilton", "Lewis Hamilton", "HAM", "ferrari", "Ferrari"),
    "Ocon": _driver("ocon", "Esteban Ocon", "OCO", "haas", "Haas"),
    "Bearman": _driver("bearman", "Oliver Bearman", "BEA", "haas", "Haas"),
    "Hulkenberg": _driver("hulkenberg", "Nico Hülkenberg", "HUL", "sauber", "Kick Sauber"),
    "Bortoleto": _driver("bortoleto", "Gabriel Bortoleto", "BOR", "sauber", "Kick Sauber"),
    "Norris": _driver("norris", "Lando Norris", "NOR", "mclaren", "McLaren"),
    "Piastri": _driver("piastri", "Oscar Piastri", "PIA", "mclaren", "McLaren"),
    "Russell": _driver("russell", "George Russell", "RUS", "mercedes", "Mercedes"),
    "Antonelli": _driver("antonelli", "Andrea Kimi Antonelli", "ANT", "mercedes", "Mercedes"),
    "Tsunoda RB": _driver("tsunoda", "Yuki Tsunoda", "TSU", "rb", "Racing Bulls"),
    "Lawson": _driver("lawson", "Liam Lawson", "LAW", "rb", "Racing Bulls"),
    "Hadjar": _driver("hadjar", "Isack Hadjar", "HAD", "rb", "Racing Bulls"),
    "Verstappen": _driver("max_verstappen", "Max Verstappen", "VER", "red_bull", "Red Bull"),
    "Tsunoda": _driver("tsunoda", "Yuki Tsunoda", "TSU", "red_bull", "Red Bull"),
    "Lawson RBR": _driver("lawson", "Liam Lawson", "LAW", "red_bull", "Red Bull"),
    "Albon": _driver("albon", "Alexander Albon", "ALB", "williams", "Williams"),
    "Sainz": _driver("sainz", "Carlos Sainz", "SAI", "williams", "Williams"),
}

DRIVER_MAPS = {2023: DRIVERS_2023, 2024: DRIVERS_2024, 2025: DRIVERS_2025}

CONSTRUCTOR_MAPS = {
    2023: {
        "Alfa Romeo": EntityMeta("alfa", "Alfa Romeo", "ALF"),
        "AlphaTauri": EntityMeta("alphatauri", "AlphaTauri", "ALT"),
        "Alpine": EntityMeta("alpine", "Alpine", "ALP"),
        "Aston Martin": EntityMeta("aston_martin", "Aston Martin", "AMR"),
        "Ferrari": EntityMeta("ferrari", "Ferrari", "FER"),
        "Haas": EntityMeta("haas", "Haas", "HAS"),
        "McLaren": EntityMeta("mclaren", "McLaren", "MCL"),
        "Mercedes": EntityMeta("mercedes", "Mercedes", "MER"),
        "Red Bull": EntityMeta("red_bull", "Red Bull", "RBR"),
        "Williams": EntityMeta("williams", "Williams", "WIL"),
    },
    2024: {},
    2025: {},
}
CONSTRUCTOR_MAPS[2024] = {
    **{k: v for k, v in CONSTRUCTOR_MAPS[2023].items() if k not in {"Alfa Romeo", "AlphaTauri"}},
    "Kick Sauber": EntityMeta("sauber", "Kick Sauber", "SAU"),
    "RB": EntityMeta("rb", "RB", "RB"),
}
CONSTRUCTOR_MAPS[2025] = {
    **{k: v for k, v in CONSTRUCTOR_MAPS[2024].items() if k != "RB"},
    "Racing Bulls": EntityMeta("rb", "Racing Bulls", "RB"),
}

EVENT_FILES = {
    2023: ("Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", "Monaco", "Spain", "Canada", "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"),
    2024: ("Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore", "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"),
    2025: ("Australia", "China", "Japan", "Bahrain", "Saudi Arabia", "Miami", "Emilia Romagna", "Monaco", "Spain", "Canada", "Austria", "Great Britain", "Belgium", "Hungary", "Netherlands", "Italy", "Azerbaijan", "Singapore", "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"),
}


def _active_driver_labels(season: int, round_no: int) -> set[str]:
    if season == 2023:
        active = set(DRIVERS_2023) - {"De Vries", "Ricciardo", "Lawson"}
        active.add("De Vries" if round_no <= 10 else "Ricciardo" if round_no in {11, 12} or round_no >= 18 else "Lawson")
        return active
    if season == 2024:
        active = {
            "Ocon", "Gasly", "Stroll", "Alonso", "Leclerc", "Sainz",
            "Magnussen", "Hulkenberg", "Bottas", "Guanyu", "Norris", "Piastri",
            "Hamilton", "Russell", "Tsunoda", "Ricciardo", "Verstappen", "Perez",
            "Albon", "Sargeant",
        }
        replacements = {
            2: (("Sainz", "Bearman"),),
            16: (("Sargeant", "Colapinto"),),
            17: (("Sargeant", "Colapinto"), ("Magnussen", "Bearman")),
            18: (("Sargeant", "Colapinto"),),
            19: (("Sargeant", "Colapinto"), ("Ricciardo", "Lawson")),
            20: (("Sargeant", "Colapinto"), ("Ricciardo", "Lawson")),
            21: (("Sargeant", "Colapinto"), ("Ricciardo", "Lawson"), ("Magnussen", "Bearman")),
            22: (("Sargeant", "Colapinto"), ("Ricciardo", "Lawson")),
            23: (("Sargeant", "Colapinto"), ("Ricciardo", "Lawson")),
            24: (("Sargeant", "Colapinto"), ("Ricciardo", "Lawson"), ("Ocon", "Doohan")),
        }
        for old, new in replacements.get(round_no, ()):
            active.discard(old)
            active.add(new)
        return active
    if season == 2025:
        active = {
            "Gasly", "Doohan", "Stroll", "Alonso", "Leclerc", "Hamilton", "Ocon",
            "Bearman", "Hulkenberg", "Bortoleto", "Norris", "Piastri", "Russell",
            "Antonelli", "Tsunoda RB", "Hadjar", "Verstappen", "Lawson RBR",
            "Albon", "Sainz",
        }
        if round_no >= 3:
            active -= {"Tsunoda RB", "Lawson RBR"}
            active |= {"Lawson", "Tsunoda"}
        if round_no >= 7:
            active.discard("Doohan")
            active.add("Colapinto")
        return active
    raise ValueError(f"Unsupported third-party season: {season}")


def _bearman_2024_constructor(round_no: int) -> tuple[str, str]:
    return ("ferrari", "Ferrari") if round_no == 2 else ("haas", "Haas")


def _event_metadata(schedule: pd.DataFrame, season: int, round_no: int, fallback_name: str) -> tuple[str, Any]:
    if schedule is not None and not schedule.empty and {"round", "raceName"}.issubset(schedule.columns):
        rows = schedule[pd.to_numeric(schedule["round"], errors="coerce") == int(round_no)]
        if not rows.empty:
            row = rows.iloc[0]
            return str(row.get("raceName") or fallback_name), row.get("date", pd.NA)
    return fallback_name, pd.NA


def _base_row(
    *, season: int, round_no: int, event_name: str, event_date: Any,
    entity_type: str, source_label: str, meta: EntityMeta,
    total: Any, price: Any, source_reference: str,
) -> dict[str, Any]:
    return {
        "season": int(season), "round": int(round_no), "event_name": event_name,
        "event_date": event_date, "entity_type": entity_type,
        "canonical_entity_id": meta.canonical_id,
        "source_entity_id": f"{THIRD_PARTY_SOURCE}:{season}:{entity_type}:{source_label}",
        "name": meta.name, "abbreviation": meta.abbreviation,
        "constructor_name": meta.constructor_name if entity_type == "driver" else pd.NA,
        "fantasy_points_total": pd.to_numeric(total, errors="coerce"),
        "qualifying_points": pd.NA, "sprint_qualifying_points": pd.NA,
        "sprint_points": pd.NA, "race_points": pd.NA, "other_points": pd.NA,
        "price": pd.to_numeric(price, errors="coerce"),
        "source_name": THIRD_PARTY_SOURCE, "source_reference": source_reference,
        "source_licence": THIRD_PARTY_LICENCE, "authority_class": "third_party_recorded",
        "is_official": False, "is_recorded_total": True, "is_reconstructed": False,
        "fantasy_score_origin": "third_party_recorded", "data_version": DATA_VERSION,
    }


def normalise_third_party_recorded(
    raw_root: str | Path,
    schedules: Mapping[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Normalise retained MIT 2023–2025 scores without traversing archived years."""
    root = Path(raw_root)
    schedules = schedules or {}
    rows: list[dict[str, Any]] = []

    for season in (2023, 2024, 2025):
        for round_no, event_file in enumerate(EVENT_FILES[season], start=1):
            relative = f"Data/{season}/Lineup/{event_file}_Results.json"
            payload = json.loads((root / relative).read_text(encoding="utf-8"))
            event_name, event_date = _event_metadata(
                schedules.get(season, pd.DataFrame()), season, round_no,
                str((payload.get("Race") or [event_file])[0]),
            )
            active_labels = _active_driver_labels(season, round_no)
            if len(active_labels) != 20:
                raise ValueError(f"{season} round {round_no}: expected 20 active driver labels, got {len(active_labels)}")
            for label in sorted(active_labels):
                if label not in payload:
                    raise ValueError(f"{season} round {round_no}: missing active driver {label}")
                meta = DRIVER_MAPS[season][label]
                if season == 2024 and label == "Bearman":
                    constructor_id, constructor_name = _bearman_2024_constructor(round_no)
                    meta = EntityMeta(meta.canonical_id, meta.name, meta.abbreviation, constructor_id, constructor_name)
                total, price = payload[label]
                rows.append(_base_row(
                    season=season, round_no=round_no, event_name=event_name, event_date=event_date,
                    entity_type="driver", source_label=label, meta=meta, total=total, price=price,
                    source_reference=f"{THIRD_PARTY_REFERENCE}/blob/main/{relative.replace(' ', '%20')}",
                ))
            for label, meta in CONSTRUCTOR_MAPS[season].items():
                total, price = payload[label]
                rows.append(_base_row(
                    season=season, round_no=round_no, event_name=event_name, event_date=event_date,
                    entity_type="constructor", source_label=label, meta=meta, total=total, price=price,
                    source_reference=f"{THIRD_PARTY_REFERENCE}/blob/main/{relative.replace(' ', '%20')}",
                ))

    out = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    validate_canonical_scores(out, expected_seasons=THIRD_PARTY_SEASONS)
    return out.sort_values(CANONICAL_KEY, kind="stable").reset_index(drop=True)


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


DRIVER_TLA_TO_ID = {
    "ALB": "albon", "ALO": "alonso", "ANT": "antonelli", "BEA": "bearman",
    "BOR": "bortoleto", "BOT": "bottas", "COL": "colapinto", "DOO": "doohan",
    "GAS": "gasly", "HAD": "hadjar", "HAM": "hamilton", "HUL": "hulkenberg",
    "LAW": "lawson", "LEC": "leclerc", "LIN": "arvid_lindblad", "NOR": "norris",
    "OCO": "ocon", "PER": "perez", "PIA": "piastri", "RUS": "russell",
    "SAI": "sainz", "STR": "stroll", "TSU": "tsunoda", "VER": "max_verstappen",
}
CONSTRUCTOR_ALIAS_TO_ID = {
    "alpine": "alpine", "astonmartin": "aston_martin", "audi": "audi",
    "cadillac": "cadillac", "ferrari": "ferrari", "haas": "haas",
    "haasf1team": "haas", "mclaren": "mclaren", "mercedes": "mercedes",
    "rb": "rb", "rbf1team": "rb", "racingbulls": "rb", "redbull": "red_bull",
    "redbullracing": "red_bull",
    "williams": "williams",
}
CANONICAL_CONSTRUCTOR_NAMES = {
    "alpine": "Alpine", "aston_martin": "Aston Martin", "audi": "Audi",
    "cadillac": "Cadillac", "ferrari": "Ferrari", "haas": "Haas",
    "mclaren": "McLaren", "mercedes": "Mercedes", "rb": "Racing Bulls",
    "red_bull": "Red Bull", "williams": "Williams",
}


def current_entity_maps(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    results: pd.DataFrame | None = None,
) -> tuple[dict[int, EntityMeta], dict[int, EntityMeta], list[str]]:
    """Map official source IDs using explicit TLA/team aliases plus season context."""
    results = results.copy(deep=True) if results is not None else pd.DataFrame()
    warnings: list[str] = []
    driver_map: dict[int, EntityMeta] = {}
    for row in players.to_dict("records"):
        source_id = int(row.get("playerId", row.get("id")))
        tla = str(row.get("tla") or "").upper()
        canonical_id = DRIVER_TLA_TO_ID.get(tla)
        name = str(row.get("name") or tla or source_id)
        team_name = str(row.get("team") or "")
        constructor_id = CONSTRUCTOR_ALIAS_TO_ID.get(_normalise_text(team_name))
        if canonical_id is None and not results.empty:
            candidates = results[
                (results.get("driver", pd.Series(index=results.index, dtype=object)).map(_normalise_text) == _normalise_text(name))
                & (results.get("constructor", pd.Series(index=results.index, dtype=object)).map(_normalise_text) == _normalise_text(team_name))
            ]
            ids = candidates.get("driverId", pd.Series(dtype=object)).dropna().astype(str).unique()
            canonical_id = ids[0] if len(ids) == 1 else None
        if canonical_id is None:
            warnings.append(f"Unresolved official driver PlayerId {source_id} ({name})")
            continue
        if not results.empty and "driverId" in results.columns:
            canonical_names = results.loc[
                results["driverId"].astype(str) == canonical_id, "driver"
            ].dropna().astype(str).unique()
            if len(canonical_names):
                name = canonical_names[-1]
        driver_map[source_id] = EntityMeta(canonical_id, name, tla, constructor_id, team_name or None)

    constructor_map: dict[int, EntityMeta] = {}
    for row in teams.to_dict("records"):
        source_id = int(row.get("teamId", row.get("id")))
        name = str(row.get("name") or row.get("tla") or source_id)
        canonical_id = CONSTRUCTOR_ALIAS_TO_ID.get(_normalise_text(name))
        if canonical_id is None:
            warnings.append(f"Unresolved official constructor PlayerId {source_id} ({name})")
            continue
        constructor_map[source_id] = EntityMeta(
            canonical_id,
            CANONICAL_CONSTRUCTOR_NAMES.get(canonical_id, name),
            str(row.get("tla") or ""),
        )
    return driver_map, constructor_map, warnings


def normalise_official_playerstats(
    driver_race_points: pd.DataFrame,
    constructor_race_points: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    results: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    driver_map, constructor_map, warnings = current_entity_maps(players, teams, results)
    rows: list[dict[str, Any]] = []
    schedule = schedule if schedule is not None else pd.DataFrame()
    for entity_type, frame, mapping in (
        ("driver", driver_race_points, driver_map),
        ("constructor", constructor_race_points, constructor_map),
    ):
        if frame is None or frame.empty:
            continue
        for source in frame.copy(deep=True).to_dict("records"):
            season = pd.to_numeric(source.get("season"), errors="coerce")
            round_no = pd.to_numeric(source.get("round"), errors="coerce")
            total = pd.to_numeric(source.get("fantasy_points"), errors="coerce")
            played = pd.to_numeric(source.get("is_played"), errors="coerce")
            source_id = pd.to_numeric(source.get("PlayerId"), errors="coerce")
            if pd.isna(season) or int(season) != 2026 or pd.isna(round_no) or pd.isna(total) or played != 1 or pd.isna(source_id):
                continue
            meta = mapping.get(int(source_id))
            if meta is None:
                warnings.append(f"Skipped unresolved official {entity_type} PlayerId {int(source_id)}")
                continue
            source_race_name = str(source.get("race_name") or f"Round {int(round_no)}")
            canonical_round = int(round_no)
            if not schedule.empty and {"round", "raceName"}.issubset(schedule.columns):
                source_name_key = _normalise_text(source_race_name).replace("grandprix", "")
                schedule_keys = schedule["raceName"].map(
                    lambda value: _normalise_text(value).replace("grandprix", "")
                )
                matched = schedule[
                    schedule_keys.map(
                        lambda value: bool(value and source_name_key)
                        and (value in source_name_key or source_name_key in value)
                    )
                ]
                if len(matched) == 1:
                    canonical_round = int(matched.iloc[0]["round"])
            event_name, event_date = _event_metadata(
                schedule, 2026, canonical_round, source_race_name
            )
            component_values = {
                key: pd.to_numeric(source.get(key), errors="coerce")
                for key in ("qualifying_points", "sprint_qualifying_points", "sprint_points", "race_points")
            }
            schedule_row = schedule[
                pd.to_numeric(schedule.get("round", pd.Series(index=schedule.index, dtype=float)), errors="coerce") == canonical_round
            ] if not schedule.empty else pd.DataFrame()
            sprint_expected = bool(
                not schedule_row.empty
                and pd.notna(schedule_row.iloc[0].get("sprint_date"))
            )
            complete_components = (
                pd.notna(component_values["qualifying_points"])
                and pd.notna(component_values["race_points"])
                and (
                    not sprint_expected
                    or pd.notna(component_values["sprint_points"])
                    or pd.notna(component_values["sprint_qualifying_points"])
                )
            )
            component_sum = sum(
                float(value)
                for value in component_values.values()
                if pd.notna(value)
            )
            row = {
                "season": 2026, "round": canonical_round, "event_name": event_name,
                "event_date": event_date, "entity_type": entity_type,
                "canonical_entity_id": meta.canonical_id,
                "source_entity_id": str(int(source_id)), "name": meta.name,
                "abbreviation": meta.abbreviation,
                "constructor_name": meta.constructor_name if entity_type == "driver" else pd.NA,
                "fantasy_points_total": float(total),
                "qualifying_points": component_values["qualifying_points"] if pd.notna(component_values["qualifying_points"]) else pd.NA,
                "sprint_qualifying_points": component_values["sprint_qualifying_points"] if pd.notna(component_values["sprint_qualifying_points"]) else pd.NA,
                "sprint_points": component_values["sprint_points"] if pd.notna(component_values["sprint_points"]) else pd.NA,
                "race_points": component_values["race_points"] if pd.notna(component_values["race_points"]) else pd.NA,
                "other_points": float(total) - component_sum if complete_components else pd.NA,
                "price": pd.to_numeric(source.get("price"), errors="coerce"),
                "source_name": "Formula 1 Fantasy playerstats",
                "source_reference": f"https://fantasy.formula1.com/feeds/popup/playerstats_{int(source_id)}.json",
                "source_licence": "Official public feed; Formula 1 terms apply",
                "authority_class": "official", "is_official": True,
                "is_recorded_total": True, "is_reconstructed": False,
                "fantasy_score_origin": "official_recorded", "data_version": DATA_VERSION,
            }
            rows.append(row)
    out = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    if not out.empty:
        validate_canonical_scores(out, expected_seasons=(2026,))
    return out.sort_values(CANONICAL_KEY, kind="stable").reset_index(drop=True), sorted(set(warnings))


def resolve_score_precedence(*frames: pd.DataFrame) -> pd.DataFrame:
    usable: list[pd.DataFrame] = []
    for source_order, frame in enumerate(frames):
        if frame is None or frame.empty:
            continue
        copied = frame.copy(deep=True)
        copied["_source_order"] = int(source_order)
        usable.append(copied)
    if not usable:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    combined = pd.concat(usable, ignore_index=True)
    ranks = {"official_recorded": 3, "third_party_recorded": 2, "reconstructed": 1, "missing": 0}
    combined["_authority_rank"] = combined["fantasy_score_origin"].map(ranks).fillna(-1)
    combined = combined.sort_values(
        CANONICAL_KEY + ["_authority_rank", "_source_order"],
        ascending=[True, True, True, True, False, False],
        kind="stable",
    )
    out = combined.drop_duplicates(CANONICAL_KEY, keep="first").drop(columns=["_authority_rank", "_source_order"])
    out = out.reindex(columns=CANONICAL_COLUMNS)
    validate_canonical_scores(out)
    return out.reset_index(drop=True)


def validate_canonical_scores(
    frame: pd.DataFrame,
    expected_seasons: Iterable[int] | None = None,
) -> dict[str, Any]:
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing canonical score columns: {missing}")
    seasons = set(pd.to_numeric(frame["season"], errors="raise").astype(int).unique())
    if not seasons.issubset(set(SUPPORTED_SEASONS)):
        raise ValueError(f"Canonical data contains unsupported seasons: {sorted(seasons)}")
    versions = set(frame["data_version"].dropna().astype(str).unique())
    if versions and versions != {DATA_VERSION}:
        raise ValueError(
            f"Canonical data version mismatch: expected {DATA_VERSION}, got {sorted(versions)}"
        )
    if expected_seasons is not None and seasons != set(int(year) for year in expected_seasons):
        raise ValueError(f"Expected seasons {sorted(expected_seasons)}, got {sorted(seasons)}")
    duplicates = int(frame.duplicated(CANONICAL_KEY, keep=False).sum())
    if duplicates:
        raise ValueError(f"Canonical data has {duplicates} duplicate-key rows")
    if not set(frame["entity_type"].dropna().unique()).issubset({"driver", "constructor"}):
        raise ValueError("Unknown entity_type in canonical data")
    numeric = pd.to_numeric(frame["fantasy_points_total"], errors="coerce")
    if numeric.isna().any():
        raise ValueError("Recorded canonical rows must contain numeric total points")
    residual_known = pd.to_numeric(frame["other_points"], errors="coerce").notna()
    if residual_known.any():
        components = frame.loc[
            residual_known,
            ["qualifying_points", "sprint_qualifying_points", "sprint_points", "race_points", "other_points"],
        ].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        delta = components.sum(axis=1) - numeric.loc[residual_known]
        if delta.abs().gt(1e-9).any():
            raise ValueError("Recorded component totals do not agree with fantasy_points_total")
    return {
        "rows": int(len(frame)), "seasons": sorted(int(season) for season in seasons), "duplicates": duplicates,
        "official_rows": int(frame["is_official"].fillna(False).astype(bool).sum()),
        "reconstructed_rows": int(frame["is_reconstructed"].fillna(False).astype(bool).sum()),
    }


def load_canonical_scores(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    frame = pd.read_csv(path)
    for column in ("event_date", "constructor_name", "qualifying_points", "sprint_qualifying_points", "sprint_points", "race_points", "other_points", "price"):
        if column in frame.columns:
            frame[column] = frame[column].where(frame[column].notna(), pd.NA)
    validate_canonical_scores(frame)
    return frame.reindex(columns=CANONICAL_COLUMNS)


def canonical_playerstats_observations(
    recorded: pd.DataFrame,
    season: int,
    entity_type: str,
) -> pd.DataFrame:
    """Expose recorded totals in the per-race observation schema used by the app.

    The canonical dataset remains the authority.  Component values stay missing
    when the source did not record them; a valid total is sufficient for a race
    to participate in points-based analysis.
    """
    if entity_type not in {"driver", "constructor"}:
        raise ValueError("entity_type must be 'driver' or 'constructor'.")
    columns = [
        "PlayerId",
        "asset_type",
        "season",
        "round",
        "race_name",
        "fantasy_points",
        "qualifying_points",
        "sprint_qualifying_points",
        "race_points",
        "sprint_points",
        "price",
        "is_played",
        "canonical_entity_id",
        "fantasy_score_origin",
        "data_version",
    ]
    if recorded is None or recorded.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "season",
        "round",
        "entity_type",
        "source_entity_id",
        "fantasy_points_total",
        "fantasy_score_origin",
    }
    missing = sorted(required - set(recorded.columns))
    if missing:
        raise ValueError(f"Canonical observations are missing columns: {missing}")

    data = recorded.copy(deep=True)
    data = data[
        pd.to_numeric(data["season"], errors="coerce").eq(int(season))
        & data["entity_type"].astype(str).eq(entity_type)
        & pd.to_numeric(data["fantasy_points_total"], errors="coerce").notna()
        & data["fantasy_score_origin"].isin(
            {"official_recorded", "third_party_recorded"}
        )
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(index=data.index)
    out["PlayerId"] = data["source_entity_id"].astype("string")
    out["asset_type"] = entity_type
    out["season"] = pd.to_numeric(data["season"], errors="raise").astype(int)
    out["round"] = pd.to_numeric(data["round"], errors="raise").astype(int)
    out["race_name"] = data.get("event_name", pd.Series(index=data.index, dtype=object))
    out["fantasy_points"] = pd.to_numeric(data["fantasy_points_total"], errors="coerce")
    for column in (
        "qualifying_points",
        "sprint_qualifying_points",
        "race_points",
        "sprint_points",
        "price",
    ):
        out[column] = pd.to_numeric(
            data.get(column, pd.Series(index=data.index, dtype=float)), errors="coerce"
        )
    out["is_played"] = 1
    out["canonical_entity_id"] = data.get(
        "canonical_entity_id", pd.Series(index=data.index, dtype=object)
    )
    out["fantasy_score_origin"] = data["fantasy_score_origin"].astype(str)
    out["data_version"] = data.get(
        "data_version", pd.Series(index=data.index, dtype=object)
    )
    return out.reindex(columns=columns).sort_values(
        ["season", "round", "PlayerId"], kind="stable"
    ).reset_index(drop=True)


def canonical_market_snapshot(
    recorded: pd.DataFrame,
    season: int,
    *,
    minimum_drivers: int = 15,
    minimum_constructors: int = 8,
) -> dict[str, Any]:
    """Build a historical event-price reference from the newest complete round.

    The returned prices belong to that completed event and must not be used as
    current-market prices for an upcoming Fantasy round.
    """
    data = recorded.copy(deep=True)
    required = {
        "season", "round", "entity_type", "source_entity_id", "name",
        "abbreviation", "constructor_name", "price", "is_official", "data_version",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Canonical market snapshot is missing columns: {missing}")
    if data.empty:
        raise ValueError("Canonical market data is empty.")
    official = data[
        (pd.to_numeric(data["season"], errors="coerce") == int(season))
        & data["is_official"].fillna(False).astype(bool)
    ].copy()
    official["round"] = pd.to_numeric(official["round"], errors="coerce")
    official["source_entity_id"] = pd.to_numeric(official["source_entity_id"], errors="coerce")
    official["price"] = pd.to_numeric(official["price"], errors="coerce")
    official = official[
        official["round"].notna()
        & official["source_entity_id"].notna()
        & official["price"].notna()
        & official["price"].gt(0)
        & official["name"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    candidate_rounds: list[int] = []
    for round_no, rows in official.groupby("round", sort=True):
        driver_count = rows.loc[rows["entity_type"].eq("driver"), "source_entity_id"].nunique()
        constructor_count = rows.loc[
            rows["entity_type"].eq("constructor"), "source_entity_id"
        ].nunique()
        if driver_count >= int(minimum_drivers) and constructor_count >= int(minimum_constructors):
            candidate_rounds.append(int(round_no))
    if not candidate_rounds:
        raise ValueError(
            f"No complete official canonical market snapshot exists for {int(season)}."
        )
    snapshot_round = max(candidate_rounds)
    snapshot = official[official["round"].eq(snapshot_round)].copy()
    drivers = snapshot[snapshot["entity_type"].eq("driver")].copy()
    constructors = snapshot[snapshot["entity_type"].eq("constructor")].copy()
    if drivers["source_entity_id"].duplicated().any() or constructors["source_entity_id"].duplicated().any():
        raise ValueError("Canonical market snapshot contains duplicate source entity IDs.")
    players = pd.DataFrame(
        {
            "playerId": drivers["source_entity_id"].astype(int),
            "name": drivers["name"].astype(str),
            "price": drivers["price"].astype(float),
            "team": drivers["constructor_name"].astype(str),
            "driver_reference": drivers["canonical_entity_id"].astype(str),
            "tla": drivers["abbreviation"].astype(str),
            "f1_player_id": drivers["source_entity_id"].astype(int),
        }
    ).reset_index(drop=True)
    teams = pd.DataFrame(
        {
            "teamId": constructors["source_entity_id"].astype(int),
            "name": constructors["name"].astype(str),
            "price": constructors["price"].astype(float),
            "tla": constructors["abbreviation"].astype(str),
            "f1_team_id": constructors["source_entity_id"].astype(int),
        }
    ).reset_index(drop=True)
    event_names = snapshot.get("event_name", pd.Series(dtype=object)).dropna().astype(str)
    event_name = event_names.iloc[0] if not event_names.empty else f"Round {snapshot_round}"
    versions = snapshot["data_version"].dropna().astype(str).unique().tolist()
    return {
        "season": int(season),
        "round": snapshot_round,
        "event_name": event_name,
        "data_version": versions[0] if len(versions) == 1 else DATA_VERSION,
        "price_semantics": "historical_event_price",
        "players": players,
        "teams": teams,
    }


def _recorded_model_rows(recorded: pd.DataFrame, entity_type: str, proxy: pd.DataFrame) -> pd.DataFrame:
    exact = recorded[recorded["entity_type"] == entity_type].copy(deep=True)
    if exact.empty:
        return pd.DataFrame()
    if entity_type == "driver":
        proxy_key = ["season", "round", "driverId"]
        exact = exact.rename(columns={"canonical_entity_id": "driverId", "name": "driver"})
        model_total = "weekend_points"
    else:
        proxy_key = ["season", "round", "constructorId"]
        exact = exact.rename(columns={"canonical_entity_id": "constructorId", "name": "constructor"})
        model_total = "constructor_weekend_points"
    structural = proxy.copy(deep=True)
    if not structural.empty:
        keep = [column for column in structural.columns if column not in exact.columns or column in proxy_key]
        exact = exact.merge(structural[keep].drop_duplicates(proxy_key), on=proxy_key, how="left", validate="one_to_one")
    exact[model_total] = pd.to_numeric(exact["fantasy_points_total"], errors="coerce")
    exact["weekend_points"] = exact[model_total]
    exact["circuitName"] = exact.get("circuitName", exact["event_name"]).fillna(exact["event_name"])
    exact["fantasy_score_origin"] = exact["fantasy_score_origin"].fillna("missing")
    for canonical, model in (("qualifying_points", "qualifying_points"), ("qualifying_points", "quali_points"), ("sprint_points", "sprint_points"), ("race_points", "race_points")):
        exact[model] = pd.to_numeric(exact[canonical], errors="coerce")
    defaults = {
        "q2_reached": 0, "q3_reached": 0, "is_dnf": 0, "is_dsq": 0,
        "has_fastest_lap": 0, "sprint_is_dnf": 0, "sprint_is_dsq": 0,
        "sprint_applicable": False, "sprint_observed": False, "dnf_drivers": 0,
        "sprint_dnf_drivers": 0, "dnf_rate": 0.0,
    }
    for column, default in defaults.items():
        if column not in exact.columns:
            exact[column] = default
        else:
            exact[column] = exact[column].fillna(default)
    if entity_type == "driver":
        constructor_lookup = {
            meta.constructor_name: meta.constructor_id
            for mapping in DRIVER_MAPS.values() for meta in mapping.values()
            if meta.constructor_name and meta.constructor_id
        }
        canonical_constructor_ids = exact["constructor_name"].map(
            lambda value: constructor_lookup.get(value)
            or CONSTRUCTOR_ALIAS_TO_ID.get(_normalise_text(value))
        )
        exact["constructor"] = exact.get("constructor", exact["constructor_name"]).fillna(exact["constructor_name"])
        exact["constructorId"] = exact.get("constructorId", canonical_constructor_ids).fillna(canonical_constructor_ids)
    return exact


def apply_recorded_scores_to_model(
    proxy_driver_weekends: pd.DataFrame,
    recorded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replace covered model totals; reconstruction survives only for uncovered keys."""
    proxy = proxy_driver_weekends.copy(deep=True)
    if proxy.empty or recorded.empty:
        if not proxy.empty:
            proxy["fantasy_score_origin"] = "reconstructed"
        return proxy, pd.DataFrame(), {"recorded_driver_rows": 0, "recorded_constructor_rows": 0, "unexpected_reconstruction_rows": int(len(proxy))}
    proxy["fantasy_score_origin"] = "reconstructed"
    recorded = recorded[recorded["season"].isin(SUPPORTED_SEASONS)].copy()
    driver_rows = _recorded_model_rows(recorded, "driver", proxy)
    proxy_ctor = pd.DataFrame()
    if not proxy.empty:
        from f1fantasy.model import _constructor_round_points
        proxy_ctor = _constructor_round_points(proxy)
        proxy_ctor["fantasy_score_origin"] = "reconstructed"
    constructor_rows = _recorded_model_rows(recorded, "constructor", proxy_ctor)

    def replace(base: pd.DataFrame, exact: pd.DataFrame, id_col: str, count_threshold: int) -> pd.DataFrame:
        if exact.empty:
            return base
        counts = exact.groupby(["season", "round"])[id_col].nunique()
        covered = {key for key, count in counts.items() if int(count) >= count_threshold}
        base_keys = list(zip(pd.to_numeric(base["season"], errors="coerce"), pd.to_numeric(base["round"], errors="coerce")))
        keep = [tuple(map(int, key)) not in covered if all(pd.notna(value) for value in key) else True for key in base_keys]
        residual = base.loc[keep].copy()
        if covered:
            exact_full = exact[[tuple(map(int, key)) in covered for key in zip(exact["season"], exact["round"])]].copy()
        else:
            exact_full = pd.DataFrame(columns=exact.columns)
        partial = exact[[tuple(map(int, key)) not in covered for key in zip(exact["season"], exact["round"])]].copy()
        if not partial.empty and not residual.empty:
            partial_keys = set(zip(partial["season"].astype(int), partial["round"].astype(int), partial[id_col].astype(str)))
            residual = residual[[
                (int(season), int(round_no), str(entity_id)) not in partial_keys
                for season, round_no, entity_id in zip(residual["season"], residual["round"], residual[id_col])
            ]]
        return pd.concat([residual, exact_full, partial], ignore_index=True, sort=False)

    # Once a source has recorded an event for an asset type, that event is an
    # exact-data event. Per-asset gaps remain missing instead of being backfilled
    # by reconstructed classifications.
    driver_out = replace(proxy, driver_rows, "driverId", 1)
    constructor_out = replace(proxy_ctor, constructor_rows, "constructorId", 1)
    covered_proxy = driver_out[driver_out["season"].isin(SUPPORTED_SEASONS)]
    unexpected = covered_proxy[covered_proxy["fantasy_score_origin"] == "reconstructed"]
    diagnostics = {
        "historical_fantasy_data_version": DATA_VERSION,
        "recorded_driver_rows": int((driver_out["fantasy_score_origin"] != "reconstructed").sum()),
        "recorded_constructor_rows": int((constructor_out["fantasy_score_origin"] != "reconstructed").sum()),
        "unexpected_reconstruction_rows": int(len(unexpected)),
        "unexpected_reconstruction_event_keys": sorted({(int(row.season), int(row.round)) for row in unexpected.itertuples()}),
    }
    return driver_out, constructor_out, diagnostics


def coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["season", "entity_type", "rows", "races", "entities", "component_rows", "price_rows", "official_rows", "reconstructed_rows"])
    data = frame.copy(deep=True)
    data["has_components"] = data[["qualifying_points", "sprint_qualifying_points", "sprint_points", "race_points", "other_points"]].notna().any(axis=1)
    data["has_price"] = pd.to_numeric(data["price"], errors="coerce").notna()
    return data.groupby(["season", "entity_type"], as_index=False).agg(
        rows=("canonical_entity_id", "size"), races=("round", "nunique"),
        entities=("canonical_entity_id", "nunique"), component_rows=("has_components", "sum"),
        price_rows=("has_price", "sum"), official_rows=("is_official", "sum"),
        reconstructed_rows=("is_reconstructed", "sum"),
    )


def approximation_comparison(recorded: pd.DataFrame, reconstructed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = recorded[[*CANONICAL_KEY, "name", "event_name", "fantasy_points_total"]].copy()
    approx = reconstructed.copy(deep=True)
    if "entity_type" not in approx.columns:
        raise ValueError("Reconstructed comparison rows require entity_type")
    merged = exact.merge(
        approx[[*CANONICAL_KEY, "fantasy_points_total"]].rename(columns={"fantasy_points_total": "approximate_points"}),
        on=CANONICAL_KEY, how="inner", validate="one_to_one",
    )
    merged["recorded_points"] = pd.to_numeric(merged.pop("fantasy_points_total"), errors="coerce")
    merged["approximate_points"] = pd.to_numeric(merged["approximate_points"], errors="coerce")
    merged["difference"] = merged["recorded_points"] - merged["approximate_points"]
    merged["absolute_difference"] = merged["difference"].abs()
    summary_rows: list[dict[str, Any]] = []
    for (season, entity_type), group in merged.groupby(["season", "entity_type"]):
        summary_rows.append({
            "season": int(season), "entity_type": entity_type, "number_of_rows": int(len(group)),
            "number_changed": int(group["absolute_difference"].gt(1e-12).sum()),
            "mean_absolute_difference": float(group["absolute_difference"].mean()),
            "median_absolute_difference": float(group["absolute_difference"].median()),
            "maximum_absolute_difference": float(group["absolute_difference"].max()),
            "correlation": float(group[["recorded_points", "approximate_points"]].corr().iloc[0, 1]) if len(group) > 1 else pd.NA,
        })
    details = merged.sort_values("absolute_difference", ascending=False, kind="stable")
    return pd.DataFrame(summary_rows), details
