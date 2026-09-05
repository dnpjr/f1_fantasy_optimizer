from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any

import pandas as pd

from f1fantasy.historical_scores import DRIVER_TLA_TO_ID


@dataclass(frozen=True)
class FantasyAssetIdentity:
    fantasy_asset_id: int
    human_driver_id: str
    history_driver_id: str | None
    driver_reference: str | None
    tla: str | None
    display_name: str
    team_id: int | None
    team_name: str | None
    active: bool
    match_method: str
    match_status: str
    diagnostic: str | None = None


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _clean_optional(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _integer_or_none(value: Any) -> int | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else int(parsed)


def _history_catalogue(history: pd.DataFrame | None) -> dict[str, Any]:
    frame = history.copy(deep=True) if isinstance(history, pd.DataFrame) else pd.DataFrame()
    if frame.empty or "driverId" not in frame.columns:
        return {"ids": set(), "names": {}, "tlas": {}, "f1_ids": {}}
    ids = set(frame["driverId"].dropna().astype(str))
    names: dict[str, set[str]] = {}
    for row in frame.to_dict("records"):
        driver_id = _clean_optional(row.get("driverId"))
        if not driver_id:
            continue
        for name_column in ("driver", "name"):
            name_key = _normalise_text(row.get(name_column))
            if name_key:
                names.setdefault(name_key, set()).add(driver_id)
    tlas: dict[str, set[str]] = {}
    for column in ("code", "tla", "abbreviation"):
        if column not in frame.columns:
            continue
        for row in frame[["driverId", column]].dropna().drop_duplicates().to_dict("records"):
            tlas.setdefault(str(row[column]).strip().upper(), set()).add(str(row["driverId"]))
    f1_ids: dict[int, set[str]] = {}
    for column in ("f1_player_id", "F1PlayerId"):
        if column not in frame.columns:
            continue
        for row in frame[["driverId", column]].dropna().drop_duplicates().to_dict("records"):
            f1_id = _integer_or_none(row[column])
            if f1_id is not None and f1_id >= 0:
                f1_ids.setdefault(f1_id, set()).add(str(row["driverId"]))
    return {"ids": ids, "names": names, "tlas": tlas, "f1_ids": f1_ids}


def _single(values: set[str] | None) -> str | None:
    return next(iter(values)) if values is not None and len(values) == 1 else None


def _row_candidate(row: dict[str, Any], catalogue: dict[str, Any]) -> tuple[str | None, str, int, str | None]:
    explicit_id = _clean_optional(row.get("human_driver_id"))
    if explicit_id is None:
        explicit_id = _clean_optional(row.get("history_driver_id"))
    if explicit_id and explicit_id in catalogue["ids"]:
        return explicit_id, "canonical_driver_id", 1, None

    f1_id = _integer_or_none(row.get("f1_player_id", row.get("F1PlayerId")))
    if f1_id is not None and f1_id >= 0:
        match = _single(catalogue["f1_ids"].get(f1_id))
        if match:
            return match, "f1_player_id", 2, None

    tla = str(row.get("tla") or row.get("DriverTLA") or "").strip().upper()
    if tla:
        mapped = DRIVER_TLA_TO_ID.get(tla)
        if mapped:
            return mapped, "tla", 3, None
        match = _single(catalogue["tlas"].get(tla))
        if match:
            return match, "tla", 3, None

    name = row.get("name", row.get("FUllName", row.get("FullName")))
    name_key = _normalise_text(name)
    exact = _single(catalogue["names"].get(name_key)) if name_key else None
    if exact:
        return exact, "normalised_name", 4, None

    if name_key and catalogue["names"]:
        scores = sorted(
            (
                SequenceMatcher(None, name_key, candidate).ratio(),
                candidate,
            )
            for candidate in catalogue["names"]
        )
        best_score, best_name = scores[-1]
        second_score = scores[-2][0] if len(scores) > 1 else 0.0
        fuzzy = _single(catalogue["names"].get(best_name))
        if fuzzy and best_score >= 0.86 and (best_score - second_score) >= 0.04:
            return fuzzy, "fuzzy_name", 5, f"Fuzzy name fallback score {best_score:.3f}."
        if best_score >= 0.86:
            return None, "ambiguous_fuzzy_name", 5, "Fuzzy name candidates were ambiguous."
    return None, "unresolved", 99, "No deterministic history identity matched."


def build_player_identity_map(
    player_assets: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map Fantasy assets to human drivers while retaining asset-level identity."""
    assets = player_assets.copy(deep=True)
    columns = [field.name for field in FantasyAssetIdentity.__dataclass_fields__.values()]
    if assets.empty:
        return pd.DataFrame(columns=columns)
    catalogue = _history_catalogue(history)
    records = assets.to_dict("records")
    candidates = [_row_candidate(row, catalogue) for row in records]
    reference_groups: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        reference = _normalise_text(row.get("driver_reference", row.get("DriverReference")))
        if reference:
            reference_groups.setdefault(reference, []).append(index)

    resolved_groups: dict[int, tuple[str | None, str, str, str | None]] = {}
    for reference, indexes in reference_groups.items():
        best_priority = min(candidates[index][2] for index in indexes)
        best_ids = {
            candidates[index][0]
            for index in indexes
            if candidates[index][2] == best_priority and candidates[index][0]
        }
        if len(best_ids) == 1:
            history_id = next(iter(best_ids))
            method = "driver_reference" if len(indexes) > 1 else candidates[indexes[0]][1]
            for index in indexes:
                resolved_groups[index] = (history_id, method, "matched", candidates[index][3])
        elif len(best_ids) > 1:
            diagnostic = (
                f"DriverReference {reference} has conflicting human candidates: "
                f"{', '.join(sorted(best_ids))}."
            )
            for index in indexes:
                resolved_groups[index] = (None, "driver_reference", "ambiguous", diagnostic)
        else:
            for index in indexes:
                resolved_groups[index] = (
                    None,
                    "driver_reference",
                    "unresolved",
                    f"DriverReference {reference} has no matching history identity.",
                )

    identities: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        asset_id = _integer_or_none(row.get("playerId", row.get("PlayerId", row.get("id"))))
        if asset_id is None:
            raise ValueError("Fantasy driver asset is missing a numeric asset ID.")
        reference = _clean_optional(row.get("driver_reference", row.get("DriverReference")))
        history_id, method, status, diagnostic = resolved_groups.get(
            index,
            (candidates[index][0], candidates[index][1], "matched" if candidates[index][0] else "unresolved", candidates[index][3]),
        )
        stable_reference = _normalise_text(reference)
        human_id = history_id or (
            f"driver-reference:{stable_reference}" if stable_reference else f"fantasy-asset:{asset_id}"
        )
        active_value = pd.to_numeric(
            row.get("is_active", row.get("IsActive", row.get("selectable", 1))),
            errors="coerce",
        )
        identity = FantasyAssetIdentity(
            fantasy_asset_id=asset_id,
            human_driver_id=human_id,
            history_driver_id=history_id,
            driver_reference=reference,
            tla=_clean_optional(row.get("tla", row.get("DriverTLA"))),
            display_name=str(row.get("name", row.get("FUllName", row.get("FullName", asset_id)))),
            team_id=_integer_or_none(row.get("team_id", row.get("TeamId"))),
            team_name=_clean_optional(row.get("team", row.get("TeamName"))),
            active=bool(pd.notna(active_value) and int(active_value) == 1),
            match_method=method,
            match_status=status,
            diagnostic=diagnostic,
        )
        identities.append(asdict(identity))
    return pd.DataFrame(identities, columns=columns).sort_values(
        "fantasy_asset_id", kind="stable"
    ).reset_index(drop=True)


def asset_to_human_identity(
    player_identity_map: pd.DataFrame,
    fantasy_asset_id: int,
) -> FantasyAssetIdentity | None:
    mapping = player_identity_map.copy(deep=True)
    if mapping.empty or "fantasy_asset_id" not in mapping.columns:
        return None
    rows = mapping[
        pd.to_numeric(mapping["fantasy_asset_id"], errors="coerce") == int(fantasy_asset_id)
    ]
    if rows.empty:
        return None
    values = rows.iloc[0].to_dict()
    return FantasyAssetIdentity(**{field: values.get(field) for field in FantasyAssetIdentity.__dataclass_fields__})


def human_assets(
    player_assets: pd.DataFrame,
    player_identity_map: pd.DataFrame,
    human_driver_id: str,
) -> pd.DataFrame:
    mapping = player_identity_map[
        player_identity_map.get("human_driver_id", pd.Series(index=player_identity_map.index, dtype=object)).astype(str)
        == str(human_driver_id)
    ].copy()
    if mapping.empty:
        return player_assets.iloc[0:0].copy(deep=True)
    ids = set(pd.to_numeric(mapping["fantasy_asset_id"], errors="coerce").dropna().astype(int))
    asset_ids = pd.to_numeric(
        player_assets.get("playerId", player_assets.get("PlayerId")), errors="coerce"
    )
    return player_assets.loc[asset_ids.isin(ids)].copy(deep=True).reset_index(drop=True)


def asset_ledger_diagnostics(
    player_assets: pd.DataFrame,
    player_identity_map: pd.DataFrame,
) -> dict[str, Any]:
    assets = player_assets.copy(deep=True)
    mapping = player_identity_map.copy(deep=True)
    active = pd.to_numeric(
        assets.get("is_active", assets.get("IsActive", pd.Series(1, index=assets.index))),
        errors="coerce",
    ).fillna(0).eq(1)
    duplicate_groups: list[dict[str, Any]] = []
    if not mapping.empty and "human_driver_id" in mapping.columns:
        for human_id, group in mapping.groupby("human_driver_id", sort=True):
            if len(group) <= 1:
                continue
            duplicate_groups.append(
                {
                    "human_driver_id": str(human_id),
                    "display_name": str(group.iloc[0].get("display_name") or human_id),
                    "assets": group[
                        [
                            "fantasy_asset_id",
                            "team_name",
                            "active",
                            "driver_reference",
                            "match_status",
                        ]
                    ].to_dict("records"),
                }
            )
    return {
        "driver_asset_count": int(len(assets)),
        "selectable_driver_asset_count": int(active.sum()),
        "inactive_driver_asset_count": int((~active).sum()),
        "duplicate_human_driver_count": int(len(duplicate_groups)),
        "duplicate_human_driver_assets": duplicate_groups,
        "player_asset_identity_mappings": mapping.to_dict("records"),
        "ambiguous_player_identity_count": int(
            mapping.get("match_status", pd.Series(index=mapping.index, dtype=object)).eq("ambiguous").sum()
        ),
        "unresolved_player_identity_count": int(
            mapping.get("match_status", pd.Series(index=mapping.index, dtype=object)).eq("unresolved").sum()
        ),
    }
