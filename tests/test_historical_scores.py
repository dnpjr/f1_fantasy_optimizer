from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from f1fantasy import app_core, recommend
from f1fantasy.historical_scores import (
    CANONICAL_COLUMNS,
    CANONICAL_KEY,
    DATA_VERSION,
    DEFAULT_CANONICAL_DATASET_PATH,
    EARLIEST_PRODUCTION_SEASON,
    apply_recorded_scores_to_model,
    canonical_market_snapshot,
    coverage_report,
    normalise_official_playerstats,
    normalise_third_party_recorded,
    resolve_score_precedence,
    validate_canonical_scores,
    load_canonical_scores,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "scripts/api_probe/raw-data/github_jm1261_fantasy_f1_league"


def _schedules() -> dict[int, pd.DataFrame]:
    return {
        year: pd.read_csv(ROOT / f"data/cache/schedule_{year}.csv")
        for year in (2023, 2024, 2025)
    }


def _third_party() -> pd.DataFrame:
    return normalise_third_party_recorded(RAW, _schedules())


def _canonical_row(origin: str, total: float, *, official: bool = False) -> pd.DataFrame:
    row = {column: pd.NA for column in CANONICAL_COLUMNS}
    row.update(
        season=2026,
        round=1,
        event_name="Australian Grand Prix",
        event_date="2026-03-08",
        entity_type="driver",
        canonical_entity_id="hamilton",
        source_entity_id="18" if official else "source:hamilton",
        name="Lewis Hamilton",
        abbreviation="HAM",
        constructor_name="Ferrari",
        fantasy_points_total=total,
        source_name="test",
        source_reference="test",
        source_licence="test",
        authority_class="official" if official else origin,
        is_official=official,
        is_recorded_total=origin != "reconstructed",
        is_reconstructed=origin == "reconstructed",
        fantasy_score_origin=origin,
        data_version=DATA_VERSION,
    )
    return pd.DataFrame([row], columns=CANONICAL_COLUMNS)


def test_recorded_production_source_has_exact_2023_to_2025_coverage():
    data = _third_party()
    validate_canonical_scores(data, expected_seasons=(2023, 2024, 2025))
    assert set(data["season"]) == {2023, 2024, 2025}
    assert data["season"].min() == EARLIEST_PRODUCTION_SEASON
    assert not data["season"].eq(2022).any()
    assert not data.duplicated(CANONICAL_KEY).any()
    report = coverage_report(data).set_index(["season", "entity_type"])
    assert report.loc[(2023, "driver"), ["rows", "races"]].tolist() == [440, 22]
    assert report.loc[(2023, "constructor"), ["rows", "races"]].tolist() == [220, 22]
    assert report.loc[(2024, "driver"), ["rows", "races"]].tolist() == [480, 24]
    assert report.loc[(2024, "constructor"), ["rows", "races"]].tolist() == [240, 24]
    assert report.loc[(2025, "driver"), ["rows", "races"]].tolist() == [480, 24]
    assert report.loc[(2025, "constructor"), ["rows", "races"]].tolist() == [240, 24]


def test_generated_production_dataset_contains_only_recorded_2023_to_2026_rows():
    data = load_canonical_scores(DEFAULT_CANONICAL_DATASET_PATH)
    assert set(data["season"]) == {2023, 2024, 2025, 2026}
    assert not data["season"].lt(2023).any()
    assert not data["is_reconstructed"].fillna(False).astype(bool).any()
    assert data.loc[data["season"] == 2026, "is_official"].fillna(False).astype(bool).all()
    assert data.loc[data["season"].isin((2023, 2024, 2025)), "authority_class"].eq("third_party_recorded").all()
    assert not data.duplicated(CANONICAL_KEY).any()


def test_2023_to_2025_totals_and_prices_are_exact_while_components_remain_null():
    data = _third_party()
    assert data["fantasy_points_total"].notna().all()
    assert data["price"].notna().all()
    assert data[["qualifying_points", "sprint_qualifying_points", "sprint_points", "race_points", "other_points"]].isna().all().all()
    assert (data["fantasy_score_origin"] == "third_party_recorded").all()
    assert not data["is_reconstructed"].fillna(False).astype(bool).any()


def test_v3_identity_and_outputs_cannot_be_confused_with_v2():
    assert DATA_VERSION == "historical_fantasy_scores_v3_recorded_2023_2026"
    assert "v3_recorded_2023_2026" in str(DEFAULT_CANONICAL_DATASET_PATH)
    assert "2023_2026" in DEFAULT_CANONICAL_DATASET_PATH.name
    assert DATA_VERSION != "historical_fantasy_scores_v2_recorded_2022_2026"
    manifest_path = DEFAULT_CANONICAL_DATASET_PATH.parent / "provenance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_version"] == DATA_VERSION
    assert manifest["supported_seasons"] == [2023, 2024, 2025, 2026]
    assert manifest["intentionally_excluded_seasons"] == [2021, 2022]
    assert manifest["official_warnings"] == []


def test_loader_rejects_an_embedded_v2_data_identity(tmp_path):
    stale = _canonical_row("third_party_recorded", 12.0)
    stale.loc[0, "season"] = 2023
    stale.loc[0, "data_version"] = "historical_fantasy_scores_v2_recorded_2022_2026"
    path = tmp_path / "stale_v2.csv"
    stale.to_csv(path, index=False)
    with pytest.raises(ValueError, match="data version mismatch"):
        load_canonical_scores(path)


def test_production_history_defaults_begin_at_2023():
    assert app_core.DEFAULT_HISTORICAL_SEASONS_BACK == 3
    assert recommend.current_season_delta == 3
    snapshot = app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2022,
        requested_seasons=(2022, 2023, 2024, 2025, 2026),
        loaded_seasons=(2022, 2023, 2024, 2025, 2026),
        season_load_failures={},
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.DataFrame(),
        players=pd.DataFrame(),
        teams=pd.DataFrame(),
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={},
    )
    coverage = app_core.season_coverage(snapshot, historical_seasons_back=5)
    assert coverage["requested_seasons"] == (2023, 2024, 2025, 2026)
    assert coverage["used_seasons"] == (2023, 2024, 2025, 2026)


def test_2026_official_coverage_is_valid_and_not_fixed_to_one_row_count():
    data = load_canonical_scores(DEFAULT_CANONICAL_DATASET_PATH)
    official = data[data["season"] == 2026]
    assert not official.empty
    assert official["round"].nunique() >= 1
    assert official["is_official"].fillna(False).astype(bool).all()
    assert official["fantasy_score_origin"].eq("official_recorded").all()
    assert official["price"].notna().all()


def test_genuine_zeroes_survive_but_inactive_replacement_placeholders_do_not():
    data = _third_party()
    genuine_zero = data[
        (data["entity_type"] == "driver")
        & (data["fantasy_points_total"] == 0)
        & pd.to_numeric(data["price"], errors="coerce").gt(0)
    ]
    assert not genuine_zero.empty
    ricciardo_2023 = data[
        (data["season"] == 2023) & (data["canonical_entity_id"] == "ricciardo")
    ]
    lawson_2023 = data[
        (data["season"] == 2023) & (data["canonical_entity_id"] == "lawson")
    ]
    assert ricciardo_2023["round"].tolist() == [11, 12, 18, 19, 20, 21, 22]
    assert lawson_2023["round"].tolist() == [13, 14, 15, 16, 17]
    assert not data.duplicated(CANONICAL_KEY).any()


def test_team_changes_use_one_driver_identity_and_event_specific_constructor():
    data = _third_party()
    tsunoda = data[(data["season"] == 2025) & (data["canonical_entity_id"] == "tsunoda")]
    lawson = data[(data["season"] == 2025) & (data["canonical_entity_id"] == "lawson")]
    assert tsunoda.loc[tsunoda["round"] <= 2, "constructor_name"].eq("Racing Bulls").all()
    assert tsunoda.loc[tsunoda["round"] >= 3, "constructor_name"].eq("Red Bull").all()
    assert lawson.loc[lawson["round"] <= 2, "constructor_name"].eq("Red Bull").all()
    assert lawson.loc[lawson["round"] >= 3, "constructor_name"].eq("Racing Bulls").all()


def test_source_precedence_is_official_then_recorded_then_reconstructed():
    reconstructed = _canonical_row("reconstructed", 1.0)
    third_party = _canonical_row("third_party_recorded", 2.0)
    official = _canonical_row("official_recorded", 3.0, official=True)
    resolved = resolve_score_precedence(reconstructed, third_party, official)
    assert len(resolved) == 1
    assert resolved.iloc[0]["fantasy_points_total"] == 3.0
    assert resolved.iloc[0]["fantasy_score_origin"] == "official_recorded"

    refreshed = _canonical_row("official_recorded", 4.0, official=True)
    refreshed.loc[0, "source_name"] = "fresh live response"
    latest = resolve_score_precedence(official, refreshed)
    assert latest.iloc[0]["fantasy_points_total"] == 4.0


def test_model_overlay_preserves_inputs_and_recorded_driver_and_constructor_totals():
    proxy = pd.DataFrame(
        [
            {
                "season": 2026, "round": 1, "circuitName": "Albert Park",
                "driverId": "hamilton", "driver": "Lewis Hamilton",
                "constructorId": "ferrari", "constructor": "Ferrari",
                "weekend_points": -99.0, "qualifying_points": 1.0, "quali_points": 1.0,
                "sprint_points": 0.0, "race_points": -100.0, "q2_reached": 1,
                "q3_reached": 1, "is_dnf": 0, "is_dsq": 0,
                "sprint_is_dnf": 0, "sprint_is_dsq": 0,
            }
        ]
    )
    driver = _canonical_row("official_recorded", 25.0, official=True)
    constructor = driver.copy(deep=True)
    constructor.loc[0, "entity_type"] = "constructor"
    constructor.loc[0, "canonical_entity_id"] = "ferrari"
    constructor.loc[0, "source_entity_id"] = "28"
    constructor.loc[0, "name"] = "Ferrari"
    constructor.loc[0, "abbreviation"] = "FER"
    constructor.loc[0, "fantasy_points_total"] = 60.0
    before_proxy = proxy.copy(deep=True)
    before_recorded = pd.concat([driver, constructor], ignore_index=True)
    drivers, constructors, diagnostics = apply_recorded_scores_to_model(proxy, before_recorded)
    assert drivers.loc[drivers["driverId"] == "hamilton", "weekend_points"].item() == 25.0
    assert constructors.loc[constructors["constructorId"] == "ferrari", "constructor_weekend_points"].item() == 60.0
    assert drivers.loc[drivers["driverId"] == "hamilton", "fantasy_score_origin"].item() == "official_recorded"
    assert diagnostics["recorded_driver_rows"] == 1
    pd.testing.assert_frame_equal(proxy, before_proxy)
    pd.testing.assert_frame_equal(before_recorded, pd.concat([driver, constructor], ignore_index=True))


def test_official_normaliser_verifies_2026_and_preserves_missing_components():
    driver_points = pd.DataFrame(
        [
            {"PlayerId": 18, "season": 2026, "round": 6, "race_name": "Miami Grand Prix", "fantasy_points": 0.0, "qualifying_points": 2.0, "sprint_qualifying_points": pd.NA, "sprint_points": 5.0, "race_points": -7.0, "price": 24.0, "is_played": 1},
            {"PlayerId": 18, "season": 2025, "round": 1, "race_name": "Ignored query echo", "fantasy_points": 99.0, "is_played": 1},
        ]
    )
    players = pd.DataFrame([{"playerId": 18, "name": "Lewis Hamilton", "tla": "HAM", "team": "Ferrari"}])
    official, warnings = normalise_official_playerstats(
        driver_points,
        pd.DataFrame(),
        players,
        pd.DataFrame(),
        schedule=pd.DataFrame([{"season": 2026, "round": 4, "raceName": "Miami Grand Prix", "date": "2026-05-03", "sprint_date": "2026-05-02"}]),
    )
    assert warnings == []
    assert len(official) == 1
    assert official.iloc[0]["season"] == 2026
    assert official.iloc[0]["fantasy_points_total"] == 0.0
    assert official.iloc[0]["round"] == 4
    assert official.iloc[0]["sprint_points"] == 5.0
    assert pd.isna(official.iloc[0]["sprint_qualifying_points"])
    assert official.iloc[0]["other_points"] == 0.0
    assert official.iloc[0]["is_official"]
    assert official.iloc[0]["fantasy_score_origin"] == "official_recorded"


def test_canonical_market_snapshot_uses_latest_complete_official_2026_round_without_mutation():
    recorded = load_canonical_scores(DEFAULT_CANONICAL_DATASET_PATH)
    original = recorded.copy(deep=True)

    snapshot = canonical_market_snapshot(recorded, 2026)

    assert snapshot["season"] == 2026
    assert snapshot["round"] == 12
    assert len(snapshot["players"]) == 22
    assert len(snapshot["teams"]) == 11
    assert snapshot["players"]["playerId"].is_unique
    assert snapshot["teams"]["teamId"].is_unique
    assert snapshot["players"]["price"].gt(0).all()
    assert snapshot["teams"]["price"].gt(0).all()
    assert {"Bottas", "Perez"}.issubset(
        set(snapshot["players"]["name"].str.split().str[-1])
    )
    pd.testing.assert_frame_equal(recorded, original)


def test_canonical_market_snapshot_rejects_an_incomplete_newer_round():
    recorded = load_canonical_scores(DEFAULT_CANONICAL_DATASET_PATH)
    newest = recorded[
        (recorded["season"] == 2026) & (recorded["round"] == 12)
    ].copy()
    partial = newest.head(1).copy()
    partial["round"] = 13
    combined = pd.concat([recorded, partial], ignore_index=True)

    snapshot = canonical_market_snapshot(combined, 2026)

    assert snapshot["round"] == 12
