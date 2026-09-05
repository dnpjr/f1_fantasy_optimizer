from __future__ import annotations

import pandas as pd
import pytest

from f1fantasy import app_core, fantasy_api, player_stats
from f1fantasy.asset_identity import (
    asset_ledger_diagnostics,
    asset_to_human_identity,
    build_player_identity_map,
    human_assets,
)
from f1fantasy.historical_scores import normalise_official_playerstats


def _raw_market() -> list[dict]:
    return [
        {
            "PlayerId": 114,
            "FUllName": "Liam Lawson",
            "PositionName": "DRIVER",
            "IsActive": 0,
            "Value": 12.4,
            "OldPlayerValue": 12.3,
            "TeamId": 9,
            "TeamName": "Racing Bulls",
            "SelectedPercentage": 1.2,
            "CaptainSelectedPercentage": 0.1,
            "DriverReference": "LIALAW01",
            "DriverTLA": "LAW",
            "F1PlayerId": -8,
            "Status": "INACTIVE",
        },
        {
            "PlayerId": 116,
            "FUllName": "Liam Lawson",
            "PositionName": "DRIVER",
            "IsActive": 1,
            "Value": 13.0,
            "OldPlayerValue": 13.0,
            "TeamId": 5,
            "TeamName": "Red Bull",
            "SelectedPercentage": 4.5,
            "CaptainSelectedPercentage": 0.2,
            "DriverReference": "LIALAW01",
            "DriverTLA": "LAW",
            "F1PlayerId": 14,
            "Status": "ACTIVE",
        },
        {
            "PlayerId": 130,
            "FUllName": "Yuki Tsunoda",
            "PositionName": "DRIVER",
            "IsActive": 1,
            "Value": 10.2,
            "OldPlayerValue": 10.2,
            "TeamId": 9,
            "TeamName": "Racing Bulls",
            "DriverReference": "YUKTSU01",
            "DriverTLA": "TSU",
            "F1PlayerId": 22,
        },
        {
            "PlayerId": 11032,
            "FUllName": "Isack Hadjar",
            "PositionName": "DRIVER",
            "IsActive": 0,
            "Value": 15.1,
            "OldPlayerValue": 15.0,
            "TeamId": 5,
            "TeamName": "Red Bull",
            "DriverReference": "ISAHAD01",
            "DriverTLA": "HAD",
            "F1PlayerId": 47,
        },
        {
            "PlayerId": 28,
            "FUllName": "Red Bull",
            "PositionName": "CONSTRUCTOR",
            "IsActive": 1,
            "Value": 30.0,
            "OldPlayerValue": 29.8,
            "SelectedPercentage": 20.0,
            "DriverTLA": "RBR",
            "F1PlayerId": 5,
        },
    ]


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"driverId": "lawson", "driver": "Liam Lawson"},
            {"driverId": "tsunoda", "driver": "Yuki Tsunoda"},
            {"driverId": "hadjar", "driver": "Isack Hadjar"},
        ]
    )


def _expected_driver_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driverId": "lawson",
                "driver": "Liam Lawson",
                "exp_score": 20.0,
                "next_race_exp_score": 20.0,
                "horizon_expected_points": 60.0,
                "dnf_rate": 0.1,
                "volatility": 4.0,
                "nn_exp_score": 21.0,
            },
            {
                "driverId": "tsunoda",
                "driver": "Yuki Tsunoda",
                "exp_score": 18.0,
                "next_race_exp_score": 18.0,
                "horizon_expected_points": 54.0,
                "dnf_rate": 0.2,
                "volatility": 5.0,
                "nn_exp_score": 19.0,
            },
        ]
    )


def _playerstats_payload(player_id: int, *, points: float | None = None) -> dict:
    stats = [] if points is None else [{"Event": "Total", "Value": points}]
    return {
        "Value": {
            "PlayerId": player_id,
            "PlayerSkill": 1,
            "GamedayWiseStats": [
                {
                    "GamedayId": 12,
                    "PlayerValue": 15.1,
                    "OldPlayerValue": 15.0,
                    "IsPlayed": int(points is not None),
                    "IsActive": 0,
                    "StatsWise": stats,
                }
            ],
            "MatchWiseStats": [
                {
                    "GamedayId": 12,
                    "RaceDayWise": [
                        {
                            "MeetingNumber": 12,
                            "MeetingName": "Test Grand Prix",
                            "Season": 2026,
                            "RaceDayId": 1200 + player_id,
                        }
                    ],
                }
            ],
        }
    }


def test_full_ledger_is_lossless_active_filter_is_compatible_and_input_is_unchanged():
    raw = _raw_market()
    original = [dict(row) for row in raw]

    ledger = fantasy_api.normalise_player_asset_ledger(raw, feed_round=12)
    selectable = fantasy_api.selectable_player_assets(ledger)

    assert raw == original
    assert set(ledger["playerId"]) == {114, 116, 130, 11032}
    assert set(selectable["playerId"]) == {116, 130}
    assert {"IsActive", "is_active", "TeamId", "team_id", "DriverReference", "driver_reference"}.issubset(ledger.columns)
    assert ledger.set_index("playerId").loc[114, "F1PlayerId"] == -8
    assert ledger.set_index("playerId").loc[11032, "price"] == pytest.approx(15.1)
    assert ledger.set_index("playerId").loc[11032, "official_price_change"] == pytest.approx(0.1)
    assert ledger["feed_round"].eq(12).all()
    assert list(selectable.columns) == [
        "playerId",
        "name",
        "price",
        "previous_price",
        "official_price_change",
        "team",
        "selected_pct",
        "captain_selected_pct",
        "driver_reference",
        "tla",
        "f1_player_id",
    ]


def test_combined_market_loader_fetches_once_and_derives_all_universes(monkeypatch):
    calls: list[int | None] = []
    monkeypatch.setattr(
        fantasy_api,
        "_get_market",
        lambda feed_round=None: calls.append(feed_round) or _raw_market(),
    )

    market = fantasy_api.fetch_market_asset_ledgers(feed_round=12)

    assert calls == [12]
    assert len(market["player_assets"]) == 4
    assert set(market["players"]["playerId"]) == {116, 130}
    assert len(market["constructor_assets"]) == len(market["teams"]) == 1


def test_verified_market_cache_retains_the_full_asset_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(fantasy_api, "MIN_CACHEABLE_DRIVER_ROWS", 1)
    monkeypatch.setattr(fantasy_api, "MIN_CACHEABLE_CONSTRUCTOR_ROWS", 1)
    player_assets = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    constructor_assets = fantasy_api.normalise_constructor_asset_ledger(
        _raw_market(), feed_round=12
    )
    players = fantasy_api.selectable_player_assets(player_assets)
    teams = fantasy_api.selectable_constructor_assets(constructor_assets)
    path = tmp_path / "verified-market.json"

    fantasy_api.save_verified_market_cache(
        12,
        players,
        teams,
        player_assets=player_assets,
        constructor_assets=constructor_assets,
        asset_ledger_complete=True,
        path=path,
    )
    cached = fantasy_api.load_verified_market_cache(path=path)

    assert cached["asset_ledger_complete"] is True
    assert set(cached["player_assets"]["playerId"]) == {114, 116, 130, 11032}
    assert set(cached["players"]["playerId"]) == {116, 130}


def test_lawson_assets_remain_distinct_but_share_one_human_identity():
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    mapping = build_player_identity_map(ledger, _history())
    lawson = mapping[mapping["fantasy_asset_id"].isin([114, 116])]

    assert set(lawson["fantasy_asset_id"]) == {114, 116}
    assert lawson["human_driver_id"].tolist() == ["lawson", "lawson"]
    assert lawson["history_driver_id"].tolist() == ["lawson", "lawson"]
    assert lawson["match_method"].eq("driver_reference").all()
    assert asset_to_human_identity(mapping, 114).active is False
    assert asset_to_human_identity(mapping, 116).active is True
    assert set(human_assets(ledger, mapping, "lawson")["playerId"]) == {114, 116}


def test_lawson_negative_inactive_f1_id_does_not_break_reference_continuity():
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    mapping = build_player_identity_map(ledger, _history()).set_index("fantasy_asset_id")

    assert mapping.loc[114, "history_driver_id"] == "lawson"
    assert mapping.loc[116, "history_driver_id"] == "lawson"


def test_driver_reference_and_tla_win_before_fuzzy_name():
    raw = _raw_market()
    raw[0]["FUllName"] = "Yuki Tsunoda"
    ledger = fantasy_api.normalise_player_asset_ledger(raw, feed_round=12)
    mapping = build_player_identity_map(ledger, _history()).set_index("fantasy_asset_id")

    assert mapping.loc[114, "human_driver_id"] == "lawson"
    assert mapping.loc[114, "match_method"] == "driver_reference"


def test_ambiguous_strong_identity_is_diagnostic_not_an_arbitrary_merge():
    raw = _raw_market()[:2]
    raw[1]["DriverTLA"] = "TSU"
    mapping = build_player_identity_map(
        fantasy_api.normalise_player_asset_ledger(raw, feed_round=12),
        _history(),
    )

    assert mapping["match_status"].eq("ambiguous").all()
    assert mapping["history_driver_id"].isna().all()
    assert mapping["diagnostic"].str.contains("conflicting human candidates").all()


def test_legacy_name_fallback_maps_when_stronger_identifiers_are_absent():
    assets = pd.DataFrame(
        [{"playerId": 999, "name": "Liam Lawson", "team": "Red Bull", "is_active": 1}]
    )
    mapping = build_player_identity_map(assets, _history()).iloc[0]

    assert mapping["human_driver_id"] == "lawson"
    assert mapping["match_method"] == "normalised_name"


def test_current_assets_use_human_ev_but_keep_current_team_context():
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    selectable = fantasy_api.selectable_player_assets(ledger)
    mapping = build_player_identity_map(ledger, _history())
    drivers = app_core._build_driver_table(selectable, _expected_driver_rows(), mapping)
    constructors = pd.DataFrame(
        [
            {"name": "Racing Bulls", "exp_score": 10.0},
            {"name": "Red Bull", "exp_score": 100.0},
        ]
    )
    adjusted = app_core._apply_team_strength_adjustment(drivers, constructors).set_index("id")

    assert adjusted.loc[116, "driverId"] == "lawson"
    assert adjusted.loc[116, "exp_score_raw"] == pytest.approx(20.0)
    assert adjusted.loc[116, "team"] == "Red Bull"
    assert adjusted.loc[116, "team_exp"] == pytest.approx(100.0)
    assert adjusted.loc[116, "exp_score"] == pytest.approx(23.0)
    assert adjusted.loc[130, "driverId"] == "tsunoda"
    assert pd.notna(adjusted.loc[130, "exp_score"])


def test_explicit_identity_bridge_does_not_change_ordinary_driver_ev():
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    selectable = fantasy_api.selectable_player_assets(ledger)
    mapping = build_player_identity_map(ledger, _history())
    expected = _expected_driver_rows()

    legacy = app_core._build_driver_table(selectable, expected)
    bridged = app_core._build_driver_table(selectable, expected, mapping)

    pd.testing.assert_series_equal(bridged["exp_score"], legacy["exp_score"], check_names=False)
    pd.testing.assert_series_equal(bridged["driverId"], legacy["driverId"], check_names=False)


def test_old_asset_playerstats_are_not_reassigned_to_new_lawson_asset():
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    players = fantasy_api.selectable_player_assets(ledger)
    old_asset_stats = pd.DataFrame(
        [
            {
                "PlayerId": 114,
                "asset_type": "driver",
                "season": 2026,
                "round": 1,
                "race_name": "Test Grand Prix",
                "fantasy_points": 10.0,
                "is_played": 1,
                "price": 12.4,
            }
        ]
    )
    official, warnings = normalise_official_playerstats(
        old_asset_stats,
        pd.DataFrame(),
        players,
        pd.DataFrame(),
        results=_history(),
    )

    assert official.empty
    assert any("PlayerId 114" in warning for warning in warnings)


def test_inactive_hadjar_is_holding_visible_and_playerstats_can_be_requested_explicitly(monkeypatch):
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    holdings = fantasy_api.holding_valid_player_assets(ledger)
    price_view = fantasy_api.price_view_player_assets(ledger)
    hadjar = holdings[holdings["playerId"] == 11032].rename(columns={"playerId": "id"})
    calls: list[int] = []
    monkeypatch.setattr(
        player_stats,
        "fetch_player_stats",
        lambda player_id: calls.append(player_id) or _playerstats_payload(player_id, points=8.0),
    )

    recent, races, _diagnostics = player_stats.fetch_recent_points_for_roster(hadjar, "driver")

    assert calls == [11032]
    assert recent.loc[0, "recent_points_1ago"] == pytest.approx(8.0)
    assert races.loc[0, "PlayerId"] == 11032
    assert hadjar.loc[hadjar.index[0], "price"] == pytest.approx(15.1)
    assert 11032 in set(price_view["playerId"])
    assert 11032 not in set(fantasy_api.selectable_player_assets(ledger)["playerId"])


def test_ledger_diagnostics_expose_counts_duplicates_and_asset_mappings():
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    mapping = build_player_identity_map(ledger, _history())
    diagnostics = asset_ledger_diagnostics(ledger, mapping)

    assert diagnostics["driver_asset_count"] == 4
    assert diagnostics["selectable_driver_asset_count"] == 2
    assert diagnostics["inactive_driver_asset_count"] == 2
    assert diagnostics["duplicate_human_driver_count"] == 1
    duplicate = diagnostics["duplicate_human_driver_assets"][0]
    assert duplicate["human_driver_id"] == "lawson"
    assert {asset["fantasy_asset_id"] for asset in duplicate["assets"]} == {114, 116}
    assert len(diagnostics["player_asset_identity_mappings"]) == 4


def test_snapshot_copy_defensively_copies_ledger_and_identity_contracts():
    ledger = fantasy_api.normalise_player_asset_ledger(_raw_market(), feed_round=12)
    mapping = build_player_identity_map(ledger, _history())
    snapshot = app_core.LiveDataSnapshot(
        current_season=2026,
        loaded_start_year=2026,
        requested_seasons=(2026,),
        loaded_seasons=(2026,),
        season_load_failures={},
        results=pd.DataFrame(),
        qualifying=pd.DataFrame(),
        sprint=pd.DataFrame(),
        schedule=pd.DataFrame(),
        players=fantasy_api.selectable_player_assets(ledger),
        teams=pd.DataFrame(),
        driver_recent_points=pd.DataFrame(),
        constructor_recent_points=pd.DataFrame(),
        driver_race_points=pd.DataFrame(),
        constructor_race_points=pd.DataFrame(),
        team_lock_payload={},
        source_diagnostics={},
        player_assets=ledger,
        player_identity_map=mapping,
    )

    copied = app_core.copy_live_data_snapshot(snapshot)
    copied.player_assets.loc[0, "price"] = 999.0
    copied.player_identity_map.loc[0, "human_driver_id"] = "changed"

    assert snapshot.player_assets.loc[0, "price"] == pytest.approx(12.4)
    assert snapshot.player_identity_map.loc[0, "human_driver_id"] == "lawson"
