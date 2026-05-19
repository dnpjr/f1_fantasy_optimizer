import pandas as pd
from pathlib import Path
import inspect

import pytest

from f1fantasy import app_core
from f1fantasy.app_core import (
    OBJECTIVE_POINTS_ONLY,
    OBJECTIVE_PRICE_GROWTH_ONLY,
    OBJECTIVE_COMBINED,
    DEFAULT_TOP_K,
    apply_no_negative_scores,
    annotate_card_expected_points,
    adjust_money_value,
    auto_budget_from_team_cost,
    build_asset_option_labels,
    build_transfer_recommendations,
    current_team_budget_from_selection,
    current_team_json,
    current_team_upload_summary,
    fantasy_asset_card_html,
    fantasy_card_grid_html,
    format_countdown,
    format_money,
    format_points,
    format_probability,
    format_signed_money,
    format_signed_points,
    format_transfer_recommendations_display,
    price_change_probability_matrix_table,
    format_selected_asset_display_table,
    load_current_team_json_text,
    projected_team_value_from_budget,
    parse_team_lock_deadline_timestamp,
    recommendation_badges,
    resolve_budget_value,
    run_optimizer,
    selected_assets_price_gain,
    transfer_asset_max_price_gain,
    transfer_candidate_filter_score,
    select_chip_boost_drivers,
    team_colour,
    team_expected_points_with_chips,
    validate_current_team,
)
from f1fantasy.player_stats import parse_team_lock_deadline_from_payload


def _drivers():
    return pd.DataFrame(
        [
            {"id": str(i), "name": f"Driver {i}", "price": 10.0, "exp_score": float(i)}
            for i in range(1, 7)
        ]
    )


def _constructors():
    return pd.DataFrame(
        [
            {"id": str(i), "name": f"Constructor {i}", "price": 10.0, "exp_score": float(i)}
            for i in range(1, 4)
        ]
    )


def test_validate_current_team_valid_shape():
    result = validate_current_team(["1", "2", "3", "4", "5"], ["1", "2"], _drivers(), _constructors(), budget=100.0)

    assert result["valid"] is True
    assert result["total_cost"] == 70.0
    assert result["projected_points"] == 18.0


def test_validate_current_team_reports_wrong_shape_and_over_budget():
    result = validate_current_team(["1", "2"], ["1"], _drivers(), _constructors(), budget=20.0)

    assert result["valid"] is False
    assert "Select exactly 5 drivers." in result["errors"]
    assert "Select exactly 2 constructors." in result["errors"]
    assert result["warnings"]


def test_validate_current_team_uses_adjustable_budget():
    result = validate_current_team(["1", "2", "3", "4", "5"], ["1", "2"], _drivers(), _constructors(), budget=70.0)
    richer = validate_current_team(["1", "2", "3", "4", "5"], ["1", "2"], _drivers(), _constructors(), budget=108.7)

    assert result["valid"] is True
    assert richer["warnings"] == []


def test_current_team_projected_points_use_same_exp_score_source():
    drivers = _drivers()
    constructors = _constructors()
    drivers.loc[drivers["id"] == "1", "exp_score"] = 99.0

    result = validate_current_team(["1", "2", "3", "4", "5"], ["1", "2"], drivers, constructors, budget=100.0)

    assert result["projected_points"] == 116.0


def test_current_team_json_export_shape():
    payload = current_team_json(["1", "hamilton"], ["25"], free_transfers=1, bank=0.25)

    assert payload == {
        "drivers": [1, "hamilton"],
        "constructors": [25],
        "free_transfers": 1,
        "bank": 0.2,
    }


def test_run_optimizer_default_top_k_is_one():
    sols = run_optimizer(_drivers(), _constructors(), budget=100.0)

    assert DEFAULT_TOP_K == 1
    assert len(sols) == 1


def test_team_colour_mapping_has_known_and_unknown_fallbacks():
    assert team_colour("Ferrari") == "#dc2626"
    assert team_colour("Definitely Unknown Team") == "#64748b"


def test_current_team_display_formatting_rounds_numbers_and_removes_raw_price_change():
    frame = pd.DataFrame(
        [
            {
                "name": "Driver One",
                "team": "Ferrari",
                "price": 24.4,
                "exp_score": 56.789,
                "effective_price_change_after_floor_ceiling": 0.3,
                "projected_price": 24.7,
                "dnf_rate": 0.12345,
                "raw_price_change": 0.3,
            }
        ]
    )

    out = format_selected_asset_display_table(frame)

    assert "raw_price_change" not in out.columns
    assert out.loc[0, "Price"] == 24.4
    assert out.loc[0, "Expected / race"] == 56.79
    assert out.loc[0, "Expected price gain"] == 0.3
    assert out.loc[0, "Projected price"] == 24.7
    assert out.loc[0, "DNF rate"] == 0.123


def test_current_team_upload_parsing_and_missing_ids_reporting():
    payload = {
        "drivers": [1, "missing-driver"],
        "constructors": [25, "missing-constructor"],
        "bank": 0.2,
        "free_transfers": 2,
    }

    summary = current_team_upload_summary(payload, available_driver_ids={"1", "2"}, available_constructor_ids={"25"})

    assert summary["drivers"] == ["1"]
    assert summary["constructors"] == ["25"]
    assert summary["missing_drivers"] == ["missing-driver"]
    assert summary["missing_constructors"] == ["missing-constructor"]
    assert summary["bank"] == 0.2
    assert summary["free_transfers"] == 2


def test_current_team_upload_invalid_json_raises_useful_error():
    with pytest.raises(ValueError, match="could not parse JSON"):
        load_current_team_json_text("{not valid json")


def test_uploaded_json_budget_is_sum_of_selected_prices_plus_bank():
    drivers = pd.DataFrame([{"price": 24.4}, {"price": 26.4}])
    constructors = pd.DataFrame([{"price": 30.5}, {"price": 28.9}])

    assert current_team_budget_from_selection(drivers, constructors, bank=0.2) == pytest.approx(110.4)


def test_app_title_uses_british_spelling():
    assert "F1 Fantasy Optimiser" in Path("streamlit_app.py").read_text(encoding="utf-8")


def test_money_and_points_format_helpers():
    assert format_money(7.6) == "7.60M"
    assert format_signed_money(0.43) == "+0.43M"
    assert format_signed_money(-0.4) == "-0.40M"
    assert format_points(35.239) == "35.24"
    assert format_signed_points(7.64) == "+7.64"
    assert format_probability(0.723) == "72.3%"
    assert format_signed_money(-0.001) == "0.00M"
    assert format_signed_points(-0.001) == "0.00"


def test_fantasy_card_html_escapes_dynamic_text_and_is_not_indented_code():
    frame = pd.DataFrame(
        [
            {
                "name": "<script>alert(1)</script>",
                "team": "Ferrari & Friends",
                "price": 24.4,
                "exp_score": 56.7,
                "expected_price_change": 0.3,
                "projected_price": 24.7,
            }
        ]
    )

    html = fantasy_card_grid_html(frame, asset_label="Driver")

    assert html.startswith('<div class="f1-card-grid">')
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Ferrari &amp; Friends" in html
    assert "\n    <div" not in html


def test_streamlit_card_wrapper_renders_html_unsafely_and_json_tools_are_combined():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "fantasy_card_grid_html" in source
    assert "unsafe_allow_html=True" in source
    assert "Advanced: JSON import / export" in source
    assert "Advanced: JSON export" not in source
    assert "Technical details" not in source


def test_main_tabs_remove_assumptions_and_trends():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert '"Assumptions"' not in source
    assert '"Trends"' not in source
    assert '"Optimise"' in source
    assert '"Price changes"' in source
    assert '"Current team"' in source
    assert '"Transfers"' in source
    assert '"Locks and exclusions"' in source
    assert '"Model settings"' in source
    assert '"Diagnostics"' in source
    assert "Load detailed playerstats on startup" not in source


def test_subtitle_and_tab_explanations_use_user_facing_copy():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "Optimise your F1 Fantasy team using live prices, race history, price-change probabilities and transfer recommendations." in source
    assert "A lightweight Streamlit wrapper around the existing Python model." not in source
    assert "OPTIMISER COCKPIT" in source
    assert "BUDGET BUILDER" in source
    assert "SQUAD BUILDER" in source
    assert "TRANSFER DESK" in source
    assert "SELECTION RULES" in source
    assert "MODEL SETTINGS" in source
    assert "DATA CHECKS" in source


def test_high_level_metric_label_uses_expected_points():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert '.metric("Expected points"' in source
    assert '.metric("Projected points / race"' not in source
    assert "DRS / boost multiplier" not in source
    assert "Use No Negative expected scores" not in source
    assert "Chips applied" in source
    assert "Advanced objective details" not in source
    assert "Allow extra transfers" not in source
    assert "Number of transfer options" in source
    assert "Show only free-transfer moves" not in source
    assert "Recommendation priority" not in source
    assert "Advanced transfer settings" not in source
    assert "Risk appetite" not in source
    assert "Recommendation objective" in source
    assert "Detailed table" not in source
    assert "Run transfer recommendations" in source
    assert 'max_value=4' in source
    assert "transfer_results" in source
    assert "Set transfer options, then run transfer recommendations." in source
    assert "Running transfer recommendations..." in source
    assert "Transfer run failed. Keeping last successful recommendations." in source
    assert '.metric("Remaining budget"' in source
    assert '.metric("Expected price gain"' in source
    assert '.metric("Projected team value"' in source
    assert "Could not load live data. Try Refresh live data, or try again later." in source
    assert "Live refresh failed. Using last loaded data." in source


def test_locks_and_exclusions_copy_and_summary_present():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "Lock assets you want to keep and exclude assets you do not want the optimiser or transfer tool to use." in source
    assert "Locks force an asset into optimiser and transfer results. Exclusions prevent an asset from being selected or recommended." in source
    assert '.metric("Locked drivers"' in source
    assert '.metric("Locked constructors"' in source
    assert '.metric("Excluded drivers"' in source
    assert '.metric("Excluded constructors"' in source


def test_current_team_json_import_export_shape_round_trips():
    payload = current_team_json(["11161", "121"], ["25"], free_transfers=2, bank=0.25)
    parsed = load_current_team_json_text(__import__("json").dumps(payload))

    assert parsed == {
        "drivers": ["11161", "121"],
        "constructors": ["25"],
        "free_transfers": 2,
        "bank": 0.2,
    }


def test_current_team_price_gain_sums_probabilistic_asset_gains():
    drivers = pd.DataFrame([{"expected_price_gain": 0.25}, {"expected_price_gain": -0.05}])
    constructors = pd.DataFrame([{"expected_price_gain": 0.1}])

    assert selected_assets_price_gain(drivers, constructors) == pytest.approx(0.30)


def test_projected_team_value_includes_budget_and_expected_gain():
    assert projected_team_value_from_budget(108.7, 2.4) == pytest.approx(111.1)


def test_load_model_data_defaults_to_include_playerstats_true():
    signature = inspect.signature(app_core.load_model_data)
    assert signature.parameters["include_playerstats"].default is True
    assert "playerstats_load_duration_seconds" in inspect.getsource(app_core.load_model_data)


def test_cached_loader_has_plain_data_signature_without_ui_callback():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    fn_block = source.split("def load_cached_model_data(", 1)[1].split("def _option_labels", 1)[0]
    signature_block = fn_block.split("):", 1)[0]
    assert "_progress_callback" not in signature_block
    assert "progress_callback=" not in fn_block
    assert "st." not in fn_block
    assert "last_good_model_payload" in source


def test_transfer_recommendations_emit_progress_stages_and_counts():
    events = []
    drivers = pd.DataFrame(
        [
            {"id": "1", "name": "D1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "2", "name": "D2", "price": 10.0, "exp_score": 9.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "3", "name": "D3", "price": 10.0, "exp_score": 8.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "4", "name": "D4", "price": 10.0, "exp_score": 7.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "5", "name": "D5", "price": 10.0, "exp_score": 6.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "6", "name": "D6", "price": 10.0, "exp_score": 15.0, "expected_price_gain": 0.2, "volatility": 6.0},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "1", "name": "C1", "price": 20.0, "exp_score": 20.0, "expected_price_gain": 0.2, "volatility": 8.0},
            {"id": "2", "name": "C2", "price": 20.0, "exp_score": 18.0, "expected_price_gain": 0.2, "volatility": 8.0},
            {"id": "3", "name": "C3", "price": 20.0, "exp_score": 25.0, "expected_price_gain": 0.3, "volatility": 9.0},
        ]
    )

    recs = build_transfer_recommendations(
        current_driver_ids=["1", "2", "3", "4", "5"],
        current_constructor_ids=["1", "2"],
        drivers=drivers,
        constructors=constructors,
        budget=100.0,
        max_transfers=2,
        top_n=5,
        progress_callback=lambda payload: events.append(payload),
    )

    assert not recs.empty
    stages = [str(event.get("stage")) for event in events]
    assert "read_current_team" in stages
    assert "apply_locks_exclusions" in stages
    assert "filter_candidates" in stages
    assert "generate_candidates" in stages
    assert "score_candidates" in stages
    assert "rank_recommendations" in stages
    assert stages[-1] == "ready"
    assert int(events[-1].get("transfer_candidate_count_total", 0) or 0) >= 1
    assert int(events[-1].get("transfer_candidates_evaluated", 0) or 0) >= 1
    assert int(events[-1].get("transfer_candidates_scored", 0) or 0) >= 1
    assert float(events[-1].get("progress", 0.0) or 0.0) == pytest.approx(1.0)
    assert all(0.0 <= float(event.get("progress", 0.0) or 0.0) <= 1.0 for event in events)

    score_events = [event for event in events if event.get("stage") == "score_candidates"]
    assert score_events, "Expected score_candidates progress events"
    first_score = score_events[0]
    assert float(first_score.get("progress", 0.0) or 0.0) < 0.50
    assert "/" in str(first_score.get("message", ""))
    assert "evaluated" in str(first_score.get("message", "")).lower()
    assert int(score_events[-1].get("transfer_candidates_evaluated", 0) or 0) == int(
        score_events[-1].get("transfer_candidate_count_total", 0) or 0
    )


def test_transfer_progress_ui_has_single_status_line_and_new_diagnostics_keys():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "transfer_line = st.empty()" not in source
    assert "transfer_line.caption(" not in source
    assert "transfer_candidate_count_total" in source
    assert "transfer_candidates_evaluated" in source
    assert "transfer_generation_duration_seconds" in source
    assert "transfer_scoring_duration_seconds" in source
    assert "transfer_total_duration_seconds" in source
    assert "candidate_filter_score_used_for_prefilter" in source
    assert "exhaustive_scoring_used_after_prefilter" in source
    assert "final_score_used_for_sorting" in source
    assert "candidate_pool_mode" in source


def test_transfer_candidate_filter_score_uses_normalised_components():
    cheap_asset = {"exp_score": 20.0, "price": 10.0, "expected_price_gain": 0.3, "volatility": 5.0}
    expensive_asset = {"exp_score": 20.0, "price": 25.0, "expected_price_gain": 0.3, "volatility": 5.0}

    assert transfer_asset_max_price_gain(10.0) == pytest.approx(0.6)
    assert transfer_asset_max_price_gain(25.0) == pytest.approx(0.3)
    assert transfer_candidate_filter_score(cheap_asset, OBJECTIVE_POINTS_ONLY, price_gain_weight=10.0) == pytest.approx(2.0)
    assert transfer_candidate_filter_score(cheap_asset, OBJECTIVE_PRICE_GROWTH_ONLY, price_gain_weight=10.0) == pytest.approx(0.5)
    # scaled weight = 10/10 = 1.0 -> 2.0 + 0.5
    assert transfer_candidate_filter_score(cheap_asset, OBJECTIVE_COMBINED, price_gain_weight=10.0) == pytest.approx(2.5)
    # expensive uses smaller max gain denominator so price-gain normalised component is larger.
    assert transfer_candidate_filter_score(expensive_asset, OBJECTIVE_PRICE_GROWTH_ONLY, price_gain_weight=10.0) == pytest.approx(1.0)


def test_transfer_filtering_balanced_caps_incoming_pool_and_excludes_bottom_constructors():
    drivers = pd.DataFrame(
        [
            {"id": f"d{i}", "name": f"D{i}", "price": 10.0, "exp_score": float(60 - i), "expected_price_gain": 0.1}
            for i in range(1, 26)
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": f"c{i}", "name": f"C{i}", "price": 12.0, "exp_score": float(30 - i), "expected_price_gain": 0.1}
            for i in range(1, 11)
        ]
    )
    events = []
    recs = build_transfer_recommendations(
        current_driver_ids=[f"d{i}" for i in range(1, 6)],
        current_constructor_ids=["c1", "c2"],
        drivers=drivers,
        constructors=constructors,
        budget=100.0,
        max_transfers=2,
        search_mode="balanced",
        progress_callback=lambda payload: events.append(payload),
    )
    assert not recs.empty
    final_event = events[-1]
    assert int(final_event.get("incoming_driver_candidates", 0) or 0) <= 15
    # 10 total ctors, 2 current -> 8 incoming, drop bottom 2 => 6
    assert int(final_event.get("incoming_constructor_candidates", 0) or 0) == 6


def test_transfer_balanced_evaluates_fewer_candidates_than_exhaustive():
    drivers = pd.DataFrame(
        [
            {"id": f"d{i}", "name": f"D{i}", "price": 8.0 + (i % 5), "exp_score": float(70 - i), "expected_price_gain": 0.05 * ((i % 4) + 1), "volatility": 5.0 + (i % 7)}
            for i in range(1, 19)
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": f"c{i}", "name": f"C{i}", "price": 10.0 + i, "exp_score": float(35 - i), "expected_price_gain": 0.05 * ((i % 3) + 1), "volatility": 8.0 + (i % 4)}
            for i in range(1, 8)
        ]
    )

    balanced_events = []
    exhaustive_events = []
    recs_balanced = build_transfer_recommendations(
        current_driver_ids=[f"d{i}" for i in range(1, 6)],
        current_constructor_ids=["c1", "c2"],
        drivers=drivers,
        constructors=constructors,
        budget=130.0,
        max_transfers=3,
        search_mode="balanced",
        progress_callback=lambda payload: balanced_events.append(payload),
    )
    recs_exhaustive = build_transfer_recommendations(
        current_driver_ids=[f"d{i}" for i in range(1, 6)],
        current_constructor_ids=["c1", "c2"],
        drivers=drivers,
        constructors=constructors,
        budget=130.0,
        max_transfers=3,
        search_mode="exhaustive",
        progress_callback=lambda payload: exhaustive_events.append(payload),
    )
    assert not recs_balanced.empty
    assert not recs_exhaustive.empty
    balanced_final = balanced_events[-1]
    exhaustive_final = exhaustive_events[-1]
    assert int(balanced_final.get("transfer_candidate_count_total", 0) or 0) < int(
        exhaustive_final.get("transfer_candidate_count_total", 0) or 0
    )
    assert int(balanced_final.get("evaluated_full_candidates", 0) or 0) <= int(
        exhaustive_final.get("evaluated_full_candidates", 0) or 0
    )
    assert bool(balanced_final.get("exhaustive_scoring_used_after_prefilter", False)) is True
    assert bool(balanced_final.get("final_score_used_for_sorting", False)) is True
    assert balanced_final.get("candidate_pool_mode") == "balanced_prefiltered"


def test_transfer_balanced_matches_exhaustive_when_pool_is_not_reduced():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 9.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 8.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 7.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 6.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d6", "name": "D6", "price": 10.0, "exp_score": 16.0, "expected_price_gain": 0.2, "volatility": 6.0},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 7.0},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 9.0, "expected_price_gain": 0.1, "volatility": 7.0},
            {"id": "c3", "name": "C3", "price": 10.0, "exp_score": 18.0, "expected_price_gain": 0.2, "volatility": 8.0},
        ]
    )
    balanced = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=70.0,
        max_transfers=2,
        objective_mode=OBJECTIVE_POINTS_ONLY,
        search_mode="balanced",
    )
    exhaustive = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=70.0,
        max_transfers=2,
        objective_mode=OBJECTIVE_POINTS_ONLY,
        search_mode="exhaustive",
    )
    assert not balanced.empty
    assert not exhaustive.empty
    assert str(balanced.iloc[0]["IN"]) == str(exhaustive.iloc[0]["IN"])
    assert float(balanced.iloc[0]["Final recommendation score"]) == pytest.approx(
        float(exhaustive.iloc[0]["Final recommendation score"])
    )


def test_transfer_balanced_max_three_keeps_and_scores_multiple_depths():
    drivers = pd.DataFrame(
        [
            {
                "id": f"d{i}",
                "name": f"D{i}",
                "price": 7.5 + (i % 6),
                "exp_score": float(80 - i),
                "expected_price_gain": 0.04 * ((i % 5) + 1),
                "volatility": 4.0 + (i % 6),
            }
            for i in range(1, 22)
        ]
    )
    constructors = pd.DataFrame(
        [
            {
                "id": f"c{i}",
                "name": f"C{i}",
                "price": 10.0 + i,
                "exp_score": float(38 - i),
                "expected_price_gain": 0.05 * ((i % 4) + 1),
                "volatility": 7.0 + (i % 4),
            }
            for i in range(1, 9)
        ]
    )
    events = []
    recs = build_transfer_recommendations(
        current_driver_ids=[f"d{i}" for i in range(1, 6)],
        current_constructor_ids=["c1", "c2"],
        drivers=drivers,
        constructors=constructors,
        budget=140.0,
        free_transfers=3,
        max_transfers=3,
        search_mode="balanced",
        progress_callback=lambda payload: events.append(payload),
    )
    assert not recs.empty
    final_event = events[-1]
    generated_by_depth = final_event.get("generated_candidates_by_depth", {})
    scored_by_depth = final_event.get("fully_scored_by_depth", {})
    assert int(generated_by_depth.get(1, 0)) > 0
    assert int(generated_by_depth.get(2, 0)) > 0
    assert int(generated_by_depth.get(3, 0)) > 0
    assert int(scored_by_depth.get(1, 0)) > 0
    assert int(scored_by_depth.get(2, 0)) > 0
    assert int(scored_by_depth.get(3, 0)) > 0


def test_transfer_points_only_does_not_rank_negative_points_move_above_positive_move():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.10, "volatility": 5.0},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.10, "volatility": 5.0},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.10, "volatility": 5.0},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.10, "volatility": 5.0},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.10, "volatility": 5.0},
            {"id": "d6", "name": "HighPtsLowGain", "price": 20.0, "exp_score": 13.0, "expected_price_gain": -0.10, "volatility": 5.0},
            {"id": "d7", "name": "LowPtsHighGain", "price": 5.0, "exp_score": 8.0, "expected_price_gain": 0.60, "volatility": 5.0},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 12.0, "expected_price_gain": 0.10, "volatility": 7.0},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 12.0, "expected_price_gain": 0.10, "volatility": 7.0},
        ]
    )
    recs = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=100.0,
        free_transfers=1,
        max_transfers=1,
        objective_mode=OBJECTIVE_POINTS_ONLY,
        search_mode="balanced",
        top_n=5,
    )
    assert not recs.empty
    assert (pd.to_numeric(recs["Final recommendation score"], errors="coerce").fillna(-999).diff().dropna() <= 0).all()
    top = recs.iloc[0]
    assert float(top["Net expected points gain"]) >= 0
    assert "HighPtsLowGain" in str(top["IN"])


def test_transfer_price_growth_only_sorts_by_expected_price_gain_delta():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d6", "name": "D6", "price": 10.0, "exp_score": 9.0, "expected_price_gain": 0.3},
            {"id": "d7", "name": "D7", "price": 10.0, "exp_score": 14.0, "expected_price_gain": -0.1},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
        ]
    )
    recs = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=80.0,
        max_transfers=1,
        objective_mode=OBJECTIVE_PRICE_GROWTH_ONLY,
        search_mode="balanced",
    )
    assert not recs.empty
    deltas = pd.to_numeric(recs["Expected price gain delta"], errors="coerce").fillna(-999)
    assert (deltas.diff().dropna() <= 0).all()


def test_transfer_final_scores_are_objective_specific_and_separate_from_filter_score():
    drivers = pd.DataFrame(
        [
            {"id": f"d{i}", "name": f"D{i}", "price": 8.0 + i, "exp_score": 18.0 - i, "expected_price_gain": 0.05 * i, "volatility": 4.0 + i}
            for i in range(1, 8)
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 15.0, "exp_score": 20.0, "expected_price_gain": 0.05, "volatility": 8.0},
            {"id": "c2", "name": "C2", "price": 16.0, "exp_score": 19.0, "expected_price_gain": 0.06, "volatility": 8.5},
            {"id": "c3", "name": "C3", "price": 12.0, "exp_score": 21.0, "expected_price_gain": 0.20, "volatility": 7.5},
        ]
    )
    recs = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=100.0,
        free_transfers=1,
        max_transfers=2,
        objective_mode=OBJECTIVE_PRICE_GROWTH_ONLY,
        search_mode="balanced",
    )
    assert not recs.empty
    assert "Candidate filter score" in recs.columns
    assert "Final recommendation score" in recs.columns
    assert (pd.to_numeric(recs["Final recommendation score"], errors="coerce").fillna(-999).diff().dropna() <= 0).all()
    final_scores = pd.to_numeric(recs["Final recommendation score"], errors="coerce").fillna(0.0)
    filter_scores = pd.to_numeric(recs["Candidate filter score"], errors="coerce").fillna(0.0)
    assert ((final_scores - filter_scores).abs() > 1e-9).any()


def test_transfer_combined_sorts_by_real_combined_objective_improvement():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 5.0},
            {"id": "d6", "name": "D6", "price": 10.0, "exp_score": 13.0, "expected_price_gain": 0.2, "volatility": 5.0},
            {"id": "d7", "name": "D7", "price": 10.0, "exp_score": 16.0, "expected_price_gain": -0.3, "volatility": 5.0},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 8.0},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1, "volatility": 8.0},
        ]
    )
    recs = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=80.0,
        max_transfers=1,
        objective_mode=OBJECTIVE_COMBINED,
        price_gain_weight=10.0,
        search_mode="balanced",
    )
    assert not recs.empty
    scores = pd.to_numeric(recs["Final recommendation score"], errors="coerce").fillna(-999)
    objectives = pd.to_numeric(recs["Objective improvement"], errors="coerce").fillna(-999)
    assert (scores.diff().dropna() <= 0).all()
    assert (objectives.diff().dropna() <= 0).all()


def test_price_change_model_projection_table_columns():
    frame = pd.DataFrame(
        [
            {
                "id": "1",
                "name": "Driver One",
                "team": "Ferrari",
                "price": 20.0,
                "exp_score": 30.0,
                "next_race_expected_points": 30.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 15.0,
                "dnf_rate": 0.1,
                "volatility": 8.0,
            }
        ]
    )
    table = price_change_probability_matrix_table(
        frame,
        {
            "terrible_max": 0.60,
            "poor_min": 0.60,
            "poor_max": 0.90,
            "good_min": 0.90,
            "good_max": 1.20,
            "great_min": 1.20,
            "terrible_price_change": -0.6,
            "poor_price_change": -0.2,
            "good_price_change": 0.2,
            "great_price_change": 0.6,
        },
        predicted_points_col="next_race_expected_points",
    )
    assert "Expected Points" in table.columns
    assert "P(Terrible)" in table.columns
    assert "P(Poor)" in table.columns
    assert "P(Good)" in table.columns
    assert "P(Great)" in table.columns
    assert "Expected price gain" in table.columns
    assert "P(Price rise)" not in table.columns


def test_fantasy_card_labels_use_expected_points_and_three_tiles_only():
    html = fantasy_card_grid_html(
        pd.DataFrame(
            [
                {
                    "name": "Driver",
                    "team": "Ferrari",
                    "price": 24.4,
                    "exp_score": 56.7,
                    "expected_price_gain": 0.3,
                    "projected_price": 24.7,
                }
            ]
        ),
        asset_label="Driver",
    )

    assert "Exp Pts" in html
    assert "Exp Gain" in html
    assert "Projected Price" not in html
    assert "Exp Points" not in html
    assert "EV/race" not in html
    assert "Driver · Projected price" not in html
    assert html.count('class="f1-stat"') == 3


def test_fantasy_card_uses_2x_token_not_drs_sticker():
    html = fantasy_card_grid_html(
        pd.DataFrame(
            [
                {
                    "name": "Kimi Antonelli",
                    "team": "Mercedes",
                    "price": 24.4,
                    "exp_score": 56.7,
                    "expected_price_gain": 0.3,
                }
            ]
        ),
        boosted_driver="Kimi Antonelli",
        asset_label="Driver",
    )

    assert ">2x<" in html
    assert ">DRS<" not in html


def test_driver_and_constructor_cards_use_same_shared_component_class():
    driver_html = fantasy_asset_card_html(
        {"name": "Driver", "team": "Ferrari", "price": 10.0, "exp_score": 5.0, "expected_price_gain": 0.1},
        asset_label="Driver",
    )
    constructor_html = fantasy_asset_card_html(
        {"name": "Ferrari", "team": "Ferrari", "price": 24.5, "exp_score": 50.0, "expected_price_gain": 0.3},
        asset_label="Constructor",
    )
    assert 'class="f1-driver-card"' in driver_html
    assert 'class="f1-driver-card"' in constructor_html


def test_no_negative_floors_negative_expected_scores():
    frame = pd.DataFrame(
        [
            {"exp_score": -5.0, "next_race_expected_points": -3.0, "combined_objective_score": -10.0},
            {"exp_score": 4.0, "next_race_expected_points": 2.0, "combined_objective_score": 1.0},
        ]
    )

    out = apply_no_negative_scores(frame)

    assert out.loc[0, "exp_score"] == 0.0
    assert out.loc[0, "next_race_expected_points"] == 0.0
    assert out.loc[0, "combined_objective_score"] == 0.0
    assert out.loc[1, "exp_score"] == 4.0


def test_chip_expected_points_apply_to_points_only():
    drivers = pd.DataFrame(
        [
            {"name": "A", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.5},
            {"name": "B", "price": 10.0, "exp_score": 8.0, "expected_price_gain": 0.4},
            {"name": "C", "price": 10.0, "exp_score": 6.0, "expected_price_gain": 0.3},
            {"name": "D", "price": 10.0, "exp_score": 4.0, "expected_price_gain": 0.2},
            {"name": "E", "price": 10.0, "exp_score": 2.0, "expected_price_gain": 0.1},
        ]
    )
    constructors = pd.DataFrame([{"name": "C1", "price": 10.0, "exp_score": 5.0, "expected_price_gain": 0.1}])

    boosted, triple = select_chip_boost_drivers(drivers, "none")
    triple_boosted, triple_driver = select_chip_boost_drivers(drivers, "triple")

    assert boosted == "A"
    assert triple is None
    assert triple_driver == "A"
    assert triple_boosted == "B"
    assert triple_driver != triple_boosted
    assert team_expected_points_with_chips(drivers, constructors, "none", boosted, triple) == pytest.approx(45.0)
    assert team_expected_points_with_chips(drivers, constructors, "limitless", boosted, triple) == pytest.approx(45.0)
    assert team_expected_points_with_chips(drivers, constructors, "triple", triple_boosted, triple_driver) == pytest.approx(63.0)
    assert selected_assets_price_gain(drivers, constructors) == pytest.approx(1.6)


def test_price_growth_objective_assigns_2x_to_highest_expected_points_driver():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 30.0, "combined_objective_score": 0.05, "expected_price_gain": 0.05},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 28.0, "combined_objective_score": 0.04, "expected_price_gain": 0.04},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 26.0, "combined_objective_score": 0.03, "expected_price_gain": 0.03},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 12.0, "combined_objective_score": 0.20, "expected_price_gain": 0.20},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 10.0, "combined_objective_score": 0.18, "expected_price_gain": 0.18},
            {"id": "d6", "name": "D6", "price": 10.0, "exp_score": 8.0, "combined_objective_score": 0.16, "expected_price_gain": 0.16},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 20.0, "combined_objective_score": 0.05, "expected_price_gain": 0.05},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 18.0, "combined_objective_score": 0.06, "expected_price_gain": 0.06},
            {"id": "c3", "name": "C3", "price": 10.0, "exp_score": 16.0, "combined_objective_score": 0.04, "expected_price_gain": 0.04},
        ]
    )
    sol = run_optimizer(
        drivers,
        constructors,
        budget=70.0,
        top_k=1,
        drs_multiplier=1.0,
        objective_col="combined_objective_score",
        boost_col="exp_score",
        triple_multiplier=None,
    )[0]

    top_driver = sol.drivers.sort_values(["exp_score", "price"], ascending=False).iloc[0]["name"]
    assert sol.boosted_driver == top_driver

    price_gain_before = selected_assets_price_gain(sol.drivers, sol.constructors)
    alt_boosted = sol.drivers.sort_values("exp_score").iloc[0]["name"]
    _ = team_expected_points_with_chips(sol.drivers, sol.constructors, "none", boosted_driver=alt_boosted)
    price_gain_after = selected_assets_price_gain(sol.drivers, sol.constructors)
    assert price_gain_after == pytest.approx(price_gain_before)


def test_three_x_mode_assigns_distinct_top_two_by_expected_points():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 30.0, "combined_objective_score": 0.2},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 25.0, "combined_objective_score": 0.2},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 20.0, "combined_objective_score": 0.2},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 18.0, "combined_objective_score": 0.2},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 16.0, "combined_objective_score": 0.2},
            {"id": "d6", "name": "D6", "price": 10.0, "exp_score": 10.0, "combined_objective_score": 0.2},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 20.0, "combined_objective_score": 0.2},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 19.0, "combined_objective_score": 0.2},
            {"id": "c3", "name": "C3", "price": 10.0, "exp_score": 18.0, "combined_objective_score": 0.2},
        ]
    )
    sol = run_optimizer(
        drivers,
        constructors,
        budget=70.0,
        top_k=1,
        drs_multiplier=1.0,
        objective_col="combined_objective_score",
        boost_col="exp_score",
        triple_multiplier=3.0,
    )[0]

    ordered = sol.drivers.sort_values(["exp_score", "price"], ascending=False)["name"].tolist()
    assert sol.triple_driver == ordered[0]
    assert sol.boosted_driver == ordered[1]
    assert sol.triple_driver != sol.boosted_driver


def test_fantasy_card_display_matches_optimizer_boost_driver_assignment():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 30.0, "combined_objective_score": 0.2},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 25.0, "combined_objective_score": 0.2},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 20.0, "combined_objective_score": 0.2},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 18.0, "combined_objective_score": 0.2},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 16.0, "combined_objective_score": 0.2},
            {"id": "d6", "name": "D6", "price": 10.0, "exp_score": 10.0, "combined_objective_score": 0.2},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 20.0, "combined_objective_score": 0.2},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 19.0, "combined_objective_score": 0.2},
            {"id": "c3", "name": "C3", "price": 10.0, "exp_score": 18.0, "combined_objective_score": 0.2},
        ]
    )
    sol = run_optimizer(
        drivers,
        constructors,
        budget=70.0,
        top_k=1,
        drs_multiplier=1.0,
        objective_col="combined_objective_score",
        boost_col="exp_score",
        triple_multiplier=None,
    )[0]

    display = annotate_card_expected_points(sol.drivers, boosted_driver=sol.boosted_driver, triple_driver=sol.triple_driver)
    boosted_row = display[display["name"].astype(str) == str(sol.boosted_driver)].iloc[0]
    assert boosted_row["display_exp_score"] == pytest.approx(float(boosted_row["exp_score"]) * 2.0)


def test_card_helper_can_show_boosted_display_points_without_changing_price_gain():
    drivers = pd.DataFrame(
        [
            {"name": "A", "team": "Ferrari", "price": 10.0, "exp_score": 10.0, "display_exp_score": 20.0, "expected_price_gain": 0.5},
        ]
    )

    html = fantasy_card_grid_html(drivers, boosted_driver="A")

    assert ">20.00<" in html
    assert "+0.50" in html


def test_transfer_recommendations_generate_one_transfer_and_sort_by_objective():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "Driver 1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d2", "name": "Driver 2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d3", "name": "Driver 3", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d4", "name": "Driver 4", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d5", "name": "Driver 5", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d6", "name": "Driver 6", "price": 10.0, "exp_score": 20.0, "expected_price_gain": 0.2},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "Constructor 1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "c2", "name": "Constructor 2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
        ]
    )

    recs = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=70.0,
        free_transfers=1,
        max_transfers=1,
        objective_mode=OBJECTIVE_POINTS_ONLY,
    )

    assert not recs.empty
    assert recs.loc[0, "Transfers"] == 1
    assert "Driver 6" in recs.loc[0, "IN"]
    assert recs.loc[0, "Transfer penalty"] == 0
    assert recs.loc[0, "Net expected points gain"] == pytest.approx(20.0)
    assert recs.loc[0, "Expected price gain delta"] == pytest.approx(0.1)
    assert recs.loc[0, "Projected team value"] == pytest.approx(70.8)
    assert isinstance(recs.loc[0, "Move rows"], list)
    assert recs.loc[0, "Move rows"][0]["asset_type"] == "driver"


def test_transfer_recommendations_respect_budget_penalty_and_exclusions():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "Driver 1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d2", "name": "Driver 2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d3", "name": "Driver 3", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d4", "name": "Driver 4", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d5", "name": "Driver 5", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d6", "name": "Driver 6", "price": 30.0, "exp_score": 100.0, "expected_price_gain": 1.0},
            {"id": "d7", "name": "Driver 7", "price": 10.0, "exp_score": 20.0, "expected_price_gain": 0.2},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "Constructor 1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "c2", "name": "Constructor 2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "c3", "name": "Constructor 3", "price": 10.0, "exp_score": 20.0, "expected_price_gain": 0.2},
        ]
    )

    recs = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=70.0,
        free_transfers=1,
        max_transfers=2,
        allow_extra_transfers=True,
        transfer_penalty=10.0,
        excluded_driver_ids=["d6"],
        locked_driver_ids=["d1"],
    )

    assert not recs.empty
    assert not recs["IN"].str.contains("Driver 6").any()
    assert not recs["OUT"].str.contains("Driver 1").any()
    assert recs["Team cost"].max() <= 70.0
    two_transfer = recs[recs["Transfers"] == 2]
    assert two_transfer["Transfer penalty"].iloc[0] == 10.0


def test_transfer_recommendations_include_constructor_move_rows_when_relevant():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "c3", "name": "C3", "price": 10.0, "exp_score": 20.0, "expected_price_gain": 0.2},
        ]
    )
    recs = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=70.0,
        free_transfers=1,
        max_transfers=1,
    )
    has_constructor = any(
        any(move.get("asset_type") == "constructor" for move in moves)
        for moves in recs["Move rows"]
        if isinstance(moves, list)
    )
    assert has_constructor


def test_transfer_deltas_include_chip_boosts_but_price_gain_does_not():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "A", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.1},
            {"id": "d2", "name": "B", "price": 10.0, "exp_score": 8.0, "expected_price_gain": 0.1},
            {"id": "d3", "name": "C", "price": 10.0, "exp_score": 6.0, "expected_price_gain": 0.1},
            {"id": "d4", "name": "D", "price": 10.0, "exp_score": 4.0, "expected_price_gain": 0.1},
            {"id": "d5", "name": "E", "price": 10.0, "exp_score": 2.0, "expected_price_gain": 0.1},
            {"id": "d6", "name": "F", "price": 10.0, "exp_score": 20.0, "expected_price_gain": 0.1},
        ]
    )
    constructors = pd.DataFrame([{"id": "c1", "name": "C1", "price": 10.0, "exp_score": 5.0, "expected_price_gain": 0.1}, {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 5.0, "expected_price_gain": 0.1}])

    none = build_transfer_recommendations(["d1", "d2", "d3", "d4", "d5"], ["c1", "c2"], drivers, constructors, budget=70, free_transfers=1, max_transfers=1, chip_mode="none")
    triple = build_transfer_recommendations(["d1", "d2", "d3", "d4", "d5"], ["c1", "c2"], drivers, constructors, budget=70, free_transfers=1, max_transfers=1, chip_mode="triple")

    assert triple.loc[0, "Net expected points gain"] > none.loc[0, "Net expected points gain"]
    assert triple.loc[0, "Expected price gain delta"] == none.loc[0, "Expected price gain delta"]


def test_transfer_display_formatting_removes_explanation_and_rounds_values():
    recs = pd.DataFrame(
        [
            {
                "Rank": 1,
                "OUT": "A",
                "IN": "B",
                "Expected points gain": 1.2345,
                "Net expected points gain": 0.9876,
                "Expected price gain delta": -0.3333,
                "Objective improvement": 4.5678,
                "Explanation": "Long text",
            }
        ]
    )

    out = format_transfer_recommendations_display(recs)

    assert "Explanation" not in out.columns
    assert out.loc[0, "Expected points gain"] == 1.23
    assert out.loc[0, "Expected price gain delta"] == -0.33


def test_selector_labels_show_price_not_id_and_keep_id_mapping():
    frame = pd.DataFrame([{"id": "121", "name": "Sergio Perez", "price": 7.6}])
    labels = build_asset_option_labels(frame)
    assert labels["121"] == "Sergio Perez (7.60M)"


def test_recommendation_badges_cover_common_cases():
    row = {
        "Expected points gain": 2.5,
        "Expected price gain delta": 0.4,
        "Remaining budget": 1.0,
        "Transfer penalty": 0.0,
        "Extra transfers": 0,
        "Incoming volatility mean": 20.0,
        "Outgoing negative gain count": 1,
    }
    badges = recommendation_badges(row, risk_appetite="Conservative")
    assert "Points upgrade" in badges
    assert "Budget builder" in badges
    assert "Frees cash" in badges
    assert "Avoids price drop" in badges
    assert "No penalty" in badges
    assert "Risky / high variance" in badges
    assert "Conservative" in badges


def test_budget_helpers_auto_initialise_and_resolve_manual_override():
    assert auto_budget_from_team_cost(108.5, 0.2) == pytest.approx(108.7)
    assert resolve_budget_value(100.0, 108.5, 0.2, user_overridden=False) == pytest.approx(108.7)
    assert resolve_budget_value(111.2, 108.5, 0.2, user_overridden=True) == pytest.approx(111.2)


def test_adjust_money_value_handles_quick_button_steps_with_floor():
    assert adjust_money_value(10.0, +1.0) == pytest.approx(11.0)
    assert adjust_money_value(10.0, +0.1) == pytest.approx(10.1)
    assert adjust_money_value(10.0, -0.1) == pytest.approx(9.9)
    assert adjust_money_value(0.05, -1.0) == pytest.approx(0.0)


def test_transfer_ui_source_uses_shared_card_html_and_simplified_metrics():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "fantasy_asset_card_html" in source
    assert "fantasy_asset_card_html(out_asset" in source
    assert "fantasy_asset_card_html(in_asset" in source
    assert '.metric("Transfers used"' in source
    assert '.metric("Transfer penalty"' in source
    assert '.metric("Δ Expected points"' in source
    assert '.metric("Δ Expected price gain"' in source
    assert '.metric("Remaining budget"' in source
    assert '.metric("Extra transfers"' not in source
    assert '.metric("Net Δ points"' not in source
    assert '.metric("Δ Team value"' not in source
    assert "_money_input_with_inline_adjustments(" not in source
    assert "prefix=\"current_budget\"" not in source
    assert "_money_adjustment_buttons" not in source
    assert "transfer_objective_col, transfer_weight_col = st.columns(2)" in source
    assert 'format_points(pd.to_numeric(row.get("Transfer penalty"), errors="coerce"))' in source


def test_cards_use_shared_size_without_constructor_specific_wide_style():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert ".f1-constructor-card" not in source
    assert "display: flex;" in source
    assert "width: 252px;" in source
    assert "grid-template-columns: 252px auto 252px;" in source


def test_transfer_explanation_not_rendered_as_duplicate_caption():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'st.caption(str(row.get("Explanation") or ""))' not in source
    assert "_render_transfer_tradeoff_box(top)" in source
    assert "_render_transfer_tradeoff_box(rec)" in source
    assert "Canadian Grand Prix" not in source
    assert 'st.markdown("### Top recommendation")' in source
    assert 'title = "Top recommendation"' not in source


def test_transfer_summary_explanation_is_concise_tradeoff_sentence():
    drivers = pd.DataFrame(
        [
            {"id": "d1", "name": "D1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.0},
            {"id": "d2", "name": "D2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.0},
            {"id": "d3", "name": "D3", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.0},
            {"id": "d4", "name": "D4", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.0},
            {"id": "d5", "name": "D5", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.0},
            {"id": "d6", "name": "D6", "price": 10.0, "exp_score": 12.0, "expected_price_gain": 0.2},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"id": "c1", "name": "C1", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.0},
            {"id": "c2", "name": "C2", "price": 10.0, "exp_score": 10.0, "expected_price_gain": 0.0},
        ]
    )
    recs = build_transfer_recommendations(
        ["d1", "d2", "d3", "d4", "d5"],
        ["c1", "c2"],
        drivers,
        constructors,
        budget=70.0,
        free_transfers=1,
        max_transfers=1,
        objective_mode=OBJECTIVE_POINTS_ONLY,
    )
    assert not recs.empty
    explanation = str(recs.iloc[0]["Explanation"])
    assert explanation.startswith("This ")
    assert "Uses" not in explanation
    assert "after transfer penalties" not in explanation


def test_team_lock_deadline_parser_and_countdown_formatting():
    parsed = parse_team_lock_deadline_timestamp("2026-05-24T14:30:00Z")
    assert parsed is not None
    assert parsed.year == 2026
    assert parsed.month == 5
    assert parsed.day == 24
    assert parse_team_lock_deadline_timestamp("") is None
    assert parse_team_lock_deadline_timestamp("not-a-timestamp") is None

    now = parse_team_lock_deadline_timestamp("2026-05-20T12:00:00Z")
    target = parse_team_lock_deadline_timestamp("2026-05-22T15:05:00Z")
    assert format_countdown(target, now_utc=now) == "02D : 03H : 05M"
    assert format_countdown(None, now_utc=now) == "Team lock deadline unavailable"


def test_playerstats_deadline_parser_prefers_status_one_future_session():
    payload = {
        "Value": {
            "FixtureWiseStats": [
                {
                    "GamedayId": 5,
                    "RaceDayWise": [
                        {
                            "SessionType": "Sprint Qualifying",
                            "SessionStartDate": "2030-05-23T12:00:00-04:00",
                            "MatchStatus": "1",
                            "MeetingName": "Canadian Grand Prix",
                        },
                        {
                            "SessionType": "Qualifying",
                            "SessionStartDate": "2030-05-23T16:00:00-04:00",
                            "MatchStatus": "0",
                            "MeetingName": "Canadian Grand Prix",
                        },
                    ],
                }
            ]
        }
    }
    out = parse_team_lock_deadline_from_payload(payload)
    assert out["team_lock_deadline_source"] == "official_feed_playerstats_session_start"
    assert out["team_lock_deadline_utc"] is not None
    assert out["team_lock_deadline_raw_field"] == "FixtureWiseStats.RaceDayWise.SessionStartDate"
    assert out["team_lock_session_type"] == "Sprint Qualifying"


def test_playerstats_deadline_parser_handles_missing_without_timestamp():
    out = parse_team_lock_deadline_from_payload({"Value": {"FixtureWiseStats": []}})
    assert out["team_lock_deadline_source"] == "unavailable"
    assert out["team_lock_deadline_utc"] is None
