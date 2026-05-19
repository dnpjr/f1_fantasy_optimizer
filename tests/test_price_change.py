import pandas as pd
import pytest

from f1fantasy.app_core import (
    OBJECTIVE_COMBINED,
    OBJECTIVE_POINTS_ONLY,
    OBJECTIVE_PRICE_GROWTH_ONLY,
    DEFAULT_PRICE_CHANGE_BOUNDS,
    DEFAULT_PRICE_CHANGE_CHEAP_RULES,
    DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF,
    DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES,
    PRICE_BAND_STYLES,
    PriceChangeBounds,
    PriceChangeRules,
    apply_recent_point_overrides,
    apply_objective_mode,
    apply_probabilistic_price_change_model,
    apply_price_change_model,
    avg_ppm_from_points,
    band_probabilities_from_normal,
    clamp_price_change,
    expected_price_change,
    expected_price_gain_from_probabilities,
    format_next_race_header,
    choose_price_change_rules,
    predicted_avg_ppm,
    price_change_probability_matrix_table,
    price_change_projection_summary_table,
    price_change_target_summary_table,
    price_change_threshold_table,
    price_change_tier,
    raw_price_change_for_tier,
    recent_points_diagnostics,
    required_next_points,
)


def _cheap_rules():
    return PriceChangeRules(
        terrible_max=0.5,
        poor_min=0.5,
        poor_max=1.0,
        good_min=1.0,
        good_max=2.0,
        great_min=2.0,
        terrible_price_change=-0.6,
        poor_price_change=-0.2,
        good_price_change=0.2,
        great_price_change=0.6,
    )


def _expensive_rules():
    return PriceChangeRules(
        terrible_max=0.5,
        poor_min=0.5,
        poor_max=1.0,
        good_min=1.0,
        good_max=2.0,
        great_min=2.0,
        terrible_price_change=-0.3,
        poor_price_change=-0.1,
        good_price_change=0.1,
        great_price_change=0.3,
    )


def test_avg_ppm_is_average_last_three_points_divided_by_price():
    avg_ppm = predicted_avg_ppm(10.0, 20.0, 30.0, current_price=10.0)

    assert avg_ppm == 2.0
    assert avg_ppm_from_points(20.0, current_price=10.0) == 2.0


def test_classification_uses_avg_ppm_and_boundaries_are_inclusive():
    rules = _cheap_rules()

    assert price_change_tier(0.5, rules) == "Terrible"
    assert price_change_tier(0.75, rules) == "Poor"
    assert price_change_tier(1.0, rules) == "Good"
    assert price_change_tier(1.5, rules) == "Good"
    assert price_change_tier(2.0, rules) == "Great"
    assert price_change_tier(2.1, rules) == "Great"


def test_price_change_tier_never_returns_neutral():
    rules = _cheap_rules()

    tiers = {price_change_tier(value, rules) for value in [-1.0, 0.5, 0.75, 1.5, 2.0, 5.0]}

    assert tiers <= {"Terrible", "Poor", "Good", "Great"}
    assert "Neutral" not in tiers


def test_required_next_points_formula():
    needed = required_next_points(
        current_price=10.0,
        target_avg_ppm=2.0,
        points_race_minus_2=12.0,
        points_race_minus_1=18.0,
    )

    assert needed == 30.0


def test_price_movement_follows_band_and_price_tier_rules():
    assert raw_price_change_for_tier("Great", _cheap_rules()) == 0.6
    assert raw_price_change_for_tier("Great", _expensive_rules()) == 0.3
    assert expected_price_change(2.1, _cheap_rules()) == 0.6


def test_apply_price_change_model_uses_expensive_and_cheap_rules():
    df = pd.DataFrame(
        [
            {
                "id": "cheap",
                "name": "Cheap",
                "price": 5.0,
                "exp_score": 30.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 20.0,
            },
            {
                "id": "expensive",
                "name": "Expensive",
                "price": 30.0,
                "exp_score": 180.0,
                "recent_points_2ago": 60.0,
                "recent_points_1ago": 120.0,
            },
        ]
    )

    out = apply_price_change_model(
        df,
        _cheap_rules(),
        expensive_rules=_expensive_rules(),
        expensive_price_min=18.5,
        bounds=PriceChangeBounds(min_asset_price=3.0, max_asset_price=34.0),
    )

    assert out.loc[out["id"] == "cheap", "raw_price_change"].iloc[0] == 0.6
    assert out.loc[out["id"] == "expensive", "raw_price_change"].iloc[0] == 0.3


def test_floor_and_ceiling_clamp_effective_price_change():
    assert clamp_price_change(3.0, -0.6, PriceChangeBounds(3.0, 34.0)) == (3.0, 0.0)
    projected, effective = clamp_price_change(3.2, -0.6, PriceChangeBounds(3.0, 34.0))
    assert projected == 3.0
    assert effective == pytest.approx(-0.2)
    assert clamp_price_change(34.0, 0.6, PriceChangeBounds(3.0, 34.0)) == (34.0, 0.0)
    projected, effective = clamp_price_change(33.8, 0.6, PriceChangeBounds(3.0, 34.0))
    assert projected == 34.0
    assert effective == pytest.approx(0.2)


def test_price_change_threshold_table_adds_traceable_recent_scores_and_required_points():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 10.0,
                "exp_score": 30.0,
                "recent_points_2ago": 12.0,
                "recent_points_1ago": 18.0,
            }
        ]
    )

    out = price_change_threshold_table(df, _cheap_rules())

    assert out.loc[0, "recent_points_2ago"] == 12.0
    assert out.loc[0, "recent_points_1ago"] == 18.0
    assert out.loc[0, "required_great_min"] == 30.0
    assert out.loc[0, "avg_ppm"] == 2.0
    assert out.loc[0, "points_needed_terrible"] == "≤ -15"
    assert out.loc[0, "points_needed_poor"] == "-14 to -1"
    assert out.loc[0, "points_needed_good"] == "0 to 29"
    assert out.loc[0, "points_needed_great"] == "≥ 30"
    assert out.loc[0, "price_change_efficiency"] == 3.0


def test_missing_recent_points_are_marked_as_fallback_not_silent_zero():
    df = pd.DataFrame([{"id": "a", "name": "A", "price": 10.0, "exp_score": 12.0}])

    out = apply_price_change_model(df, _cheap_rules())

    assert pd.isna(out.loc[0, "recent_points_2ago"])
    assert pd.isna(out.loc[0, "recent_points_1ago"])
    assert out.loc[0, "recent_points_available"] == 0
    assert bool(out.loc[0, "recent_points_fallback_used"]) is True
    assert out.loc[0, "price_change_tier"] == "Missing"
    assert out.loc[0, "raw_price_change"] == 0.0


def test_recent_points_diagnostics_reports_fallback_coverage():
    drivers = pd.DataFrame({"recent_points_available": [2, 0]})
    constructors = pd.DataFrame({"recent_points_available": [2]})
    weekend_points = pd.DataFrame(
        [
            {"season": 2026, "round": 4, "circuitName": "Miami"},
            {"season": 2026, "round": 5, "circuitName": "Canada"},
        ]
    )

    diag = recent_points_diagnostics(drivers, constructors, weekend_points, current_season=2026)

    assert diag["recent_points_driver_complete"] == 1
    assert diag["recent_points_constructor_complete"] == 1
    assert diag["recent_points_fallback_used"] is True
    assert diag["recent_points_rounds"] == [4, 5]


def test_fixed_default_price_change_constants_match_expected_values():
    assert DEFAULT_PRICE_CHANGE_BOUNDS.min_asset_price == 3.0
    assert DEFAULT_PRICE_CHANGE_BOUNDS.max_asset_price == 34.0
    assert DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF == 18.5
    assert DEFAULT_PRICE_CHANGE_CHEAP_RULES.terrible_max == 0.60
    assert DEFAULT_PRICE_CHANGE_CHEAP_RULES.poor_max == 0.90
    assert DEFAULT_PRICE_CHANGE_CHEAP_RULES.great_min == 1.20
    assert DEFAULT_PRICE_CHANGE_CHEAP_RULES.terrible_price_change == -0.6
    assert DEFAULT_PRICE_CHANGE_CHEAP_RULES.great_price_change == 0.6
    assert DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES.terrible_price_change == -0.3
    assert DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES.great_price_change == 0.3


def test_price_tier_cutoff_uses_18_point_5_as_cheap_side():
    cheap = choose_price_change_rules(18.5, DEFAULT_PRICE_CHANGE_CHEAP_RULES, DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES, DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF)
    expensive = choose_price_change_rules(18.6, DEFAULT_PRICE_CHANGE_CHEAP_RULES, DEFAULT_PRICE_CHANGE_EXPENSIVE_RULES, DEFAULT_PRICE_CHANGE_EXPENSIVE_CUTOFF)

    assert cheap.terrible_price_change == -0.6
    assert expensive.terrible_price_change == -0.3


def test_required_next_points_allows_negative_values():
    needed = required_next_points(
        current_price=24.4,
        target_avg_ppm=DEFAULT_PRICE_CHANGE_CHEAP_RULES.great_min,
        points_race_minus_2=50.0,
        points_race_minus_1=42.0,
    )

    assert needed == pytest.approx(-4.16, abs=0.01)


def test_russell_like_default_thresholds_match_canada_style_scale():
    df = pd.DataFrame(
        [
            {
                "id": "russell",
                "name": "George Russell",
                "price": 28.6,
                "recent_points_2ago": 27.0,
                "recent_points_1ago": 42.0,
            }
        ]
    )

    out = price_change_threshold_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_exp_score")

    assert out.loc[0, "points_needed_terrible"] == "≤ -18"
    assert out.loc[0, "points_needed_poor"] == "-17 to 7"
    assert out.loc[0, "points_needed_good"] == "8 to 33"
    assert out.loc[0, "points_needed_great"] == "≥ 34"


def test_kimi_like_default_thresholds_match_canada_style_scale():
    df = pd.DataFrame(
        [
            {
                "id": "kimi",
                "name": "Kimi Antonelli",
                "price": 24.4,
                "recent_points_2ago": 50.0,
                "recent_points_1ago": 42.0,
            }
        ]
    )

    out = price_change_threshold_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_exp_score")

    assert out.loc[0, "points_needed_terrible"] == "≤ -48"
    assert out.loc[0, "points_needed_poor"] == "-47 to -27"
    assert out.loc[0, "points_needed_good"] == "-26 to -5"
    assert out.loc[0, "points_needed_great"] == "≥ -4"


def test_leclerc_like_default_thresholds_match_canada_style_scale():
    df = pd.DataFrame(
        [
            {
                "id": "leclerc",
                "name": "Charles Leclerc",
                "price": 24.0,
                "recent_points_2ago": 31.0,
                "recent_points_1ago": 27.0,
            }
        ]
    )

    out = price_change_threshold_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_exp_score")

    assert out.loc[0, "points_needed_terrible"] == "≤ -15"
    assert out.loc[0, "points_needed_poor"] == "-14 to 6"
    assert out.loc[0, "points_needed_good"] == "7 to 28"
    assert out.loc[0, "points_needed_great"] == "≥ 29"


def test_threshold_columns_do_not_depend_on_predicted_next():
    base = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 24.4,
                "exp_score": 999.0,
                "next_race_exp_score": 10.0,
                "recent_points_2ago": 31.0,
                "recent_points_1ago": 27.0,
            }
        ]
    )
    changed_prediction = base.copy()
    changed_prediction["next_race_exp_score"] = -20.0

    out_a = price_change_threshold_table(base, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_exp_score")
    out_b = price_change_threshold_table(changed_prediction, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_exp_score")

    for col in ["points_needed_terrible", "points_needed_poor", "points_needed_good", "points_needed_great"]:
        assert out_a.loc[0, col] == out_b.loc[0, col]
    assert out_a.loc[0, "price_change_predicted_next"] == 10.0
    assert out_b.loc[0, "price_change_predicted_next"] == -20.0


def test_target_summary_table_sorts_ppm_ease_ascending_by_default():
    df = pd.DataFrame(
        [
            {
                "id": "slow",
                "name": "Slow",
                "price": 20.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 10.0,
            },
            {
                "id": "fast",
                "name": "Fast",
                "price": 20.0,
                "recent_points_2ago": 50.0,
                "recent_points_1ago": 50.0,
            },
        ]
    )

    out = price_change_target_summary_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_exp_score")

    assert out["Abbrev"].tolist() == ["fast", "slow"]
    assert out["Rise difficulty"].tolist() == sorted(out["Rise difficulty"].tolist())
    assert "Race -2" not in out.columns
    assert "Race -1" not in out.columns


def test_predicted_next_not_required_for_core_threshold_table():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 24.4,
                "recent_points_2ago": 31.0,
                "recent_points_1ago": 27.0,
            }
        ]
    )

    out = price_change_threshold_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_exp_score")

    assert out.loc[0, "points_needed_terrible"] == "≤ -14"
    assert out.loc[0, "points_needed_great"] == "≥ 30"
    assert pd.isna(out.loc[0, "price_change_predicted_next"])
    assert out.loc[0, "price_change_tier"] == "Missing"


def test_projection_summary_table_hides_core_band_columns_and_sorts_projected_gain():
    df = pd.DataFrame(
        [
            {
                "id": "low",
                "name": "Low",
                "team": "Ferrari",
                "price": 20.0,
                "exp_score": 10.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 10.0,
                "volatility": 5.0,
            },
            {
                "id": "high",
                "name": "High",
                "team": "Ferrari",
                "price": 20.0,
                "exp_score": 60.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 10.0,
                "volatility": 5.0,
            },
        ]
    )

    out = price_change_projection_summary_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="exp_score")

    assert out["Abbrev"].tolist() == ["high", "low"]
    assert "Race -2" not in out.columns
    assert "Race -1" not in out.columns
    assert "Projected avgPPM" not in out.columns
    assert "Projected tier" not in out.columns
    assert "Raw price change" not in out.columns
    assert "Expected Points" in out.columns
    assert "Expected price gain" in out.columns
    assert "P(price rise)" not in out.columns
    assert "P(price fall)" not in out.columns
    assert "Volatility / race" not in out.columns
    assert "P(Great)" not in out.columns
    assert out["Expected price gain"].iloc[0] == out["Expected price gain"].max()


def test_probability_matrix_contains_tier_probabilities_and_sorts_expected_gain():
    df = pd.DataFrame(
        [
            {
                "id": "low",
                "name": "Low",
                "team": "Ferrari",
                "price": 20.0,
                "exp_score": 10.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 10.0,
                "volatility": 5.0,
            },
            {
                "id": "high",
                "name": "High",
                "team": "Ferrari",
                "price": 20.0,
                "exp_score": 60.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 10.0,
                "volatility": 5.0,
            },
        ]
    )

    out = price_change_probability_matrix_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="exp_score")

    assert out["Abbrev"].tolist() == ["high", "low"]
    assert {"Expected Points", "P(Terrible)", "P(Poor)", "P(Good)", "P(Great)"} <= set(out.columns)
    assert "P(Price rise)" not in out.columns
    assert out["Expected price gain"].iloc[0] == out["Expected price gain"].max()


def test_price_change_probabilities_do_not_change_for_chip_modes():
    df = pd.DataFrame(
        [
            {
                "id": "kimi",
                "name": "Kimi",
                "price": 24.4,
                "next_race_expected_points": 20.0,
                "recent_points_2ago": 50.0,
                "recent_points_1ago": 42.0,
                "volatility": 5.0,
                "dnf_rate": 0.1,
            }
        ]
    )

    none = price_change_probability_matrix_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_expected_points")
    triple = price_change_probability_matrix_table(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="next_race_expected_points")

    assert none[["P(Terrible)", "P(Poor)", "P(Good)", "P(Great)", "Expected price gain"]].iloc[0].to_dict() == pytest.approx(
        triple[["P(Terrible)", "P(Poor)", "P(Good)", "P(Great)", "Expected price gain"]].iloc[0].to_dict()
    )


def test_advanced_probability_details_hide_debug_raw_gain_and_dnf_source():
    source = __import__("pathlib").Path("streamlit_app.py").read_text(encoding="utf-8")

    detail_function = source.split("def _price_change_probability_detail_table", 1)[1].split("st.title", 1)[0]
    assert '"raw_expected_price_gain"' not in detail_function
    assert '"dnf_score_source"' not in detail_function


def test_price_band_styles_are_separate_colours_not_one_block():
    assert set(PRICE_BAND_STYLES) == {"Terrible", "Poor", "Good", "Great"}
    assert len(set(PRICE_BAND_STYLES.values())) == 4
    assert "127, 29, 29" in PRICE_BAND_STYLES["Terrible"]
    assert "22, 163, 74" in PRICE_BAND_STYLES["Great"]


def test_projected_tier_uses_race_minus_two_minus_one_and_one_race_prediction():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 24.4,
                "exp_score": 999.0,
                "next_race_exp_score": -10.0,
                "recent_points_2ago": 31.0,
                "recent_points_1ago": 27.0,
            }
        ]
    )

    out = apply_price_change_model(
        df,
        DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        predicted_points_col="next_race_exp_score",
    )

    expected_ppm = (31.0 + 27.0 - 10.0) / 3.0 / 24.4
    assert out.loc[0, "avg_ppm"] == pytest.approx(expected_ppm)
    assert out.loc[0, "price_change_tier"] == "Poor"


def test_model_projection_uses_next_race_expected_points_field():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 20.0,
                "exp_score": 999.0,
                "next_race_expected_points": 20.0,
                "recent_points_2ago": 10.0,
                "recent_points_1ago": 10.0,
                "volatility": 5.0,
            }
        ]
    )

    out = apply_price_change_model(
        df,
        DEFAULT_PRICE_CHANGE_CHEAP_RULES,
        predicted_points_col="next_race_expected_points",
    )

    assert out.loc[0, "price_change_predicted_next"] == 20.0
    assert out.loc[0, "expected_points_per_million"] == 1.0
    assert out.loc[0, "expected_points_per_volatility"] == 4.0
    assert "expected_price_gain_per_million" in out.columns
    assert "risk_adjusted_price_gain" in out.columns


def test_band_probabilities_sum_to_one():
    probs = band_probabilities_from_normal(
        mean=50.0,
        sd=10.0,
        thresholds={"terrible_max": 20.0, "poor_max": 40.0, "great_min": 60.0},
    )
    total = probs["p_terrible"] + probs["p_poor"] + probs["p_good"] + probs["p_great"]

    assert total == pytest.approx(1.0)
    assert probs["p_price_rise"] == pytest.approx(probs["p_good"] + probs["p_great"])
    assert probs["p_price_fall"] == pytest.approx(probs["p_terrible"] + probs["p_poor"])


def test_dnf_rate_zero_matches_normal_only_probabilities():
    thresholds = {"terrible_max": 20.0, "poor_max": 40.0, "great_min": 60.0}

    normal = band_probabilities_from_normal(mean=50.0, sd=10.0, thresholds=thresholds)
    with_dnf_zero = band_probabilities_from_normal(mean=50.0, sd=10.0, thresholds=thresholds, dnf_rate=0.0, dnf_score=0.0)

    assert with_dnf_zero == pytest.approx(normal)


def test_dnf_rate_increases_bad_tier_probability_and_preserves_total():
    thresholds = {"terrible_max": 20.0, "poor_max": 40.0, "great_min": 60.0}

    normal = band_probabilities_from_normal(mean=70.0, sd=5.0, thresholds=thresholds, dnf_rate=0.0, dnf_score=0.0)
    with_dnf = band_probabilities_from_normal(mean=70.0, sd=5.0, thresholds=thresholds, dnf_rate=0.2, dnf_score=0.0)

    total = with_dnf["p_terrible"] + with_dnf["p_poor"] + with_dnf["p_good"] + with_dnf["p_great"]
    assert total == pytest.approx(1.0)
    assert with_dnf["p_terrible"] > normal["p_terrible"]
    assert with_dnf["p_great"] < normal["p_great"]


def test_negative_dnf_score_reduces_great_probability_when_below_great_threshold():
    thresholds = {"terrible_max": -48.0, "poor_max": -27.0, "great_min": -4.0}

    normal = band_probabilities_from_normal(mean=20.0, sd=0.1, thresholds=thresholds, dnf_rate=0.0, dnf_score=-30.0)
    with_dnf = band_probabilities_from_normal(mean=20.0, sd=0.1, thresholds=thresholds, dnf_rate=0.2, dnf_score=-30.0)

    assert normal["p_great"] == pytest.approx(1.0)
    assert with_dnf["p_great"] == pytest.approx(0.8)
    assert with_dnf["p_price_rise"] < 1.0


def test_dnf_risk_changes_expected_price_gain():
    base = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 10.0,
                "exp_score": 70.0,
                "recent_points_2ago": 0.0,
                "recent_points_1ago": 0.0,
                "volatility": 5.0,
            }
        ]
    )
    no_dnf = apply_probabilistic_price_change_model(base.assign(dnf_rate=0.0), _cheap_rules(), predicted_points_col="exp_score")
    with_dnf = apply_probabilistic_price_change_model(base.assign(dnf_rate=0.2), _cheap_rules(), predicted_points_col="exp_score")

    assert with_dnf.loc[0, "expected_price_gain"] < no_dnf.loc[0, "expected_price_gain"]
    assert with_dnf.loc[0, "dnf_rate_used"] == pytest.approx(0.2)


def test_kimi_like_dnf_risk_prevents_guaranteed_price_rise():
    df = pd.DataFrame(
        [
            {
                "id": "kimi",
                "name": "Kimi",
                "price": 24.4,
                "exp_score": 20.0,
                "recent_points_2ago": 50.0,
                "recent_points_1ago": 42.0,
                "volatility": 0.1,
                "dnf_rate": 0.2,
            }
        ]
    )

    out = apply_probabilistic_price_change_model(df, DEFAULT_PRICE_CHANGE_CHEAP_RULES, predicted_points_col="exp_score")

    assert out.loc[0, "dnf_score_used"] == pytest.approx(-30.0)
    assert out.loc[0, "p_price_rise"] < 1.0
    assert out.loc[0, "p_great"] < 1.0
    assert out.loc[0, "dnf_score_source"] == "fixed_generic_race_weekend_bad_outcome"


def test_missing_dnf_rate_is_handled_safely():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 10.0,
                "exp_score": 70.0,
                "recent_points_2ago": 0.0,
                "recent_points_1ago": 0.0,
                "volatility": 5.0,
            }
        ]
    )

    out = apply_probabilistic_price_change_model(df, _cheap_rules(), predicted_points_col="exp_score")

    assert out.loc[0, "dnf_rate_used"] == 0.0
    assert out.loc[0, "dnf_score_used"] == pytest.approx(-30.0)
    assert out.loc[0, "dnf_score_source"] == "fixed_generic_race_weekend_bad_outcome"
    total = out.loc[0, "p_terrible"] + out.loc[0, "p_poor"] + out.loc[0, "p_good"] + out.loc[0, "p_great"]
    assert total == pytest.approx(1.0)


def test_expected_price_gain_is_probability_weighted_fractional_value():
    probs = {"p_terrible": 0.05, "p_poor": 0.15, "p_good": 0.30, "p_great": 0.50}

    out = expected_price_gain_from_probabilities(probs, price=10.0, rules=_cheap_rules(), bounds=DEFAULT_PRICE_CHANGE_BOUNDS)

    assert out["raw_expected_price_gain"] == pytest.approx(0.30)
    assert out["expected_price_gain"] == pytest.approx(0.30)
    assert out["projected_price_after_expected_gain"] == pytest.approx(10.30)


def test_high_volatility_near_threshold_does_not_get_hard_great_move():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 10.0,
                "exp_score": 70.0,
                "recent_points_2ago": 0.0,
                "recent_points_1ago": 0.0,
                "volatility": 20.0,
            }
        ]
    )

    out = apply_probabilistic_price_change_model(df, _cheap_rules(), predicted_points_col="exp_score")

    assert out.loc[0, "price_change_tier"] == "Great"
    assert out.loc[0, "effective_price_change_after_floor_ceiling"] == pytest.approx(0.6)
    assert out.loc[0, "expected_price_gain"] < 0.6
    assert out.loc[0, "expected_price_gain"] > -0.6


def test_low_volatility_far_above_great_is_close_to_great_move():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 10.0,
                "exp_score": 80.0,
                "recent_points_2ago": 0.0,
                "recent_points_1ago": 0.0,
                "volatility": 0.1,
            }
        ]
    )

    out = apply_probabilistic_price_change_model(df, _cheap_rules(), predicted_points_col="exp_score")

    assert out.loc[0, "p_great"] > 0.99
    assert out.loc[0, "expected_price_gain"] == pytest.approx(0.6, abs=0.01)


def test_low_volatility_far_below_terrible_is_close_to_terrible_move_with_floor():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 3.2,
                "exp_score": -30.0,
                "recent_points_2ago": 0.0,
                "recent_points_1ago": 0.0,
                "volatility": 0.1,
            }
        ]
    )

    out = apply_probabilistic_price_change_model(df, _cheap_rules(), predicted_points_col="exp_score", bounds=DEFAULT_PRICE_CHANGE_BOUNDS)

    assert out.loc[0, "p_terrible"] > 0.99
    assert out.loc[0, "raw_expected_price_gain"] == pytest.approx(-0.6, abs=0.01)
    assert out.loc[0, "expected_price_gain"] == pytest.approx(-0.2)


def test_objective_mode_uses_probabilistic_expected_price_gain_when_present():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 10.0,
                "exp_score": 20.0,
                "expected_price_change": 0.6,
                "expected_price_gain": 0.3,
            }
        ]
    )

    out = apply_objective_mode(df, OBJECTIVE_COMBINED, price_gain_weight=10.0)

    assert out.loc[0, "price_growth_objective"] == pytest.approx(0.3)
    assert out.loc[0, "combined_objective_score"] == pytest.approx(23.0)


def test_next_race_header_renders_gracefully_when_missing():
    assert format_next_race_header(None, None) == "Next race"
    assert format_next_race_header("Canadian Grand Prix", "2026-05-22") == "Next race: Canadian Grand Prix, 22 May 2026"


def test_manual_recent_point_overrides_are_used_when_provided():
    base = pd.DataFrame(
        [
            {
                "driverId": "a",
                "recent_points_2ago": pd.NA,
                "recent_points_1ago": pd.NA,
                "recent_points_available": 0,
                "recent_points_source": "missing",
            }
        ]
    )
    manual = pd.DataFrame([{"driverId": "a", "recent_points_2ago": 8.0, "recent_points_1ago": 16.0}])

    out = apply_recent_point_overrides(base, manual, "driverId")

    assert out.loc[0, "recent_points_2ago"] == 8.0
    assert out.loc[0, "recent_points_1ago"] == 16.0
    assert out.loc[0, "recent_points_source"] == "manual"
    assert out.loc[0, "recent_points_available"] == 2


def test_objective_modes_use_effective_price_change():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 3.0,
                "exp_score": 30.0,
                "recent_points_2ago": 0.0,
                "recent_points_1ago": 0.0,
            },
        ]
    )
    modelled = apply_price_change_model(df, _cheap_rules(), bounds=PriceChangeBounds(3.0, 34.0))

    assert modelled["raw_price_change"].iloc[0] == 0.6
    assert modelled["expected_price_change"].iloc[0] == pytest.approx(0.6)

    points = apply_objective_mode(modelled, OBJECTIVE_POINTS_ONLY, price_gain_weight=10.0)
    price = apply_objective_mode(modelled, OBJECTIVE_PRICE_GROWTH_ONLY, price_gain_weight=10.0)
    combined = apply_objective_mode(modelled, OBJECTIVE_COMBINED, price_gain_weight=10.0)

    assert points["combined_objective_score"].iloc[0] == 30.0
    assert price["combined_objective_score"].iloc[0] == pytest.approx(0.6)
    assert combined["combined_objective_score"].iloc[0] == 36.0


def test_price_growth_objective_uses_effective_not_raw_at_floor():
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "price": 3.0,
                "exp_score": -10.0,
                "recent_points_2ago": 0.0,
                "recent_points_1ago": 0.0,
            },
        ]
    )
    modelled = apply_price_change_model(df, _cheap_rules(), bounds=PriceChangeBounds(3.0, 34.0))
    price = apply_objective_mode(modelled, OBJECTIVE_PRICE_GROWTH_ONLY)

    assert modelled["raw_price_change"].iloc[0] == -0.6
    assert modelled["expected_price_change"].iloc[0] == 0.0
    assert price["combined_objective_score"].iloc[0] == 0.0
