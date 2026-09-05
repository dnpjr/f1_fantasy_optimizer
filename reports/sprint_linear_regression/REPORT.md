# Sprint EV linear-regression research

## 1. Executive conclusion

Leakage-safe 2026 OLS candidates:

- Driver: **alpha = 8.4501, beta = 0.7320**
- Constructor: **alpha = 15.0865, beta = 0.8660**

Validation-selected historical/2026 blends:

- Driver: **alpha = 7.0530, beta = 0.7740**
  with 2026 weight 0.50.
- Constructor: **alpha = 10.1712, beta = 0.9435**
  with 2026 weight 0.25.

Leave-one-Sprint-event-out validation selects `4_2026_weighted_hybrid` for drivers and
`4_2026_weighted_hybrid` for constructors. These are research candidates only; no
coefficient is activated in production.

## 2. 2026 results

| entity_type | estimator | alpha | beta | standard_error_alpha | standard_error_beta | r_squared | mae | rmse | observation_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| driver | OLS | 8.4501 | 0.7320 | 2.0688 | 0.1137 | 0.3254 | 12.8686 | 16.3199 | 88.0000 |
| driver | Huber_IRLS | 9.6136 | 0.7246 | 1.7909 | 0.1007 | 0.3223 | 12.8309 | 16.3568 | 88.0000 |
| constructor | OLS | 15.0865 | 0.8660 | 4.7994 | 0.1124 | 0.5973 | 18.1968 | 22.6717 | 42.0000 |
| constructor | Huber_IRLS | 15.2414 | 0.8533 | 4.2897 | 0.1002 | 0.5971 | 18.2167 | 22.6763 | 42.0000 |

Event-cluster bootstrap intervals:

| entity_type | coefficient | p2_5 | p50 | p97_5 | bootstrap_samples |
| --- | --- | --- | --- | --- | --- |
| driver | alpha | 5.1176 | 8.4501 | 11.4000 | 10000.0000 |
| driver | beta | 0.6638 | 0.7320 | 0.8641 | 10000.0000 |
| constructor | alpha | 8.7258 | 15.0865 | 21.2081 | 10000.0000 |
| constructor | beta | 0.7240 | 0.8660 | 1.0166 | 10000.0000 |

Leave-one-Sprint-event coefficient sensitivity is in `leave_one_sprint_out.csv`. Because all assets
within a weekend share conditions, the event-cluster intervals—not row-level OLS standard errors—are
the preferred uncertainty statement.

## 3. Individual 2026 assets

Every driver has four Sprint observations; one constructor has only three. With two fitted parameters,
raw individual fits have at most two residual degrees of freedom. 33 fits additionally fail
the conservative variation/conditioning reliability screen. Raw coefficients are descriptive and
unstable.

Ridge penalties selected by leave-one-event-out validation are 100 for
drivers and 100 for constructors. Largest shrunk deviations:

| entity | entity_type | sprint_observations | shrunk_alpha | shrunk_beta | alpha_deviation_from_group | beta_deviation_from_group |
| --- | --- | --- | --- | --- | --- | --- |
| Aston Martin | constructor | 4.0000 | 16.2043 | 0.8430 | 1.1178 | -0.0230 |
| Audi | constructor | 3.0000 | 13.9791 | 0.8718 | -1.1074 | 0.0058 |
| Cadillac | constructor | 4.0000 | 15.6729 | 0.8550 | 0.5864 | -0.0110 |
| Racing Bulls | constructor | 4.0000 | 14.7143 | 0.8676 | -0.3722 | 0.0016 |
| Red Bull | constructor | 4.0000 | 14.8250 | 0.8619 | -0.2615 | -0.0041 |
| Lance Stroll | driver | 4.0000 | 9.3125 | 0.6837 | 0.8624 | -0.0482 |
| Alexander Albon | driver | 4.0000 | 7.8293 | 0.7430 | -0.6208 | 0.0111 |
| Arvid Lindblad | driver | 4.0000 | 7.8801 | 0.7324 | -0.5700 | 0.0005 |
| Nico Hülkenberg | driver | 4.0000 | 7.9861 | 0.7438 | -0.4640 | 0.0118 |
| Gabriel Bortoleto | driver | 4.0000 | 8.0461 | 0.7281 | -0.4040 | -0.0039 |

The group coefficients are more reliable than any individual pair. No mixed-effects package was
available, so ridge partial pooling was used without adding a dependency.

## 4. Season-by-season results

Strict predictive walk-forward regressions:

| period | entity_type | alpha | beta | r_squared | mae | observation_count | excluded_observation_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | driver | 4.0580 | 0.9411 | 0.2917 | 13.0975 | 120.0000 | 0.0000 |
| 2023 | constructor | 6.0802 | 1.0285 | 0.5364 | 17.4979 | 60.0000 | 0.0000 |
| 2024 | driver | 4.9428 | 0.9047 | 0.3037 | 10.8729 | 119.0000 | 1.0000 |
| 2024 | constructor | 8.3674 | 1.0087 | 0.5954 | 15.1474 | 60.0000 | 0.0000 |
| 2025 | driver | 6.0365 | 0.7573 | 0.2807 | 12.4773 | 120.0000 | 0.0000 |
| 2025 | constructor | 9.4087 | 0.9342 | 0.4523 | 18.1315 | 60.0000 | 0.0000 |
| 2026 | driver | 8.4501 | 0.7320 | 0.3254 | 12.8686 | 88.0000 | 0.0000 |
| 2026 | constructor | 15.0865 | 0.8660 | 0.5973 | 18.1968 | 42.0000 | 1.0000 |

Strict baselines excluded 1 driver and
1 constructor observations. The shrunk sample explicitly imputed
or materially shrank 93 and 43, respectively.

## 5. Pooled results

| period | entity_type | alpha | beta | r_squared | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- |
| 2023-2025 | driver | 5.1929 | 0.8562 | 0.2898 | 12.1420 | 15.5221 |
| 2023-2025 | constructor | 7.9856 | 0.9895 | 0.5249 | 16.8486 | 22.2558 |
| 2023-2026 | driver | 6.0117 | 0.8164 | 0.2949 | 12.2908 | 15.7193 |
| 2023-2026 | constructor | 9.6793 | 0.9532 | 0.5384 | 17.1633 | 22.4259 |

Season-normalised pooling, raw fixed-effects estimates, equal-season weighting, 50% 2026 weighting,
and validation-selected weighting are recorded in `normalised_coefficients.csv` and
`overall_coefficients.csv`.

## 6. Model comparison

| entity_type | model | mae | rmse | bias | rank_correlation | top_asset_overlap | calibration_slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| driver | 0_no_adjustment | 14.9875 | 17.8381 | -5.8852 | 0.5727 | 0.6500 | 0.7320 |
| driver | 1_global_multiplier | 15.0978 | 17.9238 | -6.1023 | 0.5713 | 0.6500 | 0.7382 |
| driver | 2_global_additive | 13.6385 | 17.0473 | -0.0000 | 0.5533 | 0.6500 | 0.7155 |
| driver | 3_hybrid_linear | 13.1859 | 16.5780 | -0.0194 | 0.5395 | 0.6500 | 0.9517 |
| driver | 4_2026_weighted_hybrid | 13.1181 | 16.5008 | -1.0075 | 0.5577 | 0.6500 | 0.9224 |
| driver | 5_shrunk_asset_hybrid | 13.2057 | 16.5792 | 0.0570 | 0.5430 | 0.6000 | 0.9513 |
| constructor | 0_no_adjustment | 20.4940 | 25.6824 | -11.2845 | 0.7700 | 0.8333 | 0.8660 |
| constructor | 1_global_multiplier | 21.3646 | 26.2414 | -8.2821 | 0.7657 | 0.8333 | 0.7640 |
| constructor | 2_global_additive | 19.8776 | 24.1568 | 0.1515 | 0.7497 | 0.8333 | 0.8373 |
| constructor | 3_hybrid_linear | 19.9741 | 24.2287 | 0.1611 | 0.7440 | 0.8333 | 0.9493 |
| constructor | 4_2026_weighted_hybrid | 18.6684 | 23.3562 | -2.6749 | 0.7602 | 0.8333 | 0.9081 |
| constructor | 5_shrunk_asset_hybrid | 20.0031 | 24.2138 | 0.3707 | 0.7432 | 0.8333 | 0.9537 |

Metrics are out-of-event predictions from leaving each completed 2026 Sprint out in turn. Historical
prices were retained for residual strength diagnostics, but the production optimiser was deliberately
not invoked or changed.

## 7. Strength behaviour

Representative 2026 OLS predictions:

| entity_type | normal_ev | predicted_sprint_ev | absolute_uplift | effective_multiplier |
| --- | --- | --- | --- | --- |
| driver | 0.0000 | 8.4501 | 8.4501 |  |
| driver | 5.0000 | 12.1099 | 7.1099 | 2.4220 |
| driver | 10.0000 | 15.7698 | 5.7698 | 1.5770 |
| driver | 15.0000 | 19.4296 | 4.4296 | 1.2953 |
| driver | 20.0000 | 23.0895 | 3.0895 | 1.1545 |
| driver | 25.0000 | 26.7493 | 1.7493 | 1.0700 |
| constructor | 0.0000 | 15.0865 | 15.0865 |  |
| constructor | 10.0000 | 23.7469 | 13.7469 | 2.3747 |
| constructor | 20.0000 | 32.4074 | 12.4074 | 1.6204 |
| constructor | 30.0000 | 41.0678 | 11.0678 | 1.3689 |
| constructor | 40.0000 | 49.7282 | 9.7282 | 1.2432 |
| constructor | 50.0000 | 58.3887 | 8.3887 | 1.1678 |

Residual summaries by within-event price and pre-event-form tiers are in
`residual_by_strength.csv`. They diagnose—not prescribe—possible non-linearity.

Constructor-specific interpretation: the recorded 2026 constructor hybrid has alpha
`15.0865` and beta `0.8660`. Its implied uplifts across the requested
strength grid are:

| normal_ev | predicted_sprint_ev | absolute_uplift | effective_multiplier |
| --- | --- | --- | --- |
| 0.0000 | 15.0865 | 15.0865 |  |
| 10.0000 | 23.7469 | 13.7469 | 2.3747 |
| 20.0000 | 32.4074 | 12.4074 | 1.6204 |
| 30.0000 | 41.0678 | 11.0678 | 1.3689 |
| 40.0000 | 49.7282 | 9.7282 | 1.2432 |
| 50.0000 | 58.3887 | 8.3887 | 1.1678 |

This directly uses recorded constructor totals. The observation file records 11 constructors in the
first three completed 2026 Sprints and 10 in round 9, compared with 10 in earlier seasons.
The positive intercept with beta below 1 explains why a large fixed Sprint opportunity can coexist
with a lower pooled percentage multiplier. For the 2026 OLS line, absolute uplift is
`alpha + (beta - 1) × normal_ev`, declining from `15.0865` at zero normal EV to
`8.3887` at normal EV 50. The current sample therefore
does **not** show stronger constructors receiving larger absolute Sprint uplifts. Per-asset scoring
prevents the additional 2026 constructor from mechanically inflating the regression.

## 8. Scoring-rule sensitivity

| period | entity_type | alpha | beta | r_squared | mae |
| --- | --- | --- | --- | --- | --- |
| 2023-2025 | driver | 0.4800 | 0.8489 | 0.2891 | 1.1004 |
| 2023-2025 | constructor | 0.2718 | 0.9925 | 0.5297 | 0.5755 |
| 2023-2026 | driver | 0.5816 | 0.8073 | 0.2981 | 1.1602 |
| 2023-2026 | constructor | 0.3395 | 0.9536 | 0.5453 | 0.5979 |

Season normalisation divides both baseline and Sprint score by that season/entity normal-event mean.
It is the main cross-season comparability view; raw coefficients remain the directly interpretable
point-scale candidates.

## 9. Recommendation

Use the validation-selected historical/2026 hybrid as the specific first shadow-production candidate:
`Sprint EV = 7.0530 + 0.7740 × normal EV`
for drivers and
`Sprint EV = 10.1712 + 0.9435 × normal EV`
for constructors. The direct 2026 OLS lines remain the contemporary sensitivity, while individual
ridge estimates remain diagnostic. Do not activate any coefficient until shadow forecasts confirm
calibration on new Sprint events.

## 10. Limitations

- Only four 2026 Sprint weekends are complete.
- Asset outcomes within a Sprint event are correlated; only four event clusters drive uncertainty.
- Crashes, DNFs, penalties, negative scores and zeroes are retained as realised outcomes.
- Individual two-parameter regressions are extremely sparse and often ill-conditioned.
- Fantasy scoring rules and point scales change across seasons.
- Replacement drivers and the one sparse constructor have less history.
- A single line may miss threshold effects or other non-linearity.
- Prices and strength evolve; price tiers are within-event diagnostics, not causal controls.
- Descriptive full-season baselines use future normal events and are explicitly non-predictive.

Source: `data/generated/historical_fantasy_scores_v3_recorded_2023_2026/historical_fantasy_scores_2023_2026.csv`. Only recorded, non-reconstructed 2023–2026 canonical totals and verified
local Sprint schedule metadata were used. No network call or production-model change was made.
