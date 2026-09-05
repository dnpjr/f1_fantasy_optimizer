# Recorded F1 Fantasy Sprint-multiplier analysis

## 1. Executive conclusion

Recommended driver Sprint multiplier: **x_driver = 1.3425**
Recommended constructor Sprint multiplier: **x_constructor = 1.2505**

Event-level bootstrap 95% intervals are 1.2265–1.4654
for drivers and 1.1636–1.3448
for constructors. These are crude pooled associations, suitable only as a transparent first approximation.

## 2. Data coverage

Source: `data/generated/historical_fantasy_scores_v3_recorded_2023_2026/historical_fantasy_scores_2023_2026.csv`. Only canonical recorded totals from 2023–2026 were used; reconstructed
scores and 2021–2022 were excluded. Schedule metadata, rather than score components, defined format.

| season | normal | sprint |
| --- | --- | --- |
| 2023 | 16 | 6 |
| 2024 | 18 | 6 |
| 2025 | 18 | 6 |
| 2026 | 7 | 4 |

Sprint events:

| season | round | event_name | event_date |
| --- | --- | --- | --- |
| 2023 | 4 | Azerbaijan Grand Prix | 2023-04-30 |
| 2023 | 9 | Austrian Grand Prix | 2023-07-02 |
| 2023 | 12 | Belgian Grand Prix | 2023-07-30 |
| 2023 | 17 | Qatar Grand Prix | 2023-10-08 |
| 2023 | 18 | United States Grand Prix | 2023-10-22 |
| 2023 | 20 | São Paulo Grand Prix | 2023-11-05 |
| 2024 | 5 | Chinese Grand Prix | 2024-04-21 |
| 2024 | 6 | Miami Grand Prix | 2024-05-05 |
| 2024 | 11 | Austrian Grand Prix | 2024-06-30 |
| 2024 | 19 | United States Grand Prix | 2024-10-20 |
| 2024 | 21 | São Paulo Grand Prix | 2024-11-03 |
| 2024 | 23 | Qatar Grand Prix | 2024-12-01 |
| 2025 | 2 | Chinese Grand Prix | 2025-03-23 |
| 2025 | 6 | Miami Grand Prix | 2025-05-04 |
| 2025 | 13 | Belgian Grand Prix | 2025-07-27 |
| 2025 | 19 | United States Grand Prix | 2025-10-19 |
| 2025 | 21 | São Paulo Grand Prix | 2025-11-09 |
| 2025 | 23 | Qatar Grand Prix | 2025-11-30 |
| 2026 | 2 | Chinese Grand Prix | 2026-03-15 |
| 2026 | 4 | Miami Grand Prix | 2026-05-03 |
| 2026 | 5 | Canadian Grand Prix | 2026-05-24 |
| 2026 | 9 | British Grand Prix | 2026-07-05 |

## 3. Main calculations

| entity_type | normal_event_count | sprint_event_count | normal_asset_observations | sprint_asset_observations | normal_mean_points | sprint_mean_points | additive_uplift | multiplicative_uplift | event_means_multiplier | median_based_multiplier | trimmed_mean_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| driver | 59 | 22 | 1191 | 448 | 10.7725 | 14.4621 | 3.6896 | 1.3425 | 1.3398 | 1.2894 | 1.3318 |
| constructor | 59 | 22 | 596 | 223 | 28.8792 | 36.1121 | 7.2329 | 1.2505 | 1.2495 | 1.2439 | 1.2391 |

Season-by-season and pooled results are in `season_summary.csv` and `pooled_summary.csv`.

| period | entity_type | normal_event_count | sprint_event_count | normal_mean_points | sprint_mean_points | additive_uplift | multiplicative_uplift | median_based_multiplier | trimmed_mean_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | driver | 16 | 6 | 11.9781 | 14.7083 | 2.7302 | 1.2279 | 1.2620 | 1.2544 |
| 2023 | constructor | 16 | 6 | 29.9875 | 35.5833 | 5.5958 | 1.1866 | 1.2021 | 1.2074 |
| 2024 | driver | 18 | 6 | 10.6778 | 14.7167 | 4.0389 | 1.3783 | 1.4371 | 1.3498 |
| 2024 | constructor | 18 | 6 | 27.5000 | 35.5833 | 8.0833 | 1.2939 | 1.3339 | 1.2720 |
| 2025 | driver | 18 | 6 | 10.4250 | 13.2333 | 2.8083 | 1.2694 | 1.2642 | 1.2713 |
| 2025 | constructor | 18 | 6 | 30.1667 | 35.3500 | 5.1833 | 1.1718 | 1.1315 | 1.1708 |
| 2026 | driver | 7 | 4 | 9.2715 | 15.4545 | 6.1830 | 1.6669 | 1.8159 | 1.6759 |
| 2026 | constructor | 7 | 4 | 26.7632 | 38.6512 | 11.8880 | 1.4442 | 1.3875 | 1.4504 |

Negative, zero and unusually poor recorded observations were retained: 387 negative
asset-event scores and 44 genuine zero scores. No outlier filtering was applied.

## 4. Stability

| entity_type | estimate | p2_5 | p50 | p97_5 | bootstrap_samples | random_seed |
| --- | --- | --- | --- | --- | --- | --- |
| driver | 1.3425 | 1.2265 | 1.3416 | 1.4654 | 10000 | 20260806 |
| constructor | 1.2505 | 1.1636 | 1.2501 | 1.3448 | 10000 | 20260806 |

Leave-one-Sprint-out ranges are **1.3141–1.3582**
for drivers and **1.2315–1.2640** for
constructors. Leave-one-season-out ranges are **1.2938–1.3913**
and **1.2164–1.2850**, respectively.
Full exclusions are in their corresponding CSV files.

Cross-season mean absolute errors for the crude alternatives:

| entity_type | season_mae_multiplicative | season_mae_separate_additive | season_mae_global_additive | season_mae_no_adjustment |
| --- | --- | --- | --- | --- |
| driver | 1.7357 | 1.4825 | 1.8868 | 3.9401 |
| constructor | 3.4686 | 2.9873 | 2.8891 | 7.6876 |

The one combined additive uplift is `c = 4.8458` points per asset. Separate pooled
uplifts are `3.6896` for drivers and
`7.2329` for constructors. In leave-one-season-out
validation, the separate additive adjustment is best for drivers and the shared additive constant is
best for constructors. The separate entity-type additive adjustment still beats multiplication for
both entity types, while no adjustment is least accurate.

Extreme events (absolute event-mean z-score at least 2) are retained, not removed:

| season | round | event_name | weekend_format | driver_mean_points_per_asset | constructor_mean_points_per_asset |
| --- | --- | --- | --- | --- | --- |
| 2023 | 9 | Austrian Grand Prix | sprint | 21.0000 | 47.8000 |
| 2023 | 13 | Dutch Grand Prix | normal | 18.9500 | 43.9000 |
| 2024 | 8 | Monaco Grand Prix | normal | 4.5500 | 14.9000 |
| 2026 | 6 | Monaco Grand Prix | normal | 6.1000 | 18.3636 |
| 2026 | 9 | British Grand Prix | sprint | 17.4545 | 47.5000 |

## 5. Tier findings

Tier estimates vary, but every tier remains observational and is not stable enough to replace the global result. Unranked assets with no prior normal-weekend form are excluded.

| entity_type | tier | normal_mean | sprint_mean | additive_uplift | multiplier | normal_observation_count | sprint_observation_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constructor | A | 54.7126 | 63.3030 | 8.5904 | 1.1570 | 174 | 66 |
| constructor | B | 23.4615 | 29.7444 | 6.2829 | 1.2678 | 234 | 90 |
| constructor | C | 10.8506 | 18.2121 | 7.3615 | 1.6784 | 174 | 66 |
| driver | A | 22.9448 | 27.1909 | 4.2461 | 1.1851 | 290 | 110 |
| driver | B | 8.0879 | 12.1545 | 4.0666 | 1.5028 | 580 | 220 |
| driver | C | 4.0651 | 6.8983 | 2.8332 | 1.6970 | 292 | 118 |

## 6. Proposed first approximation

For an asset with total recorded points `T`, `N_normal` normal races, `N_sprint` Sprint races,
and the entity-type multiplier `x`:

```text
baseline = T / (N_normal + x × N_sprint)
normal EV = baseline
Sprint EV = x × baseline
```

The numerical identity check across every asset had maximum reconstruction error
`4.547e-13`. This identity does not establish predictive accuracy.

| entity | entity_type | normal_races | sprint_races | total_points | global_multiplier | implied_normal_baseline | implied_sprint_ev | reconstructed_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Max Verstappen | driver | 59 | 22 | 2815.0000 | 1.3425 | 31.7953 | 42.6853 | 2815.0000 |
| Fernando Alonso | driver | 59 | 22 | 639.0000 | 1.3425 | 7.2175 | 9.6895 | 639.0000 |
| Isack Hadjar | driver | 25 | 10 | 249.0000 | 1.3425 | 6.4802 | 8.6996 | 249.0000 |
| Logan Sargeant | driver | 28 | 9 | -9.0000 | 1.3425 | -0.2245 | -0.3014 | -9.0000 |
| Red Bull | constructor | 59 | 22 | 4997.0000 | 1.2505 | 57.7621 | 72.2289 | 4997.0000 |
| Haas | constructor | 59 | 22 | 1269.0000 | 1.2505 | 14.6688 | 18.3427 | 1269.0000 |
| Cadillac | constructor | 7 | 4 | -58.0000 | 1.2505 | -4.8326 | -6.0429 | -58.0000 |

## 7. Limitations

- Sprint samples are small relative to normal weekends, so event-level uncertainty remains material.
- Crashes, DNFs, penalties, zeroes and negative scores are intentionally retained as realised EV.
- Scoring environments and roster sizes change across seasons.
- Team and driver strength changes over time; pooled means are not causal Sprint effects.
- Replacement drivers often lack enough prior normal-weekend form for tier assignment.
- One common multiplier assumes proportional uplift across all strengths and event conditions.
- The tier analysis is exploratory and susceptible to small samples and regression to the mean.
- This is a first approximation, not a final individual-asset forecast model.

## 8. Recommendation

**B. Use separate additive uplifts instead because they are more stable.**

This choice is based on the reported pooled effects, event-bootstrap uncertainty, leave-one-out
ranges and cross-season alternative errors. No multiplier is activated in production by this analysis.
