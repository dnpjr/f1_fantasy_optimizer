# Personalised 2026 Sprint adjustments

## 1. Executive summary

This offline study fits a separate Sprint relationship for every current 2026 asset. All coefficients are research-only; no production forecast, optimiser, cache, or UI reads these files.

The pooled reference is `bonus = 4.2291 + 0.0926 × base` for drivers and `bonus = 5.1072 + 0.1651 × base` for constructors.

## 2. Method

```text
extra Sprint points = Sprint points + Sprint Qualifying points
base weekend points = total points - extra Sprint points
personalised bonus = alpha_asset + beta_asset × current normal baseline
candidate Sprint EV = current normal baseline + personalised bonus
```

The primary baseline averages ordinary-weekend content across every valid completed round 1–11: normal weekends use total points, while Sprint weekends use total minus their official Sprint-specific components. Round 12 is explicitly excluded.

## 3. Drivers

| entity_name | current_normal_baseline | mean_observed_sprint_bonus | raw_alpha | raw_beta | raw_regression_adjustment | raw_candidate_sprint_ev | reliability_flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi Antonelli | 35.3636 | 10.0 | 14.4729 | -0.1154 | 10.3909 | 45.7545 | usable_but_uncertain |
| Lewis Hamilton | 28.7273 | 7.0 | 3.7143 | 0.1143 | 6.9974 | 35.7247 | usable_but_uncertain |
| Charles Leclerc | 24.2 | 10.75 | 8.8976 | 0.0639 | 10.4434 | 34.6434 | high_residual_error |
| George Russell | 23.1 | 11.25 | 9.5064 | 0.0775 | 11.2965 | 34.3965 | usable_but_uncertain |
| Max Verstappen | 22.2727 | 6.5 | 7.5746 | -0.0682 | 6.0549 | 28.3277 | usable_but_uncertain |
| Lando Norris | 16.9091 | 10.25 | 8.5212 | 0.1471 | 11.0091 | 27.9182 | usable_but_uncertain |
| Oscar Piastri | 11.8182 | 7.5 | 7.0792 | 0.0481 | 7.6476 | 19.4658 | usable_but_uncertain |
| Liam Lawson | 8.4545 | 8.75 | 7.057 | 0.2052 | 8.792 | 17.2465 | usable_but_uncertain |
| Franco Colapinto | 10.1 | 5.5 | 7.5019 | -0.1313 | 6.1761 | 16.2761 | usable_but_uncertain |
| Esteban Ocon | 8.4545 | 6.75 | 4.8614 | 0.1988 | 6.5422 | 14.9967 | usable_but_uncertain |
| Pierre Gasly | 10.2 | 4.3333 | 4.485 | -0.0253 | 4.2272 | 14.4272 | usable_but_uncertain |
| Isack Hadjar | 12.6364 | -0.25 | 0.5806 | -0.1007 | -0.6916 | 11.9448 | usable_but_uncertain |
| Oliver Bearman | 4.8182 | 4.0 | 3.3135 | 0.0584 | 3.595 | 8.4132 | usable_but_uncertain |
| Valtteri Bottas | -5.5455 | 0.25 | 6.807 | -1.1404 | 13.1308 | 7.5853 | extreme_slope |
| Gabriel Bortoleto | 5.8182 | 0.25 | 1.2283 | -0.301 | -0.523 | 5.2952 | usable_but_uncertain |
| Alexander Albon | -2.2727 | 7.75 | 6.512 | -0.127 | 6.8006 | 4.5279 | usable_but_uncertain |
| Arvid Lindblad | 6.5455 | -2.0 | -1.703 | -0.0914 | -2.3011 | 4.2443 | usable_but_uncertain |
| Fernando Alonso | -0.4545 | 2.5 | 4.1196 | 0.4319 | 3.9233 | 3.4687 | usable_but_uncertain |
| Sergio Perez | -0.9091 | 3.75 | 3.9659 | -0.0432 | 4.0052 | 3.0961 | usable_but_uncertain |
| Lance Stroll | -9.1818 | 5.25 | 5.25 | 0.0224 | 5.0439 | -4.1379 | usable_but_uncertain |
| Nico Hulkenberg | -1.9091 | -2.75 | -2.0492 | 0.1649 | -2.364 | -4.2731 | usable_but_uncertain |
| Carlos Sainz | 5.2222 | 8.0 |  |  |  |  | insufficient_observations |

## 4. Constructors

| entity_name | current_normal_baseline | mean_observed_sprint_bonus | raw_alpha | raw_beta | raw_regression_adjustment | raw_candidate_sprint_ev | reliability_flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mercedes | 72.4545 | 21.25 | 38.0953 | -0.2231 | 21.9295 | 94.384 | usable_but_uncertain |
| Ferrari | 63.1818 | 17.75 | -14.3295 | 0.4824 | 16.1493 | 79.3311 | usable_but_uncertain |
| McLaren | 42.4545 | 17.75 | 14.201 | 0.1036 | 18.6002 | 61.0547 | usable_but_uncertain |
| Red Bull Racing | 43.3636 | 6.25 | 27.1855 | -0.6594 | -1.4079 | 41.9558 | usable_but_uncertain |
| Racing Bulls | 28.8182 | 6.75 | 1.722 | 0.1972 | 7.4043 | 36.2225 | usable_but_uncertain |
| Alpine | 26.4545 | 8.75 | -1.5664 | 0.3275 | 7.0976 | 33.5521 | usable_but_uncertain |
| Haas F1 Team | 15.7273 | 10.75 | 7.7331 | 0.1244 | 9.6897 | 25.417 | usable_but_uncertain |
| Williams | 9.2 | 13.3333 | 15.5639 | -0.2231 | 13.5118 | 22.7118 | usable_but_uncertain |
| Audi | 12.6 | -5.0 | -6.0278 | 0.2418 | -2.9807 | 9.6193 | high_residual_error |
| Cadillac | -6.7273 | 4.0 | 7.4766 | -0.309 | 9.5556 | 2.8283 | usable_but_uncertain |
| Aston Martin | -8.0909 | 7.75 | 8.1614 | 0.1496 | 6.951 | -1.1399 | usable_but_uncertain |

## 5. Asset case studies

#### Nico Hulkenberg

Baseline -1.91; alpha -2.0492; beta 0.1649; adjustment -2.3640; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | 10.0 | -3.0 | True |
| 4 | Miami Grand Prix | -19.0 | -10.0 | True |
| 5 | Canadian Grand Prix | 2.0 | -1.0 | True |
| 9 | British Grand Prix | -10.0 | 3.0 | True |

#### Valtteri Bottas

Baseline -5.55; alpha 6.8070; beta -1.1404; adjustment 13.1308; reliability `extreme_slope`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | 11.0 | -8.0 | True |
| 4 | Miami Grand Prix | 3.0 | 1.0 | True |
| 5 | Canadian Grand Prix | 6.0 | 6.0 | True |
| 9 | British Grand Prix | 3.0 | 2.0 | True |

#### Lance Stroll

Baseline -9.18; alpha 5.2500; beta 0.0224; adjustment 5.0439; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | -19.0 | 5.0 | True |
| 4 | Miami Grand Prix | 7.0 | 10.0 | True |
| 5 | Canadian Grand Prix | 8.0 | 3.0 | True |
| 9 | British Grand Prix | 4.0 | 3.0 | True |

#### Liam Lawson

Baseline 8.45; alpha 7.0570; beta 0.2052; adjustment 8.7920; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | 21.0 | 14.0 | True |
| 4 | Miami Grand Prix | -16.0 | 4.0 | True |
| 5 | Canadian Grand Prix | 12.0 | 11.0 | True |
| 9 | British Grand Prix | 16.0 | 6.0 | True |

#### Oliver Bearman

Baseline 4.82; alpha 3.3135; beta 0.0584; adjustment 3.5950; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | 29.0 | 5.0 | True |
| 4 | Miami Grand Prix | 3.0 | 3.0 | True |
| 5 | Canadian Grand Prix | 8.0 | 2.0 | True |
| 9 | British Grand Prix | 7.0 | 6.0 | True |

#### Kimi Antonelli

Baseline 35.36; alpha 14.4729; beta -0.1154; adjustment 10.3909; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | 57.0 | 11.0 | True |
| 4 | Miami Grand Prix | 41.0 | 1.0 | True |
| 5 | Canadian Grand Prix | 50.0 | 12.0 | True |
| 9 | British Grand Prix | 7.0 | 16.0 | True |

#### Mercedes

Baseline 72.45; alpha 38.0953; beta -0.2231; adjustment 21.9295; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | 92.0 | 23.0 | True |
| 4 | Miami Grand Prix | 93.0 | 11.0 | True |
| 5 | Canadian Grand Prix | 55.0 | 21.0 | True |
| 9 | British Grand Prix | 62.0 | 30.0 | True |

#### McLaren

Baseline 42.45; alpha 14.2010; beta 0.1036; adjustment 18.6002; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | -19.0 | 12.0 | True |
| 4 | Miami Grand Prix | 88.0 | 22.0 | True |
| 5 | Canadian Grand Prix | 18.0 | 15.0 | True |
| 9 | British Grand Prix | 50.0 | 22.0 | True |

#### Aston Martin

Baseline -8.09; alpha 8.1614; beta 0.1496; adjustment 6.9510; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | -31.0 | 11.0 | True |
| 4 | Miami Grand Prix | 18.0 | 20.0 | True |
| 5 | Canadian Grand Prix | -8.0 | -6.0 | True |
| 9 | British Grand Prix | 10.0 | 6.0 | True |

#### Cadillac

Baseline -6.73; alpha 7.4766; beta -0.3090; adjustment 9.5556; reliability `usable_but_uncertain`. The fitted result reflects the small observation set below, including negative outcomes and component gaps.

| round | event_name | base_weekend_points | extra_sprint_points | included_in_regression |
| --- | --- | --- | --- | --- |
| 2 | Chinese Grand Prix | 21.0 | 1.0 | True |
| 4 | Miami Grand Prix | 19.0 | 5.0 | True |
| 5 | Canadian Grand Prix | -9.0 | 11.0 | True |
| 9 | British Grand Prix | 14.0 | -1.0 | True |

## 6. Group versus personalised estimates

`asset_predictions.csv` includes raw, group, mean, median, recency, and 25/50/75% shrunk alternatives. Raw personal fits can be dominated by four points; group shrinkage deliberately exposes a continuum rather than selecting a production weight.

## 7. Baseline sensitivity

| baseline_method | assets | mean_baseline | mean_adjustment | min_adjustment | max_adjustment |
| --- | --- | --- | --- | --- | --- |
| current_normal_baseline | 33 | 15.5698 | 7.0843 | -2.9807 | 21.9295 |
| median_normalised_baseline | 33 | 17.4697 | 7.1271 | -2.8839 | 22.254 |
| normal_weekend_only_mean | 33 | 15.1659 | 7.2703 | -5.7838 | 22.3178 |
| recency_weighted_normal_baseline | 33 | 16.091 | 7.2817 | -6.616 | 23.5126 |
| recency_weighted_normal_weekend_only_mean | 33 | 15.597 | 7.3231 | -9.9442 | 23.2477 |

## 8. Reliability

Every ordinary two-parameter asset fit needs at least three valid Sprint observations and non-trivial base variance. Extreme slopes and high residual error are retained and flagged rather than hidden.

| entity_type | entity_name | valid_sprint_observations | raw_beta | condition_number | reliability_flag |
| --- | --- | --- | --- | --- | --- |
| constructor | Audi | 4 | 0.2418 | 9.7382 | high_residual_error |
| driver | Charles Leclerc | 4 | 0.0639 | 89.6765 | high_residual_error |
| driver | Valtteri Bottas | 4 | -1.1404 | 13.615 | extreme_slope |
| driver | Carlos Sainz | 2 |  |  | insufficient_observations |

## 9. Recommendation for later implementation

A lightly shrunk personal regression (for example the explicit 25% asset / 75% group candidate) is the most defensible future shadow model among these outputs. It retains directionally personal evidence without treating four Sprints as a stable asset law. Mean personal bonus is a useful simpler benchmark. Raw personal coefficients should not be activated without more Sprint events and a walk-forward evaluation.

## 10. Limitations

There are only four completed Sprints, and outcomes within each event are correlated. Crashes, DNFs, penalties, and positions lost are genuine but can dominate individual slopes. Replacement drivers can have missing component observations. This current-season-only analysis uses all completed rounds descriptively, not a historical walk-forward design. Coefficients can change materially after every future Sprint.
