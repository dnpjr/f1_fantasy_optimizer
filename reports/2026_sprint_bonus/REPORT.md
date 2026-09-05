# 2026 Sprint-only Fantasy bonus analysis

## 1. Executive conclusion

The extra target is `sprint_points + sprint_qualifying_points`, summed only when at least one official component is present. Ordinary qualifying, Grand Prix race points, residuals, complete-weekend totals, and price changes are excluded.

Across current assets, the observed mean Sprint-only bonus is **5.24 points for drivers** and **9.94 points for constructors**. The selected pooled models are `constrained_form_price` for drivers and `constrained_form_price` for constructors.

Driver recommendation: `predicted_bonus = max(0, 0.6856 + 8.7174 × price_percentile)`.
Constructor recommendation: `predicted_bonus = max(0, 0.3394 + 17.6000 × price_percentile)`.

Adding price percentile improved constrained-model LOAO MAE by 0.315 driver points and 0.567 constructor points. In both selected fits the form coefficient collapsed to zero while price percentile remained positive: stronger/more expensive constructors therefore receive larger absolute bonuses descriptively, but the four-Sprint sample cannot separate form from price reliably.

## 2. Component semantics

The official playerstats parser classifies sessions using both session labels and scoring-event labels. The redacted official fixture demonstrates that a feed session labelled `Sprint Qualifying` can contain `Sprint Position` and `Sprint overtake` events; these are correctly classified as `sprint_points`. A separately emitted Sprint Qualifying score is retained as `sprint_qualifying_points`. Canonical nulls remain null, and target aggregation uses `min_count=1`, so an absent observation is never invented as zero.

## 3. Current 2026 asset summaries

The descriptive features use all 7 completed normal 2026 weekends; targets use 4 completed Sprint weekends. This is current-state calibration, so races after an early Sprint may contribute to current form. Current prices are from verified official feed 12 (2026-08-06T09:49:00.543442+00:00).

| entity_type | entity | current_price | normal_weekend_mean | recent_normal_form | mean_extra_sprint_points | sprint_event_count |
| --- | --- | --- | --- | --- | --- | --- |
| constructor | Mercedes | 32.6 | 70.714 | 66.546 | 21.25 | 4 |
| constructor | Ferrari | 26.6 | 61.286 | 62.731 | 17.75 | 4 |
| constructor | Red Bull Racing | 30.9 | 50.0 | 56.31 | 6.25 | 4 |
| constructor | McLaren | 31.0 | 47.143 | 49.292 | 17.75 | 4 |
| constructor | Racing Bulls | 12.9 | 30.714 | 32.001 | 6.75 | 4 |
| constructor | Alpine | 18.8 | 23.571 | 22.21 | 8.75 | 4 |
| constructor | Audi | 6.8 | 18.167 | 19.235 | -5.0 | 4 |
| constructor | Haas F1 Team | 12.0 | 10.857 | 10.841 | 10.75 | 4 |
| constructor | Williams | 13.0 | 8.857 | 8.541 | 13.333 | 3 |
| constructor | Aston Martin | 5.7 | -11.143 | -3.898 | 7.75 | 4 |
| constructor | Cadillac | 3.0 | -17.0 | -21.436 | 4.0 | 4 |
| driver | Kimi Antonelli | 25.7 | 33.429 | 31.901 | 10.0 | 4 |
| driver | Lewis Hamilton | 25.0 | 28.714 | 28.443 | 7.0 | 4 |
| driver | Max Verstappen | 27.6 | 26.0 | 28.204 | 6.5 | 4 |
| driver | George Russell | 27.9 | 23.5 | 20.268 | 11.25 | 4 |
| driver | Charles Leclerc | 23.9 | 21.0 | 22.634 | 10.75 | 4 |
| driver | Lando Norris | 26.1 | 19.857 | 23.845 | 10.25 | 4 |
| driver | Isack Hadjar | 14.5 | 15.143 | 18.623 | -0.25 | 4 |
| driver | Oscar Piastri | 24.4 | 13.571 | 11.546 | 7.5 | 4 |
| driver | Pierre Gasly | 13.0 | 12.0 | 10.309 | 4.333 | 3 |
| driver | Liam Lawson | 10.3 | 8.571 | 8.706 | 8.75 | 4 |
| driver | Arvid Lindblad | 8.0 | 8.429 | 7.236 | -2.0 | 4 |
| driver | Esteban Ocon | 10.7 | 7.857 | 6.985 | 6.75 | 4 |
| driver | Gabriel Bortoleto | 7.8 | 7.286 | 7.497 | 0.25 | 4 |
| driver | Franco Colapinto | 11.2 | 6.667 | 6.509 | 5.5 | 4 |
| driver | Carlos Sainz | 10.0 | 2.286 | 1.82 | 8.0 | 2 |
| driver | Alexander Albon | 5.8 | 2.0 | 2.061 | 7.75 | 4 |
| driver | Fernando Alonso | 6.2 | 1.429 | 3.432 | 2.5 | 4 |
| driver | Oliver Bearman | 7.6 | 0.857 | 1.644 | 4.0 | 4 |
| driver | Nico Hulkenberg | 3.0 | -0.571 | 1.767 | -2.75 | 4 |
| driver | Sergio Perez | 3.8 | -4.286 | -8.794 | 3.75 | 4 |
| driver | Valtteri Bottas | 3.0 | -12.0 | -11.901 | 0.25 | 4 |
| driver | Lance Stroll | 3.0 | -14.429 | -10.404 | 5.25 | 4 |

## 4. Model comparison

| entity_type | model | loao_mae | loao_rmse | loao_bias | loao_spearman | loao_r_squared | loao_negative_predictions | selected_pooled_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constructor | constrained_form_price | 4.9792 | 5.9882 | 0.0909 | 0.5057 | 0.2836 | 0 | True |
| constructor | price_only | 5.1122 | 6.0132 | -0.1334 | 0.5057 | 0.2776 | 0 | False |
| constructor | ridge_form_price | 5.2205 | 6.3878 | -0.1561 | 0.4055 | 0.1848 | 0 | False |
| constructor | form_price | 5.3958 | 6.7583 | -0.235 | 0.3508 | 0.0875 | 0 | False |
| constructor | constrained_hybrid | 5.5457 | 6.7281 | -0.3191 | 0.2916 | 0.0956 | 0 | False |
| constructor | hybrid | 5.5457 | 6.7281 | -0.3191 | 0.2916 | 0.0956 | 0 | False |
| constructor | constrained_proportional | 5.9706 | 6.7292 | -2.4649 | 0.4977 | 0.0953 | 0 | False |
| constructor | constant | 6.2273 | 7.7822 | 0.0 | -1.0 | -0.21 | 0 | False |
| constructor | proportional | 6.6583 | 7.4521 | -3.1526 | 0.5103 | -0.1095 | 2 | False |
| driver | constrained_form_price | 2.8234 | 3.461 | -0.009 | 0.5428 | 0.2698 | 0 | True |
| driver | price_only | 2.8272 | 3.4663 | -0.0128 | 0.5428 | 0.2676 | 0 | False |
| driver | ridge_form_price | 2.8793 | 3.5172 | -0.0483 | 0.5066 | 0.2459 | 0 | False |
| driver | form_price | 2.9176 | 3.5658 | -0.0721 | 0.4609 | 0.2249 | 0 | False |
| driver | interaction | 2.9605 | 3.6631 | -0.0279 | 0.4485 | 0.182 | 0 | False |
| driver | constrained_hybrid | 3.1387 | 3.7708 | -0.0414 | 0.4197 | 0.1332 | 0 | False |
| driver | hybrid | 3.1387 | 3.7708 | -0.0414 | 0.4197 | 0.1332 | 0 | False |
| driver | constant | 3.5332 | 4.2431 | -0.0 | -1.0 | -0.0975 | 0 | False |
| driver | constrained_proportional | 3.6448 | 4.119 | -1.863 | 0.5514 | -0.0342 | 0 | False |
| driver | proportional | 4.0984 | 4.6577 | -2.3329 | 0.536 | -0.3225 | 4 | False |
| constructor | shrunk_asset_residual |  |  |  |  |  | 0 | False |
| driver | shrunk_asset_residual |  |  |  |  |  | 0 | False |

## 5. Drivers

| entity | normal_weekend_ev | observed_mean_sprint_bonus | pooled_predicted_bonus | shrunk_asset_bonus | predicted_sprint_ev |
| --- | --- | --- | --- | --- | --- |
| George Russell | 23.5 | 11.25 | 9.403 | 10.327 | 33.827 |
| Lando Norris | 19.857 | 10.25 | 8.611 | 9.43 | 29.287 |
| Kimi Antonelli | 33.429 | 10.0 | 8.214 | 9.107 | 42.536 |
| Charles Leclerc | 21.0 | 10.75 | 7.026 | 8.888 | 29.888 |
| Max Verstappen | 26.0 | 6.5 | 9.007 | 7.753 | 33.753 |
| Oscar Piastri | 13.571 | 7.5 | 7.422 | 7.461 | 21.032 |
| Lewis Hamilton | 28.714 | 7.0 | 7.818 | 7.409 | 36.123 |
| Liam Lawson | 8.571 | 8.75 | 5.044 | 6.897 | 15.469 |
| Esteban Ocon | 7.857 | 6.75 | 5.441 | 6.095 | 13.952 |
| Carlos Sainz | 2.286 | 8.0 | 4.648 | 5.765 | 8.051 |
| Franco Colapinto | 6.667 | 5.5 | 5.837 | 5.668 | 12.335 |
| Pierre Gasly | 12.0 | 4.333 | 6.233 | 5.419 | 17.419 |
| Alexander Albon | 2.0 | 7.75 | 2.667 | 5.208 | 7.208 |
| Oliver Bearman | 0.857 | 4.0 | 3.459 | 3.73 | 4.587 |
| Lance Stroll | -14.429 | 5.25 | 1.478 | 3.364 | -11.065 |
| Isack Hadjar | 15.143 | -0.25 | 6.629 | 3.19 | 18.332 |
| Sergio Perez | -4.286 | 3.75 | 2.271 | 3.01 | -1.275 |
| Fernando Alonso | 1.429 | 2.5 | 3.063 | 2.782 | 4.21 |
| Gabriel Bortoleto | 7.286 | 0.25 | 3.856 | 2.053 | 9.338 |
| Arvid Lindblad | 8.429 | -2.0 | 4.252 | 1.126 | 9.554 |
| Valtteri Bottas | -12.0 | 0.25 | 1.478 | 0.864 | -11.136 |
| Nico Hulkenberg | -0.571 | -2.75 | 1.478 | 0.0 | -0.571 |

## 6. Constructors

| entity | normal_weekend_ev | observed_mean_sprint_bonus | pooled_predicted_bonus | shrunk_asset_bonus | predicted_sprint_ev |
| --- | --- | --- | --- | --- | --- |
| Mercedes | 70.714 | 21.25 | 17.939 | 17.939 | 88.654 |
| McLaren | 47.143 | 17.75 | 16.339 | 16.339 | 63.482 |
| Red Bull Racing | 50.0 | 6.25 | 14.739 | 14.739 | 64.739 |
| Ferrari | 61.286 | 17.75 | 13.139 | 13.139 | 74.425 |
| Alpine | 23.571 | 8.75 | 11.539 | 11.539 | 35.111 |
| Williams | 8.857 | 13.333 | 9.939 | 9.939 | 18.797 |
| Racing Bulls | 30.714 | 6.75 | 8.339 | 8.339 | 39.054 |
| Haas F1 Team | 10.857 | 10.75 | 6.739 | 6.739 | 17.597 |
| Audi | 18.167 | -5.0 | 5.139 | 5.139 | 23.306 |
| Aston Martin | -11.143 | 7.75 | 3.539 | 3.539 | -7.603 |
| Cadillac | -17.0 | 4.0 | 1.939 | 1.939 | -15.061 |

The constructor table directly answers the absolute-bonus question: compare `pooled_predicted_bonus` for leading and lower-form teams. The formula does not assume a named team tier.

## 7. Price contribution

Form/price VIF is 6.78 for drivers and 5.93 for constructors. Price is retained only if its LOAO benefit is material; otherwise the simpler form-only or constant candidate is preferred.

## 8. Stability and uncertainty

Leave-one-asset-out predictions are the primary pooled-model comparison. Bootstrap intervals resample assets with a fixed seed. Leave-one-Sprint-out refits after removing the event before target aggregation; coefficient ranges are:

| ('entity_type', '') | ('model', '') | ('alpha', 'min') | ('alpha', 'max') | ('gamma', 'min') | ('gamma', 'max') | ('delta', 'min') | ('delta', 'max') | ('mae', 'min') | ('mae', 'max') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constructor | constrained_form_price | 0.0 | 2.4264 | 0.0 | 0.0471 | 13.9444 | 19.7 | 4.5376 | 9.9856 |
| driver | constrained_form_price | 0.2895 | 1.4529 | 0.0 | 0.0102 | 7.765 | 9.9438 | 2.5237 | 5.8259 |

## 9. Recommended first production candidate

Drivers: `predicted_bonus = max(0, 0.6856 + 8.7174 × price_percentile)`; `Sprint EV = normal EV + predicted_bonus`.

Constructors: `predicted_bonus = max(0, 0.3394 + 17.6000 × price_percentile)`; `Sprint EV = normal EV + predicted_bonus`.

These are research recommendations only and are not activated in production.

## 10. Limitations

Only four completed 2026 Sprints are available. Assets within the same Sprint are correlated. A few official component observations are missing and remain excluded rather than converted to zero. Crashes, DNFs, and penalties are retained and can strongly affect four-event means. Current price partly reflects results already observed. Constructor analysis has only 11 assets. This is descriptive full-season/current-state calibration, not a leakage-safe historical walk-forward test.
