# 2026 Sprint partial-pooling research model

## 1. Executive conclusion

The full proposed model was fitted separately for drivers and constructors with centred ridge asset and event effects. It remains research-only.

| entity_type | strength_definition | mu | lambda | asset_penalty | event_penalty | residual_variance |
| --- | --- | --- | --- | --- | --- | --- |
| driver | z_form | 5.2041 | 2.3204 | 4.0 | 16.0 | 21.1748 |
| constructor | blend_form_0.75_price_0.25 | 1.091 | 16.0499 | 16.0 | 1.0 | 81.6511 |

Leave-one-Sprint-out preferred candidates: driver: `shrunk_personal_mean` MAE 4.085, RMSE 5.624, bias -0.008, Spearman 0.423; constructor: `strength_only` MAE 7.003, RMSE 10.403, bias 0.002, Spearman 0.374. The validation table explicitly shows whether full partial pooling improves on constant, strength-only, and shrunk personal mean.

## 2. Current form definition

Normal weekends use recorded total Fantasy points. Sprint weekends use `total - sprint_points - sprint_qualifying_points`. The equal-weight mean across valid completed rounds 1–11 is `normal_ev`; round 12 is excluded. Missing Sprint components remain missing.

## 3. Strength comparison

Selected driver strength: `z_form`. Selected constructor strength: `blend_form_0.75_price_0.25`.

| entity_type | form_price_correlation | form_price_percentile_correlation | form_price_vif |
| --- | --- | --- | --- |
| constructor | 0.9111 | 0.8818 | 4.4965 |
| driver | 0.911 | 0.9534 | 10.9742 |

| entity_type | strength_definition | mae | rmse | bias | spearman |
| --- | --- | --- | --- | --- | --- |
| driver | z_form | 4.3563 | 5.8324 | 0.0096 | 0.2969 |
| driver | form_percentile | 4.404 | 5.8574 | 0.0129 | 0.284 |
| driver | blend_form_1.00_price_0.00 | 4.404 | 5.8574 | 0.0129 | 0.284 |
| driver | blend_form_0.75_price_0.25 | 4.3692 | 5.8141 | 0.0128 | 0.2938 |
| driver | blend_form_0.50_price_0.50 | 4.3376 | 5.7746 | 0.0126 | 0.3198 |
| driver | blend_form_0.25_price_0.75 | 4.3052 | 5.7404 | 0.0123 | 0.3391 |
| driver | blend_form_0.00_price_1.00 | 4.2852 | 5.7127 | 0.0119 | 0.3528 |
| constructor | z_form | 7.1608 | 10.3483 | 0.002 | 0.3809 |
| constructor | form_percentile | 7.1906 | 10.548 | -0.0033 | 0.3451 |
| constructor | blend_form_1.00_price_0.00 | 7.1906 | 10.548 | -0.0033 | 0.3451 |
| constructor | blend_form_0.75_price_0.25 | 7.0034 | 10.4031 | 0.0021 | 0.3741 |
| constructor | blend_form_0.50_price_0.50 | 6.9157 | 10.2791 | 0.0076 | 0.4095 |
| constructor | blend_form_0.25_price_0.75 | 6.9399 | 10.1884 | 0.0128 | 0.4268 |
| constructor | blend_form_0.00_price_1.00 | 7.0127 | 10.1375 | 0.0174 | 0.4374 |

Price is considered useful only when a blend materially lowers whole-event validation error; high VIF values indicate that form and price cannot be interpreted independently.

## 4. Driver model

| entity | strength | group_bonus | asset_effect_u | personalised_bonus |
| --- | --- | --- | --- | --- |
| Alexander Albon | -1.0673 | 2.7275 | 2.5113 | 5.2387 |
| Lance Stroll | -1.681 | 1.3035 | 1.9732 | 3.2768 |
| Liam Lawson | -0.1145 | 4.9383 | 1.9058 | 6.8442 |
| Lando Norris | 0.6364 | 6.6808 | 1.7846 | 8.4654 |
| George Russell | 1.1863 | 7.9567 | 1.6466 | 9.6034 |
| Charles Leclerc | 1.284 | 8.1834 | 1.2833 | 9.4667 |
| Carlos Sainz | -0.4016 | 4.2722 | 1.1701 | 5.4423 |
| Oscar Piastri | 0.1842 | 5.6316 | 0.9342 | 6.5658 |
| Esteban Ocon | -0.1145 | 4.9383 | 0.9058 | 5.8442 |
| Sergio Perez | -0.9462 | 3.0085 | 0.3707 | 3.3793 |
| Franco Colapinto | 0.0316 | 5.2775 | 0.1113 | 5.3887 |
| Oliver Bearman | -0.4375 | 4.1889 | -0.0944 | 4.0944 |
| Pierre Gasly | 0.0405 | 5.2981 | -0.2104 | 5.0877 |
| Kimi Antonelli | 2.2755 | 10.4842 | -0.2421 | 10.2421 |
| Fernando Alonso | -0.9058 | 3.1022 | -0.3011 | 2.8011 |
| Max Verstappen | 1.1128 | 7.7862 | -0.6431 | 7.1431 |
| Valtteri Bottas | -1.358 | 2.053 | -0.9015 | 1.1515 |
| Lewis Hamilton | 1.6861 | 9.1165 | -1.0582 | 8.0582 |
| Gabriel Bortoleto | -0.3487 | 4.395 | -2.0725 | 2.3225 |
| Nico Hulkenberg | -1.035 | 2.8024 | -2.7762 | 0.0262 |
| Isack Hadjar | 0.2569 | 5.8002 | -3.0251 | 2.7751 |
| Arvid Lindblad | -0.2841 | 4.5449 | -3.2724 | 1.2724 |

## 5. Constructor model

| entity | strength | group_bonus | asset_effect_u | personalised_bonus |
| --- | --- | --- | --- | --- |
| Williams | 0.3409 | 6.5626 | 1.1476 | 7.7101 |
| Aston Martin | 0.1136 | 2.9148 | 0.967 | 3.8819 |
| McLaren | 0.7727 | 13.4932 | 0.8514 | 14.3446 |
| Mercedes | 1.0 | 17.1409 | 0.8218 | 17.9627 |
| Ferrari | 0.8636 | 14.9523 | 0.5595 | 15.5118 |
| Haas F1 Team | 0.4318 | 8.0216 | 0.5457 | 8.5673 |
| Cadillac | 0.1591 | 3.6444 | 0.0711 | 3.7155 |
| Alpine | 0.5682 | 10.2103 | -0.2921 | 9.9182 |
| Racing Bulls | 0.5909 | 10.575 | -0.765 | 9.81 |
| Red Bull Racing | 0.8182 | 14.2228 | -1.5946 | 12.6282 |
| Audi | 0.3409 | 6.5626 | -2.3125 | 4.25 |

## 6. Sprint event effects

| season | round | event_name | driver_event_effect | constructor_event_effect | driver_effects_centred | constructor_effects_centred |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 | 2 | Chinese Grand Prix | 1.4216 | 4.475 | True | True |
| 2026 | 4 | Miami Grand Prix | -0.9866 | -3.4417 | True | True |
| 2026 | 5 | Canadian Grand Prix | -0.8391 | -2.525 | True | True |
| 2026 | 9 | British Grand Prix | 0.4041 | 1.4918 | True | True |

The effects are centred to zero. Prediction for the next unknown Sprint always sets the event effect to zero.

## 7. Validation

| entity_type | model | mae | rmse | bias | spearman | mae_low_form_third | mae_middle_form_third | mae_high_form_third |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| driver | constant | 4.5782 | 6.2307 | 0.0092 | -0.2619 | 4.2547 | 4.4356 | 4.9729 |
| driver | strength_only | 4.3563 | 5.8324 | 0.0096 | 0.2969 | 4.2315 | 4.2753 | 4.5288 |
| driver | strength_event | 4.3556 | 5.8334 | 0.0243 | 0.2962 | 4.2283 | 4.2751 | 4.5299 |
| driver | personal_mean | 4.7608 | 5.9915 | -0.0 | 0.4128 | 4.4762 | 5.04 | 4.7917 |
| driver | shrunk_personal_mean | 4.0851 | 5.6243 | -0.0081 | 0.4225 | 3.9139 | 4.0826 | 4.2369 |
| driver | partial_asset | 4.1009 | 5.6122 | -0.0014 | 0.4388 | 3.8981 | 4.1361 | 4.2509 |
| driver | full_partial_pooling | 4.0977 | 5.6108 | 0.011 | 0.437 | 3.8969 | 4.1248 | 4.2523 |
| constructor | constant | 7.7759 | 11.304 | 0.0174 | -0.2526 | 7.217 | 6.7707 | 9.1652 |
| constructor | strength_only | 7.0034 | 10.4031 | 0.0021 | 0.3741 | 6.8404 | 6.0442 | 8.0748 |
| constructor | strength_event | 6.9978 | 10.3947 | 0.0262 | 0.3766 | 6.8282 | 6.0403 | 8.0717 |
| constructor | personal_mean | 8.5349 | 11.0102 | -0.0 | 0.3511 | 8.8788 | 8.2083 | 8.625 |
| constructor | shrunk_personal_mean | 7.1924 | 10.3997 | -0.007 | 0.3868 | 7.3237 | 6.1064 | 8.1882 |
| constructor | partial_asset | 6.9545 | 10.2751 | -0.0049 | 0.4053 | 6.9029 | 5.9695 | 7.9751 |
| constructor | full_partial_pooling | 6.9458 | 10.2584 | 0.0284 | 0.4053 | 6.8749 | 5.9635 | 7.9768 |

## 8. Case studies

- **Nico Hulkenberg:** observations [-3.0, -10.0, -1.0, 3.0]; group 2.80; raw mean -2.75; shrunk mean 0.49; u=-2.78; final bonus 0.03.
- **Valtteri Bottas:** observations [-8.0, 1.0, 6.0, 2.0]; group 2.05; raw mean 0.25; shrunk mean 1.30; u=-0.90; final bonus 1.15.
- **Lance Stroll:** observations [5.0, 10.0, 3.0, 3.0]; group 1.30; raw mean 5.25; shrunk mean 2.95; u=1.97; final bonus 3.28.
- **Liam Lawson:** observations [14.0, 4.0, 11.0, 6.0]; group 4.94; raw mean 8.75; shrunk mean 6.53; u=1.91; final bonus 6.84.
- **Oliver Bearman:** observations [5.0, 3.0, 2.0, 6.0]; group 4.19; raw mean 4.00; shrunk mean 4.11; u=-0.09; final bonus 4.09.
- **Kimi Antonelli:** observations [11.0, 1.0, 12.0, 16.0]; group 10.48; raw mean 10.00; shrunk mean 10.28; u=-0.24; final bonus 10.24.
- **George Russell:** observations [12.0, 10.0, 9.0, 14.0]; group 7.96; raw mean 11.25; shrunk mean 9.33; u=1.65; final bonus 9.60.
- **Lando Norris:** observations [5.0, 13.0, 9.0, 14.0]; group 6.68; raw mean 10.25; shrunk mean 8.17; u=1.78; final bonus 8.47.
- **Mercedes:** observations [23.0, 11.0, 21.0, 30.0]; group 17.14; raw mean 21.25; shrunk mean 18.40; u=0.82; final bonus 17.96.
- **Ferrari:** observations [36.0, 12.0, 9.0, 14.0]; group 14.95; raw mean 17.75; shrunk mean 15.81; u=0.56; final bonus 15.51.
- **McLaren:** observations [12.0, 22.0, 15.0, 22.0]; group 13.49; raw mean 17.75; shrunk mean 14.80; u=0.85; final bonus 14.34.
- **Red Bull Racing:** observations [18.0, 9.0, -8.0, 6.0]; group 14.22; raw mean 6.25; shrunk mean 11.78; u=-1.59; final bonus 12.63.
- **Williams:** observations [25.0, 9.0, 6.0, missing]; group 6.56; raw mean 13.33; shrunk mean 8.25; u=1.15; final bonus 7.71.
- **Audi:** observations [3.0, -30.0, 1.0, 6.0]; group 6.56; raw mean -5.00; shrunk mean 3.02; u=-2.31; final bonus 4.25.
- **Aston Martin:** observations [11.0, 20.0, -6.0, 6.0]; group 2.91; raw mean 7.75; shrunk mean 4.40; u=0.97; final bonus 3.88.
- **Cadillac:** observations [1.0, 5.0, 11.0, -1.0]; group 3.64; raw mean 4.00; shrunk mean 3.75; u=0.07; final bonus 3.72.

## 9. Historical shape check

The season-normalised 2023–2025 qualitative result is `supports_positive_strength_relationship`. Older raw values never enter the 2026 fit.

## 10. Recommended shadow candidate

```text
normal_ev = mean completed 2026 normal-equivalent scores
strength = selected within-class strength definition
group_bonus = mu + lambda * strength
personalised_bonus = group_bonus + shrunk asset effect u_i
Sprint EV = normal_ev + personalised_bonus
future unknown event effect = 0
```

A shadow implementation is justified only if the selected partial-pooling candidate is competitive with the simpler shrunk personal mean under leave-one-Sprint-out validation. Nothing is activated by this analysis.

## 11. Limitations

Only four 2026 Sprints exist, asset outcomes are correlated within events, and constructors provide only 11 cross-sectional units. Reliability effects remain difficult to distinguish from randomness. Form and price are collinear. This is current-state descriptive calibration and must be recalibrated after future Sprint events.
