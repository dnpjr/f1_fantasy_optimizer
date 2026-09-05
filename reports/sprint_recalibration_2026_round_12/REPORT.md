# Simplified 2026 Sprint-EV final candidate

## Executive recommendation

**A. Values look sensible enough for a shadow implementation.** This is a research recommendation only; the model is not production-approved or activated.

Normal EV is the equal-weight mean of included completed normal-equivalent scores. Normal weekends retain recorded total; Sprint weekends subtract official Sprint and Sprint-qualifying points. With application decay `p`, this generalises to a recency-weighted mean whose included scores receive weights `1, p, p², ...` newest first.

Driver formulas:

- `z_form = (normal_ev - 9.467457180501) / 10.915416414447` (within the current driver class, population SD).
- `group_bonus = 4.706549290132 + 2.266533074786 * z_form`.
- `personal_mean = arithmetic mean of valid official Sprint-only bonuses`, retaining negative values and omitting missing observations.
- `w_i = tau² / (tau² + sigma_within² / n_i)`, where `tau²=5.273338291601` and `sigma_within²=24.175000000000`.
- `Sprint bonus = w_i * personal_mean + (1 - w_i) * group_bonus`; an asset with no valid observation gets `w_i=0`.
- `Sprint EV = normal_ev + Sprint bonus`.

Constructor formulas:

- `strength = 0.75 * form_percentile + 0.25 * price_percentile`.
- `Sprint bonus = 0.992577958138 + 14.409013242204 * strength`, with the slope constrained nonnegative.
- `Sprint EV = normal_ev + Sprint bonus`.

The simplified candidate has no asset-specific constructor effect and no future Sprint-event effect; `v_next=0`.

## Driver table

| entity | normal_ev | observed_mean_sprint_bonus | group_bonus | empirical_bayes_weight | final_sprint_bonus | sprint_weekend_ev |
| --- | --- | --- | --- | --- | --- | --- |
| Alexander Albon | -2.0 | 6.4 | 2.3254 | 0.5217 | 4.451 | 2.451 |
| Arvid Lindblad | 6.5455 | -2.0 | 4.0998 | 0.466 | 1.2575 | 7.803 |
| Carlos Sainz | 5.1 | 6.3333 | 3.7997 | 0.3955 | 4.8019 | 9.9019 |
| Charles Leclerc | 24.7273 | 10.4 | 7.8752 | 0.5217 | 9.1923 | 33.9196 |
| Esteban Ocon | 6.75 | 5.8 | 4.1423 | 0.5217 | 5.0071 | 11.7571 |
| Fernando Alonso | 0.9167 | 3.6 | 2.931 | 0.5217 | 3.28 | 4.1967 |
| Franco Colapinto | 9.5455 | 4.8 | 4.7227 | 0.5217 | 4.763 | 14.3085 |
| Gabriel Bortoleto | 5.8182 | 0.25 | 3.9488 | 0.466 | 2.2253 | 8.0435 |
| George Russell | 23.1818 | 10.6 | 7.5543 | 0.5217 | 9.1432 | 32.325 |
| Isack Hadjar | 12.6364 | -0.25 | 5.3646 | 0.466 | 2.7484 | 15.3847 |
| Kimi Antonelli | 34.9167 | 9.4 | 9.991 | 0.5217 | 9.6827 | 44.5993 |
| Lance Stroll | -9.4167 | 4.6 | 0.7854 | 0.5217 | 2.7754 | -6.6413 |
| Lando Norris | 19.4167 | 10.2 | 6.7725 | 0.5217 | 8.5605 | 27.9772 |
| Lewis Hamilton | 28.0833 | 6.0 | 8.572 | 0.5217 | 7.2303 | 35.3136 |
| Liam Lawson | 8.4545 | 8.75 | 4.4962 | 0.466 | 6.4783 | 14.9329 |
| Max Verstappen | 19.0833 | 5.8 | 6.7032 | 0.5217 | 6.232 | 25.3154 |
| Nico Hulkenberg | -0.5 | -4.2 | 2.6369 | 0.5217 | -0.9298 | -1.4298 |
| Oliver Bearman | 3.0 | 4.0 | 3.3636 | 0.5217 | 3.6956 | 6.6956 |
| Oscar Piastri | 12.3333 | 6.6 | 5.3016 | 0.5217 | 5.979 | 18.3123 |
| Pierre Gasly | 9.9091 | 3.5 | 4.7983 | 0.466 | 4.1933 | 14.1024 |
| Sergio Perez | -0.0833 | 2.8 | 2.7234 | 0.5217 | 2.7633 | 2.68 |
| Valtteri Bottas | -6.6667 | 0.4 | 1.3564 | 0.5217 | 0.8575 | -5.8092 |
| Yuki Tsunoda | 6.0 | -1.0 | 3.9865 | 0.1791 | 3.0936 | 9.0936 |

## Constructor table

| entity | normal_ev | selected_strength | final_sprint_bonus | sprint_weekend_ev |
| --- | --- | --- | --- | --- |
| Alpine | 25.5833 | 0.5682 | 9.1795 | 34.7629 |
| Aston Martin | -6.75 | 0.1136 | 2.63 | -4.12 |
| Audi | 14.6364 | 0.4091 | 6.8872 | 21.5235 |
| Cadillac | -6.7273 | 0.1591 | 3.2849 | -3.4424 |
| Ferrari | 63.1667 | 0.8636 | 13.4367 | 76.6034 |
| Haas F1 Team | 12.25 | 0.3636 | 6.2322 | 18.4822 |
| McLaren | 44.5 | 0.8409 | 13.1092 | 57.6092 |
| Mercedes | 72.1667 | 1.0 | 15.4016 | 87.5683 |
| Racing Bulls | 28.3333 | 0.6136 | 9.8345 | 38.1678 |
| Red Bull Racing | 40.4167 | 0.75 | 11.7993 | 52.216 |
| Williams | 9.3636 | 0.3182 | 5.5773 | 14.9409 |

## Sanity checks

| entity | entity_type | normal_ev | observed_mean_sprint_bonus | group_bonus | empirical_bayes_weight | final_sprint_bonus | sprint_weekend_ev | sanity_conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aston Martin | constructor | -6.75 | 8.2 | 2.63 |  | 2.63 | -4.12 | smaller positive constructor bonus than the leaders |
| Audi | constructor | 14.6364 | -6.0 | 6.8872 |  | 6.8872 | 21.5235 | smaller positive constructor bonus than the leaders |
| Cadillac | constructor | -6.7273 | 4.0 | 3.2849 |  | 3.2849 | -3.4424 | smaller positive constructor bonus than the leaders |
| Ferrari | constructor | 63.1667 | 16.4 | 13.4367 |  | 13.4367 | 76.6034 | large leading-constructor strength bonus |
| Haas F1 Team | constructor | 12.25 | 9.8 | 6.2322 |  | 6.2322 | 18.4822 | smaller positive constructor bonus than the leaders |
| McLaren | constructor | 44.5 | 16.8 | 13.1092 |  | 13.1092 | 57.6092 | large leading-constructor strength bonus |
| Mercedes | constructor | 72.1667 | 20.0 | 15.4016 |  | 15.4016 | 87.5683 | large leading-constructor strength bonus |
| Red Bull Racing | constructor | 40.4167 | 5.6 | 11.7993 |  | 11.7993 | 52.216 | positive strength-only constructor bonus |
| Williams | constructor | 9.3636 | 11.0 | 5.5773 |  | 5.5773 | 14.9409 | smaller positive constructor bonus than the leaders |
| Charles Leclerc | driver | 24.7273 | 10.4 | 7.8752 | 0.5217 | 9.1923 | 33.9196 | value follows the selected simplified formula |
| George Russell | driver | 23.1818 | 10.6 | 7.5543 | 0.5217 | 9.1432 | 32.325 | material positive bonus for a stronger driver |
| Kimi Antonelli | driver | 34.9167 | 9.4 | 9.991 | 0.5217 | 9.6827 | 44.5993 | material positive bonus for a stronger driver |
| Lance Stroll | driver | -9.4167 | 4.6 | 0.7854 | 0.5217 | 2.7754 | -6.6413 | weak normal form and positive personal Sprint history are both represented |
| Lando Norris | driver | 19.4167 | 10.2 | 6.7725 | 0.5217 | 8.5605 | 27.9772 | material positive bonus for a stronger driver |
| Lewis Hamilton | driver | 28.0833 | 6.0 | 8.572 | 0.5217 | 7.2303 | 35.3136 | value follows the selected simplified formula |
| Liam Lawson | driver | 8.4545 | 8.75 | 4.4962 | 0.466 | 6.4783 | 14.9329 | reasonable positive driver bonus retained |
| Nico Hulkenberg | driver | -0.5 | -4.2 | 2.6369 | 0.5217 | -0.9298 | -1.4298 | negative personal mean retained and moderated toward the group |
| Oliver Bearman | driver | 3.0 | 4.0 | 3.3636 | 0.5217 | 3.6956 | 6.6956 | reasonable positive driver bonus retained |
| Valtteri Bottas | driver | -6.6667 | 0.4 | 1.3564 | 0.5217 | 0.8575 | -5.8092 | moderated EB bonus; old unstable +13 adjustment is absent |

Bottas receives a small shrunk bonus rather than the old unstable +13 adjustment. Hülkenberg's negative observed mean is retained but moderated. The selected formulas also preserve positive separation for stronger drivers and positive, monotonic constructor bonuses, with Mercedes/Ferrari/McLaren above the weaker constructor group.

## Validation

These are historical validation metrics (rounds 2, 4, 5, 9), NOT validation of the refreshed fit, copied from the completed partial-pooling research; no new model search was run.

| entity_type | model | mae | rmse | bias | spearman |
| --- | --- | --- | --- | --- | --- |
| constructor | constant | 7.7759 | 11.304 | 0.0174 | -0.2526 |
| constructor | full_partial_pooling | 6.9458 | 10.2584 | 0.0284 | 0.4053 |
| constructor | personal_mean | 8.5349 | 11.0102 | -0.0 | 0.3511 |
| constructor | shrunk_personal_mean | 7.1924 | 10.3997 | -0.007 | 0.3868 |
| constructor | strength_only | 7.0034 | 10.4031 | 0.0021 | 0.3741 |
| driver | constant | 4.5782 | 6.2307 | 0.0092 | -0.2619 |
| driver | full_partial_pooling | 4.0977 | 5.6108 | 0.011 | 0.437 |
| driver | personal_mean | 4.7608 | 5.9915 | -0.0 | 0.4128 |
| driver | shrunk_personal_mean | 4.0851 | 5.6243 | -0.0081 | 0.4225 |
| driver | strength_only | 4.3563 | 5.8324 | 0.0096 | 0.2969 |

## Recommendation

A. Values look sensible enough for a shadow implementation. The original validation evidence is limited to four completed 2026 Sprints, so a future implementation should remain observable and must be recalibrated as more Sprint data arrives. This report does not activate the model.
