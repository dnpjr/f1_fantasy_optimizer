# Simplified 2026 Sprint-EV final candidate

## Executive recommendation

**A. Values look sensible enough for a shadow implementation.** This is a research recommendation only; the model is not production-approved or activated.

Normal EV is the equal-weight mean of completed round-1--11 normal-equivalent scores. Normal weekends retain recorded total; Sprint weekends subtract official Sprint and Sprint-qualifying points. With application decay `p`, this generalises to a recency-weighted mean whose included scores receive weights `1, p, p², ...` newest first.

Driver formulas:

- `z_form = (normal_ev - 9.743985307622) / 11.258787528190` (within the current driver class, population SD).
- `group_bonus = 5.167337315125 + 2.328882120968 * z_form`.
- `personal_mean = arithmetic mean of valid official Sprint-only bonuses`, retaining negative values and omitting missing observations.
- `w_i = tau² / (tau² + sigma_within² / n_i)`, where `tau²=4.710348565856` and `sigma_within²=26.378787878788`.
- `Sprint bonus = w_i * personal_mean + (1 - w_i) * group_bonus`; an asset with no valid observation gets `w_i=0`.
- `Sprint EV = normal_ev + Sprint bonus`.

Constructor formulas:

- `strength = 0.75 * form_percentile + 0.25 * price_percentile`.
- `Sprint bonus = 0.941047805698 + 16.210891019808 * strength`, with the slope constrained nonnegative.
- `Sprint EV = normal_ev + Sprint bonus`.

The simplified candidate has no asset-specific constructor effect and no future Sprint-event effect; `v_next=0`.

## Driver table

| entity | normal_ev | observed_mean_sprint_bonus | group_bonus | empirical_bayes_weight | final_sprint_bonus | sprint_weekend_ev |
| --- | --- | --- | --- | --- | --- | --- |
| Alexander Albon | -2.2727 | 7.75 | 2.6817 | 0.4167 | 4.7934 | 2.5207 |
| Arvid Lindblad | 6.5455 | -2.0 | 4.5057 | 0.4167 | 1.7951 | 8.3405 |
| Carlos Sainz | 5.2222 | 8.0 | 4.232 | 0.2632 | 5.2236 | 10.4458 |
| Charles Leclerc | 24.2 | 10.75 | 8.1576 | 0.4167 | 9.2377 | 33.4377 |
| Esteban Ocon | 8.4545 | 6.75 | 4.9006 | 0.4167 | 5.6712 | 14.1257 |
| Fernando Alonso | -0.4545 | 2.5 | 3.0578 | 0.4167 | 2.8254 | 2.3708 |
| Franco Colapinto | 10.1 | 5.5 | 5.241 | 0.4167 | 5.3489 | 15.4489 |
| Gabriel Bortoleto | 5.8182 | 0.25 | 4.3553 | 0.4167 | 2.6448 | 8.463 |
| George Russell | 23.1 | 11.25 | 7.93 | 0.4167 | 9.3133 | 32.4133 |
| Isack Hadjar | 12.6364 | -0.25 | 5.7656 | 0.4167 | 3.2592 | 15.8955 |
| Kimi Antonelli | 35.3636 | 10.0 | 10.4668 | 0.4167 | 10.2723 | 45.6359 |
| Lance Stroll | -9.1818 | 5.25 | 1.2525 | 0.4167 | 2.9181 | -6.2637 |
| Lando Norris | 16.9091 | 10.25 | 6.6494 | 0.4167 | 8.1496 | 25.0587 |
| Lewis Hamilton | 28.7273 | 7.0 | 9.094 | 0.4167 | 8.2215 | 36.9488 |
| Liam Lawson | 8.4545 | 8.75 | 4.9006 | 0.4167 | 6.5045 | 14.959 |
| Max Verstappen | 22.2727 | 6.5 | 7.7589 | 0.4167 | 7.2344 | 29.5071 |
| Nico Hulkenberg | -1.9091 | -2.75 | 2.7569 | 0.4167 | 0.4624 | -1.4467 |
| Oliver Bearman | 4.8182 | 4.0 | 4.1484 | 0.4167 | 4.0866 | 8.9048 |
| Oscar Piastri | 11.8182 | 7.5 | 5.5964 | 0.4167 | 6.3895 | 18.2077 |
| Pierre Gasly | 10.2 | 4.3333 | 5.2617 | 0.3488 | 4.9378 | 15.1378 |
| Sergio Perez | -0.9091 | 3.75 | 2.9637 | 0.4167 | 3.2913 | 2.3823 |
| Valtteri Bottas | -5.5455 | 0.25 | 2.0047 | 0.4167 | 1.2736 | -4.2719 |

## Constructor table

| entity | normal_ev | selected_strength | final_sprint_bonus | sprint_weekend_ev |
| --- | --- | --- | --- | --- |
| Alpine | 26.4545 | 0.5682 | 10.1518 | 36.6063 |
| Aston Martin | -8.0909 | 0.1136 | 2.7832 | -5.3077 |
| Audi | 12.6 | 0.3409 | 6.4675 | 19.0675 |
| Cadillac | -6.7273 | 0.1591 | 3.5201 | -3.2072 |
| Ferrari | 63.1818 | 0.8636 | 14.9414 | 78.1232 |
| Haas F1 Team | 15.7273 | 0.4318 | 7.9412 | 23.6685 |
| McLaren | 42.4545 | 0.7727 | 13.4676 | 55.9222 |
| Mercedes | 72.4545 | 1.0 | 17.1519 | 89.6065 |
| Racing Bulls | 28.8182 | 0.5909 | 10.5202 | 39.3384 |
| Red Bull Racing | 43.3636 | 0.8182 | 14.2045 | 57.5681 |
| Williams | 9.2 | 0.3409 | 6.4675 | 15.6675 |

## Sanity checks

| entity | entity_type | normal_ev | observed_mean_sprint_bonus | group_bonus | empirical_bayes_weight | final_sprint_bonus | sprint_weekend_ev | sanity_conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aston Martin | constructor | -8.0909 | 7.75 | 2.7832 |  | 2.7832 | -5.3077 | smaller positive constructor bonus than the leaders |
| Audi | constructor | 12.6 | -5.0 | 6.4675 |  | 6.4675 | 19.0675 | smaller positive constructor bonus than the leaders |
| Cadillac | constructor | -6.7273 | 4.0 | 3.5201 |  | 3.5201 | -3.2072 | smaller positive constructor bonus than the leaders |
| Ferrari | constructor | 63.1818 | 17.75 | 14.9414 |  | 14.9414 | 78.1232 | large leading-constructor strength bonus |
| Haas F1 Team | constructor | 15.7273 | 10.75 | 7.9412 |  | 7.9412 | 23.6685 | smaller positive constructor bonus than the leaders |
| McLaren | constructor | 42.4545 | 17.75 | 13.4676 |  | 13.4676 | 55.9222 | large leading-constructor strength bonus |
| Mercedes | constructor | 72.4545 | 21.25 | 17.1519 |  | 17.1519 | 89.6065 | large leading-constructor strength bonus |
| Red Bull Racing | constructor | 43.3636 | 6.25 | 14.2045 |  | 14.2045 | 57.5681 | positive strength-only constructor bonus |
| Williams | constructor | 9.2 | 13.3333 | 6.4675 |  | 6.4675 | 15.6675 | smaller positive constructor bonus than the leaders |
| Charles Leclerc | driver | 24.2 | 10.75 | 8.1576 | 0.4167 | 9.2377 | 33.4377 | value follows the selected simplified formula |
| George Russell | driver | 23.1 | 11.25 | 7.93 | 0.4167 | 9.3133 | 32.4133 | material positive bonus for a stronger driver |
| Kimi Antonelli | driver | 35.3636 | 10.0 | 10.4668 | 0.4167 | 10.2723 | 45.6359 | material positive bonus for a stronger driver |
| Lance Stroll | driver | -9.1818 | 5.25 | 1.2525 | 0.4167 | 2.9181 | -6.2637 | weak normal form and positive personal Sprint history are both represented |
| Lando Norris | driver | 16.9091 | 10.25 | 6.6494 | 0.4167 | 8.1496 | 25.0587 | material positive bonus for a stronger driver |
| Lewis Hamilton | driver | 28.7273 | 7.0 | 9.094 | 0.4167 | 8.2215 | 36.9488 | value follows the selected simplified formula |
| Liam Lawson | driver | 8.4545 | 8.75 | 4.9006 | 0.4167 | 6.5045 | 14.959 | reasonable positive driver bonus retained |
| Nico Hulkenberg | driver | -1.9091 | -2.75 | 2.7569 | 0.4167 | 0.4624 | -1.4467 | negative personal mean retained and moderated toward the group |
| Oliver Bearman | driver | 4.8182 | 4.0 | 4.1484 | 0.4167 | 4.0866 | 8.9048 | reasonable positive driver bonus retained |
| Valtteri Bottas | driver | -5.5455 | 0.25 | 2.0047 | 0.4167 | 1.2736 | -4.2719 | moderated EB bonus; old unstable +13 adjustment is absent |

Bottas receives a small shrunk bonus rather than the old unstable +13 adjustment. Hülkenberg's negative observed mean is retained but moderated. The selected formulas also preserve positive separation for stronger drivers and positive, monotonic constructor bonuses, with Mercedes/Ferrari/McLaren above the weaker constructor group.

## Validation

These are the established leave-one-Sprint-out results copied from the completed partial-pooling research; no new model search was run.

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

A. Values look sensible enough for a shadow implementation. The evidence is limited to four completed 2026 Sprints, so a future implementation should remain observable and must be recalibrated as more Sprint data arrives. This report does not activate the model.
