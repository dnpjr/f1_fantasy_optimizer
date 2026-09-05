# 2026 Sprint shadow versus production EV

Inputs are frozen local data: verified official feed 12 (2026-08-10T14:56:47.943603+00:00), completed canonical rounds 1–11, and the canonical schedule identifying Dutch GP round 12 as Sprint. No network refresh or calibration was run.

## 1. Executive result

- Sprint-aware EV materially changes asset rankings; the largest driver uplift is **Kimi Antonelli +14.13**, and the largest constructor uplift is **Mercedes +19.14**.
- At the primary p=0.85/all-supported/points-only setting, the optimiser changes **4** asset slot(s), with **3/7** assets overlapping.
- Under Sprint EV, the shadow-selected team gains **34.13** points over the production-selected team. Under production EV it gives up **7.14** points.
- The shadow-selected team is not fully stable across p=1.00, 0.85 and 0.70 (3 distinct shadow team composition(s)).
- All 22 driver and 11 constructor prices exactly match the accepted verified official market.

## 2. Drivers

Top 10 by Sprint-aware EV:

| Asset | Production EV | Normal EV | Sprint bonus | Sprint EV | Rank change |
|---|---|---|---|---|---|
| Kimi Antonelli | 27.06 | 31.40 | 9.79 | 41.19 | 4 |
| Lewis Hamilton | 27.67 | 28.57 | 8.20 | 36.77 | 0 |
| Charles Leclerc | 27.63 | 26.30 | 9.49 | 35.79 | 0 |
| Max Verstappen | 30.70 | 22.83 | 7.30 | 30.13 | -3 |
| Lando Norris | 27.46 | 21.40 | 8.69 | 30.09 | -1 |
| George Russell | 24.67 | 20.07 | 8.95 | 29.01 | 0 |
| Isack Hadjar | 12.16 | 16.99 | 3.78 | 20.77 | 1 |
| Oscar Piastri | 18.77 | 11.26 | 6.32 | 17.58 | -1 |
| Liam Lawson | 7.25 | 8.98 | 6.57 | 15.55 | 1 |
| Franco Colapinto | 7.06 | 9.92 | 5.33 | 15.25 | 1 |

Top 10 production order: Max Verstappen, Lewis Hamilton, Charles Leclerc, Lando Norris, Kimi Antonelli, George Russell, Oscar Piastri, Isack Hadjar, Pierre Gasly, Liam Lawson.

Top 10 Sprint order: Kimi Antonelli, Lewis Hamilton, Charles Leclerc, Max Verstappen, Lando Norris, George Russell, Isack Hadjar, Oscar Piastri, Liam Lawson, Franco Colapinto.

Largest movers up:

| Asset | Production EV | Normal EV | Sprint bonus | Sprint EV | Rank change |
|---|---|---|---|---|---|
| Kimi Antonelli | 27.06 | 31.40 | 9.79 | 41.19 | 4 |
| Gabriel Bortoleto | 4.78 | 7.29 | 2.82 | 10.11 | 3 |
| Isack Hadjar | 12.16 | 16.99 | 3.78 | 20.77 | 1 |
| Liam Lawson | 7.25 | 8.98 | 6.57 | 15.55 | 1 |
| Franco Colapinto | 7.06 | 9.92 | 5.33 | 15.25 | 1 |

Largest movers down:

| Asset | Production EV | Normal EV | Sprint bonus | Sprint EV | Rank change |
|---|---|---|---|---|---|
| Max Verstappen | 30.70 | 22.83 | 7.30 | 30.13 | -3 |
| Arvid Lindblad | 6.43 | 6.51 | 1.79 | 8.31 | -2 |
| Pierre Gasly | 8.08 | 10.21 | 4.94 | 15.15 | -2 |
| Sergio Perez | 0.37 | -4.82 | 2.82 | -2.00 | -1 |
| Oscar Piastri | 18.77 | 11.26 | 6.32 | 17.58 | -1 |

## 3. Constructors

Top 10 by Sprint-aware EV:

| Asset | Production EV | Normal EV | Sprint bonus | Sprint EV | Rank change |
|---|---|---|---|---|---|
| Mercedes | 64.83 | 66.82 | 17.15 | 83.98 | 0 |
| Ferrari | 62.91 | 64.05 | 14.94 | 78.99 | 0 |
| Red Bull Racing | 56.93 | 49.27 | 14.20 | 63.47 | 1 |
| McLaren | 58.30 | 46.79 | 13.47 | 60.25 | -1 |
| Racing Bulls | 25.02 | 31.01 | 10.52 | 41.53 | 0 |
| Alpine | 22.16 | 25.64 | 10.15 | 35.79 | 0 |
| Audi | 13.19 | 15.32 | 7.57 | 22.90 | 1 |
| Haas F1 Team | 17.09 | 13.86 | 6.84 | 20.69 | -1 |
| Williams | 11.61 | 8.96 | 6.47 | 15.43 | 0 |
| Aston Martin | 8.73 | -1.86 | 3.89 | 2.03 | 0 |

Top 10 production order: Mercedes, Ferrari, McLaren, Red Bull Racing, Racing Bulls, Alpine, Haas F1 Team, Audi, Williams, Aston Martin.

Top 10 Sprint order: Mercedes, Ferrari, Red Bull Racing, McLaren, Racing Bulls, Alpine, Audi, Haas F1 Team, Williams, Aston Martin.

Largest movers up:

| Asset | Production EV | Normal EV | Sprint bonus | Sprint EV | Rank change |
|---|---|---|---|---|---|
| Audi | 13.19 | 15.32 | 7.57 | 22.90 | 1 |
| Red Bull Racing | 56.93 | 49.27 | 14.20 | 63.47 | 1 |

Largest movers down:

| Asset | Production EV | Normal EV | Sprint bonus | Sprint EV | Rank change |
|---|---|---|---|---|---|
| McLaren | 58.30 | 46.79 | 13.47 | 60.25 | -1 |
| Haas F1 Team | 17.09 | 13.86 | 6.84 | 20.69 | -1 |

## 4. Optimiser comparison

Budget: **117.00M**, derived from persisted current-team value 116.80M plus 0.20M bank. Standard 5-driver/2-constructor roster, one 2× driver, no chip, no locks or exclusions.

Production-selected team:

- Drivers: Arvid Lindblad, Isack Hadjar, Max Verstappen, Nico Hulkenberg, Sergio Perez
- Constructors: Ferrari, Mercedes
- 2×: Max Verstappen
- Cost: 116.10M

Sprint-shadow-selected team:

- Drivers: Esteban Ocon, Gabriel Bortoleto, Kimi Antonelli, Liam Lawson, Nico Hulkenberg
- Constructors: Ferrari, Mercedes
- 2×: Kimi Antonelli
- Cost: 116.70M

Swaps: drivers out **Arvid Lindblad, Isack Hadjar, Max Verstappen, Sergio Perez**, drivers in **Esteban Ocon, Gabriel Bortoleto, Kimi Antonelli, Liam Lawson**; constructors out **none**, constructors in **none**.

| Selected team | Scored by production EV | Scored by Sprint EV |
|---|---:|---:|
| Production-selected | 208.38 | 250.50 |
| Sprint-selected | 201.24 | 284.63 |

The separate combined-objective rows in `optimiser_comparison.csv` keep the same price-growth projection and weight 50; only the points field differs.

### Current default combined objective

Price-growth input is deterministic canonical official scoring from each asset's latest two completed observations. Production-selected: **Charles Leclerc, Gabriel Bortoleto, Isack Hadjar, Max Verstappen, Nico Hulkenberg | Ferrari, Racing Bulls | 2× Max Verstappen**. Sprint-selected: **Charles Leclerc, Gabriel Bortoleto, Isack Hadjar, Kimi Antonelli, Nico Hulkenberg | Ferrari, Racing Bulls | 2× Kimi Antonelli**.

| Selected team | Scored by production EV | Scored by Sprint EV |
|---|---:|---:|
| Production-selected | 194.18 | 247.65 |
| Sprint-selected | 186.90 | 269.78 |

## 5. Decay sensitivity

| p | Swaps | Overlap | Shadow-team Sprint score | Shadow advantage | Production penalty |
|---|---|---|---|---|---|
| 1.00 | 4 | 3 | 296.87 | 36.37 | 4.35 |
| 0.85 | 4 | 3 | 284.63 | 34.13 | 7.14 |
| 0.70 | 3 | 4 | 287.26 | 27.96 | 3.36 |

The personal Sprint histories and calibration coefficients are identical in every row; only selected-race form weighting changes.

## 6. Current-season vs all-history production comparison

| Production history | Production team / production score | Shadow team / Sprint score | Swaps | Shadow advantage | Production penalty |
|---|---|---|---|---|---|
| current_season_only | 239.75 | 284.63 | 0 | 0.00 | 0.00 |
| all_supported | 208.38 | 284.63 | 4 | 34.13 | 7.14 |

The shadow side remains `2026_only` in both cases. Differences between these rows therefore isolate the effect of the production model's older-season prior.

## 7. Difference decomposition and sanity review

`Sprint EV = normal-equivalent EV + Sprint bonus`. `baseline_difference = normal-equivalent EV - production EV` separates normal-form disagreement from the actual Sprint adjustment.

Largest absolute baseline differences:

| Asset | Production EV | Normal EV | Baseline difference | Sprint bonus |
|---|---|---|---|---|
| McLaren | 58.30 | 46.79 | -11.51 | 13.47 |
| Aston Martin | 8.73 | -1.86 | -10.59 | 3.89 |
| Max Verstappen | 30.70 | 22.83 | -7.87 | 7.30 |
| Red Bull Racing | 56.93 | 49.27 | -7.66 | 14.20 |
| Oscar Piastri | 18.77 | 11.26 | -7.52 | 6.32 |

Largest Sprint bonuses:

| Asset | Normal EV | Sprint bonus | Sprint EV |
|---|---|---|---|
| Mercedes | 66.82 | 17.15 | 83.98 |
| Ferrari | 64.05 | 14.94 | 78.99 |
| Red Bull Racing | 49.27 | 14.20 | 63.47 |
| McLaren | 46.79 | 13.47 | 60.25 |
| Racing Bulls | 31.01 | 10.52 | 41.53 |

Sanity cases:

| Asset | Production EV | Normal EV | Sprint bonus | Sprint EV | Assessment | Review |
|---|---|---|---|---|---|---|
| Kimi Antonelli | 27.06 | 31.40 | 9.79 | 41.19 | sensible | Driver bonus 9.79 combines form-linked group expectation with frozen personal history. |
| George Russell | 24.67 | 20.07 | 8.95 | 29.01 | sensible | Driver bonus 8.95 combines form-linked group expectation with frozen personal history. |
| Lando Norris | 27.46 | 21.40 | 8.69 | 30.09 | sensible | Driver bonus 8.69 combines form-linked group expectation with frozen personal history. |
| Lewis Hamilton | 27.67 | 28.57 | 8.20 | 36.77 | sensible | Driver bonus 8.20 combines form-linked group expectation with frozen personal history. |
| Charles Leclerc | 27.63 | 26.30 | 9.49 | 35.79 | sensible | Driver bonus 9.49 combines form-linked group expectation with frozen personal history. |
| Liam Lawson | 7.25 | 8.98 | 6.57 | 15.55 | sensible | Driver bonus 6.57 combines form-linked group expectation with frozen personal history. |
| Oliver Bearman | 4.98 | 3.88 | 3.97 | 7.85 | sensible | Driver bonus 3.97 combines form-linked group expectation with frozen personal history. |
| Nico Hulkenberg | 0.28 | -0.45 | 0.64 | 0.19 | sensible | Conservative 0.64-point bonus reflects the stored -2.75 personal Sprint mean. |
| Valtteri Bottas | -3.55 | -6.94 | 1.11 | -5.83 | sensible | Bonus is 1.11, avoiding the former unstable +13-style artefact. |
| Lance Stroll | -2.07 | -6.54 | 3.24 | -3.31 | sensible | Driver bonus 3.24 combines form-linked group expectation with frozen personal history. |
| Mercedes | 64.83 | 66.82 | 17.15 | 83.98 | sensible | Strength-ranked constructor bonus is 17.15; no personal/event effect is used. |
| Ferrari | 62.91 | 64.05 | 14.94 | 78.99 | sensible | Strength-ranked constructor bonus is 14.94; no personal/event effect is used. |
| McLaren | 58.30 | 46.79 | 13.47 | 60.25 | sensible | Strength-ranked constructor bonus is 13.47; no personal/event effect is used. |
| Red Bull Racing | 56.93 | 49.27 | 14.20 | 63.47 | sensible | Strength-ranked constructor bonus is 14.20; no personal/event effect is used. |
| Haas F1 Team | 17.09 | 13.86 | 6.84 | 20.69 | sensible | Strength-ranked constructor bonus is 6.84; no personal/event effect is used. |
| Williams | 11.61 | 8.96 | 6.47 | 15.43 | sensible | Strength-ranked constructor bonus is 6.47; no personal/event effect is used. |
| Audi | 13.19 | 15.32 | 7.57 | 22.90 | sensible | Strength-ranked constructor bonus is 7.57; no personal/event effect is used. |
| Aston Martin | 8.73 | -1.86 | 3.89 | 2.03 | sensible | Strength-ranked constructor bonus is 3.89; no personal/event effect is used. |
| Cadillac | -11.11 | -12.06 | 2.41 | -9.64 | sensible | Strength-ranked constructor bonus is 2.41; no personal/event effect is used. |

## 8. Recommendation

**C. Asset-level values look reasonable, but optimiser impact is unstable.**

This is an analysis recommendation only. Sprint-aware EV remains shadow-only and is not activated in production or optimisation.
