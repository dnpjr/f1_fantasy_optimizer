# Sprint recalibration: Dutch Grand Prix, 2026 round 12

Activated `sprint_ev_2026_v2` on 2026-09-05 using the existing offline fitter and explicit promotion command. The stable runtime filename remains `data/generated/sprint_ev_calibration/sprint_ev_2026_v1.json`; its model_version is v2. The exact old file is in `archive/sprint_ev_2026_v1.json`.

## Included weekends and source

Before: 2026 China (round 2, March 15), Miami (4, May 3), Canada (5, May 24), Britain (9, July 5). Normal-form history covered completed rounds 1–11.

After: the same four plus **Netherlands / Dutch Grand Prix (round 12, August 23; Sprint August 22)**. Normal-form history covers rounds 1–12. Singapore (17) is the next catalogued Sprint, still in the future. Italian round 13 is unplayed in the captured official feed. Event identity follows the repository's season/round catalogue, not another calendar's numbering.

Official public `https://fantasy.formula1.com/feeds/popup/playerstats_{player_id}.json` payloads were retrieved for both current and historical Fantasy asset IDs. The retained snapshot includes the unmodified gameday-12 entries and roster metadata, including inactive assets; other gamedays were omitted. `scripts/build_historical_fantasy_scores.py --official-snapshot` replays them through the existing parser and canonical identity normaliser. No finishing-position reconstruction was substituted for official Fantasy points. The repository's incomplete 2026 Jolpica results cache is not the calibration source.

33 Dutch rows were added: 22 played drivers and 11 constructors. All 2,458 previous canonical rows are unchanged; the new total is 2,491 unique rows. CSV and Parquet agree, and repeated snapshot imports leave the canonical CSV hash unchanged.

## Definitions and fitted quantities

The target is the sum of available official `sprint_points` and `sprint_qualifying_points`, with min_count=1. If both are missing, the target is missing, never zero. Normal-equivalent form is the recorded weekend total minus that target on Sprint weekends; ordinary weekends retain their recorded totals. GP qualifying, GP race, and residual points never enter the fitted Sprint target. Negative Sprint scores remain valid. This exactly preserves the old observation definition.

Drivers: fit a nonnegative-slope group regression against population-standardised normal form. Estimate within-driver residual variance and between-driver signal variance (tau squared), then use w = tau² / (tau² + within_variance / n) to combine each driver's personal Sprint mean with the group bonus. Form uses equal-weight completed normal-equivalent scores during fitting; runtime form follows the user's selected races and decay.

Constructors: fit a separate intercept and nonnegative slope against 0.75 * form percentile + 0.25 * current-price percentile. These weights are fixed, not newly estimated. No personal constructor effect or future-event effect is fitted.

Before: 85 valid driver and 43 constructor targets. After: 104 and 53. The observation grid also contains explicit missing rows, so grid size must not be confused with fit sample size.

## Parameter comparison

| parameter | old_active | new_active | total_change |
|---|---|---|---|
| driver.form_mean | 9.743985308 | 9.467457181 | -0.276528127 |
| driver.form_sd | 11.258787528 | 10.915416414 | -0.343371114 |
| driver.group_intercept | 5.167337315 | 4.706549290 | -0.460788025 |
| driver.group_slope | 2.328882121 | 2.266533075 | -0.062349046 |
| driver.within_variance | 26.378787879 | 24.175000000 | -2.203787879 |
| driver.tau_squared | 4.710348566 | 5.273338292 | 0.562989726 |
| constructor.intercept | 0.941047806 | 0.992577958 | 0.051530152 |
| constructor.slope | 16.210891020 | 14.409013242 | -1.801877778 |
| constructor.form_weight | 0.750000000 | 0.750000000 | 0.000000000 |
| constructor.price_weight | 0.250000000 | 0.250000000 | 0.000000000 |
| constructor.future_event_effect | 0.000000000 | 0.000000000 | 0.000000000 |

`parameter_comparison.csv` additionally separates the new-weekend effect from the market-snapshot effect. Refitting the old four weekends with today's market reproduces all six driver parameters. Constructor price ranks changed: intercept 0.941047806 → 1.116582247 and slope 16.210891020 → 15.876608818 before adding Dutch. At that fixed market, Dutch changes them by -0.124004289 and -1.467595576 respectively. Do not attribute the entire constructor change to the new race.

`personal_history_comparison.csv` lists every driver's old/new personal mean and count. `sprint_observations.csv`, `normalised_history.csv`, and `included_events.csv` expose the actual fitting inputs. `comparison_to_active.json` compares personal history by canonical identity, avoiding misleading differences from list reordering. Runtime input hashes cover canonical data, schedule, and market snapshot.

## Problems found and corrected

1. The production recalibration path inherited a research cutoff at round 11, including hidden caps in the observation and form builders. Production now requests all completed canonical rounds. The original research scripts keep their default round-11 cutoff for reproducibility. New partial weekends, duplicate identities, gaps, future dates, and source/schedule conflicts are rejected.
2. The current selectable roster dropped inactive Hadjar and caused the old research-universe equality check to abort. Historical participants are now retained using canonical identity and their latest recorded price if unavailable in the current market; old research predictions are comparison-only. Lawson's new asset ID 116 maps to the same human history as old ID 114. Newcomer Tsunoda receives one valid Dutch observation; Hadjar retains four earlier observations.
3. Production previously added a Sprint bonus to whole-weekend form, counting historical Sprint points within the baseline. On Sprint projections only, both the model-history route and official-playerstats projection route now subtract known Sprint components. Sprint rows without a usable component split are omitted from those baseline inputs. This includes older-season recorded Sprint totals without session splits. Normal projections take exactly the previous calculation path with the same supplied history. Actual recorded totals remain available for price/history displays.

The third correction changes Sprint baselines independently of refitted coefficients; it is not included in the parameter-delta table.

## Missing data and limitations

Dutch labelled Sprint components are missing for Lawson, Bortoleto, Lindblad, and Cadillac despite played GP totals. Their Sprint targets and normal-equivalent Dutch scores are omitted. Hülkenberg has -10 labelled Sprint Qualifying points in the official feed; retaining that field follows the existing methodology and yields normal-equivalent 15 from weekend total 5. No inferred zero or manually invented Sprint score was added.

The latest roster is complete, but the available driver Sprint-component coverage for this event alone is 19/22 (86.4%). Across the full driver fitting sample coverage remains above the existing 90% threshold. The original method checks aggregate coverage, not 90% per event. Constructor Dutch coverage is 10/11. Limited sample size and missing components remain reasons to treat estimates cautiously.

No explosive coefficient or sign reversal appeared. Driver within-variance decreased about 8.4%, tau² increased about 12.0%, and constructor slope decreased about 11.1% overall. The copied leave-one-Sprint-out metrics in `comparison_to_previous_models.csv` are explicitly labelled historical four-Sprint research results, not new validation metrics. The selected model was refitted without a new model-selection search.

## Reproduction

```sh
.venv/bin/python scripts/build_historical_fantasy_scores.py --official-snapshot data/generated/sprint_ev_calibration/sources/official_2026_round_12.json
.venv/bin/python scripts/recalibrate_sprint_ev.py --current-prices data/generated/sprint_ev_calibration/sources/market_2026_round_13.json --output reports/sprint_recalibration_candidate
```

Candidate generation does not change production. The v2 candidate in this directory was already promoted using the existing `--promote` command. Rebuilding now will propose v3 because production is v2. To reproduce the v2 candidate version exactly, pass `--active data/generated/sprint_ev_calibration/archive/sprint_ev_2026_v1.json` to generation and write to a separate output directory.

## Validation

150 targeted tests passed across Sprint production, raw-snapshot recalibration, final-candidate fitting, personalised adjustments, component EV, both shadow models, historical scores, and race selection. Checks cover exact parameter reproduction, the new weekend included once, canonical/source IDs, complete raw replay, missing and partial input rejection, old observation equivalence, negative points, constructor/driver separation, baseline subtraction, single bonus application, versioned promotion and archive, decay/history modes, and optimiser constraints. A normal-weekend integration test makes Sprint baseline normalisation raise if called and confirms exactly equal next-race/horizon EVs; the optimiser result also remains identical.

The local Python readline extension crashes during pytest startup. Tests were run by setting `sys.modules['readline'] = None` before invoking pytest.main; no test was skipped for this. Existing PuLP deprecation warnings remain. CSV/Parquet and idempotent-import checks passed; PyArrow emitted harmless sandbox CPU-capability probe warnings.

## Files changed by this update

Source:
- scripts/build_historical_fantasy_scores.py
- scripts/calibrate_asset_sprint_adjustments.py
- scripts/build_2026_sprint_final_candidate.py
- scripts/recalibrate_sprint_ev.py
- f1fantasy/model.py (Sprint baseline normalisation helper)
- f1fantasy/app_core.py (Sprint-only normalised inputs for both production routes)

Tests:
- tests/test_sprint_recalibration_update.py (new)
- tests/test_sprint_production.py
- tests/test_2026_sprint_final_candidate.py
- tests/test_historical_scores.py

Artifacts:
- data/generated/historical_fantasy_scores_v3_recorded_2023_2026/: canonical CSV/Parquet, coverage/summary outputs and provenance manifest regenerated by the existing builder
- data/generated/sprint_ev_calibration/sprint_ev_2026_v1.json (active v2)
- data/generated/sprint_ev_calibration/archive/sprint_ev_2026_v1.json
- data/generated/sprint_ev_calibration/sources/official_2026_round_12.json
- data/generated/sprint_ev_calibration/sources/market_2026_round_13.json
- reports/sprint_recalibration_2026_round_12/: fitter outputs and the audit/comparison files listed above

The repository already had extensive uncommitted work. This update preserves it; no unrelated refactor, UI edit, commit, or deployment was performed.
