# Repository size and artifact audit — 2026-09-05

Measured committed snapshot: `084589fdcd6e318010aba4bd2a7ee9897e36d21d` (409 files). No model fitting, activation, commits, history rewriting or pushes were performed during this audit.

## Scope and measurements

The committed file payload is **9,185,130 bytes (9.19 MB / 8.76 MiB)**. This sums Git blob contents at HEAD, rather than filesystem allocation or compressed history. The working checkout occupies approximately **477 MiB**, predominantly an ignored **438 MiB virtual environment**; `.git` occupies approximately **12 MiB**. Git reports 11.60 MiB of loose objects plus a 177.62 KiB pack. Removing a file from a future commit does not remove its historical blobs. No history cleanup is warranted.

Four cache files already had local refresh changes when this audit began: `qualifying_2026.csv.meta.json`, `results_2026.csv.meta.json`, `sprint_2026.csv.meta.json`, and `verified_fantasy_market.json`, all under `data/cache/`. They were preserved. Measurements below use HEAD, not those mutable local versions.

Only two zero-byte directory placeholders were removed: `data/.gitkeep` and `data/cache/.gitkeep`. Both directories contain many required tracked files, and no workflow references these placeholders. The release payload therefore remains **9,185,130 bytes across 407 existing files**, before adding this audit document. No data bytes were removed. HEAD itself remains unchanged until changes are committed.

### Exclusive category breakdown

Each file appears once. Python files are classified by location; non-Python report contents stay in Reports; calibration directories include their retained source snapshots. Remaining data extensions include fixtures and licensed raw `.config`/ZIP sources. Markdown outside those artifact bundles is Documentation. Text lines include headers, blank lines and serialized JSON; binary files contribute bytes but no lines. These are not all source-code lines.

| Category | Files | Text lines | Bytes | MB |
| --- | ---: | ---: | ---: | ---: |
| CI/configuration | 7 | 119 | 3,069 | 0.003 |
| other | 4 | 42 | 2,135 | 0.002 |
| documentation/Markdown | 13 | 1,086 | 75,116 | 0.075 |
| CSV/JSON/Parquet/data files | 192 | 36,703 | 4,148,732 | 4.149 |
| calibration/model artifacts | 5 | 36,954 | 1,378,494 | 1.378 |
| screenshots/images | 2 | 0 | 149,710 | 0.150 |
| Python application/source code | 24 | 23,003 | 982,892 | 0.983 |
| reports | 93 | 11,129 | 1,294,066 | 1.294 |
| scripts | 20 | 10,664 | 471,722 | 0.472 |
| tests | 49 | 17,828 | 679,194 | 0.679 |

Data, calibration artifacts and report categories together account for **6,821,292 bytes (6.82 MB; 74.3%)**. These exclusive category totals include report prose and research tables; they are not a second tally of every data extension.

### Python LOC

“Source” includes application and research/maintenance scripts, excluding tests. Physical lines are the primary reproducible LOC measure. The approximate code count removes blank lines, standalone comments and AST-recognized module/class/function docstrings; it is not a complexity metric.

| Python group | Files | Physical LOC | Nonblank LOC | Approximate code LOC |
| --- | ---: | ---: | ---: | ---: |
| Source excluding tests | 44 | 33,667 | 30,943 | 30,375 |
| Tests | 49 | 17,828 | 15,362 | 15,347 |
| Total | 93 | 51,495 | 46,305 | 45,722 |

### Largest 20 committed files

| File | Bytes |
| --- | ---: |
| `data/generated/sprint_ev_calibration/sources/official_2026_round_12.json` | 1,274,491 |
| `data/generated/historical_fantasy_scores_v2_recorded_2022_2026/historical_fantasy_scores_2022_2026.csv` | 1,143,110 |
| `data/generated/historical_fantasy_scores_v3_recorded_2023_2026/historical_fantasy_scores_2023_2026.csv` | 906,493 |
| `data/research/sprint_round_11/canonical.csv` | 897,972 |
| `reports/sprint_linear_regression/observation_dataset.csv` | 304,662 |
| `f1fantasy/app_core.py` | 281,720 |
| `reports/2026_sprint_partial_pooling/leave_one_sprint_out.csv` | 199,436 |
| `streamlit_app.py` | 199,120 |
| `scripts/api_probe/results.json` | 143,474 |
| `reports/sprint_recalibration_2026_round_12/normalised_history.csv` | 143,329 |
| `data/generated/sprint_ev_calibration/sources/market_2026_round_13.json` | 92,495 |
| `scripts/api_probe/results.csv` | 90,954 |
| `scripts/api_probe/public_archive_search_results.json` | 84,353 |
| `scripts/api_probe/fixtures/playerstats_constructor_2026_sanitised.json` | 81,740 |
| `docs/images/optimiser.png` | 79,666 |
| `docs/RELEASE_FILE_CLASSIFICATION.csv` | 73,900 |
| `data/cache/verified_fantasy_market.json` | 70,452 |
| `docs/images/market-thresholds.png` | 70,044 |
| `scripts/api_probe/fixtures/playerstats_driver_2026_sanitised.json` | 69,971 |
| `scripts/analyse_sprint_linear_regression.py` | 60,157 |

### Largest 20 committed directories

Recursive totals overlap: a parent includes its children. Do not sum this table.

| Directory | Bytes |
| --- | ---: |
| `data/` | 4,685,689 |
| `data/generated/` | 3,555,347 |
| `data/generated/sprint_ev_calibration/` | 1,375,334 |
| `data/generated/sprint_ev_calibration/sources/` | 1,366,986 |
| `reports/` | 1,294,066 |
| `scripts/` | 1,251,023 |
| `data/generated/historical_fantasy_scores_v2_recorded_2022_2026/` | 1,207,219 |
| `data/generated/historical_fantasy_scores_v3_recorded_2023_2026/` | 969,634 |
| `data/research/sprint_round_11/` | 904,064 |
| `data/research/` | 904,064 |
| `scripts/api_probe/` | 867,906 |
| `f1fantasy/` | 783,772 |
| `tests/` | 695,191 |
| `reports/sprint_linear_regression/` | 368,067 |
| `reports/2026_sprint_partial_pooling/` | 311,605 |
| `docs/` | 264,833 |
| `reports/sprint_recalibration_2026_round_12/` | 235,833 |
| `scripts/api_probe/raw-data/` | 233,978 |
| `data/cache/` | 223,421 |
| `scripts/api_probe/raw-data/github_jm1261_fantasy_f1_league/` | 179,684 |

## Artifact trace and recommendations

The following checks cover application/runtime (including Streamlit), tests, scripts, maintenance, documentation and reproducibility. “No direct runtime reader” is not evidence that a research artifact is dead. References were checked in tracked contents, script defaults, entrypoints and Git history, rather than only Python imports.

### Definitely keep

| Files/group | Runtime / deployment | Tests / scripts / maintenance | Documentation / reproducibility |
| --- | --- | --- | --- |
| `data/generated/historical_fantasy_scores_v3_recorded_2023_2026/` | Canonical history used by expected-value routes. | Historical integrity and Sprint tests; historical builder; maintenance `validate_pair` compares CSV/Parquet. | Coverage, discrepancy reports and manifest explain canonical definitions. Both representations pass exact normalized row comparison (2,491 rows). |
| `data/generated/sprint_ev_calibration/sprint_ev_2026_v1.json` | Stable production path contains **sprint_ev_2026_v2**. Filename is not the active version. | Runtime loader, recalibration and maintenance tests use it. | Active frozen parameter artifact: never discard because the filename says v1. |
| `data/generated/sprint_ev_calibration/archive/sprint_ev_2026_v1.json` | Not active. | Baseline lineage and old/new calibration comparison. | Tiny original fitted model; meaningful provenance, not a redundant copy of v2. |
| Calibration `sources/official_2026_round_12.json` and `sources/market_2026_round_13.json` | No direct app read required. | Raw import replay, inactive/replacement-driver coverage and maintenance fixtures use these snapshots; see `test_sprint_recalibration_update.py` and `test_update_sprint_model.py`. | Frozen official response and asset ledger are the evidence behind the accepted fit. Re-fetching a mutable API does not reproduce these bytes. |
| `data/generated/sprint_ev_shadow/` | Diagnostic runtime path loads its separate model. | Shadow/model tests and comparison script. | Explicitly distinct from production; not an obsolete active-model copy. |
| `data/cache/` reviewed 2023–2026 classification/schedule pairs and verified market seed | Supports validated offline/failure fallback in a clean Streamlit clone. | Race selection, current-history and data checks; feeds can refresh at runtime. | Provenance sidecars are required context. Old timestamps alone do not establish dead cache data. |
| `data/research/sprint_round_11/` | Not production live prices/history. | Fixed inputs in `compare_sprint_shadow_to_production.py` and historical/Sprint regression tests. | Freezes the earlier experiment as production history advances. Replacing it with current canonical data would alter that experiment. |
| `reports/2026_sprint_partial_pooling/` | Not loaded by app at startup. | `build_2026_sprint_final_candidate._load_prior_outputs` reads `asset_predictions.csv` and `model_comparison.csv`; recalibration defaults to this report; maintenance freezes the prior report bundle. | Sensitivity/selection evidence accompanies consumed comparison inputs. |
| `reports/sprint_recalibration_2026_round_12/` and final-candidate report | Runtime reads the promoted artifact, not report tables. | Builder and bundle-output tests; frozen accepted comparison/promotion evidence. | Keep complete accepted update report and normalized inputs. |
| `scripts/api_probe/raw-data/` with licenses/provenance | Builder inputs, not general application cache. | Historical builder consumes recorded totals; `validate_recovered_datasets.py` compares 2021 Kaggle/GitHub source data. | Original recorded scores, source attribution and cross-source validation remain useful even for seasons excluded from production. |
| Sanitised playerstats fixtures and `tests/` fixture data | Not production. | Schema, missing/partial-data and identity regressions. | No nonempty byte-identical fixture pair was found. |
| Screenshots, CI, dependencies and release docs | Real UI examples and clean-clone deployment instructions. | CI runs full tests, compilation and audit. | Portfolio presentation, not generated debug captures. |

### Safe to remove

**Removed:** `data/.gitkeep` and `data/cache/.gitkeep`, both empty. Git already retains these directories through their actual contents. No runtime/deployment, test, script, maintenance, documentation or fitting dependency on either placeholder was found. This removes clutter, not meaningful disk usage. No `.gitignore` change is needed: they are not regenerated outputs.

No substantive data/report/calibration file met the same unambiguous removal standard.

### Probably remove, but review first

| Candidate | Evidence checked | Why it remains for this release |
| --- | --- | --- |
| `f1fantasy/fantasy_prices.py` (legacy endpoint probe) | Old standalone `python -m` utility probing `mixapi.json` URLs. No tracked caller, documentation reference, test, maintenance or packaging entrypoint. Active data ingestion uses `fantasy_api.py`; later probe tooling performs broader discovery. Git history places this in the older implementation, predating the new release artifacts. | Likely obsolete diagnostic utility, but it remains manually executable. Removing an existing module is unnecessary to save roughly 2 KB without confirming that manual workflow is retired. |
| v2 historical `.parquet` alongside its CSV | Both contain 3,118 equal rows after canonical key sorting/missing normalization. No production reader selects the v2 archive; docs and classification manifest explicitly retain it. | Potentially redundant format, but preserves the original archive/schema package. Keep until archive retention policy is consciously narrowed; production v3 pairing must remain. |
| `scripts/api_probe/results.csv` | Human-readable companion to JSON discovery results; not a runtime or fitting input. `results.json` is explicitly read by `probe_playerstats_history.py` to discover asset IDs. Reports document the investigation. | CSV may be dispensable if the JSON-to-table export is verified and the research bundle is deliberately simplified. Do not remove its required JSON companion. |
| Old exploratory observation/LOSO report exports | Producers exist in `analyse_sprint_linear_regression.py`, `analyse_2026_sprint_bonus.py`, `analyse_2026_sprint_partial_pooling.py`, and `analyse_sprint_multiplier.py`. Not every exported table has a downstream code reader. | Regenerable in principle, but byte-identical replay of each historical report under frozen inputs, old cutoffs, market snapshot and package versions has not been established. They are documented model-selection evidence, not proven scratch output. |

The complete v2 dataset is superseded **for production**, but should not be deleted as a unit: it retains 2022 coverage absent from production v3, original reconstruction discrepancies and provenance. Current builder defaults and inputs are not proof of exact regeneration of that earlier archive.

### Large but intentional

- **1.27 MB official Sprint source snapshot:** the largest file is unique import/calibration evidence, including historical/inactive asset records.
- **1.21 MB v2 historical archive:** excluded 2022 observations and earlier provenance; retained research baseline.
- **0.97 MB current canonical bundle:** production data, provenance and CSV/Parquet integrity pair.
- **0.90 MB frozen round-11 research bundle:** stable experiment and regression inputs.
- **1.29 MB reports in total:** model-selection history, fitting inputs and accepted update evidence. The largest individual tables are the linear observation dataset (305 KB) and partial-pooling leave-one-Sprint-out table (199 KB).
- **150 KB of screenshots:** two real app views; negligible compared with data and code.

## Duplication, regeneration and older implementations

A SHA-256 scan of all 409 HEAD files found just one nonempty exact duplicate pair: `comparison_to_previous_models.csv` in `reports/2026_sprint_final_candidate/` and `reports/sprint_recalibration_2026_round_12/`. Both are deliberately retained in self-contained historical report bundles; the builder emits this filename and the final-candidate output test expects it. Deduplicating it would make the bundles less complete for negligible savings. Empty `.gitkeep` files also matched `f1fantasy/__init__.py`; the package initializer is required structure and was retained.

Both historical CSV/Parquet pairs have equal normalized rows. An initial comparison treating CSV numbers as strings is not a valid definition check; the successful comparison uses the maintenance workflow's key sorting and missing-value normalization with exact values. CSV/Parquet equality does not mean both production files are disposable: the maintained pair is explicitly validated by the workflow.

The retained Kaggle `dataset.zip` contains 46 files, all byte-equivalent to extracted files. The archive is only about 24 KB and its original download checksum is recorded in provenance. Extracted files serve validation; the ZIP preserves exact downloaded-source evidence. Re-zipping equivalent CSV files does not necessarily recreate the original archive checksum. Retain both rather than claiming deterministic bit-for-bit regeneration.

Canonical outputs and report tables have builders, but **re-running a builder against a live API is not reproduction of a frozen result**. Canonical source inputs, frozen market identity data, historical model versions, prior report inputs and validation receipts should be retained together. Current production history is canonical; accepted raw snapshots and prior reports are frozen evidence; cache seeds are deployment conveniences with validation; observation/LOSO exports are derived research outputs; only the empty placeholders were established as obsolete.

Git history shows the larger data/research/report bundles entering in `88e8b0e`, not a long accumulation of repeated historical commits. Earlier endpoint utilities remain in history, but active `ergast.py`/`fantasy_api.py` paths, `update_cache.py`, roster tooling, console entrypoints and shadow diagnostics have distinguishable roles. No broad source cleanup was attempted. Downloaded frontend code, transient inventories, the unversioned `reports/sprint_recalibration_candidate/`, virtual environments, personal lineups and caches excluded by policy remain local/ignored. They do not contribute to committed payload.

## Verification

- **Full suite: 710 passed, 2,385 warnings in 235.12 seconds.** Ran all `tests/` in a fresh HEAD export with only the two placeholder deletions applied; imports were asserted to resolve inside that export. The suite includes maintenance, canonical integrity, normal/Sprint EV, double-counting, optimiser constraints and Streamlit AppTest regressions. Warnings are existing PuLP deprecations. The local Python's native `readline` crash required the documented `sys.modules['readline'] = None` diagnostic workaround; no tests were skipped.
- **Sprint audit:** `python scripts/update_sprint_model.py --audit` returned `up_to_date`, active version `sprint_ev_2026_v2`, rounds `[2, 4, 5, 9, 12]`, no rounds to import and no files changed. This ran against the working repository. No fitter or activation was run.
- **Compilation:** `python -m compileall -q f1fantasy scripts tests streamlit_app.py` passed.
- **Runtime package imports:** `f1fantasy.app_core`, `f1fantasy.model`, `f1fantasy.optimize` and production calibration loader passed; loader returned `sprint_ev_2026_v2`.
- **Streamlit startup:** supported server launch returned HTTP **200 `ok`** at `/_stcore/health`; the temporary server was stopped. UI execution is covered separately by the full suite's AppTest tests. Direct bare `import streamlit_app` was also attempted and failed at `load_status.update()` without a Streamlit script context; it is not a passing standalone import check and no application change was made to accommodate it. Localhost binding required execution outside the filesystem/network sandbox.
- **CSV/Parquet:** both archived v2 (3,118 rows) and active v3 (2,491 rows) compared exactly after canonical key sorting and missing normalization; active `validate_pair` passed.
- **Final hygiene:** `git diff --check` passed. A scan of existing tracked text plus this report found no personal filesystem paths, private-key blocks or common GitHub/AWS/Slack credential-token patterns. This pattern scan is not a proof that arbitrary secrets cannot exist; it supplements the prior release audit. Private lineup, environment, frontend downloads and scratch report remain ignored.
- **Byte preservation:** SHA-256 comparison with the starting working tree found only the two intended deletions. Canonical observations, active calibration and all four pre-existing cache modifications remained byte-for-byte unchanged. Active artifact SHA-256: `25bcb8b647faec2ad17fd54568f94b0eefa6f6f11593f76cd6582e098a598386`.
- **Git:** branch `fix/live-data-caching`, three commits ahead and zero behind the existing `origin/main` reference; no remote fetch was necessary for this size audit. Nothing was staged or committed. Final audit changes are two placeholder deletions and this new report, alongside the four pre-existing cache changes.

## Conclusion

Approximately 9.2 MB is reasonable for this application's code, tests, canonical history, frozen calibration evidence and offline seeds. The largest committed file is approximately 1.27 MB. There is no size-related reason to rewrite history, adopt Git LFS, remove unique evidence, or delay the public release. Review and commit this small audit cleanup separately; review the four pre-existing cache refresh changes independently before including them in a push.
