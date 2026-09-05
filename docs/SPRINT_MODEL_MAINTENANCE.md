# Sprint-model maintenance

Use `scripts/update_sprint_model.py` from the repository root. It wraps the existing official snapshot importer, canonical dataset writer, Sprint fitter and atomic calibration promotion. It does not change the statistical model or run from the optimiser UI.

The examples below use the macOS/Linux virtual-environment path. On Windows, activate the environment as shown in the root README and substitute `python` for `.venv/bin/python`.

Production is currently `sprint_ev_2026_v2`. Its stable runtime filename still ends in `sprint_ev_2026_v1.json`; the JSON's `model_version` is authoritative. Do not rename that runtime path or edit coefficients by hand.

## Next time: the exact procedure

1. Wait until the Sprint **and Grand Prix** are complete. Use the app's existing data refresh to update the local schedule and verified Fantasy market. The wrapper reads `data/cache/schedule_2026.csv` and `data/cache/verified_fantasy_market.json`; `--fetch` refreshes playerstats only, not those two inputs. The current fitter conservatively waits until the event date is before today's UTC date.
2. Audit the prospective update:

   ```sh
   .venv/bin/python scripts/update_sprint_model.py --audit --fetch
   ```

   This reads public playerstats if a past Sprint is missing, validates prospective inputs in temporary storage, and prints included/missing rounds, available/missing observations and proposed actions. It saves no candidate, canonical data or calibration. `--dry-run` is an alias for `--audit`.
3. Generate a candidate:

   ```sh
   .venv/bin/python scripts/update_sprint_model.py --fetch
   ```

   Copy the printed `bundle` path. It will look like `reports/sprint_updates/sprint_ev_2026_v3_<input-fingerprint>` for the next version. The version, round selection and fingerprint are computed automatically. There is no round argument to maintain.
4. Inspect that directory's `UPDATE_REPORT.md`, `parameter_comparison.csv`, `comparison.json`, and `fit/sprint_observations.csv`. Run the reproducibility check and regression tests using the printed path:

   ```sh
   .venv/bin/python scripts/update_sprint_model.py --check reports/sprint_updates/sprint_ev_2026_v3_<input-fingerprint>
   ```

   Replace the example path with the exact printed path, including its fingerprint. This command refits the frozen inputs, verifies the parameters, histories and input hashes, then runs the maintenance regression suite. It writes `TEST_RESULTS.txt` and `checks.json`. A failed test prevents activation; no `--skip-tests` option exists.
5. Explicitly activate **after review**. Stop the app during this short filesystem update:

   ```sh
   .venv/bin/python scripts/update_sprint_model.py --activate reports/sprint_updates/sprint_ev_2026_v3_<input-fingerprint>
   ```

   If the report has flags, activation stops. After assessing them, acknowledge them explicitly:

   ```sh
   .venv/bin/python scripts/update_sprint_model.py --activate reports/sprint_updates/sprint_ev_2026_v3_<input-fingerprint> --accept-anomalies
   ```

   Missing components and absent historical participants are reported, not converted to zero. Parameter changes above 25% of `max(abs(old), 1)` are review flags, not automatic model-selection rules. A flag acknowledgement cannot override invalid data, changed files, a failed check or a stale baseline.
6. Reload the app. Inspect and commit the **whole bundle**, updated `data/generated/historical_fantasy_scores_v3_recorded_2023_2026/` directory, active calibration and its new archive together. Do not commit only the parameter JSON. The manifest and `activation.json` identify what was installed. Check the diff for unrelated pre-existing work before committing.

A typical future run discovers the next completed Sprint in the updated catalogue, imports its data and any missing intervening normal weekends, produces a candidate with a new version, and leaves production unchanged while you review/test it. Activation installs the staged dataset and the candidate together. Rerunning after activation reports `up_to_date`; rerunning candidate generation with identical inputs reuses the existing bundle without rewriting it.

## Offline/local inputs

Without `--fetch`, both audit and generation are offline:

```sh
.venv/bin/python scripts/update_sprint_model.py --audit
.venv/bin/python scripts/update_sprint_model.py
```

The default source directory is `data/generated/sprint_ev_calibration/sources/`. Supply a different directory with `--sources /path/to/snapshots`. Files named `official_*.json` use the existing retained-snapshot format: `season`, `round`, full `players`/`teams` identity metadata, `expected_played_counts`, and `payloads` containing `player_id` and official `payload` objects. The prior Dutch snapshot is a schema example, not a hard-coded discovery rule.

If the app or another authorised import already put a new Sprint into canonical history, this command detects that it is absent from the **active fit**, generates a candidate without reimporting the weekend, and reports the difference correctly.

The wrapper needs all completed normal-form history through the new Sprint. It fetches/sources missing intervening normal weekends as well. If any required source is unavailable, it reports `awaiting_sources`, exits with code 2, and saves no candidate. Calendar dates alone are not treated as proof of a completed, scored event. Partial results, unsupported identities or a changed source schema require investigation or another refresh, not invented points.

`--canonical`, `--active`, `--schedule`, `--market`, `--sources` and `--output` allow isolated runs. The schedule must retain the fitter's `schedule_2026.csv` filename. Activation uses the destinations recorded in the bundle; paths are shown in its manifest. Move/copy the entire bundle for review, but do not edit its files or activate an untrusted bundle.

## What is preserved and checked

- Discovery distinguishes canonical Sprint coverage from active-calibration Sprint coverage using season/round keys and the canonical schedule. It does not hard-code Dutch or the next Sprint round.
- Raw ingestion reuses `parse_player_race_points` and `normalise_official_playerstats` through the existing snapshot importer. Fetching covers the full cached roster, inactive assets and historical source IDs. Replacement Fantasy IDs still resolve to human canonical identities.
- Duplicate raw assets, gamedays and sessions are rejected. Raw session season/round/name must agree with the selected scheduled event. Missing GP results/incomplete rosters are rejected by the existing completed-event checks.
- Existing canonical events are never silently revised or supplemented. A conflicting replay is rejected. Missing historical components remain missing. Aggregate Sprint-component coverage retains the established fitting threshold and methodology.
- Canonical duplicates, event gaps, future dates, official-source requirements and component arithmetic are checked before fitting. CSV and Parquet are compared; if the current dataset has Parquet, failure to generate its staged counterpart stops the update.
- The production baseline normaliser is checked against the fitter's normal-equivalent history for both drivers and constructors. Sprint targets contain only the existing labelled Sprint/Sprint Qualifying components; GP points are not added to the target. Missing splits remain omitted.
- Candidate generation writes only an isolated, versioned bundle after successful validation/fitting. Canonical data and active parameters remain unchanged, including on a fit failure. No-op/audit modes save no output files.
- The bundle freezes before/proposed datasets, source snapshots, active baseline, schedule, market and prior research comparison inputs. Hashes bind every retained file and the relevant working-tree code. Python, NumPy and pandas versions are recorded. Reproduction checks ignore the newly generated timestamp but require identical fitted parameters and histories.
- Activation requires a passing receipt for those exact files/code, unchanged production dataset/calibration, and explicit flag acknowledgement when needed. A lock prevents concurrent activations. The existing promotion routine archives the prior calibration and atomically replaces the active JSON.
- Copied leave-one-Sprint-out research metrics are labelled historical. This workflow refits the established model; it does not rerun model selection or change the 2026 scoring assumptions.

## Tests and recovery

`--check BUNDLE` is the one follow-up test command. It runs `tests/test_update_sprint_model.py`, `tests/test_sprint_ev_components.py`, `tests/test_sprint_ev_shadow.py`, `tests/test_sprint_shadow.py`, and `tests/test_race_selection.py`, plus direct validation/refitting of the actual candidate. These tests are isolated from production and use frozen fixtures, so the next real update need not match hard-coded v2 coefficients. The wrapper avoids the local broken `readline` extension when starting pytest; no tests are skipped for this workaround.

A code change requires a newly generated candidate and check. Repeated generation under changed inputs/code receives a different fingerprint, preserving the earlier bundle for audit.

Each installed file is replaced atomically, but a multi-file dataset is not one filesystem transaction. Normal exceptions trigger restoration from the bundle's `before_dataset/` and `before_active.json`. A process or machine crash leaves `activation_pending.json` and/or `.sprint-maintenance.lock`; activation then stops rather than guessing. Recovery is deliberately manual:

1. Stop the app and ensure no maintenance process is running. Read the bundle's manifest to confirm the destination paths.
2. Restore every file in `before_dataset/` to the canonical destination directory. Remove any dataset output listed under `dataset/` that was absent from `before_dataset/`. Restore `before_active.json` to the active destination.
3. Verify these files against `before_dataset_sha256` and `before_active_sha256` in the manifest. Retain the bundle as the audit record. The archive may already contain the old calibration; identical archive content is safe.
4. Remove that bundle's `activation_pending.json` and the active directory's `.sprint-maintenance.lock` only after confirming restoration. Rerun `--check`, then the explicit activation command.

No unattended activation, deployment, commit or scoring-season migration is performed. This workflow supports future **2026** Sprint weekends using the current 2026 fitter. A new season or scoring-rule change still requires deliberate model/source support; it must not silently inherit these assumptions.

## Initial verification (2026-09-05)

Against the current repository, audit and update report `up_to_date` with canonical/active Sprint rounds `[2, 4, 5, 9, 12]`. Active `sprint_ev_2026_v2`, canonical CSV/Parquet and existing dataset reports remain byte-for-byte unchanged. The combined maintenance and existing Sprint regression run passed 177 tests during implementation.
