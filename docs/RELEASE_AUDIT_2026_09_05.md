# Portfolio release audit — 2026-09-05

## Public version versus local version

The live remote `refs/heads/main` was verified with `git ls-remote` at **86ff294af5029b6aa3fabdf0f975992e2dbfd356** (`Bug fix`). Local `main`, `fix/live-data-caching` (current branch), and `fix/upload-budget-state` all point to that commit. There are **0 commits ahead and 0 behind** public main. All newer functionality remains working-tree changes; no staging, commits, pushes, remote-history edits or calibration activation were performed during this audit.

The public version already had a Streamlit team optimiser, price-change/transfer views and chip support. This release develops those into a more complete data/modelling system:

| Area | Work becoming public |
| --- | --- |
| Current Fantasy ingestion | Verified market resolution, same-feed refresh handling, independent source domains, atomic refresh/fallback and fuller asset ledgers |
| Historical data | Canonical recorded 2023–2026 Fantasy history, season/round identities, explicit source/missingness metadata and inactive/replacement-driver history |
| Current prices and efficiency | Separation of accepted live prices from event-price history; completed-race selection and historical points-per-million views |
| Expected points | Recency and season weighting, current-only/all-supported history modes, component routes and complete-event guards |
| Sprint EV | Separate driver/constructor calibration, normal-baseline Sprint removal, production v2, and a separately frozen diagnostic shadow |
| Calibration maintenance | Automatic missing-weekend discovery, canonical importer reuse, staged/versioned candidates, input hashing, comparison reports, validation receipts and explicit activation |
| Price growth | Probabilistic estimates, current-price validation, missing/replacement-asset handling and separation from chip multipliers |
| Optimisation and transfers | Shared EV/price objectives, constrained ranked results, locks/exclusions, current holdings, transfer/budget state and robust uploaded-team handling |
| Live sessions | Official/public session ingestion, identity/completion/freshness checks, guarded optional EV emphasis (zero by default), separate diagnostics |
| Interface | Mobile/desktop views, persistent optimisation results, market/efficiency and current-team controls, exports and source diagnostics |
| Developer tooling | 50 test modules / 710 tests, explicit development dependencies, portable AppTest paths and Ubuntu/Python 3.11 CI |

## Changes made during this audit

- Removed tracked `.DS_Store`; the ignore rule prevents recurrence.
- Removed the public default `data/current_team.json`. Preserved its exact contents locally at ignored `data/current_team.local.json`; app/legacy CLI now read that optional local path. Retained the previously public lineup as an explicitly labelled deterministic research example.
- Included the reviewed 2023–2026 schedule/classification CSV seeds and public verified-market seed, previously ignored despite runtime/test/maintenance use. Retained their sidecar provenance and all canonical/calibration artifacts.
- Preserved scientifically useful archived v2 data, licensed original datasets, accepted reports and source snapshots. Documented which are production, frozen research or archival inputs.
- Excluded downloaded frontend JavaScript/sourcemaps (~11 MB), transient source inventories (including unrelated public participant/team names), and superseded unversioned calibration scratch output. These remain local; no scientific source data was deleted.
- Removed machine-specific paths from two retained historical reports and one ignored scratch promotion review.
- Added fixed round-11 research inputs and connected older regression experiments to them. The derived market fixture clearly identifies its source in the archived report; its unavailable original verification timestamp remains null. Later canonical updates no longer silently change these historical test scenarios.
- Made UI test entrypoint paths explicit after a clean install exposed a newer Streamlit path-resolution change.
- Corrected Python support to 3.11+, raised dependency minimums for APIs already used, bounded major versions, declared Pillow/Parquet explicitly, added development dependencies and verified editable installation.
- Rewrote the README around user tasks and technical architecture; added Mermaid, exact setup commands, limits, source attribution and links to maintenance/deployment guides.
- Added real optimiser and market screenshots; no UI redesign or fabricated output.
- Added a GitHub Actions suite and documented artifact/deployment/release policy.

Audit-specific file groups: `.gitignore`, `README.md`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.github/workflows/tests.yml`; `streamlit_app.py` and `f1fantasy/recommend.py` (local lineup path only); `scripts/compare_sprint_shadow_to_production.py`; four test modules (`test_2026_sprint_bonus_analysis`, `test_current_season_history_regression`, `test_sprint_linear_regression`, `test_streamlit_ui`); `data/research/sprint_round_11/`; documentation/index/image files under `docs/`, `data/README.md`, `reports/README.md`; two retained report path corrections. The much larger runtime/model diff predates this audit.

## Security, privacy and artifact review

The candidate publication scan found no private-key/token patterns, personal filesystem paths or email addresses. The only embedded API-key-like constant is the existing **public browser results key**: its presence and `NEXT_PUBLIC_GLOBAL_EVENTTRACKER_APIKEY` variable were verified directly in Formula1.com's public results HTML (HTTP 200). It is retained, with the existing environment override for rotation; it is not an account credential.

No private cookies, environment credentials or account tokens are required by default. The Git ignore policy excludes secrets, local lineups, virtual environments, caches outside the reviewed seed allowlist, logs, build outputs and downloaded frontend bundles. Data licensing and third-party trademarks are distinguished from the project's code licence.

The proposed publication is about **9.1 MB** before Git compression. Its largest files are a ~1.27 MB retained public playerstats snapshot and ~1.14 MB superseded historical CSV. Neither needs Git LFS. Archived/source duplicates are intentional provenance, not accidental runtime alternatives.

## Verification results

Initial full run: **677 passed, 25 failed, 8 errors**. The failures traced to pre-Dutch research/round-11 fixtures reading the current dataset/feed. Fixed-input targeted verification: **53 passed**.

Final full suite:

| Environment | Result |
| --- | --- |
| Existing workspace environment (Streamlit 1.60.0) | 710 passed in 238.92 s |
| Workspace with fresh declared dependencies (Streamlit 1.63.0) | 710 passed in 235.96 s |
| Isolated release copy containing only Git-eligible files, fresh dependencies | **710 passed in 234.11 s** |

All full runs include the 27 Sprint-maintenance tests, current recalibration/data-integrity regressions, normal/Sprint route and no-double-counting checks, optimiser tests and UI interactions. Final runs emitted 2,385 PuLP deprecation warnings, not failed checks. PuLP remains bounded below 4; migration of that solver API is future maintenance, outside this release.

Additional checks passed:

- `python -m pip install -r requirements-dev.txt`, `python -m pip check` and `python -m pip install -e '.[dev]'` in an isolated environment; both declared console entrypoints resolve.
- Compilation of `f1fantasy`, `scripts`, `tests` and `streamlit_app.py`; import of all 22 package modules.
- Actual clean-copy Streamlit startup and browser rendering with current public feed 13; optimiser generated ten teams; Market thresholds rendered with genuine missing-history diagnostics. Test server stopped afterward.
- Offline `python scripts/update_sprint_model.py --audit`: **up_to_date**, active **sprint_ev_2026_v2**, canonical/active Sprint rounds **[2, 4, 5, 9, 12]**, no missing/new observations, no fit or activation.
- Original data/cache/calibration inventory: **63/63 files byte-identical** after all workspace checks. CSV/Parquet agreement is covered by the maintenance/data tests.
- Candidate relative Markdown links and `git diff --check`; candidate secret/path/artifact scan.

The first clean-install run found seven UI-test path failures under newer Streamlit (703 passed). These were fixed by resolving the actual app path from the test file; the final clean-copy full run above passed. Test-server refreshes were confined to the temporary copy, and its reviewed seeds were restored before the final isolated suite.

The local Conda interpreter's broken native `readline` extension required the documented pytest line-editing workaround. No tests were skipped. macOS sandbox CPU-info warnings from Arrow were nonfatal. Linux CI has been configured but cannot run remotely until the user publishes the branch; Community Cloud deployment itself was not changed.

Resolved direct dependencies: pandas 3.0.5, NumPy 2.4.6, requests 2.34.2, PuLP 3.3.2, Streamlit 1.63.0, Pillow 12.3.0, PyArrow 24.0.0, pytest 9.1.1. Requirements remain bounded ranges rather than a complete transitive lock.

Preserved SHA-256 identifiers:

```text
active calibration: 25bcb8b647faec2ad17fd54568f94b0eefa6f6f11593f76cd6582e098a598386
canonical CSV:      9a80fa4d8241f7fb8985a04c0baab6c1f1739f582eee63be97bb7c93304d488a
canonical Parquet:  62edd0564d99d111acfadd4397466436da031180223839cb6fd2daff06bd7cbb
```

## Deployment and remaining manual steps

Select **Python 3.11**, root `requirements.txt` and `streamlit_app.py` in Community Cloud, and verify its connected branch before merging. Publish code **and** required data, reports and calibration files. The source checkout, not the wheel alone, is the deployment unit. Refresh data and inspect source status after rebuild; cached seeds can be partial/stale and the UI correctly warns about missing histories.

The legacy CLI is retained but does not share the complete Streamlit Sprint/live-session pipeline. Use Streamlit for production recommendations. No statistical methodology, active calibration or unrelated optimiser functionality was changed here.

All work is still uncommitted. Intentionally local/excluded: `data/current_team.local.json`, `.venv/`, Python/test/build caches, older non-reviewed cache seasons, ignored frontend bundles and discovery inventories, `reports/sprint_recalibration_candidate/`, and transient test/install logs kept outside the repository. Necessary canonical data and versioned reports remain eligible for publication.

Use the three proposed commit groups and safe review-branch push commands in [the release guide](REPOSITORY_RELEASE.md#reviewing-and-publishing). Review patch staging where files contain several areas of work. Do not force-push or commit only parameter JSON. After publication, check GitHub CI/README rendering and the live app's event, prices, optimiser, Team/transfer and normal/Sprint diagnostics.
