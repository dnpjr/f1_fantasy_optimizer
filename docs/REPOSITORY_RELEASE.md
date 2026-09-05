# Release and deployment guide

## Streamlit Community Cloud

Deploy the source repository with `streamlit_app.py` as the entrypoint, root `requirements.txt`, and **Python 3.11**. Check the app's actual connected branch in Community Cloud before merging a release; it cannot be inferred from local Git configuration.

Python 3.11 is required by `datetime.UTC`. The declared Streamlit minimum covers the current segmented controls, download behaviour and dataframe sizing APIs. Parquet support is an explicit dependency rather than an accidental transitive installation. PuLP is kept below 4 because the optimiser uses its existing bundled-CBC interface. Requirements use bounded compatibility ranges; record the resolved environment when diagnosing a future rebuild.

Runtime data and calibration files live outside the Python package. Deploy from the repository root, not from the wheel alone. Include every reviewed file described in [the data policy](../data/README.md); an editable installation does not supply missing data. No user credentials or Streamlit secrets are needed for default public-feed access.

Public feed requests can fail or change. Reviewed cache seeds are included to support validated fallback, but they are dated snapshots. Successful HTTP startup alone does not establish that a feed is fresh: inspect source status, event identity, prices and completed-history coverage in the running UI. Cloud filesystem refreshes are ephemeral; do not use the deployed filesystem as the canonical calibration archive.

The results client contains an explicitly public browser API key, also present in Formula1.com's public results HTML under `NEXT_PUBLIC_GLOBAL_EVENTTRACKER_APIKEY` (verified during this release audit). It is not a personal account credential. `FORMULA1_RESULTS_API_KEY` can override it for rotation. Private environment values, cookies and account tokens must never be committed.

## Before a release

From an activated Python 3.11 environment at the repository root:

```sh
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest tests -q
python -m compileall -q f1fantasy scripts tests streamlit_app.py
python scripts/update_sprint_model.py --audit
git diff --check
git status --short
```

The full suite includes Sprint maintenance, canonical data integrity, normal/Sprint expected-points routes, no-double-counting regressions, optimiser constraints and Streamlit AppTest interactions. An audit reports missing future data without importing or activating it; resolve unexpected changes before release. Production calibration is identified by the JSON's `model_version` field, not the stable filename ending in `v1`.

A local Conda base Python used during the release audit had a broken native `readline` extension. If pytest crashes before collection in that environment, recreate the virtual environment with a healthy Python installation, or use the same diagnostic workaround used for verification:

```sh
python -c "import sys; sys.modules['readline'] = None; import pytest; raise SystemExit(pytest.main(['tests', '-q']))"
```

This disables terminal line editing only; it does not skip tests. It is not needed for a normal Python installation.

After committing reviewed changes locally, verify the exact committed snapshot separately:

```sh
git worktree add --detach ../f1-fantasy-release-check HEAD
cd ../f1-fantasy-release-check
python -m venv .venv
# Activate this new environment using the platform command in the README.
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/update_sprint_model.py --audit
python -m streamlit run streamlit_app.py
```

The new checkout must work without copying personal files, credentials or an old virtual environment. The GitHub Actions workflow repeats the suite, compilation and offline Sprint audit on Ubuntu/Python 3.11 after publication.

## Reviewing and publishing

No push or commit is part of the maintenance/audit tools. Keep production model/data changes and their required artifacts together. A suitable release split is:

1. `Update validated data pipeline, modelling and decision-support UI` — runtime package/UI, canonical/seed data, historical builders, recalibration fitter and research evidence, plus their regression tests, frozen fixtures, personal-config migration and dependency declarations.
2. `Add staged Sprint calibration maintenance workflow` — `scripts/update_sprint_model.py`, `tests/test_update_sprint_model.py` and `docs/SPRINT_MODEL_MAINTENANCE.md`. The shared historical writer belongs with the ingestion code in commit 1.
3. `Prepare portfolio documentation and reproducible release checks` — README/docs/images, artifact policy, `.gitignore`, OS-artifact deletion and CI.

Some files contain work from more than one group. Use patch staging where useful; do not split a required dataset/artifact dependency just to achieve exactly three commits. Preserve existing published commits.

After reviewing and creating the local commits, publish a review branch first:

```sh
git fetch origin
git log --oneline --left-right origin/main...HEAD
git diff --stat origin/main...HEAD
git status --short
git push --dry-run origin HEAD:refs/heads/codex/portfolio-release
git push origin HEAD:refs/heads/codex/portfolio-release
```

If the destination branch already exists, inspect it first. A normal push refuses a conflicting history; never add `--force`. Open a GitHub pull request from `codex/portfolio-release` to the deployed branch (normally `main`), inspect the **Files changed** tab and CI, then merge after review. If `main` advanced, merge `origin/main` locally and rerun checks before pushing the updated review branch. There is no reason to rewrite remote history.

## Post-push checks

- GitHub renders the Mermaid diagram, relative documentation links and screenshots; no OS files, personal lineup, tokens or downloaded frontend bundles appear in the PR.
- CI passes on the actual committed files. Confirm the canonical CSV/Parquet, active v2 calibration, shadow JSON and cache seeds are present.
- Community Cloud uses the intended branch, Python version and entrypoint; inspect build/runtime logs after merging.
- Load the app in a fresh browser session. Verify the current event/feed and source freshness, run the optimiser, and inspect Market and Team/transfer views on desktop and mobile.
- Check normal-weekend projections and, when relevant, Sprint breakdowns. Confirm active calibration is still the reviewed version; no fit or activation should occur during deployment.

## Screenshots

`docs/images/optimiser.png` and `docs/images/market-thresholds.png` were captured from the unmodified running application on 2026-09-05, with the public feed for Italian GP R13 and the current v2 model. They are illustrative model outputs, not backtest results. The missing-history warning is preserved. No UI data or imagery was fabricated. The main README shows the optimiser and keeps the second view collapsed.

A useful later third image would show **Team → current squad and transfer deltas**, using a deliberately selected example squad rather than a private saved team. Save it as `docs/images/transfers.png` and keep it in the same optional screenshot section.
