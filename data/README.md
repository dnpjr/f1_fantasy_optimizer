# Data and reproducibility

Run the application and maintenance commands from the repository root. Required data is included in the source checkout; the Python wheel alone is not a self-contained deployment.

| Directory/file | Purpose and publication policy |
| --- | --- |
| `generated/historical_fantasy_scores_v3_recorded_2023_2026/` | **Production canonical history.** Keep CSV, Parquet, coverage reports and provenance together. Missing components remain missing. |
| `generated/sprint_ev_calibration/` | **Production calibration and reproducibility.** Keep active JSON, archives and retained public source snapshots. The stable `sprint_ev_2026_v1.json` filename currently contains version `sprint_ev_2026_v2`; inspect `model_version`, not the filename. |
| `generated/sprint_ev_shadow/` | Frozen diagnostic model loaded by runtime/tests; retain separately from active calibration. |
| `cache/*_2023.csv` through `*_2026.csv` | Reviewed schedule/classification seeds, with `.meta.json` provenance. Intentionally included despite the general cache ignore rule. They can be stale or partial; metadata and normal validation still apply. |
| `cache/verified_fantasy_market.json` | Reviewed public market fallback seed including the historical asset ledger. Not personal account data; never assume this snapshot is current without validation. |
| `research/sprint_round_11/` | Fixed regression experiment. Derived market fixture is explicitly distinguished from a raw response. Never used as production live prices. |
| `generated/historical_fantasy_scores_v2_recorded_2022_2026/` | Superseded research archive, retained for provenance; **not production input**. Excluded 2022 coverage lacks comparable event-price/component information. |
| `current_team.local.json` | Private optional user lineup; ignored and not deployed. |

The canonical 2023–2025 recorded totals come from the MIT-licensed `jm1261/Fantasy-F1-League` source retained under `scripts/api_probe/raw-data/` with attribution. The 2026 records and retained market/playerstats snapshots come from public official F1 Fantasy responses. Preserve each source's licensing and provenance; this project's MIT licence covers its code, not Formula 1 imagery/trademarks or an implied relicensing of official data.

The small archived datasets and modelling reports are intentionally retained rather than deleting evidence to reduce size. Downloaded third-party frontend JavaScript/sourcemaps and transient endpoint inventories are ignored. They are not calibration inputs and are not needed for the app or tests.

For later Sprint updates, use [the staged maintenance workflow](../docs/SPRINT_MODEL_MAINTENANCE.md). Review changed cache seeds and their sidecars together before committing; never replace a complete reviewed seed with a failed/partial refresh merely to make the working tree clean.
