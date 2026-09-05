# Historical Fantasy scores: implementation note

This note records the production path inspected before implementation. The new
dataset intentionally covers only 2023–2026; neither 2021 nor the archived 2022
research source is read, normalised, validated, or exposed in production.

## Existing paths

- `f1fantasy/model.py::compute_weekend_points()` reconstructs driver qualifying,
  Sprint, race, and weekend points from motorsport classifications. Its output
  feeds historical form, recency weighting, volatility, expected-points
  modelling, and the Sprint-aware shadow diagnostics.
- `f1fantasy/model.py::_constructor_round_points()` aggregates those reconstructed
  driver rows and applies constructor bonuses and penalties, so historical
  constructor totals are reconstructed as well.
- `f1fantasy/app_core.py::derive_model_data()` currently sends these reconstructed
  rows into the driver and constructor forecast paths before blending current
  official playerstats observations.
- `f1fantasy/recommend.py` has a second direct historical-model entry point.
- `f1fantasy/player_stats.py` parses current official playerstats totals,
  components, and prices. It is the authoritative 2026 source.
- `f1fantasy/ergast.py` persists ordinary results, qualifying, Sprint, and
  schedule CSVs. These caches do not contain Fantasy totals, but derived model
  identities currently have no recorded-score data-version marker.
- `f1fantasy/price_efficiency.py` and the race-selection helpers already preserve
  missing observations. Current price-efficiency UI calculations use current
  official playerstats and are outside this migration.

## Proposed changes

1. Add `f1fantasy/historical_scores.py` as the Streamlit-independent canonical
   normalisation, validation, source-precedence, overlay, and reporting layer.
2. Keep the existing cautious licensed-source downloader and archived 2022
   provenance files, while the production build reads only the MIT-licensed
   2023–2025 event files. It will not retain or
   redistribute F1 Fantasy Tools responses. The 2023–2025 MIT files record exact
   totals and prices but not reliable scoring-component splits; those component
   fields will remain null.
3. Add `scripts/build_historical_fantasy_scores.py` to build versioned CSV and,
   when a Parquet engine is installed, Parquet output plus provenance, coverage,
   and approximation-comparison reports. The builder will never traverse the
   retained 2021 folders.
4. Add explicit season-aware source-to-canonical entity maps. Source labels are
   never joined to production rows by surname, abbreviation, or display name
   alone.
5. Load the generated 2023–2025 recorded dataset through `app_core`, append
   official 2026 playerstats records at runtime, and overlay recorded totals on
   reconstructed structural rows before historical form is calculated. Exact
   driver and constructor totals will have priority; reconstruction will remain
   available only as a labelled fallback and will be diagnosed for covered
   events.
6. Add the data version
   `historical_fantasy_scores_v3_recorded_2023_2026` to snapshot/model identity
   and diagnostics so an older derived snapshot cannot masquerade as recorded
   data. Ordinary motorsport caches and unrelated user caches will not be
   deleted.
7. Add focused unit and integration tests for schema, pre-2023 exclusion,
   uniqueness, null preservation, replacements/team changes, precedence,
   official 2026 parsing, and production overlay behaviour.

The migration changes historical target provenance, not the forecast formula,
optimiser objective, user controls, or application layout.
