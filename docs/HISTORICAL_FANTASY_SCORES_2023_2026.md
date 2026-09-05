# Historical F1 Fantasy scores, 2023–2026

## Production scope

The canonical production dataset starts in 2023 and uses the stable key:

```text
season + round + entity_type + canonical_entity_id
```

The supported seasons are 2023, 2024, 2025, and 2026. Neither 2021 nor 2022
is read, normalised, validated, emitted, or exposed by the production builder.
This is a consistency decision, not a finding that the recorded 2022 totals are
invalid: the archived 2022 source lacks exact event prices and detailed
components, whereas 2023 onward provides one common period for points- and
price-based analysis. The downloaded 2022 files and the v2 generated output
remain available locally for provenance, audit, comparison, and future research.

The production data version is:

```text
historical_fantasy_scores_v3_recorded_2023_2026
```

## Sources and exact field coverage

| Season | Authority | Exact retained fields | Unavailable detail |
|---:|---|---|---|
| 2023 | Third-party recorded, MIT-licensed | Driver/constructor event totals and event prices | Q/SQ/S/R component split |
| 2024 | Third-party recorded, MIT-licensed | Driver/constructor event totals and event prices | Q/SQ/S/R component split |
| 2025 | Third-party recorded, MIT-licensed | Driver/constructor event totals and event prices | Q/SQ/S/R component split |
| 2026 | Official Formula 1 Fantasy playerstats | Available totals, prices, and session components | Explicit per-asset feed gaps remain missing |

The retained 2023–2025 source is
[`jm1261/Fantasy-F1-League`](https://github.com/jm1261/Fantasy-F1-League),
licensed under MIT. Its missing component values remain null; they are not
reconstructed and are never converted to zero. Replacement drivers and team
changes are represented by explicit season/round participation mappings.

Official 2026 rows use:

```text
https://fantasy.formula1.com/feeds/popup/playerstats_{player_id}.json
```

Historical query parameters are not used because the investigation established
that `season`, `season_name`, and `year` still return the current 2026 payload.

## Precedence and reconstruction policy

The general resolver retains this precedence for diagnostics and future cases:

1. `official_recorded`
2. `third_party_recorded`
3. explicitly diagnosed `reconstructed`
4. `missing`

The 2023–2026 production build accepts no reconstructed rows. Recorded driver
and constructor totals replace ordinary-classification proxies before form,
recency weighting, volatility, expected-points targets, price efficiency, and
points-per-million analysis. Missing component detail does not prevent an exact
weekend total from being used.

## Build and generated outputs

The allow-listed source downloader may retain the archived 2022 research files,
but the production normaliser traverses only 2023–2025 event files:

```bash
.venv/bin/python scripts/api_probe/download_licensed_datasets.py --production-scores-only
```

Rebuild using an existing valid v3 official cache:

```bash
.venv/bin/python scripts/build_historical_fantasy_scores.py
```

Refresh from the current official 2026 feed:

```bash
.venv/bin/python scripts/build_historical_fantasy_scores.py --fetch-official --feed-round 12
```

The feed number is explicit so a transient probe failure cannot silently select
an older feed. Outputs are written to:

```text
data/generated/historical_fantasy_scores_v3_recorded_2023_2026/
```

The directory contains canonical CSV and Parquet files, a provenance manifest,
coverage CSV/JSON, source-summary CSV/JSON, and approximation-comparison
diagnostics. The 2026 row count is deliberately dynamic so later official
gamedays can be added without changing the schema or completed-season checks.

## Cache and model migration

The v3 data version participates in successful live-snapshot and derived-model
identities. The canonical loader also rejects a mismatched embedded data
version. Consequently, a v2 snapshot or model built from the 2022–2026 range
cannot masquerade as v3 state. Ordinary motorsport caches and unrelated user
state are not deleted, and the old v2 generated directory is not overwritten.

## Licensing and redistribution

Each 2023–2025 row retains the MIT source reference and licence. Official 2026
rows retain the public endpoint reference and are subject to Formula 1 terms.
Restricted F1 Fantasy Tools responses are not a production dependency and are
not redistributed.
