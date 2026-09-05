# Historical F1 Fantasy score-source investigation

Investigation continued: 2026-08-05
Repository branch: `fix/live-data-caching`
Authentication used: none
Methods: public GET requests only

## Executive conclusion

Historical driver and constructor Fantasy scores were found for every requested season, but they are not all equally authoritative.

| Season | Classification | Evidence |
|---:|---|---|
| 2021 | **C — Partially recovered** | Two licensed third-party records cover all 22 races and both asset types, but they disagree on 23/440 driver and 14/220 constructor observations. Prices are available; full official scoring components are not. |
| 2022 | **B — Exact third-party recorded scores recovered** | MIT-licensed individual race totals for 20 drivers and 10 constructors across all 22 races. The individual values reproduce the stored cumulative totals exactly. A separate GPL-3.0 repository supplies price snapshots. No official payload or component breakdown was recovered. |
| 2023 | **B — Exact third-party recorded scores recovered** | The public F1 Fantasy Tools Statistics API returned 22 populated race results for drivers and constructors, including Q/S/R score components and prices. |
| 2024 | **B — Exact third-party recorded scores recovered** | The same public API returned all 24 populated race results, including Q/S/R score components and prices. |
| 2025 | **B — Exact third-party recorded scores recovered** | The same public API returned all 24 populated race results, including Q/S/R score components and prices. A separate public GitHub scrape exposes corroborating detailed 2025 records, but has no identified repository licence. |
| 2026 | **A — Exact official fantasy scores recovered** | The official public playerstats feed returned current race-by-race/session scores for a valid driver and constructor. |

The official current playerstats route does **not** expose 2021–2025 through the tested `season`, `season_name`, or `year` parameters. Each parameter returned the same 2026 `Value`. No genuine `seasonId` or `championshipId` selector was found in the current official frontend code or responses, so those parameters were not invented or tested.

This report does not claim that no other archive exists. It records only the source mechanisms tested and their observed responses.

The detailed source-by-season inventory is in [`HISTORICAL_SOURCE_MATRIX.csv`](HISTORICAL_SOURCE_MATRIX.csv).

## 1. Exact working official request

### Request

- Hostname: `fantasy.formula1.com`
- Path: `/feeds/popup/playerstats_{player_id}.json`
- Confirmed driver request: `https://fantasy.formula1.com/feeds/popup/playerstats_18.json`
- Confirmed constructor request: `https://fantasy.formula1.com/feeds/popup/playerstats_28.json`
- Required query parameters: none
- Authentication: none
- Relevant non-secret headers:
  - `Accept: application/json,text/plain,*/*`
  - browser-style `User-Agent`
- Timeout used by probes: 15 seconds

The current official frontend source map confirms the template `feeds/popup/playerstats_${playerId}.json`. The live browser network inventory also confirmed the current official config, schedule, market, statistics, and live-mix feed families.

### Response schema

Top level:

- `FeedTime`
- `Value`

`Value` contains:

- `PlayerId`
- `PlayerSkill` (`1` driver, `2` constructor in the inspected fixtures)
- `TourWiseStats`
- `GamedayWiseStats`
- `MatchWiseStats`
- `FixtureWiseStats`

Important identity and score fields:

- entity: `PlayerId`, `PlayerSkill`
- current tour: `TourId`
- event: `GamedayId`, `Season`, `MeetingNumber`, `RaceDayId`
- session: `SessionType`, `SessionName`, `SessionNumber`
- status/time: `IsPlayed`, `MatchStatus`, `SessionStartDate`, `DateTime`
- scoring: `StatsWise[]` with `Event`, `Frequency`, and `Value`
- price: `OldPlayerValue`, `PlayerValue`

The inspected 2026 responses contained 12 gameday/match/fixture records. The schema carries explicit `Season = 2026` values inside its session rows.

### Identifier stability

The official market feed maps current `PlayerId` values to entity metadata; for example player 18 was Pierre Gasly in the inspected 2026 market response. The investigation found no official cross-season identifier map. Therefore:

- current IDs are proven valid for 2026;
- `GamedayId` 1–12 are proven only within the current response;
- global or season-to-season stability of player IDs and gameday IDs is **not proven**;
- current `GamedayId` values were not treated as retired `game_period_id` values.

Sanitised full schema fixtures, with no cookies, tokens, credentials, or personal data, are stored at:

- [`fixtures/playerstats_driver_2026_sanitised.json`](fixtures/playerstats_driver_2026_sanitised.json)
- [`fixtures/playerstats_constructor_2026_sanitised.json`](fixtures/playerstats_constructor_2026_sanitised.json)

## 2. Historical variants of the working endpoint

The probe tested one evidence-backed request in each plausible query-parameter family against the confirmed route:

| Parameter | Requested | HTTP | Payload season | Result |
|---|---:|---:|---:|---|
| `season` | 2025 | 200 | 2026 | Ignored; `Value` matched baseline |
| `season_name` | 2025 | 200 | 2026 | Ignored; `Value` matched baseline |
| `year` | 2025 | 200 | 2026 | Ignored; `Value` matched baseline |

Signatures were calculated after excluding the volatile top-level `FeedTime`. Each response had the same current gameday IDs and the same payload signature as the 2026 baseline. Per the requested stopping rule, lower years were not repeated after each family was proven ignored.

Neither `seasonId` nor `championshipId` appeared in a genuine official response or relevant current frontend source. No historical official gameday IDs were discovered. Those variants were therefore not guessed.

Detailed request evidence is in:

- [`playerstats_historical_variants.csv`](playerstats_historical_variants.csv)
- [`playerstats_historical_variants.json`](playerstats_historical_variants.json)

## 3. Current official frontend assets

The public page `https://fantasy.formula1.com/en/statistics/details` loaded these relevant bundles during the live browser inspection:

- `main.cdb70fa6.js`
- `app.a2733cda.chunk.js`
- `statistics.dfbce992.chunk.js`

Their referenced source maps were publicly accessible and were downloaded without executing JavaScript. The relevant source-map findings are:

- feed base route: `feeds`
- player route: `feeds/popup/playerstats_${playerId}.json`
- current config route: `feeds/v2/apps/web_config.json`
- generic statistics route supplied by config: `statistics/driverconstructors_{tourId}.json`
- current config value: `tourId = 4`
- current resolved statistics route: `feeds/v2/statistics/driverconstructors_4.json`

The resolved bulk statistics response explicitly contained `season = 2026`, current driver/constructor aggregate categories, and current entity IDs. It did not contain race-by-race Fantasy point histories.

The official Statistics source uses the current config/tour only. No historical/archive control, season selector, `season_name`, `seasonId`, or `championshipId` mechanism was found in the relevant source modules.

Downloaded assets and search contexts are recorded under:

- [`frontend_assets/PROVENANCE.json`](frontend_assets/PROVENANCE.json)
- [`frontend_asset_findings.json`](frontend_asset_findings.json)

## 4. F1 Fantasy Tools Statistics

The public Statistics frontend at `https://f1fantasytools.com/statistics` exposes a genuine client route:

```text
GET /api/statistics/{year}
```

Its loaded code defines `ST_FIRST_SEASON = 2023` and presents a selector from 2023 through the current season. Its public pricing page also describes Statistics as containing F1 Fantasy data for every season since 2023.

The following responses were inspected directly and the requested season was verified inside each payload:

| Year | HTTP | JSON | Populated races | Driver entities | Constructor entities | Sessions | Prices |
|---:|---:|---|---:|---:|---:|---|---|
| 2023 | 200 | Yes | 22 | 22 | 10 | Q, S, R | `price`, `priceChange` |
| 2024 | 200 | Yes | 24 | 25 | 10 | Q, S, R | `price`, `priceChange` |
| 2025 | 200 | Yes | 24 | 23 | 10 | Q, S, R | `price`, `priceChange` |

The elevated driver counts reflect replacement/inactive drivers recorded during the season. The detailed data includes total Fantasy points and fields such as position, fastest lap, overtakes, positions gained/lost, DNF/disqualification, Driver of the Day, teamwork, and pit-stop bonuses where applicable.

The endpoint was public and did not require authentication in these requests. It is a third-party service, not an official Formula One source. Its [Terms of Service](https://f1fantasytools.com/terms-of-service) restrict unauthorized automated extraction, reverse engineering, and reuse beyond permitted personal use. Consequently:

- only low-volume public GETs were used;
- no authentication, subscription bypass, or access-control circumvention occurred;
- historical response bodies were inspected in memory but not saved as raw datasets;
- only response/schema summaries and the specifically requested public frontend assets were retained.

Results are in:

- [`verified_source_results.csv`](verified_source_results.csv)
- [`verified_source_results.json`](verified_source_results.json)

## 5. Public repositories, Kaggle, and Internet Archive

### 5.1 Kaggle: 2021

The public Kaggle dataset [Formula1 Fantasy 2021](https://www.kaggle.com/datasets/prathamsharma123/formula-1-fantasy-2021) is labelled **CC0: Public Domain**. It contains:

- 22 driver race-performance CSVs;
- 22 constructor race-performance CSVs;
- driver price history;
- constructor price history.

Each race file includes entity name and `Fantasy Points`; it also carries limited real-result fields, but not a complete official Fantasy scoring-component breakdown.

Because its licence and provenance were confirmed, the archive and extracted files were retained at [`raw-data/kaggle_formula_1_fantasy_2021`](raw-data/kaggle_formula_1_fantasy_2021), with checksums and provenance in `PROVENANCE.json`.

### 5.2 GitHub: 2021–2022 totals

The public repository [jm1261/Fantasy-F1-League](https://github.com/jm1261/Fantasy-F1-League) is MIT-licensed and contains both individual-race and cumulative Fantasy totals:

| Season | Drivers | Constructors | Races per entity | Internal cumulative check |
|---:|---:|---:|---:|---|
| 2021 | 20 | 10 | 22 | 660/660 observations reproduced exactly |
| 2022 | 20 | 10 | 22 | 660/660 observations reproduced exactly |

The repository says it processes data for an official Formula 1 fantasy league but does not claim to have solved official-site scraping. The files are therefore classified as third-party recorded totals, not official API responses. They do not contain scoring-component breakdowns or prices.

The licensed point files, README, licence, checksums, and provenance were retained at [`raw-data/github_jm1261_fantasy_f1_league`](raw-data/github_jm1261_fantasy_f1_league).

### 5.3 2021 cross-source comparison

The Kaggle and GitHub 2021 datasets agree on all 20 driver and 10 constructor identities and all 22 event positions, but not every score:

| Asset type | Compared | Disagreements | Agreement |
|---|---:|---:|---:|
| Drivers | 440 | 23 | 94.8% |
| Constructors | 220 | 14 | 93.6% |

Most disagreements occur at the shortened Belgian Grand Prix, where half-score treatment differs. A smaller number occur at other rounds. Without an official archived payload, neither source can be declared universally exact; this is why 2021 is classification C rather than B.

The reproducible comparison is in [`recovered_dataset_validation.json`](recovered_dataset_validation.json).

### 5.4 Other GitHub findings

- [JoshCBruce/fantasy-data](https://github.com/JoshCBruce/fantasy-data) contains detailed 2025 driver and constructor scrape results with race totals, Q/S/R component fields, values, and selection percentages. It provides independent third-party evidence for 2025, but no repository licence was found; raw files were inspected but not retained as a dataset.
- [EduardoFAFernandes/F1FantasyData](https://github.com/EduardoFAFernandes/F1FantasyData) is GPL-3.0 and contains large 2021–2022 price histories, not race-by-race Fantasy scores. Its README warns that 2021 is very incomplete.
- `JoshCBruce/formula-fantasy` is a library rather than a stored historical dataset.
- `sajal147x/Formula_1_fantasy_analysis` contains ordinary race championship results, not official Fantasy points.
- Other repository-search results either contained models/code only, current-season data, league entrant records, or unrelated race datasets.

### 5.5 Kaggle search

The two public Kaggle index searches returned the CC0 2021 dataset above. The other matching result inspected was a general F1 dataset rather than official Fantasy scores. This is the extent of the tested Kaggle queries, not a claim about all Kaggle content.

### 5.6 Internet Archive

Internet Archive CDX queries were run for:

- `fantasy.formula1.com/feeds/popup/playerstats_*.json`, 2021–2025;
- `fantasy-api.formula1.com/*game_periods_scores*`, 2021–2025;
- official frontend JavaScript assets, 2021–2025.

The first two tested patterns returned zero indexed 200 captures. The frontend query returned 200 distinct archived JavaScript captures within the configured result limit, beginning in 2023. These bundles may preserve endpoint code, but the tested archive index did not directly expose score payloads.

This does **not** establish that the Internet Archive contains no relevant material. It establishes only that these URL patterns and filters did not return archived score JSON during this run.

Search and candidate evidence is in:

- [`public_archive_search_results.json`](public_archive_search_results.json)
- [`candidate_dataset_inventories.json`](candidate_dataset_inventories.json)
- [`candidate_file_findings.json`](candidate_file_findings.json)

## 6. Legacy hostname status

The earlier probe established that `fantasy-api.formula1.com` did not resolve in DNS from this environment. This continuation did not spend further requests guessing paths on that dead hostname.

Accordingly:

- no legacy request reached HTTP;
- no 401/403/404/410 was observed from that host;
- route existence and authentication requirements remain unproven;
- the historical community endpoint documentation remains endpoint-history evidence, not a currently working source.

## 7. Exact remaining coverage gaps

1. No official 2021–2025 playerstats or legacy API response was recovered.
2. No official cross-season `PlayerId`, `GamedayId`, `game_period_id`, `seasonId`, or championship mapping was recovered.
3. 2021 has complete third-party totals and prices but unresolved score disagreements and incomplete scoring components.
4. 2022 has complete third-party race totals but no recovered Q/S/R component breakdown and no independent official validation.
5. 2023–2025 detailed data is third-party. F1 Fantasy Tools exposes it publicly, but no dataset reuse licence was identified; its terms restrict extraction and reuse.
6. Official historical price/threshold snapshots were not recovered. Available historical prices are third-party records.
7. The Internet Archive check covered selected URL patterns and a bounded bundle index, not every possible archived hostname/path.
8. Repository reconstruction from motorsport results remains category D: useful, but subject to omitted bonuses, historical rule changes, corrections, and unavailable official price state.

## 8. Safety and reproducibility

All new network scripts enforce or use:

- GET requests only;
- 15-second timeout;
- at least one second between repeated requests;
- no cookies, bearer tokens, credentials, login automation, or subscription access;
- no JavaScript execution for downloaded assets;
- bounded response sizes for retained files;
- sanitised response previews;
- explicit licence and provenance before retaining raw datasets.

Relevant scripts:

- [`probe_historical_scores.py`](probe_historical_scores.py)
- [`probe_playerstats_history.py`](probe_playerstats_history.py)
- [`inspect_frontend_assets.py`](inspect_frontend_assets.py)
- [`probe_verified_sources.py`](probe_verified_sources.py)
- [`search_public_archives.py`](search_public_archives.py)
- [`inspect_candidate_datasets.py`](inspect_candidate_datasets.py)
- [`inspect_candidate_files.py`](inspect_candidate_files.py)
- [`download_licensed_datasets.py`](download_licensed_datasets.py)
- [`validate_recovered_datasets.py`](validate_recovered_datasets.py)

No application code was changed by this investigation. No files were staged or committed.
