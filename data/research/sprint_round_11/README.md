# Frozen round-11 research inputs

These inputs preserve the original four-Sprint experiment as canonical production history advances. They are **regression/research fixtures**, never production market or calibration inputs.

- `canonical.csv`: exact CSV field values from the v3 canonical release dataset, retaining 2023–2025 and 2026 rounds 1–11. Dutch R12 is deliberately absent. The original source/provenance fields are retained.
- `market.json`: a **derived fixture, not a raw official response**. Identities come from each asset's latest canonical round through R11; accepted feed-12 prices and names come from `reports/2026_sprint_final_candidate/final_candidate.csv`. The original verification timestamp is unavailable and remains null. No prices or results were invented.
- `example_team.json`: the previously public example lineup, retained solely for deterministic research budgets. Personal app state uses ignored `data/current_team.local.json`.

The fixed comparison script and historical regression tests use these files. Current Sprint calibration tests use the current canonical dataset and v2 artifact. Keep these fixtures fixed when new weekends arrive; add new regression cases for new behaviour.
