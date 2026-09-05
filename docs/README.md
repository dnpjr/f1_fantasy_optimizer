# Developer documentation

Start with the root README for installation, usage and architecture.

- [2026-09-05 portfolio release audit](RELEASE_AUDIT_2026_09_05.md)
- [Repository release, deployment and artifact policy](REPOSITORY_RELEASE.md)
- [Sprint model maintenance](SPRINT_MODEL_MAINTENANCE.md): the supported production update procedure
- [Canonical 2023–2026 Fantasy history](HISTORICAL_FANTASY_SCORES_2023_2026.md): coverage, source definitions and missingness
- [Historical ingestion implementation notes](HISTORICAL_FANTASY_SCORES_IMPLEMENTATION_NOTE.md)
- [Frozen Sprint shadow](SPRINT_EV_SHADOW.md): diagnostic research model, separate from production
- [Data inventory and provenance](../data/README.md)
- [Retained research reports](../reports/README.md)

Historical research builders are retained to explain model selection. They are not alternative production maintenance commands. In particular the round-11 experiments use their fixed data cutoffs; do not rerun them over later live inputs and interpret that as a production update.
