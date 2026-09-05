# 2026 Sprint EV shadow

The application contains a research-only, frozen Sprint-aware EV calculation.
It is displayed only when **Show Sprint-aware EV shadow** is enabled in
Settings → Diagnostics. The calculation adds explicitly named `shadow_*`
columns; production expected points, price projections, recommendations,
budgets and optimiser objectives remain authoritative.

Runtime code reads
`data/generated/sprint_ev_shadow/sprint_ev_shadow_2026_v1.json`. It does not
import the research builders, fit coefficients, update the artefact, or make a
network request. Normal-equivalent form is calculated only from selected,
completed 2026 canonical scores. A normal weekend retains its recorded total;
a Sprint weekend subtracts the available `sprint_points` and
`sprint_qualifying_points`. A wholly missing Sprint-only component makes that
asset/event unavailable rather than turning it into zero.

## Production maintenance

The production calibration now has a separate, explicit maintenance command:
`python scripts/update_sprint_model.py`. Follow
[the maintenance procedure](SPRINT_MODEL_MAINTENANCE.md) to audit, stage, test and
activate a versioned production update. The frozen research shadow described
above remains separate and is not changed by this workflow.

## Historical research recalibration

Before a future Sprint, the intended workflow is:

1. Refresh completed official Fantasy history.
2. Manually run `scripts/calibrate_asset_sprint_adjustments.py` and/or the
   reviewed final-candidate builder.
3. Review the resulting research reports and validation tests.
4. Deliberately update the versioned frozen calibration artefact.
5. Restart or reload the application.

The application must never perform steps 2–4 automatically. A changed model
must receive a new reviewed version identifier; the report directory is not a
runtime dependency.
