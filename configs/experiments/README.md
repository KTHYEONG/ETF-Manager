# Experiment Config Taxonomy — Index

`configs/experiments/` holds 42 JSON configs. `INDEX.json` is the machine-readable source of truth; this file is the human-readable mirror. Statuses match `INDEX.json` exactly.

| File | Status | Kind | Notes | Location |
|------|--------|------|-------|----------|
| acc_qqq_baseline_120m.json | active | acc | active challenger baseline 120m | configs/experiments/acc_qqq_baseline_120m.json |
| m0_m1.json | fixture | m | M1 schema regression fixture | configs/experiments/m0_m1.json |
| m1_d_universe.json | fixture | m | M1 schema regression fixture | configs/experiments/m1_d_universe.json |
| m1_m2.json | fixture | m | M1 schema regression fixture | configs/experiments/m1_m2.json |
| m1_n_nasdaq.json | fixture | m | M1 schema regression fixture | configs/experiments/m1_n_nasdaq.json |
| m_qqq_grid.json | archived | m | FUTURE_INDUSTRY_STATIC_MIX closed | configs/experiments/archive/m_qqq_grid.json |
| m_qqq_iwf.json | archived | m | FUTURE_INDUSTRY_STATIC_MIX closed | configs/experiments/archive/m_qqq_iwf.json |
| m_thesis_ai_compute_soxx.json | active | m_thesis | active thesis mix ai_compute | configs/experiments/m_thesis_ai_compute_soxx.json |
| m_thesis_ai_compute_soxx_120m.json | active | m_thesis | active thesis mix ai_compute 120m | configs/experiments/m_thesis_ai_compute_soxx_120m.json |
| m_thesis_ai_compute_soxx_inc_5_10_15.json | active | m_thesis | active thesis incremental mix ai_compute | configs/experiments/m_thesis_ai_compute_soxx_inc_5_10_15.json |
| m_thesis_ai_power_bottleneck_grid.json | active | m_thesis | active thesis grid ai_power | configs/experiments/m_thesis_ai_power_bottleneck_grid.json |
| m_thesis_ai_power_pave.json | active | m_thesis | active thesis mix ai_power pave | configs/experiments/m_thesis_ai_power_pave.json |
| m_thesis_ai_power_pave_inc_5_10_15.json | active | m_thesis | active thesis incremental mix ai_power pave | configs/experiments/m_thesis_ai_power_pave_inc_5_10_15.json |
| m_thesis_physical_automation_botz_prospective.json | active | m_thesis | active thesis prospective physical_automation | configs/experiments/m_thesis_physical_automation_botz_prospective.json |
| m_thesis_physical_automation_robo.json | active | m_thesis | active thesis mix physical_automation robo | configs/experiments/m_thesis_physical_automation_robo.json |
| m_thesis_physical_automation_robo_inc_5_10_15.json | active | m_thesis | active thesis incremental mix physical_automation robo | configs/experiments/m_thesis_physical_automation_robo_inc_5_10_15.json |
| wf_qqq_adaptive_contribution.json | archived | wf | superseded by v5 | configs/experiments/archive/wf_qqq_adaptive_contribution.json |
| wf_qqq_adaptive_v2.json | archived | wf | superseded by v5 | configs/experiments/archive/wf_qqq_adaptive_v2.json |
| wf_qqq_adaptive_v3.json | archived | wf | superseded by v5 | configs/experiments/archive/wf_qqq_adaptive_v3.json |
| wf_qqq_adaptive_v4.json | archived | wf | superseded by v5 | configs/experiments/archive/wf_qqq_adaptive_v4.json |
| wf_qqq_adaptive_v5.json | active | wf | operational adaptive lock candidate | configs/experiments/wf_qqq_adaptive_v5.json |
| wf_qqq_soxx10_adaptive_v5.json | active | wf | compound DCA tournament QQQ vs QQQ90/SOXX10 adaptive v5 | configs/experiments/wf_qqq_soxx10_adaptive_v5.json |
| wf_qqq_cadence.json | active | wf | active cadence baseline | configs/experiments/wf_qqq_cadence.json |
| wf_qqq_cadence_twice.json | active | wf | active cadence twice_monthly | configs/experiments/wf_qqq_cadence_twice.json |
| wf_qqq_future_core.json | archived | wf | FUTURE_INDUSTRY_STATIC_MIX closed | configs/experiments/archive/wf_qqq_future_core.json |
| wf_qqq_kafi_deployment.json | active | wf | active kafi deployment | configs/experiments/wf_qqq_kafi_deployment.json |
| wf_qqq_kafi_shape.json | active | wf | active kafi contribution shape | configs/experiments/wf_qqq_kafi_shape.json |
| wf_qqq_overlay.json | active | wf | active overlay | configs/experiments/wf_qqq_overlay.json |
| wf_qqq_reserve.json | archived | wf | superseded by v4 | configs/experiments/archive/wf_qqq_reserve.json |
| wf_qqq_reserve_v2.json | archived | wf | superseded by v4 | configs/experiments/archive/wf_qqq_reserve_v2.json |
| wf_qqq_reserve_v3.json | archived | wf | superseded by v4 | configs/experiments/archive/wf_qqq_reserve_v3.json |
| wf_qqq_reserve_v4.json | active | wf | operational reserve lock candidate | configs/experiments/wf_qqq_reserve_v4.json |
| wf_thesis_ai_compute_soxx_inc_10.json | active | wf_thesis | active thesis walk-forward inc 10 | configs/experiments/wf_thesis_ai_compute_soxx_inc_10.json |
| wf_vt_ff_proxy.json | archived | wf | unreferenced proxy remnant | configs/experiments/archive/wf_vt_ff_proxy.json |
| wf_vt_vti.json | active | wf | active vt vti walk-forward | configs/experiments/wf_vt_vti.json |
| wf_vti_cadence.json | active | wf | active vti cadence | configs/experiments/wf_vti_cadence.json |
| wf_vti_currency.json | active | wf | active vti currency | configs/experiments/wf_vti_currency.json |
| wf_vti_ivv.json | active | wf | active vti ivv | configs/experiments/wf_vti_ivv.json |
| wf_vti_mapping.json | active | wf | active vti mapping | configs/experiments/wf_vti_mapping.json |
| wf_vti_overlay.json | active | wf | active vti overlay | configs/experiments/wf_vti_overlay.json |
| wf_vti_qqq.json | active | wf | active vti qqq | configs/experiments/wf_vti_qqq.json |
| wf_vti_reserve.json | active | wf | active vti reserve | configs/experiments/wf_vti_reserve.json |

## Conventions

- **active**: safe to re-run; referenced by tests/docs.
- **fixture**: `m0_m1` / `m1_*` schema regression anchors — not archived.
- **archived**: superseded or closed research; bytes identical under `archive/`. Legacy paths `configs/experiments/<name>.json` still resolve via `resolve_experiment_config_path` fallback (INFO log once).

## Loader fallback

`src/validation/experiment.py:resolve_experiment_config_path` — if a historical path is not found but its basename exists under `archive/`, the archive path is returned; otherwise `FileNotFoundError`.

