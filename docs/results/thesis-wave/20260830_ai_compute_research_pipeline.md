# ai_compute Research Pipeline — 2024-08-31 panel

**Run date:** 2026-08-30  
**Panel as-of:** `2024-08-31` (catalog prices through `2024-09-30`; `freshness_status=STALE`, `lag_days=698`, `--allow-stale`)  
**Operational invariant:** `PolicyId.QQQ` 100% unchanged; no `adoption_passes` on thesis paths.

## Pipeline

1. `ingest thesis-panel` + FX/CPI backfill (`fx` → `2024-09-30`, `cpi` from `2000-01-01`)
2. `run thesis-wave --as-of 2024-08-31 --allow-stale`
3. `run thesis-incremental --thesis-id ai_compute --as-of 2024-08-31 --allow-stale`
4. `run ablation --config configs/experiments/m_thesis_ai_compute_soxx_inc_5_10_15.json`
5. `run walk-forward` / `walk-forward-costs` — `configs/experiments/wf_thesis_ai_compute_soxx_inc_10.json`
6. `run diagnose-overlap --vehicle SOXX --baseline QQQ --as-of 2024-08-31`
7. `run accumulation-cohort --config configs/experiments/m_thesis_ai_compute_soxx_120m.json`

Machine JSON: `data/results/thesis/`, `data/results/experiments/`.

## Thesis wave

| thesis_id | decision | 120M median | overlap % |
|-----------|----------|-------------|-----------|
| ai_compute | continue_research | 1.41 (n=3) | 11.0 |
| ai_power_bottleneck | reject | 0.87 | 6.0 |
| physical_automation | reject | no 120M cohort | 2.0 |

Summary: [`2024-08-31_v2_thesis_wave.md`](2024-08-31_v2_thesis_wave.md)

## Track H incremental

`portfolio_status`: **historically_promising** (all arms path-bootstrap ok).

| Arm | median | mean realized SOXX | target |
|-----|--------|-------------------|--------|
| QQQ95/SOX5 | 1.007 | 3.3% | 5% |
| QQQ90/SOX10 | 1.017 | 9.9% | 10% |
| QQQ85/SOX15 | 1.027 | 16.8% | 15% |

JSON: [`../thesis-incremental/2024-08-31_incremental_ai_compute.json`](../thesis-incremental/2024-08-31_incremental_ai_compute.json)

## Ablation / walk-forward (qqq90_soxx10 representative)

- Ablation CE γ=2: 5% **1.013**, 10% **1.033**, 15% **1.053** — all `adopted=false` (long_horizon gate).
- WF `process_adopted_vs_baseline`: **false** (2 folds; train never adopts; OOS candidate slightly beats baseline).
- Cost grid (ideal/low/base/stress): **all_scenarios_adopted=false**.

## Overlap (SOXX vs QQQ)

`overlap_pct` **11.03%**, **13** shared holdings (PIT `2024-08-31`).

## 120M cohort gate

`cohort_count` **3** at 120M (`start=2012-08-31`); **n≥10 not met**. Max at 60M horizon: **n=8**. Observe until catalog span lengthens.

## Next (research only)

- Extend catalog ~12 months → re-run 120M accumulation for n≥10.
- Track F fundamentals (valuation/crowding still `unknown`).
- Do not change operational QQQ; do not expand SOXX weight grid beyond 5/10/15%.
