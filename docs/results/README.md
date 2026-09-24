# Research Results

Curated, human-readable results kept in git. Machine outputs of CLI runs live under `data/results/` (git-ignored, backed up to gdrive `quant-lake`). Superseded runs are deleted; git history preserves them.

| File | What it is |
|------|------------|
| [`final-historical/final_historical_campaign_v1_2012-2026.md`](final-historical/final_historical_campaign_v1_2012-2026.md) (+ `.json`) | Frozen buy-only campaign: QQQ100 vs QQQ95/90/85 + SOXX, 120M cohorts, 2012-08-31..2026-08-28 |
| [`thesis-wave/thesis_wave_summary_2026-08-28.md`](thesis-wave/thesis_wave_summary_2026-08-28.md) | Thesis decision table on the FRESH 2026-08-28 panel |
| [`thesis-wave/thesis_wave_d_exit_ai_compute_2026-08-28.md`](thesis-wave/thesis_wave_d_exit_ai_compute_2026-08-28.md) | ai_compute reference-slice exit assessment |
| [`thesis-incremental/thesis_incremental_ai_compute_2026-08-28.json`](thesis-incremental/thesis_incremental_ai_compute_2026-08-28.json) | SOXX 5/10/15 incremental portfolio evidence |

CLI default outputs: `run thesis-wave` -> `thesis-wave/{date}_v2_thesis_wave.md`; `run thesis-incremental` -> `thesis-incremental/{date}_incremental_{thesis_id}.json`. After-tax campaign reports are written to `data/results/experiments/` and promoted here by hand when they are decision evidence.
