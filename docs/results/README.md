# Research Results Archive

Curated narrative and summary JSON for operator review. Machine outputs from CLI runs also live under `data/results/` (experiments, thesis wave JSON).

## Layout

| Directory | Contents |
|-----------|----------|
| [`thesis-wave/`](thesis-wave/) | Batch thesis-wave markdown + methodology deep-dives; `data/` holds flat JSON tables |
| [`thesis-incremental/`](thesis-incremental/) | Track H incremental portfolio JSON (`QQQ95/90/85` vs `QQQ100`) |
| [`catalog-waves/`](catalog-waves/) | Historical catalog ingest and satellite-matrix reports (Wave 2–3) |
| [`archive/`](archive/) | Superseded or exploratory runs (stale panel, pre-attribution-fix, etc.) |

## Canonical (catalog `end=2024-08-31`, panel STALE allowed)

| Report | Path |
|--------|------|
| Thesis wave summary | [`thesis-wave/2024-08-31_v2_thesis_wave.md`](thesis-wave/2024-08-31_v2_thesis_wave.md) |
| Track H incremental | [`thesis-incremental/2024-08-31_incremental_ai_compute.json`](thesis-incremental/2024-08-31_incremental_ai_compute.json) |
| Full pipeline write-up | [`thesis-wave/20260830_ai_compute_research_pipeline.md`](thesis-wave/20260830_ai_compute_research_pipeline.md) |

## Reference (methodology / prior panel)

| Report | Path |
|--------|------|
| Adaptive horizon detail (`as_of=2025-04-30`) | [`thesis-wave/20260829_v2_thesis_wave_detail.md`](thesis-wave/20260829_v2_thesis_wave_detail.md) |
| Wave 2 catalog & ingest | [`catalog-waves/20260828_wave2_catalog_and_ingest.md`](catalog-waves/20260828_wave2_catalog_and_ingest.md) |
| Wave 3 satellite matrix | [`catalog-waves/20260828_wave3_satellite_matrix.md`](catalog-waves/20260828_wave3_satellite_matrix.md) |

CLI defaults: `run thesis-wave` → `docs/results/thesis-wave/{date}_v2_thesis_wave.md`; `run thesis-incremental` → `docs/results/thesis-incremental/{date}_incremental_{thesis_id}.json`.
