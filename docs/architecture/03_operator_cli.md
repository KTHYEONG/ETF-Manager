# Operator CLI & Artifacts

## 1. Entry Point

```bash
uv run python -m src.cli <command> ...
```

All execution, lint, and tests use the `uv run` prefix per project policy.

> Wave 1 layout: `src/cli.py` is a thin facade (≤450 LOC) importing `src/cli_commands/*`; domain runners live in `parser`, `resolvers`, `ingest`, `thesis`, `diagnose`, `sim_run`, `campaign` (each ≤500 LOC). Tests migrated to `tests/unit/cli/` by command family.

## 2. Commands

### Ingest

| Command | Purpose |
| --- | --- |
| `ingest history --start DATE --end DATE` | Full panel: prices (`history_price_tickers()`), FX, CPI, factors, VIX, research returns |
| `ingest static-dca --start DATE --end DATE` | CPI + prices + FX for long-horizon static-DCA panels |
| `ingest thesis-panel` | Refresh panel tickers (`BOTZ, GRID, PAVE, QQQ, ROBO, SOXX`), FX, CPI, and N-PORT |
| `ingest thesis-fundamentals` | Fetch structural macro series (PNFI, A35SNO, NEWORDER) for thesis fundamentals |
| `ingest nport --filing-quarter YYYYqN` | SEC N-PORT quarterly holdings dataset |
| `ingest prices --tickers T ... --start --end` | Single-dataset price pull |
| `ingest fx / cpi / factors / macro / research-returns` | Individual vendor datasets |

### Run — simulation

| Command | Purpose |
| --- | --- |
| `run baseline --id b0_global\|b1_us --ticker T --start --end --contribution-krw` | Single-sleeve fast DCA |
| `run policy --id <PolicyId> --start --end --contribution-krw` | Multi-sleeve allocation (operational: `qqq`) |
| `run validate --id --start --end --contribution-krw --delta0 --modules --horizon-months` | Cohort CE vs B0 |
| `run paper --id qqq --start --end --contribution-krw` | Buy-only paper replay |

Optional policy flags: `--tilt-factor`, `--tilt-intensity`, `--overlay-max-shift`, `--vix-threshold`,
`--reserve-max-withhold`, `--fx-max-defer`, `--fx-expensive-percentile`, `--rebalance-band`,
`--map-etf`, `--map-min-improvement`.

### Run — validation campaigns

| Command | Config example |
| --- | --- |
| `run ablation --config configs/experiments/m1_m2.json` | Cohort CE gate |
| `run walk-forward --config configs/experiments/wf_s1_s7.json` | Train/test adoption (single candidate) |
| `run strategy-select --config configs/experiments/wf_compound_dca_tournament.json` | Multi-candidate WF tournament selection |
| `run walk-forward-costs --config configs/experiments/wf_s0_s1.json` | Cost scenario grid |
| `run walk-forward-proxy --config configs/experiments/wf_s0_r1.json` | I9 proxy campaign |
| `run cadence-robustness --config ... --seed 7` | Growth-first cadence robustness gate |
| `run accumulation-cohort --config ... --horizon-months 120` | 120M wealth-ratio distribution (reporting only) |
| `run audit-feasibility --config ... [--write-report]` | Static DCA window / cohort-count audit |

### Run — diagnostics & thesis (reporting only)

| Command | Purpose |
| --- | --- |
| `run diagnose-us-vehicles --start --end --contribution-krw` | VTI/IVV/QQQ factor profile + identical DCA (no gate) |
| `run diagnose-compound-dca --contribution-krw` | Compound DCA tournament (12 arms, flat vs adaptive, no gate) |
| `run diagnose-qqq-regimes / diagnose-qqq-blends / diagnose-qqq-reserve / diagnose-qqq-cadence / diagnose-qqq-accumulation-alpha / diagnose-qqq-kafi / diagnose-qqq-adaptive-hp` | Specialized QQQ diagnostic runs |
| `run diagnose-overlap --vehicle SOXX --baseline QQQ` | Holdings overlap between two vehicles at PIT as-of |
| `run thesis [--id THESIS] [--config-dir configs/theses] [--compute-evidence]` | List or inspect thesis registry |
| `run thesis-report --id THESIS [--as-of ISO] [--experiment PATH]` | Build thesis report with 5-slot evidence vector |
| `run thesis-wave [--as-of ISO] [--allow-stale]` | Run thesis wave for all theses; write wave JSON + markdown |
| `run thesis-incremental [--thesis-id THESIS] [--as-of ISO] [--seed N]` | Track H incremental portfolio (QQQ95/90/85) + bootstrap |
| `run thesis-pipeline [--thesis-id THESIS] [--as-of ISO]` | Complete thesis pipeline (wave + incremental + Wave D exit) |

### Maintain

| Command | Purpose |
| --- | --- |
| `maintain prune [--apply] [--keep-latest-only] [--drop-nport-zip-mirrors] [--migrate-results-layout]` | Prune stale partitions, zip mirrors, and migrate results |

## 3. Experiment JSON (`configs/experiments/`)

```json
{
  "name": "wf_qqq_soxx10_adaptive_v5",
  "start": "2016-07-01",
  "end": "2026-06-30",
  "contribution_krw": 1000000,
  "delta0": 0.02,
  "horizon_months": 0,
  "train_months": 36,
  "test_months": 24,
  "commission_bps": 0.0,
  "fx_spread_bps": 0.0,
  "objective": "compound_growth",
  "thesis_id": "ai_compute",
  "preregistration": {
    "weights_locked": true,
    "universe_locked": true,
    "baseline_frozen": true
  },
  "baseline": { "id": "qqq_flat", "policy": "qqq", "modules": 0 },
  "candidates": [
    {
      "id": "qqq90_soxx10_adaptive_v5",
      "policy": "qqq",
      "modules": 2,
      "targets": { "QQQ": 0.9, "SOXX": 0.1 },
      "adaptive_contribution": { "name": "kafi_v5", "active": true }
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `horizon_months` | `0` = single path; `>0` = rolling cohort step for ablation |
| `train_months` / `test_months` | Walk-forward only; both required together |
| `modules` | Declared complexity for CE penalty (not auto-counted) |
| `delta0` | Per-module CE margin (default 0.02) |
| `objective` | Gate objective: `compound_growth` (no MDD veto), `contribution_growth`, or `ce` |
| `thesis_id` | Optional registered `ThesisId` (`ai_compute`, `ai_power_bottleneck`, `physical_automation`) |
| `preregistration` | Locked weights/universe and baseline hash verification |

Single-candidate walk-forward specs use `run walk-forward`; multi-candidate tournament specs use `run strategy-select`.

Thesis JSON lives under `configs/theses/` (`ThesisSpec`: falsifiers, sleeves, historical proxies,
five evidence slots).

## 4. Data Root Layout

```text
data/
├── raw/<provider>/...
├── normalized/<dataset>/schema_version=<v>/part-*.parquet
├── manifests/<dataset>/<retrieved_date>.json
└── results/
    ├── experiments/   # walk-forward / strategy-selection reports
    ├── audits/        # static DCA feasibility audits
    └── thesis/        # thesis wave / incremental / exit artifacts
```

Catalog reads go through `DataSettings` → `load_visible` / `latest_artifact`.
Partitions without a matching manifest are rejected (fail-closed).

## 5. Operator Date Window

Shipped JSON files use catalog-feasible panels. With the current ECOS CPI fixed-lag
availability model and Tiingo price coverage, operators should prefer:

| Constraint | Safe bound |
| --- | --- |
| Start | `2012-08-31` or later (CPI PIT visible at first execution) |
| End | Latest complete month-end with price + CPI + FX coverage |

Failures surface as `BaselineDataError` / `AllocationDataError` with explicit missing-row messages.

## 6. Typical Workflow

```bash
# 1. Refresh catalog / thesis panel
uv run python -m src.cli ingest history \
  --start 2012-06-01 --end 2024-10-31

# 2. Operational policy smoke
uv run python -m src.cli run policy \
  --id qqq --start 2016-07-01 --end 2026-06-30 \
  --contribution-krw 1000000

# 3. Strategy selection tournament
uv run python -m src.cli run strategy-select \
  --config configs/experiments/wf_compound_dca_tournament.json

# 4. Thesis wave and exit assessment
uv run python -m src.cli run thesis-pipeline \
  --thesis-id ai_compute
```

## 7. Logging

Structured lines use the `[DATA] event=...` prefix. CLI exit codes: `0` success, `1` data/sim
failure, `2` usage error. Campaign reports persist under `data/results/experiments/` (or `data/experiments/`)
with `process_adopted_vs_baseline` and per-fold wealth fields.
