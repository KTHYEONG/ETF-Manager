# Operator CLI & Artifacts

## 1. Entry Point

```bash
uv run python -m src.cli <command> ...
```

All execution, lint, and tests use the `uv run` prefix per project policy.

## 2. Commands

### Ingest

| Command | Purpose |
| --- | --- |
| `ingest history --start DATE --end DATE` | Full panel: prices (`history_price_tickers()`), FX, CPI, factors, VIX, research returns |
| `ingest prices --tickers T ... --start --end` | Single-dataset price pull |
| `ingest fx / cpi / factors / macro` | Individual datasets |

Default history tickers: sorted union of `all_policy_tickers()` and diagnostic `QQQ`.

### Run — simulation

| Command | Purpose |
| --- | --- |
| `run baseline --id b0_global\|b1_us --ticker T --start --end --contribution-krw` | Single-sleeve fast DCA |
| `run policy --id <PolicyId> --start --end --contribution-krw` | Multi-sleeve allocation (operational: `qqq`) |
| `run validate --id --start --end --contribution-krw --delta0 --modules --horizon-months` | Cohort CE vs B0 |
| `run paper --id qqq --start --end --contribution-krw` | Buy-only paper replay |

Optional policy flags: `--tilt-factor`, `--overlay-max-shift`, `--vix-threshold`,
`--fx-max-defer`, `--rebalance-band`, `--map-etf`.

### Run — validation campaigns

| Command | Config example |
| --- | --- |
| `run ablation --config configs/experiments/m1_m2.json` | Cohort CE gate |
| `run walk-forward --config configs/experiments/wf_s1_s7.json` | Train/test adoption |
| `run walk-forward-costs --config configs/experiments/wf_s0_s1.json` | Cost scenario grid |
| `run walk-forward-proxy --config configs/experiments/wf_s0_r1.json` | I9 proxy campaign |

### Run — diagnostics

| Command | Purpose |
| --- | --- |
| `run diagnose-us-vehicles --start --end --contribution-krw` | VTI/IVV/QQQ factor profile + identical DCA (no gate) |

## 3. Experiment JSON (`configs/experiments/`)

```json
{
  "name": "wf_s1_s7",
  "start": "2012-06-01",
  "end": "2024-10-31",
  "contribution_krw": 1000000,
  "delta0": 0.02,
  "horizon_months": 0,
  "train_months": 60,
  "test_months": 36,
  "commission_bps": 0.0,
  "fx_spread_bps": 0.0,
  "baseline": { "id": "s1_us", "policy": "s1_us", "modules": 0 },
  "candidates": [{ "id": "s7_us_large_cap", "policy": "s7_us_large_cap", "modules": 1 }]
}
```

| Field | Meaning |
| --- | --- |
| `horizon_months` | `0` = single path; `>0` = rolling cohort step for ablation |
| `train_months` / `test_months` | Walk-forward only; both required together |
| `modules` | Declared complexity for CE penalty (not auto-counted) |
| `delta0` | Per-module CE margin (default 0.02) |

Walk-forward specs require exactly one candidate.

## 4. Data Root Layout

```text
data/
├── raw/<provider>/...
├── normalized/<dataset>/schema_version=<v>/part-*.parquet
├── manifests/<dataset>/<retrieved_date>.json
└── experiments/<name>_<experiment_id>.json   # walk-forward reports
```

Catalog reads go through `DataSettings` → `load_visible` / `latest_artifact`.
Partitions without a matching manifest are rejected (fail-closed).

## 5. Operator Date Window

Shipped JSON files use `2014-01-03` / `2024-09-30`. With the current ECOS CPI fixed-lag
availability model and Tiingo price coverage, operators should prefer:

| Constraint | Safe bound |
| --- | --- |
| Start | `2012-08-31` or later (CPI PIT visible at first execution) |
| End | `2024-09-30` or earlier (avoid execution session after last price bar) |

Failures surface as `BaselineDataError` / `AllocationDataError` with explicit missing-row messages.

## 6. Typical Workflow

```bash
# 1. Refresh catalog
uv run python -m src.cli ingest history \
  --start 2012-06-01 --end 2024-10-31

# 2. Operational policy smoke
uv run python -m src.cli run policy \
  --id qqq --start 2012-06-01 --end 2024-10-31 \
  --contribution-krw 1000000

# 3. Next validation wave (cost robustness)
uv run python -m src.cli run walk-forward-costs \
  --config configs/experiments/wf_s0_s1.json

# 4. Optional diagnostics (no policy change)
uv run python -m src.cli run diagnose-us-vehicles \
  --start 2012-06-01 --end 2024-10-31 --contribution-krw 1000000
```

## 7. Logging

Structured lines use the `[DATA] event=...` prefix. CLI exit codes: `0` success, `1` data/sim
failure, `2` usage error. Campaign reports persist under `data/experiments/` with
`process_adopted_vs_baseline` and per-fold wealth fields.
