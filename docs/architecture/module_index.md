# Module Index — Agent Load Recipes

Task-oriented file lists for bounded AI context. See also `docs/code_map.json`.

## ingest

- `src/cli_commands/ingest.py`
- `src/data/fetch.py`
- `src/data/pipeline.py`
- `src/data/catalog.py`
- `src/data/pit.py`
- `src/data/settings.py`
- `src/data/panel_freshness.py`
- `configs/data/thesis_fundamentals/`

## policy-run

- `src/cli_commands/sim_run.py`
- `src/cli_commands/resolvers.py`
- `src/policy/targets.py`
- `src/sim/allocation.py`
- `src/sim/baseline.py`
- `src/sim/contribution.py`
- `src/execution/orders.py`

## validate-campaign

- `src/cli_commands/campaign.py`
- `src/validation/experiment.py`
- `src/validation/walk_forward.py`
- `src/validation/strategy_selection.py`
- `src/validation/historical_campaign.py`
- `src/validation/prospective_registry.py`
- `src/validation/research_posture.py`
- `experiments/INDEX.json`

## thesis-research

- `src/cli_commands/thesis.py`
- `src/analytics/thesis/structural.py`
- `src/analytics/thesis/valuation.py`
- `src/analytics/thesis/crowding.py`
- `src/analytics/thesis/purity.py`
- `src/analytics/thesis/wave.py`
- `src/analytics/thesis/incremental.py`
- `configs/theses/`

## diagnose-qqq

- `src/cli_commands/diagnose.py`
- `src/analytics/compound_dca.py`
- `src/analytics/regimes.py`
- `src/analytics/blends.py`
- `src/analytics/cadence.py`
- `src/analytics/reserve_usage.py`
- `src/analytics/overlap.py`
- `src/analytics/us_vehicles.py`

## maintain

- `src/cli_commands/maintenance.py`
- `src/data/doctor.py`
- `src/data/retention.py`
- `src/data/merge.py`

## experiment-config

- `src/validation/experiment.py`
- `experiments/README.md`
- `experiments/INDEX.json`
- `experiments/archive/`
- `records/prospective/` (immutable preregistration freezes)
- `data/results/<experiment>/` (run artifacts + `runs.jsonl`)
- `tests/unit/validation/test_experiment_taxonomy.py`
- `docs/architecture/overview.md`
