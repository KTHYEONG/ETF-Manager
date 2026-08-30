# Module Index — Agent Load Recipes

Task-oriented file lists for bounded AI context. See also `docs/code_map.json`.

## ingest

- `src/cli_commands/ingest.py`
- `src/data/fetch.py`
- `src/data/pipeline.py`
- `src/data/catalog.py`
- `src/data/pit.py`
- `src/data/settings.py`
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
- `src/validation/cost_grid.py`
- `src/validation/cadence_robustness.py`
- `src/validation/gate.py`
- `src/validation/ablation.py`
- `configs/experiments/INDEX.json`

## thesis-research

- `src/cli_commands/thesis.py`
- `src/analytics/thesis/evidence.py`
- `src/analytics/thesis/wave.py`
- `src/analytics/thesis/incremental.py`
- `src/analytics/thesis/report.py`
- `src/policy/thesis.py`
- `configs/theses/`
- `docs/architecture/02_policy_and_validation.md`

## diagnose-qqq

- `src/cli_commands/diagnose.py`
- `src/analytics/regimes.py`
- `src/analytics/blends.py`
- `src/analytics/cadence.py`
- `src/analytics/reserve_usage.py`
- `src/analytics/adaptive_hp_screen.py`
- `src/analytics/compound_dca.py`
- `src/features/kafi.py`

## experiment-config

- `src/validation/experiment.py`
- `configs/experiments/README.md`
- `configs/experiments/INDEX.json`
- `configs/experiments/archive/`
- `tests/unit/validation/test_experiment_taxonomy.py`
- `docs/architecture/03_operator_cli.md`
