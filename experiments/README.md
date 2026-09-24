# Experiment Configs

Research experiment definitions live here. This directory holds the JSON
configs consumed by walk-forward, cohort, and historical campaigns.

## Conventions

- **active**: safe to re-run; referenced by tests/docs.
- **fixture**: `m0_m1` / `m1_*` schema regression anchors — not archived.
- **archived**: superseded or closed research under `archive/`.
  Archived configs are never deleted: the trial-lineage census counts them
  (including archived entries) for multiple-testing haircuts.

## Catalog

`INDEX.json` is the only catalog — add an entry there when adding a config.
`tests/unit/validation/test_experiment_taxonomy.py` enforces coverage
(INDEX ↔ files consistency).

## Loader fallback

`src/validation/experiment.py:resolve_experiment_config_path` resolves an
experiment config path with this fallback order:

1. `path` as given, if it is a file.
2. Legacy `configs/experiments/<rel>` → `experiments/<rel>`.
3. Either prefix → `experiments/archive/<basename>`.
