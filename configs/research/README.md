# Research Configs

`INDEX.json` is the only catalog of every experiment ever run. The trial-lineage census counts them
(`active` and `archived` entries) for multiple-testing haircuts, so an entry is never deleted.

## What stays as a file

Only configs that a shipped command still reads: the pension pipeline (`pension_*`),
the after-tax campaigns (`after_tax_*`), and the thesis tooling (`m_thesis_*`).
Every other experiment was retired: its config file is gone and its INDEX entry carries
`"retired": true` with its original status, so the census count is unchanged. Git history
holds the retired definitions.

## Catalog

Add an INDEX entry when adding a config. `tests/unit/validation/test_experiment_taxonomy.py`
enforces that INDEX keys equal the files present plus the retired entries.

## Path resolution

`src/data/paths.py:resolve_repo_path` maps the historical `experiments/<name>` and
`configs/experiments/<name>` citations to `configs/decision/<renamed>` for the three decision
specs, else `configs/research/<name>`, so frozen configs keep their bytes and hashes.
