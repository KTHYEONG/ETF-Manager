# Project Refactor Roadmap

This document orders small implementation units. Only the linked `_spec.md` files are implementation blueprints; later units require their own focused spec after the data contracts below stabilize. Historical experiment JSON, frozen prospective records, and curated result evidence retain their identities throughout the refactor.

## Decision basis

- The repository currently implements `data/raw` and `data/normalized` plus manifests. Gold decision views and features are largely in memory; `data/results` contains run outputs, not reusable market data.
- The preceding probe reproduced duplicate visible macro vintages, non-finite adjusted prices passing the quality gate, a provenance manifest collision, and a cache hit surviving Parquet deletion.
- The first five units therefore stabilize data trust before broad module movement or configuration consolidation.

## Ordered implementation units

| Order | Unit | Target files | Completion boundary |
| --- | --- | --- | --- |
| 1 | [Logical observation identity](refactor_data_pit_spec.md) | `src/data/schema.py`, `src/data/pit.py`; coupled PIT tests | Registered revisable datasets return one then-latest observation without rewriting stored vintages. |
| 2 | [Finite market values](refactor_data_quality_spec.md) | `src/data/quality.py`; quality and pipeline tests | NaN/infinite and specified invalid price/level rows cannot enter Silver. |
| 3 | [Immutable manifest identity](refactor_data_storage_spec.md) | `src/data/storage.py`; storage tests | Same frame with different provenance has distinct, readable manifests; legacy files remain readable. |
| 4 | [Catalog cache identity](refactor_data_catalog_spec.md) | `src/data/catalog.py`; cache tests | Root and provenance isolation, deletion/change detection, and PIT behavior hold on cache hits. |
| 5 | [Retention planning](refactor_data_retention_spec.md) | `src/data/retention.py`; retention tests | Malformed lineage aborts planning and shared referenced files are retained. |

## Follow-on units to specify after order 1–5

| Order | Single-bolt unit | Files to inspect and likely change | Required contract before implementation |
| --- | --- | --- | --- |
| 6 | Bronze/Silver lineage for incremental merges | `src/data/fetch.py`, `src/data/nport_ingest.py`, `src/data/pipeline.py`, `src/data/storage.py`, `src/data/retention.py` | One derived partition records every source payload and prior trusted partition contributing rows. Re-ingest and pruning preserve that lineage. |
| 7 | Decision-ready Gold view and run-pinned catalog | `src/data/catalog.py`, `src/data/panel_freshness.py`, `src/data/query.py`, direct `DataStore.read_normalized` callers in `src/sim/`, `src/validation/`, and `src/analytics/` | A run resolves required manifest identities once; every decision consumes PIT-visible, source-labeled, coverage-checked inputs from that snapshot. Research returns and proxy prices remain distinct from tradable prices. Add a measured memory and latency budget before replacing the current cache. |
| 8 | Repository paths and settings boundary | `src/data/paths.py`, `src/data/settings.py`, `src/validation/experiment.py`, campaign loaders, `src/cli_commands/*`, `configs/`, `experiments/` | Runtime paths resolve from one typed root. Historical experiment definitions and prospective frozen records stay immutable; operational defaults have one named owner and effective date. |
| 9 | Campaign parsing, calculation, and reporting seams | `src/validation/pension_campaign.py`, `historical_campaign.py`, `pension_selection.py`, `after_tax_campaign.py` and their direct tests | Each workflow gets a separate small spec. Public report fields, decisions, input manifest hashes, and deterministic outputs remain unchanged for pinned fixtures. Split by responsibility rather than a line-count limit. |
| 10 | Simulation and policy boundaries | `src/sim/allocation.py`, `after_tax_engine.py`, `pension_engine.py`, `src/policy/after_tax_rules.py` | Each engine gets its own spec and ledger/lot/tax conservation scenarios. Profiling decides whether a hot loop is split. |
| 11 | CLI, analytics compatibility, and agent map | `src/cli.py`, `src/cli_commands/parser.py`, `src/analytics/thesis_*`, `src/analytics/thesis/`, `docs/architecture/`, `docs/code_map.json`, `README.md` | Commands and public imports retain compatibility. Documentation names actual data boundaries and generated inventories instead of fixed counts or former config paths. |

## Verification gates

- For units 1–5, run their named scenario files and coupled data integration tests with `uv run`; inspect any changed result hashes before accepting a behavior-preserving claim.
- No existing `data/`, `records/`, or `docs/results/` content is deleted or rewritten by a spec. Retention stays dry-run until separately invoked with explicit apply semantics.
- Unit 7 requires a realistic panel-size memory/latency measurement; unit 9 requires pinned regression outputs; unit 10 requires cash, units, fees, and tax conservation checks.
- A unit is complete only when its public contract, caller wiring, targeted scenarios, and relevant architecture text agree.
