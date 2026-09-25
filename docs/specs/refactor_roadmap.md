# Project Refactor Roadmap

This document orders small implementation units. Only the linked `_spec.md` files are implementation blueprints; later units require their own focused spec. Historical experiment JSON, frozen prospective records, and curated result evidence retain their identities throughout the refactor.

## Operating goal

The data layer must stay correct **without operator attention**. The user runs ingest and analysis commands; the system itself guarantees that a trusted dataset never silently shrinks, that damage is detected and repaired automatically where the source is re-downloadable, and that it stops loudly only when it cannot repair. Every unit is judged by two questions: does it remove a way for data to go wrong unnoticed, and does it reduce (not add) concepts the operator or a caller must understand?

### Reproducibility scope (deliberately narrow)

- **Required:** every run/result records the identity (manifest hash) of each Silver input it consumed, and PIT-visible reads stay causal. Frozen prospective records and pinned result evidence are immutable.
- **Not required:** keeping every historical Silver partition or Bronze payload forever. Public sources (SEC, FRED, Tiingo, ECOS, Ken French) are re-downloadable, so Bronze is a repair cache, not a permanent archive.
- Consequence: a Silver partition is trusted on its own Silver identity (manifest + Parquet hash). A missing Bronze payload is a repairable condition, not a reason to refuse reading Silver.

## Decision basis

- Units 1–5 fixed real defects (duplicate macro vintages, non-finite prices passing quality, manifest collision, stale cache hits, unsafe retention). They are complete.
- Units 1–5 also made trust depend on Bronze presence (`DataStore.read_normalized` re-hashes the raw payload on every read, 440 MB for N-PORT). Combined with an existing fail-open pattern, this produced a live incident on 2026-09-25:
  - `etf_holdings` Bronze ZIP was missing, so the 4,017-row Silver partition (2019-09 … 2026-04, 6 ETFs) became untrusted.
  - Re-ingesting `2026q2` hit `except UntrustedDatasetError: pass` in the incremental merge, skipped the prior partition, and wrote a 385-row partition (2026-02 … 2026-04, 5 ETFs).
  - `latest_artifact` picks the newest `retrieved_at`, so the truncated partition is now the dataset's latest view. No error was raised.
- The same fail-open merge exists for `prices`, `rates`, and `macro` in `src/data/fetch.py`. Any transient trust failure can therefore replace a full history with only the freshly fetched slice.

## Completed units

| Order | Unit | Target files | Status |
| --- | --- | --- | --- |
| 1 | [Logical observation identity](refactor_data_pit_spec.md) | `src/data/schema.py`, `src/data/pit.py` | Done |
| 2 | [Finite market values](refactor_data_quality_spec.md) | `src/data/quality.py` | Done |
| 3 | [Immutable manifest identity](refactor_data_storage_spec.md) | `src/data/storage.py` | Done |
| 4 | [Catalog cache identity](refactor_data_catalog_spec.md) | `src/data/catalog.py` | Done |
| 5 | [Retention planning](refactor_data_retention_spec.md) | `src/data/retention.py` | Done |

## Remaining units

| Order | Unit | Files to inspect and likely change | Required contract before implementation |
| --- | --- | --- | --- |
| 6 | Non-shrinking incremental ingest | `src/data/fetch.py` (prices, rates, macro), `src/data/nport_ingest.py`, `src/data/catalog.py`, `src/data/storage.py` | One shared merge helper replaces the four ad-hoc merges. If a prior partition exists but cannot be read, ingest aborts (`[DATA]` error) instead of writing a fresh-slice-only partition. A new partition must cover every prior key except those explicitly refreshed by this ingest; otherwise it is not written. The manifest records a flat `sources` list (raw payload hashes of this ingest plus the prior manifest hash) instead of new lineage structures. Includes a one-time repair: merge the trusted 4,017-row and 385-row `etf_holdings` partitions into one new latest partition. |
| 7 | Silver trust independent of Bronze | `src/data/storage.py`, `src/data/catalog.py`, `src/data/retention.py` | `read_normalized` verifies manifest and Parquet identity only; Bronze hash is checked at ingest/repair time, not on every read. Missing Bronze logs a `[DATA]` warning and is reported as repairable. Retention keeps only the latest Silver per dataset, the partitions referenced by recorded results/prospective records, and the Bronze of the latest partition; everything else is prunable. |
| 8 | Self-healing `maintain data` command | `src/cli_commands/parser.py`, `src/cli.py`, new `src/data/doctor.py` | One idempotent command replaces manual inspection: for each dataset, verify latest Silver, re-fetch missing Bronze from its recorded endpoint, rebuild only when needed, then apply the safe retention plan. Default is dry-run with a one-screen summary; `--apply` performs it. It never deletes a partition referenced by a result or prospective record. Ingest commands run the verification step automatically. |
| 9 | Decision-ready Gold view and run-pinned catalog | `src/data/catalog.py`, `src/data/panel_freshness.py`, `src/data/query.py`, direct `DataStore.read_normalized` callers in `src/sim/`, `src/validation/`, `src/analytics/`, and the `UntrustedDatasetError` fallbacks in `src/cli_commands/campaign.py`, `src/validation/*` | A run resolves required manifest identities once and records them in its output; every decision consumes PIT-visible, coverage-checked inputs from that snapshot. Caller-level `except UntrustedDatasetError` fallbacks are removed or made fail-closed. Research returns and proxy prices remain distinct from tradable prices. Measure memory/latency on a realistic panel before replacing the current cache. |
| 10 | Repository paths and settings boundary | `src/data/paths.py`, `src/data/settings.py`, `src/validation/experiment.py`, campaign loaders, `src/cli_commands/*`, `configs/`, `experiments/` | Runtime paths resolve from one typed root. Historical experiment definitions and prospective frozen records stay immutable; operational defaults have one named owner and effective date. |
| 11 | Campaign parsing, calculation, and reporting seams | `src/validation/pension_campaign.py`, `historical_campaign.py`, `pension_selection.py`, `after_tax_campaign.py` and their direct tests | Each workflow gets a separate small spec. Public report fields, decisions, input manifest hashes, and deterministic outputs remain unchanged for pinned fixtures. Split by responsibility rather than a line-count limit. |
| 12 | Simulation and policy boundaries | `src/sim/allocation.py`, `after_tax_engine.py`, `pension_engine.py`, `src/policy/after_tax_rules.py` | Each engine gets its own spec and ledger/lot/tax conservation scenarios. Profiling decides whether a hot loop is split. |
| 13 | CLI, analytics compatibility, and agent map | `src/cli.py`, `src/cli_commands/parser.py`, `src/analytics/thesis_*`, `src/analytics/thesis/`, `docs/architecture/`, `docs/code_map.json`, `README.md` | Commands and public imports retain compatibility. Remove dead "wiring detection" anchors (`_ = some_function`) and swallowed `except Exception: pass` blocks in ingest code. Documentation names actual data boundaries and generated inventories instead of fixed counts or former config paths. |

## Simplicity guardrails

- A unit may add a concept only if it removes a failure mode listed in its contract; prefer deleting a special case over adding a new structure.
- Lineage stays a flat list of hashes in the manifest. No lineage graph, no per-row provenance.
- Fail-closed means: stop and say what is wrong and which command repairs it. It never means silently continuing with a smaller dataset.

## Verification gates

- Each unit runs its named scenario files and coupled data integration tests with `uv run`; inspect any changed result hashes before accepting a behavior-preserving claim.
- Unit 6 must reproduce the 2026-09-25 incident as a test (prior partition untrusted → ingest aborts, no new manifest) and pass the non-shrinking invariant for all four merge paths.
- Units 7–8 must show a read-only run against the real `data/` root: all datasets readable, and the maintenance dry-run lists only expected actions.
- No existing `records/` or `docs/results/` content is deleted or rewritten. `data/` changes happen only through ingest, the unit 6 repair, or `maintain data --apply`.
- Unit 9 requires a realistic panel-size memory/latency measurement; unit 11 requires pinned regression outputs; unit 12 requires cash, units, fees, and tax conservation checks.
- A unit is complete only when its public contract, caller wiring, targeted scenarios, and relevant architecture text agree.
