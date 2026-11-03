# Operating Manual: ETF-Manager

> **Domain Identity:** Account-aware ETF asset allocation research system (Normal, Pension Savings, ISA) with statute-backed after-tax ledgers.

## 1. Domain Ground Truth & Hard Boundaries
- **Strict PIT & Look-Ahead Protection:** Every data row carries `available_at`. Access to unreleased future data raises `LookAheadError` immediately.
- **Statute-Backed Tax Rules:** Tax schedules must strictly align with statutory texts (`configs/tax/*.json`). Hardcoding tax rates in business logic is prohibited.
- **Data Integrity & Conservation:** SHA-256 manifest verification must match byte-for-byte (`UntrustedDatasetError` on mismatch). Verify portfolio accounting conservation (cash + weights = 1.0).
- **Core Quant Invariants (details: `.agents/rules/domain.md`):** Point-in-time data only, no lookahead; historical universe including delisted names; results net of friction with capacity limits; no magic numbers (typed `Spec`/`Config`); ledgers conserve exactly (integer base units/`Decimal`); on anomalies fail closed to `NO_TRADE`.
- **Config & Secrets:** Runtime settings are externalized and validated at application boundaries; never hardcode environment-dependent secrets or endpoints.
- **Workspace Hygiene:** Keep exploratory experiments and temporary files strictly isolated under `scratch/`. Never commit raw cache or scratch files.

## 2. Autonomy & Execution Contract
- **Bias Toward Action & Diagnostic Autonomy:** For data queries, exploratory scratch diagnostics, and empirical root-cause isolation, execute immediately without asking for permission. When asked open-ended questions about bugs or data anomalies, proactively run scratch experiments under `scratch/` to discover truth. Never modify production code or commit in response to open-ended diagnostic queries.
- **Skills:** user-level and project-agnostic (`probe`, `spec`, `implement`, `check`, `refactor`, `commit`, `ship`); this project's policy is the Workflow Profile in §6. Individual slash invocations execute only that skill and halt for user review; `/ship` is the explicit autonomy opt-in for the approved implement → check → commit chain. Specs live under `docs/specs/` (gitignored).

## 3. Project Toolchain & Verification
Verify code changes against the project's native toolchains before concluding tasks:
- **Quality Gate:** `uv run python tools/verify.py`
- **Test Runner:** `uv run pytest`
- **Git Commits:** Run the project's `commit` skill.
- Instruction budget: `uv run python tools/instruction_budget.py`
- Commit guard: git config core.hooksPath .githooks (one-time per clone). Claude Code also runs a Bash guard hook (`.claude/settings.json`); Codex/agy get only the git hook.

## 4. Communication & Language
- **Natural Korean:** Converse, explain rationales, and report findings in Korean (한국어). Inside structured output cards, retain English keys/badges while writing descriptions in Korean.
- **Technical English:** System instructions, rules, specifications (`docs/specs/`), code, and docstrings are written in English.

## 5. Domain Rule Routing
- Before touching `src/**`, `tests/**`, or strategy/universe/accounting logic: read `.agents/rules/domain.md`
- Before touching `tests/**`, test tooling, or fixtures: read `.agents/rules/testing.md`
- Before touching any logging call, log format, or credential handling: read `.agents/rules/logging.md`
- Before touching hot loops, bulk I/O, or parallelism: read `.agents/rules/performance.md`
- Before touching docstrings, comments, or `docs/architecture/**`: read `.agents/rules/documentation.md`

## 6. Workflow Profile
- verify: `uv run python tools/verify.py`
- test_scoped: `uv run pytest {files}`
- spec_dir: `docs/specs` (untracked; archive finished specs to `docs/specs/_done/`)
- scratch_dir: `scratch/` (never commit)
- high_risk_paths: `src/data/**`, `src/execution/**`, `src/sim/**`, `src/validation/**`, `src/policy/**`, `configs/tax/**`
- hard_stops: hardcoding tax rates or editing `configs/tax/*.json` without statutory basis; bypassing SHA-256 manifest verification (`UntrustedDatasetError`) or `LookAheadError` guards; breaking cash+weights=1.0 accounting conservation; any `rm` without previewing exact targets
- commit_style: Korean `<type>: <summary <=50 chars>` + `- **Why:** <rationale>`; types feat/fix/refactor/chore/docs; no AI or co-author trailers; every commit via `/commit`
- artifact_language: English
- report_language: Korean (card keys stay English)
- rules: `.agents/rules/*.md`
