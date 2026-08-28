# ETF-Manager v2: Thesis-Centric Evolution Plan

Review of the v2 thesis-centric direction against `docs/architecture/` (post–Wave 0), Wave 1–3 results, and completed Wave 0 implementation.

---

## 1. One-sentence shift

| | Question the system answers |
| --- | --- |
| **v1 (today)** | Which `PolicyId` / ETF mix beat QQQ on past accumulation under PIT data and CE gate? |
| **v2 (target)** | Which **economic thesis** is structurally true, investable via listed vehicles, not already overpriced, and robust enough to challenge QQQ over a 10+ year horizon? |

v2 is **not** “find the best backtest ETF.” Backtest becomes one step in a longer pipeline.

---

## 2. Accepted v2 direction

Wave 3 (2026-08-28) showed the limit of v1:

- 17 satellite arms (`QQQ + X%` via `targets_override`): **0/17** pass 36M CE > 1.02
- SOXX 15%: 120M median ratio **1.0273** — reporting only, not adoption
- BOTZ: **no** 120M cohort (history too short)
- GRID: confirms prior reject

The system cannot yet say separately:

1. “AI power thesis may be structurally valid”
2. “GRID is a dirty vehicle for that thesis”
3. “Historical backtest is the wrong regime for this thesis”

`next.md` correctly identifies that the **research unit** must change from ETF/`PolicyId` to **thesis**.

Other directions we **accept**:

- **Sleeve ≠ vehicle** — economic exposure vs listed ticker (`NASDAQ_100` / `QQQ` / `QQQM`)
- **Falsifiers required** — every thesis must state what would prove it wrong
- **Evidence as a vector** — structural / historical / valuation / overlap / crowding; **no magic composite score**
- **QQQ stays operational incumbent** — goal is validated challenge, not “beat QQQ algorithm”
- **Prospective OOS** for industries with short history (freeze weights, paper forward)
- **Preregistration / hypothesis registry** — failed experiments kept, not discarded
- **Strangler migration** — `next.md` §21 also says no big-bang rewrite

---

## 3. What we defer or reject (for now)

| `next.md` proposal | Why not now |
| --- | --- |
| Full `src/etf_manager/` tree (~50 modules) | 74 Python files today; rename breaks imports/tests with zero new research questions |
| `sim→backtest`, `policy→portfolio`, CLI split | `cli.py` is large but functional; housekeeping after identity is used |
| Holdings look-through / purity % | **No holdings code**; N-PORT PIT only from **2019Q4** (`01_data_contracts.md`) |
| PELT / Bai-Perron / CUSUM on capex, MW, grid queue | **No such `Dataset`**; manual `QQQ_REGIME_WINDOWS` in `regimes.py` is honest reporting-only |
| 120M CE as **operational adoption gate** | n=9 overlapping cohorts (Wave 2 target n≥10 unmet); dependent samples; 36M gate stays until sample rules pass |
| YAML thesis configs | Project uses JSON + Pydantic `extra="forbid"`; no PyYAML dependency |
| Research index backcast (2026 picks in 2010) | Violates I9/I10; historical leg = ETF proxy only |
| Hansen SPA / scenario IRF / `live/` package | No identification strategy; ops/research seam already exists in architecture docs |
| Collapsing five evidence slots to one number | `next.md` itself forbids this — **invariant** |

**Principle:** add capability only when **data + contract** exist; do not rename folders to simulate progress.

---

## 4. v1 architecture (current)

Source: `docs/architecture/00_system_overview.md`, `02_policy_and_validation.md`.

```text
L1 Data → L2 Features → L3 Policy (PolicyId → targets)
                              ↓
L4 Simulation (ledger SSOT) → L5 Validation (CE / WF / cohorts)
                              ↓
L6 ETF mapping (optional)     Analytics (reporting only)
```

**Identity chain today:**

```text
PolicyId (qqq, vti, …) → resolve_targets → ticker weights → run_allocation
```

**Operational lock:** `PolicyId.QQQ` 100%, adaptive contribution locked.

**Adoption gate:** 36M (and WF) CE ratio > `1 + delta0 × modules` for γ ∈ {2, 5, 10}.

**Research satellites:** `targets_override` on `PolicyId.QQQ` — still ticker-centric, not thesis-centric.

**Already built (feedback / Wave 1–3):**

| Done | Module / artifact |
| --- | --- |
| Wave 1 | 120M rolling accumulation cohort (`validation/accumulation_cohort.py`) — **reporting only** |
| Wave 2 | Historical coverage audit, static-DCA ingest, CPI to 2000, prices to 2006 |
| Wave 3 | Independent satellite matrix, `research_satellite_tickers()` |
| Earlier | `targets_override`, I9 research proxy, I12 experiment registry, factor attribution, PaperBroker |

---

## 5. v2 architecture (target)

Logical pipeline from `next.md` §2 — **phased**, not all at once:

```text
WORLD (macro / industry change)
  ↓
THESIS (hypothesis + falsifiers + lifecycle)
  ↓
FUNDAMENTAL EVIDENCE     ← needs new datasets (later)
  ↓
EXPOSURE / PURITY        ← needs N-PORT holdings (later)
  ↓
SLEEVE → VEHICLE         ← Wave 0
  ↓
VALUATION / EXPECTATIONS ← later
  ↓
HISTORICAL VALIDATION    ← existing sim + validation
  ↓
REGIME / ROBUSTNESS      ← reporting → computed (later)
  ↓
DECISION (reject / watch / prospective / operational challenger)
```

**New identity chain (after Wave 0+):**

```text
ThesisId → candidate SleeveId(s) → resolve_vehicle → VehicleId (QQQ, SOXX, …)
                ↓
         ExperimentSpec / targets_override / PolicyId (operational alias)
                ↓
         run_allocation (unchanged engine)
```

**What stays the same:**

- PIT data layer (`src/data/*`), invariants I1–I12
- Ledger-centric simulation (`src/sim/allocation.py`)
- CE gate for operational adoption (until a future wave passes explicit sample-size rules)
- QQQ operational lock unless a **separate** adoption wave succeeds

**Module placement (strangler — inside existing `src/`, not `etf_manager/`):**

| Concern | v1 location | v2 addition (incremental) |
| --- | --- | --- |
| Sleeve / vehicle IDs | `policy/targets.py` (`UsEquityUniverse`) | `etf/sleeves.py` |
| Thesis registry | — | `policy/thesis.py` + `configs/theses/*.json` |
| Exposure / overlap | — | new module **after** holdings ingest |
| Evidence slots | — | declared on `ThesisSpec`, computed in analytics **reporting only** |
| Regime | `analytics/regimes.py` (manual windows) | changepoint **only on ingested fundamental series** |
| Experiments | `validation/experiment.py` | optional `thesis_id`, preregistration flags |
| CLI | `cli.py` | `run thesis` (inspect); split later if needed |

---

## 6. Side-by-side comparison

| Dimension | v1 (`docs/architecture/`) | v2 (evolution target) |
| --- | --- | --- |
| Primary research object | `PolicyId` / ETF ticker | `ThesisId` |
| “Why buy?” | Implicit in policy name | Explicit thesis + causal chain + **falsifiers** |
| Economic vs listed | Partial (`UsEquityUniverse` → ticker) | First-class `SleeveId` / `VehicleId` |
| ETF = industry? | Often assumed (GRID = power) | Measured purity / overlap when data exists |
| Regime | Fixed reporting windows | Fundamental changepoint (future) |
| Horizon vs gate | 36M CE adoption; 120M report-only | 120M as **additional** objective when n/rules OK |
| Short history | “insufficient history → reject” | **Prospective challenger** + frozen paper OOS |
| Experiment discipline | JSON specs + registry | + preregistration, failed-run registry |
| Package layout | `src/data`, `policy`, `sim`, … | Same roots; new files beside old |
| Operational policy | QQQ 100% | QQQ 100% until formal challenger adoption |

---

## 7. Phased work order

Execute in order. **Do not skip Wave 0** — later waves depend on stable IDs.

```mermaid
flowchart LR
  W0[Wave 0 Identity] --> W1[Wave 1 Experiments]
  W1 --> W2[Wave 2 Holdings]
  W2 --> W3[Wave 3 Evidence]
  W3 --> W4[Wave 4 Long horizon]
  W4 --> W5[Wave 5 Prospective]
  W5 --> W6[Wave 6 Regime data]
  W6 --> W7[Wave 7 E2E theses]
```

### Wave 0 — Research identity kernel **(next implement)**

**Spec:** `docs/specs/v2_research_identity_contract.json`

| Deliverable | Description |
| --- | --- |
| `SleeveId`, `VehicleId`, `resolve_vehicle` | `NASDAQ_100` → `QQQ`; satellites → SOXX/GRID/BOTZ |
| `ThesisSpec` + registry loader | JSON in `configs/theses/`, falsifiers required |
| Three seed theses | `ai_compute`, `ai_power_bottleneck`, `physical_automation` |
| `run thesis` CLI | Inspect/list only; **no** adoption gate |
| Wire `UNIVERSE_VEHICLE` | Derived from `resolve_vehicle`; QQQ lock unchanged |

**Exit criteria:** contract tests pass; operational QQQ and 36M CE unchanged.

**Command:** `/implement docs/specs/v2_research_identity_contract.json`

---

### Wave 1 — Experiments speak thesis

| Deliverable | Description |
| --- | --- |
| `thesis_id` on `CandidateSpec` | Optional link from experiment JSON to registry |
| Preregistration fields | `weights_locked`, `universe_locked`, baseline frozen at run time |
| Failed experiment registry | All arms logged (extend `validation/registry.py`) |
| Update `docs/architecture/02_policy_and_validation.md` | Design line: Thesis → Sleeve → Vehicle |

**Gate:** still 36M CE; no operational change.

---

### Wave 2 — Holdings & overlap (data-blocked)

| Prerequisite | SEC N-PORT ingest, PIT from 2019Q4 |
| --- | --- |
| `Dataset` + provider | Holdings weights, as-of filing date |
| Overlap / purity | e.g. “SOXX add = mostly NVDA overlap with QQQ?” |
| GRID thesis purity | Reporting-only % buckets |

**Cannot start** until ingest schema and manifests exist.

---

### Wave 3 — Computed evidence (reporting only)

Fill `ThesisSpec` evidence vector from **existing** engines:

| Slot | Source |
| --- | --- |
| Historical accumulation | 120M cohort reporter, factor attribution |
| Structural | placeholder until Wave 6 fundamentals |
| Valuation / crowding | placeholder until valuation datasets |
| Overlap | Wave 2 holdings |

**Invariant:** evidence output never calls `adoption_passes`.

---

### Wave 4 — Long-horizon objective (optional additive)

| Deliverable | Description |
| --- | --- |
| `objective=long_horizon` on `ExperimentSpec` | Alongside `ce`, not replacing until approved |
| Fail-closed rules | Min cohort count, overlap dependence disclosure |
| SOXX-style divergence | Document when 120M median and 36M CE disagree |

**Not** an automatic operational lock change.

---

### Wave 5 — Prospective OOS

| Deliverable | Description |
| --- | --- |
| Freeze timestamp + immutable weights | Extend I12 lineage |
| Paper portfolio forward | `PaperBroker` on calendar time |
| Status `PROSPECTIVE_CHALLENGER` | Runtime transition from `ThesisSpec` |

For BOTZ-like cases: history insufficient → prospective, not permanent reject.

---

### Wave 6 — Fundamental regime (data-blocked)

| Prerequisite | Ingest capex, industry series with PIT `available_at` |
| --- | --- |
| Changepoint detector | On fundamentals only — **not** price returns |
| Thesis health | Align breaks across series → confidence, not auto-trade |

Replaces manual windows for **structural** claims only; keep manual windows for stress reports.

---

### Wave 7 — End-to-end research waves

Run three seed theses through full pipeline:

| Thesis | Role |
| --- | --- |
| AI Compute | Strong history + strong fundamentals proxy (SOXX) |
| AI Power Bottleneck | Weak history + regime story (GRID) |
| Physical Automation | Short history + prospective path (BOTZ) |

**Output shape** (from `next.md` §19): thesis report with evidence vector + decision + next falsifier — not a ticker rank table.

---

### Deferred housekeeping (after Wave 1+)

Only when thesis IDs are used in production experiments:

- Split `cli.py` into `app/cli/*.py`
- Rename modules (`sim`→`backtest`, etc.) with compatibility shims
- Optional `research/` vs `execution/` package boundaries

---

## 8. Seed theses (Wave 0 configs)

| ThesisId | Proxy vehicle | Typical question |
| --- | --- | --- |
| `ai_compute` | SOXX | Does semiconductor exposure add durable 10y accumulation vs QQQ? |
| `ai_power_bottleneck` | GRID | Is grid/power exposure structurally accelerating despite weak ETF history? |
| `physical_automation` | BOTZ | Is commercialization lag vs hype visible; is prospective OOS needed? |

Do **not** add 30 ETFs until these three paths work in code.

---

## 9. Invariants (all waves)

| Rule | Enforcement |
| --- | --- |
| I1–I12 | Unchanged |
| QQQ operational lock | Until explicit adoption wave |
| 36M CE gate | Default adoption until Wave 4 rules approved |
| Thesis registry ≠ adoption | JSON cannot set `operational_challenger` or adopt |
| Analytics / thesis inspect | Never call `adoption_passes` |
| No 2010 backcast of 2026 research index | I9/I10 |
| Five evidence dimensions | Never merged into one score |
| `from src.*` public layout | No `etf_manager/` rename without dedicated migration wave |

---

## 10. How this relates to `docs/architecture/`

| Document | Role |
| --- | --- |
| `docs/architecture/*` | **Current** normative contracts (v1 layers, thesis registry Wave 0, I1–I12, CE math) |
| `docs/plans/v2_thesis_evolution.md` (this file) | **Evolution** roadmap and deferred v2 scope |

When Wave 0 lands, update **`00_system_overview.md`** module map and **`02_policy_and_validation.md`** design principle **surgically** (English, ≤300 lines per file). Do not duplicate this plan into architecture docs.

---

## 11. Immediate next step

```text
Wave 0: implement thesis / sleeve / vehicle identity
        → configs/theses/*.json (3 files)
        → run thesis (inspect)
        → zero change to QQQ operations or CE gate
```

```bash
/implement docs/specs/v2_research_identity_contract.json
```

After Wave 0: sync architecture tables, then spec Wave 1 (experiment `thesis_id` + preregistration).
