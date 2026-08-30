# Wave 2 — Catalog Extension & Feasibility Windows

**Run date:** 2026-08-28  
**Operator goal:** Extend static-DCA data toward the 2000s and establish maximum feasible backtest windows before Wave 3 satellite tests.

---

## 1. Ingest actions performed

| Command / action | Parameters | Outcome |
|------------------|------------|---------|
| `fetch_and_persist_cpi` | 2000-01-01 → 2025-05-31 | **305 rows** (was 215) |
| `fetch_and_persist_fx` (FRED) | 2006-08-28 → 2025-05-31 | **4,719 rows** |
| `fetch_and_persist_prices` | 2006-08-28 → 2025-05-31, 8 tickers | **34,412 rows** |

Tickers ingested: `QQQ`, `GRID`, `XLI`, `SOXX`, `IBB`, `ITA`, `BOTZ`, `IWF`.

`ingest static-dca` from 2000-01-01 failed at price persist: Tiingo returns pre-2006 dates, but `stamp_availability` calls `XNYS.close_ts()` which rejects sessions before **2006-08-28**.

---

## 2. Post-ingest catalog coverage

| Dataset | First observation | Last observation | Notes |
|---------|-------------------|------------------|-------|
| CPI `period_end` | 2000-01-31 | 2025-05-31 | ECOS monthly; `FIXED_LAG` 45d |
| FX `date` | 2006-08-28 | 2025-05-30 | FRED DEXKOUS, session-filtered |
| PRICES `date` | 2006-08-28 | 2025-05-30 | Per-ticker listing floors below |

### Per-ticker price floors (after ingest)

| Ticker | First session | Binds window when included |
|--------|---------------|----------------------------|
| QQQ, XLI, SOXX, IBB, ITA, IWF | 2006-08-28 | Core satellite group |
| GRID | 2009-11-17 | GRID mix only |
| BOTZ | 2016-09-13 | BOTZ mix only |

---

## 3. Structural limits (not data-provider gaps)

### 3.1 XNYS calendar floor

```
exchange_calendars XNYS first_session = 2006-08-28
```

CPI rows exist from 2000, but **no ETF allocation path can execute before 2006-08-28** in the current engine. Dot-com (2000–2002) stress requires either calendar extension or a research-proxy path (out of Wave 2 scope).

### 3.2 Feasible end date

| Requested `end` | Feasibility | Reason |
|-----------------|-------------|--------|
| 2025-05-31 | **FAIL** (`fx`, `price`) | May signal → execution **2025-06-02**; no marks |
| 2025-04-30 | **PASS** | Last month-end with complete execution marks |
| 2025-05-30 | FAIL | Same June execution gap |

**Operator rule:** use `end = 2025-04-30` until catalog prices extend past 2025-06-02 execution.

### 3.3 Earliest feasible start (after ingest, `end = 2025-04-30`)

| Profile | Earliest month-end start | Decision months | 36M cohorts (non-overlap) | 120M cohorts (step=12) |
|---------|--------------------------|---------------|---------------------------|-------------------------|
| QQQ only / XLI / SOXX / IBB / ITA | **2006-08-31** | 225 | 16 | **9** |
| QQQ + GRID | **2009-11-30** | 186 | 13 | **6** |
| QQQ + BOTZ | **2016-09-30** | 104 | 2 | **0** |

Wave 2 target `cohort_count_120m_step12 ≥ 10` remains **unmet** (9 for QQQ-only) due to the 2006 calendar floor, not ingest failure.

---

## 4. Wave 2 engineering deliverables

| Component | Path | Role |
|-----------|------|------|
| Feasibility audit | `src/validation/feasibility_audit.py` | Static-DCA dependency profile, coverage rows, cohort count |
| Static ingest | `fetch_and_persist_static_dca_datasets`, CLI `ingest static-dca` | PRICES+FX+CPI only (no MACRO) |
| Audit CLI | `run audit-feasibility` | JSON under `data/audits/` |
| Experiment seed | `configs/experiments/acc_qqq_baseline_120m.json` | 120M smoke config |

### Dependency profile (static mix)

For `targets_override` arms with `modules ∈ {0,1}` and no overlay/reserve/currency/adaptive:

```text
required_datasets = (prices, fx, cpi)
requires_macro = False
```

MACRO is **not** a blocker for static satellite ablations.

---

## 5. Comparison to pre-ingest state

| Metric | Pre-ingest (2007-08 panel) | Post-ingest |
|--------|---------------------------|-------------|
| CPI start | 2007-08-31 | **2000-01-31** (unused for ETF sim) |
| QQQ price start | 2007-08-31 | **2006-08-28** |
| QQQ feasible start | 2007-10-31 | **2006-08-31** |
| Max feasible end | 2025-05-31 (fragile) | **2025-04-30** (stable) |
| 120M cohorts (QQQ) | 8 | **9** |

Net gain: **~10 months** earlier ETF history and **+1** overlapping 120M cohort; still short of 10-cohort Wave 2 gate.

---

## 6. Recommended operator windows (post-ingest)

```text
QQQ / XLI / SOXX / IBB / ITA :  2006-08-31  →  2025-04-30
QQQ + GRID                   :  2009-11-30  →  2025-04-30
QQQ + BOTZ                   :  2016-09-30  →  2025-04-30  (120M N/A)
```

---

## 7. Open items

1. **Calendar extension** — allow XNYS (or NASDAQ) sessions before 2006-08-28 to unlock 2000s ETF DCA.
2. **Catalog end refresh** — ingest through a month whose execution session ≤ last price row (target ≥ 2025-05-31 signal).
3. **Wave 2 gate** — 120M `n ≥ 10` blocked at 9 cohorts until span lengthens ~12 months.
