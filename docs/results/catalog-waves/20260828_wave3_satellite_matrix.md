# Wave 3 — Independent Satellite Matrix (Max-Window Backtest)

**Run date:** 2026-08-28  
**Baseline:** QQQ 100% (`PolicyId.QQQ`, `modules=0`)  
**Candidate:** QQQ + single satellite via `targets_override`  
**Costs:** `commission_bps=0`, `fx_spread_bps=0`  
**Contribution:** ₩1,000,000 / month  
**Adoption gate:** CE\(_{γ=2}\) ratio > **1.02** (unchanged)

Raw metrics: [`data/20260828_satellite_matrix.json`](data/20260828_satellite_matrix.json)

---

## 1. Methodology

### 1.1 Per-combination maximum window

Each satellite uses the **longest feasible** `(start, end)` given listing dates and post-ingest catalog (see [Wave 2 catalog report](20260828_wave2_catalog_and_ingest.md)).

| Group | Satellites | Weights | Start | End |
|-------|------------|---------|-------|-----|
| Core | XLI, SOXX, IBB, ITA | 5%, 10%, 15% | 2006-08-31 | 2025-04-30 |
| GRID | GRID | 5%, 10%, 15% | 2009-11-30 | 2025-04-30 |
| BOTZ | BOTZ | 5%, 10% | 2016-09-30 | 2025-04-30 |

### 1.2 Metrics

| Layer | Engine | Horizon | Cohort design |
|-------|--------|---------|---------------|
| **Adoption (36M)** | `run_ablation` | 36 months | Non-overlapping (`step = horizon`) |
| **Distribution (120M)** | `run_accumulation_cohort_report` | 120 months | Overlapping, `step_months=12`, `bootstrap_paths=1000`, `seed=7` |

**36M CE ratio:** `CE_candidate(γ=2) / CE_baseline(γ=2)` pooled across cohort terminal wealths.  
**120M ratios:** per-cohort `terminal_wealth_real_krw(candidate) / baseline`; summary reports median and worst.

### 1.3 BOTZ exception

Window span ≈ 8.7 years → **no 120M cohort fits**. Report adds **36M accumulation cohort** (`step=12`, n=6) for distribution context only. 36M ablation has only **n=2** non-overlapping cohorts (high variance).

---

## 2. Executive summary

| Finding | Detail |
|---------|--------|
| **Adoption** | **0 / 17** arms pass CE > 1.02 |
| **Best 36M CE** | IBB 10% → **1.0028** (still < 1.02) |
| **Best 120M median** | SOXX 15% → **1.0273** (median > 1, but gate is 36M CE) |
| **GRID** | Confirms FUTURE_INDUSTRY_STATIC_MIX_V1 reject (CE 0.979–0.995) |
| **Operational policy** | **QQQ 100% unchanged** |

---

## 3. Full results — 36M ablation (CE gate)

Hurdle: **1.02**. All `adopted_36m = false`.

| Satellite | 5% CE | 10% CE | 15% CE | 36M cohorts (n) |
|-----------|-------|--------|--------|-----------------|
| **XLI** | 0.9961 | 0.9891 | 0.9832 | 6 |
| **SOXX** | 0.9978 | 0.9969 | 0.9969 | 6 |
| **IBB** | 1.0010 | **1.0028** | 0.9982 | 6 |
| **ITA** | 0.9973 | 0.9916 | 0.9872 | 6 |
| **GRID** | 0.9951 | 0.9861 | 0.9786 | 5 |
| **BOTZ** | 0.9891 | 0.9768 | — | 2 |

### 3.1 CE interpretation

- **Monotonic degradation** with weight for XLI, ITA, GRID, BOTZ — higher satellite share reduces CE vs QQQ.
- **SOXX** is flat across 5–15% (0.997–0.997): semiconductor tilt does not change pooled CE much on 36M non-overlapping windows, but see 120M section.
- **IBB 10%** is the only arm above 1.00 on 36M CE; still **80bp short** of the 2% hurdle.

---

## 4. Full results — 120M accumulation cohorts

Baseline: QQQ 100%. `step_months=12`, overlapping. **n = 9** (core), **n = 6** (GRID).

| Satellite | 5% median / worst | 10% median / worst | 15% median / worst |
|-----------|-------------------|--------------------|--------------------|
| **XLI** | 0.991 / 0.986 | 0.975 / 0.960 | 0.960 / 0.934 |
| **SOXX** | **1.007** / 0.997 | **1.017** / 0.994 | **1.027** / 0.991 |
| **IBB** | 0.999 / 0.995 | 0.978 / 0.972 | 0.952 / 0.936 |
| **ITA** | 0.998 / 0.989 | 0.974 / 0.961 | 0.958 / 0.938 |
| **GRID** | 0.992 / 0.987 | 0.981 / 0.966 | 0.970 / 0.946 |

### 4.1 SOXX deep dive

SOXX is the **only** satellite where **median 120M ratio > 1.0** at all three weights:

```text
Weight   median_ratio   worst_ratio   Interpretation
  5%       1.0072         0.9970      Slight median edge; worst cohort still loses
 10%       1.0174         0.9938      Median +1.7%; tail still below parity
 15%       1.0273         0.9905      Median +2.7%; worst −0.95%
```

**Caution:** 120M report is **not** the adoption gate. Overlapping cohorts (n=9) are dependent; 36M CE gate uses non-overlapping n=6. Do not adopt SOXX on 120M median alone without a pre-registered Objective B.

### 4.2 GRID vs prior research

| Study | GRID 5% CE (36M) | This run (max window) |
|-------|------------------|------------------------|
| FUTURE_INDUSTRY_STATIC_MIX_V1 (2012+ safe window) | ~0.998 | **0.995** |
| 120M median (this run) | — | **0.992** |

Direction unchanged: GRID dilutes QQQ compounding. **Observation list only.**

### 4.3 BOTZ (36M cohort substitute)

| Weight | 36M ablation CE (n=2) | 36M acc. median | 36M acc. worst | n (step=12) |
|--------|----------------------|-----------------|----------------|-------------|
| 5% | 0.9891 | 0.9929 | 0.9868 | 6 |
| 10% | 0.9768 | 0.9851 | 0.9716 | 6 |

BOTZ window too short for 120M; ablation n=2 is **not statistically meaningful**.

---

## 5. Cross-metric ranking

### 5.1 By 36M CE (adoption metric)

| Rank | Arm | CE ratio |
|------|-----|----------|
| 1 | IBB 10% | 1.0028 |
| 2 | IBB 5% | 1.0010 |
| 3 | IBB 15% | 0.9982 |
| 4 | SOXX 5% | 0.9978 |
| … | … | … |
| 17 | BOTZ 10% | 0.9768 |

### 5.2 By 120M median ratio (evidence only)

| Rank | Arm | Median |
|------|-----|--------|
| 1 | SOXX 15% | 1.0273 |
| 2 | SOXX 10% | 1.0174 |
| 3 | SOXX 5% | 1.0072 |
| 4 | IBB 5% | 0.9985 |
| … | … | … |

**Metric divergence:** SOXX ranks 1–3 on 120M median but 4–6 on 36M CE. IBB ranks 1–3 on CE but 4–9 on 120M median.

---

## 6. Research freeze updates

Per architecture satellite research discipline (`02_policy_and_validation.md`):

```text
Single satellite → gate pass → combination candidate
Single satellite → gate fail → exclude from Wave 4
```

| Satellite | 36M CE pass | 120M median > 1 (any weight) | Wave 4 |
|-----------|-------------|------------------------------|--------|
| XLI | No | No | **Exclude** |
| SOXX | No | Yes (5–15%) | **Exclude** (CE gate) |
| IBB | No | No (5% ≈ 1.0) | **Exclude** |
| ITA | No | No | **Exclude** |
| GRID | No | No | **Exclude** (already rejected) |
| BOTZ | No | N/A | **Exclude** |

**Wave 4 combination search:** no eligible arms. Proceed to calendar-extension work before combination or SOXX deep-dive under a separate objective.

---

## 7. Reproduction

```bash
# Prerequisite: post-ingest catalog (2006-08-28 .. 2025-05-30 prices)
uv run python scratch/run_satellite_matrix.py   # scratch runner (purged on sync)

# Single arm example
uv run python -m src.cli run ablation --config configs/experiments/m_qqq_grid.json
uv run python -m src.cli run accumulation-cohort \
  --config configs/experiments/acc_qqq_baseline_120m.json \
  --horizon-months 120 --cohort-step-months 12 --seed 7
```

Adjust experiment JSON `start`/`end` per tables in §1.1 before running.

---

## 8. Limitations

1. **Zero costs** — favours multi-ETF candidates; QQQ advantage may be understated.
2. **Span** — core window ~18.7 years; GFC included, dot-com excluded (calendar floor).
3. **Small n** — 36M ablation n=5–6; BOTZ n=2.
4. **Overlapping 120M** — n=9 cohorts are dependent; bootstrap used but not an adoption gate.
5. **No walk-forward** — in-sample cohort screen only; Wave 5 would add WF + cost grid.
