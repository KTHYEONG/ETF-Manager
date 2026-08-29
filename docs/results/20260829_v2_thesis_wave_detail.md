# v2 Thesis Wave — Detailed Results (Adaptive Horizon)

**Report date:** 2026-08-29  
**Panel as-of:** 2025-04-30T00:00:00+00:00  
**ADR:** `ADR_20260829_V2_HORIZON_ELIGIBILITY`  
**Command:** `uv run python -m src.cli run thesis-wave --as-of 2025-04-30`  
**Methodology:** adaptive evaluation horizon within `[min_years, target_years]`; prospective iff `span < min_years` (not `target_years`)

Machine-readable flat table: [`data/20260829_v2_thesis_wave.json`](data/20260829_v2_thesis_wave.json)  
Short summary: [`2025-04-30_v2_thesis_wave.md`](2025-04-30_v2_thesis_wave.md)

---

## 1. Executive summary

| thesis_id | vehicle | decision | CE γ=2 | hist median | eval horizon | cohort n | overlap % | prospective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ai_compute` | SOXX | **continue_research** | 1.2020 | 1.2783 | 120M | 8 | 11.5 | no (span 18.67y) |
| `ai_power_bottleneck` | GRID | **reject** | 0.6515 | 0.7877 | 120M | 6 | 7.2 | no (span 15.45y) |
| `physical_automation` | BOTZ | **reject** | 0.5631 | 0.6132 | **96M** (capped) | 1 | 2.0 | no (span 8.62y) |

**Key change vs pre-fix wave:** `physical_automation` moved **prospective → reject**. Root cause was not “insufficient history” but **evaluable 8.6y panel with weak QQQ-relative accumulation** once 120M hard-code was removed.

**Operational invariant:** `PolicyId.QQQ` 100% unchanged. Thesis-wave / report / decision **never** call `adoption_passes`.

---

## 2. Methodology (reproducibility)

### 2.1 Common simulation settings

| Parameter | Value |
| --- | --- |
| Baseline | QQQ 100% (`PolicyId.QQQ`, `modules=0`) |
| Candidate | Proxy vehicle 100% via `targets_override` (`modules=1`) |
| Monthly contribution | ₩1,000,000 (identical external cashflow, I5) |
| Costs | `commission_bps=0`, `fx_spread_bps=0` |
| Fill | `fill_delay_sessions=1` |
| CE ratio | `CE_candidate(γ=2) / CE_baseline(γ=2)` on experiment window terminal wealth |

### 2.2 Adaptive evaluation horizon (new)

For each thesis, `resolve_evaluation_horizon` picks the **largest** whole-year horizon `h` (months = 12×y) such that:

- `y ∈ [min_years, target_years]` (integer years)
- `len(rolling_cohorts(start, end, horizon_months=h, step_months=12)) ≥ 1`
- `span_capped = true` when `h < target_years × 12`

| Thesis | min_years | target_years | proxy span (PIT prices) | chosen h | span_capped |
| --- | --- | --- | --- | --- | --- |
| ai_compute | 5 | 10 | 18.67y | 120M | false |
| ai_power_bottleneck | 5 | 10 | 15.45y | 120M | false |
| physical_automation | 5 | 10 | 8.62y | **96M** | **true** |

Accumulation cohort report: `step_months=12`, `bootstrap_paths=400`, `seed=7`.

### 2.3 Prospective eligibility

| Rule | Definition |
| --- | --- |
| Eligible | `proxy_span_years < horizon.min_years` |
| Not eligible | `proxy_span_years ≥ horizon.min_years` |
| **Not used** | comparison to `target_years` for prospective cutoff |

### 2.4 Long-horizon gate (reporting only)

`long_horizon_passes`: `cohort_count ≥ 10` AND `median_ratio ≥ 1.0`. Does **not** unlock operational policy.

### 2.5 Decision synthesis (no adoption gate)

| Priority | Condition | Decision |
| --- | --- | --- |
| 1 | `prospective.eligible` | `prospective` |
| 2 | `median ≥ 1.0` AND `ce < 1.02` AND NOT `lh_passes` | `watch` |
| 3 | `ce < 0.98` AND `median < 1.0` | `reject` |
| 4 | else | `continue_research` |

### 2.6 Regime proxy (structural slot)

Windows tested: `pre_ai` (2010–2019), `bear_2022` (2022), `recent_2023_2026` (2023–2026).  
Metric: real terminal wealth ratio proxy vs QQQ on identical monthly KRW.

### 2.7 Holdings overlap (overlap slot)

PIT N-PORT `ETF_HOLDINGS`, pairwise overlap vs QQQ incumbent at `as_of`.

---

## 3. Experiment windows (preregistered JSON)

| thesis_id | experiment JSON | start | end | JSON horizon_months |
| --- | --- | --- | --- | --- |
| ai_compute | `m_thesis_ai_compute_soxx_120m.json` | 2007-08-31 | 2025-04-30 | 120 |
| ai_power_bottleneck | `m_thesis_ai_power_bottleneck_grid.json` | 2009-11-30 | 2025-04-30 | 120 |
| physical_automation | `m_thesis_physical_automation_botz_prospective.json` | 2016-09-30 | 2025-04-30 | **96** |

---

## 4. Master metrics table (analysis-ready)

All ratios are **candidate / QQQ baseline** unless noted.

| field | ai_compute | ai_power_bottleneck | physical_automation |
| --- | --- | --- | --- |
| **decision** | continue_research | reject | reject |
| **rationale** | continue_research | reject weak ce 0.6515 median 0.7877 | reject weak ce 0.5631 median 0.6132 |
| **suggested_status** | research | research | research |
| **next_falsifier** | capex_structural_slowdown | capex_structural_slowdown | commercialization_lag |
| **ce_ratio_gamma_2** | 1.201984 | 0.651493 | 0.563092 |
| **evaluated_horizon_months** | 120 | 120 | 96 |
| **span_capped** | false | false | true |
| **proxy_span_years** | 18.669405 | 15.446954 | 8.624230 |
| **prospective_eligible** | false | false | false |
| **long_horizon_passes** | false | false | false |
| **lh_cohort_count** | 8 | 6 | 1 |
| **lh_reason** | n<10 | n<10 | n<10 |
| **hist_median_ratio** | 1.278312 | 0.787711 | 0.613173 |
| **hist_p10_ratio** | 1.098887 | 0.651393 | 0.613173 |
| **hist_worst_ratio** | 1.088274 | 0.625289 | 0.613173 |
| **hist_win_rate** | 1.000 | 0.000 | 0.000 |
| **hist_bootstrap_p05** | 1.174077 | 0.703078 | 0.613173 |
| **overlap_pct** | 11.506297 | 7.214877 | 2.020259 |
| **overlap_shared** | 15 | 6 | 2 |
| **overlap_a_only_wt** | 195.046 | 194.618 | 102.911 |
| **overlap_b_only_wt** | 88.455 | 92.747 | 97.941 |
| **regime_windows_beat** | 2 | 1 | 1 |
| **regime_windows_tested** | 3 | 3 | 3 |
| **regime_median_ratio** | 1.023983 | 0.953228 | 0.855515 |
| **regime_worst_ratio** | 0.854264 | 0.637033 | 0.817787 |
| **valuation_status** | unknown | unknown | unknown |
| **crowding_status** | unknown | unknown | unknown |

---

## 5. Per-thesis detail

### 5.1 `ai_compute` (SOXX)

**Thesis:** AI compute semiconductor capex → SOXX proxy.  
**Decision:** `continue_research` — strong historical median (1.28) but CE adoption hurdle (1.02 for 1 module) not met at γ=2 (1.20); long_horizon n=8 < 10.

| Evidence slot | status | headline |
| --- | --- | --- |
| historical | computed | 120M n=8 median 1.2783, win_rate 100% |
| structural | computed | 2/3 regime windows beat QQQ; bear_2022 SOXX > QQQ |
| overlap | computed | 11.5% vs QQQ, 15 shared names |
| valuation | unknown | not computed |
| crowding | unknown | not computed |

**Divergence flags:** `long_horizon_passes=false`, `overlap_dependence_disclosed=true` (step 12 < horizon 120).

**Research read:** Historical accumulation supports the thesis narrative; overlap with QQQ (~11.5%) means satellite is not a pure orthogonal sleeve. Not an operational challenger under 36M CE gate.

---

### 5.2 `ai_power_bottleneck` (GRID)

**Thesis:** Grid / power bottleneck → GRID proxy.  
**Decision:** `reject` — CE 0.65, median 0.79, win_rate 0% on 120M cohorts.

| Evidence slot | status | headline |
| --- | --- | --- |
| historical | computed | 120M n=6 median 0.7877, worst 0.6253 |
| structural | computed | 1/3 windows beat; bear_2022 GRID > QQQ |
| overlap | computed | 7.2% vs QQQ |
| valuation | unknown | not computed |
| crowding | unknown | not computed |

**Tension:** Regime slice (especially bear_2022) can beat QQQ while full accumulation cohorts lose — vehicle/regime story ≠ durable 10y DCA edge vs incumbent.

---

### 5.3 `physical_automation` (BOTZ)

**Thesis:** Physical automation / robotics → BOTZ proxy.  
**Decision:** `reject` (was `prospective` before adaptive horizon fix).

| Evidence slot | status | headline |
| --- | --- | --- |
| historical | computed | **96M** n=1 median 0.6132 (span-capped) |
| structural | computed | 1/3 windows beat; bear_2022 BOTZ > QQQ |
| overlap | computed | 2.0% vs QQQ — low overlap, weak returns |
| valuation | unknown | not computed |
| crowding | unknown | not computed |

**Horizon audit (why 96M not 120M):**

| horizon_months | step12 cohorts (BOTZ span) | feasible |
| --- | --- | --- |
| 120 | 0 | no |
| 108 | 0 | no |
| 96 | 1 | **yes (chosen)** |
| 84 | 2 | yes |
| 60 | 4 | yes |

**8.62y BOTZ history is evaluable** — not prospective-by-history. Full-window CE γ=2 ≈ 0.56 confirms reject path.

**96M single cohort:** `2016-09-30` → `2024-09-29`, ratio 0.6132. Thin sample; treat bootstrap stats as identical to single draw.

---

## 6. Cross-thesis comparison

### 6.1 Historical vs structural tension

| thesis | hist median | regime median | bear_2022 vs QQQ |
| --- | --- | --- | --- |
| ai_compute | 1.28 (strong) | 1.02 | SOXX beat |
| ai_power_bottleneck | 0.79 (weak) | 0.95 | GRID beat |
| physical_automation | 0.61 (weak) | 0.86 | BOTZ beat |

Pattern: **bear_2022 defensive/regime win does not imply long-horizon DCA dominance** for GRID/BOTZ.

### 6.2 CE vs median (decision inputs)

| thesis | CE γ=2 | vs 1.02 hurdle | median | reject rule (ce<0.98 & med<1) |
| --- | --- | --- | --- | --- |
| ai_compute | 1.20 | below adoption | 1.28 | no |
| ai_power_bottleneck | 0.65 | fail | 0.79 | no (median ≥ 0.98 threshold side) |
| physical_automation | 0.56 | fail | 0.61 | **yes** |

Note: GRID reject uses combined weak CE + median in rationale string; decision engine uses explicit thresholds in §2.5.

### 6.3 Overlap vs incremental value

| vehicle | overlap % | hist median | interpretation |
| --- | --- | --- | --- |
| SOXX | 11.5 | 1.28 | meaningful overlap but strong incremental cohort |
| GRID | 7.2 | 0.79 | moderate overlap, weak incremental |
| BOTZ | 2.0 | 0.61 | **low overlap, still weak** — not a hidden QQQ clone |

---

## 7. Prior wave delta (methodology fix)

| thesis_id | decision (pre-fix) | decision (this run) | driver |
| --- | --- | --- | --- |
| ai_compute | continue_research | continue_research | unchanged |
| ai_power_bottleneck | reject | reject | unchanged |
| physical_automation | **prospective** | **reject** | 120M n=0 masked CE; 96M eval on 8.62y span |

---

## 8. Suggested next research actions (not operational)

| thesis_id | suggested_status | next falsifier | follow-up |
| --- | --- | --- | --- |
| ai_compute | research | capex_structural_slowdown | WF + cost grid; overlap/purity deep-dive; await n≥10 cohorts |
| ai_power_bottleneck | research | capex_structural_slowdown | close or redefine vehicle; regime ≠ accumulation |
| physical_automation | research | commercialization_lag | **do not** paper-forward as prospective; revisit if thesis/vehicle reframed |

---

## 9. Artifact index

| Artifact | Path |
| --- | --- |
| This report | `docs/results/20260829_v2_thesis_wave_detail.md` |
| Flat JSON | `docs/results/data/20260829_v2_thesis_wave.json` |
| Wave JSON (runtime) | `data/thesis_reports/wave_2025-04-30T00-00-00+00-00.json` |
| Per-thesis JSON | `data/thesis_reports/{thesis_id}_2025-04-30T00-00-00+00-00.json` |
| Experiment map | `configs/theses/experiment_map.json` |

---

## 10. Column dictionary (JSON / pivot)

| column | type | description |
| --- | --- | --- |
| `ce_ratio_gamma_2` | float | Singleton-window CE ratio at γ=2 |
| `evaluated_horizon_months` | int | Adaptive cohort horizon used |
| `span_capped` | bool | true when eval h < target_years×12 |
| `proxy_span_years` | float | First/last price session span for proxy |
| `historical_median_ratio` | float | Median cohort terminal wealth ratio |
| `historical_cohort_count` | int | Number of overlapping cohorts |
| `historical_win_rate` | float | Fraction of cohorts with ratio > 1 |
| `overlap_pct` | float | Pairwise holdings overlap vs QQQ |
| `structural_windows_beat_qqq` | int | Regime windows where proxy beat QQQ |
| `prospective_eligible` | bool | History shorter than min_years |
