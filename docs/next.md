# Next — ETF-Manager Research Roadmap

> **한 줄 상태 (2026-08-28):** QQQ는 incumbent로 유지한다. Wave 1(120M cohort engine)은 완료됐다. 다음 우선순위는 **Wave 2 historical coverage**이며, 그 전에 의미 있는 120M 연구 실행은 데이터 창이 짧아 불가능하다.

근거 문서: [`docs/feedback.md`](feedback.md) · 완료 기록: `ADR_20260828_WAVE1_ROLLING_120M_COHORT`

---

## 1. 고정된 연구 결론 (변경 금지)

### FUTURE_INDUSTRY_STATIC_MIX_V1

```text
Result:
- QQQ remains operational policy.
- IWF does not replace QQQ.
- GRID 5/10/15% does not improve CE.
- QQQ80/GRID10/XLI10 does not pass walk-forward adoption.
- No further tuning of GRID weights is permitted in this research wave.
```

| 항목 | 상태 |
| --- | --- |
| Operational policy | `qqq` 100% (+ adaptive contribution v5 lock) |
| Adoption gate | `candidate CE / baseline CE > 1.02` (`delta0 = 0.02`) — **사후 완화 금지** |
| GRID | Research reject; observation list only |
| XLI (in 80/10/10 mix) | 독립 검증 전 — 조합 효과와 혼재 |

### 금지 사항

- GRID 비중 3/7/8% 등 **사후 재조정**
- CE hurdle 낮추기
- 단일 satellite gate 실패 arm을 Wave 4 조합에 포함
- Objective A (wealth max)와 Objective B (diversification) **혼합**

---

## 2. 완료 — Wave 1: Long-Horizon Validation

**ADR:** `ADR_20260828_WAVE1_ROLLING_120M_COHORT`

| 항목 | 상태 |
| --- | --- |
| 120-month rolling accumulation cohort engine | ✅ |
| Cohort step 1M / 12M / 36M | ✅ |
| median / p10 / worst / win-rate | ✅ |
| Overlap metadata + `independent_sample_warning` | ✅ |
| Moving-block bootstrap (`bootstrap_p05_ratio_mean`) | ✅ |
| Recovery time (`recovery_months`) | ✅ |
| CLI `run accumulation-cohort` | ✅ reporting-only |

### Smoke test (safe window)

현재 카탈로그 feasible start는 `2012-06-01`. 기존 `m_qqq_grid.json` (`start: 2007`)은 feasibility 실패 — **Wave 2 전까지 safe window config 사용**.

```bash
# scratch/wave1_smoke.json: start 2012-06-01, end 2024-09-30, QQQ vs QQQ95+GRID5
uv run python -m src.cli run accumulation-cohort \
  --config scratch/wave1_smoke.json \
  --horizon-months 36 --cohort-step-months 12 \
  --bootstrap-paths 500 --seed 7
```

`horizon-months=120` full run은 Wave 2 데이터 확장 후 수행. 현재 창(`2012-06`~)에서는 cohort 수가 1~2개 수준이라 분포 통계 의미 없음.

---

## 3. Wave 의존성

```text
Wave 1 (cohort engine)          ✅ DONE
        │
        ▼
Wave 2 (historical coverage)    ← NEXT: /spec → /implement
        │
        ├──────────────────┐
        ▼                  ▼
Wave 3 (single satellite)   Wave 1 full 120M baseline report
        │                  (Wave 2 창 확보 후)
        ▼
Wave 4 (combination)            ← Wave 3 gate 통과 arm만
        │
        ▼
Wave 5 (operational validation) ← Wave 4 후보 확정 후
```

| Wave | `/spec` 시점 | `/implement` 시점 |
| --- | --- | --- |
| 2 | **지금** | Wave 2 spec 승인 직후 |
| 3 | Wave 2와 **골격 병렬** 가능 | Wave 2 ingest + feasibility audit 완료 후 |
| 4 | Wave 3 단일 satellite 결과 후 | 통과 arm 목록 확정 후 |
| 5 | Wave 4 조합 후보 확정 후 | Wave 3~4 research freeze 후 |

---

## 4. Wave 2 — Historical Coverage (NEXT)

**목표:** static DCA feasibility window를 2012년 이전으로 확장해 dot-com / GFC / 2000s regime을 포함한다.

### P0 — Feasibility dependency audit

- [ ] `resolve_feasibility` / `assert_experiment_feasible`가 static mix에 불필요한 dataset(VIX, BAA10Y, macro)으로 window를 자르는지 감사
- [ ] blocking dataset별 earliest feasible `start` 리포트 (machine-readable JSON)
- [ ] static DCA + `targets_override` 실험에 필요한 최소 dataset 집합 명시

### P0 — CPI historical coverage

- [ ] Korean CPI PIT coverage를 2000년대까지 확장 (ECOS ingest)
- [ ] `FIXED_LAG` availability semantics 유지; early session positive CPI row 검증

### P0 — USD/KRW historical coverage

- [ ] FRED/ALFRED FX panel을 2000년대까지 확장
- [ ] execution session FX row 누락 없음 확인

### P0 — Price ingest (research satellites)

- [ ] `QQQ`, `XLI`, `SOXX`, `IBB`, `ITA`, `BOTZ`, `GRID` 장기 가격 history (`ingest history`)
- [ ] `research_satellite_tickers()` universe와 ingest manifest 정합

### P1 — Nasdaq-100 dot-com research proxy

- [ ] 2000 전후 stress-test용 research proxy (기존 `FF_PROXY` / `walk-forward-proxy` 패턴 재사용 검토)
- [ ] QQQ listing 이전 구간 identity isolation; ETF engine과 분리 유지 (I9)

### P1 — 2000s stress regime

- [ ] feasibility audit 통과 후 experiment window를 `2000-01`~ 수준으로 당긴 smoke config
- [ ] Wave 1 `accumulation-cohort` `horizon-months=120` baseline 리포트 (QQQ vs rejected GRID 5%)

### 완료 기준

1. Safe window start가 `2012-06-01`보다 앞당겨짐 (목표: 2000년대 초)
2. `horizon-months=120`, `cohort-step-months=12`에서 cohort `n ≥ 10`
3. Feasibility audit JSON이 operator CLI help 없이 재현 가능

### 예상 산출물

- `src/validation/feasibility_audit.py` (또는 기존 feasibility 확장)
- ingest provider / catalog manifest 갱신
- `configs/experiments/` safe-window JSON start 날짜 정리

---

## 5. Wave 3 — Independent Satellite Test

**전제:** Wave 2 완료 + Wave 1 full 120M engine으로 baseline 증거 갱신.

**규칙:** 단일 satellite → gate 통과 → 조합 후보. 실패 arm은 Wave 4에 넣지 않음.

| 순서 | 실험 | Weight grid | 비고 |
| --- | --- | --- | --- |
| 1 | QQQ + **XLI** | 5 / 10 / 15% | Physical economy 독립효과 |
| 2 | QQQ + **SOXX** | 5 / 10 / 15% | Semiconductor overweight |
| 3 | QQQ + **IBB** | 5 / 10% | Bio diversification |
| 4 | QQQ + **ITA** | 5 / 10% | Defense/aerospace |
| 5 | QQQ + **BOTZ** | 5 / 10% | Robotics pure-play |

**GRID:** v1 reject 유지 — Wave 3 매트릭스에 포함하지 않음.

### Gate (Wave 3 단일 arm)

- CE adoption: `ratio > 1.02` on safe/extended window (36M ablation + 120M cohort report)
- `contiguous_adopted_plateau` / `cohort_win_rate` 기존 gate 재사용
- 80/10/10 혼합 실험 금지 (독립 효과 오염)

### `/spec` 골격 (Wave 2와 병렬 작성 가능)

- [ ] Experiment JSON 템플릿 (`m_qqq_xli.json` 등)
- [ ] Ablation + `accumulation-cohort` wiring
- [ ] Batch runner 또는 campaign 확장 (선택)

---

## 6. Wave 4 — Combination

**전제:** Wave 3에서 CE gate를 통과한 satellite만 조합.

- [ ] 2-satellite combination only (coarse weights)
- [ ] `contiguous_adopted_plateau` — 사후 weight retuning 금지
- [ ] Wave 1 `accumulation-cohort` `horizon-months=120` on each combination candidate
- [ ] GRID 및 Wave 3 실패 satellite 제외

**`/spec`:** Wave 3 결과 테이블(통과 arm ID 목록) 확정 후 작성.

---

## 7. Wave 5 — Operational Validation

**전제:** Wave 4에서 생존한 조합 후보 1개 이상 (없으면 QQQ 유지로 freeze).

| 검증 축 | 내용 |
| --- | --- |
| 120M cohorts | step 1M / 12M / 36M; median, p10, worst, win-rate, bootstrap tail |
| Walk-forward | 기존 `walk-forward` campaign; `process_adopted` |
| Cost grid | Ideal → Base → Stress (`commission_bps`, `fx_spread_bps`) |
| CE γ | γ = 2 / 5 / 10 sensitivity |
| Worst-cohort gate | 최악 cohort ratio floor |
| Tax-aware | 배당세·해외주식 세금 sensitivity (별도 simulation layer) |

**`/spec`:** Wave 4 후보 policy + cost scenario grid 확정 후 작성.

---

## 8. 연구 트리 (목표 상태)

```text
                     QQQ 100%
                        │
              Operational Champion
                        │
        ┌───────────────┴──────────────┐
        │                              │
  Evidence Upgrade              New Satellites
  (Wave 1 ✅, Wave 2)            (Wave 3)
  120M cohort                    XLI / SOXX / IBB / ITA / BOTZ
  pre-2012 data                  (single-arm only)
  dot-com stress
  realistic costs (Wave 5)
        │                              │
        └───────────────┬──────────────┘
                        │
                 Passing arms only (Wave 4)
                        │
                   Combination
                        │
              120M cohort + WF + cost (Wave 5)
                        │
                 CE gate (>1.02)
                        │
              ┌─────────┴─────────┐
              │                   │
            FAIL                PASS
         QQQ 유지            New Policy
```

---

## 9. 즉시 실행 가능한 다음 명령

```bash
# Wave 2 planning
/spec wave2_historical_coverage

# Wave 1 regression (unit)
uv run pytest tests/unit/validation/test_accumulation_cohort.py tests/unit/test_cli.py -k "ACC-COH or CLI-accumulation-cohort"
```

---

## 10. Out of scope (이 로드맵에서 다루지 않음)

- GRID weight 재탐색
- 3+ satellite 동시 조합 탐색 (Wave 4 이전)
- Adoption gate / operational lock 변경
- Objective B (strategic diversification) operational화
- Broker / live execution wiring
