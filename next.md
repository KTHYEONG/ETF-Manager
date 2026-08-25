# ETF-Manager Final Development Roadmap

> 목적: 현재 `S8_US_NASDAQ (QQQ 100%)` 적립식 전략을 기준으로, 동일한 장기 외부 현금흐름 안에서 **투자 시점, 투자 강도, 위험 노출, 포트폴리오 배분을 동적으로 제어하여 실질 장기 자산을 최대화하는 adaptive accumulation system**으로 발전시킨다.

---

## 0. 문서 사용 목적

이 문서는 다른 AI/개발자가 ETF-Manager의 다음 개발 단계를 계획하고 구현할 때 사용할 **최상위 로드맵**이다.

핵심 원칙은 다음과 같다.

1. 새로운 ETF/전략을 무작정 추가하지 않는다.
2. 현재 operational policy인 `S8_US_NASDAQ`을 신규 실험의 기준점으로 삼는다.
3. 외부 현금흐름은 후보 간 동일하게 유지한다.
4. 새로운 기능은 반드시 독립적으로 검증한 뒤 조합한다.
5. in-sample 성과가 아니라 walk-forward / rolling cohort / bootstrap / cost stress에서 살아남는지 평가한다.
6. 단순 terminal wealth 증가만 보지 않고 catastrophic risk를 함께 통제한다.
7. 매도는 마지막 단계로 미룬다.

---

# 1. 현재 프로젝트 상태 요약

현재 ETF-Manager는 단순 DCA 백테스터가 아니다.

구조적으로 이미 다음 계층이 분리되어 있다.

```text
Data
  ↓
Features
  ↓
Policy
  ↓
Simulation
  ↓
Validation
  ↓
Analytics
  ↓
Execution
```

현재 주요 상태:

- Operational policy: `S8_US_NASDAQ`
- Vehicle: `QQQ 100%`
- External contribution: fixed monthly KRW
- Rebalancing: buy-only
- Production sell logic: 없음
- Validation: walk-forward / cohort / CE gate / cost grid / bootstrap 기반 구조 보유
- Dynamic modules:
  - overlay
  - reserve
  - currency defer
  - ETF mapping
  - cadence
- Portfolio diagnostics:
  - QQQ / VTI / IEF blend 실험 기반 존재

즉 다음 단계의 핵심은 **ETF 선택 문제를 반복하는 것이 아니라 QQQ 중심의 자본 배분 제어 문제로 전환하는 것**이다.

---

# 2. 최종 시스템 정의

ETF-Manager의 최종 목표를 다음처럼 정의한다.

> **QQQ를 핵심 성장 자산으로 사용하면서, 동일한 월별 저축 예산 안에서 시장 상태에 따라 투자 금액, 현금 reserve, 위험 노출, 포트폴리오 비중을 동적으로 조정해 장기 실질 자산을 최대화하는 PIT-safe adaptive accumulation system**

최종 아키텍처 개념:

```text
                    External Monthly Savings
                           1,000,000 KRW
                                  │
                                  ▼
                    ┌─────────────────────┐
                    │ Contribution Control │
                    │                     │
                    │ How much to deploy? │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
                  Invest              Reserve
                     │                   │
                     └─────────┬─────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Exposure Controller │
                    │                     │
                    │ QQQ 80~100% etc.    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Portfolio Allocator │
                    │                     │
                    │ QQQ / VTI / IEF     │
                    └──────────┬──────────┘
                               ▼
                            Execution
                               │
                               ▼
                             Ledger
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Validation System   │
                    │                     │
                    │ WF / Cohort / BS    │
                    │ Cost / CE / Wealth  │
                    └─────────────────────┘
```

---

# 3. 연구 목표 재정의

## 기존 질문

```text
매월 동일한 금액을 투자할 때 어떤 ETF/포트폴리오가 가장 좋은가?
```

## 앞으로의 질문

```text
동일한 장기 저축 능력 아래에서,
언제 얼마를 QQQ에 투자하고,
얼마를 reserve로 보유하고,
언제 위험 노출을 줄이고,
필요하다면 어떤 자산에 분산해야
장기 실질 terminal wealth를 최대화할 수 있는가?
```

---

# 4. 절대 유지해야 할 연구 원칙

## 4.1 동일 외부 현금흐름

모든 후보 전략은 동일한 외부 KRW cashflow를 사용해야 한다.

잘못된 비교:

```text
A: 매월 100만원
B: 평소 100만원 + 폭락 때 외부 자금 100만원 추가
```

B가 더 많은 자금을 투입했으므로 전략 비교가 아니다.

올바른 비교:

```text
매월 외부 유입 = 100만원 고정

전략 A
→ 100만원 전부 즉시 투자

전략 B
→ 일부 reserve 적립
→ 향후 drawdown에서 reserve 재투입
```

외부 자본 총량은 동일해야 한다.

---

## 4.2 PIT(Point-in-Time) 원칙

모든 decision signal은 해당 시점에 실제로 이용 가능했던 데이터만 사용한다.

금지:

- 미래 수정치 사용
- future close 사용
- 전체 기간 기준 normalization
- global forward-fill
- future regime label 사용

---

## 4.3 전략 복잡도 페널티 유지

복잡한 모듈을 추가할수록 adoption hurdle을 높인다.

```text
simple candidate
→ 낮은 hurdle

multi-module candidate
→ 높은 hurdle
```

현재의 module-count 기반 CE gate 철학을 유지한다.

---

## 4.4 독립 검증 → 조합 검증

처음부터 다음처럼 만들지 않는다.

```text
QQQ + reserve + overlay + portfolio + currency + cadence
```

이렇게 하면 어떤 요소가 성과를 만들었는지 알 수 없다.

순서:

```text
1. single module validation
2. adopted modules only
3. pairwise combination
4. full-stack challenger
```

---

# 5. 최종 추천 개발 우선순위

| Priority | Phase | 핵심 목적 |
|---|---|---|
| P0 | S8 연구 기준 통일 | 신규 연구 baseline을 QQQ로 전환 |
| P1 | S8 + Reserve V1 | 동적 적립의 기본 유효성 검증 |
| P2 | Reserve V2 | 0.8x~2.0x 수준의 연속적 자금 배분 |
| P3 | S8 + Overlay | 위험 노출 축소 효과 검증 |
| P4 | Portfolio Challenger | QQQ/VTI, QQQ/IEF 후보 승격 검토 |
| P5 | Module Combination | 채택된 모듈끼리 조합 검증 |
| P6 | Long-History Stress | 2000~2002 포함 QQQ robustness 강화 |
| P7 | Sell/Rebalance Engine | 최종 단계의 위험 기반 매도 시스템 |

---

# 6. P0 — S8를 연구 기준으로 완전히 전환

## 목표

신규 연구의 baseline을 기존 `S1_US` 중심에서 `S8_US_NASDAQ` 중심으로 이동한다.

## 해야 할 일

- 신규 experiment config는 기본적으로 `S8_US_NASDAQ`을 baseline으로 사용한다.
- 기존 S1 기반 overlay/reserve/cadence/currency 실험은 historical reference로 보존한다.
- 동일 기능에 대해 S8 전용 config를 새로 생성한다.

예시:

```text
wf_s8_reserve.json
wf_s8_overlay.json
wf_s8_cadence.json
wf_s8_currency.json
```

## 완료 조건

- S8 baseline config가 walk-forward에서 정상 동작
- 모든 신규 challenger가 S8과 동일한 cashflow/window/cost 조건 사용
- 기존 S1 실험과 이름/결과가 혼동되지 않음
- operational policy 변경 없이 연구 baseline만 정렬

---

# 7. P1 — S8 + Reserve V1 정식 검증

## 목적

현재 구현된 reserve 기능이 QQQ에서도 실제 OOS 성과 개선을 만드는지 확인한다.

현재 reserve 개념:

```text
positive trend
→ 일부 contribution reserve 보류

drawdown <= threshold
→ reserve 일부 추가 투입

otherwise
→ normal contribution
```

## 실험

```text
Baseline
S8 fixed DCA

Candidate
S8 + Reserve V1
```

## 검증 항목

- Walk-forward
- Rolling cohort
- Cost grid
- Bootstrap
- Real terminal wealth
- XIRR real
- MDD
- worst cohort
- reserve utilization ratio
- average cash drag

## 반드시 추가할 diagnostic

```text
reserve_balance_over_time
withheld_total
redeployed_total
reserve_idle_months
reserve_deployment_events
extra_investment_ratio
```

## 완료 조건

Reserve V1은 다음을 모두 만족해야 한다.

```text
1. OOS terminal wealth 개선
2. worst cohort 악화가 허용 범위 내
3. cost stress에서도 우위 유지
4. 특정 단일 crash에만 의존하지 않음
5. reserve cash drag가 장기 성과를 과도하게 훼손하지 않음
```

실패할 경우 Reserve V1은 operational candidate로 승격하지 않는다.

---

# 8. P2 — Reserve V2 설계

Reserve V1의 가장 큰 한계는 binary rule과 좁은 범위다.

현재 구조는 사실상:

```text
0.9x / 1.0x / 1.1x
```

에 가까운 동작만 한다.

최종적으로는 다음 구조가 더 적합하다.

```text
0.75x
1.00x
1.25x
1.50x
2.00x
```

단, 추가 투자분은 반드시 reserve에서만 가져온다.

---

## 8.1 권장 ReserveConfig V2

예시:

```python
ReserveConfig(
    reserve_target_months=3.0,
    reserve_max_months=6.0,
    min_invest_multiplier=0.80,
    max_invest_multiplier=2.00,
    drawdown_levels=(-0.10, -0.20, -0.30),
)
```

### 개념

```text
Normal
→ 1.0x

High valuation / strong trend / low drawdown
→ 0.8~0.9x

-10% drawdown
→ 1.25x

-20% drawdown
→ 1.50x

-30% drawdown
→ up to 2.0x
```

---

## 8.2 반드시 지킬 제약

```text
external monthly cashflow fixed
reserve >= 0
reserve <= configured maximum
no hidden borrowing
no negative cash
no leverage
```

---

## 8.3 Binary threshold 제거

다음 구조를 피한다.

```text
-14.9% → no action
-15.0% → action
```

가능하면 piecewise 또는 continuous scoring을 사용한다.

예:

```text
score = f(drawdown, trend, volatility)
multiplier = clip(1 + score, min_multiplier, max_multiplier)
```

단, 모델 복잡도는 최소화한다.

---

# 9. P3 — S8 + Overlay 정식 검증

## 역할 정의

Reserve와 Overlay의 역할을 명확히 분리한다.

```text
Reserve
→ 언제 더 많이 투자할 것인가

Overlay
→ 언제 위험 노출을 줄일 것인가
```

QQQ 100%에서는 positive overlay가 사실상 leverage로 확대되지 않고 normalization되므로, overlay의 실질 기능은 defensive exposure control이다.

예:

```text
Normal
QQQ 100%

Risk-off
QQQ 90% + cash 10%

Strong risk-off
QQQ 80% + cash 20%
```

## 사용 가능한 기존 feature

- trend
- volatility
- drawdown
- VIX

## 실험

```text
Baseline
S8 fixed DCA

Candidate
S8 + Overlay
```

## 검증 목표

Overlay는 반드시 다음 중 하나 이상의 경제적 이점을 보여야 한다.

```text
- terminal wealth 개선
- catastrophic MDD 감소
- worst cohort 개선
- risk-adjusted CE 개선
```

단순 MDD 개선만 있고 terminal wealth가 크게 감소하면 growth-first 목표와 맞지 않으므로 adoption하지 않는다.

---

# 10. P4 — Portfolio Challenger 연구

현재 QQQ/VTI, QQQ/IEF blend는 diagnostic 단계로 유지한다.

후보 예시:

```text
QQQ 90 / VTI 10
QQQ 80 / VTI 20
QQQ 80 / IEF 20
QQQ 70 / IEF 30
```

## 원칙

모든 blend를 바로 새로운 PolicyId로 승격하지 않는다.

순서:

```text
analytics diagnostics
    ↓
regime robustness
    ↓
후보 1~2개 shortlist
    ↓
formal PolicyId candidate
    ↓
walk-forward
    ↓
cohort
    ↓
bootstrap
    ↓
adoption gate
```

## 성장 목표 주의

채권 혼합은 다음을 개선할 수 있다.

```text
MDD
volatility
tail risk
```

하지만 반드시 terminal wealth를 증가시키는 것은 아니다.

따라서 최종 선택 기준은 단순 Sharpe가 아니라 다음이어야 한다.

```text
growth benefit
+ acceptable risk reduction
```

---

# 11. P5 — 채택된 모듈 조합

독립적으로 채택된 모듈만 조합한다.

예:

```text
Phase A
S8 vs S8 + Reserve

Phase B
S8 vs S8 + Overlay

Phase C
S8 + Reserve
vs
S8 + Reserve + Overlay
```

최종적으로:

```text
S8 fixed
vs
S8 + Reserve
vs
S8 + Overlay
vs
S8 + Reserve + Overlay
vs
S8 + Reserve + Overlay + Portfolio
```

## ExperimentSpec 변경 방향

현재 module mutual exclusivity는 독립 검증 단계에서는 유지한다.

조합 실험 전용 schema를 별도로 허용한다.

권장:

```text
single_module experiment
combined_modules experiment
```

을 구분한다.

조합 단계에서만 다음을 허용한다.

```text
reserve + overlay
reserve + portfolio
reserve + overlay + portfolio
```

단, module count는 명시적으로 증가시킨다.

---

# 12. P6 — QQQ Long-History Stress Validation

QQQ를 장기 core로 사용할 경우 현재 데이터 기간만으로는 충분하지 않을 수 있다.

특히 반드시 검토해야 할 시장:

```text
2000~2002 dot-com collapse
2007~2009 GFC
2020 COVID crash
2022 rate shock
```

현재 장기 진단 구간이 2006년 이후라면 dot-com 붕괴를 직접 포함하지 못한다.

## 해야 할 일

가능하면 QQQ historical dataset을 1999~2000년대 초반까지 확장한다.

반드시 확인:

- adjusted/raw dividend 처리 일관성
- split 처리
- survivorship issue
- data availability timestamp
- FX history
- CPI history
- transaction-cost assumptions

## 추가 validation

```text
rolling start-date cohorts
multiple horizon lengths
block bootstrap
regime-specific breakdown
cost stress
worst-5% outcomes
```

---

# 13. 평가 목적 함수 개편

현재 CE 기반 validation은 유지한다.

하지만 최종 목표가 “자산 증식”이라면 growth-first 구조를 명확히 한다.

권장 최종 목적:

```text
maximize:
    OOS real terminal wealth

subject to:
    catastrophic drawdown constraint
    worst-cohort constraint
    bootstrap-tail constraint
    cost robustness
```

## Primary metrics

```text
Real Terminal Wealth
Median OOS Terminal Wealth
Real XIRR
```

## Risk constraints

```text
Max Drawdown
Worst Cohort Terminal Wealth
5% Bootstrap Tail Wealth
Time Under Water
```

## Secondary metrics

```text
CE gamma
Volatility
Cash Drag
Turnover
Cost Drag
Tax Drag
```

---

# 14. 매도는 마지막 단계로 유지

단순 profit-taking 규칙은 우선순위가 낮다.

예:

```text
+30% 수익 → 익절
```

은 다음 문제를 만든다.

```text
복리 중단
거래비용
세금
재진입 문제
```

매도 전략은 `profit taking`이 아니라 `risk exposure management` 관점에서 설계한다.

예:

```text
QQQ 100%
→ extreme risk regime
→ QQQ 80% + cash 20%

risk normalized
→ QQQ 100%
```

---

# 15. P7 — Sell / Rebalance Engine

기존 buy-only engine을 직접 훼손하지 않는다.

별도 구조를 추가한다.

```text
Target Allocation
      ↓
Rebalance Decision
      ↓
TradeIntent
   ├─ BUY
   └─ SELL
      ↓
TaxLotSelector
      ↓
Execution
      ↓
Ledger
```

필요 모듈 예시:

```text
policy/sell.py
sim/rebalance.py
sim/tax_lots.py
execution/trade_intent.py
```

## 매도 validation 필수 조건

매도 전략은 반드시 full simulation을 사용한다.

반영해야 할 요소:

```text
capital gain tax
trade FX
commission
spread
lot basis
partial fills
cash settlement
```

매도 전략을 fast engine 결과만으로 adoption하지 않는다.

---

# 16. 하지 말아야 할 것

## 16.1 ETF universe 무한 확장

현재 단계에서 수십 개의 ETF를 더 추가하지 않는다.

이유:

```text
search space explosion
data snooping
multiple comparison problem
interpretability 감소
```

---

## 16.2 ML 기반 매수/매도 예측 선행

현재는 필요 없다.

먼저 단순하고 설명 가능한 feature 조합을 검증한다.

```text
trend
volatility
drawdown
VIX
FX
```

이 수준에서 robust edge가 확인되지 않으면 ML은 더 높은 과적합 위험만 만든다.

---

## 16.3 레버리지

현 단계에서 leverage를 도입하지 않는다.

동적 추가 투자는 reserve ledger 범위 내에서만 수행한다.

---

## 16.4 In-sample parameter tuning

다음 식의 exhaustive search를 피한다.

```text
MA 50/100/150/200
DD 5/10/15/20/25%
VIX 15/20/25/30/35
...
```

권장:

- economically justified small parameter set
- nested walk-forward 또는 fixed candidate grid
- OOS result 기준 선택

---

# 17. 구현 순서 상세

## Phase 0 — Baseline alignment

- [ ] 신규 experiment baseline을 S8로 통일
- [ ] S8 reserve config 추가
- [ ] S8 overlay config 추가
- [ ] S8 cost-grid config 추가
- [ ] experiment naming convention 정리
- [ ] docs synchronization

## Phase 1 — Reserve V1

- [ ] S8 + Reserve WF 실행
- [ ] reserve diagnostic metrics 추가
- [ ] cohort validation
- [ ] cost-grid validation
- [ ] bootstrap validation
- [ ] adoption report 생성

## Phase 2 — Reserve V2

- [ ] multiplier model 설계
- [ ] reserve capacity 모델 설계
- [ ] piecewise/continuous drawdown response
- [ ] hard cash invariants 테스트
- [ ] PIT leakage 테스트
- [ ] S8 fixed vs V1 vs V2 비교

## Phase 3 — Overlay

- [ ] S8 overlay experiment
- [ ] risk exposure logging
- [ ] cash residual accounting 검증
- [ ] VIX optional gate OOS 검증
- [ ] terminal wealth vs MDD tradeoff 분석

## Phase 4 — Portfolio

- [ ] blend diagnostics 결과 정리
- [ ] shortlist 1~2개
- [ ] formal candidate 생성
- [ ] WF / cohort / bootstrap
- [ ] growth-risk frontier 생성

## Phase 5 — Module composition

- [ ] combined experiment schema 설계
- [ ] reserve + overlay
- [ ] reserve + portfolio
- [ ] reserve + overlay + portfolio
- [ ] complexity penalty 적용
- [ ] ablation decomposition

## Phase 6 — Long history

- [ ] QQQ history 확장 검토
- [ ] 2000~2002 stress 포함
- [ ] data integrity tests
- [ ] extended cohorts
- [ ] long-history bootstrap

## Phase 7 — Sell engine

- [ ] TradeIntent abstraction
- [ ] sell policy
- [ ] tax lot support
- [ ] full-mode sell simulation
- [ ] risk-reduction sell candidate
- [ ] no naive profit-taking rule

---

# 18. 각 단계의 공통 Acceptance Gate

새로운 전략/모듈은 아래 조건을 통과하기 전 operational policy에 반영하지 않는다.

## Functional

- [ ] unit tests pass
- [ ] integration tests pass
- [ ] deterministic reproducibility
- [ ] no silent fallback
- [ ] all invalid states fail closed

## Data

- [ ] PIT invariant 유지
- [ ] future leakage 없음
- [ ] missing data silent imputation 없음
- [ ] manifest / dataset hash 기록

## Accounting

- [ ] cash conservation
- [ ] reserve conservation
- [ ] no hidden external inflow
- [ ] no negative reserve
- [ ] fees explicitly accounted

## Research

- [ ] identical cashflows
- [ ] walk-forward OOS
- [ ] rolling cohorts
- [ ] cost stress
- [ ] bootstrap
- [ ] result registry 저장

## Adoption

- [ ] baseline 대비 meaningful improvement
- [ ] worst-case degradation controlled
- [ ] improvement not isolated to one regime
- [ ] complexity hurdle passed

---

# 19. 권장 Experiment Naming Convention

```text
wf_s8_reserve_v1
wf_s8_reserve_v2
wf_s8_overlay
wf_s8_portfolio_q90v10
wf_s8_portfolio_q80i20
wf_s8_reserve_overlay
wf_s8_full_stack
```

Cost grid:

```text
cg_s8_reserve_v1
cg_s8_overlay
cg_s8_full_stack
```

Diagnostics:

```text
diag_s8_reserve_usage
diag_s8_regimes
diag_s8_portfolio_frontier
```

---

# 20. 결과 저장에 반드시 포함할 메타데이터

모든 실험 결과에 최소 다음 항목을 기록한다.

```text
experiment_id
git_commit
config_hash
dataset_manifest_hash
baseline_policy
candidate_policy
modules
start
end
train_months
test_months
contribution_krw
commission_bps
fx_spread_bps
terminal_wealth_real_krw
xirr_real
max_drawdown
worst_cohort
bootstrap_p05
CE_gamma_2
CE_gamma_5
CE_gamma_10
adopted
```

Reserve candidate:

```text
reserve_avg
reserve_max
reserve_utilization
withheld_total
redeployed_total
```

Overlay candidate:

```text
avg_equity_exposure
min_equity_exposure
risk_off_months
cash_residual_avg
```

---

# 21. AI 작업 지침

이 로드맵을 참고하여 작업하는 AI는 다음 절차를 따른다.

## 작업 전

1. 현재 `main` HEAD 확인
2. 관련 architecture docs 확인
3. 관련 config와 validation runner 확인
4. 기존 테스트 확인
5. 현재 behavior를 변경하지 않고 필요한 extension point부터 식별

## 구현 중

1. 기존 invariant를 깨지 않는다.
2. 하나의 PR/작업 단위에는 하나의 경제적 가설만 포함한다.
3. 새로운 parameter는 최소화한다.
4. magic threshold를 추가할 경우 경제적 근거를 문서화한다.
5. 테스트 없는 전략 로직을 추가하지 않는다.

## 구현 후

다음 순서로 검증한다.

```text
unit
→ integration
→ deterministic simulation
→ invariant tests
→ experiment smoke test
→ walk-forward
→ cohort
→ cost-grid
→ bootstrap
→ docs sync
```

---

# 22. 최종 목표 상태

최종 operational candidate는 다음과 같은 형태가 될 수 있다.

```text
External contribution
    = 1,000,000 KRW/month fixed

Core asset
    = QQQ

Contribution controller
    = adaptive reserve-based deployment

Exposure controller
    = bounded risk reduction

Portfolio allocator
    = QQQ-dominant, only if validated

Sell logic
    = optional risk-control layer, only after full-mode validation
```

그러나 최종 전략을 미리 정답으로 가정하지 않는다.

모든 모듈은 다음 기준으로 탈락할 수 있어야 한다.

```text
If it does not survive OOS validation,
remove it.
```

ETF-Manager의 최종 품질은 기능 수가 아니라 **살아남은 규칙의 수가 적고, 각 규칙의 효과가 검증되어 있는가**로 판단한다.

---

# 23. 최종 개발 방향 한 문장

> **ETF-Manager는 QQQ를 중심으로 동일한 장기 저축 예산 안에서 자금 투입 강도, 현금 reserve, 위험 노출, 포트폴리오 배분을 PIT-safe하게 동적으로 제어하고, 모든 개선안을 walk-forward·cohort·bootstrap·cost stress로 검증하는 growth-first adaptive accumulation platform으로 발전시킨다.**

