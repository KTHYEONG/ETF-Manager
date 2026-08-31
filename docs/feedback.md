최신 `main`의 HEAD는 **`e73b9d62`**이고, 이전 분석 기준점 `bf654100` 이후 11개 커밋이 추가됐습니다. 이번 개편은 방향 자체는 맞습니다. 다만 **지금 바로 새 전략 개발을 멈추고, 검증 인프라의 4개 핵심 결함만 수정한 뒤 프로젝트를 monitoring 단계로 전환하는 것**이 최적이라고 판단합니다.

## 1. 총평

현재 프로젝트의 상태를 한 문장으로 정리하면:

> **전략 연구 구조는 거의 수렴했지만, “최종 검증”과 “prospective OOS” 구현에 아직 통계적으로 중요한 결함이 남아 있다.**

이전 피드백 중 중요한 부분은 상당히 제대로 반영됐습니다.

* Operational: **QQQ90 / SOXX10 + flat contribution**
* Benchmark: **QQQ100 immutable**
* Adaptive v5: **research-only freeze**
* `≤ 2026-08-28`: seen history
* 미래 산업 thesis: operational alpha generator가 아니라 watch/research
* capital allocation과 deployment timing 분리
* KAFI: 미래 납입금을 당겨 쓰는 adaptive contribution 대신 reserve를 이용한 causal deployment
* Walk-forward reject 시 baseline identity 완전 복귀
* SOXX 후보도 5/10/15로 제한

README에도 이제 프로젝트 목표가 “동일 외부 현금흐름에서 장기 Real KRW terminal wealth 극대화”라고 정확하게 정리돼 있습니다.

특히 이전에 발견했던 WF 오류는 제대로 수정됐습니다. 후보가 train에서 탈락하면 이제 baseline을 새로 조립하지 않고 **`baseline_test_arm` 객체 자체를 chosen으로 사용**합니다.

그리고 KAFI deployment도 현재 월 납입액과 이미 축적된 reserve만 사용하며 미래 현금흐름을 빌리지 않습니다. 이 설계는 맞습니다.

따라서 **프로젝트 철학을 다시 뜯어고칠 필요는 없습니다.**

---

# 2. 하지만 지금 남은 P0 문제가 4개 있다

이 네 가지를 고치기 전에는 `FINAL_HISTORICAL_CAMPAIGN`이나 `PROSPECTIVE_2026_V1`을 최종 증거로 사용하면 안 됩니다.

## P0-1. Final Historical Campaign이 실제로는 10년 cohort 1개다

현재 설정:

```text
start = 2016-07-01
end   = 2026-06-30
horizon = 120 months
step    = 12 months
```

입니다.

그리고 `rolling_cohorts()`는 120개월 전체가 `end` 안에 들어오는 경우에만 cohort를 생성합니다.

따라서 현재 final campaign에서 생성되는 120M cohort는 사실상:

```text
2016-07-01 ~ 2026-06-30
```

**딱 하나입니다.**

코드도 기본값을 그대로 사용합니다.

```python
cohort_horizon_months=120
cohort_step_months=12
```

이건 상당히 큰 문제입니다.

기존 incremental 결과에는 이미 **120개월 cohort 10개**가 있었고:

* QQQ95/SOXX5: median 1.00696
* **QQQ90/SOXX10: 1.01988**
* QQQ85/SOXX15: 1.03202

가 나왔습니다.

그런데 새 “최종 캠페인”이 오히려 이보다 역사 표본을 좁혔습니다.

### 수정

`final_historical_campaign_v1.json`의 시작일을 임의의 2016년으로 고정하면 안 됩니다.

```text
start =
max(
  QQQ 최초 usable date,
  SOXX 최초 usable date,
  USDKRW usable date,
  CPI usable date
)
```

를 자동으로 산출하십시오.

현재 incremental 코드에는 이미 vehicle 최초 PIT 가격까지 cohort 시작점을 clip하는 로직이 있습니다. 이걸 공통화하면 됩니다.

### 최종 historical campaign은

```text
earliest common usable history
                ↓
120M cohorts / 12M step
                ↓
QQQ100
QQQ95/SOXX5
QQQ90/SOXX10
QQQ85/SOXX15
```

이어야 합니다.

**이걸 고치지 않고 final campaign 결과를 해석하면 안 됩니다.**

---

# 3. P0-2. Prospective monitor가 장기 복리를 측정하지 않는다

현재 `run_prospective_monitor()`는:

```python
start = as_of - 30 days
end   = as_of
```

형태로 매번 새로운 allocation을 만듭니다.

예를 들어:

```text
2026-09-30 관측
→ 2026-08-31 부근부터 새 시뮬레이션

2026-10-31 관측
→ 2026-10-01 부근부터 또 새 시뮬레이션

2026-11-30
→ 다시 새 시뮬레이션
```

입니다.

이건 prospective **track record**가 아닙니다.

월별 독립 backtest입니다.

프로젝트의 질문은:

> 10년 이상 적립할 때 복리 terminal wealth가 어떻게 누적되는가?

인데 현재 prospective는 장기 wealth state를 이어가지 않습니다.

더 심각하게는 KAFI deployment에서 reserve도 매번 초기화됩니다.

KAFI의 핵심은:

```text
월 1
일부 투자
일부 reserve 축적
       ↓
월 2
새 납입 + 기존 reserve
       ↓
기회가 좋으면 reserve 추가 투입
```

인데 prospective monitor를 매달 새로 실행하면 이 **reserve state가 다음 달로 전달되지 않습니다.**

따라서 현재 `incumbent_kafi_timing`의 prospective 결과는 전략을 제대로 재현하지 못합니다.

## 수정 방법

가장 단순하고 안전한 방법은 persistent state를 직접 저장하기보다:

```text
prospective_start = 2026-09-01

매월 as_of:
2026-09-01 → as_of
전체 기간을 다시 PIT simulation
```

하는 겁니다.

예:

```text
2026-09-30:
2026-09-01 ~ 2026-09-30

2026-10-31:
2026-09-01 ~ 2026-10-31

2027-08-31:
2026-09-01 ~ 2027-08-31
```

그러면:

* 동일 납입 history
* 기존 positions
* KAFI reserve
* integer lots
* dust
* cumulative compounding

이 모두 자연스럽게 보존됩니다.

### 저장해야 하는 핵심 metric

매 관측마다:

```text
cumulative_terminal_real_krw
cumulative_real_gain
cumulative_xirr_real
cumulative_ratio_vs_qqq100
cumulative_ratio_vs_incumbent
reserve_krw
realized_weights
monthly_incremental_delta
```

를 기록하면 됩니다.

---

# 4. P0-3. Strategy freeze hash가 충분히 freeze하지 않는다

현재 identity payload는 실질적으로:

```text
policy
targets
has_kafi_deployment = true/false
has_adaptive = false
```

만 사용합니다.

따라서 아래 두 전략은 현재 hash가 같을 수 있습니다.

```text
KAFI A:
min=0.7
max=1.3
rank_window=252
```

```text
KAFI B:
min=0.1
max=1.5
rank_window=63
```

둘 다:

```text
has_kafi_deployment = true
```

이기 때문입니다.

이건 prospective freeze의 핵심 목적을 깨뜨립니다.

### Full strategy identity에 최소한 포함해야 하는 것

```text
policy
targets

monthly_contribution
cadence
fill_delay_sessions

commission_bps
fx_spread_bps

full kafi_deployment config
full reserve config
full other active module configs

objective_family
prospective_start
seen_history_cutoff

engine/schema version
```

그리고 개별 arm hash 외에:

```text
bundle_hash
```

도 만드는 것이 좋습니다.

---

# 5. 코드 버전도 freeze에 포함해야 한다

현재 frozen file에는:

```text
git_commit = b99a021222f4
```

가 들어 있습니다.

좋은 시작입니다.

그런데 monitor 실행 시:

> 현재 실행 코드가 그 commit과 같은가?

를 검증하지 않습니다.

즉 strategy JSON은 그대로인데 나중에:

```text
run_allocation()
KAFI score
fill timing
FX handling
reserve logic
```

중 하나가 바뀌면 같은 hash 아래 다른 전략을 실행할 수 있습니다.

### 가장 좋은 방식

prospective observation마다:

```text
frozen_strategy_hash
frozen_bundle_hash
frozen_engine_commit
runtime_engine_commit
data_manifest_hash
```

를 기록합니다.

그리고 runtime commit이 다르면:

```text
behavior_preserving migration
```

을 명시적으로 승인하지 않는 한 fail closed.

실제 운영 엔진 bugfix 때문에 코드 변경이 필요하다면:

```text
PROSPECTIVE_2026_V1_ENGINE2
```

같은 lineage를 남깁니다.

---

# 6. P0-4. 현재 path bootstrap은 DCA용 bootstrap으로는 부정확하다

이 부분은 이번에 가장 주의해서 볼 필요가 있습니다.

현재 `monthly_simple_returns()`는:

```python
cur.mark_krw / prev.mark_krw - 1
```

을 계산합니다.

그런데 `mark_krw`에는 매달 들어온 **새 외부 적립금이 포함**되어 있습니다.

즉 예를 들어:

```text
월초 자산 1,000,000
새 납입 1,000,000
가격 변화 0%
```

인데 단순 NAV ratio만 보면 약:

```text
+100%
```

처럼 보일 수 있습니다.

물론 candidate와 baseline이 같은 납입을 받기 때문에 단순 paired comparison에서는 일부 상쇄되지만,

현재 bootstrap은 이 “cashflow가 섞인 monthly NAV growth”를 block으로 재배열한 뒤 복리합니다.

이건 실제 DCA synthetic path와 동일하지 않습니다.

## 권장 방법

둘 중 하나입니다.

### 방법 A — 최소 수정

external flow를 제거한 unitized return을 만듭니다.

개념적으로:

```text
r_t =
(NAV_t - external_cashflow_t)
/ NAV_(t-1)
- 1
```

execution timing에 맞춰 정확한 식을 정의해야 합니다.

### 방법 B — 더 정확함

아예 underlying joint return block을 bootstrap합니다.

```text
[QQQ return,
 SOXX return,
 USDKRW change]
```

를 12개월 block 단위로 함께 resample하고,

그 synthetic market path 위에서:

```text
월 1M DCA
integer lots
fees
FX
reserve
```

를 다시 실행합니다.

프로젝트의 장기 목적에는 **B가 더 정석**입니다.

개발비는 좀 있지만, 새로운 전략을 하나 더 만드는 것보다 ROI가 훨씬 높습니다.

---

# 7. 그다음 P1 문제

P0 네 개 다음입니다.

## 7.1 Cost stress가 candidate-vs-baseline stress가 아니다

현재 final campaign의:

```text
cost_stress_worst_ratio
```

는 각 arm의:

```text
stress wealth / 그 arm의 ideal wealth
```

를 계산합니다.

즉:

> 이 전략이 비용에 얼마나 민감한가?

는 알 수 있지만,

> 비용을 넣어도 SOXX10이 QQQ100을 이기는가?

는 직접 알 수 없습니다.

최종 보고에는 반드시:

```text
scenario       SOXX10 / QQQ100
ideal
low
base
stress
```

가 필요합니다.

이미 cost-grid에 0/0, 5/10, 10/20, 50/50 bps scenario가 있으므로 재사용하면 됩니다.

---

# 8. Pre-history stress도 현재는 후보 전략을 stress하지 않는다

현재 `audit_pre_history_proxy_stress()`는 `FF_PROXY` 하나를 돌리고:

* terminal wealth
* XIRR

를 출력합니다.

그런데 이것으로는:

> QQQ90/SOXX10이 dot-com에서 QQQ100보다 어땠을까?

를 알 수 없습니다.

즉 현재 pre-history proxy는 **시장 일반 stress**이지 candidate discrimination test가 아닙니다.

여기에는 개발 가치가 있습니다.

가능하면 research-only proxy로:

```text
NASDAQ-100 historical index
Semiconductor historical index
```

를 사용해

```text
QQQ proxy 100
vs
NASDAQ proxy90 + Semiconductor proxy10
```

을 dot-com/GFC까지 확장합니다.

단:

> ETF 실제 execution result와 index proxy result를 절대 같은 evidence tier로 취급하면 안 됩니다.

`proxy_stress_only`로 분리하면 됩니다.

---

# 9. Regime coverage도 기준을 조금 강화해야 한다

현재 regime은 **1개월이라도 겹치면 `covered=True`**입니다.

다행히 `overlap_months`도 출력하므로 정보는 보존됩니다.

하지만 최종 verdict에서 단순 boolean을 사용하지 말고:

```text
full
substantial
partial
none
```

정도로 나누는 것이 낫습니다.

예:

```text
coverage_fraction =
overlap_months / regime_duration
```

그리고 무엇보다 현재 2016 start를 고치면:

* dot-com
* GFC

는 ETF 실데이터 혹은 proxy를 통해 별도로 명시해야 합니다.

---

# 10. Trial lineage 구현은 좋은데 아직 “통계적 trial count”는 아니다

새 코드에서 lineage를 추가한 것은 방향이 맞습니다.

Final config에도:

```text
research_family_id = soxx
related_trial_count = 12
parameter_variants_tried = 3
```

를 기록합니다.

다만 현재 census는 파일 이름으로:

```python
if "adaptive" in filename ...
if "soxx" in filename ...
```

처럼 family를 분류하고 active/archived config 수를 셉니다.

이건 disclosure에는 좋지만:

> 실제로 과거에 몇 개의 parameter 조합을 실행했는가?

는 정확히 알 수 없습니다.

장기적으로는 `make_experiment()` registry에:

```text
research_family_id
parent_experiment_id
strategy_hash
first_seen_at
seen_history_cutoff
```

를 넣고 **실제 실행된 unique strategy hash 수**를 세는 게 맞습니다.

하지만 이건 P0 이후입니다.

---

# 11. Objective Family도 한 군데 더 닫아두는 것이 좋다

현재:

```text
CAPITAL_ALLOCATION
DEPLOYMENT_TIMING
```

으로 나눈 것은 좋습니다.

그리고 adaptive external contribution은 두 family 모두에서 금지했습니다.

하지만 generic invariant에서는:

```text
deployment_timing이면 reserve/KAFI가 있어야 한다
```

만 확인하고,

```text
capital_allocation이면 reserve/KAFI/contribution_shape가 없어야 한다
```

를 완전히 강제하지는 않습니다.

Final historical campaign에서는 별도 validation으로 막고 있어 당장 문제는 없습니다.

그래도 architecture contract 자체를 완전히 닫는 게 좋습니다.

```text
CAPITAL_ALLOCATION
→ flat external flow
→ no reserve
→ no KAFI deployment
→ no contribution shaping

DEPLOYMENT_TIMING
→ fixed external flow
→ reserve/KAFI only
```

로 명확하게 하십시오.

---

# 12. 미래산업 부분은 이제 더 개발하지 않아도 된다

이 부분은 이번 개편으로 거의 원하는 상태에 도달했습니다.

AI Power:

```text
status = dormant
research_role = watch
operational_weight = 0
min_additional_oos_years = 3
```

Physical Automation도 동일합니다.

이게 맞습니다.

### 여기에는 더 이상 개발 시간을 쓰지 않는 것을 권합니다.

즉 하지 않을 것:

* 새로운 AI 전력 ETF 찾기
* robotics ETF 추가
* LLM으로 산업 전망 점수화
* 뉴스 sentiment
* supply chain 예측 모델
* 산업 성장률 forecast
* future theme dynamic allocation

### 앞으로 미래 산업 thesis의 역할

오직:

```text
WATCH
VETO
REOPEN CONDITION
```

입니다.

이 부분은 이제 사실상 **완료**라고 봐도 됩니다.

---

# 13. SOXX thesis와 SOXX operational allocation도 잘 분리했다

`ai_compute` thesis 자체에는:

```text
research_role = challenger
operational_weight = 0
```

가 들어 있습니다.

그런데 actual operational target은 QQQ90/SOXX10입니다.

이게 모순처럼 보이지만 저는 오히려 **올바른 분리**라고 봅니다.

즉:

```text
SOXX10을 보유하는 이유
≠
AI가 미래에 엄청 성장할 것 같아서
```

이고

```text
SOXX10을 보유하는 이유
=
historical empirical evidence에서
QQQ100보다 장기 accumulation 특성이 좋았기 때문
```

입니다.

AI Compute thesis는 그 historical edge가 경제적으로 말이 되는지 관찰하는 secondary evidence일 뿐입니다.

이 철학은 유지하십시오.

---

# 14. 현재 전략 판단은 바꾸지 않는다

현재 push된 결과만으로 operational을 다시 바꿀 이유는 없습니다.

2026-08-28 결과:

| 전략               |     median |        p10 |      worst |     CE γ10 | bootstrap win |        p05 |
| ---------------- | ---------: | ---------: | ---------: | ---------: | ------------: | ---------: |
| QQQ95/SOXX5      |     1.0070 |     1.0030 |     0.9970 |     1.0007 |         71.5% |     0.9903 |
| **QQQ90/SOXX10** | **1.0199** | **1.0070** | **0.9938** | **1.0024** |    **80.25%** | **0.9772** |
| QQQ85/SOXX15     |     1.0320 |     1.0108 |     0.9905 |     1.0034 |        81.75% |     0.9625 |

그리고 새 `economic_effect_passes()`는 최소 median 1.01을 요구합니다.

따라서:

* 5%: economic hurdle 미달
* 10%: 통과
* 15%: 통과

가 됩니다.

현재 10과 15 중 하나를 고른다면 여전히 **10%가 더 방어 가능합니다.**

왜냐하면 15%의 historical 추가 이득은 존재하지만 tail deterioration도 같이 증가하기 때문입니다.

따라서 지금:

```text
QQQ90/SOXX10
→ PROVISIONAL INCUMBENT
```

를 유지하는 것이 맞습니다.

15%로 다시 올리는 개발은 하지 마십시오.

---

# 15. 이제 개발 방향은 3단계로 끝내는 게 좋다

## Phase 1 — Validation Correctness Finalization

**이것만 지금 개발하십시오.**

우선순위:

1. **Prospective cumulative simulation**

   * 30-day reset 제거
   * prospective inception 고정
   * KAFI reserve state 유지

2. **Full strategy/bundle identity hash**

   * KAFI parameter 포함
   * costs/cadence/fill 포함
   * runtime git commit 검증

3. **Final historical cohort range 수정**

   * earliest common usable history
   * 120M rolling cohort 회복

4. **Cashflow-clean bootstrap**

   * 최소 unitized return
   * 가능하면 joint underlying return replay

이 네 개는 ROI가 **매우 높습니다.**

---

# 16. Phase 2 — 딱 한 번의 진짜 Final Historical Campaign

위 네 개를 수정한 뒤:

```text
B0 QQQ100
C1 QQQ95/SOXX5
C2 QQQ90/SOXX10
C3 QQQ85/SOXX15
```

이 네 개만 실행합니다.

**새 weight 추가 금지.**

보고서에는:

* 120M cohorts
* cohort dates
* median
* p10
* worst
* CE γ2/5/10
* corrected bootstrap
* bootstrap p05
* real XIRR
* realized weights
* candidate/base ratio under each cost scenario
* FX stress
* regime coverage
* proxy dot-com/GFC stress
* trial lineage
* tax limitation

를 모두 넣습니다.

그리고 그 결과를 Git에도:

```text
docs/results/final-historical/
FINAL_HISTORICAL_CAMPAIGN_V1_<commit>.json
FINAL_HISTORICAL_CAMPAIGN_V1_<commit>.md
```

형태로 남기십시오.

현재 `docs/results`에는 아직 final-historical 결과 디렉터리가 없습니다. push된 main에서 확인되는 것은 기존 thesis/catalog 결과들입니다.

즉 새 engine은 만들어졌지만 **최종 evidence artifact는 아직 push되지 않은 상태**입니다.

---

# 17. Phase 3 — Prospective로 전환하고 전략 개발 종료

Phase 2가 끝나면 strategy research는 중단하십시오.

구조는:

```text
                 QQQ100
              immutable
                  │
                  ▼
        QQQ90 / SOXX10 flat
        provisional incumbent
                  │
        ┌─────────┴──────────┐
        │                    │
 historical validation   prospective
        │                    │
        ▼                    ▼
    no more tuning      monthly monitoring
                             │
                             ▼
                  long-term OOS evidence
```

KAFI는 별도:

```text
QQQ90/SOXX10 flat
        vs
QQQ90/SOXX10 + causal KAFI reserve
```

만 비교합니다.

---

# 18. KAFI도 더 튜닝하지 말 것

현재 prospective에는:

```text
min_multiplier = 0.7
max_multiplier = 1.3
rank_window = 252
```

가 들어 있습니다.

이 상태에서:

```text
0.6 / 1.4
0.8 / 1.2
rank 126
rank 504
새 component
새 power function
```

등을 다시 테스트하면 이전 adaptive v1~v5와 똑같은 길로 돌아갑니다.

따라서:

> **현재 KAFI deployment config를 마지막 historical design으로 freeze.**

향후 성능이 안 좋으면 개선하는 게 아니라:

> **KAFI timing hypothesis rejected**

로 처리하는 게 맞습니다.

---

# 19. Prospective bundle도 한 가지 구조 변경이 필요하다

현재 loader는 provisional incumbent target이 현재 코드의:

```text
OPERATIONAL_TARGETS_OVERRIDE
```

와 일치하는지 확인합니다.

지금은 괜찮습니다.

그런데 예를 들어 2030년에 operational target이 바뀌면:

```text
OPERATIONAL_TARGETS_OVERRIDE = QQQ85/SOXX15
```

가 될 수 있습니다.

그러면 **2026년에 freeze한 QQQ90/SOXX10 bundle 자체가 validation fail**합니다.

Frozen historical artifact가 현재 production 설정에 의존해서는 안 됩니다.

### 구조를 바꾸십시오

```text
Prospective Bundle V1
= 완전 self-contained immutable artifact
```

와

```text
CURRENT_OPERATIONAL_BUNDLE
= V1을 가리키는 pointer
```

를 분리합니다.

그러면 나중에:

```text
V1 = QQQ90/SOXX10
V2 = 다른 전략
```

이 되어도 V1의 기록은 영원히 재현 가능합니다.

이것도 Phase 1에 같이 넣으면 좋습니다.

---

# 20. 세금은 그 다음이다

현재 final historical은 세금에 대해 명시적으로:

```text
not_modelled
```

이라고 선언합니다.

당장은 괜찮습니다.

하지만 SOXX10의 historical edge가 대략 10년에 +2% 수준이기 때문에 작은 마찰도 중요합니다.

따라서 P0가 끝난 뒤에는:

### 최소 tax model

정도는 구현할 가치가 있습니다.

* dividend withholding
* terminal liquidation capital-gain tax
* annual exemption 같은 사용자별 제도는 parameterization

정도면 충분합니다.

복잡한 세법 engine까지 만들 필요는 없습니다.

---

# 21. 반대로 이제 하지 말아야 하는 개발

여기부터는 명확하게 중단을 권합니다.

| 개발                   | 판단          |
| -------------------- | ----------- |
| SOXX 20/25/30% grid  | **중단**      |
| KAFI v2/v3 튜닝        | **중단**      |
| Adaptive v6          | **중단**      |
| 새로운 미래산업 발굴          | **중단**      |
| PAVE 대체 ETF 탐색       | **중단**      |
| Robotics 대체 ETF 탐색   | **중단**      |
| ML 수익률 예측            | **중단**      |
| LLM 뉴스 alpha         | **중단**      |
| 더 많은 macro feature   | **중단**      |
| 복잡한 regime switching | **중단**      |
| 자동 strategy search   | **강하게 비권장** |

반대로 계속할 것:

| 개발                      | 판단      |
| ----------------------- | ------- |
| Prospective correctness | **최우선** |
| Bootstrap correctness   | **최우선** |
| Historical coverage     | **최우선** |
| Reproducibility/hash    | **최우선** |
| Data refresh            | 지속      |
| Data quality            | 지속      |
| Cost/tax realism        | 보완      |
| Monitoring/reporting    | 지속      |

---

# 22. 프로젝트의 최종 형태

ETF-Manager는 이제 이런 시스템이 되는 게 가장 좋습니다.

```text
┌──────────────────────────┐
│      PIT Data Layer      │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│    DCA Simulation Core   │
│ fixed external cashflow  │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Historical Falsification │
│ cohorts / costs / proxy  │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Frozen Strategy Registry │
│ hashes / commit / lineage│
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│   Prospective OOS Track  │
│ cumulative / immutable   │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ Annual Policy Review     │
└──────────────────────────┘
```

**전략 생성 엔진이 아니라 전략 검증·추적 엔진**입니다.

---

# 23. 최종 개발 우선순위

제가 지금 실제 backlog를 자른다면 이것만 남깁니다.

|       순위 | 작업                                        | 이유                    |
| -------: | ----------------------------------------- | --------------------- |
| **P0-1** | Prospective를 cumulative inception 방식으로 변경 | 현재 OOS 목적 자체가 깨짐      |
| **P0-2** | KAFI 포함 full strategy/bundle hashing      | freeze 신뢰성 문제         |
| **P0-3** | Final campaign history range 확대           | 현재 120M cohort 1개     |
| **P0-4** | DCA cashflow-clean bootstrap              | confidence metric 정확성 |
| **P1-1** | Candidate-vs-baseline cost stress         | 작은 SOXX edge 검증       |
| **P1-2** | NDX/SOX pre-history stress                | dot-com/GFC 검증        |
| **P1-3** | Objective family invariant 강화             | 미래 회귀 방지              |
| **P1-4** | Canonical trial lineage registry          | multiple-testing 공개   |
|   **P2** | 최소 after-tax terminal model               | 실사용 realism           |

**여기까지 끝나면 기능 개발 종료**를 권합니다.

---

# 24. 현재 투자전략 판단

코드가 개편됐지만 **현재 전략 결론은 바꾸지 않습니다.**

```text
QQQ100
= immutable benchmark

QQQ90/SOXX10 flat
= provisional incumbent

QQQ85/SOXX15
= historical aggressive challenger

Adaptive V5
= frozen research only

KAFI deployment
= prospective timing experiment

AI Power
= watch / 0%

Physical Automation
= watch / 0%
```

Operational lock이 실제 코드에서도 flat QQQ90/SOXX10만 붙이도록 정리됐습니다. Adaptive는 붙이지 않습니다.

따라서 **지금 QQQ90/SOXX10을 다시 바꾸지 마십시오.**

---

# 25. 최종 판단

이번 개편으로 프로젝트는 처음으로 **“계속 연구하면 더 좋아질 것”이라는 단계에서 벗어났습니다.**

이제 ROI 구조가 명확합니다.

```text
새 전략 추가 ROI        ↓↓↓
새 예측 모델 ROI        ↓↓↓
미래산업 연구 ROI       ↓↓↓

검증 정확도 ROI         ↑↑↑
재현성 ROI              ↑↑↑
prospective OOS ROI      ↑↑↑
```

따라서 앞으로의 방향은:

> **앞으로 더 좋은 전략을 찾아다니는 것이 아니라, 현재 살아남은 QQQ90/SOXX10이 정말 살아남는지를 검증하는 데 전력을 집중한다.**

입니다.

그리고 위의 **P0 네 가지 + P1 핵심 보완**까지 끝내면, 이 프로젝트에서는 의도적으로 개발 속도를 낮추는 것이 맞습니다. 이후에는 월별 데이터 갱신과 prospective monitoring, 연 1회 validation review 정도만 유지하는 것이 장기 복리 목표에 가장 적합합니다.

현재는 **“새로운 알파 기능을 추가할 단계”가 아니라 “연구를 끝낼 수 있게 만드는 마지막 correctness 단계”**입니다.
