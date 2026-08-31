# 결론부터

**프로젝트 자체를 멈추는 것은 아깝습니다. 그러나 지금부터는 개발 방향을 크게 바꾸는 것이 맞습니다.**

제가 현재 `main`의 전체 트리, 핵심 `policy / sim / validation / thesis / compound_dca` 로직, 최신 실험 설정과 2026-08-28 결과물을 다시 기준점 없이 검토한 결론은 다음입니다. 이번 세션에서는 저장소 코드를 직접 실행해 전체 테스트/데이터 파이프라인을 재실행한 것은 아니며, 현재 `main` 소스와 push된 결과 artifact를 정적 분석한 결과입니다. 현재 저장소는 이미 상당한 PIT 데이터·시뮬레이션·검증·thesis·WF 인프라를 갖춘 상태입니다.

> **ETF-Manager의 다음 단계는 "더 똑똑한 미래예측 시스템"이 아니라 "더 이상 스스로를 속이지 않는 장기 복리 전략 검증 시스템"이어야 합니다.**

구체적으로는:

* **미래산업 예측을 operational alpha의 중심에서 내린다.**
* **백테스트를 중심 증거로 사용하되, 백테스트만으로 최종 채택하지 않는다.**
* `QQQ100 flat`을 영구 benchmark로 둔다.
* 현재 가장 유의미한 실증 challenger는 `QQQ90 / SOXX10 flat`.
* `QQQ85 / SOXX15`는 더 공격적인 challenger일 뿐, 현재 데이터에서 10%보다 명백하게 우월하다고 보기 어렵다.
* 현재 operational adaptive contribution은 **장기 적립 asset-allocation 문제와 분리해야 한다.**
* 새로운 테마/ETF/파라미터 탐색은 사실상 중단한다.
* 남은 개발 ROI는 **검증 오류 수정 → 연구 과적합 방어 → 과거 regime 확대 → 진짜 prospective OOS** 순으로 높습니다.
* 이것까지 한 뒤에는 기능 개발을 거의 멈추고 **monitoring/revalidation 시스템**으로 전환하는 편이 낫습니다.

---

# 1. 현재 `main`은 이전 프로젝트와 꽤 달라졌다

가장 먼저 중요한 사실이 있습니다.

현재 코드의 operational target은 더 이상 QQQ100이 아닙니다.

```text
OPERATIONAL_POLICY_ID = QQQ
OPERATIONAL_TARGETS_OVERRIDE = {
    QQQ: 0.9,
    SOXX: 0.1
}
```

즉 이름은 `PolicyId.QQQ`이지만 실제 target은 **QQQ90/SOXX10**입니다.

반면 일부 과거 문서에는 여전히 `QQQ100 operational`이라고 적힌 것이 남아 있습니다. 예컨대 2026-08-30 AI Compute 문서도 당시에는 QQQ100 불변을 명시했습니다. 따라서 이제는 **코드가 문서보다 앞서간 상태**입니다.

이것부터 정리해야 합니다.

그리고 현재 프로젝트에는 이미:

* `compound_dca`
* adaptive contribution v5
* QQQ/SOXX mix
* risk budget
* tournament
* walk-forward
* strategy selection
* long-horizon cohort
* path bootstrap
* thesis fundamentals
* structural evidence
* valuation/crowding
* prospective research

등 상당한 양의 연구 기능이 들어가 있습니다.

문제는 이제 **기능 부족이 아닙니다.**

오히려 반대입니다.

> **연구 자유도가 데이터가 제공하는 정보량보다 빨리 증가하고 있습니다.**

이게 현재 프로젝트의 가장 큰 위험입니다.

---

# 2. 지금까지 나온 결과를 완전히 다시 해석하면

## 2.1 AI Compute / SOXX는 실제 신호가 있다

최신 2026-08-28 fresh panel에서 QQQ에 SOXX를 추가한 120개월 결과는 다음과 같습니다.

| 전략                 | 120M median |        p10 |      worst | cohort win | Bootstrap win | Bootstrap p05 |    CE γ=10 |
| ------------------ | ----------: | ---------: | ---------: | ---------: | ------------: | ------------: | ---------: |
| QQQ95 / SOXX5      |      1.0070 |     1.0030 |     0.9970 |        90% |         71.5% |        0.9903 |     1.0007 |
| **QQQ90 / SOXX10** |  **1.0199** | **1.0070** | **0.9938** |    **90%** |    **80.25%** |    **0.9772** | **1.0024** |
| QQQ85 / SOXX15     |      1.0320 |     1.0108 |     0.9905 |        90% |        81.75% |        0.9625 |     1.0034 |

이건 단순 노이즈라고 버릴 정도로 약하지 않습니다.

특히 5→10→15%로 갈수록 historical median이 단조 증가하는 것은 계속 유지되고 있습니다.

하지만 숫자의 크기도 냉정하게 봐야 합니다.

### QQQ90/SOXX10

10년에 terminal wealth 약 +1.99%.

연환산 상대 차이로 환산하면 대략:

> **연 +0.20% 수준**

입니다.

### QQQ85/SOXX15

10년에 약 +3.20%.

대략:

> **연 +0.32% 수준**

입니다.

즉 엄청난 alpha가 아닙니다.

---

# 3. 그래서 SOXX 15%보다 10%가 현재는 더 방어 가능하다

15%의 median은 더 좋습니다.

하지만 bootstrap tail은:

* SOXX10: `0.9772`
* SOXX15: `0.9625`

까지 떨어집니다.

즉 SOXX를 10→15%로 늘리면:

* median: 약 +1.21%p 추가
* bootstrap p05: 약 -1.47%p 악화

합니다.

더 중요하게 γ=10 CE를 보면:

* 10%: 1.00242
* 15%: 1.00339

입니다.

위험 회피를 강하게 반영하면 **SOXX 5%p를 더 넣어서 얻는 효용 차이가 매우 작아집니다.**

따라서 현재 데이터만 놓고 순위를 매기면:

> **QQQ90/SOXX10 = 가장 defensible한 empirical challenger**

입니다.

QQQ85/SOXX15가 잘못됐다는 뜻은 아닙니다.

다만:

> `10% → 15% → 20% → 25% ...`

식으로 결과를 보고 계속 확대하면 그 순간부터 거의 전형적인 backtest optimization으로 들어갑니다.

---

# 4. 반대로 AI Power 결과는 사실상 "없다"

현재 GRID 대신 PAVE까지 재검토했지만 결과가 중요합니다.

QQQ90/PAVE10의 96M 결과:

* median = **1.00074**
* cohort n = **2**
* win rate = 50%
* CE γ10 = **0.99913**
* bootstrap win = 56%
* bootstrap p05 = **0.9436**

입니다.

이건 실질적으로:

> **alpha evidence 없음**

에 가깝습니다.

특히 8년 동안 terminal wealth가 +0.074% 정도 좋은 것을 `historically_promising`이라고 부르는 것은 현재 상태 classifier가 너무 관대합니다.

### 이 부분은 로직을 바꾸는 게 좋습니다.

단순히:

```text
bootstrap_win_rate > threshold
```

만으로 promising 판정을 만들면 안 됩니다.

반드시 **economic effect size hurdle**이 필요합니다.

즉:

```text
통계적으로 50%보다 약간 좋은가?
```

뿐 아니라

```text
틀릴 위험을 감수할 만큼 경제적으로 큰 차이인가?
```

를 봐야 합니다.

PAVE는 현재 후자를 통과하지 못합니다.

---

# 5. Physical Automation은 더 명확하다

ROBO 결과는:

| SOXX 대신 ROBO 비중 | 120M median | win rate | bootstrap win |
| --------------- | ----------: | -------: | ------------: |
| 5%              |      0.9824 |       0% |          5.0% |
| 10%             |      0.9587 |       0% |         4.75% |
| 15%             |      0.9343 |       0% |          4.5% |

입니다.

여기는 해석이 훨씬 쉽습니다.

> **ROBO vehicle은 현재 투자 후보에서 제외하는 것이 맞습니다.**

그리고 이 결과를:

> 로봇 산업은 미래에도 실패한다.

로 읽으면 안 됩니다.

정확한 결론은:

> **현재까지 실제 상장시장에서 ROBO를 이용해 Physical Automation exposure를 overweight하는 전략은 QQQ보다 장기 복리 증식에 도움이 되었다는 증거가 없다.**

여기까지만 말할 수 있습니다.

---

# 6. 그러면 "미래산업 예측"은 멍청한 짓인가?

여기서는 두 극단을 모두 피해야 합니다.

### ① "미래 산업은 예측 가능하니 선점하면 된다"

이건 현재 증거로 지지하기 어렵습니다.

### ② "미래는 절대 알 수 없으니 fundamentals는 모두 쓸모없다"

이것도 지나칩니다.

문제는 다음 논리입니다.

```text
AI 전력 수요가 크게 증가한다
        ↓
전력 인프라 산업이 성장한다
        ↓
관련 회사 매출이 증가한다
        ↓
PAVE/GRID 가격이 QQQ보다 더 오른다
```

각 화살표가 자동으로 성립하지 않습니다.

산업이 폭발적으로 성장해도:

* 이미 가격에 반영됐을 수 있고
* 경쟁으로 마진이 낮아질 수 있고
* 신규 공급이 증가할 수 있고
* ETF가 잘못된 기업을 담을 수 있고
* 산업 성장의 경제적 rent를 다른 회사가 가져갈 수 있고
* 성장주는 이미 엄청난 valuation premium을 받을 수 있습니다.

기술혁명과 주가의 관계 자체도 사후에는 큰 bubble/boom으로 관찰되더라도 사전에 그 경로를 예측하기 어렵다는 연구가 있습니다. ([IDEAS/RePEc][1])

최근 실제 thematic investing 데이터에서도 특정 테마가 지속적으로 리더십을 유지하지 않고 성과 순위가 크게 순환합니다. Morningstar도 테마 간 성과 편차와 잦은 leadership rotation을 지적합니다. ([Morningstar Indexes][2])

따라서:

> **산업 미래예측을 자산배분의 positive signal로 직접 사용하는 것은 현재 프로젝트 목표 대비 ROI가 낮습니다.**

이 부분은 상당히 강하게 결론내릴 수 있습니다.

---

# 7. 그런데 "백테스트 결과만 믿으면 되나?" → 이것도 아니다

이 부분이 핵심입니다.

미래예측을 버린다고 해서:

> 백테스트 1등 전략 = 미래 최적 전략

이 되는 것은 아닙니다.

오히려 현재 ETF-Manager에서는 **backtest overfitting이 미래산업 예측보다 더 현실적인 위험**이 됐습니다.

현재 repo에는 이미:

* adaptive v1
* v2
* v3
* v4
* v5
* reserve 여러 버전
* contribution shaping
* KAFI deployment
* cadence
* overlay
* QQQ/SOXX 5/10/15
* SOXX90
* SOXX100
* risk-budget
* tournament

등 많은 실험 lineage가 존재합니다.

하나하나는 walk-forward를 해도 **프로젝트 전체로는 같은 역사 데이터를 반복해서 보고 다음 아이디어를 만든 것**입니다.

따라서 현재 historical dataset은 더 이상 순수한 OOS가 아닙니다.

학계에서도 수많은 전략·factor를 시도하면 일반적인 significance threshold가 충분하지 않으며, multiple testing 때문에 훨씬 높은 evidence hurdle이 필요하다고 지적합니다. ([NBER][3])

Backtest Overfitting 연구 역시 많은 configuration 중 가장 좋아 보이는 전략을 선택하는 과정 자체가 심각한 selection bias를 만든다는 점을 다룹니다. ([ScholarWorks][4])

---

# 8. 그래서 가장 중요한 원칙은 이것이다

제가 이 프로젝트의 연구 철학을 지금 다시 정의한다면:

> **Backtest는 전략을 채택하는 도구라기보다 전략을 탈락시키는 도구로 더 강하게 사용해야 합니다.**

증거 수준을 비대칭으로 둡니다.

### Negative historical evidence

상당히 강하게 사용할 수 있습니다.

예:

```text
ROBO 5/10/15
→ 전부 장기적으로 명확하게 패배
→ operational allocation = 0
```

### Positive historical evidence

그것만으로 채택하면 안 됩니다.

예:

```text
SOXX10
→ historical pass
→ robustness pass
→ prospective candidate
→ 이후 진짜 신규 데이터로 검증
```

즉:

> **FAIL은 탈락시킬 수 있지만 PASS는 다음 시험장으로 보내는 것일 뿐입니다.**

이게 이 프로젝트에 가장 적합합니다.

---

# 9. 미래산업 정보의 올바른 역할도 바꿔야 한다

앞으로 fundamentals / future industry research는:

### 기존

```text
AI 산업 좋아질 것
→ ETF 찾음
→ portfolio 추가
```

가 아니라,

### 변경

```text
Historical strategy에서 유의미한 edge 발견
        ↓
왜 존재했는지 경제적으로 해석
        ↓
현재도 causal mechanism이 존재하는지 확인
        ↓
현재 valuation / concentration 등이
edge를 깨뜨릴 정도인지 확인
        ↓
VETO 또는 MONITOR
```

가 적절합니다.

즉 fundamental은 **alpha generator보다 veto/filter**로 쓰는 것입니다.

예를 들어 SOXX가 역사적으로 좋았는데:

* AI CAPEX 붕괴
* semiconductor profitability 붕괴
* valuation 극단
* concentration 극단

등이 발견된다면:

> "과거 alpha가 미래에도 이어진다고 보기 어렵다"

는 경고를 줄 수 있습니다.

반대로 fundamentals가 아무리 좋아도 PAVE 역사성과가 1.0007이라면:

> **fundamental이 allocation을 강제로 만들어서는 안 됩니다.**

이게 훨씬 안전합니다.

---

# 10. 현재 fundamental layer도 생각보다 예측력이 강한 구조는 아니다

예를 들어 AI Compute의 primary fundamental은 현재:

```text
PNFI
```

즉 미국 Private Nonresidential Fixed Investment이며, semiconductor production series는 PIT vintage 문제 때문에 deferred 상태입니다.

따라서 현재 `AI_COMPUTE fundamental`이 실제로:

* NVIDIA accelerator shipments
* HBM orders
* semiconductor equipment backlog
* hyperscaler AI-specific CAPEX

를 직접 추적하는 수준은 아닙니다.

상당히 넓은 macro proxy입니다.

이 정도 데이터로 SOXX의 **미래 excess return**을 정밀하게 예측하려는 것은 기대하기 어렵습니다.

따라서 이 layer를 더 확장하는 데 많은 시간을 쓰는 것도 지금은 ROI가 낮습니다.

---

# 11. 현재 "valuation"도 실제 valuation이라고 보기 어렵다

현재 valuation 모듈도 기본적으로 SOXX/QQQ의 상대 가격비율을 trailing percentile로 평가하는 구조입니다.

즉 실제로 기대했던:

* forward P/E
* EV/EBITDA
* FCF yield
* earnings revisions
* expected earnings growth

같은 **fundamental valuation model**과는 다릅니다.

이걸 제대로 구현하려면:

* PIT analyst estimates
* point-in-time financial statements
* ETF look-through valuation
* constituent membership history

등이 필요합니다.

개발비는 급격히 증가합니다.

그런데 그렇게 만들어도 return forecast 정확도가 충분히 높아질지는 알 수 없습니다.

### 따라서 ROI 판단

> **지금 당장 이 방향을 깊게 파는 것은 권하지 않습니다.**

현재 valuation은 risk diagnostic으로 유지하고, 진짜 fundamental valuation은 **필요성이 다시 증명될 때만** 확장하는 게 낫습니다.

---

# 12. 더 중요한 문제: 현재 Adaptive Contribution은 원래 목표와 다른 문제를 풀고 있다

현재 adaptive v5는 KAFI signal에 따라 월 contribution을:

> `0 × base ~ 2 × base`

로 변화시킵니다.

코드 자체가 명시적으로:

> `no horizon sum conservation`

이라고 정의하고 있습니다.

시뮬레이터 역시:

* `contribution_shape`: 전체 credit conservation
* `adaptive_contribution`: conservation 없음

으로 명확히 구분합니다.

이건 매우 중요한 문제입니다.

---

# 13. 왜 문제가 되는가

우리가 알고 싶은 것이:

> 매달 100만원씩 10년 투자할 때 어떤 전략이 복리 자산을 가장 크게 만드는가?

라면,

A가 매달 100만원,

B가 시장 좋을 때 200만원, 나쁠 때 70만원 등으로 투자한다면,

두 전략의 terminal wealth 차이에는:

1. 투자 timing
2. 투자 asset
3. **총 납입액**

세 효과가 동시에 들어갑니다.

즉 이것은 pure investment strategy comparison이 아닙니다.

물론 현재 gate에서 XIRR과 real gain을 같이 보려는 장치가 있지만, 그래도 연구 질문 자체가 달라집니다.

---

# 14. Adaptive contribution을 버릴 필요는 없다

오히려 더 좋은 실험 형태가 있습니다.

현재 외부 적립금:

```text
매달 1,000,000 KRW
```

은 항상 동일하게 들어오게 합니다.

그다음:

```text
투자
or
현금 reserve
```

를 KAFI가 결정하게 합니다.

예:

```text
월 저축 1,000,000
       ↓
Portfolio system
       ↓
┌─────────────┐
│ 투자         │
│ Cash reserve │
└─────────────┘
```

시장 기회가 좋아지면 이미 축적된 reserve를 추가 투입합니다.

중요한 것은:

> **미래의 월급을 미리 빌려서 투자할 수는 없습니다.**

이 구조라면 모든 전략의 외부 cashflow가 동일합니다.

현재 repo에 이미 reserve infrastructure가 있기 때문에 완전히 새로 만들 필요도 없습니다.

### 따라서

현재:

`AdaptiveContribution`

은 operational에서 내리고,

새 연구는:

> **Causal Budget-Constrained Deployment**

로 바꾸는 것을 권합니다.

이건 개발 ROI가 높습니다.

---

# 15. KAFI 자체는 더 튜닝하지 않는 것이 좋다

현재 KAFI는:

* momentum
* drawdown depth
* equity/bond relative
* credit stress
* FX stress
* volatility

등을 percentile rank로 조합합니다.

그리고 adaptive v5에는 다시:

* rank window
* downside power
* upside power
* dispersion
* neutral deadband
* vol inclusion
* min/max multiplier

등이 있습니다.

이미 자유도가 상당합니다.

v1→v5까지 같은 과거 history를 보며 수정했다는 사실까지 고려하면:

> **v6, v7을 만드는 순간 정보량보다 parameter search가 더 커질 가능성이 높습니다.**

따라서:

> **KAFI v5를 마지막 historical version으로 freeze**

하는 것이 좋습니다.

추가 tuning은 하지 않는 것을 권합니다.

---

# 16. 현재 Walk-Forward에는 실제로 수정해야 할 중요한 문제가 있다

이번 코드 검토에서 가장 먼저 수정할 P0입니다.

현재 WF에서 candidate가 train gate를 통과하지 못하면:

```text
chosen_policy = baseline.policy
chosen_targets = baseline.targets
```

로 돌아가는데,

`chosen_adaptive_contribution`은 candidate가 reject되면 `None`으로 설정되는 경로가 있습니다.

문제는 현재 tournament baseline 자체가 adaptive일 수 있다는 것입니다.

그러면:

```text
baseline:
QQQ90/SOXX10 + Adaptive V5
```

인데 candidate가 reject됐을 때:

```text
chosen:
QQQ90/SOXX10 + flat contribution
```

가 됩니다.

즉:

> **"candidate를 선택하지 않았으니 baseline으로 돌아간다"는 contract가 깨집니다.**

---

# 17. 이건 이렇게 바꾸는 게 가장 안전하다

로직을 재조립하지 말고:

```python
chosen_test_arm = (
    candidate_test_arm
    if train_adopted
    else baseline_test_arm
)
```

처럼 해야 합니다.

이렇게 하면 baseline에 나중에:

* adaptive
* cadence
* reserve
* 기타 module

이 추가되더라도 동일 문제가 발생하지 않습니다.

그리고 invariant test를 반드시 넣습니다.

```text
IF train_adopted == false

THEN
chosen_test_arm == baseline_test_arm
```

terminal wealth만 같아서는 안 됩니다.

* total contribution
* XIRR
* TWR
* target
* adaptive config
* costs
* cadence

전체 identity를 보장해야 합니다.

### 중요

이 버그는 **SOXX static incremental 결과 자체를 무효화하지 않습니다.**

하지만 현재 adaptive baseline을 사용하는 tournament/WF의 operational evidence는 수정 후 **전부 다시 실행해야 합니다.**

---

# 18. 현재 Strategy Tournament도 "진짜 OOS"는 아니다

현재 tournament는 여러 후보 중 process를 통과한 후보를 모은 뒤 pooled OOS real gain이 가장 큰 전략을 추천합니다.

게다가 후보가 많으면 in-sample ranking을 이용해 WF 대상 수를 제한하는 로직도 있습니다.

통계적으로 생각하면:

```text
historical data
→ candidate 생성
→ candidate screening
→ WF
→ OOS score 비교
→ winner 선택
```

에서 마지막 OOS도 **winner 선택에 사용됐습니다.**

따라서 그것은 최종 test set이 아닙니다.

validation set에 가깝습니다.

더 심각한 것은 프로젝트 차원에서 이미 adaptive v1~v5 등을 같은 역사 데이터로 반복 연구했다는 점입니다.

즉 내부 WF fold가 아무리 엄격해도:

> **인간 연구자까지 포함한 전체 research process 관점에서는 historical data가 반복 사용됐습니다.**

그래서 지금부터의 진짜 OOS는 과거에서 만들 수 없습니다.

---

# 19. 진짜 OOS는 지금부터 시작해야 한다

이게 앞으로 프로젝트에서 가장 가치 있는 변화입니다.

예를 들어:

### Research epoch

```text
Historical Research Cutoff
= 2026-08-28 panel
```

를 freeze합니다.

그리고:

```text
2026-09 이후 들어오는 데이터
= Prospective OOS
```

로 정의합니다.

그 뒤에는 기존 전략을 수정하지 않습니다.

새로운 아이디어가 생기면:

```text
Strategy v2
```

로 별도 등록합니다.

기존 v1의 forward track record는 그대로 보존합니다.

이렇게 해야 나중에 실제로:

> "이건 내가 결과를 보고 만든 전략이 아니다."

라고 말할 수 있습니다.

코드 몇 천 줄 추가하는 것보다 **이 한 가지가 연구 신뢰도를 더 크게 올립니다.**

---

# 20. "백테스트만 사용한다"를 이렇게 수정해야 한다

제가 권하는 최종 원칙은:

## Empirical-first, Forecast-light

입니다.

구체적으로:

### Historical backtest

후보 발굴 + 강력한 falsification.

### Robustness

selection bias와 path dependence 검사.

### Economic/fundamental analysis

historical winner가 명백히 경제적으로 말이 안 되는 경우 veto.

### Prospective OOS

실제 adoption evidence.

### 미래산업 전망

**0% → positive allocation을 만드는 근거가 아니라 watchlist와 falsifier를 만드는 정보.**

즉:

> **Narrative는 전략을 구할 수 없다.**

PAVE가 백테스트에서 의미 있는 edge가 없는데 AI 전력수요 전망이 좋다고 해서 채택하면 안 됩니다.

---

# 21. 미래산업은 완전히 삭제하지 말고 "옵션"으로 남겨라

AI Power와 Physical Automation은 이렇게 처리하면 됩니다.

```text
AI_POWER_BOTTLENECK
status = PROSPECTIVE_WATCH
operational_weight = 0

PHYSICAL_AUTOMATION
status = PROSPECTIVE_WATCH
operational_weight = 0
```

그리고 1년에 수십 번 다시 테스트할 필요가 없습니다.

새로운 실제 데이터가 충분히 누적되거나 사전에 정의한 event가 발생했을 때만 reopen합니다.

예:

```text
structural fundamentals
+
ETF/vehicle price evidence
+
minimum history
```

가 충족되었을 때 재검토.

이렇게 하면:

> "혹시 2030년에 로봇이 폭발하면?"

이라는 가능성을 버리지 않으면서도,

현재 돈을 희생하지 않습니다.

---

# 22. 현재 main의 각 연구 방향에 대한 ROI 평가

여기서 ROI는 **개발 복잡도 대비 실제 의사결정 정보 증가량**입니다.

| 개발 항목                           | 정보 증가 |    과적합 위험 |   개발비 | 판단        |
| ------------------------------- | ----: | --------: | ----: | --------- |
| **WF fallback bug 수정**          | 매우 높음 |        낮음 |    낮음 | **즉시**    |
| **Fixed cashflow objective 분리** | 매우 높음 |        낮음 |    중간 | **즉시**    |
| **Prospective OOS registry 강화** | 매우 높음 |     매우 낮음 |    낮음 | **즉시**    |
| **QQQ/SOXX 5/10/15 최종 검증**      |    높음 |        중간 |    낮음 | **진행**    |
| **과거 regime coverage 확대**       |    높음 |     낮음~중간 |    중간 | **진행**    |
| Cost/FX/tax sensitivity         |    높음 |        낮음 |    중간 | **진행**    |
| Multiple-testing lineage        |    높음 |        낮음 |    중간 | **진행**    |
| 실제 fundamental valuation 구축     |    중간 |        중간 |    높음 | 보류        |
| AI Power 추가 ETF 탐색              |    낮음 |        높음 |    중간 | **중단**    |
| Robotics 새 ETF 탐색               |    낮음 |        높음 |    중간 | **중단**    |
| KAFI v6/v7                      |    낮음 | **매우 높음** |    중간 | **중단**    |
| ML return forecasting           |   불명확 | **매우 높음** | 매우 높음 | **하지 않음** |
| 뉴스/LLM 미래산업 예측                  |    낮음 |     매우 높음 |    높음 | **하지 않음** |
| 더 많은 weight grid                |    낮음 |     매우 높음 |    낮음 | **하지 않음** |
| 데이터 갱신/monitoring               |    높음 |        낮음 |    낮음 | **지속**    |

---

# 23. 특히 "더 많은 전략을 쉽게 테스트할 수 있게 만들자"는 이제 위험하다

처음에는 좋은 architecture였습니다.

하지만 지금은 전략을 추가하는 비용이 너무 낮아졌습니다.

예를 들어:

```text
SOXX 10 괜찮네
→ 15
→ 20
→ 25
→ adaptive
→ adaptive risk budget
→ timing
→ momentum conditional
→ valuation conditional
...
```

하면 결국 하나는 엄청나게 좋아 보입니다.

그게 바로 현재 프로젝트가 가장 경계해야 하는 것입니다.

Backtest overfitting 연구가 다루는 문제도 본질적으로 이 구조입니다. ([ScholarWorks][4])

따라서 앞으로 좋은 기능은:

> **전략을 쉽게 만드는 기능이 아니라 전략을 만들기 어렵게 만드는 기능**

입니다.

---

# 24. 그래서 `Preregistration`보다 한 단계 더 가야 한다

현재 preregistration이 있어도 충분하지 않습니다.

왜냐하면:

> "이번 전략을 freeze했다."

라고 해도 그 전략 아이디어가 이전 결과를 보고 만들어졌기 때문입니다.

앞으로 registry에:

```text
research_family_id
hypothesis_birth_date
first_test_date
historical_data_used_until
parent_experiments[]
number_of_related_trials
parameter_variants_tried
```

를 남기는 것을 권합니다.

그러면:

```text
SOXX family
```

가 실제로 몇 번 연구되었는지 알 수 있습니다.

이를 기반으로 나중에 PBO/SPA/Reality Check 같은 방식을 적용할 수도 있습니다.

하지만 **복잡한 통계기법 구현 자체보다 trial lineage를 정확히 저장하는 것이 먼저**입니다.

---

# 25. Long-Horizon test도 한 가지 착각을 조심해야 한다

현재 SOXX는:

```text
120M cohort n = 10
```

입니다.

하지만 step=12M라면 인접 10년 cohort는 대부분의 기간을 공유합니다.

따라서:

> n=10 = 독립적인 10년 실험 10개

가 아닙니다.

실제로는 동일한 하나의 역사에서 만들어낸 높은 상관의 관측치들입니다.

Path bootstrap을 추가한 것은 좋은 발전입니다.

하지만 bootstrap도:

> 과거 월별 return process가 미래를 어느 정도 대표한다.

는 가정을 피할 수 없습니다.

따라서 프로젝트의 가장 큰 제한은 이제 **데이터 부족**입니다.

코드를 더 작성해도 금융시장 history를 더 생성할 수는 없습니다.

---

# 26. 그래서 역사 coverage를 늘리는 개발은 ROI가 높다

새 전략을 만드는 대신:

> 현재 SOXX/QQQ 실험이 실제로 어떤 120M start/end cohort를 포함하는가

를 명시적으로 출력하십시오.

그리고:

* dot-com
* GFC
* 2010s low-rate
* COVID
* 2022 inflation/rates
* AI boom

중 무엇을 실제로 포함하고 있는지 audit합니다.

특히 Nasdaq/semiconductor 전략은 dot-com 같은 극단적인 valuation unwind를 포함하는지가 중요합니다.

현재 데이터 infrastructure가 해당 기간을 충분히 지원하지 못한다면:

* ETF 이전 index
* NASDAQ-100 index
* semiconductor index

등을 **research proxy**로만 사용해 이전 regime을 확장하는 것이 새로운 AI 모델을 만드는 것보다 훨씬 높은 ROI입니다.

단 execution ETF backtest와 proxy history는 반드시 분리해야 합니다.

---

# 27. 현재 SOXX10도 "확정된 답"은 아니다

SOXX10을 꽤 긍정적으로 보고 있지만 여기에는 중요한 selection bias가 있습니다.

처음부터:

```text
QQQ90/SOXX10
```

하나만 사전에 정하고 테스트한 게 아닙니다.

많은 ETF와 satellite를 연구한 뒤 **SOXX가 살아남은 것**입니다.

따라서 current historical statistics는:

> unconditional SOXX evidence

가 아니라 어느 정도:

> winner-conditioned evidence

입니다.

이 때문에 현재 가장 적절한 상태는:

### QQQ100 flat

`IMMUTABLE BENCHMARK`

### QQQ90/SOXX10 flat

`PROVISIONAL EMPIRICAL INCUMBENT`

### QQQ85/SOXX15 flat

`AGGRESSIVE CHALLENGER`

### Adaptive V5

`FROZEN RESEARCH CANDIDATE`

정도라고 판단합니다.

---

# 28. 현재 operational adaptive v5를 최종 전략으로 인정하면 안 된다

현재 adaptive는 여러 번의 연구 iteration을 거쳤고, 외부 contribution 자체를 변화시키며, 현재 WF에는 adaptive baseline fallback 문제도 발견됩니다.

따라서:

> `QQQ90/SOXX10 + adaptive v5`

를 지금 최종 operational optimum이라고 부르는 것은 너무 강합니다.

제가 코드 상태를 결정한다면:

```text
OPERATIONAL / PROVISIONAL
QQQ90/SOXX10 flat

RESEARCH ONLY
QQQ90/SOXX10 adaptive_v5
```

로 분리합니다.

그리고 adaptive v5는 더 이상 수정하지 않습니다.

수정하면 새 버전으로 분리합니다.

---

# 29. 다음 개발은 정확히 이 순서가 좋다

## Phase A — Correctness Freeze

가장 먼저:

1. Walk-forward baseline fallback 버그 수정.
2. `candidate rejected ⇒ chosen arm == baseline arm` invariant.
3. adaptive baseline을 포함한 regression tests.
4. README / architecture / operational target 상태 동기화.
5. 현재 모든 결과 artifact를 versioned snapshot으로 freeze.

여기까지는 **무조건 할 가치가 있습니다.**

---

## Phase B — Objective 분리

연구 목표를 두 개로 완전히 분리합니다.

### Objective A — Capital Allocation

```text
매월 외부 cashflow 동일
```

Primary objective:

> **120M real KRW terminal wealth 최대화**

보조:

* CE γ2/5/10
* real XIRR
* p10/p05
* worst
* MDD/recovery

여기서는 adaptive external contribution 금지.

---

### Objective B — Deployment Timing

외부 cashflow 역시:

```text
매월 1M
```

으로 유지.

다만:

```text
Invested
vs
Reserve
```

만 dynamic하게 만듭니다.

이렇게 해야 KAFI의 진짜 timing value를 측정할 수 있습니다.

---

# 30. Phase C — 마지막 Historical Strategy Campaign

후보를 딱 고정합니다.

```text
B0 = QQQ100

C1 = QQQ95 / SOXX5
C2 = QQQ90 / SOXX10
C3 = QQQ85 / SOXX15
```

**여기서 끝입니다.**

SOXX20/25/30을 추가하지 않습니다.

그리고 같은 후보를 대상으로:

* 120M cohort
* cohort start/end 출력
* CE γ2/5/10
* paired path bootstrap
* cost stress
* FX stress
* tax sensitivity
* realized weight drift
* XIRR
* worst/p05
* regime coverage
* pre-history proxy stress

를 한 번에 돌립니다.

그 결과를 `FINAL_HISTORICAL_CAMPAIGN_V1`으로 freeze합니다.

---

# 31. Phase D — multiple testing 방어

현재까지 시도한 전략 family를 registry로 복원합니다.

예:

```text
QQQ
VTI
IWF
GRID
XLI
ITA
IBB
BOTZ
ROBO
PAVE
SOXX
reserve v1~v4
adaptive v1~v5
cadence
overlay
...
```

그리고 최종 보고서에:

> 몇 개의 strategy family와 parameter variant를 봤는가?

를 반드시 공개합니다.

SOXX 결과는 그 이후에 해석해야 합니다.

---

# 32. Phase E — Prospective Freeze

이 단계부터 개발 정책이 달라집니다.

현재까지 사용한 history:

```text
≤ 2026-08-28
```

은 모두:

> **TRAINED/SEEN HISTORY**

로 취급합니다.

그리고 최종 candidate를 version freeze합니다.

예:

```text
PROSPECTIVE_2026_V1

Benchmark
QQQ100

Candidate A
QQQ90/SOXX10

Candidate B
QQQ90/SOXX10 + causal reserve KAFI
```

그 뒤 새 데이터가 들어올 때마다 기록만 합니다.

**parameter를 건드리지 않습니다.**

---

# 33. AI Power / Robotics는 여기서 동결

이 둘은 추가 ETF hunting을 중지합니다.

### AI Power

```text
thesis = WATCH
PAVE = insufficient/no meaningful edge
GRID = rejected historical proxy
weight = 0
```

### Physical Automation

```text
thesis = WATCH
ROBO = rejected vehicle
BOTZ = rejected/weak vehicle
weight = 0
```

새 뉴스가 나올 때마다 실험하지 않습니다.

그건 또 다른 researcher degrees of freedom입니다.

---

# 34. 향후 미래산업을 다시 볼 수 있는 조건

사전에 reopen condition을 정의합니다.

예를 들어:

```text
minimum additional OOS history
AND
fundamental structural evidence
AND
market-price evidence
AND
investable vehicle quality
```

가 모두 개선될 때만 재연구합니다.

그러면 미래를 무시하지 않으면서도 narrative chasing을 방지할 수 있습니다.

---

# 35. 언제 프로젝트 개발을 멈춰야 하는가

여기서 stop rule도 필요합니다.

위의:

* correctness audit
* objective separation
* frozen historical campaign
* history/regime extension
* prospective registry

까지 구현하면 **새 전략 개발은 사실상 종료하는 것을 권합니다.**

그 뒤 ETF-Manager는:

```text
Research Engine
        ↓
Validation Engine
        ↓
Frozen Strategy Registry
        ↓
Periodic Data Refresh
        ↓
Prospective Monitoring
```

시스템이 됩니다.

이 상태가 오히려 훨씬 완성도 높은 퀀트 프로젝트입니다.

---

# 36. 왜 거기서 멈추는 것이 맞나

그 이후 기능을 더 만드는 것은 대부분:

```text
정보량 증가 << 모델 자유도 증가
```

가 되기 때문입니다.

예를 들어 ML 모델 하나를 넣는다고 시장 역사 30년이 추가되지 않습니다.

100개의 feature를 추가한다고 새로운 독립 regime이 생기지 않습니다.

오히려 가능한 전략 수만 늘어납니다.

현재 프로젝트의 bottleneck은:

> **software capability가 아니라 independent evidence**

입니다.

이건 상당히 중요한 전환점입니다.

---

# 37. 최종적으로 전략 자체는 어떻게 볼 것인가

현재 evidence만 기준으로 하면 저는 이렇게 정리합니다.

| 전략                       | 현재 판단                                             |
| ------------------------ | ------------------------------------------------- |
| **QQQ100 flat**          | 영구 benchmark                                      |
| **QQQ90/SOXX10 flat**    | **현재 가장 합리적인 challenger / provisional incumbent** |
| QQQ85/SOXX15 flat        | 성장 우위 더 큼, tail trade-off 증가                      |
| QQQ95/SOXX5 flat         | 보수적이나 incremental edge 작음                         |
| QQQ90/SOXX10 Adaptive V5 | **재검증 전 research-only**                           |
| SOXX100                  | 전략 선택 evidence로 사용하지 않는 편이 좋음                     |
| PAVE satellite           | 보류/0%                                             |
| GRID satellite           | reject                                            |
| ROBO/BOTZ                | reject vehicle                                    |
| 기타 미래 테마                 | prospective watch only                            |

---

# 38. 장기 복리 극대화라는 목표에 가장 중요한 판단

여기서 MDD 최소화나 Sharpe 최대화가 목적이 아닙니다.

목표는:

> **동일한 현실적 외부 저축 조건에서 10년 이상 real terminal wealth를 최대화하는 것**

이어야 합니다.

따라서 primary objective는 계속:

```text
120M real terminal wealth
```

가 맞습니다.

다만 단순 median만 최대화하면 안 됩니다.

최종 선택은 대략:

```text
Expected long-horizon wealth
+
downside robustness
+
OOS persistence
-
complexity
-
selection uncertainty
```

라는 관점으로 봐야 합니다.

이 관점에서 현재 SOXX10이 SOXX15보다 더 방어 가능한 이유도 설명됩니다.

---

# 39. "백테스트만 정직하게 쓰면 충분한가?"에 대한 최종 답

**절반은 맞고 절반은 틀립니다.**

### 맞는 부분

미래산업을 주관적으로 예측해서 allocation을 주는 것보다:

> **실제 시장가격에서 검증된 효과를 우선하는 것이 훨씬 낫습니다.**

### 틀린 부분

많은 전략을 시도한 뒤 나온 최고의 백테스트를 그대로 믿으면:

> **그것 역시 미래를 예측하는 것만큼 위험해질 수 있습니다.**

따라서 최종 철학은:

> **미래를 예측하지 말자. 그러나 과거를 과적합해서 미래인 척하지도 말자.**

가 적절합니다.

---

# 40. 프로젝트에 대한 최종 ROI 판단

### 프로젝트 전체

**계속할 가치가 높습니다.**

이미 만든:

* PIT infrastructure
* KRW accounting
* FX/CPI
* integer lot
* buy-only simulation
* cohort
* bootstrap
* WF
* strategy registry
* fresh panel
* extensive tests

등은 버리기에는 가치가 큽니다.

### 새로운 전략 발명

**ROI 낮음. 중단 권고.**

### 미래산업 예측 시스템 확대

**ROI 낮음. operational 연구에서는 중단 권고.**

### ML/LLM/news 기반 미래 수익률 예측

**현재 단계에서는 ROI가 오히려 음수일 가능성이 높음.**

### 검증 정확성 향상

**ROI 매우 높음.**

### 과거 regime 확장

**ROI 높음.**

### 진짜 prospective OOS

**가장 높은 ROI.**

### 유지보수/데이터 갱신

**장기적으로 가장 실용적인 ROI.**

---

# 제가 지금 이 프로젝트를 맡는다면

프로젝트를 **“전략 탐색 단계”에서 “수렴·반증 단계”로 공식 전환**하겠습니다.

가장 먼저 WF adaptive-baseline 오류를 수정하고, `QQQ90/SOXX10 + adaptive_v5`를 final operational이라고 간주하지 않은 상태에서 모든 핵심 전략을 재검증합니다. 그다음 asset allocation은 **동일 월 현금흐름**으로 고정하고, adaptive는 reserve 기반 timing 문제로 분리합니다. SOXX는 5/10/15 외에 더 이상 weight를 탐색하지 않고 최종 historical falsification을 수행합니다.

그 결과 **QQQ90/SOXX10 flat**이 여전히 살아남으면 이를 `PROVISIONAL_INCUMBENT`로 freeze하는 것이 현재 가장 합리적입니다. 반대로 corrected validation에서 무너지면 QQQ100으로 돌아갑니다. AI Power/Robotics 같은 미래 thesis는 0% weight의 prospective watch로만 남깁니다.

그리고 **2026-08-28까지의 모든 데이터는 이미 본 데이터라고 선언**하는 것이 중요합니다. 앞으로 들어오는 데이터만 진짜 prospective evidence로 인정합니다.

여기까지 구현한 뒤에는 새 기능 개발을 크게 줄이는 편이 좋습니다. 그 시점부터 프로젝트의 가치는 “계속 더 좋은 백테스트를 찾아내는 것”이 아니라 **지금 가진 최선의 전략이 정말 새로운 시장에서도 살아남는지를 정직하게 기록하는 것**에 있습니다.

현재 프로젝트에서 앞으로 얻을 수 있는 가장 큰 개선은 더 정교한 예측 모델이 아니라 **잘못된 확신을 줄이는 것**입니다. 장기 복리 전략에서는 그것이 실질적으로 가장 높은 개발 ROI라고 판단합니다.

[1]: https://ideas.repec.org/a/aea/aecrev/v99y2009i4p1451-83.html "https://ideas.repec.org/a/aea/aecrev/v99y2009i4p1451-83.html"
[2]: https://indexes.morningstar.com/insights/Analysis/bltfe9deb5617110ea2/thematic-investing-performance-and-rotation-insights "https://indexes.morningstar.com/insights/Analysis/bltfe9deb5617110ea2/thematic-investing-performance-and-rotation-insights"
[3]: https://www.nber.org/papers/w20592 "https://www.nber.org/papers/w20592"
[4]: https://scholarworks.wmich.edu/math_pubs/42/ "https://scholarworks.wmich.edu/math_pubs/42/"
