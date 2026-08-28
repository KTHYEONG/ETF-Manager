
지금 결과로 방향을 크게 틀 필요는 없습니다. 오히려 **현재 단계에서 얻은 결론을 고정하고, 다음 연구가 과적합으로 흐르지 않게 범위를 좁혀야 합니다.**

핵심 해석은 세 가지입니다.

* **QQQ 100%를 바꿀 근거는 현재 없다.**
* `QQQ95 + GRID5`의 CE 0.998은 “거의 성공”이 아니라 **수익 극대화 목적에서는 QQQ보다 추가 가치가 확인되지 않은 것**이다.
* `QQQ80 + GRID10 + XLI10`이 한 OOS 구간에서 +0.53%였다는 사실은 흥미롭지만, train에서 한 번도 채택되지 않았으므로 **regime-specific 현상일 가능성**이 더 크다.

따라서 GRID 비중을 3%, 7%, 8%처럼 재조정하면서 다시 돌리는 방향은 중단하는 것이 좋습니다. 그 순간부터 researcher overfitting이 시작됩니다.

## 다음 방향

### 1. 현재 결과를 research wave 종료 결과로 고정

현재 wave의 결론은 그대로 기록합니다.

```md
FUTURE_INDUSTRY_STATIC_MIX_V1

Result:
- QQQ remains operational policy.
- IWF does not replace QQQ.
- GRID 5/10/15% does not improve CE.
- QQQ80/GRID10/XLI10 does not pass walk-forward adoption.
- No further tuning of GRID weights is permitted in this research wave.
```

이 실험은 실패한 것이 아니라 상당히 유용합니다.

특히 IWF 100%가 4개 cohort 모두 QQQ에 패배했다는 것은 적어도 2012~2025에서는 QQQ 성과가 단순히 `US Large Growth`를 보유했기 때문만은 아니라는 근거를 하나 얻은 것입니다.

---

# 2. 가장 먼저 해야 할 일은 ETF 추가가 아니라 **10년 투자 목적과 실험 horizon을 맞추는 것**

현재 가장 큰 약점은 여기에 있습니다.

실제 목표:

```text
10년 이상 적립
```

현재 핵심 ablation:

```text
36개월
n = 4
```

둘은 상당히 다릅니다.

따라서 다음 개발 우선순위는:

> **120개월 Rolling Accumulation Cohort Engine**

입니다.

예를 들어 데이터가 충분하다면:

```text
2001-01 → 2010-12
2001-02 → 2011-01
2001-03 → 2011-02
...
2016-07 → 2026-06
```

처럼 월 단위 overlapping cohort를 생성합니다.

단, overlapping cohort는 서로 독립이 아니므로 단순 승률을 통계적 독립 표본처럼 취급하면 안 됩니다.

그래서 동시에:

```text
step = 1M   → 상세 분포
step = 12M  → 연도별 robustness
step = 36M  → 낮은 중첩 robustness
```

를 보고 moving-block bootstrap을 결합하는 것이 좋습니다.

---

# 3. 데이터 시작점을 2012년보다 앞으로 당기는 것이 매우 중요

현재 결과의 가장 큰 한계는 `2012-06` 시작입니다.

QQQ를 장기 Core로 선택하는 연구에서 우리가 반드시 보고 싶은 시기는:

```text
닷컴 붕괴
2000~2002

회복기
2003~2007

금융위기
2008~2009

제로금리 성장주 시대
2010s

인플레이션/금리충격
2022
```

입니다.

특히 QQQ에 가장 불리한 역사인 **닷컴버블을 제외한 상태에서 QQQ 장기 우위를 연구하는 것은 구조적으로 불완전**합니다.

따라서 다음 데이터 작업의 우선순위는:

```text
P0
Korean CPI historical PIT coverage 확장

P0
USD/KRW historical coverage 확장

P0
QQQ / XLI / SOXX / IBB 등 가격 history 확장

P1
Nasdaq-100 research proxy
→ 2000 전후 stress-test

P1
필요 없는 macro dataset 때문에
static DCA feasibility start가 제한되지 않는지 확인
```

입니다.

특히 static mix 연구에는 VIX나 BAA10Y가 필요 없습니다.

**불필요한 dataset의 시작일이 전체 feasibility window를 2012년으로 잘라버리고 있지 않은지 반드시 확인**하는 것이 좋습니다.

---

# 4. 그 다음에야 나머지 Satellite를 하나씩 검증

현재 실제로 독립 검증된 미래산업 satellite는 GRID뿐입니다.

`80/10/10` 결과만 가지고 XLI를 평가하면 안 됩니다.

GRID와 XLI 효과가 섞여 있기 때문입니다.

다음 wave는 아래 정도면 충분합니다.

| 순서 | 실험                        | 이유                      |
| ---- | --------------------------- | ------------------------- |
| 1    | QQQ95/90/85 +**XLI**  | Physical economy 독립효과 |
| 2    | QQQ95/90/85 +**SOXX** | Semiconductor overweight  |
| 3    | QQQ95/90 +**IBB**     | Bio diversification       |
| 4    | QQQ95/90 +**ITA**     | Defense/aerospace         |
| 5    | QQQ95/90 +**BOTZ**    | Robotics pure-play        |

중요한 규칙은:

```text
단일 satellite
      ↓
통과
      ↓
조합 후보

단일 satellite
      ↓
실패
      ↓
조합에 넣지 않음
```

입니다.

따라서 지금 당장 `GRID + XLI + SOXX + BOTZ` 같은 조합 탐색으로 넘어가서는 안 됩니다.

---

# 5. GRID는 일단 research 후보에서 내려놓는 것이 맞다

현재 GRID 결과는 꽤 명확합니다.

```text
GRID 5%   0.998
GRID 10%  0.995
GRID 15%  0.992
```

비중을 늘릴수록 CE가 단조 감소합니다.

이는 현재 window에서는:

> **GRID exposure가 diversification benefit보다 QQQ의 높은 복리수익을 희석한 효과가 더 컸다**

고 해석하는 것이 가장 자연스럽습니다.

따라서 GRID는:

```text
Operational candidate → NO
Observation list → YES
```

로 내립니다.

미래 전력망 thesis 자체는 그대로 유지할 수 있습니다.

하지만:

> 좋은 산업 전망 ≠ 지금 portfolio weight를 줘야 한다

는 것을 데이터가 보여준 것입니다.

---

# 6. 2% hurdle은 지금 낮추면 안 된다

CE 0.998을 보고:

> "2% hurdle이 너무 센 것 아닐까?"

라는 생각이 생길 수 있습니다.

하지만 **이번 결과를 본 뒤 hurdle을 낮추면 안 됩니다.**

그건 명백한 사후 규칙 변경입니다.

현재 wealth-maximization operational gate는:

```text
candidate CE / QQQ CE > 1.02
```

를 그대로 유지하는 게 좋습니다.

다만 향후 전혀 다른 연구 목적을 만들 수는 있습니다.

예를 들어:

```text
Objective A
Maximum Compounding
→ 기존 >1.02 CE adoption

Objective B
Strategic Diversification
→ QQQ 대비 CE 비열등성
  + tail risk/MDD/concentration 개선
```

입니다.

하지만 두 Objective를 섞으면 안 됩니다.

현재 사용자의 핵심 목표가 **자산 증식 극대화**라면 Objective A가 operational policy를 결정해야 합니다.

---

# 7. 현실 비용도 다음 단계에서는 넣어야 한다

이번 결과는:

```text
commission = 0
FX spread = 0
```

입니다.

QQQ 100%와 다중 ETF를 비교할 때 이것은 다중 ETF에 약간 유리한 가정입니다.

실전에서는:

```text
QQQ 1개
```

보다

```text
QQQ
GRID
XLI
```

가 주문 수·잔여현금·환전/체결 구조에서 조금 더 복잡합니다.

따라서 최종 adoption 이전에는 반드시:

```text
Ideal
Base
Stress
```

비용 시나리오를 통과해야 합니다.

그리고 장기적으로는 **한국 거주자의 배당세·해외주식 세금까지 별도의 tax-aware simulation**으로 넣을 가치가 있습니다.

ETF별 배당성향이 다르기 때문에 10~20년 복리에서는 차이가 생길 수 있습니다.

---

# 8. 앞으로의 연구 Tree는 이렇게 잡는 것이 가장 깔끔하다

```text
                     QQQ 100
                        │
              Operational Champion
                        │
        ┌───────────────┴──────────────┐
        │                              │
  Evidence Upgrade              New Satellites
        │                              │
  120M cohort                     XLI 5/10/15
  pre-2012 data                   SOXX 5/10/15
  dot-com stress                  IBB 5/10
  realistic costs                 ITA 5/10
  bootstrap                       BOTZ 5/10
        │                              │
        └───────────────┬──────────────┘
                        │
                 Passing arms only
                        │
                   Combination
                        │
                120M cohort test
                        │
                 Walk-forward
                        │
                  Cost stress
                        │
                  Bootstrap
                        │
                    CE gate
                        │
              ┌─────────┴─────────┐
              │                   │
            FAIL                PASS
              │                   │
          QQQ 유지           New Policy
```

---

# 9. 개발 우선순위를 정하면

제가 지금 ETF-Manager를 이어 개발한다면 순서는 이렇습니다.

```md
## Wave 1 — Long-Horizon Validation

- [ ] 120-month rolling accumulation cohort 구현
- [ ] cohort step 1M / 12M / 36M 지원
- [ ] median / worst / p10 / win-rate 출력
- [ ] overlapping cohort 의존성 명시
- [ ] moving-block bootstrap 연동
- [ ] recovery time 추가


## Wave 2 — Historical Coverage

- [ ] static DCA feasibility dependency audit
- [ ] CPI historical coverage 확장
- [ ] USDKRW historical coverage 확장
- [ ] QQQ/XLI/SOXX/IBB 장기 가격 ingest
- [ ] Nasdaq-100 dot-com research proxy
- [ ] 2000s stress regime 추가


## Wave 3 — Independent Satellite Test

- [ ] QQQ + XLI 5/10/15
- [ ] QQQ + SOXX 5/10/15
- [ ] QQQ + IBB 5/10
- [ ] QQQ + ITA 5/10
- [ ] QQQ + BOTZ 5/10

GRID는 v1 결과로 reject 상태 유지.


## Wave 4 — Combination

단일 satellite gate를 통과한 ETF만 사용.

- [ ] 2-satellite combination
- [ ] coarse weights only
- [ ] plateau check
- [ ] no post-result retuning


## Wave 5 — Operational Validation

- [ ] 120M cohorts
- [ ] walk-forward
- [ ] realistic cost grid
- [ ] moving-block bootstrap
- [ ] CE γ=2/5/10
- [ ] worst-cohort gate
- [ ] tax-aware sensitivity
```

---

## 현재 투자 연구 상태를 한 문장으로 표현하면

> **QQQ가 최종적으로 최선이라고 증명된 것이 아니라, 현재까지 도전한 미래산업 보완안들이 QQQ를 이기지 못했다. 따라서 QQQ를 incumbent로 유지하면서 더 긴 역사와 실제 10년 적립 horizon으로 증거 수준을 높이고, XLI·SOXX·IBB·ITA·BOTZ를 독립적으로 반증해 나가는 단계로 이동해야 한다.**

특히 지금 가장 중요한 작업은 **새 ETF를 더 찾는 것이 아니라 `120개월 cohort + 2000년대 데이터 확장`입니다.** 이 두 가지 없이 satellite를 계속 늘리면 ETF-Manager가 정교한 백테스터가 아니라 2012~2025 구간에서 잘 맞는 ETF 조합 검색기로 변할 위험이 있습니다.
