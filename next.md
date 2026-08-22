당신은 **퀀트 리서처, 포트폴리오 엔지니어, 금융 데이터 엔지니어, 백테스트 시스템 아키텍트** 역할을 동시에 수행한다.

이번 작업의 목표는 단순한 ETF 자동매매 봇이나 단기 수익률 최적화 시스템을 만드는 것이 아니다.

최종 목표는 다음과 같다.

> **한국에서 장기간 근로소득을 통해 신규 자금을 지속적으로 투자하는 개인 투자자가, 수십 년의 투자기간 동안 비용·세금·환율·drawdown·실행 가능성을 고려하면서 실질 KRW 기준 자산을 견고하게 복리 증식할 수 있는 ETF 기반 퀀트 투자 연구 및 실행 시스템을 설계한다.**

현재 제시되는 모든 전략 아이디어와 아키텍처는 **확정된 정답이 아니라 초기 hypothesis**이다.

기존 아이디어를 무조건 유지하거나 그대로 구현하지 말라.
금융공학적 근거, 데이터 가용성, point-in-time 검증 가능성, 백테스트 편향, 구현 복잡성, 실전 비용 등을 근거로 필요하다면 적극적으로:

* 수정
* 단순화
* 제거
* 대체
* 재설계

하라.

다만 근거 없이 임의의 복잡한 전략이나 새로운 지표를 추가해서는 안 된다.

---

# 1. 가장 중요한 설계 원칙

프로젝트의 중심은 ETF ticker가 아니다.

반드시 다음 순서를 유지한다.

```text
Investment Objective
        ↓
Risk Budget
        ↓
Economic Exposure / Asset Classes
        ↓
Strategic Allocation
        ↓
Factor Exposure
        ↓
Optional Dynamic Risk Overlay
        ↓
Contribution / Rebalancing
        ↓
Currency Management
        ↓
ETF Mapping
        ↓
Orders / Execution
        ↓
Portfolio Ledger
        ↓
Validation
```

즉,

```text
ETF → 전략
```

이 아니라

```text
경제적 Exposure → 전략 → ETF 구현
```

구조여야 한다.

ETF는 투자전략 자체가 아니라 **원하는 자산 및 factor exposure를 구현하기 위한 implementation vehicle**로 취급한다.

---

# 2. 프로젝트 목표 정의

최종적으로 최적화하고자 하는 것은 단순 CAGR가 아니다.

장기 적립 투자 관점에서 다음을 함께 고려한다.

### 최대화 대상

* Real Terminal Wealth in KRW
* Terminal Wealth
* Money Weighted Return / XIRR
* 장기 risk-adjusted return
* 목표 exposure 유지 정확도

### 최소화 대상

* Maximum Drawdown
* Drawdown Duration
* Tail Risk
* Shortfall Risk
* Turnover
* Transaction Cost
* FX Cost
* Tax Drag
* Strategy Complexity
* Parameter Sensitivity

개념적인 목적함수는 다음처럼 생각할 수 있다.

```text
Utility
=
Return / Wealth
- Risk penalty
- Cost penalty
- Turnover penalty
- Complexity penalty
```

정확한 수식은 연구를 통해 결정한다.

단순히 가장 높은 CAGR를 만든 전략을 최종 전략으로 선택해서는 안 된다.

---

# 3. 기존 투자안에 얽매이지 말 것

과거 예시로 사용된 다음 요소들은 모두 **검증 대상일 뿐 기본 전제로 사용하지 않는다.**

```text
SPYM
VUG
AVUV
VXUS
QQQM

Fear & Greed
VIX threshold
S&P500 drawdown threshold
폭락 시 추가매수
USD/KRW 252일 이동평균
특정 ETF 고정비중
```

이들을 유지해야 할 이유가 데이터와 연구 결과로 확인될 때만 유지한다.

특히 다음과 같은 논리를 사전에 가정하지 않는다.

```text
VIX가 높음 → 앞으로 기대수익이 높음
시장 폭락 → 무조건 추가 현금 투입이 최적
미국 시장이 과거 강함 → 미래에도 US overweight가 최적
Growth ETF가 과거 좋음 → Growth tilt가 필요
환율이 이동평균보다 높음 → 달러가 비쌈
```

모두 독립적인 hypothesis로 검증한다.

---

# 4. 먼저 현재 프로젝트 상태를 검사하라

작업 시작 즉시 코드를 작성하지 말라.

현재 repository가 존재한다면 먼저 전수 조사한다.

다음을 확인한다.

```text
- repository 구조
- 기존 Python environment
- pyproject.toml / requirements
- 데이터 수집 코드
- 백테스트 코드
- portfolio 관련 코드
- config
- tests
- notebooks
- datasets
- documentation
```

그리고 다음을 먼저 보고하라.

```text
1. 현재 구현 상태
2. 재사용 가능한 코드
3. 구조적 문제
4. 기술부채
5. look-ahead / survivorship / data leakage 가능성
6. 삭제 또는 교체해야 할 코드
7. 제안 아키텍처와 현재 코드 사이의 gap
```

기존 코드가 없으면 새 프로젝트라고 판단하고 진행한다.

---

# 5. 데이터 아키텍처

데이터 계층은 전략 코드와 완전히 분리한다.

목표 구조:

```text
Data Provider
      ↓
Raw Immutable Data
      ↓
Normalization
      ↓
Point-in-Time / Availability Processing
      ↓
Quality Validation
      ↓
Normalized Parquet
      ↓
Feature Engine
```

---

# 6. 데이터 공급원 조사 및 선택

아래 공급원을 우선 검토하되, 현재 API 정책·가격·데이터 범위·라이선스가 변경되었을 수 있으므로 **공식 문서를 직접 확인한 뒤 결정**한다.

### 가격 / ETF

후보:

```text
Tiingo
Alpha Vantage
Nasdaq
SEC EDGAR
yfinance (prototype/fallback only)
```

yfinance는 빠른 탐색과 cross-check에는 사용할 수 있지만 production-grade historical truth source로 무조건 신뢰하지 않는다.

ETF 가격 데이터에서 최소한 다음이 필요하다.

```text
date
ticker
open
high
low
close
volume

adjusted prices
dividends
split factors

source
retrieved_at
```

---

# 7. Factor 데이터

우선적으로 Kenneth French Data Library를 검토한다.

연구 후보:

```text
MKT-RF
SMB
HML
RMW
CMA
Momentum
```

가능하다면:

```text
US
Developed
Developed ex-US
Emerging
```

지역별 장기 데이터를 활용한다.

Factor 연구와 ETF 실제 구현을 반드시 분리한다.

예:

```text
Small Value premium 존재 여부 연구
        ↓
실제 ETF로 구현 가능한지 연구
```

이어야 하며,

```text
AVUV 성과가 좋았음
        ↓
Small Value가 좋음
```

식의 역방향 추론을 금지한다.

---

# 8. Macro / FX 데이터

우선 조사할 데이터 소스:

```text
FRED
ALFRED
한국은행 ECOS
```

대상 예:

```text
USD/KRW
VIX
Treasury yields
T-Bill
Credit Spread
Inflation
Yield Curve
기타 macro 변수
```

Macro 데이터는 반드시 release timing / vintage를 고려한다.

예를 들어 오늘 수정된 과거 CPI나 GDP 값을 과거 시점의 전략 입력으로 사용하면 안 된다.

가능하면 각 데이터 포인트에:

```text
observation_date
release_date
available_at
vintage_date
```

개념을 적용한다.

---

# 9. 데이터 저장

권장:

```text
Parquet
+
DuckDB
+
Polars
```

구조 예:

```text
data/
├── raw/
│   ├── prices/
│   ├── fred/
│   ├── fama_french/
│   ├── sec/
│   └── ecos/
│
├── normalized/
│   ├── prices/
│   ├── factors/
│   ├── macro/
│   ├── fx/
│   └── metadata/
│
├── features/
│
└── manifests/
```

Raw 데이터는 기본적으로 immutable하게 관리한다.

가능하면 manifest에 다음을 기록한다.

```text
provider
endpoint
request parameters
download timestamp
file hash
schema version
normalization version
```

---

# 10. Data Quality Gate

전략 계산 전에 데이터는 자동 검증을 통과해야 한다.

최소한:

```text
Schema validation
Duplicate detection
Missing data detection
Trading-session validation
Timezone validation
OHLC consistency
Corporate-action validation
Adjusted-return validation
Outlier detection
Cross-provider sanity check
```

를 설계한다.

데이터 오류를 자동으로 조용히 보정하지 말 것.

특히 위험한 처리는 다음과 같다.

```python
fillna(method="ffill")
```

해당 값이 실제 당시 알려지지 않았던 데이터라면 미래정보가 들어갈 수 있다.

Missing data 처리 정책은 데이터 종류별로 명시한다.

---

# 11. 거래일 캘린더

단순 weekday calendar를 사용하지 말라.

NYSE 등 실제 거래소 calendar를 이용한다.

후보:

```text
exchange_calendars
```

휴장일, special close 등을 고려한다.

---

# 12. Point-in-Time 원칙

프로젝트 전체의 최우선 invariant다.

시점 `t`에서 계산되는 모든 signal은 반드시:

```text
available_at <= t
```

인 데이터만 사용해야 한다.

다음을 엄격히 검사한다.

```text
- revised macro data
- ETF inception date
- corporate actions
- index constituent information
- ETF metadata
- expense ratio
- AUM
- fund closure
- index changes
```

현재 시점의 metadata를 과거 전체에 적용하면 안 된다.

---

# 13. Survivorship Bias

현재 존재하는 ETF 목록을 이용해 과거 ETF selection 백테스트를 하지 말라.

예:

```text
2026년에 살아남은 ETF 목록
        ↓
2010년 optimal ETF selector
```

를 만들면 안 된다.

Historical point-in-time ETF universe가 확보되지 않으면:

```text
Dynamic ETF Selection
```

은 research limitation으로 표시하고 production candidate에서 제외한다.

학교의 WRDS / CRSP 사용 가능 여부도 검토할 것.

가능하다면 survivorship-bias-free dataset을 이용한다.

---

# 14. 연구 백테스트와 실제 ETF 백테스트를 분리

반드시 두 계층으로 만든다.

## A. Exposure Research Backtest

목적:

```text
경제적 아이디어 자체가 유효한지
```

사용 예:

```text
Fama-French
장기 index
historical factor data
macro
```

---

## B. ETF Implementation Backtest

목적:

```text
실제 거래 가능한 ETF로 구현했을 때도 효과가 유지되는지
```

반영:

```text
actual ETF inception
expense
tracking difference
spread
liquidity
FX
tax
execution costs
```

다음을 절대로 동일시하지 않는다.

```text
Factor return
≠
ETF return
```

---

# 15. Baseline 전략

가장 먼저 단순한 benchmark를 만든다.

최소 benchmark:

```text
B0:
Global market exposure
+
정기적인 동일 외부 현금흐름
+
No market timing
+
No FX timing
+
최소한의 rebalancing
```

추가 benchmark:

```text
B1: US equity DCA
B2: Global equity DCA
B3: Global equity + bond allocation
```

등을 둘 수 있다.

모든 전략은 이 단순 benchmark를 이길 충분한 이유가 있어야 한다.

---

# 16. Strategic Allocation 연구

ETF를 결정하기 전에 경제적 exposure를 정의한다.

초기 후보:

```text
US Broad Equity
Developed ex-US
Emerging Markets
Global Equity
Short Treasury
Intermediate Treasury
Broad Bonds
```

연구 후보 예:

```text
S0 Global Equity
S1 US Equity
S2 US + Developed ex-US + EM
S3 Global Equity + Bonds
S4 Global Equity + Short Treasury
```

특정 비중을 사전에 정답으로 지정하지 않는다.

---

# 17. 포트폴리오 최적화 주의사항

Expected-return 최적화는 매우 불안정할 수 있다.

처음부터:

```text
maximize expected CAGR
```

같은 unrestricted optimizer를 만들지 않는다.

초기 후보는 상대적으로 단순한 방식부터 비교한다.

```text
Market-cap weighting
Fixed allocation
Equal sleeve allocation
Risk parity
Minimum variance
Minimum semivariance
CVaR / CDaR constrained model
```

사용 가능한 라이브러리 후보:

```text
cvxpy
PyPortfolioOpt
```

최종 전략에서는 가능하면 수식과 constraint가 명확히 드러나게 한다.

---

# 18. Factor Engine

우선 연구할 후보:

```text
Market
Value
Size
Profitability / Quality
Momentum
```

CMA 등 다른 factor는 incremental value가 있을 때 추가한다.

먼저 fixed factor tilt를 연구한다.

Factor timing은 후순위 challenger로 둔다.

즉:

```text
Fixed factor exposure
        ↓
검증 성공
        ↓
그 다음 factor timing 연구
```

순서다.

---

# 19. ETF Factor Exposure 측정

실제 ETF가 원하는 factor를 제대로 구현하는지 regression으로 분석할 수 있다.

예:

```text
ETF excess return
~
MKT
+ SMB
+ HML
+ RMW
+ CMA
+ MOM
```

후보:

```text
statsmodels
```

각 ETF에 대해 최소한 다음을 계산할 수 있게 검토한다.

```text
beta_market
beta_size
beta_value
beta_profitability
beta_investment
beta_momentum
alpha
R²
tracking behavior
```

ETF의 이름이나 marketing label만 믿지 않는다.

---

# 20. Contribution Rebalancing

장기 적립식 투자에서 핵심 모듈이다.

신규 투자금:

```text
C_t
```

현재 포트폴리오와 목표 포트폴리오 차이를 계산해서 신규 자금을 우선적으로 부족한 자산에 투입한다.

단순 target weight proportional buying과 비교한다.

개념적으로:

```text
minimize
    target deviation after contribution
    + transaction costs

subject to
    sum(buys) <= new contribution
    buy_i >= 0
```

형태의 optimization을 검토한다.

후보:

```text
cvxpy
```

---

# 21. 매도 최소화

Accumulation phase에서는 기본적으로:

```text
BUY-ONLY REBALANCING
```

을 baseline으로 둔다.

다만 비중이 심각하게 벗어난 경우:

```text
rebalance band
```

을 사용한 매도를 challenger로 검토한다.

다음 parameter는 연구 대상이다.

```text
absolute band
relative band
minimum trade size
minimum holding period
tax-aware sell threshold
```

---

# 22. Dynamic Risk Overlay

Market timing은 전체 포트폴리오를 지배하게 하지 않는다.

기본 구조:

```text
Strategic Allocation
        +
Small / Bounded Tactical Overlay
```

로 둔다.

우선 검토할 signal:

```text
Trend
Momentum
Realized Volatility
Drawdown
```

그 다음:

```text
VIX
Valuation
Credit
Macro
```

를 incremental challenger로 추가한다.

---

# 23. Trend / Momentum 후보

예:

```text
12-1 momentum
6-1 momentum
10-month moving average
12-month trend
```

하나의 exact parameter를 미리 정답으로 잡지 않는다.

parameter plateau를 찾는다.

---

# 24. Realized Volatility

후보:

```text
20-day
63-day
126-day
252-day
```

및:

```text
short-vol / long-vol ratio
rolling percentile
```

등을 비교할 수 있다.

Volatility scaling을 넣는 경우 expected return 감소와 turnover 증가까지 함께 평가한다.

---

# 25. Drawdown

현재 가격과 running peak를 이용한 drawdown을 계산한다.

다음과 같이 단순 threshold만 최적화하지 말고:

```text
-10%
-15%
-20%
-25%
```

연속형 score 또는 percentile 방식도 비교한다.

---

# 26. VIX

VIX는 기본 전략이 아니라 **추가 feature**로 취급한다.

테스트:

```text
Model A:
Trend + Realized Vol + Drawdown

Model B:
Trend + Realized Vol + Drawdown + VIX
```

B가 out-of-sample에서 일관된 개선을 제공하지 못하면 VIX를 제거한다.

`VIX > 20`, `VIX > 30` 같은 인간이 임의로 만든 threshold를 기본적으로 신뢰하지 않는다.

Absolute threshold와 rolling percentile을 비교한다.

---

# 27. Defensive Timing과 Contrarian Buying을 분리

이 둘은 다른 전략이다.

### Defensive

```text
Trend deterioration
Volatility increase
↓
risk exposure 감소
```

목표:

```text
MDD / tail risk 감소
```

### Contrarian

```text
Large drawdown / stress
↓
risk exposure 증가
```

목표:

```text
mean reversion / higher expected return 활용
```

둘을 하나의 signal에 섞지 말고 독립적으로 검증한 뒤 결합한다.

---

# 28. 가상의 추가 자금 금지

전략마다 외부 현금 유입은 동일해야 한다.

금지:

```text
Normal:
100만원 투자

Crash:
300만원 투자
```

하면서 추가 200만원의 출처를 설명하지 않는 것.

허용:

```text
기존 cash reserve
defensive asset 매도
미리 누적된 현금
실제 외부 추가 cashflow
```

백테스트에서 돈을 생성하지 않는다.

---

# 29. Dynamic Overlay 제한

전략이 market timing에 과도하게 의존하지 않도록 hard bound를 둔다.

개념:

```text
target_t
=
strategic_target
+
tactical_adjustment
```

그리고:

```text
|tactical_adjustment|
<= tactical_budget
```

후보:

```text
0%
5%
10%
15%
20%
```

등을 연구한다.

최적값 하나가 아니라 안정적인 구간을 찾는다.

---

# 30. Risk Score 연구

여러 signal을 결합할 경우 즉시 복잡한 ML 모델을 사용하지 않는다.

우선:

```text
normalized trend
volatility percentile
drawdown score
optional VIX percentile
```

를 이용한 단순한 continuous risk score를 비교한다.

예:

```text
risk_score
=
a * trend
- b * volatility
- c * drawdown stress
- d * VIX stress
```

가중치는 정답으로 가정하지 않는다.

다음 방식들을 비교할 수 있다.

```text
equal weight
rank aggregation
simple linear model
regularized regression
```

ML은 단순 모델이 충분하지 않은 근거가 있을 때만 challenger로 추가한다.

---

# 31. Currency Engine

Baseline은 FX forecasting을 하지 않는다.

```text
KRW cashflow
↓
투자 시 필요한 만큼 환전
↓
investment
```

부터 시작한다.

다음은 challenger다.

```text
rolling FX percentile
MA-based timing
PPP / valuation
fixed hedge
partial hedge
dynamic hedge
```

FX timing과 FX risk management를 구분한다.

---

# 32. Trading Currency와 Economic Exposure 구분

반드시 데이터 모델에 구분한다.

```text
Trading Currency
≠
Underlying Economic Currency Exposure
```

USD로 거래되는 ETF라도 해외 자산을 담으면 underlying currency risk가 존재할 수 있다.

---

# 33. ETF Mapping Engine

경제적 target exposure가 결정된 후 ETF를 선택한다.

흐름:

```text
Target Exposure
      ↓
Candidate ETFs
      ↓
Hard Filters
      ↓
Exposure Fit
      ↓
Cost
      ↓
Tracking
      ↓
Liquidity
      ↓
Tax
      ↓
Final ETF
```

---

# 34. ETF Hard Filters

초기 production candidate에서 기본적으로 다음을 제한하는 것을 검토한다.

```text
Leveraged ETF
Inverse ETF
Single-stock ETF
극단적인 derivative 구조
극도로 낮은 AUM
극도로 낮은 liquidity
지나치게 짧은 track record
폐쇄 위험이 큰 ETF
```

연구 목적에서는 별도 override 가능하게 한다.

---

# 35. ETF Score

ETF score는 최근 performance chasing이 아니라 implementation quality 기반이어야 한다.

예:

```text
ETFScore
=
Exposure Fit
- Expense penalty
- Tracking Difference penalty
- Spread penalty
- Liquidity penalty
- Tax penalty
- Operational risk
```

검토할 필드:

```text
expense ratio
tracking difference
tracking error
AUM
volume
bid-ask spread
holdings
index methodology
fund structure
tax treatment
issuer
inception date
```

---

# 36. ETF 교체 규칙

조금 더 좋은 ETF가 발견될 때마다 매도하지 않는다.

Hysteresis를 적용한다.

개념:

```text
Expected Future Benefit
>
Switching Cost
+ Tax Cost
+ Safety Margin
```

추가로:

```text
minimum score improvement
minimum holding period
cooldown
migration-by-new-contribution
```

를 검토한다.

---

# 37. Research Engine과 Final Simulator 분리

다음 두 엔진을 구분한다.

### Research Engine

목적:

```text
빠른 parameter exploration
signal test
sensitivity analysis
```

후보:

```text
vectorbt
NumPy
Polars
```

---

### Final Validation Engine

목적:

```text
실제 cashflow
FX
orders
fills
dividends
fees
tax
rebalance
corporate actions
```

를 현실적으로 처리.

가능하면 custom event-driven simulator를 구축한다.

---

# 38. Same-Bar Look-Ahead 금지

예:

```text
Jan 31 close
↓
signal 계산
↓
Jan 31 close에 체결
```

을 금지한다.

최소:

```text
Jan 31 data 확정
↓
signal 계산
↓
다음 거래 가능한 시점
↓
execution
```

이어야 한다.

각 signal에:

```text
signal_at
execution_at
```

을 명시하는 것이 좋다.

---

# 39. Dividend 처리

다음 중 하나만 선택한다.

```text
Adjusted total-return price
```

또는

```text
Raw price + dividend cashflow
```

두 방법을 동시에 사용하여 dividend를 중복 계산하지 않는다.

---

# 40. Portfolio Ledger

최종 시뮬레이터에는 accounting ledger를 둔다.

최소 기록:

```text
timestamp

external_contribution_krw

cash_krw
cash_usd

fx_rate
fx_fee

ticker
quantity
cost_basis

price
market_value_usd
market_value_krw

dividend
commission
spread_cost
slippage
tax

realized_pnl
unrealized_pnl

target_weight
actual_weight

strategy_version
signal_snapshot_id
```

ledger를 portfolio 상태의 Single Source of Truth로 사용한다.

---

# 41. 비용 모델

처음부터 거래비용 0 하나만 사용하지 않는다.

Scenario:

```text
Ideal
Low
Base
Stress
```

를 둔다.

분리해서 기록:

```text
commission
spread
slippage
FX conversion cost
expense ratio
tax drag
```

---

# 42. 세금

세법을 strategy logic 내부에 hard-code하지 않는다.

Interface:

```python
class TaxModel:
    on_dividend(...)
    on_sale(...)
    year_end(...)
```

형식으로 분리한다.

현재 한국 거주 개인의 해외 ETF 과세 제도는 실제 구현 시 공식 최신 자료로 다시 확인한다.

---

# 43. 성과지표

최소한 다음을 구현한다.

```text
Terminal Wealth KRW
Real Terminal Wealth KRW
XIRR
TWR
CAGR when meaningful

Annualized Volatility
MDD
Drawdown Duration
Ulcer Index
Sortino
Sharpe

Worst 1Y
Worst 3Y
Tail loss

Turnover
Trade count
Total transaction cost
FX cost
Tax drag

Average portfolio drift
Tracking error against target
```

Accumulation portfolio에서는 CAGR 하나가 주요 metric이 되어서는 안 된다.

---

# 44. 한국 물가 기준 실질자산

가능하면 한국 CPI를 사용하여:

```text
Real KRW Wealth
```

를 계산한다.

명목 USD 수익만 보고 전략을 평가하지 않는다.

---

# 45. Ablation Test

모듈 하나씩 추가한다.

예:

```text
M0
Global DCA

M1
M0 + Strategic alternatives

M2
M1 + Fixed Factor Tilt

M3
M2 + Contribution Rebalancing

M4
M3 + Trend

M5
M4 + Realized Volatility

M6
M5 + Drawdown

M7
M6 + VIX

M8
M7 + Contrarian Overlay

M9
M8 + Currency Management

M10
M9 + Dynamic Exposure Selection

M11
M10 + Dynamic ETF Implementation
```

각 단계의 marginal contribution을 기록한다.

복잡성만 증가하고 개선이 없는 모듈은 삭제한다.

---

# 46. Walk-Forward Validation

Random train/test split을 사용하지 않는다.

Time-ordered walk-forward를 적용한다.

예:

```text
Train
↓
parameter/model 선정
↓
향후 Test
↓
roll forward
↓
다시 Train
↓
다음 Test
```

필요하면:

```text
scikit-learn TimeSeriesSplit
```

을 활용하되 금융 시계열 특성에 맞는 gap도 검토한다.

---

# 47. Parameter Robustness

다음을 금지한다.

```text
가장 높은 CAGR를 만든 parameter 한 개
→ 최종 parameter
```

대신 parameter surface 전체를 본다.

찾아야 하는 것은:

```text
sharp optimum ❌
broad stable plateau ✅
```

이다.

예:

```text
Momentum lookback
9
10
11
12
13
14개월
```

주변 구간에서도 성능이 유지되는지를 확인한다.

---

# 48. Bootstrap / Monte Carlo

시계열 구조를 완전히 깨는 무작위 shuffle은 피한다.

우선 검토:

```text
Stationary Bootstrap
Block Bootstrap
```

후보 library:

```text
arch
```

평가:

```text
Terminal Wealth distribution
Real Wealth distribution
MDD distribution
XIRR distribution
Probability of underperforming baseline
```

---

# 49. Multiple Testing / Data Snooping

많은 전략과 parameter를 탐색한다면 반드시 multiple-testing 문제를 인정한다.

가능하면:

```text
SPA
StepM
MCS
```

같은 방법을 검토한다.

특히 후보 전략 수가 많아질수록:

```text
best backtest
```

만 보고 선택하지 않는다.

---

# 50. Rolling Cohort Test

장기 적립 전략에서는 시작 연도에 따른 결과 차이가 매우 중요하다.

예:

```text
1995 시작 투자자
1996 시작 투자자
...
2010 시작 투자자
```

등 rolling starting cohort를 만들어 비교한다.

각 cohort에서:

```text
10Y
15Y
20Y
```

성과 분포를 본다.

---

# 51. Stress Test

최소 다음 시장환경을 별도로 분석한다.

```text
Dot-com crash
Global Financial Crisis
COVID crash
Inflation/rate shock
Long bull market
Sideways market
Rapid recovery
High inflation
Strong USD
Weak USD
```

단, 특정 사건에 맞춰 parameter를 튜닝하지 않는다.

---

# 52. 연구 합격 기준

새로운 모듈이나 전략은 최소한 다음 조건을 검토한 뒤 채택한다.

```text
1. In-sample improvement
2. Out-of-sample improvement
3. Parameter stability
4. Rolling-period stability
5. Transaction-cost robustness
6. Bootstrap robustness
7. Tail-risk degradation 여부
8. Complexity 대비 개선폭
9. 경제적 설명 가능성
10. 실제 ETF 구현 가능성
```

한두 개 metric만 좋아졌다고 채택하지 않는다.

---

# 53. Complexity Penalty

복잡한 전략에 기본적으로 불리한 기준을 적용한다.

예:

```text
Strategy A
4 ETFs
2 signals
monthly execution

Strategy B
15 ETFs
12 signals
weekly execution
```

B가 미세하게 성과가 좋다고 해서 B를 선택하지 않는다.

복잡성이 정당화될 만큼 OOS 개선이 커야 한다.

---

# 54. 필수 프로젝트 Invariant

다음은 AI가 임의로 변경하지 말아야 하는 핵심 원칙이다.

```text
1. Future information 사용 금지

2. Same-bar signal/fill 금지

3. Survivorship bias 방치 금지

4. 현재 ETF metadata를 과거에 적용 금지

5. ETF보다 economic exposure를 먼저 정의

6. 전략 간 external cashflow 조건 동일

7. 가상의 추가 투자자금 생성 금지

8. Adjusted price + dividend 중복 계산 금지

9. 최고 성능 parameter 하나만 선택 금지

10. OOS 검증 없이 production candidate 채택 금지

11. Research proxy와 actual ETF performance 혼동 금지

12. Missing data를 조용히 임의 보정 금지

13. 데이터 lineage 기록

14. 모든 전략 결과 재현 가능하게 seed/config/version 기록

15. Simple baseline을 항상 유지
```

---

# 55. 기술 스택 검토

다음 후보를 검토하되 필요하면 더 적절한 라이브러리로 교체하라.

```text
Python 3.12+

httpx
tenacity

polars
pandas
numpy
scipy

pyarrow
duckdb

statsmodels
scikit-learn

cvxpy
PyPortfolioOpt

vectorbt

arch

exchange_calendars

pydantic
PyYAML

matplotlib
plotly

pytest
hypothesis
```

각 dependency가 왜 필요한지 설명한다.

불필요하게 많은 dependency를 추가하지 않는다.

---

# 56. 권장 코드 구조

다음을 초안으로 삼되 더 좋은 구조가 있다면 수정한다.

```text
src/
├── data/
│   ├── providers/
│   ├── normalization/
│   ├── quality/
│   ├── point_in_time/
│   └── storage/
│
├── universe/
│   ├── assets/
│   ├── exposures/
│   ├── etfs/
│   └── historical_universe/
│
├── features/
│   ├── returns/
│   ├── factors/
│   ├── momentum/
│   ├── trend/
│   ├── volatility/
│   ├── drawdown/
│   ├── macro/
│   └── fx/
│
├── strategy/
│   ├── strategic/
│   ├── factors/
│   ├── risk_overlay/
│   ├── currency/
│   └── etf_mapping/
│
├── portfolio/
│   ├── targets/
│   ├── contribution/
│   ├── rebalance/
│   └── optimizer/
│
├── simulation/
│   ├── events/
│   ├── execution/
│   ├── broker/
│   ├── ledger/
│   ├── fees/
│   ├── tax/
│   └── corporate_actions/
│
├── validation/
│   ├── walk_forward/
│   ├── bootstrap/
│   ├── multiple_testing/
│   ├── sensitivity/
│   ├── ablation/
│   └── cohorts/
│
└── analytics/
    ├── performance/
    ├── risk/
    └── reporting/
```

---

# 57. Config-driven architecture

전략 parameter를 코드에 하드코딩하지 않는다.

예:

```yaml
strategic:
  model: global_market

factors:
  enabled: true
  sleeves:
    - value
    - profitability
    - momentum

risk_overlay:
  enabled: false

contribution:
  method: target_gap

execution:
  signal_frequency: monthly
  fill_delay_sessions: 1
```

단 이 YAML 값들은 예시일 뿐이며 최종 전략 parameter가 아니다.

---

# 58. 재현성

모든 실험은 다음을 남겨야 한다.

```text
experiment_id
git commit
strategy config
dataset version
data manifest hash
execution model
cost model
tax model
random seed
start/end date
metrics
```

실험 결과만 보고 어떤 데이터와 코드로 생성됐는지 알 수 없는 상태를 금지한다.

---

# 59. 테스트 전략

Unit test뿐 아니라 financial invariant test를 작성한다.

예:

```text
portfolio weights sum
cash conservation
no negative cash unless allowed
external cashflow conservation
dividend accounting
split handling
FX conversion accounting
fee accounting
same-bar prevention
PIT availability
rebalance constraint
```

Hypothesis 같은 property-based testing도 적극 검토한다.

---

# 60. 우선 구현 순서

한번에 전체 시스템을 만들지 않는다.

## Phase 0 — Research specification

먼저:

```text
목표
가설
benchmark
데이터
편향
평가 metric
```

문서화.

---

## Phase 1 — Data Foundation

구현:

```text
providers
raw storage
normalization
Parquet
DuckDB
PIT metadata
quality tests
```

아직 복잡한 전략 금지.

---

## Phase 2 — Simple Baselines

구현:

```text
Global DCA
US DCA
basic contribution schedule
KRW performance
FX conversion
```

---

## Phase 3 — Strategic Allocation

```text
regional diversification
defensive assets
simple allocation alternatives
```

---

## Phase 4 — Factors

```text
factor dataset
factor attribution
fixed factor tilt
ETF factor regression
```

---

## Phase 5 — Contribution Optimization

```text
buy-only rebalancing
rebalance band
cost-aware allocation
```

---

## Phase 6 — Dynamic Overlay

순서:

```text
trend
volatility
drawdown
VIX incremental test
contrarian
```

---

## Phase 7 — Currency

```text
baseline conversion
hedge variants
FX timing challenger
```

---

## Phase 8 — ETF Mapping

```text
metadata
exposure fit
cost
tracking
liquidity
switching rule
```

---

## Phase 9 — Robust Validation

```text
walk-forward
parameter surface
bootstrap
cohorts
multiple-testing
stress scenarios
```

---

## Phase 10 — Paper Execution

research가 안정된 후에만 구현한다.

```text
Broker interface
PaperBroker
order generation
position reconciliation
```

실거래 연결은 최종 단계로 남긴다.

---

# 61. 현재 가장 유력한 전략 hypothesis

현재 단계에서 아래 구조를 **우선 검증 후보**로 사용하라.

정답으로 취급하지 않는다.

```text
Strategic Global Beta
        +
Diversified Fixed Factor Tilts
        +
Contribution-based Rebalancing
        +
Small Bounded Trend/Risk Overlay
        +
Simple Currency Execution
        +
Low-cost ETF Implementation
```

수식으로 개념화하면:

```text
Portfolio Target_t
=
Strategic Target
+
Fixed Factor Target
+
Bounded Dynamic Adjustment_t
```

그리고:

```text
Orders_t
=
Contribution Optimizer(
    current holdings,
    target weights,
    new external cashflow,
    costs
)
```

ETF 선택은:

```text
ETF
=
Best Implementation(
    required exposure,
    tracking quality,
    cost,
    liquidity,
    tax,
    operational risk
)
```

로 본다.

---

# 62. 우선순위가 낮은 전략

다음은 삭제 대상이 아니라 후순위 challenger다.

```text
Fear & Greed timing
absolute VIX thresholds
large crash cash reserve
FX moving-average timing
aggressive factor timing
full ETF rotation
machine-learning return prediction
frequent trading
```

단순 전략보다 명확한 OOS 개선을 보여줄 때만 추가한다.

---

# 63. AI의 자율 판단 범위

다음 사항은 자유롭게 비판하고 수정할 수 있다.

```text
asset classes
factor set
factor weights
strategic weights

trend definition
momentum lookback
volatility window
drawdown metric

risk score construction
tactical budget
rebalance band

ETF ranking weights
ETF switching criteria

FX hedge
cost assumptions

specific Python library
folder structure
```

단 반드시 수정 근거를 설명한다.

---

# 64. AI가 피해야 할 행동

다음을 하지 말라.

```text
- 새로운 전략을 많이 추가하는 것이 좋은 설계라고 판단
- ML을 이유 없이 사용
- 모든 가능한 parameter grid를 brute-force하고 최고 결과 선택
- 현재 좋은 ETF를 보고 과거 전략 설계
- 샘플이 짧은 ETF 성과를 장기 factor 성과로 해석
- 벤치마크를 약하게 설정
- 비용을 0으로만 계산
- 세금과 환율을 완전히 무시
- 데이터가 없는데 임의 proxy를 조용히 사용
- 오류를 피하기 위해 missing 값을 무조건 forward-fill
- 미래정보를 사용한 결과를 좋은 백테스트라고 보고
- 과도한 abstraction으로 실제 로직을 이해하기 어렵게 만들기
```

---

# 65. 첫 번째 작업에서 요구하는 출력

바로 전체 구현에 들어가지 말고 먼저 아래 내용을 작성하라.

## A. Architecture Review

현재 제안에 대해:

```text
유지할 것
수정할 것
제거할 것
추가할 것
```

을 분류한다.

---

## B. Data Feasibility Matrix

다음 형식으로 작성한다.

| Data | Source | History | PIT 가능 | 무료 여부 | 용도 | 위험 |
| ---- | ------ | ------: | ------ | ----- | -- | -- |

특히:

```text
ETF price
historical ETF universe
factor
VIX
FX
rates
CPI
macro
ETF metadata
bid-ask
expense
```

를 포함한다.

---

## C. Bias Audit

다음을 각각 평가한다.

```text
Look-ahead bias
Survivorship bias
Selection bias
Data snooping
Revision bias
ETF inception bias
Delisting bias
Corporate-action errors
Currency timing leakage
Same-bar execution
```

---

## D. Research Experiment Matrix

최소:

```text
M0 ~ M11
```

실험군을 제안하고:

```text
hypothesis
added module
required data
parameters
benchmark
acceptance criteria
```

를 작성한다.

---

## E. Final Architecture

다음을 포함한 아키텍처를 제시한다.

```text
Data Layer
Feature Layer
Research Layer
Portfolio Layer
Simulation Layer
Validation Layer
ETF Implementation Layer
Future Live Execution Layer
```

---

## F. Implementation Roadmap

각 Phase별로:

```text
목표
구현 모듈
필요 데이터
테스트
완료 조건
```

을 작성한다.

---

# 66. 구현을 시작하기 전 반드시 나에게 보여줄 결론

최종적으로 다음을 간결하게 정리하라.

```text
1. 당신이 판단한 가장 합리적인 전체 설계

2. 기존 제안 중 수정한 핵심 사항

3. 가장 중요한 5개의 연구 가설

4. 가장 위험한 5개의 백테스트 오류 가능성

5. 확보 가능한 데이터와 확보 불가능한 데이터

6. Phase 1에서 실제로 구현할 범위

7. 아직 구현하지 않을 항목과 이유
```

---

# 67. 구현 단계의 행동 원칙

설계 검토가 끝난 뒤 구현할 때는:

```text
작은 단위 구현
→ unit test
→ data test
→ baseline 실행
→ 결과 검증
→ 다음 기능 추가
```

순서를 유지한다.

복잡한 모든 기능을 한번에 구현하지 않는다.

각 기능을 추가할 때:

```text
Before
vs
After
```

성능 및 복잡도 변화를 확인할 수 있어야 한다.

---

# 68. 최종 성공 기준

이 프로젝트의 성공은:

```text
가장 높은 과거 수익률
```

을 발견하는 것이 아니다.

성공 기준은 다음이다.

> **논리적 근거가 명확하고, point-in-time 데이터로 재현 가능하며, 동일한 외부 cashflow 조건에서 단순 benchmark와 공정하게 비교되고, 다양한 시장구간과 OOS 검증에서 견고하며, 비용과 환율과 현실적 실행 조건을 고려해도 유지되는 장기 적립식 투자 정책을 만드는 것.**

최종 전략이 단순한 Global DCA와 거의 차이가 없다면 그 결과도 받아들인다.

복잡한 퀀트 전략이 단순 전략을 확실히 개선하지 못한다면:

```text
단순 전략을 최종 선택
```

하는 것도 성공적인 연구 결과다.

---

# 69. 지금 수행할 작업

위 요구사항을 기준으로:

1. 현재 repository와 구현 상태를 전수 분석하라.
2. 필요한 경우 공식 데이터/API 문서를 확인하라.
3. 현재 제안 아키텍처를 금융공학적·소프트웨어공학적으로 비판하라.
4. 과도하거나 잘못된 설계는 수정하라.
5. 데이터 가용성과 point-in-time 제약을 확인하라.
6. 최종 권장 Architecture v2를 설계하라.
7. M0~M11 연구 실험체계를 구체화하라.
8. 구현 roadmap과 dependency를 확정하라.
9. 아직 코드를 대규모로 수정하지 말고 먼저 분석 결과와 구현 계획을 제시하라.
10. 계획의 각 선택에 대해 근거와 trade-off를 설명하라.

가장 중요한 기준은 **복잡함이 아니라 robustness, 재현성, 실전 가능성, 데이터 무결성**이다.
