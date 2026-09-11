# 컴포넌트 상세 명세 (Component Catalog)

본 문서는 ETF-Manager 플랫폼을 구성하는 핵심 모듈들의 책임, 입출력 규격, 의존 관계 및 주요 구현 코드를 상세히 설명합니다.

---

## 1. 데이터 수집 및 공급자 어댑터 (Data Ingestion)

### `DataFetcher & Provider Adapters`

* **담당 업무 (Responsibility):** 외부 금융 데이터 제공처(Tiingo, FRED, ECOS, SEC EDGAR 등)의 REST API를 호출하여 시장 가격, 거시경제 지표, ETF 분기 공시를 안정적으로 수집합니다. 호출 제한(Rate Limit)을 준수하고 지수 백오프 기반 재시도를 수행합니다.
* **입력 (Input):** 수집 대상 기간(`start`, `end`), 종목 티커 목록, API 인증 키.
* **출력 (Output):** 원본 데이터 파일 (`data/raw/<provider>/<dataset>/<sha256>`).
* **의존 모듈 (Dependencies):** `httpx`, `tenacity`, `src/data/settings.py`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/data/fetch.py`, `src/data/providers/tiingo.py`, `src/data/providers/fred.py`, `src/data/providers/ecos.py`, `src/data/providers/sec_nport.py`
  * 함수: `fetch_and_persist_prices()`, `fetch_and_persist_macro()`, `fetch_and_persist_fx()`, `fetch_and_persist_cpi()`

---

## 2. 시점 정합성 및 미래 정보 차단 엔진 (Point-in-Time Core)

### `PointInTimeEngine`

* **담당 업무 (Responsibility):** 수집된 데이터에 공식 공시 시점(`available_at`)을 계산해 붙이고, 특정 의사결정 시점 $t$ 기준으로 시장에 이미 공개되어 있던 데이터만 추려냅니다. 시뮬레이션 중 미래 시점 데이터가 유입되면 즉시 예외를 발생시킵니다.
* **입력 (Input):** 정규화된 Polars DataFrame, 데이터셋 규격(`DatasetSpec`), 의사결정 시각 $t$.
* **출력 (Output):** 공시 시점이 부여된 DataFrame, 의사결정 시점 $t$ 기준의 안전한 시계열 슬라이스.
* **의존 모듈 (Dependencies):** `polars`, `exchange-calendars`, `src/data/calendar.py`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/data/pit.py`
  * 함수: `stamp_availability()` (가용 시점 태깅), `as_of()` (시점 기준 조회), `assert_no_lookahead()` (미래 정보 누출 감지)
  * 예외: `LookAheadError`

---

## 3. 데이터 품질 검증기 (Quality Gate)

### `QualityGate`

* **담당 업무 (Responsibility):** 데이터가 저장소에 들어가기 전 10가지 품질 규칙(컬럼 스키마, 키 중복 여부, 필수 컬럼 결측치, 가격의 논리적 모순 등)을 검증하여 오염된 데이터의 유입을 원천 차단합니다.
* **입력 (Input):** 정규화된 Polars DataFrame, 데이터셋 스펙.
* **출력 (Output):** 검증 결과 리포트(`QualityReport`). 치명적 오류(`ERROR`) 발견 시 입력을 수정하지 않고 즉시 `DataQualityError`를 던집니다.
* **의존 모듈 (Dependencies):** `polars`, `src/data/schema.py`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/data/quality.py`
  * 클래스/함수: `QualityGate.enforce()`, `QualityReport`, `DataQualityError`

---

## 4. 불변 저장소 및 무결성 검증기 (Storage Engine)

### `StorageEngine`

* **담당 업무 (Responsibility):** 검증을 통과한 데이터를 불변 Parquet 파일로 저장하고, 파일 내용의 정렬된 SHA-256 해시값을 담은 매니페스트(JSON)를 함께 생성합니다. 데이터 조회 시 해시값을 다시 검산하여 1바이트의 변조도 허용하지 않습니다.
* **입력 (Input):** 품질 검증을 통과한 DataFrame, 품질 리포트.
* **출력 (Output):** Parquet 파일 (`data/normalized/...`), 매니페스트 JSON (`data/manifests/...`).
* **의존 모듈 (Dependencies):** `polars`, `pyarrow`, `hashlib`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/data/storage.py`
  * 함수: `write_dataset_partition()`, `read_dataset_partition()`, `canonical_frame_sha256()`
  * 예외: `UntrustedDatasetError`

---

## 5. 지표 및 팩터 계산기 (Feature Engine)

### `FeatureEngine`

* **담당 업무 (Responsibility):** 과거 시점 데이터만을 사용하여 이동평균, 실현 변동성, 전고점 대비 최대 낙폭(MDD), Fama-French 팩터 민감도(Beta), 한국 금융상황지수(KAFI) 등 전략에 필요한 지표들을 산출합니다.
* **입력 (Input):** 시점 정합성(PIT)이 보장된 주가, 환율, 거시지표 DataFrame.
* **출력 (Output):** 날짜 및 종목별 피처 DataFrame.
* **의존 모듈 (Dependencies):** `polars`, `src/data/pit.py`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/features/returns.py`, `src/features/risk.py`, `src/features/drawdown.py`, `src/features/factors.py`, `src/features/kafi.py`
  * 함수: `calculate_returns()`, `realized_volatility()`, `drawdown_series()`, `estimate_factor_loadings()`

---

## 6. 자산배분 정책 및 목표 비중 산출기 (Policy Layer)

### `PolicyTargetResolver`

* **담당 업무 (Responsibility):** 전략 ID(`PolicyId`, 예: `qqq`)와 경제 가설에 따라 각 자산의 목표 투자 비중(가중치 합계 = 1.0)을 계산합니다. 필요 시 하락장 방어 오버레이나 환전 지연 규칙을 결합합니다.
* **입력 (Input):** 전략 식별자, 의사결정 시각, 피처 DataFrame.
* **출력 (Output):** 종목별 목표 가중치 딕셔너리 (`{티커: 비중}`).
* **의존 모듈 (Dependencies):** `src/policy/targets.py`, `src/policy/thesis.py`, `src/policy/overlay.py`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/policy/targets.py`, `src/policy/overlay.py`, `src/policy/reserve.py`
  * 함수: `resolve_targets()`, `apply_operational_contribution_lock()`, `apply_bounded_overlay()`

---

## 7. 매수 전용 적립식 시뮬레이터 (Simulation Engine)

### `SimulationEngine & CashflowMixer`

* **담당 업무 (Responsibility):** 매월 들어오는 100만 원의 적립금을 기존 주식 매도 없이 목표 비중보다 부족한 종목에만 배분(`allocate_contribution`)하고, 익거래일($t+1$) 종가와 환율, 거래 수수료를 적용해 정수 주수 단위로 모의 체결을 진행합니다.
* **입력 (Input):** 시뮬레이션 설정(`AllocationConfig`), 과거 가격/환율/CPI 데이터.
* **출력 (Output):** 매월 자산 스냅샷과 최종 실질 원화 자산($W^{\text{real}}$), 실질 내부수익률(Real XIRR)이 담긴 `AllocationResult`.
* **의존 모듈 (Dependencies):** `src/sim/contribution.py`, `src/sim/lots.py`, `src/analytics/metrics.py`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/sim/allocation.py`, `src/sim/contribution.py`, `src/sim/lots.py`
  * 함수: `run_allocation()`, `allocate_contribution()`, `fill_integer_buys()`
  * 클래스: `AllocationConfig`, `AllocationSnapshot`, `AllocationResult`

---

## 8. 통계적 검증 및 가설 채택 게이트 (Validation Gate)

### `ValidationGate & PostureKernel`

* **담당 업무 (Responsibility):** 시뮬레이션 결과가 단순한 우연이나 과적합이 아닌지 통계적으로 검증합니다. 위험회피 성향을 반영한 확실성 등가 수익률(CE)을 계산하고, 120개월 롤링 코호트, 12개월 블록 부트스트랩, 비용 악화 시나리오를 통과한 전략만 표준 정책으로 승격시킵니다.
* **입력 (Input):** 후보 전략과 기준선 전략의 시뮬레이션 결과(`AllocationResult`).
* **출력 (Output):** 최종 채택 통과 여부(Boolean), 분위수 통계량($p_{05}$, $p_{10}$, 중위수 비율).
* **의존 모듈 (Dependencies):** `src/validation/gate.py`, `src/validation/bootstrap.py`, `src/validation/accumulation_cohort.py`, `src/validation/cost_grid.py`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/validation/gate.py`, `src/validation/bootstrap.py`, `src/validation/accumulation_cohort.py`, `src/validation/research_posture.py`
  * 함수: `certainty_equivalent()`, `adoption_passes()`, `moving_block_bootstrap()`, `run_accumulation_cohorts()`

---

## 9. ETF 종목 매핑기 (ETF Mapper)

### `ETFMapper`

* **담당 업무 (Responsibility):** 추상적인 자산군(예: 미국 대형주, 반도체)을 실제 시장에 상장된 ETF(예: QQQ, SOXX)로 연결합니다. SEC 공시 데이터를 바탕으로 운용 보수, 거래 대금, 추적 오차를 점수화하며, 사소한 점수 차이로 종목을 자주 교체해 수수료를 낭비하지 않도록 완충 기준(Hysteresis buffer)을 적용합니다.
* **입력 (Input):** 목표 자산군 비중, SEC ETF 메타데이터 DataFrame.
* **출력 (Output):** 실제 상장 ETF 티커와 매핑된 비중.
* **의존 모듈 (Dependencies):** `src/etf/sleeves.py`, `src/etf/score.py`, `src/etf/mapping.py`.
* **주요 구현 (Key Implementation):**
  * 모듈: `src/etf/mapping.py`, `src/etf/score.py`, `src/etf/sleeves.py`
  * 함수: `apply_etf_mapping()`, `score_etf()`, `resolve_vehicle()`
