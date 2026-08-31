# ETF-Manager (ETF 적립식 매수 최적화 및 퀀트 검증 플랫폼)

> **Point-in-Time(PIT) 데이터 무결성**, **현실적 거래비용·환율 모델링**, **동일 외부 현금흐름(Identical External Cashflows)** 제약 하에서 실질 원화(Real KRW) 최종 자산을 극대화하는 적립 정책을 **반증·수렴** 단계까지 검증하는 플랫폼입니다.

---

## 📌 1. 프로젝트 개요 (Overview)

**ETF-Manager**는 단순 백테스트를 넘어 Look-ahead bias, 암묵적 cashflow 왜곡, research overfitting을 엄격히 통제하는 **기관급 퀀트 리서치 엔진**입니다.

### 🎯 핵심 질문
> *"동일한 월 적립금(외부 cashflow) 하에서, PIT 제약과 현실 비용을 반영할 때 10년 이상 실질 원화 terminal wealth를 극대화하는 정책은 무엇인가?"*

### 🔒 운용 락 (2026-08-30, `docs/architecture/00_system_overview.md`와 동기)

| 항목 | 값 |
| --- | --- |
| Policy | `qqq` (`PolicyId.QQQ`) |
| Targets | **QQQ 90% / SOXX 10%** (`OPERATIONAL_TARGETS_OVERRIDE`) |
| Contribution | **flat** (adaptive 미부착) |
| Benchmark | **QQQ 100%** — immutable benchmark (`StrategyRole.IMMUTABLE_BENCHMARK`) |
| Adaptive v5 | **frozen research-only** (`FROZEN_ADAPTIVE_V5`), operational path에 미연결 |
| Seen history | `≤ 2026-08-28` — 이후 데이터만 prospective OOS |

### 핵심 불변식 (Invariants)
1. **PIT 무결성**: `available_at` / delayed fill 분리.
2. **동일 외부 cashflow**: capital-allocation 비교는 월 적립액 동일; timing 연구는 reserve/KAFI deployment로 분리.
3. **Buy-only accumulation**: 매도 없이 신규 적립금만 배분.
4. **CE adoption gate**: 복잡도 허들 `δ₀·m` (research challenger 채택용; operational은 flat lock).
5. **WF reject (I14)**: `train_adopted=False` ⇒ `chosen_test_arm == baseline_test_arm` (identity).

---

## 📊 2. 전략(Policy) 카탈로그 (요약)

| PolicyId | Sleeves | 분류 | 상태 |
| :--- | :--- | :--- | :--- |
| **`qqq`** | **QQQ 90% / SOXX 10%** | **Operational lock** | flat provisional incumbent |
| `qqq` + QQQ 100% targets | QQQ 100% | Benchmark | immutable benchmark (override) |
| `vti` | VTI 100% | CE baseline | walk-forward/ablation 기준선 |
| `vt` | VT 100% | Baseline | 글로벌 분산 참조 |
| `ff_proxy` | French daily market | Research proxy | ETF 실행과 분리 (pre-history stress) |
| 기타 | — | Research | CE gate 미통과 또는 기각 |

상세: `docs/architecture/02_policy_and_validation.md`

---

## 🏗️ 3. 아키텍처

6계층 단방향 구조 (`L1 Data` → `L6 ETF mapping`). 상세 topology 및 invariant 목록은 `docs/architecture/00_system_overview.md`.

**연구 수렴 단계 (2026-08)**  
검증 오류 수정 → objective 분리 → `FINAL_HISTORICAL_CAMPAIGN_V1` freeze → prospective registry (`PROSPECTIVE_2026_V1`) → monitoring.

---

## 🔬 4. 검증 프레임워크

- **Cohort ablation / walk-forward / cost-grid**: CE 및 growth gate.
- **Final historical campaign** (`run final-historical-campaign`): B0 QQQ100 + SOXX 5/10/15 flat only; cost·FX stress, regime coverage, trial lineage, FF proxy pre-history diagnostic, tax milestone (`not_modelled`).
- **Prospective monitoring**: frozen bundle identity hash; `as_of > 2026-08-28`만 기록.

CE 채택 조건 (research):

$$\forall \gamma \in \{2, 5, 10\}: \quad \frac{\mathrm{CE}_\gamma(k)}{\mathrm{CE}_\gamma(B)} > 1 + \delta_0 \cdot m_k$$

---

## 🚀 5. Quickstart

```bash
uv sync --all-groups
export TIINGO_API_KEY=...   # ingest 시
export FRED_API_KEY=...
export ECOS_API_KEY=...
```

### 운용 시뮬레이션 (flat QQQ90/SOXX10)

```bash
uv run python -m src.cli run policy \
  --id qqq \
  --start 2016-07-01 --end 2026-06-30 \
  --contribution-krw 1000000
```

`apply_operational_contribution_lock`이 bare `qqq` monthly path에 QQQ90/SOXX10을 부착합니다.

### 주요 validation CLI

```bash
# Walk-forward (단일 candidate)
uv run python -m src.cli run walk-forward --config configs/experiments/wf_qqq95_soxx5_adaptive_v5.json

# Cost grid (commission + FX scenarios)
uv run python -m src.cli run walk-forward-costs --config configs/experiments/wf_qqq95_soxx5_adaptive_v5.json

# Final historical campaign (reporting-only freeze)
uv run python -m src.cli run final-historical-campaign \
  --config configs/experiments/final_historical_campaign_v1.json \
  --seed 42

# Prospective monitor (post-cutoff as_of only)
uv run python -m src.cli run prospective-monitor \
  --bundle configs/prospective/registry/prospective_2026_v1_frozen.json \
  --as-of 2026-09-30
```

전체 명령: `docs/architecture/03_operator_cli.md`

---

## 🧪 6. 품질 관리

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

---

## 📜 7. 라이선스

내부 연구 및 포트폴리오 관리 목적.
