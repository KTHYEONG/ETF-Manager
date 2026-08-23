# ETF-Manager — 운영·연구 로드맵

> **최종 목표:** 한국 거주 근로소득자가 매월 꾸준히 ETF 적립식 장기투자를 하며, 비용·환율·실질 KRW 기준으로 **신뢰 가능한 복리 자산증식**을 달성하는 포트폴리오를 구축한다.

이 문서는 초기 설계 명세가 아니라 **현재 구현 상태와 검증 결과를 반영한 운영·연구 방향**이다. P0–P10 및 Wave A–E는 구현·실행 완료 또는 기각되었다.

---

## 1. 운영 정책 (현재 잠금)

| 항목 | 값 |
| --- | --- |
| **PolicyId** | `S1_US` |
| **경제적 exposure** | US 전체 주식시장 |
| **구현 ETF** | VTI 100% (1슬리브) |
| **적립** | 매월 고정 KRW (`monthly_contribution_krw`) |
| **리밸런싱** | Buy-only (신규 납입으로만 비중 조정, 기본 매도 없음) |
| **모듈** | Strategic only (`modules = 0`) |

**채택 근거:** Campaign A (S0 vs S1 walk-forward), M1 ablation (S2–S4 vs S1), Wave D (S7 IVV vs S1), M2 (S6 VTI/VTV vs S1) 모두 CE adoption gate 실패 또는 S1 유지.

다른 ETF·비중·타이밍은 **연구 후보**이며, CE gate + walk-forward OOS를 통과하기 전까지 운영 정책을 바꾸지 않는다.

---

## 2. 시스템이 최적화하는 것 (코드 계약)

단순 CAGR 최대가 아니다.

$$
\mathrm{CE}_\gamma = \left(\frac{1}{N}\sum_{i=1}^{N}\left(W_i^{\text{real}}\right)^{1-\gamma}\right)^{\frac{1}{1-\gamma}},
\quad \gamma \in \{2, 5, 10\}
$$

- $W^{\text{real}}$: 한국 CPI로 디플레이트한 실질 KRW 종료자산
- **동일 외부 현금흐름(I5):** 모든 후보는 같은 월 적립액·같은 기간
- **복잡도 패널티:** 후보 채택 조건 $\mathrm{CE}_\gamma(k)/\mathrm{CE}_\gamma(B) > 1 + \delta_0 \cdot m_k$ ($m_k$ = 추가 모듈 수)

비용·환율·세금은 목적함수 항이 아니라 **시뮬레이션 안에서 실현**된다.

---

## 3. 검증 완료 요약 (2026-08-23 기준)

### 기각된 전략 가설

| 실험 | 후보 | 결과 | 함의 |
| --- | --- | --- | --- |
| M1 | S2 지역, S3 채권, S4 방어 | 기각 | 다자산 분산·채권 슬리브가 S1을 이기지 못함 |
| Wave D | S7 IVV (S&P 500) | **기각** | 인기 SP500 ETF ≠ 더 나은 US exposure |
| M2 | S6 VTI 80% / VTV 20% | **기각** | US 내 팩터 틸트 분할도 CE gate 실패 |
| Wave E (진단) | QQQ | PolicyId 없음 | 성장 집중 베팅; 샘플 구간 수익 ↑, MDD ↑, 채택 경로 없음 |

### Wave E 진단 (설명용, 채택 아님)

동일 월 100만원 DCA (2012-06 ~ 2024-10, 카탈로그 가용 창):

| Vehicle | 실질 종료자산 | XIRR | SMB (factor) |
| --- | --- | --- | --- |
| VTI | 3.37억 | 15.8% | +0.013 |
| IVV | 3.38억 | 15.8% | −0.084 |
| QQQ | 4.89억 | 21.2% | −0.289 |

QQQ가 한 구간에서 앞선 것은 **정책 근거가 아니라** 성장·대형주 집중 베팅 + 해당 10년 운에 가깝다.

---

## 4. 개선 레버 우선순위

복리에 영향을 주는 순서. **위에서 통과하기 전 아래 레버를 운영에 넣지 않는다.**

```
1. Economic exposure (무엇을 사나)          ← S1 잠금 완료
2. 비용·OOS 견고성 (신뢰 가능한가)         ← Wave B (미운영 확인)
3. 리스크 완충 (적립 중단 방지)            ← Overlay walk-forward (다음 구현)
4. 적립 스케줄 (얼마나·언제 넣나)          ← 유보 원장 설계 후
5. 구현체 선택 (VTI vs ITOT)               ← 매핑 레이어, 후순위
6. 매도·익절                               ← overlay 실패 후 challenger
```

### 4.1 지금 하지 않는 것

| 아이디어 | 이유 |
| --- | --- |
| QQQ·나스닥 정책 편입 | `PolicyId` 금지; 단일 구간 과적합 |
| 월 적립액 가변 (폭락 시 추가 매수) | I5 위반 unless 명시적 유보 원장 + 동일 NPV 게이트 |
| 자유 비중 최적화 / 다 ETF 그리드 | Data snooping; 가설당 ablation만 허용 |
| 익절 매도 | Buy-only baseline 깨짐; overlay가 1차 대안 |

---

## 5. 다음 연구 파도 (구현 순서)

### Wave F — S1 비용 견고성 (구현 없음, CLI만)

**목적:** Ideal→Stress 비용에서도 S1 잠금이 유지되는지 확인.

```bash
uv run python -m src.etf_manager.cli ingest history --start 2012-06-01 --end 2024-10-31
uv run python -m src.etf_manager.cli run walk-forward-costs \
  --config configs/experiments/wf_s0_s1.json
```

실패 시: overlay 이전에 비용·환율 가정을 재검토.

**운영 참고:** 실험 JSON 기본 `2012-04-01` / `2024-11-30`은 CPI PIT 지연·마지막 거래일 가격 부재로 실패할 수 있음. 운영 시 `2012-06-01` / `2024-10-31` 사용.

### Wave G — S1 + Bounded Overlay walk-forward (구현 필요)

**목적:** 동일 월 적립 조건에서 리스크 완충이 CE를 올리는지 검증. 수익 극대화가 아니라 **적립 지속성·효용** 검증.

**가설:** `S1 + OverlayConfig(max_shift ≤ 0.10)` 이 순수 S1 대비 CE gate 통과.

**구현 범위 (예상):**
- `ExperimentSpec`에 overlay 필드 추가
- `validation/ablation.py`, `validation/campaign.py`의 `_arm_config`에서 overlay 전달
- `configs/experiments/wf_s1_overlay.json` 추가
- 실패 시 overlay 제거, S1 유지 (성공적인 연구 결과)

**비목표:** 매도, 레버리지, 월 납입액 변경.

### Wave H — 적립 스케줄 (장기, 별도 spec)

명시적 **유보 원장** 위에서만 “더 사고 덜 사기” 허용. 가상 현금 생성 금지(I5, I6).

### Wave I — Live execution (최종)

Paper broker(P10) 안정 후 실거래 연결. 연구 정책 변경 없이 실행 레이어만 확장.

---

## 6. 운영자 워크플로

### 데이터 적재

```bash
uv run python -m src.etf_manager.cli ingest history \
  --start 2012-06-01 --end 2024-10-31
```

기본 티커: `history_price_tickers()` = policy 슬리브 ∪ `QQQ`(진단 전용).

### 현재 정책 단일 경로 실행

```bash
uv run python -m src.etf_manager.cli run policy \
  --id s1_us --start 2012-06-01 --end 2024-10-31 \
  --contribution-krw 1000000
```

### 인기 ETF vs S1 진단 (채택 없음)

```bash
uv run python -m src.etf_manager.cli run diagnose-us-vehicles \
  --start 2012-06-01 --end 2024-10-31 --contribution-krw 1000000
```

### Paper 실행

```bash
uv run python -m src.etf_manager.cli run paper \
  --id s1_us --start 2024-01-01 --end 2024-10-31 \
  --contribution-krw 1000000
```

---

## 7. 성공 기준

프로젝트 성공은 **가장 높은 과거 수익률 발견**이 아니다.

> PIT 데이터로 재현 가능하고, 동일 cashflow·비용 조건에서 단순 baseline과 공정 비교되며, OOS·cohort·비용 시나리오에서도 견고한 **장기 적립 정책**을 확정하는 것.

최종 정책이 단순 VTI DCA와 거의 같다면 그 결과도 성공이다.

---

## 8. 아키텍처 참조

| 문서 | 내용 |
| --- | --- |
| `docs/architecture/00_system_overview.md` | 레이어·목적함수·불변식 |
| `docs/architecture/01_data_contracts.md` | 데이터 공급·PIT·품질 게이트 |
| `docs/architecture/02_policy_and_validation.md` | PolicyId·실험 매트릭스·검증 파이프라인 |
| `docs/architecture/03_operator_cli.md` | CLI·설정 파일·산출물 |
