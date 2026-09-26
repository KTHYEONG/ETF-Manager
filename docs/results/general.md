# 일반계좌 결과: QQQ 90% + SOXX 10% 매수 전용 적립

## 결론

일반계좌(월 100만 원 적립)는 QQQ 90% + SOXX 10% 매수 전용 적립을 채택한다.

## 핵심 표

기준(QQQ 100%) 대비 세후 실질 성과다.

| 조합 | 실질 연수익률 | 최악 기간 | 판정 |
| :--- | :---: | :---: | :--- |
| QQQ 90% + SOXX 10% (채택) | **24.7%** | **1.022** | 4개 기간 전부 우세 |
| QQQ 100% (기준) | 23.3% | 1.000 | — |
| QQQ 85% + SOXX 15% (기각) | — | — | 반도체 쏠림으로 기각 |

최악 기간 1.022는 실행 아티팩트의 c2 코호트 worst 1.0224와 일치한다.

## 근거

| 항목 | 값 |
| :--- | :--- |
| run id | 미기록(legacy 아티팩트) |
| run JSON sha256 | `9aa5c704c2d0e58566d0d8e752cd48022aa5ccdbb6605372faa122b7168ef7cb` |
| run JSON 위치 | `data/runs/final_historical_campaign_v1/legacy_1c7a8194f7ce5a59.json` |
| config_sha256 | `20319dc0b5a730e7af04e745c7764fdf4c8b4daaab16d5663be927e50db2eebc` |
| config 위치 | `configs/decision/general.json` |
| manifest_hashes | 미기록(캠페인 아티팩트에 없음) |
| seed | 미기록(캠페인 아티팩트에 없음) |
| git commit | `8201d9e`(md sidecar 단축 해시) |
| 누적 시험 | 39(아티팩트 내장 lineage census) |

## 재현

```bash
uv run python -m src.cli run final-historical-campaign --config configs/decision/general.json --seed 42
```

## 한계

- **표본 크기:** 120개월 코호트 4개는 서로 겹친다(CPI 가용 기간 2012-08~).
- **극단 국면:** 닷컴·GFC 구간은 실제 ETF 상장 이전이라 검증에서 제외된다.
- **독립성:** 세 결정은 같은 과거 데이터를 공유하므로 독립된 발견이 아니다.
