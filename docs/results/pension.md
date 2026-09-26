# 연금저축 결과: 나스닥100 90% + 배당 10% 계속 보유

## 결론

연금저축(연 600만 원)은 나스닥100 90% + 배당 10%(`schd10_qqq90`) 계속 보유를 채택한다.

## 핵심 표

기준 조합(나스닥100 80% + S&P500 20%) 대비 구간별 비율이다.

| 조합 | 100년 | 현대 | 실제 ETF | 판정 |
| :--- | :---: | :---: | :---: | :--- |
| schd10_qqq90 (채택) | **1.039** | **1.057** | **1.024** | 세 구간 모두 0.995 이상 |
| schd20_qqq80 (기각) | 1.069 | 1.003 | 0.992 | 실제 ETF 구간 미달 |
| spy0_qqq100 (기각) | 0.995 | 1.113 | 1.056 | 100년 구간 미달 |

표 수치는 동결 근거 run 아티팩트의 도미넌스 표와 일치한다(1.038760, 1.057112, 1.024226 등).

## 근거

| 항목 | 값 |
| :--- | :--- |
| record_id | `pension_decision_v3__2026-08-31__1175d4ef48b20a9a` |
| record sha256 | `9d99771e2285e9018f8f1e60fefbf640e339c324ade2467a1502b6123a28f852` |
| record 위치 | `data/frozen/pension/pension_decision_v3__2026-08-31__1175d4ef48b20a9a.json` |
| config_sha256 | `1175d4ef48b20a9afef2240c9c50cdc77e91d604b541c11e4c022ceeeb993992` |
| config 위치 | `configs/decision/pension.json` |
| manifest_hashes | prices `dafb32d30ebdda8da50da5599cb554eaf77b99f5499b56911969f86175942a2d`, research_monthly `8a47a46d6fb40124efd43733ef2008e3b415d8f8e726b5e28bfe85e7a60c9204` |
| seed | 미기록(동결 기록에 없음) |
| git commit | `2f0ce92ebaf7b0406ed9c396f07017f1624093ed` |
| 누적 시험 | 51(동결 근거 run 아티팩트) |

## 재현

```bash
uv run python -m src.cli run pension-decision --config configs/decision/pension.json --seed 42
```

## 한계

- **실제 ETF 구간:** 10년 기간이 5개뿐이라 참고용이다.
- **근사 모델:** 월 단위 결정이며 법정 금액은 물가에 연동하지 않은 명목값이다.
- **독립성:** 세 결정은 같은 과거 데이터를 공유하므로 독립된 발견이 아니다.
