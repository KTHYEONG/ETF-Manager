# ISA 중개형 결과: 계속 보유(`hold`)

## 결론

ISA 중개형은 연 600만·1,200만·2,000만 원 세 예산 모두에서 계속 보유(`hold`)를 채택한다.

## 핵심 표

예산별 최저 칸 점수(기준 `hold` = 1.000)다. 소득 경로 4종(서민형·일반형) 모두 결론이 같다.

| 예산 | hold | 3년마다 전액 연금 이전 | ISA 미사용 | 판정 |
| :--- | :---: | :---: | :---: | :--- |
| 연 600만 원 | 1.000 | 0.970 | 0.935 | hold |
| 연 1,200만 원 | 1.000 | 0.977 | 0.934 | hold |
| 연 2,000만 원 | 1.000 | 0.990 | 0.944 | hold |

표 수치는 동결 기록의 예산별 강건 점수와 일치한다(0.970364, 0.976861, 0.990468 등).

## 근거

| 항목 | 값 |
| :--- | :--- |
| record_id | `isa_household_v1__94cad1c1f1d4b913` |
| record sha256 | `79fba55b74ac728b3e600e1f7de40dba98677b9758c5d113980ece81df7113d0` |
| record 위치 | `data/frozen/isa/isa_household_v1__94cad1c1f1d4b913.json` |
| config_sha256 | `94cad1c1f1d4b913c44b46ff7280d3322c0261a209f95ba5f3e506fad6621f6c` |
| config 위치 | `configs/decision/isa.json` |
| manifest_hashes | prices `dafb32d30ebdda8da50da5599cb554eaf77b99f5499b56911969f86175942a2d`, research_monthly `8a47a46d6fb40124efd43733ef2008e3b415d8f8e726b5e28bfe85e7a60c9204`(run JSON) |
| seed | 42(run JSON) |
| git commit | `68f73a769206765e37dea96bb202f514d457ba98` |
| 누적 시험 | 55(동결 기록) |

## 재현

```bash
uv run python -m src.cli run isa-household --config configs/decision/isa.json --seed 42
```

## 한계

- **근사 모델:** 월 단위·환율 중립 근사이며 법정 금액은 명목값이다.
- **미모델링:** ISA 3년 미만 중도해지, 2027년 세제개편 정부안(국회 계류).
- **법 전제:** 계속 보유는 계약 기간 상한이 없는 현행법을 전제로 하므로 국회 표결 후 재확인한다.
