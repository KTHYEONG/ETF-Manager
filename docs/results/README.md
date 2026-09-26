# Research Results

| 계좌 | 채택안 | 결과 문서 | 근거 위치 (`data/frozen` 또는 `data/runs`) |
| :--- | :--- | :--- | :--- |
| 일반계좌 | QQQ 90% + SOXX 10% | [`general.md`](general.md) | `data/runs/final_historical_campaign_v1/` |
| 연금저축 | 나스닥100 90% + 배당 10% | [`pension.md`](pension.md) | `data/frozen/pension/` |
| ISA 중개형 | 계속 보유(`hold`) | [`isa.md`](isa.md) | `data/frozen/isa/` |

기계 출력물과 동결 기록은 git 무시 대상인 `data/` 아래에 살고 `tools/devops/backup.py`로 Drive에 미러링된다. 승격된 연구 근거는 `data/research/`에 둔다.
