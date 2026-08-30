# Wave D Exit ai_compute 2026-08-28

As of: 2026-08-28T20:00:00+00:00
panel_as_of: 2026-08-28T20:00:00+00:00
freshness_status: FRESH
portfolio_status: historically_promising

## Evidence Slots

| slot | status | summary | metrics |
| --- | --- | --- | --- |
| historical | computed | 120M cohorts n=3 median 1.4116 | median_ratio=1.4115671089129471, cohort_count=3, p10_ratio=1.3775315236234689, worst_ratio=1.3690226273010992, win_rate=1.0, bootstrap_p05_ratio_mean=1.3832041211717152 |
| structural | computed | fundamental: PNFI yoy 9.88% regime expansion falsifier False change none | primary_series_id=PNFI, primary_yoy_pct=9.8833705049445, falsifier_capex_structural_slowdown_active=False, change_point_date=, regime=expansion |
| valuation | computed | valuation: rich richness 94.0% ratio 0.7099 return 103.30% falsifier False | vehicle_ticker=SOXX, benchmark_ticker=QQQ, relative_ratio=0.7099367698170094, richness_percentile=94.04761904761905, richness_label=rich, trailing_return_pct=103.30073774435262, falsifier_semiconductor_pricing_collapse_active=False |
| overlap | computed | overlap 10.6% shared 12 | overlap_pct=10.625638388196, shared_holdings_count=12, a_only_weight_pct=89.37436161180399, b_only_weight_pct=89.335968309791 |
| crowding | computed | crowding: dispersed hhi 0.0473 top5 35.6% n 32 | vehicle_ticker=SOXX, hhi=0.04725778912434084, top5_weight_pct=35.5820464240689, effective_n=21.160532867267268, holdings_count=32, concentration_label=dispersed, top_n=5 |

## Track H Arms

| arm_id | soxx_weight | median_ratio | p10_ratio | worst_ratio | cohort_count | win_rate | p05 | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qqq95_soxx5 | 0.05 | 1.0070 | 1.0030 | 0.9970 | 10 | 0.7150 | 0.9903 | True |
| qqq90_soxx10 | 0.10 | 1.0199 | 1.0070 | 0.9938 | 10 | 0.8025 | 0.9772 | True |
| qqq85_soxx15 | 0.15 | 1.0320 | 1.0108 | 0.9905 | 10 | 0.8175 | 0.9625 | True |

## Exit Checklist

- track_f_complete: True
- reference_slice_ready: True
- operational_challenger_ready: True
- portfolio_status: historically_promising
- freshness_status: FRESH

## Blockers

- none

reference_slice_ready: True
operational_challenger_ready: True
