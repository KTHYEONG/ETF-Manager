# FINAL_HISTORICAL_CAMPAIGN_V1

- **Window:** 2012-08-31 → 2026-08-28
- **Git commit:** `8201d9e`
- **Operational unlock:** False
- **Thin-sample warning:** cohort_count=4 (<10 target); CPI floor limits 120M rolling cohorts

## Arm summary (120M, step 12M)

| arm | cohorts | median | p10 | worst | bootstrap p05 | win_rate | CE γ10 |
|-----|---------|--------|-----|-------|---------------|----------|--------|
| b0_qqq100 | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.00 | 1.0000 |
| c1_qqq95_soxx5 | 4 | 1.0110 | 1.0069 | 1.0058 | 1.0047 | 1.00 | 1.0112 |
| c2_qqq90_soxx10 | 4 | 1.0336 | 1.0256 | 1.0224 | 1.0123 | 1.00 | 1.0335 |
| c3_qqq85_soxx15 | 4 | 1.0556 | 1.0416 | 1.0368 | 1.0120 | 1.00 | 1.0546 |

## Regime coverage

| regime | tier | overlap_months |
|--------|------|----------------|
| dot_com | none | 0 |
| gfc | none | 0 |
| low_rate_2010s | substantial | 89 |
| covid | full | 3 |
| inflation_2022 | full | 12 |
| ai_boom_2023 | substantial | 32 |

## Pre-history mix proxy (dot_com / gfc)

- **dot_com:** unavailable — Parameter `start` receieved as '1998-03-01 00:00:00' although cannot be earlier than the first session of calendar 'XNYS' ('2006-08-31 00:00:00').
- **gfc:** unavailable — missing research_returns series: ['NDX100', 'SOX']

## Tax sensitivity

- not_modelled: buy_only_accumulation_defers_realization_tax_until_sale; no PIT tax ledger model

## Incumbent read (C2 QQQ90/SOXX10)

Median candidate/baseline ratio **1.0336**; bootstrap p05 **1.0123**; all 4 cohorts beat baseline (win_rate=1).
