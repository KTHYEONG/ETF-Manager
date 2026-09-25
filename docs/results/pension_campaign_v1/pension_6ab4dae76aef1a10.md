# Pension campaign pension_campaign_v1

experiment_id: 6ab4dae76aef1a10
market_mode: us_proxy
market_coverage: 1998-01-02..2026-07-06
real_data_status: AVAILABLE
evidence_status: INSUFFICIENT_INDEPENDENT_20Y_EVIDENCE
note: Current live Korean ETF history is too short for an observed 20-year test.
note: SOXX observations before 2021-06-21 carry the disclosed index-break label.
note: Ratios value each account as if closed at the cohort end with the non-pension tax (credit clawed back); pension-receipt values are a bracket that assumes deferral to eligibility and no further return; see years_to_draw_at_threshold.
fx_fallback: status=APPLIED sessions=244 share=0.0451
note: USD/KRW on Korean-holiday sessions comes from FRED DEXKOUS (NY-noon buying rate) instead of the ECOS base rate; see fx_provenance for count and same-day basis.
note: Household view compares the same available cash: pension (credit-optimal contributions, valued at exit) plus a general account holding leftover cash and credit refunds, versus all cash in a general account holding the same US-listed ETFs under Korean overseas-equity tax with year-end gain harvesting.

## Household same-cash view

| arm | horizon | profile | cohorts | excluded | acct adv (lump) | worst | acct adv (annuity low) | acct adv (annuity high) | asset effect household | asset effect general |
|---|---|---|---|---|---|---|---|---|---|---|
| nasdaq_100 | 120 | mid_partial | 12 | 0 | 1.0499 | 1.0384 | 1.1114 | 1.0499 | 1.2927 | 1.2853 |
| nasdaq_100 | 120 | retiree_full | 12 | 0 | 1.0862 | 1.0599 | 1.2092 | 1.0862 | 1.3116 | 1.2853 |
| nasdaq_100 | 120 | young_zero | 12 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.2853 | 1.2853 |
| nasdaq_100 | 240 | mid_partial | 2 | 0 | 1.0892 | 1.0881 | 1.1573 | 1.0892 | 1.8310 | 1.8088 |
| nasdaq_100 | 240 | retiree_full | 2 | 0 | 1.1774 | 1.1754 | 1.3137 | 1.1774 | 1.8541 | 1.8088 |
| nasdaq_100 | 240 | young_zero | 2 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.8088 | 1.8088 |
| qqq80_soxx20 | 120 | mid_partial | 12 | 0 | 1.0460 | 1.0379 | 1.1076 | 1.0460 | 1.3370 | 1.3277 |
| qqq80_soxx20 | 120 | retiree_full | 12 | 0 | 1.0799 | 1.0607 | 1.2031 | 1.0799 | 1.3604 | 1.3277 |
| qqq80_soxx20 | 120 | young_zero | 12 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.3277 | 1.3277 |
| qqq80_soxx20 | 240 | mid_partial | 2 | 0 | 1.0854 | 1.0825 | 1.1532 | 1.0854 | 1.9451 | 1.9284 |
| qqq80_soxx20 | 240 | retiree_full | 2 | 0 | 1.1717 | 1.1666 | 1.3073 | 1.1717 | 1.9670 | 1.9284 |
| qqq80_soxx20 | 240 | young_zero | 2 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.9284 | 1.9284 |
| qqq90_soxx10 | 120 | mid_partial | 12 | 0 | 1.0475 | 1.0375 | 1.1088 | 1.0475 | 1.3118 | 1.3026 |
| qqq90_soxx10 | 120 | retiree_full | 12 | 0 | 1.0824 | 1.0603 | 1.2049 | 1.0824 | 1.3339 | 1.3026 |
| qqq90_soxx10 | 120 | young_zero | 12 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.3026 | 1.3026 |
| qqq90_soxx10 | 240 | mid_partial | 2 | 0 | 1.0872 | 1.0852 | 1.1551 | 1.0872 | 1.8878 | 1.8684 |
| qqq90_soxx10 | 240 | retiree_full | 2 | 0 | 1.1743 | 1.1705 | 1.3102 | 1.1743 | 1.9100 | 1.8684 |
| qqq90_soxx10 | 240 | young_zero | 2 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.8684 | 1.8684 |
| sp500_100 | 120 | mid_partial | 12 | 0 | 1.0325 | 1.0267 | 1.0915 | 1.0325 | 1.0000 | 1.0000 |
| sp500_100 | 120 | retiree_full | 12 | 0 | 1.0513 | 1.0242 | 1.1693 | 1.0513 | 1.0000 | 1.0000 |
| sp500_100 | 120 | young_zero | 12 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| sp500_100 | 240 | mid_partial | 2 | 0 | 1.0760 | 1.0753 | 1.1416 | 1.0760 | 1.0000 | 1.0000 |
| sp500_100 | 240 | retiree_full | 2 | 0 | 1.1487 | 1.1461 | 1.2798 | 1.1487 | 1.0000 | 1.0000 |
| sp500_100 | 240 | young_zero | 2 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
note: Cohorts whose baseline after-tax wealth is zero (no contribution was made, for example a profile with no usable tax credit) have no defined paired ratio and are excluded from ratio, rate, and drawdown statistics; see the undefined column.
fully_undefined_profiles: young_zero

| arm | horizon | cohorts | undefined | independent | median | worst | annuity low | annuity high | median XIRR | drawdown |
|---|---|---|---|---|---|---|---|---|---|---|
| sp500_100 | 120 | 24 | 12 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.12227598103123197 | -0.2943 |
| nasdaq_100 | 120 | 24 | 12 | 2 | 1.3015 | 1.1881 | 1.3039 | 1.3015 | 0.16967063693459017 | -0.2966 |
| qqq90_soxx10 | 120 | 24 | 12 | 2 | 1.3173 | 1.2377 | 1.3202 | 1.3173 | 0.17309470562230292 | -0.3075 |
| qqq80_soxx20 | 120 | 24 | 12 | 2 | 1.3420 | 1.2183 | 1.3452 | 1.3420 | 0.17644389670361527 | -0.3164 |
| sp500_100 | 240 | 4 | 2 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.14322714663722272 | -0.3008 |
| nasdaq_100 | 240 | 4 | 2 | 1 | 1.8513 | 1.8500 | 1.8545 | 1.8513 | 0.19197774055495137 | -0.3066 |
| qqq90_soxx10 | 240 | 4 | 2 | 1 | 1.9061 | 1.8842 | 1.9095 | 1.9061 | 0.19424692574653188 | -0.3113 |
| qqq80_soxx20 | 240 | 4 | 2 | 1 | 1.9609 | 1.9185 | 1.9646 | 1.9609 | 0.19644136989594746 | -0.3210 |
