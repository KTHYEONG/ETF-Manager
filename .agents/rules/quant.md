---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(etf|portfolio|allocation|rebalance|execution|data|backtest).*"
  - on_file_path_glob: ["src/**/etf/**/*.py", "src/**/portfolio/**/*.py", "src/**/allocation/**/*.py", "src/**/rebalance/**/*.py", "src/**/data/**/*.py"]
priority: 10
---

# Quant & ETF Engineering Principles

> **Never leak future information, preserve the reality of capital flows and execution viability, guard against validation leakage and overfitting, and prioritize economic correctness over specific implementation mechanics.**

## 1. Temporal Integrity & Distribution Timestamps (PIT & Leakage)
- **Information Availability & Timestamps:** Define explicit semantics for `observation_time` (NAV publication, market close), `decision_time` (rebalance signal), and `execution_time` (order fill at open/close).
- **Multi-Currency & FX Alignment:** Align cross-border ETF prices, local valuation, and FX benchmark rates using strict release timestamps without look-ahead bias.
- **Distribution & Split Integrity:** Reflect dividend ex-dates, distribution payments, and corporate actions strictly at their actual point-in-time publication.

## 2. ETF Microstructure & Portfolio Accounting
- **NAV Disparity & Tracking:** Monitor iNAV vs. market price disparity `(market_price - nav) / nav`. Guard against executing during illiquid opening/closing dislocations. Track Tracking Error ($TE$) and Tracking Difference ($TD$).
- **Expense Ratios & Real Drag:** Deduct Total Expense Ratio (TER, 운용보수/기타비용) daily from NAV, and model local execution fees, exchange charges, and spreads.
- **Cash Drag & Accumulation Realism:** Model Dollar-Cost Averaging (DCA), Value Averaging, and lot size constraints with realistic cash drag and cash buffers.
- **Portfolio Weight Invariants:** Enforce structural weight invariant $\sum w_i + w_{cash} = 1.0 \pm 10^{-6}$. Employ tolerance bands to avoid unnecessary turnover friction.
- **Return Accounting:** Explicitly distinguish between Price Return (PR) and Total Return (TR, dividend reinvestment).

## 3. Numerical Integrity & Economic Correctness
- **Numerical Edge Cases:** Handle zero-division, NaNs, and infinities based on true market semantics (e.g., zero trading volume, halted ETF, missing iNAV) rather than arbitrary normal substitutions.
- **Metric Significance vs. Overfitting:** Avoid tuning allocation parameters to past samples; evaluate robust economic viability across varying macro and interest rate regimes.
- **Principles Over Mechanics:** Prioritize correct financial meaning and structural invariants over dogmatic adherence to specific library functions.
