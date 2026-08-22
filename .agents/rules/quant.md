---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(etf|portfolio|allocation|rebalance|execution|data|backtest).*"
  - on_file_path_glob: ["src/**/etf/**/*.py", "src/**/portfolio/**/*.py", "src/**/allocation/**/*.py", "src/**/rebalance/**/*.py", "src/**/data/**/*.py"]
priority: 10
---

# Quant & ETF Engineering Principles

This document provides quantitative and financial directives for building robust ETF accumulation, portfolio allocation, and rebalancing systems (global/domestic universal).

## 0. Priority Hierarchy
1. **Financial Realism Over Pure Metrics:** Model tracking errors, disparity, expense ratios, and cash drag realistically over idealized backtests.
2. **Data Integrity & Temporal Consistency:** Enforce strict point-in-time data availability (NAV, market price, distributions, FX).
3. **Deterministic Numerical Stability:** Safeguard against zero-division, precision loss, and portfolio weight floating-point drifts.

## 1. ETF Microstructure & Valuation Metrics
- **NAV & Disparity (괴리율) Accounting:**
  - Track both Market Price and Net Asset Value (NAV / iNAV).
  - Calculate Disparity Ratio: `disparity = (market_price - nav) / nav`.
  - Guard against abnormal disparity spikes (e.g. illiquid sessions, market opening/closing auction dislocations) before executing rebalance or accumulation orders.
- **Total Expense Ratio (TER) & Real Cost Drag:**
  - Account for annual expense ratios (운용보수, 기타비용) deducted daily from NAV.
  - Model broker commissions, exchange fees, and bid-ask spreads per execution venue.
- **Tracking Error & Index Replication:**
  - Measure Tracking Difference ($TD = R_{ETF} - R_{Index}$) and Tracking Error ($TE = \text{Std}(TD)$) when evaluating ETF suitability.

## 2. Accumulation (적립식) & Portfolio Rebalancing
- **Systematic Accumulation Strategies:**
  - Support Dollar-Cost Averaging (DCA), Dynamic Accumulation (Value Averaging, Volatility/Drawdown-adjusted sizing).
  - Enforce minimum trade unit (lot size / fractional shares) and cash buffer constraints.
- **Portfolio Weight Normalization & Invariants:**
  - Enforce invariant: $\sum w_i + w_{cash} = 1.0 \pm 10^{-6}$.
  - Rebalancing thresholds: Use tolerance bands (e.g., target weight $\pm 2\%$) or periodic calendar schedules to minimize excessive turnover and transaction friction.
- **Distribution (Dividend) Handling & Total Return (TR):**
  - Explicitly distinguish between Price Return (PR) and Total Return (TR).
  - Support automated dividend reinvestment (DRIP) modeling and cash buffer accumulation.

## 3. Data Integrity & Temporal Alignment
- **Explicit Timestamp Semantics:** Define `observation_time` (NAV release, market close), `decision_time` (signal calculation, rebalance decision), and `execution_time` (order fill / next market open/close).
- **Multi-Currency & FX Alignment:**
  - Explicitly track currency denominations (USD, KRW, EUR, etc.) and FX conversion timestamps for international ETF holdings.
  - Avoid look-ahead bias when aligning foreign ETF NAV/prices with local currency valuation.
- **Point-in-Time Data Availability:** Ensure split adjustments, dividend ex-dates, and NAV updates are reflected only after their actual publication timestamp.

## 4. Safe Numerical Computation
- **Safe Vectorized Division:** Use `np.divide` with explicit `out` initialization and `where` masks to avoid zero-division errors.
  ```python
  result = np.zeros_like(numerator, dtype=float)
  np.divide(numerator, denominator, out=result, where=denominator != 0)
  ```
- **Log-space Compounding:** Use `np.log1p()` and `np.expm1()` for compounding long-term returns and cumulative returns to avoid numerical underflow.
