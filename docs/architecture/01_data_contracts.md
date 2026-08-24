# Data Layer Contracts

## 1. Data Feasibility Matrix

Verified against provider documentation (2026-08).

| Data | Source / Endpoint | History | PIT capable | Free | Use | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| ETF/equity EOD OHLCV, dividend, split | Tiingo `GET /tiingo/daily/{t}/prices` (CRSP-style adjustment) | up to 50y, per-ticker bounded by inception | Partial — value final after session close, vendor corrections same evening | 50 req/h, 1,000 req/d, 500 symbols/mo; personal use | L6 ETF implementation backtest | Free tier is personal-use; corrections require re-pull + manifest re-hash; delisted-ETF coverage unconfirmed |
| Price cross-check | yfinance / Stooq | varies | No | Yes | sanity check only | Silent retroactive edits; forbidden as source of truth |
| Factors 5F + Mom (US, Developed, Developed ex-US, EM, regions) | Kenneth French Data Library CSV/ZIP | US 5F 1963-07~, Developed & Dev ex-US 1990-07~, EM 1990-01~ (monthly) | **No** — publication lag and retroactive revision, no vintage archive | Yes | L2/L5 exposure research **only** | Revision + lag ⇒ prohibited as a live signal input (I1, I9) |
| Macro, rates, VIX, credit spread | FRED/ALFRED `series/observations?realtime_start&realtime_end` | series dependent (VIX 1990~, DGS10 1962~) | **Yes** — true vintages via real-time period | Yes, 120 req/min, 6,000 req/day, commercial use permitted | overlay signals, reporting deflator | Daily cap ⇒ local cache mandatory; series ids differ by seasonal adjustment |
| USD/KRW | FRED `DEXKOUS` (1981~) / ECOS `731Y001` (daily) | 1981~ | Yes (unrevised) | Yes; ECOS 10,000 calls/month | FX conversion, valuation | Reference rate ≠ executable rate ⇒ explicit spread model required |
| Korea CPI | ECOS `901Y009` (monthly) | 1965~ | **No** — base-year rebasing retroactively restates the index | Yes | ex-post real KRW deflator | Reporting only; prohibited as signal input |
| ETF metadata: expense, AUM, holdings | SEC `N-CEN` (annual), `N-PORT` bulk datasets (2019Q4~) | 2019Q4~ for N-PORT | Yes, keyed by filing date | Yes | L6 ETF score from 2019 onward | No PIT metadata before 2019Q4 |
| ETF delisting / liquidation events | SEC `Form 25` (`effectivenessDate`), `497` supplements, N-PORT cessation | ~2000s~ | Yes | Yes | survivorship correction | Ticker↔CIK↔series reconciliation is non-trivial; no single authoritative file |
| Survivorship-free PIT ETF universe | CRSP/WRDS, commercial security master | — | Yes | **No** | M10–M11 only | Not obtainable ⇒ M11 downgraded to documented research limitation |
| Historical bid-ask spread | none free (IEX realtime only) | — | No | — | cost model | Unobtainable ⇒ parametric spread scenarios keyed on AUM/volume |
| Expense ratio history | N-CEN 2019~, prospectus `497` | 2019~ | Yes | Yes | cost model | Pre-2019 must use scenario constants, never today's value (I10) |

## 2. Availability Semantics

Every dataset declares how `available_at` is derived. No dataset may omit this declaration.

| Rule | Definition | Applies to |
| --- | --- | --- |
| `SESSION_CLOSE` | `available_at = calendar.close_ts(observation_date)` | prices, FX, VIX, daily rates |
| `RELEASE_COLUMN` | `available_at = release_date` supplied by the provider; multiple vintages per `observation_date` permitted | FRED/ALFRED series, SEC filings |
| `FIXED_LAG` | `available_at = period_end + lag`, `lag` a declared constant | Ken French factors, Korea CPI |

As-of resolution for a decision timestamp $t$:

$$
\text{as\_of}(t) = \underset{\text{observation\_key}}{\arg\max_{\;\text{available\_at}}}\;\{\text{rows}: \text{available\_at} \le t\}
$$

For non-revisable datasets this reduces to a filter; for revisable datasets it selects the latest
vintage per observation key.

## 3. Missing-Data Policy Registry

| Policy | Behavior | Assigned to |
| --- | --- | --- |
| `FAIL` | any gap versus the expected calendar raises `DataQualityError` | prices, FX |
| `DROP` | rows with nulls in required columns are removed and counted in the report | optional metadata columns |
| `EXPLICIT_GAP` | gap is preserved as a null row with `available_at = null`; downstream must handle it | low-frequency macro |

Global forward-fill is prohibited system-wide. Any carry-forward must be expressed as an
availability rule (`FIXED_LAG`), never as value imputation.

## 4. Quality Gate Checks

| Check | Predicate | Severity |
| --- | --- | --- |
| `schema_conformance` | column set and dtypes equal `DatasetSpec.columns` | ERROR |
| `key_uniqueness` | `DatasetSpec.key` has no duplicate rows | ERROR |
| `required_not_null` | required columns contain no nulls | ERROR |
| `timestamp_tz_aware` | all timestamps are `Datetime("us", "UTC")` | ERROR |
| `availability_ordering` | `available_at >= observation timestamp` for every row | ERROR |
| `ohlc_consistency` | `low <= min(open, close)`, `high >= max(open, close)`, `low <= high`, all `> 0` | ERROR |
| `session_completeness` | observed dates equal calendar sessions in range (policy `FAIL`) | ERROR |
| `monotonic_observations` | observation keys strictly increasing per group | ERROR |
| `return_outlier` | `abs(log(close_t / close_{t-1})) > z` flagged, never modified | WARN |
| `cross_provider_deviation` | secondary-source close deviates beyond tolerance | WARN |

`enforce(report)` raises on any `ERROR`. The gate never mutates input data.

## 5. Storage Layout

```text
data/
├── raw/<provider>/<dataset>/<retrieved_date>/payload.{json,csv,zip}
├── normalized/<dataset>/schema_version=<v>/part-*.parquet
├── features/<feature_set>/<config_hash>/part-*.parquet
└── manifests/<dataset>/<retrieved_date>.json
```

`raw/` is append-only and never rewritten. `normalized/` is regenerated deterministically from
`raw/` plus a normalization version. DuckDB is a read-only query view over `normalized/` and
`features/`; it never holds authoritative state.

## 6. Manifest Record

| Field | Type | Purpose |
| --- | --- | --- |
| `dataset` | str | dataset identity |
| `provider` / `endpoint` | str | lineage |
| `request_params` | mapping | exact reproduction of the call |
| `retrieved_at` | datetime UTC | download instant |
| `row_count` | int | volume check |
| `content_sha256` | str | canonical-order content hash |
| `schema_version` | str | dataset schema revision |
| `normalization_version` | str | transform revision |
| `quality_findings` | list | serialized gate result |

A normalized partition without a matching manifest is treated as untrusted and rejected on read.

## 7. Ingest Universe

| Function | Tickers | Use |
| --- | --- | --- |
| `all_policy_tickers()` | BND, IEF, IVV, TLT, VEA, VT, VTI, VTV, VWO | Policy sleeves + ablation/WF |
| `diagnostic_price_tickers()` | QQQ | Reporting only (`run diagnose-us-vehicles`) |
| `history_price_tickers()` | Union of the above | Default `ingest history` panel |

QQQ is ingested for factor/DCA diagnostics but has **no `PolicyId`**. Partial ticker ingest
(e.g. QQQ alone) replaces the latest prices partition — always run full `ingest history` before
campaigns.

## 8. Operator Date Constraints

ECOS CPI uses `FIXED_LAG` availability (roughly 6–8 weeks after `period_end`). Combined with
NYSE fill-delay semantics, the following windows are known-good with the current catalog:

| Boundary | Issue if violated |
| --- | --- |
| `start < 2012-06-01` | `missing positive CPI row` at early execution sessions |
| `end > 2024-10-31` (with current Tiingo pull) | `missing price row` on last execution session after signal month-end |

Experiment JSON under `configs/experiments/` uses `2014-01-03` / `2024-09-30` (catalog-feasible).
For longer panels use ingest `2012-08-31` / `2024-09-30`; avoid `2024-11-30` (missing last-session marks).
