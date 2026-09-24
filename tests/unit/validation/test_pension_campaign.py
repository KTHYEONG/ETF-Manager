"""Invariant guards for the pension cohort campaign."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.analytics.metrics import xirr
from src.data.calendar import load_calendar
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload
from src.sim.pension_engine import PensionDataError, PensionMarketMode
from src.validation.pension_campaign import (
    PensionCampaignReport,
    load_pension_campaign_spec,
    run_pension_campaign,
    write_pension_campaign_report,
)

_REPO = Path(__file__).resolve().parents[3]
_TAX_PATH = str(_REPO / "configs" / "tax" / "kr_pension_2026.json")
_IDENTITY_PATH = str(_REPO / "configs" / "data" / "pension_etfs_2026.json")
_RETRIEVED_AT = datetime(2024, 1, 1, 5, 0, tzinfo=UTC)


def _payload() -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="probe",
        request_params={},
        retrieved_at=_RETRIEVED_AT,
        extension="json",
        content=b"{}",
    )


def _persist_proxy_lake(settings: DataSettings, start: date, end: date) -> None:
    sessions = list(load_calendar("XNYS").sessions(start, end))
    drifts = {"SPY": (400.0, 0.10), "QQQ": (300.0, 0.16), "SOXX": (200.0, 0.22)}
    rows = []
    for ticker, (base, drift) in drifts.items():
        for index, day in enumerate(sessions):
            price = base + drift * index
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 10_000,
                    "adjusted_close": price,
                    "dividend": 0.0,
                    "split_factor": 1.0,
                    "source": "synthetic",
                    "retrieved_at": _RETRIEVED_AT,
                }
            )
    spec_columns = spec_for(Dataset.PRICES).columns
    prices = pl.DataFrame(
        rows,
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "adjusted_close": pl.Float64,
            "dividend": pl.Float64,
            "split_factor": pl.Float64,
            "source": pl.String,
            "retrieved_at": pl.Datetime("us", "UTC"),
        },
    ).select(list(spec_columns))
    persist_ingest(prices, Dataset.PRICES, _payload(), settings)
    fx = pl.DataFrame(
        {"date": sessions, "usdkrw": [1300.0] * len(sessions), "source": ["synthetic"] * len(sessions),
         "retrieved_at": [_RETRIEVED_AT] * len(sessions)},
        schema=dict(spec_for(Dataset.FX_KRW_BASE).columns),
    )
    persist_ingest(fx, Dataset.FX_KRW_BASE, _payload(), settings)


def _persist_live_lake(settings: DataSettings, start: date, end: date) -> None:
    sessions = list(load_calendar("XKRX").sessions(start, end))
    rows = []
    for ticker in ("379800", "379810", "469060"):
        for index, day in enumerate(sessions):
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "close_krw": 10000.0 + 2.0 * index,
                    "nav_krw": 10000.0 + 2.0 * index,
                    "distribution_krw": 0.0,
                    "distribution_pay_date": None,
                    "split_factor": 1.0,
                    "volume": 1000,
                    "source": "synthetic",
                    "retrieved_at": _RETRIEVED_AT,
                }
            )
    frame = pl.DataFrame(
        rows,
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "close_krw": pl.Float64,
            "nav_krw": pl.Float64,
            "distribution_krw": pl.Float64,
            "distribution_pay_date": pl.Date,
            "split_factor": pl.Float64,
            "volume": pl.Int64,
            "source": pl.String,
            "retrieved_at": pl.Datetime("us", "UTC"),
        },
    )
    persist_ingest(frame, Dataset.KR_ETF_PRICES, _payload(), settings, calendar_name="XKRX")


def _persist_cpi_lake(settings: DataSettings) -> None:
    frame = pl.DataFrame(
        {
            "period_end": [date(2022, 10, 31), date(2024, 10, 31)],
            "value": [100.0, 110.0],
            "source": ["synthetic", "synthetic"],
            "retrieved_at": [_RETRIEVED_AT, _RETRIEVED_AT],
        },
        schema=dict(spec_for(Dataset.CPI).columns),
    )
    persist_ingest(frame, Dataset.CPI, _payload(), settings)


def _profile_entry(
    profile_id: str,
    birth: str,
    opened: str,
    years: list[int],
    income: int = 50_000_000,
    national: int = 5_000_000,
    local: int = 5_000_000,
    other: int = 0,
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "birth_date": birth,
        "account_open_date": opened,
        "pension_start_date": f"{max(years)}-12-01",
        "income_kind": "wage",
        "annual_income_krw": {str(y): income for y in years},
        "remaining_national_tax_krw": {str(y): national for y in years},
        "remaining_local_tax_krw": {str(y): local for y in years},
        "other_private_pension_income_krw": {str(y): other for y in years},
    }


def _campaign_config(
    *,
    name: str = "probe",
    start: str = "2023-01-01",
    end: str = "2024-12-31",
    mode: str = "us_proxy",
    horizons: list[int] | None = None,
    step: int = 12,
    arms: list[dict[str, Any]] | None = None,
    profiles: list[dict[str, Any]] | None = None,
    retirement: int = 2030,
    withdrawals: dict[str, int] | None = None,
    drag: dict[str, float] | None = None,
) -> dict[str, Any]:
    years = list(range(int(start[:4]), int(end[:4]) + 1))
    return {
        "name": name,
        "start": start,
        "end": end,
        "market_mode": mode,
        "horizons_months": horizons or [12],
        "step_months": step,
        "available_cash_events_krw": {f"{y}-01-03": 6_000_000 for y in years},
        "contribution_dates": {str(y): [f"{y}-01-15"] for y in years},
        "tax_credit_settlement_dates": {str(y): f"{y + 1}-05-31" for y in years},
        "retirement_start_year": retirement,
        "withdrawal_amounts_krw": withdrawals or {},
        "profiles": profiles or [_profile_entry("accum", "1985-01-01", "2015-01-01", years)],
        "tax_regime_path": _TAX_PATH,
        "etf_identity_path": _IDENTITY_PATH,
        "baseline_arm_id": "sp500",
        "arms": arms
        or [
            {"arm_id": "sp500", "role": "baseline", "targets": {"SPY": 1.0}},
            {"arm_id": "nasdaq", "role": "candidate", "targets": {"QQQ": 1.0}},
        ],
        "execution_spread_bps": 0.0,
        "commission_bps": 0.0,
        "max_fx_age_days": 7,
        "max_fx_fallback_share": 0.1,
        "max_cpi_age_days": 75,
        "extra_annual_drag_by_ticker": drag or {},
    }


def _write_config(tmp_path: Path, config: dict[str, Any]) -> str:
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataSettings:
    monkeypatch.chdir(tmp_path)
    return DataSettings(data_root="data")


def test_portfolio_ordering_disclosed_without_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Semiconductor-tilted cohorts report paired ratios and overlap with no adoption verdict."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    arms = [
        {"arm_id": "sp500", "role": "baseline", "targets": {"SPY": 1.0}},
        {"arm_id": "qqq90_soxx10", "role": "candidate", "targets": {"QQQ": 0.9, "SOXX": 0.1}},
    ]
    spec = load_pension_campaign_spec(
        _write_config(tmp_path, _campaign_config(arms=arms, step=6, drag={"SPY": 0.01, "QQQ": 0.01, "SOXX": 0.01}))
    )
    report = run_pension_campaign(spec, settings, seed=7)
    assert report.evidence_status == "INSUFFICIENT_INDEPENDENT_20Y_EVIDENCE"
    assert report.market_coverage_start == date(2023, 1, 3)
    assert report.market_coverage_end == date(2024, 12, 31)
    assert len(report.summaries) == 2
    for summary in report.summaries:
        assert summary.cohort_count > 0
        assert summary.median_wealth_ratio > 0
        assert summary.worst_wealth_ratio > 0
        assert summary.evidence_status == "INSUFFICIENT_INDEPENDENT_20Y_EVIDENCE"
        assert summary.median_cashflow_normalized_rate is not None
        assert 0 <= summary.underperforming_cohorts <= summary.cohort_count
    assert {row.historical_overlap_group for row in report.cohort_rows} == {"overlapping"}
    assert all(row.paired_wealth_ratio > 0 for row in report.cohort_rows)
    assert all(row.max_drawdown <= 0 for row in report.cohort_rows)


def test_tax_capacity_gates_claimed_credits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero, partial, and sufficient capacities claim no more credit than usable."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    years = [2023, 2024]
    profiles = [
        _profile_entry("zero", "1985-01-01", "2015-01-01", years, national=0, local=0),
        _profile_entry("partial", "1985-01-01", "2015-01-01", years, national=450_000, local=45_000),
        _profile_entry("full", "1985-01-01", "2015-01-01", years, national=5_000_000, local=5_000_000),
    ]
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config(profiles=profiles, horizons=[12, 24])))
    report = run_pension_campaign(spec, settings, seed=7)
    by_profile: dict[str, dict[int, list[int]]] = {}
    for row in report.cohort_rows:
        if row.arm_id == "sp500":
            by_profile.setdefault(row.profile_id, {}).setdefault(row.horizon_months, []).append(row.credit_received_krw)
    assert by_profile["zero"] == {12: [0, 0], 24: [0]}
    assert by_profile["partial"] == {12: [0, 0], 24: [495_000]}
    assert by_profile["full"] == {12: [0, 0], 24: [990_000]}
    zero_rows = [row for row in report.cohort_rows if row.profile_id == "zero"]
    assert all(
        row.terminal_nav_krw == 0 and row.paired_wealth_ratio == 1.0 and row.cashflow_normalized_rate is None
        for row in zero_rows
    )
    two_year_balances = {
        row.profile_id: row.terminal_nav_krw
        for row in report.cohort_rows if row.arm_id == "sp500" and row.horizon_months == 24
    }
    assert two_year_balances["zero"] == 0 < two_year_balances["partial"] < two_year_balances["full"]
    two_year_rows = {
        row.profile_id: row
        for row in report.cohort_rows if row.arm_id == "sp500" and row.horizon_months == 24
    }
    assert two_year_rows["zero"].contributed_krw == 0
    assert two_year_rows["partial"].contributed_krw == 6_000_000
    assert two_year_rows["full"].contributed_krw == 12_000_000
    assert two_year_rows["partial"].gross_credit_krw == two_year_rows["partial"].usable_credit_krw == 990_000
    assert two_year_rows["full"].gross_credit_krw == two_year_rows["full"].usable_credit_krw == 1_980_000


def test_cashflow_normalized_rate_uses_settled_refund_and_terminal_nav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported annual rate uses actual contribution and refund dates exactly once."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config(horizons=[24])))
    report = run_pension_campaign(spec, settings, seed=7)
    row = next(row for row in report.cohort_rows if row.arm_id == "sp500")
    def at(day: date) -> datetime:
        return datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    expected = xirr([
        (at(date(2023, 1, 15)), -6_000_000.0),
        (at(date(2024, 1, 15)), -6_000_000.0),
        (at(date(2024, 5, 31)), 990_000.0),
        (at(row.cohort_end), float(row.terminal_nav_krw)),
    ])
    without_refund = xirr([
        (at(date(2023, 1, 15)), -6_000_000.0),
        (at(date(2024, 1, 15)), -6_000_000.0),
        (at(row.cohort_end), float(row.terminal_nav_krw)),
    ])
    assert row.cashflow_normalized_rate == pytest.approx(expected)
    assert row.cashflow_normalized_rate > without_refund
    summary = next(summary for summary in report.summaries if summary.arm_id == "sp500")
    assert summary.median_cashflow_normalized_rate == pytest.approx(expected)
    saved = json.loads(write_pension_campaign_report(report, settings, experiment_id="rate_guard").read_text())
    assert saved["rows"][0]["cashflow_normalized_rate"] == pytest.approx(expected)
    assert saved["summaries"][0]["median_cashflow_normalized_rate"] == pytest.approx(expected)


def test_real_outcomes_require_visible_trusted_cpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An available CPI partition deflates terminal KRW; absent CPI stays explicit null."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config(horizons=[24])))
    without = run_pension_campaign(spec, settings, seed=7)
    assert without.real_data_status == "UNAVAILABLE_NO_TRUSTED_CPI"
    assert all(row.after_tax_wealth_real_krw is None for row in without.cohort_rows)

    _persist_cpi_lake(settings)
    with_cpi = run_pension_campaign(spec, settings, seed=7)
    assert with_cpi.real_data_status == "AVAILABLE"
    for row in with_cpi.cohort_rows:
        assert row.terminal_nav_real_krw == pytest.approx(row.terminal_nav_krw * 100.0 / 110.0)
        assert row.after_tax_wealth_real_krw == pytest.approx(row.after_tax_wealth_krw * 100.0 / 110.0)
    saved = json.loads(write_pension_campaign_report(with_cpi, settings, experiment_id="cpi_guard").read_text())
    assert saved["real_data_status"] == "AVAILABLE"
    assert saved["rows"][0]["after_tax_wealth_real_krw"] == pytest.approx(
        with_cpi.cohort_rows[0].after_tax_wealth_real_krw
    )
    stale_config = _campaign_config(horizons=[24])
    stale_config["max_cpi_age_days"] = 30
    stale_spec = load_pension_campaign_spec(_write_config(tmp_path, stale_config))
    stale = run_pension_campaign(stale_spec, settings, seed=7)
    assert stale.real_data_status == "PARTIAL_CPI_COVERAGE"
    assert all(row.after_tax_wealth_real_krw is None for row in stale.cohort_rows)

    later_settings = DataSettings(data_root="late-cpi-data")
    _persist_proxy_lake(later_settings, date(2023, 1, 1), date(2024, 12, 31))
    late_frame = pl.DataFrame(
        {"period_end": [date(2024, 10, 31)], "value": [110.0], "source": ["synthetic"],
         "retrieved_at": [_RETRIEVED_AT]},
        schema=dict(spec_for(Dataset.CPI).columns),
    )
    persist_ingest(late_frame, Dataset.CPI, _payload(), later_settings)
    late = run_pension_campaign(spec, later_settings, seed=7)
    assert late.real_data_status == "PARTIAL_CPI_COVERAGE"
    assert all(row.terminal_nav_real_krw is None for row in late.cohort_rows)


def test_live_history_cannot_emit_20y_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 240-month live horizon fits no cohort while the proxy panel runs."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_live_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    live_path = _write_config(tmp_path, _campaign_config(mode="kr_live", horizons=[240],
        arms=[{"arm_id": "sp500", "role": "baseline", "targets": {"379800": 1.0}}]))
    live_spec = load_pension_campaign_spec(live_path)
    with pytest.raises(ValueError, match="no cohorts fit"):
        run_pension_campaign(live_spec, settings, seed=7)

    _persist_proxy_lake(settings, date(2005, 1, 1), date(2024, 12, 31))
    years = list(range(2005, 2025))
    proxy_config = _campaign_config(start="2005-01-01", end="2024-12-31", horizons=[240], step=120,
        profiles=[_profile_entry("accum", "1985-01-01", "2000-01-01", years)])
    proxy_spec = load_pension_campaign_spec(_write_config(tmp_path, proxy_config))
    proxy_report = run_pension_campaign(proxy_spec, settings, seed=7)
    assert all(row.horizon_months == 240 for row in proxy_report.cohort_rows)
    assert proxy_report.market_mode == "us_proxy"


def test_retirement_payout_uses_threshold_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A payout pushing aggregate income above 15m is taxed at 15%, not the age rate."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    years = [2023, 2024]
    profiles = [_profile_entry("retiree", "1960-01-01", "2010-01-01", years, other=14_000_000)]
    config = _campaign_config(profiles=profiles, horizons=[24], retirement=2024, withdrawals={"2024": 2_000_000})
    spec = load_pension_campaign_spec(_write_config(tmp_path, config))
    report = run_pension_campaign(spec, settings, seed=7)
    rows = [row for row in report.cohort_rows if row.arm_id == "sp500" and row.after_tax_payout_krw is not None]
    assert rows
    assert all(row.withdrawal_tax_krw == 300_000 + 30_000 for row in rows)
    assert all(row.after_tax_payout_krw == 2_000_000 - 330_000 for row in rows)
    baseline = rows[0]
    candidate = next(row for row in report.cohort_rows if row.arm_id == "nasdaq")
    for row in (baseline, candidate):
        assert row.after_tax_wealth_krw == row.terminal_nav_krw + row.credit_received_krw + row.after_tax_payout_krw
        assert row.is_retirement_terminal is False
    assert candidate.paired_wealth_ratio == pytest.approx(candidate.after_tax_wealth_krw / baseline.after_tax_wealth_krw)


def test_payout_shortfall_is_reported_without_negative_account_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retirement spending request larger than the account is reported as an unmet amount."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    years = [2023, 2024]
    retiree = _profile_entry("retiree", "1960-01-01", "2000-01-01", years)
    spec = load_pension_campaign_spec(_write_config(
        tmp_path, _campaign_config(profiles=[retiree], horizons=[24], retirement=2024,
                                   withdrawals={"2024": 500_000_000}),
    ))
    report = run_pension_campaign(spec, settings, seed=7)
    assert all(row.payout_shortfall_krw > 0 for row in report.cohort_rows)
    assert all(row.terminal_nav_krw >= 0 and not row.is_retirement_terminal for row in report.cohort_rows)
    assert all(row.payout_shortfall_krw + row.after_tax_payout_krw < 500_000_000 for row in report.cohort_rows)


def test_campaign_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical inputs produce identical cohort rows on a second run."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config()))
    first = run_pension_campaign(spec, settings, seed=7)
    second = run_pension_campaign(spec, settings, seed=7)
    assert first.cohort_rows == second.cohort_rows
    assert first.summaries == second.summaries


def test_shipped_campaign_config_loads() -> None:
    """The shipped campaign declares four arms with one baseline and a sensitivity."""
    spec = load_pension_campaign_spec(_REPO / "experiments" / "pension_campaign_v1.json")
    assert len(spec.arms) == 4
    assert [arm.arm_id for arm in spec.arms if arm.role == "baseline"] == ["sp500_100"]
    assert [arm.arm_id for arm in spec.arms if arm.role == "sensitivity"] == ["qqq80_soxx20"]
    assert spec.baseline_arm_id == "sp500_100"


def test_campaign_spec_rejects_bad_configs(tmp_path: Path) -> None:
    """Omitted horizons, unmatched cashflows, and proxy-as-live mappings fail closed."""
    base = _campaign_config()
    cases: list[tuple[dict[str, Any], str]] = [
        ({"horizons_months": []}, "nonempty list"),
        ({"horizons_months": "x"}, "nonempty list"),
        ({"horizons_months": [0]}, "positive integers"),
        ({"step_months": 0}, "positive integer"),
        ({"start": "not-a-date"}, "ISO date"),
        ({"start": "2024-01-01", "end": "2023-01-01"}, "after end"),
        ({"market_mode": "tape"}, "unknown market_mode"),
        ({"name": "  "}, "non-blank"),
        ({"baseline_arm_id": "ghost"}, "not found"),
        ({"arms": [{"arm_id": "sp500", "role": "candidate", "targets": {"SPY": 1.0}}]}, "role must be baseline"),
        ({"profiles": []}, "nonempty list"),
        ({"arms": []}, "nonempty list"),
        ({"execution_spread_bps": -1.0}, "must lie in"),
        ({"commission_bps": 20000.0}, "must lie in"),
        ({"max_fx_age_days": -1}, "nonnegative integer"),
        ({"max_cpi_age_days": "old"}, "nonnegative integer"),
        ({"extra_annual_drag_by_ticker": {"SPY": 1.5}}, "must lie in"),
        ({"extra_annual_drag_by_ticker": []}, "must be an object"),
        ({"retirement_start_year": "soon"}, "integer year"),
        ({"tax_regime_path": "missing.json"}, "not found"),
        ({"etf_identity_path": "missing.json"}, "not found"),
        ({"baseline_arm_id": ""}, "non-blank"),
        ({"available_cash_events_krw": []}, "non-empty object"),
        ({"available_cash_events_krw": {"20xx": 1}}, "ISO date"),
        ({"available_cash_events_krw": {"2023-01-03": -1}}, "nonnegative integer"),
        ({"tax_credit_settlement_dates": []}, "must be an object"),
        ({"tax_credit_settlement_dates": {"20xx": "2024-05-31"}}, "integer years"),
        ({"tax_credit_settlement_dates": {"2023": "2023-12-31"}}, "must follow the tax year"),
        ({"contribution_dates": []}, "must be an object"),
        ({"contribution_dates": {"20xx": []}}, "integer years"),
        ({"contribution_dates": {"2023": "x"}}, "must be an array"),
        ({"profiles": [1]}, "must be objects"),
        ({"profiles": [{"profile_id": "x"}]}, "missing field"),
        ({"profiles": [_profile_entry("", "1985-01-01", "2015-01-01", [2023, 2024])]}, "non-blank"),
        ({"arms": [1]}, "must be objects"),
        ({"arms": [{"arm_id": "", "role": "baseline", "targets": {"SPY": 1.0}}]}, "non-blank"),
        ({"arms": [
            {"arm_id": "sp500", "role": "baseline", "targets": {"SPY": 1.0}},
            {"arm_id": "sp500", "role": "candidate", "targets": {"SPY": 1.0}},
        ]}, "duplicate arm id"),
        ({"arms": [{"arm_id": "sp500", "role": "baseline", "targets": {"": 1.0}}]}, "blank ticker"),
        ({"arms": [{"arm_id": "sp500", "role": "baseline", "targets": {"SPY": "x"}}]}, "must be numeric"),
        ({"arms": [{"arm_id": "sp500", "role": "baseline", "targets": {"SPY": 0.0}}]}, "must lie in"),
        ({"arms": [{"arm_id": "sp500", "role": "baseline", "targets": []}]}, "non-empty object"),
    ]
    for mutate, match in cases:
        document = dict(base)
        document.update(mutate)
        with pytest.raises(ValueError, match=match):
            load_pension_campaign_spec(_write_config(tmp_path, document))
    for bad_income, match in (([], "non-empty object"), ({"20xx": 1}, "integer years")):
        invalid_profile = dict(base)
        invalid_profile["profiles"] = [
            {**_profile_entry("bad_income", "1985-01-01", "2015-01-01", [2023, 2024]),
             "annual_income_krw": bad_income}
        ]
        with pytest.raises(ValueError, match=match):
            load_pension_campaign_spec(_write_config(tmp_path, invalid_profile))
    cashless = dict(base)
    cashless["tax_credit_settlement_dates"] = {"2023": "2024-05-31"}
    with pytest.raises(ValueError, match="no entry for 2024"):
        load_pension_campaign_spec(_write_config(tmp_path, cashless))
    unmatched = dict(base)
    unmatched["available_cash_events_krw"] = {"2022-01-03": 6_000_000}
    with pytest.raises(ValueError, match="outside the campaign window"):
        load_pension_campaign_spec(_write_config(tmp_path, unmatched))
    stray = dict(base)
    stray["contribution_dates"] = {"2023": ["2022-01-15"]}
    with pytest.raises(ValueError, match="outside the campaign window"):
        load_pension_campaign_spec(_write_config(tmp_path, stray))
    early_withdrawal = dict(base)
    early_withdrawal["withdrawal_amounts_krw"] = {"2022": 100}
    with pytest.raises(ValueError, match="outside the campaign window"):
        load_pension_campaign_spec(_write_config(tmp_path, early_withdrawal))
    proxy_as_live = dict(base)
    proxy_as_live["market_mode"] = "kr_live"
    with pytest.raises(ValueError, match="not a listed Korean ETF"):
        load_pension_campaign_spec(_write_config(tmp_path, proxy_as_live))
    live_as_proxy = _campaign_config(mode="kr_live",
        arms=[{"arm_id": "sp500", "role": "baseline", "targets": {"379800": 1.0}}])
    live_as_proxy["market_mode"] = "us_proxy"
    with pytest.raises(ValueError, match="labels listed fund"):
        load_pension_campaign_spec(_write_config(tmp_path, live_as_proxy))
    both_baseline = dict(base)
    both_baseline["arms"] = [
        {"arm_id": "sp500", "role": "baseline", "targets": {"SPY": 1.0}},
        {"arm_id": "sp500b", "role": "baseline", "targets": {"SPY": 1.0}},
    ]
    with pytest.raises(ValueError, match="exactly one baseline"):
        load_pension_campaign_spec(_write_config(tmp_path, both_baseline))
    bad_targets = dict(base)
    bad_targets["arms"] = [{"arm_id": "sp500", "role": "baseline", "targets": {"SPY": 0.5}}]
    with pytest.raises(ValueError, match="must sum to 1"):
        load_pension_campaign_spec(_write_config(tmp_path, bad_targets))
    bad_role = dict(base)
    bad_role["arms"] = [{"arm_id": "sp500", "role": "operational", "targets": {"SPY": 1.0}}]
    with pytest.raises(ValueError, match="unknown arm role"):
        load_pension_campaign_spec(_write_config(tmp_path, bad_role))
    bad_kind = dict(base)
    bad_kind["profiles"] = [_profile_entry("x", "1985-01-01", "2015-01-01", [2023, 2024])]
    bad_kind["profiles"][0]["income_kind"] = "business"
    with pytest.raises(ValueError, match="unsupported"):
        load_pension_campaign_spec(_write_config(tmp_path, bad_kind))
    dup_profile = dict(base)
    dup_profile["profiles"] = [
        _profile_entry("same", "1985-01-01", "2015-01-01", [2023, 2024]),
        _profile_entry("same", "1985-01-01", "2015-01-01", [2023, 2024]),
    ]
    with pytest.raises(ValueError, match="duplicate profile_id"):
        load_pension_campaign_spec(_write_config(tmp_path, dup_profile))
    no_object = tmp_path / "campaign.json"
    no_object.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_pension_campaign_spec(no_object)
    missing_field = dict(base)
    del missing_field["arms"]
    with pytest.raises(ValueError, match="missing field"):
        load_pension_campaign_spec(_write_config(tmp_path, missing_field))


def test_campaign_run_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent sources, zero baselines, and unwritable outputs fail closed."""
    settings = _settings(tmp_path, monkeypatch)
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config()))
    with pytest.raises(PensionDataError, match="absent or stale"):
        run_pension_campaign(spec, settings, seed=7)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    broke = dict(_campaign_config())
    broke["available_cash_events_krw"] = {"2023-01-03": 0, "2024-01-03": 0}
    broke["profiles"] = [_profile_entry("accum", "1985-01-01", "2015-01-01", [2023, 2024], national=0, local=0)]
    cashless_report = run_pension_campaign(load_pension_campaign_spec(_write_config(tmp_path, broke)), settings, seed=7)
    assert all(row.terminal_nav_krw == 0 and row.paired_wealth_ratio == 1.0 for row in cashless_report.cohort_rows)
    report = PensionCampaignReport(
        name="probe", market_mode=PensionMarketMode.US_PROXY,
        market_coverage_start=date(2023, 1, 1), market_coverage_end=date(2024, 12, 31),
        cohort_rows=(), summaries=(),
        real_data_status="UNAVAILABLE_NO_TRUSTED_CPI", evidence_status="x",
    )
    write_pension_campaign_report(report, settings, experiment_id="abc123")
    live_report = PensionCampaignReport(
        name="probe", market_mode=PensionMarketMode.KR_LIVE,
        market_coverage_start=date(2023, 1, 1), market_coverage_end=date(2024, 12, 31),
        cohort_rows=(), summaries=(),
        real_data_status="UNAVAILABLE_NO_TRUSTED_CPI", evidence_status="x",
    )
    empty_settings = DataSettings(data_root="empty-data")
    write_pension_campaign_report(live_report, empty_settings, experiment_id="abc123")
    blocked = settings.resolved_data_root() / "results" / "probe"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    if blocked.exists():
        for child in blocked.iterdir():
            child.unlink()
        blocked.rmdir()
    blocked.write_text("blocked", encoding="utf-8")
    with pytest.raises(OSError, match="unwritable"):
        write_pension_campaign_report(report, settings, experiment_id="abc123")


def _persist_proxy_lake_with_fx_gap(
    settings: DataSettings, start: date, end: date, *, gap: tuple[date, ...],
) -> tuple[date, ...]:
    sessions = list(load_calendar("XNYS").sessions(start, end))
    drifts = {"SPY": (400.0, 0.10), "QQQ": (300.0, 0.16)}
    rows = []
    for ticker, (base, drift) in drifts.items():
        for index, day in enumerate(sessions):
            price = base + drift * index
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 10_000,
                    "adjusted_close": price,
                    "dividend": 0.0,
                    "split_factor": 1.0,
                    "source": "synthetic",
                    "retrieved_at": _RETRIEVED_AT,
                }
            )
    prices = pl.DataFrame(
        rows,
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "adjusted_close": pl.Float64,
            "dividend": pl.Float64,
            "split_factor": pl.Float64,
            "source": pl.String,
            "retrieved_at": pl.Datetime("us", "UTC"),
        },
    ).select(list(spec_for(Dataset.PRICES).columns))
    persist_ingest(prices, Dataset.PRICES, _payload(), settings)
    fx_dates = [day for day in sessions if day not in set(gap)]
    fx = pl.DataFrame(
        {"date": fx_dates, "usdkrw": [1300.0] * len(fx_dates), "source": ["synthetic"] * len(fx_dates),
         "retrieved_at": [_RETRIEVED_AT] * len(fx_dates)},
        schema=dict(spec_for(Dataset.FX_KRW_BASE).columns),
    )
    persist_ingest(fx, Dataset.FX_KRW_BASE, _payload(), settings)
    return tuple(sessions)


def _persist_fx_fallback(settings: DataSettings, days: tuple[date, ...], *, quote: float = 1305.0) -> None:
    frame = pl.DataFrame(
        {"date": list(days), "usdkrw": [quote] * len(days), "source": ["fred"] * len(days),
         "retrieved_at": [_RETRIEVED_AT] * len(days)},
        schema=dict(spec_for(Dataset.FX).columns),
    )
    persist_ingest(frame, Dataset.FX, _payload(), settings)


def _gap_fixture_sessions() -> tuple[tuple[date, date], tuple[date, ...]]:
    sessions = list(load_calendar("XNYS").sessions(date(2023, 1, 1), date(2024, 12, 31)))
    gap = tuple(sessions[100:112])
    return (date(2023, 1, 1), date(2024, 12, 31)), gap


def test_loader_requires_fallback_cap(tmp_path: Path) -> None:
    """The campaign config must carry an explicit fallback share cap."""
    config = _campaign_config()
    del config["max_fx_fallback_share"]
    with pytest.raises(ValueError, match="missing field"):
        load_pension_campaign_spec(_write_config(tmp_path, config))
    config["max_fx_fallback_share"] = 0.1
    spec = load_pension_campaign_spec(_write_config(tmp_path, config))
    assert spec.max_fx_fallback_share == 0.1


def test_loader_rejects_invalid_cap(tmp_path: Path) -> None:
    """Non-finite or out-of-range caps fail closed."""
    for bad in (-0.1, 1.5, float("nan"), True, "0.1"):
        config = _campaign_config()
        config["max_fx_fallback_share"] = bad  # type: ignore[assignment]
        with pytest.raises(ValueError, match=r"finite number in \[0, 1\]"):
            load_pension_campaign_spec(_write_config(tmp_path, config))


def test_holiday_gap_aborts_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A >7-day ECOS gap over US sessions aborts when no fallback partition exists."""
    settings = _settings(tmp_path, monkeypatch)
    (start, end), gap = _gap_fixture_sessions()
    _persist_proxy_lake_with_fx_gap(settings, start, end, gap=gap)
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config()))
    with pytest.raises(PensionDataError, match="stale"):
        run_pension_campaign(spec, settings, seed=7)


def test_holiday_gap_completes_with_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FRED quotes over the gap let the campaign complete without touching the staleness limit."""
    settings = _settings(tmp_path, monkeypatch)
    (start, end), gap = _gap_fixture_sessions()
    _persist_proxy_lake_with_fx_gap(settings, start, end, gap=gap)
    _persist_fx_fallback(settings, gap)
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config()))
    report = run_pension_campaign(spec, settings, seed=7)
    assert spec.max_fx_age_days == 7
    assert report.fx_provenance["fallback_status"] == "APPLIED"
    assert report.fx_provenance["fallback_session_count"] > 0


def test_fallback_share_cap_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cap below the observed share aborts and names both numbers."""
    settings = _settings(tmp_path, monkeypatch)
    (start, end), gap = _gap_fixture_sessions()
    _persist_proxy_lake_with_fx_gap(settings, start, end, gap=gap)
    _persist_fx_fallback(settings, gap)
    config = _campaign_config()
    config["max_fx_fallback_share"] = 0.0
    spec = load_pension_campaign_spec(_write_config(tmp_path, config))
    with pytest.raises(PensionDataError, match="fallback share"):
        run_pension_campaign(spec, settings, seed=7)


def test_fallback_share_at_cap_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap comparison is strict: a share equal to the cap still runs."""
    settings = _settings(tmp_path, monkeypatch)
    (start, end), gap = _gap_fixture_sessions()
    _persist_proxy_lake_with_fx_gap(settings, start, end, gap=gap)
    _persist_fx_fallback(settings, gap)
    wide = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config()))
    observed = run_pension_campaign(wide, settings, seed=7).fx_provenance["fallback_session_share"]
    assert observed > 0
    config = _campaign_config()
    config["max_fx_fallback_share"] = observed
    spec = load_pension_campaign_spec(_write_config(tmp_path, config))
    report = run_pension_campaign(spec, settings, seed=7)
    assert report.fx_provenance["fallback_session_share"] == pytest.approx(observed)


def test_report_discloses_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that uses fallback discloses it in JSON evidence notes and markdown."""
    settings = _settings(tmp_path, monkeypatch)
    (start, end), gap = _gap_fixture_sessions()
    _persist_proxy_lake_with_fx_gap(settings, start, end, gap=gap)
    _persist_fx_fallback(settings, gap)
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config()))
    report = run_pension_campaign(spec, settings, seed=7)
    json_path = write_pension_campaign_report(report, settings, experiment_id="fxdisclose")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["fx_provenance"]["fallback_status"] == "APPLIED"
    assert any("DEXKOUS" in note for note in payload["evidence_notes"])
    markdown = json_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "fx_fallback:" in markdown
    assert "DEXKOUS" in markdown


def test_no_fallback_note_when_unused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A gap-free run records zero fallback sessions and adds no note."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config()))
    report = run_pension_campaign(spec, settings, seed=7)
    assert report.fx_provenance["fallback_session_count"] == 0
    json_path = write_pension_campaign_report(report, settings, experiment_id="fxunused")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert not any("DEXKOUS" in note for note in payload["evidence_notes"])
    assert "fx_fallback:" not in json_path.with_suffix(".md").read_text(encoding="utf-8") or "sessions=0" in json_path.with_suffix(".md").read_text(encoding="utf-8")


def test_kr_live_unaffected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """KR_LIVE reports carry no FX provenance."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_live_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    config = _campaign_config(
        mode="kr_live", arms=[{"arm_id": "sp500", "role": "baseline", "targets": {"379800": 1.0}}],
    )
    spec = load_pension_campaign_spec(_write_config(tmp_path, config))
    report = run_pension_campaign(spec, settings, seed=7)
    assert report.fx_provenance == {}
    json_path = write_pension_campaign_report(report, settings, experiment_id="fxlive")
    assert json.loads(json_path.read_text(encoding="utf-8"))["fx_provenance"] == {}
    assert "fx_fallback:" not in json_path.with_suffix(".md").read_text(encoding="utf-8")


def test_manifest_hashes_include_fallback_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """US-proxy manifest hashes cover the fallback dataset, present or absent."""
    from src.validation.pension_campaign import _manifest_hashes

    from src.sim.pension_engine import PensionMarketMode

    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    hashes = _manifest_hashes(settings, PensionMarketMode.US_PROXY)
    assert str(Dataset.FX) in hashes
    assert hashes[str(Dataset.FX)] is None
    _persist_fx_fallback(settings, (date(2023, 6, 1),))
    hashes = _manifest_hashes(settings, PensionMarketMode.US_PROXY)
    assert hashes[str(Dataset.FX)] is not None


def test_invalid_fx_series_converted_to_data_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt fallback quote aborts through the campaign's documented failure type."""
    settings = _settings(tmp_path, monkeypatch)
    _persist_proxy_lake(settings, date(2023, 1, 1), date(2024, 12, 31))
    bad = pl.DataFrame(
        {"date": [date(2023, 6, 1)], "usdkrw": [0.0], "source": ["fred"],
         "retrieved_at": [_RETRIEVED_AT]},
        schema=dict(spec_for(Dataset.FX).columns),
    )
    persist_ingest(bad, Dataset.FX, _payload(), settings)
    spec = load_pension_campaign_spec(_write_config(tmp_path, _campaign_config()))
    with pytest.raises(PensionDataError, match="fx series is invalid"):
        run_pension_campaign(spec, settings, seed=7)
