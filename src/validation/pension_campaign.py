"""Pension cohort campaign: equal-cashflow arms over explicit cohorts with labeled evidence."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Final

import polars as pl

from src.analytics.metrics import max_drawdown, real_krw, xirr
from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.data.catalog import CatalogSnapshot, load_snapshot_visible, resolve_snapshot
from src.data.pension_fx import build_krw_fx_series
from src.data.query import load_as_of
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.sim.after_tax_engine import AfterTaxConfig, run_after_tax
from src.sim.pension_engine import (
    PensionBacktestConfig,
    PensionDataError,
    PensionMarketMode,
    run_pension_backtest,
)
from src.sim.pension_tax import (
    PensionExitKind,
    PensionTaxProfile,
    load_pension_tax_regime,
    quote_pension_exit,
    years_to_draw_at_threshold,
)
from src.sim.tax import KrOverseasTaxRegime, load_tax_regime
from src.validation.gate import wealth_quantile
from src.validation.pension_campaign_config import (
    PensionArmSpec,
    PensionCampaignSpec,
    load_pension_campaign_spec,
)
from src.validation.pension_campaign_report import (
    _EXIT_CONVENTION_NOTE,
    _FX_FALLBACK_NOTE,
    _HOUSEHOLD_NOTE,
    _SHORT_LIVE_NOTE,
    _SOXX_BREAK_NOTE,
    _UNDEFINED_RATIO_NOTE,
    _household_view_payload,
    _manifest_hashes,
    _snapshot_stem,
    write_pension_campaign_report,
)
from src.validation.pension_household import (
    HouseholdRow,
    PensionHouseholdReport,
    evaluate_household_arm_cohort,
    summarize_household,
)
from src.validation.windows import rolling_cohorts

logger = logging.getLogger(__name__)

__all__ = [
    "_EXIT_CONVENTION_NOTE",
    "_FX_FALLBACK_NOTE",
    "_HOUSEHOLD_NOTE",
    "_SHORT_LIVE_NOTE",
    "_SOXX_BREAK_NOTE",
    "_UNDEFINED_RATIO_NOTE",
    "PensionArmSpec",
    "PensionArmSummary",
    "PensionCampaignReport",
    "PensionCampaignSpec",
    "PensionCohortRow",
    "_household_view_payload",
    "_manifest_hashes",
    "load_pension_campaign_spec",
    "run_pension_campaign",
    "write_pension_campaign_report",
]

_INSUFFICIENT_EVIDENCE: Final[str] = "INSUFFICIENT_INDEPENDENT_20Y_EVIDENCE"


@dataclass(frozen=True, slots=True)
class PensionCohortRow:
    """One arm/cohort outcome with source status and paired tax-aware measures.

    ``paired_wealth_ratio`` is ``None`` when the baseline's liquidation wealth for
    the same profile and cohort is zero, because a ratio against zero is
    undefined and is never imputed.

    ``after_tax_wealth_krw`` is the *pre-exit* (tax-deferred) balance plus received
    cash; the liquidation fields apply the exit tax.
    """

    arm_id: str
    profile_id: str
    cohort_start: date
    cohort_end: date
    horizon_months: int
    market_mode: PensionMarketMode
    terminal_nav_krw: int
    after_tax_wealth_krw: int
    terminal_nav_real_krw: float | None
    after_tax_wealth_real_krw: float | None
    liquidation_wealth_krw: int
    liquidation_wealth_real_krw: float | None
    annuity_low_wealth_krw: int
    annuity_high_wealth_krw: int
    years_to_draw_at_threshold: int
    foreign_tax_withheld_krw: int
    after_tax_payout_krw: int | None
    contributed_krw: int
    gross_credit_krw: int
    usable_credit_krw: int
    credit_received_krw: int
    withdrawal_tax_krw: int | None
    payout_shortfall_krw: int
    is_retirement_terminal: bool
    cashflow_normalized_rate: float | None
    max_drawdown: float
    paired_wealth_ratio: float | None
    historical_overlap_group: str


@dataclass(frozen=True, slots=True)
class PensionArmSummary:
    """Median/worst paired outcomes over cohorts with a defined ratio, plus how many cohorts were excluded and the count of independent windows that carry information."""

    arm_id: str
    horizon_months: int
    cohort_count: int
    undefined_ratio_cohorts: int
    fully_undefined_profiles: tuple[str, ...]
    independent_window_count: int
    underperforming_cohorts: int
    median_wealth_ratio: float | None
    worst_wealth_ratio: float | None
    median_annuity_low_ratio: float | None
    median_annuity_high_ratio: float | None
    median_cashflow_normalized_rate: float | None
    median_max_drawdown: float | None
    evidence_status: str


@dataclass(frozen=True, slots=True)
class PensionCampaignReport:
    """Paired cohort evidence with source coverage, tax flows, and no adoption side effect."""

    name: str
    market_mode: PensionMarketMode
    market_coverage_start: date
    market_coverage_end: date
    cohort_rows: tuple[PensionCohortRow, ...]
    summaries: tuple[PensionArmSummary, ...]
    real_data_status: str
    evidence_status: str
    fx_provenance: Mapping[str, object] = field(default_factory=dict)
    foreign_tax_credit_rate: float = 0.0
    foreign_dividend_withholding_rate: float = 0.15
    household: PensionHouseholdReport | None = None
    manifest_hashes: Mapping[str, str | None] = field(default_factory=dict)

def _cutoff(end: date) -> datetime:
    return datetime(end.year, end.month, end.day, 23, 59, tzinfo=UTC)


def _optional_snapshot(settings: DataSettings, dataset: Dataset) -> CatalogSnapshot | None:
    """Snapshot holding one truly optional dataset, or None when never published.

    A missing manifests directory means absent data; manifests that exist but
    fail verification raise UntrustedDatasetError instead of degrading to absence.
    """
    manifests_dir = settings.resolved_data_root() / "manifests" / str(dataset)
    if not manifests_dir.is_dir() or not any(manifests_dir.glob("*.json")):
        return None
    return resolve_snapshot(settings, (dataset,))


def run_pension_campaign(
    spec: PensionCampaignSpec,
    settings: DataSettings,
    *,
    seed: int,
) -> PensionCampaignReport:
    """Evaluate equal-cashflow pension arms over explicit cohorts and payout scenarios.

    Returns: Paired nominal/real KRW outcomes, tax-credit and withdrawal-tax ledgers,
        drawdowns, overlap counts, source coverage, and evidence status.
    Raises: PensionDataError when a required trusted source is absent or stale.
    """
    _ = seed
    regime = load_pension_tax_regime(spec.tax_regime_path)
    general_regime = load_tax_regime(spec.household.general_tax_regime_path) if spec.household is not None else None
    coverage_end = spec.end
    if spec.household is not None:
        assert general_regime is not None
        day = spec.end
        calendar = load_calendar(DEFAULT_CALENDAR_NAME)
        for _ in range(2 + general_regime.settlement_sessions):
            day = calendar.next_session(day)
        coverage_end = day
    fx_provenance: dict[str, object] = {}
    required = (Dataset.KR_ETF_PRICES,) if spec.market_mode is PensionMarketMode.KR_LIVE else (
        Dataset.PRICES, Dataset.FX_KRW_BASE,
    )
    try:
        snapshot = resolve_snapshot(settings, required)
    except UntrustedDatasetError as exc:
        raise PensionDataError(f"pension campaign source is absent or stale: {exc}") from exc
    manifest_stems: dict[str, str | None] = {
        str(dataset): _snapshot_stem(snapshot, dataset) for dataset in required
    }
    try:
        if spec.market_mode is PensionMarketMode.KR_LIVE:
            prices = load_snapshot_visible(snapshot, Dataset.KR_ETF_PRICES, _cutoff(coverage_end))
            fx: pl.DataFrame | None = None
        else:
            prices = load_snapshot_visible(snapshot, Dataset.PRICES, _cutoff(coverage_end))
            fx = load_snapshot_visible(snapshot, Dataset.FX_KRW_BASE, _cutoff(coverage_end))
    except UntrustedDatasetError as exc:
        raise PensionDataError(f"pension campaign source is absent or stale: {exc}") from exc
    if spec.market_mode is not PensionMarketMode.KR_LIVE:
        assert fx is not None
        try:
            fallback_snapshot = _optional_snapshot(settings, Dataset.FX)
            fallback = (
                load_snapshot_visible(fallback_snapshot, Dataset.FX, _cutoff(coverage_end))
                if fallback_snapshot is not None else None
            )
        except UntrustedDatasetError as exc:
            raise PensionDataError(f"pension campaign fx fallback source is damaged: {exc}") from exc
        manifest_stems[str(Dataset.FX)] = (
            _snapshot_stem(fallback_snapshot, Dataset.FX) if fallback_snapshot is not None else None
        )
    if spec.market_mode is not PensionMarketMode.KR_LIVE:
        assert fx is not None
        try:
            series = build_krw_fx_series(fx, fallback)
        except ValueError as exc:
            raise PensionDataError(f"pension campaign fx series is invalid: {exc}") from exc
        fx = series.frame
        targets_union = {ticker for arm in spec.arms for ticker in arm.targets}
        sessions = sorted(
            {
                day
                for day in prices.filter(pl.col("ticker").is_in(sorted(targets_union)))
                .get_column("date")
                .to_list()
                if spec.start <= day <= spec.end
            }
        )
        fallback_sessions = series.fallback_session_dates(sessions)
        share = (len(fallback_sessions) / len(sessions)) if sessions else 0.0
        if share > spec.max_fx_fallback_share:
            raise PensionDataError(
                f"pension campaign fx fallback share {share:.6f} exceeds cap "
                f"{spec.max_fx_fallback_share:.6f} (source {series.fallback_source})"
            )
        fx_provenance = dict(series.provenance(sessions))
        logger.info(
            "[DATA] event=pension_fx_provenance status=%s fallback_sessions=%d share=%.6f",
            str(series.status.value),
            len(fallback_sessions),
            share,
        )
    try:
        cpi_snapshot = _optional_snapshot(settings, Dataset.CPI)
        cpi = (
            load_snapshot_visible(cpi_snapshot, Dataset.CPI, _cutoff(coverage_end))
            if cpi_snapshot is not None else None
        )
    except UntrustedDatasetError as exc:
        raise PensionDataError(f"pension campaign cpi source is damaged: {exc}") from exc
    manifest_stems[str(Dataset.CPI)] = (
        _snapshot_stem(cpi_snapshot, Dataset.CPI) if cpi_snapshot is not None else None
    )
    cpi_cache: dict[date, float | None] = {}

    def _cpi_at(day: date) -> float | None:
        if day not in cpi_cache:
            if cpi is None:
                cpi_cache[day] = None
            else:
                visible = load_as_of(cpi, Dataset.CPI, _cutoff(day))
                valid = visible.filter(pl.col("value").is_finite() & (pl.col("value") > 0.0)).sort("period_end")
                if valid.is_empty():
                    cpi_cache[day] = None
                else:
                    latest = valid.row(valid.height - 1, named=True)
                    cpi_cache[day] = (
                        float(latest["value"])
                        if (day - latest["period_end"]).days <= spec.max_cpi_age_days else None
                    )
        return cpi_cache[day]

    household_rows: list[HouseholdRow] = []
    household_excluded: dict[tuple[str, int, str], int] = {}
    household_cache: dict[tuple[int, str, date, date], float] = {}
    household_runs = 0
    household_cached = 0
    household_regime: KrOverseasTaxRegime | None = None
    if spec.household is not None:
        if cpi is None:
            raise PensionDataError("household view requires trusted CPI for the general-account engine")
        assert general_regime is not None
        household_regime = general_regime

    def _run_household_general(config: AfterTaxConfig) -> float:
        fx_frame = fx
        cpi_frame = cpi
        if fx_frame is None or cpi_frame is None:  # pragma: no cover - excluded by upfront us_proxy/CPI checks
            raise PensionDataError("household view requires us_proxy FX and trusted CPI")
        nonlocal household_runs
        household_runs += 1
        return run_after_tax(config, prices, fx_frame, cpi_frame, None).terminal_after_tax_krw

    baseline_arm = next(arm for arm in spec.arms if arm.arm_id == spec.baseline_arm_id)
    rows: list[PensionCohortRow] = []
    summaries: list[PensionArmSummary] = []
    baseline_cache: dict[tuple[int, str, str, str], tuple[int, int, int]] = {}

    def _arm_config(
        arm: PensionArmSpec, c_start: date, c_end: date
    ) -> PensionBacktestConfig:
        years = range(c_start.year, c_end.year + 1)
        return PensionBacktestConfig(
            start=c_start,
            end=c_end,
            targets=dict(arm.targets),
            available_cash_events_krw={
                day: amount for day, amount in spec.available_cash_events_krw.items() if c_start <= day <= c_end
            },
            contribution_dates={
                year: tuple(day for day in spec.contribution_dates.get(year, ()) if c_start <= day <= c_end)
                for year in years
                if year in spec.contribution_dates
            },
            tax_credit_settlement_dates={year: spec.tax_credit_settlement_dates[year] for year in years},
            retirement_start_year=spec.retirement_start_year,
            withdrawal_amounts_krw={
                year: spec.withdrawal_amounts_krw[year] for year in years if year in spec.withdrawal_amounts_krw
            },
            market_mode=spec.market_mode,
            max_fx_age_days=spec.max_fx_age_days,
            execution_spread_bps=spec.execution_spread_bps,
            commission_bps=spec.commission_bps,
            extra_annual_drag_by_ticker=dict(spec.extra_annual_drag_by_ticker),
        )

    def _exit_wealths(
        c_end: date,
        terminal_nav_krw: int,
        external_krw: int,
        uncredited_krw: int,
        withheld_krw: int,
        profile: PensionTaxProfile,
    ) -> tuple[int, int, int, int]:
        """Return (liquidation, annuity-low, annuity-high, years-to-draw) exit wealth."""
        lump = quote_pension_exit(
            PensionExitKind.LUMP_SUM,
            valuation_date=c_end,
            balance_krw=terminal_nav_krw,
            uncredited_principal_krw=uncredited_krw,
            foreign_tax_withheld_krw=withheld_krw,
            profile=profile,
            regime=regime,
        )
        low = quote_pension_exit(
            PensionExitKind.ANNUITY_LOW,
            valuation_date=c_end,
            balance_krw=terminal_nav_krw,
            uncredited_principal_krw=uncredited_krw,
            foreign_tax_withheld_krw=withheld_krw,
            profile=profile,
            regime=regime,
        )
        high = quote_pension_exit(
            PensionExitKind.ANNUITY_HIGH,
            valuation_date=c_end,
            balance_krw=terminal_nav_krw,
            uncredited_principal_krw=uncredited_krw,
            foreign_tax_withheld_krw=withheld_krw,
            profile=profile,
            regime=regime,
        )
        years = years_to_draw_at_threshold(lump.taxable_krw, regime)
        return (lump.net_krw + external_krw, low.net_krw + external_krw, high.net_krw + external_krw, years)

    def _baseline_exit(horizon: int, c_start: date, c_end: date, profile: PensionTaxProfile) -> tuple[int, int, int]:
        key = (horizon, c_start.isoformat(), c_end.isoformat(), profile.profile_id)
        cached = baseline_cache.get(key)
        if cached is not None:
            return cached
        result = run_pension_backtest(_arm_config(baseline_arm, c_start, c_end), prices, fx, profile, regime)
        external = sum(amount for _, amount in result.after_tax_external_cashflows_krw)
        liquidation, low, high, _ = _exit_wealths(
            c_end,
            result.terminal_nav_krw,
            external,
            result.terminal_uncredited_principal_krw,
            result.foreign_tax_withheld_krw,
            profile,
        )
        baseline_cache[key] = (liquidation, low, high)
        return (liquidation, low, high)

    for horizon in spec.horizons_months:
        cohorts = rolling_cohorts(spec.start, spec.end, horizon_months=horizon, step_months=spec.step_months)
        if not cohorts:
            raise ValueError(f"no cohorts fit horizon {horizon}")
        overlap_group = "overlapping" if spec.step_months < horizon else "independent"
        independent_cohorts = rolling_cohorts(
            spec.start, spec.end, horizon_months=horizon, step_months=horizon
        )
        for arm in spec.arms:
            ratios: list[float] = []
            low_ratios: list[float] = []
            high_ratios: list[float] = []
            normalized_rates: list[float] = []
            drawdowns: list[float] = []
            undefined_count = 0
            defined_by_profile: dict[str, int] = {profile.profile_id: 0 for profile in spec.profiles}
            for profile in spec.profiles:
                for c_start, c_end in cohorts:
                    result = run_pension_backtest(_arm_config(arm, c_start, c_end), prices, fx, profile, regime)
                    wealth = result.terminal_nav_krw + sum(
                        amount for _, amount in result.after_tax_external_cashflows_krw
                    )
                    external_total = sum(amount for _, amount in result.after_tax_external_cashflows_krw)
                    liquidation, low_wealth, high_wealth, years_to_draw = _exit_wealths(
                        c_end,
                        result.terminal_nav_krw,
                        external_total,
                        result.terminal_uncredited_principal_krw,
                        result.foreign_tax_withheld_krw,
                        profile,
                    )
                    lump_net = liquidation - external_total
                    if spec.household is not None:
                        assert household_regime is not None
                        low_net = low_wealth - external_total
                        high_net = high_wealth - external_total
                        cache_size_before = len(household_cache)
                        household_row = evaluate_household_arm_cohort(
                            arm_id=arm.arm_id,
                            profile_id=profile.profile_id,
                            horizon_months=horizon,
                            cohort_start=c_start,
                            cohort_end=c_end,
                            result=result,
                            available_cash_events_krw=spec.available_cash_events_krw,
                            tax_credit_settlement_dates=spec.tax_credit_settlement_dates,
                            pension_nets_krw=(lump_net, low_net, high_net),
                            arm_targets=arm.targets,
                            household_spec=spec.household,
                            general_regime=household_regime,
                            runner=_run_household_general,
                            counterfactual_cache=household_cache,
                            excluded=household_excluded,
                        )
                        if household_row is not None:
                            household_rows.append(household_row)
                            if len(household_cache) == cache_size_before:
                                household_cached += 1
                    start_cpi = _cpi_at(c_start)
                    end_cpi = _cpi_at(c_end)
                    real_nav = (
                        real_krw(result.terminal_nav_krw, cpi_index=end_cpi, cpi_base=start_cpi)
                        if start_cpi is not None and end_cpi is not None else None
                    )
                    real_wealth = (
                        real_krw(wealth, cpi_index=end_cpi, cpi_base=start_cpi)
                        if start_cpi is not None and end_cpi is not None else None
                    )
                    real_liquidation = (
                        real_krw(liquidation, cpi_index=end_cpi, cpi_base=start_cpi)
                        if start_cpi is not None and end_cpi is not None else None
                    )
                    base_liquidation, base_low, base_high = _baseline_exit(horizon, c_start, c_end, profile)
                    if base_liquidation == 0:
                        ratio: float | None = None
                        low_ratio: float | None = None
                        high_ratio: float | None = None
                    elif arm.arm_id == spec.baseline_arm_id:
                        ratio = 1.0
                        low_ratio = 1.0
                        high_ratio = 1.0
                    else:
                        ratio = liquidation / base_liquidation
                        low_ratio = low_wealth / base_low if base_low != 0 else None
                        high_ratio = high_wealth / base_high if base_high != 0 else None
                        if low_ratio is None or high_ratio is None:
                            ratio = None  # pragma: no cover - unreachable under the shipped rate ordering
                    if ratio is None or low_ratio is None or high_ratio is None:
                        undefined_count += 1
                    else:
                        ratios.append(ratio)
                        low_ratios.append(low_ratio)
                        high_ratios.append(high_ratio)
                        defined_by_profile[profile.profile_id] += 1
                    unitized_nav: list[float] = []
                    unit_value = 1.0
                    previous_nav = 0.0
                    previous_contributions = 0
                    withdrawals_by_day = {
                        quote.withdrawal_date: quote.gross_withdrawal_krw for quote in result.withdrawals
                    }
                    for snap in result.snapshots:
                        net_flow = snap.cumulative_contributions_krw - previous_contributions
                        net_flow -= withdrawals_by_day.get(snap.session, 0)
                        if previous_nav > 0:
                            unit_value *= max(snap.nav_krw - net_flow, 0) / previous_nav
                        unitized_nav.append(unit_value)
                        previous_nav = float(snap.nav_krw)
                        previous_contributions = snap.cumulative_contributions_krw
                    drawdown = max_drawdown(unitized_nav)
                    if ratio is not None:
                        drawdowns.append(drawdown)
                    credit_received = sum(
                        credit.national_credit_krw + credit.local_credit_krw
                        for credit in result.tax_credits
                        if spec.tax_credit_settlement_dates[credit.tax_year] <= c_end
                    )
                    if result.withdrawals or result.payout_shortfalls_krw:
                        payout_net = sum(
                            quote.gross_withdrawal_krw - quote.national_tax_krw - quote.local_tax_krw
                            for quote in result.withdrawals
                        )
                        withdrawal_tax = sum(
                            quote.national_tax_krw + quote.local_tax_krw for quote in result.withdrawals
                        )
                    else:
                        payout_net = None
                        withdrawal_tax = None
                    normalized_rate: float | None = None
                    if result.contribution_cashflows_krw:
                        investor_flows = [
                            (datetime.combine(day, datetime.min.time(), tzinfo=UTC), -float(amount))
                            for day, amount in result.contribution_cashflows_krw
                        ]
                        investor_flows.extend(
                            (datetime.combine(day, datetime.min.time(), tzinfo=UTC), float(amount))
                            for day, amount in result.after_tax_external_cashflows_krw
                        )
                        investor_flows.append(
                            (datetime.combine(c_end, datetime.min.time(), tzinfo=UTC), float(lump_net))
                        )
                        normalized_rate = xirr(investor_flows)
                        if ratio is not None and normalized_rate is not None:
                            normalized_rates.append(normalized_rate)
                    rows.append(
                        PensionCohortRow(
                            arm_id=arm.arm_id,
                            profile_id=profile.profile_id,
                            cohort_start=c_start,
                            cohort_end=c_end,
                            horizon_months=horizon,
                            market_mode=spec.market_mode,
                            terminal_nav_krw=result.terminal_nav_krw,
                            after_tax_wealth_krw=wealth,
                            terminal_nav_real_krw=real_nav,
                            after_tax_wealth_real_krw=real_wealth,
                            liquidation_wealth_krw=liquidation,
                            liquidation_wealth_real_krw=real_liquidation,
                            annuity_low_wealth_krw=low_wealth,
                            annuity_high_wealth_krw=high_wealth,
                            years_to_draw_at_threshold=years_to_draw,
                            foreign_tax_withheld_krw=result.foreign_tax_withheld_krw,
                            after_tax_payout_krw=payout_net,
                            contributed_krw=sum(amount for _, amount in result.contribution_cashflows_krw),
                            gross_credit_krw=sum(
                                credit.theoretical_national_credit_krw + credit.theoretical_local_credit_krw
                                for credit in result.tax_credits
                            ),
                            usable_credit_krw=sum(
                                credit.national_credit_krw + credit.local_credit_krw for credit in result.tax_credits
                            ),
                            credit_received_krw=credit_received,
                            withdrawal_tax_krw=withdrawal_tax,
                            payout_shortfall_krw=sum(amount for _, amount in result.payout_shortfalls_krw),
                            is_retirement_terminal=result.is_retirement_terminal,
                            cashflow_normalized_rate=normalized_rate,
                            max_drawdown=drawdown,
                            paired_wealth_ratio=ratio,
                            historical_overlap_group=overlap_group,
                        )
                    )
            fully_undefined = tuple(sorted(pid for pid, count in defined_by_profile.items() if count == 0))
            if ratios:
                informative_windows = sum(
                    1
                    for w_start, w_end in independent_cohorts
                    if any(
                        _baseline_exit(horizon, w_start, w_end, profile)[0] != 0 for profile in spec.profiles
                    )
                )
                median_ratio: float | None = wealth_quantile(ratios, 0.5)
                worst_ratio: float | None = min(ratios)
                median_drawdown: float | None = wealth_quantile(drawdowns, 0.5)
                median_low: float | None = wealth_quantile(low_ratios, 0.5)
                median_high: float | None = wealth_quantile(high_ratios, 0.5)
            else:
                informative_windows = 0
                median_ratio = None
                worst_ratio = None
                median_drawdown = None
                median_low = None
                median_high = None
            summaries.append(
                PensionArmSummary(
                    arm_id=arm.arm_id,
                    horizon_months=horizon,
                    cohort_count=len(ratios),
                    undefined_ratio_cohorts=undefined_count,
                    fully_undefined_profiles=fully_undefined,
                    independent_window_count=informative_windows,
                    underperforming_cohorts=sum(ratio < 1.0 for ratio in ratios),
                    median_wealth_ratio=median_ratio,
                    worst_wealth_ratio=worst_ratio,
                    median_annuity_low_ratio=median_low,
                    median_annuity_high_ratio=median_high,
                    median_cashflow_normalized_rate=(
                        wealth_quantile(normalized_rates, 0.5) if normalized_rates else None
                    ),
                    median_max_drawdown=median_drawdown,
                    evidence_status=_INSUFFICIENT_EVIDENCE,
                )
            )
            logger.info(
                "[ALGO] event=pension_arm_done arm=%s horizon=%d cohorts=%d undefined=%d",
                arm.arm_id,
                horizon,
                len(ratios),
                undefined_count,
            )
    household_report: PensionHouseholdReport | None = None
    if spec.household is not None:
        household_report = PensionHouseholdReport(
            spec=spec.household,
            rows=tuple(household_rows),
            summaries=summarize_household(
                household_rows, baseline_arm_id=spec.baseline_arm_id, excluded=household_excluded
            ),
        )
        logger.info(
            "[ALGO] event=pension_household_done rows=%d general_runs=%d cached=%d excluded=%d",
            len(household_rows),
            household_runs,
            household_cached,
            sum(household_excluded.values()),
        )
    return PensionCampaignReport(
        name=spec.name,
        market_mode=spec.market_mode,
        market_coverage_start=min(prices.get_column("date")),
        market_coverage_end=max(prices.get_column("date")),
        cohort_rows=tuple(rows),
        summaries=tuple(summaries),
        real_data_status=(
            "AVAILABLE" if all(row.after_tax_wealth_real_krw is not None for row in rows)
            else "UNAVAILABLE_NO_TRUSTED_CPI" if cpi is None else "PARTIAL_CPI_COVERAGE"
        ),
        evidence_status=_INSUFFICIENT_EVIDENCE,
        fx_provenance=fx_provenance,
        foreign_tax_credit_rate=regime.foreign_tax_credit_rate,
        foreign_dividend_withholding_rate=regime.foreign_dividend_withholding_rate,
        household=household_report,
        manifest_hashes=manifest_stems,
    )
