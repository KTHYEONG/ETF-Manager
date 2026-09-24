"""Pension cohort campaign: equal-cashflow arms over explicit cohorts with labeled evidence."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Literal, cast

import polars as pl

from src.analytics.metrics import max_drawdown, real_krw, xirr
from src.data.catalog import latest_artifact, load_visible
from src.data.pension_fx import build_krw_fx_series
from src.data.query import load_as_of
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.sim.pension_engine import (
    PensionBacktestConfig,
    PensionDataError,
    PensionMarketMode,
    run_pension_backtest,
)
from src.sim.pension_tax import PensionTaxProfile, load_pension_tax_regime
from src.validation.gate import wealth_quantile
from src.validation.windows import rolling_cohorts

logger = logging.getLogger(__name__)

__all__ = [
    "PensionArmSpec",
    "PensionArmSummary",
    "PensionCampaignReport",
    "PensionCampaignSpec",
    "PensionCohortRow",
    "load_pension_campaign_spec",
    "run_pension_campaign",
    "write_pension_campaign_report",
]

_INSUFFICIENT_EVIDENCE: Final[str] = "INSUFFICIENT_INDEPENDENT_20Y_EVIDENCE"
_FX_FALLBACK_NOTE: Final[str] = (
    "USD/KRW on Korean-holiday sessions comes from FRED DEXKOUS (NY-noon buying rate) "
    "instead of the ECOS base rate; see fx_provenance for count and same-day basis."
)
_SOXX_BREAK_NOTE: Final[str] = "SOXX observations before 2021-06-21 carry the disclosed index-break label."
_SHORT_LIVE_NOTE: Final[str] = "Current live Korean ETF history is too short for an observed 20-year test."
_UNDEFINED_RATIO_NOTE: Final[str] = (
    "Cohorts whose baseline after-tax wealth is zero (no contribution was made, for example a profile "
    "with no usable tax credit) have no defined paired ratio and are excluded from ratio, rate, "
    "and drawdown statistics; see the undefined column."
)
ArmRole = Literal["baseline", "candidate", "sensitivity"]
_ARM_ROLES: Final[tuple[str, ...]] = ("baseline", "candidate", "sensitivity")


@dataclass(frozen=True, slots=True)
class PensionArmSpec:
    """Fixed buy-only allocation and role used for paired campaign comparisons."""

    arm_id: str
    role: ArmRole
    targets: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PensionCampaignSpec:
    """Fixed market, income, contribution, and cohort scenarios loaded from one config."""

    name: str
    start: date
    end: date
    market_mode: PensionMarketMode
    horizons_months: tuple[int, ...]
    step_months: int
    available_cash_events_krw: Mapping[date, int]
    contribution_dates: Mapping[int, tuple[date, ...]]
    tax_credit_settlement_dates: Mapping[int, date]
    retirement_start_year: int
    withdrawal_amounts_krw: Mapping[int, int]
    profiles: tuple[PensionTaxProfile, ...]
    tax_regime_path: str
    etf_identity_path: str
    baseline_arm_id: str
    arms: tuple[PensionArmSpec, ...]
    max_fx_age_days: int
    max_fx_fallback_share: float
    max_cpi_age_days: int
    execution_spread_bps: float
    commission_bps: float
    extra_annual_drag_by_ticker: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PensionCohortRow:
    """One arm/cohort outcome with source status and paired tax-aware measures.

    ``paired_wealth_ratio`` is ``None`` when the baseline's after-tax wealth for
    the same profile and cohort is zero, because a ratio against zero is
    undefined and is never imputed.
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


def _parse_date(value: object, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date, got {value!r}") from exc


def _parse_krw(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return value


def _parse_year_cash(raw: object, name: str) -> dict[int, int]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{name} must be a non-empty object")
    parsed: dict[int, int] = {}
    for key, value in raw.items():
        try:
            year = int(str(key))
        except ValueError as exc:
            raise ValueError(f"{name} keys must be integer years, got {key!r}") from exc
        parsed[year] = _parse_krw(value, f"{name}[{year}]")
    return parsed


def _parse_dated_cash(raw: object) -> dict[date, int]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("available_cash_events_krw must be a non-empty object")
    return {
        _parse_date(day, "available_cash_events_krw date"): _parse_krw(amount, f"available_cash_events_krw[{day}]")
        for day, amount in raw.items()
    }


def _parse_settlement_dates(raw: object) -> dict[int, date]:
    if not isinstance(raw, dict):
        raise ValueError("tax_credit_settlement_dates must be an object")
    parsed: dict[int, date] = {}
    for key, value in raw.items():
        try:
            year = int(str(key))
        except ValueError as exc:
            raise ValueError(f"tax_credit_settlement_dates keys must be integer years, got {key!r}") from exc
        day = _parse_date(value, f"tax_credit_settlement_dates[{year}]")
        if day <= date(year, 12, 31):
            raise ValueError(f"tax_credit_settlement_dates[{year}] must follow the tax year")
        parsed[year] = day
    return parsed


def _parse_contribution_dates(raw: object) -> dict[int, tuple[date, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("contribution_dates must be an object")
    parsed: dict[int, tuple[date, ...]] = {}
    for key, value in raw.items():
        try:
            year = int(str(key))
        except ValueError as exc:
            raise ValueError(f"contribution_dates keys must be integer years, got {key!r}") from exc
        if not isinstance(value, list):
            raise ValueError(f"contribution_dates[{year}] must be an array of ISO dates")
        parsed[year] = tuple(_parse_date(item, f"contribution_dates[{year}]") for item in value)
    return parsed


def _parse_profile(raw: object) -> PensionTaxProfile:
    if not isinstance(raw, dict):
        raise ValueError("profiles entries must be objects")
    try:
        profile_id = str(raw["profile_id"]).strip()
        birth = _parse_date(raw["birth_date"], "birth_date")
        opened = _parse_date(raw["account_open_date"], "account_open_date")
        start_raw = raw.get("pension_start_date")
        pension_start = _parse_date(start_raw, "pension_start_date") if start_raw is not None else None
        kind = str(raw["income_kind"])
        income = _parse_year_cash(raw["annual_income_krw"], "annual_income_krw")
        national = _parse_year_cash(raw["remaining_national_tax_krw"], "remaining_national_tax_krw")
        local = _parse_year_cash(raw["remaining_local_tax_krw"], "remaining_local_tax_krw")
        other = _parse_year_cash(raw["other_private_pension_income_krw"], "other_private_pension_income_krw")
    except KeyError as exc:
        raise ValueError(f"profile is missing field {exc}") from exc
    if not profile_id:
        raise ValueError("profile_id must be non-blank")
    if kind not in ("wage", "comprehensive"):
        raise ValueError(f"income_kind {kind!r} is unsupported")
    income_kind = cast("Literal['wage', 'comprehensive']", kind)
    return PensionTaxProfile(
        profile_id=profile_id,
        birth_date=birth,
        account_open_date=opened,
        pension_start_date=pension_start,
        income_kind=income_kind,
        annual_income_krw=income,
        remaining_national_tax_krw=national,
        remaining_local_tax_krw=local,
        other_private_pension_income_krw=other,
    )


def load_pension_campaign_spec(path: str | Path) -> PensionCampaignSpec:
    """Load fixed arm, tax-profile, cash-availability, and source-mode contracts.

    Returns: A validated comparison with one baseline and source-labeled arms.
    Raises: ValueError for omitted horizons, unmatched cashflows, invalid fund mapping,
        or attempts to label proxy observations as live Korean ETF prices.
    """
    from src.data.pension_market import load_pension_etf_identities

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("pension campaign JSON must be an object")
    try:
        name = str(document["name"]).strip()
        if not name:
            raise ValueError("name must be non-blank")
        start = _parse_date(document["start"], "start")
        end = _parse_date(document["end"], "end")
        if start > end:
            raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
        try:
            market_mode = PensionMarketMode(str(document["market_mode"]))
        except ValueError as exc:
            raise ValueError(f"unknown market_mode {document.get('market_mode')!r}") from exc
        raw_horizons = document["horizons_months"]
        if not isinstance(raw_horizons, list) or not raw_horizons:
            raise ValueError("horizons_months must be a nonempty list")
        horizons: list[int] = []
        for item in raw_horizons:
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(f"horizons_months entries must be positive integers, got {item!r}")
            horizons.append(item)
        step = document["step_months"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(f"step_months must be a positive integer, got {step!r}")
        available = _parse_dated_cash(document["available_cash_events_krw"])
        for day in available:
            if day < start or day > end:
                raise ValueError(f"available cash date {day.isoformat()} lies outside the campaign window")
        contribution_dates = _parse_contribution_dates(document.get("contribution_dates", {}))
        for year, dates in contribution_dates.items():
            for day in dates:
                if day.year != year or day < start or day > end:
                    raise ValueError(f"contribution date {day.isoformat()} lies outside the campaign window")
        settlement_dates = _parse_settlement_dates(document["tax_credit_settlement_dates"])
        for year in range(start.year, end.year + 1):
            if year not in settlement_dates:
                raise ValueError(f"tax_credit_settlement_dates has no entry for {year}")
        retirement_start_year = document["retirement_start_year"]
        if isinstance(retirement_start_year, bool) or not isinstance(retirement_start_year, int):
            raise ValueError("retirement_start_year must be an integer year")
        raw_withdrawals = document.get("withdrawal_amounts_krw")
        withdrawals = _parse_year_cash(raw_withdrawals, "withdrawal_amounts_krw") if raw_withdrawals else {}
        for year in withdrawals:
            if year < start.year or year > end.year:
                raise ValueError(f"withdrawal year {year} lies outside the campaign window")
        raw_profiles = document["profiles"]
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ValueError("profiles must be a nonempty list")
        profiles = tuple(_parse_profile(entry) for entry in raw_profiles)
        if len({profile.profile_id for profile in profiles}) != len(profiles):
            raise ValueError("duplicate profile_id")
        tax_regime_path = str(document["tax_regime_path"]).strip()
        etf_identity_path = str(document["etf_identity_path"]).strip()
        if not tax_regime_path or not Path(tax_regime_path).is_file():
            raise ValueError(f"tax_regime_path not found: {tax_regime_path!r}")
        if not etf_identity_path or not Path(etf_identity_path).is_file():
            raise ValueError(f"etf_identity_path not found: {etf_identity_path!r}")
        baseline_arm_id = str(document["baseline_arm_id"]).strip()
        if not baseline_arm_id:
            raise ValueError("baseline_arm_id must be non-blank")
        raw_arms = document["arms"]
        if not isinstance(raw_arms, list) or not raw_arms:
            raise ValueError("arms must be a nonempty list")
        arms: list[PensionArmSpec] = []
        seen_ids: set[str] = set()
        for entry in raw_arms:
            if not isinstance(entry, dict):
                raise ValueError("arm entries must be objects")
            arm_id = str(entry["arm_id"]).strip()
            if not arm_id:
                raise ValueError("arm_id must be non-blank")
            if arm_id in seen_ids:
                raise ValueError(f"duplicate arm id {arm_id!r}")
            seen_ids.add(arm_id)
            role = str(entry["role"])
            if role not in _ARM_ROLES:
                raise ValueError(f"unknown arm role {role!r}")
            raw_targets = entry["targets"]
            if not isinstance(raw_targets, dict) or not raw_targets:
                raise ValueError(f"arm {arm_id!r} targets must be a non-empty object")
            targets: dict[str, float] = {}
            for ticker, weight in raw_targets.items():
                if not str(ticker).strip():
                    raise ValueError(f"arm {arm_id!r} has a blank ticker")
                if isinstance(weight, bool) or not isinstance(weight, float | int):
                    raise ValueError(f"arm {arm_id!r} weight for {ticker!r} must be numeric")
                if not 0.0 < float(weight) <= 1.0:
                    raise ValueError(f"arm {arm_id!r} weight for {ticker!r} must lie in (0, 1]")
                targets[str(ticker)] = float(weight)
            if abs(sum(targets.values()) - 1.0) > 1e-9:
                raise ValueError(f"arm {arm_id!r} targets must sum to 1")
            role_literal = cast("Literal['baseline', 'candidate', 'sensitivity']", role)
            arms.append(PensionArmSpec(arm_id=arm_id, role=role_literal, targets=targets))
        spread = document.get("execution_spread_bps", 0.0)
        commission = document.get("commission_bps", 0.0)
        max_fx_age_days = document["max_fx_age_days"]
        max_cpi_age_days = document["max_cpi_age_days"]
        for label, value in (("max_fx_age_days", max_fx_age_days), ("max_cpi_age_days", max_cpi_age_days)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        max_fx_fallback_share_raw = document["max_fx_fallback_share"]
        if (
            isinstance(max_fx_fallback_share_raw, bool)
            or not isinstance(max_fx_fallback_share_raw, float | int)
            or not math.isfinite(float(max_fx_fallback_share_raw))
            or not 0.0 <= float(max_fx_fallback_share_raw) <= 1.0
        ):
            raise ValueError("max_fx_fallback_share must be a finite number in [0, 1]")
        max_fx_fallback_share = float(max_fx_fallback_share_raw)
        for label, value in (("execution_spread_bps", spread), ("commission_bps", commission)):
            if isinstance(value, bool) or not isinstance(value, float | int) or not 0.0 <= float(value) < 10000.0:
                raise ValueError(f"{label} must lie in [0, 10000)")
        raw_drag = document.get("extra_annual_drag_by_ticker", {})
        if not isinstance(raw_drag, dict):
            raise ValueError("extra_annual_drag_by_ticker must be an object")
        drag: dict[str, float] = {}
        for ticker, value in raw_drag.items():
            if isinstance(value, bool) or not isinstance(value, float | int) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"drag for {ticker!r} must lie in [0, 1)")
            drag[str(ticker)] = float(value)
    except KeyError as exc:
        raise ValueError(f"pension campaign JSON missing field {exc}") from exc
    if baseline_arm_id not in {arm.arm_id for arm in arms}:
        raise ValueError(f"baseline arm {baseline_arm_id!r} not found")
    baseline_roles = [arm.role for arm in arms if arm.arm_id == baseline_arm_id]
    if baseline_roles[0] != "baseline":
        raise ValueError(f"baseline arm {baseline_arm_id!r} role must be baseline")
    if sum(1 for arm in arms if arm.role == "baseline") != 1:
        raise ValueError("exactly one baseline arm is required")
    listed = {identity.ticker for identity in load_pension_etf_identities(etf_identity_path)}
    for arm in arms:
        for ticker in arm.targets:
            is_listed = ticker in listed
            if market_mode is PensionMarketMode.KR_LIVE and not is_listed:
                raise ValueError(f"arm {arm.arm_id!r} maps {ticker!r} as live; it is not a listed Korean ETF")
            if market_mode is PensionMarketMode.US_PROXY and is_listed:
                raise ValueError(f"arm {arm.arm_id!r} labels listed fund {ticker!r} as proxy observations")
    return PensionCampaignSpec(
        name=name,
        start=start,
        end=end,
        market_mode=market_mode,
        horizons_months=tuple(horizons),
        step_months=step,
        available_cash_events_krw=available,
        contribution_dates=contribution_dates,
        tax_credit_settlement_dates=settlement_dates,
        retirement_start_year=retirement_start_year,
        withdrawal_amounts_krw=withdrawals,
        profiles=tuple(profiles),
        tax_regime_path=tax_regime_path,
        etf_identity_path=etf_identity_path,
        baseline_arm_id=baseline_arm_id,
        arms=tuple(arms),
        max_fx_age_days=max_fx_age_days,
        max_fx_fallback_share=max_fx_fallback_share,
        max_cpi_age_days=max_cpi_age_days,
        execution_spread_bps=float(spread),
        commission_bps=float(commission),
        extra_annual_drag_by_ticker=drag,
    )


def _cutoff(end: date) -> datetime:
    return datetime(end.year, end.month, end.day, 23, 59, tzinfo=UTC)


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
    fx_provenance: dict[str, object] = {}
    try:
        if spec.market_mode is PensionMarketMode.KR_LIVE:
            prices = load_visible(settings, Dataset.KR_ETF_PRICES, _cutoff(spec.end))
            fx: pl.DataFrame | None = None
        else:
            prices = load_visible(settings, Dataset.PRICES, _cutoff(spec.end))
            fx = load_visible(settings, Dataset.FX_KRW_BASE, _cutoff(spec.end))
    except UntrustedDatasetError as exc:
        raise PensionDataError(f"pension campaign source is absent or stale: {exc}") from exc
    if spec.market_mode is not PensionMarketMode.KR_LIVE:
        assert fx is not None
        try:
            fallback = load_visible(settings, Dataset.FX, _cutoff(spec.end))
        except UntrustedDatasetError:
            fallback = None
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
        cpi = load_visible(settings, Dataset.CPI, _cutoff(spec.end))
    except UntrustedDatasetError:
        cpi = None
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

    baseline_arm = next(arm for arm in spec.arms if arm.arm_id == spec.baseline_arm_id)
    rows: list[PensionCohortRow] = []
    summaries: list[PensionArmSummary] = []
    baseline_cache: dict[tuple[int, str, str, str], int] = {}

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

    def _baseline_wealth(horizon: int, c_start: date, c_end: date, profile: PensionTaxProfile) -> int:
        key = (horizon, c_start.isoformat(), c_end.isoformat(), profile.profile_id)
        cached = baseline_cache.get(key)
        if cached is not None:
            return cached
        result = run_pension_backtest(_arm_config(baseline_arm, c_start, c_end), prices, fx, profile, regime)
        wealth = result.terminal_nav_krw + sum(amount for _, amount in result.after_tax_external_cashflows_krw)
        baseline_cache[key] = wealth
        return wealth

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
                    base_wealth = _baseline_wealth(horizon, c_start, c_end, profile)
                    if base_wealth == 0:
                        ratio: float | None = None
                    else:
                        ratio = 1.0 if arm.arm_id == spec.baseline_arm_id else wealth / base_wealth
                    if ratio is not None:
                        ratios.append(ratio)
                        defined_by_profile[profile.profile_id] += 1
                    else:
                        undefined_count += 1
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
                            (datetime.combine(c_end, datetime.min.time(), tzinfo=UTC), float(result.terminal_nav_krw))
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
                        _baseline_wealth(horizon, w_start, w_end, profile) != 0 for profile in spec.profiles
                    )
                )
                median_ratio: float | None = wealth_quantile(ratios, 0.5)
                worst_ratio: float | None = min(ratios)
                median_drawdown: float | None = wealth_quantile(drawdowns, 0.5)
            else:
                informative_windows = 0
                median_ratio = None
                worst_ratio = None
                median_drawdown = None
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
    )


def _manifest_hashes(settings: DataSettings, mode: PensionMarketMode) -> dict[str, str | None]:
    datasets = (
        (Dataset.KR_ETF_PRICES, Dataset.CPI)
        if mode is PensionMarketMode.KR_LIVE else (Dataset.PRICES, Dataset.FX_KRW_BASE, Dataset.FX, Dataset.CPI)
    )
    hashes: dict[str, str | None] = {}
    for dataset in datasets:
        try:
            hashes[str(dataset)] = latest_artifact(settings, dataset).manifest.normalized_sha256
        except UntrustedDatasetError:
            hashes[str(dataset)] = None
    return hashes


def write_pension_campaign_report(
    report: PensionCampaignReport,
    settings: DataSettings,
    *,
    experiment_id: str,
    provenance: Mapping[str, str] | None = None,
) -> Path:
    """Persist a reproducible, reporting-only JSON and Markdown decision record.

    The artifact lives under the experiment's result directory.

    Returns: The JSON report path.
    Raises: OSError when an output cannot be written safely.
    """
    from src.data.result_store import ResultKind, write_result

    fx_provenance: dict[str, object] = dict(report.fx_provenance)
    evidence_notes: list[str] = [_SOXX_BREAK_NOTE, _SHORT_LIVE_NOTE]
    fallback_session_count = int(cast("int", fx_provenance.get("fallback_session_count", 0) or 0))
    if fallback_session_count > 0:
        evidence_notes.append(_FX_FALLBACK_NOTE)
    if any(summary.undefined_ratio_cohorts > 0 for summary in report.summaries):
        evidence_notes.append(_UNDEFINED_RATIO_NOTE)
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "provenance": dict(provenance or {}),
        "market_mode": str(report.market_mode),
        "market_coverage_start": report.market_coverage_start.isoformat(),
        "market_coverage_end": report.market_coverage_end.isoformat(),
        "real_data_status": report.real_data_status,
        "fx_provenance": fx_provenance,
        "evidence_status": report.evidence_status,
        "evidence_notes": evidence_notes,
        "manifest_hashes": _manifest_hashes(settings, report.market_mode),
        "household_view": None,
        "summaries": [
            {
                "arm_id": summary.arm_id,
                "horizon_months": summary.horizon_months,
                "cohort_count": summary.cohort_count,
                "undefined_ratio_cohorts": summary.undefined_ratio_cohorts,
                "fully_undefined_profiles": list(summary.fully_undefined_profiles),
                "independent_window_count": summary.independent_window_count,
                "underperforming_cohorts": summary.underperforming_cohorts,
                "median_wealth_ratio": summary.median_wealth_ratio,
                "worst_wealth_ratio": summary.worst_wealth_ratio,
                "median_cashflow_normalized_rate": summary.median_cashflow_normalized_rate,
                "median_max_drawdown": summary.median_max_drawdown,
                "evidence_status": summary.evidence_status,
            }
            for summary in report.summaries
        ],
        "rows": [
            {
                "arm_id": row.arm_id,
                "profile_id": row.profile_id,
                "cohort_start": row.cohort_start.isoformat(),
                "cohort_end": row.cohort_end.isoformat(),
                "horizon_months": row.horizon_months,
                "market_mode": str(row.market_mode),
                "terminal_nav_krw": row.terminal_nav_krw,
                "after_tax_wealth_krw": row.after_tax_wealth_krw,
                "terminal_nav_real_krw": row.terminal_nav_real_krw,
                "after_tax_wealth_real_krw": row.after_tax_wealth_real_krw,
                "after_tax_payout_krw": row.after_tax_payout_krw,
                "contributed_krw": row.contributed_krw,
                "gross_credit_krw": row.gross_credit_krw,
                "usable_credit_krw": row.usable_credit_krw,
                "credit_received_krw": row.credit_received_krw,
                "withdrawal_tax_krw": row.withdrawal_tax_krw,
                "payout_shortfall_krw": row.payout_shortfall_krw,
                "is_retirement_terminal": row.is_retirement_terminal,
                "cashflow_normalized_rate": row.cashflow_normalized_rate,
                "max_drawdown": row.max_drawdown,
                "paired_wealth_ratio": row.paired_wealth_ratio,
                "historical_overlap_group": row.historical_overlap_group,
            }
            for row in report.cohort_rows
        ],
    }
    lines = [
        f"# Pension campaign {report.name}",
        "",
        f"experiment_id: {experiment_id}",
        f"market_mode: {report.market_mode}",
        f"market_coverage: {report.market_coverage_start}..{report.market_coverage_end}",
        f"real_data_status: {report.real_data_status}",
        f"evidence_status: {report.evidence_status}",
        f"note: {_SHORT_LIVE_NOTE}",
        f"note: {_SOXX_BREAK_NOTE}",
    ]
    if fx_provenance:
        share_value = float(cast("float", fx_provenance.get("fallback_session_share") or 0.0))
        lines.append(
            f"fx_fallback: status={fx_provenance.get('fallback_status')} "
            f"sessions={fx_provenance.get('fallback_session_count')} "
            f"share={share_value:.4f}"
        )
        if fallback_session_count > 0:
            lines.append(f"note: {_FX_FALLBACK_NOTE}")
    lines.append("household_view: not provided")
    if any(summary.undefined_ratio_cohorts > 0 for summary in report.summaries):
        lines.append(f"note: {_UNDEFINED_RATIO_NOTE}")
    fully_undefined_ids = sorted(
        {profile_id for summary in report.summaries for profile_id in summary.fully_undefined_profiles}
    )
    if fully_undefined_ids:
        lines.append(f"fully_undefined_profiles: {', '.join(fully_undefined_ids)}")
    lines.extend(
        [
            "",
            "| arm | horizon | cohorts | undefined | independent | median | worst | median XIRR | drawdown |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )

    def _fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "N/A"

    lines.extend(
        f"| {summary.arm_id} | {summary.horizon_months} | {summary.cohort_count} "
        f"| {summary.undefined_ratio_cohorts} | {summary.independent_window_count} | {_fmt(summary.median_wealth_ratio)} "
        f"| {_fmt(summary.worst_wealth_ratio)} | "
        f"{summary.median_cashflow_normalized_rate if summary.median_cashflow_normalized_rate is not None else 'N/A'} "
        f"| {_fmt(summary.median_max_drawdown)} |"
        for summary in report.summaries
    )
    try:
        ref = write_result(
            settings,
            experiment=report.name,
            kind=ResultKind.PENSION,
            run_id=experiment_id,
            payload=payload,
            markdown="\n".join(lines) + "\n",
        )
    except OSError as exc:
        raise OSError(f"pension campaign report unwritable: {exc}") from exc
    return ref.json_path
