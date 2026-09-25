"""Pension campaign definitions: versioned spec parsing without financial side effects."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal, cast

from src.sim.pension_engine import PensionMarketMode
from src.sim.pension_tax import PensionTaxProfile
from src.validation.pension_household import PensionHouseholdSpec, parse_household_spec

__all__ = [
    "ArmRole",
    "PensionArmSpec",
    "PensionCampaignSpec",
    "load_pension_campaign_spec",
]

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
    household: PensionHouseholdSpec | None = None

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
    """Parse one versioned pension campaign definition without altering its financial assumptions.

    Args:
        path: Existing campaign JSON path.

    Returns:
        The existing typed campaign specification.

    Raises:
        ValueError: If dates, tax paths, cash schedules, or arm definitions are invalid.
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
        raw_household = document.get("household_view")
        household = parse_household_spec(raw_household) if raw_household is not None else None
        if household is not None and market_mode is PensionMarketMode.KR_LIVE:
            raise ValueError("household_view requires us_proxy market mode")
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
        household=household,
    )
