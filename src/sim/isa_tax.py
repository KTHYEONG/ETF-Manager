"""Korean ISA (brokerage type) tax ledger: frozen 2026-law closure tax, contribution room, and pension-transfer credit."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal

from src.sim.pension_tax import PensionCreditQuote, PensionTaxProfile, PensionTaxRegime

__all__ = [
    "IsaClosureQuote",
    "IsaTaxClass",
    "IsaTaxRegime",
    "IsaTransferCreditQuote",
    "classify_isa_tax_class",
    "credit_maximizing_transfer_krw",
    "isa_contribution_room_krw",
    "load_isa_tax_regime",
    "quote_isa_closure",
    "quote_isa_transfer_credit",
]

_EXPECTED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "regime_id",
        "policy_year",
        "statute_refs",
        "source_urls",
        "source_checked_date",
        "annual_contribution_limit_krw",
        "lifetime_contribution_limit_krw",
        "carryover_years_cap",
        "minimum_contract_years",
        "general_allowance_krw",
        "preferential_allowance_krw",
        "preferential_wage_threshold_krw",
        "preferential_comprehensive_income_threshold_krw",
        "separate_tax_rate",
        "local_surcharge_rate",
        "pension_transfer_credit_rate",
        "pension_transfer_credit_cap_krw",
    }
)
_AMOUNT_KEYS: Final[tuple[str, ...]] = (
    "annual_contribution_limit_krw",
    "lifetime_contribution_limit_krw",
    "general_allowance_krw",
    "preferential_allowance_krw",
    "preferential_wage_threshold_krw",
    "preferential_comprehensive_income_threshold_krw",
    "pension_transfer_credit_cap_krw",
)
_RATE_KEYS: Final[tuple[str, ...]] = (
    "separate_tax_rate",
    "local_surcharge_rate",
    "pension_transfer_credit_rate",
)
_SUPPORTED_POLICY_YEAR: Final[int] = 2026


class IsaTaxClass(StrEnum):
    """ISA allowance class fixed at the opening or extension date of one account."""

    GENERAL = "general"
    PREFERENTIAL = "preferential"


@dataclass(frozen=True, slots=True)
class IsaTaxRegime:
    """Validated 2026-law ISA parameters for the supported brokerage-ISA scenario.

    Only the general and preferential (서민형) classes are supported; the farmer/fisher
    class is out of scope. Amounts are integer KRW, rates are fractions in [0, 1).
    """

    regime_id: str
    policy_year: int
    statute_refs: tuple[str, ...]
    source_urls: tuple[str, ...]
    source_checked_date: date
    annual_contribution_limit_krw: int
    lifetime_contribution_limit_krw: int
    carryover_years_cap: int
    minimum_contract_years: int
    general_allowance_krw: int
    preferential_allowance_krw: int
    preferential_wage_threshold_krw: int
    preferential_comprehensive_income_threshold_krw: int
    separate_tax_rate: float
    local_surcharge_rate: float
    pension_transfer_credit_rate: float
    pension_transfer_credit_cap_krw: int


@dataclass(frozen=True, slots=True)
class IsaClosureQuote:
    """Tax settled when an ISA is terminated at or after its minimum contract term."""

    balance_krw: int
    principal_krw: int
    net_income_krw: int
    allowance_krw: int
    taxable_krw: int
    national_tax_krw: int
    local_tax_krw: int
    net_proceeds_krw: int


@dataclass(frozen=True, slots=True)
class IsaTransferCreditQuote:
    """Extra pension credit earned by moving matured ISA proceeds into the pension account."""

    tax_year: int
    transfer_krw: int
    eligible_krw: int
    national_credit_krw: int
    local_credit_krw: int
    credited_principal_krw: int
    uncredited_principal_krw: int


def _won(amount_krw: int, rate: float) -> int:
    quantized = Decimal(amount_krw) * Decimal(str(rate))
    return int(quantized.to_integral_value(rounding=ROUND_FLOOR))


def _require_amount(document: dict[str, object], key: str) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"isa regime field {key!r} must be an integer amount of KRW")
    if value < 0:
        raise ValueError(f"isa regime field {key!r} must be nonnegative, got {value!r}")
    return value


def _require_rate(document: dict[str, object], key: str) -> float:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"isa regime field {key!r} must be a finite rate")
    rate = float(value)
    if rate < 0 or rate >= 1:
        raise ValueError(f"isa regime field {key!r} must lie in [0, 1), got {value!r}")
    return rate


def _require_count(document: dict[str, object], key: str) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"isa regime field {key!r} must be an integer count")
    if value < 0:
        raise ValueError(f"isa regime field {key!r} must be nonnegative, got {value!r}")
    return value


def load_isa_tax_regime(path: str | Path) -> IsaTaxRegime:
    """Load a source-backed, frozen ISA tax scenario.

    Returns: Validated policy parameters.
    Raises: ValueError for malformed JSON, a key-set mismatch, non-integer or negative amounts,
        rates outside [0, 1), an unsupported policy year, empty statute references or URLs,
        or internally inconsistent limits.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"isa regime {str(path)!r} must be a JSON object")
    keys = frozenset(document.keys())
    if keys != _EXPECTED_KEYS:
        raise ValueError(
            f"isa regime {str(path)!r} has unexpected keys: missing={sorted(_EXPECTED_KEYS - keys)} "
            f"extra={sorted(keys - _EXPECTED_KEYS)}"
        )
    regime_id = document["regime_id"]
    if not isinstance(regime_id, str) or not regime_id:
        raise ValueError("isa regime field 'regime_id' must be a nonempty string")
    policy_year = document["policy_year"]
    if isinstance(policy_year, bool) or not isinstance(policy_year, int):
        raise ValueError("isa regime field 'policy_year' must be an integer year")
    if policy_year != _SUPPORTED_POLICY_YEAR:
        raise ValueError(f"isa regime field 'policy_year' is unknown: {policy_year!r}")
    statute_refs = document["statute_refs"]
    if (
        not isinstance(statute_refs, list)
        or not statute_refs
        or any(not isinstance(item, str) or not item for item in statute_refs)
    ):
        raise ValueError("isa regime field 'statute_refs' must be a non-empty array of nonempty strings")
    source_urls = document["source_urls"]
    if (
        not isinstance(source_urls, list)
        or not source_urls
        or any(not isinstance(url, str) or not url.startswith("http") for url in source_urls)
    ):
        raise ValueError("isa regime field 'source_urls' must be a non-empty array of http(s) URLs")
    checked_raw = document["source_checked_date"]
    if not isinstance(checked_raw, str):
        raise ValueError("isa regime field 'source_checked_date' must be a YYYY-MM-DD string")
    try:
        checked = date.fromisoformat(checked_raw)
    except ValueError as exc:
        raise ValueError(f"isa regime field 'source_checked_date' must be a YYYY-MM-DD string: {exc}") from exc
    amounts = {key: _require_amount(document, key) for key in _AMOUNT_KEYS}
    rates = {key: _require_rate(document, key) for key in _RATE_KEYS}
    carryover = _require_count(document, "carryover_years_cap")
    minimum_years = _require_count(document, "minimum_contract_years")
    if minimum_years <= 0:
        raise ValueError("isa regime field 'minimum_contract_years' must be positive")
    if amounts["annual_contribution_limit_krw"] <= 0 or amounts["lifetime_contribution_limit_krw"] <= 0:
        raise ValueError("isa regime contribution limits must be positive")
    if amounts["lifetime_contribution_limit_krw"] < amounts["annual_contribution_limit_krw"]:
        raise ValueError("isa regime limits are inconsistent: lifetime cap must cover the annual cap")
    if amounts["preferential_allowance_krw"] < amounts["general_allowance_krw"]:
        raise ValueError("isa regime allowances are inconsistent: preferential allowance must cover general allowance")
    return IsaTaxRegime(
        regime_id=regime_id,
        policy_year=policy_year,
        statute_refs=tuple(statute_refs),
        source_urls=tuple(source_urls),
        source_checked_date=checked,
        annual_contribution_limit_krw=amounts["annual_contribution_limit_krw"],
        lifetime_contribution_limit_krw=amounts["lifetime_contribution_limit_krw"],
        carryover_years_cap=carryover,
        minimum_contract_years=minimum_years,
        general_allowance_krw=amounts["general_allowance_krw"],
        preferential_allowance_krw=amounts["preferential_allowance_krw"],
        preferential_wage_threshold_krw=amounts["preferential_wage_threshold_krw"],
        preferential_comprehensive_income_threshold_krw=amounts["preferential_comprehensive_income_threshold_krw"],
        separate_tax_rate=rates["separate_tax_rate"],
        local_surcharge_rate=rates["local_surcharge_rate"],
        pension_transfer_credit_rate=rates["pension_transfer_credit_rate"],
        pension_transfer_credit_cap_krw=amounts["pension_transfer_credit_cap_krw"],
    )


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer amount of KRW")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return value


def isa_contribution_room_krw(regime: IsaTaxRegime, *, elapsed_full_years: int, cumulative_contributed_krw: int) -> int:
    """Remaining contribution room of one account on a given day.

    Unused annual room carries forward, but the carry stops growing once
    ``carryover_years_cap`` full years have elapsed, and the lifetime cap always binds.
    Reinvested interest and dividends never consume room (시행령 제93조의4 ⑫), so the
    caller passes principal contributions only.

    Raises: ValueError if either argument is negative or not an integer.
    """
    elapsed = _require_nonnegative_int(elapsed_full_years, "elapsed_full_years")
    cumulative = _require_nonnegative_int(cumulative_contributed_krw, "cumulative_contributed_krw")
    grown = regime.annual_contribution_limit_krw * (1 + min(elapsed, regime.carryover_years_cap))
    annual_room = grown - cumulative
    lifetime_room = regime.lifetime_contribution_limit_krw - cumulative
    return max(0, min(annual_room, lifetime_room))


def classify_isa_tax_class(
    regime: IsaTaxRegime, *, income_kind: Literal["wage", "comprehensive"], prior_year_income_krw: int
) -> IsaTaxClass:
    """Allowance class for an account opened or extended in a year, from the prior tax year's income.

    The preferential class requires actual prior-year income of the stated kind at or below
    its threshold; a prior year with zero income yields the general class because the
    preferential test is defined over earned or comprehensive income that must exist.

    Raises: ValueError on an unknown income kind or a negative/non-integer income.
    """
    if income_kind not in ("wage", "comprehensive"):
        raise ValueError(f"income_kind {income_kind!r} is unsupported")
    income = _require_nonnegative_int(prior_year_income_krw, "prior_year_income_krw")
    if income == 0:
        return IsaTaxClass.GENERAL
    threshold = (
        regime.preferential_wage_threshold_krw
        if income_kind == "wage"
        else regime.preferential_comprehensive_income_threshold_krw
    )
    if income <= threshold:
        return IsaTaxClass.PREFERENTIAL
    return IsaTaxClass.GENERAL


def quote_isa_closure(
    regime: IsaTaxRegime,
    *,
    tax_class: IsaTaxClass,
    balance_krw: int,
    principal_krw: int,
    account_years: int,
) -> IsaClosureQuote:
    """Quote the withholding due when the account is terminated.

    Valid only for holdings whose every gain and distribution is dividend income
    (domestic-listed funds tracking overseas equity), so account net income equals the
    closing balance minus contributed principal with fees already inside the balance and
    losses netted inside the account. Income up to the class allowance is exempt; the
    excess bears the separate rate plus the local surcharge, each floored to whole won.

    Raises: ValueError if amounts are negative or non-integer, or ``account_years`` is below
        ``minimum_contract_years`` (early termination and its clawback are unsupported).
    """
    if not isinstance(tax_class, IsaTaxClass):
        raise ValueError(f"unknown ISA tax class {tax_class!r}")
    balance = _require_nonnegative_int(balance_krw, "balance_krw")
    principal = _require_nonnegative_int(principal_krw, "principal_krw")
    if isinstance(account_years, bool) or not isinstance(account_years, int):
        raise ValueError("account_years must be an integer number of years")
    if account_years < regime.minimum_contract_years:
        raise ValueError(
            f"account_years {account_years!r} is below the minimum contract term {regime.minimum_contract_years!r}"
        )
    allowance = (
        regime.preferential_allowance_krw if tax_class is IsaTaxClass.PREFERENTIAL else regime.general_allowance_krw
    )
    net_income = balance - principal
    taxable = max(0, net_income - allowance)
    national = _won(taxable, regime.separate_tax_rate)
    local = _won(national, regime.local_surcharge_rate)
    return IsaClosureQuote(
        balance_krw=balance,
        principal_krw=principal,
        net_income_krw=net_income,
        allowance_krw=allowance,
        taxable_krw=taxable,
        national_tax_krw=national,
        local_tax_krw=local,
        net_proceeds_krw=balance - national - local,
    )


def credit_maximizing_transfer_krw(regime: IsaTaxRegime) -> int:
    """Smallest transfer whose credit base reaches the transfer-credit cap (cap ÷ rate, rounded up)."""
    cap = regime.pension_transfer_credit_cap_krw
    if cap <= 0:
        return 0
    rate = regime.pension_transfer_credit_rate
    if rate <= 0:
        raise ValueError("pension transfer credit rate must be positive to reach the cap")
    ratio = Fraction(str(rate))
    return (cap * ratio.denominator + ratio.numerator - 1) // ratio.numerator


def _profile_capacity(mapping: Mapping[int, int], year: int, name: str) -> int:
    if year not in mapping:
        raise ValueError(f"{name} has no entry for tax year {year}; no income is inferred")
    value = mapping[year]
    return _require_nonnegative_int(value, name)


def quote_isa_transfer_credit(
    tax_year: int,
    transfer_krw: int,
    *,
    regular: PensionCreditQuote,
    profile: PensionTaxProfile,
    pension_regime: PensionTaxRegime,
    isa_regime: IsaTaxRegime,
) -> IsaTransferCreditQuote:
    """Quote the transfer credit on top of the same year's ordinary pension credit.

    The transfer never counts toward the annual pension contribution limit. Its credit base
    is the smaller of ``transfer x pension_transfer_credit_rate`` (floored to won) and the
    cap, credited at the same income-dependent rate as ordinary contributions, and is paid
    only out of national and local tax capacity left after ``regular`` consumed its share.
    Principal not covered by a usable credit stays uncredited and later leaves the pension
    tax-free.

    Raises: ValueError if ``regular.tax_year`` differs from ``tax_year``, the transfer is
        negative or non-integer, or the profile lacks income or capacity for ``tax_year``.
    """
    if isinstance(tax_year, bool) or not isinstance(tax_year, int):
        raise ValueError(f"tax_year must be an integer year, got {tax_year!r}")
    transfer = _require_nonnegative_int(transfer_krw, "transfer_krw")
    if regular.tax_year != tax_year:
        raise ValueError(f"regular.tax_year {regular.tax_year!r} differs from tax_year {tax_year!r}")
    if profile.income_kind not in ("wage", "comprehensive"):
        raise ValueError(f"income_kind {profile.income_kind!r} is unsupported")
    income = _profile_capacity(profile.annual_income_krw, tax_year, "annual_income_krw")
    national_capacity = _profile_capacity(profile.remaining_national_tax_krw, tax_year, "remaining_national_tax_krw")
    local_capacity = _profile_capacity(profile.remaining_local_tax_krw, tax_year, "remaining_local_tax_krw")
    eligible = min(_won(transfer, isa_regime.pension_transfer_credit_rate), isa_regime.pension_transfer_credit_cap_krw)
    threshold = (
        pension_regime.wage_threshold_krw
        if profile.income_kind == "wage"
        else pension_regime.comprehensive_income_threshold_krw
    )
    rate = pension_regime.low_income_credit_rate if income <= threshold else pension_regime.high_income_credit_rate
    theoretical_national = _won(eligible, rate)
    theoretical_local = _won(theoretical_national, pension_regime.local_surcharge_rate)
    remaining_national = max(0, national_capacity - regular.national_credit_krw)
    remaining_local = max(0, local_capacity - regular.local_credit_krw)
    national_credit = min(theoretical_national, remaining_national)
    local_credit = min(theoretical_local, remaining_local)
    if theoretical_national == 0:
        credited = 0
    else:
        national_share = Fraction(national_credit, theoretical_national)
        local_share = Fraction(local_credit, theoretical_local) if theoretical_local else Fraction(0)
        share = max(national_share, local_share)
        scaled = eligible * share
        credited = (scaled.numerator + scaled.denominator - 1) // scaled.denominator
    return IsaTransferCreditQuote(
        tax_year=tax_year,
        transfer_krw=transfer,
        eligible_krw=eligible,
        national_credit_krw=national_credit,
        local_credit_krw=local_credit,
        credited_principal_krw=credited,
        uncredited_principal_krw=transfer - credited,
    )
