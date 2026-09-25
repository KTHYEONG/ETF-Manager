"""Korean personal-pension savings tax ledger: frozen 2026-law credit and withdrawal quotes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Final, Literal

__all__ = [
    "PensionCreditQuote",
    "PensionExitKind",
    "PensionExitQuote",
    "PensionTaxProfile",
    "PensionTaxRegime",
    "PensionWithdrawalQuote",
    "load_pension_tax_regime",
    "quote_annual_pension_credit",
    "quote_annual_pension_withdrawal",
    "quote_pension_exit",
    "years_to_draw_at_threshold",
]

_EXPECTED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "regime_id",
        "policy_year",
        "source_urls",
        "annual_contribution_limit_krw",
        "annual_credit_limit_krw",
        "wage_threshold_krw",
        "comprehensive_income_threshold_krw",
        "low_income_credit_rate",
        "high_income_credit_rate",
        "local_surcharge_rate",
        "minimum_pension_age",
        "minimum_account_years",
        "private_pension_threshold_krw",
        "age_withholding_bands",
        "above_threshold_separate_rate",
        "non_pension_rate",
        "foreign_dividend_withholding_rate",
        "foreign_tax_credit_rate",
        "pension_limit_multiplier",
        "pension_limit_final_year",
    }
)
_AMOUNT_KEYS: Final[tuple[str, ...]] = (
    "annual_contribution_limit_krw",
    "annual_credit_limit_krw",
    "wage_threshold_krw",
    "comprehensive_income_threshold_krw",
    "private_pension_threshold_krw",
)
_RATE_KEYS: Final[tuple[str, ...]] = (
    "low_income_credit_rate",
    "high_income_credit_rate",
    "local_surcharge_rate",
    "above_threshold_separate_rate",
    "non_pension_rate",
    "foreign_dividend_withholding_rate",
    "foreign_tax_credit_rate",
)
_SUPPORTED_POLICY_YEAR: Final[int] = 2026


@dataclass(frozen=True, slots=True)
class PensionTaxRegime:
    """Validated 2026-law parameters for the supported personal-pension scenario.

    Foreign-dividend withholding is lost at source inside the account; the credit rate
    converts accumulated withheld tax into an exit-tax credit (0 disables it).
    """

    regime_id: str
    policy_year: int
    annual_contribution_limit_krw: int
    annual_credit_limit_krw: int
    wage_threshold_krw: int
    comprehensive_income_threshold_krw: int
    low_income_credit_rate: float
    high_income_credit_rate: float
    local_surcharge_rate: float
    minimum_pension_age: int
    minimum_account_years: int
    private_pension_threshold_krw: int
    age_withholding_bands: tuple[tuple[int, float], ...]
    above_threshold_separate_rate: float
    non_pension_rate: float
    foreign_dividend_withholding_rate: float
    """Withholding lost at source on foreign dividends held inside the account."""
    foreign_tax_credit_rate: float
    """Share of accumulated withheld foreign tax creditable against the exit tax (0 disables)."""
    pension_limit_multiplier: float
    pension_limit_final_year: int
    source_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PensionTaxProfile:
    """Explicit age, income, and residual tax-capacity scenario; no income is inferred."""

    profile_id: str
    birth_date: date
    account_open_date: date
    income_kind: Literal["wage", "comprehensive"]
    annual_income_krw: Mapping[int, int]
    remaining_national_tax_krw: Mapping[int, int]
    remaining_local_tax_krw: Mapping[int, int]
    other_private_pension_income_krw: Mapping[int, int]
    pension_start_date: date | None = None


@dataclass(frozen=True, slots=True)
class PensionCreditQuote:
    """Annual credit and principal-basis classification after separate national/local capacity limits."""

    tax_year: int
    contributed_krw: int
    theoretical_national_credit_krw: int
    theoretical_local_credit_krw: int
    credited_principal_krw: int
    uncredited_principal_krw: int
    national_credit_krw: int
    local_credit_krw: int


@dataclass(frozen=True, slots=True)
class PensionWithdrawalQuote:
    """Legal pension receipt with tax-free/taxable amounts and remaining principal bases."""

    withdrawal_date: date
    gross_withdrawal_krw: int
    tax_free_krw: int
    taxable_krw: int
    national_tax_krw: int
    local_tax_krw: int
    remaining_uncredited_principal_krw: int
    remaining_credited_principal_krw: int


def _require_krw(document: dict[str, object], key: str) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"pension regime field {key!r} must be an integer amount of KRW")
    if value <= 0:
        raise ValueError(f"pension regime field {key!r} must be positive, got {value!r}")
    return value


def _require_rate(document: dict[str, object], key: str) -> float:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"pension regime field {key!r} must be a finite rate")
    rate = float(value)
    if rate < 0 or rate >= 1:
        raise ValueError(f"pension regime field {key!r} must lie in [0, 1), got {value!r}")
    return rate


def _require_bands(document: dict[str, object]) -> tuple[tuple[int, float], ...]:
    raw = document["age_withholding_bands"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("pension regime field 'age_withholding_bands' must be a non-empty array")
    bands: list[tuple[int, float]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError(f"pension regime age band {item!r} must be an [age, rate] pair")
        age, rate = item
        if isinstance(age, bool) or not isinstance(age, int) or age < 0:
            raise ValueError(f"pension regime age band {item!r} must start at a nonnegative integer age")
        if isinstance(rate, bool) or not isinstance(rate, int | float) or not math.isfinite(rate):
            raise ValueError(f"pension regime age band {item!r} must carry a finite rate")
        rate_value = float(rate)
        if rate_value < 0 or rate_value >= 1:
            raise ValueError(f"pension regime age band {item!r} must carry a rate in [0, 1)")
        bands.append((age, rate_value))
    ages = [age for age, _ in bands]
    if ages[0] != 0:
        raise ValueError("pension regime age bands must start at age 0")
    if any(later <= earlier for earlier, later in pairwise(ages)):
        raise ValueError(f"pension regime age bands must be strictly increasing, got {ages!r}")
    return tuple(bands)


def load_pension_tax_regime(path: str | Path) -> PensionTaxRegime:
    """Load a source-backed, frozen pension-only tax scenario.

    Returns: Validated policy parameters.
    Raises: ValueError for malformed or internally inconsistent policy data.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"pension regime {str(path)!r} must be a JSON object")
    keys = frozenset(document.keys())
    if keys != _EXPECTED_KEYS:
        raise ValueError(
            f"pension regime {str(path)!r} has unexpected keys: missing={sorted(_EXPECTED_KEYS - keys)} "
            f"extra={sorted(keys - _EXPECTED_KEYS)}"
        )
    regime_id = document["regime_id"]
    if not isinstance(regime_id, str) or not regime_id:
        raise ValueError("pension regime field 'regime_id' must be a nonempty string")
    policy_year = document["policy_year"]
    if isinstance(policy_year, bool) or not isinstance(policy_year, int):
        raise ValueError("pension regime field 'policy_year' must be an integer year")
    if policy_year != _SUPPORTED_POLICY_YEAR:
        raise ValueError(f"pension regime field 'policy_year' is unknown: {policy_year!r}")
    source_urls = document["source_urls"]
    if (
        not isinstance(source_urls, list)
        or not source_urls
        or any(not isinstance(url, str) or not url.startswith("http") for url in source_urls)
    ):
        raise ValueError("pension regime field 'source_urls' must be a non-empty array of http(s) URLs")
    amounts = {key: _require_krw(document, key) for key in _AMOUNT_KEYS}
    rates = {key: _require_rate(document, key) for key in _RATE_KEYS}
    if rates["low_income_credit_rate"] < rates["high_income_credit_rate"]:
        raise ValueError("pension regime credit rates are inconsistent: low-income rate must cover high-income rate")
    if amounts["annual_credit_limit_krw"] > amounts["annual_contribution_limit_krw"]:
        raise ValueError("pension regime thresholds are inconsistent: credit cap must not exceed contribution cap")
    minimum_age = _require_krw(document, "minimum_pension_age")
    minimum_years = _require_krw(document, "minimum_account_years")
    final_year = _require_krw(document, "pension_limit_final_year")
    multiplier = document["pension_limit_multiplier"]
    if isinstance(multiplier, bool) or not isinstance(multiplier, int | float) or not math.isfinite(multiplier):
        raise ValueError("pension regime field 'pension_limit_multiplier' must be a finite number")
    if float(multiplier) < 1:
        raise ValueError(f"pension regime field 'pension_limit_multiplier' must cover 1.0, got {multiplier!r}")
    return PensionTaxRegime(
        regime_id=regime_id,
        policy_year=policy_year,
        annual_contribution_limit_krw=amounts["annual_contribution_limit_krw"],
        annual_credit_limit_krw=amounts["annual_credit_limit_krw"],
        wage_threshold_krw=amounts["wage_threshold_krw"],
        comprehensive_income_threshold_krw=amounts["comprehensive_income_threshold_krw"],
        low_income_credit_rate=rates["low_income_credit_rate"],
        high_income_credit_rate=rates["high_income_credit_rate"],
        local_surcharge_rate=rates["local_surcharge_rate"],
        minimum_pension_age=minimum_age,
        minimum_account_years=minimum_years,
        private_pension_threshold_krw=amounts["private_pension_threshold_krw"],
        age_withholding_bands=_require_bands(document),
        above_threshold_separate_rate=rates["above_threshold_separate_rate"],
        non_pension_rate=rates["non_pension_rate"],
        foreign_dividend_withholding_rate=rates["foreign_dividend_withholding_rate"],
        foreign_tax_credit_rate=rates["foreign_tax_credit_rate"],
        pension_limit_multiplier=float(multiplier),
        pension_limit_final_year=final_year,
        source_urls=tuple(source_urls),
    )


def _won(amount_krw: int, rate: float) -> int:
    """Floor fractional won of ``amount_krw * rate`` using exact decimal arithmetic."""
    quantized = Decimal(amount_krw) * Decimal(str(rate))
    return int(quantized.to_integral_value(rounding="ROUND_FLOOR"))


def _require_quote_amount(value: object, name: str, *, allow_zero: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer amount of KRW")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _capacity(mapping: Mapping[int, int], year: int, name: str) -> int:
    if year not in mapping:
        raise ValueError(f"{name} has no entry for tax year {year}; no income is inferred")
    return _require_quote_amount(mapping[year], name, allow_zero=True)


def quote_annual_pension_credit(
    tax_year: int,
    contributed_krw: int,
    profile: PensionTaxProfile,
    regime: PensionTaxRegime,
) -> PensionCreditQuote:
    """Quote the usable annual credit against actual remaining tax capacity.

    Args: tax_year and contributed_krw are calendar-year totals; profile contains
        only income and tax capacity known for that scenario.
    Returns: National credit, local credit, credited principal, and uncredited principal.
    Raises: ValueError if the year, income kind, tax capacity, or amount is invalid.
    """
    if isinstance(tax_year, bool) or not isinstance(tax_year, int):
        raise ValueError(f"tax_year must be an integer year, got {tax_year!r}")
    contributed = _require_quote_amount(contributed_krw, "contributed_krw", allow_zero=True)
    if contributed > regime.annual_contribution_limit_krw:
        raise ValueError(
            f"contributed_krw {contributed!r} exceeds the annual contribution limit "
            f"{regime.annual_contribution_limit_krw!r}"
        )
    if profile.income_kind not in ("wage", "comprehensive"):
        raise ValueError(f"income_kind {profile.income_kind!r} is unsupported; only pension savings apply")
    income = _capacity(profile.annual_income_krw, tax_year, "annual_income_krw")
    national_capacity = _capacity(profile.remaining_national_tax_krw, tax_year, "remaining_national_tax_krw")
    local_capacity = _capacity(profile.remaining_local_tax_krw, tax_year, "remaining_local_tax_krw")
    threshold = regime.wage_threshold_krw if profile.income_kind == "wage" else regime.comprehensive_income_threshold_krw
    rate = regime.low_income_credit_rate if income <= threshold else regime.high_income_credit_rate
    eligible = min(contributed, regime.annual_credit_limit_krw)
    theoretical_national = _won(eligible, rate)
    theoretical_local = _won(theoretical_national, regime.local_surcharge_rate)
    national_credit = min(theoretical_national, national_capacity)
    local_credit = min(theoretical_local, local_capacity)
    if theoretical_national == 0:
        credited = 0
    else:
        national_share = Fraction(national_credit, theoretical_national)
        local_share = Fraction(local_credit, theoretical_local) if theoretical_local else Fraction(0)
        share = max(national_share, local_share)
        scaled = eligible * share
        credited = (scaled.numerator + scaled.denominator - 1) // scaled.denominator
    return PensionCreditQuote(
        tax_year=tax_year,
        contributed_krw=contributed,
        theoretical_national_credit_krw=theoretical_national,
        theoretical_local_credit_krw=theoretical_local,
        credited_principal_krw=credited,
        uncredited_principal_krw=contributed - credited,
        national_credit_krw=national_credit,
        local_credit_krw=local_credit,
    )


def _full_years_since(start: date, end: date) -> int:
    years = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return years


def _age_rate(regime: PensionTaxRegime, age: int) -> float:
    rate = regime.age_withholding_bands[0][1]
    for min_age, band_rate in regime.age_withholding_bands:
        if age >= min_age:
            rate = band_rate
    return rate


def quote_annual_pension_withdrawal(
    withdrawal_date: date,
    gross_withdrawal_krw: int,
    opening_account_value_krw: int,
    prior_same_year_withdrawals_krw: int,
    uncredited_principal_krw: int,
    credited_principal_krw: int,
    profile: PensionTaxProfile,
    regime: PensionTaxRegime,
) -> PensionWithdrawalQuote:
    """Quote an ordinary eligible pension withdrawal, preserving the tax-basis order.

    Returns: Tax-free amount, taxable amount, national/local tax, and remaining tax bases.
    Raises: ValueError when age, account age, annual pension limit, capacity, or
        private-pension election cannot be resolved under the supported policy.
    """
    gross = _require_quote_amount(gross_withdrawal_krw, "gross_withdrawal_krw", allow_zero=False)
    opening = _require_quote_amount(opening_account_value_krw, "opening_account_value_krw", allow_zero=True)
    prior = _require_quote_amount(prior_same_year_withdrawals_krw, "prior_same_year_withdrawals_krw", allow_zero=True)
    uncredited = _require_quote_amount(uncredited_principal_krw, "uncredited_principal_krw", allow_zero=True)
    credited = _require_quote_amount(credited_principal_krw, "credited_principal_krw", allow_zero=True)
    age = _full_years_since(profile.birth_date, withdrawal_date)
    if age < regime.minimum_pension_age:
        raise ValueError(f"pension receipt requires age {regime.minimum_pension_age}, got {age}")
    account_years = _full_years_since(profile.account_open_date, withdrawal_date)
    if account_years < regime.minimum_account_years:
        raise ValueError(f"pension receipt requires {regime.minimum_account_years} account years, got {account_years}")
    start = profile.pension_start_date
    if start is None or start > withdrawal_date:
        raise ValueError("pension receipt requires an effective pension commencement application")
    if _full_years_since(profile.birth_date, start) < regime.minimum_pension_age or _full_years_since(
        profile.account_open_date, start
    ) < regime.minimum_account_years:
        raise ValueError("pension commencement date precedes age or account-tenure eligibility")
    first_eligible_year = max(
        profile.birth_date.year + regime.minimum_pension_age,
        profile.account_open_date.year + regime.minimum_account_years,
    )
    initial_year = 6 if profile.account_open_date < date(2013, 3, 1) else 1
    pension_year = initial_year + withdrawal_date.year - first_eligible_year
    if pension_year <= regime.pension_limit_final_year:
        divisor = regime.pension_limit_final_year + 1 - pension_year
        annual_limit = int(
            (Decimal(opening) * Decimal(str(regime.pension_limit_multiplier)) / Decimal(divisor)).to_integral_value(
                rounding="ROUND_FLOOR"
            )
        )
        if gross + prior > annual_limit:
            raise ValueError(
                f"withdrawal {gross + prior!r} exceeds pension-year-{pension_year} annual limit {annual_limit!r}"
            )
    tax_free = min(gross, uncredited)
    taxable = gross - tax_free
    other_income = _capacity(profile.other_private_pension_income_krw, withdrawal_date.year, "other_private_pension_income_krw")
    aggregate = taxable + other_income
    if aggregate <= regime.private_pension_threshold_krw:
        rate = _age_rate(regime, age)
    else:
        rate = regime.above_threshold_separate_rate
    national_tax = _won(taxable, rate)
    local_tax = _won(national_tax, regime.local_surcharge_rate)
    return PensionWithdrawalQuote(
        withdrawal_date=withdrawal_date,
        gross_withdrawal_krw=gross,
        tax_free_krw=tax_free,
        taxable_krw=taxable,
        national_tax_krw=national_tax,
        local_tax_krw=local_tax,
        remaining_uncredited_principal_krw=uncredited - tax_free,
        remaining_credited_principal_krw=credited - min(taxable, credited),
    )


class PensionExitKind(StrEnum):
    """How the account balance leaves the tax wrapper at the valuation date."""

    LUMP_SUM = "lump_sum"
    ANNUITY_LOW = "annuity_low"
    ANNUITY_HIGH = "annuity_high"


@dataclass(frozen=True, slots=True)
class PensionExitQuote:
    """After-tax value of the whole balance under one exit convention."""

    kind: PensionExitKind
    balance_krw: int
    tax_free_krw: int
    taxable_krw: int
    national_tax_krw: int
    local_tax_krw: int
    foreign_tax_credit_krw: int
    net_krw: int


def quote_pension_exit(
    kind: PensionExitKind,
    *,
    valuation_date: date,
    balance_krw: int,
    uncredited_principal_krw: int,
    foreign_tax_withheld_krw: int,
    profile: PensionTaxProfile,
    regime: PensionTaxRegime,
) -> PensionExitQuote:
    """Value the entire balance as if it left the pension wrapper under ``kind``.

    Uncredited principal leaves first and tax-free; the remainder (credited principal and
    all gains) is taxable. LUMP_SUM applies the non-pension (other-income) rate, which also
    claws back the contribution credit. ANNUITY_LOW applies the age-band rate at the later of
    the valuation age and the minimum pension age, i.e. it assumes the holder defers to
    eligibility and draws within the private-pension threshold with no further return;
    ANNUITY_HIGH applies the above-threshold separate rate. The foreign-tax credit is
    ``foreign_tax_credit_rate`` times the withheld foreign tax, floored to won, and never
    exceeds the exit tax.

    Raises:
        ValueError: On negative amounts or an unknown kind.
    """
    if not isinstance(kind, PensionExitKind):
        raise ValueError(f"unknown pension exit kind {kind!r}")
    balance = _require_quote_amount(balance_krw, "balance_krw", allow_zero=True)
    uncredited = _require_quote_amount(uncredited_principal_krw, "uncredited_principal_krw", allow_zero=True)
    withheld = _require_quote_amount(foreign_tax_withheld_krw, "foreign_tax_withheld_krw", allow_zero=True)
    tax_free = min(balance, uncredited)
    taxable = balance - tax_free
    if kind is PensionExitKind.LUMP_SUM:
        rate = regime.non_pension_rate
    elif kind is PensionExitKind.ANNUITY_LOW:
        age = max(_full_years_since(profile.birth_date, valuation_date), regime.minimum_pension_age)
        rate = _age_rate(regime, age)
    else:
        rate = regime.above_threshold_separate_rate
    national_tax = _won(taxable, rate)
    local_tax = _won(national_tax, regime.local_surcharge_rate)
    credit = min(national_tax + local_tax, _won(withheld, regime.foreign_tax_credit_rate))
    return PensionExitQuote(
        kind=kind,
        balance_krw=balance,
        tax_free_krw=tax_free,
        taxable_krw=taxable,
        national_tax_krw=national_tax,
        local_tax_krw=local_tax,
        foreign_tax_credit_krw=credit,
        net_krw=balance - national_tax - local_tax + credit,
    )


def years_to_draw_at_threshold(taxable_krw: int, regime: PensionTaxRegime) -> int:
    """Whole years needed to draw ``taxable_krw`` without exceeding the private-pension threshold (0 when nothing is taxable)."""
    taxable = _require_quote_amount(taxable_krw, "taxable_krw", allow_zero=True)
    if taxable == 0:
        return 0
    threshold = regime.private_pension_threshold_krw
    return (taxable + threshold - 1) // threshold
