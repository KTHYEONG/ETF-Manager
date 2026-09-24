"""Invariant guards for pension credit quotes and withdrawal settlement."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.sim.pension_tax import (
    PensionTaxProfile,
    PensionTaxRegime,
    load_pension_tax_regime,
    quote_annual_pension_credit,
    quote_annual_pension_withdrawal,
)

_SHIPPED = Path("configs/tax/kr_pension_2026.json")


def _regime() -> PensionTaxRegime:
    return load_pension_tax_regime(_SHIPPED)


def _profile(
    *,
    birth: date = date(1990, 1, 1),
    opened: date = date(2015, 1, 1),
    kind: str = "wage",
    year: int = 2024,
    income: int = 50_000_000,
    national: int = 10_000_000,
    local: int = 10_000_000,
    other: int = 0,
    pension_start: date | None = date(2025, 6, 1),
) -> PensionTaxProfile:
    return PensionTaxProfile(
        profile_id="test",
        birth_date=birth,
        account_open_date=opened,
        income_kind=kind,  # type: ignore[arg-type]
        annual_income_krw={year: income},
        remaining_national_tax_krw={year: national},
        remaining_local_tax_krw={year: local},
        other_private_pension_income_krw={year: other},
        pension_start_date=pension_start,
    )


def test_student_with_no_tax_keeps_uncredited_basis() -> None:
    """Zero capacity yields zero credit and the contribution stays uncredited."""
    quote = quote_annual_pension_credit(2024, 6_000_000, _profile(national=0, local=0), _regime())
    assert quote.national_credit_krw == 0
    assert quote.local_credit_krw == 0
    assert (quote.theoretical_national_credit_krw, quote.theoretical_local_credit_krw) == (900_000, 90_000)
    assert quote.credited_principal_krw == 0
    assert quote.uncredited_principal_krw == 6_000_000


def test_partial_capacity_caps_second_contribution() -> None:
    """495k of combined capacity stops helping once the 3m credit is covered."""
    profile = _profile(national=450_000, local=45_000)
    first = quote_annual_pension_credit(2024, 3_000_000, profile, _regime())
    assert (first.national_credit_krw, first.local_credit_krw) == (450_000, 45_000)
    assert first.credited_principal_krw == 3_000_000
    second = quote_annual_pension_credit(2024, 6_000_000, profile, _regime())
    assert (second.national_credit_krw, second.local_credit_krw) == (450_000, 45_000)
    assert second.credited_principal_krw == 3_000_000
    assert second.uncredited_principal_krw == 3_000_000


def test_contribution_cap_boundary() -> None:
    """Credit stops at 6m; contributions above 18m are rejected."""
    regime = _regime()
    profile = _profile(national=10_000_000, local=10_000_000)
    at_cap = quote_annual_pension_credit(2024, 6_000_000, profile, regime)
    assert at_cap.national_credit_krw == 900_000
    assert at_cap.credited_principal_krw == 6_000_000
    assert at_cap.uncredited_principal_krw == 0
    over_cap = quote_annual_pension_credit(2024, 6_000_001, profile, regime)
    assert over_cap.national_credit_krw == 900_000
    assert over_cap.uncredited_principal_krw == 1
    full = quote_annual_pension_credit(2024, 18_000_000, profile, regime)
    assert full.credited_principal_krw + full.uncredited_principal_krw == 18_000_000
    with pytest.raises(ValueError, match="exceeds the annual contribution limit"):
        quote_annual_pension_credit(2024, 18_000_001, profile, regime)


def test_high_income_rate_and_split_capacity() -> None:
    """High earners use 12%; principal credited under either capacity stays credited."""
    regime = _regime()
    profile = _profile(income=60_000_000, national=10_000_000, local=0)
    quote = quote_annual_pension_credit(2024, 6_000_000, profile, regime)
    assert quote.national_credit_krw == 720_000
    assert quote.local_credit_krw == 0
    assert quote.credited_principal_krw == 6_000_000
    tiny = quote_annual_pension_credit(2024, 60, _profile(national=10, local=10), regime)
    assert tiny.national_credit_krw == 9
    assert tiny.local_credit_krw == 0
    assert tiny.credited_principal_krw == 60
    empty = quote_annual_pension_credit(2024, 0, _profile(), regime)
    assert empty.credited_principal_krw == 0
    assert empty.national_credit_krw == 0


def test_withdrawal_consumes_uncredited_first() -> None:
    """Mixed bases plus gains reconcile with uncredited principal consumed first."""
    quote = quote_annual_pension_withdrawal(
        date(2025, 6, 1),
        3_000_000,
        100_000_000,
        0,
        2_000_000,
        5_000_000,
        _profile(birth=date(1960, 1, 1), opened=date(2010, 1, 1), year=2025),
        _regime(),
    )
    assert quote.tax_free_krw == 2_000_000
    assert quote.taxable_krw == 1_000_000
    assert quote.tax_free_krw + quote.taxable_krw == quote.gross_withdrawal_krw
    assert quote.national_tax_krw == 50_000
    assert quote.local_tax_krw == 5_000
    assert quote.remaining_uncredited_principal_krw == 0
    assert quote.remaining_credited_principal_krw == 4_000_000
    gains = quote_annual_pension_withdrawal(
        date(2025, 6, 1),
        10_000_000,
        100_000_000,
        0,
        2_000_000,
        5_000_000,
        _profile(birth=date(1960, 1, 1), opened=date(2010, 1, 1), year=2025),
        _regime(),
    )
    assert gains.tax_free_krw == 2_000_000
    assert gains.taxable_krw == 8_000_000
    assert gains.remaining_uncredited_principal_krw == 0
    assert gains.remaining_credited_principal_krw == 0
    assert gains.national_tax_krw >= 0


def test_pension_eligibility_gates() -> None:
    """Only a fully eligible in-limit withdrawal receives pension treatment."""
    regime = _regime()
    underage = _profile(birth=date(1970, 6, 1), opened=date(2010, 1, 1), year=2025)
    with pytest.raises(ValueError, match="requires age 55"):
        quote_annual_pension_withdrawal(date(2025, 1, 1), 1_000_000, 100_000_000, 0, 0, 1_000_000, underage, regime)
    young_account = _profile(birth=date(1960, 1, 1), opened=date(2021, 6, 1), year=2025)
    with pytest.raises(ValueError, match="account years"):
        quote_annual_pension_withdrawal(date(2025, 6, 1), 1_000_000, 100_000_000, 0, 0, 1_000_000, young_account, regime)
    first_year = _profile(birth=date(1960, 1, 1), opened=date(2020, 6, 1), year=2025)
    with pytest.raises(ValueError, match="annual limit"):
        quote_annual_pension_withdrawal(date(2025, 6, 1), 13_000_000, 100_000_000, 0, 0, 13_000_000, first_year, regime)
    within = quote_annual_pension_withdrawal(
        date(2025, 6, 1), 12_000_000, 100_000_000, 0, 0, 12_000_000, first_year, regime
    )
    assert within.taxable_krw == 12_000_000
    mature = _profile(birth=date(1960, 1, 1), opened=date(2005, 1, 1), year=2025)
    late = quote_annual_pension_withdrawal(
        date(2025, 6, 1), 50_000_000, 100_000_000, 0, 0, 50_000_000, mature, regime
    )
    assert late.taxable_krw == 50_000_000


def test_commencement_application_and_receipt_year_boundaries() -> None:
    """The receipt year starts at first legal eligibility, with the pre-2013 year-six exception."""
    regime = _regime()
    current = _profile(birth=date(1960, 1, 1), opened=date(2018, 1, 1), year=2025)
    with pytest.raises(ValueError, match="commencement application"):
        quote_annual_pension_withdrawal(
            date(2025, 6, 1), 1, 100_000_000, 0, 0, 1, _profile(
                birth=date(1960, 1, 1), opened=date(2018, 1, 1), year=2025, pension_start=None,
            ), regime,
        )
    with pytest.raises(ValueError, match="commencement date precedes"):
        quote_annual_pension_withdrawal(
            date(2025, 6, 1), 1, 100_000_000, 0, 0, 1, _profile(
                birth=date(1960, 1, 1), opened=date(2018, 1, 1), year=2025,
                pension_start=date(2022, 12, 31),
            ), regime,
        )
    assert quote_annual_pension_withdrawal(
        date(2025, 6, 1), 15_000_000, 100_000_000, 0, 0, 15_000_000, current, regime,
    ).gross_withdrawal_krw == 15_000_000
    with pytest.raises(ValueError, match="pension-year-3 annual limit 15000000"):
        quote_annual_pension_withdrawal(
            date(2025, 6, 1), 15_000_001, 100_000_000, 0, 0, 15_000_001, current, regime,
        )
    old = _profile(
        birth=date(1960, 1, 1), opened=date(2010, 1, 1), year=2015,
        pension_start=date(2015, 6, 1),
    )
    with pytest.raises(ValueError, match="pension-year-6 annual limit 24000000"):
        quote_annual_pension_withdrawal(
            date(2015, 6, 1), 24_000_001, 100_000_000, 0, 0, 24_000_001, old, regime,
        )


def test_private_pension_threshold_branches() -> None:
    """Below, at, and above 15m select the age rate or the separate-tax election."""
    regime = _regime()
    base = {"birth": date(1965, 1, 1), "opened": date(2010, 1, 1), "year": 2025}
    below = quote_annual_pension_withdrawal(
        date(2025, 6, 1), 2_000_000, 100_000_000, 0, 0, 2_000_000, _profile(other=12_999_999, **base), regime
    )
    assert below.national_tax_krw == 100_000
    assert below.local_tax_krw == 10_000
    at_edge = quote_annual_pension_withdrawal(
        date(2025, 6, 1), 2_000_000, 100_000_000, 0, 0, 2_000_000, _profile(other=13_000_000, **base), regime
    )
    assert at_edge.national_tax_krw == 100_000
    above = quote_annual_pension_withdrawal(
        date(2025, 6, 1), 2_000_000, 100_000_000, 0, 0, 2_000_000, _profile(other=13_000_001, **base), regime
    )
    assert above.national_tax_krw == 300_000
    assert above.local_tax_krw == 30_000
    senior = quote_annual_pension_withdrawal(
        date(2025, 6, 1),
        2_000_000,
        100_000_000,
        0,
        0,
        2_000_000,
        _profile(birth=date(1948, 1, 1), opened=date(2010, 1, 1), year=2025),
        regime,
    )
    assert senior.national_tax_krw == 80_000
    elder = quote_annual_pension_withdrawal(
        date(2025, 6, 1),
        2_000_000,
        100_000_000,
        0,
        0,
        2_000_000,
        _profile(birth=date(1940, 1, 1), opened=date(2010, 1, 1), year=2025),
        regime,
    )
    assert elder.national_tax_krw == 60_000


def test_quote_inputs_fail_closed() -> None:
    """Invalid years, kinds, capacities, and amounts never produce a quote."""
    regime = _regime()
    profile = _profile()
    with pytest.raises(ValueError, match="integer year"):
        quote_annual_pension_credit(True, 1_000_000, profile, regime)
    with pytest.raises(ValueError, match="positive"):
        quote_annual_pension_credit(2024, -1, profile, regime)
    with pytest.raises(ValueError, match="integer amount"):
        quote_annual_pension_credit(2024, 1.5, profile, regime)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported"):
        quote_annual_pension_credit(2024, 1_000_000, _profile(kind="business"), regime)
    with pytest.raises(ValueError, match="no entry"):
        quote_annual_pension_credit(2030, 1_000_000, profile, regime)
    with pytest.raises(ValueError, match="positive"):
        quote_annual_pension_credit(2024, 1_000_000, _profile(national=-5), regime)
    with pytest.raises(ValueError, match="positive"):
        quote_annual_pension_withdrawal(date(2025, 6, 1), 0, 100_000_000, 0, 0, 0, profile, regime)
    with pytest.raises(ValueError, match="integer amount"):
        quote_annual_pension_withdrawal(date(2025, 6, 1), "x", 100_000_000, 0, 0, 0, profile, regime)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        quote_annual_pension_withdrawal(date(2025, 6, 1), 1_000_000, -100, 0, 0, 0, profile, regime)
    with pytest.raises(ValueError, match="positive"):
        quote_annual_pension_withdrawal(date(2025, 6, 1), 1_000_000, 100_000_000, -1, 0, 0, profile, regime)
    with pytest.raises(ValueError, match="positive"):
        quote_annual_pension_withdrawal(date(2025, 6, 1), 1_000_000, 100_000_000, 0, -1, 0, profile, regime)
    with pytest.raises(ValueError, match="positive"):
        quote_annual_pension_withdrawal(date(2025, 6, 1), 1_000_000, 100_000_000, 0, 0, -1, profile, regime)
    with pytest.raises(ValueError, match="no entry"):
        quote_annual_pension_withdrawal(
            date(2030, 6, 1),
            1_000_000,
            100_000_000,
            0,
            0,
            1_000_000,
            _profile(birth=date(1960, 1, 1), opened=date(2010, 1, 1), year=2025),
            regime,
        )
