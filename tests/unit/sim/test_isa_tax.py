"""Invariant guards for ISA contribution room, closure tax, and pension-transfer credit."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from dataclasses import replace

import pytest

from src.sim.isa_tax import (
    IsaTaxClass,
    IsaTaxRegime,
    classify_isa_tax_class,
    credit_maximizing_transfer_krw,
    isa_contribution_room_krw,
    load_isa_tax_regime,
    quote_isa_closure,
    quote_isa_transfer_credit,
)
from src.sim.pension_tax import PensionTaxProfile, load_pension_tax_regime, quote_annual_pension_credit

_SHIPPED = Path("configs/tax/kr_isa_2026.json")
_PENSION_SHIPPED = Path("configs/tax/kr_pension_2026.json")


def _regime() -> IsaTaxRegime:
    return load_isa_tax_regime(_SHIPPED)


def _profile(
    *,
    year: int = 2031,
    kind: str = "wage",
    income: int = 45_000_000,
    national: int = 10_000_000,
    local: int = 10_000_000,
) -> PensionTaxProfile:
    return PensionTaxProfile(
        profile_id="test",
        birth_date=date(1990, 1, 1),
        account_open_date=date(2020, 1, 1),
        income_kind=kind,  # type: ignore[arg-type]
        annual_income_krw={year: income},
        remaining_national_tax_krw={year: national},
        remaining_local_tax_krw={year: local},
        other_private_pension_income_krw={year: 0},
    )


def test_real_config_loads() -> None:
    regime = _regime()
    document = json.loads(_SHIPPED.read_text(encoding="utf-8"))
    assert regime.regime_id == "KR_ISA_2026"
    assert regime.policy_year == 2026
    assert document["annual_contribution_limit_krw"] == 20_000_000 == regime.annual_contribution_limit_krw
    assert document["lifetime_contribution_limit_krw"] == 100_000_000 == regime.lifetime_contribution_limit_krw
    assert regime.carryover_years_cap == 4
    assert regime.minimum_contract_years == 3
    assert regime.general_allowance_krw == 2_000_000
    assert regime.preferential_allowance_krw == 4_000_000
    assert regime.preferential_wage_threshold_krw == 50_000_000
    assert regime.preferential_comprehensive_income_threshold_krw == 38_000_000
    assert regime.separate_tax_rate == 0.09
    assert regime.local_surcharge_rate == 0.1
    assert regime.pension_transfer_credit_rate == 0.1
    assert regime.pension_transfer_credit_cap_krw == 3_000_000
    assert regime.source_checked_date == date(2026, 9, 26)
    assert len(regime.statute_refs) == 5
    assert len(regime.source_urls) == 4


def test_key_set_mismatch_rejected(tmp_path: Path) -> None:
    document = json.loads(_SHIPPED.read_text(encoding="utf-8"))
    extra_path = tmp_path / "extra.json"
    extra_path.write_text(json.dumps({**document, "bonus": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected keys"):
        load_isa_tax_regime(extra_path)
    missing = dict(document)
    del missing["separate_tax_rate"]
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected keys"):
        load_isa_tax_regime(missing_path)


def test_rate_out_of_range_rejected(tmp_path: Path) -> None:
    document = json.loads(_SHIPPED.read_text(encoding="utf-8"))
    document["separate_tax_rate"] = 1.0
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="lie in"):
        load_isa_tax_regime(bad_path)


def test_contribution_room_carries_forward_then_caps() -> None:
    regime = _regime()
    assert isa_contribution_room_krw(regime, elapsed_full_years=0, cumulative_contributed_krw=0) == 20_000_000
    assert isa_contribution_room_krw(regime, elapsed_full_years=2, cumulative_contributed_krw=20_000_000) == 40_000_000
    assert isa_contribution_room_krw(regime, elapsed_full_years=7, cumulative_contributed_krw=60_000_000) == 40_000_000
    assert isa_contribution_room_krw(regime, elapsed_full_years=9, cumulative_contributed_krw=100_000_000) == 0


def test_contribution_room_never_negative() -> None:
    assert isa_contribution_room_krw(_regime(), elapsed_full_years=0, cumulative_contributed_krw=30_000_000) == 0


def test_preferential_requires_positive_income() -> None:
    regime = _regime()
    assert classify_isa_tax_class(regime, income_kind="wage", prior_year_income_krw=0) is IsaTaxClass.GENERAL
    assert (
        classify_isa_tax_class(regime, income_kind="wage", prior_year_income_krw=5_000_000) is IsaTaxClass.PREFERENTIAL
    )
    assert (
        classify_isa_tax_class(regime, income_kind="wage", prior_year_income_krw=50_000_000) is IsaTaxClass.PREFERENTIAL
    )
    assert classify_isa_tax_class(regime, income_kind="wage", prior_year_income_krw=50_000_001) is IsaTaxClass.GENERAL


def test_comprehensive_threshold() -> None:
    regime = _regime()
    assert (
        classify_isa_tax_class(regime, income_kind="comprehensive", prior_year_income_krw=38_000_000)
        is IsaTaxClass.PREFERENTIAL
    )
    assert (
        classify_isa_tax_class(regime, income_kind="comprehensive", prior_year_income_krw=38_000_001)
        is IsaTaxClass.GENERAL
    )


def test_loss_closes_tax_free() -> None:
    quote = quote_isa_closure(
        _regime(), tax_class=IsaTaxClass.GENERAL, balance_krw=50_000_000, principal_krw=60_000_000, account_years=3
    )
    assert quote.taxable_krw == 0
    assert quote.national_tax_krw == 0
    assert quote.local_tax_krw == 0
    assert quote.net_proceeds_krw == 50_000_000


def test_allowance_by_class() -> None:
    regime = _regime()
    general = quote_isa_closure(
        regime, tax_class=IsaTaxClass.GENERAL, balance_krw=70_000_000, principal_krw=60_000_000, account_years=3
    )
    assert general.taxable_krw == 8_000_000
    assert general.national_tax_krw == 720_000
    assert general.local_tax_krw == 72_000
    preferential = quote_isa_closure(
        regime, tax_class=IsaTaxClass.PREFERENTIAL, balance_krw=70_000_000, principal_krw=60_000_000, account_years=3
    )
    assert preferential.taxable_krw == 6_000_000
    assert preferential.national_tax_krw == 540_000
    assert preferential.local_tax_krw == 54_000


def test_early_closure_unsupported() -> None:
    with pytest.raises(ValueError, match="minimum contract"):
        quote_isa_closure(
            _regime(), tax_class=IsaTaxClass.GENERAL, balance_krw=70_000_000, principal_krw=60_000_000, account_years=2
        )


def test_credit_maximizing_transfer() -> None:
    assert credit_maximizing_transfer_krw(_regime()) == 30_000_000


def test_transfer_credit_capped_at_cap_base() -> None:
    isa_regime = _regime()
    pension_regime = load_pension_tax_regime(_PENSION_SHIPPED)
    profile = _profile()
    regular = quote_annual_pension_credit(2031, 6_000_000, profile, pension_regime)
    quote = quote_isa_transfer_credit(
        2031, 70_000_000, regular=regular, profile=profile, pension_regime=pension_regime, isa_regime=isa_regime
    )
    assert quote.eligible_krw == 3_000_000
    assert quote.national_credit_krw == 450_000
    assert quote.local_credit_krw == 45_000
    assert quote.credited_principal_krw == 3_000_000
    assert quote.uncredited_principal_krw == 67_000_000


def test_transfer_credit_uses_leftover_capacity_only() -> None:
    isa_regime = _regime()
    pension_regime = load_pension_tax_regime(_PENSION_SHIPPED)
    profile = _profile(national=1_000_000, local=100_000)
    regular = quote_annual_pension_credit(2031, 6_000_000, profile, pension_regime)
    assert (regular.national_credit_krw, regular.local_credit_krw) == (900_000, 90_000)
    quote = quote_isa_transfer_credit(
        2031, 30_000_000, regular=regular, profile=profile, pension_regime=pension_regime, isa_regime=isa_regime
    )
    assert quote.national_credit_krw == 100_000
    assert quote.local_credit_krw == 10_000
    assert quote.credited_principal_krw == 666_667
    assert quote.uncredited_principal_krw == 30_000_000 - 666_667


def test_zero_capacity_yields_zero_credit() -> None:
    isa_regime = _regime()
    pension_regime = load_pension_tax_regime(_PENSION_SHIPPED)
    profile = _profile(national=0, local=0)
    regular = quote_annual_pension_credit(2031, 6_000_000, profile, pension_regime)
    quote = quote_isa_transfer_credit(
        2031, 30_000_000, regular=regular, profile=profile, pension_regime=pension_regime, isa_regime=isa_regime
    )
    assert quote.national_credit_krw == 0
    assert quote.local_credit_krw == 0
    assert quote.credited_principal_krw == 0
    assert quote.uncredited_principal_krw == 30_000_000


def test_high_income_rate() -> None:
    isa_regime = _regime()
    pension_regime = load_pension_tax_regime(_PENSION_SHIPPED)
    profile = _profile(income=70_000_000)
    regular = quote_annual_pension_credit(2031, 0, profile, pension_regime)
    quote = quote_isa_transfer_credit(
        2031, 30_000_000, regular=regular, profile=profile, pension_regime=pension_regime, isa_regime=isa_regime
    )
    assert quote.national_credit_krw == 360_000


def test_tax_year_mismatch_rejected() -> None:
    isa_regime = _regime()
    pension_regime = load_pension_tax_regime(_PENSION_SHIPPED)
    profile = _profile(year=2030)
    regular = quote_annual_pension_credit(2030, 6_000_000, profile, pension_regime)
    with pytest.raises(ValueError, match="differs"):
        quote_isa_transfer_credit(
            2031, 30_000_000, regular=regular, profile=profile, pension_regime=pension_regime, isa_regime=isa_regime
        )


def _write_config(tmp_path: Path, mutate: object) -> Path:
    import json as _json

    path = tmp_path / "isa.json"
    path.write_text(_json.dumps(mutate), encoding="utf-8")
    return path


def test_loader_rejects_malformed_fields(tmp_path: Path) -> None:
    """Each malformed or inconsistent config fails closed."""
    base = json.loads(_SHIPPED.read_text(encoding="utf-8"))
    bad_amount = dict(base)
    bad_amount["general_allowance_krw"] = "2000000"
    with pytest.raises(ValueError, match="integer amount"):
        load_isa_tax_regime(_write_config(tmp_path, bad_amount))
    negative = dict(base)
    negative["general_allowance_krw"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        load_isa_tax_regime(_write_config(tmp_path, negative))
    bad_rate = dict(base)
    bad_rate["separate_tax_rate"] = "0.09"
    with pytest.raises(ValueError, match="finite rate"):
        load_isa_tax_regime(_write_config(tmp_path, bad_rate))
    bad_count = dict(base)
    bad_count["carryover_years_cap"] = 4.5
    with pytest.raises(ValueError, match="integer count"):
        load_isa_tax_regime(_write_config(tmp_path, bad_count))
    negative_count = dict(base)
    negative_count["carryover_years_cap"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        load_isa_tax_regime(_write_config(tmp_path, negative_count))
    not_object = [1, 2]
    with pytest.raises(ValueError, match="JSON object"):
        load_isa_tax_regime(_write_config(tmp_path, not_object))
    empty_id = dict(base)
    empty_id["regime_id"] = ""
    with pytest.raises(ValueError, match="nonempty string"):
        load_isa_tax_regime(_write_config(tmp_path, empty_id))
    bad_year_type = dict(base)
    bad_year_type["policy_year"] = "2026"
    with pytest.raises(ValueError, match="integer year"):
        load_isa_tax_regime(_write_config(tmp_path, bad_year_type))
    unknown_year = dict(base)
    unknown_year["policy_year"] = 2025
    with pytest.raises(ValueError, match="unknown"):
        load_isa_tax_regime(_write_config(tmp_path, unknown_year))
    empty_refs = dict(base)
    empty_refs["statute_refs"] = []
    with pytest.raises(ValueError, match="statute_refs"):
        load_isa_tax_regime(_write_config(tmp_path, empty_refs))
    empty_urls = dict(base)
    empty_urls["source_urls"] = []
    with pytest.raises(ValueError, match="source_urls"):
        load_isa_tax_regime(_write_config(tmp_path, empty_urls))
    bad_checked = dict(base)
    bad_checked["source_checked_date"] = 20260926
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        load_isa_tax_regime(_write_config(tmp_path, bad_checked))
    malformed_checked = dict(base)
    malformed_checked["source_checked_date"] = "2026-13-40"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        load_isa_tax_regime(_write_config(tmp_path, malformed_checked))
    zero_term = dict(base)
    zero_term["minimum_contract_years"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        load_isa_tax_regime(_write_config(tmp_path, zero_term))
    zero_limit = dict(base)
    zero_limit["annual_contribution_limit_krw"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        load_isa_tax_regime(_write_config(tmp_path, zero_limit))
    inverted_limits = dict(base)
    inverted_limits["annual_contribution_limit_krw"] = 100_000_000
    inverted_limits["lifetime_contribution_limit_krw"] = 20_000_000
    with pytest.raises(ValueError, match="lifetime cap"):
        load_isa_tax_regime(_write_config(tmp_path, inverted_limits))
    inverted_allowance = dict(base)
    inverted_allowance["general_allowance_krw"] = 5_000_000
    with pytest.raises(ValueError, match="allowance"):
        load_isa_tax_regime(_write_config(tmp_path, inverted_allowance))


def test_quote_inputs_fail_closed() -> None:
    """Negative, non-integer, and unknown-kind quote inputs never produce a quote."""
    regime = _regime()
    with pytest.raises(ValueError, match="integer amount"):
        isa_contribution_room_krw(regime, elapsed_full_years=1.5, cumulative_contributed_krw=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        isa_contribution_room_krw(regime, elapsed_full_years=-1, cumulative_contributed_krw=0)
    with pytest.raises(ValueError, match="unsupported"):
        classify_isa_tax_class(regime, income_kind="business", prior_year_income_krw=1_000_000)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        classify_isa_tax_class(regime, income_kind="wage", prior_year_income_krw=-1)
    with pytest.raises(ValueError, match="integer amount"):
        classify_isa_tax_class(regime, income_kind="wage", prior_year_income_krw=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown ISA tax class"):
        quote_isa_closure(regime, tax_class="general", balance_krw=1, principal_krw=1, account_years=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer number of years"):
        quote_isa_closure(regime, tax_class=IsaTaxClass.GENERAL, balance_krw=1, principal_krw=1, account_years=3.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        quote_isa_closure(regime, tax_class=IsaTaxClass.GENERAL, balance_krw=-1, principal_krw=0, account_years=3)


def test_transfer_quote_guards_and_degenerate_bases() -> None:
    """Transfer guards reject bad years/kinds/capacity; dust transfers carry no credit."""
    isa_regime = _regime()
    pension_regime = load_pension_tax_regime(_PENSION_SHIPPED)
    profile = _profile()
    regular = quote_annual_pension_credit(2031, 6_000_000, profile, pension_regime)
    with pytest.raises(ValueError, match="integer year"):
        quote_isa_transfer_credit(
            True, 1_000_000, regular=regular, profile=profile, pension_regime=pension_regime, isa_regime=isa_regime
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported"):
        quote_isa_transfer_credit(
            2031,
            1_000_000,
            regular=regular,
            profile=_profile(kind="business"),
            pension_regime=pension_regime,
            isa_regime=isa_regime,
        )
    with pytest.raises(ValueError, match="no entry"):
        quote_isa_transfer_credit(
            2031, 1_000_000, regular=regular, profile=_profile(year=2030), pension_regime=pension_regime, isa_regime=isa_regime
        )
    dust = quote_isa_transfer_credit(
        2031, 1, regular=regular, profile=profile, pension_regime=pension_regime, isa_regime=isa_regime
    )
    assert dust.eligible_krw == 0
    assert dust.credited_principal_krw == 0
    assert dust.uncredited_principal_krw == 1
    zero_cap = replace(isa_regime, pension_transfer_credit_cap_krw=0)
    assert credit_maximizing_transfer_krw(zero_cap) == 0
    zero_rate = replace(isa_regime, pension_transfer_credit_rate=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        credit_maximizing_transfer_krw(zero_rate)
