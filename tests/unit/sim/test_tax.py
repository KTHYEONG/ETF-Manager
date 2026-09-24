"""Unit tests for the Korean overseas-equity tax regime contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.sim.tax import BasisMethod, KrOverseasTaxRegime, annual_capital_gains_tax_krw, load_tax_regime

_SHIPPED = Path("configs/tax/kr_overseas_equity.json")


def _shipped_document() -> dict[str, Any]:
    return json.loads(_SHIPPED.read_text(encoding="utf-8"))


def _write_regime(tmp_path: Path, mutate: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> Path:
    document = _shipped_document()
    if mutate:
        document.update(mutate)
    if extra:
        document.update(extra)
    path = tmp_path / "regime.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _regime() -> KrOverseasTaxRegime:
    return load_tax_regime(_SHIPPED)


def test_shipped_regime_loads() -> None:
    """Shipped config carries the 2026 overseas-equity terms."""
    regime = _regime()
    assert regime.regime_id == "KR_OVERSEAS_EQUITY_2026"
    assert regime.capital_gains_rate == pytest.approx(0.22)
    assert regime.annual_deduction_krw == pytest.approx(2_500_000)
    assert regime.basis_method is BasisMethod.FIFO
    assert regime.settlement_sessions == 1
    assert regime.payment_month == 5
    assert regime.loss_carryforward is False
    assert regime.dividend_withholding_rate == pytest.approx(0.15)


def test_deduction_boundary() -> None:
    """Gains at the deduction pay nothing; one won above pays 0.22 won."""
    regime = _regime()
    assert annual_capital_gains_tax_krw(2_500_000, regime) == 0.0
    assert annual_capital_gains_tax_krw(2_500_001, regime) == pytest.approx(0.22, abs=1e-9)


def test_loss_year_yields_zero() -> None:
    """A net loss year owes no tax and carries nothing forward."""
    assert annual_capital_gains_tax_krw(-10_000_000, _regime()) == 0.0


def test_tax_curve_shape() -> None:
    """Output is nonnegative, monotone, and zero below the deduction."""
    regime = _regime()
    assert annual_capital_gains_tax_krw(0, regime) == 0.0
    low = annual_capital_gains_tax_krw(3_000_000, regime)
    high = annual_capital_gains_tax_krw(4_000_000, regime)
    assert low >= 0.0
    assert high >= low
    assert high == pytest.approx(0.22 * 1_500_000)


def test_carryforward_rejected(tmp_path: Path) -> None:
    """A regime with loss carryforward fails closed."""
    path = _write_regime(tmp_path, {"loss_carryforward": True})
    with pytest.raises(ValueError, match="carryforward"):
        load_tax_regime(path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    """An extra key fails the exact-key contract."""
    path = _write_regime(tmp_path, extra={"withholding_note": "x"})
    with pytest.raises(ValueError, match="extra"):
        load_tax_regime(path)


def test_regime_validation_boundaries(tmp_path: Path) -> None:
    """Every malformed regime field raises ValueError without leaking values."""
    base = _shipped_document()
    cases: list[tuple[dict[str, Any], str]] = [
        ({"capital_gains_rate": 1.0}, "lie in"),
        ({"capital_gains_rate": -0.01}, "lie in"),
        ({"dividend_withholding_rate": "0.15"}, "finite number"),
        ({"annual_deduction_krw": -1}, "nonnegative"),
        ({"financial_income_threshold_krw": float("nan")}, "finite number"),
        ({"loss_carryforward": "false"}, "boolean"),
        ({"settlement_sessions": 0}, ">= 1"),
        ({"settlement_sessions": "1"}, "integer"),
        ({"payment_month": 0}, "1..12"),
        ({"payment_month": 13}, "1..12"),
        ({"payment_month": 5.0}, "integer"),
        ({"regime_id": ""}, "nonempty string"),
        ({"regime_id": 7}, "nonempty string"),
        ({"basis_method": "average"}, "unknown"),
        ({"basis_method": None}, "unknown"),
    ]
    for mutate, match in cases:
        document = dict(base)
        document.update(mutate)
        path = tmp_path / "regime.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_tax_regime(path)


def test_missing_key_rejected(tmp_path: Path) -> None:
    """A regime missing one key fails the exact-key contract."""
    document = _shipped_document()
    del document["payment_month"]
    path = tmp_path / "regime.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_tax_regime(path)


def test_non_object_regime_rejected(tmp_path: Path) -> None:
    """A regime document that is not an object fails closed."""
    path = tmp_path / "regime.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_tax_regime(path)


def test_non_finite_gain_rejected() -> None:
    """Non-finite or non-numeric gains never produce a tax figure."""
    regime = _regime()
    with pytest.raises(ValueError, match="finite"):
        annual_capital_gains_tax_krw(float("inf"), regime)
    with pytest.raises(ValueError, match="finite"):
        annual_capital_gains_tax_krw(float("nan"), regime)
    with pytest.raises(ValueError, match="number"):
        annual_capital_gains_tax_krw("100", regime)  # type: ignore[arg-type]
