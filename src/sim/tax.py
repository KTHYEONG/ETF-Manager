"""Korean overseas-equity tax regime contract and annual gains computation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

__all__ = [
    "BasisMethod",
    "KrOverseasTaxRegime",
    "annual_capital_gains_tax_krw",
    "load_tax_regime",
]

_EXPECTED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "regime_id",
        "capital_gains_rate",
        "annual_deduction_krw",
        "loss_carryforward",
        "dividend_withholding_rate",
        "interest_tax_rate",
        "settlement_sessions",
        "payment_month",
        "financial_income_threshold_krw",
        "basis_method",
    }
)
_RATE_KEYS: Final[tuple[str, ...]] = (
    "capital_gains_rate",
    "dividend_withholding_rate",
    "interest_tax_rate",
)
_AMOUNT_KEYS: Final[tuple[str, ...]] = (
    "annual_deduction_krw",
    "financial_income_threshold_krw",
)


class BasisMethod(StrEnum):
    """Cost-basis identification for disposals of the same ticker."""

    FIFO = "fifo"
    MOVING_AVERAGE = "moving_average"


@dataclass(frozen=True, slots=True)
class KrOverseasTaxRegime:
    """Korean resident tax contract for directly held US-listed ETFs (general account).

    Values are loaded from a versioned config, never hard-coded in business logic, so
    a future law change is a new regime file rather than a code edit.
    """

    regime_id: str
    capital_gains_rate: float
    annual_deduction_krw: float
    loss_carryforward: bool
    dividend_withholding_rate: float
    interest_tax_rate: float
    settlement_sessions: int
    payment_month: int
    financial_income_threshold_krw: float
    basis_method: BasisMethod


def load_tax_regime(path: str | Path) -> KrOverseasTaxRegime:
    """Load and validate a tax regime JSON.

    Raises:
        ValueError: On missing/extra keys, rates outside [0, 1), negative amounts,
            ``settlement_sessions < 1``, ``payment_month`` outside 1..12, an unknown
            basis method, or ``loss_carryforward`` set to true (not modelled; fail closed).
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"tax regime {str(path)!r} must be a JSON object")
    keys = frozenset(document.keys())
    if keys != _EXPECTED_KEYS:
        raise ValueError(
            f"tax regime {str(path)!r} has unexpected keys: missing={sorted(_EXPECTED_KEYS - keys)} "
            f"extra={sorted(keys - _EXPECTED_KEYS)}"
        )
    for key in (*_RATE_KEYS, *_AMOUNT_KEYS):
        value = document[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"tax regime field {key!r} must be a finite number")
    for key in _RATE_KEYS:
        value = document[key]
        if value < 0 or value >= 1:
            raise ValueError(f"tax regime field {key!r} must lie in [0, 1), got {value!r}")
    for key in _AMOUNT_KEYS:
        if document[key] < 0:
            raise ValueError(f"tax regime field {key!r} must be nonnegative, got {document[key]!r}")
    if document["loss_carryforward"] is True:
        raise ValueError("tax regime with loss_carryforward=true is not modelled; fail closed")
    if not isinstance(document["loss_carryforward"], bool):
        raise ValueError("tax regime field 'loss_carryforward' must be a boolean")
    settlement_sessions = document["settlement_sessions"]
    if isinstance(settlement_sessions, bool) or not isinstance(settlement_sessions, int):
        raise ValueError("tax regime field 'settlement_sessions' must be an integer")
    if settlement_sessions < 1:
        raise ValueError(f"tax regime field 'settlement_sessions' must be >= 1, got {settlement_sessions!r}")
    payment_month = document["payment_month"]
    if isinstance(payment_month, bool) or not isinstance(payment_month, int):
        raise ValueError("tax regime field 'payment_month' must be an integer")
    if payment_month < 1 or payment_month > 12:
        raise ValueError(f"tax regime field 'payment_month' must lie in 1..12, got {payment_month!r}")
    regime_id = document["regime_id"]
    if not isinstance(regime_id, str) or not regime_id:
        raise ValueError("tax regime field 'regime_id' must be a nonempty string")
    try:
        basis_method = BasisMethod(document["basis_method"])
    except ValueError as exc:
        raise ValueError(f"tax regime field 'basis_method' is unknown: {document['basis_method']!r}") from exc
    return KrOverseasTaxRegime(
        regime_id=regime_id,
        capital_gains_rate=float(document["capital_gains_rate"]),
        annual_deduction_krw=float(document["annual_deduction_krw"]),
        loss_carryforward=False,
        dividend_withholding_rate=float(document["dividend_withholding_rate"]),
        interest_tax_rate=float(document["interest_tax_rate"]),
        settlement_sessions=settlement_sessions,
        payment_month=payment_month,
        financial_income_threshold_krw=float(document["financial_income_threshold_krw"]),
        basis_method=basis_method,
    )


def annual_capital_gains_tax_krw(net_realized_gain_krw: float, regime: KrOverseasTaxRegime) -> float:
    """Tax due for one calendar year's net realized KRW gain.

    ``rate * max(0, net - deduction)``; a net loss yields zero and is discarded because
    Korean overseas-equity losses never carry into later years.

    Raises:
        ValueError: When the input is non-finite.
    """
    if isinstance(net_realized_gain_krw, bool) or not isinstance(net_realized_gain_krw, int | float):
        raise ValueError(f"net realized gain must be a number, got {net_realized_gain_krw!r}")
    if not math.isfinite(net_realized_gain_krw):
        raise ValueError(f"net realized gain must be finite, got {net_realized_gain_krw!r}")
    taxable = net_realized_gain_krw - regime.annual_deduction_krw
    if taxable <= 0:
        return 0.0
    return regime.capital_gains_rate * taxable
