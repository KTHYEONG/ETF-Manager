"""Unit tests for the target-weight rule contract (policy layer isolation)."""

from __future__ import annotations

import ast
from pathlib import Path

from src.policy.weight_rule import CASH_SLEEVE, WeightRule


class _FixedRule:
    """Minimal rule used to prove the protocol shape."""

    tickers = frozenset({"QQQ"})
    requires_cash_rate = False

    def __call__(self, signal_at, market):  # type: ignore[no-untyped-def]
        closes = market.daily_adjusted_closes("QQQ", signal_at, 1)
        assert len(closes) == 1
        return {"QQQ": 1.0}


def test_cash_sleeve_identity() -> None:
    """The cash sleeve token is the CASH string rules and engine share."""
    assert CASH_SLEEVE == "CASH"
    assert isinstance(_FixedRule().tickers, frozenset)
    assert _FixedRule().requires_cash_rate is False


def test_weight_rule_is_protocol() -> None:
    """WeightRule stays a Protocol so rules never inherit simulator behavior."""
    assert getattr(WeightRule, "_is_protocol", False) is True
    assert set(getattr(WeightRule, "__protocol_attrs__", ())) >= {"tickers", "requires_cash_rate"}


def test_weight_rule_imports_avoid_simulator() -> None:
    """The policy contract imports only the PIT market view and stdlib."""
    tree = ast.parse(Path("src/policy/weight_rule.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported
    assert not [name for name in imported if name == "src.sim" or name.startswith("src.sim.")]
