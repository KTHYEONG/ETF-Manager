"""Unit tests for horizon-conserved contribution shaping (I5h projection)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final
from collections.abc import Iterator

import polars as pl
import pytest

import src.policy.contribution_shape as shape_module
from src.policy.contribution_shape import (
    ContributionBudgetState,
    ContributionShapeConfig,
    shape_monthly_contribution,
)
from src.policy.targets import PolicyError

_BASE: Final[float] = 1_000_000.0
_HORIZON: Final[int] = 12
_SIGNAL_AT: Final[datetime] = datetime(2024, 6, 28, 21, 0, tzinfo=UTC)
_EMPTY: Final[pl.DataFrame] = pl.DataFrame()


def _config(**overrides: object) -> ContributionShapeConfig:
    return ContributionShapeConfig(**overrides)  # type: ignore[arg-type]


@pytest.fixture
def forced_scores(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Replace the KAFI lookup with a scripted score path consumed front-to-back."""
    scores: list[float] = []

    def fake_kafi_score(**_kwargs: object) -> float:
        if not scores:
            raise AssertionError("scripted KAFI path exhausted")
        return scores.pop(0)

    monkeypatch.setattr(shape_module, "kafi_score", fake_kafi_score)
    yield scores
    monkeypatch.undo()


@pytest.mark.parametrize("scenario_id", ["SHAPE-A-i5h-conservation"])
def test_shape_a_i5h_conservation(scenario_id: str, forced_scores: list[float]) -> None:
    """SHAPE-A-i5h-conservation"""
    forced_scores.extend([20.0, 80.0, *([50.0] * 10)])
    config = _config()
    state = ContributionBudgetState(horizon_months=_HORIZON)
    credits: list[float] = []
    for _ in range(_HORIZON):
        credit, state = shape_monthly_contribution(
            base_contribution_krw=_BASE,
            signal_at=_SIGNAL_AT,
            prices=_EMPTY,
            fx=_EMPTY,
            macro=_EMPTY,
            config=config,
            budget_state=state,
        )
        credits.append(credit)

    assert sum(credits) == pytest.approx(_HORIZON * _BASE, rel=1e-6)
    assert all(credit > 0.0 for credit in credits)
    multipliers = [credit / _BASE for credit in credits]
    assert all(config.min_multiplier <= m <= config.max_multiplier for m in multipliers)


@pytest.mark.parametrize("scenario_id", ["SHAPE-B-fear-frontload"])
def test_shape_b_fear_frontload(scenario_id: str, forced_scores: list[float]) -> None:
    """SHAPE-B-fear-frontload"""
    forced_scores.extend([15.0, *([85.0] * 11)])
    config = _config(budget_window_months=12)
    state = ContributionBudgetState(horizon_months=_HORIZON)
    credits: list[float] = []
    for _ in range(_HORIZON):
        credit, state = shape_monthly_contribution(
            base_contribution_krw=_BASE,
            signal_at=_SIGNAL_AT,
            prices=_EMPTY,
            fx=_EMPTY,
            macro=_EMPTY,
            config=config,
            budget_state=state,
        )
        credits.append(credit)

    assert credits[0] > _BASE
    assert any(credit < _BASE for credit in credits[1:])
    assert sum(credits) == pytest.approx(_HORIZON * _BASE, rel=1e-6)


@pytest.mark.parametrize("scenario_id", ["SHAPE-A-state-immutable"])
def test_shape_a_state_immutable(scenario_id: str, forced_scores: list[float]) -> None:
    """SHAPE-A-state-immutable"""
    forced_scores.append(50.0)
    config = _config()
    state = ContributionBudgetState(horizon_months=3)

    _credit, next_state = shape_monthly_contribution(
        base_contribution_krw=_BASE,
        signal_at=_SIGNAL_AT,
        prices=_EMPTY,
        fx=_EMPTY,
        macro=_EMPTY,
        config=config,
        budget_state=state,
    )

    assert state.months_elapsed == 0
    assert state.cumulative_dev_krw == pytest.approx(0.0)
    assert next_state.months_elapsed == 1


@pytest.mark.parametrize("scenario_id", ["SHAPE-B-fail-closed"])
def test_shape_b_fail_closed(scenario_id: str, forced_scores: list[float]) -> None:
    """SHAPE-B-fail-closed"""
    config = _config()
    with pytest.raises(Exception, match="horizon"):
        shape_monthly_contribution(
            base_contribution_krw=_BASE,
            signal_at=_SIGNAL_AT,
            prices=_EMPTY,
            fx=_EMPTY,
            macro=_EMPTY,
            config=config,
            budget_state=ContributionBudgetState(),
        )
    forced_scores.extend([50.0] * 4)
    state = ContributionBudgetState(horizon_months=2)
    for _ in range(2):
        _, state = shape_monthly_contribution(
            base_contribution_krw=_BASE,
            signal_at=_SIGNAL_AT,
            prices=_EMPTY,
            fx=_EMPTY,
            macro=_EMPTY,
            config=config,
            budget_state=state,
        )
    with pytest.raises(PolicyError, match="planned months"):
        shape_monthly_contribution(
            base_contribution_krw=_BASE,
            signal_at=_SIGNAL_AT,
            prices=_EMPTY,
            fx=_EMPTY,
            macro=_EMPTY,
            config=config,
            budget_state=state,
        )
