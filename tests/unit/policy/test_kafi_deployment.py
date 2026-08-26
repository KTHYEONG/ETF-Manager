"""Unit tests for causal KAFI deployment via an explicit reserve ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import polars as pl
import pytest

import src.policy.kafi_deployment as deployment_module
from src.policy.kafi_deployment import KafiDeploymentConfig, apply_kafi_deployment
from src.policy.targets import PolicyError

_BASE: Final[float] = 1_000_000.0
_SIGNAL_AT: Final[datetime] = datetime(2024, 6, 28, 21, 0, tzinfo=UTC)
_EMPTY: Final[pl.DataFrame] = pl.DataFrame()


def _config(**overrides: object) -> KafiDeploymentConfig:
    return KafiDeploymentConfig(**overrides)  # type: ignore[arg-type]


@pytest.fixture
def forced_scores(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    scores: list[float] = []

    def fake_opportunity_score(**_kwargs: object) -> float:
        if not scores:
            raise AssertionError("scripted opportunity score path exhausted")
        return scores.pop(0)

    monkeypatch.setattr(deployment_module, "kafi_opportunity_score", fake_opportunity_score)
    return scores


@pytest.mark.parametrize("scenario_id", ["DEPLOY-A-ledger-identity"])
def test_deploy_a_ledger_identity(scenario_id: str, forced_scores: list[float]) -> None:
    """DEPLOY-A-ledger-identity"""
    forced_scores.extend([20.0, 80.0, 50.0])
    config = _config()
    reserve = 0.0
    for _ in range(3):
        decision = apply_kafi_deployment(
            contribution_krw=_BASE,
            reserve_krw=reserve,
            signal_at=_SIGNAL_AT,
            prices=_EMPTY,
            fx=_EMPTY,
            macro=_EMPTY,
            config=config,
        )
        assert decision.investable_krw + decision.reserve_krw == pytest.approx(_BASE + reserve, rel=1e-6)
        assert decision.reserve_krw >= 0.0
        assert decision.investable_krw <= _BASE + reserve + 1e-6
        reserve = decision.reserve_krw


@pytest.mark.parametrize("scenario_id", ["DEPLOY-B-neutral-multiplier"])
def test_deploy_b_neutral_multiplier(scenario_id: str, forced_scores: list[float]) -> None:
    """DEPLOY-B-neutral-multiplier"""
    forced_scores.append(50.0)
    decision = apply_kafi_deployment(
        contribution_krw=_BASE,
        reserve_krw=0.0,
        signal_at=_SIGNAL_AT,
        prices=_EMPTY,
        fx=_EMPTY,
        macro=_EMPTY,
        config=_config(),
    )
    assert decision.investable_krw == pytest.approx(_BASE)
    assert decision.reserve_krw == pytest.approx(0.0)


@pytest.mark.parametrize("scenario_id", ["DEPLOY-C-stock-cap"])
def test_deploy_c_stock_cap(scenario_id: str, forced_scores: list[float]) -> None:
    """DEPLOY-C-stock-cap"""
    forced_scores.append(100.0)
    decision = apply_kafi_deployment(
        contribution_krw=_BASE,
        reserve_krw=0.0,
        signal_at=_SIGNAL_AT,
        prices=_EMPTY,
        fx=_EMPTY,
        macro=_EMPTY,
        config=_config(),
    )
    assert decision.investable_krw == pytest.approx(_BASE)
    assert decision.reserve_krw == pytest.approx(0.0)


def test_deploy_fail_closed_on_kafi_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_kwargs: object) -> float:
        raise ValueError("momentum")

    monkeypatch.setattr(deployment_module, "kafi_opportunity_score", boom)
    with pytest.raises(PolicyError, match="kafi deployment failed closed"):
        apply_kafi_deployment(
            contribution_krw=_BASE,
            reserve_krw=0.0,
            signal_at=_SIGNAL_AT,
            prices=_EMPTY,
            fx=_EMPTY,
            macro=_EMPTY,
            config=_config(),
        )
