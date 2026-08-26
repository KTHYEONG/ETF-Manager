"""Unit tests for stateless adaptive contribution sizing via the KAFI opportunity score."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Final

import polars as pl
import pytest

import src.policy.adaptive_contribution as adaptive_module
from src.policy.adaptive_contribution import (
    OPERATIONAL_ADAPTIVE_CONTRIBUTION,
    AdaptiveContributionConfig,
    size_adaptive_contribution,
)
from src.policy.targets import PolicyError

_BASE: Final[float] = 1_000_000.0
_SIGNAL_AT: Final[datetime] = datetime(2024, 6, 28, 21, 0, tzinfo=UTC)
_EMPTY: Final[pl.DataFrame] = pl.DataFrame()


def _size(config: AdaptiveContributionConfig, base: float = _BASE) -> float:
    return size_adaptive_contribution(
        base_contribution_krw=base,
        signal_at=_SIGNAL_AT,
        prices=_EMPTY,
        fx=_EMPTY,
        macro=_EMPTY,
        config=config,
    )


@pytest.fixture
def forced_scores(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    scores: list[float] = []

    def fake_opportunity_score(**_kwargs: object) -> float:
        if not scores:
            raise AssertionError("scripted opportunity score path exhausted")
        return scores.pop(0)

    monkeypatch.setattr(adaptive_module, "kafi_opportunity_score", fake_opportunity_score)
    return scores


@pytest.mark.parametrize("scenario_id", ["ACG-A-curve-anchors"])
def test_acg_a_curve_anchors(scenario_id: str, forced_scores: list[float]) -> None:
    """ACG-A-curve-anchors"""
    forced_scores.extend([0.0, 25.0, 50.0, 75.0, 100.0])
    outputs = [_size(AdaptiveContributionConfig()) for _ in range(5)]

    assert outputs == pytest.approx(
        [0.0, _BASE * (1 - 0.5**2.5), _BASE, _BASE * (1 + 0.5**0.7), 2 * _BASE], abs=1e-6
    )
    assert outputs == sorted(outputs)
    assert all(0.0 <= output <= 2 * _BASE for output in outputs)


@pytest.mark.parametrize("scenario_id", ["ACR-ACG-default-curve"])
def test_acr_acg_default_curve(scenario_id: str, forced_scores: list[float]) -> None:
    """ACR-ACG-default-curve"""
    config = AdaptiveContributionConfig()
    assert config.rank_window == 126
    assert config.downside_power == pytest.approx(2.5)
    assert config.upside_power == pytest.approx(0.7)
    assert config.min_multiplier == pytest.approx(0.0)
    assert config.max_multiplier == pytest.approx(2.0)

    forced_scores.extend([0.0, 50.0, 100.0])
    anchors = [_size(config) for _ in range(3)]
    assert anchors == pytest.approx([0.0, _BASE, 2 * _BASE], abs=1e-6)

    forced_scores.extend([25.0, 75.0])
    midpoints = [_size(config) for _ in range(2)]
    assert midpoints == pytest.approx(
        [_BASE * (1 - 0.5**2.5), _BASE * (1 + 0.5**0.7)], abs=1e-6
    )


@pytest.mark.parametrize("scenario_id", ["ACG-B-causal-no-conservation"])
def test_acg_b_causal_no_conservation(scenario_id: str, forced_scores: list[float]) -> None:
    """ACG-B-causal-no-conservation"""
    forced_scores.extend([100.0, 100.0, 100.0])
    months = [_size(AdaptiveContributionConfig()) for _ in range(3)]

    assert months == pytest.approx([2 * _BASE] * 3, abs=1e-6)
    # No horizon conservation: cumulative output exceeds the old conserved 3 * base.
    assert sum(months) == pytest.approx(6 * _BASE, rel=1e-9)
    # The API accepts neither ledger state nor a terminal settlement offset.
    assert set(inspect.signature(size_adaptive_contribution).parameters) == {
        "base_contribution_krw",
        "signal_at",
        "prices",
        "fx",
        "macro",
        "config",
    }


@pytest.mark.parametrize("scenario_id", ["ACG-C-fail-closed"])
def test_acg_c_fail_closed(scenario_id: str) -> None:
    """ACG-C-fail-closed"""
    config = AdaptiveContributionConfig()
    with pytest.raises(ValueError, match="base_contribution_krw"):
        _size(config, base=0.0)
    with pytest.raises(ValueError, match="base_contribution_krw"):
        _size(config, base=-_BASE)

    with pytest.raises(ValueError, match="min_multiplier"):
        AdaptiveContributionConfig(min_multiplier=float("nan"))
    with pytest.raises(ValueError, match="max_multiplier"):
        AdaptiveContributionConfig(max_multiplier=float("inf"))
    with pytest.raises(ValueError, match="downside_power"):
        AdaptiveContributionConfig(downside_power=float("nan"))
    with pytest.raises(ValueError, match="upside_power"):
        AdaptiveContributionConfig(upside_power=float("inf"))
    with pytest.raises(ValueError, match="min_multiplier"):
        AdaptiveContributionConfig(min_multiplier=-0.10)
    with pytest.raises(ValueError, match="max_multiplier"):
        AdaptiveContributionConfig(max_multiplier=2.50)
    with pytest.raises(ValueError, match="min_multiplier"):
        AdaptiveContributionConfig(min_multiplier=1.0)
    with pytest.raises(ValueError, match="max_multiplier"):
        AdaptiveContributionConfig(max_multiplier=1.0)
    with pytest.raises(ValueError, match="rank_window"):
        AdaptiveContributionConfig(rank_window=62)


@pytest.mark.parametrize("scenario_id", ["ACG-C-fail-closed"])
def test_acg_c_fail_closed_on_kafi_error(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """ACG-C-fail-closed"""

    def boom(**_kwargs: object) -> float:
        raise ValueError("insufficient momentum history")

    monkeypatch.setattr(adaptive_module, "kafi_opportunity_score", boom)
    with pytest.raises(PolicyError, match="adaptive contribution failed closed"):
        _size(AdaptiveContributionConfig())


@pytest.mark.parametrize("scenario_id", ["ACG-v2-passes-flag"])
def test_acg_v2_passes_flag(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """ACG-v2-passes-flag"""
    captured: list[dict[str, object]] = []

    def fake_opportunity_score(**kwargs: object) -> float:
        captured.append(dict(kwargs))
        return 75.0

    monkeypatch.setattr(adaptive_module, "kafi_opportunity_score", fake_opportunity_score)

    challenger_credit = _size(AdaptiveContributionConfig(include_vol_dampener=False))
    operational_credit = _size(AdaptiveContributionConfig())

    assert captured[0]["include_vol_dampener"] is False
    assert captured[1]["include_vol_dampener"] is True
    assert challenger_credit == pytest.approx(operational_credit)
    # Operational lock stays on the six-component curve in this wave.
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.include_vol_dampener is True
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.rank_window == 126
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.downside_power == pytest.approx(2.5)
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.upside_power == pytest.approx(0.7)
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.min_multiplier == pytest.approx(0.0)
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.max_multiplier == pytest.approx(2.0)


@pytest.mark.parametrize("scenario_id", ["ACG-disp-passes-flag"])
def test_acg_disp_passes_flag(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """ACG-disp-passes-flag"""
    captured: list[dict[str, object]] = []

    def fake_opportunity_score(**kwargs: object) -> float:
        captured.append(dict(kwargs))
        return 75.0

    monkeypatch.setattr(adaptive_module, "kafi_opportunity_score", fake_opportunity_score)

    _size(AdaptiveContributionConfig(dispersion=0.9))
    _size(AdaptiveContributionConfig())

    assert captured[0]["dispersion"] == pytest.approx(0.9)
    assert captured[1]["dispersion"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="dispersion"):
        AdaptiveContributionConfig(dispersion=0.0)
