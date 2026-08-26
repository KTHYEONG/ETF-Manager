"""Unit tests for ERP-preserving invest-multiplier sizing."""

from __future__ import annotations

import pytest

from src.policy.sizing import erp_preserving_multiplier

_MIN_INVEST = 0.70
_MAX_INVEST = 3.0


@pytest.mark.parametrize("scenario_id", ["SIZ-V4-erp-polarity"])
@pytest.mark.parametrize(
    ("depth", "trend", "expected"),
    [
        (0.05, 0.10, 1.0),
        (0.15, 0.10, 1.0),
        (0.22, 0.10, 3.0),
        (0.05, -0.10, 1.0),
        (0.15, -0.10, 0.70),
        (0.22, -0.10, 3.0),
    ],
)
def test_siz_v4_erp_polarity(scenario_id: str, depth: float, trend: float, expected: float) -> None:
    """SIZ-V4-erp-polarity"""
    multiplier = erp_preserving_multiplier(
        depth=depth,
        trend=trend,
        min_invest_multiplier=_MIN_INVEST,
        max_invest_multiplier=_MAX_INVEST,
    )

    assert multiplier == pytest.approx(expected)
