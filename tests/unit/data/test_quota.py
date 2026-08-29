"""Unit tests for vendor quota pacing."""

from __future__ import annotations

import pytest

from src.data.providers.base import ProviderError
from src.data.providers.quota import TIINGO_QUOTA, PacingGate


def test_pacing_gate_enforces_min_interval() -> None:
    """test_pacing_gate_enforces_min_interval"""
    assert TIINGO_QUOTA.min_interval_s == 72.0
    now = {"t": 0.0}
    sleeps: list[float] = []

    def clock() -> float:
        return now["t"]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        now["t"] += seconds

    gate = PacingGate(TIINGO_QUOTA, clock=clock, sleeper=sleeper)
    gate.acquire()
    gate.acquire()
    assert sleeps
    assert sleeps[0] >= TIINGO_QUOTA.min_interval_s - 1e-9


def test_pacing_gate_daily_budget_fail_closed() -> None:
    """test_pacing_gate_daily_budget_fail_closed"""
    from datetime import UTC, datetime

    start = datetime(2024, 6, 15, 0, 0, tzinfo=UTC).timestamp()
    now = {"t": start}

    def clock() -> float:
        return now["t"]

    gate = PacingGate(TIINGO_QUOTA, clock=clock, sleeper=lambda _s: None)
    for _ in range(TIINGO_QUOTA.requests_per_day):
        gate.acquire()
        now["t"] += TIINGO_QUOTA.min_interval_s
    with pytest.raises(ProviderError, match="rate limit"):
        gate.acquire()
