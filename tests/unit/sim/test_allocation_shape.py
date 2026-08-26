"""Unit tests for contribution shaping inside run_allocation: full invest, no reserve book."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Final

import polars as pl
import pytest

import src.policy.contribution_shape as shape_module
from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.schema import Dataset, spec_for
from src.policy.contribution_shape import ContributionShapeConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, run_allocation

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT: Final[datetime] = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)
_CONTRIBUTION_KRW: Final[float] = 1_000_000.0
_CONFIG_START: Final[date] = date(2024, 1, 2)
_CONFIG_END: Final[date] = date(2024, 3, 28)
_SESSIONS: Final[tuple[date, ...]] = _CALENDAR.sessions(date(2023, 1, 2), date(2024, 4, 5))


def _prices_panel() -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    n = len(_SESSIONS)
    rows: dict[str, list[object]] = {
        "ticker": ["QQQ"] * n,
        "date": list(_SESSIONS),
        "open": [98.0] * n,
        "high": [102.0] * n,
        "low": [97.0] * n,
        "close": [100.0] * n,
        "volume": [10_000] * n,
        "adjusted_close": [100.0] * n,
        "dividend": [0.0] * n,
        "split_factor": [1.0] * n,
        "source": ["synthetic"] * n,
        "retrieved_at": [_RETRIEVED_AT] * n,
    }
    return ingest(pl.DataFrame(rows, schema=dict(spec.columns)), Dataset.PRICES)


def _fx_panel() -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    n = len(_SESSIONS)
    return ingest(
        pl.DataFrame(
            {
                "date": list(_SESSIONS),
                "usdkrw": [1300.0] * n,
                "source": ["synthetic"] * n,
                "retrieved_at": [_RETRIEVED_AT] * n,
            },
            schema=dict(spec.columns),
        ),
        Dataset.FX,
    )


def _cpi_panel() -> pl.DataFrame:
    spec = spec_for(Dataset.CPI)
    return ingest(
        pl.DataFrame(
            {
                "period_end": [date(2023, 12, 1)],
                "value": [100.0],
                "source": ["synthetic"],
                "retrieved_at": [_RETRIEVED_AT],
            },
            schema=dict(spec.columns),
        ),
        Dataset.CPI,
    )


def _macro_panel() -> pl.DataFrame:
    spec = spec_for(Dataset.MACRO)
    days = _SESSIONS[-30:]
    return ingest(
        pl.DataFrame(
            {
                "series_id": ["BAA10Y"] * len(days),
                "observation_date": list(days),
                "release_date": [
                    datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(hours=21) for d in days
                ],
                "value": [3.5] * len(days),
            },
            schema=dict(spec.columns),
        ),
        Dataset.MACRO,
    )


@pytest.mark.parametrize("scenario_id", ["SHAPE-C-allocation-no-reserve-book"])
def test_shape_c_allocation_no_reserve_book(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """SHAPE-C-allocation-no-reserve-book"""
    scripted: Final[list[float]] = [30.0, 70.0, 50.0]

    def fake_kafi_score(**_kwargs: object) -> float:
        return scripted.pop(0)

    monkeypatch.setattr(shape_module, "kafi_score", fake_kafi_score)
    result = run_allocation(
        AllocationConfig(
            policy=PolicyId.QQQ,
            start=_CONFIG_START,
            end=_CONFIG_END,
            monthly_contribution_krw=_CONTRIBUTION_KRW,
            contribution_shape=ContributionShapeConfig(),
        ),
        _prices_panel(),
        _fx_panel(),
        _cpi_panel(),
        macro=_macro_panel(),
    )

    n = len(result.snapshots)
    assert n == 3
    assert all(snapshot.reserve_krw == 0.0 for snapshot in result.snapshots)
    total = sum(snapshot.contribution_krw for snapshot in result.snapshots)
    assert abs(total - n * _CONTRIBUTION_KRW) / (n * _CONTRIBUTION_KRW) <= 1e-6


@pytest.mark.parametrize("scenario_id", ["SHAPE-C-xor-modules"])
def test_shape_c_xor_modules(scenario_id: str) -> None:
    """SHAPE-C-xor-modules"""
    base: dict[str, object] = {
        "policy": PolicyId.QQQ,
        "start": _CONFIG_START,
        "end": _CONFIG_END,
        "monthly_contribution_krw": _CONTRIBUTION_KRW,
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_allocation(
            AllocationConfig(**base, overlay=OverlayConfig(), reserve=ReserveConfig(max_withhold=0.05)),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_allocation(
            AllocationConfig(**base, overlay=OverlayConfig(), contribution_shape=ContributionShapeConfig()),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    with pytest.raises(ValueError, match="monthly"):
        run_allocation(
            AllocationConfig(**base, cadence="month_open", contribution_shape=ContributionShapeConfig()),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
