"""Unit tests for causal KAFI deployment inside run_allocation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Final

import polars as pl
import pytest

import src.policy.kafi_deployment as deployment_module
from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.schema import Dataset, spec_for
from src.policy.contribution_shape import ContributionShapeConfig
from src.policy.kafi_deployment import KafiDeploymentConfig
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


@pytest.mark.parametrize("scenario_id", ["SIM-D-external-base-recorded"])
def test_sim_d_external_base_recorded(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """SIM-D-external-base-recorded"""
    scripted: Final[list[float]] = [20.0, 80.0, 50.0]

    def fake_opportunity_score(**_kwargs: object) -> float:
        return scripted.pop(0)

    monkeypatch.setattr(deployment_module, "kafi_opportunity_score", fake_opportunity_score)
    result = run_allocation(
        AllocationConfig(
            policy=PolicyId.QQQ,
            start=_CONFIG_START,
            end=_CONFIG_END,
            monthly_contribution_krw=_CONTRIBUTION_KRW,
            kafi_deployment=KafiDeploymentConfig(),
        ),
        _prices_panel(),
        _fx_panel(),
        _cpi_panel(),
        macro=_macro_panel(),
    )

    assert len(result.snapshots) == 3
    assert all(snapshot.contribution_krw == pytest.approx(_CONTRIBUTION_KRW) for snapshot in result.snapshots)
    assert any(snapshot.reserve_krw > 0.0 for snapshot in result.snapshots)


@pytest.mark.parametrize("scenario_id", ["SIM-D-xor-modules"])
def test_sim_d_xor_modules(scenario_id: str) -> None:
    """SIM-D-xor-modules"""
    base: dict[str, object] = {
        "policy": PolicyId.QQQ,
        "start": _CONFIG_START,
        "end": _CONFIG_END,
        "monthly_contribution_krw": _CONTRIBUTION_KRW,
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_allocation(
            AllocationConfig(**base, contribution_shape=ContributionShapeConfig(), kafi_deployment=KafiDeploymentConfig()),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_allocation(
            AllocationConfig(**base, reserve=ReserveConfig(max_withhold=0.05), kafi_deployment=KafiDeploymentConfig()),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
