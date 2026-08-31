"""Unit tests for stateless adaptive contribution inside run_allocation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Final

import polars as pl
import pytest

import src.policy.adaptive_contribution as adaptive_module
from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.schema import Dataset, spec_for
from src.policy.adaptive_contribution import AdaptiveContributionConfig  # noqa: F401
from src.policy.contribution_shape import ContributionShapeConfig
from src.policy.kafi_deployment import KafiDeploymentConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import OPERATIONAL_POLICY_ID, PolicyError, PolicyId
from src.sim.allocation import AllocationConfig, apply_operational_contribution_lock, run_allocation

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


@pytest.mark.parametrize("scenario_id", ["SIM-ACG-variable-cashflow"])
def test_sim_acg_variable_cashflow(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """SIM-ACG-variable-cashflow"""
    scripted: Final[list[float]] = [0.0, 50.0, 100.0]

    def fake_opportunity_score(**_kwargs: object) -> float:
        return scripted.pop(0)

    monkeypatch.setattr(adaptive_module, "kafi_opportunity_score", fake_opportunity_score)
    result = run_allocation(
        AllocationConfig(
            policy=PolicyId.QQQ,
            start=_CONFIG_START,
            end=_CONFIG_END,
            monthly_contribution_krw=_CONTRIBUTION_KRW,
            adaptive_contribution=AdaptiveContributionConfig(),
        ),
        _prices_panel(),
        _fx_panel(),
        _cpi_panel(),
        macro=_macro_panel(),
    )

    assert [snapshot.contribution_krw for snapshot in result.snapshots] == pytest.approx(
        [0.0, _CONTRIBUTION_KRW, 2 * _CONTRIBUTION_KRW], abs=1e-6
    )
    assert sum(snapshot.contribution_krw for snapshot in result.snapshots) == pytest.approx(
        3 * _CONTRIBUTION_KRW, rel=1e-9
    )
    # CPI deflator is flat at 100 here, so the real total equals the nominal sum.
    assert result.total_contribution_real_krw == pytest.approx(3 * _CONTRIBUTION_KRW, rel=1e-9)
    assert all(snapshot.reserve_krw == 0.0 for snapshot in result.snapshots)


@pytest.mark.parametrize("scenario_id", ["SIM-ACG-boundaries"])
def test_sim_acg_boundaries(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """SIM-ACG-boundaries"""
    base: dict[str, object] = {
        "policy": PolicyId.QQQ,
        "start": _CONFIG_START,
        "end": _CONFIG_END,
        "monthly_contribution_krw": _CONTRIBUTION_KRW,
        "adaptive_contribution": AdaptiveContributionConfig(),
    }
    with pytest.raises(ValueError, match="cadence"):
        run_allocation(
            AllocationConfig(**base, cadence="month_open"),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_allocation(
            AllocationConfig(**base, reserve=ReserveConfig(max_withhold=0.05)),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_allocation(
            AllocationConfig(**base, contribution_shape=ContributionShapeConfig()),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_allocation(
            AllocationConfig(**base, kafi_deployment=KafiDeploymentConfig()),  # type: ignore[arg-type]
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )

    def zero_score(**_kwargs: object) -> float:
        return 0.0

    monkeypatch.setattr(adaptive_module, "kafi_opportunity_score", zero_score)
    with pytest.raises(PolicyError, match="all-zero"):
        run_allocation(
            AllocationConfig(**base),  # type: ignore[arg-type]
            _prices_panel(),
            _fx_panel(),
            _cpi_panel(),
            macro=_macro_panel(),
        )


@pytest.mark.parametrize("scenario_id", ["SIM-ACG-operational-lock"])
def test_sim_acg_operational_lock(scenario_id: str) -> None:
    """SIM-ACG-operational-lock legacy alias still locked to v5"""
    bare_qqq = AllocationConfig(
        policy=OPERATIONAL_POLICY_ID,
        start=_CONFIG_START,
        end=_CONFIG_END,
        monthly_contribution_krw=_CONTRIBUTION_KRW,
    )
    locked = apply_operational_contribution_lock(bare_qqq)
    assert locked.targets_override == {"QQQ": 0.9, "SOXX": 0.1}
    assert locked.adaptive_contribution is None

    vti = apply_operational_contribution_lock(
        AllocationConfig(
            policy=PolicyId.VTI,
            start=_CONFIG_START,
            end=_CONFIG_END,
            monthly_contribution_krw=_CONTRIBUTION_KRW,
        )
    )
    assert vti.adaptive_contribution is None

    with_overlay = apply_operational_contribution_lock(
        AllocationConfig(
            policy=OPERATIONAL_POLICY_ID,
            start=_CONFIG_START,
            end=_CONFIG_END,
            monthly_contribution_krw=_CONTRIBUTION_KRW,
            overlay=OverlayConfig(),
        )
    )
    assert with_overlay.adaptive_contribution is None


@pytest.mark.parametrize("scenario_id", ["SIM-ACG-operational-lock-v5"])
def test_sim_acg_operational_lock_v5(scenario_id: str) -> None:
    """SIM-ACG-operational-lock-v5"""
    bare_qqq = AllocationConfig(
        policy=OPERATIONAL_POLICY_ID,
        start=_CONFIG_START,
        end=_CONFIG_END,
        monthly_contribution_krw=_CONTRIBUTION_KRW,
    )
    locked = apply_operational_contribution_lock(bare_qqq)
    assert locked.targets_override == {"QQQ": 0.9, "SOXX": 0.1}
    assert locked.adaptive_contribution is None

def test_apply_operational_lock_is_flat_qqq90_soxx10() -> None:
    from datetime import date

    from src.policy.targets import OPERATIONAL_POLICY_ID, OPERATIONAL_TARGETS_OVERRIDE, PolicyId
    from src.sim.allocation import AllocationConfig, apply_operational_contribution_lock

    bare = AllocationConfig(
        policy=OPERATIONAL_POLICY_ID,
        start=date(2024, 1, 2),
        end=date(2024, 3, 28),
        monthly_contribution_krw=1_000_000.0,
    )
    locked = apply_operational_contribution_lock(bare)
    assert locked.targets_override == {"QQQ": 0.9, "SOXX": 0.1}
    assert locked.targets_override == OPERATIONAL_TARGETS_OVERRIDE
    assert locked.adaptive_contribution is None
    assert locked.kafi_deployment is None
    assert locked.contribution_shape is None
    skipped = apply_operational_contribution_lock(
        AllocationConfig(
            policy=PolicyId.VTI,
            start=date(2024, 1, 2),
            end=date(2024, 3, 28),
            monthly_contribution_krw=1_000_000.0,
        )
    )
    assert skipped.targets_override is None
    assert skipped.adaptive_contribution is None


