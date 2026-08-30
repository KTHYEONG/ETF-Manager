# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Simulation run commands."""

from __future__ import annotations

import logging
from datetime import date

from src.analytics.metrics import XirrError
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.etf.mapping import MappingConfig
from src.execution.broker import replay_paper
from src.execution.orders import ExecutionError, orders_from_snapshots
from src.policy.currency import CurrencyConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import PolicyError, PolicyId, policy_sleeves
from src.policy.tilt import FactorTilt
from src.sim.allocation import (
    AllocationConfig,
    AllocationDataError,
    apply_operational_contribution_lock,
    run_allocation_from_store,
)
from src.sim.baseline import BaselineConfig, BaselineDataError, BaselineId, run_baseline_from_store
from src.validation.feasibility import require_feasibility

logger = logging.getLogger(__name__)


def run_baseline_command(
    *,
    baseline_id: str,
    ticker: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
) -> int:
    """Run a stored-data baseline and log terminal KRW / XIRR / MDD."""
    config = BaselineConfig(
        baseline=BaselineId.parse(baseline_id),
        ticker=ticker,
        start=start,
        end=end,
        monthly_contribution_krw=float(contribution_krw),
        fill_delay_sessions=1,
        commission_bps=0.0,
    )
    try:
        result = run_baseline_from_store(config, settings)
    except (BaselineDataError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=baseline_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=baseline_cli_done terminal_krw=%.3f xirr=%.6f mdd=%.4f terminal_real_krw=%.3f xirr_real=%.6f ticker=%s steps=%d",
        result.terminal_wealth_krw,
        result.xirr,
        result.max_drawdown,
        result.terminal_wealth_real_krw,
        result.xirr_real,
        ticker,
        len(result.snapshots),
    )
    return 0


def run_policy_command(
    *,
    policy_id: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
    tilt: FactorTilt | None = None,
    rebalance_band: float | None = None,
    overlay: OverlayConfig | None = None,
    reserve: ReserveConfig | None = None,
    currency: CurrencyConfig | None = None,
    mapping: MappingConfig | None = None,
) -> int:
    """Run a stored-data strategic allocation and log terminal KRW / XIRR / MDD."""
    if rebalance_band is not None and not 0.0 <= rebalance_band < 1.0:
        raise ValueError(f"rebalance_band must lie in [0, 1), got {rebalance_band!r}")
    config = apply_operational_contribution_lock(
        AllocationConfig(
            policy=PolicyId.parse(policy_id),
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
            fill_delay_sessions=1,
            commission_bps=0.0,
            tilt=tilt,
            rebalance_band=rebalance_band,
            overlay=overlay,
            reserve=reserve,
            currency=currency,
            mapping=mapping,
        )
    )
    try:
        require_feasibility(
            start=config.start,
            end=config.end,
            fill_delay_sessions=config.fill_delay_sessions,
            mark_policies=(config.policy,),
            overlay=config.overlay,
            overlay_policies=(config.policy,) if config.overlay is not None else (),
            settings=settings,
            reserve=config.reserve,
            mapping=config.mapping,
            mapping_policies=(config.policy,) if config.mapping is not None else (),
            currency=config.currency,
        )
        result = run_allocation_from_store(config, settings)
    except (
        AllocationDataError,
        PolicyError,
        BaselineDataError,
        UntrustedDatasetError,
        XirrError,
        ValueError,
    ) as exc:
        logger.error("[DATA] event=policy_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=policy_cli_done policy=%s terminal_krw=%.3f xirr=%.6f mdd=%.4f"
        " terminal_real_krw=%.3f xirr_real=%.6f sleeves=%d steps=%d",
        str(config.policy),
        result.terminal_wealth_krw,
        result.xirr,
        result.max_drawdown,
        result.terminal_wealth_real_krw,
        result.xirr_real,
        len(policy_sleeves(config.policy)),
        len(result.snapshots),
    )
    return 0


def run_paper_command(
    *,
    policy_id: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
) -> int:
    """Replay a stored-data policy onto PaperBroker and fail closed on lot mismatch."""
    config = apply_operational_contribution_lock(
        AllocationConfig(
            policy=PolicyId(policy_id),
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
            fill_delay_sessions=1,
            commission_bps=0.0,
        )
    )
    try:
        result = run_allocation_from_store(config, settings)
        replay_paper(result)
        order_count = len(orders_from_snapshots(result.snapshots))
    except (
        AllocationDataError,
        BaselineDataError,
        ExecutionError,
        PolicyError,
        UntrustedDatasetError,
        XirrError,
        ValueError,
    ) as exc:
        logger.error("[DATA] event=paper_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=paper_cli_done policy=%s orders=%d steps=%d",
        str(config.policy),
        order_count,
        len(result.snapshots),
    )
    return 0
