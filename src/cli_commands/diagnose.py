# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Diagnose runners."""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Final, cast

from src.analytics.accumulation_alpha import screen_qqq_accumulation
from src.analytics.adaptive_hp_screen import make_hp_wf_runner, screen_adaptive_contribution_hp
from src.analytics.blends import compare_qqq_blends
from src.analytics.cadence import compare_qqq_cadence
from src.analytics.metrics import XirrError
from src.analytics.regimes import QQQ_REGIME_WINDOWS, compare_policy_regimes
from src.analytics.reserve_usage import compare_qqq_reserve
from src.analytics.us_vehicles import compare_vehicle_dca, profile_us_vehicles
from src.data.calendar import DEFAULT_CALENDAR_NAME, clamp_inclusive_session_range, load_calendar
from src.data.catalog import latest_artifact, load_visible
from src.data.schedule import build_decision_schedule
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DataStore, UntrustedDatasetError
from src.features.kafi import earliest_kafi_signal_session, kafi_score
from src.policy.reserve import ReserveConfig
from src.policy.targets import PolicyError, PolicyId
from src.policy.contribution_shape import ContributionShapeConfig
from src.sim.allocation import AllocationConfig, AllocationDataError, run_allocation_from_store
from src.sim.baseline import BaselineConfig, BaselineDataError, BaselineId

logger = logging.getLogger(__name__)

_DIAGNOSE_VEHICLES: Final[tuple[str, ...]] = ("VTI", "IVV", "QQQ")
_RESERVE_SCHEDULES: Final[dict[str, ReserveConfig | None]] = {
    "v1": None,
    "v2": ReserveConfig(schedule="v2", max_withhold=0.10),
    "v3": ReserveConfig(
        schedule="v3", max_withhold=0.10, min_invest_multiplier=0.70, max_invest_multiplier=3.0
    ),
    "v4": ReserveConfig(
        schedule="v4",
        max_withhold=0.10,
        min_invest_multiplier=0.70,
        max_invest_multiplier=3.0,
        reserve_max_months=2.0,
    ),
}
_ACCUMULATION_TICKER: Final[str] = "QQQ"
_KAFI_SHAPE_CONFIG: Final[ContributionShapeConfig] = ContributionShapeConfig()


def run_diagnose_us_vehicles_command(
    *,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
) -> int:
    """Log factor profiles and identical-cashflow DCA metrics for VTI/IVV/QQQ."""
    try:
        schedule = build_decision_schedule(start, end, fill_delay_sessions=1)
        if not schedule:
            raise BaselineDataError(f"empty decision schedule over [{start.isoformat()}, {end.isoformat()}]")
        cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
        prices = load_visible(settings, Dataset.PRICES, cutoff)
        fx = load_visible(settings, Dataset.FX, cutoff)
        cpi = load_visible(settings, Dataset.CPI, cutoff)
        factors = load_visible(settings, Dataset.FACTORS, cutoff)
        profiles = profile_us_vehicles(prices, factors, tickers=_DIAGNOSE_VEHICLES, signal_at=cutoff)
        base = BaselineConfig(
            baseline=BaselineId.B1_US,
            ticker=_DIAGNOSE_VEHICLES[0],
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
        )
        paths = compare_vehicle_dca(base, prices, fx, cpi, tickers=_DIAGNOSE_VEHICLES)
    except (BaselineDataError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_us_vehicles_failed reason_type=%s", type(exc).__name__)
        return 1
    for profile in profiles:
        logger.info(
            "[DATA] event=vehicle_factor_profile ticker=%s alpha=%.6f mkt_rf=%.4f smb=%.4f hml=%.4f rmw=%.4f cma=%.4f mom=%.4f",
            profile.ticker,
            profile.alpha,
            profile.mkt_rf,
            profile.smb,
            profile.hml,
            profile.rmw,
            profile.cma,
            profile.mom,
        )
    for path in paths:
        logger.info(
            "[DATA] event=vehicle_dca_done ticker=%s terminal_krw=%.3f terminal_real_krw=%.3f xirr=%.6f steps=%d",
            path.ticker,
            path.result.terminal_wealth_krw,
            path.result.terminal_wealth_real_krw,
            path.result.xirr,
            len(path.result.snapshots),
        )
    return 0


def run_diagnose_qqq_regimes_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log QQQ/VTI real-terminal ratios per regime window on identical cashflows."""
    try:
        comparisons = compare_policy_regimes(
            contribution_krw=float(contribution_krw),
            runner=lambda config: run_allocation_from_store(config, settings),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_regimes_failed reason_type=%s", type(exc).__name__)
        return 1
    for comparison in comparisons:
        logger.info(
            "[DATA] event=qqq_regime_ratio name=%s start=%s end=%s ratio=%.6f vti_steps=%d qqq_steps=%d",
            comparison.name,
            comparison.start.isoformat(),
            comparison.end.isoformat(),
            comparison.candidate.terminal_wealth_real_krw / comparison.baseline.terminal_wealth_real_krw,
            len(comparison.baseline.snapshots),
            len(comparison.candidate.snapshots),
        )
    return 0


def run_diagnose_qqq_blends_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log QQQ blend-recipe real-terminal ratios per regime window on identical cashflows."""
    try:
        comparisons = compare_qqq_blends(
            contribution_krw=float(contribution_krw),
            runner=lambda config: run_allocation_from_store(config, settings),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_blends_failed reason_type=%s", type(exc).__name__)
        return 1
    for comparison in comparisons:
        logger.info(
            "[DATA] event=qqq_blend_ratio name=%s recipe=%s real_terminal_krw=%.2f mdd=%.4f"
            " ratio_vs_qqq=%.6f ratio_vs_vti=%.6f",
            comparison.name,
            comparison.recipe,
            comparison.candidate.terminal_wealth_real_krw,
            comparison.candidate.max_drawdown,
            comparison.candidate.terminal_wealth_real_krw / comparison.qqq_baseline.terminal_wealth_real_krw,
            comparison.candidate.terminal_wealth_real_krw / comparison.vti_baseline.terminal_wealth_real_krw,
        )
    return 0


def run_diagnose_qqq_reserve_command(
    *, contribution_krw: float, settings: DataSettings, reserve_schedule: str = "v1"
) -> int:
    """Log QQQ reserved-arm ratios, MDD, and reconstructed reserve usage per regime window."""
    try:
        if reserve_schedule not in _RESERVE_SCHEDULES:
            raise ValueError(f"unknown reserve schedule {reserve_schedule!r}")
        comparisons = compare_qqq_reserve(
            contribution_krw=float(contribution_krw),
            reserve=_RESERVE_SCHEDULES[reserve_schedule],
            runner=lambda config: run_allocation_from_store(config, settings),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_reserve_failed reason_type=%s", type(exc).__name__)
        return 1
    for comparison in comparisons:
        logger.info(
            "[DATA] event=qqq_reserve_ratio name=%s start=%s end=%s"
            " reserved_real_terminal_krw=%.2f plain_real_terminal_krw=%.2f ratio_reserved_vs_plain=%.6f"
            " reserved_mdd=%.4f plain_mdd=%.4f withheld_total=%.2f redeployed_total=%.2f"
            " extra_investment_ratio=%.8f cash_drag_ratio=%.8f reserve_idle_months=%d"
            " reserve_deployment_events=%d steps=%d",
            comparison.name,
            comparison.start.isoformat(),
            comparison.end.isoformat(),
            comparison.reserved.terminal_wealth_real_krw,
            comparison.plain.terminal_wealth_real_krw,
            comparison.reserved.terminal_wealth_real_krw / comparison.plain.terminal_wealth_real_krw,
            comparison.reserved.max_drawdown,
            comparison.plain.max_drawdown,
            comparison.reserved_usage.withheld_total,
            comparison.reserved_usage.redeployed_total,
            comparison.reserved_usage.extra_investment_ratio,
            comparison.reserved_usage.cash_drag_ratio,
            comparison.reserved_usage.reserve_idle_months,
            comparison.reserved_usage.reserve_deployment_events,
            len(comparison.reserved.snapshots),
        )
    return 0


def run_diagnose_qqq_cadence_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log QQQ month-open-cadence real-terminal ratios versus the monthly cadence per regime window."""
    try:
        comparisons = compare_qqq_cadence(
            contribution_krw=float(contribution_krw),
            runner=lambda config: run_allocation_from_store(config, settings),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_cadence_failed reason_type=%s", type(exc).__name__)
        return 1
    for comparison in comparisons:
        logger.info(
            "[DATA] event=qqq_cadence_ratio name=%s start=%s end=%s"
            " month_open_real_terminal_krw=%.2f monthly_real_terminal_krw=%.2f"
            " ratio_month_open_vs_monthly=%.6f"
            " twice_monthly_real_terminal_krw=%.2f ratio_twice_monthly_vs_monthly=%.6f steps=%d",
            comparison.name,
            comparison.start.isoformat(),
            comparison.end.isoformat(),
            comparison.month_open.terminal_wealth_real_krw,
            comparison.monthly.terminal_wealth_real_krw,
            comparison.month_open.terminal_wealth_real_krw / comparison.monthly.terminal_wealth_real_krw,
            comparison.twice_monthly.terminal_wealth_real_krw,
            comparison.twice_monthly.terminal_wealth_real_krw / comparison.monthly.terminal_wealth_real_krw,
            len(comparison.month_open.snapshots),
        )
    return 0


def run_diagnose_qqq_accumulation_alpha_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log QQQ buy-cadence accumulation-screen ratios versus month-end fills."""
    try:
        if float(contribution_krw) <= 0.0:
            raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
        prices = DataStore(settings).read_normalized(
            latest_artifact(settings, Dataset.PRICES), spec_for(Dataset.PRICES)
        )
        ticker_rows = prices.filter(prices.get_column("ticker") == _ACCUMULATION_TICKER)
        if ticker_rows.is_empty():
            raise ValueError(f"ticker {_ACCUMULATION_TICKER!r} missing from catalog prices")
        start_raw = ticker_rows.get_column("date").min()
        end_raw = ticker_rows.get_column("date").max()
        if start_raw is None or end_raw is None:
            raise ValueError(f"ticker {_ACCUMULATION_TICKER!r} has no price dates")
        report = screen_qqq_accumulation(
            prices=prices,
            ticker=_ACCUMULATION_TICKER,
            start=cast(date, start_raw),
            end=cast(date, end_raw),
            monthly_contribution=float(contribution_krw),
        )
    except (PolicyError, UntrustedDatasetError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_accumulation_alpha_failed reason_type=%s", type(exc).__name__)
        return 1
    for row in report.rows:
        logger.info(
            "[DATA] event=qqq_accumulation_arm arm=%s verdict=%s tw=%.2f ratio_vs_month_end=%.6f"
            " ci_low=%.6f ci_high=%.6f mean_log_gap=%s log_fill_p=%s",
            row.name,
            str(row.verdict),
            row.terminal_wealth,
            row.ratio_vs_month_end,
            row.bootstrap_ci_low,
            row.bootstrap_ci_high,
            row.mean_log_fill_gap_vs_end,
            row.log_fill_p_value,
        )
    logger.info(
        "[DATA] event=qqq_accumulation_screen_done ticker=%s start=%s end=%s usable_months=%d"
        " operational_unlock=%s recommended_research_arm=%s",
        report.ticker,
        report.start.isoformat(),
        report.end.isoformat(),
        report.usable_months,
        report.operational_unlock,
        report.recommended_research_arm,
    )
    return 0


def run_diagnose_qqq_kafi_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log the KAFI path and shaped-versus-flat real-TW ratios per regime window."""
    try:
        if float(contribution_krw) <= 0.0:
            raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
        store = DataStore(settings)
        prices = store.read_normalized(latest_artifact(settings, Dataset.PRICES), spec_for(Dataset.PRICES))
        fx = store.read_normalized(latest_artifact(settings, Dataset.FX), spec_for(Dataset.FX))
        macro = store.read_normalized(latest_artifact(settings, Dataset.MACRO), spec_for(Dataset.MACRO))
        qqq_rows = prices.filter(prices.get_column("ticker") == _ACCUMULATION_TICKER)
        if qqq_rows.is_empty():
            raise ValueError(f"ticker {_ACCUMULATION_TICKER!r} missing from catalog prices")
        start_raw = qqq_rows.get_column("date").min()
        end_raw = qqq_rows.get_column("date").max()
        if start_raw is None or end_raw is None:
            raise ValueError(f"ticker {_ACCUMULATION_TICKER!r} has no price dates")
        catalog_start = cast(date, start_raw)
        catalog_end = cast(date, end_raw)
        calendar = load_calendar(DEFAULT_CALENDAR_NAME)
        catalog_start, catalog_end = clamp_inclusive_session_range(calendar, catalog_start, catalog_end)
        config = _KAFI_SHAPE_CONFIG
        feasible_start = earliest_kafi_signal_session(
            prices=prices,
            fx=fx,
            macro=macro,
            equity_ticker=config.equity_ticker,
            bond_ticker=config.bond_ticker,
            start=catalog_start,
            end=catalog_end,
            rank_window=config.rank_window,
            credit_series_id=config.credit_series_id,
        )
        if feasible_start is None:
            raise ValueError(
                f"catalog lacks enough PIT history for KAFI credit series {config.credit_series_id!r}"
            )
        windows = tuple(
            (name, window_start, window_end)
            for name, window_start, window_end in QQQ_REGIME_WINDOWS
            if window_start >= catalog_start and window_end <= catalog_end
        ) or (("catalog", catalog_start, catalog_end),)
        logged = False
        for name, window_start, window_end in windows:
            effective_start = max(window_start, feasible_start)
            if effective_start > window_end:
                continue
            flat_result = run_allocation_from_store(
                AllocationConfig(
                    policy=PolicyId.QQQ,
                    start=effective_start,
                    end=window_end,
                    monthly_contribution_krw=float(contribution_krw),
                ),
                settings,
            )
            shaped_result = run_allocation_from_store(
                AllocationConfig(
                    policy=PolicyId.QQQ,
                    start=effective_start,
                    end=window_end,
                    monthly_contribution_krw=float(contribution_krw),
                    contribution_shape=config,
                ),
                settings,
            )
            logged = True
            logger.info(
                "[DATA] event=qqq_kafi_ratio name=%s start=%s end=%s"
                " flat_real_terminal_krw=%.2f shaped_real_terminal_krw=%.2f"
                " ratio_shaped_vs_flat=%.6f flat_mdd=%.4f shaped_mdd=%.4f steps=%d",
                name,
                effective_start.isoformat(),
                window_end.isoformat(),
                flat_result.terminal_wealth_real_krw,
                shaped_result.terminal_wealth_real_krw,
                (
                    shaped_result.terminal_wealth_real_krw / flat_result.terminal_wealth_real_krw
                    if flat_result.terminal_wealth_real_krw != 0.0
                    else float("nan")
                ),
                flat_result.max_drawdown,
                shaped_result.max_drawdown,
                len(shaped_result.snapshots),
            )
            band = config.max_multiplier - config.min_multiplier
            for point in build_decision_schedule(effective_start, window_end):
                score = kafi_score(
                    prices=prices,
                    fx=fx,
                    macro=macro,
                    equity_ticker=config.equity_ticker,
                    bond_ticker=config.bond_ticker,
                    signal_at=point.signal_at,
                    rank_window=config.rank_window,
                    credit_series_id=config.credit_series_id,
                )
                multiplier = config.min_multiplier + band * (100.0 - score) / 100.0
                logger.info(
                    "[DATA] event=kafi_path signal_session=%s score=%.2f multiplier=%.4f",
                    point.signal_session.isoformat(),
                    score,
                    multiplier,
                )
        if not logged:
            raise ValueError("no QQQ regime window remains after KAFI warmup clamping")
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_kafi_failed reason_type=%s", type(exc).__name__)
        return 1
    return 0


def run_diagnose_compound_dca_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Run compound DCA tournament (QQQ vs QQQ90/SOXX10 flat vs adaptive); reporting only."""
    try:
        if not math.isfinite(float(contribution_krw)) or float(contribution_krw) <= 0.0:
            raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
        from src.analytics.compound_dca import compare_compound_dca

        report = compare_compound_dca(
            runner=lambda config: run_allocation_from_store(config, settings),
            contribution_krw=float(contribution_krw),
        )
        for row in report.rows:
            logger.info(
                "[DATA] event=compound_dca_arm arm=%s terminal_real_krw=%.2f total_contribution_real_krw=%.2f real_gain=%.2f xirr=%.6f mdd=%.4f",
                row.arm_id,
                row.terminal_wealth_real_krw,
                row.total_contribution_real_krw,
                row.real_gain,
                row.xirr,
                row.max_drawdown,
            )
        logger.info(
            "[DATA] event=compound_dca_done champion=%s mdd_feasible_champion=%s mdd_baseline=%s mdd_slack=%.4f operational_unlock=false rows=%d",
            report.champion_arm_id,
            report.mdd_feasible_champion_arm_id,
            report.mdd_baseline_arm_id,
            report.mdd_slack,
            len(report.rows),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_compound_dca_failed reason_type=%s", type(exc).__name__)
        return 1
    return 0


def run_diagnose_qqq_adaptive_hp_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Run the adaptive HP neighbourhood screen versus operational v5; reporting only."""
    try:
        if not math.isfinite(float(contribution_krw)) or float(contribution_krw) <= 0.0:
            raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")

        wf_runner = make_hp_wf_runner(settings, contribution_krw=float(contribution_krw))
        report = screen_adaptive_contribution_hp(
            contribution_krw=float(contribution_krw),
            wf_runner=wf_runner,
        )
        for row in report.rows:
            logger.info(
                "[DATA] event=qqq_adaptive_hp_arm downside=%.4f upside=%.4f dispersion=%.4f deadband=%.4f ratio=%.6f adopted=%s verdict=%s",
                row.downside_power,
                row.upside_power,
                row.dispersion,
                row.neutral_deadband,
                row.pooled_tw_ratio,
                row.process_adopted_vs_baseline,
                str(row.verdict),
            )
        if report.champion is not None:
            c = report.champion
            logger.info(
                "[DATA] event=qqq_adaptive_hp_champion downside=%.4f upside=%.4f dispersion=%.4f deadband=%.4f ratio=%.6f",
                c.downside_power,
                c.upside_power,
                c.dispersion,
                c.neutral_deadband,
                c.pooled_tw_ratio,
            )
        else:
            logger.info("[DATA] event=qqq_adaptive_hp_champion none")
        logger.info(
            "[DATA] event=qqq_adaptive_hp_done evaluations=%d operational_unlock=%s champion=%s",
            report.evaluations,
            report.operational_unlock,
            "none" if report.champion is None else f"{report.champion.downside_power:.4f}/{report.champion.upside_power:.4f}/{report.champion.dispersion:.4f}/{report.champion.neutral_deadband:.4f}",
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_adaptive_hp_failed reason_type=%s", type(exc).__name__)
        return 1
    return 0
