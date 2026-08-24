"""Pre-trade feasibility preflight: prove data readiness before any allocation runner."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from src.etf_manager.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.etf_manager.data.catalog import load_visible
from src.etf_manager.data.pit import AVAILABLE_AT, TS_DTYPE
from src.etf_manager.data.query import load_as_of
from src.etf_manager.data.schedule import DecisionPoint, build_decision_schedule
from src.etf_manager.data.schema import Dataset
from src.etf_manager.data.storage import UntrustedDatasetError
from src.etf_manager.etf.mapping import (
    MappingConfig,
    apply_etf_mapping,
)
from src.etf_manager.features.drawdown import trailing_price_drawdown
from src.etf_manager.features.momentum import trailing_compound_return
from src.etf_manager.features.returns import session_returns
from src.etf_manager.features.risk import trailing_simple_vol
from src.etf_manager.policy.overlay import OverlayConfig
from src.etf_manager.policy.reserve import ReserveConfig
from src.etf_manager.policy.targets import PolicyError, PolicyId, policy_sleeves
from src.etf_manager.validation.experiment import resolve_mapping, resolve_overlay, resolve_reserve

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from src.etf_manager.data.calendar import TradingCalendar
    from src.etf_manager.data.settings import DataSettings
    from src.etf_manager.validation.experiment import ExperimentSpec

logger = logging.getLogger(__name__)

__all__ = [
    "FeasibilityError",
    "FeasibilityReport",
    "FeasibilityViolation",
    "assert_experiment_feasible",
    "overlay_warmup_sessions",
    "require_feasibility",
    "reserve_warmup_sessions",
    "resolve_feasibility",
]


class FeasibilityError(ValueError):
    """Preflight proved the requested window cannot execute as specified.

    Subclasses :class:`ValueError` so existing CLI ``except (... ValueError)``
    branches map a failed preflight to exit code 1 without new handling.
    """

    def __init__(self, message: str, report: FeasibilityReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class FeasibilityViolation:
    """One preflight blocker; ``code`` is the stable machine-readable identity."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Preflight outcome; purely diagnostic and never rewritten into inputs."""

    requested_start: date
    requested_end: date
    warmup_sessions: int
    decision_points: int
    violations: tuple[FeasibilityViolation, ...]
    earliest_safe_start: date | None
    ingest_recommended_start: date | None


def overlay_warmup_sessions(overlay: OverlayConfig | None) -> int:
    """History depth in sessions the overlay features require; 0 without an overlay."""
    if overlay is None:
        return 0
    return max(overlay.trend_window, overlay.vol_window, overlay.drawdown_window)


def reserve_warmup_sessions(reserve: ReserveConfig | None) -> int:
    """History depth in sessions the reserve features require; 0 without a reserve."""
    if reserve is None:
        return 0
    return max(reserve.trend_window, reserve.drawdown_window)


def resolve_feasibility(
    *,
    start: date,
    end: date,
    fill_delay_sessions: int,
    mark_policies: tuple[PolicyId, ...],
    overlay: OverlayConfig | None,
    overlay_policies: tuple[PolicyId, ...],
    settings: DataSettings,
    reserve: ReserveConfig | None = None,
    mapping: MappingConfig | None = None,
    mapping_policies: tuple[PolicyId, ...] = (),
) -> FeasibilityReport:
    """Check that the requested window can trade without lookahead or data gaps.

    Loads PRICES/FX/CPI (plus MACRO only for a VIX-gated overlay and
    ETF_METADATA only for mapping) once at the requested schedule's last
    execution close — the allocator's own visibility cutoff — then evaluates CPI
    visibility, first/last execution marks (plus mapped implementations), the
    exact overlay feature stack at the first signal instant, and a full
    ``apply_etf_mapping`` dry run at that instant. No allocation logic runs here
    and no input date is ever clamped.

    Raises:
        UntrustedDatasetError: When any required catalog partition fails its
            lineage checks; propagated untouched, never wrapped.
        ValueError: On an invalid fill delay or naive schedule timestamps.
    """
    calendar = load_calendar(DEFAULT_CALENDAR_NAME)
    schedule = build_decision_schedule(start, end, fill_delay_sessions=fill_delay_sessions)
    warmup = max(overlay_warmup_sessions(overlay), reserve_warmup_sessions(reserve))
    violations: list[FeasibilityViolation] = []
    earliest_safe_start: date | None = None
    ingest_recommended_start: date | None = None

    if not schedule:
        violations.append(
            FeasibilityViolation(
                code="empty_schedule",
                message=f"empty decision schedule over [{start.isoformat()}, {end.isoformat()}]",
            )
        )
    else:
        first_point, last_point = schedule[0], schedule[-1]
        first_exec_close = calendar.close_ts(first_point.execution_session)
        last_exec_close = calendar.close_ts(last_point.execution_session)
        prices = load_visible(settings, Dataset.PRICES, last_exec_close)
        fx = load_visible(settings, Dataset.FX, last_exec_close)
        cpi = load_visible(settings, Dataset.CPI, last_exec_close)
        need_macro = overlay is not None and overlay.vix_threshold is not None
        macro = load_visible(settings, Dataset.MACRO, last_exec_close) if need_macro else None
        metadata: pl.DataFrame | None = None
        if mapping is not None:
            try:
                metadata = load_visible(settings, Dataset.ETF_METADATA, last_exec_close)
            except UntrustedDatasetError as exc:
                violations.append(
                    FeasibilityViolation(
                        code="etf_metadata",
                        message=f"ETF_METADATA catalog unavailable for mapping preflight: {exc}",
                    )
                )
        mark_tickers = _union_sleeves(mark_policies)
        overlay_tickers = _union_sleeves(overlay_policies)
        if mapping is not None:
            known = set(mark_tickers)
            mapped_choices: dict[str, None] = {}
            for sleeve in _union_sleeves(mapping_policies):
                for ticker in mapping.candidates.get(sleeve, ()):
                    mapped_choices.setdefault(ticker)
            mark_tickers = (*mark_tickers, *(ticker for ticker in mapped_choices if ticker not in known))

        cpi_violation = _cpi_violation(cpi, first_exec_close)
        if cpi_violation is not None:
            violations.append(cpi_violation)
        exec_closes = [(first_point.execution_session, first_exec_close)]
        if last_point.execution_session != first_point.execution_session:
            exec_closes.append((last_point.execution_session, last_exec_close))
        for session, close_ts in exec_closes:
            visible_prices = load_as_of(prices, Dataset.PRICES, close_ts)
            visible_fx = load_as_of(fx, Dataset.FX, close_ts)
            violations.extend(_mark_violations(visible_prices, visible_fx, mark_tickers, session))

        if overlay is not None:
            warmup_violation = _overlay_warmup_violation(prices, overlay, overlay_tickers, first_point.signal_at)
            if warmup_violation is not None:
                violations.append(warmup_violation)
            if need_macro and macro is not None:
                vix_violation = _vix_violation(macro, overlay, first_point.signal_at)
                if vix_violation is not None:
                    violations.append(vix_violation)

        if mapping is not None and metadata is not None:
            mapping_violation = _mapping_warmup_violation(
                prices, metadata, mapping, _union_sleeves(mapping_policies), first_point.signal_at
            )
            if mapping_violation is not None:
                violations.append(mapping_violation)

        # Informational only: never assigned back to spec.start or any config.
        ingest_recommended_start = (
            _sessions_before(calendar, first_point.signal_session, warmup)
            if overlay is not None
            else first_point.execution_session
        )
        earliest_safe_start = _earliest_safe_start(
            calendar=calendar,
            prices=prices,
            cpi=cpi,
            macro=macro,
            overlay=overlay,
            overlay_tickers=overlay_tickers,
            start=start,
            end=end,
            fill_delay_sessions=fill_delay_sessions,
        )

    logger.info(
        "[DATA] event=feasibility_resolved start=%s end=%s points=%d warmup=%d violations=%d",
        start.isoformat(),
        end.isoformat(),
        len(schedule),
        warmup,
        len(violations),
    )
    return FeasibilityReport(
        requested_start=start,
        requested_end=end,
        warmup_sessions=warmup,
        decision_points=len(schedule),
        violations=tuple(violations),
        earliest_safe_start=earliest_safe_start,
        ingest_recommended_start=ingest_recommended_start,
    )


def require_feasibility(
    *,
    start: date,
    end: date,
    fill_delay_sessions: int,
    mark_policies: tuple[PolicyId, ...],
    overlay: OverlayConfig | None,
    overlay_policies: tuple[PolicyId, ...],
    settings: DataSettings,
    reserve: ReserveConfig | None = None,
    mapping: MappingConfig | None = None,
    mapping_policies: tuple[PolicyId, ...] = (),
) -> FeasibilityReport:
    """Resolve feasibility and fail closed with every violation code when blocked.

    Raises:
        FeasibilityError: When any violation was found; the message names every code.
        UntrustedDatasetError: Propagated untouched from the catalog read path.
    """
    report = resolve_feasibility(
        start=start,
        end=end,
        fill_delay_sessions=fill_delay_sessions,
        mark_policies=mark_policies,
        overlay=overlay,
        overlay_policies=overlay_policies,
        settings=settings,
        reserve=reserve,
        mapping=mapping,
        mapping_policies=mapping_policies,
    )
    if report.violations:
        codes = ", ".join(violation.code for violation in report.violations)
        raise FeasibilityError(f"feasibility preflight failed: [{codes}]", report=report)
    return report


def assert_experiment_feasible(spec: ExperimentSpec, settings: DataSettings) -> FeasibilityReport:
    """Preflight an experiment JSON exactly as its arms would run.

    Mirrors the ablation/campaign arm configs (``fill_delay_sessions == 1``,
    baseline plus candidate marks, candidate-only overlay and mapping sleeves);
    the spec is never mutated.

    Raises:
        FeasibilityError: When the experiment window cannot execute.
        UntrustedDatasetError: Propagated untouched from the catalog read path.
    """
    overlay = resolve_overlay(spec)
    reserve = resolve_reserve(spec)
    mapping = resolve_mapping(spec)
    return require_feasibility(
        start=spec.start,
        end=spec.end,
        fill_delay_sessions=1,
        mark_policies=(spec.baseline.policy, *(candidate.policy for candidate in spec.candidates)),
        overlay=overlay,
        overlay_policies=tuple(candidate.policy for candidate in spec.candidates) if overlay is not None else (),
        settings=settings,
        reserve=reserve,
        mapping=mapping,
        mapping_policies=tuple(candidate.policy for candidate in spec.candidates) if mapping is not None else (),
    )


def _union_sleeves(policies: Iterable[PolicyId]) -> tuple[str, ...]:
    """First-seen ordered union of sleeve tickers across policies."""
    union: dict[str, None] = {}
    for policy in policies:
        for ticker in policy_sleeves(policy):
            union.setdefault(ticker)
    return tuple(union)


def _finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _cpi_violation(cpi: pl.DataFrame, close_ts: datetime) -> FeasibilityViolation | None:
    """Same positive-finite CPI predicate as the allocator, reported instead of raised."""
    rows = load_as_of(cpi, Dataset.CPI, close_ts).filter(pl.col("value").is_finite() & (pl.col("value") > 0.0))
    if rows.is_empty():
        return FeasibilityViolation(
            code="cpi", message=f"no positive finite CPI level visible at {close_ts.isoformat()}"
        )
    return None


def _mark_violations(
    visible_prices: pl.DataFrame,
    visible_fx: pl.DataFrame,
    tickers: tuple[str, ...],
    session: date,
) -> list[FeasibilityViolation]:
    """Execution-close coverage of FX and every mark sleeve at one session."""
    violations: list[FeasibilityViolation] = []
    fx_rows = visible_fx.filter(pl.col("date") == session)
    fx_value = fx_rows.item(0, "usdkrw") if not fx_rows.is_empty() else None
    if not _finite_positive(fx_value):
        violations.append(
            FeasibilityViolation(
                code="fx",
                message=f"missing or non-positive usdkrw on {session.isoformat()} at its execution close",
            )
        )
    for ticker in tickers:
        rows = visible_prices.filter((pl.col("ticker") == ticker) & (pl.col("date") == session))
        value = rows.item(0, "adjusted_close") if not rows.is_empty() else None
        if not _finite_positive(value):
            violations.append(
                FeasibilityViolation(
                    code="price",
                    message=f"missing or non-positive adjusted_close for {ticker!r} on {session.isoformat()}",
                )
            )
    return violations


def _overlay_warmup_violation(
    prices: pl.DataFrame,
    overlay: OverlayConfig,
    tickers: tuple[str, ...],
    signal_at: datetime,
) -> FeasibilityViolation | None:
    """Run the exact overlay feature stack at the first signal; windows are never shortened."""
    for ticker in tickers:
        try:
            returns = session_returns(prices, ticker=ticker)
            trailing_compound_return(returns, as_of_ts=signal_at, window=overlay.trend_window)
            trailing_simple_vol(returns, as_of_ts=signal_at, window=overlay.vol_window)
            trailing_price_drawdown(prices, ticker=ticker, as_of_ts=signal_at, window=overlay.drawdown_window)
        except ValueError as exc:
            return FeasibilityViolation(
                code="overlay_warmup", message=f"overlay history insufficient for {ticker!r}: {exc}"
            )
    return None


def _mapping_warmup_violation(
    prices: pl.DataFrame,
    metadata: pl.DataFrame,
    mapping: MappingConfig,
    sleeves: tuple[str, ...],
    signal_at: datetime,
) -> FeasibilityViolation | None:
    """Run the exact mapping stack on equal sleeve weights at the first signal."""
    if not sleeves:
        return FeasibilityViolation(
            code="mapping_warmup", message="mapping arms own no sleeve tickers to map at the signal instant"
        )
    targets = dict.fromkeys(sleeves, 1.0 / len(sleeves))
    try:
        apply_etf_mapping(targets, prices, metadata, signal_at, mapping, {})
    except (PolicyError, ValueError) as exc:
        return FeasibilityViolation(
            code="mapping_warmup", message=f"ETF mapping failed closed at the first signal: {exc}"
        )
    return None


def _vix_violation(macro: pl.DataFrame, overlay: OverlayConfig, signal_at: datetime) -> FeasibilityViolation | None:
    """A finite VIX row must be published at or before the first signal instant."""
    cutoff = signal_at.astimezone(UTC)
    visible = macro.filter(
        (pl.col("series_id") == overlay.vix_series_id)
        & (pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
        & pl.col("value").is_finite()
    )
    if visible.is_empty():
        return FeasibilityViolation(
            code="vix",
            message=f"no finite {overlay.vix_series_id!r} macro row visible at {cutoff.isoformat()}",
        )
    return None


def _point_passes(
    point: DecisionPoint,
    *,
    calendar: TradingCalendar,
    prices: pl.DataFrame,
    cpi: pl.DataFrame,
    macro: pl.DataFrame | None,
    overlay: OverlayConfig | None,
    overlay_tickers: tuple[str, ...],
) -> bool:
    """Whether one decision point clears CPI plus overlay (+VIX gate) checks."""
    close_ts = calendar.close_ts(point.execution_session)
    if _cpi_violation(cpi, close_ts) is not None:
        return False
    if overlay is None:
        return True
    if _overlay_warmup_violation(prices, overlay, overlay_tickers, point.signal_at) is not None:
        return False
    if overlay.vix_threshold is not None and macro is not None:
        return _vix_violation(macro, overlay, point.signal_at) is None
    return True


def _earliest_safe_start(
    *,
    calendar: TradingCalendar,
    prices: pl.DataFrame,
    cpi: pl.DataFrame,
    macro: pl.DataFrame | None,
    overlay: OverlayConfig | None,
    overlay_tickers: tuple[str, ...],
    start: date,
    end: date,
    fill_delay_sessions: int,
) -> date | None:
    """Earliest month-end signal whose point passes CPI+overlay(+vix); informational."""
    raw_min = prices.get_column("date").min()
    lower_bound = raw_min if isinstance(raw_min, date) else start
    for point in build_decision_schedule(lower_bound, end, fill_delay_sessions=fill_delay_sessions):
        if _point_passes(
            point,
            calendar=calendar,
            prices=prices,
            cpi=cpi,
            macro=macro,
            overlay=overlay,
            overlay_tickers=overlay_tickers,
        ):
            return point.signal_session
    return None


def _sessions_before(calendar: TradingCalendar, session: date, count: int) -> date:
    """The exchange session exactly ``count`` sessions before ``session``."""
    if count <= 0:
        return session
    span_days = 7 * (count // 5 + 2)
    while True:
        window = calendar.sessions(session - timedelta(days=span_days), session)
        if len(window) > count:
            return window[len(window) - 1 - count]
        span_days *= 2
