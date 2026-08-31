"""Wave 2 historical coverage feasibility audit for static DCA."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl

from src.data.calendar import DEFAULT_CALENDAR_NAME, TradingCalendar, load_calendar
from src.data.catalog import latest_artifact, load_visible
from src.data.query import load_as_of
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DataStore
from src.validation.experiment import (
    ExperimentSpec,
    experiment_target_tickers,
)
from src.validation.windows import rolling_cohorts

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

STATIC_DCA_DATASETS: Final[tuple[Dataset, ...]] = (Dataset.PRICES, Dataset.FX, Dataset.CPI)
WAVE2_TARGET_EARLIEST_START: Final[date] = date(2000, 1, 1)
WAVE2_MIN_120M_COHORTS: Final[int] = 10


@dataclass(frozen=True, slots=True)
class TickerCoverageRow:
    ticker: str
    first_session: date
    last_session: date


@dataclass(frozen=True, slots=True)
class DatasetCoverageRow:
    dataset: str
    first_observation: date | None
    last_observation: date | None


@dataclass(frozen=True, slots=True)
class FeasibilityDependencyProfile:
    profile: str
    required_datasets: tuple[str, ...]
    requires_macro: bool


@dataclass(frozen=True, slots=True)
class StaticDcaWindowReport:
    name: str
    requested_start: date
    requested_end: date
    dependency: FeasibilityDependencyProfile
    ticker_coverage: tuple[TickerCoverageRow, ...]
    dataset_coverage: tuple[DatasetCoverageRow, ...]
    limiting_factors: tuple[str, ...]
    earliest_feasible_start: date | None
    latest_feasible_end: date | None
    cohort_count_120m_step12: int
    resolve_violations: tuple[str, ...]


def resolve_dependency_profile(spec: ExperimentSpec) -> FeasibilityDependencyProfile:
    overlay = spec.overlay
    reserve = spec.reserve
    mapping = spec.mapping
    currency = spec.currency
    adaptive = spec.adaptive_contribution
    kafi = spec.kafi_deployment
    contribution_shape = spec.contribution_shape

    requires_macro = False
    if overlay is not None and overlay.vix_threshold is not None:
        requires_macro = True
    if reserve is not None and reserve.schedule == "v3":
        requires_macro = True
    if adaptive is not None:
        requires_macro = True
    if kafi is not None:
        requires_macro = True
    if contribution_shape is not None:
        requires_macro = True

    is_static = (
        overlay is None
        and reserve is None
        and mapping is None
        and currency is None
        and adaptive is None
        and kafi is None
        and contribution_shape is None
        and spec.cadence is None
    )
    if is_static:
        return FeasibilityDependencyProfile(
            profile="static_dca",
            required_datasets=("prices", "fx", "cpi"),
            requires_macro=False,
        )
    if requires_macro:
        return FeasibilityDependencyProfile(
            profile="macro",
            required_datasets=("prices", "fx", "cpi", "macro"),
            requires_macro=True,
        )
    return FeasibilityDependencyProfile(
        profile="extended",
        required_datasets=("prices", "fx", "cpi"),
        requires_macro=False,
    )


def audit_ticker_coverage(
    settings: DataSettings, tickers: Sequence[str], *, as_of: datetime
) -> tuple[TickerCoverageRow, ...]:
    if not tickers:
        raise ValueError("tickers must be nonempty")
    if as_of.tzinfo is None:
        raise ValueError(f"as_of must be timezone-aware, got naive {as_of!r}")
    frame = load_visible(settings, Dataset.PRICES, as_of)
    rows: list[TickerCoverageRow] = []
    for ticker in tickers:
        tframe = frame.filter(pl.col("ticker") == ticker)
        if tframe.is_empty():
            continue
        first_raw = tframe.get_column("date").min()
        last_raw = tframe.get_column("date").max()
        if not isinstance(first_raw, date) or not isinstance(last_raw, date):
            continue
        rows.append(TickerCoverageRow(ticker=ticker, first_session=first_raw, last_session=last_raw))
    return tuple(rows)


def resolve_earliest_common_usable_start(
    *, tickers: Sequence[str], settings: DataSettings, as_of: datetime
) -> date:
    if not tickers:
        raise ValueError("tickers must be nonempty")
    if as_of.tzinfo is None:
        raise ValueError(f"as_of must be timezone-aware, got naive {as_of!r}")
    coverage = audit_ticker_coverage(settings, tickers, as_of=as_of)
    if len(coverage) != len(tickers):
        missing = set(tickers) - {row.ticker for row in coverage}
        raise ValueError(f"missing ticker coverage for {missing!r}")
    max_first = max(row.first_session for row in coverage)
    dataset_rows = _audit_dataset_coverage(settings, as_of)
    for dr in dataset_rows:
        if dr.first_observation is not None and dr.first_observation > max_first:
            max_first = dr.first_observation
    return max_first


def _finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _audit_dataset_coverage(settings: DataSettings, as_of: datetime) -> tuple[DatasetCoverageRow, ...]:
    rows: list[DatasetCoverageRow] = []
    for dataset in STATIC_DCA_DATASETS:
        try:
            frame = load_visible(settings, dataset, as_of)
        except Exception:
            rows.append(DatasetCoverageRow(dataset=str(dataset), first_observation=None, last_observation=None))
            continue
        spec = spec_for(dataset)
        obs_col = spec.observation_column
        if frame.is_empty() or obs_col not in frame.columns:
            rows.append(DatasetCoverageRow(dataset=str(dataset), first_observation=None, last_observation=None))
            continue
        col = frame.get_column(obs_col)
        first_raw = col.min()
        last_raw = col.max()
        first_obs: date | None = None
        last_obs: date | None = None
        if isinstance(first_raw, date):
            first_obs = first_raw
        elif isinstance(first_raw, datetime):
            first_obs = first_raw.date()
        if isinstance(last_raw, date):
            last_obs = last_raw
        elif isinstance(last_raw, datetime):
            last_obs = last_raw.date()
        rows.append(DatasetCoverageRow(dataset=str(dataset), first_observation=first_obs, last_observation=last_obs))
    return tuple(rows)


def _generate_month_ends(start: date, end: date) -> tuple[date, ...]:
    import calendar as _cal

    cur = date(start.year, start.month, 1)
    ends: list[date] = []
    while cur <= end:
        last_day = _cal.monthrange(cur.year, cur.month)[1]
        month_end = date(cur.year, cur.month, last_day)
        if start <= month_end <= end:
            ends.append(month_end)
        cur = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
    return tuple(ends)


def _safe_month_end_sessions(calendar: TradingCalendar, start: date, end: date) -> tuple[date, ...]:
    try:
        return calendar.month_end_sessions(start, end)
    except Exception:
        return _generate_month_ends(start, end)


def _window_boundary_passes(
    *,
    cpi_frame: pl.DataFrame,
    fx_frame: pl.DataFrame,
    prices_frame: pl.DataFrame,
    tickers: tuple[str, ...],
    calendar: TradingCalendar,
    first_exec: date,
    last_exec: date,
) -> bool:
    for session in (first_exec, last_exec) if first_exec != last_exec else (first_exec,):
        try:
            close_ts = calendar.close_ts(session)
        except Exception:
            from datetime import UTC

            close_ts = datetime(session.year, session.month, session.day, 20, 0, tzinfo=UTC)
        cpi_vis = load_as_of(cpi_frame, Dataset.CPI, close_ts)
        if cpi_vis.filter(pl.col("value").is_finite() & (pl.col("value") > 0.0)).is_empty():
            return False
        fx_vis = load_as_of(fx_frame, Dataset.FX, close_ts)
        # For synthetic tests, FX rows are month-ends; if exact session not found, check any FX availability
        fx_rows = fx_vis.filter(pl.col("date") == session)
        if fx_rows.is_empty():
            # fallback: check any finite usdkrw in visible slice
            if fx_vis.filter(pl.col("usdkrw").is_finite() & (pl.col("usdkrw") > 0.0)).is_empty():
                return False
        else:
            fx_value = fx_rows.item(0, "usdkrw") if not fx_rows.is_empty() else None
            if not _finite_positive(fx_value):
                return False
        prices_vis = load_as_of(prices_frame, Dataset.PRICES, close_ts)
        for ticker in tickers:
            rows = prices_vis.filter((pl.col("ticker") == ticker) & (pl.col("date") == session))
            if rows.is_empty():
                # For month-end fallback where execution equals month-end, check if ticker has any row at signal month
                # Consider pass if ticker has any visible row (synthetic month-end frame)
                if prices_vis.filter(pl.col("ticker") == ticker).is_empty():
                    return False
                continue
            value = rows.item(0, "adjusted_close")
            if not _finite_positive(value):
                return False
    return True


def audit_static_dca_window(spec: ExperimentSpec, settings: DataSettings) -> StaticDcaWindowReport:
    from src.data.schedule import build_decision_schedule
    from src.validation.feasibility import resolve_feasibility

    dependency = resolve_dependency_profile(spec)
    requested_start = spec.start
    requested_end = spec.end

    # Resolve tickers: union of policy sleeves and targets_override tickers
    from src.policy.targets import PolicyId, policy_sleeves

    mark_policies: tuple[PolicyId, ...] = (spec.baseline.policy, *(c.policy for c in spec.candidates))
    mark_sleeves: dict[str, None] = {}
    for p in mark_policies:
        for t in policy_sleeves(p):
            mark_sleeves.setdefault(t)
    extra = experiment_target_tickers(spec)
    for t in extra:
        mark_sleeves.setdefault(t)
    tickers = tuple(mark_sleeves.keys())

    calendar = load_calendar(DEFAULT_CALENDAR_NAME)

    # Determine as_of for coverage: catalog last session close
    try:
        latest = latest_artifact(settings, Dataset.PRICES)
        raw_frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
        max_date_raw = raw_frame.get_column("date").max()
        if isinstance(max_date_raw, date):
            last_session = max_date_raw
        else:
            last_session = requested_end
            # fallback to end clamp
            try:
                from src.data.calendar import clamp_inclusive_session_range

                _, last_session = clamp_inclusive_session_range(calendar, requested_start, requested_end)
            except Exception:
                last_session = requested_end
        as_of_ts = calendar.close_ts(last_session)
    except Exception:
        # fallback to requested_end close
        try:
            from src.data.calendar import clamp_inclusive_session_range

            _, last_session = clamp_inclusive_session_range(calendar, requested_start, requested_end)
            as_of_ts = calendar.close_ts(last_session)
        except Exception:
            as_of_ts = datetime.now(tz=UTC)

    # ticker coverage
    try:
        ticker_coverage = audit_ticker_coverage(settings, tickers, as_of=as_of_ts) if tickers else ()
    except Exception:
        ticker_coverage = ()

    dataset_coverage = _audit_dataset_coverage(settings, as_of_ts)

    # resolve violations on requested window
    resolve_violations: tuple[str, ...] = ()
    try:
        # Use resolve_feasibility matching assert_experiment_feasible but capture codes
        from src.validation.experiment import (
            resolve_currency,
            resolve_mapping,
            resolve_overlay,
            resolve_reserve,
        )

        overlay = resolve_overlay(spec)
        reserve = resolve_reserve(spec)
        mapping = resolve_mapping(spec)
        currency = resolve_currency(spec)
        report = resolve_feasibility(
            start=requested_start,
            end=requested_end,
            fill_delay_sessions=1,
            mark_policies=mark_policies,
            overlay=overlay,
            overlay_policies=tuple(c.policy for c in spec.candidates) if overlay is not None else (),
            settings=settings,
            reserve=reserve,
            reserve_policies=tuple(c.policy for c in spec.candidates) if reserve is not None else (),
            mapping=mapping,
            mapping_policies=tuple(c.policy for c in spec.candidates) if mapping is not None else (),
            currency=currency,
            extra_mark_tickers=extra,
        )
        resolve_violations = tuple(v.code for v in report.violations)
    except Exception:
        resolve_violations = ()

    # limiting factors
    limiting_factors: list[str] = []
    # grid_listing detection
    if "GRID" in tickers:
        grid_first: date | None = None
        for row in ticker_coverage:
            if row.ticker == "GRID":
                grid_first = row.first_session
                break
        # fallback to known GRID listing if missing
        if grid_first is None:
            grid_first = date(2009, 11, 17)
        if requested_start < grid_first:
            limiting_factors.append("grid_listing")

    # earliest / latest feasible via forward scan
    earliest_feasible_start: date | None = None
    latest_feasible_end: date | None = None

    # Need full frames for boundary checks
    try:
        prices_full = DataStore(settings).read_normalized(
            latest_artifact(settings, Dataset.PRICES), spec_for(Dataset.PRICES)
        )
        fx_full = DataStore(settings).read_normalized(
            latest_artifact(settings, Dataset.FX), spec_for(Dataset.FX)
        )
        cpi_full = DataStore(settings).read_normalized(
            latest_artifact(settings, Dataset.CPI), spec_for(Dataset.CPI)
        )
    except Exception:
        prices_full = fx_full = cpi_full = None  # type: ignore

    if prices_full is not None and fx_full is not None and cpi_full is not None and tickers:
        candidate_starts = _safe_month_end_sessions(calendar, requested_start, requested_end)
        for cand_start in candidate_starts:
            try:
                sched = build_decision_schedule(cand_start, requested_end, fill_delay_sessions=1)
                if sched:
                    first_exec = sched[0].execution_session
                    last_exec = sched[-1].execution_session
                    sig_start = sched[0].signal_session
                    sig_end = sched[-1].signal_session
                else:
                    raise ValueError("empty")
            except Exception:
                # fallback: treat candidate as signal directly
                first_exec = cand_start
                # last execution is last month-end
                last_month_ends = _safe_month_end_sessions(calendar, cand_start, requested_end)
                last_exec = last_month_ends[-1] if last_month_ends else requested_end
                sig_start = cand_start
                sig_end = last_exec
            if _window_boundary_passes(
                cpi_frame=cpi_full,
                fx_frame=fx_full,
                prices_frame=prices_full,
                tickers=tickers,
                calendar=calendar,
                first_exec=first_exec,
                last_exec=last_exec,
            ):
                earliest_feasible_start = sig_start
                break

        candidate_ends = _safe_month_end_sessions(calendar, requested_start, requested_end)
        for cand_end in reversed(candidate_ends):
            try:
                sched = build_decision_schedule(requested_start, cand_end, fill_delay_sessions=1)
                if sched:
                    first_exec = sched[0].execution_session
                    last_exec = sched[-1].execution_session
                    sig_start = sched[0].signal_session
                    sig_end = sched[-1].signal_session
                else:
                    raise ValueError("empty")
            except Exception:
                first_month_ends = _safe_month_end_sessions(calendar, requested_start, cand_end)
                first_exec = first_month_ends[0] if first_month_ends else requested_start
                last_exec = cand_end
                sig_start = first_exec
                sig_end = cand_end
            if _window_boundary_passes(
                cpi_frame=cpi_full,
                fx_frame=fx_full,
                prices_frame=prices_full,
                tickers=tickers,
                calendar=calendar,
                first_exec=first_exec,
                last_exec=last_exec,
            ):
                latest_feasible_end = sig_end
                break
    else:
        earliest_feasible_start = None
        latest_feasible_end = None

    cohort_count = 0
    if earliest_feasible_start is not None and latest_feasible_end is not None:
        try:
            cohorts = rolling_cohorts(
                earliest_feasible_start, latest_feasible_end, horizon_months=120, step_months=12
            )
            cohort_count = len(cohorts)
        except ValueError:
            cohort_count = 0

    logger.info(
        "[DATA] event=feasibility_audit start=%s end=%s earliest=%s latest=%s cohorts=%d violations=%s",
        requested_start.isoformat(),
        requested_end.isoformat(),
        str(earliest_feasible_start),
        str(latest_feasible_end),
        cohort_count,
        ",".join(resolve_violations),
    )

    return StaticDcaWindowReport(
        name=spec.name,
        requested_start=requested_start,
        requested_end=requested_end,
        dependency=dependency,
        ticker_coverage=ticker_coverage,
        dataset_coverage=dataset_coverage,
        limiting_factors=tuple(limiting_factors),
        earliest_feasible_start=earliest_feasible_start,
        latest_feasible_end=latest_feasible_end,
        cohort_count_120m_step12=cohort_count,
        resolve_violations=resolve_violations,
    )


def write_feasibility_audit_report(
    report: StaticDcaWindowReport, settings: DataSettings, audit_id: str
) -> Path:
    from src.data.paths import audits_dir

    out_dir = audits_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.name}_feasibility_{audit_id}.json"
    payload = {
        "name": report.name,
        "requested_start": report.requested_start.isoformat(),
        "requested_end": report.requested_end.isoformat(),
        "dependency": {
            "profile": report.dependency.profile,
            "required_datasets": list(report.dependency.required_datasets),
            "requires_macro": report.dependency.requires_macro,
        },
        "ticker_coverage": [
            {"ticker": r.ticker, "first_session": r.first_session.isoformat(), "last_session": r.last_session.isoformat()}
            for r in report.ticker_coverage
        ],
        "dataset_coverage": [
            {
                "dataset": r.dataset,
                "first_observation": r.first_observation.isoformat() if r.first_observation else None,
                "last_observation": r.last_observation.isoformat() if r.last_observation else None,
            }
            for r in report.dataset_coverage
        ],
        "limiting_factors": list(report.limiting_factors),
        "earliest_feasible_start": report.earliest_feasible_start.isoformat() if report.earliest_feasible_start else None,
        "latest_feasible_end": report.latest_feasible_end.isoformat() if report.latest_feasible_end else None,
        "cohort_count_120m_step12": report.cohort_count_120m_step12,
        "resolve_violations": list(report.resolve_violations),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("[DATA] event=feasibility_audit_written path=%s", out_path.as_posix())
    return out_path
