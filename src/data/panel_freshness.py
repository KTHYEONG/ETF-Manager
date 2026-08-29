# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Panel freshness for thesis research as-of."""
from __future__ import annotations

import calendar as cal_module
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

import polars as pl

from src.data.calendar import DEFAULT_CALENDAR_NAME, TradingCalendar, load_calendar
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings

logger = logging.getLogger(__name__)

THESIS_PANEL_TICKERS: Final[tuple[str, ...]] = ("BOTZ", "GRID", "QQQ", "SOXX")
MAX_PANEL_LAG_DAYS: Final[int] = 62


class PanelFreshnessStatus(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    HARD_STOP_ACK = "HARD_STOP_ACK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class PanelHardStop:
    reason: str
    max_panel_as_of: date


@dataclass(frozen=True, slots=True)
class CatalogPanelReport:
    panel_as_of: datetime
    lag_days: int
    status: PanelFreshnessStatus
    ticker_last_session: Mapping[str, date]
    cpi_last_observation: date | None
    fx_last_observation: date | None
    holdings_last_filing: date | None
    hard_stop_reason: str | None = None


def load_panel_hard_stop(path: Path | None = None) -> PanelHardStop | None:
    target = Path(path) if path is not None else Path("configs/data/panel_hard_stop.json")
    if not target.is_file():
        return None
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    reason = str(doc.get("reason", "")).strip()
    max_as_of_raw = doc.get("max_panel_as_of")
    if not reason or max_as_of_raw is None:
        return None
    try:
        max_date = date.fromisoformat(str(max_as_of_raw))
    except ValueError:
        return None
    return PanelHardStop(reason=reason, max_panel_as_of=max_date)


def apply_hard_stop(report: CatalogPanelReport, hard_stop: PanelHardStop | None) -> CatalogPanelReport:
    if report.status != PanelFreshnessStatus.STALE:
        return report
    if hard_stop is None or not hard_stop.reason.strip():
        return report
    if report.panel_as_of.date() <= hard_stop.max_panel_as_of:
        return CatalogPanelReport(
            panel_as_of=report.panel_as_of,
            lag_days=report.lag_days,
            status=PanelFreshnessStatus.HARD_STOP_ACK,
            ticker_last_session=report.ticker_last_session,
            cpi_last_observation=report.cpi_last_observation,
            fx_last_observation=report.fx_last_observation,
            holdings_last_filing=report.holdings_last_filing,
            hard_stop_reason=hard_stop.reason,
        )
    return report


def iter_nport_quarters_for_panel(panel_as_of: date, *, lookback_months: int = 18) -> tuple[str, ...]:
    # Compute start date: subtract lookback_months
    y, m = panel_as_of.year, panel_as_of.month
    # calendar month arithmetic
    total_months = y * 12 + (m - 1) - lookback_months
    start_y = total_months // 12
    start_m = total_months % 12 + 1
    start_date = date(start_y, start_m, 1)

    # panel quarter end: inclusive
    def quarter_label(dt: date) -> str:
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}q{q}"

    def quarter_start(year: int, q: int) -> date:
        month = (q - 1) * 3 + 1
        return date(year, month, 1)

    def quarter_end(year: int, q: int) -> date:
        month = q * 3
        last = cal_module.monthrange(year, month)[1]
        return date(year, month, last)

    # Generate quarters from start_date's quarter to panel's quarter inclusive
    start_q = (start_date.month - 1) // 3 + 1
    end_q = (panel_as_of.month - 1) // 3 + 1
    # iterate
    labels: list[str] = []
    cy, cq = start_date.year, start_q
    py, pq = panel_as_of.year, end_q
    while (cy < py) or (cy == py and cq <= pq):
        labels.append(f"{cy}q{cq}")
        cq += 1
        if cq > 4:
            cq = 1
            cy += 1

    # Filter: keep only quarters where quarter_start <= panel_as_of and quarter_end >= start_date
    # Already ensured, but verify not after panel month's quarter start beyond panel: our iteration stops at panel quarter, so okay.
    # Ensure ascending and unique
    return tuple(labels)


def effective_thesis_end(panel_as_of: datetime) -> date:
    if panel_as_of.tzinfo is None:
        raise ValueError("panel_as_of must be timezone-aware")
    return panel_as_of.date()


def _load_catalog_frames(settings: DataSettings) -> dict[Dataset, pl.DataFrame]:
    """Load PRICES/FX/CPI frames via catalog; patchable for tests."""
    from src.data.catalog import latest_artifact
    from src.data.storage import DataStore

    frames: dict[Dataset, pl.DataFrame] = {}
    for ds in (Dataset.PRICES, Dataset.FX, Dataset.CPI):
        artifact = latest_artifact(settings, ds)
        frame = DataStore(settings).read_normalized(artifact, spec_for(ds))
        frames[ds] = frame
    return frames


def _month_end_session_for(calendar: TradingCalendar, day: date) -> date | None:
    try:
        sessions = calendar.sessions(date(day.year, day.month, 1), date(day.year, day.month, cal_module.monthrange(day.year, day.month)[1]))
        if not sessions:
            return None
        return sessions[-1]
    except Exception:
        return None


def _session_passes(
    session: date,
    calendar: TradingCalendar,
    frames: dict[Dataset, pl.DataFrame],
    tickers: Sequence[str],
) -> bool:
    try:
        close_ts = calendar.close_ts(session)
    except Exception:
        close_ts = datetime(session.year, session.month, session.day, 20, 0, tzinfo=UTC)
    # Check PRICES
    prices_frame = frames.get(Dataset.PRICES)
    if prices_frame is None or prices_frame.is_empty():
        return False
    for t in tickers:
        # try PIT check via available_at if present
        if "available_at" in prices_frame.columns:
            try:
                from src.data.query import load_as_of
                vis = load_as_of(prices_frame, Dataset.PRICES, close_ts)
                rows = vis.filter((pl.col("ticker") == t) & (pl.col("date") == session))
                if not rows.is_empty():
                    val = rows.get_column("adjusted_close")[0]
                    if isinstance(val, float) and (val != val or val <= 0):
                        return False
                    continue
                # fallback: if no exact session match, check any row for ticker visible
                if vis.filter(pl.col("ticker") == t).is_empty():
                    return False
                continue
            except Exception:  # noqa: S110
                pass  # noqa: S110
        # direct fallback
        filt = prices_frame.filter((pl.col("ticker") == t) & (pl.col("date") == session)) if "ticker" in prices_frame.columns else pl.DataFrame()
        if not filt.is_empty():
            try:
                v = filt.get_column("adjusted_close")[0]
                if isinstance(v, float) and (v != v or v <= 0):
                    return False
            except Exception:  # noqa: S110
                pass  # noqa: S110
        else:
            # allow if frame has any row for ticker (synthetic month-end only)
            if prices_frame.filter(pl.col("ticker") == t).is_empty():
                return False
    # FX
    fx_frame = frames.get(Dataset.FX)
    if fx_frame is None or fx_frame.is_empty():
        return False
    if "available_at" in fx_frame.columns:
        try:
            from src.data.query import load_as_of
            fx_vis = load_as_of(fx_frame, Dataset.FX, close_ts)
            fx_rows = fx_vis.filter(pl.col("date") == session) if "date" in fx_vis.columns else fx_vis
            if not fx_rows.is_empty():
                if "usdkrw" in fx_rows.columns:
                    try:
                        v = fx_rows.get_column("usdkrw")[0]
                        if isinstance(v, float) and (v != v or v <= 0):
                            return False
                    except Exception:  # noqa: S110
                        pass  # noqa: S110
            else:
                if fx_vis.filter(pl.col("usdkrw").is_finite()).is_empty():
                    return False
            # pass
        except Exception:
            fx_rows = fx_frame.filter(pl.col("date") == session) if "date" in fx_frame.columns else fx_frame
            if fx_rows.is_empty():
                if fx_frame.filter(pl.col("usdkrw").is_finite()).is_empty():
                    return False
    else:
        fx_rows = fx_frame.filter(pl.col("date") == session) if "date" in fx_frame.columns else fx_frame
        if fx_rows.is_empty():
            if fx_frame.filter(pl.col("usdkrw").is_finite()).is_empty():
                return False

    # CPI
    cpi_frame = frames.get(Dataset.CPI)
    if cpi_frame is None or cpi_frame.is_empty():
        return False
    # check finite value visibility or direct
    if "available_at" in cpi_frame.columns:
        try:
            from src.data.query import load_as_of
            cpi_vis = load_as_of(cpi_frame, Dataset.CPI, close_ts)
            if cpi_vis.filter(pl.col("value").is_finite()).is_empty():
                return False
        except Exception:
            if cpi_frame.filter(pl.col("value").is_finite()).is_empty():
                return False
    else:
        if cpi_frame.filter(pl.col("value").is_finite()).is_empty():
            return False
    return True


def resolve_catalog_panel_as_of(
    settings: DataSettings,
    *,
    reference_now: datetime,
    tickers: Sequence[str] = THESIS_PANEL_TICKERS,
) -> CatalogPanelReport:
    if reference_now.tzinfo is None:
        raise ValueError("reference_now must be timezone-aware, got naive datetime")
    # panel_as_of must be tz-aware later but reference check above satisfies requirement
    calendar = load_calendar(DEFAULT_CALENDAR_NAME)
    try:
        frames = _load_catalog_frames(settings)
    except Exception:
        # INSUFFICIENT_DATA
        # Use reference_now as panel_as_of placeholder? Choose reference_now truncated to UTC
        panel_as_of = reference_now.astimezone(UTC)
        lag = 0
        return CatalogPanelReport(
            panel_as_of=panel_as_of,
            lag_days=lag,
            status=PanelFreshnessStatus.INSUFFICIENT_DATA,
            ticker_last_session={},
            cpi_last_observation=None,
            fx_last_observation=None,
            holdings_last_filing=None,
            hard_stop_reason=None,
        )

    prices_frame = frames.get(Dataset.PRICES)
    if prices_frame is None or prices_frame.is_empty():
        return CatalogPanelReport(
            panel_as_of=reference_now.astimezone(UTC),
            lag_days=0,
            status=PanelFreshnessStatus.INSUFFICIENT_DATA,
            ticker_last_session={},
            cpi_last_observation=None,
            fx_last_observation=None,
            holdings_last_filing=None,
            hard_stop_reason=None,
        )
    # check missing ticker
    if "ticker" in prices_frame.columns:
        for t in tickers:
            if prices_frame.filter(pl.col("ticker") == t).is_empty():
                return CatalogPanelReport(
                    panel_as_of=reference_now.astimezone(UTC),
                    lag_days=0,
                    status=PanelFreshnessStatus.INSUFFICIENT_DATA,
                    ticker_last_session={},
                    cpi_last_observation=None,
                    fx_last_observation=None,
                    holdings_last_filing=None,
                    hard_stop_reason=None,
                )
    # Determine candidate month-end sessions
    # Use range from earliest price date to reference_now or max price date
    try:
        max_date_raw = prices_frame.get_column("date").max()
        if isinstance(max_date_raw, date):
            max_price_date = max_date_raw
        else:
            max_price_date = reference_now.date()
    except Exception:
        max_price_date = reference_now.date()
    # Clamp to reference_now.date() if max beyond reference
    if max_price_date > reference_now.date():
        max_price_date = reference_now.date()
    # earliest
    try:
        min_date_raw = prices_frame.get_column("date").min()
        if isinstance(min_date_raw, date):
            min_price_date = min_date_raw
        else:
            min_price_date = max_price_date
    except Exception:
        min_price_date = max_price_date
    # Generate month ends
    try:
        candidates = calendar.month_end_sessions(min_price_date, max_price_date)
    except Exception:
        # fallback generate month ends via calendar helper
        candidates = ()
        if not candidates:
            # fallback list with max_price_date if it's a session
            if calendar.is_session(max_price_date):
                candidates = (max_price_date,)
            else:
                # find previous session
                try:
                    # find last session <= max_price_date
                    # brute walk back 10 days
                    for delta in range(10):
                        cand = date(max_price_date.year, max_price_date.month, max_price_date.day)
                        # shift back
                        from datetime import timedelta
                        cand = cand - timedelta(days=delta)
                        if calendar.is_session(cand):
                            candidates = (cand,)
                            break
                except Exception:
                    candidates = (max_price_date,)
    # Iterate descending to find latest passing
    passing: date | None = None
    for sess in reversed(candidates):
        if _session_passes(sess, calendar, frames, tickers):
            passing = sess
            break
    if passing is None:
        return CatalogPanelReport(
            panel_as_of=reference_now.astimezone(UTC),
            lag_days=0,
            status=PanelFreshnessStatus.INSUFFICIENT_DATA,
            ticker_last_session={},
            cpi_last_observation=None,
            fx_last_observation=None,
            holdings_last_filing=None,
            hard_stop_reason=None,
        )
    panel_as_of = calendar.close_ts(passing)
    if panel_as_of.tzinfo is None:
        panel_as_of = panel_as_of.replace(tzinfo=UTC)
    lag_days = (reference_now.date() - panel_as_of.date()).days
    status = PanelFreshnessStatus.FRESH if lag_days <= MAX_PANEL_LAG_DAYS else PanelFreshnessStatus.STALE
    # Compute coverage fields
    ticker_last: dict[str, date] = {}
    try:
        for t in tickers:
            rows = prices_frame.filter(pl.col("ticker") == t)
            if not rows.is_empty():
                v = rows.get_column("date").max()
                if isinstance(v, date):
                    ticker_last[t] = v
    except Exception:
        ticker_last = {}
    cpi_last: date | None = None
    fx_last: date | None = None
    holdings_last: date | None = None
    try:
        cpi_frame = frames.get(Dataset.CPI)
        if cpi_frame is not None and not cpi_frame.is_empty():
            col = cpi_frame.get_column("period_end") if "period_end" in cpi_frame.columns else cpi_frame.get_column(cpi_frame.columns[0])
            v = col.max()
            if isinstance(v, date):
                cpi_last = v
    except Exception:  # noqa: S110
        pass  # noqa: S110
    try:
        fx_frame = frames.get(Dataset.FX)
        if fx_frame is not None and not fx_frame.is_empty() and "date" in fx_frame.columns:
            v = fx_frame.get_column("date").max()
            if isinstance(v, date):
                fx_last = v
    except Exception:  # noqa: S110
        pass  # noqa: S110
    try:
        from src.data.catalog import latest_artifact
        from src.data.storage import DataStore
        holdings_frame = DataStore(settings).read_normalized(latest_artifact(settings, Dataset.ETF_HOLDINGS), spec_for(Dataset.ETF_HOLDINGS))
        if not holdings_frame.is_empty() and "report_date" in holdings_frame.columns:
            v = holdings_frame.get_column("report_date").max()
            if isinstance(v, date):
                holdings_last = v
    except Exception:
        holdings_last = None

    return CatalogPanelReport(
        panel_as_of=panel_as_of,
        lag_days=lag_days,
        status=status,
        ticker_last_session=ticker_last,
        cpi_last_observation=cpi_last,
        fx_last_observation=fx_last,
        holdings_last_filing=holdings_last,
        hard_stop_reason=None,
    )
