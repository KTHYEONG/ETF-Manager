"""Regime proxy slot (Wave 7)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime

from src.analytics.regimes import QQQ_REGIME_WINDOWS
from src.analytics.thesis_evidence import EvidenceSlot
from src.data.settings import DataSettings
from src.policy.targets import PolicyId
from src.policy.thesis import ThesisSpec
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.prospective import resolve_proxy_history_span

__all__ = ["compute_regime_proxy_slot"]

_SELECTED_WINDOWS = ("bear_2022", "pre_ai", "recent_2023_2026")


def _catalog_last_session(settings: DataSettings, as_of: datetime) -> date:
    from src.data.catalog import load_visible
    from src.data.schema import Dataset

    prices = load_visible(settings, Dataset.PRICES, as_of)
    if prices.is_empty():
        return as_of.date()
    end_raw = prices.get_column("date").max()
    return end_raw if isinstance(end_raw, date) else as_of.date()


def _clip_regime_window(
    start: date,
    end: date,
    *,
    proxy_first: date,
    proxy_last: date,
    catalog_last: date,
) -> tuple[date, date] | None:
    effective_start = max(start, proxy_first)
    effective_end = min(end, proxy_last, catalog_last)
    if effective_start > effective_end:
        return None
    return effective_start, effective_end


def compute_regime_proxy_slot(
    *,
    thesis: ThesisSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float = 1_000_000,
    settings: DataSettings | None = None,
    as_of: datetime | None = None,
) -> EvidenceSlot:
    """Compute regime proxy vs QQQ over selected windows; skip infeasible windows."""
    proxy_ticker = thesis.historical_proxies[0].value if thesis.historical_proxies else "QQQ"
    selected = tuple(w for w in QQQ_REGIME_WINDOWS if w[0] in _SELECTED_WINDOWS)

    proxy_first: date | None = None
    proxy_last: date | None = None
    catalog_last: date | None = None
    if settings is not None and as_of is not None:
        try:
            proxy_first, proxy_last = resolve_proxy_history_span(settings=settings, thesis=thesis, as_of=as_of)
            catalog_last = _catalog_last_session(settings, as_of)
        except Exception:
            proxy_first = proxy_last = catalog_last = None

    windows_beat = 0
    windows_tested = 0
    windows_skipped = 0
    ratios: list[float] = []
    details: list[str] = []

    for name, start, end in selected:
        window_start, window_end = start, end
        if proxy_first is not None and proxy_last is not None and catalog_last is not None:
            clipped = _clip_regime_window(
                start, end, proxy_first=proxy_first, proxy_last=proxy_last, catalog_last=catalog_last
            )
            if clipped is None:
                windows_skipped += 1
                details.append(f"{name}: skipped (infeasible window)")
                continue
            window_start, window_end = clipped

        baseline_config = AllocationConfig(
            policy=PolicyId.QQQ,
            start=window_start,
            end=window_end,
            monthly_contribution_krw=float(contribution_krw),
            targets_override={"QQQ": 1.0},
        )
        candidate_config = AllocationConfig(
            policy=PolicyId.QQQ,
            start=window_start,
            end=window_end,
            monthly_contribution_krw=float(contribution_krw),
            targets_override={proxy_ticker: 1.0},
        )
        try:
            baseline_result = runner(baseline_config)
            candidate_result = runner(candidate_config)
        except Exception as exc:  # noqa: BLE001
            windows_skipped += 1
            details.append(f"{name}: skipped ({str(exc)[:80]})")
            continue

        windows_tested += 1
        baseline_tw = float(baseline_result.terminal_wealth_real_krw)
        candidate_tw = float(candidate_result.terminal_wealth_real_krw)
        ratio = candidate_tw / baseline_tw if baseline_tw > 0 else 0.0
        ratios.append(ratio)
        if candidate_tw > baseline_tw:
            windows_beat += 1
            details.append(f"{name}: {proxy_ticker} beat QQQ")
        else:
            details.append(f"{name}: QQQ beat {proxy_ticker}")

    if windows_tested == 0:
        return EvidenceSlot(
            status="insufficient_data",
            summary="no feasible regime windows",
            metrics={"windows_skipped": windows_skipped, "proxy": proxy_ticker},
        )

    median_ratio = float(sorted(ratios)[len(ratios) // 2])
    worst_ratio = float(min(ratios))
    summary = f"regime proxy {proxy_ticker} vs QQQ {windows_beat}/{windows_tested} windows beat"
    metrics: dict[str, float | int | str] = {
        "windows_beat_qqq": int(windows_beat),
        "windows_tested": int(windows_tested),
        "windows_skipped": int(windows_skipped),
        "median_regime_ratio": median_ratio,
        "worst_regime_ratio": worst_ratio,
        "proxy": proxy_ticker,
        "contribution_krw": float(contribution_krw),
    }
    metrics["bear_2022"] = next((d for d in details if "bear_2022" in d), "")
    return EvidenceSlot(status="computed", summary=summary, metrics=metrics)
