# ruff: noqa: B905
"""Track H incremental portfolio: 5/10/15 SOXX vs QQQ100."""
from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from src.analytics.thesis_meaning import PortfolioEvidenceStatus, ThesisMeaningSnapshot
from src.data.panel_freshness import CatalogPanelReport, effective_thesis_end, resolve_catalog_panel_as_of
from src.data.settings import DataSettings
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.gate import certainty_equivalent, cohort_win_rate, wealth_quantile
from src.validation.windows import add_calendar_months, rolling_cohorts

__all__ = [
    "INCREMENTAL_SOXX_WEIGHTS",
    "PATH_BOOTSTRAP_WIN_FLOOR",
    "BuyOnlyAttribution",
    "IncrementalArmId",
    "IncrementalArmReport",
    "IncrementalPortfolioReport",
    "PathBootstrapVerdict",
    "apply_incremental_portfolio_status",
    "arm_targets",
    "attribute_buy_only_soxx",
    "classify_portfolio_status",
    "monthly_simple_returns",
    "paired_path_block_bootstrap",
    "run_incremental_portfolio",
    "write_incremental_portfolio_report",
]

INCREMENTAL_SOXX_WEIGHTS: Final[tuple[float, ...]] = (0.05, 0.10, 0.15)
PATH_BOOTSTRAP_WIN_FLOOR: Final[float] = 0.55
WEIGHT_UPPER_EPS: Final[float] = 1e-3
MIN_REALIZED_FRAC_OF_TARGET: Final[float] = 0.05


class IncrementalArmId(StrEnum):
    QQQ95_SOXX5 = "qqq95_soxx5"
    QQQ90_SOXX10 = "qqq90_soxx10"
    QQQ85_SOXX15 = "qqq85_soxx15"


@dataclass(frozen=True, slots=True)
class BuyOnlyAttribution:
    target_soxx_weight: float
    mean_realized_soxx_weight: float
    terminal_realized_soxx_weight: float
    mean_abs_weight_drift: float
    terminal_weight_drift: float
    incremental_wealth_ratio: float


@dataclass(frozen=True, slots=True)
class PathBootstrapVerdict:
    n_paths: int
    win_rate: float
    p05_terminal_ratio: float
    ok: bool


@dataclass(frozen=True, slots=True)
class IncrementalArmReport:
    arm_id: IncrementalArmId
    soxx_weight: float
    median_ratio: float
    p10_ratio: float
    worst_ratio: float
    win_rate: float
    cohort_count: int
    ce_gamma_2: float
    ce_gamma_5: float
    ce_gamma_10: float
    attribution: BuyOnlyAttribution
    path_bootstrap: PathBootstrapVerdict


@dataclass(frozen=True, slots=True)
class IncrementalPortfolioReport:
    thesis_id: str
    as_of: datetime
    panel_as_of: datetime
    lag_days: int
    freshness_status: str
    arms: tuple[IncrementalArmReport, ...]
    portfolio_status: PortfolioEvidenceStatus


def arm_targets(soxx_weight: float) -> dict[str, float]:
    w = float(soxx_weight)
    if w not in INCREMENTAL_SOXX_WEIGHTS:
        raise ValueError(f"soxx_weight {soxx_weight!r} not in {INCREMENTAL_SOXX_WEIGHTS}")
    return {"QQQ": 1.0 - w, "SOXX": w}


def monthly_simple_returns(result: AllocationResult) -> tuple[float, ...]:
    snaps = result.snapshots
    if len(snaps) < 1:
        return ()
    for s in snaps:
        if float(s.mark_krw) <= 0.0 or not math.isfinite(float(s.mark_krw)):
            raise ValueError(f"mark_krw must be positive, got {s.mark_krw!r} on {s.session}")
    out: list[float] = []
    for prev, cur in zip(snaps, snaps[1:], strict=False):  # noqa: RUF007
        pm = float(prev.mark_krw)
        cm = float(cur.mark_krw)
        if pm <= 0.0 or cm <= 0.0:
            raise ValueError("mark_krw must be positive")
        out.append(cm / pm - 1.0)
    return tuple(out)


def paired_path_block_bootstrap(
    candidate_returns: Sequence[float],
    baseline_returns: Sequence[float],
    *,
    block_size: int,
    n_paths: int,
    seed: int,
) -> PathBootstrapVerdict:
    cand = tuple(float(v) for v in candidate_returns)
    base = tuple(float(v) for v in baseline_returns)
    if len(cand) != len(base):
        raise ValueError("candidate and baseline returns must have equal length")
    n = len(cand)
    if n < 1:
        raise ValueError("returns must contain at least one observation")
    if not 1 <= block_size <= n:
        raise ValueError(f"block_size must lie in [1, {n}], got {block_size}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >=1, got {n_paths}")
    for v in (*cand, *base):
        if not math.isfinite(v):
            raise ValueError("returns must be finite")
    rng = random.Random(seed)  # noqa: S311
    ratios: list[float] = []
    for _ in range(n_paths):
        sampled_c: list[float] = []
        sampled_b: list[float] = []
        while len(sampled_c) < n:
            start = rng.randrange(n)
            for offset in range(block_size):
                idx = (start + offset) % n
                sampled_c.append(cand[idx])
                sampled_b.append(base[idx])
                if len(sampled_c) >= n:
                    break
        sampled_c = sampled_c[:n]
        sampled_b = sampled_b[:n]
        wealth_c = 1.0
        wealth_b = 1.0
        for rc, rb in zip(sampled_c, sampled_b, strict=False):
            wealth_c *= 1.0 + rc
            wealth_b *= 1.0 + rb
        if wealth_b == 0.0:
            raise ValueError("baseline wealth path zero")
        ratios.append(wealth_c / wealth_b)
    win_rate = sum(1 for r in ratios if r >= 1.0) / len(ratios)
    p05 = wealth_quantile(ratios, 0.05) if len(ratios) >= 1 else 0.0
    ok = win_rate >= PATH_BOOTSTRAP_WIN_FLOOR
    return PathBootstrapVerdict(n_paths=int(n_paths), win_rate=float(win_rate), p05_terminal_ratio=float(p05), ok=bool(ok))


def attribute_buy_only_soxx(
    *,
    candidate: AllocationResult,
    baseline: AllocationResult,
    soxx_weight: float,
    price_at: Callable[[date, str], float],
    fx_at: Callable[[date], float],
) -> BuyOnlyAttribution:
    if not candidate.snapshots:
        raise ValueError("candidate snapshots empty")
    if not baseline.snapshots:
        raise ValueError("baseline snapshots empty")
    target = float(soxx_weight)
    realized: list[float] = []
    for snap in candidate.snapshots:
        mk = float(snap.mark_krw)
        if not math.isfinite(mk) or mk <= 0.0:
            raise ValueError(f"mark_krw must be positive, got {mk!r}")
        try:
            px = float(price_at(snap.session, "SOXX"))
        except Exception as exc:
            raise ValueError(f"price_at failed for {snap.session}: {exc}") from exc
        if not math.isfinite(px) or px <= 0.0:
            raise ValueError(f"price_at returned non-positive {px!r} for {snap.session}")
        try:
            usdkrw = float(fx_at(snap.session))
        except Exception as exc:
            raise ValueError(f"fx_at failed for {snap.session}: {exc}") from exc
        if not math.isfinite(usdkrw) or usdkrw <= 0.0:
            raise ValueError(f"fx_at returned non-positive {usdkrw!r} for {snap.session}")
        shares = float(snap.shares.get("SOXX", 0.0))
        w = shares * px * usdkrw / mk
        if not math.isfinite(w):
            raise ValueError("realized weight non-finite")
        if w < 0.0 or w > 1.0 + WEIGHT_UPPER_EPS:
            raise ValueError(f"realized weight {w!r} outside [0, 1+WEIGHT_UPPER_EPS] on {snap.session}")
        realized.append(w)
    mean_realized = sum(realized) / len(realized)
    terminal_realized = realized[-1]
    mean_abs = sum(abs(w - target) for w in realized) / len(realized)
    terminal_drift = terminal_realized - target
    base_tw = float(baseline.terminal_wealth_real_krw)
    cand_tw = float(candidate.terminal_wealth_real_krw)
    if not math.isfinite(base_tw) or base_tw <= 0.0 or not math.isfinite(cand_tw) or cand_tw <= 0.0:
        raise ValueError("terminal wealths must be positive")
    ratio = cand_tw / base_tw
    if target > 0.0:
        has_position = any(float(s.shares.get("SOXX", 0.0)) > 0.0 for s in candidate.snapshots)
        if has_position and mean_realized < target * MIN_REALIZED_FRAC_OF_TARGET:
            raise ValueError(f"mean realized weight {mean_realized!r} below coherence floor {target * MIN_REALIZED_FRAC_OF_TARGET!r}")
    return BuyOnlyAttribution(
        target_soxx_weight=float(target),
        mean_realized_soxx_weight=float(mean_realized),
        terminal_realized_soxx_weight=float(terminal_realized),
        mean_abs_weight_drift=float(mean_abs),
        terminal_weight_drift=float(terminal_drift),
        incremental_wealth_ratio=float(ratio),
    )


def classify_portfolio_status(arms: Sequence[IncrementalArmReport]) -> PortfolioEvidenceStatus:
    if not arms:
        raise ValueError("arms must be non-empty")
    for arm in arms:
        if float(arm.median_ratio) >= 1.0 and bool(arm.path_bootstrap.ok):
            return PortfolioEvidenceStatus.HISTORICALLY_PROMISING
    return PortfolioEvidenceStatus.HISTORICALLY_WEAK


def apply_incremental_portfolio_status(
    meaning: ThesisMeaningSnapshot, portfolio_status: PortfolioEvidenceStatus
) -> ThesisMeaningSnapshot:
    return replace(meaning, portfolio_status=portfolio_status)


def _resolve_panel(settings: DataSettings, as_of: datetime, panel_report: CatalogPanelReport | None) -> CatalogPanelReport:
    if panel_report is not None:
        return panel_report
    return resolve_catalog_panel_as_of(settings, reference_now=as_of)


def _make_price_at(settings: DataSettings) -> Callable[[date, str], float]:
    """PIT adjusted close visible at session close; fail-closed on missing/non-positive."""
    import polars as pl

    from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
    from src.data.catalog import latest_artifact
    from src.data.query import load_as_of
    from src.data.schema import Dataset, spec_for
    from src.data.storage import DataStore

    latest = latest_artifact(settings, Dataset.PRICES)
    prices = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
    cal = load_calendar(DEFAULT_CALENDAR_NAME)

    def _price_at(d: date, ticker: str) -> float:
        close_ts = cal.close_ts(d)
        visible = load_as_of(prices, Dataset.PRICES, close_ts)
        rows = visible.filter((pl.col("ticker") == ticker) & (pl.col("date") == d))
        if rows.is_empty():
            raise ValueError(f"price missing for {ticker!r} on {d.isoformat()}")
        value = rows.item(0, "adjusted_close")
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"non-positive adjusted_close for {ticker!r} on {d.isoformat()}")
        return float(value)

    return _price_at


def _make_fx_at(settings: DataSettings) -> Callable[[date], float]:
    """PIT usdkrw at session close via Dataset.FX; ValueError on missing/non-positive."""
    import polars as pl

    from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
    from src.data.catalog import latest_artifact
    from src.data.query import load_as_of
    from src.data.schema import Dataset, spec_for
    from src.data.storage import DataStore

    latest = latest_artifact(settings, Dataset.FX)
    fx = DataStore(settings).read_normalized(latest, spec_for(Dataset.FX))
    cal = load_calendar(DEFAULT_CALENDAR_NAME)

    def _fx_at(d: date) -> float:
        if not isinstance(d, date):
            raise ValueError(f"fx_at requires date, got {d!r}")
        close_ts = cal.close_ts(d)
        if close_ts.tzinfo is None:
            raise ValueError(f"close_ts naive for {d.isoformat()}")
        visible = load_as_of(fx, Dataset.FX, close_ts)
        rows = visible.filter(pl.col("date") == d)
        if rows.is_empty():
            raise ValueError(f"fx missing for {d.isoformat()}")
        value = rows.item(0, "usdkrw")
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"non-positive usdkrw for {d.isoformat()}")
        return float(value)

    return _fx_at


def _cohort_start_fallback(end: date) -> date:
    # fallback start ensures at least one 120M cohort ending at end
    try:
        s = add_calendar_months(end, -120)
        # add one day to make inclusive horizon fit exactly; rolling_cohorts handles inclusive logic
        return s
    except Exception:
        return date(2007, 8, 31)


def run_incremental_portfolio(
    *,
    settings: DataSettings,
    as_of: datetime,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    bootstrap_paths: int,
    seed: int,
    panel_report: CatalogPanelReport | None = None,
) -> IncrementalPortfolioReport:
    if contribution_krw <= 0 or not math.isfinite(contribution_krw):
        raise ValueError("contribution_krw must be positive")
    if bootstrap_paths < 1:
        raise ValueError("bootstrap_paths must be >=1")
    panel = _resolve_panel(settings, as_of, panel_report)
    end = effective_thesis_end(panel.panel_as_of)
    from src.data.calendar import DEFAULT_CALENDAR_NAME, clamp_inclusive_session_range, load_calendar

    cal = load_calendar(DEFAULT_CALENDAR_NAME)
    # Determine start for cohorts: try catalog min, else fallback
    start: date
    try:
        from src.data.catalog import latest_artifact
        from src.data.schema import Dataset, spec_for
        from src.data.storage import DataStore

        latest = latest_artifact(settings, Dataset.PRICES)
        frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
        min_raw = frame.get_column("date").min()
        start = min_raw if isinstance(min_raw, date) else date(2007, 8, 31)
    except Exception:
        start = date(2007, 8, 31)
    start, end = clamp_inclusive_session_range(cal, start, end)
    cohorts = rolling_cohorts(start, end, horizon_months=120, step_months=12)
    if not cohorts:
        fb_start = _cohort_start_fallback(end)
        fb_start, end = clamp_inclusive_session_range(cal, fb_start, end)
        cohorts = rolling_cohorts(fb_start, end, horizon_months=120, step_months=12)
        if not cohorts:
            raise ValueError("span too short for 120M cohort")
        start = fb_start

    price_at = _make_price_at(settings)
    fx_at = _make_fx_at(settings)

    arms: list[IncrementalArmReport] = []
    for idx, w in enumerate(INCREMENTAL_SOXX_WEIGHTS):
        arm_id = [IncrementalArmId.QQQ95_SOXX5, IncrementalArmId.QQQ90_SOXX10, IncrementalArmId.QQQ85_SOXX15][idx]
        baseline_template = AllocationConfig(
            policy=PolicyId.QQQ,
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
        )
        candidate_template = AllocationConfig(
            policy=PolicyId.QQQ,
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
            targets_override=arm_targets(float(w)),
        )
        c_wealths: list[float] = []
        b_wealths: list[float] = []
        for c_start, c_end in cohorts:
            base_res = runner(replace(baseline_template, start=c_start, end=c_end))
            cand_res = runner(replace(candidate_template, start=c_start, end=c_end))
            c_wealths.append(float(cand_res.terminal_wealth_real_krw))
            b_wealths.append(float(base_res.terminal_wealth_real_krw))

        ratios = tuple(c / b for c, b in zip(c_wealths, b_wealths))
        median_ratio = wealth_quantile(ratios, 0.5)
        p10_ratio = wealth_quantile(ratios, 0.1)
        worst_ratio = min(ratios)
        win_rate = cohort_win_rate(c_wealths, b_wealths)
        # CE gammas on candidate wealth vector (len>=1)
        ce2 = certainty_equivalent(c_wealths, gamma=2.0) / certainty_equivalent(b_wealths, gamma=2.0)
        ce5 = certainty_equivalent(c_wealths, gamma=5.0) / certainty_equivalent(b_wealths, gamma=5.0)
        ce10 = certainty_equivalent(c_wealths, gamma=10.0) / certainty_equivalent(b_wealths, gamma=10.0)

        # Full-span attribution and path bootstrap: use most recent 120M window
        # most recent cohort is cohorts[-1]; use that window for full-span
        full_start, full_end = cohorts[-1]
        base_full = runner(replace(baseline_template, start=full_start, end=full_end))
        cand_full = runner(replace(candidate_template, start=full_start, end=full_end))
        # attribution
        attribution = attribute_buy_only_soxx(candidate=cand_full, baseline=base_full, soxx_weight=float(w), price_at=price_at, fx_at=fx_at)
        # path bootstrap on monthly returns of full-span path
        cand_rets = monthly_simple_returns(cand_full)
        base_rets = monthly_simple_returns(base_full)
        # ensure lengths match (sessions should align); if mismatch due to snapshot count diff, fail closed
        if len(cand_rets) != len(base_rets):
            raise ValueError(f"candidate/baseline return lengths diverge {len(cand_rets)} vs {len(base_rets)}")
        # use block_size 12 per spec default
        verdict = paired_path_block_bootstrap(cand_rets, base_rets, block_size=12, n_paths=int(bootstrap_paths), seed=int(seed) + idx)

        arms.append(
            IncrementalArmReport(
                arm_id=arm_id,
                soxx_weight=float(w),
                median_ratio=float(median_ratio),
                p10_ratio=float(p10_ratio),
                worst_ratio=float(worst_ratio),
                win_rate=float(win_rate),
                cohort_count=len(cohorts),
                ce_gamma_2=float(ce2),
                ce_gamma_5=float(ce5),
                ce_gamma_10=float(ce10),
                attribution=attribution,
                path_bootstrap=verdict,
            )
        )

    status = classify_portfolio_status(arms)
    return IncrementalPortfolioReport(
        thesis_id="ai_compute",
        as_of=as_of,
        panel_as_of=panel.panel_as_of,
        lag_days=int(panel.lag_days),
        freshness_status=str(panel.status.value),
        arms=tuple(arms),
        portfolio_status=status,
    )


def write_incremental_portfolio_report(report: IncrementalPortfolioReport, path: Path) -> Path:
    payload = {
        "thesis_id": report.thesis_id,
        "as_of": report.as_of.isoformat(),
        "panel_as_of": report.panel_as_of.isoformat(),
        "lag_days": int(report.lag_days),
        "freshness_status": str(report.freshness_status),
        "portfolio_status": report.portfolio_status.value,
        "arms": [
            {
                "arm_id": a.arm_id.value,
                "soxx_weight": float(a.soxx_weight),
                "median_ratio": float(a.median_ratio),
                "p10_ratio": float(a.p10_ratio),
                "worst_ratio": float(a.worst_ratio),
                "win_rate": float(a.win_rate),
                "cohort_count": int(a.cohort_count),
                "ce_gamma_2": float(a.ce_gamma_2),
                "ce_gamma_5": float(a.ce_gamma_5),
                "ce_gamma_10": float(a.ce_gamma_10),
                "attribution": {
                    "target_soxx_weight": float(a.attribution.target_soxx_weight),
                    "mean_realized_soxx_weight": float(a.attribution.mean_realized_soxx_weight),
                    "terminal_realized_soxx_weight": float(a.attribution.terminal_realized_soxx_weight),
                    "mean_abs_weight_drift": float(a.attribution.mean_abs_weight_drift),
                    "terminal_weight_drift": float(a.attribution.terminal_weight_drift),
                    "incremental_wealth_ratio": float(a.attribution.incremental_wealth_ratio),
                },
                "path_bootstrap": {
                    "n_paths": int(a.path_bootstrap.n_paths),
                    "win_rate": float(a.path_bootstrap.win_rate),
                    "p05_terminal_ratio": float(a.path_bootstrap.p05_terminal_ratio),
                    "ok": bool(a.path_bootstrap.ok),
                },
            }
            for a in report.arms
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
