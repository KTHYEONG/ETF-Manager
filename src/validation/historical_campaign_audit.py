# ruff: noqa: PERF401,S110
"""Historical campaign evidence layer: regime coverage, proxy stress, and trial lineage."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal

from src.data.settings import DataSettings
from src.policy.targets import PolicyId
from src.sim.research_proxy import (
    run_research_proxy_from_store_with_returns,
    synthesize_proxy_mix_returns,
)

__all__ = [
    "REGIME_COVERAGE_CATALOG",
    "PreHistoryMixProxyStressReport",
    "PreHistoryProxyStressReport",
    "RegimeCoverageReport",
    "RegimeCoverageRow",
    "RegimeWindow",
    "TrialLineageCensusReport",
    "TrialLineageFamilyRow",
    "audit_pre_history_mix_proxy_stress",
    "audit_pre_history_proxy_stress",
    "audit_regime_coverage",
    "build_trial_lineage_census",
    "classify_regime_coverage_tier",
]

@dataclass(frozen=True, slots=True)
class RegimeWindow:
    regime_name: str
    start: date
    end: date


@dataclass(frozen=True, slots=True)
class RegimeCoverageRow:
    regime_name: str
    covered: bool
    overlap_months: int
    coverage_tier: str = "none"
    coverage_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class RegimeCoverageReport:
    rows: tuple[RegimeCoverageRow, ...]
    independent_sample_warning: bool


@dataclass(frozen=True, slots=True)
class PreHistoryProxyStressReport:
    status: Literal["available", "unavailable"]
    reason: str
    proxy_window_start: date | None = None
    proxy_window_end: date | None = None
    terminal_wealth_real_krw: float | None = None
    xirr_real: float | None = None


@dataclass(frozen=True, slots=True)
class PreHistoryMixProxyStressReport:
    evidence_tier: str
    status: str
    regime_name: str = ""
    baseline_terminal_real_krw: float | None = None
    candidate_terminal_real_krw: float | None = None
    candidate_over_baseline_ratio: float | None = None
    window_start: date | None = None
    window_end: date | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class TrialLineageFamilyRow:
    family_id: str
    experiment_count: int
    active_count: int
    archived_count: int


@dataclass(frozen=True, slots=True)
class TrialLineageCensusReport:
    total_experiments: int
    families: tuple[TrialLineageFamilyRow, ...]


REGIME_COVERAGE_CATALOG: Final[tuple[RegimeWindow, ...]] = (
    RegimeWindow(regime_name="dot_com", start=date(1998, 3, 1), end=date(2002, 10, 31)),
    RegimeWindow(regime_name="gfc", start=date(2007, 10, 1), end=date(2009, 3, 31)),
    RegimeWindow(regime_name="low_rate_2010s", start=date(2010, 1, 4), end=date(2019, 12, 31)),
    RegimeWindow(regime_name="covid", start=date(2020, 2, 1), end=date(2020, 4, 30)),
    RegimeWindow(regime_name="inflation_2022", start=date(2022, 1, 3), end=date(2022, 12, 30)),
    RegimeWindow(regime_name="ai_boom_2023", start=date(2023, 1, 3), end=date(2026, 6, 30)),
)


def _months_between_inclusive(start: date, end: date) -> int:
    if start > end:
        return 0
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def _overlap_months(a_start: date, a_end: date, b_start: date, b_end: date) -> int:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_start > inter_end:
        return 0
    return _months_between_inclusive(inter_start, inter_end)


def classify_regime_coverage_tier(*, overlap_months: int, regime_duration_months: int) -> str:
    if regime_duration_months <= 0:
        return "none"
    fraction = float(overlap_months) / float(regime_duration_months)
    if fraction >= 0.90:
        return "full"
    if fraction >= 0.50:
        return "substantial"
    if fraction > 0:
        return "partial"
    return "none"


def audit_pre_history_mix_proxy_stress(
    settings: DataSettings,
    *,
    window_start: date,
    window_end: date,
    contribution_krw: float,
    baseline_series: str,
    candidate_weights: Mapping[str, float],
    regime_name: str = "",
) -> PreHistoryMixProxyStressReport:
    import polars as pl

    from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
    from src.data.catalog import load_snapshot_visible, resolve_snapshot
    from src.data.schedule import build_decision_schedule
    from src.data.schema import Dataset
    from src.sim.allocation import AllocationConfig
    from src.validation.prospective_registry import _allocation_end_within_as_of

    if not baseline_series or not candidate_weights:
        return PreHistoryMixProxyStressReport(
            evidence_tier="proxy_stress_only",
            status="unavailable",
            regime_name=str(regime_name),
            window_start=window_start,
            window_end=window_end,
            reason="baseline_series and candidate_weights required",
        )
    try:
        effective_end = _allocation_end_within_as_of(
            start=window_start,
            as_of=window_end,
            fill_delay_sessions=1,
        )
    except Exception:
        effective_end = window_end
    try:
        snapshot = resolve_snapshot(settings, (Dataset.RESEARCH_RETURNS, Dataset.FX, Dataset.CPI))
        schedule = build_decision_schedule(window_start, effective_end, fill_delay_sessions=1)
        if not schedule:
            raise ValueError(f"empty proxy schedule over [{window_start.isoformat()}, {effective_end.isoformat()}]")
        cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
        all_returns = load_snapshot_visible(snapshot, Dataset.RESEARCH_RETURNS, cutoff)
        required_series = {str(baseline_series), *(str(k) for k in candidate_weights)}
        available_series = {str(v) for v in all_returns.get_column("series_id").unique().to_list()}
        missing = required_series - available_series
        if missing:
            raise ValueError(f"missing research_returns series: {sorted(missing)!r}")

        base_returns = all_returns.filter(pl.col("series_id") == str(baseline_series))
        mix_source = all_returns.filter(pl.col("series_id").is_in(list(candidate_weights.keys())))
        cand_returns = synthesize_proxy_mix_returns(mix_source, candidate_weights)

        proxy_cfg = AllocationConfig(
            policy=PolicyId.FF_PROXY,
            start=window_start,
            end=effective_end,
            monthly_contribution_krw=float(contribution_krw),
            fill_delay_sessions=1,
            commission_bps=0.0,
            fx_spread_bps=0.0,
        )
        base_res = run_research_proxy_from_store_with_returns(proxy_cfg, settings, base_returns)
        cand_res = run_research_proxy_from_store_with_returns(proxy_cfg, settings, cand_returns)
        base_tw = float(base_res.terminal_wealth_real_krw)
        cand_tw = float(cand_res.terminal_wealth_real_krw)
        if not math.isfinite(base_tw) or base_tw <= 0.0:
            raise ValueError(f"non-positive baseline proxy wealth {base_tw!r}")
        ratio = float(cand_tw / base_tw)
        return PreHistoryMixProxyStressReport(
            evidence_tier="proxy_stress_only",
            status="available",
            regime_name=str(regime_name),
            baseline_terminal_real_krw=base_tw,
            candidate_terminal_real_krw=cand_tw,
            candidate_over_baseline_ratio=ratio,
            window_start=window_start,
            window_end=effective_end,
            reason="ndx_sox_proxy_mix",
        )
    except Exception as exc:
        return PreHistoryMixProxyStressReport(
            evidence_tier="proxy_stress_only",
            status="unavailable",
            regime_name=str(regime_name),
            baseline_terminal_real_krw=None,
            candidate_terminal_real_krw=None,
            candidate_over_baseline_ratio=None,
            window_start=window_start,
            window_end=effective_end if "effective_end" in locals() else window_end,
            reason=str(exc),
        )


def audit_regime_coverage(
    *,
    cohorts: Sequence[tuple[date, date]],
    catalog: Sequence[RegimeWindow] | None = None,
) -> RegimeCoverageReport:
    catalog_seq = tuple(catalog) if catalog is not None else REGIME_COVERAGE_CATALOG
    rows: list[RegimeCoverageRow] = []
    for regime in catalog_seq:
        max_overlap = 0
        covered = False
        for c_start, c_end in cohorts:
            ov = _overlap_months(c_start, c_end, regime.start, regime.end)
            if ov > 0:
                covered = True
            if ov > max_overlap:
                max_overlap = ov
        regime_duration = _months_between_inclusive(regime.start, regime.end)
        coverage_fraction = float(max_overlap) / float(regime_duration) if regime_duration > 0 else 0.0
        coverage_tier = classify_regime_coverage_tier(overlap_months=max_overlap, regime_duration_months=regime_duration)
        rows.append(
            RegimeCoverageRow(
                regime_name=regime.regime_name,
                covered=covered,
                overlap_months=max_overlap,
                coverage_tier=coverage_tier,
                coverage_fraction=float(coverage_fraction),
            )
        )
    # independent_sample_warning: step < horizon
    warning = False
    if len(cohorts) >= 2:
        # estimate horizon and step from first two cohorts
        h = _months_between_inclusive(cohorts[0][0], cohorts[0][1])
        s0 = cohorts[0][0]
        s1 = cohorts[1][0]
        step_months = (s1.year - s0.year) * 12 + (s1.month - s0.month)
        # horizon months approximate as inclusive months
        warning = step_months < h
    elif len(cohorts) == 1:
        warning = True
    else:
        warning = False
    return RegimeCoverageReport(rows=tuple(rows), independent_sample_warning=warning)


def _classify_family(filename: str) -> str:
    lower = filename.lower()
    if "adaptive" in lower:
        return "adaptive"
    if "reserve" in lower:
        return "reserve"
    if "pave" in lower or "ai_power" in lower:
        return "pave"
    if "physical_automation" in lower or "robo" in lower or "botz" in lower:
        return "robo"
    if "grid" in lower:
        return "grid"
    if "soxx" in lower or "ai_compute" in lower:
        return "soxx"
    if "cadence" in lower:
        return "cadence"
    if "overlay" in lower:
        return "overlay"
    if "mapping" in lower:
        return "mapping"
    if "currency" in lower:
        return "currency"
    if "ff_proxy" in lower:
        return "proxy"
    return "qqq_vti"


def build_trial_lineage_census(
    *,
    index_path: Path,
    experiments_dir: Path,
) -> TrialLineageCensusReport:
    """Count previously executed trial families from the existing lineage index.

    Args:
        index_path: Existing trial index path.
        experiments_dir: Existing experiment definition directory.

    Returns:
        The existing lineage census report.

    Raises:
        ValueError: If the index is malformed or required evidence is inconsistent.
    """
    # experiments_dir is noted but not strictly required; keep for wiring spec
    _ = experiments_dir
    try:
        text = index_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - missing index is an environment failure
        raise ValueError(f"malformed trial index at {index_path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:  # pragma: no cover - malformed JSON fixture
        raise ValueError(f"malformed trial index at {index_path}: {exc}") from exc
    if not isinstance(payload, dict):  # pragma: no cover - malformed envelope
        raise ValueError(f"malformed trial index at {index_path}: expected object payload")
    files = payload.get("files", {})
    if not isinstance(files, dict):  # pragma: no cover - malformed envelope
        raise ValueError(f"malformed trial index at {index_path}: 'files' must be an object")
    families: dict[str, dict[str, int]] = {}
    for filename, meta in files.items():
        if not isinstance(meta, dict):  # pragma: no cover - malformed entry envelope
            raise ValueError(f"malformed trial index entry for {filename!r}: expected object")
        status = str(meta.get("status", "")).lower()
        if status not in ("active", "archived"):
            # skip fixture and others; they are not counted in active+archived census
            continue
        fam = _classify_family(str(filename))
        entry = families.setdefault(fam, {"experiment_count": 0, "active_count": 0, "archived_count": 0})
        entry["experiment_count"] += 1
        if status == "active":
            entry["active_count"] += 1
        elif status == "archived":
            entry["archived_count"] += 1
    # ensure families for empty? If no files, empty.
    rows = tuple(
        TrialLineageFamilyRow(
            family_id=fam,
            experiment_count=v["experiment_count"],
            active_count=v["active_count"],
            archived_count=v["archived_count"],
        )
        for fam, v in sorted(families.items())
    )
    total = sum(r.experiment_count for r in rows)
    return TrialLineageCensusReport(total_experiments=total, families=rows)


def _catalog_research_returns_min_date(settings: DataSettings) -> date | None:
    from src.data.catalog import resolve_snapshot
    from src.data.schema import Dataset, spec_for
    from src.data.storage import DataStore

    snapshot = resolve_snapshot(settings, (Dataset.RESEARCH_RETURNS,))
    df = DataStore(settings).read_normalized(snapshot.artifacts[Dataset.RESEARCH_RETURNS], spec_for(Dataset.RESEARCH_RETURNS))
    if df.is_empty():
        return None
    min_val = df.get_column("date").min()
    if min_val is None:
        return None
    return min_val if isinstance(min_val, date) else date.fromisoformat(str(min_val))


def audit_pre_history_proxy_stress(
    settings: DataSettings,
    *,
    proxy_start: date,
    proxy_end: date,
    contribution_krw: float,
    fallback_starts: Sequence[date] = (),
) -> PreHistoryProxyStressReport:
    from src.sim.allocation import AllocationConfig
    from src.sim.research_proxy import run_research_proxy_from_store
    from src.validation.prospective_registry import _allocation_end_within_as_of

    catalog_min = _catalog_research_returns_min_date(settings)
    start_candidates: list[date] = [proxy_start]
    if catalog_min is not None:
        start_candidates.append(catalog_min)
    for fb in fallback_starts:
        if fb not in start_candidates:
            start_candidates.append(fb)
    last_error = "no_proxy_window_attempted"
    for effective_start in start_candidates:
        if effective_start > proxy_end:
            last_error = "proxy_start_after_end"
            continue
        reason_tag = (
            f"ff_proxy_catalog_min_{effective_start.isoformat()}"
            if catalog_min is not None and effective_start == catalog_min and proxy_start < catalog_min
            else f"ff_proxy_from_{effective_start.isoformat()}"
        )
        try:
            effective_end = _allocation_end_within_as_of(
                start=effective_start,
                as_of=proxy_end,
                fill_delay_sessions=1,
            )
            cfg = AllocationConfig(
                policy=PolicyId.FF_PROXY,
                start=effective_start,
                end=effective_end,
                monthly_contribution_krw=float(contribution_krw),
                fill_delay_sessions=1,
                commission_bps=0.0,
                fx_spread_bps=0.0,
            )
            result = run_research_proxy_from_store(cfg, settings)
            tw = float(result.terminal_wealth_real_krw)
            if not math.isfinite(tw) or tw <= 0.0:
                raise ValueError(f"non-positive proxy terminal wealth {tw!r}")
            xirr = float(result.xirr_real) if math.isfinite(float(result.xirr_real)) else None
            return PreHistoryProxyStressReport(
                status="available",
                reason=reason_tag,
                proxy_window_start=effective_start,
                proxy_window_end=effective_end,
                terminal_wealth_real_krw=tw,
                xirr_real=xirr,
            )
        except Exception as exc:
            last_error = str(exc)
    return PreHistoryProxyStressReport(
        status="unavailable",
        reason=last_error,
        proxy_window_start=start_candidates[0],
        proxy_window_end=proxy_end,
    )
