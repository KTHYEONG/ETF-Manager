"""Rolling 120M accumulation cohort report (reporting-only)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from src.analytics.metrics import recovery_months
from src.validation.bootstrap import moving_block_bootstrap
from src.validation.gate import cohort_win_rate, wealth_quantile
from src.validation.windows import rolling_cohorts

if TYPE_CHECKING:
    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.experiment import ExperimentSpec

__all__ = [
    "AccumulationCohortReport",
    "AccumulationCohortRow",
    "CohortOverlapMetadata",
    "run_accumulation_cohort_report",
    "summarize_accumulation_cohorts",
    "write_accumulation_cohort_report",
]


@dataclass(frozen=True, slots=True)
class CohortOverlapMetadata:
    """Overlap between successive rolling cohorts."""

    horizon_months: int
    step_months: int
    overlap_months: int | None = None
    independent_sample_warning: bool | None = None

    def __post_init__(self) -> None:
        if self.horizon_months < 1 or self.step_months < 1:
            raise ValueError("horizon_months and step_months must be >= 1")
        expected_overlap = self.horizon_months - self.step_months
        expected_warning = self.step_months < self.horizon_months
        if self.overlap_months is None:
            object.__setattr__(self, "overlap_months", expected_overlap)
        elif self.overlap_months != expected_overlap:
            raise ValueError(f"overlap_months must be {expected_overlap}, got {self.overlap_months!r}")
        if self.independent_sample_warning is None:
            object.__setattr__(self, "independent_sample_warning", expected_warning)
        elif self.independent_sample_warning != expected_warning:
            raise ValueError(
                f"independent_sample_warning must be {expected_warning!r}, got {self.independent_sample_warning!r}"
            )


@dataclass(frozen=True, slots=True)
class AccumulationCohortRow:
    """One rolling cohort's paired wealths and recovery timing."""

    candidate_wealth: float
    baseline_wealth: float
    ratio: float
    candidate_recovery_months: int | None
    cohort_start: date | None = None
    cohort_end: date | None = None


@dataclass(frozen=True, slots=True)
class AccumulationCohortReport:
    """Reporting-only cohort summary; never calls adoption_passes."""

    name: str
    overlap: CohortOverlapMetadata
    rows: tuple[AccumulationCohortRow, ...]
    median_ratio: float
    p10_ratio: float
    worst_ratio: float
    win_rate: float
    bootstrap_p05_ratio_mean: float
    unrecovered_cohort_count: int


def summarize_accumulation_cohorts(
    *,
    candidate_wealths: Sequence[float],
    baseline_wealths: Sequence[float],
    candidate_recovery_months: Sequence[int | None],
    overlap: CohortOverlapMetadata,
    bootstrap_paths: int,
    seed: int,
) -> AccumulationCohortReport:
    """Summarize paired wealths into cohort ratios and bootstrap tail."""
    if len(candidate_wealths) != len(baseline_wealths) or len(candidate_wealths) != len(candidate_recovery_months):
        raise ValueError("candidate_wealths, baseline_wealths, and candidate_recovery_months must have equal length")
    if len(candidate_wealths) < 1:
        raise ValueError("need at least one cohort")
    if not isinstance(bootstrap_paths, int) or isinstance(bootstrap_paths, bool) or bootstrap_paths < 1:
        raise ValueError(f"bootstrap_paths must be integer >=1, got {bootstrap_paths!r}")
    # Validate wealths finite and strictly positive
    for value in (*candidate_wealths, *baseline_wealths):
        fv = float(value)
        if not math.isfinite(fv) or fv <= 0.0:
            raise ValueError(f"wealths must be finite and strictly positive, got {value!r}")

    ratios = tuple(float(c) / float(b) for c, b in zip(candidate_wealths, baseline_wealths, strict=True))

    median_ratio = wealth_quantile(ratios, 0.5)
    p10_ratio = wealth_quantile(ratios, 0.1)
    worst_ratio = min(ratios)
    win_rate = cohort_win_rate(candidate_wealths, baseline_wealths)

    block_size = max(1, len(ratios) // 2)
    paths = moving_block_bootstrap(ratios, block_size=block_size, n_paths=bootstrap_paths, seed=seed)
    path_means = tuple(sum(p) / len(p) for p in paths)
    bootstrap_p05_ratio_mean = wealth_quantile(path_means, 0.05)

    unrecovered = sum(1 for v in candidate_recovery_months if v is None)

    rows = tuple(
        AccumulationCohortRow(
            candidate_wealth=float(c),
            baseline_wealth=float(b),
            ratio=float(c) / float(b),
            candidate_recovery_months=rm,
        )
        for c, b, rm in zip(candidate_wealths, baseline_wealths, candidate_recovery_months, strict=True)
    )

    return AccumulationCohortReport(
        name="",
        overlap=overlap,
        rows=rows,
        median_ratio=median_ratio,
        p10_ratio=p10_ratio,
        worst_ratio=worst_ratio,
        win_rate=win_rate,
        bootstrap_p05_ratio_mean=bootstrap_p05_ratio_mean,
        unrecovered_cohort_count=unrecovered,
    )


def run_accumulation_cohort_report(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    horizon_months: int = 120,
    step_months: int = 12,
    bootstrap_paths: int = 4000,
    seed: int,
) -> AccumulationCohortReport:
    """Run rolling cohorts for baseline and candidate and summarize."""
    if horizon_months < 1 or step_months < 1:
        raise ValueError("horizon_months and step_months must be >=1")
    if not isinstance(bootstrap_paths, int) or bootstrap_paths < 1:
        raise ValueError(f"bootstrap_paths must be >=1, got {bootstrap_paths!r}")
    if not spec.candidates:
        raise ValueError("spec must have at least one candidate")

    cohorts = rolling_cohorts(spec.start, spec.end, horizon_months=horizon_months, step_months=step_months)
    if not cohorts:
        raise ValueError("no rolling cohorts fit the experiment window")

    # Build templates without mutating spec; use minimal AllocationConfig
    from dataclasses import replace

    from src.sim.allocation import AllocationConfig
    from src.validation.experiment import resolve_arm_targets

    baseline_template = AllocationConfig(
        policy=spec.baseline.policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=float(spec.contribution_krw),
        fill_delay_sessions=1,
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=resolve_arm_targets(spec.baseline),
    )
    candidate_spec = spec.candidates[0]
    candidate_template = AllocationConfig(
        policy=candidate_spec.policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=float(spec.contribution_krw),
        fill_delay_sessions=1,
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=resolve_arm_targets(candidate_spec),
    )

    candidate_wealths: list[float] = []
    baseline_wealths: list[float] = []
    candidate_recovery_months: list[int | None] = []
    rows: list[AccumulationCohortRow] = []

    for c_start, c_end in cohorts:
        base_res = runner(replace(baseline_template, start=c_start, end=c_end))
        cand_res = runner(replace(candidate_template, start=c_start, end=c_end))
        cw = float(cand_res.terminal_wealth_real_krw)
        bw = float(base_res.terminal_wealth_real_krw)
        if not math.isfinite(cw) or cw <= 0.0 or not math.isfinite(bw) or bw <= 0.0:
            raise ValueError(f"wealths must be finite and strictly positive, got {cw!r} vs {bw!r}")
        ratio = cw / bw
        sessions = tuple(s.session for s in cand_res.snapshots)
        marks = tuple(float(s.mark_krw) for s in cand_res.snapshots)
        rec: int | None
        if len(sessions) == 0:
            rec = None
        else:
            # If snapshots empty or marks non-finite, recovery is None
            try:
                rec = recovery_months(sessions, marks)
            except ValueError:
                rec = None
        candidate_wealths.append(cw)
        baseline_wealths.append(bw)
        candidate_recovery_months.append(rec)
        rows.append(
            AccumulationCohortRow(
                candidate_wealth=cw,
                baseline_wealth=bw,
                ratio=ratio,
                candidate_recovery_months=rec,
                cohort_start=c_start,
                cohort_end=c_end,
            )
        )

    overlap = CohortOverlapMetadata(horizon_months=horizon_months, step_months=step_months)

    # Compute summary stats (reuses gate functions)
    ratios = tuple(float(c) / float(b) for c, b in zip(candidate_wealths, baseline_wealths, strict=True))
    median_ratio = wealth_quantile(ratios, 0.5)
    p10_ratio = wealth_quantile(ratios, 0.1)
    worst_ratio = min(ratios)
    win_rate = cohort_win_rate(candidate_wealths, baseline_wealths)
    block_size = max(1, len(ratios) // 2)
    paths = moving_block_bootstrap(ratios, block_size=block_size, n_paths=bootstrap_paths, seed=seed)
    path_means = tuple(sum(p) / len(p) for p in paths)
    bootstrap_p05_ratio_mean = wealth_quantile(path_means, 0.05)
    unrecovered = sum(1 for v in candidate_recovery_months if v is None)

    return AccumulationCohortReport(
        name=spec.name,
        overlap=overlap,
        rows=tuple(rows),
        median_ratio=median_ratio,
        p10_ratio=p10_ratio,
        worst_ratio=worst_ratio,
        win_rate=win_rate,
        bootstrap_p05_ratio_mean=bootstrap_p05_ratio_mean,
        unrecovered_cohort_count=unrecovered,
    )


def write_accumulation_cohort_report(
    report: AccumulationCohortReport, settings: DataSettings, experiment_id: str
) -> Path:
    """Persist report JSON under experiments/{name}_accumulation_{experiment_id}.json."""
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "overlap": {
            "horizon_months": report.overlap.horizon_months,
            "step_months": report.overlap.step_months,
            "overlap_months": report.overlap.overlap_months,
            "independent_sample_warning": report.overlap.independent_sample_warning,
        },
        "summary": {
            "median_ratio": report.median_ratio,
            "p10_ratio": report.p10_ratio,
            "worst_ratio": report.worst_ratio,
            "win_rate": report.win_rate,
            "bootstrap_p05_ratio_mean": report.bootstrap_p05_ratio_mean,
            "unrecovered_cohort_count": report.unrecovered_cohort_count,
            "cohort_count": len(report.rows),
        },
        "cohorts": [
            {
                "cohort_start": row.cohort_start.isoformat() if row.cohort_start is not None else None,
                "cohort_end": row.cohort_end.isoformat() if row.cohort_end is not None else None,
                "candidate_wealth": row.candidate_wealth,
                "baseline_wealth": row.baseline_wealth,
                "ratio": row.ratio,
                "candidate_recovery_months": row.candidate_recovery_months,
            }
            for row in report.rows
        ],
    }
    experiments_dir = settings.resolved_data_root() / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    # Use name or 'accumulation' fallback
    base_name = report.name if report.name else "accumulation"
    out_path = experiments_dir / f"{base_name}_accumulation_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
