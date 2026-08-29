"""Thesis wave E2E (Wave 7)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.analytics.thesis_decision import ThesisDecisionRecord, synthesize_thesis_decision
from src.analytics.thesis_report import ThesisReport, build_thesis_report, write_thesis_report
from src.data.settings import DataSettings
from src.policy.thesis import ThesisId
from src.sim.allocation import AllocationConfig, AllocationResult

logger = logging.getLogger(__name__)

__all__ = [
    "ThesisWaveEntry",
    "ThesisWaveFailure",
    "ThesisWaveReport",
    "load_thesis_experiment_map",
    "run_thesis_wave",
    "write_thesis_wave_markdown",
]


@dataclass(frozen=True, slots=True)
class ThesisWaveFailure:
    thesis_id: ThesisId
    experiment_path: Path
    error: str


@dataclass(frozen=True, slots=True)
class ThesisWaveEntry:
    thesis_id: ThesisId
    report: ThesisReport
    decision: ThesisDecisionRecord
    experiment_path: Path


@dataclass(frozen=True, slots=True)
class ThesisWaveReport:
    as_of: datetime
    entries: tuple[ThesisWaveEntry, ...]
    failures: tuple[ThesisWaveFailure, ...] = ()


def load_thesis_experiment_map(path: Path = Path("configs/theses/experiment_map.json")) -> Mapping[ThesisId, Path]:
    """Load thesis -> experiment JSON map; fails closed on missing keys."""
    if not path.is_file():
        raise ValueError(f"thesis experiment map not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"experiment map unreadable at {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"experiment map root must be object at {path}")
    result: dict[ThesisId, Path] = {}
    for key, value in doc.items():
        try:
            tid = ThesisId(key)
        except ValueError as exc:
            raise ValueError(f"unknown thesis id {key!r} in map {path}") from exc
        result[tid] = Path(str(value))
    # Require all three thesis ids present
    required = {ThesisId.AI_COMPUTE, ThesisId.AI_POWER_BOTTLENECK, ThesisId.PHYSICAL_AUTOMATION}
    missing = required - set(result.keys())
    if missing:
        raise ValueError(f"experiment map missing thesis keys: {sorted(m.value for m in missing)}")
    return result


def run_thesis_wave(
    *,
    settings: DataSettings,
    as_of: datetime,
    runner: Callable[[AllocationConfig], AllocationResult],
    include_regime: bool = True,
) -> ThesisWaveReport:
    """Iterate theses in fixed order; skip failures and emit partial wave JSON."""
    experiment_map = load_thesis_experiment_map()
    order = [ThesisId.AI_COMPUTE, ThesisId.AI_POWER_BOTTLENECK, ThesisId.PHYSICAL_AUTOMATION]
    entries: list[ThesisWaveEntry] = []
    failures: list[ThesisWaveFailure] = []
    for thesis_id in order:
        try:
            exp_path = experiment_map[thesis_id]
        except KeyError as exc:
            raise ValueError(f"experiment map missing thesis key {thesis_id.value}: {exc}") from exc
        try:
            report = build_thesis_report(
                thesis_id=thesis_id,
                settings=settings,
                as_of=as_of,
                runner=runner,
                experiment_path=exp_path,
                include_regime=include_regime,
            )
        except Exception as exc:
            msg = str(exc)[:500]
            logger.error("[DATA] event=thesis_wave_thesis_failed thesis_id=%s reason=%s", thesis_id.value, msg)
            failures.append(ThesisWaveFailure(thesis_id=thesis_id, experiment_path=exp_path, error=msg))
            continue
        decision = synthesize_thesis_decision(report)
        write_thesis_report(report, settings)
        entries.append(ThesisWaveEntry(thesis_id=thesis_id, report=report, decision=decision, experiment_path=exp_path))

    wave = ThesisWaveReport(as_of=as_of, entries=tuple(entries), failures=tuple(failures))
    # Write combined wave JSON under data/thesis_reports/wave_{as_of}.json
    out_dir = settings.resolved_data_root() / "thesis_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_as_of = as_of.isoformat().replace(":", "-")
    wave_path = out_dir / f"wave_{safe_as_of}.json"
    payload = {
        "as_of": as_of.isoformat(),
        "entries": [
            {
                "thesis_id": e.thesis_id.value,
                "experiment_path": str(e.experiment_path),
                "decision": e.decision.decision.value,
                "rationale": e.decision.rationale,
                "suggested_status": e.report.suggested_status.value,
                "historical": {
                    "status": e.report.evidence.historical.status,
                    "metrics": dict(e.report.evidence.historical.metrics),
                },
                "structural": {
                    "status": e.report.evidence.structural.status,
                    "metrics": dict(e.report.evidence.structural.metrics),
                },
                "overlap": {
                    "status": e.report.evidence.overlap.status,
                    "metrics": dict(e.report.evidence.overlap.metrics),
                },
            }
            for e in entries
        ],
        "failures": [
            {
                "thesis_id": f.thesis_id.value,
                "experiment_path": str(f.experiment_path),
                "error": f.error,
            }
            for f in failures
        ],
    }
    wave_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return wave


def write_thesis_wave_markdown(wave: ThesisWaveReport, path: Path) -> Path:
    """Emit markdown table with thesis_id, decision, historical median, overlap_pct, suggested_status."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Thesis Wave {wave.as_of.date().isoformat()}")
    lines.append("")
    lines.append(f"As of: {wave.as_of.isoformat()}")
    lines.append("")
    lines.append("| thesis_id | decision | historical median | overlap_pct | suggested_status |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in wave.entries:
        hist_metrics = entry.report.evidence.historical.metrics
        median = hist_metrics.get("median_ratio", "")
        try:
            median_str = f"{float(median):.4f}" if median != "" else ""
        except Exception:
            median_str = str(median)
        overlap_metrics = entry.report.evidence.overlap.metrics
        overlap_pct = overlap_metrics.get("overlap_pct", "")
        try:
            overlap_str = f"{float(overlap_pct):.1f}" if overlap_pct != "" else ""
        except Exception:
            overlap_str = str(overlap_pct)
        lines.append(
            f"| {entry.thesis_id.value} | {entry.decision.decision.value} | {median_str} | {overlap_str} | {entry.report.suggested_status.value} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
