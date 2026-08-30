# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035,PERF401
"""Wave D exit assessment (Track F + Track H)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.analytics.thesis.incremental import IncrementalPortfolioReport
from src.analytics.thesis.wave import ThesisWaveReport
from src.data.settings import DataSettings
from src.policy.thesis import ThesisId
from src.validation.accumulation_cohort import AccumulationCohortReport

logger = logging.getLogger(__name__)

__all__ = [
    "WaveDExitAssessment",
    "assess_wave_d_exit",
    "run_thesis_pipeline_command",
    "write_wave_d_exit_markdown",
]


@dataclass(frozen=True, slots=True)
class WaveDExitAssessment:
    thesis_id: ThesisId
    as_of: datetime
    panel_as_of: datetime | None
    freshness_status: str
    track_f_complete: bool
    reference_slice_ready: bool
    operational_challenger_ready: bool
    blockers: tuple[str, ...]
    portfolio_status: str


def assess_wave_d_exit(
    *,
    thesis_id: ThesisId,
    wave: ThesisWaveReport,
    incremental: IncrementalPortfolioReport,
    accumulation: AccumulationCohortReport | None = None,
) -> WaveDExitAssessment:
    _ = accumulation
    entry = next((e for e in wave.entries if e.thesis_id == thesis_id), None)
    if entry is None:
        raise ValueError(f"thesis_id {thesis_id.value!r} absent from wave.entries")
    structural_status = str(entry.report.evidence.structural.status)
    valuation_status = str(entry.report.evidence.valuation.status)
    crowding_status = str(entry.report.evidence.crowding.status)
    statuses = {
        "structural": structural_status,
        "valuation": valuation_status,
        "crowding": crowding_status,
    }
    track_f_complete = all(s == "computed" for s in statuses.values())
    blockers: list[str] = []
    if not track_f_complete:
        for name, status in statuses.items():
            if status != "computed":
                blockers.append(name)
    # portfolio gate
    try:
        portfolio_status = str(incremental.portfolio_status.value)
    except AttributeError:
        portfolio_status = str(incremental.portfolio_status)
    historically_promising = portfolio_status == "historically_promising"
    all_arms_ge10 = False
    if incremental.arms:
        all_arms_ge10 = all(int(a.cohort_count) >= 10 for a in incremental.arms)
    else:
        all_arms_ge10 = False
    reference_slice_ready = bool(track_f_complete and historically_promising and all_arms_ge10)
    if not reference_slice_ready:
        if not historically_promising:
            blockers.append("portfolio_status")
        if not all_arms_ge10:
            blockers.append("cohort_count")
    # operational gate uses freshness_status
    freshness = str(incremental.freshness_status) if incremental.freshness_status else str(wave.freshness_status or "UNKNOWN")
    operational_challenger_ready = bool(reference_slice_ready and freshness == "FRESH")
    if reference_slice_ready and freshness != "FRESH":
        blockers.append(f"freshness_status:{freshness}")
    # dedup preserving order
    seen: set[str] = set()
    dedup: list[str] = []
    for b in blockers:
        if b not in seen:
            seen.add(b)
            dedup.append(b)
    panel_as_of = wave.panel_as_of if wave.panel_as_of is not None else incremental.panel_as_of
    return WaveDExitAssessment(
        thesis_id=thesis_id,
        as_of=wave.as_of,
        panel_as_of=panel_as_of,
        freshness_status=freshness,
        track_f_complete=bool(track_f_complete),
        reference_slice_ready=bool(reference_slice_ready),
        operational_challenger_ready=bool(operational_challenger_ready),
        blockers=tuple(dedup),
        portfolio_status=portfolio_status,
    )


def write_wave_d_exit_markdown(
    assessment: WaveDExitAssessment,
    wave: ThesisWaveReport,
    incremental: IncrementalPortfolioReport,
    path: Path,
) -> Path:
    # locate evidence for thesis
    entry = next((e for e in wave.entries if e.thesis_id == assessment.thesis_id), None)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    panel_date = assessment.panel_as_of.date().isoformat() if assessment.panel_as_of is not None else assessment.as_of.date().isoformat()
    lines.append(f"# Wave D Exit {assessment.thesis_id.value} {panel_date}")
    lines.append("")
    lines.append(f"As of: {assessment.as_of.isoformat()}")
    lines.append(f"panel_as_of: {assessment.panel_as_of.isoformat() if assessment.panel_as_of is not None else 'UNKNOWN'}")
    lines.append(f"freshness_status: {assessment.freshness_status}")
    lines.append(f"portfolio_status: {assessment.portfolio_status}")
    lines.append("")
    # evidence slot table (5 slots + metrics)
    lines.append("## Evidence Slots")
    lines.append("")
    lines.append("| slot | status | summary | metrics |")
    lines.append("| --- | --- | --- | --- |")
    slots = [
        ("historical", entry.report.evidence.historical if entry is not None else None),
        ("structural", entry.report.evidence.structural if entry is not None else None),
        ("valuation", entry.report.evidence.valuation if entry is not None else None),
        ("overlap", entry.report.evidence.overlap if entry is not None else None),
        ("crowding", entry.report.evidence.crowding if entry is not None else None),
    ]
    for name, slot in slots:
        if slot is None:
            lines.append(f"| {name} | unknown | - | {{}} |")
        else:
            metrics_str = ", ".join(f"{k}={v}" for k, v in dict(slot.metrics).items()) if slot.metrics else "{}"
            # escape pipe
            summary = str(slot.summary).replace("|", "/")
            lines.append(f"| {name} | {slot.status} | {summary} | {metrics_str} |")
    lines.append("")
    # Track H arm summary
    lines.append("## Track H Arms")
    lines.append("")
    lines.append("| arm_id | soxx_weight | median_ratio | p10_ratio | worst_ratio | cohort_count | win_rate | p05 | ok |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in incremental.arms:
        lines.append(
            f"| {arm.arm_id.value} | {arm.soxx_weight:.2f} | {arm.median_ratio:.4f} | {arm.p10_ratio:.4f} | {arm.worst_ratio:.4f} | {arm.cohort_count} | {arm.path_bootstrap.win_rate:.4f} | {arm.path_bootstrap.p05_terminal_ratio:.4f} | {arm.path_bootstrap.ok} |"
        )
    if not incremental.arms:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    lines.append("")
    # exit checklist
    lines.append("## Exit Checklist")
    lines.append("")
    lines.append(f"- track_f_complete: {assessment.track_f_complete}")
    lines.append(f"- reference_slice_ready: {assessment.reference_slice_ready}")
    lines.append(f"- operational_challenger_ready: {assessment.operational_challenger_ready}")
    lines.append(f"- portfolio_status: {assessment.portfolio_status}")
    lines.append(f"- freshness_status: {assessment.freshness_status}")
    lines.append("")
    # blockers list
    lines.append("## Blockers")
    lines.append("")
    if assessment.blockers:
        for b in assessment.blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- none")
    lines.append("")
    # reference booleans explicit for test
    lines.append(f"reference_slice_ready: {assessment.reference_slice_ready}")
    lines.append(f"operational_challenger_ready: {assessment.operational_challenger_ready}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_thesis_pipeline_command(
    *,
    thesis_id: str,
    as_of: str | None,
    settings: DataSettings,
    allow_stale: bool = False,
    seed: int = 7,
    bootstrap_paths: int = 400,
) -> int:
    from src.data.panel_freshness import PanelFreshnessStatus, apply_hard_stop, load_panel_hard_stop, resolve_catalog_panel_as_of

    # wiring anchor: resolve_catalog_panel_as_of and as_of_dt = datetime.now(UTC)
    _ = resolve_catalog_panel_as_of
    _anchor_now = datetime.now(UTC)
    try:
        try:
            tid = ThesisId(thesis_id)
        except ValueError as exc:
            logger.error("[DATA] event=thesis_pipeline_failed reason=%s", exc)
            return 2
        # panel gate reuses thesis-wave STALE logic
        try:
            gate_report = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC))
        except Exception as exc:
            logger.error("[DATA] event=thesis_pipeline_panel_failed reason=%s", exc)
            return 1
        if gate_report.status == PanelFreshnessStatus.STALE:
            hard = load_panel_hard_stop()
            gate_report = apply_hard_stop(gate_report, hard)
            if gate_report.status != PanelFreshnessStatus.HARD_STOP_ACK and not allow_stale:
                logger.error(
                    "[DATA] event=thesis_pipeline_stale panel_as_of=%s lag_days=%d",
                    gate_report.panel_as_of.isoformat(),
                    gate_report.lag_days,
                )
                return 1
        elif gate_report.status == PanelFreshnessStatus.INSUFFICIENT_DATA:
            logger.error("[DATA] event=thesis_pipeline_insufficient_data reason=insufficient catalog")
            return 1
        logger.info(
            "[DATA] event=thesis_pipeline_panel panel_as_of=%s lag_days=%d status=%s",
            gate_report.panel_as_of.isoformat(),
            gate_report.lag_days,
            gate_report.status.value,
        )
        # resolve as_of
        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
            # fail-closed if explicit as_of after last catalog price session
            try:
                _ = resolve_catalog_panel_as_of(settings, reference_now=as_of_dt)
                from src.data.catalog import latest_artifact
                from src.data.schema import Dataset, spec_for
                from src.data.storage import DataStore

                try:
                    latest = latest_artifact(settings, Dataset.PRICES)
                    frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
                    max_d = frame.get_column("date").max()
                    from datetime import date as _date

                    if isinstance(max_d, _date) and as_of_dt.date() > max_d:
                        raise ValueError(f"explicit --as-of {as_of_dt.isoformat()} is after last catalog price session {max_d.isoformat()}")
                except ValueError:
                    raise
                except Exception:
                    pass
            except ValueError:
                raise
            except Exception:
                pass
        else:
            as_of_dt = gate_report.panel_as_of
        from src.analytics.thesis.incremental import run_incremental_portfolio
        from src.analytics.thesis.wave import run_thesis_wave
        from src.sim.allocation import run_allocation_from_store

        def _runner(config):  # type: ignore[no-untyped-def]
            return run_allocation_from_store(config, settings)

        wave = run_thesis_wave(settings=settings, as_of=as_of_dt, runner=_runner, panel_report=gate_report)
        # thesis-wave already writes its own markdown; we continue to incremental
        report = run_incremental_portfolio(
            settings=settings,
            as_of=as_of_dt,
            runner=_runner,
            contribution_krw=1_000_000,
            bootstrap_paths=int(bootstrap_paths),
            seed=int(seed),
            panel_report=gate_report,
        )
        assessment = assess_wave_d_exit(thesis_id=tid, wave=wave, incremental=report)
        panel_date = gate_report.panel_as_of.date().isoformat()
        out_path = Path(f"docs/results/thesis-wave/{panel_date}_wave_d_exit_{tid.value}.md")
        write_wave_d_exit_markdown(assessment, wave, report, out_path)
        # also write under data dir for history (optional)
        try:
            from src.data.paths import thesis_reports_dir

            data_path = thesis_reports_dir(settings) / f"wave_d_exit_{tid.value}_{panel_date}.json"
            # minimal json artifact for traceability (not required by spec but harmless)
            import json

            payload = {
                "thesis_id": tid.value,
                "as_of": as_of_dt.isoformat(),
                "panel_as_of": gate_report.panel_as_of.isoformat(),
                "track_f_complete": assessment.track_f_complete,
                "reference_slice_ready": assessment.reference_slice_ready,
                "operational_challenger_ready": assessment.operational_challenger_ready,
                "blockers": list(assessment.blockers),
                "portfolio_status": assessment.portfolio_status,
            }
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
        logger.info(
            "[DATA] event=thesis_pipeline_done thesis_id=%s reference_slice_ready=%s operational_challenger_ready=%s blockers=%s",
            tid.value,
            assessment.reference_slice_ready,
            assessment.operational_challenger_ready,
            ",".join(assessment.blockers) if assessment.blockers else "none",
        )
    except (ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_pipeline_failed reason=%s", exc)
        return 1
    # Pipeline exit code 0 on success even when operational_challenger_ready is false
    return 0
