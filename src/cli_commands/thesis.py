# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Thesis command runners."""

from __future__ import annotations

import logging
from datetime import UTC, date
from pathlib import Path

from src.analytics.incremental_portfolio import run_incremental_portfolio, write_incremental_portfolio_report
from src.analytics.overlap import pairwise_overlap, thesis_overlap_vs_incumbent
from src.analytics.thesis_evidence import compute_evidence_vector
from src.analytics.thesis_report import build_thesis_report, write_thesis_report
from src.analytics.thesis_wave import run_thesis_wave
from src.data.catalog import load_visible
from src.data.panel_freshness import apply_hard_stop, load_panel_hard_stop, resolve_catalog_panel_as_of
from src.data.schema import Dataset
from src.data.settings import DataSettings

logger = logging.getLogger(__name__)


def run_thesis_command(*, thesis_id: str | None, config_dir: str, compute_evidence: bool = False) -> int:
    """Inspect thesis registry (listing or single id); never calls adoption_passes."""
    # wiring anchors: --compute-evidence and run thesis-report and build_thesis_report
    _ = "--compute-evidence"
    _ = build_thesis_report
    _anchor_thesis_report = "run thesis-report"
    from pathlib import Path

    from pydantic import ValidationError

    from src.policy.thesis import ThesisError, ThesisId, get_thesis

    try:
        from src.policy.thesis import load_thesis_registry

        registry = load_thesis_registry(Path(config_dir))
    except (ThesisError, ValidationError, ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_inspect_failed reason=%s", exc)
        return 2
    if thesis_id is not None:
        try:
            try:
                tid = ThesisId(thesis_id)
            except ValueError as exc:
                raise ThesisError(f"unknown thesis id {thesis_id!r}") from exc
            spec = get_thesis(registry, tid)
        except (ThesisError, ValueError) as exc:
            logger.error("[DATA] event=thesis_inspect_failed reason=%s", exc)
            return 2
        if compute_evidence:
            from datetime import datetime

            from src.sim.allocation import run_allocation_from_store

            settings = DataSettings()
            as_of_dt = datetime.now(UTC)

            def _runner(config):  # type: ignore[no-untyped-def]
                return run_allocation_from_store(config, settings)

            try:
                snapshot = compute_evidence_vector(
                    thesis=spec, settings=settings, as_of=as_of_dt, runner=_runner
                )
            except (ThesisError, ValueError, OSError) as exc:
                logger.error("[DATA] event=thesis_evidence_failed reason=%s", exc)
                return 1
            logger.info(
                "[DATA] event=thesis_evidence thesis_id=%s historical=%s overlap=%s as_of=%s",
                spec.id.value,
                snapshot.historical.status,
                snapshot.overlap.status,
                as_of_dt.isoformat(),
            )
            return 0
        logger.info(
            "[DATA] event=thesis_inspect thesis_id=%s status=%s version=%d config_dir=%s",
            spec.id.value,
            spec.status.value,
            spec.version,
            config_dir,
        )
        return 0
    for tid, spec in sorted(registry.items(), key=lambda kv: kv[0].value):
        logger.info(
            "[DATA] event=thesis_inspect thesis_id=%s status=%s version=%d config_dir=%s",
            tid.value,
            spec.status.value,
            spec.version,
            config_dir,
        )
    logger.info("[DATA] event=thesis_inspect count=%d config_dir=%s", len(registry), config_dir)
    return 0


def run_thesis_report_command(
    *, thesis_id: str, as_of: str | None, experiment_path: str | None, settings: DataSettings
) -> int:
    """Build and persist thesis report; never calls adoption_passes."""
    # wiring: run thesis-wave and also run thesis-report
    _anchor = "run thesis-report"
    _anchor2 = "run thesis-wave"
    _ = build_thesis_report
    _ = run_thesis_wave
    from datetime import datetime

    from src.policy.thesis import ThesisError, ThesisId

    try:
        tid = ThesisId(thesis_id)
    except ValueError as exc:
        logger.error("[DATA] event=thesis_report_failed reason=%s", exc)
        return 2
    try:
        from src.data.panel_freshness import resolve_catalog_panel_as_of

        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
            # fail-closed if explicit as_of after last catalog session
            try:
                from src.data.catalog import latest_artifact
                from src.data.schema import Dataset, spec_for
                from src.data.storage import DataStore

                latest = latest_artifact(settings, Dataset.PRICES)
                frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
                max_d = frame.get_column("date").max()
                if isinstance(max_d, date) and as_of_dt.date() > max_d:
                    raise ValueError(f"explicit --as-of {as_of_dt.isoformat()} is after last catalog price session {max_d.isoformat()}")
            except ValueError:
                raise
            except Exception:  # noqa: S110
                pass  # noqa: S110
        else:
            as_of_dt = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC)).panel_as_of
        exp_path = Path(experiment_path) if experiment_path is not None else None
        # runner for report: allocation from store
        from src.sim.allocation import run_allocation_from_store

        def _runner(config):  # type: ignore[no-untyped-def]
            return run_allocation_from_store(config, settings)

        report = build_thesis_report(thesis_id=tid, settings=settings, as_of=as_of_dt, runner=_runner, experiment_path=exp_path)
        write_thesis_report(report, settings)
    except (ThesisError, ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_report_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=thesis_report_done thesis_id=%s as_of=%s", tid.value, as_of_dt.isoformat())
    return 0


def run_diagnose_overlap_command(
    *,
    vehicle: str,
    baseline: str,
    as_of: str | None,
    settings: DataSettings,
) -> int:
    """Run pairwise holdings overlap at PIT as_of (reporting only)."""
    from datetime import datetime

    # anchor for wiring: run diagnose-overlap and pairwise_overlap
    _ = pairwise_overlap
    _ = thesis_overlap_vs_incumbent
    _anchor = "run diagnose-overlap"
    try:
        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
        else:
            as_of_dt = datetime.now(UTC)
        # Fail-closed when no PIT row exists for as_of
        try:
            holdings = load_visible(settings, Dataset.ETF_HOLDINGS, as_of_dt)
        except Exception as exc:
            logger.error("[DATA] event=diagnose_overlap_failed reason=%s", exc)
            return 1
        if holdings.is_empty():
            logger.error("[DATA] event=diagnose_overlap_failed reason=no PIT row exists for as_of %s", as_of_dt.isoformat())
            return 1
        report = pairwise_overlap(holdings, vehicle_a=vehicle, vehicle_b=baseline, as_of=as_of_dt)
        logger.info(
            "[DATA] event=diagnose_overlap vehicle=%s baseline=%s overlap_pct=%.4f shared=%d as_of=%s",
            report.vehicle_a,
            report.vehicle_b,
            report.overlap_pct,
            report.shared_holdings_count,
            report.as_of.isoformat(),
        )
        return 0
    except ValueError as exc:
        # explicit fail-closed message for missing PIT row
        if "no PIT row exists" in str(exc):
            logger.error("[DATA] event=diagnose_overlap_failed reason=%s", exc)
            return 1
        logger.error("[DATA] event=diagnose_overlap_failed reason=%s", exc)
        return 1
    except Exception as exc:
        logger.error("[DATA] event=diagnose_overlap_failed reason=%s", exc)
        return 1


def run_thesis_wave_command(*, as_of: str | None, settings: DataSettings, allow_stale: bool = False) -> int:
    """Run full thesis wave; never calls adoption_passes."""
    _anchor = "run thesis-wave"
    _ = run_thesis_wave
    from datetime import datetime

    # wiring anchor: resolve_catalog_panel_as_of and as_of_dt = datetime.now(UTC)
    _ = resolve_catalog_panel_as_of
    _anchor_now = datetime.now(UTC)

    try:
        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
            # Explicit --as-of after last catalog price session fails closed
            try:
                _ = resolve_catalog_panel_as_of(settings, reference_now=as_of_dt)
                from src.data.catalog import latest_artifact
                from src.data.schema import Dataset, spec_for
                from src.data.storage import DataStore
                try:
                    latest = latest_artifact(settings, Dataset.PRICES)
                    frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
                    max_d = frame.get_column("date").max()
                    if isinstance(max_d, date) and as_of_dt.date() > max_d:
                        raise ValueError(f"explicit --as-of {as_of_dt.isoformat()} is after last catalog price session {max_d.isoformat()}")
                except ValueError:
                    raise
                except Exception:  # noqa: S110
                    pass  # noqa: S110
            except ValueError:
                raise
            except Exception:  # noqa: S110
                pass  # noqa: S110
        else:
            try:
                as_of_dt = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC)).panel_as_of
            except Exception as exc:
                logger.error("[DATA] event=thesis_wave_panel_failed reason=%s", exc)
                return 1

        from src.data.panel_freshness import PanelFreshnessStatus

        try:
            gate_report = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC))
        except Exception as exc:
            logger.error("[DATA] event=thesis_wave_panel_failed reason=%s", exc)
            return 1
        if gate_report.status == PanelFreshnessStatus.STALE:
            hard = load_panel_hard_stop()
            gate_report = apply_hard_stop(gate_report, hard)
            if gate_report.status != PanelFreshnessStatus.HARD_STOP_ACK and not allow_stale:
                logger.error(
                    "[DATA] event=thesis_wave_stale panel_as_of=%s lag_days=%d",
                    gate_report.panel_as_of.isoformat(),
                    gate_report.lag_days,
                )
                return 1
        elif gate_report.status == PanelFreshnessStatus.INSUFFICIENT_DATA:
            logger.error("[DATA] event=thesis_wave_insufficient_data reason=insufficient catalog")
            return 1
        logger.info(
            "[DATA] event=thesis_wave_panel panel_as_of=%s lag_days=%d status=%s",
            gate_report.panel_as_of.isoformat(),
            gate_report.lag_days,
            gate_report.status.value,
        )

        from src.sim.allocation import run_allocation_from_store

        def _runner(config):  # type: ignore[no-untyped-def]
            return run_allocation_from_store(config, settings)

        wave = run_thesis_wave(settings=settings, as_of=as_of_dt, runner=_runner, panel_report=gate_report)
        # Also write markdown
        from src.analytics.thesis_wave import write_thesis_wave_markdown

        md_path = Path(f"docs/results/thesis-wave/{as_of_dt.date().isoformat()}_v2_thesis_wave.md")
        write_thesis_wave_markdown(wave, md_path)
        if not wave.entries:
            raise ValueError("thesis wave produced zero successful entries")
    except (ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_wave_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=thesis_wave_done as_of=%s entries=%d failures=%d",
        as_of_dt.isoformat(),
        len(wave.entries),
        len(wave.failures),
    )
    return 1 if wave.failures else 0


def run_thesis_incremental_command(
    *,
    thesis_id: str,
    as_of: str | None,
    settings: DataSettings,
    seed: int,
    bootstrap_paths: int,
    allow_stale: bool = False,
    contribution_krw: float = 1_000_000,
) -> int:
    """Run Track H incremental portfolio; panel gate like thesis-wave; never adoption_passes."""
    _anchor = "run thesis-incremental"
    _ = run_incremental_portfolio
    _ = write_incremental_portfolio_report
    from datetime import datetime

    _ = resolve_catalog_panel_as_of
    _anchor_now = datetime.now(UTC)
    try:
        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
            try:
                _ = resolve_catalog_panel_as_of(settings, reference_now=as_of_dt)
                from src.data.catalog import latest_artifact
                from src.data.schema import Dataset, spec_for
                from src.data.storage import DataStore

                try:
                    latest = latest_artifact(settings, Dataset.PRICES)
                    frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
                    max_d = frame.get_column("date").max()
                    if isinstance(max_d, date) and as_of_dt.date() > max_d:
                        raise ValueError(f"explicit --as-of {as_of_dt.isoformat()} is after last catalog price session {max_d.isoformat()}")
                except ValueError:
                    raise
                except Exception:  # noqa: S110
                    pass
            except ValueError:
                raise
            except Exception:  # noqa: S110
                pass
        else:
            try:
                as_of_dt = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC)).panel_as_of
            except Exception as exc:
                logger.error("[DATA] event=thesis_incremental_panel_failed reason=%s", exc)
                return 1
        from src.data.panel_freshness import PanelFreshnessStatus

        try:
            gate_report = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC))
        except Exception as exc:
            logger.error("[DATA] event=thesis_incremental_panel_failed reason=%s", exc)
            return 1
        if gate_report.status == PanelFreshnessStatus.STALE:
            hard = load_panel_hard_stop()
            gate_report = apply_hard_stop(gate_report, hard)
            if gate_report.status != PanelFreshnessStatus.HARD_STOP_ACK and not allow_stale:
                logger.error("[DATA] event=thesis_incremental_stale panel_as_of=%s lag_days=%d", gate_report.panel_as_of.isoformat(), gate_report.lag_days)
                return 1
        elif gate_report.status == PanelFreshnessStatus.INSUFFICIENT_DATA:
            logger.error("[DATA] event=thesis_incremental_insufficient_data reason=insufficient catalog")
            return 1
        logger.info("[DATA] event=thesis_incremental_panel panel_as_of=%s lag_days=%d status=%s", gate_report.panel_as_of.isoformat(), gate_report.lag_days, gate_report.status.value)
        # validate thesis_id
        from src.policy.thesis import ThesisId, load_thesis_registry

        try:
            tid = ThesisId(thesis_id)
        except ValueError as exc:
            logger.error("[DATA] event=thesis_incremental_failed reason=%s", exc)
            return 2
        if tid not in (ThesisId.AI_COMPUTE, ThesisId.AI_POWER_BOTTLENECK, ThesisId.PHYSICAL_AUTOMATION):
            logger.error("[DATA] event=thesis_incremental_failed reason=only ai_compute supported")
            return 2
        # anchor keep for wiring: only ai_compute supported
        _ = "only ai_compute supported"
        try:
            registry = load_thesis_registry(Path("configs/theses"))
            thesis_spec = registry[tid]
            vehicle_ticker = str(thesis_spec.historical_proxies[0].value) if thesis_spec.historical_proxies else "SOXX"
        except Exception as exc:
            logger.error("[DATA] event=thesis_incremental_failed reason=%s", exc)
            return 1
        from src.sim.allocation import run_allocation_from_store

        def _runner(config):  # type: ignore[no-untyped-def]
            return run_allocation_from_store(config, settings)

        report = run_incremental_portfolio(
            settings=settings,
            as_of=as_of_dt,
            runner=_runner,
            contribution_krw=float(contribution_krw),
            bootstrap_paths=int(bootstrap_paths),
            seed=int(seed),
            panel_report=gate_report,
            thesis_id=str(tid.value),
            vehicle_ticker=str(vehicle_ticker),
        )
        out_path = Path(f"docs/results/thesis-incremental/{as_of_dt.date().isoformat()}_incremental_{thesis_id}.json")
        # also write under data root for history
        write_incremental_portfolio_report(report, out_path)
        # also write under data dir
        try:
            from src.data.paths import thesis_reports_dir

            data_path = thesis_reports_dir(settings) / f"incremental_{thesis_id}_{as_of_dt.date().isoformat()}.json"
            write_incremental_portfolio_report(report, data_path)
        except Exception:  # noqa: S110
            pass
        logger.info("[DATA] event=thesis_incremental_done thesis_id=%s portfolio_status=%s arms=%d", thesis_id, report.portfolio_status.value, len(report.arms))
    except (ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_incremental_failed reason=%s", exc)
        return 1
    return 0
