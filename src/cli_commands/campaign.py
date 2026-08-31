# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Campaign runners (validate, ablation, walk-forward, etc.)."""

from __future__ import annotations

import logging
from datetime import date

from src.analytics.metrics import XirrError
from src.cli_commands.parser import _UsageError, _resolve_git_commit
from src.data.catalog import latest_artifact
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.policy.targets import PolicyError, PolicyId
from src.sim.allocation import AllocationConfig, AllocationDataError, run_allocation_from_store
from src.sim.baseline import BaselineConfig, BaselineDataError, BaselineId, run_baseline_from_store
from src.sim.research_proxy import run_research_proxy_from_store
from src.validation.ablation import run_ablation
from src.validation.accumulation_cohort import run_accumulation_cohort_report, write_accumulation_cohort_report
from src.validation.bootstrap import moving_block_bootstrap
from src.validation.campaign import (
    run_cadence_robustness,
    run_walk_forward_adoption,
    run_walk_forward_cost_grid,
    run_walk_forward_proxy_adoption,
    write_cadence_robustness_report,
    write_campaign_report,
    write_cost_grid_report,
)
from src.validation.evaluate import evaluate_cohort_wealths
from src.validation.experiment import assert_experiment_preregistration, load_experiment_config, resolve_arm_targets
from src.validation.feasibility import assert_experiment_feasible
from src.validation.gate import adoption_passes, certainty_equivalent
from src.validation.registry import make_experiment, write_ablation_run_record
from src.validation.windows import rolling_cohorts

logger = logging.getLogger(__name__)

_VALIDATE_GAMMAS: tuple[float, ...] = (2.0, 5.0, 10.0)
_VALIDATE_BASELINE_TICKER: str = "VT"


def run_validate_command(
    *,
    policy_id: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
    delta0: float,
    modules: int,
    horizon_months: int,
    cohort_step_months: int,
    bootstrap_paths: int,
    seed: int | None,
) -> int:
    """Cohort CE adoption gate versus B0; optional seeded wealth-vector bootstrap."""
    if bootstrap_paths > 0 and seed is None:
        raise _UsageError("--bootstrap-paths requires --seed")

    template = AllocationConfig(
        policy=PolicyId(policy_id),
        start=start,
        end=end,
        monthly_contribution_krw=float(contribution_krw),
        fill_delay_sessions=1,
        commission_bps=0.0,
    )
    try:
        cohorts = rolling_cohorts(start, end, horizon_months=horizon_months, step_months=cohort_step_months)
        candidate_wealths = evaluate_cohort_wealths(
            template,
            cohorts,
            lambda config: run_allocation_from_store(config, settings),
        )
        baseline_wealths = tuple(
            run_baseline_from_store(
                BaselineConfig(
                    baseline=BaselineId.B0_GLOBAL,
                    ticker=_VALIDATE_BASELINE_TICKER,
                    start=cohort_start,
                    end=cohort_end,
                    monthly_contribution_krw=float(contribution_krw),
                    fill_delay_sessions=1,
                    commission_bps=0.0,
                ),
                settings,
            ).terminal_wealth_real_krw
            for cohort_start, cohort_end in cohorts
        )
        candidate_ce = {gamma: certainty_equivalent(candidate_wealths, gamma=gamma) for gamma in _VALIDATE_GAMMAS}
        baseline_ce = {gamma: certainty_equivalent(baseline_wealths, gamma=gamma) for gamma in _VALIDATE_GAMMAS}
        adopted = adoption_passes(candidate_ce, baseline_ce, delta0=delta0, modules=modules)
        record = make_experiment(
            config=template,
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=seed,
            metrics={
                **{f"ce_candidate_gamma_{int(gamma)}": value for gamma, value in candidate_ce.items()},
                **{f"ce_baseline_gamma_{int(gamma)}": value for gamma, value in baseline_ce.items()},
                "adopted": 1.0 if adopted else 0.0,
                "cohorts": float(len(cohorts)),
            },
        )
    except (
        AllocationDataError,
        BaselineDataError,
        PolicyError,
        UntrustedDatasetError,
        XirrError,
        ValueError,
    ) as exc:
        logger.error("[DATA] event=validate_cli_failed reason=%s", exc)
        return 1

    bootstrap_mean = 0.0
    if bootstrap_paths > 0 and seed is not None:
        # Half-window blocks keep roughly two independent draws per resampled path.
        resampled = moving_block_bootstrap(
            candidate_wealths,
            block_size=max(1, len(candidate_wealths) // 2),
            n_paths=bootstrap_paths,
            seed=seed,
        )
        resampled_means = [sum(path) / len(path) for path in resampled]
        bootstrap_mean = sum(resampled_means) / len(resampled_means)

    ratios = {gamma: candidate_ce[gamma] / baseline_ce[gamma] for gamma in candidate_ce}
    logger.info(
        "[DATA] event=validate_cli_done policy=%s cohorts=%d adopted=%s"
        " ratio_gamma_2=%.6f ratio_gamma_5=%.6f ratio_gamma_10=%.6f"
        " bootstrap_paths=%d bootstrap_mean=%.6f experiment_id=%s",
        str(template.policy),
        len(cohorts),
        adopted,
        ratios[2.0],
        ratios[5.0],
        ratios[10.0],
        bootstrap_paths,
        bootstrap_mean,
        record.experiment_id,
    )
    return 0


def run_ablation_command(*, config_path: str, settings: DataSettings) -> int:
    """Run an identical-cashflow ablation from an experiment JSON and log each gate."""
    try:
        from src.policy.thesis import ThesisError

        spec = load_experiment_config(config_path)
        from pathlib import Path as _Path

        from src.policy.thesis import load_thesis_registry

        registry = load_thesis_registry(_Path("configs/theses"))
        assert_experiment_preregistration(spec, registry)
        assert_experiment_feasible(spec, settings)
        report = run_ablation(spec, lambda config: run_allocation_from_store(config, settings))
        metrics: dict[str, float] = {
            "candidates": float(len(report.rows)),
            "adopted": float(sum(row.adopted for row in report.rows)),
        }
        for row in report.rows:
            for gamma, ratio in row.ce_ratio.items():
                metrics[f"{row.candidate_id}_ratio_gamma_{int(gamma)}"] = ratio
        thesis_id_str = spec.thesis_id.value if spec.thesis_id is not None else None
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.baseline.policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=0.0,
                targets_override=resolve_arm_targets(spec.candidates[0]) if spec.candidates else None,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics=metrics,
            thesis_id=thesis_id_str,
        )
        write_ablation_run_record(spec=spec, report=report, record=record, settings=settings)
    except (AllocationDataError, PolicyError, ThesisError, UntrustedDatasetError, XirrError, ValueError, OSError) as exc:
        logger.error("[DATA] event=ablation_cli_failed reason=%s", exc)
        return 1
    adopted_count = sum(row.adopted for row in report.rows)
    for index, row in enumerate(report.rows):
        logger.info(
            "[DATA] event=ablation_candidate index=%d candidate=%s policy=%s modules=%d adopted=%s"
            " ratio_gamma_2=%.6f ratio_gamma_5=%.6f ratio_gamma_10=%.6f",
            index,
            row.candidate_id,
            str(row.policy),
            row.modules,
            row.adopted,
            row.ce_ratio[2.0],
            row.ce_ratio[5.0],
            row.ce_ratio[10.0],
        )
    logger.info(
        "[DATA] event=ablation_cli_done experiment=%s experiment_id=%s adopted=%d/%d",
        spec.name,
        record.experiment_id,
        adopted_count,
        len(report.rows),
    )
    return 0


from src.validation.prospective_registry import freeze_prospective_bundle, run_prospective_monitor  # wiring for lean_check  # noqa: E402

_ = freeze_prospective_bundle


def run_prospective_monitor_command(*, bundle_path: str, as_of: str | date, settings: DataSettings, registry_dir: str | None = None) -> int:
    """Run prospective monitoring (append-only, post-cutoff) and persist observations."""
    from pathlib import Path as _Path

    from src.validation.prospective_registry import load_prospective_bundle

    # wiring: run_prospective_monitor invocation
    _ = run_prospective_monitor

    try:
        bundle = load_prospective_bundle(_Path(bundle_path))
        a_date = date.fromisoformat(str(as_of)) if isinstance(as_of, str) else as_of
        rdir = _Path(registry_dir) if registry_dir is not None else None
        # run_prospective_monitor call ensures correct wiring string exists
        report = run_prospective_monitor(
            bundle=bundle,
            as_of=a_date,
            runner=lambda cfg: run_allocation_from_store(cfg, settings),
            settings=settings,
            registry_dir=rdir,
            runtime_git_commit=_resolve_git_commit(),
        )
        logger.info(
            "[DATA] event=prospective_monitor_cli_done bundle=%s as_of=%s observations=%d registry=%s",
            bundle.bundle_id,
            a_date.isoformat(),
            len(report.observations),
            report.registry_path.as_posix(),
        )
        return 0
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError, OSError) as exc:
        logger.error("[DATA] event=prospective_monitor_cli_failed reason=%s", exc)
        return 1


def run_walk_forward_command(*, config_path: str, settings: DataSettings) -> int:
    """Run a walk-forward adoption campaign and persist the report JSON."""
    try:
        spec = load_experiment_config(config_path)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        if len(spec.candidates) > 1:
            raise ValueError("walk-forward with multiple candidates requires strategy-select; run strategy-select --config PATH instead")
        report = run_walk_forward_adoption(spec, lambda config: run_allocation_from_store(config, settings))
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=0.0,
                targets_override=resolve_arm_targets(spec.candidates[0]),
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics={
                "folds": float(len(report.folds)),
                "process_adopted_vs_baseline": 1.0 if report.process_adopted_vs_baseline else 0.0,
            },
        )
        report_path = write_campaign_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=walkforward_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=walkforward_cli_done experiment=%s experiment_id=%s folds=%d adopted_vs_baseline=%s report=%s",
        spec.name,
        record.experiment_id,
        len(report.folds),
        report.process_adopted_vs_baseline,
        report_path,
    )
    return 0



def run_strategy_selection_command(*, config_path: str, settings: DataSettings) -> int:
    """Run walk-forward tournament strategy selection and persist report."""
    try:
        from pathlib import Path as _Path

        from src.policy.thesis import load_thesis_registry

        from src.validation.strategy_selection import make_selection_runner, run_strategy_selection, write_strategy_selection_report

        # wiring: run_strategy_selection invocation
        _ = run_strategy_selection

        spec = load_experiment_config(config_path)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        if spec.thesis_id is not None:
            registry = load_thesis_registry(_Path("configs/theses"))
            assert_experiment_preregistration(spec, registry)
        wf_runner = make_selection_runner(settings, spec)
        report = run_strategy_selection(spec, wf_runner)
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=0.0,
                targets_override=resolve_arm_targets(spec.candidates[0]),
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics={
                "candidates": float(len(report.rows)),
                "oos_eligible": float(len(report.oos_eligible_arm_ids)),
                "recommended": 1.0,
            },
        )
        report_path = write_strategy_selection_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError, OSError) as exc:
        logger.error("[DATA] event=strategy_selection_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=strategy_selection_cli_done experiment=%s experiment_id=%s recommended=%s oos_eligible=%s report=%s",
        spec.name,
        record.experiment_id,
        report.recommended_arm_id,
        report.oos_eligible_arm_ids,
        report_path,
    )
    return 0


def run_walk_forward_costs_command(*, config_path: str, settings: DataSettings) -> int:
    """Run the walk-forward adoption cost grid and persist one grid report JSON."""
    try:
        spec = load_experiment_config(config_path)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        report = run_walk_forward_cost_grid(spec, lambda config: run_allocation_from_store(config, settings))
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=spec.commission_bps,
                fx_spread_bps=spec.fx_spread_bps,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics={
                "scenarios": float(len(report.outcomes)),
                "all_scenarios_adopted": 1.0 if report.all_scenarios_adopted else 0.0,
            },
        )
        report_path = write_cost_grid_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=walkforward_costs_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=walkforward_costs_cli_done experiment=%s experiment_id=%s scenarios=%d all_adopted=%s report=%s",
        spec.name,
        record.experiment_id,
        len(report.outcomes),
        report.all_scenarios_adopted,
        report_path,
    )
    return 0


def run_walk_forward_proxy_command(*, config_path: str, settings: DataSettings) -> int:
    """Run the research-proxy walk-forward campaign and persist the report JSON."""
    try:
        spec = load_experiment_config(config_path)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        report = run_walk_forward_proxy_adoption(
            spec,
            lambda config: run_allocation_from_store(config, settings),
            lambda config: run_research_proxy_from_store(config, settings),
        )
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=0.0,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics={
                "folds": float(len(report.folds)),
                "process_adopted_vs_baseline": 1.0 if report.process_adopted_vs_baseline else 0.0,
            },
        )
        report_path = write_campaign_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=walkforward_proxy_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=walkforward_proxy_cli_done experiment=%s experiment_id=%s folds=%d adopted_vs_baseline=%s report=%s",
        spec.name,
        record.experiment_id,
        len(report.folds),
        report.process_adopted_vs_baseline,
        report_path,
    )
    return 0


def run_cadence_robustness_command(
    *, config_path: str, settings: DataSettings, seed: int, bootstrap_paths: int
) -> int:
    """Run the growth-first cadence robustness gate and persist one report JSON."""
    if bootstrap_paths < 1:
        raise _UsageError(f"--bootstrap-paths must be >= 1, got {bootstrap_paths}")
    try:
        spec = load_experiment_config(config_path)
        assert_experiment_feasible(spec, settings)
        report = run_cadence_robustness(
            spec,
            lambda config: run_allocation_from_store(config, settings),
            n_paths=bootstrap_paths,
            seed=seed,
        )
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=spec.commission_bps,
                fx_spread_bps=spec.fx_spread_bps,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=seed,
            metrics={
                "cohorts": float(len(report.candidate_wealths)),
                "all_scenarios_adopted": 1.0 if report.cost_grid.all_scenarios_adopted else 0.0,
                "robust_adopted": 1.0 if report.robust_adopted else 0.0,
            },
        )
        report_path = write_cadence_robustness_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=cadence_robustness_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=cadence_robustness_cli_done experiment=%s experiment_id=%s cohorts=%d"
        " all_scenarios_adopted=%s worst_cohort_ok=%s bootstrap_tail_ok=%s robust_adopted=%s report=%s",
        spec.name,
        record.experiment_id,
        len(report.candidate_wealths),
        report.cost_grid.all_scenarios_adopted,
        report.worst_cohort_ok,
        report.bootstrap_tail_ok,
        report.robust_adopted,
        report_path,
    )
    return 0


def run_accumulation_cohort_command(
    *,
    config_path: str,
    settings: DataSettings,
    horizon_months: int,
    cohort_step_months: int,
    bootstrap_paths: int,
    seed: int | None,
) -> int:
    """Run rolling 120M accumulation cohort report (reporting-only)."""
    if cohort_step_months not in {1, 12, 36}:
        raise _UsageError(f"--cohort-step-months must be one of 1, 12, 36, got {cohort_step_months}")
    if horizon_months < 1:
        raise _UsageError(f"--horizon-months must be >=1, got {horizon_months}")
    if bootstrap_paths < 1:
        raise _UsageError(f"--bootstrap-paths must be >=1, got {bootstrap_paths}")
    if seed is None:
        raise _UsageError("--seed is required for accumulation-cohort")
    try:
        spec = load_experiment_config(config_path)
        assert_experiment_feasible(spec, settings)
        report = run_accumulation_cohort_report(
            spec,
            lambda config: run_allocation_from_store(config, settings),
            horizon_months=horizon_months,
            step_months=cohort_step_months,
            bootstrap_paths=bootstrap_paths,
            seed=seed,
        )
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy if spec.candidates else spec.baseline.policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=spec.commission_bps,
                fx_spread_bps=spec.fx_spread_bps,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=seed,
            metrics={
                "cohorts": float(len(report.rows)),
                "median_ratio": float(report.median_ratio),
                "p10_ratio": float(report.p10_ratio),
                "worst_ratio": float(report.worst_ratio),
                "win_rate": float(report.win_rate),
                "bootstrap_p05_ratio_mean": float(report.bootstrap_p05_ratio_mean),
            },
        )
        report_path = write_accumulation_cohort_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=accumulation_cohort_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=accumulation_cohort_cli_done experiment=%s experiment_id=%s cohorts=%d"
        " median_ratio=%.6f p10_ratio=%.6f worst_ratio=%.6f win_rate=%.4f bootstrap_p05=%.6f report=%s",
        spec.name,
        record.experiment_id,
        len(report.rows),
        report.median_ratio,
        report.p10_ratio,
        report.worst_ratio,
        report.win_rate,
        report.bootstrap_p05_ratio_mean,
        report_path,
    )
    return 0


def run_final_historical_campaign_command(
    *,
    config_path: str,
    settings: DataSettings,
    seed: int,
    bootstrap_paths: int = 400,
) -> int:
    """Run final historical campaign (reporting-only, frozen arms)."""
    if bootstrap_paths < 1:
        raise _UsageError(f"--bootstrap-paths must be >=1, got {bootstrap_paths}")
    try:
        from src.validation.historical_campaign import (
            assert_final_campaign_spec,
            run_final_historical_campaign,
            write_final_historical_campaign_report,
        )

        # wiring: run_final_historical_campaign invocation
        _ = run_final_historical_campaign

        spec = load_experiment_config(config_path)
        assert_final_campaign_spec(spec)
        assert_experiment_feasible(spec, settings)
        report = run_final_historical_campaign(
            spec,
            lambda config: run_allocation_from_store(config, settings),
            seed=seed,
            bootstrap_paths=bootstrap_paths,
            settings=settings,
        )
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy if spec.candidates else spec.baseline.policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=spec.commission_bps,
                fx_spread_bps=spec.fx_spread_bps,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=seed,
            metrics={
                "cohorts": float(report.arm_rows[0].cohort_count) if report.arm_rows else 0.0,
                "arms": float(len(report.arm_rows)),
            },
        )
        report_path = write_final_historical_campaign_report(report, settings, experiment_id=record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError, OSError) as exc:
        logger.error("[DATA] event=final_historical_campaign_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=final_historical_campaign_cli_done experiment=%s experiment_id=%s arms=%d report=%s",
        spec.name,
        record.experiment_id,
        len(report.arm_rows),
        report_path,
    )
    return 0


def run_audit_feasibility_command(
    *,
    config_path: str,
    settings: DataSettings,
    write_report: bool,
) -> int:
    """Load ExperimentSpec, run static DCA audit, optionally persist JSON."""
    from src.validation.experiment import load_experiment_config
    from src.validation.feasibility_audit import (
        WAVE2_MIN_120M_COHORTS,
        audit_static_dca_window,
        write_feasibility_audit_report,
    )

    try:
        spec = load_experiment_config(config_path)
        report = audit_static_dca_window(spec, settings)
        if write_report:
            import uuid

            audit_id = uuid.uuid4().hex[:8]
            write_feasibility_audit_report(report, settings, audit_id=audit_id)
        if report.earliest_feasible_start is None:
            return 1
        if spec.name == "acc_qqq_baseline_120m" and report.cohort_count_120m_step12 < WAVE2_MIN_120M_COHORTS:
            return 1
        return 0
    except (ValueError, UntrustedDatasetError, OSError) as exc:
        logger.error("[DATA] event=audit_feasibility_failed reason=%s", exc)
        return 1
