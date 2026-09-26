# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Campaign runners (validate, ablation, walk-forward, etc.)."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from src.analytics.metrics import XirrError
from src.cli_commands.parser import _UsageError, _resolve_git_commit
from src.data.catalog import latest_artifact
from src.data.paths import THESES_DIR, resolve_repository_paths
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.policy.targets import PolicyError, PolicyId
from src.policy.thesis import ThesisError, load_thesis_registry
from src.sim.allocation import AllocationConfig, AllocationDataError, run_allocation_from_store
from src.sim.baseline import BaselineConfig, BaselineDataError, BaselineId, run_baseline_from_store
from src.sim.research_proxy import run_research_proxy_from_store
from src.validation.ablation import run_ablation
from src.validation.accumulation_cohort import run_accumulation_cohort_report, write_accumulation_cohort_report
from src.validation.bootstrap import moving_block_bootstrap
from src.validation.campaign import (
    run_cadence_robustness, run_walk_forward_adoption, run_walk_forward_cost_grid,
    run_walk_forward_proxy_adoption, write_cadence_robustness_report, write_campaign_report, write_cost_grid_report,
)
from src.validation.evaluate import evaluate_cohort_wealths
from src.validation.experiment import assert_experiment_preregistration, load_experiment_config, resolve_arm_targets
from src.validation.feasibility import assert_experiment_feasible
from src.validation.gate import adoption_passes, certainty_equivalent
from src.validation.prospective_registry import freeze_prospective_bundle, run_prospective_monitor
from src.validation.registry import make_experiment, write_ablation_run_record
from src.validation.windows import rolling_cohorts

_ = freeze_prospective_bundle
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.validation.pension_decision import PensionDecisionReport
_VALIDATE_GAMMAS: tuple[float, ...] = (2.0, 5.0, 10.0)
_VALIDATE_BASELINE_TICKER: str = "VT"
_ERRORS = (AllocationDataError, BaselineDataError, PolicyError, ThesisError, UntrustedDatasetError, XirrError, ValueError, OSError)


def run_validate_command(*, policy_id: str, start: date, end: date, contribution_krw: float, settings: DataSettings, delta0: float, modules: int, horizon_months: int, cohort_step_months: int, bootstrap_paths: int, seed: int | None) -> int:
    """Cohort CE adoption gate versus B0; optional seeded wealth-vector bootstrap."""
    if bootstrap_paths > 0 and seed is None:
        raise _UsageError("--bootstrap-paths requires --seed")
    template = AllocationConfig(policy=PolicyId(policy_id), start=start, end=end, monthly_contribution_krw=float(contribution_krw), fill_delay_sessions=1, commission_bps=0.0)
    try:
        cohorts = rolling_cohorts(start, end, horizon_months=horizon_months, step_months=cohort_step_months)
        cand_w = evaluate_cohort_wealths(template, cohorts, lambda cfg: run_allocation_from_store(cfg, settings))
        base_w = tuple(run_baseline_from_store(BaselineConfig(baseline=BaselineId.B0_GLOBAL, ticker=_VALIDATE_BASELINE_TICKER, start=cs, end=ce, monthly_contribution_krw=float(contribution_krw), fill_delay_sessions=1, commission_bps=0.0), settings).terminal_wealth_real_krw for cs, ce in cohorts)
        cand_ce = {gamma: certainty_equivalent(cand_w, gamma=gamma) for gamma in _VALIDATE_GAMMAS}
        base_ce = {gamma: certainty_equivalent(base_w, gamma=gamma) for gamma in _VALIDATE_GAMMAS}
        adopted = adoption_passes(cand_ce, base_ce, delta0=delta0, modules=modules)
        record = make_experiment(
            config=template, manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=seed,
            metrics={**{f"ce_candidate_gamma_{int(g)}": v for g, v in cand_ce.items()}, **{f"ce_baseline_gamma_{int(g)}": v for g, v in base_ce.items()}, "adopted": 1.0 if adopted else 0.0, "cohorts": float(len(cohorts))},
        )
    except _ERRORS as exc:
        logger.error("[DATA] event=validate_cli_failed reason=%s", exc)
        return 1
    bootstrap_mean = 0.0
    if bootstrap_paths > 0 and seed is not None:
        resampled = moving_block_bootstrap(cand_w, block_size=max(1, len(cand_w) // 2), n_paths=bootstrap_paths, seed=seed)
        resampled_means = [sum(p) / len(p) for p in resampled]
        bootstrap_mean = sum(resampled_means) / len(resampled_means)
    ratios = {gamma: cand_ce[gamma] / base_ce[gamma] for gamma in cand_ce}
    logger.info("[DATA] event=validate_cli_done policy=%s cohorts=%d adopted=%s ratio_gamma_2=%.6f ratio_gamma_5=%.6f ratio_gamma_10=%.6f bootstrap_paths=%d bootstrap_mean=%.6f experiment_id=%s", str(template.policy), len(cohorts), adopted, ratios[2.0], ratios[5.0], ratios[10.0], bootstrap_paths, bootstrap_mean, record.experiment_id)
    return 0


def run_ablation_command(*, config_path: str, settings: DataSettings) -> int:
    """Run an identical-cashflow ablation from an experiment JSON and log each gate."""
    try:
        paths = resolve_repository_paths(settings)
        spec = load_experiment_config(config_path, settings=settings)
        assert_experiment_preregistration(spec, load_thesis_registry(paths.root / THESES_DIR.as_posix()))
        assert_experiment_feasible(spec, settings)
        report = run_ablation(spec, lambda cfg: run_allocation_from_store(cfg, settings))
        metrics: dict[str, float] = {"candidates": float(len(report.rows)), "adopted": float(sum(row.adopted for row in report.rows))}
        for row in report.rows:
            for gamma, ratio in row.ce_ratio.items():
                metrics[f"{row.candidate_id}_ratio_gamma_{int(gamma)}"] = ratio
        record = make_experiment(
            config=AllocationConfig(policy=spec.baseline.policy, start=spec.start, end=spec.end, monthly_contribution_krw=spec.contribution_krw, fill_delay_sessions=1, commission_bps=0.0, targets_override=resolve_arm_targets(spec.candidates[0]) if spec.candidates else None),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=None, metrics=metrics, thesis_id=spec.thesis_id.value if spec.thesis_id is not None else None,
        )
        write_ablation_run_record(spec=spec, report=report, record=record, settings=settings)
    except _ERRORS as exc:
        logger.error("[DATA] event=ablation_cli_failed reason=%s", exc)
        return 1
    for index, row in enumerate(report.rows):
        logger.info("[DATA] event=ablation_candidate index=%d candidate=%s policy=%s modules=%d adopted=%s ratio_gamma_2=%.6f ratio_gamma_5=%.6f ratio_gamma_10=%.6f", index, row.candidate_id, str(row.policy), row.modules, row.adopted, row.ce_ratio[2.0], row.ce_ratio[5.0], row.ce_ratio[10.0])
    logger.info("[DATA] event=ablation_cli_done experiment=%s experiment_id=%s adopted=%d/%d", spec.name, record.experiment_id, sum(row.adopted for row in report.rows), len(report.rows))
    return 0


def run_prospective_monitor_command(*, bundle_path: str, as_of: str | date, settings: DataSettings, registry_dir: str | None = None) -> int:
    """Run prospective monitoring (append-only, post-cutoff) and persist observations."""
    from src.validation.prospective_registry import load_prospective_bundle

    _ = run_prospective_monitor
    try:
        paths = resolve_repository_paths(settings)
        bundle_file = Path(bundle_path) if Path(bundle_path).is_absolute() else paths.root / bundle_path
        bundle = load_prospective_bundle(bundle_file)
        a_date = date.fromisoformat(str(as_of)) if isinstance(as_of, str) else as_of
        report = run_prospective_monitor(bundle=bundle, as_of=a_date, runner=lambda cfg: run_allocation_from_store(cfg, settings), settings=settings, registry_dir=Path(registry_dir) if registry_dir is not None else None, runtime_git_commit=_resolve_git_commit())
        logger.info("[DATA] event=prospective_monitor_cli_done bundle=%s as_of=%s observations=%d registry=%s", bundle.bundle_id, a_date.isoformat(), len(report.observations), report.registry_path.as_posix())
        return 0
    except _ERRORS as exc:
        logger.error("[DATA] event=prospective_monitor_cli_failed reason=%s", exc)
        return 1


def run_walk_forward_command(*, config_path: str, settings: DataSettings) -> int:
    """Run a walk-forward adoption campaign and persist the report JSON."""
    try:
        spec = load_experiment_config(config_path, settings=settings)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        if len(spec.candidates) > 1:
            raise ValueError("walk-forward with multiple candidates requires strategy-select; run strategy-select --config PATH instead")
        report = run_walk_forward_adoption(spec, lambda cfg: run_allocation_from_store(cfg, settings))
        record = make_experiment(
            config=AllocationConfig(policy=spec.candidates[0].policy, start=spec.start, end=spec.end, monthly_contribution_krw=spec.contribution_krw, fill_delay_sessions=1, commission_bps=0.0, targets_override=resolve_arm_targets(spec.candidates[0])),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=None, metrics={"folds": float(len(report.folds)), "process_adopted_vs_baseline": 1.0 if report.process_adopted_vs_baseline else 0.0},
        )
        report_path = write_campaign_report(report, settings, record.experiment_id)
    except _ERRORS as exc:
        logger.error("[DATA] event=walkforward_cli_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=walkforward_cli_done experiment=%s experiment_id=%s folds=%d adopted_vs_baseline=%s report=%s", spec.name, record.experiment_id, len(report.folds), report.process_adopted_vs_baseline, report_path)
    return 0


def run_strategy_selection_command(*, config_path: str, settings: DataSettings) -> int:
    """Run walk-forward tournament strategy selection and persist report."""
    try:
        from src.validation.strategy_selection import make_selection_runner, run_strategy_selection, write_strategy_selection_report

        _ = run_strategy_selection
        paths = resolve_repository_paths(settings)
        spec = load_experiment_config(config_path, settings=settings)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        if spec.thesis_id is not None:
            assert_experiment_preregistration(spec, load_thesis_registry(paths.root / THESES_DIR.as_posix()))
        report = run_strategy_selection(spec, make_selection_runner(settings, spec))
        record = make_experiment(
            config=AllocationConfig(policy=spec.candidates[0].policy, start=spec.start, end=spec.end, monthly_contribution_krw=spec.contribution_krw, fill_delay_sessions=1, commission_bps=0.0, targets_override=resolve_arm_targets(spec.candidates[0])),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=None, metrics={"candidates": float(len(report.rows)), "oos_eligible": float(len(report.oos_eligible_arm_ids)), "recommended": 1.0},
        )
        report_path = write_strategy_selection_report(report, settings, record.experiment_id)
    except _ERRORS as exc:
        logger.error("[DATA] event=strategy_selection_cli_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=strategy_selection_cli_done experiment=%s experiment_id=%s recommended=%s oos_eligible=%s report=%s", spec.name, record.experiment_id, report.recommended_arm_id, report.oos_eligible_arm_ids, report_path)
    return 0


def run_walk_forward_costs_command(*, config_path: str, settings: DataSettings) -> int:
    """Run the walk-forward adoption cost grid and persist one grid report JSON."""
    try:
        spec = load_experiment_config(config_path, settings=settings)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        report = run_walk_forward_cost_grid(spec, lambda cfg: run_allocation_from_store(cfg, settings))
        record = make_experiment(
            config=AllocationConfig(policy=spec.candidates[0].policy, start=spec.start, end=spec.end, monthly_contribution_krw=spec.contribution_krw, fill_delay_sessions=1, commission_bps=spec.commission_bps, fx_spread_bps=spec.fx_spread_bps),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=None, metrics={"scenarios": float(len(report.outcomes)), "all_scenarios_adopted": 1.0 if report.all_scenarios_adopted else 0.0},
        )
        report_path = write_cost_grid_report(report, settings, record.experiment_id)
    except _ERRORS as exc:
        logger.error("[DATA] event=walkforward_costs_cli_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=walkforward_costs_cli_done experiment=%s experiment_id=%s scenarios=%d all_adopted=%s report=%s", spec.name, record.experiment_id, len(report.outcomes), report.all_scenarios_adopted, report_path)
    return 0


def run_walk_forward_proxy_command(*, config_path: str, settings: DataSettings) -> int:
    """Run the research-proxy walk-forward campaign and persist the report JSON."""
    try:
        spec = load_experiment_config(config_path, settings=settings)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        report = run_walk_forward_proxy_adoption(spec, lambda cfg: run_allocation_from_store(cfg, settings), lambda cfg: run_research_proxy_from_store(cfg, settings))
        record = make_experiment(
            config=AllocationConfig(policy=spec.candidates[0].policy, start=spec.start, end=spec.end, monthly_contribution_krw=spec.contribution_krw, fill_delay_sessions=1, commission_bps=0.0),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=None, metrics={"folds": float(len(report.folds)), "process_adopted_vs_baseline": 1.0 if report.process_adopted_vs_baseline else 0.0},
        )
        report_path = write_campaign_report(report, settings, record.experiment_id)
    except _ERRORS as exc:
        logger.error("[DATA] event=walkforward_proxy_cli_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=walkforward_proxy_cli_done experiment=%s experiment_id=%s folds=%d adopted_vs_baseline=%s report=%s", spec.name, record.experiment_id, len(report.folds), report.process_adopted_vs_baseline, report_path)
    return 0


def run_cadence_robustness_command(*, config_path: str, settings: DataSettings, seed: int, bootstrap_paths: int) -> int:
    """Run the growth-first cadence robustness gate and persist one report JSON."""
    if bootstrap_paths < 1:
        raise _UsageError(f"--bootstrap-paths must be >= 1, got {bootstrap_paths}")
    try:
        spec = load_experiment_config(config_path, settings=settings)
        assert_experiment_feasible(spec, settings)
        report = run_cadence_robustness(spec, lambda cfg: run_allocation_from_store(cfg, settings), n_paths=bootstrap_paths, seed=seed)
        record = make_experiment(
            config=AllocationConfig(policy=spec.candidates[0].policy, start=spec.start, end=spec.end, monthly_contribution_krw=spec.contribution_krw, fill_delay_sessions=1, commission_bps=spec.commission_bps, fx_spread_bps=spec.fx_spread_bps),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=seed, metrics={"cohorts": float(len(report.candidate_wealths)), "all_scenarios_adopted": 1.0 if report.cost_grid.all_scenarios_adopted else 0.0, "robust_adopted": 1.0 if report.robust_adopted else 0.0},
        )
        report_path = write_cadence_robustness_report(report, settings, record.experiment_id)
    except _ERRORS as exc:
        logger.error("[DATA] event=cadence_robustness_cli_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=cadence_robustness_cli_done experiment=%s experiment_id=%s cohorts=%d all_scenarios_adopted=%s worst_cohort_ok=%s bootstrap_tail_ok=%s robust_adopted=%s report=%s", spec.name, record.experiment_id, len(report.candidate_wealths), report.cost_grid.all_scenarios_adopted, report.worst_cohort_ok, report.bootstrap_tail_ok, report.robust_adopted, report_path)
    return 0


def run_accumulation_cohort_command(*, config_path: str, settings: DataSettings, horizon_months: int, cohort_step_months: int, bootstrap_paths: int, seed: int | None) -> int:
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
        spec = load_experiment_config(config_path, settings=settings)
        assert_experiment_feasible(spec, settings)
        report = run_accumulation_cohort_report(spec, lambda cfg: run_allocation_from_store(cfg, settings), horizon_months=horizon_months, step_months=cohort_step_months, bootstrap_paths=bootstrap_paths, seed=seed)
        record = make_experiment(
            config=AllocationConfig(policy=spec.candidates[0].policy if spec.candidates else spec.baseline.policy, start=spec.start, end=spec.end, monthly_contribution_krw=spec.contribution_krw, fill_delay_sessions=1, commission_bps=spec.commission_bps, fx_spread_bps=spec.fx_spread_bps),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=seed,
            metrics={"cohorts": float(len(report.rows)), "median_ratio": float(report.median_ratio), "p10_ratio": float(report.p10_ratio), "worst_ratio": float(report.worst_ratio), "win_rate": float(report.win_rate), "bootstrap_p05_ratio_mean": float(report.bootstrap_p05_ratio_mean)},
        )
        report_path = write_accumulation_cohort_report(report, settings, record.experiment_id)
    except _ERRORS as exc:
        logger.error("[DATA] event=accumulation_cohort_cli_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=accumulation_cohort_cli_done experiment=%s experiment_id=%s cohorts=%d median_ratio=%.6f p10_ratio=%.6f worst_ratio=%.6f win_rate=%.4f bootstrap_p05=%.6f report=%s", spec.name, record.experiment_id, len(report.rows), report.median_ratio, report.p10_ratio, report.worst_ratio, report.win_rate, report.bootstrap_p05_ratio_mean, report_path)
    return 0


def run_final_historical_campaign_command(*, config_path: str, settings: DataSettings, seed: int, bootstrap_paths: int = 400) -> int:
    """Run final historical campaign (reporting-only, frozen arms)."""
    if bootstrap_paths < 1:
        raise _UsageError(f"--bootstrap-paths must be >=1, got {bootstrap_paths}")
    try:
        from src.validation.historical_campaign import assert_final_campaign_spec, run_final_historical_campaign, write_final_historical_campaign_report

        _ = run_final_historical_campaign
        spec = load_experiment_config(config_path, settings=settings)
        assert_final_campaign_spec(spec)
        assert_experiment_feasible(spec, settings)
        report = run_final_historical_campaign(spec, lambda cfg: run_allocation_from_store(cfg, settings), seed=seed, bootstrap_paths=bootstrap_paths, settings=settings)
        record = make_experiment(
            config=AllocationConfig(policy=spec.candidates[0].policy if spec.candidates else spec.baseline.policy, start=spec.start, end=spec.end, monthly_contribution_krw=spec.contribution_krw, fill_delay_sessions=1, commission_bps=spec.commission_bps, fx_spread_bps=spec.fx_spread_bps),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(), seed=seed, metrics={"cohorts": float(report.arm_rows[0].cohort_count) if report.arm_rows else 0.0, "arms": float(len(report.arm_rows))},
        )
        report_path = write_final_historical_campaign_report(report, settings, experiment_id=record.experiment_id)
    except _ERRORS as exc:
        logger.error("[DATA] event=final_historical_campaign_cli_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=final_historical_campaign_cli_done experiment=%s experiment_id=%s arms=%d report=%s", spec.name, record.experiment_id, len(report.arm_rows), report_path)
    return 0


def run_after_tax_campaign_command(*, config_path: str, settings: DataSettings, seed: int) -> int:
    """Run the after-tax cohort campaign (reporting-only; never changes the operational lock).

    Loads PRICES, FX_KRW_BASE, CPI, and RATES once at the close of the last session in
    the spec window, then runs every arm through ``run_after_tax`` on the in-memory
    frames. ``experiment_id`` is the first 16 hex chars of SHA-256 over the config file
    bytes, the resolved git commit, and the PRICES manifest hash.
    """
    from src.validation.after_tax_campaign import load_after_tax_campaign_spec, run_after_tax_campaign, write_after_tax_campaign_report

    try:
        import hashlib

        from src.data.calendar import load_calendar
        from src.data.catalog import load_visible
        from src.data.schedule import build_decision_schedule
        from src.sim.after_tax_engine import AfterTaxDataError, run_after_tax
        from src.sim.tax import load_tax_regime

        spec = load_after_tax_campaign_spec(config_path)
        regime = load_tax_regime(spec.tax_regime_path)
        schedule = build_decision_schedule(spec.start, spec.end, frequency="monthly", fill_delay_sessions=1)
        if not schedule:
            raise AfterTaxDataError(f"empty decision schedule over [{spec.start.isoformat()}, {spec.end.isoformat()}]")
        calendar = load_calendar()
        cutoff = calendar.close_ts(schedule[-1].execution_session)
        prices = load_visible(settings, Dataset.PRICES, cutoff)
        cpi = load_visible(settings, Dataset.CPI, cutoff)
        last_settle = schedule[-1].execution_session
        for _ in range(regime.settlement_sessions):
            last_settle = calendar.next_session(last_settle)
        fx = load_visible(settings, Dataset.FX_KRW_BASE, calendar.close_ts(last_settle))
        rates = load_visible(settings, Dataset.RATES, cutoff)
        manifest_hash = latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256
        config_bytes = Path(config_path).read_bytes()
        digest = hashlib.sha256(config_bytes + _resolve_git_commit().encode() + manifest_hash.encode()).hexdigest()[:16]

        report = run_after_tax_campaign(spec, lambda cfg: run_after_tax(cfg, prices, fx, cpi, rates), seed=seed)
        report_path = write_after_tax_campaign_report(report, settings, experiment_id=digest)
    except (*_ERRORS, AfterTaxDataError) as exc:
        logger.error("[DATA] event=after_tax_campaign_cli_failed reason_type=%s reason=%s", type(exc).__name__, exc)
        return 1
    logger.info("[DATA] event=after_tax_campaign_cli_done experiment=%s experiment_id=%s arms=%d report=%s", spec.name, digest, len(report.summaries), report_path)
    return 0


def run_audit_feasibility_command(*, config_path: str, settings: DataSettings, write_report: bool) -> int:
    """Load ExperimentSpec, run static DCA audit, optionally persist JSON."""
    from src.validation.feasibility_audit import WAVE2_MIN_120M_COHORTS, audit_static_dca_window, write_feasibility_audit_report

    try:
        spec = load_experiment_config(config_path, settings=settings)
        report = audit_static_dca_window(spec, settings)
        if write_report:
            import uuid

            write_feasibility_audit_report(report, settings, audit_id=uuid.uuid4().hex[:8])
        if report.earliest_feasible_start is None:
            return 1
        if spec.name == "acc_qqq_baseline_120m" and report.cohort_count_120m_step12 < WAVE2_MIN_120M_COHORTS:
            return 1
        return 0
    except (ValueError, UntrustedDatasetError, OSError) as exc:
        logger.error("[DATA] event=audit_feasibility_failed reason=%s", exc)
        return 1


def run_pension_campaign_command(*, config_path: str, settings: DataSettings, seed: int) -> int:
    """Run a reporting-only pension campaign from trusted, source-labeled inputs.

    Returns: Zero after a reproducible report is written, nonzero on invalid data or policy.
    Raises: No domain exception escapes the CLI boundary; failure is logged and returned.
    """
    from src.sim.pension_engine import PensionDataError, PensionMarketMode
    from src.validation.pension_campaign import (
        load_pension_campaign_spec, run_pension_campaign, write_pension_campaign_report,
    )

    try:
        import hashlib

        spec = load_pension_campaign_spec(config_path)
        regime_bytes = Path(spec.tax_regime_path).read_bytes()
        config_bytes = Path(config_path).read_bytes()
        general_regime_bytes = b""
        general_regime_sha: str | None = None
        if spec.household is not None:
            general_regime_bytes = Path(spec.household.general_tax_regime_path).read_bytes()
            general_regime_sha = hashlib.sha256(general_regime_bytes).hexdigest()
        report = run_pension_campaign(spec, settings, seed=seed)
        consumed = report.manifest_hashes
        if spec.market_mode is PensionMarketMode.KR_LIVE:
            manifest_hashes = [
                str(consumed[str(Dataset.KR_ETF_PRICES)]),
                str(consumed.get(str(Dataset.CPI)) or "NO_TRUSTED_CPI"),
            ]
        else:
            manifest_hashes = [
                str(consumed[str(Dataset.PRICES)]),
                str(consumed[str(Dataset.FX_KRW_BASE)]),
                str(consumed.get(str(Dataset.FX)) or "NO_FX_FALLBACK"),
                str(consumed.get(str(Dataset.CPI)) or "NO_TRUSTED_CPI"),
            ]
        git_commit = _resolve_git_commit()
        digest = hashlib.sha256(
            config_bytes + git_commit.encode() + "".join(manifest_hashes).encode()
            + regime_bytes + general_regime_bytes + str(seed).encode()
        ).hexdigest()[:16]

        provenance: dict[str, str] = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "tax_regime_sha256": hashlib.sha256(regime_bytes).hexdigest(),
            "git_commit": git_commit,
            "seed": str(seed),
        }
        if general_regime_sha is not None:
            provenance["general_tax_regime_sha256"] = general_regime_sha
        report_path = write_pension_campaign_report(
            report, settings, experiment_id=digest,
            provenance=provenance,
        )
    except (*_ERRORS, PensionDataError, UntrustedDatasetError) as exc:
        logger.error("[DATA] event=pension_campaign_cli_failed reason_type=%s reason=%s", type(exc).__name__, exc)
        return 1
    logger.info("[DATA] event=pension_campaign_cli_done experiment=%s experiment_id=%s arms=%d report=%s", spec.name, digest, len(report.summaries), report_path)
    return 0


def run_pension_selection_command(*, config_path: str, settings: DataSettings, seed: int) -> int:
    """Run the pre-registered standalone pension ETF selection and persist its evidence."""
    from src.analytics.pension_selection import PensionSelectionDataError
    from src.sim.pension_engine import PensionDataError
    from src.validation.pension_selection import (
        load_pension_selection_spec, run_pension_selection, write_pension_selection_report,
    )

    try:
        import hashlib

        spec = load_pension_selection_spec(config_path)
        config_bytes = Path(config_path).read_bytes()
        tax_bytes = Path(spec.tax_regime_path).read_bytes()
        identity_bytes = Path(spec.etf_identity_path).read_bytes()
        campaign_bytes = [Path(path).read_bytes() for path in spec.historical.campaign_config_paths]
        report = run_pension_selection(spec, settings, seed=seed)
        # 식별자는 실행이 고정한 스냅샷에서만 가져온다; 라벨은 게시된 적 없는(부재) 선택 입력에만 쓰인다.
        consumed = report.manifest_hashes
        manifest_hashes = [
            str(consumed[str(Dataset.PRICES)]),
            str(consumed[str(Dataset.FX_KRW_BASE)]),
            consumed.get(str(Dataset.FX)) or "NO_FX_FALLBACK",
            consumed.get(str(Dataset.CPI)) or "NO_TRUSTED_CPI",
        ]
        campaign_hashes = [hashlib.sha256(value).hexdigest() for value in campaign_bytes]
        git_commit = _resolve_git_commit()
        digest = hashlib.sha256(
            config_bytes
            + git_commit.encode()
            + "".join(manifest_hashes).encode()
            + tax_bytes
            + identity_bytes
            + b"".join(campaign_bytes)
            + str(seed).encode()
        ).hexdigest()[:16]
        provenance: dict[str, str] = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "tax_regime_sha256": hashlib.sha256(tax_bytes).hexdigest(),
            "etf_identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "campaign_config_sha256": ",".join(campaign_hashes),
            "git_commit": git_commit,
            "seed": str(seed),
        }
        report_path = write_pension_selection_report(
            report,
            settings,
            experiment_id=digest,
            provenance=provenance,
        )
    except (*_ERRORS, PensionDataError, PensionSelectionDataError, UntrustedDatasetError) as exc:
        logger.error(
            "[PORTFOLIO] event=pension_selection_cli_failed reason_type=%s reason=%s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return 1
    logger.info(
        "[PORTFOLIO] event=pension_selection_cli_done experiment=%s experiment_id=%s status=%s selected=%s report=%s",
        spec.name,
        digest,
        report.status,
        report.selected_arm_id or "NONE",
        report_path,
    )
    return 0


def _read_incumbent_id(record_path: str) -> str:
    """Read the held candidate id from a frozen decision record file."""
    import json

    try:
        document = json.loads(Path(record_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"pension incumbent record is unreadable: {record_path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"pension incumbent record lacks an incumbent_id string: {record_path}")
    raw_id = document.get("incumbent_id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError(f"pension incumbent record lacks an incumbent_id string: {record_path}")
    return raw_id.strip()


def _read_record_id(record_path: str) -> str | None:
    """Read the frozen record id from a decision record file, if present."""
    import json

    try:
        document = json.loads(Path(record_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"pension incumbent record is unreadable: {record_path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"pension incumbent record lacks a record_id string: {record_path}")
    raw_id = document.get("record_id")
    if raw_id is None:
        return None
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError(f"pension incumbent record lacks a record_id string: {record_path}")
    return raw_id.strip()


def run_pension_review_command(*, record_path: str, as_of: date, settings: DataSettings) -> int:
    """Report post-cutoff tracking state for a frozen pension decision.

    Returns: 0 for HOLD or INSUFFICIENT_DATA, 3 for REVIEW_DUE, 1 on invalid data.
    """
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    from src.data.catalog import load_snapshot_visible, resolve_snapshot
    from src.data.schema import Dataset as _Dataset
    from src.validation.pension_decision_record import (
        evaluate_pension_review,
        load_pension_decision_record,
    )

    try:
        record = load_pension_decision_record(record_path)
        as_of_ts = _datetime(as_of.year, as_of.month, as_of.day, 23, 59, tzinfo=_UTC)
        snapshot = resolve_snapshot(settings, (_Dataset.PRICES,))
        prices = load_snapshot_visible(snapshot, _Dataset.PRICES, as_of_ts)
        status = evaluate_pension_review(record, prices, as_of_ts)
    except (*_ERRORS, UntrustedDatasetError) as exc:
        logger.error(
            "[PORTFOLIO] event=pension_review_cli_failed reason_type=%s reason=%s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return 1
    logger.info(
        "[PORTFOLIO] event=pension_review_cli_done record=%s as_of=%s state=%s months=%d ratio=%s",
        status.record_id,
        status.as_of.isoformat(),
        status.state,
        status.months_observed,
        f"{status.incumbent_over_benchmark:.6f}" if status.incumbent_over_benchmark is not None else "NONE",
    )
    return 3 if status.state == "REVIEW_DUE" else 0


def _decision_markdown(report: PensionDecisionReport, trial_count: int, dominance_reference_id: str | None) -> str:
    """Render a Korean human summary of the pension decision verdict."""
    lines = [
        f"# 연금 결정 {report.name}",
        "",
        f"- 상태: {report.status}",
        f"- 선택: {report.selected_id or '없음'}",
        f"- 동등 후보: {', '.join(report.equivalent_ids) if report.equivalent_ids else '없음'}",
        f"- 사유: {', '.join(report.reasons) if report.reasons else '없음'}",
        f"- 시행 횟수: {trial_count}",
        "",
        "## 후보 강건 점수",
        "",
        "| 후보 | 강건 점수 | 부트스트랩 승률 |",
        "| --- | --- | --- |",
    ]
    for candidate_id, score in sorted(report.robust_scores.items()):
        share = report.bootstrap_win_share.get(candidate_id)
        share_text = f"{share:.3f}" if share is not None else "없음"
        lines.append(f"| {candidate_id} | {score:.6f} | {share_text} |")
    lines += ["", f"세후 순위 일치: {'예' if report.tax_rank_agreement else '아니오'}"]
    lines += [
        "",
        "## 도미넌스 가드",
        "",
        f"- 기준 후보: {dominance_reference_id or '없음'}",
        f"- 제외 후보: {', '.join(report.guard_excluded_ids) if report.guard_excluded_ids else '없음'}",
        "",
        "| 후보 | 기준 대비 최소 비율 |",
        "| --- | --- |",
    ]
    for candidate_id in sorted(report.dominance_min_ratio):
        lines.append(f"| {candidate_id} | {report.dominance_min_ratio[candidate_id]:.6f} |")
    lines += [
        "",
        "## 컨트롤",
        "",
        "| 컨트롤 | 현대 점수 | 기준 대비 |",
        "| --- | --- | --- |",
    ]
    for control_id in sorted(report.control_scores):
        versus = report.control_vs_reference.get(control_id)
        versus_text = f"{versus:.6f}" if versus is not None else "없음"
        lines.append(f"| {control_id} | {report.control_scores[control_id]:.6f} | {versus_text} |")
    return "\n".join(lines) + "\n"


def run_pension_decision_command(*, config_path: str, settings: DataSettings, seed: int, incumbent_record: str | None = None, freeze: bool = False) -> int:
    """Run the pre-registered robust pension holding decision and persist its evidence.

    Returns: Zero after a reproducible report is written, nonzero on invalid data or policy.
    Raises: No domain exception escapes the CLI boundary; failure is logged and returned.
    """
    from dataclasses import replace

    import polars as pl

    from src.data.catalog import load_snapshot_visible, resolve_snapshot
    from src.data.result_store import ResultKind, write_result
    from src.sim.pension_engine import PensionDataError
    from src.sim.pension_monthly import panel_from_research
    from src.sim.pension_splice import splice_modern_panel
    from src.validation.pension_campaign import load_pension_campaign_spec, run_pension_campaign
    from src.validation.pension_decision import assert_tax_rank_neutrality, evaluate_pension_decision
    from src.validation.pension_decision_config import load_pension_decision_spec

    try:
        import hashlib
        from datetime import UTC, datetime

        spec = load_pension_decision_spec(config_path)
        config_bytes = Path(config_path).read_bytes()
        incumbent_id = _read_incumbent_id(incumbent_record) if incumbent_record else None
        snapshot = resolve_snapshot(settings, (Dataset.PRICES, Dataset.RESEARCH_MONTHLY))
        modern_as_of = datetime(spec.modern_end.year, spec.modern_end.month, spec.modern_end.day, 23, 59, tzinfo=UTC)
        # 두 근거층 모두 같은 결정 시점(modern_end)에 공개된 행만 읽는다; 이후 공개분은 미래 정보다.
        modern_frame = load_snapshot_visible(snapshot, Dataset.PRICES, modern_as_of)
        research_frame_at_as_of = load_snapshot_visible(snapshot, Dataset.RESEARCH_MONTHLY, modern_as_of)
        century_frame = research_frame_at_as_of.filter(
            pl.col("period_end").is_between(spec.century_start, spec.century_end)
        )
        sleeves = sorted(
            {sleeve for schedule in spec.candidates.values() for sleeve in schedule.start_weights}
            | {sleeve for schedule in spec.controls.values() for sleeve in schedule.start_weights}
        )
        modern, splice_records = splice_modern_panel(
            modern_frame,
            research_frame_at_as_of,
            sleeves,
            spec.modern_splices,
            modern_as_of,
            spec.modern_start,
            spec.modern_end,
        )
        century = panel_from_research(century_frame, spec.century_series, modern_as_of)
        if century.months[0] != spec.century_start or century.months[-1] != spec.century_end:
            raise ValueError(
                f"pension century panel spans {century.months[0].isoformat()}..{century.months[-1].isoformat()}, "
                f"config requires {spec.century_start.isoformat()}..{spec.century_end.isoformat()} "
                f"visible at {modern_as_of.isoformat()}"
            )
        report = evaluate_pension_decision(spec, modern, century, seed=seed, incumbent_id=incumbent_id)
        consumed = {
            str(dataset): Path(snapshot.artifacts[dataset].manifest_path).stem
            for dataset in (Dataset.PRICES, Dataset.RESEARCH_MONTHLY)
        }
        report = replace(report, manifest_hashes=consumed)
        campaign_spec = load_pension_campaign_spec(spec.tax_crosscheck_campaign_path)
        campaign_report = run_pension_campaign(campaign_spec, settings, seed=seed)
        summaries = [
            {
                "arm_id": summary.arm_id,
                "horizon_months": summary.horizon_months,
                "median_wealth_ratio": summary.median_wealth_ratio,
            }
            for summary in campaign_report.summaries
        ]
        agreement = assert_tax_rank_neutrality(report, summaries, spec.tax_crosscheck_arm_map)
        if agreement:
            report = replace(report, tax_rank_agreement=True)
        else:
            logger.warning(
                "[PORTFOLIO] event=pension_decision_tax_disagreement status=%s selected=%s",
                report.status,
                report.selected_id or "NONE",
            )
            report = replace(
                report,
                status="NO_DECISION",
                selected_id=None,
                reasons=(*report.reasons, "TAX_RANK_DISAGREEMENT"),
                tax_rank_agreement=False,
            )
        campaign_bytes = Path(spec.tax_crosscheck_campaign_path).read_bytes()
        git_commit = _resolve_git_commit()
        digest = hashlib.sha256(
            config_bytes + git_commit.encode() + "".join(consumed.values()).encode()
            + campaign_bytes + str(seed).encode()
        ).hexdigest()[:16]
        provenance: dict[str, str] = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "tax_crosscheck_config_sha256": hashlib.sha256(campaign_bytes).hexdigest(),
            "git_commit": git_commit,
            "seed": str(seed),
            "freeze": str(freeze),
        }
        if incumbent_record:
            provenance["incumbent_record"] = incumbent_record
        payload: dict[str, object] = {
            "name": report.name,
            "status": report.status,
            "selected_id": report.selected_id,
            "equivalent_ids": list(report.equivalent_ids),
            "reasons": list(report.reasons),
            "trial_count": report.trial_count,
            "robust_scores": dict(report.robust_scores),
            "sensitivity_robust_scores": {
                str(gamma): dict(scores) for gamma, scores in report.sensitivity_robust_scores.items()
            },
            "bootstrap_win_share": dict(report.bootstrap_win_share),
            "tax_rank_agreement": report.tax_rank_agreement,
            "manifest_hashes": dict(report.manifest_hashes),
            "dominance_reference_id": spec.dominance_reference_id,
            "dominance_min_ratio": dict(report.dominance_min_ratio),
            "guard_excluded_ids": list(report.guard_excluded_ids),
            "control_scores": dict(report.control_scores),
            "control_vs_reference": dict(report.control_vs_reference),
            "modern_splices": [
                {
                    "sleeve": record.sleeve,
                    "proxy_first_month": record.proxy_first_month.isoformat(),
                    "proxy_last_month": record.proxy_last_month.isoformat(),
                    "etf_first_month": record.etf_first_month.isoformat(),
                    "proxy_weights": dict(record.proxy_weights),
                }
                for record in splice_records
            ],
            "sleeve_products": {
                sleeve: {
                    "krx_code": product.krx_code,
                    "name": product.name,
                    "listing_date": product.listing_date.isoformat(),
                    "total_expense_ratio": product.total_expense_ratio,
                    "currency_hedged": product.currency_hedged,
                    "source_url": product.source_url,
                    "source_checked_date": product.source_checked_date.isoformat(),
                }
                for sleeve, product in spec.sleeve_products.items()
            },
            "scores": [
                {
                    "candidate_id": score.candidate_id,
                    "tier": score.tier,
                    "horizon_years": score.horizon_years,
                    "gamma": score.gamma,
                    "ce_ratio": score.ce_ratio,
                    "median_ratio": score.median_ratio,
                    "worst_ratio": score.worst_ratio,
                    "cohort_count": score.cohort_count,
                    "median_pre_retirement_drawdown": score.median_pre_retirement_drawdown,
                }
                for score in report.scores
            ],
        }
        ref = write_result(
            settings,
            experiment=spec.name,
            kind=ResultKind.PENSION_DECISION,
            run_id=digest,
            payload=payload,
            markdown=_decision_markdown(report, report.trial_count, spec.dominance_reference_id),
        )
        if freeze and report.status != "NO_DECISION":
            from src.data.paths import resolve_repository_paths
            from src.validation.pension_decision_record import freeze_pension_decision

            previous_id: str | None = None
            if incumbent_record:
                previous_id = _read_record_id(incumbent_record)
            record_path = freeze_pension_decision(
                report,
                spec,
                output_dir=resolve_repository_paths(settings).root / "records" / "pension_decisions",
                frozen_at=datetime.now(UTC),
                git_commit=git_commit,
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                previous_record_id=previous_id,
            )
            logger.info(
                "[PORTFOLIO] event=pension_decision_frozen record=%s report=%s",
                record_path.as_posix(),
                ref.json_path.as_posix(),
            )
    except (*_ERRORS, PensionDataError, UntrustedDatasetError) as exc:
        logger.error(
            "[PORTFOLIO] event=pension_decision_cli_failed reason_type=%s reason=%s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return 1
    logger.info(
        "[PORTFOLIO] event=pension_decision_cli_done experiment=%s experiment_id=%s status=%s selected=%s freeze=%s report=%s",
        spec.name,
        digest,
        report.status,
        report.selected_id or "NONE",
        freeze,
        ref.json_path.as_posix(),
    )
    return 0
