# ruff: noqa: PERF401,S110
"""Final historical campaign (reporting-only, frozen arms)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from src.data.paths import EXPERIMENT_INDEX_PATH, EXPERIMENTS_DIR
from src.policy.targets import PolicyId
from src.validation.cost_grid import COST_SCENARIOS, CostScenario
from src.validation.experiment import ExperimentSpec
from src.validation.historical_campaign_audit import (
    REGIME_COVERAGE_CATALOG,
    PreHistoryMixProxyStressReport,
    PreHistoryProxyStressReport,
    RegimeCoverageReport,
    RegimeCoverageRow,
    RegimeWindow,
    TrialLineageCensusReport,
    TrialLineageFamilyRow,
    _catalog_research_returns_min_date,
    audit_pre_history_mix_proxy_stress,
    audit_pre_history_proxy_stress,
    audit_regime_coverage,
    build_trial_lineage_census,
    classify_regime_coverage_tier,
)
from src.validation.historical_campaign_report import write_final_historical_campaign_report

if TYPE_CHECKING:
    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.registry import TrialLineageHashCensus

__all__ = [
    "FINAL_HISTORICAL_ARMS",
    "FINAL_HISTORICAL_CAMPAIGN_ID",
    "REGIME_COVERAGE_CATALOG",
    "FinalHistoricalArmId",
    "FinalHistoricalArmMetrics",
    "FinalHistoricalArmSpec",
    "FinalHistoricalCampaignReport",
    "PairedCostStressRow",
    "PreHistoryMixProxyStressReport",
    "PreHistoryProxyStressReport",
    "RegimeCoverageReport",
    "RegimeCoverageRow",
    "RegimeWindow",
    "TaxSensitivityMilestone",
    "TrialLineageCensusReport",
    "TrialLineageFamilyRow",
    "_catalog_research_returns_min_date",
    "assert_final_campaign_spec",
    "audit_pre_history_mix_proxy_stress",
    "audit_pre_history_proxy_stress",
    "audit_regime_coverage",
    "build_trial_lineage_census",
    "classify_regime_coverage_tier",
    "compute_paired_cost_stress_ratios",
    "run_final_historical_campaign",
    "write_final_historical_campaign_report",
]

FINAL_HISTORICAL_CAMPAIGN_ID: Final[str] = "FINAL_HISTORICAL_CAMPAIGN_V1"


class FinalHistoricalArmId(StrEnum):
    B0_QQQ100 = "b0_qqq100"
    C1_QQQ95_SOXX5 = "c1_qqq95_soxx5"
    C2_QQQ90_SOXX10 = "c2_qqq90_soxx10"
    C3_QQQ85_SOXX15 = "c3_qqq85_soxx15"


@dataclass(frozen=True, slots=True)
class FinalHistoricalArmSpec:
    arm_id: FinalHistoricalArmId
    targets: dict[str, float]
    adaptive: bool


FINAL_HISTORICAL_ARMS: Final[tuple[FinalHistoricalArmSpec, ...]] = (
    FinalHistoricalArmSpec(arm_id=FinalHistoricalArmId.B0_QQQ100, targets={"QQQ": 1.0}, adaptive=False),
    FinalHistoricalArmSpec(arm_id=FinalHistoricalArmId.C1_QQQ95_SOXX5, targets={"QQQ": 0.95, "SOXX": 0.05}, adaptive=False),
    FinalHistoricalArmSpec(arm_id=FinalHistoricalArmId.C2_QQQ90_SOXX10, targets={"QQQ": 0.9, "SOXX": 0.1}, adaptive=False),
    FinalHistoricalArmSpec(arm_id=FinalHistoricalArmId.C3_QQQ85_SOXX15, targets={"QQQ": 0.85, "SOXX": 0.15}, adaptive=False),
)


@dataclass(frozen=True, slots=True)
class TaxSensitivityMilestone:
    status: Literal["not_modelled"]
    rationale: str


@dataclass(frozen=True, slots=True)
class PairedCostStressRow:
    scenario_id: str
    candidate_over_baseline_ratio: float


@dataclass(frozen=True, slots=True)
class FinalHistoricalArmMetrics:
    arm_id: str
    targets: dict[str, float]
    cohort_count: int
    median_ratio: float
    p10_ratio: float
    worst_ratio: float
    win_rate: float
    ce_gamma_10: float
    bootstrap_win_rate: float
    bootstrap_p05: float
    xirr_real: float
    cost_stress_worst_ratio: float
    fx_stress_worst_ratio: float
    cohort_starts: tuple[date, ...]
    cohort_ends: tuple[date, ...]
    paired_cost_stress: tuple[PairedCostStressRow, ...] = ()


@dataclass(frozen=True, slots=True)
class FinalHistoricalCampaignReport:
    campaign_id: str
    window_start: date
    window_end: date
    arm_rows: tuple[FinalHistoricalArmMetrics, ...]
    regime_coverage: RegimeCoverageReport
    lineage_census: TrialLineageCensusReport
    tax_sensitivity: TaxSensitivityMilestone
    pre_history_proxy: PreHistoryProxyStressReport
    operational_unlock: bool
    pre_history_mix_proxy: tuple[PreHistoryMixProxyStressReport, ...] = ()
    lineage_hash_census: TrialLineageHashCensus | None = None
    manifest_hashes: Mapping[str, str | None] = field(default_factory=dict)


FINAL_HISTORICAL_MIN_COHORTS: Final[int] = 4
FINAL_HISTORICAL_TARGET_COHORTS: Final[int] = 10
# CPI PIT visible at first month-end execution close (see system-design.md).
_STATIC_DCA_ALLOCATION_MIN_START: Final[date] = date(2012, 8, 31)


def compute_paired_cost_stress_ratios(
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    spec: ExperimentSpec,
    baseline_targets: dict[str, float],
    candidate_targets: dict[str, float],
    scenarios: Sequence[CostScenario] | None = None,
) -> tuple[PairedCostStressRow, ...]:
    from src.sim.allocation import AllocationConfig

    sc_seq: Sequence[CostScenario] = scenarios if scenarios is not None else COST_SCENARIOS
    rows: list[PairedCostStressRow] = []
    baseline_policy = spec.baseline.policy
    candidate_policy = spec.candidates[0].policy if spec.candidates else spec.baseline.policy
    for sc in sc_seq:
        base_cfg = AllocationConfig(
            policy=baseline_policy,
            start=spec.start,
            end=spec.end,
            monthly_contribution_krw=float(spec.contribution_krw),
            fill_delay_sessions=1,
            commission_bps=float(sc.commission_bps),
            fx_spread_bps=float(sc.fx_spread_bps),
            targets_override=dict(baseline_targets),
        )
        cand_cfg = AllocationConfig(
            policy=candidate_policy,
            start=spec.start,
            end=spec.end,
            monthly_contribution_krw=float(spec.contribution_krw),
            fill_delay_sessions=1,
            commission_bps=float(sc.commission_bps),
            fx_spread_bps=float(sc.fx_spread_bps),
            targets_override=dict(candidate_targets),
        )
        base_wealth = float(runner(base_cfg).terminal_wealth_real_krw)
        cand_wealth = float(runner(cand_cfg).terminal_wealth_real_krw)
        ratio = float(cand_wealth / base_wealth) if base_wealth != 0 else 0.0
        rows.append(PairedCostStressRow(scenario_id=str(sc.id), candidate_over_baseline_ratio=float(ratio)))
    return tuple(rows)


def resolve_final_campaign_window(
    spec: ExperimentSpec,
    settings: DataSettings | None = None,
    *,
    as_of: datetime | None = None,
) -> tuple[date, date, tuple[tuple[date, date], ...]]:
    from src.validation.research_posture import SEEN_HISTORY_CUTOFF
    from src.validation.windows import rolling_cohorts

    effective_start = spec.start
    effective_end = min(spec.end, SEEN_HISTORY_CUTOFF)
    if settings is not None:
        try:
            from datetime import UTC

            from src.data.catalog import resolve_snapshot
            from src.data.schema import Dataset, spec_for
            from src.data.storage import DataStore
            from src.policy.targets import policy_sleeves
            from src.validation.experiment import experiment_target_tickers
            from src.validation.feasibility_audit import resolve_earliest_common_usable_start

            mark_policies: tuple[PolicyId, ...] = (spec.baseline.policy, *(c.policy for c in spec.candidates))
            sleeves: dict[str, None] = {}
            for p in mark_policies:
                for t in policy_sleeves(p):
                    sleeves.setdefault(t)
            for t in experiment_target_tickers(spec):
                sleeves.setdefault(t)
            tickers = tuple(sleeves.keys())
            if tickers:
                use_as_of = as_of if as_of is not None else datetime.now(tz=UTC)
                if use_as_of.tzinfo is None:
                    use_as_of = use_as_of.replace(tzinfo=UTC)
                earliest = resolve_earliest_common_usable_start(
                    tickers=tickers, settings=settings, as_of=use_as_of
                )
                effective_start = max(earliest, _STATIC_DCA_ALLOCATION_MIN_START)
                snapshot = resolve_snapshot(settings, (Dataset.PRICES,))
                raw_frame = DataStore(settings).read_normalized(
                    snapshot.artifacts[Dataset.PRICES], spec_for(Dataset.PRICES)
                )
                max_date_raw = raw_frame.get_column("date").max()
                if isinstance(max_date_raw, date):
                    catalog_end = min(max_date_raw, SEEN_HISTORY_CUTOFF)
                    if catalog_end > effective_end:
                        effective_end = catalog_end
        except Exception:
            pass
    cohorts = rolling_cohorts(
        effective_start, effective_end, horizon_months=120, step_months=12
    )
    if not cohorts:
        raise ValueError("no rolling cohorts fit the experiment window: cohort_count==0")
    if len(cohorts) < FINAL_HISTORICAL_MIN_COHORTS:
        raise ValueError(
            f"final campaign requires >={FINAL_HISTORICAL_MIN_COHORTS} cohorts, got cohort_count=={len(cohorts)}"
        )
    return (effective_start, effective_end, cohorts)


def assert_final_campaign_spec(spec: ExperimentSpec) -> None:
    from src.validation.research_posture import SEEN_HISTORY_CUTOFF, ObjectiveFamily, classify_strategy_role

    if not str(spec.name).lower().startswith("final_historical"):
        raise ValueError(f"name must start with 'final_historical', got {spec.name!r}")
    if spec.objective_family is None:
        raise ValueError("objective_family is required; expected capital_allocation")
    if spec.objective_family is not ObjectiveFamily.CAPITAL_ALLOCATION:
        raise ValueError(f"objective_family must be capital_allocation, got {spec.objective_family!r}")
    if spec.end > SEEN_HISTORY_CUTOFF:
        raise ValueError(f"end {spec.end.isoformat()} exceeds SEEN_HISTORY_CUTOFF {SEEN_HISTORY_CUTOFF.isoformat()}")
    if spec.baseline.targets != {"QQQ": 1.0}:
        raise ValueError(f"baseline.targets must be {{'QQQ': 1.0}}, got {spec.baseline.targets!r}")
    if len(spec.candidates) != 3:
        raise ValueError(f"exactly 3 candidates required, got {len(spec.candidates)}")
    # Check targets match expected mixes
    expected = [
        {"QQQ": 0.95, "SOXX": 0.05},
        {"QQQ": 0.9, "SOXX": 0.1},
        {"QQQ": 0.85, "SOXX": 0.15},
    ]

    def _norm(d: dict[str, float] | None) -> frozenset[tuple[str, float]]:
        if d is None:
            return frozenset()
        return frozenset((str(k).strip().upper(), float(v)) for k, v in d.items())

    expected_norms = {_norm(e) for e in expected}
    actual_norms = {_norm(c.targets) for c in spec.candidates}
    if actual_norms != expected_norms:
        raise ValueError(f"candidates targets must be {expected}, got {[c.targets for c in spec.candidates]!r}")
    if spec.adaptive_contribution is not None:
        raise ValueError("adaptive_contribution not allowed for capital_allocation final_historical")
    if spec.baseline_adaptive_contribution is not None:
        raise ValueError("baseline_adaptive_contribution not allowed for capital_allocation final_historical")
    if spec.kafi_deployment is not None:
        raise ValueError("kafi_deployment not allowed for capital_allocation final_historical")
    if spec.reserve is not None:
        raise ValueError("reserve not allowed for capital_allocation final_historical")
    if spec.overlay is not None:
        raise ValueError("overlay not allowed for capital_allocation final_historical")
    if spec.mapping is not None:
        raise ValueError("mapping not allowed for capital_allocation final_historical")
    if spec.currency is not None:
        raise ValueError("currency not allowed for capital_allocation final_historical")
    if spec.cadence is not None:
        raise ValueError("cadence not allowed for capital_allocation final_historical")
    if spec.contribution_shape is not None:
        raise ValueError("contribution_shape not allowed for capital_allocation final_historical")
    if spec.preregistration is not None:
        if not spec.preregistration.weights_locked:
            raise ValueError("preregistration.weights_locked must be True")
        if not spec.preregistration.universe_locked:
            raise ValueError("preregistration.universe_locked must be True")
    # Ensure each arm classifies
    for arm in (spec.baseline, *spec.candidates):
        if arm.targets is not None:
            classify_strategy_role(targets=arm.targets, adaptive=False)


_TAX_SENSITIVITY_MILESTONE: Final[TaxSensitivityMilestone] = TaxSensitivityMilestone(
    status="not_modelled",
    rationale="buy_only_accumulation_defers_realization_tax_until_sale; no PIT tax ledger model",
)


def _stress_worst_ratio(
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    policy: PolicyId,
    start: date,
    end: date,
    contribution_krw: float,
    commission_bps: float,
    fx_spread_bps: float,
    targets_override: dict[str, float] | None,
    scenarios: Sequence[tuple[str, float, float]],
) -> float:
    from src.sim.allocation import AllocationConfig

    ideal_wealth: float | None = None
    wealths: list[float] = []
    for scenario_id, comm, fx in scenarios:
        cfg = AllocationConfig(
            policy=policy,
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
            fill_delay_sessions=1,
            commission_bps=float(comm),
            fx_spread_bps=float(fx),
            targets_override=targets_override,
        )
        w = float(runner(cfg).terminal_wealth_real_krw)
        wealths.append(w)
        if scenario_id in ("ideal", "fx_ideal"):
            ideal_wealth = w
    if ideal_wealth is None or not math.isfinite(ideal_wealth) or ideal_wealth <= 0:
        return 1.0
    ratios = [w / ideal_wealth for w in wealths if math.isfinite(w) and w > 0]
    return min(ratios) if ratios else 1.0


def _compute_arm_metrics(
    runner: Callable[[AllocationConfig], AllocationResult],
    spec: ExperimentSpec,
    *,
    arm_id: str,
    policy: PolicyId,
    targets: dict[str, float] | None,
    cohorts: Sequence[tuple[date, date]],
    baseline_wealths: Sequence[float],
    idx: int,
    seed: int,
    bootstrap_paths: int,
    vs_baseline: bool,
) -> FinalHistoricalArmMetrics:
    from src.sim.allocation import AllocationConfig
    from src.validation.cost_grid import COST_SCENARIOS, fx_stress_scenarios
    from src.validation.gate import certainty_equivalent, cohort_win_rate, wealth_quantile

    candidate_targets = dict(targets) if targets is not None else None
    candidate_wealths: list[float] = []
    for c_start, c_end in cohorts:
        cand_cfg = AllocationConfig(
            policy=policy,
            start=c_start,
            end=c_end,
            monthly_contribution_krw=float(spec.contribution_krw),
            fill_delay_sessions=1,
            commission_bps=float(spec.commission_bps),
            fx_spread_bps=float(spec.fx_spread_bps),
            targets_override=candidate_targets,
        )
        cand_res = runner(cand_cfg)
        cw = float(cand_res.terminal_wealth_real_krw)
        if not math.isfinite(cw) or cw <= 0:
            raise ValueError(f"wealths must be finite positive, got {cw!r}")
        candidate_wealths.append(cw)

    if vs_baseline:
        ratios = tuple(float(c) / float(b) for c, b in zip(candidate_wealths, baseline_wealths, strict=True))
        median_ratio = wealth_quantile(ratios, 0.5)
        p10_ratio = wealth_quantile(ratios, 0.1)
        worst_ratio = min(ratios) if ratios else 0.0
        win_rate = cohort_win_rate(candidate_wealths, list(baseline_wealths))
    else:
        median_ratio = 1.0
        p10_ratio = 1.0
        worst_ratio = 1.0
        win_rate = 1.0

    try:
        ce_cand = certainty_equivalent(candidate_wealths, gamma=10.0)
        ce_base = certainty_equivalent(list(baseline_wealths), gamma=10.0)
        ce_gamma_10 = float(ce_cand / ce_base) if vs_baseline and ce_base != 0 else 1.0
    except Exception:
        ce_gamma_10 = 1.0 if not vs_baseline else 0.0

    cand_full_cfg = AllocationConfig(
        policy=policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=float(spec.contribution_krw),
        fill_delay_sessions=1,
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=candidate_targets,
    )
    base_full_cfg = AllocationConfig(
        policy=spec.baseline.policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=float(spec.contribution_krw),
        fill_delay_sessions=1,
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=dict(spec.baseline.targets) if spec.baseline.targets is not None else None,
    )
    cand_full = runner(cand_full_cfg)
    base_full = runner(base_full_cfg)
    xirr_real = float(cand_full.xirr_real) if math.isfinite(float(cand_full.xirr_real)) else 0.0

    bootstrap_win_rate = 0.0
    bootstrap_p05 = 0.0
    try:
        if (
            base_full.snapshots
            and cand_full.snapshots
            and len(base_full.snapshots) >= 2
            and len(cand_full.snapshots) >= 2
        ):
            from src.analytics.thesis.incremental import monthly_unitized_returns, paired_path_block_bootstrap

            cand_rets = monthly_unitized_returns(cand_full)
            base_rets = monthly_unitized_returns(base_full)
            if cand_rets and base_rets and len(cand_rets) == len(base_rets) and len(cand_rets) >= 1:
                block_size = 12 if len(cand_rets) >= 12 else len(cand_rets)
                verdict = paired_path_block_bootstrap(
                    cand_rets,
                    base_rets,
                    block_size=block_size,
                    n_paths=int(bootstrap_paths),
                    seed=int(seed) + idx,
                )
                bootstrap_win_rate = float(verdict.win_rate)
                bootstrap_p05 = float(verdict.p05_terminal_ratio)
    except Exception:
        bootstrap_win_rate = 0.0
        bootstrap_p05 = 0.0

    cost_scenarios = [(sc.id, sc.commission_bps, sc.fx_spread_bps) for sc in COST_SCENARIOS]
    cost_stress_worst_ratio = _stress_worst_ratio(
        runner,
        policy=policy,
        start=spec.start,
        end=spec.end,
        contribution_krw=float(spec.contribution_krw),
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=candidate_targets,
        scenarios=cost_scenarios,
    )
    fx_scenarios = [(sc.id, sc.commission_bps, sc.fx_spread_bps) for sc in fx_stress_scenarios(float(spec.commission_bps))]
    fx_stress_worst_ratio = _stress_worst_ratio(
        runner,
        policy=policy,
        start=spec.start,
        end=spec.end,
        contribution_krw=float(spec.contribution_krw),
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=candidate_targets,
        scenarios=fx_scenarios,
    )

    cohort_starts = tuple(c[0] for c in cohorts)
    cohort_ends = tuple(c[1] for c in cohorts)
    try:
        if vs_baseline:
            baseline_t = dict(spec.baseline.targets) if spec.baseline.targets is not None else {}
            cand_t = dict(candidate_targets) if candidate_targets is not None else {}
            paired_cost_stress = compute_paired_cost_stress_ratios(
                runner, spec=spec, baseline_targets=baseline_t, candidate_targets=cand_t
            )
        else:
            paired_cost_stress = ()
    except Exception:
        paired_cost_stress = ()

    return FinalHistoricalArmMetrics(
        arm_id=arm_id,
        targets=dict(candidate_targets) if candidate_targets is not None else {},
        cohort_count=len(cohorts),
        median_ratio=float(median_ratio),
        p10_ratio=float(p10_ratio),
        worst_ratio=float(worst_ratio),
        win_rate=float(win_rate),
        ce_gamma_10=float(ce_gamma_10),
        bootstrap_win_rate=float(bootstrap_win_rate),
        bootstrap_p05=float(bootstrap_p05),
        xirr_real=float(xirr_real),
        cost_stress_worst_ratio=float(cost_stress_worst_ratio),
        fx_stress_worst_ratio=float(fx_stress_worst_ratio),
        cohort_starts=cohort_starts,
        cohort_ends=cohort_ends,
        paired_cost_stress=paired_cost_stress,
    )


def run_final_historical_campaign(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    seed: int,
    bootstrap_paths: int = 400,
    cohort_horizon_months: int = 120,
    cohort_step_months: int = 12,
    settings: DataSettings | None = None,
) -> FinalHistoricalCampaignReport:
    assert_final_campaign_spec(spec)
    from src.sim.allocation import AllocationConfig

    # wiring: resolve_final_campaign_window invocation
    _ = resolve_final_campaign_window

    if cohort_horizon_months == 120 and cohort_step_months == 12:
        effective_start, effective_end, cohorts = resolve_final_campaign_window(spec, settings)
    else:
        from src.validation.windows import rolling_cohorts

        effective_start = spec.start
        effective_end = spec.end
        cohorts = rolling_cohorts(spec.start, spec.end, horizon_months=cohort_horizon_months, step_months=cohort_step_months)
        if not cohorts:
            raise ValueError("no rolling cohorts fit the experiment window")

    baseline_targets = dict(spec.baseline.targets) if spec.baseline.targets is not None else None
    baseline_wealths: list[float] = []
    for c_start, c_end in cohorts:
        base_cfg = AllocationConfig(
            policy=spec.baseline.policy,
            start=c_start,
            end=c_end,
            monthly_contribution_krw=float(spec.contribution_krw),
            fill_delay_sessions=1,
            commission_bps=float(spec.commission_bps),
            fx_spread_bps=float(spec.fx_spread_bps),
            targets_override=baseline_targets,
        )
        bw = float(runner(base_cfg).terminal_wealth_real_krw)
        if not math.isfinite(bw) or bw <= 0:
            raise ValueError(f"baseline wealth must be finite positive, got {bw!r}")
        baseline_wealths.append(bw)

    arm_rows: list[FinalHistoricalArmMetrics] = []
    arm_rows.append(
        _compute_arm_metrics(
            runner,
            spec,
            arm_id=str(spec.baseline.id),
            policy=spec.baseline.policy,
            targets=baseline_targets,
            cohorts=cohorts,
            baseline_wealths=baseline_wealths,
            idx=0,
            seed=seed,
            bootstrap_paths=bootstrap_paths,
            vs_baseline=False,
        )
    )
    for idx, candidate in enumerate(spec.candidates):
        candidate_targets = dict(candidate.targets) if candidate.targets is not None else None
        arm_rows.append(
            _compute_arm_metrics(
                runner,
                spec,
                arm_id=str(candidate.id),
                policy=candidate.policy,
                targets=candidate_targets,
                cohorts=cohorts,
                baseline_wealths=baseline_wealths,
                idx=idx + 1,
                seed=seed,
                bootstrap_paths=bootstrap_paths,
                vs_baseline=True,
            )
        )

    regime_coverage = audit_regime_coverage(cohorts=cohorts)
    lineage_hash_census = None
    if settings is not None:
        try:
            from src.data.paths import results_root
            from src.validation.registry import scan_executed_strategy_hash_census

            lineage_hash_census = scan_executed_strategy_hash_census(results_root(settings))
        except Exception:
            lineage_hash_census = None
    try:
        lineage_census = build_trial_lineage_census(
            index_path=EXPERIMENT_INDEX_PATH,
            experiments_dir=EXPERIMENTS_DIR,
        )
    except Exception:
        lineage_census = TrialLineageCensusReport(total_experiments=0, families=())

    dot_com = next(r for r in REGIME_COVERAGE_CATALOG if r.regime_name == "dot_com")
    gfc = next(r for r in REGIME_COVERAGE_CATALOG if r.regime_name == "gfc")
    pre_history_mix_proxy: tuple[PreHistoryMixProxyStressReport, ...] = ()
    manifest_hashes: dict[str, str | None] = {}
    if settings is not None:
        from src.data.catalog import resolve_snapshot
        from src.data.schema import Dataset as _PinnedDataset

        for _pinned in (_PinnedDataset.PRICES, _PinnedDataset.RESEARCH_RETURNS):
            _manifests_dir = settings.resolved_data_root() / "manifests" / str(_pinned)
            if not _manifests_dir.is_dir() or not any(_manifests_dir.glob("*.json")):
                manifest_hashes[str(_pinned)] = None
                continue
            manifest_hashes[str(_pinned)] = Path(
                resolve_snapshot(settings, (_pinned,)).artifacts[_pinned].manifest_path
            ).stem
        pre_history_proxy = audit_pre_history_proxy_stress(
            settings,
            proxy_start=dot_com.start,
            proxy_end=spec.end,
            contribution_krw=float(spec.contribution_krw),
            fallback_starts=(spec.start,),
        )
        mix_reports: list[PreHistoryMixProxyStressReport] = []
        for regime in (dot_com, gfc):
            mix_reports.append(
                audit_pre_history_mix_proxy_stress(
                    settings,
                    window_start=regime.start,
                    window_end=regime.end,
                    contribution_krw=float(spec.contribution_krw),
                    baseline_series="NDX100",
                    candidate_weights={"NDX100": 0.9, "SOX": 0.1},
                    regime_name=regime.regime_name,
                )
            )
        pre_history_mix_proxy = tuple(mix_reports)
    else:
        pre_history_proxy = PreHistoryProxyStressReport(
            status="unavailable",
            reason="settings required for ff_proxy pre-history stress",
            proxy_window_start=dot_com.start,
            proxy_window_end=spec.end,
        )

    return FinalHistoricalCampaignReport(
        campaign_id=FINAL_HISTORICAL_CAMPAIGN_ID,
        window_start=effective_start if cohort_horizon_months == 120 and cohort_step_months == 12 else spec.start,
        window_end=effective_end if cohort_horizon_months == 120 and cohort_step_months == 12 else spec.end,
        arm_rows=tuple(arm_rows),
        regime_coverage=regime_coverage,
        lineage_census=lineage_census,
        tax_sensitivity=_TAX_SENSITIVITY_MILESTONE,
        pre_history_proxy=pre_history_proxy,
        pre_history_mix_proxy=pre_history_mix_proxy,
        lineage_hash_census=lineage_hash_census,
        operational_unlock=False,
        manifest_hashes=manifest_hashes,
    )
