# ruff: noqa: PERF401
"""Final historical campaign (reporting-only, frozen arms)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src.validation.experiment import ExperimentSpec

if TYPE_CHECKING:
    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig, AllocationResult

__all__ = [
    "FINAL_HISTORICAL_ARMS",
    "FINAL_HISTORICAL_CAMPAIGN_ID",
    "REGIME_COVERAGE_CATALOG",
    "FinalHistoricalArmId",
    "FinalHistoricalArmMetrics",
    "FinalHistoricalArmSpec",
    "FinalHistoricalCampaignReport",
    "RegimeCoverageReport",
    "RegimeCoverageRow",
    "RegimeWindow",
    "TrialLineageCensusReport",
    "TrialLineageFamilyRow",
    "assert_final_campaign_spec",
    "audit_regime_coverage",
    "build_trial_lineage_census",
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
class RegimeWindow:
    regime_name: str
    start: date
    end: date


@dataclass(frozen=True, slots=True)
class RegimeCoverageRow:
    regime_name: str
    covered: bool
    overlap_months: int


@dataclass(frozen=True, slots=True)
class RegimeCoverageReport:
    rows: tuple[RegimeCoverageRow, ...]
    independent_sample_warning: bool


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
    cohort_starts: tuple[date, ...]
    cohort_ends: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class FinalHistoricalCampaignReport:
    campaign_id: str
    window_start: date
    window_end: date
    arm_rows: tuple[FinalHistoricalArmMetrics, ...]
    regime_coverage: RegimeCoverageReport
    lineage_census: TrialLineageCensusReport
    operational_unlock: bool


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
        rows.append(RegimeCoverageRow(regime_name=regime.regime_name, covered=covered, overlap_months=max_overlap))
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
    # experiments_dir is noted but not strictly required; keep for wiring spec
    _ = experiments_dir
    text = index_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    files = payload.get("files", {})
    families: dict[str, dict[str, int]] = {}
    for filename, meta in files.items():
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


def run_final_historical_campaign(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    seed: int,
    bootstrap_paths: int = 400,
    cohort_horizon_months: int = 120,
    cohort_step_months: int = 12,
) -> FinalHistoricalCampaignReport:
    assert_final_campaign_spec(spec)
    from src.sim.allocation import AllocationConfig
    from src.validation.cost_grid import COST_SCENARIOS
    from src.validation.gate import certainty_equivalent, cohort_win_rate, wealth_quantile
    from src.validation.windows import rolling_cohorts

    cohorts = rolling_cohorts(spec.start, spec.end, horizon_months=cohort_horizon_months, step_months=cohort_step_months)
    if not cohorts:
        raise ValueError("no rolling cohorts fit the experiment window")

    # Prepare baseline targets
    baseline_targets = dict(spec.baseline.targets) if spec.baseline.targets is not None else None

    arm_rows: list[FinalHistoricalArmMetrics] = []

    for idx, candidate in enumerate(spec.candidates):
        candidate_targets = dict(candidate.targets) if candidate.targets is not None else None
        candidate_wealths: list[float] = []
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
            cand_cfg = AllocationConfig(
                policy=candidate.policy,
                start=c_start,
                end=c_end,
                monthly_contribution_krw=float(spec.contribution_krw),
                fill_delay_sessions=1,
                commission_bps=float(spec.commission_bps),
                fx_spread_bps=float(spec.fx_spread_bps),
                targets_override=candidate_targets,
            )
            base_res = runner(base_cfg)
            cand_res = runner(cand_cfg)
            bw = float(base_res.terminal_wealth_real_krw)
            cw = float(cand_res.terminal_wealth_real_krw)
            if not math.isfinite(bw) or bw <= 0 or not math.isfinite(cw) or cw <= 0:
                raise ValueError(f"wealths must be finite positive, got {cw!r} vs {bw!r}")
            candidate_wealths.append(cw)
            baseline_wealths.append(bw)

        ratios = tuple(float(c) / float(b) for c, b in zip(candidate_wealths, baseline_wealths, strict=True))
        median_ratio = wealth_quantile(ratios, 0.5)
        p10_ratio = wealth_quantile(ratios, 0.1)
        worst_ratio = min(ratios) if ratios else 0.0
        win_rate = cohort_win_rate(candidate_wealths, baseline_wealths)
        # CE gamma10 ratio
        try:
            ce_cand = certainty_equivalent(candidate_wealths, gamma=10.0)
            ce_base = certainty_equivalent(baseline_wealths, gamma=10.0)
            ce_gamma_10 = float(ce_cand / ce_base) if ce_base != 0 else 0.0
        except Exception:
            ce_gamma_10 = 0.0

        # Full-span runs for bootstrap and xirr and cost stress
        base_full_cfg = AllocationConfig(
            policy=spec.baseline.policy,
            start=spec.start,
            end=spec.end,
            monthly_contribution_krw=float(spec.contribution_krw),
            fill_delay_sessions=1,
            commission_bps=float(spec.commission_bps),
            fx_spread_bps=float(spec.fx_spread_bps),
            targets_override=baseline_targets,
        )
        cand_full_cfg = AllocationConfig(
            policy=candidate.policy,
            start=spec.start,
            end=spec.end,
            monthly_contribution_krw=float(spec.contribution_krw),
            fill_delay_sessions=1,
            commission_bps=float(spec.commission_bps),
            fx_spread_bps=float(spec.fx_spread_bps),
            targets_override=candidate_targets,
        )
        base_full = runner(base_full_cfg)
        cand_full = runner(cand_full_cfg)
        xirr_real = float(cand_full.xirr_real) if math.isfinite(float(cand_full.xirr_real)) else 0.0

        # bootstrap
        bootstrap_win_rate = 0.0
        bootstrap_p05 = 0.0
        try:
            # reuse incremental monthly_simple_returns when snapshots exist
            if base_full.snapshots and cand_full.snapshots and len(base_full.snapshots) >= 2 and len(cand_full.snapshots) >= 2:
                from src.analytics.thesis.incremental import monthly_simple_returns, paired_path_block_bootstrap

                cand_rets = monthly_simple_returns(cand_full)
                base_rets = monthly_simple_returns(base_full)
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
                else:
                    bootstrap_win_rate = 0.0
                    bootstrap_p05 = 0.0
            else:
                bootstrap_win_rate = 0.0
                bootstrap_p05 = 0.0
        except Exception:
            bootstrap_win_rate = 0.0
            bootstrap_p05 = 0.0

        # cost stress worst ratio
        cost_stress_worst_ratio = 1.0
        try:
            ideal_wealth: float | None = None
            scenario_ratios: list[float] = []
            # need to map ideal wealth
            temp_wealths: dict[str, float] = {}
            for sc in COST_SCENARIOS:
                cfg = AllocationConfig(
                    policy=candidate.policy,
                    start=spec.start,
                    end=spec.end,
                    monthly_contribution_krw=float(spec.contribution_krw),
                    fill_delay_sessions=1,
                    commission_bps=float(sc.commission_bps),
                    fx_spread_bps=float(sc.fx_spread_bps),
                    targets_override=candidate_targets,
                )
                res = runner(cfg)
                w = float(res.terminal_wealth_real_krw)
                temp_wealths[sc.id] = w
                if sc.id == "ideal":
                    ideal_wealth = w
            if ideal_wealth is not None and ideal_wealth > 0 and math.isfinite(ideal_wealth):
                for w in temp_wealths.values():  # noqa: PERF401
                    if math.isfinite(w) and w > 0:
                        scenario_ratios.append(w / ideal_wealth)
                if scenario_ratios:
                    cost_stress_worst_ratio = min(scenario_ratios)
            else:
                cost_stress_worst_ratio = 1.0
        except Exception:
            cost_stress_worst_ratio = 1.0

        cohort_starts = tuple(c[0] for c in cohorts)
        cohort_ends = tuple(c[1] for c in cohorts)

        arm_rows.append(
            FinalHistoricalArmMetrics(
                arm_id=str(candidate.id),
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
                cohort_starts=cohort_starts,
                cohort_ends=cohort_ends,
            )
        )

    regime_coverage = audit_regime_coverage(cohorts=cohorts)
    # lineage census: default INDEX location
    try:
        lineage_census = build_trial_lineage_census(
            index_path=Path("configs/experiments/INDEX.json"),
            experiments_dir=Path("configs/experiments"),
        )
    except Exception:
        lineage_census = TrialLineageCensusReport(total_experiments=0, families=())

    return FinalHistoricalCampaignReport(
        campaign_id=FINAL_HISTORICAL_CAMPAIGN_ID,
        window_start=spec.start,
        window_end=spec.end,
        arm_rows=tuple(arm_rows),
        regime_coverage=regime_coverage,
        lineage_census=lineage_census,
        operational_unlock=False,
    )


def write_final_historical_campaign_report(
    report: FinalHistoricalCampaignReport,
    settings: DataSettings,
    *,
    experiment_id: str,
) -> Path:
    from src.data.paths import experiments_dir

    payload = {
        "campaign_id": report.campaign_id,
        "window_start": report.window_start.isoformat(),
        "window_end": report.window_end.isoformat(),
        "window": {
            "start": report.window_start.isoformat(),
            "end": report.window_end.isoformat(),
        },
        "arm_rows": [
            {
                "arm_id": row.arm_id,
                "targets": dict(row.targets),
                "cohort_count": int(row.cohort_count),
                "median_ratio": float(row.median_ratio),
                "p10_ratio": float(row.p10_ratio),
                "worst_ratio": float(row.worst_ratio),
                "win_rate": float(row.win_rate),
                "ce_gamma_10": float(row.ce_gamma_10),
                "bootstrap_win_rate": float(row.bootstrap_win_rate),
                "bootstrap_p05": float(row.bootstrap_p05),
                "xirr_real": float(row.xirr_real),
                "cost_stress_worst_ratio": float(row.cost_stress_worst_ratio),
                "cohort_starts": [d.isoformat() for d in row.cohort_starts],
                "cohort_ends": [d.isoformat() for d in row.cohort_ends],
            }
            for row in report.arm_rows
        ],
        "regime_coverage": {
            "rows": [
                {"regime_name": r.regime_name, "covered": bool(r.covered), "overlap_months": int(r.overlap_months)}
                for r in report.regime_coverage.rows
            ],
            "independent_sample_warning": bool(report.regime_coverage.independent_sample_warning),
        },
        "lineage_census": {
            "total_experiments": int(report.lineage_census.total_experiments),
            "families": [
                {
                    "family_id": f.family_id,
                    "experiment_count": int(f.experiment_count),
                    "active_count": int(f.active_count),
                    "archived_count": int(f.archived_count),
                }
                for f in report.lineage_census.families
            ],
        },
        "operational_unlock": bool(report.operational_unlock),
    }
    out_dir = experiments_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.campaign_id}_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
