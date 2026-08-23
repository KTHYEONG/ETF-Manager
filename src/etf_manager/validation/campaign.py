"""Walk-forward adoption campaign over an injected allocation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig
from src.etf_manager.validation.experiment import ExperimentSpec
from src.etf_manager.validation.gate import adoption_passes, certainty_equivalent
from src.etf_manager.validation.windows import walk_forward_windows

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from src.etf_manager.data.settings import DataSettings
    from src.etf_manager.sim.allocation import AllocationResult

__all__ = ["CampaignReport", "FoldOutcome", "run_walk_forward_adoption", "write_campaign_report"]

_CE_GAMMAS: Final[tuple[float, ...]] = (2.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    """Train-phase adoption decision plus realized test-phase wealths."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_adopted: bool
    chosen_policy: PolicyId
    baseline_test_wealth: float
    candidate_test_wealth: float
    chosen_test_wealth: float


@dataclass(frozen=True, slots=True)
class CampaignReport:
    """Every fold outcome plus the pooled adopted-vs-baseline CE verdict.

    ``candidate_test_ce`` is the always-candidate diagnostic; the verdict comes
    only from chosen versus baseline.
    """

    name: str
    candidate_id: str
    modules: int
    folds: tuple[FoldOutcome, ...]
    baseline_test_ce: Mapping[float, float]
    candidate_test_ce: Mapping[float, float]
    chosen_test_ce: Mapping[float, float]
    process_adopted_vs_baseline: bool


def _arm_config(spec: ExperimentSpec, policy: PolicyId, start: date, end: date) -> AllocationConfig:
    """Identical cashflow/costs for every arm on one sliced window."""
    return AllocationConfig(
        policy=policy,
        start=start,
        end=end,
        monthly_contribution_krw=spec.contribution_krw,
        fill_delay_sessions=1,
        fx_spread_bps=0.0,
        commission_bps=0.0,
        tilt=None,
        rebalance_band=None,
        overlay=None,
        currency=None,
        mapping=None,
    )


def _singleton_ce(wealth: float) -> Mapping[float, float]:
    """CE gammas of a one-observation wealth vector."""
    return {gamma: certainty_equivalent((wealth,), gamma=gamma) for gamma in _CE_GAMMAS}


def run_walk_forward_adoption(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
) -> CampaignReport:
    """Select per fold on train CE, then realize chosen-versus-baseline wealth on test.

    The runner is called once per arm per phase (baseline then candidate on train;
    baseline, candidate, and chosen on test — chosen re-runs even when it repeats
    another arm); ``spec`` is never mutated.

    Raises:
        ValueError: When train/test months are absent, the candidate count is not
            one, or no walk-forward fold fits the window.
    """
    if spec.train_months is None or spec.test_months is None:
        raise ValueError("walk-forward adoption requires both train_months and test_months")
    if len(spec.candidates) != 1:
        raise ValueError(f"expected exactly one candidate, got {len(spec.candidates)}")
    candidate = spec.candidates[0]
    windows = walk_forward_windows(
        spec.start,
        spec.end,
        train_months=spec.train_months,
        test_months=spec.test_months,
    )
    if not windows:
        raise ValueError("no walk-forward folds fit the experiment window")

    def real_wealth(policy: PolicyId, start: date, end: date) -> float:
        return runner(_arm_config(spec, policy, start, end)).terminal_wealth_real_krw

    folds: list[FoldOutcome] = []
    baseline_wealths: list[float] = []
    candidate_wealths: list[float] = []
    chosen_wealths: list[float] = []
    for train_start, train_end, test_start, test_end in windows:
        baseline_train = real_wealth(spec.baseline.policy, train_start, train_end)
        candidate_train = real_wealth(candidate.policy, train_start, train_end)
        train_adopted = adoption_passes(
            _singleton_ce(candidate_train),
            _singleton_ce(baseline_train),
            delta0=spec.delta0,
            modules=candidate.modules,
        )
        chosen_policy = candidate.policy if train_adopted else spec.baseline.policy
        baseline_test = real_wealth(spec.baseline.policy, test_start, test_end)
        candidate_test = real_wealth(candidate.policy, test_start, test_end)
        chosen_test = real_wealth(chosen_policy, test_start, test_end)
        folds.append(
            FoldOutcome(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_adopted=train_adopted,
                chosen_policy=chosen_policy,
                baseline_test_wealth=baseline_test,
                candidate_test_wealth=candidate_test,
                chosen_test_wealth=chosen_test,
            )
        )
        baseline_wealths.append(baseline_test)
        candidate_wealths.append(candidate_test)
        chosen_wealths.append(chosen_test)

    baseline_ce = {gamma: certainty_equivalent(baseline_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    candidate_ce = {gamma: certainty_equivalent(candidate_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    chosen_ce = {gamma: certainty_equivalent(chosen_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    process_adopted_vs_baseline = adoption_passes(
        chosen_ce, baseline_ce, delta0=spec.delta0, modules=candidate.modules
    )
    return CampaignReport(
        name=spec.name,
        candidate_id=candidate.id,
        modules=candidate.modules,
        folds=tuple(folds),
        baseline_test_ce=baseline_ce,
        candidate_test_ce=candidate_ce,
        chosen_test_ce=chosen_ce,
        process_adopted_vs_baseline=process_adopted_vs_baseline,
    )


def write_campaign_report(report: CampaignReport, settings: DataSettings, experiment_id: str) -> Path:
    """Persist the campaign verdict under ``experiments/{name}_{experiment_id}.json``.

    Returns:
        Path: The written UTF-8 JSON artifact.
    """
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "process_adopted_vs_baseline": report.process_adopted_vs_baseline,
        "fold_count": len(report.folds),
        "folds": [
            {
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "train_adopted": fold.train_adopted,
                "chosen_policy": str(fold.chosen_policy),
                "baseline_test_wealth": fold.baseline_test_wealth,
                "candidate_test_wealth": fold.candidate_test_wealth,
                "chosen_test_wealth": fold.chosen_test_wealth,
            }
            for fold in report.folds
        ],
    }
    experiments_dir = settings.resolved_data_root() / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    out_path = experiments_dir / f"{report.name}_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
