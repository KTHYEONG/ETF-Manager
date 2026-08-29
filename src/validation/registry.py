"""I12 experiment identity records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig
    from src.validation.ablation import AblationReport
    from src.validation.experiment import ExperimentSpec

from src.validation.prospective import ProspectiveFreezeRecord

__all__ = [
    "ArmOutcome",
    "ExperimentRecord",
    "ExperimentRunRecord",
    "ProspectiveFreezeRecord",
    "build_ablation_arm_outcomes",
    "freeze_baseline_config_hash",
    "make_experiment",
    "write_ablation_run_record",
    "write_prospective_freeze_record",
]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Reproducibility stamps for one validation run."""

    experiment_id: str
    config_hash: str
    manifest_hash: str
    git_commit: str
    seed: int | None
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """One ablation arm with CE ratios at gamma 2/5/10."""

    arm_id: str
    policy: str
    modules: int
    adopted: bool
    ce_ratio_gamma_2: float
    ce_ratio_gamma_5: float
    ce_ratio_gamma_10: float


@dataclass(frozen=True, slots=True)
class ExperimentRunRecord:
    """Ablation run record with every arm outcome."""

    experiment_id: str
    experiment_name: str
    thesis_id: str | None
    baseline_config_hash: str
    arms: tuple[ArmOutcome, ...]
    manifest_hash: str
    git_commit: str


def _identity_payload(
    config: AllocationConfig,
    thesis_id: str | None,
) -> dict[str, object]:
    if config.targets_override is None:
        targets_override_payload = None
    else:
        targets_override_payload = {k: float(v) for k, v in sorted(config.targets_override.items())}
    return {
        "commission_bps": float(config.commission_bps),
        "end": config.end.isoformat(),
        "fill_delay_sessions": int(config.fill_delay_sessions),
        "fx_spread_bps": float(config.fx_spread_bps),
        "has_currency": config.currency is not None,
        "has_mapping": config.mapping is not None,
        "has_overlay": config.overlay is not None,
        "has_tilt": config.tilt is not None,
        "monthly_contribution_krw": float(config.monthly_contribution_krw),
        "policy": str(config.policy),
        "start": config.start.isoformat(),
        "targets_override": targets_override_payload,
        "thesis_id": thesis_id,
    }


def make_experiment(
    *,
    config: AllocationConfig,
    manifest_hash: str,
    git_commit: str,
    seed: int | None,
    metrics: Mapping[str, float],
    thesis_id: str | None = None,
) -> ExperimentRecord:
    """Build a hashed experiment record; empty lineage fields fail closed.

    ``config_hash`` digests a canonical JSON payload of the allocation identity —
    policy, window, cashflow and cost parameters, plus presence flags for the
    optional strategy objects — never object reprs. ``experiment_id`` binds the
    config hash with manifest lineage and the repr of the seed.
    ``thesis_id`` is included as string or null in the identity payload when provided
    or when ``config`` carries a ``thesis_id`` attribute.

    Raises:
        ValueError: When ``git_commit`` or ``manifest_hash`` is empty.
    """
    if not manifest_hash:
        raise ValueError("manifest_hash must not be empty")
    if not git_commit:
        raise ValueError("git_commit must not be empty")
    effective_thesis_id = thesis_id
    if effective_thesis_id is None:
        effective_thesis_id = getattr(config, "thesis_id", None)
        if effective_thesis_id is not None:
            try:
                effective_thesis_id = effective_thesis_id.value
            except AttributeError:
                effective_thesis_id = str(effective_thesis_id)
    identity_payload = _identity_payload(config, effective_thesis_id)
    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    lineage = f"{config_hash}|{manifest_hash}|{git_commit}|{seed!r}"
    return ExperimentRecord(
        experiment_id=hashlib.sha256(lineage.encode("utf-8")).hexdigest()[:16],
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        git_commit=git_commit,
        seed=seed,
        metrics=dict(metrics),
    )


def freeze_baseline_config_hash(spec: ExperimentSpec) -> str:
    """Return SHA-256 of canonical baseline AllocationConfig identity."""
    from src.sim.allocation import AllocationConfig

    thesis_id_str = spec.thesis_id.value if spec.thesis_id is not None else None
    # Baseline arm mirrors ablation _arm_config: no overlay/reserve/mapping/currency/tilt
    targets = None
    if spec.baseline.targets is not None:
        targets = dict(spec.baseline.targets)
    config = AllocationConfig(
        policy=spec.baseline.policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=spec.contribution_krw,
        fill_delay_sessions=1,
        fx_spread_bps=float(spec.fx_spread_bps),
        commission_bps=float(spec.commission_bps),
        tilt=None,
        rebalance_band=None,
        overlay=None,
        reserve=None,
        currency=None,
        mapping=None,
        contribution_shape=None,
        kafi_deployment=None,
        cadence="monthly",
        targets_override=targets,
    )
    payload = _identity_payload(config, thesis_id_str)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_ablation_arm_outcomes(report: AblationReport) -> tuple[ArmOutcome, ...]:
    """Emit one ArmOutcome per AblationReport row with CE ratios."""
    return tuple(
        ArmOutcome(
            arm_id=row.candidate_id,
            policy=str(row.policy),
            modules=int(row.modules),
            adopted=bool(row.adopted),
            ce_ratio_gamma_2=float(row.ce_ratio[2.0]),
            ce_ratio_gamma_5=float(row.ce_ratio[5.0]),
            ce_ratio_gamma_10=float(row.ce_ratio[10.0]),
        )
        for row in report.rows
    )


def write_ablation_run_record(
    *,
    spec: ExperimentSpec,
    report: AblationReport,
    record: ExperimentRecord,
    settings: DataSettings,
) -> Path:
    """Persist ablation run JSON under data/experiments/{name}_ablation_{experiment_id}.json."""
    arms = build_ablation_arm_outcomes(report)
    thesis_id_str = spec.thesis_id.value if spec.thesis_id is not None else None
    baseline_hash = freeze_baseline_config_hash(spec)
    payload = {
        "experiment_id": record.experiment_id,
        "experiment_name": spec.name,
        "thesis_id": thesis_id_str,
        "baseline_config_hash": baseline_hash,
        "arms": [
            {
                "arm_id": a.arm_id,
                "policy": a.policy,
                "modules": a.modules,
                "adopted": a.adopted,
                "ce_ratio_gamma_2": a.ce_ratio_gamma_2,
                "ce_ratio_gamma_5": a.ce_ratio_gamma_5,
                "ce_ratio_gamma_10": a.ce_ratio_gamma_10,
            }
            for a in arms
        ],
        "manifest_hash": record.manifest_hash,
        "git_commit": record.git_commit,
    }
    from src.data.paths import experiments_dir

    out_dir = experiments_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}_ablation_{record.experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_prospective_freeze_record(
    *,
    spec: ExperimentSpec,
    freeze: ProspectiveFreezeRecord,
    settings: DataSettings,
) -> Path:
    """Persist prospective freeze record under data/experiments."""
    from src.data.paths import experiments_dir

    payload = {
        "thesis_id": freeze.thesis_id,
        "experiment_name": freeze.experiment_name,
        "frozen_at": freeze.frozen_at.isoformat(),
        "targets_hash": freeze.targets_hash,
        "spec_name": spec.name,
        "baseline_config_hash": freeze_baseline_config_hash(spec),
    }
    out_dir = experiments_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = freeze.frozen_at.isoformat().replace(":", "-")
    out_path = out_dir / f"{spec.name}_prospective_{safe_ts}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
