"""I12 experiment identity records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.etf_manager.sim.allocation import AllocationConfig

__all__ = ["ExperimentRecord", "make_experiment"]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Reproducibility stamps for one validation run."""

    experiment_id: str
    config_hash: str
    manifest_hash: str
    git_commit: str
    seed: int | None
    metrics: Mapping[str, float]


def make_experiment(
    *,
    config: AllocationConfig,
    manifest_hash: str,
    git_commit: str,
    seed: int | None,
    metrics: Mapping[str, float],
) -> ExperimentRecord:
    """Build a hashed experiment record; empty lineage fields fail closed.

    ``config_hash`` digests a canonical JSON payload of the allocation identity —
    policy, window, cashflow and cost parameters, plus presence flags for the
    optional strategy objects — never object reprs. ``experiment_id`` binds the
    config hash with manifest lineage and the repr of the seed.

    Raises:
        ValueError: When ``git_commit`` or ``manifest_hash`` is empty.
    """
    if not manifest_hash:
        raise ValueError("manifest_hash must not be empty")
    if not git_commit:
        raise ValueError("git_commit must not be empty")
    identity_payload = {
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
    }
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
