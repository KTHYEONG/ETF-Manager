"""Prospective monitoring registry (PROSPECTIVE_2026_V1)."""

# ruff: noqa: SIM108,SIM102,SIM103,PLR0913

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.settings import DataSettings
from src.policy.targets import OPERATIONAL_TARGETS_OVERRIDE, PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.research_posture import SEEN_HISTORY_CUTOFF, ObjectiveFamily, assert_prospective_observation

logger = logging.getLogger(__name__)

PROSPECTIVE_BUNDLE_ID: Final[str] = "PROSPECTIVE_2026_V1"


class ProspectiveArmRole(StrEnum):
    IMMUTABLE_BENCHMARK = "immutable_benchmark"
    PROVISIONAL_INCUMBENT = "provisional_incumbent"
    DEPLOYMENT_TIMING = "deployment_timing"


class FrozenStrategyArm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str = Field(min_length=1)
    policy: PolicyId
    targets: dict[str, float]
    role: ProspectiveArmRole | None = None
    adaptive_contribution: Any | None = None
    kafi_deployment: dict[str, Any] | None = None
    identity_hash: str | None = None
    objective_family: ObjectiveFamily | str | None = None

    @field_validator("policy", mode="before")
    @classmethod
    def _coerce_policy(cls, value: object) -> object:
        if isinstance(value, PolicyId):
            return value
        try:
            return PolicyId.parse(value)
        except ValueError as exc:
            raise ValueError(f"unknown policy {value!r}") from exc

    @field_validator("targets", mode="before")
    @classmethod
    def _validate_targets(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError(f"targets must be a mapping, got {type(value).__name__!r}")
        if len(value) == 0:
            raise ValueError("targets must be nonempty when set")
        normalized: dict[str, float] = {}
        for raw_key, raw_weight in value.items():
            key = str(raw_key).strip().upper()
            if not key:
                raise ValueError("targets ticker must be non-blank")
            if key in normalized:
                raise ValueError(f"duplicate targets ticker after normalize: {key!r}")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"targets[{key!r}] weight must be finite, got {raw_weight!r}") from exc
            import math

            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"targets[{key!r}] must be finite nonnegative, got {raw_weight!r}")
            normalized[key] = weight
        import math

        total = sum(normalized.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
            raise ValueError(f"targets weights must sum to 1.0 within 1e-6, got {total!r}")
        return normalized

    @field_validator("adaptive_contribution", mode="before")
    @classmethod
    def _check_adaptive(cls, value: object) -> object:
        if value is None:
            return None
        # reject any non-null adaptive_contribution (empty dict also forbidden)
        raise ValueError("adaptive_contribution not allowed for prospective arms")

    @model_validator(mode="after")
    def _check_kafi_role(self) -> FrozenStrategyArm:
        if self.role is not None and self.role is not ProspectiveArmRole.DEPLOYMENT_TIMING and self.kafi_deployment is not None:
            raise ValueError(f"kafi_deployment only allowed for deployment_timing arm, got role {self.role!r}")
        return self


class ProspectiveBundleSpec(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    bundle_id: str = Field(min_length=1)
    seen_history_cutoff: date = Field(default=SEEN_HISTORY_CUTOFF)
    frozen_at: datetime | None = None
    arms: tuple[FrozenStrategyArm, ...] = Field(min_length=1)
    contribution_krw: float | None = None
    policy: PolicyId | str | None = None

    @field_validator("bundle_id", mode="before")
    @classmethod
    def _coerce_bundle(cls, value: object) -> object:
        return str(value)

    @field_validator("seen_history_cutoff", mode="before")
    @classmethod
    def _coerce_cutoff(cls, value: object) -> object:
        if value is None:
            return SEEN_HISTORY_CUTOFF
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value

    @field_validator("frozen_at", mode="before")
    @classmethod
    def _coerce_frozen(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"invalid frozen_at {value!r}") from exc
        return value


def _arm_identity_payload(*, policy: PolicyId | str, targets: dict[str, float] | None, has_kafi: bool) -> dict[str, object]:
    if targets is None:
        payload_targets = None
    else:
        payload_targets = {k: float(v) for k, v in sorted(targets.items())}
    return {
        "policy": str(policy) if isinstance(policy, PolicyId) else str(PolicyId.parse(policy)),
        "targets_override": payload_targets,
        "has_kafi_deployment": bool(has_kafi),
        "has_adaptive": False,
    }


def _hash_arm_payload(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def strategy_arm_identity_hash(config: AllocationConfig) -> str:
    targets = config.targets_override
    if targets is not None:
        # normalize to uppercase keys for determinism
        norm = {str(k).strip().upper(): float(v) for k, v in targets.items()}
    else:
        norm = None
    has_kafi = config.kafi_deployment is not None
    # ensure adaptive not set; if set, identity would diverge but we still hash with has_adaptive True
    has_adaptive = config.adaptive_contribution is not None
    payload: dict[str, object]
    if norm is None:
        payload_targets = None
    else:
        payload_targets = {k: float(v) for k, v in sorted(norm.items())}
    payload = {
        "policy": str(config.policy),
        "targets_override": payload_targets,
        "has_kafi_deployment": bool(has_kafi),
        "has_adaptive": bool(has_adaptive),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_strategy_identity_unchanged(frozen: FrozenStrategyArm, config: AllocationConfig) -> None:
    expected = frozen.identity_hash
    if expected is None:
        raise ValueError("frozen arm has no identity_hash")
    actual = strategy_arm_identity_hash(config)
    if actual != expected:
        raise ValueError(f"strategy identity mismatch: expected {expected!r} got {actual!r} (targets/policy/kafi drift)")


def load_prospective_bundle(path: Path) -> ProspectiveBundleSpec:
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    # Validate via model
    try:
        spec = ProspectiveBundleSpec.model_validate(payload)
    except Exception as exc:
        # surface adaptive_contribution error message if present
        if "adaptive_contribution" in str(exc):
            raise ValueError(f"adaptive_contribution not allowed: {exc}") from exc
        raise
    # I23: reject adaptive_contribution on any arm (already via validator, but also raw check)
    raw_arms = payload.get("arms", [])
    for arm_raw in raw_arms if isinstance(raw_arms, list) else []:
        if isinstance(arm_raw, dict) and arm_raw.get("adaptive_contribution") is not None:
            raise ValueError("adaptive_contribution not allowed for prospective arms")
    # validate roles unique
    roles = [arm.role for arm in spec.arms if arm.role is not None]
    if len(roles) != len(set(roles)):
        raise ValueError(f"prospective roles must be unique, got {roles!r}")
    # validate expected roles present
    role_set = set(roles)
    # Must contain all three roles
    expected_roles = {
        ProspectiveArmRole.IMMUTABLE_BENCHMARK,
        ProspectiveArmRole.PROVISIONAL_INCUMBENT,
        ProspectiveArmRole.DEPLOYMENT_TIMING,
    }
    if role_set and role_set != expected_roles:
        # If any role missing or extra, fail closed only when fully specified; for frozen files role set may be complete
        # Enforce that no duplicate and that each present role is valid — strict if 3 arms
        if len(spec.arms) == 3 and role_set != expected_roles:
            raise ValueError(f"expected roles {expected_roles!r}, got {role_set!r}")
    # benchmark QQQ 1.0
    for arm in spec.arms:
        if arm.role is ProspectiveArmRole.IMMUTABLE_BENCHMARK:
            if arm.targets != {"QQQ": 1.0}:
                raise ValueError(f"benchmark targets must be {{'QQQ': 1.0}}, got {arm.targets!r}")
        if arm.role is ProspectiveArmRole.PROVISIONAL_INCUMBENT:
            expected = OPERATIONAL_TARGETS_OVERRIDE
            # need normalized compare
            norm_expected = {str(k).strip().upper(): float(v) for k, v in expected.items()}
            if arm.targets != norm_expected:
                raise ValueError(f"incumbent targets must match {expected!r}, got {arm.targets!r}")
        if arm.role is ProspectiveArmRole.DEPLOYMENT_TIMING:
            expected_inc = OPERATIONAL_TARGETS_OVERRIDE
            norm_expected = {str(k).strip().upper(): float(v) for k, v in expected_inc.items()}
            if arm.targets != norm_expected:
                raise ValueError(f"deployment_timing targets must match {expected_inc!r}, got {arm.targets!r}")
            if arm.kafi_deployment is None:
                raise ValueError("deployment_timing arm requires kafi_deployment")
        # objective_family validation per arm
        if arm.objective_family is not None:
            # normalize
            try:
                fam = arm.objective_family if isinstance(arm.objective_family, ObjectiveFamily) else ObjectiveFamily(str(arm.objective_family))
            except ValueError as exc:
                raise ValueError(f"unknown objective_family {arm.objective_family!r}") from exc
            if arm.role is ProspectiveArmRole.DEPLOYMENT_TIMING and fam is not ObjectiveFamily.DEPLOYMENT_TIMING:
                raise ValueError(f"deployment_timing arm objective_family must be deployment_timing, got {fam!r}")
            if arm.role in (ProspectiveArmRole.IMMUTABLE_BENCHMARK, ProspectiveArmRole.PROVISIONAL_INCUMBENT) and fam is not ObjectiveFamily.CAPITAL_ALLOCATION:
                raise ValueError(f"capital_allocation arm objective_family must be capital_allocation, got {fam!r}")
    # Also reject non-null adaptive_contribution per arm already validated
    for arm in spec.arms:
        if arm.adaptive_contribution is not None:
            raise ValueError("adaptive_contribution not allowed for prospective arms")
    # bundle_id check
    if spec.bundle_id != PROSPECTIVE_BUNDLE_ID:
        # allow frozen files to keep same id, but raise if mismatch?
        # Keep strict: must be PROSPECTIVE_2026_V1
        if spec.bundle_id != PROSPECTIVE_BUNDLE_ID:
            raise ValueError(f"bundle_id must be {PROSPECTIVE_BUNDLE_ID!r}, got {spec.bundle_id!r}")
    return spec


@dataclass(frozen=True, slots=True)
class ProspectiveFreezeRecord:
    bundle_id: str
    frozen_at: datetime
    arm_hashes: tuple[str, ...]
    git_commit: str
    # compatibility with legacy fields
    thesis_id: str = PROSPECTIVE_BUNDLE_ID
    targets_hash: str = ""
    experiment_name: str = PROSPECTIVE_BUNDLE_ID


def freeze_prospective_bundle(*, bundle_path: Path, output_dir: Path, frozen_at: datetime, git_commit: str, settings: DataSettings) -> ProspectiveFreezeRecord:
    _ = settings
    bundle = load_prospective_bundle(bundle_path)
    arm_hashes: list[str] = []
    for arm in bundle.arms:
        has_kafi = arm.kafi_deployment is not None
        payload = _arm_identity_payload(policy=arm.policy, targets=arm.targets, has_kafi=has_kafi)
        h = _hash_arm_payload(payload)
        arm_hashes.append(h)
    hashes_tuple = tuple(arm_hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "prospective_2026_v1_frozen.json"
    # Build payload from original file plus frozen metadata and identity hashes
    raw = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    # update frozen_at and git_commit
    raw["frozen_at"] = frozen_at.isoformat()
    raw["git_commit"] = git_commit
    raw["bundle_id"] = bundle.bundle_id
    # inject identity_hash per arm
    if isinstance(raw.get("arms"), list):
        for idx, arm_dict in enumerate(raw["arms"]):
            if isinstance(arm_dict, dict) and idx < len(hashes_tuple):
                arm_dict["identity_hash"] = hashes_tuple[idx]
    # also ensure seen_history_cutoff stays
    if "seen_history_cutoff" not in raw:
        raw["seen_history_cutoff"] = bundle.seen_history_cutoff.isoformat()
    out_path.write_text(json.dumps(raw, indent=2, sort_keys=False), encoding="utf-8")
    logger.info("[DATA] event=prospective_freeze bundle=%s frozen_at=%s arms=%d", bundle.bundle_id, frozen_at.isoformat(), len(hashes_tuple))
    # first hash as targets_hash for legacy compatibility
    first_hash = hashes_tuple[0] if hashes_tuple else ""
    return ProspectiveFreezeRecord(
        bundle_id=bundle.bundle_id,
        frozen_at=frozen_at,
        arm_hashes=hashes_tuple,
        git_commit=git_commit,
        thesis_id=bundle.bundle_id,
        targets_hash=first_hash,
        experiment_name=bundle.bundle_id,
    )


@dataclass(frozen=True, slots=True)
class ProspectiveObservation:
    as_of: date
    arm_id: str
    policy: PolicyId
    targets: dict[str, float]
    terminal_wealth_krw: float
    terminal_wealth_real_krw: float
    xirr: float
    xirr_real: float
    max_drawdown: float
    total_contribution_real_krw: float


@dataclass(frozen=True, slots=True)
class ProspectiveMonitorReport:
    as_of: date
    bundle_id: str
    observations: tuple[ProspectiveObservation, ...]
    registry_path: Path


def _prev_month_start(as_of: date) -> date:
    # window [as_of-1month, as_of] via 30-day approx; calendar month subtraction alternative
    # use 30 days for determinism; tests use 2026-09-30 -> 2026-08-31
    return as_of - timedelta(days=30)


def _allocation_end_within_as_of(*, start: date, as_of: date, fill_delay_sessions: int = 1) -> date:
    """Cap allocation end so monthly fills never execute after the observation date."""
    from src.data.calendar import load_calendar
    from src.data.schedule import build_decision_schedule

    calendar = load_calendar()
    end = as_of
    while True:
        schedule = build_decision_schedule(
            start,
            end,
            frequency="monthly",
            fill_delay_sessions=fill_delay_sessions,
        )
        if not schedule or schedule[-1].execution_session <= as_of:
            return end
        last_signal = schedule[-1].signal_session
        prior_sessions = [session for session in calendar.sessions(start, last_signal) if session < last_signal]
        if not prior_sessions:
            return start
        end = prior_sessions[-1]


def run_prospective_monitor(*, bundle: ProspectiveBundleSpec, as_of: date, runner: Callable[[AllocationConfig], AllocationResult], settings: DataSettings, registry_dir: Path | None = None) -> ProspectiveMonitorReport:
    assert_prospective_observation(as_of)
    # Determine window; cap end so catalog need not cover sessions after as_of.
    start = _prev_month_start(as_of)
    end = _allocation_end_within_as_of(start=start, as_of=as_of, fill_delay_sessions=1)
    # Contribution comes from bundle or default 1M
    contribution = float(bundle.contribution_krw) if bundle.contribution_krw is not None else 1_000_000.0
    observations: list[ProspectiveObservation] = []
    for arm in bundle.arms:
        targets = dict(arm.targets) if arm.targets is not None else None
        # Build AllocationConfig
        # Resolve kafi_deployment if present
        kafi_cfg = None
        if arm.kafi_deployment is not None:
            from src.policy.kafi_deployment import KafiDeploymentConfig

            # kafi_deployment dict may have keys matching KafiDeploymentConfig fields
            try:
                kafi_cfg = KafiDeploymentConfig(**arm.kafi_deployment)
            except Exception as exc:
                raise ValueError(f"invalid kafi_deployment for arm {arm.arm_id!r}: {exc}") from exc
        cfg = AllocationConfig(
            policy=arm.policy if isinstance(arm.policy, PolicyId) else PolicyId.parse(arm.policy),
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution),
            fill_delay_sessions=1,
            commission_bps=0.0,
            fx_spread_bps=0.0,
            targets_override=targets,
            kafi_deployment=kafi_cfg,
        )
        # assert identity unchanged: compare arm hash vs config hash
        # compute expected hash from arm
        has_kafi = arm.kafi_deployment is not None
        expected_payload = _arm_identity_payload(policy=arm.policy, targets=arm.targets, has_kafi=has_kafi)
        expected_hash = _hash_arm_payload(expected_payload)
        # If arm has identity_hash (frozen bundle), verify it matches expected (fail closed if tampered)
        if arm.identity_hash is not None and arm.identity_hash != expected_hash:
            raise ValueError(f"frozen identity_hash mismatch for {arm.arm_id!r}: expected {expected_hash!r} got {arm.identity_hash!r}")
        actual_hash = strategy_arm_identity_hash(cfg)
        if actual_hash != expected_hash:
            raise ValueError(f"strategy identity mismatch for {arm.arm_id!r}: arm targets/policy/kafi drift (identity|targets)")
        # also enforce extra forbid for FrozenStrategyArm vs config: if frozen has identity_hash, use that for assert
        if arm.identity_hash is not None:
            frozen_for_assert = arm
        else:
            # create temporary frozen with same identity_hash as expected
            frozen_for_assert = FrozenStrategyArm(
                arm_id=arm.arm_id,
                policy=arm.policy,
                targets=arm.targets,
                role=arm.role,
                adaptive_contribution=None,
                kafi_deployment=arm.kafi_deployment,
                identity_hash=expected_hash,
                objective_family=arm.objective_family,
            )
        assert_strategy_identity_unchanged(frozen_for_assert, cfg)
        result = runner(cfg)
        obs = ProspectiveObservation(
            as_of=as_of,
            arm_id=arm.arm_id,
            policy=cfg.policy,
            targets=dict(targets) if targets is not None else {},
            terminal_wealth_krw=float(result.terminal_wealth_krw),
            terminal_wealth_real_krw=float(result.terminal_wealth_real_krw),
            xirr=float(result.xirr),
            xirr_real=float(result.xirr_real),
            max_drawdown=float(result.max_drawdown),
            total_contribution_real_krw=float(result.total_contribution_real_krw),
        )
        observations.append(obs)
    # Determine registry path
    if registry_dir is None:
        base = settings.resolved_data_root() / "prospective_registry"
    else:
        base = Path(registry_dir)
    base.mkdir(parents=True, exist_ok=True)
    registry_path = base / "prospective_observations.jsonl"
    # append observations as JSONL
    with registry_path.open("a", encoding="utf-8") as fh:
        for obs in observations:
            rec = {
                "as_of": obs.as_of.isoformat(),
                "arm_id": obs.arm_id,
                "policy": str(obs.policy),
                "targets": obs.targets,
                "terminal_wealth_krw": obs.terminal_wealth_krw,
                "terminal_wealth_real_krw": obs.terminal_wealth_real_krw,
                "xirr": obs.xirr,
                "xirr_real": obs.xirr_real,
                "max_drawdown": obs.max_drawdown,
                "total_contribution_real_krw": obs.total_contribution_real_krw,
                "bundle_id": bundle.bundle_id,
            }
            fh.write(json.dumps(rec) + "\n")
    logger.info("[DATA] event=prospective_monitor as_of=%s arms=%d registry=%s", as_of.isoformat(), len(observations), registry_path.as_posix())
    return ProspectiveMonitorReport(as_of=as_of, bundle_id=bundle.bundle_id, observations=tuple(observations), registry_path=registry_path)

