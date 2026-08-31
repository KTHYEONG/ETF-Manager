"""Prospective monitoring registry (PROSPECTIVE_2026_V1)."""

# ruff: noqa: SIM108,SIM102,SIM103,PLR0913,N814,B009

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.settings import DataSettings
from src.policy.targets import PolicyId
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


CURRENT_OPERATIONAL_BUNDLE_PATH: Final[Path] = Path("configs/prospective/CURRENT_OPERATIONAL_BUNDLE.json")


class ProspectiveBundleSpec(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    bundle_id: str = Field(min_length=1)
    seen_history_cutoff: date = Field(default=SEEN_HISTORY_CUTOFF)
    prospective_start: date = Field(default=date(2026, 9, 1))
    frozen_at: datetime | None = None
    git_commit: str | None = None
    bundle_hash: str | None = None
    behavior_preserving_migration: bool = False
    approved_runtime_commits: tuple[str, ...] = Field(default_factory=tuple)
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

    @field_validator("prospective_start", mode="before")
    @classmethod
    def _coerce_prospective_start(cls, value: object) -> object:
        if value is None:
            return date(2026, 9, 1)
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value

    @field_validator("approved_runtime_commits", mode="before")
    @classmethod
    def _coerce_approved_commits(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        raise ValueError(f"approved_runtime_commits must be a list, got {type(value).__name__!r}")

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


def _normalize_targets(targets: dict[str, float] | None) -> dict[str, float] | None:
    if targets is None:
        return None
    norm = {str(k).strip().upper(): float(v) for k, v in targets.items()}
    return {k: float(v) for k, v in sorted(norm.items())}


def _kafi_payload_dict(kafi: object | None) -> dict[str, object] | None:
    if kafi is None:
        return None
    if isinstance(kafi, dict):
        # from FrozenStrategyArm
        return {k: kafi[k] for k in sorted(kafi.keys())}
    # KafiDeploymentConfig object
    try:
        return {
            "bond_ticker": str(getattr(kafi, "bond_ticker")),
            "credit_series_id": str(getattr(kafi, "credit_series_id")),
            "equity_ticker": str(getattr(kafi, "equity_ticker")),
            "max_multiplier": float(getattr(kafi, "max_multiplier")),
            "min_multiplier": float(getattr(kafi, "min_multiplier")),
            "rank_window": int(getattr(kafi, "rank_window")),
        }
    except Exception:
        return {"value": str(kafi)}


def build_full_arm_identity_payload(
    config: AllocationConfig,
    *,
    bundle: ProspectiveBundleSpec | None = None,
    arm: FrozenStrategyArm | None = None,
) -> dict[str, object]:
    payload_targets = _normalize_targets(
        dict(config.targets_override) if config.targets_override is not None else None
    )
    kafi_dict = _kafi_payload_dict(config.kafi_deployment)
    if kafi_dict is None and arm is not None and arm.kafi_deployment is not None:
        kafi_dict = {k: arm.kafi_deployment[k] for k in sorted(arm.kafi_deployment.keys())}
    has_kafi = kafi_dict is not None
    obj_family: object | None = None
    if arm is not None and arm.objective_family is not None:
        obj_family = str(arm.objective_family.value if hasattr(arm.objective_family, "value") else arm.objective_family)
    elif bundle is not None and arm is None:
        obj_family = None
    payload: dict[str, object] = {
        "policy": str(config.policy),
        "targets_override": payload_targets,
        "monthly_contribution_krw": float(config.monthly_contribution_krw),
        "cadence": str(config.cadence),
        "fill_delay_sessions": int(config.fill_delay_sessions),
        "commission_bps": float(config.commission_bps),
        "fx_spread_bps": float(config.fx_spread_bps),
        "has_kafi_deployment": bool(has_kafi),
        "kafi_deployment": kafi_dict,
        "has_adaptive": bool(config.adaptive_contribution is not None),
    }
    if obj_family is not None:
        payload["objective_family"] = obj_family
    if bundle is not None:
        payload["prospective_start"] = bundle.prospective_start.isoformat()
        payload["seen_history_cutoff"] = bundle.seen_history_cutoff.isoformat()
        payload["bundle_id"] = bundle.bundle_id
    if arm is not None:
        payload["arm_id"] = arm.arm_id
        if arm.role is not None:
            payload["role"] = str(arm.role.value if hasattr(arm.role, "value") else arm.role)
    return payload


def strategy_arm_identity_hash(
    config: AllocationConfig,
    *,
    bundle: ProspectiveBundleSpec | None = None,
    arm: FrozenStrategyArm | None = None,
) -> str:
    payload = build_full_arm_identity_payload(config, bundle=bundle, arm=arm)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bundle_identity_hash(bundle: ProspectiveBundleSpec, arm_hashes: Sequence[str]) -> str:
    sorted_hashes = tuple(sorted(arm_hashes))
    payload = {
        "bundle_id": bundle.bundle_id,
        "prospective_start": bundle.prospective_start.isoformat(),
        "seen_history_cutoff": bundle.seen_history_cutoff.isoformat(),
        "arm_hashes": list(sorted_hashes),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_runtime_engine_commit(
    *,
    frozen_git_commit: str,
    runtime_git_commit: str,
    behavior_preserving_migration: bool = False,
    approved_runtime_commits: Sequence[str] = (),
) -> None:
    if not runtime_git_commit:
        raise ValueError("runtime_git_commit must not be empty")
    if frozen_git_commit == runtime_git_commit:
        return
    if behavior_preserving_migration and runtime_git_commit in approved_runtime_commits:
        return
    raise ValueError(
        f"runtime_engine_commit mismatch: frozen {frozen_git_commit!r} vs runtime {runtime_git_commit!r}"
    )


def assert_strategy_identity_unchanged(frozen: FrozenStrategyArm, config: AllocationConfig) -> None:
    expected = frozen.identity_hash
    if expected is None:
        raise ValueError("frozen arm has no identity_hash")
    actual = strategy_arm_identity_hash(config, arm=frozen)
    # also compare with config-only hash to ensure policy/targets/kafi drift detected
    # allow arm-bound payload to be richer; but verify config-only hash matches when arm bundle unknown
    if actual != expected:
        # fallback: try config-only hash for legacy bundles
        legacy = strategy_arm_identity_hash(config)
        if legacy != expected and actual != expected:
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
        if arm.role is ProspectiveArmRole.DEPLOYMENT_TIMING:
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
        # Build a temporary config to compute full identity (contribution/cadence defaults)
        from src.policy.kafi_deployment import KafiDeploymentConfig as _KDC
        from src.policy.targets import PolicyId as _Pid
        from src.sim.allocation import AllocationConfig as _AC

        kafi_cfg = None
        if arm.kafi_deployment is not None:
            try:
                kafi_cfg = _KDC(**arm.kafi_deployment)
            except Exception:
                kafi_cfg = None
        cfg = _AC(
            policy=arm.policy if isinstance(arm.policy, _Pid) else _Pid.parse(arm.policy),
            start=bundle.prospective_start,
            end=bundle.prospective_start,
            monthly_contribution_krw=float(bundle.contribution_krw) if bundle.contribution_krw is not None else 1_000_000.0,
            fill_delay_sessions=1,
            commission_bps=0.0,
            fx_spread_bps=0.0,
            targets_override=dict(arm.targets),
            kafi_deployment=kafi_cfg,
        )
        h = strategy_arm_identity_hash(cfg, bundle=bundle, arm=arm)
        arm_hashes.append(h)
    hashes_tuple = tuple(arm_hashes)
    bundle_hash_val = bundle_identity_hash(bundle, hashes_tuple)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "prospective_2026_v1_frozen.json"
    raw = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    raw["frozen_at"] = frozen_at.isoformat()
    raw["git_commit"] = git_commit
    raw["bundle_hash"] = bundle_hash_val
    raw["bundle_id"] = bundle.bundle_id
    raw["prospective_start"] = bundle.prospective_start.isoformat()
    raw["seen_history_cutoff"] = bundle.seen_history_cutoff.isoformat()
    if isinstance(raw.get("arms"), list):
        for idx, arm_dict in enumerate(raw["arms"]):
            if isinstance(arm_dict, dict) and idx < len(hashes_tuple):
                arm_dict["identity_hash"] = hashes_tuple[idx]
    out_path.write_text(json.dumps(raw, indent=2, sort_keys=False), encoding="utf-8")
    logger.info("[DATA] event=prospective_freeze bundle=%s frozen_at=%s arms=%d bundle_hash=%s", bundle.bundle_id, frozen_at.isoformat(), len(hashes_tuple), bundle_hash_val)
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
    cumulative_terminal_real_krw: float = 0.0
    cumulative_real_gain: float = 0.0
    cumulative_xirr_real: float = 0.0
    cumulative_ratio_vs_benchmark: float = 1.0
    reserve_krw: float = 0.0
    frozen_engine_commit: str = ""
    runtime_engine_commit: str = ""
    bundle_hash: str = ""


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


def run_prospective_monitor(
    *,
    bundle: ProspectiveBundleSpec,
    as_of: date,
    runner: Callable[[AllocationConfig], AllocationResult],
    settings: DataSettings,
    registry_dir: Path | None = None,
    runtime_git_commit: str | None = None,
) -> ProspectiveMonitorReport:
    assert_prospective_observation(as_of)
    if bundle.git_commit is not None:
        if runtime_git_commit is None:
            raise ValueError("runtime_git_commit required when bundle.git_commit is set")
        assert_runtime_engine_commit(
            frozen_git_commit=str(bundle.git_commit),
            runtime_git_commit=str(runtime_git_commit),
            behavior_preserving_migration=bool(bundle.behavior_preserving_migration),
            approved_runtime_commits=bundle.approved_runtime_commits,
        )
    # Determine window as [prospective_start, as_of] inclusive
    start = bundle.prospective_start
    if start > as_of:
        raise ValueError(f"prospective_start {start.isoformat()} is after as_of {as_of.isoformat()}")
    end = _allocation_end_within_as_of(start=start, as_of=as_of, fill_delay_sessions=1)
    contribution = float(bundle.contribution_krw) if bundle.contribution_krw is not None else 1_000_000.0
    # precompute bundle hash for observations
    try:
        # compute arm hashes for bundle_hash calc
        tmp_hashes: list[str] = []
        for arm in bundle.arms:
            from src.policy.kafi_deployment import KafiDeploymentConfig as _KDC
            from src.policy.targets import PolicyId as _Pid
            from src.sim.allocation import AllocationConfig as _AC

            kafi_cfg_tmp = None
            if arm.kafi_deployment is not None:
                try:
                    kafi_cfg_tmp = _KDC(**arm.kafi_deployment)
                except Exception:
                    kafi_cfg_tmp = None
            cfg_tmp = _AC(
                policy=arm.policy if isinstance(arm.policy, _Pid) else _Pid.parse(arm.policy),
                start=start,
                end=end,
                monthly_contribution_krw=float(contribution),
                fill_delay_sessions=1,
                commission_bps=0.0,
                fx_spread_bps=0.0,
                targets_override=dict(arm.targets) if arm.targets is not None else None,
                kafi_deployment=kafi_cfg_tmp,
            )
            tmp_hashes.append(strategy_arm_identity_hash(cfg_tmp, bundle=bundle, arm=arm))
        computed_bundle_hash = bundle.bundle_hash if bundle.bundle_hash is not None else bundle_identity_hash(bundle, tmp_hashes)
    except Exception:
        computed_bundle_hash = bundle.bundle_hash or ""
    frozen_engine_commit = str(bundle.git_commit) if bundle.git_commit is not None else ""
    runtime_engine_commit_str = str(runtime_git_commit) if runtime_git_commit is not None else ""
    # first pass to collect results for ratio calc
    interim: list[tuple[FrozenStrategyArm, AllocationConfig, AllocationResult]] = []
    for arm in bundle.arms:
        targets = dict(arm.targets) if arm.targets is not None else None
        kafi_cfg = None
        if arm.kafi_deployment is not None:
            from src.policy.kafi_deployment import KafiDeploymentConfig

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
        # identity check using full payload
        expected_hash = strategy_arm_identity_hash(cfg, bundle=bundle, arm=arm)
        if arm.identity_hash is not None and arm.identity_hash != expected_hash:
            # allow legacy hash mismatch if legacy hash matches
            legacy_hash = strategy_arm_identity_hash(cfg)
            if legacy_hash != arm.identity_hash and expected_hash != arm.identity_hash:
                raise ValueError(f"frozen identity_hash mismatch for {arm.arm_id!r}: expected {expected_hash!r} got {arm.identity_hash!r}")
        # also ensure if bundle frozen, hash must match
        if arm.identity_hash is not None:
            frozen_for_assert = arm
            # ensure frozen hash equals either expected or legacy
            if frozen_for_assert.identity_hash not in (expected_hash, strategy_arm_identity_hash(cfg)):
                raise ValueError(f"strategy identity mismatch for {arm.arm_id!r}")
        result = runner(cfg)
        interim.append((arm, cfg, result))
    # benchmark reference for ratio
    bench_terminal: float | None = None
    for arm, _cfg, res in interim:
        if arm.role is ProspectiveArmRole.IMMUTABLE_BENCHMARK:
            bench_terminal = float(res.terminal_wealth_real_krw)
            break
    if bench_terminal is None and interim:
        bench_terminal = float(interim[0][2].terminal_wealth_real_krw)
    observations: list[ProspectiveObservation] = []
    for arm, cfg, result in interim:
        reserve_val = 0.0
        if result.snapshots:
            try:
                reserve_val = float(result.snapshots[-1].reserve_krw)
            except Exception:
                reserve_val = 0.0
        term_real = float(result.terminal_wealth_real_krw)
        total_contrib = float(result.total_contribution_real_krw)
        gain = term_real - total_contrib if total_contrib else term_real
        ratio = 1.0
        if bench_terminal is not None and bench_terminal > 0:
            if arm.role is ProspectiveArmRole.IMMUTABLE_BENCHMARK:
                ratio = 1.0
            else:
                ratio = term_real / bench_terminal if bench_terminal else 1.0
        obs = ProspectiveObservation(
            as_of=as_of,
            arm_id=arm.arm_id,
            policy=cfg.policy,
            targets=dict(cfg.targets_override) if cfg.targets_override is not None else {},
            terminal_wealth_krw=float(result.terminal_wealth_krw),
            terminal_wealth_real_krw=term_real,
            xirr=float(result.xirr),
            xirr_real=float(result.xirr_real),
            max_drawdown=float(result.max_drawdown),
            total_contribution_real_krw=total_contrib,
            cumulative_terminal_real_krw=term_real,
            cumulative_real_gain=float(gain),
            cumulative_xirr_real=float(result.xirr_real),
            cumulative_ratio_vs_benchmark=float(ratio),
            reserve_krw=float(reserve_val),
            frozen_engine_commit=frozen_engine_commit,
            runtime_engine_commit=runtime_engine_commit_str,
            bundle_hash=computed_bundle_hash,
        )
        observations.append(obs)
    if registry_dir is None:
        base = settings.resolved_data_root() / "prospective_registry"
    else:
        base = Path(registry_dir)
    base.mkdir(parents=True, exist_ok=True)
    registry_path = base / "prospective_observations.jsonl"
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
                "cumulative_terminal_real_krw": obs.cumulative_terminal_real_krw,
                "cumulative_real_gain": obs.cumulative_real_gain,
                "cumulative_xirr_real": obs.cumulative_xirr_real,
                "cumulative_ratio_vs_benchmark": obs.cumulative_ratio_vs_benchmark,
                "reserve_krw": obs.reserve_krw,
                "frozen_engine_commit": obs.frozen_engine_commit,
                "runtime_engine_commit": obs.runtime_engine_commit,
                "bundle_hash": obs.bundle_hash,
                "bundle_id": bundle.bundle_id,
            }
            fh.write(json.dumps(rec) + "\n")
    logger.info("[DATA] event=prospective_monitor as_of=%s arms=%d registry=%s", as_of.isoformat(), len(observations), registry_path.as_posix())
    return ProspectiveMonitorReport(as_of=as_of, bundle_id=bundle.bundle_id, observations=tuple(observations), registry_path=registry_path)

