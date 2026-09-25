"""After-tax campaign preregistration: strict spec parsing without simulation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from src.policy.after_tax_rules import parse_after_tax_rule_spec
from src.validation.historical_campaign import REGIME_COVERAGE_CATALOG

if TYPE_CHECKING:
    from src.policy.after_tax_rules import AfterTaxRuleSpec
    from src.sim.after_tax_engine import ExecutionMode

__all__ = [
    "AfterTaxArmSpec",
    "AfterTaxCampaignSpec",
    "AfterTaxGateSpec",
    "ArmRole",
    "load_after_tax_campaign_spec",
]


class ArmRole(StrEnum):
    """Campaign role; only the operational candidate can pass the adoption gate."""

    BASELINE = "baseline"
    OPERATIONAL_CANDIDATE = "operational_candidate"
    PROSPECTIVE_WATCH = "prospective_watch"
    DISCLOSURE = "disclosure"


@dataclass(frozen=True, slots=True)
class AfterTaxArmSpec:
    arm_id: str
    role: ArmRole
    rule: AfterTaxRuleSpec
    mode: ExecutionMode
    rebalance_band: float | None
    harvest_gains: bool


@dataclass(frozen=True, slots=True)
class AfterTaxGateSpec:
    """Adoption thresholds on the primary horizon; informational only."""

    median_ratio_floor: float
    worst_ratio_floor: float
    bootstrap_p05_floor: float


@dataclass(frozen=True, slots=True)
class AfterTaxCampaignSpec:
    name: str
    start: date
    end: date
    contribution_krw: float
    commission_bps: float
    fx_spread_bps: float
    tax_regime_path: str
    horizons_months: tuple[int, ...]
    primary_horizon_months: int
    step_months: int
    crash_regimes: tuple[str, ...]
    crash_window_months: int
    fractional_shares: bool
    bootstrap_paths: int
    baseline_arm_id: str
    arms: tuple[AfterTaxArmSpec, ...]
    gate: AfterTaxGateSpec


def _parse_date(value: object, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date, got {value!r}") from exc


def _parse_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite positive, got {value!r}")
    return result


def _parse_nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite nonnegative, got {value!r}")
    return result


def load_after_tax_campaign_spec(path: str | Path) -> AfterTaxCampaignSpec:
    """Parse the existing after-tax campaign, arms, gates, and dated market windows.

    Args:
        path: Versioned campaign definition.

    Returns:
        Typed campaign specification.

    Raises:
        ValueError: If an arm, gate, date, weight, or friction field is invalid.
    """
    from src.sim.after_tax_engine import ExecutionMode

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("campaign JSON must be an object")
    try:
        name = str(document["name"]).strip()
        if not name:
            raise ValueError("name must be non-blank")
        start = _parse_date(document["start"], "start")
        end = _parse_date(document["end"], "end")
        if start > end:
            raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
        contribution = _parse_positive_float(document["contribution_krw"], "contribution_krw")
        commission = _parse_nonnegative_float(document["commission_bps"], "commission_bps")
        spread = _parse_nonnegative_float(document["fx_spread_bps"], "fx_spread_bps")
        tax_regime_path = str(document["tax_regime_path"]).strip()
        if not tax_regime_path:
            raise ValueError("tax_regime_path must be non-blank")
        raw_horizons = document["horizons_months"]
        if not isinstance(raw_horizons, list) or not raw_horizons:
            raise ValueError("horizons_months must be a nonempty list")
        horizons: list[int] = []
        for item in raw_horizons:
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(f"horizons_months entries must be positive integers, got {item!r}")
            horizons.append(item)
        primary = document["primary_horizon_months"]
        if isinstance(primary, bool) or not isinstance(primary, int) or primary < 1:
            raise ValueError(f"primary_horizon_months must be a positive integer, got {primary!r}")
        if primary not in horizons:
            raise ValueError(f"primary_horizon_months {primary} absent from horizons_months {horizons!r}")
        step = document["step_months"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(f"step_months must be a positive integer, got {step!r}")
        raw_crash = document.get("crash_regimes", [])
        if not isinstance(raw_crash, list):
            raise ValueError("crash_regimes must be a list")
        crash_regimes = tuple(str(item) for item in raw_crash)
        known_regimes = {window.regime_name for window in REGIME_COVERAGE_CATALOG}
        for regime_name in crash_regimes:
            if regime_name not in known_regimes:
                raise ValueError(f"unknown crash regime {regime_name!r}")
        crash_window = document["crash_window_months"]
        if isinstance(crash_window, bool) or not isinstance(crash_window, int) or crash_window < 1:
            raise ValueError(f"crash_window_months must be a positive integer, got {crash_window!r}")
        fractional = document["fractional_shares"]
        if not isinstance(fractional, bool):
            raise ValueError(f"fractional_shares must be a boolean, got {fractional!r}")
        bootstrap_paths = document["bootstrap_paths"]
        if isinstance(bootstrap_paths, bool) or not isinstance(bootstrap_paths, int) or bootstrap_paths < 1:
            raise ValueError(f"bootstrap_paths must be integer >= 1, got {bootstrap_paths!r}")
        baseline_arm_id = str(document["baseline_arm_id"]).strip()
        if not baseline_arm_id:
            raise ValueError("baseline_arm_id must be non-blank")
        raw_gate = document["gate"]
        if not isinstance(raw_gate, dict):
            raise ValueError("gate must be an object")
        gate = AfterTaxGateSpec(
            median_ratio_floor=float(raw_gate["median_ratio_floor"]),
            worst_ratio_floor=float(raw_gate["worst_ratio_floor"]),
            bootstrap_p05_floor=float(raw_gate["bootstrap_p05_floor"]),
        )
        for field_name in ("median_ratio_floor", "worst_ratio_floor", "bootstrap_p05_floor"):
            if not math.isfinite(getattr(gate, field_name)):
                raise ValueError(f"gate.{field_name} must be finite")
        raw_arms = document["arms"]
        if not isinstance(raw_arms, list) or not raw_arms:
            raise ValueError("arms must be a nonempty list")
        arms: list[AfterTaxArmSpec] = []
        seen_ids: set[str] = set()
        for entry in raw_arms:
            if not isinstance(entry, dict):
                raise ValueError("arm entries must be objects")
            arm_id = str(entry["arm_id"]).strip()
            if not arm_id:
                raise ValueError("arm_id must be non-blank")
            if arm_id in seen_ids:
                raise ValueError(f"duplicate arm id {arm_id!r}")
            seen_ids.add(arm_id)
            try:
                role = ArmRole(str(entry["role"]))
            except ValueError as exc:
                raise ValueError(f"unknown arm role {entry.get('role')!r}") from exc
            try:
                mode = ExecutionMode(str(entry["mode"]))
            except ValueError as exc:
                raise ValueError(f"unknown execution mode {entry.get('mode')!r}") from exc
            band_raw = entry.get("rebalance_band")
            band: float | None = None
            if band_raw is not None:
                if isinstance(band_raw, bool) or not isinstance(band_raw, int | float):
                    raise ValueError(f"rebalance_band must be numeric, got {band_raw!r}")
                band = float(band_raw)
                if not math.isfinite(band) or not 0.0 <= band <= 1.0:
                    raise ValueError(f"rebalance_band must lie in [0, 1], got {band_raw!r}")
            if mode is ExecutionMode.REBALANCE_BAND and band is None:
                raise ValueError(f"arm {arm_id!r} uses REBALANCE_BAND without a band")
            if mode is ExecutionMode.BUY_ONLY and band is not None:
                raise ValueError(f"arm {arm_id!r} uses BUY_ONLY with a band")
            harvest = entry.get("harvest_gains")
            if not isinstance(harvest, bool):
                raise ValueError(f"arm {arm_id!r} harvest_gains must be a boolean")
            rule_payload = entry.get("rule")
            if not isinstance(rule_payload, dict):
                raise ValueError(f"arm {arm_id!r} rule must be an object")
            rule = parse_after_tax_rule_spec(rule_payload)
            arms.append(
                AfterTaxArmSpec(
                    arm_id=arm_id, role=role, rule=rule, mode=mode, rebalance_band=band, harvest_gains=harvest
                )
            )
    except KeyError as exc:
        raise ValueError(f"campaign JSON missing field {exc}") from exc
    baseline_arms = [arm for arm in arms if arm.arm_id == baseline_arm_id]
    if not baseline_arms:
        raise ValueError(f"baseline arm {baseline_arm_id!r} not found")
    if baseline_arms[0].role is not ArmRole.BASELINE:
        raise ValueError(f"baseline arm {baseline_arm_id!r} role must be baseline")
    if sum(1 for arm in arms if arm.role is ArmRole.BASELINE) != 1:
        raise ValueError("exactly one BASELINE arm is required")
    if not Path(tax_regime_path).is_file():
        raise ValueError(f"tax_regime_path not found: {tax_regime_path!r}")
    return AfterTaxCampaignSpec(
        name=name,
        start=start,
        end=end,
        contribution_krw=contribution,
        commission_bps=commission,
        fx_spread_bps=spread,
        tax_regime_path=tax_regime_path,
        horizons_months=tuple(horizons),
        primary_horizon_months=primary,
        step_months=step,
        crash_regimes=crash_regimes,
        crash_window_months=crash_window,
        fractional_shares=fractional,
        bootstrap_paths=bootstrap_paths,
        baseline_arm_id=baseline_arm_id,
        arms=tuple(arms),
        gate=gate,
    )
