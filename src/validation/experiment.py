"""Experiment JSON spec parsing; every schema violation fails closed."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.etf.mapping import MappingConfig
from src.policy.adaptive_contribution import AdaptiveContributionConfig
from src.policy.contribution_shape import ContributionShapeConfig
from src.policy.currency import CurrencyConfig
from src.policy.kafi_deployment import KafiDeploymentConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import PolicyId
from src.policy.thesis import ThesisId, ThesisSpec

__all__ = [
    "AdaptiveContributionSpec",
    "CadenceSpec",
    "CandidateSpec",
    "ContributionShapeSpec",
    "CurrencySpec",
    "ExperimentSpec",
    "KafiDeploymentSpec",
    "MappingSpec",
    "OverlaySpec",
    "PreregistrationSpec",
    "ReserveSpec",
    "assert_experiment_preregistration",
    "experiment_target_tickers",
    "load_experiment_config",
    "resolve_adaptive_contribution",
    "resolve_arm_targets",
    "resolve_baseline_adaptive_contribution",
    "resolve_cadence",
    "resolve_contribution_shape",
    "resolve_currency",
    "resolve_kafi_deployment",
    "resolve_mapping",
    "resolve_overlay",
    "resolve_reserve",
]


def _reconcile_canonical_key(data: object, canonical: str, field: str) -> object:
    """Fold a legacy JSON key onto its canonical field; conflicting duplicates fail closed."""
    if not isinstance(data, dict) or canonical not in data:
        return data
    if field in data and data[field] != data[canonical]:
        raise ValueError(f"{canonical}={data[canonical]!r} conflicts with {field}={data[field]!r}")
    merged = dict(data)
    merged[field] = merged.pop(canonical)
    return merged


class CandidateSpec(BaseModel):
    """One allocation arm: policy identity plus its declared module count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    policy: PolicyId
    modules: int = Field(ge=0)
    targets: dict[str, float] | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_extra_rules_key(cls, data: object) -> object:
        return _reconcile_canonical_key(data, canonical="extra_rules", field="modules")

    @field_validator("policy", mode="before")
    @classmethod
    def _coerce_policy(cls, value: object) -> object:
        try:
            return PolicyId.parse(value)
        except ValueError as exc:
            raise ValueError(f"unknown policy {value!r}") from exc

    @field_validator("targets", mode="before")
    @classmethod
    def _validate_targets(cls, value: object) -> object:
        if value is None:
            return None
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
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"targets[{key!r}] must be finite nonnegative, got {raw_weight!r}")
            normalized[key] = weight
        total = sum(normalized.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
            raise ValueError(f"targets weights must sum to 1.0 within 1e-6, got {total!r}")
        return normalized


class OverlaySpec(BaseModel):
    """Bounded dynamic-overlay parameters accepted in experiment JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_shift: float = Field(gt=0.0, le=0.10)
    vix_threshold: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_max_tilt_key(cls, data: object) -> object:
        return _reconcile_canonical_key(data, canonical="max_tilt", field="max_shift")


class ReserveSpec(BaseModel):
    """Explicit reserve-ledger parameters accepted in experiment JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_withhold: float = Field(gt=0.0, le=0.10)
    schedule: Literal["v1", "v2", "v3", "v4"] = "v1"
    min_invest_multiplier: float = Field(default=0.80, gt=0.0, lt=1.0)
    max_invest_multiplier: float = Field(default=2.00, gt=1.0)
    reserve_max_months: float = Field(default=6.00, gt=0.0, le=6.0)
    vix_threshold: float = Field(default=20.0, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_withhold_cap_key(cls, data: object) -> object:
        return _reconcile_canonical_key(data, canonical="withhold_cap", field="max_withhold")

    @model_validator(mode="before")
    @classmethod
    def _rebase_v3_baselines(cls, data: object) -> object:
        """Rebase omitted or legacy-baseline multipliers onto the wider v3/v4 band; VIX stays v3-only."""
        if not isinstance(data, dict) or data.get("schedule") not in ("v3", "v4"):
            return data
        merged = dict(data)
        if merged.get("min_invest_multiplier") in (None, 0.80):
            merged["min_invest_multiplier"] = 0.70
        if merged.get("max_invest_multiplier") in (None, 2.00):
            merged["max_invest_multiplier"] = 3.00
        if data.get("schedule") == "v3" and merged.get("vix_threshold") in (None, 20.0):
            merged["vix_threshold"] = 25.0
        return merged

    @model_validator(mode="after")
    def _check_schedule_band(self) -> ReserveSpec:
        """Enforce the schedule-dependent max-invest ceiling (2.0 for v1/v2, 3.0 for v3/v4)."""
        ceiling = 3.00 if self.schedule in ("v3", "v4") else 2.00
        if not 1.0 < self.max_invest_multiplier <= ceiling:
            raise ValueError(
                f"max_invest_multiplier must lie in (1.0, {ceiling}] for schedule "
                f"{self.schedule!r}, got {self.max_invest_multiplier!r}"
            )
        return self


class MappingSpec(BaseModel):
    """Implementation-mapping hysteresis parameters accepted in experiment JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_improvement: float = Field(gt=0.0, le=1.0, default=0.02)


class CurrencySpec(BaseModel):
    """FX-defer parameters accepted in experiment JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_defer: float = Field(gt=0.0, le=1.0)
    expensive_percentile: float = Field(gt=0.0, lt=1.0, default=0.80)


class CadenceSpec(BaseModel):
    """Decision-cadence module accepted in experiment JSON; the anchor fails closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor: Literal["month_open", "twice_monthly"]


class ContributionShapeSpec(BaseModel):
    """KAFI contribution-shaping parameters accepted in experiment JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    equity_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    credit_series_id: str = "BAA10Y"
    min_multiplier: float = Field(default=0.70, gt=0.0, lt=1.0)
    max_multiplier: float = Field(default=1.50, gt=1.0, le=2.0)
    budget_window_months: int = Field(default=12, ge=3, le=24)
    rank_window: int = Field(default=252, ge=63)


class KafiDeploymentSpec(BaseModel):
    """Causal KAFI deployment parameters accepted in experiment JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    equity_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    credit_series_id: str = "BAA10Y"
    min_multiplier: float = Field(default=0.70, gt=0.0, lt=1.0)
    max_multiplier: float = Field(default=1.30, gt=1.0, le=1.5)
    rank_window: int = Field(default=252, ge=63)


class AdaptiveContributionSpec(BaseModel):
    """Stateless adaptive-contribution parameters accepted in experiment JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    equity_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    credit_series_id: str = "BAA10Y"
    min_multiplier: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_multiplier: float = Field(default=2.0, gt=1.0, le=2.0)
    downside_power: float = Field(default=2.5, gt=0.0)
    upside_power: float = Field(default=0.7, gt=0.0)
    rank_window: int = Field(default=126, ge=63)
    include_vol_dampener: bool = True
    dispersion: float = Field(default=1.0, gt=0.0)
    neutral_deadband: float = Field(default=0.0, ge=0.0)


class PreregistrationSpec(BaseModel):
    """Thesis preregistration flags; universe lock gates tickers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights_locked: bool = False
    universe_locked: bool = False
    baseline_frozen: bool = True


class ExperimentSpec(BaseModel):
    """Frozen experiment contract: shared cashflow/window plus gated arms.

    ``modules`` is declared per arm (never inferred from sleeve counts) so the
    complexity-penalized adoption gate stays explicit and reproducible.
    ``objective`` picks the verdict gate: ``ce`` (default), ``growth_first``, or
    ``adaptive_growth``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    start: date
    end: date
    contribution_krw: float = Field(gt=0)
    hurdle: float = Field(ge=0)
    objective: Literal["ce", "growth_first", "adaptive_growth"] = "ce"
    horizon_months: int = Field(ge=0)
    commission_bps: float = Field(default=0.0, ge=0)
    fx_spread_bps: float = Field(default=0.0, ge=0)
    train_months: int | None = Field(default=None, ge=1)
    test_months: int | None = Field(default=None, ge=1)
    overlay: OverlaySpec | None = None
    reserve: ReserveSpec | None = None
    mapping: MappingSpec | None = None
    currency: CurrencySpec | None = None
    cadence: CadenceSpec | None = None
    contribution_shape: ContributionShapeSpec | None = None
    kafi_deployment: KafiDeploymentSpec | None = None
    adaptive_contribution: AdaptiveContributionSpec | None = None
    baseline_adaptive_contribution: AdaptiveContributionSpec | None = None
    baseline: CandidateSpec
    candidates: list[CandidateSpec] = Field(min_length=1)
    thesis_id: ThesisId | None = None
    preregistration: PreregistrationSpec | None = None

    @field_validator("thesis_id", mode="before")
    @classmethod
    def _coerce_thesis_id(cls, value: object) -> object:
        if value is None:
            return None
        try:
            return ThesisId(value)  # type: ignore[arg-type]
        except ValueError as exc:
            raise ValueError(f"unknown thesis {value!r}") from exc

    @model_validator(mode="before")
    @classmethod
    def _accept_delta0_key(cls, data: object) -> object:
        return _reconcile_canonical_key(data, canonical="delta0", field="hurdle")

    @model_validator(mode="after")
    def _run_structure_checks(self) -> ExperimentSpec:
        return self._check_structure()

    def _check_structure(self) -> ExperimentSpec:
        if self.start > self.end:
            raise ValueError(f"start {self.start.isoformat()} is after end {self.end.isoformat()}")
        months_set = [name for name in ("train_months", "test_months") if getattr(self, name) is not None]
        if len(months_set) == 1:
            raise ValueError(f"{months_set[0]} alone is invalid; set both train_months and test_months")
        if len(months_set) == 2 and len(self.candidates) != 1:
            raise ValueError(f"walk-forward specs require exactly one candidate, got {len(self.candidates)}")
        if self.overlay is not None and any(candidate.modules < 1 for candidate in self.candidates):
            raise ValueError("overlay requires every candidate.modules >= 1")
        if self.reserve is not None:
            if self.overlay is not None:
                raise ValueError("overlay and reserve are mutually exclusive experiment modules")
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError("reserve requires every candidate.modules >= 1")
        if self.mapping is not None:
            if self.overlay is not None or self.reserve is not None:
                raise ValueError("mapping cannot be combined with overlay or reserve experiment modules")
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError("mapping requires every candidate.modules >= 1")
        if self.currency is not None:
            if self.overlay is not None or self.reserve is not None or self.mapping is not None:
                raise ValueError(
                    "currency cannot be combined with overlay, reserve, or mapping experiment modules"
                )
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError("currency requires every candidate.modules >= 1")
        if self.cadence is not None:
            if (
                self.overlay is not None
                or self.reserve is not None
                or self.mapping is not None
                or self.currency is not None
                or self.contribution_shape is not None
                or self.kafi_deployment is not None
                or self.adaptive_contribution is not None
            ):
                raise ValueError(
                    "cadence cannot be combined with overlay, reserve, mapping, currency, "
                    "contribution_shape, kafi_deployment, or adaptive_contribution experiment modules"
                )
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError("cadence requires every candidate.modules >= 1")
        if self.contribution_shape is not None:
            if any(
                module is not None
                for module in (
                    self.overlay,
                    self.reserve,
                    self.mapping,
                    self.currency,
                    self.cadence,
                    self.kafi_deployment,
                    self.adaptive_contribution,
                )
            ):
                raise ValueError(
                    "contribution_shape cannot be combined with overlay, reserve, mapping, "
                    "currency, cadence, kafi_deployment, or adaptive_contribution experiment modules"
                )
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError("contribution_shape requires every candidate.modules >= 1")
        if self.kafi_deployment is not None:
            if any(
                module is not None
                for module in (
                    self.overlay,
                    self.reserve,
                    self.mapping,
                    self.currency,
                    self.cadence,
                    self.contribution_shape,
                    self.adaptive_contribution,
                )
            ):
                raise ValueError(
                    "kafi_deployment cannot be combined with overlay, reserve, mapping, "
                    "currency, cadence, contribution_shape, or adaptive_contribution experiment modules"
                )
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError("kafi_deployment requires every candidate.modules >= 1")
        if self.adaptive_contribution is not None:
            if any(
                module is not None
                for module in (
                    self.overlay,
                    self.reserve,
                    self.mapping,
                    self.currency,
                    self.cadence,
                    self.contribution_shape,
                    self.kafi_deployment,
                )
            ):
                raise ValueError(
                    "adaptive_contribution cannot be combined with overlay, reserve, mapping, "
                    "currency, cadence, contribution_shape, or kafi_deployment experiment modules"
                )
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError("adaptive_contribution requires every candidate.modules >= 1")
        if self.baseline_adaptive_contribution is not None:
            if any(
                module is not None
                for module in (
                    self.overlay,
                    self.reserve,
                    self.mapping,
                    self.currency,
                    self.cadence,
                    self.contribution_shape,
                    self.kafi_deployment,
                )
            ):
                raise ValueError(
                    "baseline_adaptive_contribution cannot be combined with overlay, reserve, mapping, "
                    "currency, cadence, contribution_shape, or kafi_deployment experiment modules"
                )
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError(
                    "baseline_adaptive_contribution requires every candidate.modules >= 1"
                )
        growth_first_modules = (self.cadence, self.reserve, self.contribution_shape, self.kafi_deployment)
        if (
            self.objective == "growth_first"
            and sum(module is not None for module in growth_first_modules) != 1
        ):
            raise ValueError(
                "objective 'growth_first' requires exactly one of a cadence, reserve, "
                "contribution_shape, or kafi_deployment module"
            )
        if self.objective == "adaptive_growth" and self.adaptive_contribution is None:
            raise ValueError("objective 'adaptive_growth' requires exactly one adaptive_contribution module")
        if self.baseline_adaptive_contribution is not None and self.objective != "adaptive_growth":
            raise ValueError(
                f"baseline_adaptive_contribution requires objective 'adaptive_growth', got {self.objective!r}"
            )
        seen: set[str] = set()
        for candidate in self.candidates:
            if candidate.id in seen:
                raise ValueError(f"duplicate candidate id {candidate.id!r}")
            seen.add(candidate.id)
        return self


def resolve_overlay(spec: ExperimentSpec) -> OverlayConfig | None:
    """Map the JSON overlay onto the runtime config, keeping window defaults."""
    if spec.overlay is None:
        return None
    return OverlayConfig(max_shift=spec.overlay.max_shift, vix_threshold=spec.overlay.vix_threshold)


def resolve_reserve(spec: ExperimentSpec) -> ReserveConfig | None:
    """Map the JSON reserve onto the runtime config, keeping window defaults."""
    if spec.reserve is None:
        return None
    return ReserveConfig(max_withhold=spec.reserve.max_withhold, schedule=spec.reserve.schedule, min_invest_multiplier=spec.reserve.min_invest_multiplier, max_invest_multiplier=spec.reserve.max_invest_multiplier, reserve_max_months=spec.reserve.reserve_max_months, vix_threshold=spec.reserve.vix_threshold)


def resolve_mapping(spec: ExperimentSpec) -> MappingConfig | None:
    """Map the JSON mapping onto the runtime config, keeping catalog defaults."""
    if spec.mapping is None:
        return None
    return MappingConfig(min_improvement=spec.mapping.min_improvement)


def resolve_cadence(spec: ExperimentSpec) -> Literal["month_open", "twice_monthly"] | None:
    """Map the JSON cadence anchor onto the runtime schedule frequency."""
    if spec.cadence is None:
        return None
    return spec.cadence.anchor


def resolve_contribution_shape(spec: ExperimentSpec) -> ContributionShapeConfig | None:
    """Map the JSON contribution shape onto the runtime config, keeping window defaults."""
    if spec.contribution_shape is None:
        return None
    return ContributionShapeConfig(
        equity_ticker=spec.contribution_shape.equity_ticker,
        bond_ticker=spec.contribution_shape.bond_ticker,
        credit_series_id=spec.contribution_shape.credit_series_id,
        min_multiplier=spec.contribution_shape.min_multiplier,
        max_multiplier=spec.contribution_shape.max_multiplier,
        budget_window_months=spec.contribution_shape.budget_window_months,
        rank_window=spec.contribution_shape.rank_window,
    )


def resolve_kafi_deployment(spec: ExperimentSpec) -> KafiDeploymentConfig | None:
    """Map the JSON KAFI deployment onto the runtime config, keeping window defaults."""
    if spec.kafi_deployment is None:
        return None
    return KafiDeploymentConfig(
        equity_ticker=spec.kafi_deployment.equity_ticker,
        bond_ticker=spec.kafi_deployment.bond_ticker,
        credit_series_id=spec.kafi_deployment.credit_series_id,
        min_multiplier=spec.kafi_deployment.min_multiplier,
        max_multiplier=spec.kafi_deployment.max_multiplier,
        rank_window=spec.kafi_deployment.rank_window,
    )


def _to_adaptive_config(module: AdaptiveContributionSpec) -> AdaptiveContributionConfig:
    """Map one JSON adaptive-contribution spec onto the runtime config, keeping window defaults.

    Mapping shape: AdaptiveContributionConfig(equity_ticker=..., bond_ticker=..., credit_series_id=..., min_multiplier=..., max_multiplier=..., downside_power=..., upside_power=..., rank_window=..., include_vol_dampener=..., dispersion=..., neutral_deadband=...)
    """
    return AdaptiveContributionConfig(
        equity_ticker=module.equity_ticker,
        bond_ticker=module.bond_ticker,
        credit_series_id=module.credit_series_id,
        min_multiplier=module.min_multiplier,
        max_multiplier=module.max_multiplier,
        downside_power=module.downside_power,
        upside_power=module.upside_power,
        rank_window=module.rank_window,
        include_vol_dampener=module.include_vol_dampener,
        dispersion=module.dispersion,
        neutral_deadband=module.neutral_deadband,
    )


def resolve_adaptive_contribution(spec: ExperimentSpec) -> AdaptiveContributionConfig | None:
    """Map the JSON adaptive-contribution onto the runtime config, keeping window defaults."""
    if spec.adaptive_contribution is None:
        return None
    return _to_adaptive_config(spec.adaptive_contribution)


def resolve_baseline_adaptive_contribution(spec: ExperimentSpec) -> AdaptiveContributionConfig | None:
    """Mirror resolve_adaptive_contribution for the optional locked-baseline arm."""
    if spec.baseline_adaptive_contribution is None:
        return None
    return _to_adaptive_config(spec.baseline_adaptive_contribution)


def resolve_currency(spec: ExperimentSpec) -> CurrencyConfig | None:
    """Map the JSON currency onto the runtime config, keeping window defaults."""
    if spec.currency is None:
        return None
    return CurrencyConfig(
        max_defer=spec.currency.max_defer,
        expensive_percentile=spec.currency.expensive_percentile,
    )


def resolve_arm_targets(arm: CandidateSpec) -> dict[str, float] | None:
    """Return a shallow copy of validated targets, or None when absent."""
    if arm.targets is None:
        return None
    return dict(arm.targets)


def experiment_target_tickers(spec: ExperimentSpec) -> tuple[str, ...]:
    """First-seen ordered union of ticker keys from all target maps in the spec."""
    ordered: dict[str, None] = {}
    for arm in (spec.baseline, *spec.candidates):
        if arm.targets is not None:
            for ticker in arm.targets:
                ordered.setdefault(ticker)
    return tuple(ordered)


def assert_experiment_preregistration(
    spec: ExperimentSpec,
    registry: Mapping[ThesisId, ThesisSpec],
    *,
    thesis_config_dir: Path | None = None,
) -> None:
    """Fail closed on preregistration violations; never calls adoption_passes."""
    del thesis_config_dir
    prereg = spec.preregistration
    if prereg is not None and prereg.universe_locked and spec.thesis_id is None:
        raise ValueError("universe_locked requires thesis_id")
    if spec.thesis_id is not None:
        try:
            tid = spec.thesis_id if isinstance(spec.thesis_id, ThesisId) else ThesisId(spec.thesis_id)
        except ValueError as exc:
            raise ValueError(f"unknown thesis {spec.thesis_id!r}") from exc
        if tid not in registry:
            raise ValueError(f"unknown thesis {spec.thesis_id!r}")
        if prereg is not None and prereg.universe_locked:
            thesis = registry[tid]
            allowed = {"QQQ"} | {v.value for v in thesis.historical_proxies}
            for arm in (spec.baseline, *spec.candidates):
                if arm.targets is not None:
                    for ticker in arm.targets:
                        upper = str(ticker).strip().upper()
                        if upper not in allowed:
                            raise ValueError(f"ticker {ticker!r} not in allowed universe {sorted(allowed)!r}")


def load_experiment_config(path: str | Path) -> ExperimentSpec:
    """Parse an experiment JSON file into a validated spec.

    Raises:
        OSError: When the file cannot be read.
        ValueError: When the payload is not valid JSON or violates the schema.
    """
    text = Path(path).read_text(encoding="utf-8")
    # Strip // line and trailing comments to allow placeholder comments in JSON.
    import re

    text = re.sub(r"//.*", "", text)
    payload = json.loads(text)
    return ExperimentSpec.model_validate(payload)
