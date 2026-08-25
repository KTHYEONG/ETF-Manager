"""Experiment JSON spec parsing; every schema violation fails closed."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.etf.mapping import MappingConfig
from src.policy.currency import CurrencyConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import PolicyId

__all__ = [
    "CadenceSpec",
    "CandidateSpec",
    "CurrencySpec",
    "ExperimentSpec",
    "MappingSpec",
    "OverlaySpec",
    "ReserveSpec",
    "load_experiment_config",
    "resolve_cadence",
    "resolve_currency",
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
    schedule: Literal["v1", "v2"] = "v1"
    min_invest_multiplier: float = Field(default=0.80, gt=0.0, lt=1.0)
    max_invest_multiplier: float = Field(default=2.00, gt=1.0, le=2.0)
    reserve_max_months: float = Field(default=6.00, gt=0.0, le=6.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_withhold_cap_key(cls, data: object) -> object:
        return _reconcile_canonical_key(data, canonical="withhold_cap", field="max_withhold")


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


class ExperimentSpec(BaseModel):
    """Frozen experiment contract: shared cashflow/window plus gated arms.

    ``modules`` is declared per arm (never inferred from sleeve counts) so the
    complexity-penalized adoption gate stays explicit and reproducible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    start: date
    end: date
    contribution_krw: float = Field(gt=0)
    hurdle: float = Field(ge=0)
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
    baseline: CandidateSpec
    candidates: list[CandidateSpec] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_delta0_key(cls, data: object) -> object:
        return _reconcile_canonical_key(data, canonical="delta0", field="hurdle")

    @model_validator(mode="after")
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
            ):
                raise ValueError(
                    "cadence cannot be combined with overlay, reserve, mapping, or currency experiment modules"
                )
            if any(candidate.modules < 1 for candidate in self.candidates):
                raise ValueError("cadence requires every candidate.modules >= 1")
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
    return ReserveConfig(
        max_withhold=spec.reserve.max_withhold,
        schedule=spec.reserve.schedule,
        min_invest_multiplier=spec.reserve.min_invest_multiplier,
        max_invest_multiplier=spec.reserve.max_invest_multiplier,
        reserve_max_months=spec.reserve.reserve_max_months,
    )


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


def resolve_currency(spec: ExperimentSpec) -> CurrencyConfig | None:
    """Map the JSON currency onto the runtime config, keeping window defaults."""
    if spec.currency is None:
        return None
    return CurrencyConfig(
        max_defer=spec.currency.max_defer,
        expensive_percentile=spec.currency.expensive_percentile,
    )


def load_experiment_config(path: str | Path) -> ExperimentSpec:
    """Parse an experiment JSON file into a validated spec.

    Raises:
        OSError: When the file cannot be read.
        ValueError: When the payload is not valid JSON or violates the schema.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentSpec.model_validate(payload)
