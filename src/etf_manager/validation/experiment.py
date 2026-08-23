"""Experiment JSON spec parsing; every schema violation fails closed."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.etf_manager.policy.targets import PolicyId

__all__ = ["CandidateSpec", "ExperimentSpec", "load_experiment_config"]


class CandidateSpec(BaseModel):
    """One allocation arm: policy identity plus its declared module count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    policy: PolicyId
    modules: int = Field(ge=0)

    @field_validator("policy", mode="before")
    @classmethod
    def _coerce_policy(cls, value: object) -> object:
        try:
            return PolicyId(value) if isinstance(value, PolicyId) else PolicyId(str(value))
        except ValueError as exc:
            raise ValueError(f"unknown policy {value!r}") from exc


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
    delta0: float = Field(ge=0)
    horizon_months: int = Field(ge=0)
    commission_bps: float = Field(default=0.0, ge=0)
    fx_spread_bps: float = Field(default=0.0, ge=0)
    train_months: int | None = Field(default=None, ge=1)
    test_months: int | None = Field(default=None, ge=1)
    baseline: CandidateSpec
    candidates: list[CandidateSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_structure(self) -> ExperimentSpec:
        if self.start > self.end:
            raise ValueError(f"start {self.start.isoformat()} is after end {self.end.isoformat()}")
        months_set = [name for name in ("train_months", "test_months") if getattr(self, name) is not None]
        if len(months_set) == 1:
            raise ValueError(f"{months_set[0]} alone is invalid; set both train_months and test_months")
        if len(months_set) == 2 and len(self.candidates) != 1:
            raise ValueError(f"walk-forward specs require exactly one candidate, got {len(self.candidates)}")
        seen: set[str] = set()
        for candidate in self.candidates:
            if candidate.id in seen:
                raise ValueError(f"duplicate candidate id {candidate.id!r}")
            seen.add(candidate.id)
        return self


def load_experiment_config(path: str | Path) -> ExperimentSpec:
    """Parse an experiment JSON file into a validated spec.

    Raises:
        OSError: When the file cannot be read.
        ValueError: When the payload is not valid JSON or violates the schema.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentSpec.model_validate(payload)
