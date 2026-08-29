"""Research thesis registry and lifecycle (Wave 0 identity)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.etf.sleeves import SleeveId, VehicleId

__all__ = [
    "FILE_LOADABLE_STATUSES",
    "LEGAL_TRANSITIONS",
    "Evidence",
    "Horizon",
    "ThesisError",
    "ThesisId",
    "ThesisSpec",
    "ThesisStatus",
    "get_thesis",
    "load_thesis_registry",
    "transition_thesis",
]


class ThesisError(RuntimeError):
    """Thesis registry or lifecycle operation failed closed."""


class ThesisId(StrEnum):
    """Named economic theses (research unit, not ticker)."""

    AI_COMPUTE = "ai_compute"
    AI_POWER_BOTTLENECK = "ai_power_bottleneck"
    PHYSICAL_AUTOMATION = "physical_automation"


class ThesisStatus(StrEnum):
    """Thesis lifecycle states; no ADOPTED member."""

    DISCOVERED = "discovered"
    RESEARCH = "research"
    REJECTED = "rejected"
    DORMANT = "dormant"
    CONFIRMED = "confirmed"
    PROSPECTIVE_CHALLENGER = "prospective_challenger"
    OPERATIONAL_CHALLENGER = "operational_challenger"
    REOPENED = "reopened"


FILE_LOADABLE_STATUSES: Final[frozenset[ThesisStatus]] = frozenset(
    {
        ThesisStatus.DISCOVERED,
        ThesisStatus.RESEARCH,
        ThesisStatus.REJECTED,
        ThesisStatus.DORMANT,
    }
)

LEGAL_TRANSITIONS: Final[frozenset[tuple[ThesisStatus, ThesisStatus]]] = frozenset(
    {
        (ThesisStatus.DISCOVERED, ThesisStatus.RESEARCH),
        (ThesisStatus.DISCOVERED, ThesisStatus.REJECTED),
        (ThesisStatus.RESEARCH, ThesisStatus.CONFIRMED),
        (ThesisStatus.RESEARCH, ThesisStatus.REJECTED),
        (ThesisStatus.RESEARCH, ThesisStatus.DORMANT),
        (ThesisStatus.CONFIRMED, ThesisStatus.PROSPECTIVE_CHALLENGER),
        (ThesisStatus.CONFIRMED, ThesisStatus.REJECTED),
        (ThesisStatus.CONFIRMED, ThesisStatus.DORMANT),
        (ThesisStatus.PROSPECTIVE_CHALLENGER, ThesisStatus.OPERATIONAL_CHALLENGER),
        (ThesisStatus.PROSPECTIVE_CHALLENGER, ThesisStatus.REJECTED),
        (ThesisStatus.PROSPECTIVE_CHALLENGER, ThesisStatus.DORMANT),
        (ThesisStatus.OPERATIONAL_CHALLENGER, ThesisStatus.REJECTED),
        (ThesisStatus.OPERATIONAL_CHALLENGER, ThesisStatus.DORMANT),
        (ThesisStatus.REJECTED, ThesisStatus.REOPENED),
        (ThesisStatus.DORMANT, ThesisStatus.REOPENED),
        (ThesisStatus.DORMANT, ThesisStatus.REJECTED),
        (ThesisStatus.REOPENED, ThesisStatus.RESEARCH),
    }
)


class Horizon(BaseModel):
    """Investment horizon in years; target >= min."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_years: int = Field(ge=1)
    target_years: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_order(self) -> Horizon:
        if self.target_years < self.min_years:
            raise ValueError(f"target_years {self.target_years!r} must be >= min_years {self.min_years!r}")
        return self


class Evidence(BaseModel):
    """Five evidence slots plus declared source; Wave 0 all unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["declared"] = "declared"
    structural: str = "unknown"
    historical: str = "unknown"
    valuation: str = "unknown"
    expectations: str = "unknown"
    crowding: str = "unknown"




class ThesisSpec(BaseModel):
    """Frozen thesis contract; falsifiers and sleeves required; evidence is declared in Wave 0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: ThesisId
    version: int = Field(ge=1)
    title: str = Field(min_length=1)
    status: ThesisStatus
    horizon: Horizon
    causal_chain: list[str] = Field(min_length=1)
    falsifiers: list[str] = Field(min_length=1)
    candidate_sleeves: list[SleeveId] = Field(min_length=1)
    historical_proxies: list[VehicleId] = Field(min_length=1)
    evidence: Evidence = Field(default_factory=Evidence)

    @field_validator("falsifiers", mode="after")
    @classmethod
    def _check_falsifiers_non_blank(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("falsifiers entries must be non-blank strings")
        return value

    @field_validator("causal_chain", mode="after")
    @classmethod
    def _check_causal_non_blank(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("causal_chain entries must be non-blank strings")
        return value

    @field_validator("status", mode="after")
    @classmethod
    def _check_status_loadable(cls, value: ThesisStatus) -> ThesisStatus:
        if value not in FILE_LOADABLE_STATUSES:
            raise ValueError(f"status {value!r} is not file-loadable; allowed {sorted(s.value for s in FILE_LOADABLE_STATUSES)!r}")
        return value


def load_thesis_registry(directory: Path) -> Mapping[ThesisId, ThesisSpec]:
    """Load and validate every ``*.json`` in ``directory`` sorted by filename.

    Raises:
        ThesisError: On missing/non-dir/empty inputs or duplicate ids.
        ValidationError: On unknown enum members or extra keys (via pydantic).
    """
    if not directory.exists() or not directory.is_dir():
        raise ThesisError(f"thesis registry directory missing or not a directory: {directory!r}")
    files = sorted(p for p in directory.glob("*.json") if p.name != "experiment_map.json")
    if not files:
        raise ThesisError(f"thesis registry empty: {directory!r}")
    registry: dict[ThesisId, ThesisSpec] = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            raise ThesisError(f"cannot read thesis {path.name}: {exc}") from exc
        # Let ValidationError propagate for extra keys / unknown enum.
        spec = ThesisSpec.model_validate(payload)
        if spec.id in registry:
            raise ThesisError(f"duplicate thesis id {spec.id!r} in {path.name!r}")
        registry[spec.id] = spec
    return registry


def get_thesis(registry: Mapping[ThesisId, ThesisSpec], thesis_id: ThesisId) -> ThesisSpec:
    """Return the thesis for ``thesis_id`` or fail closed."""
    try:
        return registry[thesis_id]
    except KeyError as exc:
        raise ThesisError(f"unknown thesis id {thesis_id!r}") from exc


def transition_thesis(spec: ThesisSpec, new_status: ThesisStatus, *, reason: str) -> ThesisSpec:
    """Return a new frozen spec with ``new_status`` if the transition is legal.

    Raises:
        ThesisError: On blank reason or illegal transition pair.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ThesisError("transition reason must be non-blank")
    if (spec.status, new_status) not in LEGAL_TRANSITIONS:
        raise ThesisError(f"illegal transition {spec.status!r} -> {new_status!r}")
    return ThesisSpec.model_construct(
        _fields_set=spec.__pydantic_fields_set__,
        id=spec.id,
        version=spec.version,
        title=spec.title,
        status=new_status,
        horizon=spec.horizon,
        causal_chain=spec.causal_chain,
        falsifiers=spec.falsifiers,
        candidate_sleeves=spec.candidate_sleeves,
        historical_proxies=spec.historical_proxies,
        evidence=spec.evidence,
    )
