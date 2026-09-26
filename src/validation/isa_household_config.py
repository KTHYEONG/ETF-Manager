"""Pre-registered ISA household operating-policy decision config."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.data.paths import resolve_repo_path
from src.sim.isa_household import HouseholdProfile, IncomePhase, IsaArm, IsaOperatingMode
from src.sim.isa_tax import IsaTaxClass

__all__ = [
    "IsaHouseholdLineage",
    "IsaHouseholdSpec",
    "load_isa_household_spec",
]

_ISA_ANNUAL_LIMIT_KRW: int = 20_000_000

_EXPECTED_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "pension_decision_config_path",
        "pension_record_path",
        "isa_tax_regime_path",
        "pension_tax_regime_path",
        "overseas_tax_regime_path",
        "plan_start_year",
        "horizons_years",
        "step_months",
        "pension_annual_krw",
        "isa_budgets_krw",
        "annuity_drawing_years",
        "sensitivity_drawing_years",
        "baseline_arm_id",
        "arms",
        "profiles",
        "equivalence_band",
        "min_bootstrap_win_share",
        "bootstrap_paths",
        "bootstrap_block_months",
        "lineage",
        "notes",
    }
)


@dataclass(frozen=True, slots=True)
class IsaHouseholdLineage:
    """Trial-multiplicity disclosure copied into the report."""

    related_trial_count: int
    related_trials: tuple[str, ...]
    first_test_date: date
    post_hoc_disclosure: str


@dataclass(frozen=True, slots=True)
class IsaHouseholdSpec:
    """Pre-registered ISA household operating-policy decision.

    ``arms`` preserves the JSON object order, which is the tie-break preference (simplest, most liquid
    first) among arms within the equivalence band.
    """

    name: str
    pension_decision_config_path: str
    pension_record_path: str
    isa_tax_regime_path: str
    pension_tax_regime_path: str
    overseas_tax_regime_path: str
    plan_start_year: int
    horizons_years: tuple[int, ...]
    step_months: int
    pension_annual_krw: int
    isa_budgets_krw: tuple[int, ...]
    annuity_drawing_years: int
    sensitivity_drawing_years: tuple[int, ...]
    baseline_arm_id: str
    arms: Mapping[str, IsaArm]
    profiles: tuple[HouseholdProfile, ...]
    equivalence_band: float
    min_bootstrap_win_share: float
    bootstrap_paths: int
    bootstrap_block_months: int
    lineage: IsaHouseholdLineage
    notes: str


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _nonblank_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value.strip()


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float | int) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _amount(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer amount of KRW")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_date(value: object, *, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date, got {value!r}") from exc


def _parse_existing_path(value: object, *, name: str) -> str:
    raw = _nonblank_string(value, name=name)
    try:
        resolved = resolve_repo_path(raw)
    except FileNotFoundError as exc:
        raise ValueError(f"{name} does not exist: {raw}") from exc
    return str(resolved)


def _parse_arm(arm_id: str, value: object) -> IsaArm:
    if not isinstance(value, dict):
        raise ValueError(f"arms[{arm_id!r}] must be an object")
    extra = sorted(set(value) - {"arm_id", "mode", "cycle_years"})
    if extra:
        raise ValueError(f"arms[{arm_id!r}] has unknown fields: {extra}")
    for field in ("arm_id", "mode", "cycle_years"):
        if field not in value:
            raise ValueError(f"arms[{arm_id!r}] missing field: {field}")
    declared = _nonblank_string(value["arm_id"], name=f"arms[{arm_id!r}].arm_id")
    if declared != arm_id:
        raise ValueError(f"arm id {declared!r} differs from its object key {arm_id!r}")
    raw_mode = _nonblank_string(value["mode"], name=f"arms[{arm_id!r}].mode")
    try:
        mode = IsaOperatingMode(raw_mode)
    except ValueError as exc:
        raise ValueError(f"arms[{arm_id!r}].mode is unknown: {raw_mode!r}") from exc
    raw_cycle = value["cycle_years"]
    cycle: int | None
    if raw_cycle is None:
        cycle = None
    elif isinstance(raw_cycle, bool) or not isinstance(raw_cycle, int):
        raise ValueError(f"arms[{arm_id!r}].cycle_years must be an integer or null")
    else:
        cycle = raw_cycle
    try:
        return IsaArm(arm_id=declared, mode=mode, cycle_years=cycle)
    except ValueError as exc:
        raise ValueError(f"arms[{arm_id!r}] is invalid: {exc}") from exc


def _parse_phase(value: object, *, name: str) -> IncomePhase:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    extra = sorted(
        set(value)
        - {
            "first_plan_year_offset",
            "income_kind",
            "annual_income_krw",
            "remaining_national_tax_krw",
            "remaining_local_tax_krw",
        }
    )
    if extra:
        raise ValueError(f"{name} has unknown fields: {extra}")
    for field in (
        "first_plan_year_offset",
        "income_kind",
        "annual_income_krw",
        "remaining_national_tax_krw",
        "remaining_local_tax_krw",
    ):
        if field not in value:
            raise ValueError(f"{name} missing field: {field}")
    offset = value["first_plan_year_offset"]
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError(f"{name}.first_plan_year_offset must be an integer")
    kind = value["income_kind"]
    if kind not in ("wage", "comprehensive"):
        raise ValueError(f"{name}.income_kind is unsupported: {kind!r}")
    amounts = {}
    for field in ("annual_income_krw", "remaining_national_tax_krw", "remaining_local_tax_krw"):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"{name}.{field} must be an integer amount of KRW")
        if raw < 0:
            raise ValueError(f"{name}.{field} must be nonnegative")
        amounts[field] = raw
    try:
        return IncomePhase(
            first_plan_year_offset=offset,
            income_kind=kind,
            annual_income_krw=amounts["annual_income_krw"],
            remaining_national_tax_krw=amounts["remaining_national_tax_krw"],
            remaining_local_tax_krw=amounts["remaining_local_tax_krw"],
        )
    except ValueError as exc:
        raise ValueError(f"{name} is invalid: {exc}") from exc


def _parse_profile(value: object, *, index: int) -> HouseholdProfile:
    name = f"profiles[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    extra = sorted(
        set(value) - {"profile_id", "birth_date", "pension_account_open_date", "initial_isa_tax_class", "phases"}
    )
    if extra:
        raise ValueError(f"{name} has unknown fields: {extra}")
    for field in ("profile_id", "birth_date", "pension_account_open_date", "initial_isa_tax_class", "phases"):
        if field not in value:
            raise ValueError(f"{name} missing field: {field}")
    profile_id = _nonblank_string(value["profile_id"], name=f"{name}.profile_id")
    birth = _parse_date(value["birth_date"], name=f"{name}.birth_date")
    opened = _parse_date(value["pension_account_open_date"], name=f"{name}.pension_account_open_date")
    raw_class = _nonblank_string(value["initial_isa_tax_class"], name=f"{name}.initial_isa_tax_class")
    try:
        tax_class = IsaTaxClass(raw_class)
    except ValueError as exc:
        raise ValueError(f"{name}.initial_isa_tax_class is unknown: {raw_class!r}") from exc
    raw_phases = value["phases"]
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError(f"{name}.phases must be a non-empty array")
    phases = tuple(_parse_phase(entry, name=f"{name}.phases[{position}]") for position, entry in enumerate(raw_phases))
    try:
        return HouseholdProfile(
            profile_id=profile_id,
            birth_date=birth,
            pension_account_open_date=opened,
            initial_isa_tax_class=tax_class,
            phases=phases,
        )
    except ValueError as exc:
        raise ValueError(f"{name} is invalid: {exc}") from exc


def _parse_lineage(value: object) -> IsaHouseholdLineage:
    if not isinstance(value, dict):
        raise ValueError("lineage must be an object")
    extra = sorted(set(value) - {"related_trial_count", "related_trials", "first_test_date", "post_hoc_disclosure"})
    if extra:
        raise ValueError(f"lineage has unknown fields: {extra}")
    for field in ("related_trial_count", "related_trials", "first_test_date", "post_hoc_disclosure"):
        if field not in value:
            raise ValueError(f"lineage missing field: {field}")
    count = _nonnegative_int(value["related_trial_count"], name="lineage.related_trial_count")
    raw_trials = value["related_trials"]
    if not isinstance(raw_trials, list):
        raise ValueError("lineage.related_trials must be an array")
    trials = tuple(_nonblank_string(entry, name="lineage.related_trials entry") for entry in raw_trials)
    first = _parse_date(value["first_test_date"], name="lineage.first_test_date")
    disclosure = value["post_hoc_disclosure"]
    if not isinstance(disclosure, str):
        raise ValueError("lineage.post_hoc_disclosure must be a string")
    return IsaHouseholdLineage(
        related_trial_count=count,
        related_trials=trials,
        first_test_date=first,
        post_hoc_disclosure=disclosure,
    )


def load_isa_household_spec(path: str | Path) -> IsaHouseholdSpec:
    """Load and validate a pre-registered ISA household decision config.

    Raises: ValueError on a key-set mismatch, duplicate JSON keys, a missing referenced file, an arm id
        that differs from its object key, a baseline not among arms, duplicate profile ids, a non-positive
        or non-integer amount, a budget above 20,000,000 KRW (the ISA annual limit checked again at
        simulation), a horizon below 3 years, a primary drawing horizon repeated in the sensitivity list,
        an equivalence band outside (0, 0.05], a win share outside (0, 1), or a non-positive bootstrap size.
    """
    try:
        document = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except ValueError as exc:
        if "duplicate JSON key" in str(exc):
            raise
        raise ValueError(f"ISA household config is unreadable: {path}") from exc
    except OSError as exc:
        raise ValueError(f"ISA household config is unreadable: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("ISA household config must be an object")
    keys = frozenset(document.keys())
    if keys != _EXPECTED_KEYS:
        raise ValueError(
            f"ISA household config has unexpected keys: missing={sorted(_EXPECTED_KEYS - keys)} "
            f"extra={sorted(keys - _EXPECTED_KEYS)}"
        )
    name = _nonblank_string(document["name"], name="name")
    pension_decision_config_path = _parse_existing_path(
        document["pension_decision_config_path"], name="pension_decision_config_path"
    )
    pension_record_path = _parse_existing_path(document["pension_record_path"], name="pension_record_path")
    isa_tax_regime_path = _parse_existing_path(document["isa_tax_regime_path"], name="isa_tax_regime_path")
    pension_tax_regime_path = _parse_existing_path(document["pension_tax_regime_path"], name="pension_tax_regime_path")
    overseas_tax_regime_path = _parse_existing_path(
        document["overseas_tax_regime_path"], name="overseas_tax_regime_path"
    )
    plan_start_year = document["plan_start_year"]
    if isinstance(plan_start_year, bool) or not isinstance(plan_start_year, int):
        raise ValueError("plan_start_year must be an integer year")
    raw_horizons = document["horizons_years"]
    if not isinstance(raw_horizons, list) or not raw_horizons:
        raise ValueError("horizons_years must be a non-empty array")
    horizons = tuple(_positive_int(entry, name=f"horizons_years[{index}]") for index, entry in enumerate(raw_horizons))
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons_years must be unique")
    for horizon in horizons:
        if horizon < 3:
            raise ValueError(f"horizons_years entry {horizon!r} is below 3 years")
    step_months = _positive_int(document["step_months"], name="step_months")
    pension_annual_krw = _amount(document["pension_annual_krw"], name="pension_annual_krw")
    raw_budgets = document["isa_budgets_krw"]
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ValueError("isa_budgets_krw must be a non-empty array")
    budgets = tuple(_amount(entry, name=f"isa_budgets_krw[{index}]") for index, entry in enumerate(raw_budgets))
    for budget in budgets:
        if budget > _ISA_ANNUAL_LIMIT_KRW:
            raise ValueError(f"isa_budgets_krw entry {budget!r} exceeds the ISA annual limit")
    annuity_drawing_years = _positive_int(document["annuity_drawing_years"], name="annuity_drawing_years")
    raw_sensitivity = document["sensitivity_drawing_years"]
    if not isinstance(raw_sensitivity, list):
        raise ValueError("sensitivity_drawing_years must be an array")
    sensitivity = tuple(
        _positive_int(entry, name=f"sensitivity_drawing_years[{index}]") for index, entry in enumerate(raw_sensitivity)
    )
    if annuity_drawing_years in sensitivity:
        raise ValueError("sensitivity_drawing_years must not repeat the primary drawing horizon")
    baseline_arm_id = _nonblank_string(document["baseline_arm_id"], name="baseline_arm_id")
    raw_arms = document["arms"]
    if not isinstance(raw_arms, dict) or not raw_arms:
        raise ValueError("arms must be a non-empty object")
    arms: dict[str, IsaArm] = {}
    for key, entry in raw_arms.items():
        arms[key] = _parse_arm(key, entry)
    if baseline_arm_id not in arms:
        raise ValueError(f"baseline_arm_id {baseline_arm_id!r} is not among arms")
    raw_profiles = document["profiles"]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("profiles must be a non-empty array")
    profiles = tuple(_parse_profile(entry, index=index) for index, entry in enumerate(raw_profiles))
    profile_ids = [profile.profile_id for profile in profiles]
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("profile ids must be unique")
    band = _finite_number(document["equivalence_band"], name="equivalence_band")
    if not 0.0 < band <= 0.05:
        raise ValueError("equivalence_band must lie in (0, 0.05]")
    win_share = _finite_number(document["min_bootstrap_win_share"], name="min_bootstrap_win_share")
    if not 0.0 < win_share < 1.0:
        raise ValueError("min_bootstrap_win_share must lie in (0, 1)")
    bootstrap_paths = _positive_int(document["bootstrap_paths"], name="bootstrap_paths")
    bootstrap_block_months = _positive_int(document["bootstrap_block_months"], name="bootstrap_block_months")
    lineage = _parse_lineage(document["lineage"])
    notes = document["notes"]
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")
    return IsaHouseholdSpec(
        name=name,
        pension_decision_config_path=pension_decision_config_path,
        pension_record_path=pension_record_path,
        isa_tax_regime_path=isa_tax_regime_path,
        pension_tax_regime_path=pension_tax_regime_path,
        overseas_tax_regime_path=overseas_tax_regime_path,
        plan_start_year=plan_start_year,
        horizons_years=horizons,
        step_months=step_months,
        pension_annual_krw=pension_annual_krw,
        isa_budgets_krw=budgets,
        annuity_drawing_years=annuity_drawing_years,
        sensitivity_drawing_years=sensitivity,
        baseline_arm_id=baseline_arm_id,
        arms=arms,
        profiles=profiles,
        equivalence_band=band,
        min_bootstrap_win_share=win_share,
        bootstrap_paths=bootstrap_paths,
        bootstrap_block_months=bootstrap_block_months,
        lineage=lineage,
        notes=notes,
    )
