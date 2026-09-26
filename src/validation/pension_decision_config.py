"""Pre-registered pension decision config: strict parsing without market reads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.sim.pension_monthly import WeightSchedule

__all__ = [
    "PensionDecisionSpec",
    "load_pension_decision_spec",
]


@dataclass(frozen=True, slots=True)
class PensionDecisionSpec:
    """Pre-registered pension decision: candidates, evidence tiers, objective, and adoption rule."""

    name: str
    benchmark_id: str
    candidates: Mapping[str, WeightSchedule]
    neighbors: Mapping[str, tuple[str, ...]]
    century_series: Mapping[str, str]
    modern_start: date
    modern_end: date
    century_start: date
    century_end: date
    horizons_years: tuple[int, ...]
    step_months: int
    pre_retirement_months: int
    primary_gamma: float
    sensitivity_gammas: tuple[float, ...]
    equivalence_band: float
    min_bootstrap_win_share: float
    bootstrap_paths: int
    bootstrap_block_months: int
    annual_drag_by_sleeve: Mapping[str, float]
    tax_crosscheck_campaign_path: str
    tax_crosscheck_arm_map: Mapping[str, str]
    review_every_months: int
    lineage: Mapping[str, object]


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


def _parse_date(value: object, *, name: str) -> date:
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date, got {value!r}") from exc
    return parsed


def _parse_path(value: object, *, name: str) -> str:
    path = _nonblank_string(value, name=name)
    if not Path(path).is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def _parse_schedule(value: object, *, name: str) -> WeightSchedule:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object with start_weights, end_weights, and glide_years")
    for key in ("start_weights", "end_weights", "glide_years"):
        if key not in value:
            raise ValueError(f"{name} missing field: {key}")
    extra = sorted(set(value) - {"start_weights", "end_weights", "glide_years"})
    if extra:
        raise ValueError(f"{name} has unknown fields: {extra}")
    start = _parse_share_map(value["start_weights"], name=f"{name}.start_weights")
    end = _parse_share_map(value["end_weights"], name=f"{name}.end_weights")
    glide = value["glide_years"]
    if isinstance(glide, bool) or not isinstance(glide, int) or glide < 0:
        raise ValueError(f"{name}.glide_years must be a non-negative integer")
    try:
        return WeightSchedule(start_weights=start, end_weights=end, glide_years=glide)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid weight schedule: {exc}") from exc


def _parse_share_map(value: object, *, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    weights: dict[str, float] = {}
    for sleeve, raw_weight in value.items():
        if not isinstance(sleeve, str) or not sleeve.strip():
            raise ValueError(f"{name} sleeve keys must be non-blank strings")
        weight = _finite_number(raw_weight, name=f"{name}[{sleeve!r}]")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"{name}[{sleeve!r}] must lie in [0, 1]")
        weights[sleeve] = weight
    total = math.fsum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to 1, got {total!r}")
    return weights


def load_pension_decision_spec(path: str | Path) -> PensionDecisionSpec:
    """Parse and validate the versioned decision config.

    Raises:
        ValueError: On unknown keys, a benchmark not among candidates, neighbors
            referencing unknown ids, a candidate sleeve without a century series,
            non-positive gammas, a negative band, a win share outside (0, 1],
            invalid dates, or empty candidates.
    """
    document = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(document, dict):
        raise ValueError("pension decision config must be an object")
    allowed = {
        "name",
        "benchmark_id",
        "candidates",
        "neighbors",
        "century_series",
        "modern_start",
        "modern_end",
        "century_start",
        "century_end",
        "horizons_years",
        "step_months",
        "pre_retirement_months",
        "primary_gamma",
        "sensitivity_gammas",
        "equivalence_band",
        "min_bootstrap_win_share",
        "bootstrap_paths",
        "bootstrap_block_months",
        "annual_drag_by_sleeve",
        "tax_crosscheck_campaign_path",
        "tax_crosscheck_arm_map",
        "review_every_months",
        "lineage",
        "notes",
    }
    missing = sorted(allowed - {"notes"} - set(document))
    if missing:
        raise ValueError(f"pension decision config missing fields: {missing}")
    extra = sorted(set(document) - allowed)
    if extra:
        raise ValueError(f"pension decision config has unknown fields: {extra}")
    if "notes" in document and not isinstance(document["notes"], str):
        raise ValueError("notes must be a string")

    name = _nonblank_string(document["name"], name="name")
    raw_candidates = document["candidates"]
    if not isinstance(raw_candidates, dict) or not raw_candidates:
        raise ValueError("candidates must be a non-empty object")
    candidates: dict[str, WeightSchedule] = {}
    for raw_id, raw_schedule in raw_candidates.items():
        candidate_id = _nonblank_string(raw_id, name="candidate id")
        candidates[candidate_id] = _parse_schedule(raw_schedule, name=f"candidates[{candidate_id!r}]")
    benchmark_id = _nonblank_string(document["benchmark_id"], name="benchmark_id")
    if benchmark_id not in candidates:
        raise ValueError(f"benchmark_id {benchmark_id!r} is not among candidates")

    raw_neighbors = document["neighbors"]
    if not isinstance(raw_neighbors, dict):
        raise ValueError("neighbors must be an object")
    neighbors: dict[str, tuple[str, ...]] = {}
    for raw_id, raw_list in raw_neighbors.items():
        candidate_id = _nonblank_string(raw_id, name="neighbor id")
        if candidate_id not in candidates:
            raise ValueError(f"neighbors key {candidate_id!r} is not among candidates")
        if not isinstance(raw_list, list):
            raise ValueError(f"neighbors[{candidate_id!r}] must be an array")
        declared: list[str] = []
        for entry in raw_list:
            neighbor_id = _nonblank_string(entry, name=f"neighbors[{candidate_id!r}] entry")
            if neighbor_id not in candidates:
                raise ValueError(f"neighbors[{candidate_id!r}] references unknown id {neighbor_id!r}")
            if neighbor_id == candidate_id:
                raise ValueError(f"neighbors[{candidate_id!r}] must not list itself")
            declared.append(neighbor_id)
        if len(set(declared)) != len(declared):
            raise ValueError(f"neighbors[{candidate_id!r}] must be unique")
        neighbors[candidate_id] = tuple(declared)

    raw_series = document["century_series"]
    if not isinstance(raw_series, dict) or not raw_series:
        raise ValueError("century_series must be a non-empty object")
    century_series = {
        _nonblank_string(sleeve, name="century_series sleeve"): _nonblank_string(
            series, name=f"century_series[{sleeve!r}]"
        )
        for sleeve, series in raw_series.items()
    }
    for candidate_id, schedule in candidates.items():
        unmapped = sorted(set(schedule.start_weights) - set(century_series))
        if unmapped:
            raise ValueError(f"candidates[{candidate_id!r}] sleeves lack a century series: {unmapped}")

    modern_start = _parse_date(document["modern_start"], name="modern_start")
    modern_end = _parse_date(document["modern_end"], name="modern_end")
    if modern_start > modern_end:
        raise ValueError("modern_start must not be after modern_end")
    century_start = _parse_date(document["century_start"], name="century_start")
    century_end = _parse_date(document["century_end"], name="century_end")
    if century_start > century_end:
        raise ValueError("century_start must not be after century_end")

    raw_horizons = document["horizons_years"]
    if not isinstance(raw_horizons, list) or not raw_horizons:
        raise ValueError("horizons_years must be a non-empty array")
    horizons = tuple(_positive_int(value, name=f"horizons_years[{index}]") for index, value in enumerate(raw_horizons))
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons_years must be unique")
    step_months = _positive_int(document["step_months"], name="step_months")
    pre_months_raw = document["pre_retirement_months"]
    if isinstance(pre_months_raw, bool) or not isinstance(pre_months_raw, int) or pre_months_raw < 0:
        raise ValueError("pre_retirement_months must be a non-negative integer")
    if pre_months_raw > min(horizons) * 12:
        raise ValueError("pre_retirement_months must not exceed the shortest horizon")

    primary_gamma = _finite_number(document["primary_gamma"], name="primary_gamma")
    if primary_gamma <= 0.0:
        raise ValueError("primary_gamma must be positive")
    raw_sensitivity = document["sensitivity_gammas"]
    if not isinstance(raw_sensitivity, list):
        raise ValueError("sensitivity_gammas must be an array")
    sensitivity = tuple(_finite_number(value, name=f"sensitivity_gammas[{index}]") for index, value in enumerate(raw_sensitivity))
    for gamma in sensitivity:
        if gamma <= 0.0:
            raise ValueError("sensitivity_gammas must be positive")
    band = _finite_number(document["equivalence_band"], name="equivalence_band")
    if band < 0.0:
        raise ValueError("equivalence_band must be non-negative")
    win_share = _finite_number(document["min_bootstrap_win_share"], name="min_bootstrap_win_share")
    if not 0.0 < win_share <= 1.0:
        raise ValueError("min_bootstrap_win_share must lie in (0, 1]")
    bootstrap_paths = _positive_int(document["bootstrap_paths"], name="bootstrap_paths")
    bootstrap_blocks = _positive_int(document["bootstrap_block_months"], name="bootstrap_block_months")

    raw_drag = document["annual_drag_by_sleeve"]
    if not isinstance(raw_drag, dict):
        raise ValueError("annual_drag_by_sleeve must be an object")
    drags: dict[str, float] = {}
    for sleeve, raw_value in raw_drag.items():
        sleeve_id = _nonblank_string(sleeve, name="annual_drag_by_sleeve sleeve")
        drag = _finite_number(raw_value, name=f"annual_drag_by_sleeve[{sleeve_id!r}]")
        if not 0.0 <= drag < 1.0:
            raise ValueError(f"annual_drag_by_sleeve[{sleeve_id!r}] must lie in [0, 1)")
        drags[sleeve_id] = drag

    campaign_path = _parse_path(document["tax_crosscheck_campaign_path"], name="tax_crosscheck_campaign_path")
    raw_arm_map = document["tax_crosscheck_arm_map"]
    if not isinstance(raw_arm_map, dict) or not raw_arm_map:
        raise ValueError("tax_crosscheck_arm_map must be a non-empty object")
    arm_map = {
        _nonblank_string(arm, name="tax_crosscheck_arm_map arm"): _nonblank_string(
            candidate, name=f"tax_crosscheck_arm_map[{arm!r}]"
        )
        for arm, candidate in raw_arm_map.items()
    }
    for arm, candidate in arm_map.items():
        if candidate not in candidates:
            raise ValueError(f"tax_crosscheck_arm_map[{arm!r}] references unknown candidate {candidate!r}")
    review_every = _positive_int(document["review_every_months"], name="review_every_months")
    lineage = document["lineage"]
    if not isinstance(lineage, dict):
        raise ValueError("lineage must be an object")
    trial_count = lineage.get("related_trial_count")
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count < 0:
        raise ValueError("lineage.related_trial_count must be a non-negative integer")

    return PensionDecisionSpec(
        name=name,
        benchmark_id=benchmark_id,
        candidates=candidates,
        neighbors=neighbors,
        century_series=century_series,
        modern_start=modern_start,
        modern_end=modern_end,
        century_start=century_start,
        century_end=century_end,
        horizons_years=horizons,
        step_months=step_months,
        pre_retirement_months=pre_months_raw,
        primary_gamma=primary_gamma,
        sensitivity_gammas=sensitivity,
        equivalence_band=band,
        min_bootstrap_win_share=win_share,
        bootstrap_paths=bootstrap_paths,
        bootstrap_block_months=bootstrap_blocks,
        annual_drag_by_sleeve=drags,
        tax_crosscheck_campaign_path=campaign_path,
        tax_crosscheck_arm_map=arm_map,
        review_every_months=review_every,
        lineage=dict(lineage),
    )
