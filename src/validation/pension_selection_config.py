"""Pension selection preregistration: strict spec parsing without market reads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.data.paths import resolve_repo_path
from src.data.pension_market import load_pension_etf_identities

__all__ = [
    "PensionSelectionHistoricalSpec",
    "PensionSelectionSpec",
    "PensionSelectionTailSpec",
    "load_pension_selection_spec",
]

_WEIGHT_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class PensionSelectionTailSpec:
    """Pre-registered bootstrap sizes and stress-delta tail thresholds."""

    horizon_months: int
    n_paths: int
    block_months: int
    pre_retirement_months: int
    quantile: float
    min_stress_terminal_multiple: float
    max_pre_retirement_drawdown: float


@dataclass(frozen=True, slots=True)
class PensionSelectionHistoricalSpec:
    """Engine campaign inputs and the paired-ratio floor versus the baseline."""

    campaign_config_paths: tuple[str, ...]
    horizons_months: tuple[int, ...]
    worst_ratio_floor: float


@dataclass(frozen=True, slots=True)
class PensionSelectionSpec:
    """One validated, pre-registered pension-only selection experiment."""

    name: str
    etf_identity_path: str
    tax_regime_path: str
    estimation_start: date
    estimation_end: date
    max_fx_age_days: int
    max_fx_fallback_share: float
    baseline_arm_id: str
    arms: Mapping[str, Mapping[str, float]]
    market_weights: Mapping[str, float]
    anchor_ticker: str
    risk_free_annual: float
    equity_risk_premium_annual: float
    delta_prior_mean_annual: float
    delta_prior_sd_annual: float
    delta_grid_z: float
    delta_grid_points: int
    regret_tolerance_annual: float
    tail: PensionSelectionTailSpec
    historical: PensionSelectionHistoricalSpec


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _required_object(document: object, *, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(document, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(keys - set(document))
    extra = sorted(set(document) - keys)
    if missing:
        raise ValueError(f"{name} missing fields: {missing}")
    if extra:
        raise ValueError(f"{name} has unknown fields: {extra}")
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
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date, got {value!r}") from exc


def _parse_path(value: object, *, name: str) -> str:
    raw = _nonblank_string(value, name=name)
    try:
        resolved = resolve_repo_path(raw)
    except FileNotFoundError as exc:
        raise ValueError(f"{name} does not exist: {raw}") from exc
    return str(resolved)


def _parse_weights(value: object, *, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    weights: dict[str, float] = {}
    for ticker, raw_weight in value.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f"{name} ticker keys must be non-blank strings")
        weight = _finite_number(raw_weight, name=f"{name}[{ticker!r}]")
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"{name}[{ticker!r}] must lie in (0, 1]")
        weights[ticker] = weight
    total = math.fsum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to 1, got {total!r}")
    return weights


def load_pension_selection_spec(path: str | Path) -> PensionSelectionSpec:
    """Parse a pension selection preregistration with strict field and date validation.

    Args:
        path: Versioned selection definition.

    Returns:
        Existing typed selection specification.

    Raises:
        ValueError: If a gate, arm, date, weight, or identity field is invalid.
    """
    raw_document = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(raw_document, dict):
        raise ValueError("pension selection config must be an object")
    document: dict[str, object] = raw_document
    if "notes" in document and not isinstance(document["notes"], str):
        raise ValueError("notes must be a string")
    required_without_notes = {
        "name",
        "etf_identity_path",
        "tax_regime_path",
        "estimation_start",
        "estimation_end",
        "max_fx_age_days",
        "max_fx_fallback_share",
        "baseline_arm_id",
        "arms",
        "capm",
        "delta_prior",
        "delta_grid",
        "regret_tolerance_annual",
        "tail",
        "historical",
    }
    missing = sorted(required_without_notes - set(document))
    if missing:
        raise ValueError(f"pension selection config missing fields: {missing}")
    extra = sorted(set(document) - required_without_notes - {"notes"})
    if extra:
        raise ValueError(f"pension selection config has unknown fields: {extra}")

    name = _nonblank_string(document["name"], name="name")
    identity_path = _parse_path(document["etf_identity_path"], name="etf_identity_path")
    tax_path = _parse_path(document["tax_regime_path"], name="tax_regime_path")
    start = _parse_date(document["estimation_start"], name="estimation_start")
    end = _parse_date(document["estimation_end"], name="estimation_end")
    if start > end:
        raise ValueError("estimation_start must not be after estimation_end")
    max_fx_age_days = _positive_int(document["max_fx_age_days"], name="max_fx_age_days")
    max_fallback_share = _finite_number(
        document["max_fx_fallback_share"], name="max_fx_fallback_share"
    )
    if not 0.0 <= max_fallback_share <= 1.0:
        raise ValueError("max_fx_fallback_share must lie in [0, 1]")
    baseline_arm_id = _nonblank_string(document["baseline_arm_id"], name="baseline_arm_id")
    arms_raw = document["arms"]
    if not isinstance(arms_raw, dict):
        raise ValueError("arms must be an object")
    arms: dict[str, dict[str, float]] = {}
    for arm_id_raw, weights in arms_raw.items():
        arm_id = _nonblank_string(arm_id_raw, name="arm id")
        arms[arm_id] = _parse_weights(weights, name=f"arms[{arm_id!r}]")
    if not arms:
        raise ValueError("arms must be non-empty")
    if baseline_arm_id not in arms:
        raise ValueError(f"baseline_arm_id {baseline_arm_id!r} is not present in arms")
    ticker_union = {ticker for weights in arms.values() for ticker in weights}

    capm = _required_object(
        document["capm"],
        name="capm",
        keys={"market_weights", "anchor_ticker", "risk_free_annual", "equity_risk_premium_annual"},
    )
    market_weights = _parse_weights(capm["market_weights"], name="capm.market_weights")
    unknown_market = sorted(set(market_weights) - ticker_union)
    if unknown_market:
        raise ValueError(f"capm market tickers are outside the arm union: {unknown_market}")
    anchor_ticker = _nonblank_string(capm["anchor_ticker"], name="capm.anchor_ticker")
    if anchor_ticker not in ticker_union:
        raise ValueError(f"capm anchor ticker {anchor_ticker!r} is outside the arm union")
    risk_free_annual = _finite_number(capm["risk_free_annual"], name="capm.risk_free_annual")
    equity_risk_premium_annual = _finite_number(
        capm["equity_risk_premium_annual"], name="capm.equity_risk_premium_annual"
    )

    prior = _required_object(document["delta_prior"], name="delta_prior", keys={"mean_annual", "sd_annual"})
    prior_mean = _finite_number(prior["mean_annual"], name="delta_prior.mean_annual")
    prior_sd = _finite_number(prior["sd_annual"], name="delta_prior.sd_annual")
    if prior_sd <= 0.0:
        raise ValueError("delta_prior.sd_annual must be positive")
    grid = _required_object(document["delta_grid"], name="delta_grid", keys={"z", "n_points"})
    grid_z = _finite_number(grid["z"], name="delta_grid.z")
    if grid_z <= 0.0:
        raise ValueError("delta_grid.z must be positive")
    grid_points = _positive_int(grid["n_points"], name="delta_grid.n_points")
    if grid_points < 3 or grid_points % 2 == 0:
        raise ValueError("delta_grid.n_points must be odd and >= 3")
    regret_tolerance = _finite_number(
        document["regret_tolerance_annual"], name="regret_tolerance_annual"
    )
    if regret_tolerance < 0.0:
        raise ValueError("regret_tolerance_annual must be nonnegative")

    tail = _required_object(
        document["tail"],
        name="tail",
        keys={
            "horizon_months",
            "n_paths",
            "block_months",
            "pre_retirement_months",
            "quantile",
            "min_stress_terminal_multiple",
            "max_pre_retirement_drawdown",
        },
    )
    tail_spec = PensionSelectionTailSpec(
        horizon_months=_positive_int(tail["horizon_months"], name="tail.horizon_months"),
        n_paths=_positive_int(tail["n_paths"], name="tail.n_paths"),
        block_months=_positive_int(tail["block_months"], name="tail.block_months"),
        pre_retirement_months=_positive_int(
            tail["pre_retirement_months"], name="tail.pre_retirement_months"
        ),
        quantile=_finite_number(tail["quantile"], name="tail.quantile"),
        min_stress_terminal_multiple=_finite_number(
            tail["min_stress_terminal_multiple"], name="tail.min_stress_terminal_multiple"
        ),
        max_pre_retirement_drawdown=_finite_number(
            tail["max_pre_retirement_drawdown"], name="tail.max_pre_retirement_drawdown"
        ),
    )
    if not 0.0 < tail_spec.quantile < 0.5:
        raise ValueError("tail.quantile must lie in (0, 0.5)")
    if tail_spec.pre_retirement_months > tail_spec.horizon_months:
        raise ValueError("tail.pre_retirement_months must not exceed tail.horizon_months")

    historical = _required_object(
        document["historical"],
        name="historical",
        keys={"campaign_config_paths", "horizons_months", "worst_ratio_floor"},
    )
    campaign_paths_raw = historical["campaign_config_paths"]
    if not isinstance(campaign_paths_raw, list) or not campaign_paths_raw:
        raise ValueError("historical.campaign_config_paths must be a non-empty array")
    campaign_paths = tuple(
        _parse_path(value, name=f"historical.campaign_config_paths[{index}]")
        for index, value in enumerate(campaign_paths_raw)
    )
    if len(set(campaign_paths)) != len(campaign_paths):
        raise ValueError("historical.campaign_config_paths must be unique")
    horizons_raw = historical["horizons_months"]
    if not isinstance(horizons_raw, list) or not horizons_raw:
        raise ValueError("historical.horizons_months must be a non-empty array")
    horizons = tuple(
        _positive_int(value, name=f"historical.horizons_months[{index}]")
        for index, value in enumerate(horizons_raw)
    )
    if len(set(horizons)) != len(horizons):
        raise ValueError("historical.horizons_months must be unique")
    historical_spec = PensionSelectionHistoricalSpec(
        campaign_config_paths=campaign_paths,
        horizons_months=horizons,
        worst_ratio_floor=_finite_number(
            historical["worst_ratio_floor"], name="historical.worst_ratio_floor"
        ),
    )

    identities = load_pension_etf_identities(identity_path)
    identities_by_proxy: dict[str, list[object]] = {}
    for identity in identities:
        if identity.pension_eligible:
            identities_by_proxy.setdefault(identity.proxy_ticker, []).append(identity)
    for ticker in sorted(ticker_union):
        matches = identities_by_proxy.get(ticker, [])
        if len(matches) != 1:
            raise ValueError(
                f"arm ticker {ticker!r} requires exactly one pension-eligible Korean ETF identity; got {len(matches)}"
            )

    return PensionSelectionSpec(
        name=name,
        etf_identity_path=identity_path,
        tax_regime_path=tax_path,
        estimation_start=start,
        estimation_end=end,
        max_fx_age_days=max_fx_age_days,
        max_fx_fallback_share=max_fallback_share,
        baseline_arm_id=baseline_arm_id,
        arms=arms,
        market_weights=market_weights,
        anchor_ticker=anchor_ticker,
        risk_free_annual=risk_free_annual,
        equity_risk_premium_annual=equity_risk_premium_annual,
        delta_prior_mean_annual=prior_mean,
        delta_prior_sd_annual=prior_sd,
        delta_grid_z=grid_z,
        delta_grid_points=grid_points,
        regret_tolerance_annual=regret_tolerance,
        tail=tail_spec,
        historical=historical_spec,
    )
