"""Pre-registered standalone pension ETF selection verdict and evidence report."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, TypeVar

import polars as pl

from src.analytics.pension_selection import (
    DcaTailStats,
    DeltaEstimate,
    GrowthRegretTable,
    bootstrap_dca_tail,
    build_monthly_krw_panel,
    delta_grid,
    estimate_tilt_delta,
    fit_capm_scenario_model,
    growth_regret_table,
)
from src.data.catalog import load_visible
from src.data.pension_fx import build_krw_fx_series
from src.data.pension_market import load_pension_etf_identities
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.sim.pension_engine import PensionDataError, proxy_krw_marks
from src.sim.pension_tax import load_pension_tax_regime
from src.validation.pension_campaign import load_pension_campaign_spec, run_pension_campaign

logger = logging.getLogger(__name__)

__all__ = [
    "PensionArmVerdict",
    "PensionSelectionHistoricalSpec",
    "PensionSelectionReport",
    "PensionSelectionSpec",
    "PensionSelectionTailSpec",
    "SelectionStatus",
    "decide_pension_selection",
    "load_pension_selection_spec",
    "run_pension_selection",
    "write_pension_selection_report",
]

SelectionStatus = Literal["SELECTED", "NO_SELECTION"]
_WEIGHT_SUM_TOLERANCE = 1e-9
_T = TypeVar("_T")


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


@dataclass(frozen=True, slots=True)
class PensionArmVerdict:
    """Per-arm outcomes and ordered failure codes for all three gates."""

    arm_id: str
    ticker_count: int
    max_regret: float
    stress_low_quantile_terminal_multiple: float
    stress_high_quantile_pre_retirement_drawdown: float
    historical_worst_ratio: float | None
    regret_pass: bool
    tail_pass: bool
    historical_pass: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PensionSelectionReport:
    """Complete scenario, tail, historical, and final selection evidence."""

    name: str
    panel_start: date
    panel_end: date
    panel_months: int
    delta_estimate: DeltaEstimate
    growth_table: GrowthRegretTable
    stress_delta: float
    stress_tail: tuple[DcaTailStats, ...]
    central_tail: tuple[DcaTailStats, ...]
    historical_worst_ratios: Mapping[str, float | None]
    verdicts: tuple[PensionArmVerdict, ...]
    status: SelectionStatus
    selected_arm_id: str | None
    selected_kr_targets: Mapping[str, float]
    fx_provenance: Mapping[str, object]


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
    path = _nonblank_string(value, name=name)
    if not Path(path).is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


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
    """Load one strict, pre-registered pension ETF selection config.

    Raises:
        ValueError: On missing or unknown fields, invalid economic inputs, missing files, or
            any arm ticker without exactly one pension-eligible Korean ETF identity.
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


def _coverage_by_arm(rows: Sequence[_T], *, expected: set[str], label: str) -> dict[str, _T]:
    by_arm: dict[str, _T] = {}
    for row in rows:
        arm_id = getattr(row, "arm_id", None)
        if not isinstance(arm_id, str) or arm_id in by_arm:
            raise ValueError(f"{label} has a missing or duplicate arm id")
        by_arm[arm_id] = row
    if set(by_arm) != expected:
        raise ValueError(f"{label} must cover exactly the configured arms")
    return by_arm


def decide_pension_selection(
    spec: PensionSelectionSpec,
    growth_table: GrowthRegretTable,
    stress_tail: Sequence[DcaTailStats],
    historical_worst_ratios: Mapping[str, float | None],
) -> tuple[tuple[PensionArmVerdict, ...], str | None]:
    """Apply the pre-registered robustness, tail, and historical gates.

    Returns:
        Verdicts in config arm order and the selected arm id, or ``None`` when all fail.
    """
    expected = set(spec.arms)
    if growth_table.baseline_arm_id != spec.baseline_arm_id:
        raise ValueError("growth table baseline does not match the configured baseline")
    growth_by_arm = _coverage_by_arm(growth_table.rows, expected=expected, label="growth table")
    tail_by_arm = _coverage_by_arm(stress_tail, expected=expected, label="stress tail")
    if set(historical_worst_ratios) != expected:
        raise ValueError("historical ratios must cover exactly the configured arms")
    if not all(
        value is None
        or (not isinstance(value, bool) and isinstance(value, float | int) and math.isfinite(float(value)))
        for value in historical_worst_ratios.values()
    ):
        raise ValueError("historical ratios must be finite numbers or None")

    min_regret = min(row.max_regret for row in growth_by_arm.values())
    regret_ceiling = min_regret + spec.regret_tolerance_annual
    verdicts: list[PensionArmVerdict] = []
    for arm_id, targets in spec.arms.items():
        growth = growth_by_arm[arm_id]
        tail = tail_by_arm[arm_id]
        historical_worst = historical_worst_ratios[arm_id]
        reasons: list[str] = []
        regret_pass = growth.max_regret <= regret_ceiling
        if not regret_pass:
            reasons.append("REGRET_ABOVE_TOLERANCE")
        if tail.low_quantile_terminal_multiple < spec.tail.min_stress_terminal_multiple:
            reasons.append("STRESS_PRINCIPAL_TAIL")
        if tail.high_quantile_pre_retirement_drawdown > spec.tail.max_pre_retirement_drawdown:
            reasons.append("STRESS_PRE_RETIREMENT_DRAWDOWN")
        tail_pass = not any(
            reason in {"STRESS_PRINCIPAL_TAIL", "STRESS_PRE_RETIREMENT_DRAWDOWN"}
            for reason in reasons
        )
        if historical_worst is None:
            historical_pass = False
            reasons.append("NO_HISTORICAL_EVIDENCE")
        else:
            historical_pass = historical_worst >= spec.historical.worst_ratio_floor
            if not historical_pass:
                reasons.append("HISTORICAL_BELOW_FLOOR")
        verdict = PensionArmVerdict(
            arm_id=arm_id,
            ticker_count=len(targets),
            max_regret=growth.max_regret,
            stress_low_quantile_terminal_multiple=tail.low_quantile_terminal_multiple,
            stress_high_quantile_pre_retirement_drawdown=tail.high_quantile_pre_retirement_drawdown,
            historical_worst_ratio=historical_worst,
            regret_pass=regret_pass,
            tail_pass=tail_pass,
            historical_pass=historical_pass,
            reasons=tuple(reasons),
        )
        verdicts.append(verdict)
        logger.debug(
            "[PORTFOLIO] event=pension_selection_arm arm=%s regret_pass=%s tail_pass=%s historical_pass=%s reasons=%s",
            arm_id,
            regret_pass,
            tail_pass,
            historical_pass,
            ",".join(reasons) or "NONE",
        )

    passers = [verdict for verdict in verdicts if not verdict.reasons]
    selected = (
        min(
            passers,
            key=lambda verdict: (
                verdict.ticker_count,
                growth_by_arm[verdict.arm_id].volatility_annual,
                verdict.arm_id,
            ),
        ).arm_id
        if passers
        else None
    )
    status: SelectionStatus = "SELECTED" if selected is not None else "NO_SELECTION"
    logger.info(
        "[PORTFOLIO] event=pension_selection_verdict status=%s selected=%s arms=%d",
        status,
        selected or "NONE",
        len(verdicts),
    )
    return tuple(verdicts), selected


def _growth_payload(table: GrowthRegretTable) -> dict[str, object]:
    return {
        "deltas": list(table.deltas),
        "baseline_arm_id": table.baseline_arm_id,
        "rows": [
            {
                "arm_id": row.arm_id,
                "volatility_annual": row.volatility_annual,
                "growth_by_delta": list(row.growth_by_delta),
                "max_regret": row.max_regret,
                "mean_regret": row.mean_regret,
            }
            for row in table.rows
        ],
        "breakeven_vs_baseline": dict(table.breakeven_vs_baseline),
    }


def _tail_payload(rows: Sequence[DcaTailStats]) -> list[dict[str, object]]:
    return [
        {
            "arm_id": row.arm_id,
            "delta": row.delta,
            "horizon_months": row.horizon_months,
            "n_paths": row.n_paths,
            "quantile": row.quantile,
            "low_quantile_terminal_multiple": row.low_quantile_terminal_multiple,
            "median_terminal_multiple": row.median_terminal_multiple,
            "high_quantile_pre_retirement_drawdown": row.high_quantile_pre_retirement_drawdown,
            "prob_below_principal": row.prob_below_principal,
        }
        for row in rows
    ]


def _verdict_payload(verdicts: Sequence[PensionArmVerdict]) -> list[dict[str, object]]:
    return [
        {
            "arm_id": verdict.arm_id,
            "ticker_count": verdict.ticker_count,
            "max_regret": verdict.max_regret,
            "stress_low_quantile_terminal_multiple": verdict.stress_low_quantile_terminal_multiple,
            "stress_high_quantile_pre_retirement_drawdown": verdict.stress_high_quantile_pre_retirement_drawdown,
            "historical_worst_ratio": verdict.historical_worst_ratio,
            "regret_pass": verdict.regret_pass,
            "tail_pass": verdict.tail_pass,
            "historical_pass": verdict.historical_pass,
            "reasons": list(verdict.reasons),
        }
        for verdict in verdicts
    ]


def _report_payload(report: PensionSelectionReport, provenance: Mapping[str, str]) -> dict[str, object]:
    delta = report.delta_estimate
    return {
        "name": report.name,
        "panel_start": report.panel_start.isoformat(),
        "panel_end": report.panel_end.isoformat(),
        "panel_months": report.panel_months,
        "delta_estimate": {
            "anchor_ticker": delta.anchor_ticker,
            "n_months": delta.n_months,
            "sample_alpha_annual": delta.sample_alpha_annual,
            "sample_alpha_se_annual": delta.sample_alpha_se_annual,
            "prior_mean_annual": delta.prior_mean_annual,
            "prior_sd_annual": delta.prior_sd_annual,
            "posterior_mean_annual": delta.posterior_mean_annual,
            "posterior_sd_annual": delta.posterior_sd_annual,
        },
        "growth_table": _growth_payload(report.growth_table),
        "stress_delta": report.stress_delta,
        "stress_tail": _tail_payload(report.stress_tail),
        "central_tail": _tail_payload(report.central_tail),
        "historical_worst_ratios": dict(report.historical_worst_ratios),
        "verdicts": _verdict_payload(report.verdicts),
        "status": report.status,
        "selected_arm_id": report.selected_arm_id,
        "selected_kr_targets": dict(report.selected_kr_targets),
        "fx_provenance": dict(report.fx_provenance),
        "provenance": dict(provenance),
    }


def _markdown(report: PensionSelectionReport) -> str:
    delta = report.delta_estimate
    lines = [
        f"# {report.name} — 연금 ETF 선택 판정",
        "",
        "## Decision",
        "",
        f"- status: `{report.status}`",
        f"- selected_arm_id: `{report.selected_arm_id or 'NONE'}`",
        f"- selected_kr_targets: `{dict(report.selected_kr_targets)}`",
        f"- panel: `{report.panel_start.isoformat()}` ~ `{report.panel_end.isoformat()}` ({report.panel_months}개월)",
        "",
        "## Delta Estimate",
        "",
        f"- anchor: `{delta.anchor_ticker}`",
        f"- n_months: `{delta.n_months}`",
        f"- prior_mean_annual: `{delta.prior_mean_annual:.8f}`",
        f"- prior_sd_annual: `{delta.prior_sd_annual:.8f}`",
        f"- sample_alpha_annual: `{delta.sample_alpha_annual:.8f}`",
        f"- sample_alpha_se_annual: `{delta.sample_alpha_se_annual:.8f}`",
        f"- posterior_mean_annual: `{delta.posterior_mean_annual:.8f}`",
        f"- posterior_sd_annual: `{delta.posterior_sd_annual:.8f}`",
        f"- delta_grid: `{list(report.growth_table.deltas)}`",
        "",
        "## Growth and Regret",
        "",
        "| arm_id | volatility_annual | max_regret | mean_regret | growth_by_delta |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report.growth_table.rows:
        growth = ", ".join(
            f"δ={scenario:.6f}: {value:.6f}"
            for scenario, value in zip(report.growth_table.deltas, row.growth_by_delta, strict=True)
        )
        lines.append(
            f"| {row.arm_id} | {row.volatility_annual:.6f} | {row.max_regret:.6f} | "
            f"{row.mean_regret:.6f} | {growth} |"
        )
    lines.extend(["", "### Breakeven vs Baseline", "", "| arm_id | breakeven_delta |", "|---|---:|"])
    for arm_id, value in report.growth_table.breakeven_vs_baseline.items():
        rendered = "N/A" if value is None else f"{value:.8f}"
        lines.append(f"| {arm_id} | {rendered} |")

    def _tail_section(title: str, rows: Sequence[DcaTailStats]) -> list[str]:
        section = [
            "",
            f"## {title}",
            "",
            (
                f"- horizon_months: `{rows[0].horizon_months}` · n_paths: `{rows[0].n_paths}` · "
                f"quantile: `{rows[0].quantile}`"
            ),
            "",
            "| arm_id | delta | low_terminal_multiple | median_terminal_multiple | high_drawdown | prob_below_principal |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        section.extend(
            f"| {row.arm_id} | {row.delta:.8f} | {row.low_quantile_terminal_multiple:.6f} | "
            f"{row.median_terminal_multiple:.6f} | {row.high_quantile_pre_retirement_drawdown:.6f} | "
            f"{row.prob_below_principal:.6f} |"
            for row in rows
        )
        return section

    lines.extend(_tail_section("Stress Tail", report.stress_tail))
    lines.extend(_tail_section("Central Tail", report.central_tail))
    lines.extend(
        [
            "",
            "## Historical After-tax Gate",
            "",
            "| arm_id | worst_ratio |",
            "|---|---:|",
        ]
    )
    for arm_id, value in report.historical_worst_ratios.items():
        lines.append(f"| {arm_id} | {'N/A' if value is None else f'{value:.6f}'} |")
    lines.extend(
        [
            "",
            "## Verdicts",
            "",
            "| arm_id | max_regret | stress_terminal | stress_drawdown | historical | regret_pass | tail_pass | historical_pass | reasons |",
            "|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for verdict in report.verdicts:
        historical = "N/A" if verdict.historical_worst_ratio is None else f"{verdict.historical_worst_ratio:.6f}"
        lines.append(
            f"| {verdict.arm_id} | {verdict.max_regret:.6f} | "
            f"{verdict.stress_low_quantile_terminal_multiple:.6f} | "
            f"{verdict.stress_high_quantile_pre_retirement_drawdown:.6f} | {historical} | "
            f"{verdict.regret_pass} | {verdict.tail_pass} | {verdict.historical_pass} | "
            f"{','.join(verdict.reasons) or 'NONE'} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- SOXX의 2021-06-21 이전 관측은 서로 다른 지수 레짐을 반영하며 break flag를 유지한다.",
            "- CAPM 시나리오는 beta에 보상이 있다는 가정과 표본 불확실성을 노출한 분석이다.",
            "- 본 결과는 시뮬레이션이며 투자 권유가 아니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _cutoff(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 23, 59, tzinfo=UTC)


def _historical_worst_ratios(
    spec: PensionSelectionSpec,
    settings: DataSettings,
    *,
    seed: int,
) -> dict[str, float | None]:
    worst: dict[str, float | None] = dict.fromkeys(spec.arms)
    undefined: set[str] = set()
    allowed_horizons = set(spec.historical.horizons_months)
    for campaign_path in spec.historical.campaign_config_paths:
        campaign_spec = load_pension_campaign_spec(campaign_path)
        if campaign_spec.household is not None:
            raise ValueError(f"historical campaign {campaign_path} must not define household_view")
        if campaign_spec.baseline_arm_id != spec.baseline_arm_id:
            raise ValueError(
                f"historical campaign {campaign_path} baseline {campaign_spec.baseline_arm_id!r} "
                f"differs from selection baseline {spec.baseline_arm_id!r}"
            )
        campaign_arms = {arm.arm_id: arm for arm in campaign_spec.arms}
        for arm_id in spec.arms.keys() & campaign_arms.keys():
            if dict(spec.arms[arm_id]) != dict(campaign_arms[arm_id].targets):
                raise ValueError(
                    f"historical campaign {campaign_path} arm {arm_id!r} targets differ from selection targets"
                )
        report = run_pension_campaign(campaign_spec, settings, seed=seed)
        for summary in report.summaries:
            arm_id = summary.arm_id
            if arm_id not in worst or summary.horizon_months not in allowed_horizons:
                continue
            ratio = summary.worst_wealth_ratio
            if ratio is None:
                undefined.add(arm_id)
                worst[arm_id] = None
            elif arm_id not in undefined:
                current = worst[arm_id]
                worst[arm_id] = ratio if current is None else min(current, ratio)
    return worst


def run_pension_selection(
    spec: PensionSelectionSpec,
    settings: DataSettings,
    *,
    seed: int,
) -> PensionSelectionReport:
    """Build all scenario, tail, and historical evidence and return the fixed verdict.

    Returns:
        The complete report, including Korean ETF targets when one arm passes every gate.
    Raises:
        PensionSelectionDataError: If a synchronized monthly return panel cannot be built.
        PensionDataError: If certified market or FX data are absent, stale, or over the FX cap.
        ValueError: If tax, scenario, tail, identity, or historical campaign inputs are invalid.
        OSError: If a configured tax or identity source cannot be read.
    """
    regime = load_pension_tax_regime(spec.tax_regime_path)
    cutoff = _cutoff(spec.estimation_end)
    try:
        prices = load_visible(settings, Dataset.PRICES, cutoff)
        fx_base = load_visible(settings, Dataset.FX_KRW_BASE, cutoff)
    except UntrustedDatasetError as exc:
        raise PensionDataError(f"pension selection source is absent or stale: {exc}") from exc
    try:
        fallback = load_visible(settings, Dataset.FX, cutoff)
    except UntrustedDatasetError:
        fallback = None
    try:
        fx_series = build_krw_fx_series(fx_base, fallback)
    except ValueError as exc:
        raise PensionDataError(f"pension selection fx series is invalid: {exc}") from exc

    tickers = sorted({ticker for targets in spec.arms.values() for ticker in targets})
    window_prices = prices.filter(
        pl.col("ticker").is_in(tickers)
        & pl.col("date").is_between(spec.estimation_start, spec.estimation_end)
    )
    sessions = sorted(set(window_prices.get_column("date").to_list()))
    fallback_sessions = fx_series.fallback_session_dates(sessions)
    fallback_share = len(fallback_sessions) / len(sessions) if sessions else 0.0
    if fallback_share > spec.max_fx_fallback_share:
        raise PensionDataError(
            f"pension selection fx fallback share {fallback_share:.6f} exceeds cap "
            f"{spec.max_fx_fallback_share:.6f} (source {fx_series.fallback_source})"
        )
    fx_provenance = dict(fx_series.provenance(sessions))
    logger.info(
        "[DATA] event=pension_selection_fx_provenance status=%s fallback_sessions=%d share=%.6f",
        str(fx_series.status.value),
        len(fallback_sessions),
        fallback_share,
    )

    marks = proxy_krw_marks(
        window_prices,
        fx_series.frame,
        max_fx_age_days=spec.max_fx_age_days,
        withholding_rate=regime.foreign_dividend_withholding_rate,
    )
    panel = build_monthly_krw_panel(
        marks,
        tickers=tickers,
        start=spec.estimation_start,
        end=spec.estimation_end,
    )
    model = fit_capm_scenario_model(
        panel,
        market_weights=spec.market_weights,
        anchor_ticker=spec.anchor_ticker,
        risk_free_annual=spec.risk_free_annual,
        equity_risk_premium_annual=spec.equity_risk_premium_annual,
    )
    delta_estimate = estimate_tilt_delta(
        panel,
        model,
        prior_mean_annual=spec.delta_prior_mean_annual,
        prior_sd_annual=spec.delta_prior_sd_annual,
    )
    deltas = delta_grid(
        delta_estimate,
        z=spec.delta_grid_z,
        n_points=spec.delta_grid_points,
    )
    growth = growth_regret_table(
        model,
        spec.arms,
        deltas,
        baseline_arm_id=spec.baseline_arm_id,
    )
    stress_tail = bootstrap_dca_tail(
        panel,
        model,
        spec.arms,
        delta=deltas[0],
        horizon_months=spec.tail.horizon_months,
        n_paths=spec.tail.n_paths,
        block_months=spec.tail.block_months,
        pre_retirement_months=spec.tail.pre_retirement_months,
        quantile=spec.tail.quantile,
        seed=seed,
    )
    central_tail = bootstrap_dca_tail(
        panel,
        model,
        spec.arms,
        delta=delta_estimate.posterior_mean_annual,
        horizon_months=spec.tail.horizon_months,
        n_paths=spec.tail.n_paths,
        block_months=spec.tail.block_months,
        pre_retirement_months=spec.tail.pre_retirement_months,
        quantile=spec.tail.quantile,
        seed=seed,
    )
    historical_worst = _historical_worst_ratios(spec, settings, seed=seed)
    verdicts, selected_arm_id = decide_pension_selection(
        spec,
        growth,
        stress_tail,
        historical_worst,
    )
    status: SelectionStatus = "SELECTED" if selected_arm_id is not None else "NO_SELECTION"
    selected_kr_targets: dict[str, float] = {}
    if selected_arm_id is not None:
        identity_by_proxy = {
            identity.proxy_ticker: identity.ticker
            for identity in load_pension_etf_identities(spec.etf_identity_path)
        }
        selected_kr_targets = {
            identity_by_proxy[proxy]: weight
            for proxy, weight in spec.arms[selected_arm_id].items()
        }
    return PensionSelectionReport(
        name=spec.name,
        panel_start=spec.estimation_start,
        panel_end=spec.estimation_end,
        panel_months=len(panel.returns),
        delta_estimate=delta_estimate,
        growth_table=growth,
        stress_delta=deltas[0],
        stress_tail=stress_tail,
        central_tail=central_tail,
        historical_worst_ratios=historical_worst,
        verdicts=verdicts,
        status=status,
        selected_arm_id=selected_arm_id,
        selected_kr_targets=selected_kr_targets,
        fx_provenance=fx_provenance,
    )


def write_pension_selection_report(
    report: PensionSelectionReport,
    settings: DataSettings,
    *,
    experiment_id: str,
    provenance: Mapping[str, str],
) -> Path:
    """Persist JSON evidence and a Korean-readable Markdown decision record.

    Returns:
        The JSON report path.
    Raises:
        OSError: If the result artifact or Markdown sidecar cannot be written.
    """
    from src.data.result_store import ResultKind, write_result

    reference = write_result(
        settings,
        experiment=report.name,
        kind=ResultKind.SELECTION,
        run_id=experiment_id,
        payload=_report_payload(report, provenance),
        markdown=_markdown(report),
    )
    return reference.json_path
