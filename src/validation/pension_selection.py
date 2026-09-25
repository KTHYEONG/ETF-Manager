"""Pre-registered standalone pension ETF selection verdict and evidence report."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
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
from src.data.catalog import CatalogSnapshot, load_snapshot_visible, resolve_snapshot
from src.data.pension_fx import build_krw_fx_series
from src.data.pension_market import load_pension_etf_identities
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.sim.pension_engine import PensionDataError, proxy_krw_marks
from src.sim.pension_tax import load_pension_tax_regime
from src.validation.pension_campaign import _optional_snapshot, load_pension_campaign_spec, run_pension_campaign
from src.validation.pension_selection_config import (
    PensionSelectionHistoricalSpec,
    PensionSelectionSpec,
    PensionSelectionTailSpec,
    load_pension_selection_spec,
)
from src.validation.pension_selection_report import write_pension_selection_report

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
    # 실행이 실제로 소비한 Silver의 normalized_sha256; None은 한 번도 게시되지 않은 선택 입력(부재)이다.
    manifest_hashes: Mapping[str, str | None] = field(default_factory=dict)


def _normalized_hash(snapshot: CatalogSnapshot | None, dataset: Dataset) -> str | None:
    return snapshot.artifacts[dataset].manifest.normalized_sha256 if snapshot is not None else None


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
        snapshot = resolve_snapshot(settings, (Dataset.PRICES, Dataset.FX_KRW_BASE))
        prices = load_snapshot_visible(snapshot, Dataset.PRICES, cutoff)
        fx_base = load_snapshot_visible(snapshot, Dataset.FX_KRW_BASE, cutoff)
    except UntrustedDatasetError as exc:
        raise PensionDataError(f"pension selection source is absent or stale: {exc}") from exc
    try:
        fallback_snapshot = _optional_snapshot(settings, Dataset.FX)
        fallback = (
            load_snapshot_visible(fallback_snapshot, Dataset.FX, cutoff) if fallback_snapshot is not None else None
        )
        # CPI는 계산에 쓰지 않지만 기존 결과 식별자(digest)에 포함되므로 같은 규칙으로 고정한다.
        cpi_snapshot = _optional_snapshot(settings, Dataset.CPI)
    except UntrustedDatasetError as exc:
        raise PensionDataError(f"pension selection optional source is damaged: {exc}") from exc
    manifest_hashes = {
        str(Dataset.PRICES): _normalized_hash(snapshot, Dataset.PRICES),
        str(Dataset.FX_KRW_BASE): _normalized_hash(snapshot, Dataset.FX_KRW_BASE),
        str(Dataset.FX): _normalized_hash(fallback_snapshot, Dataset.FX),
        str(Dataset.CPI): _normalized_hash(cpi_snapshot, Dataset.CPI),
    }
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
        manifest_hashes=manifest_hashes,
    )
