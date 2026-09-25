"""Preregistered weight-rule family for the after-tax cohort campaign."""

from __future__ import annotations

import calendar
import math
import statistics
from collections.abc import Mapping
from datetime import date, datetime

from src.data.universe import UniverseMembership, eligible_at
from src.features.pit_market import PitMarket
from src.policy.after_tax_rule_parse import AfterTaxRuleId, AfterTaxRuleSpec, parse_after_tax_rule_spec
from src.policy.targets import PolicyError
from src.policy.weight_rule import CASH_SLEEVE, WeightRule

__all__ = [
    "AfterTaxRuleId",
    "AfterTaxRuleSpec",
    "build_weight_rule",
    "parse_after_tax_rule_spec",
]


def _add_calendar_months(day: date, months: int) -> date:
    total = day.month - 1 + months
    year = day.year + total // 12
    month = total % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day))


def _k_month_return(market: PitMarket, ticker: str, as_of: datetime, months: int) -> float:
    closes = market.month_end_adjusted_closes(ticker, as_of, months + 1)
    first = closes[0]
    last = closes[-1]
    if first <= 0.0:
        raise ValueError(f"non-positive base close for {ticker!r}")
    return last / first - 1.0


def _hurdle_for(market: PitMarket, series: str, as_of: datetime, months: int) -> float:
    rate_pct = market.rate_percent(series, as_of)
    return (float(months) / 12.0) * (float(rate_pct) / 100.0)


def _above_sma(market: PitMarket, ticker: str, as_of: datetime, months: int) -> bool:
    closes = market.month_end_adjusted_closes(ticker, as_of, months)
    mean = sum(closes) / len(closes)
    return closes[-1] > mean


def _realized_vol_annual(market: PitMarket, ticker: str, as_of: datetime, window: int) -> float:
    closes = market.daily_adjusted_closes(ticker, as_of, window + 1)
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    if len(returns) < 2:
        return 0.0
    return statistics.pstdev(returns) * math.sqrt(252.0)


def _eligible_universe(
    spec: AfterTaxRuleSpec, as_of: datetime, membership: UniverseMembership | None
) -> tuple[str, ...]:
    if membership is None:
        return spec.universe
    return eligible_at(membership, spec.universe, as_of.date())


def _eligible_safe_asset(
    spec: AfterTaxRuleSpec, as_of: datetime, membership: UniverseMembership | None
) -> str:
    assert spec.safe_asset is not None
    if membership is None or spec.safe_asset == CASH_SLEEVE:
        return spec.safe_asset
    if spec.safe_asset not in membership.entries or not eligible_at(
        membership, (spec.safe_asset,), as_of.date()
    ):
        raise PolicyError(
            f"safe asset {spec.safe_asset!r} is not eligible on {as_of.date().isoformat()}"
        )
    return spec.safe_asset


def _taa_weights(
    spec: AfterTaxRuleSpec,
    as_of: datetime,
    market: PitMarket,
    membership: UniverseMembership | None = None,
) -> dict[str, float]:
    assert spec.safe_asset is not None
    assert spec.top_n is not None
    assert spec.momentum_months
    universe = _eligible_universe(spec, as_of, membership)
    safe_asset = _eligible_safe_asset(spec, as_of, membership)
    if not universe:
        return {safe_asset: 1.0}
    rate_pct = market.rate_percent(spec.hurdle_rate_series, as_of)
    mean_k = sum(spec.momentum_months) / len(spec.momentum_months)
    hurdle = (mean_k / 12.0) * (rate_pct / 100.0)
    scored: list[tuple[str, float]] = []
    for ticker in universe:
        rets = [_k_month_return(market, ticker, as_of, k) for k in spec.momentum_months]
        scored.append((ticker, sum(rets) / len(rets)))
    scored.sort(key=lambda item: item[1], reverse=True)
    picks = scored[: spec.top_n]
    weights: dict[str, float] = {}
    slot = 1.0 / float(len(picks))
    for ticker, score in picks:
        target = ticker if score >= hurdle else safe_asset
        weights[target] = weights.get(target, 0.0) + slot
    return weights


class _AfterTaxWeightRule:
    """Concrete PIT rule dispatching on a parsed spec."""

    __slots__ = ("_horizon_end", "_membership", "_spec")

    def __init__(
        self,
        spec: AfterTaxRuleSpec,
        horizon_end: date,
        membership: UniverseMembership | None = None,
    ) -> None:
        self._spec = spec
        self._horizon_end = horizon_end
        self._membership = membership

    @property
    def tickers(self) -> frozenset[str]:
        spec = self._spec
        names: set[str] = set(spec.core_targets.keys()) | set(spec.universe)
        if spec.signal_ticker is not None:
            names.add(spec.signal_ticker)
        if spec.safe_asset is not None and spec.safe_asset != CASH_SLEEVE:
            names.add(spec.safe_asset)
        names.discard(CASH_SLEEVE)
        return frozenset(names)

    @property
    def requires_cash_rate(self) -> bool:
        spec = self._spec
        hurdle_rules = {
            AfterTaxRuleId.DUAL_MOMENTUM_EXIT,
            AfterTaxRuleId.TAA_TOP_N,
            AfterTaxRuleId.GEM,
            AfterTaxRuleId.CORE_SATELLITE_TAA,
        }
        if spec.rule_id in hurdle_rules:
            return True
        return spec.safe_asset == CASH_SLEEVE

    def __call__(self, signal_at: datetime, market: PitMarket) -> Mapping[str, float]:
        spec = self._spec
        if spec.rule_id is AfterTaxRuleId.STATIC:
            return dict(spec.core_targets)
        if spec.rule_id is AfterTaxRuleId.DUAL_MOMENTUM_EXIT:
            assert spec.signal_ticker is not None
            assert spec.sma_months is not None
            assert spec.momentum_months
            assert spec.safe_asset is not None
            if _above_sma(market, spec.signal_ticker, signal_at, spec.sma_months):
                return dict(spec.core_targets)
            k = spec.momentum_months[0]
            if _k_month_return(market, spec.signal_ticker, signal_at, k) >= _hurdle_for(
                market, spec.hurdle_rate_series, signal_at, k
            ):
                return dict(spec.core_targets)
            return {spec.safe_asset: 1.0}
        if spec.rule_id is AfterTaxRuleId.TREND_PARTIAL:
            assert spec.signal_ticker is not None
            assert spec.sma_months is not None
            assert spec.risk_on_fraction_when_off is not None
            assert spec.safe_asset is not None
            if _above_sma(market, spec.signal_ticker, signal_at, spec.sma_months):
                return dict(spec.core_targets)
            frac = spec.risk_on_fraction_when_off
            trend_weights = {ticker: weight * frac for ticker, weight in spec.core_targets.items()}
            trend_weights[spec.safe_asset] = trend_weights.get(spec.safe_asset, 0.0) + (1.0 - frac)
            return trend_weights
        if spec.rule_id is AfterTaxRuleId.VOL_TARGET:
            assert spec.signal_ticker is not None
            assert spec.vol_target_annual is not None
            assert spec.vol_window_sessions is not None
            assert spec.safe_asset is not None
            realized = _realized_vol_annual(market, spec.signal_ticker, signal_at, spec.vol_window_sessions)
            scale = 1.0 if realized <= 0.0 else min(1.0, spec.vol_target_annual / realized)
            vol_weights = {ticker: weight * scale for ticker, weight in spec.core_targets.items()}
            vol_weights[spec.safe_asset] = vol_weights.get(spec.safe_asset, 0.0) + (1.0 - scale)
            return vol_weights
        if spec.rule_id is AfterTaxRuleId.TAA_TOP_N:
            return _taa_weights(spec, signal_at, market, self._membership)
        if spec.rule_id is AfterTaxRuleId.GEM:
            assert spec.signal_ticker is not None
            assert spec.safe_asset is not None
            universe = _eligible_universe(spec, signal_at, self._membership)
            safe_asset = _eligible_safe_asset(spec, signal_at, self._membership)
            if not universe:
                return {safe_asset: 1.0}
            signal_ret = _k_month_return(market, spec.signal_ticker, signal_at, 12)
            if signal_ret < _hurdle_for(market, spec.hurdle_rate_series, signal_at, 12):
                return {safe_asset: 1.0}
            best_ticker = universe[0]
            best_ret = _k_month_return(market, best_ticker, signal_at, 12)
            for ticker in universe[1:]:
                ret = _k_month_return(market, ticker, signal_at, 12)
                if ret > best_ret:
                    best_ret = ret
                    best_ticker = ticker
            return {best_ticker: 1.0}
        if spec.rule_id is AfterTaxRuleId.CORE_SATELLITE_TAA:
            assert spec.satellite_weight is not None
            universe = _eligible_universe(spec, signal_at, self._membership)
            if not universe:
                return {_eligible_safe_asset(spec, signal_at, self._membership): 1.0}
            satellite = _taa_weights(spec, signal_at, market, self._membership)
            sat = spec.satellite_weight
            blended: dict[str, float] = {}
            for ticker, weight in spec.core_targets.items():
                blended[ticker] = blended.get(ticker, 0.0) + (1.0 - sat) * weight
            for ticker, weight in satellite.items():
                blended[ticker] = blended.get(ticker, 0.0) + sat * weight
            return blended
        assert spec.glide_months is not None
        assert spec.safe_asset is not None
        boundary = _add_calendar_months(self._horizon_end, -spec.glide_months)
        if signal_at.date() <= boundary:
            return dict(spec.core_targets)
        return {spec.safe_asset: 1.0}


def build_weight_rule(
    spec: AfterTaxRuleSpec,
    *,
    horizon_end: date,
    membership: UniverseMembership | None = None,
) -> WeightRule:
    """Instantiate a PIT ``WeightRule`` for one cohort.

    ``horizon_end`` is the cohort's last calendar day; only GLIDE_PATH reads it.
    When supplied, ``membership`` filters momentum universes at the signal date
    without exposing later delisting events.

    Rule semantics (all signals from ``PitMarket`` at the month-end signal instant):
    - STATIC: always ``core_targets``.
    - DUAL_MOMENTUM_EXIT: ``core_targets`` while ``signal_ticker`` is above its
      ``sma_months`` month-end mean OR its ``momentum_months[0]`` return beats the hurdle;
      otherwise 100% ``safe_asset``.
    - TREND_PARTIAL: ``core_targets`` when above the SMA; otherwise ``core_targets`` scaled
      by ``risk_on_fraction_when_off`` with the remainder in ``safe_asset``.
    - VOL_TARGET: ``core_targets`` scaled by ``min(1, vol_target_annual / realized_vol)``
      over ``vol_window_sessions`` daily adjusted returns (annualized √252), remainder in ``safe_asset``.
    - TAA_TOP_N: rank ``universe`` by the mean of the ``momentum_months`` returns; hold
      the top ``top_n`` equally; a pick whose score fails the hurdle is replaced by ``safe_asset``.
    - GEM: if the ``signal_ticker`` 12-month return fails the hurdle -> 100% ``safe_asset``;
      else 100% in the higher-momentum member of ``universe``.
    - CORE_SATELLITE_TAA: ``(1 - satellite_weight) x core_targets`` plus
      ``satellite_weight x`` the TAA_TOP_N weights.
    - GLIDE_PATH: ``core_targets`` until ``glide_months`` before ``horizon_end``; then 100%
      ``safe_asset`` (used with BUY_ONLY so only new money moves; never sells).

    Raises:
        ValueError: When required parameters are absent (defensive re-check).
        PolicyError: When a non-cash safe asset is not eligible under membership.
    """
    if not isinstance(horizon_end, date):
        raise ValueError(f"horizon_end must be a date, got {horizon_end!r}")
    rule_id = spec.rule_id
    if rule_id in (
        AfterTaxRuleId.STATIC,
        AfterTaxRuleId.DUAL_MOMENTUM_EXIT,
        AfterTaxRuleId.TREND_PARTIAL,
        AfterTaxRuleId.VOL_TARGET,
        AfterTaxRuleId.CORE_SATELLITE_TAA,
        AfterTaxRuleId.GLIDE_PATH,
    ) and not spec.core_targets:
        raise ValueError(f"{rule_id.value} requires core_targets")
    if rule_id is AfterTaxRuleId.DUAL_MOMENTUM_EXIT and (
        spec.signal_ticker is None or spec.sma_months is None or not spec.momentum_months or spec.safe_asset is None
    ):
        raise ValueError("DUAL_MOMENTUM_EXIT requires signal, sma, momentum, and safe_asset")
    if rule_id is AfterTaxRuleId.TREND_PARTIAL and (
        spec.signal_ticker is None
        or spec.sma_months is None
        or spec.risk_on_fraction_when_off is None
        or spec.safe_asset is None
    ):
        raise ValueError("TREND_PARTIAL requires signal, sma, fraction, and safe_asset")
    if rule_id is AfterTaxRuleId.VOL_TARGET and (
        spec.signal_ticker is None
        or spec.vol_target_annual is None
        or spec.vol_window_sessions is None
        or spec.safe_asset is None
    ):
        raise ValueError("VOL_TARGET requires signal, target, window, and safe_asset")
    if rule_id is AfterTaxRuleId.TAA_TOP_N and (
        not spec.universe or not spec.momentum_months or spec.top_n is None or spec.safe_asset is None
    ):
        raise ValueError("TAA_TOP_N requires universe, momentum, top_n, and safe_asset")
    if rule_id is AfterTaxRuleId.GEM and (
        spec.signal_ticker is None or not spec.universe or spec.safe_asset is None
    ):
        raise ValueError("GEM requires signal, universe, and safe_asset")
    if rule_id is AfterTaxRuleId.CORE_SATELLITE_TAA and (
        not spec.universe
        or not spec.momentum_months
        or spec.top_n is None
        or spec.satellite_weight is None
        or spec.safe_asset is None
    ):
        raise ValueError("CORE_SATELLITE_TAA requires core, universe, momentum, top_n, satellite, and safe")
    if rule_id is AfterTaxRuleId.GLIDE_PATH and (spec.glide_months is None or spec.safe_asset is None):
        raise ValueError("GLIDE_PATH requires glide_months and safe_asset")
    return _AfterTaxWeightRule(spec, horizon_end, membership)
