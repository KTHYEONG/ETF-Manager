"""Preregistered weight-rule family for the after-tax cohort campaign."""

from __future__ import annotations

import calendar
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from src.features.pit_market import PitMarket
from src.policy.weight_rule import CASH_SLEEVE, WeightRule

__all__ = [
    "AfterTaxRuleId",
    "AfterTaxRuleSpec",
    "build_weight_rule",
    "parse_after_tax_rule_spec",
]

_SIMPLEX_TOL: float = 1e-9


class AfterTaxRuleId(StrEnum):
    """Preregistered weight-rule identities for the after-tax campaign."""

    STATIC = "static"
    DUAL_MOMENTUM_EXIT = "dual_momentum_exit"
    TREND_PARTIAL = "trend_partial"
    VOL_TARGET = "vol_target"
    TAA_TOP_N = "taa_top_n"
    GEM = "gem"
    CORE_SATELLITE_TAA = "core_satellite_taa"
    GLIDE_PATH = "glide_path"


@dataclass(frozen=True, slots=True)
class AfterTaxRuleSpec:
    """Typed, config-sourced parameters of one rule; unused fields stay ``None``.

    ``core_targets`` is the risk-on (or static) simplex. ``safe_asset`` is an ETF ticker
    or ``CASH_SLEEVE``. Absolute-momentum hurdles compare a k-month total return with
    ``k/12`` times the mean of the last 12 visible month-end ``hurdle_rate_series`` levels.
    """

    rule_id: AfterTaxRuleId
    core_targets: Mapping[str, float]
    safe_asset: str | None = None
    signal_ticker: str | None = None
    universe: tuple[str, ...] = ()
    sma_months: int | None = None
    momentum_months: tuple[int, ...] = ()
    risk_on_fraction_when_off: float | None = None
    vol_target_annual: float | None = None
    vol_window_sessions: int | None = None
    top_n: int | None = None
    satellite_weight: float | None = None
    glide_months: int | None = None
    hurdle_rate_series: str = "DTB3"


_KNOWN_RULE_KEYS: frozenset[str] = frozenset(
    {
        "rule_id",
        "core_targets",
        "safe_asset",
        "signal_ticker",
        "universe",
        "sma_months",
        "momentum_months",
        "risk_on_fraction_when_off",
        "vol_target_annual",
        "vol_window_sessions",
        "top_n",
        "satellite_weight",
        "glide_months",
        "hurdle_rate_series",
    }
)


def _check_simplex(weights: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(weights, Mapping) or len(weights) == 0:
        raise ValueError(f"{name} must be a nonempty mapping")
    normalized: dict[str, float] = {}
    total = 0.0
    for raw_key, raw_value in weights.items():
        key = str(raw_key).strip().upper()
        if not key:
            raise ValueError(f"{name} ticker must be non-blank")
        if key in normalized:
            raise ValueError(f"duplicate {name} ticker after normalize: {key!r}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(f"{name}[{key!r}] must be a number, got {raw_value!r}")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}[{key!r}] must be finite nonnegative, got {raw_value!r}")
        normalized[key] = value
        total += value
    if not math.isfinite(total) or abs(total - 1.0) > _SIMPLEX_TOL:
        raise ValueError(f"{name} weights must sum to 1.0 within 1e-9, got {total!r}")
    return normalized


def _check_fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return result


def _check_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return value


def parse_after_tax_rule_spec(payload: Mapping[str, object]) -> AfterTaxRuleSpec:
    """Validate a rule block from campaign JSON.

    Raises:
        ValueError: On an unknown rule id, a non-simplex ``core_targets``, missing
            parameters required by the rule, parameters that do not apply to the rule,
            fractions outside [0, 1], or non-positive windows.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("rule payload must be a mapping")
    unknown = set(payload.keys()) - _KNOWN_RULE_KEYS
    if unknown:
        raise ValueError(f"unknown rule fields: {sorted(unknown)!r}")
    raw_id = payload.get("rule_id")
    try:
        rule_id = AfterTaxRuleId(str(raw_id))
    except ValueError as exc:
        raise ValueError(f"unknown rule id {raw_id!r}") from exc

    raw_core = payload.get("core_targets")
    core: dict[str, float] = {}
    if raw_core is not None:
        if not isinstance(raw_core, Mapping):
            raise ValueError("core_targets must be a mapping")
        core = _check_simplex(raw_core, "core_targets")

    def _opt_str(name: str) -> str | None:
        value = payload.get(name)
        if value is None:
            return None
        text = str(value).strip().upper()
        if not text:
            raise ValueError(f"{name} must be non-blank when set")
        return text

    safe_asset = _opt_str("safe_asset")
    signal_ticker = _opt_str("signal_ticker")

    raw_universe = payload.get("universe")
    universe: tuple[str, ...] = ()
    if raw_universe is not None:
        if not isinstance(raw_universe, list | tuple):
            raise ValueError("universe must be a list of tickers")
        seen: dict[str, None] = {}
        for item in raw_universe:
            ticker = str(item).strip().upper()
            if not ticker:
                raise ValueError("universe ticker must be non-blank")
            if ticker in seen:
                raise ValueError(f"duplicate universe ticker {ticker!r}")
            seen[ticker] = None
        universe = tuple(seen.keys())

    def _opt_int(name: str) -> int | None:
        value = payload.get(name)
        if value is None:
            return None
        return _check_positive_int(value, name)

    sma_months = _opt_int("sma_months")
    vol_window_sessions = _opt_int("vol_window_sessions")
    top_n = _opt_int("top_n")
    glide_months = _opt_int("glide_months")

    raw_momentum = payload.get("momentum_months")
    momentum_months: tuple[int, ...] = ()
    if raw_momentum is not None:
        if not isinstance(raw_momentum, list | tuple):
            raise ValueError("momentum_months must be a list of positive integers")
        momentum_months = tuple(_check_positive_int(item, "momentum_months") for item in raw_momentum)

    def _opt_float(name: str, *, positive: bool = False) -> float:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a number, got {value!r}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if positive and result <= 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}")
        return result

    risk_frac: float | None = None
    if payload.get("risk_on_fraction_when_off") is not None:
        risk_frac = _check_fraction(payload.get("risk_on_fraction_when_off"), "risk_on_fraction_when_off")
    vol_target: float | None = None
    if payload.get("vol_target_annual") is not None:
        vol_target = _opt_float("vol_target_annual", positive=True)
    satellite_weight: float | None = None
    if payload.get("satellite_weight") is not None:
        satellite_weight = _check_fraction(payload.get("satellite_weight"), "satellite_weight")

    hurdle_series = "DTB3"
    if payload.get("hurdle_rate_series") is not None:
        hurdle_series = str(payload.get("hurdle_rate_series")).strip()
        if not hurdle_series:
            raise ValueError("hurdle_rate_series must be non-blank when set")
        hurdle_series = hurdle_series.upper()

    def _present(name: str, value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, tuple | list | dict):
            return len(value) > 0
        return True

    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def _forbid(name: str, value: object) -> None:
        if _present(name, value):
            raise ValueError(f"{name} does not apply to rule {rule_id.value!r}")

    has_core = len(core) > 0
    if rule_id is AfterTaxRuleId.STATIC:
        _require(has_core, "STATIC requires core_targets")
        _forbid("safe_asset", safe_asset)
        _forbid("signal_ticker", signal_ticker)
        _forbid("universe", universe)
        _forbid("sma_months", sma_months)
        _forbid("momentum_months", momentum_months)
        _forbid("risk_on_fraction_when_off", risk_frac)
        _forbid("vol_target_annual", vol_target)
        _forbid("vol_window_sessions", vol_window_sessions)
        _forbid("top_n", top_n)
        _forbid("satellite_weight", satellite_weight)
        _forbid("glide_months", glide_months)
    elif rule_id is AfterTaxRuleId.DUAL_MOMENTUM_EXIT:
        _require(has_core, "DUAL_MOMENTUM_EXIT requires core_targets")
        _require(safe_asset is not None, "DUAL_MOMENTUM_EXIT requires safe_asset")
        _require(signal_ticker is not None, "DUAL_MOMENTUM_EXIT requires signal_ticker")
        _require(sma_months is not None, "DUAL_MOMENTUM_EXIT requires sma_months")
        _require(len(momentum_months) == 1, "DUAL_MOMENTUM_EXIT requires exactly one momentum_months entry")
        _forbid("universe", universe)
        _forbid("risk_on_fraction_when_off", risk_frac)
        _forbid("vol_target_annual", vol_target)
        _forbid("vol_window_sessions", vol_window_sessions)
        _forbid("top_n", top_n)
        _forbid("satellite_weight", satellite_weight)
        _forbid("glide_months", glide_months)
    elif rule_id is AfterTaxRuleId.TREND_PARTIAL:
        _require(has_core, "TREND_PARTIAL requires core_targets")
        _require(safe_asset is not None, "TREND_PARTIAL requires safe_asset")
        _require(signal_ticker is not None, "TREND_PARTIAL requires signal_ticker")
        _require(sma_months is not None, "TREND_PARTIAL requires sma_months")
        _require(risk_frac is not None, "TREND_PARTIAL requires risk_on_fraction_when_off")
        _forbid("universe", universe)
        _forbid("momentum_months", momentum_months)
        _forbid("vol_target_annual", vol_target)
        _forbid("vol_window_sessions", vol_window_sessions)
        _forbid("top_n", top_n)
        _forbid("satellite_weight", satellite_weight)
        _forbid("glide_months", glide_months)
    elif rule_id is AfterTaxRuleId.VOL_TARGET:
        _require(has_core, "VOL_TARGET requires core_targets")
        _require(safe_asset is not None, "VOL_TARGET requires safe_asset")
        _require(signal_ticker is not None, "VOL_TARGET requires signal_ticker")
        _require(vol_target is not None, "VOL_TARGET requires vol_target_annual")
        _require(vol_window_sessions is not None, "VOL_TARGET requires vol_window_sessions")
        _forbid("universe", universe)
        _forbid("sma_months", sma_months)
        _forbid("momentum_months", momentum_months)
        _forbid("risk_on_fraction_when_off", risk_frac)
        _forbid("top_n", top_n)
        _forbid("satellite_weight", satellite_weight)
        _forbid("glide_months", glide_months)
    elif rule_id is AfterTaxRuleId.TAA_TOP_N:
        _forbid("core_targets", core if has_core else None)
        _require(safe_asset is not None, "TAA_TOP_N requires safe_asset")
        _require(len(universe) > 0, "TAA_TOP_N requires universe")
        _require(len(momentum_months) > 0, "TAA_TOP_N requires momentum_months")
        _require(top_n is not None, "TAA_TOP_N requires top_n")
        _require(top_n is not None and top_n <= len(universe), "TAA_TOP_N requires top_n <= len(universe)")
        _forbid("signal_ticker", signal_ticker)
        _forbid("sma_months", sma_months)
        _forbid("risk_on_fraction_when_off", risk_frac)
        _forbid("vol_target_annual", vol_target)
        _forbid("vol_window_sessions", vol_window_sessions)
        _forbid("satellite_weight", satellite_weight)
        _forbid("glide_months", glide_months)
    elif rule_id is AfterTaxRuleId.GEM:
        _forbid("core_targets", core if has_core else None)
        _require(signal_ticker is not None, "GEM requires signal_ticker")
        _require(len(universe) > 0, "GEM requires universe")
        _require(safe_asset is not None, "GEM requires safe_asset")
        _forbid("sma_months", sma_months)
        _forbid("momentum_months", momentum_months)
        _forbid("risk_on_fraction_when_off", risk_frac)
        _forbid("vol_target_annual", vol_target)
        _forbid("vol_window_sessions", vol_window_sessions)
        _forbid("top_n", top_n)
        _forbid("satellite_weight", satellite_weight)
        _forbid("glide_months", glide_months)
    elif rule_id is AfterTaxRuleId.CORE_SATELLITE_TAA:
        _require(has_core, "CORE_SATELLITE_TAA requires core_targets")
        _require(safe_asset is not None, "CORE_SATELLITE_TAA requires safe_asset")
        _require(len(universe) > 0, "CORE_SATELLITE_TAA requires universe")
        _require(len(momentum_months) > 0, "CORE_SATELLITE_TAA requires momentum_months")
        _require(top_n is not None, "CORE_SATELLITE_TAA requires top_n")
        _require(top_n is not None and top_n <= len(universe), "CORE_SATELLITE_TAA requires top_n <= len(universe)")
        _require(satellite_weight is not None, "CORE_SATELLITE_TAA requires satellite_weight")
        _forbid("signal_ticker", signal_ticker)
        _forbid("sma_months", sma_months)
        _forbid("risk_on_fraction_when_off", risk_frac)
        _forbid("vol_target_annual", vol_target)
        _forbid("vol_window_sessions", vol_window_sessions)
        _forbid("glide_months", glide_months)
    else:
        _require(has_core, "GLIDE_PATH requires core_targets")
        _require(safe_asset is not None, "GLIDE_PATH requires safe_asset")
        _require(glide_months is not None, "GLIDE_PATH requires glide_months")
        _forbid("signal_ticker", signal_ticker)
        _forbid("universe", universe)
        _forbid("sma_months", sma_months)
        _forbid("momentum_months", momentum_months)
        _forbid("risk_on_fraction_when_off", risk_frac)
        _forbid("vol_target_annual", vol_target)
        _forbid("vol_window_sessions", vol_window_sessions)
        _forbid("top_n", top_n)
        _forbid("satellite_weight", satellite_weight)

    return AfterTaxRuleSpec(
        rule_id=rule_id,
        core_targets=dict(core),
        safe_asset=safe_asset,
        signal_ticker=signal_ticker,
        universe=tuple(universe),
        sma_months=sma_months,
        momentum_months=tuple(momentum_months),
        risk_on_fraction_when_off=risk_frac,
        vol_target_annual=vol_target,
        vol_window_sessions=vol_window_sessions,
        top_n=top_n,
        satellite_weight=satellite_weight,
        glide_months=glide_months,
        hurdle_rate_series=hurdle_series,
    )


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


def _taa_weights(spec: AfterTaxRuleSpec, as_of: datetime, market: PitMarket) -> dict[str, float]:
    assert spec.safe_asset is not None
    assert spec.top_n is not None
    assert spec.universe
    assert spec.momentum_months
    rate_pct = market.rate_percent(spec.hurdle_rate_series, as_of)
    mean_k = sum(spec.momentum_months) / len(spec.momentum_months)
    hurdle = (mean_k / 12.0) * (rate_pct / 100.0)
    scored: list[tuple[str, float]] = []
    for ticker in spec.universe:
        rets = [_k_month_return(market, ticker, as_of, k) for k in spec.momentum_months]
        scored.append((ticker, sum(rets) / len(rets)))
    scored.sort(key=lambda item: item[1], reverse=True)
    picks = scored[: spec.top_n]
    weights: dict[str, float] = {}
    slot = 1.0 / float(spec.top_n)
    for ticker, score in picks:
        target = ticker if score >= hurdle else spec.safe_asset
        weights[target] = weights.get(target, 0.0) + slot
    return weights


class _AfterTaxWeightRule:
    """Concrete PIT rule dispatching on a parsed spec."""

    __slots__ = ("_horizon_end", "_spec")

    def __init__(self, spec: AfterTaxRuleSpec, horizon_end: date) -> None:
        self._spec = spec
        self._horizon_end = horizon_end

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
            return _taa_weights(spec, signal_at, market)
        if spec.rule_id is AfterTaxRuleId.GEM:
            assert spec.signal_ticker is not None
            assert spec.safe_asset is not None
            assert spec.universe
            signal_ret = _k_month_return(market, spec.signal_ticker, signal_at, 12)
            if signal_ret < _hurdle_for(market, spec.hurdle_rate_series, signal_at, 12):
                return {spec.safe_asset: 1.0}
            best_ticker = spec.universe[0]
            best_ret = _k_month_return(market, best_ticker, signal_at, 12)
            for ticker in spec.universe[1:]:
                ret = _k_month_return(market, ticker, signal_at, 12)
                if ret > best_ret:
                    best_ret = ret
                    best_ticker = ticker
            return {best_ticker: 1.0}
        if spec.rule_id is AfterTaxRuleId.CORE_SATELLITE_TAA:
            assert spec.satellite_weight is not None
            satellite = _taa_weights(spec, signal_at, market)
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


def build_weight_rule(spec: AfterTaxRuleSpec, *, horizon_end: date) -> WeightRule:
    """Instantiate a PIT ``WeightRule`` for one cohort.

    ``horizon_end`` is the cohort's last calendar day; only GLIDE_PATH reads it.

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
    return _AfterTaxWeightRule(spec, horizon_end)
