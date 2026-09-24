"""Boundary tests: rule spec validation, ticker metadata, and rarely-hit signal branches."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from src.policy.after_tax_rules import AfterTaxRuleId, AfterTaxRuleSpec, build_weight_rule, parse_after_tax_rule_spec
from src.policy.weight_rule import CASH_SLEEVE
from tests.unit.policy.test_after_tax_rules import MONTHS, SIGNAL_AT, _market, _rate_levels

_HORIZON_END = date(2031, 8, 30)
_CORE = {"QQQ": 0.9, "SOXX": 0.1}


def _dual(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rule_id": "dual_momentum_exit",
        "core_targets": dict(_CORE),
        "safe_asset": "CASH",
        "signal_ticker": "QQQ",
        "sma_months": 10,
        "momentum_months": [12],
    }
    payload.update(overrides)
    return payload


_PARSE_CASES: list[tuple[str, dict[str, Any], str]] = [
    ("payload_type", {}, "rule payload must be a mapping"),
    ("core_not_mapping", _dual(core_targets=[1]), "core_targets must be a mapping"),
    ("core_empty", _dual(core_targets={}), "nonempty mapping"),
    ("core_blank_ticker", _dual(core_targets={" ": 1.0}), "ticker must be non-blank"),
    ("core_duplicate", _dual(core_targets={"qqq": 0.5, "QQQ": 0.5}), "duplicate core_targets ticker"),
    ("core_non_number", _dual(core_targets={"QQQ": "1"}), "must be a number"),
    ("core_negative", _dual(core_targets={"QQQ": 1.5, "SOXX": -0.5}), "finite nonnegative"),
    ("core_not_simplex", _dual(core_targets={"QQQ": 0.5}), "sum to 1.0"),
    ("safe_blank", _dual(safe_asset="  "), "safe_asset must be non-blank"),
    ("universe_type", _dual(universe="QQQ"), "universe must be a list"),
    ("universe_blank", _dual(universe=["QQQ", " "]), "universe ticker must be non-blank"),
    ("universe_duplicate", _dual(universe=["QQQ", "qqq"]), "duplicate universe ticker"),
    ("sma_type", _dual(sma_months="10"), "must be a positive integer"),
    ("sma_zero", _dual(sma_months=0), "must be >= 1"),
    ("momentum_type", _dual(momentum_months=12), "momentum_months must be a list"),
    ("fraction_type", _dual(risk_on_fraction_when_off="half"), "must be a number"),
    ("fraction_range", _dual(risk_on_fraction_when_off=1.5), "must lie in \\[0, 1\\]"),
    ("vol_target_type", _dual(vol_target_annual="x"), "must be a number"),
    ("vol_target_nonfinite", _dual(vol_target_annual=float("inf")), "must be finite"),
    ("vol_target_nonpositive", _dual(vol_target_annual=0.0), "must be positive"),
    ("satellite_range", _dual(satellite_weight=2.0), "must lie in \\[0, 1\\]"),
    ("hurdle_blank", _dual(hurdle_rate_series=" "), "hurdle_rate_series must be non-blank"),
]


@pytest.mark.parametrize(("case_id", "payload", "message"), _PARSE_CASES, ids=[c[0] for c in _PARSE_CASES])
def test_parse_rejects_malformed_payload(case_id: str, payload: Any, message: str) -> None:
    """Malformed rule blocks fail closed at the config boundary."""
    if case_id == "payload_type":
        payload = ["not", "a", "mapping"]

    with pytest.raises(ValueError, match=message):
        parse_after_tax_rule_spec(payload)


def test_parse_normalizes_hurdle_series_and_tickers() -> None:
    """Explicit hurdle series and lower-case tickers normalize to upper case."""
    spec = parse_after_tax_rule_spec(_dual(hurdle_rate_series="dtb6", signal_ticker="qqq"))

    assert spec.hurdle_rate_series == "DTB6"
    assert spec.signal_ticker == "QQQ"


_BUILD_CASES: list[tuple[AfterTaxRuleId, dict[str, Any], str]] = [
    (AfterTaxRuleId.STATIC, {}, "static requires core_targets"),
    (AfterTaxRuleId.DUAL_MOMENTUM_EXIT, {"core_targets": _CORE}, "DUAL_MOMENTUM_EXIT requires"),
    (AfterTaxRuleId.TREND_PARTIAL, {"core_targets": _CORE}, "TREND_PARTIAL requires"),
    (AfterTaxRuleId.VOL_TARGET, {"core_targets": _CORE}, "VOL_TARGET requires"),
    (AfterTaxRuleId.TAA_TOP_N, {}, "TAA_TOP_N requires"),
    (AfterTaxRuleId.GEM, {}, "GEM requires"),
    (AfterTaxRuleId.CORE_SATELLITE_TAA, {"core_targets": _CORE}, "CORE_SATELLITE_TAA requires"),
    (AfterTaxRuleId.GLIDE_PATH, {"core_targets": _CORE}, "GLIDE_PATH requires"),
]


@pytest.mark.parametrize(("rule_id", "kwargs", "message"), _BUILD_CASES, ids=[c[0].value for c in _BUILD_CASES])
def test_build_rejects_incomplete_spec(rule_id: AfterTaxRuleId, kwargs: dict[str, Any], message: str) -> None:
    """A directly-constructed spec missing rule parameters cannot produce a rule."""
    spec = AfterTaxRuleSpec(rule_id=rule_id, core_targets=kwargs.get("core_targets", {}))

    with pytest.raises(ValueError, match=message):
        build_weight_rule(spec, horizon_end=_HORIZON_END)


def test_build_rejects_non_date_horizon_end() -> None:
    """GLIDE_PATH reads the horizon end, so any non-date value is rejected up front."""
    spec = parse_after_tax_rule_spec({"rule_id": "static", "core_targets": {"QQQ": 1.0}})

    with pytest.raises(ValueError, match="horizon_end must be a date"):
        build_weight_rule(spec, horizon_end="2031-08-30")  # type: ignore[arg-type]


def test_rule_tickers_include_signal_safe_and_universe_but_not_cash() -> None:
    """Data loading sees every ETF a rule may touch; the cash sleeve is never a ticker."""
    spec = parse_after_tax_rule_spec(_dual(safe_asset="ief"))
    gem = parse_after_tax_rule_spec(
        {"rule_id": "gem", "safe_asset": "IEF", "signal_ticker": "SPY", "universe": ["SPY", "EFA"]}
    )
    cash_spec = parse_after_tax_rule_spec(_dual(safe_asset="CASH"))

    assert build_weight_rule(spec, horizon_end=_HORIZON_END).tickers == frozenset({"QQQ", "SOXX", "IEF"})
    assert build_weight_rule(gem, horizon_end=_HORIZON_END).tickers == frozenset({"SPY", "EFA", "IEF"})
    cash_rule = build_weight_rule(cash_spec, horizon_end=_HORIZON_END)
    assert CASH_SLEEVE not in cash_rule.tickers
    assert cash_rule.requires_cash_rate is True


def _dip_market(closes: list[float]) -> Any:
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, closes, strict=True)]
    return _market(entries, _rate_levels(5.0), SIGNAL_AT)


_DIP_BELOW_SMA_MOMENTUM_UP = [100.0, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 112.0]


def test_dual_exit_momentum_alone_keeps_core() -> None:
    """Below the SMA but 12-month return above the hurdle still holds the core mix."""
    rule = build_weight_rule(parse_after_tax_rule_spec(_dual()), horizon_end=_HORIZON_END)

    assert dict(rule(SIGNAL_AT, _dip_market(_DIP_BELOW_SMA_MOMENTUM_UP))) == _CORE


def test_trend_partial_above_sma_keeps_core() -> None:
    """Above the moving average the partial-exit rule leaves the core mix untouched."""
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "trend_partial",
            "core_targets": dict(_CORE),
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "risk_on_fraction_when_off": 0.5,
        }
    )
    rule = build_weight_rule(spec, horizon_end=_HORIZON_END)

    assert dict(rule(SIGNAL_AT, _dip_market([100.0 + i for i in range(13)]))) == _CORE


def test_vol_target_single_return_window_means_full_risk() -> None:
    """A one-return window carries no dispersion, so the volatility scale stays at one."""
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "vol_target",
            "core_targets": dict(_CORE),
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "vol_target_annual": 0.2,
            "vol_window_sessions": 1,
        }
    )
    rule = build_weight_rule(spec, horizon_end=_HORIZON_END)

    weights = dict(rule(SIGNAL_AT, _dip_market([100.0 + i for i in range(13)])))

    assert weights == pytest.approx({**_CORE, "CASH": 0.0})


def test_gem_prefers_later_universe_member_with_higher_return() -> None:
    """When the second candidate out-returns the first, GEM holds it."""
    entries = [("SPY", day, 100.0 + i) for i, day in enumerate(MONTHS)]
    entries += [("EFA", day, 100.0 + 3 * i) for i, day in enumerate(MONTHS)]
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "gem",
            "safe_asset": "IEF",
            "signal_ticker": "SPY",
            "universe": ["SPY", "EFA"],
        }
    )
    rule = build_weight_rule(spec, horizon_end=_HORIZON_END)

    assert dict(rule(SIGNAL_AT, _market(entries, _rate_levels(1.0), SIGNAL_AT))) == {"EFA": 1.0}


def test_non_positive_base_close_fails_closed() -> None:
    """A zero adjusted close in the return window is a data anomaly, never a silent inf."""
    rule = build_weight_rule(parse_after_tax_rule_spec(_dual()), horizon_end=_HORIZON_END)
    closes = [0.0] + [130.0 - i for i in range(12)]

    with pytest.raises(ValueError, match="non-positive base close"):
        rule(SIGNAL_AT, _dip_market(closes))
