"""Invariant guard tests for the after-tax weight-rule family."""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.data.universe import MembershipEntry, UniverseMembership
from src.features.pit_market import PitMarket
from src.policy.after_tax_rules import AfterTaxRuleId, AfterTaxRuleSpec, build_weight_rule, parse_after_tax_rule_spec
from src.policy.targets import PolicyError
from src.policy.weight_rule import CASH_SLEEVE


def _month_dates(start: date, count: int) -> list[date]:
    dates: list[date] = []
    year, month = start.year, start.month
    for _ in range(count):
        last_day = calendar.monthrange(year, month)[1]
        dates.append(date(year, month, last_day))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def _prices_frame(entries: list[tuple[str, date, float]], signal_at: datetime) -> pl.DataFrame:
    rows = [
        {
            "ticker": ticker,
            "date": day,
            "adjusted_close": price,
            "available_at": signal_at - timedelta(days=1),
        }
        for ticker, day, price in entries
    ]
    return pl.DataFrame(rows)


def _rates_frame(levels: list[tuple[date, float]], signal_at: datetime, series: str = "DTB3") -> pl.DataFrame:
    rows = [
        {
            "series_id": series,
            "observation_date": day,
            "value": value,
            "available_at": signal_at - timedelta(days=1),
        }
        for day, value in levels
    ]
    return pl.DataFrame(rows)


def _market(
    price_entries: list[tuple[str, date, float]],
    rate_levels: list[tuple[date, float]],
    signal_at: datetime,
) -> PitMarket:
    return PitMarket(_prices_frame(price_entries, signal_at), _rates_frame(rate_levels, signal_at))


SIGNAL_AT = datetime(2021, 8, 31, 20, 0, tzinfo=UTC)
MONTHS = _month_dates(date(2020, 8, 31), 13)


def _rate_levels(value: float = 5.0) -> list[tuple[date, float]]:
    return [(day, value) for day in MONTHS]


def _membership(last_trading_dates: dict[str, date | None] | None = None) -> UniverseMembership:
    last_dates = last_trading_dates or {}
    entries = {
        ticker: MembershipEntry(
            ticker=ticker,
            listing_date=date(2010, 1, 1),
            last_trading_date=last_dates.get(ticker),
            evidence_url=f"https://example.com/{ticker}",
        )
        for ticker in ("QQQ", "SPY", "EFA", "IEF")
    }
    return UniverseMembership(entries=entries, sha256="test")


def _taa_market() -> PitMarket:
    entries = [("QQQ", day, 100.0 + 2.0 * index) for index, day in enumerate(MONTHS)]
    entries += [("SPY", day, 100.0 + index) for index, day in enumerate(MONTHS)]
    entries += [("EFA", day, 100.0 + 4.0 * index) for index, day in enumerate(MONTHS)]
    return _market(entries, _rate_levels(), SIGNAL_AT)


def _taa_spec() -> AfterTaxRuleSpec:
    return parse_after_tax_rule_spec(
        {
            "rule_id": "taa_top_n",
            "universe": ["QQQ", "SPY", "EFA"],
            "momentum_months": [12],
            "top_n": 2,
            "safe_asset": "IEF",
        }
    )


def test_dual_exit_risk_on_keeps_core() -> None:
    """Dual exit risk-on keeps core."""
    closes = [100.0 + i for i in range(13)]
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, closes, strict=True)]
    entries += [("SOXX", day, 50.0) for day in MONTHS]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "dual_momentum_exit",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "momentum_months": [12],
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"QQQ": 0.9, "SOXX": 0.1}


def test_dual_exit_needs_both_signals_off() -> None:
    """Dual exit with SMA off but momentum on stays risk-on, both off goes safe."""
    # Recent dip below SMA but 12-month gain still beats a 5% hurdle.
    closes = [100.0, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 115.0]
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, closes, strict=True)]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "dual_momentum_exit",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "momentum_months": [12],
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"QQQ": 0.9, "SOXX": 0.1}
    falling = [130.0 - i for i in range(13)]
    entries2 = [("QQQ", day, price) for day, price in zip(MONTHS, falling, strict=True)]
    market2 = _market(entries2, _rate_levels(), SIGNAL_AT)
    assert dict(rule(SIGNAL_AT, market2)) == {"CASH": 1.0}


def test_trend_partial_scales_risk() -> None:
    """Trend partial scales risk when below the SMA."""
    falling = [130.0 - i for i in range(13)]
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, falling, strict=True)]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "trend_partial",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "risk_on_fraction_when_off": 0.5,
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    weights = dict(rule(SIGNAL_AT, market))
    assert weights["QQQ"] == pytest.approx(0.45)
    assert weights["SOXX"] == pytest.approx(0.05)
    assert weights[CASH_SLEEVE] == pytest.approx(0.5)


def test_vol_target_caps_at_one() -> None:
    """Vol target scales down high vol and caps low vol at one."""
    import math

    signal_day = date(2021, 8, 31)
    days = [signal_day - timedelta(days=i) for i in range(63, -1, -1)]

    def _closes(amplitude: float) -> list[float]:
        closes = [100.0]
        for i in range(64 - 1):
            ret = amplitude if i % 2 == 0 else -amplitude
            closes.append(closes[-1] * (1.0 + ret))
        return closes

    amp_high = 0.40 / math.sqrt(252.0)
    amp_low = 0.10 / math.sqrt(252.0)
    high_entries = [("QQQ", day, price) for day, price in zip(days, _closes(amp_high), strict=True)]
    low_entries = [("QQQ", day, price) for day, price in zip(days, _closes(amp_low), strict=True)]
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "vol_target",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "vol_target_annual": 0.20,
            "vol_window_sessions": 63,
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    high_market = _market(high_entries, _rate_levels(), SIGNAL_AT)
    high = dict(rule(SIGNAL_AT, high_market))
    assert high["QQQ"] == pytest.approx(0.45, abs=0.05)
    assert high[CASH_SLEEVE] == pytest.approx(0.5, abs=0.05)
    low_market = _market(low_entries, _rate_levels(), SIGNAL_AT)
    low = dict(rule(SIGNAL_AT, low_market))
    assert low["QQQ"] == pytest.approx(0.9, abs=0.02)
    assert low["SOXX"] == pytest.approx(0.1, abs=0.02)


def test_taa_replaces_failing_pick() -> None:
    """TAA top-2 replaces a hurdle-failing pick with the safe asset."""
    qqq = [100.0 + 2 * i for i in range(13)]
    spy = [100.0 - 0.5 * i for i in range(13)]
    efa = [100.0 - i for i in range(13)]
    entries: list[tuple[str, date, float]] = []
    for day, v in zip(MONTHS, qqq, strict=True):
        entries.append(("QQQ", day, v))
    for day, v in zip(MONTHS, spy, strict=True):
        entries.append(("SPY", day, v))
    for day, v in zip(MONTHS, efa, strict=True):
        entries.append(("EFA", day, v))
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "taa_top_n",
            "universe": ["QQQ", "SPY", "EFA"],
            "momentum_months": [12],
            "top_n": 2,
            "safe_asset": "IEF",
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"QQQ": 0.5, "IEF": 0.5}


def test_taa_excludes_delisted_highest_momentum_member() -> None:
    membership = _membership({"EFA": date(2021, 8, 30)})
    rule = build_weight_rule(
        _taa_spec(),
        horizon_end=date(2031, 8, 30),
        membership=membership,
    )

    weights = dict(rule(SIGNAL_AT, _taa_market()))

    assert "EFA" not in weights
    assert weights == {"QQQ": 0.5, "SPY": 0.5}


def test_future_delisting_is_invisible_to_taa() -> None:
    spec = _taa_spec()
    market = _taa_market()
    unguarded = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    guarded = build_weight_rule(
        spec,
        horizon_end=date(2031, 8, 30),
        membership=_membership({"EFA": date(2021, 9, 30)}),
    )

    assert dict(guarded(SIGNAL_AT, market)) == dict(unguarded(SIGNAL_AT, market))


def test_taa_with_fewer_eligible_members_remains_simplex() -> None:
    membership = _membership({"SPY": date(2021, 8, 30), "EFA": date(2021, 8, 30)})
    rule = build_weight_rule(
        _taa_spec(),
        horizon_end=date(2031, 8, 30),
        membership=membership,
    )

    assert dict(rule(SIGNAL_AT, _taa_market())) == {"QQQ": 1.0}


def test_no_eligible_universe_member_holds_safe_asset() -> None:
    membership = _membership(
        {
            "QQQ": date(2021, 8, 30),
            "SPY": date(2021, 8, 30),
            "EFA": date(2021, 8, 30),
        }
    )
    taa = build_weight_rule(
        _taa_spec(),
        horizon_end=date(2031, 8, 30),
        membership=membership,
    )
    core_satellite_spec = parse_after_tax_rule_spec(
        {
            "rule_id": "core_satellite_taa",
            "core_targets": {"QQQ": 1.0},
            "universe": ["QQQ", "SPY", "EFA"],
            "momentum_months": [12],
            "top_n": 2,
            "satellite_weight": 0.3,
            "safe_asset": "IEF",
        }
    )
    core_satellite = build_weight_rule(
        core_satellite_spec,
        horizon_end=date(2031, 8, 30),
        membership=membership,
    )
    gem_spec = parse_after_tax_rule_spec(
        {"rule_id": "gem", "signal_ticker": "SPY", "universe": ["SPY", "EFA"], "safe_asset": "IEF"}
    )
    gem = build_weight_rule(
        gem_spec,
        horizon_end=date(2031, 8, 30),
        membership=membership,
    )

    assert dict(taa(SIGNAL_AT, _taa_market())) == {"IEF": 1.0}
    assert dict(core_satellite(SIGNAL_AT, _taa_market())) == {"IEF": 1.0}
    assert dict(gem(SIGNAL_AT, _taa_market())) == {"IEF": 1.0}


def test_ineligible_safe_asset_fails_policy() -> None:
    rule = build_weight_rule(
        _taa_spec(),
        horizon_end=date(2031, 8, 30),
        membership=_membership({"IEF": date(2021, 8, 30)}),
    )

    with pytest.raises(PolicyError, match="safe asset 'IEF' is not eligible"):
        rule(SIGNAL_AT, _taa_market())


def test_gem_excludes_delisted_universe_member() -> None:
    spec = parse_after_tax_rule_spec(
        {"rule_id": "gem", "signal_ticker": "SPY", "universe": ["SPY", "EFA"], "safe_asset": "IEF"}
    )
    rule = build_weight_rule(
        spec,
        horizon_end=date(2031, 8, 30),
        membership=_membership({"EFA": date(2021, 8, 30)}),
    )

    assert dict(rule(SIGNAL_AT, _taa_market())) == {"SPY": 1.0}


def test_gem_hurdle_to_safety() -> None:
    """GEM below-hurdle signal moves fully to safety."""
    spy = [130.0 - i for i in range(13)]
    efa = [100.0 + i for i in range(13)]
    entries = [("SPY", day, v) for day, v in zip(MONTHS, spy, strict=True)]
    entries += [("EFA", day, v) for day, v in zip(MONTHS, efa, strict=True)]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {"rule_id": "gem", "signal_ticker": "SPY", "universe": ["SPY", "EFA"], "safe_asset": "IEF"}
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"IEF": 1.0}


def test_glide_path_switches_at_boundary() -> None:
    """Glide path holds core before the boundary and safety after it."""
    market = _market([("QQQ", date(2021, 8, 31), 100.0)], _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "glide_path",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "IEF",
            "glide_months": 60,
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2026, 7, 31))
    before = datetime(2021, 7, 30, 20, 0, tzinfo=UTC)
    after = datetime(2021, 8, 31, 20, 0, tzinfo=UTC)
    assert dict(rule(before, market)) == {"QQQ": 0.9, "SOXX": 0.1}
    assert dict(rule(after, market)) == {"IEF": 1.0}


def test_spec_validation_rejects_missing_fraction() -> None:
    """TREND_PARTIAL without its fraction is rejected."""
    with pytest.raises(ValueError, match="risk_on_fraction_when_off"):
        parse_after_tax_rule_spec(
            {
                "rule_id": "trend_partial",
                "core_targets": {"QQQ": 1.0},
                "safe_asset": "CASH",
                "signal_ticker": "QQQ",
                "sma_months": 10,
            }
        )


def test_rule_look_ahead_invariance() -> None:
    """Perturbing rows after the signal instant leaves weights unchanged."""
    closes = [100.0 + i for i in range(13)]
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, closes, strict=True)]
    base = _market(entries, _rate_levels(), SIGNAL_AT)
    future_day = date(2021, 9, 30)
    base_prices = _prices_frame(entries, SIGNAL_AT)
    extended = pl.concat(
        [
            base_prices,
            pl.DataFrame(
                [
                    {
                        "ticker": "QQQ",
                        "date": future_day,
                        "adjusted_close": 9999.0,
                        "available_at": SIGNAL_AT + timedelta(days=5),
                    }
                ]
            ),
        ]
    )
    rates = _rates_frame(_rate_levels(), SIGNAL_AT)
    extended_market = PitMarket(extended, rates)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "dual_momentum_exit",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "momentum_months": [12],
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, base)) == dict(rule(SIGNAL_AT, extended_market))


def test_static_rule_returns_core_and_metadata() -> None:
    """STATIC always returns core with correct tickers and no cash-rate need."""
    market = _market([("QQQ", date(2021, 8, 31), 100.0)], _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec({"rule_id": "static", "core_targets": {"QQQ": 0.9, "SOXX": 0.1}})
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"QQQ": 0.9, "SOXX": 0.1}
    assert rule.tickers == frozenset({"QQQ", "SOXX"})
    assert rule.requires_cash_rate is False


def test_gem_picks_higher_momentum_member() -> None:
    """GEM above-hurdle signal buys the stronger universe member."""
    spy = [100.0 + 2 * i for i in range(13)]
    efa = [100.0 + 0.5 * i for i in range(13)]
    entries = [("SPY", day, v) for day, v in zip(MONTHS, spy, strict=True)]
    entries += [("EFA", day, v) for day, v in zip(MONTHS, efa, strict=True)]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {"rule_id": "gem", "signal_ticker": "SPY", "universe": ["SPY", "EFA"], "safe_asset": "IEF"}
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"SPY": 1.0}
    assert rule.requires_cash_rate is True


def test_core_satellite_blends_taa_weights() -> None:
    """CORE_SATELLITE blends scaled core with the satellite TAA sleeve."""
    qqq = [100.0 + 2 * i for i in range(13)]
    spy = [100.0 - 0.5 * i for i in range(13)]
    efa = [100.0 - i for i in range(13)]
    entries: list[tuple[str, date, float]] = []
    for day, v in zip(MONTHS, qqq, strict=True):
        entries.append(("QQQ", day, v))
    for day, v in zip(MONTHS, spy, strict=True):
        entries.append(("SPY", day, v))
    for day, v in zip(MONTHS, efa, strict=True):
        entries.append(("EFA", day, v))
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "core_satellite_taa",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "universe": ["QQQ", "SPY", "EFA"],
            "momentum_months": [12],
            "top_n": 2,
            "satellite_weight": 0.3,
            "safe_asset": "IEF",
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    weights = dict(rule(SIGNAL_AT, market))
    assert weights["QQQ"] == pytest.approx(0.63 + 0.15)
    assert weights["SOXX"] == pytest.approx(0.07)
    assert weights["IEF"] == pytest.approx(0.15)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_parse_rejects_unknown_and_misplaced_fields() -> None:
    """Unknown ids, unknown fields, and misplaced params fail closed."""
    with pytest.raises(ValueError, match="unknown rule id"):
        parse_after_tax_rule_spec({"rule_id": "nope", "core_targets": {"QQQ": 1.0}})
    with pytest.raises(ValueError, match="unknown rule fields"):
        parse_after_tax_rule_spec({"rule_id": "static", "core_targets": {"QQQ": 1.0}, "bogus": 1})
    with pytest.raises(ValueError, match="does not apply"):
        parse_after_tax_rule_spec(
            {"rule_id": "static", "core_targets": {"QQQ": 1.0}, "safe_asset": "CASH"}
        )
    with pytest.raises(ValueError, match="top_n <= len"):
        parse_after_tax_rule_spec(
            {
                "rule_id": "taa_top_n",
                "universe": ["QQQ", "SPY"],
                "momentum_months": [12],
                "top_n": 3,
                "safe_asset": "IEF",
            }
        )
    with pytest.raises(ValueError, match="requires core_targets"):
        build_weight_rule(
            AfterTaxRuleSpec(rule_id=AfterTaxRuleId.STATIC, core_targets={}),
            horizon_end=date(2031, 8, 30),
        )


def test_parse_and_weight_parity_across_extraction() -> None:
    """Parser and weights agree before/after extraction; weights stay simplex."""
    import src.policy.after_tax_rules as rules_module
    from src.policy.after_tax_rule_parse import (
        AfterTaxRuleSpec as ExtractedSpec,
    )
    from src.policy.after_tax_rule_parse import (
        parse_after_tax_rule_spec as extracted_parse,
    )

    assert rules_module.parse_after_tax_rule_spec is extracted_parse
    assert rules_module.AfterTaxRuleSpec is ExtractedSpec
    payload = {"rule_id": "static", "core_targets": {"QQQ": 0.9, "SOXX": 0.1}}
    assert extracted_parse(payload) == rules_module.parse_after_tax_rule_spec(payload)
    rule = rules_module.build_weight_rule(extracted_parse(payload), horizon_end=date(2024, 12, 31))
    market = _market(
        [(ticker, day, 100.0) for ticker in ("QQQ", "SOXX") for day in MONTHS],
        _rate_levels(),
        SIGNAL_AT,
    )
    weights = rule(SIGNAL_AT, market)
    assert abs(sum(weights.values()) - 1.0) <= 1e-9
    assert weights == {"QQQ": 0.9, "SOXX": 0.1}
    custom_hurdle = extracted_parse(
        {"rule_id": "static", "core_targets": {"QQQ": 1.0}, "hurdle_rate_series": "dtb3"}
    )
    assert custom_hurdle.hurdle_rate_series == "DTB3"


def _trend_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "rule_id": "trend_partial",
        "core_targets": {"QQQ": 1.0},
        "safe_asset": "CASH",
        "signal_ticker": "QQQ",
        "sma_months": 10,
        "risk_on_fraction_when_off": 0.5,
    }
    payload.update(overrides)
    return payload


def _dual_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "rule_id": "dual_momentum_exit",
        "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
        "safe_asset": "CASH",
        "signal_ticker": "QQQ",
        "sma_months": 10,
        "momentum_months": [12],
    }
    payload.update(overrides)
    return payload


def _taa_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "rule_id": "taa_top_n",
        "universe": ["QQQ", "SPY"],
        "momentum_months": [12],
        "top_n": 1,
        "safe_asset": "IEF",
    }
    payload.update(overrides)
    return payload


def _vol_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "rule_id": "vol_target",
        "core_targets": {"QQQ": 1.0},
        "safe_asset": "CASH",
        "signal_ticker": "QQQ",
        "vol_target_annual": 0.15,
        "vol_window_sessions": 60,
    }
    payload.update(overrides)
    return payload


_MALFORMED_VARIANTS: list[tuple[str, dict[str, object] | list[object], str]] = [
    ("empty_core", {"rule_id": "static", "core_targets": {}}, "nonempty mapping"),
    ("blank_ticker", {"rule_id": "static", "core_targets": {"  ": 1.0}}, "ticker must be non-blank"),
    ("dup_ticker", {"rule_id": "static", "core_targets": {"QQQ": 0.5, "qqq": 0.5}}, "duplicate"),
    ("non_numeric_weight", {"rule_id": "static", "core_targets": {"QQQ": "x"}}, "must be a number"),
    ("non_finite_weight", {"rule_id": "static", "core_targets": {"QQQ": float("nan")}}, "finite nonnegative"),
    ("off_simplex", {"rule_id": "static", "core_targets": {"QQQ": 0.5}}, "must sum to 1.0"),
    ("fraction_type", _trend_payload(risk_on_fraction_when_off="x"), "must be a number"),
    ("fraction_range", _trend_payload(risk_on_fraction_when_off=1.5), r"must lie in \[0, 1\]"),
    ("window_type", _dual_payload(sma_months="x"), "positive integer"),
    ("window_range", _dual_payload(sma_months=0), "must be >= 1"),
    ("payload_not_mapping", [], "must be a mapping"),
    ("core_not_mapping", {"rule_id": "static", "core_targets": ["QQQ"]}, "core_targets must be a mapping"),
    ("blank_safe_asset", {"rule_id": "static", "core_targets": {"QQQ": 1.0}, "safe_asset": "  "}, "non-blank when set"),
    ("universe_not_list", _taa_payload(universe="QQQ"), "must be a list of tickers"),
    ("universe_blank", _taa_payload(universe=["  ", "SPY"]), "universe ticker must be non-blank"),
    ("universe_dup", _taa_payload(universe=["QQQ", "qqq"]), "duplicate universe ticker"),
    ("momentum_not_list", _dual_payload(momentum_months=12), "must be a list of positive integers"),
    ("vol_target_type", _vol_payload(vol_target_annual="x"), "must be a number"),
    ("vol_target_finite", _vol_payload(vol_target_annual=float("nan")), "must be finite"),
    ("vol_target_positive", _vol_payload(vol_target_annual=-1.0), "must be positive"),
    (
        "blank_hurdle_series",
        {"rule_id": "static", "core_targets": {"QQQ": 1.0}, "hurdle_rate_series": "  "},
        "hurdle_rate_series must be non-blank",
    ),
]


@pytest.mark.parametrize(("case_id", "payload", "message"), _MALFORMED_VARIANTS, ids=[case[0] for case in _MALFORMED_VARIANTS])
def test_parse_rejects_malformed_field_variants(case_id: str, payload: object, message: str) -> None:
    """Every malformed rule field fails closed with a ValueError naming the field."""
    with pytest.raises(ValueError, match=message):
        parse_after_tax_rule_spec(payload)  # type: ignore[arg-type]
