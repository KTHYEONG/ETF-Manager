"""Invariant guards for the modern-tier pre-inception splice."""

from __future__ import annotations

import calendar as _calendar
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.sim.pension_monthly import panel_from_prices
from src.sim.pension_splice import (
    SpliceRecord,
    SpliceRule,
    realized_panel_from_prices,
    realized_window_start,
    splice_modern_panel,
)

_AS_OF = datetime(2001, 1, 1, tzinfo=UTC)


def _month_ends(year: int, first: int, last: int) -> tuple[date, ...]:
    return tuple(date(year, month, _calendar.monthrange(year, month)[1]) for month in range(first, last + 1))


def _prices_frame(sessions: list[tuple[str, date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [ticker for ticker, _, _ in sessions],
            "date": [day for _, day, _ in sessions],
            "adjusted_close": [close for _, _, close in sessions],
            "available_at": [datetime(2000, 1, 1, tzinfo=UTC)] * len(sessions),
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def _research_frame(rows: list[tuple[str, date, float]], lag_days: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": [series for series, _, _ in rows],
            "period_end": [month for _, month, _ in rows],
            "simple_return": [value for _, _, value in rows],
            "label": ["research_proxy"] * len(rows),
            "source": ["synthetic"] * len(rows),
            "available_at": [
                datetime(month.year, month.month, month.day, tzinfo=UTC) + timedelta(days=lag_days)
                for _, month, _ in rows
            ],
        },
        schema={
            "series_id": pl.String,
            "period_end": pl.Date,
            "simple_return": pl.Float64,
            "label": pl.String,
            "source": pl.String,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def _schd_sessions() -> list[tuple[str, date, float]]:
    return [
        ("SCHD", date(2000, 3, 31), 100.0),
        ("SCHD", date(2000, 4, 30), 110.0),
        ("SCHD", date(2000, 5, 31), 121.0),
        ("SCHD", date(2000, 6, 30), 133.1),
    ]


def test_splice_modern_panel_empty_splices_reproduce_price_panel() -> None:
    """Without splices the panel equals panel_from_prices and no record is emitted."""
    sessions = [
        ("SPY", date(1999, 12, 31), 100.0),
        ("SPY", date(2000, 1, 31), 101.0),
        ("SPY", date(2000, 2, 29), 102.0),
        ("QQQ", date(1999, 12, 31), 50.0),
        ("QQQ", date(2000, 1, 31), 55.0),
        ("QQQ", date(2000, 2, 29), 60.5),
    ]
    prices = _prices_frame(sessions)
    research = _research_frame([("ff_mkt_monthly", date(2000, 1, 31), 0.01)])
    start, end = date(2000, 1, 31), date(2000, 2, 29)

    panel, records = splice_modern_panel(prices, research, ["SPY", "QQQ"], {}, _AS_OF, start, end)

    assert records == ()
    assert panel == panel_from_prices(prices, ["SPY", "QQQ"], _AS_OF, start, end)


def test_splice_modern_panel_proxy_fills_only_pre_inception_months() -> None:
    """Proxy months precede the first ETF month; the record spans the proxy range."""
    months = _month_ends(2000, 1, 6)
    spy_sessions = [("SPY", date(1999, 12, 31), 200.0)]
    spy_sessions.extend(("SPY", month, 200.0 + 10.0 * index) for index, month in enumerate(months))
    prices = _prices_frame(spy_sessions + _schd_sessions())
    research = _research_frame(
        [
            ("ff_dp_hi30_monthly", date(2000, 1, 31), 0.01),
            ("ff_dp_hi30_monthly", date(2000, 2, 29), 0.02),
            ("ff_dp_hi30_monthly", date(2000, 3, 31), 0.03),
        ]
    )
    rule = SpliceRule(
        sleeve="SCHD",
        proxy_weights={"ff_dp_hi30_monthly": 1.0},
        etf_first_month=date(2000, 4, 30),
    )

    panel, records = splice_modern_panel(
        prices, research, ["SPY", "SCHD"], {"SCHD": rule}, _AS_OF, date(2000, 1, 31), date(2000, 6, 30)
    )

    assert panel.tier == "modern"
    assert list(panel.months) == list(months)
    assert panel.returns["SCHD"] == pytest.approx((0.01, 0.02, 0.03, 0.10, 0.10, 0.10))
    expected_spy = panel_from_prices(prices, ["SPY"], _AS_OF, date(2000, 1, 31), date(2000, 6, 30))
    assert panel.returns["SPY"] == pytest.approx(expected_spy.returns["SPY"])
    assert records == (
        SpliceRecord(
            sleeve="SCHD",
            proxy_first_month=date(2000, 1, 31),
            proxy_last_month=date(2000, 3, 31),
            etf_first_month=date(2000, 4, 30),
            proxy_weights={"ff_dp_hi30_monthly": 1.0},
        ),
    )


def test_splice_modern_panel_blend_weights_combine_series() -> None:
    """A two-series proxy month returns the weighted sum within 1e-12."""
    prices = _prices_frame(
        [
            ("SCHD", date(2000, 1, 31), 100.0),
            ("SCHD", date(2000, 2, 29), 110.0),
        ]
    )
    research = _research_frame(
        [
            ("series_a", date(2000, 1, 31), 0.01),
            ("series_b", date(2000, 1, 31), 0.02),
        ]
    )
    rule = SpliceRule(
        sleeve="SCHD",
        proxy_weights={"series_a": 0.6, "series_b": 0.4},
        etf_first_month=date(2000, 2, 29),
    )

    panel, _ = splice_modern_panel(
        prices, research, ["SCHD"], {"SCHD": rule}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
    )

    assert panel.returns["SCHD"][0] == pytest.approx(0.014, abs=1e-12)


def test_splice_modern_panel_missing_proxy_month_fails_closed() -> None:
    """A proxy series without the pre-inception month names the series and month."""
    prices = _prices_frame(
        [
            ("SCHD", date(2000, 2, 29), 100.0),
            ("SCHD", date(2000, 3, 31), 110.0),
        ]
    )
    research = _research_frame([("ff_dp_hi30_monthly", date(2000, 1, 31), 0.01)])
    rule = SpliceRule(
        sleeve="SCHD",
        proxy_weights={"ff_dp_hi30_monthly": 1.0},
        etf_first_month=date(2000, 3, 31),
    )

    with pytest.raises(ValueError, match=r"ff_dp_hi30_monthly.*2000-02-29"):
        splice_modern_panel(
            prices, research, ["SCHD"], {"SCHD": rule}, _AS_OF, date(2000, 1, 31), date(2000, 3, 31)
        )


def test_splice_modern_panel_duplicate_proxy_month_fails_closed() -> None:
    """A proxy series with two rows for one month refuses to pick a winner."""
    prices = _prices_frame(
        [
            ("SCHD", date(2000, 1, 31), 100.0),
            ("SCHD", date(2000, 2, 29), 110.0),
        ]
    )
    base = _research_frame([("series_a", date(2000, 1, 31), 0.01)])
    research = pl.concat([base, base], how="vertical")
    rule = SpliceRule(
        sleeve="SCHD", proxy_weights={"series_a": 1.0}, etf_first_month=date(2000, 2, 29)
    )

    with pytest.raises(ValueError, match="duplicate month-ends"):
        splice_modern_panel(
            prices, research, ["SCHD"], {"SCHD": rule}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
        )


def test_splice_modern_panel_late_visible_proxy_row_fails_closed() -> None:
    """A proxy month whose only row postdates as_of fails closed."""
    prices = _prices_frame(
        [
            ("SCHD", date(2000, 1, 31), 100.0),
            ("SCHD", date(2000, 2, 29), 110.0),
        ]
    )
    research = _research_frame([("series_a", date(2000, 1, 31), 0.01)], lag_days=10_000)
    rule = SpliceRule(
        sleeve="SCHD", proxy_weights={"series_a": 1.0}, etf_first_month=date(2000, 2, 29)
    )

    with pytest.raises(ValueError, match="lacks a visible row"):
        splice_modern_panel(
            prices, research, ["SCHD"], {"SCHD": rule}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
        )


def test_splice_modern_panel_first_etf_month_without_prior_close_fails() -> None:
    """An ETF segment missing its prior month-end close propagates the price failure."""
    prices = _prices_frame(
        [
            ("SCHD", date(2000, 4, 15), 100.0),
            ("SCHD", date(2000, 4, 30), 110.0),
            ("SCHD", date(2000, 5, 31), 121.0),
        ]
    )
    research = _research_frame(
        [
            ("series_a", date(2000, 1, 31), 0.01),
            ("series_a", date(2000, 2, 29), 0.02),
            ("series_a", date(2000, 3, 31), 0.03),
        ]
    )
    rule = SpliceRule(
        sleeve="SCHD", proxy_weights={"series_a": 1.0}, etf_first_month=date(2000, 4, 30)
    )

    with pytest.raises(ValueError, match="prior month-end close"):
        splice_modern_panel(
            prices, research, ["SCHD"], {"SCHD": rule}, _AS_OF, date(2000, 1, 31), date(2000, 5, 31)
        )


def test_splice_modern_panel_first_month_on_start_is_rejected() -> None:
    """An etf_first_month equal to the window start leaves no proxy prefix."""
    prices = _prices_frame(
        [
            ("SCHD", date(1999, 12, 31), 100.0),
            ("SCHD", date(2000, 1, 31), 110.0),
        ]
    )
    research = _research_frame([("series_a", date(2000, 1, 31), 0.01)])
    rule = SpliceRule(
        sleeve="SCHD", proxy_weights={"series_a": 1.0}, etf_first_month=date(2000, 1, 31)
    )

    with pytest.raises(ValueError, match="must lie in"):
        splice_modern_panel(
            prices, research, ["SCHD"], {"SCHD": rule}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
        )


def test_splice_rule_rejects_degenerate_inputs() -> None:
    """Blank sleeves, bad weights, and non-month-end first months fail closed."""
    with pytest.raises(ValueError, match="non-blank string"):
        SpliceRule(sleeve="  ", proxy_weights={"a": 1.0}, etf_first_month=date(2000, 2, 29))
    with pytest.raises(ValueError, match="non-empty mapping"):
        SpliceRule(sleeve="SCHD", proxy_weights={}, etf_first_month=date(2000, 2, 29))
    with pytest.raises(ValueError, match="invalid series id"):
        SpliceRule(sleeve="SCHD", proxy_weights={"  ": 1.0}, etf_first_month=date(2000, 2, 29))
    with pytest.raises(ValueError, match="must be finite"):
        SpliceRule(
            sleeve="SCHD",
            proxy_weights={"a": float("inf")},
            etf_first_month=date(2000, 2, 29),
        )
    with pytest.raises(ValueError, match="non-negative"):
        SpliceRule(
            sleeve="SCHD",
            proxy_weights={"a": 1.5, "b": -0.5},
            etf_first_month=date(2000, 2, 29),
        )
    with pytest.raises(ValueError, match="must sum to 1"):
        SpliceRule(
            sleeve="SCHD",
            proxy_weights={"a": 0.6, "b": 0.3},
            etf_first_month=date(2000, 2, 29),
        )
    with pytest.raises(ValueError, match="must be a date"):
        SpliceRule(sleeve="SCHD", proxy_weights={"a": 1.0}, etf_first_month="2000-02-29")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a month-end"):
        SpliceRule(sleeve="SCHD", proxy_weights={"a": 1.0}, etf_first_month=date(2000, 2, 15))


def test_splice_modern_panel_rejects_bad_request_shapes() -> None:
    """Naive instants, duplicate sleeves, and unknown splice keys fail closed."""
    prices = _prices_frame([("SPY", date(2000, 1, 31), 100.0)])
    research = _research_frame([("a", date(2000, 1, 31), 0.01)])
    rule = SpliceRule(sleeve="SPY", proxy_weights={"a": 1.0}, etf_first_month=date(2000, 2, 29))
    with pytest.raises(ValueError, match="timezone-aware"):
        splice_modern_panel(
            prices, research, ["SPY"], {}, datetime(2001, 1, 1), date(2000, 1, 31), date(2000, 2, 29)
        )
    with pytest.raises(ValueError, match="non-empty and unique"):
        splice_modern_panel(prices, research, [], {}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29))
    with pytest.raises(ValueError, match="non-empty and unique"):
        splice_modern_panel(
            prices, research, ["SPY", "SPY"], {}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
        )
    with pytest.raises(ValueError, match="not a requested sleeve"):
        splice_modern_panel(
            prices, research, ["SPY"], {"BOND": rule}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
        )
    mismatched = SpliceRule(sleeve="QQQ", proxy_weights={"a": 1.0}, etf_first_month=date(2000, 2, 29))
    with pytest.raises(ValueError, match="differs from rule sleeve"):
        splice_modern_panel(
            prices, research, ["SPY"], {"SPY": mismatched}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
        )


def test_splice_modern_panel_rejects_bad_windows() -> None:
    """Inverted windows, month-free windows, and out-of-range first months fail."""
    prices = _prices_frame([("SPY", date(2000, 1, 31), 100.0)])
    research = _research_frame([("a", date(2000, 1, 31), 0.01)])
    rule = SpliceRule(sleeve="SPY", proxy_weights={"a": 1.0}, etf_first_month=date(2000, 2, 29))
    with pytest.raises(ValueError, match="is after"):
        splice_modern_panel(
            prices, research, ["SPY"], {"SPY": rule}, _AS_OF, date(2000, 3, 31), date(2000, 1, 31)
        )
    with pytest.raises(ValueError, match="covers no month-end"):
        splice_modern_panel(
            prices, research, ["SPY"], {"SPY": rule}, _AS_OF, date(2000, 1, 15), date(2000, 1, 20)
        )
    late = SpliceRule(sleeve="SPY", proxy_weights={"a": 1.0}, etf_first_month=date(2000, 3, 31))
    with pytest.raises(ValueError, match="must lie in"):
        splice_modern_panel(
            prices, research, ["SPY"], {"SPY": late}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
        )


def test_splice_modern_panel_empty_proxy_prefix_fails_closed() -> None:
    """A first month equal to the first grid month leaves no proxy month to blend."""
    prices = _prices_frame(
        [
            ("SPY", date(1999, 12, 31), 100.0),
            ("SPY", date(2000, 1, 31), 110.0),
            ("SPY", date(2000, 2, 29), 121.0),
        ]
    )
    research = _research_frame([("a", date(2000, 1, 31), 0.01)])
    rule = SpliceRule(sleeve="SPY", proxy_weights={"a": 1.0}, etf_first_month=date(2000, 1, 31))

    with pytest.raises(ValueError, match="covers no proxy month"):
        splice_modern_panel(
            prices, research, ["SPY"], {"SPY": rule}, _AS_OF, date(2000, 1, 15), date(2000, 2, 29)
        )


def test_splice_modern_panel_missing_research_column_fails_closed() -> None:
    """A research frame without the availability stamp cannot filter proxy months."""
    prices = _prices_frame(
        [
            ("SPY", date(2000, 1, 31), 100.0),
            ("SPY", date(2000, 2, 29), 110.0),
        ]
    )
    research = _research_frame([("a", date(2000, 1, 31), 0.01)]).drop("available_at")
    rule = SpliceRule(sleeve="SPY", proxy_weights={"a": 1.0}, etf_first_month=date(2000, 2, 29))

    with pytest.raises(ValueError, match="misses required column"):
        splice_modern_panel(
            prices, research, ["SPY"], {"SPY": rule}, _AS_OF, date(2000, 1, 31), date(2000, 2, 29)
        )


def test_realized_window_start_uses_latest_candidate_listing() -> None:
    """The window opens at the latest candidate etf_first_month."""
    splices = {
        "A": SpliceRule(sleeve="A", proxy_weights={"a": 1.0}, etf_first_month=date(2011, 11, 30)),
        "B": SpliceRule(sleeve="B", proxy_weights={"b": 1.0}, etf_first_month=date(2008, 7, 31)),
    }

    assert (
        realized_window_start(splices, {"A", "B"}, date(1999, 4, 30)) == date(2011, 11, 30)
    )


def test_realized_window_start_ignores_control_splice() -> None:
    """A splice on a sleeve no candidate holds leaves the window unaffected."""
    splices = {
        "A": SpliceRule(sleeve="A", proxy_weights={"a": 1.0}, etf_first_month=date(2011, 11, 30)),
        "CTRL": SpliceRule(
            sleeve="CTRL", proxy_weights={"a": 1.0}, etf_first_month=date(2020, 1, 31)
        ),
    }

    assert realized_window_start(splices, {"A"}, date(1999, 4, 30)) == date(2011, 11, 30)


def test_realized_window_start_without_splices_returns_modern_start() -> None:
    """With no candidate splice the realized window opens at modern_start."""
    assert realized_window_start({}, {"A"}, date(1999, 4, 30)) == date(1999, 4, 30)


def test_realized_panel_has_no_proxy_months() -> None:
    """The realized panel starts at the listing month with price-derived returns."""
    months = _month_ends(2000, 1, 6)
    sessions = [("SCHD", date(1999, 12, 31), 200.0)]
    sessions.extend(("SCHD", month, 200.0 + 10.0 * index) for index, month in enumerate(months))
    prices = _prices_frame(sessions)
    research = _research_frame(
        [
            ("ff_dp_hi30_monthly", date(2000, 1, 31), 0.01),
            ("ff_dp_hi30_monthly", date(2000, 2, 29), 0.02),
            ("ff_dp_hi30_monthly", date(2000, 3, 31), 0.03),
        ]
    )
    del research

    panel = realized_panel_from_prices(
        prices, ["SCHD"], _AS_OF, date(2000, 4, 30), date(2000, 6, 30)
    )
    expected = panel_from_prices(prices, ["SCHD"], _AS_OF, date(2000, 4, 30), date(2000, 6, 30))

    assert panel.tier == "realized"
    assert list(panel.months) == [date(2000, 4, 30), date(2000, 5, 31), date(2000, 6, 30)]
    assert panel.returns["SCHD"] == pytest.approx(expected.returns["SCHD"])


def test_realized_panel_without_prior_close_fails_closed() -> None:
    """Prices starting inside the first requested month fail with ValueError."""
    prices = _prices_frame(
        [
            ("SCHD", date(2000, 4, 30), 110.0),
            ("SCHD", date(2000, 5, 31), 121.0),
        ]
    )

    with pytest.raises(ValueError, match="prior month-end close"):
        realized_panel_from_prices(prices, ["SCHD"], _AS_OF, date(2000, 4, 30), date(2000, 5, 31))
