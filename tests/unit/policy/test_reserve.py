"""Unit tests for the explicit contribution reserve ledger."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.pit import AVAILABLE_AT
from src.policy.reserve import ReserveConfig, apply_reserve_schedule
from src.policy.targets import PolicyError

_CALENDAR = load_calendar("XNYS")
_TICKER = "VTI"
_PANEL_DAYS = _CALENDAR.sessions(date(2022, 12, 1), date(2024, 3, 28))
_SIGNAL_AT = _CALENDAR.close_ts(_PANEL_DAYS[-1])
_CONTRIBUTION_KRW = 1_000_000.0


def _price_panel(closes: list[float]) -> pl.DataFrame:
    """Single-ticker PIT frame whose ``adjusted_close`` row order matches the session list."""
    return pl.DataFrame(
        {
            "ticker": [_TICKER] * len(closes),
            "date": list(_PANEL_DAYS[: len(closes)]),
            "adjusted_close": closes,
            AVAILABLE_AT: [_CALENDAR.close_ts(day) for day in _PANEL_DAYS[: len(closes)]],
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            AVAILABLE_AT: pl.Datetime("us", "UTC"),
        },
    )


def _rising_closes() -> list[float]:
    """Monotonic 0.1%/session closes: positive trend, no drawdown."""
    return [100.0 * 1.001**index for index in range(len(_PANEL_DAYS))]


def _crash_closes() -> list[float]:
    """Flat closes that fall 22% below the running peak inside the final window."""
    flat = [100.0] * (len(_PANEL_DAYS) - 5)
    return [*flat, *[78.0] * 5]


def _moderate_drawdown_closes() -> list[float]:
    """Flat closes that fall 12% below the running peak (depth in [0.10, 0.20))."""
    flat = [100.0] * (len(_PANEL_DAYS) - 5)
    return [*flat, *[88.0] * 5]


def _vix_frame(value: float, series_id: str = "VIXCLS") -> pl.DataFrame:
    """Single vintage macro row published one day before the signal instant."""
    return pl.DataFrame(
        {
            "series_id": [series_id],
            "observation_date": [date(2024, 2, 1)],
            "value": [value],
            AVAILABLE_AT: [_SIGNAL_AT - timedelta(days=1)],
        },
        schema={
            "series_id": pl.String,
            "observation_date": pl.Date,
            "value": pl.Float64,
            AVAILABLE_AT: pl.Datetime("us", "UTC"),
        },
    )


_V3_CONFIG = ReserveConfig(
    max_withhold=0.10, schedule="v3", min_invest_multiplier=0.70, max_invest_multiplier=3.0
)


@pytest.mark.parametrize("scenario_id", ["RSV-A-identity-and-bounds"])
def test_rsv_a_identity_and_bounds(scenario_id: str) -> None:
    """RSV-A-identity-and-bounds"""
    assert ReserveConfig(max_withhold=0.10).max_withhold == pytest.approx(0.10)
    with pytest.raises(ValueError, match="max_withhold"):
        ReserveConfig(max_withhold=0.0)
    with pytest.raises(ValueError, match="max_withhold"):
        ReserveConfig(max_withhold=0.11)


@pytest.mark.parametrize("scenario_id", ["RSV-A-identity-and-bounds"])
def test_rsv_a_withhold_on_rising_trend(scenario_id: str) -> None:
    """RSV-A-identity-and-bounds"""
    config = ReserveConfig(max_withhold=0.10)
    cap = 0.10 * _CONTRIBUTION_KRW

    withheld = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )

    assert withheld.investable_krw == pytest.approx(_CONTRIBUTION_KRW * 0.90)
    assert withheld.reserve_krw == pytest.approx(500_000.0 + cap)


@pytest.mark.parametrize("scenario_id", ["RSV-A-identity-and-bounds"])
def test_rsv_a_deploy_on_deep_drawdown(scenario_id: str) -> None:
    """RSV-A-identity-and-bounds"""
    config = ReserveConfig(max_withhold=0.10)
    panel = _price_panel(_crash_closes())
    cap = 0.10 * _CONTRIBUTION_KRW

    full_deploy = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=panel,
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert full_deploy.investable_krw == pytest.approx(_CONTRIBUTION_KRW + min(500_000.0, cap))
    assert full_deploy.reserve_krw == pytest.approx(500_000.0 - min(500_000.0, cap))

    partial_deploy = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=50_000.0,
        prices=panel,
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert partial_deploy.investable_krw == pytest.approx(_CONTRIBUTION_KRW + 50_000.0)
    assert partial_deploy.reserve_krw == pytest.approx(0.0)

    # Flat closes keep the trend at exactly zero, so neither rule fires.
    passthrough = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=0.0,
        prices=_price_panel([100.0] * len(_PANEL_DAYS)),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert passthrough.investable_krw == pytest.approx(_CONTRIBUTION_KRW)
    assert passthrough.reserve_krw == pytest.approx(0.0)


@pytest.mark.parametrize("scenario_id", ["RSV-A-identity-and-bounds"])
def test_rsv_a_fail_closed_guards(scenario_id: str) -> None:
    """RSV-A-identity-and-bounds"""
    config = ReserveConfig(max_withhold=0.10)
    naive_signal = datetime(2024, 3, 28, 21)
    with pytest.raises((ValueError, PolicyError)):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=0.0,
            prices=_price_panel(_rising_closes()),
            ticker=_TICKER,
            signal_at=naive_signal,
            config=config,
        )
    short_panel = _price_panel(_rising_closes()[-30:])
    with pytest.raises(PolicyError):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=0.0,
            prices=short_panel,
            ticker=_TICKER,
            signal_at=_SIGNAL_AT,
            config=config,
        )



@pytest.mark.parametrize("scenario_id", ["RSV-V2-piecewise-and-i6"])
def test_rsv_v2_piecewise_and_i6(scenario_id: str) -> None:
    """RSV-V2-piecewise-and-i6"""
    config = ReserveConfig(max_withhold=0.10, schedule="v2")

    fill = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert fill.investable_krw == pytest.approx(0.80 * _CONTRIBUTION_KRW)
    assert fill.reserve_krw == pytest.approx(700_000.0)
    assert fill.investable_krw + fill.reserve_krw == pytest.approx(1_500_000.0)

    mild_panel = _price_panel([*[100.0] * (len(_PANEL_DAYS) - 1), 85.0])
    expected_m = 1.25 + 0.25 * 0.5
    deploy = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=mild_panel,
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert deploy.investable_krw == pytest.approx(min(expected_m * _CONTRIBUTION_KRW, 1_500_000.0))
    assert abs((deploy.investable_krw + deploy.reserve_krw) - 1_500_000.0) <= 1e-9

    broke = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=0.0,
        prices=mild_panel,
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert broke.investable_krw == pytest.approx(min(expected_m * _CONTRIBUTION_KRW, _CONTRIBUTION_KRW))
    assert broke.reserve_krw == pytest.approx(0.0)

    deep_panel = _price_panel([*[100.0] * (len(_PANEL_DAYS) - 1), 70.0])
    deep = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=1_500_000.0,
        prices=deep_panel,
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert deep.investable_krw == pytest.approx(2.0 * _CONTRIBUTION_KRW)
    assert deep.reserve_krw == pytest.approx(500_000.0)
    assert abs((deep.investable_krw + deep.reserve_krw) - 2_500_000.0) <= 1e-9
    assert deep.investable_krw <= _CONTRIBUTION_KRW + 1_500_000.0


@pytest.mark.parametrize("scenario_id", ["RSV-V2-stock-cap-and-fail-closed"])
def test_rsv_v2_stock_cap_and_fail_closed(scenario_id: str) -> None:
    """RSV-V2-stock-cap-and-fail-closed"""
    config = ReserveConfig(max_withhold=0.10, schedule="v2")

    capped = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=6.0 * _CONTRIBUTION_KRW,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert capped.reserve_krw == pytest.approx(6.0 * _CONTRIBUTION_KRW)
    assert capped.investable_krw == pytest.approx(_CONTRIBUTION_KRW)
    assert abs((capped.investable_krw + capped.reserve_krw) - 7.0 * _CONTRIBUTION_KRW) <= 1e-9

    with pytest.raises(ValueError, match="min_invest_multiplier"):
        ReserveConfig(max_withhold=0.10, schedule="v2", min_invest_multiplier=1.0)
    with pytest.raises(ValueError, match="max_invest_multiplier"):
        ReserveConfig(max_withhold=0.10, schedule="v2", max_invest_multiplier=2.5)
    with pytest.raises(ValueError, match="reserve_max_months"):
        ReserveConfig(max_withhold=0.10, schedule="v2", reserve_max_months=6.5)
    bad_schedule: str = "v5"
    with pytest.raises(ValueError, match="schedule"):
        ReserveConfig(max_withhold=0.10, schedule=bad_schedule)  # type: ignore[arg-type]

    with pytest.raises(PolicyError):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=0.0,
            prices=_price_panel(_rising_closes()),
            ticker=_TICKER,
            signal_at=datetime(2024, 3, 28, 21),
            config=config,
        )
    with pytest.raises(PolicyError):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=0.0,
            prices=_price_panel(_rising_closes()[-30:]),
            ticker=_TICKER,
            signal_at=_SIGNAL_AT,
            config=config,
        )


@pytest.mark.parametrize("scenario_id", ["RSV-V3-identity-and-erp"])
def test_rsv_v3_complacent_bull_withholds(scenario_id: str) -> None:
    """RSV-V3-identity-and-erp"""
    decision = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=_V3_CONFIG,
        macro=_vix_frame(15.0),
    )

    assert decision.investable_krw == pytest.approx(700_000.0)
    assert decision.reserve_krw == pytest.approx(800_000.0)
    assert abs((decision.investable_krw + decision.reserve_krw) - 1_500_000.0) <= 1e-9


@pytest.mark.parametrize("scenario_id", ["RSV-V3-identity-and-erp"])
def test_rsv_v3_elevated_vix_deploys(scenario_id: str) -> None:
    """RSV-V3-identity-and-erp"""
    decision = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=_V3_CONFIG,
        macro=_vix_frame(40.0),
    )

    # VIX crisis branch: the 1.5m stock caps the boosted buy.
    assert decision.investable_krw == pytest.approx(1_500_000.0)
    assert decision.reserve_krw == pytest.approx(0.0)
    assert abs((decision.investable_krw + decision.reserve_krw) - 1_500_000.0) <= 1e-9


@pytest.mark.parametrize("scenario_id", ["RSV-V3-identity-and-erp"])
def test_rsv_v3_subthreshold_vix_withholds(scenario_id: str) -> None:
    """RSV-V3-identity-and-erp"""
    decision = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=0.0,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=_V3_CONFIG,
        macro=_vix_frame(22.0),
    )

    assert decision.investable_krw == pytest.approx(700_000.0)
    assert decision.reserve_krw == pytest.approx(300_000.0)
    assert abs((decision.investable_krw + decision.reserve_krw) - _CONTRIBUTION_KRW) <= 1e-9


@pytest.mark.parametrize("scenario_id", ["RSV-V3-identity-and-erp"])
def test_rsv_v3_drawdown_depth_scales(scenario_id: str) -> None:
    """RSV-V3-identity-and-erp"""
    expected_m = 1.0 + (0.22 / 0.30) * (_V3_CONFIG.max_invest_multiplier - 1.0)
    decision = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=_price_panel(_crash_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=_V3_CONFIG,
        macro=_vix_frame(15.0),
    )

    assert decision.investable_krw == pytest.approx(min(expected_m * _CONTRIBUTION_KRW, 1_500_000.0))
    assert decision.reserve_krw == pytest.approx(max(0.0, 1_500_000.0 - expected_m * _CONTRIBUTION_KRW))
    assert abs((decision.investable_krw + decision.reserve_krw) - 1_500_000.0) <= 1e-9


@pytest.mark.parametrize("scenario_id", ["RSV-V3-identity-and-erp"])
def test_rsv_v3_fail_closed_without_macro(scenario_id: str) -> None:
    """RSV-V3-identity-and-erp"""
    with pytest.raises(PolicyError):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=500_000.0,
            prices=_price_panel(_rising_closes()),
            ticker=_TICKER,
            signal_at=_SIGNAL_AT,
            config=_V3_CONFIG,
        )
    with pytest.raises(PolicyError):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=500_000.0,
            prices=_price_panel(_rising_closes()),
            ticker=_TICKER,
            signal_at=_SIGNAL_AT,
            config=_V3_CONFIG,
            macro=_vix_frame(15.0).clear(),
        )
    with pytest.raises(PolicyError):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=500_000.0,
            prices=_price_panel(_rising_closes()),
            ticker=_TICKER,
            signal_at=_SIGNAL_AT,
            config=_V3_CONFIG,
            macro=_vix_frame(15.0, series_id="OTHER"),
        )


@pytest.mark.parametrize("scenario_id", ["RSV-V3-identity-and-erp"])
def test_rsv_v3_bounds_and_defaults(scenario_id: str) -> None:
    """RSV-V3-identity-and-erp"""
    with pytest.raises(ValueError, match="max_invest_multiplier"):
        ReserveConfig(max_withhold=0.10, schedule="v2", max_invest_multiplier=2.5)
    assert (
        ReserveConfig(max_withhold=0.10, schedule="v3", max_invest_multiplier=3.0).max_invest_multiplier
        == pytest.approx(3.0)
    )
    with pytest.raises(ValueError, match="max_invest_multiplier"):
        ReserveConfig(max_withhold=0.10, schedule="v3", max_invest_multiplier=3.1)

    defaults = ReserveConfig(max_withhold=0.10, schedule="v3")
    assert defaults.min_invest_multiplier == pytest.approx(0.70)
    assert defaults.max_invest_multiplier == pytest.approx(3.0)
    assert defaults.vix_threshold == pytest.approx(25.0)
    assert defaults.vix_series_id == "VIXCLS"


@pytest.mark.parametrize("scenario_id", ["RSV-V3-mid-pass-through"])
def test_rsv_v3_mid_pass_through(scenario_id: str) -> None:
    """RSV-V3-mid-pass-through"""
    decision = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=0.0,
        prices=_price_panel(_moderate_drawdown_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=_V3_CONFIG,
        macro=_vix_frame(15.0),
    )

    assert decision.investable_krw == pytest.approx(_CONTRIBUTION_KRW)
    assert decision.reserve_krw == pytest.approx(0.0)
    assert abs((decision.investable_krw + decision.reserve_krw) - _CONTRIBUTION_KRW) <= 1e-9


_V4_CONFIG = ReserveConfig(
    max_withhold=0.10, schedule="v4", min_invest_multiplier=0.70, max_invest_multiplier=3.0
)


@pytest.mark.parametrize("scenario_id", ["RSV-V4-ledger"])
def test_rsv_v4_ledger(scenario_id: str) -> None:
    """RSV-V4-ledger"""
    passthrough = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=0.0,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=_V4_CONFIG,
    )
    assert passthrough.investable_krw == pytest.approx(_CONTRIBUTION_KRW)
    assert passthrough.reserve_krw == pytest.approx(0.0)

    withheld = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=_V4_CONFIG,
    )
    assert withheld.investable_krw == pytest.approx(_CONTRIBUTION_KRW)
    assert withheld.reserve_krw == pytest.approx(500_000.0)
    assert abs((withheld.investable_krw + withheld.reserve_krw) - 1_500_000.0) <= 1e-9

    deploy = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=_price_panel(_crash_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=_V4_CONFIG,
    )
    assert deploy.investable_krw == pytest.approx(1_500_000.0)
    assert deploy.reserve_krw == pytest.approx(0.0)
    assert abs((deploy.investable_krw + deploy.reserve_krw) - 1_500_000.0) <= 1e-9

    defaults = ReserveConfig(max_withhold=0.10, schedule="v4")
    assert defaults.min_invest_multiplier == pytest.approx(0.70)
    assert defaults.max_invest_multiplier == pytest.approx(3.0)

    bad_schedule: str = "v5"
    with pytest.raises(ValueError, match="schedule"):
        ReserveConfig(max_withhold=0.10, schedule=bad_schedule)  # type: ignore[arg-type]
