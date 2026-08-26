"""Unit tests for the KAFI composite: bounds, PIT fail-closed, and vol dampener polarity."""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Final

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.features.kafi import KAFI_COMPONENT_IDS, kafi_components, kafi_opportunity_components, kafi_opportunity_score, kafi_score

_CALENDAR = load_calendar("XNYS")
_SESSIONS: Final[tuple[date, ...]] = tuple(_CALENDAR.sessions(date(2022, 1, 3), date(2023, 12, 29)))
_RANK_WINDOW = 63
_CUTOFF_INDEX = 340
_SIGNAL_AT = _CALENDAR.close_ts(_SESSIONS[_CUTOFF_INDEX - 1])
_PRICE_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "ticker": pl.String,
        "date": pl.Date,
        "adjusted_close": pl.Float64,
        "available_at": pl.Datetime("us", "UTC"),
    }
)
_FX_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "date": pl.Date,
        "usdkrw": pl.Float64,
        "available_at": pl.Datetime("us", "UTC"),
    }
)
_MACRO_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "series_id": pl.String,
        "observation_date": pl.Date,
        "release_date": pl.Datetime("us", "UTC"),
        "value": pl.Float64,
        "available_at": pl.Datetime("us", "UTC"),
    }
)


def _walk(seed: int, *, start: float, n: int, drift: float, vol: float) -> list[float]:
    rng = random.Random(seed)  # noqa: S311
    level = start
    path: list[float] = []
    for _ in range(n):
        level *= 1.0 + drift + rng.gauss(0.0, vol)
        path.append(round(level, 6))
    return path


def _prices_panel(paths: dict[str, list[float]], *, sessions: tuple[date, ...] | None = None) -> pl.DataFrame:
    days = sessions if sessions is not None else _SESSIONS
    frame = pl.DataFrame(
        {
            "ticker": [ticker for ticker, path in paths.items() for _ in path],
            "date": [day for path in paths.values() for day in days[: len(path)]],
            "adjusted_close": [close for path in paths.values() for close in path],
            "available_at": [
                _CALENDAR.close_ts(day) for path in paths.values() for day in days[: len(path)]
            ],
        },
        schema=_PRICE_SCHEMA,
    )
    return frame


def _fx_panel(
    path: list[float],
    *,
    sessions: tuple[date, ...] | None = None,
    available_shift: timedelta | None = None,
) -> pl.DataFrame:
    days = (sessions if sessions is not None else _SESSIONS)[: len(path)]
    stamps = [_CALENDAR.close_ts(day) for day in days]
    if available_shift is not None:
        stamps = [stamp + available_shift for stamp in stamps]
    return pl.DataFrame(
        {
            "date": list(days),
            "usdkrw": path,
            "available_at": stamps,
        },
        schema=_FX_SCHEMA,
    )


def _macro_panel(path: list[float], series_id: str = "BAA10Y") -> pl.DataFrame:
    days = _SESSIONS[: len(path)]
    releases = [_CALENDAR.close_ts(day) for day in days]
    return pl.DataFrame(
        {
            "series_id": [series_id] * len(path),
            "observation_date": list(days),
            "release_date": releases,
            "value": path,
            "available_at": releases,
        },
        schema=_MACRO_SCHEMA,
    )


def _default_panels() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    prices = _prices_panel(
        {
            "QQQ": _walk(7, start=100.0, n=len(_SESSIONS), drift=0.0002, vol=0.012),
            "IEF": _walk(11, start=100.0, n=len(_SESSIONS), drift=0.0001, vol=0.003),
        }
    )
    fx = _fx_panel(_walk(13, start=1300.0, n=len(_SESSIONS), drift=0.0001, vol=0.004))
    macro = _macro_panel(_walk(17, start=3.5, n=len(_SESSIONS), drift=0.0, vol=0.05))
    return prices, fx, macro


@pytest.mark.parametrize("scenario_id", ["KAFI-A-score-bounds"])
def test_kafi_a_score_bounds(scenario_id: str) -> None:
    """KAFI-A-score-bounds"""
    prices, fx, macro = _default_panels()

    components = kafi_components(
        prices=prices,
        fx=fx,
        macro=macro,
        equity_ticker="QQQ",
        bond_ticker="IEF",
        signal_at=_SIGNAL_AT,
        rank_window=_RANK_WINDOW,
    )
    score = kafi_score(
        prices=prices,
        fx=fx,
        macro=macro,
        equity_ticker="QQQ",
        bond_ticker="IEF",
        signal_at=_SIGNAL_AT,
        rank_window=_RANK_WINDOW,
    )

    assert set(components) == set(KAFI_COMPONENT_IDS)
    for name, value in components.items():
        assert 0.0 <= value <= 100.0, name
        assert isinstance(value, float)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0
    assert score == pytest.approx(sum(components.values()) / len(components))


@pytest.mark.parametrize("scenario_id", ["KAFI-B-pit-fail-closed"])
def test_kafi_b_pit_fail_closed(scenario_id: str) -> None:
    """KAFI-B-pit-fail-closed"""
    prices, fx, macro = _default_panels()
    shift = timedelta(days=800)
    future_macro = macro.with_columns(pl.col("release_date").add(shift)).with_columns(
        pl.col("available_at").add(shift)
    )
    future_fx = _fx_panel(
        _walk(13, start=1300.0, n=len(_SESSIONS), drift=0.0001, vol=0.004),
        available_shift=shift,
    )

    with pytest.raises(ValueError, match="credit_oas"):
        kafi_score(
            prices=prices,
            fx=fx,
            macro=future_macro,
            equity_ticker="QQQ",
            bond_ticker="IEF",
            signal_at=_SIGNAL_AT,
            rank_window=_RANK_WINDOW,
        )
    with pytest.raises(ValueError, match="fx_stress"):
        kafi_score(
            prices=prices,
            fx=future_fx,
            macro=macro,
            equity_ticker="QQQ",
            bond_ticker="IEF",
            signal_at=_SIGNAL_AT,
            rank_window=_RANK_WINDOW,
        )
    short_prices = _prices_panel(
        {
            "QQQ": _walk(7, start=100.0, n=180, drift=0.0002, vol=0.012),
            "IEF": _walk(11, start=100.0, n=180, drift=0.0001, vol=0.003),
        }
    )
    with pytest.raises(ValueError, match="momentum"):
        kafi_components(
            prices=short_prices,
            fx=fx,
            macro=macro,
            equity_ticker="QQQ",
            bond_ticker="IEF",
            signal_at=_SIGNAL_AT,
            rank_window=_RANK_WINDOW,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        kafi_score(
            prices=prices,
            fx=fx,
            macro=macro,
            equity_ticker="QQQ",
            bond_ticker="IEF",
            signal_at=datetime(2023, 6, 30),
            rank_window=_RANK_WINDOW,
        )


def _constant_panels() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    flat = [100.0] * (_CUTOFF_INDEX + 40)
    prices = _prices_panel({"QQQ": list(flat), "IEF": list(flat)})
    fx = _fx_panel([1300.0] * len(flat))
    macro = _macro_panel([3.5] * len(flat))
    return prices, fx, macro


@pytest.mark.parametrize("scenario_id", ["KAFI-C-vol-not-buy-trigger"])
def test_kafi_c_vol_not_buy_trigger(scenario_id: str) -> None:
    """KAFI-C-vol-not-buy-trigger"""
    base_prices, base_fx, base_macro = _constant_panels()
    oscillating = [100.0] * (_CUTOFF_INDEX - 40)
    for index in range(40):
        oscillating.append(round(100.0 * (1.02 if index % 2 == 0 else 0.98), 6))
    spiky_prices = _prices_panel({"QQQ": oscillating, "IEF": [100.0] * len(oscillating)})

    kwargs: dict[str, object] = {
        "equity_ticker": "QQQ",
        "bond_ticker": "IEF",
        "signal_at": _SIGNAL_AT,
        "rank_window": _RANK_WINDOW,
    }
    base = kafi_components(prices=base_prices, fx=base_fx, macro=base_macro, **kwargs)  # type: ignore[arg-type]
    spiky = kafi_components(prices=spiky_prices, fx=base_fx, macro=base_macro, **kwargs)  # type: ignore[arg-type]

    for value in base.values():
        assert value == pytest.approx(50.0)
    # The realized-vol spike must move only the dampener, upward, and stay compressed around neutral.
    assert spiky["vol_dampener"] > base["vol_dampener"]
    assert spiky["vol_dampener"] <= 50.0 + 12.5 + 1e-9
    isolated = dict(base)
    isolated["vol_dampener"] = spiky["vol_dampener"]
    isolated_score = sum(isolated.values()) / len(isolated)
    assert abs(isolated_score - 50.0) <= 100.0 / 6.0 + 1e-9
    assert isolated_score >= 50.0 - 1e-9


@pytest.mark.parametrize("scenario_id", ["KAFI-D-credit-series-wiring"])
def test_kafi_d_credit_series_wiring(scenario_id: str) -> None:
    """KAFI-D-credit-series-wiring"""
    prices, fx, _macro = _default_panels()
    baa_macro = _macro_panel(_walk(19, start=3.5, n=len(_SESSIONS), drift=0.0, vol=0.05), series_id="BAA10Y")
    other_macro = _macro_panel(_walk(23, start=9.0, n=len(_SESSIONS), drift=0.0, vol=0.02), series_id="OTHER")

    baa_score = kafi_score(
        prices=prices,
        fx=fx,
        macro=baa_macro,
        equity_ticker="QQQ",
        bond_ticker="IEF",
        signal_at=_SIGNAL_AT,
        rank_window=_RANK_WINDOW,
        credit_series_id="BAA10Y",
    )
    with pytest.raises(ValueError, match="credit_oas"):
        kafi_score(
            prices=prices,
            fx=fx,
            macro=other_macro,
            equity_ticker="QQQ",
            bond_ticker="IEF",
            signal_at=_SIGNAL_AT,
            rank_window=_RANK_WINDOW,
            credit_series_id="BAA10Y",
        )
    assert 0.0 <= baa_score <= 100.0


@pytest.mark.parametrize("scenario_id", ["KAFI-D-pit-macro-dedup"])
def test_kafi_d_pit_macro_dedup(scenario_id: str) -> None:
    """KAFI-D-pit-macro-dedup"""
    prices, fx, _ = _default_panels()
    days = _SESSIONS[:_CUTOFF_INDEX]
    last_day = days[-1]
    macro_base = _macro_panel([3.0] * len(days))
    macro = pl.concat(
        [
            macro_base,
            pl.DataFrame(
                {
                    "series_id": ["BAA10Y"],
                    "observation_date": [last_day],
                    "release_date": [_SIGNAL_AT + timedelta(days=30)],
                    "value": [3.0],
                    "available_at": [_SIGNAL_AT + timedelta(days=30)],
                },
                schema=_MACRO_SCHEMA,
            ),
            pl.DataFrame(
                {
                    "series_id": ["BAA10Y"],
                    "observation_date": [last_day],
                    "release_date": [_SIGNAL_AT],
                    "value": [9.0],
                    "available_at": [_SIGNAL_AT],
                },
                schema=_MACRO_SCHEMA,
            ),
        ]
    )
    components = kafi_components(
        prices=prices,
        fx=fx,
        macro=macro,
        equity_ticker="QQQ",
        bond_ticker="IEF",
        signal_at=_SIGNAL_AT,
        rank_window=_RANK_WINDOW,
        credit_series_id="BAA10Y",
    )
    assert components["credit_oas"] > 98.0


@pytest.mark.parametrize("scenario_id", ["KAFI-OPP-neutral-score"])
def test_kafi_opp_neutral_score(scenario_id: str) -> None:
    """KAFI-OPP-neutral-score"""
    prices, fx, macro = _constant_panels()
    kwargs: dict[str, object] = {
        "equity_ticker": "QQQ",
        "bond_ticker": "IEF",
        "signal_at": _SIGNAL_AT,
        "rank_window": _RANK_WINDOW,
    }
    components = kafi_opportunity_components(prices=prices, fx=fx, macro=macro, **kwargs)  # type: ignore[arg-type]
    score = kafi_opportunity_score(prices=prices, fx=fx, macro=macro, **kwargs)  # type: ignore[arg-type]

    for value in components.values():
        assert value == pytest.approx(50.0)
    assert score == pytest.approx(50.0)


@pytest.mark.parametrize("scenario_id", ["KAFI-OPP-invert-momentum"])
def test_kafi_opp_invert_momentum(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """KAFI-OPP-invert-momentum"""
    import src.features.kafi as kafi_module

    def fake_greed_components(**_kwargs: object) -> dict[str, float]:
        return {
            "momentum": 80.0,
            "drawdown_depth": 60.0,
            "equity_bond_rel": 70.0,
            "credit_oas": 40.0,
            "fx_stress": 30.0,
            "vol_dampener": 55.0,
        }

    monkeypatch.setattr(kafi_module, "kafi_components", fake_greed_components)
    components = kafi_opportunity_components(
        prices=pl.DataFrame(),
        fx=pl.DataFrame(),
        macro=pl.DataFrame(),
        equity_ticker="QQQ",
        bond_ticker="IEF",
        signal_at=_SIGNAL_AT,
        rank_window=_RANK_WINDOW,
    )

    assert components["momentum"] == pytest.approx(20.0)
    assert components["drawdown_depth"] == pytest.approx(60.0)
    assert components["equity_bond_rel"] == pytest.approx(30.0)
    assert components["fx_stress"] == pytest.approx(70.0)
    assert components["vol_dampener"] == pytest.approx(50.0)


@pytest.mark.parametrize("scenario_id", ["KAFI-OPP-v2-exclude-vol"])
def test_kafi_opp_v2_exclude_vol(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """KAFI-OPP-v2-exclude-vol"""
    import src.features.kafi as kafi_module

    def fake_greed_components(**_kwargs: object) -> dict[str, float]:
        return {
            "momentum": 80.0,
            "drawdown_depth": 60.0,
            "equity_bond_rel": 70.0,
            "credit_oas": 40.0,
            "fx_stress": 30.0,
            "vol_dampener": 55.0,
        }

    monkeypatch.setattr(kafi_module, "kafi_components", fake_greed_components)
    kwargs: dict[str, object] = {
        "equity_ticker": "QQQ",
        "bond_ticker": "IEF",
        "signal_at": _SIGNAL_AT,
        "rank_window": _RANK_WINDOW,
    }
    components = kafi_opportunity_components(
        prices=pl.DataFrame(), fx=pl.DataFrame(), macro=pl.DataFrame(), **kwargs
    )  # type: ignore[arg-type]
    assert components["vol_dampener"] == pytest.approx(50.0)
    expected_five = (
        sum(value for name, value in components.items() if name != "vol_dampener")
        / (len(components) - 1)
    )

    score_five = kafi_opportunity_score(
        prices=pl.DataFrame(),
        fx=pl.DataFrame(),
        macro=pl.DataFrame(),
        include_vol_dampener=False,
        **kwargs,
    )  # type: ignore[arg-type]
    score_six = kafi_opportunity_score(
        prices=pl.DataFrame(),
        fx=pl.DataFrame(),
        macro=pl.DataFrame(),
        include_vol_dampener=True,
        **kwargs,
    )  # type: ignore[arg-type]

    assert abs(score_five - expected_five) <= 1e-9
    assert abs(score_five - score_six) > 1e-9


@pytest.mark.parametrize("scenario_id", ["KAFI-OPP-disp-identity"])
def test_kafi_opp_disp_identity(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """KAFI-OPP-disp-identity"""
    import src.features.kafi as kafi_module

    def fake_greed_components(**_kwargs: object) -> dict[str, float]:
        return {
            "momentum": 80.0,
            "drawdown_depth": 60.0,
            "equity_bond_rel": 70.0,
            "credit_oas": 40.0,
            "fx_stress": 30.0,
            "vol_dampener": 55.0,
        }

    monkeypatch.setattr(kafi_module, "kafi_components", fake_greed_components)
    kwargs: dict[str, object] = {
        "equity_ticker": "QQQ",
        "bond_ticker": "IEF",
        "signal_at": _SIGNAL_AT,
        "rank_window": _RANK_WINDOW,
    }
    score_default = kafi_opportunity_score(
        prices=pl.DataFrame(),
        fx=pl.DataFrame(),
        macro=pl.DataFrame(),
        include_vol_dampener=True,
        dispersion=1.0,
        **kwargs,
    )  # type: ignore[arg-type]
    score_half = kafi_opportunity_score(
        prices=pl.DataFrame(),
        fx=pl.DataFrame(),
        macro=pl.DataFrame(),
        include_vol_dampener=True,
        dispersion=0.5,
        **kwargs,
    )  # type: ignore[arg-type]

    assert abs(score_default - score_half) > 1e-9
    assert abs(score_half - (50.0 + 0.5 * (score_default - 50.0))) <= 1e-9
    with pytest.raises(ValueError, match="dispersion"):
        kafi_opportunity_score(
            prices=pl.DataFrame(),
            fx=pl.DataFrame(),
            macro=pl.DataFrame(),
            dispersion=0.0,
            **kwargs,
        )  # type: ignore[arg-type]


@pytest.mark.parametrize("scenario_id", ["KAFI-OPP-disp-novol-bridge"])
def test_kafi_opp_disp_novol_bridge(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """KAFI-OPP-disp-novol-bridge"""
    import src.features.kafi as kafi_module

    def fake_greed_components(**_kwargs: object) -> dict[str, float]:
        return {
            "momentum": 80.0,
            "drawdown_depth": 60.0,
            "equity_bond_rel": 70.0,
            "credit_oas": 40.0,
            "fx_stress": 30.0,
            "vol_dampener": 55.0,
        }

    monkeypatch.setattr(kafi_module, "kafi_components", fake_greed_components)
    kwargs: dict[str, object] = {
        "equity_ticker": "QQQ",
        "bond_ticker": "IEF",
        "signal_at": _SIGNAL_AT,
        "rank_window": _RANK_WINDOW,
    }
    score_equal6 = kafi_opportunity_score(
        prices=pl.DataFrame(),
        fx=pl.DataFrame(),
        macro=pl.DataFrame(),
        include_vol_dampener=True,
        dispersion=1.0,
        **kwargs,
    )  # type: ignore[arg-type]
    score_bridge = kafi_opportunity_score(
        prices=pl.DataFrame(),
        fx=pl.DataFrame(),
        macro=pl.DataFrame(),
        include_vol_dampener=False,
        dispersion=5.0 / 6.0,
        **kwargs,
    )  # type: ignore[arg-type]

    assert abs(score_bridge - score_equal6) <= 1e-9
