"""Unit tests for the pre-trade feasibility preflight."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import polars as pl
import pytest

import src.validation.feasibility as feasibility_module
from src.data.calendar import load_calendar
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload, UntrustedDatasetError
from src.policy.currency import CurrencyConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import PolicyId
from src.validation.experiment import (
    CandidateSpec,
    CurrencySpec,
    ExperimentSpec,
    MappingSpec,
)
from src.validation.feasibility import (
    FeasibilityError,
    assert_experiment_feasible,
    currency_warmup_sessions,
    overlay_warmup_sessions,
    require_feasibility,
    reserve_warmup_sessions,
    resolve_feasibility,
)

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2022, 6, 1, 5, 0, tzinfo=UTC)
_CPI_VISIBLE_PERIOD_END: Final[date] = date(2023, 11, 1)
_CPI_INVISIBLE_PERIOD_END: Final[date] = date(2023, 12, 1)
_SHORT_WINDOW: Final[tuple[date, date]] = (date(2023, 12, 20), date(2024, 1, 31))
_SHORT_PANEL: Final[tuple[date, date]] = (date(2023, 12, 20), date(2024, 2, 1))
_THIN_WINDOW: Final[tuple[date, date]] = (date(2024, 1, 15), date(2024, 1, 31))
_THIN_PANEL: Final[tuple[date, date]] = (date(2024, 1, 15), date(2024, 2, 1))
_MAPPING_PANEL: Final[tuple[date, date]] = (date(2023, 5, 1), date(2024, 2, 1))


def _panel_days(start: date, end: date) -> tuple[date, ...]:
    return _CALENDAR.sessions(start, end)


def _payload() -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="synthetic",
        request_params={},
        retrieved_at=_RETRIEVED_AT,
        extension="json",
        content=b"{}",
    )


def _prices_frame(days: tuple[date, ...], tickers: tuple[str, ...], close_of_day: Callable[[date], float]) -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    rows_ticker: list[str] = []
    rows_date: list[date] = []
    closes: list[float] = []
    for ticker in tickers:
        for day in days:
            rows_ticker.append(ticker)
            rows_date.append(day)
            closes.append(close_of_day(day))
    n = len(rows_date)
    return pl.DataFrame(
        {
            "ticker": rows_ticker,
            "date": rows_date,
            "open": [close * 0.98 for close in closes],
            "high": [close * 1.02 for close in closes],
            "low": [close * 0.97 for close in closes],
            "close": closes,
            "volume": [10_000] * n,
            "adjusted_close": closes,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )


def _fx_frame(days: tuple[date, ...]) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    ordered = list(days)
    return pl.DataFrame(
        {
            "date": ordered,
            "usdkrw": [1300.0] * len(ordered),
            "source": ["synthetic"] * len(ordered),
            "retrieved_at": [_RETRIEVED_AT] * len(ordered),
        },
        schema=dict(spec.columns),
    )


def _cpi_frame(period_end: date, value: float) -> pl.DataFrame:
    spec = spec_for(Dataset.CPI)
    return pl.DataFrame(
        {
            "period_end": [period_end],
            "value": [value],
            "source": ["synthetic"],
            "retrieved_at": [_RETRIEVED_AT],
        },
        schema=dict(spec.columns),
    )


def _catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    days: tuple[date, ...],
    tickers: tuple[str, ...],
    close_of_day: Callable[[date], float],
    cpi_period_end: date,
    name: str = "catalog",
) -> DataSettings:
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    persist_ingest(_prices_frame(days, tickers, close_of_day), Dataset.PRICES, _payload(), settings)
    persist_ingest(_fx_frame(days), Dataset.FX, _payload(), settings)
    persist_ingest(_cpi_frame(cpi_period_end, 100.0), Dataset.CPI, _payload(), settings)
    return settings


def _constant_closes(day: date) -> float:
    return 100.0


@pytest.mark.parametrize("scenario_id", ["FEAS-A01-cpi-invisible"])
def test_feas_a01_cpi_invisible(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAS-A01-cpi-invisible"""
    days = _panel_days(*_SHORT_PANEL)
    settings = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VT",),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_INVISIBLE_PERIOD_END,
    )
    kwargs = {
        "start": _SHORT_WINDOW[0],
        "end": _SHORT_WINDOW[1],
        "fill_delay_sessions": 1,
        "mark_policies": (PolicyId.VT,),
        "overlay": None,
        "overlay_policies": (),
        "settings": settings,
    }

    report = resolve_feasibility(**kwargs)  # type: ignore[arg-type]

    assert {violation.code for violation in report.violations} == {"cpi"}
    assert report.requested_start == _SHORT_WINDOW[0]
    assert report.requested_end == _SHORT_WINDOW[1]
    # Informational only: the first execution lacks CPI but the 2024-01-31 signal does not.
    assert report.earliest_safe_start == date(2024, 1, 31)
    assert report.ingest_recommended_start == date(2024, 1, 2)
    with pytest.raises(FeasibilityError, match="cpi"):
        require_feasibility(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("scenario_id", ["FEAS-A02-overlay-short-history"])
def test_feas_a02_overlay_short_history(scenario_id: str, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAS-A02-overlay-short-history"""
    days = _panel_days(date(2023, 12, 11), date(2024, 2, 1))
    settings = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VTI",),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
    )
    kwargs = {
        "start": _THIN_WINDOW[0],
        "end": _THIN_WINDOW[1],
        "fill_delay_sessions": 1,
        "mark_policies": (PolicyId.VTI,),
        "overlay": OverlayConfig(),
        "overlay_policies": (PolicyId.VTI,),
        "settings": settings,
    }

    report = resolve_feasibility(**kwargs)  # type: ignore[arg-type]

    assert {violation.code for violation in report.violations} == {"overlay_warmup"}
    assert report.warmup_sessions == 252
    warmup_violation = next(v for v in report.violations if v.code == "overlay_warmup")
    match = re.search(r"found (\d+) usable", warmup_violation.message)
    assert match is not None
    assert int(match.group(1)) < 252
    with pytest.raises(FeasibilityError):
        require_feasibility(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("scenario_id", ["FEAS-A03-pass-no-clamp"])
def test_feas_a03_pass_no_clamp(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAS-A03-pass-no-clamp"""
    days = _panel_days(*_SHORT_PANEL)
    settings = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VT", "VTI"),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
    )
    spec = ExperimentSpec.model_validate(
        {
            "name": "feas_a03",
            "start": _SHORT_WINDOW[0].isoformat(),
            "end": _SHORT_WINDOW[1].isoformat(),
            "contribution_krw": 1_000_000,
            "delta0": 0.02,
            "horizon_months": 0,
            "baseline": {"id": "m0_global", "policy": "s0_global", "modules": 0},
            "candidates": [{"id": "s1_us", "policy": "s1_us", "modules": 1}],
        }
    )

    report = assert_experiment_feasible(spec, settings)

    assert report.violations == ()
    assert report.requested_start == spec.start == _SHORT_WINDOW[0]
    assert report.requested_end == spec.end == _SHORT_WINDOW[1]
    assert report.earliest_safe_start == date(2023, 12, 29)


@pytest.mark.parametrize("scenario_id", ["FEAS-A04-overlay-pass-warmup"])
def test_feas_a04_overlay_pass_warmup(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAS-A04-overlay-pass-warmup"""
    days = _panel_days(date(2022, 12, 1), date(2024, 2, 1))
    rising = {day: 100.0 * 1.001**index for index, day in enumerate(days)}
    settings = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VTI",),
        close_of_day=rising.__getitem__,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
    )

    report = resolve_feasibility(
        start=_THIN_WINDOW[0],
        end=_THIN_WINDOW[1],
        fill_delay_sessions=1,
        mark_policies=(PolicyId.VTI,),
        overlay=OverlayConfig(max_shift=0.10),
        overlay_policies=(PolicyId.VTI,),
        settings=settings,
    )

    assert report.violations == ()
    assert report.warmup_sessions == 252
    assert report.earliest_safe_start is not None


@pytest.mark.parametrize("scenario_id", ["FEAS-A05-warmup-sessions-helper"])
def test_feas_a05_warmup_sessions_helper(scenario_id: str) -> None:
    """FEAS-A05-warmup-sessions-helper"""
    assert overlay_warmup_sessions(None) == 0
    custom = OverlayConfig(trend_window=10, vol_window=30, drawdown_window=20)
    assert overlay_warmup_sessions(custom) == 30


def _metadata_row(ticker: str, sleeve: str = "VTI") -> dict[str, object]:
    return {
        "ticker": ticker,
        "effective_date": date(2022, 6, 1),
        "filing_date": _RETRIEVED_AT,
        "sleeve": sleeve,
        "expense_ratio": 0.03,
        "aum_usd": 5e10,
        "avg_dollar_volume": 5e8,
        "is_leveraged": 0,
        "is_inverse": 0,
        "inception_date": date(2000, 5, 1),
        "source": "synthetic",
        "retrieved_at": _RETRIEVED_AT,
    }


def _persist_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tickers: tuple[str, ...], name: str = "catalog"
) -> None:
    """Extend the shared catalog with visible ETF_METADATA rows for ``tickers``."""
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    monkeypatch.chdir(root)
    rows = [_metadata_row(ticker) for ticker in tickers]
    frame = pl.DataFrame(rows, schema=dict(spec_for(Dataset.ETF_METADATA).columns))
    persist_ingest(frame, Dataset.ETF_METADATA, _payload(), DataSettings(data_root="data"))


def _mapping_spec() -> ExperimentSpec:
    """S1 versus S1+implementation-mapping on the shared short window."""
    return ExperimentSpec(
        name="feas_j_mapping",
        start=_SHORT_WINDOW[0],
        end=_SHORT_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        mapping=MappingSpec(min_improvement=0.02),
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_mapping", policy="s1_us", modules=1)],
    )


@pytest.mark.parametrize("scenario_id", ["FEAS-J-mapping-metadata"])
def test_feas_j_mapping_metadata(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAS-J-mapping-metadata"""
    days = _panel_days(*_MAPPING_PANEL)
    spec = _mapping_spec()

    full_prices = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VTI", "ITOT"),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
        name="pass",
    )
    _persist_metadata(monkeypatch, tmp_path, ("VTI", "ITOT"), name="pass")

    report = assert_experiment_feasible(spec, full_prices)

    assert report.violations == ()
    assert report.requested_start == spec.start

    no_metadata_prices = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VTI", "ITOT"),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
        name="no_metadata",
    )
    with pytest.raises(FeasibilityError) as excinfo:
        assert_experiment_feasible(spec, no_metadata_prices)
    assert excinfo.value.report is not None
    assert {violation.code for violation in excinfo.value.report.violations} == {"etf_metadata"}

    missing_itot = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VTI",),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
        name="missing_itot",
    )
    _persist_metadata(monkeypatch, tmp_path, ("VTI", "ITOT"), name="missing_itot")
    with pytest.raises(FeasibilityError) as excinfo:
        assert_experiment_feasible(spec, missing_itot)
    assert excinfo.value.report is not None
    assert any("ITOT" in violation.message for violation in excinfo.value.report.violations)


def _currency_spec() -> ExperimentSpec:
    """S1 versus S1+currency on the shared short window."""
    return ExperimentSpec(
        name="feas_k_currency",
        start=_SHORT_WINDOW[0],
        end=_SHORT_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        currency=CurrencySpec(max_defer=0.10),
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_currency", policy="s1_us", modules=1)],
    )


@pytest.mark.parametrize("scenario_id", ["FEAS-K-currency-warmup"])
def test_feas_k_currency_warmup(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAS-K-currency-warmup"""
    assert currency_warmup_sessions(None) == 0
    assert currency_warmup_sessions(CurrencyConfig(max_defer=0.10)) == 252

    days = _panel_days(*_SHORT_PANEL)
    settings = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VTI",),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
    )

    with pytest.raises(FeasibilityError) as excinfo:
        assert_experiment_feasible(_currency_spec(), settings)

    assert excinfo.value.report is not None
    codes = {violation.code for violation in excinfo.value.report.violations}
    assert "currency_warmup" in codes


@pytest.mark.parametrize("scenario_id", ["FEAS-H-reserve-warmup"])
def test_feas_h_reserve_warmup(scenario_id: str, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAS-H-reserve-warmup"""
    assert reserve_warmup_sessions(None) == 0
    assert reserve_warmup_sessions(ReserveConfig(max_withhold=0.10)) == 252

    days = _panel_days(date(2023, 12, 11), date(2024, 2, 1))
    settings = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("QQQ",),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
    )
    kwargs = {
        "start": _THIN_WINDOW[0],
        "end": _THIN_WINDOW[1],
        "fill_delay_sessions": 1,
        "mark_policies": (PolicyId.QQQ,),
        "overlay": None,
        "overlay_policies": (),
        "settings": settings,
        "reserve": ReserveConfig(max_withhold=0.10),
        "reserve_policies": (PolicyId.QQQ,),
    }

    report = resolve_feasibility(**kwargs)  # type: ignore[arg-type]

    assert {violation.code for violation in report.violations} == {"reserve_warmup"}
    assert report.warmup_sessions == 252
    warmup_violation = next(v for v in report.violations if v.code == "reserve_warmup")
    match = re.search(r"found (\d+) usable", warmup_violation.message)
    assert match is not None
    assert int(match.group(1)) < 252
    with pytest.raises(FeasibilityError):
        require_feasibility(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("scenario_id", ["FEAS-ACG-macro-trust"])
def test_feas_acg_macro_trust(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAS-ACG-macro-trust"""
    days = _panel_days(*_SHORT_PANEL)
    settings = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("VT", "VTI"),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
    )
    macro_spec = spec_for(Dataset.MACRO)
    persist_ingest(
        pl.DataFrame(
            {
                "series_id": ["BAA10Y"],
                "observation_date": [date(2023, 12, 20)],
                "release_date": [datetime(2023, 12, 20, 21, 0, tzinfo=UTC)],
                "value": [3.5],
            },
            schema=dict(macro_spec.columns),
        ),
        Dataset.MACRO,
        _payload(),
        settings,
    )
    spec = ExperimentSpec.model_validate(
        {
            "name": "feas_acg",
            "start": _SHORT_WINDOW[0].isoformat(),
            "end": _SHORT_WINDOW[1].isoformat(),
            "contribution_krw": 1_000_000,
            "delta0": 0.02,
            "horizon_months": 0,
            "objective": "adaptive_growth",
            "adaptive_contribution": {},
            "baseline": {"id": "m0_global", "policy": "s0_global", "modules": 0},
            "candidates": [{"id": "s1_us", "policy": "s1_us", "modules": 1}],
        }
    )

    requested: list[Dataset] = []
    real_latest_artifact = feasibility_module.latest_artifact

    def spy(settings: DataSettings, dataset: Dataset) -> object:
        requested.append(dataset)
        return real_latest_artifact(settings, dataset)

    monkeypatch.setattr(feasibility_module, "latest_artifact", spy)

    report = assert_experiment_feasible(spec, settings)

    # The trusted MACRO artifact is requested before any allocation can run.
    assert Dataset.MACRO in requested
    assert report.violations == ()
    assert report.requested_start == spec.start

    def boom(_settings: DataSettings, _dataset: Dataset) -> object:
        raise UntrustedDatasetError("untrusted MACRO partition")

    monkeypatch.setattr(feasibility_module, "latest_artifact", boom)
    with pytest.raises(UntrustedDatasetError, match="MACRO"):
        assert_experiment_feasible(spec, settings)


def _mix_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="feas_mix",
        start=_SHORT_WINDOW[0],
        end=_SHORT_WINDOW[1],
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0),
        candidates=[
            CandidateSpec(
                id="qqq_grid_mix",
                policy=PolicyId.QQQ,
                modules=1,
                targets={"QQQ": 0.9, "GRID": 0.1},
            )
        ],
    )


@pytest.mark.parametrize("scenario_id", ["FEA-MIX-extra-tickers"])
def test_fea_mix_extra_tickers(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEA-MIX-extra-tickers"""
    days = _panel_days(*_SHORT_PANEL)
    qqq_only = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("QQQ",),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
        name="qqq_only",
    )
    with pytest.raises(FeasibilityError, match="price"):
        assert_experiment_feasible(_mix_spec(), qqq_only)

    both = _catalog(
        monkeypatch,
        tmp_path,
        days=days,
        tickers=("QQQ", "GRID"),
        close_of_day=_constant_closes,
        cpi_period_end=_CPI_VISIBLE_PERIOD_END,
        name="qqq_grid",
    )
    report = assert_experiment_feasible(_mix_spec(), both)
    assert report.violations == ()
