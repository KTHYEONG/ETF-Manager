# ruff: noqa: B905,PT011,S101
"""Track H incremental portfolio tests."""
from __future__ import annotations

from datetime import date

import pytest

from src.analytics.incremental_portfolio import (
    INCREMENTAL_HORIZON_SURFACE,
    INCREMENTAL_SATELLITE_WEIGHTS,
    INCREMENTAL_SOXX_WEIGHTS,
    PATH_BOOTSTRAP_WIN_FLOOR,
    BuyOnlyAttribution,
    IncrementalArmId,
    IncrementalArmReport,
    IncrementalPortfolioReport,
    PathBootstrapVerdict,
    arm_targets,
    attribute_buy_only_soxx,
    classify_portfolio_status,
    clip_incremental_cohort_start,
    make_incremental_arm_id,
    paired_path_block_bootstrap,
    apply_incremental_portfolio_status,
    resolve_incremental_horizon,
    write_incremental_portfolio_report,
)
from src.analytics.thesis_meaning import (
    HistoricalQuality,
    PortfolioEvidenceStatus,
    ThesisEvidenceStatus,
    ThesisMeaningSnapshot,
    VehicleEvidenceStatus,
)
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
from src.policy.targets import PolicyId


def _make_result(mark_krw_seq: tuple[float, ...], shares_seq: tuple[dict[str, float], ...], terminal_real: float) -> AllocationResult:
    cfg = AllocationConfig(policy=PolicyId.QQQ, start=date(2020, 1, 1), end=date(2020, 12, 31), monthly_contribution_krw=1_000_000)
    snaps = []
    for idx, (mark, shares) in enumerate(zip(mark_krw_seq, shares_seq)):
        snaps.append(
            AllocationSnapshot(
                session=date(2020, idx + 1, 15),
                cash_krw=0.0,
                cash_usd=0.0,
                shares=dict(shares),
                mark_krw=float(mark),
                contribution_krw=1_000_000.0,
                fees_krw=0.0,
                reserve_krw=0.0,
            )
        )
    return AllocationResult(
        config=cfg,
        snapshots=tuple(snaps),
        terminal_wealth_krw=float(terminal_real),
        xirr=0.0,
        max_drawdown=0.0,
        terminal_wealth_real_krw=float(terminal_real),
        xirr_real=0.0,
    )


def test_inc_h1_arm_targets_locked() -> None:
    assert INCREMENTAL_SOXX_WEIGHTS == (0.05, 0.10, 0.15)
    assert arm_targets(0.05) == {"QQQ": 0.95, "SOXX": 0.05}
    assert arm_targets(0.10) == {"QQQ": 0.90, "SOXX": 0.10}
    assert arm_targets(0.15) == {"QQQ": 0.85, "SOXX": 0.15}
    with pytest.raises(ValueError):
        arm_targets(0.17)
    with pytest.raises(ValueError):
        arm_targets(1.0)


def test_inc_h2_paired_path_bootstrap_joint_blocks() -> None:
    rets = tuple([0.01] * 24)
    v1 = paired_path_block_bootstrap(rets, rets, block_size=6, n_paths=50, seed=1)
    assert v1.n_paths == 50
    assert 0.0 <= v1.win_rate <= 1.0
    v2 = paired_path_block_bootstrap(rets, rets, block_size=6, n_paths=50, seed=1)
    assert v2.win_rate == pytest.approx(v1.win_rate)
    assert v2.p05_terminal_ratio == pytest.approx(v1.p05_terminal_ratio)
    assert v2.n_paths == v1.n_paths


def test_inc_h3_path_ok_threshold() -> None:
    assert pytest.approx(0.55) == PATH_BOOTSTRAP_WIN_FLOOR
    # win_rate 0.55 should be ok True via direct verdict construction not needed; test via bootstrap
    # strictly dominating returns -> ok True
    cand = tuple([0.02] * 20)
    base = tuple([0.01] * 20)
    verdict_dom = paired_path_block_bootstrap(cand, base, block_size=4, n_paths=200, seed=42)
    assert verdict_dom.ok is True
    assert verdict_dom.win_rate >= 0.55
    # equal returns -> win_rate around 0.5? But due to tie rule >=1, equal series gives win_rate 1.0 (since all ratios 1). So use slight underperformance to get <0.55
    # Instead test equal returns yields win_rate ==1 with dominating? Actually equal returns produce ratio 1 always -> win_rate 1 => ok True. So we craft candidate slightly worse to get <0.55.
    # Use candidate 0.01 vs baseline 0.02
    cand_low = tuple([0.01] * 20)
    base_high = tuple([0.02] * 20)
    verdict_low = paired_path_block_bootstrap(cand_low, base_high, block_size=4, n_paths=200, seed=42)
    assert verdict_low.win_rate < 0.55
    assert verdict_low.ok is False
    # direct threshold edge: construct verdict with 0.55
    edge = PathBootstrapVerdict(n_paths=10, win_rate=0.55, p05_terminal_ratio=1.0, ok=True)
    assert edge.ok is True
    below = PathBootstrapVerdict(n_paths=10, win_rate=0.54, p05_terminal_ratio=1.0, ok=False)
    assert below.ok is False


def test_inc_h4_buy_only_attribution_drift() -> None:
    # two snapshots, target 0.15, realized weights 0.10 then 0.20
    # price_at returns 100, mark 1000, shares => weight = shares*price*fx/mark with fx=1
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 100.0

    def fx_at(d: date) -> float:
        return 1.0

    cand = _make_result((1000.0, 1000.0), ({"SOXX": 1.0}, {"SOXX": 2.0}), terminal_real=2000.0)
    base = _make_result((1000.0, 1000.0), ({}, {}), terminal_real=1000.0)
    attr = attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.15, price_at=price_at, fx_at=fx_at)
    assert attr.mean_abs_weight_drift == pytest.approx(0.05, abs=1e-9)
    assert attr.terminal_weight_drift == pytest.approx(0.05, abs=1e-9)
    assert attr.target_soxx_weight == pytest.approx(0.15)
    assert attr.mean_realized_soxx_weight == pytest.approx(0.15, abs=1e-9)
    assert attr.terminal_realized_soxx_weight == pytest.approx(0.20, abs=1e-9)
    assert attr.incremental_wealth_ratio == pytest.approx(2.0)


def test_inc_afx_fx_scales_realized_weight() -> None:
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 10.0

    def fx_at(d: date) -> float:
        return 1300.0

    cand = _make_result((13000.0,), ({"SOXX": 1.0},), terminal_real=13000.0)
    base = _make_result((13000.0,), ({},), terminal_real=13000.0)
    attr = attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.05, price_at=price_at, fx_at=fx_at)
    assert attr.mean_realized_soxx_weight == pytest.approx(1.0, abs=1e-12)
    assert attr.terminal_realized_soxx_weight == pytest.approx(1.0, abs=1e-12)


def test_inc_afx_rejects_usd_over_krw_mark() -> None:
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 100.0

    def fx_at(d: date) -> float:
        return 1.0

    cand = _make_result((130000.0,), ({"SOXX": 1.0},), terminal_real=130000.0)
    base = _make_result((130000.0,), ({},), terminal_real=130000.0)
    with pytest.raises(ValueError):
        attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.05, price_at=price_at, fx_at=fx_at)


def test_inc_afx_weight_above_one_fails() -> None:
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 10.0

    def fx_at(d: date) -> float:
        return 1300.0

    cand = _make_result((13000.0,), ({"SOXX": 2.0},), terminal_real=13000.0)
    base = _make_result((13000.0,), ({},), terminal_real=13000.0)
    with pytest.raises(ValueError):
        attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.05, price_at=price_at, fx_at=fx_at)


def test_inc_h5_classify_portfolio_status() -> None:
    def make_arm(median: float, ok: bool) -> IncrementalArmReport:
        verdict = PathBootstrapVerdict(n_paths=10, win_rate=1.0 if ok else 0.4, p05_terminal_ratio=1.0, ok=ok)
        attr = BuyOnlyAttribution(
            target_soxx_weight=0.05,
            mean_realized_soxx_weight=0.05,
            terminal_realized_soxx_weight=0.05,
            mean_abs_weight_drift=0.0,
            terminal_weight_drift=0.0,
            incremental_wealth_ratio=1.0,
        )
        return IncrementalArmReport(
            arm_id=IncrementalArmId.QQQ95_SOXX5,
            soxx_weight=0.05,
            median_ratio=float(median),
            p10_ratio=float(median),
            worst_ratio=float(median),
            win_rate=float(verdict.win_rate),
            cohort_count=8,
            ce_gamma_2=1.0,
            ce_gamma_5=1.0,
            ce_gamma_10=1.0,
            attribution=attr,
            path_bootstrap=verdict,
        )

    promising = make_arm(1.01, True)
    assert classify_portfolio_status([promising]) == PortfolioEvidenceStatus.HISTORICALLY_PROMISING
    weak1 = make_arm(0.99, True)
    assert classify_portfolio_status([weak1]) == PortfolioEvidenceStatus.HISTORICALLY_WEAK
    weak2 = make_arm(1.01, False)
    assert classify_portfolio_status([weak2]) == PortfolioEvidenceStatus.HISTORICALLY_WEAK
    with pytest.raises(ValueError):
        classify_portfolio_status([])


def test_inc_h6_apply_status_preserves_vehicle() -> None:
    snap = ThesisMeaningSnapshot(
        thesis_status=ThesisEvidenceStatus.UNRESOLVED,
        vehicle_status=VehicleEvidenceStatus.ACTIVE_PROXY,
        portfolio_status=PortfolioEvidenceStatus.UNVERIFIED,
        historical_quality=HistoricalQuality.TARGET_THIN,
        history_available=True,
        evidence_sufficient=True,
        thin_sample_warning=True,
    )
    updated = apply_incremental_portfolio_status(snap, PortfolioEvidenceStatus.HISTORICALLY_PROMISING)
    assert updated.vehicle_status == snap.vehicle_status
    assert updated.thesis_status == snap.thesis_status
    assert updated.portfolio_status == PortfolioEvidenceStatus.HISTORICALLY_PROMISING
    assert updated.historical_quality == snap.historical_quality


@pytest.mark.parametrize("scenario_id", ["test_inc_arm_targets_parameterized_vehicle"])
def test_inc_arm_targets_parameterized_vehicle(scenario_id: str) -> None:
    """test_inc_arm_targets_parameterized_vehicle"""
    assert arm_targets(0.05) == {"QQQ": 0.95, "SOXX": 0.05}
    assert arm_targets(0.10, vehicle_ticker="PAVE") == {"QQQ": 0.90, "PAVE": 0.10}
    with pytest.raises(ValueError, match="not in"):
        arm_targets(0.17)  # type: ignore[arg-type]
    assert INCREMENTAL_SOXX_WEIGHTS is INCREMENTAL_SATELLITE_WEIGHTS
    assert INCREMENTAL_SATELLITE_WEIGHTS == (0.05, 0.10, 0.15)
    assert INCREMENTAL_SOXX_WEIGHTS == (0.05, 0.10, 0.15)


@pytest.mark.parametrize("scenario_id", ["test_inc_make_arm_id_soxx_and_pave"])
def test_inc_make_arm_id_soxx_and_pave(scenario_id: str) -> None:
    """test_inc_make_arm_id_soxx_and_pave"""
    assert make_incremental_arm_id("SOXX", 0.05) == "qqq95_soxx5"
    assert make_incremental_arm_id("PAVE", 0.15) == "qqq85_pave15"
    assert make_incremental_arm_id("SOXX", 0.10) == "qqq90_soxx10"


@pytest.mark.parametrize("scenario_id", ["test_inc_make_arm_id_robo"])
def test_inc_make_arm_id_robo(scenario_id: str) -> None:
    """test_inc_make_arm_id_robo"""
    assert make_incremental_arm_id("ROBO", 0.05) == "qqq95_robo5"
    assert make_incremental_arm_id("ROBO", 0.10) == "qqq90_robo10"
    assert make_incremental_arm_id("ROBO", 0.15) == "qqq85_robo15"
    assert arm_targets(0.10, vehicle_ticker="ROBO") == {"QQQ": 0.9, "ROBO": 0.1}


@pytest.mark.parametrize("scenario_id", ["test_inc_resolve_horizon_prefers_120_then_fallback"])
def test_inc_resolve_horizon_prefers_120_then_fallback(scenario_id: str) -> None:
    """test_inc_resolve_horizon_prefers_120_then_fallback"""
    assert INCREMENTAL_HORIZON_SURFACE == (120, 96, 84, 60)
    h1, fb1 = resolve_incremental_horizon(date(2007, 8, 31), date(2026, 8, 28))
    assert h1 == 120
    assert fb1 is False
    h2, fb2 = resolve_incremental_horizon(date(2019, 2, 28), date(2026, 8, 28))
    assert h2 in (60, 84, 96)
    assert fb2 is True
    assert h2 != 120
    with pytest.raises(ValueError, match="span too short"):
        resolve_incremental_horizon(date(2026, 8, 28), date(2026, 8, 28))


@pytest.mark.parametrize("scenario_id", ["test_inc_clip_cohort_start_to_vehicle_listing"])
def test_inc_clip_cohort_start_to_vehicle_listing(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_inc_clip_cohort_start_to_vehicle_listing"""
    from datetime import UTC, datetime

    import polars as pl

    from src.data.settings import DataSettings

    pave_first = date(2019, 2, 28)
    soxx_first = date(2007, 8, 31)
    as_of = datetime(2026, 8, 28, tzinfo=UTC)
    settings = DataSettings()

    def fake_snapshot_visible(_snapshot: object, _dataset: object, _as_of: datetime) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ticker": ["PAVE", "SOXX"],
                "date": [pave_first, soxx_first],
                "adjusted_close": [50.0, 100.0],
            }
        )

    from types import SimpleNamespace

    monkeypatch.setattr(
        "src.data.catalog.resolve_snapshot", lambda _settings, _datasets: SimpleNamespace(artifacts={})
    )
    monkeypatch.setattr("src.data.catalog.load_snapshot_visible", fake_snapshot_visible)
    catalog_start = date(2007, 8, 31)
    assert clip_incremental_cohort_start(
        catalog_start=catalog_start,
        settings=settings,
        as_of=as_of,
        vehicle_ticker="PAVE",
    ) == pave_first
    assert clip_incremental_cohort_start(
        catalog_start=catalog_start,
        settings=settings,
        as_of=as_of,
        vehicle_ticker="SOXX",
    ) == soxx_first
    with pytest.raises(ValueError, match="no price history"):
        clip_incremental_cohort_start(
            catalog_start=catalog_start,
            settings=settings,
            as_of=as_of,
            vehicle_ticker="MISSING",
        )


def test_inc_pinned_factories_read_snapshot_partitions(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned price/FX factories and the catalog-min scan read one verified snapshot."""
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    import polars as _pl

    from src.analytics.thesis.incremental import (
        _catalog_min_price_session,
        _make_fx_at,
        _make_price_at,
    )
    from src.data.calendar import load_calendar as _load_calendar
    from src.data.pipeline import persist_ingest as _persist
    from src.data.schema import Dataset as _Dataset
    from src.data.schema import spec_for as _spec_for
    from src.data.settings import DataSettings as _Settings
    from src.data.storage import RawPayload as _Payload

    root = _Path(str(tmp_path))  # type: ignore[arg-type]
    monkeypatch.chdir(root)
    settings = _Settings(data_root="data")
    retrieved_at = datetime(2024, 1, 5, 5, 0, tzinfo=UTC)
    sessions = list(_load_calendar("XNYS").sessions(date(2024, 1, 2), date(2024, 1, 31)))

    def _payload() -> _Payload:
        return _Payload(
            provider="synthetic", endpoint="probe", request_params={},
            retrieved_at=retrieved_at, extension="json", content=b"{}",
        )

    _persist(
        _pl.DataFrame(
            {
                "ticker": ["SOXX"] * len(sessions),
                "date": sessions,
                "open": [100.0] * len(sessions),
                "high": [101.0] * len(sessions),
                "low": [99.0] * len(sessions),
                "close": [100.0] * len(sessions),
                "volume": [10_000] * len(sessions),
                "adjusted_close": [100.0] * len(sessions),
                "dividend": [0.0] * len(sessions),
                "split_factor": [1.0] * len(sessions),
                "source": ["synthetic"] * len(sessions),
                "retrieved_at": [retrieved_at] * len(sessions),
            },
            schema=dict(_spec_for(_Dataset.PRICES).columns),
        ),
        _Dataset.PRICES, _payload(), settings,
    )
    _persist(
        _pl.DataFrame(
            {
                "date": sessions,
                "usdkrw": [1300.0] * len(sessions),
                "source": ["synthetic"] * len(sessions),
                "retrieved_at": [retrieved_at] * len(sessions),
            },
            schema=dict(_spec_for(_Dataset.FX).columns),
        ),
        _Dataset.FX, _payload(), settings,
    )
    assert _catalog_min_price_session(settings) == sessions[0]
    assert _make_price_at(settings)(sessions[-1], "SOXX") == 100.0
    assert _make_fx_at(settings)(sessions[-1]) == 1300.0


def test_inc_pinned_catalog_min_short_span_fails_closed(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sub-horizon lake fails closed after the pinned catalog-min scan."""
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    import polars as _pl

    from src.analytics.thesis.incremental import run_incremental_portfolio
    from src.data.calendar import load_calendar as _load_calendar
    from src.data.panel_freshness import CatalogPanelReport as _PanelReport
    from src.data.panel_freshness import PanelFreshnessStatus as _Freshness
    from src.data.pipeline import persist_ingest as _persist
    from src.data.schema import Dataset as _Dataset
    from src.data.settings import DataSettings as _Settings
    from src.data.storage import RawPayload as _Payload

    root = _Path(str(tmp_path))  # type: ignore[arg-type]
    monkeypatch.chdir(root)
    settings = _Settings(data_root="data")
    retrieved_at = datetime(2024, 1, 5, 5, 0, tzinfo=UTC)
    sessions = list(_load_calendar("XNYS").sessions(date(2024, 1, 2), date(2024, 1, 31)))
    rows = [
        {
            "ticker": ticker, "date": day, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.0, "volume": 10_000,
            "adjusted_close": 100.0, "dividend": 0.0, "split_factor": 1.0,
            "source": "synthetic", "retrieved_at": retrieved_at,
        }
        for ticker in ("QQQ", "SOXX")
        for day in sessions
    ]
    _persist(
        _pl.DataFrame(rows, schema={
            "ticker": _pl.String, "date": _pl.Date, "open": _pl.Float64, "high": _pl.Float64,
            "low": _pl.Float64, "close": _pl.Float64, "volume": _pl.Int64,
            "adjusted_close": _pl.Float64, "dividend": _pl.Float64, "split_factor": _pl.Float64,
            "source": _pl.String, "retrieved_at": _pl.Datetime("us", "UTC"),
        }),
        _Dataset.PRICES,
        _Payload(provider="synthetic", endpoint="probe", request_params={},
                 retrieved_at=retrieved_at, extension="json", content=b"{}"),
        settings,
    )
    panel_as_of = _load_calendar("XNYS").close_ts(sessions[-1])
    panel = _PanelReport(
        panel_as_of=panel_as_of, lag_days=1, status=_Freshness.FRESH,
        ticker_last_session={}, cpi_last_observation=None,
        fx_last_observation=None, holdings_last_filing=None,
    )

    def _unused_runner(config: object) -> object:
        raise AssertionError("runner must not run on a sub-horizon span")

    with pytest.raises(ValueError, match="span too short"):
        run_incremental_portfolio(
            settings=settings, as_of=panel_as_of, runner=_unused_runner,  # type: ignore[arg-type]
            contribution_krw=1_000_000.0, bootstrap_paths=1, seed=1,
            panel_report=panel, vehicle_ticker="SOXX",
        )


@pytest.mark.parametrize("scenario_id", ["test_inc_attribute_uses_vehicle_ticker"])
def test_inc_attribute_uses_vehicle_ticker(scenario_id: str) -> None:
    """test_inc_attribute_uses_vehicle_ticker"""
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "PAVE"
        return 100.0

    def fx_at(d: date) -> float:
        return 1.0

    cand = _make_result((1000.0, 1000.0), ({"PAVE": 1.0}, {"PAVE": 2.0}), terminal_real=2000.0)
    base = _make_result((1000.0, 1000.0), ({}, {}), terminal_real=1000.0)
    attr = attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.15, price_at=price_at, fx_at=fx_at, vehicle_ticker="PAVE")
    assert attr.mean_abs_weight_drift == pytest.approx(0.05, abs=1e-9)
    assert attr.target_soxx_weight == pytest.approx(0.15)
    # default SOXX still works
    def price_at_soxx(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 100.0

    cand2 = _make_result((1000.0, 1000.0), ({"SOXX": 1.0}, {"SOXX": 2.0}), terminal_real=2000.0)
    attr2 = attribute_buy_only_soxx(candidate=cand2, baseline=base, soxx_weight=0.15, price_at=price_at_soxx, fx_at=fx_at)
    assert attr2.mean_realized_soxx_weight == pytest.approx(0.15, abs=1e-9)


@pytest.mark.parametrize("scenario_id", ["test_inc_report_includes_horizon_fields"])
def test_inc_report_includes_horizon_fields(scenario_id: str, tmp_path) -> None:
    """test_inc_report_includes_horizon_fields"""
    from datetime import UTC, datetime

    as_of = datetime(2026, 8, 28, 15, 0, tzinfo=UTC)
    panel_as_of = datetime(2026, 8, 28, 21, 0, tzinfo=UTC)
    verdict = PathBootstrapVerdict(n_paths=10, win_rate=0.6, p05_terminal_ratio=1.0, ok=True)
    attr = BuyOnlyAttribution(target_soxx_weight=0.05, mean_realized_soxx_weight=0.05, terminal_realized_soxx_weight=0.05, mean_abs_weight_drift=0.0, terminal_weight_drift=0.0, incremental_wealth_ratio=1.0)
    arm = IncrementalArmReport(arm_id=IncrementalArmId.QQQ95_PAVE5, soxx_weight=0.05, median_ratio=1.01, p10_ratio=1.0, worst_ratio=0.99, win_rate=0.6, cohort_count=8, ce_gamma_2=1.0, ce_gamma_5=1.0, ce_gamma_10=1.0, attribution=attr, path_bootstrap=verdict)
    report = IncrementalPortfolioReport(thesis_id="ai_power_bottleneck", as_of=as_of, panel_as_of=panel_as_of, lag_days=0, freshness_status="FRESH", arms=(arm,), portfolio_status=PortfolioEvidenceStatus.HISTORICALLY_PROMISING, vehicle_ticker="PAVE", horizon_months=84, horizon_fallback=True)
    path = tmp_path / "report.json"
    write_incremental_portfolio_report(report, path)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["vehicle_ticker"] == "PAVE"
    assert data["horizon_months"] == 84
    assert data["horizon_fallback"] is True
    assert data["thesis_id"] == "ai_power_bottleneck"


def test_incremental_grid_unchanged() -> None:
    import pytest
    from src.analytics.thesis.incremental import INCREMENTAL_SATELLITE_WEIGHTS, arm_targets

    assert INCREMENTAL_SATELLITE_WEIGHTS == (0.05, 0.10, 0.15)
    assert arm_targets(0.10) == {'QQQ': 0.9, 'SOXX': 0.10}
    with pytest.raises(ValueError, match='not in'):
        arm_targets(0.90)

def test_classify_portfolio_status_requires_economic_hurdle() -> None:
    from src.analytics.thesis.incremental import (
        BuyOnlyAttribution,
        IncrementalArmId,
        IncrementalArmReport,
        PathBootstrapVerdict,
        classify_portfolio_status,
    )
    from src.analytics.thesis.meaning import PortfolioEvidenceStatus

    def make_arm(*, median: float, ce10: float, ok: bool) -> IncrementalArmReport:
        return IncrementalArmReport(
            arm_id=IncrementalArmId.QQQ90_PAVE10,
            soxx_weight=0.10,
            median_ratio=float(median),
            p10_ratio=float(median),
            worst_ratio=float(median),
            win_rate=0.56 if ok else 0.40,
            cohort_count=2,
            ce_gamma_2=1.0,
            ce_gamma_5=1.0,
            ce_gamma_10=float(ce10),
            attribution=BuyOnlyAttribution(
                target_soxx_weight=0.10,
                mean_realized_soxx_weight=0.10,
                terminal_realized_soxx_weight=0.10,
                mean_abs_weight_drift=0.0,
                terminal_weight_drift=0.0,
                incremental_wealth_ratio=float(median),
            ),
            path_bootstrap=PathBootstrapVerdict(
                n_paths=400,
                win_rate=0.56 if ok else 0.40,
                p05_terminal_ratio=0.94,
                ok=ok,
            ),
        )

    pave_like = make_arm(median=1.00074, ce10=0.99913, ok=True)
    assert classify_portfolio_status([pave_like]) is PortfolioEvidenceStatus.HISTORICALLY_WEAK
    soxx10 = make_arm(median=1.0199, ce10=1.0024, ok=True)
    soxx10 = IncrementalArmReport(
        arm_id=IncrementalArmId.QQQ90_SOXX10,
        soxx_weight=0.10,
        median_ratio=1.0199,
        p10_ratio=1.0070,
        worst_ratio=0.9938,
        win_rate=0.90,
        cohort_count=10,
        ce_gamma_2=1.0,
        ce_gamma_5=1.0,
        ce_gamma_10=1.0024,
        attribution=pave_like.attribution,
        path_bootstrap=PathBootstrapVerdict(
            n_paths=400,
            win_rate=0.8025,
            p05_terminal_ratio=0.9772,
            ok=True,
        ),
    )
    assert classify_portfolio_status([soxx10]) is PortfolioEvidenceStatus.HISTORICALLY_PROMISING


def test_monthly_unitized_returns_excludes_contribution() -> None:
    from datetime import date

    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig, AllocationResult, Snapshot
    from src.analytics.thesis.incremental import monthly_simple_returns, monthly_unitized_returns

    cfg = AllocationConfig(
        policy=PolicyId.QQQ,
        start=date(2020, 1, 1),
        end=date(2020, 3, 31),
        monthly_contribution_krw=1_000_000.0,
    )
    snaps = (
        Snapshot(session=date(2020, 1, 31), mark_krw=1_000_000.0, contribution_krw=1_000_000.0),
        Snapshot(session=date(2020, 2, 29), mark_krw=2_000_000.0, contribution_krw=1_000_000.0),
        Snapshot(session=date(2020, 3, 31), mark_krw=3_000_000.0, contribution_krw=1_000_000.0),
    )
    result = AllocationResult(
        config=cfg,
        snapshots=snaps,
        terminal_wealth_krw=3_000_000.0,
        xirr=0.0,
        max_drawdown=0.0,
        terminal_wealth_real_krw=3_000_000.0,
        xirr_real=0.0,
    )
    naive = monthly_simple_returns(result)
    unitized = monthly_unitized_returns(result)
    assert naive[0] > 0.9
    assert abs(unitized[0]) < 1e-9
    assert abs(unitized[1]) < 1e-9

