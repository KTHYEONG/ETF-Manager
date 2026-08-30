"""Tests for thesis evidence vector."""
from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.policy.thesis import ThesisId, ThesisSpec, Horizon
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot



def _fake_runner(policy_wealth: dict[PolicyId, float]):
    def runner(config: AllocationConfig) -> AllocationResult:
        # Use terminal_wealth based on targets_override or policy
        # For evidence we differentiate by targets_override
        if config.targets_override is not None:
            # Check proxy
            if "SOXX" in config.targets_override:
                wealth = policy_wealth.get(PolicyId.QQQ, 110.0) if "SOXX" in str(config.targets_override) else 100.0
                # differentiate candidate vs baseline via targets
                # Baseline QQQ 100, SOXX candidate 108
                if config.targets_override == {"QQQ": 1.0}:
                    wealth = 100.0
                elif config.targets_override == {"SOXX": 1.0}:
                    wealth = 108.0
                else:
                    wealth = 105.0
            else:
                wealth = 100.0
        else:
            wealth = policy_wealth.get(config.policy, 100.0)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    return runner


@pytest.mark.parametrize("scenario_id", ["EV-HIST-120m-summary"])
def test_ev_hist_120m_summary(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector

    # Need a thesis spec
    thesis = ThesisSpec(
        id=ThesisId.AI_COMPUTE,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_semiconductor"],
        historical_proxies=["SOXX"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    # Mock runner that yields distinct wealths for QQQ vs SOXX
    # run_accumulation_cohort_report will call runner multiple times for cohorts
    # Our runner returns baseline 100, candidate 110 to get ratio >1
    def runner(config: AllocationConfig) -> AllocationResult:
        # Distinguish by targets_override
        if config.targets_override is not None and "SOXX" in config.targets_override:
            wealth = 110.0
        elif config.targets_override is not None and "QQQ" in config.targets_override:
            wealth = 100.0
        else:
            # fallback by policy
            wealth = 110.0 if config.policy is PolicyId.QQQ and config.targets_override == {"SOXX": 1.0} else 100.0
            # Actually policy is QQQ for both, so use targets
            if config.targets_override == {"SOXX": 1.0}:  # noqa: SIM108
                wealth = 110.0
            else:
                wealth = 100.0
        # Need snapshots for recovery_months maybe empty
        return AllocationResult(
            config=config,
            snapshots=(
                AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=wealth, contribution_krw=0, fees_krw=0),
            ),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    # Need to ensure load_visible for holdings doesn't crash; patch it to raise to test insufficient_data fallback? For this test we want historical computed even if overlap fails.
    # Patch load_visible to raise for holdings, but our compute_evidence_vector should catch and set overlap to insufficient_data, not fail.

    original_load_visible = None
    try:
        from src.data.catalog import load_visible as lv

        original_load_visible = lv
    except Exception:  # noqa: S110
        pass

    def fake_load_visible(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            raise ValueError("no holdings")
        return pl.DataFrame()

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)
    # Ensure compute doesn't try to load holdings via catalog
    snapshot = compute_evidence_vector(thesis=thesis, settings=settings, as_of=as_of, runner=runner)
    assert snapshot.historical.status == "computed"
    assert math.isfinite(snapshot.historical.metrics["median_ratio"])
    assert snapshot.historical.metrics["median_ratio"] > 0
    # cohort_count should be >=1
    assert snapshot.historical.metrics["cohort_count"] >= 1


@pytest.mark.parametrize("scenario_id", ["EV-OVL-holdings"])
def test_ev_ovl_holdings(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector

    thesis = ThesisSpec(
        id=ThesisId.AI_COMPUTE,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_semiconductor"],
        historical_proxies=["SOXX"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2020, 1, 15, tzinfo=UTC)

    # Build synthetic holdings matching OVL-A: A {X:60,Y:40}, B {X:50,Z:50} overlap 50
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report_date = date(2019, 12, 31)
    rows = [
        {"etf_ticker": "SOXX", "report_date": report_date, "filing_date": filing, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 60.0, "value_usd": 60, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "SOXX", "report_date": report_date, "filing_date": filing, "holding_id": "Y", "issuer_name": "Y Inc", "cusip": "Y-cusip", "isin": None, "lei": None, "weight_pct": 40.0, "value_usd": 40, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "QQQ", "report_date": report_date, "filing_date": filing, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "QQQ", "report_date": report_date, "filing_date": filing, "holding_id": "Z", "issuer_name": "Z Inc", "cusip": "Z-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    from src.data.pit import stamp_availability

    stamped = stamp_availability(df, spec)

    def fake_load_visible(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            return stamped
        raise ValueError("unexpected dataset")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)

    def runner(config: AllocationConfig) -> AllocationResult:
        wealth = 100.0
        if config.targets_override == {"SOXX": 1.0}:
            wealth = 110.0
        return AllocationResult(
            config=config,
            snapshots=(AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=wealth, contribution_krw=0, fees_krw=0),),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    snapshot = compute_evidence_vector(thesis=thesis, settings=settings, as_of=as_of, runner=runner)
    assert snapshot.overlap.status == "computed"
    assert snapshot.overlap.metrics["overlap_pct"] == pytest.approx(50.0)


@pytest.mark.parametrize("scenario_id", ["EV-NO-adoption-import"])
def test_ev_no_adoption_import(scenario_id: str) -> None:
    text_ev = Path("src/analytics/thesis_evidence.py").read_text(encoding="utf-8")
    assert "adoption_passes" not in text_ev
    text_rpt = Path("src/analytics/thesis_report.py").read_text(encoding="utf-8")
    assert "adoption_passes" not in text_rpt


def test_ev_adaptive_horizon_not_hardcoded_120(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector
    from src.validation.accumulation_cohort import AccumulationCohortReport, AccumulationCohortRow, CohortOverlapMetadata
    from src.validation.prospective import EvaluationHorizon

    thesis = ThesisSpec(
        id=ThesisId.AI_COMPUTE,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_semiconductor"],
        historical_proxies=["SOXX"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    captured: dict[str, int] = {}

    def fake_resolve(*args, **kwargs):
        return EvaluationHorizon(horizon_months=96, target_months=120, min_months=60, span_years=8.6, span_capped=True, reason="ok")

    def fake_proxy(*args, **kwargs):
        return (date(2016, 9, 30), date(2025, 4, 30))

    def fake_cohort(spec, runner, horizon_months=120, step_months=12, bootstrap_paths=400, seed=7):
        captured["horizon_months"] = int(horizon_months)
        # minimal report
        overlap = CohortOverlapMetadata(horizon_months=horizon_months, step_months=step_months)
        rows = (
            AccumulationCohortRow(candidate_wealth=110.0, baseline_wealth=100.0, ratio=1.1, candidate_recovery_months=None, cohort_start=date(2016, 9, 30), cohort_end=date(2024, 9, 29)),
        )
        return AccumulationCohortReport(
            name=spec.name,
            overlap=overlap,
            rows=rows,
            median_ratio=1.1,
            p10_ratio=1.1,
            worst_ratio=1.1,
            win_rate=1.0,
            bootstrap_p05_ratio_mean=1.0,
            unrecovered_cohort_count=0,
        )

    monkeypatch.setattr("src.validation.prospective.resolve_evaluation_horizon", fake_resolve)
    monkeypatch.setattr("src.validation.prospective.resolve_proxy_history_span", fake_proxy)
    monkeypatch.setattr("src.validation.accumulation_cohort.run_accumulation_cohort_report", fake_cohort)

    def fake_load_visible(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            raise ValueError("no holdings")
        return pl.DataFrame()

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)

    def runner(config: AllocationConfig) -> AllocationResult:
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )

    snapshot = compute_evidence_vector(thesis=thesis, settings=settings, as_of=as_of, runner=runner)
    assert captured.get("horizon_months") == 96
    assert snapshot.historical.status == "computed"


