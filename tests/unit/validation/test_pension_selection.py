"""Invariant guards for the pre-registered pension ETF selection verdict."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from src.analytics.pension_selection import ArmGrowthRow, DcaTailStats, GrowthRegretTable
from src.data.calendar import load_calendar
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload
from src.sim.pension_engine import PensionDataError
from src.validation import pension_selection as pension_selection_module
from src.validation.pension_selection import (
    PensionSelectionSpec,
    decide_pension_selection,
    load_pension_selection_spec,
    run_pension_selection,
    write_pension_selection_report,
)

_REPO = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO / "experiments" / "pension_selection_v1.json"


def _document() -> dict[str, object]:
    document = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _write_config(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "pension_selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_committed_pension_selection_config_loads() -> None:
    """The committed experiment fixes six standalone arms and the 17-point delta grid."""
    spec = load_pension_selection_spec(_CONFIG_PATH)
    assert spec.name == "pension_selection_v1"
    assert len(spec.arms) == 6
    assert spec.baseline_arm_id == "sp500_100"
    assert spec.delta_grid_points == 17
    assert all(Path(path).is_file() for path in spec.historical.campaign_config_paths)


def test_experiment_index_catalogs_pension_selection() -> None:
    """The active selection config is discoverable in the canonical experiment taxonomy."""
    index = json.loads((_REPO / "experiments" / "INDEX.json").read_text(encoding="utf-8"))
    assert index["files"]["pension_selection_v1.json"] == {
        "status": "active",
        "kind": "pension",
        "notes": "standalone pension ETF selection: delta-regret, stress tail, dot-com historical gates",
    }


def test_non_eligible_arm_ticker_is_rejected(tmp_path: Path) -> None:
    """An arm without a pension-eligible Korean identity fails before any run."""
    document = _document()
    arms = document["arms"]
    assert isinstance(arms, dict)
    arms["unsafe"] = {"IWM": 1.0}
    with pytest.raises(ValueError, match="IWM"):
        load_pension_selection_spec(_write_config(tmp_path, document))


def test_unknown_top_level_key_is_rejected_but_notes_are_allowed(tmp_path: Path) -> None:
    """Only optional free text may be added outside the pre-registered schema."""
    document = _document()
    document["foo"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        load_pension_selection_spec(_write_config(tmp_path, document))
    without_notes = _document()
    without_notes.pop("notes")
    assert load_pension_selection_spec(_write_config(tmp_path, without_notes)).name == "pension_selection_v1"
    notes_only = _document()
    notes_only["notes"] = "prior rationale"
    assert load_pension_selection_spec(_write_config(tmp_path, notes_only)).name == "pension_selection_v1"


def test_even_delta_grid_is_rejected(tmp_path: Path) -> None:
    """The scenario grid must preserve its posterior midpoint."""
    document = _document()
    grid = document["delta_grid"]
    assert isinstance(grid, dict)
    grid["n_points"] = 16
    with pytest.raises(ValueError, match="odd"):
        load_pension_selection_spec(_write_config(tmp_path, document))


def test_zero_fallback_cap_and_duplicate_json_keys_are_handled(tmp_path: Path) -> None:
    """A zero fallback budget is valid, while duplicate arm keys never collapse silently."""
    document = _document()
    document["max_fx_fallback_share"] = 0.0
    assert load_pension_selection_spec(_write_config(tmp_path, document)).max_fx_fallback_share == 0.0
    document = _document()
    tail = _dict_field(document, "tail")
    tail["block_months"] = 241
    assert load_pension_selection_spec(_write_config(tmp_path, document)).tail.block_months == 241
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"arms":{"sp500_100":{"SPY":1.0},"sp500_100":{"QQQ":1.0}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_pension_selection_spec(duplicate)


def _dict_field(document: dict[str, object], key: str) -> dict[str, object]:
    value = document[key]
    assert isinstance(value, dict)
    return value


def test_non_object_config_is_rejected(tmp_path: Path) -> None:
    """The top-level campaign contract must be a JSON object."""
    path = tmp_path / "selection.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_pension_selection_spec(path)


def _set_unknown_market_ticker(document: dict[str, object]) -> None:
    weights = _dict_field(_dict_field(document, "capm"), "market_weights")
    weights.clear()
    weights["IWM"] = 1.0


def _set_blank_arm_ticker(document: dict[str, object]) -> None:
    document["arms"] = {"bad": {"": 1.0}}


def _set_arm_weight_type(document: dict[str, object]) -> None:
    _dict_field(document, "arms")["nasdaq_100"] = []


@pytest.mark.parametrize(
    ("case_id", "mutate", "message"),
    [
        ("missing_field", lambda d: d.pop("name"), "missing fields"),
        ("notes_type", lambda d: d.__setitem__("notes", 1), "notes must be a string"),
        ("blank_name", lambda d: d.__setitem__("name", " "), "name must be a non-blank"),
        ("bad_identity_path", lambda d: d.__setitem__("etf_identity_path", "missing.json"), "does not exist"),
        ("bad_tax_path", lambda d: d.__setitem__("tax_regime_path", "missing.json"), "does not exist"),
        ("bad_date", lambda d: d.__setitem__("estimation_start", "not-a-date"), "must be an ISO date"),
        ("reversed_window", lambda d: d.__setitem__("estimation_start", "2027-01-01"), "must not be after"),
        ("zero_fx_age", lambda d: d.__setitem__("max_fx_age_days", 0), "positive integer"),
        ("missing_baseline", lambda d: d.__setitem__("baseline_arm_id", "ghost"), "not present"),
        ("empty_arms", lambda d: d.__setitem__("arms", {}), "arms must be non-empty"),
        ("blank_arm", lambda d: d.__setitem__("arms", {" ": {"SPY": 1.0}}), "non-blank"),
        ("arm_weight_sum", lambda d: _dict_field(d, "arms")["nasdaq_100"].__setitem__("QQQ", 0.5), "sum to 1"),
        ("market_weight_sum", lambda d: _dict_field(_dict_field(d, "capm"), "market_weights").__setitem__("SPY", 0.5), "sum to 1"),
        ("market_unknown", _set_unknown_market_ticker, "outside the arm union"),
        ("anchor_unknown", lambda d: _dict_field(d, "capm").__setitem__("anchor_ticker", "IWM"), "anchor ticker"),
        ("nonfinite_rate", lambda d: _dict_field(d, "capm").__setitem__("risk_free_annual", float("nan")), "finite number"),
        ("zero_prior_sd", lambda d: _dict_field(d, "delta_prior").__setitem__("sd_annual", 0.0), "must be positive"),
        ("zero_grid_z", lambda d: _dict_field(d, "delta_grid").__setitem__("z", 0.0), "must be positive"),
        ("negative_regret_tolerance", lambda d: d.__setitem__("regret_tolerance_annual", -0.1), "nonnegative"),
        ("zero_tail_horizon", lambda d: _dict_field(d, "tail").__setitem__("horizon_months", 0), "positive integer"),
        ("tail_quantile", lambda d: _dict_field(d, "tail").__setitem__("quantile", 0.5), "must lie in"),
        ("pre_retirement_long", lambda d: _dict_field(d, "tail").__setitem__("pre_retirement_months", 241), "must not exceed"),
        ("empty_campaigns", lambda d: _dict_field(d, "historical").__setitem__("campaign_config_paths", []), "non-empty array"),
        ("duplicate_campaigns", lambda d: _dict_field(d, "historical").__setitem__("campaign_config_paths", ["experiments/pension_campaign_v2_dotcom.json", "experiments/pension_campaign_v2_dotcom.json"]), "must be unique"),
        ("empty_horizons", lambda d: _dict_field(d, "historical").__setitem__("horizons_months", []), "non-empty array"),
        ("duplicate_horizons", lambda d: _dict_field(d, "historical").__setitem__("horizons_months", [120, 120]), "must be unique"),
        ("nonfinite_floor", lambda d: _dict_field(d, "historical").__setitem__("worst_ratio_floor", float("inf")), "finite number"),
        ("nested_unknown", lambda d: _dict_field(d, "capm").__setitem__("foo", 1), "unknown fields"),
        ("nested_missing", lambda d: _dict_field(d, "capm").pop("anchor_ticker"), "missing fields"),
        ("fallback_above_one", lambda d: d.__setitem__("max_fx_fallback_share", 1.1), "must lie in"),
        ("arms_type", lambda d: d.__setitem__("arms", []), "must be an object"),
        ("arm_weights_type", _set_arm_weight_type, "non-empty object"),
        ("blank_arm_ticker", _set_blank_arm_ticker, "ticker keys"),
        ("boolean_arm_weight", lambda d: _dict_field(d, "arms")["nasdaq_100"].__setitem__("QQQ", True), "finite number"),
        ("zero_arm_weight", lambda d: _dict_field(d, "arms")["nasdaq_100"].__setitem__("QQQ", 0.0), "must lie in"),
        ("market_weights_type", lambda d: _dict_field(d, "capm").__setitem__("market_weights", []), "non-empty object"),
        ("capm_type", lambda d: d.__setitem__("capm", []), "must be an object"),
        ("prior_type", lambda d: d.__setitem__("delta_prior", []), "must be an object"),
        ("grid_type", lambda d: d.__setitem__("delta_grid", []), "must be an object"),
        ("grid_too_small", lambda d: _dict_field(d, "delta_grid").__setitem__("n_points", 1), "odd and"),
        ("tail_type", lambda d: d.__setitem__("tail", []), "must be an object"),
        ("zero_tail_paths", lambda d: _dict_field(d, "tail").__setitem__("n_paths", 0), "positive integer"),
        ("zero_tail_block", lambda d: _dict_field(d, "tail").__setitem__("block_months", 0), "positive integer"),
        ("zero_pre_retirement", lambda d: _dict_field(d, "tail").__setitem__("pre_retirement_months", 0), "positive integer"),
        ("historical_type", lambda d: d.__setitem__("historical", []), "must be an object"),
        ("campaign_paths_type", lambda d: _dict_field(d, "historical").__setitem__("campaign_config_paths", "bad"), "non-empty array"),
        ("historical_horizons_type", lambda d: _dict_field(d, "historical").__setitem__("horizons_months", "bad"), "non-empty array"),
    ],
)
def test_config_loader_rejects_invalid_contract_fields(
    tmp_path: Path, case_id: str, mutate: Callable[[dict[str, object]], None], message: str
) -> None:
    """Every malformed economic or schema field fails closed with a named error."""
    document = _document()
    mutate(document)
    with pytest.raises(ValueError, match=message):
        load_pension_selection_spec(_write_config(tmp_path, document))


def _growth_row(arm_id: str, max_regret: float, volatility: float) -> ArmGrowthRow:
    return ArmGrowthRow(
        arm_id=arm_id,
        volatility_annual=volatility,
        growth_by_delta=(0.0, 0.0),
        max_regret=max_regret,
        mean_regret=max_regret / 2.0,
    )


def _tail_row(arm_id: str, multiple: float, drawdown: float) -> DcaTailStats:
    return DcaTailStats(
        arm_id=arm_id,
        delta=-0.02,
        horizon_months=240,
        n_paths=100,
        quantile=0.05,
        low_quantile_terminal_multiple=multiple,
        median_terminal_multiple=multiple + 0.1,
        high_quantile_pre_retirement_drawdown=drawdown,
        prob_below_principal=0.0,
    )


def _decision_inputs(
    regrets: dict[str, float],
    tails: dict[str, tuple[float, float]],
    historical: dict[str, float | None],
    volatilities: dict[str, float] | None = None,
) -> tuple[PensionSelectionSpec, GrowthRegretTable, tuple[DcaTailStats, ...], dict[str, float | None]]:
    loaded = load_pension_selection_spec(_CONFIG_PATH)
    selected_arms = {arm_id: loaded.arms[arm_id] for arm_id in regrets}
    spec = replace(loaded, arms=selected_arms, baseline_arm_id=next(iter(selected_arms)))
    growth = GrowthRegretTable(
        deltas=(-0.02, 0.02),
        baseline_arm_id=spec.baseline_arm_id,
        rows=tuple(
            _growth_row(arm_id, value, (volatilities or {}).get(arm_id, 0.2))
            for arm_id, value in regrets.items()
        ),
        breakeven_vs_baseline=dict.fromkeys(regrets),
    )
    tail = tuple(_tail_row(arm_id, *values) for arm_id, values in tails.items())
    return spec, growth, tail, historical


def test_regret_gate_uses_tolerance_above_minimum() -> None:
    """Only regret more than the minimum plus the declared tolerance fails."""
    spec, growth, tail, historical = _decision_inputs(
        {"sp500_100": 0.010, "nasdaq_100": 0.014, "qqq90_soxx10": 0.016},
        dict.fromkeys(("sp500_100", "nasdaq_100", "qqq90_soxx10"), (1.1, 0.2)),
        dict.fromkeys(("sp500_100", "nasdaq_100", "qqq90_soxx10"), 1.0),
    )
    verdicts, _ = decide_pension_selection(spec, growth, tail, historical)
    assert [verdict.regret_pass for verdict in verdicts] == [True, True, False]
    assert verdicts[2].reasons == ("REGRET_ABOVE_TOLERANCE",)


def test_stress_tail_gates_fire_independently() -> None:
    """Principal and drawdown failures remain separate and each clears tail_pass."""
    spec, growth, tail, historical = _decision_inputs(
        {"sp500_100": 0.01, "nasdaq_100": 0.01},
        {"sp500_100": (0.9, 0.2), "nasdaq_100": (1.2, 0.6)},
        {"sp500_100": 1.0, "nasdaq_100": 1.0},
    )
    verdicts, _ = decide_pension_selection(spec, growth, tail, historical)
    assert verdicts[0].reasons == ("STRESS_PRINCIPAL_TAIL",)
    assert verdicts[1].reasons == ("STRESS_PRE_RETIREMENT_DRAWDOWN",)
    assert not any(verdict.tail_pass for verdict in verdicts)


def test_historical_ratio_below_floor_is_named() -> None:
    """Observed but insufficient historical evidence carries its own fail-closed code."""
    spec, growth, tail, historical = _decision_inputs(
        {"sp500_100": 0.01, "nasdaq_100": 0.01},
        {"sp500_100": (1.1, 0.2), "nasdaq_100": (1.1, 0.2)},
        {"sp500_100": 1.0, "nasdaq_100": 0.89},
    )
    verdicts, selected = decide_pension_selection(spec, growth, tail, historical)
    assert verdicts[1].reasons == ("HISTORICAL_BELOW_FLOOR",)
    assert selected == "sp500_100"


def test_missing_historical_evidence_fails_closed() -> None:
    """An arm absent from the historical campaigns cannot pass the final gate."""
    spec, growth, tail, historical = _decision_inputs(
        {"sp500_100": 0.01, "nasdaq_100": 0.01},
        {"sp500_100": (1.1, 0.2), "nasdaq_100": (1.1, 0.2)},
        {"sp500_100": 1.0, "nasdaq_100": None},
    )
    verdicts, selected = decide_pension_selection(spec, growth, tail, historical)
    assert verdicts[1].reasons == ("NO_HISTORICAL_EVIDENCE",)
    assert selected == "sp500_100"


def test_parsimony_breaks_ties_by_ticker_count_volatility_then_id() -> None:
    """Passing arms prefer fewer tickers, lower volatility, then lexical id."""
    spec, growth, tail, historical = _decision_inputs(
        {"sp500_100": 0.01, "nasdaq_100": 0.01, "qqq90_soxx10": 0.01},
        dict.fromkeys(("sp500_100", "nasdaq_100", "qqq90_soxx10"), (1.1, 0.2)),
        dict.fromkeys(("sp500_100", "nasdaq_100", "qqq90_soxx10"), 1.0),
        volatilities={"sp500_100": 0.3, "nasdaq_100": 0.1, "qqq90_soxx10": 0.1},
    )
    _, selected = decide_pension_selection(spec, growth, tail, historical)
    assert selected == "nasdaq_100"

    spec, growth, tail, historical = _decision_inputs(
        {"sp500_100": 0.01, "nasdaq_100": 0.01, "qqq90_soxx10": 0.01},
        dict.fromkeys(("sp500_100", "nasdaq_100", "qqq90_soxx10"), (1.1, 0.2)),
        dict.fromkeys(("sp500_100", "nasdaq_100", "qqq90_soxx10"), 1.0),
        volatilities=dict.fromkeys(("sp500_100", "nasdaq_100", "qqq90_soxx10"), 0.2),
    )
    _, selected = decide_pension_selection(spec, growth, tail, historical)
    assert selected == "nasdaq_100"


def test_no_passer_yields_no_selection() -> None:
    """The baseline is never silently substituted when all arms fail."""
    spec, growth, tail, historical = _decision_inputs(
        {"sp500_100": 0.1, "nasdaq_100": 0.1},
        {"sp500_100": (0.5, 0.8), "nasdaq_100": (0.5, 0.8)},
        {"sp500_100": 0.5, "nasdaq_100": 0.5},
    )
    verdicts, selected = decide_pension_selection(spec, growth, tail, historical)
    assert selected is None
    assert all(verdict.reasons for verdict in verdicts)


def test_decision_rejects_incomplete_or_mismatched_coverage() -> None:
    """Growth, tail, historical, and baseline coverage must match the config exactly."""
    spec, growth, tail, historical = _decision_inputs(
        {"sp500_100": 0.01, "nasdaq_100": 0.01},
        {"sp500_100": (1.1, 0.2), "nasdaq_100": (1.1, 0.2)},
        {"sp500_100": 1.0, "nasdaq_100": 1.0},
    )
    with pytest.raises(ValueError, match="stress tail"):
        decide_pension_selection(spec, growth, tail[:-1], historical)
    with pytest.raises(ValueError, match="historical ratios"):
        decide_pension_selection(spec, growth, tail, {"sp500_100": 1.0})
    wrong_baseline = GrowthRegretTable(
        deltas=growth.deltas,
        baseline_arm_id="nasdaq_100",
        rows=growth.rows,
        breakeven_vs_baseline=growth.breakeven_vs_baseline,
    )
    with pytest.raises(ValueError, match="baseline"):
        decide_pension_selection(spec, wrong_baseline, tail, historical)
    duplicate_tail = (tail[0], tail[0])
    with pytest.raises(ValueError, match="duplicate arm"):
        decide_pension_selection(spec, growth, duplicate_tail, historical)
    with pytest.raises(ValueError, match="finite"):
        decide_pension_selection(spec, growth, tail, {**historical, "sp500_100": float("nan")})


def _runtime_config(
    tmp_path: Path,
    *,
    tax_regime_path: str | None = None,
) -> tuple[Path, PensionSelectionSpec]:
    document = _document()
    document["estimation_start"] = "2021-01-01"
    document["estimation_end"] = "2024-12-31"
    if tax_regime_path is not None:
        document["tax_regime_path"] = tax_regime_path
    tail = _dict_field(document, "tail")
    tail.update(
        {
            "horizon_months": 12,
            "n_paths": 16,
            "block_months": 3,
            "pre_retirement_months": 4,
        }
    )
    _dict_field(document, "historical")["horizons_months"] = [12]
    path = _write_config(tmp_path, document)
    return path, load_pension_selection_spec(path)


def _stub_campaign_reports(
    campaign_spec: object,
    _settings: DataSettings,
    *,
    seed: int,
) -> SimpleNamespace:
    assert seed == 17
    summaries = tuple(
        SimpleNamespace(arm_id=arm.arm_id, horizon_months=12, worst_wealth_ratio=1.0)
        for arm in campaign_spec.arms  # type: ignore[attr-defined]
    )
    return SimpleNamespace(summaries=summaries)


def _persist_dividend_proxy_lake(settings: DataSettings, start: date, end: date) -> None:
    retrieved_at = datetime(2025, 1, 1, tzinfo=UTC)
    sessions = list(load_calendar("XNYS").sessions(start, end))
    rows: list[dict[str, object]] = []
    for ticker, base, drift in (("SPY", 400.0, 0.10), ("QQQ", 300.0, 0.16), ("SOXX", 200.0, 0.22)):
        for index, day in enumerate(sessions):
            price = base + drift * index
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 10_000,
                    "adjusted_close": price,
                    "dividend": 0.01 * price if ticker == "QQQ" and index % 63 == 0 else 0.0,
                    "split_factor": 1.0,
                    "source": "synthetic",
                    "retrieved_at": retrieved_at,
                }
            )
    prices = pl.DataFrame(rows, schema=dict(spec_for(Dataset.PRICES).columns)).select(
        list(spec_for(Dataset.PRICES).columns)
    )
    payload = RawPayload(
        provider="synthetic",
        endpoint="pension-selection",
        request_params={},
        retrieved_at=retrieved_at,
        extension="json",
        content=b"{}",
    )
    persist_ingest(prices, Dataset.PRICES, payload, settings)
    fx = pl.DataFrame(
        {"date": sessions, "usdkrw": [1300.0] * len(sessions), "source": ["synthetic"] * len(sessions),
         "retrieved_at": [retrieved_at] * len(sessions)},
        schema=dict(spec_for(Dataset.FX_KRW_BASE).columns),
    )
    persist_ingest(fx, Dataset.FX_KRW_BASE, payload, settings)


def test_historical_worst_ratio_is_minimum_and_ignores_other_horizons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aggregation takes the minimum allowed horizon and poisons on any undefined summary."""
    del tmp_path
    spec = load_pension_selection_spec(_CONFIG_PATH)
    reports = iter(
        (
            SimpleNamespace(
                summaries=(
                    SimpleNamespace(arm_id="sp500_100", horizon_months=120, worst_wealth_ratio=1.05),
                    SimpleNamespace(arm_id="sp500_100", horizon_months=240, worst_wealth_ratio=1.01),
                    SimpleNamespace(arm_id="sp500_100", horizon_months=60, worst_wealth_ratio=0.10),
                    SimpleNamespace(arm_id="nasdaq_100", horizon_months=120, worst_wealth_ratio=0.97),
                    SimpleNamespace(arm_id="nasdaq_100", horizon_months=240, worst_wealth_ratio=0.99),
                )
            ),
            SimpleNamespace(
                summaries=(
                    SimpleNamespace(arm_id="sp500_100", horizon_months=120, worst_wealth_ratio=1.10),
                    SimpleNamespace(arm_id="nasdaq_100", horizon_months=240, worst_wealth_ratio=1.10),
                    SimpleNamespace(arm_id="qqq90_soxx10", horizon_months=240, worst_wealth_ratio=None),
                    SimpleNamespace(arm_id="qqq90_soxx10", horizon_months=120, worst_wealth_ratio=0.95),
                )
            ),
        )
    )
    monkeypatch.setattr(pension_selection_module, "run_pension_campaign", lambda *args, **kwargs: next(reports))
    result = pension_selection_module._historical_worst_ratios(
        spec, DataSettings(data_root="unused"), seed=17
    )
    assert result["sp500_100"] == pytest.approx(1.01)
    assert result["nasdaq_100"] == pytest.approx(0.97)
    assert result["qqq90_soxx10"] is None


def test_historical_baseline_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A campaign anchored to another baseline cannot be aggregated silently."""
    spec = load_pension_selection_spec(_CONFIG_PATH)
    campaign = pension_selection_module.load_pension_campaign_spec(
        spec.historical.campaign_config_paths[0]
    )
    wrong_campaign = replace(campaign, baseline_arm_id="other")
    monkeypatch.setattr(pension_selection_module, "load_pension_campaign_spec", lambda _path: wrong_campaign)
    with pytest.raises(ValueError, match="baseline"):
        pension_selection_module._historical_worst_ratios(
            spec, DataSettings(data_root="unused"), seed=17
        )
    assert campaign.baseline_arm_id == spec.baseline_arm_id


def test_historical_target_or_household_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical evidence cannot change targets or reintroduce a household account view."""
    spec = load_pension_selection_spec(_CONFIG_PATH)
    campaign = pension_selection_module.load_pension_campaign_spec(
        spec.historical.campaign_config_paths[0]
    )
    changed_arm = replace(campaign.arms[0], targets={"QQQ": 1.0})
    wrong_targets = replace(campaign, arms=(changed_arm, *campaign.arms[1:]))
    monkeypatch.setattr(pension_selection_module, "load_pension_campaign_spec", lambda _path: wrong_targets)
    with pytest.raises(ValueError, match="targets differ"):
        pension_selection_module._historical_worst_ratios(
            spec, DataSettings(data_root="unused"), seed=17
        )

    household = replace(campaign, household=SimpleNamespace())
    monkeypatch.setattr(pension_selection_module, "load_pension_campaign_spec", lambda _path: household)
    with pytest.raises(ValueError, match="household_view"):
        pension_selection_module._historical_worst_ratios(
            spec, DataSettings(data_root="unused"), seed=17
        )


def test_run_selection_is_deterministic_and_writes_korean_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Certified synthetic data produces a repeatable report and Korean ETF decision record."""
    from tests.unit.validation.test_pension_campaign import _persist_proxy_lake

    settings = DataSettings(data_root=str(tmp_path / "data"))
    _persist_proxy_lake(settings, date(2021, 1, 1), date(2024, 12, 31))
    _, spec = _runtime_config(tmp_path)
    monkeypatch.setattr(pension_selection_module, "run_pension_campaign", _stub_campaign_reports)
    first = run_pension_selection(spec, settings, seed=17)
    second = run_pension_selection(spec, settings, seed=17)
    assert first == second
    assert first.panel_months == 47
    assert first.selected_arm_id is not None
    assert set(first.selected_kr_targets).issubset({"379800", "379810", "469060"})
    assert first.selected_kr_targets

    path = write_pension_selection_report(
        first,
        settings,
        experiment_id="selection123",
        provenance={"config_sha256": "a", "seed": "17"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == first.status
    assert payload["selected_kr_targets"] == dict(first.selected_kr_targets)
    assert payload["provenance"]["seed"] == "17"
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "연금 ETF 선택 판정" in markdown
    assert "투자 권유가 아니다" in markdown
    assert next(iter(first.selected_kr_targets)) in markdown


def test_withholding_rate_comes_from_tax_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the certified regime's dividend withholding changes the scenario panel."""
    settings = DataSettings(data_root=str(tmp_path / "data"))
    _persist_dividend_proxy_lake(settings, date(2021, 1, 1), date(2024, 12, 31))
    _, default_spec = _runtime_config(tmp_path)
    regime = json.loads((_REPO / "configs" / "tax" / "kr_pension_2026.json").read_text(encoding="utf-8"))
    regime["foreign_dividend_withholding_rate"] = 0.30
    regime["regime_id"] = "KR_PENSION_TEST_WITHHOLDING"
    regime_path = tmp_path / "tax.json"
    regime_path.write_text(json.dumps(regime), encoding="utf-8")
    _, changed_spec = _runtime_config(tmp_path, tax_regime_path=str(regime_path))
    monkeypatch.setattr(pension_selection_module, "run_pension_campaign", _stub_campaign_reports)
    default = run_pension_selection(default_spec, settings, seed=17)
    changed = run_pension_selection(changed_spec, settings, seed=17)
    assert default.growth_table != changed.growth_table


def test_missing_certified_market_data_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent required prices or FX are translated to the public pension data error."""
    _, spec = _runtime_config(tmp_path)
    monkeypatch.setattr(pension_selection_module, "run_pension_campaign", _stub_campaign_reports)
    with pytest.raises(PensionDataError, match="absent or stale"):
        run_pension_selection(spec, DataSettings(data_root=str(tmp_path / "empty")), seed=17)


def test_invalid_or_over_cap_fx_series_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed FX merges and excessive fallback use both abort with pension data errors."""
    from tests.unit.validation.test_pension_campaign import _persist_proxy_lake

    settings = DataSettings(data_root=str(tmp_path / "data"))
    _persist_proxy_lake(settings, date(2021, 1, 1), date(2024, 12, 31))
    _, spec = _runtime_config(tmp_path)

    def invalid(_base: pl.DataFrame, _fallback: pl.DataFrame | None) -> object:
        raise ValueError("malformed quotes")

    monkeypatch.setattr(pension_selection_module, "build_krw_fx_series", invalid)
    with pytest.raises(PensionDataError, match="fx series is invalid"):
        run_pension_selection(spec, settings, seed=17)

    def over_cap(_base: pl.DataFrame, _fallback: pl.DataFrame | None) -> SimpleNamespace:
        return SimpleNamespace(
            frame=pl.DataFrame(),
            status=SimpleNamespace(value="APPLIED"),
            fallback_source="fred",
            fallback_session_dates=lambda sessions: sessions,
            provenance=lambda _sessions: {},
        )

    monkeypatch.setattr(pension_selection_module, "build_krw_fx_series", over_cap)
    with pytest.raises(PensionDataError, match="fallback share"):
        run_pension_selection(spec, settings, seed=17)
