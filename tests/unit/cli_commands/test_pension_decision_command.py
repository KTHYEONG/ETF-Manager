"""CLI contract tests for the robust pension decision command."""

from __future__ import annotations

import calendar as _calendar
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

import src.cli_commands.campaign as campaign_mod
import src.data.catalog as catalog_mod
import src.data.result_store as result_store_mod
from src import cli
from src.cli import main
from src.cli_commands.parser import _build_parser
from src.data.settings import DataSettings
from src.data.schema import Dataset
from src.validation import pension_campaign as pension_campaign_mod
from src.validation import pension_decision as pension_decision_mod

_REPO = Path(__file__).resolve().parents[3]
_GIT_COMMIT = "0" * 40


def _months(first: date, count: int) -> list[date]:
    out: list[date] = []
    year, month = first.year, first.month
    for _ in range(count):
        out.append(date(year, month, _calendar.monthrange(year, month)[1]))
        month += 1
        if month > 12:
            year += 1
            month = 1
    return out


def _config(tmp_path: Path) -> str:
    document = json.loads((_REPO / "experiments" / "pension_decision_v1.json").read_text(encoding="utf-8"))
    document["modern_start"] = "2000-01-31"
    document["modern_end"] = "2012-06-30"
    document["century_start"] = "2000-01-31"
    # 60일 공개 지연 때문에 2012-06-30 결정 시점에 보이는 마지막 월은 2012-04-30이다.
    document["century_end"] = "2012-04-30"
    document["horizons_years"] = [10, 12]
    document["bootstrap_paths"] = 10
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _prices_frame(months: list[date]) -> pl.DataFrame:
    tickers, dates, closes = [], [], []
    for ticker, rate in (("SPY", 0.004), ("QQQ", 0.006)):
        for index, day in enumerate(months):
            tickers.append(ticker)
            dates.append(day)
            closes.append(100.0 * (1.0 + rate) ** index)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "adjusted_close": closes,
            "available_at": [datetime(2000, 1, 1, tzinfo=UTC)] * len(dates),
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def _research_frame(months: list[date]) -> pl.DataFrame:
    series, ends, returns, stamps = [], [], [], []
    for series_id, rate in (("ff_mkt_monthly", 0.004), ("ff_hitec_monthly", 0.006)):
        for day in months:
            series.append(series_id)
            ends.append(day)
            returns.append(rate)
            stamps.append(datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=60))
    return pl.DataFrame(
        {
            "series_id": series,
            "period_end": ends,
            "simple_return": returns,
            "label": ["research_proxy"] * len(ends),
            "source": ["synthetic"] * len(ends),
            "available_at": stamps,
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


def _summaries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in (120, 144):
        rows.extend(
            [
                {"arm_id": "nasdaq_100", "horizon_months": horizon, "median_wealth_ratio": 1.2},
                {"arm_id": "sp50_nq50", "horizon_months": horizon, "median_wealth_ratio": 1.1},
                {"arm_id": "sp70_nq30", "horizon_months": horizon, "median_wealth_ratio": 1.05},
                {"arm_id": "sp500_100", "horizon_months": horizon, "median_wealth_ratio": 1.0},
            ]
        )
    return rows


def _install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, object],
    *,
    agreement: bool = True,
) -> None:
    months = _months(date(1999, 12, 31), 151)
    frames = {"prices": _prices_frame(months), "research_monthly": _research_frame(months[1:])}
    snapshot = SimpleNamespace(
        artifacts={
            Dataset.PRICES: SimpleNamespace(manifest_path="/m/prices.json"),
            Dataset.RESEARCH_MONTHLY: SimpleNamespace(manifest_path="/m/research.json"),
        }
    )
    monkeypatch.setattr(catalog_mod, "resolve_snapshot", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(
        catalog_mod, "load_snapshot_visible", lambda _snap, dataset, _ts: frames[str(dataset)]
    )
    monkeypatch.setattr(
        pension_campaign_mod,
        "run_pension_campaign",
        lambda *_a, **_k: SimpleNamespace(summaries=[SimpleNamespace(**row) for row in _summaries()]),
    )
    if not agreement:
        monkeypatch.setattr(pension_decision_mod, "assert_tax_rank_neutrality", lambda *_a, **_k: False)
    monkeypatch.setattr(campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)

    def write(
        _settings: DataSettings,
        *,
        experiment: str,
        kind: object,
        run_id: str,
        payload: dict[str, object],
        markdown: str | None = None,
        written_at: object = None,
    ) -> SimpleNamespace:
        captured["experiment"] = experiment
        captured["kind"] = str(kind)
        captured["run_id"] = run_id
        captured["payload"] = payload
        captured["markdown"] = markdown
        return SimpleNamespace(json_path=tmp_path / f"decision_{run_id}.json")

    monkeypatch.setattr(result_store_mod, "write_result", write)


def test_pension_decision_command_records_consumed_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command exits zero and the payload pins the snapshot manifest identities."""
    captured: dict[str, object] = {}
    _install(tmp_path, monkeypatch, captured)
    code = campaign_mod.run_pension_decision_command(
        config_path=_config(tmp_path),
        settings=DataSettings(data_root=str(tmp_path / "data")),
        seed=11,
    )
    assert code == 0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["manifest_hashes"] == {"prices": "prices", "research_monthly": "research"}
    assert payload["status"] == "ADOPT_CANDIDATE"
    assert captured["experiment"] == "pension_decision_v1"
    assert isinstance(captured["markdown"], str)
    assert "상태" in captured["markdown"]


def test_pension_decision_tax_disagreement_forces_no_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A false tax agreement writes NO_DECISION with TAX_RANK_DISAGREEMENT."""
    captured: dict[str, object] = {}
    _install(tmp_path, monkeypatch, captured, agreement=False)
    with caplog.at_level("WARNING"):
        code = campaign_mod.run_pension_decision_command(
            config_path=_config(tmp_path),
            settings=DataSettings(data_root=str(tmp_path / "data")),
            seed=11,
        )
    assert code == 0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["status"] == "NO_DECISION"
    assert "TAX_RANK_DISAGREEMENT" in payload["reasons"]
    assert "pension_decision_tax_disagreement" in caplog.text


def test_pension_decision_incumbent_record_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record holding the selected candidate keeps it; bad records fail with 1."""
    captured: dict[str, object] = {}
    _install(tmp_path, monkeypatch, captured)
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"record_id": "r1", "incumbent_id": "spy0_qqq100"}), encoding="utf-8")
    code = campaign_mod.run_pension_decision_command(
        config_path=_config(tmp_path),
        settings=DataSettings(data_root=str(tmp_path / "data")),
        seed=11,
        incumbent_record=str(record),
    )
    assert code == 0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["status"] == "KEEP_INCUMBENT"
    assert payload["selected_id"] == "spy0_qqq100"
    assert (
        campaign_mod.run_pension_decision_command(
            config_path=_config(tmp_path),
            settings=DataSettings(data_root=str(tmp_path / "data")),
            seed=11,
            incumbent_record=str(tmp_path / "missing.json"),
        )
        == 1
    )
    blank = tmp_path / "blank.json"
    blank.write_text(json.dumps({"incumbent_id": "  "}), encoding="utf-8")
    assert (
        campaign_mod.run_pension_decision_command(
            config_path=_config(tmp_path),
            settings=DataSettings(data_root=str(tmp_path / "data")),
            seed=11,
            incumbent_record=str(blank),
        )
        == 1
    )
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    assert (
        campaign_mod.run_pension_decision_command(
            config_path=_config(tmp_path),
            settings=DataSettings(data_root=str(tmp_path / "data")),
            seed=11,
            incumbent_record=str(array),
        )
        == 1
    )


def test_parser_exposes_pension_decision_target() -> None:
    """The facade parser exposes config, seed, incumbent record, and freeze inputs."""
    args = _build_parser().parse_args(["run", "pension-decision", "--config", "d.json", "--seed", "1"])
    assert args.target == "pension-decision"
    assert args.config == "d.json"
    assert args.seed == 1
    assert args.incumbent_record is None
    assert args.freeze is False
    args = _build_parser().parse_args(
        ["run", "pension-decision", "--config", "d.json", "--seed", "1", "--incumbent-record", "r.json", "--freeze"]
    )
    assert args.incumbent_record == "r.json"
    assert args.freeze is True


def test_pension_decision_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The facade dispatches once with config, seed, record, and freeze; seed stays mandatory."""
    captured: dict[str, object] = {}

    def fake_command(*, config_path: str, settings: object, seed: int, incumbent_record: object, freeze: object) -> int:
        captured["config_path"] = config_path
        captured["settings"] = settings
        captured["seed"] = seed
        captured["incumbent_record"] = incumbent_record
        captured["freeze"] = freeze
        return 0

    monkeypatch.setattr(cli, "run_pension_decision_command", fake_command)
    assert main(["run", "pension-decision", "--config", "c.json", "--seed", "7"]) == 0
    assert captured["config_path"] == "c.json"
    assert captured["seed"] == 7
    assert captured["incumbent_record"] is None
    assert captured["freeze"] is False
    assert isinstance(captured["settings"], DataSettings)
    assert main(["run", "pension-decision", "--config", "c.json"]) == 2


def test_pension_review_exit_codes_follow_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stubbed HOLD and REVIEW_DUE statuses map to exit codes 0 and 3."""
    from types import SimpleNamespace

    import src.validation.pension_decision_record as record_mod

    record_file = tmp_path / "record.json"
    record_file.write_text(json.dumps({"record_id": "r1", "incumbent_id": "spy0_qqq100"}), encoding="utf-8")
    monkeypatch.setattr(
        record_mod, "load_pension_decision_record", lambda _path: SimpleNamespace(record_id="r1")
    )
    monkeypatch.setattr(catalog_mod, "resolve_snapshot", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(catalog_mod, "load_snapshot_visible", lambda *_a, **_k: SimpleNamespace())
    hold = SimpleNamespace(
        record_id="r1", as_of=date(2021, 1, 31), months_observed=6,
        incumbent_over_benchmark=0.9, state="HOLD",
    )
    monkeypatch.setattr(record_mod, "evaluate_pension_review", lambda *_a, **_k: hold)
    assert campaign_mod.run_pension_review_command(
        record_path=str(record_file),
        as_of=date(2021, 1, 31),
        settings=DataSettings(data_root=str(tmp_path / "data")),
    ) == 0
    due = SimpleNamespace(
        record_id="r1", as_of=date(2021, 2, 28), months_observed=12,
        incumbent_over_benchmark=0.9, state="REVIEW_DUE",
    )
    monkeypatch.setattr(record_mod, "evaluate_pension_review", lambda *_a, **_k: due)
    assert campaign_mod.run_pension_review_command(
        record_path=str(record_file),
        as_of=date(2021, 2, 28),
        settings=DataSettings(data_root=str(tmp_path / "data")),
    ) == 3
    monkeypatch.setattr(
        record_mod, "load_pension_decision_record", lambda _path: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert campaign_mod.run_pension_review_command(
        record_path=str(record_file),
        as_of=date(2021, 2, 28),
        settings=DataSettings(data_root=str(tmp_path / "data")),
    ) == 1


def test_pension_decision_freeze_links_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--freeze records the incumbent record id as previous_record_id."""
    import src.validation.pension_decision_record as record_mod

    captured: dict[str, object] = {}
    _install(tmp_path, monkeypatch, captured)
    captured_freeze: dict[str, object] = {}

    def fake_freeze(report: object, spec: object, **kwargs: object) -> Path:
        captured_freeze["kwargs"] = kwargs
        captured_freeze["report"] = report
        return tmp_path / "frozen.json"

    monkeypatch.setattr(record_mod, "freeze_pension_decision", fake_freeze)
    record = tmp_path / "incumbent.json"
    record.write_text(
        json.dumps({"record_id": "prev-record", "incumbent_id": "spy0_qqq100"}), encoding="utf-8"
    )
    code = campaign_mod.run_pension_decision_command(
        config_path=_config(tmp_path),
        settings=DataSettings(data_root=str(tmp_path / "data")),
        seed=11,
        incumbent_record=str(record),
        freeze=True,
    )
    assert code == 0
    assert isinstance(captured_freeze.get("kwargs"), dict)
    assert captured_freeze["kwargs"]["previous_record_id"] == "prev-record"  # type: ignore[index]


def test_parser_exposes_pension_review_target() -> None:
    """The facade parser exposes record and as-of inputs for pension-review."""
    args = _build_parser().parse_args(["run", "pension-review", "--record", "r.json", "--as-of", "2021-01-31"])
    assert args.target == "pension-review"
    assert args.record == "r.json"
    assert args.as_of == date(2021, 1, 31)


def test_pension_review_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The facade dispatches pension-review once with record and as-of."""
    captured: dict[str, object] = {}

    def fake_review(*, record_path: str, as_of: object, settings: object) -> int:
        captured["record_path"] = record_path
        captured["as_of"] = as_of
        captured["settings"] = settings
        return 0

    monkeypatch.setattr(cli, "run_pension_review_command", fake_review)
    assert main(["run", "pension-review", "--record", "r.json", "--as-of", "2021-01-31"]) == 0
    assert captured["record_path"] == "r.json"
    assert captured["as_of"] == date(2021, 1, 31)
    assert isinstance(captured["settings"], DataSettings)


def test_read_record_id_branches(tmp_path: Path) -> None:
    """Record-id extraction fails closed and returns None when absent."""
    with pytest.raises(ValueError, match="unreadable"):
        campaign_mod._read_record_id(str(tmp_path / "absent.json"))
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="record_id"):
        campaign_mod._read_record_id(str(array))
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"incumbent_id": "spy0_qqq100"}), encoding="utf-8")
    assert campaign_mod._read_record_id(str(legacy)) is None
    blank = tmp_path / "blank.json"
    blank.write_text(json.dumps({"record_id": "  "}), encoding="utf-8")
    with pytest.raises(ValueError, match="record_id"):
        campaign_mod._read_record_id(str(blank))


def test_pension_decision_rejects_century_month_not_yet_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A century month whose 60-day availability falls after the decision instant fails closed."""
    config = _config(tmp_path)
    document = json.loads(Path(config).read_text(encoding="utf-8"))
    document["century_end"] = "2012-06-30"
    Path(config).write_text(json.dumps(document), encoding="utf-8")
    captured: dict[str, object] = {}
    _install(tmp_path, monkeypatch, captured)
    with caplog.at_level("ERROR"):
        code = campaign_mod.run_pension_decision_command(
            config_path=config, settings=DataSettings(data_root=str(tmp_path / "data")), seed=7
        )
    assert code == 1
    assert "config requires 2000-01-31..2012-06-30" in caplog.text
    assert "payload" not in captured


def _splice_months(first: date, count: int) -> list[date]:
    return _months(first, count)


def _splice_config(tmp_path: Path) -> str:
    document = json.loads((_REPO / "experiments" / "pension_decision_v1.json").read_text(encoding="utf-8"))
    document["candidates"] = {
        "spy_only": {
            "start_weights": {"SPY": 1.0},
            "end_weights": {"SPY": 1.0},
            "glide_years": 0,
        },
        "schd_tilt": {
            "start_weights": {"SPY": 0.5, "SCHD": 0.5},
            "end_weights": {"SPY": 0.5, "SCHD": 0.5},
            "glide_years": 0,
        },
    }
    document["benchmark_id"] = "spy_only"
    document["neighbors"] = {"spy_only": [], "schd_tilt": []}
    document["century_series"] = {"SPY": "ff_mkt_monthly", "SCHD": "ff_dp_hi30_monthly"}
    document["modern_start"] = "2000-01-31"
    document["modern_end"] = "2002-06-30"
    document["century_start"] = "2000-01-31"
    document["century_end"] = "2002-04-30"
    document["horizons_years"] = [1, 2]
    document["step_months"] = 12
    document["pre_retirement_months"] = 12
    document["bootstrap_paths"] = 4
    document["bootstrap_block_months"] = 6
    document["annual_drag_by_sleeve"] = {}
    document["tax_crosscheck_arm_map"] = {"sp500_100": "spy_only"}
    document["controls"] = {
        "world": {
            "start_weights": {"VT": 1.0},
            "end_weights": {"VT": 1.0},
            "glide_years": 0,
        }
    }
    document["dominance_reference_id"] = "spy_only"
    document["modern_splices"] = {
        "SCHD": {"proxy_weights": {"ff_dp_hi30_monthly": 1.0}, "etf_first_month": "2001-01-31"}
    }
    path = tmp_path / "splice_decision.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _splice_prices_frame() -> pl.DataFrame:
    months = _splice_months(date(1999, 12, 31), 31)
    tickers, dates, closes = [], [], []
    for index, day in enumerate(months):
        tickers.append("SPY")
        dates.append(day)
        closes.append(100.0 * (1.004) ** index)
    for index, day in enumerate(months):
        tickers.append("VT")
        dates.append(day)
        closes.append(100.0 * (1.005) ** index)
    schd_months = _splice_months(date(2000, 12, 31), 19)
    for index, day in enumerate(schd_months):
        tickers.append("SCHD")
        dates.append(day)
        closes.append(100.0 * (1.006) ** index)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "adjusted_close": closes,
            "available_at": [datetime(2000, 1, 1, tzinfo=UTC)] * len(dates),
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def _splice_research_frame(*, with_proxy: bool = True) -> pl.DataFrame:
    months = _splice_months(date(2000, 1, 31), 28)
    series, ends, returns, stamps = [], [], [], []
    for day in months:
        series.append("ff_mkt_monthly")
        ends.append(day)
        returns.append(0.004)
        stamps.append(datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=60))
    if with_proxy:
        for day in months:
            series.append("ff_dp_hi30_monthly")
            ends.append(day)
            returns.append(0.005)
            stamps.append(datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=60))
    return pl.DataFrame(
        {
            "series_id": series,
            "period_end": ends,
            "simple_return": returns,
            "label": ["research_proxy"] * len(ends),
            "source": ["synthetic"] * len(ends),
            "available_at": stamps,
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


def _install_splice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured: dict[str, object], *, with_proxy: bool = True
) -> None:
    frames = {"prices": _splice_prices_frame(), "research_monthly": _splice_research_frame(with_proxy=with_proxy)}
    snapshot = SimpleNamespace(
        artifacts={
            Dataset.PRICES: SimpleNamespace(manifest_path="/m/prices.json"),
            Dataset.RESEARCH_MONTHLY: SimpleNamespace(manifest_path="/m/research.json"),
        }
    )
    monkeypatch.setattr(catalog_mod, "resolve_snapshot", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(
        catalog_mod, "load_snapshot_visible", lambda _snap, dataset, _ts: frames[str(dataset)]
    )
    monkeypatch.setattr(
        pension_campaign_mod,
        "run_pension_campaign",
        lambda *_a, **_k: SimpleNamespace(
            summaries=[
                SimpleNamespace(arm_id="sp500_100", horizon_months=12, median_wealth_ratio=1.0),
                SimpleNamespace(arm_id="sp500_100", horizon_months=24, median_wealth_ratio=1.0),
            ]
        ),
    )
    monkeypatch.setattr(campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)

    def write(
        _settings: DataSettings,
        *,
        experiment: str,
        kind: object,
        run_id: str,
        payload: dict[str, object],
        markdown: str | None = None,
        written_at: object = None,
    ) -> SimpleNamespace:
        captured["experiment"] = experiment
        captured["kind"] = str(kind)
        captured["run_id"] = run_id
        captured["payload"] = payload
        captured["markdown"] = markdown
        return SimpleNamespace(json_path=tmp_path / f"decision_{run_id}.json")

    monkeypatch.setattr(result_store_mod, "write_result", write)


def test_pension_decision_spliced_sleeve_and_controls_reach_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late-listed sleeve, one control, and a reference surface in the payload."""
    captured: dict[str, object] = {}
    _install_splice(tmp_path, monkeypatch, captured)
    code = campaign_mod.run_pension_decision_command(
        config_path=_splice_config(tmp_path),
        settings=DataSettings(data_root=str(tmp_path / "data")),
        seed=11,
    )
    assert code == 0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["dominance_reference_id"] == "spy_only"
    assert payload["modern_splices"] == [
        {
            "sleeve": "SCHD",
            "proxy_first_month": "2000-01-31",
            "proxy_last_month": "2000-12-31",
            "etf_first_month": "2001-01-31",
            "proxy_weights": {"ff_dp_hi30_monthly": 1.0},
        }
    ]
    assert set(payload["control_scores"]) == {"world"}  # type: ignore[arg-type]
    assert isinstance(captured["markdown"], str)
    assert "도미넌스 가드" in captured["markdown"]
    assert "컨트롤" in captured["markdown"]


def test_pension_decision_missing_proxy_series_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proxy series absent from the store fails the splice with no result file."""
    captured: dict[str, object] = {}
    _install_splice(tmp_path, monkeypatch, captured, with_proxy=False)
    code = campaign_mod.run_pension_decision_command(
        config_path=_splice_config(tmp_path),
        settings=DataSettings(data_root=str(tmp_path / "data")),
        seed=11,
    )
    assert code == 1
    assert "payload" not in captured
