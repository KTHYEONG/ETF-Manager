"""Invariant guards for the frozen pension decision record and annual review."""

from __future__ import annotations

import calendar as _calendar
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.sim.pension_monthly import WeightSchedule
from src.validation.pension_decision import PensionDecisionReport
from src.validation.pension_decision_config import PensionDecisionSpec
from src.validation.pension_decision_record import (
    evaluate_pension_review,
    freeze_pension_decision,
    load_pension_decision_record,
)

_BENCH = "spy100_qqq0"
_INC = "spy0_qqq100"


def _mix(spy: float, qqq: float) -> WeightSchedule:
    return WeightSchedule(
        start_weights={"SPY": spy, "QQQ": qqq},
        end_weights={"SPY": spy, "QQQ": qqq},
        glide_years=0,
    )


def _spec(cutoff: date = date(2020, 1, 31)) -> PensionDecisionSpec:
    return PensionDecisionSpec(
        name="probe",
        benchmark_id=_BENCH,
        candidates={_BENCH: _mix(1.0, 0.0), _INC: _mix(0.0, 1.0)},
        neighbors={_BENCH: (_INC,), _INC: (_BENCH,)},
        century_series={"SPY": "ff_mkt_monthly", "QQQ": "ff_hitec_monthly"},
        modern_start=date(2000, 1, 31),
        modern_end=cutoff,
        century_start=date(2000, 1, 31),
        century_end=date(2019, 12, 31),
        horizons_years=(2,),
        step_months=12,
        pre_retirement_months=12,
        primary_gamma=1.0,
        sensitivity_gammas=(),
        equivalence_band=0.005,
        min_bootstrap_win_share=0.6,
        bootstrap_paths=10,
        bootstrap_block_months=3,
        annual_drag_by_sleeve={},
        tax_crosscheck_campaign_path="configs/research/pension_campaign_v2_dotcom.json",
        tax_crosscheck_arm_map={},
        review_every_months=12,
        lineage={"related_trial_count": 1},
        modern_splices={},
        sleeve_products={},
        dominance_reference_id=None,
        controls={},
        realized_horizons_years=(),
    )


def _report(status: str = "ADOPT_CANDIDATE", selected: str | None = _INC) -> PensionDecisionReport:
    return PensionDecisionReport(
        name="probe",
        status=status,  # type: ignore[arg-type]
        selected_id=selected,
        equivalent_ids=(_BENCH, _INC),
        reasons=(),
        scores=(),
        robust_scores={_BENCH: 1.0, _INC: 1.01},
        sensitivity_robust_scores={},
        bootstrap_win_share={_BENCH: 0.4, _INC: 0.8},
        tax_rank_agreement=True,
        manifest_hashes={"prices": "p", "research_monthly": "r"},
        trial_count=3,
        dominance_min_ratio={},
        dominance_min_ratio_by_tier={},
        guard_excluded_ids=(),
        control_scores={},
        control_vs_reference={},
    )


def _freeze_kwargs() -> dict[str, object]:
    return {
        "frozen_at": datetime(2021, 2, 1, tzinfo=UTC),
        "git_commit": "0" * 40,
        "config_sha256": "a" * 64,
    }


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


def _prices(months: list[date], spy: float, qqq: float) -> pl.DataFrame:
    tickers, dates, closes = [], [], []
    for ticker, rate in (("SPY", spy), ("QQQ", qqq)):
        for index, day in enumerate(months):
            tickers.append(ticker)
            dates.append(day)
            closes.append(100.0 * (1.0 + rate) ** index)
    stamps = [datetime(2019, 1, 1, tzinfo=UTC)] * len(dates)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "adjusted_close": closes,
            "available_at": stamps,
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def test_freeze_writes_once(tmp_path: Path) -> None:
    """Freezing twice to the same directory fails and leaves bytes unchanged."""
    out = tmp_path / "records"
    first = freeze_pension_decision(_report(), _spec(), output_dir=out, **_freeze_kwargs())  # type: ignore[arg-type]
    before = first.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        freeze_pension_decision(_report(), _spec(), output_dir=out, **_freeze_kwargs())  # type: ignore[arg-type]
    assert first.read_bytes() == before


def test_no_decision_cannot_be_frozen(tmp_path: Path) -> None:
    """A NO_DECISION report is never frozen."""
    with pytest.raises(ValueError, match="NO_DECISION"):
        freeze_pension_decision(
            _report(status="NO_DECISION", selected=None),
            _spec(),
            output_dir=tmp_path,
            **_freeze_kwargs(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        freeze_pension_decision(
            _report(),
            _spec(),
            output_dir=tmp_path,
            frozen_at=datetime(2021, 2, 1),
            git_commit="0" * 40,
            config_sha256="a" * 64,
        )


def test_record_round_trips(tmp_path: Path) -> None:
    """Every frozen field survives a load."""
    path = freeze_pension_decision(
        _report(), _spec(), output_dir=tmp_path, previous_record_id="prev-1", **_freeze_kwargs()  # type: ignore[arg-type]
    )
    record = load_pension_decision_record(path)
    assert record.record_id == path.stem
    assert record.frozen_at == datetime(2021, 2, 1, tzinfo=UTC)
    assert record.git_commit == "0" * 40
    assert record.config_sha256 == "a" * 64
    assert record.manifest_hashes == {"prices": "p", "research_monthly": "r"}
    assert record.seen_history_cutoff == date(2020, 1, 31)
    assert record.status == "ADOPT_CANDIDATE"
    assert record.incumbent_id == _INC
    assert record.incumbent_schedule == _mix(0.0, 1.0)
    assert record.equivalent_ids == (_BENCH, _INC)
    assert record.benchmark_id == _BENCH
    assert record.benchmark_schedule == _mix(1.0, 0.0)
    assert record.review_every_months == 12
    assert record.previous_record_id == "prev-1"


def test_record_rejects_malformed_content(tmp_path: Path) -> None:
    """Malformed record bytes fail closed on load."""
    path = freeze_pension_decision(_report(), _spec(), output_dir=tmp_path, **_freeze_kwargs())  # type: ignore[arg-type]
    document = json.loads(path.read_text(encoding="utf-8"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**document, "status": "NO_DECISION"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not freezable"):
        load_pension_decision_record(bad)
    missing = tmp_path / "missing.json"
    trimmed = dict(document)
    del trimmed["benchmark_id"]
    missing.write_text(json.dumps(trimmed), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_pension_decision_record(missing)
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({**document, "surprise": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_pension_decision_record(extra)


def _frozen_record(cutoff: date = date(2020, 1, 31), incumbent: str = _INC) -> object:
    spec = _spec(cutoff)
    schedule = spec.candidates[incumbent]
    bench = spec.candidates[_BENCH]
    from src.validation.pension_decision_record import PensionDecisionRecord

    return PensionDecisionRecord(
        record_id="probe-record",
        frozen_at=datetime(2020, 2, 1, tzinfo=UTC),
        git_commit="0" * 40,
        config_sha256="a" * 64,
        manifest_hashes={},
        seen_history_cutoff=cutoff,
        status="ADOPT_CANDIDATE",
        incumbent_id=incumbent,
        incumbent_schedule=schedule,
        equivalent_ids=(_BENCH, _INC),
        benchmark_id=_BENCH,
        benchmark_schedule=bench,
        review_every_months=12,
        previous_record_id=None,
    )


def test_review_insufficient_data_before_first_month() -> None:
    """as_of inside the cutoff month reports INSUFFICIENT_DATA."""
    record = _frozen_record()
    frame = _prices(_months(date(2020, 1, 31), 14), 0.005, 0.005)
    status = evaluate_pension_review(record, frame, datetime(2020, 1, 31, 23, 59, tzinfo=UTC))  # type: ignore[arg-type]
    assert status.state == "INSUFFICIENT_DATA"
    assert status.months_observed == 0
    assert status.incumbent_over_benchmark is None


def test_review_falls_due_after_cadence() -> None:
    """Twelve complete months after the cutoff report REVIEW_DUE."""
    record = _frozen_record()
    frame = _prices(_months(date(2020, 1, 31), 14), 0.005, 0.005)
    status = evaluate_pension_review(record, frame, datetime(2021, 1, 31, 23, 59, tzinfo=UTC))  # type: ignore[arg-type]
    assert status.months_observed == 12
    assert status.state == "REVIEW_DUE"


def test_review_underperformance_never_forces_switch() -> None:
    """A trailing incumbent before the cadence stays HOLD with the ratio reported."""
    record = _frozen_record()
    months = _months(date(2020, 1, 31), 8)
    tickers, dates, closes = [], [], []
    for index, day in enumerate(months):
        tickers += ["SPY", "QQQ"]
        dates += [day, day]
        closes += [100.0 * 1.005**index, 100.0 * 0.95**index]
    frame = pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "adjusted_close": closes,
            "available_at": [datetime(2019, 1, 1, tzinfo=UTC)] * len(dates),
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )
    status = evaluate_pension_review(record, frame, datetime(2020, 7, 31, 23, 59, tzinfo=UTC))  # type: ignore[arg-type]
    assert status.state == "HOLD"
    assert status.incumbent_over_benchmark is not None
    assert status.incumbent_over_benchmark < 0.9


def test_review_historical_months_are_invisible() -> None:
    """Corrupting prices on or before the cutoff changes nothing."""
    record = _frozen_record()
    months = _months(date(2019, 11, 30), 16)
    base = _prices(months, 0.005, 0.006)
    as_of = datetime(2020, 7, 31, 23, 59, tzinfo=UTC)
    first = evaluate_pension_review(record, base, as_of)  # type: ignore[arg-type]
    corrupted = base.with_columns(
        pl.when(pl.col("date") <= date(2020, 1, 31))
        .then(pl.col("adjusted_close") * 100.0)
        .otherwise(pl.col("adjusted_close"))
        .alias("adjusted_close")
    )
    second = evaluate_pension_review(record, corrupted, as_of)  # type: ignore[arg-type]
    assert second == first


def test_review_benchmark_incumbent_ratio_is_one() -> None:
    """A record keeping the benchmark reports exactly 1.0."""
    record = _frozen_record(incumbent=_BENCH)
    frame = _prices(_months(date(2020, 1, 31), 14), 0.005, -0.02)
    status = evaluate_pension_review(record, frame, datetime(2020, 7, 31, 23, 59, tzinfo=UTC))  # type: ignore[arg-type]
    assert status.state == "HOLD"
    assert status.incumbent_over_benchmark == 1.0


def test_review_rejects_naive_or_preceding_as_of() -> None:
    """Naive or pre-cutoff review instants fail closed."""
    record = _frozen_record()
    frame = _prices(_months(date(2020, 1, 31), 14), 0.005, 0.005)
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_pension_review(record, frame, datetime(2020, 7, 31))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="precedes cutoff"):
        evaluate_pension_review(record, frame, datetime(2019, 12, 31, 23, 59, tzinfo=UTC))  # type: ignore[arg-type]


def test_freeze_rejects_blank_and_unknown_selection(tmp_path: Path) -> None:
    """Blank identities and unknown selections fail closed."""
    with pytest.raises(ValueError, match="git_commit"):
        freeze_pension_decision(
            _report(), _spec(), output_dir=tmp_path,
            frozen_at=datetime(2021, 2, 1, tzinfo=UTC), git_commit="  ", config_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="no selected candidate"):
        freeze_pension_decision(
            _report(status="ADOPT_CANDIDATE", selected=None), _spec(),
            output_dir=tmp_path, **_freeze_kwargs(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="not among candidates"):
        freeze_pension_decision(
            _report(status="ADOPT_CANDIDATE", selected="ghost"), _spec(),
            output_dir=tmp_path, **_freeze_kwargs(),  # type: ignore[arg-type]
        )


def test_record_rejects_structural_defects(tmp_path: Path) -> None:
    """Duplicate keys, bad timestamps, schedules, and containers fail closed."""
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_pension_decision_record(tmp_path / "absent.json")
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_pension_decision_record(array)
    dupes = tmp_path / "dupes.json"
    dupes.write_text('{"record_id": "a", "record_id": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_pension_decision_record(dupes)

    path = freeze_pension_decision(_report(), _spec(), output_dir=tmp_path / "ok", **_freeze_kwargs())  # type: ignore[arg-type]
    document = json.loads(path.read_text(encoding="utf-8"))

    def _bad(name: str, mutated: tuple[str, object], match: str) -> None:
        bad = tmp_path / f"{name}.json"
        altered = dict(document)
        altered["record_id"] = name
        altered[mutated[0]] = mutated[1]
        bad.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_pension_decision_record(bad)

    _bad("frozen", ("frozen_at", "not-a-datetime"), "ISO datetime")
    _bad("naive", ("frozen_at", "2021-02-01T00:00:00"), "timezone-aware")
    _bad("cutoff", ("seen_history_cutoff", "not-a-date"), "ISO date")
    _bad("midmonth", ("seen_history_cutoff", "2020-01-15"), "month-end")
    _bad("manifest", ("manifest_hashes", ["prices"]), "must be an object")
    _bad("equiv", ("equivalent_ids", {"a": 1}), "must be an array")
    _bad("cadence", ("review_every_months", 0), "positive integer")
    _bad("sched-type", ("incumbent_schedule", ["SPY"]), "must be an object")
    _bad("sched-bad", ("incumbent_schedule", {"start_weights": {"SPY": 2.0}, "end_weights": {"SPY": 1.0}, "glide_years": 0}), "valid weight schedule")


def test_review_rejects_bad_price_cells() -> None:
    """Non-numeric, non-positive, or missing price columns fail closed."""
    record = _frozen_record()
    months = _months(date(2020, 1, 31), 6)
    frame = _prices(months, 0.005, 0.005)
    as_of = datetime(2020, 5, 31, 23, 59, tzinfo=UTC)
    with pytest.raises(ValueError, match="required column"):
        evaluate_pension_review(record, frame.drop("ticker"), as_of)  # type: ignore[arg-type]
    broken = frame.with_columns(pl.lit(None).alias("adjusted_close"))
    with pytest.raises(ValueError, match="must be numeric"):
        evaluate_pension_review(record, broken, as_of)  # type: ignore[arg-type]
    negative = frame.with_columns(
        pl.when(pl.col("ticker") == "SPY")
        .then(-5.0)
        .otherwise(pl.col("adjusted_close"))
        .alias("adjusted_close")
    )
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_pension_review(record, negative, as_of)  # type: ignore[arg-type]


def test_review_without_visible_post_cutoff_months_is_insufficient() -> None:
    """No visible month after the cutoff reports INSUFFICIENT_DATA."""
    record = _frozen_record()
    frame = _prices(_months(date(2019, 10, 31), 3), 0.005, 0.005)
    status = evaluate_pension_review(record, frame, datetime(2020, 7, 31, 23, 59, tzinfo=UTC))  # type: ignore[arg-type]
    assert status.state == "INSUFFICIENT_DATA"
    assert status.months_observed == 0
    assert status.incumbent_over_benchmark is None
