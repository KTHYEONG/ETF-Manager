"""Invariant guards for the ISA household operating-policy decision."""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.sim.isa_household import HouseholdProfile, IncomePhase, IsaArm, IsaOperatingMode
from src.sim.isa_tax import IsaTaxClass
from src.sim.pension_monthly import MonthlyReturnPanel, WeightSchedule
from src.validation.isa_household_config import IsaHouseholdLineage, IsaHouseholdSpec
from src.validation.isa_household_decision import (
    IsaHouseholdReport,
    evaluate_isa_household,
    freeze_isa_household_decision,
)
from src.validation.pension_decision_config import PensionDecisionSpec
from src.validation.pension_decision_record import PensionDecisionRecord

_INC = WeightSchedule(
    start_weights={"QQQ": 0.9, "SCHD": 0.1},
    end_weights={"QQQ": 0.9, "SCHD": 0.1},
    glide_years=0,
)
_ALT = WeightSchedule(
    start_weights={"QQQ": 1.0},
    end_weights={"QQQ": 1.0},
    glide_years=0,
)


def _months(start: date, count: int) -> tuple[date, ...]:
    out: list[date] = []
    year, month = start.year, start.month
    for _ in range(count):
        out.append(date(year, month, calendar.monthrange(year, month)[1]))
        month += 1
        if month > 12:
            year += 1
            month = 1
    return tuple(out)


def _constant_panel(months: tuple[date, ...], tier: str) -> MonthlyReturnPanel:
    count = len(months)
    return MonthlyReturnPanel(
        tier=tier,
        months=months,
        returns={"QQQ": tuple([0.008] * count), "SCHD": tuple([0.004] * count)},
    )


_MODERN = _constant_panel(_months(date(2015, 1, 31), 96), "modern")
_CENTURY = _constant_panel(_months(date(1990, 1, 31), 144), "century")
_LOSS_MIX_CENTURY = MonthlyReturnPanel(
    tier="century",
    months=_months(date(1990, 1, 31), 144),
    returns={
        "QQQ": tuple([-0.02] * 72 + [0.025] * 72),
        "SCHD": tuple([-0.01] * 72 + [0.0125] * 72),
    },
)

_HOLD = IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None)
_ALL = IsaArm(arm_id="all", mode=IsaOperatingMode.ROLL_TO_PENSION_ALL, cycle_years=3)
_SIDE = IsaArm(arm_id="side", mode=IsaOperatingMode.ROLL_TO_SIDE, cycle_years=3)
_NO_ISA = IsaArm(arm_id="no_isa", mode=IsaOperatingMode.NO_ISA, cycle_years=None)


def _phase(*, income: int = 45_000_000, national: int = 10_000_000, local: int = 10_000_000) -> IncomePhase:
    return IncomePhase(
        first_plan_year_offset=0,
        income_kind="wage",
        annual_income_krw=income,
        remaining_national_tax_krw=national,
        remaining_local_tax_krw=local,
    )


def _profile(
    profile_id: str, *, income: int = 45_000_000, national: int = 10_000_000, local: int = 10_000_000
) -> HouseholdProfile:
    return HouseholdProfile(
        profile_id=profile_id,
        birth_date=date(1990, 1, 1),
        pension_account_open_date=date(2020, 1, 1),
        initial_isa_tax_class=IsaTaxClass.GENERAL,
        phases=(_phase(income=income, national=national, local=local),),
    )


def _decision_spec(**overrides) -> PensionDecisionSpec:
    fields: dict = {
        "name": "probe",
        "benchmark_id": "inc",
        "candidates": {"inc": _INC, "alt": _ALT},
        "neighbors": {},
        "century_series": {"QQQ": "x", "SCHD": "y"},
        "modern_start": date(2015, 1, 31),
        "modern_end": date(2022, 12, 31),
        "century_start": date(1990, 1, 31),
        "century_end": date(2001, 12, 31),
        "horizons_years": (6,),
        "step_months": 12,
        "pre_retirement_months": 0,
        "primary_gamma": 1.0,
        "sensitivity_gammas": (),
        "equivalence_band": 0.005,
        "min_bootstrap_win_share": 0.6,
        "bootstrap_paths": 50,
        "bootstrap_block_months": 12,
        "annual_drag_by_sleeve": {},
        "tax_crosscheck_campaign_path": "x",
        "tax_crosscheck_arm_map": {},
        "review_every_months": 12,
        "lineage": {},
        "modern_splices": {},
        "sleeve_products": {},
        "dominance_reference_id": None,
        "controls": {},
        "realized_horizons_years": (),
    }
    fields.update(overrides)
    return PensionDecisionSpec(**fields)


def _record() -> PensionDecisionRecord:
    return PensionDecisionRecord(
        record_id="probe",
        frozen_at=datetime(2026, 1, 1, tzinfo=UTC),
        git_commit="a" * 40,
        config_sha256="b" * 64,
        manifest_hashes={},
        seen_history_cutoff=date(2022, 12, 31),
        status="ADOPT_CANDIDATE",
        incumbent_id="inc",
        incumbent_schedule=_INC,
        equivalent_ids=(),
        benchmark_id="inc",
        benchmark_schedule=_INC,
        review_every_months=12,
        previous_record_id=None,
    )


def _spec(
    arms: dict[str, IsaArm],
    profiles: list[HouseholdProfile],
    *,
    baseline: str = "hold",
    band: float = 0.005,
    win_share: float = 0.6,
    drawing: int = 10,
    sensitivity: tuple[int, ...] = (20,),
    budget: int = 20_000_000,
    horizons: tuple[int, ...] = (6,),
) -> IsaHouseholdSpec:
    return IsaHouseholdSpec(
        name="probe",
        pension_decision_config_path="",
        pension_record_path="",
        isa_tax_regime_path="",
        pension_tax_regime_path="",
        overseas_tax_regime_path="",
        plan_start_year=2027,
        horizons_years=horizons,
        step_months=12,
        pension_annual_krw=6_000_000,
        isa_budgets_krw=(budget,),
        annuity_drawing_years=drawing,
        sensitivity_drawing_years=sensitivity,
        baseline_arm_id=baseline,
        arms=dict(arms),
        profiles=tuple(profiles),
        equivalence_band=band,
        min_bootstrap_win_share=win_share,
        bootstrap_paths=50,
        bootstrap_block_months=12,
        lineage=IsaHouseholdLineage(0, (), date(2026, 9, 26), ""),
        notes="",
    )


def _regimes():
    from pathlib import Path as _Path

    from src.sim.isa_tax import load_isa_tax_regime
    from src.sim.pension_tax import load_pension_tax_regime
    from src.sim.tax import load_tax_regime

    return (
        load_isa_tax_regime(_Path("configs/tax/kr_isa_2026.json")),
        load_pension_tax_regime(_Path("configs/tax/kr_pension_2026.json")),
        load_tax_regime(_Path("configs/tax/kr_overseas_equity.json")),
    )


def _run(
    spec: IsaHouseholdSpec,
    decision_spec: PensionDecisionSpec | None = None,
    *,
    century: MonthlyReturnPanel = _CENTURY,
    seed: int = 7,
) -> IsaHouseholdReport:
    isa_regime, pension_regime, overseas_regime = _regimes()
    return evaluate_isa_household(
        spec,
        decision_spec or _decision_spec(),
        _record(),
        _MODERN,
        century,
        isa_regime=isa_regime,
        pension_regime=pension_regime,
        overseas_regime=overseas_regime,
        seed=seed,
    )


def test_baseline_scores_exactly_one() -> None:
    """Paired baseline ratios are identically one, so its robust score is 1.0."""
    report = _run(_spec({"hold": _HOLD, "all": _ALL}, [_profile("ample")]))
    baseline_cells = [cell for cell in report.cells if cell.arm_id == "hold"]
    assert baseline_cells
    assert all(cell.ce_ratio == 1.0 for cell in baseline_cells)
    assert report.decisions[0].robust_scores["hold"] == 1.0


def test_tie_break_prefers_registered_order() -> None:
    """Two identical policies tie; the first-registered id wins and both stay equivalent."""
    twin = IsaArm(arm_id="hold_twin", mode=IsaOperatingMode.HOLD, cycle_years=None)
    report = _run(_spec({"hold": _HOLD, "hold_twin": twin}, [_profile("ample")]))
    decision = report.decisions[0]
    assert decision.selected_arm_id == "hold"
    assert decision.equivalent_arm_ids == ("hold", "hold_twin")


def test_worst_profile_governs() -> None:
    """A policy winning one life path but losing another by more than the band is rejected."""
    report = _run(
        _spec(
            {"hold": _HOLD, "all": _ALL},
            [_profile("ample"), _profile("zero", income=30_000_000, national=0, local=0)],
        )
    )
    decision = report.decisions[0]
    assert decision.selected_arm_id != "all"
    assert decision.per_profile_best["ample"] == "all"


def test_keep_baseline_needs_no_bootstrap() -> None:
    """Keeping the baseline skips the bootstrap and reports no win share."""
    report = _run(_spec({"hold": _HOLD, "no_isa": _NO_ISA}, [_profile("ample")]))
    decision = report.decisions[0]
    assert decision.status == "KEEP_BASELINE"
    assert decision.bootstrap_win_share is None


def test_bootstrap_weak_demotes() -> None:
    """A robust winner losing too many bootstrap paths is demoted with a reason."""
    report = _run(
        _spec(
            {"hold": _HOLD, "all": _ALL},
            [_profile("zero", income=30_000_000, national=0, local=0)],
            band=0.0001,
            win_share=0.99,
        ),
        century=_LOSS_MIX_CENTURY,
    )
    decision = report.decisions[0]
    assert decision.selected_arm_id == "all"
    assert decision.bootstrap_win_share is not None
    assert decision.bootstrap_win_share < 0.99
    assert decision.status == "BOOTSTRAP_WEAK"
    assert "BOOTSTRAP_WEAK" in decision.reasons


def test_exit_sensitivity_flagged() -> None:
    """A pension-heavy winner under long drawing horizons loses under a one-year exit."""
    report = _run(_spec({"hold": _HOLD, "all": _ALL}, [_profile("ample")], drawing=30, sensitivity=(1,)))
    decision = report.decisions[0]
    assert decision.status == "EXIT_SENSITIVE"
    assert decision.sensitivity_selected[1] != decision.selected_arm_id
    assert "EXIT_SENSITIVE" in decision.reasons


def test_reasons_list_every_failed_check() -> None:
    """When bootstrap and exit sensitivity both fail, status follows precedence but both reasons are kept."""
    report = _run(
        _spec({"hold": _HOLD, "all": _ALL}, [_profile("ample")], drawing=30, sensitivity=(1,), win_share=0.9999),
        century=_LOSS_MIX_CENTURY,
    )
    decision = report.decisions[0]
    assert decision.selected_arm_id == "all"
    assert decision.bootstrap_win_share is not None
    assert decision.bootstrap_win_share < 0.9999
    assert decision.status == "BOOTSTRAP_WEAK"
    assert decision.reasons[:2] == ("BOOTSTRAP_WEAK", "EXIT_SENSITIVE")


def test_holding_review_never_overrides() -> None:
    """A better alternative holding flags review without changing the stage-1 verdict."""
    report = _run(_spec({"hold": _HOLD, "all": _ALL}, [_profile("ample")]))
    decision = report.decisions[0]
    first_status = decision.status
    assert decision.holding_consistent is False
    assert "HOLDING_REVIEW" in decision.reasons
    assert decision.status == first_status


def test_infeasible_cell_skipped() -> None:
    """A horizon beyond the modern panel scores century cells only."""
    short_modern = _constant_panel(_months(date(2015, 1, 31), 96), "modern")
    long_century = _constant_panel(_months(date(1990, 1, 31), 144), "century")
    isa_regime, pension_regime, overseas_regime = _regimes()
    spec = _spec({"hold": _HOLD, "all": _ALL}, [_profile("ample")], horizons=(10,))
    report = evaluate_isa_household(
        spec,
        _decision_spec(horizons_years=(10,)),
        _record(),
        short_modern,
        long_century,
        isa_regime=isa_regime,
        pension_regime=pension_regime,
        overseas_regime=overseas_regime,
        seed=7,
    )
    assert report.cells
    assert all(cell.tier == "century" for cell in report.cells)


def test_deterministic_with_seed() -> None:
    """Identical inputs and seed produce equal reports."""
    spec = _spec({"hold": _HOLD, "all": _ALL, "side": _SIDE}, [_profile("ample")])
    assert _run(spec) == _run(spec)


def test_freeze_refuses_undecided(tmp_path: Path) -> None:
    """A weak-bootstrap verdict cannot be frozen and writes nothing."""
    report = _run(
        _spec(
            {"hold": _HOLD, "all": _ALL},
            [_profile("zero", income=30_000_000, national=0, local=0)],
            band=0.0001,
            win_share=0.99,
        ),
        century=_LOSS_MIX_CENTURY,
    )
    assert report.decisions[0].status == "BOOTSTRAP_WEAK"
    with pytest.raises(ValueError, match="not freezable"):
        freeze_isa_household_decision(
            report,
            output_dir=tmp_path,
            frozen_at=datetime(2026, 9, 26, tzinfo=UTC),
            git_commit="a" * 40,
            config_sha256="b" * 64,
            pension_record_id="pension_decision_v3__x",
            run_id="seed7",
        )
    assert list(tmp_path.iterdir()) == []


def test_freeze_writes_once(tmp_path: Path) -> None:
    """A decisive verdict freezes to JSON once; a second freeze is refused."""
    report = _run(_spec({"hold": _HOLD, "no_isa": _NO_ISA}, [_profile("ample")]))
    assert report.decisions[0].status == "KEEP_BASELINE"
    frozen_at = datetime(2026, 9, 26, tzinfo=UTC)
    path = freeze_isa_household_decision(
        report,
        output_dir=tmp_path,
        frozen_at=frozen_at,
        git_commit="a" * 40,
        config_sha256="b" * 64,
        pension_record_id="pension_decision_v3__x",
        run_id="seed7",
    )
    import json as _json

    document = _json.loads(path.read_text(encoding="utf-8"))
    assert document["decisions"][0]["selected_arm_id"] == "hold"
    with pytest.raises(ValueError, match="already exists"):
        freeze_isa_household_decision(
            report,
            output_dir=tmp_path,
            frozen_at=frozen_at,
            git_commit="a" * 40,
            config_sha256="b" * 64,
            pension_record_id="pension_decision_v3__x",
            run_id="seed7",
        )


def test_evaluation_rejects_unknown_incumbent_and_sleeves() -> None:
    """An incumbent outside the candidates or a panel missing sleeves fails closed."""
    isa_regime, pension_regime, overseas_regime = _regimes()
    spec = _spec({"hold": _HOLD}, [_profile("ample")])
    decision_spec = _decision_spec()
    bad_record = PensionDecisionRecord(
        record_id="probe",
        frozen_at=datetime(2026, 1, 1, tzinfo=UTC),
        git_commit="a" * 40,
        config_sha256="b" * 64,
        manifest_hashes={},
        seen_history_cutoff=date(2022, 12, 31),
        status="ADOPT_CANDIDATE",
        incumbent_id="ghost",
        incumbent_schedule=_INC,
        equivalent_ids=(),
        benchmark_id="inc",
        benchmark_schedule=_INC,
        review_every_months=12,
        previous_record_id=None,
    )
    with pytest.raises(ValueError, match="not among candidates"):
        evaluate_isa_household(
            spec,
            decision_spec,
            bad_record,
            _MODERN,
            _CENTURY,
            isa_regime=isa_regime,
            pension_regime=pension_regime,
            overseas_regime=overseas_regime,
            seed=7,
        )
    thin = MonthlyReturnPanel(tier="modern", months=_months(date(2015, 1, 31), 96), returns={"QQQ": tuple([0.01] * 96)})
    with pytest.raises(ValueError, match="lacks sleeves"):
        evaluate_isa_household(
            spec,
            decision_spec,
            _record(),
            thin,
            _CENTURY,
            isa_regime=isa_regime,
            pension_regime=pension_regime,
            overseas_regime=overseas_regime,
            seed=7,
        )


def test_bootstrap_horizon_beyond_century_rejected() -> None:
    """A max horizon longer than the century panel fails closed before simulating."""
    isa_regime, pension_regime, overseas_regime = _regimes()
    thin_century = _constant_panel(_months(date(1990, 1, 31), 60), "century")
    with pytest.raises(ValueError, match="bootstrap horizon"):
        evaluate_isa_household(
            _spec({"hold": _HOLD}, [_profile("ample")]),
            _decision_spec(),
            _record(),
            _MODERN,
            thin_century,
            isa_regime=isa_regime,
            pension_regime=pension_regime,
            overseas_regime=overseas_regime,
            seed=7,
        )


def test_no_feasible_cell_rejected() -> None:
    """Horizons beyond both panels leave no evidence and fail closed."""
    isa_regime, pension_regime, overseas_regime = _regimes()
    with pytest.raises(ValueError, match="no feasible tier-horizon cell"):
        evaluate_isa_household(
            _spec({"hold": _HOLD}, [_profile("ample")], horizons=(20,)),
            _decision_spec(),
            _record(),
            _MODERN,
            _CENTURY,
            isa_regime=isa_regime,
            pension_regime=pension_regime,
            overseas_regime=overseas_regime,
            seed=7,
        )


def test_markdown_summarizes_budgets() -> None:
    """The human summary carries one table per budget with verdicts and scores."""
    from src.validation.isa_household_decision import isa_household_markdown

    report = _run(_spec({"hold": _HOLD, "all": _ALL}, [_profile("ample")]))
    text = isa_household_markdown(report)
    assert report.name in text
    assert "20000000" in text
    assert "hold" in text
    assert "all" in text
    assert "상태" in text


def test_freeze_rejects_naive_instant_and_blank_identities(tmp_path: Path) -> None:
    """Naive freeze instants and blank lineage identities fail closed without writing."""
    report = _run(_spec({"hold": _HOLD, "no_isa": _NO_ISA}, [_profile("ample")]))
    naive = datetime(2026, 9, 26)
    aware = datetime(2026, 9, 26, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        freeze_isa_household_decision(
            report, output_dir=tmp_path, frozen_at=naive, git_commit="a" * 40,
            config_sha256="b" * 64, pension_record_id="r", run_id="s",
        )
    with pytest.raises(ValueError, match="git_commit"):
        freeze_isa_household_decision(
            report, output_dir=tmp_path, frozen_at=aware, git_commit="  ",
            config_sha256="b" * 64, pension_record_id="r", run_id="s",
        )
    with pytest.raises(ValueError, match="config_sha256"):
        freeze_isa_household_decision(
            report, output_dir=tmp_path, frozen_at=aware, git_commit="a" * 40,
            config_sha256="  ", pension_record_id="r", run_id="s",
        )
    assert list(tmp_path.iterdir()) == []
