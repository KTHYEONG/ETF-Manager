"""Invariant guards for the ISA household decision config."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.validation.isa_household_config import load_isa_household_spec
from src.validation.pension_decision_record import load_pension_decision_record

_REGISTERED = Path("configs/decision/isa.json")


def _document() -> dict:
    return json.loads(_REGISTERED.read_text(encoding="utf-8"))


def _write(tmp_path: Path, mutate) -> Path:
    path = tmp_path / "isa_household.json"
    path.write_text(json.dumps(mutate), encoding="utf-8")
    return path


def test_registered_config_loads() -> None:
    """The registered decision config carries six arms, four profiles, and the frozen plan."""
    spec = load_isa_household_spec(_REGISTERED)
    assert tuple(spec.arms) == (
        "hold",
        "roll3_pension_cap",
        "roll5_pension_all",
        "roll3_pension_all",
        "roll3_side",
        "no_isa",
    )
    assert spec.baseline_arm_id == "hold"
    assert tuple(profile.profile_id for profile in spec.profiles) == (
        "student_then_seomin",
        "student_then_general_mid",
        "student_then_general_high",
        "student_then_no_capacity",
    )
    assert spec.isa_budgets_krw == (6_000_000, 12_000_000, 20_000_000)
    assert spec.horizons_years == (20, 30)


def test_arm_order_preserved() -> None:
    """Arms iterate in JSON object order for deterministic tie-breaks."""
    spec = load_isa_household_spec(_REGISTERED)
    assert tuple(spec.arms) == tuple(_document()["arms"])


def test_unknown_baseline_rejected(tmp_path: Path) -> None:
    """A baseline id outside the arms fails closed."""
    document = _document()
    document["baseline_arm_id"] = "missing"
    with pytest.raises(ValueError, match="baseline_arm_id"):
        load_isa_household_spec(_write(tmp_path, document))


def test_budget_above_isa_limit_rejected(tmp_path: Path) -> None:
    """A budget above the 20M annual limit fails closed."""
    document = _document()
    document["isa_budgets_krw"] = [20_000_001]
    with pytest.raises(ValueError, match="annual limit"):
        load_isa_household_spec(_write(tmp_path, document))


def test_duplicate_key_rejected(tmp_path: Path) -> None:
    """A document repeating a key fails closed."""
    raw = _REGISTERED.read_text(encoding="utf-8").replace('"name": "isa_household_v1"', '"name": "a", "name": "b"', 1)
    path = tmp_path / "isa_household.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_isa_household_spec(path)


def test_drawing_horizon_overlap_rejected(tmp_path: Path) -> None:
    """The primary drawing horizon must not repeat in the sensitivity list."""
    document = _document()
    document["sensitivity_drawing_years"] = [10, 20]
    with pytest.raises(ValueError, match="sensitivity_drawing_years"):
        load_isa_household_spec(_write(tmp_path, document))


def test_capacity_absorbs_full_credits() -> None:
    """Employed-phase capacity exactly covers the ordinary plus transfer credit bases."""
    spec = load_isa_household_spec(_REGISTERED)
    for profile in spec.profiles:
        employed = profile.phases[1]
        if profile.profile_id == "student_then_no_capacity":
            assert employed.remaining_national_tax_krw == 0
            assert employed.remaining_local_tax_krw == 0
            continue
        rate = 0.12 if profile.profile_id == "student_then_general_high" else 0.15
        expected_national = int(6_000_000 * rate) + int(3_000_000 * rate)
        assert employed.remaining_national_tax_krw == expected_national
        assert employed.remaining_local_tax_krw == expected_national // 10


def test_paths_resolve() -> None:
    """Every referenced file exists and the pension record holds the frozen incumbent."""
    spec = load_isa_household_spec(_REGISTERED)
    for field in (
        "pension_decision_config_path",
        "pension_record_path",
        "isa_tax_regime_path",
        "pension_tax_regime_path",
        "overseas_tax_regime_path",
    ):
        assert Path(getattr(spec, field)).is_file()
    record = load_pension_decision_record(spec.pension_record_path)
    assert record.incumbent_id == "schd10_qqq90"


def test_spec_rejects_malformed_fields(tmp_path: Path) -> None:
    """Key-set, arm, profile, and range violations fail closed."""
    document = _document()
    del document["notes"]
    with pytest.raises(ValueError, match="unexpected keys"):
        load_isa_household_spec(_write(tmp_path, document))
    renamed = _document()
    renamed["arms"]["hold"]["arm_id"] = "other"
    with pytest.raises(ValueError, match="differs from its object key"):
        load_isa_household_spec(_write(tmp_path, renamed))
    duplicated = _document()
    duplicated["profiles"].append(dict(duplicated["profiles"][0]))
    with pytest.raises(ValueError, match="unique"):
        load_isa_household_spec(_write(tmp_path, duplicated))
    short_horizon = _document()
    short_horizon["horizons_years"] = [2]
    with pytest.raises(ValueError, match="below 3 years"):
        load_isa_household_spec(_write(tmp_path, short_horizon))
    wide_band = _document()
    wide_band["equivalence_band"] = 0.06
    with pytest.raises(ValueError, match="equivalence_band"):
        load_isa_household_spec(_write(tmp_path, wide_band))
    bad_share = _document()
    bad_share["min_bootstrap_win_share"] = 1.0
    with pytest.raises(ValueError, match="min_bootstrap_win_share"):
        load_isa_household_spec(_write(tmp_path, bad_share))
    missing_file = _document()
    missing_file["pension_record_path"] = "records/does_not_exist.json"
    with pytest.raises(ValueError, match="does not exist"):
        load_isa_household_spec(_write(tmp_path, missing_file))
    assert date(2026, 9, 26) == load_isa_household_spec(_REGISTERED).lineage.first_test_date


def test_config_rejects_malformed_fields(tmp_path: Path) -> None:
    """Every malformed scalar, entry, and structural variant fails closed."""
    cases = [
        ("name", "", "non-blank string"),
        ("equivalence_band", "wide", "finite number"),
        ("step_months", 0, "positive integer"),
        ("pension_annual_krw", "6000000", "integer amount"),
        ("pension_annual_krw", 0, "positive integer"),
        ("plan_start_year", "2027", "integer year"),
        ("horizons_years", {}, "non-empty array"),
        ("isa_budgets_krw", {}, "non-empty array"),
        ("sensitivity_drawing_years", {}, "an array"),
        ("arms", [], "non-empty object"),
        ("profiles", {}, "non-empty array"),
        ("notes", 1, "must be a string"),
    ]
    for field, bad, pattern in cases:
        document = _document()
        document[field] = bad
        with pytest.raises(ValueError, match=pattern):
            load_isa_household_spec(_write(tmp_path, document))
    duplicated_horizons = _document()
    duplicated_horizons["horizons_years"] = [20, 20]
    with pytest.raises(ValueError, match="unique"):
        load_isa_household_spec(_write(tmp_path, duplicated_horizons))

    arm_cases = [
        ({"arm_id": "hold", "mode": "hold", "cycle_years": None, "extra": 1}, "unknown fields"),
        ({"arm_id": "hold", "mode": "hold"}, "missing field"),
        ({"arm_id": "hold", "mode": "hover", "cycle_years": None}, "mode is unknown"),
        ({"arm_id": "hold", "mode": "hold", "cycle_years": "3"}, "integer or null"),
        ({"arm_id": "hold", "mode": "hold", "cycle_years": 3}, "is invalid"),
    ]
    for bad_arm, pattern in arm_cases:
        document = _document()
        document["arms"]["hold"] = bad_arm
        with pytest.raises(ValueError, match=pattern):
            load_isa_household_spec(_write(tmp_path, document))
    not_object_arm = _document()
    not_object_arm["arms"]["hold"] = "hold"
    with pytest.raises(ValueError, match="must be an object"):
        load_isa_household_spec(_write(tmp_path, not_object_arm))

    base_phase = {
        "first_plan_year_offset": 0,
        "income_kind": "wage",
        "annual_income_krw": 5_000_000,
        "remaining_national_tax_krw": 0,
        "remaining_local_tax_krw": 0,
    }

    def _with_phases(phases) -> Path:
        document = _document()
        document["profiles"][0]["phases"] = phases
        return _write(tmp_path, document)

    with pytest.raises(ValueError, match="must be an object"):
        load_isa_household_spec(_with_phases(["x"]))
    unknown_field_phase = dict(base_phase)
    unknown_field_phase["extra"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        load_isa_household_spec(_with_phases([unknown_field_phase]))
    missing_field_phase = dict(base_phase)
    del missing_field_phase["income_kind"]
    with pytest.raises(ValueError, match="missing field"):
        load_isa_household_spec(_with_phases([missing_field_phase]))
    for bad_value, pattern in [("0", "must be an integer"), ("business", "unsupported"), ("1", "integer amount"), (-1, "nonnegative")]:
        mutated = dict(base_phase)
        if bad_value in ("0", "business"):
            mutated["first_plan_year_offset" if bad_value == "0" else "income_kind"] = bad_value
        else:
            mutated["annual_income_krw"] = bad_value
        with pytest.raises(ValueError, match=pattern):
            load_isa_household_spec(_with_phases([mutated]))
    second_negative = [base_phase, {**base_phase, "first_plan_year_offset": -1}]
    with pytest.raises(ValueError, match="is invalid"):
        load_isa_household_spec(_with_phases(second_negative))

    profile_cases = [
        (1, "must be an object"),
        ({"profile_id": "x", "extra": 1}, "unknown fields"),
        ({"profile_id": "x"}, "missing field"),
    ]
    for bad_profile, pattern in profile_cases:
        document = _document()
        document["profiles"][0] = bad_profile
        with pytest.raises(ValueError, match=pattern):
            load_isa_household_spec(_write(tmp_path, document))
    bad_class = _document()
    bad_class["profiles"][0]["initial_isa_tax_class"] = "rich"
    with pytest.raises(ValueError, match="is unknown"):
        load_isa_household_spec(_write(tmp_path, bad_class))
    empty_phases = _document()
    empty_phases["profiles"][0]["phases"] = []
    with pytest.raises(ValueError, match="non-empty array"):
        load_isa_household_spec(_write(tmp_path, empty_phases))
    shifted = _document()
    shifted["profiles"][0]["phases"][0]["first_plan_year_offset"] = 1
    with pytest.raises(ValueError, match="is invalid"):
        load_isa_household_spec(_write(tmp_path, shifted))

    lineage_not_object = _document()
    lineage_not_object["lineage"] = []
    with pytest.raises(ValueError, match="must be an object"):
        load_isa_household_spec(_write(tmp_path, lineage_not_object))
    lineage_extra = _document()
    lineage_extra["lineage"]["extra"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        load_isa_household_spec(_write(tmp_path, lineage_extra))
    lineage_missing = _document()
    del lineage_missing["lineage"]["related_trials"]
    with pytest.raises(ValueError, match="missing field"):
        load_isa_household_spec(_write(tmp_path, lineage_missing))
    lineage_negative = _document()
    lineage_negative["lineage"]["related_trial_count"] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        load_isa_household_spec(_write(tmp_path, lineage_negative))
    bad_trials = _document()
    bad_trials["lineage"]["related_trials"] = "x"
    with pytest.raises(ValueError, match="must be an array"):
        load_isa_household_spec(_write(tmp_path, bad_trials))
    bad_disclosure = _document()
    bad_disclosure["lineage"]["post_hoc_disclosure"] = 1
    with pytest.raises(ValueError, match="must be a string"):
        load_isa_household_spec(_write(tmp_path, bad_disclosure))
    bad_date = _document()
    bad_date["lineage"]["first_test_date"] = "not-a-date"
    with pytest.raises(ValueError, match="ISO date"):
        load_isa_household_spec(_write(tmp_path, bad_date))
    with pytest.raises(ValueError, match="unreadable"):
        load_isa_household_spec(tmp_path / "does_not_exist.json")
    not_object = tmp_path / "array.json"
    not_object.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_isa_household_spec(not_object)


def test_malformed_json_unreadable(tmp_path: Path) -> None:
    """Truncated JSON bytes fail closed as unreadable."""
    path = tmp_path / "broken.json"
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_isa_household_spec(path)


def test_committed_isa_records_path_resolves() -> None:
    """An ISA config citing a frozen records/ pension record loads with the file on disk."""
    spec = load_isa_household_spec(_REGISTERED)
    assert Path(spec.pension_record_path).is_file()


def test_missing_cited_record_path_names_field(tmp_path: Path) -> None:
    """An ISA config citing a file that exists nowhere fails closed naming the field."""
    document = _document()
    document["pension_record_path"] = "records/does_not_exist.json"
    with pytest.raises(ValueError, match="pension_record_path"):
        load_isa_household_spec(_write(tmp_path, document))
