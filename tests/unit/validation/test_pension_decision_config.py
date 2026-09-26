"""Invariant guards for the pension decision pre-registration loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.validation.pension_decision_config import load_pension_decision_spec

_REPO = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO / "experiments" / "pension_decision_v1.json"


def _document() -> dict[str, object]:
    return dict(json.loads(_CONFIG_PATH.read_text(encoding="utf-8")))


def _write(tmp_path: Path, document: dict[str, object]) -> str:
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_load_committed_decision_config(tmp_path: Path) -> None:
    """The committed config carries 12 candidates over the log-utility objective."""
    del tmp_path
    spec = load_pension_decision_spec(_CONFIG_PATH)
    assert spec.name == "pension_decision_v1"
    assert len(spec.candidates) == 12
    assert spec.benchmark_id == "spy100_qqq0"
    assert spec.primary_gamma == pytest.approx(1.0)
    assert "glide_qqq100_to_50" in spec.candidates
    assert "SOXX" not in "".join(spec.candidates)


def test_load_rejects_unknown_keys(tmp_path: Path) -> None:
    """Unknown top-level keys fail closed naming the field."""
    document = _document()
    document["unknown_field"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_unknown_neighbor_ids(tmp_path: Path) -> None:
    """Neighbors referencing unknown ids fail closed naming the field."""
    document = _document()
    neighbors = dict(document["neighbors"])  # type: ignore[arg-type]
    neighbors["spy100_qqq0"] = ["no_such_candidate"]
    document["neighbors"] = neighbors
    with pytest.raises(ValueError, match="neighbors"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_sleeve_without_century_series(tmp_path: Path) -> None:
    """A candidate sleeve without a century series fails closed naming the field."""
    document = _document()
    series = dict(document["century_series"])  # type: ignore[arg-type]
    del series["QQQ"]
    document["century_series"] = series
    with pytest.raises(ValueError, match="century series"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_benchmark_outside_candidates(tmp_path: Path) -> None:
    """A benchmark not among candidates fails closed naming the field."""
    document = _document()
    document["benchmark_id"] = "spy0_qqq0"
    with pytest.raises(ValueError, match="benchmark_id"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_win_share_above_one(tmp_path: Path) -> None:
    """A win share above 1 fails closed naming the field."""
    document = _document()
    document["min_bootstrap_win_share"] = 1.5
    with pytest.raises(ValueError, match="min_bootstrap_win_share"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_empty_candidates(tmp_path: Path) -> None:
    """Empty candidates fail closed."""
    document = _document()
    document["candidates"] = {}
    with pytest.raises(ValueError, match="candidates"):
        load_pension_decision_spec(_write(tmp_path, document))


def _mutated(base: dict[str, object], key: str, value: object) -> dict[str, object]:
    document = json.loads(json.dumps(base))
    assert isinstance(document, dict)
    document[key] = value
    return document


_CASES: list[tuple[str, str, object, str]] = [
    ("blank name", "name", "  ", "non-blank string"),
    ("non-finite gamma", "primary_gamma", float("inf"), "finite number"),
    ("zero step", "step_months", 0, "positive integer"),
    ("bad modern date", "modern_start", "31-01-2000", "ISO date"),
    ("missing campaign file", "tax_crosscheck_campaign_path", "no/such.json", "does not exist"),
    ("non-object schedule", "candidates", {"x": []}, "must be an object with"),
    ("schedule missing field", "candidates", {"x": {"start_weights": {"SPY": 1.0}, "end_weights": {"SPY": 1.0}}}, "missing field"),
    ("schedule unknown field", "candidates", {"x": {"start_weights": {"SPY": 1.0}, "end_weights": {"SPY": 1.0}, "glide_years": 0, "extra": 1}}, "unknown fields"),
    ("negative glide", "candidates", {"x": {"start_weights": {"SPY": 1.0}, "end_weights": {"SPY": 1.0}, "glide_years": -1}}, "glide_years"),
    ("bad schedule weights", "candidates", {"x": {"start_weights": {"SPY": 1.0}, "end_weights": {"QQQ": 1.0}, "glide_years": 0}}, "not a valid weight schedule"),
    ("empty weights", "candidates", {"x": {"start_weights": {}, "end_weights": {"SPY": 1.0}, "glide_years": 0}}, "non-empty object"),
    ("blank sleeve", "candidates", {"x": {"start_weights": {"  ": 1.0}, "end_weights": {"SPY": 1.0}, "glide_years": 0}}, "sleeve keys"),
    ("weight above one", "candidates", {"x": {"start_weights": {"SPY": 1.5}, "end_weights": {"SPY": 1.0}, "glide_years": 0}}, r"lie in \[0, 1\]"),
    ("weights off simplex", "candidates", {"x": {"start_weights": {"SPY": 0.5, "QQQ": 0.4}, "end_weights": {"SPY": 1.0}, "glide_years": 0}}, "must sum to 1"),
    ("neighbors non-object", "neighbors", [], "neighbors must be an object"),
    ("neighbors unknown key", "neighbors", {"ghost": []}, "not among candidates"),
    ("neighbors non-array", "neighbors", {"spy100_qqq0": "spy90_qqq10"}, "must be an array"),
    ("neighbors self", "neighbors", {"spy100_qqq0": ["spy100_qqq0"]}, "must not list itself"),
    ("neighbors dupes", "neighbors", {"spy100_qqq0": ["spy90_qqq10", "spy90_qqq10"]}, "must be unique"),
    ("empty series", "century_series", {}, "non-empty object"),
    ("modern inverted", "modern_start", "2026-09-30", "modern_start must not be after"),
    ("century inverted", "century_start", "2026-08-31", "century_start must not be after"),
    ("empty horizons", "horizons_years", [], "non-empty array"),
    ("dup horizons", "horizons_years", [20, 20], "must be unique"),
    ("negative pre", "pre_retirement_months", -1, "non-negative integer"),
    ("pre over horizon", "pre_retirement_months", 241, "shortest horizon"),
    ("zero gamma", "primary_gamma", 0.0, "must be positive"),
    ("sensitivity non-array", "sensitivity_gammas", {}, "must be an array"),
    ("sensitivity non-positive", "sensitivity_gammas", [0.0], "must be positive"),
    ("negative band", "equivalence_band", -0.1, "non-negative"),
    ("drag non-object", "annual_drag_by_sleeve", [], "must be an object"),
    ("drag over one", "annual_drag_by_sleeve", {"SPY": 1.0}, r"lie in \[0, 1\)"),
    ("empty arm map", "tax_crosscheck_arm_map", {}, "non-empty object"),
    ("arm unknown candidate", "tax_crosscheck_arm_map", {"sp500_100": "ghost"}, "unknown candidate"),
    ("lineage non-object", "lineage", [], "must be an object"),
    ("lineage bad count", "lineage", {"related_trial_count": -1}, "non-negative integer"),
]


@pytest.mark.parametrize(("label", "key", "value", "match"), _CASES, ids=[case[0] for case in _CASES])
def test_load_rejects_invalid_fields(tmp_path: Path, label: str, key: str, value: object, match: str) -> None:
    """Every documented config rejection names its field."""
    del label
    with pytest.raises(ValueError, match=match):
        load_pension_decision_spec(_write(tmp_path, _mutated(_document(), key, value)))


def test_load_rejects_duplicate_keys_and_non_objects(tmp_path: Path) -> None:
    """Duplicate JSON keys and non-object documents fail closed."""
    dupes = tmp_path / "dupes.json"
    dupes.write_text('{"name": "a", "name": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_pension_decision_spec(dupes)
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_pension_decision_spec(array)
    notes = _document()
    notes["notes"] = 7
    with pytest.raises(ValueError, match="notes must be a string"):
        load_pension_decision_spec(_write(tmp_path, notes))
    missing = _document()
    del missing["name"]
    with pytest.raises(ValueError, match="missing fields"):
        load_pension_decision_spec(_write(tmp_path, missing))
