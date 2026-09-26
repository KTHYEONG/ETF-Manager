"""Invariant guards for the pension decision pre-registration loader."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.validation.pension_decision_config import load_pension_decision_spec

_REPO = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO / "experiments" / "pension_decision_v1.json"
_V2_CONFIG_PATH = _REPO / "experiments" / "pension_decision_v2.json"


def test_shipped_v2_config_loads_with_products_and_reference(tmp_path: Path) -> None:
    """Shipped v2 config loads with 15 candidates, 2 controls, and priced sleeves."""
    del tmp_path
    spec = load_pension_decision_spec(_V2_CONFIG_PATH)
    assert spec.name == "pension_decision_v2"
    assert spec.benchmark_id == "spy100_qqq0"
    assert spec.dominance_reference_id == "spy20_qqq80"
    assert len(spec.candidates) == 15
    assert len(spec.controls) == 2
    sleeves = {sleeve for schedule in spec.candidates.values() for sleeve in schedule.start_weights}
    assert sleeves == {"SPY", "QQQ", "SCHD"}
    for sleeve in sleeves | {s for schedule in spec.controls.values() for s in schedule.start_weights}:
        product = spec.sleeve_products[sleeve]
        assert not product.currency_hedged
        assert spec.annual_drag_by_sleeve[sleeve] >= product.total_expense_ratio > 0.0


def test_v2_spy_line_matches_v1_schedules(tmp_path: Path) -> None:
    """v2 SPY line matches v1 schedules for every shared candidate id."""
    del tmp_path
    v2 = load_pension_decision_spec(_V2_CONFIG_PATH)
    v1 = load_pension_decision_spec(_CONFIG_PATH)
    shared = set(v1.candidates) & set(v2.candidates)
    assert len(shared) == 11
    for candidate_id in shared:
        assert v2.candidates[candidate_id] == v1.candidates[candidate_id]


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


def _product(code: str, *, expense: float = 0.0009, hedged: bool = False) -> dict[str, object]:
    return {
        "krx_code": code,
        "name": f"Product {code}",
        "listing_date": "2000-01-01",
        "total_expense_ratio": expense,
        "currency_hedged": hedged,
        "source_url": "https://example.com/product",
        "source_checked_date": "2026-01-01",
    }


def _document_with_splices_and_products() -> dict[str, object]:
    document = _document()
    document["modern_splices"] = {
        "SPY": {"proxy_weights": {"ff_mkt_monthly": 1.0}, "etf_first_month": "2000-04-30"},
        "QQQ": {
            "proxy_weights": {"ff_hitec_monthly": 0.5, "ff_mkt_monthly": 0.5},
            "etf_first_month": "2001-01-31",
        },
    }
    document["sleeve_products"] = {"SPY": _product("123456"), "QQQ": _product("654321")}
    document["annual_drag_by_sleeve"] = {"SPY": 0.001, "QQQ": 0.001}
    return document


def test_load_v1_config_has_empty_splice_mappings(tmp_path: Path) -> None:
    """A v1 config without the new fields loads with empty splice mappings."""
    del tmp_path
    spec = load_pension_decision_spec(_CONFIG_PATH)
    assert spec.modern_splices == {}
    assert spec.sleeve_products == {}


def test_load_accepts_valid_splices_and_products(tmp_path: Path) -> None:
    """Well-formed splice and product declarations parse into their dataclasses."""
    spec = load_pension_decision_spec(_write(tmp_path, _document_with_splices_and_products()))

    assert spec.modern_splices["SPY"].etf_first_month == date(2000, 4, 30)
    assert spec.modern_splices["SPY"].proxy_weights == {"ff_mkt_monthly": 1.0}
    assert spec.modern_splices["QQQ"].proxy_weights == {"ff_hitec_monthly": 0.5, "ff_mkt_monthly": 0.5}
    assert spec.sleeve_products["SPY"].krx_code == "123456"
    assert spec.sleeve_products["QQQ"].total_expense_ratio == pytest.approx(0.0009)
    assert spec.sleeve_products["SPY"].currency_hedged is False


def test_load_rejects_hedged_product(tmp_path: Path) -> None:
    """A currency-hedged product breaks the USD invariance and fails closed."""
    document = _document_with_splices_and_products()
    products = dict(document["sleeve_products"])  # type: ignore[arg-type]
    products["QQQ"] = _product("654321", hedged=True)
    document["sleeve_products"] = products

    with pytest.raises(ValueError, match="currency-hedged"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_drag_below_expense_ratio(tmp_path: Path) -> None:
    """An annual drag understating the product fee fails closed."""
    document = _document_with_splices_and_products()
    products = dict(document["sleeve_products"])  # type: ignore[arg-type]
    products["SPY"] = _product("123456", expense=0.0025)
    document["sleeve_products"] = products
    document["annual_drag_by_sleeve"] = {"SPY": 0.001, "QQQ": 0.001}

    with pytest.raises(ValueError, match="understates"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_products_missing_a_sleeve(tmp_path: Path) -> None:
    """Products must cover every candidate sleeve and name the gap."""
    document = _document_with_splices_and_products()
    products = dict(document["sleeve_products"])  # type: ignore[arg-type]
    del products["QQQ"]
    document["sleeve_products"] = products

    with pytest.raises(ValueError, match="QQQ"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_splice_on_unknown_sleeve(tmp_path: Path) -> None:
    """A splice on a sleeve no candidate uses fails closed."""
    document = _document_with_splices_and_products()
    splices = dict(document["modern_splices"])  # type: ignore[arg-type]
    splices["BOND"] = {"proxy_weights": {"ff_mkt_monthly": 1.0}, "etf_first_month": "2000-04-30"}
    document["modern_splices"] = splices

    with pytest.raises(ValueError, match="not a sleeve used"):
        load_pension_decision_spec(_write(tmp_path, document))


_DELETE: object = object()


def _nested(document: dict[str, object], keys: list[str], value: object) -> dict[str, object]:
    clone: dict[str, object] = json.loads(json.dumps(document))
    node: dict[str, object] = clone
    for key in keys[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    if value is _DELETE:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value
    return clone


_SPLICE_PRODUCT_CASES: list[tuple[str, list[str], object, str]] = [
    ("splices non-object", ["modern_splices"], [], "modern_splices must be an object"),
    ("splice entry non-object", ["modern_splices", "SPY"], [], "must be an object"),
    ("splice unknown field", ["modern_splices", "SPY", "extra"], 1, "unknown fields"),
    ("splice missing field", ["modern_splices", "SPY", "etf_first_month"], _DELETE, "missing field"),
    ("splice empty weights", ["modern_splices", "SPY", "proxy_weights"], {}, "non-empty object"),
    ("splice weights off simplex", ["modern_splices", "SPY", "proxy_weights"], {"a": 0.5}, "must sum to 1"),
    ("splice bad date", ["modern_splices", "SPY", "etf_first_month"], "not-a-date", "ISO date"),
    ("splice on modern start", ["modern_splices", "SPY", "etf_first_month"], "1999-04-30", "must lie in"),
    ("splice past modern end", ["modern_splices", "SPY", "etf_first_month"], "2026-09-30", "must lie in"),
    ("splice mid-month", ["modern_splices", "SPY", "etf_first_month"], "2000-04-15", "is invalid"),
    ("products non-object", ["sleeve_products"], [], "sleeve_products must be an object"),
    ("product non-object", ["sleeve_products", "SPY"], [], "must be an object"),
    ("product unknown field", ["sleeve_products", "SPY", "extra"], 1, "unknown fields"),
    ("product missing field", ["sleeve_products", "SPY", "name"], _DELETE, "missing field"),
    ("product short code", ["sleeve_products", "SPY", "krx_code"], "ABC12", "alphanumeric"),
    ("product symbol code", ["sleeve_products", "SPY", "krx_code"], "ABC-12", "alphanumeric"),
    ("product negative fee", ["sleeve_products", "SPY", "total_expense_ratio"], -0.1, r"lie in \[0, 0\.02\]"),
    ("product high fee", ["sleeve_products", "SPY", "total_expense_ratio"], 0.03, r"lie in \[0, 0\.02\]"),
    ("product non-bool hedge", ["sleeve_products", "SPY", "currency_hedged"], "yes", "must be a boolean"),
    ("product bad url", ["sleeve_products", "SPY", "source_url"], "ftp://example.com", "must start with"),
    ("product bad listing date", ["sleeve_products", "SPY", "listing_date"], "01-01-2000", "ISO date"),
]


def _schedule_entry(spy: float = 1.0, qqq: float = 0.0) -> dict[str, object]:
    return {"start_weights": {"SPY": spy, "QQQ": qqq}, "end_weights": {"SPY": spy, "QQQ": qqq}, "glide_years": 0}


def test_load_rejects_non_object_controls(tmp_path: Path) -> None:
    """A non-object controls declaration fails closed."""
    document = _document()
    document["controls"] = ["world"]

    with pytest.raises(ValueError, match="controls must be an object"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_reference_naming_a_control(tmp_path: Path) -> None:
    """A dominance reference must be a candidate, never a control."""
    document = _document()
    document["controls"] = {"world": _schedule_entry()}
    document["dominance_reference_id"] = "world"

    with pytest.raises(ValueError, match="not among candidates"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_control_id_collision(tmp_path: Path) -> None:
    """A control id colliding with a candidate id fails closed."""
    document = _document()
    document["controls"] = {"spy100_qqq0": _schedule_entry()}

    with pytest.raises(ValueError, match="collides"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_rejects_control_in_neighbors(tmp_path: Path) -> None:
    """Neighbors referencing a control id fail closed."""
    document = _document()
    document["controls"] = {"world": _schedule_entry()}
    neighbors = dict(document["neighbors"])  # type: ignore[arg-type]
    neighbors["spy100_qqq0"] = ["spy90_qqq10", "world"]
    document["neighbors"] = neighbors

    with pytest.raises(ValueError, match="references control"):
        load_pension_decision_spec(_write(tmp_path, document))


def test_load_accepts_reference_and_controls(tmp_path: Path) -> None:
    """A declared reference and disjoint controls parse into the spec."""
    document = _document()
    document["controls"] = {"world": _schedule_entry()}
    document["dominance_reference_id"] = "spy100_qqq0"

    spec = load_pension_decision_spec(_write(tmp_path, document))

    assert spec.dominance_reference_id == "spy100_qqq0"
    assert set(spec.controls) == {"world"}
    assert spec.modern_splices == {}
    assert spec.sleeve_products == {}


@pytest.mark.parametrize(
    ("label", "keys", "value", "match"),
    _SPLICE_PRODUCT_CASES,
    ids=[case[0] for case in _SPLICE_PRODUCT_CASES],
)
def test_load_rejects_invalid_splice_and_product_fields(
    tmp_path: Path, label: str, keys: list[str], value: object, match: str
) -> None:
    """Every malformed splice or product declaration names its field."""
    del label
    document = _nested(_document_with_splices_and_products(), keys, value)
    with pytest.raises(ValueError, match=match):
        load_pension_decision_spec(_write(tmp_path, document))
