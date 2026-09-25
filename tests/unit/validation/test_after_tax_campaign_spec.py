"""Boundary tests: campaign JSON validation and runner fail-closed guards."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.validation.after_tax_campaign import run_after_tax_campaign
from src.validation.after_tax_campaign_config import load_after_tax_campaign_spec

_STATIC_QQQ = {"rule_id": "static", "core_targets": {"QQQ": 1.0}}


def _base() -> dict[str, Any]:
    return {
        "name": "spec_boundary",
        "start": "2000-04-01",
        "end": "2022-04-30",
        "contribution_krw": 1_000_000,
        "commission_bps": 10.0,
        "fx_spread_bps": 20.0,
        "tax_regime_path": "configs/tax/kr_overseas_equity.json",
        "horizons_months": [120],
        "primary_horizon_months": 120,
        "step_months": 12,
        "crash_regimes": ["dot_com"],
        "crash_window_months": 36,
        "fractional_shares": False,
        "bootstrap_paths": 10,
        "baseline_arm_id": "b",
        "arms": [
            {"arm_id": "b", "role": "baseline", "rule": dict(_STATIC_QQQ), "mode": "buy_only",
             "rebalance_band": None, "harvest_gains": False},
            {"arm_id": "c", "role": "operational_candidate", "rule": dict(_STATIC_QQQ), "mode": "buy_only",
             "rebalance_band": None, "harvest_gains": True},
        ],
        "gate": {"median_ratio_floor": 1.0, "worst_ratio_floor": 0.97, "bootstrap_p05_floor": 1.0},
    }


def _arm(document: dict[str, Any], index: int) -> dict[str, Any]:
    arm: dict[str, Any] = document["arms"][index]
    return arm


_CASES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("blank_name", lambda d: d.__setitem__("name", "  "), "name must be non-blank"),
    ("bad_start", lambda d: d.__setitem__("start", "not-a-date"), "start must be an ISO date"),
    ("start_after_end", lambda d: d.__setitem__("start", "2030-01-01"), "is after end"),
    ("contribution_type", lambda d: d.__setitem__("contribution_krw", "1000"), "contribution_krw must be a number"),
    ("contribution_zero", lambda d: d.__setitem__("contribution_krw", 0), "finite positive"),
    ("commission_type", lambda d: d.__setitem__("commission_bps", True), "commission_bps must be a number"),
    ("commission_negative", lambda d: d.__setitem__("commission_bps", -1.0), "finite nonnegative"),
    ("blank_tax_path", lambda d: d.__setitem__("tax_regime_path", " "), "tax_regime_path must be non-blank"),
    ("horizons_empty", lambda d: d.__setitem__("horizons_months", []), "nonempty list"),
    ("horizon_entry", lambda d: d.__setitem__("horizons_months", [0]), "positive integers"),
    ("primary_type", lambda d: d.__setitem__("primary_horizon_months", "120"), "primary_horizon_months must be"),
    ("primary_absent", lambda d: d.__setitem__("primary_horizon_months", 60), "absent from horizons_months"),
    ("step_zero", lambda d: d.__setitem__("step_months", 0), "step_months must be"),
    ("crash_not_list", lambda d: d.__setitem__("crash_regimes", "gfc"), "crash_regimes must be a list"),
    ("crash_unknown", lambda d: d.__setitem__("crash_regimes", ["nope"]), "unknown crash regime"),
    ("crash_window", lambda d: d.__setitem__("crash_window_months", 0), "crash_window_months must be"),
    ("fractional_type", lambda d: d.__setitem__("fractional_shares", "no"), "fractional_shares must be a boolean"),
    ("bootstrap_zero", lambda d: d.__setitem__("bootstrap_paths", 0), "bootstrap_paths must be"),
    ("blank_baseline", lambda d: d.__setitem__("baseline_arm_id", ""), "baseline_arm_id must be non-blank"),
    ("gate_type", lambda d: d.__setitem__("gate", []), "gate must be an object"),
    ("gate_nonfinite", lambda d: d["gate"].__setitem__("median_ratio_floor", float("nan")), "must be finite"),
    ("arms_empty", lambda d: d.__setitem__("arms", []), "arms must be a nonempty list"),
    ("arm_not_object", lambda d: d["arms"].append("x"), "arm entries must be objects"),
    ("arm_blank_id", lambda d: _arm(d, 1).__setitem__("arm_id", " "), "arm_id must be non-blank"),
    ("arm_unknown_role", lambda d: _arm(d, 1).__setitem__("role", "star"), "unknown arm role"),
    ("arm_unknown_mode", lambda d: _arm(d, 1).__setitem__("mode", "yolo"), "unknown execution mode"),
    ("band_type", lambda d: _arm(d, 1).update(mode="rebalance_band", rebalance_band="x"), "rebalance_band must be numeric"),
    ("band_range", lambda d: _arm(d, 1).update(mode="rebalance_band", rebalance_band=1.5), "must lie in \\[0, 1\\]"),
    ("band_missing", lambda d: _arm(d, 1).update(mode="rebalance_band", rebalance_band=None), "without a band"),
    ("band_on_buy_only", lambda d: _arm(d, 1).__setitem__("rebalance_band", 0.05), "BUY_ONLY with a band"),
    ("harvest_type", lambda d: _arm(d, 1).__setitem__("harvest_gains", "yes"), "harvest_gains must be a boolean"),
    ("rule_type", lambda d: _arm(d, 1).__setitem__("rule", "static"), "rule must be an object"),
    ("missing_field", lambda d: d.pop("step_months"), "missing field"),
    ("baseline_absent", lambda d: d.__setitem__("baseline_arm_id", "zzz"), "not found"),
    ("baseline_wrong_role", lambda d: _arm(d, 0).__setitem__("role", "disclosure"), "role must be baseline"),
    ("two_baselines", lambda d: _arm(d, 1).__setitem__("role", "baseline"), "exactly one BASELINE"),
    ("duplicate_arm", lambda d: _arm(d, 1).__setitem__("arm_id", "b"), "duplicate arm id"),
    ("tax_path_missing", lambda d: d.__setitem__("tax_regime_path", "configs/tax/nope.json"), "tax_regime_path not found"),
]


@pytest.mark.parametrize(("case_id", "mutate", "message"), _CASES, ids=[case[0] for case in _CASES])
def test_campaign_spec_rejects_invalid_field(
    tmp_path: Path, case_id: str, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Every malformed campaign field fails closed with a ValueError naming the field."""
    document = copy.deepcopy(_base())
    mutate(document)
    path = tmp_path / f"{case_id}.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_after_tax_campaign_spec(path)


def test_campaign_spec_rejects_non_object_document(tmp_path: Path) -> None:
    """A top-level JSON array is not a campaign spec."""
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be an object"):
        load_after_tax_campaign_spec(path)


def test_campaign_runner_rejects_horizon_without_cohorts(tmp_path: Path) -> None:
    """A window shorter than the horizon yields no cohorts and never reaches the runner."""
    document = _base()
    document["start"] = "2020-01-01"
    document["end"] = "2021-01-01"
    path = tmp_path / "short.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    spec = load_after_tax_campaign_spec(path)

    def runner(_config: Any) -> Any:
        raise AssertionError("runner must not be called")

    with pytest.raises(ValueError, match="no cohorts fit horizon 120"):
        run_after_tax_campaign(spec, runner, seed=1)


def test_campaign_runner_rejects_nonpositive_baseline_wealth(tmp_path: Path) -> None:
    """A wiped-out baseline cannot anchor ratios; the campaign aborts instead of dividing by zero."""
    from tests.unit.validation.test_after_tax_campaign import _make_result

    document = _base()
    document["horizons_months"] = [12]
    document["primary_horizon_months"] = 12
    document["end"] = "2001-06-30"
    path = tmp_path / "zero.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    spec = load_after_tax_campaign_spec(path)

    with pytest.raises(ValueError, match="baseline wealth must be positive"):
        run_after_tax_campaign(spec, lambda _config: _make_result(0.0), seed=1)

    assert spec.start == date(2000, 4, 1)
