"""Unit tests for thesis registry and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.etf.sleeves import VehicleId
from src.policy.thesis import (
    ThesisError,
    ThesisId,
    ThesisSpec,
    ThesisStatus,
    get_thesis,
    load_thesis_registry,
    transition_thesis,
)

_SEED_TEMPLATE: dict[str, object] = {
    "id": "ai_compute",
    "version": 1,
    "title": "AI compute",
    "status": "research",
    "horizon": {"min_years": 5, "target_years": 10},
    "causal_chain": ["ai_compute_demand"],
    "falsifiers": ["capex_structural_slowdown"],
    "candidate_sleeves": ["ai_semiconductor"],
    "historical_proxies": ["SOXX"],
    "evidence": {
        "source": "declared",
        "structural": "unknown",
        "historical": "unknown",
        "valuation": "unknown",
        "expectations": "unknown",
        "crowding": "unknown",
    },
}


@pytest.mark.parametrize("scenario_id", ["THESIS-01-load-three-seeds"])
def test_thesis_01_load_three_seeds(scenario_id: str) -> None:
    """THESIS-01-load-three-seeds"""
    registry = load_thesis_registry(Path("configs/theses"))
    assert set(registry) == {
        ThesisId.AI_COMPUTE,
        ThesisId.AI_POWER_BOTTLENECK,
        ThesisId.PHYSICAL_AUTOMATION,
    }
    for spec in registry.values():
        assert len(spec.falsifiers) >= 1
        assert spec.status is ThesisStatus.RESEARCH
        assert spec.evidence.source == "declared"
        assert spec.evidence.structural == "unknown"
        assert spec.evidence.historical == "unknown"
        assert spec.evidence.valuation == "unknown"
        assert spec.evidence.expectations == "unknown"
        assert spec.evidence.crowding == "unknown"
        assert spec.horizon.min_years == 5
        assert spec.horizon.target_years == 10
    assert VehicleId.SOXX in registry[ThesisId.AI_COMPUTE].historical_proxies
    assert VehicleId.PAVE in registry[ThesisId.AI_POWER_BOTTLENECK].historical_proxies
    assert VehicleId.ROBO in registry[ThesisId.PHYSICAL_AUTOMATION].historical_proxies


@pytest.mark.parametrize("scenario_id", ["THESIS-02-reject-empty-falsifiers"])
def test_thesis_02_reject_empty_falsifiers(scenario_id: str, tmp_path: Path) -> None:
    """THESIS-02-reject-empty-falsifiers"""
    payload = dict(_SEED_TEMPLATE)
    payload["falsifiers"] = []
    with pytest.raises(ValidationError):
        ThesisSpec.model_validate(payload)
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValidationError, ThesisError)):
        load_thesis_registry(tmp_path)


@pytest.mark.parametrize("scenario_id", ["THESIS-03-reject-extra-keys"])
def test_thesis_03_reject_extra_keys(scenario_id: str) -> None:
    """THESIS-03-reject-extra-keys"""
    payload = dict(_SEED_TEMPLATE)
    payload["magic_score"] = 87.3
    with pytest.raises(ValidationError):
        ThesisSpec.model_validate(payload)


@pytest.mark.parametrize("scenario_id", ["THESIS-04-file-status-not-challenger"])
def test_thesis_04_file_status_not_challenger(scenario_id: str) -> None:
    """THESIS-04-file-status-not-challenger"""
    assert not hasattr(ThesisStatus, "ADOPTED")
    for status in ("operational_challenger", "confirmed", "prospective_challenger", "reopened"):
        payload = dict(_SEED_TEMPLATE)
        payload["status"] = status
        with pytest.raises((ValidationError, ThesisError)):
            ThesisSpec.model_validate(payload)


@pytest.mark.parametrize("scenario_id", ["THESIS-05-unknown-id-and-dupes"])
def test_thesis_05_unknown_id_and_dupes(scenario_id: str, tmp_path: Path) -> None:
    """THESIS-05-unknown-id-and-dupes"""
    full_registry = load_thesis_registry(Path("configs/theses"))
    partial = {ThesisId.AI_COMPUTE: full_registry[ThesisId.AI_COMPUTE]}
    with pytest.raises(ThesisError, match="unknown thesis"):
        get_thesis(partial, ThesisId.AI_POWER_BOTTLENECK)

    one = tmp_path / "a.json"
    two = tmp_path / "b.json"
    one.write_text(json.dumps(_SEED_TEMPLATE), encoding="utf-8")
    two.write_text(json.dumps(_SEED_TEMPLATE), encoding="utf-8")
    with pytest.raises(ThesisError, match="duplicate"):
        load_thesis_registry(tmp_path)

    missing = tmp_path / "empty"
    missing.mkdir()
    with pytest.raises(ThesisError, match="empty"):
        load_thesis_registry(missing)

    with pytest.raises(ThesisError, match="missing"):
        load_thesis_registry(tmp_path / "nope")


@pytest.mark.parametrize("scenario_id", ["LIFE-01-legal-research-to-confirmed"])
def test_life_01_legal_research_to_confirmed(scenario_id: str) -> None:
    """LIFE-01-legal-research-to-confirmed"""
    research_spec = load_thesis_registry(Path("configs/theses"))[ThesisId.AI_COMPUTE]
    assert research_spec.status is ThesisStatus.RESEARCH
    updated = transition_thesis(research_spec, ThesisStatus.CONFIRMED, reason="wf-note")
    assert updated.status is ThesisStatus.CONFIRMED
    assert updated.id is research_spec.id
    assert updated.falsifiers == research_spec.falsifiers
    assert updated.candidate_sleeves == research_spec.candidate_sleeves
    assert research_spec.status is ThesisStatus.RESEARCH


@pytest.mark.parametrize("scenario_id", ["LIFE-02-illegal-and-blank-reason"])
def test_life_02_illegal_and_blank_reason(scenario_id: str) -> None:
    """LIFE-02-illegal-and-blank-reason"""
    discovered = ThesisSpec.model_validate({**_SEED_TEMPLATE, "status": "discovered"})
    research = load_thesis_registry(Path("configs/theses"))[ThesisId.AI_COMPUTE]
    operational = ThesisSpec.model_construct(
        _fields_set=research.__pydantic_fields_set__,
        id=research.id,
        version=research.version,
        title=research.title,
        status=ThesisStatus.OPERATIONAL_CHALLENGER,
        horizon=research.horizon,
        causal_chain=research.causal_chain,
        falsifiers=research.falsifiers,
        candidate_sleeves=research.candidate_sleeves,
        historical_proxies=research.historical_proxies,
        evidence=research.evidence,
    )

    with pytest.raises(ThesisError, match="illegal transition"):
        transition_thesis(discovered, ThesisStatus.CONFIRMED, reason="skip")
    with pytest.raises(ThesisError, match="illegal transition"):
        transition_thesis(research, ThesisStatus.OPERATIONAL_CHALLENGER, reason="skip")
    with pytest.raises(ThesisError, match="illegal transition"):
        transition_thesis(operational, ThesisStatus.CONFIRMED, reason="skip")
    with pytest.raises(ThesisError, match="non-blank"):
        transition_thesis(research, ThesisStatus.CONFIRMED, reason="")
    with pytest.raises(ThesisError, match="non-blank"):
        transition_thesis(research, ThesisStatus.CONFIRMED, reason="   ")


@pytest.mark.parametrize("scenario_id", ["LIFE-03-reopen-cycle"])
def test_life_03_reopen_cycle(scenario_id: str) -> None:
    """LIFE-03-reopen-cycle"""
    research = load_thesis_registry(Path("configs/theses"))[ThesisId.AI_COMPUTE]
    dormant = transition_thesis(research, ThesisStatus.DORMANT, reason="gate-fail")
    reopened = transition_thesis(dormant, ThesisStatus.REOPENED, reason="structural-break")
    back = transition_thesis(reopened, ThesisStatus.RESEARCH, reason="revalidate")
    assert back.status is ThesisStatus.RESEARCH
    with pytest.raises(ThesisError, match="illegal transition"):
        transition_thesis(dormant, ThesisStatus.CONFIRMED, reason="skip")


@pytest.mark.parametrize("scenario_id", ["test_physical_automation_thesis_proxy_is_robo"])
def test_physical_automation_thesis_proxy_is_robo(scenario_id: str) -> None:
    """test_physical_automation_thesis_proxy_is_robo"""
    registry = load_thesis_registry(Path("configs/theses"))
    assert registry[ThesisId.PHYSICAL_AUTOMATION].historical_proxies == [VehicleId.ROBO]


@pytest.mark.parametrize("scenario_id", ["test_experiment_map_points_physical_automation_to_robo"])
def test_experiment_map_points_physical_automation_to_robo(scenario_id: str) -> None:
    """test_experiment_map_points_physical_automation_to_robo"""
    exp_map = json.loads(Path("configs/theses/experiment_map.json").read_text(encoding="utf-8"))
    path_str = exp_map["physical_automation"]
    assert path_str.endswith("m_thesis_physical_automation_robo.json")
    assert Path(path_str).exists()
    doc = json.loads(Path(path_str).read_text(encoding="utf-8"))
    assert doc["candidates"][0]["targets"]["ROBO"] == 1.0
    assert doc["thesis_id"] == "physical_automation"
    assert doc["start"] == "2013-10-31"
    assert doc["horizon_months"] == 120
    inc_path = Path("configs/experiments/m_thesis_physical_automation_robo_inc_5_10_15.json")
    assert inc_path.exists()
    inc_doc = json.loads(inc_path.read_text(encoding="utf-8"))
    weights = {tuple(sorted(c["targets"].items())) for c in inc_doc["candidates"]}
    assert any(("ROBO", 0.05) in dict(c["targets"]).items() for c in inc_doc["candidates"])
    assert any(("ROBO", 0.10) in dict(c["targets"]).items() for c in inc_doc["candidates"])
    assert any(("ROBO", 0.15) in dict(c["targets"]).items() for c in inc_doc["candidates"])
    assert Path("configs/experiments/m_thesis_physical_automation_botz_prospective.json").exists()


@pytest.mark.parametrize("scenario_id", ["test_ai_power_thesis_proxy_is_pave"])
def test_ai_power_thesis_proxy_is_pave(scenario_id: str) -> None:
    """test_ai_power_thesis_proxy_is_pave"""
    registry = load_thesis_registry(Path("configs/theses"))
    proxies = registry[ThesisId.AI_POWER_BOTTLENECK].historical_proxies
    assert proxies == [VehicleId.PAVE]
    assert VehicleId.GRID not in proxies


@pytest.mark.parametrize("scenario_id", ["test_experiment_map_points_ai_power_to_pave"])
def test_experiment_map_points_ai_power_to_pave(scenario_id: str) -> None:
    """test_experiment_map_points_ai_power_to_pave"""
    exp_map = json.loads(Path("configs/theses/experiment_map.json").read_text(encoding="utf-8"))
    path_str = exp_map["ai_power_bottleneck"]
    assert path_str.endswith("m_thesis_ai_power_pave.json")
    assert Path(path_str).exists()
    inc_path = Path("configs/experiments/m_thesis_ai_power_pave_inc_5_10_15.json")
    assert inc_path.exists()
    doc = json.loads(inc_path.read_text(encoding="utf-8"))
    assert doc["preregistration"]["weights_locked"] is True
    cands = doc["candidates"]
    assert len(cands) == 3
    ids = {c["id"] for c in cands}
    assert ids == {"qqq95_pave5", "qqq90_pave10", "qqq85_pave15"}
    for c in cands:
        targets = c["targets"]
        assert "PAVE" in targets
        assert "QQQ" in targets
    # ensure legacy grid file still exists but not mapped
    assert Path("configs/experiments/m_thesis_ai_power_bottleneck_grid.json").exists()
