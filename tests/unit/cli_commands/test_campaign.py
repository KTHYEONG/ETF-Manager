"""Ensure co-modification wiring for campaign."""
from __future__ import annotations

from pathlib import Path

import pytest

_THESIS_INC_10_CONFIG = """{
  "name": "wf_thesis_ai_compute_soxx_inc_10",
  "start": "2012-08-31",
  "end": "2024-08-31",
  "contribution_krw": 1000000,
  "hurdle": 0.02,
  "objective": "ce",
  "horizon_months": 36,
  "train_months": 60,
  "test_months": 36,
  "thesis_id": "ai_compute",
  "preregistration": {
    "weights_locked": true,
    "universe_locked": true,
    "baseline_frozen": true
  },
  "baseline": {
    "id": "qqq_baseline",
    "policy": "qqq",
    "modules": 0,
    "targets": {
      "QQQ": 1.0
    }
  },
  "candidates": [
    {
      "id": "qqq90_soxx10",
      "policy": "qqq",
      "modules": 1,
      "targets": {
        "QQQ": 0.9,
        "SOXX": 0.1
      }
    }
  ]
}"""



@pytest.mark.parametrize("scenario_id", ["test_campaign_import"])
def test_campaign_import(scenario_id: str) -> None:
    import src.cli_commands.campaign as mod
    assert hasattr(mod, "run_final_historical_campaign_command")
    assert hasattr(mod, "run_accumulation_cohort_command")


@pytest.mark.parametrize("scenario_id", ["test_strategy_selection_thesis_preregistration"])
def test_strategy_selection_thesis_preregistration(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Thesis walk-forward selection passes preregistration via THESES_DIR."""
    from types import SimpleNamespace

    import src.cli_commands.campaign as camp_mod
    import src.validation.strategy_selection as sel_mod
    from src.data.settings import DataSettings
    from src.validation.strategy_selection import StrategySelectionReport

    monkeypatch.setattr(camp_mod, "assert_experiment_feasible", lambda spec, settings: None)
    monkeypatch.setattr(camp_mod, "latest_artifact", lambda settings, dataset: SimpleNamespace(manifest=SimpleNamespace(normalized_sha256="f" * 64)))
    monkeypatch.setattr(sel_mod, "make_selection_runner", lambda settings, spec: (lambda config: None))
    fake_report = StrategySelectionReport(
        name="wf_thesis_ai_compute_soxx_inc_10",
        baseline_arm_id="base",
        objective="ce",
        rows=(),
        in_sample_champion_arm_id=None,
        oos_eligible_arm_ids=(),
        recommended_arm_id="base",
        operational_unlock=False,
        selection_reason="test",
    )
    monkeypatch.setattr(sel_mod, "run_strategy_selection", lambda spec, runner: fake_report)
    written = tmp_path / "selection_report.json"
    monkeypatch.setattr(sel_mod, "write_strategy_selection_report", lambda report, settings, experiment_id: written)

    config_path = tmp_path / "wf_thesis_ai_compute_soxx_inc_10.json"
    config_path.write_text(_THESIS_INC_10_CONFIG, encoding="utf-8")
    settings = DataSettings(data_root=str(tmp_path / "data"))

    assert camp_mod.run_strategy_selection_command(config_path=str(config_path), settings=settings) == 0


def test_campaign_commands_fail_closed_on_missing_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing definitions resolve through the repository anchor and fail closed."""
    import src.cli_commands.campaign as camp_mod
    from src.data.settings import DataSettings

    source_text = Path("configs/research/m_thesis_ai_compute_soxx.json").read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root=tmp_path / "data")
    missing = str(tmp_path / "no-such-experiment.json")
    definition = tmp_path / "m_thesis_ai_compute_soxx.json"
    definition.write_text(source_text, encoding="utf-8")
    assert camp_mod.run_ablation_command(config_path=missing, settings=settings) == 1
    assert camp_mod.run_walk_forward_command(config_path=missing, settings=settings) == 1
    assert camp_mod.run_strategy_selection_command(config_path=missing, settings=settings) == 1
    assert camp_mod.run_walk_forward_costs_command(config_path=missing, settings=settings) == 1
    assert camp_mod.run_walk_forward_proxy_command(config_path=missing, settings=settings) == 1
    assert camp_mod.run_cadence_robustness_command(config_path=missing, settings=settings, seed=1, bootstrap_paths=1) == 1
    assert camp_mod.run_accumulation_cohort_command(
        config_path=missing, settings=settings, horizon_months=120, cohort_step_months=12, bootstrap_paths=1, seed=1
    ) == 1
    assert camp_mod.run_final_historical_campaign_command(config_path=missing, settings=settings, seed=1) == 1
    assert camp_mod.run_audit_feasibility_command(config_path=missing, settings=settings, write_report=False) == 1
    assert camp_mod.run_prospective_monitor_command(bundle_path=missing, as_of="2026-10-01", settings=settings) == 1
    assert camp_mod.run_ablation_command(config_path=str(definition), settings=settings) == 1
