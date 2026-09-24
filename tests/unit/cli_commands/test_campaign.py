"""Ensure co-modification wiring for campaign."""
from __future__ import annotations

from pathlib import Path

import pytest


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

    source = Path("experiments/wf_thesis_ai_compute_soxx_inc_10.json")
    config_path = tmp_path / source.name
    config_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    settings = DataSettings(data_root=str(tmp_path / "data"))

    assert camp_mod.run_strategy_selection_command(config_path=str(config_path), settings=settings) == 0
