"""Layout bounds for validation campaign split."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario_id", ["test_campaign_facade_under_120_lines"])
def test_campaign_facade_under_120_lines(scenario_id: str) -> None:
    p = Path("src/validation/campaign.py")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 120, f"{p} has {len(lines)} > 120"


@pytest.mark.parametrize("scenario_id", ["test_campaign_split_modules_under_450_lines"])
def test_campaign_split_modules_under_450_lines(scenario_id: str) -> None:
    for name in ("walk_forward.py", "cost_grid.py", "cadence_robustness.py"):
        p = Path("src/validation") / name
        assert p.exists(), f"missing {p}"
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 450, f"{p} has {len(lines)} > 450"


@pytest.mark.parametrize("scenario_id", ["test_campaign_facade_reexports_runners"])
def test_campaign_facade_reexports_runners(scenario_id: str) -> None:
    from src.validation.campaign import run_cadence_robustness, run_walk_forward_adoption, run_walk_forward_cost_grid
    from src.validation.cadence_robustness import run_cadence_robustness as cr
    from src.validation.cost_grid import run_walk_forward_cost_grid as cg
    from src.validation.walk_forward import run_walk_forward_adoption as wf

    assert run_walk_forward_adoption is wf
    assert run_walk_forward_cost_grid is cg
    assert run_cadence_robustness is cr
