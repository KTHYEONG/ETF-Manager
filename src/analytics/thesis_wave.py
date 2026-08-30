# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.wave."""
import sys
import src.analytics.thesis.wave as _real
sys.modules[__name__] = _real
from src.analytics.thesis.wave import ThesisWaveEntry, ThesisWaveFailure, ThesisWaveReport, load_thesis_experiment_map, run_thesis_wave, write_thesis_wave_markdown
__all__ = ["ThesisWaveEntry", "ThesisWaveFailure", "ThesisWaveReport", "load_thesis_experiment_map", "run_thesis_wave", "write_thesis_wave_markdown"]
