"""Canonical result and data layout helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.settings import DataSettings


def experiments_dir(settings: DataSettings) -> Path:
    return settings.resolved_data_root() / "results" / "experiments"


def audits_dir(settings: DataSettings) -> Path:
    return settings.resolved_data_root() / "results" / "audits"


def thesis_reports_dir(settings: DataSettings) -> Path:
    return settings.resolved_data_root() / "results" / "thesis"


def thesis_wave_dir(settings: DataSettings) -> Path:
    return thesis_reports_dir(settings)


# Backward compat alias for older code referencing thesis_reports
def _thesis_reports_compat(settings: DataSettings) -> Path:
    return thesis_reports_dir(settings)
