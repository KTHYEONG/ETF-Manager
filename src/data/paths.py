"""Canonical result and data layout helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.data.settings import DataSettings

LEGACY_FLAT_RESULT_SUBDIRS: Final[tuple[str, ...]] = ("experiments", "audits", "thesis")


# Repo-relative, git-tracked inputs; resolved against the process cwd like every other
# relative path in the CLI (the CLI is always run from the repo root).
EXPERIMENTS_DIR: Final[Path] = Path("experiments")
EXPERIMENT_ARCHIVE_DIR: Final[Path] = EXPERIMENTS_DIR / "archive"
EXPERIMENT_INDEX_PATH: Final[Path] = EXPERIMENTS_DIR / "INDEX.json"
LEGACY_EXPERIMENTS_PREFIX: Final[str] = "configs/experiments"
PROSPECTIVE_RECORDS_DIR: Final[Path] = Path("records/prospective")
CURRENT_OPERATIONAL_BUNDLE_PATH: Final[Path] = Path("configs/prospective/CURRENT_OPERATIONAL_BUNDLE.json")
THESES_DIR: Final[Path] = Path("configs/theses")
THESIS_EXPERIMENT_MAP_PATH: Final[Path] = THESES_DIR / "experiment_map.json"
THESIS_FUNDAMENTALS_DIR: Final[Path] = Path("configs/data/thesis_fundamentals")
ETF_METADATA_BOOTSTRAP_PATH: Final[Path] = Path("configs/etf_metadata/bootstrap.json")
NPORT_SERIES_MAP_PATH: Final[Path] = Path("configs/etf_metadata/nport_series_map.json")
PANEL_HARD_STOP_PATH: Final[Path] = Path("configs/data/panel_hard_stop.json")


def results_root(settings: DataSettings) -> Path:
    """Return the git-ignored root for machine-generated research outputs.

    Every run artifact lives under ``<data_root>/results/<experiment_slug>/``; nothing
    below this root is curated evidence — promotion to ``docs/results/`` is explicit.
    """
    return settings.resolved_data_root() / "results"


def legacy_results_dir(settings: DataSettings) -> Path:
    """Return the quarantine directory for pre-layout outputs whose experiment cannot be inferred.

    Files here are preserved evidence and are never pruned automatically.
    """
    return results_root(settings) / "_legacy"
