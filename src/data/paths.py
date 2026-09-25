"""Canonical repository path resolution for inputs, records, and generated outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.data.settings import DataSettings

LEGACY_FLAT_RESULT_SUBDIRS: Final[tuple[str, ...]] = ("experiments", "audits", "thesis")


# Repo-relative, git-tracked inputs; resolved against the repository root so every
# caller shares one anchor regardless of process working directory.
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


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class RepositoryPaths:
    """Resolved repository roots for versioned inputs, frozen records, and generated outputs.

    Attributes:
        root: Canonical project root.
        experiments: Versioned experiment definitions.
        prospective_records: Frozen prospective evidence directory.
        data: Configured data root.
        results: Generated run output root.
    """

    root: Path
    experiments: Path
    prospective_records: Path
    data: Path
    results: Path


def resolve_repository_paths(settings: DataSettings) -> RepositoryPaths:
    """Resolve every runtime path from one typed repository root.

    Args:
        settings: Data root configuration.

    Returns:
        Canonical paths shared by CLI, experiment loading, and report writers.

    Raises:
        ValueError: If a configured path escapes its permitted root.
    """
    root = _repository_root()
    experiments = root / EXPERIMENTS_DIR.as_posix()
    prospective_records = root / PROSPECTIVE_RECORDS_DIR.as_posix()
    data = settings.resolved_data_root()
    results = data / "results"
    for guarded, label in (
        (experiments, "experiments"),
        (prospective_records, "records/prospective"),
    ):
        if results == guarded or _is_within(results, guarded) or _is_within(guarded, results):
            raise ValueError(
                f"generated results {str(results)!r} escapes permitted root into {label} {str(guarded)!r}"
            )
    return RepositoryPaths(
        root=root,
        experiments=experiments,
        prospective_records=prospective_records,
        data=data,
        results=results,
    )


def results_root(settings: DataSettings) -> Path:
    """Return the git-ignored root for machine-generated research outputs.

    Every run artifact lives under ``<data_root>/results/<experiment_slug>/``; nothing
    below this root is curated evidence — promotion to ``docs/results/`` is explicit.
    """
    return resolve_repository_paths(settings).results


def legacy_results_dir(settings: DataSettings) -> Path:
    """Return the quarantine directory for pre-layout outputs whose experiment cannot be inferred.

    Files here are preserved evidence and are never pruned automatically.
    """
    return resolve_repository_paths(settings).results / "_legacy"
