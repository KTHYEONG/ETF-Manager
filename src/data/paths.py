"""Canonical repository path resolution for inputs, records, and generated outputs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.data.settings import DataSettings

logger = logging.getLogger(__name__)

LEGACY_FLAT_RESULT_SUBDIRS: Final[tuple[str, ...]] = ("experiments", "audits", "thesis")


# Repo-relative, git-tracked inputs; resolved against the repository root so every
# caller shares one anchor regardless of process working directory.
EXPERIMENTS_DIR: Final[Path] = Path("configs/research")
EXPERIMENT_ARCHIVE_DIR: Final[Path] = EXPERIMENTS_DIR / "archive"
EXPERIMENT_INDEX_PATH: Final[Path] = EXPERIMENTS_DIR / "INDEX.json"
DECISION_CONFIGS_DIR: Final[Path] = Path("configs/decision")
LEGACY_EXPERIMENTS_PREFIX: Final[str] = "configs/experiments"
LEGACY_ROOT_EXPERIMENTS_PREFIX: Final[str] = "experiments"
LEGACY_RECORDS_PREFIX: Final[str] = "records"
DECISION_CONFIG_RENAMES: Final[Mapping[str, str]] = {
    "final_historical_campaign_v1.json": "general.json",
    "pension_decision_v3.json": "pension.json",
    "isa_household_v1.json": "isa.json",
}
CURRENT_OPERATIONAL_BUNDLE_PATH: Final[Path] = Path("configs/prospective/CURRENT_OPERATIONAL_BUNDLE.json")
THESES_DIR: Final[Path] = Path("configs/theses")
THESIS_EXPERIMENT_MAP_PATH: Final[Path] = THESES_DIR / "experiment_map.json"
THESIS_FUNDAMENTALS_DIR: Final[Path] = Path("configs/data/thesis_fundamentals")
ETF_METADATA_BOOTSTRAP_PATH: Final[Path] = Path("configs/etf_metadata/bootstrap.json")
NPORT_SERIES_MAP_PATH: Final[Path] = Path("configs/etf_metadata/nport_series_map.json")
PANEL_HARD_STOP_PATH: Final[Path] = Path("configs/data/panel_hard_stop.json")
PRICE_CORRECTIONS_PATH: Final[Path] = Path("configs/data/price_corrections.json")
MACRO_BACKFILL_PATH: Final[Path] = Path("configs/data/macro_backfill.json")
UNIVERSE_MEMBERSHIP_PATH: Final[Path] = Path("configs/data/universe_membership.json")

_RECORDS_FROZEN_SUBDIRS: Final[tuple[tuple[str, str], ...]] = (
    ("pension_decisions", "pension"),
    ("isa_household", "isa"),
    ("prospective", "prospective"),
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_input_path(path: str | Path) -> Path:
    """Anchor a repo-relative versioned input to the repository root.

    Args:
        path: Repo-relative config path, or an already absolute path.

    Returns:
        The absolute path every caller shares regardless of process working directory.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repository_root() / candidate


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _relative_after_prefix(posix_path: str, prefix: str) -> tuple[str, ...] | None:
    parts = [part for part in posix_path.split("/") if part not in ("", ".")]
    head = prefix.split("/")
    if parts[: len(head)] != head or len(parts) == len(head):
        return None
    return tuple(parts[len(head):])


@dataclass(frozen=True, slots=True)
class RepositoryPaths:
    """Resolved repository roots for versioned inputs, frozen records, and generated outputs.

    Attributes:
        root: Canonical project root.
        experiments: Versioned research experiment definitions (``configs/research``).
        decisions: Versioned final decision specs (``configs/decision``).
        data: Configured data root (git-ignored, Drive-mirrored).
        results: Generated run output root (``<data>/runs``).
        frozen: Immutable decision records root (``<data>/frozen``).
        research: Promoted evidence root (``<data>/research``).
    """

    root: Path
    experiments: Path
    decisions: Path
    data: Path
    results: Path
    frozen: Path
    research: Path


def resolve_repository_paths(settings: DataSettings) -> RepositoryPaths:
    """Resolve every runtime path from one typed repository root.

    Raises:
        ValueError: If the configured data root sits inside a versioned config directory
            (research or decision), which would let generated files land in tracked inputs.
    """
    root = _repository_root()
    experiments = root / EXPERIMENTS_DIR.as_posix()
    decisions = root / DECISION_CONFIGS_DIR.as_posix()
    data = settings.resolved_data_root()
    results = data / "runs"
    frozen = data / "frozen"
    research = data / "research"
    generated = (("results", results), ("frozen", frozen), ("research", research))
    versioned = (("experiments", experiments), ("decisions", decisions))
    for generated_label, generated_dir in generated:
        for versioned_label, versioned_dir in versioned:
            if _is_within(generated_dir, versioned_dir):
                raise ValueError(
                    f"generated {generated_label} {str(generated_dir)!r} sits inside versioned "
                    f"{versioned_label} {str(versioned_dir)!r}"
                )
            if _is_within(versioned_dir, generated_dir):
                raise ValueError(
                    f"versioned {versioned_label} {str(versioned_dir)!r} sits inside generated "
                    f"{generated_label} {str(generated_dir)!r}"
                )
    return RepositoryPaths(
        root=root,
        experiments=experiments,
        decisions=decisions,
        data=data,
        results=results,
        frozen=frozen,
        research=research,
    )


def _repo_path_aliases(candidate: Path) -> list[Path]:
    """Ordered new-layout candidates for a frozen experiments/records reference."""
    posix = candidate.as_posix()
    aliases: list[Path] = []
    for prefix in (LEGACY_ROOT_EXPERIMENTS_PREFIX, LEGACY_EXPERIMENTS_PREFIX):
        remainder = _relative_after_prefix(posix, prefix)
        if remainder is None:
            continue
        renamed = DECISION_CONFIG_RENAMES.get(candidate.name)
        if renamed is not None:
            aliases.append(DECISION_CONFIGS_DIR / renamed)
        else:
            aliases.append(EXPERIMENTS_DIR.joinpath(*remainder))
            aliases.append(EXPERIMENT_ARCHIVE_DIR / candidate.name)
    for records_dir, frozen_subdir in _RECORDS_FROZEN_SUBDIRS:
        remainder = _relative_after_prefix(posix, f"{LEGACY_RECORDS_PREFIX}/{records_dir}")
        if remainder is None:
            continue
        data = resolve_repository_paths(DataSettings()).data
        aliases.append(data.joinpath("frozen", frozen_subdir, *remainder))
    return aliases


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path string cited by a frozen config or record to its current location.

    Frozen configs and records cite the locations that existed when they were written. Rewriting those
    strings would change the file bytes and break its recorded hash, so old prefixes are mapped here instead.
    An existing file at the given path always wins. Otherwise, in order: ``experiments/<name>`` and
    ``configs/experiments/<name>`` map to ``configs/decision/<new-name>`` when ``<name>`` is a key of
    ``DECISION_CONFIG_RENAMES``, else to ``configs/research/<name>`` and then ``configs/research/archive/<basename>``;
    ``records/pension_decisions/<file>`` maps to ``<data>/frozen/pension/<file>``, ``records/isa_household/<file>``
    to ``<data>/frozen/isa/<file>``, ``records/prospective/<file>`` to ``<data>/frozen/prospective/<file>``.
    Every candidate is also tried anchored at the repository root.

    Returns: Absolute path of the first existing candidate.
    Raises: FileNotFoundError: If no candidate exists.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    root = _repository_root()
    for alias in _repo_path_aliases(candidate):
        if alias.is_file():
            resolved = alias.resolve()
            logger.info("[DATA] event=repo_path_alias requested=%s resolved=%s", path, resolved)
            return resolved
        rooted = root / alias.as_posix()
        if rooted.is_file():
            resolved = rooted.resolve()
            logger.info("[DATA] event=repo_path_alias requested=%s resolved=%s", path, resolved)
            return resolved
    raise FileNotFoundError(f"repo path not found: {path}")


def results_root(settings: DataSettings) -> Path:
    """Return the git-ignored root for machine-generated research outputs.

    Every run artifact lives under ``<data_root>/runs/<experiment_slug>/``; nothing
    below this root is curated evidence — promotion to ``data/research/`` is explicit.
    """
    return resolve_repository_paths(settings).results


def legacy_results_dir(settings: DataSettings) -> Path:
    """Return the quarantine directory for pre-layout outputs whose experiment cannot be inferred.

    Files here are preserved evidence and are never pruned automatically.
    """
    return resolve_repository_paths(settings).results / "_legacy"
