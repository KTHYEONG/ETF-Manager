"""Fail-closed guards against partial ingest shrinking operational partitions."""

from __future__ import annotations

from src.data.catalog import latest_artifact
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError

_MIN_PRIOR_ROWS = 100
_SHRINK_RATIO = 0.5
_GUARDED = frozenset({Dataset.PRICES, Dataset.FX})


class PartitionShrinkError(ValueError):
    """Raised when a new partition would replace a substantially larger trusted one."""


def assert_safe_partition_replacement(
    dataset: Dataset,
    new_row_count: int,
    settings: DataSettings,
    *,
    allow_shrink: bool = False,
) -> None:
    """Reject writes that would demote a full panel to a tiny partial partition."""
    if allow_shrink or dataset not in _GUARDED:
        return
    try:
        prior = latest_artifact(settings, dataset)
    except UntrustedDatasetError:
        return
    prior_rows = prior.manifest.row_count
    if prior_rows < _MIN_PRIOR_ROWS:
        return
    if new_row_count < prior_rows * _SHRINK_RATIO:
        raise PartitionShrinkError(
            f"refusing to replace {dataset!s} partition: new_rows={new_row_count} "
            f"prior_rows={prior_rows} (>{_SHRINK_RATIO:.0%} shrink); use full-panel ingest"
        )
