"""Read-path seam: expose only rows a decision at ``decision_ts`` could have seen."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from src.etf_manager.data.pit import as_of, assert_no_lookahead
from src.etf_manager.data.schema import Dataset, spec_for


def load_as_of(frame: pl.DataFrame, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
    """Resolve the visible vintage of a dataset for a given decision timestamp.

    Args:
        frame: Availability-stamped dataset frame.
        dataset: Dataset identity resolved through the spec registry.
        decision_ts: Timezone-aware decision instant.

    Returns:
        Rows whose ``available_at`` precedes ``decision_ts``, one row per observation key.

    Raises:
        LookAheadError: If any surviving row is not yet available at ``decision_ts``.
        ValueError: On a naive ``decision_ts`` or missing ``available_at`` column.
    """
    spec = spec_for(dataset)
    visible = as_of(frame, spec, decision_ts)
    assert_no_lookahead(visible, decision_ts)
    return visible
