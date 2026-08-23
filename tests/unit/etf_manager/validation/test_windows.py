"""Unit tests for walk-forward folds and rolling cohorts."""

from __future__ import annotations

from datetime import date

import pytest

from src.etf_manager.validation.windows import (
    add_calendar_months,
    rolling_cohorts,
    walk_forward_windows,
)


@pytest.mark.parametrize("scenario_id", ["VAL-V01-walk-forward-order"])
def test_val_v01_walk_forward_order(scenario_id: str) -> None:
    """VAL-V01-walk-forward-order"""
    folds = walk_forward_windows(
        date(2020, 1, 15),
        date(2021, 7, 15),
        train_months=12,
        test_months=3,
        embargo_months=0,
    )

    assert folds
    for fold in folds:
        train_start, train_end, test_start, test_end = fold
        assert date(2020, 1, 15) <= train_start <= train_end < test_start <= test_end <= date(2021, 7, 15)
        assert test_start == add_calendar_months(train_end, 1)
    assert any(fold[1] < fold[2] for fold in folds)
    assert folds[-1][0] > folds[0][0]

    with pytest.raises(ValueError, match="train_months"):
        walk_forward_windows(date(2020, 1, 15), date(2021, 7, 15), train_months=0, test_months=3)

    cohorts = rolling_cohorts(date(2020, 1, 1), date(2022, 1, 1), horizon_months=12, step_months=12)
    assert [c_start for c_start, _ in cohorts] == [date(2020, 1, 1), date(2021, 1, 1)]
    assert all(c_end <= date(2022, 1, 1) for _, c_end in cohorts)
