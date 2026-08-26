"""Unit tests for walk-forward folds and rolling cohorts."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.validation.windows import (
    add_calendar_months,
    rolling_cohorts,
    walk_forward_windows,
)


@pytest.mark.parametrize("scenario_id", ["VAL-V01-walk-forward-order"])
def test_val_v01_walk_forward_order(scenario_id: str) -> None:
    """VAL-V01-walk-forward-order"""
    folds = walk_forward_windows(
        date(2020, 1, 15),
        date(2021, 10, 15),
        train_months=12,
        test_months=3,
        embargo_months=0,
    )

    assert folds
    for fold in folds:
        train_start, train_end, test_start, test_end = fold
        assert date(2020, 1, 15) <= train_start <= train_end < test_start <= test_end <= date(2021, 10, 15)
        assert test_end == add_calendar_months(test_start, 3) - timedelta(days=1)
    assert any(fold[1] < fold[2] for fold in folds)
    assert folds[-1][0] > folds[0][0]

    with pytest.raises(ValueError, match="train_months"):
        walk_forward_windows(date(2020, 1, 15), date(2021, 7, 15), train_months=0, test_months=3)

    cohorts = rolling_cohorts(date(2020, 1, 1), date(2022, 1, 1), horizon_months=12, step_months=12)
    assert [c_start for c_start, _ in cohorts] == [date(2020, 1, 1), date(2021, 1, 1)]
    assert all(c_end <= date(2022, 1, 1) for _, c_end in cohorts)


@pytest.mark.parametrize("scenario_id", ["ACR-WIN-full-test-default"])
def test_acr_win_full_test_default(scenario_id: str) -> None:
    """ACR-WIN-full-test-default"""
    folds = walk_forward_windows(
        date(2015, 6, 1),
        date(2026, 6, 30),
        train_months=60,
        test_months=36,
    )

    assert len(folds) == 2
    for fold in folds:
        train_start, train_end, test_start, test_end = fold
        assert train_start < train_end < test_start <= test_end <= date(2026, 6, 30)
        assert test_end == add_calendar_months(test_start, 36) - timedelta(days=1)

    legacy = walk_forward_windows(
        date(2015, 6, 1),
        date(2026, 6, 30),
        train_months=60,
        test_months=36,
        require_full_test=False,
    )
    assert len(legacy) == 3
    assert legacy[-1][2] == legacy[-1][3] == date(2026, 6, 30)
