"""Certainty equivalent, complexity-penalized adoption gate, plateau selection."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_ALLOWED_GAMMAS = (2.0, 5.0, 10.0)

__all__ = ["adoption_passes", "certainty_equivalent", "select_plateau"]


def certainty_equivalent(wealths: Sequence[float], *, gamma: float) -> float:
    """Power CE of strictly positive wealths for gamma in {2, 5, 10}.

    Raises:
        ValueError: When ``gamma`` is unsupported or any wealth is not finite and positive.
    """
    if gamma not in _ALLOWED_GAMMAS:
        raise ValueError(f"unsupported gamma {gamma!r}; expected one of {_ALLOWED_GAMMAS}")
    if len(wealths) < 1:
        raise ValueError("wealths must contain at least one observation")
    power = 1.0 - gamma
    total = 0.0
    for wealth in wealths:
        value = float(wealth)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"wealths must be finite and strictly positive, got {value!r}")
        total += value**power
    return float((total / len(wealths)) ** (1.0 / power))


def adoption_passes(
    ce_candidate: Mapping[float, float],
    ce_baseline: Mapping[float, float],
    *,
    delta0: float,
    modules: int,
) -> bool:
    """Return True iff every gamma beats the baseline by more than ``delta0 * modules``.

    The hurdle applies strictly, so an exact tie at ``1 + delta0 * modules`` fails.

    Raises:
        ValueError: When gammas mismatch or ``delta0`` / ``modules`` are invalid.
    """
    if set(ce_candidate) != set(ce_baseline):
        raise ValueError("candidate and baseline must expose identical gamma keys")
    if delta0 < 0:
        raise ValueError(f"delta0 must be non-negative, got {delta0!r}")
    if isinstance(modules, bool) or not isinstance(modules, int) or modules < 0:
        raise ValueError(f"modules must be a non-negative integer, got {modules!r}")
    threshold = 1.0 + delta0 * modules
    return all(float(ce_candidate[gamma]) / float(ce_baseline[gamma]) > threshold for gamma in ce_candidate)


def select_plateau(grid: Sequence[float], scores: Sequence[float], *, rel_tol: float = 0.05) -> float:
    """Median grid point of the contiguous near-maximum score band.

    The band keeps scores within ``rel_tol`` of the maximum; the lower median grid
    value of that contiguous block wins, and disconnected peaks fail closed.

    Raises:
        ValueError: When the grid is not strictly increasing or the near-max set is disconnected.
    """
    if len(grid) != len(scores):
        raise ValueError(f"grid length {len(grid)} must equal scores length {len(scores)}")
    if len(grid) < 1:
        raise ValueError("grid must contain at least one point")
    if not all(previous < current for previous, current in pairwise(grid)):
        raise ValueError("grid must be strictly increasing")
    if not 0.0 < rel_tol < 1.0:
        raise ValueError(f"rel_tol must lie in (0, 1), got {rel_tol!r}")
    cutoff = max(scores) * (1.0 - rel_tol)
    band = [index for index, score in enumerate(scores) if score >= cutoff]
    if band != list(range(band[0], band[-1] + 1)):
        raise ValueError("near-maximum scores form a disconnected band")
    return float(grid[band[(len(band) - 1) // 2]])
