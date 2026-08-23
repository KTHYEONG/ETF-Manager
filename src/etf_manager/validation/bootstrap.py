"""Circular moving-block bootstrap of a wealth vector."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["moving_block_bootstrap"]


def moving_block_bootstrap(
    values: Sequence[float],
    *,
    block_size: int,
    n_paths: int,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    """Resample ``values`` with wrapping blocks; deterministic in ``seed``.

    Block starts are uniform over ``0..n-1`` and blocks wrap circularly, so each
    path concatenates whole blocks until it holds exactly ``n`` samples.

    Raises:
        ValueError: When ``block_size`` or ``n_paths`` is invalid or ``values`` is empty.
    """
    n = len(values)
    if n < 1:
        raise ValueError("values must contain at least one observation")
    if not 1 <= block_size <= n:
        raise ValueError(f"block_size must lie in [1, {n}], got {block_size}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    rng = random.Random(seed)  # noqa: S311 -- deterministic seeded resampling, not cryptographic
    paths: list[tuple[float, ...]] = []
    for _ in range(n_paths):
        sampled: list[float] = []
        while len(sampled) < n:
            block_start = rng.randrange(n)
            sampled.extend(float(values[(block_start + offset) % n]) for offset in range(block_size))
        paths.append(tuple(sampled[:n]))
    return tuple(paths)
