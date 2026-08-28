"""Certainty equivalent, complexity-penalized adoption gate, plateau selection."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

from src.validation.bootstrap import moving_block_bootstrap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_ALLOWED_GAMMAS = (2.0, 5.0, 10.0)

__all__ = [
    "adoption_passes",
    "bootstrap_tail_passes",
    "certainty_equivalent",
    "cohort_win_rate",
    "contiguous_adopted_plateau",
    "contribution_growth_process_passes",
    "contribution_growth_train_passes",
    "growth_first_process_passes",
    "growth_first_train_passes",
    "select_plateau",
    "wealth_quantile",
    "worst_cohort_passes",
]


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


def growth_first_train_passes(
    *,
    candidate_tw: float,
    baseline_tw: float,
    candidate_mdd: float,
    baseline_mdd: float,
    mdd_slack: float = 0.02,
) -> bool:
    """Growth-first train gate: any strict TW gain wins if MDD stays within ``mdd_slack``.

    Passes iff ``candidate_tw > baseline_tw`` and drawdowns (negative fractions) do
    not deepen beyond ``mdd_slack`` (default 0.02):
    ``candidate_mdd >= baseline_mdd - mdd_slack``. No complexity hurdle applies.

    Raises:
        ValueError: When any input is non-finite or ``mdd_slack`` is negative.
    """
    if not all(math.isfinite(float(value)) for value in (candidate_tw, baseline_tw, candidate_mdd, baseline_mdd)):
        raise ValueError("growth-first train gate requires finite TW and MDD inputs")
    if mdd_slack < 0 or not math.isfinite(mdd_slack):
        raise ValueError(f"mdd_slack must be finite and non-negative, got {mdd_slack!r}")
    return float(candidate_tw) > float(baseline_tw) and float(candidate_mdd) >= float(baseline_mdd) - mdd_slack


def growth_first_process_passes(
    *,
    chosen_test: Sequence[float],
    baseline_test: Sequence[float],
    worst_fold_floor: float = 0.97,
) -> bool:
    """Growth-first process gate: pooled chosen gain with no fold below ``worst_fold_floor``.

    Passes iff ``sum(chosen_test) > sum(baseline_test)`` and every per-fold ratio
    ``chosen/baseline`` is at least ``worst_fold_floor`` (default 0.97); one bad
    fold vetoes adoption.

    Raises:
        ValueError: When the fold sequences mismatch, are empty, contain non-finite
            or non-positive baseline wealths, or ``worst_fold_floor`` is non-finite.
    """
    if len(chosen_test) != len(baseline_test):
        raise ValueError("chosen_test and baseline_test must have equal fold length")
    if len(chosen_test) < 1:
        raise ValueError("growth-first process gate needs at least one fold")
    if not math.isfinite(worst_fold_floor):
        raise ValueError(f"worst_fold_floor must be finite, got {worst_fold_floor!r}")
    chosen = tuple(float(value) for value in chosen_test)
    baseline = tuple(float(value) for value in baseline_test)
    if not all(math.isfinite(value) for value in (*chosen, *baseline)):
        raise ValueError("growth-first process gate requires finite test wealths")
    if any(value <= 0.0 for value in baseline):
        raise ValueError(f"baseline_test must be strictly positive, got {baseline!r}")
    pooled_gain = sum(chosen) > sum(baseline)
    floor_ok = all(
        chosen_value / baseline_value >= worst_fold_floor
        for chosen_value, baseline_value in zip(chosen, baseline, strict=True)
    )
    return pooled_gain and floor_ok

def contribution_growth_train_passes(
    *,
    candidate_tw: float,
    baseline_tw: float,
    candidate_real_gain: float,
    baseline_real_gain: float,
    candidate_xirr_real: float,
    baseline_xirr_real: float,
    candidate_mdd: float,
    baseline_mdd: float,
    mdd_slack: float = 0.02,
) -> bool:
    """Capital-aware train gate for variable external cashflows.

    Passes iff the candidate strictly gains TW and real profit (terminal wealth minus
    contributed capital), its real XIRR is non-inferior, and drawdowns do not deepen
    beyond ``mdd_slack`` (default 0.02): ``candidate_mdd >= baseline_mdd - mdd_slack``.
    The paired real-gain requirement keeps extra invested inflows from winning adoption.

    Raises:
        ValueError: When any input is non-finite or ``mdd_slack`` is negative.
    """
    metrics = (
        candidate_tw,
        baseline_tw,
        candidate_real_gain,
        baseline_real_gain,
        candidate_xirr_real,
        baseline_xirr_real,
        candidate_mdd,
        baseline_mdd,
    )
    if not all(math.isfinite(float(value)) for value in metrics):
        raise ValueError("contribution-growth train gate requires finite TW, gain, XIRR, and MDD inputs")
    if mdd_slack < 0 or not math.isfinite(mdd_slack):
        raise ValueError(f"mdd_slack must be finite and non-negative, got {mdd_slack!r}")
    return (
        float(candidate_tw) > float(baseline_tw)
        and float(candidate_real_gain) > float(baseline_real_gain)
        and float(candidate_xirr_real) >= float(baseline_xirr_real)
        and float(candidate_mdd) >= float(baseline_mdd) - mdd_slack
    )


def contribution_growth_process_passes(
    *,
    chosen_test_tw: Sequence[float],
    baseline_test_tw: Sequence[float],
    chosen_test_real_gain: Sequence[float],
    baseline_test_real_gain: Sequence[float],
    chosen_test_xirr_real: Sequence[float],
    baseline_test_xirr_real: Sequence[float],
    worst_fold_floor: float = 0.97,
) -> bool:
    """Capital-aware process gate over pooled walk-forward folds.

    Passes iff pooled chosen TW and pooled chosen real gain strictly exceed their
    baselines, every fold TW ratio ``chosen/baseline`` reaches ``worst_fold_floor``
    (default 0.97), and every chosen fold's real XIRR is at least its baseline's;
    one bad fold vetoes adoption.

    Raises:
        ValueError: When the fold sequences mismatch in length, are empty, contain
            non-finite metrics or non-positive baseline TWs, or ``worst_fold_floor``
            is non-finite.
    """
    lengths = {
        len(chosen_test_tw),
        len(baseline_test_tw),
        len(chosen_test_real_gain),
        len(baseline_test_real_gain),
        len(chosen_test_xirr_real),
        len(baseline_test_xirr_real),
    }
    if len(lengths) != 1:
        raise ValueError("fold sequences must all have equal length")
    if len(chosen_test_tw) < 1:
        raise ValueError("contribution-growth process gate needs at least one fold")
    if not math.isfinite(worst_fold_floor):
        raise ValueError(f"worst_fold_floor must be finite, got {worst_fold_floor!r}")
    chosen_tw = tuple(float(value) for value in chosen_test_tw)
    baseline_tw = tuple(float(value) for value in baseline_test_tw)
    chosen_gain = tuple(float(value) for value in chosen_test_real_gain)
    baseline_gain = tuple(float(value) for value in baseline_test_real_gain)
    chosen_xirr = tuple(float(value) for value in chosen_test_xirr_real)
    baseline_xirr = tuple(float(value) for value in baseline_test_xirr_real)
    if not all(
        math.isfinite(value)
        for value in (*chosen_tw, *baseline_tw, *chosen_gain, *baseline_gain, *chosen_xirr, *baseline_xirr)
    ):
        raise ValueError("contribution-growth process gate requires finite fold metrics")
    if any(value <= 0.0 for value in baseline_tw):
        raise ValueError(f"baseline_test_tw must be strictly positive, got {baseline_tw!r}")
    pooled_gain = sum(chosen_tw) > sum(baseline_tw) and sum(chosen_gain) > sum(baseline_gain)
    floor_ok = all(
        chosen_value / baseline_value >= worst_fold_floor
        for chosen_value, baseline_value in zip(chosen_tw, baseline_tw, strict=True)
    )
    xirr_ok = all(
        chosen_value >= baseline_value
        for chosen_value, baseline_value in zip(chosen_xirr, baseline_xirr, strict=True)
    )
    return pooled_gain and floor_ok and xirr_ok


def wealth_quantile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated ``q``-quantile of finite wealths.

    Interpolates at position ``q * (n - 1)`` on the sorted values.

    Raises:
        ValueError: When ``q`` lies outside ``(0, 1)``, the sequence is empty,
            or any value is non-finite.
    """
    if len(values) < 1:
        raise ValueError("values must contain at least one observation")
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie in (0, 1), got {q!r}")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("values must be finite")
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def worst_cohort_passes(
    candidate_wealths: Sequence[float],
    baseline_wealths: Sequence[float],
    *,
    floor: float = 0.97,
) -> bool:
    """Worst-cohort gate: every paired ratio reaches ``floor`` (default 0.97).

    Passes iff ``min(candidate_i / baseline_i) >= floor``; one weak cohort vetoes.

    Raises:
        ValueError: When lengths mismatch or are empty, any wealth is non-finite
            or non-positive, or ``floor`` is non-finite.
    """
    if len(candidate_wealths) != len(baseline_wealths):
        raise ValueError("candidate_wealths and baseline_wealths must have equal length")
    if len(candidate_wealths) < 1:
        raise ValueError("worst-cohort gate needs at least one cohort pair")
    if not math.isfinite(floor):
        raise ValueError(f"floor must be finite, got {floor!r}")
    candidate = tuple(float(value) for value in candidate_wealths)
    baseline = tuple(float(value) for value in baseline_wealths)
    if not all(math.isfinite(value) and value > 0.0 for value in (*candidate, *baseline)):
        raise ValueError(f"cohort wealths must be finite and strictly positive, got {candidate!r} vs {baseline!r}")
    return min(cand / base for cand, base in zip(candidate, baseline, strict=True)) >= floor


def bootstrap_tail_passes(
    candidate_wealths: Sequence[float],
    baseline_wealths: Sequence[float],
    *,
    n_paths: int,
    seed: int,
    quantile: float = 0.05,
    floor: float = 0.97,
) -> bool:
    """Seeded bootstrap-tail gate on cohort wealth ratios.

    Resamples the per-cohort ratios with half-window moving blocks; passes iff
    the lower-tail quantile of path means stays at or above ``floor``. The same
    seed reproduces the identical boolean.

    Raises:
        ValueError: When pairing rules fail, ``n_paths`` is below one, or
            ``quantile`` / ``floor`` are invalid.
    """
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile!r}")
    if not math.isfinite(floor):
        raise ValueError(f"floor must be finite, got {floor!r}")
    if len(candidate_wealths) != len(baseline_wealths):
        raise ValueError("candidate_wealths and baseline_wealths must have equal length")
    if len(candidate_wealths) < 1:
        raise ValueError("bootstrap-tail gate needs at least one cohort pair")
    candidate = tuple(float(value) for value in candidate_wealths)
    baseline = tuple(float(value) for value in baseline_wealths)
    if not all(math.isfinite(value) and value > 0.0 for value in (*candidate, *baseline)):
        raise ValueError(f"cohort wealths must be finite and strictly positive, got {candidate!r} vs {baseline!r}")
    ratios = [cand / base for cand, base in zip(candidate, baseline, strict=True)]
    paths = moving_block_bootstrap(
        ratios,
        block_size=max(1, len(ratios) // 2),
        n_paths=n_paths,
        seed=seed,
    )
    path_means = [sum(path) / len(path) for path in paths]
    return wealth_quantile(path_means, quantile) >= floor


def contiguous_adopted_plateau(
    grid: Sequence[float], adopted: Sequence[bool], *, min_width: int = 2
) -> bool:
    """True iff some contiguous adopted run has length >= min_width."""
    if not isinstance(min_width, int) or isinstance(min_width, bool) or min_width < 2:
        raise ValueError(f"min_width must be an integer >= 2, got {min_width!r}")
    if len(grid) == 0 or len(adopted) == 0:
        raise ValueError("grid and adopted must be nonempty")
    if len(grid) != len(adopted):
        raise ValueError(f"grid length {len(grid)} must equal adopted length {len(adopted)}")
    if not all(previous < current for previous, current in pairwise(grid)):
        raise ValueError("grid must be strictly increasing")
    max_run = 0
    current_run = 0
    for flag in adopted:
        if flag:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run >= min_width


def cohort_win_rate(
    candidate_wealths: Sequence[float], baseline_wealths: Sequence[float]
) -> float:
    """Paired win rate: count(candidate > baseline) / n."""
    if len(candidate_wealths) != len(baseline_wealths):
        raise ValueError("candidate_wealths and baseline_wealths must have equal length")
    if len(candidate_wealths) < 1:
        raise ValueError("cohort win rate needs at least one pair")
    candidate = tuple(float(v) for v in candidate_wealths)
    baseline = tuple(float(v) for v in baseline_wealths)
    for value in (*candidate, *baseline):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"wealths must be finite and strictly positive, got {value!r}")
    wins = sum(1 for c, b in zip(candidate, baseline, strict=True) if c > b)
    return wins / len(candidate)


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
