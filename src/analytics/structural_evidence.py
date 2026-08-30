"""Structural evidence from PIT fundamental series (Track F)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.analytics.thesis_evidence import EvidenceSlot
from src.data.catalog import load_visible
from src.data.pit import AVAILABLE_AT, TS_DTYPE
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.thesis_fundamentals import FalsifierSpec, ThesisFundamentalsSpec
from src.policy.thesis import ThesisSpec

__all__ = [
    "compute_structural_slot",
    "detect_yoy_regime_change",
    "evaluate_falsifier_slowdown",
    "pit_macro_series_levels",
    "resolve_primary_falsifier",
    "yoy_growth_pct",
]


def pit_macro_series_levels(*, macro: pl.DataFrame, series_id: str, as_of: datetime) -> pl.DataFrame:
    """PIT-resolve macro to latest vintage per observation_date for series_id."""
    if macro.is_empty():
        return macro.clear()
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    cutoff = as_of.astimezone(UTC)

    # ensure available_at
    frame = macro
    if AVAILABLE_AT not in frame.columns:
        if "release_date" in frame.columns:
            frame = frame.with_columns(pl.col("release_date").cast(TS_DTYPE).alias(AVAILABLE_AT))
        else:
            raise ValueError("macro frame missing release_date/available_at")

    # ensure observation_date is Date
    # filter by series_id and visibility
    filtered = frame.filter(
        (pl.col("series_id") == series_id) & (pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
    )
    if filtered.is_empty():
        return filtered.sort("observation_date")
    # sort by observation_date and available_at, keep last per observation_date
    ordered = filtered.sort(["observation_date", AVAILABLE_AT], maintain_order=True)
    deduped = ordered.filter(pl.struct(["observation_date"]).is_last_distinct())
    return deduped.sort("observation_date")


def yoy_growth_pct(levels: pl.DataFrame, *, periods: int) -> pl.DataFrame:
    """Compute YoY % on levels sorted by observation_date."""
    if levels.is_empty():
        # return empty with yoy_pct column
        return levels.with_columns(pl.lit(None, dtype=pl.Float64).alias("yoy_pct"))
    sorted_levels = levels.sort("observation_date")
    if "value" not in sorted_levels.columns:
        raise ValueError("levels missing value column")
    result = sorted_levels.with_columns(
        ((pl.col("value") / pl.col("value").shift(periods) - 1) * 100).alias("yoy_pct")
    )
    return result


def evaluate_falsifier_slowdown(*, yoy: pl.DataFrame, threshold_pct: float, consecutive_periods: int) -> bool:
    """Active if last consecutive_periods YoY values are strictly below threshold."""
    if yoy.is_empty() or "yoy_pct" not in yoy.columns:
        return False
    # filter non-null yoy_pct
    valid = yoy.filter(pl.col("yoy_pct").is_not_null() & pl.col("yoy_pct").is_finite())
    if valid.height < consecutive_periods:
        return False
    tail = valid.tail(consecutive_periods).get_column("yoy_pct").to_list()
    return all(float(v) < float(threshold_pct) for v in tail)


def detect_yoy_regime_change(
    *, yoy: pl.DataFrame, lookback_periods: int, min_positive_periods: int
) -> tuple[str, date | None]:
    """Detect crossing from >=0 to <0 after sustained positives."""
    if yoy.is_empty() or "yoy_pct" not in yoy.columns:
        return ("unknown", None)
    sorted_yoy = yoy.sort("observation_date")
    # consider only rows with finite yoy_pct for detection window
    valid = sorted_yoy.filter(pl.col("yoy_pct").is_not_null() & pl.col("yoy_pct").is_finite())
    if valid.is_empty():
        return ("unknown", None)
    # take trailing lookback window
    window = valid.tail(lookback_periods) if lookback_periods > 0 else valid
    yoy_vals = window.get_column("yoy_pct").to_list()
    obs_dates = window.get_column("observation_date").to_list()
    # normalize dates
    # detect first crossing where prior min_positive_periods are >=0
    for idx in range(min_positive_periods, len(yoy_vals)):
        if float(yoy_vals[idx]) < 0:
            # check preceding min_positive_periods are >=0
            prior = yoy_vals[idx - min_positive_periods : idx]
            if all(float(v) >= 0 for v in prior):
                # found
                cp_date = obs_dates[idx]
                # normalize to date
                if isinstance(cp_date, datetime):
                    cp_date = cp_date.date()
                return ("slowdown", cp_date)
    # no crossing: regime from latest yoy
    latest = float(yoy_vals[-1])
    if latest >= 0:
        return ("expansion", None)
    if latest < 0:
        return ("slowdown", None)
    return ("unknown", None)


def resolve_primary_falsifier(*, spec: ThesisFundamentalsSpec) -> FalsifierSpec | None:
    """Prefer falsifier whose series_id equals primary_series_id, else first."""
    if not spec.falsifiers:
        return None
    for fals in spec.falsifiers:
        if fals.series_id == spec.primary_series_id:
            return fals  # type: ignore[no-any-return]
    return spec.falsifiers[0]  # type: ignore[no-any-return]


def _infer_periods(levels: pl.DataFrame) -> int:
    """Infer YoY lag from observation_date spacing."""
    if levels.height < 2 or "observation_date" not in levels.columns:
        return 4
    sorted_levels = levels.sort("observation_date")
    dates = sorted_levels.get_column("observation_date").to_list()
    # compute median days diff
    diffs: list[int] = []
    for i in range(1, len(dates)):
        d0 = dates[i - 1]
        d1 = dates[i]
        if isinstance(d0, datetime):
            d0 = d0.date()
        if isinstance(d1, datetime):
            d1 = d1.date()
        diffs.append((d1 - d0).days)
    if not diffs:
        return 4
    diffs.sort()
    median = diffs[len(diffs) // 2]
    # monthly ~30 days, quarterly ~90 days
    if median <= 45:
        return 12
    return 4


def compute_structural_slot(*, thesis: ThesisSpec, settings: DataSettings, as_of: datetime) -> EvidenceSlot:
    """Compute structural EvidenceSlot from PIT fundamentals."""
    # load registry
    try:
        from src.data.thesis_fundamentals import load_thesis_fundamentals

        spec = load_thesis_fundamentals(thesis_id=thesis.id)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    # load visible macro
    try:
        macro_visible = load_visible(settings, Dataset.MACRO, as_of)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    # PIT levels for primary
    try:
        levels = pit_macro_series_levels(macro=macro_visible, series_id=spec.primary_series_id, as_of=as_of)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    if levels.is_empty():
        return EvidenceSlot(
            status="insufficient_data",
            summary=f"no macro series {spec.primary_series_id} at as_of",
            metrics={"error": f"no series {spec.primary_series_id}"},
        )
    if levels.height < spec.min_history_periods:
        return EvidenceSlot(
            status="insufficient_data",
            summary=f"insufficient history {levels.height} < {spec.min_history_periods}",
            metrics={"error": "insufficient_history", "observed": levels.height, "required": spec.min_history_periods},
        )

    # infer periods: try registry frequency, else spacing
    periods = 4
    # try reading raw json for frequency
    try:
        reg_path = Path("configs/data/thesis_fundamentals") / f"{thesis.id.value}.json"
        if reg_path.is_file():
            payload = json.loads(reg_path.read_text(encoding="utf-8"))
            freq = payload.get("primary_frequency")
            if freq == "M":
                periods = 12
            elif freq == "Q":
                periods = 4
            else:
                # check structural yoy_lag map
                yoy_lag = payload.get("structural", {}).get("yoy_lag", {})
                if isinstance(yoy_lag, dict) and freq in yoy_lag:
                    periods = int(yoy_lag[freq])
                else:
                    periods = _infer_periods(levels)
        else:
            periods = _infer_periods(levels)
    except Exception:
        periods = _infer_periods(levels)

    # yoy
    yoy = yoy_growth_pct(levels, periods=periods)
    # filter valid yoy
    valid_yoy = yoy.filter(pl.col("yoy_pct").is_not_null() & pl.col("yoy_pct").is_finite())
    if valid_yoy.is_empty():
        return EvidenceSlot(
            status="insufficient_data",
            summary="no yoy observations",
            metrics={"error": "no_yoy"},
        )
    last_yoy = float(valid_yoy.tail(1).get_column("yoy_pct").to_list()[0])

    # falsifier via resolve_primary_falsifier
    falsifier_active = False
    falsifier_id = "capex_structural_slowdown"
    threshold_pct = 0.0
    consecutive = 2
    try:
        resolved = resolve_primary_falsifier(spec=spec)
        if resolved is not None:
            falsifier_id = str(resolved.id)
            threshold_pct = float(resolved.threshold_pct)
            consecutive = int(resolved.consecutive_periods)
        falsifier_active = evaluate_falsifier_slowdown(yoy=yoy, threshold_pct=threshold_pct, consecutive_periods=consecutive)
    except Exception:
        falsifier_active = False

    # regime change
    try:
        # read min_positive_periods from registry if present
        min_positive = 4
        try:
            reg_path2 = Path("configs/data/thesis_fundamentals") / f"{thesis.id.value}.json"
            if reg_path2.is_file():
                payload2 = json.loads(reg_path2.read_text(encoding="utf-8"))
                min_positive = int(payload2.get("structural", {}).get("min_positive_periods", 4))
        except Exception:
            min_positive = 4
        regime, cp_date = detect_yoy_regime_change(yoy=yoy, lookback_periods=spec.lookback_periods, min_positive_periods=min_positive)
    except Exception:
        regime, cp_date = "unknown", None

    change_point_str = cp_date.isoformat() if isinstance(cp_date, date) else ""
    metrics: dict[str, float | int | str] = {
        "primary_series_id": spec.primary_series_id,
        "primary_yoy_pct": float(last_yoy),
        f"falsifier_{falsifier_id}_active": bool(falsifier_active),
        "change_point_date": change_point_str,
        "regime": str(regime),
    }
    summary = f"fundamental: {spec.primary_series_id} yoy {last_yoy:.2f}% regime {regime} falsifier {falsifier_active} change {change_point_str or 'none'}"
    return EvidenceSlot(status="computed", summary=summary, metrics=metrics)
