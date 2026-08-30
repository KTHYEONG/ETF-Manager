"""Thesis fundamental registry: PIT series, falsifier, and ingest wiring."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call,unused-ignore,type-arg"

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from src.policy.thesis import ThesisId

if TYPE_CHECKING:
    from src.data.secrets import ProviderSecrets
    from src.data.settings import DataSettings
    from src.data.storage import DatasetArtifact


@dataclass(frozen=True, slots=True)
class FalsifierSpec:
    """Single falsifier contract for a fundamental series."""

    id: str
    series_id: str
    threshold_pct: float
    consecutive_periods: int
    metric: str = "yoy_pct"

    @property
    def falsifier_id(self) -> str:
        return self.id

    @property
    def name(self) -> str:
        return self.id


class FalsifierCollection(tuple):  # type: ignore[type-arg]
    """Tuple of FalsifierSpec that also supports dict-like lookup."""

    def __new__(cls, specs):  # type: ignore[no-untyped-def]
        return super().__new__(cls, specs)  # type: ignore[arg-type]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(getattr(s, "id", None) == item for s in self)
        return super().__contains__(item)  # type: ignore[arg-type]

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            for spec in self:
                if getattr(spec, "id", None) == key:
                    return spec
            raise KeyError(key)
        return super().__getitem__(key)  # type: ignore[call-arg]

    def get(self, key: str, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def keys(self):
        return [getattr(s, "id", None) for s in self]


@dataclass(frozen=True, slots=True)
class ThesisFundamentalsSpec:
    thesis_id: ThesisId
    primary_series_id: str
    secondary_series_ids: tuple[str, ...]
    falsifiers: FalsifierCollection
    min_history_periods: int
    lookback_periods: int


@dataclass(frozen=True, slots=True)
class ValuationSpec:
    vehicle_ticker: str
    benchmark_ticker: str
    trailing_sessions: int
    rich_percentile: float
    cheap_percentile: float
    min_sessions: int
    return_lookback_sessions: int
    collapse_return_pct: float


@dataclass(frozen=True, slots=True)
class CrowdingSpec:
    vehicle_ticker: str
    top_n: int
    concentrated_hhi_threshold: float
    concentrated_top5_pct: float


@dataclass(frozen=True, slots=True)
class ExposureNote:
    isin: str | None = None
    cusip: str | None = None
    role: str = ""
    note: str = ""


@dataclass(frozen=True, slots=True)
class PuritySpec:
    vehicle_ticker: str
    incumbent_ticker: str
    pure_min_pct: float
    impure_max_pct: float
    exposure_notes: tuple[ExposureNote, ...]


def load_purity_spec(*, thesis_id: ThesisId, path: Path | None = None) -> PuritySpec | None:
    file_path = Path(path) if path is not None else _default_registry_path(thesis_id)
    payload = _load_json_payload(file_path)
    if payload is None:
        return None
    raw = payload.get("purity")
    if not isinstance(raw, dict):
        return None
    try:
        vehicle_ticker = str(raw["vehicle_ticker"]).strip()
        incumbent_ticker = str(raw["incumbent_ticker"]).strip()
        pure_min_pct = float(raw["pure_min_pct"])
        impure_max_pct = float(raw["impure_max_pct"])
        raw_notes = raw.get("exposure_notes")
        if not isinstance(raw_notes, list):
            return None
        if not vehicle_ticker or not incumbent_ticker:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    notes: list[ExposureNote] = []
    for entry in raw_notes:
        if not isinstance(entry, dict):
            continue
        raw_isin = entry.get("isin")
        raw_cusip = entry.get("cusip")
        isin: str | None = None
        cusip: str | None = None
        if raw_isin is not None:
            s = str(raw_isin).strip()
            if s:
                isin = s
        if raw_cusip is not None:
            s = str(raw_cusip).strip()
            if s:
                cusip = s
        role = str(entry.get("role", "")).strip()
        note = str(entry.get("note", "")).strip()
        if not role or not note:
            continue
        if isin is None and cusip is None:
            continue
        notes.append(ExposureNote(isin=isin, cusip=cusip, role=role, note=note))
    try:
        return PuritySpec(
            vehicle_ticker=vehicle_ticker,
            incumbent_ticker=incumbent_ticker,
            pure_min_pct=float(pure_min_pct),
            impure_max_pct=float(impure_max_pct),
            exposure_notes=tuple(notes),
        )
    except (TypeError, ValueError):
        return None


def _default_registry_path(thesis_id: ThesisId) -> Path:
    return Path("configs/data/thesis_fundamentals") / f"{thesis_id.value}.json"


def load_thesis_fundamentals(*, thesis_id: ThesisId, path: Path | None = None) -> ThesisFundamentalsSpec:
    """Load and validate the fundamental registry for a thesis."""
    file_path = Path(path) if path is not None else _default_registry_path(thesis_id)
    try:
        text = file_path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise FileNotFoundError(f"thesis fundamentals registry missing at {file_path.as_posix()}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"fundamentals payload must be object: {file_path.as_posix()}")

    # thesis_id field
    raw_tid = payload.get("thesis_id", thesis_id.value)
    try:
        tid = ThesisId(str(raw_tid))
    except ValueError as exc:
        raise ValueError(f"unknown thesis_id {raw_tid!r} in {file_path.as_posix()}") from exc
    if tid is not thesis_id:
        raise ValueError(f"registry thesis_id {tid!r} mismatches requested {thesis_id!r}")

    primary_series_id = str(payload.get("primary_series_id", ""))
    if not primary_series_id:
        raise ValueError(f"primary_series_id missing in {file_path.as_posix()}")

    # secondary ids
    secondary_ids: list[str] = []
    raw_secondary = payload.get("secondary_series_ids")
    if raw_secondary is None:
        raw_secondary = payload.get("secondary_series")
    if raw_secondary is not None:
        if isinstance(raw_secondary, list):
            for entry in raw_secondary:
                if isinstance(entry, str):
                    secondary_ids.append(entry)
                elif isinstance(entry, dict) and "series_id" in entry:
                    secondary_ids.append(str(entry["series_id"]))
        elif isinstance(raw_secondary, dict):
            secondary_ids.extend(str(k) for k in raw_secondary)

    # falsifiers
    raw_falsifiers = payload.get("falsifiers", {})
    falsifier_specs: list[FalsifierSpec] = []
    if isinstance(raw_falsifiers, dict):
        for fid, cfg in raw_falsifiers.items():
            if not isinstance(cfg, dict):
                continue
            falsifier_specs.append(
                FalsifierSpec(
                    id=str(fid),
                    series_id=str(cfg.get("series_id", primary_series_id)),
                    threshold_pct=float(cfg.get("threshold_pct", 0.0)),
                    consecutive_periods=int(cfg.get("consecutive_periods", 2)),
                    metric=str(cfg.get("metric", "yoy_pct")),
                )
            )
    elif isinstance(raw_falsifiers, list):
        for cfg in raw_falsifiers:
            if not isinstance(cfg, dict):
                continue
            fid = str(cfg.get("id") or cfg.get("falsifier_id") or cfg.get("name") or "")
            if not fid:
                continue
            falsifier_specs.append(
                FalsifierSpec(
                    id=fid,
                    series_id=str(cfg.get("series_id", primary_series_id)),
                    threshold_pct=float(cfg.get("threshold_pct", 0.0)),
                    consecutive_periods=int(cfg.get("consecutive_periods", 2)),
                    metric=str(cfg.get("metric", "yoy_pct")),
                )
            )

    # periods from structural or top-level
    structural = payload.get("structural", {}) if isinstance(payload.get("structural"), dict) else {}
    min_history = payload.get("min_history_periods", structural.get("min_history_periods", 8))
    lookback = payload.get("lookback_periods", structural.get("lookback_periods", 20))
    try:
        min_history_periods = int(min_history)
        lookback_periods = int(lookback)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid period fields in {file_path.as_posix()}: {exc}") from exc

    return ThesisFundamentalsSpec(
        thesis_id=tid,
        primary_series_id=primary_series_id,
        secondary_series_ids=tuple(secondary_ids),
        falsifiers=FalsifierCollection(falsifier_specs),
        min_history_periods=min_history_periods,
        lookback_periods=lookback_periods,
    )


def _load_json_payload(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_valuation_spec(*, thesis_id: ThesisId, path: Path | None = None) -> ValuationSpec | None:
    file_path = Path(path) if path is not None else _default_registry_path(thesis_id)
    payload = _load_json_payload(file_path)
    if payload is None:
        return None
    raw = payload.get("valuation")
    if not isinstance(raw, dict):
        return None
    try:
        return ValuationSpec(
            vehicle_ticker=str(raw["vehicle_ticker"]),
            benchmark_ticker=str(raw["benchmark_ticker"]),
            trailing_sessions=int(raw["trailing_sessions"]),
            rich_percentile=float(raw["rich_percentile"]),
            cheap_percentile=float(raw["cheap_percentile"]),
            min_sessions=int(raw["min_sessions"]),
            return_lookback_sessions=int(raw["return_lookback_sessions"]),
            collapse_return_pct=float(raw["collapse_return_pct"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_crowding_spec(*, thesis_id: ThesisId, path: Path | None = None) -> CrowdingSpec | None:
    file_path = Path(path) if path is not None else _default_registry_path(thesis_id)
    payload = _load_json_payload(file_path)
    if payload is None:
        return None
    raw = payload.get("crowding")
    if not isinstance(raw, dict):
        return None
    try:
        return CrowdingSpec(
            vehicle_ticker=str(raw["vehicle_ticker"]),
            top_n=int(raw["top_n"]),
            concentrated_hhi_threshold=float(raw["concentrated_hhi_threshold"]),
            concentrated_top5_pct=float(raw["concentrated_top5_pct"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def fundamental_series_ids(spec: ThesisFundamentalsSpec) -> tuple[str, ...]:
    """Sorted unique series ids for the thesis."""
    ids = {spec.primary_series_id, *spec.secondary_series_ids}
    # remove empty strings
    ids = {s for s in ids if s}
    return tuple(sorted(ids))


def fetch_and_persist_thesis_fundamentals(
    *,
    start: date,
    end: date,
    settings: DataSettings,
    secrets: ProviderSecrets,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch all registry fundamentals via single MACRO partition."""
    registry_dir = Path("configs/data/thesis_fundamentals")
    if not registry_dir.is_dir():
        raise FileNotFoundError(f"fundamentals registry dir missing: {registry_dir.as_posix()}")
    series_set: set[str] = set()
    for path in sorted(registry_dir.glob("*.json")):
        try:
            tid = ThesisId(path.stem)
        except ValueError:
            continue
        spec = load_thesis_fundamentals(thesis_id=tid, path=path)
        series_set.update(fundamental_series_ids(spec))
    if not series_set:
        raise ValueError("no thesis fundamental series ids found")
    series_ids = tuple(sorted(series_set))
    from src.data.fetch import fetch_and_persist_macro

    return fetch_and_persist_macro(
        series_ids, start, end, secrets=secrets, settings=settings, client=client, retain_other_series=True
    )
