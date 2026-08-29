"""Retention prune planning and application."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.catalog import latest_artifact
from src.data.pipeline import persist_ingest
from src.data.retention import apply_prune, plan_prune
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload

_RETRIEVED_EARLY = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
_RETRIEVED_LATE = datetime(2024, 2, 2, 5, 0, tzinfo=UTC)


def _fx_frame(dates: list[date], rates: list[float], retrieved_at: datetime) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    return pl.DataFrame(
        {
            "date": list(dates),
            "usdkrw": list(rates),
            "source": ["synthetic"] * len(dates),
            "retrieved_at": [retrieved_at] * len(dates),
        },
        schema=dict(spec.columns),
    )


def _payload(retrieved_at: datetime) -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="usdkrw/daily",
        request_params={"interval": "daily"},
        retrieved_at=retrieved_at,
        extension="json",
        content=b"{}",
    )


def test_plan_prune_keeps_latest_drops_stale_and_nport_mirrors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "retention"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    early_art = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload(_RETRIEVED_EARLY),
        settings,
    )
    late_art = persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload(_RETRIEVED_LATE),
        settings,
    )

    nport_zip = settings.resolved_data_root() / "raw" / "sec" / "nport" / "2019q4.zip"
    nport_zip.parent.mkdir(parents=True, exist_ok=True)
    nport_zip.write_bytes(b"fake nport zip mirror")

    plan = plan_prune(settings, migrate_results_layout=False)

    assert early_art.manifest_path in plan.to_delete
    assert early_art.normalized_path in plan.to_delete
    assert late_art.manifest_path in plan.retained_manifests
    assert late_art.normalized_path in plan.retained_parquets
    assert nport_zip in plan.to_delete

    dry_report = apply_prune(plan, dry_run=True)
    assert dry_report.dry_run is True
    assert early_art.manifest_path.is_file()
    assert nport_zip.is_file()

    apply_report = apply_prune(plan, dry_run=False)
    assert apply_report.dry_run is False
    assert not early_art.manifest_path.is_file()
    assert not nport_zip.is_file()
    assert late_art.manifest_path.is_file()

    latest = latest_artifact(settings, Dataset.FX)
    assert latest.manifest.normalized_sha256 == late_art.manifest.normalized_sha256
