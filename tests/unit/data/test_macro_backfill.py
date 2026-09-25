"""Unit tests for unrevised MACRO backfill before the first ALFRED vintage."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from src.cli import main
from src.data.calendar import TradingCalendar, load_calendar
from src.data.macro_backfill import (
    UnrevisedMacroSeries,
    fetch_and_persist_macro_backfill,
    load_unrevised_macro_series,
)
from src.data.merge import PriorPartitionUntrustedError
from src.data.pit import as_of
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, DatasetSpec, spec_for
from src.data.secrets import ProviderSecrets
from src.data.settings import DataSettings
from src.data.storage import DataStore, DatasetArtifact, RawPayload

_SECRETS = ProviderSecrets(tiingo_api="wire-tiingo-token", fred_api="wire-fred-key", ecos_api="wire-ecos-key")
_FIRST_VINTAGE = date(2012, 6, 1)
_FRIDAY_BEFORE_HOLIDAY = date(2012, 5, 25)
_HOLIDAY = date(2012, 5, 28)
_SESSION_AFTER_HOLIDAY = date(2012, 5, 29)
_SESSION_AFTER_LAG = date(2012, 5, 30)
_DOTCOM_START = date(2000, 1, 3)
_BACKFILL_OBSERVATIONS = (
    ("2000-01-03", "26.15"),
    ("2000-01-04", "27.30"),
    ("2000-01-05", "25.44"),
    ("2012-06-01", "99.99"),
)


class _FredTransport:
    """Mock FRED observations transport that records every requested series id."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.served: list[str] = []
        self._bodies = bodies

    def __call__(self, request: httpx.Request) -> httpx.Response:
        series_id = str(request.url.params.get("series_id", ""))
        self.served.append(series_id)
        body = self._bodies.get(series_id)
        return httpx.Response(200 if body is not None else 404, content=body or b"{}")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def _calendar() -> TradingCalendar:
    return load_calendar("XNYS")


def _fresh_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DataSettings:
    monkeypatch.chdir(tmp_path)
    return DataSettings(data_root="data")


def _fred_body(observations: tuple[tuple[str, str], ...]) -> bytes:
    document = {"observations": [{"date": day, "value": value} for day, value in observations]}
    return json.dumps(document).encode("utf-8")


def _vintage_rows(series_id: str) -> list[tuple[str, date, datetime, float]]:
    calendar = _calendar()
    return [
        (series_id, day, calendar.close_ts(calendar.next_session(day)), value)
        for day, value in ((date(2012, 6, 1), 25.0), (date(2012, 6, 4), 26.5))
    ]


def _vintage_frame(rows: list[tuple[str, date, datetime, float]]) -> pl.DataFrame:
    spec = spec_for(Dataset.MACRO)
    return pl.DataFrame(
        {
            "series_id": [series_id for series_id, _, _, _ in rows],
            "observation_date": [day for _, day, _, _ in rows],
            "release_date": [release for _, _, release, _ in rows],
            "value": [value for _, _, _, value in rows],
        },
        schema=dict(spec.columns),
    )


def _persist_seed(settings: DataSettings, frame: pl.DataFrame, series_ids: tuple[str, ...]) -> None:
    persist_ingest(
        frame,
        Dataset.MACRO,
        RawPayload(
            provider="alfred",
            endpoint=f"series/observations/{'+'.join(series_ids)}",
            request_params={"series_ids": list(series_ids)},
            retrieved_at=datetime(2012, 7, 1, tzinfo=UTC),
            extension="json",
            content=b"seed",
        ),
        settings,
    )


def _seed_macro(settings: DataSettings, series_ids: tuple[str, ...] = ("BAA10Y", "VIXCLS")) -> pl.DataFrame:
    rows = [row for series_id in series_ids for row in _vintage_rows(series_id)]
    frame = _vintage_frame(rows)
    _persist_seed(settings, frame, series_ids)
    return frame


def _stored(settings: DataSettings, artifact: DatasetArtifact, spec: DatasetSpec) -> pl.DataFrame:
    return DataStore(settings).read_normalized(artifact, spec)


def _backfill(
    settings: DataSettings,
    series_ids: tuple[str, ...],
    start: date,
    body: bytes,
) -> DatasetArtifact:
    transport = _FredTransport(dict.fromkeys(series_ids, body))
    with transport.client() as http:
        return fetch_and_persist_macro_backfill(
            series_ids, start, secrets=_SECRETS, settings=settings, client=http
        )


def test_backfill_stops_at_first_vintage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Backfilled rows stay strictly before the first vintage and never touch prior rows."""
    settings = _fresh_settings(monkeypatch, tmp_path)
    spec = spec_for(Dataset.MACRO)
    seeded = _seed_macro(settings)

    artifact = _backfill(settings, ("VIXCLS",), _DOTCOM_START, _fred_body(_BACKFILL_OBSERVATIONS))
    stored = _stored(settings, artifact, spec)
    backfilled = stored.filter(pl.col("series_id") == "VIXCLS")
    assert sorted(day for day in backfilled.get_column("observation_date").to_list() if day < _FIRST_VINTAGE) == [
        date(2000, 1, 3),
        date(2000, 1, 4),
        date(2000, 1, 5),
    ]
    # The observation on the cutoff date stays the prior vintage row instead of duplicating.
    cutoff_rows = backfilled.filter(pl.col("observation_date") == _FIRST_VINTAGE)
    assert cutoff_rows.height == 1
    assert cutoff_rows.get_column("value").to_list() == [25.0]
    assert stored.join(seeded, on=list(spec.key), how="semi").height == seeded.height


def test_release_lag_is_causal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Friday observation with lag 1 is released at the next session close, never earlier."""
    settings = _fresh_settings(monkeypatch, tmp_path)
    spec = spec_for(Dataset.MACRO)
    calendar = _calendar()
    _seed_macro(settings)
    body = _fred_body(((str(_FRIDAY_BEFORE_HOLIDAY), "30.0"),))

    artifact = _backfill(settings, ("VIXCLS",), _FRIDAY_BEFORE_HOLIDAY, body)
    stored = _stored(settings, artifact, spec)
    row = stored.filter(pl.col("observation_date") == _FRIDAY_BEFORE_HOLIDAY).row(0, named=True)
    assert row["release_date"] == calendar.close_ts(_SESSION_AFTER_HOLIDAY)
    assert row["value"] == 30.0

    at_friday_close = as_of(stored, spec, calendar.close_ts(_FRIDAY_BEFORE_HOLIDAY))
    assert _FRIDAY_BEFORE_HOLIDAY not in at_friday_close.get_column("observation_date").to_list()
    at_release_close = as_of(stored, spec, calendar.close_ts(_SESSION_AFTER_HOLIDAY))
    assert _FRIDAY_BEFORE_HOLIDAY in at_release_close.get_column("observation_date").to_list()


def test_non_session_observation_rolls_forward_before_lag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A holiday-dated print releases after the next session plus the lag, keeping its vendor date."""
    settings = _fresh_settings(monkeypatch, tmp_path)
    spec = spec_for(Dataset.MACRO)
    calendar = _calendar()
    assert not calendar.is_session(_HOLIDAY)
    _seed_macro(settings)
    body = _fred_body(((str(_HOLIDAY), "18.75"),))

    artifact = _backfill(settings, ("VIXCLS",), _HOLIDAY, body)
    stored = _stored(settings, artifact, spec)
    holiday_rows = stored.filter(pl.col("observation_date") == _HOLIDAY)
    assert holiday_rows.height == 1
    assert holiday_rows.row(0, named=True)["release_date"] == calendar.close_ts(_SESSION_AFTER_LAG)


def test_other_series_survive_backfill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Backfilling one allowlisted series leaves every other series row identical."""
    settings = _fresh_settings(monkeypatch, tmp_path)
    spec = spec_for(Dataset.MACRO)
    _seed_macro(settings)
    expected_baa = _vintage_frame(_vintage_rows("BAA10Y"))

    artifact = _backfill(settings, ("VIXCLS",), _DOTCOM_START, _fred_body(_BACKFILL_OBSERVATIONS))
    stored = _stored(settings, artifact, spec)
    assert stored.filter(pl.col("series_id") == "BAA10Y").select(*spec.columns).equals(expected_baa)
    assert artifact.manifest.prior_manifest_sha256 is not None
    assert artifact.manifest.row_count > expected_baa.height


def test_unlisted_or_absent_series_is_rejected_without_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rejections happen before any HTTP call and leave no new manifest behind."""
    settings = _fresh_settings(monkeypatch, tmp_path)
    _seed_macro(settings, ("VIXCLS",))
    manifests = settings.resolved_data_root() / "manifests" / "macro"
    before_manifests = sorted(manifests.glob("*.json"))
    body = _fred_body(_BACKFILL_OBSERVATIONS)
    transport = _FredTransport({"VIXCLS": body, "BAA10Y": body})

    with transport.client() as http:
        with pytest.raises(ValueError, match="not allowlisted"):
            fetch_and_persist_macro_backfill(
                ("T10YIE",), _DOTCOM_START, secrets=_SECRETS, settings=settings, client=http
            )
        with pytest.raises(ValueError, match="no MACRO rows"):
            fetch_and_persist_macro_backfill(
                ("BAA10Y",), _DOTCOM_START, secrets=_SECRETS, settings=settings, client=http
            )
        with pytest.raises(ValueError, match="is not before"):
            fetch_and_persist_macro_backfill(
                ("VIXCLS",), _FIRST_VINTAGE, secrets=_SECRETS, settings=settings, client=http
            )
        with pytest.raises(ValueError, match="duplicate series ids"):
            fetch_and_persist_macro_backfill(
                ("VIXCLS", "VIXCLS"), _DOTCOM_START, secrets=_SECRETS, settings=settings, client=http
            )
    assert transport.served == []
    assert sorted(manifests.glob("*.json")) == before_manifests


def test_missing_observations_persist_as_null_gaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A '.' observation persists as an explicit null gap row instead of being dropped or filled."""
    settings = _fresh_settings(monkeypatch, tmp_path)
    spec = spec_for(Dataset.MACRO)
    assert spec.missing_policy.name == "EXPLICIT_GAP"
    _seed_macro(settings)
    body = _fred_body((("2000-01-03", "."), ("2000-01-04", "26.15")))

    artifact = _backfill(settings, ("VIXCLS",), _DOTCOM_START, body)
    backfilled = _stored(settings, artifact, spec).filter(pl.col("observation_date") < _FIRST_VINTAGE)
    assert backfilled.height == 2
    assert backfilled.filter(pl.col("observation_date") == date(2000, 1, 3)).get_column("value").to_list() == [None]
    assert backfilled.filter(pl.col("observation_date") == date(2000, 1, 4)).get_column("value").to_list() == [26.15]


def test_backfill_manifest_records_lineage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The manifest names the backfill endpoint, requested series, first-vintage cutoff, and lag."""
    settings = _fresh_settings(monkeypatch, tmp_path)
    _seed_macro(settings)

    artifact = _backfill(settings, ("VIXCLS",), _DOTCOM_START, _fred_body(_BACKFILL_OBSERVATIONS))
    params = artifact.manifest.request_params
    assert artifact.manifest.provider == "fred"
    assert params["series_ids"] == ["VIXCLS"]
    assert params["observation_start"] == _DOTCOM_START.isoformat()
    assert params["first_vintage_cutoff"] == {"VIXCLS": _FIRST_VINTAGE.isoformat()}
    assert params["release_lag_sessions"] == {"VIXCLS": "1"}


def test_untrusted_prior_partition_stops_backfill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A MACRO partition that fails verification aborts instead of publishing a thinner one."""
    from src.data.catalog import clear_catalog_frame_cache, latest_artifact

    settings = _fresh_settings(monkeypatch, tmp_path)
    _seed_macro(settings, ("VIXCLS",))
    latest_artifact(settings, Dataset.MACRO).normalized_path.unlink()
    clear_catalog_frame_cache()

    with _FredTransport({"VIXCLS": _fred_body(_BACKFILL_OBSERVATIONS)}).client() as http, pytest.raises(PriorPartitionUntrustedError):
        fetch_and_persist_macro_backfill(
            ("VIXCLS",), _DOTCOM_START, secrets=_SECRETS, settings=settings, client=http
        )


def test_backfill_requires_a_stored_first_vintage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without an empty request, a stored first vintage, or a session window, the backfill refuses."""
    import src.data.macro_backfill as backfill_module

    settings = _fresh_settings(monkeypatch, tmp_path)
    with _FredTransport({}).client() as http, pytest.raises(ValueError, match="at least one series id"):
        fetch_and_persist_macro_backfill((), _DOTCOM_START, secrets=_SECRETS, settings=settings, client=http)
    with _FredTransport({}).client() as http, pytest.raises(ValueError, match="existing MACRO partition"):
        fetch_and_persist_macro_backfill(("VIXCLS",), _DOTCOM_START, secrets=_SECRETS, settings=settings, client=http)

    # A first vintage dated on a non-session leaves no session between it and the requested start.
    _persist_seed(
        settings,
        _vintage_frame([("VIXCLS", date(1998, 12, 26), datetime(1998, 12, 28, 21, 0, tzinfo=UTC), 25.0)]),
        ("VIXCLS",),
    )
    with _FredTransport({}).client() as http, pytest.raises(ValueError, match="no exchange session"):
        fetch_and_persist_macro_backfill(
            ("VIXCLS",), date(1998, 12, 25), secrets=_SECRETS, settings=settings, client=http
        )
    assert backfill_module.MACRO_BACKFILL_PATH.name == "macro_backfill.json"


def test_backfill_opens_its_own_client_when_none_is_injected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without an injected client the module owns one timeout-bounded session for the window."""
    import src.data.macro_backfill as backfill_module

    settings = _fresh_settings(monkeypatch, tmp_path)
    _seed_macro(settings, ("VIXCLS",))
    transport = _FredTransport({"VIXCLS": _fred_body(_BACKFILL_OBSERVATIONS)})
    real_client = httpx.Client
    opened: list[dict[str, object]] = []

    def owned_client(**kwargs: object) -> httpx.Client:
        opened.append(kwargs)
        return real_client(transport=httpx.MockTransport(transport))

    monkeypatch.setattr(backfill_module.httpx, "Client", owned_client)
    artifact = fetch_and_persist_macro_backfill(
        ("VIXCLS",), _DOTCOM_START, secrets=_SECRETS, settings=settings
    )
    assert len(opened) == 1
    assert "timeout" in opened[0]
    assert transport.served == ["VIXCLS"]
    assert artifact.manifest.row_count > 0


def test_shipped_allowlist_holds_unrevised_series_with_one_session_lag() -> None:
    """The git-tracked allowlist names VIXCLS and BAA10Y, each lagged one session."""
    assert load_unrevised_macro_series() == (
        UnrevisedMacroSeries(series_id="BAA10Y", release_lag_sessions=1),
        UnrevisedMacroSeries(series_id="VIXCLS", release_lag_sessions=1),
    )


def _allowlist_document(*entries: object) -> object:
    return {"series": list(entries)}


def _write_allowlist(tmp_path: Path, document: object) -> Path:
    target = tmp_path / "macro_backfill.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        pytest.param(
            _allowlist_document(
                {"series_id": "VIXCLS", "release_lag_sessions": 1},
                {"series_id": "VIXCLS", "release_lag_sessions": 2},
            ),
            "duplicate macro backfill series",
            id="duplicate-id",
        ),
        pytest.param(
            _allowlist_document({"series_id": "VIXCLS", "release_lag_sessions": 0}),
            "at least 1",
            id="zero-lag",
        ),
        pytest.param(
            _allowlist_document({"series_id": "VIXCLS", "release_lag_sessions": "1"}),
            "at least 1",
            id="non-integer-lag",
        ),
        pytest.param(
            _allowlist_document({"series_id": "VIXCLS"}),
            "at least 1",
            id="missing-lag",
        ),
        pytest.param(
            _allowlist_document({"release_lag_sessions": 1}),
            "declares no 'series_id'",
            id="missing-id",
        ),
        pytest.param(_allowlist_document("VIXCLS"), "is not a JSON object", id="entry-not-object"),
        pytest.param({}, "declares no 'series' list", id="missing-list"),
        pytest.param([], "is not a JSON object", id="document-not-object"),
    ],
)
def test_load_unrevised_macro_series_rejects_invalid_allowlists(
    tmp_path: Path, document: object, expected: str
) -> None:
    """Invalid allowlists fail closed with a message naming the offending entry."""
    with pytest.raises(ValueError, match=expected):
        load_unrevised_macro_series(_write_allowlist(tmp_path, document))


def test_load_unrevised_macro_series_rejects_malformed_json(tmp_path: Path) -> None:
    """A non-JSON allowlist fails closed instead of yielding an empty, silently trusted set."""
    target = tmp_path / "macro_backfill.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        load_unrevised_macro_series(target)


def test_cli_dispatches_macro_backfill(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ingest macro-backfill` forwards the comma list and start; usage gaps exit two."""
    import src.cli as cli_mod
    import src.cli_commands.ingest as ingest_mod

    seen: list[tuple[tuple[str, ...], date]] = []

    def fake_backfill(series_ids: tuple[str, ...], start: date, **_kwargs: object) -> object:
        seen.append((series_ids, start))
        return type("_Artifact", (), {"manifest": type("M", (), {"row_count": 1})()})()

    monkeypatch.setattr(ingest_mod, "fetch_and_persist_macro_backfill", fake_backfill)
    monkeypatch.setattr(cli_mod, "load_provider_secrets", lambda: object())

    assert main(["ingest", "macro-backfill", "--series-id", "VIXCLS", "--start", "1998-01-02"]) == 0
    assert main(["ingest", "macro-backfill", "--series-id", "VIXCLS, BAA10Y", "--start", "1998-01-02"]) == 0
    assert seen == [(("VIXCLS",), date(1998, 1, 2)), (("VIXCLS", "BAA10Y"), date(1998, 1, 2))]
    assert main(["ingest", "macro-backfill", "--start", "1998-01-02"]) == 2
    assert main(["ingest", "macro-backfill", "--series-id", "VIXCLS"]) == 2
    assert main(["ingest", "macro-backfill", "--series-id", ",", "--start", "1998-01-02"]) == 2
    assert len(seen) == 2
