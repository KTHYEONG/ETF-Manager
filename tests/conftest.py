"""Pin pytest and stdlib temporary roots inside the repository tmp/ directory."""

from __future__ import annotations

import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_TMP = _REPO_ROOT / "tmp"
_PROJECT_TMP.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(_PROJECT_TMP)

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    basetemp = config.option.basetemp
    if basetemp is None:
        config.option.basetemp = str(_PROJECT_TMP / "pytest-of-pytest")


def pytest_sessionstart(session: pytest.Session) -> None:
    import os

    os.environ["TMPDIR"] = str(_PROJECT_TMP)


@pytest.fixture(autouse=True)
def _isolate_shared_provider_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep tests hermetic from the developer's real parent-folder shared secrets file."""
    import src.data.secrets as _secrets

    monkeypatch.setattr(_secrets, "SHARED_ENV_FILE", tmp_path_factory.mktemp("secrets") / "absent.quant.env")
