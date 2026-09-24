"""Unit tests for SOPS-aware provider secret loading."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.data.secrets import ProviderSecrets, load_provider_secrets

_NAMES = ("TIINGO_API", "FRED_API", "ECOS_API")


def _patch_sops(
    monkeypatch: pytest.MonkeyPatch, stdout: bytes
) -> list[tuple[list[str], dict[str, Any]]]:
    """Replace subprocess.run; records every (argv, kwargs) invocation."""
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_env_layer_wins_without_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SEC-C01-env-then-sops"""
    monkeypatch.setenv("TIINGO_API", "env-tiingo-token")
    monkeypatch.setenv("FRED_API", "env-fred-token")
    monkeypatch.setenv("ECOS_API", "env-ecos-token")
    calls = _patch_sops(monkeypatch, b"")

    secrets = load_provider_secrets(env_enc=tmp_path / "absent.enc", env_file=tmp_path / "absent.env")

    assert secrets == ProviderSecrets(
        tiingo_api="env-tiingo-token", fred_api="env-fred-token", ecos_api="env-ecos-token"
    )
    assert calls == []


def test_sops_stdout_resolves_missing_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SEC-C01-env-then-sops"""
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    stdout = b"TIINGO_API=sops-tiingo\nFRED_API='sops-fred'\nECOS_API=\"sops-ecos\"\n"
    calls = _patch_sops(monkeypatch, stdout)
    env_enc = tmp_path / ".env.enc"
    env_enc.touch()

    secrets = load_provider_secrets(env_enc=env_enc, env_file=tmp_path / "absent.env")

    assert secrets == ProviderSecrets(tiingo_api="sops-tiingo", fred_api="sops-fred", ecos_api="sops-ecos")
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == ["sops", "-d", "--input-type", "dotenv", "--output-type", "dotenv", str(env_enc)]
    timeout = kwargs.get("timeout")
    assert isinstance(timeout, (int, float))
    assert 0 < timeout <= 30


@pytest.mark.parametrize("scenario_id", ["SEC-E01-sops-dotenv-flags"])
def test_sops_decrypt_argv_carries_dotenv_flags(
    scenario_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SEC-E01-sops-dotenv-flags"""
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    stdout = b"TIINGO_API=sops-tiingo\nFRED_API=sops-fred\nECOS_API=sops-ecos\n"
    calls = _patch_sops(monkeypatch, stdout)
    env_enc = tmp_path / ".env.enc"
    env_enc.touch()

    secrets = load_provider_secrets(env_enc=env_enc, env_file=tmp_path / "absent.env")

    assert secrets == ProviderSecrets(tiingo_api="sops-tiingo", fred_api="sops-fred", ecos_api="sops-ecos")
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == ["sops", "-d", "--input-type", "dotenv", "--output-type", "dotenv", str(env_enc)]
    assert kwargs.get("timeout") == 30


def test_dotenv_file_layer_fills_partial_gaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SEC-C01-env-then-sops"""
    monkeypatch.delenv("TIINGO_API", raising=False)
    monkeypatch.delenv("FRED_API", raising=False)
    monkeypatch.delenv("ECOS_API", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# local overrides\nFRED_API=file-fred\nEMPTY=\n", encoding="utf-8")
    calls = _patch_sops(monkeypatch, b"TIINGO_API=sops-tiingo\nECOS_API=sops-ecos\n")
    env_enc = tmp_path / ".env.enc"
    env_enc.touch()

    secrets = load_provider_secrets(env_enc=env_enc, env_file=env_file)

    assert secrets == ProviderSecrets(
        tiingo_api="sops-tiingo", fred_api="file-fred", ecos_api="sops-ecos"
    )
    assert [argv[0] for argv, _ in calls] == ["sops"]


def test_absent_env_enc_skips_sops_without_invoking_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SEC-C01-env-then-sops: sops 폐기 후 .env.enc가 없으면 조용히 건너뛴다."""
    monkeypatch.setenv("TIINGO_API", "env-tiingo-token")
    monkeypatch.setenv("FRED_API", "env-fred-token")
    monkeypatch.setenv("ECOS_API", "env-ecos-token")
    calls = _patch_sops(monkeypatch, b"should-not-be-read")

    secrets = load_provider_secrets(env_enc=tmp_path / "absent.env.enc", env_file=tmp_path / "absent.env")

    assert secrets == ProviderSecrets(
        tiingo_api="env-tiingo-token", fred_api="env-fred-token", ecos_api="env-ecos-token"
    )
    assert calls == []


def test_missing_name_raises_value_error_listing_names_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SEC-C01-env-then-sops"""
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    _patch_sops(monkeypatch, b"TIINGO_API=sops-only-tiingo-token\nECOS_API=sops-only-ecos\n")
    env_enc = tmp_path / ".env.enc"
    env_enc.touch()

    with pytest.raises(ValueError, match="FRED_API") as excinfo:
        load_provider_secrets(env_enc=env_enc, env_file=tmp_path / "absent.env")

    message = str(excinfo.value)
    assert "sops-only-tiingo-token" not in message


def test_shared_file_export_prefixed_lines_parse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Export-prefixed shared lines resolve without env or sops."""
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    _patch_sops(monkeypatch, b"")
    shared = tmp_path / ".quant.env"
    shared.write_text(
        "export TIINGO_API=a\nexport FRED_API=\"b\"\nexport ECOS_API='c'\n", encoding="utf-8"
    )

    secrets = load_provider_secrets(
        env_enc=tmp_path / "absent.enc", env_file=tmp_path / "absent.env", shared_env_file=shared
    )

    assert secrets == ProviderSecrets(tiingo_api="a", fred_api="b", ecos_api="c")


def test_shared_layer_fills_only_gaps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Process env wins; the shared file fills only the remaining gaps."""
    monkeypatch.setenv("TIINGO_API", "env")
    monkeypatch.delenv("FRED_API", raising=False)
    monkeypatch.delenv("ECOS_API", raising=False)
    _patch_sops(monkeypatch, b"")
    shared = tmp_path / ".quant.env"
    shared.write_text("TIINGO_API=shared\nFRED_API=shared-b\nECOS_API=shared-c\n", encoding="utf-8")

    secrets = load_provider_secrets(
        env_enc=tmp_path / "absent.enc", env_file=tmp_path / "absent.env", shared_env_file=shared
    )

    assert secrets == ProviderSecrets(tiingo_api="env", fred_api="shared-b", ecos_api="shared-c")


def test_absent_shared_file_is_empty_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing shared path never errors when env already covers every token."""
    monkeypatch.setenv("TIINGO_API", "env-tiingo")
    monkeypatch.setenv("FRED_API", "env-fred")
    monkeypatch.setenv("ECOS_API", "env-ecos")
    calls = _patch_sops(monkeypatch, b"should-not-be-read")

    secrets = load_provider_secrets(
        env_enc=tmp_path / "absent.enc",
        env_file=tmp_path / "absent.env",
        shared_env_file=tmp_path / "absent.quant.env",
    )

    assert secrets == ProviderSecrets(tiingo_api="env-tiingo", fred_api="env-fred", ecos_api="env-ecos")
    assert calls == []


def test_conftest_isolation_never_reads_real_shared_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an explicit path the redirected SHARED_ENV_FILE keeps tests hermetic."""
    import src.data.secrets as secrets_mod

    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    _patch_sops(monkeypatch, b"")
    assert secrets_mod.SHARED_ENV_FILE.name == "absent.quant.env"
    assert not secrets_mod.SHARED_ENV_FILE.is_file()

    with pytest.raises(ValueError, match="TIINGO_API") as excinfo:
        load_provider_secrets(env_enc=Path("absent.enc"), env_file=Path("absent.env"))

    message = str(excinfo.value)
    assert "TIINGO_API" in message
    assert "FRED_API" in message
    assert "ECOS_API" in message
