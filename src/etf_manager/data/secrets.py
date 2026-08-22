"""Load provider API tokens from env, .env, or SOPS-encrypted .env.enc."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_SECRET_NAMES: Final[tuple[str, ...]] = ("TIINGO_API", "FRED_API", "ECOS_API")
_SOPS_TIMEOUT_S: Final[int] = 30


@dataclass(frozen=True, slots=True)
class ProviderSecrets:
    """Plaintext tokens in memory only; never logged or persisted to manifests."""

    tiingo_api: str
    fred_api: str
    ecos_api: str


def load_provider_secrets(*, env_enc: Path = Path(".env.enc"), env_file: Path = Path(".env")) -> ProviderSecrets:
    """Resolve TIINGO_API, FRED_API, and ECOS_API without writing a decrypted file.

    Lookup order: process environment, then ``env_file``, then ``sops -d``
    stdout of ``env_enc``; later layers are consulted only while tokens are
    still missing. Empty strings count as missing values.

    Raises:
        ValueError: When any token is missing after all lookups; the message
            names the missing keys and never includes secret values.
    """
    layers: tuple[Callable[[], Mapping[str, str]], ...] = (
        lambda: dict(os.environ),
        lambda: _parse_dotenv(_read_text(env_file)),
        lambda: _decrypt_dotenv(env_enc),
    )
    resolved: dict[str, str] = {}
    for layer in layers:
        values = layer()
        for name in _SECRET_NAMES:
            if name not in resolved and values.get(name, "") != "":
                resolved[name] = values[name]
        if len(resolved) == len(_SECRET_NAMES):
            break
    missing = [name for name in _SECRET_NAMES if name not in resolved]
    if missing:
        raise ValueError(
            f"missing provider API tokens (provide via env, {str(env_file)!r}, or {str(env_enc)!r}): "
            + ", ".join(missing)
        )
    return ProviderSecrets(
        tiingo_api=resolved["TIINGO_API"],
        fred_api=resolved["FRED_API"],
        ecos_api=resolved["ECOS_API"],
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_dotenv(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines; blank lines, comments, and paired quotes are handled."""
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[key.strip()] = value
    return parsed


def _decrypt_dotenv(env_enc: Path) -> dict[str, str]:
    """Decrypt via ``sops -d`` stdout only; nothing is ever written to disk."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["sops", "-d", str(env_enc)],  # noqa: S607 - pinned vendor executable
            capture_output=True,
            check=False,
            timeout=_SOPS_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        # Exit code only; stderr may echo environment details.
        raise ValueError(f"sops decryption of {str(env_enc)!r} failed with exit code {completed.returncode}")
    return _parse_dotenv(completed.stdout.decode("utf-8", errors="replace"))
