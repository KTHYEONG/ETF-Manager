"""Runtime settings boundary for the repository-local data root."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    """Own typed runtime roots and dated operational defaults in one validated settings boundary.

    Args:
        data_root: Allowed repository-local generated data root.

    Raises:
        ValueError: If roots or effective dates are malformed or escape the repository.
    """

    data_root: Path = Path("data")
    model_config = SettingsConfigDict(env_prefix="ETF_MANAGER_", env_file=".env", extra="ignore")

    def resolved_data_root(self) -> Path:
        """Resolve ``data_root`` beneath the current working directory.

        Relative roots anchor at ``Path.cwd()``; symlinks are resolved. The
        directory is never created here — storage layers own creation.

        Raises:
            ValueError: When the resolved root is neither equal to nor a
                descendant of the current working directory.
        """
        cwd = Path.cwd().resolve()
        candidate = self.data_root if self.data_root.is_absolute() else cwd / self.data_root
        resolved = candidate.resolve()
        try:
            resolved.relative_to(cwd)
        except ValueError as exc:
            raise ValueError(
                f"data_root {str(self.data_root)!r} resolves to {str(resolved)!r}, outside cwd {str(cwd)!r}"
            ) from exc
        return resolved
