"""Public data-layer entry points."""

from src.etf_manager.data.fetch import (
    fetch_and_persist_cpi,
    fetch_and_persist_fx,
    fetch_and_persist_macro,
    fetch_and_persist_prices,
)
from src.etf_manager.data.secrets import ProviderSecrets, load_provider_secrets

__all__ = [
    "ProviderSecrets",
    "fetch_and_persist_cpi",
    "fetch_and_persist_fx",
    "fetch_and_persist_macro",
    "fetch_and_persist_prices",
    "load_provider_secrets",
]
