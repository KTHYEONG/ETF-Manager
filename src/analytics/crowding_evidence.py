# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.crowding."""
import sys
import src.analytics.thesis.crowding as _real
sys.modules[__name__] = _real
from src.analytics.thesis.crowding import compute_crowding_slot, holdings_concentration_metrics
__all__ = ["compute_crowding_slot", "holdings_concentration_metrics"]
