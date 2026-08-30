# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.valuation."""
import sys
import src.analytics.thesis.valuation as _real
sys.modules[__name__] = _real
from src.analytics.thesis.valuation import compute_valuation_slot, pit_price_series, relative_richness_percentile, trailing_total_return_pct
__all__ = ["compute_valuation_slot", "pit_price_series", "relative_richness_percentile", "trailing_total_return_pct"]
