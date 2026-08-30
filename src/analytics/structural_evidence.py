# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.structural."""
import sys
import src.analytics.thesis.structural as _real
sys.modules[__name__] = _real
from src.analytics.thesis.structural import compute_structural_slot, detect_yoy_regime_change, evaluate_falsifier_slowdown, pit_macro_series_levels, resolve_primary_falsifier, yoy_growth_pct
__all__ = ["compute_structural_slot", "detect_yoy_regime_change", "evaluate_falsifier_slowdown", "pit_macro_series_levels", "resolve_primary_falsifier", "yoy_growth_pct"]
