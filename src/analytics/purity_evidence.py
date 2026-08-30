# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.purity."""
import sys
import src.analytics.thesis.purity as _real
sys.modules[__name__] = _real
from src.analytics.thesis.purity import INDUSTRIAL_AUTOMATION_ROLES, HUMANOID_OPTIONALITY_ROLES, thesis_aligned_weight_pct, role_aligned_weight_pct, compute_purity_slot
__all__ = ["INDUSTRIAL_AUTOMATION_ROLES", "HUMANOID_OPTIONALITY_ROLES", "thesis_aligned_weight_pct", "role_aligned_weight_pct", "compute_purity_slot"]
