# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.meaning."""
import sys
import src.analytics.thesis.meaning as _real
sys.modules[__name__] = _real
from src.analytics.thesis.meaning import HistoricalQuality, VehicleEvidenceStatus, ThesisEvidenceStatus, PortfolioEvidenceStatus, ThesisMeaningSnapshot, classify_thesis_meaning
__all__ = ["HistoricalQuality", "VehicleEvidenceStatus", "ThesisEvidenceStatus", "PortfolioEvidenceStatus", "ThesisMeaningSnapshot", "classify_thesis_meaning"]
