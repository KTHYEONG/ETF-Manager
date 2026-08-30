# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.evidence."""
import sys
import src.analytics.thesis.evidence as _real
sys.modules[__name__] = _real
from src.analytics.thesis.evidence import EvidenceSlot, EvidenceSnapshot, compute_evidence_vector
__all__ = ["EvidenceSlot", "EvidenceSnapshot", "compute_evidence_vector"]
