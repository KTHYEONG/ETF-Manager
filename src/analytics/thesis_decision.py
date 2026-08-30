# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.decision."""
import sys
import src.analytics.thesis.decision as _real
sys.modules[__name__] = _real
from src.analytics.thesis.decision import ThesisDecision, ThesisDecisionRecord, synthesize_thesis_decision
__all__ = ["ThesisDecision", "ThesisDecisionRecord", "synthesize_thesis_decision"]
