# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.report."""
import sys
import src.analytics.thesis.report as _real
sys.modules[__name__] = _real
from src.analytics.thesis.report import ThesisReport, build_thesis_report, write_thesis_report
__all__ = ["ThesisReport", "build_thesis_report", "write_thesis_report"]
