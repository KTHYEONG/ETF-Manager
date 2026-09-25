# ruff: noqa: I001,RUF022
"""Keep the published thesis-report import path bound to canonical objects.

The compatibility module is intentionally small so historical imports and
monkeypatch targets continue to resolve to the canonical implementation.
"""
from src.analytics.thesis.report import ThesisReport, build_thesis_report, write_thesis_report

__all__ = ["ThesisReport", "build_thesis_report", "write_thesis_report"]
