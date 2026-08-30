# ruff: noqa: I001,E402,RUF022
"""Legacy shim re-exporting thesis.incremental."""
import sys
import src.analytics.thesis.incremental as _real
sys.modules[__name__] = _real
from src.analytics.thesis.incremental import INCREMENTAL_HORIZON_SURFACE, INCREMENTAL_SATELLITE_WEIGHTS, INCREMENTAL_SOXX_WEIGHTS, PATH_BOOTSTRAP_WIN_FLOOR, BuyOnlyAttribution, IncrementalArmId, IncrementalArmReport, IncrementalPortfolioReport, PathBootstrapVerdict, apply_incremental_portfolio_status, arm_targets, attribute_buy_only_soxx, classify_portfolio_status, clip_incremental_cohort_start, make_incremental_arm_id, monthly_simple_returns, paired_path_block_bootstrap, resolve_incremental_horizon, run_incremental_portfolio, write_incremental_portfolio_report
__all__ = ["INCREMENTAL_HORIZON_SURFACE", "INCREMENTAL_SATELLITE_WEIGHTS", "INCREMENTAL_SOXX_WEIGHTS", "PATH_BOOTSTRAP_WIN_FLOOR", "BuyOnlyAttribution", "IncrementalArmId", "IncrementalArmReport", "IncrementalPortfolioReport", "PathBootstrapVerdict", "apply_incremental_portfolio_status", "arm_targets", "attribute_buy_only_soxx", "classify_portfolio_status", "clip_incremental_cohort_start", "make_incremental_arm_id", "monthly_simple_returns", "paired_path_block_bootstrap", "resolve_incremental_horizon", "run_incremental_portfolio", "write_incremental_portfolio_report"]
