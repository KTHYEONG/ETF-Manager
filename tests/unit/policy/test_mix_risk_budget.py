# ruff: noqa: S101, PT018, RUF043
"""Mix risk budget tests."""
from __future__ import annotations


def test_mrb_weight_hits_theta_then_clips() -> None:
    import math
    import pytest
    from src.policy.mix_risk_budget import satellite_risk_budget_weight
    from src.policy.targets import PolicyError

    def rc_frac(w: float, sig_q: float, sig_s: float, rho: float) -> float:
        a = sig_s ** 2
        b = sig_q ** 2
        c = rho * sig_q * sig_s
        var = w * w * a + (1.0 - w) ** 2 * b + 2.0 * w * (1.0 - w) * c
        rc_s = w * (w * a + (1.0 - w) * c)
        return rc_s / var

    w = satellite_risk_budget_weight(sigma_core=0.01, sigma_satellite=0.016, rho=0.85, theta=0.10)
    assert w == pytest.approx(0.0730, abs=1e-4)
    assert rc_frac(w, 0.01, 0.016, 0.85) == pytest.approx(0.10, abs=1e-9)
    raw = satellite_risk_budget_weight(sigma_core=0.01, sigma_satellite=0.01, rho=0.0, theta=0.10)
    assert raw == pytest.approx(0.25)
    from src.policy.mix_risk_budget import OPERATIONAL_MIX_RISK_BUDGET
    clipped = min(OPERATIONAL_MIX_RISK_BUDGET.satellite_weight_cap, max(OPERATIONAL_MIX_RISK_BUDGET.satellite_weight_floor, raw))
    assert clipped == pytest.approx(0.15)
    with pytest.raises(PolicyError):
        satellite_risk_budget_weight(sigma_core=0.0, sigma_satellite=0.01, rho=0.5, theta=0.10)
    with pytest.raises(PolicyError):
        satellite_risk_budget_weight(sigma_core=0.01, sigma_satellite=0.01, rho=math.nan, theta=0.10)


def test_mrb_resolve_simplex_and_naive_signal() -> None:
    from datetime import date, datetime
    import polars as pl
    import pytest
    from src.data.calendar import load_calendar
    from src.data.pit import TS_DTYPE
    from src.policy.mix_risk_budget import OPERATIONAL_MIX_RISK_BUDGET, resolve_mix_risk_budget_targets

    cal = load_calendar('XNYS')
    days = cal.sessions(date(2023, 10, 2), date(2024, 1, 31))
    assert len(days) >= 70
    rows: list[dict[str, object]] = []
    for i, day in enumerate(days):
        stamp = cal.close_ts(day)
        q = 0.001 * (1.0 if i % 2 == 0 else -1.0)
        s = 0.002 * (1.0 if i % 3 == 0 else -1.0)
        rows.append({'date': day, 'ticker': 'QQQ', 'adjusted_close': 100.0 * (1.0 + q * 0.01 * i), 'available_at': stamp})
        rows.append({'date': day, 'ticker': 'SOXX', 'adjusted_close': 50.0 * (1.0 + s * 0.01 * i), 'available_at': stamp})
    # rebuild monotonic prices
    q_px = 100.0
    s_px = 50.0
    rows = []
    for i, day in enumerate(days):
        q_px *= 1.0 + 0.001 * ((i % 5) - 2)
        s_px *= 1.0 + 0.002 * ((i % 7) - 3)
        stamp = cal.close_ts(day)
        rows.append({'date': day, 'ticker': 'QQQ', 'adjusted_close': q_px, 'available_at': stamp})
        rows.append({'date': day, 'ticker': 'SOXX', 'adjusted_close': s_px, 'available_at': stamp})
    prices = pl.DataFrame(rows, schema={'date': pl.Date, 'ticker': pl.String, 'adjusted_close': pl.Float64, 'available_at': TS_DTYPE})
    signal = cal.close_ts(days[-1])
    cfg = OPERATIONAL_MIX_RISK_BUDGET
    assert cfg.core_ticker == 'QQQ' and cfg.satellite_ticker == 'SOXX'
    assert cfg.satellite_risk_budget == pytest.approx(0.10)
    assert cfg.satellite_weight_cap == pytest.approx(0.15)
    assert cfg.vol_window == 63
    targets = resolve_mix_risk_budget_targets(prices, signal, cfg)
    assert set(targets) == {'QQQ', 'SOXX'}
    assert min(targets.values()) >= 0.0
    assert abs(sum(targets.values()) - 1.0) <= 1e-6
    assert targets['SOXX'] <= cfg.satellite_weight_cap + 1e-12
    with pytest.raises(ValueError, match='timezone-aware|naive'):
        resolve_mix_risk_budget_targets(prices, datetime(2024, 1, 31), cfg)
