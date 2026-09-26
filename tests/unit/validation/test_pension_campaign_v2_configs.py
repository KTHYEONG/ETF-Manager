"""Invariant guards for the historical pension selection campaign configs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.data.pension_market import load_pension_etf_identities
from src.validation.pension_campaign import load_pension_campaign_spec

_REPO = Path(__file__).resolve().parents[3]
_DOTCOM_PATH = _REPO / "configs" / "research" / "pension_campaign_v2_dotcom.json"
_SEMIS_PATH = _REPO / "configs" / "research" / "pension_campaign_v2_semis.json"
_V1_PATH = _REPO / "configs" / "research" / "pension_campaign_v1.json"
_IDENTITY_PATH = _REPO / "configs" / "data" / "pension_etfs_2026.json"
_INDEX_PATH = _REPO / "configs" / "research" / "INDEX.json"
_CONFIG_PATHS = (_DOTCOM_PATH, _SEMIS_PATH)


def _read_json(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_dotcom_campaign_config_loads_standalone() -> None:
    """The dot-com campaign loads with its historical start and SPY baseline."""
    spec = load_pension_campaign_spec(_DOTCOM_PATH)
    assert spec.name == "pension_campaign_v2_dotcom"
    assert spec.start == date(1999, 5, 1)
    assert spec.household is None
    assert "household_view" not in _read_json(_DOTCOM_PATH)
    assert [arm.arm_id for arm in spec.arms if arm.role == "baseline"] == ["sp500_100"]


def test_semis_campaign_config_loads_standalone() -> None:
    """The semiconductor campaign starts after SOXX inception and has no household view."""
    spec = load_pension_campaign_spec(_SEMIS_PATH)
    assert spec.name == "pension_campaign_v2_semis"
    assert spec.start == date(2001, 8, 1)
    assert spec.household is None
    assert "household_view" not in _read_json(_SEMIS_PATH)


def test_campaign_arm_tickers_map_to_pension_eligible_korean_etfs() -> None:
    """Every US proxy ticker resolves uniquely to a pension-eligible Korean ETF identity."""
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    for config_path in _CONFIG_PATHS:
        spec = load_pension_campaign_spec(config_path)
        for arm in spec.arms:
            for ticker in arm.targets:
                matches = [identity for identity in identities if identity.proxy_ticker == ticker]
                assert len(matches) == 1
                assert matches[0].pension_eligible is True


def test_campaign_contributions_follow_account_openings() -> None:
    """No scheduled contribution predates any participating pension account."""
    for config_path in _CONFIG_PATHS:
        spec = load_pension_campaign_spec(config_path)
        contribution_dates = tuple(day for dates in spec.contribution_dates.values() for day in dates)
        assert contribution_dates
        for profile in spec.profiles:
            assert all(profile.account_open_date <= day for day in contribution_dates)


def test_campaign_frictions_and_guards_match_v1() -> None:
    """Historical selection campaigns preserve v1 execution, FX, and CPI guardrails."""
    fields = (
        "execution_spread_bps",
        "commission_bps",
        "max_fx_age_days",
        "max_fx_fallback_share",
        "max_cpi_age_days",
    )
    v1 = _read_json(_V1_PATH)
    for config_path in _CONFIG_PATHS:
        document = _read_json(config_path)
        assert {field: document[field] for field in fields} == {field: v1[field] for field in fields}


def test_experiment_index_catalogs_historical_pension_campaigns() -> None:
    """Both active historical pension campaigns are present in the experiment taxonomy."""
    index = _read_json(_INDEX_PATH)
    files = index["files"]
    assert isinstance(files, dict)
    assert files["pension_campaign_v2_dotcom.json"] == {
        "status": "active",
        "kind": "pension",
        "notes": "standalone pension campaign incl. dot-com, SPY/QQQ blends 120/240m",
    }
    assert files["pension_campaign_v2_semis.json"] == {
        "status": "active",
        "kind": "pension",
        "notes": "standalone pension campaign from SOXX inception, QQQ/SOXX blends 120/240m",
    }
