"""Unit tests for the frozen 2026 pension regime file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.sim.pension_tax import load_pension_tax_regime

_SHIPPED = Path("configs/tax/kr_pension_2026.json")


def _shipped_document() -> dict[str, Any]:
    return json.loads(_SHIPPED.read_text(encoding="utf-8"))


def _write_regime(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "regime.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_shipped_regime_is_explicit_and_finite() -> None:
    """The 2026 file carries every threshold, rate, band, and source URL."""
    regime = load_pension_tax_regime(_SHIPPED)
    assert regime.regime_id == "KR_PENSION_2026"
    assert regime.policy_year == 2026
    assert regime.annual_contribution_limit_krw == 18_000_000
    assert regime.annual_credit_limit_krw == 6_000_000
    assert regime.wage_threshold_krw == 55_000_000
    assert regime.comprehensive_income_threshold_krw == 45_000_000
    assert regime.low_income_credit_rate == pytest.approx(0.15)
    assert regime.high_income_credit_rate == pytest.approx(0.12)
    assert regime.local_surcharge_rate == pytest.approx(0.10)
    assert regime.minimum_pension_age == 55
    assert regime.minimum_account_years == 5
    assert regime.private_pension_threshold_krw == 15_000_000
    assert regime.age_withholding_bands == ((0, 0.05), (70, 0.04), (80, 0.03))
    assert regime.above_threshold_separate_rate == pytest.approx(0.15)
    assert regime.non_pension_rate == pytest.approx(0.15)
    assert regime.pension_limit_multiplier == pytest.approx(1.2)
    assert regime.pension_limit_final_year == 10
    assert len(regime.source_urls) == 4
    assert all(url.startswith("http") for url in regime.source_urls)


def test_missing_and_extra_keys_rejected(tmp_path: Path) -> None:
    """A regime missing one key or carrying an extra key fails closed."""
    document = _shipped_document()
    del document["non_pension_rate"]
    with pytest.raises(ValueError, match="missing"):
        load_pension_tax_regime(_write_regime(tmp_path, document))
    document = _shipped_document()
    document["irp_rule"] = True
    with pytest.raises(ValueError, match="extra"):
        load_pension_tax_regime(_write_regime(tmp_path, document))


def test_non_object_regime_rejected(tmp_path: Path) -> None:
    """A regime document that is not an object fails closed."""
    path = tmp_path / "regime.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_pension_tax_regime(path)


def test_regime_field_boundaries(tmp_path: Path) -> None:
    """Every malformed regime field raises ValueError."""
    base = _shipped_document()
    cases: list[tuple[dict[str, Any], str]] = [
        ({"regime_id": ""}, "nonempty string"),
        ({"regime_id": 7}, "nonempty string"),
        ({"policy_year": "2026"}, "integer year"),
        ({"policy_year": 2025}, "unknown"),
        ({"source_urls": []}, "source_urls"),
        ({"source_urls": ["ftp://x"]}, "source_urls"),
        ({"source_urls": "https://x"}, "source_urls"),
        ({"annual_contribution_limit_krw": -1}, "positive"),
        ({"annual_contribution_limit_krw": 6.5}, "integer amount"),
        ({"annual_contribution_limit_krw": True}, "integer amount"),
        ({"wage_threshold_krw": 0}, "positive"),
        ({"low_income_credit_rate": 1.0}, "lie in"),
        ({"low_income_credit_rate": -0.01}, "lie in"),
        ({"low_income_credit_rate": "0.15"}, "finite rate"),
        ({"high_income_credit_rate": float("nan")}, "finite rate"),
        ({"local_surcharge_rate": float("inf")}, "finite rate"),
        ({"minimum_pension_age": 0}, "positive"),
        ({"minimum_account_years": -2}, "positive"),
        ({"pension_limit_final_year": 0}, "positive"),
        ({"pension_limit_multiplier": "1.2"}, "finite number"),
        ({"pension_limit_multiplier": 0.9}, "cover 1.0"),
        ({"pension_limit_multiplier": float("nan")}, "finite number"),
        ({"above_threshold_separate_rate": 2.0}, "lie in"),
        ({"non_pension_rate": -0.1}, "lie in"),
        ({"annual_credit_limit_krw": 19_000_000}, "must not exceed contribution cap"),
        ({"low_income_credit_rate": 0.10}, "must cover high-income rate"),
        ({"age_withholding_bands": []}, "non-empty array"),
        ({"age_withholding_bands": "x"}, "non-empty array"),
        ({"age_withholding_bands": [[0]]}, "pair"),
        ({"age_withholding_bands": [[-1, 0.05], [70, 0.04]]}, "nonnegative integer age"),
        ({"age_withholding_bands": [[True, 0.05]]}, "nonnegative integer age"),
        ({"age_withholding_bands": [[0, "x"]]}, "finite rate"),
        ({"age_withholding_bands": [[0, float("nan")]]}, "finite rate"),
        ({"age_withholding_bands": [[0, 1.5]]}, "in \\[0, 1\\)"),
        ({"age_withholding_bands": [[10, 0.05]]}, "start at age 0"),
        ({"age_withholding_bands": [[0, 0.05], [70, 0.04], [70, 0.03]]}, "strictly increasing"),
        ({"age_withholding_bands": [[0, 0.05], [65, 0.04], [60, 0.03]]}, "strictly increasing"),
    ]
    for mutate, match in cases:
        document = dict(base)
        document.update(mutate)
        with pytest.raises(ValueError, match=match):
            load_pension_tax_regime(_write_regime(tmp_path, document))
