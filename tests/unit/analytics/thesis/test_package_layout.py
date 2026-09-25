"""Wave 2 thesis package layout contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_thesis_package_modules_exist() -> None:
    from src.analytics.thesis import THESIS_MODULES

    expected = (
        "evidence",
        "meaning",
        "decision",
        "report",
        "wave",
        "structural",
        "valuation",
        "crowding",
        "purity",
        "wave_d_exit",
        "incremental",
    )
    assert expected == THESIS_MODULES
    for stem in expected:
        assert stem in THESIS_MODULES
    # each file exists
    base = Path("src/analytics/thesis")
    for stem in THESIS_MODULES:
        assert (base / f"{stem}.py").is_file(), f"missing {stem}.py"


def test_thesis_package_real_modules_under_560_lines() -> None:
    base = Path("src/analytics/thesis")
    for p in base.glob("*.py"):
        if p.name == "__init__.py":
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 560, f"{p} has {len(lines)} > 560"


def test_thesis_legacy_shims_under_40_lines() -> None:
    legacy = (
        "thesis_evidence",
        "thesis_meaning",
        "thesis_decision",
        "thesis_report",
        "thesis_wave",
        "structural_evidence",
        "valuation_evidence",
        "crowding_evidence",
        "purity_evidence",
        "wave_d_exit",
        "incremental_portfolio",
    )
    for stem in legacy:
        p = Path(f"src/analytics/{stem}.py")
        assert p.is_file(), f"legacy shim missing {p}"
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 40, f"{p} has {len(lines)} > 40"


def test_thesis_legacy_import_compute_evidence_vector() -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector
    from src.analytics.thesis.evidence import compute_evidence_vector as cv2

    assert compute_evidence_vector is cv2


def test_thesis_package_no_import_of_shims() -> None:
    base = Path("src/analytics/thesis")
    legacy_imports = [
        "from src.analytics.thesis_evidence",
        "from src.analytics.thesis_meaning",
        "from src.analytics.thesis_decision",
        "from src.analytics.thesis_report",
        "from src.analytics.thesis_wave",
        "from src.analytics.structural_evidence",
        "from src.analytics.valuation_evidence",
        "from src.analytics.crowding_evidence",
        "from src.analytics.purity_evidence",
        "from src.analytics.wave_d_exit",
        "from src.analytics.incremental_portfolio",
        "import src.analytics.thesis_evidence",
        "import src.analytics.structural_evidence",
        "import src.analytics.valuation_evidence",
        "import src.analytics.crowding_evidence",
        "import src.analytics.purity_evidence",
        "import src.analytics.thesis_meaning",
        "import src.analytics.thesis_decision",
        "import src.analytics.thesis_report",
        "import src.analytics.thesis_wave",
        "import src.analytics.wave_d_exit",
        "import src.analytics.incremental_portfolio",
        # also check without 'from ' prefix inside thesis package should be qualified
        "src.analytics.thesis_evidence",
        "src.analytics.structural_evidence",
        "src.analytics.valuation_evidence",
        "src.analytics.crowding_evidence",
        "src.analytics.purity_evidence",
    ]
    # need to avoid false positive for thesis.* imports containing legacy substring?
    # we check exact legacy import patterns; for wave_d_exit and incremental, ensure not importing legacy path without thesis prefix
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        for needle in legacy_imports:
            # For incremental_portfolio and wave_d_exit, the new path is src.analytics.thesis.incremental / thesis.wave_d_exit
            # So searching for "src.analytics.wave_d_exit" would match "src.analytics.thesis.wave_d_exit" as substring -> avoid.
            # Use explicit checks for legacy without thesis prefix.
            if needle in text:
                # if needle is like "src.analytics.thesis_evidence", it won't appear in thesis/* if we use thesis.evidence, so any occurrence is bad
                # For wave_d_exit, if text contains "from src.analytics.thesis.wave_d_exit", it contains substring "src.analytics.wave_d_exit" but we should not flag.
                # So refine: only flag if needle without thesis. prefix appears not as part of thesis.
                if "src.analytics.thesis.wave_d_exit" in text and needle == "src.analytics.wave_d_exit":
                    # skip because it's part of new import
                    # check if legacy import without thesis. exists separately
                    # look for "from src.analytics.wave_d_exit" specifically
                    if "from src.analytics.wave_d_exit" in text or "import src.analytics.wave_d_exit" in text:
                        pytest.fail(f"{p} imports legacy shim via {needle}")
                    continue
                if "src.analytics.thesis.incremental" in text and needle == "src.analytics.incremental_portfolio":
                    if "from src.analytics.incremental_portfolio" in text or "import src.analytics.incremental_portfolio" in text:
                        pytest.fail(f"{p} imports legacy shim via {needle}")
                    continue
                pytest.fail(f"{p} imports legacy shim via {needle!r}: found {needle}")


def test_thesis_report_shim_resolves_canonical_identities() -> None:
    """Legacy thesis-report imports resolve to the canonical report objects."""
    import inspect

    import src.analytics.thesis.report as canonical
    import src.analytics.thesis_report as shim

    assert shim.ThesisReport is canonical.ThesisReport
    assert shim.build_thesis_report is canonical.build_thesis_report
    assert shim.write_thesis_report is canonical.write_thesis_report
    assert set(shim.__all__) == {"ThesisReport", "build_thesis_report", "write_thesis_report"}
    assert inspect.signature(shim.build_thesis_report) == inspect.signature(canonical.build_thesis_report)
    assert inspect.signature(shim.write_thesis_report) == inspect.signature(canonical.write_thesis_report)


def test_thesis_shim_imports_have_no_cycle() -> None:
    """Fresh processes import the thesis package and all shims in either order."""
    import subprocess
    import sys

    shims = [
        "src.analytics.thesis_evidence",
        "src.analytics.thesis_meaning",
        "src.analytics.thesis_decision",
        "src.analytics.thesis_report",
        "src.analytics.thesis_wave",
    ]
    orders = [
        ["src.analytics.thesis", *shims],
        [*shims, "src.analytics.thesis"],
    ]
    for modules in orders:
        imports = "; ".join(f"import {name}" for name in modules)
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", imports],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, f"import order {modules} failed: {completed.stderr[-500:]}"


def test_ev_no_adoption_import() -> None:
    # behavior freeze check also required in this package test
    for path in [Path("src/analytics/thesis/evidence.py"), Path("src/analytics/thesis_evidence.py")]:
        text = path.read_text(encoding="utf-8")
        assert "adoption_passes" not in text, f"{path} contains adoption_passes"
    # also ensure report not containing
    for path in [Path("src/analytics/thesis/report.py"), Path("src/analytics/thesis_report.py")]:
        text = path.read_text(encoding="utf-8")
        assert "adoption_passes" not in text
