"""Minimal packaging/release-metadata checks.

Not a full build+install integration test (that requires network access to
fetch torch into an isolated venv and is exercised manually per RELEASE
checklist / STATE.md, not in CI). This just guards against metadata drift
between ``pyproject.toml``, the installed distribution, and the public API
that a real `pip install` would expose.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import ckptplan

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_license_file_exists_and_is_referenced() -> None:
    license_path = _REPO_ROOT / "LICENSE"
    assert license_path.exists(), "release metadata claims an MIT license but LICENSE is missing"
    assert "MIT License" in license_path.read_text()


def test_installed_metadata_matches_package_version() -> None:
    dist_version = importlib.metadata.version("ckptplan")
    assert dist_version == ckptplan.__version__


def test_public_api_is_importable_from_top_level_package() -> None:
    # A minimal guard that declare_blocks -> profile_blocks -> plan_checkpoints
    # -> apply_plan -> run_benchmark (the documented pipeline) all resolve from
    # the top-level namespace, the way a real installed user would import them.
    names = [
        "declare_blocks",
        "profile_blocks",
        "plan_checkpoints",
        "validate_plan",
        "apply_plan",
        "run_benchmark",
        "compare_results",
    ]
    for name in names:
        assert hasattr(ckptplan, name), f"ckptplan.{name} is not exported"
    for name in names:
        assert name in ckptplan.__all__


def test_project_metadata_has_license_and_urls() -> None:
    metadata = importlib.metadata.metadata("ckptplan")
    # importlib.metadata exposes PEP 639 license expressions under "License-Expression"
    # on modern packaging; fall back to the classic "License" field for older setuptools.
    license_value = metadata.get("License-Expression") or metadata.get("License")
    assert license_value, "no license recorded in built package metadata"
    project_urls = metadata.get_all("Project-URL") or []
    assert any("github.com" in url for url in project_urls), "no repository URL recorded"
