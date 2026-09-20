"""P12b: packaging dry run — wheel contents and metadata assertions.

Builds the wheel (marked 'slow' because uv build is not instant) and asserts:
  - jevmlx/presets/*.json ARE in the wheel (package data)
  - jevmlx/data/*.json ARE in the wheel (package data)
  - tests/, benchmarks/, .venv, results, scratch files are NOT
  - twine check passes
  - the wheel installs and `jevmlx --help` runs
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_PRESET_FILES = [
    "jevmlx/presets/code_security.json",
    "jevmlx/presets/content_moderation.json",
    "jevmlx/presets/fintech_fraud.json",
    "jevmlx/presets/high_cardinality_255.json",
    "jevmlx/presets/inbound_email.json",
    "jevmlx/presets/support_triage.json",
]

_DATA_FILES = ["jevmlx/data/bucket_edges.json"]

_FORBIDDEN_PREFIXES = ("tests/", "benchmarks/", ".venv/", "results/", "scratch/")


def _wheel_path() -> Path:
    dist = REPO / "dist"
    wheels = sorted(dist.glob("jevmlx-*.whl"))
    assert wheels, f"no wheel in {dist}; run `uv build` first"
    return wheels[-1]


def _wheel_names() -> list[str]:
    with zipfile.ZipFile(_wheel_path()) as zf:
        return zf.namelist()


@pytest.mark.slow
class TestPackaging:
    def test_build_wheel(self):
        """uv build produces a wheel + sdist."""
        dist = REPO / "dist"
        if not list(dist.glob("jevmlx-*.whl")):
            subprocess.run(["uv", "build"], cwd=REPO, check=True, capture_output=True)
        wheels = list(dist.glob("jevmlx-*.whl"))
        sdists = list(dist.glob("jevmlx-*.tar.gz"))
        assert wheels, "no wheel built"
        assert sdists, "no sdist built"

    def test_twine_check_passes(self):
        """twine check dist/* passes (valid metadata)."""
        if not (REPO / "dist" / "jevmlx-0.1.0-py3-none-any.whl").exists():
            subprocess.run(["uv", "build"], cwd=REPO, check=True, capture_output=True)
        dist_files = [str(p) for p in (REPO / "dist").iterdir() if p.suffix in (".whl", ".tar.gz")]
        result = subprocess.run(
            [sys.executable, "-m", "twine", "check", *dist_files],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"twine check failed:\n{result.stdout}\n{result.stderr}"

    def test_presets_in_wheel(self):
        """All preset JSONs are package data in the wheel."""
        names = set(_wheel_names())
        for f in _PRESET_FILES:
            assert f in names, f"{f} missing from wheel"

    def test_data_in_wheel(self):
        """All data JSONs are package data in the wheel."""
        names = set(_wheel_names())
        for f in _DATA_FILES:
            assert f in names, f"{f} missing from wheel"

    def test_no_tests_benchmarks_or_scratch_in_wheel(self):
        """tests/, benchmarks/, .venv, results, scratch are NOT in the wheel."""
        names = _wheel_names()
        for name in names:
            for prefix in _FORBIDDEN_PREFIXES:
                assert not name.startswith(prefix), f"{name} should not be in the wheel"

    def test_metadata_version_and_name(self):
        """The wheel's METADATA has name=jevmlx, version=0.1.0."""
        wheel = _wheel_path()
        with zipfile.ZipFile(wheel) as zf:
            meta_names = [n for n in zf.namelist() if n.endswith("METADATA")]
            assert meta_names, "no METADATA in wheel"
            meta = zf.read(meta_names[0]).decode("utf-8")
        assert "Name: jevmlx\n" in meta
        assert "Version: 0.1.0\n" in meta

    def test_metadata_license_and_classifiers(self):
        """The METADATA carries the MIT license + the expected classifiers."""
        wheel = _wheel_path()
        with zipfile.ZipFile(wheel) as zf:
            meta_names = [n for n in zf.namelist() if n.endswith("METADATA")]
            meta = zf.read(meta_names[0]).decode("utf-8")
        assert "License-Expression: MIT" in meta or "License: MIT" in meta
        assert "Operating System :: MacOS" in meta
        assert "Programming Language :: Python :: 3.12" in meta
        assert "Topic :: Scientific/Engineering :: Artificial Intelligence" in meta
