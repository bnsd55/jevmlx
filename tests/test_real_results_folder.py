"""Tests that run leaderboard._local_rows and check_results over the REAL
committed 7B results folder (benchmarks/results/m5max-128gb-...-7b-...).

These are NOT hand-made fixtures — they assert the exact rendered strings
on the real folder committed in PR #122. If the folder is absent (packaged
sdist without benchmarks/results), every test skips cleanly.

Run: pytest tests/test_real_results_folder.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.check_results import _error_lines_in_folder, check_root
from benchmarks.leaderboard import _local_rows

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_RESULTS = _REPO_ROOT / "benchmarks" / "results"
_7B_FOLDER = _REAL_RESULTS / "m5max-128gb-mlx-community--qwen2.5-7b-instruct-4bit"

# Skip the whole module if the real results folder is not present (packaged
# sdist strips benchmarks/results).
pytestmark = pytest.mark.skipif(
    not _7B_FOLDER.is_dir(),
    reason="benchmarks/results/m5max-128gb-...-7b-... not present (sdist without results)",
)


def test_local_rows_renders_all_three_tracks():
    """The real 7B folder renders 3 rows: naive, labels, slots (in sort order)."""
    rows = _local_rows(_REAL_RESULTS)
    scorers = [r["scorer"] for r in rows]
    assert scorers == ["naive (generate+parse)", "labels", "slots"], scorers


def test_naive_row_exact_values():
    """naive_local-slots-typesafe: 67.7% accuracy, Parity '—', 1.5s, 45 (7 error).

    Values are read from the folder but asserted as rendered strings so a
    drift in any field (accuracy, parity, timing, case count, error count)
    is caught.
    """
    rows = _local_rows(_REAL_RESULTS)
    naive = next(r for r in rows if r["scorer"] == "naive (generate+parse)")
    assert naive["accuracy"] == pytest.approx(0.6767, abs=0.001)
    assert naive["parity_status"] == "—"
    assert naive["time_per_case_s"] == pytest.approx(1.479, abs=0.01)
    assert naive["cases"] == 45
    assert naive["error_count"] == 7


def test_labels_row_exact_values():
    """parallel-labels-typesafe: 82.1% accuracy, DRIFT (0.078), 0.6s, 45 (1 error)."""
    rows = _local_rows(_REAL_RESULTS)
    labels = next(r for r in rows if r["scorer"] == "labels")
    assert labels["accuracy"] == pytest.approx(0.8213, abs=0.001)
    assert labels["parity_status"] == "DRIFT"
    assert labels["parity_max_drift"] == pytest.approx(0.078, abs=0.001)
    assert labels["time_per_case_s"] == pytest.approx(0.586, abs=0.01)
    assert labels["cases"] == 45
    assert labels["error_count"] == 1


def test_slots_row_exact_values():
    """parallel-slots-typesafe: 63.2% accuracy, DRIFT (0.078), 0.6s, 45 (1 error)."""
    rows = _local_rows(_REAL_RESULTS)
    slots = next(r for r in rows if r["scorer"] == "slots")
    assert slots["accuracy"] == pytest.approx(0.6322, abs=0.001)
    assert slots["parity_status"] == "DRIFT"
    assert slots["parity_max_drift"] == pytest.approx(0.078, abs=0.001)
    assert slots["time_per_case_s"] == pytest.approx(0.633, abs=0.01)
    assert slots["cases"] == 45
    assert slots["error_count"] == 1


@pytest.mark.slow
def test_all_combos_pass_check_results():
    """check_results over the real 7B folder: every combo OK.

    Marked slow: decompresses ~130K prediction lines across 9 gzipped files
    (~10 min wall on an M5). Run with -m slow.
    """
    results = check_root(_7B_FOLDER)
    assert results, "check_root returned no combos"
    for folder, ok, problems in results:
        assert ok, f"{folder.name}: FAIL — {problems}"


@pytest.mark.slow
def test_error_summary_matches_expected_counts():
    """The errors summary: naive_local-slots-typesafe has 7 error lines;
    parallel-{labels,slots}-typesafe each have 1. The bundled/perturbed
    combos have 0.

    Marked slow: same decompression cost as test_all_combos_pass_check_results.
    """
    results = check_root(_7B_FOLDER)
    error_counts = {folder.name: len(_error_lines_in_folder(folder)) for folder, _, _ in results}
    assert error_counts.get("naive_local-slots-typesafe") == 7
    assert error_counts.get("parallel-labels-typesafe") == 1
    assert error_counts.get("parallel-slots-typesafe") == 1
    for combo in ("naive_local-slots-bundled", "parallel-labels-bundled", "parallel-slots-bundled"):
        assert error_counts.get(combo) == 0, f"{combo}: expected 0 errors"
