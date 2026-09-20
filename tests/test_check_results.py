"""Tests for benchmarks/check_results.py: valid folder passes, mismatched
report fails, a line missing a key fails. No MLX, no model."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.check_results import check_folder, main
from jevmlx.evalmetrics import compute_metrics
from jevmlx.evalreport import environment, write_report

FIXTURE = Path(__file__).parent / "fixtures" / "results_ok"


def _record(**overrides):
    base = {
        "run_id": "fix",
        "case_id": "c1",
        "group_id": "g",
        "source": "fix",
        "workflow": None,
        "field": "risk",
        "type": "enum",
        "track": "parallel",
        "model": "fake",
        "permutation": "canonical",
        "label": "HIGH",
        "prediction": "HIGH",
        "valid": True,
        "correct": True,
        "log_scores": {"LOW": -2.0, "HIGH": -0.1},
        "probability": 0.7,
        "per_option": None,
        "latency_ms": 5.0,
        "per_item_end_to_end_ms": 4.5,
        "rows": 2,
        "passes": 1,
        "error": None,
        "salvage_prediction": None,
    }
    base.update(overrides)
    return base


def _write_valid(folder: Path, records=None):
    folder.mkdir(parents=True, exist_ok=True)
    if records is None:
        records = [
            _record(),
            _record(case_id="c2", label="LOW", prediction="HIGH", correct=False, probability=0.3),
        ]
    with open(folder / "predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    run = {
        "run_id": "fix",
        "environment": environment(),
        "config": {
            "model": "fake",
            "track": "parallel",
            "temperature": 1.0,
            "dataset_path": "fix",
            "permutations": "none",
            "split": "all",
            "model_revision": None,
            "quantization": None,
            "prompt_version": None,
        },
        "counts": {"cases": 2, "fields": 2, "prediction_lines": len(records)},
    }
    (folder / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    (folder / "dataset.lock.json").write_text(json.dumps({"source": "fix"}) + "\n")
    # Results contract v2: the parallel track's honest timing split medians
    # (the same keys the engine's _meta carries, incl. failed_attempts and
    # peak memory).
    (folder / "timing.json").write_text(
        json.dumps(
            {
                "calls": 2,
                "median": {
                    "latency_ms": 5.5,
                    "prior_ms": 0.0,
                    "prefill_ms": 2.0,
                    "plan_compile_ms": 0.1,
                    "cache_broadcast_ms": 0.2,
                    "suffix_eval_ms": 2.5,
                    "lm_head_gather_ms": 0.3,
                    "second_pass_ms": 0.0,
                    "total_ms": 5.5,
                    "per_item_end_to_end_ms": 4.5,
                    "peak_active_bytes": 1024,
                    "peak_incremental_bytes": 512,
                    "failed_attempts": 0,
                    "padded_token_positions": 6,
                    "naive_branch_prompt_tokens": 100,
                    "shared_prefix_tokens": 50,
                    "logical_suffix_token_positions": 10,
                    "computed_suffix_token_positions": 12,
                    "computed_prompt_token_positions": 62,
                    "retry_wasted_ms": 0.0,
                    "rescored_fields_count": 0,
                    "rerun_fields_count": 0,
                    "num_fields": 1,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_report(
        folder / "report.json",
        {"environment": environment(), "metrics": compute_metrics(records)},
    )


def test_valid_folder_passes(tmp_path):
    folder = tmp_path / "combo"
    _write_valid(folder)
    ok, problems = check_folder(folder)
    assert ok, problems
    assert problems == []


def test_committed_fixture_passes():
    """The fixture committed under tests/fixtures/results_ok is itself valid."""
    ok, problems = check_folder(FIXTURE)
    assert ok, problems


def test_mismatched_report_fails(tmp_path):
    folder = tmp_path / "combo"
    _write_valid(folder)
    # Corrupt one metric in the committed report.json.
    report = json.loads((folder / "report.json").read_text())
    report["metrics"]["accuracy"] = 0.0
    (folder / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ok, problems = check_folder(folder)
    assert not ok
    assert any("accuracy" in p for p in problems)


def test_line_missing_key_fails(tmp_path):
    folder = tmp_path / "combo"
    records = [_record()]
    # Drop a required key from the only line.
    del records[0]["correct"]
    _write_valid(folder, records=records)
    ok, problems = check_folder(folder)
    assert not ok
    assert any("missing keys" in p and "correct" in p for p in problems)


def test_wrong_type_fails(tmp_path):
    folder = tmp_path / "combo"
    records = [_record(valid="yes")]  # valid must be bool
    _write_valid(folder, records=records)
    ok, problems = check_folder(folder)
    assert not ok
    assert any("valid" in p and "wrong type" in p for p in problems)


def test_missing_dataset_lock_fails(tmp_path):
    folder = tmp_path / "combo"
    _write_valid(folder)
    (folder / "dataset.lock.json").unlink()
    ok, problems = check_folder(folder)
    assert not ok
    assert any("dataset.lock.json" in p for p in problems)


def test_missing_run_key_fails(tmp_path):
    folder = tmp_path / "combo"
    _write_valid(folder)
    run = json.loads((folder / "run.json").read_text())
    del run["counts"]
    (folder / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    ok, problems = check_folder(folder)
    assert not ok
    assert any("counts" in p for p in problems)


def test_main_exits_1_on_failure(tmp_path, capsys):
    folder = tmp_path / "combo"
    _write_valid(folder)
    report = json.loads((folder / "report.json").read_text())
    report["metrics"]["accuracy"] = 0.99
    (folder / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rc = main([str(folder)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL" in captured.out


def test_main_exits_0_on_success(tmp_path, capsys):
    folder = tmp_path / "combo"
    _write_valid(folder)
    rc = main([str(folder)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "OK" in captured.out


def test_summary_row_has_majority_and_exact_columns(tmp_path):
    """P7: SUMMARY.md rows carry 'majority' (mean over fields) and 'exact'
    columns next to field accuracy."""
    from benchmarks.summarize_results import _row_from_folder

    folder = tmp_path / "combo"
    _write_valid(folder)
    row = _row_from_folder(folder)
    assert row is not None
    assert "majority_baseline" in row
    assert "exact_record" in row
    # Both are numeric (compute_metrics always produces them for labelled
    # records).
    assert row["majority_baseline"] is not None
    assert row["exact_record"] is not None


def test_imports_without_mlx(monkeypatch):
    """The check path must import on a no-MLX CI runner (ubuntu-latest)."""
    import sys

    # Simulate mlx absent by marking the modules as None (import raises).
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    # Force a re-import of the check module's dependencies.
    # W5-C-fix: jevmlx.calibrate is NOT an mlx dependency and holds the
    # CalibrationBundle class; evicting it while jevmlx.engine survives
    # makes isinstance(bundle, CalibrationBundle) stale across the suite
    # (engine holds the old class object, later imports get a new one).
    # Keep it resident so the typed object's identity is stable.
    for mod in list(sys.modules):
        if mod.startswith("jevmlx.") and mod not in (
            "jevmlx.engine",
            "jevmlx.calibrate",
        ):
            del sys.modules[mod]
    import importlib

    import benchmarks.check_results  # already imported; re-import deps

    importlib.reload(benchmarks.check_results)
    # compute_metrics / environment must be callable without mlx.
    records = [_record()]
    metrics = benchmarks.check_results.compute_metrics(records)
    assert isinstance(metrics, dict)


def test_bench_machine_tag_patches_live_module(monkeypatch):
    """Guard for the module-eviction above: machine_tag must read the _sysctl
    binding in ITS OWN live module (whatever instance sys.modules holds now).

    When test_imports_without_mlx evicts jevmlx.* and a later file's
    module-level `from jevmlx.bench import machine_tag` still points at the
    evicted instance, monkeypatching the re-imported module's _sysctl would
    patch a dict machine_tag never reads — the tag silently falls back to
    real sysctl. Pinning through machine_tag.__globals__ catches that rot
    regardless of import order."""
    from jevmlx.bench import machine_tag

    monkeypatch.setitem(
        machine_tag.__globals__,
        "_sysctl",
        lambda args: (
            "Apple M5 Max"
            if args == ["machdep.cpu.brand_string"]
            else (str(128 * 2**30) if args == ["hw.memsize"] else None)
        ),
    )
    assert machine_tag() == "m5max-128gb"


class TestParityGate:
    """W4-A: a model enters the README compat table only with a passing
    slow parity test (parity.json in the model folder)."""

    def test_passing_parity_passes(self, tmp_path):
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                    "prompt_version": "jevmlx-parallel-v8",
                    "test": "test_w1a_scoring_parity_batch_vs_chunked_real_model",
                    "passed": True,
                    "max_abs_drift_nats": 0.027,
                    "max_raw_row_drift_nats": 0.031,
                    "atol": 0.05,
                    "max_gap_drift_nats": 0.01,
                    "run_at": "2026-09-18T12:00:00Z",
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert ok
        assert problems == []

    def test_missing_parity_fails(self, tmp_path):
        from benchmarks.check_results import check_parity

        ok, problems = check_parity(tmp_path)
        assert not ok
        assert any("missing parity.json" in p for p in problems)

    def test_failed_parity_fails(self, tmp_path):
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                    "test": "test_w1a_scoring_parity_batch_vs_chunked_real_model",
                    "passed": False,
                    "max_abs_drift_nats": 0.15,
                    "max_raw_row_drift_nats": 0.02,
                    "atol": 0.05,
                    "max_gap_drift_nats": 0.01,
                    "max_margin_drift_nats": 0.01,
                    "winners_identical": True,
                    "run_at": "2026-09-18T12:00:00Z",
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert not ok
        assert any("did not pass" in p and "log-score drift" in p for p in problems)

    def test_gap_drift_stage_named(self, tmp_path):
        """A batched pairwise GAP drift failure names THAT stage (W5c-1
        review section 2): the final-decision drift can stay inside the
        band while the pairwise gaps drift past it — raw-logit drift is a
        diagnostic, the gap is the gate."""
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "passed": False,
                    "max_abs_drift_nats": 0.01,
                    "max_raw_row_drift_nats": 0.2,
                    "max_gap_drift_nats": 0.2,
                    "max_margin_drift_nats": 0.01,
                    "atol": 0.05,
                    "winners_identical": True,
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert not ok
        assert any("batched pairwise gap drift" in p for p in problems)

    def test_pre_v2_parity_payload_fails(self, tmp_path):
        """A v1 parity.json (no raw-gate key) fails: the bench regenerates
        it with the batched raw gate before the model enters the table."""
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "model": "m",
                    "test": "w1a",
                    "passed": True,
                    "max_drift_nats": 0.02,
                    "atol": 0.05,
                    "run_at": "2026-09-18T12:00:00Z",
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert not ok
        assert any("pre-v2" in p for p in problems)

    def test_missing_gap_drift_key_fails(self, tmp_path):
        """W5c-1 review: a v2 payload that carries max_raw_row_drift_nats but
        is missing max_gap_drift_nats is a CONTRACT failure (not a fallback)
        — the gated decomposition key is mandatory."""
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "model": "m",
                    "test": "w1a",
                    "passed": True,
                    "max_abs_drift_nats": 0.02,
                    "max_raw_row_drift_nats": 0.03,
                    "atol": 0.05,
                    "run_at": "2026-09-18T12:00:00Z",
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert not ok
        assert any("pre-v2" in p and "max_gap_drift_nats" in p for p in problems)

    def test_corrupt_parity_fails(self, tmp_path):
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text("{not valid json")
        ok, problems = check_parity(tmp_path)
        assert not ok
        assert any("unreadable" in p for p in problems)

    def test_leaderboard_excludes_model_without_parity(self, tmp_path):
        """_local_rows skips a model folder that has no parity.json."""
        from benchmarks.leaderboard import _local_rows

        model_dir = tmp_path / "m2pro--qwen2.5-7b"
        combo = model_dir / "parallel-trie-typesafe"
        combo.mkdir(parents=True)
        # No parity.json — model must be excluded.
        (combo / "report.json").write_text(
            json.dumps({"metrics": {"agreement": {"agreement_common_subset": 0.9}}})
        )
        (combo / "run.json").write_text(
            json.dumps(
                {
                    "run_id": "x",
                    "environment": {},
                    "config": {"track": "parallel", "dataset_path": "typesafe", "model": "qwen7b"},
                    "counts": {"cases": 1, "fields": 1, "prediction_lines": 1},
                }
            )
        )
        rows = _local_rows(tmp_path)
        assert rows == []

    def test_leaderboard_includes_model_with_passing_parity(self, tmp_path):
        """_local_rows includes a model folder that has a passing parity.json."""
        from benchmarks.leaderboard import _local_rows

        model_dir = tmp_path / "m2pro--qwen2.5-7b"
        combo = model_dir / "parallel-trie-typesafe"
        combo.mkdir(parents=True)
        (model_dir / "parity.json").write_text(
            json.dumps(
                {
                    "model": "qwen7b",
                    "test": "w1a",
                    "passed": True,
                    "max_drift_nats": 0.02,
                    "atol": 0.05,
                }
            )
        )
        (combo / "report.json").write_text(
            json.dumps({"metrics": {"agreement": {"agreement_common_subset": 0.9, "n_cases": 20}}})
        )
        (combo / "run.json").write_text(
            json.dumps(
                {
                    "run_id": "x",
                    "environment": {},
                    "config": {"track": "parallel", "dataset_path": "typesafe", "model": "qwen7b"},
                    "counts": {"cases": 1, "fields": 1, "prediction_lines": 1},
                }
            )
        )
        # Results contract v2: _local_rows reads ONLY the per-item end-to-end
        # median — without it the leaderboard FAILS the folder.
        (combo / "predictions.jsonl").write_text(
            json.dumps(
                {
                    "case_id": "c1",
                    "field": "f",
                    "per_item_end_to_end_ms": 810.0,
                    "valid": True,
                    "correct": True,
                    "label": "A",
                    "prediction": "A",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rows = _local_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["model"] == "qwen7b"
