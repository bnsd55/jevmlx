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
    # parity-gates: the dataset lock lives at the MODEL folder level as
    # <dataset>.dataset.lock.json (the parent of the combo), and run.json
    # carries dataset_lock_sha256. The combo folder does NOT have a
    # per-combo dataset.lock.json.
    import hashlib

    lock_content = json.dumps({"source": "fix"}) + "\n"
    lock_path = folder.parent / "fix.dataset.lock.json"
    lock_path.write_text(lock_content)
    run["config"]["dataset_lock_sha256"] = hashlib.sha256(lock_content.encode()).hexdigest()
    (folder / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
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


def test_ordinal_keys_accepted(tmp_path):
    """ordered/ordinal/ordinal_choices (OrdinalTelemetry, ordered=True enum
    fields) are optional add-on keys, not contract violations. The 7B
    typesafe combos failed 'unexpected keys [ordinal, ordinal_choices]' until
    the contract was updated."""
    folder = tmp_path / "combo"
    rec1 = _record()
    rec1["ordered"] = True
    rec1["ordinal_choices"] = ["low", "medium", "high"]
    rec1["ordinal"] = {"argmax_level": 2, "expected_index": 2.0, "variance": 0.0}
    rec2 = _record(case_id="c2", label="LOW", prediction="HIGH", correct=False, probability=0.3)
    _write_valid(folder, records=[rec1, rec2])
    ok, problems = check_folder(folder)
    assert ok, problems


def test_unknown_key_still_fails(tmp_path):
    """An unrecognized key (not in the contract, not in the optional add-on
    list) is still a contract violation."""
    folder = tmp_path / "combo"
    rec = _record()
    rec["totally_unknown_key"] = 42
    _write_valid(folder, records=[rec])
    ok, problems = check_folder(folder)
    assert not ok
    assert any("unexpected keys" in p and "totally_unknown_key" in p for p in problems)


def test_missing_dataset_lock_fails(tmp_path):
    folder = tmp_path / "model" / "combo"
    _write_valid(folder)
    # Remove the model-folder lock (the one the bench writes).
    (folder.parent / "fix.dataset.lock.json").unlink()
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
                    "status": "PASS",
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

    def test_drift_parity_passes_with_note(self, tmp_path):
        """parity-gates: status=DRIFT is publishable (OK) — the note is
        informational (batch-shape noise, not a regression). Before the
        fix, DRIFT set passed=False and check_parity returned FAIL."""
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                    "test": "test_w1a_scoring_parity_batch_vs_chunked_real_model",
                    "passed": False,
                    "status": "DRIFT",
                    "max_abs_drift_nats": 0.078,
                    "max_raw_row_drift_nats": 0.03,
                    "max_gap_drift_nats": 0.078,
                    "max_margin_drift_nats": 0.05,
                    "atol": 0.05,
                    "winners_identical": True,
                    "drift_envelope": {"band": 0.14},
                    "run_at": "2026-09-18T12:00:00Z",
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert ok, f"DRIFT should be OK (publishable), got problems={problems}"
        assert len(problems) == 1
        assert "DRIFT" in problems[0]
        assert "batched drift" in problems[0]
        assert "inside envelope band 0.14" in problems[0]

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

    def test_drift_status_prints_drift_word(self, tmp_path):
        """parity-gates: a parity.json with status=DRIFT (drift >= atol,
        winners identical, inside envelope band) is OK (publishable) and
        prints the DRIFT sentence as an informational note. Before the fix,
        DRIFT set passed=False and check_parity returned FAIL."""
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "model": "m",
                    "test": "w1a",
                    "passed": False,
                    "status": "DRIFT",
                    "max_abs_drift_nats": 0.051,
                    "max_raw_row_drift_nats": 0.03,
                    "max_gap_drift_nats": 0.051,
                    "max_margin_drift_nats": 0.051,
                    "winners_identical": True,
                    "atol": 0.05,
                    "drift_envelope": {"band": 0.141, "shape_bucket": "M>16"},
                    "run_at": "2026-09-20T12:00:00Z",
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert ok  # DRIFT is publishable (batch-shape noise, not a regression)
        msg = problems[0]  # the informational note
        assert "DRIFT:" in msg
        assert "winners identical" in msg
        assert "inside envelope band 0.141" in msg
        assert "envelope band 0.141" in msg

    def test_fail_status_prints_fail_word(self, tmp_path):
        """P4/I7: a parity.json with a winner changed (status=FAIL) prints
        'FAIL: ...' not 'DRIFT: ...'."""
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "model": "m",
                    "test": "w1a",
                    "passed": False,
                    "status": "FAIL",
                    "max_abs_drift_nats": 0.15,
                    "max_raw_row_drift_nats": 0.02,
                    "max_gap_drift_nats": 0.15,
                    "max_margin_drift_nats": 0.15,
                    "winners_identical": False,
                    "atol": 0.05,
                    "run_at": "2026-09-20T12:00:00Z",
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert not ok
        msg = problems[0]
        assert "FAIL:" in msg
        assert "DRIFT:" not in msg

    def test_pass_status_passes(self, tmp_path):
        """P4/I7: a parity.json with status=PASS passes check_parity."""
        from benchmarks.check_results import check_parity

        (tmp_path / "parity.json").write_text(
            json.dumps(
                {
                    "model": "m",
                    "test": "w1a",
                    "passed": True,
                    "status": "PASS",
                    "max_abs_drift_nats": 0.02,
                    "max_raw_row_drift_nats": 0.03,
                    "max_gap_drift_nats": 0.01,
                    "max_margin_drift_nats": 0.01,
                    "winners_identical": True,
                    "atol": 0.05,
                    "run_at": "2026-09-20T12:00:00Z",
                }
            )
        )
        ok, problems = check_parity(tmp_path)
        assert ok

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
        # Results contract v2: _local_rows reads the per-item end-to-end
        # median from timing.json — without it the leaderboard FAILS.
        (combo / "timing.json").write_text(
            json.dumps({"median": {"per_item_end_to_end_ms": 810.0}}), encoding="utf-8"
        )
        rows = _local_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["model"] == "qwen7b"


def test_check_folder_lock_sha_mismatch_fails(tmp_path):
    """parity-gates (review fix 1): the dataset lock lives at the MODEL
    folder level as <dataset>.dataset.lock.json; check_results verifies
    its sha256 matches run.json's dataset_lock_sha256."""

    model_dir = tmp_path / "model"
    folder = model_dir / "combo"
    _write_valid(folder)
    # Corrupt the lock: rewrite with different content.
    lock_path = model_dir / "fix.dataset.lock.json"
    lock_path.write_text(json.dumps({"source": "different"}) + "\n")
    # run.json's sha still points at the original content.
    ok, problems = check_folder(folder)
    assert not ok
    assert any("sha256 mismatch" in p for p in problems)


def test_check_folder_missing_model_lock_fails(tmp_path):
    """A combo whose model folder has no <dataset>.dataset.lock.json fails."""
    model_dir = tmp_path / "model"
    folder = model_dir / "combo"
    _write_valid(folder)
    (model_dir / "fix.dataset.lock.json").unlink()
    ok, problems = check_folder(folder)
    assert not ok
    assert any("fix.dataset.lock.json" in p and "missing" in p for p in problems)


def test_leaderboard_cases_from_run_json_counts(tmp_path):
    """parity-gates (review fix 3): the 'Cases' column comes from run.json's
    counts.cases (the source of truth), not from the agreement metrics'
    n_cases (which can undercount when a case has no valid prediction)."""
    from benchmarks.leaderboard import _local_rows

    model_dir = tmp_path / "m2pro--qwen7b"
    combo = model_dir / "parallel-trie-typesafe"
    combo.mkdir(parents=True)
    (model_dir / "parity.json").write_text(
        json.dumps({"passed": True, "status": "PASS", "max_abs_drift_nats": 0.01, "atol": 0.05})
    )
    (combo / "report.json").write_text(
        json.dumps({"metrics": {"agreement": {"agreement_common_subset": 0.9, "n_cases": 44}}})
    )
    (combo / "run.json").write_text(
        json.dumps(
            {
                "run_id": "x",
                "environment": {},
                "config": {"track": "parallel", "dataset_path": "typesafe", "model": "qwen7b"},
                "counts": {"cases": 45, "fields": 45, "prediction_lines": 44},
            }
        )
    )
    (combo / "timing.json").write_text(
        json.dumps({"median": {"per_item_end_to_end_ms": 590.0}}), encoding="utf-8"
    )
    rows = _local_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["cases"] == 45  # from run.json counts.cases, not n_cases=44
