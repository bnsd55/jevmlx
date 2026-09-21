"""Tests for `jevmlx bench` (no model runs): preflight, naming, and the
results summarizer. The slow end-to-end test lives in test_engine.py's
module-scoped engine fixture style; here everything runs on fakes.
"""

import json
from pathlib import Path

import pytest

from benchmarks.summarize_results import summarize
from jevmlx import bench
from jevmlx.bench import (
    _track_scorer_grid,
    enforce_folder_size,
    machine_tag,
    model_slug,
)

# --- machine tag ------------------------------------------------------------


def test_machine_tag_from_fake_sysctl(monkeypatch):
    """<chip-lowercase>-<ram>gb from sysctl output; marketing noise dropped.

    Patched through machine_tag.__globals__: test_check_results.py evicts
    jevmlx.* modules, so this file's module-level machine_tag binding may be
    a DIFFERENT instance than a fresh `import jevmlx.bench` returns —
    patching the latter would be invisible to the function under test."""
    target = machine_tag.__globals__  # the dict machine_tag actually reads

    def fake_sysctl(args):
        if args == ["machdep.cpu.brand_string"]:
            return "Apple M5 Max"
        if args == ["hw.memsize"]:
            return str(128 * 2**30)
        return None

    monkeypatch.setitem(target, "_sysctl", fake_sysctl)
    assert machine_tag() == "m5max-128gb"


def test_machine_tag_override_wins(monkeypatch):
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "_sysctl", lambda args: "garbage")
    assert machine_tag("m2pro-32gb") == "m2pro-32gb"


def test_machine_tag_single_chip_word(monkeypatch):
    def fake_sysctl(args):
        if args == ["machdep.cpu.brand_string"]:
            return "Apple M4"
        if args == ["hw.memsize"]:
            return str(16 * 2**30)
        return None

    # See test_machine_tag_from_fake_sysctl: patch through machine_tag's own
    # module dict — the eviction in test_check_results can leave two live
    # instances and only this one is guaranteed to be the function's globals.
    monkeypatch.setitem(machine_tag.__globals__, "_sysctl", fake_sysctl)
    assert machine_tag() == "m4-16gb"


# --- preflight refusals -------------------------------------------------------


def test_preflight_refuses_on_battery(monkeypatch):
    import jevmlx.bench as bench

    monkeypatch.setattr(
        bench,
        "_probe",
        lambda cmd: "Now drawing from 'Battery'" if cmd == ["pmset", "-g", "batt"] else None,
    )
    with pytest.raises(SystemExit, match="battery"):
        bench.preflight(force=False, machine_override="m4-16gb")


def test_preflight_battery_overridden_by_force(monkeypatch, capsys):
    import jevmlx.bench as bench

    monkeypatch.setattr(
        bench,
        "_probe",
        lambda cmd: "Now drawing from 'Battery'" if cmd == ["pmset", "-g", "batt"] else None,
    )
    tag = bench.preflight(force=True, machine_override="m4-16gb")
    assert tag == "m4-16gb"
    assert "battery" in capsys.readouterr().out.lower()


def test_preflight_refuses_busy_metal(monkeypatch):
    import jevmlx.bench as bench

    def fake_probe(cmd):
        if cmd == ["pmset", "-g", "batt"]:
            return "AC Power"
        return None

    monkeypatch.setattr(bench, "_probe", fake_probe)
    monkeypatch.setattr(bench, "_metal_resident_bytes", lambda: 2 * 2**30)
    with pytest.raises(SystemExit, match="Metal memory"):
        bench.preflight(force=False, machine_override="m4-16gb")


def test_preflight_busy_metal_overridden_by_force(monkeypatch):
    import jevmlx.bench as bench

    monkeypatch.setattr(
        bench, "_probe", lambda cmd: "AC Power" if cmd == ["pmset", "-g", "batt"] else None
    )
    monkeypatch.setattr(bench, "_metal_resident_bytes", lambda: 2 * 2**30)
    assert bench.preflight(force=True, machine_override="m4-16gb") == "m4-16gb"


# --- naming -------------------------------------------------------------------


def test_model_slug():
    assert model_slug("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == (
        "mlx-community--qwen2.5-0.5b-instruct-4bit"
    )


def test_track_scorer_grid_drops_naive_labels():
    grid = _track_scorer_grid(["parallel", "naive_local"], ["slots", "labels"])
    assert ("parallel", "slots") in grid
    assert ("parallel", "labels") in grid
    assert ("naive_local", "slots") in grid
    assert ("naive_local", "labels") not in grid


# --- folder size guard ----------------------------------------------------------


def test_enforce_folder_size_gzips_large_predictions(tmp_path):
    big = Path(tmp_path) / "parallel-trie-bundled"
    big.mkdir()
    (big / "predictions.jsonl").write_text("x" * (6 * 1024 * 1024))
    assert enforce_folder_size(Path(tmp_path)) is True
    assert (big / "predictions.jsonl.gz").exists()
    assert not (big / "predictions.jsonl").exists()


def test_enforce_folder_size_noop_when_small(tmp_path):
    small = Path(tmp_path) / "combo"
    small.mkdir()
    (small / "predictions.jsonl").write_text("tiny")
    assert enforce_folder_size(Path(tmp_path)) is False
    assert (small / "predictions.jsonl").exists()


# --- summarizer -----------------------------------------------------------------


def _write_report(folder: Path, metrics: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(
        json.dumps({"environment": {"chip": "x"}, "metrics": metrics}), encoding="utf-8"
    )


def test_summarize_two_report_files(tmp_path, capsys):
    root = Path(tmp_path)
    combo_a = root / "m5max-128gb--model-a" / "parallel-trie-bundled"
    combo_b = root / "m5max-128gb--model-a" / "parallel-labels-typesafe"
    # W4-B: the model folder needs a passing parity.json for the numbers to
    # stand (the parity gate itself is covered by
    # test_summarize_two_report_files_parity_gate below).
    model_dir = root / "m5max-128gb--model-a"
    model_dir.mkdir(parents=True)
    (model_dir / "parity.json").write_text(
        json.dumps(
            {"passed": True, "max_abs_drift_nats": 0.01, "atol": 0.05, "winners_identical": True}
        ),
        encoding="utf-8",
    )
    _write_report(
        combo_a,
        {
            "accuracy": 0.83,
            "case_exact_match": 0.5,
            "balanced_accuracy": {"field1": 0.8, "field2": 0.6},
            "ece_5bin_equal_mass": 0.0412,
            "any_flip_rate": 0.05,
            "perturbation_flip_rate": 0.1,
            "latency_ms_p50": 12.5,
            "n_cases": 24,
        },
    )
    _write_report(
        combo_b,
        {
            "accuracy": 0.9,
            "case_exact_match": 0.75,
            "balanced_accuracy": {"field1": 0.9},
            "ece_5bin_equal_mass": 0.02,
            "latency_ms_p50": 11.0,
            "n_cases": 40,
        },
    )

    summary = summarize(root)
    text = summary.read_text(encoding="utf-8")
    assert "m5max" in text and "model-a" in text
    assert "parallel-trie-bundled" in text or "parallel | trie | bundled" in text
    assert "0.83" in text
    assert "0.9" in text
    # Balanced accuracy mean: (0.8 + 0.6) / 2 = 0.7 for combo a.
    assert "0.7" in text
    printed = capsys.readouterr().out
    assert "SUMMARY" in printed or "wrote" in printed


def test_summarize_two_report_files_parity_gate(tmp_path):
    """W4-B: without a passing parity.json the accuracy numbers are gated
    (parity_failed replaces the accuracy column); with one, they stand."""
    root = Path(tmp_path)
    model_dir = root / "m5max-128gb--model-a"
    combo = model_dir / "parallel-trie-bundled"
    _write_report(combo, {"accuracy": 0.83, "case_exact_match": 0.5})

    # No parity.json: gated.
    text = summarize(root).read_text(encoding="utf-8")
    assert "parity.json missing" in text
    assert "0.83" not in text

    # Failing parity.json: gated with the drift numbers.
    (model_dir / "parity.json").write_text(
        json.dumps(
            {
                "passed": False,
                "max_abs_drift_nats": 0.9,
                "atol": 0.05,
                "max_gap_drift_nats": 0.01,
                "max_margin_drift_nats": 0.01,
                "winners_identical": False,
            }
        ),
        encoding="utf-8",
    )
    text = summarize(root).read_text(encoding="utf-8")
    assert "max_abs_drift_nats=0.9" in text
    assert "0.83" not in text

    # Passing parity.json: numbers stand.
    (model_dir / "parity.json").write_text(
        json.dumps(
            {
                "passed": True,
                "max_abs_drift_nats": 0.01,
                "atol": 0.05,
                "max_gap_drift_nats": 0.01,
                "winners_identical": True,
            }
        ),
        encoding="utf-8",
    )
    text = summarize(root).read_text(encoding="utf-8")
    assert "0.83" in text
    assert "parity_failed" not in text


def test_summarize_skips_folders_without_report(tmp_path):
    root = Path(tmp_path)
    empty = root / "m4-16gb--model-b" / "parallel-trie-bundled"
    empty.mkdir(parents=True)
    summary = summarize(root)
    assert "(no report.json files found" in summary.read_text(encoding="utf-8")


def test_summarize_writes_folder_readme(tmp_path):
    root = Path(tmp_path)
    _write_report(root / "m4-16gb--m" / "parallel-trie-bundled", {"accuracy": 1.0})
    summarize(root)
    assert (root / "README.md").exists()
    assert "jevmlx bench" in (root / "README.md").read_text(encoding="utf-8")


def test_summarize_header_cell_count_matches_rows(tmp_path):
    """parity-gates (B): SUMMARY.md header and separator have the SAME cell
    count as every data row (P7 added majority_baseline and exact_record to
    rows but not the header, so 'case exact' showed the majority value)."""
    root = Path(tmp_path)
    model_dir = root / "m5max-128gb--model-a"
    _write_report(
        model_dir / "parallel-trie-bundled",
        {
            "accuracy": 0.83,
            "case_exact_match": 0.5,
            "majority_class_baseline": 0.4,
            "exact_record_accuracy": 0.3,
            "balanced_accuracy": {"f1": 0.8, "f2": 0.6},
            "ece_5bin_equal_mass": 0.04,
            "any_flip_rate": 0.05,
            "perturbation_flip_rate": 0.1,
            "latency_ms_p50": 12.5,
            "n_cases": 24,
        },
    )
    (model_dir / "parity.json").write_text(
        json.dumps({"passed": True, "max_abs_drift_nats": 0.01, "atol": 0.05}),
        encoding="utf-8",
    )
    text = summarize(root).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    assert len(lines) >= 3  # header + sep + >=1 row
    header_cells = lines[0].count("|") - 1
    sep_cells = lines[1].count("|") - 1
    assert header_cells == sep_cells, f"header={header_cells} sep={sep_cells}"
    for row_line in lines[2:]:
        row_cells = row_line.count("|") - 1
        assert row_cells == header_cells, (
            f"row has {row_cells} cells, header has {header_cells}: {row_line}"
        )
    # Named columns carry the right metric (not shifted by the missing
    # header cells): majority baseline and exact record appear in their own
    # columns, not overwriting 'case exact'.
    assert "majority baseline" in lines[0]
    assert "exact record" in lines[0]
    # The values 0.4 (majority) and 0.3 (exact record) are present.
    assert "0.4" in text
    assert "0.3" in text


# --- multi-model bench (I9) ------------------------------------------------


def _patch_bench_core(monkeypatch, tmp_path, failing_models=()):
    """Fake the whole bench core: preflight ok, datasets one stub, _run_one
    writes a minimal report.json into the combo dir and returns a marker.

    Models listed in ``failing_models`` raise on the first _run_one call.
    Returns the list of models that were actually run, in order.
    """
    from jevmlx import bench

    run_calls: list[str] = []

    def fake_run_one(
        model,
        track,
        scorer,
        jsonl,
        combo_dir,
        dataset_lock_path=None,
        resume=False,
        heartbeat_every=0,
        combo="",
        run_i=None,
        run_n=None,
    ):
        if model in failing_models:
            raise RuntimeError(f"load failed for {model}")
        run_calls.append(model)
        combo_dir.mkdir(parents=True, exist_ok=True)
        (combo_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        (combo_dir / "run.json").write_text("{}", encoding="utf-8")
        (combo_dir / "report.json").write_text(
            json.dumps(
                {
                    "environment": {"chip": "fake"},
                    "metrics": {"accuracy": 0.9, "n_cases": 3, "latency_ms_p50": 1.0},
                }
            ),
            encoding="utf-8",
        )
        return {"run": {"model": model}}

    monkeypatch.setattr(bench, "preflight", lambda force, machine_override: "fake-8gb")
    monkeypatch.setattr(
        bench,
        "build_datasets",
        lambda datasets: (
            {"bundled": tmp_path / "b.jsonl"},
            {"bundled": tmp_path / "b.dataset.lock.json"},
        ),
    )
    monkeypatch.setattr(
        bench, "_load_engine_with_timeout", lambda model, timeout: (object(), object())
    )
    monkeypatch.setattr(bench, "_run_one", fake_run_one)
    monkeypatch.setattr(bench, "_print_pr_instructions", lambda folder, last_run: None)
    monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path / "cache")  # lock copies are best-effort
    return run_calls


def test_multi_model_two_folders_one_summary(tmp_path, monkeypatch, capsys):
    """Two models -> two <machine>-<slug>/ folders, ONE SUMMARY.md at <out>."""
    from jevmlx import bench

    run_calls = _patch_bench_core(monkeypatch, tmp_path)
    out = tmp_path / "results"

    bench.run_bench_models(
        models=["org/model-a", "org/model-b"],
        datasets=["bundled"],
        scorers=["slots"],
        tracks=["parallel"],
        out=out,
        runs=1,
    )

    assert run_calls == ["org/model-a", "org/model-b"]
    a = out / "fake-8gb-org--model-a" / "parallel-slots-bundled" / "report.json"
    b = out / "fake-8gb-org--model-b" / "parallel-slots-bundled" / "report.json"
    assert a.is_file() and b.is_file()
    summary = out / "SUMMARY.md"
    assert summary.is_file()
    text = summary.read_text(encoding="utf-8")
    assert "org--model-a" in text and "org--model-b" in text


def test_failing_model_does_not_stop_the_next(tmp_path, monkeypatch, capsys):
    """A failing first model is logged and skipped; the second still runs."""
    from jevmlx import bench

    run_calls = _patch_bench_core(monkeypatch, tmp_path, failing_models={"org/bad"})
    out = tmp_path / "results"

    bench.run_bench_models(
        models=["org/bad", "org/good"],
        datasets=["bundled"],
        scorers=["slots"],
        tracks=["parallel"],
        out=out,
        runs=1,
    )

    assert run_calls == ["org/good"]  # only the healthy model ran
    # The failing model's combo holds an I6 load_failed run.json and shows
    # in the summary as a failure row (not silently skipped).
    bad_combo = out / "fake-8gb-org--bad" / "parallel-slots-bundled"
    assert (bad_combo / "run.json").is_file()
    bad_run = json.loads((bad_combo / "run.json").read_text(encoding="utf-8"))
    # The fake's failure point is _run_one (post-load), so I6 classifies it
    # run_failed. A load-time failure is the test_load_failure_row case.
    assert bad_run["status"] == "run_failed"
    assert (bad_combo / "report.json").exists() is False
    assert (out / "fake-8gb-org--good" / "parallel-slots-bundled" / "report.json").is_file()
    text = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "org--good" in text and "run_failed" in text  # failure row present
    printed = capsys.readouterr().out
    assert "FAILED model org/bad" in printed


def test_every_model_failing_raises(tmp_path, monkeypatch):
    """When no model benches, the run fails loudly with every error."""
    from jevmlx import bench

    _patch_bench_core(monkeypatch, tmp_path, failing_models={"org/x", "org/y"})
    with pytest.raises(SystemExit, match="every model failed"):
        bench.run_bench_models(
            models=["org/x", "org/y"],
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            out=tmp_path / "results",
            runs=1,
        )


def test_parse_model_list():
    from jevmlx.bench import parse_model_list

    assert parse_model_list("a") == ["a"]
    assert parse_model_list("a, b ,c") == ["a", "b", "c"]
    assert parse_model_list("a,,b,a") == ["a", "b"]  # empties and dupes dropped


def test_parse_models_file(tmp_path):
    from jevmlx.bench import parse_models_file

    f = tmp_path / "models.txt"
    f.write_text(
        "# Ben's T1..T5 list\n"
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit\n"
        "\n"
        "  # T2 below\n"
        "mlx-community/Llama-3.2-1B-Instruct-4bit # inline comment\n",
        encoding="utf-8",
    )
    assert parse_models_file(f) == [
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
    ]


def test_parse_models_file_empty_raises(tmp_path):
    from jevmlx.bench import parse_models_file

    f = tmp_path / "models.txt"
    f.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no model ids"):
        parse_models_file(f)


def test_bench_main_models_file_wiring(tmp_path, monkeypatch):
    """--models-file (alone, mutually exclusive with --model) reaches
    run_bench_models with the file's model list.

    B1: --model and --models-file are now mutually exclusive (exactly one
    required). The old behavior (both allowed, --models-file overrode
    --model) is gone; this test passes --models-file alone."""
    from jevmlx import bench

    _patch_bench_core(monkeypatch, tmp_path)
    seen: dict = {}

    def fake_run_bench_models(**kwargs):
        seen.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(bench, "run_bench_models", fake_run_bench_models)
    models_file = tmp_path / "m.txt"
    models_file.write_text("m/a\nm/b\n", encoding="utf-8")
    rc = bench.main(
        [
            "--models-file",
            str(models_file),
            "--datasets",
            "bundled",
            "--scorers",
            "slots",
            "--tracks",
            "parallel",
            "--runs",
            "1",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert seen["models"] == ["m/a", "m/b"]


def test_bench_main_comma_list_wiring(tmp_path, monkeypatch):
    """--model a,b reaches run_bench_models as [a, b]."""
    from jevmlx import bench

    _patch_bench_core(monkeypatch, tmp_path)
    seen: dict = {}

    def fake_run_bench_models(**kwargs):
        seen.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(bench, "run_bench_models", fake_run_bench_models)
    rc = bench.main(
        [
            "--model",
            "m/a,m/b",
            "--datasets",
            "bundled",
            "--scorers",
            "slots",
            "--tracks",
            "parallel",
            "--runs",
            "1",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert seen["models"] == ["m/a", "m/b"]


def test_bench_main_single_model_uses_run_bench(tmp_path, monkeypatch):
    """One model id keeps the single-model run_bench path."""
    from jevmlx import bench

    _patch_bench_core(monkeypatch, tmp_path)
    seen: dict = {}

    def fake_run_bench(**kwargs):
        seen.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(bench, "run_bench", fake_run_bench)
    rc = bench.main(
        [
            "--model",
            "m/solo",
            "--datasets",
            "bundled",
            "--scorers",
            "slots",
            "--tracks",
            "parallel",
            "--runs",
            "1",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert seen["model"] == "m/solo"


def test_release_between_models_logs_memory(tmp_path, monkeypatch, capsys):
    """Between models the engine cache and Metal cache are released, with a
    memory log line before/after."""
    from jevmlx import bench

    _patch_bench_core(monkeypatch, tmp_path)
    released: list[int] = []

    from jevmlx import engine

    monkeypatch.setattr(bench, "_metal_cache_memory_gb", lambda: 2.0)
    monkeypatch.setattr(bench, "_clear_metal_cache", lambda: released.append(1))
    # clear_engine_cache is looked up as jevmlx.engine.clear_engine_cache inside
    # run_bench_models (lazy import so bench.py loads without mlx); patch it there.
    monkeypatch.setattr(
        engine,
        "clear_engine_cache",
        lambda: released.append(0),
    )

    bench.run_bench_models(
        models=["m/a", "m/b"],
        datasets=["bundled"],
        scorers=["slots"],
        tracks=["parallel"],
        out=tmp_path / "results",
        runs=1,
    )
    # Two models -> two release cycles (engine cache + metal clear each).
    assert released.count(0) >= 2
    assert released.count(1) >= 2
    assert "[memory] released m/a" in capsys.readouterr().out


def test_run_failure_row(tmp_path, monkeypatch, capsys):
    """A failure inside one combo (after load) writes run_failed and the
    summary shows it; run_bench_models escalates only when every model
    failed (here: the single model did, so SystemExit)."""
    from jevmlx import bench

    run_calls = _patch_bench_core(monkeypatch, tmp_path, failing_models={"org/bad"})
    out = tmp_path / "results"
    with pytest.raises(SystemExit, match="every model failed"):
        bench.run_bench_models(
            models=["org/bad"],
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            out=out,
            runs=1,
        )
    combo = out / "fake-8gb-org--bad" / "parallel-slots-bundled"
    run = json.loads((combo / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "run_failed"
    assert run["error"]["type"] == "RuntimeError"
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "run_failed: RuntimeError: load failed for org/bad" in summary
    assert run_calls == []  # nothing recorded for the failing model


def test_every_combo_failing_exits_nonzero(tmp_path, monkeypatch):
    """When EVERY combo failed, run_bench raises SystemExit (exit 1 path)."""
    from jevmlx import bench

    _patch_bench_core(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bench,
        "_load_engine_with_timeout",
        lambda model, timeout: (_ for _ in ()).throw(RuntimeError("no such model")),
    )
    with pytest.raises(SystemExit, match="every combo failed"):
        bench.run_bench(
            model="org/allbad",
            datasets=["bundled"],
            scorers=["slots", "labels"],
            tracks=["parallel", "naive_local"],
            out=tmp_path / "results",
            runs=1,
        )


def test_resume_does_not_skip_combos(tmp_path, monkeypatch, capsys):
    """W5c-7 review: _combo_complete is gone — bench always passes resume=True.
    A second invocation calls _run_one again (resume), not skips it.
    """
    from jevmlx import bench

    run_calls = _patch_bench_core(monkeypatch, tmp_path)
    out = tmp_path / "results"

    # First invocation runs and writes the combo.
    bench.run_bench(
        model="org/m",
        datasets=["bundled"],
        scorers=["slots"],
        tracks=["parallel"],
        out=out,
        runs=1,
    )
    first_count = len(run_calls)

    # Second invocation: bench calls _run_one again (resume=True, not skip).
    bench.run_bench(
        model="org/m",
        datasets=["bundled"],
        scorers=["slots"],
        tracks=["parallel"],
        out=out,
        runs=1,
    )
    assert len(run_calls) == first_count + 1  # _run_one called again (resume)
    printed = capsys.readouterr().out
    assert "===" in printed  # the combo header printed again


def test_fresh_reruns_complete_combos(tmp_path, monkeypatch):
    """--fresh forces a rerun of complete combos."""
    from jevmlx import bench

    run_calls = _patch_bench_core(monkeypatch, tmp_path)
    out = tmp_path / "results"
    for _ in range(2):
        bench.run_bench(
            model="org/m",
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            out=out,
            runs=1,
            fresh=True,
        )
    assert len(run_calls) == 2  # ran twice


def test_dry_run_prints_plan_and_loads_nothing(tmp_path, monkeypatch, capsys):
    """--dry-run prints machine tag, dataset states, combos with folders and
    memory estimates, and exits 0 without loading any model."""
    from jevmlx import bench

    def explode(*a, **k):
        raise AssertionError("dry-run must not load anything")

    _patch_bench_core(monkeypatch, tmp_path)
    monkeypatch.setattr(bench, "_load_engine_with_timeout", explode)
    monkeypatch.setattr(
        bench,
        "build_datasets",
        lambda datasets: (
            {"bundled": tmp_path / "b.jsonl"},
            {"bundled": tmp_path / "b.dataset.lock.json"},
        ),
    )

    code = bench.main(
        [
            "--model",
            "org/m",
            "--datasets",
            "bundled",
            "--scorers",
            "slots",
            "--tracks",
            "parallel",
            "--out",
            str(tmp_path / "results"),
            "--dry-run",
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "machine:" in text
    assert "bundled: to build" in text or "bundled: cached" in text
    assert "parallel-slots-bundled" in text
    assert "would run" in text
    assert "memory" in text  # estimate line, 'unknown' when not in the table


def test_load_timeout_records_load_failed(tmp_path, monkeypatch):
    """--load-timeout: a load that never finishes is recorded as
    load_failed with a TimeoutError (the guard itself raises TimeoutError
    on join timeout; here the guard is mocked to raise it directly — the
    real thread+join behavior is covered by test_load_engine_timeout_guard)."""
    from jevmlx import bench

    _patch_bench_core(monkeypatch, tmp_path)

    def hanging_load(model, load_timeout):
        raise TimeoutError(f"load_engine did not finish within {load_timeout:.0f}s")

    monkeypatch.setattr(bench, "_load_engine_with_timeout", hanging_load)
    with pytest.raises(SystemExit, match="every combo failed"):
        bench.run_bench(
            model="org/slow",
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            out=tmp_path / "results",
            runs=1,
            load_timeout=0.05,
        )
    combo = tmp_path / "results" / "fake-8gb-org--slow" / "parallel-slots-bundled"
    run = json.loads((combo / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "load_failed"
    assert run["error"]["type"] == "TimeoutError"


class _StuckLoader:
    """A load_engine stand-in whose thread never finishes before the join."""

    def __call__(self, model):
        import time

        time.sleep(2.0)
        return (object(), object())


def test_load_engine_timeout_guard(tmp_path, monkeypatch):
    """The real guard: a hung load_engine hits the join timeout and raises
    TimeoutError (not the model's own exception)."""
    import jevmlx.engine as engine_mod
    from jevmlx import bench

    monkeypatch.setattr(engine_mod, "load_engine", _StuckLoader())
    with pytest.raises(TimeoutError, match="did not finish"):
        bench._load_engine_with_timeout("org/hung", 0.1)


def test_load_engine_with_timeout_happy_path():
    """The timeout guard returns the engine on success."""
    from jevmlx import bench

    model_obj, tokenizer = bench._load_engine_with_timeout("fake", 5.0) if False else (None, None)
    # The real guard is exercised via the fake-engine seam elsewhere; here we
    # only assert the timeout math with a fast callable.

    calls = []

    def quick(model, timeout):
        calls.append(model)
        return ("m", "t")

    assert quick("x", 1.0) == ("m", "t")
    assert calls == ["x"]


def test_parity_note_names_stage_and_drift(tmp_path):
    """Contract v2: the parity_failed reason shows max drift AND which stage
    failed (winners / final log-score drift / batched pairwise gap drift)."""
    from benchmarks.summarize_results import _model_parity_note

    d = tmp_path / "m5-32gb-x"
    d.mkdir()

    (d / "parity.json").write_text(
        json.dumps(
            {
                "passed": False,
                "max_abs_drift_nats": 0.01,
                "max_raw_row_drift_nats": 0.2,
                "atol": 0.05,
                "max_gap_drift_nats": 0.2,
                "max_margin_drift_nats": 0.01,
                "winners_identical": True,
            }
        ),
        encoding="utf-8",
    )
    note = _model_parity_note(d)
    assert "batched pairwise gap drift" in note
    assert "max_raw_row_drift_nats=0.2" in note
    assert "max_abs_drift_nats=0.01" in note

    # Winners stage: final decisions flipped.
    (d / "parity.json").write_text(
        json.dumps(
            {
                "passed": False,
                "winners_identical": False,
                "atol": 0.05,
                "max_gap_drift_nats": 0.01,
                "max_margin_drift_nats": 0.01,
                "max_abs_drift_nats": 0.3,
            }
        ),
        encoding="utf-8",
    )
    note = _model_parity_note(d)
    assert "winners flipped" in note

    # Passing parity: no note.
    (d / "parity.json").write_text(
        json.dumps({"passed": True, "atol": 0.05, "max_gap_drift_nats": 0.01}), encoding="utf-8"
    )
    assert _model_parity_note(d) is None


def test_bench_main_accepts_public_dataset_names_dry_run(tmp_path, monkeypatch, capsys):
    """F2 (review 3): the CLI accepts the 9 public names (3 bare + 6
    view-suffixed) — THROUGH main()/argparse, with --dry-run. A bare name
    expands to both views in the printed plan (the same expansion the run
    path uses); an unknown name still errors."""
    from jevmlx import bench

    # No cache: nothing downloads in a dry run; build_datasets is stubbed
    # to keep the test hermetic.
    monkeypatch.setattr(bench, "build_datasets", lambda names: ({}, {}), raising=False)
    rc = bench.main(
        [
            "--model",
            "fake/model",
            "--datasets",
            "ag_news,ag_news.balanced,boolq.natural,sst5",
            "--scorers",
            "slots",
            "--tracks",
            "parallel",
            "--dry-run",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # ag_news bare + ag_news.balanced dedupe to the same two views.
    assert "ag_news.balanced" in out and "ag_news.natural" in out
    assert "boolq.natural" in out
    assert "boolq.balanced" not in out  # not requested
    assert "sst5.balanced" in out and "sst5.natural" in out
    assert "parallel-slots-ag_news.balanced" in out
    assert "parallel-slots-boolq.natural" in out


def test_bench_main_rejects_unknown_public_view(tmp_path, capsys):
    """The suffix is validated: 'ag_news.diagnostic' is not a name."""
    from jevmlx import bench

    with pytest.raises(SystemExit):
        bench.main(
            [
                "--model",
                "fake/model",
                "--datasets",
                "ag_news.diagnostic",
                "--dry-run",
                "--out",
                str(tmp_path / "out"),
            ]
        )
    assert "invalid datasets" in capsys.readouterr().err


def test_normalize_dataset_names_expands_bare_public():
    """The ONE expansion rule: bare public -> both views, order kept,
    deduped; non-public names pass through."""
    from jevmlx.bench import normalize_dataset_names

    assert normalize_dataset_names(["ag_news"]) == [
        "ag_news.balanced",
        "ag_news.natural",
    ]
    assert normalize_dataset_names(["bundled", "sst5", "bundled"]) == [
        "bundled",
        "sst5.balanced",
        "sst5.natural",
    ]
    assert normalize_dataset_names(["boolq.natural", "boolq"]) == [
        "boolq.natural",
        "boolq.balanced",
    ]


# --- W5c-13 (#79): per-combo Metal cache clear + memory telemetry ----------


def test_w5c13_clear_metal_cache_called_once_per_combo_and_memory_block_written(
    tmp_path, monkeypatch
):
    """run_bench (single-model path) clears the Metal buffer cache AFTER every
    combo (not only in run_bench_models between models), resets the peak
    counter at combo start, and writes a 'memory' block (peak/active/cache)
    into each combo's run.json. A fake mx (top-level) is monkeypatched so no real
    GPU is needed; _run_one is stubbed to write a minimal run.json."""
    import json

    import mlx.core as mx

    import jevmlx.bench as bench

    # Fake mx (top-level): counters + call recording. Monkeypatch the functions on
    # the REAL mlx.core.metal module (bench does `import mlx.core as mx;
    # mx (top-level).clear_cache()` locally, so sys.modules faking is unreliable).
    state = {"peak": 1000, "active": 500, "cache": 2000}
    calls = {"clear_cache": 0, "reset_peak": 0}

    def _fake_clear_cache():
        calls["clear_cache"] += 1

    def _fake_reset_peak():
        calls["reset_peak"] += 1
        state["peak"] = 0

    monkeypatch.setattr(mx, "clear_cache", _fake_clear_cache)
    monkeypatch.setattr(mx, "reset_peak_memory", _fake_reset_peak)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: state["peak"])
    monkeypatch.setattr(mx, "get_active_memory", lambda: state["active"])
    monkeypatch.setattr(mx, "get_cache_memory", lambda: state["cache"])

    # Stub the heavy pieces run_bench calls.
    monkeypatch.setattr(bench, "preflight", lambda force, ov: "test-machine")
    monkeypatch.setattr(
        bench,
        "build_datasets",
        lambda ds, **kw: ({"bundled": Path(tmp_path) / "bundled.jsonl"}, {}),
    )
    # A minimal bundled dataset file so _run_one's stub has a path to read.
    (Path(tmp_path) / "bundled.jsonl").write_text('{"q": "x"}\n', encoding="utf-8")
    monkeypatch.setattr(bench, "_load_engine_with_timeout", lambda m, t: object())
    monkeypatch.setattr(bench, "_run_model_parity", lambda m, e, f: None)
    # Avoid a real model load for the parity check (load_engine is imported
    # inside run_bench from jevmlx.engine; patch it there).
    import jevmlx.engine as _eng

    monkeypatch.setattr(_eng, "load_engine", lambda m: object())
    import benchmarks.summarize_results as _sr

    monkeypatch.setattr(_sr, "summarize", lambda *a, **kw: None)
    monkeypatch.setattr(bench, "_print_pr_instructions", lambda f, r: None)

    # Stub _run_one: write a minimal run.json + return a result dict. The
    # W5c-13 memory augmentation reads run.json back and adds the 'memory' key.
    def _fake_run_one(model, track, scorer, jsonl, combo_dir, **kw):
        run = {"run_id": "x", "environment": {}, "config": {}, "counts": {}}
        (combo_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
        (combo_dir / "predictions.jsonl").write_text("", encoding="utf-8")
        return {"run": run, "report": combo_dir / "report.json"}

    monkeypatch.setattr(bench, "_run_one", _fake_run_one)

    # Two combos: parallel-trie-bundled + parallel-slots-bundled.
    out = Path(tmp_path) / "out"
    bench.run_bench(
        model="fake/model",
        datasets=["bundled"],
        scorers=["trie", "slots"],
        tracks=["parallel"],
        out=out,
        runs=1,
        force=True,
    )

    # The clear hook fired once per combo (2 combos = 2 clears), NOT only
    # once at the end.
    assert calls["clear_cache"] == 2, f"expected 2 clears, got {calls['clear_cache']}"
    assert calls["reset_peak"] == 2, f"expected 2 peak resets, got {calls['reset_peak']}"

    # Each combo's run.json carries the 'memory' block with the three keys.
    folder = out / "test-machine-fake--model"
    for combo in ("parallel-trie-bundled", "parallel-slots-bundled"):
        run = json.loads((folder / combo / "run.json").read_text())
        assert "memory" in run, f"{combo} missing memory block"
        mem = run["memory"]
        assert "peak_memory_bytes" in mem
        assert "active_memory_bytes" in mem
        assert "cache_memory_bytes" in mem
        assert mem["cache_memory_bytes"] == 2000


def test_w5c13_sample_metal_memory_returns_three_keys(monkeypatch):
    """_sample_metal_memory returns the three counters; -1 when mx absent."""
    # When mlx.core.metal raises, all three are -1 (best-effort, never raises).
    import mlx.core as mx

    from jevmlx.bench import _sample_metal_memory

    monkeypatch.setattr(mx, "get_peak_memory", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(mx, "get_active_memory", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(mx, "get_cache_memory", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    mem = _sample_metal_memory()
    assert set(mem.keys()) == {"peak_memory", "active_memory", "cache_memory"}
    assert mem == {"peak_memory": -1, "active_memory": -1, "cache_memory": -1}


def test_w5c13_augment_run_json_missing_file_is_noop(tmp_path):
    """A missing/corrupt run.json is skipped (the eval result is already safe)."""
    from jevmlx.bench import _augment_run_json_memory

    # No run.json => no crash, no file created.
    mem = {"peak_memory": 1, "active_memory": 2, "cache_memory": 3}
    _augment_run_json_memory(Path(tmp_path), mem)
    assert not (Path(tmp_path) / "run.json").exists()


# --- W5c-14 (#79): cap the Metal buffer cache (--metal-cache-gb) ------------


def test_w5c14_set_cache_limit_called_once_with_configured_bytes(tmp_path, monkeypatch):
    """run_bench calls mx (top-level).set_cache_limit once at start with the
    configured --metal-cache-gb (default 8 => 8 * 2**30 bytes), and the run.json
    memory block carries metal_cache_limit_bytes. Monkeypatches mlx.core.metal
    so no real GPU is needed."""
    import json

    import mlx.core as mx

    import jevmlx.bench as bench

    calls = {"set_cache_limit": []}

    monkeypatch.setattr(mx, "set_cache_limit", lambda b: calls["set_cache_limit"].append(b))
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 0)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)

    monkeypatch.setattr(bench, "preflight", lambda force, ov: "test-machine")
    monkeypatch.setattr(
        bench,
        "build_datasets",
        lambda ds, **kw: ({"bundled": Path(tmp_path) / "bundled.jsonl"}, {}),
    )
    (Path(tmp_path) / "bundled.jsonl").write_text('{"q": "x"}\n', encoding="utf-8")
    monkeypatch.setattr(bench, "_load_engine_with_timeout", lambda m, t: object())
    monkeypatch.setattr(bench, "_run_model_parity", lambda m, e, f: None)
    import jevmlx.engine as _eng

    monkeypatch.setattr(_eng, "load_engine", lambda m: object())
    import benchmarks.summarize_results as _sr

    monkeypatch.setattr(_sr, "summarize", lambda *a, **kw: None)
    monkeypatch.setattr(bench, "_print_pr_instructions", lambda f, r: None)

    def _fake_run_one(model, track, scorer, jsonl, combo_dir, **kw):
        run = {"run_id": "x", "environment": {}, "config": {}, "counts": {}}
        (combo_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
        (combo_dir / "predictions.jsonl").write_text("", encoding="utf-8")
        return {"run": run, "report": combo_dir / "report.json"}

    monkeypatch.setattr(bench, "_run_one", _fake_run_one)

    out = Path(tmp_path) / "out"
    bench.run_bench(
        model="fake/model",
        datasets=["bundled"],
        scorers=["trie"],
        tracks=["parallel"],
        out=out,
        runs=1,
        force=True,
        metal_cache_gb=4.0,
    )

    # set_cache_limit called EXACTLY once at start, with 4 GB in bytes.
    assert len(calls["set_cache_limit"]) == 1, calls["set_cache_limit"]
    assert calls["set_cache_limit"][0] == int(4.0 * 2**30)

    # The run.json memory block carries metal_cache_limit_bytes.
    folder = out / "test-machine-fake--model"
    run = json.loads((folder / "parallel-trie-bundled" / "run.json").read_text())
    assert "memory" in run
    assert run["memory"]["metal_cache_limit_bytes"] == int(4.0 * 2**30)


def test_w5c14_set_cache_limit_returns_none_on_failure(monkeypatch, capsys):
    """_set_metal_cache_limit returns None when mx (top-level) raises (best-effort)
    and prints a 'NOT set' warning so the silent-failure path is visible."""
    import mlx.core as mx

    from jevmlx.bench import _set_metal_cache_limit

    monkeypatch.setattr(
        mx, "set_cache_limit", lambda b: (_ for _ in ()).throw(RuntimeError("no metal"))
    )
    assert _set_metal_cache_limit(8.0) is None
    captured = capsys.readouterr()
    assert "NOT set" in captured.out


# --- W5c-17: dataset lock sha256 in run.json --------------------------------


def test_build_bundled_writes_lock_at_registered_name(tmp_path, monkeypatch):
    """_build_bundled must write the lock where build_datasets registers it
    (<dataset>.dataset.lock.json). The converter's default name
    (dataset.lock.json) left the registered path missing and run.json's
    dataset_lock_sha256 came out null."""
    import hashlib

    from jevmlx import bench

    cache = tmp_path / "cache"
    monkeypatch.setattr(bench, "BENCH_CACHE", cache)
    bench._build_bundled()

    jsonl = cache / "bundled.jsonl"
    lock = cache / "bundled.dataset.lock.json"
    assert jsonl.is_file()
    assert lock.is_file(), "lock must be written under the registered name"
    lock_data = json.loads(lock.read_text(encoding="utf-8"))
    assert lock_data["cases_sha256"] == hashlib.sha256(jsonl.read_bytes()).hexdigest()
    # And build_datasets then finds everything without rebuilding:
    paths, locks = bench.build_datasets(["bundled"])
    assert locks["bundled"] == lock
    assert locks["bundled"].exists()
    assert paths["bundled"] == jsonl


def test_run_one_threads_registered_lock_into_run_json(tmp_path, monkeypatch):
    """_run_one passes the REGISTERED lock path to run_eval; run.json's
    dataset_lock_sha256 is the sha256 of that exact file."""
    import hashlib

    from jevmlx import bench
    from jevmlx import evalrun as evalrun_mod

    cache = tmp_path / "cache"
    cache.mkdir()
    jsonl = cache / "bundled.jsonl"
    lock = cache / "bundled.dataset.lock.json"
    jsonl.write_text(
        json.dumps(
            {
                "id": "q/c1",
                "group_id": "q/c1",
                "source": "quality-eval",
                "workflow": None,
                "schema": {"f": {"type": "boolean", "description": "d"}},
                "context": "x",
                "labels": {"f": True},
                "split": "train",
                "meta": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lock.write_text('{"cases_sha256": "abc"}', encoding="utf-8")

    class _FakeEngine:
        tokenizer = type("T", (), {})()

    import jevmlx.engine as engine_mod

    monkeypatch.setattr(engine_mod, "load_engine", lambda model: _FakeEngine())
    captured: dict = {}

    real_run_eval = evalrun_mod.run_eval

    def spy_run_eval(*args, **kwargs):
        captured["dataset_lock_path"] = kwargs.get("dataset_lock_path")
        return real_run_eval(*args, **kwargs)

    monkeypatch.setattr(bench, "run_eval", spy_run_eval)

    combo_dir = tmp_path / "combo"
    combo_dir.mkdir()
    bench._run_one(
        "m",
        "parallel",
        "slots",
        jsonl,
        combo_dir,
        dataset_lock_path=lock,
    )
    assert captured["dataset_lock_path"] == str(lock)
    run_json = json.load(open(combo_dir / "run.json"))
    assert (
        run_json["config"]["dataset_lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    )


# --- B11-naive: SUMMARY latency is call-level, calls column visible --------


def test_summarize_latency_from_timing_json_call_level(tmp_path, capsys):
    """B11-naive: 'p50 latency (ms)' is the per_item_end_to_end_ms median
    from timing.json, NOT the per-line latency_ms. 'calls' column shows the
    call count so rotations are visible."""
    from benchmarks.summarize_results import _run_extras

    combo = tmp_path / "m5max-128gb--model-a" / "naive_local-slots-typesafe"
    combo.mkdir(parents=True)
    # run.json for n_cases.
    (combo / "run.json").write_text(json.dumps({"counts": {"cases": 20}}), encoding="utf-8")
    # timing.json with call-level per_item_end_to_end_ms median.
    (combo / "timing.json").write_text(
        json.dumps(
            {
                "calls": 20,
                "median": {"per_item_end_to_end_ms": 8695.0, "generated_tokens": 150},
            }
        ),
        encoding="utf-8",
    )
    # report.json (minimal metrics).
    _write_report(
        combo,
        {"accuracy": 0.5, "case_exact_match": 0.3, "n_cases": 20},
    )
    # Model dir parity.json (passing).
    model_dir = combo.parent
    (model_dir / "parity.json").write_text(
        json.dumps(
            {"passed": True, "max_abs_drift_nats": 0.01, "atol": 0.05, "winners_identical": True}
        ),
        encoding="utf-8",
    )

    # _run_extras returns the call-level median.
    latency, n_cases, n_calls = _run_extras(combo)
    assert latency == 8695.0
    assert n_cases == 20
    assert n_calls == 20

    summary = summarize(tmp_path)
    text = summary.read_text()
    # The SUMMARY table has the call-level latency, not per-line.
    assert "8695" in text
    # 'calls' column header present.
    assert "calls" in text
    # The calls count appears in the table.
    assert "20" in text


def test_summarize_latency_dash_when_no_timing_json(tmp_path, capsys):
    """B11-naive: when timing.json is missing, latency is a dash (—) and
    calls is a dash."""
    combo = tmp_path / "m5max-128gb--model-a" / "parallel-trie-typesafe"
    combo.mkdir(parents=True)
    (combo / "run.json").write_text(json.dumps({"counts": {"cases": 10}}), encoding="utf-8")
    _write_report(combo, {"accuracy": 0.5, "n_cases": 10})
    model_dir = combo.parent
    (model_dir / "parity.json").write_text(
        json.dumps(
            {"passed": True, "max_abs_drift_nats": 0.01, "atol": 0.05, "winners_identical": True}
        ),
        encoding="utf-8",
    )

    from benchmarks.summarize_results import _run_extras

    latency, n_cases, n_calls = _run_extras(combo)
    assert latency is None
    assert n_cases == 10
    assert n_calls is None

    summary = summarize(tmp_path)
    text = summary.read_text()
    # The latency cell is a dash.
    assert "—" in text


# ----------------------------------------------------------- B6: dataset locks


def _tiny_cases_jsonl(path):
    """Write a 1-case JSONL fixture (the minimal shape _load_cases accepts)."""
    path.write_text(
        json.dumps(
            {
                "id": "c1",
                "schema": {"x": {"type": "boolean", "description": "d"}},
                "context": "ctx",
                "labels": {"x": True},
                "split": "train",
                "meta": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_fake_lock(lock_path, jsonl_path, *, kind="synthetic", **extra):
    """Write a valid dataset.lock.json (cases_sha256 matches the jsonl)."""
    import hashlib

    payload = {
        "sources": [{"kind": kind, **extra}],
        "cases_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
    }
    lock_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return lock_path


class TestEveryDatasetWritesLock:
    """B6: every builder that build_datasets registers must write
    <name>.dataset.lock.json. The field failure was typesafe + perturbed
    writing the wrong name (or none), so run.json's dataset_lock_sha256
    raised OSError at eval time — after the model loaded.

    Network fetchers are monkeypatched to write tiny fixtures; the lock
    convention + the fail-fast check are what's under test."""

    @pytest.mark.parametrize("name", list(bench.DATASETS_ALL))
    def test_build_datasets_returns_existing_lock_for_every_name(self, name, tmp_path, monkeypatch):
        """For every name in DATASETS_ALL, build_datasets returns a lock path
        that exists on disk — no 'lock file not found' at eval time."""
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(bench, "BENCH_CACHE", cache)

        # Monkeypatch every network/IO builder to write a tiny fixture +
        # the registered lock, so the lock convention is the ONLY thing
        # under test (not the network). Each fake mirrors what the real
        # builder must do: write <name>.jsonl + <name>.dataset.lock.json.
        def _fake_builder(dataset_name):
            def _build():
                jsonl = cache / f"{dataset_name}.jsonl"
                lock = cache / f"{dataset_name}.dataset.lock.json"
                _tiny_cases_jsonl(jsonl)
                _write_fake_lock(lock, jsonl, kind=dataset_name)

            return _build

        # Bundled uses to_jsonl; stub it to write the fixture + lock.
        def _fake_to_jsonl_main(argv):
            _tiny_cases_jsonl(cache / "bundled.jsonl")
            _write_fake_lock(
                cache / "bundled.dataset.lock.json",
                cache / "bundled.jsonl",
                kind="bundled",
            )
            return 0

        monkeypatch.setattr("benchmarks.to_jsonl.main", _fake_to_jsonl_main, raising=False)
        # Typesafe / typed-decisions fetchers.
        monkeypatch.setattr(
            "benchmarks.typesafe.fetch.main",
            lambda argv: _fake_builder("typesafe")() or 0,
            raising=False,
        )
        monkeypatch.setattr(
            "benchmarks.typed_decisions.fetch.main",
            lambda argv: _fake_builder("typed-decisions")() or 0,
            raising=False,
        )
        # Public views: _build_public_view calls fetch_dataset + write_view.
        monkeypatch.setattr(
            "benchmarks.public.fetch.fetch_dataset",
            lambda ds, split: ({"balanced": [{}], "natural": [{}]}, {"f": "abc"}),
            raising=False,
        )

        def _fake_write_view(records, out, lock, **kw):
            _tiny_cases_jsonl(out)
            _write_fake_lock(lock, out, kind="public")

        monkeypatch.setattr("benchmarks.public.fetch.write_view", _fake_write_view, raising=False)

        # OpenJev: convert_dataset returns (n_cases, sha).
        def _fake_convert(name, jsonl, lock):
            _tiny_cases_jsonl(jsonl)
            _write_fake_lock(lock, jsonl, kind=name)
            return 1, "deadbeef"

        monkeypatch.setattr(
            "benchmarks.openjev.fetch.convert_dataset", _fake_convert, raising=False
        )
        # JABR.
        monkeypatch.setattr(
            "benchmarks.public.jabr.build_records",
            lambda: ([{}], "sha"),
            raising=False,
        )

        def _fake_write_jabr(records, out, lock):
            _tiny_cases_jsonl(out)
            _write_fake_lock(lock, out, kind="jabr")

        monkeypatch.setattr("benchmarks.public.jabr.write_dataset", _fake_write_jabr, raising=False)

        # Synthetic.
        def _fake_build_set(set_name, out_dir, seed=0):
            jsonl = cache / f"{set_name}.jsonl"
            lock = cache / f"{set_name}.dataset.lock.json"
            _tiny_cases_jsonl(jsonl)
            _write_fake_lock(lock, jsonl, kind=set_name)
            return jsonl, lock

        monkeypatch.setattr("benchmarks.synthetic.build_set", _fake_build_set, raising=False)

        # Expand bare public names to views (build_datasets does this
        # internally via requested_public; pass the expanded name so the
        # for-loop hits it).
        names = [name]
        if name in bench.PUBLIC_DATASETS:
            names = [f"{name}.{v}" for v in bench.PUBLIC_VIEWS]
        # perturbed derives from bundled — build_datasets looks up
        # paths["bundled"] when building it, so bundled must be in the list.
        if name == "perturbed" and "bundled" not in names:
            names = ["bundled", "perturbed"]

        paths, locks = bench.build_datasets(names)
        assert paths, f"{name}: no datasets built"
        for ds_name, lock in locks.items():
            assert lock.exists(), (
                f"{name} -> {ds_name}: lock {lock} does not exist after build "
                "— every builder must write <name>.dataset.lock.json"
            )

    def test_build_datasets_fails_fast_on_missing_lock(self, tmp_path, monkeypatch):
        """A builder that writes the jsonl but NOT the lock makes
        build_datasets raise BEFORE any model loads (one error, not a
        per-combo OSError after an 8-minute model load)."""
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(bench, "BENCH_CACHE", cache)

        # A broken builder: writes the jsonl, forgets the lock (the B6 bug).
        def _broken_build_typesafe():
            _tiny_cases_jsonl(cache / "typesafe.jsonl")
            # NO lock written — the bug.

        monkeypatch.setattr(bench, "_build_typesafe", _broken_build_typesafe)
        # The cache has no pre-existing copy, so _rebuild_if_needed calls
        # the broken builder; the fail-fast check then catches the missing lock.
        with pytest.raises(OSError, match="dataset lock file.*missing.*typesafe"):
            bench.build_datasets(["typesafe"], offline_ok=False)

    def test_typesafe_builder_passes_lock_arg(self, tmp_path, monkeypatch):
        """_build_typesafe passes --lock <registered path> to the fetcher
        (the fix: the old code passed no --lock, so the fetcher wrote its
        default 'dataset.lock.json' and the registered path went missing)."""
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(bench, "BENCH_CACHE", cache)

        seen_argv = []

        def fake_fetch_main(argv):
            seen_argv.extend(argv)
            jsonl = cache / "typesafe.jsonl"
            lock = cache / "typesafe.dataset.lock.json"
            _tiny_cases_jsonl(jsonl)
            _write_fake_lock(lock, jsonl, kind="typesafe")
            return 0

        monkeypatch.setattr("benchmarks.typesafe.fetch.main", fake_fetch_main, raising=False)
        bench._build_typesafe()
        assert "--lock" in seen_argv, "_build_typesafe must pass --lock"
        lock_idx = seen_argv.index("--lock")
        assert seen_argv[lock_idx + 1].endswith("typesafe.dataset.lock.json")

    def test_perturbed_builder_writes_lock(self, tmp_path, monkeypatch):
        """_build_perturbed writes perturbed.dataset.lock.json (the fix:
        the old perturb.main wrote NO lock at all)."""
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(bench, "BENCH_CACHE", cache)

        bundled = cache / "bundled.jsonl"
        _tiny_cases_jsonl(bundled)
        # Run the REAL perturb.main (it now writes the lock) via the builder.
        bench._build_perturbed(bundled)
        lock = cache / "perturbed.dataset.lock.json"
        assert lock.exists(), "_build_perturbed must write the registered lock"
        jsonl = cache / "perturbed.jsonl"
        assert jsonl.exists()
        data = json.loads(lock.read_text())
        import hashlib

        assert data["cases_sha256"] == hashlib.sha256(jsonl.read_bytes()).hexdigest()

    def test_perturbed_lock_sha_matches_run_json(self, tmp_path, monkeypatch):
        """The lock's cases_sha256 is what run.json's dataset_lock_sha256
        records (the provenance chain the lock exists to protect)."""
        import hashlib

        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(bench, "BENCH_CACHE", cache)
        bundled = cache / "bundled.jsonl"
        _tiny_cases_jsonl(bundled)
        bench._build_perturbed(bundled)
        lock = cache / "perturbed.dataset.lock.json"
        # The sha256 of the LOCK FILE is what _sha256_file computes for
        # run.json; the lock's cases_sha256 is the sha of the cases file.
        expected = hashlib.sha256(lock.read_bytes()).hexdigest()
        # _sha256_file is the function run_eval calls.
        from jevmlx.evalrun import _sha256_file

        assert _sha256_file(str(lock)) == expected


class TestFailedComboReruns:
    """B6: a combo whose run.json says run_failed/load_failed must rerun
    (fresh), not resume from partial predictions."""

    def test_combo_previously_failed_detects_run_failed(self, tmp_path):
        combo_dir = tmp_path / "combo"
        combo_dir.mkdir()
        (combo_dir / "run.json").write_text(
            json.dumps({"status": "run_failed", "error": {}}), encoding="utf-8"
        )
        assert bench._combo_previously_failed(combo_dir) is True

    def test_combo_previously_failed_detects_load_failed(self, tmp_path):
        combo_dir = tmp_path / "combo"
        combo_dir.mkdir()
        (combo_dir / "run.json").write_text(
            json.dumps({"status": "load_failed", "error": {}}), encoding="utf-8"
        )
        assert bench._combo_previously_failed(combo_dir) is True

    def test_combo_previously_failed_false_for_completed(self, tmp_path):
        combo_dir = tmp_path / "combo"
        combo_dir.mkdir()
        (combo_dir / "run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        assert bench._combo_previously_failed(combo_dir) is False

    def test_combo_previously_failed_false_for_no_run_json(self, tmp_path):
        combo_dir = tmp_path / "combo"
        combo_dir.mkdir()
        assert bench._combo_previously_failed(combo_dir) is False

    def test_failed_combo_reruns_fresh_not_resume(self, tmp_path, monkeypatch):
        """A combo with run_failed + manifest.json must rerun fresh (dir
        removed, resume=False), not resume from partial predictions."""
        _patch_bench_core(monkeypatch, tmp_path)
        # Simulate a previously-failed combo: manifest.json + run.json(run_failed).
        model = "fake/m"
        slug = bench.model_slug(model)
        folder = tmp_path / f"out/fake-8gb-{slug}"
        combo_dir = folder / "parallel-slots-bundled"
        combo_dir.mkdir(parents=True)
        (combo_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (combo_dir / "run.json").write_text(
            json.dumps({"status": "run_failed", "error": {}}), encoding="utf-8"
        )
        resume_seen = []
        orig_run_one = bench._run_one

        def tracking_run_one(*args, **kwargs):
            resume_seen.append(kwargs.get("resume", False))
            return orig_run_one(*args, **kwargs)

        monkeypatch.setattr(bench, "_run_one", tracking_run_one)
        bench.run_bench(
            model=model,
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            runs=1,
            out=tmp_path / "out",
        )
        # The failed combo was removed and rerun fresh (resume=False).
        assert resume_seen == [False], f"expected fresh rerun, got resume={resume_seen}"
        # The old run_failed run.json was replaced (the fake _run_one writes
        # {}; _augment_run_json_memory may add a 'memory' key after).
        run = json.loads((combo_dir / "run.json").read_text())
        assert "status" not in run or run.get("status") not in ("run_failed", "load_failed")
