"""Tests for benchmarks/leaderboard.py: exact table text, stale README check,
missing published file -> local rows only, and L1's real published_agreement()
output shape (by_workflow values are dicts {agreed, total, agreement})."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.leaderboard import build_table, check_readme, main, write_readme_block
from benchmarks.typesafe.published import published_agreement

_OFFICIAL = {
    "_note": "CITATION, not measurement.",
    "source_url": "https://evals.typesafe.ai/",
    "retrieved": "2026-09-17",
    "models": [
        {
            "name": "Jev",
            "accuracy": 0.678,
            "by_workflow": {
                "customer_service": 0.760,
                "agent_trace_observability": 0.716,
                "security_incidents": 0.617,
                "invoice_processing": 0.618,
            },
            "time_per_case_s": 0.4,
            "cost_per_case_usd": 0.0004,
        },
        {
            "name": "GPT-5.6 Terra",
            "accuracy": 0.679,
            "by_workflow": {},
            "time_per_case_s": 10.1,
            "cost_per_case_usd": 0.0304,
        },
    ],
}


def _write_official(tmp_path: Path) -> Path:
    p = tmp_path / "official.json"
    p.write_text(json.dumps(_OFFICIAL, indent=2), encoding="utf-8")
    return p


def _published_records():
    """Reuse L1's hand-built fixture shape (3 records, 2 models: opus, sol).

    Mirrors tests/test_published.py's `records` fixture so the published
    block is driven by the REAL published_agreement() function, not a
    hand-rolled JSON blob.
    """
    bool_schema = {"type": "boolean", "description": "d"}
    enum_schema = {"type": "enum", "description": "d", "choices": ["refund", "dispute"]}
    score_schema = {"type": "enum", "description": "d", "choices": ["0", "1", "2", "3"]}

    def _record(rid, workflow, schema, labels, models, ambiguous=()):
        rec = {
            "id": rid,
            "workflow": workflow,
            "schema": schema,
            "labels": labels,
            "meta": {"models": models},
        }
        if ambiguous:
            rec["meta"]["ambiguous"] = list(ambiguous)
        return rec

    return [
        _record(
            "typesafe/customer_service/case-1",
            "customer_service",
            {"bool_field": bool_schema, "enum_field": enum_schema},
            {"bool_field": False, "enum_field": "refund"},
            {
                "bool_field": {"opus": 0.9, "sol": 0.2},
                "enum_field": {"opus": "refund", "sol": "refund"},
            },
        ),
        _record(
            "typesafe/security_incidents/case-2",
            "security_incidents",
            {"score_field": score_schema, "ambiguous_field": enum_schema},
            {"score_field": "2", "ambiguous_field": "refund"},
            {
                "score_field": {"opus": 1.6, "sol": 0.4},
                "ambiguous_field": {"opus": "refund", "sol": "refund"},
            },
            ambiguous=["ambiguous_field"],
        ),
        _record(
            "typesafe/customer_service/case-3",
            "customer_service",
            {"missing_field": enum_schema},
            {"missing_field": "refund"},
            {"missing_field": {"opus": "refund"}},
        ),
    ]


def _write_published(tmp_path: Path) -> Path:
    """Write L1's real published_agreement() output to a JSON file."""
    result = published_agreement(_published_records())
    p = tmp_path / "published_agreement.json"
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return p


def _write_local_result(tmp_path: Path) -> Path:
    """A minimal valid results folder for a typesafe parallel run."""
    root = tmp_path / "results"
    machine_dir = root / "m1-8gb-fake"
    combo = machine_dir / "parallel-trie-typesafe"
    combo.mkdir(parents=True)
    run = {
        "run_id": "r1",
        "environment": {"chip": "fake"},
        "config": {
            "model": "fake-1b",
            "track": "parallel",
            "dataset_path": "/cache/jevmlx/bench/typesafe.jsonl",
        },
        "counts": {"cases": 4, "fields": 4, "prediction_lines": 4},
    }
    (combo / "run.json").write_text(json.dumps(run), encoding="utf-8")
    # W4-A: a model needs a passing parity.json to enter the leaderboard.
    (machine_dir / "parity.json").write_text(
        json.dumps(
            {
                "model": "fake-1b",
                "test": "w1a",
                "passed": True,
                "max_drift_nats": 0.02,
                "atol": 0.05,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "environment": {"chip": "fake"},
        "metrics": {
            "agreement": {
                "agreement_common_subset": 0.75,
                "by_workflow": {"customer_service": 0.8, "invoice_processing": 0.7},
                "n_fields": 4,
                "n_cases": 4,
            }
        },
    }
    (combo / "report.json").write_text(json.dumps(report), encoding="utf-8")
    records = []
    for i in range(4):
        records.append(
            json.dumps(
                {
                    "case_id": f"c{i}",
                    "field": "f",
                    "latency_ms": 500.0 + i * 100,
                    "valid": True,
                    "correct": True,
                    "label": "A",
                    "prediction": "A",
                }
            )
        )
    (combo / "predictions.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    return root


def _write_local_result_v2(root: Path) -> Path:
    """Same fixture, but predictions carry the results-contract-v2 per-item
    end-to-end timing (the only shape the leaderboard accepts now)."""
    combo = next(p for p in (root / "m1-8gb-fake").iterdir() if p.is_dir())
    # parity-gates: time/case comes from timing.json median
    # per_item_end_to_end_ms (same source as summarize_results since #99),
    # not from prediction lines.
    (combo / "timing.json").write_text(
        json.dumps({"median": {"per_item_end_to_end_ms": 850.0}}), encoding="utf-8"
    )
    return root


def _add_typed_decisions_combo(root: Path) -> None:
    """Clone the typesafe combo into a typed-decisions one (accuracy 0.6)."""
    machine_dir = root / "m1-8gb-fake"
    src = machine_dir / "parallel-trie-typesafe"
    dst = machine_dir / "parallel-trie-typed-decisions"
    dst.mkdir()
    run = json.loads((src / "run.json").read_text())
    run["config"]["dataset_path"] = "/cache/jevmlx/bench/typed-decisions.jsonl"
    run["counts"]["cases"] = 400  # match the report's n_cases
    (dst / "run.json").write_text(json.dumps(run), encoding="utf-8")
    # parity-gates: timing.json at the combo level (time/case source).
    (dst / "timing.json").write_text(
        json.dumps({"median": {"per_item_end_to_end_ms": 850.0}}), encoding="utf-8"
    )
    report = json.loads((src / "report.json").read_text())
    report["metrics"]["agreement"]["agreement_common_subset"] = 0.6
    report["metrics"]["agreement"]["n_cases"] = 400
    (dst / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (dst / "predictions.jsonl").write_bytes((src / "predictions.jsonl").read_bytes())
    lock = {
        "sources": [
            {
                "repo_id": "LocalLLaMA/typed-decisions",
                "revision": "0af3f0e9dc6d28c2f8f1c9d1ba2e4a55f0e6c9d3",
            }
        ],
        "parser_version": "1",
        "counts": {},
        "cases_sha256": "x",
    }
    (machine_dir / "typed-decisions.dataset.lock.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )


def test_typed_decisions_rows_render_in_their_own_group(tmp_path):
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    results = _write_local_result(tmp_path)
    _write_local_result_v2(results)
    _add_typed_decisions_combo(results)
    table = build_table(results, published, official)
    lines = table.splitlines()
    group_c = next(i for i, ln in enumerate(lines) if "**jevmlx, local (measured)**" in ln)
    group_d = next(i for i, ln in enumerate(lines) if "LocalLLaMA/typed-decisions test split" in ln)
    assert group_c < group_d
    # Each group holds exactly its own dataset's row: typesafe 75.0% / 4
    # cases under C, typed-decisions 60.0% / 400 cases under D.
    assert lines[group_c + 1].startswith("| fake-1b |") and "75.0%" in lines[group_c + 1]
    assert lines[group_d + 1].startswith("| fake-1b |") and "60.0%" in lines[group_d + 1]
    assert lines[group_d + 1].rstrip().endswith("| 400 |")
    assert group_d == group_c + 2  # exactly one typesafe row between the headers
    # Caption derives cases from the row and names the pinned revision —
    # no hardcoded 400/test split.
    assert "400 cases" in table
    assert "revision 0af3f0e9dc6d28c2f8f1c9d1ba2e4a55f0e6c9d3" in table
    assert "official `test` split" not in table
    assert "No local results yet" not in table


def test_official_block_exact_text(tmp_path):
    """Official + published blocks render the exact expected rows."""
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    table = build_table(None, published, official)
    # Official header + group separator line + two cited rows.
    assert "TypeSafe official (cited, retrieved 2026-09-17)" in table
    assert "| Jev | official (cited) | — | — | 67.8% | — | 76.0% |" in table
    assert "71.6% | 61.7% | 61.8% | 0.4s | $0.0004 | — |" in table
    assert "| GPT-5.6 Terra | official (cited) | — | — | 67.9% | — |" in table
    # Published group — driven by the REAL published_agreement() output.
    assert "Published models on the public examples (computed)" in table
    assert "| opus | computed from published answers |" in table
    assert "| sol | computed from published answers |" in table
    # by_workflow values are dicts {agreed, total, agreement}; _fmt_pct
    # extracts the agreement float. The fixture's workflows are
    # customer_service and security_incidents (L1's real WORKFLOWS names),
    # so the workflow columns populate. Overall accuracy = agreed/total =
    # 66.7% for both models.
    assert "66.7%" in table
    # opus customer_service: 1/2 = 50.0%; security_incidents: 1/1 = 100.0%.
    assert "50.0%" in table
    assert "100.0%" in table
    # No local rows yet -> the contribute line.
    assert "No local results yet" in table


def test_published_uses_real_l1_shape(tmp_path):
    """The published block reads L1's actual output: model key 'name',
    accuracy 'agreement', by_workflow dicts {agreed, total, agreement},
    cases = total."""
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    table = build_table(None, published, official)
    # opus: 2 agreed out of 3 total (bool disagree, enum agree, score agree).
    assert "| opus | computed from published answers | — | — | 66.7% |" in table
    # Cases column = subset.n_cases (2 cases in the subset).
    # Find the opus row and check it ends with | 2 |.
    opus_line = next(line for line in table.splitlines() if line.startswith("| opus |"))
    assert opus_line.rstrip().endswith("| 2 |")


def test_local_rows_render_when_results_present(tmp_path):
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    results = _write_local_result(tmp_path)
    _write_local_result_v2(results)
    table = build_table(results, published, official)
    assert "jevmlx, local (measured)" in table
    assert "fake-1b" in table
    assert "local" in table
    # Accuracy 0.75 -> 75.0%, time per case = per-item end-to-end
    # median(700,800,900,1000)ms = 0.85s -> 0.8s at 1dp (f-string rounding).
    # The per-field latency_ms median (0.65s) must NOT be used — contract v2
    # has no fallback.
    assert "75.0%" in table
    # Per-workflow agreement must land in its column (guards the by_workflow
    # key-name mapping: report uses full names customer_service /
    # invoice_processing, not short aliases).
    local_line = next(line for line in table.splitlines() if line.startswith("| fake-1b |"))
    assert "80.0%" in local_line  # customer_service 0.8
    assert "70.0%" in local_line  # invoice_processing 0.7
    assert "0.8s" in table
    assert "$0 (local)" in table
    assert "| 4 |" in table  # cases
    # The "No local results" line must NOT appear when local rows exist.
    assert "No local results yet" not in table


def test_missing_published_file_local_rows_only(tmp_path):
    """No published file -> official + local only, no published group."""
    official = _write_official(tmp_path)
    results = _write_local_result(tmp_path)
    _write_local_result_v2(results)
    table = build_table(results, None, official)
    assert "Published models on the public examples" not in table
    assert "official (cited)" in table
    assert "jevmlx, local (measured)" in table


def test_malformed_published_file_raises(tmp_path):
    """A published file missing 'subset' or 'models' raises ValueError, no fallback."""
    import pytest

    official = _write_official(tmp_path)
    bad = tmp_path / "published_agreement.json"
    bad.write_text(json.dumps([{"model": "x"}]), encoding="utf-8")  # bare list
    with pytest.raises(ValueError, match="missing keys"):
        build_table(None, bad, official)


def test_published_missing_models_key_raises(tmp_path):
    """A published file with 'subset' but no 'models' raises ValueError."""
    import pytest

    official = _write_official(tmp_path)
    bad = tmp_path / "published_agreement.json"
    bad.write_text(json.dumps({"subset": {"n_fields": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        build_table(None, bad, official)


def test_stale_readme_block_fails_check(tmp_path):
    """A README whose marker block differs from the built table fails --check-readme."""
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n\n<!-- leaderboard:start -->\n| old |\n<!-- leaderboard:end -->\n",
        encoding="utf-8",
    )
    table = build_table(None, published, official)
    assert not check_readme(readme, table)


def test_fresh_readme_block_passes_check(tmp_path):
    """After write_readme_block, check_readme passes."""
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n\n<!-- leaderboard:start -->\n<!-- leaderboard:end -->\n", encoding="utf-8"
    )
    table = build_table(None, published, official)
    write_readme_block(readme, table)
    assert check_readme(readme, table)


def test_main_check_readme_exits_1_when_stale(tmp_path, capsys):
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n\n<!-- leaderboard:start -->\n| old |\n<!-- leaderboard:end -->\n",
        encoding="utf-8",
    )
    rc = main(
        [
            "--check-readme",
            "--readme",
            str(readme),
            "--official",
            str(official),
            "--published",
            str(published),
        ]
    )
    assert rc == 1
    assert "STALE" in capsys.readouterr().err


def test_main_check_readme_exits_0_when_fresh(tmp_path, capsys):
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("intro\n", encoding="utf-8")
    empty_results = tmp_path / "noresults"
    table = build_table(empty_results, published, official)
    write_readme_block(readme, table)
    rc = main(
        [
            "--check-readme",
            "--readme",
            str(readme),
            "--official",
            str(official),
            "--published",
            str(published),
            "--results",
            str(empty_results),
        ]
    )
    assert rc == 0
    assert "up to date" in capsys.readouterr().out.lower()


def test_local_rows_accept_single_path_folder(tmp_path):
    """W5c-3: the single path (run_parallel_generation) also reports
    per_item_end_to_end_ms — a parallel folder whose timing.json carries
    it is a valid folder: the leaderboard must accept it."""
    official = _write_official(tmp_path)
    results = _write_local_result(tmp_path)
    combo = next(p for p in (results / "m1-8gb-fake").iterdir() if p.is_dir())
    # parity-gates: time/case from timing.json median per_item_end_to_end_ms
    (combo / "timing.json").write_text(
        json.dumps({"median": {"per_item_end_to_end_ms": 650.0}}), encoding="utf-8"
    )
    table = build_table(results, None, official)
    assert "fake-1b" in table
    assert "0.7s" in table  # per-item e2e median, not the latency_ms median


def test_local_rows_fail_without_per_item_timing(tmp_path):
    """Contract v2: a parallel combo whose predictions lack
    per_item_end_to_end_ms FAILS loudly — no latency_ms fallback (there are
    no pre-v2 folders on main)."""
    import pytest as _pytest

    official = _write_official(tmp_path)
    results = _write_local_result(tmp_path)  # v1 lines: latency_ms only
    with _pytest.raises(ValueError, match="per_item_end_to_end_ms"):
        build_table(results, None, official)


def _write_local_result_with_parity(tmp_path: Path, *, status: str, passed: bool) -> Path:
    """A results folder whose parity.json has a given status + passed flag.

    Reuses _write_local_result_v2's shape (valid per-item timing) and
    overwrites parity.json with the requested status."""
    root = _write_local_result(tmp_path)
    _write_local_result_v2(root)  # upgrade predictions to v2 timing
    machine_dir = root / "m1-8gb-fake"
    parity = {
        "model": "fake-1b",
        "test": "w1a",
        "passed": passed,
        "status": status,
        "max_abs_drift_nats": 0.078 if status == "DRIFT" else 0.02,
        "max_gap_drift_nats": 0.078 if status == "DRIFT" else 0.01,
        "max_margin_drift_nats": 0.05 if status == "DRIFT" else 0.01,
        "max_raw_row_drift_nats": 0.03,
        "atol": 0.05,
        "winners_identical": True,
        "drift_envelope": {"band": 0.14} if status == "DRIFT" else {},
    }
    (machine_dir / "parity.json").write_text(json.dumps(parity), encoding="utf-8")
    return root


def test_leaderboard_includes_drift_model(tmp_path):
    """A model with parity status=DRIFT (publishable batch-shape noise)
    appears in the leaderboard, and the Parity column shows 'DRIFT' + max drift."""
    official = _write_official(tmp_path)
    results = _write_local_result_with_parity(tmp_path, status="DRIFT", passed=False)
    table = build_table(results, None, official)
    assert "fake-1b" in table
    assert "DRIFT" in table
    # The Parity column shows the word (not just the status code).
    assert "DRIFT (0.078)" in table


def test_leaderboard_includes_pass_model(tmp_path):
    """A model with parity status=PASS appears in the leaderboard."""
    official = _write_official(tmp_path)
    results = _write_local_result_with_parity(tmp_path, status="PASS", passed=True)
    table = build_table(results, None, official)
    assert "fake-1b" in table
    assert "PASS" in table


def test_leaderboard_excludes_fail_model(tmp_path):
    """A model with parity status=FAIL (a winner changed, or drift beyond
    the band) is excluded from the leaderboard — no row."""
    official = _write_official(tmp_path)
    results = _write_local_result_with_parity(tmp_path, status="FAIL", passed=False)
    table = build_table(results, None, official)
    assert "fake-1b" not in table


def test_leaderboard_excludes_missing_parity(tmp_path):
    """A model with no parity.json is excluded."""
    official = _write_official(tmp_path)
    root = _write_local_result(tmp_path)
    _write_local_result_v2(root)
    (root / "m1-8gb-fake" / "parity.json").unlink()
    table = build_table(root, None, official)
    assert "fake-1b" not in table


def _write_naive_combo(root: Path) -> Path:
    """Add a naive_local-slots-typesafe combo (the generate+parse baseline).

    Mirrors the real 7B results folder: track=naive_local, timing.json with
    call-level median, error lines (invalid enum values).
    """
    machine_dir = root / "m1-8gb-fake"
    combo = machine_dir / "naive_local-slots-typesafe"
    combo.mkdir(parents=True, exist_ok=True)
    run = {
        "run_id": "r-naive",
        "environment": {"chip": "fake"},
        "config": {
            "model": "fake-1b",
            "track": "naive_local",
            "dataset_path": "/cache/jevmlx/bench/typesafe.jsonl",
        },
        "counts": {"cases": 5, "fields": 5, "prediction_lines": 5},
    }
    (combo / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (combo / "timing.json").write_text(
        json.dumps({"median": {"per_item_end_to_end_ms": 1480.0}}), encoding="utf-8"
    )
    report = {
        "environment": {"chip": "fake"},
        "metrics": {
            "agreement": {
                "agreement_common_subset": 0.6,
                "by_workflow": {"customer_service": 0.7},
                "n_fields": 5,
                "n_cases": 5,
            }
        },
    }
    (combo / "report.json").write_text(json.dumps(report), encoding="utf-8")
    records = []
    for i in range(5):
        if i < 2:
            # 2 error lines (invalid enum value — the naive parser failed).
            records.append(
                json.dumps(
                    {
                        "case_id": f"c{i}",
                        "field": "f",
                        "valid": False,
                        "correct": None,
                        "label": "A",
                        "prediction": None,
                        "error": f"invalid value for f: {i}",
                        "latency_ms": 1500.0,
                    }
                )
            )
        else:
            records.append(
                json.dumps(
                    {
                        "case_id": f"c{i}",
                        "field": "f",
                        "valid": True,
                        "correct": True,
                        "label": "A",
                        "prediction": "A",
                        "latency_ms": 1400.0,
                    }
                )
            )
    (combo / "predictions.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    return root


def test_leaderboard_includes_naive_baseline_row(tmp_path):
    """The naive_local track (generate JSON + parse) is the baseline the
    project argues against. It must appear as a row: Scorer 'naive
    (generate+parse)', Parity '—', Time per case from timing.json, Cases
    with '(M error)' note.
    """
    official = _write_official(tmp_path)
    root = _write_local_result(tmp_path)
    _write_local_result_v2(root)
    _write_naive_combo(root)
    table = build_table(root, None, official)
    # The naive row appears.
    assert "naive (generate+parse)" in table
    # Parity is '—' for the naive row (no batch/chunked test).
    # Find the naive row line and check the Parity cell.
    naive_line = next(line for line in table.splitlines() if "naive (generate+parse)" in line)
    cells = [c.strip() for c in naive_line.split("|")]
    # cells: ['', model, source, scorer, machine, accuracy, parity, ...]
    parity_cell = cells[6]
    assert parity_cell == "—", f"naive row parity should be '—', got {parity_cell!r}"
    # Time per case from timing.json (1.48s -> rounds to 1.5s).
    assert "1.5s" in naive_line
    # Cases with error note.
    assert "5 (2 error)" in naive_line


def test_leaderboard_case_count_derived_from_results(tmp_path):
    """The footnote 'N public example cases' is derived from the results'
    counts.cases, never a literal 20."""
    official = _write_official(tmp_path)
    root = _write_local_result(tmp_path)
    _write_local_result_v2(root)
    _write_naive_combo(root)
    table = build_table(root, None, official)
    # The footnote derives the count from the local rows (4 cases for the
    # parallel combo, 5 for naive — min=4, max=5; when there's a single
    # count it shows that number).
    assert "20 public example cases" not in table
    assert "public example cases" in table
