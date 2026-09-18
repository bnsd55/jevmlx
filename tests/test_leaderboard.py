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


def test_official_block_exact_text(tmp_path):
    """Official + published blocks render the exact expected rows."""
    official = _write_official(tmp_path)
    published = _write_published(tmp_path)
    table = build_table(None, published, official)
    # Official header + group separator line + two cited rows.
    assert "TypeSafe official (cited, retrieved 2026-09-17)" in table
    assert "| Jev | official (cited) | — | — | 67.8% | 76.0% |" in table
    assert "71.6% | 61.7% | 61.8% | 0.4s | $0.0004 | — |" in table
    assert "| GPT-5.6 Terra | official (cited) | — | — | 67.9% |" in table
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
    table = build_table(results, published, official)
    assert "jevmlx, local (measured)" in table
    assert "fake-1b" in table
    assert "local" in table
    # Accuracy 0.75 -> 75.0%, time per case = median(500,600,700,800)ms = 0.65s -> 0.7s (1dp).
    assert "75.0%" in table
    # Per-workflow agreement must land in its column (guards the by_workflow
    # key-name mapping: report uses full names customer_service /
    # invoice_processing, not short aliases).
    local_line = next(line for line in table.splitlines() if line.startswith("| fake-1b |"))
    assert "80.0%" in local_line  # customer_service 0.8
    assert "70.0%" in local_line  # invoice_processing 0.7
    assert "0.7s" in table
    assert "$0 (local)" in table
    assert "| 4 |" in table  # cases
    # The "No local results" line must NOT appear when local rows exist.
    assert "No local results yet" not in table


def test_missing_published_file_local_rows_only(tmp_path):
    """No published file -> official + local only, no published group."""
    official = _write_official(tmp_path)
    results = _write_local_result(tmp_path)
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
