"""Tests for benchmarks.typed_decisions.fetch (LocalLLaMA/typed-decisions -> JSONL).

Pure conversion tests on in-memory rows shaped like the dataset's parquet
columns (nested columns are JSON strings). No network, no pyarrow.
"""

from __future__ import annotations

import hashlib
import json

from benchmarks.typed_decisions.fetch import (
    PARSER_VERSION,
    field_schema,
    row_to_record,
    write_outputs,
)
from jevmlx.evalmetrics import typesafe_agreement


def _row(**overrides) -> dict:
    """One dataset row: choice + noul + score questions, one ambiguous field."""
    questions = {
        "action": {
            "type": "choice",
            "instructions": "What should happen?",
            "criteria": {"continue": "Keep going.", "stop": "Halt now."},
        },
        "needs_review": {
            "type": "noul",
            "instructions": "A human should look.",
            "criteria": {"false": "No.", "true": "Yes."},
        },
        "risk": {
            "type": "score",
            "instructions": "How risky?",
            "criteria": ["Benign.", "Low.", "Moderate.", "High."],
        },
        "notes": {"type": "free_text", "instructions": "Anything else?"},
    }
    gold = {
        "action": {
            "type": "choice",
            "label": "continue",
            "probabilities": {"continue": 0.52, "stop": 0.48},
        },
        "needs_review": {
            "type": "noul",
            "label": "false",
            "probabilities": {"false": 0.9, "true": 0.1},
        },
        "risk": {
            "type": "score",
            "label": "1",
            "probabilities": {"0": 0.2, "1": 0.7, "2": 0.1, "3": 0.0},
        },
    }
    row = {
        "id": "agent_trace_observability_000007",
        "workflow": "agent_trace_observability",
        "split": "test",
        "state": json.dumps({"task": "rotate cert", "steps": [1, 2]}),
        "questions": json.dumps(questions),
        "gold": json.dumps(gold),
        "label_agreement": json.dumps({"action": {"argmax_agree": False}}),
        "factors": json.dumps({"turns": 3}),
        "n_questions": 4,
    }
    row.update(overrides)
    return row


def test_field_schema_maps_question_types():
    assert field_schema({"type": "noul", "instructions": "x", "criteria": {}}) == {
        "type": "boolean",
        "description": "x",
    }
    choice = field_schema({"type": "choice", "instructions": "x", "criteria": {"b": "B", "a": "A"}})
    # Published order, not sorted.
    assert choice == {"type": "enum", "description": "x", "choices": ["b", "a"]}
    score = field_schema({"type": "score", "instructions": "x", "criteria": ["z", "o", "t", "h"]})
    assert score["choices"] == ["0", "1", "2", "3"]
    assert "0 = z; 1 = o" in score["description"]
    assert field_schema({"type": "free_text", "instructions": "x"}) is None


def test_row_to_record_contract_and_labels():
    record = row_to_record(_row())
    skipped = record.pop("skipped_questions")
    assert skipped == 1  # the free_text question
    # Same frozen contract as the typesafe fetcher (plus nothing extra).
    assert set(record) == {
        "id",
        "group_id",
        "source",
        "workflow",
        "benchmark_only",
        "schema",
        "context",
        "labels",
        "split",
        "meta",
    }
    assert record["id"] == (
        "typed-decisions/agent_trace_observability/agent_trace_observability_000007"
    )
    assert record["source"] == "typed-decisions"
    assert record["workflow"] == "agent_trace_observability"
    assert record["split"] == "test"
    assert record["benchmark_only"] is True
    # Labels in jevmlx form: bool for noul, strings for choice/score.
    assert record["labels"] == {"action": "continue", "needs_review": False, "risk": "1"}
    assert {k: v["type"] for k, v in record["schema"].items()} == {
        "action": "enum",
        "needs_review": "boolean",
        "risk": "enum",
    }
    # Context is the state rendered as JSON (what the model reads).
    assert json.loads(record["context"]) == {"task": "rotate cert", "steps": [1, 2]}


def test_consensus_margin_and_ambiguity():
    meta = row_to_record(_row())["meta"]
    assert meta["consensus"]["action"] == {"continue": 0.52, "stop": 0.48}
    assert abs(meta["margin"]["action"] - 0.04) < 1e-9
    assert abs(meta["margin"]["needs_review"] - 0.8) < 1e-9
    # action margin 0.04 < AMBIGUOUS_MARGIN (0.1) -> flagged; the others are not.
    assert meta["ambiguous"] == ["action"]
    # Dataset-native agreement/factor columns ride along for slicing.
    assert meta["label_agreement"] == {"action": {"argmax_agree": False}}
    assert meta["factors"] == {"turns": 3}


def test_nested_columns_accept_dicts_too():
    """Parquet readers may already decode the JSON columns; both forms work."""
    raw = _row()
    decoded = {
        **raw,
        "state": json.loads(raw["state"]),
        "questions": json.loads(raw["questions"]),
        "gold": json.loads(raw["gold"]),
        "label_agreement": json.loads(raw["label_agreement"]),
        "factors": json.loads(raw["factors"]),
    }
    assert row_to_record(decoded) == row_to_record(raw)


def test_agreement_metric_counts_typed_decisions_source():
    """evalmetrics treats typed-decisions like typesafe (consensus labels)."""
    lines = [
        {
            "source": "typed-decisions",
            "workflow": "customer_service",
            "case_id": "c1",
            "field": "action",
            "label": "continue",
            "prediction": "continue",
            "valid": True,
            "correct": True,
            "ambiguous": ["action"],
        },
        {
            "source": "typed-decisions",
            "workflow": "customer_service",
            "case_id": "c1",
            "field": "risk",
            "label": "1",
            "prediction": "2",
            "valid": True,
            "correct": False,
            "ambiguous": ["action"],
        },
    ]
    agreement = typesafe_agreement(lines)
    assert agreement["overall"] == 0.5
    assert agreement["n_fields"] == 2 and agreement["n_cases"] == 1
    # Common subset excludes the ambiguous field: only the wrong one remains.
    assert agreement["agreement_common_subset"] == 0.0
    assert typesafe_agreement([{**lines[0], "source": "bundled"}]) == {}


def test_write_outputs_lock_pins_revision_and_hash(tmp_path):
    record = row_to_record(_row())
    record.pop("skipped_questions")
    out = tmp_path / "cache" / "typed-decisions.jsonl"
    lock = tmp_path / "cache" / "typed-decisions.dataset.lock.json"
    source = {
        "repo_id": "LocalLLaMA/typed-decisions",
        "revision": "abc123",
        "files": [{"file": "customer_service/test-00000-of-00001.parquet", "sha256": "ff"}],
    }
    counts = {"records": 1}
    assert write_outputs([record], out, source, counts, lock) == lock
    written = json.loads(out.read_text().splitlines()[0])
    assert written == record
    lock_data = json.loads(lock.read_text())
    assert lock_data["parser_version"] == PARSER_VERSION
    assert lock_data["sources"] == [source]
    assert lock_data["counts"] == counts
    assert lock_data["cases_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    # Default lock location mirrors the typesafe fetcher.
    default_lock = write_outputs([record], tmp_path / "d" / "c.jsonl", source, counts)
    assert default_lock == tmp_path / "d" / "dataset.lock.json"
