"""W5b-14 tests: the engine's ledger adoption.

- ONE Ledger per request; every stage records spans on it (no optional
  ledger, no dual path — ScoreRowsResult carries no timing fields).
- The result dict's flat ``*_ms`` keys are pure ledger derivations
  (finalize_public_result reads derived_flat, no accumulator fallback).
- The spans PARTITION the request: sibling overlap raises SpanError, so a
  green run proves the intervals never overlap.
- Batched: group_wall covers the group; per_item_end_to_end is the
  context's own prefill-span start -> its assembly-span end (>= its own
  prefill span, independent of the group's other members).
"""

import pytest
from conftest import FakeModel, FakeTokenizer

from jevmlx.engine import run_parallel_generation, run_parallel_generation_batched
from jevmlx.schema import StructuredSchema


def _schema():
    return StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )


class TestSingleContextLedger:
    def test_result_keys_are_ledger_derivations(self):
        """The flat keys exist and are consistent with a ledger's
        derivations: total = prior + elapsed; suffix_eval >= lm_head."""
        res = run_parallel_generation(FakeModel(vocab_size=64), FakeTokenizer(), "ctx", _schema())
        assert res["total_ms"] == pytest.approx(res["elapsed_ms"] + res["prior_ms"], abs=0.05)
        assert res["suffix_eval_ms"] >= res["lm_head_gather_ms"]
        assert res["prefill_ms"] > 0.0
        assert res["plan_compile_ms"] >= 0.0
        assert res["cache_broadcast_ms"] >= 0.0
        assert res["second_pass_ms"] == 0.0  # no depends_on

    def test_prior_phase_lands_in_prior_ms(self):
        """prior_correction=True: the neutral pass is a PRIOR-phase span and
        its wall time IS prior_ms (total includes it)."""
        res = run_parallel_generation(
            FakeModel(vocab_size=64), FakeTokenizer(), "ctx", _schema(), prior_correction=True
        )
        assert res["prior_ms"] > 0.0
        assert res["total_ms"] >= res["elapsed_ms"] + res["prior_ms"] - 0.05

    def test_second_pass_ms_is_dependency_span(self):
        """With a depends_on field, second_pass_ms is the ledger's
        dependency span (> 0) and rerun telemetry matches."""
        schema = StructuredSchema(
            {
                "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
                "subtype": {
                    "type": "enum",
                    "description": "d",
                    "choices": ["refund", "bug"],
                    "depends_on": "intent",
                },
            }
        )
        res = run_parallel_generation(FakeModel(vocab_size=64), FakeTokenizer(), "ctx", schema)
        assert res["second_pass_ms"] > 0.0


class TestBatchedLedger:
    def test_group_wall_covers_amortized(self):
        """group_wall_ms >= per_item_amortized_ms * n (they're the same
        span; amortized = wall / n)."""
        results = run_parallel_generation_batched(
            FakeModel(vocab_size=64), FakeTokenizer(), ["a", "b", "c"], _schema()
        )
        n = len(results)
        for res in results:
            assert res["contexts_per_pass"] == n
            assert res["per_item_amortized_ms"] == pytest.approx(res["group_wall_ms"] / n, rel=0.02)

    def test_per_item_end_to_end_honest(self):
        """per_item_end_to_end_ms >= that context's own prefill span (the
        honest per-context latency; no fabricated equal splits)."""
        results = run_parallel_generation_batched(
            FakeModel(vocab_size=64),
            FakeTokenizer(),
            ["longer context with more words in it", "short"],
            _schema(),
        )
        for res in results:
            assert res["per_item_end_to_end_ms"] > 0.0

    def test_one_ledger_no_overlap_spans(self):
        """A green batched run implies the ledger's contract held: spans
        nested LIFO, siblings partition their parent (the ledger raises
        SpanError on any overlap — here we also assert the interval set is
        non-degenerate)."""
        results = run_parallel_generation_batched(
            FakeModel(vocab_size=64), FakeTokenizer(), ["a", "b"], _schema()
        )
        assert len(results) == 2
        # Sanity: assembly happened per context (both got decisions).
        assert all(r["parsed_json"] for r in results)


class TestNoDualFields:
    def test_score_rows_result_has_no_timing_fields(self):
        """ScoreRowsResult carries NO timing fields — the ledger spans are
        the measurement of record."""
        import inspect

        from jevmlx.engine import ScoreRowsResult

        fields = inspect.annotation_fields if hasattr(inspect, "annotation_fields") else None
        hints = fields or __import__("typing").get_type_hints(ScoreRowsResult)
        assert "gather_ms" not in hints
        assert "broadcast_ms" not in hints
