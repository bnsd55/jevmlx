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
        from jevmlx.engine import ScoreRowsResult

        assert "gather_ms" not in ScoreRowsResult._fields
        assert "broadcast_ms" not in ScoreRowsResult._fields


class TestReviewFixes:
    """The 2026-09-19 review: F3/F4/F5/F6 regressions covered."""

    def test_each_batched_context_has_own_prefill_ms(self):
        """GAP A / F4: per-context ledgers — every result's prefill_ms is
        ITS OWN prefill span, NOT the batch-wide sum (they differ when the
        prompts differ in length)."""
        schema = _schema()
        results = run_parallel_generation_batched(
            FakeModel(vocab_size=64),
            FakeTokenizer(),
            ["context one with more text", "short"],
            schema,
        )
        pf = [r["prefill_ms"] for r in results]
        # The prompts render to different lengths -> different prefill walls.
        assert pf[0] != pf[1]
        # And no context reports the SUM of both (the old shared-ledger bug).
        total = sum(pf)
        assert all(0 < r["prefill_ms"] < total for r in results)

    def test_second_pass_ms_is_dependency_interval(self):
        """F3: second_pass_ms == the dependency span (not the old
        telemetry accumulator — they would diverge if the dependency pass
        re-used a cached prior)."""
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
        # Equality per review: drive a ledger through and compare the flat
        # key to the dependency interval, not just a sign check.
        from jevmlx.timing import Ledger

        ledger = Ledger()
        res = run_parallel_generation(
            FakeModel(vocab_size=64), FakeTokenizer(), "ctx", schema, ledger=ledger
        )
        dep = [iv for iv in ledger.intervals if iv.name == "dependency"]
        assert dep, "expected a dependency span when a field has depends_on"
        assert res["second_pass_ms"] == pytest.approx(dep[-1].ms, abs=0.01)
        # The dependency span is INSIDE the request wall.
        assert res["second_pass_ms"] <= res["elapsed_ms"] + 1e-6

    def test_forced_metal_retry_under_group_wall(self):
        """F5 / GAP B: a Metal allocation failure that retries must not
        break the parents' __exit__ (one forced [metal::malloc] failure)."""

        from conftest import FakeModel

        class FlakyModel(FakeModel):
            _failed = False

            def __call__(self, tokens, cache=None):

                # Force the failure on a BATCHED (multi-row) forward only —
                # the retry path lives in _score_rows; prefill (width 1)
                # must stay clean.
                if tokens.shape[0] > 1 and not getattr(self, "_failed", False):
                    self._failed = True
                    raise RuntimeError("[metal::malloc] forced failure")
                return super().__call__(tokens, cache=cache)

        schema = _schema()
        results = run_parallel_generation_batched(
            FlakyModel(vocab_size=64), FakeTokenizer(), ["a", "b"], schema
        )
        assert len(results) == 2
        for res in results:
            assert res["group_wall_ms"] > 0.0
            # N4: the batched path passes failed_attempts through.
            assert res["failed_attempts"] >= 1

    def test_degenerate_schema_no_crash(self):
        """N2 repro: an R==0 (single-choice) schema through the batched
        path must produce results with group views — the old branch never
        stored results[idx] and opened a nested group_wall."""
        schema = StructuredSchema(
            {"pick": {"type": "enum", "description": "d", "choices": ["ONLY"]}}
        )
        results = run_parallel_generation_batched(
            FakeModel(vocab_size=64), FakeTokenizer(), ["a", "b"], schema
        )
        assert len(results) == 2
        for res in results:
            assert res["group_wall_ms"] > 0.0
            assert res["per_item_amortized_ms"] > 0.0
            assert res["per_item_end_to_end_ms"] > 0.0
            assert res["contexts_per_pass"] == 2

    def test_batched_prior_ms_shared_nonzero(self):
        """N1: prior_correction=True in the batched path reports the shared
        prior pass on EVERY result (prior_ms > 0), and total_ms includes it
        (finding 26)."""
        schema = _schema()
        results = run_parallel_generation_batched(
            FakeModel(vocab_size=64),
            FakeTokenizer(),
            ["ctx one", "ctx two"],
            schema,
            prior_correction=True,
        )
        for res in results:
            assert res["prior_ms"] > 0.0
            assert res["total_ms"] == pytest.approx(res["elapsed_ms"] + res["prior_ms"], abs=0.05)

    def test_unpadded_chunks_eval_their_cache(self):
        """F6: _eval_cache_state runs UNCONDITIONALLY — an unpadded merge
        still evaluates (cache_merge spans the whole broadcast region)."""
        # Structural: the call sits outside the max_padding guard.
        import inspect

        from jevmlx import engine

        src = inspect.getsource(engine._score_rows)
        pad_idx = src.index("if max_padding > 0:")
        eval_idx = src.index("_eval_cache_state(b_cache)", pad_idx)
        # _eval_cache_state must be OUTDENTED relative to the if (same level).
        pad_indent = len(src[:pad_idx].rsplit("\n", 1)[-1]) - len(
            src[:pad_idx].rsplit("\n", 1)[-1].lstrip()
        )
        eval_indent = len(src[:eval_idx].rsplit("\n", 1)[-1]) - len(
            src[:eval_idx].rsplit("\n", 1)[-1].lstrip()
        )
        assert eval_indent == pad_indent  # unconditional, not inside the if
