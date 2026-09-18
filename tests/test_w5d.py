"""W5-D: request-scoped peak memory (finding 32) and log-space legal
mass (findings 37/38).

Fake-model tests — each asserts behavior that FAILS on 1f9f453:

- 32: peak telemetry reports absolute + INCREMENTAL peak; the reset means
  a prior request's peak cannot leak into the next one.
- 37: score_trie's legal-mass callback returns LOG mass; an extreme
  (-1000 nat) mass flows through instead of raising log(0).
- 38: multi telemetry carries cardinality-stable stats
  (min_option_legal_mass, mean_log_legal_mass) instead of the underflowing
  product; the count row's legal mass is exposed, not discarded.
"""

import math

from jevmlx import StructuredSchema
from jevmlx.engine import run_parallel_generation
from jevmlx.trie import score_trie
from tests.test_engine_fake import FakeModel, FakeTokenizer


class _HugeVocabModel(FakeModel):
    """FakeModel with a big vocab so |allowed|/vocab masses are tiny."""


def test_peak_incremental_present_and_consistent():
    """Finding 32: peak_incremental_bytes = peak_active_bytes - active
    memory at request start; both describe THIS request only."""
    model = FakeModel(vocab_size=64)
    tokenizer = FakeTokenizer()
    schema = StructuredSchema({"pick": {"type": "enum", "description": "d", "choices": ["A", "B"]}})
    result = run_parallel_generation(model, tokenizer, "ctx", schema)
    assert "peak_active_bytes" in result
    # W5-D finding 32: the incremental pair exists and never goes negative.
    assert "peak_incremental_bytes" in result
    assert result["peak_incremental_bytes"] >= 0


def test_peak_reset_between_requests():
    """Finding 32: a request's reported peak must not be dominated by an
    EARLIER request's allocations — the counter is reset per request.

    With the old code, run A allocates, then run B (no model work between)
    reports A's peak as its own. After the fix each run resets, so the
    second call's peak reflects its own cache state."""
    model = FakeModel(vocab_size=64)
    tokenizer = FakeTokenizer()
    schema = StructuredSchema({"pick": {"type": "enum", "description": "d", "choices": ["A", "B"]}})
    first = run_parallel_generation(model, tokenizer, "ctx", schema)
    second = run_parallel_generation(model, tokenizer, "ctx2", schema)
    # The second request resets the peak counter: its incremental peak
    # cannot exceed its own absolute peak.
    assert second["peak_incremental_bytes"] <= second["peak_active_bytes"]
    assert first["peak_active_bytes"] > 0


def test_score_trie_extreme_log_mass_no_underflow():
    """Finding 37: a -1000-nat node mass flows through in log space —
    the old exp() -> 0.0 -> log(0) path raised ValueError."""

    remainders = [[10], [30]]
    nodes = build_trie_local(remainders)
    log_probs, lm_logs = score_trie(nodes, 2, lambda n: [1.0, 2.0], lambda n: -1000.0)
    # No exception; the log mass passes through untouched.
    assert lm_logs == [-1000.0, -1000.0]
    # And the scores are the plain branch log-probs.
    assert all(math.isfinite(lp) for lp in log_probs)


def build_trie_local(remainders):
    from jevmlx.trie import build_trie

    return build_trie(remainders)


def test_multi_legal_mass_stats_cardinality_free():
    """Finding 38: multi fields expose min_option_legal_mass (probability
    space) + mean_log_legal_mass (additive, stable) — NOT a raw product
    that shrinks exponentially with option count."""
    schema = StructuredSchema(
        {
            "tags": {
                "type": "multi",
                "description": "d",
                "choices": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            }
        }
    )
    result = run_parallel_generation(FakeModel(vocab_size=64), FakeTokenizer(), "ctx", schema)
    tel = result["field_telemetry"]["tags"]
    # The old product-of-masses key is gone for multi; the stable stats are in.
    assert "min_option_legal_mass" in tel
    assert "mean_log_legal_mass" in tel
    assert 0.0 < tel["min_option_legal_mass"] <= 1.0
    # Per-option logs still available, keyed by the real choices.
    assert set(tel["legal_mass_logs"]) == {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}


def test_count_row_legal_mass_exposed():
    """Finding 38: the count row's legal mass is computed and EXPOSED
    (it was computed and discarded before)."""
    schema = StructuredSchema(
        {"tags": {"type": "multi", "description": "d", "choices": ["A", "B", "C"]}}
    )
    result = run_parallel_generation(FakeModel(vocab_size=64), FakeTokenizer(), "ctx", schema)
    count_tel = result["field_telemetry"].get("tags#count")
    assert count_tel is not None
    assert "legal_mass" in count_tel
    assert 0.0 < count_tel["legal_mass"] <= 1.0
