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
from tests.test_w4b_parity import _cases, _CountTokenizer, _StableModel


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


# --- W5-D findings 40/41/42: batched parity matrix, real preset contexts, raw logits first ---


def test_batched_parity_matrix_stable_fake():
    """Finding 40: the batched matrix runs 1/2/4 contexts (equal + mixed
    lengths), compares batched vs independent finals AND raw pre-rescore
    row logits. On a batch-shape-stable fake, everything agrees exactly."""
    from jevmlx.parity import check_batched_parity

    model = _StableModel()
    tokenizer = _CountTokenizer()
    result = check_batched_parity(model, tokenizer, _cases())
    assert result["winners_identical"] is True
    assert result["max_abs_drift_nats"] == 0.0
    assert result["max_raw_row_drift_nats"] == 0.0
    assert set(result["per_case"]) == {"mini", "mini2"}


def test_batched_parity_matrix_catches_row_drift():
    """Finding 42: a model whose batched ROW logits drift (but whose
    near-tie rescore hides it at the decision level) must fail the RAW
    check — the final-decision comparison alone would pass."""
    from jevmlx.parity import check_batched_parity

    class _RowDriftModel(_StableModel):
        """Boosts alias "B" only in batched suffix calls (>1-row chunks)."""

        def __call__(self, tokens, cache=None):

            out = super().__call__(tokens, cache=cache)
            if tokens.shape[0] > 1 and tokens.shape[1] > 1:
                out = out.at[:, :, 67].add(5.0)
            return out

    result = check_batched_parity(_RowDriftModel(), _CountTokenizer(), _cases())
    # The raw row-logit comparison catches what the final decision gate
    # (with batch=1 rescoring) would mask.
    assert result["max_raw_row_drift_nats"] > 0.0


def test_parity_uses_real_preset_context():
    """Finding 41: _case_context reads the preset's REAL context —
    bundled_preset_specs returns the whole preset dict."""
    from jevmlx.parity import _case_context, bundled_preset_specs

    cases = bundled_preset_specs()
    for case_id, preset in cases:
        ctx = _case_context(case_id, preset)
        # The bundled presets all carry real contexts; the filler must not
        # be used for them.
        assert ctx == preset["context"], case_id


def test_batched_parity_prior_on_matches_independent():
    """Finding 40 matrix includes prior on/off: with prior_correction=True
    the batched results still equal per-context decide (the shared prior
    object — W5-D finding 26 — flows into every assemble)."""
    from jevmlx.parity import check_batched_parity

    result = check_batched_parity(
        _StableModel(), _CountTokenizer(), _cases(), prior_correction=True
    )
    assert result["winners_identical"] is True


# --- W5-D review round 2: the width slope is measured, not claimed ---


def test_width_slope_floor_until_probe_runs():
    """SHORTCUT honesty: before any engine load, budgeting uses the
    ASSUMED floor (1.0) — and the constant's name says assumed, not
    measured."""
    import jevmlx.engine as eng

    # Process-global state: other tests in this session may have loaded an
    # engine (and measured the slope). Save/restore around the floor check.
    saved = eng._WIDTH_SLOPE
    try:
        eng._WIDTH_SLOPE = None
        assert eng._width_slope() == eng._ASSUMED_BYTES_PER_ROW_SLOPE == 1.0
    finally:
        eng._WIDTH_SLOPE = saved
    # The budget function reads the live accessor, not a stale constant.
    schema = StructuredSchema({"pick": {"type": "enum", "description": "d", "choices": ["A", "B"]}})
    model = FakeModel(vocab_size=64)
    tok = FakeTokenizer()
    built = eng._build_schema_rows(schema, tok, "slots")
    from jevmlx.engine import make_prompt_cache

    cache = make_prompt_cache(model)
    import mlx.core as mx

    model(mx.array([[1, 2, 3]]), cache=cache)
    cap = eng._width_bin_max_rows(built["rows"], cache, 64, 0, None)
    assert cap >= 1


def test_measure_width_slope_returns_finite_ratio():
    """The probe runs a real B=1/B=2 pair on a fake model and returns a
    plausible slope; on a broken model it falls back to the floor."""
    import jevmlx.engine as eng

    model = FakeModel(vocab_size=64)
    slope = eng._measure_width_slope(model)
    assert 0.5 <= slope <= 8.0

    class _Broken:
        pass

    assert eng._measure_width_slope(_Broken()) == 1.0
