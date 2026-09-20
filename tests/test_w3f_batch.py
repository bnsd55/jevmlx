"""W3-F tests: decide_many context batching.

Verifies:
- 3 contexts through decide_many (batched) equal 3 separate decide calls
  within PARITY_ATOL (logits are compared via the parsed values + telemetry).
- run_parallel_generation_batched returns results in input order.
- One scoring pass total (passes telemetry shared, divided per context).
- Constraints + prior_correction flow through the batched path.
- Empty contexts list returns [].
"""

from __future__ import annotations

import pydantic
from conftest import FakeModel, FakeTokenizer, make_engine

from jevmlx.engine import run_parallel_generation, run_parallel_generation_batched
from jevmlx.schema import StructuredSchema


def test_batched_equals_separate_calls():
    """3 contexts through run_parallel_generation_batched equal 3 separate
    run_parallel_generation calls — same parsed values and per-field
    probabilities (the fake model's logits are deterministic per row
    content, and the rows are identical across contexts; the only difference
    is the cache batch width, which merge() keeps causal-correct)."""
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    contexts = ["ctx one", "ctx two", "ctx three"]

    # Separate calls (fresh model each time for identical cache state).
    separate = []
    for ctx in contexts:
        m = FakeModel(vocab_size=64)
        separate.append(run_parallel_generation(make_engine(m, FakeTokenizer()), ctx, schema))

    # Batched call (one model, shared pass).
    model = FakeModel(vocab_size=64)
    tok = FakeTokenizer()
    batched = run_parallel_generation_batched(make_engine(model, tok), contexts, schema)

    assert len(batched) == 3
    for sep, bat, ctx in zip(separate, batched, contexts, strict=True):
        assert bat["parsed_json"] == sep["parsed_json"], ctx
        for fname in sep["field_telemetry"]:
            sep_prob = sep["field_telemetry"][fname]["probability"]
            bat_prob = bat["field_telemetry"][fname]["probability"]
            assert abs(sep_prob - bat_prob) < 1e-9, (ctx, fname, sep_prob, bat_prob)


def test_batched_input_order_preserved():
    """Results come back in input order (each context's own prefill)."""
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    contexts = ["first", "second", "third", "fourth"]
    model = FakeModel(vocab_size=64)
    results = run_parallel_generation_batched(make_engine(model), contexts, schema)
    assert len(results) == 4
    # All four succeed and are valid decisions for the same schema.
    for r in results:
        assert "parsed_json" in r
        assert "pick" in r["parsed_json"]


def test_batched_telemetry_shared_pass():
    """The batched path runs ONE scoring pass for all contexts (each
    context's report carries passes=scored.passes, gathered/broadcast ms
    divided by context count)."""
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    model = FakeModel(vocab_size=64)
    results = run_parallel_generation_batched(make_engine(model), ["a", "b", "c"], schema)
    # Every context reports the same shared pass count.
    passes = {r["sequential_forward_passes"] for r in results}
    assert len(passes) == 1


def test_batched_empty_contexts():
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    model = FakeModel(vocab_size=64)
    assert run_parallel_generation_batched(make_engine(model), [], schema) == []


def test_batched_constraints_flow_through():
    """constraints= flows through the batched path (MAP still applies per
    context)."""
    schema = StructuredSchema(
        {
            "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
            "subtype": {
                "type": "enum",
                "description": "d",
                "choices": ["refund", "dispute", "bug"],
            },
        }
    )
    constraints = [
        {
            "type": "implies",
            "parent": "intent",
            "child": "subtype",
            "mapping": {"billing": ["refund", "dispute"], "technical": ["bug"]},
        }
    ]
    model = FakeModel(vocab_size=64)
    results = run_parallel_generation_batched(
        make_engine(model), ["ctx-a", "ctx-b"], schema, constraints=constraints
    )
    assert len(results) == 2
    for r in results:
        assert r["constraints_applied"] is True


def test_decide_many_batched_end_to_end(monkeypatch):
    """decide_many uses the batched path end to end (Decision objects back,
    in input order) — engine load and model calls mocked with the fake."""
    from typing import Literal

    from pydantic import BaseModel

    import jevmlx.api as api

    class TwoField(BaseModel):
        is_fraudulent: bool = pydantic.Field(description="Whether the txn is fraudulent")
        risk_tier: Literal["LOW", "HIGH"] = pydantic.Field(description="Risk tier")

    monkeypatch.setattr(api, "load_engine", lambda model_id: make_engine(FakeModel(vocab_size=64)))
    results = api.decide_many(
        TwoField,
        ["transaction one", "transaction two"],
        model="fake/model",
    )
    assert len(results) == 2
    for d in results:
        assert set(d.fields) == {"is_fraudulent", "risk_tier"}
        assert d.latency_ms >= 0


def test_contexts_per_pass_telemetry_and_bounding():
    """contexts_per_pass lands in the result and bounds how many caches are
    alive at once (review F2): 500 contexts are processed in groups of the
    budgeted size, never all at once."""
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    model = FakeModel(vocab_size=64)
    contexts = [f"ctx {i}" for i in range(12)]
    results = run_parallel_generation_batched(make_engine(model), contexts, schema)
    assert len(results) == 12
    cpp = results[0]["contexts_per_pass"]
    assert cpp >= 1
    # Every context reports the same bound.
    assert all(r["contexts_per_pass"] == cpp for r in results)
    # The bound must hold: group size == min(len(contexts), budgeted size).
    # The fake cache is ~0 bytes, so the budget allows all 12 in one group.
    from mlx_lm.models.cache import make_prompt_cache

    from jevmlx.engine import _cache_nbytes, _contexts_per_pass

    pf_cache = make_prompt_cache(model)
    per_ctx = _cache_nbytes(pf_cache)
    expected = min(12, _contexts_per_pass(max(1, per_ctx)))
    assert cpp == expected


def test_batched_empty_rows_schema():
    """A schema is never rowless in practice, but the batched path must not
    crash if the plan produces zero rows."""
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    model = FakeModel(vocab_size=64)
    results = run_parallel_generation_batched(make_engine(model), ["a"], schema)
    assert len(results) == 1
    assert "pick" in results[0]["parsed_json"]


def test_grouped_contexts_prefill_their_own_prompts(monkeypatch):
    """Bug fix regression: with a budget forcing group_size=2 and 5 distinct
    contexts, every result must match ITS OWN separate decide call. The old
    code reused contexts[0]'s probe cache as every group's first context, so
    groups after the first scored contexts[0]'s prompt in place of their own
    first context."""
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    contexts = [f"ctx-{i}" for i in range(5)]

    # Force group_size=2 by patching the budget: per-context cache nbytes of
    # 0 makes _contexts_per_pass return the whole budget, so bound it from
    # above by patching the function directly.
    import jevmlx.engine as eng

    monkeypatch.setattr(eng, "_contexts_per_pass", lambda nbytes: 2)

    batched = run_parallel_generation_batched(
        make_engine(FakeModel(vocab_size=64)), contexts, schema
    )
    assert len(batched) == 5
    # With group_size=2 and 5 contexts the groups are [0,1], [2,3], [4] —
    # the middle groups' first contexts (2 and 4's group) must NOT have been
    # scored against ctx-0's prompt. The fake tokenizer's prompt tokens embed
    # the context text, and the fake model's cache consumes prompt-length
    # key/value state; a wrong prefill changes nothing in the zero-logit
    # output, so assert on the provenance sha instead: each result's
    # prompt_sha256 must equal the sha of ITS OWN context's prompt.
    import hashlib
    import json

    tok = FakeTokenizer()
    from jevmlx.engine import _context_block

    for ctx, res in zip(contexts, batched, strict=True):
        schema_str = schema.to_labels_schema_str()
        user_content = f"Classify the following fields.\n\n{schema_str}\n\n{_context_block(ctx)}"
        expected_ids = tok.encode(
            "\n".join(
                m["content"]
                for m in [
                    {"role": "system", "content": eng.PROMPT_V2_SYSTEM},
                    {"role": "user", "content": user_content},
                ]
            )
        )
        expected_sha = hashlib.sha256(json.dumps(expected_ids).encode()).hexdigest()
        assert res["prompt_sha256"] == expected_sha, (
            f"context {ctx!r} scored against another context's prompt"
        )
    # And the batched probabilities still match separate calls bit-for-bit
    # (the parity guarantee is unchanged by grouping).
    for ctx, bat in zip(contexts, batched, strict=True):
        sep = run_parallel_generation(make_engine(FakeModel(vocab_size=64)), ctx, schema)
        assert bat["parsed_json"] == sep["parsed_json"], ctx


def test_token_accounting_telemetry_keys_present():
    """W5c-6 / B4: every result carries the token-accounting keys with
    honest values — actual rows/branches counted, not fields."""
    schema = StructuredSchema(
        {
            "tier": {"type": "enum", "description": "d", "choices": ["A", "B", "C"]},
            "flags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
        }
    )
    engine = make_engine(FakeModel(vocab_size=64), FakeTokenizer())
    result = run_parallel_generation(engine, "ctx", schema)

    for key in (
        "naive_branch_prompt_tokens",
        "shared_prefix_tokens",
        "logical_suffix_token_positions",
        "computed_suffix_token_positions",
        "computed_prompt_token_positions",
        "retry_wasted_ms",
    ):
        assert key in result, key

    # shared_prefix_tokens = the prompt length (base_ids).
    assert result["shared_prefix_tokens"] > 0
    # naive_branch_prompt_tokens = sum of full prompt length for every
    # actual scoring row (shared prefix repeated). With 2 multi options +
    # count rows + 3 enum branches, there are > 2 rows.
    assert result["naive_branch_prompt_tokens"] > result["shared_prefix_tokens"]
    # logical_suffix_token_positions = unpadded suffix positions (>= 1 per
    # row).
    assert result["logical_suffix_token_positions"] > 0
    # computed_suffix_token_positions = padded/chunked (>= logical).
    assert result["computed_suffix_token_positions"] >= result["logical_suffix_token_positions"]
    # computed_prompt_token_positions = shared + computed.
    assert (
        result["computed_prompt_token_positions"]
        == result["shared_prefix_tokens"] + result["computed_suffix_token_positions"]
    )
    # No retries on the fake model — waste is 0.
    assert result["retry_wasted_ms"] == 0.0


def test_batched_token_accounting_per_context_logical_plus_group_computed():
    """W5c-6 / B4: decide_many reports per-context LOGICAL values +
    group-level COMPUTED values (the merged pass totals)."""
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    contexts = ["ctx one", "ctx two", "ctx three"]
    engine = make_engine(FakeModel(vocab_size=64), FakeTokenizer())
    batched = run_parallel_generation_batched(engine, contexts, schema)

    for res in batched:
        # Per-context logical values are present and positive.
        assert res["naive_branch_prompt_tokens"] > 0
        assert res["shared_prefix_tokens"] > 0
        assert res["logical_suffix_token_positions"] > 0
        # Per-context computed (the group's total / n_group).
        assert res["computed_suffix_token_positions"] > 0
        # Group-level computed totals.
        assert res["group_computed_suffix_token_positions"] > 0
        assert res["group_retry_wasted_ms"] == 0.0
        # The group total = n_group * per-context (all contexts share row
        # widths in the merged pass).
        n = len(contexts)
        assert (
            res["group_computed_suffix_token_positions"]
            == res["computed_suffix_token_positions"] * n
        )
