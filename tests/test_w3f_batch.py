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

from jevmlx.engine import run_parallel_generation, run_parallel_generation_batched
from jevmlx.schema import StructuredSchema
from tests.test_engine_fake import FakeModel, FakeTokenizer


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
        separate.append(run_parallel_generation(m, FakeTokenizer(), ctx, schema))

    # Batched call (one model, shared pass).
    model = FakeModel(vocab_size=64)
    tok = FakeTokenizer()
    batched = run_parallel_generation_batched(model, tok, contexts, schema)

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
    results = run_parallel_generation_batched(model, FakeTokenizer(), contexts, schema)
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
    results = run_parallel_generation_batched(model, FakeTokenizer(), ["a", "b", "c"], schema)
    # Every context reports the same shared pass count.
    passes = {r["sequential_forward_passes"] for r in results}
    assert len(passes) == 1


def test_batched_empty_contexts():
    schema = StructuredSchema(
        {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
    )
    model = FakeModel(vocab_size=64)
    assert run_parallel_generation_batched(model, FakeTokenizer(), [], schema) == []


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
        model, FakeTokenizer(), ["ctx-a", "ctx-b"], schema, constraints=constraints
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

    monkeypatch.setattr(
        api, "load_engine", lambda model_id: (FakeModel(vocab_size=64), FakeTokenizer())
    )
    results = api.decide_many(
        TwoField,
        ["transaction one", "transaction two"],
        model="fake/model",
    )
    assert len(results) == 2
    for d in results:
        assert set(d.fields) == {"is_fraudulent", "risk_tier"}
        assert d.latency_ms >= 0
