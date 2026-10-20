"""#194: the post-dependency MAP must rebuild the decision record.

A constrained case with a depends_on field where the dependency wave's
re-scored argmax VIOLATES a case constraint: the constraint MAP forces the
value back, and the field's telemetry (probability, log_score_margin,
probability_margin, tie, alternatives, parsed prob) must describe the
RETURNED value — the same recompute reconcile_case_constraints runs on
the pre-dependency MAP.

The scenario (deterministic token-id model, see _TokenBiasModel):

- intent (billing/technical) prefers "billing" (high margin, confident
  parent);
- subtype (bug/refund) depends_on intent and prefers "bug" — which the
  implies-constraint forbids under billing;
- first pass: subtype=bug -> MAP forces refund (child is a reconciled
  field, so the dependency wave re-scores it conditioned on the parent);
- the wave's argmax is "bug" AGAIN (the bias is content-independent);
- the post-dependency MAP forces refund a second time.

The RETURNED value is "refund" while the wave's winner was "bug": on main
the telemetry keeps the wave's record (P(bug) ≈ 1, positive margin); with
the fix it describes "refund" (tiny probability, negative margin).
"""

from __future__ import annotations

from conftest import FakeModel, FakeTokenizer, make_engine

from jevmlx.engine import run_parallel_generation
from jevmlx.schema import StructuredSchema


class _TokenBiasModel(FakeModel):
    """Constant per-token-id logits over the zero base."""

    def __init__(self, bias: dict[int, float], vocab_size: int = 128):
        super().__init__(vocab_size=vocab_size)
        self._bias = bias

    def __call__(self, tokens, cache=None):
        out = super().__call__(tokens, cache)
        for tid, val in self._bias.items():
            out = out.at[:, :, tid].add(val)
        return out


_SCHEMA = {
    "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
    "subtype": {
        "type": "enum",
        "description": "d",
        "choices": ["bug", "refund"],
        "depends_on": "intent",
    },
}

_CONSTRAINT = {
    "type": "implies",
    "parent": "intent",
    "child": "subtype",
    "mapping": {"billing": ["refund"], "technical": ["bug"]},
}


def _biased_model() -> _TokenBiasModel:
    # The trie decides on each choice's FIRST-char token id: intent ->
    # [b(illing), t(echnical)], subtype -> [b(ug), r(efund)]. billing and
    # bug share id 'b' — the parent's confident choice and the child's
    # violating choice light up together; technical and refund stay flat.
    tok = FakeTokenizer()
    return _TokenBiasModel({tok.encode("b", add_special_tokens=False)[0]: 5.0})


def test_post_dependency_map_rebuilds_decision_record():
    model = _biased_model()
    result = run_parallel_generation(
        make_engine(model, FakeTokenizer()),
        "ctx",
        StructuredSchema(_SCHEMA),
        constraints=[_CONSTRAINT],
    )

    # The wave re-ran the reconciled child...
    assert "subtype" in result["rerun_fields"]
    # ...and the constraint forces the returned value back to refund (both
    # on main and with the fix — winners are not the question).
    assert result["parsed_json"]["subtype"]["value"] == "refund"

    tel = result["field_telemetry"]["subtype"]
    # The record describes the RETURNED value (issue #194). refund is the wave
    # distribution's LOSER (bug carries the +5 bias), so its probability is
    # tiny and its margin to the best competitor negative.
    assert tel["value"] == "refund"
    assert tel["probability"] < 0.5, (
        f"telemetry probability {tel['probability']} looks like the wave's "
        "winner, not the returned value"
    )
    assert tel["log_score_margin"] < 0, (
        f"log_score_margin {tel['log_score_margin']} is not the returned "
        "value's gap to its best competitor"
    )
    assert result["parsed_json"]["subtype"]["prob"] == tel["probability"]
    # The returned value's competitors: bug first (the distribution's top
    # choice OTHER than the returned value).
    assert tel["alternatives"][0]["choice"] == "bug"
    assert tel["semantics"]["constraint_changed"] is True
