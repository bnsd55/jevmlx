"""W2-D: legal_mass telemetry — the per-branch leakage signal.

legal_mass = P(model wanted any valid code at the branch) =
sum(exp(z_allowed)) / sum(exp(z_vocab)). A branch can confidently pick A
over B even when almost all unconstrained mass is on a reasoning token;
legal_mass flags that leakage. Always computed (the ~0.002 Metal FP drift
from the full-vocab logsumexp means bit-identical batch parity was never a
real invariant on Metal; W1-A asserts winners identical + log_scores
within atol PARITY_ATOL).
"""

import math

import pytest

from jevmlx import StructuredSchema
from jevmlx.engine import run_parallel_generation
from jevmlx.trie import score_trie
from tests.test_prompt_v2 import FakeModel, FakeTokenizer

SCHEMA = StructuredSchema(
    {
        "risk": {
            "type": "enum",
            "description": "risk",
            "choices": ["LOW", "HIGH"],
        },
    }
)


def _run():
    return run_parallel_generation(
        FakeModel(vocab_size=64),
        FakeTokenizer(),
        "ctx",
        SCHEMA,
    )


def test_legal_mass_always_present():
    """legal_mass is always in the telemetry (no opt-in flag)."""
    result = _run()
    tel = result["field_telemetry"]["risk"]
    assert "legal_mass" in tel
    assert "legal_mass_logs" in tel
    # FakeModel emits zero logits: every token is equally likely, so
    # legal_mass = |allowed| / vocab_size. The enum has one branch node with
    # 2 children (LOW, HIGH aliases), so |allowed| = 2, vocab = 64.
    assert tel["legal_mass"] == pytest.approx(2 / 64, abs=1e-5)


def test_legal_mass_in_unit_interval():
    """legal_mass is a probability: strictly in (0, 1]."""
    result = _run()
    tel = result["field_telemetry"]["risk"]
    assert 0.0 < tel["legal_mass"] <= 1.0


def test_legal_mass_logs_keyed_by_real_choice():
    """legal_mass_logs uses the same real-choice keying as log_scores."""
    result = _run()
    tel = result["field_telemetry"]["risk"]
    assert set(tel["legal_mass_logs"]) == set(tel["log_scores"])


def test_score_trie_returns_tuple():
    """score_trie returns (log_probs, legal_mass_logs) — one function."""
    from jevmlx.trie import build_trie

    # Two choices, each a single-token remainder; root branches on token 10 vs 30.
    remainders = [[10], [30]]
    nodes = build_trie(remainders)
    logits = [1.0, 2.0]  # branch node 0, children in sorted-token order
    log_probs, lm_logs = score_trie(nodes, 2, lambda n: logits, lambda n: 0.5)
    # log_probs are the constrained-path log-probs (log_softmax of the logits).
    # legal_mass_at_node=0.5 -> log = ln(0.5); both choices pass the one branch.
    assert lm_logs == [math.log(0.5), math.log(0.5)]
    # Without the callback, legal mass logs are 0.0 (mass 1.0) — for the
    # MLX-free unit tests.
    _lp, lm_logs_none = score_trie(nodes, 2, lambda n: logits, None)
    assert lm_logs_none == [0.0, 0.0]
    # log_probs are identical with or without the legal-mass callback.
    assert log_probs == _lp
