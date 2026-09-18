"""Shared test fixtures and constants.

PARITY_ATOL is the tolerance for real-model batch-vs-chunked log_score
parity: Metal batched matmuls tile differently at different batch shapes,
and the legal-mass full-vocab logsumexp (W2-D) adds a reduction that
perturbs the lazy evaluation graph.

Measured (W3-C, Qwen2.5-0.5B-4bit on M5, fintech_fraud + support_triage,
max_rows=3 vs full): worst drift 0.0293 nats — IDENTICAL on main before and
after the W3-C branch, i.e. pre-existing batch-shape noise, not a W3-C
regression. Coder1's F3 proposed 1e-2; the measurement shows 1e-2 fails on
main itself, so the constant ships at 5e-2 pending a remeasure. Both the
W3-A/W3-C parity suite and coder3's W2-D tests import this constant.

W2-A's per-field prompt tails (40-100 tokens/row vs 4 for the old lead_in)
increase Metal batch-shape tiling drift; measured ~0.027 nats on the action
row — inside the existing 5e-2 band, so the constant is unchanged.
The FakeModel path stays exact (deterministic zeros).
"""

# Real-model log_score parity tolerance (nats). See module docstring.
PARITY_ATOL = 5e-2


def make_test_renderer(tokenizer, schema, scoring="labels"):
    """Build a render_field_prompt for tests that compile plans directly.

    Tests that only check token-level structure (remainders, codebook, codes)
    still need a render_field_prompt now that W2-A made it mandatory. This
    helper creates one with an empty context — the prompt tails are
    irrelevant to those tests; they only inspect shared_ids/remainders/etc.
    """
    from jevmlx.engine import make_field_prompt_renderer

    return make_field_prompt_renderer(tokenizer, "", schema, scoring)
