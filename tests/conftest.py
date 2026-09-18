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
The FakeModel path stays exact (deterministic zeros).
"""

# Real-model log_score parity tolerance (nats). See module docstring.
PARITY_ATOL = 5e-2
