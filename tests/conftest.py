"""Shared test fixtures and constants.

PARITY_ATOL is the tolerance for real-model batch-vs-chunked log_score
parity: Metal batched matmuls tile differently at different batch shapes,
and the legal-mass full-vocab logsumexp (W2-D) adds a reduction that
perturbs the lazy evaluation graph.

Measured (W3-C, Qwen2.5-0.5B-4bit on M5, fintech_fraud + support_triage,
max_rows=3 vs full): worst drift 0.0293 nats — IDENTICAL on main before and
after the W3-C branch, i.e. pre-existing batch-shape noise, not a W3-C
regression. Coder1's F3 proposed 1e-2; the measurement shows 1e-2 fails on
main itself, so the constant ships at 5e-2 pending a remeasure.

W3-E: the constant now LIVES in jevmlx.engine as INSTABILITY_BAND (the
near-tie rescore band uses the same measurement) and is re-exported here so
tests and engine share exactly one number.
The FakeModel path stays exact (deterministic zeros).
"""

# Real-model log_score parity tolerance (nats) == engine.INSTABILITY_BAND;
# re-exported under the tests' name for the parity suites.
from jevmlx.engine import INSTABILITY_BAND

PARITY_ATOL = INSTABILITY_BAND
