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

import pytest

# Real-model log_score parity tolerance (nats) == engine.INSTABILITY_BAND;
# re-exported under the tests' name for the parity suites.
from jevmlx.engine import INSTABILITY_BAND

PARITY_ATOL = INSTABILITY_BAND

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.fixture(scope="module")
def engine():
    """The real 0.5B model, loaded once per module. Shared by every slow
    test that needs a live engine (test_engine, test_w4b_parity)."""
    from jevmlx.engine import load_engine

    return load_engine(MODEL_ID)
