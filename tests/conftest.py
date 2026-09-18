"""Shared test fixtures and constants.

PARITY_ATOL is the tolerance for real-model batch-vs-chunked log_score
parity: Metal batched matmuls tile differently at different batch shapes,
and the legal-mass full-vocab logsumexp (W2-D) adds a reduction that
perturbs the lazy evaluation graph. Measured drift ~0.004 nats on main;
winners stay stable. coder1's W3-A parity suite imports the same constant
after W2-D merges. The FakeModel path stays exact (deterministic zeros).
"""

# Real-model log_score parity tolerance (nats). See module docstring.
PARITY_ATOL = 1e-2
