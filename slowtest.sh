#!/usr/bin/env bash
# Run the slow (real-model) test suite with a capped worker count.
#
# Slow tests load a real MLX model; running many in parallel OOMs Apple
# Silicon's unified memory. This runner caps to 2 workers (set
# SLOWTEST_WORKERS to override) and uses a fresh cache provider.
#
# MODEL_ID is NOT set here — each slow test that needs a specific model
# pins it (e.g. test_issue105_prior_order pins the 1.5B). Tests that are
# model-agnostic (mechanical / deterministic identity checks) inherit the
# conftest default (0.5B).
set -euo pipefail

WORKERS="${SLOWTEST_WORKERS:-1}"

# Find the venv: the worktree's own .venv, else the main repo's .venv.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PY="$SCRIPT_DIR/.venv/bin/python"
elif [[ -x "$SCRIPT_DIR/../jevmlx/.venv/bin/python" ]]; then
    PY="$SCRIPT_DIR/../jevmlx/.venv/bin/python"
else
    PY="python"
fi

# Slow tests load a real MLX model into unified memory; run serially by
# default (one model in memory at a time). pytest-xdist is not installed,
# so -n is omitted — set SLOWTEST_WORKERS only if xdist is added later.
exec "$PY" -m pytest \
    -m slow \
    -p no:cacheprovider \
    -o "addopts=" \
    "$@" tests/
