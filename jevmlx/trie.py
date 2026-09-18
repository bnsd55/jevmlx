"""Pure token-trie construction and scoring for parallel constrained decisions.

No MLX here: the trie and the scoring math are plain Python so they can be
unit-tested without a model (see tests/test_trie.py).

The probability model is the *constrained path probability*: the probability
that masked greedy-per-branch decoding yields a choice, i.e. the product over
the choice's branch points of the locally-masked next-token distribution. It
is a proper distribution over the schema's choices, but it is NOT a normalized
full-sequence likelihood — unary-token evidence and full-vocabulary
normalizers outside the allowed continuations are discarded.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def build_trie(remainders: list[list[int]]) -> list[dict]:
    """Build the branch-point list for one field's choice token remainders.

    Each remainder is the token continuation that distinguishes one choice
    after the field's shared token prefix. A node is a *branch point* when two
    or more choices diverge there (>= 2 distinct next tokens). The rows for a
    field are exactly its branch points: one row per node, holding
    ``shared_ids + path``.

    Returns a list of branch nodes in trie order, each::

        {"path": [token ids from the remainder start to this node, inclusive],
         "children": {token_id: [choice indices reaching that child]}}

    ``children`` is ordered by ascending token id so scoring is deterministic.
    Choices whose remainders never branch keep log-probability 0 (uniquely
    determined once their field's branch factors are applied). A strict-prefix
    remainder (one that is a prefix of another) can never be scored this way
    and is rejected at plan-compile time.
    """
    root: dict = {"children": {}, "choices": []}
    for choice_index, remainder in enumerate(remainders):
        root["choices"].append(choice_index)
        node = root
        for token in remainder:
            node = node["children"].setdefault(token, {"children": {}, "choices": []})
            node["choices"].append(choice_index)

    branch_nodes: list[dict] = []

    # Iterative pre-order walk with an explicit stack (N1: a long remainder
    # must not depend on the interpreter recursion limit). Children are
    # pushed in REVERSE sorted order so the pop order — and therefore the
    # branch_nodes output order — matches the old recursive pre-order walk
    # exactly.
    stack: list[tuple[dict, list[int]]] = [(root, [])]
    while stack:
        node, path = stack.pop()
        if len(node["children"]) >= 2:
            branch_nodes.append(
                {
                    "path": list(path),
                    "children": {
                        token: node["children"][token]["choices"]
                        for token in sorted(node["children"])
                    },
                }
            )
        for token in sorted(node["children"], reverse=True):
            stack.append((node["children"][token], [*path, token]))

    return branch_nodes


def _validate_finite(values: list[float]) -> None:
    """Raise ValueError if any value is NaN or infinite."""
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"non-finite value in logits: {values!r}")


def logsumexp(values: list[float]) -> float:
    """Numerically stable log-sum-exp over finite values.

    Computed from the shifted values (s = v - max) so huge-but-finite
    magnitudes cannot cancel (C1).
    """
    _validate_finite(values)
    largest = max(values)
    shifted = [v - largest for v in values]
    return largest + math.log(sum(math.exp(s) for s in shifted))


def log_softmax(values: list[float]) -> list[float]:
    """Numerically stable log-softmax over finite values.

    Computed directly from the shifted logits (s = v - max) as
    s - log(sum(exp(s))): no cancellation between huge values and the
    log-sum-exp (C1). [1e300, 1e300] -> [-ln2, -ln2].
    """
    _validate_finite(values)
    largest = max(values)
    shifted = [v - largest for v in values]
    lse_shifted = math.log(sum(math.exp(s) for s in shifted))
    return [s - lse_shifted for s in shifted]


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over finite values.

    Scales BEFORE the max shift ((v / T) - max(v / T)), so the exponent
    argument never exceeds 0: no overflow for any finite logits and any
    finite positive temperature. Temperature must be finite and > 0.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    _validate_finite(values)
    largest = max(values)
    # Shift BEFORE scaling ((v - max) / T): finite values stay finite for any
    # finite positive T, however small (B3).
    exps = [math.exp((v - largest) / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def score_trie(
    branch_nodes: list[dict],
    n_choices: int,
    logits_at_node: Callable[[dict], list[float]],
    legal_mass_at_node: Callable[[dict], float] | None = None,
) -> tuple[list[float], list[float]]:
    """Natural-log constrained-path probability per choice, at temperature 1.

    ``logits_at_node(node)`` must return the child logits in the same order as
    ``list(node["children"])``. At each branch node the children's full-vocab
    logits are log-softmaxed over the allowed continuations and every choice
    under a child accumulates that child's log-probability. Temperature is NOT
    applied here: callers score at T=1 and apply their confidence temperature
    once to the final per-choice scores (see calibrate.py), so ranking is
    invariant to it.

    This is the constrained path probability — the probability that masked
    greedy-per-branch decoding yields the choice — not a normalized
    full-sequence likelihood. It is a proper distribution over the choices:
    the probabilities exp(scores) sum to 1.

    Choices with no branch nodes on their path score log P = 0.0: their value
    is fully determined by the field's other branch decisions. A field with a
    single choice therefore scores log P = 0 (probability 1.0) with no rows.

    When ``legal_mass_at_node`` is provided, also returns the per-choice
    legal-mass product. The legal mass at a branch node is the probability
    the model assigns to the union of allowed continuations, against the full
    vocabulary: ``sum(exp(z_allowed)) / sum(exp(z_vocab))``. It answers "did
    the model want *any* valid code here?" — a branch can confidently pick A
    over B even when almost all unconstrained mass is on a reasoning token,
    newline, or label text. Legal mass is that leakage signal. The callback
    returns the per-node mass float (the caller computes it from the
    full-vocab logits it holds; the trie is MLX-free). None (default) leaves
    every legal mass at 1.0 — the MLX-free unit tests use this.

    Returns ``(log_probs, legal_mass_logs)``: the natural-log constrained-path
    probability per choice and the natural log of the per-choice legal-mass
    product along the branch path. A choice with no branch nodes scores
    ``log legal_mass = 0.0`` (mass 1.0): nothing branched, so there was
    nowhere to leak.
    """
    log_probs = [0.0] * n_choices
    legal_mass_logs = [0.0] * n_choices
    for node in branch_nodes:
        child_logits = logits_at_node(node)
        if not all(math.isfinite(value) for value in child_logits):
            raise ValueError(f"non-finite logits at branch node {node['path']!r}: {child_logits!r}")
        child_log_probs = log_softmax(child_logits)
        node_legal_mass_log = (
            math.log(legal_mass_at_node(node)) if legal_mass_at_node is not None else 0.0
        )
        for token, log_prob in zip(node["children"], child_log_probs, strict=True):
            for choice_index in node["children"][token]:
                log_probs[choice_index] += log_prob
                legal_mass_logs[choice_index] += node_legal_mass_log
    return log_probs, legal_mass_logs
