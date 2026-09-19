"""Question -> schema/label conversions shared by the dataset fetchers.

The TypeSafe published task and its Hugging Face mirror
(``LocalLLaMA/typed-decisions``) use the same question shape — the same
four workflows, the same ``noul``/``choice``/``score`` question types, the
same 0-3 score scale and the same consensus-distribution ambiguity rule —
so one module owns the mapping and both fetchers import it.
"""

from __future__ import annotations

import json

# The workflows TypeSafe publishes today (mirrored in the HF dataset).
WORKFLOWS = (
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
)

# score questions are answered on a fixed 0-3 scale (the criteria list has
# one description per level).
SCORE_CHOICES = ("0", "1", "2", "3")

# Consensus distributions with top1 - top2 below this are marked ambiguous.
# ONE threshold rule for every fetcher's empty-distribution/margin handling.
AMBIGUOUS_MARGIN = 0.1


def as_obj(value):
    """Nested parquet/viewer columns may be JSON strings; accept dicts too."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def field_schema(question: dict, *, score_choices=None) -> dict | None:
    """Map one question to a jevmlx schema field, or None to skip.

    ``noul`` -> boolean; ``choice`` -> enum over the criteria keys (published
    order); ``score`` -> enum over the score scale with the level texts
    folded into the description, ORDERED (W6-B1: the level order is the
    scale order, so the engine derives ordinal telemetry). The scale is
    SCORE_CHOICES ("0".."3") by default; ``score_choices`` overrides it for
    datasets with a different ordinal range (e.g. SST-5's 0..4) — still ONE
    mapping owner, no local copy. Unknown types (free text) are not
    decidable by the engine and are skipped.
    """
    instructions = question["instructions"]
    criteria = question.get("criteria")
    qtype = question["type"]
    if qtype == "noul":
        return {"type": "boolean", "description": instructions}
    if qtype == "choice":
        return {"type": "enum", "description": instructions, "choices": list(criteria)}
    if qtype == "score":
        levels = "; ".join(f"{i} = {text}" for i, text in enumerate(criteria))
        return {
            "type": "enum",
            "description": f"{instructions} Scale: {levels}.",
            "choices": list(score_choices or SCORE_CHOICES),
            # W6-B1: a score question IS an ordinal scale — the level order
            # is the scale order, so the engine derives ordinal telemetry.
            "ordered": True,
        }
    return None


def label_form(gold_label, qtype: str):
    """A gold label in jevmlx label form (booleans become real bools).

    ``None`` when the answer carries no label (skipped by the caller).
    """
    if gold_label is None:
        return None
    if qtype == "noul":
        if isinstance(gold_label, bool):
            return gold_label
        return str(gold_label).lower() == "true"
    return str(gold_label)


def margin_of(distribution: dict[str, float]) -> float:
    """top1 - top2 of a consensus distribution (0.0 when empty)."""
    probs = sorted((float(p) for p in distribution.values()), reverse=True)
    if not probs:
        return 0.0
    return probs[0] - (probs[1] if len(probs) > 1 else 0.0)
