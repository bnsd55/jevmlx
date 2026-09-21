"""Issue #105 reproduction: does prior_correction reintroduce option-order
dependence in labels mode?

6 cases, 1 enum (risk_tier: low/medium/high) with choice_descriptions,
ordered=False, 3 option orderings. Only the order changes between runs.

Runs under the slow marker (loads a real model). Mirrors the contributor's
public repro (github.com/imaddde867/jev-position-test) but is self-contained.

Usage:
    MODEL_ID=mlx-community/Qwen2.5-1.5B-Instruct-4bit \
        .venv/bin/python -m pytest tests/test_issue105_prior_order.py -s -m slow
"""

from __future__ import annotations

from typing import Literal

import pytest
from conftest import MODEL_ID
from pydantic import BaseModel, Field

import jevmlx

# The 6 cases from the issue (paraphrased — same semantic content).
CASES: dict[str, str] = {
    "clear_high_risk_scam": (
        "URGENT: Your account will be suspended in 1 hour. Click this link "
        "and enter your card number and PIN immediately to avoid permanent "
        "loss of funds: bit.ly/verify-now-2847"
    ),
    "benign_smalltalk": (
        "Hey, just checking in -- how was your weekend? We should grab "
        "coffee sometime next week if you're free."
    ),
    "ambiguous_urgent": (
        "I need you to process this wire transfer today, the client is "
        "waiting and I can't get on a call right now, just use the account "
        "details from last time."
    ),
    "technical_neutral": (
        "Sensor 14B reported a temperature reading of 22.4C at 03:00 UTC, "
        "within the configured normal operating range."
    ),
    "mild_concern": (
        "I noticed a $40 charge I don't remember making, probably a "
        "subscription I forgot about, can you check it when you get a "
        "chance?"
    ),
    "empty_string": "",
}

TIER_DESCRIPTIONS = {
    "low": "Low risk: no red flags, safe to proceed as normal.",
    "medium": "Some risk indicators present, needs human review before acting.",
    "high": "Clear risk indicators: likely fraudulent, malicious, or urgent harm.",
}

ORDERS = {
    "low_med_high": ("low", "medium", "high"),
    "high_med_low": ("high", "medium", "low"),
    "med_high_low": ("medium", "high", "low"),
}


def _model_cls(order: tuple[str, str, str]) -> type[BaseModel]:
    lit = Literal[order]  # type: ignore[valid-type]

    class RiskGate(BaseModel):
        is_risky: bool = Field(
            description="Does this message show signs of risk, scam, fraud, or urgent harm?"
        )
        risk_tier: lit = Field(  # type: ignore[valid-type]
            description="What risk tier does this message fall into?",
            json_schema_extra={"choice_descriptions": TIER_DESCRIPTIONS},
        )

    RiskGate.__name__ = f"RiskGate_{'_'.join(order)}"
    return RiskGate


def _run_grid(scoring: str, prior_correction: bool) -> dict[str, dict[str, str]]:
    """Run all 6 cases x 3 orderings; return {case: {order_name: chosen_tier}}."""
    results: dict[str, dict[str, str]] = {}
    for case_name, ctx in CASES.items():
        results[case_name] = {}
        for order_name, order in ORDERS.items():
            d = jevmlx.decide(
                _model_cls(order),
                ctx,
                model=MODEL_ID,
                scoring=scoring,
                prior_correction=prior_correction,
            )
            results[case_name][order_name] = d.fields["risk_tier"].value
    return results


def _count_flips(results: dict[str, dict[str, str]]) -> int:
    """How many cases change their chosen tier across the 3 orderings."""
    flips = 0
    for _case_name, by_order in results.items():
        values = set(by_order.values())
        if len(values) > 1:
            flips += 1
    return flips


def _slot_distribution(results: dict[str, dict[str, str]]) -> dict[int, int]:
    """Count of chosen values landing in slot 1/2/3 (position in the ordering)."""
    dist = {1: 0, 2: 0, 3: 0}
    for by_order in results.values():
        for order_name, order in ORDERS.items():
            chosen = by_order[order_name]
            slot = order.index(chosen) + 1
            dist[slot] += 1
    return dist


@pytest.mark.slow
def test_issue105_reproduce_table(capsys):
    """Reproduce the issue #105 4-row table on the current main."""
    print(f"\nmodel: {MODEL_ID}")
    print(f"jevmlx: {jevmlx.__version__}\n")

    for scoring, prior in [
        ("labels", False),
        ("labels", True),
        ("slots", False),
        ("slots", True),
    ]:
        results = _run_grid(scoring, prior)
        flips = _count_flips(results)
        dist = _slot_distribution(results)
        label = f"{'labels' if scoring == 'labels' else 'slots'}"
        if prior:
            label += " + prior_correction"
        print(f"| {label:<30} | {flips}/6 | {dist[1]}/{dist[2]}/{dist[3]} |")

    # The assertion is informational — we print the table. The root cause
    # test (order-invariant prior) is separate.
    assert True


def test_canonicalized_for_prior_sorts_enum_choices():
    """StructuredSchema.canonicalized_for_prior returns a schema with enum
    choices sorted; booleans/multi unaffected; no-op when already sorted."""
    from jevmlx.schema import StructuredSchema

    schema = StructuredSchema(
        {
            "risk_tier": {
                "name": "risk_tier",
                "type": "enum",
                "description": "tier",
                "choices": ["high", "low", "medium"],
                "choice_descriptions": {
                    "low": "low",
                    "medium": "med",
                    "high": "high",
                },
            }
        }
    )
    canon = schema.canonicalized_for_prior()
    assert canon.fields["risk_tier"].choices == ("high", "low", "medium")
    # Already sorted -> no-op (returns self).
    assert canon.canonicalized_for_prior() is canon
    # Boolean field is untouched.
    bool_schema = StructuredSchema(
        {"flag": {"name": "flag", "type": "boolean", "description": "f"}}
    )
    assert bool_schema.canonicalized_for_prior() is bool_schema


def test_prior_is_order_invariant_on_fake_engine(monkeypatch):
    """Issue #105 fix: the prior computed by _get_or_compute_prior is
    IDENTICAL for all orderings of the same choice set (the prior cache
    has one entry, applied by choice name). Tested on the fake engine so
    no model load is needed."""
    from conftest import make_engine

    from jevmlx.api import _prepare_schema
    from jevmlx.engine import _PRIOR_CACHE, _get_or_compute_prior

    engine = make_engine()
    _PRIOR_CACHE.clear()
    NEUTRAL = ""
    priors = {}
    for order_name, order in ORDERS.items():
        schema = _prepare_schema(_model_cls(order), allow_none_of_above=False)
        prior = _get_or_compute_prior(engine, schema, "labels", None, NEUTRAL)
        priors[order_name] = prior["risk_tier"]["log_scores"]
    # All three orderings produce the SAME prior (by choice name).
    assert priors["low_med_high"] == priors["high_med_low"] == priors["med_high_low"], (
        f"prior differs across orderings: {priors}"
    )
    # One cache entry for all three orderings.
    assert len(_PRIOR_CACHE) == 1


@pytest.mark.slow
def test_prior_corrected_labels_order_invariant_on_15b():
    """Issue #105 slow: with prior_correction=True and labels scoring, the
    canonical-prior fix reduces order-dependent flips on the 1.5B.

    Before the fix: 2/6 cases flipped (the prior itself was order-dependent).
    After: at most 1/6 (residual evidence-pass position sensitivity on a
    thin-content case — the fix removed the PRIOR's order dependence, not
    the model's inherent evidence-position sensitivity, which is a separate,
    harder problem). The fake-engine test asserts exact invariance; this
    slow test guards the improvement does not regress past 1.
    """
    results = _run_grid("labels", prior_correction=True)
    flips = _count_flips(results)
    print(f"\nlabels + prior_correction flips: {flips}/6")
    for case_name, by_order in results.items():
        print(f"  {case_name}: {by_order}")
    assert flips <= 1, (
        f"prior_correction flipped {flips}/6 cases across orderings "
        f"(expected <= 1 after the canonical-prior fix): {results}"
    )
