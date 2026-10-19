"""Issue #105 / P12: the prior is measured under the same presentation as
the evidence.

History: issue #105 reported order-dependent decisions under prior
correction; the first fix canonicalized (sorted) the schema for the prior
pass, sharing one prior across orderings. P12 (F8) replaces that: sorting
made the prior measure "dog under a different alias" than the evidence
scored, so the subtraction did not cancel position bias. The prior is now
measured under EXACTLY the evidence presentation (same schema, same choice
order, same alias per choice; only the context is neutral), and each
presentation gets its own cache entry.

The 6-case grid (3 orderings, labels/slots, prior on/off) mirrors the
contributor's public repro (github.com/imaddde867/jev-position-test) and
runs under the slow marker (loads a real model). The fake-engine tests
here are the root-cause tests and run offline.

Usage:
    MODEL_ID=mlx-community/Qwen2.5-1.5B-Instruct-4bit \
        .venv/bin/python -m pytest tests/test_issue105_prior_order.py -s -m slow
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
from conftest import FakeModel
from pydantic import BaseModel, Field

import jevmlx

# Issue #105: the order-invariance thresholds in this file were measured on
# the 1.5B. The test NAME says _on_15b, so the test must PIN the 1.5B
# explicitly — never silently inherit conftest.MODEL_ID (which defaults to
# the 0.5B and flips the scam case to 'low', failing the <= 1 flip gate).
# Anyone running -m slow without MODEL_ID set still gets the 1.5B here.
ISSUE105_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# A4/O2: the slow-on-main CI job caches ONLY the 0.5B (it must not download
# the 1.5B on every merge). When JEVMLX_CI_SLOW=1 is set, these 1.5B-pinned
# tests skip cleanly rather than attempting a download.
_skip_in_ci = pytest.mark.skipif(
    os.environ.get("JEVMLX_CI_SLOW") == "1",
    reason="JEVMLX_CI_SLOW=1: CI caches only the 0.5B; this test pins the 1.5B",
)

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
                model=ISSUE105_MODEL,
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


# ----------------------------- P12: prior under the evidence presentation --


class _AliasBiasModel(FakeModel):
    """FakeModel +2.0 logits at the alias-'A' token (the fake tokenizer's
    ord(c) % 60 id): in slots mode the row candidate for the FIRST alias
    scores higher, so prior numbers depend on which alias a choice sits at
    (the presentation)."""

    def __call__(self, tokens, cache=None):
        out = super().__call__(tokens, cache)
        return out.at[:, :, ord("A") % 60].add(2.0)


NEUTRAL = "(no context provided)"


def _enum_schema(choices):
    from jevmlx.schema import StructuredSchema

    return StructuredSchema(
        {"risk": {"type": "enum", "description": "tier", "choices": list(choices)}}
    )


def test_T1_prior_measured_under_evidence_presentation_slots():
    """P12 T1 (the F8 probe, slots): the prior for a choice is measured
    under the EXACT presentation the evidence used — same schema object,
    same alias per choice. Evidence ["dog", "cat"] binds dog to alias 'A';
    the prior must measure dog under alias 'A'. Main canonicalized the
    schema first, measuring dog under its SORTED-position alias 'B'."""
    from conftest import make_engine

    from jevmlx.engine import _get_or_compute_prior, run_parallel_generation

    schema = _enum_schema(["dog", "cat"])
    engine = make_engine(_AliasBiasModel())
    prior = _get_or_compute_prior(engine, schema, "slots", None, NEUTRAL)
    # The expected numbers: one manual neutral pass over the SAME schema —
    # the evidence presentation.
    manual = run_parallel_generation(
        engine, NEUTRAL, schema, temperature=1.0, scoring="slots", _prior_mode=True
    )
    assert prior["risk"]["log_scores"] == manual["field_telemetry"]["risk"]["log_scores"]


def test_T2_rotated_presentation_gets_own_prior_entry():
    """P12 T2: each rotation is its own presentation — its own prior cache
    entry, computed once, cached. Main shared ONE canonical entry across
    orderings (with the sorted presentation's numbers)."""
    from conftest import make_engine

    from jevmlx.engine import _PRIOR_CACHE, _get_or_compute_prior

    engine = make_engine(_AliasBiasModel())
    _PRIOR_CACHE.clear()
    priors = {}
    for order in [("dog", "cat"), ("cat", "dog")]:
        schema = _enum_schema(order)
        priors[order] = _get_or_compute_prior(engine, schema, "slots", None, NEUTRAL)["risk"][
            "log_scores"
        ]
    # One entry per presentation, not one per choice set.
    assert len(_PRIOR_CACHE) == 2
    # The alias-position bias makes the per-presentation priors differ.
    assert priors[("dog", "cat")] != priors[("cat", "dog")]


def test_T3_apply_prior_missing_choice_raises():
    """P12 T3: a prior missing a choice raises, naming the field and the
    choice — never a silent correction by an implicit 0.0 (main subtracted
    prior_scores.get(c, 0.0))."""
    from jevmlx.engine import _apply_prior

    with pytest.raises(ValueError, match="'risk'.*'dog'"):
        _apply_prior([-0.1, -2.0], ["dog", "cat"], {"log_scores": {"cat": -0.5}}, "risk")


def test_T4_neutral_prompt_lists_choices_in_evidence_order():
    """P12 T4 (labels): the neutral pass runs on the EVIDENCE schema, so
    the neutral prompt lists the choices in the evidence order (the spy
    captures the schema the neutral pass actually used). Main passed a
    sorted canonical schema."""
    from conftest import make_engine

    import jevmlx.engine as eng
    from jevmlx.engine import _get_or_compute_prior

    captured = []
    real = eng.run_parallel_generation

    def spy(*a, **k):
        captured.append(a[2] if len(a) > 2 else k["schema"])
        return real(*a, **k)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(eng, "run_parallel_generation", spy)
    try:
        engine = make_engine()
        schema = _enum_schema(["dog", "cat"])
        _get_or_compute_prior(engine, schema, "labels", None, NEUTRAL)
    finally:
        monkeypatch.undo()
    assert captured, "the neutral pass must run"
    # The neutral pass used the EVIDENCE schema — same object, same order.
    assert captured[0] is schema
    assert captured[0].fields["risk"].choices == ("dog", "cat")


# ------------------------------------------------------------- slow grid ---


@pytest.mark.slow
@_skip_in_ci
def test_issue105_reproduce_table(capsys):
    """Reproduce the issue #105 4-row table on the current main."""
    print(f"\nmodel: {ISSUE105_MODEL}")
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
    # test (per-presentation priors) is separate.
    assert True


@pytest.mark.slow
@_skip_in_ci
def test_prior_corrected_labels_order_stability_on_15b():
    """Issue #105 slow, P12 mechanism: with prior_correction=True and
    labels scoring, order-dependent flips on the 1.5B stay at the
    pre-P12 level or better. The MECHANISM changed (per-presentation
    priors instead of a shared canonical prior — the prior now cancels the
    model's position bias exactly for each ordering), but the promise is
    the same: prior correction must not make order flips WORSE.

    Pinned to ``ISSUE105_MODEL`` (mlx-community/Qwen2.5-1.5B-Instruct-4bit),
    NOT conftest.MODEL_ID — the flip threshold was measured on the 1.5B.
    The 0.5B default flips the scam case to 'low' (3/6) and fails this
    gate; that is a model-capacity difference, not a regression. Run with::

        MODEL_ID=mlx-community/Qwen2.5-1.5B-Instruct-4bit \
            .venv/bin/python -m pytest tests/test_issue105_prior_order.py \
            -m slow -p no:cacheprovider -o "addopts="

    Under the canonical-prior fix: at most 1/6 flips (residual
    evidence-pass position sensitivity on a thin-content case). The
    fake-engine tests above assert the presentation rule; this slow test
    guards the flip gate.
    """
    results = _run_grid("labels", prior_correction=True)
    flips = _count_flips(results)
    print(f"\nlabels + prior_correction flips: {flips}/6")
    for case_name, by_order in results.items():
        print(f"  {case_name}: {by_order}")
    assert flips <= 1, (
        f"prior_correction flipped {flips}/6 cases across orderings (expected <= 1): {results}"
    )
