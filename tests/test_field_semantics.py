"""W5b-13 step 1: FieldSemantics contract tests (design-note C9).

The dataclass lives on the public API NOW; the ENGINE fills it in step 2
(set in the #50 stages: score_scalar_field / score_multi_field /
reconcile_case_constraints / run_dependency_waves / finalize_public_result).
Until step 2 lands, decide() cannot supply semantics — the end-to-end
pinning test is xfail(strict=True) so it fails loudly today and flips the
suite green exactly when step 2 removes the mark.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import jevmlx
from jevmlx.api import FieldResult, FieldSemantics

# --- the dataclass contract ----------------------------------------------------


def test_field_semantics_frozen_and_field_names():
    """Frozen dataclass; the six C9 field names in the design order."""
    s = FieldSemantics(
        score_source="batched",
        temperature=1.0,
        calibrator_id=None,
        prior_mode="off",
        constraint_changed=False,
        dependency_rescored=False,
    )
    assert dataclasses.is_dataclass(s)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.temperature = 0.7  # type: ignore[misc]
    assert [f.name for f in dataclasses.fields(s)] == [
        "score_source",
        "temperature",
        "calibrator_id",
        "prior_mode",
        "constraint_changed",
        "dependency_rescored",
    ]


def test_field_result_requires_semantics():
    """FieldResult.semantics is REQUIRED (keyword-only, no default): a
    construction without it is a TypeError, never a silent default."""
    kwargs = dict(
        value="A",
        score=0.0,
        log_score_margin=None,
        probability_margin=None,
        threshold_distance=None,
        probability=0.9,
        calibrated=False,
        model="slots",
        alternatives=(),
    )
    with pytest.raises(TypeError, match="semantics"):
        FieldResult(**kwargs)
    s = FieldSemantics("batched", 1.0, None, "off", False, False)
    fr = FieldResult(**kwargs, semantics=s)
    assert fr.semantics is s


def test_field_semantics_exported():
    """Public surface: importable from jevmlx.api, listed in __all__.

    Both bindings come from ONE import path inside the test: the
    test_check_results module eviction can leave this file's collection-time
    binding on a stale module instance, and cross-instance identity would
    fail for the wrong reason (see test_bench_machine_tag_patches_live_module)."""
    import jevmlx.api as api
    from jevmlx.api import FieldSemantics as FS  # noqa: N806 — same-path binding

    assert api.FieldSemantics is FS
    assert "FieldSemantics" in api.__all__


# --- engine-filling pinning test (step 2) --------------------------------------


def _run_pinned_decide(monkeypatch):
    """decide() over the shared fake engine with one scalar field: the
    minimal path whose FieldResult must carry a complete semantics record
    once the engine sets it (W5b-13 step 2)."""
    import jevmlx.api as api
    from tests.conftest import make_engine_result, make_field_telemetry

    monkeypatch.setattr(
        api,
        "run_parallel_generation",
        lambda *a, **k: make_engine_result(
            fields={
                "risk_tier": make_field_telemetry(
                    value="HIGH",
                    choices=["HIGH", "LOW", "CRITICAL"],
                )
            },
        ),
    )
    monkeypatch.setattr(api, "load_engine", lambda model_id: ("engine", "tokenizer"))

    class Ticket(BaseModel):
        risk_tier: Literal["HIGH", "LOW", "CRITICAL"] = Field(description="Risk tier")

    return jevmlx.decide(Ticket, "ctx", model="fake/model")


@pytest.mark.xfail(
    strict=True,
    reason="W5b-13 step 2: the engine does not set FieldResult.semantics yet "
    "(lands after the '#51 merged' ping; #50 stages set it per the design table)",
)
def test_engine_fills_semantics_per_field(monkeypatch):
    """PIN (step 2): the engine's decided fields carry per-field semantics —
    scalar path: batched evidence, the caller temperature, no calibrator,
    prior off, no constraint change, no dependency rescore."""
    d = _run_pinned_decide(monkeypatch)
    fr = d.fields["risk_tier"]
    assert isinstance(fr.semantics, FieldSemantics)
    assert fr.semantics == FieldSemantics(
        score_source="batched",
        temperature=1.0,
        calibrator_id=None,
        prior_mode="off",
        constraint_changed=False,
        dependency_rescored=False,
    )


@pytest.mark.xfail(
    strict=True,
    reason="W5b-13 step 2: probability_status stays the old global string until "
    "the stages set semantics and finalize_public_result builds the summary",
)
def test_probability_status_summarizes_semantics_groups(monkeypatch):
    """PIN (step 2): the result-level probability_status is a SUMMARY over the
    distinct (score_source, temperature, calibrator_id, prior_mode) groups —
    it names the count of fields per group and never a per-field claim."""
    d = _run_pinned_decide(monkeypatch)
    status = d.fields["risk_tier"].semantics  # placeholder to force attr use
    result_status = _run_result_status(monkeypatch)
    assert isinstance(status, FieldSemantics)
    # One distinct group (scalar, T=1, uncalibrated, prior off) -> the
    # summary must mention that group, with no per-field value claims.
    assert "1 field" in result_status
    assert "batched" in result_status
    assert "T=1" in result_status


def _run_result_status(monkeypatch) -> str:
    """The raw engine result's probability_status through the fake path."""
    import jevmlx.api as api
    from tests.conftest import make_engine_result, make_field_telemetry

    res = make_engine_result(
        fields={
            "risk_tier": make_field_telemetry(
                value="HIGH",
                choices=["HIGH", "LOW", "CRITICAL"],
            )
        },
    )
    monkeypatch.setattr(api, "run_parallel_generation", lambda *a, **k: res)
    monkeypatch.setattr(api, "load_engine", lambda model_id: ("engine", "tokenizer"))

    from typing import Literal as L

    from pydantic import BaseModel

    class Ticket(BaseModel):
        risk_tier: L["HIGH", "LOW", "CRITICAL"] = Field(description="Risk tier")

    d = jevmlx.decide(Ticket, "ctx", model="fake/model")
    assert d.fields["risk_tier"].semantics is not None  # step-2 gate
    return res["probability_status"]
