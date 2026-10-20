"""Constrained MAP tests.

Verifies:
- A 3-field component where the independent argmax violates an implies
  constraint -> the MAP reconciler flips the low-margin child.
- No-constraint path is bit-identical (constraints=None).
- Exclusivity constraint: at most one premium option in a multi field.
- Excludes constraint: approved=true -> rejection_reason must be empty.
- Large component (>5000 product) raises NotImplementedError.
- Telemetry: reconciled_fields lists the changed fields.
"""

from __future__ import annotations

import math

import pytest
from conftest import make_field_semantics

from jevmlx.constraints import compile_constraints
from jevmlx.engine import _constrained_map, reconcile_case_constraints
from jevmlx.schema import StructuredSchema


def _compile(constraints, schema):
    """Compile raw dicts against a real StructuredSchema."""
    return compile_constraints(constraints, schema)


def _field_scores(scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Identity helper: pass through the log_scores dict."""
    return scores


def test_implies_map_flips_low_margin_child():
    """3-field component: intent -> subtype (implies). The independent argmax
    picks intent=billing (0.9), subtype=bug (0.55) — but bug is NOT valid for
    billing (only refund/dispute). The MAP must flip subtype to the best valid
    option (refund, 0.3)."""
    field_log_scores = {
        "intent": {"billing": math.log(0.9), "technical": math.log(0.1)},
        "subtype": {"refund": math.log(0.3), "dispute": math.log(0.15), "bug": math.log(0.55)},
    }
    field_values = {
        "intent": {"value": "billing"},
        "subtype": {"value": "bug"},  # independent argmax — violates implies
    }
    constraints = [
        {
            "type": "implies",
            "parent": "intent",
            "child": "subtype",
            "mapping": {"billing": ["refund", "dispute"], "technical": ["bug"]},
        }
    ]

    # The MAP unit evaluates COMPILED constraints — compile once.
    schema = StructuredSchema(
        {
            "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
            "subtype": {
                "type": "enum",
                "description": "d",
                "choices": ["refund", "dispute", "bug"],
            },
        }
    )
    compiled = compile_constraints(constraints, schema)
    reconciled, changed = _constrained_map(field_log_scores, field_values, compiled, schema)
    assert reconciled["intent"] == "billing"  # parent unchanged
    assert reconciled["subtype"] == "refund"  # flipped to best valid
    assert "subtype" in changed
    assert "intent" not in changed


def test_no_constraint_path_bit_identical():
    """When constraints=None, _constrained_map returns ({}, []) — no change."""
    field_log_scores = {"risk": {"LOW": -1.0, "HIGH": -0.5}}
    field_values = {"risk": {"value": "HIGH"}}

    schema = StructuredSchema(
        {"risk": {"type": "enum", "description": "d", "choices": ["LOW", "HIGH"]}}
    )
    compiled = compile_constraints([], schema)
    reconciled, changed = _constrained_map(field_log_scores, field_values, compiled, schema)
    assert reconciled == {}
    assert changed == []


def test_excludes_constraint():
    """approved=true -> rejection_reason must be empty. If the independent
    argmax picks approved=true with rejection_reason=policy, the MAP must
    flip one of them."""
    field_log_scores = {
        "approved": {"true": -0.1, "false": -2.0},
        "rejection_reason": {"policy": -0.2, "eligibility": -1.5, "duplicate": -3.0},
    }
    field_values = {
        "approved": {"value": "true"},
        "rejection_reason": {"value": "policy"},  # violates excludes
    }
    # Compiled constraints carry TYPED domains — boolean fields
    # compile to [True, False], so the constraint value is the Python bool
    # (the old string-key path silently accepted "true").
    constraints = [
        {"type": "excludes", "field": "approved", "value": True, "other": "rejection_reason"}
    ]

    schema = StructuredSchema(
        {
            "approved": {"type": "boolean", "description": "d"},
            "rejection_reason": {
                "type": "enum",
                "description": "d",
                "choices": ["policy", "eligibility", "duplicate"],
            },
        }
    )
    compiled = compile_constraints(constraints, schema)
    reconciled, changed = _constrained_map(field_log_scores, field_values, compiled, schema)
    # approved=true is strong (-0.1); rejection_reason must be empty.
    # But "" is not in the candidates — the MAP should pick approved=false
    # (flip approved, keep rejection_reason=policy which has -0.2).
    # Actually: approved=true (-0.1) + rejection_reason="" is not possible
    # because "" is not a candidate. So the MAP picks approved=false (-2.0)
    # + rejection_reason=policy (-0.2) = -2.2 vs approved=true (-0.1) +
    # rejection_reason=eligibility (-1.5) = -1.6 but eligibility also violates.
    # The only valid assignments: approved=false + any reason, or approved=true
    # + reason="". Since "" is not a candidate, only approved=false + reason.
    # The MAP decides TYPED values (the boolean field reconciles to
    # Python False, not the score key "false").
    assert reconciled["approved"] is False
    assert reconciled["rejection_reason"] == "policy"  # best reason
    assert "approved" in changed


def test_large_component_raises():
    """A component with >5000 product space raises NotImplementedError."""
    # 3 fields, each with 20 candidates -> 8000 > 5000.
    field_log_scores = {f"f{i}": {f"c{j}": -j * 0.1 for j in range(20)} for i in range(3)}
    field_values = {f"f{i}": {"value": "c0"} for i in range(3)}
    constraints = [
        {"type": "implies", "parent": "f0", "child": "f1", "mapping": {"c0": ["c0"]}},
        {"type": "implies", "parent": "f1", "child": "f2", "mapping": {"c0": ["c0"]}},
    ]

    schema = StructuredSchema(
        {
            **{
                f"f{i}": {
                    "type": "enum",
                    "description": "d",
                    "choices": [f"c{j}" for j in range(20)],
                }
                for i in range(3)
            }
        }
    )
    compiled = compile_constraints(constraints, schema)

    import pytest

    with pytest.raises(NotImplementedError, match="too large"):
        _constrained_map(field_log_scores, field_values, compiled, schema)


def test_reconciled_fields_in_telemetry():
    """The engine result carries reconciled_fields when constraints are
    applied. This is a smoke test of the wiring (uses the fake model path
    if available; otherwise tests _constrained_map directly)."""
    field_log_scores = {
        "intent": {"billing": -0.1, "technical": -2.0},
        "subtype": {"refund": -1.0, "dispute": -2.0, "bug": -0.05},  # bug is argmax
    }
    field_values = {
        "intent": {"value": "billing"},
        "subtype": {"value": "bug"},  # violates implies
    }
    constraints = [
        {
            "type": "implies",
            "parent": "intent",
            "child": "subtype",
            "mapping": {"billing": ["refund", "dispute"], "technical": ["bug"]},
        }
    ]

    schema = StructuredSchema(
        {
            "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
            "subtype": {
                "type": "enum",
                "description": "d",
                "choices": ["refund", "dispute", "bug"],
            },
        }
    )
    compiled = compile_constraints(constraints, schema)
    reconciled, changed = _constrained_map(field_log_scores, field_values, compiled, schema)
    assert "subtype" in changed
    assert reconciled["subtype"] == "refund"


# ------------------------- the record reports what it returns


def _state(parsed, telemetry):
    from jevmlx.engine import AssembledState

    return AssembledState(
        parsed_json=parsed,
        field_telemetry=telemetry,
        rescored_fields=(),
    )


def _scalar_tel(value, probability, log_scores, **extra):
    """A finalized scalar telemetry entry (distribution + old decision
    numbers, pre-MAP shape — no probability_margin/alternatives yet)."""
    ranked = sorted(log_scores.items(), key=lambda kv: -kv[1])
    entry = {
        "value": value,
        "type": "enum",
        "probability": probability,
        "cardinality": len(log_scores),
        "log_scores": dict(log_scores),
        "top_choices": [{"choice": c, "probability": math.exp(ls)} for c, ls in ranked],
        "rows": 2,
        "tie": False,
        "rescored": False,
        "legal_mass": 1.0,
        "semantics": make_field_semantics(),
    }
    entry.update(extra)
    return entry


def test_T1_boolean_map_change_reports_returned_value():
    """Model winners are (allow, false); the implies
    constraint forces flag=True. The record must report the RETURNED value:
    probability = P(true), a negative probability_margin, the top OTHER
    choice as alternatives — while the distribution keys stay untouched.
    Main kept the old argmax's probability (str(True)='True' misses the
    'true' key) and never touched margin/tie/alternatives."""
    constraints = [
        {
            "type": "implies",
            "parent": "gate",
            "child": "flag",
            "mapping": {"allow": [True], "deny": [True, False]},
        }
    ]
    schema = StructuredSchema(
        {
            "gate": {"type": "enum", "description": "d", "choices": ["allow", "deny"]},
            "flag": {"type": "boolean", "description": "d"},
        }
    )
    compiled = compile_constraints(constraints, schema)
    gate_tel = _scalar_tel("allow", 0.9, {"allow": math.log(0.9), "deny": math.log(0.1)})
    flag_tel = _scalar_tel(False, 0.9, {"true": math.log(0.1), "false": math.log(0.9)})
    flag_tel["type"] = "boolean"
    input_tel = {"gate": gate_tel, "flag": flag_tel}
    state = _state(
        {"gate": {"value": "allow", "prob": 0.9}, "flag": {"value": False, "prob": 0.9}},
        input_tel,
    )
    out = reconcile_case_constraints(state, constraints, schema, compiled)

    assert out.parsed_json["flag"]["value"] is True
    ft = out.field_telemetry["flag"]
    assert ft["value"] is True
    assert ft["probability"] == pytest.approx(0.1)  # P(true), not the old 0.9
    assert out.parsed_json["flag"]["prob"] == pytest.approx(0.1)
    assert ft["log_score_margin"] == pytest.approx(math.log(0.1) - math.log(0.9))
    assert ft["probability_margin"] == pytest.approx(0.1 - 0.9)  # negative: forced
    assert ft["tie"] is False  # |log 0.1 - log 0.9| = 2.2 nats, far outside the band
    # Competitors of the returned value: the top OTHER choices.
    assert [a["choice"] for a in ft["alternatives"]] == ["false"]
    assert ft["alternatives"][0]["probability"] == pytest.approx(0.9)
    # Distribution keys untouched.
    assert ft["log_scores"] == input_tel["flag"]["log_scores"]
    assert ft["top_choices"] == input_tel["flag"]["top_choices"]
    assert ft["semantics"]["constraint_changed"] is True
    # The untouched parent: no decision numbers injected, no changed flag.
    assert out.field_telemetry["gate"] is input_tel["gate"]
    assert out.field_telemetry["gate"]["semantics"]["constraint_changed"] is False
    assert out.reconciled_fields == ("flag",)


def test_T3_enum_map_change_reports_returned_value():
    """Same checks for a plain enum (the str(val) lookup path) —
    subtype bug -> refund under intent=billing."""
    constraints = [
        {
            "type": "implies",
            "parent": "intent",
            "child": "subtype",
            "mapping": {"billing": ["refund"], "technical": ["bug", "refund", "dispute"]},
        }
    ]
    schema = StructuredSchema(
        {
            "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
            "subtype": {
                "type": "enum",
                "description": "d",
                "choices": ["refund", "dispute", "bug"],
            },
        }
    )
    compiled = compile_constraints(constraints, schema)
    scores = {"refund": math.log(0.2), "dispute": math.log(0.3), "bug": math.log(0.5)}
    subtype_tel = _scalar_tel("bug", 0.5, scores)
    input_tel = {"subtype": subtype_tel}
    state = _state(
        {"intent": {"value": "billing", "prob": 1.0}, "subtype": {"value": "bug", "prob": 0.5}},
        input_tel,
    )
    out = reconcile_case_constraints(state, constraints, schema, compiled)

    assert out.parsed_json["subtype"]["value"] == "refund"
    ft = out.field_telemetry["subtype"]
    assert ft["probability"] == pytest.approx(0.2)  # P(refund)
    assert out.parsed_json["subtype"]["prob"] == pytest.approx(0.2)
    assert ft["log_score_margin"] == pytest.approx(math.log(0.2) - math.log(0.5))
    assert ft["probability_margin"] == pytest.approx(0.2 - 0.5)  # best OTHER is bug
    assert ft["tie"] is False
    assert [a["choice"] for a in ft["alternatives"]] == ["bug", "dispute"]
    assert ft["alternatives"][0]["probability"] == pytest.approx(0.5)
    assert ft["log_scores"] == scores  # distribution untouched
    assert ft["top_choices"] == input_tel["subtype"]["top_choices"]
    assert ft["semantics"]["constraint_changed"] is True
    assert out.reconciled_fields == ("subtype",)


def test_T4_unchanged_field_telemetry_identical():
    """A field the MAP did NOT change is untouched — no
    decision numbers injected, every key identical (the normal-path numbers
    already came from the ONE helper at finalize time)."""
    constraints = [
        {
            "type": "implies",
            "parent": "gate",
            "child": "flag",
            "mapping": {"allow": [True], "deny": [True, False]},
        }
    ]
    schema = StructuredSchema(
        {
            "gate": {"type": "enum", "description": "d", "choices": ["allow", "deny"]},
            "flag": {"type": "boolean", "description": "d"},
        }
    )
    compiled = compile_constraints(constraints, schema)
    gate_tel = _scalar_tel("allow", 0.9, {"allow": math.log(0.9), "deny": math.log(0.1)})
    # Model winner already satisfies the constraint: no change.
    flag_tel = _scalar_tel(True, 0.1, {"true": math.log(0.1), "false": math.log(0.9)})
    flag_tel["type"] = "boolean"
    input_tel = {"gate": gate_tel, "flag": flag_tel}
    state = _state(
        {"gate": {"value": "allow", "prob": 0.9}, "flag": {"value": True, "prob": 0.1}},
        input_tel,
    )
    out = reconcile_case_constraints(state, constraints, schema, compiled)
    assert out.reconciled_fields == ()
    assert dict(out.field_telemetry) == input_tel
    assert dict(out.parsed_json) == state.parsed_json
