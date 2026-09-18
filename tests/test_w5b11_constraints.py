"""W5b-11 tests: compiled constraint objects (GPT-REVIEW-2 C6).

- compile_constraints is the ONLY entry: unknown fields/values/types and
  multi-field violations all fail at compile time, before any model work.
- The MAP on the compiled form picks IDENTICAL reconciliations to the old
  dict-walking path on the dependent fixture (intent=billing forces
  subtype into {refund, dispute}; independent argmax bug must flip).
- evalmetrics._check_constraint accepts compiled objects.
"""

import dataclasses
import math

import pytest

from jevmlx.constraints import (
    CompiledImplication,
    ConstraintError,
    compile_constraints,
)
from jevmlx.engine import _constrained_map
from jevmlx.schema import StructuredSchema


def _schema():
    return StructuredSchema(
        {
            "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
            "subtype": {
                "type": "enum",
                "description": "d",
                "choices": ["refund", "dispute", "bug"],
            },
        }
    )


_IMPLIES = {
    "type": "implies",
    "parent": "intent",
    "child": "subtype",
    "mapping": {"billing": ["refund", "dispute"], "technical": ["bug"]},
}


class TestCompileRejects:
    def test_unknown_field(self):
        with pytest.raises(ConstraintError, match="unknown field 'intent2'"):
            compile_constraints(
                [{**_IMPLIES, "parent": "intent2"}],
                _schema(),
            )

    def test_unknown_parent_value(self):
        with pytest.raises(ConstraintError, match="mapping key 'enterprise'"):
            compile_constraints(
                [{**_IMPLIES, "mapping": {"enterprise": ["refund"]}}],
                _schema(),
            )

    def test_unknown_child_value(self):
        with pytest.raises(ConstraintError, match="mapped child value 'escalate'"):
            compile_constraints(
                [{**_IMPLIES, "mapping": {"billing": ["escalate"]}}],
                _schema(),
            )

    def test_unknown_type(self):
        with pytest.raises(ConstraintError, match="type must be one of"):
            compile_constraints(
                [{"type": "xor", "field": "intent", "value": "billing"}],
                _schema(),
            )

    def test_multi_field_case_constraint_rejected(self):
        schema = StructuredSchema(
            {
                "tags": {"type": "multi", "description": "d", "choices": ["a", "b"]},
                "flag": {"type": "boolean", "description": "d"},
            }
        )
        with pytest.raises(ConstraintError, match="multi field"):
            compile_constraints(
                [{"type": "implies", "parent": "tags", "child": "flag", "mapping": {"a": [True]}}],
                schema,
            )


class TestCompiledShape:
    def test_frozen_typed_objects(self):
        compiled = compile_constraints([_IMPLIES], _schema())
        assert isinstance(compiled.implications[0], CompiledImplication)
        with pytest.raises(dataclasses.FrozenInstanceError):
            compiled.implications[0].parent_idx = 9  # type: ignore[misc]

    def test_index_space(self):
        """The compiled object carries FIELD INDICES and value-index sets —
        no string keys in the hot path."""
        compiled = compile_constraints([_IMPLIES], _schema())
        c = compiled.implications[0]
        assert c.parent_idx == 0  # 'intent' is schema field 0
        assert c.child_idx == 1  # 'subtype' is field 1
        assert c.mapping == {0: frozenset({0, 1}), 1: frozenset({2})}
        assert compiled.domain_of[0] == ("billing", "technical")
        assert compiled.domain_of[1] == ("refund", "dispute", "bug")

    def test_components_precomputed(self):
        compiled = compile_constraints([_IMPLIES], _schema())
        assert len(compiled.components) == 1
        assert compiled.components[0].field_idxes == frozenset({0, 1})

    def test_satisfied_typed_values(self):
        compiled = compile_constraints([_IMPLIES], _schema())
        assert compiled.satisfied({"intent": "billing", "subtype": "refund"})
        assert not compiled.satisfied({"intent": "billing", "subtype": "bug"})
        # Unmeasured fields cannot violate.
        assert compiled.satisfied({"intent": "billing"})
        assert compiled.satisfied({})


class TestMapIdenticalOnDependentFixture:
    def test_map_result_identical_to_dict_path(self):
        """The compiled MAP picks the same joint assignment the dict-walking
        path picked on the dependent fixture: subtype flips bug -> refund."""
        field_log_scores = {
            "intent": {"billing": math.log(0.9), "technical": math.log(0.1)},
            "subtype": {"refund": math.log(0.3), "dispute": math.log(0.15), "bug": math.log(0.55)},
        }
        field_values = {"intent": {"value": "billing"}, "subtype": {"value": "bug"}}
        schema = _schema()
        compiled = compile_constraints([_IMPLIES], schema)
        reconciled, changed = _constrained_map(field_log_scores, field_values, compiled, schema)
        # The old path's answer (existing test_w3d_map expectation).
        assert reconciled["intent"] == "billing"
        assert reconciled["subtype"] == "refund"
        assert "subtype" in changed
        assert "intent" not in changed

    def test_no_constraints_noop(self):
        compiled = compile_constraints([], _schema())
        reconciled, changed = _constrained_map(
            {"intent": {"billing": -0.1}}, {"intent": {"value": "billing"}}, compiled, _schema()
        )
        assert reconciled == {}
        assert changed == []


def test_evalmetrics_accepts_compiled(monkeypatch):
    """evalmetrics._check_constraint takes compiled objects (constraint_
    violation_rate consumes the compiled form)."""
    from jevmlx.evalmetrics import _check_constraint

    compiled = compile_constraints([_IMPLIES], _schema())
    # Compiled objects expose .satisfied — the name-keyed view.
    assert _check_constraint(
        compiled.implications[0].__class__ and compiled,
        {
            "intent": "billing",
            "subtype": "refund",
        },
    )
