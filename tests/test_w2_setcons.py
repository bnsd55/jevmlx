"""W2-SETCONS: hard set constraints for multi fields.

Schema-declared set constraints (mutually_exclusive, at_most_one,
at_most_k, at_least_one, exact_k, implies between options) are validated at
COMPILE time (contradictory sets raise SchemaCompileError at construction);
the engine's multi selection runs the threshold proposal through the exact
solver in jevmlx.setcons.py, which picks the score-maximizing feasible set.
Telemetry: set_constraints (verbatim) + set_selection
("constraints"|"per_option") on the field entry; absent when the field
declares no set constraints.
"""

import pytest
from conftest import YNLogitModel as _YNModel
from conftest import _Mod97Tokenizer as _CountTokenizer

from jevmlx.schema import SchemaCompileError, StructuredSchema


def _schema(constraints, choices=("x", "y", "z")):
    spec = {"type": "multi", "description": "d", "choices": list(choices)}
    if constraints is not None:
        spec["set_constraints"] = constraints
    return StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": list(choices)}}
        if constraints is None
        else {"flags": spec}
    )


# ---------------------------------------------------------------- compile --


def test_valid_constraints_compile_and_roundtrip():
    """Accepted constraint types attach to the field and survive to_dict."""
    constraints = [
        {"type": "mutually_exclusive", "options": ["x", "y"]},
        {"type": "at_most_k", "options": ["z"], "k": 1},
        {"type": "at_least_one", "options": ["x", "z"]},
        {"type": "exact_k", "options": ["x", "y", "z"], "k": 2},
        {"type": "implies", "if_option": "z", "then_option": "y"},
    ]
    schema = StructuredSchema(
        {
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["x", "y", "z"],
                "set_constraints": constraints,
            }
        }
    )
    assert [dict(c) for c in schema["flags"].set_constraints] == constraints
    assert schema["flags"].to_dict()["set_constraints"] == constraints


def test_contrapositive_of_implies_inside_exclusive_group_raises():
    """x implies y with x,y in one mutually_exclusive group: selecting x
    forces y, violating at-most-one — unsatisfiable for any set containing
    x, so the constraint set is contradictory. Compile-time error."""
    with pytest.raises(SchemaCompileError, match="contradiction"):
        StructuredSchema(
            {
                "flags": {
                    "type": "multi",
                    "description": "d",
                    "choices": ["x", "y", "z"],
                    "set_constraints": [
                        {"type": "mutually_exclusive", "options": ["x", "y"]},
                        {"type": "implies", "if_option": "x", "then_option": "y"},
                    ],
                }
            }
        )


def test_implies_cycle_raises():
    """x -> y -> x loops: compile-time SchemaCompileError."""
    with pytest.raises(SchemaCompileError, match="cycle"):
        StructuredSchema(
            {
                "flags": {
                    "type": "multi",
                    "description": "d",
                    "choices": ["x", "y"],
                    "set_constraints": [
                        {"type": "implies", "if_option": "x", "then_option": "y"},
                        {"type": "implies", "if_option": "y", "then_option": "x"},
                    ],
                }
            }
        )


def test_exact_k_above_group_size_raises():
    with pytest.raises(SchemaCompileError, match="unsatisfiable"):
        StructuredSchema(
            {
                "flags": {
                    "type": "multi",
                    "description": "d",
                    "choices": ["x", "y"],
                    "set_constraints": [{"type": "exact_k", "options": ["x", "y"], "k": 3}],
                }
            }
        )


def test_unknown_option_raises():
    with pytest.raises(SchemaCompileError, match="not in field choices"):
        StructuredSchema(
            {
                "flags": {
                    "type": "multi",
                    "description": "d",
                    "choices": ["x", "y"],
                    "set_constraints": [{"type": "mutually_exclusive", "options": ["x", "q"]}],
                }
            }
        )


def test_set_constraints_on_non_multi_raises():
    schema = StructuredSchema(
        {"topic": {"type": "enum", "description": "d", "choices": ["a", "b"]}}
    )
    with pytest.raises(SchemaCompileError, match="multi fields only"):
        schema["topic"].compile_set_constraints([{"type": "mutually_exclusive", "options": ["a"]}])


def test_unknown_constraint_type_raises():
    with pytest.raises(SchemaCompileError, match="type must be one of"):
        StructuredSchema(
            {
                "flags": {
                    "type": "multi",
                    "description": "d",
                    "choices": ["x", "y"],
                    "set_constraints": [{"type": "whatever", "options": ["x"]}],
                }
            }
        )


def test_set_constraints_require_dict_list():
    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    with pytest.raises(SchemaCompileError):
        schema["flags"].compile_set_constraints(["not-a-dict"])
    with pytest.raises(SchemaCompileError):
        schema["flags"].compile_set_constraints([])


# ----------------------------------------------------------------- solver --


def test_solver_picks_best_option_when_group_binds():
    """mutually_exclusive over an all-yes proposal: exactly one of the group
    survives — the higher-scoring one; unconstrained options are untouched."""
    from jevmlx.setcons import select_constrained_set

    sel, rule = select_constrained_set(
        ["x", "y", "z"],
        {"x": 2.0, "y": 3.0, "z": 0.5},
        [{"type": "mutually_exclusive", "options": ["x", "y"]}],
        {"x", "y", "z"},
    )
    assert sel == ["y", "z"]
    assert rule == "constraints"


def test_solver_non_binding_keeps_per_option():
    """Constraint satisfied by the proposal: selection unchanged, and the
    rule reports per_option (bit-stable with the no-constraint path)."""
    from jevmlx.setcons import select_constrained_set

    sel, rule = select_constrained_set(
        ["x", "y", "z"],
        {"x": 2.0, "y": -1.0, "z": 0.5},
        [{"type": "mutually_exclusive", "options": ["x", "y"]}],
        {"x", "z"},
    )
    assert sel == ["x", "z"]
    assert rule == "per_option"


def test_solver_implies_forces_option():
    from jevmlx.setcons import select_constrained_set

    sel, rule = select_constrained_set(
        ["x", "y", "z"],
        {"x": 2.0, "y": -1.0, "z": 0.5},
        [{"type": "implies", "if_option": "x", "then_option": "y"}],
        {"x", "z"},
    )
    assert sel == ["x", "y", "z"]
    assert rule == "constraints"


def test_solver_exact_k_single_best():
    from jevmlx.setcons import select_constrained_set

    sel, _rule = select_constrained_set(
        ["x", "y", "z"],
        {"x": 2.0, "y": 3.0, "z": 0.5},
        [{"type": "exact_k", "options": ["x", "y", "z"], "k": 1}],
        {"x", "y", "z"},
    )
    assert sel == ["y"]


def test_solver_at_least_one_forces_best_when_empty():
    from jevmlx.setcons import select_constrained_set

    sel, rule = select_constrained_set(
        ["x", "y"],
        {"x": 0.1, "y": -0.2},
        [{"type": "at_least_one", "options": ["x", "y"]}],
        set(),
    )
    assert sel == ["x"]
    assert rule == "constraints"


def test_solver_implies_across_groups():
    """x in an at-most-one group implies z outside it: the solver picks x
    (2.0 > 1.0) and drags z in."""
    from jevmlx.setcons import select_constrained_set

    sel, _rule = select_constrained_set(
        ["x", "y", "z"],
        {"x": 2.0, "y": 1.0, "z": 0.5},
        [
            {"type": "mutually_exclusive", "options": ["x", "y"]},
            {"type": "implies", "if_option": "x", "then_option": "z"},
        ],
        {"x", "y"},
    )
    assert sel == ["x", "z"]


# ----------------------------------------------------------------- engine --


def test_engine_mutually_exclusive_collapses_all_yes():
    """All options P(yes)=0.88: unconstrained selects everything; the
    mutually_exclusive group keeps exactly one; telemetry carries the
    constraints and the 'constraints' selection rule."""
    from jevmlx.engine import run_parallel_generation

    schema = _schema([{"type": "mutually_exclusive", "options": ["x", "y"]}])
    result = run_parallel_generation(_YNModel(), _CountTokenizer(), "ctx", schema)
    value = result["parsed_json"]["flags"]["value"]
    assert len([o for o in value if o in ("x", "y")]) <= 1
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["set_selection"] == "constraints"
    assert telemetry["set_constraints"] == [{"type": "mutually_exclusive", "options": ["x", "y"]}]


def test_engine_no_constraints_no_telemetry_keys():
    """No set constraints: no set_constraints/set_selection keys — the
    absent-key policy (same as calibrated_log_odds)."""
    from jevmlx.engine import run_parallel_generation

    schema = _schema(None)
    result = run_parallel_generation(_YNModel(), _YNModel(), "ctx", schema) if False else None
    result = run_parallel_generation(_YNModel(), _CountTokenizer(), "ctx", schema)
    telemetry = result["field_telemetry"]["flags"]
    assert "set_constraints" not in telemetry
    assert "set_selection" not in telemetry


def test_engine_non_binding_constraints_keep_per_option():
    """Proposal already satisfies the constraints: value unchanged and
    set_selection == 'per_option'."""
    from jevmlx.engine import run_parallel_generation

    schema = _schema([{"type": "at_most_one", "options": ["y", "z"]}])
    # Only x says yes (no_logit strong): proposal = [] ... build a yes-only-x
    # by an all-no model with y's row unaffected — simplest: all yes model but
    # the constraint group {y,z} with all-yes violates, so use per-option
    # biasing via a model whose N wins: selection [] satisfies at_most_one.
    model = _YNModel(yes_logit=-2.0, no_logit=2.0)
    result = run_parallel_generation(model, _CountTokenizer(), "ctx", schema)
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["set_selection"] == "per_option"
    assert result["parsed_json"]["flags"]["value"] == []
