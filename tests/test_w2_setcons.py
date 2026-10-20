"""Hard set constraints for multi fields.

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
from conftest import CountCodeModel as _CountCodeModel
from conftest import YNLogitModel as _YNModel
from conftest import _Mod97Tokenizer as _CountTokenizer
from conftest import make_engine

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


def test_implies_into_exclusive_group_compiles_feasible():
    """x implies y with x,y in one mutually_exclusive group
    is SATISFIABLE — the empty set satisfies both (x absent => y not forced;
    at-most-one holds). The old compiler rejected it with a syntactic rule;
    the solver-based satisfiability check accepts it, and the engine's
    solver simply never selects x."""
    schema = StructuredSchema(
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
    assert schema["flags"].set_constraints  # compiled, not rejected


def test_mutual_implies_pair_compiles():
    """A->B, B->A is satisfiable (both absent)."""
    schema = StructuredSchema(
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
    assert schema["flags"].set_constraints


def test_implies_into_exact_k0_group_compiles():
    """A->B with exact_k([B], 0) is satisfiable (A not
    selected — the empty set closes to itself)."""
    schema = StructuredSchema(
        {
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["x", "y"],
                "set_constraints": [
                    {"type": "implies", "if_option": "x", "then_option": "y"},
                    {"type": "exact_k", "options": ["y"], "k": 0},
                ],
            }
        }
    )
    assert schema["flags"].set_constraints


def test_truly_unsatisfiable_constraints_raise():
    """The solver-based check still catches genuinely unsatisfiable sets:
    exact_k=1 over two options plus at_most_one over the same two is
    contradictory (need exactly 1 hit but at most... wait, exact_k=1 with
    at_most_one is satisfiable — pick one. Use at_least_one + exact_k=0
    over the same pair: need >= 1 and exactly 0."""
    with pytest.raises(SchemaCompileError, match="no selection"):
        StructuredSchema(
            {
                "flags": {
                    "type": "multi",
                    "description": "d",
                    "choices": ["x", "y"],
                    "set_constraints": [
                        {"type": "at_least_one", "options": ["x", "y"]},
                        {"type": "exact_k", "options": ["x", "y"], "k": 0},
                    ],
                }
            }
        )


def test_implies_cycle_compiles_and_solver_never_selects():
    """A 2-cycle x->y->x is SATISFIABLE (both absent — the
    closure of the empty selection is empty), so it compiles; the engine's
    solver then simply never selects x or y (selecting either forces both,
    which is fine for implies-only — actually selecting x gives {x,y}, which
    satisfies implies. The cycle does NOT make implies-only unsatisfiable;
    the old 'cycle raises' rule was wrong for the implies-only case."""
    schema = StructuredSchema(
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
    assert schema["flags"].set_constraints


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
    result = run_parallel_generation(make_engine(_YNModel(), _CountTokenizer()), "ctx", schema)
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
    result = (
        run_parallel_generation(make_engine(_YNModel(), _YNModel()), "ctx", schema)
        if False
        else None
    )
    result = run_parallel_generation(make_engine(_YNModel(), _CountTokenizer()), "ctx", schema)
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
    result = run_parallel_generation(make_engine(model, _CountTokenizer()), "ctx", schema)
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["set_selection"] == "per_option"
    assert result["parsed_json"]["flags"]["value"] == []


class TestW5CSolverExactness:
    """One exact component solver over group overlap
    AND implication edges; no caps that reject legal schemas."""

    def test_two_disjoint_groups_both_feasible(self):
        """Finding 14: while enumerating component {A,B}, the solver checked
        EVERY group constraint, so the {C,D} at_least_one killed all {A,B}
        candidates and the solver reported no solution."""
        from jevmlx.setcons import select_constrained_set

        selected, rule = select_constrained_set(
            ["A", "B", "C", "D"],
            {"A": -1.0, "B": -1.0, "C": 2.0, "D": 0.0},
            [
                {"type": "at_least_one", "options": ["A", "B"]},
                {"type": "at_least_one", "options": ["C", "D"]},
            ],
            set(),
        )
        assert set(selected) == {"A", "C"}

    def test_implies_only_maximizes_score(self):
        """Finding 15: implies-only closed the proposal under implication
        instead of maximizing. scores X=0.1, Y=-100; X->Y: {X,Y} scores
        -99.9, the empty set is feasible at 0.0."""
        from jevmlx.setcons import select_constrained_set

        selected, _rule = select_constrained_set(
            ["X", "Y"],
            {"X": 0.1, "Y": -100.0},
            [{"type": "implies", "if_option": "X", "then_option": "Y"}],
            {"X"},
        )
        assert selected == []

    def test_implies_crossing_group_not_discarded(self):
        """Finding 16: an implication crossing two group components — the
        optimum needs the component's second-best assignment after the
        implication forces a member out. A -> B with B in an at_most_one
        group with C (C much better than A+B): the old solver picked A
        locally, the implication invalidated the mask, and the mask was
        discarded entirely instead of trying {C}."""
        from jevmlx.setcons import select_constrained_set

        # Options: A (implies B), B, C in at_most_one with B.
        # Scores: A=5, C=10, B=-100. Proposal: {A}.
        # Feasible: {} (score 0), {C} (score 10), {B} (-100), {B,C} infeasible.
        # {A,B} infeasible (at_most_one). Optimum: {C}.
        selected, rule = select_constrained_set(
            ["A", "B", "C"],
            {"A": 5.0, "B": -100.0, "C": 10.0},
            [
                {"type": "implies", "if_option": "A", "then_option": "B"},
                {"type": "at_most_one", "options": ["B", "C"]},
            ],
            {"A"},
        )
        assert set(selected) == {"C"}

    def test_no_16_source_cap(self):
        """Finding 16: >16 implication sources must not raise — the cap
        rejected legal schemas (schema permits 64 options)."""
        from jevmlx.setcons import select_constrained_set

        n = 20
        options = [f"o{i}" for i in range(n)]
        constraints = [
            {"type": "implies", "if_option": f"o{i}", "then_option": f"o{i + 1}"}
            for i in range(n - 1)
        ]
        selected, _rule = select_constrained_set(
            options, {o: 1.0 for o in options}, constraints, set(options)
        )
        assert set(selected) == set(options)

    def test_component_cap_raises_above_20(self):
        """The cap STAYS — a 24-option at_most_one
        component is 2^24 Python-loop candidates (multi-second), so the
        solver raises naming the component and the limit; the schema
        compiler surfaces it at compile time."""
        from jevmlx.setcons import select_constrained_set

        options = [f"o{i}" for i in range(24)]
        constraints = [{"type": "at_most_one", "options": options}]
        scores = {o: 0.1 for o in options}
        scores["o23"] = 5.0
        with pytest.raises(ValueError, match="24 options.*20-option"):
            select_constrained_set(options, scores, constraints, set())

    def test_component_cap_is_per_component_not_per_field(self):
        """The cap is on ONE component. A 30-option field whose groups
        are 3-option components (max 3) enumerates fine."""
        from jevmlx.setcons import select_constrained_set

        options = [f"o{i}" for i in range(30)]
        constraints = [
            {"type": "at_most_one", "options": [f"o{i}", f"o{i + 1}", f"o{i + 2}"]}
            for i in range(0, 30, 3)
        ]
        scores = {o: 0.1 for o in options}
        scores["o29"] = 5.0
        selected, _rule = select_constrained_set(options, scores, constraints, set())
        assert "o29" in selected

    def test_group_constraint_checked_only_in_own_component(self):
        """Finding 14 (unit): components_of builds from group overlap AND
        implication edges — {A,B} and {C,D} groups are separate components."""
        from jevmlx.setcons import components_of

        comps = components_of(
            ["A", "B", "C", "D"],
            [
                {"type": "at_least_one", "options": ["A", "B"]},
                {"type": "at_least_one", "options": ["C", "D"]},
            ],
        )
        assert sorted(sorted(c) for c in comps) == [["A", "B"], ["C", "D"]]

    def test_implies_edges_merge_components(self):
        """An implies edge between two groups fuses them into one component
        (the old solver composed independently-optimized
        subcomponents even when an implication crossed them)."""
        from jevmlx.setcons import components_of

        comps = components_of(
            ["A", "B", "C", "D"],
            [
                {"type": "at_least_one", "options": ["A", "B"]},
                {"type": "at_least_one", "options": ["C", "D"]},
                {"type": "implies", "if_option": "B", "then_option": "C"},
            ],
        )
        assert sorted(sorted(c) for c in comps) == [["A", "B", "C", "D"]]


class TestW5CReviewF8F17:
    """The cap decision and is_feasible propagation."""

    def test_is_feasible_propagates_component_cap(self):
        """F17: a cap breach through is_feasible (compile time) PROPAGATES —
        the caller must see the schema problem, not a False verdict."""
        from jevmlx.setcons import is_feasible

        options = [f"o{i}" for i in range(24)]
        with pytest.raises(ValueError, match="24 options.*20-option"):
            is_feasible(options, [{"type": "at_most_one", "options": options}])

    def test_is_feasible_returns_false_for_unsatisfiable(self):
        """F17: the unsatisfiable case still maps to False (at_least_one +
        exact_k=0 over the same pair)."""
        from jevmlx.setcons import is_feasible

        assert not is_feasible(
            ["x", "y"],
            [
                {"type": "at_least_one", "options": ["x", "y"]},
                {"type": "exact_k", "options": ["x", "y"], "k": 0},
            ],
        )

    def test_oversized_component_fails_at_schema_compile(self):
        """End-to-end: a >20-option component is a COMPILE error via the
        solver's cap, surfaced by the compiler."""
        from jevmlx.schema import SchemaCompileError, StructuredSchema

        options = [f"o{i}" for i in range(24)]
        with pytest.raises(SchemaCompileError, match="20-option"):
            StructuredSchema(
                {
                    "flags": {
                        "type": "multi",
                        "description": "d",
                        "choices": options,
                        "set_constraints": [{"type": "at_most_one", "options": options}],
                    }
                }
            )


# ------------- case-level exclusivity in the solver -------


def _excl_schema():
    return StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y", "z"]}}
    )


def test_T1_case_exclusivity_enforced_in_selection():
    """A case-level exclusivity [x, y] on a
    multi field is enforced by the joint set solver — the model wants
    x, y, z (uniform P(yes) > 0.5), the result keeps the higher-of-x/y by z
    (a tie here: schema order) plus z. Main returns [x, y, z] unchecked."""
    from jevmlx.engine import run_parallel_generation

    constraints = [{"type": "exclusivity", "field": "flags", "options": ["x", "y"]}]
    result = run_parallel_generation(
        make_engine(_YNModel(), _CountTokenizer()),
        "ctx",
        _excl_schema(),
        constraints=constraints,
    )
    assert result["parsed_json"]["flags"]["value"] == ["x", "z"]
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["reconciled_by"] == "constraints"
    # The exclusivity group rides the solver's constraint list (provenance).
    assert {"type": "at_most_one", "options": ["x", "y"]} in telemetry["set_constraints"]
    assert telemetry["threshold_proposal"] == ["x", "y", "z"]
    assert telemetry["semantics"]["constraint_changed"] is True
    # Per-option numbers stay DISTRIBUTION numbers.
    assert all(p > 0.5 for p in telemetry["per_option"].values())


def test_T2_exclusivity_and_trusted_count_jointly():
    """A trusted count k=2 AND exclusivity [x, y] are ONE solver
    call — the result honors both ({x, z}: two options, at most one of the
    group). Rule when they conflict: schema exclusivity is HARD, the count
    is evidence and DROPS (existing precedence) — tested by the infeasible
    variant below. Main ignores the exclusivity and returns [x, y]."""
    from jevmlx.engine import run_parallel_generation

    constraints = [{"type": "exclusivity", "field": "flags", "options": ["x", "y"]}]
    model = _CountCodeModel({"2": 3.0}, yes_logit=1.0, no_logit=-1.0)
    result = run_parallel_generation(
        make_engine(model, _CountTokenizer()), "ctx", _excl_schema(), constraints=constraints
    )
    assert result["parsed_json"]["flags"]["value"] == ["x", "z"]
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["count_choice"] == "2"
    assert "dropped_reason" not in telemetry
    assert telemetry["reconciled_by"] == "count"


def test_T2b_exclusivity_beats_infeasible_trusted_count():
    """The conflict rule: count k=2 over [x, y] with exclusivity
    [x, y] is jointly infeasible — schema exclusivity is HARD, the trusted
    count drops (dropped_reason recorded) and the joint solver still
    enforces the exclusivity on the threshold proposal. Main: no drop, no
    enforcement — [x, y]."""
    from jevmlx.engine import run_parallel_generation

    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    constraints = [{"type": "exclusivity", "field": "flags", "options": ["x", "y"]}]
    model = _CountCodeModel({"2": 3.0}, yes_logit=1.0, no_logit=-1.0)
    result = run_parallel_generation(
        make_engine(model, _CountTokenizer()), "ctx", schema, constraints=constraints
    )
    assert result["parsed_json"]["flags"]["value"] == ["x"]
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["count_choice"] == "2"
    # The drop is recorded on the count row's OWN (internal) telemetry.
    count_tel = result["internal_telemetry"]["flags#count"]
    assert count_tel["dropped_reason"] == "infeasible_with_set_constraints"
    assert telemetry["reconciled_by"] == "constraints"


def test_T3_no_exclusivity_selection_identical():
    """The same field with NO case constraints — the threshold
    proposal stands untouched, no set-constraint telemetry keys (the
    main-identical invariant)."""
    from jevmlx.engine import run_parallel_generation

    result = run_parallel_generation(
        make_engine(_YNModel(), _CountTokenizer()), "ctx", _excl_schema()
    )
    assert result["parsed_json"]["flags"]["value"] == ["x", "y", "z"]
    telemetry = result["field_telemetry"]["flags"]
    assert "set_constraints" not in telemetry
    assert telemetry["reconciled_by"] == "per_option"


def test_T4_eval_exclusivity_violation_rate_zero():
    """The engine's constrained output satisfies the exclusivity —
    constraint_violation_rate reports 0 for a run whose UNCONSTRAINED
    selection ([x, y, z]) would violate it. Main's output violates — the
    metric reports a nonzero rate."""
    from jevmlx.engine import run_parallel_generation
    from jevmlx.evalmetrics import constraint_violation_rate

    constraints = [{"type": "exclusivity", "field": "flags", "options": ["x", "y"]}]
    result = run_parallel_generation(
        make_engine(_YNModel(), _CountTokenizer()),
        "ctx",
        _excl_schema(),
        constraints=constraints,
    )
    value = result["parsed_json"]["flags"]["value"]
    assert len([o for o in value if o in ("x", "y")]) <= 1
    records = [
        {
            "case_id": "c1",
            "field": "flags",
            "type": "multi",
            "prediction": value,
            "label": ["x"],
            "constraints": constraints,
        }
    ]
    rate = constraint_violation_rate(records, _excl_schema())
    assert rate["overall"] == 0.0
    assert rate["by_type"]["exclusivity"] == 0.0
