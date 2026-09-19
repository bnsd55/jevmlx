"""W2-SETCONS (W5-C rewrite): exact hard set-constraint selection.

Given per-option scores (the calibrated log-odds when a calibrator ran, else
a monotone transform of P(yes)) and the field's validated set constraints,
select the option SET that maximizes the summed score under every
constraint. This is a HARD selection: the unconstrained threshold rule
proposes, the solver disposes — the result always satisfies the schema's
declared constraints.

Solver shape (exact, W5-C: one solver, complete components):

The solver is a single exact search over CONNECTED COMPONENTS built from
BOTH group-overlap edges (mutually_exclusive / at_most_one / at_most_k /
at_least_one / exact_k share options) AND implication edges (implies links
its if_option and then_option). Splitting groups and implications into
separate structures is what made the previous versions unsound (a
component-local optimum can be invalidated by implication propagation
crossing components; implies-only inputs returned the proposal's closure
rather than the score-maximizing feasible set).

Per component: bitmask enumeration with implication closure applied inside
the enumeration (a selection is feasible iff its closed set satisfies every
constraint whose options intersect the component). Components are
independent — no constraint spans two components by construction — so the
global optimum is the union of per-component optima. A component larger
than MAX_COMPONENT_OPTIONS (20) cannot be enumerated in acceptable time
(2^20 is already millions of Python-loop candidates; 2^30 is minutes) and
raises ValueError naming the component — the schema compiler surfaces it
at compile time, before any inference. The cap is on ONE component, not
the field: a 64-option field whose largest component has 12 options is
fine (F8 decision, review 2026-09-19: the cap stays).

Unconstrained options (touched by no constraint) keep their threshold
decision: they are free, their score contribution is fixed, and the solver
never has a reason to flip them.

Feasibility (used by the schema compiler too, finding 17):
:func:`is_feasible` runs the same component enumeration without scores.
The compiler calls it instead of inferring unsatisfiability from syntactic
patterns (A->B + B->A, A->B + at_most_one(A,B), A->B + exact_k([B], 0) are
all satisfiable — both absent / A forbidden / A not selected).
"""

from __future__ import annotations

__all__ = [
    "MAX_COMPONENT_OPTIONS",
    "select_constrained_set",
    "is_feasible",
    "components_of",
]

# F8 (review 2026-09-19): a single component above this size is not
# enumerated — 2^20 bitmask candidates is already a multi-second Python
# loop and 2^30 is minutes. is_feasible runs the same search at schema
# COMPILE time, so an over-large component fails loudly there (the
# compiler's SchemaCompileError), never as a mid-inference hang. The cap
# is per component: a big field with many small components is unaffected.
MAX_COMPONENT_OPTIONS = 20


def _options_of(constraint: dict) -> list[str]:
    return list(constraint.get("options", []))


def _check_local(constraint: dict, chosen: set[str]) -> bool:
    """Feasibility of one set constraint against a candidate selection."""
    ctype = constraint.get("type")
    options = set(_options_of(constraint))
    hits = len(chosen & options)
    if ctype in ("mutually_exclusive", "at_most_one"):
        return hits <= 1
    if ctype == "at_most_k":
        return hits <= constraint["k"]
    if ctype == "at_least_one":
        return hits >= 1
    if ctype == "at_least_k":
        return hits >= constraint["k"]
    if ctype == "exact_k":
        return hits == constraint["k"]
    if ctype == "implies":
        return constraint["then_option"] in chosen or constraint["if_option"] not in chosen
    return True  # unknown: satisfied (same policy as constraints.py)


def components_of(options: list[str], constraints: list[dict]) -> list[frozenset[str]]:
    """Connected components over group overlap AND implication edges.

    Two options are in the same component when they share a group constraint
    or one implies the other. Only options touched by at least one constraint
    appear in a component; untouched options are free.
    """
    touched: set[str] = set()
    adjacency: dict[str, set[str]] = {}

    def link(a: str, b: str) -> None:
        touched.add(a)
        touched.add(b)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    for c in constraints:
        ctype = c.get("type")
        if ctype == "implies":
            link(c["if_option"], c["then_option"])
        else:
            opts = _options_of(c)
            for o in opts:
                touched.add(o)
            for other in opts[1:]:
                link(opts[0], other)
    # Link group members pairwise (link above chains opts[0] to each other
    # member, which is enough for connectivity but let's be explicit for
    # every pair so a constraint listing options in any order still connects
    # all of them).
    for c in constraints:
        if c.get("type") == "implies":
            continue
        opts = _options_of(c)
        for i, a in enumerate(opts):
            for b in opts[i + 1 :]:
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)

    comps: list[frozenset[str]] = []
    for start in sorted(touched, key=options.index):
        if start in {o for comp in comps for o in comp}:
            continue
        stack, comp = [start], set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(adjacency.get(node, ()) - comp)
        comps.append(frozenset(comp))
    return comps


def _implied_closure(constraints: list[dict]) -> dict[str, set[str]]:
    """Transitive closure of the implies relation: option -> forced set."""
    edges: dict[str, set[str]] = {}
    for c in constraints:
        if c.get("type") == "implies":
            edges.setdefault(c["if_option"], set()).add(c["then_option"])
    closure: dict[str, set[str]] = {}

    def closure_of(opt: str, seen: frozenset[str]) -> set[str]:
        if opt in closure:
            return closure[opt]
        forced: set[str] = set()
        for nxt in edges.get(opt, ()):  # direct
            forced.add(nxt)
            if nxt not in seen:
                forced |= closure_of(nxt, seen | {nxt})
        closure[opt] = forced
        return forced

    for opt in list(edges):
        closure_of(opt, frozenset({opt}))
    return closure


def select_constrained_set(
    options: list[str],
    scores: dict[str, float],
    constraints: list[dict],
    unconstrained_selected: set[str],
) -> tuple[list[str], str]:
    """Select the score-maximizing set satisfying every constraint.

    ``options``: the field's choices (schema order — ties resolve to the
    earlier option, mirroring the engine's tie policy).
    ``scores``: per-option value to maximize (calibrated log-odds when a
    calibrator ran; else log-odds of P(yes), monotone in P(yes) — the same
    ranking the threshold rule uses).
    ``constraints``: the field's validated set constraints.
    ``unconstrained_selected``: the threshold rule's proposal — the DP's
    preference floor (a tie between feasible sets of equal score resolves
    toward the proposal's membership, keeping the no-constraint behavior
    bit-stable when constraints don't bind).

    Returns ``(selected_options_sorted_by_schema_order, rule)`` where
    ``rule`` is "constraints" (the solver changed something or bound) —
    telemetry's reconciled_by analog for the set-constraint path.
    """
    if not constraints:
        return [o for o in options if o in unconstrained_selected], "per_option"

    closure = _implied_closure(constraints)
    opts = list(options)
    order = {o: i for i, o in enumerate(opts)}
    proposal = set(unconstrained_selected)

    def forced_by(chosen: set[str]) -> set[str]:
        forced: set[str] = set()
        for o in chosen:
            forced |= closure.get(o, set())
        return forced

    def total_score(sel: set[str]) -> float:
        return sum(scores.get(o, 0.0) for o in sel)

    comps = components_of(opts, constraints)

    # Free options keep their threshold decision (untouched by constraints —
    # the solver never has a reason to flip them; their score is constant).
    comp_opts: set[str] = {o for comp in comps for o in comp}
    # comps is never empty here: constraints is non-empty (early return
    # above) and every constraint's options are linked into a component —
    # even a constraint type with unknown semantics contributes its options
    # via _options_of.
    best_sel: set[str] = set()

    for comp in comps:
        comp_sorted = sorted(comp, key=order.__getitem__)
        n = len(comp_sorted)
        if n > MAX_COMPONENT_OPTIONS:
            raise ValueError(
                f"set-constraint component has {n} options, above the "
                f"{MAX_COMPONENT_OPTIONS}-option enumeration cap "
                f"(component={comp_sorted}); split the constraint or shrink "
                f"the field"
            )
        # Bitmask enumeration with implication closure applied per
        # candidate. Closure targets outside the component are added to
        # the assembled set after per-component selection (finding 16:
        # components were built OVER implication edges, so a forced
        # option is in this component or free-but-implication-linked;
        # either way adding it at assembly cannot invalidate a sibling
        # component — verified by the final re-check below).
        best_c: set[str] | None = None
        best_key: tuple[float, int] | None = None
        local_constraints = [c for c in constraints if set(_options_of(c)) & comp]
        for mask in range(1 << n):
            cand = {comp_sorted[i] for i in range(n) if mask >> i & 1}
            # Close under implication BEFORE the constraint check so the
            # enumeration explores full closures, not raw selections.
            closed = cand | forced_by(cand)
            if not all(_check_local(c, closed) for c in local_constraints):
                continue
            # A source not chosen but forced by another chosen source is
            # inconsistent: the closure already pulled it in, and the
            # check above would catch violations — but the source itself
            # must be IN (it was forced): closure guarantees that. Skip
            # nothing further; closed sets are the search space.
            cs = total_score(closed)
            tie_break = len(closed & proposal)
            if (
                best_key is None
                or cs > best_key[0]
                or (cs == best_key[0] and tie_break > best_key[1])
            ):
                best_c, best_key = closed, (cs, tie_break)
        if best_c is None:
            raise ValueError(
                "no selection satisfies the field's set constraints "
                f"(component={comp_sorted}, constraints={constraints})"
            )
        best_sel |= best_c

    # Free (untouched) options keep their threshold decision.
    free_selected = {o for o in proposal if o not in comp_opts}
    best_sel |= free_selected

    # Implies targets outside every component (forced free options).
    best_sel |= forced_by(best_sel)

    # Defense in depth: the assembled set must satisfy EVERY constraint.
    if not all(_check_local(c, best_sel) for c in constraints):
        raise ValueError(
            "internal error: solver assembled a constraint-violating set "
            f"({sorted(best_sel)} vs {constraints})"
        )

    final = [o for o in opts if o in best_sel]
    proposal_set = [o for o in opts if o in proposal]
    rule = "constraints" if final != proposal_set else "per_option"
    return final, rule


def is_feasible(options: list[str], constraints: list[dict]) -> bool:
    """Satisfiability of ``constraints`` over ``options`` — no scores.

    The schema compiler runs this (finding 17) instead of syntactic
    contradiction rules: A->B + B->A, A->B + at_most_one(A,B), and A->B +
    exact_k([B], 0) are all satisfiable (both absent / A forbidden / A not
    selected). Same enumeration as the scored solver, empty proposal.

    Only the "no selection satisfies" failure maps to False (F17): the
    solver's internal-error guard and the component-size cap raise their
    own ValueErrors, which PROPAGATE — an internal error is a bug and a
    cap breach is a schema the caller must see, not a satisfiability
    verdict. Both are distinguishable by message prefix.
    """
    try:
        select_constrained_set(options, {}, constraints, set())
    except ValueError as exc:
        message = str(exc)
        if message.startswith("internal error"):
            raise  # a solver bug, not an unsatisfiable schema
        if message.startswith("set-constraint component has"):
            raise  # F8: the cap breach is the caller's problem to see
        return False
    return True
