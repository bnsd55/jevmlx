"""W2-SETCONS: hard set-constraint selection for multi fields.

Given per-option scores (the calibrated log-odds when a calibrator ran, else
a monotone transform of P(yes)) and the field's validated set constraints,
select the option SET that maximizes the summed score under every
constraint. This is a HARD selection: the unconstrained threshold rule
proposes, the solver disposes — the result always satisfies the schema's
declared constraints.

Solver shape (exact, no heuristic):

The constraints partition into independent structures over the option set:

- ``implies`` edges form a functional graph (option -> forced option).
  Options not on any implies chain are FREE; each chain tail forces its
  whole downstream (selection propagates transitively — the schema
  compiler has already rejected cycles).
- The remaining options form GROUPS (mutually_exclusive / at_most_one /
  at_most_k / at_least_one / exact_k), keyed by overlapping membership.
  Options in no group are free.

Exact DP over connected group components: for each component (a set of
overlapping groups, at most a handful in practice), enumerate the
component's feasible selections in product space bounded by the
component's option count (<= 64 choices per multi field keeps every
component tiny — a full 2^n walk over n <= 64 would not, so components
are the natural split: feasibility is component-local). The final answer
stitches per-component bests with the free options' threshold decisions
and the implies propagation, then re-checks every constraint on the
assembled set (defense in depth: the DP must never emit a violating set).
"""

from __future__ import annotations

__all__ = ["select_constrained_set"]


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
    if ctype == "exact_k":
        return hits == constraint["k"]
    if ctype == "implies":
        return constraint["then_option"] in chosen or constraint["if_option"] not in chosen
    return True  # unknown: satisfied (same policy as constraints.py)


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


def _group_constraints(constraints: list[dict]) -> list[dict]:
    return [c for c in constraints if c.get("type") != "implies"]


def _components(groups: list[dict], options: list[str]) -> list[set[str]]:
    """Connected components over group-overlap (union-find). Only components
    containing at least one constrained option are returned — unconstrained
    options don't enumerate."""
    parent: dict[str, str] = {o: o for o in options}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in groups:
        opts = _options_of(c)
        for o in opts[1:]:
            union(opts[0], o)
    comps: dict[str, set[str]] = {}
    for o in options:
        comps.setdefault(find(o), set()).add(o)
    constrained_opts: set[str] = set()
    for c in groups:
        constrained_opts |= set(_options_of(c))
    return [set(v) for v in comps.values() if v & constrained_opts]


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
    ``constraints``: the field's validated set constraints (schema compile
    has already rejected malformed/contradictory ones; the solver still
    re-checks feasibility of the final set).
    ``unconstrained_selected``: the threshold rule's proposal — used as the
    DP's preference floor (a tie between feasible sets of equal score
    resolves toward the proposal's membership, keeping the no-constraint
    behavior bit-stable when constraints don't bind).

    Returns ``(selected_options_sorted_by_schema_order, rule)`` where
    ``rule`` is "constraints" (the solver changed something or bound) —
    telemetry's reconciled_by analog for the set-constraint path.
    """
    if not constraints:
        return [o for o in options if o in unconstrained_selected], "per_option"

    closure = _implied_closure(constraints)
    opts = list(options)

    def forced_by(chosen: set[str]) -> set[str]:
        forced: set[str] = set()
        for o in chosen:
            forced |= closure.get(o, set())
        return forced

    def total_score(sel: set[str]) -> float:
        return sum(scores.get(o, 0.0) for o in sel)

    # Options touched by any constraint (group members or implies endpoints).
    touched: set[str] = set()
    for c in constraints:
        if c.get("type") == "implies":
            touched.add(c["if_option"])
            touched.add(c["then_option"])
        else:
            touched |= set(_options_of(c))
    proposal = set(unconstrained_selected)

    # Per connected component: enumerate feasible selections (each component
    # is small — bounded by the field's choice count), pick the best score.
    # Options outside every group but implies-linked still matter: implies
    # edges are checked globally on the assembled set, so components here are
    # GROUP components only; implies propagation happens at assembly.
    # Options NOT touched by any constraint keep their threshold decision:
    # they are free, and the DP never has a reason to flip them (their score
    # contribution is fixed). Seed the base set from the proposal minus
    # touched options; touched options come from the enumeration.
    base_free = {o for o in proposal if o not in touched}

    group_comps = _components([c for c in constraints if c.get("type") != "implies"], opts)

    best_sel: set[str] | None = None
    best_score = float("-inf")

    if group_comps:
        # Group decisions are independent per component; implies edges span
        # components, so enumerate per-component bests under each implies
        # closure scenario instead of the full cross product: enumerate the
        # implied-forced set over the Touched-but-ungrouped closure targets,
        # which is small (chains, not powersets). Practical shape: iterate
        # over candidate FORCED sets = closure targets of any selection of
        # implies sources; sources are the threshold-selected options, so
        # the forced set is determined by which sources are in. Enumerate
        # subsets of the implies SOURCE options that appear in groups (tiny:
        # sources are single options).
        implies_sources = sorted(
            {c["if_option"] for c in constraints if c.get("type") == "implies"}
        )
        n_sources = len(implies_sources)
        if n_sources > 16:
            raise ValueError(
                "set constraint implies fan-in too large: "
                f"{n_sources} source options; the exact enumeration bound is 16"
            )

        for src_mask in range(1 << n_sources):
            chosen_sources = {implies_sources[i] for i in range(n_sources) if src_mask >> i & 1}
            forced = forced_by(chosen_sources)
            # Every source NOT chosen must not be forced by another source —
            # if it is, this mask is inconsistent (the source is in whether
            # we like it or not): normalize by adding it.
            # Every source NOT chosen must not be forced by another —
            # otherwise the mask is inconsistent (the source is in whether
            # the solver likes it or not).
            consistent = all(src in chosen_sources or src not in forced for src in implies_sources)
            if not consistent:
                continue
            # Per component: best feasible selection given the forced set.
            comp_sels: list[set[str]] = []
            feasible = True
            for comp in group_comps:
                comp_opts = sorted(comp, key=opts.index)
                n = len(comp_opts)
                if n > 20:
                    raise ValueError(
                        f"set constraint group component too large: {n} options; "
                        "the exact enumeration bound is 20 per component"
                    )
                best_c: set[str] | None = None
                best_cs = float("-inf")
                group_checks = [c for c in constraints if c.get("type") != "implies"]
                for mask in range(1 << n):
                    cand = {comp_opts[i] for i in range(n) if mask >> i & 1}
                    if not all(_check_local(c, cand) for c in group_checks):
                        continue
                    cs = sum(scores.get(o, 0.0) for o in cand)
                    # Tie-break toward the proposal (and schema order via the
                    # ascending mask enumeration: earlier masks favor earlier
                    # options, and a later candidate must strictly beat the
                    # incumbent's proposal overlap to displace it).
                    tie_break = len(cand & proposal)
                    if cs > best_cs or (
                        cs == best_cs and best_c is not None and tie_break > len(best_c & proposal)
                    ):
                        best_c, best_cs = cand, cs
                if best_c is None:
                    feasible = False
                    break
                comp_sels.append(best_c)
            if not feasible:
                continue
            cand_sel = set(chosen_sources) | forced | base_free
            for cs in comp_sels:
                cand_sel |= cs
            if not all(_check_local(c, cand_sel) for c in constraints):
                continue
            sc = total_score(cand_sel)
            # Tie toward the unconstrained proposal, then schema order.
            cand_key = (sc, len(cand_sel & proposal))
            best_key = (
                best_score if best_sel is None else best_score,
                len(best_sel & proposal) if best_sel is not None else -1,
            )
            if best_sel is None or cand_key > best_key:
                best_sel, best_score = cand_sel, sc
        if best_sel is None:
            raise ValueError(
                "no selection satisfies the field's set constraints "
                f"(options={opts}, constraints={constraints})"
            )
    else:
        # Implies-only: the threshold selection's closure (compile rejected
        # cycles; the closure of a consistent selection is consistent). Free
        # options outside all constraints keep their threshold decision.
        best_sel = set(unconstrained_selected) | forced_by(set(unconstrained_selected))

    final = [o for o in opts if o in best_sel]
    proposal_set = [o for o in opts if o in proposal]
    rule = "constraints" if final != proposal_set else "per_option"
    return final, rule
