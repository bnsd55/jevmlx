"""Schema definitions and batch-plan compilation for parallel constrained decisions.

Supports booleans, categorical enums (cardinality up to 255), and multi fields
(subset of choices, 2-64 options, decided as one boolean decision per option).
"""

import dataclasses
import hashlib
import json
import logging
import weakref
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

# W5-A finding 39: ONE canonical JSON serializer (see jevmlx/json_text.py).
from jevmlx.json_text import json_text

_LOGGER = logging.getLogger(__name__)


def _freeze_plan(plan: Any) -> Any:
    """Deep-freeze a compiled plan for read-only exposure (W5b-1, C7).

    Dicts become ``MappingProxyType`` views over recursively frozen copies;
    lists become tuples. Token id lists are leaf data (ints), so tuples are
    safe everywhere. Cached once at compile time; callers can no longer
    corrupt a shared plan (e.g. strip lead-in twice) — mutation attempts
    raise.
    """
    if isinstance(plan, dict):
        return MappingProxyType({k: _freeze_plan(v) for k, v in plan.items()})
    if isinstance(plan, list):
        return tuple(_freeze_plan(v) for v in plan)
    return plan


def _thaw_plan(plan: Any) -> Any:
    """The inverse of :func:`_freeze_plan`: plain dict/list containers.

    For JSON serialization (run.json's ``compiled_plan_sha256``,
    ``schema.plan_hash``): json.dumps raises on ``MappingProxyType``, and a
    ``default=list`` fallback would collapse every mapping to its key list —
    thaw recursively so the payload carries the plan's CONTENT. Hash
    consumers must thaw before dumping, never dump the frozen container.
    """
    if isinstance(plan, dict) or isinstance(plan, MappingProxyType):
        return {k: _thaw_plan(v) for k, v in plan.items()}
    if isinstance(plan, (list, tuple)):
        return [_thaw_plan(v) for v in plan]
    return plan


def _freeze_choice_descriptions(
    name: str, choices: tuple[str, ...], descriptions: Mapping[str, str] | None
) -> Mapping[str, str]:
    """Validate and freeze per-choice glosses (W5b-1, C7).

    Keys must be declared choices; duplicates rejected by the caller. The
    returned mapping is a ``MappingProxyType`` over a copy — mutation via
    the schema raises ``TypeError``.
    """
    d = dict(descriptions or {})
    if d:
        unknown = sorted(set(d) - set(choices))
        if unknown:
            raise ValueError(
                f"Field '{name}': choice_descriptions keys not in choices: "
                f"{', '.join(repr(u) for u in unknown)}"
            )
    return MappingProxyType(d)


_LOGGER = logging.getLogger(__name__)


def _alias_code(index: int) -> str:
    """Neutral choice alias for slot scoring: A..Z, then AA..ZZ (base 26).

    Fallback only — slot plans now SEARCH a per-field codebook first
    (see :func:`_search_codebook`); this remains for index-based callers
    and the degenerate >2-char case.
    """
    if index < 0:
        raise ValueError(f"alias index must be >= 0, got {index}")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < 26:
        return letters[index]
    code = ""
    while index >= 0:
        code = letters[index % 26] + code
        index = index // 26 - 1
    return code


# Tokenizer-specific codebook search (review Q3 'Tokenizer-specific codebook
# search' + 'First-token-only scoring', R5, bug 7): slot aliases are no longer
# pinned to the choice's index — the compiler SEARCHES candidate codes and
# picks the set whose complete candidate rows tokenize most cleanly.
_CODEBOOK_SINGLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CODEBOOK_DIGITS = "0123456789"

# W2-E step 3 (count row): the candidate codes for the per-multi-field count
# row. '4' means "four or more". A multi field has at most 64 choices, but
# the reconciliation only needs a coarse bucket: k is capped at the field's
# option count, so 4 behaves as k = min(4, len(options)).
COUNT_CODES: list[str] = ["0", "1", "2", "3", "4"]


def count_key(fname: str) -> str:
    """The '<field>#count' telemetry/trie/prior key for a multi field's count
    row. One helper so the key format lives in exactly one place (F4, PR #24
    review) — callers must not rebuild it by concatenation. Field names
    cannot contain '#' (C3), so the key is injective against plain field
    names and never collides with '<field>/<code>' option-row keys."""
    return f"{fname}#count"


def is_count_key(key: str) -> bool:
    """True when `key` names a count row (a `count_key` output). Structural
    check via the helper, not substring sniffing."""
    return key.endswith("#count") and len(key) > len("#count")


def _variance(values: list[int]) -> float:
    """Population variance of a non-empty int list (0.0 for a single value)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _search_codebook(
    tokenizer, candidate_text_fn, n_choices: int, field_name: str = "<field>"
) -> tuple[list[str], bool]:
    """Search a codebook for the best alias set for one field.

    BOUNDED SEARCH (W5-A finding 2, was F1 greedy): tokenize each candidate
    row ONCE per code (a code's tokens do not depend on the other codes),
    build the prefix-conflict graph over the pool, and search for the best
    size-n prefix-free independent set with bounded backtracking. Greedy
    never revisited: an A-is-a-prefix-of-B/C/D pool raised even though
    {B, C, D} was valid. The lexicographic objective (branch nodes, max
    trie depth, max candidate tokens, token-length variance, code length)
    scores every COMPLETE set; the best set wins. Pools are small (36
    singles, 1296 pairs) and the visited-node cap (200k) bounds the work.

    Candidate codes: A-Z, then digits, then 2-char alphanumerics. Homogeneous
    pools first (letters, then digits, then mixed), then 2-char. DETERMINISTIC.

    Returns ``(codes, single_branch)``. Raises :class:`SchemaCompileError`
    when no valid set exists (F2: no silent fallback).
    """
    from jevmlx.trie import build_trie

    pair_codes = [
        a + b
        for a in _CODEBOOK_SINGLE + _CODEBOOK_DIGITS
        for b in _CODEBOOK_SINGLE + _CODEBOOK_DIGITS
    ]

    def _is_prefix_pair(a: list[int], b: list[int]) -> bool:
        return a[: len(b)] == b or b[: len(a)] == a

    def _greedy_pick(pool: list[str]) -> list[str] | None:
        # W5-A finding 2: BOUNDED backtracking search, not greedy. Greedy
        # never revisits: if code A is a token-prefix of B, C and D, it
        # commits to A, rejects B/C/D, and raises even though {B, C, D} is a
        # valid set. Pre-tokenize each candidate row once (a code's tokens do
        # not depend on the other codes), then search the conflict graph for
        # a size-n prefix-free independent set, scoring every COMPLETE set
        # with the existing lexicographic objective. Pools are small (36
        # singles, 1296 pairs); n <= 64; bounded by a visited-node cap.
        tokenized = {
            code: tokenizer.encode(candidate_text_fn(code), add_special_tokens=False)
            for code in pool
        }

        # Conflict iff one candidate's remainder (under the pair's common
        # prefix) equals or is a token-prefix of the other's — those two
        # codes can never coexist in a distinguishable set.
        def conflicts(a: str, b: str) -> bool:
            full_a, full_b = tokenized[a], tokenized[b]
            shared = _common_token_prefix([full_a, full_b])
            rem_a = full_a[len(shared) :]
            rem_b = full_b[len(shared) :]
            return _is_prefix_pair(rem_a, rem_b)

        best: tuple[tuple, list[str]] | None = None
        chosen: list[str] = []
        visited = 0
        # Bounded: 20k node visits per pool. Pools are priority-ordered and
        # the caller picks the lexicographic best across pools, so the cap
        # trades a hair of optimality for compile latency (the 26-choose-26
        # worst case must stay <1s).
        # Tighter cap for large n: the tree is C(pool, n)-shaped; for
        # n > 16 the exhaustive part must give way to the cap quickly.
        max_visits = 20_000 if n_choices <= 16 else 2_000

        def record(complete: list[str]) -> None:
            nonlocal best
            key = _score(complete)
            if best is None or key < best[0]:
                best = (key, list(complete))

        def backtrack(start: int) -> None:
            nonlocal visited
            if visited >= max_visits:
                return
            visited += 1
            if len(chosen) == n_choices:
                record(chosen)
                return
            # Bound 1: can we still reach n_choices from here?
            if len(pool) - start < n_choices - len(chosen):
                return
            # Bound 2: are there enough NON-CONFLICTING codes left? A node
            # whose residual conflict graph cannot supply the rest is cut
            # without visiting its subtree.
            remaining = [pool[i] for i in range(start, len(pool))]
            still_ok = sum(1 for c in remaining if not any(conflicts(c, x) for x in chosen))
            if still_ok < n_choices - len(chosen):
                return
            # Prune: if a complete set is already recorded with a better
            # possible prefix... (no cheap admissible bound beyond the pool
            # check; the visited cap bounds the search).
            for idx in range(start, len(pool)):
                code = pool[idx]
                if any(conflicts(code, c) for c in chosen):
                    continue
                chosen.append(code)
                backtrack(idx + 1)
                chosen.pop()
                if best is not None and visited >= max_visits:
                    return

        backtrack(0)
        return best[1] if best is not None else None

    def _score(codes: list[str]) -> tuple[int, int, int, float, int]:
        fulls = [tokenizer.encode(candidate_text_fn(c), add_special_tokens=False) for c in codes]
        shared = _common_token_prefix(fulls)
        remainders = [f[len(shared) :] for f in fulls]
        trie = build_trie(remainders)
        max_depth = max((len(n["path"]) for n in trie), default=0)
        return (
            len(trie),
            max_depth,
            max(len(r) for r in fulls),
            float(_variance([len(r) for r in fulls])),
            max(len(c) for c in codes),
        )

    single_pool = list(_CODEBOOK_SINGLE) + list(_CODEBOOK_DIGITS)
    pools = [list(_CODEBOOK_SINGLE), list(_CODEBOOK_DIGITS), single_pool, pair_codes]
    best: tuple[tuple, list[str]] | None = None
    for pool in pools:
        if not pool or n_choices > len(pool):
            continue
        # _greedy_pick now returns the best COMPLETE set it found (bounded
        # backtracking + lexicographic objective), or None.
        codes = _greedy_pick(pool)
        if codes is None:
            continue
        key = _score(codes)
        if best is None or key < best[0]:
            best = (key, codes)
        # Early exit on a single-branch-node set: 1 branch node is the
        # objective's first component at its minimum (n >= 2 needs exactly
        # one node), and pool order is priority — later pools can only tie
        # on node count while paying full search cost. Keeps the
        # 26-choice compile under a second (the C(36,26) backtracking tree
        # never runs to its cap); determinism unaffected (fixed pool
        # order). If no pool reaches 1 node, the full bounded search
        # across all pools still picks the lexicographic best.
        if key[0] == 1:
            best = (key, codes)
            break
    if best is None:
        raise SchemaCompileError(
            field_name,
            f"field '{field_name}': no codebook set of {n_choices} codes "
            f"tokenizes to distinguishable rows under tokenizer "
            f"{type(tokenizer).__name__}",
        )
    codes = best[1]
    fulls = [tokenizer.encode(candidate_text_fn(c), add_special_tokens=False) for c in codes]
    shared = _common_token_prefix(fulls)
    remainders = [f[len(shared) :] for f in fulls]
    trie = build_trie(remainders)
    single_branch = len(trie) <= 1
    return codes, single_branch


def _common_token_prefix(sequences: list[list[int]]) -> list[int]:
    """Longest common token-ID prefix of every sequence (at least one required)."""
    if not sequences:
        return []
    shared: list[int] = []
    shortest = min(len(sequence) for sequence in sequences)
    for position in range(shortest):
        tokens_at = {sequence[position] for sequence in sequences}
        if len(tokens_at) != 1:
            break
        shared.append(sequences[0][position])
    return shared


class SchemaCompileError(ValueError):
    """The plan compilers cannot produce a scorable plan for one field.

    Raised for token-identical choices, strict token-prefix continuations,
    candidates with no shared token prefix, and duplicate-choice keys.
    Carries the offending field name so linters can attribute the failure.
    """

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


@dataclasses.dataclass(frozen=True, slots=True)
class FieldDefinition:
    """One schema field, IMMUTABLE after construction (W5b-1, C7).

    Frozen dataclass: every attribute is set exactly once, at construction;
    any later mutation raises ``FrozenInstanceError``. The mutable inputs are
    deep-copied and re-frozen at the boundary: ``choices`` becomes a tuple,
    ``choice_descriptions`` a read-only mapping, and ``set_constraints`` —
    populated only through :meth:`compile_set_constraints`, which swaps the
    frozen field for a fresh one (validated copies, tuples inside) — is a
    tuple of read-only constraint views, so nothing reachable from a
    compiled schema can be mutated in place.
    """

    name: str
    field_type: str
    description: str
    choices: tuple[str, ...]
    choice_descriptions: Mapping[str, str]
    depends_on: str | None
    set_constraints: tuple[Mapping[str, Any], ...] = ()

    def __init__(
        self,
        name: str,
        field_type: str,
        description: str,
        choices: Sequence[str] | None = None,
        choice_descriptions: Mapping[str, str] | None = None,
        depends_on: str | None = None,
        set_constraints: tuple[Mapping[str, Any], ...] = (),
    ):
        # Frozen dataclass with a custom __init__: the validation logic is
        # the constructor's contract; freeze happens through object.__setattr__.
        field_type = field_type.lower()
        if field_type == "boolean":
            resolved_choices: tuple[str, ...] = ("true", "false")
            resolved_descriptions: Mapping[str, str] = MappingProxyType({})
        elif field_type == "multi":
            if not choices or len(choices) < 2:
                raise ValueError(f"Field '{name}' of type multi must have at least 2 choices.")
            if len(choices) > 64:
                raise ValueError(
                    f"Field '{name}' exceeds maximum cardinality of 64 choices "
                    f"for type multi (got {len(choices)})."
                )
            resolved_choices = tuple(choices)
            resolved_descriptions = _freeze_choice_descriptions(
                name, resolved_choices, choice_descriptions
            )
        elif field_type in ("enum", "choice", "selection"):
            if not choices or len(choices) == 0:
                raise ValueError(f"Field '{name}' of type enum must have choices defined.")
            if len(choices) > 255:
                raise ValueError(
                    f"Field '{name}' exceeds maximum cardinality of 255 choices "
                    f"(got {len(choices)})."
                )
            resolved_choices = tuple(choices)
            resolved_descriptions = _freeze_choice_descriptions(
                name, resolved_choices, choice_descriptions
            )
        else:
            raise ValueError(
                f"Unsupported field type '{field_type}'. "
                "Supported types: 'boolean', 'enum' and 'multi'."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "field_type", field_type)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "depends_on", depends_on)
        object.__setattr__(self, "set_constraints", set_constraints)
        object.__setattr__(self, "choices", resolved_choices)
        object.__setattr__(self, "choice_descriptions", resolved_descriptions)

        if self.field_type in ("multi", "enum", "choice", "selection"):
            seen: set[str] = set()
            for choice in resolved_choices:
                if choice in seen:
                    raise ValueError(
                        f"Field '{name}': duplicate choice '{choice}'; the engine "
                        "cannot distinguish duplicate choices"
                    )
                seen.add(choice)

    @property
    def cardinality(self) -> int:
        return len(self.choices)

    def compile_set_constraints(
        self, constraints: Sequence[Mapping[str, Any]] | None
    ) -> "FieldDefinition":
        """Validate W2-SETCONS hard set constraints and return a NEW frozen
        field carrying them (W5b-1, C7: the field is immutable, so "attach"
        means replace — :class:`StructuredSchema` swaps the field in its
        dict; a bare reference to the old field keeps the empty tuple).

        Accepted types (all validated at SET time — contradictory or
        malformed sets raise SchemaCompileError here, at compile time, never
        mid-run):

        - ``{"type": "mutually_exclusive", "options": [...]}`` — at most
          one option of the group may be selected. Alias of at_most_one
          with k=1 (kept as a distinct type so the prompt can name it).
        - ``{"type": "at_most_one", "options": [...]}`` — at most one.
        - ``{"type": "at_most_k", "options": [...], "k": int}`` — at most
          k options of the group.
        - ``{"type": "at_least_one", "options": [...]}`` — at least one.
        - ``{"type": "exact_k", "options": [...], "k": int}`` — exactly k.
        - ``{"type": "implies", "if_option": X, "then_option": Y}`` —
          selecting X forces Y in (between options of ONE multi field; the
          case-level implies between FIELDS lives in jevmlx/constraints.py
          and is untouched).

        Every referenced option must be a declared choice. Malformed
        constraints (unknown options, bad k, empty option lists, unknown
        types) raise SchemaCompileError immediately. Satisfiability is
        decided by running the same feasibility solver the engine uses
        (jevmlx.setcons, no scores): a constraint set with no feasible
        selection raises SchemaCompileError at compile time (W5-C finding
        17 — the old syntactic rules falsely rejected satisfiable sets like
        A->B + B->A, both absent).
        """
        if constraints is None:
            return self  # immutable: already constraint-free
        if self.field_type != "multi":
            raise SchemaCompileError(
                self.name,
                f"set constraints apply to multi fields only; field "
                f"'{self.name}' is of type '{self.field_type}'",
            )
        if not isinstance(constraints, list) or not constraints:
            raise SchemaCompileError(
                self.name,
                "set constraints must be a non-empty list of constraint dicts",
            )
        for i, c in enumerate(constraints):
            if not isinstance(c, dict):
                raise SchemaCompileError(
                    self.name, f"set constraint [{i}] must be a dict, got {type(c).__name__}"
                )
            ctype = c.get("type")
            if ctype in (
                "mutually_exclusive",
                "at_most_one",
                "at_least_one",
                "at_least_k",
                "exact_k",
                "at_most_k",
            ):
                options = c.get("options")
                if not isinstance(options, list) or not options:
                    raise SchemaCompileError(
                        self.name,
                        f"set constraint [{i}] ({ctype}): 'options' must be a "
                        "non-empty list of option names",
                    )
                unknown = [o for o in options if o not in self.choices]
                if unknown:
                    raise SchemaCompileError(
                        self.name,
                        f"set constraint [{i}] ({ctype}): options not in field "
                        f"choices: {', '.join(repr(o) for o in unknown)}",
                    )
                if len(set(options)) != len(options):
                    raise SchemaCompileError(
                        self.name,
                        f"set constraint [{i}] ({ctype}): duplicate options in {options!r}",
                    )
                if ctype in ("exact_k", "at_most_k", "at_least_k"):
                    k = c.get("k")
                    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                        raise SchemaCompileError(
                            self.name,
                            f"set constraint [{i}] ({ctype}): 'k' must be a non-negative int",
                        )
                    if k > len(options):
                        raise SchemaCompileError(
                            self.name,
                            f"set constraint [{i}] ({ctype}): k={k} exceeds the "
                            f"group size {len(options)} — unsatisfiable",
                        )
            elif ctype == "implies":
                if_opt = c.get("if_option")
                then_opt = c.get("then_option")
                for key, val in (("if_option", if_opt), ("then_option", then_opt)):
                    if not isinstance(val, str) or not val:
                        raise SchemaCompileError(
                            self.name,
                            f"set constraint [{i}] (implies): '{key}' must be a "
                            "non-empty option name",
                        )
                    if val not in self.choices:
                        raise SchemaCompileError(
                            self.name,
                            f"set constraint [{i}] (implies): '{key}' {val!r} is "
                            "not a declared choice of this field",
                        )
                if if_opt == then_opt:
                    raise SchemaCompileError(
                        self.name,
                        f"set constraint [{i}] (implies): if_option and "
                        "then_option are the same option; a self-implication "
                        "is vacuous and masks a config error",
                    )
            else:
                raise SchemaCompileError(
                    self.name,
                    f"set constraint [{i}].type must be one of "
                    f"['at_least_one', 'at_least_k', 'at_most_k', 'at_most_one', 'exact_k', "
                    f"'implies', 'mutually_exclusive'], got {ctype!r}",
                )
        # W5-C finding 17: NO syntactic contradiction rules. The old rules
        # falsely rejected satisfiable sets — A->B + B->A (both absent),
        # A->B + at_most_one(A,B) (A forbidden), A->B + exact_k([B], 0) (A
        # not selected) — and even an implies 2-cycle is satisfiable (both
        # absent). Satisfiability is decided by running the SAME feasibility
        # solver the engine uses (jevmlx.setcons, no scores, empty
        # proposal); a constraint set with no feasible selection is a
        # compile error.
        from jevmlx.setcons import is_feasible

        try:
            feasible = is_feasible(list(self.choices), list(constraints))
        except ValueError as exc:
            # F8: an over-large component (setcons.MAX_COMPONENT_OPTIONS)
            # raises from the solver — surface it as a compile error naming
            # the field, not a raw ValueError from deep inside the engine.
            raise SchemaCompileError(
                self.name,
                f"set constraint enumeration cap exceeded: {exc}",
            ) from exc

        if not feasible:
            raise SchemaCompileError(
                self.name,
                f"set constraint contradiction: no selection of "
                f"{list(self.choices)} satisfies {constraints!r}",
            )
        # W5b-1 (C7): deep-freeze the output — one validated copy per
        # constraint, wrapped read-only, in an immutable tuple. Nothing
        # reachable from the compiled field is caller-mutable. Idiomatic
        # clone on a frozen dataclass: dataclasses.replace with the
        # set_constraints field passed through the custom __init__.
        frozen = tuple(MappingProxyType(dict(c)) for c in constraints)
        return dataclasses.replace(self, set_constraints=frozen)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.field_type,
            "description": self.description,
            "choices": self.choices,
            "cardinality": self.cardinality,
            "choice_descriptions": self.choice_descriptions,
        }
        if self.depends_on is not None:
            d["depends_on"] = self.depends_on
        if self.set_constraints:
            d["set_constraints"] = list(self.set_constraints)
        return d


class StructuredSchema:
    """Immutable schema (W5b-1, C7): field set and field definitions are
    frozen after construction.

    ``fields`` exposes a read-only mapping (insert/clear/pop raise);
    ``FieldDefinition`` attributes are frozen. Mutation attempts raise in
    tests and at runtime — a schema is compiled once and shared across
    engines, threads, and compiled-plan caches; in-place mutation would
    silently invalidate every cached plan keyed to it.
    """

    def __init__(self, schema_dict: dict[str, Any]):
        fields: dict[str, FieldDefinition] = {}
        for field_name, spec in schema_dict.items():
            if "." in field_name:
                # Field names become JSON row keys (json.dumps'd at candidate
                # build time); a dot inside a field name could collide with
                # another field's literal name (C3: row keys must be
                # injective).
                raise ValueError(
                    f"Field name '{field_name}' contains '.'; dot-free field "
                    "names keep multi row keys injective"
                )
            if "/" in field_name:
                # Multi option rows are keyed '<field>/<code>' where code is
                # the option's zero-padded 2-digit index in choices order
                # (W2-E row codes); a slash inside a field name would collide
                # across fields (C3: row keys must be injective).
                raise ValueError(
                    f"Field name '{field_name}' contains '/'; slash-free field "
                    "names keep multi option row keys '<field>/<code>' injective"
                )
            if "#" in field_name:
                # The count row for a multi field is keyed '<field>#count'
                # (W2-E step 3); a '#' inside a field name would collide with
                # a field's count row (C3: row keys must be injective).
                raise ValueError(
                    f"Field name '{field_name}' contains '#'; hash-free field "
                    "names keep the '<field>#count' count-row keys injective"
                )
            fields[field_name] = FieldDefinition(
                name=field_name,
                field_type=spec.get("type", "enum"),
                description=spec.get("description", ""),
                choices=spec.get("choices", None),
                choice_descriptions=spec.get("choice_descriptions", None),
                depends_on=spec.get("depends_on", None),
            )
            if spec.get("set_constraints"):
                # W2-SETCONS: schema-dict-declared set constraints are
                # validated at construction (compile time) — contradictory
                # sets raise SchemaCompileError before any engine runs.
                # W5b-1 (C7): the field is frozen; compile_set_constraints
                # returns a NEW frozen field — swap it into the dict.
                fields[field_name] = fields[field_name].compile_set_constraints(
                    spec["set_constraints"]
                )

        # W5-B (review 5): depends_on must name an existing earlier field and
        # must never target a multi child — dependency conditioning scores a
        # scalar candidate row for the child; a multi child has per-option
        # Y/N rows and no joint set solver exists yet. Also reject cycles:
        # the selective second pass runs in topological waves, which need a
        # DAG.
        for field_name, fd in fields.items():
            dep = fd.depends_on
            if dep is None:
                continue
            if dep not in fields:
                raise SchemaCompileError(
                    field_name,
                    f"field '{field_name}': depends_on references unknown field '{dep}'",
                )
            if dep == field_name:
                raise SchemaCompileError(field_name, f"field '{field_name}': depends_on itself")
            if fields[dep].field_type == "multi":
                raise SchemaCompileError(
                    field_name,
                    f"field '{field_name}': depends_on '{dep}' is a multi field; "
                    "dependency conditioning on multi parents is not supported "
                    "(no joint set solver) — declare the parent as an enum or boolean",
                )
            if fd.field_type == "multi":
                raise SchemaCompileError(
                    field_name,
                    f"field '{field_name}' is a multi field with depends_on; "
                    "conditioned multi scoring is not supported yet — drop "
                    "depends_on or make the field an enum/boolean",
                )
        # Cycle check (review 10: topological waves need a DAG).
        _TOPSORT_STATE = {}
        for start in fields:
            if _TOPSORT_STATE.get(start) == 2:
                continue
            stack = [
                (
                    start,
                    iter(fd.depends_on for fd in [fields[start]])
                    if fields[start].depends_on
                    else iter(()),
                )
            ]
            _TOPSORT_STATE[start] = 1
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    state = _TOPSORT_STATE.get(nxt, 0)
                    if state == 1:
                        raise SchemaCompileError(node, f"depends_on cycle through '{nxt}'")
                    if state == 0:
                        _TOPSORT_STATE[nxt] = 1
                        nxt_deps = [fields[nxt].depends_on] if fields[nxt].depends_on else []
                        stack.append((nxt, iter(nxt_deps)))
                        advanced = True
                        break
                if not advanced:
                    _TOPSORT_STATE[node] = 2
                    stack.pop()

        # Compiled plans, keyed by tokenizer OBJECT IDENTITY (P2: a
        # WeakKeyDictionary keys by __eq__/__hash__, so two equal-but-distinct
        # tokenizers would wrongly share one plan). dict[id] =
        # (weakref.ref(tok), plan); weakref.finalize evicts the entry when the
        # tokenizer dies, so an id can never be reused by a live object while
        # its entry lingers. One schema object can be reused with several
        # models, and token IDs are tokenizer-specific.
        # W5b-1 (C7): the field set is frozen after construction — exposed
        # as a read-only mapping over the built dict.
        self.fields: Mapping[str, FieldDefinition] = MappingProxyType(fields)
        self._plans: dict[int, tuple[weakref.ref, dict[str, Any]]] = {}
        self._logged_non_weakref = False

    def get_field_names(self) -> list[str]:
        return list(self.fields.keys())

    def __getitem__(self, key: str) -> FieldDefinition:
        return self.fields[key]

    def __len__(self) -> int:
        return len(self.fields)

    def to_json_schema_prompt_str(self) -> str:
        """Returns a clean TypeScript/JSON schema representation for naive LLM prompting."""
        lines = ["{"]
        for name, field in self.fields.items():
            if field.field_type == "boolean":
                lines.append(f'  "{name}": boolean, // {field.description}')
            elif field.field_type == "multi":
                choices_str = " | ".join(f'"{c}"' for c in field.choices)
                lines.append(
                    f'  "{name}": [{choices_str}], // {field.description} (select all that apply)'
                )
            else:
                # Every choice, always: the parallel path sees all choices,
                # so a naive baseline that truncates large enums would not
                # be a fair comparison.
                choices_str = " | ".join(f'"{c}"' for c in field.choices)
                lines.append(f'  "{name}": {choices_str}, // {field.description}')
        lines.append("}")
        return "\n".join(lines)

    def _multi_field_header(self, field: FieldDefinition) -> str:
        """Schema-block text for one multi field: options as a described
        yes/no menu mapping code = option (W2-E row codes: the engine's
        decision rows are keyed '<field>/<code>', so the model must see the
        exact code for every option) with explicit Y/N meanings (Q6-6: the
        scorer expects quoted Y/N), plus the count question (W2-E step 3:
        the engine always asks '<field>#count' how many options apply and
        the model must see the exact answer codes 0..4, where 4 means
        # four or more). Every displayed
        name, label, option and gloss is json.dumps-escaped so
        quotes/newlines cannot break the schema block (Q2 'System text' /
        'Schema block format')."""
        parts = []
        for i, choice in enumerate(field.choices):
            safe_choice = json_text(choice)
            gloss = field.choice_descriptions.get(choice)
            gloss_part = f" — {json_text(gloss)}" if gloss else ""
            parts.append(f"{self.code_for_index(i)} = {safe_choice}{gloss_part}")
        return (
            "; ".join(parts)
            + " — how many of these apply? Answer one of "
            + ", ".join(json_text(c) for c in COUNT_CODES)
            + " ("
            + json_text(COUNT_CODES[-1])
            + " means four or more)."
        )

    @staticmethod
    def code_for_index(index: int) -> str:
        """The 2-digit zero-padded row code for the option at ``index``
        (choices order: 00, 01, ...; FieldDefinition caps multi at 64
        choices, so two digits always suffice)."""
        return f"{index:02d}"

    def to_schema_str(self, mode: str = "slots", tokenizer=None) -> str:
        """Schema block for prompt v2, rendered per scoring mode.

        mode ``"slots"`` lists each field's choices under the aliases THE
        COMPILED PLAN scored for this tokenizer (``A) <choice>`` when the
        search picks the index aliases, ``0) <choice>`` when it picks
        digits, etc.), plus `` — <gloss>`` when a gloss exists; mode
        ``"labels"`` lists the real choice strings — exactly the text the
        scorer reads.

        W5-A finding 1: the compiled plan OWNS the displayed aliases. The
        old independent ``_alias_code(i)`` rendering path taught the model
        one protocol (A, B, C...) while the scorer judged another (whatever
        ``_search_codebook`` picked, e.g. digits). Prompt and scorer can no
        longer disagree because they are the same data: the slot block is
        rendered from ``plan['fields'][name]['aliases']``.

        ``tokenizer`` is REQUIRED for mode ``"slots"`` (the plan must be
        compiled to know the searched codes) and ignored for ``"labels"``.

        Multi fields render once as a described yes/no menu — the field
        header states that each option is answered yes or no; the per-option
        decision rows below (``"<field>/<code>"``) are answered with Y/N.
        """
        if mode not in ("slots", "labels"):
            raise ValueError(f"mode must be 'slots' or 'labels', got {mode!r}")
        slot_plan: dict[str, dict[str, Any]] | None = None
        if mode == "slots":
            if tokenizer is None:
                raise ValueError(
                    "to_schema_str(mode='slots') needs the tokenizer: the "
                    "displayed aliases come from the COMPILED plan (W5-A "
                    "finding 1 — the prompt must show the codebook the "
                    "scorer reads), and that is tokenizer-specific"
                )
            slot_plan = self._cached_plan(tokenizer, mode="slots") or self.compile_slot_plan(
                tokenizer
            )
        lines = []
        for name, field in self.fields.items():
            safe_name = json_text(name)
            desc = field.description.split("\n")[0].strip()
            safe_desc = json_text(desc)
            if field.field_type == "multi":
                menu = self._multi_field_header(field)
                lines.append(
                    f"  {safe_name}: {menu}  // {safe_desc} (select all that apply; "
                    'each coded option is answered "Y" = applies or "N" = does not apply)'
                )
                continue
            choices_list = (
                ["true", "false"] if field.field_type == "boolean" else list(field.choices)
            )
            if mode == "slots":
                # W5-A finding 1: display the aliases the compiled plan
                # scored (tokenizer-specific searched codes), never an
                # independent index-derived rendering.
                assert slot_plan is not None
                displayed_aliases = slot_plan["fields"][name]["aliases"]
                assert len(displayed_aliases) == len(choices_list), (
                    f"plan aliases for {name!r} do not cover the choices"
                )
                parts = []
                for i, choice in enumerate(choices_list):
                    alias = displayed_aliases[i]
                    safe_choice = json_text(choice)
                    gloss = field.choice_descriptions.get(choice)
                    gloss_part = f" — {json_text(gloss)}" if gloss else ""
                    parts.append(f"{alias}) {safe_choice}{gloss_part}")
            else:
                parts = []
                for choice in choices_list:
                    safe_choice = json_text(choice)
                    gloss = field.choice_descriptions.get(choice)
                    gloss_part = f" — {json_text(gloss)}" if gloss else ""
                    parts.append(f"{safe_choice}{gloss_part}")
            lines.append(f"  {safe_name}: {'  '.join(parts)}  // {safe_desc}")
        return "\n".join(lines)

    def to_alias_schema_str(self, tokenizer) -> str:
        """Schema block in slots mode: aliases from the COMPILED plan for
        ``tokenizer`` (W5-A finding 1 — the prompt shows the codebook the
        scorer reads)."""
        return self.to_schema_str("slots", tokenizer=tokenizer)

    def to_labels_schema_str(self) -> str:
        """Schema block in labels mode (real choice strings)."""
        return self.to_schema_str("labels")

    @staticmethod
    def alias_for_index(index: int) -> str:
        """The index-derived alias (A, B, ..., AA, AB...).

        W5-A finding 1: this is NO LONGER the prompt's displayed alias —
        prompts render from the compiled plan. It remains only for the
        OpenAI-compatible endpoint adapter's per-field requests, which do
        not go through the compiled plan (openai_slots.py:113)."""
        return _alias_code(index)

    def plan_hash(self, tokenizer, mode: str) -> str:
        """sha256 of the compiled plan for ``mode`` (stable within a process).

        The plan captures the schema block, choice order, and token
        segmentation the scoring pass depends on — everything the neutral
        prior must match. Compiled on demand; the result equals the hash of
        ``json.dumps(plan, sort_keys=True)`` with token ids as ints.

        W5b-1 (C7): plans are read-only mappings; they are converted to
        plain containers here (mappingproxy→dict) so the hash sees the
        plan's CONTENT — the stdlib default (``default=list``) would
        collapse every mappingproxy to its key list and hash slots/labels
        plans identically.
        """
        if mode == "slots":
            plan = self.compile_slot_plan(tokenizer)
        elif mode == "labels":
            plan = self.compile_labels_plan(tokenizer)
        else:
            raise ValueError(f"mode must be 'slots' or 'labels', got {mode!r}")

        def thaw(value: Any) -> Any:
            if isinstance(value, dict) or isinstance(value, MappingProxyType):
                return {k: thaw(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [thaw(v) for v in value]
            return value

        return hashlib.sha256(
            json.dumps(thaw(plan), sort_keys=True, default=list).encode("utf-8")
        ).hexdigest()

    def _cached_plan(self, tokenizer, mode: str) -> dict[str, Any] | None:
        """Return the live cached plan for (tokenizer, mode), or None.

        Identity check: the stored ref() must resolve to THIS tokenizer, not
        just an equal one (P2 — two equal-but-distinct tokenizers must not
        share a plan; token ids are tokenizer-specific). Callers compile and
        call _cache_plan when this returns None.
        """
        entry = self._plans.get((id(tokenizer), mode))
        if entry is not None and entry[0]() is tokenizer:
            return entry[1]
        return None

    def _cache_plan(self, tokenizer, plan: dict[str, Any], mode: str) -> None:
        """Store a (pre-frozen, read-only) plan keyed by tokenizer identity,
        evicted on tokenizer death (W5b-1, C7: compiled plans are frozen at
        compile time — mutating a returned plan raises).

        Non-weak-referenceable tokenizers are not cached at all (N3): an
        id()-keyed entry without a liveness check could be returned for a
        different object after id reuse. The finalizer is registered only
        when the key is new — re-storing an existing key must not stack
        finalizer objects on the tokenizer.
        """
        try:
            ref = weakref.ref(tokenizer)
        except TypeError:
            if not self._logged_non_weakref:
                _LOGGER.debug(
                    "tokenizer %s is not weak-referenceable; plan cache disabled "
                    "for it (compiling on every call)",
                    type(tokenizer).__name__,
                )
                self._logged_non_weakref = True
            return
        key = id(tokenizer)
        cache_key = (key, mode)
        if cache_key not in self._plans:
            weakref.finalize(tokenizer, self._plans.pop, cache_key, None)
        self._plans[cache_key] = (ref, plan)

    def compile_slot_plan(self, tokenizer) -> dict[str, dict[str, Any]]:
        """Slot-trie plan (the default scoring mode): the decision row stays
        JSON — ``'{\n  "<field>": '`` — and the scored candidates are the
        QUOTED neutral aliases ``'"A"'``, `'"B"'``, ... (base-26 codes beyond
        26 choices), mapped back to the real choice strings on assembly.

        Multi fields share the same per-option yes/no rows in both modes
        (see compile_labels_plan): the row text is the natural question and
        the scored candidates are the quoted "Y"/"N" aliases. Booleans get
        aliases too (A -> true, B -> false).

        Returns the same shape as the labels plan
        (``{"lead_in_ids": ..., "fields": {fname: plan}}``) with per-field
        ``alias_map`` ({alias: real value}) added; the engine scores
        ``shared_ids``/``remainders`` through the token trie unchanged and
        maps winners back via ``alias_map``.
        """
        cached = self._cached_plan(tokenizer, "slots")
        if cached is not None:
            return cached
        fields_plan: dict[str, dict[str, Any]] = {}

        def slot_candidate_text(name: str, alias: str) -> str:
            """The complete one-field JSON object for one alias row (W2-B:
            the candidate row protocol is the complete object, not a
            dangling '{\n  "field": "A",\n')."""
            return json_text({name: alias})

        # Lead-in candidates: scalar fields' shared prefixes. Computed after
        # the per-field plans exist (same two-pass shape as labels mode).
        for fname, fdef in self.fields.items():
            if fdef.field_type == "multi":
                continue
            if fdef.field_type == "boolean":
                values = ["true", "false"]
            else:
                values = list(fdef.choices)
            from functools import partial

            aliases, single_branch = _search_codebook(
                tokenizer,
                partial(slot_candidate_text, fname),
                len(values),
                field_name=fname,
            )
            alias_map = dict(zip(aliases, values, strict=True))

            candidates = [
                tokenizer.encode(slot_candidate_text(fname, alias), add_special_tokens=False)
                for alias in aliases
            ]
            shared = _common_token_prefix(candidates)
            remainders = [full[len(shared) :] for full in candidates]
            for i, remainder in enumerate(remainders):
                for j, other in enumerate(remainders):
                    if i == j or other[: len(remainder)] != remainder:
                        continue
                    if other == remainder:
                        raise SchemaCompileError(
                            fname,
                            f"field '{fname}': aliases '{aliases[i]}' and '{aliases[j]}' "
                            "are token-identical; the engine cannot distinguish them",
                        )
                    raise SchemaCompileError(
                        fname,
                        f"field '{fname}': alias '{aliases[i]}' is a strict token-prefix "
                        f"of '{aliases[j]}' in token space; the engine would never "
                        "distinguish them",
                    )
            if not shared and len({r[0] for r in remainders}) > 1:
                raise SchemaCompileError(
                    fname,
                    f"field '{fname}': alias candidates share no token prefix "
                    f"(tokenizer {type(tokenizer).__name__}); cannot place the "
                    "decision row",
                )
            fields_plan[fname] = {
                "shared_ids": shared,
                "remainders": remainders,
                "alias_map": alias_map,
                "aliases": aliases,
                "choices": values,
                # W2-C telemetry: the searched codebook, whether a search
                # validated (False = index fallback), and whether the winning
                # set scores through a single branch node.
                "codebook": list(aliases),
                "single_branch": single_branch,
            }

        if any(f.field_type == "multi" for f in self.fields.values()):
            multi_plan = self._compile_multi_plan(tokenizer)
            for fname, fdef in self.fields.items():
                if fdef.field_type == "multi":
                    fields_plan[fname] = multi_plan[fname]
        # Factor the global lead-in once over scalar shared_ids AND multi
        # suffix_ids_list (B1/Q6-1: the lead-in must be the common prefix of
        # ALL row prefixes, never computed from scalars alone). Strip exactly
        # once, after the complete final mode plan exists.
        row_prefixes = (
            [p["shared_ids"] for p in fields_plan.values() if "shared_ids" in p]
            + [
                ids
                for p in fields_plan.values()
                if "suffix_ids_list" in p
                for ids in p["suffix_ids_list"]
            ]
            + [
                # W2-E step 3: the count row is a real scored row — its shared
                # prefix must join the lead-in candidates (the count row's
                # shared_ids start with the same '{\n  "field#' text).
                p["count"]["shared_ids"]
                for p in fields_plan.values()
                if "count" in p
            ]
        )
        lead_in = _common_token_prefix(row_prefixes) if row_prefixes else []
        if lead_in:
            for p in fields_plan.values():
                if "shared_ids" in p:
                    p["shared_ids"] = p["shared_ids"][len(lead_in) :]
                if "suffix_ids_list" in p:
                    # lead_in is the common prefix of all row_prefixes by
                    # construction — strip unconditionally, no fallback.
                    p["suffix_ids_list"] = [ids[len(lead_in) :] for ids in p["suffix_ids_list"]]
                if "count" in p:
                    p["count"]["shared_ids"] = p["count"]["shared_ids"][len(lead_in) :]
        result = {"lead_in_ids": list(lead_in), "fields": fields_plan}
        result = _freeze_plan(result)  # W5b-1 (C7): read-only, frozen once
        self._cache_plan(tokenizer, result, mode="slots")
        return result

    def compile_labels_plan(self, tokenizer) -> dict[str, dict[str, Any]]:
        """Labels scoring plan (choice-text trie): candidates are the real
        choice strings; the decision row is the full JSON row text. The
        engine maps winners straight to the choice strings (no alias hop).
        """
        return self._compile_labels(tokenizer)

    def _compile_multi_plan(self, tokenizer) -> dict[str, dict[str, Any]]:
        """Build ONLY the multi-field plans, returning UNSTRIPPED full option
        prefixes (suffix_ids_list + remainders per option, before any
        schema-wide lead-in stripping).

        This is the private multi-plan builder shared by both scoring modes.
        compile_slot_plan calls this instead of compile_labels_plan (Q6-1/Q6-2:
        slot mode must not invoke the whole labels compiler — a scalar
        real-label collision should not fail slot mode, and the multi option
        prefixes must NOT have the labels lead-in pre-stripped).
        """
        plan: dict[str, dict[str, Any]] = {}

        def candidate_text(name: str, value_text: str) -> str:
            """The complete one-field JSON object for one row (W2-B: the
            candidate row protocol is the complete object, not a dangling
            '{\n  "field": value,\n'). value_text is a pre-serialized JSON
            value (e.g. '"LOW"', 'true'), so we build the object string
            directly rather than double-encoding through json.dumps."""
            return "{" + f"{json_text(name)}: {value_text}" + "}"

        for fname, fdef in self.fields.items():
            if fdef.field_type != "multi":
                continue
            suffix_ids_list = []
            remainders_per_option = []
            codes = []
            for oi, option in enumerate(fdef.choices):
                # W2-E row codes: the decision row key is '<field>/<code>'
                # (code = zero-padded 2-digit choices-order index), not
                # '<field>/<option>' — the raw option text never reaches the
                # scored token stream, so hostile/garbage option strings can
                # no longer fragment or collide the rows (bug 18). The
                # prompt's schema block maps code = option (same choices
                # order), and telemetry maps codes back to option names.
                code = self.code_for_index(oi)
                codes.append(code)
                pair = [
                    tokenizer.encode(
                        candidate_text(f"{fname}/{code}", f'"{alias}"'),
                        add_special_tokens=False,
                    )
                    for alias in ("Y", "N")
                ]
                option_shared = _common_token_prefix(pair)
                option_remainders = [full[len(option_shared) :] for full in pair]
                if not option_shared:
                    raise SchemaCompileError(
                        fname,
                        f"field '{fname}': option '{code}' ({option!r}) Y/N "
                        f"candidates share no token prefix (tokenizer "
                        f"{type(tokenizer).__name__}); cannot place the "
                        "decision row",
                    )
                if option_remainders[0] == option_remainders[1]:
                    raise SchemaCompileError(
                        fname,
                        f"field '{fname}': option '{code}' ({option!r}) tokenizes to "
                        "identical Y/N candidates; the engine cannot "
                        "distinguish them",
                    )
                if len(option_remainders[0]) < len(option_remainders[1]):
                    shorter, longer = option_remainders
                    short_name, long_name = "Y", "N"
                else:
                    shorter, longer = option_remainders[1], option_remainders[0]
                    short_name, long_name = "N", "Y"
                if longer[: len(shorter)] == shorter:
                    raise SchemaCompileError(
                        fname,
                        f"field '{fname}': option '{code}' ({option!r}) has a strict "
                        f"token-prefix continuation ({short_name} is a prefix of "
                        f"{long_name} in token space); the engine would never "
                        "distinguish them",
                    )
                suffix_ids_list.append(option_shared)
                remainders_per_option.append(option_remainders)
            # W2-E step 3: the COUNT row. One extra row per multi field asking
            # how many options apply, scored like a scalar enum: the
            # candidates are the quoted count codes ('0', '1', '2', '3',
            # '4' — 4 = four or more) scored through the same trie
            # machinery as any scalar field. The row key '<field>#count' is
            # injective (C3: '#' is rejected in field names) and never
            # collides with '<field>/<code>' option rows. The count is a
            # RECONCILIATION signal only: the engine gates its use on the
            # row's top-2 margin (COUNT_MARGIN_MIN) and falls back to the
            # per-option rule otherwise.
            count_shared_ids = []
            for count_code in COUNT_CODES:
                candidate = tokenizer.encode(
                    candidate_text(count_key(fname), f'"{count_code}"'),
                    add_special_tokens=False,
                )
                count_shared_ids.append(candidate)
            count_shared = _common_token_prefix(count_shared_ids)
            count_remainders = [full[len(count_shared) :] for full in count_shared_ids]
            for i, remainder in enumerate(count_remainders):
                for j, other in enumerate(count_remainders):
                    if i == j or other[: len(remainder)] != remainder:
                        continue
                    if other == remainder:
                        raise SchemaCompileError(
                            fname,
                            f"field '{fname}': count codes '{COUNT_CODES[i]}' and "
                            f"'{COUNT_CODES[j]}' are token-identical; the engine "
                            "cannot distinguish them",
                        )
                    raise SchemaCompileError(
                        fname,
                        f"field '{fname}': count code '{COUNT_CODES[i]}' is a strict "
                        f"token-prefix of '{COUNT_CODES[j]}' in token space; the "
                        "engine would never distinguish them",
                    )
            if not count_shared and len({r[0] for r in count_remainders}) > 1:
                raise SchemaCompileError(
                    fname,
                    f"field '{fname}': count candidates share no token prefix "
                    f"(tokenizer {type(tokenizer).__name__}); cannot place the "
                    "count row",
                )
            plan[fname] = {
                "options": list(fdef.choices),
                # W2-E row codes, choices order — telemetry maps code ->
                # option via this list.
                "codes": codes,
                "suffix_ids_list": suffix_ids_list,
                "remainders": remainders_per_option,
                # W2-E step 3: the always-on count row (built above). Its
                # shared_ids/remainders ride the field plan; the engine
                # scores it like a scalar enum and reconciles with the
                # per-option rule through COUNT_MARGIN_MIN.
                "count": {
                    "shared_ids": count_shared,
                    "remainders": count_remainders,
                    "codes": list(COUNT_CODES),
                },
            }
        return plan

    def _compile_labels(self, tokenizer) -> dict[str, dict[str, Any]]:  # noqa: D401
        """Pre-index everything the engine needs for the batched suffix pass.

        Token-aligned at both boundaries: every choice is encoded as ONE
        tokenization of its complete assistant-tail candidate — ``{\n`` + the
        JSON row text + ``,\n`` (the row text being ``  "name": "choice"``
        or the bare literal, built with json.dumps so quotes and backslashes
        survive). Tokenizing a prefix and its remainder separately would be
        wrong: BPE merges are not compositional, so the concatenated ids would
        not be the tokenization of the full candidate. The prefill prompt ends
        exactly at the chat template's generation marker; ``{\n`` belongs to
        the candidate tail, not the prompt.

        Per enum/boolean field the plan carries ``shared_ids`` (the common
        token-ID prefix across the candidates of ALL fields — typically the
        ``{\n  "`` lead-in) and per-choice ``remainders``.
        Per multi field it carries one ``suffix_ids_list`` entry per option —
        that option's shared token prefix (everything before its yes/no
        divergence, i.e. the row the engine runs) — and ``remainders`` with
        the option's two Y/N continuations. The returned mapping is
        ``{"lead_in_ids": [...], "fields": {field_name: plan}}`` — metadata
        sits beside the field plans, never inside them (D1: a field could
        legally be named "_lead_in_ids"). Plans are cached per tokenizer
        identity (name_or_path + vocab size).
        """
        cached = self._cached_plan(tokenizer, "labels")
        if cached is not None:
            return cached
        plan: dict[str, dict[str, Any]] = {}

        def candidate_text(name: str, value_text: str) -> str:
            """The complete one-field JSON object for one row (W2-B: the
            candidate row protocol is the complete object, not a dangling
            '{\n  "field": value,\n'). value_text is a pre-serialized JSON
            value (e.g. '"Y"', '"N"'), so we build the object string
            directly rather than double-encoding through json.dumps.

            The row key is always json.dumps(name) (C3: no raw interpolation;
            field names are dot- and slash-free, so '<field>.<option>' scalar
            keys and '<field>/<option>' option keys are injective across
            (field, option) pairs and field names).
            """
            return "{" + f"{json_text(name)}: {value_text}" + "}"

        multi_plan = self._compile_multi_plan(tokenizer)
        for fname, fdef in self.fields.items():
            if fdef.field_type == "multi":
                # Built by the shared private multi-plan builder (Q6-1: same
                # UNSTRIPPED prefixes used by slot mode; the lead-in strip
                # happens once, below, after the full plan exists).
                plan[fname] = multi_plan[fname]
                continue

            if fdef.field_type == "boolean":
                value_texts = ["true", "false"]
            else:
                value_texts = [json_text(choice) for choice in fdef.choices]
            candidates = [
                tokenizer.encode(candidate_text(fname, value_text), add_special_tokens=False)
                for value_text in value_texts
            ]

            shared = _common_token_prefix(candidates)
            remainders = [full[len(shared) :] for full in candidates]
            for i, remainder in enumerate(remainders):
                for j, other in enumerate(remainders):
                    if i == j or other[: len(remainder)] != remainder:
                        continue
                    if other == remainder:
                        raise SchemaCompileError(
                            fname,
                            f"field '{fname}': choices '{fdef.choices[i]}' and "
                            f"'{fdef.choices[j]}' are token-identical; the engine "
                            "cannot distinguish them",
                        )
                    raise SchemaCompileError(
                        fname,
                        f"field '{fname}': choice '{fdef.choices[i]}' is a strict "
                        f"token-prefix of '{fdef.choices[j]}' in token space; the "
                        "engine would never distinguish them",
                    )
            if not shared and len({r[0] for r in remainders}) > 1:
                # After lead-in removal this field would have an empty row: a
                # branch directly at the generation boundary cannot be scored
                # by a broadcast row (B2).
                raise SchemaCompileError(
                    fname,
                    f"field '{fname}': candidates share no token prefix "
                    f"(tokenizer {type(tokenizer).__name__}); cannot place the "
                    "decision row",
                )
            plan[fname] = {
                "shared_ids": shared,
                "remainders": remainders,
            }

        # Schema-wide lead-in (typically '{\n  "') shared by every field's
        # candidates: lifted out of shared_ids so the engine can keep it in
        # the prefill broadcast cache. Remainders stay relative to the full
        # per-field shared prefix; rows are lead_in + shared_ids + path.
        # Lead-in candidates: every row prefix — scalar fields' shared_ids
        # AND multi option prefixes (B1: with only a multi field, the lead-in
        # must still be the common prefix of the option rows, never their
        # longer per-option text).
        field_shared_prefixes = (
            [p["shared_ids"] for p in plan.values() if "shared_ids" in p]
            + [ids for p in plan.values() if "suffix_ids_list" in p for ids in p["suffix_ids_list"]]
            + [
                # W2-E step 3: the count row is a real scored row — its shared
                # prefix joins the lead-in candidates.
                p["count"]["shared_ids"]
                for p in plan.values()
                if "count" in p
            ]
        )
        if not field_shared_prefixes:
            wrapped: dict[str, Any] = {"lead_in_ids": [], "fields": plan}
            self._cache_plan(tokenizer, wrapped, mode="labels")
            return wrapped
        lead_in = _common_token_prefix(field_shared_prefixes)
        # An empty schema-wide lead-in is legal (e.g. char-level tokenizers
        # where '{\n' fuses with the field name): the engine then runs one row
        # per field with no broadcast prefix — each row still carries that
        # field's full shared_ids.
        # Apply the same strip to multi option prefixes so the engine can
        # prepend lead_in uniformly to every row (R1: one rule for all rows).
        # lead_in is the common prefix by construction — strip unconditionally.
        for p in plan.values():
            if "suffix_ids_list" in p and lead_in:
                p["suffix_ids_list"] = [ids[len(lead_in) :] for ids in p["suffix_ids_list"]]
            if "count" in p and lead_in:
                p["count"]["shared_ids"] = p["count"]["shared_ids"][len(lead_in) :]
        for p in plan.values():
            if "shared_ids" in p:
                p["shared_ids"] = p["shared_ids"][len(lead_in) :]
        # Metadata lives beside the field plans, never mixed into them (D1:
        # a field could legally be named "_lead_in_ids").
        result = {"lead_in_ids": list(lead_in), "fields": plan}
        result = _freeze_plan(result)  # W5b-1 (C7): read-only, frozen once
        self._cache_plan(tokenizer, result, mode="labels")
        return result
