"""Case-level constraint checking (EV1 / W3-D).

Single source of truth for constraint semantics — imported by both
:mod:`jevmlx.engine` (constrained MAP) and :mod:`jevmlx.evalmetrics`
(constraint_violation_rate).

Constraint types (mirroring EV1's declarative shape):
    - implies / requires_parent: parent -> child mapping
    - excludes: field==value -> other must be empty/falsy
    - exclusivity: at most one of the group options in a multi field
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "check_constraint",
    "validate_constraints",
    "validate_constraints_for_schema",
    "compile_constraints",
    "CompiledImplication",
    "CompiledExclusion",
    "CompiledExclusivity",
    "CompiledConstraints",
    "ConstraintComponent",
    "ConstraintError",
]


class ConstraintError(ValueError):
    """A case-level constraint is invalid against the schema (W5-B review 12).

    Raised BEFORE any model work: unknown types, unknown fields, values
    outside the field's domain and multi-field case constraints all fail
    here. There is no 'unknown means satisfied' fallback anywhere.
    """


def check_constraint(constraint: dict, assignment: dict[str, object]) -> bool:
    """Return True if the constraint is SATISFIED, False if VIOLATED.

    ``assignment`` maps field name -> predicted/labelled value. Multi
    fields carry a list; scalar fields carry a single value.
    """
    ctype = constraint.get("type")
    if ctype in ("implies", "requires_parent"):
        parent = constraint.get("parent")
        child = constraint.get("child")
        mapping = constraint.get("mapping", {})
        parent_val = assignment.get(parent)
        child_val = assignment.get(child)
        if parent_val is None or child_val is None:
            return True  # unmeasured field can't violate
        allowed = mapping.get(parent_val, [])
        return child_val in allowed
    if ctype == "excludes":
        field = constraint.get("field")
        value = constraint.get("value")
        other = constraint.get("other")
        field_val = assignment.get(field)
        other_val = assignment.get(other)
        if field_val is None or other_val is None:
            return True
        if field_val == value:
            # When field==value, other must be empty/falsy
            if isinstance(other_val, list):
                return len(other_val) == 0
            return other_val in (None, "", False)
        return True
    if ctype == "exclusivity":
        field = constraint.get("field")
        options = set(constraint.get("options", []))
        field_val = assignment.get(field)
        if field_val is None:
            return True
        if not isinstance(field_val, list):
            field_val = [field_val] if field_val else []
        selected = set(field_val) & options
        return len(selected) <= 1  # at most one from the exclusivity group
    # W5-B (review 12): no 'unknown means satisfied' fallback. Unvalidated
    # constraint types are a caller bug — validate_constraints_for_schema
    # rejects them before any model work; reaching here with an unknown type
    # means the caller bypassed validation.
    raise ConstraintError(
        f"unknown constraint type {ctype!r}; run validate_constraints_for_schema "
        "before calling the engine"
    )


def validate_constraints(constraints: list[dict]) -> list[dict]:
    """Validate the shape of a constraint list. Returns the list if valid,
    raises ValueError with a descriptive message if any constraint is
    malformed.
    """
    if not isinstance(constraints, list):
        raise ValueError(f"constraints must be a list, got {type(constraints).__name__}")
    valid_types = {"implies", "requires_parent", "excludes", "exclusivity"}
    for i, c in enumerate(constraints):
        if not isinstance(c, dict):
            raise ValueError(f"constraint[{i}] must be a dict, got {type(c).__name__}")
        ctype = c.get("type")
        if ctype not in valid_types:
            raise ConstraintError(
                f"constraint[{i}].type must be one of {sorted(valid_types)}, got {ctype!r}"
            )
        if ctype in ("implies", "requires_parent"):
            for key in ("parent", "child", "mapping"):
                if key not in c:
                    raise ValueError(f"constraint[{i}] ({ctype}): missing key {key!r}")
            if not isinstance(c["mapping"], dict):
                raise ValueError(f"constraint[{i}] ({ctype}): mapping must be a dict")
        elif ctype == "excludes":
            for key in ("field", "value", "other"):
                if key not in c:
                    raise ValueError(f"constraint[{i}] (excludes): missing key {key!r}")
        elif ctype == "exclusivity":
            for key in ("field", "options"):
                if key not in c:
                    raise ValueError(f"constraint[{i}] (exclusivity): missing key {key!r}")
            if not isinstance(c["options"], list):
                raise ValueError(f"constraint[{i}] (exclusivity): options must be a list")
    return constraints


def validate_constraints_for_schema(constraints: list[dict], schema) -> list[dict]:
    """Compile case-level constraints against a StructuredSchema (W5-B rev 12).

    Everything that could silently no-op or blow up mid-engine fails HERE,
    before any model work:

    - shape validation (validate_constraints) — unknown types fail, no
      'unknown means satisfied';
    - every referenced field must exist in the schema;
    - implies/requires_parent: the parent must be a scalar (enum/boolean),
      every mapping key must be a value in the parent's domain, and every
      mapped child value must be in the child's domain;
    - excludes: both fields scalar, ``value`` inside ``field``'s domain;
    - exclusivity: a multi field, every option in the field's choices
      (review 13: case-level constraints cannot be reconciled on multi
      fields today — exclusivity IS the supported multi shape);
    - implies/requires_parent/excludes on a multi field: rejected (no joint
      set solver; error says so).

    Returns the constraint list unchanged when valid; raises ConstraintError
    otherwise.
    """
    validate_constraints(constraints)
    for i, c in enumerate(constraints):
        ctype = c["type"]
        if ctype in ("implies", "requires_parent"):
            parent, child, mapping = c["parent"], c["child"], c["mapping"]
            for fname in (parent, child):
                if fname not in schema.fields:
                    raise ConstraintError(f"constraint[{i}] ({ctype}): unknown field {fname!r}")
                if schema.fields[fname].field_type == "multi":
                    raise ConstraintError(
                        f"constraint[{i}] ({ctype}): field {fname!r} is a multi field; "
                        "case-level constraints on multi fields are not supported "
                        "(no joint set solver)"
                    )
            parent_domain = set(_scalar_domain(schema.fields[parent]))
            child_domain = set(_scalar_domain(schema.fields[child]))
            for parent_val in mapping:
                if parent_val not in parent_domain:
                    raise ConstraintError(
                        f"constraint[{i}] ({ctype}): mapping key {parent_val!r} is not "
                        f"a value of parent field {parent!r} "
                        f"(domain: {sorted(map(str, parent_domain))})"
                    )
                for child_val in mapping[parent_val]:
                    if child_val not in child_domain:
                        raise ConstraintError(
                            f"constraint[{i}] ({ctype}): mapped child value {child_val!r} "
                            f"is not a value of child field {child!r} "
                            f"(domain: {sorted(map(str, child_domain))})"
                        )
        elif ctype == "excludes":
            field, value, other = c["field"], c["value"], c["other"]
            for fname in (field, other):
                if fname not in schema.fields:
                    raise ConstraintError(f"constraint[{i}] (excludes): unknown field {fname!r}")
                if schema.fields[fname].field_type == "multi":
                    raise ConstraintError(
                        f"constraint[{i}] (excludes): field {fname!r} is a multi field; "
                        "case-level constraints on multi fields are not supported "
                        "(no joint set solver)"
                    )
            domain = set(_scalar_domain(schema.fields[field]))
            if value not in domain:
                raise ConstraintError(
                    f"constraint[{i}] (excludes): value {value!r} is not a value of "
                    f"field {field!r} (domain: {sorted(map(str, domain))})"
                )
        elif ctype == "exclusivity":
            field, options = c["field"], c["options"]
            if field not in schema.fields:
                raise ConstraintError(f"constraint[{i}] (exclusivity): unknown field {field!r}")
            fdef = schema.fields[field]
            if fdef.field_type != "multi":
                raise ConstraintError(
                    f"constraint[{i}] (exclusivity): field {field!r} is not a multi "
                    "field; exclusivity applies to multi (set) fields"
                )
            domain = set(fdef.choices or [])
            for option in options:
                if option not in domain:
                    raise ConstraintError(
                        f"constraint[{i}] (exclusivity): option {option!r} is not an "
                        f"option of multi field {field!r} (options: {sorted(fdef.choices or [])})"
                    )
    return constraints


def _scalar_domain(fdef) -> list[object]:
    """The typed value domain of a scalar (enum/boolean) field."""
    if fdef.field_type == "boolean":
        return [True, False]
    return list(fdef.choices or [])


# --------------------------------------------------------------------------
# W5b-11 (GPT-REVIEW-2 C6): compiled constraint objects.
#
# validate_constraints_for_schema walks the dict list on EVERY engine call
# and _constrained_map does STRING-key lookups per candidate combination.
# compile_constraints does all of that ONCE per (constraints, schema) into
# frozen typed objects carrying FIELD INDICES and value-index sets; the MAP
# evaluates by index with no dict lookups, and evalmetrics consumes the
# compiled form.


@dataclass(frozen=True)
class CompiledImplication:
    """implies/requires_parent: parent_idx -> child_idx value-index mapping.

    ``mapping`` maps PARENT VALUE-INDEX -> frozenset of allowed CHILD
    VALUE-INDEXes — indices into the compiled field domains, so evaluation
    is integer set membership (no string-key dict lookups in the MAP).
    """

    parent_idx: int
    child_idx: int
    mapping: dict[int, frozenset[int]]

    def satisfied(self, values_by_idx: dict[int, object]) -> bool:
        parent_val = values_by_idx.get(self.parent_idx)
        child_val = values_by_idx.get(self.child_idx)
        if parent_val is None or child_val is None:
            return True  # unmeasured field can't violate
        domain = self._parent_domain
        try:
            parent_index = domain.index(parent_val)
        except ValueError:
            return True  # value outside the domain cannot drive the rule
        allowed = self.mapping.get(parent_index, frozenset())
        try:
            child_index = self._child_domain.index(child_val)
        except ValueError:
            return True
        return child_index in allowed


@dataclass(frozen=True)
class CompiledExclusion:
    """excludes: when field_idx == value, other_idx must be empty/falsy."""

    field_idx: int
    value: object
    other_idx: int

    def satisfied(self, values_by_idx: dict[int, object]) -> bool:
        field_val = values_by_idx.get(self.field_idx)
        other_val = values_by_idx.get(self.other_idx)
        if field_val is None or other_val is None:
            return True
        if field_val == self.value:
            if isinstance(other_val, list):
                return len(other_val) == 0
            return other_val in (None, "", False)
        return True


@dataclass(frozen=True)
class CompiledExclusivity:
    """exclusivity: at most one of option_idxes selected in the multi field.

    Options are compiled as CHOICE VALUE-INDICES; the multi value is a list
    of the field's choice values, so intersection stays on values (a multi
    selection is values, not indices).
    """

    field_idx: int
    options: frozenset[object]

    def satisfied(self, values_by_idx: dict[int, object]) -> bool:
        field_val = values_by_idx.get(self.field_idx)
        if field_val is None:
            return True
        if not isinstance(field_val, list):
            field_val = [field_val] if field_val else []
        selected = set(field_val) & set(self.options)
        return len(selected) <= 1


@dataclass(frozen=True)
class ConstraintComponent:
    """One connected component of constrained fields (by field index)."""

    field_idxes: frozenset[int]
    constraint_indices: tuple[int, ...]


@dataclass(frozen=True)
class CompiledConstraints:
    """The compiled constraint set: frozen typed objects + component graph.

    Index space: ``field_names[idx]`` is the schema field at that index
    (schema field order); ``domain_of[idx]`` is the field's typed domain.
    ``components`` precomputes the connected components the MAP enumerates
    (adjacency from implies/excludes edges; exclusivity fields are
    singletons the MAP never reconciles).
    """

    implications: tuple[CompiledImplication, ...]
    exclusions: tuple[CompiledExclusion, ...]
    exclusivities: tuple[CompiledExclusivity, ...]
    field_names: tuple[str, ...]
    domain_of: dict[int, tuple[object, ...]]  # field_idx -> typed domain
    components: tuple[ConstraintComponent, ...]

    @property
    def constrained_field_idxes(self) -> frozenset[int]:
        out: set[int] = set()
        for c in self.implications:
            out.add(c.parent_idx)
            out.add(c.child_idx)
        for c in self.exclusions:
            out.add(c.field_idx)
            out.add(c.other_idx)
        for c in self.exclusivities:
            out.add(c.field_idx)
        return frozenset(out)

    def idx_of(self, name: str) -> int:
        return self.field_names.index(name)

    def satisfied(self, assignment: dict[str, object]) -> bool:
        """Every compiled constraint satisfied under a NAME-keyed
        assignment (evalmetrics' view) — evaluates by index internally."""
        values_by_idx = {idx: assignment.get(name) for name, idx in self.name_idx_pairs()}
        return all(
            c.satisfied(values_by_idx)
            for c in (*self.implications, *self.exclusions, *self.exclusivities)
        )

    def name_idx_pairs(self) -> tuple[tuple[str, int], ...]:
        return tuple((name, idx) for idx, name in enumerate(self.field_names))


def compile_constraints(constraints: list[dict], schema) -> CompiledConstraints:
    """Compile case-level constraints ONCE against a StructuredSchema.

    The ONLY validation entry (W5b-11): shape validation, schema-field
    existence, domain membership, multi-field support rules — everything
    validate_constraints_for_schema checked — plus index compilation. All
    failures raise ConstraintError here, before any model work.

    Returns a frozen CompiledConstraints; the engine's _constrained_map
    evaluates by index, and evalmetrics.constraint_violation_rate consumes
    the compiled form.
    """
    validate_constraints_for_schema(constraints, schema)
    field_names = tuple(schema.fields)
    idx_of = {name: i for i, name in enumerate(field_names)}
    domain_of: dict[int, tuple[object, ...]] = {
        idx_of[name]: tuple(_scalar_domain(fdef)) for name, fdef in schema.fields.items()
    }
    # Multi fields: domain = choices (for exclusivity option indices).
    for name, fdef in schema.fields.items():
        if fdef.field_type == "multi" and idx_of[name] not in domain_of:
            domain_of[idx_of[name]] = tuple(fdef.choices or ())

    implications: list[CompiledImplication] = []
    exclusions: list[CompiledExclusion] = []
    exclusivities: list[CompiledExclusivity] = []

    for c in constraints:
        ctype = c["type"]
        if ctype in ("implies", "requires_parent"):
            parent_idx = idx_of[c["parent"]]
            child_idx = idx_of[c["child"]]
            parent_domain = domain_of[parent_idx]
            child_domain = domain_of[child_idx]
            p_domain_index = {v: i for i, v in enumerate(parent_domain)}
            c_domain_index = {v: i for i, v in enumerate(child_domain)}
            mapping: dict[int, frozenset[int]] = {}
            for parent_val, child_vals in c["mapping"].items():
                p_idx = p_domain_index[parent_val]
                mapping[p_idx] = frozenset(c_domain_index[cv] for cv in child_vals)
            implications.append(CompiledImplication(parent_idx, child_idx, mapping))
        elif ctype == "excludes":
            exclusions.append(CompiledExclusion(idx_of[c["field"]], c["value"], idx_of[c["other"]]))
        elif ctype == "exclusivity":
            exclusivities.append(CompiledExclusivity(idx_of[c["field"]], frozenset(c["options"])))

    compiled = CompiledConstraints(
        implications=tuple(implications),
        exclusions=tuple(exclusions),
        exclusivities=tuple(exclusivities),
        field_names=field_names,
        domain_of=domain_of,
        components=(),  # filled below (frozen: build then replace)
    )
    # Wire the domain lookups the implication evaluator needs (frozen
    # dataclass: assign through object.__setattr__).
    for c in compiled.implications:
        object.__setattr__(c, "_parent_domain", list(domain_of[c.parent_idx]))
        object.__setattr__(c, "_child_domain", list(domain_of[c.child_idx]))

    # Connected components over implies/excludes edges (field indices).
    constrained = compiled.constrained_field_idxes
    adj: dict[int, set[int]] = {i: set() for i in constrained}
    for c in compiled.implications:
        adj[c.parent_idx].add(c.child_idx)
        adj[c.child_idx].add(c.parent_idx)
    for c in compiled.exclusions:
        adj[c.field_idx].add(c.other_idx)
        adj[c.other_idx].add(c.field_idx)
    visited: set[int] = set()
    components: list[ConstraintComponent] = []
    for start in sorted(constrained):
        if start in visited:
            continue
        queue, comp = [start], set()
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            queue.extend(adj[node] - visited)
        components.append(ConstraintComponent(frozenset(comp), ()))

    object.__setattr__(compiled, "components", tuple(components))
    return compiled
