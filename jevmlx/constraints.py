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

__all__ = [
    "check_constraint",
    "validate_constraints",
    "validate_constraints_for_schema",
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
