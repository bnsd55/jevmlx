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

__all__ = ["check_constraint", "validate_constraints"]


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
    return True  # unknown constraint type: assume satisfied


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
            raise ValueError(
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
