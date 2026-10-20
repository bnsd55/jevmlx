"""Schema linting: detect enum choices the engine cannot score in one row.

The batched suffix pass gives each field one row keyed by the choice's first
token. Two choices that share a first token force the slower fallback: one
teacher-forced row per choice, scored and compared across different token
paths. ``lint_schema`` reports that (and related problems) before a schema
ships.
"""

from __future__ import annotations

from dataclasses import dataclass

from jevmlx.schema import SchemaCompileError, StructuredSchema


@dataclass(frozen=True)
class Finding:
    """One lint result for a schema field.

    Attributes:
        field: Name of the schema field the finding applies to.
        kind: One of ``"collision"``, ``"compile_error"``, ``"single_choice"``.
        message: Human-readable description of the problem.
        suggestion: A concrete rename, when one can be computed mechanically.
    """

    field: str
    kind: str
    message: str
    suggestion: str | None = None


def _rotation_suggestion(choices: list[str], all_choices: list[str], tokenizer) -> str | None:
    """Suggest renames that break a first-token tie, or None if the rule does not apply.

    Rule: when every colliding choice shares the same first underscore-separated
    word, move that word to the end so the distinguishing word leads
    (``BLOCK_TRANSACTION`` / ``BLOCK_USER`` -> ``TRANSACTION_BLOCK`` / ``USER_BLOCK``).
    This is best-effort; if any choice is a single word, or the rotated forms
    are not unique within the colliding subset, or any rotated name already
    exists in the field's COMPLETE choice set (N2: suggesting a name the
    schema already uses would trade a collision for a duplicate), no
    suggestion is made.
    """
    parts_list = [choice.split("_") for choice in choices]
    if any(len(parts) < 2 for parts in parts_list):
        return None
    first_words = {parts[0] for parts in parts_list}
    if len(first_words) != 1:
        return None
    rotated = ["_".join(parts[1:] + parts[:1]) for parts in parts_list]
    if len(set(rotated)) != len(rotated):
        return None
    # Validate against the complete choice set after renaming: the colliding
    # choices are replaced by their rotations, everything else keeps its name.
    renamed_list = [rotated[choices.index(c)] if c in choices else c for c in all_choices]
    if len(set(renamed_list)) != len(all_choices):
        return None

    # Tokenizer-verified: the renamed set must compile AND score in one row
    # per field (no first-token collision) — rotation can merely move the
    # collision to the next word (P1), e.g. BLOCK_* -> *_BLOCK that still
    # shares its first token with another choice.
    try:
        from jevmlx.schema import StructuredSchema

        probe = StructuredSchema(
            {"_probe": {"type": "enum", "description": "", "choices": renamed_list}}
        )
        from jevmlx.engine import make_field_prompt_renderer

        _rfp = make_field_prompt_renderer(tokenizer, "", probe, "labels")
        probe_entry = probe.compile_labels_plan(tokenizer, _rfp)["fields"]["_probe"]
    except ValueError:
        return None
    seen_first: set[int] = set()
    for remainder in probe_entry["remainders"]:
        if remainder[0] in seen_first:
            return None
        seen_first.add(remainder[0])
    return ", ".join(rotated)


def lint_schema(schema: StructuredSchema, tokenizer) -> list[Finding]:
    """Lint a schema's enum choices for engine-visible problems.

    Token lists come from ``StructuredSchema.compile_labels_plan`` — the exact
    lists the engine scores — so the lint never re-tokenizes by hand and can
    never disagree with the engine about token boundaries.

    Only categorical enum fields are linted. Boolean and multi fields decide
    true/false per option, so their choice token lists are fixed and cannot
    collide.

    The plan is compiled ONCE for the whole schema before any per-field loop
    (M1): a compile failure caused by a boolean or multi field is attributed
    to that field (SchemaCompileError.field), not reported per enum with the
    wrong name — and boolean/multi-only schemas get real coverage.

    Checks:
    - compile_error: the schema cannot be compiled for this tokenizer at all
      (token-identical choices, strict token-prefix continuations, choices
      with no shared token prefix, in ANY field type). Exactly one finding
      naming SchemaCompileError.field; linting stops there because nothing
      else can be linted without a plan.
    - collision (enum fields): two or more choices share their first
      diverging token. The engine still scores them correctly, but the field
      needs one row per trie branch point instead of one row total (slower).
    - single_choice (informational, enum fields): a cardinality-1 enum has
      no branch points; the engine scores it deterministically (P = 1.0).
      Reported so the degenerate schema is visible, not an error.
    """
    findings: list[Finding] = []

    try:
        from jevmlx.engine import make_field_prompt_renderer

        _rfp = make_field_prompt_renderer(tokenizer, "", schema, "labels")
        compiled = schema.compile_labels_plan(tokenizer, _rfp)
    except SchemaCompileError as exc:
        return [
            Finding(
                field=exc.field,
                kind="compile_error",
                message=f"schema cannot be compiled: {exc}",
            )
        ]

    for fname, fdef in schema.fields.items():
        if fdef.field_type not in ("enum", "choice", "selection"):
            continue

        if len(fdef.choices) == 1:
            # Cardinality-1 enum: empty remainder, no branch points, nothing
            # to collide — informational only (L1a).
            findings.append(
                Finding(
                    field=fname,
                    kind="single_choice",
                    message=(
                        "single-choice enum: the value is fully determined by "
                        "the schema; the engine scores it deterministically (P=1.0)"
                    ),
                )
            )
            continue

        entry = compiled["fields"][fname]
        groups: dict[int, list[int]] = {}
        for idx, tokens in enumerate(entry["remainders"]):
            groups.setdefault(tokens[0], []).append(idx)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            colliding = [fdef.choices[i] for i in indices]
            findings.append(
                Finding(
                    field=fname,
                    kind="collision",
                    message=(
                        "choices share their first token; the field needs one row per "
                        "trie branch point instead of one row per field (slower). "
                        f"Rename to keep one row per field: {', '.join(colliding)}"
                    ),
                    suggestion=_rotation_suggestion(colliding, fdef.choices, tokenizer),
                )
            )

    return findings
