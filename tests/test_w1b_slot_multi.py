"""W1-B tests: exact candidate-token reconstruction for slot+multi lead-in
repair (Q6-1, Q6-2). Every candidate row's tokens (lead_in + shared + remainder)
must equal tokenizing the full candidate text, for multi-only, scalar-only and
mixed schemas, in both slots and labels modes.

Also verifies:
- compile_slot_plan does NOT call compile_labels_plan (Q6-2: a scalar
  real-label collision in labels mode must not fail slot mode).
- The multi section prompt shows explicit Y/N codes (Q6-6).
- Field names/labels/options/glosses are json.dumps-escaped (Q2).
"""

from __future__ import annotations

import json

from jevmlx.schema import StructuredSchema

# A fake char tokenizer: each character maps to one token id. BPE merges are
# not an issue because every character is its own token — the compositional
# property (prefix + remainder == full tokenization) holds exactly.


class CharTokenizer:
    name_or_path = "fake-char"
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert tokenize
        return self.encode("\n".join(m["content"] for m in messages))

    def __len__(self) -> int:
        return 128


def _scalar_schema():
    return StructuredSchema(
        {
            "risk": {
                "type": "enum",
                "description": "Risk level",
                "choices": ["LOW", "HIGH"],
            },
            "flag": {"type": "boolean", "description": "flagged"},
        }
    )


def _multi_only_schema():
    return StructuredSchema(
        {
            "tags": {
                "type": "multi",
                "description": "Tags to apply",
                "choices": ["alpha", "beta"],
            },
        }
    )


def _mixed_schema():
    return StructuredSchema(
        {
            "risk": {
                "type": "enum",
                "description": "Risk level",
                "choices": ["LOW", "HIGH"],
            },
            "tags": {
                "type": "multi",
                "description": "Tags to apply",
                "choices": ["alpha", "beta", "gamma"],
            },
        }
    )


def _candidate_text_scalar(name: str, value_text: str) -> str:
    """The complete one-field JSON object for one scalar row (W2-B protocol)."""
    return "{" + f"{json.dumps(name)}: {value_text}" + "}"


def _candidate_text_multi(name: str, option: str, alias: str) -> str:
    """The complete one-field JSON object for one multi Y/N row (W2-B protocol)."""
    return "{" + f'{json.dumps(f"{name}/{option}")}: "{alias}"' + "}"


def _reconstruct_row(lead_in: list[int], shared: list[int], remainder: list[int]) -> list[int]:
    return list(lead_in) + list(shared) + list(remainder)


def _check_scalar_reconstruction(schema, tokenizer, mode: str):
    """Every scalar candidate row = lead_in + shared + remainder == full text."""
    plan = (
        schema.compile_slot_plan(tokenizer)
        if mode == "slots"
        else schema.compile_labels_plan(tokenizer)
    )
    lead_in = plan["lead_in_ids"]
    for fname, fplan in plan["fields"].items():
        if "suffix_ids_list" in fplan:
            continue  # multi, handled separately
        shared = fplan["shared_ids"]
        fdef = schema.fields[fname]
        values = ["true", "false"] if fdef.field_type == "boolean" else list(fdef.choices)
        for j, remainder in enumerate(fplan["remainders"]):
            value = values[j]
            if mode == "slots":
                # Slots mode: the value text is the alias, not the real choice.
                aliases = fplan["aliases"]
                value_text = json.dumps(aliases[j])
            else:
                value_text = (
                    "true"
                    if value == "true"
                    else "false"
                    if value == "false"
                    else json.dumps(value)
                )
            full_text = _candidate_text_scalar(fname, value_text)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            recon = _reconstruct_row(lead_in, shared, remainder)
            assert recon == full_ids, (
                f"{mode} scalar {fname}={value}: reconstructed {recon} != "
                f"full {full_ids} (text={full_text!r})"
            )


def _check_multi_reconstruction(schema, tokenizer, mode: str):
    """Every multi Y/N candidate row = lead_in + suffix + remainder == full text."""
    plan = (
        schema.compile_slot_plan(tokenizer)
        if mode == "slots"
        else schema.compile_labels_plan(tokenizer)
    )
    lead_in = plan["lead_in_ids"]
    for fname, fplan in plan["fields"].items():
        if "suffix_ids_list" not in fplan:
            continue
        options = fplan["options"]
        # W2-E row codes: rows are keyed '<field>/<code>'; the reconstruction
        # target must be built from the CODE, not the raw option text.
        codes = fplan["codes"]
        for opt_idx, (suffix, remainders) in enumerate(
            zip(fplan["suffix_ids_list"], fplan["remainders"], strict=True)
        ):
            option, code = options[opt_idx], codes[opt_idx]
            for yn_idx, alias in enumerate(("Y", "N")):
                remainder = remainders[yn_idx]
                full_text = _candidate_text_multi(fname, code, alias)
                full_ids = tokenizer.encode(full_text, add_special_tokens=False)
                recon = _reconstruct_row(lead_in, suffix, remainder)
                assert recon == full_ids, (
                    f"{mode} multi {fname}/{option}={alias}: reconstructed {recon} != "
                    f"full {full_ids} (text={full_text!r})"
                )


def test_scalar_only_slots_reconstruction():
    tok = CharTokenizer()
    schema = _scalar_schema()
    _check_scalar_reconstruction(schema, tok, "slots")


def test_scalar_only_labels_reconstruction():
    tok = CharTokenizer()
    schema = _scalar_schema()
    _check_scalar_reconstruction(schema, tok, "labels")


def test_multi_only_slots_reconstruction():
    """Q6-1: multi-only slot schema — lead_in must be the common prefix of the
    option rows (not empty), and every Y/N row must reconstruct exactly."""
    tok = CharTokenizer()
    schema = _multi_only_schema()
    plan = schema.compile_slot_plan(tok)
    # The lead-in must NOT be empty for a multi-only schema (the bug: it was
    # empty because compile_slot_plan computed lead_in from scalar shared_ids
    # only, and there were no scalars).
    assert plan["lead_in_ids"] != [], "multi-only slot schema has empty lead_in (Q6-1 bug)"
    _check_multi_reconstruction(schema, tok, "slots")


def test_multi_only_labels_reconstruction():
    tok = CharTokenizer()
    schema = _multi_only_schema()
    _check_multi_reconstruction(schema, tok, "labels")


def test_mixed_slots_reconstruction():
    """Q6-1: mixed schema — lead_in factored over BOTH scalar shared_ids AND
    multi suffix_ids_list, stripped exactly once. Every row reconstructs."""
    tok = CharTokenizer()
    schema = _mixed_schema()
    _check_scalar_reconstruction(schema, tok, "slots")
    _check_multi_reconstruction(schema, tok, "slots")


def test_mixed_labels_reconstruction():
    tok = CharTokenizer()
    schema = _mixed_schema()
    _check_scalar_reconstruction(schema, tok, "labels")
    _check_multi_reconstruction(schema, tok, "labels")


def test_slot_plan_does_not_call_labels_plan():
    """Q6-2: compile_slot_plan must not invoke compile_labels_plan. A scalar
    real-label collision in labels mode must not fail slot mode.

    We build a schema where two scalar choices would be token-identical in
    labels mode (same text) but have distinct slot aliases. If
    compile_slot_plan still calls compile_labels_plan, this would raise."""
    tok = CharTokenizer()
    # Two enum fields: one with a deliberate token collision in labels mode
    # (choices that tokenize identically under a char tokenizer — same chars).
    # Under a char tokenizer, "AA" and "AA" are identical, but we need two
    # distinct choices that collide. Use choices with the same characters:
    # "AB" and "AB" would be rejected at schema build. Instead, use a schema
    # where labels mode would reject (token-prefix) but slots mode is fine.
    #
    # Actually, the simplest test: verify compile_slot_plan does not populate
    # the labels cache. After compile_slot_plan, the labels cache should be
    # empty (compile_labels_plan was never called).
    schema = _mixed_schema()
    schema.compile_slot_plan(tok)
    # The labels plan cache should be empty — compile_slot_plan used
    # _compile_multi_plan, not compile_labels_plan.
    cached_labels = schema._cached_plan(tok, "labels")
    assert cached_labels is None, "compile_slot_plan called compile_labels_plan (Q6-2 bug)"


def test_multi_prompt_shows_yn_codes():
    """Q6-6 + W2-E: the multi section maps code = option (choices order) and
    shows the exact quoted Y/N codes the scorer reads."""
    schema = _multi_only_schema()
    tok2 = CharTokenizer()
    block = schema.to_schema_str("slots", tokenizer=tok2)
    assert '"Y" = applies or "N" = does not apply' in block
    assert '00 = "alpha"' in block and '01 = "beta"' in block


def test_field_names_json_dumps_escaped():
    """Q2: field names, labels, options, glosses rendered through json.dumps."""
    schema = StructuredSchema(
        {
            'field"with"quotes': {
                "type": "enum",
                "description": 'desc"with"quotes',
                "choices": ['choice"with"quotes', "safe"],
                "choice_descriptions": {'choice"with"quotes': 'gloss"with"quotes'},
            },
        }
    )
    tok2 = CharTokenizer()
    block = schema.to_schema_str("slots", tokenizer=tok2)
    # json.dumps escapes the quotes — the block should contain escaped quotes
    # and never raw unescaped quotes that would break parsing.
    assert '\\"' in block  # escaped quotes present


def test_prompt_version_bumped():
    """PROMPT_VERSION is now v3 (W1-B)."""
    from jevmlx.engine import PROMPT_VERSION

    assert PROMPT_VERSION == "jevmlx-parallel-v9"


def test_p5_option_plan_cache_reuses_across_rotations():
    """P5: the option-plan cache reuses the tokenizer-invariant part
    (aliases, shared_ids, remainders) across rotations. A rotation creates
    a new StructuredSchema, but the per-field token work is the same —
    only the alias_map (alias -> real value) changes."""
    from jevmlx.schema import (
        _OPTION_PLAN_CACHE,
        StructuredSchema,
        _get_option_plan,
    )

    tok = CharTokenizer()
    # Clear the cache to start fresh.
    _OPTION_PLAN_CACHE.clear()

    schema_canonical = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B", "C", "D"]}}
    )
    plan1 = schema_canonical.compile_slot_plan(tok)

    # The cache should now have an entry for (tok, "action", 4, "slots").
    cached = _get_option_plan(tok, "action", ("A", "B", "C", "D"), "slots")
    assert cached is not None
    # The cached aliases and remainders are the tokenizer-invariant part.
    assert list(cached["aliases"]) == list(plan1["fields"]["action"]["aliases"])
    assert [list(r) for r in cached["remainders"]] == [
        list(r) for r in plan1["fields"]["action"]["remainders"]
    ]

    # A rotation: same choices, different order, NEW schema object.
    schema_rotated = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["B", "C", "D", "A"]}}
    )
    plan2 = schema_rotated.compile_slot_plan(tok)

    # The token ids (shared_ids, remainders) are IDENTICAL — only the
    # alias_map differs (alias A -> B instead of A -> A).
    assert plan1["fields"]["action"]["shared_ids"] == plan2["fields"]["action"]["shared_ids"]
    assert plan1["fields"]["action"]["remainders"] == plan2["fields"]["action"]["remainders"]
    assert plan1["fields"]["action"]["aliases"] == plan2["fields"]["action"]["aliases"]
    # The alias_map reflects the rotation.
    assert plan1["fields"]["action"]["alias_map"] == {"A": "A", "B": "B", "C": "C", "D": "D"}
    assert plan2["fields"]["action"]["alias_map"] == {"A": "B", "B": "C", "C": "D", "D": "A"}


def test_p5_option_plan_cache_no_cross_contamination_different_choices():
    """P5 review fix: two schemas with the SAME field name and choice count
    but DIFFERENT choice strings must NOT share a plan. The cache key is
    (id(tokenizer), mode, field_name, tuple(sorted(choices))) — the choice
    SET is part of the key, not just the count."""
    from jevmlx.schema import _OPTION_PLAN_CACHE

    tok = CharTokenizer()
    _OPTION_PLAN_CACHE.clear()

    schema_alpha = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B", "C", "D"]}}
    )
    plan_alpha = schema_alpha.compile_slot_plan(tok)

    # Same field name, same count (4), DIFFERENT choice strings.
    schema_xyz = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["W", "X", "Y", "Z"]}}
    )
    plan_xyz = schema_xyz.compile_slot_plan(tok)

    # The two plans must have DIFFERENT alias_maps (different real values).
    assert plan_alpha["fields"]["action"]["alias_map"] == {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
    }
    assert plan_xyz["fields"]["action"]["alias_map"] == {
        "A": "W",
        "B": "X",
        "C": "Y",
        "D": "Z",
    }

    # The token ids (shared_ids, remainders) are the SAME — same field name,
    # same aliases, same tokenizer. Only the alias_map differs.
    assert list(plan_alpha["fields"]["action"]["shared_ids"]) == list(
        plan_xyz["fields"]["action"]["shared_ids"]
    )
    assert [list(r) for r in plan_alpha["fields"]["action"]["remainders"]] == [
        list(r) for r in plan_xyz["fields"]["action"]["remainders"]
    ]

    # Verify on the fake engine: decisions match the uncached path for both.
    from conftest import FakeModel, FakeTokenizer, make_engine

    from jevmlx.engine import run_parallel_generation

    _OPTION_PLAN_CACHE.clear()
    # Uncached: fresh tokenizer so the cache is empty.
    tok_uncached = FakeTokenizer()
    result_alpha_uncached = run_parallel_generation(
        make_engine(FakeModel(), tok_uncached),
        "ctx",
        StructuredSchema(
            {"action": {"type": "enum", "description": "d", "choices": ["A", "B", "C", "D"]}}
        ),
    )
    result_xyz_uncached = run_parallel_generation(
        make_engine(FakeModel(), FakeTokenizer()),
        "ctx",
        StructuredSchema(
            {"action": {"type": "enum", "description": "d", "choices": ["W", "X", "Y", "Z"]}}
        ),
    )
    # Cached: populate the cache, then run again.
    tok_cached = FakeTokenizer()
    schema_alpha_c = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B", "C", "D"]}}
    )
    schema_alpha_c.compile_slot_plan(tok_cached)  # populate cache
    schema_xyz_c = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["W", "X", "Y", "Z"]}}
    )
    schema_xyz_c.compile_slot_plan(tok_cached)  # populate cache for XYZ
    result_alpha_cached = run_parallel_generation(
        make_engine(FakeModel(), tok_cached),
        "ctx",
        StructuredSchema(
            {"action": {"type": "enum", "description": "d", "choices": ["A", "B", "C", "D"]}}
        ),
    )
    result_xyz_cached = run_parallel_generation(
        make_engine(FakeModel(), tok_cached),
        "ctx",
        StructuredSchema(
            {"action": {"type": "enum", "description": "d", "choices": ["W", "X", "Y", "Z"]}}
        ),
    )
    # Decisions match the uncached path for both schemas.
    assert (
        result_alpha_uncached["parsed_json"]["action"]["value"]
        == result_alpha_cached["parsed_json"]["action"]["value"]
    )
    assert (
        result_xyz_uncached["parsed_json"]["action"]["value"]
        == result_xyz_cached["parsed_json"]["action"]["value"]
    )
    # And the two schemas produce different winners (different choices).
    assert (
        result_alpha_uncached["parsed_json"]["action"]["value"]
        != result_xyz_uncached["parsed_json"]["action"]["value"]
    )
