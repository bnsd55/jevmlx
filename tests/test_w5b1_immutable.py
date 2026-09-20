"""W5b-1 (GPT-REVIEW-2 C7 + C13): schema immutability + no compat re-exports.

C7: StructuredSchema and FieldDefinition are immutable after construction;
set_constraints is deep-frozen (read-only constraint views in a tuple);
compiled plans are returned read-only. Mutation attempts raise.
C13: jevmlx.api carries no backward-compatibility re-exports (ARCHITECTURE
rule: "No backward compatibility" — old names are deleted, not aliased).
"""

from __future__ import annotations

import pytest

from jevmlx.schema import FieldDefinition, StructuredSchema


@pytest.fixture()
def schema() -> StructuredSchema:
    return StructuredSchema(
        {
            "risk_tier": {
                "type": "enum",
                "description": "d",
                "choices": ["LOW", "HIGH"],
                "choice_descriptions": {"LOW": "calm"},
            },
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["x", "y", "z"],
            },
        }
    )


# --------------------------------------------------------- C7: fields -----


def test_field_definition_attribute_mutation_raises():
    f = FieldDefinition("f", "enum", "d", choices=["a", "b"])
    with pytest.raises(Exception, match="cannot assign"):
        f.name = "g"
    with pytest.raises(Exception, match="cannot assign"):
        f.field_type = "boolean"
    with pytest.raises(Exception, match="cannot assign"):
        f.description = "new"
    with pytest.raises(Exception, match="cannot assign"):
        f.depends_on = "f2"


def test_field_choices_are_frozen_tuple():
    f = FieldDefinition("f", "enum", "d", choices=["a", "b"])
    assert isinstance(f.choices, tuple)
    with pytest.raises(AttributeError):
        f.choices.append("c")  # type: ignore[attr-defined]


def test_field_choice_descriptions_readonly(schema):
    d = schema["risk_tier"].choice_descriptions
    with pytest.raises(TypeError):
        d["LOW"] = "changed"
    with pytest.raises(TypeError):
        del d["LOW"]


def test_schema_fields_mapping_readonly(schema):
    with pytest.raises(TypeError):
        schema.fields["new"] = schema["risk_tier"]
    with pytest.raises(TypeError):
        del schema.fields["risk_tier"]
    # Reads still work.
    assert schema["risk_tier"].name == "risk_tier"
    assert list(schema.fields) == ["risk_tier", "flags"]


def test_field_dict_input_not_shared_with_schema():
    """The caller's dict/list inputs are copied at the boundary: mutating
    them after construction cannot change the schema."""
    choices = ["a", "b"]
    descs = {"a": "g"}
    f = FieldDefinition("f", "enum", "d", choices=choices, choice_descriptions=descs)
    choices.append("c")
    descs["a"] = "hacked"
    assert f.choices == ("a", "b")
    assert f.choice_descriptions["a"] == "g"


# ------------------------------------------- C7: set constraints ----------


def test_set_constraints_frozen_tuple_of_readonly_views():
    constraints = [
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
    sc = schema["flags"].set_constraints
    assert isinstance(sc, tuple)
    # Constraint contents are read-only:
    with pytest.raises(TypeError):
        sc[0]["k"] = 1
    # And the caller's original dicts are copies, not views:
    constraints[0]["k"] = 99
    assert dict(sc[0])["k"] == 2


def test_compile_set_constraints_returns_new_frozen_field():
    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    before = schema["flags"]
    after = before.compile_set_constraints([{"type": "at_most_one", "options": ["x", "y"]}])
    assert after is not before
    assert len(after.set_constraints) == 1
    assert before.set_constraints == ()  # the old reference is untouched


def test_compile_set_constraints_none_returns_same_field():
    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    assert schema["flags"].compile_set_constraints(None) is schema["flags"]


# --------------------------------------------- C7: compiled plans --------


class _Tok:
    name_or_path = "fake-w5b1"
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 97 + 1 for c in text]

    def __len__(self) -> int:
        return 128


def test_compiled_plan_is_readonly(schema):
    plan = schema.compile_slot_plan(_Tok())
    with pytest.raises(TypeError):
        plan["fields"]["flags"]["shared_ids"] = [1]  # type: ignore[index]
    with pytest.raises(TypeError):
        plan["fields"]["flags"]["count"]["shared_ids"] = [1]  # type: ignore[index]
    # Frozen sequences: no append/assignment (tuple: AttributeError is fine
    # — the surface is read-only either way).
    with pytest.raises((TypeError, AttributeError)):
        plan["lead_in_ids"].append(1)  # type: ignore[attr-defined]


def test_plan_cache_returns_same_frozen_object(schema):
    """Cache identity holds AND the returned object is the frozen one —
    no per-read re-freezing that would hand out fresh mutable copies."""
    tok = _Tok()
    p1 = schema.compile_slot_plan(tok)
    p2 = schema.compile_slot_plan(tok)
    assert p1 is p2


def test_labels_plan_is_readonly(schema):
    plan = schema.compile_labels_plan(_Tok())
    with pytest.raises(TypeError):
        plan["fields"]["risk_tier"]["shared_ids"] = []  # type: ignore[index]


# --------------------------------------------------- C13: api re-exports --


def test_api_has_no_compat_reexports():
    """jevmlx.api exposes its own surface only — no models-module re-exports
    (ARCHITECTURE: no backward compatibility, no aliases)."""
    import jevmlx.api as api

    exported = set(api.__all__)
    assert "resolve_model" not in exported
    assert "MODEL_ALIASES" not in exported
    # The API's own surface is intact:
    assert {
        "decide",
        "decide_many",
        "schema_from_model",
        "DEFAULT_MODEL",
        "DEFAULT_SCORING",
    } <= exported
