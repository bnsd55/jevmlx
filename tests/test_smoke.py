import json

import pytest
from conftest import MODEL_ID

from jevmlx.cli import load_preset
from jevmlx.engine import load_engine, run_parallel_generation
from jevmlx.schema import StructuredSchema


@pytest.mark.slow
def test_fintech_fraud_decisions():
    engine = load_engine(MODEL_ID)
    preset = load_preset("fintech_fraud")
    schema = StructuredSchema(preset["schema"])
    result = run_parallel_generation(engine, preset["context"], schema)

    # Every schema key present, every value in its choices.
    for fname, fdef in schema.fields.items():
        assert fname in result["parsed_json"]
        val = result["parsed_json"][fname]["value"]
        expected = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
        assert str(val).lower() in [c.lower() for c in expected], f"{fname}={val!r} not in choices"

    # Real inference ran: telemetry for every field with valid confidences.
    assert set(result["field_telemetry"]) == set(schema.fields)
    for entry in result["field_telemetry"].values():
        assert 0.0 <= entry["probability"] <= 1.0

    # The assembled JSON serializes.
    assert isinstance(json.dumps(result["parsed_json"]), str)


def test_validate_json_multi_list_semantics():
    """C4a: multi values compared as lists (set equality, no duplicates)."""
    from jevmlx.engine import _validate_json

    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["a", "b", "c"]}}
    )

    # Valid subset -> schema_match True.
    parsed, valid, _, missing, invalid, match = _validate_json('{"flags": ["a", "c"]}', schema)
    assert valid and match and not missing and not invalid

    # Same items different order -> still valid.
    _, _, _, _, _, match = _validate_json('{"flags": ["c", "a"]}', schema)
    assert match

    # Duplicate items -> rejected.
    _, _, _, _, invalid, match = _validate_json('{"flags": ["a", "a"]}', schema)
    assert not match and invalid

    # Unknown item -> rejected.
    _, _, _, _, invalid, match = _validate_json('{"flags": ["a", "zzz"]}', schema)
    assert not match and invalid

    # Non-list -> rejected.
    _, _, _, _, invalid, match = _validate_json('{"flags": "a"}', schema)
    assert not match and invalid


def test_validate_json_non_object_is_valid_but_no_match():
    """C4b: a parsed JSON that is not a dict -> is_valid_json=True, match=False."""
    from jevmlx.engine import _validate_json

    schema = StructuredSchema({"flag": {"type": "boolean", "description": "d"}})
    for text in ("[1, 2, 3]", '"hello"', "42", "null"):
        parsed, valid, _, _, _, match = _validate_json(text, schema)
        assert valid is True, text
        assert match is False, text
    parsed, valid, _, _, _, match = _validate_json("[1, 2, 3]", schema)
    assert parsed == [1, 2, 3]


def test_validate_json_boolean_and_enum_type_checks():
    """D2: booleans must be real JSON bools; enums compare as str only."""
    from jevmlx.engine import _validate_json

    schema = StructuredSchema(
        {
            "flag": {"type": "boolean", "description": "d"},
            "action": {"type": "enum", "description": "d", "choices": ["A", "B"]},
        }
    )

    # Boolean given as the string "true" -> invalid.
    _, _, _, _, invalid, match = _validate_json('{"flag": "true", "action": "A"}', schema)
    assert not match and any("flag" in i for i in invalid)

    # Boolean given as 1 -> invalid.
    _, _, _, _, invalid, match = _validate_json('{"flag": 1, "action": "A"}', schema)
    assert not match and any("flag" in i for i in invalid)

    # Real boolean -> valid.
    _, _, _, _, invalid, match = _validate_json('{"flag": true, "action": "A"}', schema)
    assert match and not invalid

    # Enum given as int -> invalid (no str coercion).
    _, _, _, _, invalid, match = _validate_json('{"flag": true, "action": 1}', schema)
    assert not match and any("action" in i for i in invalid)

    # Enum given as a str that happens to stringify to a choice -> still invalid.
    _, _, _, _, invalid, match = _validate_json('{"flag": true, "action": ["A"]}', schema)
    assert not match and any("action" in i for i in invalid)
