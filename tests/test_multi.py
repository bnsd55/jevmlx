"""Fast tests for the multi field type: batch-plan expansion, API mapping,
and the pure fold function. No model loading."""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from jevmlx.api import schema_from_model
from jevmlx.engine import _fold_multi
from jevmlx.schema import FieldDefinition, StructuredSchema


class FakeTokenizer:
    """Deterministic tokenizer: one id per character (offset so ids start at 1)."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % 97 + 1 for c in text] or [1]


def test_compile_labels_plan_expands_multi_field():
    schema = StructuredSchema(
        {
            "categories": {
                "type": "multi",
                "description": "categories that apply",
                "choices": ["billing", "technical"],
            }
        }
    )
    plan = schema.compile_labels_plan(FakeTokenizer())
    p = plan["fields"]["categories"]

    # One yes/no row per option, row key '<field>/<option>' (V4: the natural
    # question, not a synthetic 'field.option' JSON key), scored candidates
    # are the quoted aliases "Y"/"N".
    assert p["options"] == ["billing", "technical"]
    assert len(p["suffix_ids_list"]) == 2
    assert p["suffix_ids_list"][0] != p["suffix_ids_list"][1]
    # The common row lead-in ('{\n  "categories/' with the slash) is lifted
    # into plan["lead_in_ids"]; each suffix continues with '<option>": "'.
    full_row = plan["lead_in_ids"] + p["suffix_ids_list"][0]
    slash_id = ord("/") % 97 + 1
    assert slash_id in full_row  # the row key really is '<field>/<option>'
    # Per-option Y/N remainder pairs (2 options x 2), each starting where
    # that option's row ends — the branch compares "Y" vs "N" there.
    assert len(p["remainders"]) == 2
    assert all(len(pair) == 2 for pair in p["remainders"])
    assert all(len(toks) >= 1 for pair in p["remainders"] for toks in pair)
    # Pair order: ["Y"] first, ["N"] second (engine folds index 0 = yes).
    assert all(pair[0][0] != pair[1][0] for pair in p["remainders"])
    for pair in p["remainders"]:
        yes_first = ord("Y") % 97 + 1
        no_first = ord("N") % 97 + 1
        assert {pair[0][0], pair[1][0]} == {yes_first, no_first}
    # Fold mapping: the engine folds rows back by option index.
    assert p["options"].index("billing") == 0
    assert p["options"].index("technical") == 1


def test_multi_field_validation():
    with pytest.raises(ValueError, match="at least 2"):
        FieldDefinition("x", "multi", "d", choices=["only"])
    with pytest.raises(ValueError, match="64"):
        FieldDefinition("x", "multi", "d", choices=[f"c{i}" for i in range(65)])
    with pytest.raises(ValueError, match="multi"):
        FieldDefinition("x", "multi", "d")


def test_schema_from_model_list_and_set_literal():
    class WithTags(BaseModel):
        tags: list[Literal["a", "b"]] = Field(description="tags that apply")
        badges: set[Literal["x", "y"]] = Field(description="badges")

    assert schema_from_model(WithTags) == {
        "tags": {
            "type": "multi",
            "choices": ["a", "b"],
            "description": "tags that apply",
            "choice_descriptions": {},
        },
        "badges": {
            "type": "multi",
            "choices": ["x", "y"],
            "description": "badges",
            "choice_descriptions": {},
        },
    }


def test_schema_from_model_non_literal_list_raises():
    class Bad(BaseModel):
        nums: list[int] = Field(description="numbers")

    with pytest.raises(TypeError, match="'nums'"):
        schema_from_model(Bad)


def test_fold_multi():
    # Threshold semantics: p_yes >= threshold selects; margin is the closest
    # option's distance to the threshold, over ALL options (accepted or not).
    selected, prob, margin = _fold_multi({"a": 0.9, "b": 0.6, "c": 0.2}, 0.5)
    assert selected == ["a", "b"]
    assert prob is None  # no field-level probability is claimed
    assert margin == pytest.approx(0.1)  # b sits 0.1 above the threshold

    # A rejected option can be the closest: c sits just under threshold.
    selected, prob, margin = _fold_multi({"a": 0.99, "b": 0.55, "c": 0.48}, 0.5)
    assert selected == ["a", "b"]
    assert margin == pytest.approx(0.02)  # 0.5 - 0.48

    # Empty selection: margin from the strongest rejected option.
    selected, prob, margin = _fold_multi({"a": 0.2, "b": 0.4, "c": 0.49}, 0.5)
    assert selected == []
    assert margin == pytest.approx(0.01)

    # A higher threshold moves both the selection and the margin.
    selected, prob, margin = _fold_multi({"a": 0.9, "b": 0.6}, 0.7)
    assert selected == ["a"]
    assert margin == pytest.approx(0.1)  # |0.6 - 0.7|

    # Boundary p_yes == threshold selects.
    selected, _prob, margin = _fold_multi({"a": 0.5, "b": 0.49}, 0.5)
    assert selected == ["a"]
    assert margin == pytest.approx(0.0)  # a sits exactly on the threshold

    # No options at all: nothing selected, zero margin (vacuous).
    selected, prob, margin = _fold_multi({}, 0.5)
    assert selected == []
    assert prob is None
    assert margin == 0.0


def test_multi_prompt_describes_options_and_yes_no_contract():
    """W2-E (v4): the schema block describes the multi field once, mapping
    code = option in choices order (the engine's decision rows are keyed
    '<field>/<code>'), with glosses and the Y/N contract; the block must
    never show a synthetic 'field.option' key."""
    schema = StructuredSchema(
        {
            "topics": {
                "type": "multi",
                "description": "topics that apply",
                "choices": ["billing", "tech"],
                "choice_descriptions": {"billing": "money issues", "tech": "software faults"},
            }
        }
    )
    for mode in ("slots", "labels"):
        block = schema.to_schema_str(mode)
        # W2-E row codes: the block maps code = option in choices order.
        assert '00 = "billing" — "money issues"' in block
        assert '01 = "tech" — "software faults"' in block
        assert '"Y" = applies or "N" = does not apply' in block
        assert "select all that apply" in block
        assert '"topics.billing"' not in block and '"topics.billing"' not in block
        assert "topics/billing" not in block  # raw option never a row key in prompt
        assert "topics/00" not in block  # row keys stay engine-side


def test_multi_engine_result_semantics():
    """V4 result semantics: value = selected list, prob = None (no field
    probability), margin = min |p_yes - threshold|, alternatives = per-option
    list sorted by P(yes)."""

    from tests.test_engine_fake import FakeModel, FakeTokenizer

    # Bias logits so option 0's "Y" token wins strongly and option 1's "N"
    # wins: model returns zeros, so probabilities are uniform at 0.5 — the
    # boundary case selects everything at threshold 0.5 and nothing at 0.51.
    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    model, tokenizer = FakeModel(), FakeTokenizer()
    at_half = run_generation(schema, model, tokenizer, 0.5)
    telemetry = at_half["field_telemetry"]["flags"]
    assert at_half["parsed_json"]["flags"]["value"] == ["x", "y"]
    assert at_half["parsed_json"]["flags"]["prob"] is None
    assert telemetry["probability"] is None
    assert telemetry["margin"] == pytest.approx(0.0)  # |0.5 - 0.5|
    assert telemetry["threshold"] == 0.5
    assert telemetry["alternatives"] == (("x", 0.5), ("y", 0.5))
    # Raising the threshold deselects everything; margin grows.
    above = run_generation(schema, model, tokenizer, 0.51)
    assert above["parsed_json"]["flags"]["value"] == []
    assert above["field_telemetry"]["flags"]["margin"] == pytest.approx(0.01)
    # Lowering it keeps everything selected with a wider margin.
    below = run_generation(schema, model, tokenizer, 0.4)
    assert below["parsed_json"]["flags"]["value"] == ["x", "y"]
    assert below["field_telemetry"]["flags"]["margin"] == pytest.approx(0.1)
    assert "log_scores" not in telemetry  # calibrate skips multi fields


def run_generation(schema, model, tokenizer, threshold):
    from jevmlx.engine import run_parallel_generation

    return run_parallel_generation(model, tokenizer, "ctx", schema, multi_threshold=threshold)


def test_multi_threshold_validation():
    """Threshold must live in (0, 1); the engine rejects anything else."""

    from tests.test_engine_fake import FakeModel, FakeTokenizer

    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="multi_threshold"):
            run_generation(schema, FakeModel(), FakeTokenizer(), bad)


def test_field_name_with_slash_rejected():
    with pytest.raises(ValueError, match="/"):
        StructuredSchema({"a/b": {"type": "enum", "description": "d", "choices": ["x", "y"]}})


def test_field_name_with_dot_still_rejected():
    with pytest.raises(ValueError, match="\\."):
        StructuredSchema({"a.b": {"type": "enum", "description": "d", "choices": ["x", "y"]}})
