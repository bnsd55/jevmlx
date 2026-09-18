"""W1-C regression tests: prompt profiles, PEP 604 optionals, small fixes.

Fast fake-tokenizer/model tests (no model loads); real-model coverage is in
test_engine.py's slow module.
"""

import enum
from typing import Literal

import pytest
from pydantic import BaseModel

import jevmlx.api as api
from jevmlx.engine import (
    PromptProfile,
    _chat_ids,
    _profile_for,
    _resolve_profile,
    _stop_token_ids,
    run_parallel_generation,
)
from jevmlx.schema import StructuredSchema
from tests.test_prompt_v2 import FakeModel, FakeTokenizer

# --- bug 3 / Q5: PromptProfile -------------------------------------------------


def test_profile_for_qwen3_family_disables_thinking():
    for model_id in [
        "mlx-community/Qwen3-8B-Instruct-4bit",
        "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        "qwen3-8b",  # bare name, case-insensitive
    ]:
        profile = _profile_for(model_id)
        assert profile.template_kwargs == {"enable_thinking": False}, model_id


def test_profile_for_non_qwen3_is_default():
    for model_id in [
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "mlx-community/gemma-3-12b-it-4bit",
    ]:
        assert _profile_for(model_id).template_kwargs == {}, model_id


def test_chat_ids_passes_profile_template_kwargs():
    seen_kwargs = []

    class Recording(FakeTokenizer):
        name_or_path = "recording-qwen3"

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            seen_kwargs.append(kwargs)
            return super().apply_chat_template(messages, add_generation_prompt, tokenize)

    profile = PromptProfile(template_kwargs={"enable_thinking": False})
    _chat_ids(Recording(), "user text", "system text", profile)
    assert seen_kwargs == [{"enable_thinking": False}]


def test_resolve_profile_is_self_contained_without_load_engine():
    # run_parallel_generation / run_naive_generation receive (model,
    # tokenizer) directly — from eval, serve and tests — often without
    # load_engine. _resolve_profile must derive the full profile from the
    # tokenizer alone: Qwen3-named fake gets enable_thinking=False, no
    # registry, no load.
    class Qwen3Fake(FakeTokenizer):
        name_or_path = "mlx-community/Qwen3-8B-Instruct-4bit"

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            return super().apply_chat_template(messages, add_generation_prompt, tokenize)

    profile = _resolve_profile(Qwen3Fake())
    assert profile.template_kwargs == {"enable_thinking": False}
    assert profile.supports_system  # FakeTokenizer accepts system role
    ids = _chat_ids(Qwen3Fake(), "u", "s", profile)
    assert isinstance(ids, list) and ids


def test_system_role_probe_merged_rendering():
    seen = []

    class Gemma(FakeTokenizer):
        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            seen.append(messages)
            if any(m["role"] == "system" for m in messages):
                raise __import__("jinja2.exceptions", fromlist=["TemplateError"]).TemplateError(
                    "system role not supported"
                )
            return super().apply_chat_template(messages, add_generation_prompt, tokenize)

    profile = _resolve_profile(Gemma())
    assert not profile.supports_system
    # The probe was exactly one tiny system+user rendering.
    assert [sorted(m["role"] for m in msgs) for msgs in seen] == [["system", "user"]]

    # _chat_ids with a system-rejecting profile merges into one user turn
    # and never raises.
    seen.clear()
    ids = _chat_ids(Gemma(), "user text", "system text", profile)
    assert isinstance(ids, list) and ids
    assert all(m["role"] != "system" for m in seen[-1])
    assert "system text" in seen[-1][0]["content"]


def test_parallel_generation_qwen3_named_tokenizer_without_load_engine():
    """End-to-end WITHOUT load_engine (the eval/serve/test path): a
    Qwen3-named fake tokenizer's apply_chat_template receives
    enable_thinking=False through run_parallel_generation."""
    seen_kwargs = []

    class Recording(FakeTokenizer):
        name_or_path = "mlx-community/Qwen3-8B-Instruct-4bit"

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            seen_kwargs.append(kwargs)
            return super().apply_chat_template(messages, add_generation_prompt, tokenize)

    schema = StructuredSchema({"tier": {"type": "enum", "choices": ["A", "B"], "description": "d"}})
    run_parallel_generation(FakeModel(), Recording(), "ctx", schema)
    assert seen_kwargs and all(kw == {"enable_thinking": False} for kw in seen_kwargs)


# --- bug 11: PEP 604 optionals -------------------------------------------------


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Pep604Literal(BaseModel):
    x: Literal["A", "B"] | None = None


class Pep604Enum(BaseModel):
    c: Color | None = None


class TypingOptionalLiteral(BaseModel):
    x: Literal["A", "B"] | None = None


def test_pep604_literal_optional_schema_identical():
    pep = api.schema_from_model(Pep604Literal)
    old = api.schema_from_model(TypingOptionalLiteral)
    assert (
        pep
        == old
        == {
            "x": {
                "type": "enum",
                "choices": ["A", "B"],
                "description": "x",
                "choice_descriptions": {},
            }
        }
    )


def test_pep604_enum_optional_schema():
    # The actual failure mode: pydantic keeps EnumClass | None as a raw
    # types.UnionType, which the old `is typing.Union` check rejected.
    schema = api.schema_from_model(Pep604Enum)
    assert schema["c"]["type"] == "enum"
    assert schema["c"]["choices"] == ["red", "blue"]


def test_pep604_enum_optional_allow_unknown_accepted():
    # _prepare_schema's Optional check must also accept the PEP 604 form:
    # with allow_unknown the synthetic UNKNOWN choice is appended.
    schema = api._prepare_schema(Pep604Enum, allow_unknown=True)
    assert list(schema.fields["c"].choices) == ["red", "blue", "UNKNOWN"]


def test_pep604_union_of_two_still_rejected():
    class Bad(BaseModel):
        x: Literal["A"] | Literal["B"] | None = None  # two non-None args

    with pytest.raises(TypeError, match="Optional"):
        api.schema_from_model(Bad)


# --- bug 23: duplicate load log -------------------------------------------------


def test_load_engine_logs_model_load_once(capsys):
    # Source-level check: the load_engine body contains exactly one
    # "Engine loaded" line (the old code logged it twice).
    import inspect

    from jevmlx import engine

    src = inspect.getsource(engine.load_engine)
    assert src.count('"Engine loaded in %.2fs.') == 1


# --- bug 24: unk token must not become a stop token -----------------------------


class UnkTokenizer(FakeTokenizer):
    """Mistral-style: absent token strings resolve to unk (0), not None."""

    unk_token_id = 0
    eos_token_id = 2

    def convert_tokens_to_ids(self, tok_str):
        return {"<end_of_turn>": 0, "<|im_end|>": 0, "<eos>": 2}.get(tok_str, 0)


class NoneTokenizer(FakeTokenizer):
    """Qwen-style: absent token strings resolve to None."""

    unk_token_id = None
    eos_token_id = 151643

    def convert_tokens_to_ids(self, tok_str):
        return {"<end_of_turn>": None, "<|im_end|>": 151645, "<eos>": None}.get(tok_str)


def test_unk_token_not_added_to_stop_tokens():
    stops = _stop_token_ids(UnkTokenizer())
    assert 0 not in stops  # unk excluded
    assert stops == {2}  # eos + real <eos>; absent strings mapped to unk


def test_none_returning_tokenizer_stop_tokens():
    stops = _stop_token_ids(NoneTokenizer())
    assert 151645 in stops
    assert None not in stops


# --- bug 20: naive baseline temperature argument removed -------------------------


def test_naive_generation_signature_has_no_temperature():
    import inspect

    from jevmlx.engine import run_naive_generation

    params = inspect.signature(run_naive_generation).parameters
    assert "temperature" not in params
    with pytest.raises(TypeError):
        run_naive_generation(FakeModel(), FakeTokenizer(), "ctx", None, temperature=0.2)


# --- bug 21: naive baseline shows all choices -----------------------------------


def test_json_schema_prompt_shows_all_choices():
    many = [f"choice_{i:02d}" for i in range(60)]
    schema = StructuredSchema({"big": {"type": "enum", "choices": many, "description": "d"}})
    text = schema.to_json_schema_prompt_str()
    for choice in many:
        assert f'"{choice}"' in text
    assert "..." not in text
    assert "total options" not in text
