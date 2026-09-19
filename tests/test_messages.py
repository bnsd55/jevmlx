"""W6-B2: the messages form of context (v10-messages prompt version).

Unit tests on the protocol block construction, the chat-template rendering
(supports_system + no-system merge), validation, the parity between the
string form and the messages form (same fields/values, different
prompt_sha256 + prompt_version), and the batched path + prior correction
in the messages form. No live model.
"""

from __future__ import annotations

import pytest
from conftest import FakeModel, FakeTokenizer, FakeTokenizerNoSystem, make_engine

from jevmlx.engine import (
    PROMPT_VERSION,
    PROMPT_VERSION_MESSAGES,
    PROTOCOL_DELIMITER,
    _chat_ids_messages,
    _probe_system_role,
    _profile_for,
    _protocol_messages,
    run_parallel_generation,
    run_parallel_generation_batched,
    validate_messages,
)
from jevmlx.schema import StructuredSchema

SCHEMA = StructuredSchema(
    {
        "risk_tier": {
            "type": "enum",
            "description": "tier",
            "choices": ["LOW", "HIGH"],
        },
    }
)


# ---- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        [{"role": "tool", "content": "x"}],  # exotic role
        [{"role": "user", "content": [{"type": "image"}]}],  # image content
        [{"role": "user", "content": None}],  # null content
        [{"role": "user", "content": ""}],  # empty content
        [{"role": "user", "content": "x", "tool_calls": []}],  # extra keys
        [{"role": "assistant", "content": "only assistant"}],  # no user
        [],  # empty list
    ],
)
def test_validate_messages_rejects(bad):
    with pytest.raises(ValueError):
        validate_messages(bad)


def test_validate_messages_rejects_non_list():
    with pytest.raises(ValueError, match="non-empty list"):
        validate_messages("just a string")  # type: ignore[arg-type]


def test_validate_messages_returns_fresh_copy():
    src = [{"role": "user", "content": "x"}]
    out = validate_messages(src)
    assert out is not src and out[0] is not src[0]


# ---- protocol block construction --------------------------------------------


def test_protocol_block_is_library_owned_system_message():
    """The protocol block is the first message, role system, library-owned:
    caller system text concatenates INTO it behind the delimiter, never
    replaces or precedes it. All other messages keep their order."""
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "app pays late"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "final"},
    ]
    out = _protocol_messages(validate_messages(msgs), SCHEMA, None, "labels", None)
    assert len(out) == 4
    assert out[0]["role"] == "system"
    assert "You are a classifier" in out[0]["content"]
    # protocol block comes first, caller system text after the delimiter
    assert out[0]["content"].index("You are a classifier") < out[0]["content"].index("be terse")
    assert PROTOCOL_DELIMITER in out[0]["content"]
    # non-system messages preserved verbatim and in order
    assert [m["content"] for m in out[1:]] == ["app pays late", "ok", "final"]
    assert [m["role"] for m in out[1:]] == ["user", "assistant", "user"]


def test_protocol_block_no_caller_system():
    """A caller conversation with no system turn still gets the protocol
    system message prepended."""
    out = _protocol_messages(
        validate_messages([{"role": "user", "content": "x"}]), SCHEMA, None, "labels", None
    )
    assert out[0]["role"] == "system"
    assert PROTOCOL_DELIMITER not in out[0]["content"]
    assert out[1] == {"role": "user", "content": "x"}


def test_protocol_block_multiple_caller_systems_concat_in_order():
    out = _protocol_messages(
        validate_messages(
            [
                {"role": "system", "content": "first"},
                {"role": "user", "content": "u"},
                {"role": "system", "content": "second"},
            ]
        ),
        SCHEMA,
        None,
        "labels",
        None,
    )
    # exactly one system turn: protocol + delimiter + "first" + delimiter + "second"
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"].index("first") < out[0]["content"].index("second")


# ---- rendering: system + no-system merge ------------------------------------


def test_chat_ids_messages_passes_profile_template_kwargs():
    seen: list[dict] = []

    class Recording(FakeTokenizer):
        name_or_path = "recording-qwen3"

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            seen.append(kwargs)
            return super().apply_chat_template(messages, add_generation_prompt, tokenize)

    profile = _profile_for("mlx-community/Qwen3-8B-Instruct-4bit")
    _chat_ids_messages(
        Recording(),
        _protocol_messages(
            validate_messages([{"role": "user", "content": "x"}]), SCHEMA, None, "labels", profile
        ),
        profile,
    )
    assert seen == [{"enable_thinking": False}]


def test_chat_ids_messages_no_system_merges_into_first_user():
    """A no-system profile: the (single) protocol system turn merges into
    the FIRST user turn with \\n\\n — the template never sees a system role,
    and the remaining messages keep their order."""
    seen: list[list[dict]] = []

    class Gemma(FakeTokenizerNoSystem):
        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            seen.append([dict(m) for m in messages])
            return super().apply_chat_template(messages, add_generation_prompt, tokenize)

    profile = _probe_system_role(Gemma(), _profile_for("test/gemma"))
    assert not profile.supports_system
    _chat_ids_messages(
        Gemma(),
        _protocol_messages(
            validate_messages(
                [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "app"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "final"},
                ]
            ),
            SCHEMA,
            None,
            "labels",
            profile,
        ),
        profile,
    )
    rendered = seen[-1]
    assert all(m["role"] != "system" for m in rendered)
    assert rendered[0]["role"] == "user"
    # the merged first user turn carries the protocol block, the delimiter
    # and the caller system text, then the original user content
    first = rendered[0]["content"]
    assert "You are a classifier" in first and "be terse" in first and "app" in first
    assert first.index("You are a classifier") < first.index("be terse") < first.index("app")
    # the remaining messages keep their order
    assert [m["role"] for m in rendered[1:]] == ["assistant", "user"]


# ---- parity: string form vs messages form -----------------------------------


def _engine():
    return make_engine(FakeModel(), FakeTokenizer())


def test_parity_messages_vs_string_same_fields_different_prompt():
    """The messages form goes through the SAME _prefill/_score_rows path:
    same fields decided, same values (the fake's logits are constant), but
    a DIFFERENT prompt_version and prompt_sha256."""
    str_result = run_parallel_generation(_engine(), "ctx text", SCHEMA)
    msg_result = run_parallel_generation(
        _engine(), [{"role": "user", "content": "ctx text"}], SCHEMA
    )
    assert str_result["prompt_version"] == PROMPT_VERSION == "jevmlx-parallel-v9"
    assert (
        msg_result["prompt_version"] == PROMPT_VERSION_MESSAGES == ("jevmlx-parallel-v10-messages")
    )
    assert str_result["prompt_sha256"] != msg_result["prompt_sha256"]
    # same field set, same decided values
    assert set(str_result["field_telemetry"]) == set(msg_result["field_telemetry"])
    for fname in str_result["field_telemetry"]:
        assert (
            str_result["field_telemetry"][fname]["value"]
            == msg_result["field_telemetry"][fname]["value"]
        )


def test_messages_form_renders_no_nonce_fence():
    """The messages form does NOT use the nonce fence — the template's own
    turn boundaries are the boundary (the protocol block is a system turn,
    the data is the caller's user turns)."""

    class Recording(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.seen: list[str] = []

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            self.seen.extend(m["content"] for m in messages)
            return super().apply_chat_template(messages, add_generation_prompt, tokenize)

    tok = Recording()
    run_parallel_generation(
        make_engine(FakeModel(), tok), [{"role": "user", "content": "ctx"}], SCHEMA
    )
    rendered = "\n".join(tok.seen)
    assert "<<<CONTEXT" not in rendered and "CONTEXT:" not in rendered
    assert "ctx" in rendered


def test_messages_form_neutral_prior_renders_same_form():
    """prior_correction in the messages form: the neutral pass is the
    messages-form neutral (a single user turn with '(no context provided)'),
    not the v9 string neutral — the prior cache key captures that."""
    result = run_parallel_generation(
        _engine(),
        [{"role": "user", "content": "evidence"}],
        SCHEMA,
        prior_correction=True,
    )
    assert result["prompt_version"] == PROMPT_VERSION_MESSAGES
    assert result["prior_correction"] is True


# ---- batched path ------------------------------------------------------------


def test_batched_accepts_mixed_forms():
    """decide_many-style batched path: contexts may MIX str and messages;
    each result reports its own form's prompt_version."""
    engine = _engine()
    results = run_parallel_generation_batched(
        engine,
        ["string ctx", [{"role": "user", "content": "messages ctx"}]],
        SCHEMA,
    )
    assert len(results) == 2
    assert results[0]["prompt_version"] == PROMPT_VERSION
    assert results[1]["prompt_version"] == PROMPT_VERSION_MESSAGES
    assert results[0]["prompt_sha256"] != results[1]["prompt_sha256"]


def test_batched_messages_prior_correction_per_form():
    """A mixed-form batched call with prior_correction computes a prior per
    form (two neutral renders) and routes each context the matching one."""
    engine = _engine()
    results = run_parallel_generation_batched(
        engine,
        ["string ctx", [{"role": "user", "content": "messages ctx"}]],
        SCHEMA,
        prior_correction=True,
    )
    assert results[0]["prior_correction"] and results[1]["prior_correction"]
    assert results[0]["prompt_version"] == PROMPT_VERSION
    assert results[1]["prompt_version"] == PROMPT_VERSION_MESSAGES
