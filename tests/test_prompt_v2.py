"""Prompt v2 (jevmlx-parallel-v10) contract tests: system + user messages,
hard-delimited context, and the neutral alias schema block."""

import pytest
from conftest import FakeModel, FakeTokenizer, make_engine
from jinja2.exceptions import TemplateError

from jevmlx.engine import PROMPT_V2_SYSTEM, PROMPT_VERSION, run_parallel_generation
from jevmlx.schema import StructuredSchema, _alias_code
from tests.conftest import make_test_renderer

SCHEMA = StructuredSchema(
    {
        "risk_tier": {
            "type": "enum",
            "description": "Credit risk tier",
            "choices": ["LOW", "MEDIUM", "HIGH"],
            "choice_descriptions": {"LOW": "stable income", "HIGH": "many missed payments"},
        },
        "flag": {"type": "boolean", "description": "manually flagged"},
    }
)


class RecordingTokenizer(FakeTokenizer):
    """Captures the messages handed to apply_chat_template."""

    def __init__(self):
        super().__init__()
        self.messages = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        self.messages = messages
        return super().apply_chat_template(messages, add_generation_prompt, tokenize)


def test_alias_codes_base26():
    assert _alias_code(0) == "A"
    assert _alias_code(25) == "Z"
    assert _alias_code(26) == "AA"
    assert _alias_code(27) == "AB"
    assert _alias_code(701) == "ZZ"


def test_alias_schema_block_lists_aliases_with_gloss():
    tok = FakeTokenizer()
    block = SCHEMA.to_alias_schema_str(tok)
    assert 'A) "LOW" — "stable income"' in block  # gloss present, json.dumps-escaped
    assert 'C) "HIGH" — "many missed payments"' in block  # gloss present
    assert 'B) "MEDIUM"\n' in block or 'B) "MEDIUM" ' in block  # MEDIUM: no gloss part
    assert 'B) "MEDIUM" —' not in block
    assert 'A) "true"' in block and 'B) "false"' in block  # boolean uses aliases too


def test_labels_schema_block_lists_real_choices():
    block = SCHEMA.to_labels_schema_str()
    assert '"LOW" — "stable income"' in block  # gloss present, json.dumps-escaped
    assert "A)" not in block  # labels mode shows no aliases
    assert '"true"' in block and '"false"' in block  # boolean as real values


def test_prompt_v2_sends_system_and_user():
    tok = RecordingTokenizer()
    run_parallel_generation(make_engine(FakeModel(), tok), "ctx text", SCHEMA)
    roles = [m["role"] for m in tok.messages]
    assert roles == ["system", "user"]
    assert tok.messages[0]["content"] == PROMPT_V2_SYSTEM
    user = tok.messages[1]["content"]
    # W5-A finding 44: the delimiter carries the deterministic per-context
    # nonce on both fences (old bare CONTEXT>>> could be impersonated by an
    # interior context line).
    import hashlib

    tag = "C" + hashlib.sha256(b"ctx text").hexdigest()[:16]
    assert f"<<<CONTEXT:{tag}" in user and f"CONTEXT:{tag}>>>" in user
    assert "ctx text" in user
    assert "Classify" in user


def test_prompt_version_is_v10():
    tok = FakeTokenizer()
    result = run_parallel_generation(make_engine(FakeModel(), tok), "ctx", SCHEMA)
    assert result["prompt_version"] == PROMPT_VERSION == "jevmlx-parallel-v10"


def test_slot_plan_maps_aliases_to_values():
    tok = FakeTokenizer()
    plan = SCHEMA.compile_slot_plan(tok, make_test_renderer(tok, SCHEMA, "slots"))
    p = plan["fields"]["risk_tier"]
    assert p["alias_map"] == {"A": "LOW", "B": "MEDIUM", "C": "HIGH"}
    bool_p = plan["fields"]["flag"]
    assert bool_p["alias_map"] == {"A": "true", "B": "false"}


def test_engine_assembles_real_values_from_aliases():
    tok = FakeTokenizer()
    result = run_parallel_generation(make_engine(FakeModel(), tok), "ctx", SCHEMA)
    telemetry = result["field_telemetry"]["risk_tier"]
    # FakeModel's distribution makes the last-listed choice win; whatever it
    # is, the assembled value and the log_scores keys must be REAL values.
    assert telemetry["value"] in {"LOW", "MEDIUM", "HIGH"}
    assert set(telemetry["log_scores"]) == {"LOW", "MEDIUM", "HIGH"}


def test_gemma_style_template_rejects_system_role():
    seen = []

    class GemmaTokenizer(FakeTokenizer):
        name_or_path = "gemma-fake"

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
            seen.append(messages)
            if any(m["role"] == "system" for m in messages):
                raise TemplateError("system role not supported")
            return super().apply_chat_template(messages, add_generation_prompt, tokenize)

    # The system-role rejection is probed ONCE at Engine build, never inside
    # the scoring request: the real call goes out as a single user turn with
    # the system text merged in (the engine carries the resolved profile).
    from jevmlx.engine import _probe_system_role, _profile_for

    tok = GemmaTokenizer()
    profile = _probe_system_role(tok, _profile_for("test/gemma"))
    assert not profile.supports_system
    assert seen[-1] == [
        {"role": "system", "content": "probe"},
        {"role": "user", "content": "probe"},
    ]

    result = run_parallel_generation(make_engine(FakeModel(), tok), "ctx", SCHEMA)
    assert result["prompt_version"] == "jevmlx-parallel-v10"
    # The scoring prompt is a single user turn with the merged system text.
    assert all(m["role"] != "system" for m in seen[-1])
    assert PROMPT_V2_SYSTEM in seen[-1][0]["content"]
    assert len(seen) == 4  # probe (test) + probe (renderer) + one render per field
    # No system-role message ever reached a scoring render: only the two
    # probes (seen[0], seen[1]) contain a system role, and both raised.
    assert not any(m["role"] == "system" for m in seen[2])


def test_scoring_accepts_only_slots_and_labels():
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="scoring"):
        run_parallel_generation(make_engine(FakeModel(), tok), "ctx", SCHEMA, scoring="trie")
    with pytest.raises(ValueError, match="scoring"):
        run_parallel_generation(make_engine(FakeModel(), tok), "ctx", SCHEMA, scoring="letters")
