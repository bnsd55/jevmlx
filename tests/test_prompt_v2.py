"""Prompt v2 (jevmlx-parallel-v2) contract tests: system + user messages,
hard-delimited context, and the neutral alias schema block."""

import mlx.core as mx
import pytest
from jinja2.exceptions import TemplateError

from jevmlx.engine import PROMPT_V2_SYSTEM, PROMPT_VERSION, run_parallel_generation
from jevmlx.schema import StructuredSchema, _alias_code


class FakeModel:
    """Minimal model: zeros logits, real KVCache objects sized by layers."""

    def __init__(self, vocab_size: int = 64, n_layers: int = 2):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.args = type("Args", (), {"vocab_size": vocab_size})()
        self.layers = [None] * n_layers  # make_prompt_cache counts these

    def parameters(self):
        return {}

    def __call__(self, tokens, cache=None):
        batch, seq_len = tokens.shape
        if cache is not None:
            for c in cache:
                # Drive the cache like a real layer (update_and_fetch): the
                # merge-based broadcast produces BatchKVCache whose offsets a
                # direct keys/values stomp cannot maintain.
                c.update_and_fetch(
                    mx.zeros((batch, 2, seq_len, 8)), mx.zeros((batch, 2, seq_len, 8))
                )
        return mx.zeros((batch, seq_len, self.vocab_size))


class FakeTokenizer:
    """Character tokenizer mirroring test_engine_fake's fake."""

    name_or_path = "fake-prompt"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 60 for c in text]

    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert tokenize
        return self.encode("\n".join(m["content"] for m in messages))

    def __len__(self) -> int:
        return 64


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
    block = SCHEMA.to_alias_schema_str()
    assert "A) LOW — stable income" in block  # gloss present
    assert "C) HIGH — many missed payments" in block  # gloss present
    assert "B) MEDIUM\n" in block or "B) MEDIUM " in block  # MEDIUM: no gloss part
    assert "B) MEDIUM —" not in block
    assert "A) true" in block and "B) false" in block  # boolean uses aliases too


def test_labels_schema_block_lists_real_choices():
    block = SCHEMA.to_labels_schema_str()
    assert "LOW — stable income" in block  # gloss present
    assert "A)" not in block  # labels mode shows no aliases
    assert "true | false" in block or "true  false" in block  # boolean as real values


def test_prompt_v2_sends_system_and_user():
    tok = RecordingTokenizer()
    run_parallel_generation(FakeModel(), tok, "ctx text", SCHEMA)
    roles = [m["role"] for m in tok.messages]
    assert roles == ["system", "user"]
    assert tok.messages[0]["content"] == PROMPT_V2_SYSTEM
    user = tok.messages[1]["content"]
    assert "<<<CONTEXT" in user and "ctx text" in user and "CONTEXT>>>" in user
    assert "Classify" in user


def test_prompt_version_is_v2():
    tok = FakeTokenizer()
    result = run_parallel_generation(FakeModel(), tok, "ctx", SCHEMA)
    assert result["prompt_version"] == PROMPT_VERSION == "jevmlx-parallel-v2"


def test_slot_plan_maps_aliases_to_values():
    tok = FakeTokenizer()
    plan = SCHEMA.compile_slot_plan(tok)
    p = plan["fields"]["risk_tier"]
    assert p["alias_map"] == {"A": "LOW", "B": "MEDIUM", "C": "HIGH"}
    bool_p = plan["fields"]["flag"]
    assert bool_p["alias_map"] == {"A": "true", "B": "false"}


def test_engine_assembles_real_values_from_aliases():
    tok = FakeTokenizer()
    result = run_parallel_generation(FakeModel(), tok, "ctx", SCHEMA)
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

    # The system-role rejection is probed by the self-contained
    # _resolve_profile, never inside the scoring request: the real call
    # goes out as a single user turn with the system text merged in.
    # Probing is deliberately uncached, so each generation call probes
    # once more (microseconds); seen records: probe, probe (from
    # run_parallel_generation's own resolution), merged scoring render.
    from jevmlx.engine import _resolve_profile

    tok = GemmaTokenizer()
    profile = _resolve_profile(tok)
    assert not profile.supports_system
    assert seen[-1] == [
        {"role": "system", "content": "probe"},
        {"role": "user", "content": "probe"},
    ]

    result = run_parallel_generation(FakeModel(), tok, "ctx", SCHEMA)
    assert result["prompt_version"] == "jevmlx-parallel-v2"
    # The scoring prompt is a single user turn with the merged system text.
    assert all(m["role"] != "system" for m in seen[-1])
    assert PROMPT_V2_SYSTEM in seen[-1][0]["content"]
    assert len(seen) == 3  # probe (test) + probe (generation) + one merged render
    # No system-role message ever reached a scoring render: only the two
    # probes (seen[0], seen[1]) contain a system role, and both raised.
    assert not any(m["role"] == "system" for m in seen[2])


def test_scoring_accepts_only_slots_and_labels():
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="scoring"):
        run_parallel_generation(FakeModel(), tok, "ctx", SCHEMA, scoring="trie")
    with pytest.raises(ValueError, match="scoring"):
        run_parallel_generation(FakeModel(), tok, "ctx", SCHEMA, scoring="letters")
