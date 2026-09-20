"""Golden prompt vectors (W5c-5, GPT-review B3).

The byte-exact unit is prompt version x profile x tokenizer revision x
representative request. These tests compare the COMMITTED vector bytes to
the live renderer — never regenerating both sides through the same code
path (not circular). The qwen2.5/gemma vectors render with the REAL
transformers tokenizer (cheap; no model, no mlx); the qwen3 vector pins
the profile's template kwargs against the fake tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmarks.golden_prompts as gp
from jevmlx.engine import PROMPT_VERSION, PROMPT_VERSION_MESSAGES

VECTORS = sorted((Path(__file__).parent / "golden" / "prompts").glob("*.json"))

# F3: the real-tokenizer vectors (qwen2.5, gemma) load PINNED tokenizer
# files from the HF cache — the network marker keeps the fast CI suite
# offline-clean; CI verifies them via the golden-prompt --check step (which
# runs with an actions/cache on the hub dir). Fake-tokenizer vectors run
# everywhere.
network = pytest.mark.network


@network
def test_vectors_exist():
    """One vector per (profile, tokenizer revision, representative case):
    3 profiles (qwen2.5 + gemma on the real tokenizer, qwen3 on the fake)
    x 3 cases (slots, labels, multi) = 9 vectors.
    """
    names = {p.stem for p in VECTORS}
    assert any(n.startswith("qwen2.5__a5339a41") for n in names), names
    assert any(n.startswith("gemma__") for n in names), names
    assert any(n.startswith("qwen3__fake__") for n in names), names
    cases = ("risk_enum_bool", "risk_enum_bool_labels", "tags_multi")
    for case in cases:
        assert sum(1 for n in names if n.endswith(f"__{case}")) == 3, (case, names)
    assert len(names) == len(VECTORS) == 12  # 9 string-form + 3 messages-form


_VALID_VERSIONS = (PROMPT_VERSION, PROMPT_VERSION_MESSAGES)


def test_every_vector_pins_version_profile_revision():
    for path in VECTORS:
        vector = json.loads(path.read_text(encoding="utf-8"))
        assert vector["prompt_version"] in _VALID_VERSIONS, (path.name, vector["prompt_version"])
        assert vector["profile"] and vector["tokenizer_revision"] and vector["case"]
        # The full provenance triple: prompt sha, plan hash, token-ids sha.
        for key in ("prompt_sha256", "plan_hash", "token_ids_sha256"):
            assert len(vector[key]) == 64, (path.name, key)
        # Input block carries everything a re-render needs.
        assert vector["input"]["system"] and "schema" in vector["input"]
        assert vector["input"]["scoring"] in ("slots", "labels")
        assert "template_kwargs" in vector["input"] and "supports_system" in vector["input"]
        # W6-B2: the messages-form vectors carry a message list; the string
        # form carries a context string — never both, never neither.
        if vector["prompt_version"] == PROMPT_VERSION_MESSAGES:
            assert "messages" in vector["input"] and "context" not in vector["input"]
        else:
            assert "context" in vector["input"] and "messages" not in vector["input"]


@network
def test_committed_vectors_match_the_renderer():
    """NOT circular: the committed file is the only expected; re-render and
    diff bytes."""
    assert gp.check_vectors() == []


@network
def test_gemma_vector_merges_system_into_user_turn():
    """supports_system=false profile: the rendered text is a single user
    turn whose content starts with the system text + blank line."""
    vector = next(
        json.loads(p.read_text(encoding="utf-8")) for p in VECTORS if p.stem.startswith("gemma__")
    )
    assert vector["input"]["supports_system"] is False
    rendered = vector["rendered_text"]
    assert "<start_of_turn>user" in rendered
    assert rendered.index(vector["input"]["system"]) < rendered.index(
        "Classify the following fields."
    )
    assert "<|im_start|>system" not in rendered  # no dedicated system turn


@network
def test_qwen25_vector_has_dedicated_system_turn():
    vector = next(
        json.loads(p.read_text(encoding="utf-8")) for p in VECTORS if p.stem.startswith("qwen2.5__")
    )
    assert vector["input"]["supports_system"] is True
    rendered = vector["rendered_text"]
    assert rendered.startswith("<|im_start|>system")
    assert "<|im_start|>assistant" in rendered  # generation marker ends the prompt


def test_qwen3_vector_pins_thinking_off():
    vector = next(
        json.loads(p.read_text(encoding="utf-8"))
        for p in VECTORS
        if p.stem.startswith("qwen3__fake__")
    )
    assert vector["input"]["template_kwargs"] == {"enable_thinking": False}


@network
def test_labels_vector_shows_real_choices_no_aliases():
    vector = next(
        json.loads(p.read_text(encoding="utf-8"))
        for p in VECTORS
        if "__risk_enum_bool_labels" in p.stem and p.stem.startswith("qwen2.5__")
    )
    assert vector["input"]["scoring"] == "labels"
    rendered = vector["rendered_text"]
    assert '"LOW" — "stable income"' in rendered
    assert "A)" not in rendered  # labels mode shows no aliases


def test_multi_vector_renders_count_and_yn_menu():
    vector = next(
        json.loads(p.read_text(encoding="utf-8"))
        for p in VECTORS
        if "__tags_multi" in p.stem and p.stem.startswith("qwen3__")
    )
    rendered = vector["rendered_text"]
    assert "select all that apply" in rendered
    assert "Y" in rendered and "N" in rendered


@network
def test_cold_offline_cache_miss_is_a_clear_error(capsys):
    """F3: a cache miss exits with a clear message (and the local remedy),
    never a raw traceback."""
    with pytest.raises(SystemExit) as excinfo:
        gp._real_tokenizer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", "0" * 40)
    message = str(excinfo.value)
    assert "tokenizer not available" in message
    assert "huggingface-cli download" in message


def test_vector_detects_a_renderer_change(monkeypatch):
    """A renderer change must make --check fail (the diff is the alarm, not
    a silent regen). Uses ONLY fake-profile vectors (drift applied to qwen3,
    whose vector never touches the Hub) so this runs in the offline fast
    suite."""
    real = gp.build_vector

    def drifted_build_vector(profile, case, tok, revision):
        vector = real(profile, case, tok, revision)
        if profile == "qwen3":  # the fake-profile vector: no Hub access
            vector["rendered_text"] = vector["rendered_text"] + "\nDRIFT"
        return vector

    monkeypatch.setattr(gp, "build_vector", drifted_build_vector)
    problems = [p for p in gp.check_vectors() if "qwen3" in p]
    assert problems and all("stale" in p and "rendered_text" in p for p in problems)


@network
def test_stale_revision_vector_fails_check(tmp_path):
    """A committed vector from a revision the renderer no longer resolves is
    stale by definition: --check must fail, not silently ignore it."""
    stale = tmp_path / ("qwen2.5__00000000000000000000000000000000__risk_enum_bool.json")
    stale.write_text(json.dumps({"prompt_version": PROMPT_VERSION}), encoding="utf-8")
    import benchmarks.golden_prompts as g

    original = g.GOLDEN_DIR
    g.GOLDEN_DIR = tmp_path
    try:
        problems = g.check_vectors()
    finally:
        g.GOLDEN_DIR = original
    assert any(stale.name in p and "no longer matches" in p for p in problems)
