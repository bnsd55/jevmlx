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

import benchmarks.golden_prompts as gp
from jevmlx.engine import PROMPT_VERSION

VECTORS = sorted((Path(__file__).parent / "golden" / "prompts").glob("*.json"))


def test_vectors_exist():
    """One vector per (profile, tokenizer revision, representative case):
    qwen2.5 + gemma on the real tokenizer, qwen3 on the fake (profile-only).
    """
    names = {p.stem for p in VECTORS}
    assert any(n.startswith("qwen2.5__a5339a41") for n in names), names
    assert any(n.startswith("gemma__") for n in names), names
    assert any(n.startswith("qwen3__fake__") for n in names), names
    assert len(names) == len(VECTORS) == 3


def test_every_vector_pins_version_profile_revision():
    for path in VECTORS:
        vector = json.loads(path.read_text(encoding="utf-8"))
        assert vector["prompt_version"] == PROMPT_VERSION
        assert vector["profile"] and vector["tokenizer_revision"] and vector["case"]
        # The full provenance triple: prompt sha, plan hash, token-ids sha.
        for key in ("prompt_sha256", "plan_hash", "token_ids_sha256"):
            assert len(vector[key]) == 64, (path.name, key)
        # Input block carries everything a re-render needs.
        assert vector["input"]["system"] and "schema" in vector["input"]
        assert vector["input"]["scoring"] in ("slots", "labels")
        assert "template_kwargs" in vector["input"] and "supports_system" in vector["input"]


def test_committed_vectors_match_the_renderer():
    """NOT circular: the committed file is the only expected; re-render and
    diff bytes."""
    assert gp.check_vectors() == []


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


def test_vector_detects_a_renderer_change(tmp_path, monkeypatch):
    """A renderer change must make --check fail (the diff is the alarm, not
    a silent regen)."""
    real = gp.build_vector

    def drifted_build_vector(profile, case, tok, revision):
        vector = real(profile, case, tok, revision)
        vector["rendered_text"] = vector["rendered_text"] + "\nDRIFT"
        return vector

    monkeypatch.setattr(gp, "build_vector", drifted_build_vector)
    problems = gp.check_vectors()
    assert problems and all("stale" in p and "rendered_text" in p for p in problems)


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
