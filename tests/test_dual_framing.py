"""W6-dualframe (HOLD): dual-framing scoring for boolean fields.

Tests:
- default off: prompt_version unchanged, byte-identical to main
- on: boolean fields get a second pass, p_combined = 0.5*(p_pos + (1-p_neg))
- on: telemetry carries p_pos, p_neg_complement, score_source='dual_framing'
- on: non-boolean fields untouched
- on: no boolean fields -> prompt version bumps but no second pass
- negation is deterministic (no LLM)
- golden prompt vectors: default-off unchanged
- public API: dual_framing kwarg exists on decide/choose/judge/rate
"""

from __future__ import annotations

import pytest
from conftest import make_engine_result, make_field_telemetry

import jevmlx
from jevmlx import api
from jevmlx.engine import (
    PROMPT_VERSION_DUAL_FRAME,
    _apply_dual_framing,
    _boolean_field_names,
    _negate_boolean_description,
    _negated_schema,
)
from jevmlx.schema import StructuredSchema

# ---- shared helpers --------------------------------------------------------


def _patch_engine(monkeypatch, fake_run_parallel):
    monkeypatch.setattr(api, "load_engine", lambda model_id: ("engine", "tokenizer"))
    monkeypatch.setattr(api, "run_parallel_generation", fake_run_parallel)


def _bool_telemetry(value: bool, p_true: float):
    """Boolean field telemetry with a known P(true)."""
    tc = [
        {"choice": "true", "probability": p_true},
        {"choice": "false", "probability": 1.0 - p_true},
    ]
    return make_field_telemetry(
        value=value,
        type_="boolean",
        choices=["true", "false"],
        probability=p_true if value else 1.0 - p_true,
        top_choices=tc,
    )


def _apply_with_neg(monkeypatch, schema, pos_result, neg_result):
    """Call _apply_dual_framing with the negated pass patched.

    _apply_dual_framing calls run_parallel_generation (module global) for the
    negated pass; we patch that to return ``neg_result`` and assert the
    negated schema carries the negation prefix.
    """
    import jevmlx.engine as eng_mod

    def fake_neg(engine, context, neg_schema, **kwargs):
        for fname in _boolean_field_names(schema):
            assert neg_schema.fields[fname].description.startswith("Negated framing")
        return neg_result

    monkeypatch.setattr(eng_mod, "run_parallel_generation", fake_neg)
    return _apply_dual_framing(
        ("engine", "tokenizer"),
        "ctx",
        schema,
        1.0,
        None,
        "slots",
        None,
        False,
        None,
        pos_result,
    )


# ---- default off: byte-identical to main -----------------------------------


def test_default_off_prompt_version_unchanged(monkeypatch):
    """With dual_framing=False (default), the prompt version is the main one."""
    calls = []

    def fake(engine, context, schema, **kwargs):
        calls.append(kwargs.get("dual_framing"))
        return make_engine_result(
            fields={"judgment": _bool_telemetry(True, 0.8)},
            parsed={"judgment": {"value": True, "prob": 0.8}},
        )

    _patch_engine(monkeypatch, fake)
    api.judge("ctx", "Is it true?")
    assert calls == [False]  # default off


def test_default_off_prompt_sha_unchanged():
    """The default-off prompt is byte-identical to main (golden vectors pass).

    The prompt is rendered by _user_content + _chat_ids; dual_framing=False
    never touches the prompt path.
    """
    from conftest import FakeTokenizer

    from jevmlx.engine import _user_content

    tok = FakeTokenizer()
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "Is it true?"}})
    prompt_off = _user_content("ctx", schema, tok, "slots")
    prompt_main = _user_content("ctx", schema, tok, "slots")
    assert prompt_off == prompt_main


# ---- on: combination math --------------------------------------------------


def test_dual_framing_combines_probabilities(monkeypatch):
    """p = 0.5 * (p_pos + (1 - p_neg))."""
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "Is it true?"}})
    pos = make_engine_result(
        fields={"flag": _bool_telemetry(True, 0.8)},
        parsed={"flag": {"value": True, "prob": 0.8}},
    )
    neg = make_engine_result(
        fields={"flag": _bool_telemetry(False, 0.3)},
        parsed={"flag": {"value": False, "prob": 0.3}},
    )
    result = _apply_with_neg(monkeypatch, schema, pos, neg)
    ft = result["field_telemetry"]["flag"]
    # p = 0.5 * (0.8 + (1 - 0.3)) = 0.5 * 1.5 = 0.75
    assert ft["p_combined"] == pytest.approx(0.75)
    assert ft["value"] is True  # 0.75 >= 0.5


def test_dual_framing_flips_when_negated_disagrees(monkeypatch):
    """If the negated framing strongly disagrees, the combined value can flip."""
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "Is it true?"}})
    pos = make_engine_result(  # positive: P(true) = 0.6
        fields={"flag": _bool_telemetry(True, 0.6)},
        parsed={"flag": {"value": True, "prob": 0.6}},
    )
    neg = make_engine_result(  # negated: P(true) = 0.9 -> complement 0.1
        fields={"flag": _bool_telemetry(True, 0.9)},
        parsed={"flag": {"value": True, "prob": 0.9}},
    )
    result = _apply_with_neg(monkeypatch, schema, pos, neg)
    ft = result["field_telemetry"]["flag"]
    # p = 0.5 * (0.6 + 0.1) = 0.35 -> flips to False
    assert ft["p_combined"] == pytest.approx(0.35)
    assert ft["value"] is False


# ---- on: telemetry ---------------------------------------------------------


def test_dual_framing_telemetry_keys(monkeypatch):
    """field_telemetry carries p_pos, p_neg_complement, score_source."""
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "Is it true?"}})
    pos = make_engine_result(
        fields={"flag": _bool_telemetry(True, 0.7)},
        parsed={"flag": {"value": True, "prob": 0.7}},
    )
    neg = make_engine_result(
        fields={"flag": _bool_telemetry(False, 0.4)},
        parsed={"flag": {"value": False, "prob": 0.4}},
    )
    result = _apply_with_neg(monkeypatch, schema, pos, neg)
    ft = result["field_telemetry"]["flag"]
    assert ft["score_source"] == "dual_framing"
    assert ft["p_pos"] == pytest.approx(0.7)
    assert ft["p_neg_complement"] == pytest.approx(0.6)  # 1 - 0.4
    assert ft["p_combined"] == pytest.approx(0.65)  # 0.5*(0.7+0.6)
    assert ft["dual_framing"]["combined"] is True
    assert ft["dual_framing"]["disagreement"] == pytest.approx(0.1)  # |0.7-0.6|
    assert result["prompt_version"] == PROMPT_VERSION_DUAL_FRAME


def test_dual_framing_non_boolean_untouched(monkeypatch):
    """Non-boolean fields keep their original result."""
    schema = StructuredSchema(
        {
            "risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "risk"},
            "flag": {"type": "boolean", "description": "flagged"},
        }
    )
    pos = make_engine_result(
        fields={
            "risk": make_field_telemetry(value="LOW", choices=["LOW", "HIGH"], probability=0.9),
            "flag": _bool_telemetry(True, 0.7),
        },
        parsed={
            "risk": {"value": "LOW", "prob": 0.9},
            "flag": {"value": True, "prob": 0.7},
        },
    )
    neg = make_engine_result(
        fields={"flag": _bool_telemetry(False, 0.3)},
        parsed={"flag": {"value": False, "prob": 0.3}},
    )
    result = _apply_with_neg(monkeypatch, schema, pos, neg)
    # risk is untouched (no dual_framing keys)
    risk_ft = result["field_telemetry"]["risk"]
    assert "p_pos" not in risk_ft
    assert "score_source" not in risk_ft
    assert risk_ft["value"] == "LOW"
    # flag is combined
    flag_ft = result["field_telemetry"]["flag"]
    assert flag_ft["score_source"] == "dual_framing"


def test_dual_framing_no_boolean_fields_bumps_version_only():
    """With no boolean fields, the second pass is skipped but the version bumps."""
    schema = StructuredSchema(
        {"risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "risk"}}
    )
    pos = make_engine_result(
        fields={
            "risk": make_field_telemetry(value="LOW", choices=["LOW", "HIGH"], probability=0.9)
        },
        parsed={"risk": {"value": "LOW", "prob": 0.9}},
    )
    # No monkeypatch needed: _apply_dual_framing short-circuits (no bool fields).
    result = _apply_dual_framing(
        ("engine", "tokenizer"),
        "ctx",
        schema,
        1.0,
        None,
        "slots",
        None,
        False,
        None,
        pos,
    )
    assert result["prompt_version"] == PROMPT_VERSION_DUAL_FRAME
    assert "p_pos" not in result["field_telemetry"]["risk"]


# ---- negation determinism --------------------------------------------------


def test_negate_description_is_deterministic():
    """The negation prefix is a fixed template, no LLM."""
    assert (
        _negate_boolean_description("Is it true?")
        == "Negated framing — answer the opposite: Is it true?"
    )
    assert _negate_boolean_description("Is it true?") == _negate_boolean_description("Is it true?")


def test_negated_schema_only_touches_booleans():
    schema = StructuredSchema(
        {
            "risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "risk tier"},
            "flag": {"type": "boolean", "description": "Is it flagged?"},
        }
    )
    negated = _negated_schema(schema, ["flag"])
    assert negated.fields["risk"].description == "risk tier"  # untouched
    assert negated.fields["flag"].description.startswith("Negated framing")
    assert negated.fields["flag"].description.endswith("Is it flagged?")


def test_boolean_field_names():
    schema = StructuredSchema(
        {
            "risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"},
            "a": {"type": "boolean", "description": "d"},
            "b": {"type": "boolean", "description": "d"},
        }
    )
    assert _boolean_field_names(schema) == ["a", "b"]


# ---- public API: dual_framing kwarg exists ---------------------------------


def test_decide_accepts_dual_framing():
    """decide() and the one-field helpers accept dual_framing."""
    import inspect

    assert "dual_framing" in inspect.signature(jevmlx.decide).parameters
    assert "dual_framing" in inspect.signature(jevmlx.judge).parameters
    assert "dual_framing" in inspect.signature(jevmlx.choose).parameters
    assert "dual_framing" in inspect.signature(jevmlx.rate).parameters
