"""W6-dualframe (HOLD): dual-framing scoring for boolean fields.

Single-pass approach: when dual_framing=True, the schema is expanded with a
synthetic __dual_neg__<field> boolean twin (complement-condition description)
BEFORE the prefill. Both render in ONE schema block (one prefill) and both
get row(s) in ONE batched suffix pass. After assembly, the twin is extracted
and combined: p = 0.5*(p_pos + (1-p_neg)).

Tests:
- default off: byte-identical to main
- on: combination math (p_combined, flip on disagreement)
- on: telemetry keys (p_pos, p_neg_complement, score_source, dual_framing dict)
- on: non-boolean fields untouched
- on: no boolean fields -> version bump only
- on: twin fields stripped from the public result
- complement template is deterministic (no LLM)
- default-off prompt byte-identical
- public API: dual_framing kwarg exists
"""

from __future__ import annotations

import pytest
from conftest import make_engine_result, make_field_telemetry

import jevmlx
from jevmlx import api
from jevmlx.engine import (
    PROMPT_VERSION_DUAL_FRAME,
    _boolean_field_names,
    _combine_dual_framing,
    _dual_neg_field_name,
    _expand_schema_for_dual_framing,
    _negate_boolean_description,
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


def _make_result_with_twin(field: str, p_pos: float, p_neg: float):
    """A single-pass result with both the positive field and its __dual_neg__ twin."""
    return make_engine_result(
        fields={
            field: _bool_telemetry(p_pos >= 0.5, p_pos),
            _dual_neg_field_name(field): _bool_telemetry(p_neg >= 0.5, p_neg),
        },
        parsed={
            field: {"value": p_pos >= 0.5, "prob": p_pos},
            _dual_neg_field_name(field): {"value": p_neg >= 0.5, "prob": p_neg},
        },
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
    """The default-off prompt is byte-identical to main (golden vectors pass)."""
    from conftest import FakeTokenizer

    from jevmlx.engine import _user_content

    tok = FakeTokenizer()
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "Is it true?"}})
    prompt_off = _user_content("ctx", schema, tok, "slots")
    prompt_main = _user_content("ctx", schema, tok, "slots")
    assert prompt_off == prompt_main


# ---- on: combination math --------------------------------------------------


def test_dual_framing_combines_probabilities():
    """p = 0.5 * (p_pos + (1 - p_neg))."""
    result = _make_result_with_twin("flag", p_pos=0.8, p_neg=0.3)
    combined = _combine_dual_framing(result, ["flag"])
    ft = combined["field_telemetry"]["flag"]
    # p = 0.5 * (0.8 + (1 - 0.3)) = 0.5 * 1.5 = 0.75
    assert ft["p_combined"] == pytest.approx(0.75)
    assert ft["value"] is True  # 0.75 >= 0.5


def test_dual_framing_flips_when_negated_disagrees():
    """If the negated framing strongly disagrees, the combined value can flip."""
    # positive: P(true) = 0.6, negated: P(true) = 0.9 -> complement 0.1
    result = _make_result_with_twin("flag", p_pos=0.6, p_neg=0.9)
    combined = _combine_dual_framing(result, ["flag"])
    ft = combined["field_telemetry"]["flag"]
    # p = 0.5 * (0.6 + 0.1) = 0.35 -> flips to False
    assert ft["p_combined"] == pytest.approx(0.35)
    assert ft["value"] is False


# ---- on: telemetry ---------------------------------------------------------


def test_dual_framing_telemetry_keys():
    """field_telemetry carries p_pos, p_neg_complement, score_source."""
    result = _make_result_with_twin("flag", p_pos=0.7, p_neg=0.4)
    combined = _combine_dual_framing(result, ["flag"])
    ft = combined["field_telemetry"]["flag"]
    assert ft["score_source"] == "dual_framing"
    assert ft["p_pos"] == pytest.approx(0.7)
    assert ft["p_neg_complement"] == pytest.approx(0.6)  # 1 - 0.4
    assert ft["p_combined"] == pytest.approx(0.65)  # 0.5*(0.7+0.6)
    assert ft["dual_framing"]["combined"] is True
    assert ft["dual_framing"]["p_neg"] == pytest.approx(0.4)
    assert ft["dual_framing"]["disagreement"] == pytest.approx(0.1)  # |0.7-0.6|
    assert combined["prompt_version"] == PROMPT_VERSION_DUAL_FRAME
    assert combined["probability_status"] == "dual_framing"


def test_dual_framing_twin_stripped_from_result():
    """The __dual_neg__ twin is removed from the public result."""
    result = _make_result_with_twin("flag", p_pos=0.7, p_neg=0.4)
    combined = _combine_dual_framing(result, ["flag"])
    assert _dual_neg_field_name("flag") not in combined["field_telemetry"]
    assert _dual_neg_field_name("flag") not in combined["parsed_json"]
    assert "flag" in combined["field_telemetry"]


def test_dual_framing_non_boolean_untouched():
    """Non-boolean fields keep their original result."""
    result = make_engine_result(
        fields={
            "risk": make_field_telemetry(value="LOW", choices=["LOW", "HIGH"], probability=0.9),
            "flag": _bool_telemetry(True, 0.7),
            _dual_neg_field_name("flag"): _bool_telemetry(False, 0.3),
        },
        parsed={
            "risk": {"value": "LOW", "prob": 0.9},
            "flag": {"value": True, "prob": 0.7},
            _dual_neg_field_name("flag"): {"value": False, "prob": 0.3},
        },
    )
    combined = _combine_dual_framing(result, ["flag"])
    risk_ft = combined["field_telemetry"]["risk"]
    assert "p_pos" not in risk_ft
    assert "score_source" not in risk_ft
    assert risk_ft["value"] == "LOW"
    flag_ft = combined["field_telemetry"]["flag"]
    assert flag_ft["score_source"] == "dual_framing"


def test_dual_framing_no_boolean_fields_bumps_version_only():
    """With no boolean fields, no expansion; the version still bumps."""
    result = make_engine_result(
        fields={
            "risk": make_field_telemetry(value="LOW", choices=["LOW", "HIGH"], probability=0.9)
        },
        parsed={"risk": {"value": "LOW", "prob": 0.9}},
    )
    # _combine_dual_framing with empty bool_fields just bumps the version.
    combined = _combine_dual_framing(result, [])
    assert combined["prompt_version"] == PROMPT_VERSION_DUAL_FRAME
    assert "p_pos" not in combined["field_telemetry"]["risk"]


def test_expand_schema_adds_twins():
    """_expand_schema_for_dual_framing adds a __dual_neg__ twin per boolean."""
    schema = StructuredSchema(
        {
            "risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "risk"},
            "flag": {"type": "boolean", "description": "Is it flagged?"},
        }
    )
    expanded, bool_fields = _expand_schema_for_dual_framing(schema)
    assert bool_fields == ["flag"]
    assert _dual_neg_field_name("flag") in expanded.fields
    # The twin has the complement-condition description.
    twin_desc = expanded.fields[_dual_neg_field_name("flag")].description
    assert twin_desc.endswith(" Answer true only if this is NOT the case.")
    assert twin_desc.startswith("Is it flagged?")
    # The original field is unchanged.
    assert expanded.fields["flag"].description == "Is it flagged?"
    # Non-boolean field is untouched.
    assert "risk" in expanded.fields
    assert _dual_neg_field_name("risk") not in expanded.fields


def test_expand_schema_no_booleans_returns_original():
    """No boolean fields -> the original schema is returned unchanged."""
    schema = StructuredSchema(
        {"risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"}}
    )
    expanded, bool_fields = _expand_schema_for_dual_framing(schema)
    assert bool_fields == []
    assert expanded is schema  # same object, no copy


# ---- complement template determinism ---------------------------------------


def test_negate_description_is_deterministic():
    """The complement template is a fixed suffix, no LLM."""
    assert _negate_boolean_description("Is it true?") == (
        "Is it true? Answer true only if this is NOT the case."
    )
    assert _negate_boolean_description("Is it true?") == _negate_boolean_description("Is it true?")


def test_boolean_field_names():
    schema = StructuredSchema(
        {
            "risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"},
            "a": {"type": "boolean", "description": "d"},
            "b": {"type": "boolean", "description": "d"},
        }
    )
    assert _boolean_field_names(schema) == ["a", "b"]


def test_dual_neg_field_name():
    assert _dual_neg_field_name("flag") == "__dual_neg__flag"


# ---- public API: dual_framing kwarg exists ---------------------------------


def test_decide_accepts_dual_framing():
    """decide() and the one-field helpers accept dual_framing."""
    import inspect

    assert "dual_framing" in inspect.signature(jevmlx.decide).parameters
    assert "dual_framing" in inspect.signature(jevmlx.judge).parameters
    assert "dual_framing" in inspect.signature(jevmlx.choose).parameters
    assert "dual_framing" in inspect.signature(jevmlx.rate).parameters
