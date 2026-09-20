"""W6-B5: choose / judge / rate one-field convenience helpers + CLI verbs.

The wrappers synthesize a one-field schema and return the single
FieldResult (not a full Decision). No engine change — they reuse the
exact run_parallel_generation path decide() takes. Tests use the fake
engine path from conftest (monkeypatched run_parallel_generation +
load_engine).
"""

from __future__ import annotations

import json

import pytest
from conftest import (
    make_engine_result,
    make_field_telemetry,
)

import jevmlx
from jevmlx import api
from jevmlx.cli import main as cli_main

# ---- shared fake-engine plumbing -------------------------------------------


def _patch_engine(monkeypatch, fake_run_parallel):
    # Patch the module object OUR `api` binding points to (not
    # sys.modules["jevmlx.api"]) — test_check_results::test_imports_without_mlx
    # evicts jevmlx.api from sys.modules, and a bare "jevmlx.api.X" string
    # would patch the re-imported module our functions don't live in.
    monkeypatch.setattr(api, "load_engine", lambda model_id: ("engine", "tokenizer"))
    monkeypatch.setattr(api, "run_parallel_generation", fake_run_parallel)
    # The CLI verbs do `from jevmlx import api` at CALL time, which resolves
    # via sys.modules — after the eviction test that may be a DIFFERENT
    # module object than our `api` binding. Patch sys.modules' entry too so
    # the CLI picks up the same fakes.
    import sys as _sys

    cached = _sys.modules.get("jevmlx.api")
    if cached is not api:
        monkeypatch.setattr(cached, "load_engine", lambda model_id: ("engine", "tokenizer"))
        monkeypatch.setattr(cached, "run_parallel_generation", fake_run_parallel)


# ---- choose ----------------------------------------------------------------


def test_choose_builds_enum_schema_and_returns_field_result(monkeypatch):
    captured = {}

    def fake(engine, context, schema, **kwargs):
        captured["context"] = context
        captured["schema"] = schema
        captured["field_name"] = list(schema.fields)
        field = schema.fields["choice"]
        captured["choices"] = field.choices
        captured["descriptions"] = field.choice_descriptions
        captured["ordered"] = field.ordered
        return make_engine_result(
            fields={
                "choice": make_field_telemetry(
                    value="B",
                    choices=["A", "B", "C"],
                    probability=0.7,
                    log_scores={"A": -2.0, "B": -0.7, "C": -3.0},
                    top_choices=[
                        {"choice": "B", "probability": 0.7},
                        {"choice": "A", "probability": 0.2},
                        {"choice": "C", "probability": 0.1},
                    ],
                )
            },
            parsed={"choice": {"value": "B", "prob": 0.7}},
        )

    _patch_engine(monkeypatch, fake)
    field = api.choose("a context", {"A": "desc A", "B": "desc B", "C": "desc C"})

    # The single field is named "choice" and is an UNORDERED enum.
    assert captured["field_name"] == ["choice"]
    assert captured["choices"] == ("A", "B", "C")
    assert captured["descriptions"] == {"A": "desc A", "B": "desc B", "C": "desc C"}
    assert captured["ordered"] is False
    # context flows through unchanged.
    assert captured["context"] == "a context"
    # The FieldResult carries the winner + probability + margin.
    assert field.value == "B"
    assert field.probability == pytest.approx(0.7)
    assert field.reason is None


def test_choose_list_options_uses_name_as_description(monkeypatch):
    captured = {}

    def fake(engine, context, schema, **kwargs):
        field = schema.fields["choice"]
        captured["descriptions"] = field.choice_descriptions
        captured["choices"] = field.choices
        return make_engine_result(
            fields={"choice": make_field_telemetry(value="x", choices=["x", "y"], probability=0.6)},
            parsed={"choice": {"value": "x", "prob": 0.6}},
        )

    _patch_engine(monkeypatch, fake)
    api.choose("ctx", ["x", "y"])
    assert captured["choices"] == ("x", "y")
    assert captured["descriptions"] == {"x": "x", "y": "y"}


def test_choose_instructions_become_field_description(monkeypatch):
    captured = {}

    def fake(engine, context, schema, **kwargs):
        captured["description"] = schema.fields["choice"].description
        return make_engine_result(
            fields={"choice": make_field_telemetry(value="a", choices=["a", "b"], probability=0.6)},
            parsed={"choice": {"value": "a", "prob": 0.6}},
        )

    _patch_engine(monkeypatch, fake)
    api.choose("ctx", {"a": "x", "b": "y"}, instructions="Pick the best fit.")
    assert captured["description"] == "Pick the best fit."


def test_choose_reuses_decide_kwargs(monkeypatch):
    """temperature/scoring/prior_correction pass through to the engine."""
    captured = {}

    def fake(engine, context, schema, **kwargs):
        captured["kwargs"] = kwargs
        return make_engine_result(
            fields={"choice": make_field_telemetry(value="a", choices=["a", "b"], probability=0.6)},
            parsed={"choice": {"value": "a", "prob": 0.6}},
        )

    _patch_engine(monkeypatch, fake)
    api.choose(
        "ctx",
        {"a": "x", "b": "y"},
        model="m",
        temperature=0.5,
        scoring="labels",
        prior_correction=True,
    )
    assert captured["kwargs"]["temperature"] == 0.5
    assert captured["kwargs"]["scoring"] == "labels"
    assert captured["kwargs"]["prior_correction"] is True


def test_choose_validation_too_few_options():
    with pytest.raises(ValueError, match="at least 2"):
        api.choose("ctx", {"a": "x"})


def test_choose_validation_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        api.choose("ctx", ["a", "a"])


def test_choose_validation_bad_type():
    with pytest.raises(TypeError, match="dict.*list"):
        api.choose("ctx", "not a dict or list")  # type: ignore[arg-type]


# ---- judge -----------------------------------------------------------------


def test_judge_builds_boolean_field(monkeypatch):
    captured = {}

    def fake(engine, context, schema, **kwargs):
        field = schema.fields["judgment"]
        captured["field_name"] = list(schema.fields)
        captured["type"] = field.field_type
        captured["description"] = field.description
        return make_engine_result(
            fields={
                "judgment": make_field_telemetry(
                    value=True, type_="boolean", choices=["false", "true"], probability=0.8
                )
            },
            parsed={"judgment": {"value": True, "prob": 0.8}},
        )

    _patch_engine(monkeypatch, fake)
    field = api.judge("a passage", "Is the claim supported?")

    assert captured["field_name"] == ["judgment"]
    assert captured["type"] == "boolean"
    assert captured["description"] == "Is the claim supported?"
    # Returns the full FieldResult (not a bare float): probability of True.
    assert field.value is True
    assert field.probability == pytest.approx(0.8)


def test_judge_empty_question_raises():
    with pytest.raises(ValueError, match="question"):
        api.judge("ctx", "   ")


# ---- rate ------------------------------------------------------------------


def test_rate_builds_ordered_enum_with_ordinal(monkeypatch):
    captured = {}

    def fake(engine, context, schema, **kwargs):
        field = schema.fields["rating"]
        captured["field_name"] = list(schema.fields)
        captured["choices"] = field.choices
        captured["ordered"] = field.ordered
        return make_engine_result(
            fields={
                "rating": make_field_telemetry(
                    value="medium",
                    choices=["low", "medium", "high"],
                    probability=0.6,
                    log_scores={"low": -3.0, "medium": -0.6, "high": -2.0},
                    top_choices=[
                        {"choice": "medium", "probability": 0.6},
                        {"choice": "high", "probability": 0.25},
                        {"choice": "low", "probability": 0.15},
                    ],
                    ordinal={
                        "argmax_level": 1,
                        "expected_index": 1.35,
                        "variance": 0.4275,
                        "expected_score_normalized": 0.675,
                    },
                )
            },
            parsed={"rating": {"value": "medium", "prob": 0.6}},
        )

    _patch_engine(monkeypatch, fake)
    field = api.rate("a review", {"low": "poor", "medium": "ok", "high": "great"})

    # The field is an ORDERED enum — the declaration order is the scale.
    assert captured["field_name"] == ["rating"]
    assert captured["choices"] == ("low", "medium", "high")
    assert captured["ordered"] is True
    # The ordinal record is derived (argmax_level = index of winner).
    assert field.ordinal is not None
    assert field.ordinal.argmax_level == 1  # "medium" is index 1
    assert field.value == "medium"


def test_rate_list_levels_preserves_order(monkeypatch):
    captured = {}

    def fake(engine, context, schema, **kwargs):
        captured["choices"] = schema.fields["rating"].choices
        captured["ordered"] = schema.fields["rating"].ordered
        return make_engine_result(
            fields={
                "rating": make_field_telemetry(value="b", choices=["a", "b", "c"], probability=0.6)
            },
            parsed={"rating": {"value": "b", "prob": 0.6}},
        )

    _patch_engine(monkeypatch, fake)
    api.rate("ctx", ["a", "b", "c"])
    assert captured["choices"] == ("a", "b", "c")
    assert captured["ordered"] is True


def test_rate_validation_too_few_levels():
    with pytest.raises(ValueError, match="at least 2"):
        api.rate("ctx", {"low": "x"})


# ---- public API exports ----------------------------------------------------


def test_helpers_are_in_all():
    for name in ("choose", "judge", "rate"):
        assert name in jevmlx.__all__
        assert getattr(jevmlx, name) is not None


# ---- CLI verbs -------------------------------------------------------------


def _write_context(tmp_path, text="a context"):
    ctx = tmp_path / "ctx.txt"
    ctx.write_text(text, encoding="utf-8")
    return ctx


def test_cli_choose_parses_and_prints(monkeypatch, tmp_path, capsys):
    def fake(engine, context, schema, **kwargs):
        return make_engine_result(
            fields={
                "choice": make_field_telemetry(
                    value="B",
                    choices=["A", "B"],
                    probability=0.7,
                    top_choices=[
                        {"choice": "B", "probability": 0.7},
                        {"choice": "A", "probability": 0.3},
                    ],
                )
            },
            parsed={"choice": {"value": "B", "prob": 0.7}},
        )

    _patch_engine(monkeypatch, fake)
    ctx = _write_context(tmp_path)
    cli_main(
        [
            "choose",
            "--context",
            str(ctx),
            "--option",
            "A=desc A",
            "--option",
            "B=desc B",
            "--model",
            "fake/m",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == "B"
    assert out["prob"] == pytest.approx(0.7)


def test_cli_judge_parses_and_prints(monkeypatch, tmp_path, capsys):
    def fake(engine, context, schema, **kwargs):
        return make_engine_result(
            fields={
                "judgment": make_field_telemetry(
                    value=True, type_="boolean", choices=["false", "true"], probability=0.9
                )
            },
            parsed={"judgment": {"value": True, "prob": 0.9}},
        )

    _patch_engine(monkeypatch, fake)
    ctx = _write_context(tmp_path)
    cli_main(["judge", "--context", str(ctx), "--question", "Is it true?", "--model", "fake/m"])
    out = json.loads(capsys.readouterr().out)
    assert out["value"] is True
    assert out["prob"] == pytest.approx(0.9)


def test_cli_rate_parses_and_prints(monkeypatch, tmp_path, capsys):
    def fake(engine, context, schema, **kwargs):
        return make_engine_result(
            fields={
                "rating": make_field_telemetry(
                    value="medium",
                    choices=["low", "medium", "high"],
                    probability=0.6,
                    top_choices=[
                        {"choice": "medium", "probability": 0.6},
                        {"choice": "high", "probability": 0.25},
                        {"choice": "low", "probability": 0.15},
                    ],
                    ordinal={
                        "argmax_level": 1,
                        "expected_index": 1.35,
                        "variance": 0.4275,
                        "expected_score_normalized": 0.675,
                    },
                )
            },
            parsed={"rating": {"value": "medium", "prob": 0.6}},
        )

    _patch_engine(monkeypatch, fake)
    ctx = _write_context(tmp_path)
    cli_main(
        [
            "rate",
            "--context",
            str(ctx),
            "--level",
            "low=poor",
            "--level",
            "medium=ok",
            "--level",
            "high=great",
            "--model",
            "fake/m",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == "medium"
    assert "ordinal" in out
    assert out["ordinal"]["argmax_level"] == 1


def test_cli_choose_bare_option_no_equals(monkeypatch, tmp_path, capsys):
    """A bare --option name (no =description) maps name -> name."""

    def fake(engine, context, schema, **kwargs):
        assert schema.fields["choice"].choice_descriptions == {"A": "A", "B": "B"}
        return make_engine_result(
            fields={"choice": make_field_telemetry(value="A", choices=["A", "B"], probability=0.6)},
            parsed={"choice": {"value": "A", "prob": 0.6}},
        )

    _patch_engine(monkeypatch, fake)
    ctx = _write_context(tmp_path)
    cli_main(
        ["choose", "--context", str(ctx), "--option", "A", "--option", "B", "--model", "fake/m"]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == "A"


def test_cli_choose_stdin_context(monkeypatch, tmp_path, capsys):
    def fake(engine, context, schema, **kwargs):
        assert context == "from stdin"
        return make_engine_result(
            fields={"choice": make_field_telemetry(value="A", choices=["A", "B"], probability=0.6)},
            parsed={"choice": {"value": "A", "prob": 0.6}},
        )

    _patch_engine(monkeypatch, fake)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("from stdin"))
    cli_main(["choose", "--context", "-", "--option", "A", "--option", "B", "--model", "fake/m"])
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == "A"
