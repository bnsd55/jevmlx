import enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import jevmlx
from jevmlx.api import schema_from_model
from jevmlx.cli import load_preset
from jevmlx.schema import StructuredSchema


class Severity(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Case(BaseModel):
    is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk tier")
    severity: Severity = Field(description="Case severity")
    notes: bool  # no description -> field name with underscores replaced


def test_schema_from_model_exact_dict():
    assert schema_from_model(Case) == {
        "is_fraudulent": {
            "type": "boolean",
            "description": "Whether the transaction is fraudulent",
        },
        "risk_tier": {
            "type": "enum",
            "choices": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            "description": "Risk tier",
            "choice_descriptions": {},
        },
        "severity": {
            "type": "enum",
            "choices": ["LOW", "MEDIUM", "HIGH"],
            "description": "Case severity",
            "choice_descriptions": {},
        },
        "notes": {"type": "boolean", "description": "notes"},
    }


def test_unsupported_type_raises():
    class Bad(BaseModel):
        count: int = Field(description="how many")

    with pytest.raises(TypeError, match=r"'count'"):
        schema_from_model(Bad)


def test_decide_many_uses_one_engine_and_one_schema(monkeypatch):
    class TwoField(BaseModel):
        is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
        risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Risk tier")

    load_calls = []
    run_calls = []

    def fake_load_engine(model_id):
        load_calls.append(model_id)
        return ("engine", "tokenizer")

    def fake_run_parallel(
        engine_model,
        tokenizer,
        context,
        schema,
        *,
        temperature=1.0,
        scoring="slots",
        multi_threshold=0.5,
        prior_correction=False,
    ):
        run_calls.append((engine_model, tokenizer, context, schema, temperature))
        return {
            "parsed_json": {
                "is_fraudulent": {"value": True},
                "risk_tier": {"value": "HIGH"},
            },
            "field_telemetry": {
                "is_fraudulent": {"value": True, "probability": 0.9},
                "risk_tier": {"value": "HIGH", "probability": 0.8},
            },
            "confidence_model": "slots",
            "elapsed_ms": 5.0,
        }

    monkeypatch.setattr("jevmlx.api.load_engine", fake_load_engine)
    monkeypatch.setattr("jevmlx.api.run_parallel_generation", fake_run_parallel)

    decisions = jevmlx.decide_many(
        TwoField,
        ["context one", "context two", "context three"],
        model="fake/model",
        temperature=0.7,
    )

    assert len(decisions) == 3
    for d in decisions:
        assert d.value == TwoField(is_fraudulent=True, risk_tier="HIGH")
        assert set(d.fields) == {"is_fraudulent", "risk_tier"}
        assert d.fields["is_fraudulent"].probability == 0.9
        assert d.fields["risk_tier"].probability == 0.8
        assert d.latency_ms == 5.0

    assert load_calls == ["fake/model"]  # engine loaded exactly once
    assert [c[2] for c in run_calls] == ["context one", "context two", "context three"]
    assert all(c[0] == "engine" and c[1] == "tokenizer" for c in run_calls)
    # One shared StructuredSchema object across every context.
    assert len({id(c[3]) for c in run_calls}) == 1
    assert isinstance(run_calls[0][3], StructuredSchema)
    assert all(c[4] == 0.7 for c in run_calls)


def test_decide_many_input_validation(monkeypatch):
    """Y7: str/bytes contexts and non-str items raise TypeError before any load."""

    class TwoField(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] = Field(description="Risk tier")

    load_calls: list[str] = []
    monkeypatch.setattr("jevmlx.api.load_engine", lambda model_id: load_calls.append(model_id))

    with pytest.raises(TypeError, match="sequence of str"):
        jevmlx.decide_many(TwoField, "one lone context")
    with pytest.raises(TypeError, match="sequence of str"):
        jevmlx.decide_many(TwoField, b"bytes contexts")
    with pytest.raises(TypeError, match=r"contexts\[1\] must be str"):
        jevmlx.decide_many(TwoField, ["fine", 123])
    assert load_calls == []  # validation happens before the engine loads


def test_decide_many_empty_contexts_returns_empty_without_loading(monkeypatch):
    """Y7: an empty contexts list short-circuits to [] with no engine load."""

    class TwoField(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] = Field(description="Risk tier")

    def fail_load(model_id):
        raise AssertionError("engine must not be loaded for empty contexts")

    monkeypatch.setattr("jevmlx.api.load_engine", fail_load)
    assert jevmlx.decide_many(TwoField, []) == []


def test_schema_from_model_rejects_non_string_literal_values():
    """Y8: Literal[1, 2] must raise, not be silently str()-coerced."""

    class Bad(BaseModel):
        level: Literal[1, 2] = Field(description="level")

    with pytest.raises(TypeError, match="'level'"):
        schema_from_model(Bad)


def test_schema_from_model_rejects_non_string_enum_values():
    """Y8: IntEnum members are not str; must raise naming the field."""

    class Priority(enum.IntEnum):
        LOW = 1
        HIGH = 2

    class Bad(BaseModel):
        priority: Priority = Field(description="priority")

    with pytest.raises(TypeError, match="'priority'"):
        schema_from_model(Bad)


def test_schema_from_model_accepts_strenum():
    """Y8: enum.StrEnum members are str and must work as enum choices."""

    class Color(enum.StrEnum):
        RED = "red"
        GREEN = "green"

    class Ok(BaseModel):
        color: Color = Field(description="color")

    assert schema_from_model(Ok) == {
        "color": {
            "type": "enum",
            "choices": ["red", "green"],
            "description": "color",
            "choice_descriptions": {},
        }
    }


def test_choice_values_rejects_duplicate_values_directly():
    """Y8: the strictness helper rejects duplicates it is handed.

    Tested directly because duplicates cannot reach it through Pydantic:
    Literal deduplicates at annotation level and Python enums alias members
    with equal values, so both collapse before _choice_values runs.
    """
    from jevmlx.api import _choice_values

    assert _choice_values("x", ["a", "b"]) == ["a", "b"]
    with pytest.raises(TypeError, match="duplicate"):
        _choice_values("x", ["a", "b", "a"])
    with pytest.raises(TypeError, match="non-string"):
        _choice_values("x", ["a", 1])


def test_clear_engine_cache_is_public_and_idempotent():
    """Y9: clear_engine_cache exists on the package and is safe to call twice."""
    assert callable(jevmlx.clear_engine_cache)
    jevmlx.clear_engine_cache()
    jevmlx.clear_engine_cache()  # must not raise with nothing cached


def test_load_engine_cache_is_single_slot():
    """Y9: one model in unified memory at a time — maxsize=1."""
    assert jevmlx.load_engine.cache_info().maxsize == 1


def test_decide_end_to_end():
    class TwoField(BaseModel):
        is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
        risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk tier")

    fraud_preset = load_preset("fintech_fraud")
    d = jevmlx.decide(
        TwoField,
        fraud_preset["context"],
        model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    )
    assert isinstance(d.value, TwoField)
    assert set(d.fields) == {"is_fraudulent", "risk_tier"}
    assert all(0.0 <= fr.probability <= 1.0 for fr in d.fields.values())
    assert d.latency_ms > 0


# --- V3: per-choice glosses -------------------------------------------------


class SeverityGlossed(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    descriptions = {  # becomes an enum MEMBER whose value is this dict
        "LOW": "routine case",
        "MEDIUM": "needs review today",
        "HIGH": "escalate immediately",
    }


def test_choice_glosses_from_json_schema_extra():
    """Literal glosses come from Field(json_schema_extra={'choice_descriptions': ...})."""

    class WithGlosses(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] = Field(
            description="Risk tier",
            json_schema_extra={"choice_descriptions": {"LOW": "nothing to do", "HIGH": "act now"}},
        )

    schema = schema_from_model(WithGlosses)
    assert schema["risk_tier"]["choice_descriptions"] == {
        "LOW": "nothing to do",
        "HIGH": "act now",
    }
    # Flows through StructuredSchema into FieldDefinition.
    fs = StructuredSchema(schema).fields["risk_tier"]
    assert fs.choice_descriptions == {"LOW": "nothing to do", "HIGH": "act now"}
    assert fs.to_dict()["choice_descriptions"] == fs.choice_descriptions


def test_choice_glosses_from_enum_class_descriptions():
    class WithEnum(BaseModel):
        severity: SeverityGlossed = Field(description="Severity")

    schema = schema_from_model(WithEnum)
    assert schema["severity"]["choice_descriptions"] == {
        "LOW": "routine case",
        "MEDIUM": "needs review today",
        "HIGH": "escalate immediately",
    }


def test_choice_glosses_absent_means_empty_dict():
    class Plain(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] = Field(description="Risk tier")

    schema = schema_from_model(Plain)
    assert schema["risk_tier"]["choice_descriptions"] == {}


def test_choice_glosses_malformed_raise():
    class Bad(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] = Field(
            description="Risk tier",
            json_schema_extra={"choice_descriptions": {"LOW": 3}},
        )

    with pytest.raises(TypeError, match="choice_descriptions"):
        schema_from_model(Bad)


def test_choice_glosses_unknown_choice_rejected_by_schema():
    with pytest.raises(ValueError, match="choice_descriptions keys not in choices"):
        StructuredSchema(
            {
                "f": {
                    "type": "enum",
                    "description": "d",
                    "choices": ["a", "b"],
                    "choice_descriptions": {"nope": "gloss"},
                }
            }
        )


# --- W2-D: allow_none_of_above (explicit opt-out, was allow_unknown) --------


def test_allow_none_of_above_requires_optional_enum():
    class Strict(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] = Field(description="Risk tier")

    with pytest.raises(TypeError, match=r"'risk_tier'.*Optional"):
        jevmlx.decide(Strict, "ctx", model="fake/model", allow_none_of_above=True)


def test_allow_none_of_above_appends_choice_and_maps_to_none(monkeypatch):
    class Maybe(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] | None = Field(description="Risk tier")

    captured = {}

    def fake_run_parallel(engine_model, tokenizer, context, schema, **kwargs):
        captured["choices"] = schema.fields["risk_tier"].choices
        captured["descriptions"] = schema.fields["risk_tier"].choice_descriptions
        return {
            "parsed_json": {"risk_tier": {"value": "NONE_OF_ABOVE"}},
            "field_telemetry": {
                "risk_tier": {
                    "value": "NONE_OF_ABOVE",
                    "probability": 0.4,
                    "log_scores": {"LOW": -2.1, "HIGH": -3.0, "NONE_OF_ABOVE": -1.1},
                    "top_choices": [{"choice": "NONE_OF_ABOVE", "probability": 0.4}],
                }
            },
            "confidence_model": "slots",
            "elapsed_ms": 5.0,
        }

    monkeypatch.setattr("jevmlx.api.load_engine", lambda model_id: ("engine", "tokenizer"))
    monkeypatch.setattr("jevmlx.api.run_parallel_generation", fake_run_parallel)

    d = jevmlx.decide(Maybe, "ctx", model="fake/model", allow_none_of_above=True)
    # NONE_OF_ABOVE was appended to the engine's choices with the opt-out gloss...
    assert captured["choices"] == ["LOW", "HIGH", "NONE_OF_ABOVE"]
    assert captured["descriptions"]["NONE_OF_ABOVE"] == "none of the options apply"
    # ...and mapped to None in the validated model, with the explicit reason.
    assert d.value.risk_tier is None
    assert d.fields["risk_tier"].value == "NONE_OF_ABOVE"
    assert d.fields["risk_tier"].reason == "none_of_above"


def test_allow_unknown_gone_no_alias(monkeypatch):
    """The old allow_unknown kwarg is gone — not deprecated, not aliased.

    Also: a model answer of the old synthetic 'UNKNOWN' string is just a
    regular choice now; with allow_none_of_above it's still a decided value
    (no None mapping) unless the field declares it.
    """

    class Maybe(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] | None = Field(description="Risk tier")

    with pytest.raises(TypeError, match="unexpected keyword argument 'allow_unknown'"):
        jevmlx.decide(Maybe, "ctx", model="fake/model", allow_unknown=True)


def test_allow_none_of_above_off_leaves_schema_untouched(monkeypatch):
    class Maybe(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] | None = Field(description="Risk tier")

    captured = {}

    def fake_run_parallel(engine_model, tokenizer, context, schema, **kwargs):
        captured["choices"] = schema.fields["risk_tier"].choices
        return {
            "parsed_json": {"risk_tier": {"value": "LOW"}},
            "field_telemetry": {"risk_tier": {"value": "LOW", "probability": 0.9}},
            "confidence_model": "slots",
            "elapsed_ms": 5.0,
        }

    monkeypatch.setattr("jevmlx.api.load_engine", lambda model_id: ("engine", "tokenizer"))
    monkeypatch.setattr("jevmlx.api.run_parallel_generation", fake_run_parallel)

    jevmlx.decide(Maybe, "ctx", model="fake/model")
    assert captured["choices"] == ["LOW", "HIGH"]


def test_allow_none_of_above_rejects_existing_choice():
    class Collide(BaseModel):
        risk_tier: Literal["LOW", "NONE_OF_ABOVE"] | None = Field(description="Risk tier")

    with pytest.raises(TypeError, match="already exists"):
        jevmlx.decide(Collide, "ctx", model="fake/model", allow_none_of_above=True)


# --- W2-D: abstention (confidence gate, not a choice) ------------------------


def _abstain_result(margin: float):
    """Engine result shaped so risk_tier's probability_margin == margin."""
    p1, p2 = 0.5 + margin / 2, 0.5 - margin / 2
    return {
        "parsed_json": {"risk_tier": {"value": "HIGH"}, "tags": {"value": ["a"]}},
        "field_telemetry": {
            "risk_tier": {
                "value": "HIGH",
                "probability": p1,
                "log_scores": {"LOW": -1.0, "HIGH": -0.5},
                "top_choices": [
                    {"choice": "HIGH", "probability": p1},
                    {"choice": "LOW", "probability": p2},
                ],
            },
            "tags": {
                "value": ["a"],
                "type": "multi",
                "probability": None,
                "margin": margin,
                "per_option": {"a": 0.5 + margin / 2, "b": 0.5 - margin / 2},
                "top_choices": [],
                "rows": 2,
            },
        },
        "confidence_model": "slots",
        "elapsed_ms": 5.0,
    }


class AbstainModel(BaseModel):
    risk_tier: Literal["LOW", "HIGH"] | None = Field(description="Risk tier")
    tags: list[Literal["a", "b"]] | None = Field(default_factory=list)


def _abstain_monkeypatch(monkeypatch, margin: float):
    monkeypatch.setattr("jevmlx.api.load_engine", lambda model_id: ("engine", "tokenizer"))
    monkeypatch.setattr(
        "jevmlx.api.run_parallel_generation",
        lambda *a, **k: _abstain_result(margin),
    )


def test_abstain_scalar_field_below_margin(monkeypatch):
    # margin 0.04 < cut 0.1 -> abstain: model gets None, FieldResult keeps
    # the engine's raw value, reason='abstain'.
    _abstain_monkeypatch(monkeypatch, 0.04)
    d = jevmlx.decide(AbstainModel, "ctx", model="fake/model", abstain_below_margin=0.1)
    assert d.value.risk_tier is None
    fr = d.fields["risk_tier"]
    assert fr.reason == "abstain"
    assert fr.value == "HIGH"  # provenance kept


def test_abstain_scalar_field_above_margin(monkeypatch):
    _abstain_monkeypatch(monkeypatch, 0.6)
    d = jevmlx.decide(AbstainModel, "ctx", model="fake/model", abstain_below_margin=0.1)
    assert d.value.risk_tier == "HIGH"
    assert d.fields["risk_tier"].reason is None
    assert d.fields["risk_tier"].reason is None


def test_abstain_no_threshold_keeps_values(monkeypatch):
    _abstain_monkeypatch(monkeypatch, 0.01)
    d = jevmlx.decide(AbstainModel, "ctx", model="fake/model")
    assert d.value.risk_tier == "HIGH"
    assert d.fields["risk_tier"].reason is None


def test_abstain_multi_field_uses_threshold_distance(monkeypatch):
    # Multi field: probability_margin is None; threshold_distance drives it.
    _abstain_monkeypatch(monkeypatch, 0.02)
    d = jevmlx.decide(AbstainModel, "ctx", model="fake/model", abstain_below_margin=0.1)
    assert d.value.tags is None
    fr = d.fields["tags"]
    assert fr.reason == "abstain"
    assert fr.value == ["a"]  # provenance kept


def test_abstain_and_none_of_above_are_independent(monkeypatch):
    # NONE_OF_ABOVE is a decided answer (reason='none_of_above', not abstained);
    # abstention is a confidence gate. A NONE_OF_ABOVE pick with a wide margin
    # is not abstained even when the threshold is set.

    def fake_run_parallel(*a, **k):
        result = _abstain_result(0.6)  # margin 0.6 >> 0.1: no abstention
        result["parsed_json"] = {"risk_tier": {"value": "NONE_OF_ABOVE"}, "tags": {"value": ["a"]}}
        telemetry = result["field_telemetry"]["risk_tier"]
        telemetry.update(
            {
                "value": "NONE_OF_ABOVE",
                "probability": 0.8,
                "log_scores": {"LOW": -2.0, "HIGH": -3.0, "NONE_OF_ABOVE": -0.2},
            }
        )
        telemetry["top_choices"] = [
            {"choice": "NONE_OF_ABOVE", "probability": 0.8},
            {"choice": "LOW", "probability": 0.1},
            {"choice": "HIGH", "probability": 0.1},
        ]
        return result

    monkeypatch.setattr("jevmlx.api.load_engine", lambda model_id: ("engine", "tokenizer"))
    monkeypatch.setattr("jevmlx.api.run_parallel_generation", fake_run_parallel)

    d = jevmlx.decide(
        AbstainModel,
        "ctx",
        model="fake/model",
        allow_none_of_above=True,
        abstain_below_margin=0.1,
    )
    assert d.value.risk_tier is None  # NONE_OF_ABOVE maps to None
    fr = d.fields["risk_tier"]
    assert fr.reason == "none_of_above"
    assert fr.reason == "none_of_above"  # not a confidence abstention
    assert fr.value == "NONE_OF_ABOVE"


def test_abstain_below_margin_validation():
    with pytest.raises(TypeError, match="abstain_below_margin must be in"):
        jevmlx.decide(AbstainModel, "ctx", model="fake/model", abstain_below_margin=1.0)
    with pytest.raises(TypeError, match="abstain_below_margin must be in"):
        jevmlx.decide(AbstainModel, "ctx", model="fake/model", abstain_below_margin=-0.5)
    with pytest.raises(TypeError, match="abstain_below_margin must be in"):
        jevmlx.decide_many(AbstainModel, ["a"], model="fake/model", abstain_below_margin=1.5)


def test_decide_many_abstain(monkeypatch):
    _abstain_monkeypatch(monkeypatch, 0.04)
    results = jevmlx.decide_many(
        AbstainModel, ["ctx1", "ctx2"], model="fake/model", abstain_below_margin=0.1
    )
    assert all(r.value.risk_tier is None for r in results)
    assert all(r.fields["risk_tier"].reason == "abstain" for r in results)


# --- V3: Decision.fields / FieldResult --------------------------------------


def _fake_result(confidence_model="slots"):
    return {
        "parsed_json": {"risk_tier": {"value": "HIGH"}, "tags": {"value": ["a"]}},
        "field_telemetry": {
            "risk_tier": {
                "value": "HIGH",
                "probability": 0.7,
                "log_scores": {"LOW": -0.5, "HIGH": -0.1, "CRITICAL": -2.0},
                "top_choices": [
                    {"choice": "HIGH", "probability": 0.7},
                    {"choice": "LOW", "probability": 0.2},
                    {"choice": "CRITICAL", "probability": 0.1},
                ],
            },
            "tags": {
                "value": ["a"],
                "probability": None,
                "per_option": {"a": 0.8, "b": 0.1},
                "margin": 0.3,
            },
        },
        "confidence_model": confidence_model,
        "elapsed_ms": 5.0,
    }


def test_field_result_built_from_fake_engine_result(monkeypatch):
    import jevmlx.api as api

    monkeypatch.setattr(api, "run_parallel_generation", lambda *a, **k: _fake_result())
    monkeypatch.setattr(api, "load_engine", lambda model_id: ("engine", "tokenizer"))

    class TwoField(BaseModel):
        risk_tier: Literal["LOW", "HIGH", "CRITICAL"] = Field(description="Risk tier")
        tags: list[Literal["a", "b"]] = Field(description="Tags")

    d = jevmlx.decide(TwoField, "ctx", model="fake/model")
    fr = d.fields["risk_tier"]
    assert isinstance(fr, api.FieldResult)
    assert fr.value == "HIGH"
    assert fr.score == -0.1  # log P of the winner
    assert abs(fr.log_score_margin - (-0.1 - -0.5)) < 1e-9  # top1-top2 at T=1
    assert abs(fr.probability_margin - (0.7 - 0.2)) < 1e-9  # post-temperature top1-top2
    assert fr.threshold_distance is None  # scalar fields carry no threshold distance
    assert fr.probability == 0.7
    assert fr.calibrated is False
    assert fr.model == "slots"
    assert fr.alternatives == (("HIGH", 0.7), ("LOW", 0.2), ("CRITICAL", 0.1))
    # multi: no log_scores -> score 0.0, threshold_distance from telemetry,
    # alternatives from per_option
    multi = d.fields["tags"]
    assert multi.value == ["a"]
    assert multi.log_score_margin is None
    assert multi.probability_margin is None
    assert multi.threshold_distance == 0.3  # min(|0.8-0.5|, |0.1-0.5|) from telemetry
    assert multi.alternatives == (("a", 0.8), ("b", 0.1))
    assert isinstance(d.value, TwoField)


def test_field_result_model_matches_scoring_mode(monkeypatch):
    import jevmlx.api as api

    monkeypatch.setattr(api, "run_parallel_generation", lambda *a, **k: _fake_result("slots"))
    monkeypatch.setattr(api, "load_engine", lambda model_id: ("engine", "tokenizer"))

    class OneField(BaseModel):
        risk_tier: Literal["LOW", "HIGH", "CRITICAL"] = Field(description="Risk tier")

    d = jevmlx.decide(OneField, "ctx", model="fake/model")
    assert d.fields["risk_tier"].model == "slots"


def test_decision_has_no_confidence_attribute():
    """HARD RULE: .confidence is gone, replaced by .fields."""
    import dataclasses

    from jevmlx.api import Decision

    assert not any(f.name == "confidence" for f in dataclasses.fields(Decision))
