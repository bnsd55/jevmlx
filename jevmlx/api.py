"""Typed Pydantic API: decide() maps a BaseModel schema onto parallel
constrained decisions and returns validated, typed results.

    from typing import Literal
    from pydantic import BaseModel, Field
    import jevmlx

    class Fraud(BaseModel):
        is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
        risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk tier")

    d = jevmlx.decide(Fraud, context)
    d.value                  # Fraud(is_fraudulent=True, risk_tier="CRITICAL")
    d.fields["risk_tier"].probability  # constrained or slot P of the winner
    d.latency_ms
"""

from __future__ import annotations

import dataclasses
import enum
import typing
from collections.abc import Sequence

from pydantic import BaseModel

from jevmlx.engine import load_engine, run_parallel_generation
from jevmlx.schema import StructuredSchema

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

_SUPPORTED = (
    "supported field types: bool, Literal[str, ...], enum.Enum/enum.StrEnum with str values, "
    "list[Literal[...]] / set[Literal[...]] (multi)"
)

# Synthetic unknown choice appended to every enum field when allow_unknown is
# set; the engine decides it like any other choice and the API maps it to None.
UNKNOWN = "UNKNOWN"
UNKNOWN_DESCRIPTION = "insufficient evidence or none of the options"


def _choice_values(name: str, values: list) -> list[str]:
    """Validate and return a field's choice values as strict strings.

    Every value must already be a str — no ``str()`` coercion, which would
    silently turn ``Literal[1, 2]`` into a schema about the strings "1" and
    "2". Duplicates are rejected because the engine scores one row per
    distinct first token and cannot distinguish duplicate literals.
    """
    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                f"Field '{name}' has non-string choice value {value!r} "
                f"({type(value).__name__}); use str values. {_SUPPORTED}"
            )
    if len(set(values)) != len(values):
        dupes = sorted({v for v in values if values.count(v) > 1})
        raise TypeError(
            f"Field '{name}' has duplicate choice values: {', '.join(repr(v) for v in dupes)}"
        )
    return values


@dataclasses.dataclass(frozen=True)
class FieldResult:
    """Provenance for one decided field.

    Attributes:
        value: The decided value (engine-side: str for enums, bool for
            booleans, list[str] for multi).
        score: Log P of the winning choice (constrained-path log score);
            0.0 for multi fields (no field-level log score exists).
        log_score_margin: Scalar fields only: top-1 minus top-2 log score at
            T=1 (decision units: log odds). None for multi fields.
        probability_margin: Scalar fields only: top-1 minus top-2
            probability (post-temperature). None for multi fields.
        threshold_distance: Multi fields only: min |P(yes) - threshold|
            over the field's options — how close the closest yes/no decision
            sat to the selection cut. None for scalar fields.
        probability: P of the winner — constrained-path probability in
            both scoring modes. In [0, 1]; None for multi fields (no
            field-level probability is claimed).
        calibrated: True only after a fitted calibrator has been applied to
            ``probability``. The engine never calibrates; this is False in
            every decide() result until calibration runs.
        model: Which scoring model produced the probability: "slots"
            (neutral aliases through the token trie; the default) or
            "labels" (real choice text through the token trie).
        alternatives: Top 3 (choice, probability) pairs, most probable
            first. For multi fields: all per-option (option, P(yes)) pairs
            sorted by P(yes) descending.
    """

    value: object
    score: float
    log_score_margin: float | None
    probability_margin: float | None
    threshold_distance: float | None
    probability: float | None
    calibrated: bool
    model: str
    alternatives: tuple[tuple[str, float], ...]


@dataclasses.dataclass
class Decision[T: BaseModel]:
    value: T
    fields: dict[str, FieldResult]
    latency_ms: float


def _description(name: str, info) -> str:
    return info.description or name.replace("_", " ")


def _choice_descriptions(name: str, info) -> dict[str, str]:
    """Per-choice glosses from Field(json_schema_extra={"choice_descriptions": ...})."""
    extra = info.json_schema_extra
    if not isinstance(extra, dict):
        return {}
    glosses = extra.get("choice_descriptions")
    if glosses is None:
        return {}
    if not isinstance(glosses, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in glosses.items()
    ):
        raise TypeError(
            f"Field '{name}': json_schema_extra['choice_descriptions'] must be "
            "a dict mapping choice strings to gloss strings"
        )
    return dict(glosses)


def _enum_class_descriptions(ann) -> dict[str, str]:
    """Per-choice glosses from an Enum class attribute ``descriptions``.

    A plain dict class attribute on an Enum becomes an enum member whose
    value is the dict (Python enum semantics), so it must be excluded from
    the choice list (done by the str-value filter) and read back through the
    member value here. The required shape is a dict mapping member values to
    gloss strings.
    """
    descriptions = None
    for member in ann:
        if member.name == "descriptions":
            descriptions = member.value
            break
    if descriptions is None:
        return {}
    if not isinstance(descriptions, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in descriptions.items()
    ):
        raise TypeError(
            f"Enum '{ann.__name__}': class attribute 'descriptions' must be a dict "
            "mapping member values to gloss strings"
        )
    return dict(descriptions)


def _optional_inner(ann):
    """Inner type of Optional[X] (Union with exactly one non-None arg), else None."""
    if typing.get_origin(ann) is not typing.Union:
        return None
    args = [a for a in typing.get_args(ann) if a is not type(None)]
    return args[0] if len(args) == 1 else None


def schema_from_model(model_cls: type[BaseModel]) -> dict:
    """Map a Pydantic model to the engine's schema dict (bool / enum fields).

    Literal fields take per-choice glosses from
    ``Field(json_schema_extra={"choice_descriptions": {...}})``; enum classes
    take them from a class attribute ``descriptions`` (a dict mapping member
    values to glosses). Glosses are carried into the schema dict as the
    optional ``choice_descriptions`` key.

    Optional[Literal[...]] and Optional[enum] are accepted: the schema is
    identical to the non-Optional form — None is the UNKNOWN mapping, not a
    decided value. Any other Optional raises.
    """
    schema: dict = {}
    for name, info in model_cls.model_fields.items():
        ann = info.annotation
        origin = typing.get_origin(ann)
        if origin is typing.Union:
            inner = _optional_inner(ann)
            if inner is None:
                raise TypeError(
                    f"Field '{name}' has unsupported type {ann!r}; only "
                    f"Optional[<enum>] is supported. {_SUPPORTED}"
                )
            ann = inner
            origin = typing.get_origin(ann)
        if ann is bool:
            schema[name] = {"type": "boolean", "description": _description(name, info)}
        elif origin in (list, set) and typing.get_args(ann):
            (lit,) = typing.get_args(ann)
            if typing.get_origin(lit) is not typing.Literal:
                raise TypeError(f"Field '{name}' has unsupported type {ann!r}. {_SUPPORTED}")
            schema[name] = {
                "type": "multi",
                "choices": _choice_values(name, list(typing.get_args(lit))),
                "description": _description(name, info),
                "choice_descriptions": _choice_descriptions(name, info),
            }
        elif origin is typing.Literal:
            schema[name] = {
                "type": "enum",
                "choices": _choice_values(name, list(typing.get_args(ann))),
                "description": _description(name, info),
                "choice_descriptions": _choice_descriptions(name, info),
            }
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            # Collect member values, skipping the synthetic 'descriptions'
            # member (its value is the gloss dict, not a choice). Any OTHER
            # non-str member value must raise: silent str() coercion or a
            # silently-dropped member would both produce a wrong schema.
            values = _choice_values(
                name,
                [m.value for m in ann if m.name != "descriptions"],
            )
            schema[name] = {
                "type": "enum",
                "choices": values,
                "description": _description(name, info),
                "choice_descriptions": _enum_class_descriptions(ann),
            }
        else:
            raise TypeError(f"Field '{name}' has unsupported type {ann!r}. {_SUPPORTED}")
    return schema


def _is_optional_enum(model_cls: type[BaseModel], name: str) -> bool:
    """True when the field's annotation is Optional[<enum-like>]."""
    ann = model_cls.model_fields[name].annotation
    origin = typing.get_origin(ann)
    if origin is not typing.Union:
        return False
    args = [a for a in typing.get_args(ann) if a is not type(None)]
    if len(args) != 1:
        return False
    inner = args[0]
    return (typing.get_origin(inner) is typing.Literal) or (
        isinstance(inner, type) and issubclass(inner, enum.Enum)
    )


def _build_field_results(result: dict, confidence_model: str) -> dict[str, FieldResult]:
    """Build Decision.fields from the engine's field_telemetry.

    Telemetry contract per field: ``log_scores`` ({choice: log P}, enum and
    boolean fields only), ``probability`` (P of the winner), ``top_choices"
    (top 5 {choice, probability}). Multi fields carry ``per_option`` (P(yes)
    per option), ``margin`` (min |P(yes) - threshold|) and no field-level
    probability (None — an exact-set probability is not claimed); their
    alternatives are the per-option pairs sorted by P(yes).

    Margins are unit-split (bug 14): scalar fields get ``log_score_margin``
    (top1-top2 at T=1 log scores) and ``probability_margin`` (top1-top2
    post-temperature); multi fields get ``threshold_distance`` from the
    engine's ``margin``. No field carries more than one of the three.
    """
    fields: dict[str, FieldResult] = {}
    for name, telemetry in result["field_telemetry"].items():
        probability = telemetry["probability"]
        log_scores: dict[str, float] | None = telemetry.get("log_scores")
        if log_scores:
            ranked = sorted(log_scores.items(), key=lambda kv: -kv[1])
            score = ranked[0][1]
            log_score_margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
            # Probability margin: top1-top2 of the post-temperature
            # distribution, from the same ranking as the log scores.
            probs_sorted = [entry["probability"] for entry in telemetry.get("top_choices", [])]
            probability_margin = probs_sorted[0] - probs_sorted[1] if len(probs_sorted) > 1 else 0.0
            threshold_distance = None
            # Top 3 by probability, from top_choices (same ranking as log
            # scores; probabilities are monotone in the log scores).
            alternatives = tuple(
                (entry["choice"], entry["probability"])
                for entry in telemetry.get("top_choices", [])[:3]
            )
        else:
            per_option = telemetry.get("per_option") or {}
            # Multi: no field-level probability is claimed; score stays 0.0
            # (log-space has no value for an unclaimed probability) and the
            # threshold distance comes straight from the engine telemetry.
            score = 0.0
            log_score_margin = None
            probability_margin = None
            threshold_distance = telemetry.get("margin")
            alternatives = tuple(
                (choice, prob) for choice, prob in sorted(per_option.items(), key=lambda kv: -kv[1])
            )
        fields[name] = FieldResult(
            value=telemetry["value"],
            score=score,
            log_score_margin=log_score_margin,
            probability_margin=probability_margin,
            threshold_distance=threshold_distance,
            probability=probability,
            calibrated=False,
            model=confidence_model,
            alternatives=alternatives,
        )
    return fields


def _decide_once[T: BaseModel](
    model_cls: type[T],
    context: str,
    engine_model,
    tokenizer,
    schema: StructuredSchema,
    temperature: float,
    scoring: str = "slots",
    allow_unknown: bool = False,
    multi_threshold: float = 0.5,
    prior_correction: bool = False,
) -> Decision[T]:
    """Decide one context with a loaded engine and a compiled schema."""
    result = run_parallel_generation(
        engine_model,
        tokenizer,
        context,
        schema,
        temperature=temperature,
        scoring=scoring,
        multi_threshold=multi_threshold,
        prior_correction=prior_correction,
    )

    kwargs = {}
    for name, info in model_cls.model_fields.items():
        value = result["parsed_json"][name]["value"]
        ann = info.annotation
        origin = typing.get_origin(ann)
        if allow_unknown and value == UNKNOWN:
            # Synthetic unknown choice maps to None (the field is Optional;
            # _prepare_schema already verified that).
            value = None
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            value = ann(value)
        elif origin in (list, set):
            value = origin(value)  # list or set of the selected literal strings
        kwargs[name] = value

    return Decision(
        value=model_cls(**kwargs),
        fields=_build_field_results(result, result["confidence_model"]),
        latency_ms=result["elapsed_ms"],
    )


def _prepare_schema(model_cls: type[BaseModel], allow_unknown: bool) -> StructuredSchema:
    """Schema dict for the model, with the UNKNOWN choice folded in if asked.

    Raises the caller-facing TypeError when allow_unknown targets a
    non-Optional enum field: without Optional there is no None to map UNKNOWN
    to, so the request is a usage error, not a runtime fallback.
    """
    schema_dict = schema_from_model(model_cls)
    if not allow_unknown:
        return StructuredSchema(schema_dict)
    for name, spec in schema_dict.items():
        if spec["type"] != "enum":
            continue
        if not _is_optional_enum(model_cls, name):
            raise TypeError(
                f"Field '{name}': allow_unknown requires the field to be "
                f"Optional (e.g. {name}: Literal[...] | None) so UNKNOWN can "
                "map to None"
            )
        if UNKNOWN in spec["choices"]:
            raise TypeError(
                f"Field '{name}': a choice named '{UNKNOWN}' already exists; "
                "allow_unknown cannot be used"
            )
        spec["choices"] = [*spec["choices"], UNKNOWN]
        spec["choice_descriptions"] = {
            **spec.get("choice_descriptions", {}),
            UNKNOWN: UNKNOWN_DESCRIPTION,
        }
    return StructuredSchema(schema_dict)


def decide[T: BaseModel](
    model_cls: type[T],
    context: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    scoring: str = "slots",
    allow_unknown: bool = False,
    multi_threshold: float = 0.5,
    prior_correction: bool = False,
) -> Decision[T]:
    """Run parallel constrained decisions and return a validated model instance.

    ``scoring`` selects the engine mode: ``"slots"`` (default) decides
    through neutral aliases listed in the prompt; ``"labels"`` scores the
    real choice text via the token trie.

    ``allow_unknown`` adds a synthetic ``UNKNOWN`` choice (gloss: insufficient
    evidence or none of the options) to every enum field and maps it to None
    in the returned model; declare the field Optional to receive it. Every
    enum field must be Optional when this is set — ``decide`` raises a
    TypeError naming the first non-Optional enum field otherwise.

    ``prior_correction`` subtracts the model's neutral-context prior (one
    batched pass with "(no context provided)" in the delimiters) from the
    per-choice log scores and renormalises; the neutral pass is cached per
    (model, tokenizer, prompt version, scoring mode, plan hash), so
    decide_many pays it once per schema.
    """
    # Validate allow_unknown against the model BEFORE touching the engine: a
    # usage error must not pay for a model load (same rule as decide_many's
    # input validation).
    schema = _prepare_schema(model_cls, allow_unknown)
    engine_model, tokenizer = load_engine(model)
    return _decide_once(
        model_cls,
        context,
        engine_model,
        tokenizer,
        schema,
        temperature,
        scoring=scoring,
        allow_unknown=allow_unknown,
        multi_threshold=multi_threshold,
        prior_correction=prior_correction,
    )


def decide_many[T: BaseModel](
    model_cls: type[T],
    contexts: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    scoring: str = "slots",
    allow_unknown: bool = False,
    multi_threshold: float = 0.5,
    prior_correction: bool = False,
) -> list[Decision[T]]:
    """Decide many contexts against one schema and return one Decision per context.

    Loads the model once and compiles the schema once (the engine's batch plan
    is cached on the StructuredSchema instance, so the compiled suffix and
    choice tokens are reused across contexts); the parallel decision pass then
    runs once per context. Results are returned in input order. ``allow_unknown``
    behaves exactly as in :func:`decide`.

    Raises:
        TypeError: If ``contexts`` is a bare str or bytes (a common mistake
            that would otherwise be decided one character at a time), or if
            any item is not a str.
    """
    if isinstance(contexts, (str, bytes)):
        raise TypeError(
            f"contexts must be a sequence of str, not {type(contexts).__name__}; "
            "wrap a single context in a list"
        )
    contexts = list(contexts)
    for index, item in enumerate(contexts):
        if not isinstance(item, str):
            raise TypeError(f"contexts[{index}] must be str, not {type(item).__name__}")
    if not contexts:
        return []

    schema = _prepare_schema(model_cls, allow_unknown)
    engine_model, tokenizer = load_engine(model)
    return [
        _decide_once(
            model_cls,
            context,
            engine_model,
            tokenizer,
            schema,
            temperature,
            scoring=scoring,
            allow_unknown=allow_unknown,
            multi_threshold=multi_threshold,
            prior_correction=prior_correction,
        )
        for context in contexts
    ]
