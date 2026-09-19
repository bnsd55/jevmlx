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
import types
import typing
from collections.abc import Sequence

from pydantic import BaseModel

from jevmlx.calibrate import CalibrationBundle
from jevmlx.engine import (
    load_engine,
    run_parallel_generation,
    run_parallel_generation_batched,
)
from jevmlx.models import DEFAULT_MODEL
from jevmlx.schema import StructuredSchema

__all__ = [
    "DEFAULT_MODEL",
    "NONE_OF_ABOVE",
    "NONE_OF_ABOVE_DESCRIPTION",
    "Decision",
    "FieldResult",
    "FieldSemantics",
    "OrdinalFieldRecord",
    "Ordered",
    "decide",
    "decide_many",
    "schema_from_model",
]

_SUPPORTED = (
    "supported field types: bool, Literal[str, ...], enum.Enum/enum.StrEnum with str values, "
    "list[Literal[...]] / set[Literal[...]] (multi)"
)

# Explicit opt-out choice appended to enum fields when allow_none_of_above
# is set. Means exactly "none of the options apply" and maps to None; the
# old allow_unknown conflated this with low confidence, which is now the
# separate, thresholded ``abstain`` (see FieldResult.abstain).
NONE_OF_ABOVE = "NONE_OF_ABOVE"
NONE_OF_ABOVE_DESCRIPTION = "none of the options apply"


class Ordered:
    """Marks an enum field as an ORDINAL scale (W6-B1).

    Attach via ``Field(json_schema_extra={"ordered": True})`` or the plain
    marker ``Field(json_schema_extra=Ordered())`` — the choices' DECLARATION
    order is the scale order (level 0 = first Literal member). The decided
    value stays the winning level (a plain string); ordering only adds
    ordinal telemetry (expected index / variance) and ordinal metrics.

        class Sentiment(BaseModel):
            level: Literal["0", "1", "2", "3", "4"] = Field(
                description="Sentiment level",
                json_schema_extra={"ordered": True},
            )
    """

    def __repr__(self) -> str:
        return "Ordered()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ordered)

    def __hash__(self) -> int:
        return hash("Ordered")


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
class FieldSemantics:
    """How ONE field's reported probabilities were produced (W5b-13, §C9).

    The per-field truth a global ``probability_status`` string cannot
    carry: a result mixes temperature-scaled scalars, calibrated multi
    options that ignore the caller temperature, count-row bucket scores,
    prior-corrected fields, and dependency/oracle re-scores. Each field
    records its own semantics; the result-level ``probability_status``
    becomes a summary of these records.

    Attributes:
        score_source: Which scoring path produced the final evidence:
            "batched" (batched pass), "rescored_batch1" (canonical batch=1
            rescore replaced it), "dependency" (second pass conditioned on
            the parent), "oracle" (forced re-score under oracle_overrides).
        temperature: The temperature actually applied to the reported
            distribution. None for count rows (fixed T=1 bucket scores)
            and for calibrated multi selections (the calibrated log-odds
            ``a*(yes-no)+b`` cut ignores the caller temperature).
        calibrator_id: Identity of the CalibrationBundle whose fitted
            calibrator set the selection (multi only); None = uncalibrated.
        prior_mode: "off" or "neutral_v1" — whether (and how) the scores
            were prior-corrected against the neutral-context pass.
        constraint_changed: A reconciler (count / set constraints / case
            MAP) overrode the raw winner.
        dependency_rescored: Re-scored conditioned on the parent value in
            a dependency wave.
    """

    score_source: str
    temperature: float | None
    calibrator_id: str | None
    prior_mode: str
    constraint_changed: bool
    dependency_rescored: bool


@dataclasses.dataclass(frozen=True, kw_only=True)
class OrdinalFieldRecord:
    """W6-B1: ordinal telemetry for ONE ordered-enum field.

    Derived from the finalized distribution over the field's declared
    (scale) order — no extra model call, the decided value unchanged.

    Attributes:
        argmax_level: index of the winning level in scale order.
        expected_index: distribution mean over the scale (sum p_i * i).
        variance: distribution spread around the mean.
        expected_score_normalized: expected_index / (n - 1), in [0, 1].
    """

    argmax_level: int
    expected_index: float
    variance: float
    expected_score_normalized: float


@dataclasses.dataclass(frozen=True, kw_only=True)
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
        reason: Why the field carries no decided value, or None when it
            does. "none_of_above": the caller opted in via
            ``allow_none_of_above=True`` and the model picked the explicit
            NONE_OF_ABOVE option ("none of the options apply" -> None).
            "abstain": the caller set ``abstain_below_margin`` and the
            field's confidence sat below that cut — the value is withheld
            even though the engine produced one. The two are deliberately
            separate: one is a schema-level answer, the other a confidence
            gate (calibrated abstention is a later milestone). The
            only values are None, "none_of_above" and "abstain"; a
            withheld decision is exactly ``reason == "abstain"``.
        semantics: HOW this field's reported probabilities were produced —
            see :class:`FieldSemantics`. Required: the engine's stages set
            the record and _build_field_results coerces it; a telemetry
            entry without one raises (never a silent None).
        ordinal: W6-B1, ORDERED enums only — the derived ordinal record
            (see :class:`OrdinalFieldRecord`); None for unordered fields.
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
    reason: str | None
    semantics: FieldSemantics
    ordinal: OrdinalFieldRecord | None = None


@dataclasses.dataclass
class Decision[T: BaseModel]:
    value: T
    fields: dict[str, FieldResult]
    latency_ms: float


def _description(name: str, info) -> str:
    return info.description or name.replace("_", " ")


def _ordered_flag(name: str, info) -> bool:
    """Read the W6-B1 ordered marker from a field's json_schema_extra.

    Accepted: ``{"ordered": True}`` or an ``Ordered()`` instance (bare or
    merged in a dict). Anything else (including truthy non-bool junk) is a
    TypeError — an accidentally-ordered field would silently corrupt every
    ordinal metric, so this fails loudly.
    """
    extra = info.json_schema_extra
    if extra is None:
        return False
    if isinstance(extra, Ordered):
        return True
    if isinstance(extra, dict):
        if "ordered" not in extra:
            return False
        flag = extra["ordered"]
        if isinstance(flag, Ordered):
            return True
        if flag is True:
            return True
        if flag is False:
            return False
    raise TypeError(
        f"Field '{name}': json_schema_extra['ordered'] must be True, False, "
        f"or Ordered(); got {extra!r}"
    )


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
    """Inner type of Optional[X] (Union with exactly one non-None arg), else None.

    Both Optional spellings count: ``typing.Union[X, None]`` (origin
    ``typing.Union``) and PEP 604 ``X | None`` (origin ``types.UnionType``).
    Pydantic normalizes some annotations to one form and not the other
    (Literal[...] | None arrives as typing.Union, an enum class | None
    arrives as types.UnionType), so both must be accepted.
    """
    if typing.get_origin(ann) not in (typing.Union, types.UnionType):
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
    identical to the non-Optional form — None is the NONE_OF_ABOVE or
    abstention mapping, not a decided value. Any other Optional raises.
    """
    schema: dict = {}
    for name, info in model_cls.model_fields.items():
        ann = info.annotation
        origin = typing.get_origin(ann)
        if origin in (typing.Union, types.UnionType):
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
                "ordered": _ordered_flag(name, info),
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
                "ordered": _ordered_flag(name, info),
            }
        else:
            raise TypeError(f"Field '{name}' has unsupported type {ann!r}. {_SUPPORTED}")
    return schema


def _is_optional_enum(model_cls: type[BaseModel], name: str) -> bool:
    """True when the field's annotation is Optional[<enum-like>].

    Both Optional spellings (typing.Union and PEP 604 types.UnionType) —
    pydantic keeps ``EnumClass | None`` as a raw types.UnionType, which the
    old ``is typing.Union`` check silently rejected.
    """
    ann = model_cls.model_fields[name].annotation
    origin = typing.get_origin(ann)
    if origin not in (typing.Union, types.UnionType):
        return False
    args = [a for a in typing.get_args(ann) if a is not type(None)]
    if len(args) != 1:
        return False
    inner = args[0]
    return (typing.get_origin(inner) is typing.Literal) or (
        isinstance(inner, type) and issubclass(inner, enum.Enum)
    )


def _build_field_results(
    result: dict,
    confidence_model: str,
    *,
    abstain_below_margin: float | None = None,
) -> dict[str, FieldResult]:
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

    Abstention (W2-D): with ``abstain_below_margin`` set, a scalar field
    whose ``probability_margin`` is below the cut gets ``abstain=True`` and
    ``reason="abstain"``; a multi field uses ``threshold_distance`` the
    same way. The decided value is kept on the FieldResult (provenance),
    but the validated model instance maps the field to None.
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
        reason = None
        abstain_below_margin = abstain_below_margin
        if abstain_below_margin is not None:
            margin = probability_margin if log_scores else threshold_distance
            if margin is not None and margin < abstain_below_margin:
                reason = "abstain"
        # W5-C finding 23: calibrated reflects the APPLIED calibrator for
        # THIS field — scalar fields carry {"temperature": T} when the
        # fitted temperature was applied, multi fields the (a, b) pair.
        # Provenance (the bundle identity) belongs to the telemetry, not to
        # confidence_model: model stays the clean decision-model name.
        calibrated_flag = telemetry.get("calibrated") is not None
        # W5b-13 step 2: the engine's stages set the semantics record; the
        # public API coerces it into the frozen FieldSemantics. A telemetry
        # entry without one is a contract violation — fail loudly, never
        # ship a None semantics. (This is the boundary CHECK, not a
        # tolerant read: the index below raises when the record is absent
        # or not a dict.)
        sem_dict = telemetry["semantics"]
        if not isinstance(sem_dict, dict):
            raise ValueError(
                f"field telemetry for {name!r} carries no semantics record "
                "(results-contract violation; engine stages must set "
                "field_telemetry[fname]['semantics'])"
            )
        # W6-B1: ordered enums carry the engine's derived ordinal record;
        # unordered fields have no key and keep ordinal=None.
        ordinal_record = None
        ord_dict = telemetry.get("ordinal")
        if ord_dict is not None:
            ordinal_record = OrdinalFieldRecord(
                argmax_level=ord_dict["argmax_level"],
                expected_index=ord_dict["expected_index"],
                variance=ord_dict["variance"],
                expected_score_normalized=ord_dict["expected_score_normalized"],
            )
        fields[name] = FieldResult(
            value=telemetry["value"],
            score=score,
            log_score_margin=log_score_margin,
            probability_margin=probability_margin,
            threshold_distance=threshold_distance,
            probability=probability,
            calibrated=calibrated_flag,
            model=confidence_model,
            alternatives=alternatives,
            reason=reason,
            semantics=FieldSemantics(**sem_dict),
            ordinal=ordinal_record,
        )
    return fields


def _decide_once[T: BaseModel](
    model_cls: type[T],
    context: str,
    engine,
    schema: StructuredSchema,
    temperature: float,
    scoring: str = "slots",
    allow_none_of_above: bool = False,
    abstain_below_margin: float | None = None,
    calibration: str | CalibrationBundle | None = None,
    prior_correction: bool = False,
    constraints: list[dict] | None = None,
) -> Decision[T]:
    """Decide one context with a loaded :class:`Engine` and a compiled schema
    The Engine is the single handle; no (model, tokenizer) pairs."""
    result = run_parallel_generation(
        engine,
        context,
        schema,
        temperature=temperature,
        scoring=scoring,
        calibration=_resolve_calibration(calibration),
        prior_correction=prior_correction,
        constraints=constraints,
    )
    return _assemble_decision(
        model_cls,
        result,
        abstain_below_margin=abstain_below_margin,
        allow_none_of_above=allow_none_of_above,
    )


def _assemble_decision[T: BaseModel](
    model_cls: type[T],
    result: dict,
    *,
    abstain_below_margin: float | None = None,
    allow_none_of_above: bool = False,
) -> Decision[T]:
    """Turn a run_parallel_generation result into a validated Decision.

    Shared by decide/_decide_once and the W3-F batched decide_many path.
    """
    field_results = _build_field_results(
        result, result["confidence_model"], abstain_below_margin=abstain_below_margin
    )

    kwargs = {}
    for name, info in model_cls.model_fields.items():
        value = result["parsed_json"][name]["value"]
        ann = info.annotation
        origin = typing.get_origin(ann)
        if allow_none_of_above and value == NONE_OF_ABOVE:
            # Explicit opt-out choice maps to None (the field is Optional;
            # _prepare_schema already verified that).
            value = None
            field_results[name] = dataclasses.replace(field_results[name], reason="none_of_above")
        elif field_results[name].reason == "abstain":
            # Confidence-gated abstention: keep the engine's value on the
            # FieldResult (provenance) but withhold it from the model.
            value = None
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            value = ann(value)
        elif origin in (list, set):
            value = origin(value)  # list or set of the selected literal strings
        kwargs[name] = value

    return Decision(
        value=model_cls(**kwargs),
        fields=field_results,
        latency_ms=result["elapsed_ms"],
    )


def _resolve_calibration(
    calibration: str | CalibrationBundle | None,
) -> CalibrationBundle | None:
    """W5-C F3 boundary: load the bundle HERE, before the engine.

    ``decide``/``decide_many`` accept a JSON file path (what ``jevmlx
    calibrate --out`` writes) or a constructed bundle — NO dict payload
    (a second shape is a dual path; ``CalibrationBundle.load`` is the only
    boundary for files, and the ONE bundle shape is what the CLI writes).
    The ENGINE itself takes only ``CalibrationBundle | None`` — no file
    I/O, no duck typing. One helper, one load, shared by all entry points.
    """
    if calibration is None or isinstance(calibration, CalibrationBundle):
        return calibration
    if isinstance(calibration, str):
        try:
            return CalibrationBundle.load(calibration)
        except ValueError as exc:
            raise ValueError(f"calibration bundle invalid: {exc}") from exc
    raise TypeError(
        "calibration must be a JSON file path, a CalibrationBundle, or None, "
        f"got {type(calibration).__name__}"
    )


def _check_abstain_margin(abstain_below_margin: float | None) -> None:
    """Validate the abstention cut: None (disabled) or a float in [0, 1)."""
    if abstain_below_margin is not None and not 0.0 <= abstain_below_margin < 1.0:
        raise TypeError("abstain_below_margin must be in [0, 1) or None")


def _check_abstention_schema[T: BaseModel](model_cls: type[T]) -> None:
    """W5-C finding 25: abstention maps a withheld field to None — that is
    only representable when the field's annotation is Optional. A
    non-Optional field with abstain_below_margin set would fail Pydantic
    validation AFTER inference (the worst possible time). The contract is
    enforced at decide() start: every field of the model must be Optional
    (any spelling) or the request is a usage error naming the first
    offending field."""
    for name, info in model_cls.model_fields.items():
        ann = info.annotation
        origin = typing.get_origin(ann)
        is_optional = origin in (typing.Union, types.UnionType) and type(None) in typing.get_args(
            ann
        )
        if not is_optional:
            raise TypeError(
                f"Field '{name}': abstain_below_margin requires every field to be "
                f"Optional (e.g. {name}: ... | None) so a withheld field can map "
                "to None; abstention would otherwise fail validation after "
                "inference"
            )


def _prepare_schema(model_cls: type[BaseModel], allow_none_of_above: bool) -> StructuredSchema:
    """Schema dict for the model, with the NONE_OF_ABOVE choice folded in if asked.

    Raises the caller-facing TypeError when allow_none_of_above targets a
    non-Optional enum field: without Optional there is no None to map the
    explicit opt-out to, so the request is a usage error, not a runtime
    fallback.
    """
    schema_dict = schema_from_model(model_cls)
    if not allow_none_of_above:
        return StructuredSchema(schema_dict)
    for name, spec in schema_dict.items():
        if spec["type"] != "enum":
            continue
        if not _is_optional_enum(model_cls, name):
            raise TypeError(
                f"Field '{name}': allow_none_of_above requires the field to be "
                f"Optional (e.g. {name}: Literal[...] | None) so NONE_OF_ABOVE "
                "can map to None"
            )
        if NONE_OF_ABOVE in spec["choices"]:
            raise TypeError(
                f"Field '{name}': a choice named '{NONE_OF_ABOVE}' already exists; "
                "allow_none_of_above cannot be used"
            )
        spec["choices"] = [*spec["choices"], NONE_OF_ABOVE]
        spec["choice_descriptions"] = {
            **spec.get("choice_descriptions", {}),
            NONE_OF_ABOVE: NONE_OF_ABOVE_DESCRIPTION,
        }
    return StructuredSchema(schema_dict)


def decide[T: BaseModel](
    model_cls: type[T],
    context: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    scoring: str = "slots",
    allow_none_of_above: bool = False,
    abstain_below_margin: float | None = None,
    calibration: str | CalibrationBundle | None = None,
    prior_correction: bool = False,
    constraints: list[dict] | None = None,
) -> Decision[T]:
    """Run parallel constrained decisions and return a validated model instance.

    ``scoring`` selects the engine mode: ``"slots"`` (default) decides
    through neutral aliases listed in the prompt; ``"labels"`` scores the
    real choice text via the token trie.

    ``allow_none_of_above`` adds an explicit ``NONE_OF_ABOVE`` choice (gloss:
    "none of the options apply") to every enum field and maps it to None in
    the returned model — the field-level answer is literally "none of
    these". Declare the field Optional to receive it; every enum field must
    be Optional when this is set — ``decide`` raises a TypeError naming the
    first non-Optional enum field otherwise. This is a schema-level answer,
    distinct from:

    ``abstain_below_margin`` — a confidence gate, not a choice. When set
    (a float in [0, 1)), any scalar field whose ``probability_margin``
    (top1-top2, post-temperature) or multi field whose
    ``threshold_distance`` (min |P(yes) - threshold|) sits below the cut is
    abstained: the validated model maps it to None, the FieldResult keeps
    the engine's raw value for provenance and carries ``abstain=True`` /
    ``reason="abstain"``. The threshold is a raw margin cut for now; the
    calibrated correctness model is a later milestone.

    ``prior_correction`` subtracts the model's neutral-context prior (one
    batched pass with "(no context provided)" in the delimiters) from the
    per-choice log scores and renormalises; the neutral pass is cached per
    (model, tokenizer, prompt version, scoring mode, plan hash), so
    decide_many pays it once per schema.
    """
    # Validate allow_none_of_above against the model BEFORE touching the
    # engine: a usage error must not pay for a model load (same rule as
    # decide_many's input validation).
    _check_abstain_margin(abstain_below_margin)
    if abstain_below_margin is not None:
        _check_abstention_schema(model_cls)
    schema = _prepare_schema(model_cls, allow_none_of_above)
    engine = load_engine(model)
    return _decide_once(
        model_cls,
        context,
        engine,
        schema,
        temperature,
        scoring=scoring,
        allow_none_of_above=allow_none_of_above,
        abstain_below_margin=abstain_below_margin,
        calibration=_resolve_calibration(calibration),
        prior_correction=prior_correction,
        constraints=constraints,
    )


def decide_many[T: BaseModel](
    model_cls: type[T],
    contexts: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    scoring: str = "slots",
    allow_none_of_above: bool = False,
    abstain_below_margin: float | None = None,
    calibration: str | CalibrationBundle | None = None,
    prior_correction: bool = False,
    constraints: list[dict] | None = None,
) -> list[Decision[T]]:
    """Decide many contexts against one schema and return one Decision per context.

    Loads the model once and compiles the schema once (the engine's batch plan
    is cached on the StructuredSchema instance, so the compiled suffix and
    choice tokens are reused across contexts); the parallel decision pass then
    runs once per context. Results are returned in input order.
    ``allow_none_of_above`` and ``abstain_below_margin`` behave exactly as in
    :func:`decide`.

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

    _check_abstain_margin(abstain_below_margin)
    if abstain_below_margin is not None:
        _check_abstention_schema(model_cls)
    schema = _prepare_schema(model_cls, allow_none_of_above)
    engine = load_engine(model)
    # W3-F: batch the contexts (one merged scoring pass per context group).
    # Each Decision is assembled through the same post-processing as
    # decide() (_assemble_decision), so validation / abstain / NONE_OF_ABOVE
    # behave identically; only the forward passes are shared.
    raws = run_parallel_generation_batched(
        engine,
        contexts,
        schema,
        temperature=temperature,
        scoring=scoring,
        calibration=_resolve_calibration(calibration),
        prior_correction=prior_correction,
        constraints=constraints,
    )
    return [
        _assemble_decision(
            model_cls,
            raw,
            abstain_below_margin=abstain_below_margin,
            allow_none_of_above=allow_none_of_above,
        )
        for raw in raws
    ]
