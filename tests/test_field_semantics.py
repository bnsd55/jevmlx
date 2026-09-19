"""W5b-13: per-field probability semantics (GPT-REVIEW-2 §C item 9).

The engine's stages set a ``semantics`` record on every field_telemetry
entry (score_source / temperature / calibrator_id / prior_mode /
constraint_changed / dependency_rescored); the public API coerces it into
the frozen :class:`jevmlx.api.FieldSemantics`; the result-level
``probability_status`` summarizes the distinct semantic groups.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import jevmlx
from jevmlx.api import FieldResult, FieldSemantics
from jevmlx.engine import run_parallel_generation

# --- the dataclass contract ----------------------------------------------------


def test_field_semantics_frozen_and_field_names():
    """Frozen dataclass; the six §C9 field names in the design order."""
    s = FieldSemantics(
        score_source="batched",
        temperature=1.0,
        calibrator_id=None,
        prior_mode="off",
        constraint_changed=False,
        dependency_rescored=False,
    )
    assert dataclasses.is_dataclass(s)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.temperature = 0.7  # type: ignore[misc]
    assert [f.name for f in dataclasses.fields(s)] == [
        "score_source",
        "temperature",
        "calibrator_id",
        "prior_mode",
        "constraint_changed",
        "dependency_rescored",
    ]


def test_field_result_requires_semantics():
    """FieldResult is kw_only-frozen and semantics is REQUIRED (no default):
    a construction without it is a TypeError, never a silent default."""
    kwargs = dict(
        value="A",
        score=0.0,
        log_score_margin=None,
        probability_margin=None,
        threshold_distance=None,
        probability=0.9,
        calibrated=False,
        model="slots",
        alternatives=(),
        reason=None,
    )
    with pytest.raises(TypeError, match="semantics"):
        FieldResult(**kwargs)
    s = FieldSemantics("batched", 1.0, None, "off", False, False)
    fr = FieldResult(**kwargs, semantics=s)
    assert fr.semantics is s


def test_field_semantics_exported():
    """Public surface: importable from jevmlx.api, listed in __all__.

    Both bindings come from ONE import path inside the test: the
    test_check_results module eviction can leave this file's collection-time
    binding on a stale module instance, and cross-instance identity would
    fail for the wrong reason (see test_bench_machine_tag_patches_live_module)."""
    import jevmlx.api as api
    from jevmlx.api import FieldSemantics as FS  # noqa: N806 — same-path binding

    assert api.FieldSemantics is FS
    assert "FieldSemantics" in api.__all__


# --- engine filling: one pin per behavior ---------------------------------------


def _decide_with(monkeypatch, *, fields: dict, model_fields: type | None = None, **request_kwargs):
    """decide() over the shared fake engine with the given telemetry fields.

    ``model_fields``: the pydantic model to decide against — defaults to a
    single-scalar Ticket; pass a custom model when the fixture carries
    other fields."""
    import jevmlx.api as api
    from tests.conftest import make_engine_result

    result = make_engine_result(fields=fields)
    monkeypatch.setattr(api, "run_parallel_generation", lambda *a, **k: result)
    monkeypatch.setattr(api, "load_engine", lambda model_id: ("engine", "tokenizer"))

    if model_fields is None:

        class Ticket(BaseModel):
            risk_tier: Literal["HIGH", "LOW", "CRITICAL"] = Field(description="Risk tier")

        model_fields = Ticket

    return jevmlx.decide(model_fields, "ctx", model="fake/model", **request_kwargs), result


def _scalar_field(**overrides):
    from tests.conftest import make_field_semantics, make_field_telemetry

    return make_field_telemetry(
        value="HIGH",
        choices=["HIGH", "LOW", "CRITICAL"],
        semantics=make_field_semantics(**overrides),
    )


def test_decide_fields_carry_semantics(monkeypatch):
    """Every decided field carries a complete frozen FieldSemantics — the
    engine's record is coerced at the public boundary; None is impossible.

    Both FieldSemantics bindings come from ONE import path inside the test
    (the test_check_results module eviction can leave this file's
    collection-time class object on a stale module instance — compare via
    the same-path binding, not collection-time identity)."""
    from jevmlx.api import FieldSemantics as FS  # noqa: N806 — same-path binding

    d, _ = _decide_with(
        monkeypatch,
        fields={
            "risk_tier": _scalar_field(score_source="rescored_batch1", temperature=0.5),
        },
    )
    sem = d.fields["risk_tier"].semantics
    assert isinstance(sem, FS)
    # Same-path binding for the equality too (cross-instance equality is by
    # value here, but the constructor must come from the same module as the
    # coercion — see the module-eviction note above).
    assert dataclasses.asdict(sem) == dataclasses.asdict(
        FS(
            score_source="rescored_batch1",
            temperature=0.5,
            calibrator_id=None,
            prior_mode="off",
            constraint_changed=False,
            dependency_rescored=False,
        )
    )


def test_missing_semantics_record_fails_loudly(monkeypatch):
    """A telemetry entry without a semantics record is a contract violation:
    _build_field_results raises naming the field — never a silent None
    (F2: the boundary indexes directly; a KeyError and the ValueError are
    both loud, so the pin accepts either)."""
    import jevmlx.api as api
    from tests.conftest import make_engine_result, make_field_telemetry

    result = make_engine_result(
        fields={"risk_tier": make_field_telemetry(value="HIGH", choices=["HIGH", "LOW"])}
    )
    del result["field_telemetry"]["risk_tier"]["semantics"]
    monkeypatch.setattr(api, "run_parallel_generation", lambda *a, **k: result)
    monkeypatch.setattr(api, "load_engine", lambda model_id: ("engine", "tokenizer"))

    class Ticket(BaseModel):
        risk_tier: Literal["HIGH", "LOW"] = Field(description="Risk tier")

    with pytest.raises((ValueError, KeyError), match="semantics"):
        jevmlx.decide(Ticket, "ctx", model="fake/model")


def _finalize_with(fields: dict[str, dict]) -> dict:
    """finalize_public_result over a typed AssembledState carrying the given
    field_telemetry — the stage that builds probability_status."""
    from jevmlx.engine import Ledger, ScoreRowsResult, finalize_public_result
    from jevmlx.schema import StructuredSchema

    schema = StructuredSchema(
        {
            "risk_tier": {"type": "enum", "description": "d", "choices": ["HIGH", "LOW"]},
            "tags": {"type": "multi", "description": "d", "choices": ["a", "b"]},
        }
    )
    state = _AssembledStateShim(fields)
    scored = ScoreRowsResult(row_logits={}, row_legal_mass_log={}, passes=1, chunk_shapes=[])
    built = {"scoring": "slots", "rows": []}
    return finalize_public_result(
        schema=schema,
        state=state,  # type: ignore[arg-type]
        scored=scored,
        built=built,
        second_pass_telemetry={"rerun_fields": [], "rerun_rows": 0},
        timings={"peak_active_bytes": 1, "peak_incremental_bytes": 1},
        temperature=1.0,
        prior_correction=False,
        prior_ms=0.0,
        constraints=None,
        base_ids=[],
        active_start=0,
        ledger=Ledger(),
    )


class _AssembledStateShim:
    """Duck-typed AssembledState: finalize_public_result reads .field_telemetry
    plus the pass-through tuples; a real AssembledState requires frozen dict
    shapes the tests don't need."""

    def __init__(self, field_telemetry):
        self.field_telemetry = field_telemetry
        self.rescored_fields = ()
        self.reconciled_fields = ()
        self.internal_telemetry = types.MappingProxyType({})
        self.parsed_json = {
            name: {"value": ft.get("value")} for name, ft in field_telemetry.items()
        }


def test_probability_status_is_a_group_summary():
    """The result-level probability_status is a SUMMARY over the distinct
    (score_source, temperature, calibrator_id, prior_mode) groups: one clause
    per group with its field count — not authoritative for any single field."""
    from tests.conftest import make_field_semantics, make_field_telemetry

    result = _finalize_with(
        {
            # Two fields share one group (scalar, T=1, uncalibrated, prior off)...
            "risk_tier": make_field_telemetry(
                value="HIGH", choices=["HIGH", "LOW"], semantics=make_field_semantics()
            ),
            "tags": make_field_telemetry(
                value=["a"],
                type_="multi",
                choices=["a", "b"],
                per_option={"a": 0.9, "b": 0.1},
                semantics=make_field_semantics(temperature=None),
            ),
        }
    )
    status = result["probability_status"]
    # ...but here they differ (multi carries temperature=None) -> 2 groups.
    assert status.count("1 field:") == 2
    # Group clauses name the semantics dimensions.
    assert "batched" in status
    assert "prior-corrected" not in status  # both groups are prior_mode=off


def test_probability_status_names_prior_and_calibrator_groups():
    """A prior-corrected scalar and a calibrated multi land in DISTINCT
    groups; each clause carries its own prior_mode / calibrator identity."""
    from tests.conftest import make_field_semantics, make_field_telemetry

    result = _finalize_with(
        {
            "risk_tier": make_field_telemetry(
                value="HIGH",
                choices=["HIGH", "LOW"],
                semantics=make_field_semantics(prior_mode="neutral_v1"),
            ),
            "tags": make_field_telemetry(
                value=["a"],
                type_="multi",
                choices=["a", "b"],
                per_option={"a": 0.9, "b": 0.1},
                semantics=make_field_semantics(temperature=None, calibrator_id="test-rev-abc"),
            ),
        }
    )
    status = result["probability_status"]
    assert "prior-corrected against the neutral-context pass" in status
    assert "calibrated (bundle test-rev-abc)" in status


# --- engine-stage pins (real fake-engine path, one behavior each) ----------------


def test_engine_sets_semantics_on_all_shapes():
    """The engine's stages set a semantics record on EVERY telemetry shape:
    scalar, multi, cardinality-1, and the dependency re-decided child."""
    from jevmlx.schema import StructuredSchema
    from tests.conftest import FakeModel, FakeTokenizer, make_engine
    from tests.test_w5b import BijectiveTokenizer, RoutingBiasModel, _chain_schema

    # Scalar + multi (FakeModel ties -> rescored_batch1 on both).
    schema = StructuredSchema(
        {
            "action": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "flags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
        }
    )
    r = run_parallel_generation(make_engine(FakeModel(), FakeTokenizer()), "ctx", schema)
    sem_action = r["field_telemetry"]["action"]["semantics"]
    assert sem_action["score_source"] == "rescored_batch1"
    assert sem_action["temperature"] == 1.0
    assert sem_action["prior_mode"] == "off"
    # Multi: uncalibrated fake run — the caller T (1.0) IS applied to the
    # P(yes) softmax, so the record carries it (None only under calibration).
    sem_flags = r["field_telemetry"]["flags"]["semantics"]
    assert sem_flags["score_source"] == "rescored_batch1"
    assert sem_flags["temperature"] == 1.0
    assert sem_flags["prior_mode"] == "off"

    # Cardinality-1 + dependency re-decided child (chain schema: pa has one
    # choice -> schema-determined; cb re-decided in a wave).
    chain = StructuredSchema(_chain_schema())
    r2 = run_parallel_generation(
        make_engine(RoutingBiasModel(), BijectiveTokenizer()), "ctx", chain
    )
    sem_pa = r2["field_telemetry"]["pa"]["semantics"]
    assert sem_pa["score_source"] == "batched"
    assert sem_pa["temperature"] is None  # schema-determined: no T applied
    sem_cb = r2["field_telemetry"]["cb"]["semantics"]
    assert sem_cb["score_source"] == "dependency"
    assert sem_cb["dependency_rescored"] is True


def test_constraint_flip_records_constraint_changed():
    """A case-constraint MAP flip sets constraint_changed=True on exactly the
    flipped field's semantics record."""
    from jevmlx.schema import StructuredSchema
    from tests.conftest import make_engine
    from tests.test_w5b import BijectiveTokenizer, RoutingBiasModel

    schema = StructuredSchema(
        {
            "a": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "b": {"type": "enum", "description": "d", "choices": ["X", "Y"]},
        }
    )
    # Both fields bias to their first choice (A, X); A and X are forbidden
    # jointly, so the MAP must flip one of them.
    cons = [{"type": "excludes", "field": "a", "value": "A", "other": "b", "other_value": "X"}]
    r = run_parallel_generation(
        make_engine(RoutingBiasModel(), BijectiveTokenizer()), "ctx", schema, constraints=cons
    )
    assert r["reconciled_fields"]
    for fname in r["reconciled_fields"]:
        assert r["field_telemetry"][fname]["semantics"]["constraint_changed"] is True
    untouched = set(r["field_telemetry"]) - set(r["reconciled_fields"])
    for fname in untouched:
        assert r["field_telemetry"][fname]["semantics"]["constraint_changed"] is False


def test_calibrated_multi_and_scalar_temperature_split_groups():
    """A calibrated multi ignores the caller temperature (record temperature
    None + bundle id); the scalar next to it carries the caller T — two
    distinct groups in the summary."""
    from jevmlx.calibrate import CalibrationBundle
    from jevmlx.schema import StructuredSchema
    from tests.conftest import FakeModel, FakeTokenizer, make_engine

    schema = StructuredSchema(
        {
            "action": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "flags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
        }
    )
    bundle = CalibrationBundle.from_payload({"multi": {"a": 1.0, "b": 0.0}})
    r = run_parallel_generation(
        make_engine(FakeModel(), FakeTokenizer()),
        "ctx",
        schema,
        temperature=0.7,
        calibration=bundle,
    )
    sem_flags = r["field_telemetry"]["flags"]["semantics"]
    assert sem_flags["temperature"] is None
    assert sem_flags["calibrator_id"] == bundle.identity()
    sem_action = r["field_telemetry"]["action"]["semantics"]
    assert sem_action["temperature"] == 0.7
    assert sem_action["calibrator_id"] is None
    # The summary carries both groups separately.
    status = r["probability_status"]
    assert "T=0.7" in status
    assert f"calibrated (bundle {bundle.identity()})" in status
    assert "temperature not applied (fixed rule selection)" in status


def test_run_parallel_generation_imported_here():
    """Guard: the module-level import used by the pins above resolves to the
    live engine module (the test_check_results eviction trap)."""
    from jevmlx.engine import run_parallel_generation  # noqa: F401


def test_prior_correction_lands_neutral_v1_on_real_run():
    """F6a: prior_correction=True on a REAL engine path lands
    prior_mode='neutral_v1' on every field's semantics record — asserted on
    the run, not a hand-made dict. Fails if the scalar/multi setters stop
    reading decision.prior_corrected / prior_entry."""
    from jevmlx.schema import StructuredSchema
    from tests.conftest import FakeModel, FakeTokenizer, make_engine

    schema = StructuredSchema(
        {
            "action": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "flags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
        }
    )
    corrected = run_parallel_generation(
        make_engine(FakeModel(), FakeTokenizer()), "ctx", schema, prior_correction=True
    )
    raw = run_parallel_generation(make_engine(FakeModel(), FakeTokenizer()), "ctx", schema)
    for fname in ("action", "flags"):
        assert corrected["field_telemetry"][fname]["semantics"]["prior_mode"] == "neutral_v1"
        assert raw["field_telemetry"][fname]["semantics"]["prior_mode"] == "off"


def test_count_row_semantics_temperature_none_and_score_source_copied():
    """F6b: the count row's semantics carries temperature None (fixed T=1
    bucket softmaxes) and copies the parent multi's score_source + prior
    mode. Fails if the count setter stops deriving from the parent."""
    from jevmlx.schema import StructuredSchema
    from tests.conftest import FakeModel, FakeTokenizer, make_engine

    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    r = run_parallel_generation(
        make_engine(FakeModel(), FakeTokenizer()), "ctx", schema, prior_correction=True
    )
    count = r["internal_telemetry"]["flags#count"]
    parent = r["field_telemetry"]["flags"]
    assert count["semantics"]["temperature"] is None
    assert count["semantics"]["score_source"] == parent["semantics"]["score_source"]
    assert count["semantics"]["prior_mode"] == parent["semantics"]["prior_mode"] == "neutral_v1"


def test_uncalibrated_multi_carries_caller_temperature():
    """F6c: an UNCALIBRATED multi records the caller temperature (the P(yes)
    softmax used it) and prior_mode off without prior correction. Fails if
    the multi setter stops passing the caller T on the uncalibrated path."""
    from jevmlx.schema import StructuredSchema
    from tests.conftest import FakeModel, FakeTokenizer, make_engine

    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    r = run_parallel_generation(
        make_engine(FakeModel(), FakeTokenizer()), "ctx", schema, temperature=0.6
    )
    sem = r["field_telemetry"]["flags"]["semantics"]
    assert sem["temperature"] == 0.6
    assert sem["calibrator_id"] is None
    assert sem["prior_mode"] == "off"


def test_trusted_nonbinding_count_reports_constraint_changed_false():
    """F3/F4: neither an untrusted count (margin below COUNT_MARGIN_MIN: no
    constraint ran) nor a trusted count whose constraint was non-binding
    may claim constraint_changed — only a reconciler that CHANGED the
    selection (or the count row's own trusted constraint running) may."""
    from jevmlx.schema import StructuredSchema
    from tests.conftest import FakeModel, FakeTokenizer, make_engine

    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}}
    )
    # Zero logits -> P(yes)=0.5 for every option; the count row ties across
    # buckets (untrusted), the threshold proposal selects all, nothing
    # changed anything.
    r = run_parallel_generation(make_engine(FakeModel(), FakeTokenizer()), "ctx", schema)
    sem = r["field_telemetry"]["flags"]["semantics"]
    assert sem["constraint_changed"] is False
    assert r["internal_telemetry"]["flags#count"]["semantics"]["constraint_changed"] is False
