"""Fix: finalize_public_result raised on an empty schema (zero fields).

Root cause: the typesafe dataset carries cases whose schema is {} (a
workflow variant with no scorable fields). The engine loops over
schema.fields (empty), field_telemetry stays {}, and finalize_public_result
raised 'no field carries a semantics record' — a results-contract violation
that is actually a legal degenerate case.

Fix at the owning layer: an empty schema is legal. The engine returns a
valid result with num_fields=0 and a documented 'empty schema' probability_status.
The ValueError still fires for a NON-empty schema where no field has a
semantics record (a real contract violation).
"""

from __future__ import annotations

import pytest

from jevmlx.engine import finalize_public_result, run_parallel_generation
from jevmlx.schema import StructuredSchema


def _make_state(field_telemetry: dict | None = None):
    """A minimal AssembledState-like object for testing finalize."""
    from types import SimpleNamespace

    return SimpleNamespace(field_telemetry=field_telemetry if field_telemetry is not None else {})


def test_empty_schema_returns_valid_result():
    """An empty schema (zero fields) returns a valid result, not an error.

    This is the typesafe field failure: a case whose schema is {} raised
    'finalize_public_result: no field carries a semantics record'.
    """
    from tests.conftest import FakeModel, _Mod97Tokenizer, make_engine

    engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())
    schema = StructuredSchema({})

    result = run_parallel_generation(engine, "ctx", schema)

    assert result["num_fields"] == 0
    assert result["parsed_json"] == {}
    assert "empty schema" in result["probability_status"]
    assert result["field_telemetry"] == {}


def test_empty_schema_probability_status():
    """finalize_public_result: empty schema + empty field_telemetry ->
    'empty schema' status, not ValueError."""
    from tests.conftest import FakeModel, _Mod97Tokenizer, make_engine

    engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())
    schema = StructuredSchema({})

    result = run_parallel_generation(engine, "ctx", schema)
    assert result["probability_status"] == "empty schema (0 fields; no scoring needed)"


def test_nonempty_schema_with_no_semantics_still_raises():
    """A non-empty schema where no field has a semantics record still raises.

    This is the real contract violation (a bug in the stages) — the empty-schema
    exemption does NOT mask it.
    """
    # We can't easily construct this via run_parallel_generation (every path
    # sets semantics for non-empty schemas). Test the finalizer's guard
    # directly by constructing a state with a field but no 'semantics' key.
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "d"}})
    state = _make_state({"flag": {"value": True}})  # no 'semantics' key

    from unittest.mock import MagicMock

    with pytest.raises((KeyError, ValueError), match="semantics|no field carries"):
        finalize_public_result(
            schema=schema,
            state=state,
            scored=MagicMock(),
            built={"field_plans": {}, "tries": {}},
            second_pass_telemetry={},
            timings={},
            temperature=0.0,
            prior_correction=False,
            prior_ms=0.0,
            constraints=None,
            base_ids=[],
            active_start=0,
            ledger=MagicMock(),
        )


def test_typesafe_empty_schema_case_runs():
    """The actual typesafe case (id typesafe/security_incidents/...t2/n1)
    has an empty schema; the engine handles it without error."""
    from tests.conftest import FakeModel, _Mod97Tokenizer, make_engine

    engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())
    schema = StructuredSchema({})

    # This is the exact shape of the typesafe case: empty schema, empty labels.
    result = run_parallel_generation(engine, "", schema)

    assert result["num_fields"] == 0
    assert result["parsed_json"] == {}
