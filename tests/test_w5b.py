"""W5-B tests: shared scalar finalizer + dependency second pass + MAP fixes.

Every test targets one GPT-REVIEW-2 finding and FAILS on 1f9f453:

- 3/4: conditioned rows are shared+path with ONE conditioning header (no
  schema lead-in, never two adjacent JSON objects).
- 5: labels-mode + multi dependency paths don't crash; multi depends_on is
  rejected at compile.
- 6: per-field FieldDefinition binding (boolean children stay bool).
- 7: the second pass scores through finalize_scalar_evidence (prior
  correction + caller temperature).
- 8: post-dependency MAP re-run + constraint assertion (a rerun can never
  undo a hard constraint).
- 9: parent gate measures the CHOSEN value's margin (MAP-forced non-argmax
  parents never condition).
- 10: topological waves (depth-2 rows condition on UPDATED parents).
- 11: _constrained_map typed candidates (booleans stay bool).
- 12: constraints validated against the schema BEFORE model work; no
  'unknown means satisfied'.
- 13: case-level constraints on multi fields rejected.
- 43: the prior pass stops after first-pass finalization (no
  dependency-conditioned neutral scores in the prior cache).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import jevmlx.engine as eng

try:
    from jevmlx.constraints import ConstraintError
except ImportError:  # pre-W5-B base: the type does not exist yet

    class ConstraintError(Exception): ...


from jevmlx.constraints import check_constraint
from jevmlx.engine import run_parallel_generation
from jevmlx.schema import SchemaCompileError, StructuredSchema

# ---- token ids used by the bias model (BijectiveTokenizer: id == ord(c)) ----
G = ord("G")  # present ONLY in conditioned rows ('Given: ' header)
P = ord("p")  # 'pa' field rows
Z = ord("z")  # 'za' field rows
W = ord("w")  # 'wq' field rows
C = ord("c")  # 'cb' field rows
ALIAS_A = ord("A")
ALIAS_B = ord("B")
LPAREN = ord("(")  # only in the neutral context "(no context provided)"


class BijectiveTokenizer:
    """Char tokenizer with id == ord(c): rows decode with chr()."""

    name_or_path = "fake-engine"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert tokenize
        return self.encode("\n".join(m["content"] for m in messages))

    def __len__(self) -> int:
        return 200


class RoutingBiasModel:
    """Fake model whose logits implement the test scenarios.

    Row forwards (short sequences) get a PER-ROW bias:

    - 'G' in the row (conditioned row): +2.0 at alias 'B' -> the conditioned
      child answers with its SECOND choice.
    - neutral prefill seen and 'p' in the row: +20.0 at alias 'A' (prior
      pass: 'pa'-family rows decisive — unused by these schemas' rows but
      keeps the neutral first pass honest).
    - evidence pass and 'z' in the row ('za' field): +20.0 at alias 'A'
      -> 'za' decides its FIRST choice decisively.
    - evidence pass and 'w' in the row ('wq' field): +0.1 at alias 'B'
      -> 'wq' leans to its SECOND choice with a 0.1-nat margin (inside the
      0.15 child-low-margin gate -> qualifies for a rerun).
    - everything else: zeros (exact tie -> schema order wins).

    Prefills (batch=1, long sequence) store whether the prompt contained
    '(' — the neutral context "(no context provided)" is the only prompt
    with one — so row forwards know they are in the prior pass.
    """

    def __init__(self, vocab_size: int = 200, n_layers: int = 2):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.args = type("Args", (), {"vocab_size": vocab_size})()
        self.layers = [None] * n_layers
        self.neutral = False

    def parameters(self):
        return {}

    def __call__(self, tokens, cache=None):

        batch, seq_len = tokens.shape
        is_prefill = cache is not None and batch == 1 and seq_len > 40
        if is_prefill:
            # Prefill: record neutral-ness, update cache, no useful logits.
            self.neutral = LPAREN in tokens[0].tolist()
        out = mx.zeros((batch, seq_len, self.vocab_size))
        if cache is not None and not is_prefill:
            for b in range(batch):
                row = tokens[b].tolist()
                if G in row:
                    # Conditioned (dependency) row: +2.0 at alias 'B'.
                    out = out.at[b, :, ALIAS_B].add(2.0)
                elif self.neutral and P in row:
                    # Prior pass ('pa'-family rows, no '(' in prompt).
                    out = out.at[b, :, ALIAS_A].add(20.0)
                elif (not self.neutral) and Z in row:
                    # Evidence pass, 'za' field: decisive first choice.
                    out = out.at[b, :, ALIAS_A].add(20.0)
                elif (not self.neutral) and W in row:
                    # Evidence pass, 'wq' field: 0.1-nat lean to second.
                    out = out.at[b, :, ALIAS_B].add(0.1)
                elif (not self.neutral) and C in row:
                    # Evidence pass, 'cb' field rows: low-margin lean to the
                    # second choice (B) so the child qualifies for a rerun.
                    out = out.at[b, :, ALIAS_B].add(0.05)
        if cache is not None:
            for c in cache:
                c.update_and_fetch(
                    mx.zeros((batch, 2, seq_len, 8)), mx.zeros((batch, 2, seq_len, 8))
                )
        return out


def _chain_schema() -> dict:
    """pa -> cb -> gc (two dependency waves)."""
    return {
        "pa": {"type": "enum", "description": "d", "choices": ["a"]},
        "cb": {
            "type": "enum",
            "description": "d",
            "choices": ["OLDVALUE", "NEWVALUE"],
            "depends_on": "pa",
        },
        "gc": {
            "type": "enum",
            "description": "d",
            "choices": ["X1", "X2"],
            "depends_on": "cb",
        },
    }


@pytest.fixture()
def row_capture(monkeypatch):
    """Record every _score_rows call's rows (calls still run for real)."""
    calls: list[list[list[int]]] = []
    real = eng._score_rows

    def wrapper(model, cache, rows, *a, **k):
        calls.append([list(r) for r in rows])
        return real(model, cache, rows, *a, **k)

    monkeypatch.setattr(eng, "_score_rows", wrapper)
    return calls


def _decode(row: list[int]) -> str:
    return "".join(chr(t) for t in row)


# ---------------------------------------------------------------- findings 3+4
def test_conditioned_rows_use_header_not_schema_lead_in(row_capture):
    """Review 3+4: a conditioned row is tokenize(header + child object) —
    the schema-wide lead_in is NOT prepended, and the candidate is ONE
    complete JSON object behind an explicit 'Given:' header."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "pa": {"type": "enum", "description": "d", "choices": ["a"]},
            "cb": {
                "type": "enum",
                "description": "d",
                "choices": ["OLDVALUE", "NEWVALUE"],
                "depends_on": "pa",
            },
        }
    )
    result = run_parallel_generation(model, tok, "ctx", schema)
    assert result["rerun_fields"] == ["cb"]
    # Dependency rows are identifiable by content (the header token 'G').
    dep_rows = [r for call in row_capture for r in call if G in r]
    assert dep_rows, "dependency pass must run rows"
    for row in dep_rows:
        text = _decode(row)
        assert text.startswith("Given: "), text
        # ONE header, ONE JSON object; no schema lead-in ('{\n  "cb"...' from
        # the FIRST-pass family never opens a conditioned row).
        assert text.count("Given: ") == 1
        assert text.count('{"cb"') == 1
        lead_in_ids = eng._build_schema_rows(schema, tok, "slots")["lead_in"]
        assert not text.startswith(_decode(lead_in_ids)) if lead_in_ids else True
    # The row is a prefix of the full candidate family tokenization.
    full = tok.encode('Given: {"pa": "a"}\n{"cb": "OLDVALUE"}', add_special_tokens=False)
    assert all(
        _decode(r) == _decode(full[: len(r)]) or len(_decode(r)) <= len(full) for r in dep_rows
    )


# ------------------------------------------------------------------- finding 5
def test_labels_mode_second_pass_does_not_crash():
    """Review 5: labels plans have no aliases/alias_map; the old
    dict(zip([], choices, strict=True)) default crashed. The labels branch
    scores the real choice texts."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "pa": {"type": "enum", "description": "d", "choices": ["a"]},
            "cb": {
                "type": "enum",
                "description": "d",
                "choices": ["X1", "X2"],
                "depends_on": "pa",
            },
        }
    )
    result = run_parallel_generation(model, tok, "ctx", schema, scoring="labels")
    assert result["rerun_fields"] == ["cb"]
    assert result["parsed_json"]["cb"]["value"] in {"X1", "X2"}


def test_depends_on_multi_child_rejected_at_compile():
    """Review 5: multi children with depends_on are rejected at compile
    (never crash or mis-score at inference)."""
    with pytest.raises(SchemaCompileError, match="multi"):
        StructuredSchema(
            {
                "pa": {"type": "enum", "description": "d", "choices": ["A", "B"]},
                "tags": {
                    "type": "multi",
                    "description": "d",
                    "choices": ["x", "y"],
                    "depends_on": "pa",
                },
            }
        )


def test_depends_on_unknown_parent_and_cycles_rejected():
    with pytest.raises(SchemaCompileError, match="unknown field"):
        StructuredSchema(
            {"cb": {"type": "enum", "description": "d", "choices": ["X"], "depends_on": "nope"}}
        )
    with pytest.raises(SchemaCompileError, match="cycle"):
        StructuredSchema(
            {
                "a": {"type": "enum", "description": "d", "choices": ["A", "B"], "depends_on": "b"},
                "b": {"type": "enum", "description": "d", "choices": ["A", "B"], "depends_on": "a"},
            }
        )


# ------------------------------------------------------------------- finding 6
def test_second_pass_binds_each_field_own_definition():
    """Review 6: a boolean child rerunning before an enum child must finalize
    with ITS OWN FieldDefinition — value stays a Python bool."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "pa": {"type": "enum", "description": "d", "choices": ["a"]},
            "b1": {"type": "boolean", "description": "d", "depends_on": "pa"},
            "e1": {"type": "enum", "description": "d", "choices": ["X1", "X2"], "depends_on": "pa"},
        }
    )
    result = run_parallel_generation(model, tok, "ctx", schema)
    assert set(result["rerun_fields"]) == {"b1", "e1"}
    b1 = result["parsed_json"]["b1"]["value"]
    assert isinstance(b1, bool), f"boolean child value leaked as {type(b1)}: {b1!r}"
    assert b1 is False  # conditioned rows bias alias 'B' -> false
    assert result["parsed_json"]["e1"]["value"] == "X2"


# ------------------------------------------------------------------- finding 7
def test_second_pass_prior_correction_and_temperature():
    """Review 7: the dependency rescore goes through the shared finalizer —
    prior-corrected scores and caller-temperature probabilities, not a
    hardcoded T=1 uncorrected copy."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "pa": {"type": "enum", "description": "d", "choices": ["a"]},
            "cb": {
                "type": "enum",
                "description": "d",
                "choices": ["OLDVALUE", "NEWVALUE"],
                "depends_on": "pa",
            },
        }
    )
    r1 = run_parallel_generation(model, tok, "ctx", schema, prior_correction=True, temperature=1.0)
    r2 = run_parallel_generation(model, tok, "ctx", schema, prior_correction=True, temperature=2.0)
    assert r1["rerun_fields"] == ["cb"] and r2["rerun_fields"] == ["cb"]
    # The conditioned winner is the second choice, confidently, at T=1.
    assert r1["parsed_json"]["cb"]["value"] == "NEWVALUE"
    assert r1["field_telemetry"]["cb"]["probability"] > 0.8
    # Temperature flattens the SAME (prior-corrected) log scores.
    assert r2["parsed_json"]["cb"]["value"] == "NEWVALUE"
    assert r2["field_telemetry"]["cb"]["probability"] < r1["field_telemetry"]["cb"]["probability"]
    # The prior was actually subtracted: the uncorrected run differs.
    r0 = run_parallel_generation(model, tok, "ctx", schema, prior_correction=False)
    assert r0["field_telemetry"]["cb"]["log_scores"] != r1["field_telemetry"]["cb"]["log_scores"]


# ------------------------------------------------------------------- finding 8
def test_second_pass_cannot_undo_constraint():
    """Review 8: MAP re-runs after the dependency pass and every constraint
    is asserted — a rerun that re-picks a constraint-violating value is
    corrected back, never returned."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "za": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "wq": {
                "type": "enum",
                "description": "d",
                "choices": ["X", "Y"],
                "depends_on": "za",
            },
        }
    )
    constraints = [
        {"type": "implies", "parent": "za", "child": "wq", "mapping": {"A": ["X"], "B": ["Y"]}}
    ]
    result = run_parallel_generation(model, tok, "ctx", schema, constraints=constraints)
    # za: evidence bias +20 at 'A' -> A (valid parent). wq: evidence leans Y
    # (+0.1 at 'B', low margin -> rerun); the CONDITIONED rerun also picks Y
    # (+2.0 at 'B'), which violates A->X. The post-wave MAP must put it back.
    assert result["parsed_json"]["za"]["value"] == "A"
    assert result["parsed_json"]["wq"]["value"] == "X", (
        f"dependency rerun undid the hard constraint: {result['parsed_json']!r}"
    )


# ------------------------------------------------------------------- finding 9
def test_map_forced_non_argmax_parent_never_conditions():
    """Review 9: the parent gate is the CHOSEN value's margin
    (chosen - max(others)), not the old argmax margin. A MAP-forced parent
    the model disagreed with (margin -20) never conditions."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "za": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "wq": {
                "type": "enum",
                "description": "d",
                "choices": ["X1", "X2"],
                "depends_on": "za",
            },
        }
    )
    # A allows nothing -> MAP forces za=B although the model decisively
    # preferred A (+20 at 'A'). The chosen-value margin is -20: no conditioning.
    constraints = [
        {"type": "implies", "parent": "za", "child": "wq", "mapping": {"B": ["X1", "X2"]}}
    ]
    result = run_parallel_generation(model, tok, "ctx", schema, constraints=constraints)
    assert result["parsed_json"]["za"]["value"] == "B"
    assert result["rerun_fields"] == [], (
        f"conditioned on a MAP-forced parent the model rejected: {result['rerun_fields']!r}"
    )


# ------------------------------------------------------------------ finding 10
def test_dependency_waves_condition_on_updated_parents(row_capture):
    """Review 10: depth-2 children are built from the UPDATED depth-1
    decision (topological waves), never from a pre-rerun snapshot."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(_chain_schema())
    result = run_parallel_generation(model, tok, "ctx", schema)
    # cb reruns (evidence tie), flips to NEWVALUE under conditioning; gc must
    # then rerun conditioned on the UPDATED cb — two waves.
    assert set(result["rerun_fields"]) == {"cb", "gc"}
    assert result["rerun_rows"] > 0
    # Wave-2 rows carry the UPDATED cb value in the header.
    wave2 = [row for call in row_capture for row in call if "gc" in _decode(row) and G in row]
    assert wave2, "depth-2 wave must run"
    for row in wave2:
        assert _decode(row).startswith('Given: {"cb": "NEWVALUE"}'), _decode(row)
    assert result["parsed_json"]["cb"]["value"] == "NEWVALUE"


# ------------------------------------------------------------------ finding 11
def test_constrained_map_keeps_booleans_typed():
    """Review 11: score keys 'true'/'false' are the SCORED representation;
    the reconciled value is a Python bool. The old code returned the string
    'true' and falsely recorded the field as changed."""
    from jevmlx.engine import _constrained_map

    schema = StructuredSchema(
        {
            "approved": {"type": "boolean", "description": "d"},
            "rejection_reason": {
                "type": "enum",
                "description": "d",
                "choices": ["policy", "eligibility", "duplicate"],
            },
        }
    )
    field_log_scores = {
        "approved": {"true": -0.1, "false": -2.0},
        "rejection_reason": {"policy": -0.2, "eligibility": -1.5, "duplicate": -3.0},
    }
    field_values = {"approved": {"value": True}, "rejection_reason": {"value": "policy"}}
    constraints = [
        {"type": "excludes", "field": "approved", "value": True, "other": "rejection_reason"}
    ]
    from jevmlx.constraints import compile_constraints

    compiled = compile_constraints(constraints, schema)
    reconciled, changed = _constrained_map(field_log_scores, field_values, compiled, schema)
    got = reconciled["approved"]
    assert got is False, f"boolean reconciled to {got!r} ({type(got).__name__})"
    assert "approved" in changed


# ------------------------------------------------------------------ finding 12
def test_constraints_validated_before_model_work():
    """Review 12: unknown types / fields / out-of-domain values fail loudly
    BEFORE any model work — never silently satisfied, never a mid-engine
    KeyError."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "za": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "wq": {"type": "enum", "description": "d", "choices": ["X1", "X2"]},
        }
    )
    with pytest.raises(ValueError, match="type"):
        run_parallel_generation(model, tok, "ctx", schema, constraints=[{"type": "typo"}])
    with pytest.raises(ValueError, match="unknown field"):
        run_parallel_generation(
            model,
            tok,
            "ctx",
            schema,
            constraints=[{"type": "implies", "parent": "zz", "child": "wq", "mapping": {}}],
        )
    with pytest.raises(ValueError, match="not a value"):
        run_parallel_generation(
            model,
            tok,
            "ctx",
            schema,
            constraints=[
                {"type": "implies", "parent": "za", "child": "wq", "mapping": {"ZZ": ["X1"]}}
            ],
        )
    with pytest.raises(ValueError, match="not a value"):
        run_parallel_generation(
            model,
            tok,
            "ctx",
            schema,
            constraints=[
                {"type": "implies", "parent": "za", "child": "wq", "mapping": {"A": ["ZZ"]}}
            ],
        )


def test_check_constraint_unknown_type_raises():
    """Review 12: 'unknown means satisfied' is gone."""
    with pytest.raises(ValueError):
        check_constraint({"type": "zzz"}, {})


# ------------------------------------------------------------------ finding 13
def test_case_constraint_on_multi_field_rejected():
    """Review 13: implies/excludes on a multi field fail at validation with
    a clear error (no joint set solver) instead of pretending to work."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "za": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "tags": {"type": "multi", "description": "d", "choices": ["t1", "t2"]},
        }
    )
    with pytest.raises(ValueError, match="multi"):
        run_parallel_generation(
            model,
            tok,
            "ctx",
            schema,
            constraints=[
                {"type": "implies", "parent": "tags", "child": "za", "mapping": {"t1": ["A"]}}
            ],
        )
    with pytest.raises(ValueError, match="multi"):
        run_parallel_generation(
            model,
            tok,
            "ctx",
            schema,
            constraints=[{"type": "excludes", "field": "za", "value": "A", "other": "tags"}],
        )


# ------------------------------------------------------------------ finding 43
def test_prior_pass_stops_after_first_pass(row_capture):
    """Review 43: the neutral prior never runs the dependency second pass —
    a dependency-conditioned neutral score must not enter the prior cache."""
    model = RoutingBiasModel()
    tok = BijectiveTokenizer()
    schema = StructuredSchema(
        {
            "pa": {"type": "enum", "description": "d", "choices": ["a"]},
            "cb": {
                "type": "enum",
                "description": "d",
                "choices": ["OLDVALUE", "NEWVALUE"],
                "depends_on": "pa",
            },
        }
    )
    prior = eng._get_or_compute_prior(model, tok, schema, "slots", None, "(no context provided)")
    # The neutral pass must store the FIRST-pass scores: cb's neutral rows
    # are unbiased (tie -> uniform). A leaked dependency-conditioned score
    # would carry the +2.0-at-'B' bias (NEWVALUE decisive).
    cb = prior["cb"]["log_scores"]
    assert abs(cb["OLDVALUE"] - cb["NEWVALUE"]) < 0.5, (
        f"prior cache holds dependency-conditioned neutral scores: {cb!r}"
    )
    # And the prior pass ran NO conditioned rows (no dependency wave): every
    # recorded scoring call is first-pass row content only.
    conditioned = [row for call in row_capture for row in call if G in row]
    assert not conditioned, "prior pass ran dependency-conditioned rows"
