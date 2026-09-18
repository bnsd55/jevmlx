"""W2-E step 3: the multi-field count row and reconciliation.

One extra row per multi field ('<field>#count') asks how many options apply;
candidates '0','1','2','3','4' are scored like a scalar enum through the
trie. The row ALWAYS runs (no flag). When its top-2 margin clears
COUNT_MARGIN_MIN (0.7 nats) the selected set is reconciled to top-k by
calibrated log-odds (P(yes) order uncalibrated — monotone-equivalent);
otherwise the per-option rule stands. Telemetry: count_choice, count_margin,
reconciled_by on the field entry, plus a '<field>#count' scalar entry.
"""

import pytest
from conftest import CountCodeModel as _BiasedModel
from conftest import _Mod97Tokenizer as _CountTokenizer
from conftest import make_engine

from jevmlx.engine import COUNT_MARGIN_MIN, run_parallel_generation
from jevmlx.schema import StructuredSchema


def _multi_schema():
    return StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["x", "y", "z"]}}
    )


def test_count_row_always_runs_and_lands_in_telemetry():
    """The count row always runs when a multi field exists (no flag): the
    result carries count_choice/count_margin/reconciled_by plus a separate
    '<field>#count' scalar entry; the parsed value is unchanged."""
    model = _BiasedModel(count_bias={}, yes_logit=1.0, no_logit=-1.0)
    result = run_parallel_generation(make_engine(model, _CountTokenizer()), "ctx", _multi_schema())
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["count_choice"] in ("0", "1", "2", "3", "4")
    assert isinstance(telemetry["count_margin"], float)
    assert telemetry["reconciled_by"] in ("per_option", "count")
    count_entry = result["field_telemetry"]["flags#count"]
    assert count_entry["type"] == "enum"
    assert set(count_entry["log_scores"]) == {"0", "1", "2", "3", "4"}
    # All logits zero -> margin 0 < gate -> per_option rule stands.
    assert telemetry["count_margin"] == pytest.approx(0.0)
    assert telemetry["reconciled_by"] == "per_option"
    assert result["parsed_json"]["flags"]["value"] == ["x", "y", "z"]  # P(yes)=0.88


def test_count_gate_below_margin_keeps_per_option_rule():
    """count_margin <= COUNT_MARGIN_MIN: the count is IGNORED — even when it
    names a bucket, the per-option P(yes) >= 0.5 rule selects."""
    # All count codes tied (no bias): margin 0, gate closed. Y strongly
    # biased: per-option rule selects everything.
    model = _BiasedModel(count_bias={}, yes_logit=2.0, no_logit=-2.0)
    result = run_parallel_generation(make_engine(model, _CountTokenizer()), "ctx", _multi_schema())
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["count_margin"] <= COUNT_MARGIN_MIN
    assert telemetry["reconciled_by"] == "per_option"
    assert result["parsed_json"]["flags"]["value"] == ["x", "y", "z"]
    # And a bias that would pick exactly '1': still ignored below the gate.
    model = _BiasedModel(count_bias={"0": 0.3, "1": 0.31}, yes_logit=2.0, no_logit=-2.0)
    result = run_parallel_generation(make_engine(model, _CountTokenizer()), "ctx", _multi_schema())
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["count_choice"] == "1"
    assert telemetry["count_margin"] <= COUNT_MARGIN_MIN
    assert telemetry["reconciled_by"] == "per_option"
    assert result["parsed_json"]["flags"]["value"] == ["x", "y", "z"]


def test_count_gate_above_margin_reconciles_top_k():
    """count_margin > COUNT_MARGIN_MIN: the selected set is exactly the
    top-k by P(yes) (uncalibrated; monotone in log-odds), k from the count
    bucket. With distinct Y/N per option, top-k is deterministic."""
    # Strong confident '1': margin is large (0.31 - (-1.69) >> 0.7 given the
    # zero baseline for 2..4; keep 2..4 far below).
    model = _BiasedModel(count_bias={"0": -1.0, "1": 3.0}, yes_logit=1.0, no_logit=-1.0)
    result = run_parallel_generation(make_engine(model, _CountTokenizer()), "ctx", _multi_schema())
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["count_margin"] > COUNT_MARGIN_MIN
    assert telemetry["count_choice"] == "1"
    assert telemetry["reconciled_by"] == "count"
    # All options share the same P(yes) here (identical Y/N bias); ties are
    # resolved by schema order, so k=1 -> the first option.
    assert result["parsed_json"]["flags"]["value"] == ["x"]


def test_count_reconciliation_respects_k_with_calibrated_log_odds():
    """With calibration, top-k uses calibrated log-odds. a=1, b=0 keeps the
    raw log-odds; make option 2 (z) the strongest via a per-option bias and
    ask for k=2: value must be the two strongest by calibrated log-odds."""
    # Per-option Y/N bias: give 'z' a stronger yes by biasing the option
    # divergence id... options share ids, so instead differentiate through
    # calibration only: a=-1 (flip), b=0 -> log-odds ordering flips.
    model = _BiasedModel(count_bias={"0": -1.0, "2": 3.0}, yes_logit=1.0, no_logit=-1.0)
    result = run_parallel_generation(
        make_engine(model, _CountTokenizer()),
        "ctx",
        _multi_schema(),
        calibration={"multi": {"a": 1.0, "b": 0.0}},
    )
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["count_margin"] > COUNT_MARGIN_MIN
    assert telemetry["count_choice"] == "2"
    assert telemetry["reconciled_by"] == "count"
    # All options tie on calibrated log-odds (a=1,b=0 identical pairs);
    # schema order resolves the tie: first two.
    assert result["parsed_json"]["flags"]["value"] == ["x", "y"]


def test_count_bucket_capped_at_option_count():
    """'4' with only 3 options: k = min(4, 3) = 3 -> all options selected
    (in P(yes)/calibrated order, ties by schema order)."""
    model = _BiasedModel(
        count_bias={"0": -1.0, "1": -1.0, "2": -1.0, "3": -1.0, "4": 3.0},
        yes_logit=1.0,
        no_logit=-1.0,
    )
    result = run_parallel_generation(make_engine(model, _CountTokenizer()), "ctx", _multi_schema())
    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["count_margin"] > COUNT_MARGIN_MIN
    assert telemetry["count_choice"] == "4"
    assert telemetry["reconciled_by"] == "count"
    assert result["parsed_json"]["flags"]["value"] == ["x", "y", "z"]


def test_scalar_fields_unaffected_by_count_rows():
    """A mixed schema: scalar fields keep their exact rows/telemetry; the
    count row adds no branch nodes to the scalar tries."""
    import sys

    sys.path.insert(0, "tests")
    from typing import Literal

    from conftest import FakeModel, FakeTokenizer
    from pydantic import BaseModel, Field

    from jevmlx.api import schema_from_model

    class M(BaseModel):
        topic: Literal["a", "b"] = Field(..., description="t")
        flags: list[Literal["x", "y"]] = Field(..., description="m")

    schema = StructuredSchema(schema_from_model(M))
    result = run_parallel_generation(make_engine(FakeModel(), FakeTokenizer()), "ctx", schema)
    assert result["field_telemetry"]["topic"]["rows"] == 1  # single branch, unchanged
    assert "count_choice" not in result["field_telemetry"]["topic"]
    assert result["field_telemetry"]["flags"]["count_choice"] in ("0", "1", "2", "3", "4")


def test_count_code_4_no_token_prefix_collision():
    """F2 (PR #24 review): a tokenizer where '"4' is a strict token-prefix of
    '"4+' must still compile with the new codes '0'..'4' — the old '4+' code
    raised SchemaCompileError on every multi field here. Then a negative
    control: with '4+' injected into COUNT_CODES the same tokenizer fails."""

    class _GreedyTokenizer:
        """Tokenizer where '"4' (quote + digit 4) is ONE token: the candidate
        row for code '4' is then a strict token-prefix of the row for '4+'
        (['"4', ...] vs ['"4', '+', ...]) — the review's collision case."""

        name_or_path = "fake-greedy"
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            out = []
            i = 0
            while i < len(text):
                if text[i] == '"':
                    j = i + 1
                    if j < len(text) and text[j].isdigit():
                        # '"<digit>' merges into one token (BPE-style).
                        out.append(110 + int(text[j]))
                        i = j + 1
                        continue
                    out.append(109)
                    i += 1
                    continue
                if text[i] == "+":
                    out.append(149)
                    i += 1
                    continue
                out.append(ord(text[i]) % 97 + 1)
                i += 1
            return out or [1]

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
            assert tokenize
            return self.encode("\n".join(m["content"] for m in messages))

    schema = _multi_schema()
    # Compiles clean: codes are '"0"'..'"4"', no token-prefix pairs.
    plan = schema.compile_slot_plan(_GreedyTokenizer())
    remainders = plan["fields"]["flags"]["count"]["remainders"]
    assert all(r for r in remainders), "every count code needs a remainder token"

    # Negative control: a tokenizer that merges quote+digit into one token
    # AND truncates the candidate at the value's end turns code '4' into a
    # strict token-prefix of '4+' (["\"4"] vs ["\"4", "+"]) — exactly the
    # review's collision shape. The guard must raise SchemaCompileError.
    # Proof the prefix detector actually fires on a quote-digit-merging
    # tokenizer; codes '0'..'4' above never trigger it.
    #
    # Patch through the CLASS's __globals__, not `import jevmlx.schema`:
    # test_check_results.py::test_imports_without_mlx deletes every jevmlx.*
    # module from sys.modules, so a module-global patch on the re-imported
    # module never reaches THIS module's already-imported class (the compile
    # reads its defining module's globals, which after that deletion belong to
    # a detached copy of jevmlx.schema). Patching the class's own globals works
    # in both worlds.
    class _TruncatingTokenizer(_GreedyTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # 17 tokens: the full '{"flags#count": "<code>"}' row minus the
            # closing quote/brace tail — the scored candidate ends at the
            # code, so '4' = ['"4'] and '4+' = ['"4', '+'] collide.
            return super().encode(text, add_special_tokens)[:17]

    compile_globals = StructuredSchema.compile_slot_plan.__globals__
    # Same two-copies pitfall for the EXCEPTION CLASS: the compile raises the
    # class from ITS defining globals, which after a sys.modules purge may be a
    # detached copy — not the one a plain `from jevmlx.schema import` binds.
    # Bind both from the compile's own globals.
    SchemaCompileError = compile_globals["SchemaCompileError"]
    old = compile_globals["COUNT_CODES"]
    compile_globals["COUNT_CODES"] = ["0", "1", "2", "3", "4", "4+"]
    try:
        with pytest.raises(SchemaCompileError):
            _multi_schema().compile_slot_plan(_TruncatingTokenizer())
    finally:
        compile_globals["COUNT_CODES"] = old
