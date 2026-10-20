"""W2-E step 1: multi decision-row codes.

The multi row key changes from '<field>/<option>' to '<field>/<code>' where
code is the option's zero-padded 2-digit index in choices order (00, 01, ...).
The schema block maps code = option; telemetry maps codes back to option
names; PROMPT_VERSION bumps to v4.
"""

import json

from jevmlx.schema import StructuredSchema
from tests.conftest import make_test_renderer


class CharTokenizer:
    """Deterministic char tokenizer (ids start at 1)."""

    name_or_path = "fake-w2e"
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 97 + 1 for c in text] or [1]


def _decode(ids: list[int]) -> str:
    inv = {}
    for c in map(chr, range(32, 127)):
        inv[ord(c) % 97 + 1] = c
    return "".join(inv.get(t, f"<{t}>") for t in ids)


def _multi_schema():
    return StructuredSchema(
        {
            "tags": {
                "type": "multi",
                "description": "Tags to apply",
                "choices": ["alpha", "beta", "gamma"],
            }
        }
    )


def test_row_key_uses_zero_padded_codes():
    """The decision row key is '<field>/<code>' (00, 01, 02...), never the
    raw option text: garbage/hostile option strings cannot reach the scored
    token stream (bug 18)."""
    tok = CharTokenizer()
    for mode in ("slots", "labels"):
        schema = _multi_schema()
        plan = (
            schema.compile_slot_plan(tok, make_test_renderer(tok, schema, "slots"))
            if mode == "slots"
            else schema.compile_labels_plan(tok, make_test_renderer(tok, schema, "labels"))
        )
        p = plan["fields"]["tags"]
        assert list(p["codes"]) == ["00", "01", "02"]
        assert list(p["options"]) == ["alpha", "beta", "gamma"]
        for i, suffix in enumerate(p["suffix_ids_list"]):
            row = _decode(list(suffix))
            assert f'"tags/{i:02d}"' in row, (mode, i, row)
            assert "alpha" not in row and "beta" not in row and "gamma" not in row


def test_exact_token_reconstruction_with_codes():
    """Token-exact: lead_in + suffix + remainder reconstructs the full
    candidate text '{\\n  "<field>/<code>": "Y"|"N",\\n' for every option in
    BOTH modes (the row text the engine would emit is exactly what was
    compiled)."""
    tok = CharTokenizer()
    for mode in ("slots", "labels"):
        schema = _multi_schema()
        plan = (
            schema.compile_slot_plan(tok, make_test_renderer(tok, schema, "slots"))
            if mode == "slots"
            else schema.compile_labels_plan(tok, make_test_renderer(tok, schema, "labels"))
        )
        p = plan["fields"]["tags"]
        for i, (suffix, remainders) in enumerate(
            zip(p["suffix_ids_list"], p["remainders"], strict=True)
        ):
            code = p["codes"][i]
            for alias, remainder in zip(("Y", "N"), remainders, strict=True):
                full_text = "{" + f'{json.dumps(f"tags/{code}")}: "{alias}"' + "}"
                full_ids = tok.encode(full_text, add_special_tokens=False)
                recon = list(suffix) + list(remainder)
                assert recon == full_ids, (mode, code, alias)


def test_codes_injective_across_options_and_schema_order():
    """Codes are zero-padded, choices-ordered, and unique per option; beyond
    9 options they stay 2-digit (10, 11, ...)."""
    tok = CharTokenizer()
    schema = StructuredSchema(
        {
            "m": {
                "type": "multi",
                "description": "d",
                "choices": [f"c{i}" for i in range(12)],
            }
        }
    )
    plan = schema.compile_labels_plan(tok, make_test_renderer(tok, schema, "labels"))
    assert list(plan["fields"]["m"]["codes"]) == [f"{i:02d}" for i in range(12)]


def test_engine_telemetry_maps_codes_to_option_names():
    """Fake-model end-to-end: telemetry keys (per_option, option_logit_pairs,
    alternatives, top_choices) are OPTION NAMES in choices order, not codes —
    the engine maps '<field>/<code>' rows back through plan['codes']."""
    from conftest import FakeModel, FakeTokenizer, make_engine

    from jevmlx.engine import run_parallel_generation

    schema = _multi_schema()
    result = run_parallel_generation(make_engine(FakeModel(), FakeTokenizer()), "ctx", schema)
    telemetry = result["field_telemetry"]["tags"]
    assert set(telemetry["per_option"]) == {"alpha", "beta", "gamma"}
    assert set(telemetry["option_logit_pairs"]) == {"alpha", "beta", "gamma"}
    assert [c for c, _ in telemetry["alternatives"]] == ["alpha", "beta", "gamma"]
    assert [e["choice"] for e in telemetry["top_choices"]] == ["alpha", "beta", "gamma"]
    assert result["parsed_json"]["tags"]["value"] == ["alpha", "beta", "gamma"]
    assert result["prompt_version"] == "jevmlx-parallel-v10"
