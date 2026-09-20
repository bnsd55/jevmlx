"""Tests for benchmarks.perturb: deterministic, label-preserving variants."""

from __future__ import annotations

import json

from benchmarks.perturb import _format_numbers, _norm_whitespace, main, perturb_case, perturb_cases


def _case(**overrides) -> dict:
    case = {
        "id": "q/case-1",
        "group_id": "q/case-1",
        "source": "quality-eval",
        "workflow": None,
        "benchmark_only": False,
        "schema": {"action": {"type": "enum", "description": "a", "choices": ["A", "B"]}},
        "context": "Block  one.\n\nBlock two with 1,850.00 USD.  \n\n Block three.",
        "labels": {"action": "A"},
        "split": "train",
        "meta": {},
    }
    case.update(overrides)
    return case


def test_ws_collapses_whitespace_runs():
    assert _norm_whitespace("a   b\t\tc\n   \n\nd  \n") == "a b c\n\nd"


def test_numfmt_rewrites_thousands_and_trailing_zero_zero():
    assert _format_numbers("1,850.00 USD and 2,350 items and 3.5 kept") == (
        "1850 USD and 2350 items and 3.5 kept"
    )


def test_variants_carry_id_group_and_kind():
    variants = perturb_case(_case(), variants=3, seed=0)
    assert len(variants) == 3
    for k, variant in enumerate(variants, start=1):
        assert variant["id"] == "q/case-1#p" + str(k)
        assert variant["group_id"] == "q/case-1"
        assert variant["meta"]["perturbation"] in {"ws", "preamble", "numfmt", "shuffle"}
    kinds = [v["meta"]["perturbation"] for v in variants]
    # Fixed kind order for a context where all three deterministic kinds apply.
    assert kinds == ["ws", "preamble", "numfmt"]
    # Labels and schema are untouched — perturbations must not change labels.
    for variant in variants:
        assert variant["labels"] == {"action": "A"}
        assert variant["schema"] == _case()["schema"]


def test_preamble_and_ws_change_context_as_documented():
    variants = perturb_case(_case(), variants=3, seed=0)
    ws, preamble, numfmt = variants
    # ws collapses the double space, trailing spaces, and leading space.
    assert ws["context"] == "Block one.\n\nBlock two with 1,850.00 USD.\n\nBlock three."
    # preamble is verbatim + "\n\n"; everything else untouched.
    assert preamble["context"] == (
        "Record follows.\n\nBlock  one.\n\nBlock two with 1,850.00 USD.  \n\n Block three."
    )
    assert numfmt["context"] == "Block  one.\n\nBlock two with 1850 USD.  \n\n Block three."


def test_noop_kinds_are_skipped():
    """A single-block, already-normalised context skips ws/numfmt/shuffle.

    ws and numfmt are no-ops there and shuffle is structurally impossible
    (one block); preamble fires, plus the schema kinds (optrev/criterion)
    which are context-independent. Ids stay contiguous from p1 despite the
    skipped kinds.
    """
    case = _case(context="Block one.")
    variants = perturb_case(case, variants=3, seed=0)
    kinds = [v["meta"]["perturbation"] for v in variants]
    assert "preamble" in kinds
    assert "ws" not in kinds and "numfmt" not in kinds and "shuffle" not in kinds
    assert [v["id"] for v in variants] == [f"q/case-1#p{k}" for k in range(1, len(variants) + 1)]
    pre = next(v for v in variants if v["meta"]["perturbation"] == "preamble")
    assert pre["context"] == "Record follows.\n\nBlock one."


def test_shuffle_is_deterministic_and_preserves_blocks():
    case = _case(context="Block one.\n\nBlock two.\n\nBlock three.\n\nBlock four.")
    # variants=6 so the shuffle kind (after ws/preamble/numfmt/optrev/criterion)
    # is reached; the schema kinds are context-independent and always fire.
    run_1 = perturb_case(case, variants=6, seed=7)
    run_2 = perturb_case(case, variants=6, seed=7)
    assert run_1 == run_2  # same seed -> byte-identical plan

    shuffles = [v for v in run_1 if v["meta"]["perturbation"] == "shuffle"]
    assert shuffles, "a 4-block context must produce a shuffle variant"

    def blocks_of(ctx):
        return sorted(ctx.split("\n\n"))

    original_blocks = blocks_of(case["context"])
    for variant in shuffles:
        assert blocks_of(variant["context"]) == original_blocks  # same blocks, order permuted
    assert shuffles[0]["context"] != case["context"]  # actually reshuffled for seed 7


def test_originals_are_kept_and_unchanged():
    cases = [_case(id="a"), _case(id="b")]
    out = perturb_cases(cases, variants=2, seed=0)
    assert out[:2] == cases  # originals first, byte-equal
    assert len(out) == 2 + sum(len(perturb_case(c, 2, 0)) for c in cases)
    # Variants reference their original via group_id; originals keep their own.
    variant_groups = {v["group_id"] for v in out[2:]}
    assert variant_groups <= {"a", "b"}


def test_cli_end_to_end_is_deterministic(tmp_path, capsys):
    src = tmp_path / "cases.jsonl"
    with src.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_case(id="x/1")) + "\n")
        f.write(json.dumps(_case(id="x/2")) + "\n")

    out_1 = tmp_path / "p1.jsonl"
    out_2 = tmp_path / "p2.jsonl"
    assert main(["--in", str(src), "--out", str(out_1), "--variants", "3", "--seed", "0"]) == 0
    assert main(["--in", str(src), "--out", str(out_2), "--variants", "3", "--seed", "0"]) == 0
    assert out_1.read_bytes() == out_2.read_bytes()  # fully deterministic

    lines = [json.loads(line) for line in out_1.read_text(encoding="utf-8").splitlines()]
    originals = [line for line in lines if "#" not in line["id"]]
    variants = [line for line in lines if "#p" in line["id"]]
    assert len(originals) == 2 and len(variants) == 6
    # An original's group_id passes through untouched; a variant's group_id
    # points at its original (overridden in perturb_case).
    assert {line["group_id"] for line in originals} == {"q/case-1"}
    assert all(line["id"].startswith(line["group_id"] + "#p") for line in variants)
    printed = capsys.readouterr().out
    assert "cases: 2" in printed and "variants: 6" in printed
