"""W2-C tests: tokenizer-specific codebook search (review Q3 + bug 7).

Verifies:
- A fake tokenizer where ALL letters collide but digits don't -> picks digits.
- single_branch detection.
- Determinism: same schema + tokenizer -> same codes.
- SchemaCompileError when no valid set exists (F2: no silent fallback).
- Telemetry: codebook and single_branch land in the plan.
- Codes differ from index fallback when searched.
- F1 perf: a 26-choice field compiles in under 1 second (greedy, not O(C(n,k))).
"""

from __future__ import annotations

import json
import time

import pytest

from jevmlx.schema import SchemaCompileError, StructuredSchema, _alias_code, _search_codebook
from tests.conftest import make_test_renderer


class CollidingLetterTokenizer:
    """ALL letters tokenize to the same token (collision); digits 0 and 1 are
    distinct. The codebook search must reject every letter set and pick
    digits instead."""

    name_or_path = "fake-colliding-letters"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for c in text:
            if c.isalpha():
                out.append(1)  # collision: ALL letters map to token 1
            elif c == "0":
                out.append(10)
            elif c == "1":
                out.append(11)
            else:
                out.append(ord(c))
        return out

    def __len__(self) -> int:
        return 128


class DistinctTokenizer:
    """Every character is its own token — no collisions."""

    name_or_path = "fake-distinct"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def __len__(self) -> int:
        return 128


class SingleBranchTokenizer:
    """Makes A and B share the first token, then diverge: 'A' -> [5, 6],
    'B' -> [5, 7]. The complete candidate rows share a long prefix and
    diverge at the alias position -> single branch node."""

    name_or_path = "fake-single-branch"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for c in text:
            if c == "A":
                out.extend([5, 6])
            elif c == "B":
                out.extend([5, 7])
            else:
                out.append(ord(c))
        return out

    def __len__(self) -> int:
        return 128


def _candidate_text(name: str, alias: str) -> str:
    return json.dumps({name: alias}, ensure_ascii=False)


def test_letters_collide_digits_dont():
    """When ALL letters tokenize identically, the search rejects the letter
    set and picks digits (0, 1) instead."""
    tok = CollidingLetterTokenizer()
    codes, _single_branch = _search_codebook(
        tok, lambda alias: _candidate_text("risk", alias), 2, field_name="risk"
    )
    assert codes != ["A", "B"], f"colliding letter set was not rejected: {codes}"
    assert codes == ["0", "1"], f"expected digits, got {codes}"


def test_single_branch_detection():
    """When all candidates share their first token and diverge once, the trie
    has exactly one branch node -> single_branch=True."""
    tok = SingleBranchTokenizer()
    codes, single_branch = _search_codebook(
        tok, lambda alias: _candidate_text("risk", alias), 2, field_name="risk"
    )
    assert single_branch, f"expected single_branch=True, got codes={codes}"


def test_determinism_same_codes():
    """Same schema + tokenizer -> same codes."""
    tok = DistinctTokenizer()
    schema = StructuredSchema(
        {"risk": {"type": "enum", "description": "d", "choices": ["LOW", "MEDIUM", "HIGH"]}}
    )
    plan1 = schema.compile_slot_plan(tok, make_test_renderer(tok, schema, "slots"))
    plan2 = schema.compile_slot_plan(tok, make_test_renderer(tok, schema, "slots"))
    codes1 = plan1["fields"]["risk"]["codebook"]
    codes2 = plan2["fields"]["risk"]["codebook"]
    assert codes1 == codes2, f"non-deterministic codes: {codes1} vs {codes2}"
    assert plan1 is plan2  # cached


def test_no_valid_set_raises():
    """F2: when no pool set validates, raise SchemaCompileError (no fallback)."""

    class AllCollidingTokenizer:
        name_or_path = "fake-all-colliding"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # Every character (including digits) maps to the same token.
            return [1] * len(text)

        def __len__(self) -> int:
            return 128

    tok = AllCollidingTokenizer()
    with pytest.raises(SchemaCompileError, match="no codebook set"):
        _search_codebook(tok, lambda alias: _candidate_text("risk", alias), 2, field_name="risk")


def test_telemetry_in_plan():
    """The plan carries codebook and single_branch per field (no
    codebook_searched — F2 removed it)."""
    tok = DistinctTokenizer()
    schema = StructuredSchema(
        {"risk": {"type": "enum", "description": "d", "choices": ["LOW", "HIGH"]}}
    )
    plan = schema.compile_slot_plan(tok, make_test_renderer(tok, schema, "slots"))
    fp = plan["fields"]["risk"]
    assert "codebook" in fp
    assert "single_branch" in fp
    assert "codebook_searched" not in fp  # F2 removed
    assert isinstance(fp["codebook"], (list, tuple))  # frozen plan: tuple
    assert isinstance(fp["single_branch"], bool)
    assert len(fp["codebook"]) == 2


def test_codes_not_index_derived_when_searched():
    """When the search validates, codes are tokenizer-specific — they may
    differ from the index-derived A, B, C..."""
    tok = CollidingLetterTokenizer()
    schema = StructuredSchema(
        {"risk": {"type": "enum", "description": "d", "choices": ["LOW", "HIGH"]}}
    )
    plan = schema.compile_slot_plan(tok, make_test_renderer(tok, schema, "slots"))
    codes = plan["fields"]["risk"]["codebook"]
    index_codes = [_alias_code(i) for i in range(2)]
    assert codes != index_codes, f"codes match index fallback: {codes}"


def test_26_choice_field_compiles_under_1_second():
    """F1 perf: greedy search is O(pool), not O(C(n,k)). A 26-choice field
    must compile in under 1 second with the fake tokenizer."""
    tok = DistinctTokenizer()
    choices = [f"choice_{i}" for i in range(26)]
    schema = StructuredSchema({"big": {"type": "enum", "description": "d", "choices": choices}})
    start = time.monotonic()
    plan = schema.compile_slot_plan(tok, make_test_renderer(tok, schema, "slots"))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"26-choice compile took {elapsed:.2f}s (greedy should be <1s)"
    assert len(plan["fields"]["big"]["codebook"]) == 26
