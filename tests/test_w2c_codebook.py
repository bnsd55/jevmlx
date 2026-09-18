"""W2-C tests: tokenizer-specific codebook search (review Q3 + bug 7).

Verifies:
- A fake tokenizer where letters collide (A, B tokenize to the same token)
  but digits don't (0, 1 are distinct) -> the search picks digits over letters.
- single_branch detection: a code set whose trie has exactly one branch node.
- Determinism: same schema + tokenizer -> same codes.
- codebook_searched=False when the fallback (index-derived A..) is used.
- Telemetry: codebook, codebook_searched, single_branch land in the plan.
"""

from __future__ import annotations

import json

from jevmlx.schema import StructuredSchema, _alias_code, _search_codebook


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
    """When A and B tokenize identically, the search rejects the letter set
    and picks digits (0, 1) instead."""
    tok = CollidingLetterTokenizer()
    codes, _single_branch, searched = _search_codebook(
        tok, lambda alias: _candidate_text("risk", alias), 2
    )
    assert searched, "codebook search should have validated a set"
    assert codes != ["A", "B"], f"colliding letter set was not rejected: {codes}"
    assert codes == ["0", "1"], f"expected digits, got {codes}"


def test_single_branch_detection():
    """When all candidates share their first token and diverge once, the trie
    has exactly one branch node -> single_branch=True."""
    tok = SingleBranchTokenizer()
    codes, single_branch, searched = _search_codebook(
        tok, lambda alias: _candidate_text("risk", alias), 2
    )
    assert searched
    assert single_branch, f"expected single_branch=True, got codes={codes}"


def test_determinism_same_codes():
    """Same schema + tokenizer -> same codes."""
    tok = DistinctTokenizer()
    schema = StructuredSchema(
        {"risk": {"type": "enum", "description": "d", "choices": ["LOW", "MEDIUM", "HIGH"]}}
    )
    plan1 = schema.compile_slot_plan(tok)
    plan2 = schema.compile_slot_plan(tok)
    codes1 = plan1["fields"]["risk"]["codebook"]
    codes2 = plan2["fields"]["risk"]["codebook"]
    assert codes1 == codes2, f"non-deterministic codes: {codes1} vs {codes2}"
    assert plan1 is plan2  # cached


def test_codebook_searched_false_on_fallback():
    """When no pool set validates, the fallback (A..) is used and
    codebook_searched=False."""

    class AllCollidingTokenizer:
        name_or_path = "fake-all-colliding"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # All chars map to token 1 EXCEPT the structural chars that
            # differ — so candidates DO differ (different lengths) but every
            # alias char collides. The search cannot find a valid 2-code set
            # from single-char codes because A==B and 0==1 under this tok.
            return [1 if c.isalnum() else ord(c) for c in text]

        def __len__(self) -> int:
            return 128

    tok = AllCollidingTokenizer()
    codes, _single_branch, searched = _search_codebook(
        tok, lambda alias: _candidate_text("risk", alias), 2
    )
    assert not searched, "should have fallen back to index-derived codes"
    assert codes == ["A", "B"]  # fallback


def test_telemetry_in_plan():
    """The plan carries codebook, codebook_searched, and single_branch per
    field."""
    tok = DistinctTokenizer()
    schema = StructuredSchema(
        {"risk": {"type": "enum", "description": "d", "choices": ["LOW", "HIGH"]}}
    )
    plan = schema.compile_slot_plan(tok)
    fp = plan["fields"]["risk"]
    assert "codebook" in fp
    assert "codebook_searched" in fp
    assert "single_branch" in fp
    assert isinstance(fp["codebook"], list)
    assert isinstance(fp["codebook_searched"], bool)
    assert isinstance(fp["single_branch"], bool)
    assert len(fp["codebook"]) == 2


def test_codes_not_index_derived_when_searched():
    """When the search validates, codes are tokenizer-specific — they may
    differ from the index-derived A, B, C..."""
    tok = CollidingLetterTokenizer()
    schema = StructuredSchema(
        {"risk": {"type": "enum", "description": "d", "choices": ["LOW", "HIGH"]}}
    )
    plan = schema.compile_slot_plan(tok)
    codes = plan["fields"]["risk"]["codebook"]
    index_codes = [_alias_code(i) for i in range(2)]
    assert codes != index_codes, f"codes match index fallback: {codes}"
