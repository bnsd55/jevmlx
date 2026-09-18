"""Fast tests for token-aligned batch plans and trie scoring (no model).

The non-compositional fake tokenizer proves Y1: plans must come from the
full-candidate tokenization, not from tokenizing a character prefix and its
remainder separately.
"""

import math

import pytest

from jevmlx.schema import StructuredSchema
from jevmlx.trie import build_trie, log_softmax, logsumexp, score_trie, softmax

_QUOTE = ord('"')


class NonCompositionalTokenizer:
    """Tokenizes the word LOWER as ONE token, LOW and ER as single tokens.

    encode("LOW") + encode("ER") != encode("LOWER") by construction, so any
    plan built from a character prefix + separately tokenized remainder is
    detectably wrong.
    """

    name_or_path = "fake-non-compositional"

    _SPECIAL = (("LOWER", [999]), ("LOW", [7]), ("ER", [8]))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Structural text ({, newline, quotes, spaces, , : , \n) tokenizes
        # char-wise; only the value words map to special (non-compositional)
        # token sequences.
        out: list[int] = []
        i = 0
        while i < len(text):
            for word, ids in self._SPECIAL:
                if text.startswith(word, i):
                    out.extend(ids)
                    i += len(word)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def __len__(self) -> int:
        return 1000


class OtherTokenizer(NonCompositionalTokenizer):
    """Same interface, different vocabulary — must get its own cached plan."""

    name_or_path = "fake-other"

    _SPECIAL = (("LOWER", [555]), ("LOW", [3]), ("ER", [4]))


def test_plan_uses_full_sequence_tokenization():
    """Y1: the LOW/LOWER remainders come from encoding the full candidates."""
    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["LOW", "LOWER"]}}
    )
    tok = NonCompositionalTokenizer()
    plan = schema.compile_labels_plan(tok)
    remainders = plan["fields"]["action"]["remainders"]
    # '  "action": "LOW"'+',\n' -> [7, _QUOTE, comma, newline]
    # '  "action": "LOWER"'+',\n' -> [999, _QUOTE, comma, newline]
    # (LOWER is ONE token; the old character-prefix plan would have produced
    # token 8 (ER) somewhere and missed the terminator.)
    assert remainders == [[7, _QUOTE, 44, 10], [999, _QUOTE, 44, 10]]
    flat = [t for remainder in remainders for t in remainder]
    assert 8 not in flat
    # The shared lead-in is the '{\n  "action": "' structure, kept out of
    # shared_ids as the schema-wide prefix (the engine's prefill tail).
    assert plan["lead_in_ids"] == tok.encode('{\n  "action": "')
    assert plan["fields"]["action"]["shared_ids"] == []
    assert 999 not in plan["lead_in_ids"]


def test_plan_cache_is_per_tokenizer():
    """Y6: one schema, two tokenizers -> two distinct cached plans."""
    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["LOW", "LOWER"]}}
    )
    # Keep the tokenizer instances alive: the cache is weakref-keyed, so a
    # dead tokenizer's entry disappears with it.
    tok_a = NonCompositionalTokenizer()
    tok_b = OtherTokenizer()
    plan_a = schema.compile_labels_plan(tok_a)
    plan_b = schema.compile_labels_plan(tok_b)
    assert plan_a["fields"]["action"]["remainders"] == [[7, _QUOTE, 44, 10], [999, _QUOTE, 44, 10]]
    assert plan_b["fields"]["action"]["remainders"] == [[3, _QUOTE, 44, 10], [555, _QUOTE, 44, 10]]
    assert len(schema._plans) == 2
    assert schema.compile_labels_plan(tok_a) is plan_a
    assert schema.compile_labels_plan(tok_b) is plan_b


def test_trie_rows_branch_vs_distinct():
    """Y2: shared first token -> 2 branch rows; distinct first tokens -> 1."""
    # APPROVE diverges at the root; BLOCK_TRANSACTION/BLOCK_USER diverge after
    # their shared 'BLOCK_' characters -> two branch nodes.
    colliding = build_trie(
        [
            [ord(c) for c in "APPROVE"],
            [ord(c) for c in "BLOCK_TRANSACTION"],
            [ord(c) for c in "BLOCK_USER"],
        ]
    )
    assert len(colliding) == 2

    distinct = build_trie([[ord(c) for c in "APPROVE"], [ord(c) for c in "REVIEW"]])
    assert len(distinct) == 1
    assert distinct[0]["path"] == []  # the root row: suffix only


def _logits_lookup(nodes: list[dict], table: dict[tuple, list[float]]):
    by_path = {
        tuple(node["path"]): values for node, values in zip(nodes, table.values(), strict=True)
    }
    return lambda node: by_path[tuple(node["path"])]


def test_trie_probabilities_match_manual_computation():
    """T3: one-collision trie, hand-built logits; sums to 1 and matches math.

    Choices: A (remainder [1]); B ([2, 3]); C ([2, 4]). Branch nodes: the root
    ({1, 2}) and the node after token 2 ({3, 4}).
    """
    remainders = [[1], [2, 3], [2, 4]]
    nodes = build_trie(remainders)
    assert len(nodes) == 2

    root_logits = [1.0, 2.0]  # token 1 vs token 2 at the root
    split_logits = [0.5, -0.5]  # token 3 vs token 4 after token 2
    tables = [root_logits, split_logits]
    by_path = {tuple(node["path"]): values for node, values in zip(nodes, tables, strict=True)}
    scores = score_trie(nodes, 3, lambda node: by_path[tuple(node["path"])])

    # Manual computation, natural log.
    lse_root = math.log(math.exp(1.0) + math.exp(2.0))
    lse_split = math.log(math.exp(0.5) + math.exp(-0.5))
    expected = [
        1.0 - lse_root,
        2.0 - lse_root + 0.5 - lse_split,
        2.0 - lse_root - 0.5 - lse_split,
    ]
    assert scores == pytest.approx(expected, abs=1e-12)

    probs = [math.exp(lp) for lp in scores]
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)

    # Temperature applies ONCE to the final scores (softmax(scores / T));
    # per-branch softmax stays at T=1, so ranking is invariant to T.
    probs_hot = softmax(scores, temperature=2.0)
    raw = [math.exp(e / 2.0) for e in expected]
    z = sum(raw)
    assert probs_hot == pytest.approx([p / z for p in raw], abs=1e-12)
    assert max(range(3), key=probs.__getitem__) == max(range(3), key=probs_hot.__getitem__)


def test_softmax_temperature():
    values = [1.0, 2.0]
    assert softmax(values) == pytest.approx(
        [
            math.exp(1.0) / (math.exp(1.0) + math.exp(2.0)),
            math.exp(2.0) / (math.exp(1.0) + math.exp(2.0)),
        ]
    )
    hot = softmax(values, temperature=2.0)
    assert hot == pytest.approx(
        [
            math.exp(0.5) / (math.exp(0.5) + math.exp(1.0)),
            math.exp(1.0) / (math.exp(0.5) + math.exp(1.0)),
        ]
    )


def test_identical_remainders_rejected():
    """Two choices with the same token sequence cannot be distinguished.

    B4 rejects duplicate literals at construction, so this drives the
    compile-time check through two distinct literals that tokenize alike.
    """

    class SameTokens(NonCompositionalTokenizer):
        name_or_path = "fake-same-tokens"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # 'OK1' and 'OK2' tokenize identically (both become [42]).
            if "OK1" in text or "OK2" in text:
                return [ord(c) for c in text.replace("OK1", "OK").replace("OK2", "OK")]
            return super().encode(text, add_special_tokens)

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["OK1", "OK2"]}}
    )
    with pytest.raises(ValueError, match="token-identical"):
        schema.compile_labels_plan(SameTokens())


def test_choice_with_double_quote_is_json_escaped():
    """F1: candidates are json.dumps-escaped, never f-string interpolated.

    A choice containing a double quote must appear in the candidate text with
    the backslash escape, exactly as the assembled JSON will contain it.
    """
    tok = NonCompositionalTokenizer()
    schema = StructuredSchema(
        {
            "quote": {
                "type": "enum",
                "description": "d",
                "choices": ['say "hi"', "plain"],
            }
        }
    )
    plan = schema.compile_labels_plan(tok)
    lead_in = plan["lead_in_ids"]
    shared = plan["fields"]["quote"]["shared_ids"]
    remainders = plan["fields"]["quote"]["remainders"]

    bs_quote = chr(92) + chr(34)  # backslash + double quote, the JSON escape
    candidate_text = (
        chr(123)
        + chr(10)
        + "  "
        + chr(34)
        + "quote"
        + chr(34)
        + ": "
        + chr(34)
        + "say "
        + bs_quote
        + "hi"
        + bs_quote
        + chr(34)
        + ","
        + chr(10)
    )
    expected_escaped = tok.encode(candidate_text)
    assert lead_in + shared + remainders[0] == expected_escaped
    # A naive f-string candidate (invalid JSON) would tokenize differently.
    assert tok.encode('{\n  "quote": "say ""hi"""\n') != expected_escaped


def test_strict_token_prefix_remainder_rejected():
    """F2: remainder A strict-prefix of B -> never distinguished -> ValueError."""

    class Prefixing:
        name_or_path = "fake-prefixing"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if 'ABC"' in text:
                return [10, 11]
            if 'AB"' in text:
                return [10]
            return [ord(c) for c in text]

        def __len__(self) -> int:
            return 100

    schema = StructuredSchema(
        {"x": {"type": "enum", "description": "d", "choices": ["AB", "ABC", "OK"]}}
    )
    with pytest.raises(ValueError, match="strict token-prefix"):
        schema.compile_labels_plan(Prefixing())


def test_choice_that_is_token_prefix_of_another_is_rejected():
    """A remainder that is a strict prefix of another leaves a choice unscored.

    Tokenizer maps AB -> [10], ABC -> [10, 11], ABD -> [10, 12] (AB's token is
    a strict prefix of the other two): scoring can never separate AB from its
    siblings, so plan compilation must reject the field.
    """

    class Nested:
        name_or_path = "fake-nested"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # The candidate text ends with 'AB"' / 'ABC"' / 'ABD"'; map the
            # value token(s) so AB's single id is a strict prefix of the rest.
            if 'AB"' in text:
                return [10]
            if 'ABC"' in text:
                return [10, 11]
            if 'ABD"' in text:
                return [10, 12]
            return [ord(c) for c in text]

        def __len__(self) -> int:
            return 100

    schema = StructuredSchema(
        {"x": {"type": "enum", "description": "d", "choices": ["AB", "ABC", "ABD"]}}
    )
    with pytest.raises(ValueError, match="strict token-prefix"):
        schema.compile_labels_plan(Nested())


def test_score_trie_rejects_non_finite_logits():
    """T6: NaN/inf logits raise ValueError naming the branch node."""
    nodes = build_trie([[1], [2]])
    with pytest.raises(ValueError, match="non-finite"):
        score_trie(nodes, 2, lambda node: [float("nan"), 1.0])


def test_score_trie_tiny_temperature_ranking_invariant():
    """T5/T6: temperature applied once to final scores; T=1e-3 keeps ranking."""
    remainders = [[1], [2, 3], [2, 4]]
    nodes = build_trie(remainders)
    tables = [[1.0, 2.0], [0.5, -0.5]]
    by_path = {tuple(n["path"]): v for n, v in zip(nodes, tables, strict=True)}
    scores = score_trie(nodes, 3, lambda node: by_path[tuple(node["path"])])

    probs_t1 = softmax(scores)
    probs_tiny = softmax(scores, temperature=1e-3)
    assert all(math.isfinite(p) for p in probs_t1)
    assert all(math.isfinite(p) for p in probs_tiny)
    winner = max(range(3), key=probs_t1.__getitem__)
    assert winner == max(range(3), key=probs_tiny.__getitem__)


def test_single_choice_enum_scores_one_point_oh():
    """T7: cardinality-1 enum -> P=1.0, no branch rows, no crash."""
    schema = StructuredSchema({"only": {"type": "enum", "description": "d", "choices": ["ONLY"]}})
    tok = NonCompositionalTokenizer()
    plan = schema.compile_labels_plan(tok)
    remainders = plan["fields"]["only"]["remainders"]
    assert len(remainders) == 1
    nodes = build_trie(remainders)
    assert nodes == []  # single leaf: no branch points
    scores = score_trie(nodes, 1, lambda node: [])
    assert scores == [0.0]
    assert softmax(scores)[0] == pytest.approx(1.0)


def test_mixed_schema_rows_carry_lead_in_exactly_once():
    """R1: every row (enum, boolean, multi) starts with the lead-in exactly once."""
    from jevmlx.trie import build_trie as _bt

    tok = NonCompositionalTokenizer()
    schema = StructuredSchema(
        {
            "flag": {"type": "boolean", "description": "d"},
            "action": {"type": "enum", "description": "d", "choices": ["LOW", "LOWER"]},
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["opt_a", "opt_b"],
            },
        }
    )
    plan = schema.compile_labels_plan(tok)
    lead_in = plan["lead_in_ids"]
    assert lead_in, "fake tokenizer must produce a shared lead-in"

    # Assemble the rows exactly like the engine does.
    rows: list[list[int]] = []
    for p in plan["fields"].values():
        if not isinstance(p, dict):
            continue
        if "options" in p:
            rows.extend(lead_in + list(s) for s in p["suffix_ids_list"])
        elif "remainders" in p:
            trie_nodes = _bt(p["remainders"])
            rows.extend(lead_in + list(p["shared_ids"]) + list(n["path"]) for n in trie_nodes)

    assert rows, "mixed schema must produce rows"
    for row in rows:
        assert row[: len(lead_in)] == lead_in
        assert row[len(lead_in) : len(lead_in) * 2] != lead_in  # not duplicated
    # And the enum candidate must still round-trip to its full text.
    p = plan["fields"]["action"]
    full = lead_in + p["shared_ids"] + p["remainders"][0]
    assert full == tok.encode('{\n  "action": "LOW",\n')


def test_multi_option_strict_prefix_pair_rejected():
    """R5: an option whose true/false continuations are prefix-related raises."""
    from jevmlx.schema import StructuredSchema as _SS

    class PrefixPair(NonCompositionalTokenizer):
        name_or_path = "fake-prefix-pair"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            base = super().encode(text, add_special_tokens)
            # Make option_a's Y candidate a strict token-prefix of its N
            # candidate: encode the Y row as the N row minus its last token.
            if "/00" in text and ': "Y"' in text:
                n_cand = self.encode(text.replace(': "Y"', ': "N"'), add_special_tokens)
                return n_cand[:-1]
            return base

    schema = _SS(
        {
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["opt_a", "opt_b"],
            }
        }
    )
    with pytest.raises(ValueError, match="strict"):
        schema.compile_labels_plan(PrefixPair())


def test_softmax_extreme_temperature_no_nan():
    """B3: (v - max)/T order — [-1,-2] at T=1e-300 stays finite, no NaN."""
    result = softmax([-1.0, -2.0], temperature=1e-300)
    assert all(math.isfinite(p) for p in result)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_mixed_enum_multi_lead_in_round_trip():
    """B1: lead-in spans scalar AND multi prefixes; rows round-trip."""
    tok = NonCompositionalTokenizer()
    schema = StructuredSchema(
        {
            "action": {"type": "enum", "description": "d", "choices": ["LOW", "LOWER"]},
            "flags": {"type": "multi", "description": "d", "choices": ["opt_a", "opt_b"]},
        }
    )
    plan = schema.compile_labels_plan(tok)
    lead_in = plan["lead_in_ids"]

    # With only one scalar field, the lead-in must NOT contain that field's
    # name (it is the common prefix of ALL row prefixes, multi included).
    assert b"action".decode() not in "".join(
        chr(t) if 32 <= t < 127 else "?" for t in lead_in
    ) or lead_in == tok.encode('{\n  "')

    rows: list[tuple[str, list[int], str]] = []  # (kind, row, full candidate text)
    for _fname, p in plan["fields"].items():
        if not isinstance(p, dict):
            continue
        if "options" in p:
            for oi, ids in enumerate(p["suffix_ids_list"]):
                option = p["options"][oi]
                full = lead_in + list(ids)
                rows.append(("multi", full, "{\n  " + '"flags.' + option + '": '))
        elif "remainders" in p:
            for node in build_trie(p["remainders"]):
                full = lead_in + list(p["shared_ids"]) + list(node["path"])
                rows.append(("enum", full, '{\n  "action": "LOW"'))

    kinds = {kind for kind, _, _ in rows}
    assert kinds == {"enum", "multi"}
    for _kind, row, _ in rows:
        assert row[: len(lead_in)] == lead_in
        assert row[len(lead_in) : 2 * len(lead_in)] != lead_in


def test_zero_length_row_rejected_when_no_common_prefix():
    """B2: per-field shared empty AND branching root -> ValueError at compile."""

    class NoCommon(NonCompositionalTokenizer):
        name_or_path = "fake-no-common"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # The whole candidate collapses to ONE token keyed by its choice
            # letter: candidates share no prefix (different single tokens).
            if '"action": "A' in text:
                return [65]  # one token for the whole A-candidate
            if '"action": "B' in text:
                return [66]  # different single token for the B-candidate
            return super().encode(text, add_special_tokens)

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )
    with pytest.raises(ValueError, match="share no token prefix"):
        schema.compile_labels_plan(NoCommon())


def test_duplicate_multi_choices_rejected():
    """B4: duplicate multi values raise ValueError naming the field."""
    with pytest.raises(ValueError, match="duplicate choice 'opt_a'"):
        StructuredSchema(
            {
                "flags": {
                    "type": "multi",
                    "description": "d",
                    "choices": ["opt_a", "opt_a", "opt_b"],
                }
            }
        )


def test_log_softmax_no_cancellation_huge_values():
    """C1: [1e300, 1e300] -> [-ln2, -ln2], computed from shifted logits."""
    import math as _math

    result = log_softmax([1e300, 1e300])
    expected = -_math.log(2)
    assert result[0] == pytest.approx(expected, rel=1e-12)
    assert result[1] == pytest.approx(expected, rel=1e-12)


def test_log_softmax_no_cancellation_huge_negative():
    """C1: [-1e300, 0] -> [-1e300, 0] exactly (shift = max = 0)."""
    result = log_softmax([-1e300, 0.0])
    assert result[1] == pytest.approx(0.0, abs=1e-12)
    assert result[0] == pytest.approx(-1e300, rel=1e-12)


def test_logsumexp_no_cancellation():
    """C1: logsumexp([1e300, 1e300]) = 1e300 + ln 2, not 1e300."""
    import math as _math

    result = logsumexp([1e300, 1e300])
    assert result == pytest.approx(1e300 + _math.log(2), rel=1e-12)


def test_multi_option_no_common_prefix_rejected():
    """C2: an option pair sharing no first token raises at compile time."""

    class NoCommonPair(NonCompositionalTokenizer):
        name_or_path = "fake-no-common-pair"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if ': "Y"' in text:
                return [84]  # single token, unique to the Y candidate
            if ': "N"' in text:
                return [70]  # different single token for the N candidate
            return super().encode(text, add_special_tokens)

    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["opt_a", "opt_b"]}}
    )
    with pytest.raises(ValueError, match=r"option '00' \('opt_a'\).*share no token prefix"):
        schema.compile_labels_plan(NoCommonPair())


def test_field_name_with_dot_rejected():
    """C3: dot-free field names keep '<field>.<option>' keys injective."""
    with pytest.raises(ValueError, match="contains '.'"):
        StructuredSchema(
            {
                "a.b": {"type": "boolean", "description": "d"},
                "flags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
            }
        )


def test_multi_option_key_never_collides_with_field_name():
    """C3: field 'a' + option 'b.c' vs field 'a.b' — the dot rule rejects it."""
    with pytest.raises(ValueError, match="contains '.'"):
        StructuredSchema(
            {
                "a.b": {"type": "boolean", "description": "d"},
                "a": {"type": "multi", "description": "d", "choices": ["b.c", "c"]},
            }
        )


def test_field_named_lead_in_ids_does_not_collide():
    """D1: metadata lives beside field plans — a field named _lead_in_ids
    (or lead_in_ids) cannot be clobbered by plan metadata."""
    tok = NonCompositionalTokenizer()
    schema = StructuredSchema(
        {
            "_lead_in_ids": {"type": "boolean", "description": "d"},
            "lead_in_ids": {"type": "enum", "description": "d", "choices": ["A", "B"]},
        }
    )
    plan = schema.compile_labels_plan(tok)
    # Metadata key present and correct.
    assert plan["lead_in_ids"] == tok.encode('{\n  "')
    # Both fields have their own untouched plans.
    assert "_lead_in_ids" in plan["fields"]
    assert "lead_in_ids" in plan["fields"]
    assert set(plan["fields"]["_lead_in_ids"]) == {"shared_ids", "remainders"}
    assert set(plan["fields"]["lead_in_ids"]) == {"shared_ids", "remainders"}
    # The fields' plans are exactly the per-field data, no metadata mixed in.
    all_values = {v for p in plan["fields"].values() for v in p}
    assert all_values <= {"shared_ids", "remainders", "options", "suffix_ids_list"}


def test_build_trie_handles_5000_token_remainder():
    """N1: build_trie walks iteratively — a 5000-token remainder builds."""
    remainders = [
        list(range(1, 5001)),  # 1..5000
        list(range(1, 4001)) + [9999] + list(range(4001, 5001)),
    ]
    nodes = build_trie(remainders)
    # The remainders share the first 4000 tokens, diverge at token 4001
    # (one branch point holding the shared path), and re-converge after.
    assert len(nodes) == 1
    assert len(nodes[0]["path"]) == 4000
    assert nodes[0]["children"] == {4001: [0], 9999: [1]}


def test_build_trie_output_order_unchanged_vs_reference():
    """N1: the iterative walk reproduces the recursive pre-order exactly."""
    import sys

    sys.setrecursionlimit(60)  # would explode on the recursive version

    # Reference: independent recursive implementation.
    def recursive_nodes(remainders):
        root: dict = {"children": {}, "choices": []}
        for idx, remainder in enumerate(remainders):
            root["choices"].append(idx)
            node = root
            for token in remainder:
                node = node["children"].setdefault(token, {"children": {}, "choices": []})
                node["choices"].append(idx)
        out = []

        def walk(node, path):
            if len(node["children"]) >= 2:
                out.append(
                    (
                        list(path),
                        {t: node["children"][t]["choices"] for t in sorted(node["children"])},
                    )
                )
            for token in sorted(node["children"]):
                walk(node["children"][token], [*path, token])

        walk(root, [])
        return out

    remainders = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 5],
        [6, 7],
        [6, 8, 9],
    ]
    got = [(n["path"], n["children"]) for n in build_trie(remainders)]
    assert got == recursive_nodes(remainders)


def test_equal_but_distinct_tokenizers_get_distinct_plans():
    """P2: WeakKeyDictionary keyed by __eq__/__hash__; two equal-but-distinct
    fakes with different encodings must get two different plans."""

    class EqTokenizer:
        def __init__(self, shift: int):
            self.shift = shift
            self.name_or_path = f"fake-eq-{shift}"

        def __eq__(self, other):
            return isinstance(other, EqTokenizer)  # all instances equal

        def __hash__(self):
            return 42  # identical hash

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [(ord(c) + self.shift) % 65536 for c in text]

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )
    plan_a = schema.compile_labels_plan(EqTokenizer(shift=0))
    plan_b = schema.compile_labels_plan(EqTokenizer(shift=10))
    # The plans must differ: under a shared (equal-keyed) cache entry the
    # second tokenizer would silently reuse the first one's token ids.
    assert plan_a["fields"]["action"]["remainders"] != plan_b["fields"]["action"]["remainders"]
    # And each matches a fresh compile with the same tokenizer.
    again = schema.compile_labels_plan(EqTokenizer(shift=10))
    assert again["fields"]["action"]["remainders"] == plan_b["fields"]["action"]["remainders"]


def test_cache_evicts_entry_when_tokenizer_dies():
    """P2: weakref.finalize evicts the plan when the tokenizer is collected."""
    import gc
    import weakref as _weakref

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )
    tok = NonCompositionalTokenizer()
    schema.compile_labels_plan(tok)
    cache_key = (id(tok), "labels")
    assert cache_key in schema._plans
    ref = _weakref.ref(tok)
    del tok
    gc.collect()
    assert ref() is None
    assert cache_key not in schema._plans


def test_compile_slot_plan_reads_the_cache():
    """C2: compile_slot_plan must READ the plan cache like compile_labels_plan
    does — two calls with the same tokenizer return the same object and add
    no new weakrefs (previously it recompiled and stacked a finalizer per
    call); a different tokenizer recompiles."""

    class CountingTokenizer:
        name_or_path = "fake-count"

        def __init__(self):
            self.encodes = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            self.encodes += 1
            return [ord(c) % 65536 for c in text]

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )
    tok = CountingTokenizer()
    import weakref as _weakref

    plan1 = schema.compile_slot_plan(tok)
    encodes_after_first = tok.encodes
    weakrefs_after_first = _weakref.getweakrefcount(tok)

    plan2 = schema.compile_slot_plan(tok)
    assert plan2 is plan1  # same object: served from the cache
    assert tok.encodes == encodes_after_first  # nothing re-encoded
    assert _weakref.getweakrefcount(tok) == weakrefs_after_first  # no new finalizers

    # A different tokenizer (same schema) recompiles.
    tok2 = CountingTokenizer()
    plan3 = schema.compile_slot_plan(tok2)
    assert plan3 is not plan1
    assert tok2.encodes > 0


def test_labels_plan_cache_hit_is_the_same_object():
    """C2: labels path via the shared helper — same object, flat weakrefs."""
    import weakref as _weakref

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )
    tok = NonCompositionalTokenizer()
    plan1 = schema.compile_labels_plan(tok)
    weakrefs_after_first = _weakref.getweakrefcount(tok)
    plan2 = schema.compile_labels_plan(tok)
    assert plan2 is plan1
    assert _weakref.getweakrefcount(tok) == weakrefs_after_first


def test_re_storing_a_plan_does_not_stack_finalizers():
    """C2: _cache_plan registers the eviction finalizer only when the key is
    new — re-storing must not accumulate finalizer objects."""
    import gc
    import weakref as _weakref

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )
    tok = NonCompositionalTokenizer()
    schema.compile_labels_plan(tok)
    n = _weakref.getweakrefcount(tok)
    schema.compile_labels_plan(tok)  # hit: no new weakrefs
    assert _weakref.getweakrefcount(tok) == n
    # Force a re-store on the same key (evict-less overwrite path).
    schema._cache_plan(tok, {"lead_in_ids": [], "fields": {}}, mode="labels")
    assert _weakref.getweakrefcount(tok) == n
    # The eviction still works.
    cache_key = (id(tok), "labels")
    del tok
    gc.collect()
    assert cache_key not in schema._plans
