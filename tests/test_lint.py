"""Tests for jevmlx.lint using a deterministic fake tokenizer (no downloads)."""

import zlib

import pytest

from jevmlx.lint import lint_schema
from jevmlx.schema import StructuredSchema
from tests.conftest import make_test_renderer
from tests.test_trie import NonCompositionalTokenizer


class FakeTokenizer:
    """Char-level structural text; crc32 ids for value words (non-compositional)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            if text[i] in '{}\n ":,':
                out.append(ord(text[i]))
                i += 1
                continue
            if text[i] == "_":
                out.append(zlib.crc32(b"_"))
                i += 1
                continue
            j = i
            while j < len(text) and text[j] not in '{}\n ":,_':
                j += 1
            word = text[i:j]
            out.append(zlib.crc32(word.encode()))
            i = j
        return out


def _findings(schema_dict: dict, tokenizer=None) -> list:
    return lint_schema(StructuredSchema(schema_dict), tokenizer or FakeTokenizer())


def test_collision_detected_with_suggestion():
    """Two choices sharing their first token are flagged, with a rotation rename."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "ALLOW"],
            }
        }
    )
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert "BLOCK_TRANSACTION, BLOCK_USER" in collisions[0].message
    assert collisions[0].suggestion == "TRANSACTION_BLOCK, USER_BLOCK"


def test_clean_schema_has_no_findings():
    """Distinct first tokens produce no findings."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "ALLOW", "REVIEW_MANUALLY"],
            }
        }
    )
    assert findings == []


def test_shared_first_word_alone_is_not_a_collision():
    """A shared first word that IS the common prefix is stripped before scoring.

    ["BLOCK_TRANSACTION", "BLOCK_USER"] share the prefix BLOCK; the engine
    strips it and compares TRANSACTION vs USER at the decision position.
    """
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER"],
            }
        }
    )
    assert findings == []


def test_duplicate_choice_rejected_at_construction():
    """B4: duplicate enum values raise ValueError in FieldDefinition."""
    with pytest.raises(ValueError, match="duplicate choice 'ALLOW'"):
        StructuredSchema(
            {
                "action": {
                    "type": "enum",
                    "description": "next action",
                    "choices": ["ALLOW", "ALLOW", "BLOCK_USER"],
                }
            }
        )


def test_quote_fusion_gives_prefix_choice_its_own_branch():
    """The closing quote fuses into the final word token (non-compositional).

    BLOCK's candidate ends ...BLOCK" (quote fused into the token), while
    BLOCK_TRANSACTION/BLOCK_USER carry a bare ...BLOCK token followed by the
    rest. BLOCK therefore diverges at the root (no empty_choice), and the two
    longer choices share their first token (collision, with rotation fix).
    """
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK", "BLOCK_TRANSACTION", "BLOCK_USER"],
            }
        }
    )
    assert [f.kind for f in findings] == ["collision"]
    assert "BLOCK_TRANSACTION, BLOCK_USER" in findings[0].message
    assert findings[0].suggestion == "TRANSACTION_BLOCK, USER_BLOCK"


def test_boolean_fields_are_skipped():
    """Booleans always decide true/false; the lint must not touch them."""
    findings = _findings({"approved": {"type": "boolean", "description": "ok?"}})
    assert findings == []


def test_multi_fields_are_skipped():
    """Multi fields decide true/false per option; their plan entry is not per-choice.

    The plan for a multi field carries 'options' and choice_token_lists of
    [true, false], so a per-choice lint loop would index the wrong lists.
    Options are boolean decisions by construction and cannot collide.
    """
    findings = _findings(
        {
            "tags": {
                "type": "multi",
                "description": "select all that apply",
                "choices": ["BLOCK_USER", "BLOCK_TRANSACTION", "ALLOW"],
            }
        }
    )
    assert findings == []


def test_findings_are_dataclasses_with_kind_and_field():
    """Every finding carries field, kind, message; suggestion only when computed."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "ALLOW"],
            }
        }
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.field == "action"
    assert finding.kind == "collision"
    assert finding.message
    assert finding.suggestion


def test_collision_without_rotation_suggestion():
    """A collision the rotation rule cannot fix (single-word choice) gets None.

    X and X_Y share their leading crc32(X) token; X cannot be rotated (single
    word), so the finding carries suggestion=None.
    """
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["W", "X", "X_Y"],
            }
        }
    )
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert "X, X_Y" in collisions[0].message
    assert collisions[0].suggestion is None


def test_single_choice_enum_lints_without_raising():
    """L1a: a cardinality-1 enum has an empty remainder — no crash, and an
    informational single_choice finding (chosen behavior: informational)."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["ONLY"],
            }
        }
    )
    assert [f.kind for f in findings] == ["single_choice"]
    assert findings[0].field == "action"


def test_token_identical_pair_yields_compile_error_finding():
    """L1b: compile-time ValueErrors become compile_error findings, not
    silently-clean lint results."""

    class SameTokens(NonCompositionalTokenizer):
        name_or_path = "fake-lint-same-tokens"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if "OK1" in text or "OK2" in text:
                return [ord(c) for c in text.replace("OK1", "OK").replace("OK2", "OK")]
            return super().encode(text, add_special_tokens)

    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["OK1", "OK2"],
            }
        },
        tokenizer=SameTokens(),
    )
    assert [f.kind for f in findings] == ["compile_error"]
    assert findings[0].field == "action"
    assert "token-identical" in findings[0].message


def test_no_common_prefix_yields_compile_error_finding():
    """L1b: a no-common-prefix schema also lints as compile_error."""

    class NoCommon(NonCompositionalTokenizer):
        name_or_path = "fake-lint-no-common"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if '"action": "A' in text:
                return [65]
            if '"action": "B' in text:
                return [66]
            return super().encode(text, add_special_tokens)

    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["A", "B"],
            }
        },
        tokenizer=NoCommon(),
    )
    assert [f.kind for f in findings] == ["compile_error"]
    assert "share no token prefix" in findings[0].message


def test_valid_schema_lints_with_no_compile_error():
    """A healthy schema produces no compile_error findings."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "ALLOW"],
            }
        }
    )
    assert all(f.kind != "compile_error" for f in findings)


def test_multi_only_schema_strict_prefix_pair_yields_one_compile_error():
    """M1: a multi-only schema with a strict-prefix pair is attributed to the
    multi field — one compile_error finding, not a clean lint."""

    class PrefixPair(NonCompositionalTokenizer):
        name_or_path = "fake-m1-prefix-pair"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # opt_a's Y candidate is a strict token-prefix of its N candidate.
            if "/00" in text and ': "Y"' in text:
                n_cand = self.encode(text.replace(': "Y"', ': "N"'), add_special_tokens)
                return n_cand[:-1]
            return super().encode(text, add_special_tokens)

    findings = _findings(
        {
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["opt_a", "opt_b"],
            }
        },
        tokenizer=PrefixPair(),
    )
    assert len(findings) == 1
    assert findings[0].kind == "compile_error"
    assert findings[0].field == "flags"
    assert "strict" in findings[0].message


def test_mixed_schema_failing_boolean_reports_once_with_right_field():
    """M1: a mixed schema whose BOOLEAN field fails compilation yields exactly
    one compile_error finding, attributed to the boolean field."""

    class BoolKills(NonCompositionalTokenizer):
        name_or_path = "fake-m1-bool-kills"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # Only the boolean field's candidates share no token prefix.
            if '"flag": ' in text:
                if "true" in text:
                    return [84]
                if "false" in text:
                    return [70]
            return super().encode(text, add_special_tokens)

    findings = _findings(
        {
            "flag": {"type": "boolean", "description": "d"},
            "action": {
                "type": "enum",
                "description": "d",
                "choices": ["ALLOW", "BLOCK"],
            },
            "backup": {
                "type": "enum",
                "description": "d",
                "choices": ["ON", "OFF"],
            },
        },
        tokenizer=BoolKills(),
    )
    assert len(findings) == 1
    assert findings[0].kind == "compile_error"
    assert findings[0].field == "flag"


def test_schema_compile_error_is_value_error():
    """SchemaCompileError stays a ValueError subclass (back-compat for
    engine-side except ValueError handlers)."""
    from jevmlx.schema import SchemaCompileError

    err = SchemaCompileError("flag", "boom")
    assert isinstance(err, ValueError)
    assert err.field == "flag"
    assert str(err) == "boom"


def test_rotation_suggestion_rejected_if_name_exists_in_full_choice_set():
    """N2: BLOCK_TRANSACTION, BLOCK_USER + pre-existing TRANSACTION_BLOCK ->
    rotation would create a duplicate, so suggestion is None."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "TRANSACTION_BLOCK"],
            }
        }
    )
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert collisions[0].suggestion is None


def test_rotation_suggestion_still_works_without_conflicts():
    """N2: same rotation but no pre-existing rotated name -> suggestion kept."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "ALLOW"],
            }
        }
    )
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert collisions[0].suggestion == "TRANSACTION_BLOCK, USER_BLOCK"


def test_non_weakrefable_tokenizer_compiles_fresh_each_time():
    """N3: no id() fallback — a non-weakrefable tokenizer gets two
    independent plans (no dead-object cache reuse)."""

    class Uncacheable:
        __slots__ = ("name_or_path",)  # no __dict__/__weakref__: not weakrefable

        def __init__(self):
            self.name_or_path = "fake-uncacheable"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(c) for c in text]

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )
    tok_a = Uncacheable()
    tok_b = Uncacheable()
    plan_a = schema.compile_labels_plan(tok_a, make_test_renderer(tok_a, schema, "labels"))
    plan_b = schema.compile_labels_plan(tok_b, make_test_renderer(tok_b, schema, "labels"))
    assert plan_a is not plan_b
    # Both plans are complete and correct.
    assert plan_a["fields"]["action"]["remainders"] == (plan_b["fields"]["action"]["remainders"])


def test_rotation_rejected_when_it_merely_moves_the_collision():
    """P1: BLOCK_TRANSACTION, BLOCK_USER, TRANSACTION_ALLOW — rotation gives
    TRANSACTION_BLOCK, USER_BLOCK, TRANSACTION_ALLOW which still collide on
    the first word token, so the tokenizer-verified suggestion is None."""

    class WordTok:
        """Underscore-separated words each become one crc32 token."""

        name_or_path = "fake-p1-word"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            out: list[int] = []
            i = 0
            while i < len(text):
                if text[i] in '{}\n ":,':
                    out.append(ord(text[i]))
                    i += 1
                    continue
                if text[i] == "_":
                    out.append(zlib.crc32(b"_"))
                    i += 1
                    continue
                j = i
                while j < len(text) and text[j] not in '{}\n ":,_':
                    j += 1
                out.append(zlib.crc32(text[i:j].encode()))
                i = j
            return out

    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "TRANSACTION_ALLOW"],
            }
        },
        tokenizer=WordTok(),
    )
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert collisions[0].suggestion is None


def test_rotation_kept_when_renamed_set_compiles_collision_free():
    """P1: BLOCK_TRANSACTION, BLOCK_USER, ALLOW — renamed set compiles and has
    no first-token collision, so the suggestion survives."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "ALLOW"],
            }
        }
    )
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert collisions[0].suggestion == "TRANSACTION_BLOCK, USER_BLOCK"
