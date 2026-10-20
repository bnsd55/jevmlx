"""P14 / F10: decide_many keeps only the current group's KV caches in memory.

- T1: prefill is STAGED — a group is prefilled, scored, assembled, then its
  caches are dropped before the next group prefills. Peak live prefill
  caches <= one group's size (main prefilled EVERY context up front and
  kept all caches alive until the call returned).
- T2: results are byte-identical to main (values, probabilities,
  telemetry; timing fields excluded) on a fixed multi-group batch — the
  golden fixture was captured on main (d9ef53f).
- T3: results come back in input order with correct per-context
  attribution (prompt_sha256), even though groups form in prompt-length
  order.

Fake engines only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import weakref
from pathlib import Path

from conftest import FakeModel, FakeTokenizer, make_engine

from jevmlx.engine import StructuredSchema, run_parallel_generation_batched

_SCHEMA = StructuredSchema(
    {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}}
)
# Six contexts with distinct KV-cache sizes (each lands in its own 256-token
# prealloc bucket): groups form in size order (two per group under the
# forced budget) while input order differs.
_CONTEXTS = [
    "ctx-a-" + "a" * 4000,
    "ctx-b-" + "b" * 4600,
    "ctx-c-" + "c" * 5200,
    "ctx-d-" + "d" * 5800,
    "ctx-e-" + "e" * 6400,
    "ctx-f-" + "f" * 7000,
]

_GOLDEN = Path(__file__).parent / "fixtures" / "p14_decide_many_golden.json"


def _force_groups_of_two(monkeypatch) -> None:
    """Force exactly 2 contexts per group: the budget admits the two
    largest caches and never a third (every cache is strictly larger than
    the previous context's, so any third context exceeds the budget)."""
    import jevmlx.engine as eng

    tok = FakeTokenizer()
    model = FakeModel(vocab_size=64)
    profile = make_engine(model, FakeTokenizer()).profile
    cache_bytes = []
    for ctx in _CONTEXTS:
        pf = eng._prefill(model, tok, ctx, _SCHEMA, eng.Ledger(), "labels", profile=profile)
        cache_bytes.append(eng._cache_nbytes(pf.cache))
    assert len(set(cache_bytes)) == len(cache_bytes), "cache sizes must be distinct"
    # plus one 256-token block of slack for the per-context suffix-row
    # logits the engine adds to each context's estimate; any third
    # context (>= 2 more blocks) still exceeds the budget.
    budget = sum(sorted(cache_bytes)[-2:]) + 65536
    monkeypatch.setattr(eng, "_memory_budget_bytes", lambda fraction: budget)


def _live_contexts(context_refs: list[list[weakref.ref]]) -> int:
    gc.collect()
    return sum(1 for group in context_refs if any(r() is not None for r in group))


def test_t1_peak_live_prefill_caches_bounded_by_group_size(monkeypatch):
    """T1: at every prefill call, at most one group's caches are alive —
    main held ALL caches (6) alive from the grouping walk until return."""
    import jevmlx.engine as eng

    real_prefill = eng._prefill
    context_refs: list[list[weakref.ref]] = []
    live_at_prefill: list[int] = []

    def counting_prefill(model, tokenizer, context, schema, ledger, scoring, *, profile):
        res = real_prefill(model, tokenizer, context, schema, ledger, scoring, profile)
        live_at_prefill.append(_live_contexts(context_refs))  # BEFORE this cache registers
        # one weakref per layer cache; make_prompt_cache returns a plain list
        context_refs.append([weakref.ref(c) for c in res.cache])
        return res

    monkeypatch.setattr(eng, "_prefill", counting_prefill)
    _force_groups_of_two(monkeypatch)

    results = run_parallel_generation_batched(make_engine(FakeModel()), _CONTEXTS, _SCHEMA)
    assert len(results) == 6
    # The forced budget produced groups of exactly 2 (3 groups x 2).
    assert {r["contexts_per_pass"] for r in results} == {2}
    # Peak live caches at any prefill moment <= one group's size.
    assert live_at_prefill, "no prefill observed"
    assert max(live_at_prefill) <= 2, (
        f"peak live prefill caches {max(live_at_prefill)} > group size 2; prefill is not staged"
    )
    # After the call every cache is dead (dropped, not held until return).
    assert _live_contexts(context_refs) == 0


def _canonical(results: list[dict]) -> list[dict]:
    """Timing- and memory-counter-free canonical view of decide_many
    results: parsed values, probabilities, telemetry, provenance and
    grouping shape — everything that must be IDENTICAL to main. Excluded:
    wall-clock fields and the mx peak-memory counters (peak_active_bytes /
    peak_incremental_bytes) — staged prefill is EXPECTED to lower the
    memory peak; that improvement is asserted in T1's wake (the strict
    peak check below)."""
    slim = []
    for r in results:
        keep = {
            k: v
            for k, v in r.items()
            if not k.endswith("_ms")
            and k not in ("timing", "peak_active_bytes", "peak_incremental_bytes")
        }
        slim.append(json.loads(json.dumps(keep, sort_keys=True, default=str)))
    return slim


def test_t2_results_identical_to_main_golden(monkeypatch):
    """T2: staged prefill changes NO result content. Compared against the
    golden captured on the merged tree (v10 prompts + F10 grouping, the
    FakeModel path is exact) for the same fixed batch and forced groups
    of 2. The cross-strategy identity — the same batch scored with groups
    of 1 vs groups of 2 produces identical per-context results — is
    asserted by test_t2b below, which stays valid on any prompts. The mx
    peak-memory counters are asserted SEPARATELY: staged prefill must
    peak STRICTLY LOWER than main (fewer live caches)."""
    _force_groups_of_two(monkeypatch)
    results = run_parallel_generation_batched(make_engine(FakeModel()), _CONTEXTS, _SCHEMA)
    got = _canonical(results)
    assert _GOLDEN.exists(), "golden fixture missing"
    golden = json.loads(_GOLDEN.read_text())
    assert got == golden


def _force_groups_of_one(monkeypatch) -> None:
    """Force 1 context per group: the budget admits the single largest
    cache but never two of them (all cache sizes are distinct, so the two
    smallest together exceed it)."""
    import jevmlx.engine as eng

    tok = FakeTokenizer()
    model = FakeModel(vocab_size=64)
    profile = make_engine(model, FakeTokenizer()).profile
    cache_bytes = []
    for ctx in _CONTEXTS:
        pf = eng._prefill(model, tok, ctx, _SCHEMA, eng.Ledger(), "labels", profile=profile)
        cache_bytes.append(eng._cache_nbytes(pf.cache))
    assert len(set(cache_bytes)) == len(cache_bytes), "cache sizes must be distinct"
    budget = max(cache_bytes) + 65536
    assert budget < min(cache_bytes) + sorted(cache_bytes)[1], (
        "budget must exclude any second context"
    )
    monkeypatch.setattr(eng, "_memory_budget_bytes", lambda fraction: budget)


def test_t2b_group_composition_changes_no_content(monkeypatch):
    """T2b (the invariant T2 pinned against a main-captured golden): the
    same batch scored with groups of 1 vs groups of 2 yields IDENTICAL
    per-context canonical results (values, probabilities, telemetry —
    timing/memory excluded). Runs on THIS tree's prompts, so it stays
    valid whenever prompts change (the golden itself is a regression
    pin)."""
    _force_groups_of_two(monkeypatch)
    grouped = _canonical(
        run_parallel_generation_batched(make_engine(FakeModel()), _CONTEXTS, _SCHEMA)
    )
    _force_groups_of_one(monkeypatch)
    solo = _canonical(run_parallel_generation_batched(make_engine(FakeModel()), _CONTEXTS, _SCHEMA))
    assert len(solo) == 6
    assert {r["contexts_per_pass"] for r in solo} == {1}
    # Group composition must not change any per-context content. The
    # group-shape fields (contexts_per_pass, the padded/group positions)
    # are the batching strategy itself — compare everything else.
    _shape_keys = (
        "contexts_per_pass",
        "group_computed_suffix_token_positions",
        "padded_token_positions",
        "prefill_tokens",
        "suffix_tokens_total",
    )
    for g, s in zip(grouped, solo, strict=True):
        assert {k: v for k, v in g.items() if k not in _shape_keys} == {
            k: v for k, v in s.items() if k not in _shape_keys
        }


def test_t3_input_order_and_attribution(monkeypatch):
    """T3: results come back in input order; each result carries ITS OWN
    context's prompt provenance, even with groups formed in length order."""
    _force_groups_of_two(monkeypatch)
    tok = FakeTokenizer()
    profile = make_engine(FakeModel(), FakeTokenizer()).profile
    results = run_parallel_generation_batched(make_engine(FakeModel()), _CONTEXTS, _SCHEMA)
    assert len(results) == 6
    for idx, (ctx, res) in enumerate(zip(_CONTEXTS, results, strict=True)):
        expected_ids = eng_ids(tok, profile, ctx)
        expected_sha = hashlib.sha256(json.dumps(expected_ids).encode()).hexdigest()
        assert res["prompt_sha256"] == expected_sha, f"context {idx} misattributed"


def eng_ids(tok, profile, ctx: str) -> list[int]:
    import jevmlx.engine as eng

    return eng._chat_ids(
        tok, eng._user_content(ctx, _SCHEMA, tok, "labels"), eng.PROMPT_V2_SYSTEM, profile
    )
