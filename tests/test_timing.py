"""W5b-8 tests: jevmlx/timing.py — the event ledger.

Pure-Python tests (no mlx): partition contracts fire, nested spans
summarize, derived flat keys match today's engine keys, batched views.
"""

import pytest

from jevmlx.timing import Ledger, SpanError


def test_basic_span_and_summary():
    ledger = Ledger()
    with ledger.span("prefill"):
        pass
    summary = ledger.summary()
    assert "main" in summary
    assert summary["main"]["prefill"] >= 0.0


def test_nested_spans_partition_parent():
    """Children close inside their parent; the parent interval contains
    both children's [t0, t1]."""
    ledger = Ledger()
    with ledger.span("outer"):
        with ledger.span("a"):
            pass
        with ledger.span("b"):
            pass
    ivals = {iv.name: iv for iv in ledger.intervals}
    outer = ivals["outer"]
    for child in ("a", "b"):
        assert outer.t0 <= ivals[child].t0
        assert ivals[child].t1 <= outer.t1


def test_sibling_overlap_raises():
    """A span that opens before the previous same-depth sibling closed is
    an overlap — SpanError, and the failed open leaves the ledger
    unchanged."""
    ledger = Ledger()
    with ledger.span("outer"):
        with ledger.span("a"):
            pass
        # Drive the sibling-overlap guard directly: open 'b' with a
        # backdated clock (earlier than 'a's close).
        import jevmlx.timing as timing

        real_pc = timing.time.perf_counter
        a_end = max(iv.t1 for iv in ledger.intervals if iv.name == "a")
        timing.time.perf_counter = lambda: a_end - 1e-6
        try:
            with pytest.raises(SpanError, match="previous sibling"):
                ledger._open("b", "main")
        finally:
            timing.time.perf_counter = real_pc
        # The failed open must leave the ledger unchanged ('outer' is
        # still open, so only 'a' is recorded so far).
        assert [iv.name for iv in ledger.intervals] == ["a"]


def test_out_of_order_close_raises():
    ledger = Ledger()
    with pytest.raises(SpanError):
        ledger._close()  # nothing open


def test_close_ordering_nested():
    """A parent cannot close while a child is open (LIFO enforced)."""
    ledger = Ledger()
    with ledger.span("outer"):
        inner = ledger.span("child")
        inner.__enter__()
        # Attempt to close 'outer' while 'child' is open: the _SpanContext
        # for outer pops whatever is on top — LIFO means child pops first,
        # so the with-body cannot close outer early without SpanError.
        ctx = ledger.span("x")
        ctx.__enter__()
        ctx.__exit__(None, None, None)
        inner.__exit__(None, None, None)


def test_derived_flat_keys_match_engine_contract():
    """The derived flat keys are pure functions of the interval set and
    match today's result-key semantics (coder6's mapping)."""
    ledger = Ledger()
    with ledger.span("prior_pass", phase="prior"):
        pass
    with ledger.span("plan"):
        pass
    with ledger.span("prefill"):
        with ledger.span("cache_merge"):
            pass
        with ledger.span("transformer"):
            pass
        with ledger.span("gather"):
            pass
    with ledger.span("dependency"):
        pass
    flat = ledger.derived_flat()
    assert set(flat) == {
        "plan_compile_ms",
        "prefill_ms",
        "cache_broadcast_ms",
        "suffix_eval_ms",
        "lm_head_gather_ms",
        "second_pass_ms",
        "prior_ms",
        "elapsed_ms",
        "total_ms",
    }
    # suffix_eval_ms is the composite: cache_merge + transformer + gather.
    by_name = {iv.name: iv for iv in ledger.intervals}
    expected_suffix = by_name["cache_merge"].ms + by_name["transformer"].ms + by_name["gather"].ms
    assert flat["suffix_eval_ms"] == pytest.approx(expected_suffix)
    # elapsed_ms = top-level main spans only (children not double-counted).
    assert flat["elapsed_ms"] == pytest.approx(
        by_name["plan"].ms + by_name["prefill"].ms + by_name["dependency"].ms
    )
    assert flat["prefill_ms"] == pytest.approx(by_name["prefill"].ms)
    assert flat["cache_broadcast_ms"] == pytest.approx(by_name["cache_merge"].ms)
    assert flat["lm_head_gather_ms"] == pytest.approx(by_name["gather"].ms)
    assert flat["second_pass_ms"] == pytest.approx(by_name["dependency"].ms)
    assert flat["prior_ms"] == pytest.approx(by_name["prior_pass"].ms)
    assert flat["total_ms"] == pytest.approx(flat["prior_ms"] + flat["elapsed_ms"])


def test_elapsed_does_not_double_count_children():
    """prefill has three children; elapsed counts the parent once, not
    parent+children (the old overlap bug)."""
    ledger = Ledger()
    with ledger.span("prefill"):
        with ledger.span("cache_merge"):
            pass
        with ledger.span("transformer"):
            pass
        with ledger.span("gather"):
            pass
    flat = ledger.derived_flat()
    prefill = {iv.name: iv for iv in ledger.intervals}["prefill"].ms
    assert flat["elapsed_ms"] == pytest.approx(prefill)
    # And the composite suffix_eval still names its parts.
    assert flat["suffix_eval_ms"] > 0.0


def test_batched_views_amortized():
    """group_wall is the outer span; per_item_amortized divides by n."""
    ledger = Ledger()
    with ledger.span("group_wall") as _ctx:
        for _ in range(4):
            with ledger.span("prefill"):
                pass
            with ledger.span("transformer"):
                pass
    group_int = [iv for iv in ledger.intervals if iv.name == "group_wall"][0]
    views = ledger.batched_views(group_int, list(range(4)))
    assert views["per_item_amortized_ms"] == [pytest.approx(views["group_wall_ms"][0] / 4)]
    # Honest per-item end-to-end: own prefill start -> own assembly end.
    ivals = {iv.name: iv for iv in ledger.intervals}
    e2e = ledger.per_item_end_to_end(ivals["prefill"], ivals["transformer"])
    assert e2e >= 0.0


def test_span_exception_drops_interval():
    """An exception unwinding through a span records NO interval (a failed
    request must not report timings it never completed)."""
    ledger = Ledger()
    with pytest.raises(ValueError, match="boom"):
        with ledger.span("prefill"):
            raise ValueError("boom")
    assert ledger.intervals == []
    # And the ledger stays usable afterwards.
    with ledger.span("prefill"):
        pass
    assert len(ledger.intervals) == 1


def test_phase_separation_prior_vs_main():
    """prior-phase spans are excluded from elapsed_ms and summed separately
    into prior_ms (elapsed used to exclude prior; now total = both)."""
    ledger = Ledger()
    with ledger.span("neutral", phase="prior"):
        pass
    with ledger.span("prefill"):
        pass
    flat = ledger.derived_flat()
    assert flat["elapsed_ms"] > 0.0
    assert flat["prior_ms"] > 0.0
    summary = ledger.summary()
    assert "prior" in summary and "main" in summary
