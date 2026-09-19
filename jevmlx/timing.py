"""W5b-8: the event ledger for engine timing (GPT-REVIEW-2 §C item 12).

Standalone module — pure Python, NO mlx import, NO engine wiring yet.
Engine adoption (deleting the t_gather_ms/t_broadcast_ms/t_scored_ms
accumulators) lands after W5-B.

Why: the current ``*_ms`` keys overlap (``suffix_eval_ms`` includes
broadcast + gather, which are ALSO reported separately), ``elapsed_ms``
excludes the prior pass, and decide_many divides group timers per context.
One ledger measures each interval ONCE, non-overlapping; every reported
key is a derivation of the same interval set.

Phases: ``prior`` (the neutral pass) and ``main``
(everything else). Names: plan, prompt_render, prefill, cache_merge,
transformer, lm_head, gather, rescore, dependency, reconciliation,
assembly — plus the batched wrappers group_wall / per-context assembly.

Derived keys (one place: :meth:`Ledger.derived_flat`) exactly match
today's result keys:

- ``plan_compile_ms`` = plan
- ``prefill_ms`` = prefill
- ``cache_broadcast_ms`` = cache_merge
- ``suffix_eval_ms`` = cache_merge + transformer + gather  (composite,
  marked derived — it exists only so old readers keep working)
- ``lm_head_gather_ms`` = gather
- ``second_pass_ms`` = dependency
- ``prior_ms`` = Σ(prior-phase spans)
- ``elapsed_ms`` = Σ(main-phase TOP-LEVEL spans — children live inside
  their parents, so top-level sums cannot double-count the partition)
- ``total_ms`` = prior_ms + elapsed_ms

Batched views (:meth:`Ledger.batched_views` / helpers): ``group_wall`` is
the outer group span (prefill loop + one merged scoring pass);
``per_item_amortized`` divides it by the group size; ``per_item_end_to_end``
is, per context, its own prefill-span start → its assembly-span end —
honest per-context latency, no fabricated splits.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["Interval", "Ledger", "SpanError"]


class SpanError(RuntimeError):
    """A span contract violation: sibling overlap, parent escape, or a
    close without a matching open. The ledger refuses to record a lie."""


@dataclass(frozen=True)
class Interval:
    """One measured interval: phase, name, perf_counter seconds."""

    name: str
    phase: str
    t0: float
    t1: float

    @property
    def ms(self) -> float:
        return (self.t1 - self.t0) * 1000.0


class _OpenSpan:
    __slots__ = ("name", "phase", "t0", "parent", "child_start")

    def __init__(self, name: str, phase: str, t0: float, parent: _OpenSpan | None):
        self.name = name
        self.phase = phase
        self.t0 = t0
        self.parent = parent
        self.child_start: float | None = None  # t0 of the most recent child


class _SpanContext:
    """The context manager ``with ledger.span(name, phase):`` yields."""

    __slots__ = ("_ledger", "_name", "_phase", "_opened")

    def __init__(self, ledger: Ledger, name: str, phase: str):
        self._ledger = ledger
        self._name = name
        self._phase = phase
        self._opened: _OpenSpan | None = None

    def __enter__(self) -> None:
        self._opened = self._ledger._open(self._name, self._phase)

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # An exception unwinds through THIS span: drop it and any spans
            # nested INSIDE it (they cannot outlive their parent) WITHOUT
            # recording fake completed intervals. Spans ABOVE it on the
            # stack — its ancestors — stay open: the caller may catch the
            # exception and continue, and the parents close normally later.
            # (Dropping the whole stack broke parents that outlive a caught
            # child failure — e.g. a Metal retry inside a group span.)
            opened = self._opened
            self._opened = None
            if opened is None or opened not in self._ledger._stack:
                # Never opened, or already dropped by an inner __exit__ with
                # exc_info — a double unwind is a no-op.
                return
            stack = self._ledger._stack
            while stack:
                span = stack.pop()
                if span is opened:
                    break
            return
        self._opened = None
        self._ledger._close()


class Ledger:
    """One non-overlapping event ledger.

    Usage::

        ledger = Ledger()
        with ledger.span("prefill"):
            ...
        flat = ledger.derived_flat()
        by_phase = ledger.summary()

    Contracts (asserted, not conventions — the ledger refuses to record a
    lie):
    - Spans nest LIFO; closing out of order raises SpanError.
    - A child must start within its parent.
    - Siblings partition their parent: a child must start at/after the
      previous sibling's end (no sibling overlap), so children always
      partition their parent in time.
    - Top-level spans must not overlap either — the ledger is ONE timeline.
    """

    def __init__(self) -> None:
        self._intervals: list[Interval] = []
        self._stack: list[_OpenSpan] = []
        self._last_end_at_depth: dict[int, float] = {}  # depth -> last close t1

    # -- measuring ---------------------------------------------------------

    def span(self, name: str, phase: str = "main") -> _SpanContext:
        """Open a span: ``with ledger.span("prefill"): ...``."""
        return _SpanContext(self, name, phase)

    @property
    def intervals(self) -> list[Interval]:
        return list(self._intervals)

    def _open(self, name: str, phase: str) -> _OpenSpan:
        parent = self._stack[-1] if self._stack else None
        t0 = time.perf_counter()
        if parent is not None and t0 < parent.t0:
            raise SpanError(f"span {name!r} starts before its parent")
        depth = len(self._stack)
        prev_end = self._last_end_at_depth.get(depth)
        if prev_end is not None and t0 < prev_end - 1e-9:
            raise SpanError(
                f"span {name!r} at depth {depth} starts at {t0:.6f}, before "
                f"the previous sibling closed at {prev_end:.6f} (overlap)"
            )
        opened = _OpenSpan(name, phase, t0, parent)
        self._stack.append(opened)
        return opened

    def _close(self) -> None:
        if not self._stack:
            raise SpanError("span close with no open span")
        span = self._stack.pop()
        t1 = time.perf_counter()
        if t1 < span.t0:
            raise SpanError(f"span {span.name!r} closed before it opened")
        depth = len(self._stack)
        prev_end = self._last_end_at_depth.get(depth)
        if prev_end is not None and t1 < prev_end - 1e-9:
            raise SpanError(
                f"span {span.name!r} closes at {t1:.6f}, before the previous "
                f"same-level span closed at {prev_end:.6f} (overlap)"
            )
        self._last_end_at_depth[depth] = t1
        self._intervals.append(Interval(span.name, span.phase, span.t0, t1))

    # -- views --------------------------------------------------------------

    def summary(self) -> dict[str, dict[str, float]]:
        """{phase: {name: total_ms}} — the ledger's own accounting."""
        out: dict[str, dict[str, float]] = {}
        for iv in self._intervals:
            out.setdefault(iv.phase, {})
            out[iv.phase][iv.name] = out[iv.phase].get(iv.name, 0.0) + iv.ms
        return out

    def derived_flat(self, prior_ms: float | None = None) -> dict[str, float]:
        """Today's flat ``*_ms`` keys, derived from the interval set.

        One place computes them (per the note): the flat keys exist only so
        existing readers keep working; they are pure derivations, never
        separately measured. ``elapsed_ms`` sums only TOP-LEVEL main spans
        (children are inside their parents' [t0, t1] — the partition makes
        the top-level sum the true elapsed time without double counting).

        ``prior_ms`` overrides the prior-phase derivation for ledgers that
        carry no prior span (the batched per-context ledgers: the neutral
        pass runs ONCE on the request ledger — finding 26 — and is exposed
        on every result as the shared value).
        """
        top_level = self._top_level_intervals()

        # Composite keys (cache_merge/transformer/gather/plan/...) are
        # children of prefill — they must derive from the FULL interval
        # set, not the top-level cut (only elapsed_ms is top-level, so the
        # partition never double-counts).
        def name_ms(name: str) -> float:
            return sum(iv.ms for iv in self._intervals if iv.name == name)

        if prior_ms is None:
            prior_ms = sum(iv.ms for iv in self._top_level_intervals(phase="prior"))
        # elapsed_ms = the ONE top-level ``request`` span when present
        # (W5b-14 review F10: true wall time — the stage spans are children
        # of it and never double-count). Older paths without a request span
        # fall back to the top-level MAIN-span sum (the partition).
        request_ivs = [iv for iv in top_level if iv.name == "request" and iv.phase == "main"]
        if request_ivs:
            elapsed_ms = sum(iv.ms for iv in request_ivs)
        else:
            elapsed_ms = sum(iv.ms for iv in top_level if iv.phase == "main")
        cache_merge = name_ms("cache_merge")
        gather = name_ms("gather")
        return {
            "plan_compile_ms": name_ms("plan"),
            "prefill_ms": name_ms("prefill"),
            "cache_broadcast_ms": cache_merge,
            "suffix_eval_ms": cache_merge + name_ms("transformer") + gather,
            "lm_head_gather_ms": gather,
            "second_pass_ms": name_ms("dependency"),
            "prior_ms": prior_ms,
            "elapsed_ms": elapsed_ms,
            "total_ms": prior_ms + elapsed_ms,
        }

    def _top_level_intervals(self, phase: str | None = None) -> list[Interval]:
        """Intervals not strictly contained in another interval (same phase).

        Runs in O(n²) — interval counts per request are small (dozens), and
        this is only the derivation step, not the measurement.
        """
        pool = [iv for iv in self._intervals if phase is None or iv.phase == phase]
        top: list[Interval] = []
        for iv in pool:
            contained = any(
                other is not iv
                and other.t0 <= iv.t0
                and other.t1 >= iv.t1
                and (other.t0, other.t1) != (iv.t0, iv.t1)
                for other in pool
            )
            if not contained:
                top.append(iv)
        return top

    def last_interval(self, name: str) -> Interval:
        """The most recent CLOSED interval with this name (SpanError if none
        closed yet) — the call-site replacement for bare ``intervals[-1]``."""
        for iv in reversed(self._intervals):
            if iv.name == name:
                return iv
        raise SpanError(f"no closed interval named {name!r}")

    def batched_views(self, group_int: Interval, n_items: int) -> dict[str, list[float]]:
        """The group-level decide_many views from one ledger.

        ``group_int`` is the outer group span (``last_interval("group_wall")``);
        ``n_items`` is the group's context count. Returns ``group_wall_ms``
        (once) and ``per_item_amortized_ms`` (the group wall divided by the
        group — the ONE amortized share, ``n_items`` entries). Per-context
        end-to-end is NOT derived here: it is each context's own prefill
        span + amortized share + own assembly span, assembled by the engine
        from the per-context ledgers.
        """
        group_wall_ms = group_int.ms
        n = max(1, n_items)
        return {
            "group_wall_ms": [group_wall_ms],
            "per_item_amortized_ms": [group_wall_ms / n] * n,
        }

    def per_item_end_to_end(self, prefill_iv: Interval, assembly_iv: Interval) -> float:
        """Honest per-context latency: own prefill start → assembly end."""
        return (assembly_iv.t1 - prefill_iv.t0) * 1000.0


@contextmanager
def span(ledger: Ledger, name: str, phase: str = "main") -> Iterator[None]:
    """Functional alternative to the method: ``with span(l, "prefill"):``."""
    with ledger.span(name, phase):
        yield
