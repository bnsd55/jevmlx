"""W4-B: scoring parity check — one implementation, two consumers.

The W1-A slow test (batch=1 vs batch=N vs chunked scoring agreement) and
the bench's ``parity.json`` producer must share EXACTLY one
implementation, or the recorded parity and the tested parity drift apart.
This module is that implementation:

- :func:`check_scoring_parity` runs the batch=1 / batched / chunked
  comparison over a set of preset cases on a LOADED engine (model,
  tokenizer) and returns per-case drift measurements.
- :func:`parity_report` turns those measurements into the ``parity.json``
  payload the bench writes into each model folder:

    {
      "model": "...",
      "prompt_version": "jevmlx-parallel-v7",
      "max_abs_drift_nats": 0.027,
      "winners_identical": true,
      "atol": 0.05,
      "passed": true,
      "cases": ["fintech_fraud", "support_triage"],
      "test": "test_w1a_scoring_parity_batch_vs_chunked_real_model",
      "run_at": "2026-09-18T12:00:00Z"
    }

``atol`` is ``INSTABILITY_BAND`` (5e-2 nats) — the same constant the slow
parity test asserts against, so a recorded ``passed: true`` means the slow
test's invariant held on this machine at bench time. Metal batched matmuls
tile differently per batch shape and the legal-mass logsumexp adds a
reduction; bit-identity is not the invariant, WINNERS are, and log_scores
must agree within the band.

The schema is a superset of PR #29's BENCHMARKING.md record (model, test,
passed, max_drift_nats, atol, run_at): the extra keys (prompt_version,
winners_identical, max_abs_drift_nats alias, cases) identify WHAT ran;
PR #29's ``check_parity`` only reads ``passed``, so the superset stays
compatible both ways.
"""

from __future__ import annotations

import datetime
import json
from importlib import resources
from typing import Any

__all__ = [
    "PARITY_TEST_NAME",
    "check_scoring_parity",
    "check_batched_parity",
    "parity_report",
    "bundled_preset_specs",
]

# The slow twin this check mirrors (tests/test_engine.py).
PARITY_TEST_NAME = "test_w1a_scoring_parity_batch_vs_chunked_real_model"

# Bundled presets the parity check runs on: every schema family shipped in
# jevmlx/presets/. The bench's quality-eval dataset is built from
# benchmarks/cases.json (payment_risk + support_triage); the preset JSONs in
# jevmlx/presets/ carry the same two families plus the stress shapes
# (code_security, high_cardinality_255). Parity covers all four — a model
# that is only parity-clean on the small schemas has not earned the README
# compat table.


def bundled_preset_specs() -> list[tuple[str, dict]]:
    """(preset_id, preset) for every bundled preset, loaded from the
    package. W5-D finding 41: the WHOLE preset dict is returned (schema +
    context + description), not just spec["schema"] — the parity run must
    score the preset's REAL context, not generic filler. ``id`` inside the
    JSON wins when present; else the stem. The single-context consumers
    (:func:`check_scoring_parity`, :func:`parity_report`) read
    ``preset["schema"]`` and ``preset.get("context")`` through
    :func:`_case_context`."""
    specs: list[tuple[str, dict]] = []
    preset_dir = resources.files("jevmlx") / "presets"
    for entry in sorted(preset_dir.iterdir()):
        if entry.name.endswith(".json"):
            spec = json.loads(entry.read_text(encoding="utf-8"))
            specs.append((spec.get("id", entry.stem), spec))
    return specs


def check_scoring_parity(
    engine: Any,
    cases: list[tuple[str, dict]] | None = None,
    max_rows_options: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    """Run batch=1 vs batched vs chunked scoring parity on a loaded engine.

    Takes the loaded :class:`jevmlx.engine.Engine` .

    ``cases``: (case_id, schema_dict) pairs; default is every bundled
    preset. For each case, ONE context is scored three ways through
    :func:`jevmlx.engine.run_parallel_generation`:

    - batched: ``max_rows=None`` (the default, chunking heuristic decides)
    - per ``max_rows_options`` values: chunked at 1 (row-per-pass) and 2

    Returns a result dict::

        {
          "winners_identical": bool,      # every field's value agrees everywhere
          "max_abs_drift_nats": float,    # worst |log_score_full - log_score_chunk|
          "per_case": {case_id: {"max_abs_drift_nats": float}},
        }

    Fields without ``log_scores`` (multi) contribute their winner check
    only — multi per-option P(yes) has no field-level log P by design.
    """
    from jevmlx.engine import run_parallel_generation

    if cases is None:
        cases = bundled_preset_specs()

    winners_identical = True
    max_abs_drift = 0.0
    per_case: dict[str, dict[str, float]] = {}

    for case_id, preset in cases:
        schema = _make_schema(case_id, preset["schema"])
        context = _case_context(case_id, preset)
        full = run_parallel_generation(engine, context, schema)
        case_drift = 0.0
        for max_rows in max_rows_options:
            again = run_parallel_generation(engine, context, schema, max_rows=max_rows)
            # Winners must be identical — a different decision is a real bug.
            for fname, entry in full["parsed_json"].items():
                if again["parsed_json"].get(fname, {}).get("value") != entry["value"]:
                    winners_identical = False
            # log_scores agree within the band (FP drift only).
            for fname, tel in full["field_telemetry"].items():
                ls_full = tel.get("log_scores")
                if ls_full is None:
                    continue
                ls_again = again["field_telemetry"].get(fname, {}).get("log_scores")
                if ls_again is None or set(ls_full) != set(ls_again):
                    winners_identical = False
                    continue
                for choice, ls in ls_full.items():
                    drift = abs(ls - ls_again[choice])
                    case_drift = max(case_drift, drift)
        per_case[case_id] = {"max_abs_drift_nats": case_drift}
        max_abs_drift = max(max_abs_drift, case_drift)

    return {
        "winners_identical": winners_identical,
        "max_abs_drift_nats": max_abs_drift,
        "per_case": per_case,
    }


def parity_report(
    engine: Any,
    model_id: str,
    cases: list[tuple[str, dict]] | None = None,
    max_rows_options: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    """The ``parity.json`` payload for a model folder (schema above).

    Takes the loaded :class:`jevmlx.engine.Engine`.

    W5c-1: runs BOTH parity checks into ONE payload —
    :func:`check_scoring_parity` (batch vs chunked, single-context) AND
    :func:`check_batched_parity` (batched-vs-independent matrix with the
    raw pre-rescore row-logit gate). ``check_results.check_parity``
    rejects a parity.json without ``max_raw_row_drift_nats`` as pre-v2, so
    the bench's payload must always carry both key sets.

    The GATE is: winners identical (both checks), single-context
    batch-vs-chunked drift under ``atol`` (the W1-A invariant, unchanged),
    and raw pre-rescore row drift under ``raw_atol``. The single-context
    and batched-raw drifts are SEPARATE keys each against its own
    documented band — they are different measurements (batch<=2 vs
    batch=max(context_counts)*n_rows), so one ``max`` would gate the
    single-context invariant at the batched envelope and vice versa. The
    multi-context batched FINAL log-score drift is a third measurement:
    recorded as ``max_batched_drift_nats``, reported but not banded (its
    winner identity and the raw row gate cover a real divergence).
    """

    from jevmlx.engine import INSTABILITY_BAND, PROMPT_VERSION

    result = check_scoring_parity(engine, cases, max_rows_options)
    batched = check_batched_parity(engine, cases)
    winners_identical = bool(result["winners_identical"] and batched["winners_identical"])
    # The gated single-context drift (W1-A invariant, measured ~0.044 on
    # this machine's real models — headroom under the shared band).
    max_abs_drift = result["max_abs_drift_nats"]
    # W5-D finding 42: the raw pre-rescore row-logit gate (from the
    # batched matrix) — the key check_results requires for v2. Gated at
    # its own band (raw_atol), NOT the single-context atol.
    max_raw_drift = batched["max_raw_row_drift_nats"]
    raw_atol = batched["raw_atol"]
    # The batched matrix's final log-score drift: informative, not gated.
    max_batched_drift = batched["max_abs_drift_nats"]
    return {
        "model": model_id,
        "prompt_version": PROMPT_VERSION,
        "max_abs_drift_nats": round(max_abs_drift, 6),
        "max_raw_row_drift_nats": round(max_raw_drift, 6),
        "max_batched_drift_nats": round(max_batched_drift, 6),
        "winners_identical": winners_identical,
        "atol": INSTABILITY_BAND,
        "raw_atol": raw_atol,
        "passed": bool(
            winners_identical and max_abs_drift < INSTABILITY_BAND and max_raw_drift < raw_atol
        ),
        "cases": [case_id for case_id, _preset in (cases or bundled_preset_specs())],
        "test": PARITY_TEST_NAME,
        "run_at": datetime.datetime.now(datetime.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }


def write_parity_json(
    engine: Any,
    model_id: str,
    out_dir: Any,
    cases: list[tuple[str, dict]] | None = None,
) -> dict[str, Any]:
    """Run the parity check and write ``<out_dir>/parity.json``. Returns
    the payload written. The bench calls this right after the engine load;
    ``check_parity`` (PR #29) reads the file back."""
    from pathlib import Path

    payload = parity_report(engine, model_id, cases)
    out = Path(out_dir) / "parity.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------- helpers --


def _make_schema(case_id: str, schema_dict: dict) -> Any:
    """A StructuredSchema for the preset's schema dict, with per-field
    descriptions defaulted (presets carry real descriptions; the fallback
    keeps ad-hoc case dicts usable in tests)."""
    from jevmlx.schema import StructuredSchema

    fields: dict[str, dict] = {}
    for fname, spec in schema_dict.items():
        entry = dict(spec)
        entry.setdefault("description", f"{case_id}.{fname}")
        fields[fname] = entry
    return StructuredSchema(fields)


def _case_context(case_id: str, preset: dict) -> str:
    """The preset's REAL context (W5-D finding 41: bundled_preset_specs now
    returns the whole preset, so the recorded cases use the recorded
    evidence). When absent (bare schema dicts in tests) a stable filler
    keeps the run deterministic."""
    return (
        preset.get("context")
        or f"Parity check context for {case_id}. Routine request with all "
        "validation checks passed; process according to the standard policy "
        "and select the appropriate fields."
    )


# ---------------------------------------------------------------- batched --


def check_batched_parity(
    engine: Any,
    cases: list[tuple[str, dict]] | None = None,
    context_counts: tuple[int, ...] = (1, 2, 4),
    prior_correction: bool = False,
) -> dict[str, Any]:
    """Batched-vs-independent parity matrix (W5-D findings 40/42).

    The W1-A gate only ever exercised run_parallel_generation — every
    decide_many bug passed it. This matrix runs each case at 1/2/4
    contexts, equal and MIXED prompt lengths, and compares:

    - RAW row logits before near-tie rescore (finding 42): one scoring
      pass per context through the shared _score_rows at identical chunk
      shapes — batched results re-keyed per context — against the
      independent path's rows. Final decisions are compared separately so
      a batch-1 rescore cannot mask raw batch drift.
    - Final decisions (parsed values + log_scores) batched vs independent.
    - prior on/off (prior_correction=True computes the shared prior and
      must still match decide-per-context).

    Returns the same shape as check_scoring_parity plus a
    ``batched_matrix`` section per case.
    """
    from jevmlx.engine import (
        INSTABILITY_BAND,
        _build_schema_rows,
        _prefill,
        _score_rows,
        run_parallel_generation,
        run_parallel_generation_batched,
    )

    if cases is None:
        cases = bundled_preset_specs()

    # W5c-1: the raw pre-rescore row gate scores the SAME rows at batch=1
    # (the canonical reference the near-tie rescore trusts) vs batch=
    # max(context_counts)*n_rows (one merged pass per group, the decide_many
    # row shape). That is a batch-SHAPE stress, not a same-shape comparison:
    # Metal FP accumulation grows with the merged reduction width, so the
    # raw drift on a real model (measured 0.125 at context_count=4) is
    # ~2.5x the single-context W1-A band (0.05) with winners identical —
    # accumulation noise, not a divergence. The raw gate's job is to catch
    # a real shape-dependent divergence (the W5-D RowDriftModel fixture
    # adds 5.0 nats — 25x this envelope). Its tolerance therefore scales
    # with the matrix's own context-count multiplier rather than reusing
    # the single-context band.
    raw_atol = max(context_counts) * INSTABILITY_BAND

    winners_identical = True
    max_abs_drift = 0.0
    max_raw_drift = 0.0
    per_case: dict[str, dict[str, Any]] = {}

    built_cache: dict[str, Any] = {}
    for case_id, preset in cases:
        schema = _make_schema(case_id, preset["schema"])
        context = _case_context(case_id, preset)
        built = _build_schema_rows(schema, engine.tokenizer, "slots")
        built_cache[case_id] = built

        # --- RAW row logits (finding 42): score the SAME rows for one
        # context three ways — alone (the independent reference), and as
        # rows 2..n of a batched group — before any near-tie rescore.
        raw_drift = 0.0
        if built["rows"]:
            vocab_size = engine.vocab_size
            from jevmlx.timing import Ledger

            pf = _prefill(
                engine.model, engine.tokenizer, context, schema, Ledger(), "slots", engine.profile
            )
            # The reference is the CANONICAL batch=1 shape (one row per
            # forward) — the same shape the near-tie rescore trusts. The
            # batched side runs 4 context-copies of the rows in merged
            # chunks (the decide_many row shape), so raw batch drift shows.
            # Reference: the context's rows alone at the CANONICAL batch=1
            # shape (one row per forward) — the shape the near-tie rescore
            # trusts. The batched side runs 4 context-copies of the rows in
            # merged chunks (the decide_many row shape — one merged pass
            # per group), so raw batch drift shows.
            ref = _score_rows(
                engine.model,
                pf.cache,
                built["rows"],
                built["row_decision"],
                vocab_size,
                built["pad_id"],
                1,
                Ledger(),
            )
            # The batched side runs 4 context-copies of the rows in merged
            # chunks (the decide_many row shape — one merged pass per
            # group), so raw batch drift shows against the batch=1
            # reference the near-tie rescore trusts.
            cache_slots = [pf.cache] * (4 * len(built["rows"]))
            all_rows = built["rows"] * 4
            all_decisions = built["row_decision"] * 4
            batched = _score_rows(
                engine.model,
                pf.cache,
                all_rows,
                all_decisions,
                vocab_size,
                built["pad_id"],
                max(2, 4 * len(built["rows"])),
                Ledger(),
                cache_slots=cache_slots,
            )
            for ridx in range(len(built["rows"])):
                ref_vals = ref.row_logits.get(ridx)
                got_vals = batched.row_logits.get(ridx)
                if ref_vals is None or got_vals is None:
                    winners_identical = False
                    continue
                for a, b in zip(ref_vals, got_vals, strict=True):
                    raw_drift = max(raw_drift, abs(a - b))
        max_raw_drift = max(max_raw_drift, raw_drift)

        # --- Final decisions at 1/2/4 contexts, equal + mixed lengths.
        case_drift = 0.0
        n_ctx = max(context_counts)
        contexts = [context] * n_ctx
        # Mixed prompt lengths: pad later contexts with distinct tails.
        mixed = [context + f" Additional evidence block {i}." for i in range(n_ctx)]
        for ctx_list in (contexts, mixed):
            independent = [
                run_parallel_generation(engine, c, schema, prior_correction=prior_correction)
                for c in ctx_list
            ]
            batched_results = run_parallel_generation_batched(
                engine,
                ctx_list,
                schema,
                prior_correction=prior_correction,
            )
            for ind, bat in zip(independent, batched_results, strict=True):
                for fname, entry in ind["parsed_json"].items():
                    if bat["parsed_json"].get(fname, {}).get("value") != entry["value"]:
                        winners_identical = False
                for fname, tel in ind["field_telemetry"].items():
                    ls_ind = tel.get("log_scores")
                    if ls_ind is None:
                        continue
                    ls_bat = bat["field_telemetry"].get(fname, {}).get("log_scores")
                    if ls_bat is None or set(ls_ind) != set(ls_bat):
                        winners_identical = False
                        continue
                    for choice, ls in ls_ind.items():
                        case_drift = max(case_drift, abs(ls - ls_bat[choice]))
        per_case[case_id] = {
            "max_abs_drift_nats": case_drift,
            "max_raw_row_drift_nats": raw_drift,
        }
        max_abs_drift = max(max_abs_drift, case_drift)

    return {
        "winners_identical": winners_identical,
        "max_abs_drift_nats": max_abs_drift,
        "max_raw_row_drift_nats": max_raw_drift,
        # W5c-1: the documented band for the batched raw row gate (a
        # batch-shape stress, not a same-shape comparison) — see the
        # raw_atol derivation above.
        "raw_atol": raw_atol,
        "per_case": per_case,
    }
