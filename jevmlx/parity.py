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


def _field_margin(tel: dict) -> float:
    """Top-two probability margin from a field's telemetry (W5c-1 review §4).

    ``top_choices`` is sorted by probability descending; the margin is
    p1 - p2 (the same quantity the near-tie rescore keys on, in
    probability space). A field with one choice has margin inf.
    """
    tc = tel.get("top_choices") or []
    if len(tc) < 2:
        return float("inf")
    return float(tc[0]["probability"]) - float(tc[1]["probability"])


def _environment_metadata(engine: Any) -> dict[str, Any]:
    """Machine + library + model metadata for parity.json (W5c-1 review §3).

    Best-effort: every field degrades to None when the source is
    unavailable (the parity check runs on the LOADED engine, so mlx is
    present, but the machine tag may not resolve on a non-Apple-Silicon
    box). None is recorded explicitly — the schema is fixed, not inferred
    from what happened to resolve.
    """
    import platform

    from jevmlx.bench import machine_tag
    from jevmlx.engine import engine_metadata

    meta = engine_metadata(engine.model_id)
    chip = None
    try:
        chip = machine_tag()
    except Exception:  # noqa: BLE001 — best-effort provenance
        chip = None
    return {
        "chip": chip,
        "macos": platform.platform(),
        "mlx_version": meta["mlx_version"],
        "mlx_lm_version": meta["mlx_lm_version"],
        "activation_dtype": "float32",
        "quantization": meta["quantization"],
        "mask_path": "explicit_merge",
        "max_rows": 4,
        "context_count": 4,
    }


def _rescore_gate_safety_assertion(
    engine: Any,
    cases: list[tuple[str, dict]] | None,
) -> dict[str, Any]:
    """Per-field rescore-gate safety check (W5c-1 review §2 item 4).

    For every field in every case, computes batch-one evidence
    (single-context ``run_parallel_generation``) and batched evidence
    (``run_parallel_generation_batched`` at max contexts), then records:

    - ``batch_one_margin``: top-two probability margin at batch=1
    - ``batched_margin``: top-two probability margin at batch=N
    - ``reference_near_tie``: batch_one_margin < 0.05 (the field sat
      inside the near-tie band at the reference shape)
    - ``production_rescored``: did the batched path rescore this field?
    - ``winner_differs``: did the batched winner disagree with batch=1?
    - ``escaped_near_tie``: reference_near_tie AND NOT production_rescored
      — the exact failure mode the near-tie rescore exists to catch,
      missed at batch. A non-empty escaped list FAILS parity.
    """
    from jevmlx.engine import (
        INSTABILITY_BAND,
        run_parallel_generation,
        run_parallel_generation_batched,
    )

    if cases is None:
        cases = bundled_preset_specs()
    n_ctx = 4
    escaped: list[dict[str, Any]] = []
    per_field: list[dict[str, Any]] = []
    for case_id, preset in cases:
        schema = _make_schema(case_id, preset["schema"])
        context = _case_context(case_id, preset)
        one = run_parallel_generation(engine, context, schema)
        many = run_parallel_generation_batched(engine, [context] * n_ctx, schema)
        # The batched result is a list; index 0 is the first context copy.
        bat0 = many[0] if isinstance(many, list) else many
        for fname, one_tel in one["field_telemetry"].items():
            bat_tel = bat0["field_telemetry"].get(fname, {})
            one_margin = _field_margin(one_tel)
            bat_margin = _field_margin(bat_tel)
            one_val = one["parsed_json"].get(fname, {}).get("value")
            bat_val = bat0["parsed_json"].get(fname, {}).get("value")
            ref_near_tie = one_margin < INSTABILITY_BAND
            prod_rescored = bool(bat_tel.get("rescored", False))
            winner_differs = one_val != bat_val
            esc = ref_near_tie and not prod_rescored
            entry = {
                "case": case_id,
                "field": fname,
                "batch_one_margin": round(one_margin, 6),
                "batched_margin": round(bat_margin, 6),
                "reference_near_tie": ref_near_tie,
                "production_rescored": prod_rescored,
                "winner_differs": winner_differs,
                "escaped_near_tie": esc,
            }
            per_field.append(entry)
            if esc:
                escaped.append(entry)
    return {
        "escaped_near_tie": escaped,
        "per_field": per_field,
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

    The GATE is fail-closed at the FIXED 0.05 contract (review section 2:
    no widening): winners identical (both checks), single-context
    batch-vs-chunked drift, the batched pairwise GAP drift (d_gap), and
    the batched top-two margin drift — all under ``atol``. Raw logits
    carry an arbitrary additive offset, so ``max_raw_row_drift_nats`` is a
    DIAGNOSTIC only (not gated); the gap drift is what catches a real
    divergence. On the 0.5B: d_raw 0.125 but d_gap 0.070 — the drift is
    partly real, so the batched gate FAILS (parity_passed=false, fail
    closed). The single-context W1-A gate still passes.
    """

    from jevmlx.engine import INSTABILITY_BAND, PROMPT_VERSION

    result = check_scoring_parity(engine, cases, max_rows_options)
    batched = check_batched_parity(engine, cases)
    winners_identical = bool(result["winners_identical"] and batched["winners_identical"])
    # Single-context W1-A drift (the invariant that predates the matrix).
    max_abs_drift = result["max_abs_drift_nats"]
    # Batched final log-score drift (across the 1/2/4-context matrix).
    max_batched_drift = batched["max_abs_drift_nats"]
    # The decomposition (review section 2): d_raw = diagnostic; d_gap and
    # margin drift = the gate.
    max_raw_drift = batched["max_raw_row_drift_nats"]
    max_gap_drift = batched["max_gap_drift_nats"]
    max_logp_drift = batched["max_logp_drift_nats"]
    max_margin_drift = batched["max_margin_drift_nats"]
    max_merged_rows = batched.get("max_merged_rows", 0)
    # Rescore-gate safety assertion (review section 2, item 4): did any
    # field sit inside the 0.05 near-tie band at batch=1 but escape the
    # rescore under batching? That is the actual failure mode the raw gate
    # is a proxy for.
    rescore_flag = _rescore_gate_safety_assertion(engine, cases)
    escaped = rescore_flag["escaped_near_tie"]
    # W5c-9: the batched matrix's measured d_gap PERSISTS as a drift-envelope
    # record (user cache + probes folder) — the measured envelope the engine
    # band reads. The PARITY GATE below stays at the FIXED 0.05: recording
    # the envelope never redefines parity; it bounds the rescore band.
    recorded = _record_envelope_from_report(engine, model_id, max_gap_drift, max_merged_rows)
    passed = bool(
        winners_identical
        and max_abs_drift < INSTABILITY_BAND
        and max_gap_drift < INSTABILITY_BAND
        and max_margin_drift < INSTABILITY_BAND
        and not escaped
    )
    return {
        "model": model_id,
        "prompt_version": PROMPT_VERSION,
        "max_abs_drift_nats": round(max_abs_drift, 6),
        "max_raw_row_drift_nats": round(max_raw_drift, 6),
        # W5c-1 review: the gated decomposition.
        "max_gap_drift_nats": round(max_gap_drift, 6),
        "max_logp_drift_nats": round(max_logp_drift, 6),
        "max_margin_drift_nats": round(max_margin_drift, 6),
        "max_batched_drift_nats": round(max_batched_drift, 6),
        "winners_identical": winners_identical,
        "atol": INSTABILITY_BAND,
        # W5c-9: the drift-envelope resolution this report persisted (bound,
        # band, source) — informational; the gate above is untouched.
        "drift_envelope": recorded,
        # Rescore-gate safety assertion (review item 4).
        "rescore_gate": rescore_flag,
        # Environment metadata (review item 3).
        "environment": _environment_metadata(engine),
        "passed": passed,
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


def _record_envelope_from_report(
    engine: Any, model_id: str, max_gap_drift: float, merged_rows: int
) -> dict:
    """Persist the report's measured d_gap as a drift-envelope record (W5c-9).

    The batched matrix measured pairwise-gap drift at the 4-context merged
    shape (M = 4 * rows — the M>16 bucket for every bundled schema). The
    record lands in the user cache (keyed by the envelope tuple) and the
    model's probes folder. Best-effort: a failure NEVER fails parity — the
    envelope is an optimization for the band, not a gate input.
    """
    try:
        from jevmlx.driftenv import (
            MAX_GAP_DRIFT_KEY,
            envelope_key,
            probes_dir_for_record,
            record_envelope,
            shape_bucket,
        )

        key = envelope_key(engine)
        bucket = shape_bucket(merged_rows)  # the real merged pass M (C4)
        record = {
            "key": key,
            "shape_bucket": bucket,
            MAX_GAP_DRIFT_KEY: float(max_gap_drift),
            "source": "parity_report",
            "model": model_id,
            "matrix_rows": 4,
        }
        probes_dir = probes_dir_for_record(model_id, chip=key.get("chip") or "")
        record_envelope(record, probes_dir=probes_dir)
        from jevmlx.driftenv import rescore_band

        return {
            "shape_bucket": bucket,
            MAX_GAP_DRIFT_KEY: round(float(max_gap_drift), 6),
            "band": round(rescore_band(max_gap_drift), 6),
        }
    except Exception as exc:  # noqa: BLE001 — never fails parity
        return {"error": str(exc)}


def _slug(model_id: str) -> str:
    """A filesystem-safe slug for the model id (probes folder naming)."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-") or "model"


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
    import mlx.core as _mx
    import mlx.nn as _mlx_nn

    from jevmlx.engine import (
        _build_schema_rows,
        _prefill,
        _score_rows,
        run_parallel_generation,
        run_parallel_generation_batched,
    )

    if cases is None:
        cases = bundled_preset_specs()

    # W5c-1 review section 2: raw logits carry an arbitrary additive
    # offset, so uncentered d_raw is a DIAGNOSTIC, not a correctness gate.
    # The gate is the pairwise GAP drift (d_gap) and the final candidate
    # log-score drift — both at the FIXED 0.05 contract (no widening). A
    # common-mode raw shift leaves d_gap ~0; a real divergence moves the
    # gaps and fails the gate. measured on the 0.5B: d_raw 0.125 but
    # d_gap 0.070 — the drift is partly real (fails the gate, fail closed).
    n_ctx = max(context_counts)

    def _decomp(ref_vals, bat_vals):
        """Per-row raw-logit decomposition (review section 2)."""
        d_raw = max(abs(a - b) for a, b in zip(ref_vals, bat_vals, strict=True))
        d_gap = 0.0
        for i in range(len(ref_vals)):
            for j in range(len(ref_vals)):
                d_gap = max(d_gap, abs((bat_vals[i] - bat_vals[j]) - (ref_vals[i] - ref_vals[j])))
        ls_ref = _mlx_nn.log_softmax(_mx.array(ref_vals)).tolist()
        ls_bat = _mlx_nn.log_softmax(_mx.array(bat_vals)).tolist()
        d_logp = max(abs(a - b) for a, b in zip(ls_ref, ls_bat, strict=True))
        return d_raw, d_gap, d_logp

    def _top_two_margin(vals):
        s = sorted(vals, reverse=True)
        return s[0] - s[1] if len(s) >= 2 else float("inf")

    winners_identical = True
    max_abs_drift = 0.0
    max_raw_drift = 0.0
    max_gap_drift = 0.0
    max_logp_drift = 0.0
    max_margin_drift = 0.0
    max_merged_rows = 0  # W5c-9 C4: the largest merged pass M the matrix ran
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
        # W5c-1 review: decompose into d_raw (diagnostic) / d_gap (gate) /
        # d_logp (diagnostic) + margin drift (gate).
        raw_drift = gap_drift = logp_drift = margin_drift = 0.0
        if built["rows"]:
            vocab_size = engine.vocab_size
            from jevmlx.timing import Ledger

            pf = _prefill(
                engine.model, engine.tokenizer, context, schema, Ledger(), "slots", engine.profile
            )
            # Reference: the CANONICAL batch=1 shape (one row per forward) —
            # the same shape the near-tie rescore trusts.
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
            # The batched side runs n_ctx context-copies of the rows in
            # merged chunks (the decide_many row shape — one merged pass
            # per group), so raw batch drift shows against the batch=1
            # reference the near-tie rescore trusts.
            cache_slots = [pf.cache] * (n_ctx * len(built["rows"]))
            all_rows = built["rows"] * n_ctx
            all_decisions = built["row_decision"] * n_ctx
            batched = _score_rows(
                engine.model,
                pf.cache,
                all_rows,
                all_decisions,
                vocab_size,
                built["pad_id"],
                max(2, n_ctx * len(built["rows"])),
                Ledger(),
                cache_slots=cache_slots,
            )
            for ridx in range(len(built["rows"])):
                ref_vals = ref.row_logits.get(ridx)
                got_vals = batched.row_logits.get(ridx)
                if ref_vals is None or got_vals is None:
                    winners_identical = False
                    continue
                dr, dg, dl = _decomp(ref_vals, got_vals)
                raw_drift = max(raw_drift, dr)
                gap_drift = max(gap_drift, dg)
                logp_drift = max(logp_drift, dl)
                margin_drift = max(
                    margin_drift, abs(_top_two_margin(ref_vals) - _top_two_margin(got_vals))
                )
        max_raw_drift = max(max_raw_drift, raw_drift)
        max_gap_drift = max(max_gap_drift, gap_drift)
        max_logp_drift = max(max_logp_drift, logp_drift)
        max_margin_drift = max(max_margin_drift, margin_drift)
        max_merged_rows = max(max_merged_rows, n_ctx * len(built["rows"]))

        # --- Final decisions at 1/2/4 contexts, equal + mixed lengths.
        case_drift = 0.0
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
            # W5c-1 review: the gated decomposition (d_gap + margin drift)
            # and the diagnostics (d_raw + d_logp).
            "max_gap_drift_nats": gap_drift,
            "max_logp_drift_nats": logp_drift,
            "max_margin_drift_nats": margin_drift,
        }
        max_abs_drift = max(max_abs_drift, case_drift)

    return {
        "winners_identical": winners_identical,
        "max_abs_drift_nats": max_abs_drift,
        "max_raw_row_drift_nats": max_raw_drift,
        # W5c-1 review section 2: the GATE metrics — pairwise gap drift and
        # top-two margin drift, both at the FIXED 0.05 contract (no
        # widening). d_raw is a diagnostic only (raw logits carry an
        # arbitrary additive offset).
        "max_gap_drift_nats": max_gap_drift,
        "max_logp_drift_nats": max_logp_drift,
        "max_margin_drift_nats": max_margin_drift,
        # W5c-9 C4: the largest merged pass M the matrix ran (n_ctx * R
        # per case) — the bucket the persisted envelope record belongs to.
        "max_merged_rows": max_merged_rows,
        "per_case": per_case,
    }
