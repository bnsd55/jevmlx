"""Evaluation runner: batch cases through a decision track, emit per-field
prediction lines plus a run manifest.

One line per (case, field, permutation) in ``predictions.jsonl`` and a
``run.json`` manifest next to it — the shapes are the frozen eval-harness
contract. The runner itself is a thin loop around a per-track
``decide_fn(schema_dict, context) -> per-field results`` seam, so tests can
drive it entirely with fakes.

Tracks:
- ``parallel``: the jevmlx engine (``run_parallel_generation`` at T=1),
  log scores from field telemetry.
- ``naive_local``: the same local model free-writes the whole JSON object
  (``run_naive_generation``), parsed strictly by
  :func:`jevmlx.baseline.parse_baseline_output`.
- ``api_baseline``: an OpenAI-compatible chat API via
  :func:`jevmlx.baseline.baseline_decide` (product-comparison track).

Permutations (parallel track only) probe choice/field order sensitivity:
``rotations`` cycles every enum field's choices, ``fieldperm`` permutes the
field order; each permutation runs against a fresh schema and its
predictions are remapped back to canonical choice strings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import statistics
import time
from collections.abc import Callable
from typing import Any

from jevmlx.baseline import baseline_decide, parse_baseline_output
from jevmlx.evalreport import environment
from jevmlx.schema import StructuredSchema, _thaw_plan

__all__ = [
    "DecideFn",
    "api_baseline_decide_fn",
    "make_run_id",
    "openai_slots_decide_fn",
    "naive_local_decide_fn",
    "parallel_decide_fn",
    "rotations",
    "run_eval",
]

# decide_fn(schema_dict, context) -> {field: {"prediction", "valid", "error",
# "log_scores", "probability", "per_option", "type"}, "_meta": {...}}.
# W5c-3: parallel _meta carries per_item_end_to_end_ms (single and batched
# engine paths both report it); it rides every parallel prediction line.
DecideFn = Callable[[dict, str], dict[str, Any]]

logger = logging.getLogger(__name__)

# Frozen prediction-line contract: every line in predictions.jsonl carries
# exactly these keys (sorted). check_results.py imports this list instead of
# retyping the contract. (Optional keys perturbation/consensus are added by
# run_eval only when the case carries them; they are not in the frozen set.)
PREDICTION_LINE_KEYS: tuple[str, ...] = (
    "case_id",
    "correct",
    "error",
    "field",
    "group_id",
    "label",
    "latency_ms",
    "log_scores",
    "model",
    "per_item_end_to_end_ms",
    "per_option",
    "passes",
    "permutation",
    "prediction",
    "probability",
    "rows",
    "run_id",
    "salvage_prediction",
    "source",
    "track",
    "type",
    "valid",
    "workflow",
)

# run.json required top-level keys (config and counts are dicts).
RUN_REQUIRED_KEYS: tuple[str, ...] = ("run_id", "environment", "config", "counts")


def make_run_id() -> str:
    """Run id: UTC timestamp + short random suffix (sortable, collision-safe)."""
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.urandom(3).hex()}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _full_line_skeleton(
    run_id: str,
    case: dict,
    tag: str | None,
    track: str,
    model: str,
    *,
    field: str = "_error",
    error: str | None = None,
) -> dict[str, Any]:
    """A prediction-line-shaped dict for error/marker lines.

    Every key in PREDICTION_LINE_KEYS is present (None for irrelevant),
    so error/marker lines pass the frozen-contract check in check_results.
    Written to errors.jsonl (NOT predictions.jsonl) so they do not survive
    resume (R2).
    """
    return {
        "run_id": run_id,
        "case_id": case.get("id"),
        "group_id": case.get("group_id"),
        "source": case.get("source"),
        "workflow": case.get("workflow"),
        "field": field,
        "type": None,
        "track": track,
        "model": model,
        "permutation": tag or "canonical",
        "label": case.get("labels", {}).get(field),
        "prediction": None,
        "valid": False,
        "correct": None,
        "log_scores": None,
        "probability": None,
        "per_option": None,
        "latency_ms": None,
        "per_item_end_to_end_ms": None,
        "rows": None,
        "passes": None,
        "error": error,
        "salvage_prediction": None,
        "oracle_prediction": None,
    }


def _next_timing_segment(out_dir: str) -> int:
    """Glob timing.segment-*.json and return max+1, or 0 if none exist."""
    import glob

    segments = []
    for p in glob.glob(os.path.join(out_dir, "timing.segment-*.json")):
        try:
            n = int(os.path.basename(p).split("-")[1].split(".")[0])
            segments.append(n)
        except (ValueError, IndexError):
            continue
    return (max(segments) + 1) if segments else 0


def _sha256_file(path: str | None) -> str | None:
    """sha256 hex of the file at ``path``; None only when path is None.

    A non-None path that does not exist raises OSError — callers must not
    write null into run.json for a lock they claimed exists (W5c-17: the
    bench passed a lock path that was never written, and the silent None
    hid the broken provenance chain).
    """
    if not path:
        return None
    if not os.path.exists(path):
        raise OSError(f"dataset lock file not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_tokenizer_metrics(
    schema: StructuredSchema,
    plan_provider: Callable[..., dict],
    field_telemetry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Per-model tokenizer metrics for run.json (W4-A).

    Quantifies how the tokenizer segments the schema's quoted codes:
    - ``quoted_code_token_lengths``: list of candidate token lengths
      (shared_ids + remainder per choice) across all scalar fields.
    - ``single_branch_fraction``: fraction of scalar fields whose codebook
      search produced a single-branch trie (all choices diverge at one node).
    - ``max_trie_depth``: deepest branch-node path across all fields.
    - ``mean_legal_mass``: mean legal_mass from the first result's field
      telemetry (1.0 = no leakage; lower = model wanted off-schema tokens).
    """
    from jevmlx.trie import build_trie

    plan = plan_provider(schema)
    fields_plan = plan.get("fields")
    if not isinstance(fields_plan, dict):
        # Synthetic plan (e.g. tests pass {"compiled": True}); no token-level
        # structure to measure. Telemetry (legal_mass) may still be present.
        fields_plan = {}

    code_lengths: list[int] = []
    single_branch_count = 0
    scalar_count = 0
    max_depth = 0

    for fp in fields_plan.values():
        if "remainders" in fp:
            # Scalar field (enum/boolean).
            scalar_count += 1
            shared = fp["shared_ids"]
            if fp["single_branch"]:
                single_branch_count += 1
            for remainder in fp["remainders"]:
                code_lengths.append(len(shared) + len(remainder))
            for node in build_trie(fp["remainders"]):
                depth = len(node["path"])
                if depth > max_depth:
                    max_depth = depth
        elif "suffix_ids_list" in fp:
            # Multi field: each option pairs with its own Y/N remainders.
            for suffix, remainders in zip(fp["suffix_ids_list"], fp["remainders"], strict=True):
                # Y candidate: suffix + first Y remainder token length.
                code_lengths.append(len(suffix) + len(remainders[0]))

    mean_legal_mass = None
    if field_telemetry:
        masses = [
            ft["legal_mass"] for ft in field_telemetry.values() if ft["legal_mass"] is not None
        ]
        if masses:
            mean_legal_mass = sum(masses) / len(masses)

    return {
        "quoted_code_token_lengths": code_lengths,
        "quoted_code_token_length_mean": (
            sum(code_lengths) / len(code_lengths) if code_lengths else 0
        ),
        "single_branch_fraction": (single_branch_count / scalar_count if scalar_count else 0),
        "max_trie_depth": max_depth,
        "mean_legal_mass": mean_legal_mass,
    }


def parallel_decide_fn(engine, scoring: str = "slots", prior_correction: bool = False) -> DecideFn:
    """Track ``parallel``: the jevmlx engine at T=1.

    Takes the loaded :class:`Engine`.

    Log scores come from field telemetry's finalized ``log_scores`` key — a
    dict mapping choice to constrained-path log P at T=1 (prior-corrected
    and renormalised when ``prior_correction`` is set). Multi fields report
    ``per_option`` instead; log_scores stays None for them.
    """

    def decide(
        schema_dict: dict, context: str, constraints=None, oracle_overrides=None
    ) -> dict[str, dict[str, Any]]:
        from jevmlx.engine import run_parallel_generation

        schema = StructuredSchema(schema_dict)
        result = run_parallel_generation(
            engine,
            context,
            schema,
            temperature=1.0,
            scoring=scoring,
            prior_correction=prior_correction,
            constraints=constraints,
            oracle_overrides=oracle_overrides,
        )
        out: dict[str, dict[str, Any]] = {}
        for fname, telemetry in result["field_telemetry"].items():
            field = schema.fields.get(fname)
            entry: dict[str, Any] = {
                "prediction": telemetry["value"],
                "probability": telemetry.get("probability"),
                "per_option": telemetry.get("per_option"),
                "type": telemetry.get("type") or (field.field_type if field else None),
            }
            # W6-B1: ordered enums carry the scale (choice order = level
            # order) + the derived ordinal telemetry, so ordinal metrics can
            # be computed from the predictions file alone.
            if field is not None and field.ordered:
                entry["ordered"] = True
                entry["ordinal_choices"] = list(field.choices)
                ord_record = telemetry.get("ordinal")
                if isinstance(ord_record, dict):
                    entry["ordinal"] = ord_record
            if field is not None and field.field_type != "multi":
                raw = telemetry.get("log_scores")
                if isinstance(raw, dict):
                    entry["log_scores"] = raw
            out[fname] = entry
        out["_meta"] = {
            "latency_ms": result["elapsed_ms"],
            # W3-R: the engine's full timing split rides _meta so bench
            # writes the timing JSON per combo from the SAME decide calls
            # (no second run). Strict keys: the engine always sets these —
            # a missing key must be a visible bug (KeyError), not a silent
            # zero in timing.json (review F2).
            "prior_ms": result["prior_ms"],
            "prefill_ms": result["prefill_ms"],
            "plan_compile_ms": result["plan_compile_ms"],
            "cache_broadcast_ms": result["cache_broadcast_ms"],
            "suffix_eval_ms": result["suffix_eval_ms"],
            "lm_head_gather_ms": result["lm_head_gather_ms"],
            "second_pass_ms": result["second_pass_ms"],
            "total_ms": result["total_ms"],
            # W5c-3: the single path's honest per-request latency — the same
            # key the batched path reports per context (results contract v2
            # must not depend on which engine path produced the lines).
            "per_item_end_to_end_ms": result["per_item_end_to_end_ms"],
            # Results contract v2 (W5-D findings 30/32): retry count and
            # request-scoped peak ride the split so timing.json shows them
            # next to the absolute peak.
            "failed_attempts": result["failed_attempts"],
            "peak_active_bytes": result["peak_active_bytes"],
            "peak_incremental_bytes": result["peak_incremental_bytes"],
            "padded_token_positions": result["padded_token_positions"],
            # W5c-6 / B4: token-accounting telemetry in timing.json.
            "naive_branch_prompt_tokens": result["naive_branch_prompt_tokens"],
            "shared_prefix_tokens": result["shared_prefix_tokens"],
            "logical_suffix_token_positions": result["logical_suffix_token_positions"],
            "computed_suffix_token_positions": result["computed_suffix_token_positions"],
            "computed_prompt_token_positions": result["computed_prompt_token_positions"],
            "retry_wasted_ms": result["retry_wasted_ms"],
            "rescored_fields_count": len(result["rescored_fields"]),
            "rerun_fields_count": len(result["rerun_fields"]),
            "num_fields": result["num_fields"],
        }
        return out

    return decide


def _attach_ordinal(field, entry: dict) -> None:
    """W6-B1/F5: every track's decide_fn output carries the ordered scale +
    the derived ordinal telemetry, so ordinal_mae/confusion compute on
    EVERY track (no tolerated-absence path).

    The parallel track reads the engine's distribution-derived record from
    its result telemetry. The non-parallel tracks (naive_local,
    api_baseline, openai_slots) free-write or call out — they have no
    distribution, only the hard prediction. For those, derive a REAL hard
    record: argmax_level = index of the predicted level in the declared
    scale order, expected_index = argmax_level (no soft evidence — the hard
    pick IS the distribution's entire mass), variance 0.0,
    expected_score_normalized = argmax/(n-1), tagged ``source="hard"`` so
    ordinal_mae_expected skips it EXPLICITLY by source (the soft metric is
    undefined on a hard-only record), never by an absent key. An invalid
    (off-scale) prediction still emits the record with a worst-distance
    argmax (len-1), matching the invalid-is-wrong rule.
    """
    if field is None or not field.ordered:
        return
    choices = list(field.choices)
    prediction = entry.get("prediction")
    entry["ordered"] = True
    entry["ordinal_choices"] = choices
    # Index of the predicted level; invalid => worst distance (len-1),
    # the same rule ordinal_mae applies to the hard argmax.
    try:
        argmax = choices.index(str(prediction))
    except (ValueError, TypeError):
        argmax = len(choices) - 1
    n = len(choices)
    entry["ordinal"] = {
        "argmax_level": argmax,
        "expected_index": float(argmax),
        "variance": 0.0,
        "expected_score_normalized": (argmax / (n - 1)) if n > 1 else 1.0,
        "source": "hard",
    }


def naive_local_decide_fn(engine) -> DecideFn:
    """Track ``naive_local``: the same local model free-writes the JSON object.
    Takes the loaded :class:`Engine`.

    Output is parsed strictly by :func:`jevmlx.baseline.parse_baseline_output`;
    unsalvageable fields become invalid predictions (a measurement, not a
    crash).
    """

    def decide(
        schema_dict: dict, context: str, constraints=None, oracle_overrides=None
    ) -> dict[str, dict[str, Any]]:
        from jevmlx.engine import run_naive_generation

        schema = StructuredSchema(schema_dict)
        result = run_naive_generation(engine, context, schema)
        strict_values, salvage_values, errors = parse_baseline_output(
            result.get("raw_text", ""), schema
        )
        out: dict[str, dict[str, Any]] = {}
        for fname in schema.fields:
            strict = strict_values.get(fname)
            entry: dict[str, Any] = {
                "prediction": strict,
                "valid": strict is not None,
                "salvage_prediction": salvage_values.get(fname),
                "error": next((e for e in errors if fname in e), None),
            }
            # W6-B1/F5: ordered enums carry the scale on EVERY track.
            _attach_ordinal(schema.fields.get(fname), entry)
            out[fname] = entry
        out["_meta"] = {
            "latency_ms": result.get("elapsed_ms"),
            "rows": None,
            "passes": result.get("total_tokens"),
        }
        return out

    return decide


def api_baseline_decide_fn(
    base_url: str, model: str, api_key: str | None
) -> tuple[DecideFn, dict[str, str]]:
    """Track ``api_baseline``: an OpenAI-compatible chat API decides.

    Returns the decide_fn plus the request parameters to record in run.json.
    """

    def decide(
        schema_dict: dict, context: str, constraints=None, oracle_overrides=None
    ) -> dict[str, dict[str, Any]]:
        schema = StructuredSchema(schema_dict)
        result = baseline_decide(base_url, model, api_key, schema, context)
        out: dict[str, dict[str, Any]] = {}
        for fname in schema.fields:
            value = result["values"].get(fname)
            entry: dict[str, Any] = {
                "prediction": value,
                "valid": value is not None,
                "error": next((e for e in result["errors"] if fname in e), None),
            }
            # W6-B1/F5: ordered enums carry the scale on EVERY track.
            _attach_ordinal(schema.fields.get(fname), entry)
            out[fname] = entry
        out["_meta"] = {
            "latency_ms": result.get("latency_ms"),
            "rows": None,
            "passes": None,
        }
        return out

    return decide, {"api_base": base_url, "api_model": model}


def openai_slots_decide_fn(
    base_url: str, model: str, api_key: str | None, timeout: float = 120.0
) -> tuple[DecideFn, dict[str, str]]:
    """Track ``openai_slots``: the slots decision semantics via an API.

    Same result shape as the parallel track (log_scores, probability,
    alternatives) with ``confidence_model="openai_slots"``; one request per
    field. Returns the decide_fn plus the request parameters for run.json.
    """

    def decide(
        schema_dict: dict, context: str, constraints=None, oracle_overrides=None
    ) -> dict[str, dict[str, Any]]:
        from jevmlx.openai_slots import decide_openai

        schema = StructuredSchema(schema_dict)
        result = decide_openai(base_url, model, api_key, schema, context, timeout=timeout)
        out: dict[str, dict[str, Any]] = {}
        for fname, telemetry in result["field_telemetry"].items():
            field = schema.fields.get(fname)
            entry = {
                "prediction": telemetry["value"],
                "probability": telemetry.get("probability"),
                "per_option": telemetry.get("per_option"),
                "type": telemetry.get("type") or (field.field_type if field else None),
                "log_scores": telemetry.get("log_scores"),
                "truncated": telemetry.get("truncated"),
            }
            # W6-B1/F5: ordered enums carry the scale on EVERY track.
            _attach_ordinal(field, entry)
            out[fname] = entry
        out["_meta"] = {
            "latency_ms": result.get("elapsed_ms"),
            "rows": None,
            "passes": result.get("sequential_forward_passes"),
        }
        return out

    return decide, {"api_base": base_url, "api_model": model, "track_kind": "openai_slots"}


def rotations(choices: list[str]) -> list[list[str]]:
    """Cyclic rotations of a choice list: every k in 1..n-1 for n <= 8, else 8
    seeded rotations (deterministic for a given choice list)."""
    n = len(choices)
    if n <= 8:
        return [choices[k:] + choices[:k] for k in range(1, n)]
    rng = random.Random(0)
    return [choices[k:] + choices[:k] for k in sorted(rng.sample(range(1, n), 7))]


def _field_permutations(field_names: list[str], count: int = 3) -> list[list[str]]:
    """``count`` seeded non-identity permutations of the field order."""
    rng = random.Random(0)
    perms: list[list[str]] = []
    while len(perms) < count:
        candidate = field_names[:]
        rng.shuffle(candidate)
        if candidate != field_names and candidate not in perms:
            perms.append(candidate)
    return perms


def _heartbeat(
    combo: str,
    done: int,
    pred_lines: int,
    start: float,
    out_dir: str,
) -> None:
    """W5c-16: print + append one heartbeat record for every N completed cases.

    One line to stdout (the same GB formatting as the bench [memory] line,
    reusing the local _sample_metal_memory + _memory_block_gb) and one
    JSON object appended to ``<out_dir>/heartbeat.jsonl`` (machine-readable).
    Best-effort: a Metal read failure yields -1 and never breaks the run.
    """
    elapsed = int(time.perf_counter() - start)
    mem = _sample_metal_memory()
    print(
        f"[heartbeat] {combo} cases_done={done} pred_lines={pred_lines} "
        f"elapsed_s={elapsed} {_memory_block_gb(mem)}",
        flush=True,
    )
    rec = {
        "combo": combo,
        "cases_done": done,
        "pred_lines": pred_lines,
        "elapsed_s": elapsed,
        "peak_memory_bytes": mem["peak_memory"],
        "active_memory_bytes": mem["active_memory"],
        "cache_memory_bytes": mem["cache_memory"],
    }
    try:
        with open(os.path.join(out_dir, "heartbeat.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"[heartbeat] could not append heartbeat.jsonl: {exc}", flush=True)


# W5c-13/16: Metal memory helpers. Defined here (not in jevmlx.bench) so the
# eval infra never imports from the bench CLI; bench imports these from here.
_MEMORY_KEYS = ("peak_memory", "active_memory", "cache_memory")


def _sample_metal_memory() -> dict[str, int]:
    """Sample the three Metal memory counters (bytes), -1 when unreadable.

    - peak_memory:   mx.get_peak_memory() — the high-water mark since
                      the last reset_peak_memory() (reset at combo start).
    - active_memory: mx.get_active_memory() — buffers currently held.
    - cache_memory:  mx.get_cache_memory() — the buffer cache Metal
                      keeps after frees (the #79 root cause: not returned to
                      macOS until clear_cache()).
    """
    out: dict[str, int] = {k: -1 for k in _MEMORY_KEYS}
    try:
        import mlx.core as mx

        out["peak_memory"] = int(mx.get_peak_memory())
        out["active_memory"] = int(mx.get_active_memory())
        out["cache_memory"] = int(mx.get_cache_memory())
    except Exception:  # noqa: BLE001 - telemetry must never break the run
        pass
    return out


def _memory_block_gb(mem: dict[str, int]) -> str:
    """A one-line bench-log string of the memory block in GB."""

    def _gb(v: int) -> str:
        return f"{v / 2**30:.2f} GB" if v >= 0 else "n/a"

    return (
        f"peak={_gb(mem['peak_memory'])} "
        f"active={_gb(mem['active_memory'])} "
        f"cache={_gb(mem['cache_memory'])}"
    )


def _schema_variants(
    case: dict, schema: StructuredSchema, mode: str
) -> list[tuple[str | None, dict]]:
    """(permutation tag, schema dict) variants for one case.

    ``canonical`` (tag None) first, then the tagged permutations: every cyclic
    rotation of every enum field's choices for ``rotations``/``all``, and 3
    seeded field-order permutations for ``fieldperm``/``all``.
    """
    variants: list[tuple[str | None, dict]] = [(None, case["schema"])]
    if mode in ("rotations", "all"):
        for fname, fdef in schema.fields.items():
            if fdef.field_type != "enum" or len(fdef.choices) < 2:
                continue
            for k, perm in enumerate(rotations(fdef.choices), start=1):
                variant = json.loads(json.dumps(case["schema"]))
                variant[fname]["choices"] = perm
                variants.append((f"rot{k}:{fname}", variant))
    if mode in ("fieldperm", "all"):
        base = case["schema"]
        for k, order in enumerate(_field_permutations(list(schema.fields)), start=1):
            variants.append((f"fieldperm{k}", {name: base[name] for name in order}))
    return variants


def run_eval(
    cases: list[dict],
    decide_fn: DecideFn,
    *,
    track: str,
    model: str,
    permutations: str = "none",
    split: str = "all",
    out_dir: str,
    run_id: str | None = None,
    extra_config: dict | None = None,
    chat_template: str | None = None,
    plan_provider: Callable[..., dict] | None = None,
    dataset_lock_path: str | None = None,
    dataset_path: str | None = None,
    carry_perturbation: bool = False,
    carry_consensus: bool = False,
    resume: bool = False,
    heartbeat_every: int = 0,
    combo: str = "",
) -> dict:
    """Run the batch and write ``predictions.jsonl`` + ``run.json`` into out_dir.

    ``dataset_lock_path`` is the dataset lock produced next to the cases
    file; run.json's ``dataset_lock_sha256`` is the sha256 of THAT lock —
    the provenance anchor for leaderboard rows. A non-None path that does
    not exist is an ERROR (OSError), never a silent null: a missing lock
    means the dataset was built by a stale builder and the sha would be
    untraceable.

    ``decide_fn(schema_dict, context) -> per-field results`` is the seam: each
    per-field result carries ``prediction``, optionally ``valid``/``error``/
    ``log_scores``/``probability``/``per_option``; a ``"_meta"`` entry carries
    run-level ``latency_ms``/``rows``/``passes``. Returns the run manifest.

    With ``carry_perturbation=True`` each prediction line also carries
    ``"perturbation"``: the case's ``meta.perturbation`` (a string such as
    ``"ws"`` or ``"shuffle3"``) or null for original cases. Pairs of lines
    sharing a ``group_id`` (original vs variant) feed
    :func:`jevmlx.evalmetrics.perturbation_flip_rate`.

    With ``carry_consensus=True`` each prediction line whose case carries a
    ``meta.consensus[field]`` distribution (TypeSafe / typed-decisions
    fetchers) gets a ``"consensus"`` key with that field's ``{choice: p}``
    dict, feeding :func:`jevmlx.evalmetrics.tvd_vs_consensus`.
    """
    run_id = run_id or make_run_id()
    os.makedirs(out_dir, exist_ok=True)

    # W5c-7: crash-safe resumable eval infrastructure.
    from datetime import UTC, datetime

    from jevmlx.resume import (
        CircuitBreaker,
        ManifestMismatchError,
        ResultsLock,
        build_manifest,
        completed_case_keys,
        failure_signature,
        load_manifest,
        machine_probe,
        truncate_to_last_commit,
        verify_manifest_matches,
        write_case_blob,
        write_manifest,
    )
    from jevmlx.resume import (
        case_key as _case_key,
    )

    _code_hash = {
        "jevmlx_evalrun": _sha256_text(open(__file__, encoding="utf-8").read()),
    }
    _machine = machine_probe()
    _model_revision = None
    _tokenizer_revision = None
    _prompt_version = None
    if track == "parallel":
        try:
            from jevmlx.engine import PROMPT_VERSION, engine_metadata

            _metadata = engine_metadata(model)
            _model_revision = _metadata["revision"]
            _tokenizer_revision = _metadata.get("revision")
            _prompt_version = PROMPT_VERSION
        except Exception:  # noqa: BLE001
            pass
    manifest = build_manifest(
        config={
            "model": model,
            "temperature": 1.0,
            "track": track,
            "permutations": permutations if track == "parallel" else "none",
            "split": split,
            "prompt_version": _prompt_version,
        },
        dataset_lock_sha256=_sha256_file(dataset_lock_path),
        prompt_version=_prompt_version,
        prompt_sha256=None,
        model_id=model,
        model_revision=_model_revision,
        tokenizer_revision=_tokenizer_revision,
        tokenizer_chat_template_sha256=(_sha256_text(chat_template) if chat_template else None),
        machine=_machine,
        code_hash=_code_hash,
        run_id=run_id,
    )
    # N2: sessions list — durable per-session record for the segment counter.
    manifest["sessions"] = [
        {"segment": 0, "started_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    ]
    _resume_keys: set[str] = set()
    _segment = 0
    _run_id_for_run = run_id
    _stored_prompt_sha: str | None = None
    if resume:
        existing_manifest = load_manifest(out_dir)
        if existing_manifest is None:
            raise RuntimeError(
                f"--resume requested but no manifest.json in {out_dir} "
                "(the run was never started). Re-run without --resume."
            )
        verify_manifest_matches(
            existing_manifest, manifest, skip_fields=("prompt_sha256", "sessions")
        )
        _stored_prompt_sha = existing_manifest.get("prompt_sha256")
        _resume_keys = completed_case_keys(out_dir)
        _run_id_for_run = existing_manifest.get("run_id", run_id)
        with ResultsLock(out_dir):
            truncate_to_last_commit(out_dir)
            # R1: mutate the EXISTING manifest (which has the stored
            # prompt_sha256) — never write the new manifest on resume
            # (it has prompt_sha256=None and would overwrite the sha).
            _sessions = existing_manifest.get("sessions", [])
            _segment = len(_sessions)
            existing_manifest["sessions"] = _sessions + [
                {
                    "segment": _segment,
                    "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            ]
            write_manifest(out_dir, existing_manifest)
            manifest = existing_manifest
    else:
        # S4: refuse to start fresh into a folder that already has
        # predictions.jsonl (would double the file). Use --resume.
        _pred_path = os.path.join(out_dir, "predictions.jsonl")
        if os.path.exists(_pred_path):
            raise RuntimeError(
                f"{out_dir}/predictions.jsonl already exists. "
                "Use --resume to continue, or remove the directory."
            )
        write_manifest(out_dir, manifest)

    selected = [c for c in cases if split == "all" or c.get("split", "train") == split]

    timing_meta: list[dict[str, Any]] = []  # W3-R: parallel _meta timings
    lines: list[dict] = []
    n_canonical = 0
    first_field_telemetry: dict[str, Any] | None = None
    _breaker = CircuitBreaker()
    _circuit_tripped: str | None = None
    # W5c-16: heartbeat counters (cases committed + run start time).
    _hb_done = 0
    _hb_start = time.perf_counter()
    with ResultsLock(out_dir):
        for case in selected:
            schema = StructuredSchema(case["schema"])
            variants = (
                _schema_variants(case, schema, permutations)
                if track == "parallel" and permutations != "none"
                else [(None, case["schema"])]
            )
            for tag, schema_dict in variants:
                _ckey = _case_key(case, tag or "canonical")
                if _ckey in _resume_keys:
                    # Already committed in a prior session — skip.
                    continue
                variant_lines: list[dict] = []
                marker_lines: list[dict] = []
                if _circuit_tripped is not None:
                    # Breaker tripped: mark remaining variants and skip.
                    marker_lines.append(
                        _full_line_skeleton(
                            _run_id_for_run,
                            case,
                            tag,
                            track,
                            model,
                            field="_skipped",
                            error=f"circuit_breaker_tripped:{_circuit_tripped}",
                        )
                    )
                    # R2: markers go to errors.jsonl, NOT predictions.jsonl.
                    _errors_path = os.path.join(out_dir, "errors.jsonl")
                    _blob = "".join(json.dumps(ml, sort_keys=True) + "\n" for ml in marker_lines)
                    with open(_errors_path, "a", encoding="utf-8") as _mf:
                        _mf.write(_blob)
                        _mf.flush()
                        os.fsync(_mf.fileno())
                    continue
                try:
                    results = decide_fn(
                        schema_dict, case["context"], constraints=case.get("constraints")
                    )
                except TypeError:
                    results = decide_fn(schema_dict, case["context"])
                except Exception as exc:  # noqa: BLE001 — circuit breaker
                    sig = failure_signature(exc)
                    # P2/P3: write the error line (full shape) for BOTH infra
                    # and non-infra — the failure IS the result.
                    variant_lines.append(
                        _full_line_skeleton(
                            _run_id_for_run,
                            case,
                            tag,
                            track,
                            model,
                            field="_error",
                            error=f"{type(exc).__name__}:{exc}",
                        )
                    )
                    lines.extend(variant_lines)
                    # R2: infra failures go to errors.jsonl, NOT predictions.jsonl
                    # (they must not survive resume). Validation errors (sig
                    # is None) ARE journaled via write_case_blob — they are
                    # permanent results.
                    if sig is None:
                        write_case_blob(
                            os.path.join(out_dir, "predictions.jsonl"),
                            os.path.join(out_dir, "completed_cases.jsonl"),
                            _ckey,
                            variant_lines,
                        )
                        # W5c-16: heartbeat every N committed cases.
                        _hb_done += 1
                        if heartbeat_every and _hb_done % heartbeat_every == 0:
                            _heartbeat(combo, _hb_done, len(lines), _hb_start, out_dir)
                    else:
                        _errors_path = os.path.join(out_dir, "errors.jsonl")
                        _blob = "".join(
                            json.dumps(vl, sort_keys=True) + "\n" for vl in variant_lines
                        )
                        with open(_errors_path, "a", encoding="utf-8") as _ef:
                            _ef.write(_blob)
                            _ef.flush()
                            os.fsync(_ef.fileno())
                    if sig is not None and _breaker.record(exc):
                        _circuit_tripped = sig
                        logger.error(
                            "circuit breaker tripped (%s) after %d consecutive "
                            "infra failures; stopping model %s",
                            sig,
                            _breaker._consecutive.get(sig, 0),
                            model,
                        )
                    continue
                _breaker.record_success()
                meta = results.pop("_meta", {})
                # W3-R: parallel-track _meta carries the engine's timing split.
                if track == "parallel" and tag is None and meta:
                    timing_meta.append(meta)
                if tag is None and first_field_telemetry is None and meta.get("field_telemetry"):
                    first_field_telemetry = meta["field_telemetry"]
                if tag is None:
                    n_canonical += len(results)
                # R1: verify prompt_sha256 on resume (manifest IS existing_manifest,
                # has the stored sha). Fill from first case on fresh run.
                _prompt_sha = meta.get("prompt_sha256")
                if resume and _prompt_sha is not None:
                    if _prompt_sha != manifest.get("prompt_sha256"):
                        raise ManifestMismatchError(
                            {"prompt_sha256": (manifest.get("prompt_sha256"), _prompt_sha)}
                        )
                elif (
                    not resume and _prompt_sha is not None and manifest.get("prompt_sha256") is None
                ):
                    manifest["prompt_sha256"] = _prompt_sha
                    write_manifest(out_dir, manifest)
                # W3-D part 2: oracle pass.
                oracle_results: dict[str, Any] = {}
                if track == "parallel" and tag is None:
                    labels = case.get("labels", {})
                    has_depends = any(
                        isinstance(spec, dict) and spec.get("depends_on")
                        for spec in schema_dict.values()
                    )
                    if has_depends and labels:
                        oracle_parents: dict[str, object] = {}
                        for spec in schema_dict.values():
                            if isinstance(spec, dict) and spec.get("depends_on"):
                                parent = spec["depends_on"]
                                if parent in labels:
                                    oracle_parents[parent] = labels[parent]
                        if oracle_parents:
                            try:
                                oracle_out = decide_fn(
                                    schema_dict,
                                    case["context"],
                                    constraints=case.get("constraints"),
                                    oracle_overrides=oracle_parents,
                                )
                                oracle_out.pop("_meta", None)
                                for ofname, ores in oracle_out.items():
                                    if "oracle_prediction" in ores:
                                        oracle_results[ofname] = ores["oracle_prediction"]
                            except TypeError:
                                pass
                for fname, res in results.items():
                    field_def = schema.fields.get(fname)
                    prediction = res.get("prediction")
                    # W6-B1: ordered-enum lines carry the scale + telemetry.
                    ordinal_choices = res.get("ordinal_choices")
                    ordinal_record = res.get("ordinal")
                    if tag and tag.startswith("rot") and field_def is not None:
                        if isinstance(prediction, str) and prediction not in field_def.choices:
                            prediction = None
                    label = case.get("labels", {}).get(fname)
                    line = {
                        "run_id": _run_id_for_run,
                        "case_id": case.get("id"),
                        "group_id": case.get("group_id"),
                        "source": case.get("source"),
                        "workflow": case.get("workflow"),
                        "field": fname,
                        "type": res.get("type") or (field_def.field_type if field_def else None),
                        "track": track,
                        "model": model,
                        "permutation": tag or "canonical",
                        "label": label,
                        "prediction": prediction,
                        "valid": bool(res.get("valid", prediction is not None)),
                        "correct": None
                        if label is None
                        else (prediction is not None and prediction == label),
                        "log_scores": res.get("log_scores"),
                        "probability": res.get("probability"),
                        "per_option": res.get("per_option"),
                        "latency_ms": meta.get("latency_ms"),
                        "per_item_end_to_end_ms": meta.get("per_item_end_to_end_ms"),
                        "rows": meta.get("rows"),
                        "passes": meta.get("passes"),
                        "error": res.get("error"),
                        "salvage_prediction": res.get("salvage_prediction"),
                        "oracle_prediction": oracle_results.get(fname),
                    }
                    if ordinal_choices:
                        line["ordinal_choices"] = ordinal_choices
                        line["ordinal"] = ordinal_record
                    if carry_perturbation:
                        line["perturbation"] = (case.get("meta") or {}).get("perturbation")
                    if carry_consensus:
                        consensus = ((case.get("meta") or {}).get("consensus") or {}).get(fname)
                        if isinstance(consensus, dict) and consensus:
                            line["consensus"] = consensus
                    constraints = case.get("constraints")
                    if isinstance(constraints, list) and constraints:
                        line["constraints"] = constraints
                    variant_lines.append(line)
                    # B1: commit this variant's lines with its own journal key.
                if variant_lines:
                    write_case_blob(
                        os.path.join(out_dir, "predictions.jsonl"),
                        os.path.join(out_dir, "completed_cases.jsonl"),
                        _ckey,
                        variant_lines,
                    )
                    lines.extend(variant_lines)
                    # W5c-16: heartbeat every N committed cases.
                    _hb_done += 1
                    if heartbeat_every and _hb_done % heartbeat_every == 0:
                        _heartbeat(combo, _hb_done, len(lines), _hb_start, out_dir)

    config: dict[str, Any] = {
        "model": model,
        "temperature": 1.0,
        "track": track,
        "dataset_path": dataset_path,
        "permutations": permutations if track == "parallel" else "none",
        "split": split,
        # Provenance (X2): engine metadata + prompt version. Populated for the
        # parallel track; other tracks leave them None.
        "model_revision": None,
        "quantization": None,
        "prompt_version": None,
    }
    if track == "parallel" and selected:
        from jevmlx.engine import PROMPT_VERSION, engine_metadata

        metadata = engine_metadata(config["model"])
        config["model_revision"] = metadata["revision"]
        config["quantization"] = metadata["quantization"]
        config["prompt_version"] = PROMPT_VERSION
    if extra_config:
        config.update(extra_config)
    if chat_template is not None:
        config["tokenizer_chat_template_sha256"] = _sha256_text(chat_template)
    if plan_provider is not None and selected:
        # W5b-1 (C7): compiled plans are read-only mappings; hash the CONTENT
        # (thawed via schema._thaw_plan), never the frozen container —
        # json.dumps would otherwise raise on MappingProxyType (or, with
        # default=list, collapse every mapping to its key list and hash
        # every plan identically).
        plan = plan_provider(StructuredSchema(selected[0]["schema"]))
        plan_json = json.dumps(_thaw_plan(plan), sort_keys=True)
        config["compiled_plan_sha256"] = hashlib.sha256(plan_json.encode()).hexdigest()

    # W4-A: per-model tokenizer metrics — quantified how the tokenizer
    # segments the schema's quoted codes. These are model-specific (different
    # tokenizers split JSON differently) and feed the compatibility table.
    if track == "parallel" and plan_provider is not None and selected:
        config["tokenizer_metrics"] = _compute_tokenizer_metrics(
            StructuredSchema(selected[0]["schema"]),
            plan_provider,
            first_field_telemetry,
        )

    config["dataset_lock_sha256"] = _sha256_file(dataset_lock_path)

    # On resume, count ALL prediction lines in the file (including prior
    # sessions). On fresh run, just len(lines).
    _total_lines = len(lines)
    if resume:
        _pred_path = os.path.join(out_dir, "predictions.jsonl")
        if os.path.exists(_pred_path):
            with open(_pred_path, encoding="utf-8") as _pf:
                _total_lines = sum(1 for _line in _pf if _line.strip())

    run = {
        "run_id": _run_id_for_run,
        "environment": environment(),
        "config": config,
        "counts": {
            "cases": len(selected),
            "fields": n_canonical,
            "prediction_lines": _total_lines,
        },
        "circuit_breaker": _circuit_tripped,
    }

    # predictions.jsonl is written per-case via write_case_blob (above);
    # do NOT overwrite it here.
    if timing_meta:
        timing_summary: dict[str, float] = {}
        for key in timing_meta[0]:
            values = [m[key] for m in timing_meta if key in m]
            if values and all(isinstance(v, (int, float)) for v in values):
                timing_summary[key] = round(statistics.median(values), 4)
        _timing_payload = {
            "calls": len(timing_meta),
            "median": timing_summary,
            "segment": _segment,
            "segment_id": _run_id_for_run,
        }
        # Write timing.segment-N.json (per-session, for resume) AND
        # timing.json (backward compat for check_results/bench).
        _timing_path = os.path.join(out_dir, f"timing.segment-{_segment}.json")
        with open(_timing_path, "w", encoding="utf-8") as f:
            json.dump(_timing_payload, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
            json.dump(_timing_payload, f, indent=2, sort_keys=True)
            f.write("\n")
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(
        "eval run %s: %d cases, %d prediction lines -> %s",
        _run_id_for_run,
        len(selected),
        _total_lines,
        out_dir,
    )
    return run


def load_cases(path: str) -> list[dict]:
    """Load the cases JSONL (one JSON object per line, # comments skipped)."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(json.loads(line))
    return cases
