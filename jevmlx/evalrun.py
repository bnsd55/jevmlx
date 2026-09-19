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


def _sha256_file(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
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


def parallel_decide_fn(
    model, tokenizer, scoring: str = "slots", prior_correction: bool = False
) -> DecideFn:
    """Track ``parallel``: the jevmlx engine at T=1.

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
            model,
            tokenizer,
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
            # Results contract v2 (W5-D findings 30/32): retry count and
            # request-scoped peak ride the split so timing.json shows them
            # next to the absolute peak.
            "failed_attempts": result["failed_attempts"],
            "peak_active_bytes": result["peak_active_bytes"],
            "peak_incremental_bytes": result["peak_incremental_bytes"],
            "padded_token_positions": result["padded_token_positions"],
            "rescored_fields_count": len(result["rescored_fields"]),
            "rerun_fields_count": len(result["rerun_fields"]),
            "num_fields": result["num_fields"],
        }
        return out

    return decide


def naive_local_decide_fn(model, tokenizer) -> DecideFn:
    """Track ``naive_local``: the same local model free-writes the JSON object.

    Output is parsed strictly by :func:`jevmlx.baseline.parse_baseline_output`;
    unsalvageable fields become invalid predictions (a measurement, not a
    crash).
    """

    def decide(
        schema_dict: dict, context: str, constraints=None, oracle_overrides=None
    ) -> dict[str, dict[str, Any]]:
        from jevmlx.engine import run_naive_generation

        schema = StructuredSchema(schema_dict)
        result = run_naive_generation(model, tokenizer, context, schema)
        strict_values, salvage_values, errors = parse_baseline_output(
            result.get("raw_text", ""), schema
        )
        out: dict[str, dict[str, Any]] = {}
        for fname in schema.fields:
            strict = strict_values.get(fname)
            out[fname] = {
                "prediction": strict,
                "valid": strict is not None,
                "salvage_prediction": salvage_values.get(fname),
                "error": next((e for e in errors if fname in e), None),
            }
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
            out[fname] = {
                "prediction": value,
                "valid": value is not None,
                "error": next((e for e in result["errors"] if fname in e), None),
            }
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
            out[fname] = {
                "prediction": telemetry["value"],
                "probability": telemetry.get("probability"),
                "per_option": telemetry.get("per_option"),
                "type": telemetry.get("type") or (field.field_type if field else None),
                "log_scores": telemetry.get("log_scores"),
                "truncated": telemetry.get("truncated"),
            }
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
) -> dict:
    """Run the batch and write ``predictions.jsonl`` + ``run.json`` into out_dir.

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
    ``meta.consensus`` distribution (TypeSafe fetcher) gets a ``"consensus"``
    key with that dict, feeding :func:`jevmlx.evalmetrics.tvd_vs_consensus`.
    """
    run_id = run_id or make_run_id()
    os.makedirs(out_dir, exist_ok=True)

    selected = [c for c in cases if split == "all" or c.get("split", "train") == split]

    timing_meta: list[dict[str, Any]] = []  # W3-R: parallel _meta timings
    lines: list[dict] = []
    n_canonical = 0
    # Capture the first parallel result's field_telemetry for tokenizer metrics
    # (legal_mass, rows). Only the first case is needed — the schema is the
    # same across cases in a run; later cases would add noise, not signal.
    first_field_telemetry: dict[str, Any] | None = None
    for case in selected:
        schema = StructuredSchema(case["schema"])
        variants = (
            _schema_variants(case, schema, permutations)
            if track == "parallel" and permutations != "none"
            else [(None, case["schema"])]
        )
        for tag, schema_dict in variants:
            try:
                results = decide_fn(
                    schema_dict, case["context"], constraints=case.get("constraints")
                )
            except TypeError:
                # decide_fn doesn't accept constraints (mock/baseline) —
                # call without it (constraints only apply to the parallel track).
                results = decide_fn(schema_dict, case["context"])
            meta = results.pop("_meta", {})
            # W3-R: parallel-track _meta carries the engine's timing split —
            # collect per-case for the combo timing.json (median over cases).
            if track == "parallel" and tag is None and meta:
                timing_meta.append(meta)
            if tag is None and first_field_telemetry is None and meta.get("field_telemetry"):
                first_field_telemetry = meta["field_telemetry"]
            if tag is None:
                n_canonical += len(results)

            # W3-D part 2: oracle pass — teacher-force the TRUE parent for
            # fields with depends_on, so oracle_parent_gap can be computed.
            oracle_results: dict[str, Any] = {}
            if track == "parallel" and tag is None:
                labels = case.get("labels", {})
                has_depends = any(
                    isinstance(spec, dict) and spec.get("depends_on")
                    for spec in schema_dict.values()
                )
                if has_depends and labels:
                    # Build oracle overrides: for each field with depends_on,
                    # set the parent to its TRUE label value.
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
                            pass  # decide_fn doesn't support oracle_overrides
            for fname, res in results.items():
                field_def = schema.fields.get(fname)
                prediction = res.get("prediction")
                # Rotated variants reorder the SAME canonical choice strings,
                # so a prediction string is already canonical; the remap seam
                # only validates membership (and would translate a positional
                # encoding here if a future track produced one).
                if tag and tag.startswith("rot") and field_def is not None:
                    if isinstance(prediction, str) and prediction not in field_def.choices:
                        prediction = None
                label = case.get("labels", {}).get(fname)
                line = {
                    "run_id": run_id,
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
                    "rows": meta.get("rows"),
                    "passes": meta.get("passes"),
                    "error": res.get("error"),
                    "salvage_prediction": res.get("salvage_prediction"),
                    "oracle_prediction": oracle_results.get(fname),
                }
                if carry_perturbation:
                    line["perturbation"] = (case.get("meta") or {}).get("perturbation")
                if carry_consensus:
                    consensus = (case.get("meta") or {}).get("consensus")
                    if isinstance(consensus, dict) and consensus:
                        line["consensus"] = consensus
                # Constraints always carried when present (case-level, not
                # schema-level — F1: StructuredSchema would parse a 'constraints'
                # key as a field).
                constraints = case.get("constraints")
                if isinstance(constraints, list) and constraints:
                    line["constraints"] = constraints
                lines.append(line)

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

    run = {
        "run_id": run_id,
        "environment": environment(),
        "config": config,
        "counts": {
            "cases": len(selected),
            "fields": n_canonical,
            "prediction_lines": len(lines),
        },
    }

    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, sort_keys=True) + "\n")
    if timing_meta:
        # W3-R: per-combo timing JSON — the median of each split across the
        # canonical decide calls. Same calls the predictions came from (no
        # second run).
        timing_summary: dict[str, float] = {}
        for key in timing_meta[0]:
            values = [m[key] for m in timing_meta if key in m]
            if values and all(isinstance(v, (int, float)) for v in values):
                timing_summary[key] = round(statistics.median(values), 4)
        with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"calls": len(timing_meta), "median": timing_summary},
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(
        "eval run %s: %d cases, %d prediction lines -> %s",
        run_id,
        len(selected),
        len(lines),
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
