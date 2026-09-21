#!/usr/bin/env python3
"""Validate a bench results folder: contract, size, report reproducibility.

Usage::

    python -m benchmarks.check_results <results_dir> [<results_dir> ...]

Each ``results_dir`` is a combo folder (predictions.jsonl + run.json +
report.json) or a parent containing combo folders (the
``<machine>-<model>/<track>-<scorer>-<dataset>`` layout produced by
``jevmlx bench``). The dataset lock file
(``<dataset>.dataset.lock.json``) lives at the MODEL folder level (the
parent of the combo), not in the combo folder itself. The checker:

1. Finds every combo folder under the given roots.
2. For each combo: every predictions.jsonl line has the frozen contract keys
   (:data:`jevmlx.evalrun.PREDICTION_LINE_KEYS`) with the right types;
   run.json has the required top-level keys; the dataset lock
   (``<dataset>.dataset.lock.json`` in the model folder) is present and its
   sha256 matches run.json's ``dataset_lock_sha256``;
   the folder is under 5 MB (predictions may be gzipped).
3. Recomputes report.json's metrics from predictions.jsonl via
   :func:`jevmlx.evalmetrics.compute_metrics` and diffs against the
   committed report.json (numeric tolerance 1e-9).
4. Checks the W5 timing contract (results contract v2): a parallel-track
   run.json must carry the honest timing keys in ``config``-adjacent
   ``timing.json``/predictions — ``group_wall_ms`` / ``per_item_amortized_ms``
   / ``per_item_end_to_end_ms`` / ``contexts_per_pass`` on batched paths,
   ``failed_attempts`` and ``peak_incremental_bytes`` in the result
   telemetry. A parallel track.json whose ``median`` block misses the split
   keys (or a combo whose predictions carry a parallel ``_meta`` without
   ``peak_incremental_bytes`` / ``failed_attempts``) is a FAIL.
5. Prints a SUMMARY table (one row per combo) and exits 1 on any failure
   with a clear, per-folder list.

No MLX dependency: imports jevmlx.evalmetrics/evalreport only (mlx lives
inside jevmlx.engine, never imported here), so this runs on ubuntu-latest.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import sys
from pathlib import Path

from jevmlx.bench import MAX_FOLDER_BYTES
from jevmlx.evalmetrics import compute_metrics
from jevmlx.evalrun import PREDICTION_LINE_KEYS, RUN_REQUIRED_KEYS

__all__ = ["check_folder", "check_root", "check_parity", "main"]

_NUMERIC_TOLERANCE = 1e-9

# Results contract v2 (W5): the timing split keys every parallel-track combo
# must carry in timing.json's ``median`` block (ride the parallel _meta).
# Batched decide_many keys land there only when the run used the batched
# path; the single-context split below is required either way.
# W5b-14: the keys are UNCHANGED but their provenance changed — every one
# is a derivation of the request's timing.Ledger (derived_flat), measured
# once per interval; suffix_eval_ms stays the documented composite.
TIMING_SPLIT_KEYS = (
    "prior_ms",
    "prefill_ms",
    "plan_compile_ms",
    "cache_broadcast_ms",
    "suffix_eval_ms",
    "lm_head_gather_ms",
    "second_pass_ms",
    "total_ms",
)

# W5-D finding 27/30/32 + W5c-3: the per-item timing split. Since W5c-3 the
# SINGLE path also reports per_item_end_to_end_ms (own prefill span + own
# assembly span, from the same ledger), so that key is required on every
# parallel prediction line; group_wall_ms / per_item_amortized_ms stay the
# batched-only pair (they land together when decide_many produced the
# records). RUN_TIMING_KEYS are the request-scoped memory + retry counters
# every parallel result reports.
PER_ITEM_END_TO_END_KEY = "per_item_end_to_end_ms"
BATCHED_TIMING_KEYS = ("group_wall_ms", "per_item_amortized_ms")
RUN_TIMING_KEYS = ("peak_active_bytes", "peak_incremental_bytes", "failed_attempts")

# W5c-6 / B4: token-accounting telemetry — every parallel-track combo must
# carry these in timing.json's ``median`` block (ride the parallel _meta).
# The ratio of naive_branch_prompt_tokens to computed_prompt_token_positions
# is NOT a speedup (a suffix query still attends over the cached prefix).
TOKEN_ACCOUNTING_KEYS = (
    "naive_branch_prompt_tokens",
    "shared_prefix_tokens",
    "logical_suffix_token_positions",
    "computed_suffix_token_positions",
    "computed_prompt_token_positions",
    "retry_wasted_ms",
)

# W6-B6b/F2: results contract v2 — every report.json metrics dict must carry
# these CI keys (additive; they ride alongside the point estimates). The
# _metrics_match key-set check catches a missing key, but this explicit
# list documents the contract and produces a clear error message.
REQUIRED_CI_KEYS = (
    "accuracy_ci",
    "valid_accuracy",
    "per_field_accuracy",
)


def _read_predictions_lines(path: Path) -> list[dict]:
    """Load predictions.jsonl, transparently handling a gzipped file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _metrics_match(recomputed: dict, committed: dict, path: str, problems: list[str]) -> bool:
    """Deep-compare two metrics dicts over numbers (tolerance 1e-9) and exact
    non-numbers. Returns True when they agree."""

    def walk(r, c, trail: str) -> bool:
        ok = True
        if set(r) != set(c):
            problems.append(f"{trail}: key set mismatch {sorted(r)} != {sorted(c)}")
            return False
        for key in r:
            sub = f"{trail}.{key}" if trail else key
            rv, cv = r[key], c[key]
            if isinstance(rv, dict) and isinstance(cv, dict):
                ok = walk(rv, cv, sub) and ok
            elif _is_number(rv) and _is_number(cv):
                if abs(rv - cv) > _NUMERIC_TOLERANCE:
                    problems.append(f"{sub}: {rv} != {cv} (tol {_NUMERIC_TOLERANCE})")
                    ok = False
            elif _is_number(rv) or _is_number(cv):
                problems.append(f"{sub}: number vs non-number {rv!r} != {cv!r}")
                ok = False
            elif rv != cv:
                problems.append(f"{sub}: {rv!r} != {cv!r}")
                ok = False
        return ok

    return walk(recomputed, committed, path)


def check_folder(folder: Path) -> tuple[bool, list[str]]:
    """Validate one combo folder. Returns (ok, problems)."""
    problems: list[str] = []
    name = str(folder)

    # Required files. NOTE: the dataset lock is NOT a per-combo file —
    # since #84/#110 the bench writes <dataset>.dataset.lock.json at the
    # MODEL folder level (the parent of this combo) and run.json carries
    # dataset_lock_sha256. The lock is resolved + sha-verified below, after
    # run.json is read.
    pred_path = folder / "predictions.jsonl"
    run_path = folder / "run.json"
    report_path = folder / "report.json"
    for required, label in (
        (pred_path, "predictions.jsonl"),
        (run_path, "run.json"),
        (report_path, "report.json"),
    ):
        # predictions.jsonl may be gzipped.
        if not required.exists() and not Path(str(required) + ".gz").exists():
            problems.append(f"{name}: missing {label}")
    if problems:
        return False, problems

    # Folder size (predictions may be gzipped; the on-disk folder must fit).
    folder_bytes = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    if folder_bytes > MAX_FOLDER_BYTES:
        problems.append(
            f"{name}: folder is {folder_bytes} bytes (> {MAX_FOLDER_BYTES}); gzip the predictions"
        )

    # predictions.jsonl contract: every line has the frozen keys with types.
    actual_pred_path = pred_path if pred_path.exists() else Path(str(pred_path) + ".gz")
    try:
        records = _read_predictions_lines(actual_pred_path)
    except (OSError, json.JSONDecodeError) as e:
        problems.append(f"{name}: predictions.jsonl unreadable: {e}")
        return False, problems
    if not records:
        problems.append(f"{name}: predictions.jsonl is empty")
    for index, record in enumerate(records):
        missing = sorted(set(PREDICTION_LINE_KEYS) - set(record))
        extra = sorted(set(record) - set(PREDICTION_LINE_KEYS))
        # perturbation / consensus / oracle_prediction are optional add-ons,
        # not contract violations (oracle_prediction rides under
        # oracle_overrides evaluation; W3-D). ordered / ordinal /
        # ordinal_choices are written by OrdinalTelemetry (ordered=True enum
        # fields); they are optional — only present on ordinal fields.
        extra = [
            k
            for k in extra
            if k
            not in (
                "perturbation",
                "consensus",
                "oracle_prediction",
                "ordered",
                "ordinal",
                "ordinal_choices",
            )
        ]
        if missing:
            problems.append(f"{name}: line {index} missing keys {missing}")
        if extra:
            problems.append(f"{name}: line {index} unexpected keys {extra}")
        for key in PREDICTION_LINE_KEYS:
            if key not in record:
                continue
            if not _type_ok(key, record[key]):
                problems.append(
                    f"{name}: line {index} key {key!r} wrong type "
                    f"{type(record[key]).__name__} (value {record[key]!r})"
                )

    # run.json required keys.
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        problems.append(f"{name}: run.json unreadable: {e}")
        run = {}
    for key in RUN_REQUIRED_KEYS:
        if key not in run:
            problems.append(f"{name}: run.json missing top-level key {key!r}")

    # Dataset lock: since #84/#110 the lock lives at the MODEL folder level
    # as <dataset>.dataset.lock.json (the parent of this combo), and
    # run.json carries dataset_lock_sha256. Resolve the lock by dataset name
    # and verify the sha matches — FAIL only if the file is missing or the
    # sha differs. (The old code required a per-combo dataset.lock.json, which
    # the bench never writes — every combo failed 'missing dataset.lock.json'.)
    config = run.get("config", {}) if isinstance(run, dict) else {}
    dataset_path = config.get("dataset_path", "") or ""
    dataset_name = dataset_path if "/" not in dataset_path else Path(dataset_path).stem
    if dataset_name:
        model_dir = folder.parent
        lock_path = model_dir / f"{dataset_name}.dataset.lock.json"
        expected_sha = config.get("dataset_lock_sha256") or run.get("dataset_lock_sha256")
        if not lock_path.exists():
            problems.append(
                f"{name}: missing {lock_path.name} in model folder "
                f"({model_dir}) — the bench writes <dataset>.dataset.lock.json "
                "at the model folder level"
            )
        elif expected_sha:
            import hashlib

            actual_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                problems.append(
                    f"{name}: {lock_path.name} sha256 mismatch — "
                    f"file={actual_sha[:12]} run.json={expected_sha[:12]} "
                    "(the dataset was rebuilt after this run; rerun the combo)"
                )

    # Results contract v2 (W5): a parallel-track combo must carry the
    # honest timing split. predictions lines ride per-field latency_ms;
    # the _meta-split keys land in timing.json (same calls, medians).
    is_parallel = bool(records) and all(r.get("track") in (None, "parallel") for r in records)
    if is_parallel:
        timing_path = _latest_timing_path(folder)
        if timing_path is None or not timing_path.exists():
            problems.append(
                f"{name}: missing timing.json (parallel track must record the "
                "timing split; naive/openai tracks legitimately have none)"
            )
        else:
            timing = _load_json(timing_path)
            median = (timing or {}).get("median")
            if not isinstance(median, dict):
                problems.append(f"{name}: timing.json has no median object")
            else:
                for key in TIMING_SPLIT_KEYS:
                    if key not in median:
                        problems.append(f"{name}: timing.json median missing key {key!r}")
                for key in ("peak_active_bytes", "failed_attempts"):
                    if key not in median:
                        problems.append(
                            f"{name}: timing.json median missing key {key!r} "
                            "(results contract v2: failed_attempts + peak memory "
                            "ride the split)"
                        )
                peak_incremental = median.get("peak_incremental_bytes")
                if peak_incremental is not None and not _is_number(peak_incremental):
                    problems.append(
                        f"{name}: timing.json median peak_incremental_bytes is "
                        f"{peak_incremental!r}, expected a number"
                    )
                # W5c-6 / B4: token-accounting keys must be present in the
                # median block (ride the parallel _meta).
                for key in TOKEN_ACCOUNTING_KEYS:
                    if key not in median:
                        problems.append(
                            f"{name}: timing.json median missing key {key!r} "
                            "(results contract v2: token-accounting telemetry)"
                        )
                # W5c-7 item 5: timing segment isolation. A resumed run
                # carries segment > 0 and a segment_id; the report/leaderboard
                # read per-item timings only from the LATEST segment — never
                # pool across segments (a resumed run's timings are from a
                # different process/machine-state, not comparable).
                segment = (timing or {}).get("segment", 0)
                segment_id = (timing or {}).get("segment_id")
                if isinstance(segment, int) and segment > 0 and not segment_id:
                    problems.append(
                        f"{name}: timing.json segment={segment} but no "
                        "segment_id (results contract v2: resumed runs must "
                        "carry segment_id for timing isolation)"
                    )
        # Batched-path per-item timing: when the predictions carry
        # batched per-item keys (decide_many), BOTH must be present.
        per_item_keys = {key for key in BATCHED_TIMING_KEYS if any(key in r for r in records)}
        if per_item_keys and per_item_keys != set(BATCHED_TIMING_KEYS):
            missing = sorted(set(BATCHED_TIMING_KEYS) - per_item_keys)
            problems.append(
                f"{name}: batched per-item timing incomplete — missing {missing} "
                "(group_wall_ms / per_item_amortized_ms land together)"
            )

    # W5c-3: per_item_end_to_end_ms is required on parallel lines — the
    # single path reports it since W5c-3 (elapsed - plan), the batched path
    # since W5-D finding 27/N6. A parallel line without it is a broken
    # folder (no fallback; the leaderboard reads this key).
    if is_parallel:
        for index, record in enumerate(records):
            ms = record.get(PER_ITEM_END_TO_END_KEY)
            if not _is_number(ms):
                problems.append(
                    f"{name}: line {index} missing/invalid "
                    f"{PER_ITEM_END_TO_END_KEY} ({ms!r}) — results contract v2 "
                    "(single and batched paths both report it)"
                )

    # Report reproducibility: recompute metrics and diff against committed.
    try:
        committed_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        problems.append(f"{name}: report.json unreadable: {e}")
        committed_report = {}
    if records:
        recomputed = compute_metrics(records)
        committed_metrics = committed_report.get("metrics", {})
        if not _metrics_match(recomputed, committed_metrics, name, problems):
            pass  # problems already appended by _metrics_match
        # W6-B6b/F2: results contract v2 — the CI keys must be present.
        for key in REQUIRED_CI_KEYS:
            if key not in committed_metrics:
                problems.append(
                    f"{name}: report.json metrics missing key {key!r} "
                    "(results contract v2: CI keys required)"
                )

    return len(problems) == 0, problems


def _type_ok(key: str, value) -> bool:
    """Contract type check for one prediction-line value. None is allowed for
    every nullable field (latency_ms, log_scores, etc.); the engine writes
    None when a metric is not computable for that track."""
    nullable = {
        "label",
        "prediction",
        "log_scores",
        "per_option",
        "latency_ms",
        "rows",
        "passes",
        "error",
        "salvage_prediction",
        "confidence",
        "probability",
        "group_id",
        "source",
        "workflow",
        "type",
    }
    if value is None:
        return key in nullable or key in (
            "perturbation",
            "consensus",
            "ordinal",
            "ordinal_choices",
        )
    string_keys = {
        "run_id",
        "case_id",
        "group_id",
        "source",
        "workflow",
        "field",
        "track",
        "model",
        "permutation",
        "type",
    }
    if key in string_keys:
        return isinstance(value, str)
    if key in ("valid",):
        return isinstance(value, bool)
    if key in ("correct",):
        return value is None or isinstance(value, bool)
    if key in ("latency_ms", "rows", "passes", "probability"):
        return isinstance(value, int | float)
    if key in ("log_scores", "per_option"):
        return isinstance(value, dict)
    if key in ("prediction",):
        return isinstance(value, str | bool | list)
    if key in ("error", "salvage_prediction", "label"):
        return isinstance(value, str | bool | list | int | float)
    # ordinal telemetry (ordered=True enum fields): 'ordered' is a bool flag,
    # 'ordinal_choices' is the list of choice strings, 'ordinal' is the
    # per-level probability/variance dict.
    if key == "ordered":
        return isinstance(value, bool)
    if key == "ordinal_choices":
        return isinstance(value, list)
    if key == "ordinal":
        return isinstance(value, dict)
    return True  # unknown-but-present optional keys are not type-checked


def _find_combo_folders(root: Path) -> list[Path]:
    """Combo folders under a root: either root itself (has predictions.jsonl)
    or root/<machine-model>/<combo>/, or root/<combo>/."""
    if (root / "predictions.jsonl").exists() or (root / "predictions.jsonl.gz").exists():
        return [root]
    combos: list[Path] = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir():
            continue
        if (child / "predictions.jsonl").exists() or (child / "predictions.jsonl.gz").exists():
            combos.append(child)
        else:
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and (
                    (grandchild / "predictions.jsonl").exists()
                    or (grandchild / "predictions.jsonl.gz").exists()
                ):
                    combos.append(grandchild)
    return combos


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_timing_path(folder: Path) -> Path | None:
    """The latest timing segment file (P4).

    W5c-7: timing segments are separate files (timing.segment-N.json).
    Return the highest-numbered segment, or timing.json (segment 0) if
    no segments exist. Never merge across segments.
    """
    import glob

    segments = []
    for p in glob.glob(str(folder / "timing.segment-*.json")):
        try:
            seg = int(Path(p).name.split("segment-")[1].split(".")[0])
            segments.append((seg, Path(p)))
        except (ValueError, IndexError):
            continue
    if segments:
        return max(segments, key=lambda x: x[0])[1]
    timing_json = folder / "timing.json"
    return timing_json if timing_json.exists() else None


def check_parity(model_dir: Path) -> tuple[bool, list[str]]:
    """Check that a model's results folder has a passing slow parity test.

    A model enters the README compatibility table only with a passing slow
    parity test recorded in its results folder as ``parity.json``. Results
    contract v2 (W5): the file is produced by the parity producer over
    BOTH the batch=1/chunked scoring comparison AND the batched matrix's
    RAW pre-rescore row-logit gate (finding 42: a batch-1 rescore can mask
    raw batch drift, so the raw gate is its own pass/fail stage).

    The ``parity.json`` schema (v2 — the raw gate keys are required)::

        {
          "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
          "prompt_version": "jevmlx-parallel-v8",
          "max_abs_drift_nats": 0.027,
          "max_raw_row_drift_nats": 0.031,
          "max_batched_drift_nats": 0.05,
          "winners_identical": true,
          "atol": 0.05,
          "max_gap_drift_nats": 0.07,
          "max_margin_drift_nats": 0.06,
          "passed": true,
          "cases": ["code_security", "fintech_fraud", ...],
          "test": "test_w1a_scoring_parity_batch_vs_chunked_real_model",
          "run_at": "2026-09-18T12:00:00Z"
        }

    ``passed`` covers three stages: winners identical, final log-score
    drift within atol, batched pairwise gap drift within atol, and the
    rescore-gate escape flag (W5c-1 review: fail-closed at the fixed band)
    (W5c-1: the batched raw gate is a batch-shape stress with its own
    documented band, distinct from the single-context atol). The failure
    message names the stage that failed.
    """
    problems: list[str] = []
    name = str(model_dir)
    parity_path = model_dir / "parity.json"
    if not parity_path.exists():
        problems.append(f"{name}: missing parity.json (slow parity test not recorded)")
        return False, problems
    try:
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        problems.append(f"{name}: parity.json unreadable: {e}")
        return False, problems
    status = parity.get("status", "PASS" if parity.get("passed") else "FAIL")
    # parity-gates: PASS and DRIFT are publishable (OK); only FAIL (a winner
    # changed, or drift beyond the band) or missing/unreadable parity is a
    # hard FAIL. DRIFT is OK-with-note — the one-sentence explanation so an
    # operator sees the batch-shape noise is expected, not a regression.
    if status == "DRIFT":
        envelope = parity.get("drift_envelope") or {}
        band = envelope.get("band", "?")
        max_drift = max(
            parity.get("max_abs_drift_nats", 0),
            parity.get("max_gap_drift_nats", 0),
            parity.get("max_margin_drift_nats", 0),
        )
        sentence = (
            f"DRIFT: batched drift {max_drift} >= atol {parity['atol']}, "
            f"winners identical on all cases, inside envelope band {band}; "
            "near-tie rescore applies"
        )
        # DRIFT is OK (publishable) — the note is informational, not a problem.
        return True, [f"{name}: parity.json status=DRIFT — {sentence}"]
    if status == "FAIL" or not parity.get("passed"):
        stages = _parity_failed_stages(parity)
        sentence = (
            f"FAIL: {'; '.join(stages)} "
            f"(max_drift={parity['max_abs_drift_nats']}, "
            f"raw_row_drift={parity['max_raw_row_drift_nats']}, "
            f"atol={parity['atol']})"
        )
        problems.append(f"{name}: parity.json shows test did not pass — {sentence}")
        return False, problems
    # status == PASS — a v1 file (no v2 keys) predates the decomposition; a
    # folder regenerated by the current bench always carries them.
    missing = [k for k in ("max_raw_row_drift_nats", "max_gap_drift_nats") if k not in parity]
    if missing:
        problems.append(
            f"{name}: parity.json is pre-v2 (missing {', '.join(missing)}) — "
            "rerun the bench to regenerate with the per-row decomposition"
        )
        return False, problems
    return True, []


def _parity_failed_stages(parity: dict) -> list[str]:
    """Which parity stages failed, from the recorded payload.

    W5c-1 review section 2: the gate is fail-closed at the fixed atol.
    Stages: winners flipped, single-context log-score drift over atol,
    batched pairwise GAP drift over atol (d_gap — raw logits carry an
    arbitrary offset, so d_raw is diagnostic only), batched top-two margin
    drift over atol, and the rescore-gate escape flag
    (reference_near_tie && !production_rescored).
    """
    atol = parity["atol"]
    stages: list[str] = []
    if not parity.get("winners_identical", True):
        stages.append("winners flipped (final decisions disagree)")
    drift = parity["max_abs_drift_nats"]
    if isinstance(drift, int | float) and drift >= atol:
        stages.append("single-context log-score drift >= atol")
    gap = parity["max_gap_drift_nats"]
    if isinstance(gap, int | float) and gap >= atol:
        stages.append("batched pairwise gap drift >= atol")
    margin = parity["max_margin_drift_nats"]
    if isinstance(margin, int | float) and margin >= atol:
        stages.append("batched top-two margin drift >= atol")
    rescore = parity.get("rescore_gate", {})
    if isinstance(rescore, dict) and rescore.get("escaped_near_tie"):
        stages.append("rescore-gate escape (reference_near_tie && !production_rescored)")
    # A passing payload has NO failed stages — an empty list is the honest
    # answer, not a junk sentinel.
    return stages


def check_root(root: Path) -> list[tuple[Path, bool, list[str]]]:
    """Validate every combo folder under root. Returns per-folder results."""
    folders = _find_combo_folders(root)
    if not folders:
        return [(root, False, ["no combo folders found under this root"])]
    results = []
    for folder in folders:
        ok, problems = check_folder(folder)
        results.append((folder, ok, problems))
    return results


def _summary(results: list[tuple[Path, bool, list[str]]]) -> str:
    """One-line-per-folder summary table."""
    out = io.StringIO()
    out.write("| folder | status | checks |\n")
    out.write("|---|---|---|\n")
    for folder, ok, problems in results:
        status = "OK" if ok else "FAIL"
        n = len(problems) if problems else 0
        out.write(f"| {folder.name or folder} | {status} | {n} |\n")
    return out.getvalue()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Validate bench results folders and README leaderboard freshness."
    )
    ap.add_argument("results_dirs", nargs="*", help="results folders to validate")
    ap.add_argument(
        "--check-readme",
        metavar="README.md",
        default=None,
        help="regenerate the leaderboard table in memory and fail if the README's "
        "<!-- leaderboard:start --> block differs",
    )
    ap.add_argument(
        "--results",
        default="benchmarks/results",
        help="results root for --check-readme (default: benchmarks/results)",
    )
    ap.add_argument(
        "--published",
        default=None,
        help="published_agreement.json for --check-readme (L1 output, optional)",
    )
    ap.add_argument(
        "--check-parity",
        action="store_true",
        help="also verify each model folder has a passing slow parity test "
        "(parity.json); a model without one cannot enter the README compat table",
    )
    args = ap.parse_args(argv)

    # --check-readme: build the leaderboard table and compare the README block.
    if args.check_readme:
        from benchmarks.leaderboard import build_table, check_readme

        table = build_table(
            Path(args.results),
            Path(args.published) if args.published else None,
        )
        readme_path = Path(args.check_readme)
        if check_readme(readme_path, table):
            print("README leaderboard block is up to date.")
        else:
            print("README leaderboard block is STALE — regenerate with:", file=sys.stderr)
            cmd = f"  python -m benchmarks.leaderboard --results {args.results}"
            if args.published:
                cmd += f" --published {args.published}"
            cmd += f" --readme {args.check_readme}"
            print(cmd, file=sys.stderr)
            return 1
        # Fall through to folder validation if dirs were also given.
        if not args.results_dirs:
            return 0

    argv_dirs = args.results_dirs
    if not argv_dirs:
        print(
            "usage: python -m benchmarks.check_results <results_dir> [...] "
            "[--check-readme README.md]",
            file=sys.stderr,
        )
        return 2
    all_results: list[tuple[Path, bool, list[str]]] = []
    for arg in argv_dirs:
        all_results.extend(check_root(Path(arg)))

    # W4-A: optional slow parity gate. A model enters the README compat table
    # only with a passing slow parity test (parity.json in the model folder).
    if args.check_parity:
        checked_models: set[Path] = set()
        for arg in argv_dirs:
            root = Path(arg)
            # Model folders are either the root itself (single model) or
            # children of the root (the <machine>-<model> layout). A model
            # folder is any dir that contains parity.json OR contains combo
            # subdirs (the <machine>-<model> level).
            model_dirs: list[Path] = []
            if (root / "parity.json").exists():
                model_dirs = [root]
            elif root.is_dir():
                for child in sorted(root.iterdir()):
                    if child.is_dir():
                        model_dirs.append(child)
            for model_dir in model_dirs:
                if model_dir in checked_models:
                    continue
                checked_models.add(model_dir)
                ok, problems = check_parity(model_dir)
                all_results.append((model_dir, ok, problems))

    print(_summary(all_results))
    any_fail = False
    for folder, ok, problems in all_results:
        if ok:
            continue
        any_fail = True
        print(f"\n{folder}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
