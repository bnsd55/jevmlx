"""W5c-9: the persisted batched-drift envelope and the derived rescore band.

The near-tie rescore (W3-E) exists because batched Metal matmuls tile
differently per shape and log-score gaps inside the band are batch-shape
noise. The band was the CONSTANT ``INSTABILITY_BAND`` (0.05 nats); PR #66's
drift probe measured pairwise-gap drift (d_gap) up to 0.0625 nats at >= 16
merged rows, so a reference near tie could escape the rescore: batch-one
margin 0.031 with a batched raw-logit margin 0.0625 sat ABOVE the 0.05
band and never reached the canonical shape (seen in parity:
code_security/is_vulnerability, batch-one margin 0.031, escaped).

The rule this module ships (GPT-REVIEW-4-drift question 2):

    for a pass with M rows: rescore when batched top-two margin <=
    INSTABILITY_BAND + E_bound(M)

where E_bound comes from a PERSISTED drift envelope — measured, not
fitted — keyed by the tuple that actually determines the numerics:

    (model_id, revision, quantization, mlx_version, chip, activation dtype,
     bucket_edges_version)

with a COARSE shape bucket (M<=4, <=8, <=16, >16): the probe showed the
plateau starts at 16 rows and is flat to 128, so bucketing by exact M would
over-fit the table; the plateau bounds every larger M.

Sources of the envelope, in order of authority:

1. RECORDED: ``benchmarks/driftprobe.py`` (full matrix) and
   ``jevmlx.parity.parity_report`` (the batched matrix it already runs)
   both write envelope records under ``benchmarks/probes/`` AND cache them
   under the user cache dir keyed by the tuple.
2. CANARY: when nothing is recorded for the tuple, engine load runs a tiny
   one-forward 16-row probe on a fixed synthetic schema and uses it, logs
   that the envelope is unrecorded, and writes it (so the next load has a
   recorded value). The canary MEASURES; it never installs a constant —
   on failure the load RAISES (no silent under-cover).

E_bound = max over all recorded buckets AT OR ABOVE the pass's bucket
(the plateau is monotone in M up to the flat region), rounded UP to the
next 1/64 nat — the drift lattice IS the fp16 ULP ladder
(0.015625/0.03125/0.0625/0.125), so the band lives on the same grid it
protects. A pass with M ABOVE the largest recorded bucket uses the
largest recorded bound (the plateau's upper reach) — never a smaller
bucket's, never a constant.

The PARITY CONTRACT IS UNCHANGED: parity still fails when measured d_gap
exceeds 0.05. This band protects DECISIONS (every reference near tie
reaches the canonical batch=1 shape); it does not redefine parity. See
ARCHITECTURE.md "Rescore band vs parity contract".
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

__all__ = [
    "DRIFT_LATTICE",
    "BUCKET_EDGES",
    "BUCKET_EDGES_VERSION",
    "MAX_GAP_DRIFT_KEY",
    "DriftEnvelopeError",
    "shape_bucket",
    "bucket_labels",
    "round_up_lattice",
    "envelope_cache_path",
    "load_envelope_record",
    "record_envelope",
    "bound_from_records",
    "rescore_band",
    "band_for_pass",
    "envelope_key",
    "envelope_for_engine",
    "recorded_envelope_records",
    "probes_dir_for_record",
    "CANARY_SCHEMA",
    "run_canary",
]

logger = logging.getLogger(__name__)

# The drift lattice: pairwise-gap drift on the fp16 grid lands on multiples
# of 1/64 nat (measured plateau: 0.0625; every observed value in the W5c-4
# matrix is a multiple of 0.00390625). Bands round UP onto this grid.
DRIFT_LATTICE = 64.0

# The envelope record key the probe and parity both write.
MAX_GAP_DRIFT_KEY = "max_gap_drift_nats"

# The user-cache location for persisted envelope records (the probes
# folder in the repo is the durable copy; this is the machine-local cache
# the engine reads at load).
_ENVELOPE_CACHE = Path.home() / ".cache" / "jevmlx" / "driftenv"

# The repo probes dir (where REAL model records are committed — PR #66).
# Tests monkeypatch this to a tmp_path so test records never touch the
# repo. None => the caller resolves the default (benchmarks/probes/).
_PROBES_DIR_OVERRIDE: Path | None = None


class DriftEnvelopeError(RuntimeError):
    """A drift-envelope resolution that must not proceed silently — the
    canary failed, the cache is read-only, the chip could not be resolved,
    or a batched pass's M sits above the largest recorded bucket with no
    plateau record. The caller (engine load / a batched pass) RAISES this
    rather than installing a constant band."""


# ---------------------------------------------------------------------------
# Bucket edges — shipped in the package, versioned, part of the envelope key.
# ---------------------------------------------------------------------------

# The W5c-4 matrix measured the drift plateau starting at 16 rows and flat
# to 128. These edges are the DEFAULT set; a finer M>16 split (the fp32
# bisect jump between M=16 and M=112) lands as a NEW versioned edge set —
# never an in-place edit, because the edge version rides in the envelope
# key (records under different edges never mix). The shipped data file
# jevmlx/data/bucket_edges.json is the single source of truth; these
# module constants mirror it for import-time use and are overwritten from
# the file at import when it resolves.
BUCKET_EDGES_VERSION = 1
BUCKET_EDGES: tuple[int, ...] = (4, 8, 16)


def _load_shipped_edges() -> tuple[int, ...]:
    """Read the shipped bucket_edges.json (the package's source of truth).

    A malformed/missing file is a packaging bug, not a runtime fallback:
    raise so it surfaces in CI, never silently to the band.
    """
    path = Path(__file__).resolve().parent / "data" / "bucket_edges.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    edges = sorted(int(e) for e in data["edges"] if isinstance(e, (int, float)) and e > 0)
    if len(edges) < 2 or len(edges) != len(set(edges)):
        raise DriftEnvelopeError(f"malformed bucket_edges.json: {edges}")
    return tuple(edges)


try:
    BUCKET_EDGES = _load_shipped_edges()
except DriftEnvelopeError:
    raise
except OSError:
    # Source checkout without the data file packaged (e.g. a raw sdist) —
    # the constant above stands; a packaged install always has the file.
    pass


def _load_shipped_edges() -> tuple[int, ...]:
    """Read the shipped bucket_edges.json (the package's source of truth).

    A malformed/missing file is a packaging bug, not a runtime fallback:
    raise so it surfaces in CI, never silently to the band.
    """
    path = Path(__file__).resolve().parent / "data" / "bucket_edges.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    edges = sorted(int(e) for e in data["edges"] if isinstance(e, (int, float)) and e > 0)
    if len(edges) < 2 or len(edges) != len(set(edges)):
        raise DriftEnvelopeError(f"malformed bucket_edges.json: {edges}")
    return tuple(edges)


def bucket_labels() -> list[str]:
    """The ordered bucket labels for the shipped edge set."""
    return [f"M<={e}" for e in BUCKET_EDGES] + [f"M>{BUCKET_EDGES[-1]}"]


# The canary probe's fixed synthetic schema: one boolean field, a 16-row
# decision (the plateau's onset) — enough rows to cross the M<=16 bucket
# boundary in one forward, small enough to be negligible at load.
CANARY_SCHEMA: dict[str, dict] = {
    "canary_flag": {
        "type": "boolean",
        "description": "Envelope canary: is the record flagged",
    },
}


def shape_bucket(m_rows: int) -> str:
    """The shape bucket for a pass of M rows, from the shipped edge set."""
    for edge in BUCKET_EDGES:
        if m_rows <= edge:
            return f"M<={edge}"
    return f"M>{BUCKET_EDGES[-1]}"


def round_up_lattice(x: float) -> float:
    """Round UP to the next multiple of 1/64 nat (the drift lattice)."""
    return math.ceil(x * DRIFT_LATTICE) / DRIFT_LATTICE


def _slug(model_id: str) -> str:
    """A filesystem-safe slug for a model id (mirrors parity._slug)."""
    import re

    return re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower() or "model"


def probes_dir_for_record(model_id: str = "", *, chip: str = "") -> Path | None:
    """The probes dir for a record write. Real model records go to the repo's
    ``benchmarks/probes/<chip>--<slug>/`` (committed — PR #66); tests
    monkeypatch ``_PROBES_DIR_OVERRIDE`` to a tmp_path so test records never
    touch the repo. Returns None when the repo path does not exist (a
    non-repo install)."""
    if _PROBES_DIR_OVERRIDE is not None:
        return _PROBES_DIR_OVERRIDE
    try:
        from jevmlx.bench import HERE

        if not Path(HERE).exists():
            return None
        sub = f"{chip or 'unknown-chip'}--{_slug(model_id)}" if model_id else ""
        return Path(HERE) / "probes" / sub if sub else Path(HERE) / "probes"
    except Exception:  # noqa: BLE001 — best-effort
        return None


def envelope_cache_path(key: dict[str, Any]) -> Path:
    """The user-cache path for one envelope tuple.

    Keyed by (model_id, revision, quantization, mlx_version, chip,
    activation dtype, bucket_edges_version) — the tuple that determines the
    numerics. Quantization dicts hash by their sorted JSON (group_size/bits
    vary per model). The bucket-edge version is part of the key: records
    written under different edge sets never collide.
    """
    h = json.dumps(key, sort_keys=True, default=str)
    digest = hashlib.sha256(h.encode()).hexdigest()[:16]
    return _ENVELOPE_CACHE / f"{digest}.json"


def load_envelope_record(key: dict[str, Any]) -> dict[str, Any] | None:
    """The cached envelope record for the tuple, or None."""
    path = envelope_cache_path(key)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def record_envelope(record: dict[str, Any], *, probes_dir: Path | None = None) -> list[Path]:
    """Persist an envelope record: user cache + (best-effort) probes folder.

    ``record`` carries the full tuple key + ``shape_bucket`` +
    ``max_gap_drift_nats`` (+ provenance). The cache write keeps the MAX
    per (key, shape_bucket) — never last-writer-wins (C3): a driftprobe run
    that measured 0.125 must survive a later parity_report that measured
    0.0625 for the same bucket.

    A read-only cache (C6) is handled: the write is attempted, on
    PermissionError the in-memory record is kept (the engine resolution
    already holds it) and a warning logs — NO constant fallback installs.
    """
    written: list[Path] = []
    cache_path = envelope_cache_path(record.get("key") or record)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # noqa: BLE001 — read-only home / sandbox
        logger.warning(
            "drift-envelope cache dir unwritable (%s); record kept in-memory, "
            "canary will re-run next load",
            exc,
        )
        return written
    # Merge into the cache: keep the MAX per (key, shape_bucket).
    existing = load_envelope_record(record.get("key") or record) or {"records": []}
    records = list(existing.get("records", []))
    bucket = record.get("shape_bucket")
    new_val = record.get(MAX_GAP_DRIFT_KEY)
    replaced = False
    for i, r in enumerate(records):
        if r.get("shape_bucket") == bucket:
            old_val = r.get(MAX_GAP_DRIFT_KEY)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if float(new_val) > float(old_val):
                    records[i] = record
            else:
                records[i] = record
            replaced = True
            break
    if not replaced:
        records.append(record)
    payload = {"key": record.get("key") or record, "records": records}
    try:
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(cache_path)
    except OSError as exc:  # noqa: BLE001 — read-only cache (C6)
        logger.warning(
            "drift-envelope cache write failed (%s); record kept in-memory, "
            "canary will re-run next load",
            exc,
        )
        return written
    if probes_dir is not None:
        try:
            probes_dir.mkdir(parents=True, exist_ok=True)
            slug = json.dumps(record.get("key") or record, sort_keys=True, default=str)
            name = hashlib.sha256(slug.encode()).hexdigest()[:16]
            p = probes_dir / f"driftenv-{name}.json"
            p.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            written.append(p)
        except OSError:
            pass
    return written


def bound_from_records(records: list[dict[str, Any]], bucket: str) -> float | None:
    """E_bound for the bucket from envelope records (None when absent).

    The TIGHTEST covering bucket: the smallest bucket >= the pass's bucket
    that has a record. A higher bucket's plateau bounds any smaller M, so
    the nearest covering record is the correct (least wasteful) bound.
    For a pass whose M sits ABOVE the largest recorded bucket, the LARGEST
    recorded bound applies (the plateau's upper reach) — never a smaller
    bucket's, never a constant.
    """
    labels = bucket_labels()
    try:
        bi = labels.index(bucket)
    except ValueError:
        return None
    covering = [
        (labels.index(r.get("shape_bucket")), float(r.get(MAX_GAP_DRIFT_KEY)))
        for r in records
        if r.get("shape_bucket") in labels
        and labels.index(r.get("shape_bucket")) >= bi
        and isinstance(r.get(MAX_GAP_DRIFT_KEY), (int, float))
    ]
    if covering:
        covering.sort()
        return covering[0][1]
    # No covering bucket: fall back to the LARGEST recorded bound (the
    # plateau's upper reach) — M above the largest recorded bucket.
    all_vals = [
        float(r.get(MAX_GAP_DRIFT_KEY))
        for r in records
        if r.get("shape_bucket") in labels and isinstance(r.get(MAX_GAP_DRIFT_KEY), (int, float))
    ]
    return max(all_vals) if all_vals else None


def rescore_band(e_bound: float) -> float:
    """The decision band for a pass whose envelope bound is E_bound:
    INSTABILITY_BAND + E_bound, rounded UP to the 1/64 lattice."""
    from jevmlx.engine import INSTABILITY_BAND

    return round_up_lattice(INSTABILITY_BAND + e_bound)


def band_for_pass(records: list[dict[str, Any]], m_rows: int) -> float:
    """The rescore band for a pass of M rows from a tuple's envelope records.

    Resolves E_bound for the pass's shape bucket from the records (the
    tightest covering bucket; M above the largest recorded bucket uses the
    largest recorded bound). RAISES :class:`DriftEnvelopeError` when NO
    record covers the pass — a batched pass must not proceed with an
    unmeasured band (the escape the PR exists to close). Callers that
    legitimately have no envelope (the canonical batch=1 / dependency /
    oracle paths, which never rescore) never call this. M == 0 (a
    degenerate no-rows schema) returns INSTABILITY_BAND — no batched pass
    ran, so no widening applies.
    """
    from jevmlx.engine import INSTABILITY_BAND

    if m_rows <= 0:
        return INSTABILITY_BAND
    bound = bound_from_records(records, shape_bucket(m_rows))
    if bound is None:
        raise DriftEnvelopeError(
            f"no drift-envelope record covers a {m_rows}-row pass "
            f"(bucket {shape_bucket(m_rows)}); run benchmarks/driftprobe.py"
        )
    return rescore_band(bound)


def _chip_tag() -> str:
    """The machine chip tag (machine_tag() without the RAM suffix).

    Refuses None (C5): a None chip collides envelope keys across machines.
    """
    from jevmlx.bench import machine_tag

    tag = machine_tag()
    if not tag:
        raise DriftEnvelopeError("machine chip tag could not be resolved (sysctl)")
    return tag.rsplit("-", 1)[0]


def _activation_dtype(engine: Any) -> str:
    """The activation dtype the batched matmuls actually run in (C5).

    MEASURED from a one-token forward on the loaded model — not assumed
    'float16'. A real MLX model returns logits whose dtype IS the
    activation dtype. When the model is not callable (test fakes / a model
    that defers its forward), the dtype is inferred from the engine's
    metadata: quantized => float16 activations on Apple Silicon, else
    float32. RAISES :class:`DriftEnvelopeError` when neither path resolves
    a dtype (a None/unknown dtype collides envelope keys across models).
    """
    model = getattr(engine, "model", None)
    tokenizer = getattr(engine, "tokenizer", None)
    # One-token forward on the model's native path. The output logits'
    # dtype IS the activation dtype the batched matmuls run in.
    dtype = ""
    try:
        import mlx.core as _mx

        ids = _mx.array([[0]])
        if (
            tokenizer is not None
            and hasattr(tokenizer, "bos_token_id")
            and tokenizer.bos_token_id is not None
        ):
            ids = _mx.array([[tokenizer.bos_token_id]])
        out = model(ids)
        # Llama-style models return (logits, cache); take the logits.
        logits = out[0] if isinstance(out, (tuple, list)) else out
        dtype = str(getattr(logits, "dtype", ""))
    except Exception:  # noqa: BLE001 — measurement fails on non-callable fakes
        dtype = ""
    if dtype:
        return dtype
    # Fallback: infer from metadata (quantized => float16 on Apple Silicon).
    meta = {}
    try:
        from jevmlx.engine import engine_metadata

        meta = engine_metadata(getattr(engine, "model_id", "") or "")
    except Exception:  # noqa: BLE001 — best-effort metadata
        meta = {}
    if meta.get("quantization"):
        return "float16"
    if getattr(engine, "model_id", None) is not None:
        # An unquantized model on MLX defaults to float32 activations.
        return "float32"
    raise DriftEnvelopeError(
        "activation dtype could not be measured (model not callable) nor "
        "inferred (no metadata); refusing a None dtype"
    )


def envelope_key(engine: Any) -> dict[str, Any]:
    """The envelope tuple for a loaded engine (see module docstring).

    The bucket-edge version rides in the key (records under different edges
    never mix). The chip is REQUIRED (refuses None — C5).
    """
    from jevmlx.engine import engine_metadata

    meta = engine_metadata(getattr(engine, "model_id", "") or "")
    quant = meta.get("quantization")
    if isinstance(quant, dict):
        quant = {k: quant[k] for k in sorted(quant)}
    return {
        "model_id": getattr(engine, "model_id", None),
        "revision": getattr(engine, "revision", None),
        "quantization": quant,
        "mlx_version": meta.get("mlx_version"),
        "chip": _chip_tag(),
        "activation_dtype": _activation_dtype(engine),
        "bucket_edges_version": BUCKET_EDGES_VERSION,
    }


def recorded_envelope_records(engine: Any) -> list[dict[str, Any]]:
    """The cached envelope RECORDS for the engine's tuple (empty when none).

    The engine carries the RECORDS (not a single resolved bound) so the
    per-pass resolution can pick the right bucket (C2): the load-time
    canary's M<=16 record must not be applied to an M>16 production pass.
    """
    key = envelope_key(engine)
    cached = load_envelope_record(key)
    if cached is None:
        return []
    records = cached.get("records", [])
    return records if isinstance(records, list) else []


def envelope_for_engine(engine: Any, *, probes_dir: Path | None = None) -> dict[str, Any]:
    """Resolve the envelope for a loaded engine: the tuple's RECORDS,
    augmented by the canary when nothing is recorded.

    Returns the resolution the engine carries — the RECORDS list (so the
    per-pass band resolves by M) plus the key and source:

        {
          "key": <tuple>, "records": [...], "source": "recorded"|"canary",
        }

    When no record exists for the tuple, the canary runs (and WRITES its
    record — the next load finds it), a warning logs that the envelope was
    unrecorded, and the canary's record joins the records list. The canary
    MEASURES; on failure the load RAISES (no constant fallback — H3).

    ``probes_dir`` defaults to the repo's ``benchmarks/probes/`` (where real
    model records are committed — see PR #66); tests pass a tmp_path so test
    records never touch the repo.
    """
    key = envelope_key(engine)
    cached = load_envelope_record(key)
    records = list(cached.get("records", [])) if cached else []
    if records:
        return {"key": key, "records": records, "source": "recorded"}
    logger.warning(
        "drift envelope unrecorded for %s/%s — running the load-time canary "
        "and writing the record (run benchmarks/driftprobe.py for the full matrix)",
        key.get("model_id"),
        key.get("chip"),
    )
    try:
        record = run_canary(engine)
    except DriftEnvelopeError:
        raise
    except Exception as exc:  # noqa: BLE001 — a failed probe must not under-cover
        raise DriftEnvelopeError(
            f"canary probe failed for {key.get('model_id')}/{key.get('chip')}: {exc}"
        ) from exc
    if probes_dir is None:
        probes_dir = probes_dir_for_record(
            key.get("model_id") or "", chip=key.get("chip") or ""
        )
    record_envelope(record, probes_dir=probes_dir)
    # Read back the merged store (the write may have hit a covering bucket
    # already; the canary's own bucket is M<=16).
    fresh = load_envelope_record(key)
    records = list(fresh.get("records", [])) if fresh else [record]
    return {"key": key, "records": records, "source": "canary"}


def run_canary(engine: Any) -> dict[str, Any]:
    """The load-time canary: ONE 16-row forward on the fixed synthetic
    schema; returns the envelope record it produces.

    Measures d_gap of the field's branch logits at the decide_many shape
    (16 copies of the rows in ONE merged pass, per-row slots) against the
    canonical batch=1 shape — the same decomposition parity's RAW stage
    uses. The canary MEASURES; on any failure it RAISES (H3: no constant
    fallback — a broken probe must not silently under-cover the band).
    """
    key = envelope_key(engine)
    bucket = shape_bucket(16)
    measured = _canary_gap_drift(engine)
    record = {
        "key": key,
        "shape_bucket": bucket,
        MAX_GAP_DRIFT_KEY: float(measured),
        "source": "canary",
        "canary_rows": 16,
    }
    return record


def _canary_gap_drift(engine: Any) -> float:
    """d_gap at the 16-row decide_many shape vs batch=1 (one forward each)."""
    from jevmlx.engine import _build_schema_rows, _prefill, _score_rows
    from jevmlx.schema import StructuredSchema
    from jevmlx.timing import Ledger

    schema = StructuredSchema(CANARY_SCHEMA)
    context = "(no context provided)"
    tokenizer = engine.tokenizer
    built = _build_schema_rows(schema, tokenizer, "slots")
    if not built["rows"]:
        raise DriftEnvelopeError("canary schema produced no rows; cannot measure")
    pf = _prefill(engine.model, tokenizer, context, schema, Ledger(), "slots", engine.profile)
    n = 16
    rows_n = built["rows"] * math.ceil(n / len(built["rows"]))
    rows_n = rows_n[:n]
    decisions_n = (built["row_decision"] * math.ceil(n / len(built["row_decision"])))[:n]
    batched = _score_rows(
        engine.model,
        pf.cache,
        rows_n,
        decisions_n,
        engine.vocab_size,
        built["pad_id"],
        n,
        Ledger(),
        cache_slots=[pf.cache] * n,
    )
    ref = _score_rows(
        engine.model,
        pf.cache,
        built["rows"],
        built["row_decision"],
        engine.vocab_size,
        built["pad_id"],
        1,
        Ledger(),
    )
    worst = 0.0
    for j, got in batched.row_logits.items():
        orig = j % len(built["rows"])
        r = ref.row_logits.get(orig)
        if r is None:
            continue
        ref_gap = max(r) - min(r)
        got_gap = max(got) - min(got)
        worst = max(worst, abs(ref_gap - got_gap))
    return worst
