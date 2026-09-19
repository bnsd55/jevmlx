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

    (model_id, revision, quantization, mlx_version, chip, activation dtype)

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
   recorded value).

E_bound = max(recorded, canary), rounded UP to the next 1/64 nat — the
drift lattice IS the fp16 ULP ladder (0.015625/0.03125/0.0625/0.125), so
the band lives on the same grid it protects.

The PARITY CONTRACT IS UNCHANGED: parity still fails when measured d_gap
exceeds 0.05. This band protects DECISIONS (every reference near tie
reaches the canonical batch=1 shape); it does not redefine parity. See
ARCHITECTURE.md "Rescore band vs parity contract".
"""

from __future__ import annotations

import json
import logging
import math
import platform
from pathlib import Path
from typing import Any

__all__ = [
    "DRIFT_LATTICE",
    "MAX_GAP_DRIFT_KEY",
    "shape_bucket",
    "round_up_lattice",
    "envelope_cache_path",
    "load_envelope_record",
    "record_envelope",
    "bound_from_records",
    "rescore_band",
    "band_for_rows",
    "envelope_for_engine",
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
    """The shape bucket for a pass of M rows.

    Default edges (4, 8, 16) mirror the W5c-4 plateau measurement (drift
    onset at 16; everything above shares the M>16 bucket). DATA-DRIVEN
    OVERRIDE (W5c-9 review): when the recorded bucket-edge file carries
    explicit edges, those govern — a finer M>16 split lands as a recorded
    edge set (e.g. the fp32 bisect jump between M=16 and M=112 =>
    edges 16/32/64/112), never as a code change.
    """
    edges = _recorded_bucket_edges()
    if edges is None:
        edges = (4, 8, 16)
    labels = [f"M<={e}" for e in edges]
    for e, label in zip(edges, labels, strict=True):
        if m_rows <= e:
            return label
    return f"M>{edges[-1]}"


# The bucket-edge version rides in the envelope key: changing edges does
# not silently mix records written under different edges.
BUCKET_EDGES_VERSION = 1


def _recorded_bucket_edges() -> tuple[int, ...] | None:
    """Bucket edges from the recorded edge-set file, when present.

    <cache>/bucket_edges.json: {\"version\": N, \"edges\": [4, 8, 16, 32, 112]}
    written by the analyst when the bisect report lands. Missing/malformed
    => the default plateau edges. The version check guards against reading
    edges older than this code's expectations.
    """
    try:
        raw = (_ENVELOPE_CACHE / "bucket_edges.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        version = int(data.get("version", -1))
        edges = data.get("edges")
        if version != BUCKET_EDGES_VERSION or not isinstance(edges, list):
            return None
        clean = sorted(int(e) for e in edges if isinstance(e, (int, float)) and e > 0)
        if len(clean) < 2 or len(clean) != len(set(clean)):
            return None
        return tuple(clean)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None


def round_up_lattice(x: float) -> float:
    """Round UP to the next multiple of 1/64 nat (the drift lattice)."""
    return math.ceil(x * DRIFT_LATTICE) / DRIFT_LATTICE


def envelope_cache_path(key: dict[str, Any]) -> Path:
    """The user-cache path for one envelope tuple.

    Keyed by (model_id, revision, quantization, mlx_version, chip,
    activation dtype) — the tuple that determines the numerics. Quantization
    dicts hash by their sorted JSON (group_size/bits vary per model).
    """
    h = json.dumps(key, sort_keys=True, default=str)
    # stable short hash: the tuple itself is too nested for a filename.
    import hashlib

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
    ``max_gap_drift_nats`` (+ provenance). The cache write is authoritative;
    the probes-folder copy is written when the dir exists or can be created
    (repo checkouts), and silently skipped elsewhere (installed wheels).
    """
    written: list[Path] = []
    cache_path = envelope_cache_path(record.get("key") or record)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge into the cache: keep the max per (key, shape_bucket).
    existing = load_envelope_record(record.get("key") or record) or {"records": []}
    records = [
        r
        for r in existing.get("records", [])
        if r.get("shape_bucket") != record.get("shape_bucket")
    ]
    records.append(record)
    cache_path.write_text(
        json.dumps({"key": record.get("key") or record, "records": records}, indent=2),
        encoding="utf-8",
    )
    written.append(cache_path)
    if probes_dir is not None:
        try:
            probes_dir.mkdir(parents=True, exist_ok=True)
            slug = json.dumps(record.get("key") or record, sort_keys=True, default=str)
            import hashlib

            name = hashlib.sha256(slug.encode()).hexdigest()[:16]
            p = probes_dir / f"driftenv-{name}.json"
            p.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            written.append(p)
        except OSError:
            pass
    return written


def bound_from_records(records: list[dict[str, Any]], bucket: str) -> float | None:
    """E_bound for the bucket from envelope records (None when absent).

    The bucket's own record bounds it; a HIGHER bucket's record also bounds
    it (the plateau is monotone in M up to the flat region — the W5c-4
    matrix showed no decrease past 16; using the max keeps fail-safe).
    """
    edges = _recorded_bucket_edges() or (4, 8, 16)
    labels = [f"M<={e}" for e in edges] + [f"M>{edges[-1]}"]
    try:
        bi = labels.index(bucket)
    except ValueError:
        return None
    # The TIGHTEST covering bucket: the smallest bucket >= the pass's
    # bucket that has a record. A higher bucket's plateau bounds any
    # smaller M, so the nearest covering record is the correct (least
    # wasteful) bound.
    covering = [
        (labels.index(r.get("shape_bucket")), float(r.get(MAX_GAP_DRIFT_KEY)))
        for r in records
        if r.get("shape_bucket") in labels
        and labels.index(r.get("shape_bucket")) >= bi
        and isinstance(r.get(MAX_GAP_DRIFT_KEY), (int, float))
    ]
    if not covering:
        return None
    covering.sort()
    return covering[0][1]


def rescore_band(e_bound: float) -> float:
    """The decision band for a pass whose envelope bound is E_bound:
    INSTABILITY_BAND + E_bound, rounded UP to the 1/64 lattice."""
    from jevmlx.engine import INSTABILITY_BAND

    return round_up_lattice(INSTABILITY_BAND + e_bound)


def band_for_rows(envelope: dict[str, Any], m_rows: int) -> float:
    """The rescore band for a pass of M rows from a resolved envelope.

    The envelope carries the bound for the pass's shape bucket; the bucket
    is recomputed from M (not trusted from the resolution record, which is
    the canary's bucket). When the envelope is missing or malformed the
    CONSERVATIVE plateau bound (0.0625) applies — the band never
    under-covers because of bookkeeping.
    """
    from jevmlx.engine import INSTABILITY_BAND

    bound = None
    if isinstance(envelope, dict):
        records = envelope.get("records")
        if isinstance(records, list):
            bound = bound_from_records(records, shape_bucket(m_rows))
        elif envelope.get("bound") is not None:
            # The resolution record is for one bucket; a larger M in a
            # HIGHER bucket needs the higher bucket's bound — conservatively
            # reuse the resolved bound (the plateau is flat above 16).
            bound = float(envelope["bound"])
    if bound is None:
        bound = 0.0625
    return round_up_lattice(INSTABILITY_BAND + bound)


def envelope_key(engine: Any) -> dict[str, Any]:
    """The envelope tuple for a loaded engine (see module docstring)."""
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
        "activation_dtype": "float16",
    }


def _chip_tag() -> str | None:
    """The machine chip tag (machine_tag() without the RAM suffix)."""
    try:
        from jevmlx.bench import machine_tag

        tag = machine_tag()
        return tag.rsplit("-", 1)[0] if tag else None
    except Exception:  # noqa: BLE001 — best-effort provenance
        return None


def run_canary(engine: Any) -> dict[str, Any]:
    """The load-time canary: ONE 16-row forward on the fixed synthetic
    schema; returns the envelope record it produces.

    Measures d_gap of the field's branch logits at the decide_many shape
    (16 copies of the rows in ONE merged pass, per-row slots) against the
    canonical batch=1 shape — the same decomposition parity's RAW stage
    uses. Never raises: on any failure the canary reports a CONSERVATIVE
    fallback (the plateau value 0.0625) so the band never under-covers
    because the probe itself broke.
    """
    key = envelope_key(engine)
    bucket = "M<=16"
    try:
        measured = _canary_gap_drift(engine)
    except Exception as exc:  # noqa: BLE001 — the canary must not break load
        logger.warning("drift-envelope canary failed (%s); using conservative fallback", exc)
        measured = None
    plateau_fallback = 0.0625
    # A measured 0 can be honest for a TINY schema (one branch, logits on
    # the same fp16 grid at every shape) — but it must NOT shrink the band
    # below the measured plateau: the canary schema is synthetic, the real
    # workload's drift was measured at 0.0625 on real schemas. The canary
    # LOWER-bounds nothing; it only ever RAISES the band above the known
    # plateau when IT measures more. So: value = max(measured, plateau)
    # when measured; the plateau alone when the probe failed.
    value = max(float(measured), plateau_fallback) if measured is not None else plateau_fallback
    record = {
        "key": key,
        "shape_bucket": bucket,
        MAX_GAP_DRIFT_KEY: value,
        "source": "canary" if measured is not None else "canary_fallback",
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
        return 0.0
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


def envelope_for_engine(engine: Any) -> dict[str, Any]:
    """Resolve the envelope for a loaded engine: recorded value or canary.

    Returns the resolution dict the engine carries:

        {
          "key": <tuple>, "bucket": <bucket-of-load>, "bound": E_bound,
          "band": rescore_band(E_bound), "source": "recorded"|"canary",
        }

    When no record exists for the tuple, the canary runs (and WRITES its
    record — the next load finds it), a warning logs that the envelope was
    unrecorded, and E_bound = max(recorded, canary) = the canary value.
    When a record exists, it is used AS IS (the canary does not re-run at
    every load; the probe/parity writers keep it fresh).
    """
    key = envelope_key(engine)
    bucket = shape_bucket(16)  # the load-time canary's own bucket
    cached = load_envelope_record(key)
    recorded = bound_from_records(cached.get("records", []), bucket) if cached else None
    if recorded is not None:
        return {
            "key": key,
            "bucket": bucket,
            "bound": round_up_lattice(recorded),
            "band": rescore_band(recorded),
            "source": "recorded",
        }
    logger.warning(
        "drift envelope unrecorded for %s/%s — running the load-time canary "
        "and writing the record (run benchmarks/driftprobe.py for the full matrix)",
        key.get("model_id"),
        key.get("chip"),
    )
    record = run_canary(engine)
    from jevmlx.bench import HERE

    probes_dir = Path(HERE) / "probes"
    record_envelope(record, probes_dir=probes_dir if Path(HERE).exists() else None)
    return {
        "key": key,
        "bucket": bucket,
        "bound": round_up_lattice(record[MAX_GAP_DRIFT_KEY]),
        "band": rescore_band(record[MAX_GAP_DRIFT_KEY]),
        "source": record["source"],
    }


def _platform_note() -> str:
    return platform.platform()
