"""W6-2 prep: slope probe + adapter parity, standalone benchmark commands.

Two commands (no engine wiring):

``slope``
    Load a model (alias ok), run the B=1/2/4/8 suffix probe under
    ``mx.reset_peak_memory`` at widths 4/8/16/32, print a table of peak
    bytes and the fitted per-row slope per width bin, and write JSON.

``adapters``
    For the loaded model, compare
    ``adapter.lm_head(adapter.backbone(x)[:, pos])`` vs ``model(x)[:, pos]``
    on the bundled preset rows: max abs diff and timing for full-width head
    vs decision-position head.

Fake-model tests cover the table/JSON builders only
(``tests/test_probe.py``); real runs land in benchmarks/m5.py as optional
steps behind ``--probe``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

__all__ = ["probe_slope", "probe_adapters", "slope_table_lines", "adapters_table_lines"]


def probe_slope(
    model,
    widths: tuple[int, ...] = (4, 8, 16, 32),
    batches: tuple[int, ...] = (1, 2, 4, 8),
    vocab_size: int | None = None,
    hidden_size: int | None = None,
) -> dict:
    """Peak-activation slope probe: [B, W, V] logits slab at each (width, B).

    Runs a real allocation+reduction per (batch, width) under
    ``mx.reset_peak_memory`` and records the measured peak bytes. The
    FITTED per-row slope per width bin is the B-scaling of peak bytes:
    slope(bin) = peak(B=8) / (peak(B=1) * 8) — 1.0 means the slab scales
    perfectly linearly with batch (no tiling overhead); >1.0 is the
    overhead the width-bin budget must charge.

    Returns the JSON-able dict: {"widths": [...], "batches": [...],
    "peak_bytes": {width: {batch: int}}, "fitted_slope_per_bin": {width:
    float}, "vocab": int, "hidden": int}.
    """
    if vocab_size is None:
        vocab_size = (
            model.args.vocab_size
            if hasattr(model, "args") and hasattr(model.args, "vocab_size")
            else model.model.embed_tokens.weight.shape[0]
        )
    if hidden_size is None:
        # args first (every mlx_lm ModelArgs carries hidden_size); only then
        # the embedding width as fallback.
        args = getattr(model, "args", None)
        hs = getattr(args, "hidden_size", None)
        if hs is None:
            embed = getattr(getattr(model, "model", None), "embed_tokens", None)
            hs = embed.weight.shape[1] if embed is not None else 2048
        hidden_size = int(hs)
    peak_bytes: dict[int, dict[int, int]] = {}
    for width in widths:
        peak_bytes[width] = {}
        for batch in batches:
            mx.reset_peak_memory()
            active_before = mx.get_active_memory()
            slab = mx.zeros((batch, width, vocab_size), dtype=mx.float32)
            total = mx.sum(slab)
            mx.eval(total)
            peak = mx.get_peak_memory() - active_before
            peak_bytes[width][batch] = int(max(0, peak))
            del slab, total
    fitted = {}
    for width in widths:
        b1 = max(1, peak_bytes[width][1])
        top_batch = max(batches)
        fitted[width] = round((peak_bytes[width][top_batch] / b1) / top_batch, 4)
    return {
        "widths": list(widths),
        "batches": list(batches),
        "peak_bytes": {str(w): {str(b): v for b, v in d.items()} for w, d in peak_bytes.items()},
        "fitted_slope_per_bin": {str(w): v for w, v in fitted.items()},
        "vocab": int(vocab_size),
        "hidden": int(hidden_size),
    }


def probe_adapters(
    model,
    tokenizer,
    preset_specs: list[tuple[str, dict]],
    max_cases: int = 8,
) -> dict:
    """Adapter split vs model call on bundled preset rows.

    For each preset (id, whole preset dict as returned by
    ``jevmlx.parity.bundled_preset_specs``): build the schema rows, and
    compare at ONE decision position of a row:
    ``model(row)[:, pos]`` (full-width head — what the engine does today)
    vs ``adapter.lm_head(adapter.backbone(row)[:, pos])`` (the W6 flow).
    Records max abs diff over all cases and wall time of both head
    strategies (per case and total).

    Real-model probe: needs a live engine; fake-model tests cover only the
    table/JSON builders.
    """
    from jevmlx.adapters import adapter_for
    from jevmlx.engine import _build_schema_rows
    from jevmlx.parity import _make_schema

    adapter = adapter_for(model)
    cases: list[dict] = []
    overall = 0.0
    full_total_ms = 0.0
    dec_total_ms = 0.0

    for case_id, preset in preset_specs[:max_cases]:
        schema = _make_schema(case_id, preset["schema"])
        built = _build_schema_rows(schema, tokenizer, "slots")
        if not built["rows"]:
            continue
        row = mx.array([built["rows"][0]])
        pos = built["row_decision"][0][0]

        full = model(row)
        split_hidden = adapter.backbone(row)
        split = adapter.lm_head(split_hidden[:, pos, :])
        mx.eval(full, split)
        diff = mx.max(mx.abs(full[0, pos, :] - split[0, :])).item()
        overall = max(overall, diff)

        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            mx.eval(model(row))
        full_ms = (time.perf_counter() - t0) / reps * 1000
        t0 = time.perf_counter()
        for _ in range(reps):
            mx.eval(adapter.lm_head(adapter.backbone(row)[:, pos, :]))
        dec_ms = (time.perf_counter() - t0) / reps * 1000
        full_total_ms += full_ms
        dec_total_ms += dec_ms

        cases.append(
            {
                "case": case_id,
                "row_tokens": len(built["rows"][0]),
                "decision_pos": int(pos),
                "max_abs_diff": float(diff),
                "full_width_ms": full_ms,
                "decision_pos_ms": dec_ms,
            }
        )

    return {
        "cases": cases,
        "max_abs_diff": overall,
        "full_width_ms_total": full_total_ms,
        "decision_pos_ms_total": dec_total_ms,
    }


def slope_table_lines(result: dict) -> list[str]:
    """Render the slope probe result as a printed table."""
    lines = [f"{'width':>6} | " + " | ".join(f"B={b:<8}" for b in result["batches"]) + " | slope"]
    for width in result["widths"]:
        key = str(width)
        cells = " | ".join(f"{result['peak_bytes'][key][str(b)]:>9}" for b in result["batches"])
        lines.append(f"{width:>6} | {cells} | {result['fitted_slope_per_bin'][key]:.3f}")
    return lines


def adapters_table_lines(result: dict) -> list[str]:
    """Render the adapters probe result as a printed table."""
    lines = [f"{'case':<24} {'max_abs_diff':>14} {'full_ms':>10} {'dec_ms':>10}"]
    for case in result["cases"]:
        lines.append(
            f"{case['case']:<24} {case['max_abs_diff']:>14.3e} "
            f"{case['full_width_ms']:>10.3f} {case['decision_pos_ms']:>10.3f}"
        )
    lines.append(f"overall max abs diff: {result['max_abs_diff']:.3e}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.probe",
        description="W6-2 prep: memory slope probe + adapter parity probe.",
    )
    parser.add_argument("--model", default="quality", help="model id or alias (quality/fast/test)")
    parser.add_argument(
        "--out", default=None, help="JSON output path (default: probe-<model>.json)"
    )
    parser.add_argument(
        "--command",
        choices=("slope", "adapters", "both"),
        default="both",
        help="which probe to run",
    )
    args = parser.parse_args(argv)

    from jevmlx.engine import load_engine

    model, tokenizer = load_engine(args.model)
    out_path = Path(args.out) if args.out else Path(f"probe-{args.model.replace('/', '_')}.json")
    payload: dict = {"model": args.model}

    if args.command in ("slope", "both"):
        payload["slope"] = probe_slope(model)
        for line in slope_table_lines(payload["slope"]):
            print(line)
    if args.command in ("adapters", "both"):
        from jevmlx.parity import bundled_preset_specs

        payload["adapters"] = probe_adapters(model, tokenizer, bundled_preset_specs())
        for line in adapters_table_lines(payload["adapters"]):
            print(line)

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
