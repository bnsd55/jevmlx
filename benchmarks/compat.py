"""Model compatibility matrix for jevmlx.

Loads each model, runs the fintech_fraud and support_triage presets through
run_parallel_generation, and prints a markdown table with load status,
schema validity, warm latency, prompt size, and peak GPU memory.

Each model is probed in its own subprocess so peak-memory readings are
isolated (models share a process-wide Metal memory high-water mark).

Usage: .venv/bin/python tools/compat.py [model_id ...]
"""

import json
import subprocess
import sys

import mlx.core as mx

from jevmlx.cli import load_preset
from jevmlx.engine import load_engine, run_parallel_generation
from jevmlx.schema import StructuredSchema

MODELS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/gemma-2-2b-it-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
]

PRESETS = ["fintech_fraud", "support_triage"]

EMPTY_ROW = {
    "loads": "n",
    "presets": "",
    "latency_ms": "-",
    "prompt_tokens": "-",
    "peak_mem_gb": "-",
    "error": "",
}


def probe_row(model_id: str) -> dict:
    """Probe one model; returns the table row (never raises)."""
    row = {"model": model_id, **EMPTY_ROW}
    try:
        engine = load_engine(model_id)
        row["loads"] = "y"
        presets_ok = []
        latencies = []
        prompt_tokens = 0
        for name in PRESETS:
            preset = load_preset(name)
            schema = StructuredSchema(preset["schema"])
            run_parallel_generation(engine, preset["context"], schema)  # warmup
            result = run_parallel_generation(engine, preset["context"], schema)
            latencies.append(result["elapsed_ms"])
            valid = all(
                str(entry["value"]).lower() in [c.lower() for c in schema.fields[f].choices]
                if schema.fields[f].field_type != "boolean"
                else isinstance(result["parsed_json"][f]["value"], bool)
                for f, entry in result["parsed_json"].items()
            )
            presets_ok.append("ok" if valid else "INVALID")
            prompt_tokens += len(engine.tokenizer.encode(preset["context"]))
        row["presets"] = ", ".join(presets_ok)
        row["latency_ms"] = f"{sum(latencies) / len(latencies):.0f}"
        row["prompt_tokens"] = str(prompt_tokens)
        row["peak_mem_gb"] = f"{mx.metal.get_peak_memory() / 1e9:.2f}"
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:60]
    return row


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--row":
        # Child mode: probe one model, print the row as the last stdout line.
        row = probe_row(args[1])
        print("ROW_JSON=" + json.dumps(row), flush=True)
        return

    models = args or MODELS
    rows = []
    for model_id in models:
        print(f"Probing {model_id} ...", flush=True)
        # Subprocess isolation: load_engine caches models and Metal tracks a
        # process-wide peak, so later models would inherit earlier peaks.
        proc = subprocess.run(
            [sys.executable, __file__, "--row", model_id],
            capture_output=True,
            text=True,
        )
        row = None
        for line in proc.stdout.splitlines():
            if line.startswith("ROW_JSON="):
                row = json.loads(line[len("ROW_JSON=") :])
        if row is None:
            first_err = (proc.stderr or "unknown error").strip().splitlines()
            row = {
                "model": model_id,
                **EMPTY_ROW,
                "error": (first_err[-1] if first_err else "unknown error")[:60],
            }
        rows.append(row)

    print()
    print(
        "| Model | loads | presets valid "
        "| warm latency (ms, avg of 2 presets) | prompt tokens | peak GPU mem (GB) |"
    )
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r["loads"] == "y":
            print(
                f"| `{r['model']}` | y | {r['presets']} | {r['latency_ms']} "
                f"| {r['prompt_tokens']} | {r['peak_mem_gb']} |"
            )
        else:
            print(f"| `{r['model']}` | n | - | - | - | {r['error']} |")


if __name__ == "__main__":
    main()
