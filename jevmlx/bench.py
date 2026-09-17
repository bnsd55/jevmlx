"""``jevmlx bench``: one command, a complete PR-ready benchmark results folder.

Runs the eval harness across tracks x scorers x bundled datasets on this
machine, writes predictions/run manifests/reports per combination, then
summarizes everything into ``SUMMARY.md`` with PR instructions. Datasets are
built once into ``~/.cache/jevmlx/bench/`` and reused while their lock files
match.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from jevmlx.evalmetrics import compute_metrics, load_predictions
from jevmlx.evalreport import environment, write_report
from jevmlx.evalrun import parallel_decide_fn, run_eval

BENCH_CACHE = Path.home() / ".cache" / "jevmlx" / "bench"
HERE = Path(__file__).resolve().parent.parent / "benchmarks"
RESULTS_DIR = HERE / "results"
MAX_FOLDER_BYTES = 5 * 1024 * 1024

DATASETS = (
    "bundled",
    "typesafe",
    "perturbed",
    "synthetic-labels",
    "synthetic-cardinality",
    "synthetic-injection",
    "synthetic-dependent",
)
SCORERS = ("slots", "labels")
TRACKS = ("parallel", "naive_local")


def machine_tag(override: str | None = None) -> str:
    """Machine tag ``<chip-lowercase>-<ram>gb`` (e.g. ``m5max-128gb``).

    The chip comes from ``sysctl machdep.cpu.brand_string`` with the marketing
    noise dropped (Apple M5 Max -> m5max); RAM from ``hw.memsize`` in GB.
    """
    if override:
        return override
    chip_raw = _sysctl(["machdep.cpu.brand_string"]) or "unknown-chip"
    words = [w for w in chip_raw.replace("Apple", "").split() if w]
    chip = "".join(w.lower() for w in words if w.lower() != "apple")
    ram_raw = _sysctl(["hw.memsize"])
    ram_gb = int(int(ram_raw) / 2**30) if (ram_raw or "").isdigit() else 0
    return f"{chip}-{ram_gb}gb"


def _sysctl(name_args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", *name_args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def preflight(force: bool, machine_override: str | None) -> str:
    """Refuse to run on machines that cannot produce comparable numbers.

    Checks Apple Silicon, free Metal memory (a second GPU-heavy process would
    skew latency), and battery power (thermal throttling). ``--force``
    overrides the battery and Metal checks (never the platform check) with
    the reason printed. Returns the machine tag.
    """
    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("bench requires Apple Silicon (macOS + arm64).")

    tag = machine_tag(machine_override)

    battery = _probe(["pmset", "-g", "batt"])
    on_battery = battery is not None and "AC Power" not in battery
    if on_battery:
        if not force:
            raise SystemExit(
                "on battery power (thermal throttling skews latency); plug in or pass --force"
            )
        print("WARNING: on battery power (--force); latency numbers may be throttled.")

    if not force:
        resident = _metal_resident_bytes()
        if resident is not None and resident > 1 * 2**30:
            raise SystemExit(
                f"another process holds ~{resident / 2**30:.1f} GB of Metal memory; "
                "close it or pass --force"
            )
    return tag


def _probe(command: list[str]) -> str | None:
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _metal_resident_bytes() -> int | None:
    """Wired memory in bytes as a proxy for Metal pressure (other GPU processes
    would show up as wired/in-use memory)."""
    raw = _sysctl(["vm.wire_count"])
    if raw is None or not raw.isdigit():
        return None
    page = _sysctl(["hw.pagesize"])
    page_size = int(page) if (page or "").isdigit() else 16384
    return int(raw) * page_size


def model_slug(model_id: str) -> str:
    """Folder-safe model slug: lowercased, '/' -> '--', dots kept."""
    return model_id.lower().replace("/", "--")


def build_datasets(datasets: list[str], offline_ok: bool = True) -> dict[str, Path]:
    """Build eval JSONL datasets into the bench cache, reusing lock matches.

    Returns dataset name -> JSONL path. ``typesafe`` is skipped with a clear
    message when offline; ``perturbed`` derives from ``bundled``.
    """
    BENCH_CACHE.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if "bundled" in datasets:
        jsonl = BENCH_CACHE / "bundled.jsonl"
        lock = BENCH_CACHE / "bundled.dataset.lock.json"
        _rebuild_if_needed(jsonl, lock, _build_bundled)
        paths["bundled"] = jsonl

    if "typesafe" in datasets:
        jsonl = BENCH_CACHE / "typesafe.jsonl"
        lock = BENCH_CACHE / "typesafe.dataset.lock.json"
        try:
            _rebuild_if_needed(jsonl, lock, _build_typesafe)
        except OSError as exc:
            if offline_ok and not (jsonl.exists() and lock.exists()):
                print(f"typesafe dataset skipped (offline): {exc}")
            elif jsonl.exists() and lock.exists():
                print(f"typesafe dataset: reusing cached copy (fetch failed: {exc})")
            else:
                raise
        paths["typesafe"] = jsonl

    if "perturbed" in datasets:
        jsonl = BENCH_CACHE / "perturbed.jsonl"
        lock = BENCH_CACHE / "perturbed.dataset.lock.json"
        _rebuild_if_needed(jsonl, lock, lambda: _build_perturbed(paths["bundled"]))
        paths["perturbed"] = jsonl

    for name in (
        "synthetic-labels",
        "synthetic-cardinality",
        "synthetic-injection",
        "synthetic-dependent",
    ):
        if name in datasets:
            set_name = name.removeprefix("synthetic-")
            jsonl = BENCH_CACHE / f"{set_name}.jsonl"
            lock = BENCH_CACHE / f"{set_name}.dataset.lock.json"
            _rebuild_if_needed(jsonl, lock, lambda n=set_name: _build_synthetic(n))
            paths[name] = jsonl

    return paths


def _rebuild_if_needed(jsonl: Path, lock: Path, build) -> None:
    """Rebuild the dataset when missing or when the previous build was partial."""
    if jsonl.exists() and lock.exists():
        return
    build()


def _build_bundled() -> None:
    from benchmarks.to_jsonl import main as to_jsonl_main

    out = str(BENCH_CACHE / "bundled.jsonl")
    print("building bundled dataset...")
    to_jsonl_main(["--out", out])


def _build_typesafe() -> None:
    from benchmarks.typesafe.fetch import main as fetch_main

    out = str(BENCH_CACHE / "typesafe.jsonl")
    print("building typesafe dataset (downloads from the network)...")
    rc = fetch_main(["--out", out])
    if rc != 0:
        raise OSError("typesafe fetch failed")


def _build_perturbed(bundled: Path) -> None:
    from benchmarks.perturb import main as perturb_main

    print("building perturbed dataset from bundled...")
    perturb_main(
        [
            "--in",
            str(bundled),
            "--out",
            str(BENCH_CACHE / "perturbed.jsonl"),
            "--variants",
            "3",
            "--seed",
            "0",
        ]
    )


def _build_synthetic(set_name: str) -> None:
    """Generate one synthetic set straight into the bench cache."""
    from benchmarks.synthetic import build_set

    print(f"building synthetic dataset '{set_name}'...")
    build_set(set_name, BENCH_CACHE)


def _track_scorer_grid(tracks: list[str], scorers: list[str]) -> list[tuple[str, str]]:
    """Valid (track, scorer) pairs: naive_local is scorer-independent."""
    grid = []
    for track in tracks:
        for scorer in scorers:
            if track == "naive_local" and scorer != "slots":
                continue  # naive generation has no scorer dimension
            grid.append((track, scorer))
    return grid


def _load_engine_with_timeout(model: str, load_timeout: float) -> tuple:
    """load_engine in a thread, joined with ``load_timeout`` seconds.

    Returns (model, tokenizer). On timeout the thread is abandoned (daemon;
    the process may keep it alive but the bench moves on) and TimeoutError
    is raised — callers treat it like any load failure.
    """
    import threading

    from jevmlx.engine import load_engine

    outcome: dict[str, Any] = {}

    def _load() -> None:
        try:
            outcome["result"] = load_engine(model)
        except BaseException as exc:  # noqa: BLE001 - the thread must not die silently
            outcome["error"] = exc

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    thread.join(load_timeout)
    if thread.is_alive():
        raise TimeoutError(f"load_engine did not finish within {load_timeout:.0f}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _write_failure_run(combo_dir: Path, status: str, exc: BaseException) -> Path:
    """Write <combo>/run.json with status load_failed/run_failed; return it."""
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
    payload = {
        "status": status,
        "error": {
            "type": type(exc).__name__,
            "message": first_line,
        },
        "environment": environment(),
    }
    run_path = combo_dir / "run.json"
    run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_path


def _combo_complete(combo_dir: Path) -> bool:
    """Resume rule: predictions.jsonl + run.json + report.json all exist."""
    return (
        (combo_dir / "predictions.jsonl").is_file()
        and (combo_dir / "run.json").is_file()
        and (combo_dir / "report.json").is_file()
    )


def run_bench(
    model: str,
    datasets: list[str],
    scorers: list[str],
    tracks: list[str],
    out: Path,
    runs: int,
    machine_override: str | None = None,
    force: bool = False,
    fresh: bool = False,
    load_timeout: float = 900.0,
) -> Path:
    """Run the full bench matrix for one model; returns the results folder.

    Failure is a result, not a crash: a model that cannot load (error or
    ``--load-timeout``) writes a ``load_failed`` run.json into every combo
    folder of that model, and a combo that fails mid-run writes
    ``run_failed``. Both get a SUMMARY.md row (accuracy column: ``load
    failed: <msg>`` / ``run failed: <msg>``). Completed combos
    (predictions + run.json + report.json) are skipped on re-invocation
    unless ``fresh`` forces a rerun — state is derived from files, no
    manifest. Exit code 0 unless EVERY combo failed.
    """
    tag = preflight(force, machine_override)
    print(f"machine: {tag}")

    dataset_paths = build_datasets(datasets)
    if not dataset_paths:
        raise SystemExit("no datasets selected")

    slug = model_slug(model)
    folder = out / f"{tag}-{slug}"
    folder.mkdir(parents=True, exist_ok=True)

    # Dataset locks travel with the results.
    for name in dataset_paths:
        lock_src = BENCH_CACHE / f"{name}.dataset.lock.json"
        if lock_src.exists():
            shutil.copy(lock_src, folder / f"{name}.dataset.lock.json")

    last_run: dict[str, dict[str, Any]] = {}
    failed_combos: dict[str, str] = {}
    combos = [
        (track, scorer, dataset)
        for track, scorer in _track_scorer_grid(tracks, scorers)
        for dataset in datasets
        if dataset in dataset_paths
    ]
    engine_loaded = False
    try:
        for track, scorer, dataset in combos:
            combo = f"{track}-{scorer}-{dataset}"
            combo_dir = folder / combo
            combo_dir.mkdir(parents=True, exist_ok=True)
            if not fresh and _combo_complete(combo_dir):
                print(f"=== {combo}: complete, skipping (--fresh to rerun) ===")
                continue
            print(f"=== {combo} ({runs} run(s)) ===")
            try:
                if not engine_loaded:
                    # Load once per model, lazily, inside the timeout guard.
                    _load_engine_with_timeout(model, load_timeout)
                    engine_loaded = True
                result = None
                for run_index in range(runs):
                    result = _run_one(model, track, scorer, dataset_paths[dataset], combo_dir)
                    print(f"  run {run_index + 1}/{runs} done")
                assert result is not None
                last_run[combo] = result
            except Exception as exc:  # noqa: BLE001 - failure is a result
                status = "load_failed" if not engine_loaded else "run_failed"
                _write_failure_run(combo_dir, status, exc)
                failed_combos[combo] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED combo {combo} ({status}): {failed_combos[combo]}", flush=True)
    finally:
        # One model load per track group is enough; drop it between tracks so
        # memory returns to baseline before the next track's runs.
        from jevmlx.engine import clear_engine_cache

        clear_engine_cache()

    if failed_combos:
        summarize(folder)
        for combo, err in failed_combos.items():
            print(f"combo {combo} FAILED: {err}")
        if len(failed_combos) == len(combos):
            detail = "; ".join(f"{c}: {e}" for c, e in failed_combos.items())
            raise SystemExit(f"every combo failed — {detail}")
    else:
        summarize(folder)
    _print_pr_instructions(folder, last_run)
    return folder


def run_bench_models(
    models: list[str],
    datasets: list[str],
    scorers: list[str],
    tracks: list[str],
    out: Path,
    runs: int,
    machine_override: str | None = None,
    force: bool = False,
    fresh: bool = False,
    load_timeout: float = 900.0,
) -> Path:
    """Run the bench matrix for several models, sequentially.

    Each model gets its own ``<machine>-<slug>/`` folder under ``out``; ONE
    ``SUMMARY.md`` is written at ``out`` covering all of them. Between models
    the engine cache and the Metal buffer cache are released and memory is
    logged before/after.

    Per-model failure handling is delegated to :func:`run_bench`, which
    records ``load_failed``/``run_failed`` run.json rows per combo (the one
    failure mechanism): a model whose engine cannot load leaves failure rows
    in every one of its combos and the remaining models still run. This
    wrapper only escalates when EVERY model left zero successful combos.
    """
    if not models:
        raise SystemExit("no models selected")
    all_combos = 0
    failed_combos = 0
    for model in models:
        print(f"\n=== model {model} ===", flush=True)
        try:
            run_bench(
                model=model,
                datasets=datasets,
                scorers=scorers,
                tracks=tracks,
                out=out,
                runs=runs,
                machine_override=machine_override,
                force=force,
                fresh=fresh,
                load_timeout=load_timeout,
            )
        except SystemExit as exc:
            # run_bench exits 1 only when EVERY of its combos failed.
            failed_combos += 1
            print(f"FAILED model {model}: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one model must not stop the next
            failed_combos += 1
            print(f"FAILED model {model}: {type(exc).__name__}: {exc}", flush=True)
        else:
            all_combos += 1
        finally:
            before = _metal_cache_memory_gb()
            from jevmlx.engine import clear_engine_cache

            clear_engine_cache()
            _clear_metal_cache()
            after = _metal_cache_memory_gb()
            print(
                f"[memory] released {model}: metal cache {before} GB -> {after} GB",
                flush=True,
            )
    if all_combos == 0 and failed_combos:
        summarize(out)  # failure rows still get a summary
        raise SystemExit("every model failed")
    summarize(out)
    return out


def _metal_cache_memory_gb() -> float:
    """Metal buffer-cache bytes in GB, or -1.0 when unreadable."""
    try:
        import mlx.core as mx

        return round(mx.metal.get_cache_memory() / 2**30, 2)
    except Exception:  # noqa: BLE001 - memory logging must never break the run
        return -1.0


def _clear_metal_cache() -> None:
    """Release the Metal buffer cache; never raises."""
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except Exception:  # noqa: BLE001 - cleanup must never break the run
        pass


def parse_model_list(comma_list: str) -> list[str]:
    """Split a comma-separated --model list, dropping empties and duplicates."""
    models: list[str] = []
    for part in comma_list.split(","):
        stripped = part.strip()
        if stripped and stripped not in models:
            models.append(stripped)
    return models


def parse_models_file(path: Path) -> list[str]:
    """One model id per line; blank lines and '#' comments ignored."""
    models: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and line not in models:
            models.append(line)
    if not models:
        raise SystemExit(f"{path}: no model ids found")
    return models


def _run_one(model: str, track: str, scorer: str, jsonl: Path, combo_dir: Path) -> dict:
    """One eval run (in-process) + metrics + report, into combo_dir."""
    cases = _load_cases(jsonl)
    from jevmlx.engine import load_engine

    model_obj, tokenizer = load_engine(model)
    chat_template = getattr(tokenizer, "chat_template", None)

    if track == "parallel":
        decide_fn = parallel_decide_fn(model_obj, tokenizer, scoring=scorer)
    else:
        from jevmlx.evalrun import naive_local_decide_fn

        decide_fn = naive_local_decide_fn(model_obj, tokenizer)

    permutations = "rotations" if track == "parallel" else "none"
    run = run_eval(
        cases,
        decide_fn,
        track=track,
        model=model,
        permutations=permutations,
        split="all",
        out_dir=str(combo_dir),
        extra_config={"scoring": scorer if track == "parallel" else "slots"},
        chat_template=chat_template,
        dataset_path=str(jsonl),
    )

    records = load_predictions(combo_dir / "predictions.jsonl")
    report_path = combo_dir / "report.json"
    write_report(report_path, {"environment": environment(), "metrics": compute_metrics(records)})
    print(f"  wrote {combo_dir}/predictions.jsonl, run.json, report.json, report.md")
    return {"run": run, "report": report_path}


def _load_cases(jsonl: Path) -> list[dict]:
    cases: list[dict] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(__import__("json").loads(line))
    return cases


def summarize(out: Path) -> Path:
    """Summarize every report.json under ``out`` into ``out/SUMMARY.md``."""
    from benchmarks.summarize_results import summarize as _summarize

    return _summarize(out)


def _print_pr_instructions(folder: Path, last_run: dict) -> None:
    n_combos = len(last_run)
    print(
        "\nBench complete.\n"
        f"Results folder: {folder}\n"
        f"Combos: {n_combos}; summary: {folder / 'SUMMARY.md'}\n\n"
        "To publish these numbers:\n"
        f"  1. git checkout -b bench-results-{folder.name}\n"
        f"  2. git add {folder}  (predictions may be gzipped; see the folder README)\n"
        f"  3. git commit -m 'Bench results: {folder.name}'\n"
        "  4. Open a PR against main with SUMMARY.md pasted into the description.\n"
        "Do NOT commit anything outside the results folder (no caches, no models)."
    )


def enforce_folder_size(folder: Path) -> bool:
    """Gzip any predictions.jsonl pushing the folder over 5 MB.

    Returns True when any file was compressed (the report should say so).
    """
    total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    if total <= MAX_FOLDER_BYTES:
        return False
    compressed = False
    for pred in folder.rglob("predictions.jsonl"):
        gz_path = pred.with_suffix(".jsonl.gz")
        with open(pred, "rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        pred.unlink()
        compressed = True
    return compressed


# Rough constant-bytes-per-parameter factors for common quantizations; the
# compat table's probe is the precise source but loading it would violate the
# dry-run never-loads rule, so these heuristics only power the plan printout.
_Q_SUFFIXES = ("4bit", "8bit", "mlxfp4")


def _model_memory_estimate(model: str) -> str:
    """Human memory estimate from the model id, else 'unknown'.

    Parses the parameter count from the id (e.g. ``7b``, ``1.5b``,
    ``0.5b``) and a quantization suffix (4bit ≈ 0.5 B/param + overhead,
    8bit ≈ 1.0). Never loads anything — this is a plan-time guess, the
    doctor/compat probe is the precise source.
    """
    import re

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*b\b", model.lower())
    if not m:
        return "unknown"
    params_b = float(m.group(1))
    if model.lower().endswith(("8bit", "8-bit")):
        bytes_per_param = 1.0
    elif model.lower().endswith(_Q_SUFFIXES) or "4bit" in model.lower():
        bytes_per_param = 0.55
    else:
        bytes_per_param = 2.0
    gb = params_b * bytes_per_param + 0.5  # + runtime overhead
    return f"~{gb:.1f} GB (estimate)"


def dry_run(
    models: list[str],
    datasets: list[str],
    scorers: list[str],
    tracks: list[str],
    out: Path,
    machine_override: str | None = None,
) -> int:
    """Print the run plan and exit 0 without loading anything."""
    tag = machine_tag(machine_override)
    print(f"machine: {tag}")
    print("datasets:")
    cached = build_datasets(datasets) if datasets else {}
    for name in datasets:
        path = cached.get(name)
        state = "cached" if path is not None and Path(path).is_file() else "to build"
        print(f"  {name}: {state}")
    print("combos:")
    for model in models:
        slug = model_slug(model)
        folder = out / f"{tag}-{slug}"
        print(f"  model {model} (memory {_model_memory_estimate(model)}) -> {folder}")
        for track, scorer in _track_scorer_grid(tracks, scorers):
            for dataset in datasets:
                combo = f"{track}-{scorer}-{dataset}"
                combo_dir = folder / combo
                state = "complete, would skip" if _combo_complete(combo_dir) else "would run"
                print(f"    {combo} -> {combo_dir} [{state}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jevmlx bench",
        description="One command: complete PR-ready benchmark results folder.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id(s) for mlx-lm; comma-separated list runs "
        "them sequentially, each into its own <machine>-<slug>/ folder, one "
        "SUMMARY.md across all",
    )
    parser.add_argument(
        "--models-file",
        default=None,
        help="path to a file with one model id per line ('#' comments allowed); overrides --model",
    )
    parser.add_argument(
        "--datasets",
        default="bundled,typesafe,perturbed",
        help="comma list: bundled,typesafe,perturbed,synthetic-*",
    )
    parser.add_argument("--scorers", default="slots,labels", help="comma list: slots,labels")
    parser.add_argument(
        "--tracks", default="parallel,naive_local", help="comma list: parallel,naive_local"
    )
    parser.add_argument(
        "--out", default=str(RESULTS_DIR), help="results root (default benchmarks/results)"
    )
    parser.add_argument("--runs", type=int, default=2, help="eval runs per combo (last one kept)")
    parser.add_argument("--machine", default=None, help="override the machine tag")
    parser.add_argument(
        "--force",
        action="store_true",
        help="run despite battery power or busy Metal memory (reasons are printed)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="rerun combos that already have complete results (default: skip them)",
    )
    parser.add_argument(
        "--load-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for load_engine before recording a load_failed row (default 900)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the machine tag, dataset build plan, combo list with output "
        "folders, and a memory estimate per model, then exit without loading anything",
    )
    args = parser.parse_args(argv)

    if args.models_file:
        models = parse_models_file(Path(args.models_file))
    else:
        models = parse_model_list(args.model)
    if not models:
        parser.error("no model ids given (--model or --models-file)")

    datasets = [d for d in (s.strip() for s in args.datasets.split(",")) if d]
    scorers = [s for s in (s.strip() for s in args.scorers.split(",")) if s]
    tracks = [t for t in (t.strip() for t in args.tracks.split(",")) if t]
    for name, values, allowed in (
        ("datasets", datasets, DATASETS),
        ("scorers", scorers, SCORERS),
        ("tracks", tracks, TRACKS),
    ):
        bad = [v for v in values if v not in allowed]
        if bad:
            parser.error(f"invalid {name}: {', '.join(bad)} (allowed: {', '.join(allowed)})")
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    if args.dry_run:
        return dry_run(
            models=models,
            datasets=datasets,
            scorers=scorers,
            tracks=tracks,
            out=Path(args.out),
            machine_override=args.machine,
        )

    if len(models) == 1:
        run_bench(
            model=models[0],
            datasets=datasets,
            scorers=scorers,
            tracks=tracks,
            out=Path(args.out),
            runs=args.runs,
            machine_override=args.machine,
            force=args.force,
            fresh=args.fresh,
            load_timeout=args.load_timeout,
        )
    else:
        run_bench_models(
            models=models,
            datasets=datasets,
            scorers=scorers,
            tracks=tracks,
            out=Path(args.out),
            runs=args.runs,
            machine_override=args.machine,
            force=args.force,
            fresh=args.fresh,
            load_timeout=args.load_timeout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
