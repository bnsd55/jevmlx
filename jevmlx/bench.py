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
import hashlib
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
    "typed-decisions",
    "perturbed",
    "synthetic-labels",
    "synthetic-cardinality",
    "synthetic-injection",
    "synthetic-dependent",
)

# W6-B5 public gold datasets (F2): the BARE name is accepted on the CLI as
# shorthand for both views; build_datasets only produces the VIEW-SUFFIXED
# cases files, so combos expand bare -> suffixed.
PUBLIC_DATASETS = ("ag_news", "boolq", "sst5")
PUBLIC_VIEWS = ("balanced", "natural")
PUBLIC_DATASET_NAMES = tuple(f"{name}.{view}" for name in PUBLIC_DATASETS for view in PUBLIC_VIEWS)
DATASETS_ALL = DATASETS + PUBLIC_DATASETS + PUBLIC_DATASET_NAMES


def normalize_dataset_names(names: list[str]) -> list[str]:
    """Expand bare public names to their two views, drop duplicates, keep
    order. Non-public names pass through unchanged.

    This is the ONE place bare-vs-suffixed is resolved (the CLI validator
    and the combo builder both call it), so the two can never drift.
    """
    expanded: list[str] = []
    for name in names:
        if name in PUBLIC_DATASETS:
            for view in PUBLIC_VIEWS:
                suffixed = f"{name}.{view}"
                if suffixed not in expanded:
                    expanded.append(suffixed)
        elif name not in expanded:
            expanded.append(name)
    return expanded


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


def build_datasets(
    datasets: list[str], offline_ok: bool = True
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Build eval JSONL datasets into the bench cache, reusing lock matches.

    Returns (dataset name -> JSONL path, dataset name -> lock path).
    ``typesafe`` is skipped with a clear message when offline; ``perturbed``
    derives from ``bundled``.
    """
    BENCH_CACHE.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths_locks: dict[str, Path] = {}

    if "bundled" in datasets:
        jsonl = BENCH_CACHE / "bundled.jsonl"
        lock = BENCH_CACHE / "bundled.dataset.lock.json"
        _rebuild_if_needed(jsonl, lock, _build_bundled)
        paths["bundled"] = jsonl
        paths_locks["bundled"] = lock

    for name, build in (("typesafe", _build_typesafe), ("typed-decisions", _build_typed_decisions)):
        if name not in datasets:
            continue
        jsonl = BENCH_CACHE / f"{name}.jsonl"
        lock = BENCH_CACHE / f"{name}.dataset.lock.json"
        try:
            _rebuild_if_needed(jsonl, lock, build)
        except OSError as exc:
            if offline_ok and not (jsonl.exists() and lock.exists()):
                print(f"{name} dataset skipped (offline): {exc}")
            elif jsonl.exists() and lock.exists():
                print(f"{name} dataset: reusing cached copy (fetch failed: {exc})")
            else:
                raise
        paths[name] = jsonl
        paths_locks[name] = lock

    # W6-B5 public gold datasets: two views each (balanced diagnostic /
    # natural distribution), each view its own cases file + lock. The
    # verifier re-checks the pinned file sha256s on cache reuse (F5). Both
    # the bare name and the view-suffixed names are accepted here (F2); the
    # requested views are tracked so only the asked-for files are built.
    requested_public = {
        (name[: -len(f".{view}")], view)
        for name in datasets
        for name_, view in [(name, name.split(".", 1)[-1])]
        if name in PUBLIC_DATASET_NAMES
    } | {(name, view) for name in datasets if name in PUBLIC_DATASETS for view in PUBLIC_VIEWS}
    for name in PUBLIC_DATASETS:
        for view in PUBLIC_VIEWS:
            if (name, view) not in requested_public:
                continue
            jsonl = BENCH_CACHE / f"{name}.{view}.jsonl"
            lock = BENCH_CACHE / f"{name}.{view}.dataset.lock.json"
            try:
                _rebuild_if_needed(
                    jsonl,
                    lock,
                    lambda n=name, v=view: _build_public_view(n, v),
                    verify=_public_pin_problem,
                )
            except OSError as exc:
                if offline_ok and not (jsonl.exists() and lock.exists()):
                    print(f"{name}.{view} dataset skipped (offline): {exc}")
                elif jsonl.exists() and lock.exists():
                    print(f"{name}.{view}: reusing cached copy (fetch failed: {exc})")
                else:
                    raise
            paths[f"{name}.{view}"] = jsonl
            paths_locks[f"{name}.{view}"] = lock

    if "perturbed" in datasets:
        jsonl = BENCH_CACHE / "perturbed.jsonl"
        lock = BENCH_CACHE / "perturbed.dataset.lock.json"
        _rebuild_if_needed(jsonl, lock, lambda: _build_perturbed(paths["bundled"]))
        paths["perturbed"] = jsonl
        paths_locks["perturbed"] = lock

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
            paths_locks[name] = lock

    return paths, paths_locks


def _rebuild_if_needed(jsonl: Path, lock: Path, build, verify=None) -> None:
    """Rebuild the dataset when missing, partial, or no longer matching its lock.

    The cached cases file is only trusted when it still hashes to the lock's
    ``cases_sha256`` (the same integrity rule the fetchers apply); a
    mismatched pair (truncated write, stale cache, edited file) rebuilds.
    ``verify`` (a lock -> problem-string-or-None callable) adds an extra
    validity gate on cache reuse — the public fetchers' pin re-check (F5):
    a lock whose recorded file sha256 no longer equals the hardcoded
    expectation forces a rebuild.
    """
    if jsonl.exists() and lock.exists():
        try:
            expected = json.loads(lock.read_text(encoding="utf-8")).get("cases_sha256")
        except (OSError, json.JSONDecodeError):
            expected = None
        if expected is not None and hashlib.sha256(jsonl.read_bytes()).hexdigest() == expected:
            if verify is not None:
                problem = verify(lock)
                if problem is None:
                    return
                print(f"{jsonl.name}: {problem} — rebuilding...")
            else:
                return
        print(f"{jsonl.name}: cached copy does not match its lock — rebuilding...")
    build()


def _build_bundled() -> None:
    from benchmarks.to_jsonl import main as to_jsonl_main

    out = str(BENCH_CACHE / "bundled.jsonl")
    print("building bundled dataset...")
    # --lock: the registered lock name is <dataset>.dataset.lock.json (build_datasets
    # reads it back from there); the converter's default 'dataset.lock.json'
    # would collide with other datasets sharing the cache dir and leave the
    # registered path missing — run.json's dataset_lock_sha256 came out null.
    to_jsonl_main(["--out", out, "--lock", str(BENCH_CACHE / "bundled.dataset.lock.json")])


def _build_typesafe() -> None:
    from benchmarks.typesafe.fetch import main as fetch_main

    out = str(BENCH_CACHE / "typesafe.jsonl")
    print("building typesafe dataset (downloads from the network)...")
    rc = fetch_main(["--out", out])
    if rc != 0:
        raise OSError("typesafe fetch failed")


def _public_pin_problem(lock: Path) -> str | None:
    """F5: on cache reuse, re-check the lock's recorded file sha256s against
    the hardcoded pin expectations. None = still the pinned bytes."""
    from benchmarks.public.fetch import DATASETS

    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "lock unreadable"
    sources = data.get("sources") or []
    if not sources:
        return "lock has no sources"
    source = sources[0]
    repo_id = source.get("repo_id")
    ds = next((d for d in DATASETS.values() if d.repo_id == repo_id), None)
    if ds is None:
        return f"unknown repo {repo_id}"
    if source.get("revision") != ds.revision:
        return f"revision drift: {source.get('revision')} != pin {ds.revision}"
    for file_path, recorded in (source.get("files") or {}).items():
        expected = ds.expected_sha256.get(file_path)
        if expected is None:
            return f"no pin expectation for {file_path}"
        if recorded != expected:
            return f"file sha256 drift for {file_path}"
    return None


def _build_public_view(name: str, view: str) -> None:
    """One public-gold view: fetch the pinned revision, sample, write.

    Uses the fetcher's DEFAULT sample sizes (50/class balanced, 500
    natural) — the bench does not thread --per-class/--natural-rows; those
    CLI knobs exist for direct `python -m benchmarks.public.fetch` runs.

    Both views come from one download (fetch_dataset downloads the split
    once and samples twice), so rebuilding 'balanced' also refreshes
    'natural' — _rebuild_if_needed's cases_sha256 check catches drift on
    either file independently.
    """
    from benchmarks.public.fetch import DATASETS, EVAL_SPLITS, fetch_dataset, write_view

    ds = DATASETS[name]
    print(f"building {name}.{view} dataset (downloads from the Hugging Face Hub)...")
    views, sha_by_file = fetch_dataset(ds, EVAL_SPLITS[name])
    for v, records in views.items():
        out = BENCH_CACHE / f"{name}.{v}.jsonl"
        lock = BENCH_CACHE / f"{name}.{v}.dataset.lock.json"
        write_view(records, out, lock, files_sha256=sha_by_file)
        print(f"  wrote {out.name} ({len(records)} cases)")


def _build_typed_decisions() -> None:
    from benchmarks.typed_decisions.fetch import main as fetch_main

    out = BENCH_CACHE / "typed-decisions.jsonl"
    lock = BENCH_CACHE / "typed-decisions.dataset.lock.json"
    print("building typed-decisions dataset (downloads from the Hugging Face Hub)...")
    # No except-wrapper: hub errors already subclass OSError, so letting them
    # propagate keeps the real cause visible (a wrapper would misreport every
    # bug as "offline"). The rc check covers fetchers that return nonzero
    # instead of raising.
    rc = fetch_main(["--out", str(out), "--lock", str(lock)])
    if rc != 0:
        raise OSError("typed-decisions fetch failed")


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


def _load_engine_with_timeout(model: str, load_timeout: float) -> Any:
    """load_engine in a thread, joined with ``load_timeout`` seconds.

    Returns the loaded :class:`Engine`. On timeout the thread is
    abandoned (daemon; the process may keep it alive but the bench moves on)
    and TimeoutError is raised — callers treat it like any load failure.
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


def _run_model_parity(model: str, engine, folder: Path) -> str | None:
    """W4-B: run the scoring-parity check on a loaded engine and write
    ``<folder>/parity.json``. Returns None on pass, else a short failure
    note for the summary rows. Takes the loaded :class:`Engine`."""
    from jevmlx.parity import write_parity_json

    payload = write_parity_json(engine, model, folder)
    if payload["passed"]:
        print(
            f"parity: PASS (max drift {payload['max_abs_drift_nats']} nats, "
            f"atol {payload['atol']}, {len(payload['cases'])} cases)"
        )
        return None
    print(
        f"parity: FAIL (winners_identical={payload['winners_identical']}, "
        f"max drift {payload['max_abs_drift_nats']} nats, atol {payload['atol']})"
    )
    return (
        f"parity_failed: max_abs_drift_nats={payload['max_abs_drift_nats']} "
        f"atol={payload['atol']} winners_identical={payload['winners_identical']}"
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
    metal_cache_gb: float = 8.0,
    heartbeat_every: int = 25,
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
    # Import here, not at module level: summarize_results imports
    # enforce_folder_size from this module, so a top-level import would be
    # circular. (No wrapper: callers import summarize_results directly.)
    from benchmarks.summarize_results import summarize

    tag = preflight(force, machine_override)
    print(f"machine: {tag}")

    # W5c-14 (#79): cap the Metal buffer cache for the whole run. The
    # per-combo clear (W5c-13) releases the cache between combos, but inside
    # a long combo (typesafe, 426+ cases with rotations) the Metal allocator
    # hoards freed buffers and pushes the machine into swap. set_cache_limit
    # makes the allocator evict buffers above the cap instead of hoarding —
    # no per-case clear needed. Default 8 GB (the 7B weights are ~4 GB; the
    # cap bounds the GPU leftover, not the live working set). Best-effort:
    # a non-Metal build logs and continues.
    metal_cache_limit_bytes = _set_metal_cache_limit(metal_cache_gb)
    if metal_cache_limit_bytes is not None:
        print(
            f"[memory] Metal buffer cache cap: {metal_cache_gb} GB "
            f"({metal_cache_limit_bytes} bytes)",
            flush=True,
        )

    dataset_paths, dataset_locks = build_datasets(datasets)
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

    # Bare public names expand to both views through the ONE normalizer
    # (the same helper the CLI validator used) — no second, drifting
    # expansion rule. Only names with a built cases file become combos.
    combo_dataset_names = [
        name for name in normalize_dataset_names(datasets) if name in dataset_paths
    ]
    combos = [
        (track, scorer, dataset)
        for track, scorer in _track_scorer_grid(tracks, scorers)
        for dataset in combo_dataset_names
    ]
    engine_loaded = False
    parity_note: str | None = None  # W4-B: set when the parity check fails
    try:
        for track, scorer, dataset in combos:
            combo = f"{track}-{scorer}-{dataset}"
            combo_dir = folder / combo
            combo_dir.mkdir(parents=True, exist_ok=True)
            # N3: --fresh deletes the combo dir (explicit, logged) so the
            # run starts clean. Otherwise: manifest present -> resume,
            # absent -> fresh.
            if fresh and combo_dir.exists():
                import shutil as _shutil

                print(f"=== {combo}: --fresh, removing existing dir ===")
                _shutil.rmtree(combo_dir)
                combo_dir.mkdir(parents=True, exist_ok=True)
            _has_manifest = (combo_dir / "manifest.json").is_file()
            print(f"=== {combo} ({runs} run(s)) [{'resume' if _has_manifest else 'fresh'}] ===")
            # W5c-13 (#79): reset the Metal peak-memory counter at combo
            # start so each combo's peak is its own (not cumulative across
            # the matrix). The clear happens AFTER the combo (below).
            _reset_metal_peak_memory()
            try:
                if not engine_loaded:
                    # Load once per model, lazily, inside the timeout guard.
                    _load_engine_with_timeout(model, load_timeout)
                    engine_loaded = True
                    # W4-B: scoring parity (batch=1 vs batched vs chunked)
                    # over the bundled presets, recorded as parity.json in
                    # the model folder. A model that cannot demonstrate
                    # scoring parity gets NO accuracy numbers: every combo
                    # row is marked parity_failed in SUMMARY.md (the README
                    # compat gate, PR #29's check_parity, reads the same
                    # file). Runs BEFORE any eval combo so a parity-failing
                    # model never produces unvetted accuracy rows. The engine
                    # comes from _load_engine_with_timeout (already run above
                    # — its result was discarded; recover it from the cache
                    # via load_engine, which is a lru-cached call) and parity
                    # failures are a RESULT, not an exception: a parity crash
                    # marks the model parity_failed and the combos continue.
                    from jevmlx.engine import load_engine

                    try:
                        engine = load_engine(model)
                        parity_note = _run_model_parity(model, engine, folder)
                    except Exception as parity_exc:  # noqa: BLE001 - failure is a result
                        parity_note = f"parity_failed: {type(parity_exc).__name__}: {parity_exc}"
                        print(f"parity check error: {parity_note}", flush=True)
                result = None
                for run_index in range(runs):
                    if run_index > 0:
                        # W5c-12: repeat runs start from a clean combo dir.
                        # 'last one kept' semantics: run 1's output is
                        # discarded (rmtree + recreate), then run 2+ runs
                        # fresh (resume=False). --resume applies to the FIRST
                        # run only (manifest present -> resume run 1).
                        import shutil as _shutil

                        print(f"=== {combo}: run {run_index + 1}/{runs}, cleaning combo dir ===")
                        _shutil.rmtree(combo_dir)
                        combo_dir.mkdir(parents=True, exist_ok=True)
                    result = _run_one(
                        model,
                        track,
                        scorer,
                        dataset_paths[dataset],
                        combo_dir,
                        dataset_lock_path=dataset_locks.get(dataset),
                        # --resume applies to the first run only (manifest
                        # present -> resume). Repeat runs are always fresh.
                        resume=_has_manifest and run_index == 0,
                        heartbeat_every=heartbeat_every,
                        combo=combo,
                    )
                    print(f"  run {run_index + 1}/{runs} done")
                assert result is not None
                last_run[combo] = result
                # W5c-13 (#79): sample Metal memory AFTER the combo, then
                # clear the buffer cache so it does not accumulate across
                # combos (the single-model run_bench path never called
                # _clear_metal_cache before — only run_bench_models did,
                # between models). The memory block lands in the bench log
                # and the combo's run.json (additive 'memory' key; the
                # check_results contract accepts optional keys).
                mem = _sample_metal_memory()
                print(f"  [memory] {combo}: {_memory_block_gb(mem)}", flush=True)
                _augment_run_json_memory(
                    combo_dir, mem, metal_cache_limit_bytes=metal_cache_limit_bytes
                )
                _clear_metal_cache()
            except Exception as exc:  # noqa: BLE001 - failure is a result
                status = "load_failed" if not engine_loaded else "run_failed"
                _write_failure_run(combo_dir, status, exc)
                failed_combos[combo] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED combo {combo} ({status}): {failed_combos[combo]}", flush=True)
    finally:
        # After ALL combos: drop the engine/weights cache so memory returns
        # to baseline before the caller (or the next model in
        # run_bench_models) proceeds. The per-combo Metal buffer-cache
        # clear is above (W5c-13 #79); this releases the model itself.
        from jevmlx.engine import clear_engine_cache

        clear_engine_cache()

    if failed_combos:
        summarize(folder, parity_note=parity_note)
        for combo, err in failed_combos.items():
            print(f"combo {combo} FAILED: {err}")
        if len(failed_combos) == len(combos):
            detail = "; ".join(f"{c}: {e}" for c, e in failed_combos.items())
            raise SystemExit(f"every combo failed — {detail}")
    else:
        summarize(folder, parity_note=parity_note)
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
    metal_cache_gb: float = 8.0,
    heartbeat_every: int = 25,
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
    # Same-cycle note as run_bench: imported here, not at module level.
    from benchmarks.summarize_results import summarize
    from jevmlx.engine import clear_engine_cache

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
                metal_cache_gb=metal_cache_gb,
                heartbeat_every=heartbeat_every,
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
    # W4-B: parity.json lives in each model folder; the summarizer reads it
    # per model folder and marks parity_failed rows itself.
    summarize(out)
    return out


def _metal_cache_memory_gb() -> float:
    """Metal buffer-cache bytes in GB, or -1.0 when unreadable."""
    try:
        import mlx.core as mx

        return round(mx.get_cache_memory() / 2**30, 2)
    except Exception:  # noqa: BLE001 - memory logging must never break the run
        return -1.0


def _clear_metal_cache() -> None:
    """Release the Metal buffer cache; never raises."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001 - cleanup must never break the run
        pass


def _set_metal_cache_limit(cache_gb: float) -> int | None:
    """Cap the Metal buffer cache (W5c-14 #79); returns the bytes set, or None.

    The per-combo clear (W5c-13) releases the cache between combos, but
    inside a long combo the Metal allocator hoards freed buffers and pushes
    the machine into swap. ``mx.set_cache_limit`` makes the allocator
    evict buffers above the cap instead of hoarding — no per-case clear
    needed. Best-effort: a non-Metal build returns None and never raises.
    """
    try:
        import mlx.core as mx

        limit_bytes = int(float(cache_gb) * 2**30)
        mx.set_cache_limit(limit_bytes)
        return limit_bytes
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the run
        print(
            f"[memory] Metal buffer cache cap NOT set ({exc!r}); the allocator may hoard",
            flush=True,
        )
        return None


# W5c-13 (#79): per-combo Metal memory telemetry. The three counters the
# Metal API exposes, sampled after a combo finishes, plus a peak reset at
# combo start so each combo's peak is its own (not cumulative across the
# matrix). All best-effort: a non-Metal build / import failure yields -1
# and never breaks the run.
_MEMORY_KEYS = ("peak_memory", "active_memory", "cache_memory")


def _reset_metal_peak_memory() -> None:
    """Reset the Metal peak-memory counter; never raises (best-effort)."""
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:  # noqa: BLE001 - telemetry must never break the run
        pass


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


def _augment_run_json_memory(
    combo_dir: Path, mem: dict[str, int], *, metal_cache_limit_bytes: int | None = None
) -> None:
    """Add the 'memory' block to the combo's run.json (W5c-13 #79, W5c-14).

    run.json is written by jevmlx.evalrun.run_eval (out of this PR's
    scope); this reads it back, attaches the per-combo Metal memory
    sample as an additive 'memory' key, and writes it. Best-effort: a
    missing/corrupt run.json is skipped (the eval result is already on
    disk). The check_results contract accepts unknown-but-present optional
    keys (benchmarks/check_results.py: 'unknown-but-present optional keys
    are not type-checked'). W5c-14: the cache cap set at run start is
    recorded as metal_cache_limit_bytes (None when the cap could not be set).
    """
    import json

    run_json = combo_dir / "run.json"
    try:
        run = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    run["memory"] = {
        "peak_memory_bytes": mem["peak_memory"],
        "active_memory_bytes": mem["active_memory"],
        "cache_memory_bytes": mem["cache_memory"],
        "metal_cache_limit_bytes": metal_cache_limit_bytes,
    }
    try:
        run_json.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
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


def _run_one(
    model: str,
    track: str,
    scorer: str,
    jsonl: Path,
    combo_dir: Path,
    dataset_lock_path: Path | None = None,
    resume: bool = False,
    heartbeat_every: int = 0,
    combo: str = "",
) -> dict:
    """One eval run (in-process) + metrics + report, into combo_dir."""
    cases = _load_cases(jsonl)
    from jevmlx.engine import load_engine

    engine = load_engine(model)
    chat_template = getattr(engine.tokenizer, "chat_template", None)

    if track == "parallel":
        decide_fn = parallel_decide_fn(engine, scoring=scorer)
    else:
        from jevmlx.evalrun import naive_local_decide_fn

        decide_fn = naive_local_decide_fn(engine)

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
        # Provenance (F7): the dataset lock's sha256 lands in run.json so
        # leaderboard rows are traceable to a pinned dataset revision.
        dataset_lock_path=str(dataset_lock_path) if dataset_lock_path else None,
        # Consensus datasets (typesafe, typed-decisions) carry per-field
        # distributions; lines get theirs so tvd_vs_consensus can run. Other
        # datasets have no meta.consensus and are unaffected.
        carry_consensus=True,
        resume=resume,
        heartbeat_every=heartbeat_every,
        combo=combo,
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


def _print_pr_instructions(folder: Path, last_run: dict) -> None:
    n_combos = len(last_run)
    print(
        "\nBench complete.\n"
        f"Results folder: {folder}\n"
        f"Combos: {n_combos}; summary: {folder / 'SUMMARY.md'}\n\n"
        "To publish these numbers (as a human contributor):\n"
        f"  1. git checkout -b bench-results-{folder.name}\n"
        f"  2. git add {folder}  (predictions may be gzipped; see the folder README)\n"
        f"  3. git commit -m 'Bench results: {folder.name}'\n"
        "  4. Push the branch to your fork and open a pull request against\n"
        "     main with SUMMARY.md pasted into the description.\n"
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
    # Bare public names expand here too — the plan must show the SAME
    # combos the run would execute.
    combo_names = normalize_dataset_names(datasets)
    print("datasets:")
    cached, _locks = build_datasets(datasets) if datasets else ({}, {})
    for name in combo_names:
        path = cached.get(name)
        state = "cached" if path is not None and Path(path).is_file() else "to build"
        print(f"  {name}: {state}")
    print("combos:")
    for model in models:
        slug = model_slug(model)
        folder = out / f"{tag}-{slug}"
        print(f"  model {model} (memory {_model_memory_estimate(model)}) -> {folder}")
        for track, scorer in _track_scorer_grid(tracks, scorers):
            for dataset in combo_names:
                combo = f"{track}-{scorer}-{dataset}"
                combo_dir = folder / combo
                state = (
                    "would resume (has predictions)"
                    if (combo_dir / "predictions.jsonl").is_file()
                    else "would run"
                )
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
        help="comma list: bundled,typesafe,typed-decisions,perturbed,"
        "synthetic-*,ag_news,ag_news.balanced,ag_news.natural,boolq,"
        "boolq.balanced,boolq.natural,sst5,sst5.balanced,sst5.natural",
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
        "--metal-cache-gb",
        type=float,
        default=8.0,
        help=(
            "cap the Metal buffer cache in GB (default 8); the allocator evicts "
            "freed buffers above the cap instead of hoarding them (#79)"
        ),
    )
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=25,
        help=(
            "print a heartbeat every N completed cases (default 25; 0 disables). "
            "One line to stdout + one JSON record per heartbeat to "
            "<combo>/heartbeat.jsonl"
        ),
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
        ("datasets", datasets, DATASETS_ALL),
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
            metal_cache_gb=args.metal_cache_gb,
            heartbeat_every=args.heartbeat_every,
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
            metal_cache_gb=args.metal_cache_gb,
            heartbeat_every=args.heartbeat_every,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
