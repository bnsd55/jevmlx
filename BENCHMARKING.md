# BENCHMARKING.md — contributing results from your Apple Silicon Mac

One PR adds one results folder under `benchmarks/results/`, produced by a
single command. Four steps:

## 1. Clone and set up

```bash
git clone https://github.com/bnsd55/jevmlx && cd jevmlx
./setup.sh
```

## 2. Run the bench

```bash
.venv/bin/jevmlx bench --model quality
```

(or `source .venv/bin/activate` first, then plain `jevmlx bench ...`)

Several models in one sitting: pass a comma list or a file (one id per line,
`#` comments allowed) — models run sequentially, each into its own folder, and
one `SUMMARY.md` covers all of them:

```bash
.venv/bin/jevmlx bench --models-file models.txt
```

Plug in first; the command refuses to run on battery (override: `--force`).
It preflights the machine, builds the datasets into `~/.cache/jevmlx/bench/`,
runs every (track, scorer, dataset) combination twice and keeps the warm run,
writes `benchmarks/results/<machine>-<model-slug>/...`, and prints
`SUMMARY.md` plus the exact PR instructions.

Check the plan before spending an evening of GPU: `--dry-run` prints the
machine tag, which datasets are cached vs to-build, every combo with its
output folder, and a memory estimate per model — then exits without loading
anything.

Failure is a result, not a crash: a model that cannot load (or exceeds
`--load-timeout`, default 900s) writes a `load_failed` `run.json` into each
of its combos, and a combo that fails mid-run writes `run_failed` — both
appear in `SUMMARY.md` with the error in the field-acc column. Exit code is
0 as long as at least one combo succeeded; 1 only when EVERY combo failed.
Reruns resume: a combo with `predictions.jsonl` + `run.json` + `report.json`
tracks as complete and is skipped with a log line (state comes from the
files, no manifest). Pass `--fresh` to force reruns.

## 3. Commit the results folder

```bash
git checkout -b bench-results-<machine>-<model-slug>
git add benchmarks/results/<machine>-<model-slug>
git commit -m "Bench results: <machine>-<model-slug>"
git push -u origin HEAD
```

## 4. Open the PR

Paste `SUMMARY.md` into the PR description and link the machine specs
(chip, RAM, macOS version from `report.json`).

## What the command does

- Preflight: Apple Silicon check; refuses on battery or when another process
  holds significant Metal memory (`--force` overrides with a printed warning).
- Datasets: bundled cases, TypeSafe's public set (skipped offline), and
  deterministic perturbations of the bundled cases — built once into
  `~/.cache/jevmlx/bench/` and reused while the lock files match.
- Runs: for every (track, scorer, dataset) — `eval` in-process `--runs` times
  (default 2), last run kept, order-rotation permutations on for the parallel
  track. Writes `predictions.jsonl`, `run.json`, `report.json`, `report.md`
  per combination, then `SUMMARY.md` across all of them. Completed combos
  are skipped on rerun (`--fresh` overrides); load/run failures become
  `load_failed`/`run_failed` rows in the summary instead of crashing.

## What NOT to commit

Anything outside `benchmarks/results/<machine>-<model-slug>/`: caches
(`~/.cache/jevmlx/`), model weights, logs, editor files. Predictions larger
than 5 MB total are gzipped automatically (the folder README says so).

## PR checklist

- [ ] Folder contains only the bench output (predictions, run.json,
      report.json/.md, dataset lock files, SUMMARY.md, folder README)
- [ ] `parity.json` in the model folder — records the passing slow parity
      test (W1-A batch vs chunked log_score agreement). A model cannot
      enter the README leaderboard without it. Schema:
      `{"model": "...", "test": "test_w1a_...", "passed": true,
      "max_drift_nats": 0.027, "atol": 0.05, "run_at": "..."}`
- [ ] `SUMMARY.md` pasted into the PR description
- [ ] Machine specs (chip, RAM, macOS) mentioned in the PR body
- [ ] No hand-edited numbers — recompute instead of fixing up
- [ ] `results-check` CI workflow is green (runs `python -m
      benchmarks.check_results` on the changed folders: contract keys, folder
      size, dataset lock, and report reproducibility — all must pass before
      merge)
- [ ] Ran `python -m benchmarks.leaderboard --results benchmarks/results
      --readme README.md` so the README leaderboard block is up to date
      (the `--check-readme` freshness gate in `results-check` CI enforces this)

## M5 runbook: one command for the whole gate sequence

`benchmarks/m5.py` chains the full milestone-gate sequence in order, each step
logged with wall time and exit status into `<out>/RUNBOOK.md`:

```bash
python -m benchmarks.m5 --out m5-2026-09-18 [--ab-branch w2a-field-local]
```

Steps in order: (1) `jevmlx doctor --json` (gate — any FAIL aborts); (2)
`pytest -m slow` once per parity model (Qwen3-8B, Llama-3.1-8B, Gemma-3-12B by
default; override with `--parity-models` or `--models-file`, the id rides the
`MODEL_ID` env var); (3) `jevmlx bench --model quality`; (4)
`benchmarks.invariance` on quality with `--extra 1,5,20,40` over the TypeSafe
cases (fetched automatically when missing); (5) `benchmarks.timing --model
quality --reps 5`; (6) `jevmlx bench --models-file` for the remaining parity
models; (7) with `--ab-branch`: a temp worktree of that branch with its own
venv, steps 3+4 rerun there, worktree removed afterwards; (8) `<out>/SUMMARY.md`
comparing main vs A/B (agreement/accuracy, flip rate, log-odds drift, time per
case, peak memory) from the produced json files.

Idempotent: steps whose output markers already exist are skipped (`--fresh`
reruns everything). Per-step logs land in `<out>/<step-id>.log`. Interrupted
runs resume; the gate step keeps a half-finished evening from wasting GPU time
on a broken environment.

### MODEL_ID env: the slow suite's model selector

`tests/conftest.py` reads `MODEL_ID` from the environment in ONE place
(`os.environ.get("MODEL_ID", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")`)
and exposes it as the `MODEL_ID` constant. Every slow test that loads a real
model reads that constant (or the `engine` fixture, which does); no test
hardcodes a model id. So when the M5 parity step runs `pytest -m slow` with
`MODEL_ID=<parity model>`, every slow test exercises THAT model — not the
0.5B dev default.

Tolerances are model-sensitive: `PARITY_ATOL` (== `INSTABILITY_BAND = 5e-2`)
is the batch-vs-chunked log_score drift measured on the 0.5B dev model.
Larger models (1.5B, 7B) see higher raw drift from Metal batch-shape matmul
variation, so `test_w4b_parity.py::test_parity_real_model_twin` (which runs
all four bundled presets and asserts `< PARITY_ATOL`) is **default-model-
only** — it skips when `MODEL_ID` is not the 0.5B default. The W1-A / T4
batch-vs-chunked parity tests use the same tolerance but only on the two
`fintech_fraud` / `support_triage` presets (smaller drift), so they stay on
all models; if a parity model ever drifts past `PARITY_ATOL` there, raise
the constant in `jevmlx/engine.py` (it is shared by the engine's tie flag).
