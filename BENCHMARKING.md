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
