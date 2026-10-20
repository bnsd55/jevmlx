# BENCHMARKING.md — contributing results from your Apple Silicon Mac

One PR adds one results folder under `benchmarks/results/`, produced by a
single command. Four steps:

## 1. Clone and set up

```bash
git clone https://github.com/bnsd55/jevmlx && cd jevmlx
./setup.sh
```

One venv PER checkout/worktree: `./setup.sh` creates `.venv/` inside the
checkout it runs in. If you run from a git worktree, run `./setup.sh`
there too — `jevmlx doctor` intentionally FAILS when the python you run
imports a different jevmlx tree than the one the venv installed (the
benchmarked code would not be the installed code).

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
Reruns resume: a combo with `manifest.json` present resumes (`resume=True`
passed to `run_eval`); completed cases are skipped via the commit journal
(`completed_cases.jsonl`). A combo without a manifest is a fresh run. Use
`jevmlx eval --resume` to resume a crashed/interrupted eval run into an
existing output dir — the manifest is verified (config/code/model/tokenizer/
prompt/machine must match), `predictions.jsonl` is truncated to the last
committed byte offset (discarding any half-written trailing data), and
completed cases are skipped. Pass `--fresh` to force reruns.
With `--runs N > 1`, the first run resumes if a manifest is present, and
each subsequent run starts from a clean combo dir (the prior run's output
is removed) so only the last run's predictions are kept — 'last one kept'
semantics, no cross-run contamination.

### Watching a run

A many-hour bench/m5 run is blind without a watcher. `jevmlx watch` is a
read-only dashboard that renders from the files the run already writes — the
run process is never touched.

```bash
# Terminal mode (default): rich Live redraw in the terminal
jevmlx watch benchmarks/results/<machine>-<model-slug> [--refresh 2]

# Web mode: serve the control-room page (live via SSE, no reload)
jevmlx watch <out-dir> --web [--port 8765] [--no-tty]

# Or start the watcher in the same terminal as the bench:
jevmlx bench --ui --model <model> --out <out-dir> ...
jevmlx bench --ui --web --no-tty --model <model> --out <out-dir> ...
```

Terminal mode (default) is a compact rich `Live` view: step states,
combo progress, and the last few prediction lines. Keys: `q` quit,
`p` pause.

`--web` serves a static control-room page at `http://127.0.0.1:PORT/` — a
single self-contained HTML file (inline CSS + vanilla JS, no framework, no
CDN) that is **live via SSE**: the first paint fetches `/dashboard.json` once,
then an `EventSource('/events')` connection pushes a new dashboard payload
whenever a watched file changes (no full-page reload, no `root.innerHTML`
wipe). The server watches the mtimes of `RUNBOOK.md` and every
`heartbeat.jsonl` / `run.json` / `predictions.jsonl` under `<out>` (a cheap
`os.stat` scan every `refresh` seconds) and emits `event: dashboard` with the
full `build_dashboard` JSON on change, or a `: keepalive` comment every 15 s
if nothing changed. The browser patches the DOM in place (text, bar widths,
keyed row insert/update/remove) so selection, filters, sort, scroll, and open
`<details>` are preserved across updates. On disconnect a small
"disconnected" pill appears in the top bar; `EventSource` reconnects
automatically. The page renders client-side and has, top to bottom:

- **Top bar**: run name, freeze hash, machine + mlx version, attempt number
  + start time, awake time + sleep-guard state, state badge (running /
  stopped / done) and alert count.
- **NOW**: the live combo's model, track, scorer, dataset, run i/N, a
  progress bar (cases done/total, cases/h, ETA), pred lines, combo elapsed,
  and running accuracy vs majority — from the latest heartbeat.
- **MEMORY**: cache, active, and peak GB as bars against the machine total,
  with the 8 GB cap tick and the 10 GB stop-line tick; cache/peak turn red
  over the stop line.
- **HEALTH**: one pill per stop/flag rule (cache over stop, sleep windows,
  run_failed combos, parity-fail models, metal alloc retries, heartbeat
  age) plus any active alert text.
- **PIPELINE**: the m5 steps of the current attempt only, each with state
  (ok / running / failed / skipped / waiting), wall, exit code; click a
  step for its argv, stdout tail, and error in a popover.
- **FAILURE DIAGNOSIS**: the first failed step's argv, stdout tail, and
  error; previous failures of the run are collapsed in a `<details>`.
- **AGGREGATES SO FAR**: field accuracy (parallel labels + naive), exact
  record, time per case (parallel + naive), parity counts (pass / fail), cases scored.
- **RESULTS**: one row per combo (model × dataset × scorer × track) with
  the same columns as the README leaderboard (field accuracy + Wilson CI,
  majority, exact, parity status word (PASS/FAIL), time/case, calls, cases, A/B Δ).
  Filter chips (model / dataset / scorer / track / status) + a free-text
  filter; click a header to sort; click a row to load its questions below.
- **QUESTIONS**: every decision of the selected combo (time, case +
  rotation, field, predicted, label, ok, p(pred), margin, call ms, acc so
  far). Filter chips (wrong only / near-tie / order flips / workflow /
  field) + free-text; click a row to open the question card: context text,
  options with probability bars (predicted + label tagged), margin,
  same-field rotations, rescored flag, log-odds drift.
- **EVENTS**: heartbeats, combo completions, alloc retries, step completions,
  attempt markers, failures — newest first.
- **HISTORY**: previous attempts of this run dir, collapsed in
  `<details>` blocks.

`/dashboard.json` returns the raw 9-key contract for scripts;
`/questions.json?combo=<id>` returns the flat question list;
`/events` is the SSE stream (live updates). Null fields
render as a dash, never break the page. Half-written trailing lines are
truncated (same rule as `--resume`); missing files show `—`; no parse error
ever raises. `--no-tty` runs web only; `--web` without `--no-tty` runs both
the terminal render and the server.

When run via `bench --ui`, bench stdout goes to `<out>/bench.log` so the
screen stays clean; on exit the `SUMMARY.md` path is printed.

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

- The SUMMARY `p50 latency (ms)` column is the call-level
  `per_item_end_to_end_ms` median from `timing.json` (one value per decide
  call, not per prediction line) so parallel and naive tracks are compared
  fairly; the `calls` column shows the call count so rotations are visible.
- Preflight: Apple Silicon check; refuses on battery or when another process
  holds significant Metal memory (`--force` overrides with a printed warning).
- Datasets: bundled cases, TypeSafe's public set (skipped offline), the
  typed-decisions Hugging Face mirror, the three PUBLIC gold datasets
  (below), the two OpenJev datasets (below), and deterministic
  perturbations of the bundled cases — built once into
  `~/.cache/jevmlx/bench/` and reused while the lock files match.
  **Lock convention:** every builder writes `<name>.dataset.lock.json` next
  to `<name>.jsonl` (one lock per dataset, registered name); `build_datasets`
  fails fast if any registered lock is missing after build — before any model
  loads — so a broken lock is one error, not a per-combo `OSError` after an
  8-minute model load.
  Perturbation kinds: four context-side (`ws` whitespace
  normalisation, `preamble` neutral preamble, `numfmt` number reformatting,
  `shuffle` block-order shuffle) plus two label-preserving SCHEMA kinds —
  `optrev` (reverse every enum field's option order; labels are option
  descriptions so they stay valid) and `criterion` (prefix each field
  description with an evidence-grounding instruction). All are deterministic;
  `perturbation_flip_rate` picks up every kind via `group_id` +
  `meta.perturbation` with no metric change.
- Public gold datasets (pass `--datasets` `ag_news`, `boolq`,
  `sst5`, or view-suffixed names like `ag_news.balanced`; a bare name
  runs both views): AG News (4-class topic enum), BoolQ (yes/no reading
  comprehension), SST-5 (ordinal 0-4 sentiment). Each is PINNED to a
  dataset repo commit sha with a hardcoded per-file EXPECTED sha256:
  verified on download, and re-checked on cache reuse by comparing the
  LOCK's recorded sha256 against the pin (the cached cases bytes are
  covered by the lock's own cases_sha256) — a mismatch fails closed
  (nothing is sampled or locked). TWO DISJOINT
  sampling views are written as separate cases files:
  `<name>.balanced.jsonl` (class-balanced diagnostic, 50 rows/class —
  per-class accuracy, macro-F1, confusion, ordinal MAE) and
  `<name>.natural.jsonl` (the dataset's own class prevalence, 500 rows
  drawn from rows the balanced view did NOT take, so calibration rows
  are never the reported diagnostic rows — the view NLL/Brier/ECE
  describe). Pass `--datasets authored144,perturbations108` for the two
  OpenJev datasets: `authored144` (144 rows, 36 groups × 4 variants,
  3-way evidence interpretation — `supported`/`insufficient`/`contradicted`)
  and `perturbations108` (108 rows, split `rebase_stability`, 36 base rows
  × 3 label-preserving perturbation variants — `option_reversal`,
  `criterion_wrapper`, `irrelevant_context`; each row's
  `provenance.base_id` clusters it with its authored144 base under
  `group_id`). Source: github.com/TheoLeeCJ/openjev, pinned by commit sha +
  per-file EXPECTED sha256 (fail-closed, same guard as the public gold).
  Both are **model-reviewed, not human-adjudicated**
  (`meta.annotation_status` copied verbatim from provenance); the
  leaderboard row says so. No sampling views — every row is an eval row.
  describe). Selection is deterministic: `hash(seed, source_row_id)`.
  License/terms are recorded per dataset in the lock and redistribution
  is NOT cleared for any of them: the cached cases files (under
  `~/.cache/jevmlx/bench/`, never committed) carry the source text the
  engine classifies, while every committed RESULT artifact
  (predictions.jsonl, run.json, report.json, dataset.lock.json) stores
  row ids, split, option order and input hashes only — never the text.
  Each combo's run.json carries `dataset_lock_sha256`: the sha256 of the
  dataset lock file for the combo's dataset (provenance: leaderboard rows
  trace to a pinned dataset revision; a missing lock aborts the run with
  an error, it never records null).
- **Engine provenance (`environment.git_sha`):** every run.json records
  the jevmlx commit the engine ran on under `environment.git_sha`. Commits
  before 2026-09-22 were rewritten with identical trees (author metadata
  correction); the old sha no longer resolves but the tree is unchanged.
  Mapping: `dbb1ff1f04a0bce4b20817e4c09eae2b5a3d2ad3` →
  `4f74bd87fd19509aa6b019175b59aad2733a0b7f`. `check_results` does not
  fail on a git_sha absent from history (it never resolved the sha).
- **jabr classifier-benchmark** (pass `--datasets jabr`): a
  public-domain (CC0) benchmark from
  https://github.com/jabr/classifier-benchmark — 8 classification tasks /
  78 cases covering the three System One primitives: `choice`
  (support_department 5-way, email_intent 5-way) -> enum, `noul`
  (secret_leak, urgency, refund_eligible) -> boolean, `score`
  (frustration_level 3-level, incident_severity 5-level,
  review_sentiment 5-level) -> ordered enum (OrdinalTelemetry applies).
  Single-view (all 78 cases are the benchmark; no balanced/natural
  sampling — the set is hand-curated). The case definitions live in
  upstream's `bench/cases.py` as Python source that imports a
  package we do not depend on; we parse the source with `ast`
  the fetcher downloads the PINNED commit sha from raw.githubusercontent.com,
  verifies its sha256 against a hardcoded expectation (fail-closed on
  mismatch, the same fail-closed rule as the HF datasets), and parses it with the `ast`
  module — never `exec`. Each case carries `workflow = task_id` so the
  report's `per_workflow_accuracy` carries per-task accuracy, and the
  leaderboard renders a separate jabr table with 8 per-task columns.
- Runs: for every (track, scorer, dataset) — `eval` in-process `--runs` times
  (default 2), last run kept, order-rotation permutations on for the parallel
  track. Writes `predictions.jsonl`, `run.json`, `report.json`, `report.md`
  per combination, then `SUMMARY.md` across all of them. Completed combos
  are skipped on rerun (`--fresh` overrides); load/run failures become
  `load_failed`/`run_failed` rows in the summary instead of crashing.

## What NOT to commit

## RAM model (#79)

Activity Monitor's "used" during a `jevmlx bench` run is NOT `weights ×
number of cases`. It is three things, summed:

1. **Model weights** — the 4-bit 7B is ~4 GB, held by the engine cache
   (`load_engine` LRU, maxsize=1) for the whole run. The bench does NOT
   unload between cases (a reload is 4 GB × hundreds of questions).
2. **The live logits slab** for the CURRENT case — `rows × suffix_tokens ×
   vocab` float32 per chunk. The chunking heuristic limits *concurrent*
   rows (it ran 1 row/pass for the 7B smoke), but a long suffix still
   builds a large `width × vocab` tensor. This dies when the decide()
   call returns.
3. **The Metal buffer cache** — Metal keeps freed GPU buffers in a cache
   for reuse. After a huge case, a tiny case still shows high RAM because
   bucket 3 stays; it is NOT returned to macOS until `mx.metal.clear_cache()`.

The single-model `jevmlx bench --model <one>` path goes through `run_bench`,
which clears the Metal cache **after every combo** (it once only cleared
in its `finally`, after ALL combos), resets the peak memory counter at combo
start, and logs the three Metal counters (peak /
active / cache) per combo into the bench log and the combo's `run.json`
`memory` block. The model is still loaded once and kept; only the GPU
leftover is released. `run_bench_models` (multi-model) already cleared
between models; that is unchanged.

A **cache cap** (`--metal-cache-gb`, default 8): at `run_bench`
start `mx.metal.set_cache_limit` makes the allocator evict freed buffers
above the cap instead of hoarding them — so inside a long combo (typesafe,
426+ cases with rotations) the Metal allocator can no longer hoard ~94 GB
and push the machine into swap. The cap is recorded in each combo's
`run.json` `memory` block as `metal_cache_limit_bytes`.

A **heartbeat** (`--heartbeat-every`, default 25, 0 disables):
the per-case eval loop prints `[heartbeat] <combo> cases_done=N pred_lines=M
elapsed_s=S peak=X active=Y cache=Z` every N completed cases (reusing the
same GB formatting as the `[memory]` line) and appends the same fields as
one JSON record per heartbeat to `<combo>/heartbeat.jsonl`. The Metal
memory API moved to the top-level `mx` names (`mx.set_cache_limit` etc.)
now that mlx 0.32 deprecates the `mx.metal.*` aliases.

Anything outside `benchmarks/results/<machine>-<model-slug>/`: caches
(`~/.cache/jevmlx/`), model weights, logs, editor files. Predictions larger
than 5 MB total are gzipped automatically (the folder README says so).

Exception — `benchmarks/probes/<machine-tag>--<model-slug>/`: diagnostic
probe artifacts (e.g. `driftprobe.json`/`driftprobe.md` from
`benchmarks/driftprobe.py`). These are NOT results-contract files and are
not validated by the results-check CI job; they document one-off
diagnostic measurements (batched drift matrix, kernel-shape probes) that
accompany a PR but are not part of the results ledger.

`benchmarks/two_stage.py` measures whether coarse→fine two-stage
choice (pick a category group, then pick within it) is more accurate or
faster than a single trie-constrained pass on 255-option enums. Output:
`two_stage.{json,md}` in the same probes folder.

## PR checklist

- [ ] Folder contains only the bench output (predictions, run.json,
      report.json/.md, dataset lock files, SUMMARY.md, folder README)
- [ ] `parity.json` in the model folder — records the passing slow parity
      test (batch vs chunked log_score agreement, single-context)
      AND the batched matrix (batched-vs-independent
      finals plus the raw pre-rescore row-logit gate). A model cannot
      enter the README leaderboard without it; `check_results` rejects a
      payload without `max_raw_row_drift_nats` / `max_gap_drift_nats` as
      pre-v2. The gate checks DECISIONS, not raw numbers (issue #184):
      PASS = every field's batched winner equals the batch=1 winner AND no
      escaped near-tie; the drift quantities (raw-row, pairwise gap,
      top-two margin, logp — in nats and in fp16 ulps of the measured
      logit magnitude) are batch-kernel lattice noise, reported as
      diagnostics and never gated. Schema:
      `{"model": "...", "test": "test_w1a_...", "passed": false,
      "max_abs_drift_nats": 0.044, "max_raw_row_drift_nats": 0.125,
      "max_gap_drift_nats": 0.070, "max_margin_drift_nats": 0.063,
      "max_batched_drift_nats": 0.05, "winners_identical": true,
      "atol": 0.05, "rescore_gate": {...}, "environment": {...},
      "status": "PASS", "run_at": "..."}`

      ``status`` is the reporting word — exactly PASS or FAIL. The gate
      (``passed``) follows it: FAIL sets ``passed: false``. PASS = every
      field's batched winner equals the batch=1 winner AND no escaped
      near-tie (every field whose margin sat inside the engine's rescore
      band was actually rescored). FAIL = a winner differs, or an escaped
      near-tie. A scored payload still carrying status DRIFT is
      pre-contract and rejected with a regenerate hint; a payload without
      a scoring key is UNVERIFIED (pre-F9) in the leaderboard. The drift
      numbers ride the payload as diagnostics only.
      ``check_results --check-parity`` prints the status word and one
      sentence; the leaderboard shows the status word in the Parity column.

      **Publishability** (parity-gates): a model with status PASS is
      publishable — it appears in the leaderboard and ``check_results
      --check-parity`` returns OK. Status FAIL (a winner changed, or an
      escaped near-tie) is excluded from the leaderboard and
      ``check_results`` returns FAIL. A pre-F9 payload without a scoring
      key shows UNVERIFIED (pre-F9) in the leaderboard's Parity column.
- [ ] `SUMMARY.md` pasted into the PR description
- [ ] Machine specs (chip, RAM, macOS) mentioned in the PR body
- [ ] No hand-edited numbers — recompute instead of fixing up
- [ ] `results-check` CI workflow is green (runs `python -m
      benchmarks.check_results` on the changed folders: contract keys, folder
      size, dataset lock, and report reproducibility — all must pass before
      merge)
- **Error lines**: a prediction line whose `error` key is a non-empty string
  is an error record (the field's scoring failed — e.g. a context too long
  for the model's window). Error lines may have null `correct`,
  `probability`, `prediction`, `per_item_end_to_end_ms`, and `log_scores`; they
  are allowed by the contract and counted in an errors summary (per combo:
  count + first error text) printed by `check_results` and shown in the
  leaderboard 'Cases' column as `N (M error)`. A line with null values and
  NO `error` key is still a contract violation.
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

### A/B install contract (issue #63)

Two guards make sure the A/B venv gets the A/B branch's jevmlx, never the
main repo's (an install with cwd = the main repo measures main twice):

- The install runs `-e <ab-worktree>[dev,bench]` with cwd=<ab-worktree> —
  the worktree path is absolute, never relative to the caller's cwd. After
  the install, `<ab-venv>/python -I -c` imports jevmlx, prints the resolved
  path into `ab-setup.log`, and FAILS the step unless it resolves inside
  the A/B worktree (`-I` keeps the import off the cwd, so it cannot mask a
  broken install).
- After the A/B bench, the combos' run.json `environment.git_sha` are
  compared against the main side. An equal sha fails the ab-bench step with
  'A/B measured the base code', skips the remaining A/B steps, and SUMMARY
  discards the A/B side instead of comparing main against itself.

Extra eval flags for the A/B side ride `--ab-eval-args` (repeatable):

```bash
python -m benchmarks.m5 --out m5-2026-09-23 --ab-branch w6-dualframe \
    --ab-eval-args=--dual-framing
```

The flags are appended verbatim to the A/B bench and invariance command
lines. A flag the A/B branch's bench/invariance do not expose fails that
step loudly (argparse error in the step log) — it is never silently
dropped.

**macOS idle sleep**: on darwin the runbook re-execs under
`caffeinate -dimsu` so macOS does not idle-sleep mid-run (during the M5 7B
smoke it did, and Metal parks while wall time runs). The `sleep_blocked`
flag is recorded in `RUNBOOK.md`'s header. Opt out with `--allow-sleep`.

### MODEL_ID env: the slow suite's model selector

`tests/conftest.py` reads `MODEL_ID` from the environment in ONE place
(`os.environ.get("MODEL_ID", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")`)
and exposes it as the `MODEL_ID` constant. Every slow test that loads a real
model reads that constant (or the `engine` fixture, which does); no test
hardcodes a model id. So when the M5 parity step runs `pytest -m slow` with
`MODEL_ID=<parity model>`, every slow test exercises THAT model — not the
0.5B dev default.

Tolerances are decision-based (issue #184): the parity gate checks
WINNERS (winners identical + no escaped near-tie), never raw drifts — the
recorded drift numbers are fp16 lattice noise of the batched kernels
(4-8 ulp steps), reported as diagnostics in nats + ulps and never gated.
`test_w4b_parity.py::test_parity_real_model_twin` (which runs all four
bundled presets and asserts the DECISION invariant) is **default-model-
only** — it skips when `MODEL_ID` is not the 0.5B default (the drift
magnitudes were characterized on the 0.5B; larger models see higher raw
drift from Metal batch-shape matmul variation, and the M5 parity step
injects those models via MODEL_ID env). The batch-vs-chunked parity tests
run on the two `fintech_fraud` / `support_triage` presets (smaller drift)
and stay on all models; they also assert the decision invariant.

## Statistics

Every accuracy number in `report.json` and the markdown summary carries a
confidence interval. No interval = the row says "n too small" rather than
an unqualified number.

### Wilson interval (single Bernoulli per independent case)

`wilson_interval(k, n, z=1.96)` — used ONLY for per-field accuracy where
each case is an independent Bernoulli trial. No continuity correction.
Not sufficient for correlated fields, balanced sampling, or macro-averaged
 tasks. Canonical lines only — permutation/rotation rows are excluded so n
is not inflated ~3x.

### Case-cluster bootstrap (the default CI)

`cluster_bootstrap_ci(records, metric_fn, draws=2000, seed=0)` — resamples
whole cases (all their fields move together) B=2000 times with a fixed
seed. This is the DEFAULT CI for aggregate accuracy, macro-F1, NLL, Brier,
and ECE everywhere fields are correlated within a case or sampling is
balanced. The seed is fixed for reproducibility — same seed → identical CI.

### Paired comparison (McNemar + paired bootstrap)

`paired_bootstrap_difference(records_a, records_b, draws=2000, seed=0)` —
for two conditions on the SAME cases (perturbation tracks, A/B branches).
Returns the accuracy difference CI plus the exact McNemar test
(`mcnemar_exact(b, c)`) on the correct→wrong vs wrong→correct discordance
counts. The exact two-sided p-value is the binomial tail under H₀: Binom(b+c, 0.5);
no continuity correction. The McNemar discordance is per case; the paired
bootstrap difference is per line (the two are aligned by case_id but the
units differ — documented here, not reconciled).

### Valid-only accuracy

`valid_accuracy` — excludes invalid predictions from the denominator
(denominator = labelled AND valid lines), alongside `field_accuracy`
(failure-inclusive: invalid counts as wrong, denominator = all labelled).
The gap between the two is the invalid-prediction rate — if they're equal,
every prediction was valid.

### Report contract

Results contract v2 (additive keys): `accuracy_ci`, `log_loss_ci`,
`brier_ci`, `ece_ci`, `macro_f1_ci` are additive — they ride alongside the
point estimates. `per_field_accuracy` carries a `ci` (Wilson) per field.
`valid_accuracy` is the valid-only companion to `field_accuracy`.
`check_results` enforces `REQUIRED_CI_KEYS` (accuracy_ci, valid_accuracy,
per_field_accuracy). The leaderboard prints the interval next to every
point or 'n too small'. `report.md` and `SUMMARY.md` show the majority
baseline (mean over fields) and exact-record accuracy as first-class
columns right after field accuracy; the per-field table marks a field
below its majority baseline with †.


## Slow tests on the shared macOS runner

Four GPU-bound slow tests assert one-pass batching over the 28-row fintech_fraud
schema, or batched-vs-chunked parity on a real model. The GitHub Actions
`macos-14` runner's GPU times out on the full-batch Metal command buffer
(`kIOGPUCommandBufferCallbackErrorTimeout`), forcing a 2-chunk retry that
breaks the `sequential_forward_passes == 1` assertion and crashes the parity
twin. Under `JEVMLX_CI_SLOW=1` (set in `.github/workflows/slow.yml`) these 4
tests are skipped via `skip_on_shared_runner()`:

- `test_chunking_matches_full_batch_and_counts_passes`
- `test_slots_scoring_fintech_fraud`
- `test_labels_scoring_fintech_fraud_valid`
- `test_parity_real_model_twin`

They are verified on real hardware (M5) via `slowtest.sh`. The GitHub Actions
slow workflow is manual-dispatch only — it no longer runs automatically on
merge to main, because the shared runner GPU is not stable enough for model
tests (Metal command-buffer timeouts, NaN logits observed).
