# Changelog

## 0.1.0 - 2026-09-17

First release.

- Parallel constrained decisions: every schema field decided in one batched
  forward pass on Apple Silicon (MLX), with per-choice probabilities from a
  token trie over the choices — probabilities sum to 1 with no extra softmax.
- Slot scoring by default: neutral alias codes chosen per field and tokenizer
  by a deterministic codebook search (greedy over letters, digits, 2-char
  pools; the set whose complete candidate rows tokenize most cleanly wins).
  `--scoring labels` scores real choice text when spelling is the signal.
- Multi-select fields as independent per-option yes/no decisions plus an
  always-on count row (`<field>#count`, answer codes '0'..'4' where 4 means
  four or more) that reconciles the
  selected set to the top-k options when its own confidence clears a margin
  gate; otherwise the per-option rule stands.
- Hard set constraints for multi fields: declare mutually-exclusive / at-most
  / at-least / exact-size / implies groups in the schema; an exact solver
  picks the score-maximizing feasible set and contradictions are rejected at
  compile time.
- Case-level constraints (implies, excludes, exclusivity) reconciled by
  constrained MAP: the joint assignment maximizes summed log scores over
  every field, sharing one constraint evaluator with the metrics code.
- Parent-conditioned second pass: declare `depends_on` between fields and
  low-confidence children (or children whose parent's MAP reconciliation
  changed the value) are re-decided in one extra batched pass, conditioned
  on the parent's answer; oracle overrides supported for DAG evaluation.
- Near-tie rescoring: candidates whose logit gap falls inside a single
  instability band are re-scored at the canonical batch=1 shape so results
  do not depend on batch composition; the same constant bounds the
  batch-vs-chunked parity tolerance.
- Scoring parity as a gate: one shared check compares batch=1, batched, and
  chunked scoring over the bundled presets; the benchmark records its verdict
  per model as `parity.json` and a model without a passing verdict cannot
  enter the compatibility table.
- Measured memory budget: rows per chunk come from a working-set estimate,
  wide rows are bucketed together, and a Metal allocation failure halves the
  chunk once and still scores every row; peak memory is reported.
- Legal-mass telemetry: per-branch leakage signal (probability the model
  assigned to any allowed continuation, against the full vocabulary) on
  every field, raw and pre-correction — calibration feature for abstention.
- Typed Python API: `jevmlx.decide(PydanticModel, context)` returns a
  validated instance plus per-field provenance (probability, score, unit-
  split margins, top alternatives, calibration state, scoring mode, and a
  single `reason` field covering NONE_OF_ABOVE and abstention);
  `decide_many` for batches. `Optional` fields map None to an explicit
  "none of the options apply" choice, and `abstain_below_margin` withholds
  low-confidence values while keeping the raw decision.
- Multi calibration: `jevmlx calibrate --out` fits a pooled logistic on raw
  multi option log-odds alongside the scalar temperature and writes both to
  one JSON file; decide runs select multi options by calibrated log-odds.
- Prompt v7: neutral alias menu with per-choice glosses, coded multi option
  rows, the count question, and an injection-resistant context block; the
  prompt version is read from the engine, never restated by callers.
- Prompt profiles: per-model-family chat-template handling resolved once at
  load (thinking mode off for Qwen3, system-role probing instead of per-
  request retries); no hand-built prompts, no system-role assumptions.
- `jevmlx decide` CLI: run a bundled preset or your own schema/context; table
  output or `--json`.
- New backend: the same decision semantics through any OpenAI-compatible
  chat endpoint that returns logprobs (`jevmlx decide --backend openai`),
  one request per field; missing candidates in the endpoint's top-k get an
  explicit floor probability and are flagged in the output.
- `jevmlx doctor`: environment checks (platform, versions, memory, power,
  Metal, model cache, network) before filing an issue or running a
  benchmark.
- `jevmlx eval` / `jevmlx report`: labeled-case evaluation with per-field
  predictions and run manifests, offline report summaries, TypeSafe-consensus
  agreement metrics, and constraint-violation rate.
- `jevmlx bench`: one command producing a complete, PR-ready results folder
  (parity verdict, compatibility table inputs, eval reports, environment
  metadata), single model or a comma-separated list run sequentially; a
  model that fails to load or fails parity is recorded and skipped, not
  fatal.
- `jevmlx serve`: a local HTTP server exposing `POST /decide` so one Metal
  GPU can back several clients, serially.
- `jevmlx validate` / `jevmlx lint`: schema validation for structural
  problems and choice-collision findings before a run.
- Benchmarks: bundled and TypeSafe-fetched eval cases with dataset locks and
  provenance, deterministic label-preserving perturbations, an
  irrelevant-field invariance benchmark, a naive-autoregressive baseline
  track, and results auditing that recomputes committed metrics from raw
  predictions.
- Apple Silicon only at model-load time: importing the package, compiling
  schemas, and computing metrics work on any OS; the platform check fires
  with a clear error only when a model is actually loaded.
- Provenance fields everywhere: prompt SHA-256, prompt version, compiled-plan
  and chat-template hashes, dataset locks, run manifests, and parity verdicts
  so any number in a report traces back to the exact prompt, data, and code
  that produced it.
