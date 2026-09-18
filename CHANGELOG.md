# Changelog

## Unreleased

- W4-A named model defaults + compat metrics: `--model` accepts aliases
  `fast` (Qwen2.5-3B), `quality` (Qwen2.5-7B, **default**), `test` (1.5B,
  tests only). `resolve_model()` resolves aliases in `load_engine`, `doctor`,
  and all CLI commands. Hub ids verified by a slow HEAD test (never at import).
  `run.json` gains `config.tokenizer_metrics` (quoted-code token lengths,
  single_branch_fraction, max_trie_depth, mean_legal_mass from telemetry).
  `check_results.py --check-parity` gates the README compatibility table: a
  model enters only with a passing slow parity test (`parity.json` in its
  results folder). `_local_rows` in leaderboard.py enforces the same gate.
  Also fixed a pre-existing test isolation bug (`sys.setrecursionlimit(60)`
  leaked from `test_trie.py`).
- Multi count row (W2-E step 3): every multi field gets an extra decision row
  `<field>#count` (candidates `0`, `1`, `2`, `3`, `4+`, scored like a scalar
  enum through the trie; the row always runs). When the count row's top-2
  margin clears `COUNT_MARGIN_MIN` (0.7 nats) the selected set is reconciled
  to the top-k options by calibrated log-odds (P(yes) order uncalibrated —
  monotone-equivalent), k capped at the option count; otherwise the
  per-option rule stands. Telemetry: `count_choice`, `count_margin`,
  `reconciled_by` on the field entry plus a `<field>#count` scalar entry.


- W2-D legal_mass telemetry: per-branch leakage signal added to engine
  field telemetry. legal_mass = sum(exp(z_allowed)) / sum(exp(z_vocab)) —
  the probability the model assigned to the union of allowed continuations
  at the branch position, against the full vocabulary. A branch can
  confidently pick A over B even when almost all unconstrained mass is on a
  reasoning token, newline, or label text; legal_mass flags that leakage.
  Product over the winner's branch path (scalar fields) / per-option Y/N
  branches (multi fields); 1.0 for cardinality-1 fields. Raw, pre-prior-
  correction logits. Always computed: the ~0.002 Metal FP drift from the
  full-vocab mx.logsumexp means bit-identical batch=1 vs batch=N parity was
  never a real invariant on Metal (GPT Q4 confirms); the W1-A parity test
  now asserts winners identical + log_scores within tests/conftest.py's
  PARITY_ATOL = 1e-2 (measured Metal batch-shape drift ~0.004 nats, winners
  stable; coder1's PR #21 imports the same constant post-merge; exact
  equality still holds on the deterministic FakeModel path). trie.py:
  score_trie returns (log_probs, legal_mass_logs) — one function, the
  legal_mass_at_node callback optional (None = mass 1.0, for the MLX-free
  unit tests); score_trie_with_legal_mass removed. Calibration feature for
  the abstention model (R9, bug 10). forced_prefix_logprob deferred (needs
  prefix-position logits the engine currently discards; conflicts with
  W3-A's gather-only constraint — noted in ROADMAP).

- Apple Silicon platform check moved from engine.py import time into
  `load_engine` (via `_require_mlx`): `import jevmlx`, `import jevmlx.engine`,
  and schema/plan/metrics tooling now work on Linux (ubuntu-latest CI);
  the clear RuntimeError fires only when a model is actually loaded.
  `engine_metadata` returns `mlx_version`/`mlx_lm_version` as None when mlx
  isn't installed (best-effort provenance, no raise).
- Doc sweep: ARCHITECTURE updated to the current contracts — prompt v5,
  corrected line citations (engine.py:293/430/471-490/508-516/621-629),
  PromptProfile in the engine module-map row, a new FieldResult contract
  table (margins unit-split, reason as the single source of truth), Y/N
  codes in the assembly line. Stale `prompt v2` / `jevmlx-parallel-v2`
  mentions fixed; no UNKNOWN/allow_unknown/prompt-v2 references remain.
- UNKNOWN split into two concepts (was one conflated kwarg):
  `allow_none_of_above=True` adds an explicit `NONE_OF_ABOVE` choice
  ("none of the options apply") mapped to `None` with
  `FieldResult.reason="none_of_above"`; `abstain_below_margin=X` is a
  separate confidence gate — fields whose margin falls below the cut are
  withheld from the model (`FieldResult.reason="abstain"`; the raw value
  stays for provenance). `allow_unknown` is removed with no alias. The
  calibrated correctness model is W2.
- Multi calibration (W2-E step 2): `jevmlx calibrate --out` fits a pooled
  logistic (a, b) on raw multi option log-odds alongside the scalar
  temperature and writes both to one JSON file; `decide`/`decide_many`/
  `run_parallel_generation`/`decide_openai` accept `calibration=<path or
  dict>` and select multi options by calibrated log-odds > 0. The
  `multi_threshold` parameter and `--multi-threshold` flag are DELETED (no
  dual path); uncalibrated selection stays at the fixed P(yes) >= 0.5 rule.
- Prompt profiles: a frozen `PromptProfile` (template kwargs + system-role
  support) is resolved once at engine load. Qwen3-family models get
  `enable_thinking=False` so answers land in the direct channel; the
  system-role probe replaces the per-request TemplateError retry.
- PEP 604 optionals accepted everywhere: `Literal[...] | None` and
  `EnumClass | None` now work like `Optional[...]` (pydantic keeps the enum
  form as a raw `types.UnionType`).
- Naive baseline honesty: `run_naive_generation` is greedy by definition —
  the never-applied `temperature` argument is removed — and its JSON-schema
  prompt now shows every choice instead of truncating enums over 50 options
  to 20.
- Small fixes: duplicate engine-load log line removed; stop-token discovery
  no longer treats the unknown token as a stop (Mistral-style tokenizers
  map absent strings to unk).

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
  fatal. Every combo folder carries a `timing.json` (median timing split
  over the same decide calls the predictions came from).
- Standalone timing report (`benchmarks/timing.py`): decide() per bundled
  preset for N repetitions with the engine's full timing split — prior,
  prefill, plan compile, cache broadcast, suffix eval, lm-head gather,
  second pass, total — plus rows, padded token positions, forward passes,
  rescored fields, and the second-pass rerun rate; median/p95/min/max over
  repetitions with per-rep raw numbers alongside.
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
