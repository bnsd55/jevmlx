# Changelog

## Unreleased

- W2-A field-local prompts (HOLD for M5 A/B): the global schema block is
  replaced by per-field prompt blocks. The engine renders one complete chat
  prompt per field (system + nonce-delimited context + that field's block +
  lead-in), the plan compiler takes the exact token-ID LCP across all
  per-field prompts as the prefill, and each row carries its field's
  post-LCP prompt tail + candidate remainder. Context moves ABOVE the field
  block (GPT Q2: final contract nearest generation). Every displayed string
  is json.dumps-escaped. Multi fields render one block per option. New
  telemetry: prefill_tokens, suffix_tokens_total. PROMPT_VERSION v8. No
  fallback path — render_field_prompt is mandatory on compile_*_plan; the
  old lead_in_ids / global-schema prompt path is deleted. Plan cache key
  includes a context hash (prompt tails are context-dependent). PARITY_ATOL
  bumped to 5e-2 (W2-A's longer rows increase Metal batch-shape drift to
  ~0.027 nats). This PR is marked HOLD — it merges only after Ben runs EV2
  + TypeSafe on the M5 against pre-W2-A main and the gates pass (EV2 drift
  decreases, TypeSafe accuracy non-worse).

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

- Parallel constrained decisions: every schema field decided in one batched forward pass on Apple Silicon (MLX), with per-choice probabilities from a token trie over the choices — probabilities sum to 1 with no extra softmax.
- `jevmlx decide` CLI: run a bundled preset or your own schema/context; table output or `--json`; slot scoring (neutral aliases) by default, `--scoring labels` for real choice text; opt-in prior correction.
- Multi-select fields decided as per-option yes/no rows; no field-level probability claimed, a margin instead (0.5-threshold distance uncalibrated, calibrated log-odds gap when a calibrator runs).
- Typed Python API: `jevmlx.decide(PydanticModel, context)` returns a validated instance plus per-field provenance (probability, score, margin, top alternatives, calibration state, scoring mode); `decide_many` for batches.
- Temperature calibration on labeled JSONL data (`jevmlx calibrate`).
- New backend: the same decision semantics through any OpenAI-compatible chat endpoint that returns logprobs (`jevmlx decide --backend openai --base-url URL --api-model M`), one request per field; missing candidates in the endpoint's top-k get an explicit floor probability and are flagged in the output.
- `jevmlx doctor`: environment checks (platform, versions, memory, power, Metal, model cache, network) before filing an issue or running a benchmark.
- `jevmlx eval` / `jevmlx report`: labeled-case evaluation with per-field predictions and run manifests, offline report summaries, TypeSafe-consensus agreement metrics.
- `jevmlx bench`: one command producing a complete, PR-ready results folder (compatibility table inputs, eval reports, environment metadata), single model or a comma-separated list run sequentially.
- Chat-template handling that works across model families (no hand-built prompts, no system-role assumptions).
- Schema validation (`jevmlx validate`) for structural problems before a run.
- `jevmlx serve`: a local HTTP server exposing `POST /decide` (`{"schema": {...}, "context": "..."}`) so one Metal GPU can back several clients, serially.
- Typesafe fetcher (`benchmarks.typesafe.fetch`) for the published eval examples, plus synthetic labeled cases covering known failure modes.
# v8
