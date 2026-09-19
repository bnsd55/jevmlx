# Changelog

## Unreleased

- **W5-C** — set constraints, joint count + set optimization, typed
  calibration, internal telemetry, abstention contract (PROMPT_VERSION
  `jevmlx-parallel-v9`). The set-constraint solver is exact: connected
  components from group-overlap AND implication edges, one bitmask
  enumeration per component with implication closure applied per
  candidate, score-maximizing under (score, proposal overlap, schema
  order). A single component above 20 options raises at COMPILE time
  (`SchemaCompileError`); the cap is per component, not per field. The
  compiler has NO syntactic contradiction rules — satisfiability is
  decided by running the same solver unscored (`setcons.is_feasible`),
  so `A→B + B→A`, `A→B + at_most_one(A,B)`, `A→B + exact_k([B],0)`, and
  implies cycles are accepted (both absent / A forbidden / A not
  selected); genuinely unsatisfiable sets still raise. A trusted count
  enters the SAME solver run as the schema's set constraints (bucket
  0–3 `exact_k`, bucket 4 `at_least_k(k=4)` — `"4"` means four or
  more); schema constraints are HARD and a jointly-infeasible count drops
  as unreliable evidence (logged). `CalibrationBundle` is the one typed
  object (model_revision, prompt_version, scoring, prior_mode,
  scalar.temperature, multi.a/b); `jevmlx calibrate --out` writes it,
  `CalibrationBundle.load` reads it at the public boundary
  (`str | CalibrationBundle | None`), the engine takes `CalibrationBundle
  | None` only — no file I/O, no dict, no duck typing. `prior_mode`
  records what the multi (a,b) was fitted on; the engine rejects
  fit/apply mismatches. Count rows live in `internal_telemetry` keyed
  `<field>#count` (never `field_telemetry`, so `Decision.fields` is
  clean). `FieldResult.calibrated` reflects the applied calibrator per
  field (telemetry carries `calibration_id`). `abstain_below_margin`
  requires every model field to be Optional — `TypeError` at `decide()`
  start naming the first offender, not a Pydantic failure after
  inference. `decision_margin` is the min over ALL options of the
  per-option distance from its threshold side, floored at 0, recomputed
  after every reconciler.

## Released

- The engine is an object: `load_engine` returns a frozen `Engine`
  dataclass carrying `model`, `tokenizer`, `model_id` (resolved),
  `revision`, `profile`, `vocab_size`, `weight_bytes`,
  `cache_capabilities`, and the measured `width_slope` — every per-model
  property resolved ONCE at load (the prompt profile is probed there from
  the resolved id; the hot paths read `engine.profile`). The generation
  entry points take an Engine only — `run_parallel_generation(engine,
  context, schema, ...)`, `run_parallel_generation_batched(engine,
  contexts, ...)`, `run_naive_generation(engine, ...)` — and all callers
  (`api`, `calibrate`, `cli`, `evalrun`, `parity`, `serve`, `bench`, the
  benchmark scripts) pass the object through; `_prefill` /
  `_get_or_compute_prior` take the load-time `profile` (the prior helper
  takes the engine itself). `width_slope` lives on the Engine (the
  process-global and its accessor are gone); the chunking budget
  (`_width_bin_max_rows`) takes the slope as a parameter. Tests build
  engines through the `conftest.make_engine` factory. `load_engine(alias)`
  and `load_engine(full id)` return the ONE cached object.
- W5b-14 engine timing ledger adoption (in #58, on main): one
  `jevmlx.timing.Ledger` per request measures every interval ONCE; the
  result dict's `*_ms` keys are pure `Ledger.derived_flat()` derivations.
  This branch's engine-object work is rebased on top of it.
- W5b-9 CLI error contract: user-input failures (missing file, bad JSON,
  schema/constraint rejection, engine environment errors) exit 1 with the
  full error message on stderr — no traceback (`-v` re-raises for debug);
  Ctrl-C exits 130. `decide --json` no longer prints the `Loading ...`
  line to stdout (stdout is pure JSON). `doctor --model quality` resolves
  aliases before the tokenizer check. `serve` logs the bound port after
  binding.
- W5b-1 immutable schema and plans: `StructuredSchema` and
  `FieldDefinition` are frozen dataclasses — mutation raises
  `FrozenInstanceError`; derive with `dataclasses.replace`
  (`compile_set_constraints` returns a NEW frozen field carrying a tuple of
  read-only constraint views). Compiled plans are exposed as read-only
  mappings (`_freeze_plan` over `MappingProxyType`). `jevmlx.api` no longer
  re-exports `jevmlx.models.*` — `MODEL_ALIASES`/`DEFAULT_MODEL`/`resolve_model`
  import from `jevmlx.models` directly.
- W5b-2 shared test fakes: `tests/conftest.py` owns one tokenizer/model per
  family (`FakeTokenizer`, `FakeModel`, `_Mod97Tokenizer`, `YNLogitModel`,
  `CountCodeModel`) plus the engine-result factories `make_engine_result` /
  `make_field_telemetry` (the full `run_parallel_generation` result shape,
  per ARCHITECTURE.md). Contract tests pin the factory's key sets to the
  engine's real output (top level + scalar/multi/count telemetry shapes) and
  assert the per-file aliases stay the same objects — the per-file stub
  copies are gone; a stub that only passed because it was wrong is now
  visible.
- W5-D decide_many semantics: the prior pass runs ONCE per
  `run_parallel_generation_batched` call (every result reports the shared
  `prior_ms`); context groups are built INCREMENTALLY from actual cumulative
  cache bytes plus projected suffix cost over prompt-length-sorted contexts
  (a short first context no longer sizes a group that admits 30K-token
  prompts); honest per-result timing — `group_wall_ms`,
  `per_item_amortized_ms`, `per_item_end_to_end_ms`, and `contexts_per_pass`
  as the ACTUAL group size (the final partial group reports its own size).
- W5-D per-chunk retry + measured memory budget: `_score_rows` halves and
  retries a chunk on a Metal allocation failure (`_is_metal_allocation_error`
  — resource exhaustion only, programming errors propagate) and records the
  retries as `failed_attempts` on `ScoreRowsResult` and the result dict —
  never folded into `sequential_forward_passes`; retry state is per chunk, so
  a retried chunk does not stop later chunks in the same bucket. The chunk
  cap moves to the width-bin budget `_width_bin_max_rows` (each row's cache
  bytes + its logits slab at the row's OWN width bin, scaled by the tiling
  slope); the slope is MEASURED at engine load (`_measure_width_slope`, the
  B=1/B=2 peak-activation ratio inside the warmup, floor 1.0 on any failure)
  instead of assumed. Peak memory is honest: absolute
  `mx.reset_peak_memory()`-based `peak_active_bytes` plus
  `peak_incremental_bytes` (peak minus the request's starting active
  memory).
- W5-D prior cache keys: the process-lifetime prior cache
  (`_prior_cache_key`) is keyed by id(model)/id(tokenizer) with LIVE weakref
  verification on every hit (an id can be reused after free), the neutral
  prompt's full sha256 (the plan hash alone omits descriptions/glosses/field
  order), prompt_version, and scoring mode; eviction is LRU
  (`_PRIOR_CACHE_MAX`). Non-weak-referenceable tokenizers are not cached.
- W5-D resolved-id engine cache: `load_engine`'s lru_cache sits on the
  RESOLVED model id (`jevmlx.models.resolve_model`), so `load_engine("quality")`
  and `load_engine("mlx-community/Qwen2.5-7B-Instruct-4bit")` share one entry
  instead of loading the model twice.
- W5-D log-space legal mass: `score_trie`'s `legal_mass_at_node` callback
  returns natural-LOG masses (the trie stays MLX-free; log space end to
  end). Multi fields replace the underflowing, cardinality-confounded
  field-level product with cardinality-free stats: `min_option_legal_mass`
  (worst option's leakage, probability space) and `mean_log_legal_mass`
  (additive, stable); the per-option logs stay on `legal_mass_logs`. The
  count row exposes its own `legal_mass` (winner) and `min_option_legal_mass`
  (worst code) on the `<field>#count` entry.
- W5-D parity matrix: `parity.check_batched_parity` exercises decide_many —
  each case at 1/2/4 contexts, equal and mixed prompt lengths, comparing RAW
  pre-rescore row logits (so a batch-1 rescore cannot mask raw batch drift),
  final decisions, and prior on/off. Parity cases use the bundled presets'
  REAL contexts (`_case_context`), not synthetic filler.
- W5-A prompt/plan correctness (PR #38): the prompt renders FROM THE COMPILED
  PLAN — `to_schema_str(mode, tokenizer=…)` shows the aliases the scorer
  actually scores (`alias_for_index` survives only for the OpenAI adapter's
  per-field requests); the codebook search is BOUNDED BACKTRACKING over a
  prefix-conflict graph (`_search_codebook`), not greedy — an A-is-a-prefix-
  of-B/C/D pool now finds the valid {B, C, D} set instead of raising. ONE
  canonical JSON serializer (`jevmlx/json_text.py`, `ensure_ascii=False`) for
  every prompt and candidate path — no more escaped-non-ASCII mismatch
  between what is compiled and what is displayed. The context fence carries a
  sha256-derived nonce (`_context_nonce`/`_context_block`), so a context
  containing a fake `CONTEXT>>>` line can no longer close the block early.
  `PROMPT_VERSION` is `jevmlx-parallel-v8`.
- Doctor environment hardening (PR #39): `check_venv` fails a conda
  interpreter (`sys.base_prefix` under miniconda/anaconda — conda Pythons
  have been observed to hang at exec under endpoint-security load) and probes
  a trivial subprocess within 2 s; `check_editable_install` fails an editable
  install that points at a DIFFERENT checkout (`direct_url.json`), while
  plain package installs stay OK.

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
  stable; PR #21 imports the same constant post-merge; exact
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
- W5-B engine restructure (PR #45): every scalar scoring path — normal
  batched pass, batch=1 rescore, dependency rescore, oracle rescore — now
  produces a frozen `ScalarEvidence` (per-choice T=1 log-scores +
  legal-mass logs, `source_shape` batch/batch1/dependency/oracle), and
  every decision goes through the ONE finalizer
  `finalize_scalar_evidence` (INSTABILITY_BAND rescore gate, prior
  correction, confidence temperature, tie flag, margins, top_choices).
  `ScalarDecision` carries the complete public telemetry (including
  `evidence_source`, new on scalar `field_telemetry` entries);
  `Candidate` separates the typed value from its score-key so booleans no
  longer decide as "true"/"false" strings in reconciled assignments.
  Case-level constraints are validated against the schema BEFORE any
  model work (`constraints.validate_constraints_for_schema` — unknown
  types/fields, out-of-domain values, and implies/requires_parent/excludes
  on multi fields fail at call time; exclusivity remains the supported
  multi shape). The dependency second pass runs in topological waves with
  an explicit `Given: {"parent": …}` conditioning header per row family
  and conditions depth-2 children on the UPDATED depth-1 decisions; after
  the final MAP re-run a violating assignment raises
  `InternalConstraintViolationError` instead of returning a decision.
- W5b-8 timing ledger (PR #46): `jevmlx/timing.py` — a standalone event
  ledger (`Ledger`, `Interval`, `SpanError`) for engine timing. Pure
  Python, no mlx import, NOT yet wired into the engine: measured
  non-overlapping spans in `prior`/`main` phases, `Ledger.derived_flat`
  derives today's exact result keys (`suffix_eval_ms` as a marked
  composite so old readers keep working), `Ledger.batched_views` produces
  `group_wall` / `per_item_amortized` / `per_item_end_to_end`. Engine
  adoption (deleting the ad-hoc `t_*_ms` accumulators) lands after W5-B.
- W6-1 LM-head adapters (PR #42): `jevmlx/adapters.py` — a registry of
  backbone/lm_head splits per installed mlx_lm model family
  (`LMHeadAdapter` protocol, `adapter_for`, `UnsupportedModelError`
  carrying the supported set, `list_supported_model_types`). Mirrors the
  exact attribute paths and call orders of qwen2/qwen3/qwen3_moe/llama/
  mixtral/phi3 (tied vs untied head), gemma3_text (always untied head),
  phi (biased head), and mistral3 (delegation). Registry-only, NOT wired
  into the engine yet — engine call sites adopt it in W6-2+.
- M5 runbook (PR #43): `benchmarks/m5.py` — the one-command milestone-gate
  sequence (doctor gate, slow parity per model, quality bench, invariance,
  timing, remaining parity models, optional `--ab-branch` A/B against a
  temp worktree) logged into `<out>/RUNBOOK.md` with a summary builder
  comparing main vs A/B from produced JSON only. Idempotent steps
  (`--fresh` reruns); the step planner and summary builder are pure
  functions tested with fakes — no model loads in a unit test.
- W6-2 prep probes (PR #47): `benchmarks/probe.py` — standalone `slope`
  (per-row peak-memory slope per width bin under `mx.reset_peak_memory`,
  B=1/2/4/8 at widths 4/8/16/32) and `adapters` (max abs diff + timing of
  full-width head vs decision-position head). No engine wiring.
- Per-field probability semantics (W5b-13): every engine scoring stage —
  scalar finalization, multi selection, count rows, case-constraint MAP,
  dependency waves — sets a frozen semantics record
  (`api.FieldSemantics`: score_source / temperature / calibrator_id /
  prior_mode / constraint_changed / dependency_rescored) on its
  field_telemetry entry, and `FieldResult.semantics` (required, kw-only)
  coerces it at the public API boundary — a missing record raises, so
  'required' holds at every decide() return. The result-level
  `probability_status` is now a SUMMARY over the distinct
  (score_source, temperature, calibrator_id, prior_mode) groups — one
  clause per group with its field count; the old global-only statement is
  gone (a result mixing temperature-scaled scalars, calibrated multi
  options and dependency re-scores cannot be described by one sentence).
  A calibrated multi records temperature None (the log-odds cut ignores
  the caller T); count rows always do (fixed T=1 buckets);
  constraint_changed reflects an actual selection change, not merely a
  binding constraint.
- Results contract v2 in the results tooling (PR #48):
  `check_results.py` requires `timing.json` on parallel-track combos with
  the full timing-split median (incl. `peak_incremental_bytes` +
  `failed_attempts`), requires the batched per-item keys
  (`group_wall_ms`/`per_item_amortized_ms`/`per_item_end_to_end_ms`) to
  land together when present, treats a pre-v2 `parity.json` (no
  `max_raw_row_drift_nats`) as a FAIL with a regenerate hint, and names
  failing parity stages via `_parity_failed_stages` (winners / final
  log-score drift / raw pre-rescore row drift, shared with
  `summarize_results`, whose `parity_failed` reason now carries stage +
  max drifts + atol). `leaderboard.py` reads ONLY the per-item
  end-to-end median for local time-per-case — a parallel combo without it
  raises `ValueError` naming the folder (no pre-v2 results exist on
  main; no fallback). `oracle_prediction` joins the optional prediction
  keys.
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
