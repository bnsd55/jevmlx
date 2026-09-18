# ARCHITECTURE.md

For people who want to read or change the code. References cite function,
constant, and key names (they survive refactors; line numbers don't) —
trust the code over this document when they drift.

## Module map

| Module | Role |
|---|---|
| [`jevmlx/schema.py`](jevmlx/schema.py) | Immutable schema model (`StructuredSchema`, `FieldDefinition` — frozen dataclasses, derived via `dataclasses.replace`; compiled plans exposed as read-only mappings), slot/labels/multi plan compilation, tokenizer-specific bounded codebook search (`_search_codebook`: prefix-conflict graph + backtracking over an independent set), plan-driven prompt rendering (`to_schema_str` takes the tokenizer — the compiled plan owns the displayed aliases), count-row compilation, per-tokenizer plan cache, compile-time rejections, hard set-constraint validation (`FieldDefinition.compile_set_constraints` returns a NEW frozen field). |
| [`jevmlx/trie.py`](jevmlx/trie.py) | Branch-point trie over candidate token remainders; `score_trie` (constrained-path log probs + legal-mass logs, callback in LOG space); softmax/logsumexp helpers. |
| [`jevmlx/json_text.py`](jevmlx/json_text.py) | THE canonical JSON serializer (`json_text`, `ensure_ascii=False`) for every prompt and candidate path — never `json.dumps` directly (non-ASCII labels must tokenize as displayed). |
| [`jevmlx/engine.py`](jevmlx/engine.py) | Model load (`load_engine`, lru_cached per RESOLVED model id, the only platform check), `PROMPT_VERSION`, `PromptProfile` (Qwen3 thinking-off, system-role probe), prompt rendering from the compiled plan (`_user_content`) with the nonce context delimiter (`_context_block`), prefill + broadcast KV, chunked batched passes (`_score_rows` with per-chunk halve-and-retry and `failed_attempts`), width-bin memory budget (`_width_bin_max_rows`, slope measured at load by `_measure_width_slope`), constrained-path trie scoring, multi count row + reconciliation (`COUNT_MARGIN_MIN`), constrained MAP (`_constrained_map`), constraint validation before model work (`constraints.validate_constraints_for_schema` at the top of `run_parallel_generation`), scalar finalization (`ScalarEvidence` / `ScalarDecision` / `Candidate`, `finalize_scalar_evidence` — the ONE scalar finalizer every scoring path goes through), dependency second pass rebuilt on it (`_selective_second_pass`: topological waves, `Given:` conditioning header, `InternalConstraintViolationError` on post-MAP violation), near-tie batch=1 rescore, prior cache (`_prior_cache_key`: neutral-prompt sha + id(model)/id(tokenizer), LRU), multi calibration (`_load_calibration`), result dict. |
| [`jevmlx/setcons.py`](jevmlx/setcons.py) | Hard set-constraint selection for multi fields: exact DP over group components (mutually_exclusive, at_most_one, at_most_k, at_least_one, exact_k) plus implies propagation; score-maximizing feasible set. |
| [`jevmlx/constraints.py`](jevmlx/constraints.py) | Case-level constraint checking (implies/requires_parent, excludes, exclusivity): single source of truth shared by the engine's MAP and `evalmetrics`' violation rate. |
| [`jevmlx/parity.py`](jevmlx/parity.py) | Scoring parity (batch=1 vs batched vs chunked) over the bundled presets with their REAL contexts, plus `check_batched_parity` — the decide_many parity matrix (1/2/4 contexts, raw pre-rescore row logits + final decisions, prior on/off). ONE implementation shared by the slow parity test and the bench's `parity.json` producer. |
| [`jevmlx/api.py`](jevmlx/api.py) | Public API: `decide`, `decide_many`, `Decision`/`FieldResult`, Pydantic → schema, NONE_OF_ABOVE + abstention handling, unit-split margins. No compat re-exports (model aliases live in `jevmlx.models`). |
| [`jevmlx/cli.py`](jevmlx/cli.py) | Subcommands: decide, serve, validate, eval, report, calibrate, bench, doctor. |
| [`jevmlx/serve.py`](jevmlx/serve.py) | HTTP server (`POST /decide`), one serial worker on the single Metal GPU. |
| [`jevmlx/models.py`](jevmlx/models.py) | Model aliases (`MODEL_ALIASES`, `DEFAULT_MODEL`) and `resolve_model` — the leaf module every loader and CLI command resolves ids through. |
| [`jevmlx/lint.py`](jevmlx/lint.py) | `lint_schema`: collision / duplicate / empty-choice findings from a compiled plan. |
| [`jevmlx/calibrate.py`](jevmlx/calibrate.py) | `fit_temperature` by NLL over labeled JSONL and `fit_logistic` (pooled multi calibration); ECE before/after; one JSON output for both. |
| [`jevmlx/evalrun.py`](jevmlx/evalrun.py) | Eval tracks (`parallel_decide_fn`, `naive_local_decide_fn`); `run_eval` writes `predictions.jsonl` + `run.json`. |
| [`jevmlx/evalmetrics.py`](jevmlx/evalmetrics.py) | Offline metrics from prediction lines (accuracy, calibration, bias, agreement, constraint_violation_rate). |
| [`jevmlx/evalreport.py`](jevmlx/evalreport.py) | `report.json` + markdown rendering of a run. |
| [`jevmlx/baseline.py`](jevmlx/baseline.py) | OpenAI-compatible chat client used by the API track. |
| [`jevmlx/openai_slots.py`](jevmlx/openai_slots.py) | OpenAI-compatible slot backend: one request per option, top-k logprobs with an explicit floor and a `truncated` flag. |
| [`jevmlx/bench.py`](jevmlx/bench.py) | Dataset × scorer × track matrix runner writing results directories; writes `parity.json` per model folder right after the engine load. |
| [`jevmlx/log.py`](jevmlx/log.py) | Logging configuration (`-v`, `JEVMLX_LOG=json`). |
| [`jevmlx/timing.py`](jevmlx/timing.py) | W5b-8 event ledger for engine timing (`Ledger`, `Interval`, `SpanError`): non-overlapping named spans in two phases (`prior`, `main`), with `Ledger.derived_flat` deriving today's `*_ms` result keys (`plan_compile_ms`, `prefill_ms`, `cache_broadcast_ms`, composite `suffix_eval_ms`, `lm_head_gather_ms`, `second_pass_ms`, `prior_ms`, `elapsed_ms`, `total_ms`) and `Ledger.batched_views` producing `group_wall` / `per_item_amortized` / `per_item_end_to_end`. Standalone — pure Python, no mlx import, NO engine wiring yet: the engine's ad-hoc timers are untouched until W6 adoption. |
| [`jevmlx/adapters.py`](jevmlx/adapters.py) | W6-1 LM-head adapters (`LMHeadAdapter` protocol, `adapter_for`, `UnsupportedModelError`, `list_supported_model_types`): backbone/lm_head split per installed mlx_lm family (untied `lm_head`, tied `embed_tokens.as_linear`, gemma3 always-head, biased phi head, mistral3 delegation). Registry-only — NO engine wiring yet; engine call sites adopt it in W6-2+. |
| [`benchmarks/to_jsonl.py`](benchmarks/to_jsonl.py) | Bundled `cases.json` → eval JSONL (+ lock). |
| [`benchmarks/typesafe/fetch.py`](benchmarks/typesafe/fetch.py) | TypeSafe public pages → eval JSONL (+ lock, `benchmark_only`). |
| [`benchmarks/perturb.py`](benchmarks/perturb.py) | Deterministic label-preserving context perturbations. |
| [`benchmarks/synthetic.py`](benchmarks/synthetic.py) | Synthetic labeled case generator. |
| [`benchmarks/invariance.py`](benchmarks/invariance.py) | Irrelevant-field invariance benchmark (an invariance gate). |
| [`benchmarks/check_results.py`](benchmarks/check_results.py) | Results-folder audit: recomputes metrics from `predictions.jsonl`, verifies committed `report.json` numbers, enforces the results contract v2 gates (timing.json median split incl. `peak_incremental_bytes`/`failed_attempts`; batched per-item keys together; pre-v2 parity.json fails with a regenerate hint), and `--check-readme` compares the README leaderboard block against a rebuilt table. |
| [`benchmarks/summarize_results.py`](benchmarks/summarize_results.py) | Results directories → `SUMMARY.md` (marks `parity_failed` rows for models whose parity check failed). |
| [`benchmarks/timing.py`](benchmarks/timing.py) | Timing report: decide() per bundled preset, N reps; the engine's full split (median/p95/min/max) plus rows, padded token positions, passes, rescored count, rerun rate; per-rep raw numbers alongside. |
| [`benchmarks/leaderboard.py`](benchmarks/leaderboard.py) | README leaderboard table (TypeSafe agreement + local rows). Local time-per-case reads ONLY the per-item end-to-end median (contract v2) — a parallel combo without it raises `ValueError` naming the folder (no pre-v2 folders on main; no fallback). |
| [`benchmarks/compat.py`](benchmarks/compat.py) | Model compatibility table (latency/memory) generator. |
| [`benchmarks/naive_vs_parallel.py`](benchmarks/naive_vs_parallel.py) | Quick parallel-vs-naive side-by-side comparison. |
| [`benchmarks/m5.py`](benchmarks/m5.py) | One-command M5 runbook: doctor gate → slow parity per model → quality bench → invariance → timing → remaining parity models → optional `--ab-branch` A/B worktree → `SUMMARY.md` comparison. Steps are subprocesses logged into `<out>/RUNBOOK.md`; idempotent via output markers (`--fresh` reruns); step planner and summary builder are pure functions. |
| [`benchmarks/probe.py`](benchmarks/probe.py) | W6-2 prep probes, standalone (no engine wiring): `slope` fits the per-row peak-memory slope per width bin (B=1/2/4/8 at widths 4/8/16/32 under `mx.reset_peak_memory`) and `adapters` compares `adapter.lm_head(adapter.backbone(x)[:, pos])` vs `model(x)[:, pos]` on bundled preset rows. |

## Data flow

```
schema (Pydantic or JSON)
  │  StructuredSchema compile (slot / labels / multi plans, per tokenizer)
  │    - bounded codebook search per scalar field (_search_codebook:
  │      pre-tokenized pool, prefix-conflict graph, bounded backtracking for
  │      a size-n prefix-free independent set; lexicographic objective on
  │      every complete set — never greedy)
  │    - multi fields: '<field>/<code>' option rows + '<field>#count' row
  │    - set constraints validated at construction (SchemaCompileError on
  │      contradictions: cycles, implies×exclusivity, k bounds)
  ▼
plan {lead_in_ids, fields: {shared_ids, remainders/trie, alias_map?, count?}}
  │  (identity-keyed plan cache per tokenizer; weakref-evicted)
  ▼
prompt  (PROMPT_VERSION = "jevmlx-parallel-v8" from the engine;
  │      PROMPT_V2_SYSTEM paragraph, user schema block + <<<CONTEXT:<nonce>
  │      … CONTEXT:<nonce>>> built inside run_parallel_generation; the
  │      schema block renders FROM THE COMPILED PLAN (to_alias_schema_str —
  │      the prompt shows the codebook the scorer reads); the nonce is the
  │      context's sha256-derived tag so a fake interior fence cannot close
  │      the block; chat template via PromptProfile)
  │  prefill ONCE (make_prompt_cache + model(base_arr))  →  KV cache
  │    →  broadcast ×rows (prior pass runs first only with prior_correction)
  ▼
batched suffix pass(es)  (_score_rows -> ScoreRowsResult(row_logits,
  │    row_legal_mass_log, passes, gather_ms, broadcast_ms, chunk_shapes,
  │    failed_attempts); chunked by the width-bin memory budget
  │    _width_bin_max_rows (slope MEASURED at engine load by
  │    _measure_width_slope, B=1/B=2 peak-activation ratio), halve-and-retry
  │    PER CHUNK on Metal allocation failure — failed_attempts counts the
  │    retries, never folded into passes)
  │  branch-point logits at each row's decision position (gather-only eval)
  ▼
trie scoring  (score_trie: P(choice) = Π branch softmax factors, T applied
  │    once downstream; legal-mass LOGS per branch from the full-vocab
  │    logsumexp — log space end to end)
  ▼
near-tie rescore  (scalar top candidates and multi Y/N pairs inside
  │    INSTABILITY_BAND (5e-2 nats) rescored at the canonical batch=1 shape,
  │    multi branch at option_pair, scalar at scores; rescored_fields telemetry)
  ▼
multi selection  (calibrated log-odds a*(yes-no)+b > 0, else P(yes) >= 0.5;
  │    count row <field>#count reconciles the set to top-k by calibrated
  │    COUNT_MARGIN_MIN = 0.7 nats)
  ▼
hard set constraints  (setcons.select_constrained_set: threshold/count
  │    propose, the exact DP picks the score-maximizing feasible set;
  │    applies LAST, after count reconciliation)
  ▼
constrained MAP  (case-level constraints: _constrained_map maximizes summed
  │    log scores over the joint assignment; constraints are validated
  │    against the schema BEFORE any model work —
  │    constraints.validate_constraints_for_schema, called at the top of
  │    run_parallel_generation, rejects unknown types/fields, out-of-domain
  │    values, and implies/requires_parent/excludes on multi fields)
  ▼
selective second pass  (depends_on children whose parent is confident and
  │    own margin low / MAP-changed: _selective_second_pass, rebuilt on the
  │    shared scalar finalizer — rows run in TOPOLOGICAL WAVES, each row a
  │    complete one-field JSON object prefixed by a `Given: {"parent": …}\n`
  │    header; depth-2 children condition on the UPDATED depth-1 decisions;
  │    after the last wave MAP re-runs over affected components and a
  │    constraint violation after that raises
  │    InternalConstraintViolationError — an internal error, never a
  │    returned decision)
  ▼
assembly  (winners → typed values via alias_map; multi = per-option Y/N
  │    codes at T=1; row codes '00','01',… map back to choices)
  ▼
result dict  {parsed_json, field_telemetry, prompt_sha256, full timing
  │    split incl. plan_compile_ms / cache_broadcast_ms / padded_token_positions,
  │    peak_active_bytes + peak_incremental_bytes, failed_attempts, …}
  │
  ├──► api.Decision / FieldResult        (Python)
  └──► evalrun predictions.jsonl lines   (eval) / CLI table       (decide)

scalar decisions (W5-B, PR #45): every scoring path — normal batched pass,
batch=1 rescore, dependency rescore, oracle rescore — produces a
ScalarEvidence (per-choice T=1 log-scores + legal-mass logs, source_shape
'batch' | 'batch1' | 'dependency' | 'oracle'), and every decision goes
through finalize_scalar_evidence, the ONE finalizer (rescore gate, prior
correction, temperature, tie flag, margins, top_choices). No scoring
semantics live outside it. ScalarDecision carries the complete public
telemetry (evidence_source included); Candidate is a typed value + its
score-key ('true'/'false' for booleans — conflating the scored and typed
representations turned True into "true" in reconciled assignments).

batched (decide_many): run_parallel_generation_batched prefills each
context, builds the rows ONCE, then processes context groups built
INCREMENTALLY from actual cumulative cache bytes + projected suffix cost
(contexts sorted by prompt length); ONE merged scoring pass per group with
per-row cache slots; prior computed ONCE per call; per-result timing keys
group_wall_ms / per_item_amortized_ms / per_item_end_to_end_ms /
contexts_per_pass (the ACTUAL group size).

bench: after the engine load, jevmlx/parity.py runs the batch=1 vs batched
vs chunked parity check over the four bundled presets (their real contexts)
and writes <model folder>/parity.json BEFORE any eval combo
(bench._run_model_parity, called in run_bench_models right after the engine
load).
```

## Contracts

### Engine result dict — the dict `run_parallel_generation` returns

| Key | Meaning |
|---|---|
| `elapsed_ms` | Wall clock for the decision (excludes the prior pass). |
| `prior_ms` / `prefill_ms` / `plan_compile_ms` / `cache_broadcast_ms` / `suffix_eval_ms` / `lm_head_gather_ms` / `total_ms` | The honest timing split: neutral prior pass (0.0 when `prior_correction` is off), prefill, plan compilation (plan cache makes it ~0 warm), broadcast+prepare+eval of the per-chunk cache copies (distinct from the forwards), batched suffix, decision gather inside the suffix window, and everything (`total_ms == elapsed_ms + prior_ms`). |
| `second_pass_ms` / `rerun_fields` / `rerun_rows` | The `depends_on` second pass: wall time, which fields were re-decided, how many conditioned rows ran (0.0/[] when no `depends_on`). |
| `padded_token_positions` | W3-R: total suffix token positions including right padding — sum of (chunk width x chunk rows), the tiling shape the forwards actually ran at. |
| `rescored_fields` | Fields whose batched result was replaced by the batch=1 canonical rescore. |
| `total_tokens_generated` | Always 0: no text is generated. |
| `peak_active_bytes` / `peak_incremental_bytes` | W5-D finding 32: absolute Metal peak since the last `mx.reset_peak_memory()`, and the INCREMENTAL peak over the request's starting active memory (the old single number could describe an earlier request or the warmup). |
| `sequential_forward_passes` | 1, or the chunk count from the width-bin memory budget. |
| `failed_attempts` | W5-D finding 30: Metal allocation failures that were retried at a smaller chunk size (`ScoreRowsResult.failed_attempts`). Never folded into `sequential_forward_passes` — a pass is a forward that produced rows. |
| `schema_match` | Always True (keys/enums guaranteed by construction). |
| `confidence_model` | `"slots"` or `"labels"`. |
| `prompt_sha256` / `prompt_version` | SHA-256 over the full prompt token ids; the version is read from `engine.PROMPT_VERSION` (v8) — never a literal elsewhere. |
| `probability_status` | How to read the probabilities. |
| `prior_correction` / `constraints_applied` | Whether the prior pass ran / case-level constraints were applied. |
| `reconciled_fields` | Fields whose value changed under constrained MAP. |
| `parsed_json` | `{field: {"value": …, "prob": …}}`. |
| `field_telemetry` | `{field: entry}` — see next table. |
| `num_fields` | Field count. |

Batched-only keys (`run_parallel_generation_batched`, every result): `group_wall_ms` (the group's wall time incl. prefill+scoring+assembly), `per_item_amortized_ms` (group wall / group size), `per_item_end_to_end_ms` (this context's own prefill + its share), `contexts_per_pass` (the ACTUAL group size — the final partial group reports its own smaller size). With `prior_correction=True` the neutral pass is computed ONCE per call and every result reports the shared `prior_ms`. |

### `field_telemetry` entry — built in `run_parallel_generation`'s field loop (multi), scalar branch, and the `<field>#count` branch

| Key | Meaning |
|---|---|
| `value` | Decided value (str / bool / list[str]). |
| `type` | `boolean`, `enum`, or `multi`. |
| `probability` | P of the winner; multi: `None` — no field-level probability is claimed (per-option decisions only). |
| `cardinality` | Number of choices. |
| `log_scores` | Constrained-path log P per choice at T=1, keyed by real choice string. Absent for multi (no field-level log P exists). |
| `per_option` | multi only: independent per-option P(yes) (post prior/softmax). |
| `option_logit_pairs` | multi only: raw [yes, no] logits per option at T=1 (what the prior cache and the multi calibrator consume). |
| `calibrated` | multi only: the applied `{"a", "b"}` pooled-logistic calibrator, or None (fixed P(yes) >= 0.5 rule; the threshold knob is deleted). With calibration, `calibrated_log_odds` rides alongside. |
| `count_choice` / `count_margin` / `reconciled_by` | multi only: the `<field>#count` row's winning bucket ('0'..'4', where 4 means four or more), its top-2 log-score margin (nats), and which rule produced the selected set ('count' when the margin cleared `COUNT_MARGIN_MIN`, else 'per_option'). The count row itself surfaces as a separate `<field>#count` telemetry entry (scalar-shaped, `margin_nats`). |
| `set_constraints` / `set_selection` | multi only, present only when the field declares set constraints: the constraints verbatim and whether the solver changed the selection ("constraints") or they didn't bind ("per_option"). |
| `margin` | Multi only: min \|P(yes) - cut\| over the final set in probability units (engine-side name; the API exposes it as `threshold_distance`; an option forced in against its P(yes) shows 0). |
| `top_choices` | Top (choice, probability) pairs, most probable first (top 5). |
| `rows` | Rows the field consumed (0 for cardinality-1 fields). |
| `tie` / `rescored` | Scalar: whether the top-2 gap is inside `INSTABILITY_BAND` (subsumes exact-equality ties), and whether the batch=1 rescore replaced the batched result. Multi: `rescored` when any option's Y/N pair was rescored. |
| `evidence_source` | Scalar only (W5-B, PR #45): which scoring path produced the final evidence — `batch` (batched pass), `batch1` (canonical rescore replaced it), `dependency` (second pass), `oracle` (forced re-score). Rides on scalar entries and on second-pass re-decides. |
| `legal_mass` | Probability the model assigned to the union of allowed continuations at the winner's branch point(s), against the full vocabulary = sum(exp(z_allowed)) / sum(exp(z_vocab)). Per-branch leakage signal — the constrained distribution can confidently pick A over B even when almost all unconstrained mass is on a reasoning token/newline/label text. Product over the winner's branch path (scalar); per-option Y/N branches (multi); the count row has its own (`legal_mass` + `min_option_legal_mass` on the `<field>#count` entry). 1.0 for cardinality-1 fields (nothing branched). Always computed. Raw, pre-prior-correction logits. |
| `legal_mass_logs` | Per-choice (scalar) / per-option (multi) natural-log legal-mass product along the branch path, keyed by the real choice/option string. Raw, T=1. Calibration feature for the abstention model. The legal-mass callback is LOG-space end to end (W5-D finding 37: `score_trie`'s `legal_mass_at_node` returns natural-log floats; the trie stays MLX-free). |
| `min_option_legal_mass` / `mean_log_legal_mass` | Multi only (W5-D finding 38): cardinality-free field-level stats replacing the old underflowing, cardinality-confounded product as the headline numbers — the worst option's legal mass in probability space, and the mean per-option log mass (additive, stable). The per-option logs stay on `legal_mass_logs`. |
| `oracle_prediction` / `oracle_log_scores` | Present only under `oracle_overrides` (DAG evaluation): the field was forced to the given value and re-scored conditioned on it. |
| `prior_log_scores` / `prior_corrected` (scalar) / `prior_option_pairs` / `prior_corrected` (multi) | Present only under `prior_correction`: the neutral-context prior the field's scores were corrected against. |

### `FieldResult` — `api.FieldResult` (built by `_build_field_results`)

| Field | Meaning |
|---|---|
| `value` | Decided value (engine-side). |
| `score` | Log P of the winner; 0.0 for multi (no field-level log score). |
| `log_score_margin` | Scalar only: top1-top2 log score at T=1 (decision units). None for multi. |
| `probability_margin` | Scalar only: top1-top2 probability (post-temperature). None for multi. |
| `threshold_distance` | Multi only: min \|P(yes) - cut\| from the engine's `margin`. None for scalar. Margins are unit-split: no field carries more than one of the three. |
| `probability` | P of the winner; None for multi. |
| `calibrated` | False until a fitted calibrator is applied (engine never calibrates). |
| `model` | `"slots"` or `"labels"`. |
| `alternatives` | Top 3 (choice, probability) pairs; multi: per-option (option, P(yes)) sorted desc. |
| `reason` | None, `"none_of_above"` (caller opted in via `allow_none_of_above=True`, model picked the explicit opt-out → None), or `"abstain"` (`abstain_below_margin` set and the field's margin — `probability_margin` scalar / `threshold_distance` multi — fell below the cut; value withheld from the validated instance, raw kept for provenance). The single source of truth — no separate abstain flag. |

### Case-level constraints — `constraints.py`, applied by `_constrained_map`

| Type | Shape | Meaning |
|---|---|---|
| `implies` / `requires_parent` | `{parent, child, mapping}` | child's value must be in `mapping[parent_value]` (unmeasured fields can't violate). |
| `excludes` | `{field, value, other}` | when `field == value`, `other` must be empty/falsy. |
| `exclusivity` | `{group}` | at most one member of the group may hold a value. |

`check_constraint` is the one evaluator, shared by the
engine's MAP reconciliation and `evalmetrics.constraint_violation_rate`;
`validate_constraints` checks shapes up front. W5-B (PR #45) added
`validate_constraints_for_schema`: the engine calls it at the top of
`run_parallel_generation` — BEFORE any model work — so unknown constraint
types, unknown fields, out-of-domain values, and implies/requires_parent/
excludes on multi fields all fail loudly at call time (review 13:
case-level constraints cannot be reconciled on multi fields today;
exclusivity is the supported multi shape). The neutral prior pass runs
with constraints=None, so validation never fires twice. After the
dependency pass's final MAP re-run, a still-violating assignment raises
`InternalConstraintViolationError` (engine.py) — an internal error, never
a returned decision.

### Per-field set constraints — `FieldDefinition.compile_set_constraints`, solver `setcons.select_constrained_set`

Schema-dict key `set_constraints` on a multi field; validated at
construction, so contradictory sets raise `SchemaCompileError` before any
model load. The field is a frozen dataclass, so "attaching" constraints
means deriving: `compile_set_constraints` returns a NEW `FieldDefinition`
via `dataclasses.replace` (a bare reference to the old field keeps the empty
tuple), and the stored constraints are a tuple of read-only views.
Types: `mutually_exclusive` / `at_most_one` / `at_most_k` (k),
`at_least_one`, `exact_k` (k), `implies` (`if_option` → `then_option`).
Compile-time contradictions: implies cycles, an implies pair inside an
at-most-one group, implies into an `exact_k=0` group, k above group size,
unknown options, self-implication. The solver selects the score-maximizing
feasible set (calibrated log-odds, else raw yes/no log-odds); non-binding
constraint sets reproduce the proposal exactly.

### `parity.json` — producer `parity.write_parity_json`, gate `summarize_results`

Written by the bench into each model folder right after the engine load
(`bench._run_model_parity`), before any eval combo:

```json
{
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "prompt_version": "jevmlx-parallel-v8",
  "max_abs_drift_nats": 0.027,
  "winners_identical": true,
  "atol": 0.05,
  "passed": true,
  "cases": ["code_security", "fintech_fraud", "high_cardinality_255", "support_triage"],
  "test": "test_w1a_scoring_parity_batch_vs_chunked_real_model",
  "run_at": "2026-09-18T12:00:00Z"
}
```

(Four bundled presets on disk; the docstring example shows two. Field names
match `parity.py` verbatim: `model`, `prompt_version`, `max_abs_drift_nats`,
`winners_identical`, `atol`, `passed`, `cases`, `test` = `PARITY_TEST_NAME`,
`run_at`.)

One check, two consumers: the slow parity test and the bench share
`check_scoring_parity`. A model without a passing `parity.json` gets
`parity_failed` rows in SUMMARY.md (`summarize_results._model_parity_note`,
which also gates a MISSING parity.json) and cannot enter the README compat
table. W5 (PR #48) made the gate v2-aware: a parity.json without
`max_raw_row_drift_nats` is pre-v2 and FAILS with a regenerate hint, and a
failing payload names its stages via the shared
`_parity_failed_stages` (winners flipped / final log-score drift ≥ atol /
raw pre-rescore row-logit drift ≥ atol) plus the max drifts and atol;
`check_results.check_parity` uses the same classifier.

### Batched parity matrix — `parity.check_batched_parity`

The W1-A gate only exercised `run_parallel_generation`, so a `decide_many`
bug passed it. `check_batched_parity` runs each case at 1/2/4 contexts,
equal and mixed prompt lengths, and compares RAW row logits BEFORE the
near-tie rescore (W5-D finding 42: one scoring pass per context through the
shared `_score_rows` at identical chunk shapes vs the batched group's rows —
so a batch-1 rescore cannot mask raw batch drift), final decisions (parsed
values + log_scores), and prior on/off (`prior_correction=True` computes the
shared prior and must still match decide-per-context). Returns the
`check_scoring_parity` shape plus `max_raw_row_drift_nats` and a
per-case `max_raw_row_drift_nats`. Parity cases use the bundled presets'
REAL contexts (W5-D finding 41: `bundled_preset_specs` returns the whole
preset; `_case_context` falls back to a stable filler only for bare test
schemas).

### Eval `cases.jsonl` line — writers: `to_jsonl.py`, `typesafe/fetch.py`

| Key | Meaning |
|---|---|
| `id` | Record id (`typesafe/<wf>/<case>[/n<k>]`, `quality-eval/…`). |
| `group_id` | Grouping key (original case id; = id for bundled cases). |
| `source` | `typesafe` or `quality-eval`. |
| `workflow` | Workflow name or null. |
| `benchmark_only` | True for TypeSafe-derived pseudo-labels (never calibration/routing data). |
| `schema` | Field → {type, description, choices}. |
| `context` | Documents rendered into one string. |
| `labels` | Field → consensus label. |
| `split` | `train`/`holdout` (TypeSafe: deterministic sha1-of-id rule in the fetcher's `split_for`; synthetic: every fifth case). |
| `meta` | Provenance: consensus distributions, ambiguity, per-model answers. |

### `predictions.jsonl` line — `evalrun.PREDICTION_LINE_KEYS` (frozen), written by `run_eval`

Frozen contract; `check_results.py` imports this list instead of retyping it.

| Key | Meaning |
|---|---|
| `run_id`, `case_id`, `group_id`, `source`, `workflow` | Provenance of the decision. |
| `field`, `type`, `track`, `model`, `permutation` | What was decided, by what. |
| `label`, `prediction`, `valid`, `correct` | Scored comparison (correct=None when unlabeled). |
| `log_scores`, `probability`, `per_option` | Distribution evidence (multi uses per_option). |
| `latency_ms`, `rows`, `passes` | Engine timing telemetry. |
| `error`, `salvage_prediction` | Failure provenance. |
| `perturbation`, `consensus`, `oracle_prediction` | Optional; present only with `carry_perturbation` / `carry_consensus`, or on dependency/oracle re-decide lines (DAG evaluation). |

### `run.json` — written by `run_eval`

`run_id`, `environment`, `config` (model, temperature 1.0, track,
dataset_path, permutations, split, model_revision, quantization,
prompt_version read from the engine, tokenizer chat-template SHA-256,
compiled plan SHA-256, dataset lock SHA-256), and `counts` (cases, fields,
prediction_lines).

### `timing.json` (per combo) — written by `run_eval` when the parallel track ran

`{"calls": <canonical decide calls>, "median": {<split key>: <median over
cases>}}` — split keys ride the parallel `_meta` (`latency_ms`, the full
engine split, `peak_active_bytes`, `padded_token_positions`,
`rescored_fields_count`, `rerun_fields_count`, `num_fields`; results
contract v2 adds `peak_incremental_bytes` + `failed_attempts` —
check_results.REQUIRED `RUN_TIMING_KEYS`). Batched-path runs also land
`group_wall_ms` / `per_item_amortized_ms` / `per_item_end_to_end_ms`
(check_results `BATCHED_TIMING_KEYS`, required together when present).
Same calls the predictions came from — no second run. Naive/openai tracks
write none (no engine split exists there).

### `benchmarks/timing.py` report (standalone) — `run_timing` / `aggregate`

`{model, reps, prior_correction, presets: {<preset>: {title,
num_fields, aggregate, raw}}}`; aggregate = median + p95 (nearest-rank) +
min/max per key over `TIMING_KEYS` (prior_ms, prefill_ms, plan_compile_ms,
cache_broadcast_ms, suffix_eval_ms, lm_head_gather_ms, second_pass_ms,
total_ms) plus `COUNT_KEYS` (peak_active_bytes, rows,
padded_token_positions, sequential_forward_passes, rescored_fields_count,
num_fields) and `rerun_rate` (rerun fields / fields). CLI:
`python -m benchmarks.timing --model <id> [--reps N] [--out DIR]`.

### `dataset.lock.json` — `typesafe/fetch.write_outputs` and `to_jsonl.main`

TypeSafe: `sources` [{url, sha256, fetched_at, etag}], `parser_version`,
`counts`, `cases_sha256` (sha256 of the JSONL written next to it). Bundled
conversion: same shape with empty `sources` plus `fetched_at`.

## Invariants

- **Token-aligned candidates.** The full candidate text (field key + value)
  is tokenized as one string, and the row holds `lead_in + shared + branch`
  token ids. Splitting candidate text anywhere but at token boundaries would
  score tokens the model never sees.
- **One lead-in for every row.** Plans store suffixes *without* the schema
  lead-in; the engine prepends it exactly once per row, one rule for scalar,
  multi, count, and cardinality-1 rows (the row-build loop in
  `run_parallel_generation`).
- **Compile-time rejections** (`StructuredSchema.__init__` + plan
  compilation): dots/slashes/'#' in field names (row-key injectivity),
  duplicate choices, token-identical alias candidates (alias, count-row,
  and labels compilers), strict token-prefix pairs, candidates sharing no
  token prefix, contradictory set constraints (`compile_set_constraints`) —
  all raise before any model load.
- **Probability semantics.** `log_scores` are constrained-path log
  probabilities at T=1; temperature is applied once to the final per-choice
  distribution (ranking-invariant); `FieldResult.calibrated` is False until
  a fitted calibrator runs. For multi fields `probability` is `None` — the
  engine does not claim a field-level probability for an option set
  (per-option decisions only).
- **One tolerance number.** `engine.INSTABILITY_BAND` (5e-2 nats) is the
  single source of truth for batch-shape FP noise; the test suite re-exports
  it as `PARITY_ATOL` and `parity.json` records it as `atol`. Near-tie
  rescoring, the parity test, the memory probe, and the parity producer all
  read the same constant — no second tolerance literal anywhere.
- **One parity implementation.** The slow test and the bench's
  `parity.json` producer call the same `jevmlx/parity.py` function; recorded
  parity and tested parity cannot drift apart.
- **No dual paths.** The multi threshold knob is deleted (calibrated
  log-odds > 0 or the fixed P(yes) >= 0.5 rule); `set_constraints` is a real
  frozen attribute read directly (no getattr fallback); `summarize` lives in
  `benchmarks/summarize_results.py` and bench calls it directly.
- **Schemas and plans are immutable.** `StructuredSchema` and
  `FieldDefinition` are frozen dataclasses; mutation raises
  `FrozenInstanceError` (derive with `dataclasses.replace`). Compiled plans
  and `set_constraints` are exposed as read-only mappings/tuples
  (`_freeze_plan` / `MappingProxyType`). A schema is compiled once and
  shared across engines, threads, and plan caches — in-place mutation would
  silently invalidate every cached plan keyed to it. `api` no longer
  re-exports `models.*`: aliases resolve via `jevmlx.models.resolve_model`
  (imported by the engine) and `jevmlx.models` is the public home of
  `MODEL_ALIASES`/`DEFAULT_MODEL`.
- **Prompt version read, never written.** `PROMPT_VERSION` lives once in
  `jevmlx.engine`; every consumer (result dict, prior cache key, parity
  payload) reads it.
- **One serializer.** `jevmlx.json_text.json_text` (`ensure_ascii=False`) is
  the only sanctioned JSON rendering for prompt and candidate text (W5-A
  finding 39): the stdlib default escaped non-ASCII labels in candidate
  compilation while prompts showed the real characters, so the scorer judged
  tokenizations the model never saw.
- **The compiled plan owns the prompt.** `to_schema_str(mode, tokenizer=…)`
  renders the aliases THE COMPILED PLAN scored for this tokenizer (W5-A
  finding 1) — prompt and scorer can never disagree about the protocol.
  `alias_for_index` survives only for the OpenAI adapter's per-field
  requests, which bypass the plan.
- **The context cannot impersonate its own fence.** `_context_block` fences
  the context with a sha256-derived nonce tag (`_context_nonce`): open and
  close fences match per context, and an interior `CONTEXT>>>` line cannot
  close the block early (W5-A finding 44).
- **The prior cache is identity-keyed.** `_prior_cache_key` carries
  id(model)/id(tokenizer) with live weakref verification on every hit (an id
  can be reused after free), plus the neutral prompt's full sha256 (the plan
  hash alone omits descriptions/glosses/order), prompt_version, and scoring
  mode; eviction is LRU (`_PRIOR_CACHE`, `_PRIOR_CACHE_MAX`), and
  non-weak-referenceable tokenizers are simply not cached.
- **Retry is per chunk and honest.** A Metal allocation failure halves that
  chunk and retries; later chunks in the same bucket still run
  (`_score_rows`, W5-D finding 30). Retries surface as `failed_attempts`,
  never as `sequential_forward_passes`.
- **No backward compatibility.** Changes replace: old paths, keys, flags and
  names are deleted with their callers and tests in the same change. No
  aliases, no shims, no deprecation periods.

## Testing

- Fast tests (`pytest -m "not slow" -q`) never touch a model: the shared
  fakes live in `tests/conftest.py` (`FakeModel`, `FakeTokenizer`, the
  %97+1 tokenizer, `YNLogitModel`, `CountCodeModel`) — one class per
  behaviour, re-exported by the per-file aliases — plus the engine-result
  factories `make_engine_result` / `make_field_telemetry`, whose key sets
  are pinned to `run_parallel_generation`'s real output by contract tests
  in that file (a stub that drifts from the engine contract fails loudly).
  A fake tokenizer mapping words to crc32 ids lives in
  `tests/test_lint.py`, and `decide_fn` seams in `tests/test_evalrun.py`.
- Slow tests (marker `slow`) load
  `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Run one locally with
  `uv run pytest tests/test_engine.py -m slow -k slots -q`.
- The parity check has both paths by design: exact assertions on the
  deterministic fake model, `INSTABILITY_BAND`-bounded assertions on a real
  model (`tests/test_w4b_parity.py`).
- Eval/benchmark code is tested against synthetic viewer payloads and
  fixture cases; nothing downloads from the network (see
  `tests/test_typesafe_fetch.py`).

## Design decisions

1. **Trie over first tokens.** Scoring whole choices as branch paths gives a
   proper distribution and lets shared token prefixes be paid for once.
2. **Codebook search, not pinned letters.** The compiler searches alias
   codes per field and tokenizer (`_search_codebook`) and picks the best
   size-n prefix-free INDEPENDENT set over the pre-tokenized pool's
   prefix-conflict graph — bounded backtracking, scored by the lexicographic
   objective (branch nodes, trie depth, length variance, code length) on
   every complete set; greedy committed to a prefix-colliding code and
   rejected valid sets. The pinned index-based mapping survives only for
   index-based callers (the OpenAI adapter).
3. **Slots default over labels.** Neutral aliases decouple the scored
   vocabulary from choice text — no tokenization collisions by construction;
   labels mode stays for schemas where spelling is the signal.
4. **JSON rows kept.** The decision row stays `'<field>": '` so assembly is
   plain JSON parsing, not a positional protocol. Multi option rows are
   keyed `'<field>/<code>'` (zero-padded 2-digit index) and the count row
   `'<field>#count'` — field names reject `.`/`/`/`#` at compile time so
   these keys stay injective.
5. **Multi as one-vs-rest.** Each option is an independent true/false
   decision at its own divergence point; subset probability semantics would
   not be additive. The count row is a reconciliation signal gated on its
   own confidence, never a hard answer.
6. **Measured budget, not hard bounds.** Rows per chunk come from an
   active-memory budget charged per width bin (`_width_bin_max_rows`: one
   row's cache bytes + the logits slab at the row's OWN width bin, scaled by
   the tiling slope); the B=1/B=2 slope is MEASURED at engine load
   (`_measure_width_slope`, one probe inside the warmup, floor 1.0 on any
   failure) instead of assumed; a warning logs when a pass is split, and a
   Metal allocation failure halves the chunk and retries (counted in
   `failed_attempts`) while still scoring every row.
7. **Identity-keyed plan cache.** Compiled plans are cached per tokenizer
   object identity (weakref, evicted on death) so repeated decisions skip
   recompilation without leaking tokenizers — and two equal-but-distinct
   tokenizers never share a plan.
8. **Failure is a result.** A bench model that cannot load writes
   `load_failed` rows and keeps going; a parity failure is recorded and
   gates the model's rows instead of crashing the run or producing unvetted
   accuracy numbers.
9. **Provenance fields everywhere.** `prompt_sha256`, `prompt_version`,
   `dataset.lock.json`, `run.json`, and `parity.json` exist so any number in
   a report can be traced to the exact prompt, data, and parity state that
   produced it.
