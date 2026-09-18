# ARCHITECTURE.md

For people who want to read or change the code. Line references cite
`file:line` at the time of writing (branch point `a26a114`); trust the code
over this document when they drift.

## Module map

| Module | Role |
|---|---|
| [`jevmlx/schema.py`](jevmlx/schema.py) | Schema model (`StructuredSchema`, `FieldDefinition`), slot/labels/multi plan compilation, tokenizer-specific codebook search (`_search_codebook`), count-row compilation, per-tokenizer plan cache, compile-time rejections, hard set-constraint validation (`FieldDefinition.compile_set_constraints`). |
| [`jevmlx/trie.py`](jevmlx/trie.py) | Branch-point trie over candidate token remainders; `score_trie` (constrained-path log probs + legal-mass logs); softmax/logsumexp helpers. |
| [`jevmlx/engine.py`](jevmlx/engine.py) | Model load (lru_cached, the only platform check), `PROMPT_VERSION`, `PromptProfile` (Qwen3 thinking-off, system-role probe), prefill + broadcast KV, chunked batched passes (`_score_rows`), constrained-path trie scoring, multi count row + reconciliation (`COUNT_MARGIN_MIN`), constrained MAP (`_constrained_map`), selective parent-conditioned second pass (`_selective_second_pass`), near-tie batch=1 rescore, multi calibration (`_load_calibration`), result dict. |
| [`jevmlx/setcons.py`](jevmlx/setcons.py) | Hard set-constraint selection for multi fields: exact DP over group components (mutually_exclusive, at_most_one, at_most_k, at_least_one, exact_k) plus implies propagation; score-maximizing feasible set. |
| [`jevmlx/constraints.py`](jevmlx/constraints.py) | Case-level constraint checking (implies/requires_parent, excludes, exclusivity): single source of truth shared by the engine's MAP and `evalmetrics`' violation rate. |
| [`jevmlx/parity.py`](jevmlx/parity.py) | Scoring parity (batch=1 vs batched vs chunked) over the bundled presets — ONE implementation shared by the slow parity test and the bench's `parity.json` producer. |
| [`jevmlx/api.py`](jevmlx/api.py) | Public API: `decide`, `decide_many`, `Decision`/`FieldResult`, Pydantic → schema, NONE_OF_ABOVE + abstention handling, unit-split margins. |
| [`jevmlx/cli.py`](jevmlx/cli.py) | Subcommands: decide, serve, validate, eval, report, calibrate, bench, doctor. |
| [`jevmlx/serve.py`](jevmlx/serve.py) | HTTP server (`POST /decide`), one serial worker on the single Metal GPU. |
| [`jevmlx/lint.py`](jevmlx/lint.py) | `lint_schema`: collision / duplicate / empty-choice findings from a compiled plan. |
| [`jevmlx/calibrate.py`](jevmlx/calibrate.py) | `fit_temperature` by NLL over labeled JSONL and `fit_logistic` (pooled multi calibration); ECE before/after; one JSON output for both. |
| [`jevmlx/evalrun.py`](jevmlx/evalrun.py) | Eval tracks (`parallel_decide_fn`, `naive_local_decide_fn`); `run_eval` writes `predictions.jsonl` + `run.json`. |
| [`jevmlx/evalmetrics.py`](jevmlx/evalmetrics.py) | Offline metrics from prediction lines (accuracy, calibration, bias, agreement, constraint_violation_rate). |
| [`jevmlx/evalreport.py`](jevmlx/evalreport.py) | `report.json` + markdown rendering of a run. |
| [`jevmlx/baseline.py`](jevmlx/baseline.py) | OpenAI-compatible chat client used by the API track. |
| [`jevmlx/openai_slots.py`](jevmlx/openai_slots.py) | OpenAI-compatible slot backend: one request per option, top-k logprobs with an explicit floor and a `truncated` flag. |
| [`jevmlx/bench.py`](jevmlx/bench.py) | Dataset × scorer × track matrix runner writing results directories; writes `parity.json` per model folder right after the engine load. |
| [`jevmlx/log.py`](jevmlx/log.py) | Logging configuration (`-v`, `JEVMLX_LOG=json`). |
| [`benchmarks/to_jsonl.py`](benchmarks/to_jsonl.py) | Bundled `cases.json` → eval JSONL (+ lock). |
| [`benchmarks/typesafe/fetch.py`](benchmarks/typesafe/fetch.py) | TypeSafe public pages → eval JSONL (+ lock, `benchmark_only`). |
| [`benchmarks/perturb.py`](benchmarks/perturb.py) | Deterministic label-preserving context perturbations. |
| [`benchmarks/synthetic.py`](benchmarks/synthetic.py) | Synthetic labeled case generator. |
| [`benchmarks/invariance.py`](benchmarks/invariance.py) | Irrelevant-field invariance benchmark (an invariance gate). |
| [`benchmarks/check_results.py`](benchmarks/check_results.py) | Results-folder audit: recomputes metrics from `predictions.jsonl`, verifies committed `report.json` numbers, and `--check-readme` compares the README leaderboard block against a rebuilt table. |
| [`benchmarks/summarize_results.py`](benchmarks/summarize_results.py) | Results directories → `SUMMARY.md` (marks `parity_failed` rows for models whose parity check failed). |
| [`benchmarks/leaderboard.py`](benchmarks/leaderboard.py) | README leaderboard table (TypeSafe agreement + local rows). |
| [`benchmarks/compat.py`](benchmarks/compat.py) | Model compatibility table (latency/memory) generator. |
| [`benchmarks/naive_vs_parallel.py`](benchmarks/naive_vs_parallel.py) | Quick parallel-vs-naive side-by-side comparison. |

## Data flow

```
schema (Pydantic or JSON)
  │  StructuredSchema compile (slot / labels / multi plans, per tokenizer)
  │    - codebook search per scalar field (_search_codebook: greedy over
  │      A-Z, digits, 2-char pools; lexicographic objective on the final set)
  │    - multi fields: '<field>/<code>' option rows + '<field>#count' row
  │    - set constraints validated at construction (SchemaCompileError on
  │      contradictions: cycles, implies×exclusivity, k bounds)
  ▼
plan {lead_in_ids, fields: {shared_ids, remainders/trie, alias_map?, count?}}
  │  (identity-keyed plan cache per tokenizer; weakref-evicted)
  ▼
prompt  (PROMPT_VERSION = "jevmlx-parallel-v7", engine.py:36;
  │      system paragraph engine.py:300, user schema block + <<<CONTEXT …>>>:
  │      engine.py:1430-1434; chat template via PromptProfile)
  │  prefill ONCE (engine.py:1446-1453)  →  KV cache  →  broadcast ×rows
  │    (prior pass runs first only with prior_correction, engine.py:1365)
  ▼
batched suffix pass(es)  (_score_rows engine.py:740; chunked by the memory
  │    heuristic _rows_per_chunk engine.py:564, halve-and-retry on Metal
  │    allocation failure, width bucketing)
  │  branch-point logits at each row's decision position (gather-only eval)
  ▼
trie scoring  (score_trie: P(choice) = Π branch softmax factors, T applied
  │    once downstream; legal-mass logs per branch from the full-vocab
  │    logsumexp, trie.py:125)
  ▼
near-tie rescore  (scalar top candidates and multi Y/N pairs inside
  │    INSTABILITY_BAND (5e-2 nats) rescored at the canonical batch=1 shape,
  │    engine.py:1608-1645, 1959-1990; rescored_fields telemetry)
  ▼
multi selection  (calibrated log-odds a*(yes-no)+b > 0, else P(yes) >= 0.5;
  │    count row <field>#count reconciles the set to top-k by calibrated
  │    log-odds when its margin clears COUNT_MARGIN_MIN = 0.7 nats,
  │    engine.py:1731)
  ▼
hard set constraints  (setcons.select_constrained_set: threshold/count
  │    propose, the exact DP picks the score-maximizing feasible set;
  │    applies LAST, after count reconciliation — engine.py:1754-1770)
  ▼
constrained MAP  (case-level constraints: _constrained_map maximizes summed
  │    log scores over the joint assignment, engine.py:2085-2110)
  ▼
selective second pass  (depends_on children whose parent is confident and
  │    own margin low / MAP-changed: one conditioned batch=1 pass,
  │    _selective_second_pass engine.py:994)
  ▼
assembly  (winners → typed values via alias_map; multi = per-option Y/N
  │    codes at T=1; row codes '00','01',… map back to choices)
  ▼
result dict  {parsed_json, field_telemetry, prompt_sha256, timing split, …}
  │
  ├──► api.Decision / FieldResult        (Python)
  └──► evalrun predictions.jsonl lines   (eval) / CLI table       (decide)

bench: after the engine load, jevmlx/parity.py runs the batch=1 vs batched
vs chunked parity check over the four bundled presets and writes
<model folder>/parity.json BEFORE any eval combo (bench.py:285-303, called
at bench.py:403).
```

## Contracts

### Engine result dict — `engine.py:2164-2204`

| Key | Meaning |
|---|---|
| `elapsed_ms` | Wall clock for the decision (excludes the prior pass). |
| `prior_ms` / `prefill_ms` / `suffix_eval_ms` / `lm_head_gather_ms` / `total_ms` | The honest timing split: neutral prior pass (0.0 when `prior_correction` is off), prefill, batched suffix, decision gather+eval inside the suffix window, and everything (`total_ms == elapsed_ms + prior_ms`). |
| `second_pass_ms` / `rerun_fields` / `rerun_rows` | The `depends_on` second pass: wall time, which fields were re-decided, how many conditioned rows ran (0.0/[] when no `depends_on`). |
| `rescored_fields` | Fields whose batched result was replaced by the batch=1 canonical rescore. |
| `total_tokens_generated` | Always 0: no text is generated. |
| `peak_active_bytes` | Peak Metal active memory from `mx.get_peak_memory()`. |
| `sequential_forward_passes` | 1, or the chunk count from the memory heuristic. |
| `schema_match` | Always True (keys/enums guaranteed by construction). |
| `confidence_model` | `"slots"` or `"labels"`. |
| `prompt_sha256` / `prompt_version` | SHA-256 over the full prompt token ids; the version is read from `engine.PROMPT_VERSION` (v7) — never a literal elsewhere. |
| `probability_status` | How to read the probabilities. |
| `prior_correction` / `constraints_applied` | Whether the prior pass ran / case-level constraints were applied. |
| `reconciled_fields` | Fields whose value changed under constrained MAP. |
| `parsed_json` | `{field: {"value": …, "prob": …}}`. |
| `field_telemetry` | `{field: entry}` — see next table. |
| `num_fields` | Field count. |

### `field_telemetry` entry — `engine.py:1781-1845` (multi), `:2044-2084` (scalar), `:1862-1882` (count row)

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
| `count_choice` / `count_margin` / `reconciled_by` | multi only: the `<field>#count` row's winning bucket ('0'..'4+'), its top-2 log-score margin (nats), and which rule produced the selected set ('count' when the margin cleared `COUNT_MARGIN_MIN`, else 'per_option'). The count row itself surfaces as a separate `<field>#count` telemetry entry (scalar-shaped, `margin_nats`). |
| `set_constraints` / `set_selection` | multi only, present only when the field declares set constraints: the constraints verbatim and whether the solver changed the selection ("constraints") or they didn't bind ("per_option"). |
| `margin` | Multi only: min \|P(yes) - cut\| over the final set in probability units (engine-side name; the API exposes it as `threshold_distance`; an option forced in against its P(yes) shows 0). |
| `top_choices` | Top (choice, probability) pairs, most probable first (top 5). |
| `rows` | Rows the field consumed (0 for cardinality-1 fields). |
| `tie` / `rescored` | Scalar: whether the top-2 gap is inside `INSTABILITY_BAND` (subsumes exact-equality ties), and whether the batch=1 rescore replaced the batched result. Multi: `rescored` when any option's Y/N pair was rescored. |
| `legal_mass` | Probability the model assigned to the union of allowed continuations at the winner's branch point(s), against the full vocabulary = sum(exp(z_allowed)) / sum(exp(z_vocab)). Per-branch leakage signal — the constrained distribution can confidently pick A over B even when almost all unconstrained mass is on a reasoning token/newline/label text. Product over the winner's branch path (scalar); per-option Y/N branches (multi); count row has its own. 1.0 for cardinality-1 fields (nothing branched). Always computed. Raw, pre-prior-correction logits. |
| `legal_mass_logs` | Per-choice (scalar) / per-option (multi) natural-log legal-mass product along the branch path, keyed by the real choice/option string. Raw, T=1. Calibration feature for the abstention model. |
| `oracle_prediction` / `oracle_log_scores` | Present only under `oracle_overrides` (DAG evaluation): the field was forced to the given value and re-scored conditioned on it. |

### `FieldResult` — `api.py:69-109` (built by `_build_field_results`, `api.py:276-350`)

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

### Case-level constraints — `constraints.py`, applied at `engine.py:2085`

| Type | Shape | Meaning |
|---|---|---|
| `implies` / `requires_parent` | `{parent, child, mapping}` | child's value must be in `mapping[parent_value]` (unmeasured fields can't violate). |
| `excludes` | `{field, value, other}` | when `field == value`, `other` must be empty/falsy. |
| `exclusivity` | `{group}` | at most one member of the group may hold a value. |

`check_constraint` (constraints.py:18) is the one evaluator, shared by the
engine's MAP reconciliation and `evalmetrics.constraint_violation_rate`;
`validate_constraints` (constraints.py:62) checks shapes up front.

### Per-field set constraints — `schema.py:261` (`compile_set_constraints`), solver `setcons.py`

Schema-dict key `set_constraints` on a multi field; validated at
construction, so contradictory sets raise `SchemaCompileError` before any
model load. Types: `mutually_exclusive` / `at_most_one` / `at_most_k` (k),
`at_least_one`, `exact_k` (k), `implies` (`if_option` → `then_option`).
Compile-time contradictions: implies cycles, an implies pair inside an
at-most-one group, implies into an `exact_k=0` group, k above group size,
unknown options, self-implication. The solver selects the score-maximizing
feasible set (calibrated log-odds, else raw yes/no log-odds); non-binding
constraint sets reproduce the proposal exactly.

### `parity.json` — producer `parity.py:171` (`write_parity_json`), gate `check_results.py`

Written by the bench into each model folder right after the engine load
(`bench.py:285-303`), before any eval combo:

```json
{
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "prompt_version": "jevmlx-parallel-v7",
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
`check_scoring_parity` (parity.py:73). A model without a passing
`parity.json` gets `parity_failed` rows in SUMMARY.md
(`summarize_results.py:158`, which also gates a MISSING parity.json) and
cannot enter the README compat table.

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
| `split` | `train`/`holdout` (TypeSafe: deterministic sha1-of-id rule, fetch.py:63; synthetic: every fifth case). |
| `meta` | Provenance: consensus distributions, ambiguity, per-model answers. |

### `predictions.jsonl` line — `evalrun.py:59-83` (`PREDICTION_LINE_KEYS`), written at `:504`

Frozen contract; `check_results.py` imports this list instead of retyping it.

| Key | Meaning |
|---|---|
| `run_id`, `case_id`, `group_id`, `source`, `workflow` | Provenance of the decision. |
| `field`, `type`, `track`, `model`, `permutation` | What was decided, by what. |
| `label`, `prediction`, `valid`, `correct` | Scored comparison (correct=None when unlabeled). |
| `log_scores`, `probability`, `per_option` | Distribution evidence (multi uses per_option). |
| `latency_ms`, `rows`, `passes` | Engine timing telemetry. |
| `error`, `salvage_prediction` | Failure provenance. |
| `perturbation`, `consensus` | Optional; present only with `carry_perturbation` / `carry_consensus`. |

### `run.json` — `evalrun.py:334, 462-502`

`run_id`, `environment`, `config` (model, temperature 1.0, track,
dataset_path, permutations, split, model_revision, quantization,
prompt_version read from the engine, tokenizer chat-template SHA-256,
compiled plan SHA-256, dataset lock SHA-256), and `counts` (cases, fields,
prediction_lines).

### `dataset.lock.json` — `typesafe/fetch.py:455-472` (`write_outputs`), `to_jsonl.py:61-75`

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
  multi, count, and cardinality-1 rows (`engine.py:1398-1424`).
- **Compile-time rejections** (`schema.py:443-495`): dots/slashes/'#' in
  field names (row-key injectivity), duplicate choices (schema.py:252),
  token-identical alias candidates (schema.py:737, count:917, labels:1024),
  strict token-prefix pairs (schema.py:741), candidates sharing no token
  prefix (schema.py:748), contradictory set constraints (schema.py:402) —
  all raise before any model load.
- **Probability semantics.** `log_scores` are constrained-path log
  probabilities at T=1; temperature is applied once to the final per-choice
  distribution (ranking-invariant); `FieldResult.calibrated` is False until
  a fitted calibrator runs. For multi fields `probability` is `None` — the
  engine does not claim a field-level probability for an option set
  (per-option decisions only).
- **One tolerance number.** `INSTABILITY_BAND` (engine.py:561, 5e-2 nats)
  is the single source of truth for batch-shape FP noise; the test suite
  re-exports it as `PARITY_ATOL` (tests/conftest.py:22) and `parity.json`
  records it as `atol`. Near-tie rescoring, the parity test, the memory probe,
  and the parity producer all read the same constant — no second
  tolerance literal anywhere.
- **One parity implementation.** The slow test and the bench's
  `parity.json` producer call the same `jevmlx/parity.py` function; recorded
  parity and tested parity cannot drift apart.
- **No dual paths.** The multi threshold knob is deleted (calibrated
  log-odds > 0 or the fixed P(yes) >= 0.5 rule); `set_constraints` is a real
  attribute read directly (no getattr fallback); `summarize` lives in
  `benchmarks/summarize_results.py` and bench calls it directly.
- **Prompt version read, never written.** `PROMPT_VERSION` lives once in
  `engine.py:36`; every consumer (result dict, prior cache key, parity
  payload) reads it.
- **No backward compatibility.** Changes replace: old paths, keys, flags and
  names are deleted with their callers and tests in the same change. No
  aliases, no shims, no deprecation periods.

## Testing

- Fast tests (`pytest -m "not slow" -q`) never touch a model: the engine is
  exercised through `tests/test_engine_fake.py` (`FakeModel` +
  `FakeTokenizer`), a fake tokenizer mapping words to crc32 ids in
  `tests/test_lint.py`, and `decide_fn` seams in `tests/test_evalrun.py`.
- Slow tests (marker `slow`, `pyproject.toml:54-56`) load
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
   codes per field and tokenizer (`schema.py:72`) and picks the set whose
   complete rows tokenize most cleanly (branch nodes, trie depth, length
   variance); a pinned index-based mapping survives only as a documented
   fallback for index-based callers.
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
6. **Chunking heuristic, not hard bounds.** Rows per chunk come from a
   working-set estimate (cache bytes + one float32 logits slab); a warning
   logs when a pass is split, and a Metal allocation failure halves the
   chunk once and still scores every row.
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
