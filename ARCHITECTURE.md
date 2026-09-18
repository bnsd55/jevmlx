# ARCHITECTURE.md

For people who want to read or change the code. Line references cite
`file:line` at the time of writing (branch point `334bdaf`); trust the code
over this document when they drift.

## Module map

| Module | Role |
|---|---|
| [`jevmlx/schema.py`](jevmlx/schema.py) | Schema model (`StructuredSchema`), slot/labels plan compilation, per-tokenizer plan cache, compile-time rejections. |
| [`jevmlx/trie.py`](jevmlx/trie.py) | Branch-point trie over candidate token remainders; softmax/logsumexp helpers; `score_trie`. |
| [`jevmlx/engine.py`](jevmlx/engine.py) | Model load (lru_cached), prompt v5, PromptProfile (Qwen3 thinking-off, system-role probe), prefill + broadcast KV, chunked batched passes, trie scoring, result dict. |
| [`jevmlx/api.py`](jevmlx/api.py) | Public API: `decide`, `decide_many`, `Decision`/`FieldResult`, Pydantic → schema, NONE_OF_ABOVE + abstention handling. |
| [`jevmlx/cli.py`](jevmlx/cli.py) | Subcommands: decide, serve, validate, eval, report, calibrate, bench. |
| [`jevmlx/serve.py`](jevmlx/serve.py) | HTTP server, one serial worker on the single Metal GPU. |
| [`jevmlx/lint.py`](jevmlx/lint.py) | `lint_schema`: collision / duplicate / empty-choice findings from a compiled plan. |
| [`jevmlx/calibrate.py`](jevmlx/calibrate.py) | Temperature fitting by NLL over labeled JSONL; ECE before/after. |
| [`jevmlx/evalrun.py`](jevmlx/evalrun.py) | Eval tracks (parallel, naive_local, api_baseline); writes `predictions.jsonl` + `run.json`. |
| [`jevmlx/evalmetrics.py`](jevmlx/evalmetrics.py) | Offline metrics from prediction lines (accuracy, calibration, bias, agreement). |
| [`jevmlx/evalreport.py`](jevmlx/evalreport.py) | `report.json` + markdown rendering of a run. |
| [`jevmlx/baseline.py`](jevmlx/baseline.py) | OpenAI-compatible chat client used by the `naive_local`/API tracks. |
| [`jevmlx/bench.py`](jevmlx/bench.py) | Dataset × scorer × track matrix runner writing results directories. |
| [`jevmlx/log.py`](jevmlx/log.py) | Logging configuration (`-v`, `JEVMLX_LOG=json`). |
| [`jevmlx/openai_slots.py`](jevmlx/openai_slots.py) | OpenAI-compatible slot backend: one request per option, top-k logprobs with an explicit floor and a `truncated` flag. |
| [`benchmarks/to_jsonl.py`](benchmarks/to_jsonl.py) | Bundled `cases.json` → eval JSONL (+ lock). |
| [`benchmarks/typesafe/fetch.py`](benchmarks/typesafe/fetch.py) | TypeSafe public pages → eval JSONL (+ lock, `benchmark_only`). |
| [`benchmarks/perturb.py`](benchmarks/perturb.py) | Deterministic label-preserving context perturbations. |
| [`benchmarks/synthetic.py`](benchmarks/synthetic.py) | Synthetic labeled case generator. |
| [`benchmarks/summarize_results.py`](benchmarks/summarize_results.py) | Results directories → `SUMMARY.md` with PR instructions. |
| [`benchmarks/compat.py`](benchmarks/compat.py) | Model compatibility table (latency/memory) generator. |
| [`benchmarks/naive_vs_parallel.py`](benchmarks/naive_vs_parallel.py) | Quick parallel-vs-naive side-by-side comparison. |

## Data flow

```
schema (Pydantic or JSON)
  │  StructuredSchema.compile_slot_plan / compile_labels_plan   (per tokenizer)
  ▼
plan {lead_in_ids, fields: {shared_ids, remainders, alias_map?}}
  │  rows = lead_in + shared_ids + branch path   (one row per branch point)
  ▼
prompt v5  (engine.py:291 system, :428 user: schema block + <<<CONTEXT …>>>)
  │  prefill ONCE  →  KV cache  →  broadcast ×rows
  ▼
batched suffix pass(es)  (chunked by the memory heuristic, engine.py:506-514)
  │  branch-point logits at each row's decision position
  ▼
trie scoring  (trie.py: P(choice) = Π branch softmax factors; T applied once)
  ▼
assembly  (winners → typed values via alias_map; multi = per-option Y/N codes at T=1)
  ▼
result dict  {parsed_json, field_telemetry, prompt_sha256, …}
  │
  ├──► api.Decision / FieldResult        (Python)
  └──► evalrun predictions.jsonl lines   (eval) / CLI table       (decide)
```

## Contracts

### Engine result dict — `engine.py:469-488`

| Key | Meaning |
|---|---|
| `elapsed_ms` / `prefill_ms` / `suffix_eval_ms` | Wall-clock breakdown, rounded to 2 decimals. |
| `total_tokens_generated` | Always 0: no text is generated. |
| `sequential_forward_passes` | 1, or the chunk count from the memory heuristic. |
| `schema_match` | Always True (keys/enums guaranteed by construction). |
| `confidence_model` | `"slots"` or `"labels"`. |
| `prompt_sha256` / `prompt_version` | SHA-256 over the full prompt token ids; `jevmlx-parallel-v5`. |
| `probability_status` | How to read the probabilities. |
| `parsed_json` | `{field: {"value": …, "prob": …}}`. |
| `field_telemetry` | `{field: entry}` — see next table. |
| `num_fields` | Field count. |

### `field_telemetry` entry — `engine.py:619-627`

| Key | Meaning |
|---|---|
| `value` | Decided value (str / bool / list[str]). |
| `type` | `boolean`, `enum`, or `multi`. |
| `probability` | P of the winner; multi: `None` — no field-level probability is claimed (per-option decisions only). |
| `cardinality` | Number of choices. |
| `log_scores` | Constrained-path log P per choice at T=1, keyed by real choice string. Absent for multi. |
| `per_option` | multi only: independent per-option P(yes). |
| `option_logit_pairs` | multi only: raw [yes, no] logits per option at T=1 (what the prior cache stores). |
| `top_choices` | Top (choice, probability) pairs, most probable first (top 5). |
| `rows` | Rows the field consumed (0 for cardinality-1 fields). |
| `margin` | Multi only: min |P(yes) - threshold| (engine-side name; the API exposes it as `threshold_distance`). |

### `FieldResult` — `api.py` (built by `_build_field_results`)

| Field | Meaning |
|---|---|
| `value` | Decided value (engine-side). |
| `score` | Log P of the winner; 0.0 for multi (no field-level log score). |
| `log_score_margin` | Scalar only: top1-top2 log score at T=1. None for multi. |
| `probability_margin` | Scalar only: top1-top2 probability (post-temperature). None for multi. |
| `threshold_distance` | Multi only: min |P(yes) - threshold|. None for scalar. |
| `probability` | P of the winner; None for multi. |
| `calibrated` | False until a fitted calibrator is applied (engine never calibrates). |
| `model` | `"slots"` or `"labels"`. |
| `alternatives` | Top 3 (choice, probability) pairs; multi: per-option (option, P(yes)) sorted desc. |
| `reason` | None, `"none_of_above"` (caller opted in, model picked the explicit opt-out → None), or `"abstain"` (margin below `abstain_below_margin`; value withheld, raw kept for provenance). The single source of truth — no separate `abstain` flag. |

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
| `split` | `train`/`holdout` (deterministic sha1 rule). |
| `meta` | Provenance: consensus distributions, ambiguity, per-model answers. |

### `predictions.jsonl` line — `evalrun.py:292-335`

| Key | Meaning |
|---|---|
| `run_id`, `case_id`, `group_id`, `source`, `workflow` | Provenance of the decision. |
| `field`, `type`, `track`, `model`, `permutation` | What was decided, by what. |
| `label`, `prediction`, `valid`, `correct` | Scored comparison (correct=None when unlabeled). |
| `log_scores`, `probability`, `per_option` | Distribution evidence (multi uses per_option). |
| `latency_ms`, `rows`, `passes` | Engine timing telemetry. |
| `error`, `salvage_prediction` | Failure provenance. |
| `perturbation`, `consensus` | Optional; present only with `carry_perturbation` / `carry_consensus`. |

### `run.json` — `evalrun.py:349-381`

`run_id`, `created_at`, `config` (model, temperature 1.0, track, permutations,
split, model_revision, quantization, prompt_version, tokenizer chat-template
SHA-256, compiled plan SHA-256, dataset lock SHA-256), and
`counts` (cases, prediction_lines).

### `dataset.lock.json` — `typesafe/fetch.py:455-475`, `to_jsonl.py` `main`

`sources` [{url, sha256, fetched_at, etag}], `parser_version`, `counts`,
`cases_sha256` (sha256 of the JSONL written next to it).

## Invariants

- **Token-aligned candidates.** The full candidate text (field key + value)
  is tokenized as one string, and the row holds `lead_in + shared + branch`
  token ids. Splitting candidate text anywhere but at token boundaries would
  score tokens the model never sees.
- **One lead-in for every row.** Plans store suffixes *without* the schema
  lead-in; the engine prepends it exactly once per row, one rule for scalar,
  multi, and cardinality-1 fields (`engine.py:407-425`).
- **Compile-time rejections** (`schema.py:437-510`): token-identical choices,
  strict token-prefix pairs, candidates sharing no token prefix, duplicate
  values, dots in field names — all raise `SchemaCompileError` before any
  model load.
- **Probability semantics.** `log_scores` are constrained-path log
  probabilities at T=1; temperature is applied once to the final per-choice
  distribution (ranking-invariant); `FieldResult.calibrated` is False until
  a fitted calibrator runs. For multi fields `probability` is `None` — the
  engine does not claim a field-level probability for an option set
  (per-option decisions only).
- **No backward compatibility.** Changes replace: old paths, keys, flags and
  names are deleted with their callers and tests in the same change. No
  aliases, no shims, no deprecation periods.

## Testing

- Fast tests (`pytest -m "not slow" -q`) never touch a model: the engine is
  exercised through `tests/test_engine_fake.py` (`FakeModel` +
  `FakeTokenizer`), a fake tokenizer mapping words to crc32 ids in
  `tests/test_lint.py`, and `decide_fn` seams in `tests/test_evalrun.py`.
- Slow tests (marker `slow`, `pyproject.toml:43`) load
  `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Run one locally with
  `uv run pytest tests/test_engine.py -m slow -k slots -q`.
- Eval/benchmark code is tested against synthetic viewer payloads and
  fixture cases; nothing downloads from the network (see
  `tests/test_typesafe_fetch.py`).

## Design decisions

1. **Trie over first tokens.** Scoring whole choices as branch paths gives a
   proper distribution and lets shared token prefixes be paid for once.
2. **Slots default over labels.** Neutral aliases decouple the scored
   vocabulary from choice text — no tokenization collisions by construction;
   labels mode stays for schemas where spelling is the signal.
3. **JSON rows kept.** The decision row stays `'<field>": '` so assembly is
   plain JSON parsing, not a positional protocol.
4. **Multi as one-vs-rest.** Each option is an independent true/false
   decision at its own divergence point; subset probability semantics would
   not be additive.
5. **Chunking heuristic, not hard bounds.** Rows per chunk come from a
   working-set estimate (cache bytes + one float32 logits slab); a warning
   logs when a pass is split.
6. **Identity-keyed plan cache.** Compiled plans are cached per tokenizer
   identity (weakref, evicted on death) so repeated decisions skip
   recompilation without leaking tokenizers.
7. **No letters mode.** Slot scoring superseded it entirely; the old mode
   was deleted, not deprecated.
8. **Provenance fields everywhere.** `prompt_sha256`, `prompt_version`,
   `dataset.lock.json`, and `run.json` exist so any number in a report can
   be traced to the exact prompt and data that produced it.
