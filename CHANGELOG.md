# Changelog

## Unreleased

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
- Multi-select fields decided as per-option yes/no rows with an exposed threshold (`--multi-threshold`); no field-level probability claimed, a threshold margin instead.
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
