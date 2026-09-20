# jevmlx

[![CI](https://github.com/bnsd55/jevmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/bnsd55/jevmlx/actions/workflows/ci.yml) [![Build](https://github.com/bnsd55/jevmlx/actions/workflows/build.yml/badge.svg)](https://github.com/bnsd55/jevmlx/actions/workflows/build.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

jevmlx turns a schema of fields (booleans, enums, multi-selects) and a context string into a single batched forward pass on a local Apple Silicon model. Every allowed option for every field is scored from logits in one prefill — no text generation — and the JSON is assembled from the winners, with a probability per field.

## Quickstart

```bash
git clone https://github.com/bnsd55/jevmlx && cd jevmlx && ./setup.sh
jevmlx decide --preset support_triage --json          # one decision, CLI
jevmlx serve --model fast --port 8000                  # HTTP server
curl -s localhost:8000/decide -d '{"schema":{"x":{"type":"enum","choices":["a","b"],"description":"d"}},"context":"pick one"}'
```

Requires an Apple Silicon Mac (M1+) and Python 3.12+. First use of a model alias downloads weights (~2 GB `fast`, ~4.5 GB `quality`).

## Install

```bash
pip install git+https://github.com/bnsd55/jevmlx          # library
uv tool install git+https://github.com/bnsd55/jevmlx     # CLI only
git clone https://github.com/bnsd55/jevmlx && cd jevmlx && ./setup.sh   # dev
```

### Model aliases

| Alias | Resolves to | Use |
|---|---|---|
| `quality` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | **default** — best accuracy |
| `fast` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | lower latency |
| `test` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | tests only (too small for production) |

A full Hub id also works (`--model mlx-community/Llama-3.2-3B-Instruct-4bit`).

## Why not structured output

Structured output asks the model to write the JSON, token by token, then parses it and retries on failure. When the answer set is known and finite, that is the wrong tool. jevmlx reads one logits vector per field, applies a restricted softmax over the allowed options, and assembles the JSON itself — valid by construction, every field scored in one forward pass. When free text must be written (a summary, a rewrite), use generation; when the answer is one of N known choices, use scoring.

## Design your schema

- **Options must be mutually exclusive.** A restricted softmax puts all probability mass on the listed options. If two can both be true, split into separate boolean fields or use a multi-select.
- **Add an escape option when the list may not be exhaustive.** A restricted softmax cannot say "none of these" unless you give it one. Add an `other` or `escalate` choice, or use `allow_none_of_above=True` (adds an explicit NONE_OF_ABOVE that maps to `None`; the field must be `Optional`).
- **Use `ordered=True` for scales.** An enum declared `ordered` (schema `"ordered": True`, or Pydantic `Field(json_schema_extra={"ordered": True})`) adds ordinal telemetry — `argmax_level`, `expected_index` (Σ pᵢ·i), `expected_score_normalized` — with no extra model call. Use it for severity, priority, or any monotonic scale.
- **Set review thresholds from labeled data.** `abstain_below_margin=X` withholds a field whose `probability_margin` (top1 − top2) falls below the cut. Fit the threshold on labeled examples — not by feel — so the abstention rate matches your review capacity. `probability_margin` and `threshold_distance` (multi) are on every `FieldResult`.

## Use it from Python

```python
from typing import Literal
from pydantic import BaseModel, Field
from jevmlx import decide


class Ticket(BaseModel):
    urgent: bool
    category: Literal["BILLING", "TECHNICAL", "FEEDBACK"] = Field(
        description="What the ticket is about",
        json_schema_extra={"choice_descriptions": {"BILLING": "invoices, charges, refunds"}},
    )
    tags: list[Literal["refund", "login", "performance"]] = []


result = decide(Ticket, "Customer was charged twice and wants the duplicate refunded.")
for name, f in result.fields.items():
    print(f"{name}: {f.value} (p={f.probability})")
```

`decide(...)` returns a `Decision`: `.value` (a validated `Ticket`), `.latency_ms`, `.fields` — one `FieldResult` per field with `value`, `probability`, `alternatives` (top 3), `probability_margin`, `legal_mass` (probability mass in allowed continuations — a leakage signal when low), `ordinal` (ordered enums only), and `semantics` (how the probability was produced). See [ARCHITECTURE.md](ARCHITECTURE.md) for the full `FieldResult` contract.

Options: `decide_many(model_cls, contexts)` for batches; `constraints=[...]` for constrained MAP (implies / excludes / requires_parent); `calibration=<path>` for fitted multi-selection; `prior_correction=True` to subtract the neutral-context prior.

### One-question helpers

For a single-field decision you don't need a Pydantic model — `choose`, `judge`, and `rate` synthesize a one-field schema and return a `FieldResult` directly.

```python
from jevmlx import choose, judge, rate

# enum: pick one of N (dict = name -> description)
f = choose(context, {"supported": "evidence supports", "contradicted": "evidence refutes"})
print(f.value, f.probability, f.probability_margin)

# boolean: yes/no (probability of True is f.probability)
f = judge(passage, "Is the claim supported by the passage?")

# ordinal: rate on a scale (declaration order = scale order)
f = rate(review, {"low": "poor", "medium": "ok", "high": "great"})
print(f.ordinal.argmax_level, f.ordinal.expected_score_normalized)
```

CLI mirrors: `jevmlx choose --context ctx.txt --option name=description ...`, `jevmlx judge --context ctx.txt --question "..."`, `jevmlx rate --context ctx.txt --level name=description ...`.

## Use it from any OpenAI-compatible server

Instead of loading a model locally, jevmlx can send the same prompts to a chat server that returns logprobs (Ollama, vLLM, etc.):

```bash
jevmlx decide --backend openai --base-url http://localhost:11434/v1 --api-model llama3.2 --schema ticket.json --context ticket.txt
```

Tradeoff: one request per field (slower than one pass), and only the server's top-k logprobs are visible — options missing from that list get a floor probability and `truncated: true` in the telemetry.

## Serve over HTTP

`jevmlx serve` loads a model once and serves decisions on a local port. One Metal GPU, one serial worker, a bounded admission queue in front.

```bash
jevmlx serve --model fast --port 8000
```

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/decide` | `{"schema": {...}, "context": "...", "temperature": 1.0}` -> per-field result dict |
| `POST` | `/v1/systemone` | `{"state": str\|object, "questions": {id: {type, instructions, criteria}}}` -> `{model, answers, usage}` (maps to one schema, same queue as /decide) |
| `GET` | `/v1/models` | `{"data": [{"id": <model id>, "owned_by": "jevmlx"}]}` |
| `GET` | `/health` | Process liveness (always 200). Carries `queue_depth`, `queue_capacity`, `worker_alive` |
| `GET` | `/ready` | `503` until model load + warm-up complete AND worker alive, then `200` |

Backpressure: **429 + `Retry-After`** when the queue is full (`--queue-size`, default 16); **413** when a request exceeds `--max-rows` (default 2048) or `--max-prompt-tokens` (default 8192), before any model work. The client's `X-Request-Id` is echoed; `queue_depth` and `queue_wait_ms` ride every response.

### Bundled presets

`decide --preset <name>` loads a bundled schema + context:

| Preset | Fields | Description |
|---|---|---|
| `fintech_fraud` | 28 | Financial fraud detection, sanctions verification, autonomous containment |
| `support_triage` | 30 | Enterprise incident triage and routing (includes ordered `frustration_level` and `churn_risk`) |
| `code_security` | 28 | SAST/DAST pull-request vulnerability triage |
| `high_cardinality_255` | 4 | 255-choice customs tariff router (latency scaling demo) |
| `content_moderation` | 20 | Trust-and-safety policy enforcement (violation category, ordered severity) |
| `inbound_email` | 19 | Email routing, spam/phishing detection, ordered reply priority |

## What the numbers mean

[BENCHMARKING.md](BENCHMARKING.md) documents the eval methodology: accuracy against the TypeSafe public consensus, the majority baseline, and exact-record agreement; parity (batched vs. single-call within `PARITY_ATOL`); and the drift band (log-score and probability-margin drift between runs). The leaderboard below cites official TypeSafe accuracies; local rows are measured by contributors on the 20 public examples.

A **TypeScript client** (`@jevmlx/client`) lives in [`js/`](js/) — zero runtime deps, mirrors the server's JSON shapes exactly (types derived from fixture responses dumped by the server). Install from the GitHub path (no npm publish yet): `npm install github:bnsd55/jevmlx#main`. See [`js/README.md`](js/README.md) for the full API.

## Leaderboard

Agreement with the TypeSafe public eval consensus. Official rows are cited from TypeSafe's page; local rows are measured by contributors on the 20 public examples.

<!-- leaderboard:start -->
| Model | Source | Scorer | Machine | Accuracy | Customer service | Agent trace | Security | Invoices | Time per case | Cost per case | Cases |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **TypeSafe official (cited, retrieved 2026-09-17)** | | | | | | | | | | | |
| Jev | official (cited) | — | — | 67.8% | 76.0% | 71.6% | 61.7% | 61.8% | 0.4s | $0.0004 | — |
| GPT-5.6 Terra | official (cited) | — | — | 67.9% | — | — | — | — | 10.1s | $0.0304 | — |
| Claude Sonnet 5 | official (cited) | — | — | 67.8% | — | — | — | — | 78.1s | $0.1174 | — |
| Claude Opus 5 | official (cited) | — | — | 73.1% | — | — | — | — | 37.8s | $0.1761 | — |
| GPT-5.6 Sol | official (cited) | — | — | 74.1% | — | — | — | — | 23.3s | $0.0836 | — |
| Claude Haiku 4.5 | official (cited) | — | — | 53.6% | — | — | — | — | 12.5s | $0.0195 | — |

_Official accuracies are on TypeSafe's full private eval; ours are on the 20 public example cases, so the numbers are indicative, not the same test._
_Consensus label = the agreement of GPT-6 Astra + Claude Fable 5.1 (TypeSafe's reference)._

No local results yet — contribute one with `jevmlx bench`.
<!-- leaderboard:end -->

## Run the benchmark on your Mac

```bash
.venv/bin/jevmlx bench --model quality   # commit the results folder, open a PR
```

## CLI

| Command | What it does |
|---|---|
| `decide` | Decide a preset or schema + context, print JSON with probabilities |
| `choose` / `judge` / `rate` | One-field helpers (enum / boolean / ordinal) |
| `serve` | Serve decisions over HTTP (one Metal GPU, serial worker) |
| `validate` | Lint a schema for engine-visible problems (no model download) |
| `calibrate` | Fit a temperature + multi calibrator on labeled JSONL |
| `eval` | Run labeled cases through a track, write predictions + timing |
| `report` | Build a JSON + markdown eval report from predictions (offline) |
| `bench` | Full benchmark: all combos, parity gate, `SUMMARY.md` |
| `doctor` | Environment checks before filing an issue or a bench run |

`-v` for progress logs; `JEVMLX_LOG=json` for machine-readable logs.

## How it works

The schema compiles per tokenizer: a codebook search picks the neutral alias codes whose candidate rows tokenize most cleanly, and the prompt renders from the compiled plan. The context is fenced with a per-context nonce so no interior line can impersonate the closing fence. The rendered prompt is a tested contract — [PROMPT_PROTOCOL.md](PROMPT_PROTOCOL.md) documents it, and golden vectors (`benchmarks/golden_prompts.py --check`, run in CI) fail on any renderer drift.

The model prefills once and the KV cache is shared. One scoring row per field (extra trie rows for multi-token options), a restricted softmax over each field's allowed options, and the JSON is assembled from the winners. Chunks are sized by a measured active-memory budget; Metal allocation failures halve the chunk and retry. Temperature is applied once at the end; ties resolve deterministically. [ARCHITECTURE.md](ARCHITECTURE.md) has the full picture.

## Contributing

Code and docs: [CONTRIBUTING.md](CONTRIBUTING.md) · benchmark results: [BENCHMARKING.md](BENCHMARKING.md).

## License

Not affiliated with TypeSafe AI. MIT, see [LICENSE](LICENSE).
