# jevmlx

[![CI](https://github.com/bnsd55/jevmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/bnsd55/jevmlx/actions/workflows/ci.yml) [![Build](https://github.com/bnsd55/jevmlx/actions/workflows/build.yml/badge.svg)](https://github.com/bnsd55/jevmlx/actions/workflows/build.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Typed decisions from a local model on Apple Silicon. One batched forward pass, every field at once, with a probability per field.

Asking an LLM for JSON means parsing what it wrote and retrying until it parses. jevmlx takes a schema of booleans, enums, and multi-selects, scores every allowed answer for every field in one forward pass, and assembles the JSON itself — valid by construction, every field with a probability.

```bash
pip install git+https://github.com/bnsd55/jevmlx
jevmlx decide --preset fintech_fraud --json
```

```json
{
  "is_fraudulent": {"value": true, "prob": 0.6077},
  "risk_tier": {"value": "HIGH", "prob": 0.8392},
  "recommended_action": {"value": "CHALLENGE_2FA", "prob": 0.8891}
}
```

## Install

```bash
pip install git+https://github.com/bnsd55/jevmlx   # library
uv tool install git+https://github.com/bnsd55/jevmlx   # CLI only
git clone https://github.com/bnsd55/jevmlx && cd jevmlx && ./setup.sh   # dev
```

Requires an Apple Silicon Mac (M1 or later) and Python 3.12+.

### Model aliases

| Alias | Resolves to | Use |
|---|---|---|
| `quality` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | **default** — best accuracy |
| `fast` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | lower latency |
| `test` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | tests only (too small for production) |

```bash
jevmlx decide --model quality --schema ticket.json --context ticket.txt   # or --model fast
```

A full Hub id also works (`--model mlx-community/Llama-3.2-3B-Instruct-4bit`); first use downloads the weights (~4.5 GB `quality`, ~2 GB `fast`).

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

Read the result. `decide(...)` returns a `Decision`: `.value` (a validated
`Ticket`), `.latency_ms`, `.fields` — one `FieldResult` per field:

- `value`, `probability`, `alternatives` (top 3, winner first; multi: per-option
  P(yes)), `score`, `model` (`"slots"`/`"labels"`), `calibrated`, `legal_mass`
  (probability mass in allowed continuations at the branch points — a leakage
  signal when low despite a confident decision).
- `ordinal` (ordered enums only, `None` otherwise): an enum field may declare
  itself an ORDINAL scale — `Field(json_schema_extra={"ordered": True})` (or
  bare `Ordered()`); the choices' declaration order is the scale order. The
  decided value stays the winning level; ordering only adds derived
  telemetry — `argmax_level` (winning index), `expected_index` (Σ pᵢ·i), its
  `variance`, and `expected_score_normalized` in [0, 1] — computed from the
  finalized distribution (after prior correction and temperature), no extra
  model call. Eval runs on ordered fields
  also report `ordinal_mae` (mean |argmax − gold|), `ordinal_mae_expected`
  (soft, |E − gold|) and an ordinal confusion matrix.
- `semantics` (a frozen `FieldSemantics`): how THIS field's reported
  probabilities were produced — which scoring path (`score_source`:
  `batched` / `rescored_batch1` / `dependency` / `oracle`), the temperature
  actually applied (`None` for count rows and calibrated multi selections,
  whose log-odds cut ignores the caller temperature), the calibrator bundle
  id when a fitted calibrator set the selection, the prior mode
  (`off`/`neutral_v1`), and whether a constraint or a dependency wave
  overrode the raw winner. The result-level `probability_status` summarizes
  the distinct semantic groups (one clause per
  `(score_source, temperature, calibrator_id, prior_mode)` group with its
  field count) and is not authoritative for any single field.
- Margins, one per field: `log_score_margin` / `probability_margin` (scalar,
  top1-top2 gap in log/probability units), `threshold_distance` (multi, how
  close the closest yes/no call sat to the cut). Multi fields carry
  `probability=None` — only per-option decisions are claimed.
- `reason`: None, `"none_of_above"` (`allow_none_of_above=True` adds an explicit
  NONE_OF_ABOVE choice that maps to `None`; fields must be Optional), or
  `"abstain"` (`abstain_below_margin=X` withholds fields whose margin falls
  below the cut; the raw value stays on the FieldResult).
- Options: `decide_many(model_cls, contexts)` for batches (one prior pass, one merged scoring pass per context group — results match separate `decide` calls within `PARITY_ATOL`); `constraints=[...]`
  reconciles the joint answer by constrained MAP (implies / excludes /
  requires_parent / exclusivity; plus `depends_on` / `set_constraints` in the
  schema); `calibration=<path-or-dict>` (from `jevmlx calibrate --out`) selects
  multi options by fitted log-odds with the always-on count row reconciling to
  top-k above `COUNT_MARGIN_MIN` (0.7 nats); `prior_correction=True` subtracts
  the neutral-context prior; timing: `latency_ms` here, the full split
  (prior / prefill / plan compile / cache broadcast / suffix eval / lm-head
  gather / second pass) plus `peak_active_bytes`/`peak_incremental_bytes` and
  `failed_attempts` on the engine result, per-combo `timing.json` in bench
  output.

## Use it from any OpenAI-compatible server

Instead of loading a model on this Mac, jevmlx can send the same prompts to a chat server that returns logprobs: Ollama, oMLX, MTPLX, vLLM.

```bash
jevmlx decide --backend openai --base-url http://localhost:11434/v1 --api-model llama3.2 --schema ticket.json --context ticket.txt
```

Two tradeoffs: one request per field (slower than one pass), and only the server's top-k logprobs are visible — options missing from that list get a floor probability and the telemetry flags `truncated: true`.

## Serve over HTTP

`jevmlx serve` loads a model once and exposes `POST /decide`, `GET /health`, and `GET /ready` on a local port. One Metal GPU, one serial worker, a bounded admission queue in front.

```bash
jevmlx serve --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8000
```

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/decide` | `{"schema": {...}, "context": "...", "temperature": 1.0}` -> the per-field result dict |
| `POST` | `/v1/systemone` | `{"state": str|object, "questions": {id: {type, instructions, criteria}}}` -> `{model, answers: {id: ChoiceAnswer\|NoulAnswer\|ScoreAnswer}, usage}` (maps questions to one schema, same queue/backpressure as /decide) |
| `GET` | `/v1/models` | `{"data": [{"id": <resolved model id>, "owned_by": "jevmlx"}]}` (advertises only our model) |
| `GET` | `/health` | Process liveness (always 200). Carries `queue_depth`, `queue_capacity`, `worker_alive`, `requests_served` |
| `GET` | `/ready` | `503` until model load + warm-up complete AND the worker is alive, then `200` |

**Backpressure + admission limits** (W6-B7):

- **429 + `Retry-After`** when the admission queue is full (not 529). `--queue-size` (default 16). The server is a `ThreadingHTTPServer`; one thread per connection, but a single serial worker processes GPU work — two decide calls never overlap.
- **413** when a request exceeds a hard limit, before any model work: `--max-rows` (expanded scoring rows, default 2048; O(schema) arithmetic on the raw dict, not `_build_schema_rows`), `--max-prompt-tokens` (default 8192). The projected-memory limit was dropped — the engine already chunks rows to its measured width-bin budget, so a whole-request projection describes memory the engine never allocates.
- **Request id**: the client's `X-Request-Id` is echoed in the response header + body; if absent one is generated.
- **Queue telemetry**: `queue_depth` and `queue_wait_ms` ride every `/decide` response; `queue_depth` + `queue_capacity` ride `/health`.
- **Worker resilience**: an exception in the decide worker is a 500 to that request; the worker catches it and continues (it does not die). `/ready` goes 503 if the worker dies.
- **Ready vs live**: the port binds BEFORE warm-up, so `/ready` is reachable during warm-up (a 503 is a real response, not a connection refusal).

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

_Official accuracies: TypeSafe's full private eval; ours: the 20 public examples — indicative, not the same test. Consensus = GPT-6 Astra + Claude Fable 5.1._

No local results yet — contribute one with `jevmlx bench`.
<!-- leaderboard:end -->

## Run the benchmark on your Mac

```bash
git clone https://github.com/bnsd55/jevmlx && cd jevmlx && ./setup.sh
.venv/bin/jevmlx bench --model quality   # commit the results folder, open a PR
```
[BENCHMARKING.md](BENCHMARKING.md) has the model list, what the command does, and the PR checklist.

## CLI

| Command | What it does |
|---|---|
| `decide` | Decide a preset or schema + context, print the JSON with probabilities; `--model fast/quality/…`, `--scoring slots/labels`, `--constraints`, `--calibration`, `--prior-correction`, `--backend openai` |
| `serve` | Serve decisions over HTTP (`POST /decide`, one Metal GPU, serial) |
| `validate` | Lint a schema for engine-visible problems (no model download) |
| `calibrate` | Fit a temperature + pooled multi calibrator on labeled JSONL, report ECE, `--out` writes what `decide --calibration` reads |
| `eval` | Run labeled cases through a track (`parallel`/`naive_local`/`api_baseline`/`openai_slots`), with optional permutations; writes `predictions.jsonl` + `run.json` + per-combo `timing.json` |
| `report` | Build a JSON + markdown eval report from `predictions.jsonl` (offline) |
| `bench` | Full benchmark: all (track, scorer, dataset) combos for one or more models, parity gate + one `SUMMARY.md` |
| `doctor` | Environment checks (platform, versions, venv/conda + subprocess hang, editable-install checkout, memory, power, Metal, model cache, network): run before filing an issue or a bench run |

`-v` for progress logs; `JEVMLX_LOG=json` for machine-readable logs.

### Bundled presets

`decide --preset <name>` loads a bundled schema + context:

| Preset | Fields | Description |
|---|---|---|
| `fintech_fraud` | 28 | Real-time financial fraud detection, sanctions verification, and autonomous containment |
| `support_triage` | 30 | Enterprise incident triage and routing (includes ordered `frustration_level` and `churn_risk`) |
| `code_security` | 28 | SAST/DAST pull-request vulnerability triage |
| `high_cardinality_255` | 4 | 255-choice customs tariff router (latency scaling demo) |
| `content_moderation` | 20 | Trust-and-safety policy enforcement (violation category, ordered severity, human-review flag) |
| `inbound_email` | 19 | Email routing, spam/phishing detection, ordered reply priority |

## How it works

The schema compiles per tokenizer: a bounded codebook search picks the neutral alias codes whose candidate rows tokenize most cleanly, and the prompt renders FROM the compiled plan — the model is taught exactly the protocol the scorer judges. The context is fenced with a per-context nonce, so no interior line can impersonate the closing fence. The rendered prompt is a tested contract: [PROMPT_PROTOCOL.md](PROMPT_PROTOCOL.md) documents it, and committed golden vectors (`benchmarks/golden_prompts.py --check`, run in CI) fail on any renderer drift.

The model prefills once and the KV cache is shared. One scoring row per field (extra trie rows for multi-token options), a restricted softmax over each field's allowed options, and the JSON is assembled from the winners — with a probability per field. Chunks are sized by a measured active-memory budget (the B=1/B=2 tiling slope is probed at engine load; Metal allocation failures halve the chunk and retry, counted in `failed_attempts`). Temperature is applied once at the end; ties resolve deterministically. Batched `decide_many` runs one prior pass and one merged scoring pass per context group. [ARCHITECTURE.md](ARCHITECTURE.md) has the full picture.

## Contributing

Code and docs: [CONTRIBUTING.md](CONTRIBUTING.md) · benchmark results: [BENCHMARKING.md](BENCHMARKING.md).

## Credits and license

jevmlx started from [rorshopping/jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop) (parallel constrained decoding on a laptop) and descends from [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), an MLX demo of the technique. jevmlx is the maintained, generic version — any MLX instruct model or OpenAI-compatible server, multi-select, none-of-the-above and abstention, per-field probability, an eval harness.

Not affiliated with TypeSafe. MIT — see [LICENSE](LICENSE); third-party credits in [NOTICE](NOTICE).
