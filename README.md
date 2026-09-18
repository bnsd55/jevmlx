# jevmlx

[![CI](https://github.com/bnsd55/jevmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/bnsd55/jevmlx/actions/workflows/ci.yml) [![Build](https://github.com/bnsd55/jevmlx/actions/workflows/build.yml/badge.svg)](https://github.com/bnsd55/jevmlx/actions/workflows/build.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Typed decisions from a local model on Apple Silicon. One batched forward pass, every field at once, with a probability per field.

Asking an LLM for JSON means parsing what it wrote, fixing what drifted, and retrying until it parses. jevmlx takes a schema of booleans, enums, and multi-selects, scores every allowed answer for every field in one forward pass, and assembles the JSON itself. Nothing is generated token by token: the output is valid by construction and every field carries a probability.

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
# library into your project
pip install git+https://github.com/bnsd55/jevmlx
# CLI only
uv tool install git+https://github.com/bnsd55/jevmlx
# from a clone (dev)
git clone https://github.com/bnsd55/jevmlx && cd jevmlx && ./setup.sh
```

Requires an Apple Silicon Mac (M1 or later) and Python 3.12+.

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
    tags: list[Literal["refund", "login", "performance"]] = Field(default_factory=list)


result = decide(Ticket, "Customer was charged twice and wants the duplicate refunded.")
for name, f in result.fields.items():
    print(f"{name}: {f.value} (p={f.probability})")
```

- `decide_many(model, contexts)` decides many texts with one model load.
- `allow_unknown=True` adds an `UNKNOWN` choice when the model is unsure (fields map to `None`).
- `alternatives` lists the other options with their probabilities.
- `multi_threshold=0.5` sets the P(yes) cut for multi-select options.
- CLI equivalent: `jevmlx decide --help`.

## Use it from any OpenAI-compatible server

Instead of loading a model on this Mac, jevmlx can send the same prompts to a chat server that returns logprobs: Ollama, oMLX, MTPLX, vLLM.

```bash
jevmlx decide --backend openai --base-url http://localhost:11434/v1 --api-model llama3.2 --schema ticket.json --context ticket.txt
```

Two tradeoffs: one request per field instead of one pass (slower), and only the server's top-k logprobs are visible, so options missing from that list get a floor probability and the field's telemetry flags `truncated: true`.

## Leaderboard

Agreement with the TypeSafe public eval consensus (GPT-6 Astra + Claude Fable 5.1). Official rows are cited from TypeSafe's page; local rows are measured by contributors on the 20 public examples.

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
| **jevmlx, local (measured)** | | | | | | | | | | | |
| mlx-community/Llama-3.2-1B-Instruct-4bit | local | labels | m5max-128gb | 28.9% | 18.6% | 37.5% | 36.5% | 29.7% | 1.5s | $0 (local) | 44 |
| mlx-community/Llama-3.2-1B-Instruct-4bit | local | slots | m5max-128gb | 32.8% | 25.3% | 39.2% | 32.3% | 33.5% | 1.5s | $0 (local) | 44 |
| mlx-community/Qwen2.5-0.5B-Instruct-4bit | local | labels | m5max-128gb | 31.8% | 17.1% | 38.3% | 37.5% | 33.1% | 0.8s | $0 (local) | 44 |
| mlx-community/Qwen2.5-0.5B-Instruct-4bit | local | slots | m5max-128gb | 31.6% | 34.4% | 40.0% | 37.8% | 31.1% | 0.7s | $0 (local) | 44 |
| mlx-community/Qwen2.5-1.5B-Instruct-4bit | local | labels | m5max-128gb | 36.5% | 57.4% | 42.5% | 40.6% | 34.3% | 2.3s | $0 (local) | 44 |
| mlx-community/Qwen2.5-1.5B-Instruct-4bit | local | slots | m5max-128gb | 25.5% | 41.6% | 43.3% | 46.2% | 23.3% | 2.3s | $0 (local) | 44 |

_Official accuracies are on TypeSafe's full private eval; ours are on the 20 public example cases, so the numbers are indicative, not the same test._
_Consensus label = the agreement of GPT-6 Astra + Claude Fable 5.1 (TypeSafe's reference)._

<!-- leaderboard:end -->

## Run the benchmark on your Mac

```bash
git clone https://github.com/bnsd55/jevmlx && cd jevmlx
./setup.sh
.venv/bin/jevmlx bench --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
# commit the results folder and open a PR
```

[BENCHMARKING.md](BENCHMARKING.md) has the model list, what the command does, and the PR checklist.

## CLI

| Command | What it does |
|---|---|
| `decide` | Decide a preset or schema + context, print the JSON with probabilities |
| `serve` | Serve decisions over HTTP (`POST /decide`, one Metal GPU, serial) |
| `validate` | Lint a schema for engine-visible problems (no model download) |
| `calibrate` | Fit a temperature on labeled JSONL cases and report ECE |
| `eval` | Run labeled cases through a decision track; writes predictions + manifest |
| `report` | Build a JSON + markdown eval report from `predictions.jsonl` (offline) |
| `bench` | Full benchmark: all (track, scorer, dataset) combos, one `SUMMARY.md` |
| `doctor` | Environment checks: run before filing an issue or a bench run |

`-v` for progress logs; `JEVMLX_LOG=json` for machine-readable logs.

## How it works

The prompt lists every field and its allowed options as neutral aliases. The model prefills once and the KV cache is shared. One scoring row per field (extra trie rows for multi-token options), a restricted softmax over each field's allowed options, and the JSON is assembled from the winners — with a probability per field. Temperature is applied once at the end; ties resolve deterministically. [ARCHITECTURE.md](ARCHITECTURE.md) has the full picture.

## Contributing

Code and docs: [CONTRIBUTING.md](CONTRIBUTING.md). Benchmark results: [BENCHMARKING.md](BENCHMARKING.md).

## Credits and license

jevmlx started from [rorshopping/jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop), which reproduced the parallel constrained decoding technique on a laptop.

The engine descends from [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), an MLX demo of parallel constrained decoding with one 1.5B checkpoint; jevmlx is the maintained, generic version: any MLX instruct model or OpenAI-compatible server, choices in the prompt, multi-select, UNKNOWN, per-field probability, an eval harness.

Not affiliated with TypeSafe.

MIT — see [LICENSE](LICENSE); third-party credits in [NOTICE](NOTICE).
