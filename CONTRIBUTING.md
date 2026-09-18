# Contributing

## Setup

Apple Silicon Mac (M1+), macOS 13+, Python 3.12.

```bash
git clone https://github.com/bnsd55/jevmlx && cd jevmlx
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
uv run pytest -m "not slow" -q      # fast suite: no model download
uv run ruff check --fix . && uv run ruff format .
```

Slow tests load a real 0.5B model (`pytest -m slow`); run them once before
touching the engine. Code layout and module responsibilities:
[ARCHITECTURE.md](ARCHITECTURE.md).

## Ground rules

- Branch off `main`, small focused PRs, one logical change per PR.
- Apple Silicon only; the engine fails fast on other platforms.
- No new dependencies without an issue describing why.
- **No compatibility shims.** When behavior changes, delete the old path,
  keys, flags, and names together with their callers and tests. No aliases,
  no fallbacks, no deprecation periods.
- Run `ruff check --fix . && ruff format .` and the fast tests before pushing.

## Pull requests

- Fill the PR template; keep the description to what and why.
- Update tests in the same change as the code they cover.
- Never commit datasets, model weights, or files over 5 MB.

## Benchmark results

To contribute accuracy or latency numbers from your own Mac, follow
[BENCHMARKING.md](BENCHMARKING.md) — one results directory per
(model × scorer × dataset), raw predictions included.

## Conduct

Be civil and assume good faith; this project follows the
[Contributor Covenant](https://www.contributor-covenant.org/) spirit — see
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md) for
how to report conduct or security issues.
