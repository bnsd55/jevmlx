#!/usr/bin/env bash
# One-command setup: Python env + MLX + the jevmlx package (editable install).
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -m)" != "arm64" ] || [ "$(uname -s)" != "Darwin" ]; then
  echo "This setup targets Apple Silicon Macs (MLX)."
  exit 1
fi

command -v uv >/dev/null 2>&1 || { echo "uv not found. Install it first: brew install uv"; exit 1; }

echo "==> Creating Python 3.12 environment (.venv)"
uv venv --python 3.12 .venv

echo "==> Installing jevmlx (editable; pulls mlx-lm)"
uv pip install --python .venv/bin/python -e .

echo
echo "Done. Next steps:"
echo "  .venv/bin/jevmlx decide --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --preset fintech_fraud"
