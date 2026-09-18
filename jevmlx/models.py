"""Model aliases and resolution (leaf module — no jevmlx imports).

This module holds the model alias map, the default model, and the resolver.
It is a dependency-free leaf so that ``engine``, ``api``, ``cli``, and
``doctor`` can all import it at module level without creating a cycle
(``api`` imports ``engine``, so ``engine`` cannot import ``api``).
"""

from __future__ import annotations

#: Human-friendly aliases for common MLX instruct models. ``--model fast`` /
#: ``--model quality`` resolve to a specific Hub id; a bare alias keeps
#: demos short and makes the default model swappable in one place.
MODEL_ALIASES: dict[str, str] = {
    "fast": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "quality": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # The 1.5B stays only in tests — it's too small for production use.
    "test": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
}

#: The default model alias used by the CLI and API when ``--model`` is not
#: given. ``quality`` (7B) is the production default; ``fast`` (3B) is a
#: lighter alternative for latency-sensitive use.
DEFAULT_MODEL = "quality"


def resolve_model(model: str) -> str:
    """Resolve a model alias to a Hugging Face Hub id.

    Aliases (``fast``, ``quality``, ``test``) are case-insensitive and map to
    full Hub ids. A literal Hub id (contains ``/``) is returned as-is. This
    runs at every call, never at import — no network request is made here;
    a slow test verifies the Hub ids exist.
    """
    if "/" in model:
        return model
    key = model.lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    return model
