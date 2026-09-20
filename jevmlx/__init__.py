"""Parallel Constrained Structured Generation Engine."""

import importlib.metadata

__version__ = importlib.metadata.version("jevmlx")

# Schema/metrics modules import cleanly without MLX (engine.py is the only
# module that imports mlx). The engine-backed symbols below are exposed lazily
# so that `import jevmlx` (and `from jevmlx.evalmetrics import ...`) work on a
# machine without MLX installed — e.g. the ubuntu-latest results-check CI,
# which only needs the metrics path. Importing decide/load_engine/... still
# requires MLX and raises ModuleNotFoundError on a non-Apple-Silicon box.

from jevmlx.schema import FieldDefinition, StructuredSchema  # noqa: E402

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SCORING",
    "Decision",
    "FieldDefinition",
    "FieldResult",
    "FieldSemantics",
    "OrdinalFieldRecord",
    "Ordered",
    "StructuredSchema",
    "clear_engine_cache",
    "choose",
    "decide",
    "decide_many",
    "judge",
    "load_engine",
    "rate",
    "run_parallel_generation",
    "run_naive_generation",
    "schema_from_model",
]


def __getattr__(name: str):
    # Lazy engine-backed symbols: api.decide/decide_many/Decision/... and the
    # engine functions. Only imported when actually accessed.
    if name in {
        "DEFAULT_MODEL",
        "DEFAULT_SCORING",
        "Decision",
        "FieldResult",
        "FieldSemantics",
        "OrdinalFieldRecord",
        "Ordered",
        "choose",
        "decide",
        "decide_many",
        "judge",
        "rate",
        "schema_from_model",
    }:
        from jevmlx import api

        return getattr(api, name)
    if name in {
        "clear_engine_cache",
        "load_engine",
        "run_parallel_generation",
        "run_naive_generation",
    }:
        from jevmlx import engine

        return getattr(engine, name)
    raise AttributeError(f"module 'jevmlx' has no attribute {name!r}")
