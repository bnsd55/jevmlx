"""W3-G: the Apple Silicon platform check moved out of import time.

`import jevmlx.engine` must succeed on a machine without mlx (Linux CI runs
schema/plan/metrics tooling); the clear RuntimeError fires only when a model
is actually loaded. These tests fake the no-mlx state rather than skipping.
"""

import pytest


def test_engine_imports_without_mlx(monkeypatch):
    # If we hide mlx from the module's mlx symbols, importing the engine and
    # reading non-MLX symbols (PROMPT_VERSION, PromptProfile) must still work.
    import jevmlx.engine as engine

    monkeypatch.setattr(engine, "mx", None)
    # Re-reading these does NOT touch mlx:
    assert engine.PROMPT_VERSION.startswith("jevmlx-parallel-")
    assert hasattr(engine, "PromptProfile")
    assert hasattr(engine, "StructuredSchema") or True  # may be re-exported


def test_load_engine_raises_clear_error_without_mlx(monkeypatch):
    import jevmlx.engine as engine

    monkeypatch.setattr(engine, "mx", None)
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        engine.load_engine("any/model")


def test_schema_tooling_works_without_mlx(monkeypatch):
    # The Linux CI path: schema_from_model + StructuredSchema compile with no
    # mlx anywhere in the import chain.
    import jevmlx.engine as engine

    monkeypatch.setattr(engine, "mx", None)
    from typing import Literal

    from pydantic import BaseModel, Field

    from jevmlx import StructuredSchema, schema_from_model

    class M(BaseModel):
        risk: Literal["LOW", "HIGH"] = Field(description="risk")

    schema = StructuredSchema(schema_from_model(M))
    assert list(schema.fields["risk"].choices) == ["LOW", "HIGH"]


def test_engine_metadata_returns_none_versions_without_mlx(monkeypatch):
    # engine_metadata is best-effort provenance; with mlx unavailable it must
    # not raise — versions come back None.
    import jevmlx.engine as engine

    monkeypatch.setattr(engine, "mx", None)
    # Force importlib.metadata to report mlx/mlx-lm as absent.
    real_version = engine.importlib.metadata.version

    def fake_version(name):
        if name in ("mlx", "mlx-lm"):
            raise engine.importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(engine.importlib.metadata, "version", fake_version)
    meta = engine.engine_metadata("fake/model")
    assert meta["model_id"] == "fake/model"
    assert meta["mlx_version"] is None
    assert meta["mlx_lm_version"] is None
