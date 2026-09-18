"""W6-1: LM-head adapters — backbone/lm_head split, standalone module.

No engine wiring: this module only describes HOW a supported mlx_lm model
splits into a backbone (token ids -> hidden states, cache-aware) and an LM
head (hidden states -> vocab logits). Engine call sites adopt it later
(W6-2+).

Ground truth is the installed mlx_lm model classes: every family computes
``Model.__call__(x) = head(backbone(x))`` — qwen2/qwen3/qwen3_moe/llama/
mixtral/phi3 call ``self.lm_head`` when untied and
``self.model.embed_tokens.as_linear`` when tied; gemma3_text always calls
its (untied) ``lm_head``; phi's head is biased; mistral3 delegates to an
inner llama/ministral3 Model. The adapters here mirror those exact
attribute paths and call orders.
"""

from __future__ import annotations

from typing import Any, Protocol

import mlx.core as mx

__all__ = [
    "LMHeadAdapter",
    "UnsupportedModelError",
    "adapter_for",
    "list_supported_model_types",
]


class UnsupportedModelError(RuntimeError):
    """The model's type has no adapter — refuse instead of guessing.

    Carries the offending ``model_type`` and the supported set so callers
    can render an actionable message.
    """

    def __init__(self, model_type: str | None, supported: list[str]):
        self.model_type = model_type
        self.supported = supported
        super().__init__(
            f"unsupported model_type {model_type!r}: no LM-head adapter; "
            f"supported: {', '.join(supported)}"
        )


class LMHeadAdapter(Protocol):
    """Split a mlx_lm model into backbone and LM head.

    ``backbone(tokens, cache)`` runs the transformer up to (excluding) the
    LM head: tokens [B, W] -> hidden [B, W, H], cache semantics identical
    to the model's own ``__call__``. ``lm_head(hidden)`` projects hidden
    states [B, H] -> logits [B, V] (any leading dims work; the head is a
    linear map). The two together must reproduce ``model(tokens, cache)``
    exactly — the equivalence is enforced by the slow 0.5B test.
    """

    def backbone(self, tokens: mx.array, cache: Any = None) -> mx.array: ...  # noqa: E704
    def lm_head(self, hidden: mx.array) -> mx.array: ...  # noqa: E704


class _StandardAdapter:
    """The dominant mlx_lm layout: ``model.model`` backbone + tied-or-not head.

    - untied: ``model.lm_head`` (nn.Linear, possibly QuantizedLinear, any bias)
    - tied:   ``model.model.embed_tokens.as_linear`` — the SAME kernel the
      model's own ``__call__`` uses (verified call sites, e.g. qwen2.py:175,
      llama.py, qwen3.py, qwen3_moe.py, mixtral.py, phi3.py)
    """

    def __init__(self, model: Any):
        self._model = model
        self._backbone = model.model
        self._args = getattr(model, "args", None)
        tied = bool(getattr(self._args, "tie_word_embeddings", True))
        self._tied = tied
        self._head = None if tied else model.lm_head

    def backbone(self, tokens: mx.array, cache: Any = None) -> mx.array:
        return self._backbone(tokens, cache)

    def lm_head(self, hidden: mx.array) -> mx.array:
        if self._tied:
            return self._backbone.embed_tokens.as_linear(hidden)
        return self._head(hidden)


class _Gemma3TextAdapter:
    """Gemma 3 text: always-untied head (gemma3_text.py hardcodes
    ``tie_word_embeddings=False`` and builds ``self.lm_head``)."""

    def __init__(self, model: Any):
        self._backbone = model.model
        self._head = model.lm_head

    def backbone(self, tokens: mx.array, cache: Any = None) -> mx.array:
        return self._backbone(tokens, cache)

    def lm_head(self, hidden: mx.array) -> mx.array:
        return self._head(hidden)


class _PhiAdapter:
    """Phi-4 (phi.py): untied head WITH bias — the head module carries it."""

    def __init__(self, model: Any):
        self._backbone = model.model
        self._head = model.lm_head

    def backbone(self, tokens: mx.array, cache: Any = None) -> mx.array:
        return self._backbone(tokens, cache)

    def lm_head(self, hidden: mx.array) -> mx.array:
        return self._head(hidden)


class _Mistral3Adapter:
    """mistral3 wraps an inner llama or ministral3 Model — delegate.

    The inner model has the standard layout (its own .model/.lm_head or
    tied embedding), so the adapter resolves the INNER adapter once.
    """

    def __init__(self, model: Any):
        inner = model.language_model
        self._inner: LMHeadAdapter = adapter_for(inner)

    def backbone(self, tokens: mx.array, cache: Any = None) -> mx.array:
        return self._inner.backbone(tokens, cache)

    def lm_head(self, hidden: mx.array) -> mx.array:
        return self._inner.lm_head(hidden)


class _Gemma3Adapter:
    """gemma3 (multimodal) delegates to its gemma3_text language_model."""

    def __init__(self, model: Any):
        self._inner: LMHeadAdapter = adapter_for(model.language_model)

    def backbone(self, tokens: mx.array, cache: Any = None) -> mx.array:
        return self._inner.backbone(tokens, cache)

    def lm_head(self, hidden: mx.array) -> mx.array:
        return self._inner.lm_head(hidden)


# model_type -> adapter factory. Keys mirror the ``model_type`` strings the
# installed mlx_lm configs carry (config.json "model_type").
_REGISTRY: dict[str, type] = {
    "qwen2": _StandardAdapter,  # covers Qwen2 / Qwen2.5 (incl. 0.5B-7B)
    "qwen3": _StandardAdapter,
    "qwen3_moe": _StandardAdapter,
    "llama": _StandardAdapter,  # Llama 3.x
    "mixtral": _StandardAdapter,
    "phi3": _StandardAdapter,  # phi-3 / phi-3.5 (phi.py's type is "phi")
    "phi": _PhiAdapter,
    "gemma3_text": _Gemma3TextAdapter,
    "gemma3": _Gemma3Adapter,
    "mistral3": _Mistral3Adapter,
}


def _model_type_of(model: Any) -> str | None:
    """The ONE source mlx_lm guarantees: ``model.model_type``.

    Every installed Model sets ``self.model_type = args.model_type`` in
    ``__init__`` (verified: 108 model classes, e.g. qwen2.py:162,
    gemma3_text.py:219, phi.py:159). No config-dict probing, no args
    fallback — a family that somehow lacks the attribute is registered
    explicitly by adding a registry entry, not discovered by guessing.
    """
    mt = getattr(model, "model_type", None)
    return mt if isinstance(mt, str) else None


def adapter_for(model: Any) -> LMHeadAdapter:
    """Return the LMHeadAdapter for a loaded mlx_lm model.

    Dispatch on ``model.model_type`` ONLY (the attribute every mlx_lm
    Model.__init__ sets). An unknown type raises
    :class:`UnsupportedModelError` — full stop. There is deliberately NO
    structural fallback: guessing a layout from attribute shapes is the
    dual path the review forbids. Adding a family = adding a registry
    entry + a test.
    """
    mt = _model_type_of(model)
    adapter_cls = _REGISTRY.get(mt) if mt else None
    if adapter_cls is not None:
        return adapter_cls(model)
    raise UnsupportedModelError(mt, list_supported_model_types())


def list_supported_model_types() -> list[str]:
    """Sorted supported ``model_type`` keys (for error messages/docs)."""
    return sorted(_REGISTRY)
