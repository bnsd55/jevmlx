"""W6-1 tests: jevmlx/adapters.py — fake models for every attribute layout,
plus one slow real-model equivalence test.

The fakes mirror the REAL mlx_lm attribute layouts the adapters navigate:

- _StandardAdapter (untied): model.model + model.lm_head (qwen3/llama untied)
- _StandardAdapter (tied):   model.model + model.model.embed_tokens.as_linear
                             (qwen2/llama/qwen3/mixtral/phi3 tied default)
- _Gemma3TextAdapter:        model.model + model.lm_head (always untied)
- _PhiAdapter:               model.model + model.lm_head (biased)
- _Mistral3Adapter:          model.language_model -> inner standard model
- _Gemma3Adapter:            model.language_model -> inner gemma3_text model

Each adapter must reproduce model(x) exactly: hidden = backbone(x), and
lm_head(hidden) == model(x) — the adapter split is byte-exact by construction
(see the slow test for the real-model proof).
"""

from typing import Any

import mlx.core as mx
import pytest

from jevmlx.adapters import (
    LMHeadAdapter,
    UnsupportedModelError,
    adapter_for,
    list_supported_model_types,
)


class _Embed:
    """nn.Embedding stand-in with a quantizable as_linear."""

    def __init__(self, vocab: int, hidden: int, scale: float = 1.0):
        self.weight = (
            mx.arange(vocab * hidden, dtype=mx.float32).reshape(vocab, hidden)
            * scale
            / (vocab * hidden)
        )
        self.quantized = False

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.weight[tokens]

    def as_linear(self, hidden: mx.array) -> mx.array:
        # as_linear(embedding) = x @ W.T  ([.., H] @ [H, V] -> [.., V])
        return hidden @ self.weight.T


class _Linear:
    def __init__(self, hidden: int, vocab: int, bias: bool = False):
        self.weight = mx.zeros((vocab, hidden)) + 0.01
        self.bias = mx.zeros((vocab,)) if bias else None

    def __call__(self, hidden: mx.array) -> mx.array:
        out = hidden @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class _InnerModel:
    """Stands in for Qwen2Model/Gemma3Model: embedding + norm."""

    def __init__(self, vocab: int, hidden: int, tied: bool):
        self.embed_tokens = _Embed(vocab, hidden)

    def __call__(self, tokens: mx.array, cache: Any = None) -> mx.array:
        return self.embed_tokens(tokens)


class _FakeModelBase:
    model_type = None
    args = None

    def __call__(self, tokens: mx.array, cache: Any = None) -> mx.array:
        inner = getattr(self, "language_model", None)
        if inner is not None:  # wrapper (mistral3/gemma3): mlx_lm delegates
            return inner(tokens, cache=cache)
        hidden = self.model(tokens, cache)
        return self._full_head(hidden)

    def _full_head(self, hidden: mx.array) -> mx.array:
        raise NotImplementedError


def _std_model(tied: bool, vocab: int = 32, hidden: int = 8, mt: str = "qwen2") -> _FakeModelBase:
    class _Std(_FakeModelBase):
        pass

    m = _Std()
    m.model_type = mt
    m.args = type("Args", (), {"tie_word_embeddings": tied, "model_type": mt})()
    m.model = _InnerModel(vocab, hidden, tied)
    if not tied:
        m.lm_head = _Linear(hidden, vocab)
    else:
        m._full_head = m.model.embed_tokens.as_linear
    if not tied:
        m._full_head = m.lm_head
    return m


class TestStandardAdapter:
    def test_untied_uses_lm_head(self):
        model = _std_model(tied=False, mt="llama")
        adapter: LMHeadAdapter = adapter_for(model)
        tokens = mx.array([[1, 2, 3]])
        hidden = adapter.backbone(tokens)
        assert hidden.shape == (1, 3, 8)
        out = adapter.lm_head(hidden)
        ref = model(tokens)
        assert mx.allclose(out, ref).item()

    def test_tied_uses_embed_as_linear(self):
        model = _std_model(tied=True, mt="qwen2")
        adapter = adapter_for(model)
        assert adapter._tied is True
        tokens = mx.array([[1, 2]])
        out = adapter.lm_head(adapter.backbone(tokens))
        ref = model(tokens)
        assert mx.allclose(out, ref).item()

    def test_split_matches_call_batched(self):
        """backbone is cache-compatible: adapter split == model call at B=2."""
        model = _std_model(tied=False, mt="qwen3")
        adapter = adapter_for(model)
        tokens = mx.array([[1, 2, 3], [4, 5, 6]])
        assert mx.allclose(adapter.lm_head(adapter.backbone(tokens)), model(tokens)).item()


def test_gemma3_text_adapter():
    model = _std_model(tied=False, mt="gemma3_text")
    # gemma3_text is ALWAYS untied regardless of args.
    adapter = adapter_for(model)
    tokens = mx.array([[2, 5]])
    assert mx.allclose(adapter.lm_head(adapter.backbone(tokens)), model(tokens)).item()


def test_phi_biased_adapter():
    model = _std_model(tied=False, mt="phi")
    model.lm_head = _Linear(8, 32, bias=True)
    model._full_head = model.lm_head
    adapter = adapter_for(model)
    tokens = mx.array([[1, 2]])
    assert mx.allclose(adapter.lm_head(adapter.backbone(tokens)), model(tokens)).item()


def test_mistral3_delegates_to_inner():
    inner = _std_model(tied=True, mt="llama")

    class _Wrapper(_FakeModelBase):
        pass

    w = _Wrapper()
    w.model_type = "mistral3"
    w.args = type("Args", (), {"model_type": "mistral3"})()
    w.language_model = inner
    adapter = adapter_for(w)
    tokens = mx.array([[1, 2]])
    assert mx.allclose(adapter.lm_head(adapter.backbone(tokens)), inner(tokens)).item()
    # And the wrapper's own __call__ delegates too (mlx_lm behavior).
    assert mx.allclose(adapter.lm_head(adapter.backbone(tokens)), w(tokens)).item()


def test_gemma3_multimodal_delegates():
    inner = _std_model(tied=False, mt="gemma3_text")

    class _Wrapper(_FakeModelBase):
        pass

    w = _Wrapper()
    w.model_type = "gemma3"
    w.args = type("Args", (), {"model_type": "gemma3"})()
    w.language_model = inner
    adapter = adapter_for(w)
    tokens = mx.array([[1, 2]])
    assert mx.allclose(adapter.lm_head(adapter.backbone(tokens)), inner(tokens)).item()


def test_unsupported_model_type_raises():
    class _Weird:
        model_type = "hf_echo"

    with pytest.raises(UnsupportedModelError) as exc:
        adapter_for(_Weird())
    assert exc.value.model_type == "hf_echo"
    assert "gemma3_text" in exc.value.supported


def test_unknown_type_raises_even_with_standard_layout():
    """No structural fallback, full stop (review F1): a model whose type is
    unknown raises UnsupportedModelError even though its attribute layout
    matches _StandardAdapter — adding a family means adding a registry
    entry + a test, never attribute-shape guessing."""
    model = _std_model(tied=False, mt="brand_new_family")
    with pytest.raises(UnsupportedModelError) as exc:
        adapter_for(model)
    assert exc.value.model_type == "brand_new_family"


def test_missing_model_type_raises():
    """The single source of truth is model.model_type (review F2): an
    object without it raises instead of being probed through args/config
    dicts."""

    class _NoType:
        args = type("Args", (), {"model_type": "qwen2"})()  # decoy

    with pytest.raises(UnsupportedModelError):
        adapter_for(_NoType())


def test_list_supported_model_types():
    supported = list_supported_model_types()
    for expected in (
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "llama",
        "mixtral",
        "phi3",
        "phi",
        "gemma3_text",
        "gemma3",
        "mistral3",
    ):
        assert expected in supported


@pytest.mark.slow
def test_real_0_5b_adapter_split_equals_call(engine):
    """The adapter split reproduces the model's own __call__ bit-for-bit
    within 1e-4 on the slow-suite model (conftest.MODEL_ID — TIED
    embeddings exercise the as_linear path end to end)."""
    model, tokenizer = engine.model, engine.tokenizer
    adapter = adapter_for(model)
    tokens = mx.array([tokenizer.encode("Decide now")])

    full = model(tokens)
    split = adapter.lm_head(adapter.backbone(tokens))

    assert mx.max(mx.abs(full[:, -1, :] - split[:, -1, :])).item() < 1e-4
