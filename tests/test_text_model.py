"""Multimodal wrappers load as their text model (qwen3_5, gemma3, mistral3)."""

import inspect

from jevmlx.engine import _load_engine_resolved, _text_model


class _Inner:
    def __init__(self):
        self.model = object()


class _Wrapper:
    """The mlx-lm vision-language layout: no ``model``, text stack under ``language_model``."""

    def __init__(self):
        self.language_model = _Inner()


def test_wrapper_resolves_to_language_model():
    wrapper = _Wrapper()
    assert _text_model(wrapper) is wrapper.language_model


def test_text_model_is_returned_unchanged():
    inner = _Inner()
    assert _text_model(inner) is inner


def test_loader_unwraps_before_anything_reads_the_model():
    src = inspect.getsource(_load_engine_resolved.__wrapped__)
    load_at = src.index("model, tokenizer = load(model_id)")
    unwrap_at = src.index("model = _text_model(model)")
    assert load_at < unwrap_at < src.index("make_prompt_cache(model)")
