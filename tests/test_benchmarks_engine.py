"""B5: every benchmark script that calls load_engine has an executed
fake-engine test.

Root cause of the M5 field failure: benchmarks/probe.py unpacked
``model, tokenizer = load_engine(args.model)`` but load_engine returns the
frozen Engine dataclass (since the Engine PR). Tests only PARSED these
scripts' argv; nothing EXECUTED main() with a fake Engine.

This file runs each script's main() end to end with a fake Engine
(monkeypatched load_engine) and asserts the output files exist with the
expected keys. MLX-conditional: scripts that call mx.* directly are guarded
by a has_mlx skip so Linux CI (no MLX) doesn't fail.

Scripts covered:
- benchmarks/probe.py (slope + adapters) — the reported failure
- benchmarks/naive_vs_parallel.py (run_naive vs run_parallel)
- benchmarks/timing.py (run_timing with injectable fakes)
- benchmarks/driftprobe.py (repeatability + width matrix)
- benchmarks/layer_bisect.py (layer bisect)
- benchmarks/compat.py (probe_row)
- benchmarks/invariance.py (parallel_decide_fn path)
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeModel, _Mod97Tokenizer, make_engine


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


requires_mlx = pytest.mark.skipif(not _has_mlx(), reason="MLX not available")


# ---- probe.py (the reported failure) ---------------------------------------


@requires_mlx
def test_probe_slope_main_with_fake_engine(tmp_path, monkeypatch):
    """probe.py --command slope: main() runs with a fake Engine and writes
    a JSON file with the expected keys."""
    from benchmarks import probe

    fake_engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())

    monkeypatch.setattr("jevmlx.engine.load_engine", lambda model: fake_engine)

    out_path = tmp_path / "slope.json"
    rc = probe.main(["--model", "fake/model", "--command", "slope", "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["model"] == "fake/model"
    assert "slope" in payload
    assert "widths" in payload["slope"]
    assert "peak_bytes" in payload["slope"]
    assert "fitted_slope_per_bin" in payload["slope"]


@requires_mlx
def test_probe_adapters_with_fake_engine(monkeypatch):
    """probe_adapters() runs with a fake Engine + a mocked adapter_for (the
    fake model has no model_type) and returns the expected keys.

    Tested directly (not via main()) because the 255-choice preset hangs the
    fake-tokenizer codebook search; main() calls bundled_preset_specs() which
    includes it. The slope command is tested via main() above."""
    import mlx.core as mx

    from benchmarks import probe
    from jevmlx.parity import bundled_preset_specs

    fake_engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())

    class _FakeAdapter:
        def backbone(self, tokens, cache=None):
            batch, seq = tokens.shape
            return mx.zeros((batch, seq, 64))

        def lm_head(self, hidden):
            shape = hidden.shape
            return mx.zeros(shape[:-1] + (64,))

    monkeypatch.setattr("jevmlx.adapters.adapter_for", lambda model: _FakeAdapter())

    # Use only the first preset (code_security, 28 fields — fast).
    result = probe.probe_adapters(
        fake_engine.model, fake_engine.tokenizer, bundled_preset_specs()[:1], max_cases=1
    )
    assert "cases" in result
    assert "max_abs_diff" in result or "overall_max_abs_diff" in result
    assert len(result["cases"]) == 1


# ---- naive_vs_parallel.py --------------------------------------------------


@requires_mlx
def test_naive_vs_parallel_main_with_fake_engine(tmp_path, monkeypatch):
    """naive_vs_parallel.py main(): runs with a fake Engine and writes a
    JSON results file."""
    from benchmarks import naive_vs_parallel
    from tests.conftest import FakeTokenizer

    class _TokWithDecode(FakeTokenizer):
        eos_token_id = 0

        def decode(self, ids, **kw):
            return "".join(chr(i % 60 + 32) for i in ids)

        def convert_tokens_to_ids(self, token):
            return 0

    fake_engine = make_engine(FakeModel(vocab_size=64), _TokWithDecode())
    monkeypatch.setattr("jevmlx.engine.load_engine", lambda model: fake_engine)
    monkeypatch.setattr("benchmarks.naive_vs_parallel.DEFAULT_PRESETS", ["fintech_fraud.json"])
    monkeypatch.setattr(
        "sys.argv",
        ["naive_vs_parallel.py", "fake/model", "--outdir", str(tmp_path)],
    )

    naive_vs_parallel.main()
    out_files = list(tmp_path.glob("*.json"))
    assert len(out_files) == 1
    payload = json.loads(out_files[0].read_text())
    assert payload["model"] == "fake/model"
    assert "load_seconds" in payload
    assert "results" in payload
    assert len(payload["results"]) >= 1
    row = payload["results"][0]
    assert "naive_ms" in row
    assert "parallel_ms" in row
    assert "speedup" in row


# ---- timing.py -------------------------------------------------------------


@requires_mlx
def test_timing_main_with_fake_engine(tmp_path, monkeypatch):
    """timing.py main(): runs with a fake Engine and writes a timing JSON."""
    from benchmarks import timing

    fake_engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())

    monkeypatch.setattr("jevmlx.engine.load_engine", lambda model: fake_engine)

    out_path = tmp_path / "timing-fake_model.json"
    timing.main(
        [
            "--model",
            "fake/model",
            "--out",
            str(tmp_path),
            "--reps",
            "1",
            "--presets",
            "fintech_fraud.json",
        ]
    )
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["model"] == "fake/model"
    assert "presets" in payload
    assert len(payload["presets"]) >= 1


# ---- driftprobe.py ---------------------------------------------------------


@requires_mlx
def test_driftprobe_main_with_fake_engine(tmp_path, monkeypatch):
    """driftprobe.py main(): runs with a fake Engine and writes a JSON report."""
    from benchmarks import driftprobe

    fake_engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())
    out_dir = tmp_path / "driftprobe_out"

    monkeypatch.setattr("benchmarks.driftprobe.load_engine", lambda model: fake_engine)
    monkeypatch.setattr(
        "sys.argv",
        ["driftprobe.py", "--model", "fake/model", "--out", str(out_dir), "--quick"],
    )

    rc = driftprobe.main()
    assert rc == 0
    out_path = out_dir / "driftprobe.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["model"] == "fake/model"
    assert "repeatability" in payload
    assert "width_matrix" in payload


# ---- layer_bisect.py -------------------------------------------------------


@requires_mlx
def test_layer_bisect_main_with_fake_engine(tmp_path, monkeypatch):
    """layer_bisect.py main(): runs with a fake Engine + mocked _prefill and
    run_bisect (the bisect walks model.model.embed_tokens which the FakeModel
    doesn't have). Asserts the JSON report has the expected keys."""
    from benchmarks import layer_bisect

    fake_model = FakeModel(vocab_size=64, n_layers=2)
    fake_model.model = type("M", (), {"layers": fake_model.layers})()
    fake_engine = make_engine(fake_model, _Mod97Tokenizer())
    out_dir = tmp_path / "bisect_out"

    monkeypatch.setattr("benchmarks.layer_bisect.load_engine", lambda model: fake_engine)
    monkeypatch.setattr(
        "sys.argv",
        ["layer_bisect.py", "--model", "fake/model", "--out", str(out_dir)],
    )

    # Mock _prefill to return a fake PrefillResult with a cache list.

    class _FakePrefill:
        cache = [None, None]

    monkeypatch.setattr(
        "benchmarks.layer_bisect._prefill",
        lambda *a, **kw: _FakePrefill(),
    )
    # Mock run_bisect to return a minimal result dict.
    fake_bisect_result = {
        "tables": {},
        "layers": [],
        "label": "fp16",
    }
    monkeypatch.setattr(
        "benchmarks.layer_bisect.run_bisect",
        lambda *a, **kw: fake_bisect_result,
    )
    # Mock the fp32 dequantize + followups (need a real mlx_lm model).
    monkeypatch.setattr("benchmarks.layer_bisect._dequantize_model_fp32", lambda model: None)
    monkeypatch.setattr(
        "benchmarks.layer_bisect._capture_reference_batch1",
        lambda *a, **kw: ({}, {}),
    )
    monkeypatch.setattr(
        "benchmarks.layer_bisect.followup_width_curve", lambda *a, **kw: {"rows": []}
    )
    monkeypatch.setattr(
        "benchmarks.layer_bisect.followup_relative_diff",
        lambda *a, **kw: {
            "M=16": {"abs_diff": {"residual_in": []}},
            "M=112": {"abs_diff": {"residual_in": []}},
        },
    )
    monkeypatch.setattr(
        "benchmarks.layer_bisect.followup_gemm_isolation",
        lambda *a, **kw: {
            "jump_M": 16,
            "full_vs_row_max_abs_diff": 0.0,
            "max_abs_activation": 0.0,
            "relative_diff": 0.0,
            "gemm_path_changes_with_M": "none",
        },
    )
    monkeypatch.setattr(
        "benchmarks.layer_bisect.followup_attention_isolation",
        lambda *a, **kw: {
            m: {
                step: {"max_abs_diff": 0.0, "max_abs_activation": 0.0, "relative_diff": 0.0}
                for step in ("qkv", "qk_rope", "sdpa", "out")
            }
            for m in ("M=16", "M=32", "M=112")
        },
    )
    monkeypatch.setattr(
        "benchmarks.layer_bisect.followup_cache_and_mask_info",
        lambda *a, **kw: {
            "cache_type": "fake",
            "cache_dtype": "float16",
            "cache_has_bits": False,
            "sdpa_function": "none",
            "per_M": {
                m: {
                    "mask_type": "none",
                    "mask_shape": "(1,1)",
                    "q_shape": "(1,1)",
                    "k_shape": "(1,1)",
                    "v_shape": "(1,1)",
                    "cache_offset": 0,
                }
                for m in ("M=16", "M=32", "M=112")
            },
        },
    )
    monkeypatch.setattr(
        "benchmarks.layer_bisect.followup_insitu_vs_recomputed",
        lambda *a, **kw: {
            "per_layer_diff": [],
            "first_above_1e-4": None,
            "insitu_vs_recomputed": {},
            "decision_token_match": True,
        },
    )
    # engine.model.model(...) is called for the fp32 warmup.
    import mlx.core as mx

    fake_model.model = type(
        "M",
        (),
        {
            "layers": fake_model.layers,
            "__call__": lambda self, tokens: mx.zeros((tokens.shape[0], tokens.shape[1], 64)),
        },
    )()

    rc = layer_bisect.main()
    assert rc == 0
    out_path = out_dir / "layer_bisect.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["model"] == "fake/model"
    assert "fp16_body" in payload or "fp16" in payload


# ---- compat.py -------------------------------------------------------------


@requires_mlx
def test_compat_probe_row_with_fake_engine(tmp_path, monkeypatch):
    """compat.py probe_row(): runs with a fake Engine and returns a row dict."""
    from benchmarks import compat

    fake_engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())

    monkeypatch.setattr("benchmarks.compat.load_engine", lambda model: fake_engine)
    monkeypatch.setattr("benchmarks.compat.PRESETS", ["fintech_fraud.json"])

    row = compat.probe_row("fake/model")
    assert row["model"] == "fake/model"
    assert "loads" in row
    assert row["loads"] == "y"


# ---- invariance.py (parallel_decide_fn path) -------------------------------


@requires_mlx
def test_invariance_main_with_fake_engine(tmp_path, monkeypatch):
    """invariance.py main(): runs with a fake Engine and writes
    invariance.json + summary.md."""
    from benchmarks import invariance

    fake_engine = make_engine(FakeModel(vocab_size=64), _Mod97Tokenizer())

    monkeypatch.setattr("jevmlx.engine.load_engine", lambda model: fake_engine)

    # Write a small cases jsonl.
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "schema": {
                    "risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"}
                },
                "context": "test context",
                "labels": {"risk": "LOW"},
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "inv"
    rc = invariance.main(
        ["--model", "fake/model", "--data", str(cases_path), "--out", str(out_dir), "--extra", "1"]
    )
    assert rc == 0
    assert (out_dir / "invariance.json").exists()
    payload = json.loads((out_dir / "invariance.json").read_text())
    assert "targets" in payload
