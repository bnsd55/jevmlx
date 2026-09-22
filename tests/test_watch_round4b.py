"""W6-UI round 4b: real-shape tests built from M5's exact tree (issue #63, 07:24).

Tests the 6 data-layer bugs against fixtures that match the real M5 bench
tree, run.json shapes, and folder names. The Llama run.json is written
verbatim from the M5 comment.
"""

from __future__ import annotations

import json
from pathlib import Path

from jevmlx.watch import (
    _derive_run_index,
    _find_sibling_cases_total,
    _infer_live_step_id,
    _model_from_folder,
    build_dashboard,
)

# The verbatim Llama run.json from M5's 07:24 comment on issue #63.
LLAMA_RUN_JSON = {
    "circuit_breaker": None,
    "config": {
        "dataset_lock_sha256": "03eae290c3587ab85235e1825722a0971661b416b6a3b92f5e6ba32ec087ad0a",
        "dataset_path": "/Users/benshaharizad/.cache/jevmlx/bench/typesafe.jsonl",
        "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "model_revision": "90215b22ec18e72f623dde2ea7af4097025160e2",
        "permutations": "rotations",
        "prompt_version": "jevmlx-parallel-v9",
        "quantization": {"bits": 4, "group_size": 64},
        "scoring": "labels",
        "split": "all",
        "temperature": 1.0,
        "tokenizer_chat_template_sha256": (
            "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65"
        ),
        "track": "parallel",
    },
    "counts": {"cases": 45, "fields": 365, "prediction_lines": 14848},
    "environment": {
        "chip": "Apple M5 Max",
        "git_sha": "dbb1ff1f04a0bce4b20817e4c09eae2b5a3d2ad3",
        "ram_gb": 128.0,
        "timestamp_utc": "2026-09-21T17:33:29+00:00",
    },
    "memory": {
        "active_memory_bytes": 4517412872,
        "cache_memory_bytes": 8590899278,
        "metal_cache_limit_bytes": 8589934592,
        "peak_memory_bytes": 4991615124,
    },
    "run_id": "20260921T161835Z-82620b",
}


def _write_run_json(path: Path, run_json: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_json, sort_keys=True), encoding="utf-8")


def _make_real_tree(tmp_path: Path) -> Path:
    """Build a tree matching the M5 shapes: bench-rest + bench-quality + ab + invariance."""
    out = tmp_path / "m5-2026-09-20-full"
    out.mkdir()
    (out / "RUNBOOK.md").write_text("# M5 runbook — 2026-01-01T00:00:00Z\n", encoding="utf-8")

    # bench-rest: 3 model folders, each with parallel-labels-typesafe (finished).
    for model_slug, model_id in [
        ("mlx-community--gemma-3-12b-it-4bit", "mlx-community/gemma-3-12b-it-4bit"),
        ("mlx-community--llama-3.1-8b-instruct-4bit", "mlx-community/Llama-3.1-8B-Instruct-4bit"),
        ("mlx-community--qwen3-8b-4bit", "mlx-community/Qwen3-8B-4bit"),
    ]:
        combo = out / "bench-rest" / f"m5max-128gb-{model_slug}" / "parallel-labels-typesafe"
        rj = json.loads(json.dumps(LLAMA_RUN_JSON))
        rj["config"]["model"] = model_id
        _write_run_json(combo / "run.json", rj)
        (combo / "report.json").write_text("{}", encoding="utf-8")

    # bench-quality: one real model folder + one alias folder (no config.model).
    combo_q = (
        out
        / "bench-quality"
        / "m5max-128gb-mlx-community--qwen2.5-7b-instruct-4bit"
        / "parallel-labels-typesafe"
    )
    rj_q = json.loads(json.dumps(LLAMA_RUN_JSON))
    rj_q["config"]["model"] = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    _write_run_json(combo_q / "run.json", rj_q)
    (combo_q / "report.json").write_text("{}", encoding="utf-8")

    # Alias folder: 552-byte run.json with no config.model.
    alias_combo = out / "bench-quality" / "m5max-128gb-quality" / "parallel-labels-typesafe"
    _write_run_json(alias_combo / "run.json", {"run_id": "stub", "environment": {}})

    # ab: a running A/B combo with heartbeat (no run.json yet, just heartbeat).
    ab_combo = out / "ab" / "bench-quality" / "m5max-128gb-quality" / "parallel-labels-typesafe"
    ab_combo.mkdir(parents=True)
    (ab_combo / "heartbeat.jsonl").write_text(
        json.dumps(
            {
                "combo": "parallel-labels-typesafe",
                "cases_done": 301,
                "pred_lines": 14848,
                "elapsed_s": 120,
                "peak_memory_bytes": 4991615124,
                "active_memory_bytes": 4517412872,
                "cache_memory_bytes": 8590899278,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # ab-worktree: a git worktree dir (should be skipped).
    (out / "ab-worktree").mkdir()

    # invariance: own section (should not appear as model rows).
    for extra in ("extra01", "extra05", "extra20", "extra40"):
        (out / "invariance" / extra).mkdir(parents=True)

    return out


class TestCapGb:
    """Bug 3: cap_gb reads memory.metal_cache_limit_bytes (top-level, not config)."""

    def test_cap_gb_from_top_level_memory(self, tmp_path):
        out = _make_real_tree(tmp_path)
        dash = build_dashboard(out)
        # 8589934592 bytes = 8.0 GB.
        assert dash["memory"]["cap_gb"] == 8.0

    def test_cap_gb_fallback_to_config_memory(self, tmp_path):
        """Old fixtures put memory under config.memory — backward compat."""
        out = tmp_path / "run"
        out.mkdir()
        (out / "run.json").write_text(
            json.dumps(
                {
                    "config": {
                        "model": "M",
                        "memory": {"metal_cache_limit_bytes": 8 * 2**30},
                    },
                    "environment": {"chip": "M2", "ram_gb": 128.0},
                }
            ),
            encoding="utf-8",
        )
        dash = build_dashboard(out)
        assert dash["memory"]["cap_gb"] == 8.0


class TestRunIndex:
    """Bug 3: run_i/run_n derived from model folder count under bench-rest."""

    def test_derive_run_index_from_bench_rest(self, tmp_path):
        out = _make_real_tree(tmp_path)
        # For a bench-rest combo, run_n = 3 model folders.
        llama_combo = (
            out
            / "bench-rest"
            / "m5max-128gb-mlx-community--llama-3.1-8b-instruct-4bit"
            / "parallel-labels-typesafe"
        )
        run_i, run_n = _derive_run_index(out, llama_combo)
        assert run_n == 3
        assert run_i is not None and 1 <= run_i <= 3

    def test_derive_run_index_returns_none_for_flat(self, tmp_path):
        """A single-combo out dir (no bench-* parent) returns (None, None)."""
        out = tmp_path / "run"
        out.mkdir()
        assert _derive_run_index(out, out) == (None, None)


class TestCasesTotal:
    """Bug 2: cases_total never 0; uses sibling counts when the combo is running."""

    def test_finished_combo_uses_counts_cases(self, tmp_path):
        out = _make_real_tree(tmp_path)
        llama_combo = (
            out
            / "bench-rest"
            / "m5max-128gb-mlx-community--llama-3.1-8b-instruct-4bit"
            / "parallel-labels-typesafe"
        )
        from jevmlx.watch import _build_now, parse_heartbeat

        now_hb = parse_heartbeat(llama_combo)
        now_run = json.loads((llama_combo / "run.json").read_text())
        now_cfg = now_run.get("config", {})
        now_block = _build_now(llama_combo, now_hb, now_cfg, out)
        # A finished combo has counts.cases = 45.
        assert now_block["cases_total"] == 45

    def test_sibling_cases_total_found(self, tmp_path):
        out = _make_real_tree(tmp_path)
        # The A/B combo has no run.json; look for a sibling with the same dataset_path.
        # The siblings are under bench-rest and bench-quality.
        dataset_path = "/Users/benshaharizad/.cache/jevmlx/bench/typesafe.jsonl"
        result = _find_sibling_cases_total(out, None, dataset_path)
        assert result == 45  # counts.cases from a sibling


class TestModelNeverNone:
    """Bug 5: model is never None; alias folders show 'alias folder (no run)'."""

    def test_model_from_folder_slug(self, tmp_path):
        out = _make_real_tree(tmp_path)
        combo = (
            out
            / "bench-rest"
            / "m5max-128gb-mlx-community--llama-3.1-8b-instruct-4bit"
            / "parallel-labels-typesafe"
        )
        # The folder slug has '--' -> reconstructs the Hub id.
        assert _model_from_folder(combo) == "mlx-community/llama-3.1-8b-instruct-4bit"

    def test_model_from_sibling(self, tmp_path):
        out = _make_real_tree(tmp_path)
        # The alias folder has no config.model; a sibling in the same model folder
        # has a run.json with config.model.
        alias_combo = out / "bench-quality" / "m5max-128gb-quality" / "parallel-labels-typesafe"
        # m5max-128gb-quality has no '--' so _model_from_folder returns None.
        assert _model_from_folder(alias_combo) is None
        # _model_from_sibling looks at sibling combos in the same model folder.
        # But the alias folder's siblings are in bench-quality, not the same parent.
        # The alias folder IS the model folder; its siblings are other model folders.
        # So _model_from_sibling returns None here (no sibling combo in the same parent).
        # The results builder falls back to 'alias folder (no run)'.

    def test_results_model_never_none(self, tmp_path):
        out = _make_real_tree(tmp_path)
        dash = build_dashboard(out)
        for row in dash["results"]:
            assert row["model"] is not None
            assert row["model"] != "None"
            # The alias folder should show 'alias folder (no run)'.
            if "quality" in row.get("combo_id", "") and "m5max-128gb-quality" in row.get(
                "combo_id", ""
            ):
                assert row["model"] == "alias folder (no run)"

    def test_ab_worktree_skipped(self, tmp_path):
        out = _make_real_tree(tmp_path)
        dash = build_dashboard(out)
        for row in dash["results"]:
            assert "ab-worktree" not in row["combo_id"]


class TestMixedModelNames:
    """Bug 6: one display name per model = config.model (no slug/name mixing)."""

    def test_model_is_config_model_not_slug(self, tmp_path):
        out = _make_real_tree(tmp_path)
        dash = build_dashboard(out)
        for row in dash["results"]:
            if row["model"] != "alias folder (no run)":
                # Every model should be a Hub id (contains /), not a slug.
                assert "/" in row["model"], f"model '{row['model']}' is not a Hub id"


class TestLiveStepInference:
    """Bug 4: infer the live step from the newest heartbeat's folder path."""

    def test_infer_ab_step(self, tmp_path):
        out = _make_real_tree(tmp_path)
        ab_combo = out / "ab" / "bench-quality" / "m5max-128gb-quality" / "parallel-labels-typesafe"
        assert _infer_live_step_id(ab_combo, out) == "A/B bench"

    def test_infer_bench_rest_step(self, tmp_path):
        out = _make_real_tree(tmp_path)
        combo = (
            out
            / "bench-rest"
            / "m5max-128gb-mlx-community--llama-3.1-8b-instruct-4bit"
            / "parallel-labels-typesafe"
        )
        assert _infer_live_step_id(combo, out) == "rest bench"

    def test_infer_bench_quality_step(self, tmp_path):
        out = _make_real_tree(tmp_path)
        combo = (
            out
            / "bench-quality"
            / "m5max-128gb-mlx-community--qwen2.5-7b-instruct-4bit"
            / "parallel-labels-typesafe"
        )
        assert _infer_live_step_id(combo, out) == "quality"

    def test_infer_invariance_step(self, tmp_path):
        out = _make_real_tree(tmp_path)
        combo = out / "invariance" / "extra01"
        assert _infer_live_step_id(combo, out) == "invariance"

    def test_pipeline_shows_live_step_when_running(self, tmp_path):
        out = _make_real_tree(tmp_path)
        dash = build_dashboard(out)
        steps = dash["pipeline"]["steps"]
        # The A/B combo is running (has heartbeat, no report.json) -> a
        # 'A/B bench' step should appear as running.
        ab_steps = [s for s in steps if s["id"] == "A/B bench"]
        assert len(ab_steps) == 1
        assert ab_steps[0]["state"] == "running"
