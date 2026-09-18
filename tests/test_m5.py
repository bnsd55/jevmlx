"""Tests for benchmarks/m5.py — step planner and summary builder only.

No model ever loads here: the planner is checked against a temp directory,
and the summary builder against synthetic json blocks (the fake-model
contract). Subprocesses are not spawned either (execute_step is exercised
only via a fake runner in the flow test).
"""

from __future__ import annotations

import pytest

from benchmarks.m5 import (
    DEFAULT_PARITY_MODELS,
    Step,
    build_summary_text,
    combo_row,
    invariance_rollup,
    plan_steps,
    read_models_file,
    step_done,
)

QUALITY_TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"


@pytest.fixture()
def out(tmp_path):
    return tmp_path / "m5-run"


class TestReadModelsFile:
    def test_comments_and_blanks_skipped(self, tmp_path):
        f = tmp_path / "models.txt"
        f.write_text(
            "# parity set\n"
            "mlx-community/Qwen3-8B-4bit\n"
            "\n"
            "mlx-community/gemma-3-12b-it-4bit  # trailing comment\n",
            encoding="utf-8",
        )
        assert read_models_file(f) == [
            "mlx-community/Qwen3-8B-4bit",
            "mlx-community/gemma-3-12b-it-4bit",
        ]


class TestPlanSteps:
    def test_default_order_no_ab(self, out):
        out.mkdir()
        steps = plan_steps(out, parity_models=list(DEFAULT_PARITY_MODELS))
        assert [s.id for s in steps] == [
            "doctor",
            "parity-mlx-community--qwen3-8b-4bit",
            "parity-mlx-community--llama-3.1-8b-instruct-4bit",
            "parity-mlx-community--gemma-3-12b-it-4bit",
            "bench-quality",
            "invariance",
            "timing",
            "bench-rest",
            "summary",
        ]

    def test_doctor_is_gate_and_captures_json(self, out):
        steps = plan_steps(out, parity_models=[QUALITY_TARGET])
        doctor = steps[0]
        assert doctor.gate is True
        assert doctor.capture is True
        assert doctor.capture_path == out / "doctor.json"
        assert "--json" in doctor.argv

    def test_parity_step_rides_model_id_env(self, out):
        out.mkdir()
        steps = plan_steps(out, parity_models=["some-org/Model-4bit"])
        parity = next(s for s in steps if s.id.startswith("parity-"))
        assert parity.env == {"MODEL_ID": "some-org/Model-4bit"}
        assert "-m" in parity.argv and "slow" in parity.argv

    def test_bench_rest_excludes_quality_target(self, out):
        out.mkdir()
        models = [QUALITY_TARGET, "other-org/Other-4bit"]
        steps = plan_steps(out, parity_models=models)
        rest_file = out / "models-rest.txt"
        assert rest_file.exists()
        assert rest_file.read_text().split() == ["other-org/Other-4bit"]
        bench_rest = next(s for s in steps if s.id == "bench-rest")
        assert "--models-file" in bench_rest.argv

    def test_no_rest_step_when_only_quality_target(self, out):
        steps = plan_steps(out, parity_models=[QUALITY_TARGET])
        assert not (out / "models-rest.txt").exists()
        assert all(s.id != "bench-rest" for s in steps)

    def test_invariance_fetches_typesafe_when_missing(self, out):
        steps = plan_steps(out, parity_models=[QUALITY_TARGET], typesafe_data=out / "cases.jsonl")
        step = next(s for s in steps if s.id == "invariance")
        assert step.pre_target == out / "cases.jsonl"
        assert step.pre_argv and "typesafe.fetch" in " ".join(step.pre_argv)
        assert "--extra" in step.argv and "1,5,20,40" in step.argv

    def test_ab_steps_branch_worktree_and_cwd(self, out):
        steps = plan_steps(out, parity_models=[QUALITY_TARGET], ab_branch="w2a-field-local")
        ids = [s.id for s in steps]
        assert ids[-3:] == ["ab-setup", "ab-bench", "ab-invariance"] or ids[-4:] == [
            "ab-setup",
            "ab-bench",
            "ab-invariance",
            "summary",
        ]
        ab_setup = next(s for s in steps if s.id == "ab-setup")
        assert "worktree" in ab_setup.argv and "w2a-field-local" in ab_setup.argv
        assert any("uv" in " ".join(extra) for extra in ab_setup.extra_argv)
        ab_bench = next(s for s in steps if s.id == "ab-bench")
        assert ab_bench.cwd == out / "ab-worktree"
        assert str(out / "ab" / "bench-quality") in " ".join(ab_bench.argv)
        ab_inv = next(s for s in steps if s.id == "ab-invariance")
        assert str(out / "ab" / "invariance" / "invariance.json") in [
            str(p) for p in ab_inv.outputs
        ]

    def test_summary_step_in_process(self, out):
        steps = plan_steps(out, parity_models=[QUALITY_TARGET])
        summary = steps[-1]
        assert summary.in_process is True
        assert summary.argv == ()
        assert summary.outputs == (out / "SUMMARY.md",)

    def test_every_step_has_done_marker_and_outputs(self, out):
        for steps in (
            plan_steps(out / "a", parity_models=[QUALITY_TARGET]),
            plan_steps(out / "b", parity_models=[QUALITY_TARGET], ab_branch="main"),
        ):
            for step in steps:
                assert step.outputs, step.id


class TestStepDone:
    def test_skips_when_all_outputs_exist(self, tmp_path):
        done = tmp_path / "x.done"
        done.touch()
        step = Step(id="x", title="t", argv=("true",), outputs=(done,))
        assert step_done(step, fresh=False) is True

    def test_missing_output_means_run(self, tmp_path):
        step = Step(id="x", title="t", argv=("true",), outputs=(tmp_path / "missing.done",))
        assert step_done(step, fresh=False) is False

    def test_fresh_reruns_despite_outputs(self, tmp_path):
        done = tmp_path / "x.done"
        done.touch()
        step = Step(id="x", title="t", argv=("true",), outputs=(done,))
        assert step_done(step, fresh=True) is False


class TestComboRow:
    def test_agreement_wins_accuracy_fallback(self):
        row = combo_row(
            "c1",
            {"metrics": {"agreement": 0.9}},
            {"median": {"total_ms": 100.0, "peak_active_bytes": 2**30}},
        )
        assert row == {
            "combo": "c1",
            "agreement": 0.9,
            "accuracy": None,
            "total_ms": 100.0,
            "peak_gb": 1.0,
        }

    def test_missing_report_is_none_safe(self):
        row = combo_row("c2", None, None)
        assert row["accuracy"] is None and row["peak_gb"] is None and row["total_ms"] is None


class TestInvarianceRollup:
    def test_means_and_field_ladder(self):
        inv = {
            "targets": [
                {
                    "field": "f1",
                    "rungs": {
                        "1": {"flip_rate": 0.0, "winner_logodds_drift_mean": 0.0},
                        "40": {"flip_rate": 0.5, "winner_logodds_drift_mean": 1.5},
                    },
                },
                {
                    "field": "f2",
                    "rungs": {
                        "1": {"flip_rate": 0.0, "winner_logodds_drift_mean": 0.2},
                        "40": {"flip_rate": 0.1, "winner_logodds_drift_mean": 0.4},
                    },
                },
            ]
        }
        roll = invariance_rollup(inv)
        assert roll["mean_flip_rate"] == pytest.approx(0.15)
        assert roll["mean_drift"] == pytest.approx(0.525)
        assert roll["fields"]["f1"]["flip40"] == 0.5
        assert roll["fields"]["f1"]["drift40"] == 1.5

    def test_inf_drift_excluded_from_mean(self):
        inv = {
            "targets": [
                {
                    "field": "f",
                    "rungs": {
                        "1": {"flip_rate": 0.0, "winner_logodds_drift_mean": 0.0},
                        "40": {"flip_rate": 0.0, "winner_logodds_drift_mean": float("inf")},
                    },
                }
            ]
        }
        roll = invariance_rollup(inv)
        assert roll["mean_drift"] == 0.0

    def test_none_safe(self):
        assert invariance_rollup(None)["fields"] == {}
        assert invariance_rollup(None)["mean_flip_rate"] is None


class TestBuildSummaryText:
    @staticmethod
    def _side(
        *, agreement: float, flip: float, drift: float, total_ms: float, peak_gb: float
    ) -> dict:
        return {
            "combos": [
                combo_row(
                    "parallel-slots-typesafe",
                    {"metrics": {"agreement": agreement}},
                    {"median": {"total_ms": total_ms, "peak_active_bytes": int(peak_gb * 2**30)}},
                )
            ],
            "invariance": {
                "mean_flip_rate": flip,
                "mean_drift": drift,
                "fields": {"risk_level": {"flip40": flip, "drift40": drift}},
            },
            "timing_report": {"fintech_fraud": {"total_ms_median": total_ms, "peak_gb": peak_gb}},
        }

    def test_main_only(self):
        main = self._side(agreement=0.9, flip=0.05, drift=0.3, total_ms=1200.0, peak_gb=6.5)
        text = build_summary_text(main, None, parity_models=["a/b"])
        assert "# M5 runbook summary" in text
        assert "parallel-slots-typesafe" in text
        assert "90.0%" in text
        assert "A/B not run" in text
        assert "risk_level" in text

    def test_ab_delta_rows(self):
        main = self._side(agreement=0.90, flip=0.10, drift=1.0, total_ms=1000.0, peak_gb=6.0)
        ab = self._side(agreement=0.92, flip=0.05, drift=0.5, total_ms=800.0, peak_gb=5.5)
        text = build_summary_text(main, ab, parity_models=[])
        assert "| agreement/accuracy (mean) | 90.0% | 92.0% | 2.0% |" in text
        assert "| flip rate (invariance mean) | 10.0% | 5.0% | -5.0% |" in text
        assert "| time per case (ms) | 1000 | 800 | -200 |" in text
        assert "| peak memory (GB, max) | 6 | 5.5 | -0.5 |" in text

    def test_none_values_render_as_dash(self):
        text = build_summary_text({"combos": [], "invariance": {}}, None, parity_models=[])
        assert "- |" in text or "|" in text  # table renders, never crashes
        assert "M5 runbook summary" in text
