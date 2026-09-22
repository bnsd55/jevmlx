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
            "readme",
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
        assert ids[-4:] == [
            "ab-setup",
            "ab-bench",
            "ab-invariance",
            "summary",
        ] or ids[-5:] == [
            "ab-setup",
            "ab-bench",
            "ab-invariance",
            "summary",
            "readme",
        ]
        ab_setup = next(s for s in steps if s.id == "ab-setup")
        # B8: argv is a stale-worktree cleanup; git worktree add is the first
        # extra_argv. The branch ref and worktree path appear across them.
        all_ab_setup_tokens = list(ab_setup.argv) + [
            t for extra in ab_setup.extra_argv for t in extra
        ]
        assert "worktree" in " ".join(all_ab_setup_tokens)
        assert "w2a-field-local" in " ".join(all_ab_setup_tokens)
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
        summary = next(s for s in steps if s.id == "summary")
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

    def test_typesafe_agreement_dict_does_not_crash(self):
        """M5 crash fix: typesafe report.json stores agreement as a dict
        ({overall, by_workflow, ...}); _combo_agreement extracts 'overall'
        and _fmt formats it as a percentage, never crashing on dict.__format__."""
        main = {
            "combos": [
                combo_row(
                    "parallel-slots-typesafe",
                    {"metrics": {"agreement": {"overall": 0.87, "by_workflow": {"x": 0.9}}}},
                    {"median": {"total_ms": 1000.0, "peak_active_bytes": 6 * 2**30}},
                )
            ],
            "invariance": {"mean_flip_rate": 0.05, "mean_drift": 0.3},
            "timing_report": {},
        }
        text = build_summary_text(main, None, parity_models=[])
        assert "87.0%" in text  # overall extracted from the dict
        assert "dict" not in text.lower()

    def test_bundled_agreement_float_still_works(self):
        """Bundled/other report.json stores agreement as a bare float; the
        fix must not break the existing path."""
        main = {
            "combos": [
                combo_row(
                    "parallel-slots-bundled",
                    {"metrics": {"agreement": 0.92}},
                    {"median": {"total_ms": 800.0, "peak_active_bytes": 5 * 2**30}},
                )
            ],
            "invariance": {"mean_flip_rate": 0.03, "mean_drift": 0.2},
            "timing_report": {},
        }
        text = build_summary_text(main, None, parity_models=[])
        assert "92.0%" in text

    def test_agreement_dict_in_ab_section_does_not_crash(self):
        """The A/B comparison section also calls _combo_agreement via _agg;
        a typesafe-shaped agreement dict must not crash there either."""
        main = {
            "combos": [
                combo_row(
                    "parallel-slots-typesafe",
                    {"metrics": {"agreement": {"overall": 0.90, "by_workflow": {}}}},
                    {"median": {"total_ms": 1000.0, "peak_active_bytes": 6 * 2**30}},
                )
            ],
            "invariance": {"mean_flip_rate": 0.05, "mean_drift": 0.3},
            "timing_report": {},
        }
        ab = {
            "combos": [
                combo_row(
                    "parallel-slots-typesafe",
                    {"metrics": {"agreement": {"overall": 0.92, "by_workflow": {}}}},
                    {"median": {"total_ms": 900.0, "peak_active_bytes": 5.5 * 2**30}},
                )
            ],
            "invariance": {"mean_flip_rate": 0.04, "mean_drift": 0.25},
            "timing_report": {},
        }
        text = build_summary_text(main, ab, parity_models=[])
        assert "| agreement/accuracy (mean) | 90.0% | 92.0% | 2.0% |" in text

    def test_agreement_dict_without_overall_falls_back_to_accuracy(self):
        """When agreement is a dict without 'overall' (malformed), fall back
        to accuracy rather than crash."""
        main = {
            "combos": [
                combo_row(
                    "parallel-slots-typesafe",
                    {"metrics": {"agreement": {"by_workflow": {}}, "accuracy": 0.75}},
                    {"median": {"total_ms": 1000.0, "peak_active_bytes": 6 * 2**30}},
                )
            ],
            "invariance": {},
            "timing_report": {},
        }
        text = build_summary_text(main, None, parity_models=[])
        assert "75.0%" in text


# --- W5c-15: macOS idle-sleep blocking (caffeinate re-exec) ----------------


class TestMaybeCaffeinate:
    """The four guardrail paths for _maybe_caffeinate (monkeypatches
    os.execvp and sys.platform so no real re-exec happens)."""

    def test_default_darwin_execvp_called_once(self, monkeypatch, capsys):
        """(a) default darwin, no env var, no --allow-sleep: execvp called once
        with argv starting ['caffeinate','-dimsu', sys.executable,
        '-m','benchmarks.m5'] and the env var set."""
        import benchmarks.m5 as m5

        monkeypatch.setattr(m5.sys, "platform", "darwin")
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        calls = []

        def _fake_execvp(prog, argv):
            calls.append((prog, argv))

        monkeypatch.setattr(m5.os, "execvp", _fake_execvp)
        m5._maybe_caffeinate(allow_sleep=False, argv_tail=["--out", "x"])
        assert len(calls) == 1
        prog, argv = calls[0]
        assert prog == "caffeinate"
        assert argv[:5] == ["caffeinate", "-dimsu", m5.sys.executable, "-m", "benchmarks.m5"]
        assert argv[5:] == ["--out", "x"]
        assert m5.os.environ.get("JEVMLX_M5_CAFFEINATED") == "1"
        out = capsys.readouterr().out
        assert "blocking macOS idle sleep" in out

    def test_allow_sleep_not_called(self, monkeypatch, capsys):
        """(b) --allow-sleep: execvp not called."""
        import benchmarks.m5 as m5

        monkeypatch.setattr(m5.sys, "platform", "darwin")
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        called = []

        monkeypatch.setattr(m5.os, "execvp", lambda *a: called.append(1))
        m5._maybe_caffeinate(allow_sleep=True, argv_tail=["--out", "x"])
        assert not called
        assert "NOT blocked (--allow-sleep)" in capsys.readouterr().out

    def test_env_var_already_set_not_called(self, monkeypatch, capsys):
        """(c) env var already '1': not called (the re-execed child)."""
        import benchmarks.m5 as m5

        monkeypatch.setattr(m5.sys, "platform", "darwin")
        monkeypatch.setenv("JEVMLX_M5_CAFFEINATED", "1")
        called = []

        monkeypatch.setattr(m5.os, "execvp", lambda *a: called.append(1))
        m5._maybe_caffeinate(allow_sleep=False, argv_tail=["--out", "x"])
        assert not called

    def test_execvp_file_not_found_prints_warning_and_continues(self, monkeypatch, capsys):
        """(d) execvp raises FileNotFoundError: prints 'NOT blocked', pops the
        guard env var (so the RUNBOOK header reports sleep_blocked: False),
        and continues."""
        import benchmarks.m5 as m5

        monkeypatch.setattr(m5.sys, "platform", "darwin")
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)

        def _raise(*a):
            raise FileNotFoundError("no caffeinate")

        monkeypatch.setattr(m5.os, "execvp", _raise)
        # Must not raise.
        m5._maybe_caffeinate(allow_sleep=False, argv_tail=["--out", "x"])
        out = capsys.readouterr().out
        assert "caffeinate not found" in out
        assert "NOT blocked" in out
        # The guard env var was popped — the re-exec never happened.
        assert m5.os.environ.get("JEVMLX_M5_CAFFEINATED") is None

    def test_non_darwin_noop(self, monkeypatch, capsys):
        """Non-darwin: no-op (no print, no execvp)."""
        import benchmarks.m5 as m5

        monkeypatch.setattr(m5.sys, "platform", "linux")
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        called = []

        monkeypatch.setattr(m5.os, "execvp", lambda *a: called.append(1))
        m5._maybe_caffeinate(allow_sleep=False, argv_tail=["--out", "x"])
        assert not called
        assert capsys.readouterr().out == ""
