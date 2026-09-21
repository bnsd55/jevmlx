"""E2e test: the readme step regenerates the README leaderboard and verifies
freshness.

Uses a real results folder fixture written by the real eval writers
(run_eval + write_report + compute_metrics) — no hand-made fixture, no fake
subprocess, no temp git repo.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.m5 import Step, _execute_readme, plan_steps
from jevmlx import evalrun
from jevmlx.evalmetrics import compute_metrics, load_predictions
from jevmlx.evalreport import write_report

QUALITY_TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def _make_cases() -> list[dict]:
    """A tiny bundled-style cases dataset (2 cases, 1 enum field)."""
    return [
        {
            "id": "case-1",
            "group_id": "case-1",
            "schema": {"risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"}},
            "context": "The applicant pays late.",
            "labels": {"risk": "LOW"},
            "split": "test",
            "meta": {},
        },
        {
            "id": "case-2",
            "group_id": "case-2",
            "schema": {"risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"}},
            "context": "The applicant pays on time.",
            "labels": {"risk": "LOW"},
            "split": "test",
            "meta": {},
        },
    ]


def _write_results_folder(repo_root: Path, model_folder: str) -> Path:
    """Write a real results folder using the actual eval writers.

    Creates ``benchmarks/results/<model_folder>/bundled.dataset.lock.json``
    + a combo folder with predictions.jsonl, run.json, report.json — the
    same shape the bench writes.
    """
    results_root = repo_root / "benchmarks" / "results" / model_folder
    results_root.mkdir(parents=True, exist_ok=True)

    # Write the dataset lock (the bench writes it at the MODEL folder level).
    lock_path = results_root / "bundled.dataset.lock.json"
    lock_data = {
        "dataset": "bundled",
        "cases_sha256": "abc123",
        "path": str(repo_root / "benchmarks" / "bundled.jsonl"),
    }
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")

    combo_dir = results_root / "parallel-slots-bundled"
    combo_dir.mkdir(exist_ok=True)

    cases = _make_cases()

    def decide_fn(schema_dict, context, constraints=None, oracle_overrides=None):
        return {"risk": {"prediction": "LOW", "probability": 0.9, "valid": True}}

    evalrun.run_eval(
        cases,
        decide_fn,
        track="parallel",
        model="fake",
        out_dir=str(combo_dir),
        dataset_lock_path=str(lock_path),
        dataset_path=str(repo_root / "benchmarks" / "bundled.jsonl"),
    )

    records = load_predictions(combo_dir / "predictions.jsonl")
    write_report(
        combo_dir / "report.json", {"environment": {}, "metrics": compute_metrics(records)}
    )

    return results_root


def test_readme_step_regenerates_and_verifies(tmp_path, monkeypatch):
    """The readme step:
    1. regenerates the README leaderboard block,
    2. verifies it is fresh (check_results --check-readme passes).

    Uses a real results folder written by run_eval + write_report +
    compute_metrics (no hand-made fixture, no fake subprocess).
    """
    import benchmarks.m5 as m5

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    # Write the official.json (the leaderboard needs it for agreement).
    official_dir = repo_root / "benchmarks" / "typesafe"
    official_dir.mkdir(parents=True, exist_ok=True)
    (official_dir / "official.json").write_text("{}", encoding="utf-8")

    # Write a README.md with leaderboard markers (the block is a stale placeholder).
    readme = repo_root / "README.md"
    readme.write_text(
        "# jevmlx\n\n<!-- leaderboard:start -->\n| stale |\n<!-- leaderboard:end -->\n",
        encoding="utf-8",
    )

    # Write the results folder with real writers.
    model_folder = "test-host-mlx-community--qwen2.5-7b-instruct-4bit"
    _write_results_folder(repo_root, model_folder)

    # Safety assertion: the real repo README.md must be unchanged after the test.
    import hashlib

    real_readme = Path(__file__).resolve().parent.parent / "README.md"
    real_readme_sha_before = hashlib.sha256(real_readme.read_bytes()).hexdigest()

    # Patch REPO_ROOT so the step runs in our temp repo.
    monkeypatch.setattr(m5, "REPO_ROOT", repo_root)

    step = Step(
        id="readme",
        title="Regenerate README leaderboard and verify freshness",
        argv=(),
        outputs=(repo_root / "README.md",),
        in_process=True,
    )

    # Use a real file for the log (subprocess.run needs fileno).
    log_path = tmp_path / "readme.log"
    log = log_path.open("w", encoding="utf-8")
    rc = _execute_readme(step, log)
    log.close()
    log_text = log_path.read_text(encoding="utf-8")
    assert rc == 0, f"readme step failed:\n{log_text}"

    # The README.md now has a fresh leaderboard block (not the stale placeholder).
    readme_text = readme.read_text(encoding="utf-8")
    assert "<!-- leaderboard:start -->" in readme_text
    assert "<!-- leaderboard:end -->" in readme_text
    assert "| stale |" not in readme_text

    # The real repo README.md was NOT touched by the test.
    real_readme_sha_after = hashlib.sha256(real_readme.read_bytes()).hexdigest()
    assert real_readme_sha_before == real_readme_sha_after, (
        "the real repo README.md was modified by the test"
    )


def test_readme_step_appears_last_in_plan_by_default(tmp_path):
    """The readme step is always appended after summary (no flag needed)."""
    out = tmp_path / "run"
    out.mkdir()

    steps = plan_steps(out, parity_models=[QUALITY_TARGET])
    assert steps, "plan_steps returned no steps"
    # The readme step is present.
    readme_steps = [s for s in steps if s.id == "readme"]
    assert len(readme_steps) == 1
    assert readme_steps[0].in_process is True
    assert readme_steps[0].argv == ()
    # It is the last step (probe steps are optional and appended after).
    # When probe=False, readme is the last step.
    assert steps[-1].id == "readme"
