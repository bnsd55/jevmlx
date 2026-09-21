"""E2e test: the results-PR step regenerates the README leaderboard and
commits a fresh block.

Uses a real temp git repo with a real results folder fixture written by the
real eval writers (run_eval + write_report + compute_metrics) — no hand-made
fixture, no fake subprocess.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from benchmarks.m5 import Step, _execute_results_pr
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


def _init_git_repo(path: Path) -> None:
    """Init a git repo with a README.md (with leaderboard markers)."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True
    )

    # Write a README.md with leaderboard markers (the block is empty initially).
    readme = path / "README.md"
    readme.write_text(
        "# jevmlx\n\n<!-- leaderboard:start -->\n| empty |\n<!-- leaderboard:end -->\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_results_pr_step_regenerates_readme_and_commits(tmp_path, monkeypatch):
    """The results-PR step:
    1. regenerates the README leaderboard block,
    2. verifies it is fresh (check_results --check-readme passes),
    3. commits README.md with the fresh block.

    Uses a real temp git repo + real results written by run_eval +
    write_report + compute_metrics.
    """
    import benchmarks.m5 as m5

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    # Write the official.json (the leaderboard needs it for agreement).
    official_dir = repo_root / "benchmarks" / "typesafe"
    official_dir.mkdir(parents=True, exist_ok=True)
    (official_dir / "official.json").write_text("{}", encoding="utf-8")

    # Write the results folder with real writers.
    model_folder = "test-host-mlx-community--qwen2.5-7b-instruct-4bit"
    _write_results_folder(repo_root, model_folder)

    # Patch REPO_ROOT so the step runs in our temp repo.
    monkeypatch.setattr(m5, "REPO_ROOT", repo_root)

    step = Step(
        id="results-pr",
        title="Regenerate README leaderboard + commit results + open PR",
        argv=(),
        outputs=(repo_root / "README.md",),
        in_process=True,
    )

    # Capture the log output (use a real file — subprocess.run needs fileno).
    log_path = tmp_path / "results-pr.log"
    log = log_path.open("w", encoding="utf-8")
    rc = _execute_results_pr(step, log)
    log.close()
    log_text = log_path.read_text(encoding="utf-8")
    assert rc == 0, f"results-pr step failed:\n{log_text}"

    # The README.md now has a non-empty leaderboard block (not the initial
    # '| empty |' placeholder).
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "<!-- leaderboard:start -->" in readme
    assert "<!-- leaderboard:end -->" in readme
    assert "| empty |" not in readme

    # The git commit contains README.md.
    diff = subprocess.run(
        ["git", "diff", "HEAD~1", "--name-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "README.md" in diff, f"README.md not in commit: {diff}"


def test_results_pr_flag_adds_step_to_plan(tmp_path):
    """The --results-pr flag adds the results-pr step to plan_steps."""
    from benchmarks.m5 import plan_steps

    out = tmp_path / "run"
    out.mkdir()

    # Without the flag: no results-pr step.
    steps = plan_steps(out, parity_models=[QUALITY_TARGET], results_pr=False)
    assert not any(s.id == "results-pr" for s in steps)

    # With the flag: the results-pr step is the last step.
    steps = plan_steps(out, parity_models=[QUALITY_TARGET], results_pr=True)
    results_steps = [s for s in steps if s.id == "results-pr"]
    assert len(results_steps) == 1
    assert results_steps[0].in_process is True
    assert results_steps[0].argv == ()
