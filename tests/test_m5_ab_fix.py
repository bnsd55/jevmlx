"""M5 CRITICAL B2+B3: A/B worktree from origin/<branch>; failed A/B setup
skips A/B steps instead of crashing.

B2: benchmarks/m5.py ran 'git worktree add --detach <path> w2a-field-local'
but on the M5 clone only origin/w2a-field-local exists (no local branch).
Fix: _resolve_ab_ref tries the local branch first, then origin/<branch>.

B3: after B2 failed, the A/B bench step ran with cwd=ab-worktree (which did
not exist) -> uncaught FileNotFoundError killed the runbook before SUMMARY.
Fix at the owning layer:
  (a) a step whose declared cwd is missing is recorded as failed (exit -1),
      never raised;
  (b) when the A/B setup step fails, every later A/B step is SKIPPED with a
      RUNBOOK line 'skipped: A/B setup failed', and SUMMARY still runs for
      the main side with an 'A/B: not run (setup failed)' note;
  (c) any unexpected exception inside a step is caught, recorded with its
      type+message, and the runbook continues to the next non-dependent step;
  (d) exit code of m5 stays non-zero when any step failed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import benchmarks.m5 as m5
from benchmarks.m5 import Step, _resolve_ab_ref, build_summary_text, execute_step, runbook_append

QUALITY_TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"


# ---- B2: _resolve_ab_ref ---------------------------------------------------


def test_resolve_ab_ref_prefers_local_branch(tmp_path):
    """When a local branch exists, it is used (not origin/<branch>)."""
    repo = _make_git_repo(tmp_path)
    _make_commit(repo)
    subprocess.run(["git", "branch", "feature-x"], cwd=repo, check=True, capture_output=True)

    # Patch REPO_ROOT so _resolve_ab_ref uses our temp repo.
    orig_root = m5.REPO_ROOT
    m5.REPO_ROOT = repo
    try:
        ref = _resolve_ab_ref("feature-x")
    finally:
        m5.REPO_ROOT = orig_root
    assert ref == "feature-x"


def test_resolve_ab_ref_falls_back_to_origin(tmp_path):
    """When only origin/<branch> exists (remote-tracking ref), it is used.

    This is the M5 field failure: the branch exists only as a remote-tracking
    ref on the clone, not as a local branch.
    """
    repo = _make_git_repo(tmp_path)
    _make_commit(repo)
    # Create a remote-tracking ref: add a remote pointing to a bare repo that
    # has the branch, then fetch.
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    # Make a commit on the branch in the main repo, push it to the bare remote.
    _make_commit(repo)
    subprocess.run(["git", "branch", "w2a-field-local"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "push", "origin", "w2a-field-local"], cwd=repo, check=True, capture_output=True
    )
    # Delete the local branch so only the remote-tracking ref remains.
    subprocess.run(
        ["git", "branch", "-D", "w2a-field-local"], cwd=repo, check=True, capture_output=True
    )

    orig_root = m5.REPO_ROOT
    m5.REPO_ROOT = repo
    try:
        ref = _resolve_ab_ref("w2a-field-local")
    finally:
        m5.REPO_ROOT = orig_root
    assert ref == "origin/w2a-field-local"


def test_resolve_ab_ref_returns_bare_name_when_neither_resolves(tmp_path):
    """If neither local nor origin/<branch> resolves, return the bare name so
    'git worktree add' emits the real error (a missing ref), not our guess."""
    repo = _make_git_repo(tmp_path)
    orig_root = m5.REPO_ROOT
    m5.REPO_ROOT = repo
    try:
        ref = _resolve_ab_ref("nonexistent-branch")
    finally:
        m5.REPO_ROOT = orig_root
    assert ref == "nonexistent-branch"


# ---- B3(a): missing cwd ----------------------------------------------------


def test_execute_step_missing_cwd_recorded_not_raised(tmp_path):
    """A step whose declared cwd does not exist is recorded as failed
    (exit -1), never raised."""
    step = Step(
        id="ab-bench",
        title="A/B bench",
        argv=("echo", "hello"),
        outputs=(tmp_path / "ab-bench.done",),
        cwd=tmp_path / "does-not-exist",
    )
    # Must NOT raise.
    rc = execute_step(step)
    assert rc == -1
    # The failure is recorded in the log.
    log = (tmp_path / "ab-bench.log").read_text()
    assert "cwd does not exist" in log


# ---- B3(c): unexpected exception caught ------------------------------------


def test_execute_step_exception_caught_and_recorded(tmp_path, monkeypatch):
    """An unexpected exception inside a step is caught, recorded with its
    type+message, and the runbook continues (exit -1).

    The exception is caught in main()'s step loop (B3c); execute_step itself
    propagates, but the loop wraps it so the runbook never crashes.
    """
    from tests.test_m5_e2e import TestM5MainEndToEnd

    out = tmp_path / "run-exc"
    out.mkdir()
    calls: list[tuple[str, int]] = []
    fake_run = TestM5MainEndToEnd._fake_run_factory(out, calls)

    def exploding_run(argv, **kwargs):
        # The timing step raises mid-flight (an unexpected exception).
        if "benchmarks.timing" in " ".join(argv):
            raise RuntimeError("unexpected kaboom")
        return fake_run(argv, **kwargs)

    monkeypatch.setattr(m5, "subprocess", TestM5MainEndToEnd._fake_subprocess(exploding_run))
    monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)

    rc = m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
    # B3(d): exit non-zero (a step failed).
    assert rc == 1
    # The exception was recorded in the step's log.
    log = (out / "timing.log").read_text()
    assert "EXCEPTION (RuntimeError)" in log
    assert "unexpected kaboom" in log
    # The runbook continued past the exception to the summary.
    runbook = (out / "RUNBOOK.md").read_text()
    assert "Traceback" not in runbook
    assert (out / "SUMMARY.md").exists()


# ---- B3(b): A/B setup failure skips later A/B steps ------------------------


def test_ab_setup_failure_skips_ab_steps_and_summaries(tmp_path, monkeypatch):
    """When the A/B setup step fails, every later A/B step is SKIPPED with a
    RUNBOOK line, SUMMARY still runs for the main side with 'A/B: not run
    (setup failed)', and m5 exits non-zero.

    Uses the real plan_steps + execute_step + runbook_append + main() loop
    with subprocess.run monkeypatched: the main-side steps succeed, the
    ab-setup step fails (its 'git worktree add' returns rc=1), and the
    ab-bench / ab-invariance steps are skipped.
    """
    from tests.test_m5_e2e import TestM5MainEndToEnd

    out = tmp_path / "run-ab-fail"
    out.mkdir()
    calls: list[tuple[str, int]] = []
    fake_run = TestM5MainEndToEnd._fake_run_factory(out, calls)

    def ab_failing_run(argv, **kwargs):
        joined = " ".join(argv)
        # The ab-setup step runs 'git worktree add --detach <path> <ref>'.
        if "worktree" in joined and "add" in joined:
            calls.append(("ab-setup", 1))
            return subprocess.CompletedProcess(tuple(argv), 1, stdout="", stderr="bad ref")
        return fake_run(argv, **kwargs)

    monkeypatch.setattr(m5, "subprocess", TestM5MainEndToEnd._fake_subprocess(ab_failing_run))
    monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)

    rc = m5.main(
        [
            "--out",
            str(out),
            "--parity-models",
            QUALITY_TARGET,
            "--ab-branch",
            "w2a-field-local",
            "--allow-sleep",
        ]
    )
    # B3(d): exit non-zero (a step failed).
    assert rc == 1

    runbook = (out / "RUNBOOK.md").read_text()
    # The ab-setup step is recorded as failed.
    assert "ab-setup" in runbook or "A/B setup" in runbook
    assert "exit 1" in runbook
    # Later A/B steps are skipped (not run, not failed).
    assert "skipped: A/B setup failed" in runbook
    # No traceback leaked.
    assert "Traceback" not in runbook

    # B3(b): SUMMARY still ran for the main side.
    summary_path = out / "SUMMARY.md"
    assert summary_path.exists(), "SUMMARY.md must be written even when A/B setup failed"
    summary = summary_path.read_text()
    assert "# M5 runbook summary" in summary
    assert "## Main — bench combos" in summary
    # The A/B section says 'not run (setup failed)', not all-dashes.
    assert "A/B: not run (setup failed)" in summary

    # The ab-failed marker exists.
    assert (out / "ab-failed.txt").exists()


def test_ab_setup_success_runs_ab_steps(tmp_path, monkeypatch):
    """When A/B setup succeeds, the ab-bench and ab-invariance steps DO run
    (the skip logic does not fire). This guards against over-skipping."""
    from tests.test_m5_e2e import TestM5MainEndToEnd

    out = tmp_path / "run-ab-ok"
    out.mkdir()
    calls: list[tuple[str, int]] = []
    fake_run = TestM5MainEndToEnd._fake_run_factory(out, calls)

    def ab_ok_run(argv, **kwargs):
        joined = " ".join(argv)
        # ab-setup: git worktree add + uv venv + uv pip install -> all exit 0.
        if "worktree" in joined and "add" in joined:
            # Create the worktree dir so later steps' cwd exists.
            wt_path = Path(argv[argv.index("--detach") + 1])
            wt_path.mkdir(parents=True, exist_ok=True)
            calls.append(("ab-setup", 0))
            return subprocess.CompletedProcess(tuple(argv), 0, stdout="", stderr="")
        if "uv" in argv[0].split("/")[-1]:
            calls.append(("uv", 0))
            return subprocess.CompletedProcess(tuple(argv), 0, stdout="", stderr="")
        return fake_run(argv, **kwargs)

    monkeypatch.setattr(m5, "subprocess", TestM5MainEndToEnd._fake_subprocess(ab_ok_run))
    monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)

    rc = m5.main(
        [
            "--out",
            str(out),
            "--parity-models",
            QUALITY_TARGET,
            "--ab-branch",
            "w2a-field-local",
            "--allow-sleep",
        ]
    )
    # All steps succeeded.
    assert rc == 0, f"runbook failed: {calls}"
    # The ab-bench and ab-invariance steps ran (not skipped).
    runbook = (out / "RUNBOOK.md").read_text()
    assert "skipped: A/B setup failed" not in runbook


# ---- B3(b) summary note ----------------------------------------------------


def test_build_summary_text_ab_failed_note():
    """build_summary_text shows 'A/B: not run (setup failed)' when ab_failed."""
    main = {"combos": [], "parity_status": "-"}
    text = build_summary_text(main, None, parity_models=[], ab_failed=True)
    assert "A/B: not run (setup failed)" in text


def test_build_summary_text_no_ab_branch_note():
    """build_summary_text shows 'A/B not run (no --ab-branch)' when ab_failed
    is False and ab is None (the default, no --ab-branch given)."""
    main = {"combos": [], "parity_status": "-"}
    text = build_summary_text(main, None, parity_models=[], ab_failed=False)
    assert "A/B not run (no --ab-branch)" in text


# ---- runbook_append skipped flag -------------------------------------------


def test_runbook_append_skipped(tmp_path):
    """runbook_append with skipped=True writes 'skipped: A/B setup failed'."""
    step = Step(
        id="ab-bench",
        title="A/B bench quality",
        argv=("echo", "hello"),
        outputs=(tmp_path / "ab-bench.done",),
    )
    runbook_append(tmp_path, 1, step, rc=None, secs=None, skipped=True)
    text = (tmp_path / "RUNBOOK.md").read_text()
    assert "skipped: A/B setup failed" in text


# ---- helpers ---------------------------------------------------------------


def _make_git_repo(path: Path) -> Path:
    """Create a real git repo at path and return it."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True
    )
    return path


def _make_commit(repo: Path) -> None:
    """Make a trivial commit so the repo has a HEAD."""
    (repo / "file.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
