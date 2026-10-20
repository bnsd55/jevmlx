"""The A/B setup installs THE A/B WORKTREE, and the guards make a
main-repo install impossible to repeat.

Field evidence (M5, issue #63, 15:08 UTC): ab-setup.log ran
``uv pip install ... -e .[dev]`` with cwd = the main repo, so the A/B venv
got jevmlx from MAIN (dbb1ff1); the A/B run.json recorded
``config.prompt_version = jevmlx-parallel-v9`` although the w2a branch only
knows v8 — every A/B so far measured main twice.

The fix, guarded at both ends:
  (1) ab-setup installs ``-e <worktree>[dev,bench]`` with cwd=<worktree>
      (never relative to the caller's cwd);
  (2) GUARD 1: after install, ``<ab-venv>/python -I -c`` imports jevmlx and
      FAILS the step unless the resolved path is inside the A/B worktree
      (-I keeps the import off the cwd, so it cannot mask a bad install);
  (3) GUARD 2: after the A/B bench, the combos' run.json
      ``environment.git_sha`` are compared against the main side — an equal
      sha fails the step with 'A/B measured the base code' and the remaining
      A/B steps are skipped;
  (4) ``--ab-eval-args`` ride the ab-bench + ab-invariance argvs verbatim.

The e2e tests here run on a REAL temp git repo with TWO commits and REAL
venvs: the guard must fail when the venv resolves the base repo and pass
when it resolves the worktree. The venv's import path is seeded the way an
editable install does it (a .pth line naming the source dir), so no
network/pip is involved — the resolution path the guard inspects is real.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import benchmarks.m5 as m5
from benchmarks.m5 import ab_enforce_base_guard, ab_git_shas, ab_measured_base

QUALITY_TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"


@pytest.fixture(autouse=True)
def _planned_root(tmp_path, monkeypatch):
    """Point the planner's REPO_ROOT at a temp repo (same guard as
    TestM5MainEndToEnd and test_m5_ab_fix): every m5.main() here fires the
    readme step, which regenerates the PLANNED root's README — left on the
    module global it wrote the real checkout's README.md. The dir must
    exist: execute_step fails any step whose planned cwd is missing."""
    planned_root = tmp_path / "planned-repo"
    planned_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(m5, "REPO_ROOT", planned_root)


# --------------------------------------------------------------- repo helpers


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


def _commit_all(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _base_repo_with_two_commits(tmp_path: Path) -> tuple[Path, Path]:
    """A real temp git repo (commit 1 = base) + a worktree at commit 2.

    The package under test is a minimal ``jevmlx`` (an __init__ that records
    its origin): the guards check WHERE the import resolves, not what it
    contains, so the repo stays offline-safe. Returns (base_repo, worktree).
    """
    repo = tmp_path / "base-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repo, check=True, capture_output=True
    )
    (repo / "jevmlx").mkdir()
    (repo / "jevmlx" / "__init__.py").write_text('ORIGIN = "base"\n', encoding="utf-8")
    _commit_all(repo, "base commit")
    # Worktree at a SECOND commit — the A/B side of the story.
    (repo / "jevmlx" / "__init__.py").write_text('ORIGIN = "ab"\n', encoding="utf-8")
    subprocess.run(["git", "checkout", "-b", "ab"], cwd=repo, check=True, capture_output=True)
    _commit_all(repo, "ab commit")
    worktree = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "ab"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo, worktree


def _venv_with_pth(venv_dir: Path, package_dir: Path) -> Path:
    """A real venv whose site-packages has a .pth resolving jevmlx to
    package_dir — what an editable install of that directory produces."""
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv_dir)],
        check=True,
        capture_output=True,
    )
    site = next(venv_dir.glob("lib/python*/site-packages"))
    (site / "_m5_ab_install.pth").write_text(f"{package_dir}\n", encoding="utf-8")
    return venv_dir / "bin" / "python"


def _run_guard(venv_python: Path, snippet: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run the REAL guard snippet (the ab-setup extra_argv command) with the
    given venv python and cwd."""
    return subprocess.run(
        [str(venv_python), "-I", "-c", snippet], cwd=cwd, capture_output=True, text=True
    )


# ---- GUARD 1: the post-install import must resolve inside the worktree ------


class TestInstallGuardE2E:
    """GUARD 1 end to end: real temp git repo (two commits), real venvs, the
    real snippet text m5 plans. The install step's own argv is asserted in
    tests/test_m5.py::test_ab_steps_branch_worktree_and_cwd."""

    def test_fails_when_venv_resolves_the_base_repo(self, tmp_path):
        """The field failure: the venv holds MAIN's jevmlx. The guard must
        FAIL the step (nonzero exit, worktree named in the message)."""
        repo, worktree = _base_repo_with_two_commits(tmp_path)
        venv = tmp_path / "venv-main"
        venv_python = _venv_with_pth(venv, repo)  # .pth -> the BASE repo
        proc = _run_guard(venv_python, m5._ab_install_guard_snippet(worktree), cwd=tmp_path)
        assert proc.returncode != 0, f"guard passed on a main-repo install: {proc.stdout}"
        assert str(worktree) in proc.stderr
        assert "ab-install guard FAILED" in proc.stderr

    def test_passes_when_venv_resolves_the_worktree(self, tmp_path):
        """The fixed install: the venv's jevmlx comes from the worktree. The
        guard PASSES. cwd = the base repo proves -I keeps the cwd off
        sys.path — without -I, importing from cwd = base would win and the
        guard would false-fail exactly like the field bug."""
        repo, worktree = _base_repo_with_two_commits(tmp_path)
        venv = tmp_path / "venv-wt"
        venv_python = _venv_with_pth(venv, worktree)
        proc = _run_guard(venv_python, m5._ab_install_guard_snippet(worktree), cwd=repo)
        assert proc.returncode == 0, f"guard failed on a worktree install: {proc.stderr}"
        resolved = proc.stdout.strip().splitlines()[-1]
        assert resolved.startswith(str(worktree))

    def test_fails_when_jevmlx_missing_from_venv(self, tmp_path):
        """No install at all -> ImportError -> nonzero (never a silent pass)."""
        _, worktree = _base_repo_with_two_commits(tmp_path)
        bare_venv = tmp_path / "venv-empty"
        subprocess.run(
            ["python3", "-m", "venv", "--without-pip", str(bare_venv)],
            check=True,
            capture_output=True,
        )
        proc = _run_guard(
            bare_venv / "bin" / "python", m5._ab_install_guard_snippet(worktree), cwd=tmp_path
        )
        assert proc.returncode != 0


# ---- GUARD 2: run.json git_sha comparison (pure functions) ------------------


def _write_run_json(bench_dir: Path, combo: str, git_sha: str | None) -> None:
    """One combo folder with a bench-shaped run.json (environment.git_sha)."""
    combo_dir = bench_dir / f"machine-slug-{combo}" / "parallel-slots-bundled"
    combo_dir.mkdir(parents=True, exist_ok=True)
    env: dict = {"git_sha": git_sha} if git_sha else {"git_sha": None}
    (combo_dir / "run.json").write_text(
        json.dumps({"environment": env, "config": {}, "counts": {}}), encoding="utf-8"
    )


def test_ab_git_shas_collects_non_null_shas(tmp_path):
    bench = tmp_path / "bench"
    _write_run_json(bench, "a", "sha-1")
    _write_run_json(bench, "b", "sha-2")
    _write_run_json(bench, "null", None)  # null never enters the set
    (bench / "broken" / "x").mkdir(parents=True)
    (bench / "broken" / "x" / "run.json").write_text("{not json", encoding="utf-8")
    assert ab_git_shas(bench) == {"sha-1", "sha-2"}


def test_ab_measured_base_true_when_shas_equal(tmp_path):
    main_bench, ab_bench = tmp_path / "main", tmp_path / "ab"
    _write_run_json(main_bench, "a", "same-sha")
    _write_run_json(ab_bench, "a", "same-sha")
    assert ab_measured_base(main_bench, ab_bench) is True


def test_ab_measured_base_false_when_branch_sha_differs(tmp_path):
    main_bench, ab_bench = tmp_path / "main", tmp_path / "ab"
    _write_run_json(main_bench, "a", "main-sha")
    _write_run_json(ab_bench, "a", "ab-sha")
    assert ab_measured_base(main_bench, ab_bench) is False


def test_ab_measured_base_false_without_evidence(tmp_path):
    main_bench, ab_bench = tmp_path / "main", tmp_path / "ab"
    # No run.json at all, or null on one side — the guard cannot judge.
    main_bench.mkdir()
    ab_bench.mkdir()
    assert ab_measured_base(main_bench, ab_bench) is False
    _write_run_json(main_bench, "a", "main-sha")
    _write_run_json(ab_bench, "a", None)
    assert ab_measured_base(main_bench, ab_bench) is False


def test_ab_enforce_base_guard_writes_and_clears_marker(tmp_path):
    """ab_enforce_base_guard maintains ab-measured-base.txt symmetrically: a
    tripped A/B writes the marker and returns True; a clean A/B REMOVES a
    stale marker (a rerun whose A/B measured the branch for real must not
    be discarded because an earlier run tripped the guard)."""
    out = tmp_path / "run"
    out.mkdir()
    _write_run_json(out / "bench-quality", "a", "main-sha")
    _write_run_json(out / "ab" / "bench-quality", "a", "ab-sha")
    stale = out / "ab-measured-base.txt"
    stale.write_text("stale from a tripped run\n", encoding="utf-8")
    assert ab_enforce_base_guard(out) is False
    assert not stale.exists()  # stale marker cleared on a clean A/B
    # Now trip it: the A/B bench measured main's sha.
    _write_run_json(out / "ab" / "bench-quality", "b", "main-sha")
    assert ab_enforce_base_guard(out) is True
    assert "A/B measured the base code" in stale.read_text(encoding="utf-8")


# ---- GUARD 2 wiring: the runbook fails the ab-bench step --------------------


def test_ab_bench_measured_base_fails_runbook(tmp_path, monkeypatch):
    """When the A/B bench's run.json git_sha equals main's, the ab-bench step
    FAILS with 'A/B measured the base code', ab-measured-base.txt exists,
    ab-invariance is skipped, SUMMARY discards the A/B side, and the exit
    code is non-zero."""
    from tests.test_m5_e2e import TestM5MainEndToEnd

    out = tmp_path / "run-base-guard"
    out.mkdir()
    calls: list[tuple[str, int]] = []
    fake_run = TestM5MainEndToEnd._fake_run_factory(out, calls)

    def base_guard_run(argv, **kwargs):
        joined = " ".join(argv)
        bin0 = argv[0].split("/")[-1]
        is_python = bin0.startswith("python")
        if is_python and "-c" in argv and "rmtree" in joined:
            return subprocess.CompletedProcess(tuple(argv), 0, stdout="", stderr="")
        if is_python and "-I" in argv and "-c" in argv:
            return subprocess.CompletedProcess(tuple(argv), 0, stdout="", stderr="")
        if bin0 == "git" and "rev-parse" in joined:
            return subprocess.CompletedProcess(tuple(argv), 0, stdout="deadbeef\n", stderr="")
        if "worktree" in joined and "add" in joined:
            Path(argv[argv.index("--detach") + 1]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(tuple(argv), 0, stdout="", stderr="")
        if "uv" in bin0:
            return subprocess.CompletedProcess(tuple(argv), 0, stdout="", stderr="")
        # The A/B bench writes REAL run.json shapes (real environment()) —
        # with git_sha equal to main's, exactly the field failure. Left
        # unpatched on purpose: GUARD 2 must trip on it.
        return fake_run(argv, **kwargs)

    monkeypatch.setattr(m5, "subprocess", TestM5MainEndToEnd._fake_subprocess(base_guard_run))
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
    assert rc == 1, f"guard 2 did not fail the runbook: {calls}"
    # The marker records the reason.
    marker = out / "ab-measured-base.txt"
    assert marker.exists()
    assert "A/B measured the base code" in marker.read_text(encoding="utf-8")
    # ab-invariance was skipped with the guard reason (not run, not failed).
    runbook = (out / "RUNBOOK.md").read_text()
    assert "skipped: A/B measured the base code" in runbook
    assert "skipped: A/B setup failed" not in runbook
    # SUMMARY ran for the main side and discards the tainted A/B numbers.
    summary = (out / "SUMMARY.md").read_text()
    assert "# M5 runbook summary" in summary
    assert "measured the base code" in summary
    assert "## Main — bench combos" in summary


# ---- summary note -----------------------------------------------------------


def test_build_summary_text_ab_base_code_note():
    """The A/B note names the base-code discard, not 'setup failed'."""
    main = {"combos": [], "parity_status": "-"}
    text = m5.build_summary_text(main, None, parity_models=[], ab_base_code=True)
    assert "measured the base code" in text
    assert "setup failed" not in text
