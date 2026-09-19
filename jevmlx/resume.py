"""Crash-safe resumable eval infrastructure (W5c-7 / B6 part 1).

Three concerns, one module:

- **Immutable run manifest** (``manifest.json``): written at the START of a
  run, before any model work. Captures everything that must NOT change
  across a resume: config, dataset lock sha, code hash (git HEAD + dirty
  flag), model id + revision, tokenizer revision, prompt version + sha,
  and the machine (chip, macOS, mlx/mlx-lm versions). ``--resume`` refuses
  to mix a run whose manifest differs on any of these fields.

- **One-writer lock** (``.lock``): a file-based lock per results folder.
  Acquired at run start; released on clean exit. A stale lock (owner PID
  not alive) is breakable so a crashed run can be resumed.

- **Crash-safe per-case commit** (``predictions.jsonl`` + a journal): a
  case is complete only when ALL its prediction lines are durably written.
  Predictions are appended per-case (not all at the end), and each case's
  lines land atomically via a temp blob + rename-free append guarded by
  the lock. A ``completed_cases.jsonl`` journal records each committed
  case key so ``--resume`` can skip them without re-reading the full
  predictions file.

The contract (from GPT-REVIEW-3 B6): a case becomes complete only after
all its prediction lines are durably written. Appending field lines
directly can leave a case half-complete after a crash — so each case is
written as one atomic blob (all its lines in one ``write`` + ``flush`` +
``fsync``), and the journal entry lands AFTER the blob is durable.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "JOURNAL_PATH",
    "LOCK_PATH",
    "MANIFEST_PATH",
    "CircuitBreaker",
    "ManifestMismatchError",
    "ResultsLock",
    "build_manifest",
    "case_key",
    "circuit_breaker_threshold",
    "completed_case_keys",
    "failure_signature",
    "last_committed_offset",
    "load_manifest",
    "machine_probe",
    "truncate_to_last_commit",
    "write_case_blob",
    "write_manifest",
    "verify_manifest_matches",
]

MANIFEST_PATH = "manifest.json"
LOCK_PATH = ".lock"
JOURNAL_PATH = "completed_cases.jsonl"

# B6: trip the infra breaker after this many consecutive failures with
# the same broad signature. Five is the review's number.
circuit_breaker_threshold = 5


# ---------------------------------------------------------------------------
# Code hash
# ---------------------------------------------------------------------------


def _git_code_hash() -> dict[str, Any]:
    """git HEAD sha + dirty flag for the jevmlx checkout.

    W5c-7 review: uses the PACKAGE directory (where this file lives), not
    the process cwd — the process cwd may be / or a results dir, returning
    None,None and making the manifest useless for resume verification.
    Falls back to None when not in a git checkout (e.g. site-packages).
    """
    import jevmlx

    pkg_dir = os.path.dirname(os.path.abspath(jevmlx.__file__))
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=pkg_dir,
        )
        if head.returncode != 0:
            return {"git_sha": None, "git_dirty": None}
        sha = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=pkg_dir,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return {"git_sha": sha, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    config: dict[str, Any],
    dataset_lock_sha256: str | None,
    prompt_version: str | None,
    prompt_sha256: str | None,
    model_id: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    tokenizer_chat_template_sha256: str | None,
    machine: dict[str, Any],
    code_hash: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the immutable run manifest.

    The manifest is the resume contract: ``--resume`` refuses if ANY of
    these fields differ from the on-disk manifest. The fields captured are
    exactly those that would make a resumed run's results non-comparable:
    config (model/track/split/permutations/temperature), the dataset lock
    sha (the data didn't change), the code hash (the engine didn't change),
    the model id + revision (same weights), the tokenizer revision +
    chat-template sha (same prompt rendering), the prompt version + sha
    (same protocol), and the machine (same hardware/MLX).
    """
    return {
        "manifest_version": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run_id,
        "config": dict(config),
        "dataset_lock_sha256": dataset_lock_sha256,
        "code_hash": dict(code_hash),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_chat_template_sha256": tokenizer_chat_template_sha256,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "machine": dict(machine),
    }


def write_manifest(out_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write the manifest atomically to ``out_dir/manifest.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST_PATH
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_manifest(out_dir: str | Path) -> dict[str, Any] | None:
    """Load the manifest, or None if it does not exist."""
    path = Path(out_dir) / MANIFEST_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class ManifestMismatchError(RuntimeError):
    """``--resume`` was asked to continue a run whose manifest differs."""

    def __init__(self, differences: dict[str, tuple[Any, Any]]):
        self.differences = differences
        lines = [f"  {k}: {old!r} -> {new!r}" for k, (old, new) in differences.items()]
        super().__init__(
            "manifest mismatch — refusing to resume (the resumed run would "
            "not be comparable to the committed results):\n" + "\n".join(lines)
        )


# The manifest fields that MUST match for a resume. ``created_utc`` and
# ``manifest_version`` are deliberately excluded — they describe the
# manifest itself, not the run's comparability.
_RESUME_MATCH_FIELDS: tuple[str, ...] = (
    "config",
    "dataset_lock_sha256",
    "code_hash",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "tokenizer_chat_template_sha256",
    "prompt_version",
    "prompt_sha256",
    "machine",
)


def verify_manifest_matches(
    existing: dict[str, Any],
    new: dict[str, Any],
    *,
    skip_fields: tuple[str, ...] = (),
) -> None:
    """Raise ``ManifestMismatchError`` if ``new`` differs from ``existing``
    on any resume-critical field.

    The comparison is deep (dicts compared by content, not identity). The
    ``machine`` dict is compared on the resume-critical subset (chip,
    macOS, mlx/mlx-lm versions) — RAM and timestamp are allowed to differ
    (a resume on the same machine after a reboot is fine).
    """
    differences: dict[str, tuple[Any, Any]] = {}
    for field in _RESUME_MATCH_FIELDS:
        if field in skip_fields:
            continue
        old_val = existing.get(field)
        new_val = new.get(field)
        if field == "machine":
            old_val = _machine_resume_subset(old_val)
            new_val = _machine_resume_subset(new_val)
        if old_val != new_val:
            differences[field] = (old_val, new_val)
    if differences:
        raise ManifestMismatchError(differences)


def _machine_resume_subset(machine: dict[str, Any] | None) -> dict[str, Any]:
    """The machine fields that must match across a resume.

    chip + macOS + mlx/mlx-lm versions define the numerical environment.
    RAM and timestamp are allowed to differ.
    """
    if not isinstance(machine, dict):
        return {}
    return {
        "chip": machine.get("chip"),
        "macos_version": machine.get("macos_version"),
        "mlx_version": machine.get("mlx_version"),
        "mlx_lm_version": machine.get("mlx_lm_version"),
    }


# ---------------------------------------------------------------------------
# One-writer lock
# ---------------------------------------------------------------------------


class ResultsLock:
    """A file-based one-writer lock for a results folder.

    Acquired on ``__enter__``; released on ``__exit__``. A stale lock
    (owner PID no longer alive) is breakable so a crashed run can be
    resumed. The lock file holds the owner PID + start time for
    diagnostics.
    """

    def __init__(self, out_dir: str | Path, *, pid: int | None = None):
        self._path = Path(out_dir) / LOCK_PATH
        self._pid = pid or os.getpid()
        self._held = False

    def __enter__(self) -> ResultsLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def acquire(self) -> None:
        """Acquire the lock, breaking a stale lock if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            if self._is_stale():
                self._path.unlink(missing_ok=True)
            else:
                owner = self._read_owner()
                raise RuntimeError(
                    f"results folder {self._path.parent} is locked by another "
                    f"writer (pid {owner}). Remove {self._path} only if that "
                    "process is dead."
                )
        content = json.dumps(
            {"pid": self._pid, "started_utc": datetime.now(UTC).isoformat(timespec="seconds")},
            sort_keys=True,
        )
        # O_EXCL create: atomic acquire on POSIX.
        fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)
        self._held = True

    def release(self) -> None:
        """Release the lock (only if we hold it)."""
        if not self._held:
            return
        self._held = False
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def _is_stale(self) -> bool:
        """True if the lock's owner PID is not alive.

        W5c-7 review: a malformed/unreadable lock is NOT auto-stale —
        that would allow breaking a lock we can't identify. Returns False
        (refuse to break) so the operator investigates. PID reuse is
        mitigated by ``started_utc``: if the PID is alive but the lock is
        older than 24h, it's stale (the process can't still be running
        this run — a resume after a reboot has a different PID space).
        """
        owner, started = self._read_owner_and_started()
        if owner is None:
            # Malformed lock: refuse to break it silently.
            return False
        if not _pid_alive(owner):
            return True
        # PID is alive — check the timestamp to rule out PID reuse.
        if started is not None:
            try:
                started_dt = datetime.fromisoformat(started)
                age = datetime.now(UTC) - started_dt
                if age.total_seconds() > 86400:  # 24h
                    return True
            except (ValueError, TypeError):
                pass
        return False

    def _read_owner_and_started(self) -> tuple[int | None, str | None]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0)) or None
            started = data.get("started_utc")
            return pid, started if isinstance(started, str) else None
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return None, None

    def _read_owner(self) -> int | None:
        pid, _ = self._read_owner_and_started()
        return pid


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` is a live process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but we can't signal it — it's alive.
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Crash-safe per-case commit
# ---------------------------------------------------------------------------


def case_key(case: dict, permutation: str) -> str:
    """The stable key for one (case, permutation) pair.

    Used by the journal to track which cases are durably committed so
    ``--resume`` can skip them. The key is (case_id, permutation) — a
    rotated/fieldperm variant of the same case is a separate key.
    """
    return f"{case.get('id', '')}::{permutation}"


def completed_case_keys(out_dir: str | Path) -> set[str]:
    """Read the journal of completed case keys (for ``--resume``)."""
    path = Path(out_dir) / JOURNAL_PATH
    keys: set[str] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = entry.get("key")
                if isinstance(key, str):
                    keys.add(key)
            except json.JSONDecodeError:
                # A malformed journal line means the journal write was
                # interrupted — stop reading here (later entries are
                # unreliable). The committed offset up to this point is
                # still valid.
                break
    return keys


def last_committed_offset(out_dir: str | Path) -> int:
    """The byte offset of the last durably-committed case in predictions.jsonl.

    The journal records the offset AFTER each case's blob is fsynced. On
    resume, predictions.jsonl is truncated to this offset — any trailing
    half-written data (from a crash mid-blob) is discarded, so the file
    is always a clean sequence of complete JSON lines.
    """
    path = Path(out_dir) / JOURNAL_PATH
    if not path.exists():
        return 0
    last_offset = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                offset = entry.get("offset")
                if isinstance(offset, int) and offset >= 0:
                    last_offset = offset
            except json.JSONDecodeError:
                break  # interrupted journal write — stop here
    return last_offset


def truncate_to_last_commit(out_dir: str | Path) -> int:
    """Truncate predictions.jsonl to the last committed byte offset.

    Returns the offset truncated to. After this call, predictions.jsonl
    is a clean sequence of complete JSON lines — no half-written trailing
    data from a crash mid-blob.
    """
    offset = last_committed_offset(out_dir)
    pred_path = Path(out_dir) / "predictions.jsonl"
    if not pred_path.exists():
        return 0
    current = pred_path.stat().st_size
    if current > offset:
        with open(pred_path, "r+", encoding="utf-8") as f:
            f.truncate(offset)
            f.flush()
            os.fsync(f.fileno())
    return offset


def write_case_blob(
    predictions_path: str | Path,
    journal_path: str | Path,
    case_key_str: str,
    lines: list[dict],
) -> None:
    """Durably commit one case's prediction lines.

    W5c-7 review S1/S2/S3: the blob is appended (write+flush+fsync), then
    the journal entry — carrying the byte offset AFTER the append — lands.
    On resume, ``truncate_to_last_commit`` discards any trailing data past
    the last journal offset, so a crash mid-blob leaves a clean file
    (no truncated half-line). A crash after the blob but before the
    journal means the offset is stale — the resume truncates the orphaned
    blob and re-runs the case.

    The journal key is per (case, permutation) — a case with rotation
    variants commits each variant separately (S3).
    """
    pred_path = Path(predictions_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
    with open(pred_path, "a", encoding="utf-8") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
        offset = f.tell()
    # Journal entry: the commit record. Lands AFTER the blob is durable.
    jpath = Path(journal_path)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        json.dumps(
            {"key": case_key_str, "lines": len(lines), "offset": offset},
            sort_keys=True,
        )
        + "\n"
    )
    with open(jpath, "a", encoding="utf-8") as f:
        f.write(entry)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Machine probe (for the manifest)
# ---------------------------------------------------------------------------


def machine_probe() -> dict[str, Any]:
    """The machine fields captured in the manifest.

    Mirrors ``evalreport.environment()`` but returns only the
    resume-critical subset (chip, macOS, mlx/mlx-lm versions). RAM and
    timestamp are excluded — they don't affect numerical comparability.
    """
    macos = platform.mac_ver()[0] or None

    def _probe(cmd: list[str]) -> str | None:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def _dist_version(name: str) -> str | None:
        try:
            from importlib import metadata

            return metadata.version(name)
        except Exception:  # noqa: BLE001 — provenance is best-effort
            return None

    return {
        "chip": _probe(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "macos_version": macos,
        "mlx_version": _dist_version("mlx"),
        "mlx_lm_version": _dist_version("mlx-lm"),
    }


# ---------------------------------------------------------------------------
# Circuit breaker (B6)
# ---------------------------------------------------------------------------

# The broad failure signatures that count toward the breaker. A malformed
# dataset row or a schema validation failure is NOT an infrastructure
# failure — it trips a different breaker (or none) so one bad row cannot
# stop a whole model run.
_INFRA_SIGNATURES: tuple[str, ...] = (
    "metal_allocation",
    "metal",
    "out_of_memory",
    "oom",
    "cuda",
    "runtime_error",
    "connection",
    "timeout",
    "broken_pipe",
)


def failure_signature(exc: BaseException) -> str | None:
    """Classify an exception into a broad infra signature, or None.

    Returns None for expected/validation failures (malformed rows, schema
    errors, unsupported field types) — those are "failure is a result"
    cases that must NOT trip the infra breaker. Only infrastructure
    failures (Metal OOM, allocation, network, timeout) count.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    # Validation/schema errors are expected — never infra.
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return None
    for sig in _INFRA_SIGNATURES:
        if sig in msg or sig in name.lower():
            return sig
    # A bare RuntimeError with no infra keyword is ambiguous — treat it
    # as infra only if it mentions allocation or memory.
    if isinstance(exc, RuntimeError):
        if "alloc" in msg or "memory" in msg:
            return "metal_allocation"
        return None
    return None


class CircuitBreaker:
    """Trip after N consecutive infrastructure failures with the same
    broad signature (B6).

    W5c-7 review: "consecutive" means consecutive — a success RESETS the
    count. Five infra failures spread over 1000 successful cases do NOT
    trip the breaker. A malformed dataset row or expected validation
    failure does NOT trip it ("failure is a result") and does NOT reset
    the infra count (it is not an infra failure). Five consecutive infra
    failures of the same signature trips the breaker; the caller marks
    remaining work for that model and continues to the next.
    """

    def __init__(self, threshold: int = circuit_breaker_threshold):
        self._threshold = threshold
        self._consecutive: dict[str, int] = {}
        self._tripped: str | None = None

    def record_success(self) -> None:
        """Record a successful case — resets the consecutive infra count."""
        if self._tripped is not None:
            return
        self._consecutive.clear()

    def record(self, exc: BaseException) -> bool:
        """Record a failure. Returns True if the breaker TRIPPED on this
        call (the caller should stop the current model and move on)."""
        if self._tripped is not None:
            return True
        sig = failure_signature(exc)
        if sig is None:
            # Expected/validation failure: does not count toward infra,
            # and does NOT reset the infra count (it is not a success).
            return False
        # Reset OTHER signatures' counts — only the SAME signature
        # consecutively accumulates.
        self._consecutive = {sig: self._consecutive.get(sig, 0) + 1}
        if self._consecutive[sig] >= self._threshold:
            self._tripped = sig
            return True
        return False

    @property
    def tripped(self) -> str | None:
        """The signature that tripped the breaker, or None."""
        return self._tripped

    @property
    def threshold(self) -> int:
        """The configured trip threshold."""
        return self._threshold
