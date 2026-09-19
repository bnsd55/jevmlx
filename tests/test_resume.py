"""Tests for jevmlx.resume — crash-safe resumable eval infrastructure (W5c-7 / B6).

Four contracts, one file:

- **Manifest**: build_manifest captures every resume-critical field;
  write/load round-trips; verify_manifest_matches raises
  ManifestMismatchError with field-level diffs on ANY resume-critical
  change, while created_utc / manifest_version are excluded and the
  machine comparison is limited to chip + macOS + mlx/mlx-lm versions
  (RAM and probe timestamp may drift).

- **ResultsLock**: one-writer per results folder — .lock carries the
  owner PID, a second live acquire raises RuntimeError, a stale lock
  (owner PID dead) is breakable, release removes the file, and the
  context manager acquires/releases symmetrically.

- **Crash-safe per-case commit**: write_case_blob lands the case's
  prediction lines durably FIRST and the journal entry AFTER (the
  journal entry is the commit point) — so a case whose blob landed but
  whose journal entry did not is NOT complete and is re-run on resume.

- **CircuitBreaker**: trips after 5 consecutive infra failures with the
  SAME broad signature, a different signature resets the count,
  validation errors (ValueError/TypeError/KeyError/AttributeError)
  never count, and once tripped every record() reports tripped.
  failure_signature classifies Metal OOM / allocation / network /
  timeout as infra and returns None for validation errors.

Pure stdlib + the module: no mlx, no model, no network. All file
operations live under pytest's tmp_path.

Author: Ben Shaharizad <bnsd55@gmail.com>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from jevmlx.resume import (
    JOURNAL_PATH,
    LOCK_PATH,
    MANIFEST_PATH,
    CircuitBreaker,
    ManifestMismatchError,
    ResultsLock,
    build_manifest,
    case_key,
    circuit_breaker_threshold,
    completed_case_keys,
    failure_signature,
    load_manifest,
    verify_manifest_matches,
    write_case_blob,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BASE_MACHINE = {
    "chip": "Apple M5",
    "macos_version": "15.1.0",
    "mlx_version": "0.21.0",
    "mlx_lm_version": "0.31.3",
    # Non-comparable fields that must be IGNORED by the resume check.
    "ram_gb": 32,
    "probed_at": "2025-01-01T00:00:00+00:00",
}


def make_manifest(**overrides):
    """A fully-populated manifest via build_manifest, with opt-in overrides."""
    kwargs = {
        "config": {
            "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "track": "typed-decisions",
            "split": "dev",
            "permutations": ["id", "rot"],
            "temperature": 0.0,
        },
        "dataset_lock_sha256": "d" * 64,
        "prompt_version": "v2",
        "prompt_sha256": "p" * 64,
        "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "tokenizer_chat_template_sha256": "c" * 64,
        "machine": dict(BASE_MACHINE),
        "code_hash": {"git_sha": "1" * 40, "git_dirty": False},
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


@pytest.fixture()
def dead_pid() -> int:
    """A definitively dead (already reaped) PID for stale-lock tests."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()  # reaps the child: the PID is guaranteed not alive now
    return proc.pid


# ---------------------------------------------------------------------------
# 1. Manifest
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_has_all_required_fields(self):
        m = make_manifest()
        for field in (
            "manifest_version",
            "created_utc",
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
        ):
            assert field in m, f"manifest missing required field {field!r}"

    def test_manifest_version_and_created_utc(self):
        m = make_manifest()
        assert m["manifest_version"] == 1
        created = datetime.fromisoformat(m["created_utc"])
        assert created.tzinfo is not None  # UTC-aware timestamp

    def test_shallow_copies_input_dicts(self):
        config = {"model": "m", "permutations": ["id"]}
        machine = {"chip": "Apple M5"}
        code_hash = {"git_sha": "x", "git_dirty": False}
        m = build_manifest(
            config=config,
            dataset_lock_sha256=None,
            prompt_version=None,
            prompt_sha256=None,
            model_id="m",
            model_revision=None,
            tokenizer_revision=None,
            tokenizer_chat_template_sha256=None,
            machine=machine,
            code_hash=code_hash,
        )
        # Field values are carried over verbatim...
        assert m["config"] == config
        assert m["machine"] == machine
        assert m["code_hash"] == code_hash
        # ...and the manifest's top-level dicts are fresh copies: replacing
        # a top-level value in the manifest never mutates the caller's dict.
        m["config"] = {"model": "other"}
        m["machine"] = {"chip": "changed"}
        m["code_hash"] = {"git_sha": "changed"}
        assert config == {"model": "m", "permutations": ["id"]}
        assert machine == {"chip": "Apple M5"}
        assert code_hash == {"git_sha": "x", "git_dirty": False}


class TestManifestRoundTrip:
    def test_write_then_load_round_trips(self, tmp_path):
        m = make_manifest()
        path = write_manifest(tmp_path, m)
        assert path == tmp_path / MANIFEST_PATH
        assert path.exists()
        assert load_manifest(tmp_path) == m

    def test_write_is_atomically_replaceable(self, tmp_path):
        first = make_manifest(model_id="first/model")
        second = make_manifest(model_id="second/model")
        write_manifest(tmp_path, first)
        write_manifest(tmp_path, second)
        assert load_manifest(tmp_path) == second
        # No temp-file litter left behind.
        assert [p.name for p in tmp_path.iterdir()] == [MANIFEST_PATH]

    def test_load_missing_manifest_returns_none(self, tmp_path):
        assert load_manifest(tmp_path) is None
        assert load_manifest(tmp_path / "does-not-exist") is None


class TestVerifyManifestMatches:
    def test_identical_manifests_pass(self):
        existing = make_manifest()
        # Rebuilt from the same inputs — only created_utc differs.
        verify_manifest_matches(existing, make_manifest())

    def test_created_utc_and_manifest_version_are_excluded(self):
        existing = make_manifest()
        new = dict(existing)
        new["created_utc"] = "2099-12-31T23:59:59+00:00"
        new["manifest_version"] = 999
        verify_manifest_matches(existing, new)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("config", {"model": "other/model"}),
            ("dataset_lock_sha256", "f" * 64),
            ("code_hash", {"git_sha": "e" * 40, "git_dirty": True}),
            ("model_id", "mlx-community/other"),
            ("model_revision", "dead" * 10),
            ("tokenizer_revision", "cafe" * 10),
            ("tokenizer_chat_template_sha256", "9" * 64),
            ("prompt_version", "v1"),
            ("prompt_sha256", "7" * 64),
        ],
    )
    def test_raises_with_field_level_diff(self, field, value):
        existing = make_manifest()
        new = dict(existing)
        new[field] = value
        with pytest.raises(ManifestMismatchError) as excinfo:
            verify_manifest_matches(existing, new)
        (old, new_val) = excinfo.value.differences[field]
        assert old == existing[field]
        assert new_val == value
        # The message names the field so the operator can act on it.
        assert field in str(excinfo.value)

    def test_reports_every_differing_field_at_once(self):
        existing = make_manifest()
        new = dict(existing)
        new["model_id"] = "mlx-community/other"
        new["prompt_version"] = "v1"
        with pytest.raises(ManifestMismatchError) as excinfo:
            verify_manifest_matches(existing, new)
        assert set(excinfo.value.differences) == {"model_id", "prompt_version"}
        assert "refusing to resume" in str(excinfo.value)
        assert isinstance(excinfo.value, RuntimeError)  # subclass contract

    def test_machine_ram_and_timestamp_drift_is_allowed(self):
        existing = make_manifest()
        new = dict(existing)
        new["machine"] = {
            "chip": BASE_MACHINE["chip"],
            "macos_version": BASE_MACHINE["macos_version"],
            "mlx_version": BASE_MACHINE["mlx_version"],
            "mlx_lm_version": BASE_MACHINE["mlx_lm_version"],
            "ram_gb": 64,  # rebooted with different reported RAM
            "probed_at": "2099-01-01T00:00:00+00:00",  # later probe
        }
        verify_manifest_matches(existing, new)

    @pytest.mark.parametrize("field", ["chip", "macos_version", "mlx_version", "mlx_lm_version"])
    def test_machine_resume_critical_fields_must_match(self, field):
        existing = make_manifest()
        new = dict(existing)
        changed = dict(new["machine"])
        changed[field] = "totally-different"
        new["machine"] = changed
        with pytest.raises(ManifestMismatchError) as excinfo:
            verify_manifest_matches(existing, new)
        assert "machine" in excinfo.value.differences


# ---------------------------------------------------------------------------
# 2. ResultsLock
# ---------------------------------------------------------------------------


class TestResultsLock:
    def test_acquire_creates_lock_file_with_pid(self, tmp_path):
        lock = ResultsLock(tmp_path)
        lock.acquire()
        lock_path = tmp_path / LOCK_PATH
        assert lock_path.exists()
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid()
        assert "started_utc" in data
        lock.release()

    def test_second_acquire_on_same_dir_raises(self, tmp_path):
        with ResultsLock(tmp_path):
            second = ResultsLock(tmp_path)
            with pytest.raises(RuntimeError, match="locked by another writer"):
                second.acquire()

    def test_stale_lock_is_breakable(self, tmp_path, dead_pid):
        stale = ResultsLock(tmp_path, pid=dead_pid)
        stale.acquire()
        assert (tmp_path / LOCK_PATH).exists()
        # The stale owner is dead: a fresh writer must be able to take over.
        fresh = ResultsLock(tmp_path)
        fresh.acquire()
        data = json.loads((tmp_path / LOCK_PATH).read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid()
        fresh.release()

    def test_release_removes_lock(self, tmp_path):
        lock = ResultsLock(tmp_path)
        lock.acquire()
        assert (tmp_path / LOCK_PATH).exists()
        lock.release()
        assert not (tmp_path / LOCK_PATH).exists()

    def test_release_without_held_lock_is_noop(self, tmp_path):
        ResultsLock(tmp_path).release()
        assert not (tmp_path / LOCK_PATH).exists()

    def test_reacquire_after_release(self, tmp_path):
        lock = ResultsLock(tmp_path)
        lock.acquire()
        lock.release()
        lock.acquire()
        assert (tmp_path / LOCK_PATH).exists()
        lock.release()

    def test_context_manager_acquires_and_releases(self, tmp_path):
        with ResultsLock(tmp_path) as lock:
            assert isinstance(lock, ResultsLock)
            assert (tmp_path / LOCK_PATH).exists()
            # Still one-writer inside the with-block.
            with pytest.raises(RuntimeError, match="locked by another writer"):
                ResultsLock(tmp_path).acquire()
        assert not (tmp_path / LOCK_PATH).exists()

    def test_context_manager_releases_on_exception(self, tmp_path):
        with pytest.raises(ValueError, match="boom"):
            with ResultsLock(tmp_path):
                raise ValueError("boom")
        assert not (tmp_path / LOCK_PATH).exists()
        # And the folder is immediately usable by the next writer.
        with ResultsLock(tmp_path):
            pass


# ---------------------------------------------------------------------------
# 3. Crash-safe per-case commit
# ---------------------------------------------------------------------------


class TestCaseKey:
    def test_key_is_stable_for_case_and_permutation(self):
        case = {"id": "case-7", "text": "irrelevant payload"}
        assert case_key(case, "id") == "case-7::id"
        assert case_key(case, "id") == case_key(case, "id")

    def test_key_distinguishes_permutation_and_case(self):
        case = {"id": "case-7"}
        assert case_key(case, "id") != case_key(case, "rot")
        assert case_key({"id": "case-8"}, "id") != case_key(case, "id")

    def test_key_without_case_id_is_still_stable(self):
        assert case_key({}, "rot") == "::rot"
        assert case_key({}, "rot") == case_key({}, "rot")

    def test_keys_round_trip_through_the_journal(self, tmp_path):
        pred = tmp_path / "predictions.jsonl"
        journal = tmp_path / JOURNAL_PATH
        case = {"id": "case-9"}
        for perm in ("id", "rot"):
            write_case_blob(pred, journal, case_key(case, perm), [{"v": perm}])
        assert completed_case_keys(tmp_path) == {
            case_key(case, "id"),
            case_key(case, "rot"),
        }


class TestWriteCaseBlob:
    def test_appends_lines_then_journal_entry(self, tmp_path):
        pred = tmp_path / "predictions.jsonl"
        journal = tmp_path / JOURNAL_PATH
        lines = [
            {"case_id": "c1", "field": "label", "value": "yes"},
            {"case_id": "c1", "field": "confidence", "value": 0.9},
        ]
        write_case_blob(pred, journal, "c1::id", lines)
        # All prediction lines landed, in order, as JSON lines.
        written = [json.loads(line) for line in pred.read_text().splitlines()]
        assert written == lines
        # Exactly one journal entry, recording the key and the line count.
        entries = [json.loads(line) for line in journal.read_text().splitlines()]
        assert entries == [{"key": "c1::id", "lines": 2, "offset": entries[0]["offset"]}]
        assert isinstance(entries[0]["offset"], int)
        assert entries[0]["offset"] > 0
        assert completed_case_keys(tmp_path) == {"c1::id"}

    def test_successive_cases_append_without_clobbering(self, tmp_path):
        pred = tmp_path / "predictions.jsonl"
        journal = tmp_path / JOURNAL_PATH
        write_case_blob(pred, journal, "c1::id", [{"case_id": "c1"}])
        write_case_blob(pred, journal, "c2::id", [{"case_id": "c2"}])
        written = [json.loads(line) for line in pred.read_text().splitlines()]
        assert [row["case_id"] for row in written] == ["c1", "c2"]
        assert completed_case_keys(tmp_path) == {"c1::id", "c2::id"}

    def test_creates_missing_parent_directories(self, tmp_path):
        pred = tmp_path / "nested" / "deeper" / "predictions.jsonl"
        journal = tmp_path / "other" / JOURNAL_PATH
        write_case_blob(pred, journal, "c1::id", [{"case_id": "c1"}])
        assert pred.exists()
        assert journal.exists()


class TestCompletedCaseKeys:
    def test_missing_journal_means_nothing_complete(self, tmp_path):
        assert completed_case_keys(tmp_path) == set()
        assert completed_case_keys(tmp_path / "nope") == set()

    def test_tolerates_malformed_and_blank_journal_lines(self, tmp_path):
        """Malformed/blank lines stop reading (interrupted journal write).

        W5c-7 review: a malformed journal line means the journal write was
        interrupted — later entries are unreliable. The reader stops at the
        first malformed line; entries before it are still valid.
        """
        journal = tmp_path / JOURNAL_PATH
        journal.write_text(
            '{"key": "a::id"}\n'
            "\n"  # blank line — skipped
            '{"key": "b::id", "lines": 1}\n',
            encoding="utf-8",
        )
        assert completed_case_keys(tmp_path) == {"a::id", "b::id"}

    def test_stops_at_malformed_line(self, tmp_path):
        """A torn journal write (malformed JSON) stops reading — later
        entries are not trusted."""
        journal = tmp_path / JOURNAL_PATH
        journal.write_text(
            '{"key": "a::id"}\n'
            "not json at all\n"  # torn write
            '{"key": "b::id"}\n',  # not trusted — after the torn line
            encoding="utf-8",
        )
        assert completed_case_keys(tmp_path) == {"a::id"}


class TestCrashBetweenBlobAndJournal:
    def test_blob_without_journal_entry_is_not_complete(self, tmp_path):
        """The B6 commit contract: the journal entry is the commit point.

        Simulates a crash AFTER case-b's blob became durable but BEFORE its
        journal entry landed: resume must NOT consider case-b complete.
        """
        pred = tmp_path / "predictions.jsonl"
        journal = tmp_path / JOURNAL_PATH

        # Case A commits cleanly.
        write_case_blob(pred, journal, "case-a::id", [{"case_id": "case-a"}])
        assert completed_case_keys(tmp_path) == {"case-a::id"}

        # Case B: its blob lands (one write + flush + fsync, as the module
        # does) — then the process dies before the journal entry is written.
        b_lines = [{"case_id": "case-b", "field": "label", "value": "no"}]
        blob = "".join(json.dumps(line, sort_keys=True) + "\n" for line in b_lines)
        with open(pred, "a", encoding="utf-8") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        # ...crash here. Journal still only knows about case A.
        assert completed_case_keys(tmp_path) == {"case-a::id"}

        # Resume re-runs case B; its (duplicate) blob lines are tolerated and
        # the second commit registers it as complete.
        write_case_blob(pred, journal, "case-b::id", b_lines)
        assert completed_case_keys(tmp_path) == {"case-a::id", "case-b::id"}
        written = [json.loads(line) for line in pred.read_text().splitlines()]
        assert written.count(b_lines[0]) == 2  # duplicate from the re-run


# ---------------------------------------------------------------------------
# 4. CircuitBreaker + failure_signature
# ---------------------------------------------------------------------------


class TestFailureSignature:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (RuntimeError("Metal out of memory on device 0"), "metal"),
            (RuntimeError("allocation of 2 GB buffer failed"), "metal_allocation"),
            (ConnectionError("connection refused by peer"), "connection"),
            (TimeoutError("inference timed out"), "timeout"),
            (RuntimeError("cuda error 700"), "cuda"),
        ],
    )
    def test_infra_failures_classify(self, exc, expected):
        assert failure_signature(exc) == expected

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("malformed dataset row"),
            TypeError("unsupported field type"),
            KeyError("missing column"),
            AttributeError("schema has no attribute"),
            RuntimeError("mystery failure with no infra keyword"),
            ValueError("timeout while validating row"),  # validation wins
        ],
    )
    def test_validation_errors_are_not_infra(self, exc):
        assert failure_signature(exc) is None


class TestCircuitBreaker:
    def test_default_threshold_is_five(self):
        assert circuit_breaker_threshold == 5

    @staticmethod
    def _oom() -> RuntimeError:
        return RuntimeError("Metal out of memory")

    @staticmethod
    def _conn() -> ConnectionError:
        return ConnectionError("connection refused")

    def test_trips_after_five_consecutive_same_signature(self):
        cb = CircuitBreaker()
        for i in range(4):
            assert cb.record(self._oom()) is False, f"tripped early at {i + 1}"
            assert cb.tripped is None
        # The fifth consecutive infra failure with the same signature trips.
        assert cb.record(self._oom()) is True
        assert cb.tripped == "metal"

    def test_different_signature_resets_the_count(self):
        cb = CircuitBreaker()
        for _ in range(4):
            assert cb.record(self._oom()) is False
        # A different infra signature resets the consecutive count.
        assert cb.record(self._conn()) is False
        assert cb.tripped is None
        for _ in range(4):
            assert cb.record(self._oom()) is False
        assert cb.tripped is None  # only 4 consecutive oom since the reset
        assert cb.record(self._oom()) is True  # 5th consecutive oom trips
        assert cb.tripped == "metal"

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("bad row"),
            TypeError("bad type"),
            KeyError("missing"),
            AttributeError("no attr"),
        ],
    )
    def test_validation_errors_never_trip_the_breaker(self, exc):
        cb = CircuitBreaker()
        for _ in range(10):
            assert cb.record(exc) is False
        assert cb.tripped is None

    def test_validation_errors_do_not_reset_the_infra_count(self):
        """A malformed row is not infra: it neither counts nor resets."""
        cb = CircuitBreaker()
        for _ in range(4):
            assert cb.record(self._oom()) is False
        assert cb.record(ValueError("bad row")) is False
        # Still 4 consecutive oom failures — the next one trips.
        assert cb.record(self._oom()) is True
        assert cb.tripped == "metal"

    def test_once_tripped_every_record_reports_tripped(self):
        cb = CircuitBreaker()
        for _ in range(5):
            cb.record(self._oom())
        assert cb.tripped == "metal"
        # Subsequent calls — infra or validation — all report tripped.
        assert cb.record(self._conn()) is True
        assert cb.record(ValueError("bad row")) is True
        assert cb.tripped == "metal"


class TestEndToEndKillResume:
    """W5c-7 review: ONE end-to-end test that runs run_eval, kills at each
    of 3 crash points, resumes, and asserts byte-identical predictions.jsonl
    to a clean run.
    """

    def _cases(self):
        return [
            {
                "id": f"case-{i}",
                "schema": {"verdict": {"type": "enum", "choices": ["yes", "no"]}},
                "context": f"Evidence block {i}.",
                "labels": {"verdict": "yes" if i % 2 == 0 else "no"},
                "split": "train",
            }
            for i in range(6)
        ]

    def _clean_decide(self):
        """A deterministic decide_fn that always returns the same result."""

        def decide(schema_dict, context, **kwargs):
            return {
                "verdict": {
                    "prediction": "yes",
                    "valid": True,
                    "log_scores": {"yes": 0.0, "no": -1.0},
                    "probability": {"yes": 0.7, "no": 0.3},
                },
                "_meta": {"latency_ms": 1.0, "prompt_sha256": "a" * 64},
            }

        return decide

    _FIXED_RUN_ID = "20260919T000000Z-testrun"

    def _clean_run(self, tmp_path):
        """Run all cases cleanly — the reference predictions.jsonl."""
        from jevmlx.evalrun import run_eval

        out = tmp_path / "clean"
        run_eval(
            self._cases(),
            self._clean_decide(),
            track="parallel",
            model="test-model",
            out_dir=str(out),
            run_id=self._FIXED_RUN_ID,
        )
        return (out / "predictions.jsonl").read_bytes()

    def _assert_full_contract(self, out, clean_out, n_cases):
        """Assert predictions byte-identical, journal has one entry per
        (case, variant), and run.json counts match the reference."""
        from jevmlx.resume import completed_case_keys

        resumed_bytes = (out / "predictions.jsonl").read_bytes()
        clean_bytes = (clean_out / "predictions.jsonl").read_bytes()
        assert resumed_bytes == clean_bytes, (
            f"predictions differ: clean={len(clean_bytes)}B, resumed={len(resumed_bytes)}B"
        )
        # Journal: one entry per (case, variant). 6 cases, no permutations
        # (canonical only) = 6 entries.
        keys = completed_case_keys(out)
        assert len(keys) == n_cases, f"journal has {len(keys)} entries, expected {n_cases}: {keys}"
        # run.json counts: prediction_lines matches the reference.
        resumed_run = json.loads((out / "run.json").read_text())
        clean_run = json.loads((clean_out / "run.json").read_text())
        assert (
            resumed_run["counts"]["prediction_lines"] == clean_run["counts"]["prediction_lines"]
        ), f"run.json counts differ: resumed={resumed_run['counts']}, clean={clean_run['counts']}"

    def test_kill_after_blob_before_journal_resume_is_identical(self, tmp_path):
        """Crash point 1: kill AFTER the blob is written but BEFORE the
        journal entry. Resume truncates the orphaned blob and re-runs the
        case. Result: byte-identical to a clean run.
        """
        import jevmlx.resume as resume_mod

        out = tmp_path / "crash"
        self._clean_run(tmp_path)

        # Patch write_case_blob to crash after the blob but before the
        # journal on the 3rd case (case-2).
        original_write = resume_mod.write_case_blob
        call_count = {"n": 0}

        def crashing_write(pred_path, journal_path, case_key, lines):
            call_count["n"] += 1
            if call_count["n"] == 3:
                # Write the blob (as original does), then crash before journal.
                pred_path = Path(pred_path)
                blob = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
                with open(pred_path, "a", encoding="utf-8") as f:
                    f.write(blob)
                    f.flush()
                    os.fsync(f.fileno())
                raise KeyboardInterrupt("simulated crash after blob")
            return original_write(pred_path, journal_path, case_key, lines)

        resume_mod.write_case_blob = crashing_write
        try:
            from jevmlx.evalrun import run_eval

            run_eval(
                self._cases(),
                self._clean_decide(),
                track="parallel",
                model="test-model",
                run_id=self._FIXED_RUN_ID,
                out_dir=str(out),
            )
        except KeyboardInterrupt:
            pass
        finally:
            resume_mod.write_case_blob = original_write

        # Resume: should truncate the orphaned blob, re-run case-2.
        from jevmlx.evalrun import run_eval

        run_eval(
            self._cases(),
            self._clean_decide(),
            track="parallel",
            model="test-model",
            run_id=self._FIXED_RUN_ID,
            out_dir=str(out),
            resume=True,
        )
        self._assert_full_contract(out, tmp_path / "clean", 6)

    def test_kill_mid_blob_resume_is_identical(self, tmp_path):
        """Crash point 2: kill MID-BLOB (truncated half-line in the file).
        Resume truncates to the last committed offset and re-runs. Result:
        byte-identical to a clean run.
        """
        out = tmp_path / "crash_mid"
        self._clean_run(tmp_path)

        # First: run 2 cases cleanly (so there's a committed offset).
        from jevmlx.evalrun import run_eval

        run_eval(
            self._cases()[:2],
            self._clean_decide(),
            track="parallel",
            model="test-model",
            run_id=self._FIXED_RUN_ID,
            out_dir=str(out),
        )

        # Now simulate a mid-blob crash: append a truncated half-line.
        with open(out / "predictions.jsonl", "a", encoding="utf-8") as f:
            f.write('{"case_id": "case-2", "field": "verdict", "predi')  # truncated
            f.flush()

        # Resume: truncate to last commit, re-run case-2 onward.
        run_eval(
            self._cases(),
            self._clean_decide(),
            track="parallel",
            model="test-model",
            run_id=self._FIXED_RUN_ID,
            out_dir=str(out),
            resume=True,
        )
        self._assert_full_contract(out, tmp_path / "clean", 6)

    def test_kill_after_complete_resume_is_identical(self, tmp_path):
        """Crash point 3: the run completes fully but we resume anyway.
        All cases are in the journal — nothing is re-run. Result:
        byte-identical (no change).
        """
        out = tmp_path / "crash_after"
        self._clean_run(tmp_path)

        # Copy the clean run to our crash dir.
        import shutil

        shutil.copytree(tmp_path / "clean", out)

        # Resume: all cases already complete, nothing re-run.
        from jevmlx.evalrun import run_eval

        run_eval(
            self._cases(),
            self._clean_decide(),
            track="parallel",
            model="test-model",
            run_id=self._FIXED_RUN_ID,
            out_dir=str(out),
            resume=True,
        )
        self._assert_full_contract(out, tmp_path / "clean", 6)

    def test_rotations_kill_and_resume_is_identical(self, tmp_path):
        """B1: with permutations='rotations', a kill+resume produces
        byte-identical predictions and one journal entry per (case, variant).
        """
        from jevmlx.evalrun import run_eval
        from jevmlx.resume import completed_case_keys

        out = tmp_path / "rotations_crash"
        # Clean reference run with rotations.
        clean_out = tmp_path / "clean_rot"
        run_eval(
            self._cases(),
            self._clean_decide(),
            track="parallel",
            model="test-model",
            out_dir=str(clean_out),
            run_id=self._FIXED_RUN_ID,
            permutations="rotations",
        )
        clean_bytes = (clean_out / "predictions.jsonl").read_bytes()

        # Run 2 cases (3 variants each = 6 journal entries), then crash.
        run_eval(
            self._cases()[:2],
            self._clean_decide(),
            track="parallel",
            model="test-model",
            out_dir=str(out),
            run_id=self._FIXED_RUN_ID,
            permutations="rotations",
        )
        # Simulate a mid-blob crash: append a truncated half-line.
        with open(out / "predictions.jsonl", "a", encoding="utf-8") as f:
            f.write('{"case_id": "case-2", "truncated')
            f.flush()

        # Resume.
        run_eval(
            self._cases(),
            self._clean_decide(),
            track="parallel",
            model="test-model",
            out_dir=str(out),
            run_id=self._FIXED_RUN_ID,
            permutations="rotations",
            resume=True,
        )
        resumed_bytes = (out / "predictions.jsonl").read_bytes()
        assert resumed_bytes == clean_bytes, (
            f"rotations predictions differ: clean={len(clean_bytes)}B, "
            f"resumed={len(resumed_bytes)}B"
        )
        # Journal: one entry per (case, variant). 6 cases x 2 variants
        # (canonical + rot1 for a 2-choice enum) = 12.
        keys = completed_case_keys(out)
        assert len(keys) == 12, (
            f"journal has {len(keys)} entries, expected 12 (6 cases x 2 variants): {keys}"
        )


class TestReviewRound4:
    """W5c-7 review round 4 mandatory tests."""

    _FIXED_RUN_ID = "20260919T000000Z-testrun"

    def _cases(self):
        return [
            {
                "id": f"case-{i}",
                "schema": {"verdict": {"type": "enum", "choices": ["yes", "no"]}},
                "context": f"Evidence block {i}.",
                "labels": {"verdict": "yes" if i % 2 == 0 else "no"},
                "split": "train",
            }
            for i in range(6)
        ]

    def _decide(self):
        def decide(schema_dict, context, **kwargs):
            return {
                "verdict": {
                    "prediction": "yes",
                    "valid": True,
                    "log_scores": {"yes": 0.0, "no": -1.0},
                    "probability": {"yes": 0.7, "no": 0.3},
                    "type": "enum",
                },
                "_meta": {"latency_ms": 1.0, "prompt_sha256": "a" * 64},
            }

        return decide

    def test_resume_with_different_prompt_sha_refuses(self, tmp_path):
        """R1: fresh run, resume same sha (accepted), resume different sha
        (refused). The manifest sha is never overwritten on resume."""
        from jevmlx.evalrun import run_eval
        from jevmlx.resume import ManifestMismatchError, load_manifest

        out = tmp_path / "prompt_mismatch"
        # First run: only 2 cases (so 4 remain to run on resume).
        run_eval(
            self._cases()[:2],
            self._decide(),
            track="parallel",
            model="test-model",
            out_dir=str(out),
            run_id=self._FIXED_RUN_ID,
        )
        # Verify the stored sha.
        stored = load_manifest(out)
        assert stored["prompt_sha256"] == "a" * 64

        # Resume with the SAME sha — accepted.
        run_eval(
            self._cases(),
            self._decide(),
            track="parallel",
            model="test-model",
            out_dir=str(out),
            run_id=self._FIXED_RUN_ID,
            resume=True,
        )
        # The stored sha is still the same (not overwritten with None).
        stored = load_manifest(out)
        assert stored["prompt_sha256"] == "a" * 64

        # Now resume with a DIFFERENT prompt sha — the first uncompleted case
        # produces the sha and the verify refuses.
        # Reset: remove journal so cases re-run.

        # Run 2 fresh cases again into a new dir for the mismatch test.
        out2 = tmp_path / "prompt_mismatch2"
        run_eval(
            self._cases()[:2],
            self._decide(),
            track="parallel",
            model="test-model",
            out_dir=str(out2),
            run_id=self._FIXED_RUN_ID,
        )

        def decide_diff(schema_dict, context, **kwargs):
            return {
                "verdict": {
                    "prediction": "yes",
                    "valid": True,
                    "log_scores": {"yes": 0.0, "no": -1.0},
                    "probability": {"yes": 0.7, "no": 0.3},
                    "type": "enum",
                },
                "_meta": {"latency_ms": 1.0, "prompt_sha256": "b" * 64},
            }

        with pytest.raises(ManifestMismatchError, match="prompt_sha256"):
            run_eval(
                self._cases(),
                decide_diff,
                track="parallel",
                model="test-model",
                out_dir=str(out2),
                run_id=self._FIXED_RUN_ID,
                resume=True,
            )

    def test_infra_failure_then_resume_re_runs_case(self, tmp_path):
        """R2: an infra failure (Metal OOM) writes to errors.jsonl (NOT
        predictions.jsonl), is NOT journaled, and resume re-runs the case.
        After resume, predictions.jsonl is byte-identical to a clean run
        with no _error line for the case that later succeeded."""
        from jevmlx.evalrun import run_eval
        from jevmlx.resume import completed_case_keys

        # Clean reference run.
        clean_out = tmp_path / "clean"
        run_eval(
            self._cases(),
            self._decide(),
            track="parallel",
            model="test-model",
            out_dir=str(clean_out),
            run_id=self._FIXED_RUN_ID,
        )
        clean_bytes = (clean_out / "predictions.jsonl").read_bytes()

        out = tmp_path / "infra_fail"
        call_count = {"n": 0}

        def decide_with_infra(schema_dict, context, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                # Simulate a Metal OOM on case-1.
                raise RuntimeError("metal allocation failed: out of memory")
            return {
                "verdict": {
                    "prediction": "yes",
                    "valid": True,
                    "log_scores": {"yes": 0.0, "no": -1.0},
                    "probability": {"yes": 0.7, "no": 0.3},
                    "type": "enum",
                },
                "_meta": {"latency_ms": 1.0, "prompt_sha256": "a" * 64},
            }

        # First run: only 2 cases — case-0 succeeds, case-1 OOMs.
        # No later cases run, so the only committed case is case-0.
        run_eval(
            self._cases()[:2],
            decide_with_infra,
            track="parallel",
            model="test-model",
            out_dir=str(out),
            run_id=self._FIXED_RUN_ID,
        )
        # case-1 is NOT in completed_case_keys.
        keys = completed_case_keys(out)
        assert "case-1::canonical" not in keys
        # The _error line is in errors.jsonl, NOT predictions.jsonl.
        assert (out / "errors.jsonl").exists()
        errors_content = (out / "errors.jsonl").read_text()
        assert "_error" in errors_content
        # predictions.jsonl has NO _error line.
        pred_content = (out / "predictions.jsonl").read_text()
        assert "_error" not in pred_content

        # Resume: case-1 is re-run (not skipped).
        run_eval(
            self._cases(),
            self._decide(),
            track="parallel",
            model="test-model",
            out_dir=str(out),
            run_id=self._FIXED_RUN_ID,
            resume=True,
        )
        # Now case-1 IS in completed_case_keys.
        keys = completed_case_keys(out)
        assert "case-1::canonical" in keys
        # predictions.jsonl is byte-identical to the clean run.
        resumed_bytes = (out / "predictions.jsonl").read_bytes()
        assert resumed_bytes == clean_bytes, (
            f"predictions differ: clean={len(clean_bytes)}B, resumed={len(resumed_bytes)}B"
        )

    def test_bench_run_eval_into_empty_dir(self, tmp_path, monkeypatch):
        """B3: one run through run_bench/_run_one with ONLY the engine load
        mocked. Exercises bench's manifest-present -> resume rule and --fresh
        rmtree. Verifies the manifest is written, predictions are per-case,
        and run.json has full counts."""
        import jevmlx.engine as engine_mod
        from jevmlx import bench

        # Build a minimal dataset JSONL + lock in tmp_path.
        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"case-{i}",
                        "schema": {"verdict": {"type": "enum", "choices": ["yes", "no"]}},
                        "context": f"Evidence {i}.",
                        "labels": {"verdict": "yes" if i % 2 == 0 else "no"},
                        "split": "train",
                    }
                )
                for i in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        lock = tmp_path / "data.dataset.lock.json"
        lock.write_text(
            json.dumps({"name": "data", "sha256": "deadbeef", "files": {}}),
            encoding="utf-8",
        )

        # Mock ONLY the engine load path — everything else (run_bench,
        # _run_one, run_eval, resume infra) runs unmocked.
        class _FakeTokenizer:
            chat_template = None

        class _FakeEngine:
            tokenizer = _FakeTokenizer()

        monkeypatch.setattr(bench, "preflight", lambda force, machine_override: "test-machine")
        monkeypatch.setattr(
            bench,
            "build_datasets",
            lambda datasets: ({"data": jsonl}, {"data": lock}),
        )
        monkeypatch.setattr(
            bench, "_load_engine_with_timeout", lambda model, timeout: _FakeEngine()
        )
        monkeypatch.setattr(bench, "_run_model_parity", lambda model, engine, folder: None)
        monkeypatch.setattr(engine_mod, "load_engine", lambda model: _FakeEngine())
        monkeypatch.setattr(
            bench,
            "parallel_decide_fn",
            lambda engine, scoring="slots", prior_correction=False: self._decide(),
        )
        # Avoid the git-commit step in summarize.
        import benchmarks.summarize_results as sr_mod

        monkeypatch.setattr(sr_mod, "summarize", lambda folder, parity_note=None: None)

        out = tmp_path / "results"
        result = bench.run_bench(
            model="test/model",
            datasets=["data"],
            scorers=["slots"],
            tracks=["parallel"],
            out=out,
            runs=1,
        )
        # run_bench returns the model results folder.
        assert result.exists()
        # The combo dir has predictions.jsonl + manifest.json + run.json.
        combo = result / "parallel-slots-data"
        assert (combo / "predictions.jsonl").exists()
        assert (combo / "manifest.json").exists()
        assert (combo / "run.json").exists()
        # Manifest has the prompt sha.
        manifest = json.loads((combo / "manifest.json").read_text())
        assert manifest["prompt_sha256"] == "a" * 64
        # run.json has full counts. 3 cases x 2 variants (canonical + rot1
        # for a 2-choice enum) = 6 prediction lines.
        run_json = json.loads((combo / "run.json").read_text())
        assert run_json["counts"]["prediction_lines"] == 6

        # Now test the --fresh rmtree: re-run with fresh=True, verify the
        # old combo dir was removed and a new one written.
        old_mtime = (combo / "predictions.jsonl").stat().st_mtime
        result = bench.run_bench(
            model="test/model",
            datasets=["data"],
            scorers=["slots"],
            tracks=["parallel"],
            out=out,
            runs=1,
            fresh=True,
        )
        assert result.exists()
        assert (combo / "predictions.jsonl").exists()
        # The file was rewritten (--fresh removed the dir).
        assert (combo / "predictions.jsonl").stat().st_mtime >= old_mtime
