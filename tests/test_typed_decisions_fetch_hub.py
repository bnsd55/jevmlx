"""fetch_all end-to-end with the Hub and parquet reads monkeypatched.

Covers F6/F9: the default revision is DEFAULT_REVISION, any --revision value
(branch name or sha) is resolved to a commit sha BEFORE downloading, every
shard shares the resolved revision, per-shard sha256 files land in the lock
sources, and write_outputs pins them into dataset.lock.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import benchmarks.typed_decisions.fetch as fetch_mod
from benchmarks.typed_decisions.fetch import DEFAULT_REVISION, fetch_all, write_outputs


class _FakeApi:
    """dataset_info stub mapping revisions (branch or sha) to commit shas."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def dataset_info(self, repo_id: str, revision: str | None = None):
        class _Info:
            sha = self._mapping[revision or "main"]

        return _Info()


def _mk_row(workflow: str, i: int) -> dict:
    return {
        "id": f"{workflow}_{i:06d}",
        "workflow": workflow,
        "split": "test",
        "state": json.dumps({"i": i}),
        "questions": json.dumps({"q": {"type": "noul", "instructions": "ok?", "criteria": {}}}),
        "gold": json.dumps({"q": {"type": "noul", "label": "true"}}),
    }


def test_fetch_all_resolves_default_revision_and_records_shards(tmp_path, monkeypatch):
    resolved: list[str] = []

    def fake_resolve(revision: str) -> str:
        resolved.append(revision)
        return "sha-main-0001" if revision == "main" else f"sha-for-{revision}"

    def fake_download(workflow: str, split: str, revision: str):
        assert revision == f"sha-for-{DEFAULT_REVISION}"
        path = tmp_path / f"{workflow}-{split}.parquet"
        path.write_bytes(f"bytes-{workflow}-{split}".encode())
        return path, revision

    monkeypatch.setattr(fetch_mod, "_resolve_revision", fake_resolve)
    monkeypatch.setattr(fetch_mod, "_download", fake_download)
    monkeypatch.setattr(fetch_mod, "iter_rows", lambda path: iter([_mk_row("w1", 1)]))

    records, summary, source = fetch_all(["w1"], ["test"])

    # The default pins to DEFAULT_REVISION, resolved to a sha before the loop.
    assert resolved == [DEFAULT_REVISION]
    assert source["revision"] == f"sha-for-{DEFAULT_REVISION}"
    assert len(source["files"]) == 1
    shard = source["files"][0]
    assert shard["file"] == "w1/test-00000-of-00001.parquet"
    assert shard["sha256"] == hashlib.sha256(b"bytes-w1-test").hexdigest()
    assert summary["records"] == 1
    assert records[0]["labels"] == {"q": True}


def test_fetch_all_resolves_branch_revision_to_one_sha(tmp_path, monkeypatch):
    """A branch name resolves once; every shard downloads the same sha."""
    downloads: list[str] = []

    def fake_resolve(revision: str) -> str:
        return "sha-v2-fixed" if revision == "main-branch" else revision

    def fake_download(workflow: str, split: str, revision: str):
        downloads.append((workflow, split, revision))
        path = tmp_path / f"{workflow}-{split}.parquet"
        path.write_bytes(f"bytes-{workflow}-{split}".encode())
        return path, revision

    monkeypatch.setattr(fetch_mod, "_resolve_revision", fake_resolve)
    monkeypatch.setattr(fetch_mod, "_download", fake_download)
    monkeypatch.setattr(
        fetch_mod, "iter_rows", lambda path: iter([_mk_row("w1", 1), _mk_row("w2", 2)])
    )

    records, summary, source = fetch_all(["w1", "w2"], ["train", "test"], "main-branch")

    assert {rev for _, _, rev in downloads} == {"sha-v2-fixed"}
    assert source["revision"] == "sha-v2-fixed"
    assert len(source["files"]) == 4  # 2 workflows x 2 splits, each with its sha256
    assert {f["sha256"] for f in source["files"]} == {
        hashlib.sha256(b"bytes-w1-train").hexdigest(),
        hashlib.sha256(b"bytes-w1-test").hexdigest(),
        hashlib.sha256(b"bytes-w2-train").hexdigest(),
        hashlib.sha256(b"bytes-w2-test").hexdigest(),
    }


def test_write_outputs_lock_carries_sources(tmp_path):
    record = {
        "id": "x",
        "group_id": "x",
        "source": "typed-decisions",
        "workflow": "w1",
        "benchmark_only": True,
        "schema": {},
        "context": "{}",
        "labels": {},
        "split": "test",
        "meta": {},
    }
    source = {
        "repo_id": "LocalLLaMA/typed-decisions",
        "revision": "sha-v2-fixed",
        "files": [{"file": "w1/test-00000-of-00001.parquet", "sha256": "aa"}],
    }
    lock_path = write_outputs([record], tmp_path / "c.jsonl", source, {}, None)
    lock = json.loads(Path(lock_path).read_text())
    assert lock["sources"] == [source]
    assert lock["cases_sha256"] == hashlib.sha256((tmp_path / "c.jsonl").read_bytes()).hexdigest()
