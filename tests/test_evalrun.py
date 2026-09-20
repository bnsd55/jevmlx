"""Tests for the eval runner: contract-shaped output from fake decide_fns."""

from __future__ import annotations

import json

import pytest
from conftest import make_engine, make_engine_result, make_field_telemetry

from jevmlx import evalrun


def _two_cases() -> list[dict]:
    return [
        {
            "id": "wf/case-1",
            "group_id": "g1",
            "source": "quality-eval",
            "workflow": None,
            "schema": {
                "action": {
                    "type": "enum",
                    "description": "action to take",
                    "choices": ["APPROVE", "REVIEW", "BLOCK"],
                },
                "flag": {"type": "boolean", "description": "is it urgent"},
            },
            "context": "customer text",
            "labels": {"action": "REVIEW", "flag": True},
            "split": "train",
            "meta": {},
        },
        {
            "id": "wf/case-2",
            "group_id": "g1",
            "source": "typesafe",
            "workflow": "triage",
            "schema": {
                "priority": {
                    "type": "enum",
                    "description": "priority",
                    "choices": ["P1", "P2", "P3"],
                }
            },
            "context": "app crashes",
            "labels": {},
            "split": "holdout",
            "meta": {},
        },
    ]


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_predictions_contract_lines(tmp_path):
    cases = _two_cases()
    run = evalrun.run_eval(
        cases,
        lambda s, c: {
            "action": {"prediction": "APPROVE", "probability": 0.9},
            "flag": {"prediction": True, "probability": 0.8},
            "_meta": {"latency_ms": 1.0, "rows": 3, "passes": 1},
        },
        track="parallel",
        model="fake/model",
        out_dir=str(tmp_path),
        run_id="r1",
        chat_template="TEMPLATE",
    )
    assert run["counts"] == {"cases": 2, "fields": 4, "prediction_lines": 4}

    lines = _read_lines(tmp_path / "predictions.jsonl")
    assert len(lines) == 4
    first = lines[0]
    assert set(first) == {
        "run_id",
        "case_id",
        "group_id",
        "source",
        "workflow",
        "field",
        "type",
        "track",
        "model",
        "permutation",
        "label",
        "prediction",
        "valid",
        "correct",
        "log_scores",
        "probability",
        "per_option",
        "latency_ms",
        "per_item_end_to_end_ms",
        "rows",
        "passes",
        "error",
        "salvage_prediction",
        "oracle_prediction",
    }
    assert first["run_id"] == "r1"
    assert first["track"] == "parallel"
    assert first["permutation"] == "canonical"
    assert first["latency_ms"] == 1.0 and first["rows"] == 3 and first["passes"] == 1
    # correct is (prediction == label) when the label exists, else None
    assert first["correct"] is (first["prediction"] == first["label"])
    holdout = [line for line in lines if line["case_id"] == "wf/case-2"]
    assert all(line["correct"] is None and line["label"] is None for line in holdout)

    run_json = json.load(open(tmp_path / "run.json"))
    assert run_json["run_id"] == "r1"
    assert run_json["config"]["temperature"] == 1.0
    assert run_json["config"]["track"] == "parallel"
    assert len(run_json["config"]["tokenizer_chat_template_sha256"]) == 64
    assert run_json["environment"]["python_version"]


def test_rotation_permutation_remaps_to_canonical(tmp_path):
    case = {
        "id": "c1",
        "group_id": "g1",
        "source": "custom",
        "workflow": None,
        "schema": {
            "risk": {
                "type": "enum",
                "description": "risk level",
                "choices": ["LOW", "MED", "HIGH"],
            }
        },
        "context": "ctx",
        "labels": {"risk": "HIGH"},
        "split": "train",
        "meta": {},
    }
    seen: list[dict] = []

    def fake_decide(schema_dict, context):
        # Always emit the choice that sits FIRST under the current order, so a
        # rotation visibly changes the winner and the remap must map it back.
        seen.append(json.loads(json.dumps(schema_dict)))
        return {"risk": {"prediction": schema_dict["risk"]["choices"][0]}}

    evalrun.run_eval(
        [case],
        fake_decide,
        track="parallel",
        model="fake/model",
        permutations="rotations",
        out_dir=str(tmp_path),
        run_id="r-rot",
        chat_template="T",
        plan_provider=lambda schema: {"compiled": True},
    )
    lines = _read_lines(tmp_path / "predictions.jsonl")
    # canonical + 2 rotations = 3 lines for the single enum field
    assert len(lines) == 3
    tags = {line["permutation"] for line in lines}
    assert tags == {"canonical", "rot1:risk", "rot2:risk"}
    # every remapped prediction is a canonical choice string
    for line in lines:
        assert line["prediction"] in ("LOW", "MED", "HIGH")
    # the rotated schemas the fake actually saw had permuted choices
    assert seen[0]["risk"]["choices"] == ["LOW", "MED", "HIGH"]
    assert seen[1]["risk"]["choices"] == ["MED", "HIGH", "LOW"]
    assert seen[2]["risk"]["choices"] == ["HIGH", "LOW", "MED"]
    # and the canonical winner remaps: rot1 first choice MED -> canonical MED,
    # rot2 first choice HIGH -> canonical HIGH
    by_tag = {line["permutation"]: line["prediction"] for line in lines}
    assert by_tag["canonical"] == "LOW"
    assert by_tag["rot1:risk"] == "MED"
    assert by_tag["rot2:risk"] == "HIGH"
    # correctness is judged against the label AFTER remapping
    assert [line["correct"] for line in lines] == [False, False, True]

    config = json.load(open(tmp_path / "run.json"))["config"]
    assert config["permutations"] == "rotations"
    assert len(config["tokenizer_chat_template_sha256"]) == 64
    assert len(config["compiled_plan_sha256"]) == 64


def test_fieldperm_variants(tmp_path):
    case = {
        "id": "c",
        "group_id": "g",
        "source": "custom",
        "workflow": None,
        "schema": {
            "a": {"type": "boolean", "description": "a"},
            "b": {"type": "boolean", "description": "b"},
            "c": {"type": "boolean", "description": "c"},
        },
        "context": "ctx",
        "labels": {},
        "split": "train",
        "meta": {},
    }
    evalrun.run_eval(
        [case],
        lambda s, c: {name: {"prediction": True} for name in s},
        track="parallel",
        model="m",
        permutations="fieldperm",
        out_dir=str(tmp_path),
    )
    lines = _read_lines(tmp_path / "predictions.jsonl")
    tags = {line["permutation"] for line in lines}
    assert tags == {"canonical", "fieldperm1", "fieldperm2", "fieldperm3"}
    # 4 variants x 3 fields
    assert len(lines) == 12


def test_split_and_dataset_lock(tmp_path):
    cases = [
        {
            "id": f"c{i}",
            "group_id": "g",
            "source": "custom",
            "workflow": None,
            "schema": {"f": {"type": "boolean", "description": "d"}},
            "context": "x",
            "labels": {},
            "split": "holdout" if i % 2 else "train",
            "meta": {},
        }
        for i in range(4)
    ]
    lock = tmp_path / "dataset.lock.json"
    lock.write_text("{}")

    run = evalrun.run_eval(
        cases,
        lambda s, c: {name: {"prediction": True} for name in s},
        track="parallel",
        model="m",
        split="holdout",
        out_dir=str(tmp_path),
        dataset_lock_path=str(lock),
    )
    assert run["counts"]["cases"] == 2
    lines = _read_lines(tmp_path / "predictions.jsonl")
    assert all(line["label"] is None and line["correct"] is None for line in lines)
    config = json.load(open(tmp_path / "run.json"))["config"]
    assert len(config["dataset_lock_sha256"]) == 64


def test_dataset_lock_sha256_is_the_lock_file_hash(tmp_path):
    """W5c-17: run.json's dataset_lock_sha256 == sha256 of the exact lock
    file that was passed in (was silently null when the bench registered a
    lock path nothing had written)."""
    import hashlib

    cases = [
        {
            "id": "c1",
            "group_id": "g",
            "source": "custom",
            "workflow": None,
            "schema": {"f": {"type": "boolean", "description": "d"}},
            "context": "x",
            "labels": {"f": True},
            "split": "train",
            "meta": {},
        }
    ]
    lock = tmp_path / "bundled.dataset.lock.json"
    lock.write_text('{"cases_sha256": "abc"}', encoding="utf-8")

    run = evalrun.run_eval(
        cases,
        lambda s, c: {name: {"prediction": True} for name in s},
        track="parallel",
        model="m",
        out_dir=str(tmp_path),
        dataset_lock_path=str(lock),
    )
    expected = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert run["config"]["dataset_lock_sha256"] == expected
    written = json.load(open(tmp_path / "run.json"))["config"]["dataset_lock_sha256"]
    assert written == expected


def test_missing_dataset_lock_is_an_error_not_null(tmp_path):
    """W5c-17: a non-None dataset_lock_path whose file does not exist must
    raise OSError — a silent null in run.json breaks the provenance chain
    (leaderboard rows could not be traced to a dataset revision)."""
    cases = [
        {
            "id": "c1",
            "group_id": "g",
            "source": "custom",
            "workflow": None,
            "schema": {"f": {"type": "boolean", "description": "d"}},
            "context": "x",
            "labels": {"f": True},
            "split": "train",
            "meta": {},
        }
    ]
    with pytest.raises(OSError, match="dataset lock file not found"):
        evalrun.run_eval(
            cases,
            lambda s, c: {name: {"prediction": True} for name in s},
            track="parallel",
            model="m",
            out_dir=str(tmp_path),
            dataset_lock_path=str(tmp_path / "nope.dataset.lock.json"),
        )


def test_naive_invalid_prediction_counts_wrong(tmp_path):
    """An unparseable naive output yields valid=false, correct=false (label present)."""
    from jevmlx.baseline import parse_baseline_output
    from jevmlx.schema import StructuredSchema

    schema = StructuredSchema({"flag": {"type": "boolean", "description": "urgent"}})
    strict, salvage, errors = parse_baseline_output("not json at all", schema)
    assert strict["flag"] is None and salvage["flag"] is None and errors

    case = {
        "id": "c",
        "group_id": "g",
        "source": "custom",
        "workflow": None,
        "schema": {"flag": {"type": "boolean", "description": "urgent"}},
        "context": "ctx",
        "labels": {"flag": True},
        "split": "train",
        "meta": {},
    }
    evalrun.run_eval(
        [case],
        lambda s, c: {
            "flag": {
                "prediction": None,
                "valid": False,
                "salvage_prediction": True,  # e.g. JSON valid but trailing text
                "error": "trailing text after JSON object",
            }
        },
        track="naive_local",
        model="m",
        out_dir=str(tmp_path),
    )
    line = _read_lines(tmp_path / "predictions.jsonl")[0]
    assert line["valid"] is False
    assert line["correct"] is False  # invalid => wrong when label present
    assert line["error"]
    assert line["salvage_prediction"] is True  # salvage validity is a diagnostic


def test_load_cases_skips_comments_and_blank(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "# a comment\n"
        "\n" + json.dumps({"id": "c1", "schema": {}, "context": "", "labels": {}}) + "\n"
    )
    cases = evalrun.load_cases(str(path))
    assert len(cases) == 1 and cases[0]["id"] == "c1"


def test_parallel_log_scores_reads_finalized_dict(tmp_path, monkeypatch):
    """Accessor reads the finalized 'log_scores' dict (dict-only contract)."""
    from jevmlx import evalrun as er

    class FakeField:
        field_type = "enum"
        choices = ["A", "B"]

    class FakeSchema:
        fields = {"x": FakeField()}

    class FakeSchemaCtor:
        def __init__(self, schema_dict):
            pass

        def __getattr__(self, name):
            return FakeSchema().fields  # not used

    # parallel_decide_fn returns decide(); we test decide() by monkeypatching
    # run_parallel_generation inside jevmlx.engine.
    calls = {}

    def fake_rpg(
        engine,
        context,
        schema,
        temperature=1.0,
        scoring="slots",
        prior_correction=False,
        constraints=None,
        oracle_overrides=None,
        **kwargs,
    ):
        calls["temperature"] = temperature
        return make_engine_result(
            fields={
                "x": make_field_telemetry(
                    value="A",
                    choices=["A", "B"],
                    probability=0.9,
                    log_scores={"A": -0.1, "B": -2.0},
                )
            }
        )

    import jevmlx.engine as engine_mod

    monkeypatch.setattr(engine_mod, "run_parallel_generation", fake_rpg)
    decide = er.parallel_decide_fn(make_engine())
    result = decide({"x": {"type": "enum", "description": "d", "choices": ["A", "B"]}}, "ctx")
    assert result["x"]["log_scores"] == {"A": -0.1, "B": -2.0}
    assert calls["temperature"] == 1.0


def test_run_json_carries_provenance_keys(tmp_path, monkeypatch):
    """X2: run.json config gains model_revision, quantization, prompt_version
    (parallel track), sourced from engine_metadata."""
    import jevmlx.engine as engine_mod
    import jevmlx.evalrun as er

    monkeypatch.setattr(
        engine_mod,
        "engine_metadata",
        lambda model_id: {
            "model_id": model_id,
            "revision": "abc123",
            "mlx_version": "0.0.0",
            "mlx_lm_version": "0.0.0",
            "quantization": {"bits": 4},
        },
    )
    monkeypatch.setattr(engine_mod, "PROMPT_VERSION", "jevmlx-parallel-v1", raising=False)
    case = {
        "id": "c",
        "group_id": "g",
        "source": "custom",
        "workflow": None,
        "schema": {"f": {"type": "boolean", "description": "d"}},
        "context": "x",
        "labels": {},
        "split": "train",
        "meta": {},
    }
    er.run_eval(
        [case],
        lambda s, c: {name: {"prediction": True} for name in s},
        track="parallel",
        model="fake/model",
        out_dir=str(tmp_path),
    )
    config = json.load(open(tmp_path / "run.json"))["config"]
    assert config["model_revision"] == "abc123"
    assert config["quantization"] == {"bits": 4}
    assert config["prompt_version"] == "jevmlx-parallel-v1"


def test_engine_metadata_resolves_snapshot_sha(tmp_path):
    """X2: engine_metadata resolves the revision from a fake HF cache dir
    (refs/main), including quantization from config.json."""
    import jevmlx.engine as engine_mod

    cache = tmp_path / "hub"
    snap = cache / "models--org--m" / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (cache / "models--org--m" / "refs").mkdir()
    (cache / "models--org--m" / "refs" / "main").write_text("deadbeef\n")
    (snap / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))

    from huggingface_hub import constants

    orig = constants.HF_HUB_CACHE
    constants.HF_HUB_CACHE = str(cache)
    try:
        meta = engine_mod.engine_metadata("org/m")
    finally:
        constants.HF_HUB_CACHE = orig
    assert meta["model_id"] == "org/m"
    assert meta["revision"] == "deadbeef"
    assert meta["quantization"] == {"bits": 4}
    assert meta["mlx_version"]
    assert meta["mlx_lm_version"]


def test_carry_perturbation_flag_passes_meta_through(tmp_path):
    """carry_perturbation=True adds 'perturbation' (str|null) to every line."""
    cases = [
        {**_two_cases()[0], "group_id": "wf/case-1", "meta": {}},  # original: perturbation null
        {
            **_two_cases()[0],
            "id": "wf/case-1#p1",
            "group_id": "wf/case-1",
            "meta": {"perturbation": "ws"},
        },
    ]

    def decide(schema_dict, context):
        return {
            "action": {"prediction": "APPROVE"},
            "flag": {"prediction": True},
        }

    # Default: no perturbation key at all (contract unchanged).
    run_default = evalrun.run_eval(
        cases, decide, track="parallel", model="m", out_dir=str(tmp_path / "a"), run_id="r0"
    )
    lines_default = _read_lines(tmp_path / "a" / "predictions.jsonl")
    assert run_default["counts"]["prediction_lines"] == 4
    assert all("perturbation" not in line for line in lines_default)

    # With the flag: original lines carry None, variant lines carry the kind.
    evalrun.run_eval(
        cases,
        decide,
        track="parallel",
        model="m",
        out_dir=str(tmp_path / "b"),
        run_id="r1",
        carry_perturbation=True,
    )
    lines = _read_lines(tmp_path / "b" / "predictions.jsonl")
    assert len(lines) == 4
    by_case = {line["case_id"]: line["perturbation"] for line in lines}
    assert by_case["wf/case-1"] is None
    assert by_case["wf/case-1#p1"] == "ws"
    # Everything else on the line is unchanged; feeding the two lines sharing
    # group_id into the metric pairs them as original vs variant.
    from jevmlx.evalmetrics import perturbation_flip_rate

    assert perturbation_flip_rate(lines) == 0.0  # same predictions -> no flip


def test_carry_consensus_flag_adds_consensus_distribution(tmp_path):
    """carry_consensus=True puts each line's OWN field distribution on it.

    meta.consensus is the fetchers' ``{field: {choice: p}}``; the line for
    field X carries ``consensus[X]`` (the shape tvd_vs_consensus reads), and a
    field absent from the map gets no key.
    """
    base = _two_cases()[0]
    flag_dist = {"true": 0.9, "false": 0.1}
    cases = [
        {**base, "source": "typesafe", "meta": {"consensus": {"flag": flag_dist}}},
        {**base, "id": "wf/case-2", "group_id": "wf/case-2", "source": "typesafe", "meta": {}},
    ]

    def decide(schema_dict, context):
        return {"action": {"prediction": "APPROVE"}, "flag": {"prediction": True}}

    # Default: no consensus key (contract unchanged).
    evalrun.run_eval(
        cases, decide, track="parallel", model="m", out_dir=str(tmp_path / "a"), run_id="r0"
    )
    lines_default = _read_lines(tmp_path / "a" / "predictions.jsonl")
    assert all("consensus" not in line for line in lines_default)

    # With the flag: lines from cases with a consensus distribution carry it;
    # lines from cases without one do not get an empty key.
    evalrun.run_eval(
        cases,
        decide,
        track="parallel",
        model="m",
        out_dir=str(tmp_path / "b"),
        run_id="r1",
        carry_consensus=True,
    )
    lines = _read_lines(tmp_path / "b" / "predictions.jsonl")
    by_line = {(line["case_id"], line["field"]): line.get("consensus") for line in lines}
    assert by_line[("wf/case-1", "flag")] == flag_dist
    assert by_line[("wf/case-1", "action")] is None  # not in the consensus map
    assert by_line[("wf/case-2", "flag")] is None
    assert by_line[("wf/case-2", "action")] is None


def test_run_eval_writes_timing_json_for_parallel_track(tmp_path):
    """W3-R: the parallel track's _meta timing split is aggregated into
    <combo>/timing.json (median over canonical calls; no second run). A
    _meta without the split keys (mock decide_fn) still yields a file with
    defaults, and the naive track writes none."""
    cases = _two_cases()

    def parallel_decide(schema_dict, context):
        return {
            "priority": {"prediction": "P1", "probability": 0.9},
            "_meta": {
                "latency_ms": 10.0,
                "rows": 3,
                "passes": 1,
                "prior_ms": 0.0,
                "prefill_ms": 5.0,
                "plan_compile_ms": 0.1,
                "cache_broadcast_ms": 0.2,
                "suffix_eval_ms": 4.0,
                "lm_head_gather_ms": 0.3,
                "second_pass_ms": 0.0,
                "total_ms": 10.0,
                "peak_active_bytes": 2048,
                "padded_token_positions": 24,
                "rescored_fields_count": 0,
                "rerun_fields_count": 0,
                "num_fields": 1,
            },
        }

    evalrun.run_eval(
        cases,
        parallel_decide,
        track="parallel",
        model="fake/model",
        out_dir=str(tmp_path),
        run_id="r-timing",
    )
    timing = json.load(open(tmp_path / "timing.json"))
    assert timing["calls"] == 2
    assert timing["median"]["total_ms"] == 10.0
    assert timing["median"]["prefill_ms"] == 5.0
    assert timing["median"]["padded_token_positions"] == 24
    # Contract keys all present with 0 defaults for anything a caller omits.
    assert timing["median"]["prior_ms"] == 0.0
    assert timing["median"]["rescored_fields_count"] == 0

    # The naive track: no timing.json (no engine split exists there).
    out2 = tmp_path / "naive"
    evalrun.run_eval(
        cases,
        lambda s, c: {"priority": {"prediction": "P1", "valid": True}},
        track="naive_local",
        model="fake/model",
        out_dir=str(out2),
        run_id="r-naive",
    )
    assert not (out2 / "timing.json").exists()


# --- W5c-16: heartbeat every N completed cases ------------------------------


def _n_cases(n: int) -> list[dict]:
    """N minimal cases for the heartbeat loop (one enum field each)."""
    return [
        {
            "id": f"hb/case-{i}",
            "group_id": "g1",
            "source": "quality-eval",
            "workflow": None,
            "schema": {
                "action": {
                    "type": "enum",
                    "description": "d",
                    "choices": ["A", "B"],
                }
            },
            "context": "ctx",
            "labels": {"action": "A"},
            "split": "train",
            "meta": {},
        }
        for i in range(n)
    ]


def _hb_decide(s, c):
    return {
        "action": {"prediction": "A", "probability": 0.9},
        "_meta": {"latency_ms": 1.0, "rows": 1, "passes": 1},
    }


class TestHeartbeat:
    def test_heartbeat_every_2_over_5_cases(self, tmp_path, capsys, monkeypatch):
        """N=2 over 5 cases: 2 heartbeat lines printed (at case 2 and 4),
        heartbeat.jsonl has 2 records with the keys."""
        import mlx.core as mx

        # Stub Metal memory so _heartbeat does not touch the real GPU.
        monkeypatch.setattr(mx, "get_peak_memory", lambda: 0)
        monkeypatch.setattr(mx, "get_active_memory", lambda: 0)
        monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)

        evalrun.run_eval(
            _n_cases(5),
            _hb_decide,
            track="parallel",
            model="fake/model",
            out_dir=str(tmp_path),
            run_id="hb1",
            heartbeat_every=2,
            combo="parallel-trie-bundled",
        )
        out = capsys.readouterr().out
        hb_lines = [line for line in out.splitlines() if line.startswith("[heartbeat]")]
        assert len(hb_lines) == 2, f"expected 2 heartbeats, got {len(hb_lines)}: {hb_lines}"
        assert "cases_done=2" in hb_lines[0]
        assert "cases_done=4" in hb_lines[1]
        assert "parallel-trie-bundled" in hb_lines[0]
        assert "peak=" in hb_lines[0] and "active=" in hb_lines[0] and "cache=" in hb_lines[0]

        hb_records = [
            json.loads(line) for line in (tmp_path / "heartbeat.jsonl").read_text().splitlines()
        ]
        assert len(hb_records) == 2
        for rec in hb_records:
            assert set(rec) == {
                "combo",
                "cases_done",
                "pred_lines",
                "elapsed_s",
                "peak_memory_bytes",
                "active_memory_bytes",
                "cache_memory_bytes",
            }
        assert hb_records[0]["cases_done"] == 2
        assert hb_records[1]["cases_done"] == 4

    def test_heartbeat_disabled_when_zero(self, tmp_path, capsys, monkeypatch):
        """N=0: no heartbeat lines, no heartbeat.jsonl."""
        import mlx.core as mx

        monkeypatch.setattr(mx, "get_peak_memory", lambda: 0)
        monkeypatch.setattr(mx, "get_active_memory", lambda: 0)
        monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)

        evalrun.run_eval(
            _n_cases(5),
            _hb_decide,
            track="parallel",
            model="fake/model",
            out_dir=str(tmp_path),
            run_id="hb2",
            heartbeat_every=0,
            combo="parallel-trie-bundled",
        )
        out = capsys.readouterr().out
        assert "[heartbeat]" not in out
        assert not (tmp_path / "heartbeat.jsonl").exists()
