"""End-to-end test of benchmarks/m5.py on REAL-shaped artifacts (W5c-M5-E2E).

Why this file exists: the M5 7B smoke ran 8 hours and crashed at the LAST
step (SUMMARY) on a report.json shape the unit tests never saw — typesafe's
``agreement`` is a dict ({overall, by_workflow, ...}), not a float. PR #98
patched the formatter; this file makes sure every step of the runbook is
exercised against artifacts produced by the REAL writers, so the next shape
drift fails here — in seconds, offline — instead of after an 8-hour run.

What runs against REAL code (no model ever loads):
- jevmlx.evalrun.run_eval with the fake engine (conftest.make_engine /
  parallel_decide_fn / naive_local_decide_fn) over three dataset shapes:
  (a) bundled-shaped (float agreement absent, accuracy present),
  (b) typesafe-shaped with workflows (agreement = dict with overall/
      by_workflow, consensus distributions carried),
  (c) perturbed-shaped (#p variants sharing group_id, meta.perturbation),
  on both tracks (parallel with rotations, naive_local) and both scorers
  (slots/labels) — exactly the combos the M5 bench step produces. Every
  report.json / run.json / timing.json / predictions.jsonl comes from
  jevmlx.evalrun + evalreport + evalmetrics, byte-for-byte the shapes M5 saw.
- benchmarks.timing.run_timing (real aggregate/extract_telemetry) with an
  injectable fake engine -> timing-<model>.json, the file the timing step
  writes.
- benchmarks.invariance.main() (the real ladder + real
  invariance.json/summary.md writers) with a fake engine over a typesafe
  dataset.
- jevmlx.parity.write_parity_json (real, small hand-built cases — the
  bundled 255-choice preset hangs the fake-tokenizer codebook search, so
  the parity.json the bench step writes is reproduced with equivalent
  shape over tiny schemas) and benchmarks.probe (slope + adapters) for the
  --probe step payloads.
- benchmarks.summarize_results.summarize (the bench step's REAL SUMMARY
  writer) over the results folder built above.
- benchmarks.m5.main() END TO END with subprocess.run monkeypatched to a
  fake runner that maps each planned step's argv to the real artifact
  builders from this file. RUNBOOK.md, SUMMARY.md, skip/fresh markers,
  failure recording, and the caffeinate re-exec guard are all asserted on
  the real planner/executor code.

The _fmt coverage check at the bottom walks EVERY f-string _fmt call site
(build_summary_text + summarize path) and asserts every key it touches
exists in the real artifacts with the type _fmt expects — the test that
would have caught the M5 crash.
"""

from __future__ import annotations

import json
import subprocess  # noqa: F401 - re-exported for the fake runner's subprocess.run guard
from pathlib import Path

import pytest
from conftest import YNLogitModel, _Mod97Tokenizer, make_engine

import benchmarks.m5 as m5
from benchmarks.m5 import build_summary_text, collect_side, plan_steps

QUALITY_TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"
REST_MODEL = "mlx-community/Llama-3.1-8B-Instruct-4bit"


# ----------------------------------------------------------------- builders


def _eid(c: str) -> int:
    """Scripted-tokenizer char id: printable 32..126 -> ids 2..96; id 1 = eos."""
    return ord(c) - 30


class _ScriptTokenizer(_Mod97Tokenizer):
    """Char tokenizer with an encode/decode pair the naive track can round-trip."""

    eos_token_id = 1
    unk_token_id = -1

    def convert_tokens_to_ids(self, tok):
        return None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [_eid(c) for c in text]

    def decode(self, ids, **kwargs) -> str:
        return "".join(chr(t + 30) for t in ids if t >= 2)


class _ScriptedNaiveModel(YNLogitModel):
    """Naive-generation model that emits a fixed JSON body token by token.

    The engine seeds ``current_text`` with ``'{\\n  '``; this model replays
    the remainder ('"action": "APPROVE"}') then eos, so parse_baseline_output
    sees a valid JSON object on every case.
    """

    def __init__(self, body: str = '"action": "APPROVE"}'):
        super().__init__()
        self._body_ids = [_eid(c) for c in body] + [1]
        self._i = 0

    def __call__(self, tokens, cache=None):
        out = super().__call__(tokens, cache)
        n = int(tokens.shape[1])
        self._i = 0 if n > 1 else self._i + 1
        nxt = self._body_ids[min(self._i, len(self._body_ids) - 1)]
        return out.at[:, :, nxt].add(50.0)


def _bundled_cases(n: int = 4) -> list[dict]:
    """Bundled-shaped cases (source=quality-eval, no workflow/consensus)."""
    return [
        {
            "id": f"quality-eval/payment_risk/A{i:02d}",
            "group_id": f"quality-eval/payment_risk/A{i:02d}",
            "source": "quality-eval",
            "workflow": None,
            "schema": {
                "action": {
                    "type": "enum",
                    "description": "Recommended action",
                    "choices": ["APPROVE", "REVIEW", "BLOCK"],
                },
                "fraud": {"type": "boolean", "description": "Whether fraudulent"},
            },
            "context": f"Transaction alert #{i}. Clear low risk.",
            "labels": {"action": "APPROVE", "fraud": False},
            "split": "train",
            "meta": {},
        }
        for i in range(1, n + 1)
    ]


def _typesafe_cases(n: int = 3) -> list[dict]:
    """TypeSafe-shaped cases: source=typesafe, workflows, consensus dicts."""
    cases = []
    workflows = ["security_incidents", "invoice_processing"]
    for i in range(1, n + 1):
        wf = workflows[i % len(workflows)]
        cases.append(
            {
                "id": f"typesafe/{wf}/case-{i}",
                "group_id": f"typesafe/{wf}/case-{i}",
                "source": "typesafe",
                "workflow": wf,
                "schema": {
                    "evidence_strength": {
                        "type": "enum",
                        "description": "Evidence strength",
                        "choices": ["SPECULATIVE", "MODERATE", "STRONG"],
                    },
                    "is_true_positive": {"type": "boolean", "description": "TP?"},
                },
                "context": f"Incident record {i}: unauthorized access flagged.",
                "labels": {"evidence_strength": "MODERATE", "is_true_positive": True},
                "split": "train",
                "meta": {
                    "consensus": {
                        "evidence_strength": {"SPECULATIVE": 0.2, "MODERATE": 0.6, "STRONG": 0.2},
                        "is_true_positive": {"true": 0.75, "false": 0.25},
                    },
                    "ambiguous": ["evidence_strength"],
                    "margin": {"evidence_strength": 0.4, "is_true_positive": 0.5},
                },
            }
        )
    return cases


def _write_jsonl(path: Path, cases: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


def _run_combo(
    out: Path,
    combo: str,
    cases: list[dict],
    engine,
    *,
    scorer: str,
    track: str,
    dataset_name: str,
) -> Path:
    """One bench combo EXACTLY as jevmlx.bench._run_one produces it: run_eval
    (real, fake engine) + write_report (real) + the parity.json writer the
    bench runs per model folder. Returns the model folder."""
    from jevmlx.evalmetrics import compute_metrics, load_predictions
    from jevmlx.evalreport import environment, write_report
    from jevmlx.evalrun import naive_local_decide_fn, parallel_decide_fn, run_eval

    model_slug = QUALITY_TARGET.lower().replace("/", "--")
    model_dir = out / f"m2pro-32gb-{model_slug}"
    combo_dir = model_dir / combo
    combo_dir.mkdir(parents=True, exist_ok=True)

    if track == "parallel":
        decide_fn = parallel_decide_fn(engine, scoring=scorer)
        permutations = "rotations"
    else:
        # The naive track needs a tokenizer with decode(); the scripted model
        # free-writes the JSON the parser wants. Same writer, real code.
        naive_engine = make_engine(
            _ScriptedNaiveModel(), _ScriptTokenizer(), model_id="fake/stable"
        )
        decide_fn = naive_local_decide_fn(naive_engine)
        permutations = "none"

    run_eval(
        cases,
        decide_fn,
        track=track,
        model=QUALITY_TARGET,
        permutations=permutations,
        split="all",
        out_dir=str(combo_dir),
        extra_config={"scoring": scorer if track == "parallel" else "slots"},
        dataset_path=str(out / f"{dataset_name}.jsonl"),
        carry_consensus=True,
    )
    records = load_predictions(combo_dir / "predictions.jsonl")
    write_report(
        combo_dir / "report.json",
        {"environment": environment(), "metrics": compute_metrics(records)},
    )
    return model_dir


def _write_model_parity(model_dir: Path, engine) -> Path:
    """The bench's per-model parity.json (real writer, tiny hand-built cases).

    The bundled presets include high_cardinality_255, whose 255-choice
    codebook search takes >25 s under the fake tokenizer's %97 codebook
    (real tokenizers have dense contiguous ids; the fake does not), so the
    parity payload is produced over equivalent tiny schemas instead — same
    writer, same shape (status/max_*_drift_nats/winners_identical/passed).
    """
    import jevmlx.driftenv as dv
    from jevmlx.parity import write_parity_json

    dv._PROBES_DIR_OVERRIDE = model_dir / "parity-probes"
    try:
        write_parity_json(
            engine,
            QUALITY_TARGET,
            model_dir,
            [
                (
                    "mini",
                    {
                        "schema": {
                            "flag": {"type": "boolean", "description": "d"},
                            "pick": {"type": "enum", "description": "d", "choices": ["yes", "no"]},
                        },
                        "context": "Parity mini context: routine, all checks passed.",
                    },
                ),
                (
                    "mini2",
                    {
                        "schema": {
                            "grade": {
                                "type": "enum",
                                "description": "d",
                                "choices": ["a", "b", "c"],
                            },
                            "ok": {"type": "boolean", "description": "d"},
                        },
                        "context": "Parity mini2 context: escalation, two checks failing.",
                    },
                ),
            ],
        )
    finally:
        dv._PROBES_DIR_OVERRIDE = None
    return model_dir


def _build_bench_results(out: Path) -> Path:
    """The bench step's results tree (real writers): 6 combos (2 tracks x
    2 scorers x 2 bundled/typesafe datasets — the shape M5 actually ran)
    + perturbed + model parity.json, under <out>/bench-quality/ (the
    directory the m5 bench step passes as --out; collect_side globs
    bench-quality/**/report.json)."""
    bench_out = out if out.name == "bench-quality" else out / "bench-quality"
    bench_out.mkdir(parents=True, exist_ok=True)
    engine = make_engine(YNLogitModel(), _Mod97Tokenizer(), model_id="fake/stable")
    bundled = _write_jsonl(out / "bundled.jsonl", _bundled_cases())
    typesafe = _write_jsonl(out / "typesafe.jsonl", _typesafe_cases())
    del bundled, typesafe  # paths only referenced via dataset_path above

    model_dir: Path | None = None
    for scorer in ("slots", "labels"):
        for dataset_name, cases in (("bundled", _bundled_cases()), ("typesafe", _typesafe_cases())):
            model_dir = _run_combo(
                bench_out,
                f"parallel-{scorer}-{dataset_name}",
                cases,
                engine,
                scorer=scorer,
                track="parallel",
                dataset_name=dataset_name,
            )
    # The naive track (scorer-independent; dataset bundled + perturbed).
    model_dir = _run_combo(
        bench_out,
        "naive_local-slots-bundled",
        _bundled_cases(2),
        engine,
        scorer="slots",
        track="naive_local",
        dataset_name="bundled",
    )
    # Perturbed dataset shape: originals + #p variants sharing group_id.
    perturbed = _bundled_cases(2)
    for k in (1, 2):
        variant = dict(perturbed[0])
        variant["id"] = f"{perturbed[0]['id']}#p{k}"
        variant["group_id"] = perturbed[0]["id"]
        variant["meta"] = {"perturbation": ("ws", "numfmt")[k - 1]}
        perturbed.append(variant)
    _run_combo(
        bench_out,
        "parallel-slots-perturbed",
        perturbed,
        engine,
        scorer="slots",
        track="parallel",
        dataset_name="perturbed",
    )
    assert model_dir is not None
    return _write_model_parity(model_dir, engine)


def _build_timing_report(out: Path, engine) -> Path:
    """The timing step's artifact via benchmarks.timing.run_timing (real
    aggregate/extract) with the fake engine, then written under the name
    the timing step's argv produces (timing-<resolved model>.json)."""
    from benchmarks.timing import run_timing
    from jevmlx.cli import load_preset
    from jevmlx.engine import run_parallel_generation
    from jevmlx.models import resolve_model

    def fake_load_preset(_rel: str) -> dict:
        preset = dict(load_preset("fintech_fraud"))
        preset["schema"] = dict(list(preset["schema"].items())[:3])
        preset["id"] = "mini_fraud"
        return preset

    report = run_timing(
        QUALITY_TARGET,
        ["fintech_fraud.json"],
        2,
        False,
        load_engine_fn=lambda _m: engine,
        run_fn=lambda eng, ctx, schema, **kw: run_parallel_generation(eng, ctx, schema, **kw),
        load_preset_fn=fake_load_preset,
    )
    name = f"timing-{resolve_model(QUALITY_TARGET).replace('/', '_')}.json"
    (out / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out / name


def _build_invariance(out: Path, engine) -> Path:
    """The invariance step's artifacts via benchmarks.invariance.main() —
    the REAL ladder, run_eval loop, invariance_metrics, and JSON writers —
    with the fake engine over a small typesafe dataset."""
    import benchmarks.invariance as inv
    import jevmlx.engine as engine_mod

    data = _write_jsonl(out / "typesafe.jsonl", _typesafe_cases(2))
    original = engine_mod.load_engine
    engine_mod.load_engine = lambda _m: engine
    try:
        rc = inv.main(
            [
                "--model",
                "fake/stable",
                "--data",
                str(data),
                "--out",
                str(out / "invariance"),
                "--extra",
                "1,5",
            ]
        )
    finally:
        engine_mod.load_engine = original
    assert rc == 0
    return out / "invariance" / "invariance.json"


def _build_summary_md(out: Path) -> Path:
    """The bench step's SUMMARY.md via benchmarks.summarize_results.summarize
    (the real writer) over the bench step's results dir (the step runs
    `jevmlx bench ... --out <runbook>/bench-quality`, and summarize reads
    one level of machine-model folders under it)."""
    from benchmarks.summarize_results import summarize

    return summarize(out / "bench-quality")


def _build_probe_json(out: Path, command: str) -> Path:
    """A --probe step payload via benchmarks.probe's REAL probe functions
    (the m5 probe steps shell out to `python -m benchmarks.probe`, whose
    main() calls exactly these after load_engine — the fake model replaces
    the load)."""
    from benchmarks.probe import probe_adapters, probe_slope
    from tests.test_adapters import _std_model

    model = _std_model(tied=False, vocab=64, hidden=16, mt="qwen2")
    payload: dict = {"model": QUALITY_TARGET}
    if command == "slope":
        payload["slope"] = probe_slope(model, widths=(4, 8), batches=(1, 2))
    else:
        from jevmlx.adapters import adapter_for  # noqa: F401 - probe imports it too

        specs = [
            (
                "mini",
                {
                    "schema": {
                        "flag": {"type": "boolean", "description": "d"},
                        "pick": {"type": "enum", "description": "d", "choices": ["yes", "no"]},
                    },
                    "context": "c",
                },
            ),
            (
                "mini2",
                {
                    "schema": {
                        "grade": {"type": "enum", "description": "d", "choices": ["a", "b", "c"]},
                        "ok": {"type": "boolean", "description": "d"},
                    },
                    "context": "c2",
                },
            ),
        ]
        payload["adapters"] = probe_adapters(model, _ScriptTokenizer(), specs, max_cases=2)
    path = out / f"probe-{command}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


_TEMPLATE_DIR: Path | None = None


def _artifact_template() -> Path:
    """Build the full artifact tree ONCE per pytest session (the real writers
    take ~7.5 s: bench ~1.7 s, invariance ~4.3 s, timing ~1.5 s); every test
    then gets an INSTANT shutil.copytree copy (~0.03 s). Copies are writable
    and independent — tests that mutate the tree (skip/fresh markers) never
    see each other's state.

    The template carries the three PIECE dirs the fake runner copies into
    step targets too: m5-run/bench-quality, m5-run/invariance, and the
    timing-*.json file."""
    global _TEMPLATE_DIR
    if _TEMPLATE_DIR is None:
        import tempfile

        _TEMPLATE_DIR = Path(tempfile.mkdtemp(prefix="m5-e2e-template-"))
        out = _TEMPLATE_DIR / "m5-run"
        out.mkdir()
        _build_bench_results(out)

        def _engine():
            return make_engine(YNLogitModel(), _Mod97Tokenizer(), model_id="fake/stable")

        _build_timing_report(out, _engine())
        _build_invariance(out, _engine())
        _build_summary_md(out)
    return _TEMPLATE_DIR


def _copy_template_piece(piece: str, dest: Path) -> None:
    """Copy one template piece (bench-quality | invariance | timing) into a
    fresh run dir — the fake runner's builders, ~0.03 s instead of ~7.5 s.
    The pieces are byte-identical to what the real writers produce (they
    ARE the real writers' output, built once)."""
    import shutil

    template_run = _artifact_template() / "m5-run"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if piece == "timing":
        for timing in sorted(template_run.glob("timing-*.json")):
            shutil.copy2(timing, dest / timing.name)
        return
    if piece == "bench-quality":
        target = dest if dest.name == "bench-quality" else dest / "bench-quality"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(template_run / "bench-quality", target)
        # The template's bench SUMMARY.md already exists; regenerate is not
        # needed — it was written by the real summarizer at template build.
        return
    if piece == "invariance":
        target = dest / "invariance"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(template_run / "invariance", target)
        return
    raise ValueError(f"unknown template piece: {piece}")


@pytest.fixture(autouse=True)
def _reset_logging_handlers():
    """The real run_eval/evalreport code calls jevmlx.log.configure(), which
    adds a StreamHandler bound to the CURRENT sys.stderr. pytest swaps
    sys.stderr per-test for capture; after teardown that stream closes, and a
    later log emit (from another test) raises 'I/O operation on closed file'.
    Snapshot the root logger's handlers before each test and restore them
    after, so no stale captured-stream handler leaks across tests."""
    import logging

    root = logging.getLogger()
    saved = list(root.handlers)
    yield
    for h in list(root.handlers):
        if h not in saved:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass


@pytest.fixture
def artifacts(tmp_path_factory):
    """A fresh, writable copy of the real-writer artifact tree (see
    _artifact_template for why: build once, copy per test)."""
    import shutil

    root = tmp_path_factory.mktemp("m5-e2e")
    shutil.copytree(_artifact_template() / "m5-run", root / "m5-run")
    return root / "m5-run"


# ------------------------------------------------------- real-shaped checks


class TestRealArtifactShapes:
    """The artifacts (1) produce: every shape the summary builder consumes."""

    def test_bundled_report_has_float_accuracy_no_agreement(self, artifacts):
        report = json.loads(
            (
                artifacts
                / "bench-quality"
                / "m2pro-32gb-mlx-community--qwen2.5-7b-instruct-4bit"
                / "parallel-slots-bundled"
                / "report.json"
            ).read_text()
        )
        metrics = report["metrics"]
        assert isinstance(metrics.get("accuracy"), float)
        assert metrics.get("agreement") is None  # bundled: no typesafe source

    def test_typesafe_report_agreement_is_dict_with_overall(self, artifacts):
        """THE shape that crashed the 8-hour run: agreement = dict."""
        report = json.loads(
            (
                artifacts
                / "bench-quality"
                / "m2pro-32gb-mlx-community--qwen2.5-7b-instruct-4bit"
                / "parallel-slots-typesafe"
                / "report.json"
            ).read_text()
        )
        agreement = report["metrics"]["agreement"]
        assert isinstance(agreement, dict)
        assert isinstance(agreement["overall"], float)
        assert isinstance(agreement["by_workflow"], dict)
        assert set(agreement) == {
            "overall",
            "by_workflow",
            "agreement_common_subset",
            "n_fields",
            "n_cases",
        }

    def test_perturbed_report_has_perturbation_flip_rate(self, artifacts):
        report = json.loads(
            (
                artifacts
                / "bench-quality"
                / "m2pro-32gb-mlx-community--qwen2.5-7b-instruct-4bit"
                / "parallel-slots-perturbed"
                / "report.json"
            ).read_text()
        )
        rate = report["metrics"]["perturbation_flip_rate"]
        assert isinstance(rate, float) and 0.0 <= rate <= 1.0

    def test_naive_combo_writes_timing_json_call_level(self, artifacts):
        combo = (
            artifacts
            / "bench-quality"
            / "m2pro-32gb-mlx-community--qwen2.5-7b-instruct-4bit"
            / "naive_local-slots-bundled"
        )
        timing = json.loads((combo / "timing.json").read_text())
        assert timing["calls"] >= 1
        assert isinstance(timing["median"]["total_ms"], (int, float))

    def test_run_json_counts_and_lock_sha(self, artifacts):
        run = json.loads(
            (
                artifacts
                / "bench-quality"
                / "m2pro-32gb-mlx-community--qwen2.5-7b-instruct-4bit"
                / "parallel-slots-typesafe"
                / "run.json"
            ).read_text()
        )
        assert set(run) >= {"run_id", "environment", "config", "counts"}
        assert run["config"]["track"] == "parallel"
        assert run["config"]["permutations"] == "rotations"
        # No lock file passed -> sha stays None here; the sha contract itself
        # is owned by test_evalrun (W5c-17).
        assert "dataset_lock_sha256" in run["config"]

    def test_timing_json_preset_aggregate_shape(self, artifacts):
        timing = json.loads(
            (artifacts / "timing-mlx-community_Qwen2.5-7B-Instruct-4bit.json").read_text()
        )
        for preset in timing["presets"].values():
            agg = preset["aggregate"]
            assert isinstance(agg["total_ms"]["median"], (int, float))
            assert isinstance(agg["peak_active_bytes"]["median"], (int, float))

    def test_invariance_json_targets_rungs_shape(self, artifacts):
        inv = json.loads((artifacts / "invariance" / "invariance.json").read_text())
        assert inv["benchmark"] == "invariance"
        for target in inv["targets"]:
            assert isinstance(target["field"], str)
            for rung in target["rungs"].values():
                assert "flip_rate" in rung and "winner_logodds_drift_mean" in rung

    def test_parity_json_gate_shape(self, artifacts):
        parity = json.loads(
            (
                artifacts
                / "bench-quality"
                / "m2pro-32gb-mlx-community--qwen2.5-7b-instruct-4bit"
                / "parity.json"
            ).read_text()
        )
        assert parity["status"] in ("PASS", "DRIFT", "FAIL")
        assert isinstance(parity["passed"], bool)
        assert isinstance(parity["winners_identical"], bool)
        assert isinstance(parity["max_abs_drift_nats"], (int, float))
        assert isinstance(parity["max_raw_row_drift_nats"], (int, float))
        assert isinstance(parity["atol"], (int, float))

    def test_bench_summary_md_rows_have_real_numbers(self, artifacts):
        summary = (artifacts / "bench-quality" / "SUMMARY.md").read_text()
        # The summarizer splits combo names into track/scorer/dataset columns.
        for track, scorer, dataset in (
            ("parallel", "slots", "typesafe"),
            ("parallel", "labels", "bundled"),
            ("naive_local", "slots", "bundled"),
            ("parallel", "slots", "perturbed"),
        ):
            assert f"| {track} | {scorer} | {dataset} |" in summary
        assert "parity_failed" not in summary  # the fake engine passes parity

    def test_collect_side_reads_the_real_tree(self, artifacts):
        """collect_side (what the m5 summary step feeds build_summary_text)
        over the REAL artifacts: typesafe agreement dict must survive."""
        side = collect_side(artifacts)
        combos = {row["combo"]: row for row in side["combos"]}
        ts = combos["parallel-slots-typesafe"]
        assert isinstance(ts["agreement"], dict)  # collect_side does NOT flatten it
        assert isinstance(ts["accuracy"], float)
        assert combos["naive_local-slots-bundled"]["total_ms"] is not None
        assert side["timing_report"]  # timing-*.json picked up
        assert side["invariance"]["fields"]


class TestFmtCoverageOnRealArtifacts:
    """(3) EVERY key _fmt touches exists in the real artifacts with the type
    _fmt expects. This is the test that would have caught the M5 crash: a
    new metric shape (a dict where _fmt expects a number) fails HERE."""

    @staticmethod
    def _fmt(value, pct: bool = False) -> str:
        return m5._fmt(value, pct=pct)

    def test_summary_table_cells_format_without_crash(self, artifacts):
        side = collect_side(artifacts)
        # The exact render path the m5 summary step runs, on real artifacts.
        text = build_summary_text(side, None, parity_models=[QUALITY_TARGET, REST_MODEL])
        assert "# M5 runbook summary" in text
        assert "A/B not run" in text
        assert "|" in text
        assert "dict" not in text.lower()  # no dict.__format__ leak, no crash text

    def test_combo_row_metric_types_match_fmt_expectations(self, artifacts):
        for row in collect_side(artifacts)["combos"]:
            for key in ("agreement", "accuracy", "total_ms", "peak_gb"):
                value = row[key]
                if value is None:
                    continue
                # _fmt formats numbers; a dict is only allowed for agreement
                # (its 'overall' is extracted downstream) — everything else
                # must be a plain number.
                if key == "agreement":
                    assert isinstance(value, float | dict), (row["combo"], key, type(value))
                    if isinstance(value, dict):
                        assert isinstance(value.get("overall"), float)
                else:
                    assert isinstance(value, int | float), (row["combo"], key, type(value))

    def test_every_build_summary_text_key_present_with_right_type(self, artifacts):
        """Walk every (dict, key) access build_summary_text/_fmt can make and
        assert the real artifacts carry it with the expected type."""
        side = collect_side(artifacts)
        # main combos
        for row in side["combos"]:
            assert isinstance(row["combo"], str)
        # timing report entries: _fmt(total_ms_median), _fmt(peak_gb)
        for entry in (side.get("timing_report") or {}).values():
            total = entry.get("total_ms_median")
            peak = entry.get("peak_gb")
            assert total is None or isinstance(total, int | float)
            assert peak is None or isinstance(peak, int | float)
        # invariance rollup: _fmt(mean_flip_rate, pct), _fmt(mean_drift),
        # per-field _fmt(flip40, pct) + _fmt(drift40)
        inv = side.get("invariance") or {}
        if inv.get("fields"):
            mean_flip = inv.get("mean_flip_rate")
            mean_drift = inv.get("mean_drift")
            assert mean_flip is None or isinstance(mean_flip, int | float)
            assert mean_drift is None or isinstance(mean_drift, int | float)
            for entry in inv["fields"].values():
                flip40 = entry.get("flip40")
                drift40 = entry.get("drift40")
                assert flip40 is None or isinstance(flip40, int | float)
                assert drift40 is None or isinstance(drift40, int | float)

    def test_typesafe_agreement_overall_is_a_float(self, artifacts):
        """The crash invariant: wherever a report.json stores agreement as a
        dict, _combo_agreement extracts a float and _fmt renders a percent."""
        side = collect_side(artifacts)
        ts_rows = [r for r in side["combos"] if "typesafe" in r["combo"]]
        assert ts_rows, "no typesafe combo in the artifact tree"
        for row in ts_rows:
            value = m5._combo_agreement(row)
            assert isinstance(value, float), (row["combo"], value)
            # And the formatter accepts it end to end.
            assert self._fmt(value, pct=True).endswith("%")

    def test_new_metric_shape_would_fail_here(self, artifacts):
        """Guard the guard: if a combo's accuracy became a dict (the exact
        M5 failure mode one key over), this test fails instead of the 8-hour
        run's SUMMARY step."""
        for row in collect_side(artifacts)["combos"]:
            accuracy = row["accuracy"]
            assert accuracy is None or isinstance(accuracy, int | float), (
                f"{row['combo']}: accuracy is {type(accuracy).__name__} — a shape "
                "_fmt cannot format; extend _fmt/_combo_agreement BEFORE the next M5 run"
            )


class TestPlannedArgvParses:
    """B1 ROOT-CAUSE guard: every argv that benchmarks/m5.py plan_steps builds
    for OUR OWN CLI/module must parse against the REAL argparse — parse-only,
    no execution.

    The old e2e test faked subprocess.run and never validated argv, so a CLI
    contract break (``bench --models-file`` requiring ``--model``) passed the
    e2e test and only failed in the field (M5 full list, 9d4ddbc). This class
    catches that class of bug in seconds, offline.
    """

    @staticmethod
    def _parse_planned_argv(argv: tuple[str, ...]) -> None:
        """Parse one planned step's argv against the real parser for its
        target CLI/module. Raises (SystemExit from argparse) if the argv
        does not parse — the failure mode that escaped the old e2e.

        argv[0] is the interpreter/binary path; we key on basename + the
        subcommand (``jevmlx <cmd>`` or ``python -m benchmarks.<mod>``).
        """
        bin0 = argv[0].split("/")[-1]
        submod = argv[2] if len(argv) > 2 and argv[1] == "-m" else ""
        rest = list(argv[1:])

        # jevmlx <subcommand> ... -> jevmlx.cli.build_parser()
        if bin0 == "jevmlx":
            from jevmlx.cli import build_parser

            build_parser().parse_args(rest)
            return
        # python -m benchmarks.<module> ... -> that module's build_parser().
        # Every benchmarks.<x> module m5 shells out to exposes build_parser()
        # so the parse-only guard validates the REAL parser, not a duplicate.
        if submod.startswith("benchmarks."):
            import importlib

            mod = importlib.import_module(submod)
            assert hasattr(mod, "build_parser"), (
                f"{submod} has no build_parser() — expose it so the planned "
                "argv is validated against the real parser, not a duplicate"
            )
            mod.build_parser().parse_args(rest[2:])  # drop '-m benchmarks.<mod>'
            return
        # Not our CLI (git, pytest, uv): skip.

    def test_every_planned_step_argv_parses(self, tmp_path):
        """Every Step.argv (and pre_argv / extra_argv) that targets our CLI
        or a benchmarks.* module parses against the real argparse."""
        out = tmp_path / "run"
        out.mkdir()  # plan_steps writes models-rest.txt into <out>
        steps = plan_steps(
            out,
            parity_models=[QUALITY_TARGET, REST_MODEL],
            ab_branch=None,
            probe=True,
        )
        assert steps, "plan_steps returned no steps"
        parsed_any = False
        for step in steps:
            for argv in (step.argv, step.pre_argv, *(step.extra_argv or ())):
                if not argv:
                    continue
                bin0 = argv[0].split("/")[-1]
                submod = argv[2] if len(argv) > 2 and argv[1] == "-m" else ""
                is_ours = bin0 == "jevmlx" or submod.startswith("benchmarks.")
                if not is_ours:
                    continue
                parsed_any = True
                self._parse_planned_argv(argv)  # raises if it doesn't parse
        assert parsed_any, "no jevmlx/benchmarks.* argv was checked"

    def test_bench_models_file_alone_parses(self):
        """B1 fix: ``jevmlx bench --models-file <f>`` no longer requires --model."""
        from jevmlx.cli import build_parser

        ns = build_parser().parse_args(["bench", "--models-file", "/tmp/m.txt", "--out", "/tmp/o"])
        assert ns.models_file == "/tmp/m.txt"
        assert ns.model is None

    def test_bench_model_alone_parses(self):
        from jevmlx.cli import build_parser

        ns = build_parser().parse_args(["bench", "--model", "quality", "--out", "/tmp/o"])
        assert ns.model == "quality"
        assert ns.models_file is None

    def test_bench_model_and_models_file_mutually_exclusive(self):
        from jevmlx.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["bench", "--model", "quality", "--models-file", "/tmp/m.txt", "--out", "/tmp/o"]
            )

    def test_no_alias_token_as_model_value_in_planned_argv(self, tmp_path):
        """B4: every planned argv passes the RESOLVED Hub id (not an alias)
        as the --model value. An A/B branch may predate the alias resolver,
        so a cross-branch argv carrying 'quality' asks the Hub for a repo
        named 'quality' (401). Aliases are for humans at the CLI only."""
        from jevmlx.models import MODEL_ALIASES

        out = tmp_path / "run"
        out.mkdir()  # plan_steps writes models-rest.txt into <out>
        steps = plan_steps(
            out,
            parity_models=[QUALITY_TARGET, REST_MODEL],
            ab_branch="w2a-field-local",  # exercise the A/B argv too
            probe=True,
        )
        alias_tokens = set(MODEL_ALIASES)
        checked = 0
        for step in steps:
            for argv in (step.argv, step.pre_argv, *(step.extra_argv or ())):
                if not argv or "--model" not in argv:
                    continue
                idx = argv.index("--model")
                model_val = argv[idx + 1]
                checked += 1
                assert model_val not in alias_tokens, (
                    f"step {step.id}: --model value {model_val!r} is an alias, "
                    "not a resolved Hub id; cross-branch argv must carry the "
                    "concrete id"
                )
        assert checked, "no --model argv was checked"

    def test_bench_requires_one_of_model_or_models_file(self):
        from jevmlx.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["bench", "--out", "/tmp/o"])


class TestM5MainEndToEnd:
    """benchmarks.m5.main() end to end: subprocess.run monkeypatched to a
    fake runner that produces each step's REAL outputs (via the builders
    above), so execute_step/runbook_append/step_done run for real."""

    @staticmethod
    def _fake_run_factory(out: Path, calls: list[tuple[str, int]]):
        """Map a planned step's argv to its real artifact builder; record
        (step-id, rc) calls. Returns the subprocess.run replacement.

        The runner RECEIVES the real argv m5 planned and keys on it, so a
        planner change that renames a step fails here visibly."""

        def fake_run(argv, **_kwargs):
            argv = tuple(argv)
            joined = " ".join(argv)
            # Match by (basename(argv[0]), subcommand) — never by absolute path,
            # which differs per runner (local /Users/ben vs CI /Users/runner).
            bin0 = argv[0].split("/")[-1]
            submod = argv[2] if len(argv) > 2 and argv[1] == "-m" else ""
            is_doctor = "doctor" in joined and "--json" in argv
            if is_doctor:
                # doctor --json: real exit-0 stdout JSON (the shape the gate
                # step captures into doctor.json).
                calls.append(("doctor", 0))
                stdout = '{"exit_code": 0, "checks": []}'
                return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
            # The parity step is `pytest -m slow`; the invariance/timing/probe/
            # fetch steps are `python -m benchmarks.<x>` (submod set). A plain
            # `python -m <not benchmarks.*>` is treated as pytest too.
            is_pytest = "pytest" in bin0
            is_plain_dash_m = (
                len(argv) > 1 and argv[1] == "-m" and not submod.startswith("benchmarks.")
            )
            if is_pytest or is_plain_dash_m:
                calls.append(("parity", 0))
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if submod == "benchmarks.typesafe.fetch":
                # The invariance step's pre-step: fetch typesafe cases. On a
                # fresh checkout (CI) the cached typesafe.jsonl is absent, so
                # this pre-step runs. Write a minimal dataset at --out so
                # pre_target is satisfied and the real invariance step gets
                # its data; return exit 0 (the fake run never hits the network).
                fetch_out = Path(argv[argv.index("--out") + 1])
                fetch_out.parent.mkdir(parents=True, exist_ok=True)
                if not fetch_out.exists():
                    _write_jsonl(fetch_out, _typesafe_cases(2))
                calls.append(("typesafe-fetch", 0))
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if submod == "benchmarks.invariance" or "benchmarks.invariance" in joined:
                inv_out = Path(argv[argv.index("--out") + 1])
                target = inv_out.parent if inv_out.name == "invariance" else inv_out
                _copy_template_piece("invariance", target)
                calls.append(("invariance", 0))
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if submod == "benchmarks.timing" or "benchmarks.timing" in joined:
                timing_out = Path(argv[argv.index("--out") + 1])
                _copy_template_piece("timing", timing_out)
                calls.append(("timing", 0))
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if submod == "benchmarks.probe" or "benchmarks.probe" in joined:
                probe_out = Path(argv[argv.index("--out") + 1])
                if "--command" in argv and argv[argv.index("--command") + 1] == "slope":
                    command = "slope"
                else:
                    command = "adapters"
                _build_probe_json(probe_out.parent, command)
                calls.append((f"probe-{command}", 0))
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            # bench-rest: jevmlx bench --models-file (basename jevmlx + bench subcommand).
            is_bench_rest = (
                bin0 == "jevmlx"
                and "--models-file" in argv
                and any(a == "bench" for a in argv[1:3])
            )
            if is_bench_rest:
                rest_out = Path(argv[argv.index("--out") + 1])
                rest_file = Path(argv[argv.index("--models-file") + 1])
                _build_rest_model_results(rest_out=rest_out, models_file=rest_file)
                calls.append(("bench-rest", 0))
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            # bench-quality: jevmlx bench --model ... --out ...
            is_bench_quality = (
                bin0 == "jevmlx"
                and any(a == "bench" for a in argv[1:3])
                and "--model" in argv
                and "--out" in argv
            )
            if is_bench_quality:
                combo_out = Path(argv[argv.index("--out") + 1])
                combo_out.parent.mkdir(parents=True, exist_ok=True)
                # The builders write the shared dataset jsonls next to the
                # bench out dir; make sure they exist for this fake run too.
                for name in ("bundled.jsonl", "typesafe.jsonl"):
                    src = out / name
                    if src.is_file() and not (combo_out.parent / name).exists():
                        (combo_out.parent / name).write_text(src.read_text(), encoding="utf-8")
                _build_bench_results_into(combo_out)
                calls.append(("bench", 0))
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            calls.append(("unknown:" + joined[:60], 1))
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected argv")

        return fake_run

    def test_full_runbook_every_step_zero_and_summaries_written(self, tmp_path, monkeypatch):
        out = tmp_path / "run1"
        calls: list[tuple[str, int]] = []
        monkeypatch.setattr(
            m5, "subprocess", self._fake_subprocess(self._fake_run_factory(out, calls))
        )
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)

        rc = m5.main(
            [
                "--out",
                str(out),
                "--parity-models",
                f"{QUALITY_TARGET},{REST_MODEL}",
                "--allow-sleep",
            ]
        )
        assert rc == 0, f"runbook failed: {calls}"

        # Every planned step got a RUNBOOK section with exit 0 and a wall.
        steps = plan_steps(out, parity_models=[QUALITY_TARGET, REST_MODEL])
        runbook = (out / "RUNBOOK.md").read_text()
        for step in steps:
            title = f". {step.title}"
            assert title in runbook, f"missing RUNBOOK section for {step.id}"
        assert "exit 0" in runbook
        assert "ABORT" not in runbook
        # The planner's expected step ids all ran (doctor, 2 parity, bench,
        # invariance, timing, bench-rest, summary).
        ids = [s.id for s in steps]
        assert ids == [
            "doctor",
            "parity-mlx-community--qwen2.5-7b-instruct-4bit",
            "parity-mlx-community--llama-3.1-8b-instruct-4bit",
            "bench-quality",
            "invariance",
            "timing",
            "bench-rest",
            "summary",
        ]

        # SUMMARY.md: main table with one row per combo, timing block,
        # invariance rollup.
        summary = (out / "SUMMARY.md").read_text()
        assert "# M5 runbook summary" in summary
        assert "## Main — bench combos" in summary
        for combo in (
            "parallel-slots-bundled",
            "parallel-labels-typesafe",
            "naive_local-slots-bundled",
            "parallel-slots-perturbed",
        ):
            assert combo in summary, f"combo row missing: {combo}"
        assert "## Main — timing report (quality, decide() presets)" in summary
        assert "mini_fraud" in summary
        assert "## Main — invariance" in summary
        assert "evidence_strength" in summary  # per-field rollup rows
        # Without --ab-branch the executor still passes collect_side(out/ab) —
        # an empty side dict — so the comparison table renders with '-' A/B
        # columns (real behavior, asserted here rather than the unit-test
        # branch where ab=None).
        assert "## A/B comparison — main vs A/B (bench quality + invariance)" in summary
        # All 5 A/B cells are '-' (agreement, flip rate, drift, time/case, peak);
        # invariance per-field rows also render '-/-' (empty ab side has no data).
        ab_rows = [
            line
            for line in summary.splitlines()
            if line.endswith("| - | - |") and line.startswith("|")
        ]
        assert len(ab_rows) == 7, ab_rows  # 2 invariance fields + 5 A/B metrics

        # The real bench writer (jevmlx.bench.run_bench) ends with
        # summarize(out) — SUMMARY.md at the bench out root.
        bench_summary = out / "bench-quality" / "SUMMARY.md"
        assert bench_summary.exists()
        assert "# Bench summary" in bench_summary.read_text()

    def test_probe_steps_added_and_produce_json(self, tmp_path, monkeypatch):
        out = tmp_path / "run-probe"
        calls: list[tuple[str, int]] = []
        monkeypatch.setattr(
            m5, "subprocess", self._fake_subprocess(self._fake_run_factory(out, calls))
        )
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)

        rc = m5.main(
            ["--out", str(out), "--parity-models", QUALITY_TARGET, "--probe", "--allow-sleep"]
        )
        assert rc == 0
        steps = plan_steps(out, parity_models=[QUALITY_TARGET], probe=True)
        assert "probe-slope" in [s.id for s in steps]
        assert "probe-adapters" in [s.id for s in steps]
        assert json.loads((out / "probe-slope.json").read_text())["slope"]
        adapters = json.loads((out / "probe-adapters.json").read_text())["adapters"]
        assert adapters["max_abs_diff"] == 0.0

    def test_second_run_skips_completed_steps(self, tmp_path, monkeypatch):
        out = tmp_path / "run-resume"
        calls: list[tuple[str, int]] = []
        fake_run = self._fake_run_factory(out, calls)
        monkeypatch.setattr(m5, "subprocess", self._fake_subprocess(fake_run))
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
        first = len(calls)
        assert first > 0

        calls.clear()
        m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
        # B9: the summary step is NEVER skipped (always regenerated); every
        # OTHER step is skipped because its outputs exist.
        assert calls == []
        runbook = (out / "RUNBOOK.md").read_text()
        steps = plan_steps(out, parity_models=[QUALITY_TARGET])
        non_summary_steps = [s for s in steps if not s.in_process]
        assert runbook.count("skip (outputs exist)") == len(non_summary_steps)

    def test_fresh_reruns_despite_markers(self, tmp_path, monkeypatch):
        out = tmp_path / "run-fresh"
        calls: list[tuple[str, int]] = []
        fake_run = self._fake_run_factory(out, calls)
        monkeypatch.setattr(m5, "subprocess", self._fake_subprocess(fake_run))
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
        first = len(calls)
        calls.clear()
        rc = m5.main(
            ["--out", str(out), "--parity-models", QUALITY_TARGET, "--fresh", "--allow-sleep"]
        )
        assert rc == 0
        assert len(calls) == first  # every step ran again

    def test_failing_gate_step_aborts_remaining(self, tmp_path, monkeypatch):
        out = tmp_path / "run-abort"
        calls: list[tuple[str, int]] = []
        fake_run = self._fake_run_factory(out, calls)

        def failing_run(argv, **kwargs):
            result = fake_run(argv, **kwargs)
            if "doctor" in " ".join(argv):
                return subprocess.CompletedProcess(tuple(argv), 1, stdout="doctor FAIL", stderr="")
            return result

        monkeypatch.setattr(m5, "subprocess", self._fake_subprocess(failing_run))
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        rc = m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
        assert rc == 1
        # doctor is the gate: its failure aborts everything after it.
        step_ids = [s.id for s in plan_steps(out, parity_models=[QUALITY_TARGET])]
        assert step_ids[0] == "doctor"
        runbook = (out / "RUNBOOK.md").read_text()
        assert "ABORT: doctor failed (exit 1); remaining steps skipped." in runbook
        summary = out / "SUMMARY.md"
        assert not summary.exists()  # the runbook never reached the summary step

    def test_failing_non_gate_step_recorded_and_run_continues(self, tmp_path, monkeypatch):
        out = tmp_path / "run-nonflat"
        calls: list[tuple[str, int]] = []
        fake_run = self._fake_run_factory(out, calls)

        def failing_timing(argv, **kwargs):
            if "benchmarks.timing" in " ".join(argv):
                # The timing step STREAMS stdout into the log (only the capture
                # steps route stderr separately), so the failure marker goes
                # out via stdout.
                return subprocess.CompletedProcess(tuple(argv), 1, stdout="boom timing", stderr="")
            return fake_run(argv, **kwargs)

        monkeypatch.setattr(m5, "subprocess", self._fake_subprocess(failing_timing))
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        rc = m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
        assert rc == 1  # the failure is recorded...
        runbook = (out / "RUNBOOK.md").read_text()
        # The runbook header carries the step TITLE, not the id: the failing
        # step's section shows exit 1 and the rest still ran.
        assert "timing on quality (5 reps) — exit 1" in runbook
        # ...the summary step still ran (no gate) and the timing columns
        # render as '-' (None-safe). The fake runner streams stdout into the
        # log, but the m5 executor only writes the argv line for streamed
        # steps (stdout goes to the log FILE HANDLE, not through the fake's
        # return value) — assert the executor's contract instead: the step
        # ran, failed, and its rc reached the runbook.
        assert (out / "SUMMARY.md").exists()
        summary = (out / "SUMMARY.md").read_text()
        assert "## Main — bench combos" in summary

    def test_caffeinate_guard_env_var_honoured(self, tmp_path, monkeypatch):
        """JEVMLX_M5_CAFFEINATED=1: the re-exec child must NOT re-exec."""
        import benchmarks.m5 as m5mod

        out = tmp_path / "run-caff"
        calls: list[tuple[str, int]] = []
        monkeypatch.setattr(
            m5mod,
            "subprocess",
            self._fake_subprocess(self._fake_run_factory(out, calls)),
        )
        monkeypatch.setenv("JEVMLX_M5_CAFFEINATED", "1")
        exec_calls: list = []

        monkeypatch.setattr(m5mod.os, "execvp", lambda *a: exec_calls.append(1))
        monkeypatch.setattr(m5mod.sys, "platform", "darwin")
        rc = m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET])  # no --allow-sleep
        assert rc == 0
        assert exec_calls == []  # the guard env var suppressed the re-exec
        runbook = (out / "RUNBOOK.md").read_text()
        assert "sleep_blocked: True" in runbook

    def test_caffeinate_reexec_when_no_guard(self, tmp_path, monkeypatch):
        """No guard env var + darwin + no --allow-sleep -> re-exec under
        caffeinate with argv_tail passed through. The fake execvp stops
        main() right after the re-exec attempt."""
        import benchmarks.m5 as m5mod

        exec_calls: list = []
        monkeypatch.setattr(m5mod.sys, "platform", "darwin")
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        out = tmp_path / "run-reexec"

        def exec_capture(prog, argv):
            exec_calls.append((prog, argv))
            raise SystemExit(0)  # stop main() right after the re-exec attempt

        monkeypatch.setattr(m5mod.os, "execvp", exec_capture)
        # _maybe_caffeinate replays sys.argv[1:] (NOT the parsed argv), so
        # patch sys.argv to the tail main() would have been given.
        monkeypatch.setattr(
            m5mod.sys,
            "argv",
            ["benchmarks.m5", "--out", str(out), "--parity-models", QUALITY_TARGET],
        )
        with pytest.raises(SystemExit):
            m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET])
        assert exec_calls, "main() should have re-execed under caffeinate"
        prog, argv = exec_calls[0]
        assert prog == "caffeinate"
        assert argv[:5] == ["caffeinate", "-dimsu", m5mod.sys.executable, "-m", "benchmarks.m5"]
        assert "--out" in argv[5:] and str(out) in argv[5:]
        assert "--parity-models" in argv[5:] and QUALITY_TARGET in argv[5:]

    def test_caffeinate_missing_binary_pops_guard(self, tmp_path, monkeypatch):
        """caffeinate not found: the guard env var is popped (the re-exec
        never happened), a warning prints, and main() continues."""
        import benchmarks.m5 as m5mod

        exec_calls: list = []
        monkeypatch.setattr(m5mod.sys, "platform", "darwin")
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        out = tmp_path / "run-nocaff"
        calls: list[tuple[str, int]] = []

        def exec_missing(prog, argv):
            exec_calls.append((prog, argv))
            raise FileNotFoundError(2, "No such file or directory", "caffeinate")

        monkeypatch.setattr(m5mod.os, "execvp", exec_missing)
        monkeypatch.setattr(
            m5mod,
            "subprocess",
            self._fake_subprocess(self._fake_run_factory(out, calls)),
        )
        rc = m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET])
        assert rc == 0
        assert exec_calls  # the re-exec was attempted
        import os as real_os

        assert real_os.environ.get("JEVMLX_M5_CAFFEINATED") != "1"  # popped
        runbook = (out / "RUNBOOK.md").read_text()
        assert "sleep_blocked: False" in runbook

    def test_b7_bench_reruns_when_combo_failed(self, tmp_path, monkeypatch):
        """B7: a bench step whose combos include a run_failed/load_failed
        combo is NOT done on rerun — it reruns so the failed combos get
        retried (bench itself skips the clean combos). The .done marker
        alone is insufficient because bench exits 0 when at least one
        combo succeeded."""
        import json as _json

        out = tmp_path / "run-b7"
        calls: list[tuple[str, int]] = []
        fake_run = self._fake_run_factory(out, calls)
        monkeypatch.setattr(m5, "subprocess", self._fake_subprocess(fake_run))
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])

        # Sabotage one combo's run.json to record a run_failed status.
        model_dir = out / "bench-quality" / "m2pro-32gb-mlx-community--qwen2.5-7b-instruct-4bit"
        failed_combo = next(model_dir.iterdir())
        run_json = failed_combo / "run.json"
        run_json.write_text(
            _json.dumps({"status": "run_failed", "error": {"type": "Test", "message": "sim"}}),
            encoding="utf-8",
        )

        calls.clear()
        m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
        # The bench-quality step RERAN (not skipped) because a combo failed.
        bench_calls = [c for c in calls if c[0] == "bench"]
        assert bench_calls, "bench-quality should have rerun (a combo failed)"
        runbook = (out / "RUNBOOK.md").read_text()
        assert "jevmlx bench --model quality" in runbook

    def test_b9_summary_always_regenerates(self, tmp_path, monkeypatch):
        """B9: SUMMARY.md is a pure function of the JSON artifacts — it is
        ALWAYS regenerated, never marker-skipped. A second run rewrites it
        even when every other step is skipped."""
        out = tmp_path / "run-b9"
        calls: list[tuple[str, int]] = []
        fake_run = self._fake_run_factory(out, calls)
        monkeypatch.setattr(m5, "subprocess", self._fake_subprocess(fake_run))
        monkeypatch.delenv("JEVMLX_M5_CAFFEINATED", raising=False)
        m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
        summary1_text = (out / "SUMMARY.md").read_text()
        summary1_mtime = (out / "SUMMARY.md").stat().st_mtime_ns

        calls.clear()
        m5.main(["--out", str(out), "--parity-models", QUALITY_TARGET, "--allow-sleep"])
        # No subprocess steps ran (all skipped), but SUMMARY was regenerated.
        assert calls == []
        summary2_text = (out / "SUMMARY.md").read_text()
        summary2_mtime = (out / "SUMMARY.md").stat().st_mtime_ns
        assert summary1_text == summary2_text  # same content (same artifacts)
        assert summary2_mtime > summary1_mtime  # but rewritten (regenerated)
        runbook = (out / "RUNBOOK.md").read_text()
        # The summary step section shows it RAN (exit 0), not 'skip'.
        assert "SUMMARY.md (main vs A/B) — exit 0" in runbook

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _fake_subprocess(fake_run):
        """A module-shaped stand-in for benchmarks.m5's subprocess import:
        run() is replaced; the constants and CompletedProcess stay real."""

        class _SubprocessModule:
            CompletedProcess = subprocess.CompletedProcess
            STDOUT = subprocess.STDOUT
            PIPE = subprocess.PIPE

        module = _SubprocessModule()
        module.run = fake_run
        return module


# --------------------------------------------- shared builders for the runner


def _build_bench_results_into(out: Path) -> None:
    """The bench step's --out target: the full results tree + SUMMARY.md,
    byte-identical to what jevmlx.bench.run_bench leaves behind — copied
    from the session template (see _copy_template_piece) instead of rebuilt
    (~7.5 s real-writer build -> ~0.03 s copy)."""
    _copy_template_piece("bench-quality", out)


def _build_invariance_dir(out: Path, engine) -> None:
    """Invariance artifacts into <out>/invariance — template copy (see
    _copy_template_piece)."""
    _copy_template_piece("invariance", out)


def _build_rest_model_results(*, rest_out: Path, models_file: Path) -> None:
    """The bench-rest step's tree: the same model folder shape, one combo,
    for each model in models-rest.txt (real writers)."""
    rest_out.mkdir(parents=True, exist_ok=True)
    engine = make_engine(YNLogitModel(), _Mod97Tokenizer(), model_id="fake/stable")
    cases = _bundled_cases(2)
    dataset = rest_out / "bundled.jsonl"
    _write_jsonl(dataset, cases)
    for model in models_file.read_text().split():
        if not model.strip():
            continue
        _run_combo(
            rest_out,
            "parallel-slots-bundled",
            cases,
            engine,
            scorer="slots",
            track="parallel",
            dataset_name="rest-bundled",
        )
