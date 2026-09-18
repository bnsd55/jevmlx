"""jevmlx command-line interface.

jevmlx decide --preset fintech_fraud
jevmlx decide --schema FILE --context FILE|-
jevmlx decide --json --preset support_triage
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from importlib import resources

from jevmlx import __version__
from jevmlx.api import _SUPPORTED, NONE_OF_ABOVE, NONE_OF_ABOVE_DESCRIPTION
from jevmlx.models import DEFAULT_MODEL
from jevmlx.engine import load_engine, run_parallel_generation
from jevmlx.lint import lint_schema
from jevmlx.log import configure
from jevmlx.schema import StructuredSchema


def load_preset(name: str) -> dict:
    """Load a preset: a filesystem path if it exists, else a bundled preset
    by name ('fintech_fraud' or 'fintech_fraud.json')."""
    if os.path.isfile(name):
        with open(name, encoding="utf-8") as f:
            return json.load(f)
    filename = name if name.endswith(".json") else f"{name}.json"
    return json.loads(
        resources.files("jevmlx.presets").joinpath(filename).read_text(encoding="utf-8")
    )


def _load_constraints(path: str) -> list[dict]:
    """Load and validate a constraints JSON file (EV1 shape)."""
    from jevmlx.constraints import validate_constraints

    with open(path, encoding="utf-8") as f:
        constraints = json.load(f)
    return validate_constraints(constraints)


def _fmt_confidence(value: float, decimals: int) -> str:
    """Presentation rounding for confidences: 3 decimals in the table, 4 in --json.
    The engine returns full-precision floats; rounding happens only here."""
    return f"{value:.{decimals}f}"


def print_result(preset_title: str, model_id: str, result: dict) -> None:
    """The demo table: preset, latency split, per-field values + confidences."""
    print()
    print(f"Preset : {preset_title}")
    print(f"Model  : {model_id}")
    print(
        f"Latency: {result['elapsed_ms']:.1f} ms "
        f"(prefill {result['prefill_ms']:.1f} + batched pass {result['suffix_eval_ms']:.1f})"
    )
    print()

    rows = []
    for name, entry in result["field_telemetry"].items():
        prob = entry["probability"]
        # Multi fields claim no field-level probability; the table shows the
        # closest option's decision margin to the threshold instead.
        conf = _fmt_confidence(prob, 3) if prob is not None else f"m={entry['margin']:.3f}"
        rows.append((name, str(entry["value"]), conf, entry["type"]))
    width = max(len(r[0]) for r in rows) if rows else 10

    print(f"{'field':<{width}}  {'value':<22}  conf   type")
    print(f"{'-' * width}  {'-' * 22}  -----  -----")
    for name, value, conf, ftype in rows:
        print(f"{name:<{width}}  {value:<22}  {conf}  {ftype}")

    print()
    print("Assembled JSON (never generated token-by-token, so always well-formed):")
    print(json.dumps(result["parsed_json"], indent=2, default=str))


def _rounded_json_payload(result: dict) -> dict:
    """--json output: engine values with confidences rounded to 4 decimals.

    Multi fields carry prob=None (no field-level probability is claimed);
    round() would fail on None, so they are left as-is.
    """
    parsed = copy.deepcopy(result["parsed_json"])
    for field in parsed.values():
        if isinstance(field, dict) and isinstance(field.get("prob"), float):
            field["prob"] = round(field["prob"], 4)
    return parsed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="jevmlx", description=__doc__)
    ap.add_argument("--version", action="version", version=f"jevmlx {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    decide = sub.add_parser(
        "decide", help="Run parallel constrained decisions on a preset or schema/context"
    )
    decide.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or alias (fast, quality, test)",
    )
    decide.add_argument(
        "--preset",
        help="preset name (bundled: fintech_fraud, support_triage, ...) or path to a .json file",
    )
    decide.add_argument("--schema", help="path to a schema .json file")
    decide.add_argument("--context", help="path to a context .txt file, or - for stdin")
    decide.add_argument(
        "--json", action="store_true", dest="as_json", help="print the assembled JSON only"
    )
    decide.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="softmax temperature for choice probabilities (1.0 = raw)",
    )
    decide.add_argument(
        "--scoring",
        choices=["slots", "labels"],
        default="slots",
        help="scoring mode: slots (neutral aliases in the prompt, quoted "
        "alias candidates; default) or labels (real choice text through the "
        "token trie)",
    )
    decide.add_argument(
        "--calibration",
        default=None,
        help='JSON file with the fitted multi calibrator ({"multi": {"a": .., '
        '"b": ..}} as written by jevmlx calibrate --out); selection = '
        "calibrated log-odds > 0. Without it an option is selected at "
        "P(yes) >= 0.5",
    )
    decide.add_argument(
        "--backend",
        choices=["native", "openai"],
        default="native",
        help="decision executor: native (local MLX engine, default) or "
        "openai (any OpenAI-compatible chat endpoint with logprobs; slower, "
        "degraded — one request per field, top_logprobs only)",
    )
    decide.add_argument(
        "--base-url",
        help="chat-completions base URL (openai backend)",
    )
    decide.add_argument(
        "--api-model",
        help="model name sent to the API (openai backend)",
    )
    decide.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="env var holding the API key (openai backend)",
    )
    decide.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request timeout in seconds (openai backend)",
    )
    decide.add_argument(
        "--prior-correction",
        action="store_true",
        help="subtract the neutral-context prior (one batched pass with "
        "'(no context provided)') from the per-choice log scores before "
        "selecting; the neutral pass is cached per schema",
    )
    decide.add_argument(
        "--constraints",
        default=None,
        help="path to a JSON file with case-level constraints (implies / "
        "excludes / requires_parent / exclusivity); wires through to the "
        "constrained MAP solver",
    )

    calib = sub.add_parser(
        "calibrate", help="Fit a temperature on labeled JSONL cases and report ECE"
    )
    calib.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or alias (fast, quality, test)",
    )
    calib.add_argument(
        "--data", required=True, help="JSONL file: {schema, context, labels} per line"
    )
    calib.add_argument("--bins", type=int, default=10, help="ECE bin count")
    calib.add_argument(
        "--out",
        default=None,
        help='write the fitted calibrators to this JSON file ({"temperature": .., '
        '"multi": {"a": .., "b": ..}}); pass it to jevmlx decide --calibration',
    )

    serve_p = sub.add_parser("serve", help="Serve decisions over HTTP (one Metal GPU, serial)")
    serve_p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or alias (fast, quality, test)",
    )
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true", help="info-level logs on stderr")

    validate_p = sub.add_parser(
        "validate", help="Lint a schema for engine-visible problems (no model download)"
    )
    validate_p.add_argument("schema", help="path to a schema .json file")
    validate_p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id whose tokenizer decides choice token boundaries",
    )
    validate_p.add_argument(
        "--json", action="store_true", dest="as_json", help="print findings as JSON"
    )

    eval_p = sub.add_parser(
        "eval",
        help="Run labeled cases through a decision track; writes predictions + run manifest",
    )
    eval_p.add_argument("--data", required=True, help="cases JSONL (see the eval contract)")
    eval_p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or alias (fast, quality, test)",
    )
    eval_p.add_argument(
        "--track",
        required=True,
        choices=["parallel", "naive_local", "api_baseline", "openai_slots"],
        help="decision track to run",
    )
    eval_p.add_argument("--api-base", help="chat-completions base URL (api_baseline track)")
    eval_p.add_argument("--api-model", help="model name sent to the API (api_baseline track)")
    eval_p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="env var holding the API key (api_baseline and openai_slots tracks)",
    )
    eval_p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request timeout in seconds (openai_slots track)",
    )
    eval_p.add_argument(
        "--scoring",
        choices=["slots", "labels"],
        default="slots",
        help="parallel-track scoring mode (slots = neutral aliases, the "
        "default; labels = real choice text through the token trie)",
    )
    eval_p.add_argument(
        "--prior-correction",
        action="store_true",
        help="parallel track: subtract the neutral-context prior from the "
        "per-choice log scores; the neutral pass runs once per schema",
    )
    eval_p.add_argument(
        "--permutations",
        default="none",
        choices=["none", "rotations", "fieldperm", "all"],
        help="order-sensitivity probe (parallel track only)",
    )
    eval_p.add_argument("--limit", type=int, default=None, help="only the first N cases")
    eval_p.add_argument(
        "--split",
        default="all",
        choices=["train", "holdout", "all"],
        help="case split to run",
    )
    eval_p.add_argument(
        "--out", required=True, help="output directory (predictions.jsonl, run.json)"
    )
    bench_p = sub.add_parser(
        "bench",
        help="One command: complete PR-ready benchmark results folder.",
    )
    bench_p.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id(s) or alias (fast, quality, test) for mlx-lm; "
        "comma-separated list runs them sequentially with one SUMMARY.md across all",
    )
    bench_p.add_argument(
        "--models-file",
        default=None,
        help="file with one model id per line ('#' comments allowed); overrides --model",
    )
    bench_p.add_argument(
        "--datasets",
        default="bundled,typesafe,perturbed",
        help="comma list: bundled,typesafe,perturbed",
    )
    bench_p.add_argument("--scorers", default="slots,labels", help="comma list: slots,labels")
    bench_p.add_argument(
        "--tracks", default="parallel,naive_local", help="comma list: parallel,naive_local"
    )
    bench_p.add_argument("--out", default=None, help="results root (default benchmarks/results)")
    bench_p.add_argument("--runs", type=int, default=2, help="eval runs per combo (last kept)")
    bench_p.add_argument("--machine", default=None, help="override the machine tag")
    bench_p.add_argument(
        "--force",
        action="store_true",
        help="run despite battery power or busy Metal memory (reasons are printed)",
    )
    bench_p.add_argument(
        "--fresh",
        action="store_true",
        help="rerun combos that already have complete results (default: skip them)",
    )
    bench_p.add_argument(
        "--load-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for load_engine before recording a load_failed row (default 900)",
    )
    bench_p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the machine tag, dataset build plan, combo list with output "
        "folders, and a memory estimate per model, then exit without loading anything",
    )

    report_p = sub.add_parser(
        "report",
        help="Build a JSON + markdown eval report from predictions.jsonl (offline)",
    )
    report_p.add_argument("--predictions", required=True, help="path to predictions.jsonl")
    report_p.add_argument("--out", required=True, help="output path for the JSON report")

    doctor_p = sub.add_parser(
        "doctor",
        help="Environment checks: run before filing an issue or a bench run",
    )
    doctor_p.add_argument(
        "--model",
        default=None,
        help="model id to dry-run check (tokenizer load only, no weights)",
    )
    doctor_p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="print the checks as JSON instead of a table",
    )
    args = ap.parse_args(argv)

    # serve defaults to INFO: the user must see the listen address. -v is a no-op there.
    if args.command == "serve":
        level = logging.INFO
    else:
        level = logging.INFO if args.verbose else logging.WARNING
    configure(level=level, json_mode=os.environ.get("JEVMLX_LOG") == "json")

    if args.command == "decide":
        if args.preset and (args.schema or args.context):
            decide.error("--preset cannot be combined with --schema/--context")
        if not args.preset and not (args.schema and args.context):
            missing = [
                flag
                for flag, given in (("--schema", args.schema), ("--context", args.context))
                if not given
            ]
            decide.error(
                "exactly one of --preset or --schema AND --context is required; "
                f"missing: {', '.join(missing)}"
            )

        if args.preset:
            preset = load_preset(args.preset)
            schema_dict = preset["schema"]
            context = preset["context"]
            title = preset.get("title", args.preset)
        else:
            with open(args.schema, encoding="utf-8") as f:
                schema_dict = json.load(f)
            if args.context == "-":
                context = sys.stdin.read()
            else:
                with open(args.context, encoding="utf-8") as f:
                    context = f.read()
            title = args.schema

        if args.backend == "openai":
            from jevmlx.openai_slots import decide_openai

            if not (args.base_url and args.api_model):
                decide.error("--backend openai requires --base-url and --api-model")
            if args.prior_correction:
                decide.error("--prior-correction is native-backend only")
            schema = StructuredSchema(schema_dict)
            result = decide_openai(
                args.base_url,
                args.api_model,
                os.environ.get(args.api_key_env),
                schema,
                context,
                timeout=args.timeout,
                calibration=args.calibration,
            )
            model_label = args.api_model
        else:
            print(f"Loading {args.model} ...", flush=True)
            model, tokenizer = load_engine(args.model)

            schema = StructuredSchema(schema_dict)
            result = run_parallel_generation(
                model,
                tokenizer,
                context,
                schema,
                temperature=args.temperature,
                scoring=args.scoring,
                calibration=args.calibration,
                prior_correction=args.prior_correction,
                constraints=_load_constraints(args.constraints) if args.constraints else None,
            )
            model_label = args.model

        if args.as_json:
            print(json.dumps(_rounded_json_payload(result), indent=2))
        else:
            print_result(title, model_label, result)

    elif args.command == "calibrate":
        from jevmlx import calibrate

        print(f"Loading {args.model} ...", flush=True)
        model, tokenizer = load_engine(args.model)
        cases = calibrate.load_cases(args.data)
        print(f"collecting scores from {len(cases)} labeled cases ...", flush=True)
        samples = calibrate.collect(model, tokenizer, cases)

        calibrators: dict = {}
        if samples:
            t_fit = calibrate.fit_temperature(samples)
            ece_before = calibrate.ece(samples, 1.0, bins=args.bins)
            ece_after = calibrate.ece(samples, t_fit, bins=args.bins)
            acc = calibrate.accuracy(samples)
            calibrators["temperature"] = t_fit
            print(f"n samples      : {len(samples)}")
            print(f"fitted T       : {t_fit}")
            print(f"ECE before     : {ece_before:.4f}  (T=1.0)")
            print(f"ECE after      : {ece_after:.4f}  (T={t_fit})")
            print(f"accuracy       : {acc:.4f}")
        else:
            print("no scalar (enum/boolean) labels; skipping temperature fit")

        # Pooled multi logistic: one (a, b) across every multi option's raw
        # log-odds (W2-E step 2). Skipped when no multi labels exist.
        multi_samples = calibrate.collect_multi(model, tokenizer, cases)
        if multi_samples:
            a_fit, b_fit = calibrate.fit_logistic(multi_samples)
            calibrators["multi"] = {"a": a_fit, "b": b_fit}
            n_yes = sum(y for _, y in multi_samples)
            print(
                f"n multi pairs  : {len(multi_samples)}  ({n_yes} yes / "
                f"{len(multi_samples) - n_yes} no)"
            )
            print(f"fitted (a, b)  : ({a_fit}, {b_fit})")
        else:
            print("no multi labels; skipping pooled logistic fit")

        if args.out:
            if not calibrators:
                print("nothing fitted; no --out file written")
            else:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(calibrators, f, indent=2, sort_keys=True)
                    f.write("\n")
                print(f"calibrators -> {args.out}")

    elif args.command == "serve":
        from jevmlx.serve import serve

        serve(args.model, args.host, args.port)

    elif args.command == "bench":
        from jevmlx.bench import main as bench_main

        argv = [
            "--model",
            args.model,
            "--datasets",
            args.datasets,
            "--scorers",
            args.scorers,
            "--tracks",
            args.tracks,
            "--runs",
            str(args.runs),
        ]
        if args.out:
            argv += ["--out", args.out]
        if args.machine:
            argv += ["--machine", args.machine]
        if args.models_file:
            argv += ["--models-file", args.models_file]
        if args.force:
            argv.append("--force")
        if args.fresh:
            argv.append("--fresh")
        if args.load_timeout != 900.0:
            argv += ["--load-timeout", str(args.load_timeout)]
        if args.dry_run:
            argv.append("--dry-run")
        raise SystemExit(bench_main(argv))

    elif args.command == "report":
        # Offline: pure-python metrics over predictions.jsonl -> evalreport.
        from jevmlx.evalmetrics import compute_metrics, load_predictions
        from jevmlx.evalreport import environment, write_report

        records = load_predictions(args.predictions)
        write_report(args.out, {"environment": environment(), "metrics": compute_metrics(records)})
        print(f"wrote {args.out} (+ .md)")

    elif args.command == "doctor":
        from jevmlx.doctor import run_doctor

        raise SystemExit(run_doctor(model=args.model, as_json=args.as_json))

    elif args.command == "validate":
        from dataclasses import asdict

        # transformers is imported lazily: slow to import and only validate needs it.
        from transformers import AutoTokenizer

        with open(args.schema, encoding="utf-8") as f:
            schema = StructuredSchema(json.load(f))
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        findings = lint_schema(schema, tokenizer)

        if args.as_json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2))
        else:
            if not findings:
                print("OK: no findings.")
            for finding in findings:
                print(f"{finding.field}: [{finding.kind}] {finding.message}")
                if finding.suggestion:
                    print(f"  suggestion: {finding.suggestion}")
        # compile_error findings mean the schema cannot run at all: exit 1.
        # (collisions only slow the engine down; they stay exit 0 with a
        # warning printed above).
        if any(finding.kind == "compile_error" for finding in findings):
            sys.exit(1)

    elif args.command == "eval":
        _run_eval_command(args)


def _run_eval_command(args) -> None:
    """jevmlx eval: batch labeled cases through one decision track."""
    import logging

    from jevmlx import evalrun

    cases = evalrun.load_cases(args.data)
    if args.limit is not None:
        cases = cases[: args.limit]

    extra: dict = {
        "dataset_path": os.path.abspath(args.data),
        "scoring": args.scoring if args.track == "parallel" else "slots",
        "prior_correction": bool(args.prior_correction) if args.track == "parallel" else False,
    }
    lock = os.path.join(os.path.dirname(os.path.abspath(args.data)), "dataset.lock.json")

    if args.track == "parallel":
        print(f"Loading {args.model} ...", flush=True)
        model, tokenizer = load_engine(args.model)
        decide_fn = evalrun.parallel_decide_fn(
            model, tokenizer, scoring=args.scoring, prior_correction=args.prior_correction
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        plan_provider = lambda schema: schema.compile_labels_plan(tokenizer)  # noqa: E731
    elif args.track == "naive_local":
        print(f"Loading {args.model} ...", flush=True)
        model, tokenizer = load_engine(args.model)
        decide_fn = evalrun.naive_local_decide_fn(model, tokenizer)
        chat_template = getattr(tokenizer, "chat_template", None)
        plan_provider = None
    elif args.track == "openai_slots":
        if not (args.api_base and args.api_model):
            raise SystemExit("openai_slots track requires --api-base and --api-model")
        api_key = os.environ.get(args.api_key_env)
        decide_fn, api_params = evalrun.openai_slots_decide_fn(
            args.api_base, args.api_model, api_key, timeout=args.timeout
        )
        extra.update(api_params)
        chat_template = None
        plan_provider = None
    else:
        if not (args.api_base and args.api_model):
            eval_p_error = "api_baseline track requires --api-base and --api-model"
            raise SystemExit(eval_p_error)
        api_key = os.environ.get(args.api_key_env)
        decide_fn, api_params = evalrun.api_baseline_decide_fn(
            args.api_base, args.api_model, api_key
        )
        extra.update(api_params)
        chat_template = None
        plan_provider = None

    logging.getLogger("jevmlx.evalrun").setLevel(logging.INFO)
    run = evalrun.run_eval(
        cases,
        decide_fn,
        track=args.track,
        model=args.api_model if args.track in ("api_baseline", "openai_slots") else args.model,
        permutations=args.permutations,
        split=args.split,
        out_dir=args.out,
        extra_config=extra,
        chat_template=chat_template,
        plan_provider=plan_provider,
        dataset_lock_path=lock if os.path.exists(lock) else None,
        dataset_path=os.path.abspath(args.data),
    )
    print(
        f"run {run['run_id']}: {run['counts']['cases']} cases, "
        f"{run['counts']['prediction_lines']} prediction lines -> {args.out}/"
    )
