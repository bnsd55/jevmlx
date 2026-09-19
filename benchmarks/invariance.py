#!/usr/bin/env python3
"""Irrelevant-field invariance benchmark: does the global schema block hurt?

GPT-review Q1's diagnostic before the W2-A change (field-local prompt tails
over one shared context prefix). For each target field of the bundled and
TypeSafe cases, build schema variants with 1, 5, 20, 40 UNRELATED extra
fields appended (deterministic, from a fixed pool of plausible fields with
2-6 options), run the parallel track, and measure per target field:

- ``label_flip_rate``     share of (case, extra-count) variants where the
                          target field's decided label differs from the
                          1-extra-field baseline (the reference every
                          variant is compared against — not the case label,
                          so schema growth is isolated from model error).
- ``winner_logodds_drift`` mean absolute drift of the winner's log-odds
                          (``log p - log(1 - p)``) vs the baseline. A raw
                          log-probability drift is unbounded when the
                          baseline probability saturates at 1.0; log-odds
                          drift is the calibrated-comparable quantity.
- ``latency_ms``/``rows``/``passes`` per extra-count (engine telemetry).

Every variant runs the full track so ``run.json``/``predictions.jsonl``
stay compatible with the bench contracts (jevmlx.evalrun.run_eval); a
summary markdown table lands next to the results.

Usage:
    python -m benchmarks.invariance --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --data benchmarks/cases.jsonl --out results/invariance --extra 1,5,20,40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

__all__ = [
    "EXTRA_COUNTS",
    "FIELD_POOL",
    "build_variant_schema",
    "target_fields",
    "winner_logodds",
    "invariance_metrics",
    "write_summary",
    "main",
]

# Extra-count ladder: 1 is the baseline every other rung is compared against.
EXTRA_COUNTS = (1, 5, 20, 40)

# Fixed pool of plausible unrelated fields, deterministic order. Names are
# chosen not to collide with bundled/TypeSafe fields; every entry is a valid
# jevmlx schema field (boolean or 2-6 choice enum) with a one-line gloss.
# Size: the top rung needs 40 unrelated fields, so the pool holds 48.
FIELD_POOL: list[dict] = [
    {
        "name": "account_status",
        "type": "enum",
        "choices": ["TRIAL", "ACTIVE", "SUSPENDED", "CLOSED"],
        "description": "Current account status",
    },
    {
        "name": "billing_cycle",
        "type": "enum",
        "choices": ["MONTHLY", "QUARTERLY", "ANNUAL"],
        "description": "Billing cycle of the account",
    },
    {
        "name": "channel",
        "type": "enum",
        "choices": ["WEB", "MOBILE_APP", "PHONE", "IN_STORE"],
        "description": "Channel the interaction came through",
    },
    {
        "name": "compliance_review",
        "type": "boolean",
        "description": "Whether a compliance review was requested",
    },
    {
        "name": "customer_tenure",
        "type": "enum",
        "choices": ["NEW", "ESTABLISHED", "LONG_TERM"],
        "description": "How long the customer has been with the service",
    },
    {
        "name": "data_sensitivity",
        "type": "enum",
        "choices": ["PUBLIC", "INTERNAL", "CONFIDENTIAL"],
        "description": "Sensitivity of the data involved",
    },
    {
        "name": "delivery_status",
        "type": "enum",
        "choices": ["PENDING", "SHIPPED", "DELIVERED", "RETURNED"],
        "description": "Fulfilment status of the order",
    },
    {
        "name": "discount_applied",
        "type": "boolean",
        "description": "Whether a discount code was applied",
    },
    {"name": "escalation_flag", "type": "boolean", "description": "Whether the case was escalated"},
    {
        "name": "feedback_signed",
        "type": "boolean",
        "description": "Whether signed feedback was provided",
    },
    {
        "name": "invoice_type",
        "type": "enum",
        "choices": ["STANDARD", "CREDIT_NOTE", "PREPAID"],
        "description": "Type of invoice issued",
    },
    {
        "name": "language",
        "type": "enum",
        "choices": ["EN", "DE", "FR", "ES"],
        "description": "Language of the request",
    },
    {
        "name": "loyalty_tier",
        "type": "enum",
        "choices": ["BRONZE", "SILVER", "GOLD", "PLATINUM"],
        "description": "Loyalty programme tier",
    },
    {
        "name": "notification_pref",
        "type": "enum",
        "choices": ["EMAIL", "SMS", "PUSH", "NONE"],
        "description": "Preferred notification channel",
    },
    {
        "name": "payment_method",
        "type": "enum",
        "choices": ["CARD", "TRANSFER", "WALLET"],
        "description": "Payment method used",
    },
    {
        "name": "region",
        "type": "enum",
        "choices": ["AMER", "EMEA", "APAC"],
        "description": "Customer region",
    },
    {
        "name": "repeat_contact",
        "type": "boolean",
        "description": "Whether the customer contacted support before",
    },
    {
        "name": "reporting_consent",
        "type": "boolean",
        "description": "Whether usage reporting was consented to",
    },
    {
        "name": "reseller_flag",
        "type": "boolean",
        "description": "Whether the account is managed by a reseller",
    },
    {
        "name": "retry_count",
        "type": "enum",
        "choices": ["NONE", "ONE", "MANY"],
        "description": "How often the operation was retried",
    },
    {
        "name": "season",
        "type": "enum",
        "choices": ["Q1", "Q2", "Q3", "Q4"],
        "description": "Quarter of the interaction",
    },
    {
        "name": "sentiment",
        "type": "enum",
        "choices": ["POSITIVE", "NEUTRAL", "NEGATIVE"],
        "description": "Overall sentiment of the message",
    },
    {
        "name": "service_outage",
        "type": "boolean",
        "description": "Whether an outage affected the service",
    },
    {
        "name": "sla_tier",
        "type": "enum",
        "choices": ["STANDARD", "PRIORITY", "EXPRESS", "EMERGENCY"],
        "description": "Service-level tier of the request",
    },
    {
        "name": "subscription_active",
        "type": "boolean",
        "description": "Whether a subscription is currently active",
    },
    {
        "name": "support_plan",
        "type": "enum",
        "choices": ["FREE", "BASIC", "PRO", "ENTERPRISE"],
        "description": "Support plan of the account",
    },
    {
        "name": "survey_channel",
        "type": "enum",
        "choices": ["IN_APP", "EMAIL", "PHONE"],
        "description": "How the survey reached the customer",
    },
    {
        "name": "survey_completed",
        "type": "boolean",
        "description": "Whether the follow-up survey was completed",
    },
    {
        "name": "tax_region",
        "type": "enum",
        "choices": ["DOMESTIC", "EU", "EXPORT"],
        "description": "Tax region of the transaction",
    },
    {
        "name": "ticket_channel",
        "type": "enum",
        "choices": ["EMAIL", "CHAT", "FORUM"],
        "description": "Ticket intake channel",
    },
    {
        "name": "ticket_priority",
        "type": "enum",
        "choices": ["P4", "P3", "P2", "P1"],
        "description": "Ticket priority as filed",
    },
    {
        "name": "training_consent",
        "type": "boolean",
        "description": "Whether data may be used for model training",
    },
    {
        "name": "two_factor",
        "type": "boolean",
        "description": "Whether two-factor authentication is enabled",
    },
    {
        "name": "user_role",
        "type": "enum",
        "choices": ["ADMIN", "MEMBER", "VIEWER", "GUEST"],
        "description": "Role of the user in the workspace",
    },
    {
        "name": "verified_identity",
        "type": "boolean",
        "description": "Whether the identity was verified",
    },
    {
        "name": "volume_band",
        "type": "enum",
        "choices": ["LOW", "MEDIUM", "HIGH"],
        "description": "Usage volume band of the account",
    },
    {
        "name": "vpn_connection",
        "type": "boolean",
        "description": "Whether the session came through a VPN",
    },
    {
        "name": "weekend_flag",
        "type": "boolean",
        "description": "Whether the interaction happened on a weekend",
    },
    {
        "name": "workflow_state",
        "type": "enum",
        "choices": ["DRAFT", "REVIEW", "APPROVED", "ARCHIVED"],
        "description": "Workflow state of the record",
    },
    {
        "name": "billing_country",
        "type": "enum",
        "choices": ["US", "CA", "GB", "DE", "OTHER"],
        "description": "Billing country of the account",
    },
    {
        "name": "chargeback_history",
        "type": "enum",
        "choices": ["NONE", "ONE", "MULTIPLE"],
        "description": "Chargeback history of the account",
    },
    {
        "name": "data_retention",
        "type": "enum",
        "choices": ["30D", "90D", "1Y", "INDEFINITE"],
        "description": "Data retention window selected",
    },
    {
        "name": "device_count",
        "type": "enum",
        "choices": ["ONE", "FEW", "MANY"],
        "description": "Number of devices on the account",
    },
    {
        "name": "email_domain",
        "type": "enum",
        "choices": ["CONSUMER", "BUSINESS", "EDU", "OTHER"],
        "description": "Domain class of the email address",
    },
    {
        "name": "fraud_training",
        "type": "boolean",
        "description": "Whether the fraud model was retrained recently",
    },
    {
        "name": "marketing_optin",
        "type": "boolean",
        "description": "Whether marketing e-mails are opted in",
    },
    {
        "name": "onboarding_stage",
        "type": "enum",
        "choices": ["SIGNUP", "VERIFY", "FIRST_RUN", "DONE"],
        "description": "Onboarding stage of the user",
    },
    {
        "name": "partner_referral",
        "type": "boolean",
        "description": "Whether the account came from a partner referral",
    },
]

# Bump when the variant construction or metrics change.
BENCH_VERSION = "1"


def _pool_schema(rng: random.Random, count: int, exclude: set[str]) -> dict:
    """``count`` deterministic unrelated fields from FIELD_POOL.

    Drawn without replacement (the pool has 18 entries, enough for the
    40-extra rung); names colliding with target fields are skipped. The rng
    is seeded by the caller with (case id, extra count), so the same case
    always gets the same extras at a given rung.
    """
    schema: dict = {}
    candidates = [f for f in FIELD_POOL if f["name"] not in exclude]
    if len(candidates) < count:
        raise ValueError(f"FIELD_POOL has {len(candidates)} usable entries, need {count}")
    for f in rng.sample(candidates, count):
        spec = {"type": f["type"], "description": f["description"]}
        if f["type"] == "enum":
            spec["choices"] = list(f["choices"])
        schema[f["name"]] = spec
    return schema


def build_variant_schema(base_schema: dict, extra_count: int, seed_key: str) -> dict:
    """Base schema + ``extra_count`` unrelated fields appended, deterministically.

    Order matters to the engine (schema-block text), so extras are appended
    after the base fields in drawn order. ``seed_key`` (usually the case id)
    makes the draw reproducible per case.
    """
    digest = hashlib.sha256(f"{seed_key}#{extra_count}#{BENCH_VERSION}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    variant = {k: dict(v) for k, v in base_schema.items()}
    variant.update(_pool_schema(rng, extra_count, set(base_schema)))
    return variant


def target_fields(case: dict) -> list[str]:
    """Fields to probe for a case: the primary field when marked, else all.

    The bundled cases carry ``primary_field``; TypeSafe records do not, so
    every field of their schema is a target.
    """
    schema = case["schema"]
    primary = case.get("primary_field")
    if primary and primary in schema:
        return [primary]
    return list(schema)


def winner_logodds(entry: dict) -> float | None:
    """Winner's log-odds from a decide_fn field entry, None when unavailable.

    log-odds = log p - log(1 - p) over the field's choice distribution (from
    ``log_scores``; the parallel track's normalized T=1 log P). Saturating
    probabilities map to +-inf, which is informative (drift from saturation
    is exactly what alias interference does) but is excluded from means.
    """
    log_scores = entry.get("log_scores")
    if not isinstance(log_scores, dict) or not log_scores:
        return None
    p = entry.get("probability")
    if p is None:
        return None
    if not 0.0 < p < 1.0:
        return math.inf if p >= 1.0 else -math.inf
    return math.log(p) - math.log(1.0 - p)


def _finite_mean(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def invariance_metrics(
    runs: dict[int, list[dict]],
    targets: list[str],
) -> list[dict]:
    """Per-target-field metrics over the extra-count ladder.

    ``runs`` maps extra-count -> predictions.jsonl records (evalrun lines,
    one per (case, field)). The 1-extra run is the baseline. Returns one
    dict per (target field): flip rate, winner log-odds drift, and
    telemetry means per rung.
    """
    by_rung: dict[int, dict[tuple[str, str], dict]] = {}
    for extra, records in runs.items():
        index: dict[tuple[str, str], dict] = {}
        for rec in records:
            if rec.get("permutation") not in (None, "canonical"):
                continue
            # Variant ids carry a per-rung suffix ("<id>#x<extra>"); strip it
            # so the same case across rungs shares one key. group_id is the
            # original case id.
            case_id = rec.get("group_id") or rec.get("case_id").split("#")[0]
            index[(case_id, rec["field"])] = rec
        by_rung[extra] = index

    baseline = 1
    if baseline not in by_rung:
        raise ValueError("no baseline run (extra=1) present")
    if not by_rung[baseline]:
        raise ValueError(
            "no canonical baseline records (extra=1); every line was "
            "filtered out (non-canonical permutations?)"
        )
    base_index = by_rung[baseline]

    results: list[dict] = []
    for field in targets:
        base_keys = [k for k in base_index if k[1] == field]
        if not base_keys:
            continue
        row: dict = {"field": field, "cases": len(base_keys), "rungs": {}}
        flips_total = 0
        drifts: list[float] = []
        for extra in sorted(runs):
            idx = by_rung[extra]
            flips = 0
            compared = 0
            rung_drifts: list[float] = []
            latencies: list[float] = []
            rows_list: list[int] = []
            for key in base_keys:
                variant = idx.get(key)
                if variant is None:
                    continue
                compared += 1
                base_rec = base_index[key]
                if _canonical(variant.get("prediction")) != _canonical(base_rec.get("prediction")):
                    flips += 1
                base_lo = winner_logodds(base_rec)
                var_lo = winner_logodds(variant)
                if base_lo is not None and var_lo is not None:
                    rung_drifts.append(abs(var_lo - base_lo))
                if variant.get("latency_ms") is not None:
                    latencies.append(variant["latency_ms"])
                if variant.get("rows") is not None:
                    rows_list.append(variant["rows"])
            if compared:
                flip_rate = flips / compared
                flips_total += flips
                drifts.extend(rung_drifts)
            row["rungs"][extra] = {
                "compared": compared,
                "flips": flips,
                "flip_rate": flip_rate if compared else None,
                "winner_logodds_drift_mean": _finite_mean(rung_drifts),
                "latency_ms_mean": _finite_mean(latencies) if latencies else None,
                "rows_mean": _finite_mean([float(r) for r in rows_list]) if rows_list else None,
            }
        row["flip_rate_total"] = (
            flips_total / sum(row["rungs"][e]["compared"] for e in row["rungs"])
            if row["rungs"]
            else None
        )
        row["winner_logodds_drift_mean"] = _finite_mean(drifts)
        results.append(row)
    return results


def _canonical(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(sorted(str(v) for v in value))
    return str(value)


def write_summary(out_dir: Path, rows: list[dict], config: dict) -> Path:
    """Small markdown table next to the results (one row per target field)."""
    lines = [
        "# Irrelevant-field invariance",
        "",
        f"- model: `{config['model']}`",
        f"- extra-count ladder: {config['extra']}",
        f"- dataset: `{config['dataset_path']}`",
        "- baseline: extra=1 (label flip and log-odds drift measured against it)",
        "",
        "| field | cases | flip@5 | flip@20 | flip@40 | drift@5 | drift@20 "
        "| drift@40 | rows@1 | rows@40 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _pct(entry: dict, key: str) -> str:
        value = entry.get(key)
        return f"{value:.1%}" if isinstance(value, float) else "-"

    def _num(entry: dict, key: str) -> str:
        value = entry.get(key)
        return f"{value:.2f}" if isinstance(value, float) else "-"

    for row in rows:
        rungs = row["rungs"]
        lines.append(
            f"| {row['field']} | {row['cases']} "
            f"| {_pct(rungs.get(5, {}), 'flip_rate')} "
            f"| {_pct(rungs.get(20, {}), 'flip_rate')} "
            f"| {_pct(rungs.get(40, {}), 'flip_rate')} "
            f"| {_num(rungs.get(5, {}), 'winner_logodds_drift_mean')} "
            f"| {_num(rungs.get(20, {}), 'winner_logodds_drift_mean')} "
            f"| {_num(rungs.get(40, {}), 'winner_logodds_drift_mean')} "
            f"| {_num(rungs.get(1, {}), 'rows_mean')} "
            f"| {_num(rungs.get(40, {}), 'rows_mean')} |"
        )
    lines += [
        "",
        "flip@N = share of variants where the target label differs from the",
        "extra=1 baseline; drift = mean |winner log-odds drift|; rows = engine",
        "row count (schema growth drives rows/latency).",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


def _load_cases(path: str) -> list[dict]:
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        records: list[dict] = []
        for family in data.get("families", {}).values():
            for case in family["cases"]:
                records.append(
                    {
                        "id": f"quality-eval/{case['id']}",
                        "group_id": case["id"],
                        "source": "quality-eval",
                        "workflow": None,
                        "benchmark_only": False,
                        "schema": family["schema"],
                        "context": case["context"],
                        "labels": case["labels"],
                        "split": "train",
                        "meta": {},
                        "primary_field": case.get("primary_field"),
                    }
                )
        return records
    cases: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(json.loads(line))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.invariance",
        description="Irrelevant-field invariance benchmark (W2-A gate).",
    )
    parser.add_argument("--model", required=True, help="mlx-lm model id")
    parser.add_argument("--data", required=True, help="cases: .json (bundled) or .jsonl (eval)")
    parser.add_argument("--out", required=True, help="results directory (bench contract)")
    parser.add_argument("--extra", default="1,5,20,40", help="comma list of extra-field counts")
    parser.add_argument("--scoring", default="slots", choices=["slots", "labels"])
    parser.add_argument("--limit", type=int, default=None, help="cap cases (smoke runs)")
    args = parser.parse_args(argv)

    extra_counts = [int(x) for x in args.extra.split(",") if x.strip()]
    if not extra_counts or 1 not in extra_counts:
        parser.error("--extra must include the 1 baseline")
    if extra_counts != sorted(extra_counts):
        parser.error("--extra must be ascending")

    from jevmlx.engine import load_engine
    from jevmlx.evalrun import parallel_decide_fn, run_eval

    cases = _load_cases(args.data)
    if args.limit:
        cases = cases[: args.limit]
    decide_fn = parallel_decide_fn(load_engine(args.model), scoring=args.scoring)

    out_root = Path(args.out)
    runs: dict[int, list[dict]] = {}
    for extra in extra_counts:
        run_dir = out_root / f"extra{extra:02d}"
        variant_cases = []
        for case in cases:
            variant = dict(case)
            variant["schema"] = build_variant_schema(case["schema"], extra, case["id"])
            variant["id"] = f"{case['id']}#x{extra}"
            variant["group_id"] = case["id"]
            variant["meta"] = dict(case.get("meta") or {}, extra_fields=extra)
            variant_cases.append(variant)
        run = run_eval(
            variant_cases,
            decide_fn,
            track="parallel",
            model=args.model,
            out_dir=str(run_dir),
            extra_config={
                "benchmark": "invariance",
                "bench_version": BENCH_VERSION,
                "extra_fields": extra,
            },
        )
        print(f"extra={extra}: {run['counts']} -> {run_dir}")
        with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
            runs[extra] = [json.loads(line) for line in f if line.strip()]

    targets: list[str] = []
    seen: set[str] = set()
    for case in cases:
        for field in target_fields(case):
            if field not in seen:
                seen.add(field)
                targets.append(field)

    rows = invariance_metrics(runs, targets)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "invariance.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": "invariance",
                "bench_version": BENCH_VERSION,
                "config": {
                    "model": args.model,
                    "dataset_path": args.data,
                    "extra": extra_counts,
                    "scoring": args.scoring,
                },
                "targets": rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    summary = write_summary(
        out_root, rows, {"model": args.model, "extra": extra_counts, "dataset_path": args.data}
    )
    print(f"wrote {out_root / 'invariance.json'}")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
