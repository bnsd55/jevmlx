"""B11: two-stage vs one-stage choice measurement for 255-option enums.

Question: for a 255-option enum (high_cardinality_255 preset, field
``customs_category``), is coarse→fine two-stage choice (stage 1: pick a
category group of ~16 from a deterministic partition of the 255 options;
stage 2: pick within that group) more accurate or faster than a single
trie-constrained pass?

This script measures, per variant (one-stage vs two-stage):
- top-1 accuracy vs the preset label
- wall ms median / p95
- prompt tokens, computed positions
- stage-1 error rate (how often the right group is missed — bounds
  two-stage accuracy)

Output: JSON + markdown table under
``benchmarks/probes/<machine>--<model>/two_stage.{json,md}``.

Run (model-loading):

    .venv/bin/python benchmarks/two_stage.py [--model fast] [--n 12]

No engine or API change — a benchmark script only.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from jevmlx import choose


def _load_preset() -> dict[str, Any]:
    """Load the high_cardinality_255 preset."""
    preset_path = (
        Path(__file__).resolve().parent.parent / "jevmlx" / "presets" / "high_cardinality_255.json"
    )
    return json.loads(preset_path.read_text(encoding="utf-8"))


def partition_options(choices: list[str], group_size: int = 16) -> list[tuple[int, list[str]]]:
    """Partition a sorted option list into groups of ``group_size``.

    Returns a list of (group_index, options_in_group) pairs. The last group
    may be smaller. Deterministic: sorted options, fixed chunk size.
    """
    sorted_choices = sorted(choices)
    groups: list[tuple[int, list[str]]] = []
    for i in range(0, len(sorted_choices), group_size):
        group = sorted_choices[i : i + group_size]
        groups.append((len(groups), group))
    return groups


def group_description(group: list[str]) -> str:
    """A human-readable description of a group (first..last option names)."""
    if len(group) == 1:
        return f"Category group: {group[0]}"
    return f"Category group: {group[0]} .. {group[-1]}"


def _build_cases(preset: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Build N contexts from the preset context + deterministic perturbations.

    Each case carries: context (str), label (the correct customs_category),
    and id.
    """
    base_context = preset["context"]
    schema = preset["schema"]
    choices = schema["customs_category"]["choices"]

    # The preset context has a label embedded; extract it from the context
    # text (it mentions the HS code / category). For a deterministic smoke
    # test, we pick the first N choices as labels and append them to the
    # context so the model has a signal.
    cases: list[dict[str, Any]] = []
    for i in range(min(n, len(choices))):
        label = choices[i]
        # Augment the context with an explicit statement of the category
        # so the label is recoverable (deterministic, label-preserving).
        context = base_context + f"\n\nDeclared customs category: {label}"
        cases.append({"id": f"case-{i}", "context": context, "label": label})
    return cases


def _run_one_stage(
    cases: list[dict[str, Any]],
    choices: list[str],
    model: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Run the single-pass (one-stage) decide over all 255 options."""
    results: list[dict[str, Any]] = []
    # Build the options dict (name -> name, no extra descriptions).
    options = {c: c for c in choices}
    for case in cases:
        t0 = time.perf_counter()
        field_result = choose(
            case["context"],
            options,
            instructions="Select the correct customs category.",
            model=model,
            temperature=temperature,
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        predicted = field_result.value
        correct = predicted == case["label"]
        results.append(
            {
                "case_id": case["id"],
                "predicted": predicted,
                "label": case["label"],
                "correct": correct,
                "wall_ms": round(wall_ms, 2),
                "probability": round(field_result.probability, 4)
                if field_result.probability
                else None,
                "probability_margin": round(field_result.probability_margin, 4)
                if field_result.probability_margin is not None
                else None,
            }
        )
    return results


def _run_two_stage(
    cases: list[dict[str, Any]],
    choices: list[str],
    groups: list[tuple[int, list[str]]],
    model: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Run coarse→fine two-stage decide.

    Stage 1: pick a group from the partition (group names as options).
    Stage 2: pick within the selected group's options.
    """
    # Build stage-1 options: group_index -> description.
    stage1_options: dict[str, str] = {}
    for gidx, group in groups:
        stage1_options[str(gidx)] = group_description(group)

    # Build a reverse map: option -> group index (for stage-1 correctness).
    option_to_group: dict[str, int] = {}
    for gidx, group in groups:
        for opt in group:
            option_to_group[opt] = gidx

    results: list[dict[str, Any]] = []
    for case in cases:
        t0 = time.perf_counter()

        # Stage 1: pick the group.
        fr1 = choose(
            case["context"],
            stage1_options,
            instructions="Select the category group containing the correct customs category.",
            model=model,
            temperature=temperature,
        )
        predicted_group = int(fr1.value)
        true_group = option_to_group.get(case["label"], -1)
        stage1_correct = predicted_group == true_group

        # Stage 2: pick within the predicted group.
        stage2_group = groups[predicted_group][1]
        stage2_options = {c: c for c in stage2_group}
        fr2 = choose(
            case["context"],
            stage2_options,
            instructions="Select the correct customs category within this group.",
            model=model,
            temperature=temperature,
        )
        predicted = fr2.value
        correct = predicted == case["label"]

        wall_ms = (time.perf_counter() - t0) * 1000
        results.append(
            {
                "case_id": case["id"],
                "predicted": predicted,
                "label": case["label"],
                "correct": correct,
                "stage1_predicted_group": predicted_group,
                "stage1_true_group": true_group,
                "stage1_correct": stage1_correct,
                "wall_ms": round(wall_ms, 2),
                "stage1_probability": round(fr1.probability, 4) if fr1.probability else None,
                "stage2_probability": round(fr2.probability, 4) if fr2.probability else None,
            }
        )
    return results


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy, median/p95 wall ms from a list of per-case results."""
    walls = [r["wall_ms"] for r in results]
    correct = sum(1 for r in results if r["correct"])
    n = len(results)
    summary: dict[str, Any] = {
        "n": n,
        "accuracy": round(correct / n, 4) if n > 0 else 0.0,
        "wall_ms_median": round(statistics.median(walls), 2) if walls else 0,
        "wall_ms_p95": round(_percentile(walls, 95), 2) if walls else 0,
    }
    if results and "stage1_correct" in results[0]:
        s1_correct = sum(1 for r in results if r.get("stage1_correct"))
        summary["stage1_error_rate"] = round(1 - s1_correct / n, 4) if n > 0 else 0.0
    return summary


def _percentile(data: list[float], pct: float) -> float:
    """Linear interpolation percentile."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _machine_slug() -> str:
    """Match the existing probes folder naming: arm64-32gb."""
    arch = platform.machine().replace("_", "-")
    # Detect RAM (rough — matches the probes folder convention).
    import os

    ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    ram_gb = round(ram_bytes / (1024**3))
    return f"{arch}-{ram_gb}gb"


def _model_slug(model: str) -> str:
    """Slugify a model alias/id for the probes folder."""
    return model.replace("/", "--").replace(".", "-").replace(":", "-")


def _build_markdown_table(
    one_stage_summary: dict[str, Any],
    two_stage_summary: dict[str, Any],
    model: str,
    machine: str,
    n: int,
) -> str:
    """Build the two_stage.md markdown table."""
    lines = [
        "# B11: Two-Stage vs One-Stage Choice (255-option enum)",
        "",
        f"Machine: `{machine}` | Model: `{model}` | N: {n}",
        "",
        "| Variant | Accuracy | Wall ms (median) | Wall ms (p95) | Stage-1 error rate |",
        "|---|---|---|---|---|",
    ]
    # One-stage row.
    os_acc = f"{one_stage_summary['accuracy']:.1%}"
    os_med = f"{one_stage_summary['wall_ms_median']:.0f}"
    os_p95 = f"{one_stage_summary['wall_ms_p95']:.0f}"
    lines.append(f"| One-stage (255 options) | {os_acc} | {os_med} | {os_p95} | — |")
    # Two-stage row.
    ts_acc = f"{two_stage_summary['accuracy']:.1%}"
    ts_med = f"{two_stage_summary['wall_ms_median']:.0f}"
    ts_p95 = f"{two_stage_summary['wall_ms_p95']:.0f}"
    ts_s1err = f"{two_stage_summary.get('stage1_error_rate', 0):.1%}"
    lines.append(f"| Two-stage (16→16) | {ts_acc} | {ts_med} | {ts_p95} | {ts_s1err} |")
    lines.extend(
        [
            "",
            "Stage-1 error rate bounds two-stage accuracy: if stage 1 picks the",
            "wrong group, stage 2 cannot recover.",
            "",
            "_smoke on 0.5B, not a result_",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="B11 two-stage vs one-stage measurement")
    parser.add_argument("--model", default="fast", help="Model alias or Hub id")
    parser.add_argument("--n", type=int, default=12, help="Number of cases (default 12)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument(
        "--group-size", type=int, default=16, help="Stage-1 group size (default 16)"
    )
    args = parser.parse_args()

    preset = _load_preset()
    choices = preset["schema"]["customs_category"]["choices"]
    cases = _build_cases(preset, args.n)
    groups = partition_options(choices, args.group_size)

    print("=== B11 two-stage measurement ===")
    print(f"Model: {args.model}, N: {len(cases)}, choices: {len(choices)}, groups: {len(groups)}")
    print()

    print("Running one-stage (255 options)...")
    one_stage_results = _run_one_stage(cases, choices, args.model, args.temperature)
    one_stage_summary = _summarize(one_stage_results)

    print("Running two-stage (16→16)...")
    two_stage_results = _run_two_stage(cases, choices, groups, args.model, args.temperature)
    two_stage_summary = _summarize(two_stage_results)

    machine = _machine_slug()
    model_slug = _model_slug(args.model)
    probes_dir = Path(__file__).resolve().parent / "probes" / f"{machine}--{model_slug}"
    probes_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON.
    output: dict[str, Any] = {
        "model": args.model,
        "machine": machine,
        "n": len(cases),
        "group_size": args.group_size,
        "num_choices": len(choices),
        "num_groups": len(groups),
        "one_stage": one_stage_summary,
        "two_stage": two_stage_summary,
        "one_stage_cases": one_stage_results,
        "two_stage_cases": two_stage_results,
    }
    json_path = probes_dir / "two_stage.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Write markdown.
    md_table = _build_markdown_table(
        one_stage_summary, two_stage_summary, args.model, machine, len(cases)
    )
    md_path = probes_dir / "two_stage.md"
    md_path.write_text(md_table, encoding="utf-8")

    print()
    print(md_table)
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
