"""``jevmlx doctor``: environment checks before filing an issue or a bench run.

Checks are read-only probes; the only network touch is a 3-second HEAD to
huggingface.co (skippable by monkeypatching in tests). Exit code is 1 when
any check FAILs, 0 otherwise (WARN does not fail).

Check status semantics:

- OK: the probe ran and the condition holds.
- WARN: not fatal for an issue report, but degrades comparability (battery,
  low memory, stale mlx-lm, HF unreachable, tokenizer without system role).
- FAIL: the environment cannot run jevmlx at all (wrong platform, no Metal).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

__all__ = ["Check", "doctor_checks", "run_doctor"]


@dataclasses.dataclass
class Check:
    """One doctor check result."""

    name: str
    status: str  # "OK" | "WARN" | "FAIL"
    detail: str
    fix: str = ""  # one-line fix; only set for WARN/FAIL


def _ok(name: str, detail: str) -> Check:
    return Check(name, "OK", detail)


def _warn(name: str, detail: str, fix: str) -> Check:
    return Check(name, "WARN", detail, fix)


def _fail(name: str, detail: str, fix: str) -> Check:
    return Check(name, "FAIL", detail, fix)


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _mlx_lm_floor() -> str | None:
    """The mlx-lm minimum from pyproject dependencies (mlx-lm>=X.Y.Z)."""
    try:
        text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    except OSError:
        return None
    match = re.search(r"mlx-lm>=([\d.]+)", text)
    return match.group(1) if match else None


def _version_at_least(version: str, floor: str) -> bool:
    def parts(v: str) -> list[int]:
        return [int(p) for p in re.findall(r"\d+", v)[:3]]

    return parts(version) >= parts(floor)


# ------------------------------------------------------------------- checks


def check_platform(env: dict) -> Check:
    """Apple Silicon + macOS, or FAIL."""
    system = platform.system()
    machine = platform.machine()
    if system != "Darwin" or machine != "arm64":
        return _fail(
            "platform",
            f"{system}/{machine} (need Darwin/arm64)",
            "run on Apple Silicon macOS; jevmlx is Metal-only",
        )
    macos = env.get("macos_version") or "unknown"
    chip = env.get("chip") or "unknown chip"
    return _ok("platform", f"{chip}, macOS {macos}")


def check_python_and_versions(env: dict) -> list[Check]:
    """Python >= 3.12, then package versions; WARN if mlx-lm is older than pin."""
    checks = []
    version = env.get("python_version") or ".".join(map(str, sys.version_info[:3]))
    minor = int(re.findall(r"\d+", version)[:3][1]) if re.findall(r"\d+", version) else 0
    if version == "unknown" or minor < 12:
        checks.append(
            _fail(
                "python",
                f"{version} (need >= 3.12)",
                "uv python install 3.12 && uv venv --python 3.12 .venv",
            )
        )
    else:
        checks.append(_ok("python", version))

    jevmlx = env.get("jevmlx_version") or _dist_version("jevmlx")
    mlx = env.get("mlx_version") or _dist_version("mlx")
    mlx_lm = env.get("mlx_lm_version") or _dist_version("mlx-lm")
    if mlx is None:
        checks.append(_fail("mlx", "not installed", "uv pip install -e '.[dev]'"))
    else:
        checks.append(_ok("mlx", mlx))
    if mlx_lm is None:
        checks.append(_fail("mlx-lm", "not installed", "uv pip install -e '.[dev]'"))
    else:
        floor = _mlx_lm_floor()
        if floor and not _version_at_least(mlx_lm, floor):
            checks.append(
                _warn(
                    "mlx-lm",
                    f"{mlx_lm} (pyproject pins >= {floor})",
                    f"uv pip install -U 'mlx-lm>={floor}'",
                )
            )
        else:
            checks.append(_ok("mlx-lm", mlx_lm))
    checks.append(_ok("jevmlx", jevmlx or "unknown (not installed as a package)"))
    return checks


def check_memory(env: dict) -> Check:
    """Unified memory total (WARN < 16 GB) and currently free (vm_stat)."""
    total_gb = env.get("ram_gb")
    free_gb = _free_memory_gb()
    if total_gb is None:
        return _warn(
            "memory",
            f"total unknown, free ~{free_gb if free_gb is not None else '?'} GB",
            "unsupported platform for hw.memsize; numbers not comparable",
        )
    detail = f"{total_gb} GB total, {free_gb if free_gb is not None else '?'} GB free"
    if total_gb < 16:
        return _warn("memory", detail, "models >= 4B quantized may not fit; use smaller models")
    return _ok("memory", detail)


def _free_memory_gb() -> float | None:
    """Currently free pages from vm_stat (pagesize from hw.pagesize)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"Pages free:\s+(\d+)", out.stdout)
    if not match:
        return None
    page_raw = _sysctl("hw.pagesize")
    page_size = int(page_raw) if page_raw and page_raw.isdigit() else 16384
    return round(int(match.group(1)) * page_size / 2**30, 1)


def check_power() -> Check:
    """On battery -> WARN (bench refuses; doctor only warns)."""
    try:
        battery = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return _warn("power", "pmset unavailable", "bench will refuse to run on battery")
    if "AC Power" not in battery:
        return _warn(
            "power",
            "on battery",
            "plug in: bench refuses battery power (thermal throttling)",
        )
    return _ok("power", "AC power")


def check_metal() -> Check:
    """Metal available + a 1-element op runs, or FAIL."""
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return _fail(
                "metal",
                "mx.metal.is_available() is False",
                "run on Apple Silicon; check macOS GPU restrictions",
            )
        result = mx.add(mx.array([1.0]), mx.array([1.0]))
        _ = float(result[0])
    except Exception as e:  # noqa: BLE001 - any import/exec failure is a FAIL
        return _fail("metal", f"{type(e).__name__}: {e}", "reinstall mlx: uv pip install -U mlx")
    return _ok("metal", "available; smoke op ran")


def _hf_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _cached_models(cache: Path) -> list[tuple[str, float]]:
    """mlx-community models in the HF cache: (name, size GB) from snapshots."""
    models = []
    if not cache.is_dir():
        return models
    for model_dir in sorted(cache.glob("models--mlx-community--*")):
        snapshots = model_dir / "snapshots"
        size = _dir_size_bytes(snapshots) if snapshots.is_dir() else _dir_size_bytes(model_dir)
        name = model_dir.name.removeprefix("models--mlx-community--").replace("--", "/")
        models.append((name, round(size / 2**30, 2)))
    return models


def check_hf_cache() -> tuple[Check, list[tuple[str, float]]]:
    """Cache path, size, cached mlx-community models; default model cached?"""
    import jevmlx.api as api

    cache = _hf_cache_dir()
    if not cache.is_dir():
        return (
            _warn(
                "hf-cache",
                f"{cache} does not exist",
                "run any jevmlx command once, or: jevmlx calibrate (downloads the model)",
            ),
            [],
        )
    models = _cached_models(cache)
    total_gb = round(_dir_size_bytes(cache) / 2**30, 2)
    default_hub_id = api.resolve_model(api.DEFAULT_MODEL)
    slug = "models--" + default_hub_id.replace("/", "--")
    default_cached = (cache / slug).is_dir()
    detail = (
        f"{cache} ({total_gb} GB, {len(models)} mlx-community models); "
        f"default {api.DEFAULT_MODEL} ({default_hub_id}): "
        f"{'cached' if default_cached else 'NOT cached'}"
    )
    if not default_cached:
        return (
            _warn("hf-cache", detail, f"first run will download {default_hub_id}"),
            models,
        )
    return _ok("hf-cache", detail), models


def check_network(timeout: float = 3.0) -> Check:
    """huggingface.co reachable -> OK; offline -> WARN (cached models still work)."""
    try:
        request = urllib.request.Request(
            "https://huggingface.co", method="HEAD", headers={"User-Agent": "jevmlx-doctor"}
        )
        urllib.request.urlopen(request, timeout=timeout)
    except Exception:  # noqa: BLE001 - any failure means unreachable
        return _warn(
            "network",
            "huggingface.co unreachable",
            "offline: only cached models will work; check network or proxy",
        )
    return _ok("network", "huggingface.co reachable")


def check_model(model: str | None) -> Check:
    """Optional: dry-run tokenizer load; chat template + system-role support."""
    if model is None:
        return _ok("model", "skipped (pass --model M to check a tokenizer)")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    except Exception as e:  # noqa: BLE001 - surfaced as FAIL with the reason
        return _fail(
            "model",
            f"tokenizer load failed: {type(e).__name__}: {e}",
            f"check the model id '{model}' and your HF cache/network",
        )
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return _warn(
            "model",
            f"{model}: no chat template",
            "pick an instruct model with a chat template (engine needs one)",
        )
    system_ok = _tokenizer_accepts_system(tokenizer)
    if not system_ok:
        return _warn(
            "model",
            f"{model}: chat template rejects a system role",
            "jevmlx falls back to a user-only prompt (V1 fallback); quality may differ",
        )
    return _ok("model", f"{model}: chat template ok, system role accepted")


def _tokenizer_accepts_system(tokenizer) -> bool:
    """True if the chat template renders a system message without error.

    Some templates (V1 fallback case) drop or reject the system role.
    """
    try:
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            tokenize=False,
        )
    except Exception:  # noqa: BLE001 - template errors mean the role is unsupported
        return False
    return True


# ----------------------------------------------------------------- assembly


def doctor_checks(model: str | None = None) -> tuple[list[Check], dict]:
    """Run all checks; returns (checks, environment dict).

    Platform FAIL short-circuits nothing — later checks still run but their
    results are informational.
    """
    from jevmlx.evalreport import environment

    env = environment()
    checks: list[Check] = [check_platform(env)]
    checks += check_python_and_versions(env)
    checks.append(check_memory(env))
    checks.append(check_power())
    checks.append(check_metal())
    hf_check, _models = check_hf_cache()
    checks.append(hf_check)
    checks.append(check_network())
    checks.append(check_model(model))
    return checks, env


def _render_table(checks: list[Check]) -> str:
    """Plain table, no colour codes."""
    width = max((len(c.name) for c in checks), default=8)
    lines = []
    for check in checks:
        line = f"{check.status:<4}  {check.name:<{width}}  {check.detail}"
        lines.append(line)
        if check.status != "OK" and check.fix:
            lines.append(f"         fix: {check.fix}")
    return "\n".join(lines)


def run_doctor(model: str | None = None, as_json: bool = False) -> int:
    """Entry point for the CLI subcommand. Returns the process exit code."""
    checks, env = doctor_checks(model)
    if as_json:
        payload = {
            "environment": env,
            "checks": [dataclasses.asdict(check) for check in checks],
            "exit_code": 1 if any(c.status == "FAIL" for c in checks) else 0,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_render_table(checks))
    return 1 if any(c.status == "FAIL" for c in checks) else 0
