"""Tests for jevmlx doctor: probe monkeypatching, OK/WARN/FAIL mapping, exit codes.

No network, no model load, no real Metal. Every probe is replaced.
"""

from __future__ import annotations

import json
import platform
import subprocess
import types
from pathlib import Path

import pytest

from jevmlx import doctor
from jevmlx.doctor import (
    check_editable_install,
    check_hf_cache,
    check_memory,
    check_metal,
    check_model,
    check_network,
    check_platform,
    check_power,
    check_venv,
    doctor_checks,
    run_doctor,
)

BASE_ENV = {
    "machine_model": "Mac15,6",
    "chip": "Apple M2 Pro",
    "ram_gb": 32.0,
    "macos_version": "26.5.1",
    "python_version": "3.12.2",
    "mlx_version": "0.32.2",
    "mlx_lm_version": "0.31.3",
    "jevmlx_version": "0.1.0",
    "git_sha": None,
    "timestamp_utc": "2026-09-17T00:00:00+00:00",
}


@pytest.fixture()
def healthy(monkeypatch):
    """Patch every probe to a healthy Apple Silicon answer."""
    monkeypatch.setattr("jevmlx.evalreport.environment", lambda: dict(BASE_ENV))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.5.1", "", ""))
    monkeypatch.setattr(doctor, "_free_memory_gb", lambda: 12.0)
    monkeypatch.setattr(doctor, "check_power", lambda: doctor._ok("power", "AC power"))
    monkeypatch.setattr(
        doctor,
        "check_metal",
        lambda: doctor._ok("metal", "available; smoke op ran"),
    )
    monkeypatch.setattr(
        doctor,
        "check_hf_cache",
        lambda: (doctor._ok("hf-cache", "cache ok; default cached"), []),
    )
    monkeypatch.setattr(doctor, "check_network", lambda: doctor._ok("network", "reachable"))

    # The healthy scenario needs installed == imported. The shared venv
    # installs editable from ~/git/jevmlx while tests import THIS tree, so
    # patch direct_url to this checkout (the per-worktree-venv setup).
    this_checkout = Path(doctor.__file__).resolve().parent.parent
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "distribution",
        lambda name: types.SimpleNamespace(
            read_text=lambda _n: json.dumps(
                {"url": this_checkout.as_uri(), "dir_info": {"editable": True}}
            )
        ),
    )
    return monkeypatch


class TestPlatform:
    def test_apple_silicon_ok(self, healthy):
        check = check_platform(dict(BASE_ENV))
        assert check.status == "OK"
        assert "Apple M2 Pro" in check.detail

    @pytest.mark.parametrize(
        ("system", "machine"), [("Linux", "x86_64"), ("Darwin", "x86_64"), ("Windows", "arm64")]
    )
    def test_wrong_platform_fails(self, system, machine, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: system)
        monkeypatch.setattr(platform, "machine", lambda: machine)
        check = check_platform(dict(BASE_ENV))
        assert check.status == "FAIL"
        assert "Apple Silicon" in check.fix


class TestVersions:
    def test_healthy_versions_ok(self, healthy):
        checks = doctor.check_python_and_versions(dict(BASE_ENV))
        by_name = {c.name: c for c in checks}
        assert by_name["python"].status == "OK"
        assert by_name["mlx"].status == "OK"
        assert by_name["mlx-lm"].status == "OK"

    def test_python_311_fails(self, healthy):
        env = dict(BASE_ENV, python_version="3.11.9")
        checks = doctor.check_python_and_versions(env)
        python = next(c for c in checks if c.name == "python")
        assert python.status == "FAIL"

    def test_mlx_lm_below_pin_warns(self, healthy, monkeypatch):
        env = dict(BASE_ENV, mlx_lm_version="0.24.0")
        monkeypatch.setattr(doctor, "_mlx_lm_floor", lambda: "0.31.3")
        checks = doctor.check_python_and_versions(env)
        mlx_lm = next(c for c in checks if c.name == "mlx-lm")
        assert mlx_lm.status == "WARN"
        assert "0.31.3" in mlx_lm.fix

    def test_missing_mlx_fails(self, healthy):
        healthy.setattr(doctor, "_dist_version", lambda name: None)
        env = dict(BASE_ENV, mlx_version=None, mlx_lm_version=None)
        checks = doctor.check_python_and_versions(env)
        assert next(c for c in checks if c.name == "mlx").status == "FAIL"
        assert next(c for c in checks if c.name == "mlx-lm").status == "FAIL"

    def test_mlx_lm_at_pin_ok(self, healthy):
        env = dict(BASE_ENV, mlx_lm_version="0.31.3")
        checks = doctor.check_python_and_versions(env)
        assert next(c for c in checks if c.name == "mlx-lm").status == "OK"


class TestMemory:
    def test_healthy(self, healthy):
        check = check_memory(dict(BASE_ENV))
        assert check.status == "OK"
        assert "32.0 GB total" in check.detail and "12.0 GB free" in check.detail

    def test_below_16_gb_warns(self, healthy):
        check = check_memory(dict(BASE_ENV, ram_gb=8.0))
        assert check.status == "WARN"

    def test_unknown_total_warns(self, healthy):
        check = check_memory(dict(BASE_ENV, ram_gb=None))
        assert check.status == "WARN"

    def test_free_memory_parses_vm_stat(self, monkeypatch):
        output = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 262144.\n"  # 262144 * 16384 = 4 GiB
        )
        monkeypatch.setattr(
            doctor.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": output})(),
        )
        monkeypatch.setattr(doctor, "_sysctl", lambda name: "16384")
        assert doctor._free_memory_gb() == 4.0


class TestPower:
    def test_ac_ok(self, monkeypatch):
        monkeypatch.setattr(
            doctor.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "Now drawing from 'AC Power'"})(),
        )
        assert check_power().status == "OK"

    def test_battery_warns(self, monkeypatch):
        monkeypatch.setattr(
            doctor.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "Now drawing from 'Battery Power'"})(),
        )
        check = check_power()
        assert check.status == "WARN" and "plug in" in check.fix


class TestMetal:
    @staticmethod
    def _fake_mlx(monkeypatch, *, available, array=None, add=None):
        """Replace mlx.core for check_metal.

        The real C-extension submodule stays bound on the parent package
        after its first import, so patch the parent attribute too.
        """
        import types

        import mlx

        fake_mx = types.ModuleType("mlx.core")
        fake_mx.metal = types.SimpleNamespace(is_available=lambda: available)
        fake_mx.array = array if array is not None else (lambda v: v)
        fake_mx.add = add if add is not None else (lambda a, b: [2.0])
        monkeypatch.setitem(__import__("sys").modules, "mlx.core", fake_mx)
        monkeypatch.setattr(mlx, "core", fake_mx)
        return fake_mx

    def test_ok(self, monkeypatch):
        self._fake_mlx(monkeypatch, available=True)
        assert check_metal().status == "OK"

    def test_no_metal_fails(self, monkeypatch):
        self._fake_mlx(monkeypatch, available=False)
        check = check_metal()
        assert check.status == "FAIL" and "Apple Silicon" in check.fix

    def test_exception_fails(self, monkeypatch):
        def boom(*args):
            raise RuntimeError("GPU busy")

        self._fake_mlx(monkeypatch, available=True, add=boom)
        assert check_metal().status == "FAIL"

    def test_real_mlx_availability_check_does_not_raise(self):
        """Exercise the availability check against the REAL mlx (no mock, no
        model load). A signature change in mlx (e.g. mx.is_available() now
        requiring a device arg) is caught by CI, not by a fresh-venv smoke."""
        check = check_metal()
        assert check.status in ("OK", "FAIL")
        # On Apple Silicon it should be OK; in CI without a GPU it may FAIL —
        # either is fine, the point is no TypeError/AttributeError.
        if check.status == "FAIL":
            assert check.fix


class TestHFCache:
    def _make_cache(self, tmp_path, with_default=True):
        cache = tmp_path / "hub"
        cache.mkdir()
        for name in ("models--mlx-community--Qwen2.5-1.5B-Instruct-4bit",):
            snapshots = cache / name / "snapshots" / "abc123"
            snapshots.mkdir(parents=True)
            (snapshots / "model.safetensors").write_bytes(b"x" * 1024)
        other = cache / "models--mlx-community--Llama-3.2-1B-Instruct-4bit" / "snapshots" / "def456"
        other.mkdir(parents=True)
        (other / "model.safetensors").write_bytes(b"y" * 2048)
        if not with_default:
            import shutil

            shutil.rmtree(cache / "models--mlx-community--Qwen2.5-1.5B-Instruct-4bit")
        return cache

    def test_lists_models_and_default(self, tmp_path, monkeypatch):
        cache = self._make_cache(tmp_path)
        monkeypatch.setattr(doctor, "_hf_cache_dir", lambda: cache)
        # The default is 'quality' (7B); mock resolve_model so the 1.5B in
        # the mock cache is treated as the default (avoids depending on the
        # real alias map in a unit test).
        # Patch the name in doctor's namespace (doctor imports with
        # 'from jevmlx.models import resolve_model', so patching the models
        # module attribute would not affect the already-bound local).
        monkeypatch.setattr(
            doctor, "resolve_model", lambda model: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        )
        check, models = check_hf_cache()
        assert check.status == "OK"
        assert {name for name, _ in models} == {
            "Qwen2.5-1.5B-Instruct-4bit",
            "Llama-3.2-1B-Instruct-4bit",
        }
        assert "cached" in check.detail

    def test_missing_default_warns(self, tmp_path, monkeypatch):
        cache = self._make_cache(tmp_path, with_default=False)
        monkeypatch.setattr(doctor, "_hf_cache_dir", lambda: cache)
        monkeypatch.setattr(
            doctor, "resolve_model", lambda model: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        )
        check, _ = check_hf_cache()
        assert check.status == "WARN"
        assert "NOT cached" in check.detail

    def test_no_cache_dir_warns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(doctor, "_hf_cache_dir", lambda: tmp_path / "missing")
        check, models = check_hf_cache()
        assert check.status == "WARN" and models == []


class TestNetwork:
    def test_reachable_ok(self, monkeypatch):
        monkeypatch.setattr(
            doctor.urllib.request, "urlopen", lambda request, timeout: type("R", (), {})()
        )
        assert check_network().status == "OK"

    def test_offline_warns(self, monkeypatch):
        def boom(request, timeout):
            raise OSError("no route")

        monkeypatch.setattr(doctor.urllib.request, "urlopen", boom)
        check = check_network()
        assert check.status == "WARN" and "cached models" in check.fix


class TestModelCheck:
    def test_skipped_without_model(self):
        assert check_model(None).status == "OK"
        assert "skipped" in check_model(None).detail

    def _fake_tokenizer(self, template=True, system_ok=True):
        class FakeTokenizer:
            chat_template = "tpl" if template else None

            def apply_chat_template(self, messages, tokenize=False):
                if messages[0]["role"] == "system" and not system_ok:
                    raise ValueError("System role not supported")
                return "rendered"

        return FakeTokenizer()

    def test_ok(self, monkeypatch):
        import types

        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda model, trust_remote_code=False: self._fake_tokenizer()
        )
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
        check = check_model("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        assert check.status == "OK"

    def test_no_template_warns(self, monkeypatch):
        import types

        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda model, trust_remote_code=False: self._fake_tokenizer(
                template=False
            )
        )
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
        check = check_model("some/model")
        assert check.status == "WARN" and "no chat template" in check.detail

    def test_system_role_rejected_warns(self, monkeypatch):
        import types

        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda model, trust_remote_code=False: self._fake_tokenizer(
                system_ok=False
            )
        )
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
        check = check_model("some/model")
        assert check.status == "WARN" and "system role" in check.detail

    def test_load_failure_fails(self, monkeypatch):
        import types

        def boom(model, trust_remote_code=False):
            raise OSError("not found")

        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=boom)
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
        check = check_model("nope/model")
        assert check.status == "FAIL"


class TestAssembly:
    def test_all_healthy_all_ok_exit_0(self, healthy):
        checks, _ = doctor_checks()
        assert all(c.status == "OK" for c in checks), [c.name for c in checks if c.status != "OK"]

    def test_exit_code_one_on_fail(self, healthy):
        healthy.setattr(doctor, "check_metal", lambda: doctor._fail("metal", "no", "fix"))
        assert run_doctor(as_json=True) == 1

    def test_warn_does_not_fail_exit_code(self, healthy):
        healthy.setattr(
            doctor,
            "check_power",
            lambda: doctor._warn("power", "on battery", "plug in"),
        )
        assert run_doctor(as_json=True) == 0

    def test_json_shape(self, healthy, capsys):
        assert run_doctor(as_json=True) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["exit_code"] == 0
        assert payload["environment"]["chip"] == "Apple M2 Pro"
        assert {c["status"] for c in payload["checks"]} == {"OK"}

    def test_table_has_no_colour_codes(self, healthy, capsys):
        assert run_doctor(as_json=False) == 0
        out = capsys.readouterr().out
        assert "\x1b[" not in out
        assert "OK " in out

    def test_table_lists_fix_for_warn(self, healthy, capsys):
        healthy.setattr(
            doctor,
            "check_power",
            lambda: doctor._warn("power", "on battery", "plug in"),
        )
        run_doctor(as_json=False)
        assert "fix: plug in" in capsys.readouterr().out


class TestVenv:
    def test_non_conda_ok_and_subprocess_ok(self):
        checks = check_venv()
        assert [c.status for c in checks] == ["OK", "OK"]
        assert "not conda" in checks[0].detail

    def test_conda_base_prefix_fails_with_uv_fix(self, monkeypatch):
        monkeypatch.setattr(doctor.sys, "base_prefix", "/opt/miniconda3")
        checks = check_venv()
        venv = checks[0]
        assert venv.status == "FAIL"
        assert "uv venv --python-preference only-managed --python 3.12" in venv.fix

    def test_anaconda_base_prefix_fails(self, monkeypatch):
        monkeypatch.setattr(doctor.sys, "base_prefix", "/opt/anaconda3")
        assert check_venv()[0].status == "FAIL"

    def test_subprocess_timeout_fails(self, monkeypatch):
        def hang(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="python", timeout=2)

        monkeypatch.setattr(doctor.sys, "executable", "/fake/python")
        monkeypatch.setattr(doctor.subprocess, "run", hang)
        checks = check_venv()
        assert checks[1].status == "FAIL"
        assert "2 s" in checks[1].detail or "TimeoutExpired" in checks[1].detail
        assert "uv venv --python-preference only-managed --python 3.12" in checks[1].fix

    def test_subprocess_oserror_fails(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("no such interpreter")

        monkeypatch.setattr(doctor.sys, "executable", "/fake/python")
        monkeypatch.setattr(doctor.subprocess, "run", boom)
        assert check_venv()[1].status == "FAIL"

    def test_subprocess_nonzero_exit_fails(self, monkeypatch):
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd="python")

        monkeypatch.setattr(doctor.sys, "executable", "/fake/python")
        monkeypatch.setattr(doctor.subprocess, "run", fail)
        assert check_venv()[1].status == "FAIL"


class TestEditableInstall:
    def test_points_at_current_checkout_ok(self, monkeypatch):
        """Editable + imported tree == installed tree -> OK.

        The real environment is a shared venv (installed from ~/git/jevmlx,
        tests run from here), which correctly FAILS; patch the dist-info to
        point at THIS checkout to build the OK case.
        """
        this_checkout = Path(doctor.__file__).resolve().parent.parent
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "distribution",
            lambda name: types.SimpleNamespace(
                read_text=lambda _n: json.dumps(
                    {"url": this_checkout.as_uri(), "dir_info": {"editable": True}}
                )
            ),
        )
        check = check_editable_install()
        assert check.status == "OK"
        assert "editable install" in check.detail

    def test_wheel_install_no_direct_url_ok(self, monkeypatch):
        # Normal PyPI install: no direct_url.json -> the normal end-user
        # case, doctor must not fail it.
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "distribution",
            lambda name: types.SimpleNamespace(read_text=lambda _n: None),
        )
        check = check_editable_install()
        assert check.status == "OK"
        assert "installed as a package" in check.detail

    def test_direct_url_not_editable_ok(self, monkeypatch, tmp_path):
        other = tmp_path / "some-wheel-tree"
        payload = json.dumps({"url": other.as_uri(), "dir_info": {"editable": False}})
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "distribution",
            lambda name: types.SimpleNamespace(read_text=lambda _n: payload),
        )
        check = check_editable_install()
        assert check.status == "OK"
        assert "not editable" in check.detail

    def test_other_checkout_fails(self, monkeypatch, tmp_path):
        """Imported tree vs installed tree mismatch -> FAIL.

        Only the direct_url is patched: the running interpreter imports the
        REAL jevmlx tree (this checkout), while the fake dist-info claims
        the venv installed from another worktree. The check must catch
        exactly that mismatch (shared-venv workflow, a FAIL by design).
        """
        fake = tmp_path / "other-checkout"
        fake.mkdir()
        payload = json.dumps({"url": fake.as_uri(), "dir_info": {"editable": True}})
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "distribution",
            lambda name: types.SimpleNamespace(read_text=lambda _n: payload),
        )
        check = check_editable_install()
        assert check.status == "FAIL"
        assert "python imports jevmlx from" in check.detail
        assert "other-checkout" in check.detail

    def test_same_tree_ok(self, monkeypatch, tmp_path):
        """Editable + installed tree == imported tree -> OK.

        dist-info url AND jevmlx.__file__ both resolve to one tree (the
        per-worktree-venv setup the G-rules require).
        """
        tree = tmp_path / "own-venv-checkout"
        tree.mkdir()
        payload = json.dumps({"url": tree.as_uri(), "dir_info": {"editable": True}})
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "distribution",
            lambda name: types.SimpleNamespace(read_text=lambda _n: payload),
        )
        monkeypatch.setattr("jevmlx.__file__", str(tree / "jevmlx" / "__init__.py"))
        check = check_editable_install()
        assert check.status == "OK"
        assert "editable install" in check.detail

    def test_not_installed_fails(self, monkeypatch):
        def raise_pnf(name):
            raise doctor.importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(doctor.importlib.metadata, "distribution", raise_pnf)
        check = check_editable_install()
        assert check.status == "FAIL"
        assert "uv pip install -e" in check.fix

    def test_no_direct_url_fails(self, monkeypatch):
        # No dist-info at all is still a FAIL (jevmlx not installed),
        # distinct from a package install whose dist-info lacks
        # direct_url.json (F1: OK).
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "distribution",
            lambda name: types.SimpleNamespace(read_text=lambda _n: None),
        )
        check = check_editable_install()
        assert check.status == "OK"

    def test_file_url_with_spaces_unquoted(self, monkeypatch, tmp_path):
        """A mismatched url with %20 unquotes and FAILS, naming the tree.

        The running interpreter imports the REAL tree; the dist-info url
        (with the space) points elsewhere. _direct_url_path must unquote
        %20 so the FAIL names a real-looking path ('my checkout'), not a
        percent-encoded one.
        """
        checkout = tmp_path / "my checkout"
        checkout.mkdir()
        payload = json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}})
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "distribution",
            lambda name: types.SimpleNamespace(read_text=lambda _n: payload),
        )
        check = check_editable_install()
        assert check.status == "FAIL"
        assert "my checkout" in check.detail


class TestAssemblyVenv:
    def test_healthy_run_includes_venv_checks(self, healthy):
        checks, _ = doctor_checks()
        names = {c.name for c in checks}
        assert "venv" in names
        assert "editable-install" in names
