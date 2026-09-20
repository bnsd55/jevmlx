"""Model alias and Hub-id verification tests.

The alias map (fast/quality/test) is pure data and tested without network.
The slow test verifies the resolved Hub ids actually exist on the Hub via a
HEAD request — never at import time, only in the slow suite.
"""

from __future__ import annotations

import urllib.request

import pytest

from jevmlx.models import DEFAULT_MODEL, DEFAULT_SCORING, MODEL_ALIASES, resolve_model


class TestDefaultScoring:
    """The default scorer is 'labels' (measured better on the 7B; slots
    stays available via scoring='slots')."""

    def test_default_scoring_is_labels(self):
        assert DEFAULT_SCORING == "labels"

    def test_decide_signature_uses_default_scoring(self):
        import inspect

        from jevmlx.api import decide, decide_many

        for fn in (decide, decide_many):
            sig = inspect.signature(fn)
            assert sig.parameters["scoring"].default is DEFAULT_SCORING

    def test_cli_default_is_labels(self):
        # The decide subcommand's --scoring default must be DEFAULT_SCORING.
        # We parse ``jevmlx decide --help`` and inspect the help text.
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from jevmlx.cli import main

        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                main(["decide", "--help"])
        except SystemExit:
            pass
        help_text = buf.getvalue()
        assert "labels" in help_text
        assert "default" in help_text

    def test_default_scoring_exported_from_package(self):
        import jevmlx

        assert hasattr(jevmlx, "DEFAULT_SCORING")
        assert jevmlx.DEFAULT_SCORING == "labels"


class TestAliasResolution:
    """Pure-data: alias resolution is deterministic, no network."""

    def test_default_model_is_quality_alias(self):
        assert DEFAULT_MODEL == "quality"

    def test_fast_alias_resolves_to_3b(self):
        assert resolve_model("fast") == "mlx-community/Qwen2.5-3B-Instruct-4bit"

    def test_quality_alias_resolves_to_7b(self):
        assert resolve_model("quality") == "mlx-community/Qwen2.5-7B-Instruct-4bit"

    def test_test_alias_resolves_to_1_5b(self):
        assert resolve_model("test") == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

    def test_aliases_are_case_insensitive(self):
        assert resolve_model("FAST") == resolve_model("fast")
        assert resolve_model("Quality") == resolve_model("quality")

    def test_literal_hub_id_passes_through(self):
        literal = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        assert resolve_model(literal) == literal

    def test_unknown_string_passes_through(self):
        # A string without "/" and not in the alias map is returned as-is;
        # load_engine will fail with a clear error if it doesn't exist.
        assert resolve_model("some-unknown-model") == "some-unknown-model"

    def test_every_alias_resolves_to_a_hub_id_with_slash(self):
        for alias in MODEL_ALIASES:
            resolved = resolve_model(alias)
            assert "/" in resolved, f"alias {alias!r} did not resolve to a Hub id: {resolved}"


@pytest.mark.slow
@pytest.mark.network  # HEAD https://huggingface.co/<id> — never runs offline
class TestAliasHubIdsExist:
    """Verify the resolved Hub ids exist on huggingface.co (slow, network)."""

    @pytest.mark.parametrize("alias", list(MODEL_ALIASES))
    def test_hub_id_exists(self, alias):
        hub_id = resolve_model(alias)
        url = f"https://huggingface.co/{hub_id}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                assert resp.status == 200, f"{hub_id}: HTTP {resp.status}"
        except Exception as exc:
            pytest.fail(f"Hub id {hub_id} (alias {alias!r}) not reachable: {exc}")
