"""B10: content_moderation and inbound_email preset tests.

Exercises schema validity and the CLI 'decide --preset' path with the
fake engine (no real model load), matching the pattern in test_smoke.py
and test_cli_smoke.py.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeModel, FakeTokenizer, make_engine, make_engine_result, make_field_telemetry

from jevmlx import cli
from jevmlx.cli import load_preset
from jevmlx.schema import StructuredSchema


@pytest.fixture()
def fake_engine(monkeypatch):
    """cli's lazy engine seam -> fake engine returning valid choices."""
    calls: list[str] = []

    def fake_load_engine(model_id: str):
        calls.append(model_id)
        return make_engine(FakeModel(), FakeTokenizer(), model_id=model_id)

    def fake_rpg(engine, context, schema, **kwargs):
        # Return the first valid choice for each field so the result matches.
        fields = {}
        parsed = {}
        for fname, fdef in schema.fields.items():
            if fdef.field_type == "boolean":
                fields[fname] = make_field_telemetry(
                    value=True, type_="boolean", choices=["true", "false"], probability=0.9
                )
                parsed[fname] = {"value": True, "prob": 0.9}
            else:
                val = fdef.choices[0]
                fields[fname] = make_field_telemetry(
                    value=val, choices=list(fdef.choices), probability=0.9
                )
                parsed[fname] = {"value": val, "prob": 0.9}
        return make_engine_result(fields=fields, parsed=parsed)

    monkeypatch.setattr(cli, "_engine", lambda: (fake_load_engine, fake_rpg))
    import sys

    import jevmlx.engine as engine_mod

    monkeypatch.setattr(engine_mod, "load_engine", fake_load_engine)
    monkeypatch.setattr(engine_mod, "run_parallel_generation", fake_rpg)
    monkeypatch.setitem(sys.modules, "jevmlx.cli", cli)
    monkeypatch.setitem(sys.modules, "jevmlx.engine", engine_mod)
    return calls


def _run(args):
    """Run cli.main, return (code, stdout, stderr)."""
    import contextlib
    import io

    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main(args)
        except SystemExit as exc:
            code = exc.code or 0
    return code, out.getvalue(), err.getvalue()


class TestContentModerationPreset:
    def test_schema_validity(self):
        """Schema parses with correct field count and ordered severity."""
        preset = load_preset("content_moderation")
        schema = StructuredSchema(preset["schema"])
        assert len(schema.fields) == 20
        assert "violation_category" in schema.fields
        assert "severity" in schema.fields
        assert schema.fields["severity"].ordered is True
        assert schema.fields["severity"].choices == (
            "none",
            "low",
            "medium",
            "high",
            "critical",
        )
        assert "requires_human_review" in schema.fields
        assert schema.fields["requires_human_review"].field_type == "boolean"
        assert "user_intent" in schema.fields
        assert "action_priority" in schema.fields
        assert schema.fields["action_priority"].ordered is True
        assert "strike_count_tier" in schema.fields
        assert schema.fields["strike_count_tier"].ordered is True

    def test_decide_via_cli(self, fake_engine, capsys):
        """`decide --preset content_moderation --json` runs end to end."""
        code, out, err = _run(["decide", "--preset", "content_moderation", "--json"])
        assert code == 0, err
        result = json.loads(out)
        assert "violation_category" in result
        assert "severity" in result
        assert "requires_human_review" in result
        assert "user_intent" in result


class TestInboundEmailPreset:
    def test_schema_validity(self):
        """Schema parses with correct field count and ordered priority."""
        preset = load_preset("inbound_email")
        schema = StructuredSchema(preset["schema"])
        assert len(schema.fields) == 19
        assert "destination" in schema.fields
        assert "is_spam_or_phishing" in schema.fields
        assert schema.fields["is_spam_or_phishing"].field_type == "boolean"
        assert "priority" in schema.fields
        assert schema.fields["priority"].ordered is True
        assert schema.fields["priority"].choices == ("low", "normal", "high", "urgent")
        assert "needs_reply" in schema.fields
        assert schema.fields["needs_reply"].field_type == "boolean"
        assert "sla_deadline_hours" in schema.fields
        assert schema.fields["sla_deadline_hours"].ordered is True
        assert "confidence_tier" not in schema.fields

    def test_decide_via_cli(self, fake_engine, capsys):
        """`decide --preset inbound_email --json` runs end to end."""
        code, out, err = _run(["decide", "--preset", "inbound_email", "--json"])
        assert code == 0, err
        result = json.loads(out)
        assert "destination" in result
        assert "is_spam_or_phishing" in result
        assert "priority" in result
        assert "needs_reply" in result


class TestSupportTriageExtension:
    def test_new_ordered_fields_present(self):
        """support_triage now has frustration_level and churn_risk (ordered)."""
        preset = load_preset("support_triage")
        schema = StructuredSchema(preset["schema"])
        assert "frustration_level" in schema.fields
        assert schema.fields["frustration_level"].ordered is True
        assert schema.fields["frustration_level"].choices == (
            "calm",
            "annoyed",
            "frustrated",
            "angry",
            "enraged",
        )
        assert "churn_risk" in schema.fields
        assert schema.fields["churn_risk"].ordered is True
        assert schema.fields["churn_risk"].choices == (
            "none",
            "low",
            "medium",
            "high",
            "critical",
        )
        # Original churn_risk_level still present (not replaced).
        assert "churn_risk_level" in schema.fields
        # Now 30 fields (was 28 + 2 new).
        assert len(schema.fields) == 30


class TestAllPresetsValid:
    """Every bundled preset has a valid schema and loadable context."""

    @pytest.mark.parametrize(
        "name",
        [
            "content_moderation",
            "inbound_email",
            "support_triage",
            "fintech_fraud",
            "code_security",
            "high_cardinality_255",
        ],
    )
    def test_preset_loads_and_validates(self, name):
        preset = load_preset(name)
        assert preset["id"] == name
        assert "title" in preset
        assert "description" in preset
        assert "context" in preset
        schema = StructuredSchema(preset["schema"])
        assert len(schema.fields) > 0


class TestReadmeFieldCounts:
    """The README bundled-presets table field counts must match the actual
    preset JSON field counts (guard against drift)."""

    def test_readme_counts_match_presets(self):
        import re
        from pathlib import Path

        readme = Path(__file__).resolve().parent.parent / "README.md"
        text = readme.read_text(encoding="utf-8")
        # Parse the bundled-presets table rows: | `name` | N | desc |
        rows = re.findall(r"\| `(\w+)` \| (\d+) \|", text)
        counts = {name: int(n) for name, n in rows}
        for name, expected in counts.items():
            preset = load_preset(name)
            schema = StructuredSchema(preset["schema"])
            actual = len(schema.fields)
            assert actual == expected, (
                f"README says {name} has {expected} fields but the preset has {actual}"
            )
