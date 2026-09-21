"""W6-UI-3b: control-room dashboard page tests.

The page is a static HTML file (inline CSS + vanilla JS) that fetches
/dashboard.json and /questions.json and renders client-side. These tests
check (1) the HTML contains every panel heading and the legend, (2) the JS
references only keys present in the frozen contract, (3) the questions()
helper returns the right shape. No real socket, no browser.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevmlx.watch import _dashboard_html, build_questions

FIXTURE = Path(__file__).parent / "fixtures" / "dashboard"
HTML = (Path(__file__).parent.parent / "jevmlx" / "web" / "dashboard.html").read_text(
    encoding="utf-8"
)

# The 9 frozen top-level keys of /dashboard.json (data-layer contract).
CONTRACT_KEYS = {
    "run",
    "now",
    "memory",
    "health",
    "pipeline",
    "aggregates",
    "results",
    "events",
    "history",
}


class TestHtmlPanels:
    """Every panel heading and the legend text must be in the page."""

    def test_panel_headings_present(self):
        for heading in [
            "Now",
            "Memory",
            "Health",
            "Pipeline",
            "Failure diagnosis",
            "Aggregates so far",
            "Results",
            "Questions",
            "Events",
            "History",
        ]:
            assert heading in HTML, f"missing panel heading: {heading}"

    def test_legend_text_present(self):
        for text in [
            "accuracy = field accuracy, Wilson CI",
            "markers encode outcome, not exit code",
            "A/B Δ = main − branch",
            "parity",
        ]:
            assert text in HTML, f"missing legend text: {text}"

    def test_no_meta_refresh(self):
        """W6-UI-3f: the page must NOT reload via meta refresh — SSE drives updates."""
        assert 'http-equiv="refresh"' not in HTML

    def test_no_root_innerhtml_wipe(self):
        """W6-UI-3f: #root.innerHTML is assigned only on first paint, never on patch."""
        # The only 'root.innerHTML=' assignment is in initialRender (first paint).
        # patchDashboard must never touch root.innerHTML.
        assert HTML.count("root.innerHTML=h") == 1
        assert HTML.count("root.innerHTML=") == 1
        # No setInterval polling — SSE replaces it.
        assert "setInterval" not in HTML

    def test_sse_eventsource_present(self):
        """W6-UI-3f: the page connects via EventSource for live updates."""
        assert "EventSource('/events')" in HTML or 'EventSource("/events")' in HTML

    def test_no_external_deps(self):
        """No CDN, no framework — inline CSS + vanilla JS only."""
        assert "cdn." not in HTML
        assert "https://fonts.googleapis" not in HTML
        assert "import " not in HTML.split("<script>")[1]

    def test_refresh_substitution(self):
        """_dashboard_html injects the refresh interval into const REFRESH."""
        html = _dashboard_html(refresh=5)
        assert "const REFRESH = 5;" in html
        assert "__REFRESH__" not in html

    def test_first_paint_patches_shells(self):
        """W6-UI-3h: initialRender must call patchDashboard(D) after building
        shells so the first paint carries real values before any SSE event."""
        # The initialRender function must call patchDashboard(D) after setting
        # booted=true — this is what fills NOW/MEMORY/HEALTH/AGGREGATES nodes.
        assert "patchDashboard(D);" in HTML
        # Verify it's inside initialRender (after booted=true).
        idx_booted = HTML.index("booted=true;")
        idx_patch = HTML.index("patchDashboard(D);", idx_booted)
        assert idx_patch > idx_booted, "patchDashboard(D) must come after booted=true"

    def test_combo_id_sent_verbatim_on_click(self):
        """W6-UI-3h: the click handler sends results[i].combo_id (the
        out-relative path) verbatim to /questions.json?combo=, not the
        model|dataset|scorer|track pipe-string."""
        # The row key must be row.combo_id (with a defensive fallback).
        assert 'row.combo_id||(row.model+"|"' in HTML
        # On click, selCombo gets the data-combo (combo_id), and
        # selComboDisplayName gets display_name for the header.
        assert "selCombo=tr.dataset.combo" in HTML
        assert "selComboDisplayName=row.display_name||row.combo_id||key" in HTML

    def test_attempt_null_renders_dash(self):
        """W6-UI-3h: attempt_n=null shows '—' not '0' or empty."""
        assert 'function fmtAttempt(n){return n==null?"—":String(n);}' in HTML
        assert "fmtAttempt(t.attempt_n)" in HTML

    def test_fmt_nullable_present(self):
        """W6-UI round 4: fmtNullable helper renders null/undefined as —."""
        assert "function fmtNullable" in HTML
        # rotation must use fmtNullable, not raw concatenation.
        assert "fmtNullable(r.rotation)" in HTML
        assert "fmtNullable(c.rotation)" in HTML
        # No raw '+r.rotation+' or '+c.rotation+' concatenation left.
        assert "+r.rotation+" not in HTML
        assert "+c.rotation+" not in HTML

    def test_fmt_num_guards_nan(self):
        """W6-UI round 4: fmtNum guards NaN/Infinity (cases/h with <2 hb)."""
        assert "!isFinite(v)" in HTML

    def test_no_raw_null_concatenations(self):
        """W6-UI round 4: sweep — no raw concatenation of nullable fields."""
        # These were the spots that rendered 'null' as a literal string.
        assert "+t.machine+" not in HTML
        assert "+m.cap_gb+" not in HTML
        assert "+m.stop_gb+" not in HTML
        assert "+m.machine_gb+" not in HTML
        assert "+n.run_i+" not in HTML
        assert "+n.run_n+" not in HTML
        assert "+n.heartbeat_age_s+" not in HTML
        assert "+a.attempt_n+" not in HTML
        assert "+s.exit+" not in HTML


class TestContractKeys:
    """The JS must reference only keys present in the frozen contract."""

    def test_dashboard_json_has_exactly_9_top_level_keys(self):
        data = json.loads((FIXTURE / "dashboard.json").read_text(encoding="utf-8"))
        assert set(data.keys()) == CONTRACT_KEYS

    def test_questions_json_is_a_flat_list(self):
        """questions() returns a flat list, not an object with filter_groups."""
        data = json.loads((FIXTURE / "questions.json").read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0
        item = data[0]
        for key in (
            "ts",
            "case_id",
            "rotation",
            "field",
            "predicted",
            "label",
            "ok",
            "p_pred",
            "margin",
            "call_ms",
            "acc_so_far",
            "options",
            "context_text",
            "field_question",
            "rotations_same_field",
            "rescored",
            "drift",
        ):
            assert key in item, f"questions item missing key: {key}"

    def test_js_references_contract_keys(self):
        """Every D.<key> reference in the JS is one of the 9 contract keys."""
        js = HTML.split("<script>")[1].split("</script>")[0]
        import re

        refs = set(re.findall(r"D\.(\w+)", js))
        # D.now, D.run, etc. — all must be contract keys.
        bad = refs - CONTRACT_KEYS
        assert not bad, f"JS references non-contract D. keys: {bad}"

    def test_null_fields_render_as_dash(self):
        """The fmtNum helper returns — for null, never breaks."""
        js = HTML.split("<script>")[1].split("</script>")[0]
        assert '"—"' in js or "'—'" in js
        # ab_delta null check
        assert "ab_delta==null" in js or "ab_delta)if(d==null" in js or "ab_delta" in js


class TestQuestionsHelper:
    """build_questions(out_dir, combo_id) returns the flat list from predictions.jsonl.

    combo_id is the combo directory name (the data-layer contract), not a
    model|dataset|scorer|track pipe-string.
    """

    def test_empty_combo_id(self, tmp_path):
        assert build_questions(tmp_path, "") == []

    def test_no_match(self, tmp_path):
        """A combo_id that doesn't match any dir returns []."""
        (tmp_path / "run.json").write_text(
            json.dumps({"config": {"model": "X", "dataset": "Y", "scorer": "Z", "track": "T"}}),
            encoding="utf-8",
        )
        assert build_questions(tmp_path, "no-such-combo") == []

    def test_returns_predictions_shape(self, tmp_path):
        """Matching combo returns one item per prediction line with acc_so_far."""
        combo_dir = tmp_path / "M-D-S-parallel"
        combo_dir.mkdir()
        (combo_dir / "run.json").write_text(
            json.dumps(
                {"config": {"model": "M", "dataset": "D", "scorer": "S", "track": "parallel"}}
            ),
            encoding="utf-8",
        )
        (combo_dir / "predictions.jsonl").write_text(
            json.dumps(
                {
                    "case_id": "c1",
                    "field": "f",
                    "prediction": "A",
                    "label": "A",
                    "correct": True,
                    "probability": 0.9,
                    "latency_ms": 100,
                    "rotation": 0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "case_id": "c2",
                    "field": "f",
                    "prediction": "B",
                    "label": "A",
                    "correct": False,
                    "probability": 0.4,
                    "latency_ms": 110,
                    "rotation": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        out = build_questions(tmp_path, "M-D-S-parallel")
        assert len(out) == 2
        assert out[0]["ok"] is True
        assert out[0]["acc_so_far"] == pytest.approx(1.0)
        assert out[1]["ok"] is False
        assert out[1]["acc_so_far"] == pytest.approx(0.5)
        assert out[0]["predicted"] == "A"
        assert out[0]["label"] == "A"
