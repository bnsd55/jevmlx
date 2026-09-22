"""Tests for jevmlx.baseline: prompt building, parsing, HTTP call. No network."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jevmlx.baseline import (
    baseline_decide,
    build_baseline_messages,
    call_chat_completions,
    parse_baseline_output,
)
from jevmlx.http import ChatCompletionsError
from jevmlx.schema import StructuredSchema

SCHEMA = StructuredSchema(
    {
        "action": {
            "type": "enum",
            "choices": ["APPROVE", "REJECT", "ESCALATE"],
            "description": "Decision on the transaction.",
        },
        "amount_valid": {"type": "boolean", "description": "Is the amount plausible?"},
        "tags": {"type": "multi", "choices": ["fraud", "velocity"], "description": "Risk tags."},
    }
)


class TestBuildBaselineMessages:
    def test_exact_text_for_three_field_schema(self):
        messages = build_baseline_messages(SCHEMA, "Wire of $9,000 to a new payee.")
        assert messages == [
            {
                "role": "user",
                "content": (
                    "You will be given a context and a list of fields to decide.\n"
                    "Output ONLY a JSON object — no markdown, no explanation —"
                    " with exactly these keys:\n"
                    '- "action" (one of exactly: "APPROVE", "REJECT", "ESCALATE")'
                    " — Decision on the transaction.\n"
                    '- "amount_valid" (boolean: true or false) — Is the amount plausible?\n'
                    '- "tags" (an array, possibly empty, of allowed strings: "fraud", "velocity")'
                    " — Risk tags.\n"
                    "\n"
                    "Context:\n"
                    "Wire of $9,000 to a new payee."
                ),
            }
        ]
        assert len(messages) == 1 and messages[0]["role"] == "user"


class TestParseBaselineOutput:
    def test_valid_output(self):
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": true, "tags": ["fraud"]}', SCHEMA
        )
        assert errors == []
        assert strict == {"action": "APPROVE", "amount_valid": True, "tags": ["fraud"]}
        assert salvage == strict

    def test_object_followed_only_by_whitespace_is_valid(self):
        _, _, errors = parse_baseline_output(
            '{"action": "REJECT", "amount_valid": false, "tags": []}\n\n', SCHEMA
        )
        assert errors == []

    def test_trailing_text_after_object_is_an_error(self):
        strict, salvage, errors = parse_baseline_output(
            '{"action": "REJECT", "amount_valid": false, "tags": []} Hope that helps!', SCHEMA
        )
        assert errors == ["trailing text after JSON object"]
        assert all(value is None for value in strict.values())
        # salvage still keeps the individually parseable fields
        assert salvage == {"action": "REJECT", "amount_valid": False, "tags": []}

    def test_json_inside_prose_is_parsed_then_salvage_only(self):
        """Leading prose before the object is fine (raw_decode at first '{');
        trailing text after it is not. Either way strict stays all-None."""
        strict, salvage, errors = parse_baseline_output(
            'Sure! Here is the result:\n{"action": "REJECT", "amount_valid": false, "tags": []}\n',
            SCHEMA,
        )
        assert errors == []
        assert strict == {"action": "REJECT", "amount_valid": False, "tags": []}
        assert salvage == strict

    def test_malformed_json_reports_error_and_never_raises(self):
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": tru}', SCHEMA
        )
        assert strict == {"action": None, "amount_valid": None, "tags": None}
        assert salvage == strict
        assert len(errors) == 1 and errors[0].startswith("invalid JSON")

    def test_no_braces_at_all(self):
        strict, salvage, errors = parse_baseline_output("I cannot help with that.", SCHEMA)
        assert set(strict) == {"action", "amount_valid", "tags"}
        assert salvage == strict
        assert errors == ["no JSON object found in output"]

    def test_wrong_enum_value(self):
        strict, salvage, errors = parse_baseline_output(
            '{"action": "MAYBE", "amount_valid": true, "tags": []}', SCHEMA
        )
        assert strict["action"] is None and salvage["action"] is None
        assert any("invalid value for action" in e and "MAYBE" in e for e in errors)

    def test_numeric_scalar_coerced_for_string_digit_enum(self):
        """int 2 for choices ('0','1','2','3') is accepted as '2' — a baseline
        must not lose a semantically-correct case to a type-only mismatch."""
        schema = StructuredSchema(
            {"level": {"type": "enum", "choices": ["0", "1", "2", "3"], "description": "Level"}}
        )
        strict, salvage, errors = parse_baseline_output('{"level": 2}', schema)
        assert errors == []
        assert strict["level"] == "2"
        assert salvage["level"] == "2"

    def test_numeric_scalar_not_in_choices_still_errors(self):
        """A numeric scalar that str-coerces to a non-choice still errors
        (str(2.5)=='2.5' is not in ('0','1','2','3'))."""
        schema = StructuredSchema(
            {"level": {"type": "enum", "choices": ["0", "1", "2", "3"], "description": "Level"}}
        )
        strict, salvage, errors = parse_baseline_output('{"level": 2.5}', schema)
        assert strict["level"] is None
        assert salvage["level"] is None
        assert any("invalid value for level" in e and "2.5" in e for e in errors)

    def test_wrong_string_enum_still_errors_after_coercion(self):
        """A wrong string ('MAYBE') still errors — coercion only rescues
        numeric scalars, never wrong strings."""
        schema = StructuredSchema(
            {"action": {"type": "enum", "choices": ["APPROVE", "REJECT"], "description": "Act"}}
        )
        strict, salvage, errors = parse_baseline_output('{"action": "MAYBE"}', schema)
        assert strict["action"] is None
        assert salvage["action"] is None
        assert any("invalid value for action" in e and "MAYBE" in e for e in errors)

    def test_extra_keys_are_errors(self):
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": true, "tags": [], "notes": "hi"}', SCHEMA
        )
        assert errors == ["extra key: notes"]
        assert all(value is None for value in strict.values())  # strict: schema fields only
        # per-field values were individually fine, so salvage keeps them
        assert salvage == {"action": "APPROVE", "amount_valid": True, "tags": []}

    def test_missing_key(self):
        strict, salvage, errors = parse_baseline_output('{"action": "APPROVE"}', SCHEMA)
        assert strict["amount_valid"] is None and strict["tags"] is None
        assert salvage["amount_valid"] is None and salvage["tags"] is None
        assert sorted(errors) == ["missing key: amount_valid", "missing key: tags"]

    def test_boolean_wrong_type(self):
        _, _, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": "yes", "tags": []}', SCHEMA
        )
        assert any("wrong type for amount_valid" in e for e in errors)

    def test_boolean_string_true_coerced(self):
        """'true' (string) is accepted as bool True — a baseline must not
        lose a semantically-correct case to a type-only mismatch."""
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": "true", "tags": []}', SCHEMA
        )
        assert errors == []
        assert strict["amount_valid"] is True
        assert salvage["amount_valid"] is True

    def test_boolean_string_false_coerced(self):
        """'False' (string, mixed-case) is accepted as bool False."""
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": "False", "tags": []}', SCHEMA
        )
        assert errors == []
        assert strict["amount_valid"] is False
        assert salvage["amount_valid"] is False

    def test_boolean_int_one_coerced(self):
        """int 1 is accepted as bool True."""
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": 1, "tags": []}', SCHEMA
        )
        assert errors == []
        assert strict["amount_valid"] is True
        assert salvage["amount_valid"] is True

    def test_boolean_int_zero_coerced(self):
        """int 0 is accepted as bool False."""
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": 0, "tags": []}', SCHEMA
        )
        assert errors == []
        assert strict["amount_valid"] is False
        assert salvage["amount_valid"] is False

    def test_boolean_yes_still_errors(self):
        """'yes' is not a recognized boolean spelling and still errors
        with the same message as before the coercion was added."""
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": "yes", "tags": []}', SCHEMA
        )
        assert strict["amount_valid"] is None
        assert salvage["amount_valid"] is None
        assert any("wrong type for amount_valid" in e for e in errors)

    def test_boolean_none_still_errors(self):
        """None is not a boolean and still errors."""
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": null, "tags": []}', SCHEMA
        )
        assert strict["amount_valid"] is None
        assert salvage["amount_valid"] is None
        assert any("wrong type for amount_valid" in e for e in errors)

    def test_boolean_string_true_stripped_and_case_insensitive(self):
        """'  TRUE  ' (stripped, upper-case) is accepted as bool True."""
        strict, salvage, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": "  TRUE  ", "tags": []}', SCHEMA
        )
        assert errors == []
        assert strict["amount_valid"] is True

    def test_multi_item_not_in_choices(self):
        _, _, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": true, "tags": ["fraud", "nope"]}', SCHEMA
        )
        assert any("invalid items for tags" in e and "nope" in e for e in errors)

    def test_multi_duplicate_items_are_errors(self):
        _, _, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": true, "tags": ["fraud", "fraud"]}', SCHEMA
        )
        assert any("duplicate items for tags" in e and "fraud" in e for e in errors)

    def test_multi_empty_array_is_allowed(self):
        strict, _, errors = parse_baseline_output(
            '{"action": "APPROVE", "amount_valid": true, "tags": []}', SCHEMA
        )
        assert errors == [] and strict["tags"] == []


class _CannedHandler(BaseHTTPRequestHandler):
    """Returns a canned OpenAI-style body; records the last request."""

    last_headers: dict = {}
    last_body: dict = {}
    status = 200
    payload = {"choices": [{"message": {"content": '{"action": "APPROVE"}'}}]}

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        type(self).last_headers = dict(self.headers)
        type(self).last_body = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps(type(self).payload).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output clean
        pass


class TestCallChatCompletions:
    @pytest.fixture()
    def server(self):
        httpd = HTTPServer(("127.0.0.1", 0), _CannedHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{httpd.server_port}", httpd
        httpd.shutdown()
        httpd.server_close()

    def test_returns_assistant_content(self, server):
        base_url, _ = server
        content = call_chat_completions(
            base_url, "glm-5.3", [{"role": "user", "content": "x"}], api_key=None
        )
        assert content == '{"action": "APPROVE"}'

    def test_text_mode_is_default_and_sends_no_response_format(self, server):
        base_url, _ = server
        call_chat_completions(base_url, "glm-5.3", [{"role": "user", "content": "x"}], api_key=None)
        assert "response_format" not in _CannedHandler.last_body

    def test_json_mode_sends_response_format(self, server):
        base_url, _ = server
        call_chat_completions(
            base_url,
            "glm-5.3",
            [{"role": "user", "content": "x"}],
            api_key=None,
            mode="json",
        )
        assert _CannedHandler.last_body["response_format"] == {"type": "json_object"}

    def test_invalid_mode_raises_value_error(self, server):
        base_url, _ = server
        with pytest.raises(ValueError, match="mode must be"):
            call_chat_completions(
                base_url,
                "glm-5.3",
                [{"role": "user", "content": "x"}],
                api_key=None,
                mode="yaml",
            )

    def test_bearer_header_and_payload_shape(self, server):
        base_url, _ = server
        call_chat_completions(
            base_url + "/", "glm-5.3", [{"role": "user", "content": "x"}], api_key="secret-key"
        )
        assert _CannedHandler.last_headers.get("Authorization") == "Bearer secret-key"
        assert _CannedHandler.last_body["model"] == "glm-5.3"
        assert _CannedHandler.last_body["temperature"] == 0.0

    def test_no_bearer_header_without_api_key(self, server):
        base_url, _ = server
        call_chat_completions(base_url, "glm-5.3", [{"role": "user", "content": "x"}], api_key=None)
        assert "Authorization" not in _CannedHandler.last_headers

    def test_500_raises_baseline_error_with_status_and_snippet(self, server):
        base_url, _ = server
        _CannedHandler.status = 500
        _CannedHandler.payload = {"error": "x" * 500}
        try:
            with pytest.raises(ChatCompletionsError) as excinfo:
                call_chat_completions(
                    base_url, "glm-5.3", [{"role": "user", "content": "x"}], api_key=None
                )
            assert excinfo.value.status == 500
            assert len(str(excinfo.value)) < 260  # first 200 chars of body + prefix
        finally:
            _CannedHandler.status = 200
            _CannedHandler.payload = {"choices": [{"message": {"content": "{}"}}]}

    def test_error_is_runtimeerror(self):
        assert issubclass(ChatCompletionsError, RuntimeError)


class TestBaselineDecide:
    def test_end_to_end_against_canned_server(self):
        httpd = HTTPServer(("127.0.0.1", 0), _CannedHandler)
        _CannedHandler.payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"action": "APPROVE", "amount_valid": true, "tags": ["fraud"]}'
                    }
                }
            ]
        }
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            result = baseline_decide(
                f"http://127.0.0.1:{httpd.server_port}",
                "glm-5.3",
                None,
                SCHEMA,
                "wire transfer context",
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
        assert result["strict_valid"] is True
        assert result["schema_valid"] is True
        assert result["salvage_values"] == result["values"]
        assert result["values"] == {
            "action": "APPROVE",
            "amount_valid": True,
            "tags": ["fraud"],
        }
        assert result["latency_ms"] >= 0.0 and isinstance(result["raw"], str)
