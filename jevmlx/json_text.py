"""Canonical JSON serialization (W5-A, finding 39).

One serializer for every prompt and candidate path: ``ensure_ascii=False``.
The stdlib default escaped non-ASCII labels ("\u00e9") in candidate
compilation while prompts showed the real characters, so the scorer judged
tokenizations the model never saw. Import this everywhere; do not call
``json.dumps`` directly for prompt or candidate text.
"""

from __future__ import annotations

import json


def json_text(value: object) -> str:
    """Canonical JSON text for prompts and scored candidates."""
    return json.dumps(value, ensure_ascii=False)
