"""Decision prompt: SemIf's direct-options-v1 format, extended from 16 to 26 answer labels.

For 2-16 options the rendered prompt is byte-identical to SemIf's torch scorer, which is what the
llama.cpp readout was validated against (768/777 argmax agreement on SemIf's shape777 fixture).
"""

from __future__ import annotations

import json

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT_VERSION = "direct-options-v1"
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def messages(state, criterion, descriptions: list) -> list[dict]:
    """Chat turns for one decision; `state`, `criterion`, and descriptions may be any JSON value."""
    if not 2 <= len(descriptions) <= len(LABELS):
        raise ValueError(f"A decision needs 2-{len(LABELS)} options")
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LABELS[index], "description": value} for index, value in enumerate(descriptions)],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def evidence_opening(state) -> str:
    """The serialized payload up to and including the evidence value, without the closing brace."""
    return json.dumps({"evidence": state}, ensure_ascii=False)[:-1]
