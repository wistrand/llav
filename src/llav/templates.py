"""Per-template defaults, chosen from the rendered chat template rather than from a model name.

Most templates end where the assistant's answer begins, so the next token can be an answer letter. Some end
earlier. Muse Glimmer's ends at `<|start|>assistant`, and the model first writes a recipient header
(` to=user<|message|>`); OpenAI's gpt-oss models use the same Harmony format and first name a channel
(`<|channel|>final<|message|>`, the final answer rather than the analysis channel, so no reasoning is
generated). Without the header no letter is among the likely next tokens and every request fails. The
header is template text, appended to the template tail and tokenized with special-token parsing.

`low_mass` is the candidate mass below which an answer is flagged as one the options do not fit. It is a
display cue measured per model family: Qwen3.5-4B puts about 0.999 on the letters and 0.57 on a question no
option fits; Muse Glimmer answers correctly at a median of 0.62, and answers below 0.35 were right 35% of the
time against 86% above (agent_docs/comparisons.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    assistant_prefix: str = ""
    low_mass: float = 0.9


DEFAULT = Profile("default")

# (profile, text the rendered template must end with, text it must contain)
_KNOWN = [
    (Profile("muse-glimmer", " to=user<|message|>", 0.35), "<|start|>assistant", '# Valid recipients: "self", "user".'),
    (Profile("gpt-oss", "<|channel|>final<|message|>"), "<|start|>assistant",
     "# Valid channels: analysis, commentary, final."),
]


def detect(rendered: str) -> Profile:
    """The profile for a rendered chat prompt that ends with the assistant's generation prompt."""
    for profile, ending, marker in _KNOWN:
        if rendered.endswith(ending) and marker in rendered:
            return profile
    return DEFAULT
