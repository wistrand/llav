"""Optional temperature calibration: one temperature per question type, fitted on a caller's labelled data.

A readout's option probabilities are a softmax over the declared labels' logits. Dividing those logits by a
temperature T > 1 softens an overconfident model, T < 1 sharpens an underconfident one, and the ranking of
the options, and with it every answer, stays the same. `softmax(log p / T)` equals `softmax(logits / T)`,
because the per-question constant cancels, so calibration applies to finished probabilities and neither the
engine nor the native helper needs to know about it.

The file is written by `scripts/evaluate.py fit` and names the model file's SHA-256 and the prompt version it
was fitted for; llav refuses it for anything else. Calibration holds only for data like what it was fitted on,
so llav ships no calibration file.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .prompt import PROMPT_VERSION

FORMAT = "llav-calibration-1"
TYPES = ("noul", "choice", "score")
# log(0) is undefined; a probability this small is already 0 at any temperature a fit can produce.
TINY = 1e-300


class CalibrationError(Exception):
    """The calibration file is malformed or was fitted for another model or prompt."""


def apply(probabilities: list[float], temperature: float) -> list[float]:
    if temperature == 1:
        return probabilities
    logs = [math.log(max(p, TINY)) / temperature for p in probabilities]
    top = max(logs)
    weights = [math.exp(value - top) for value in logs]
    total = sum(weights)
    return [weight / total for weight in weights]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


class Calibration:
    def __init__(self, temperatures: dict[str, float], ident: str, model_file: str, model_sha256: str):
        self.temperatures = temperatures
        self.id = ident
        self.model_file = model_file
        self.model_sha256 = model_sha256

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        raw = Path(path).read_bytes()
        try:
            data = json.loads(raw)
        except ValueError as error:
            raise CalibrationError(f"{path} is not JSON: {error}") from error
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise CalibrationError(f"{path} is not a {FORMAT} file")
        if data.get("prompt_version") != PROMPT_VERSION:
            raise CalibrationError(f"{path} was fitted for prompt {data.get('prompt_version')!r}; "
                                   f"this llav uses {PROMPT_VERSION!r}")
        model = data.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("sha256"), str) or not model.get("file"):
            raise CalibrationError(f"{path} does not name the model it was fitted for")
        temperatures = data.get("temperature")
        if not isinstance(temperatures, dict) or not temperatures or set(temperatures) - set(TYPES):
            raise CalibrationError(f"{path}: temperature must map one or more of {TYPES} to numbers")
        for kind, value in temperatures.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.01 <= value <= 100:
                raise CalibrationError(f"{path}: temperature for {kind!r} must be a number from 0.01 to 100")
        # The id names the exact file, so a caller can tell which calibration produced an answer.
        ident = hashlib.sha256(raw).hexdigest()[:12]
        return cls({kind: float(value) for kind, value in temperatures.items()}, ident, str(model["file"]),
                   model["sha256"].lower())

    def check_model(self, gguf: Path | None, served_file: str) -> str | None:
        """Refuse a file fitted for another model. Returns a warning when the model cannot be verified."""
        if gguf is None:
            # An external llama-server's weights are not readable from here; only the file name can be compared.
            if Path(served_file).name != Path(self.model_file).name:
                raise CalibrationError(f"calibration was fitted for {self.model_file}, but llama-server serves "
                                       f"{served_file}")
            return f"calibration: model contents not verified (external llama-server serves {served_file})"
        actual = file_sha256(gguf)
        if actual != self.model_sha256:
            raise CalibrationError(f"calibration was fitted for {self.model_file} (sha256 {self.model_sha256[:12]}), "
                                   f"not {gguf.name} (sha256 {actual[:12]})")
        return None

    def apply(self, kind: str, probabilities: list[float]) -> list[float]:
        return apply(probabilities, self.temperatures.get(kind, 1.0))
