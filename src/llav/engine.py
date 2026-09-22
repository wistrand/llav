"""Option-logit readout through llama-server.

One forward pass per question reads the next-token log-probabilities of the answer labels and
softmaxes them over the declared options; nothing is generated.

Several questions about one state share its prefill. llama-server cannot roll a recurrent model
(such as Qwen3.5) back to a shared prefix without context checkpoints, and saving a checkpoint copies
the recurrent state off the GPU on every request. So the state prefix is evaluated once, saved to a
slot file, and restored before each question; every question then extends the cached tokens exactly.
llama-server must run with --ctx-checkpoints 0 and --slot-save-path.
"""

from __future__ import annotations

import http.client
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid

from .prompt import LABELS, evidence_opening, messages
from .questions import Question

TEMPLATE_KWARGS = {"enable_thinking": False}


class EngineError(Exception):
    """The backend failed or returned something unusable; reported as HTTP 500."""


class Overloaded(Exception):
    """No slot became free in time; reported as HTTP 529."""


class ContextTooLong(Exception):
    """A question's prompt does not fit a slot; reported as HTTP 422."""

    def __init__(self, key: str, tokens: int, limit: int):
        super().__init__(f"prompt for question {key!r} has {tokens} tokens; the slot limit is {limit}")
        self.key = key


class LlamaClient:
    """Minimal JSON client for llama-server."""

    def __init__(self, url: str, timeout: float = 600):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, path: str, request, timeout: float) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            with error:
                detail = error.read().decode(errors="replace")[:500]
            raise EngineError(f"llama-server {path} returned {error.code}: {detail}") from error
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise EngineError(f"llama-server {path} failed: {error}") from error
        if not isinstance(result, dict):
            raise EngineError(f"llama-server {path} returned a non-object response")
        return result

    def _field(self, path: str, result: dict, name: str):
        if name not in result:
            raise EngineError(f"llama-server {path} response has no {name!r}")
        return result[name]

    def get(self, path: str) -> dict:
        return self._call(path, self.url + path, 30)

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        return self._call(path, request, self.timeout)

    def render(self, turns: list[dict]) -> str:
        body = {"messages": turns, "chat_template_kwargs": TEMPLATE_KWARGS}
        return self._field("/apply-template", self.post("/apply-template", body), "prompt")

    def tokenize(self, text: str, special: bool = True) -> list[int]:
        """`special=False` keeps control-token text such as <|im_end|> as plain text."""
        body = {"content": text, "add_special": False, "parse_special": special}
        return self._field("/tokenize", self.post("/tokenize", body), "tokens")


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    weights = [math.exp(value - top) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


class Engine:
    def __init__(self, client: LlamaClient, slots: int, slot_ctx: int, slot_dir: Path,
                 queue_timeout: float = 30, n_probs: int = 128):
        self.client = client
        self.slot_ctx = slot_ctx
        self.slot_dir = Path(slot_dir)
        self.queue_timeout = queue_timeout
        self.n_probs = n_probs
        self.free = queue.Queue()
        for slot in range(slots):
            self.free.put(slot)
        self.label_ids = []
        for label in LABELS:
            encoded = client.tokenize(label)
            if len(encoded) != 1:
                raise EngineError(f"Answer label {label!r} is not a single token for this model")
            self.label_ids.append(encoded[0])
        if len(set(self.label_ids)) != len(LABELS):
            raise EngineError("Answer label tokens collide")
        self._boundary = {}
        self._boundary_lock = threading.Lock()

    def _boundary_ok(self, tail: str, count: int) -> bool:
        """Appending a label to the template tail must add exactly its token. Cached per tail."""
        with self._boundary_lock:
            ok = self._boundary.get(tail)
        if ok is None:
            base = self.client.tokenize(tail)
            ok = [self.client.tokenize(tail + label) == base + [token]
                  for label, token in zip(LABELS, self.label_ids)]
            with self._boundary_lock:
                self._boundary[tail] = ok
        return all(ok[:count])

    def _split(self, turns: list[dict]) -> tuple[str, str, str]:
        """Rendered prompt as (template head, user payload, template tail)."""
        prompt = self.client.render(turns)
        payload = turns[-1]["content"]
        if prompt.count(payload) != 1:
            raise EngineError("Cannot locate the user payload in the chat template")
        start = prompt.index(payload)
        return prompt[:start], payload, prompt[start + len(payload):]

    def encode(self, state, question: Question) -> list[int]:
        # Caller text is tokenized without special-token parsing so it cannot forge chat turns. The
        # payload sits between a newline and a control token, both token boundaries, so the split
        # matches whole-prompt tokenization whenever the text has no control tokens.
        head, payload, tail = self._split(messages(state, question.instructions, list(question.descriptions)))
        ids = self.client.tokenize(head) + self.client.tokenize(payload, special=False) + self.client.tokenize(tail)
        if len(ids) >= self.slot_ctx:
            raise ContextTooLong(question.key, len(ids), self.slot_ctx - 1)
        if not self._boundary_ok(tail, len(question.descriptions)):
            raise EngineError("Answer boundary changes tokenization for this chat template")
        return ids

    def state_prefix(self, state) -> list[int]:
        """Tokens up to the evidence value, minus the last token (it can merge with what follows)."""
        head, _, _ = self._split(messages(state, "prefix boundary placeholder", ["Yes", "No"]))
        return self.client.tokenize(head) + self.client.tokenize(evidence_opening(state), special=False)[:-1]

    def _readout(self, ids: list[int], count: int, slot: int, cache: bool) -> tuple[list[float], int]:
        response = self.client.post("/completion", {
            "prompt": ids, "n_predict": 1, "temperature": -1, "n_probs": self.n_probs,
            "cache_prompt": cache, "id_slot": slot,
        })
        # With temperature < 0, n_probs are a plain softmax over the raw logits, unaffected by samplers.
        entries = (response.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
        top = {entry["id"]: entry["logprob"] for entry in entries}
        found = [top.get(token) for token in self.label_ids[:count]]
        if not any(value is not None for value in found):
            raise EngineError("No answer label among the returned token probabilities")
        # A label outside the top n_probs has at most the smallest returned logprob; use that bound.
        floor = min(top.values())
        logprobs = [floor if value is None else value for value in found]
        return softmax(logprobs), int((response.get("timings") or {}).get("prompt_n") or 0)

    def evaluate(self, state, questions: list[Question]) -> tuple[list[list[float]], dict, dict]:
        """Score every question against `state`; returns per-question probabilities, usage, and timing."""
        encoded = [self.encode(state, question) for question in questions]
        counts = [len(question.descriptions) for question in questions]
        prefix = self.state_prefix(state) if len(encoded) > 1 else []
        try:
            slot = self.free.get(timeout=self.queue_timeout)
        except queue.Empty:
            raise Overloaded("All llama-server slots are busy") from None
        try:
            return self._evaluate_on(slot, prefix, encoded, counts)
        finally:
            self.free.put(slot)

    def _evaluate_on(self, slot: int, prefix: list[int], encoded: list[list[int]], counts: list[int]):
        started = time.perf_counter()
        computed = 0
        results = []
        shared = bool(prefix) and all(ids[: len(prefix)] == prefix and len(ids) > len(prefix) for ids in encoded)
        if not shared:
            for ids, count in zip(encoded, counts):
                probabilities, n = self._readout(ids, count, slot, cache=False)
                results.append(probabilities)
                computed += n
        else:
            name = f"llav-{uuid.uuid4().hex}.bin"
            self.client.post(f"/slots/{slot}?action=erase", {})
            primed = self.client.post("/completion", {
                "prompt": prefix, "n_predict": 0, "cache_prompt": True, "id_slot": slot,
            })
            computed += int((primed.get("timings") or {}).get("prompt_n") or len(prefix))
            self.client.post(f"/slots/{slot}?action=save", {"filename": name})
            try:
                for ids, count in zip(encoded, counts):
                    self.client.post(f"/slots/{slot}?action=restore", {"filename": name})
                    probabilities, n = self._readout(ids, count, slot, cache=True)
                    results.append(probabilities)
                    computed += n
            finally:
                try:
                    os.remove(self.slot_dir / name)
                except OSError:
                    pass
        usage = {"input_tokens": computed, "output_tokens": 0}
        meta = {"shared_state_tokens": len(prefix) if shared else 0, "seconds": time.perf_counter() - started}
        return results, usage, meta
