"""Option-logit readout through llama-server.

One forward pass per question reads the next-token log-probabilities of the answer labels and
softmaxes them over the declared options; nothing is generated.

Several questions about one state share its prefill. llama-server cannot roll a recurrent model
(such as Qwen3.5) back to a shared prefix without context checkpoints, and saving a checkpoint copies
the recurrent state off the GPU on every request. So the state prefix is evaluated once, saved to a
slot file, and restored before each question; every question then extends the cached tokens exactly.
llama-server must run with --ctx-checkpoints 0 and --slot-save-path, and with --swa-full, without which a
sliding-window model cannot extend a cached state.

Images and audio (with --mmproj) take a second path: llama-server accepts media only as a prompt string
plus `multimodal_data`, never as token ids, so `encode_media` builds strings around the same template head,
payload and tail, and the server tokenizes them. The readout, the slot files and the shared-state flow are
the same; the native helper is not used.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from .native import NativeError, NativeReadout
from .prompt import LABELS, evidence_opening, messages
from .questions import Question
from .templates import Profile, detect

TEMPLATE_KWARGS = {"enable_thinking": False}

# A single question is primed separately only above this many state tokens, where a later cache hit saves
# more than the extra pass costs.
STATE_CACHE_MIN_TOKENS = 256


class EngineError(Exception):
    """The backend failed or returned something unusable; reported as HTTP 500."""


class Overloaded(Exception):
    """No slot became free in time; reported as HTTP 529."""


class ContextTooLong(Exception):
    """A question's prompt does not fit a slot; reported as HTTP 422."""

    def __init__(self, key: str, tokens: int, limit: int):
        super().__init__(f"prompt for question {key!r} has {tokens} tokens; the slot limit is {limit}")
        self.key = key


class MediaRequestError(Exception):
    """A media request this server or this text cannot take; reported as HTTP 422 at `loc`."""

    def __init__(self, loc: list, message: str):
        super().__init__(message)
        self.loc = loc
        self.message = message


# llama-server's wording when a prompt, media tokens included, does not fit the slot, and when a media file
# is not one its decoders read (an image stb_image knows, or WAV, MP3 or FLAC).
_CONTEXT_EXCEEDED = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+)")
_MEDIA_UNREADABLE = "Failed to load image or audio file"


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

    def tokenize(self, text: str, special: bool = True, add_special: bool = False) -> list[int]:
        """`special=False` keeps control-token text such as <|im_end|> as plain text. `add_special` adds
        the vocabulary's BOS (and EOS) if it has one, as llama-server's own chat path does."""
        body = {"content": text, "add_special": add_special, "parse_special": special}
        return self._field("/tokenize", self.post("/tokenize", body), "tokens")

    def detokenize(self, tokens: list[int]) -> str:
        return self._field("/detokenize", self.post("/detokenize", {"tokens": tokens}), "content")


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    weights = [math.exp(value - top) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


class Engine:
    def __init__(self, client: LlamaClient, slots: int, slot_ctx: int, slot_dir: Path,
                 queue_timeout: float = 30, n_probs: int = 128, state_cache: int = 4,
                 native: NativeReadout | None = None, assistant_prefix: str | None = None):
        self.client = client
        self.slot_ctx = slot_ctx
        self.slot_dir = Path(slot_dir)
        self.queue_timeout = queue_timeout
        self.n_probs = n_probs
        self.state_cache = max(0, state_cache)
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
        self._template_parts = None
        self._template_lock = threading.Lock()
        self._run = uuid.uuid4().hex[:8]  # so stale files from an earlier run are never restored
        self._cached = OrderedDict()  # cache filename -> tokens in the file, least recently used first
        self._cache_lock = threading.Lock()
        # The template decides what, if anything, goes between its generation prompt and the answer letter.
        self.profile: Profile = detect(client.render(messages("a", "b", ["Yes", "No"])))
        self.assistant_prefix = self.profile.assistant_prefix if assistant_prefix is None else assistant_prefix
        self.trims = self._probe_trim()
        self._probe_labels()
        self.native = native  # opt-in fast path; a failure here falls back to llama-server for good
        self.native_error: str | None = None
        # The media marker is random per llama-server run, so it is read, never hardcoded, and a caller
        # cannot know it. The modalities are what the loaded projector supports, not what the model name
        # suggests: Qwen3.5's projector sees, Gemma 4's sees and hears.
        props = client.get("/props")
        self.media_marker = str(props.get("media_marker") or "")
        modalities = props.get("modalities") or {}
        self.modalities = {kind: bool(modalities.get(reported)) and bool(self.media_marker)
                           for kind, reported in (("image", "vision"), ("audio", "audio"))}

    def backend_status(self) -> dict:
        """What `/v1/models` reports about state reuse now, not at startup: the helper can be dropped later."""
        status = {"prefix_reuse": "native" if self.native else ("trim" if self.trims else "slot-file"),
                  "images": self.modalities["image"], "audio": self.modalities["audio"]}
        if self.native_error:
            status["native_error"] = self.native_error
        return status

    def _probe_trim(self) -> bool:
        """Can the backend roll its cache back to a shared prefix, or must a slot file restore it?

        Attention-only models trim the cache and re-evaluate just the new tokens. Recurrent and hybrid
        models (Qwen3.5) recompute the whole prompt and answer from a state that never rolled back, so
        they need the slot file. Probing beats a model list: it asks the backend what it actually does.
        """
        base = self.client.tokenize("probe " * 40, special=False)
        try:
            self.client.post("/slots/0?action=erase", {})
            for label in self.label_ids[:2]:
                answer = self.client.post("/completion", {
                    "prompt": base + [label], "n_predict": 1, "temperature": -1, "cache_prompt": True,
                    "id_slot": 0,
                })
            evaluated = int((answer.get("timings") or {}).get("prompt_n") or len(base))
            self.client.post("/slots/0?action=erase", {})
        except EngineError:
            return False  # a backend that cannot answer the probe gets the path that always works
        return evaluated <= len(base) // 2

    def _probe_labels(self) -> None:
        """Refuse to start when the model does not answer with a letter after this template.

        Otherwise every request would fail with "No answer label", which says nothing about the cause: a
        template that stops before the answer can begin, or a model that wants to start a sentence.
        """
        probe = Question("probe", "noul", "Is the sky blue?", ("true", "false"), ("Yes", "No"))
        try:
            ids = self.encode("The sky is blue.", probe)
        except ContextTooLong as error:
            raise EngineError(f"The per-slot context ({self.slot_ctx} tokens) cannot hold even a one-line "
                              f"question; raise --ctx. {error}") from error
        response = self.client.post("/completion", {
            "prompt": ids, "n_predict": 1, "temperature": -1,
            "n_probs": self.n_probs, "cache_prompt": False, "id_slot": 0,
        })
        entries = (response.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
        returned = {entry.get("id") for entry in entries}
        if not returned & set(self.label_ids[:2]):
            likely = ", ".join(repr(entry.get("token")) for entry in entries[:3])
            raise EngineError(
                f"The model does not answer with an option letter after this chat template (template profile "
                f"{self.profile.name!r}, most likely next tokens: {likely}). If its answer needs a header first, "
                "pass it with --assistant-prefix; see agent_docs/gotchas.md")

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
        # A header the template leaves to the model, such as Muse Glimmer's recipient, belongs to the tail.
        return prompt[:start], payload, prompt[start + len(payload):] + self.assistant_prefix

    def _template(self) -> tuple[str, str, list[int], list[int]]:
        """The template text around the payload, with its tokens. Rendered once, not once per question.

        The head is tokenized with `add_special`, which prepends BOS for a vocabulary that has one. The
        rendered template does not carry it as text (llama.cpp strips a leading BOS from the rendering so
        that tokenization adds exactly one), so without this Gemma read its prompts without BOS.

        Two probes with different payloads must give the same head and tail; a template that folds the
        payload into either one is refused here rather than producing wrong prompts.
        """
        with self._template_lock:
            if self._template_parts is None:
                head, _, tail = self._split(messages("a", "b", ["Yes", "No"]))
                other = self._split(messages({"x": ["y"]}, "c" * 40, ["Yes", "No", "Maybe"]))
                if (head, tail) != (other[0], other[2]):
                    raise EngineError("The chat template depends on the payload; llav cannot split it")
                self._template_parts = (head, tail, self.client.tokenize(head, add_special=True),
                                        self.client.tokenize(tail))
            return self._template_parts

    def encode_all(self, state, questions: list[Question]) -> tuple[list[int], list[list[int]]]:
        """The state prefix and every question's tokens, tokenizing the state once instead of per question.

        The state dominates the payload, so tokenizing it for each question is most of llav's own cost per
        request. The first question is checked against the whole-payload tokenization; on any difference the
        whole request falls back to `encode`, which tokenizes each payload in full.
        """
        head, _, head_ids, tail_ids = self._template()
        opening = evidence_opening(state)
        opening_ids = self.client.tokenize(opening, special=False)
        # The last token can merge with the text after the evidence value, so the shared part stops before
        # it and every question re-tokenizes from the character where that token began.
        prefix = head_ids + opening_ids[:-1]
        shared_text = self.client.detokenize(opening_ids[:-1])
        if not opening.startswith(shared_text):
            return self.state_prefix(state), [self.encode(state, question) for question in questions]
        encoded = []
        for index, question in enumerate(questions):
            payload = messages(state, question.instructions, list(question.descriptions))[-1]["content"]
            ids = prefix + self.client.tokenize(payload[len(shared_text):], special=False) + tail_ids
            if index == 0 and ids != self.encode(state, question):
                return self.state_prefix(state), [self.encode(state, q) for q in questions]
            if len(ids) >= self.slot_ctx:
                raise ContextTooLong(question.key, len(ids), self.slot_ctx - 1)
            if not self._boundary_ok(self._template()[1], len(question.descriptions)):
                raise EngineError("Answer boundary changes tokenization for this chat template")
            encoded.append(ids)
        return prefix, encoded

    def encode(self, state, question: Question) -> list[int]:
        # Caller text is tokenized without special-token parsing so it cannot forge chat turns. The
        # payload sits between a newline and a control token, both token boundaries, so the split
        # matches whole-prompt tokenization whenever the text has no control tokens.
        head, payload, tail = self._split(messages(state, question.instructions, list(question.descriptions)))
        ids = (self.client.tokenize(head, add_special=True) + self.client.tokenize(payload, special=False)
               + self.client.tokenize(tail))
        if len(ids) >= self.slot_ctx:
            raise ContextTooLong(question.key, len(ids), self.slot_ctx - 1)
        if not self._boundary_ok(tail, len(question.descriptions)):
            raise EngineError("Answer boundary changes tokenization for this chat template")
        return ids

    def state_prefix(self, state) -> list[int]:
        """Tokens up to the evidence value, minus the last token (it can merge with what follows)."""
        head, _, _ = self._split(messages(state, "prefix boundary placeholder", ["Yes", "No"]))
        return (self.client.tokenize(head, add_special=True)
                + self.client.tokenize(evidence_opening(state), special=False)[:-1])

    def encode_media(self, state, questions: list[Question], media: list[tuple[str, bytes]]) -> tuple[dict | None, list[dict]]:
        """Prompt objects for a media request: the shareable state prefix (None when it is not) and one
        prompt per question, each `{"prompt_string", "multimodal_data"}` with the media, in order, before
        the payload. `media` pairs a kind, "image" or "audio", with the file bytes; the server tells them
        apart itself, the kind only decides which projector modality the request needs.

        The server tokenizes the strings itself, with special-token parsing on and BOS added as `encode`
        adds it, and splits them at the media marker. The pieces are the template head, then the payload
        and tail as one piece, so the token ids are those of `encode` with the media chunks in between;
        what `encode` guarantees by
        tokenizing caller text without special parsing is guaranteed here by refusing caller text that
        special parsing would change. The prefix ends where `state_prefix` ends, and is used only when
        re-tokenizing it gives the same ids, or the server could not extend it exactly.
        """
        for kind, field in (("image", "images"), ("audio", "audio")):
            if any(item == kind for item, _ in media) and not self.modalities[kind]:
                if not any(self.modalities.values()):
                    raise MediaRequestError(["body", field], "this server has no projector; start llav with --mmproj")
                raise MediaRequestError(["body", field], f"this server's projector does not take {kind} input; "
                                        "start llav with an --mmproj that does")
        head, tail, head_ids, tail_ids = self._template()
        marker = self.media_marker
        markers = marker * len(media)
        data = [base64.b64encode(item).decode() for _, item in media]
        opening = evidence_opening(state)
        opening_ids = self.client.tokenize(opening, special=False)
        if marker in opening or self.client.tokenize(opening) != opening_ids:
            raise MediaRequestError(["body", "state"], "state contains chat control text; a media request cannot carry it")
        shared_text = self.client.detokenize(opening_ids[:-1])
        shared = opening.startswith(shared_text) and self.client.tokenize(shared_text, special=False) == opening_ids[:-1]
        prompts = []
        for index, question in enumerate(questions):
            payload = messages(state, question.instructions, list(question.descriptions))[-1]["content"]
            rest = payload[len(shared_text):] if shared else payload
            rest_ids = self.client.tokenize(rest, special=False)
            if marker in rest or self.client.tokenize(rest) != rest_ids:
                raise MediaRequestError(["body", "questions", question.key],
                                        "question contains chat control text; a media request cannot carry it")
            if shared and index == 0 and opening_ids[:-1] + rest_ids != self.client.tokenize(payload, special=False):
                shared = False
                rest_ids = self.client.tokenize(payload, special=False)
            text_tokens = len(head_ids) + (len(opening_ids) - 1 if shared else 0) + len(rest_ids) + len(tail_ids)
            if text_tokens >= self.slot_ctx:
                raise ContextTooLong(question.key, text_tokens, self.slot_ctx - 1)
            if not self._boundary_ok(tail, len(question.descriptions)):
                raise EngineError("Answer boundary changes tokenization for this chat template")
            prompts.append({"prompt_string": head + markers + payload + tail, "multimodal_data": data})
        prefix = {"prompt_string": head + markers + shared_text, "multimodal_data": data} if shared else None
        return prefix, prompts

    def _readout(self, prompt, count: int, slot: int, cache: bool) -> tuple[list[float], float, int]:
        """Option probabilities, their candidate mass, and the tokens evaluated; `prompt` is token ids or
        a media prompt object."""
        response = self.client.post("/completion", {
            "prompt": prompt, "n_predict": 1, "temperature": -1, "n_probs": self.n_probs,
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
        # The logprobs are normalized over the vocabulary, so their sum is the share the model gave the
        # declared options before the softmax over them discards the rest. With a floored label it is an
        # upper bound.
        mass = min(1.0, sum(math.exp(value) for value in logprobs))
        return softmax(logprobs), mass, int((response.get("timings") or {}).get("prompt_n") or 0)

    def _cache_name(self, prefix) -> str:
        """The slot file for a state prefix: token ids, or a media prompt object keyed by its text and
        the media bytes."""
        if isinstance(prefix, dict):
            material = b"media:" + prefix["prompt_string"].encode() + b"".join(
                hashlib.sha256(base64.b64decode(item)).digest() for item in prefix["multimodal_data"])
        else:
            material = b"tokens:" + b"".join(token.to_bytes(4, "big") for token in prefix)
        return f"llav-{self._run}-{hashlib.sha256(material).hexdigest()[:32]}.bin"

    def _drop(self, name: str) -> None:
        self._cached.pop(name, None)
        try:
            os.remove(self.slot_dir / name)
        except OSError:
            pass

    def _load_prefix(self, slot: int, prefix) -> tuple[str, int, int]:
        """Leave `slot` holding exactly `prefix`; returns the slot file (or ""), the tokens evaluated, and
        the tokens the prefix holds (media tokens included, which only the server can count).

        A cached state is restored in about 20 ms, against seconds to evaluate it again. The file is kept
        only when a later question or request can use it: a trimming backend serving one question needs
        none.
        """
        name = self._cache_name(prefix)
        with self._cache_lock:
            held = self._cached.get(name)
            if held is not None:
                self._cached.move_to_end(name)
        if held is not None:
            try:
                self.client.post(f"/slots/{slot}?action=restore", {"filename": name})
                return name, 0, held
            except EngineError:
                with self._cache_lock:  # the file went missing; fall through and evaluate the state again
                    self._drop(name)
        self.client.post(f"/slots/{slot}?action=erase", {})
        primed = self.client.post("/completion", {
            "prompt": prefix, "n_predict": 0, "cache_prompt": True, "id_slot": slot,
        })
        evaluated = int((primed.get("timings") or {}).get("prompt_n") or (0 if isinstance(prefix, dict) else len(prefix)))
        if not self.state_cache and self.trims:
            return "", evaluated, evaluated  # nothing will restore it: no file
        self.client.post(f"/slots/{slot}?action=save", {"filename": name})
        if not self.state_cache:
            return name, evaluated, evaluated  # this request's questions restore it; the caller deletes it
        with self._cache_lock:
            self._cached[name] = evaluated
            self._cached.move_to_end(name)
            while len(self._cached) > self.state_cache:
                self._drop(next(iter(self._cached)))
        return name, evaluated, evaluated

    def clear_cache(self) -> None:
        """Remove every slot file this engine wrote. Called on shutdown; the files are useless after it."""
        with self._cache_lock:
            for name in list(self._cached):
                self._drop(name)

    def _evaluate_native(self, prefix: list[int], encoded: list[list[int]], counts: list[int]):
        """All questions in one batched pass. Returns None when the helper cannot take them."""
        native = self.native
        if native is None or not prefix or len(encoded) > native.max_questions:
            return None
        try:
            logits, normalizers, evaluated = native.evaluate(prefix, [ids[len(prefix):] for ids in encoded])
        except NativeError as error:
            self.native = None  # one broken helper must not break every later request
            self.native_error = str(error)
            native.kill()  # a helper that answered wrongly may still be running, holding a copy of the model
            print(f"llav: native readout disabled: {error}", file=sys.stderr)
            return None
        results = [softmax(values[:count]) for values, count in zip(logits, counts)]
        masses = [min(1.0, sum(math.exp(value - norm) for value in values[:count]))
                  for values, norm, count in zip(logits, normalizers, counts)]
        return results, masses, evaluated

    def evaluate(self, state, questions: list[Question], media: list[tuple[str, bytes]] = ()) -> tuple[list[list[float]], dict, dict]:
        """Score every question against `state`; returns per-question probabilities, usage, and meta.

        `meta["candidate_mass"]` is, per question, the vocabulary probability on its declared labels: near 1
        when the model answered with an option letter, low when it wanted another token and the softmax over
        the options is noise. With `media` (see `encode_media`), the model reads it before the state.
        """
        counts = [len(question.descriptions) for question in questions]
        if media:
            prefix, prompts = self.encode_media(state, questions, list(media))
            # Encoding the media dominates, so one question shares its prefix whenever it can be cached.
            if len(prompts) == 1 and not self.state_cache:
                prefix = None
            try:
                slot = self.free.get(timeout=self.queue_timeout)
            except queue.Empty:
                raise Overloaded("All llama-server slots are busy") from None
            try:
                return self._evaluate_prompts(slot, prefix, prompts, counts, [q.key for q in questions],
                                              time.perf_counter())
            except MediaRequestError as error:
                # The server does not say which file it could not decode; point at a field that has one.
                if error.loc == ["body", "images"] and not any(kind == "image" for kind, _ in media):
                    error.loc = ["body", "audio"]
                raise
            finally:
                self.free.put(slot)
        prefix, encoded = self.encode_all(state, questions)
        # One question is worth a shared prefix only when the state can be cached for a later request:
        # priming it costs an extra pass, and pays for itself the next time the same state arrives.
        if len(encoded) == 1 and not self.state_cache and self.native is None:
            prefix = []
        elif len(encoded) == 1 and len(prefix) < STATE_CACHE_MIN_TOKENS:
            prefix = []
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
        shared = bool(prefix) and all(ids[: len(prefix)] == prefix and len(ids) > len(prefix) for ids in encoded)
        native = self._evaluate_native(prefix, encoded, counts) if shared else None
        if native is not None:
            results, masses, computed = native
            # The helper keeps the last state resident, so a repeat evaluates only the questions' own tokens.
            cached = computed <= sum(len(ids) - len(prefix) for ids in encoded)
            usage = {"input_tokens": computed, "output_tokens": 0}
            meta = {"shared_state_tokens": len(prefix), "seconds": time.perf_counter() - started,
                    "state_cache": "hit" if cached else "miss", "candidate_mass": masses}
            return results, usage, meta
        return self._evaluate_prompts(slot, prefix if shared else None, encoded, counts, None, started)

    def _evaluate_prompts(self, slot: int, prefix, prompts: list, counts: list[int], keys: list[str] | None,
                          started: float):
        """Read every prompt out on `slot`, through the shared prefix when there is one.

        `prompts` are token lists or media prompt objects. `keys` names the questions when the server, not
        llav, counts the prompt (media tokens), so that its context error becomes a 422 for that question.
        """
        computed = 0
        results = []
        masses = []
        cached = False
        held = 0
        if prefix is None:
            for index, (prompt, count) in enumerate(zip(prompts, counts)):
                probabilities, mass, n = self._call_readout(prompt, count, slot, False, keys, index)
                results.append(probabilities)
                masses.append(mass)
                computed += n
        else:
            try:
                name, evaluated, held = self._load_prefix(slot, prefix)
            except EngineError as error:
                raise self._caller_error(error, keys, 0) from error
            cached = evaluated == 0
            computed += evaluated
            try:
                for index, (prompt, count) in enumerate(zip(prompts, counts)):
                    # A trimming backend rolls back to the prefix by itself; the rest need the slot file.
                    if index and not self.trims:
                        self.client.post(f"/slots/{slot}?action=restore", {"filename": name})
                    probabilities, mass, n = self._call_readout(prompt, count, slot, True, keys, index)
                    results.append(probabilities)
                    masses.append(mass)
                    computed += n
            finally:
                if name and not self.state_cache:
                    self._drop(name)
        usage = {"input_tokens": computed, "output_tokens": 0}
        # Token prefixes are counted by llav; a media prefix only by the server, media tokens included.
        meta = {"shared_state_tokens": len(prefix) if isinstance(prefix, list) else held,
                "seconds": time.perf_counter() - started,
                "state_cache": ("hit" if cached else "miss") if prefix is not None and self.state_cache else "off",
                "candidate_mass": masses}
        return results, usage, meta

    def _call_readout(self, prompt, count: int, slot: int, cache: bool, keys: list[str] | None, index: int):
        try:
            return self._readout(prompt, count, slot, cache)
        except EngineError as error:
            raise self._caller_error(error, keys, index) from error

    @staticmethod
    def _caller_error(error: EngineError, keys: list[str] | None, index: int) -> Exception:
        """A server error that a media request's caller caused, as its 422: the context error as
        `ContextTooLong` for the question it hit, an undecodable file as `MediaRequestError`. Any other
        error as is. `keys` is None on the token path, where llav checked the prompt itself."""
        if keys is None:
            return error
        match = _CONTEXT_EXCEEDED.search(str(error))
        if match is not None:
            return ContextTooLong(keys[index], int(match.group(1)), int(match.group(2)))
        if _MEDIA_UNREADABLE in str(error):
            return MediaRequestError(["body", "images"], "a file could not be decoded: images must be PNG, JPEG "
                                     "or another format stb_image reads, audio must be WAV, MP3 or FLAC")
        return error
