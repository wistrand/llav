"""Optional fast path: the `llav-readout` helper, which batches every question into one forward pass.

llama-server evaluates each question in its own pass and restores the state between them. The helper keeps
the state resident and decodes all the suffixes together, which is worth most on a fast GPU, where those
fixed costs dominate. It is opt-in (`--native-readout`); without it nothing changes.

Requests and responses are length-prefixed binary, little-endian, so neither side needs a parser:
    request   b"LLVR" n_prefix n_labels n_suffix, prefix tokens, label tokens, then per suffix: n, tokens
    response  b"LLVA" status evaluated, then per suffix n_labels float32 raw logits and the float32 log of
              the vocabulary normalizer; llav needs the normalizer only for the candidate mass

The ready line names the protocol version, so a helper built from older source is refused at startup instead
of being read with the wrong response layout.
"""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess
import threading


PROTOCOL = "protocol 2"


class NativeError(Exception):
    """The helper could not answer. The caller falls back to llama-server."""


class NativeReadout:
    """One helper process, one request at a time."""

    def __init__(self, binary: str, model: Path, labels: list[int], slot_ctx: int, max_questions: int = 16,
                 gpu_layers: int = 999):
        self.labels = list(labels)
        self.max_questions = max_questions
        self._lock = threading.Lock()
        self._errors: list[str] = []
        command = [binary, "--model", str(model), "--ctx", str(slot_ctx), "--seq", str(max_questions),
                   "--ngl", str(gpu_layers)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, bufsize=0)
        except OSError as error:
            raise NativeError(f"cannot start {binary}: {error}") from error
        try:
            self._wait_for_ready()
        except NativeError:
            self.process.kill()  # a helper with the wrong protocol is still running
            self.process.wait(timeout=10)
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                pipe.close()
            raise
        # Loading prints to stderr for the process's whole life; drain it so a full pipe cannot block it.
        threading.Thread(target=self._drain, daemon=True).start()

    def _wait_for_ready(self) -> None:
        for line in self.process.stderr:
            text = line.decode(errors="replace").rstrip()
            self._errors.append(text)
            if "llav-readout ready" in text:
                if PROTOCOL not in text:
                    raise NativeError(f"the helper does not speak {PROTOCOL}; rebuild it from native/")
                return
        raise NativeError("the helper exited before it was ready: " + " | ".join(self._errors[-3:]))

    def _drain(self) -> None:
        for line in self.process.stderr:
            self._errors.append(line.decode(errors="replace").rstrip())
            del self._errors[:-60]  # room for a debugger backtrace after the line that says why

    def _read(self, count: int) -> bytes:
        try:
            data = self.process.stdout.read(count)
        except (OSError, ValueError) as error:
            raise NativeError(f"the helper closed its output: {error}") from error
        if not data or len(data) != count:
            raise NativeError("the helper stopped answering: " + self._recent())
        return data

    def evaluate(self, prefix: list[int], suffixes: list[list[int]]) -> tuple[list[list[float]], list[float], int]:
        """Logits of every label and the log normalizer for each suffix, and the tokens the helper evaluated."""
        if not suffixes or len(suffixes) > self.max_questions:
            raise NativeError(f"{len(suffixes)} questions does not fit the helper's {self.max_questions}")
        body = [b"LLVR", struct.pack("<3i", len(prefix), len(self.labels), len(suffixes)),
                struct.pack(f"<{len(prefix)}i", *prefix), struct.pack(f"<{len(self.labels)}i", *self.labels)]
        for suffix in suffixes:
            body.append(struct.pack("<i", len(suffix)) + struct.pack(f"<{len(suffix)}i", *suffix))
        with self._lock:
            try:
                self.process.stdin.write(b"".join(body))
                self.process.stdin.flush()
            except (OSError, ValueError) as error:  # ValueError: the pipe was already closed
                raise NativeError(f"the helper closed its input: {error}") from error
            if self._read(4) != b"LLVA":
                raise NativeError("the helper sent a malformed response: " + self._recent())
            status, evaluated = struct.unpack("<2i", self._read(8))
            if status != 0:
                raise NativeError(f"the helper returned status {status}: " + self._recent())
            width = len(self.labels) + 1
            raw = self._read(4 * width * len(suffixes))
        values = struct.unpack(f"<{width * len(suffixes)}f", raw)
        rows = [values[index * width:(index + 1) * width] for index in range(len(suffixes))]
        return [list(row[:-1]) for row in rows], [row[-1] for row in rows], evaluated

    def _recent(self) -> str:
        """The helper's last messages. A failing helper is often exiting; give its stderr a moment to arrive."""
        try:
            self.process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        # A crashing helper ends with a debugger backtrace; the line that says why comes before it.
        causes = [line for line in self._errors if any(mark in line for mark in ("GGML_ASSERT", "error", "llav-readout:"))]
        return " | ".join((causes or self._errors)[-3:]) or "no message on stderr"

    def kill(self) -> None:
        """Stop a helper llav no longer trusts, without waiting; `close` reaps it at shutdown."""
        if self.process.poll() is None:
            self.process.kill()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
