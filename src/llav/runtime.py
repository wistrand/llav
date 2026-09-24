"""Start and stop a llama-server process configured for llav."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .engine import TEMPLATE_KWARGS

# Runs beside llama-server holding the read end of a pipe from llav. The kernel closes the pipe when llav
# dies for any reason, including SIGKILL, which no cleanup handler can catch; on EOF this stops the child
# and removes llav's scratch directory, if one was given.
_WATCHDOG = """
import os, shutil, signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
pid, scratch = int(sys.argv[1]), sys.argv[2]
sys.stdin.buffer.read()
try:
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(0.2)
        os.kill(pid, 0)
    os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
except ProcessLookupError:
    pass
if scratch:
    shutil.rmtree(scratch, ignore_errors=True)
"""


def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


class LlamaProcess:
    def __init__(self, binary: str, gguf: Path, port: int, slots: int, slot_ctx: int, slot_dir: Path,
                 log_path: Path, extra: list[str], load_timeout: float = 900, scratch: Path | None = None):
        # Another server on the port would answer the readiness probe, and llav would silently share its
        # slots while this llama-server fails to bind and exits.
        if _port_in_use(port):
            raise RuntimeError(f"Port {port} is already in use; choose another with --llama-port")
        self.url = f"http://127.0.0.1:{port}"
        self.command = [
            binary, "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
            "-ngl", "all", "-np", str(slots), "-c", str(slots * slot_ctx),
            "--ctx-checkpoints", "0", "--slot-save-path", str(slot_dir),
            # A sliding-window model keeps only the last window by default, so a question cannot extend a
            # cached state and re-reads all of it: 19.7 s against 0.62 s for 10 questions on Muse Glimmer.
            # Models without sliding windows ignore the flag (agent_docs/research.md).
            "--swa-full",
            "--jinja", "--chat-template-kwargs", json.dumps(TEMPLATE_KWARGS),
            "--no-webui", *extra,
        ]
        self.log_path = log_path
        self.log = log_path.open("a")
        try:
            self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise
        self.watchdog = None
        try:
            self.watchdog = subprocess.Popen(
                [sys.executable, "-c", _WATCHDOG, str(self.process.pid), str(scratch or "")], stdin=subprocess.PIPE
            )
            self._wait_ready(load_timeout)
        except BaseException:
            # Includes SIGTERM/Ctrl-C during a long model load: do not orphan the child.
            self.stop()
            raise

    def _log_tail(self, lines: int = 5) -> str:
        """The end of llama-server's log. The log lives in llav's scratch directory, which is removed on exit,
        so an error that only named the file would point at nothing."""
        try:
            self.log.flush()
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return ""
        kept = [line.strip() for line in text.splitlines() if line.strip()][-lines:]
        return (":\n  " + "\n  ".join(kept)) if kept else ""

    def _wait_ready(self, load_timeout: float) -> None:
        deadline = time.monotonic() + load_timeout
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}{self._log_tail()}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as response:
                    if json.load(response).get("status") == "ok":
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"llama-server not ready after {load_timeout}s{self._log_tail()}")
            time.sleep(0.5)

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        # Retire the watchdog before reaping the child, so it can never signal a reused pid.
        if self.watchdog:
            self.watchdog.kill()
            self.watchdog.wait()
            self.watchdog.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
