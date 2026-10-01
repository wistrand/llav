"""Cost estimates for the X-Llav-Cost-* headers, from a price per hour of server time given at startup."""

from __future__ import annotations

from collections import deque
import threading
import time

WINDOW = 100  # requests in the rolling averages


class CostMeter:
    """Turns an hourly price into a per-request cost and rolling per-token figures.

    The price is the operator's (a rented GPU's hourly rate, a Space's hardware tier); llav cannot know what
    the machine costs. A figure per million tokens depends on throughput, so it is a rolling average over the
    last WINDOW requests, taken two ways: over the engine time they took, which is what the tokens cost while
    the server was busy, and over the wall time they span, which is what they cost at the current load, idle
    between requests included. Tokens are `usage.input_tokens`: the tokens evaluated, so a cached state
    counts nothing, as it costs nothing.
    """

    def __init__(self, usd_per_hour: float, clock=time.monotonic):
        self.usd_per_hour = usd_per_hour
        self.rate = usd_per_hour / 3600  # USD per second
        self.clock = clock
        self.window: deque = deque(maxlen=WINDOW)  # (started, finished, engine seconds, tokens)
        self.lock = threading.Lock()

    def now(self) -> float:
        return self.clock()

    def record(self, started: float, seconds: float, tokens: int) -> dict[str, str]:
        """Add one finished request and return its headers."""
        finished = self.clock()
        with self.lock:
            self.window.append((started, finished, seconds, tokens))
            busy = sum(entry[2] for entry in self.window)
            total = sum(entry[3] for entry in self.window)
            elapsed = finished - min(entry[0] for entry in self.window)
        headers = {"X-Llav-Cost": f"{seconds * self.rate:.6f}"}
        if total > 0:
            headers["X-Llav-Cost-Per-Mtok"] = f"{busy * self.rate / total * 1e6:.4f}"
            headers["X-Llav-Cost-Per-Mtok-Elapsed"] = f"{elapsed * self.rate / total * 1e6:.4f}"
        return headers
