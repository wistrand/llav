#!/usr/bin/env python3
"""A research proxy in front of llav that answers each choice question as the average over several option
orders, for benchmark harnesses that speak /v1/systemone.

    scripts/orders-proxy.py http://127.0.0.1:8765 --port 8767 --orders 4

Every request is forwarded to llav as one request holding, for each choice question, the caller's order and
`--orders` - 1 further orders (the cyclic rotations first, then random permutations), and the answers are
averaged per option and renormalized; the chosen option and confidence are recomputed from the average.
Nouls and scores are forwarded once: llav fixes a noul's Yes, No order, and a score's levels are ordinal.
The response keeps the System One shape and llav's own model id; `X-Llav-Orders` says how many orders each
question got. Not part of llav: the server itself still answers one order (see agent_docs/experiments/).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:8765"
ORDERS = 4
NOULS_AS_CHOICE = False  # ask a noul as a yes/no choice in both orders and average; changes the prompt
MAX_ORDERS = 16  # the native helper's batch, so the whole request still goes through it in one pass


def orders_for(keys: list[str], count: int, rng: random.Random) -> list[list[str]]:
    """The caller's order, then cyclic rotations, then random permutations, `count` distinct orders in all."""
    seen = [list(keys)]
    for shift in range(1, len(keys)):
        if len(seen) >= count:
            break
        seen.append(keys[shift:] + keys[:shift])
    while len(seen) < count and len(seen) < math.factorial(len(keys)):
        order = rng.sample(keys, len(keys))
        if order not in seen:
            seen.append(order)
    return seen


def expand(body: dict, count: int) -> tuple[dict, dict]:
    """The upstream body, and for each caller key the upstream keys that answer it."""
    questions = {}
    groups = {}
    for key, question in body["questions"].items():
        if NOULS_AS_CHOICE and isinstance(question, dict) and question.get("type") == "noul":
            criteria = question.get("criteria") or {}
            if isinstance(criteria, dict):
                # A noul's Yes, No order is fixed on the server; as a choice the two can be swapped.
                question = {"type": "choice", "instructions": question.get("instructions"),
                            "criteria": {"yes": criteria.get("true"), "no": criteria.get("false")}}
        is_choice = isinstance(question, dict) and question.get("type") == "choice"
        if not (is_choice and isinstance(question.get("criteria"), dict)):
            questions[key] = question
            groups[key] = [key]
            continue
        criteria = question["criteria"]
        rng = random.Random(f"{key}:{sorted(criteria)}")
        names = []
        for index, order in enumerate(orders_for(list(criteria), count, rng)):
            name = key if index == 0 else f"{key}\u0000{index}"
            questions[name] = {**question, "criteria": {k: criteria[k] for k in order}}
            names.append(name)
        groups[key] = names
    return {**body, "questions": questions}, groups


def confidence(probabilities: list[float]) -> float:
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, min(1.0, 1 - entropy / math.log(len(probabilities)))) if len(probabilities) > 1 else 1.0


def collapse(upstream: dict, groups: dict, original: dict) -> dict:
    answers = {}
    for key, names in groups.items():
        parts = [upstream["answers"][name] for name in names]
        if original["questions"][key].get("type") == "noul" and parts[0]["type"] == "choice":
            p_yes = sum(part["probabilities"]["yes"] for part in parts) / len(parts)  # asked as a choice
            answers[key] = {"type": "noul", "noul": p_yes}
            continue
        if len(parts) == 1:
            answers[key] = parts[0]
            continue
        keys = list(parts[0]["probabilities"])
        mean = {k: sum(part["probabilities"][k] for part in parts) / len(parts) for k in keys}
        total = sum(mean.values())
        mean = {k: v / total for k, v in mean.items()}
        answers[key] = {"type": "choice", "choice": max(mean, key=mean.get), "probabilities": mean,
                        "confidence": confidence(list(mean.values()))}
    return {"model": upstream["model"], "answers": answers, "usage": upstream["usage"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status: int, payload: dict, extra: dict | None = None) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._forward("GET", None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path != "/v1/systemone":
            self._forward("POST", raw)
            return
        try:
            body = json.loads(raw)
            questions = body["questions"] if isinstance(body, dict) else None
            assert isinstance(questions, dict)
        except (ValueError, KeyError, AssertionError):
            self._forward("POST", raw)  # let llav produce the 422
            return
        expanded, groups = expand(body, ORDERS)
        status, parsed, headers = post(json.dumps(expanded).encode(), self.headers.get("Authorization"))
        if status != 200:
            self._send(status, parsed if isinstance(parsed, dict) else {"detail": str(parsed)})
            return
        counts = ",".join(str(len(groups[key])) for key in body["questions"])
        self._send(200, collapse(parsed, groups, body),
                   {"X-Llav-Orders": counts, **{k: v for k, v in headers.items() if k.lower().startswith("x-llav-")}})

    def _forward(self, method: str, raw: bytes | None) -> None:
        request = urllib.request.Request(UPSTREAM + self.path, data=raw, method=method,
                                         headers={"Content-Type": "application/json"} if raw else {})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as error:
            data = error.read()
            self.send_response(error.code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def post(data: bytes, authorization: str | None) -> tuple[int, object, dict]:
    headers = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(UPSTREAM + "/v1/systemone", data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.load(response), dict(response.headers)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error), {}
        except ValueError:
            return error.code, {"detail": error.reason}, {}


def main(argv=None) -> None:
    global UPSTREAM, ORDERS, NOULS_AS_CHOICE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("upstream", nargs="?", default=UPSTREAM, help="llav base URL")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--orders", type=int, default=ORDERS, help=f"Orders per choice question, at most {MAX_ORDERS}")
    parser.add_argument("--nouls-as-choice", action="store_true",
                        help="Ask nouls as yes/no choices in both orders and average (a different prompt)")
    args = parser.parse_args(argv)
    if not 1 <= args.orders <= MAX_ORDERS:
        sys.exit(f"--orders must be 1 to {MAX_ORDERS}")
    UPSTREAM = args.upstream.rstrip("/")
    ORDERS = args.orders
    NOULS_AS_CHOICE = args.nouls_as_choice
    print(f"orders-proxy: {args.orders} orders per choice question, llav at {UPSTREAM}, listening on {args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
