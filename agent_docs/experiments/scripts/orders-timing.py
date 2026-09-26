#!/usr/bin/env python3
"""Runs on the rented box (it starts scripts/orders-proxy.py from /root/llav). The cost of asking a choice
question in K orders: one long state, one 7-option question, K = 1, 2, 4, 7, 14
through scripts/orders-proxy.py, against a llav with the native helper and one without. Median of repeats
with the state already cached (the per-question cost), plus the first request that reads the state.

    python3 orders-timing.py http://127.0.0.1:8765 http://127.0.0.1:8766
"""
import json
import statistics
import subprocess
import sys
import time
import urllib.request

WORDS = ("The customer wrote to say that the invoice for the March order arrived late and listed a different "
         "shipping address than the one on file, and asked whether the discount agreed by phone still applied. ")
STATE = (WORDS * 40)[:9000]  # about 1,800 tokens
QUESTION = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments, invoices, refunds", "shipping": "Delivery and addresses",
                         "sales": "Discounts, quotes, renewals", "technical": "Bugs and outages",
                         "legal": "Contracts and compliance", "account": "Logins and profile changes",
                         "other": "None of the above"}}


def ask(url: str) -> float:
    body = json.dumps({"state": STATE, "model": "jev-latest", "questions": {"q": QUESTION}}).encode()
    started = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url + "/v1/systemone", data=body,
                                                       headers={"Content-Type": "application/json"}), timeout=300) as r:
        r.read()
    return time.perf_counter() - started


def main() -> None:
    results = []
    for name, upstream in zip(("native helper", "llama-server"), sys.argv[1:3]):
        for k in (1, 2, 4, 7, 14):
            proxy = subprocess.Popen([sys.executable, "/root/llav/scripts/orders-proxy.py", upstream, "--port", "8769",
                                      "--orders", str(k)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
            try:
                ask("http://127.0.0.1:8769")  # warm: the proxy, the tokenizer, the model
                # A new state each K so the first request reads it; then repeats hit the cache.
                global STATE
                STATE = STATE[:-1] + str(k)
                first = ask("http://127.0.0.1:8769")
                repeats = [ask("http://127.0.0.1:8769") for _ in range(8)]
            finally:
                proxy.terminate()
                proxy.wait()
            row = {"path": name, "orders": k, "first_s": round(first, 3), "repeat_median_s": round(statistics.median(repeats), 3),
                   "repeat_min_s": round(min(repeats), 3)}
            results.append(row)
            print(json.dumps(row), flush=True)
    json.dump(results, open("/root/perturb/orders-timing.json", "w"), indent=1)


if __name__ == "__main__":
    main()
