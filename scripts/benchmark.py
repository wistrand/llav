#!/usr/bin/env python3
"""Measure a running llav: answer quality, request timing, and where a request's time goes.

    scripts/benchmark.py accuracy http://127.0.0.1:8080
    scripts/benchmark.py timing   http://127.0.0.1:8080
    PYTHONPATH=src scripts/benchmark.py phases http://127.0.0.1:8089 /tmp/llav-XXXX/slots

`accuracy` and `timing` talk to llav. `phases` talks to llama-server directly, so it needs that server's
URL and its --slot-save-path; it disturbs slot 0, so do not point it at a server that is serving.

The numbers in agent_docs/research.md come from these three commands. Question sets are small on purpose:
they separate a broken model from a working one and show where a model is confidently wrong. They are not
an accuracy benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid

SENTIMENT = {"positive": "Happy, pleased, satisfied", "neutral": "Neither positive nor negative",
             "negative": "Unhappy, angry, disappointed"}
LANGUAGES = {"english": None, "french": None, "german": None, "spanish": None}
TOPICS = {"sports": None, "politics": None, "cooking": None, "finance": None}
DEPARTMENTS = {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
               "sales": "Pricing, upgrades, new accounts"}


def noul(instructions):
    return {"type": "noul", "instructions": instructions}


def choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def keys(*names):
    return {name: None for name in names}


# (state, [(answer key, question, expected)]); expected is True/False for noul, an option key for choice.
EASY = [
    ("I love this blender, best purchase I've made all year!", [
        ("praise", noul("Is the customer praising the product?"), True),
        ("urgent", noul("Does this need an immediate response?"), False),
        ("sentiment", choice("What is the sentiment?", SENTIMENT), "positive")]),
    ("The whole site is down and none of our customers can log in. Fix this NOW.", [
        ("urgent", noul("Does this need an immediate response?"), True),
        ("praise", noul("Is the customer praising the product?"), False),
        ("department", choice("Which team should handle this?", DEPARTMENTS), "technical"),
        ("sentiment", choice("What is the sentiment?", SENTIMENT), "negative")]),
    ("I was charged twice for my March invoice. Please refund the duplicate.", [
        ("department", choice("Which team should handle this?", DEPARTMENTS), "billing"),
        ("hours", noul("Is the writer asking about opening hours?"), False)]),
    ("Do you offer a discount if we upgrade 50 seats to the enterprise plan?", [
        ("department", choice("Which team should handle this?", DEPARTMENTS), "sales"),
        ("complaint", noul("Is this a complaint?"), False)]),
    ("Bonjour, je voudrais réserver une table pour deux personnes ce soir.", [
        ("language", choice("Which language is this written in?", LANGUAGES), "french")]),
    ("Guten Morgen, ich habe eine Frage zu meiner Bestellung.", [
        ("language", choice("Which language is this written in?", LANGUAGES), "german")]),
    ("Whisk the eggs with sugar until pale, then fold in the flour and bake for 25 minutes.", [
        ("topic", choice("What is the topic?", TOPICS), "cooking"),
        ("sentiment", choice("What is the sentiment?", SENTIMENT), "neutral")]),
    ("The striker scored twice in the second half to win the cup final 3-1.", [
        ("topic", choice("What is the topic?", TOPICS), "sports")]),
    ("The central bank raised interest rates by half a point, and bond yields jumped.", [
        ("topic", choice("What is the topic?", TOPICS), "finance")]),
    ("It is raining today.", [
        ("rain", noul("Does the text say it is raining?"), True),
        ("snow", noul("Does the text say it is snowing?"), False)]),
]

# Sarcasm, negation, implicature, pronoun reference, and small counting and date steps.
HARD = [
    ("Oh great, another update that deleted all my settings. Just what I needed.", [
        ("sentiment", choice("What is the sentiment?", SENTIMENT), "negative"),
        ("praise", noul("Is the writer praising the update?"), False)]),
    ("The food wasn't bad at all. Honestly the best pasta I've had in years.", [
        ("sentiment", choice("What is the sentiment?", SENTIMENT), "positive")]),
    ("Please do not cancel my subscription; I only wanted to change the card on file.", [
        ("cancel", noul("Is the customer asking to cancel the subscription?"), False),
        ("department", choice("Which team should handle this?", DEPARTMENTS), "billing")]),
    ("My invoice shows the correct amount, but the button to download it as a PDF throws a 500 error.", [
        ("department", choice("Which team should handle this?", DEPARTMENTS), "technical")]),
    ("We're a five-person startup. If you have a cheaper tier than Pro we'd switch today; otherwise we'll "
     "look elsewhere.", [
        ("department", choice("Which team should handle this?", DEPARTMENTS), "sales"),
        ("churn", noul("Is there a risk of losing this customer?"), True)]),
    ("The app works fine on my phone; it's only the desktop site that won't load since yesterday.", [
        ("mobile", noul("Does the mobile app have a problem?"), False)]),
    ("I told my manager the tool is great, but between us, I'm still doing half the work in spreadsheets.", [
        ("satisfied", noul("Is the user fully satisfied with the tool?"), False)]),
    ("Ich liebe dieses Produkt, aber der Versand war eine Katastrophe.", [
        ("shipping", noul("Does the writer complain about shipping?"), True)]),
    ("Anna gave the keys to Maria because she was leaving town for a month.", [
        ("who", choice("Who is leaving town?", keys("Anna", "Maria")), "Anna")]),
    ("The meeting was moved from Tuesday to Thursday, then pushed back one more day.", [
        ("day", choice("On which day is the meeting now?",
                       keys("Tuesday", "Wednesday", "Thursday", "Friday")), "Friday")]),
    ("I had three apples, ate one, and then bought two more.", [
        ("count", choice("How many apples do I have now?", keys("2", "3", "4", "5")), "4")]),
    ("Unless the customer replies by Friday, we close the ticket. The customer replied on Thursday.", [
        ("closed", noul("Should the ticket be closed under this rule?"), False)]),
    ("Not a single one of the 40 testers reported a crash.", [
        ("crash", noul("Did any tester report a crash?"), False)]),
    ("Thanks for the quick fix! Although the dashboard now takes 30 seconds to load instead of 3.", [
        ("problem", noul("Is there still a problem?"), True)]),
]

# About 75 tokens; 24 copies make the 1,800-token state the research numbers use.
PARAGRAPH = ("Customer: Hi, I have been trying to export my monthly report since Tuesday. Every time I click "
             "export the page spins and then shows an error saying the request timed out. Agent: Sorry to "
             "hear that. Could you tell me the browser and the size of the report? Customer: Chrome, and the "
             "report has about 40,000 rows. ")
LONG_QUESTIONS = ["Is the customer frustrated?", "Is this about billing?", "Is the customer asking for a "
                  "refund?", "Does the issue involve exporting data?", "Is the customer using Chrome?",
                  "Is this a sales inquiry?", "Did the agent apologize?", "Is the report small?",
                  "Is there an error message?", "Is the customer rude?"]


def ask(url: str, state, questions: dict) -> tuple[dict, dict]:
    body = json.dumps({"state": state, "model": "llav-latest", "questions": questions}).encode()
    request = urllib.request.Request(url + "/v1/systemone", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response), dict(response.headers)


def picked(answer: dict):
    return answer["noul"] > 0.5 if answer["type"] == "noul" else answer["choice"]


def certainty(answer: dict) -> float:
    return answer["noul"] if answer["type"] == "noul" else answer["probabilities"][answer["choice"]]


def score(url: str, cases: list, label: str) -> list[str]:
    """Answer each case and report the misses, with how sure the model was of the wrong answer."""
    correct = total = 0
    misses = []
    for state, items in cases:
        body, _ = ask(url, state, {key: question for key, question, _ in items})
        answers = body["answers"]
        for key, _, expected in items:
            answer = answers[key]
            total += 1
            if picked(answer) == expected:
                correct += 1
            else:
                misses.append(f"  {label} {state[:44]!r} {key}: {picked(answer)} at "
                              f"{certainty(answer):.2f}, expected {expected}")
    print(f"{label}: {correct}/{total}")
    return misses


def order_bias(url: str) -> None:
    """Ask every choice question again with its options reversed. A flip is position bias, not judgement."""
    flips = total = 0
    shifts = []
    for state, items in EASY:
        forward = {key: question for key, question, _ in items if question["type"] == "choice"}
        if not forward:
            continue
        reversed_options = {key: choice(question["instructions"],
                                        dict(reversed(list(question["criteria"].items()))))
                            for key, question in forward.items()}
        first = ask(url, state, forward)[0]["answers"]
        second = ask(url, state, reversed_options)[0]["answers"]
        for key in forward:
            total += 1
            flips += first[key]["choice"] != second[key]["choice"]
            shifts.append(max(abs(first[key]["probabilities"][option] - second[key]["probabilities"][option])
                              for option in first[key]["probabilities"]))
    print(f"order flips: {flips}/{total}, median probability shift {sorted(shifts)[len(shifts) // 2]:.2f}, "
          f"max {max(shifts):.2f}")


def accuracy(args) -> None:
    misses = score(args.url, EASY, "easy") + score(args.url, HARD, "hard")
    order_bias(args.url)
    for miss in misses:
        print(miss)


def timing(args) -> None:
    """Cold and repeat requests on one state. The tag keeps the state out of a previous run's cache."""
    state = PARAGRAPH * 24 + f"Run {uuid.uuid4().hex}. "
    ten = {f"q{index}": noul(text) for index, text in enumerate(LONG_QUESTIONS)}
    five = {key: ten[key] for key in list(ten)[:5]}
    for label, questions in [("10 questions, cold", ten), ("10 questions, repeat", ten),
                             ("5 questions, repeat", five), ("1 question, repeat", dict(list(five.items())[:1]))]:
        started = time.perf_counter()
        answers, headers = ask(args.url, state, questions)
        print(f"{label:22} {time.perf_counter() - started:5.2f} s  "
              f"cache={headers.get('X-Llav-State-Cache', '?'):4} "
              f"state={headers.get('X-Llav-Shared-State-Tokens', '?')} tokens  "
              f"evaluated={answers['usage']['input_tokens']}")


def phases(args) -> None:
    """Split a shared-state request into prime, save, restore and readout, against llama-server itself."""
    from llav.engine import Engine, LlamaClient  # noqa: PLC0415 - only this command needs llav importable
    from llav.questions import Question

    client = LlamaClient(args.llama_url)
    engine = Engine(client, 1, 32768, args.slot_dir, state_cache=0)
    state = PARAGRAPH * 24
    questions = [Question(key=f"q{index}", type="noul", instructions=text, option_ids=("true", "false"),
                          descriptions=("Yes", "No")) for index, text in enumerate(LONG_QUESTIONS)]
    engine.evaluate(state, questions[:1])  # warm the weights

    started = time.perf_counter()
    encoded = [engine.encode(state, question) for question in questions]
    encode_seconds = time.perf_counter() - started
    prefix = engine.state_prefix(state)

    started = time.perf_counter()
    client.post("/slots/0?action=erase", {})
    client.post("/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
    prime_seconds = time.perf_counter() - started
    started = time.perf_counter()
    client.post("/slots/0?action=save", {"filename": "benchmark.bin"})
    save_seconds = time.perf_counter() - started

    restore_seconds = readout_seconds = 0.0
    for ids in encoded:
        started = time.perf_counter()
        client.post("/slots/0?action=restore", {"filename": "benchmark.bin"})
        restored = time.perf_counter()
        client.post("/completion", {"prompt": ids, "n_predict": 1, "temperature": -1, "n_probs": 128,
                                    "cache_prompt": True, "id_slot": 0})
        restore_seconds += restored - started
        readout_seconds += time.perf_counter() - restored

    count = len(encoded)
    print(f"trim probe: {'trims, no slot file needed' if engine.trims else 'needs the slot file'}")
    print(f"state {len(prefix)} tokens, {len(encoded[0]) - len(prefix)} tokens per question, {count} questions")
    print(f"  prime            {prime_seconds:6.2f} s")
    print(f"  save             {save_seconds * 1000:6.0f} ms")
    print(f"  restore          {restore_seconds / count * 1000:6.0f} ms per question")
    print(f"  readout          {readout_seconds / count * 1000:6.0f} ms per question")
    print(f"  llav's own work  {encode_seconds / count * 1000:6.0f} ms per question")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="benchmark.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function, help_text in [
        ("accuracy", accuracy, "Easy and hard question sets plus an option-order check"),
        ("timing", timing, "Cold and repeat requests on a 1,800-token state"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("url", help="llav base URL, for example http://127.0.0.1:8080")
        command.set_defaults(run=function)
    command = commands.add_parser("phases", help="Per-phase timing against llama-server; disturbs slot 0")
    command.add_argument("llama_url", help="llama-server base URL, for example http://127.0.0.1:8089")
    command.add_argument("slot_dir", help="That server's --slot-save-path")
    command.set_defaults(run=phases)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
