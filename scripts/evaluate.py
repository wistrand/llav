#!/usr/bin/env python3
"""Labelled evaluation of a running llav: calibration, selective accuracy, and option-order sensitivity.

    scripts/evaluate.py fetch  ~/llav-eval [--semif ../SemIf]       # once; needs network, not llav
    scripts/evaluate.py score  http://127.0.0.1:8080 ~/llav-eval [--out predictions.jsonl]
    scripts/evaluate.py shifts http://127.0.0.1:8080 ~/llav-eval [--out shifts.jsonl]
    scripts/evaluate.py fit    predictions.jsonl --gguf MODEL.gguf --out calibration.json

`fetch` writes one JSONL file per source. Each line is one labelled question:

    {"id": "...", "source": "ag_news", "state": ..., "question": {System One question}, "label": ...}

`label` is `true` or `false` for a noul, the option key for a choice, and the level index for a score. Any
file in that format works, so a caller's own labelled data can be scored the same way. `score` and `shifts`
talk only the System One API, so `--model` points them at another server of that shape, such as CLM, and
`--api-key-file` adds the bearer token a hosted one needs. Jev through OpenRouter, for example:

    scripts/evaluate.py score https://openrouter.ai/api ~/llav-eval --model typesafe/jev-1.13 \
        --api-key-file ~/.config/openrouter-key

`score` asks every question once, as a caller would, and reports per source: accuracy; Brier score summed
over the declared options (so a noul's is twice the binary Brier score); negative log-likelihood of the
label; expected calibration error (ECE) of the top answer's probability over ten equal-width bins; coverage,
the largest share of questions that can be answered automatically, most confident first, while keeping the
error rate at or under 1%, 5% or 10%; accuracy split by candidate mass; and, per source of one question
type, Jevals' Decision Score (dscore: 100 x (1 - loss / loss of answering the label base rates), with the
Brier score for noul and choice and the ranked probability score for score).

`fetch` also rebuilds two Jevals tasks item for item (jevals.com, release 2026-09-18, CC-BY-4.0): PubMedQA
(noul) and HelpSteer2 helpfulness (score), 300 items each, with Jevals' own wording and labels and each text
checked against its published hash. Their dscore compares with the board's numbers for Jev and six LLMs.
Jevals' third task, the full 77-intent Banking77, exceeds llav's 26 options and is not fetched.

`shifts` asks every choice and score question once per cyclic rotation of its options, all rotations in one
request so they share the state. It reports how often the answer changes with the order, how far the
probabilities move, where in the list the chosen option sat, and the same metrics for the caller's own
order against the geometric mean over rotations (the order-debiasing that AnyJev calls cyclic-shift
marginalization). Nouls are skipped: llav fixes their order as Yes, No. Positions are counted in the order
requested; llav shows a single-letter option key at its own letter, so for such questions every rotation
reaches the model in the same order.

`fit` reads the predictions `score --out` wrote and fits one temperature per question type by minimizing
the negative log-likelihood of the labels. It first reports what the fit does on data it was not fitted on
(two folds split by question id, each fitted on the other), per source, then fits on everything and writes
the calibration file llav loads with `--calibration`. A type with fewer than MIN_FIT questions keeps T = 1.
The file is valid only for the model file given with `--gguf` and the current prompt version; llav checks
both at startup, and `fit` refuses predictions that `score` recorded from another model file. A type is left
uncalibrated unless each half has at least MIN_FIT questions and MIN_ERRORS wrong answers, so every
temperature written was checked on held-out data; a temperature that ends at the search limit is refused.
Fit on the workload the thresholds will run on: calibration measured here differs by task.

The fetched sources: BoolQ, AG News, DBpedia, Banking77's 13 card intents, Yelp stars, the two Jevals tasks,
and, added for the option-order experiment, full Banking77 and CLINC150 as 12-option choices among the gold
intent's nearest neighbours by name, TREC fine question types within their coarse group, MNLI, a 12-emotion
GoEmotions subset, and ChaosNLI's 1,599 MNLI items with their 100 human labels in a `human` field.
Sources are samples of public datasets, taken evenly across each split so class-sorted splits are covered.
They are downloaded for local evaluation only; their licences differ and none are redistributed here.
Results go in agent_docs/research.md (llav) and agent_docs/comparisons.md (other models and systems).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

EPSILON = 1e-12
BINS = 10
ERROR_TARGETS = (0.01, 0.05, 0.10)
# The web UI's "No option fits" cue for servers that do not report their own (backend.low_candidate_mass).
LOW_MASS = 0.9

ROWS_API = "https://datasets-server.huggingface.co/rows"
# Jevals (jevals.com, CC-BY-4.0) publishes the items, wording and labels it scores Jev and six LLMs on, but not
# the texts; those come from the public datasets. Pinned to one data commit so the item list cannot move.
JEVALS_SUITE = ("https://raw.githubusercontent.com/Jevals/jevals-data/21bb47b72814cf661539b313844d2d2e26166e54/"
                "suites/0.1.0/{bench}.json")
AG_NEWS = {"World": "World news, politics and international affairs", "Sports": "Sports",
           "Business": "Business, companies and the economy", "Sci/Tech": "Science and technology"}
DBPEDIA = {
    "Company": "A company or business", "EducationalInstitution": "A school, college or university",
    "Artist": "An artist, musician, writer or performer", "Athlete": "An athlete or sports player",
    "OfficeHolder": "A politician or holder of a public office",
    "MeanOfTransportation": "A vehicle, ship, aircraft or other means of transport",
    "Building": "A building or structure", "NaturalPlace": "A mountain, river, lake or other natural place",
    "Village": "A village or small settlement", "Animal": "An animal species", "Plant": "A plant species",
    "Album": "A music album", "Film": "A film", "WrittenWork": "A book, journal or other written work",
}
# Banking77 has 77 intents, above llav's 26 options. These 13 are the card intents: close to each other, so
# they test calibration where the options are genuinely confusable.
CARD_INTENTS = ["activate_my_card", "card_about_to_expire", "card_acceptance", "card_arrival",
                "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
                "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
                "compromised_card", "lost_or_stolen_card"]
STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
# Full Banking77 and CLINC150 have 77 and 150 intents; each question offers this many, the gold one and the
# closest by name (_confusable).
INTENT_OPTIONS = 12
MNLI = {"entailment": "The premise shows the hypothesis is true",
        "neutral": "The premise neither confirms nor contradicts the hypothesis",
        "contradiction": "The premise shows the hypothesis is false"}
# A GoEmotions subset whose emotions overlap (anger, annoyance, disapproval; joy, amusement, love).
EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "curiosity", "disapproval",
            "gratitude", "joy", "love", "sadness", "neutral"]
MAX_STATE_CHARS = 2000
MIN_FIT = 30
# With every answer right, NLL keeps falling as T shrinks and the fit runs to its limit, calling every answer
# certain. A few wrong answers pin the temperature down.
MIN_ERRORS = 5
T_LIMITS = (0.05, 20.0)
SRC = Path(__file__).resolve().parents[1] / "src"


# Fetching

def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "llav-evaluate (local evaluation)"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503) or attempt == 7:
                raise
            time.sleep(int(error.headers.get("Retry-After") or 0) or 15 * (attempt + 1))
    raise RuntimeError("unreachable")


def _page(dataset: str, config: str, split: str, offset: int, length: int) -> tuple[list[dict], int]:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": offset, "length": length})
    result = _get_json(f"{ROWS_API}?{query}")
    time.sleep(1)  # the rows API rate-limits bursts
    return [item["row"] for item in result["rows"]], result["num_rows_total"]


def _spread(dataset: str, config: str, split: str, rows: int, pages: int) -> list[dict]:
    """About `rows` rows from `pages` evenly spaced pages, so a split sorted by class still covers every class.

    Not truncated to `rows`: cutting the last page short would drop the last class of a sorted split.
    """
    per_page = math.ceil(rows / pages)
    _, total = _page(dataset, config, split, 0, 1)
    sample = []
    for index in range(pages):
        page, _ = _page(dataset, config, split, index * total // pages, per_page)
        sample += page
    return sample


def _everything(dataset: str, config: str, split: str) -> list[dict]:
    rows, total = _page(dataset, config, split, 0, 100)
    while len(rows) < total:
        page, _ = _page(dataset, config, split, len(rows), 100)
        if not page:
            break
        rows += page
    return rows


def _item(source: str, index: int, state: str, question: dict, label) -> dict:
    return {"id": f"{source}-{index}", "source": source, "state": state[:MAX_STATE_CHARS],
            "question": question, "label": label}


def fetch_boolq(rows: int) -> list[dict]:
    items = []
    for index, row in enumerate(_spread("google/boolq", "default", "validation", rows, 20)):
        text = row["question"].strip()
        question = {"type": "noul", "instructions": text[0].upper() + text[1:] + "?"}
        items.append(_item("boolq", index, row["passage"], question, bool(row["answer"])))
    return items


def fetch_ag_news(rows: int) -> list[dict]:
    names = list(AG_NEWS)
    question = {"type": "choice", "instructions": "What is the topic of this news article?", "criteria": AG_NEWS}
    return [_item("ag_news", index, row["text"], question, names[row["label"]])
            for index, row in enumerate(_spread("fancyzhx/ag_news", "default", "test", rows, 20))]


def fetch_dbpedia(rows: int) -> list[dict]:
    names = list(DBPEDIA)
    question = {"type": "choice", "instructions": "What kind of thing is this text about?", "criteria": DBPEDIA}
    return [_item("dbpedia", index, f"{row['title']}. {row['content'].strip()}", question, names[row["label"]])
            for index, row in enumerate(_spread("fancyzhx/dbpedia_14", "dbpedia_14", "test", rows, 28))]


def fetch_banking_cards(rows: int) -> list[dict]:
    names = _get_json(f"{ROWS_API}?" + urllib.parse.urlencode({
        "dataset": "legacy-datasets/banking77", "config": "default", "split": "test", "offset": 0, "length": 1,
    }))["features"][1]["type"]["names"]
    cap = math.ceil(rows / len(CARD_INTENTS))
    counts = {intent: 0 for intent in CARD_INTENTS}
    question = {"type": "choice", "instructions": "What does the customer need help with?",
                "criteria": {intent: None for intent in CARD_INTENTS}}
    items = []
    for row in _everything("legacy-datasets/banking77", "default", "test"):
        intent = names[row["label"]]
        if intent in counts and counts[intent] < cap:
            counts[intent] += 1
            items.append(_item("banking_cards", len(items), row["text"], question, intent))
    return items


def fetch_yelp(rows: int) -> list[dict]:
    question = {"type": "score", "instructions": "How many stars did the reviewer give?", "criteria": STARS}
    return [_item("yelp_stars", index, row["text"].replace("\\n", "\n"), question, row["label"])
            for index, row in enumerate(_spread("Yelp/yelp_review_full", "yelp_review_full", "test", rows, 20))]


def semif_authored(path: Path) -> list[dict]:
    """SemIf's 144 project-authored, labelled decisions (MIT): evidence, rules and candidate selection."""
    items = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        criteria = {option["id"]: option["description"] for option in row["options"]}
        question = {"type": "choice", "instructions": row["question"], "criteria": criteria}
        items.append({"id": f"semif-{row['id']}", "source": f"semif_{row['family']}", "state": row["state"],
                      "question": question, "label": row["options"][row["label"]]["id"]})
    return items


def jevals_state(row: dict, fields: list[str]) -> dict:
    """The state Jevals sends: each field under its first path segment, so `context.contexts` becomes
    `{"context": [...]}`. Its hash is over the compact JSON, as JavaScript's JSON.stringify writes it."""
    state = {}
    for field in fields:
        value = row
        for part in field.split("."):
            value = value[part]
        state[field.split(".")[0]] = value
    return state


def fetch_jevals(bench: str) -> list[dict]:
    """A Jevals task, item for item: its questions verbatim, its labels, the texts rebuilt from the dataset.

    Jevals' Decision Score for these items is on its board (release 2026-09-18), so `score` output for them
    compares with Jev directly. A text whose hash differs from Jevals' is dropped, not scored: the dataset
    changed since Jevals sampled it, and the item would no longer be the same.
    """
    suite = _get_json(JEVALS_SUITE.format(bench=bench))
    needed = {item["row_idx"] for item in suite["items"]}
    rows = {}
    for offset in sorted({index // 100 * 100 for index in needed}):
        page, _ = _page(suite["dataset"], suite["config"], suite["split"], offset, 100)
        rows.update({offset + position: row for position, row in enumerate(page)})
    question = {"type": suite["primitive"], "instructions": suite["instructions"], "criteria": suite["criteria"]}
    items, dropped = [], 0
    for item in suite["items"]:
        row = rows.get(item["row_idx"])
        state = jevals_state(row, suite["state_fields"]) if row else None
        text = json.dumps(state, separators=(",", ":"), ensure_ascii=False) if state else ""
        if hashlib.sha256(text.encode()).hexdigest() != item["state_sha256"]:
            dropped += 1
            continue
        option = suite["options"][item["target"]]
        label = option == "yes" if suite["primitive"] == "noul" else item["target"]
        items.append({"id": item["item_id"], "source": f"jevals_{bench}", "state": state, "question": question,
                      "label": label})
    if dropped:
        print(f"jevals_{bench}: {dropped} of {len(suite['items'])} texts differ from Jevals' and were left out")
    return items


def _confusable(gold: str, names: list[str], count: int, rng: random.Random) -> list[str]:
    """`gold` and the `count` - 1 names sharing the most underscore-separated words with it, in a seeded
    random order: intent names that share words ("card_arrival", "card_delivery_estimate") are the ones a
    model confuses."""
    words = set(gold.split("_"))
    others = [name for name in names if name != gold]
    rng.shuffle(others)  # breaks ties between equally close names
    others.sort(key=lambda name: -len(words & set(name.split("_"))) / len(words | set(name.split("_"))))
    options = [gold] + others[:count - 1]
    rng.shuffle(options)
    return options


def _intents(source: str, dataset: str, config: str, label: str, rows: int, skip: set[str]) -> list[dict]:
    """An intent dataset as choices of INTENT_OPTIONS confusable intents each, the gold one among them."""
    # Test splits are sorted by intent with a few dozen rows each: many small pages, or whole intents are missed.
    sample = _spread(dataset, config, "test", rows, max(20, rows // 10))
    names = _get_json(f"{ROWS_API}?" + urllib.parse.urlencode({
        "dataset": dataset, "config": config, "split": "test", "offset": 0, "length": 1,
    }))["features"]
    names = next(f for f in names if f["name"] == label)["type"]["names"]
    items = []
    for row in sample:
        gold = names[row[label]]
        if gold in skip:
            continue
        rng = random.Random(f"{source}:{len(items)}")
        options = _confusable(gold, [name for name in names if name not in skip], INTENT_OPTIONS, rng)
        question = {"type": "choice", "instructions": "What does the customer need help with?",
                    "criteria": {name: None for name in options}}
        items.append(_item(source, len(items), row["text"], question, gold))
    return items


def fetch_banking77(rows: int) -> list[dict]:
    return _intents("banking77", "legacy-datasets/banking77", "default", "label", rows, set())


def fetch_clinc150(rows: int) -> list[dict]:
    # "oos" marks queries no intent fits; they belong in a no-fit set, not a choice with a right answer.
    return _intents("clinc150", "clinc/clinc_oos", "plus", "intent", rows, {"oos"})


def fetch_trec_fine(rows: int) -> list[dict]:
    """TREC question types: the fine types within the gold one's coarse group (13 numeric types, 22 entity
    types, ...), which differ by little."""
    sample = _everything("SetFit/TREC-QC", "default", "test")
    groups: dict[str, list[str]] = {}
    for row in sample:
        group = groups.setdefault(row["label_coarse_original"], [])
        if row["label_text"] not in group:
            group.append(row["label_text"])
    items = []
    for row in sample[:rows]:
        options = sorted(groups[row["label_coarse_original"]])
        if len(options) < 2:
            continue
        question = {"type": "choice", "instructions": "What kind of answer does this question ask for?",
                    "criteria": {name: None for name in options}}
        items.append(_item("trec_fine", len(items), row["text"], question, row["label_text"]))
    return items


def fetch_mnli(rows: int) -> list[dict]:
    names = list(MNLI)
    question = {"type": "choice", "instructions": "How does the hypothesis relate to the premise?", "criteria": MNLI}
    return [_item("mnli", index, f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}", question,
                  names[row["label"]])
            for index, row in enumerate(_spread("nyu-mll/glue", "mnli", "validation_matched", rows, 20))
            if row["label"] in (0, 1, 2)]


def fetch_goemotions(rows: int) -> list[dict]:
    names = _get_json(f"{ROWS_API}?" + urllib.parse.urlencode({
        "dataset": "google-research-datasets/go_emotions", "config": "simplified", "split": "test",
        "offset": 0, "length": 1,
    }))["features"]
    names = next(f for f in names if f["name"] == "labels")["type"]["feature"]["names"]
    question = {"type": "choice", "instructions": "Which emotion does this comment express most?",
                "criteria": {emotion: None for emotion in EMOTIONS}}
    items = []
    for row in _everything("google-research-datasets/go_emotions", "simplified", "test"):
        labels = [names[index] for index in row["labels"]]
        if len(labels) == 1 and labels[0] in EMOTIONS:
            items.append(_item("goemotions", len(items), row["text"], question, labels[0]))
    step = max(1, len(items) // rows)  # evenly across the split, so no emotion is cut off
    return [{**item, "id": f"goemotions-{index}"} for index, item in enumerate(items[::step][:rows])]


def fetch_chaos_mnli(rows: int) -> list[dict]:
    """ChaosNLI's 1,599 MNLI dev items (Nie et al., 2020), each with 100 human labels, from the tasksource
    mirror. The label is the 100-annotator majority; `human` keeps the counts, so an item's ambiguity is
    measured rather than judged. `rows` is ignored: the set is small and the point is its completeness."""
    names = {"e": "entailment", "n": "neutral", "c": "contradiction"}
    question = {"type": "choice", "instructions": "How does the hypothesis relate to the premise?", "criteria": MNLI}
    items = []
    for row in _everything("tasksource/chaos-mnli-ambiguity", "default", "train"):
        counts = dict(zip(names.values(), row["label_count"]))  # ChaosNLI's order: e, n, c
        item = _item("chaos_mnli", len(items), f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
                     question, names[row["majority_label"]])
        item["human"] = {"counts": counts, "original": names[row["old_label"]]}
        items.append(item)
    return items


FETCHERS = {"boolq": fetch_boolq, "ag_news": fetch_ag_news, "dbpedia": fetch_dbpedia,
            "banking_cards": fetch_banking_cards, "yelp_stars": fetch_yelp,
            "banking77": fetch_banking77, "clinc150": fetch_clinc150, "trec_fine": fetch_trec_fine,
            "mnli": fetch_mnli, "goemotions": fetch_goemotions, "chaos_mnli": fetch_chaos_mnli,
            "jevals_pubmedqa": lambda rows: fetch_jevals("pubmedqa"),
            "jevals_helpsteer2": lambda rows: fetch_jevals("helpsteer2")}


def fetch(args) -> None:
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, fetcher in FETCHERS.items():
        target = directory / f"{name}.jsonl"
        if target.exists():
            continue
        items = fetcher(args.rows)
        target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
        print(f"{name}: {len(items)} questions")
    semif = Path(args.semif) if args.semif else None
    if semif and semif.is_dir():
        semif = semif / "benchmarks" / "data" / "authored144.jsonl"
    target = directory / "semif_authored.jsonl"
    if semif and semif.exists() and not target.exists():
        items = semif_authored(semif)
        target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
        print(f"semif_authored: {len(items)} questions")
    elif not target.exists():
        print("semif_authored: skipped; pass --semif with a SemIf checkout")


# Asking

MODEL = "llav-latest"  # --model changes it, to score another System One server such as CLM
HEADERS = {"Content-Type": "application/json"}  # --api-key-file adds Authorization, for hosted servers
RETRY_STATUSES = (429, 502, 503, 529)  # rate limits and overload on a shared service; worth a wait


def use_model(name: str) -> None:
    global MODEL
    MODEL = name


def use_api_key(path: str | None) -> None:
    """Send the key in `path` as a bearer token. It is read from a file so it never appears in a command line,
    shell history or process list, and it is never printed."""
    if not path:
        return
    key_file = Path(path).expanduser()
    if key_file.stat().st_mode & 0o077:
        print(f"warning: {key_file} can be read by other users; chmod 600 it", file=sys.stderr)
    key = key_file.read_text().strip()
    if not key:
        sys.exit(f"{key_file} is empty")
    HEADERS["Authorization"] = f"Bearer {key}"


def ask(url: str, state, questions: dict) -> tuple[dict, dict]:
    body = json.dumps({"state": state, "model": MODEL, "questions": questions}).encode()
    for attempt in range(6):
        request = urllib.request.Request(url + "/v1/systemone", data=body, headers=HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.load(response), dict(response.headers)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            if error.code in RETRY_STATUSES and attempt < 5:
                time.sleep(int(error.headers.get("Retry-After") or 0) or 5 * (attempt + 1))
                continue
            sys.exit(f"{url} answered {error.code}: {detail}")
    raise RuntimeError("unreachable")


def served_backend(url: str) -> dict:
    """llav's backend details from /v1/models: the model file `fit` checks, and the low-mass cue."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url + "/v1/models", headers=HEADERS), timeout=30) as response:
            backend = json.load(response)["data"][0]["backend"]
            return backend if isinstance(backend, dict) else {}
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return {}  # not llav, or an older one


def load(paths: list[str], limit: int | None) -> list[dict]:
    files = []
    for path in map(Path, paths):
        files += sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    items = []
    for file in files:
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        items += rows[:limit] if limit else rows
    return items


def label_key(item: dict) -> str:
    """The label as a key of the probabilities `probabilities()` returns."""
    label = item["label"]
    if item["question"]["type"] == "noul":
        return "true" if label else "false"
    return str(label)


def probabilities(answer: dict) -> dict:
    """Option key to probability, in declared order; a noul's keys are "true" and "false"."""
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return dict(answer["probabilities"])


def masses(headers: dict, count: int) -> list[float | None]:
    values = headers.get("X-Llav-Candidate-Mass")
    return list(map(float, values.split(","))) if values else [None] * count


# Metrics

def top(probs: dict) -> tuple[str, float]:
    key = max(probs, key=probs.get)
    return key, probs[key]


def metrics(records: list[dict]) -> dict:
    """Accuracy, Brier, NLL, ECE and coverage over records of {"probs": {key: p}, "label": key}."""
    n = len(records)
    tops = [top(record["probs"]) for record in records]
    correct = [key == record["label"] for (key, _), record in zip(tops, records)]
    confidence = [p for _, p in tops]
    brier = statistics.fmean(sum((p - (key == record["label"])) ** 2 for key, p in record["probs"].items())
                             for record in records)
    nll = statistics.fmean(-math.log(max(record["probs"].get(record["label"], 0.0), EPSILON)) for record in records)
    bins = reliability(confidence, correct)
    ece = sum(count * abs(accuracy - mean) for count, mean, accuracy in bins if count) / n
    return {"n": n, "accuracy": sum(correct) / n, "brier": brier, "nll": nll, "ece": ece, "bins": bins,
            "coverage": {target: coverage(confidence, correct, target) for target in ERROR_TARGETS},
            "decision": decision_score(records)}


def _decision_loss(probs: list[float], target: int, ordinal: bool) -> float:
    if not ordinal:
        return sum((p - (index == target)) ** 2 for index, p in enumerate(probs))
    # Ranked probability score: squared gaps between the cumulative distributions, over K - 1 cut points.
    total = cumulative = 0.0
    for index, p in enumerate(probs[:-1]):
        cumulative += p
        total += (cumulative - (index >= target)) ** 2
    return total / (len(probs) - 1)


def decision_score(records: list[dict]) -> float | None:
    """Jevals' Decision Score: 100 x (1 - loss / loss of always answering the label base rates).

    The loss is the multiclass Brier score for noul and choice and the ranked probability score for score,
    whose levels are ordered. None for records of mixed or unknown question types, where it means nothing.
    Checked against Jevals' board: it reproduces Jev's 69.03 (PubMedQA) and 9.20 (HelpSteer2) from their logs.
    """
    kinds = {record.get("type") for record in records}
    if len(kinds) != 1 or None in kinds:
        return None
    ordinal = kinds == {"score"}
    keys = list(records[0]["probs"])
    if ordinal:
        keys.sort(key=int)
    targets = [keys.index(record["label"]) for record in records]
    prior = [targets.count(index) / len(targets) for index in range(len(keys))]
    loss = statistics.fmean(_decision_loss([record["probs"][key] for key in keys], target, ordinal)
                            for record, target in zip(records, targets))
    base = statistics.fmean(_decision_loss(prior, target, ordinal) for target in targets)
    return 100 * (1 - loss / base) if base else None


def reliability(confidence: list[float], correct: list[bool]) -> list[tuple[int, float, float]]:
    """Per equal-width bin: count, mean confidence, accuracy."""
    grouped = [[] for _ in range(BINS)]
    for p, ok in zip(confidence, correct):
        grouped[min(int(p * BINS), BINS - 1)].append((p, ok))
    return [(len(group), statistics.fmean(p for p, _ in group) if group else 0.0,
             statistics.fmean(ok for _, ok in group) if group else 0.0) for group in grouped]


def coverage(confidence: list[float], correct: list[bool], target: float) -> tuple[float, float | None]:
    """Largest share answerable, most confident first, with error rate <= target; and the threshold used.

    A threshold cannot split tied confidences, so a cut is only allowed where the confidence changes.
    """
    order = sorted(range(len(confidence)), key=lambda index: -confidence[index])
    best, threshold, errors = 0, None, 0
    for rank, index in enumerate(order, 1):
        errors += not correct[index]
        boundary = rank == len(order) or confidence[order[rank]] < confidence[index]
        if boundary and errors <= target * rank:
            best, threshold = rank, confidence[index]
    return best / len(confidence), threshold


def geometric_mean(rows: list[dict]) -> dict:
    """Probabilities combined in log space and renormalized: cyclic-shift marginalization."""
    logs = {key: statistics.fmean(math.log(max(row[key], EPSILON)) for row in rows) for key in rows[0]}
    peak = max(logs.values())
    weights = {key: math.exp(value - peak) for key, value in logs.items()}
    total = sum(weights.values())
    return {key: weight / total for key, weight in weights.items()}


# Reporting

def _row(name: str, result: dict) -> str:
    cov = " ".join(f"{100 * result['coverage'][target][0]:5.1f}%" for target in ERROR_TARGETS)
    decision = "     -" if result["decision"] is None else f"{result['decision']:6.1f}"
    return (f"{name:28} {result['n']:5} {100 * result['accuracy']:6.1f}% {result['brier']:6.3f} "
            f"{result['nll']:6.3f} {result['ece']:6.3f}  {cov}  {decision}")


HEADER = (f"{'source':28} {'n':>5} {'acc':>7} {'brier':>6} {'nll':>6} {'ece':>6}  "
          + " ".join(f"{'cov@' + str(round(100 * t)) + '%':>6}" for t in ERROR_TARGETS) + f"  {'dscore':>6}")


def report(groups: dict[str, list[dict]]) -> None:
    print(HEADER)
    for name, records in groups.items():
        print(_row(name, metrics(records)))


def print_thresholds(groups: dict[str, list[dict]]) -> None:
    """The top-answer probability to act at, per source, for each error target: the cut `coverage` found."""
    print("act on answers at or above this probability for at most this error rate (and the share it covers):")
    for name, records in groups.items():
        cuts = []
        for target, (share, threshold) in metrics(records)["coverage"].items():
            cut = "none" if threshold is None else f">= {threshold:.3f} ({100 * share:.0f}%)"
            cuts.append(f"{100 * target:.0f}%: {cut}")
        print(f"  {name:28} " + "   ".join(cuts))


def print_bins(records: list[dict]) -> None:
    print("reliability, all sources (bin: count, mean confidence, accuracy):")
    for index, (count, mean, accuracy) in enumerate(metrics(records)["bins"]):
        if count:
            print(f"  {index / BINS:.1f}-{(index + 1) / BINS:.1f}: {count:5}  {mean:.3f}  {accuracy:.3f}")


def by_source(records: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for record in records:
        groups.setdefault(record["source"], []).append(record)
    if len(groups) > 1:
        groups["all"] = records
    return groups


def score(args) -> None:
    items = load(args.paths, args.limit)
    by_state = {}
    for item in items:  # questions about the same state go in one request, as a caller would send them
        by_state.setdefault(json.dumps(item["state"], sort_keys=True), []).append(item)
    records, started = [], time.perf_counter()
    backend = served_backend(args.url)
    model_file = backend.get("model_file")
    for group in by_state.values():
        body, headers = ask(args.url, group[0]["state"], {f"q{i}": item["question"] for i, item in enumerate(group)})
        answers = list(body["answers"].values())
        for item, answer, mass in zip(group, answers, masses(headers, len(group))):
            records.append({"id": item["id"], "source": item["source"], "type": item["question"]["type"],
                            "label": label_key(item), "probs": probabilities(answer), "mass": mass,
                            "model": body["model"], "model_file": model_file,
                            "calibration": headers.get("X-Llav-Calibration")})
        if args.progress:
            print(f"\r{len(records)}/{len(items)}", end="", file=sys.stderr, flush=True)
    if args.progress:
        print(file=sys.stderr)
    print(f"{len(records)} questions in {time.perf_counter() - started:.0f} s\n")
    report(by_source(records))
    print()
    print_thresholds(by_source(records))
    print()
    print_bins(records)
    print_mass(records, backend.get("low_candidate_mass", LOW_MASS))
    if args.out:
        Path(args.out).write_text("".join(json.dumps(record) + "\n" for record in records))


def print_mass(records: list[dict], low_mass: float) -> None:
    known = [record for record in records if record["mass"] is not None]
    if not known:
        print("candidate mass not reported (not llav, or an llav without X-Llav-Candidate-Mass)")
        return
    low = [record for record in known if record["mass"] < low_mass]
    high = [record for record in known if record["mass"] >= low_mass]
    print(f"candidate mass: median {statistics.median(r['mass'] for r in known):.4f}, "
          f"min {min(r['mass'] for r in known):.4f}; below {low_mass}: {len(low)} questions")
    for name, group in (("below", low), ("at or above", high)):
        if group:
            accuracy = statistics.fmean(top(record["probs"])[0] == record["label"] for record in group)
            print(f"  accuracy {name} {low_mass}: {100 * accuracy:.1f}% of {len(group)}")


# Option order

def rotations(question: dict, limit: int) -> list[tuple[dict, list[str]]]:
    """Each cyclic rotation of the options, with the original key at each displayed position.

    Choice keys are the option keys. Score levels are keyed by their original index, so a rotated answer's
    level index maps back through the list.
    """
    criteria = question["criteria"]
    keys = list(criteria) if question["type"] == "choice" else [str(i) for i in range(len(criteria))]
    shifted = []
    for shift in range(min(len(keys), limit)):
        order = keys[shift:] + keys[:shift]
        if question["type"] == "choice":
            rotated = {**question, "criteria": {key: criteria[key] for key in order}}
        else:
            rotated = {**question, "criteria": [criteria[int(key)] for key in order]}
        shifted.append((rotated, order))
    return shifted


def unrotate(answer: dict, order: list[str]) -> dict:
    """Probabilities keyed by the original option keys."""
    if answer["type"] == "choice":
        return dict(answer["probabilities"])
    return {order[int(index)]: p for index, p in answer["probabilities"].items()}


def shifts(args) -> None:
    items = [item for item in load(args.paths, args.limit) if item["question"]["type"] != "noul"]
    results, started = [], time.perf_counter()
    for count, item in enumerate(items, 1):
        variants = rotations(item["question"], args.max_shifts)
        body, _ = ask(args.url, item["state"], {f"r{i}": question for i, (question, _) in enumerate(variants)})
        rows = [unrotate(answer, order) for answer, (_, order) in zip(body["answers"].values(), variants)]
        positions = [order.index(top(row)[0]) for row, (_, order) in zip(rows, variants)]
        results.append({"id": item["id"], "source": item["source"], "label": label_key(item), "rows": rows,
                        "positions": positions, "options": len(variants[0][1])})
        if args.progress:
            print(f"\r{count}/{len(items)}", end="", file=sys.stderr, flush=True)
    if args.progress:
        print(file=sys.stderr)
    print(f"{len(results)} questions, {sum(len(r['rows']) for r in results)} rotations, "
          f"{time.perf_counter() - started:.0f} s\n")
    groups = by_source(results)
    print(f"{'source':28} {'n':>5} {'K':>3} {'flip':>6} {'vs first':>8} {'shift med':>9} {'max':>5} "
          f"{'first pos':>9} {'last pos':>8}")
    for name, group in groups.items():
        print(order_row(name, group))
    print("\nfirst position: share of answers that picked whatever option was listed first; with every rotation "
          "asked, an order-blind model scores 1/K there")
    for label, pick in (("caller's order", lambda r: r["rows"][0]),
                        ("every rotation", None),
                        ("geometric mean", lambda r: geometric_mean(r["rows"]))):
        print(f"\n{label}:")
        if pick is None:  # each rotation counted as its own answer
            report({name: [{"probs": row, "label": r["label"]} for r in group for row in r["rows"]]
                    for name, group in groups.items()})
        else:
            report({name: [{"probs": pick(r), "label": r["label"]} for r in group] for name, group in groups.items()})
    if args.out:
        Path(args.out).write_text("".join(json.dumps(result) + "\n" for result in results))


def order_row(name: str, group: list[dict]) -> str:
    flips = [len({top(row)[0] for row in r["rows"]}) > 1 for r in group]
    # Answers that differ from the caller's order, over all the other rotations.
    against = [top(row)[0] != top(r["rows"][0])[0] for r in group for row in r["rows"][1:]]
    spread = [max(max(row[key] for row in r["rows"]) - min(row[key] for row in r["rows"]) for key in r["rows"][0])
              for r in group]
    picks = [p for r in group for p in r["positions"]]
    last = [p == r["options"] - 1 for r in group for p in r["positions"]]
    sizes = {r["options"] for r in group}
    k = str(sizes.pop()) if len(sizes) == 1 else "mix"
    return (f"{name:28} {len(group):5} {k:>3} {100 * statistics.fmean(flips):5.1f}% "
            f"{100 * statistics.fmean(against) if against else 0:7.1f}% {statistics.median(spread):9.3f} "
            f"{max(spread):5.2f} {100 * statistics.fmean(p == 0 for p in picks):8.1f}% "
            f"{100 * statistics.fmean(last):7.1f}%")


# Calibration

def _nll(records: list[dict], temperature: float, apply) -> float:
    total = 0.0
    for record in records:
        keys = list(record["probs"])
        calibrated = apply([record["probs"][key] for key in keys], temperature)
        total -= math.log(max(calibrated[keys.index(record["label"])], EPSILON))
    return total / len(records)


def fit_temperature(records: list[dict], apply) -> float:
    """The temperature minimizing NLL. NLL is convex in 1/T, so a golden-section search on log T finds it."""
    low, high = math.log(T_LIMITS[0]), math.log(T_LIMITS[1])
    ratio = (math.sqrt(5) - 1) / 2
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    fa, fb = _nll(records, math.exp(a), apply), _nll(records, math.exp(b), apply)
    for _ in range(60):
        if fa < fb:
            high, b, fb = b, a, fa
            a = high - ratio * (high - low)
            fa = _nll(records, math.exp(a), apply)
        else:
            low, a, fa = a, b, fb
            b = low + ratio * (high - low)
            fb = _nll(records, math.exp(b), apply)
    return math.exp((low + high) / 2)


def fit_types(records: list[dict], apply) -> tuple[dict[str, float], dict[str, str]]:
    """Temperatures per question type, and why each type without one was left at T = 1."""
    by_type = {}
    for record in records:
        by_type.setdefault(record["type"], []).append(record)
    temperatures, skipped = {}, {}
    for kind, group in by_type.items():
        errors = sum(top(record["probs"])[0] != record["label"] for record in group)
        if len(group) < MIN_FIT:
            skipped[kind] = f"{len(group)} questions, fewer than {MIN_FIT}"
        elif errors < MIN_ERRORS:
            skipped[kind] = f"{errors} wrong answers, fewer than {MIN_ERRORS}: too few to fit a temperature"
        else:
            value = fit_temperature(group, apply)
            if not T_LIMITS[0] * 1.05 < value < T_LIMITS[1] / 1.05:
                skipped[kind] = f"the fit ran to its limit (T = {value:.3f})"
            else:
                temperatures[kind] = value
    return temperatures, skipped


def calibrated(record: dict, temperatures: dict, apply) -> dict:
    keys = list(record["probs"])
    values = apply([record["probs"][key] for key in keys], temperatures.get(record["type"], 1.0))
    return {**record, "probs": dict(zip(keys, values))}


def fit(args) -> None:
    sys.path.insert(0, str(SRC))
    from llav.calibration import FORMAT, apply, file_sha256  # noqa: PLC0415 - llav is only needed here
    from llav.prompt import PROMPT_VERSION  # noqa: PLC0415

    records = [json.loads(line) for path in args.predictions for line in Path(path).read_text().splitlines()
               if line.strip()]
    if any(record.get("calibration") not in (None, "none") for record in records):
        sys.exit("these predictions were already calibrated; score again against llav without --calibration")
    gguf = Path(args.gguf)
    models = {record["model"] for record in records if record.get("model")} or {"an unnamed model"}
    named = [record.get("model_file") for record in records]
    files = set(named) - {None}
    if files and files != {gguf.name}:
        sys.exit(f"these predictions came from {', '.join(sorted(files))}, not {gguf.name}; "
                 "a temperature fitted on one model does not transfer to another")
    unnamed = named.count(None)
    if unnamed:
        print(f"warning: {unnamed} of {len(records)} predictions do not name their model file (scored before "
              f"`score` recorded it); make sure they came from {gguf.name}")
    counts = {}
    for record in records:
        counts[record["type"]] = counts.get(record["type"], 0) + 1
    print(f"{len(records)} predictions from {', '.join(sorted(map(str, models)))}; "
          + ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items())))

    # Held out: each half calibrated with temperatures fitted on the other half.
    folds = ([], [])
    for record in records:
        folds[hashlib.sha256(record["id"].encode()).digest()[0] % 2].append(record)
    held_out, reasons = [], {}
    fold_fits = [fit_types(fold, apply) for fold in folds]
    for index in (0, 1):
        temperatures, skipped = fold_fits[1 - index]
        reasons.update({kind: f"not fittable on half the data: {reason}" for kind, reason in skipped.items()})
        held_out += [calibrated(record, temperatures, apply) for record in folds[index]]
    # Only a type fitted on both halves has been checked on data it was not fitted on.
    checked = set(fold_fits[0][0]) & set(fold_fits[1][0])
    before, after = by_source(records), by_source(held_out)
    # No coverage column: the two halves carry different temperatures, so ranking them together would
    # measure the split, not the calibration. One temperature barely changes coverage; `score` measures it.
    print("\nheld out (fitted on the other half), before -> after:")
    print(f"{'source':28} {'n':>5} {'ece':>15} {'nll':>15} {'brier':>15}")
    for name in before:
        b, a = metrics(before[name]), metrics(after[name])
        print(f"{name:28} {b['n']:5} {b['ece']:6.3f} -> {a['ece']:5.3f} {b['nll']:6.3f} -> {a['nll']:5.3f} "
              f"{b['brier']:6.3f} -> {a['brier']:5.3f}")

    fitted, skipped = fit_types(records, apply)
    reasons.update(skipped)
    temperatures = {kind: value for kind, value in fitted.items() if kind in checked}
    print()
    for kind in sorted(counts):
        if kind in temperatures:
            print(f"{kind}: T = {temperatures[kind]:.3f}")
        else:
            print(f"{kind}: left at T = 1 ({reasons.get(kind, 'not fitted on both halves')})")
    if not temperatures:
        sys.exit("no question type could be calibrated; nothing written")
    document = {
        "format": FORMAT,
        "prompt_version": PROMPT_VERSION,
        "model": {"file": gguf.name, "sha256": file_sha256(gguf)},
        "temperature": {kind: round(value, 4) for kind, value in sorted(temperatures.items())},
        "fitted": {"date": time.strftime("%Y-%m-%d"), "questions": counts, "served_model": sorted(map(str, models)),
                   "sources": sorted({record["source"] for record in records})},
    }
    Path(args.out).write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {args.out}; start llav with --calibration {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("fetch", help="Download the labelled sources (needs network, not llav)")
    command.add_argument("dir", help="Directory to write one JSONL file per source into")
    command.add_argument("--rows", type=int, default=200, help="Questions per public source")
    command.add_argument("--semif", help="SemIf checkout or its authored144.jsonl, for the owned labelled set")
    command.set_defaults(run=fetch)
    for name, function, help_text in (("score", score, "Accuracy, Brier, NLL, ECE and coverage"),
                                      ("shifts", shifts, "Every cyclic rotation of each question's options")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("url", help="llav base URL, for example http://127.0.0.1:8080")
        command.add_argument("paths", nargs="+", help="JSONL files, or directories of them")
        command.add_argument("--limit", type=int, help="At most this many questions per file")
        command.add_argument("--out", help="Write per-question results as JSONL, for later analysis")
        command.add_argument("--progress", action="store_true", help="Show a counter on stderr")
        command.add_argument("--model", default=MODEL, help="Model name to request (default: llav-latest)")
        command.add_argument("--api-key-file", help="File holding an API key, sent as a bearer token; for a "
                                                   "hosted server such as OpenRouter's Jev")
        if name == "shifts":
            command.add_argument("--max-shifts", type=int, default=26, help="At most this many rotations")
        command.set_defaults(run=function)
    command = commands.add_parser("fit", help="Fit a temperature per question type from score --out predictions")
    command.add_argument("predictions", nargs="+", help="JSONL written by score --out, from an uncalibrated llav")
    command.add_argument("--gguf", required=True, help="The model file the predictions came from; its SHA-256 "
                                                       "goes into the calibration file")
    command.add_argument("--out", required=True, help="Calibration file to write")
    command.set_defaults(run=fit)
    args = parser.parse_args(argv)
    use_model(getattr(args, "model", MODEL))
    use_api_key(getattr(args, "api_key_file", None))
    args.run(args)


if __name__ == "__main__":
    main()
