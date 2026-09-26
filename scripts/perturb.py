#!/usr/bin/env python3
"""Does disagreement under option permutations predict wrong answers? The experiment in
agent_docs/experiments/permutation-uncertainty.md.

    scripts/perturb.py run     http://127.0.0.1:8080 ~/llav-eval --out perturb.jsonl [--limit 100]
    scripts/perturb.py analyze perturb.jsonl

Further subcommands, each documented in its function: `h2` (the flip effect adjusted for confidence and
source), `human` (against ChaosNLI's and GoEmotions' human label counts), `pride` (PriDe's id-prior
debiasing from the recorded rotations, against averaging), `text` and `text-analyze` (the same items
answered by generation through llama-server directly), `nofit` and `nofit-analyze` (candidate mass with
the gold option removed), and `run --reword` (the same base order under paraphrased criteria).

`run` asks every choice question in several option orders and records each answer. The caller's order is
drawn at random per question (seeded by `--seed` and the question id), so a dataset's own option order is
never the reference. For each question it sends:

- `repeat`: the base order `--repeats` times. Identical prompts; any spread is numeric noise, the floor that
  permutation spread has to clear.
- `rotation`: every other cyclic rotation of the base order, at most `--rotations`.
- `perm`: `--perms` random permutations other than the base order.

Requests carry at most MAX_PER_REQUEST questions, the native helper's batch, so a run stays on one readout
path instead of falling back to llama-server for large requests; `repeat` and `perm` share one request,
`rotation` goes in another. Nouls are skipped (llav fixes their order), and so are scores, whose levels have
an order that permuting breaks. Only the System One API is used; the prompt format is llav's own.

`run` resumes: rerun with the same arguments after a crash and it keeps the complete lines already in `--out`
and asks only the questions missing from it.

`analyze` reads one or more `run` outputs and reports: the noise floor against permutation spread; how well
each signal separates wrong answers from right ones (AUROC, errors positive) for the base-order answer and
for the answer averaged over permutations; the gain from adding permutation spread to the averaged answer's
confidence, with a paired bootstrap interval and a leave-one-source-out logistic fit; and wrong-answer rates
of confident answers that do and do not change under permutation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402  (shared loading and the System One client)
sys.path.insert(0, str(evaluate.SRC))
from llav.prompt import LABELS  # noqa: E402
from llav.questions import _align_letters  # noqa: E402  (the order llav actually displays)

MAX_PER_REQUEST = 16  # native.py's max_questions default: larger requests leave the helper
CONFIDENT = 0.9
BOOTSTRAP = 2000


# Run

def variants(item: dict, seed: int, repeats: int, rotations: int, perms: int,
             rewords: int = 0) -> list[list[tuple[str, list[str]]]]:
    """Requests for one question, each a list of (kind, option order). With `rewords`, a further request asks
    the base order under each of that many paraphrased criteria (kind "reword:<index>"), the control for
    whether any meaning-preserving change flags the same answers that reordering does."""
    keys = list(item["question"]["criteria"])
    rng = random.Random(f"{seed}:{item['id']}")
    base = rng.sample(keys, len(keys))
    first = [("repeat", base)] * repeats
    seen = {tuple(base)}
    for _ in range(perms):
        if len(seen) == math.factorial(len(keys)):
            break
        order = rng.sample(keys, len(keys))
        while tuple(order) in seen:
            order = rng.sample(keys, len(keys))
        seen.add(tuple(order))
        first.append(("perm", order))
    second = [("rotation", base[shift:] + base[:shift]) for shift in range(1, min(len(keys), rotations + 1))]
    third = [(f"reword:{index}", base) for index in range(rewords)]
    return [request for request in (first, second, third) if request]


def run(args) -> None:
    items = [item for item in evaluate.load(args.paths, args.limit) if item["question"]["type"] == "choice"]
    rewords = json.loads(Path(args.reword).read_text()) if args.reword else {}
    if rewords:  # only sources with paraphrases take part in the control
        items = [item for item in items if item["source"] in rewords]
    done = set()
    if Path(args.out).exists():  # resume: keep complete lines, ask only what is missing
        lines = Path(args.out).read_text().splitlines(keepends=True)
        complete = [line for line in lines if line.endswith("\n")]
        Path(args.out).write_text("".join(complete))  # drops a line cut short by a crash
        done = {json.loads(line)["id"] for line in complete}
        items = [item for item in items if item["id"] not in done]
        print(f"resuming: {len(done)} questions already in {args.out}")
    backend = evaluate.served_backend(args.url)
    started = time.perf_counter()
    with open(args.out, "a") as out:
        for count, item in enumerate(items, 1):
            answers = []
            paraphrases = rewords.get(item["source"], [])
            for request in variants(item, args.seed, args.repeats, args.rotations, args.perms, len(paraphrases)):
                if len(request) > MAX_PER_REQUEST:
                    sys.exit(f"{item['id']}: {len(request)} variants exceed {MAX_PER_REQUEST} per request")
                criteria = item["question"]["criteria"]
                questions = {}
                for i, (kind, order) in enumerate(request):
                    questions[f"v{i}"] = {**item["question"], "criteria": {key: criteria[key] for key in order}}
                    if kind.startswith("reword:"):
                        questions[f"v{i}"]["instructions"] = paraphrases[int(kind.split(":")[1])]
                body, headers = evaluate.ask(args.url, item["state"], questions)
                masses = evaluate.masses(headers, len(request))
                for (kind, order), answer, mass in zip(request, body["answers"].values(), masses):
                    answers.append({"kind": kind, "order": order, "probs": dict(answer["probabilities"]),
                                    "mass": mass})
            out.write(json.dumps({"id": item["id"], "source": item["source"], "label": evaluate.label_key(item),
                                  "answers": answers, "backend": backend}) + "\n")
            out.flush()
            if args.progress:
                print(f"\r{count}/{len(items)}", end="", file=sys.stderr, flush=True)
    if args.progress:
        print(file=sys.stderr)
    print(f"{len(items)} questions in {time.perf_counter() - started:.0f} s -> {args.out}")


# Analysis

def _vector(probs: dict, keys: list[str]) -> list[float]:
    return [probs[key] for key in keys]


def _tv(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def _spread(vectors: list[list[float]]) -> float:
    """Mean total-variation distance of each answer from the answers' mean."""
    mean = [statistics.fmean(column) for column in zip(*vectors)]
    return statistics.fmean(_tv(v, mean) for v in vectors)


def _argmax(v: list[float]) -> int:
    return max(range(len(v)), key=v.__getitem__)


def _changes(vectors: list[list[float]]) -> float:
    """Share of answers that differ from the most common pick."""
    picks = [_argmax(v) for v in vectors]
    return 1 - max(picks.count(pick) for pick in set(picks)) / len(picks)


def features(record: dict) -> dict:
    keys = sorted(record["answers"][0]["probs"])
    base = [a for a in record["answers"] if a["kind"] == "repeat"]
    reworded = [a for a in record["answers"] if a["kind"].startswith("reword:")]
    permuted = [a for a in record["answers"] if a["kind"] != "repeat" and not a["kind"].startswith("reword:")]
    first = _vector(base[0]["probs"], keys)
    family = [first] + [_vector(a["probs"], keys) for a in permuted]  # every distinct order, base included
    mean = [statistics.fmean(column) for column in zip(*family)]
    top, runner = sorted(first, reverse=True)[:2]
    mtop, mrunner = sorted(mean, reverse=True)[:2]
    label = keys.index(record["label"])
    masses = [a["mass"] for a in base if a["mass"] is not None]
    return {
        "id": record["id"], "source": record["source"], "options": len(keys),
        "wrong": _argmax(first) != label, "wrong_avg": _argmax(mean) != label,
        "conf": top, "margin": top - runner, "entropy": -sum(p * math.log(p) for p in first if p > 0),
        "mass": masses[0] if masses else None,
        "conf_avg": mtop, "margin_avg": mtop - mrunner,
        "spread": _spread(family), "changes": _changes(family),
        # The same flip test on a fixed budget, the base order and the random permutations: counting every
        # rotation gives a 12-option question more chances to flip than a 3-option one.
        "changes_fixed": _changes([first] + [_vector(a["probs"], keys) for a in permuted if a["kind"] == "perm"]),
        # The control: does the answer change under a paraphrased criterion, same order?
        "changes_reword": _changes([first] + [_vector(a["probs"], keys) for a in reworded]) if reworded else None,
        "spread_reword": _spread([first] + [_vector(a["probs"], keys) for a in reworded]) if reworded else None,
        "noise": _spread([_vector(a["probs"], keys) for a in base]) if len(base) > 1 else None,
        "noise_changes": _changes([_vector(a["probs"], keys) for a in base]) if len(base) > 1 else None,
        "passes": len(family),
    }


def auroc(scores: list[float], wrong: list[bool]) -> float:
    """Chance a wrong answer scores higher than a right one; ties count half (rank-sum form)."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        for k in range(i, j):
            ranks[order[k]] = (i + j + 1) / 2
        i = j
    positives = sum(wrong)
    negatives = len(wrong) - positives
    if not positives or not negatives:
        return float("nan")
    return (sum(r for r, w in zip(ranks, wrong) if w) - positives * (positives + 1) / 2) / (positives * negatives)


# Each signal oriented so that higher means "more likely wrong".
SIGNALS = {
    "1-conf": lambda f: -f["conf"], "-margin": lambda f: -f["margin"], "entropy": lambda f: f["entropy"],
    "1-mass": lambda f: -(f["mass"] if f["mass"] is not None else 1.0),
    "1-conf_avg": lambda f: -f["conf_avg"], "-margin_avg": lambda f: -f["margin_avg"],
    "spread": lambda f: f["spread"], "changes": lambda f: f["changes"],
}


def _standardize(columns: list[list[float]]) -> list[list[float]]:
    out = []
    for column in columns:
        mean, sd = statistics.fmean(column), statistics.pstdev(column) or 1.0
        out.append([(x - mean) / sd for x in column])
    return out


def logistic(rows: list[list[float]], y: list[bool], iterations: int = 30) -> list[float]:
    """Newton-Raphson logistic regression with an intercept and a small ridge, standard library only."""
    x = [[1.0] + row for row in rows]
    n = len(x[0])
    w = [0.0] * n
    for _ in range(iterations):
        grad = [0.0] * n
        hess = [[1e-3 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for xi, yi in zip(x, y):
            z = sum(a * b for a, b in zip(w, xi))
            p = 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
            for i in range(n):
                grad[i] += (p - yi) * xi[i]
                for j in range(n):
                    hess[i][j] += p * (1 - p) * xi[i] * xi[j]
        step = _solve(hess, grad)
        w = [a - b for a, b in zip(w, step)]
        if max(map(abs, step)) < 1e-8:
            break
    return w


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [value] for row, value in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col and m[col][col]:
                factor = m[r][col] / m[col][col]
                m[r] = [a - factor * b for a, b in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] if m[i][i] else 0.0 for i in range(n)]


def held_out_scores(rows: list[dict], names: list[str], target: str) -> list[float]:
    """Each source scored by a logistic model fitted on the other sources, so no weight sees its own task."""
    columns = _standardize([[SIGNALS[name](row) for row in rows] for name in names])
    matrix = [list(values) for values in zip(*columns)]
    scores = [0.0] * len(rows)
    for source in {row["source"] for row in rows}:
        train = [i for i, row in enumerate(rows) if row["source"] != source]
        w = logistic([matrix[i] for i in train], [rows[i][target] for i in train])
        for i, row in enumerate(rows):
            if row["source"] == source:
                scores[i] = w[0] + sum(a * b for a, b in zip(w[1:], matrix[i]))
    return scores


def bootstrap_gain(rows: list[dict], target: str, base: str, added: str) -> tuple[float, float, float]:
    """AUROC of held-out [base, added] minus AUROC of [base] alone, with a paired 95% bootstrap interval
    stratified by source. The models are fitted once; resampling covers the evaluation questions."""
    wrong = [row[target] for row in rows]
    alone = held_out_scores(rows, [base], target)
    joint = held_out_scores(rows, [base, added], target)
    point = auroc(joint, wrong) - auroc(alone, wrong)
    by_source: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_source.setdefault(row["source"], []).append(i)
    rng = random.Random(0)
    gains = []
    for _ in range(BOOTSTRAP):
        sample = [rng.choice(indexes) for indexes in by_source.values() for _ in indexes]
        w = [wrong[i] for i in sample]
        gains.append(auroc([joint[i] for i in sample], w) - auroc([alone[i] for i in sample], w))
    gains = sorted(g for g in gains if not math.isnan(g))
    return point, gains[int(0.025 * len(gains))], gains[int(0.975 * len(gains)) - 1]


def fixed_order(record: dict) -> bool:
    """True when every variant reaches the model in the same displayed order: a single-letter key is shown at
    its own letter (llav's `_align_letters`), so such questions cannot be permuted."""
    return len({tuple(_align_letters(answer["order"])) for answer in record["answers"]}) == 1


def analyze(args) -> None:
    records = [json.loads(line) for path in args.files for line in Path(path).read_text().splitlines()
               if line.strip()]
    skipped = [r for r in records if fixed_order(r)]
    rows = [features(r) for r in records if not fixed_order(r)]
    if skipped:
        print(f"left out {len(skipped)} questions whose single-letter keys fix the displayed order")
    print(f"{len(rows)} questions, {sum(r['wrong'] for r in rows)} wrong in the base order, "
          f"{sum(r['wrong_avg'] for r in rows)} wrong averaged over {statistics.median(r['passes'] for r in rows):.0f} "
          f"orders (median)\n")

    noisy = [r for r in rows if r["noise"] is not None]
    if noisy:
        print("noise floor: identical prompts against permuted ones (median, mean, share of questions > 0)")
        for name, key in (("identical, spread", "noise"), ("permuted, spread", "spread"),
                          ("identical, answer changes", "noise_changes"), ("permuted, answer changes", "changes")):
            values = [r[key] for r in noisy]
            print(f"  {name:28} {statistics.median(values):.4f}  {statistics.fmean(values):.4f}  "
                  f"{sum(v > 0 for v in values) / len(values):6.1%}")
        print()

    sources = sorted({r["source"] for r in rows})
    for target, title in (("wrong", "base-order answer"), ("wrong_avg", "answer averaged over orders")):
        print(f"AUROC for a wrong {title} (0.5 = chance)")
        print(f"  {'source':30} {'n':>5} {'err':>6} " + " ".join(f"{name:>11}" for name in SIGNALS))
        for source in sources + ["all"]:
            group = rows if source == "all" else [r for r in rows if r["source"] == source]
            wrong = [r[target] for r in group]
            print(f"  {source:30} {len(group):5d} {sum(wrong) / len(group):6.1%} " + " ".join(
                f"{auroc([signal(r) for r in group], wrong):11.3f}" for signal in SIGNALS.values()))
        print()

    print("H1: does spread add to the averaged answer's confidence? held-out AUROC gain, 95% bootstrap")
    for target in ("wrong", "wrong_avg"):
        for base in ("1-conf_avg", "-margin_avg"):
            point, low, high = bootstrap_gain(rows, target, base, "spread")
            print(f"  {target:10} {base:12} + spread: {point:+.4f}  [{low:+.4f}, {high:+.4f}]")
    print()

    with_reword = [r for r in rows if r["changes_reword"] is not None]
    if with_reword:
        print("Reworded-criterion control (sources with paraphrases): wrong-answer rate by which perturbation "
              "changes the answer")
        wrong = [r["wrong"] for r in with_reword]
        signal = lambda key: auroc([float(r[key] > 0) for r in with_reword], wrong)
        print(f"  AUROC for a wrong answer: order flip {signal('changes_fixed'):.3f}, "
              f"reword flip {signal('changes_reword'):.3f}, "
              f"order spread {auroc([r['spread'] for r in with_reword], wrong):.3f}, "
              f"reword spread {auroc([r['spread_reword'] for r in with_reword], wrong):.3f}")
        confident = [r for r in with_reword if r["conf"] >= CONFIDENT]
        for title, subset in (("all answers", with_reword), (f"confidence >= {CONFIDENT}", confident)):
            print(f"  {title}:")
            for o in (0, 1):
                for w in (0, 1):
                    cell = [r for r in subset if (r["changes_fixed"] > 0) == o and (r["changes_reword"] > 0) == w]
                    label = f"order {'flips ' if o else 'stable'}, reword {'flips ' if w else 'stable'}"
                    rate = statistics.fmean(r["wrong"] for r in cell) if cell else float("nan")
                    print(f"    {label}: n={len(cell):5d}  wrong {rate:6.1%}")
        print()
    print(f"H2: base-order answers with confidence >= {CONFIDENT}: wrong-answer rate by whether any order changes it")
    confident = [r for r in rows if r["conf"] >= CONFIDENT]
    for name, group in (("stable", [r for r in confident if r["changes"] == 0]),
                        ("changes", [r for r in confident if r["changes"] > 0])):
        wrong = sum(r["wrong"] for r in group)
        print(f"  {name:8} n={len(group):5d}  wrong {wrong:4d}  ({wrong / max(len(group), 1):.1%})")


# H2 adjusted: does a flip predict a wrong confident answer beyond confidence and source?

def _logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def _spline(values: list[float], knots: list[float]) -> list[list[float]]:
    """Restricted cubic spline basis (linear beyond the outer knots), Harrell's form."""
    k = len(knots)
    scale = (knots[-1] - knots[0]) ** 2
    last, inner = knots[k - 1], knots[k - 2]
    cube = lambda v: max(v, 0.0) ** 3
    rows = []
    for x in values:
        row = [x]
        for j in range(k - 2):
            row.append((cube(x - knots[j]) - cube(x - inner) * (last - knots[j]) / (last - inner)
                        + cube(x - last) * (inner - knots[j]) / (last - inner)) / scale)
        rows.append(row)
    return rows


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _design(rows: list[dict], flip: str, sources: list[str], knots: list[float]) -> list[list[float]]:
    spline = _spline([_logit(r["conf"]) for r in rows], knots)
    return [basis + [float(r[flip] > 0)] + [float(r["source"] == s) for s in sources[1:]]
            for basis, r in zip(spline, rows)]


def _predict(w: list[float], row: list[float]) -> float:
    z = w[0] + sum(a * b for a, b in zip(w[1:], row))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def adjusted_effect(rows: list[dict], flip: str, knots: list[float]) -> tuple[float, float, float]:
    """Log-odds of the flip term, and the marginal risk ratio and difference by standardization: every
    answer predicted as flipped and as stable, averaged."""
    sources = sorted({r["source"] for r in rows})
    x = _design(rows, flip, sources, knots)
    w = logistic(x, [r["wrong"] for r in rows])
    at = len(knots) - 1  # index of the flip column in a design row
    flipped = statistics.fmean(_predict(w, row[:at] + [1.0] + row[at + 1:]) for row in x)
    stable = statistics.fmean(_predict(w, row[:at] + [0.0] + row[at + 1:]) for row in x)
    return w[1 + at], flipped / stable, flipped - stable


def mantel_haenszel(rows: list[dict], flip: str, strata) -> tuple[float, float, float]:
    """Common odds ratio of wrong given flip over strata, with the Robins-Breslow-Greenland 95% interval."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(strata(r), []).append(r)
    sr = ss = srp = srq_sp = ssq = 0.0
    for group in groups.values():
        n = len(group)
        a = sum(1 for r in group if r[flip] > 0 and r["wrong"])
        b = sum(1 for r in group if r[flip] > 0 and not r["wrong"])
        c = sum(1 for r in group if r[flip] == 0 and r["wrong"])
        d = sum(1 for r in group if r[flip] == 0 and not r["wrong"])
        if n < 2:
            continue
        rk, sk = a * d / n, b * c / n
        pk, qk = (a + d) / n, (b + c) / n
        sr, ss = sr + rk, ss + sk
        srp += pk * rk
        srq_sp += qk * rk + pk * sk
        ssq += qk * sk
    if not sr or not ss:
        return float("nan"), float("nan"), float("nan")
    odds = sr / ss
    var = srp / (2 * sr ** 2) + srq_sp / (2 * sr * ss) + ssq / (2 * ss ** 2)
    return odds, math.exp(math.log(odds) - 1.96 * math.sqrt(var)), math.exp(math.log(odds) + 1.96 * math.sqrt(var))


def h2(args) -> None:
    records = [json.loads(line) for path in args.files for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [features(r) for r in records if not fixed_order(r)]
    confident = [r for r in rows if r["conf"] >= CONFIDENT]
    logits = [_logit(r["conf"]) for r in confident]
    knots = [_quantile(logits, q) for q in (0.05, 0.35, 0.65, 0.95)]
    print(f"{len(confident)} base-order answers with confidence >= {CONFIDENT}; spline knots on logit(confidence) "
          + ", ".join(f"{k:.2f}" for k in knots) + "\n")
    for flip, title in (("changes", "flip across every order"),
                        ("changes_fixed", "flip across base + random permutations")):
        flipped = [r for r in confident if r[flip] > 0]
        stable = [r for r in confident if r[flip] == 0]
        raw_f = sum(r["wrong"] for r in flipped) / len(flipped)
        raw_s = sum(r["wrong"] for r in stable) / len(stable)
        print(f"{title}: flipped {len(flipped)} ({raw_f:.1%} wrong), stable {len(stable)} ({raw_s:.1%} wrong), "
              f"raw risk ratio {raw_f / raw_s:.2f}")
        beta, ratio, diff = adjusted_effect(confident, flip, knots)
        rng = random.Random(0)
        by_source: dict = {}
        for r in confident:
            by_source.setdefault(r["source"], []).append(r)
        boots = []
        for _ in range(args.bootstrap):
            sample = [rng.choice(group) for group in by_source.values() for _ in group]
            boots.append(adjusted_effect(sample, flip, knots))
        interval = lambda values: (sorted(values)[int(0.025 * len(values))],
                                   sorted(values)[int(0.975 * len(values)) - 1])
        (ol, oh), (rl, rh), (dl, dh) = (interval([math.exp(b[0]) for b in boots]), interval([b[1] for b in boots]),
                                        interval([b[2] for b in boots]))
        print(f"  adjusted (spline of logit confidence + source): "
              f"odds ratio {math.exp(beta):.2f} [{ol:.2f}, {oh:.2f}], "
              f"risk ratio {ratio:.2f} [{rl:.2f}, {rh:.2f}], risk difference {diff:+.3f} [{dl:+.3f}, {dh:+.3f}]"
              f"  ({args.bootstrap} bootstrap resamples within source)")
        deciles = [_quantile([r["conf"] for r in confident], q / 10) for q in range(1, 10)]
        band = lambda r: sum(r["conf"] >= d for d in deciles)
        odds, low, high = mantel_haenszel(confident, flip, lambda r: (r["source"], band(r)))
        print(f"  Mantel-Haenszel over source x confidence decile: odds ratio {odds:.2f} [{low:.2f}, {high:.2f}]")
        loso = []
        for source in sorted(by_source):
            rest = [r for r in confident if r["source"] != source]
            b, ratio_rest, _ = adjusted_effect(rest, flip, knots)
            loso.append((source, math.exp(b), ratio_rest))
        print("  leave one source out, adjusted odds ratio / risk ratio: "
              + ", ".join(f"-{s} {o:.2f}/{rr:.2f}" for s, o, rr in loso))
        print("  per source, Mantel-Haenszel over confidence deciles (sources with 10+ flips):")
        for source, group in sorted(by_source.items()):
            if sum(r[flip] > 0 for r in group) >= 10:
                odds, low, high = mantel_haenszel(group, flip, band)
                flips = sum(r[flip] > 0 for r in group)
                print(f"    {source:30} flips {flips:4d}  odds ratio {odds:5.2f} [{low:.2f}, {high:.2f}]")
        print()

    print("What predicts a flip (all answers, fixed budget): logistic on averaged margin, option count, source")
    sources = sorted({r["source"] for r in rows})
    x = [[r["margin_avg"], math.log(r["options"])] + [float(r["source"] == s) for s in sources[1:]] for r in rows]
    x = [list(v) for v in zip(*_standardize([list(c) for c in zip(*x)][:2]))]
    x = [a + [float(r["source"] == s) for s in sources[1:]] for a, r in zip(x, rows)]
    w = logistic(x, [r["changes_fixed"] > 0 for r in rows])
    print(f"  per standard deviation: averaged margin {w[1]:+.2f} log-odds, log(option count) {w[2]:+.2f} log-odds")


# Against human label distributions: ChaosNLI (100 labels per item) and GoEmotions' per-rater rows.

def goemotions_raters(items: list[dict], raw_dir: Path) -> dict[str, dict]:
    """Per item, how many raters chose each emotion, joined to the fetched items by text."""
    import csv
    wanted = {item["state"]: item["id"] for item in items if item["source"] == "goemotions"}
    counts: dict[str, dict] = {}
    for path in sorted(raw_dir.glob("goemotions_*.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item_id = wanted.get(row["text"])
                if item_id is None:
                    continue
                entry = counts.setdefault(item_id, {"raters": 0, "counts": {}})
                entry["raters"] += 1
                for emotion, value in row.items():
                    if value == "1" and emotion not in ("rater_id", "example_very_unclear"):
                        entry["counts"][emotion] = entry["counts"].get(emotion, 0) + 1
    return counts


def _share(human: dict, label: str) -> float:
    """Share of the human votes on `label`."""
    total = human.get("raters") or sum(human["counts"].values())
    return human["counts"].get(label, 0) / total if total else 0.0


def _jsd(p: list[float], q: list[float]) -> float:
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl = lambda x, y: sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0)
    return (kl(p, m) + kl(q, m)) / 2


def human(args) -> None:
    items = {item["id"]: item for item in evaluate.load(args.paths, None)}
    if args.raters:
        for item_id, entry in goemotions_raters(list(items.values()), Path(args.raters)).items():
            items[item_id]["human"] = entry
    records = [json.loads(line) for path in args.files for line in Path(path).read_text().splitlines() if line.strip()]
    records = [r for r in records if r["id"] in items and "human" in items[r["id"]] and not fixed_order(r)]
    print(f"{len(records)} questions with human label counts\n")
    rows = []
    for record in records:
        row = features(record)
        human_counts = items[record["id"]]["human"]
        keys = sorted(record["answers"][0]["probs"])
        # Human share of the gold label and of the model's answers; human distribution over the options.
        row["gold_share"] = _share(human_counts, record["label"])
        base = record["answers"][0]["probs"]
        family = [base] + [a["probs"] for a in record["answers"] if a["kind"] != "repeat"]
        mean = {k: statistics.fmean(p[k] for p in family) for k in keys}
        row["base_share"] = _share(human_counts, max(base, key=base.get))
        row["avg_share"] = _share(human_counts, max(mean, key=mean.get))
        total = sum(human_counts["counts"].get(k, 0) for k in keys)
        if total:
            h = [human_counts["counts"].get(k, 0) / total for k in keys]
            row["jsd_base"] = _jsd([base[k] for k in keys], h)
            row["jsd_avg"] = _jsd([mean[k] for k in keys], h)
        rows.append(row)

    for source in sorted({r["source"] for r in rows}):
        group = [r for r in rows if r["source"] == source]
        print(f"== {source}: {len(group)} questions")
        for flip in ("changes_fixed", "changes"):
            name = "fixed-budget flip" if flip == "changes_fixed" else "any-order flip"
            print(f"  {name}")
            # H2 with label noise removed: only items whose gold label most humans chose.
            for cut in (0.8, 0.9):
                clean = [r for r in group if r["conf"] >= CONFIDENT and r["gold_share"] >= cut]
                fl = [r for r in clean if r[flip] > 0]
                st = [r for r in clean if r[flip] == 0]
                if fl and st:
                    wf, ws = sum(r["wrong"] for r in fl), sum(r["wrong"] for r in st)
                    print(f"    H2 on items with human agreement >= {cut:.0%} (confidence >= {CONFIDENT}): "
                          f"flipped {wf}/{len(fl)} wrong ({wf / len(fl):.1%}), "
                          f"stable {ws}/{len(st)} ({ws / len(st):.1%})")
            # Does a flip mark items humans disagree on?  Contested: the majority label got under 60% of votes.
            contested = [r["gold_share"] < 0.6 for r in group]
            print(f"    contested items (majority under 60%): {sum(contested)}/{len(group)}; AUROC for contested: "
                  f"flip {auroc([float(r[flip] > 0) for r in group], contested):.3f}, "
                  f"spread {auroc([r['spread'] for r in group], contested):.3f}, "
                  f"1-conf {auroc([-r['conf'] for r in group], contested):.3f}, "
                  f"1-conf_avg {auroc([-r['conf_avg'] for r in group], contested):.3f}")
            conf_group = [r for r in group if r["conf"] >= CONFIDENT]
            fl = [r for r in conf_group if r[flip] > 0]
            st = [r for r in conf_group if r[flip] == 0]
            if fl and st:
                share_fl = statistics.fmean(r["base_share"] for r in fl)
                share_st = statistics.fmean(r["base_share"] for r in st)
                print(f"    confident answers: mean human share of the model's answer, flipped {share_fl:.2f} "
                      f"(n={len(fl)}), stable {share_st:.2f} (n={len(st)}); "
                      f"contested share: flipped {statistics.fmean(r['gold_share'] < 0.6 for r in fl):.1%}, "
                      f"stable {statistics.fmean(r['gold_share'] < 0.6 for r in st):.1%}")
        with_jsd = [r for r in group if "jsd_base" in r]
        if with_jsd:
            jsd_base = statistics.fmean(r["jsd_base"] for r in with_jsd)
            jsd_avg = statistics.fmean(r["jsd_avg"] for r in with_jsd)
            print(f"  distance to the human distribution (Jensen-Shannon, bits): base order {jsd_base:.3f}, "
                  f"averaged over orders {jsd_avg:.3f}; "
                  f"human share of the answer: base {statistics.fmean(r['base_share'] for r in group):.3f}, "
                  f"averaged {statistics.fmean(r['avg_share'] for r in group):.3f}")
        print()


# Generated-text arm: the same items and orders answered by generation, through llama-server directly (llav
# itself never generates). "bare" uses llav's own prompt, which asks for a lone letter; "text" asks for the
# letter and the option's description. The answer is parsed from the text; the first generated token's
# alternatives are kept, so the first-token readout can be compared on the same call.

TEXT_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
               "Answer with its uppercase letter, a colon, and the option's description, and nothing else.")
GEN_TOKENS = 32


def _generate(url: str, turns: list[dict]) -> dict:
    body = json.dumps({"messages": turns, "temperature": 0, "max_tokens": GEN_TOKENS, "logprobs": True,
                       "top_logprobs": 10, "chat_template_kwargs": {"enable_thinking": False},
                       "cache_prompt": False}).encode()
    import urllib.error
    import urllib.request
    for attempt in range(6):
        request = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code in (429, 503) and attempt < 5:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    choice = data["choices"][0]
    tokens = choice.get("logprobs", {}).get("content") or []
    first = tokens[0] if tokens else None
    return {"text": choice["message"]["content"], "finish": choice.get("finish_reason"),
            "first": {"token": first["token"], "logprob": first["logprob"],
                      "top": {e["token"]: e["logprob"] for e in first.get("top_logprobs", [])}} if first else None,
            "seq_logprob": sum(e["logprob"] for e in tokens)}


def parse_answer(text: str, option_ids: list[str], descriptions: list) -> tuple[int | None, str]:
    """The option a generated answer names: a leading letter, else the longest option key or description
    the text contains. Returns (label index, how)."""
    import re
    stripped = text.strip()
    match = re.match(r"^[^A-Za-z0-9]*([A-Z])(?![A-Za-z])", stripped)
    if match and LABELS.index(match.group(1)) < len(option_ids):
        return LABELS.index(match.group(1)), "letter"
    lowered = stripped.lower()
    best = None
    for index, (key, description) in enumerate(zip(option_ids, descriptions)):
        for candidate in (str(key), description if isinstance(description, str) else json.dumps(description)):
            candidate = candidate.lower()
            if candidate and candidate in lowered and (best is None or len(candidate) > best[1]):
                best = (index, len(candidate))
    return (best[0], "description") if best else (None, "none")


def text(args) -> None:
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(evaluate.SRC))
    from llav.prompt import messages
    from llav.questions import parse_request
    items = [item for item in evaluate.load(args.paths, args.limit) if item["question"]["type"] == "choice"]
    done = set()
    if Path(args.out).exists():
        complete = [line for line in Path(args.out).read_text().splitlines(keepends=True) if line.endswith("\n")]
        Path(args.out).write_text("".join(complete))
        done = {json.loads(line)["id"] for line in complete}
        items = [item for item in items if item["id"] not in done]
        print(f"resuming: {len(done)} questions already in {args.out}")
    started = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def one(item: dict, kind: str, order: list[str]) -> dict:
        criteria = item["question"]["criteria"]
        body = {"state": item["state"], "model": "llav-latest",
                "questions": {"q": {**item["question"], "criteria": {key: criteria[key] for key in order}}}}
        state, _, questions = parse_request(body)
        question = questions[0]
        turns = messages(state, question.instructions, list(question.descriptions))
        if args.prompt == "text":
            turns[0] = {"role": "system", "content": TEXT_SYSTEM}
        generated = _generate(args.url, turns)
        index, how = parse_answer(generated["text"], list(question.option_ids), list(question.descriptions))
        return {"kind": kind, "order": order, "label_order": list(question.option_ids),
                "answer": question.option_ids[index] if index is not None else None, "how": how, **generated}

    with open(args.out, "a") as out:
        for count, item in enumerate(items, 1):
            requests = variants(item, args.seed, 1, 0, args.perms)[0]  # base once and the random permutations
            if len({tuple(_align_letters(order)) for _, order in requests}) == 1:
                continue  # single-letter keys fix the displayed order
            answers = list(pool.map(lambda pair: one(item, *pair), requests))
            out.write(json.dumps({"id": item["id"], "source": item["source"], "label": evaluate.label_key(item),
                                  "prompt": args.prompt, "answers": answers}) + "\n")
            out.flush()
            if args.progress:
                print(f"\r{count}/{len(items)}", end="", file=sys.stderr, flush=True)
    if args.progress:
        print(file=sys.stderr)
    print(f"{len(items)} questions in {time.perf_counter() - started:.0f} s -> {args.out}")


def text_analyze(args) -> None:
    """Accuracy, majority vote over orders, flips and parse failures of the generated answers; the same for
    the first-token readout taken from the same calls; and, with --readout, agreement with `run`'s answers."""
    records = [json.loads(line) for path in args.files for line in Path(path).read_text().splitlines()
               if line.strip()]
    readout = {}
    if args.readout:
        for line in Path(args.readout).read_text().splitlines():
            record = json.loads(line)
            readout[record["id"]] = {tuple(a["order"]): a for a in record["answers"]}
    rows = []
    for record in records:
        base = record["answers"][0]
        picks = [a["answer"] for a in record["answers"]]
        counts = {}
        for pick in picks:
            counts[pick] = counts.get(pick, 0) + 1
        majority = max(counts, key=lambda k: (counts[k], k == base["answer"]))
        # First-token readout from the same call: softmax over the declared letters among the alternatives.
        first_pick = first_conf = None
        if base["first"]:
            letters = {label: base["first"]["top"].get(label) for label in LABELS[:len(base["label_order"])]}
            found = {k: v for k, v in letters.items() if v is not None}
            if found:
                top_letter = max(found, key=found.get)
                total = sum(math.exp(v) for v in found.values())
                first_pick = base["label_order"][LABELS.index(top_letter)]
                first_conf = math.exp(found[top_letter]) / total
        row = {"source": record["source"], "wrong": base["answer"] != record["label"],
               "wrong_majority": majority != record["label"], "unparsed": base["answer"] is None,
               "how": base["how"], "flip": len({p for p in picks}) > 1,
               "conf": math.exp(base["first"]["logprob"]) if base["first"] else 0.0,
               "first_pick": first_pick, "first_conf": first_conf,
               "first_wrong": first_pick != record["label"] if first_pick else None,
               "text_vs_first": (base["answer"] == first_pick) if first_pick and base["answer"] else None}
        if record["id"] in readout:
            r = readout[record["id"]].get(tuple(base["order"]))
            if r:
                r_pick = max(r["probs"], key=r["probs"].get)
                row["readout_wrong"] = r_pick != record["label"]
                row["text_vs_readout"] = base["answer"] == r_pick
                r_family = [readout[record["id"]].get(tuple(a["order"])) for a in record["answers"]]
                r_family = [x for x in r_family if x]
                row["readout_flip"] = len({max(x["probs"], key=x["probs"].get) for x in r_family}) > 1
        rows.append(row)
    print(f"{len(rows)} questions, prompt '{records[0]['prompt']}'\n")
    sources = sorted({r["source"] for r in rows})
    print(f"{'source':30} {'n':>5} {'wrong':>6} {'major':>6} {'flip':>6} {'unpars':>6} {'letter':>6} "
          f"{'1st-tok wrong':>13} {'text=1st':>8} {'AUROC conf':>10}")
    for source in sources + ["all"]:
        g = rows if source == "all" else [r for r in rows if r["source"] == source]
        first = [r for r in g if r["first_wrong"] is not None]
        agree = [r for r in g if r["text_vs_first"] is not None]
        print(f"{source:30} {len(g):5d} {statistics.fmean(r['wrong'] for r in g):6.1%} "
              f"{statistics.fmean(r['wrong_majority'] for r in g):6.1%} "
              f"{statistics.fmean(r['flip'] for r in g):6.1%} {statistics.fmean(r['unparsed'] for r in g):6.1%} "
              f"{statistics.fmean(r['how'] == 'letter' for r in g):6.1%} "
              f"{statistics.fmean(r['first_wrong'] for r in first) if first else float('nan'):13.1%} "
              f"{statistics.fmean(r['text_vs_first'] for r in agree) if agree else float('nan'):8.1%} "
              f"{auroc([-r['conf'] for r in g], [r['wrong'] for r in g]):10.3f}")
    print("\nwrong-answer rate by confidence of the generated first token, stable vs flipped (text answers)")
    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.97), (0.97, 0.99), (0.99, 0.999), (0.999, 1.01)]
    for lo, hi in bands:
        b = [r for r in rows if lo <= r["conf"] < hi]
        st = [r for r in b if not r["flip"]]
        fl = [r for r in b if r["flip"]]
        f = lambda g: f"{statistics.fmean(r['wrong'] for r in g):5.1%} ({len(g)})" if g else "   -"
        print(f"  {lo}-{min(hi, 1)}: stable {f(st):16} flipped {f(fl)}")
    if readout:
        joined = [r for r in rows if "readout_wrong" in r]
        print(f"\nagainst the readout run ({len(joined)} questions matched by base order):")
        print(f"  text answer = readout answer: {statistics.fmean(r['text_vs_readout'] for r in joined):.1%}; "
              f"wrong: text {statistics.fmean(r['wrong'] for r in joined):.1%}, readout "
              f"{statistics.fmean(r['readout_wrong'] for r in joined):.1%}; flips: text "
              f"{statistics.fmean(r['flip'] for r in joined):.1%}, readout "
              f"{statistics.fmean(r['readout_flip'] for r in joined):.1%}")
        for name, key in (("text flips", "flip"), ("readout flips", "readout_flip")):
            fl = [r for r in joined if r[key]]
            st = [r for r in joined if not r[key]]
            print(f"  {name}: text answer wrong {statistics.fmean(r['wrong'] for r in fl):.1%} when it flips "
                  f"({len(fl)}) vs {statistics.fmean(r['wrong'] for r in st):.1%} when stable ({len(st)})")


# PriDe (Zheng et al., ICLR 2024) from the recorded rotations: the prior over option ids is the geometric mean
# of the observed id probabilities over a question's cyclic rotations, averaged over a small sample of
# questions; every other question's base-order answer is divided by it. Compared with averaging over orders.

def _pride_rows(record: dict) -> dict:
    """Base-order probabilities by displayed position, and the full cycle of rotations by position."""
    base = record["answers"][0]
    by_position = lambda a: [a["probs"][key] for key in a["order"]]
    cycle = [by_position(base)] + [by_position(a) for a in record["answers"] if a["kind"] == "rotation"]
    perms = [a for a in record["answers"] if a["kind"] == "perm"]
    return {"id": record["id"], "source": record["source"], "label": record["label"], "order": base["order"],
            "base": by_position(base), "cycle": cycle, "n": len(base["order"]),
            "perm_probs": [base["probs"]] + [a["probs"] for a in perms]}


def pride(args) -> None:
    rng = random.Random(args.seed)
    for path in args.files:
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        rows = [_pride_rows(r) for r in records if not fixed_order(r)]
        rows = [r for r in rows if len(r["cycle"]) == r["n"]]  # a full cycle is needed to estimate the prior
        groups: dict = {}
        for r in rows:
            groups.setdefault((r["source"], r["n"]), []).append(r)
        results = []
        priors = {}
        for key, group in groups.items():
            rng.shuffle(group)
            sample = group[:max(2, int(args.share * len(group)))]
            rest = group[len(sample):]
            n = key[1]
            log_prior = [0.0] * n
            for r in sample:
                for i in range(n):
                    log_prior[i] += statistics.fmean(math.log(max(c[i], 1e-12)) for c in r["cycle"])
            log_prior = [v / len(sample) for v in log_prior]
            top = max(log_prior)
            prior = [math.exp(v - top) for v in log_prior]
            prior = [v / sum(prior) for v in prior]
            priors[key] = prior
            for r in rest:
                debiased = [p / q for p, q in zip(r["base"], prior)]
                debiased = [v / sum(debiased) for v in debiased]
                keys = list(r["perm_probs"][0])
                mean = {k: statistics.fmean(p[k] for p in r["perm_probs"]) for k in keys}
                results.append({"source": r["source"], "n": n,
                                "base_wrong": r["order"][r["base"].index(max(r["base"]))] != r["label"],
                                "pride_wrong": r["order"][debiased.index(max(debiased))] != r["label"],
                                "avg_wrong": max(mean, key=mean.get) != r["label"],
                                "base_conf": max(r["base"]), "pride_conf": max(debiased),
                                "avg_conf": max(mean.values())})
        print(f"{path}: {len(results)} evaluated after a {args.share:.0%} sample per source for the prior")
        print(f"  {'source':30} {'n':>5} {'base':>6} {'PriDe':>6} {'avg':>6}   AUROC base/PriDe/avg   "
              "prior (by position)")
        for source in sorted({r["source"] for r in results}) + ["all"]:
            g = results if source == "all" else [r for r in results if r["source"] == source]
            w = lambda k: statistics.fmean(r[k] for r in g)
            au = lambda k, c: auroc([-r[c] for r in g], [r[k] for r in g])
            shown = ""
            if source != "all":
                prior = priors[next(k for k in priors if k[0] == source)]
                shown = " ".join(f"{v:.2f}" for v in prior[:8]) + (" ..." if len(prior) > 8 else "")
            aurocs = "/".join(f"{au(k, c):.3f}" for k, c in (("base_wrong", "base_conf"), ("pride_wrong", "pride_conf"),
                                                              ("avg_wrong", "avg_conf")))
            print(f"  {source:30} {len(g):5d} {w('base_wrong'):6.1%} {w('pride_wrong'):6.1%} {w('avg_wrong'):6.1%}   "
                  f"{aurocs}   {shown}")
        print()


# Candidate mass with the gold option removed: does the header flag a question none of the options fit?

def nofit(args) -> None:
    """Each choice question asked as given and with its gold option removed (matched pairs, same order
    otherwise); candidate mass and the answer's confidence recorded for both, then how well mass and
    confidence separate the two."""
    items = [item for item in evaluate.load(args.paths, args.limit) if item["question"]["type"] == "choice"
             and len(item["question"]["criteria"]) >= 3]
    rows = []
    started = time.perf_counter()
    with open(args.out, "w") as out:
        for count, item in enumerate(items, 1):
            criteria = item["question"]["criteria"]
            gold = evaluate.label_key(item)
            if gold not in criteria:
                continue
            removed = {k: v for k, v in criteria.items() if k != gold}
            questions = {"full": item["question"], "cut": {**item["question"], "criteria": removed}}
            body, headers = evaluate.ask(args.url, item["state"], questions)
            mass = evaluate.masses(headers, 2)
            answers = body["answers"]
            row = {"id": item["id"], "source": item["source"], "gold": gold, "options": len(criteria),
                   "full": {"mass": mass[0], "conf": max(answers["full"]["probabilities"].values()),
                            "answer": answers["full"]["choice"]},
                   "cut": {"mass": mass[1], "conf": max(answers["cut"]["probabilities"].values()),
                           "answer": answers["cut"]["choice"]}}
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            if args.progress:
                print(f"\r{count}/{len(items)}", end="", file=sys.stderr, flush=True)
    if args.progress:
        print(file=sys.stderr)
    print(f"{len(rows)} pairs in {time.perf_counter() - started:.0f} s -> {args.out}\n")
    nofit_report(rows)


def nofit_report(rows: list[dict]) -> None:
    full_right = [r for r in rows if r["full"]["answer"] == r["gold"]]
    print(f"{len(rows)} pairs; the full question is answered right on {len(full_right)} "
          f"({len(full_right)/len(rows):.1%})")
    print(f"{'source':30} {'n':>5} {'mass full':>10} {'mass cut':>9} {'conf full':>10} {'conf cut':>9} "
          f"{'AUROC mass':>10} {'AUROC conf':>10} {'cut<0.9':>8} {'full<0.9':>8}")
    for source in sorted({r["source"] for r in rows}) + ["all"]:
        g = rows if source == "all" else [r for r in rows if r["source"] == source]
        med = lambda side, key: statistics.median(r[side][key] for r in g if r[side][key] is not None)
        # Which side is the cut one: higher score = more likely cut.
        pairs = [(r["full"], False) for r in g] + [(r["cut"], True) for r in g]
        au_mass = auroc([-(p["mass"] if p["mass"] is not None else 1.0) for p, _ in pairs], [c for _, c in pairs])
        au_conf = auroc([-p["conf"] for p, _ in pairs], [c for _, c in pairs])
        low = lambda side: statistics.fmean((r[side]["mass"] or 1.0) < 0.9 for r in g)
        print(f"{source:30} {len(g):5d} {med('full', 'mass'):10.4f} {med('cut', 'mass'):9.4f} "
              f"{med('full', 'conf'):10.3f} {med('cut', 'conf'):9.3f} {au_mass:10.3f} {au_conf:10.3f} "
              f"{low('cut'):8.1%} {low('full'):8.1%}")
    print("\nshare of cut questions flagged at a mass threshold, and share of full questions wrongly flagged:")
    for cut in (0.99, 0.95, 0.9, 0.8, 0.7, 0.5):
        tp = statistics.fmean((r["cut"]["mass"] or 1.0) < cut for r in rows)
        fp = statistics.fmean((r["full"]["mass"] or 1.0) < cut for r in rows)
        print(f"  mass < {cut}: cut flagged {tp:6.1%}   full flagged {fp:6.1%}")
    print("\nmedian mass of the cut question by the confidence of its answer:")
    for lo, hi in ((0.0, 0.5), (0.5, 0.9), (0.9, 1.01)):
        g = [r for r in rows if lo <= r["cut"]["conf"] < hi and r["cut"]["mass"] is not None]
        if g:
            median = statistics.median(r["cut"]["mass"] for r in g)
            print(f"  cut confidence {lo}-{min(hi, 1)}: n={len(g):5d}  median mass {median:.4f}")


def nofit_analyze(args) -> None:
    rows = [json.loads(line) for path in args.files for line in Path(path).read_text().splitlines() if line.strip()]
    nofit_report(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="Ask every choice question in several option orders")
    command.add_argument("url", help="llav base URL, for example http://127.0.0.1:8080")
    command.add_argument("paths", nargs="+", help="JSONL files from evaluate.py fetch, or directories of them")
    command.add_argument("--out", required=True, help="JSONL to write, one line per question")
    command.add_argument("--limit", type=int, help="At most this many questions per file")
    command.add_argument("--repeats", type=int, default=3, help="Identical base-order asks, for the noise floor")
    command.add_argument("--rotations", type=int, default=13, help="At most this many further cyclic rotations")
    command.add_argument("--perms", type=int, default=6, help="Random permutations besides the base order")
    command.add_argument("--seed", type=int, default=0, help="Seeds the base order and the permutations")
    command.add_argument("--reword", help="JSON of source -> paraphrased criteria; adds a request per question "
                                          "asking the base order under each, and keeps only those sources")
    command.add_argument("--progress", action="store_true", help="Show a counter on stderr")
    command.set_defaults(run=run)
    command = commands.add_parser("h2", help="Flip effect on confident answers, adjusted for confidence and source")
    command.add_argument("files", nargs="+", help="JSONL written by run")
    command.add_argument("--bootstrap", type=int, default=200, help="Resamples for the adjusted intervals")
    command.set_defaults(run=h2)
    command = commands.add_parser("human", help="Flips and averaging against human label distributions")
    command.add_argument("files", nargs="+", help="JSONL written by run")
    command.add_argument("--data", dest="paths", nargs="+", required=True,
                         help="The fetched question files, for their human label counts")
    command.add_argument("--raters", help="Directory with GoEmotions' goemotions_*.csv per-rater files")
    command.set_defaults(run=human)
    command = commands.add_parser("text", help="Generated-text arm through llama-server directly")
    command.add_argument("url", help="llama-server base URL (not llav), for example http://127.0.0.1:8091")
    command.add_argument("paths", nargs="+", help="JSONL files from evaluate.py fetch, or directories of them")
    command.add_argument("--out", required=True, help="JSONL to write, one line per question")
    command.add_argument("--prompt", choices=("bare", "text"), default="text",
                         help="bare: llav's prompt (a lone letter); text: letter, colon and description")
    command.add_argument("--limit", type=int, help="At most this many questions per file")
    command.add_argument("--perms", type=int, default=6, help="Random permutations besides the base order")
    command.add_argument("--seed", type=int, default=0, help="Same seed as run, for the same orders")
    command.add_argument("--workers", type=int, default=8, help="Parallel requests; match the server's slots")
    command.add_argument("--progress", action="store_true", help="Show a counter on stderr")
    command.set_defaults(run=text)
    command = commands.add_parser("text-analyze", help="Accuracy, majority vote, flips of generated answers")
    command.add_argument("files", nargs="+", help="JSONL written by text")
    command.add_argument("--readout", help="A run output for the same model, to compare per item")
    command.set_defaults(run=text_analyze)
    command = commands.add_parser("pride", help="PriDe id-prior debiasing from the recorded rotations, vs averaging")
    command.add_argument("files", nargs="+", help="JSONL written by run (needs every rotation)")
    command.add_argument("--share", type=float, default=0.05, help="Share of each source used to estimate the prior")
    command.add_argument("--seed", type=int, default=0)
    command.set_defaults(run=pride)
    command = commands.add_parser("nofit", help="Candidate mass with the gold option removed, matched pairs")
    command.add_argument("url", help="llav base URL")
    command.add_argument("paths", nargs="+", help="JSONL files from evaluate.py fetch, or directories of them")
    command.add_argument("--out", required=True, help="JSONL to write, one line per pair")
    command.add_argument("--limit", type=int, help="At most this many questions per file")
    command.add_argument("--progress", action="store_true")
    command.set_defaults(run=nofit)
    command = commands.add_parser("nofit-analyze", help="Report from a nofit output")
    command.add_argument("files", nargs="+")
    command.set_defaults(run=nofit_analyze)
    command = commands.add_parser("analyze", help="Noise floor, AUROC per signal, H1 gain, H2 table")
    command.add_argument("files", nargs="+", help="JSONL written by run")
    command.set_defaults(run=analyze)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
