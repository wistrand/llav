# Plan: how a noul should reach the model

Drafted 2026-09-26. Status: not pursued. The rule-only set is written and its first result (below) shows the
fold doing what it is for; the counter-evidence is JevBench's nouls, and llav is not tuned toward a
benchmark. Reopen only if a caller-style set (PubMedQA, or a caller's own labelled nouls) shows the same
loss under the fold.

## Why

A noul reaches the model as a two-option decision, `Yes` and `No`, with the caller's `criteria.true` and
`criteria.false` folded into the criterion as `Answer Yes when: ...` lines and repeated in the option
descriptions (`_fold_noul` and `_describe` in `questions.py`). The fold was adopted on 2026-09-23 for one
reason, measured on one question: Granite 4.2 3B answers the criterion and ignores option descriptions, so a
rule written only in `criteria.true` scored 0.017 on Granite where Qwen3.5-4B scored 0.987; the fold fixed
both (research.md, "Where the decision rule is written matters").

The trigger for this plan, from [comparisons.md](../comparisons.md#llav-on-jevbenchs-public-items-2026-09-26):
on JevBench's public nouls, all of which carry criteria, the same questions asked as two-option choices
(`{"yes": criteria.true, "no": criteria.false}`, no fold) scored 23 of 24 and 27 of 38 with Qwen3.5-4B,
against 20 and 24 with the fold. Asking both orders on top changed nothing. Small counts, same direction
twice. So the fold may cost Qwen accuracy on the very questions it was meant to help, and the two
measurements behind the current design total one question and 62 items.

## What differs between the two framings

| Element                    | Fold (current)                                          | Choice framing (proxy run)                    |
|----------------------------|---------------------------------------------------------|-----------------------------------------------|
| Criterion the model reads  | instructions + `Answer Yes when: <true>` + `Answer No when: <false>` | instructions only                  |
| Option A                   | `Yes: <true>`                                           | `yes: <true>`                                 |
| Option B                   | `No: <false>`                                           | `no: <false>`                                 |
| Order                      | fixed, Yes first                                        | fixed, yes first (both orders added nothing)  |

Three things change at once: the fold in the criterion, the case of the labels, and nothing else. The
experiment separates them.

## Candidates

- F0, current: fold in the criterion, `Yes: <true>` / `No: <false>`.
- F1, no fold: instructions only, `Yes: <true>` / `No: <false>`. The pre-fold behaviour, the Granite failure
  case.
- F2, choice framing: instructions only, `yes: <true>` / `no: <false>`. What the proxy sent.
- F3, fold, bare labels: fold in the criterion, options `Yes` / `No`. Tests whether repeating the rule in the
  descriptions hurts once it is in the criterion.
- F4, F2 in both orders averaged: reflex's construction. Expected to add nothing (JevBench), kept as the
  control for the order question.
- F5, F0 in both orders averaged: the same control for the current framing.
- F6, fold in the criterion with `yes: <true>` / `no: <false>` labels: the fold's rule-keeping plus whatever
  F2 gains on JevBench, if that gain is the label form rather than the absence of the fold.

Without criteria (BoolQ, most caller nouls today) F0, F1 and F3 coincide (`Yes` / `No`) and only the label
case differs from F2. Those items measure the case effect alone.

## Data

| Set                                   | Nouls | Criteria | Why                                                        |
|---------------------------------------|------:|----------|------------------------------------------------------------|
| JevBench public, original + hard      |    62 | yes      | The trigger; rubric-style criteria                          |
| Jevals PubMedQA (`jevals_pubmedqa`)   |   300 | yes      | Board-comparable; criteria are Jevals' own wording          |
| BoolQ (`boolq`)                       | 1,000 | no       | The no-criteria case; the case effect alone                 |
| `benchmark.py accuracy` shapes        |     4 | mixed    | The four shapes the fold was measured on; must stay right  |
| Rule-only nouls (`data/rule-only-nouls.jsonl`) | 50 | yes  | Written 2026-09-26: rule in `criteria.true` only, vague instructions, the Granite failure shape; 25 true, 25 false; in 41 the rule's answer is the opposite of the naive reading of the instruction (`naive` field) |

The rule-only set is the one the fold was built for; without it the experiment cannot say whether removing
the fold brings the Granite failure back. Its labels hold by construction: each rule names something
checkable in the text (a word, an amount, a date, a phone number, a language), present in the true items
and absent in the false ones, and the false items are written to sound important so that a model answering
the vague instruction instead of the rule gets them wrong.

## Models

Qwen3.5-4B (the default), Qwen3.5-2B (the browser demo), Granite 4.2 3B (the fold's reason), SmolLM3-3B.
The decision must not regress Granite on the rule-only set, or it needs a per-template choice.

## Method

1. A research-only switch in `questions.py`, `LLAV_NOUL_FRAMING` read from the environment, selecting F0 to
   F3 (F4 and F5 are the proxy's job, `scripts/orders-proxy.py --nouls-as-choice` plus one order or both).
   Default F0, so nothing changes for anyone; the switch is removed when the plan closes. `messages()` is
   untouched, so `PROMPT_VERSION` stays: the framing lives in what `parse_question` builds, as the fold does.
2. One llav per model on the box, `evaluate.py score` per framing over the sets above, and the JevBench
   harness for its 62. Metrics: accuracy, Brier, ECE, Jevals dscore for PubMedQA; per set and model.
3. `benchmark.py accuracy` per framing, to keep the four shapes.
4. Cost: 4 models x 4 framings x about 1,400 nouls, one forward pass each, under an hour on the RTX 3090.

## Decision rule

Adopt a framing other than F0 only if, on every model, it is not worse than F0 by more than the bootstrap
interval on any set, and better on at least PubMedQA or the JevBench nouls. If the best framing for Qwen
regresses Granite on the rule-only set, keep F0 as the default and note the trade-off; a per-template
framing is possible (`templates.py` chooses per-model behaviour from the rendered template, not the name),
but a startup probe that picks a framing is a bigger change and needs its own evidence.

If a framing is adopted: `questions.py`, [design.md](../design.md) (the fold section), research.md (a new
table replacing the one-question evidence), the README's noul description, the demo port `docs/demo/llav.js`
(`foldNoul` and `describe` mirror the Python), `openapi.py` only if the response changes (it does not), and
the calibration note: every noul answer changes, so fitted temperatures for nouls are invalid.

## First result: the two framings trade off (2026-09-26)

Qwen3.5-4B on the rule-only set, through the box (native helper), `evaluate.py score`; F2 through
`orders-proxy.py --nouls-as-choice --orders 1`. Results in `local/results/jevbench/rule-only-qwen4b-*.jsonl`.

| Framing                     | Rule-only nouls (50) | JevBench nouls, original (24) | JevBench nouls, hard (38) |
|-----------------------------|----------------------|-------------------------------|---------------------------|
| F0, the fold                | 50/50, ECE 0.009     | 20                            | 24                        |
| F2, yes/no choice, no fold  | 43/50, ECE 0.078     | 23                            | 27                        |

F2's seven misses are the mundane texts where the rule holds (a lunch menu mentioning America, a postcard
from Stockholm, a cat on the windowsill): the model answered the vague instruction instead of the rule,
the Granite failure, now on Qwen without the fold. So the fold costs something on JevBench's rubric wording and is
not removable: the framings trade off. The fold stays; F6 is the candidate to try if the question is ever
reopened on caller-relevant data. Qwen is one model; the table would need the other three.

## Order

The order question is settled enough to leave: on JevBench, both orders added nothing on top of F2, and the
permutation experiment found averaging helps confusable multi-option choices, not two-option ones. F4 and F5
stay in the plan as controls only. A server-side option to average a noul's two orders is not planned.

## Not in scope

Score questions and choices keep their construction. The prompt format (`messages()`) is not touched.
