# Research behind the design

Measurements of llav's readout: whether its prompt matches the reference, how accurate and calibrated its
answers are, and the approaches tried and rejected. Timings are in [performance.md](performance.md); other
models and System One systems are in [comparisons.md](comparisons.md).

Measurements that shaped llav. The first sections were made in the SemIf repository before llav existed,
with its `benchmarks/llama_server_probe.py`; those outputs were kept outside any repository, so the numbers
below are the record. Later sections use llav's own `scripts/benchmark.py` and `scripts/evaluate.py` and
name their date and machine. Unless a section says otherwise: a laptop with an Intel Arc B390 iGPU,
llama.cpp build 10809, Qwen3.5-4B Q8_0 GGUF (bartowski, the revision pinned in `scripts/fetch-model.sh`).

## Contents
- Prompt fidelity and agreement
- Real articles, not hand-written sentences
- Where the decision rule is written matters, per model
- Rejected approaches
- Candidate mass
- Labelled evaluation: calibration and option order
- Temperature calibration
- Open questions

## Prompt fidelity and agreement

Workload: SemIf's owned `shape777` fixture, 37 states of about 1,800 tokens, 21 binary questions each, 777
decisions. Reference: SemIf's published BF16 PyTorch predictions from an RTX 3090.

- Rendering through llama-server's `/apply-template` and `/tokenize` reproduced all 777 prompt hashes,
  token counts and answer-token ids of SemIf's torch scorer.
- Serial-restored readout agreed on the argmax for 768/777 decisions against fresh reference scoring, 771
  against reference serial reuse, 772 against reference parallel reuse. Median absolute probability
  difference 0.005, 99th percentile 0.065, maximum 0.149.
- All 9 disagreements with fresh scoring were near-ties (0.47 to 0.53 in both runs), and 5 were exact
  0.5/0.5 ties in the reference. The reference's own reuse modes differ from its fresh mode on 5 to 6
  decisions, so this is within the reference's own noise.
- The model put at least 99.6% of its probability on the answer letters.

These results used SemIf's prompt wording. llav's choice-key folding and `Yes`/`No` noul options were
checked only for internal consistency: one 21-question request versus 21 single-question requests agreed on
21/21 choices, with probabilities within 0.008.


## Real articles, not hand-written sentences

The easy and hard sets are sentences written to probe one thing each. This check uses 29 Wikipedia articles
truncated to 10,000 characters (1,809 to 2,653 tokens), one request per article with four questions sharing
the state: a 7-way category choice, "is the subject a person?", the same rule written only into a vague
question's `criteria`, and a technicality score with no ground truth. Labels are by hand; `Chess` and
`Go (game)` are filed under sport, which is arguable. Run on 2026-09-23 against Qwen3.5-4B on the RTX 3090
with the native helper:

| Check                                      | Result |
|---------------------------------------------|-------:|
| Category, 7 options                         |  29/29 |
| Is the subject a person?                    |  29/29 |
| Same rule, written only in `criteria`       |  29/29 |

- Every article is a new state, so every request is a cache miss: 0.85 s on the server (`X-Llav-Seconds`),
  3.31 s measured from the laptop, the difference being the tailnet round trip to a rented box. Timing
  through a network says nothing about the server.
- The criteria fold holds on real prose, but a vague criterion still costs confidence: Apollo 11 answered
  0.003 through `instructions` and 0.355 through `criteria`, Chernobyl 0.002 against 0.142. Both land on the
  right side of 0.5; folding recovers the answer, not the certainty.
- Category confidence was 0.97 to 1.00 everywhere except Mount Everest at 0.80, the one subject that is
  arguably a natural feature rather than a place.
- The technicality score has no ground truth but orders sensibly: CRISPR 1.91 and Chernobyl 1.75 at the top,
  Reykjavík 0.23 and Hip-hop 0.26 at the bottom.
- `scripts/benchmark.py fetch DIR` downloads the set, `scripts/benchmark.py articles URL DIR` runs it. The
  articles are not vendored: Wikipedia text changes, so a future miss is worth reading before it counts as a
  regression.

## Where the decision rule is written matters, per model

Reported 2026-09-23: Granite 4.2 3B answered "no" to a question whose rule sat only in the option
description. One noul question asked four ways, 400-word state that mentions America throughout:

| Question shape                                                            | Qwen3.5-4B | Granite 4.2 3B |
|---------------------------------------------------------------------------|-----------:|---------------:|
| `instructions` "is this important", `criteria.true` "america is mentioned" |      0.993 |          0.018 |
| Rule in `instructions`, no criteria                                        |      0.999 |          0.999 |
| Rule in `instructions`, both sides described                               |      0.999 |          0.997 |
| Vague `instructions`, both sides described                                 |      0.993 |          0.183 |

Granite answers the criterion and reads the option descriptions as labels; Qwen weighs both. The prompt is
the same shape in every row, so this is the model, not the format. Advice for callers, and for the README:
llav now repeats a noul's criteria in the criterion (`_fold_noul`), so the caller can write the rule in
either field. Candidate fixes, same question, before the change:

| What llav puts in the prompt                | Qwen3.5-4B | Granite 4.2 3B |
|----------------------------------------------|-----------:|---------------:|
| `Yes: <rule>` / `No`, as it was              |      0.987 |          0.017 |
| Fill the empty side with `No: otherwise`     |      0.987 |          0.008 |
| Negate it: `No: not the case that <rule>`    |      0.994 |          0.646 |
| Fold the rule into the criterion             |      0.996 |          1.000 |
| Fold it and keep the descriptions (adopted)  |      0.999 |          1.000 |

Patching the option descriptions does not work; only the criterion does. After the change all four shapes
score 0.997 to 1.000 on both models, and the easy and hard scores are unchanged (Qwen 19/19 and 17/17,
Granite 18/19 and 13/17). `scripts/benchmark.py accuracy` runs the four shapes against any model.
`choice` and `score` are not folded and remain exposed to the same effect; unmeasured.

## Rejected approaches

- **Context checkpoints for prefix reuse.** Debug logs showed llama-server saving two 50 MB checkpoints per
  request (one duplicate of the checkpoint it had just restored, one at the end of the prompt), each a copy
  off the GPU of about 170 ms. `--ctx-checkpoints 2` and `--cache-ram 0` changed little. Replaced by
  save/restore through files, since copying onto the GPU is fast.
- **Parallel slots for one state.** llama.cpp does not batch several sequences of this hybrid model
  efficiently; 21 concurrent questions took as long as or longer than serial.
- **Forking llama.cpp.** Rejected while the system package works; a fix belongs upstream.
- **PyTorch runtime.** Faster for shared states (batched suffixes), but needs per-hardware torch builds,
  Triton toolchains and a workaround for a CPU fallback in Qwen3.5's attention on Intel XPU.

## Candidate mass

`X-Llav-Candidate-Mass` checked on 2026-09-24, Qwen3.5-4B Q8_0 on the laptop, one 110-character support
ticket, six questions. Both paths agree within 0.01.

| Question                                       | llama-server | Native | Answer              |
|------------------------------------------------|-------------:|-------:|---------------------|
| Urgent? (noul)                                 |       0.9998 | 0.9997 | Yes, 0.9998         |
| Team, three options                            |       0.9997 | 0.9997 | billing             |
| Anger, three levels                            |       0.9997 | 0.9997 | 1.68                |
| Colour of the customer's car? red or blue      |       0.5694 | 0.5606 | red, 0.79           |
| "Write a short poem about this ticket." (noul) |       0.9976 | 0.9972 | No, 0.93            |
| First letter of the first word, 26 options     |       0.9990 | 0.9989 | O (the answer is C) |

- A question no option fits is the case the metric catches. The missing 0.43 went to `NA` (0.16), `NONE`,
  `N`, `None` and `Neither`: the model declined both options, and the softmax over A and B still reported
  red at 0.79.
- High mass does not mean a right answer: the 26-option question answered `O` at 0.999 mass, and the
  poem instruction was forced into a Yes/No answer without the mass dropping.
- One ticket, six questions: enough to show the header works and what a drop looks like, not to set a
  threshold. No threshold is known yet. The web UI flags an answer below 0.9 (`LOW_MASS` in
  `webui.html`) as "No option fits"; that value is a display cue chosen between the 0.999 of working
  answers and the 0.57 above, not a measurement.
- `scripts/benchmark.py accuracy` reports the lowest and median mass per question set and repeats the
  car-colour check. On 2026-09-24 it gave 0.9996 minimum on the easy set and 0.9990 on the hard set, which
  supersedes the hand-read "Letter mass" column for Qwen3.5-4B under "Other models" in
  [comparisons.md](comparisons.md#other-models).

## Labelled evaluation: calibration and option order

Run on 2026-09-24 with `scripts/evaluate.py`, Qwen3.5-4B Q8_0, on the RTX PRO 4000 Blackwell box with the
native helper. `score` took 107 s there against 865 s on the laptop through llama-server, with the same
answers within noise (81.3% against 81.0% overall). 1,176 labelled questions: samples of BoolQ (noul),
AG News (4 options), DBpedia-14, the 13 card intents of Banking77, Yelp stars (score, 5 levels), and
SemIf's `authored144` (3 options, three families of 48). The public samples are downloaded by `fetch`,
not committed.

`score`, each question asked once in the caller's order. Coverage is the share answerable, most confident
first, at an error rate of at most 5%.

| Source                          |    n |   Acc |   ECE | Brier |  NLL | Coverage at 5% |
|---------------------------------|-----:|------:|------:|------:|-----:|---------------:|
| DBpedia, 14 options             |  224 | 98.2% | 0.013 | 0.027 | 0.05 |           100% |
| BoolQ, noul                     |  200 | 92.5% | 0.038 | 0.123 | 0.23 |            94% |
| SemIf evidence interpretation   |   48 | 95.8% | 0.052 | 0.104 | 0.23 |           100% |
| SemIf rule application          |   48 | 89.6% | 0.076 | 0.160 | 0.33 |            83% |
| AG News, 4 options              |  200 | 80.5% | 0.113 | 0.299 | 0.76 |            44% |
| Banking77 card intents, 13      |  208 | 75.0% | 0.124 | 0.381 | 1.23 |            59% |
| Yelp stars, score               |  200 | 59.0% | 0.204 | 0.617 | 1.38 |            12% |
| SemIf candidate selection       |   48 | 56.2% | 0.242 | 0.523 | 0.92 |            27% |
| All                             | 1176 | 81.3% | 0.073 | 0.281 | 0.69 |            65% |

- Overconfident at the top: 831 answers above 0.9 had mean probability 0.986 and were right 92.3% of the
  time. Below 0.9 the bins are roughly calibrated. A threshold of 0.99 does not mean 1% errors.
- Calibration depends on the task: ECE is 0.013 on DBpedia and 0.20 on Yelp. One temperature for all
  workloads would not fit; this supports calibrating per workload.
- Candidate mass never fell below 0.96 on these questions, so it separates nothing here; its use is the
  no-fit case above.
- This is the first labelled measurement of llav's own wording. Against SemIf's published direct-logit
  results on `authored144` (its prompt, BF16 torch), llav scores higher on evidence interpretation (95.8%
  against 85.4%) and rule application (89.6% against 87.5%) and lower on candidate selection (56.2%
  against 68.8%). 48 rows per family: one standard error is 5 to 7 points.

**Letter-named options.** Candidate selection's options are keyed `A`, `B` and `insufficient`, in varying
order, and the state names "Candidate A" and "Candidate B". llav folds the key into the description, so
answer letter A can read "B: Candidate B". The model answers the name, not the letter: 21 of 48 wrong,
mostly `A` and `B` swapped or pushed onto `insufficient`. Averaging over rotations fixes it (below), which
shows the cause is the letter collision and not the judgement.

The fix, `_align_letters` in `questions.py`, shows a single-letter key at its own letter. Rerun on the same
box the same day:

| Candidate selection, 48 | Before | After | SemIf direct logits |
|-------------------------|-------:|------:|--------------------:|
| Accuracy                |  56.2% | 95.8% |               68.8% |
| ECE                     |  0.242 | 0.065 |                     |
| NLL                     |  0.919 | 0.155 |               0.554 |
| Coverage at 1% error    |    27% |   85% |                     |

- Every other source scored identically, as it must: none has single-letter keys, so their prompts did not
  change. Overall accuracy rose from 81.3% to 82.9%, all of it from this family.
- Under `shifts` the family now flips 0 of 48: every rotation reaches the model aligned. Probabilities still
  differ by up to 0.09 between rotations of an identical prompt, from the helper's batched decode on CUDA
  (median 0.000); the answers do not.
- The tables below were measured before the fix; their candidate-selection rows show the collision.

`shifts`, every cyclic rotation of each choice and score question's options (8,072 rotations, 445 s):

| Source                        |  K | Answer changes under some order | vs caller's order | Median shift | Picked first listed |
|-------------------------------|---:|--------------------------------:|------------------:|-------------:|--------------------:|
| DBpedia                       | 14 |                            3.6% |              2.2% |        0.006 |   7.3% (1/K = 7.1%) |
| AG News                       |  4 |                            8.5% |              4.7% |        0.007 |  25.4% (1/K = 25%)  |
| SemIf evidence interpretation |  3 |                           12.5% |              6.2% |        0.052 |  34.7% (1/K = 33%)  |
| SemIf rule application        |  3 |                           16.7% |             11.5% |        0.143 |  38.9% (1/K = 33%)  |
| Banking77 card intents        | 13 |                           32.2% |             13.7% |        0.147 |   9.2% (1/K = 7.7%) |
| Yelp stars                    |  5 |                           35.5% |             20.8% |        0.255 |  17.8% (1/K = 20%)  |
| SemIf candidate selection     |  3 |                           60.4% |             50.0% |        0.510 |  25.7% (1/K = 33%)  |
| All                           |    |                           21.1% |              9.3% |        0.057 |                     |

Geometric mean over rotations against the caller's order:

| Source                    | Acc, caller's order | Acc, geometric mean | NLL        | Coverage at 1% |
|---------------------------|--------------------:|--------------------:|------------|---------------:|
| SemIf candidate selection |               60.4% |               91.7% | 0.91, 0.21 |      27%, 81%  |
| Banking77 card intents    |               75.5% |               77.9% | 1.24, 1.02 |       7%, 43%  |
| AG News                   |               81.0% |               83.5% | 0.77, 0.67 |      10%, 6%   |
| Yelp stars                |               57.5% |               56.5% | 1.38, 1.24 |       4%, 6%   |
| All 976                   |               79.0% |               81.5% | 0.79, 0.65 |       4%, 21%  |

- Order sensitivity is real and larger than the reversal check suggested: "no order flips" on the 11 easy
  reversals in `benchmark.py accuracy` holds for easy questions only. On confusable options (Banking77,
  Yelp) a third of answers change under some rotation.
- It is instability, not a preference for a position: the share picking whatever was listed first stays
  near 1/K everywhere. That fits a noisy readout on near-ties better than the additive position bias that
  AnyJev's prior correction targets (inference).
- The geometric mean lowers NLL on every choice source, raises or keeps accuracy, fixes the letter-named
  options, and raises coverage at 1% error where the caller's order had little. It did not improve ECE
  (0.086 to 0.092 overall): the combined answers are sharper, not better calibrated, so calibration stays a
  separate step.
- Yelp gains nothing. Rotating a score's levels breaks their low-to-high order, which the model reads; a
  cyclic shift is the wrong perturbation for an ordinal scale (inference).
- Cost: K passes per question. On the GPU box with the native helper a rotation cost about 55 ms.

How many orders does averaging need? Recomputed offline from the same rotations, leaving out candidate
selection (fixed since by `_align_letters`) and Yelp (rotation breaks a scale's order), 832 questions:

| Averaged over           | Accuracy |   NLL |   ECE | Coverage at 5% error |
|-------------------------|---------:|------:|------:|---------------------:|
| Caller's order          |    86.1% | 0.618 | 0.061 |                76.5% |
| 2 orders (0 and K/2)    |    87.2% | 0.559 | 0.066 |                79.0% |
| 3 orders, evenly spaced |    87.4% | 0.520 | 0.070 |                79.7% |
| All K                   |    87.6% | 0.518 | 0.069 |                79.8% |

- Three orders get nearly all of the full average's gain at 3 passes instead of up to 14.
- The gain is small but not noise: across Banking77 and AG News, averaging fixed 14 answers and broke 4.
- Averaging does not calibrate: ECE rises as the combined answers sharpen. A temperature would have to be
  fitted on averaged probabilities.
- Not implemented in llav. An opt-in form would be three orders, choice questions only, requested per call
  by a header, since the body has no room for it.

## Temperature calibration

`scripts/evaluate.py fit` on the `score` predictions above (after the letter fix), 2026-09-24, same box. One
temperature per question type. Held out: two halves split by question id, each calibrated with the
temperatures fitted on the other.

Fitted on all 1,176: choice T = 1.66, noul T = 1.47, score T = 1.94. All three above 1: the model is
overconfident on every type.

| Source                     | ECE raw | Pooled fit, held out | Fitted on this source only, held out | T, this source |
|----------------------------|--------:|---------------------:|-------------------------------------:|---------------:|
| AG News                    |   0.113 |                0.060 |                                0.043 |           2.15 |
| Banking77 card intents     |   0.124 |                0.047 |                                0.100 |           1.90 |
| BoolQ                      |   0.038 |                0.026 |                                0.026 |           1.47 |
| DBpedia                    |   0.013 |                0.061 |                                0.011 |           0.84 |
| SemIf, three families      |   0.036 |                0.107 |                                0.032 |           0.96 |
| Yelp stars                 |   0.204 |                0.083 |                                0.083 |           1.94 |
| All                        |   0.069 |                0.023 |                                      |                |

- Pooled, calibration works on average: ECE 0.069 to 0.023 and NLL 0.660 to 0.571 held out. With the file
  loaded (`--calibration`), `score` on the same questions gave ECE 0.024; the top bin went from mean
  probability 0.986 at 92.3% accuracy to 0.966 at 97.5%. Accuracy is unchanged, 82.9%, as it must be.
- Pooled, it harms the tasks that were already calibrated: DBpedia 0.013 to 0.061, SemIf's 144 0.036 to
  0.107. Their best temperatures are near or below 1 (0.84, 0.96); the pooled 1.66 softens
  answers that were right. This is why llav ships no calibration file: fit on the workload the thresholds
  will run on.
- `fit` now refuses the DBpedia and SemIf per-source fits in this table: DBpedia has 4 wrong answers in 224
  and SemIf 12 in 144, below `MIN_ERRORS` (`scripts/evaluate.py`) per half, where the temperature is not
  pinned down. Their rows above come from before that guard.
- Banking77 fitted alone did worse held out (0.100) than pooled (0.047). Each half has about 100 questions;
  one temperature from 100 labels is noisy. Treat a few hundred labels per workload as the minimum
  (inference from this one case).
- Coverage barely moves under one temperature, as expected: Banking77 at 5% error 59.1% to 61.5%, AG News
  44.0% to 45.0%, Yelp 11.5% to 12.0%. What changes is the threshold that achieves it: for Banking77, 5%
  errors meant cutting at 0.96 raw and at 0.69 calibrated. A first held-out report showed coverage collapsing
  (Banking77 59% to 21%); that came from ranking the two halves together under different temperatures, and
  `fit` no longer reports it.

## Open questions

- Accuracy of llav's own wording against labelled data: measured on public samples and `authored144` (see
  "Labelled evaluation"); not yet against `shape777`-scale data or callers' own workloads.
- Whether a BF16 GGUF closes the remaining probability differences, which would attribute them to Q8_0
  quantization: untested.
- Names that collide with answer letters only inside the state ("Candidate A" with keys `first`, `second`):
  `_align_letters` does not cover them, and no labelled set tests them.
