# Comparisons

Other models under llav, and other System One systems on the same labelled questions. llav's own readout
is in [research.md](research.md), its timings in [performance.md](performance.md). Every comparison names
its date, machine and model; results from before a fix say so. Unless a section says otherwise: a laptop
with an Intel Arc B390 iGPU, llama.cpp build 10809, Q8_0 GGUFs.

## Contents
- The System One landscape
- Other models
- Comparison with CLM
- Muse Glimmer 30B, a test
- Comparison with Laya and von
- Against the Jevals board
- Jev through OpenRouter
- Open questions

## The System One landscape

TypeSafe launched Jev on 2026-09-15 and named the category "System One models": typed `noul`, `choice` and
`score` answers with probabilities, from one parallel pass instead of generated text. Within ten days
community lists counted over a hundred related repositories. What follows was read on 2026-09-24 from
launch material, write-ups and project READMEs; only the rows marked measured were run here.

| Approach                                        | Examples                                              | Here                          |
|-------------------------------------------------|-------------------------------------------------------|-------------------------------|
| Hosted, trained, undisclosed                    | Jev (TypeSafe)                                        | Jev measured                  |
| Letter probabilities of an untrained chat model | SemIf, OneForward, AnyJev, poorjev, decidr, **llav**  | llav measured                 |
| Trained bidirectional encoder                   | Laya, von, openJev Verdict                            | Laya and von measured         |
| Frozen LLM plus trained head                    | CLM, minojev, kev, JevForge                           | CLM measured                  |
| Diffusion model                                 | OpenJev (DiffusionGemma)                              | not run                       |

- **Jev** claims 70 to 500 ms, $0.042 per million input tokens, up to 255 options and calibrated
  probabilities from a training method it calls RLCD; size, base model and training data are undisclosed.
  Independent write-ups put it level with mid-price LLMs and 6.5 to 11.5 points behind the frontier, with
  ECE 0.071 to 0.161, overconfident on multi-class public sets, and fixed largely by one fitted temperature;
  swapping option names changed 32.5% of its answers in one study. Jevals reports 69.0 on PubMedQA and 67.8
  on the full 77-intent Banking77. Accuracy, calibration, order and speed were since measured here; see
  [Jev through OpenRouter](#jev-through-openrouter).
- The same weaknesses showed in llav's own measurements: calibration that depends on the task and yields to
  one temperature per workload ([research.md](research.md#temperature-calibration)), and answers that move
  with option order and names ([research.md](research.md#labelled-evaluation-calibration-and-option-order)).
- llav's 26-option limit rules out the full Banking77 that Jev is often measured on; our Banking77 rows use
  its 13 card intents.
- Sources: [TypeSafe launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
  [eight days of independent tests](https://dev.to/gde/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1kln),
  [Jevals](https://jevals.com/), [awesome-jev](https://github.com/cobanov/awesome-jev),
  [awesome-open-system-one](https://github.com/MorrisZJ/awesome-open-system-one).

## Other models

IBM Granite 4.0 Micro and SmolLM3-3B (Q8_0) were the first alternatives tried. Checked on 2026-09-22 with
the four-question support-ticket request from the README plus one yes/no question ("Is the customer
praising the product?"), nothing more:

- Both pass `Engine`'s startup checks: `A`–`Z` are single tokens and the template tail does not merge
  with a label.
- Both put at least 99.9% of the next-token probability on the declared letters for every question.
- SmolLM3 renders `Reasoning Mode: /no_think` and an empty `<think>` block with `enable_thinking: false`.
- Answers compared with Qwen3.5-4B on the same request:

| Question                    | Qwen3.5-4B        | Granite 4.0 Micro  | SmolLM3-3B          |
|-----------------------------|-------------------|--------------------|---------------------|
| Urgent (noul)               | 1.00              | 1.00               | 0.99                |
| Praising the product (noul) | 0.00              | 0.00               | 0.11                |
| Department (choice)         | billing 0.85      | technical 1.00     | technical 0.77      |
| Frustration (score, 0 to 2) | 0.94              | 1.00               | 1.01                |

Granite is near-certain on every question, including the department question, where billing is the
better answer for failing payouts. That points to worse calibration than Qwen's, from one example only.
Neither model's accuracy against labelled data has been measured, and neither has been timed on a long
state.

### Model screening

Checked on 2026-09-22 with a scratch eval against each model, one at a time on the same laptop: 19
clear-cut questions with known answers (nouls and choices over sentiment, language, topic and department),
the same 11 choice questions with their options reversed, and a 10-question request on a 1,800-token state.
All Q8_0, and all before the state cache and the trim probe existed, so the times here are higher than the
ranking table's further down; accuracy is unaffected. `scripts/benchmark.py` runs these checks
(`accuracy`, `timing`, `phases`); the numbers below predate it but its question sets are the same.
Nineteen easy questions separate broken models from working ones; they do not rank the working ones.

| Model              | Released | Correct | Flips when reversed | Letter mass | 10 q, long state |
|--------------------|----------|--------:|--------------------:|------------:|-----------------:|
| Qwen3.5-4B         | 2026-02  |   19/19 |                0/11 |        1.00 |           4.76 s |
| Qwen3.5-2B         | 2026-02  |   18/19 |                1/11 |        1.00 |           2.24 s |
| Granite 4.2 3B     | 2026-08  |   18/19 |                0/11 |        1.00 |           4.90 s |
| Granite 4.0 H Tiny | 2025-10  |   18/19 |                1/11 |        1.00 |           4.64 s |
| Granite 4.0 Micro  | 2025-10  |   18/19 |                1/11 |        1.00 |           5.84 s |
| Ministral 3 3B     | 2025-12  |   18/19 |                0/11 |  0.99–1.00  |           5.71 s |
| Phi-4-mini         | 2025-02  |   17/19 |                0/11 |  0.99–1.00  |           5.34 s |
| SmolLM3-3B         | 2025-07  |   17/19 |                0/11 |        1.00 |           5.01 s |
| Gemma 4 E2B        | 2026-03  |   17/19 |                0/11 |        1.00 |          21.67 s |
| Gemma 3 4B         | 2025-03  |    9/19 |                7/11 |        1.00 |          31.58 s |
| Gemma 3 1B         | 2025-03  |   13/19 |                9/11 |  0.99–1.00  |           7.21 s |
| Llama 3.2 1B       | 2024-09  |    8/19 |                9/11 |  0.98–0.99  |           1.95 s |
| Granite 4.0 1B     | 2025-10  |    6/19 |               11/11 |  0.00–0.01  |           3.95 s |
| Granite 4.0 350M   | 2025-10  |    6/19 |               10/11 |  0.99–1.00  |           1.54 s |
| LFM2.5-2.6B        | 2026-07  |    6/19 |               11/11 |        0.00 |           3.13 s |

Findings:
- **Working models are all about 3B and up**, except Qwen3.5-2B, which is also the only one clearly faster
  than Qwen3.5-4B (about 2x on the long state). Its department answer on the README example was a near-tie
  (billing 0.53, technical 0.46).
- **Granite 4.2 3B and Granite 4.0 Micro put essentially all probability on one option** for every
  question, and reversing the options did not move Granite 4.2 at all. Stable, but the probabilities carry
  little information beyond the argmax. Worse calibration than Qwen's is likely; unmeasured.
- **Gemma models are slow with llav**: Gemma 4 E2B was accurate but 4.5x slower than Qwen3.5-4B on the long
  state, Gemma 3 4B 6.6x. Both use sliding-window attention; an interaction with slot save and restore is
  the suspected cause, not confirmed. Gemma 3 4B was also biased toward `A`.
- **Failures come in two kinds.** Granite 4.0 1B (`**`) and LFM2.5-2.6B (`The`, 92–95%) want to start a
  sentence, so the letters get no mass and the softmax over them is noise. Gemma 3, Llama 3.2 1B and Granite
  4.0 350M do answer with a letter but by position: reversing the options flips most choices. Since
  2026-09-24 the first kind is refused at startup when no letter is among the likely next tokens
  (`Engine._probe_labels`); when the letters get some mass but little, `X-Llav-Candidate-Mass` shows it.
- **The speed gain from small models is capped** by per-question overhead on this laptop (see
  [performance.md](performance.md#where-a-requests-time-goes)).

**Harder questions.** The pinned models were then run on 17 harder questions with a defensible answer:
sarcasm, negation, implicature, pronoun reference, and small counting and date steps (for example "moved
from Tuesday to Thursday, then pushed back one more day"). Same day, same laptop.

| Model              | Hard  | Misses (probability on the wrong answer)                                          |
|--------------------|------:|-----------------------------------------------------------------------------------|
| Qwen3.5-4B         | 17/17 | none                                                                              |
| Granite 4.0 H Tiny | 14/17 | meeting day (1.00), apple count (0.64), ticket rule (0.99)                        |
| Qwen3.5-2B         | 13/17 | sarcasm (0.57), pronoun (0.50), meeting day (0.78), apple count (0.53)            |
| Granite 4.2 3B     | 13/17 | pronoun (1.00), meeting day (0.94), apple count (0.55), "still a problem?" (0.11) |
| SmolLM3-3B         | 10/17 | seven, including two plain negations at 0.80 and 0.89                             |

- Only Qwen3.5-4B handled the multi-step items; every other pinned model missed the meeting day.
- How a model is wrong matters as much as how often. Qwen3.5-2B's misses sit near 0.5, so its probabilities
  flag its uncertainty. The Granites miss at 0.94 to 1.00, which no threshold can catch.
- SmolLM3 misreads negation ("not a single tester reported a crash"), which puts it below its easy-set
  score.

**Larger mixture-of-experts models.** Two models with about 3B active parameters were run through the same
easy, hard, order and timing checks, Q8_0, after stopping every other llama-server:

| Model                            | Total / active | Q8_0    | Easy  | Hard  | Flips | Letter mass | 10 q, long state |
|----------------------------------|---------------:|--------:|------:|------:|------:|------------:|-----------------:|
| Qwen3.5-4B (reference)           |      4B dense  |  4.6 GB | 19/19 | 17/17 |  0/11 |        1.00 |           4.56 s |
| Qwen3.6-35B-A3B                  |      36B / 3B  | 36.9 GB | 19/19 | 17/17 |  0/11 |        1.00 |          10.44 s |
| Nemotron 3.5 Lightning 30B-A3B   |      32B / 3B  | 33.6 GB | 19/19 | 16/17 |  0/11 |  0.94–0.98  |          16.50 s |

- Neither beat Qwen3.5-4B here, because Qwen3.5-4B already answers every question right; these checks
  cannot show what a larger model adds. That needs harder labelled data.
- Active parameters do not set the speed on this iGPU: at 3B active, both were 2.3x and 3.6x slower than the
  4B dense model. Granite 4.0 H Tiny (about 1B active) showed the same, matching the dense 4B.
- Nemotron leaks 2 to 6% of its probability to tokens like `The` and `Yes`, and picked technical (0.93) for
  the README's failing-payouts example.
- Neither is pinned in `scripts/fetch-model.sh`. Qwen3.6-35B-A3B is the candidate for a quality option on a
  machine with a large GPU; unmeasured there.

Ranking of the pinned models as candidates for the default, revised 2026-09-22 after the harder questions
and re-timed with the state cache and the trim probe in place. Same laptop, 1,800-token state, 5 questions;
"repeat" is the same state arriving in a later request. Qwen3.5-4B stays the default; switching needs an
accuracy run comparable to its `shape777` result, not this screening.

| Rank | Model              | Path      | Cold 5 q | Repeat 5 q |  Easy |  Hard | Reason                                                                               |
|-----:|--------------------|-----------|---------:|-----------:|------:|------:|--------------------------------------------------------------------------------------|
|    1 | Qwen3.5-4B         | slot-file |   4.07 s |     0.88 s | 19/19 | 17/17 | Only validated model; the only one that answers every question right; no order flips |
|    2 | Qwen3.5-2B         | slot-file |   1.54 s |     0.42 s | 18/19 | 13/17 | Fastest pinned model, 2.1 GB, same family and template; misses sit near 0.5          |
|    3 | Granite 4.0 H Tiny | slot-file |   2.30 s |     0.89 s | 18/19 | 14/17 | Best non-Chinese, and fastest of them; wrong at 0.99 to 1.00 when it misses; 7.4 GB  |
|    4 | Granite 4.2 3B     | trim      |   3.70 s |     0.98 s | 18/19 | 13/17 | No order bias at all, but probabilities saturate at 0 or 1, including on misses      |
|    5 | SmolLM3-3B         | trim      |   3.02 s |     0.77 s | 17/19 | 10/17 | Misreads negation; the prompt carries today's date (see [gotchas.md](gotchas.md))    |

This ranking weighs accuracy first, and with the native helper on a fast GPU it needs no trade-off at all: the
default is then the fastest on repeats as well (see
[performance.md](performance.md#what-the-helper-does-to-the-model-choice)). Without the helper, on speed alone
Granite 4.2 3B answers a first request faster than the default on both machines measured (4.52 s against 4.98
s on the laptop, 0.52 s against 0.92 s on an RTX 3090, 10 questions), because it needs no slot restore. On
repeat requests the default is still ahead on the laptop and behind on the 3090.

The optimizations did not reorder anything. They cut every model's repeat cost to between 0.42 s and 0.98 s,
which narrows the speed argument for a weaker model, and the trim path is worth less than the model's own
size: Granite 4.2 3B trims and is still slower than Granite 4.0 H Tiny, which restores a slot file but
activates about 1B parameters.

Muse Glimmer 30B was pinned later (2026-09-24) and is not in this ranking: it needs about 18 GB of GPU memory,
which the laptop does not have. On the labelled set it matched the default's accuracy with better
calibration; see "Muse Glimmer 30B, a test".

`scripts/fetch-model.sh` pins the default plus Qwen3.5-2B, Granite 4.0 H Tiny, Granite 4.2 3B and SmolLM3-3B.
Granite 4.0 Micro was dropped: Granite 4.2 3B matched or beat it on every measure. A few-shot prompt or a
fine-tuned readout might rescue the failing models, but either changes the prompt format and needs its own
validation.

## Comparison with CLM

[CLM](https://github.com/Contrastive-LM/CLM) (commit 7956937, head `CLM-v0.1-8B`, 2026-09-19 release) is a
trained System One model: a frozen Qwen3-8B, last-token pooled through vLLM, with a 20M-parameter projection
head trained contrastively on about 90M examples. It embeds the state plus instructions and each option
separately and softmaxes their scaled cosines, so it has no answer letters and no option order. It serves
`POST /v1/systemone` with TypeSafe's `type` field (its README example nests the type instead; the server
does not). Run on 2026-09-24 on the same RTX PRO 4000 box with `scripts/evaluate.py --model clm-latest`,
vLLM 0.30.0, BF16, llav stopped. The setup reproduced CLM's README example within 0.05 on the choice and
score answers (0.83 against 0.41 on its noul).

| Source                        | llav acc | CLM acc | llav ECE | CLM ECE |
|-------------------------------|---------:|--------:|---------:|--------:|
| DBpedia, 14 options           |    98.2% |   22.8% |    0.013 |   0.305 |
| BoolQ, noul                   |    92.5% |   79.5% |    0.038 |   0.064 |
| SemIf evidence interpretation |    95.8% |   81.2% |    0.052 |   0.053 |
| SemIf rule application        |    89.6% |   25.0% |    0.076 |   0.707 |
| SemIf candidate selection     |    95.8% |   50.0% |    0.065 |   0.276 |
| AG News, 4 options            |    80.5% |   41.0% |    0.113 |   0.223 |
| Banking77 card intents        |    75.0% |   18.8% |    0.124 |   0.411 |
| Yelp stars, score             |    59.0% |   30.5% |    0.204 |   0.365 |
| All                           |    82.9% |   39.7% |    0.069 |   0.261 |

llav's column is after the letter fix, uncalibrated.

- CLM is order-invariant by construction and measured so: 0 flips in 8,072 rotations. It is also wrong on
  most of these tasks, and confidently: its answers above 0.9 were right 62% of the time. Stability does not
  make an answer right, which matches what SemIf found for its per-option reranker.
- Spot checks rule out a broken setup for the classification failures: "Lionel Messi scored twice as
  Barcelona won the league" came out Business over Sports, "Wenamu River is a river in South America" came
  out WrittenWork. CLM's published results are on agent action choice (computer use, games, tool calls,
  verifying agent runs), which this set does not cover; it may be strong there and weak on topic, intent
  and evidence questions (inference).
- Speed is where CLM wins: 1,176 questions in 61 s against llav's 107 s, with an 8B encoder, because the
  state is one embedding and options are cached vectors. The shift run's 8,072 rotations took 2 s, all
  cache hits.
- For llav: no reason to adopt per-option embedding scoring, and the order-sensitivity numbers above are
  worth their cost in context: a readout that compares the options in one prompt answered these tasks far
  better than one that scores them apart. Agent action choice is untested for llav and would need its own
  labelled source before comparing there.

## Muse Glimmer 30B, a test

Meta's `Muse-Glimmer-30B` (Apache 2.0), `meta-models/Muse-Glimmer-30B-GGUF` Q4_K_M, 16.8 GB, SHA-256
checked, on the RTX PRO 4000 box, 2026-09-24, llama.cpp built that day. Through llama-server only: the helper
would need a second copy of the weights, which does not fit 24 GB. A dense 30B with hybrid attention: local
sliding-window layers (2,048 tokens) between global ones.

- **Unusable as llav stands.** Every request failed with "No answer label among the returned token
  probabilities". The chat template ends at `<|start|>assistant`, and the model's next token is ` to` at
  probability 1.0: its messages open with a recipient header (`to=user` or `to=self`, as the system turn
  lists), so no letter can come next. The template also adds "Reasoning strength: high." to the system
  turn; `enable_thinking: false` does nothing to it.
- **With a header prefilled it works.** Appending ` to=user<|message|>` to the template tail (a box-only
  patch, not in the repo) put 0.63 on A, 0.06 on B and 0.03 on C for the README ticket. With
  ` to=user<|channel|>final<|message|>` it preferred to open JSON (`{"`, 0.47).
- **Sliding-window attention defeats prefix reuse without `--swa-full`.** The startup probe chose the
  slot-file path, and every question re-read the whole state even after a restore: 10 questions on a
  1,759-token state evaluated 18,048 tokens on a repeat request and took 19.7 s. With
  `--llama-arg=--swa-full` the probe chose trimming and the same request evaluated 399 tokens:

  | 10 questions, 1,754-token state | Default | `--swa-full` |
  |---------------------------------|--------:|-------------:|
  | First request                   | 14.25 s |       1.94 s |
  | Repeat request                  | 19.72 s |       0.62 s |

  GPU memory was about the same (16 GB); the model has 2 KV heads, so the full window costs little at llav's
  8,192-token slots. This is likely the cause of the Gemma slowness under "Other models" (inference; Gemma
  was not rerun).
- **Accuracy matches Qwen3.5-4B overall, calibration is better.** `benchmark.py accuracy`: 19/19 easy, 17/17
  hard, 0/11 order flips, rule placement 0.93 to 1.00. `evaluate.py score` on the 1,176 labelled questions
  (with the prefill):

  | Source                        | Muse acc | Qwen acc | Muse ECE | Qwen ECE |
  |-------------------------------|---------:|---------:|---------:|---------:|
  | DBpedia                       |    98.7% |    98.2% |    0.057 |    0.013 |
  | SemIf evidence interpretation |    97.9% |    95.8% |    0.080 |    0.052 |
  | SemIf candidate selection     |   100.0% |    95.8% |    0.176 |    0.065 |
  | BoolQ                         |    90.5% |    92.5% |    0.046 |    0.038 |
  | Banking77 card intents        |    86.1% |    75.0% |    0.053 |    0.124 |
  | SemIf rule application        |    83.3% |    89.6% |    0.114 |    0.076 |
  | AG News                       |    75.5% |    80.5% |    0.078 |    0.113 |
  | Yelp stars                    |    54.0% |    59.0% |    0.225 |    0.204 |
  | All                           |    82.9% |    82.9% |    0.037 |    0.069 |

  Answers above 0.9 averaged 0.957 and were right 97.3% of the time, against Qwen's 0.986 and 93.2%.
- **Candidate mass is low even when right:** median 0.62, minimum 0.18, every answer below 0.9; the rest goes
  to tokens like `{"` and `The`. The web UI's "No option fits" cue (`LOW_MASS`, 0.9) would flag every Muse
  answer, so that cue is Qwen's, not general. The no-fit check still separated: 0.67 when an option fits,
  0.45 when none does.
- **Speed:** 321 s for the labelled set without `--swa-full` and without the helper, against 107 s for Qwen
  with it. With `--swa-full` the repeat request above (0.62 s) is about three times Qwen's with the helper.

Since 2026-09-24 llav serves it with no options: `templates.detect` recognises the template and appends
the header, `--swa-full` is a default, and the profile's low-mass cue is 0.35, where the labelled set split
into 35% right below and 86% right above (78 and 1,098 questions). Run that way on the box it reported
`prefix_reuse: trim`, answered 19/19 easy and 17/17 hard, and took 1.87 s for a first request of 10
questions and 0.63 s for a repeat. `muse-glimmer-30b` is pinned in `scripts/fetch-model.sh`.

The native helper is a different case. It never rolls a sequence back: it decodes the state once, copies it
to one sequence per question and extends each copy, which needs only the last window. Measured on Gemma 3 1B
(sliding-window, Q8_0) on the box the same day, llav with the helper, 10 questions on a 636-token state:

| Helper context           | GPU memory, llav total | Largest probability difference vs llama-server | Repeat request |
|--------------------------|-----------------------:|-----------------------------------------------:|---------------:|
| `swa_full = true`        |              6,856 MiB |                                         0.0001 |        0.057 s |
| default (no full window) |              4,222 MiB |                                         0.0001 |        0.054 s |

The helper runs with the default; a full window costs memory in proportion to its (sequences + 1) x context
positions and bought nothing.

`--swa-full` is now in the flags `LlamaProcess` passes. On Qwen3.5-4B, which has no sliding-window layers, it
changed nothing, measured the same day with `benchmark.py timing` and `accuracy` before and after:

| Qwen3.5-4B, 10 questions, 1,820 tokens | First request, before / after | Repeat, before / after |
|----------------------------------------|------------------------------:|-----------------------:|
| RTX PRO 4000, llama-server             |              0.73 s / 0.75 s |        0.42 s / 0.42 s |
| Arc B390 laptop, Vulkan, llama-server  |              4.57 s / 4.47 s |        1.71 s / 1.73 s |

Evaluated tokens, GPU memory (4,971 MiB on the box), answers and candidate masses were identical; the probe
still chose the slot-file path. The native helper creates its own context and does not read the flag; its
repeat request stayed at 124 ms.

## Comparison with Laya and von

Two open, trained System One models, run on 2026-09-24 on the RTX PRO 4000 box beside llav with
`scripts/evaluate.py --model`, on the same 1,176 labelled questions. Both serve `/v1/systemone` with the
`type` field.

- **Laya** 0.3.20 (Convai Innovations, Apache 2.0): ModernBERT-large, 421M parameters, fine-tuned with a
  decision head and trained against a proper scoring rule (its "RLCD"). Two checkpoints were run: `laya`
  (English, reads 512 tokens) and `laya-typed-decisions` (1,024 tokens).
- **von** 1.2 (`von-sdk` 1.2.2, Apache 2.0): ModernBERT-large, trained on NLI and operational data; each
  option attends only to the text and itself, so option order cannot matter.

| Source                          | llav, Qwen3.5-4B | llav, Muse 30B |  Laya | Laya typed |   von |   CLM |
|---------------------------------|-----------------:|---------------:|------:|-----------:|------:|------:|
| DBpedia                         |            98.2% |          98.7% | 90.2% |      87.5% | 90.2% | 22.8% |
| SemIf evidence interpretation   |            95.8% |          97.9% | 66.7% |      70.8% | 58.3% | 81.2% |
| SemIf rule application          |            89.6% |          83.3% | 60.4% |      68.8% | 62.5% | 25.0% |
| SemIf candidate selection       |            95.8% |         100.0% | 58.3% |      58.3% | 31.2% | 50.0% |
| BoolQ                           |            92.5% |          90.5% | 74.0% |      77.0% | 84.0% | 79.5% |
| Banking77 card intents          |            75.0% |          86.1% | 75.5% |      77.4% | 80.8% | 18.8% |
| AG News                         |            80.5% |          75.5% | 93.5% |      93.5% | 86.0% | 41.0% |
| Yelp stars                      |            59.0% |          54.0% | 38.0% |      43.5% | 58.0% | 30.5% |
| All                             |            82.9% |          82.9% | 73.0% |      74.8% | 76.4% | 39.7% |
| ECE, all                        |            0.069 |          0.037 | 0.102 |      0.042 | 0.115 | 0.261 |
| Right when above 0.9            |            93.2% |          97.3% | 90.7% |      96.0% | 89.3% | 62.2% |
| Answer changes under some order |              19% |                |   32% |        30% |    0% |    0% |

llav's order figure leaves out candidate selection, which `_align_letters` fixed after that run; Muse was
not run through `shifts`.

- **llav with Qwen3.5-4B is ahead overall** by 6.5 to 10 points, with the largest leads on SemIf's reasoning
  questions (evidence, rules, candidates: 21 to 65 points) and DBpedia. The encoders read a claim against
  evidence poorly; they were trained for classification and routing.
- **The encoders win on news topics**: both Laya checkpoints reach 93.5% on AG News against llav's 80.5%.
  von is best on the Banking77 card intents among the small models, 80.8% against llav's 75.0%.
- **Laya typed-decisions is the best calibrated of the small models** (ECE 0.042) without any fitting; its
  answers above 0.9 were right 96.0% of the time. Laya's claim of calibrated probabilities holds for that
  checkpoint here, not for the English one (0.102).
- **von is order-invariant as designed**, 0 of 8,072 rotations flipped, and still reached 76.4%, unlike CLM.
  Order invariance and accuracy are not in conflict for it; its weak spot is the SemIf items (31.2% on
  candidate selection, below chance with three options).
- **Laya is order-sensitive**, more than llav: 30 to 32% of answers change under some rotation. On Yelp 80 to
  93% change, and the English checkpoint picked whichever level was listed first 38% of the time against
  20% for an order-blind model; it reads a rating scale's order.
- **Speed**: the labelled set took 28 to 31 s for all three encoders against 107 s for llav with the native
  helper on the same GPU. Laya's 512-token English checkpoint truncates the longest BoolQ passages and Yelp
  reviews (not measured how many).

Request time measured by the client on the box over localhost, a 300-token support ticket, 3-option choice
questions, median of 10; "first" is a text the server has not seen, "repeat" the same text again. Each system
ran without the others competing for the GPU.

| Questions per request | llav, Qwen3.5-4B + helper |       Laya | Laya typed |          von |
|-----------------------|--------------------------:|-----------:|-----------:|-------------:|
| 1, first / repeat     |                88 / 29 ms | 26 / 26 ms | 28 / 27 ms |   26 / 26 ms |
| 5, first / repeat     |               147 / 88 ms | 34 / 34 ms | 34 / 34 ms | 126 / 125 ms |
| 10, first / repeat    |              231 / 171 ms | 59 / 59 ms | 60 / 60 ms | 249 / 249 ms |

- Laya is 3 to 4 times faster than llav and grows slowly with the number of questions: it batches them.
- von costs about 25 ms per question, as if it scored each question separately; at 10 questions it is
  slower than llav.
- Neither encoder caches: first and repeat cost the same. llav's repeat saves the state read (about 60 ms
  here), which grows with the text; the encoders cannot read texts past 512 or 1,024 tokens at all.
- Memory: llav with the helper held 15.2 GB (llama-server 5 GB plus the helper's own copy and a context for
  17 sequences), Laya 5.9 GB for its checkpoints, von 3.5 GB.

A first timing run started Laya and von beside llav with the helper, 24.5 GB against the card's 24 GB. On
llav's next request the helper failed ("the helper sent a malformed response") and exited; llav answered
through llama-server from then on, as designed, but `/v1/models` kept reporting `prefix_reuse: native`. What
the helper wrote is lost; running out of GPU memory is the likely cause (inference). Fixed since: the helper
answers on a private channel with stdout pointed at stderr, a failure quotes its stderr, llav stops a dropped
helper, and `/v1/models` reports the fallback with `native_error`. A second run with both encoders loaded
(20 GB in use) did not fail, so whether the stray output caused the first failure is unconfirmed.

## Against the Jevals board

[Jevals](https://jevals.com/) scores Jev and six LLMs on the same 300 items per task against human labels
(release 2026-09-18, suite 0.1.0, CC-BY-4.0). `scripts/evaluate.py fetch` rebuilds two of its tasks item for
item: Jevals' own instructions, criteria and labels, with every text checked against its published hash
(600 of 600 matched, and the label counts match the suite's). `score` reports Jevals' Decision Score (dscore),
100 x (1 - loss / loss of answering the label base rates), Brier for noul and ranked probability score for
score; recomputed from Jevals' logs it reproduces Jev's 69.03 and 9.20. Run on 2026-09-24 on the RTX PRO
4000 box, llav with Qwen3.5-4B and the native helper, uncalibrated.

The two tasks:

- **PubMedQA** (noul): passages from a medical research paper and a research question; is the answer yes?
  62% of the answers are yes.
- **HelpSteer2 helpfulness** (score): a prompt and a chatbot's reply; how helpful was the reply, on five
  levels from "not helpful at all" to "extremely helpful"? Rated by human annotators.

How to read the numbers:

- **Decision Score** rates the probabilities, not only the answer. 0 is what always answering the label
  frequencies scores (on PubMedQA, "62% yes" for every question, without reading it); 100 is always right
  with full confidence; below 0 is worse than that blind guess, which confident mistakes cause.
- **Accuracy** is the share of items where the most probable answer was the right one. The last row answers
  with the label frequencies without reading anything: that is a score of 0 by definition, and its accuracy
  is that of always picking the most common label.

| System                              | PubMedQA score | PubMedQA accuracy | HelpSteer2 score | HelpSteer2 accuracy |
|-------------------------------------|---------------:|------------------:|-----------------:|--------------------:|
| Gemini 3.8 Flash                    |           73.0 |             92.5% |              4.6 |               42.4% |
| Jev                                 |           69.0 |             91.3% |              9.2 |               41.3% |
| Jev, run here through OpenRouter    |           69.1 |             91.3% |              9.5 |               41.3% |
| Qwen3.8 Flash                       |           62.4 |             89.7% |             -1.4 |               36.1% |
| GLM-5.3                             |           60.6 |             88.7% |              7.8 |               43.0% |
| Mistral Medium 3.5                  |           58.0 |             88.8% |            -13.7 |               43.9% |
| Mercury 2.5                         |           55.7 |             87.1% |             -5.5 |               41.7% |
| DeepSeek V4.1 Flash                 |           47.5 |             83.7% |            -19.0 |               34.7% |
| **llav, Qwen3.5-4B**                |       **46.5** |         **80.7%** |        **-11.1** |           **39.0%** |
| llav, temperature fitted (held out) |           47.4 |             80.7% |             -6.3 |               39.0% |
| von 1.2                             |            7.7 |             68.3% |            -17.0 |               37.3% |
| Laya typed-decisions                |           -6.3 |             62.3% |              6.2 |               37.3% |
| Laya English                        |          -31.3 |             59.3% |              2.0 |               35.0% |
| Label frequencies, reads nothing    |              0 |             62.0% |                0 |               41.7% |

The first seven rows are Jevals' board; "Jev, run here" and the llav, von and Laya rows were run here the
same day, and the Jev rerun matching the board shows the rebuilt items are the same. Laya's
English checkpoint reads at most 512 tokens and its typed-decisions checkpoint 1,024, so the longer
PubMedQA passages and HelpSteer2 replies were cut short for them. "Temperature fitted (held out)" corrects llav's overconfidence with a
temperature fitted on half of each task and scored on the other half, both ways round; it changes how sure
llav is, never which answer it picks, so accuracy stays the same.

- **PubMedQA: last, just behind DeepSeek V4.1 Flash.** 80.7% right against Jev's 91.3%. The questions ask
  whether biomedical passages support a yes, a harder reading task than most of llav's own set. A fitted
  temperature (1.27) adds under a point: the misses are wrong answers, not overconfidence.
- **The small trained encoders cannot read these passages.** On PubMedQA von (68.3%) and Laya (59.3 to 62.3%)
  are at or near the 62% that answering yes every time gets; their scores run from 7.7 down to -31.3. On
  llav's own set they were 6.5 to 10 points behind; here the gap to llav is 12 to 21 points of accuracy. Why
  is below.
- **HelpSteer2: nobody beats guessing by much.** Every score is within 20 points of 0, and every accuracy is
  within a few points of the 41.7% that always answering "extremely helpful" gets; llav's 39.0% is below it.
  llav's -11.1 rises to -6.3 with a temperature of 2.3, between Mercury 2.5 and Mistral Medium 3.5. Jevals'
  own verdict is that no model clearly beats the label base rates here.
- **Not identical conditions.** Jevals asked each item five times; llav is deterministic, so one pass stands
  for all five. Its LLM rows reason (at a low setting) and state their probabilities in text; Jev and llav
  read them from the model. Jevals' latencies are over the network; llav's 600 answers took 76 s on the box.
- Jevals' third task, Banking77 with all 77 intents, needs more than llav's 26 options and was not run.

**Why the encoders fail on PubMedQA.** Jevals puts the research question inside the state and gives every item
the same instruction, "is the answer to the research question yes?". The same 300 items were run again, the
same day, with each item's own research question as the instruction and only the passages as the state
(a scratch variant, not in `fetch`):

| PubMedQA, 300 items  | Says yes, Jevals' format | Right on no, Jevals' format | Accuracy (score), Jevals' format | Accuracy (score), question as instruction | Right on yes / no, question as instruction |
|----------------------|-------------------------:|----------------------------:|---------------------------------:|------------------------------------------:|-------------------------------------------:|
| llav, Qwen3.5-4B     |                      164 |                         84% |                     80.7% (46.5) |                          **86.0% (57.0)** |                                  87% / 84% |
| von 1.2              |                      229 |                         39% |                      68.3% (7.7) |                               64.7% (4.0) |                                  60% / 73% |
| Laya typed-decisions |                      291 |                          4% |                     62.3% (-6.3) |                               66.0% (7.3) |                                  89% / 29% |
| Laya English         |                      274 |                          8% |                    59.3% (-31.3) |                              63.7% (-2.1) |                                  88% / 24% |

The truth is 186 yes and 114 no.

- **Mainly, the encoders cannot tell a yes item from a no item.** With the question stated directly, von's
  probability of yes stays between 0.36 and 0.63 for 80% of items, and Laya typed-decisions' between 0.45
  and 0.70: near a coin toss whatever the passages say. llav's runs from 0.02 to 1.00. Answering needs
  reading findings such as "no significant difference" against the question; these ModernBERT-sized
  encoders, trained for classification, routing and simple entailment, do not do that reading (inference,
  consistent with their 20 to 65 point deficit on SemIf's evidence and rule items above).
- **Jevals' format makes it worse, most for Laya.** With the generic instruction Laya answers yes almost
  regardless: 4 to 8% right on the no items. The item's own question lifts it about 4 points of accuracy.
- **Truncation is a minor factor.** The median state is about 1,450 characters, roughly 350 tokens, inside
  even Laya's 512. Accuracy falls clearly only on the 11 states over 2,000 characters.
- **llav also gains from the question as instruction**: 80.7% to 86.0%, and a Decision Score of 57.0 that
  would sit between GLM-5.3 and Mistral Medium 3.5 on the board. The comparable number stays 46.5, since every
  row on the board used Jevals' format. The wording advice this gives callers is in
  [design.md](design.md#question-mapping).

## Jev through OpenRouter

TypeSafe's Jev (`typesafe/jev-1.13`, served as `typesafe/jev-1.13-20260917`) run on 2026-09-24 through
OpenRouter's `https://openrouter.ai/api/v1/systemone`, with `scripts/evaluate.py --model typesafe/jev-1.13
--api-key-file`, from the laptop. Same 1,176 labelled questions and the same rotations as every other row.
The whole run cost well under a dollar at $0.042 per million input tokens.

**Reproducing Jevals.** On Jevals' two rebuilt tasks this run gave PubMedQA 69.1 (91.3%) and HelpSteer2 9.5
(41.3%), against Jevals' published 69.0 (91.3%) and 9.2 (41.3%): same accuracy, scores within 0.3. The item
reconstruction and the Decision Score match Jevals', so the llav, von and Laya rows beside Jev there compare
fairly.

| Source                          |   Jev | llav, Qwen3.5-4B | llav, Muse 30B |   von | Laya typed |
|---------------------------------|------:|-----------------:|---------------:|------:|-----------:|
| DBpedia                         | 99.1% |            98.2% |          98.7% | 90.2% |      87.5% |
| SemIf evidence interpretation   | 95.8% |            95.8% |          97.9% | 58.3% |      70.8% |
| SemIf rule application          | 97.9% |            89.6% |          83.3% | 62.5% |      68.8% |
| SemIf candidate selection       | 95.8% |            95.8% |         100.0% | 31.2% |      58.3% |
| BoolQ                           | 93.5% |            92.5% |          90.5% | 84.0% |      77.0% |
| Banking77 card intents          | 88.5% |            75.0% |          86.1% | 80.8% |      77.4% |
| AG News                         | 85.0% |            80.5% |          75.5% | 86.0% |      93.5% |
| Yelp stars                      | 72.5% |            59.0% |          54.0% | 58.0% |      43.5% |
| All                             | 89.0% |            82.9% |          82.9% | 76.4% |      74.8% |
| ECE, all                        | 0.040 |            0.069 |          0.037 | 0.115 |      0.042 |
| Right when above 0.9            | 96.3% |            93.2% |          97.3% | 89.3% |      96.0% |
| Answer changes under some order |    6% |              19% |                |    0% |        30% |

llav's order figure leaves out candidate selection, fixed since by `_align_letters`; Muse was not run through
`shifts`.

- **Jev is the most accurate system measured, 6 points ahead of llav overall.** The lead is small where the
  answer is clear (DBpedia, BoolQ, SemIf's evidence and candidates: 0 to 1 point), 8 points on SemIf's rule
  application, and largest on confusable options: 13.5 points on the Banking77 card intents and on Yelp's
  star ratings. Muse Glimmer
  closes most of the Banking gap (86.1%) but not Yelp's.
- **Jev is well calibrated without fitting** (ECE 0.040; answers above 0.9 right 96.3% of the time), as
  TypeSafe claims for these tasks. Muse under llav is comparable (0.037); Qwen under llav needs a fitted
  temperature to get there (0.023 held out, see [research.md](research.md#temperature-calibration)).
- **Jev is less order-sensitive than llav, not immune.** 6.4% of questions changed answer under some option
  rotation: Yelp 15.5%, Banking 10.1%, AG News 4.0%, DBpedia 0.4%, against llav's 35.5%, 32.2%, 8.5% and 3.6%.
  Part of that is not order: Jev is not deterministic, and Jevals measured 2.3 to 2.7% of identical choice and
  score requests changing answer. The questions that flipped were near-ties (median top probability 0.6).
- Jev rounds probabilities to two decimals, exact zeros included, so a wrong answer at 0 costs NLL the
  clipped maximum; its NLL column is inflated by that, not by bad probabilities (AG News 1.735 against an
  ECE of 0.083).

**Speed.** Measured from the laptop, end to end over the network:

| Request                                  |  Jev, median |  Jev, fastest |
|------------------------------------------|-------------:|--------------:|
| 1 question, short text                   |       345 ms |        313 ms |
| 1 question, 300-token text               |       316 ms |        303 ms |
| 5 questions, same text                   |       327 ms |        283 ms |
| 10 questions, same text                  |       416 ms |        330 ms |
| Jevals' logs, PubMedQA / HelpSteer2      | 438 / 479 ms | 357 / 362 ms |

- About 140 ms of that is the network (a plain request to OpenRouter took that long), so Jev's own time is
  roughly 150 to 300 ms per request (estimate; the network figure is from another endpoint). Jevals' 95th
  percentiles are 653 to 693 ms. TypeSafe's "70 to 500 ms end to end" is its upper half from outside.
- More questions per request cost little: 10 questions took about 100 ms more than one.
- TypeSafe's "40 to 200 times faster than frontier LLMs" holds only against slow generators: Jevals' fastest
  LLM, Gemini 3.8 Flash, answered in a median of 1,636 ms, about 4 times Jev's time.
- A local llav on a GPU next to the client is faster than Jev over the network: 88 ms for one question and
  231 ms for ten on a new 300-token text, 29 and 171 ms on a repeated one, on the RTX PRO 4000
  ([performance and memory above](#comparison-with-laya-and-von)). On the laptop's iGPU llav is slower than
  Jev (1.7 s for a repeat of 10 questions on an 1,800-token text).
- Throughput, one request at a time: 1,176 questions (about 1,145 requests) in 413 s and 8,072 rotations
  (976 requests) in 363 s, about 2.7 to 2.8 requests a second.

## Open questions

- Accuracy of the alternative models in `scripts/fetch-model.sh` on labelled data: measured for Muse Glimmer
  30B only. `scripts/evaluate.py score` against each of the others would answer it.
- Muse Glimmer 30B through the native helper: untested. The helper holds a second copy of the weights, so it
  needs about 35 GB of GPU memory; the 24 GB box cannot hold both. The helper runs without a full
  sliding-window cache, which Gemma 3 1B showed is enough (see "Muse Glimmer 30B, a test"); that it holds
  for Muse is an inference.
- Agent action choice (tool calls, commands, UI actions), where CLM reports its results: no labelled source
  in `scripts/evaluate.py`, so llav is unmeasured there.
- Why Gemma models are 4 to 7 times slower under llav: sliding-window attention without `--swa-full`
  is the likely cause, since it caused the same symptom on Muse Glimmer (see that section); Gemma has
  not been rerun with the flag.
