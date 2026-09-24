# Research behind the design

Measurements that shaped llav. The first sections were made in the SemIf repository before llav existed,
with its `benchmarks/llama_server_probe.py`; those outputs were kept outside any repository, so the numbers
below are the record. Later sections use llav's own `scripts/benchmark.py` and `scripts/evaluate.py` and
name their date and machine. Unless a section says otherwise: a laptop with an Intel Arc B390 iGPU,
llama.cpp build 10809, Qwen3.5-4B Q8_0 GGUF (bartowski, the revision pinned in `scripts/fetch-model.sh`).

## Contents
- Prompt fidelity and agreement
- Reusing a shared state
- Backend and runtime comparisons
- Where a request's time goes
- Other models
- Rejected approaches
- Candidate mass
- Labelled evaluation: calibration and option order
- Temperature calibration
- The normalizer's cost in the native helper
- Comparison with CLM
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

## Reusing a shared state

Per 21-question state:

| Approach                                              | Decisions/s | Notes                                            |
|-------------------------------------------------------|-------------|--------------------------------------------------|
| Fresh scoring, no cache                               | 0.27        | Full prompt every time                           |
| Serial with prompt cache                              | 0.69        | Partial reuse through checkpoints                |
| Serial, primed with the state prefix                  | 1.1 to 1.25 | Full reuse, but two checkpoint saves per request |
| Parallel slots (21 at once), primed or not            | 0.24        | Every slot recomputes the whole state            |
| Primed, copied to 21 slots via save/restore, parallel | 1.05 to 1.57| Reuse works; parallel suffixes do not batch      |
| **Serial-restored, `--ctx-checkpoints 0`** (llav)     | 2.3 to 2.8  | About 20 ms per restore, about 0.2 s per question |
| SemIf torch XPU shared mode, same GPU (reference)     | 5.6 to 6.7  | One batched pass for all 21 questions            |

The full 777-decision serial-restored run averaged 2.28 decisions/s; a 3-state run reached 2.75. The slower
full run had longer state reads, possibly GPU throttling (not confirmed).

## Backend and runtime comparisons

`llama-bench`, prompt processing, tokens per second:

| Test                                  | Vulkan | SYCL  |
|---------------------------------------|--------|-------|
| 77 tokens                             | 506    | 405   |
| 512 tokens                            | 872    | 1,306 |
| 1,891 tokens                          | 721    | 804   |
| 77 tokens after 1,812 in context      | 305    | 335   |

SYCL gained about 3% end to end on serial-restored, but needs the oneAPI toolkit and
`LD_LIBRARY_PATH=/opt/intel/oneapi/<version>/lib`. Vulkan is the default.

## Where a request's time goes

Measured 2026-09-22 with `scripts/benchmark.py phases` on the laptop above, Qwen3.5-4B, 10 questions on a
1,793-token state, before the state cache and the trim probe existed. Every request paid for the state; the machine comparison below is with
both in place.

| Phase                                | Time   | Share |
|--------------------------------------|-------:|------:|
| Erase, prime and save the state      | 2.80 s |   60% |
| Readouts (10 x 157 ms)               | 1.57 s |   34% |
| Restores (10 x 20 ms)                | 0.20 s |    4% |
| llav's own work (template, tokenize) | 0.07 s |  1.3% |

The readout cost is nearly fixed: 124 ms on an 83-token state, 137 ms at 1,049, 149 ms at 2,057. About
120 ms per question is llama.cpp's small-batch overhead, not context. `n_probs` 16 against 128 changes
nothing.

Measured on two more machines, 2026-09-22, with `scripts/benchmark.py` on all three: the laptop, a MacBook
Air M3 (10-core GPU, 24 GB, Metal, llama.cpp build 10964, Python 3.14) and a vast.ai RTX 3090 (24 GB, CUDA,
Python 3.12), same Qwen3.5-4B Q8_0. The 39 unit tests pass on all three.

| Phase, 10 questions on a 1,800-token state | Arc B390 iGPU | MacBook Air M3 | RTX 3090 |
|--------------------------------------------|--------------:|---------------:|---------:|
| Prime the state                            |        3.09 s |         5.40 s |   0.29 s |
| Save the slot file                         |        433 ms |          26 ms |    59 ms |
| Restore per question                       |         17 ms |          14 ms |    34 ms |
| Readout per question                       |        234 ms |         203 ms |    27 ms |
| llav's own work per question               |          6 ms |           3 ms |    24 ms |
| First request, 10 questions                |        5.11 s |         7.69 s |   1.27 s |
| Repeat request, 10 questions               |        1.89 s |         2.20 s |   0.88 s |
| Repeat request, 5 questions                |        0.92 s |         1.09 s |   0.48 s |
| Repeat request, 1 question                 |        0.18 s |         0.22 s |   0.12 s |

All three answer identically: 19/19 easy, 17/17 hard, no order flips, probabilities within 0.02.

Readout timings vary between runs on the laptop, 157 ms to 234 ms for the same work, the higher figures
measured while a second llav held the GPU; treat single-run phase numbers as approximate and compare whole
requests where possible.

- **The M3 is 1.5 to 1.9 times slower than the Arc iGPU** on prefill. Apple's base chips are not a speed
  upgrade here; the Pro and Max parts have several times the GPU cores and are what the estimates below
  extrapolate to.
- **On the 3090 the slot restore costs more than the answer**, 34 ms against 27 ms, and it does not shrink
  with the GPU: it is the upload of a 107 MB saved state, not disk. Putting the slot directory on a RAM disk
  (`TMPDIR=/dev/shm`) changed nothing, 36 ms.
- **So the trim path is worth much more on a fast GPU than on the laptop.** Granite 4.2 3B restores
  nothing, which makes it the faster model for a first request on both machines once the trim path exists:
  0.52 s against Qwen's 0.92 s on the 3090, and 4.52 s against 4.98 s on the laptop, for 10 questions. On
  repeats the machines disagree: 0.31 s against 0.44 s on the 3090, but 0.98 s against 0.88 s on the
  laptop, where Qwen's faster readout outweighs its restore. The faster the GPU, the more a model that
  passes the probe is worth, which the pinned-model ranking does not capture.
- **llav's own per-question work is hardware-dependent too**: 24 ms on the vast.ai host against 6 ms on the
  laptop, from slower per-core CPU. It is 1% of a laptop request but 25% of a 3090 one, so the template and
  tokenize round trips are worth removing if llav is ever tuned for fast GPUs.
- **A saved state is about 107 MB for this model**, so `--state-cache 4` can hold roughly 430 MB in the slot
  directory.
- **Writing that file costs wildly different amounts**: 433 ms on the Arc laptop, 59 ms on the 3090, 26 ms on
  the M3, where unified memory spares the copy. It is paid once per state with the cache on, so it matters
  most for states asked about only once.

### The pinned models on a fast GPU

Same scripts on the RTX 3090, 2026-09-23, 1,800-token state:

| Model          | Path      | Cold 10 q | Repeat 10 q | Repeat 1 q |  Easy |  Hard | Restore | Readout | Slot file |
|----------------|-----------|----------:|------------:|-----------:|------:|------:|--------:|--------:|----------:|
| Qwen3.5-4B     | slot-file |    1.16 s |      0.65 s |     0.08 s | 19/19 | 17/17 |   31 ms |   26 ms |    106 MB |
| Qwen3.5-2B     | slot-file |    0.66 s |      0.40 s |     0.05 s | 18/19 | 14/17 |   16 ms |   17 ms |     40 MB |
| Granite 4.2 3B | trim      |    0.71 s |      0.30 s |     0.08 s | 18/19 | 13/17 |   41 ms |   21 ms |    134 MB |

- Qwen3.5-2B's state is 40 MB against the 4B's 106 MB, so its restore costs half as much. It is the fastest
  first request of the three and close to Granite on repeats, while keeping the default's family, template
  and a better hard score. On a fast GPU it, not Granite, is the speed pick.
- Granite pays a 136 ms save once and then no restores, which is why it leads on repeats.
- Hard scores move by one between runs (Qwen3.5-2B: 13/17 on the laptop, 14/17 here) because several of its
  answers sit near 0.5.

### The native readout helper

`native/llav-readout.cpp` keeps the state resident and decodes every question of a request in one
`llama_decode`, instead of one llama-server pass per question. Measured through llav's own API with
`--native-readout`, Qwen3.5-4B, 10 questions on an 1,800-token state:

| Machine  | Path         | Cold 10 q | Repeat 10 q | Repeat 5 q | Repeat 1 q |
|----------|--------------|----------:|------------:|-----------:|-----------:|
| Arc iGPU | llama-server |    5.11 s |      1.89 s |     0.92 s |     0.18 s |
| Arc iGPU | native       |    3.78 s |      1.26 s |     0.72 s |     0.19 s |
| RTX 3090 | llama-server |    1.16 s |      0.65 s |     0.38 s |     0.08 s |
| RTX 3090 | native       |    0.82 s |      0.34 s |     0.16 s |     0.04 s |

- The gain grows with the GPU, as predicted: 1.5x on the laptop's repeat requests, 1.9x on the 3090's, and
  2.4x for five questions there, because the fixed costs it removes are a larger share when the compute is
  fast.
- With the helper, the default model matches the trim models: 0.34 s against Granite 4.2 3B's 0.30 s on a
  repeat of 10, while keeping 17/17 on the hard questions. That weakens the case for a faster, weaker model
  on fast hardware.
- Answers are unchanged: 19/19 easy, 17/17 hard, no order flips. Probabilities differ from the llama-server
  path by at most 1.3e-5 for an attention model and 0.002 for Qwen3.5, against 0.02 between machines; the
  batched arithmetic is not bit-identical.
- The helper compiled unchanged against the Arch package's `llama.h` and against a from-source CUDA build.

### What the helper does to the model choice

The three pinned models that fit the box, measured on the RTX 3090 with `--native-readout`, 2026-09-23,
1,800-token state:

| Model          | Cold 10 q | Repeat 10 q |  Easy |  Hard |
|----------------|----------:|------------:|------:|------:|
| Qwen3.5-4B     |    0.75 s |      0.21 s | 19/19 | 17/17 |
| Granite 4.2 3B |    0.62 s |      0.27 s | 18/19 | 13/17 |
| SmolLM3-3B     |    0.62 s |      0.25 s | 17/19 | 10/17 |

With the helper the default is both the fastest on repeat requests and the most accurate, so the reason to
run a weaker model on fast hardware is gone. The smaller models keep a small lead on a first request, where
the state prefill still dominates and their smaller weights help; once a state is cached, the per-question
fixed costs the helper removed were exactly what they had been saving.

Without the helper the order is the opposite on repeats (Granite 0.30 s against Qwen's 0.65 s), so the
advice depends on whether the helper is built: see "The pinned models on a fast GPU" above.
SmolLM3's run predates the criteria fold, so its rule placement is unmeasured.

### Cutting llav's own cost per question

Two changes on 2026-09-23, after the helper made llav's own work a visible share of a request on a fast GPU:

- **The state is tokenized once per request, not once per question** (`Engine.encode_all`). The chat
  template is rendered once per process, checked against a second probe payload so a template that folded
  the payload into its head or tail is refused; the evidence opening is tokenized once; each question
  re-tokenizes only from the character where the shared part stops. The first question is compared with the
  whole-payload tokenization and the whole request falls back to `encode` on any difference. Prompt building
  for 10 questions on an 1,800-token state went from 99 ms to 9 ms, tokens identical.
- **The helper returns raw logits instead of normalized log-probabilities.** llav softmaxes them over the
  declared options, which cancels the normalizer, so computing it meant a pass over all 248,320 vocabulary
  entries per question for nothing. The normalizer came back on 2026-09-24 for candidate mass, vectorized
  and threaded so it costs about 2 ms per 10-question request; see "The normalizer's cost in the native
  helper" below.

Effect on a repeat request of 10 questions, same workload as above:

| Machine  | llama-server | Native before | Native after |
|----------|-------------:|--------------:|-------------:|
| Arc iGPU |       1.73 s |        1.26 s |       1.22 s |
| RTX 3090 |       0.65 s |        0.34 s |       0.21 s |

The 3090 gains where the laptop barely moves: prompt building was 26% of its request and under 1% of the
laptop's. Against the llama-server path it is now 3.1x on repeats there. Answers are unchanged: 19/19 easy,
17/17 hard, no order flips.

### Real articles, not hand-written sentences

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

### Where the decision rule is written matters, per model

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

### Against generating text

The usual way to get a decision from a local model is to send the state and question as a chat prompt, let
the model write an answer, and parse it. llav avoids three costs of that approach.

- **No generation.** Each question is one forward pass, and the answer is read from the logits of the next
  token. A text answer needs one sequential decode step per output token, plus whatever reasoning the model
  writes first. With thinking enabled, Qwen3.5 can write hundreds of tokens before it answers.
- **The state is read once per request.** Reading the prompt is most of the cost. A 1,800-token state
  takes 2.5–3.4 s to evaluate, and a question on top of it 0.2–0.25 s. Sending each question as its own
  prompt re-reads the state every time. llama-server's built-in prompt cache helps less than expected with
  this model, because its hybrid layers force a checkpoint copy off the GPU on every request (see How it
  works).
- **No parsing or retries.** The answer is a probability over the declared options, so there is no
  malformed output to detect and re-ask, and no grammar to maintain.

Asking all the questions in one text prompt would also read the state once. But the model then writes the
answers one token at a time, each answer can sway the ones after it, and you get labels without
probabilities.

Measured on the setup below, for 21 yes/no questions about one state:

| Approach                                               | Decisions/s |
|--------------------------------------------------------|------------:|
| A fresh prompt per question, no cache                  |        0.27 |
| A prompt per question with llama-server's prompt cache |        0.69 |
| llav (state evaluated once, restored per question)     |     2.3–2.8 |

All three rows use the single-pass readout; none of them generates text, and all three evaluate the state
for the first time; a cached state raises llav's row to about 5 decisions/s. A generate-and-parse baseline
was not measured, so the no-generation gain comes on top of these numbers but has no figure of its own.
the sections above has the details.

### Expected scaling on other hardware

**Only the rows marked measured are measurements.** The rest extrapolate from them; llav has not been run
on that hardware. The Apple estimates now scale from the measured M3 by GPU core count, so they moved down
from an earlier guess.

A request has two parts that scale differently:
- **Evaluating the state** is a large batch of tokens, limited by GPU compute. It speeds up roughly in line
  with the GPU, and a state the cache already holds skips it entirely.
- **Each question** is a pass of about 80 tokens, two HTTP round trips, and, on a backend that needs the
  slot file, a restore. A pass that short is limited by per-step overhead rather than by context: on the
  laptop a readout takes 124 ms on an 83-token state and 149 ms on a 2,057-token one. Expect a floor of
  roughly 20–40 ms per question even on the fastest cards.

So the number of questions, not the length of the state, sets the time on a large GPU, and it is all that
is left once the state is cached.

Estimated, 21 questions on a 1,800-token state, first request and a repeat of the same state, with the
default model. "Readout + restore" splits the per-question cost, because only the readout gets faster with
the GPU; a model that passes the trim probe drops the restore entirely. The measured rows carry their
measured phase timings out to 21 questions; the three-machine table above has the runs themselves.

| Hardware                    |      State | Readout + restore | First request | Repeat request | Decisions/s, repeat |
|-----------------------------|-----------:|------------------:|--------------:|---------------:|--------------------:|
| Arc B390 iGPU (measured)    |  2.5–3.4 s |       157 + 20 ms |         6.5 s |          3.7 s |                   6 |
| MacBook Air M3 (measured)   |     5.35 s |       203 + 12 ms |        10.0 s |          4.5 s |                   5 |
| Apple M4 Pro/Max (Metal)    |  1.3–2.1 s |    80–150 + 15 ms |     3.5–5.3 s |      2.0–3.5 s |                6–11 |
| RTX 3090 (measured)         |     0.35 s |        26 + 33 ms |         1.6 s |          1.2 s |                  17 |
| RTX 4090 / 5090 (CUDA)      | 0.1–0.25 s |  15–25 + 25–30 ms |     1.0–1.4 s |      0.8–1.2 s |               18–26 |
| H100                        |     ~0.1 s |  15–20 + 20–30 ms |     0.8–1.2 s |      0.7–1.1 s |               19–30 |
| 16-core desktop CPU, no GPU |    10–20 s |   0.5–1 s + 50 ms |       20–40 s |        11–22 s |                 1–2 |

- **The largest uncertainty is Qwen3.5's recurrent layers.** llama.cpp's kernels for them are newer and
  less tuned than its attention kernels. The GPU rows could be off by a factor of two.
- **An H100 gains little over a 4090.** A 4B model is too small to use it; the per-question overhead sets
  the limit, and on a repeat request it is the only cost left.
- **On a fast GPU the slot restore costs more than the answer** (33 ms against 26 ms on the 3090), because
  it uploads the saved state, about 107 MB for this model. A model that passes the trim probe skips it
  entirely, and the gap grows with the GPU: Granite 4.2 3B answered a first 10-question request in 0.52 s
  against the default's 0.92 s on that 3090, and 4.52 s against 4.98 s on the laptop.
- **Concurrency across states** with `--slots` may scale better on large GPUs than on the laptop. Untested.
- **Past about 20 decisions/s, software matters more than hardware.** The per-question floor comes from
  llama.cpp evaluating each question's tokens in its own pass; only a batched decode over a shared prefix
  removes it, and llama-server's HTTP API cannot express that. See
  the section above.

Tried and rejected, same workload:
- **llama-server flags.** Flash attention is already on (`-fa auto`); forcing it off costs 30% (6.01 s
  against 4.65 s). `-ub 1024 -b 2048` is slower (5.47 s), and with `-fa on` slower still (5.99 s).
- **Parallel slots, now on an attention-only model too.** Granite 4.2 3B, 12 questions: 2.14 s serial
  against 4.51 s over 4 slots, and probabilities moved by up to 0.29. The earlier hybrid result holds for
  dense models.

Adopted:
- **Skipping the slot file where the backend allows it.** Granite 4.2 3B and SmolLM3 reuse the prompt cache
  across questions with identical logprobs (max difference 0.0000), saving the 0.4 s save and 20 ms per
  question. Qwen3.5-4B fails the same test: 13.38 s against 1.08 s for five questions, recomputing the whole
  prompt each time, and its logprobs differ by 0.06, so it must keep the file. `Engine` decides with a
  startup probe.
- **Caching evaluated states across requests.** Qwen3.5-4B, same 1,800-token state: five questions 3.89 s
  cold against 0.85 s on a repeat, one question 3.20 s against 0.18 s, probabilities identical. Turning the
  cache on leaves a cold request unchanged for a restoring backend, which already wrote the file (3.85 s
  against 3.91 s with `--state-cache 0`, 10 questions on a 1,200-token state), and costs a trimming one the
  save it would otherwise skip (Granite 4.2 3B, same workload, 3.57 s against 3.18 s). The first repeat
  pays that back.

Still open: batching every question's suffix into one `llama_decode` over a shared prefix would remove the
120 ms floor. SemIf's torch shared mode reached 5.6 to 6.7 decisions/s that way against 2.3 to 2.8 here.
llama-server's HTTP API cannot express it.

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
  4.0 350M do answer with a letter but by position: reversing the options flips most choices.
- **The speed gain from small models is capped** by per-question overhead on this laptop (see "Where a
  request's time goes").

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

This ranking weighs accuracy first, and with the native helper on a fast GPU it needs no trade-off at all:
the default is then the fastest on repeats as well (see "What the helper does to the model choice"). Without
the helper, on speed alone Granite 4.2 3B answers a first request faster than the default on both machines
measured (4.52 s against 4.98 s on the laptop, 0.52 s against 0.92 s on an RTX
3090, 10 questions), because it needs no slot restore. On repeat requests the default is still ahead on the
laptop and behind on the 3090.

The optimizations did not reorder anything. They cut every model's repeat cost to between 0.42 s and 0.98 s,
which narrows the speed argument for a weaker model, and the trim path is worth less than the model's own
size: Granite 4.2 3B trims and is still slower than Granite 4.0 H Tiny, which restores a slot file but
activates about 1B parameters.

`scripts/fetch-model.sh` pins the default plus Qwen3.5-2B, Granite 4.0 H Tiny, Granite 4.2 3B and SmolLM3-3B.
Granite 4.0 Micro was dropped: Granite 4.2 3B matched or beat it on every measure. A few-shot prompt or a
fine-tuned readout might rescue the failing models, but either changes the prompt format and needs its own
validation.

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
  supersedes the hand-read "Letter mass" column for Qwen3.5-4B under "Other models".

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

## The normalizer's cost in the native helper

Candidate mass needs the log-sum-exp over the whole vocabulary (248k logits for Qwen3.5) per question. The
first version computed it serially with `std::exp` in double after the batched decode. On the RTX PRO 4000
box, a repeat request of 10 questions took 310 ms with it and 122 ms with a helper that skipped it: the
pass cost about 19 ms per question there, far more than the 1.5 ms the same loop took on an ordinary array
in a microbenchmark (why is not investigated; the logits buffer's memory is the suspect). The current
helper uses a polynomial exp (at most 5e-6 relative error) that the compiler vectorizes and splits the
vocabulary across `--threads`: 124 ms, 2 ms above not computing it at all.

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

## Open questions

- Accuracy of llav's own wording against labelled data: measured on public samples and `authored144` (see
  "Labelled evaluation"); not yet against `shape777`-scale data or callers' own workloads.
- Whether a BF16 GGUF closes the remaining probability differences, which would attribute them to Q8_0
  quantization: untested.
- The native helper batches questions on libllama as torch's shared mode does; a direct throughput
  comparison with SemIf's torch runner on the same GPU has not been run.
- Accuracy of the alternative models in `scripts/fetch-model.sh` on labelled data: unmeasured.
  `scripts/evaluate.py score` against each would answer it.
- Why reading llama.cpp's logits buffer costs about ten times more than reading an ordinary array
  (see "The normalizer's cost in the native helper"): not investigated.
- Agent action choice (tool calls, commands, UI actions), where CLM reports its results: no labelled source
  in `scripts/evaluate.py`, so llav is unmeasured there.
- Names that collide with answer letters only inside the state ("Candidate A" with keys `first`, `second`):
  `_align_letters` does not cover them, and no labelled set tests them.
- Why Gemma models are 4 to 7 times slower under llav (sliding-window attention with slot restore is the
  suspect): not investigated.
- Throughput beyond the three machines measured here: the scaling table is an extrapolation; replace its
  rows with measurements when a machine becomes available.
