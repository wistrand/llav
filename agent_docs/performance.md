# Performance

Where a request's time goes, how state reuse and the native helper cut it, and how fast llav is on the
machines measured. Accuracy is in [research.md](research.md); other models' speed is in
[comparisons.md](comparisons.md).

Measurements that shaped llav. The first sections were made in the SemIf repository before llav existed,
with its `benchmarks/llama_server_probe.py`; those outputs were kept outside any repository, so the numbers
below are the record. Later sections use llav's own `scripts/benchmark.py` and `scripts/evaluate.py` and
name their date and machine. Unless a section says otherwise: a laptop with an Intel Arc B390 iGPU,
llama.cpp build 10809, Qwen3.5-4B Q8_0 GGUF (bartowski, the revision pinned in `scripts/fetch-model.sh`).

## Contents
- Reusing a shared state
- Backend and runtime comparisons
- Where a request's time goes
- Against generating text
- Expected scaling on other hardware
- The normalizer's cost in the native helper
- Asking a question in several orders
- Open questions

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
1,793-token state, before the state cache and the trim probe existed. Every request paid for the state; the
machine comparison below is with both in place.

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

## Against generating text

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

## Expected scaling on other hardware

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

## The normalizer's cost in the native helper

Candidate mass needs the log-sum-exp over the whole vocabulary (248k logits for Qwen3.5) per question. The
first version computed it serially with `std::exp` in double after the batched decode. On the RTX PRO 4000
box, a repeat request of 10 questions took 310 ms with it and 122 ms with a helper that skipped it: the
pass cost about 19 ms per question there, far more than the 1.5 ms the same loop took on an ordinary array
in a microbenchmark (why is not investigated; the logits buffer's memory is the suspect). The current
helper uses a polynomial exp (at most 5e-6 relative error) that the compiler vectorizes and splits the
vocabulary across `--threads`: 124 ms, 2 ms above not computing it at all.

## Asking a question in several orders

Measured 2026-09-26 on an RTX PRO 4000 Blackwell, Qwen3.5-4B Q8_0: one 1,800-token state, one 7-option choice
with described options, asked in K orders through `scripts/orders-proxy.py` (all orders in one request).
"Repeat" is the median of 8 requests with the state already cached, the per-question cost; "first" reads
the state. Script and raw numbers: `agent_docs/experiments/scripts/orders-timing.py`,
`local/results/perturb/orders-timing.json`.

| Orders K | Helper, first | Helper, repeat | llama-server, first | llama-server, repeat |
|---------:|--------------:|---------------:|--------------------:|---------------------:|
|        1 |       0.30 s |        0.046 s |             0.31 s |              0.062 s |
|        2 |       0.32 s |        0.069 s |             0.36 s |              0.110 s |
|        4 |       0.38 s |        0.121 s |             0.21 s |              0.208 s |
|        7 |       0.44 s |        0.192 s |             0.60 s |              0.354 s |
|       14 |       0.67 s |        0.420 s |             0.95 s |              0.697 s |

- The cost is close to linear in K on both paths: about 27 ms per extra order through the helper and 48 ms
  through llama-server, for a suffix of about 150 tokens (7 described options). The helper's batching
  removes the per-pass overhead, not the decode of each suffix.
- Reading the state dominates the first request (0.30 s); the orders add to it the same way.
- On JevBench's short questions (3 to 5 options), 4 orders cost 30 to 60 ms per decision
  ([comparisons.md](comparisons.md)). The gain those orders buy is in
  [experiments/permutation-uncertainty.md](experiments/permutation-uncertainty.md): 8% fewer errors on this
  model, 17 to 22% on the smaller pinned ones.

## Open questions

- The native helper batches questions on libllama as torch's shared mode does; a direct throughput
  comparison with SemIf's torch runner on the same GPU has not been run.
- Why reading llama.cpp's logits buffer costs about ten times more than reading an ordinary array
  (see "The normalizer's cost in the native helper"): not investigated.
- Throughput beyond the three machines measured here: the scaling table is an extrapolation; replace its
  rows with measurements when a machine becomes available.
