# Research behind the design

Measurements that shaped llav, made in the SemIf repository before llav existed. The probe script is
`benchmarks/llama_server_probe.py` in SemIf; its outputs were kept outside any repository, so the numbers
below are the record. Hardware for all timings: a laptop with an Intel Arc B390 iGPU, llama.cpp build 10809,
Qwen3.5-4B Q8_0 GGUF (bartowski, the revision pinned in `scripts/fetch-model.sh`).

## Contents
- Prompt fidelity and agreement
- Reusing a shared state
- Backend and runtime comparisons
- Where a request's time goes
- Other models
- Rejected approaches
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
  upgrade here; the Pro and Max parts have several times the GPU cores and are what the README's estimates
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
- **The speed gain from small models is capped** by per-question overhead on this laptop (see the README's
  scaling notes).

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

This ranking weighs accuracy first; on speed alone Granite 4.2 3B now answers a first request faster than
the default on both machines measured (4.52 s against 4.98 s on the laptop, 0.52 s against 0.92 s on an RTX
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

## Open questions

- Accuracy of llav's own wording (choice keys folded, `Yes`/`No` nouls) against labelled data: unmeasured.
- Whether a BF16 GGUF closes the remaining probability differences, which would attribute them to Q8_0
  quantization: untested.
- A native program on libllama (`llama_memory_seq_cp` or `llama_state_seq_*` plus batched `llama_decode`)
  might match torch's batched shared mode. Untested; the Arch `llama-cpp` package ships `llama.h`.
- Accuracy of the alternative models in `scripts/fetch-model.sh` on the `shape777` workload or other labelled
  data: unmeasured.
- Why Gemma models are 4 to 7 times slower under llav (sliding-window attention with slot restore is the
  suspect): not investigated.
- Throughput on other GPUs (CUDA, Metal): unmeasured. The README's scaling table is an extrapolation from the
  laptop figures; replace it with measurements when available.
