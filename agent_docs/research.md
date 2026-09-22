# Research behind the design

Measurements that shaped llav, made in the SemIf repository before llav existed. The probe script is
`benchmarks/llama_server_probe.py` in SemIf; its outputs were kept outside any repository, so the numbers
below are the record. Hardware for all timings: a laptop with an Intel Arc B390 iGPU, llama.cpp build 10809,
Qwen3.5-4B Q8_0 GGUF (bartowski, the revision pinned in `scripts/fetch-model.sh`).

## Contents
- Prompt fidelity and agreement
- Reusing a shared state
- Backend and runtime comparisons
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
All Q8_0. The script was not kept; the numbers below are the record. Nineteen easy questions separate
broken models from working ones; they do not rank the working ones.

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

Ranking of the pinned models as candidates for the default, decided 2026-09-22. Qwen3.5-4B stays the
default; switching needs an accuracy run comparable to its `shape777` result, not this screening.

| Rank | Model              | Reason                                                                                                                                                                           |
|-----:|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    1 | Qwen3.5-4B         | Only validated model; 19/19, no flips; probabilities informative (billing 0.85, technical 0.15)                                                                                  |
|    2 | Qwen3.5-2B         | Twice as fast, 2.1 GB, same family and template, so the validation likely transfers best; one flip moved a probability by 0.86 and the README department question was a near-tie |
|    3 | Granite 4.2 3B     | Best non-Chinese; no order bias, but probabilities saturate at 0 or 1                                                                                                            |
|    4 | Granite 4.0 H Tiny | 18/19 with less saturated probabilities than the other Granites; 7.4 GB and no faster on the laptop                                                                              |
|    5 | SmolLM3-3B         | 17/19; softer probabilities; the prompt carries today's date (see [gotchas.md](gotchas.md))                                                                                      |

`scripts/fetch-model.sh` pins the default plus Qwen3.5-2B, Granite 4.2 3B, Granite 4.0 H Tiny and SmolLM3-3B.
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
