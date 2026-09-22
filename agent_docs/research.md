# Research behind the design

Measurements that shaped llav, made in the SemIf repository before llav existed. The probe script is
`benchmarks/llama_server_probe.py` in SemIf; its outputs were kept outside any repository, so the numbers
below are the record. Hardware for all timings: a laptop with an Intel Arc B390 iGPU, llama.cpp build 10809,
Qwen3.5-4B Q8_0 GGUF (bartowski, the revision pinned in `scripts/fetch-model.sh`).

## Contents
- Prompt fidelity and agreement
- Reusing a shared state
- Backend and runtime comparisons
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
- Throughput on other GPUs (CUDA, Metal): unmeasured.
