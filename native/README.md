# llav-readout, the optional native fast path

`llav-readout` answers every question of one request in a single forward pass. llama-server evaluates each
question separately and restores the state in between; this helper keeps the state resident and decodes all
the question suffixes together. llav uses it only when started with `--native-readout`, and falls back to
llama-server whenever it is absent or fails.

Worth it on a fast GPU, where the per-question fixed costs dominate. A repeat request of 10 questions went
from 1.73 s to 1.22 s on an Arc iGPU laptop and from 0.65 s to 0.21 s on an RTX 3090. See
[../agent_docs/performance.md](../agent_docs/performance.md) for the full numbers.

## Build

The helper needs `llama.h` and `libllama` from the same llama.cpp version llama-server comes from. Every
install route ships them:

```bash
# Arch (llama-cpp package), or any system where llama.h is on the default include path
g++ -O2 -std=c++17 -pthread -o llav-readout native/llav-readout.cpp -lllama

# Homebrew
g++ -O2 -std=c++17 -pthread -o llav-readout native/llav-readout.cpp \
    -I/opt/homebrew/include -L/opt/homebrew/lib -lllama

# A llama.cpp built from source
g++ -O2 -std=c++17 -pthread -o llav-readout native/llav-readout.cpp \
    -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
    -L/path/to/llama.cpp/build/bin -lllama -Wl,-rpath,/path/to/llama.cpp/build/bin
```

## Run

```bash
PYTHONPATH=src python3 -m llav --gguf MODEL.gguf --native-readout ./llav-readout
```

On a rented GPU box, `NATIVE=1 scripts/remote-gpu.sh HOST PORT` builds the helper there and starts llav
with it.

The helper loads the model itself, so it needs `--gguf`, not `--llama-url`, and it holds a second copy of
the weights in memory alongside llama-server's. For Qwen3.5-4B that is about 5 GB more; Muse Glimmer 30B
(16.8 GB) with the helper does not fit a 24 GB GPU. `--native-questions N` sets how many questions it takes in
one pass (default 16); a request with more goes to llama-server. The helper decodes the prefix and the
suffixes in chunks of its batch size, so a long state or many long suffixes do not exceed it. `GET /v1/models` reports
`backend.prefix_reuse: native` while it is in use.

## Notes

- The helper also returns each question's log-sum-exp over the vocabulary, for `X-Llav-Candidate-Mass`.
  The pass is split across the helper's `--threads`, 4 since llav does not pass the flag. Computed serially
  it cost 19 ms per question on the RTX PRO 4000 box, more than the batched decode (see
  [agent_docs/performance.md](../agent_docs/performance.md)).
- The helper does not keep a full sliding-window cache, unlike llama-server with `--swa-full`. It never rolls
  back, only extends a resident prefix forward, which needs just the last window. On Gemma 3 1B a full cache
  cost 2.6 GB more for identical answers (see [agent_docs/comparisons.md](../agent_docs/comparisons.md)).
- The helper keeps only the most recent state resident, so `X-Llav-State-Cache` reports a hit when a request
  repeats the state its predecessor used. llav's slot-file cache serves the llama-server path.
- Requests are serialized through the one helper process. With `--slots` above 1, llama-server can answer
  several requests at once and the helper cannot.
- Probabilities are not bit-identical to the llama-server path: batching changes the arithmetic. Measured
  differences are 1.3e-5 for an attention model and 0.002 for Qwen3.5, against 0.02 between machines.
  Within one batch on CUDA, identical prompts in different sequences differed by up to 0.09 (median 0) on
  Qwen3.5; no answer changed.
- The helper answers on a private copy of its original stdout and points stdout at stderr at startup, so
  anything llama.cpp, ggml or a GPU driver prints cannot corrupt an answer. When llav still gets an answer
  it cannot read, the error quotes the helper's last stderr lines.
- When llav drops the helper after a failure, it stops the process so its copy of the model leaves GPU
  memory, and `GET /v1/models` reports the fallback (`backend.prefix_reuse`) and the cause
  (`backend.native_error`).
- The helper exits when llav closes its input, so it cannot outlive the server that started it.
- Rebuild the helper when llama.cpp is upgraded. It uses the C API, so a mismatch is a compile error rather
  than silent corruption.
- Rebuild it when `native/llav-readout.cpp` changes, too. Its ready line names the protocol version, and llav
  refuses a helper built from older source ("rebuild it from native/") rather than misread its answers.
