# Gotchas and findings

## Contents
- Traps
- Findings

## Traps

- **llama-server cannot roll back a recurrent state without checkpoints.** Qwen3.5 mixes attention with
  recurrent layers. With `--ctx-checkpoints 0`, a request that only partially matches a slot's cache is
  recomputed from scratch, never trimmed. This is why the shared path erases, primes, saves and restores,
  and why single questions use `cache_prompt: false`.
- **Context checkpoints are expensive on this model.** Each save copies about 50 MB of recurrent state off
  the GPU. Turning them back on roughly triples per-question latency; see [research.md](research.md).
- **Never reuse a slot's cache across questions without checking `Engine.trims`.** On Qwen3.5 a partial
  cache match recomputes the whole prompt (13.38 s against 1.08 s for five questions) and returns
  probabilities that differ by up to 0.06, because the recurrent state never rolled back. The startup probe
  decides; a wrong answer there is slow and quietly wrong, not an error.
- **A cached slot file is only valid for the run that wrote it.** Filenames carry a per-run id, so a file
  left in a `--slot-dir` by an earlier llav is never restored. Restoring a file from another model or
  context size would be silently wrong.
- **`n_probs` probabilities depend on temperature.** Only `temperature < 0` gives a plain softmax over raw
  logits. With sampling settings or `post_sampling_probs`, the numbers reflect the sampler chain and are
  not the readout llav was validated with.
- **A label can fall outside the top `n_probs`.** `_readout` then gives it the smallest returned logprob, an
  upper bound on its true value. Raising `n_probs` (an `Engine` argument) tightens the bound at the cost of
  response size.
- **Labels must be single tokens with a clean boundary.** `Engine.__init__` refuses to start if any of
  `A`–`Z` is not one token, and `_boundary_ok` rejects a chat template whose tail merges with a label. A
  new model or template can fail here at startup.
- **SmolLM3's chat template puts today's date in the system turn.** Its prompt changes daily, so answers
  can shift slightly from one day to the next. Nothing is cached across requests, so the date change does
  not break prefix reuse. The template also switches thinking off only through `enable_thinking`; check the
  rendered prompt ends in an empty `<think>` block after any template change.
- **More than 26 choice options needs a different readout.** Two-letter labels are not single tokens in
  Qwen's vocabulary; raising `MAX_OPTIONS` alone breaks the boundary check.
- **The `--chat-template-kwargs` flag logs a deprecation warning** in current llama.cpp, which suggests
  `--reasoning off`. The prompt fidelity result was measured with `--chat-template-kwargs` plus
  `chat_template_kwargs` on `/apply-template`; re-verify prompt hashes before switching.
- **Slot files go to the llama-server's `--slot-save-path`, not llav's working directory.** With
  `--llama-url`, `--slot-dir` must name that same directory, or llav cannot delete the files after use.
- **SYCL needs oneAPI libraries on the library path.** `ggml-sycl` loads only with
  `LD_LIBRARY_PATH=/opt/intel/oneapi/<version>/lib` (or `setvars.sh`); otherwise llama.cpp silently lists
  only the Vulkan device.
- **`HTTPServer` does a reverse DNS lookup on bind.** Its `server_bind` calls `socket.getfqdn()`, which took
  35 s for `127.0.0.1` with Homebrew Python 3.14 on macOS (Apple's Python 3.9 on the same host: instant).
  `server.Server` skips it; use `Server`, not `ThreadingHTTPServer`, for new listeners, including fakes in
  tests.
- **Responses sent before reading the body must close the connection.** With HTTP/1.1 keep-alive, an unread
  body would be parsed as the next request. `do_POST` adds `Connection: close` to every early response.

## Findings

### Two checkpoint copies per request

- **Symptom:** questions on a primed state took 0.6 to 1.2 s through llama-server, against 0.25 s for the
  same work in `llama-bench`.
- **Diagnosis:** with `-lv 4`, each request restored the prefix checkpoint (2 ms), then erased it as "too
  close" and saved it again (about 180 ms), then saved another checkpoint near the end of the prompt (about
  170 ms).
- **Fix:** `--ctx-checkpoints 0`, with the prefix saved once to a slot file and restored before each question.
- **Takeaway:** now a global invariant in [CLAUDE.md](../CLAUDE.md#invariants).

### Caller text could forge chat turns

- **Symptom:** tokenizing the whole rendered prompt with special-token parsing turned `<|im_end|>`-style
  text inside the state or criteria into real control tokens.
- **Fix:** `Engine._split` tokenizes the user payload with `parse_special: false` and only the template
  head and tail with parsing on. Tests: `test_caller_text_cannot_inject_control_tokens`,
  `test_split_tokenization_matches_whole_prompt_for_plain_text`.
- **Takeaway:** a global invariant in [CLAUDE.md](../CLAUDE.md#invariants).
