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
- **`criteria` text does not weigh the same as `instructions` on every model.** Granite 4.2 3B answers the
  criterion and treats option descriptions as labels: a rule written only into `criteria.true` scored 0.017
  where Qwen3.5-4B scored 0.987. `_fold_noul` now repeats a noul's criteria in the criterion, which fixed
  both; `scripts/benchmark.py accuracy` checks it on any model. Keep the fold when touching
  `parse_question`, and remember that `choice` and `score` are still exposed to this.
- **`n_probs` probabilities depend on temperature.** Only `temperature < 0` gives a plain softmax over raw
  logits. With sampling settings or `post_sampling_probs`, the numbers reflect the sampler chain and are
  not the readout llav was validated with.
- **A label can fall outside the top `n_probs`.** `_readout` then gives it the smallest returned logprob, an
  upper bound on its true value. Raising `n_probs` (an `Engine` argument) tightens the bound at the cost of
  response size. The candidate mass on the llama-server path inherits the bound; the native helper's is
  exact.
- **Option keys that are letters collide with the answer letters.** Keys `A`, `B` shown in another order put
  "B: Candidate B" at answer letter A, and the model answers the name. `_align_letters` in `questions.py`
  prevents it by showing such keys at their own letter; keep it when touching `parse_question`. Names in the
  state alone ("Candidate A" with keys `first`, `second`) are not covered.
- **The native helper's protocol is versioned in its ready line.** A binary built before the candidate-mass
  normalizer was added is refused at startup with "rebuild it from native/". Rebuild after pulling.
- **`llama_decode` aborts the process on a batch larger than `n_batch`.** It fails
  `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` instead of returning an error. The helper decoded every
  suffix in one batch, so 12 Banking77 questions (2,108 tokens, `n_batch` 2048) killed it, and llav then ran
  without the helper for the rest of its life; a state over 2,048 tokens did the same through the prefix
  decode. The helper now splits both into chunks of `llama_n_batch(ctx)` tokens and copies each chunk's
  logits before the next decode. Chunked answers match unchunked ones as closely as the batched decode
  matches llama-server. Found 2026-09-25; a helper built from older source still crashes, so rebuild.
- **Scripts that stop llav on a box.** A background job in a non-interactive shell ignores SIGINT, so
  `kill -INT` never stops it; send SIGTERM, which `cli.py` handles like Ctrl-C. And a `pgrep -f` pattern
  that appears literally in the calling script's own command line matches that shell: anchor it
  (`pgrep -f "^python3 -m llav"`) or bracket a character. Both cost runs on 2026-09-25 and 26.
- **Reading llama.cpp's logits buffer is slow.** A serial pass over the 248k-entry vocabulary per question
  cost about 19 ms on the RTX PRO 4000 box, against 1.5 ms over an ordinary array, and made a repeat request
  2.5 times slower. `log_normalizers` in the helper vectorizes and threads it; keep any new per-logit work
  there (see [performance.md](performance.md#the-normalizers-cost-in-the-native-helper)).
- **A calibration file is tied to one model file and one prompt version.** llav refuses it for any other,
  so bumping `PROMPT_VERSION` or changing the pinned GGUF invalidates every caller's file; say so in the
  change. `scripts/evaluate.py fit` refuses predictions scored against a calibrated llav
  (`X-Llav-Calibration` other than `none`), since fitting on calibrated numbers would compound them.
- **Labels must be single tokens with a clean boundary.** `Engine.__init__` refuses to start if any of
  `A`–`Z` is not one token, and `_boundary_ok` rejects a chat template whose tail merges with a label. A
  new model or template can fail here at startup.
- **SmolLM3's chat template puts today's date in the system turn.** Its prompt changes daily, so answers
  can shift slightly from one day to the next. Nothing is cached across requests, so the date change does
  not break prefix reuse. The template also switches thinking off only through `enable_thinking`; check the
  rendered prompt ends in an empty `<think>` block after any template change.
- **llama-server's log is deleted with llav's scratch directory.** A startup failure therefore quotes the log's
  last lines (`LlamaProcess._log_tail`) rather than naming the file; keep that for any new startup error.
- **Sliding-window models need `--swa-full` for prefix reuse.** Without it llama-server re-reads the whole
  state for every question even after a slot restore, and the startup probe picks the slot-file path: 19.7 s
  against 0.62 s for a repeat request of 10 questions on Muse Glimmer 30B. `LlamaProcess` passes it; an
  external llama-server (`--llama-url`) must be started with it. Models without sliding windows ignore it. The
  native helper deliberately does not set `swa_full`: it only extends a prefix, and a full window over its
  many sequences costs gigabytes. See [comparisons.md](comparisons.md#muse-glimmer-30b-a-test).
- **Some chat templates stop before the answer can start.** Muse Glimmer's template ends at
  `<|start|>assistant`, and the model must write a recipient header (` to=user<|message|>`) first; without
  it no label is in the top `n_probs`. gpt-oss (OpenAI's Harmony format) is the same case: the next token
  is `<|channel|>` with all the mass, and the profile supplies `<|channel|>final<|message|>` (found
  2026-09-26; gpt-oss-20b then scores 18/19 and 16/17 on `benchmark.py accuracy`). `templates.detect`
  recognises both templates and supplies the header; a new template of this kind needs an entry in `_KNOWN`
  there, or `--assistant-prefix`. `_probe_labels` makes an unknown one fail at startup with a hint instead
  of on every request.
- **A model can pass the startup probe and still not answer with letters.** The probe only requires a
  label among the top `n_probs` on one easy question. Gemma 4 12B passes it, then on real questions puts 2%
  of its mass on the letters and wants to write prose (`Based`, `The`; with thinking correctly off), and no
  assistant prefix tried (`Answer: `, `The answer is `, `Letter: `) made a letter its next token. It ignores
  "respond with only its uppercase letter"; the E4B of the same family follows it and scores like
  Qwen3.5-4B. Screening (`agent_docs/comparisons.md`) catches this, the probe does not; a stricter probe
  (mass on the labels above some share) is an open question.
- **The low candidate-mass cue is per template profile.** 0.9 suits Qwen3.5-4B (0.999 when answering, 0.57
  when no option fits); Muse Glimmer answers correctly at a median of 0.62, so its profile uses 0.35. The web
  UI and `scripts/evaluate.py` read it from `backend.low_candidate_mass`. Muse's no-fit signal is weaker
  (0.47 against 0.71 on one ticket), so the cue misses cases that Qwen's catches.
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
  The cost: a client still writing a rejected body gets a broken pipe rather than the status. Measured with
  a 9 MiB body against both a local server and one over a network; draining the body first would hand back
  a readable 413 but reintroduce exactly the work the early reject avoids.

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
