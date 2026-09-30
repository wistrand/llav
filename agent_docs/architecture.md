# Architecture

## Contents
- Overview
- Invariants
- Components
- Request flow
- Shared-state flow
- Media requests
- Tokenization
- Slots and concurrency
- Process lifecycle
- Error mapping

## Overview

llav is a thin HTTP layer in front of `llama-server`. All model work happens in llama-server, or in the
optional `llav-readout` helper (`--native-readout`); llav validates requests, builds prompts, drives
llama-server's HTTP API, and shapes answers. Global rules are in
[CLAUDE.md](../CLAUDE.md#invariants).

**Key files**: `src/llav/engine.py` (`Engine.evaluate`, `Engine._evaluate_on`, `Engine.encode`,
`Engine._readout`), `src/llav/server.py` (`Handler.do_POST`), `src/llav/questions.py` (`parse_question`,
`build_answer`).

## Invariants

- `Engine.encode` and `Engine.state_prefix` must tokenize the same way, or the prefix check in
  `_evaluate_on` fails and every multi-question request falls back to unshared scoring. The fallback is
  correct but slow, so a regression here shows up only as latency.
- The state prefix drops its last token. That token can merge with the text after the evidence value
  (`,` or `}`), so keeping it would make the prefix disagree with the full prompt.
- The shared path erases the slot before priming a state it has not cached. Without context checkpoints,
  llama-server cannot roll a recurrent state back to a partial match, so a stale slot cache must not be
  reused.
- `Engine.trims` comes from a startup probe, not a model list: the engine primes a short prefix, extends it
  twice with different tokens, and reads `prompt_n`. An attention-only backend evaluates only the new token
  and needs no slot file between questions; a recurrent one recomputes everything and answers from a state
  that never rolled back, so it must restore the file.

## Components

| Unit            | File             | Responsibility                                                             |
|-----------------|------------------|----------------------------------------------------------------------------|
| `main`          | `cli.py`         | Parse flags; start `LlamaProcess` or attach via `--llama-url`; cleanup     |
| `Handler`       | `server.py`      | Routes, bearer auth, body limits, JSON errors, `X-Llav-*` headers          |
| `document`      | `openapi.py`     | OpenAPI 3.1 spec from the code; served at `/openapi.json`                  |
| `NativeReadout` | `native.py`      | Optional helper process: one batched pass for all questions                |
| `Calibration`   | `calibration.py` | Optional per-type temperature from `--calibration`; checks the model       |
| `detect`        | `templates.py`   | Template profile: assistant prefix and low-mass cue from the rendered text |
| `parse_request` | `questions.py`   | Validate the body into `(state, model, [Question])`                        |
| `parse_media`   | `questions.py`   | Decode the optional `images` and `audio` data URLs                         |
| `Question`      | `questions.py`   | Frozen per-question data: option ids, what the model reads, caller order   |
| `build_answer`  | `questions.py`   | Turn probabilities into a `noul` / `choice` / `score` answer               |
| `messages`      | `prompt.py`      | System instruction plus JSON payload `{evidence, criterion, options}`      |
| `LlamaClient`   | `engine.py`      | JSON over HTTP to llama-server; any failure becomes `EngineError`          |
| `Engine`        | `engine.py`      | Label token ids, encoding, slot pool, readout, shared-state orchestration  |
| `LlamaProcess`  | `runtime.py`     | Spawn llama-server with required flags, wait for `/health`, stop on exit   |
| web UI          | `webui.html`     | `--web-ui` page at `GET /`; calls the public API like any client           |

## Request flow

1. `Handler.do_POST` checks path, auth, `Transfer-Encoding` and `Content-Length` before reading the body.
2. The body is parsed with `parse_constant` rejecting `NaN`/`Infinity`, then `parse_request` validates it
   and `parse_media` decodes any `images` and `audio`. `parse_question` folds a noul's criteria into its
   criterion (`_fold_noul`) and moves single-letter choice keys to their own letter (`_align_letters`),
   keeping the caller's order in `Question.declared`.
3. The model name is checked against the served id and its aliases.
4. `Engine.evaluate` encodes every question and, for more than one question, the state prefix. All of
   this happens before a slot is taken, so tokenization does not hold capacity. A request with media
   takes the [media path](#media-requests) from here.
5. A slot is taken from `Engine.free` (or `Overloaded` after `queue_timeout`).
6. `_evaluate_on` scores each question and returns probabilities, `usage` and metadata. With the native
   helper and a shared prefix, `_evaluate_native` sends every question in one batch and gets raw label
   logits plus each question's log-sum-exp over the vocabulary; on any `NativeError` the engine drops the
   helper, stops its process and continues through llama-server; `Engine.backend_status` then reports the
   fallback and its cause to `/v1/models`. Candidate mass is the declared labels' share of the vocabulary:
   the sum of their vocabulary-normalized `n_probs` on the llama-server path, `exp(logit - normalizer)` on the
   helper path.
7. With `--calibration`, each question's probabilities are divided by its type's temperature in log space
   (`Calibration.apply`); the answer letter never changes. Candidate mass stays raw.
8. `build_answer` shapes each answer; timing goes into `X-Llav-Seconds` and `X-Llav-Shared-State-Tokens`,
   each question's candidate mass into `X-Llav-Candidate-Mass`, and the calibration id or `none` into
   `X-Llav-Calibration`.

## Shared-state flow

Used when every question's tokens start with the state prefix: any multi-question request, and a
single-question one whose prefix is at least `STATE_CACHE_MIN_TOKENS` while the state cache is on.

1. `Engine._load_prefix` leaves the slot holding exactly the prefix:
   - Cache hit: `action=restore` of the state's file, about 20 ms, and no tokens evaluated.
   - Cache miss: `action=erase`, `/completion` with the prefix and `n_predict: 0`, then `action=save` to
     `llav-<run>-<prefix digest>.bin`. The save is skipped only when nothing will restore it: a trimming
     backend with the cache off.
2. For each question: `_readout` with `cache_prompt: true`. Between questions, a non-trimming backend gets
   `action=restore` first; a trimming one rolls back by itself. The first question needs no restore either
   way, since the slot already holds the prefix.
3. With the cache off, the file is deleted in a `finally`; otherwise it stays for later requests, and
   `Engine` evicts the least recently used beyond `--state-cache` and deletes the rest in `clear_cache` at
   shutdown.

Cached files are keyed by the prefix tokens and by a per-run id, so a file from an earlier llav run is
never restored. A request whose prefix check fails uses `_readout` with `cache_prompt: false`.

## Media requests

A request with `images` or `audio` takes `Engine.encode_media` instead of `encode_all`, and never the
native helper. The two kinds share one path; the kind only decides which projector modality is required.
llama-server takes media only as `{"prompt_string", "multimodal_data"}`, tokenizing the string itself
with special-token parsing on and splitting it at the media marker, so this path cannot send the token ids
`encode` builds.

1. The marker comes from `/props` at startup (`Engine.media_marker`); llama-server randomizes it per run.
   `Engine.modalities` is what `/props` reports the projector takes (vision, audio); a request with a kind
   the projector lacks is a 422 before any backend call.
2. Each prompt string is template head, one marker per file (images then audio), the payload and the
   tail. The server sees the head as one text piece and the payload plus tail as another, so their token
   ids are those of `encode`, with the media chunks between. Caller text is checked instead of protected:
   the state's evidence opening and each question's remainder must tokenize the same with and without
   special-token parsing, or the request is a 422 (`MediaRequestError`). This replaces the
   `parse_special: false` guarantee of the text path for this path only.
3. The shareable prefix is the head, the markers and the evidence opening minus its last token, as
   `state_prefix` does, but as text: `detokenize` of those ids. It is used only when re-tokenizing that
   text gives the same ids and the first question's payload tokenizes as prefix plus remainder;
   otherwise every question is read unshared. A non-trimming backend that extended a prefix inexactly
   would answer from a state that never rolled back (see [gotchas.md](gotchas.md)), so this check is what
   makes the slot-file restore safe here.
4. `_evaluate_prompts` runs the same erase, prime, save, restore loop as the text path over prompt objects.
   Slot files carry the media chunks (llama-server serializes them, and restoring one needs the projector),
   and their names hash the prefix text and the media bytes. The server counts the prefix, media tokens
   included; that count is what `X-Llav-Shared-State-Tokens` reports for media requests.
5. llav cannot count media tokens or decode the files, so a prompt that does not fit the slot and a file
   the server cannot read are the server's errors: `_caller_error` matches them (`_CONTEXT_EXCEEDED`,
   `_MEDIA_UNREADABLE`) and raises `ContextTooLong` for the question it hit or `MediaRequestError`.

## Tokenization

`Engine._split` renders the chat turns with `/apply-template` and splits the result into template head, user
payload and template tail. Head and tail are tokenized with special-token parsing, the head also with
`add_special`, which prepends BOS for a vocabulary that adds one (the rendering never carries it as text);
the payload (all caller text) is tokenized with `special=False`. The payload sits between a newline and a control token, both token
boundaries, so for text without control tokens the split gives the same ids as tokenizing the whole prompt.

`Engine._boundary_ok` checks that appending each label to the template tail adds exactly that label's
token. The result is cached per tail string, which is constant for a given chat template, so the check
costs a handful of `/tokenize` calls once per process.

## Slots and concurrency

`Engine.free` is a queue of slot ids, sized to llama-server's slot count. Each request holds one slot for its
whole duration and pins every call to it with `id_slot`. More slots allow concurrent requests, but
llama.cpp does not batch sequences of this hybrid model efficiently, so throughput barely improves (see
[research.md](research.md)).

## Process lifecycle

`cli.main` first binds the HTTP port with `server.bind`, so a busy `--port` fails before the model load;
the socket starts listening only when `serve` runs, so clients are refused rather than queued during the
load. With `--gguf`, `cli.main` then creates a temporary directory holding the slot directory and
`llama-server.log`, starts `LlamaProcess`, and removes the directory on exit. SIGTERM is turned into
`KeyboardInterrupt` so both signals take the same cleanup path; once cleanup starts, both are ignored, so
a second Ctrl-C cannot abort it. llama-server runs in its own session, so a terminal's Ctrl-C reaches only
llav, which stops it with one SIGTERM (two SIGINTs make llama-server terminate immediately, and it has
been seen to hang doing so). `LlamaProcess.__init__` stops the child if
anything, including an interrupt, arrives during the model load. SIGKILL skips that cleanup, so
`LlamaProcess` also starts a watchdog process (`_WATCHDOG` in `runtime.py`) that reads a pipe from llav and
stops llama-server on EOF, which the kernel delivers when llav dies, then removes the temporary directory.
`stop()` kills the watchdog before reaping llama-server so the watchdog never signals a reused pid.
`LlamaProcess` refuses a `--llama-port` that already accepts connections: another server there would
answer the readiness probe while the new llama-server fails to bind, and llav would share its slots.

With `--llama-url`, llav reads the slot count and per-slot context from `/props` and uses `--slot-dir` for
slot files; that server must already run with the flags `LlamaProcess` would pass.

`Engine.__init__` also renders the chat template once and picks its profile (`templates.detect`): the
assistant prefix `_split` appends to the template tail, unless `--assistant-prefix` overrides it, and the
low-mass cue reported in `/v1/models`. After the trim probe, `_probe_labels` asks one easy question and
raises `EngineError` when no answer letter is among the returned `n_probs`, so a template that stops before
the answer, or a model that wants to write a sentence, fails at startup instead of on every request.

`--calibration` is loaded before anything starts, and a malformed file or one for another prompt version is
a usage error. With `--gguf`, the file's model hash is checked against the model file before llama-server
starts, so a mismatch exits in seconds; with `--llama-url` only the file name from `/props` can be compared,
and llav warns that the contents are unverified.

## Error mapping

| Condition                                           | Status              | Where                         |
|-----------------------------------------------------|---------------------|-------------------------------|
| `ValidationError`, unknown model, bad JSON          | 422                 | `Handler.do_POST`             |
| `ContextTooLong`                                    | 422                 | raised in `Engine.encode`, or from the server's context error on media requests |
| `MediaRequestError`                                 | 422                 | raised in `Engine.encode_media`, or from the server's decode error |
| `Overloaded`                                        | 529                 | raised in `Engine.evaluate`   |
| `EngineError`, any other exception                  | 500                 | `Handler.do_POST`             |
| Bad auth, bad path, chunked body, bad or large size | 401/404/411/400/413 | before the body is read       |

Responses in the last row carry `Connection: close`. A body that ends early, or stalls for longer than
`SOCKET_TIMEOUT` in `server.py`, gets no response; the connection is closed to free its thread.
