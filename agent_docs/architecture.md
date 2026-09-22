# Architecture

## Contents
- Overview
- Invariants
- Components
- Request flow
- Shared-state flow
- Tokenization
- Slots and concurrency
- Process lifecycle
- Error mapping

## Overview

llav is a thin HTTP layer in front of `llama-server`. All model work happens in llama-server; llav
validates requests, builds prompts, drives llama-server's HTTP API, and shapes answers. Global rules are in
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
- The shared path erases the slot before priming. Without context checkpoints, llama-server cannot roll a
  recurrent state back to a partial match, so a stale slot cache must not be reused.

## Components

| Unit            | File           | Responsibility                                                            |
|-----------------|----------------|---------------------------------------------------------------------------|
| `main`          | `cli.py`       | Parse flags; start `LlamaProcess` or attach via `--llama-url`; cleanup    |
| `Handler`       | `server.py`    | Routes, bearer auth, body limits, JSON errors, `X-Llav-*` headers         |
| `parse_request` | `questions.py` | Validate the body into `(state, model, [Question])`                       |
| `Question`      | `questions.py` | Frozen per-question data: option ids, what the model reads, score legend  |
| `build_answer`  | `questions.py` | Turn probabilities into a `noul` / `choice` / `score` answer              |
| `messages`      | `prompt.py`    | System instruction plus JSON payload `{evidence, criterion, options}`     |
| `LlamaClient`   | `engine.py`    | JSON over HTTP to llama-server; any failure becomes `EngineError`         |
| `Engine`        | `engine.py`    | Label token ids, encoding, slot pool, readout, shared-state orchestration |
| `LlamaProcess`  | `runtime.py`   | Spawn llama-server with required flags, wait for `/health`, stop on exit  |
| web UI          | `webui.html`   | `--web-ui` page at `GET /`; calls the public API like any client          |

## Request flow

1. `Handler.do_POST` checks path, auth, `Transfer-Encoding` and `Content-Length` before reading the body.
2. The body is parsed with `parse_constant` rejecting `NaN`/`Infinity`, then `parse_request` validates it.
3. The model name is checked against the served id and its aliases.
4. `Engine.evaluate` encodes every question and, for more than one question, the state prefix. All of
   this happens before a slot is taken, so tokenization does not hold capacity.
5. A slot is taken from `Engine.free` (or `Overloaded` after `queue_timeout`).
6. `_evaluate_on` scores each question and returns probabilities, `usage` and timing metadata.
7. `build_answer` shapes each answer; timing goes into `X-Llav-Seconds` and `X-Llav-Shared-State-Tokens`.

## Shared-state flow

Used when a request has more than one question and every question's tokens start with the state prefix.

1. `POST /slots/{slot}?action=erase`.
2. `/completion` with the prefix, `n_predict: 0`, `cache_prompt: true`: evaluates the state once.
3. `POST /slots/{slot}?action=save` to a unique `llav-<uuid>.bin` in the slot directory.
4. For each question: `action=restore`, then `_readout` with `cache_prompt: true`. The question's tokens
   extend the restored tokens exactly, so only the question and options are computed.
5. The slot file is deleted in a `finally`.

Single-question requests, and any request whose prefix check fails, use `_readout` with
`cache_prompt: false`.

## Tokenization

`Engine._split` renders the chat turns with `/apply-template` and splits the result into template head, user
payload and template tail. Head and tail are tokenized with special-token parsing; the payload (all caller
text) is tokenized with `special=False`. The payload sits between a newline and a control token, both token
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
`KeyboardInterrupt` so both signals take the same cleanup path. `LlamaProcess.__init__` stops the child if
anything, including an interrupt, arrives during the model load. SIGKILL skips that cleanup, so
`LlamaProcess` also starts a watchdog process (`_WATCHDOG` in `runtime.py`) that reads a pipe from llav and
stops llama-server on EOF, which the kernel delivers when llav dies, then removes the temporary directory.
`stop()` kills the watchdog before reaping llama-server so the watchdog never signals a reused pid.
`LlamaProcess` refuses a `--llama-port` that already accepts connections: another server there would
answer the readiness probe while the new llama-server fails to bind, and llav would share its slots.

With `--llama-url`, llav reads the slot count and per-slot context from `/props` and uses `--slot-dir` for
slot files; that server must already run with the flags `LlamaProcess` would pass.

## Error mapping

| Condition                                           | Status              | Where                         |
|-----------------------------------------------------|---------------------|-------------------------------|
| `ValidationError`, unknown model, bad JSON          | 422                 | `Handler.do_POST`             |
| `ContextTooLong`                                    | 422                 | raised in `Engine.encode`     |
| `Overloaded`                                        | 529                 | raised in `Engine.evaluate`   |
| `EngineError`, any other exception                  | 500                 | `Handler.do_POST`             |
| Bad auth, bad path, chunked body, bad or large size | 401/404/411/400/413 | before the body is read       |

Responses in the last row carry `Connection: close`. A body that ends early, or stalls for longer than
`SOCKET_TIMEOUT` in `server.py`, gets no response; the connection is closed to free its thread.
