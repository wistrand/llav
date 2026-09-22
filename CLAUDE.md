Guidance for agents working in this repo. Read this first, then the relevant file in `agent_docs/`.

## What this is

llav is an HTTP server that answers typed decision questions (`noul`, `choice`, `score`) about a caller's
`state`, using a local GGUF model through `llama-server`. Its API follows the public shape of TypeSafe's
System One API (`POST /v1/systemone`). Each question is one forward pass. llav reads the next-token
log-probabilities of the answer letters `A`–`Z` and softmaxes them over the declared options; it never
generates text. When a request asks several questions about one state, llav evaluates the state prefix
once and reuses it for every question; evaluated states are cached as slot files for later requests.

```
client ──HTTP──> server.py ──> questions.py (validate, build answers)
                     │
                     └──> engine.py ──HTTP──> llama-server (managed by runtime.py, or external)
                            prompt.py builds the prompt
```

## Layout

| Path                       | Role                                                                   |
|----------------------------|------------------------------------------------------------------------|
| `src/llav/cli.py`          | Entry point `main()`: flags, managed vs external llama-server, cleanup |
| `src/llav/server.py`       | `Handler`: routes, auth, body limits, error-to-status mapping          |
| `src/llav/questions.py`    | `parse_request()`, `Question`, `build_answer()`, `confidence()`        |
| `src/llav/prompt.py`       | `messages()`: SemIf `direct-options-v1` prompt; `LABELS`               |
| `src/llav/engine.py`       | `LlamaClient`, `Engine.evaluate()`: tokenization, slots, readout       |
| `src/llav/runtime.py`      | `LlamaProcess`: start, health-wait, stop llama-server                  |
| `src/llav/webui.html`      | Optional browser UI (`--web-ui`); self-contained, no external requests |
| `src/llav/openapi.py`      | `document()`: the OpenAPI 3.1 spec, built from the code                |
| `tests/test_llav.py`       | Unit tests with a fake llama-server; no model needed                   |
| `scripts/write-openapi.py` | Writes the committed `openapi.json` from `openapi.py`                  |
| `scripts/benchmark.py`     | `accuracy`, `timing`, `phases` against a running llav                  |
| `scripts/fetch-model.sh`   | Downloads a pinned Q8_0 GGUF (Qwen3.5-4B default) and checks SHA-256   |
| `scripts/remote-gpu.sh`    | Sets up and starts llav on a rented CUDA box over SSH (Vast.ai etc.)   |
| `agent_docs/`              | Deep dives, linked below                                               |
| `docs/`                    | README screenshots; retake them when the web UI changes                |

## Commands

```bash
PYTHONPATH=src python3 -m llav --gguf PATH.gguf          # serve (starts llama-server)
PYTHONPATH=src python3 -m llav --help                    # flags; the source of truth for options
python3 -m unittest discover -s tests                    # unit tests
scripts/fetch-model.sh DIR [MODEL]                       # get a pinned model (default qwen3.5-4b)
scripts/benchmark.py timing http://127.0.0.1:8080        # also: accuracy, phases (see its --help)
```

Runtime requirement: `llama-server` from llama.cpp on `PATH`, or passed with `--llama-server`. The README's
Quick start lists install options per platform.

## Docs

- [agent_docs/architecture.md](agent_docs/architecture.md): modules, request flow, shared-state flow,
  tokenization split, slot pool, process lifecycle. Read before changing `engine.py` or `server.py`.
- [agent_docs/design.md](agent_docs/design.md): API compatibility decisions: how each question type maps to
  options, confidence and score formulas, aliases, usage semantics, error format, deliberate differences
  from Jev.
- [agent_docs/research.md](agent_docs/research.md): the measurements behind the design (prompt fidelity,
  agreement with SemIf's reference, timings, approaches tried and rejected) and open questions.
- [agent_docs/gotchas.md](agent_docs/gotchas.md): llama.cpp and hybrid-model traps. Skim before touching
  llama-server flags, the readout, or slot handling.

## Invariants

- Never add a third-party Python dependency. llav is standard library only; the only runtime is
  `llama-server`.
- Never change the prompt format (system text, payload keys and order, JSON serialization, label letters)
  without bumping `PROMPT_VERSION` in `prompt.py` and re-measuring agreement. Given the same descriptions,
  `messages()` must produce SemIf's `direct-options-v1` prompt byte for byte; the validation evidence in
  [agent_docs/research.md](agent_docs/research.md) depends on it.
- Never sample or generate. The readout is `/completion` with `n_predict: 1`, `temperature: -1` and
  `n_probs`, which returns a plain softmax over the raw logits. Do not use `post_sampling_probs`, grammars or
  `logit_bias`.
- Always tokenize caller text (state, instructions, criteria) with `parse_special: false`, so it cannot
  forge chat-template control tokens. Only template text is tokenized with special-token parsing.
- Always run llama-server with `--ctx-checkpoints 0` and `--slot-save-path`. Whether a slot's cache may be
  reused between questions is decided by `Engine`'s startup probe (`trims`), never by a model name; a
  backend that fails the probe restores the saved prefix before every question.
- Never let a cached slot file outlive the state that produced it: files are keyed by the prefix tokens and
  a per-run id, evicted by `--state-cache`, and deleted in `Engine.clear_cache` at shutdown.
- Always return a slot to `Engine.free` in a `finally`, or the server loses capacity permanently.
- Keep the response body to the public System One shape: `model`, `answers`, `usage` with only
  `input_tokens` and `output_tokens`. Put llav-specific data in `X-Llav-*` headers.
- Never present llav as Jev or TypeSafe. Responses name llav's own model id, and the README keeps its
  independence statement. Accepting `jev-latest` as a request alias is allowed for SDK compatibility.

## Conventions

- Python 3.10+, 4-space indent, double quotes, lines up to about 120 characters.
- Match the surrounding code's comment density; comments state why, not what.
- New llama-server interactions go through `LlamaClient`, which turns every backend failure into
  `EngineError` (HTTP 500).
- Map new failure modes to the existing statuses in `server.py`: 422 for caller mistakes, 529 for
  capacity, 500 for backend faults.
- The README's Options block is hand-written. When flags change in `cli.py`, update it in the same change.
- The API is described once, in `openapi.py`. When routes, question types, limits or statuses change, update
  it and run `PYTHONPATH=src python3 scripts/write-openapi.py`; a test fails while `openapi.json` is stale.

## Documentation Style

- Markdown links for doc references you want an agent to follow, not backticks. Backticks are fine for
  source paths in tables and inline code. Align table columns.
- No AI-isms (no "powerful", "seamlessly", "leverage", rule-of-three, "not just X but Y"). No em dashes or
  emojis in project copy. State the point directly.
- Concise; assume the agent is competent. Add only what it can't infer: project names, rules,
  constraints, and the why.
- State each rule on its own line as always/never; a rule buried mid-paragraph gets skipped.
- Don't copy constants from code into docs; name the constant and where it lives.
- Mark inferred claims and open questions; don't present a guess as a fact.
- Docs change in the same change as the code they point at. Keep this file the routing entry point; move
  subsystem detail into `agent_docs/`.
