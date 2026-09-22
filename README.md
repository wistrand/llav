# llav

```
╭───────────────────────────╮
│    ╷   ╷                  │
│    │   │   ╶─╮  ╲    ╱    │
│    │   │   ╭─┤   ╲  ╱     │
│    ╰─  ╰─  ╰─╯    ╲╱      │
│                           │
│  decisions from logprobs  │
╰───────────────────────────╯
```

Typed semantic decisions from a local open model, served over an HTTP API shaped like TypeSafe's
System One API (`POST /v1/systemone`).

*Independent project; not affiliated with or endorsed by TypeSafe. Jev and TypeSafe are the property of
their respective owners. llav reproduces the public request/response shape, not Jev's model.*

You send a `state` and a map of typed questions (`noul`, `choice`, `score`). llav asks a local model
([Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) GGUF via
[llama.cpp](https://github.com/ggml-org/llama.cpp)) each question in a single forward pass and reads the
probabilities of the answer labels straight from the logits. Nothing is generated or parsed.

- **No Python dependencies.** Standard library only. The runtime is `llama-server`.
- **Runs on any llama.cpp backend**, including Vulkan, CUDA, Metal, SYCL and CPU.
- **Shared state.** When a request asks several questions about one state, the state is evaluated once and
  reused for every question.
- **State cache.** Evaluated states are kept for later requests, so follow-up questions about the same text
  skip the expensive part: about 3 s down to 0.2 s on the laptop below.

## Quick start

1. Install llama.cpp so that `llama-server` is on your `PATH`. Any of these work:

   | Platform         | Install                                                                                    |
   |------------------|--------------------------------------------------------------------------------------------|
   | macOS, Linux     | `brew install llama.cpp` (Metal on Apple Silicon)                                          |
   | Windows          | `winget install llama.cpp`                                                                 |
   | Arch             | `pacman -S llama-cpp` plus a backend package such as `ggml-vulkan`                         |
   | Any              | A [release build](https://github.com/ggml-org/llama.cpp/releases) for your GPU             |
   | Any, from source | llama.cpp's [build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) |

   Pick a build with a GPU backend (CUDA, Vulkan, Metal, ROCm or SYCL) if you have one; CPU-only works but
   is slower. If `llama-server` is not on `PATH`, point llav at it with `--llama-server PATH`.

2. Download the model (4.6 GB, SHA-256 checked):

   ```bash
   scripts/fetch-model.sh ~/models
   ```

   Other pinned models, all Apache 2.0, take the name as a second argument
   (`scripts/fetch-model.sh ~/models qwen3.5-2b`). Times are on the laptop under Performance, 1,800-token
   state: a first request of 10 questions, and 5 questions on a state the cache already holds. Rows are in
   order of preference; the reasons are in [agent_docs/research.md](agent_docs/research.md).

   | Model                                                                           | Argument             |   Size | Cold 10 q | Repeat 5 q |
   |---------------------------------------------------------------------------------|----------------------|-------:|----------:|-----------:|
   | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (default)                  | `qwen3.5-4b`         | 4.6 GB |     5.0 s |     0.88 s |
   | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)                            | `qwen3.5-2b`         | 2.1 GB |     1.9 s |     0.42 s |
   | [IBM Granite 4.0 H Tiny](https://huggingface.co/ibm-granite/granite-4.0-h-tiny) | `granite-4.0-h-tiny` | 7.4 GB |     3.1 s |     0.89 s |
   | [IBM Granite 4.2 3B](https://huggingface.co/ibm-granite/granite-4.2-3b)         | `granite-4.2-3b`     | 3.9 GB |     4.5 s |     0.98 s |
   | [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)                   | `smollm3-3b`         | 3.3 GB |     4.0 s |     0.77 s |

   The agreement figures under How it works were measured with Qwen3.5-4B only. The others passed a
   19-question screening and scored 10 to 14 of 17 harder questions, against 17 for the default; their
   accuracy on labelled data is unmeasured. Several smaller or older models (1B and under, Gemma 3, LFM2.5)
   failed the screening; see [agent_docs/research.md](agent_docs/research.md).

3. Serve from the checkout. Nothing needs installing:

   ```bash
   PYTHONPATH=src python3 -m llav --gguf ~/models/Qwen_Qwen3.5-4B-Q8_0.gguf --port 8080
   ```

   Add `--web-ui` and open http://localhost:8080/ to try questions in a browser.

   <a href="docs/webui.png"><img src="docs/webui-thumb.png" width="480" alt="The llav web UI answering a rating and a choice question about a support ticket"></a>

   For a `llav` command on your `PATH`, for example to run it as a service, use `pipx install .`.

4. Ask:

   ```bash
   curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
     "state": "Help! My payouts have been failing for 3 days.",
     "model": "llav-latest",
     "questions": {
       "is_urgent":   {"type": "noul", "instructions": "Does this convey urgency?"},
       "department":  {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds",
                                    "technical": "Bugs, outages, integrations",
                                    "sales": "Pricing, upgrades, new accounts"}},
       "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                       "criteria": ["Calm", "Frustrated", "Very angry"]}
     }
   }'
   ```

   ```json
   {
     "model": "llav-qwen_qwen3.5-4b-q8_0",
     "answers": {
       "is_urgent": {"type": "noul", "noul": 0.997},
       "department": {"type": "choice", "choice": "billing",
                      "probabilities": {"billing": 0.849, "technical": 0.150, "sales": 0.002},
                      "confidence": 0.605},
       "frustration": {"type": "score", "score": 0.937,
                       "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                       "probabilities": {"0": 0.065, "1": 0.933, "2": 0.002},
                       "confidence": 0.769}
     },
     "usage": {"input_tokens": 249, "output_tokens": 0}
   }
   ```

   The response above is abridged: real responses carry full-precision floats. Probabilities and token
   counts vary slightly with the llama.cpp build and backend.

## API

### `POST /v1/systemone`

The request is `{"state", "model", "questions"}`:

- **`state`** (required) is a nonempty string, object or array.
- **`model`** (required) is the served model id, `llav-latest` or `jev-latest`. Use `--no-jev-alias` to
  refuse `jev-latest`.
- **`questions`** (required) maps your keys to questions. Each question has a `type`, `instructions`
  (string, object or array) and `criteria`:

| Type | `criteria` | Answer |
|---|---|---|
| `noul` | optional `{"true": ..., "false": ...}` | `{"type": "noul", "noul": p_yes}` |
| `choice` | `{option: description or null}`, 2–26 options | `{"type": "choice", "choice", "probabilities", "confidence"}` |
| `score` | ordered array of 2–10 levels | `{"type": "score", "score", "legend", "probabilities", "confidence"}` |

- **`score`** is the probability-weighted level index, `Σ i·p_i` with 0-based levels. It can land between
  levels.
- **`confidence`** is `1 − H(p)/ln(n)`: normalized entropy, 1 when all the probability is on one option and 0
  when it is uniform. TypeSafe does not publish its formula, so this is llav's own definition.
- **`usage.input_tokens`** counts the prompt tokens llav actually evaluated. A shared state is counted once,
  and not at all when it comes from the state cache. **`output_tokens`** is always 0.
- **Timing headers:** each response carries `X-Llav-Seconds`, `X-Llav-Shared-State-Tokens` and
  `X-Llav-State-Cache` (`hit`, `miss` or `off`).

Errors return a JSON `detail`:

| Status | Meaning |
|---|---|
| 400 | `Content-Length` is not a nonnegative integer. |
| 401 | Missing or invalid API key |
| 404 | Unknown path |
| 411 | The body was sent with `Transfer-Encoding` (for example chunked). Send it with `Content-Length`. |
| 413 | The body exceeds 8 MiB. |
| 422 | Validation failed. This covers a malformed body, an unknown model, and a prompt too long for a slot. The body gives the field path in the style `{"detail": [{"loc": [...], "msg": ...}]}`. |
| 500 | The llama-server backend failed or returned an unusable response, or llav hit an internal error. |
| 529 | All slots were busy for `--queue-timeout` seconds. Retry with backoff. |

A 400, 401, 404, 411 or 413 response to a `POST` is sent before the body is read, so it carries
`Connection: close` and the server closes the connection.

### `GET /v1/models`

Returns the served model id, its aliases and the backend details.

### `GET /openapi.json`

The OpenAPI 3.1 document for this server, built from the code that serves it, so the model ids it lists are
the ones this process accepts. No key is needed. A committed copy sits in
[openapi.json](openapi.json); `scripts/write-openapi.py` regenerates it.

### `GET /health`

Returns `{"status": "ok"}` while the managed llama-server is running.

### `GET /` (optional web UI)

With `--web-ui`, llav serves a page for trying questions in a browser: write a state, add yes/no, choice and
score questions, and see each answer's probabilities as bars. The page calls `/v1/systemone` like any other
client, keeps your draft in the browser's local storage, and loads nothing from outside llav. The page
itself needs no key; with `--api-key`, enter the key in the page to send requests.

### Authentication

Set `--api-key` or `LLAV_API_KEY` to require `Authorization: Bearer <key>` (the scheme is matched without
regard to case). Without a key, auth is off.
llav binds to `127.0.0.1` by default.

## How it works

Every question becomes one prompt:
- a fixed system instruction
- a JSON payload: `{"evidence": state, "criterion": instructions, "options": [{"letter": "A", "description": ...}, ...]}`

llav renders the prompt with the model's own chat template, with thinking disabled. It then reads the
next-token log-probabilities of the letters `A`, `B`, … and softmaxes them over the declared options.
Nothing is sampled.

What the model sees for each option:
- **Choice:** the option key and its description (`"billing: Payments, invoicing, refunds"`).
- **Noul:** two options, `Yes` and `No`, with their `criteria` text if given.
- **Score:** the levels, in order.

This is the readout from [SemIf](https://github.com/TheoLeeCJ/SemIf) (`direct-options-v1`). Before llav
existed, its llama.cpp path was checked against SemIf's published BF16 PyTorch predictions on SemIf's 777
owned decisions:
- **Prompts:** 777/777 prompt hashes and token sequences were identical.
- **Answers:** 768/777 decisions agreed. All 9 disagreements were near-ties between 0.47 and 0.53, and 5 of
  them were exact 0.5/0.5 ties in the reference. The reference's own prefix-reuse modes differ from its
  fresh mode on 5–6 decisions.
- **Probabilities:** the median difference was 0.005 and the largest 0.149.

That check used SemIf's own option wording. llav's choice folding (`key: description`) and its `Yes`/`No`
noul options change the prompt, so the same accuracy is expected but not measured.

**Shared state.** Evaluating the state is most of the work: about 3 s for 1,800 tokens against 0.15 s for a
question on top of it. llav evaluates it once per request and saves the slot to a file with
`/slots/{id}?action=save`.

Whether that file is needed between questions depends on the model, so llav probes the backend at startup
rather than keeping a list of model names. Attention-only models (Granite, SmolLM3) roll their cache back to
the shared prefix by themselves. Qwen3.5 mixes attention with recurrent layers, which llama.cpp cannot roll
back without context checkpoints, and those copy about 50 MB off the GPU per request; such a model gets the
saved file restored before every question, about 20 ms each. `GET /v1/models` reports which path is in use
as `backend.prefix_reuse`.

**State cache.** The saved file is kept and reused by later requests about the same state, keyed by the
state's tokens. A hit restores in about 20 ms instead of evaluating the state again, and those tokens are
not counted in `usage`. `--state-cache N` sets how many states are kept (default 4, `0` disables); each
costs a file in the slot directory. Measured on the laptop below, 1,800-token state:

| Request                                  | Cold   | Repeat state |
|------------------------------------------|-------:|-------------:|
| 5 questions                              | 3.89 s |       0.85 s |
| 1 question                               | 3.20 s |       0.18 s |

llama-server runs with `--ctx-checkpoints 0 --slot-save-path <tmp>`, which llav sets when it manages the
process.

## Why it is faster than prompting for text

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
[agent_docs/research.md](agent_docs/research.md) has the details.

## Performance

These figures come from a laptop with an Intel Arc B390 iGPU on Vulkan, using Q8_0. Each state is about
1,800 tokens with 21 yes/no questions.

| Workload                                                 |                    Time |
|----------------------------------------------------------|------------------------:|
| 21 questions in one request, state not seen before       | 6.5 s (3.2 decisions/s) |
| The same 21 questions as separate requests, state cached |                   4.4 s |
| The same, with `--state-cache 0`                         |                    67 s |
| Evaluating a state the cache does not hold               |               2.5–3.4 s |
| Each further question on that state                      |              0.2–0.25 s |
| Restoring a cached state                                 |             about 20 ms |

Throughput depends heavily on the GPU. llama.cpp does not batch several sequences of this hybrid model
efficiently, so extra `--slots` add concurrency but little throughput. Once a state is cached, splitting
questions across requests costs no more than asking them together.

### Expected scaling on other hardware

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

Estimated, 21 questions on a 1,800-token state, first request and a repeat of the same state. The two
measured rows carry their measured phase timings out to 21 questions; the Performance table above has the
runs themselves.

| Hardware                     |      State | Per question | First request | Repeat request | Decisions/s, repeat |
|------------------------------|-----------:|-------------:|--------------:|---------------:|--------------------:|
| Arc B390 iGPU (measured)     |  2.5–3.4 s |   0.2–0.25 s |         6.5 s |          3.7 s |                   6 |
| MacBook Air M3 (measured)    |     5.35 s |       203 ms |        10.0 s |          4.5 s |                   5 |
| Apple M4 Pro/Max (Metal)     |  1.3–2.1 s |    80–150 ms |     3.5–5.3 s |      1.7–3.2 s |                7–12 |
| RTX 4070 / 3090 class (CUDA) |  0.2–0.4 s |     30–50 ms |     0.8–1.4 s |      0.7–1.1 s |               19–30 |
| RTX 4090 / 5090 (CUDA)       | 0.1–0.25 s |     20–35 ms |     0.5–1.0 s |      0.5–0.8 s |               26–46 |
| H100                         |     ~0.1 s |     20–30 ms |     0.5–0.7 s |      0.5–0.7 s |               32–46 |
| 16-core desktop CPU, no GPU  |    10–20 s |      0.5–1 s |       20–40 s |        10–21 s |                 1–2 |

- **The largest uncertainty is Qwen3.5's recurrent layers.** llama.cpp's kernels for them are newer and
  less tuned than its attention kernels. The GPU rows could be off by a factor of two.
- **An H100 gains little over a 4090.** A 4B model is too small to use it; the per-question overhead sets
  the limit, and on a repeat request it is the only cost left.
- **Concurrency across states** with `--slots` may scale better on large GPUs than on the laptop. Untested.
- **Past about 20 decisions/s, software matters more than hardware.** The per-question floor comes from
  llama.cpp evaluating each question's tokens in its own pass; only a batched decode over a shared prefix
  removes it, and llama-server's HTTP API cannot express that. See
  [agent_docs/research.md](agent_docs/research.md).

## Differences from Jev

- **Model and accuracy.** A 4B open model with an untrained readout. Probabilities are uncalibrated
  conditional scores: calibrate on your own labelled data before thresholding on them.
- **Choice size.** llav allows at most 26 options (single-token letters `A`–`Z`); Jev allows 255.
- **Confidence formula.** Jev's formula is unpublished, so llav uses its own (see above).
- **Usage and limits.** Token counts follow llav's evaluation, not Jev's billing. There are no
  rate-limit tiers: overload returns `529`.
- **Instructions stay literal.** Backticked references in `instructions` are passed through verbatim to the
  model, which sees the state as JSON.

## Options

```
llav --gguf FILE [--port 8080] [--host 127.0.0.1] [--slots 1] [--ctx 8192]
     [--api-key KEY] [--model-id ID] [--no-jev-alias] [--queue-timeout 30] [--state-cache 4] [--web-ui]
     [--llama-server BIN] [--llama-port 8089] [--llama-arg ARG ...]

llav --llama-url http://127.0.0.1:8089 --slot-dir DIR   # attach to your own llama-server
```

Pass llama.cpp flags through with `--llama-arg`, for example `--llama-arg=-dev --llama-arg=Vulkan0`. When
attaching to your own server, start it with `--ctx-checkpoints 0 --slot-save-path DIR --jinja`.

## License

MIT; see [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
