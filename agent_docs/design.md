# API design

## Contents
- Source of the API shape
- Question mapping
- Answers
- Model names
- Usage and headers
- Errors
- Deliberate differences from Jev

## Source of the API shape

The request and response shapes follow TypeSafe's public API reference at
<https://docs.typesafe.ai/api> (the full docs dump is at <https://docs.typesafe.ai/llms-full.txt>). If
llav and the reference disagree, check the current reference before changing anything: llav tracks the
public shape, not an SDK. TypeSafe also documents `GET /v1/models`; its response shape is not published, so
llav's `/v1/models` body is its own (inferred OpenAI-style list).

Questions carry their kind in a `type` field (`{"type": "noul", "instructions": ...}`), as TypeSafe's docs
state: "A `Question` is one of three types, set by its `type` field" (checked 2026-09-24). CLM's README shows
the kind as a wrapper key instead (`{"noul": {...}}`); its server accepts `type`, like llav. CLM also takes a
`temperature` field in the request body, which llav, following TypeSafe's shape, rejects as unknown.

## Question mapping

Every question becomes one decision over 2 to 26 options, labelled `A`, `B`, … in order, except that a
single-letter choice key takes its own letter (see below). What the model
reads for each option is built in `parse_question` (`questions.py`):

| Type     | Options, in order                 | What the model reads per option                                                               |
|----------|-----------------------------------|-----------------------------------------------------------------------------------------------|
| `noul`   | `true`, `false`                   | `Yes` / `No`, or `Yes: <criteria.true>` / `No: <criteria.false>`                              |
| `choice` | `criteria` keys, letters aligned  | `key: description`, just `key` for null, or `{"option", "description"}` for structured values |
| `score`  | level indexes `0..n-1`            | the level text or structured value as given                                                   |

Why a noul's criteria are repeated in the criterion (`_fold_noul`): `Yes` and `No` say nothing on their
own, so a caller's rule often lives only in `criteria`, and models weigh the two fields differently.
Granite 4.2 3B answers the criterion and reads option descriptions as labels: a rule written only into
`criteria.true` scored 0.017 where Qwen3.5-4B scored 0.987. Appending `Answer Yes when: <criteria.true>`
(and the `No` side when given) to the criterion fixed both, 1.000 and 0.999, with the option descriptions
left alone. `choice` and `score` are not folded: there the descriptions are the options. Measurements in
[research.md](research.md).

Why choice keys are folded into the description: SemIf's fixtures had self-contained descriptions, but
System One criteria are often `{"billing": "Payments"}` or `{"billing": null}`, where the key carries the
meaning. Dropping the key would lose it. The cost: for choices, the prompt is no longer byte-identical to
SemIf's, so the SemIf agreement result does not transfer directly (see [research.md](research.md)).

Why a choice key that is a single letter moves to that letter (`_align_letters`): with keys `B`,
`insufficient`, `A` in request order, answer letter A would read "B: Candidate B", and the model answers the
name rather than the letter. On SemIf's candidate-selection family that scored 56%; averaging over option
rotations, which dissolves the collision, scored 92%. A single-letter key (either case) within the first
`n` letters takes its own letter, uppercase keys before lowercase ones, so `a` cannot take `A`'s letter; the
others fill the free letters in request order. Choices without
letter keys keep the request order exactly. `messages()` is unchanged, so this needs no `PROMPT_VERSION`
bump; like the noul fold, it changes the prompt such questions produce. `probabilities` in the answer keeps
the caller's key order. Measurements in [research.md](research.md).

Why some templates get an assistant prefix (`templates.py`): the prompt must end where the answer letter can
be the next token. Muse Glimmer's template ends at `<|start|>assistant`, and its model first writes a
recipient header, so llav appends ` to=user<|message|>` as template text. The profile is chosen from the
rendered template, never from a model name; `--assistant-prefix` overrides it. The prefix is not part of
`messages()`, so SemIf's prompt format is unchanged for every template that needs none.

`instructions` and structured criteria are placed into the JSON payload as JSON values, not stringified.

## Answers

Built by `build_answer` (`questions.py`):

- `noul`: `p(true)`.
- `choice`: argmax key, full `probabilities` map, `confidence`.
- `score`: `Σ i·p_i` over 0-based levels, `legend` (index to level text, structured levels JSON-encoded),
  `probabilities` keyed by index string, `confidence`.
- With `--calibration`, every probability above is the calibrated one, `softmax(log p / T)` with `T` from
  the file for the question's type, so `noul`, `score`, `probabilities` and `confidence` all change and the
  chosen option does not. One `T` per type, not per option or per source: vector or matrix scaling would
  overfit the few hundred labels a caller typically has, and a per-source `T` needs a source the API does
  not carry.
- `confidence()` is `1 − H(p)/ln(n)`. TypeSafe says only that confidence is derived from the
  distribution; the two examples in its docs do not match normalized entropy, Gini, or top-two margin, so
  its formula is unknown. llav's formula is its own and is documented as such.

## Model names

The served id is `--model-id` or `llav-<gguf stem>`. Requests may name the served id, `llav-latest`, or
`jev-latest` (unless `--no-jev-alias`). `jev-latest` is accepted so code written for the TypeSafe SDK works
unchanged; responses always carry llav's own id.

## Usage and headers

- `usage.input_tokens` is the number of prompt tokens llama-server actually computed (`timings.prompt_n`),
  so a shared state counts once and restored tokens count zero. It is not a billing figure.
- `usage.output_tokens` is always 0.
- `X-Llav-Calibration` names the calibration file in use (the first 12 hex digits of its SHA-256) or
  `none`, so a caller can tell calibrated answers from raw ones.
- Candidate mass goes in `X-Llav-Candidate-Mass`, not in the answers, for the same reason. It is the share of
  the full vocabulary on the declared labels, from OneForward's `candidate_mass`. It flags a model that did
  not answer with a letter; it is not calibration.
- `X-Llav-Candidate-Mass` is positional, in the order of `answers`. Two known limits, both unhandled: it
  grows by about 7 bytes per question, so a request with several hundred questions produces a header some
  proxies reject; and a JavaScript client that sends hand-written JSON with integer-like question keys gets
  `answers` back reordered by its JSON parser, which puts integer keys first, so positions no longer line
  up. Clients that build the body from their own parsed object are unaffected.
- Timing and sharing details go in `X-Llav-Seconds` and `X-Llav-Shared-State-Tokens`, not in `usage`, so
  strict clients that validate `usage` keep working.

## Errors

422 bodies use the `{"detail": [{"loc": [...], "msg": ..., "type": "value_error"}]}` shape. TypeSafe's
docs say only that the body "details the offending field"; the FastAPI-style shape is an assumption.
Other errors use `{"detail": "<message>"}`. 529 carries `Retry-After`. The status table lives in the
README; the exception mapping is in [architecture.md](architecture.md).

## Deliberate differences from Jev

- At most 26 choice options (single-token letters `A`–`Z`); TypeSafe allows 255. Going further needs
  multi-token labels or another readout. See [gotchas.md](gotchas.md).
- The confidence formula is llav's own.
- Probabilities are uncalibrated option scores from a 4B model, not a trained decision head, unless the
  operator loads a temperature file fitted on their own labelled data.
- No rate-limit tiers; capacity limits surface as 529.
- Backticked references in `instructions` are passed through verbatim; nothing resolves them against the
  state.
- A noul's `criteria` are repeated in the criterion text, so a rule written only there is read by every
  model. This changes the prompt for such questions, and with it their answers.
- A choice key that is a single letter is shown at that answer letter, so the options may reach the model
  in a different order than the request gave them.
