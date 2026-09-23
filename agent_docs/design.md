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

## Question mapping

Every question becomes one decision over 2 to 26 options, labelled `A`, `B`, … in order. What the model
reads for each option is built in `parse_question` (`questions.py`):

| Type     | Options, in order                 | What the model reads per option                                                               |
|----------|-----------------------------------|-----------------------------------------------------------------------------------------------|
| `noul`   | `true`, `false`                   | `Yes` / `No`, or `Yes: <criteria.true>` / `No: <criteria.false>`                              |
| `choice` | `criteria` keys, in request order | `key: description`, just `key` for null, or `{"option", "description"}` for structured values |
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

`instructions` and structured criteria are placed into the JSON payload as JSON values, not stringified.

## Answers

Built by `build_answer` (`questions.py`):

- `noul`: `p(true)`.
- `choice`: argmax key, full `probabilities` map, `confidence`.
- `score`: `Σ i·p_i` over 0-based levels, `legend` (index to level text, structured levels JSON-encoded),
  `probabilities` keyed by index string, `confidence`.
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
- Probabilities are uncalibrated option scores from a 4B model, not a trained decision head.
- No rate-limit tiers; capacity limits surface as 529.
- Backticked references in `instructions` are passed through verbatim; nothing resolves them against the
  state.
- A noul's `criteria` are repeated in the criterion text, so a rule written only there is read by every
  model. This changes the prompt for such questions, and with it their answers.
