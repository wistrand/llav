"""Request validation and answer construction for the System One style API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

from .prompt import LABELS

MAX_OPTIONS = len(LABELS)
MAX_LEVELS = 10


class ValidationError(Exception):
    """A request field failed validation; reported as HTTP 422."""

    def __init__(self, loc: list, message: str):
        super().__init__(message)
        self.loc = loc
        self.message = message


@dataclass(frozen=True)
class Question:
    key: str
    type: str  # "noul" | "choice" | "score"
    instructions: object
    option_ids: tuple  # answer keys: choice keys, ("true", "false"), or level indexes, in label order
    descriptions: tuple  # what the model reads for each option, in label order
    legend: tuple = ()  # score levels as strings
    declared: tuple = ()  # choice keys in request order, when that differs from label order


def _is_nonempty_json(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, (dict, list)) and bool(value)


def _fold_noul(instructions, criteria: dict):
    """Repeat a noul's criteria in the criterion, where every model reads them.

    `Yes` and `No` carry no meaning by themselves, so a caller's rule often lives only in `criteria`. Models
    weigh the two fields differently: Granite 4.2 3B answers the criterion and reads option descriptions as
    labels, scoring 0.017 where Qwen3.5-4B scored 0.987 on the same question. Folding fixed it for both
    (1.000 and 0.999); see agent_docs/research.md. The options keep their descriptions.
    """
    rules = [(side, criteria.get(side)) for side in ("true", "false") if _is_nonempty_json(criteria.get(side))]
    if not rules:
        return instructions
    answers = {"true": "Yes", "false": "No"}
    if isinstance(instructions, str) and all(isinstance(value, str) for _, value in rules):
        lines = [instructions.strip()]
        lines += [f"Answer {answers[side]} when: {value.strip()}" for side, value in rules]
        return "\n\n".join(lines)
    # Non-string instructions or criteria keep their structure: the prompt serializes any JSON value.
    folded = {"criterion": instructions}
    folded.update({f"answer_{answers[side].lower()}_when": value for side, value in rules})
    return folded


def _describe(label: str, value):
    """Fold an option key into its description so the model sees both."""
    if value is None:
        return label
    if isinstance(value, str):
        return f"{label}: {value}" if value.strip() else label
    return {"option": label, "description": value}


def _align_letters(keys: list) -> list:
    """Order choice keys so a key that is itself a letter sits at that answer letter.

    With keys `B`, `insufficient`, `A` in that order, answer letter A would read "B: Candidate B", and the model
    answers the name instead of the letter: SemIf's candidate-selection family scored 56% that way against 92%
    once the collision was averaged out (agent_docs/research.md). A single-letter key (either case) within the
    first len(keys) letters takes its own letter; the other keys fill the remaining letters in request order.
    """
    slots = [None] * len(keys)
    rest = []
    for key in keys:
        index = LABELS.find(key.upper()) if len(key) == 1 else -1
        if 0 <= index < len(keys) and slots[index] is None:
            slots[index] = key
        else:
            rest.append(key)
    remaining = iter(rest)
    return [key if key is not None else next(remaining) for key in slots]


def _check_value(value, loc: list, allow_null: bool = False) -> None:
    if value is None and allow_null:
        return
    if not isinstance(value, (str, dict, list)):
        raise ValidationError(loc, "must be a string, object, or array")


def parse_question(key: str, raw) -> Question:
    loc = ["body", "questions", key]
    if not isinstance(raw, dict):
        raise ValidationError(loc, "question must be an object")
    kind = raw.get("type")
    if kind not in ("noul", "choice", "score"):
        raise ValidationError([*loc, "type"], "type must be 'noul', 'choice', or 'score'")
    unknown = set(raw) - {"type", "instructions", "criteria"}
    if unknown:
        raise ValidationError([*loc, sorted(unknown)[0]], "unknown field")
    instructions = raw.get("instructions")
    if not _is_nonempty_json(instructions):
        raise ValidationError([*loc, "instructions"], "instructions must be a nonempty string, object, or array")
    criteria = raw.get("criteria")

    if kind == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ValidationError([*loc, "criteria"], "noul criteria may only contain 'true' and 'false'")
        for side in ("true", "false"):
            _check_value(criteria.get(side), [*loc, "criteria", side], allow_null=True)
        return Question(
            key, kind, _fold_noul(instructions, criteria), ("true", "false"),
            (_describe("Yes", criteria.get("true")), _describe("No", criteria.get("false"))),
        )

    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValidationError([*loc, "criteria"], "choice criteria must map at least two options to descriptions")
        if len(criteria) > MAX_OPTIONS:
            raise ValidationError([*loc, "criteria"], f"llav supports at most {MAX_OPTIONS} options per choice")
        for option, value in criteria.items():
            if not option:
                raise ValidationError([*loc, "criteria"], "option keys must be nonempty")
            _check_value(value, [*loc, "criteria", option], allow_null=True)
        order = _align_letters(list(criteria))
        return Question(
            key, kind, instructions, tuple(order), tuple(_describe(option, criteria[option]) for option in order),
            declared=tuple(criteria) if order != list(criteria) else (),
        )

    if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_LEVELS:
        raise ValidationError([*loc, "criteria"], f"score criteria must be an array of 2-{MAX_LEVELS} levels")
    for index, level in enumerate(criteria):
        _check_value(level, [*loc, "criteria", index])
        if not _is_nonempty_json(level):
            raise ValidationError([*loc, "criteria", index], "levels must be nonempty")
    return Question(
        key, kind, instructions, tuple(str(index) for index in range(len(criteria))), tuple(criteria),
        tuple(level if isinstance(level, str) else json.dumps(level, ensure_ascii=False) for level in criteria),
    )


def parse_request(body) -> tuple[object, str, list[Question]]:
    """Validate a /v1/systemone body and return (state, model, questions)."""
    if not isinstance(body, dict):
        raise ValidationError(["body"], "request body must be a JSON object")
    unknown = set(body) - {"state", "model", "questions"}
    if unknown:
        raise ValidationError(["body", sorted(unknown)[0]], "unknown field")
    state = body.get("state")
    if not _is_nonempty_json(state):
        raise ValidationError(["body", "state"], "state must be a nonempty string, object, or array")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValidationError(["body", "model"], "model is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValidationError(["body", "questions"], "questions must be a nonempty object")
    return state, model, [parse_question(key, raw) for key, raw in questions.items()]


def confidence(probabilities: list[float]) -> float:
    """One minus normalized Shannon entropy: 1 for a one-hot distribution, 0 for a uniform one.

    TypeSafe does not publish its confidence formula; this is llav's own definition.
    """
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, min(1.0, 1 - entropy / math.log(len(probabilities))))


def build_answer(question: Question, probabilities: list[float]) -> dict:
    if question.type == "noul":
        return {"type": "noul", "noul": probabilities[0]}
    by_id = dict(zip(question.option_ids, probabilities))
    if question.type == "choice":
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        if question.declared:  # the caller's order, not the order the model saw
            by_id = {option: by_id[option] for option in question.declared}
        return {
            "type": "choice", "choice": question.option_ids[best],
            "probabilities": by_id, "confidence": confidence(probabilities),
        }
    return {
        "type": "score",
        "score": sum(index * p for index, p in enumerate(probabilities)),
        "legend": dict(zip(question.option_ids, question.legend)),
        "probabilities": by_id,
        "confidence": confidence(probabilities),
    }
