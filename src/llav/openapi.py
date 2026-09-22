"""OpenAPI document for llav's HTTP API, built from the code it describes.

The limits, question types and statuses come from the modules that enforce them, so a change there shows up
in the spec. `GET /openapi.json` serves it; `scripts/write-openapi.py` writes the committed copy.
"""

from __future__ import annotations

from . import __version__
from .questions import MAX_LEVELS, MAX_OPTIONS

_PROBABILITIES = {
    "type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
    "description": "Probability per option, summed to 1 over the declared options.",
}
_CONFIDENCE = {
    "type": "number", "minimum": 0, "maximum": 1,
    "description": "1 - H(p)/ln(n): 1 when all the probability is on one option, 0 when it is uniform. "
                   "llav's own definition, not Jev's.",
}
_TEXT = {
    "description": "A string, or a JSON object or array, which llav serializes into the prompt.",
    "oneOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}],
}


def _error(description: str) -> dict:
    return {"description": description, "content": {"application/json": {
        "schema": {"$ref": "#/components/schemas/Error"}}}}


def document(model_id: str = "llav-<model>", aliases: tuple[str, ...] = ("llav-latest", "jev-latest")) -> dict:
    """The served document. `model_id` and `aliases` name what this process actually accepts."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "llav",
            "version": __version__,
            "summary": "Typed semantic decisions read from a local model's answer-token logits.",
            "description": (
                "Independent project, not affiliated with or endorsed by TypeSafe. llav reproduces the "
                "public request and response shape of the System One API, not Jev's model. Option limits, "
                "the confidence formula and the X-Llav-* headers are llav's own."
            ),
            "license": {"name": "MIT"},
        },
        "servers": [{"url": "/"}],
        "paths": {
            "/v1/systemone": {"post": {
                "summary": "Answer typed questions about one state",
                "operationId": "systemone",
                "security": [{"bearerAuth": []}],
                "requestBody": {"required": True, "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/Request"}}}},
                "responses": {
                    "200": {
                        "description": "Every question answered.",
                        "headers": {
                            "X-Llav-Seconds": {"schema": {"type": "string"}, "description": "Engine time."},
                            "X-Llav-Shared-State-Tokens": {
                                "schema": {"type": "string"},
                                "description": "Prefix tokens shared by the questions, or 0.",
                            },
                            "X-Llav-State-Cache": {
                                "schema": {"type": "string", "enum": ["hit", "miss", "off"]},
                                "description": "Whether the state came from llav's state cache.",
                            },
                        },
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Response"}}},
                    },
                    "400": _error("Content-Length is not a nonnegative integer."),
                    "401": _error("Missing or invalid API key."),
                    "411": _error("The body was sent with Transfer-Encoding; send it with Content-Length."),
                    "413": _error("The body exceeds the server's limit."),
                    "422": _error("Validation failed; detail lists the field paths."),
                    "500": _error("The llama-server backend failed, or llav hit an internal error."),
                    "529": _error("All slots were busy; retry with backoff."),
                },
            }},
            "/v1/models": {"get": {
                "summary": "The served model, its aliases and backend details",
                "operationId": "models",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {"description": "One model.", "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/ModelList"}}}},
                    "401": _error("Missing or invalid API key."),
                },
            }},
            "/openapi.json": {"get": {
                "summary": "This document",
                "operationId": "openapi",
                "security": [],
                "responses": {"200": {"description": "The OpenAPI document.",
                                      "content": {"application/json": {"schema": {"type": "object"}}}}},
            }},
            "/health": {"get": {
                "summary": "Whether the managed llama-server is running",
                "operationId": "health",
                "security": [],
                "responses": {
                    "200": {"description": "Running.", "content": {"application/json": {"schema": {
                        "type": "object", "properties": {"status": {"const": "ok"}}}}}},
                    "503": {"description": "The backend is gone.", "content": {"application/json": {"schema": {
                        "type": "object", "properties": {"status": {"const": "unavailable"}}}}}},
                },
            }},
        },
        "components": {
            "securitySchemes": {"bearerAuth": {
                "type": "http", "scheme": "bearer",
                "description": "Required only when llav runs with --api-key or LLAV_API_KEY.",
            }},
            "schemas": {
                "Request": {
                    "type": "object", "required": ["state", "model", "questions"], "additionalProperties": False,
                    "properties": {
                        "state": dict(_TEXT, description="The text the model reads. " + _TEXT["description"]),
                        "model": {"type": "string", "enum": [model_id, *aliases],
                                  "description": "The served model id or one of its aliases."},
                        "questions": {
                            "type": "object", "minProperties": 1,
                            "description": "Caller keys mapped to questions; each answer comes back under "
                                           "its key.",
                            "additionalProperties": {"oneOf": [
                                {"$ref": "#/components/schemas/NoulQuestion"},
                                {"$ref": "#/components/schemas/ChoiceQuestion"},
                                {"$ref": "#/components/schemas/ScoreQuestion"},
                            ]},
                        },
                    },
                },
                "NoulQuestion": {
                    "type": "object", "required": ["type", "instructions"], "additionalProperties": False,
                    "properties": {
                        "type": {"const": "noul"},
                        "instructions": _TEXT,
                        "criteria": {
                            "type": "object", "additionalProperties": False,
                            "description": "Optional wording for the two options; Yes and No by default.",
                            "properties": {"true": _TEXT, "false": _TEXT},
                        },
                    },
                },
                "ChoiceQuestion": {
                    "type": "object", "required": ["type", "instructions", "criteria"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {"const": "choice"},
                        "instructions": _TEXT,
                        "criteria": {
                            "type": "object", "minProperties": 2, "maxProperties": MAX_OPTIONS,
                            "description": f"Option key to description, or null. At most {MAX_OPTIONS} "
                                           "options, one per answer letter.",
                            "additionalProperties": {"oneOf": [_TEXT, {"type": "null"}]},
                        },
                    },
                },
                "ScoreQuestion": {
                    "type": "object", "required": ["type", "instructions", "criteria"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {"const": "score"},
                        "instructions": _TEXT,
                        "criteria": {
                            "type": "array", "minItems": 2, "maxItems": MAX_LEVELS, "items": _TEXT,
                            "description": "Levels in order, lowest first.",
                        },
                    },
                },
                "Response": {
                    "type": "object", "required": ["model", "answers", "usage"], "additionalProperties": False,
                    "properties": {
                        "model": {"type": "string", "description": "The model that answered."},
                        "answers": {"type": "object", "additionalProperties": {"oneOf": [
                            {"$ref": "#/components/schemas/NoulAnswer"},
                            {"$ref": "#/components/schemas/ChoiceAnswer"},
                            {"$ref": "#/components/schemas/ScoreAnswer"},
                        ]}},
                        "usage": {
                            "type": "object", "required": ["input_tokens", "output_tokens"],
                            "additionalProperties": False,
                            "properties": {
                                "input_tokens": {
                                    "type": "integer",
                                    "description": "Prompt tokens llav evaluated. A shared state counts once, "
                                                   "and not at all when it came from the state cache.",
                                },
                                "output_tokens": {"const": 0, "description": "Nothing is generated."},
                            },
                        },
                    },
                },
                "NoulAnswer": {
                    "type": "object", "required": ["type", "noul"], "additionalProperties": False,
                    "properties": {
                        "type": {"const": "noul"},
                        "noul": {"type": "number", "minimum": 0, "maximum": 1,
                                 "description": "Probability of Yes."},
                    },
                },
                "ChoiceAnswer": {
                    "type": "object", "required": ["type", "choice", "probabilities", "confidence"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {"const": "choice"},
                        "choice": {"type": "string", "description": "The option key with the most probability."},
                        "probabilities": _PROBABILITIES,
                        "confidence": _CONFIDENCE,
                    },
                },
                "ScoreAnswer": {
                    "type": "object", "required": ["type", "score", "legend", "probabilities", "confidence"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {"const": "score"},
                        "score": {
                            "type": "number", "minimum": 0,
                            "description": "Probability-weighted level index with 0-based levels; it can land "
                                           "between levels.",
                        },
                        "legend": {"type": "object", "additionalProperties": {"type": "string"},
                                   "description": "Level index to the level's text."},
                        "probabilities": _PROBABILITIES,
                        "confidence": _CONFIDENCE,
                    },
                },
                "ModelList": {
                    "type": "object", "required": ["object", "data"],
                    "properties": {
                        "object": {"const": "list"},
                        "data": {"type": "array", "items": {
                            "type": "object", "required": ["id", "object", "aliases", "owned_by", "backend"],
                            "properties": {
                                "id": {"type": "string"},
                                "object": {"const": "model"},
                                "aliases": {"type": "array", "items": {"type": "string"}},
                                "owned_by": {"type": "string"},
                                "backend": {
                                    "type": "object",
                                    "description": "llav-specific: runtime, model file, slots, per-slot "
                                                   "context and the prefix-reuse path the startup probe chose.",
                                    "additionalProperties": True,
                                },
                            },
                        }},
                    },
                },
                "Error": {
                    "type": "object", "required": ["detail"],
                    "properties": {"detail": {
                        "description": "A message, or the field paths that failed validation.",
                        "oneOf": [{"type": "string"}, {"type": "array", "items": {
                            "type": "object",
                            "properties": {"loc": {"type": "array"}, "msg": {"type": "string"}},
                        }}],
                    }},
                },
            },
        },
    }
