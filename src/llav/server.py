"""HTTP front end: POST /v1/systemone, GET /v1/models, GET /health, GET /openapi.json."""

from __future__ import annotations

import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socketserver
import sys
import time
import traceback
from urllib.parse import urlsplit

from .engine import ContextTooLong, Engine, EngineError, Overloaded
from .openapi import document
from .questions import ValidationError, build_answer, parse_request

MAX_BODY = 8 * 1024 * 1024
# Seconds a client may stall while sending a request or idling on keep-alive before its thread is freed.
SOCKET_TIMEOUT = 60
WEB_UI = Path(__file__).with_name("webui.html")
# The page is self-contained: inline script and style, requests only back to llav.
WEB_UI_POLICY = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
                 "img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def _reject_constant(name: str):
    raise ValueError(f"{name} is not valid JSON")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # HTTPServer.server_bind names the server with getfqdn(), a reverse DNS lookup that can stall for tens
        # of seconds (35 s for 127.0.0.1 with Homebrew Python on macOS). llav never uses the name.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class App:
    def __init__(self, engine: Engine, model_id: str, aliases: list[str], api_key: str | None,
                 backend: dict, health=lambda: True, web_ui: bytes | None = None):
        self.engine = engine
        self.model_id = model_id
        self.accepted = {model_id, *aliases}
        self.aliases = aliases
        self.api_key = api_key
        self.backend = backend
        self.health = health
        self.web_ui = web_ui


class Handler(BaseHTTPRequestHandler):
    server_version = "llav"
    protocol_version = "HTTP/1.1"
    timeout = SOCKET_TIMEOUT
    app: App

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        sys.stderr.write(f"{self.address_string()} {format % args}\n")

    def _send(self, status: int, body: dict, headers: dict | None = None) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _send_page(self, page: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Content-Security-Policy", WEB_UI_POLICY)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(page)

    def _invalid(self, loc: list, message: str) -> None:
        self._send(422, {"detail": [{"loc": loc, "msg": message, "type": "value_error"}]})

    def _authorized(self) -> bool:
        if not self.app.api_key:
            return True
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        token = token.strip() if scheme.lower() == "bearer" else ""
        if token and hmac.compare_digest(token.encode(), self.app.api_key.encode()):
            return True
        # The body is left unread, so the connection cannot be reused.
        self._send(401, {"detail": "Missing or invalid API key"}, {"WWW-Authenticate": "Bearer", "Connection": "close"})
        return False

    def _route(self) -> str:
        return urlsplit(self.path).path

    def do_GET(self):  # noqa: N802 - stdlib naming
        path = self._route()
        if path == "/" and self.app.web_ui:
            self._send_page(self.app.web_ui)
        elif path == "/health":
            ok = self.app.health()
            self._send(200 if ok else 503, {"status": "ok" if ok else "unavailable"})
        elif path == "/openapi.json":
            # Served unauthenticated: it describes the API, including that a key may be required.
            self._send(200, document(self.app.model_id, tuple(self.app.aliases)))
        elif path == "/v1/models":
            if self._authorized():
                self._send(200, {"object": "list", "data": [{
                    "id": self.app.model_id, "object": "model", "aliases": self.app.aliases,
                    "owned_by": "local", "backend": self.app.backend,
                }]})
        else:
            self._send(404, {"detail": "Not found"})

    def do_POST(self):  # noqa: N802 - stdlib naming
        # Responses sent before the body is read close the connection so the unread body is not
        # parsed as the next request.
        close = {"Connection": "close"}
        if self._route() != "/v1/systemone":
            self._send(404, {"detail": "Not found"}, close)
            return
        if not self._authorized():
            return
        if "Transfer-Encoding" in self.headers:
            self._send(411, {"detail": "Send the body with Content-Length"}, close)
            return
        raw_length = (self.headers.get("Content-Length") or "0").strip()
        # Headers are decoded as Latin-1, where "\xb2" (superscript two) passes isdigit() but not int().
        if not (raw_length.isascii() and raw_length.isdigit()):
            self._send(400, {"detail": "Invalid Content-Length"}, close)
            return
        length = int(raw_length)
        if length > MAX_BODY:
            self._send(413, {"detail": f"Request body exceeds {MAX_BODY} bytes"}, close)
            return
        try:
            data = self.rfile.read(length)
        except TimeoutError:
            self.close_connection = True
            return
        if len(data) < length:
            self.close_connection = True
            return
        try:
            body = json.loads(data or b"null", parse_constant=_reject_constant)
        except (ValueError, UnicodeDecodeError) as error:
            self._invalid(["body"], f"Invalid JSON: {error}")
            return
        try:
            state, model, questions = parse_request(body)
        except ValidationError as error:
            self._invalid(error.loc, error.message)
            return
        if model not in self.app.accepted:
            self._invalid(["body", "model"], f"Unknown model {model!r}; available: {sorted(self.app.accepted)}")
            return
        try:
            results, usage, meta = self.app.engine.evaluate(state, questions)
        except ContextTooLong as error:
            self._invalid(["body", "questions", error.key], str(error))
            return
        except Overloaded as error:
            self._send(529, {"detail": str(error)}, {"Retry-After": "1"})
            return
        except EngineError as error:
            self._send(500, {"detail": str(error)})
            return
        except Exception:  # noqa: BLE001 - answer with 500 instead of dropping the connection
            traceback.print_exc()
            self._send(500, {"detail": "Internal error"})
            return
        answers = {question.key: build_answer(question, p) for question, p in zip(questions, results)}
        self._send(200, {"model": self.app.model_id, "answers": answers, "usage": usage}, {
            "X-Llav-Seconds": f"{meta['seconds']:.3f}",
            "X-Llav-Shared-State-Tokens": str(meta["shared_state_tokens"]),
            "X-Llav-State-Cache": meta["state_cache"],
        })


def bind(host: str, port: int) -> Server:
    """Claim the port before the slow model load; it accepts connections only once `serve` runs."""
    httpd = Server((host, port), Handler, bind_and_activate=False)
    try:
        httpd.server_bind()
    except BaseException:
        httpd.server_close()
        raise
    return httpd


def serve(httpd: Server, app: App) -> None:
    httpd.RequestHandlerClass = type("BoundHandler", (Handler,), {"app": app})
    httpd.server_activate()
    host, port = httpd.server_address[:2]
    sys.stderr.write(f"llav serving {app.model_id} on http://{host}:{port}\n")
    if app.web_ui:
        sys.stderr.write(f"web UI at http://{host}:{port}/\n")
    started = time.monotonic()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        sys.stderr.write(f"llav stopped after {time.monotonic() - started:.0f}s\n")
