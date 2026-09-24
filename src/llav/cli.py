"""llav command line: start llama-server (or attach to one) and serve the decision API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile

from . import __version__
from .calibration import Calibration, CalibrationError
from .engine import Engine, EngineError, LlamaClient
from .native import NativeError, NativeReadout
from .runtime import LlamaProcess
from .server import WEB_UI, App, bind, serve

LOGO = """\
╭───────────────────────────╮
│    ╷   ╷                  │
│    │   │   ╶─╮  ╲    ╱    │
│    │   │   ╭─┤   ╲  ╱     │
│    ╰─  ╰─  ╰─╯    ╲╱      │
│                           │
│  decisions from logprobs  │
╰───────────────────────────╯
"""


def _banner() -> None:
    # A console without UTF-8 (a legacy Windows code page) cannot encode the box characters; skip the logo there.
    try:
        sys.stderr.write(LOGO)
        sys.stderr.flush()
    except UnicodeEncodeError:
        pass


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llav", description=__doc__)
    parser.add_argument("--version", action="version", version=f"llav {__version__}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default=os.environ.get("LLAV_API_KEY"),
                        help="Require 'Authorization: Bearer <key>' (default: $LLAV_API_KEY; unset disables auth)")
    parser.add_argument("--model-id", help="Model id returned in responses (default: llav-<gguf name>)")
    parser.add_argument("--no-jev-alias", action="store_true",
                        help="Do not accept model 'jev-latest' as an alias (accepted by default for SDK compatibility)")
    parser.add_argument("--web-ui", action="store_true", help="Serve a browser UI for trying questions at /")
    parser.add_argument("--queue-timeout", type=float, default=30, help="Seconds to wait for a free slot before 529")
    parser.add_argument("--native-readout", metavar="BIN",
                        help="Path to the llav-readout helper: answers every question of a request in one "
                             "batched pass (needs --gguf; see native/README.md)")
    parser.add_argument("--native-questions", type=int, default=16,
                        help="Questions the helper takes in one pass; larger requests use llama-server")
    parser.add_argument("--state-cache", type=int, default=4,
                        help="Evaluated states kept as slot files for reuse by later requests (0 disables)")
    parser.add_argument("--calibration", type=Path, metavar="FILE",
                        help="Temperature calibration from 'scripts/evaluate.py fit'; must match the model")

    spawn = parser.add_argument_group("managed llama-server")
    spawn.add_argument("--gguf", type=Path, help="Model file; llav starts llama-server with the required flags")
    spawn.add_argument("--llama-server", default="llama-server", help="llama-server binary")
    spawn.add_argument("--llama-port", type=int, default=8089)
    spawn.add_argument("--slots", type=int, default=1, help="Concurrent requests served by llama-server")
    spawn.add_argument("--ctx", type=int, default=8192, help="Context tokens per slot")
    spawn.add_argument("--llama-arg", action="append", default=[],
                       help="Extra llama-server argument, repeatable, e.g. --llama-arg=-dev --llama-arg=Vulkan0")

    attach = parser.add_argument_group("external llama-server")
    attach.add_argument("--llama-url", help="Use a running llama-server started with --ctx-checkpoints 0 "
                                            "and --slot-save-path")
    attach.add_argument("--slot-dir", type=Path, help="That server's --slot-save-path directory")
    args = parser.parse_args(argv)

    if bool(args.gguf) == bool(args.llama_url):
        parser.error("Give exactly one of --gguf or --llama-url")
    if args.llama_url and not args.slot_dir:
        parser.error("--llama-url needs --slot-dir (the server's --slot-save-path)")
    for flag, value in (("--port", args.port), ("--llama-port", args.llama_port)):
        if not 1 <= value <= 65535:
            parser.error(f"{flag} must be between 1 and 65535")
    if args.slots < 1:
        parser.error("--slots must be at least 1")
    if args.ctx < 1:
        parser.error("--ctx must be at least 1")
    if not args.queue_timeout >= 0:
        parser.error("--queue-timeout must not be negative")
    if args.state_cache < 0:
        parser.error("--state-cache must not be negative")
    if args.native_readout and not args.gguf:
        parser.error("--native-readout needs --gguf: the helper loads the model itself")
    if args.native_questions < 1:
        parser.error("--native-questions must be at least 1")

    calibration = None
    if args.calibration:
        try:
            calibration = Calibration.load(args.calibration)
        except (CalibrationError, OSError) as error:
            parser.error(str(error))

    signal.signal(signal.SIGTERM, _raise_interrupt)
    process = None
    temporary = None
    httpd = None
    engine = None
    native = None
    try:
        httpd = bind(args.host, args.port)
        _banner()
        if args.gguf:
            if not args.gguf.is_file():
                parser.error(f"No such model file: {args.gguf}")
            if calibration:  # before the slow start, so a mismatched file fails fast
                sys.stderr.write(f"checking calibration {calibration.id} against {args.gguf.name}\n")
                calibration.check_model(args.gguf, args.gguf.name)
            temporary = Path(tempfile.mkdtemp(prefix="llav-"))
            slot_dir = temporary / "slots"
            slot_dir.mkdir()
            sys.stderr.write(f"starting llama-server (log: {temporary / 'llama-server.log'})\n")
            process = LlamaProcess(args.llama_server, args.gguf, args.llama_port, args.slots, args.ctx,
                                   slot_dir, temporary / "llama-server.log", args.llama_arg, scratch=temporary)
            client = LlamaClient(process.url)
            slots, slot_ctx, source = args.slots, args.ctx, args.gguf
        else:
            client = LlamaClient(args.llama_url)
            props = client.get("/props")
            slots = int(props.get("total_slots") or 1)
            slot_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
            if slot_ctx <= 0:
                parser.error("Could not read the per-slot context size from /props")
            slot_dir, source = args.slot_dir, Path(props.get("model_path") or "model")
            warning = calibration.check_model(None, source.name) if calibration else None
            if warning:
                sys.stderr.write(f"llav: {warning}\n")
        if args.native_readout:
            sys.stderr.write(f"starting the native readout helper ({args.native_readout})\n")
            native = NativeReadout(args.native_readout, args.gguf, [], slot_ctx, args.native_questions)
        engine = Engine(client, slots, slot_ctx, slot_dir, args.queue_timeout, state_cache=args.state_cache,
                        native=native)
        if native:
            native.labels = engine.label_ids  # the helper reads whatever labels the engine tokenized
        model_id = args.model_id or f"llav-{source.stem.lower()}"
        aliases = ["llav-latest"] + ([] if args.no_jev_alias else ["jev-latest"])
        health = (lambda: process.alive()) if process else (lambda: True)
        backend = {"runtime": "llama.cpp", "model_file": source.name, "slots": slots, "slot_ctx": slot_ctx,
                   "prefix_reuse": "native" if engine.native else ("trim" if engine.trims else "slot-file"),
                   "calibration": calibration.id if calibration else None}
        web_ui = WEB_UI.read_bytes() if args.web_ui else None
        serve(httpd, App(engine, model_id, aliases, args.api_key, backend, health, web_ui, calibration))
    except (CalibrationError, EngineError, NativeError, RuntimeError, OSError) as error:
        sys.stderr.write(f"llav: {error}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        if native:
            native.close()
        if engine:
            engine.clear_cache()
        if httpd:
            httpd.server_close()
        if process:
            process.stop()
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
