"""Starts Edit Studio: the local server plus a desktop window (or the web browser)."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8765


def _setup_logging(home: Path, verbose: bool) -> None:
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(logs / "edit-studio.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    ]
    if sys.stderr is not None:  # pythonw has no console
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "multipart", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for chatty in ("diffusers", "transformers", "accelerate"):  # dtype/offload notices are not actionable
        logging.getLogger(chatty).setLevel(logging.ERROR)


def _free_port(host: str, start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host if host != "0.0.0.0" else "", port))
                return port
            except OSError:
                continue
    raise SystemExit("No free port found")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="edit-studio", description="Edit Studio - AI images & video")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--listen", action="store_true", help="allow other devices on your network (no protection!)")
    parser.add_argument("--browser", action="store_true", help="open in the web browser instead of an app window")
    parser.add_argument("--no-open", action="store_true", help="only start the server")
    parser.add_argument("--data", type=Path, default=None, help="data folder (outputs, settings, logs)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from .config import Paths

    paths = Paths(args.data)
    paths.ensure()
    _setup_logging(paths.home, args.verbose)
    log = logging.getLogger("edit_studio")

    import uvicorn

    from .app import create_app
    from .services import Services

    log.info("Starting Edit Studio (data: %s)", paths.home)
    services = Services.create(paths)
    hw = services.hw.summary()
    log.info("Hardware: %s | GPUs: %s | RAM %.0f GB", hw["backend"], [g["name"] for g in hw["gpus"]], hw["ram_total_gb"])
    if not services.local.models:
        services.jobs.submit("models.scan", {})  # first run: look for models you already have

    host = "0.0.0.0" if args.listen else "127.0.0.1"
    port = _free_port(host, args.port)
    url = f"http://127.0.0.1:{port}/"
    app = create_app(services, allow_remote=args.listen)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False))

    window = None
    if not args.browser and not args.no_open:
        try:
            import webview  # pywebview

            window = webview
        except Exception as e:
            log.info("Desktop window unavailable (%s); using the browser", e)

    print(f"\n  Edit Studio: {url}\n  (κλείσε αυτό το παράθυρο για τερματισμό / close this window to quit)\n", flush=True)
    try:
        if window is not None:
            t = threading.Thread(target=server.run, daemon=True)
            t.start()
            while not server.started and t.is_alive():
                time.sleep(0.05)
            window.create_window("Edit Studio", url, width=1480, height=940, min_size=(1024, 700))
            window.start()
            server.should_exit = True
            t.join(timeout=5)
        else:
            if not args.no_open:
                threading.Timer(1.0, lambda: webbrowser.open(url)).start()
            server.run()
    finally:
        services.shutdown()
        log.info("Edit Studio stopped")


if __name__ == "__main__":
    main()
