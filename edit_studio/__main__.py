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
_log_file: Path | None = None


def _setup_logging(home: Path, verbose: bool, console: bool) -> None:
    global _log_file
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    _log_file = logs / "edit-studio.log"
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(_log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    ]
    if console:
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


def _running_instance(home: Path, start: int) -> str | None:
    """URL of an Edit Studio that already serves this data folder (opening the app twice shares one server)."""
    import json
    import urllib.request

    for port in range(start, start + 10):
        url = f"http://127.0.0.1:{port}/"
        try:
            req = urllib.request.Request(url + "api/system", headers={"Host": f"127.0.0.1:{port}"})
            with urllib.request.urlopen(req, timeout=0.5) as r:
                info = json.loads(r.read())
        except Exception:
            continue
        if Path(info.get("data_dir", "")).resolve() == home.resolve():
            return url
    return None


def _open_window(url: str, paths) -> bool:
    """Native window (WebView2 on Windows). Returns False when unavailable, so the caller can use the browser."""
    try:
        import webview  # pywebview
    except Exception as e:
        logging.getLogger("edit_studio").info("Desktop window unavailable (%s); using the browser", e)
        return False
    if hasattr(webview, "settings"):
        webview.settings["ALLOW_DOWNLOADS"] = True
    webview.create_window("Edit Studio", url, width=1480, height=940, min_size=(1024, 700))
    kwargs = {"private_mode": False, "storage_path": str(paths.cache / "webview")}  # keep language / form choices
    if sys.platform == "win32":
        kwargs["gui"] = "edgechromium"  # never the old Internet Explorer engine
    try:
        webview.start(**kwargs)
    except Exception as e:
        logging.getLogger("edit_studio").warning("Desktop window failed (%s); using the browser", e)
        return False
    return True


def _free_port(host: str, start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if sys.platform != "win32":  # like uvicorn: a just-closed port (TIME_WAIT) is reusable
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host if host != "0.0.0.0" else "", port))
                return port
            except OSError:
                continue
    raise SystemExit("No free port found")


def _error_box(text: str) -> None:
    """Started without a console (desktop shortcut), a failure would be invisible: show it in a message box."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "Edit Studio", 0x10)  # MB_ICONERROR
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    console = sys.stdout is not None and sys.stderr is not None
    if not console:  # pythonw.exe (the desktop shortcut) has no console; printing must not crash anything
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    try:
        _main(argv, console)
    except Exception as e:
        logging.getLogger("edit_studio").exception("Edit Studio stopped with an error")
        if not console:
            _error_box(
                "Το Edit Studio δεν μπόρεσε να ξεκινήσει.\nEdit Studio could not start.\n\n"
                f"{type(e).__name__}: {e}\n\n"
                f"{_log_file or ''}\n(Τρέξε το EditStudio.bat για λεπτομέρειες / run EditStudio.bat for details)"
            )
        raise


def _main(argv: list[str] | None, console: bool) -> None:
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
    _setup_logging(paths.home, args.verbose, console)
    if not console:  # tracebacks from worker threads, library warnings
        sys.stdout = sys.stderr = open(paths.home / "logs" / "console.log", "a", encoding="utf-8", buffering=1)
    log = logging.getLogger("edit_studio")

    existing = None if args.listen else _running_instance(paths.home, args.port)
    if existing:
        log.info("Edit Studio is already running at %s", existing)
        if args.no_open:
            return
        if args.browser or not _open_window(existing, paths):
            webbrowser.open(existing)
        return

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
    # log_config=None: uvicorn's own console formatter crashes without a console (pythonw); ours logs to the file
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False, log_config=None))

    if console:
        print(f"\n  Edit Studio: {url}\n  (κλείσε αυτό το παράθυρο για τερματισμό / close this window to quit)\n", flush=True)
    try:
        if args.browser or args.no_open:
            if not args.no_open:
                threading.Timer(1.0, lambda: webbrowser.open(url)).start()
            server.run()
            return
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        while not server.started and t.is_alive():
            time.sleep(0.05)
        if _open_window(url, paths):
            server.should_exit = True  # window closed: quit
            t.join(timeout=5)
        else:
            webbrowser.open(url)
            t.join()
    finally:
        services.shutdown()
        log.info("Edit Studio stopped")


if __name__ == "__main__":
    main()
