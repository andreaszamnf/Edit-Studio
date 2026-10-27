"""Application paths and persistent user settings."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
WEB_DIR = PACKAGE_DIR / "web"


def default_home() -> Path:
    env = os.environ.get("EDIT_STUDIO_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return PROJECT_DIR / "data"


class Paths:
    """All on-disk locations used by the app (everything lives under one folder)."""

    def __init__(self, home: Path | None = None):
        self.home = Path(home or default_home())
        self.outputs = self.home / "outputs"
        self.inputs = self.home / "inputs"
        self.tools = self.home / "tool-models"  # upscalers, background removal, eraser
        self.cache = self.home / "cache"
        self.settings_file = self.home / "settings.json"
        self.local_index = self.cache / "local-models.json"

    def ensure(self) -> None:
        for p in (self.home, self.outputs, self.inputs, self.tools, self.cache):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    language: str = "el"
    # Hugging Face
    hf_token: str = ""
    hf_cache_dir: str = ""  # empty = the standard HF cache, shared with ComfyUI/diffusers scripts
    # Extra folders with model files (ComfyUI / Forge / A1111 / Fooocus / your own)
    model_folders: list[str] = field(default_factory=list)
    auto_detect_folders: bool = True
    # Hardware
    enabled_gpus: list[int] | None = None  # None = use every GPU
    memory_mode: str = "auto"  # auto | single | split | offload | lowvram
    compression: str = "auto"  # auto | none | fp8 | nf4
    keep_models_loaded: bool = True
    # Output
    image_format: str = "png"  # png | webp | jpg
    video_quality: int = 18  # x264 CRF (lower = better)

    CHOICES = {
        "language": ("el", "en"),
        "memory_mode": ("auto", "single", "split", "offload", "lowvram"),
        "compression": ("auto", "none", "fp8", "nf4"),
        "image_format": ("png", "webp", "jpg"),
    }

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["hf_token_set"] = bool(self.hf_token)
        data.pop("hf_token")
        return data


class SettingsStore:
    def __init__(self, paths: Paths):
        self.paths = paths
        self._lock = threading.Lock()
        self.settings = self._load()

    def _load(self) -> Settings:
        s = Settings()
        try:
            raw = json.loads(self.paths.settings_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return s
        except (OSError, ValueError) as e:
            log.warning("Could not read settings (%s); using defaults", e)
            return s
        try:
            self._apply(s, raw)
        except ValueError as e:
            log.warning("Ignoring invalid settings: %s", e)
        return s

    @staticmethod
    def _apply(s: Settings, changes: dict[str, Any]) -> None:
        names = {f.name: f for f in fields(Settings)}
        for key, value in changes.items():
            if key not in names:
                continue
            if key in Settings.CHOICES and value not in Settings.CHOICES[key]:
                raise ValueError(f"{key}: invalid value {value!r}")
            if key == "model_folders":
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ValueError("model_folders must be a list of paths")
                value = [v.strip() for v in value if v.strip()]
            elif key == "enabled_gpus":
                if value is not None and (
                    not isinstance(value, list) or not all(isinstance(v, int) for v in value)
                ):
                    raise ValueError("enabled_gpus must be null or a list of integers")
            elif key == "video_quality":
                value = int(value)
                if not 0 <= value <= 51:
                    raise ValueError("video_quality must be between 0 and 51")
            elif key in ("auto_detect_folders", "keep_models_loaded"):
                value = bool(value)
            elif key in ("hf_token", "hf_cache_dir", "language"):
                value = str(value).strip()
            setattr(s, key, value)

    def update(self, changes: dict[str, Any]) -> Settings:
        with self._lock:
            draft = Settings(**asdict(self.settings))
            self._apply(draft, changes)
            self.settings = draft
            self.paths.home.mkdir(parents=True, exist_ok=True)
            tmp = self.paths.settings_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(draft), indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.paths.settings_file)
            return draft

    def hf_token(self) -> str | None:
        return self.settings.hf_token or None

    def hf_cache_dir(self) -> str | None:
        return self.settings.hf_cache_dir or None
