"""Gallery of results (outputs/) and user uploads (inputs/), each file with a JSON sidecar."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin

from . import media
from .config import Paths

log = logging.getLogger(__name__)


@dataclass
class MediaItem:
    id: str
    kind: str  # image | video
    file: str  # path relative to the data folder
    created: float
    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: float = 0.0
    has_audio: bool = False
    task: str = ""
    model: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    favorite: bool = False
    thumb: str = ""
    name: str = ""  # original file name for uploads
    hidden: bool = False  # masks and other helper uploads

    @property
    def is_upload(self) -> bool:
        return self.id.startswith("u-")

    def public(self) -> dict:
        d = asdict(self)
        d["url"] = "/files/" + self.file
        d["thumb_url"] = ("/files/" + self.thumb) if self.thumb else d["url"]
        return d


class Storage:
    def __init__(self, paths: Paths, image_format=lambda: "png"):
        self.paths = paths
        self.image_format = image_format
        self._lock = threading.Lock()
        self.items: dict[str, MediaItem] = {}
        self._load()

    # ------------------------------------------------------------------ index
    def _load(self) -> None:
        for root in (self.paths.outputs, self.paths.inputs):
            if not root.is_dir():
                continue
            for meta in root.rglob("*.json"):
                try:
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    item = MediaItem(**data)
                except (OSError, ValueError, TypeError):
                    continue
                if (self.paths.home / item.file).exists():
                    self.items[item.id] = item

    def _write_meta(self, item: MediaItem) -> None:
        path = (self.paths.home / item.file).with_suffix(".json")
        path.write_text(json.dumps(asdict(item), ensure_ascii=False, indent=1), encoding="utf-8")

    def _new_output(self, ext: str) -> tuple[str, Path]:
        now = time.localtime()
        day = self.paths.outputs / time.strftime("%Y-%m-%d", now)
        day.mkdir(parents=True, exist_ok=True)
        oid = "o-" + time.strftime("%Y%m%d-%H%M%S", now) + "-" + secrets.token_hex(3)
        return oid, day / f"{oid}{ext}"

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.paths.home).as_posix()

    def _finish(self, item: MediaItem, path: Path, is_video: bool) -> MediaItem:
        thumb = path.with_suffix(".thumb.jpg")
        if media.make_thumbnail(path, thumb, is_video):
            item.thumb = self._rel(thumb)
        self._write_meta(item)
        with self._lock:
            self.items[item.id] = item
        return item

    # ------------------------------------------------------------------ outputs
    def video_path(self, ext: str = ".mp4") -> tuple[str, Path]:
        return self._new_output(ext)

    def add_image(self, img: Image.Image, *, task: str, model: str = "", params: dict | None = None,
                  sources: list[str] | None = None, fmt: str | None = None) -> MediaItem:
        fmt = (fmt or self.image_format() or "png").lower()
        if img.mode == "RGBA":
            fmt = "png" if fmt == "jpg" else fmt  # keep transparency
        ext = {"jpg": ".jpg", "webp": ".webp"}.get(fmt, ".png")
        oid, path = self._new_output(ext)
        params = params or {}
        if ext == ".png":
            info = PngImagePlugin.PngInfo()
            info.add_text("edit-studio", json.dumps({"task": task, "model": model, "params": params}, ensure_ascii=False))
            img.save(path, "PNG", pnginfo=info)
        elif ext == ".webp":
            img.save(path, "WEBP", quality=95, method=5)
        else:
            media.flatten(img).save(path, "JPEG", quality=95)
        item = MediaItem(id=oid, kind="image", file=self._rel(path), created=time.time(), width=img.width,
                         height=img.height, task=task, model=model, params=params, sources=sources or [])
        return self._finish(item, path, False)

    def add_video(self, oid: str, path: Path, *, task: str, model: str = "", params: dict | None = None,
                  sources: list[str] | None = None) -> MediaItem:
        info = media.probe_video(path)
        item = MediaItem(id=oid, kind="video", file=self._rel(path), created=time.time(), width=info.width,
                         height=info.height, duration=info.duration, fps=info.fps, has_audio=info.has_audio,
                         task=task, model=model, params=params or {}, sources=sources or [])
        return self._finish(item, path, True)

    # ------------------------------------------------------------------ uploads
    def add_upload(self, data_path: Path, filename: str, hidden: bool = False) -> MediaItem:
        ext = Path(filename).suffix.lower()
        if ext not in media.IMAGE_EXTS | media.VIDEO_EXTS:
            raise ValueError("unsupported_file")
        digest = hashlib.sha1()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        uid = "u-" + digest.hexdigest()[:16]
        with self._lock:
            if uid in self.items:
                data_path.unlink(missing_ok=True)
                existing = self.items[uid]
                if existing.hidden and not hidden:  # a mask later uploaded as a normal picture
                    existing.hidden = False
                    self._write_meta(existing)
                return existing
        self.paths.inputs.mkdir(parents=True, exist_ok=True)
        dest = self.paths.inputs / f"{uid}{ext}"
        shutil.move(str(data_path), dest)
        is_video = ext in media.VIDEO_EXTS and ext != ".gif"
        item = MediaItem(id=uid, kind="video" if is_video else "image", file=self._rel(dest), created=time.time(),
                         task="upload", name=Path(filename).name, hidden=hidden)
        try:
            if is_video:
                info = media.probe_video(dest)
                item.width, item.height, item.duration = info.width, info.height, info.duration
                item.fps, item.has_audio = info.fps, info.has_audio
            else:
                with Image.open(dest) as im:
                    item.width, item.height = im.size
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise ValueError("unreadable_file") from e
        return self._finish(item, dest, is_video)

    # ------------------------------------------------------------------ queries
    def get(self, item_id: str) -> MediaItem | None:
        return self.items.get(item_id)

    def path_of(self, item_id: str) -> Path:
        item = self.items.get(item_id)
        if item is None:
            raise KeyError(item_id)
        return self.paths.home / item.file

    def list(self, kind: str | None = None, favorites: bool = False, uploads: bool = False,
             offset: int = 0, limit: int = 100, q: str = "", tasks: tuple[str, ...] = ()) -> tuple[list[MediaItem], int]:
        with self._lock:
            items = list(self.items.values())
        items = [i for i in items if i.is_upload == uploads and not i.hidden]
        if kind:
            items = [i for i in items if i.kind == kind]
        if tasks:
            items = [i for i in items if i.task in tasks]
        if favorites:
            items = [i for i in items if i.favorite]
        if q:
            ql = q.lower()
            items = [i for i in items if ql in json.dumps(i.params, ensure_ascii=False).lower() or ql in i.model.lower()]
        items.sort(key=lambda i: i.created, reverse=True)
        return items[offset: offset + limit], len(items)

    def set_favorite(self, item_id: str, value: bool) -> MediaItem:
        item = self.items[item_id]
        item.favorite = bool(value)
        self._write_meta(item)
        return item

    def delete(self, item_id: str) -> None:
        with self._lock:
            item = self.items.pop(item_id, None)
        if item is None:
            raise KeyError(item_id)
        base = self.paths.home / item.file
        for p in (base, base.with_suffix(".json"), base.with_suffix(".thumb.jpg")):
            p.unlink(missing_ok=True)
