"""Image / video helpers built on the ffmpeg binary that ships with imageio-ffmpeg."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool

    @property
    def frames(self) -> int:
        return max(1, int(round(self.duration * self.fps)))


def probe_video(path: str | Path) -> VideoInfo:
    gen = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(gen)
    finally:
        gen.close()
    w, h = meta.get("size") or (0, 0)
    return VideoInfo(
        width=int(w),
        height=int(h),
        fps=float(meta.get("fps") or 24.0),
        duration=float(meta.get("duration") or 0.0),
        has_audio=bool(meta.get("audio_codec")),
    )


def iter_frames(path: str | Path, max_frames: int | None = None, fps: float | None = None) -> Iterator[np.ndarray]:
    """Yields RGB uint8 frames (H, W, 3)."""
    params = ["-vf", f"fps={fps}"] if fps else None
    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", output_params=params)
    try:
        meta = next(gen)
        w, h = meta["size"]
        for i, raw in enumerate(gen):
            if max_frames is not None and i >= max_frames:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        gen.close()


def read_frames(path: str | Path, max_frames: int | None = None, fps: float | None = None) -> list[Image.Image]:
    return [Image.fromarray(f) for f in iter_frames(path, max_frames, fps)]


class VideoWriter:
    """Streams frames into an H.264 MP4 (or VP9 WebM with alpha for RGBA frames)."""

    def __init__(self, path: str | Path, fps: float, crf: int = 18, alpha: bool = False):
        self.path = Path(path)
        self.fps = fps
        self.crf = crf
        self.alpha = alpha
        self._gen = None
        self.size: tuple[int, int] | None = None
        self.count = 0

    def _open(self, w: int, h: int) -> None:
        self.size = (w - w % 2, h - h % 2)
        if self.alpha:
            kw = dict(codec="libvpx-vp9", pix_fmt_in="rgba", pix_fmt_out="yuva420p",
                      output_params=["-crf", str(max(self.crf, 20)), "-b:v", "0", "-auto-alt-ref", "0"])
        else:
            kw = dict(codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                      output_params=["-crf", str(self.crf), "-preset", "medium", "-movflags", "+faststart"])
        self._gen = imageio_ffmpeg.write_frames(
            str(self.path), self.size, fps=self.fps, quality=None, macro_block_size=1, ffmpeg_log_level="error", **kw
        )
        self._gen.send(None)

    def write(self, frame) -> None:
        if isinstance(frame, Image.Image):
            frame = np.asarray(frame.convert("RGBA" if self.alpha else "RGB"))
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._gen is None:
            self._open(frame.shape[1], frame.shape[0])
        w, h = self.size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = np.ascontiguousarray(frame[:h, :w])
        self._gen.send(frame.tobytes())
        self.count += 1

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def to_uint8_frames(frames) -> list[np.ndarray]:
    """diffusers video output (list of PIL / float array in [0,1]) -> list of uint8 arrays."""
    if isinstance(frames, np.ndarray):
        arr = frames
        if arr.ndim == 5:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
        return list(arr)
    out = []
    for f in frames:
        if isinstance(f, Image.Image):
            out.append(np.asarray(f.convert("RGB")))
        else:
            a = np.asarray(f)
            out.append(a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255).round().astype(np.uint8))
    return out


def write_video(frames, path: str | Path, fps: float, crf: int = 18) -> Path:
    with VideoWriter(path, fps, crf) as w:
        for f in to_uint8_frames(frames):
            w.write(f)
    return Path(path)


_TIME_RE = re.compile(r"out_time_(?:us|ms)=(\d+)")


def run_ffmpeg(
    args: list[str],
    duration: float | None = None,
    progress: Callable[[float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-nostats", "-loglevel", "error", *args]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        creationflags=_NO_WINDOW,
    )
    err_lines: list[str] = []
    t = threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True)
    t.start()
    try:
        for line in proc.stdout:
            if should_stop and should_stop():
                proc.kill()
                break
            m = _TIME_RE.search(line)
            if m and duration and progress:
                progress(min(1.0, int(m.group(1)) / 1e6 / duration))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
        t.join(timeout=2)
    if should_stop and should_stop():
        from .jobs import Cancelled

        raise Cancelled()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + ("".join(err_lines[-8:]).strip() or f"exit code {proc.returncode}"))


def mux_audio(video: Path, audio_source: Path, out: Path, start: float = 0.0, speed: float = 1.0) -> bool:
    """Copies the audio track of `audio_source` onto `video`. Returns False if there was no audio."""
    try:
        if not probe_video(audio_source).has_audio:
            return False
    except Exception:
        return False
    af = []
    if speed != 1.0:
        af = ["-filter:a", _atempo_chain(speed)]
    run_ffmpeg(
        ["-i", str(video), "-ss", f"{start:.3f}", "-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?",
         "-c:v", "copy", *af, "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    )
    return True


def _atempo_chain(speed: float) -> str:
    parts = []
    s = speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return img


def flatten(img: Image.Image, color=(255, 255, 255)) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, color)
    bg.paste(img, mask=img.split()[-1])
    return bg


def snap_size(w: int, h: int, multiple: int, max_pixels: int | None = None) -> tuple[int, int]:
    """Rounds a size to the model's granularity, optionally scaling down to a pixel budget."""
    if max_pixels and w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        w, h = w * scale, h * scale
    return max(multiple, int(round(w / multiple)) * multiple), max(multiple, int(round(h / multiple)) * multiple)


def make_thumbnail(src: Path, dst: Path, is_video: bool, size: int = 480) -> bool:
    try:
        if is_video:
            run_ffmpeg(["-ss", "0.3", "-i", str(src), "-frames:v", "1", "-vf", f"scale={size}:-2", "-q:v", "4", str(dst)])
            if not dst.exists():  # very short clips
                run_ffmpeg(["-i", str(src), "-frames:v", "1", "-vf", f"scale={size}:-2", "-q:v", "4", str(dst)])
        else:
            img = load_image(src)
            img.thumbnail((size, size))
            flatten(img, (32, 32, 36)).save(dst, "JPEG", quality=85)
        return dst.exists()
    except Exception as e:
        log.info("thumbnail failed for %s: %s", src, e)
        return False
