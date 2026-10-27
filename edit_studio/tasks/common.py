"""Shared helpers for task handlers."""

from __future__ import annotations

import hashlib
import inspect
import logging
import random
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter
from pydantic import BaseModel, ConfigDict, Field

from .. import media
from ..jobs import Cancelled, TaskContext, TaskError
from ..models.catalog import TOOLS_BY_ID, ToolModel
from ..models.manager import ModelError, ModelSource

log = logging.getLogger(__name__)


class Params(BaseModel):
    model_config = ConfigDict(extra="ignore")
    gpu: int | None = None  # pin to one graphics card


class LoraRef(BaseModel):
    id: str
    weight: float = Field(0.8, ge=-2, le=2)


class ModelParams(Params):
    model: str
    prompt: str = Field("", max_length=8000)
    negative_prompt: str = Field("", max_length=4000)
    steps: int | None = Field(None, ge=1, le=200)
    guidance: float | None = Field(None, ge=0, le=50)
    seed: int = -1
    loras: list[LoraRef] = Field(default_factory=list, max_length=6)


def new_seed(seed: int) -> int:
    return seed if seed is not None and seed >= 0 else random.randint(0, 2**32 - 1)


def generator(seed: int):
    import torch

    return torch.Generator("cpu").manual_seed(int(seed))


def filter_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Only pass arguments the pipeline's __call__ accepts (they differ between model families)."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if k in params and v is not None}


def input_image(ctx: TaskContext, ref: str) -> Image.Image:
    try:
        return media.load_image(ctx.services.storage.path_of(ref))
    except KeyError:
        raise TaskError("input_missing", ref=ref)


def input_mask(ctx: TaskContext, ref: str, size: tuple[int, int], grow: int = 0, blur: int = 0) -> Image.Image:
    """White = area to change. Accepts painted masks with transparency too."""
    m = input_image(ctx, ref)
    if m.mode == "RGBA" and m.getextrema()[3][0] < 255:
        m = m.split()[-1]  # painted strokes on a transparent layer
    m = m.convert("L").resize(size, Image.Resampling.BILINEAR)
    m = m.point(lambda v: 255 if v > 127 else 0)
    if grow > 0:
        m = m.filter(ImageFilter.MaxFilter(grow * 2 + 1))
    if blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(blur))
    if m.getbbox() is None:
        raise TaskError("empty_mask")
    return m


def input_video(ctx: TaskContext, ref: str) -> Path:
    try:
        p = ctx.services.storage.path_of(ref)
    except KeyError:
        raise TaskError("input_missing", ref=ref)
    if ctx.services.storage.get(ref).kind != "video":
        raise TaskError("not_a_video", ref=ref)
    return p


def resolve_model(ctx: TaskContext, model_id: str, task: str) -> ModelSource:
    src = ctx.services.models.resolve(model_id)
    if task not in src.tasks():
        raise ModelError("task_unsupported", task=task, model=src.name)
    missing = ctx.services.models.missing_downloads(src)
    if missing:
        raise ModelError("model_missing", repo=missing[0]["repo"], model=src.id)
    return src


def resolve_loras(ctx: TaskContext, src: ModelSource, loras: list[LoraRef]) -> list[tuple[str, float]]:
    out = []
    for ref in loras:
        lm = ctx.services.local.get(ref.id)
        if lm is None or lm.kind != "lora":
            raise TaskError("lora_missing", id=ref.id)
        if lm.family and not src.family.id.startswith(lm.family):
            raise TaskError("lora_incompatible", lora=lm.name, model=src.name)
        out.append((lm.path, ref.weight))
    if out and not src.family.lora:
        raise TaskError("lora_unsupported", model=src.name)
    return out


def run_pipeline(ctx: TaskContext, src: ModelSource, task: str, call, loras: list[tuple[str, float]] | None = None):
    """Acquires the pipeline and runs `call(pipe)`; on out-of-memory retries once with a safer plan."""
    import torch

    models = ctx.services.models
    for attempt in range(2):
        ctx.check()
        pipe, plan = models.acquire(src, task, ctx.slots, loras, ctx.status)
        if ctx.services.hw.backend == "cuda" and plan.devices and torch.cuda.is_available():
            torch.cuda.set_device(plan.devices[0])
        try:
            with torch.inference_mode():
                return call(pipe)
        except torch.OutOfMemoryError:
            log.warning("OOM while running %s (attempt %d)", src.id, attempt + 1)
            del pipe
            if attempt == 1 or not models.report_oom(src, ctx.slots):
                raise ModelError("oom")
            ctx.progress(stage="fallback")
        except Cancelled:
            raise
    raise ModelError("oom")


def public_params(p: BaseModel, **extra) -> dict:
    d = p.model_dump(exclude={"gpu"})
    d.update(extra)
    return d


# ----------------------------------------------------------------------------- tool models
def ensure_tool(ctx: TaskContext, tool_id: str) -> Path:
    """Downloads a small non-diffusers model (upscaler, background removal, eraser) once."""
    tool = TOOLS_BY_ID.get(tool_id)
    if tool is None:
        raise TaskError("tool_missing", id=tool_id)
    dest = ctx.services.paths.tools / tool.filename
    if dest.exists():
        return dest
    download_tool(tool, dest, ctx)
    return dest


def download_tool(tool: ToolModel, dest: Path, ctx: TaskContext | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    md5 = hashlib.md5()
    if ctx:
        ctx.progress(stage="downloading_tool", force=True)
    req = urllib.request.Request(tool.url, headers={"User-Agent": "EditStudio/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or tool.size_mb * 1024 * 1024)
            done = 0
            while True:
                if ctx:
                    ctx.check()
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                md5.update(chunk)
                done += len(chunk)
                if ctx:
                    ctx.progress(min(0.99, done / total), "downloading_tool", detail=f"{done >> 20}/{total >> 20} MB")
    except Cancelled:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise TaskError("network", str(e), url=tool.url)
    if tool.md5 and md5.hexdigest() != tool.md5:
        tmp.unlink(missing_ok=True)
        raise TaskError("checksum", url=tool.url)
    tmp.replace(dest)
