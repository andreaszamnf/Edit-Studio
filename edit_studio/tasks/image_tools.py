"""Image tools: remove background, upscale, erase objects."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import Field

from .. import tools
from ..jobs import TaskContext, TaskError, task
from ..models.catalog import TOOLS_BY_ID
from .common import Params, ensure_tool, input_image, input_mask

BG_MODELS = ("birefnet", "birefnet-lite", "isnet")


class RemoveBackground(Params):
    image: str
    model: str = Field("birefnet", pattern="^(birefnet|birefnet-lite|isnet)$")
    mode: str = Field("transparent", pattern="^(transparent|color|blur)$")
    color: str = Field("#ffffff", pattern="^#[0-9a-fA-F]{6}$")


@task("image.remove_bg", "cpu", RemoveBackground)
def remove_bg(ctx: TaskContext, p: RemoveBackground):
    remover = tools.background_remover(ensure_tool(ctx, p.model), p.model)
    img = input_image(ctx, p.image)
    ctx.progress(0.2, "processing", force=True)
    mask = remover.mask(img)
    ctx.progress(0.9, "saving")
    out = tools.apply_background(img, mask, p.mode, p.color)
    item = ctx.services.storage.add_image(out, task="image.remove_bg", model=p.model, params=p.model_dump(exclude={"gpu"}),
                                          sources=[p.image], fmt="png" if p.mode == "transparent" else None)
    return [item]


def upscaler_path(ctx: TaskContext, model: str) -> tuple[Path, int | None]:
    if model in TOOLS_BY_ID and TOOLS_BY_ID[model].kind == "upscaler":
        return ensure_tool(ctx, model), TOOLS_BY_ID[model].scale
    lm = ctx.services.local.get(model)
    if lm is None or lm.kind != "upscaler":
        raise TaskError("tool_missing", id=model)
    return Path(lm.path), None


def gpu_retry(ctx: TaskContext, fn):
    """Small tools share a GPU with cached diffusion models; free it once if memory runs out."""
    import torch

    try:
        return fn()
    except torch.OutOfMemoryError:
        ctx.services.models.unload_slots(ctx.slots)
        return fn()


class UpscaleImage(Params):
    image: str
    model: str = "realesrgan-x4"
    scale: int = Field(4, ge=1, le=8)
    tile: int = Field(512, ge=128, le=2048)


@task("image.upscale", "gpu", UpscaleImage)
def upscale_image(ctx: TaskContext, p: UpscaleImage):
    path, _ = upscaler_path(ctx, p.model)
    img = input_image(ctx, p.image)
    ctx.progress(0.02, "loading", force=True)

    def run():
        desc = tools.load_spandrel(path, ctx.device)
        out = tools.upscale(desc, img, ctx.device, tile=p.tile,
                            progress=lambda f: ctx.progress(0.05 + 0.9 * f, "upscaling"), check=ctx.check)
        return out, int(desc.scale)

    out, native = gpu_retry(ctx, run)
    if p.scale != native:
        out = out.resize((img.width * p.scale, img.height * p.scale), Image.Resampling.LANCZOS)
    ctx.progress(0.97, "saving")
    item = ctx.services.storage.add_image(out, task="image.upscale", model=p.model, params=p.model_dump(exclude={"gpu"}),
                                          sources=[p.image], fmt="png")
    return [item]


class EraseObjects(Params):
    image: str
    mask: str
    grow: int = Field(8, ge=0, le=64)


@task("image.erase", "gpu", EraseObjects)
def erase_objects(ctx: TaskContext, p: EraseObjects):
    path = ensure_tool(ctx, "lama")
    img = input_image(ctx, p.image)
    mask = input_mask(ctx, p.mask, img.size, grow=p.grow)
    ctx.progress(0.3, "processing", force=True)

    def run():
        desc = tools.load_spandrel(path, ctx.device, reduced_precision=False)
        return tools.erase(desc, img, mask, ctx.device)

    out = gpu_retry(ctx, run)
    if img.mode == "RGBA":
        out.putalpha(img.split()[-1])
    item = ctx.services.storage.add_image(out, task="image.erase", model="lama", params=p.model_dump(exclude={"gpu"}),
                                          sources=[p.image])
    return [item]
