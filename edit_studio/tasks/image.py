"""AI image tasks: create, variations, inpaint, outpaint, instruction editing."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter
from pydantic import Field

from .. import media
from ..jobs import TaskContext, task
from ..models.manager import ModelSource
from .common import (
    ModelParams,
    filter_kwargs,
    generator,
    input_image,
    input_mask,
    new_seed,
    public_params,
    resolve_loras,
    resolve_model,
    run_pipeline,
)

# Pixel budget each family is trained around (keeps speed and quality predictable).
FAMILY_PIXELS = {"sd15": 512 * 640, "qwen_image": 1328 * 1328, "qwen_image_edit": 1024 * 1024}
MULTI_IMAGE_FAMILIES = ("qwen_image_edit", "flux2_klein", "qwen_image_21")
INPAINT_STRENGTH = {"sd15": 0.99, "sdxl": 0.99, "flux1_fill": 1.0, "z_image": 1.0, "qwen_image": 0.9}


def _settings(src: ModelSource, p: ModelParams) -> tuple[int, float]:
    d = src.defaults
    steps = p.steps or int(d.get("steps", 30))
    guidance = float(d.get("guidance", 4.0)) if p.guidance is None else float(p.guidance)
    return steps, guidance


def _base_kwargs(src: ModelSource, p: ModelParams, steps: int, guidance: float, seed: int, cb) -> dict:
    fam = src.family
    kw = {
        "prompt": p.prompt,
        "num_inference_steps": steps,
        "generator": generator(seed),
        "callback_on_step_end": cb,
        fam.guidance_arg: guidance,
    }
    neg = p.negative_prompt.strip()
    if fam.guidance_arg == "true_cfg_scale":
        if guidance > 1.0:  # Qwen runs true CFG only when a negative prompt is present
            kw["negative_prompt"] = neg or " "
    elif fam.negative_prompt and neg and guidance > (0.0 if fam.id == "z_image" else 1.0):
        kw["negative_prompt"] = neg
    return kw


def _work_size(src: ModelSource, w: int, h: int) -> tuple[int, int]:
    budget = FAMILY_PIXELS.get(src.family.id, 1024 * 1024)
    if w * h < budget * 0.6:  # small pictures: work bigger, the model needs room
        scale = (budget / (w * h)) ** 0.5
        w, h = w * scale, h * scale
    return media.snap_size(int(w), int(h), src.family.multiple_of, max_pixels=int(budget * 1.15))


def _run_batch(ctx: TaskContext, src: ModelSource, task_name: str, pipe_task: str, p, count: int, make_kwargs,
               post=None, loras=None, sources=None, extra_params=None):
    """Runs `count` generations (seed, seed+1, ...) and saves each result to the gallery."""
    storage = ctx.services.storage
    steps, guidance = _settings(src, p)
    seed0 = new_seed(p.seed)
    outputs = []

    def call(pipe):
        for i in range(len(outputs), count):
            seed = seed0 + i
            cb = ctx.step_callback(steps, i / count, (i + 1) / count)
            kw = _base_kwargs(src, p, steps, guidance, seed, cb)
            kw.update(make_kwargs(pipe))
            img = pipe(**filter_kwargs(pipe.__call__, kw)).images[0]
            if post:
                img = post(img)
            ctx.progress(stage="saving")
            params = public_params(p, seed=seed, steps=steps, guidance=guidance, **(extra_params or {}))
            outputs.append(storage.add_image(img, task=task_name, model=src.id, params=params, sources=sources or []))
        return outputs

    return run_pipeline(ctx, src, pipe_task, call, loras)


# ----------------------------------------------------------------------------- create
class GenerateImage(ModelParams):
    width: int = Field(0, ge=0, le=4096)  # 0 = model default
    height: int = Field(0, ge=0, le=4096)
    count: int = Field(1, ge=1, le=8)


@task("image.generate", "gpu", GenerateImage, model_param="model")
def generate(ctx: TaskContext, p: GenerateImage):
    src = resolve_model(ctx, p.model, "txt2img")
    loras = resolve_loras(ctx, src, p.loras)
    d = src.defaults
    w, h = media.snap_size(p.width or d.get("width", 1024), p.height or d.get("height", 1024), src.family.multiple_of)
    return _run_batch(ctx, src, "image.generate", "txt2img", p, p.count, lambda pipe: {"width": w, "height": h},
                      loras=loras, extra_params={"width": w, "height": h})


# ----------------------------------------------------------------------------- variations
class Img2Img(ModelParams):
    image: str
    strength: float | None = Field(None, ge=0.05, le=1.0)
    count: int = Field(1, ge=1, le=8)


@task("image.img2img", "gpu", Img2Img, model_param="model")
def img2img(ctx: TaskContext, p: Img2Img):
    src = resolve_model(ctx, p.model, "img2img")
    loras = resolve_loras(ctx, src, p.loras)
    base = media.flatten(input_image(ctx, p.image))
    w, h = _work_size(src, *base.size)
    init = base.resize((w, h), Image.Resampling.LANCZOS)
    strength = p.strength if p.strength is not None else float(src.defaults.get("strength", 0.6))
    return _run_batch(
        ctx, src, "image.img2img", "img2img", p, p.count,
        lambda pipe: {"image": init, "strength": strength, "width": w, "height": h},
        loras=loras, sources=[p.image], extra_params={"strength": strength},
    )


# ----------------------------------------------------------------------------- inpaint
class InpaintImage(ModelParams):
    image: str
    mask: str
    strength: float | None = Field(None, ge=0.05, le=1.0)
    grow: int = Field(6, ge=0, le=64)
    feather: int = Field(10, ge=0, le=64)
    reference: str | None = None  # FLUX.2 klein: put this object in the painted area
    count: int = Field(1, ge=1, le=8)


def _inpaint(ctx: TaskContext, p, src: ModelSource, base: Image.Image, mask_full: Image.Image, hard_mask: Image.Image,
             task_name: str, sources: list[str], extra=None):
    """Runs an inpaint pipeline at a model-friendly size and pastes the result back seamlessly."""
    W, H = base.size
    w, h = _work_size(src, W, H)
    work = base.resize((w, h), Image.Resampling.LANCZOS)
    m_work = hard_mask.resize((w, h), Image.Resampling.BILINEAR)
    strength = p.strength if getattr(p, "strength", None) is not None else INPAINT_STRENGTH.get(src.family.id, 0.9)
    ref = media.flatten(input_image(ctx, p.reference)) if getattr(p, "reference", None) else None
    loras = resolve_loras(ctx, src, p.loras)

    def kwargs(pipe):
        kw = {"image": work, "mask_image": m_work, "strength": strength, "width": w, "height": h}
        if ref is not None:
            kw["image_reference"] = ref
        return kw

    def paste_back(img: Image.Image) -> Image.Image:
        return Image.composite(img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS), base, mask_full)

    return _run_batch(ctx, src, task_name, "inpaint", p, p.count, kwargs, post=paste_back, loras=loras,
                      sources=sources, extra_params={"strength": strength, **(extra or {})})


@task("image.inpaint", "gpu", InpaintImage, model_param="model")
def inpaint(ctx: TaskContext, p: InpaintImage):
    src = resolve_model(ctx, p.model, "inpaint")
    base = media.flatten(input_image(ctx, p.image))
    hard = input_mask(ctx, p.mask, base.size, grow=p.grow)
    soft = input_mask(ctx, p.mask, base.size, grow=p.grow, blur=p.feather)
    return _inpaint(ctx, p, src, base, soft, hard, "image.inpaint", [p.image])


class OutpaintImage(ModelParams):
    image: str
    left: int = Field(0, ge=0, le=4096)
    right: int = Field(0, ge=0, le=4096)
    top: int = Field(0, ge=0, le=4096)
    bottom: int = Field(0, ge=0, le=4096)
    count: int = Field(1, ge=1, le=8)
    strength: float | None = None


@task("image.outpaint", "gpu", OutpaintImage, model_param="model")
def outpaint(ctx: TaskContext, p: OutpaintImage):
    src = resolve_model(ctx, p.model, "inpaint")
    base = media.flatten(input_image(ctx, p.image))
    W, H = base.size
    NW, NH = W + p.left + p.right, H + p.top + p.bottom
    if (NW, NH) == (W, H):
        from ..jobs import TaskError

        raise TaskError("nothing_to_extend")
    # Start the new area from a blurred stretch of the photo: gives the model the right colours.
    canvas = base.resize((NW, NH), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(max(NW, NH) // 24))
    canvas.paste(base, (p.left, p.top))
    overlap = max(8, min(W, H) // 40)
    hard = Image.new("L", (NW, NH), 255)
    keep = (
        p.left + (overlap if p.left else 0),
        p.top + (overlap if p.top else 0),
        p.left + W - (overlap if p.right else 0),
        p.top + H - (overlap if p.bottom else 0),
    )
    ImageDraw.Draw(hard).rectangle(keep, fill=0)
    soft = hard.filter(ImageFilter.GaussianBlur(overlap / 2))
    p.strength = 1.0
    return _inpaint(ctx, p, src, canvas, soft, hard, "image.outpaint", [p.image],
                    extra={"left": p.left, "right": p.right, "top": p.top, "bottom": p.bottom})


# ----------------------------------------------------------------------------- instruction edit
class EditImage(ModelParams):
    images: list[str] = Field(min_length=1, max_length=8)
    width: int = Field(0, ge=0, le=4096)
    height: int = Field(0, ge=0, le=4096)
    count: int = Field(1, ge=1, le=8)


@task("image.edit", "gpu", EditImage, model_param="model")
def edit(ctx: TaskContext, p: EditImage):
    src = resolve_model(ctx, p.model, "edit")
    loras = resolve_loras(ctx, src, p.loras)
    images = [media.flatten(input_image(ctx, ref)) for ref in p.images]
    if p.width and p.height:
        w, h = media.snap_size(p.width, p.height, src.family.multiple_of)
    else:
        w, h = media.snap_size(*images[0].size, src.family.multiple_of, max_pixels=1024 * 1024)
    multi = src.family.id in MULTI_IMAGE_FAMILIES

    def kwargs(pipe):
        return {"image": images if multi else images[0], "width": w, "height": h}

    return _run_batch(ctx, src, "image.edit", "edit", p, p.count, kwargs, loras=loras, sources=list(p.images),
                      extra_params={"width": w, "height": h})
