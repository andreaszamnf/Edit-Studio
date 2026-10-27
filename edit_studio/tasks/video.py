"""AI video tasks: text/image-to-video, restyle (video-to-video), continue a clip."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import Field

from .. import media
from ..jobs import TaskContext, TaskError, task
from ..models.catalog import WAN_NEGATIVE
from ..models.manager import ModelSource
from .common import (
    ModelParams,
    filter_kwargs,
    generator,
    input_image,
    input_video,
    new_seed,
    public_params,
    resolve_model,
    run_pipeline,
)


def _frames_4k1(n: int) -> int:
    """Wan needs 4k+1 frames."""
    return max(5, 4 * round((n - 1) / 4) + 1)


def _video_size(src: ModelSource, w: int, h: int, aspect: float | None) -> tuple[int, int]:
    """Model default size, flipped to portrait/landscape to match the input picture/video."""
    d = src.defaults
    w = w or d.get("width", 832)
    h = h or d.get("height", 480)
    if aspect is not None and (aspect < 1) != (w / h < 1) and abs(aspect - 1) > 0.05:
        w, h = h, w
    m = src.family.multiple_of
    return max(m, w // m * m), max(m, h // m * m)


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), (w, h), Image.Resampling.LANCZOS)


def _negative(src: ModelSource, p: ModelParams) -> str | None:
    neg = p.negative_prompt.strip()
    if src.family.id.startswith("wan"):
        return neg or WAN_NEGATIVE
    return neg or None


def _save_frames(ctx: TaskContext, frames, fps: float, *, task_name: str, src: ModelSource, params: dict,
                 sources: list[str], audio_from: Path | None = None, audio_start: float = 0.0):
    storage = ctx.services.storage
    crf = ctx.services.settings.settings.video_quality
    ctx.progress(stage="encoding", force=True)
    oid, path = storage.video_path(".mp4")
    if audio_from is not None:
        tmp = path.with_suffix(".noaudio.mp4")
        media.write_video(frames, tmp, fps, crf)
        if not media.mux_audio(tmp, audio_from, path, start=audio_start):
            tmp.replace(path)
        tmp.unlink(missing_ok=True)
    else:
        media.write_video(frames, path, fps, crf)
    return storage.add_video(oid, path, task=task_name, model=src.id, params=params, sources=sources)


# ----------------------------------------------------------------------------- generate
class GenerateVideo(ModelParams):
    image: str | None = None  # start picture -> image-to-video
    width: int = Field(0, ge=0, le=2048)
    height: int = Field(0, ge=0, le=2048)
    frames: int = Field(0, ge=0, le=241)  # 0 = model default (~5 s)
    fps: int = Field(0, ge=0, le=60)
    guidance_2: float | None = Field(None, ge=0, le=20)


@task("video.generate", "gpu", GenerateVideo, model_param="model")
def generate_video(ctx: TaskContext, p: GenerateVideo):
    kind = "img2vid" if p.image else "txt2vid"
    src = resolve_model(ctx, p.model, kind)
    start = input_image(ctx, p.image) if p.image else None
    w, h = _video_size(src, p.width, p.height, (start.width / start.height) if start else None)
    frames = _frames_4k1(p.frames or src.defaults.get("frames", 81))
    fps = p.fps or src.defaults.get("fps", 16)
    steps = p.steps or int(src.defaults.get("steps", 30))
    guidance = float(src.defaults.get("guidance", 5.0)) if p.guidance is None else p.guidance
    guidance_2 = p.guidance_2 if p.guidance_2 is not None else src.defaults.get("guidance_2")
    seed = new_seed(p.seed)

    def call(pipe):
        kw = {
            "prompt": p.prompt,
            "negative_prompt": _negative(src, p),
            "height": h,
            "width": w,
            "num_frames": frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator(seed),
            "output_type": "np",
            "callback_on_step_end": ctx.step_callback(steps, 0.0, 0.92, final_stage="decoding"),
        }
        if guidance_2 is not None and getattr(pipe, "transformer_2", None) is not None:
            kw["guidance_scale_2"] = guidance_2
        if start is not None:
            kw["image"] = _fit(start, w, h)
        return pipe(**filter_kwargs(pipe.__call__, kw)).frames[0]

    out = run_pipeline(ctx, src, kind, call)
    params = public_params(p, seed=seed, width=w, height=h, frames=frames, fps=fps, steps=steps, guidance=guidance)
    return [_save_frames(ctx, out, fps, task_name="video.generate", src=src, params=params,
                         sources=[p.image] if p.image else [])]


# ----------------------------------------------------------------------------- restyle
class RestyleVideo(ModelParams):
    video: str
    strength: float = Field(0.7, ge=0.1, le=1.0)
    max_seconds: float = Field(5.0, ge=1.0, le=12.0)
    width: int = Field(0, ge=0, le=2048)
    height: int = Field(0, ge=0, le=2048)


@task("video.restyle", "gpu", RestyleVideo, model_param="model")
def restyle_video(ctx: TaskContext, p: RestyleVideo):
    src = resolve_model(ctx, p.model, "vid2vid")
    path = input_video(ctx, p.video)
    info = media.probe_video(path)
    fps = src.defaults.get("fps", 16)
    w, h = _video_size(src, p.width, p.height, info.width / max(1, info.height))
    ctx.progress(0.01, "reading", force=True)
    n = _frames_4k1(min(int(p.max_seconds * fps), int(info.duration * fps) or 5))
    frames = [_fit(f, w, h) for f in media.read_frames(path, max_frames=n, fps=fps)]
    if len(frames) < 5:
        raise TaskError("video_too_short")
    frames = frames[: (len(frames) - 1) // 4 * 4 + 1]  # Wan needs 4k+1 frames
    steps = p.steps or int(src.defaults.get("steps", 30))
    guidance = float(src.defaults.get("guidance", 5.0)) if p.guidance is None else p.guidance
    seed = new_seed(p.seed)

    def call(pipe):
        kw = {
            "video": frames,
            "prompt": p.prompt,
            "negative_prompt": _negative(src, p),
            "height": h,
            "width": w,
            "strength": p.strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator(seed),
            "output_type": "np",
            "callback_on_step_end": ctx.step_callback(steps, 0.02, 0.92, final_stage="decoding"),
        }
        return pipe(**filter_kwargs(pipe.__call__, kw)).frames[0]

    out = run_pipeline(ctx, src, "vid2vid", call)
    params = public_params(p, seed=seed, width=w, height=h, fps=fps, steps=steps, guidance=guidance)
    return [_save_frames(ctx, out, fps, task_name="video.restyle", src=src, params=params, sources=[p.video],
                         audio_from=path)]


# ----------------------------------------------------------------------------- continue
class ExtendVideo(ModelParams):
    video: str
    frames: int = Field(0, ge=0, le=241)


@task("video.extend", "gpu", ExtendVideo, model_param="model")
def extend_video(ctx: TaskContext, p: ExtendVideo):
    src = resolve_model(ctx, p.model, "img2vid")
    path = input_video(ctx, p.video)
    info = media.probe_video(path)
    ctx.progress(0.01, "reading", force=True)
    last = None
    for f in media.iter_frames(path):
        last = f
    if last is None:
        raise TaskError("video_too_short")
    start = Image.fromarray(last)
    w, h = _video_size(src, 0, 0, start.width / start.height)
    frames = _frames_4k1(p.frames or src.defaults.get("frames", 81))
    fps = src.defaults.get("fps", 16)
    steps = p.steps or int(src.defaults.get("steps", 30))
    guidance = float(src.defaults.get("guidance", 5.0)) if p.guidance is None else p.guidance
    seed = new_seed(p.seed)

    def call(pipe):
        kw = {
            "image": _fit(start, w, h),
            "prompt": p.prompt,
            "negative_prompt": _negative(src, p),
            "height": h,
            "width": w,
            "num_frames": frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator(seed),
            "output_type": "np",
            "callback_on_step_end": ctx.step_callback(steps, 0.02, 0.9, final_stage="decoding"),
        }
        return pipe(**filter_kwargs(pipe.__call__, kw)).frames[0]

    cont = media.to_uint8_frames(run_pipeline(ctx, src, "img2vid", call))[1:]  # frame 0 repeats the last frame
    ctx.progress(0.93, "encoding", force=True)
    storage = ctx.services.storage
    oid, out = storage.video_path(".mp4")
    with tempfile.TemporaryDirectory() as td:
        cont_path = Path(td) / "cont.mp4"
        media.write_video(cont, cont_path, fps, 12)
        W, H = info.width - info.width % 2, info.height - info.height % 2
        F = info.fps or fps
        graph = (
            f"[0:v]fps={F},scale={W}:{H},setsar=1[a];"
            f"[1:v]fps={F},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[b];"
            "[a][b]concat=n=2:v=1:a=0[v]"
        )
        crf = ctx.services.settings.settings.video_quality
        media.run_ffmpeg(["-i", str(path), "-i", str(cont_path), "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
                          "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
                         should_stop=ctx.job.cancel_event.is_set)
    params = public_params(p, seed=seed, frames=frames, fps=fps, steps=steps, guidance=guidance)
    return [storage.add_video(oid, out, task="video.extend", model=src.id, params=params, sources=[p.video])]
