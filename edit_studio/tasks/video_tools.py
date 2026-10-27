"""Video tools: upscale, remove background, smooth motion, cut/speed/crop, GIF, frames, join."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image
from pydantic import Field

from .. import media, tools
from ..jobs import TaskContext, TaskError, task
from .common import Params, ensure_tool, input_video
from .image_tools import gpu_retry, upscaler_path


def _new_video(ctx: TaskContext, ext: str = ".mp4"):
    return ctx.services.storage.video_path(ext)


def _save(ctx: TaskContext, oid: str, path: Path, task_name: str, p: Params, sources: list[str], model: str = ""):
    return ctx.services.storage.add_video(oid, path, task=task_name, model=model, params=p.model_dump(exclude={"gpu"}),
                                          sources=sources)


def _ffmpeg(ctx: TaskContext, args: list[str], duration: float, start: float = 0.0, end: float = 1.0):
    media.run_ffmpeg(args, duration=duration, progress=lambda f: ctx.progress(start + (end - start) * f, "processing"),
                     should_stop=ctx.job.cancel_event.is_set)


# ----------------------------------------------------------------------------- AI per-frame tools
class UpscaleVideo(Params):
    video: str
    model: str = "realesr-general-x4"
    scale: int = Field(2, ge=1, le=4)
    tile: int = Field(512, ge=128, le=2048)


@task("video.upscale", "gpu", UpscaleVideo)
def upscale_video(ctx: TaskContext, p: UpscaleVideo):
    path = input_video(ctx, p.video)
    info = media.probe_video(path)
    model_path, _ = upscaler_path(ctx, p.model)
    oid, out = _new_video(ctx)
    tmp = out.with_suffix(".noaudio.mp4")
    total = info.frames
    crf = ctx.services.settings.settings.video_quality
    target = (info.width * p.scale, info.height * p.scale)

    def run():
        desc = tools.load_spandrel(model_path, ctx.device)
        with media.VideoWriter(tmp, info.fps, crf) as w:
            for i, frame in enumerate(media.iter_frames(path)):
                ctx.check()
                img = tools.upscale(desc, Image.fromarray(frame), ctx.device, tile=p.tile)
                if img.size != target:
                    img = img.resize(target, Image.Resampling.LANCZOS)
                w.write(img)
                ctx.progress(0.95 * (i + 1) / total, "upscaling", step=i + 1, steps=total)

    gpu_retry(ctx, run)
    if not media.mux_audio(tmp, path, out):
        tmp.replace(out)
    tmp.unlink(missing_ok=True)
    return [_save(ctx, oid, out, "video.upscale", p, [p.video], p.model)]


class RemoveVideoBackground(Params):
    video: str
    model: str = Field("birefnet-lite", pattern="^(birefnet|birefnet-lite|isnet)$")
    mode: str = Field("color", pattern="^(transparent|color|blur)$")
    color: str = Field("#00ff00", pattern="^#[0-9a-fA-F]{6}$")


@task("video.remove_bg", "cpu", RemoveVideoBackground)
def remove_video_bg(ctx: TaskContext, p: RemoveVideoBackground):
    path = input_video(ctx, p.video)
    info = media.probe_video(path)
    remover = tools.background_remover(ensure_tool(ctx, p.model), p.model)
    alpha = p.mode == "transparent"
    oid, out = _new_video(ctx, ".webm" if alpha else ".mp4")
    tmp = out.with_suffix(".noaudio" + out.suffix)
    total = info.frames
    crf = ctx.services.settings.settings.video_quality
    with media.VideoWriter(tmp, info.fps, crf, alpha=alpha) as w:
        for i, frame in enumerate(media.iter_frames(path)):
            ctx.check()
            img = Image.fromarray(frame)
            w.write(tools.apply_background(img, remover.mask(img), p.mode, p.color))
            ctx.progress(0.95 * (i + 1) / total, "processing", step=i + 1, steps=total)
    if alpha or not media.mux_audio(tmp, path, out):
        tmp.replace(out)
    tmp.unlink(missing_ok=True)
    return [_save(ctx, oid, out, "video.remove_bg", p, [p.video], p.model)]


# ----------------------------------------------------------------------------- ffmpeg tools
class SmoothVideo(Params):
    video: str
    factor: int = Field(2, ge=2, le=4)
    slowmo: bool = False  # keep the frame rate and play the extra frames slower


@task("video.interpolate", "cpu", SmoothVideo)
def interpolate_video(ctx: TaskContext, p: SmoothVideo):
    path = input_video(ctx, p.video)
    info = media.probe_video(path)
    target = round(info.fps * p.factor, 3)
    mi = f"minterpolate=fps={target}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    oid, out = _new_video(ctx)
    crf = str(ctx.services.settings.settings.video_quality)
    if p.slowmo:
        vf = f"{mi},setpts={p.factor}*PTS"
        args = ["-i", str(path), "-vf", vf, "-r", str(info.fps), "-an"]
        duration = info.duration * p.factor
    else:
        args = ["-i", str(path), "-vf", mi, "-c:a", "aac"]
        duration = info.duration
    _ffmpeg(ctx, [*args, "-c:v", "libx264", "-crf", crf, "-pix_fmt", "yuv420p", str(out)], duration)
    return [_save(ctx, oid, out, "video.interpolate", p, [p.video])]


ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5, "4:3": 4 / 3, "21:9": 21 / 9}


class TransformVideo(Params):
    video: str
    start: float = Field(0.0, ge=0)
    end: float = Field(0.0, ge=0)  # 0 = until the end
    speed: float = Field(1.0, ge=0.25, le=4.0)
    reverse: bool = False
    aspect: str = Field("", pattern="^(|16:9|9:16|1:1|4:5|4:3|21:9)$")
    height: int = Field(0, ge=0, le=4320)  # output height, 0 = keep
    rotate: int = Field(0, ge=0, le=270, multiple_of=90)
    flip: bool = False
    mute: bool = False


@task("video.transform", "cpu", TransformVideo)
def transform_video(ctx: TaskContext, p: TransformVideo):
    path = input_video(ctx, p.video)
    info = media.probe_video(path)
    end = p.end if p.end and p.end > p.start else info.duration
    length = max(0.05, end - p.start)
    vf: list[str] = []
    af: list[str] = []
    if p.aspect:
        r = ASPECTS[p.aspect]
        vf.append(f"crop='if(gt(iw/ih,{r}),ih*{r},iw)':'if(gt(iw/ih,{r}),ih,iw/{r})'")
    if p.rotate:
        vf.append({90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}[p.rotate])
    if p.flip:
        vf.append("hflip")
    if p.height:
        vf.append(f"scale=-2:{p.height - p.height % 2}")
    if p.reverse:
        vf.append("reverse")
        af.append("areverse")
    if p.speed != 1.0:
        vf.append(f"setpts=PTS/{p.speed}")
        af.append(media._atempo_chain(p.speed))
    vf.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")
    oid, out = _new_video(ctx)
    args = ["-ss", f"{p.start:.3f}", "-t", f"{length:.3f}", "-i", str(path), "-vf", ",".join(vf)]
    if p.mute or not info.has_audio:
        args.append("-an")
    else:
        if af:
            args += ["-af", ",".join(af)]
        args += ["-c:a", "aac", "-b:a", "192k"]
    crf = str(ctx.services.settings.settings.video_quality)
    _ffmpeg(ctx, [*args, "-c:v", "libx264", "-crf", crf, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            length / p.speed)
    return [_save(ctx, oid, out, "video.transform", p, [p.video])]


class VideoToGif(Params):
    video: str
    start: float = Field(0.0, ge=0)
    end: float = Field(0.0, ge=0)
    fps: int = Field(12, ge=4, le=30)
    width: int = Field(480, ge=120, le=1280)


@task("video.to_gif", "cpu", VideoToGif)
def video_to_gif(ctx: TaskContext, p: VideoToGif):
    path = input_video(ctx, p.video)
    info = media.probe_video(path)
    end = p.end if p.end and p.end > p.start else min(info.duration, p.start + 15)
    length = max(0.1, end - p.start)
    oid, out = _new_video(ctx, ".gif")
    graph = (f"fps={p.fps},scale={p.width}:-1:flags=lanczos,split[a][b];"
             "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a")
    _ffmpeg(ctx, ["-ss", f"{p.start:.3f}", "-t", f"{length:.3f}", "-i", str(path), "-filter_complex", graph,
                  "-loop", "0", str(out)], length)
    return [_save(ctx, oid, out, "video.to_gif", p, [p.video])]


class ExtractFrame(Params):
    video: str
    time: float = Field(-1.0)  # seconds; negative = last frame


@task("video.extract_frame", "cpu", ExtractFrame)
def extract_frame(ctx: TaskContext, p: ExtractFrame):
    path = input_video(ctx, p.video)
    info = media.probe_video(path)
    if p.time < 0:
        frame = None
        for frame in media.iter_frames(path):
            pass
        if frame is None:
            raise TaskError("video_too_short")
        img = Image.fromarray(frame)
    else:
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "frame.png"
            media.run_ffmpeg(["-ss", f"{min(p.time, max(0.0, info.duration - 0.05)):.3f}", "-i", str(path),
                              "-frames:v", "1", str(dst)])
            img = media.load_image(dst)
            img.load()
    item = ctx.services.storage.add_image(img, task="video.extract_frame", params=p.model_dump(exclude={"gpu"}),
                                          sources=[p.video], fmt="png")
    return [item]


class ConcatVideos(Params):
    videos: list[str] = Field(min_length=2, max_length=20)


@task("video.concat", "cpu", ConcatVideos)
def concat_videos(ctx: TaskContext, p: ConcatVideos):
    paths = [input_video(ctx, v) for v in p.videos]
    infos = [media.probe_video(x) for x in paths]
    W, H = infos[0].width - infos[0].width % 2, infos[0].height - infos[0].height % 2
    F = infos[0].fps or 24
    with_audio = all(i.has_audio for i in infos)
    args: list[str] = []
    for x in paths:
        args += ["-i", str(x)]
    parts = []
    for i in range(len(paths)):
        parts.append(f"[{i}:v]fps={F},scale={W}:{H}:force_original_aspect_ratio=decrease,"
                     f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]")
        if with_audio:
            parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
    joined = "".join(f"[v{i}]" + (f"[a{i}]" if with_audio else "") for i in range(len(paths)))
    graph = ";".join(parts) + f";{joined}concat=n={len(paths)}:v=1:a={1 if with_audio else 0}[v]" + ("[a]" if with_audio else "")
    oid, out = _new_video(ctx)
    maps = ["-map", "[v]"] + (["-map", "[a]", "-c:a", "aac"] if with_audio else [])
    crf = str(ctx.services.settings.settings.video_quality)
    _ffmpeg(ctx, [*args, "-filter_complex", graph, *maps, "-c:v", "libx264", "-crf", crf, "-pix_fmt", "yuv420p",
                  str(out)], sum(i.duration for i in infos))
    return [_save(ctx, oid, out, "video.concat", p, list(p.videos))]
