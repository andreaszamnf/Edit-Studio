"""Non-diffusion AI tools: background removal (ONNX), upscaling and object erasing (spandrel)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger(__name__)

_cache: dict[tuple, object] = {}
_cache_lock = threading.Lock()


def _cached(key: tuple, factory: Callable[[], object]):
    with _cache_lock:
        if key not in _cache:
            if len(_cache) > 4:
                _cache.clear()
            _cache[key] = factory()
        return _cache[key]


# ----------------------------------------------------------------------------- background removal
BG_PREPROCESS = {
    "birefnet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True),
    "birefnet-lite": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True),
    "isnet": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), False),
}


def _ort_providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available


class BackgroundRemover:
    def __init__(self, model_path: Path, kind: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=_ort_providers())
        self.mean, self.std, self.sigmoid = BG_PREPROCESS[kind]
        self.input_name = self.session.get_inputs()[0].name

    def mask(self, img: Image.Image) -> Image.Image:
        """Alpha mask (L) of the foreground, same size as `img`."""
        im = np.asarray(img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS), dtype=np.float32)
        im = im / max(float(im.max()), 1e-6)
        x = (im - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.session.run(None, {self.input_name: x})[0][:, 0, :, :]
        if self.sigmoid:
            pred = 1.0 / (1.0 + np.exp(-pred))
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / max(hi - lo, 1e-6)
        m = Image.fromarray((np.squeeze(pred) * 255).astype(np.uint8), mode="L")
        return m.resize(img.size, Image.Resampling.LANCZOS)


def background_remover(model_path: Path, kind: str) -> BackgroundRemover:
    return _cached(("bg", str(model_path)), lambda: BackgroundRemover(model_path, kind))


def apply_background(img: Image.Image, mask: Image.Image, mode: str, color: str = "#ffffff") -> Image.Image:
    rgb = img.convert("RGB")
    if mode == "transparent":
        out = rgb.copy()
        out.putalpha(mask)
        return out
    if mode == "blur":
        bg = rgb.filter(ImageFilter.GaussianBlur(max(img.size) // 40 + 4))
    else:
        bg = Image.new("RGB", img.size, color)
    return Image.composite(rgb, bg, mask)


# ----------------------------------------------------------------------------- spandrel models
def _torch():
    import torch

    return torch


def load_spandrel(path: Path, device: str, reduced_precision: bool = True):
    from spandrel import ModelLoader

    def factory():
        torch = _torch()
        desc = ModelLoader(device="cpu").load_from_file(str(path))
        desc.model.eval()
        if reduced_precision and device.startswith("cuda"):
            if desc.supports_half:
                desc.model.half()
            elif getattr(desc, "supports_bfloat16", False):
                desc.model.to(torch.bfloat16)
        desc.model.to(device)
        return desc

    return _cached(("spandrel", str(path), device, reduced_precision), factory)


def _to_tensor(img: Image.Image, device: str, dtype):
    torch = _torch()
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _to_image(t) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8))


def upscale(desc, img: Image.Image, device: str, tile: int = 512, overlap: int = 24,
            progress: Callable[[float], None] | None = None, check: Callable[[], None] | None = None) -> Image.Image:
    """Tiled super-resolution; each tile is processed with context and the borders are cropped away."""
    torch = _torch()
    dtype = next(desc.model.parameters()).dtype
    scale = int(desc.scale)
    alpha = img.split()[-1] if img.mode == "RGBA" else None
    rgb = img.convert("RGB")
    W, H = rgb.size
    out = Image.new("RGB", (W * scale, H * scale))
    xs = list(range(0, W, tile))
    ys = list(range(0, H, tile))
    total = len(xs) * len(ys)
    req = getattr(desc, "size_requirements", None)
    mult = max(1, int(getattr(req, "multiple_of", 1) or 1))
    minimum = int(getattr(req, "minimum", 0) or 0)
    n = 0
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                if check:
                    check()
                x0, y0 = max(0, x - overlap), max(0, y - overlap)
                x1, y1 = min(W, x + tile + overlap), min(H, y + tile + overlap)
                patch = rgb.crop((x0, y0, x1, y1))
                pw, ph = patch.size
                pad_w = max(minimum, pw + (-pw) % mult) - pw  # architectures with window sizes
                pad_h = max(minimum, ph + (-ph) % mult) - ph
                t = _to_tensor(patch, device, dtype)
                if pad_w or pad_h:
                    t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
                res = _to_image(desc(t)[:, :, : ph * scale, : pw * scale])
                cx, cy = (x - x0) * scale, (y - y0) * scale
                cw, ch = (min(W, x + tile) - x) * scale, (min(H, y + tile) - y) * scale
                out.paste(res.crop((cx, cy, cx + cw, cy + ch)), (x * scale, y * scale))
                n += 1
                if progress:
                    progress(n / total)
    if alpha is not None:
        out.putalpha(alpha.resize(out.size, Image.Resampling.BICUBIC))
    return out


def erase(desc, img: Image.Image, mask: Image.Image, device: str) -> Image.Image:
    """Removes the masked area with LaMa, working on a crop around the mask for detail."""
    torch = _torch()
    rgb = img.convert("RGB")
    W, H = rgb.size
    box = mask.getbbox()
    if box is None:
        return rgb
    bw, bh = box[2] - box[0], box[3] - box[1]
    margin = max(64, int(max(bw, bh) * 0.6))
    x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
    x1, y1 = min(W, box[2] + margin), min(H, box[3] + margin)
    crop, mcrop = rgb.crop((x0, y0, x1, y1)), mask.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    limit = 1024
    s = min(1.0, limit / max(cw, ch))
    tw, th = max(16, int(cw * s) // 8 * 8), max(16, int(ch * s) // 8 * 8)
    c_in = crop.resize((tw, th), Image.Resampling.LANCZOS)
    m_in = mcrop.resize((tw, th), Image.Resampling.NEAREST).point(lambda v: 255 if v > 127 else 0)
    dtype = next(desc.model.parameters()).dtype
    with torch.inference_mode():
        t_img = _to_tensor(c_in, device, dtype)
        t_mask = torch.from_numpy(np.asarray(m_in, dtype=np.float32) / 255.0)[None, None].to(device=device, dtype=dtype)
        res = _to_image(desc(t_img, t_mask)).resize((cw, ch), Image.Resampling.LANCZOS)
    soft = mcrop.filter(ImageFilter.GaussianBlur(3))
    out = rgb.copy()
    out.paste(Image.composite(res, crop, soft), (x0, y0))
    return out
