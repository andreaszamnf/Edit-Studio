"""Finds model files that already exist on this computer and figures out what they are.

Looks inside ComfyUI, Stable Diffusion WebUI (A1111 / Forge), Fooocus, StabilityMatrix, SwarmUI
and any folder the user adds. Files are classified from their tensor names (read from the
safetensors / GGUF header only - nothing is loaded into memory).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import string
import struct
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_EXTS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".sft")

COMFY_DIRS = {
    "checkpoints": "checkpoint",
    "diffusion_models": "denoiser",
    "unet": "denoiser",
    "loras": "lora",
    "vae": "vae",
    "text_encoders": "text_encoder",
    "clip": "text_encoder",
    "upscale_models": "upscaler",
    "controlnet": "controlnet",
}
WEBUI_DIRS = {
    "Stable-diffusion": "checkpoint",
    "Lora": "lora",
    "LyCORIS": "lora",
    "VAE": "vae",
    "ESRGAN": "upscaler",
    "RealESRGAN": "upscaler",
    "SwinIR": "upscaler",
    "DAT": "upscaler",
    "HAT": "upscaler",
    "text_encoder": "text_encoder",
    "ControlNet": "controlnet",
    "diffusion_models": "denoiser",
}
SWARM_DIRS = {
    "Stable-Diffusion": "checkpoint",
    "diffusion_models": "denoiser",
    "Lora": "lora",
    "VAE": "vae",
    "clip": "text_encoder",
    "upscale_models": "upscaler",
}

# Base Hugging Face repos that provide the text encoders / VAE / configs for transformer-only files.
BASE_REPOS = {
    ("flux1", "dev"): "black-forest-labs/FLUX.1-dev",
    ("flux1", "schnell"): "black-forest-labs/FLUX.1-schnell",
    ("flux1_kontext", "dev"): "black-forest-labs/FLUX.1-Kontext-dev",
    ("flux1_fill", "dev"): "black-forest-labs/FLUX.1-Fill-dev",
    ("flux2_klein", "4b"): "black-forest-labs/FLUX.2-klein-4B",
    ("flux2_klein", "4b-base"): "black-forest-labs/FLUX.2-klein-base-4B",
    ("flux2_klein", "9b"): "black-forest-labs/FLUX.2-klein-9B",
    ("flux2_klein", "9b-base"): "black-forest-labs/FLUX.2-klein-base-9B",
    ("z_image", "turbo"): "Tongyi-MAI/Z-Image-Turbo",
    ("z_image", "base"): "Tongyi-MAI/Z-Image",
    ("qwen_image", ""): "Qwen/Qwen-Image-2512",
    ("qwen_image_edit", ""): "Qwen/Qwen-Image-Edit-2511",
    ("qwen_image_21", ""): "Qwen/Qwen-Image-2.1",
    ("wan_t2v", "1.3b"): "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ("wan_t2v", "14b"): "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    ("wan_ti2v", "5b"): "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}
SINGLE_FILE_CHECKPOINT_FAMILIES = ("sd15", "sdxl")


@dataclass
class LocalModel:
    id: str
    path: str
    name: str
    kind: str  # checkpoint | denoiser | lora | vae | text_encoder | upscaler | controlnet | unknown
    family: str | None
    variant: str
    fmt: str
    dtype: str
    size: int
    mtime: float
    source: str
    supported: bool
    note: str = ""
    base_repo: str | None = None

    def public(self) -> dict:
        return asdict(self)


def model_id(path: str) -> str:
    return "local:" + hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- headers
def read_safetensors_header(path: str) -> tuple[dict, dict]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("truncated file")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 200 * 1024 * 1024:
            raise ValueError("not a safetensors file")
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    return header, meta


def read_gguf_tensors(path: str) -> tuple[dict, dict]:
    from gguf import GGUFReader

    reader = GGUFReader(path)
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = {"dtype": t.tensor_type.name, "shape": [int(x) for x in reversed(t.shape)]}
    meta = {}
    for key in ("general.architecture", "general.file_type"):
        field = reader.fields.get(key)
        if field is not None:
            try:
                meta[key] = field.contents()
            except Exception:
                pass
    return tensors, meta


DTYPE_NAMES = {
    "BF16": "bf16",
    "F16": "fp16",
    "F32": "fp32",
    "F8_E4M3": "fp8",
    "F8_E5M2": "fp8",
}


def summarize_dtype(tensors: dict, fmt: str) -> str:
    counts: dict[str, int] = {}
    for info in tensors.values():
        d = info.get("dtype", "?")
        counts[d] = counts.get(d, 0) + 1
    if not counts:
        return "?"
    top = max(counts, key=counts.get)
    if fmt == "gguf":
        quant = [d for d in counts if d not in ("F32", "F16", "BF16")]
        return ("gguf-" + max(quant, key=counts.get)) if quant else "gguf-" + top.lower()
    return DTYPE_NAMES.get(top, top.lower())


def param_count(tensors: dict) -> int:
    total = 0
    for info in tensors.values():
        n = 1
        for s in info.get("shape", []):
            n *= int(s)
        total += n
    return total


# ----------------------------------------------------------------------------- classify
def _shape(tensors: dict, *names: str) -> list[int] | None:
    for n in names:
        if n in tensors:
            return list(tensors[n].get("shape", []))
    return None


def classify(tensors: dict, filename: str, kind_hint: str | None) -> tuple[str, str | None, str, bool, str]:
    """Returns (kind, family, variant, supported, note)."""
    keys = tensors.keys()
    lname = filename.lower()

    def has(*names: str) -> bool:
        return any(n in tensors for n in names)

    def has_prefixed(name: str) -> bool:
        return name in tensors or ("model.diffusion_model." + name) in tensors

    def contains(fragment: str) -> bool:
        return any(fragment in k for k in keys)

    # ComfyUI "fp8_scaled" files carry per-tensor scales diffusers can't read.
    scaled_fp8 = has("scaled_fp8") or any(k.endswith(".scale_weight") for k in keys)

    # ---- LoRA
    if contains("lora_up") or contains("lora_down") or contains(".lora_A") or contains(".lora_B") or contains("lora_unet_"):
        family = None
        if contains("lora_te2_") or contains("lora_te1_") or contains("text_encoder_2"):
            family = "sdxl"
        elif contains("double_blocks") or contains("single_blocks") or contains("single_transformer_blocks"):
            family = "flux1"
        elif contains("txt_mlp") or contains("img_mlp"):
            family = "qwen_image"
        elif contains("adaLN_modulation") and contains("layers."):
            family = "z_image"
        elif contains("cross_attn") and contains("blocks."):
            family = "wan"
        elif contains("lora_unet_"):
            k = next((k for k in keys if "attn2_to_k" in k and "lora_down" in k), None)
            dim = tensors[k]["shape"][-1] if k else 0
            family = "sdxl" if dim == 2048 else "sd15" if dim == 768 else None
        return "lora", family, "", family is not None, "" if family else "unknown_base"

    # ---- full SD/SDXL checkpoints
    if has("model.diffusion_model.input_blocks.0.0.weight"):
        in_ch = (_shape(tensors, "model.diffusion_model.input_blocks.0.0.weight") or [0, 4])[1]
        variant = "inpaint" if in_ch == 9 else ""
        if has("conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias") or contains(
            "conditioner.embedders.1."
        ):
            return "checkpoint", "sdxl", variant, True, ""
        if contains("conditioner.embedders.0.model."):
            return "checkpoint", "sdxl_refiner", variant, False, "refiner"
        if has("cond_stage_model.model.token_embedding.weight"):
            return "checkpoint", "sd2", variant, False, "sd2"
        if has("model.diffusion_model.output_blocks.11.0.skip_connection.weight") or contains("cond_stage_model."):
            return "checkpoint", "sd15", variant, True, ""
        return "checkpoint", None, variant, False, "unknown"

    kind = kind_hint if kind_hint in ("checkpoint", "denoiser") else "denoiser"
    note = "fp8_scaled" if scaled_fp8 else ""

    # ---- FLUX.1
    if has_prefixed("double_blocks.0.img_attn.norm.key_norm.scale") and not has_prefixed(
        "single_stream_modulation.lin.weight"
    ):
        img_in = _shape(tensors, "img_in.weight", "model.diffusion_model.img_in.weight") or [0, 64]
        if img_in[1] == 384:
            family, variant = "flux1_fill", "dev"
        elif img_in[1] == 128:
            return kind, "flux1_control", "", False, "flux_control"
        else:
            guided = has_prefixed("guidance_in.in_layer.weight")
            variant = "dev" if guided else "schnell"
            family = "flux1_kontext" if ("kontext" in lname and guided) else "flux1"
        return kind, family, variant, not scaled_fp8, note
    # ---- FLUX.2
    if has_prefixed("single_stream_modulation.lin.weight"):
        params = param_count(tensors)
        if params > 20e9:
            return kind, "flux2_dev", "", False, "too_big"
        size = "4b" if params < 6.5e9 else "9b"
        variant = size + ("-base" if "base" in lname else "")
        return kind, "flux2_klein", variant, not scaled_fp8, note
    # ---- Qwen-Image 2.1 / Qwen-Image (+ Edit)
    if has_prefixed("txt_in.text_norm.weight"):
        return kind, "qwen_image_21", "", not scaled_fp8, note
    if has_prefixed("txt_norm.weight") and has_prefixed("img_in.weight") and contains("img_mod"):
        family = "qwen_image_edit" if "edit" in lname else "qwen_image"
        return kind, family, "", not scaled_fp8, note
    # ---- Z-Image
    if has_prefixed("layers.0.adaLN_modulation.0.weight"):
        variant = "base" if ("base" in lname and "turbo" not in lname) else "turbo"
        return kind, "z_image", variant, not scaled_fp8, note
    # ---- Wan
    if has_prefixed("head.modulation"):
        pe = _shape(tensors, "patch_embedding.weight", "model.diffusion_model.patch_embedding.weight") or [0, 0]
        dim, in_ch = pe[0], pe[1]
        if contains("vace_blocks"):
            return kind, "wan_vace", "", False, "wan_vace"
        if dim == 1536:
            return kind, "wan_t2v", "1.3b", not scaled_fp8, note
        if dim == 3072 and in_ch == 48:
            return kind, "wan_ti2v", "5b", not scaled_fp8, note
        if dim == 5120:
            expert = "high_noise" if "high" in lname else "low_noise" if "low" in lname else ""
            if expert:  # Wan 2.2 A14B experts come in pairs - not loadable one at a time here
                return kind, "wan_a14b", expert, False, "wan_expert"
            return kind, "wan_t2v" if in_ch == 16 else "wan_i2v", "14b", (in_ch == 16) and not scaled_fp8, note
        return kind, "wan", "", False, "unknown"
    # ---- LTX-2
    if has_prefixed("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight"):
        return kind, "ltx2", "", False, "not_yet"

    # ---- components
    if has("encoder.conv_in.weight", "decoder.conv_out.weight") or has("first_stage_model.decoder.conv_out.weight"):
        return "vae", None, "", False, "component"
    if contains("encoder.block.0.layer.0.SelfAttention") or contains("text_model.encoder.layers") or contains(
        "model.layers.0.self_attn"
    ):
        return "text_encoder", None, "", False, "component"
    if kind_hint == "upscaler":
        return "upscaler", None, "", True, ""
    return kind_hint or "unknown", None, "", False, "unknown"


# ----------------------------------------------------------------------------- roots
@dataclass
class Root:
    path: Path
    source: str
    mapping: dict[str, str] | None  # subfolder -> kind; None = generic recursive scan


def _layout(p: Path) -> Root | None:
    """Recognises a program folder (or its models folder)."""
    name = p.name.lower()
    m = p / "models"
    if (m / "Stable-diffusion").is_dir():
        src = "Forge" if "forge" in str(p).lower() else "A1111"
        return Root(m, src, WEBUI_DIRS)
    if (m / "checkpoints").is_dir() or (m / "diffusion_models").is_dir() or (m / "unet").is_dir():
        src = "Fooocus" if "fooocus" in name or (p / "fooocus_version.py").exists() else "ComfyUI"
        return Root(m, src, COMFY_DIRS)
    if (p / "checkpoints").is_dir() or (p / "diffusion_models").is_dir():
        return Root(p, "ComfyUI", COMFY_DIRS)
    if (p / "Stable-diffusion").is_dir():
        return Root(p, "A1111", WEBUI_DIRS)
    if (p / "Stable-Diffusion").is_dir() and (p / "Lora").is_dir():
        return Root(p, "SwarmUI", SWARM_DIRS)
    return None


def _known_program_dirs(base: Path) -> list[Path]:
    out = []
    try:
        children = [c for c in base.iterdir() if c.is_dir()]
    except OSError:
        return out
    for c in children:
        n = c.name.lower()
        if n.startswith("comfyui"):
            out += [c, c / "ComfyUI"]
        elif "stable-diffusion-webui" in n or n.startswith("sd-webui") or n.startswith("webui_forge"):
            out += [c, c / "webui"]
        elif n.startswith("fooocus"):
            out += [c, c / "Fooocus"]
        elif n.startswith("stabilitymatrix"):
            out += [c / "Data" / "Models", c / "Models"]
        elif n == "swarmui":
            out += [c / "Models"]
    return out


def candidate_bases() -> list[Path]:
    home = Path.home()
    bases = [home, home / "Desktop", home / "Documents", home / "Downloads", home / "AI"]
    if sys.platform == "win32":
        for letter in string.ascii_uppercase[2:]:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                bases += [drive, drive / "AI", drive / "Programs", drive / "Tools", drive / "Apps"]
        appdata = os.environ.get("APPDATA")
        if appdata:  # ComfyUI Desktop keeps its base path in a config file
            try:
                cfg = json.loads((Path(appdata) / "ComfyUI" / "config.json").read_text(encoding="utf-8"))
                if cfg.get("basePath"):
                    bases.append(Path(cfg["basePath"]).parent)
            except (OSError, ValueError):
                pass
    return bases


def detect_roots() -> list[Root]:
    found: dict[str, Root] = {}
    for base in candidate_bases():
        for p in _known_program_dirs(base):
            root = _layout(p)
            if root:
                found.setdefault(os.path.normcase(str(root.path.resolve())), root)
    return list(found.values())


def roots_for(folders: list[str], auto: bool) -> list[Root]:
    roots: dict[str, Root] = {}
    for f in folders:
        p = Path(f).expanduser()
        if not p.is_dir():
            continue
        root = _layout(p) or Root(p, "Folder", None)
        roots.setdefault(os.path.normcase(str(root.path.resolve())), root)
    if auto:
        for root in detect_roots():
            roots.setdefault(os.path.normcase(str(root.path.resolve())), root)
    return list(roots.values())


# ----------------------------------------------------------------------------- scanning
def _iter_files(folder: Path, max_depth: int):
    base_depth = len(folder.parts)
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=True):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", ".git")]
        for fn in filenames:
            if fn.lower().endswith(MODEL_EXTS):
                yield Path(dirpath) / fn


def _file_kind_hint(path: Path, root: Root) -> str | None:
    if root.mapping is None:
        return None
    try:
        rel = path.relative_to(root.path)
    except ValueError:
        return None
    return root.mapping.get(rel.parts[0]) if rel.parts else None


def inspect_file(path: Path, source: str, kind_hint: str | None) -> LocalModel:
    st = path.stat()
    ext = path.suffix.lower()
    fmt = {".safetensors": "safetensors", ".sft": "safetensors", ".gguf": "gguf"}.get(ext, ext.lstrip("."))
    base = dict(
        id=model_id(str(path)),
        path=str(path),
        name=path.stem,
        fmt=fmt,
        size=st.st_size,
        mtime=st.st_mtime,
        source=source,
    )
    if fmt in ("safetensors", "gguf"):
        try:
            tensors, _meta = read_safetensors_header(str(path)) if fmt == "safetensors" else read_gguf_tensors(str(path))
        except Exception as e:
            return LocalModel(kind=kind_hint or "unknown", family=None, variant="", dtype="?", supported=False,
                              note=f"unreadable: {e}", **base)
        kind, family, variant, supported, note = classify(tensors, path.name, kind_hint)
        dtype = summarize_dtype(tensors, fmt)
    else:  # pickled formats: only trusted for upscalers / eraser (loaded with a restricted unpickler)
        kind = kind_hint or "unknown"
        family, variant, dtype = None, "", "?"
        supported = kind == "upscaler"
        note = "" if supported else "pickle"
    base_repo = None
    if kind in ("denoiser", "checkpoint") and family and family not in SINGLE_FILE_CHECKPOINT_FAMILIES:
        base_repo = BASE_REPOS.get((family, variant))
        if base_repo is None:
            supported = False
            note = note or "no_base"
    if fmt == "gguf" and kind == "checkpoint":
        kind = "denoiser"
    return LocalModel(kind=kind, family=family, variant=variant, dtype=dtype, supported=supported, note=note,
                      base_repo=base_repo, **base)


class LocalLibrary:
    """Scans and caches local model files."""

    def __init__(self, index_file: Path):
        self.index_file = index_file
        self._lock = threading.Lock()
        self.models: dict[str, LocalModel] = {}
        self.roots: list[Root] = []
        self._load_index()

    def _load_index(self) -> None:
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8"))
            self.models = {m["id"]: LocalModel(**m) for m in data.get("models", [])}
        except (OSError, ValueError, TypeError):
            self.models = {}

    def _save_index(self) -> None:
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": [m.public() for m in self.models.values()]}
        tmp = self.index_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.index_file)

    def scan(self, folders: list[str], auto: bool, progress=None) -> list[LocalModel]:
        roots = roots_for(folders, auto)
        previous = dict(self.models)
        found: dict[str, LocalModel] = {}
        files: list[tuple[Path, Root]] = []
        for root in roots:
            if root.mapping is None:
                files += [(f, root) for f in _iter_files(root.path, 5)]
            else:
                for sub in root.mapping:
                    d = root.path / sub
                    if d.is_dir():
                        files += [(f, root) for f in _iter_files(d, 4)]
        for i, (f, root) in enumerate(files):
            if progress:
                progress(i, len(files))
            mid = model_id(str(f))
            try:
                st = f.stat()
            except OSError:
                continue
            old = previous.get(mid)
            if old and old.size == st.st_size and abs(old.mtime - st.st_mtime) < 1:
                found[mid] = old
                continue
            try:
                found[mid] = inspect_file(f, root.source, _file_kind_hint(f, root))
            except OSError as e:
                log.info("skip %s: %s", f, e)
        with self._lock:
            self.models = found
            self.roots = roots
            self._save_index()
        return list(found.values())

    def get(self, mid: str) -> LocalModel | None:
        return self.models.get(mid)

    def list(self) -> list[LocalModel]:
        return sorted(self.models.values(), key=lambda m: (m.kind, m.family or "~", m.name.lower()))

    def roots_public(self) -> list[dict]:
        return [{"path": str(r.path), "source": r.source} for r in self.roots]
