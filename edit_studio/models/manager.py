"""Loads diffusers pipelines onto the GPUs according to a memory plan and keeps them cached."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import logging
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import SettingsStore
from ..hardware import GB, Hardware
from .catalog import CATALOG_BY_ID, FAMILIES, Family, ModelSpec, family_from_index
from .hub import HubClient
from .local import LocalLibrary, LocalModel
from .planner import CPU, SHARDED, Component, Options, Plan, plan_options, sequential_plan, size_of

log = logging.getLogger(__name__)

# After an out-of-memory error the model is planned again with this much more room for activations;
# past the last step it runs in the slow sequential mode.
OOM_HEADROOM = (1.0, 1.6, 2.5, 4.0)

# Defaults for models that don't come from the catalog (local files / other HF repos).
FAMILY_DEFAULTS: dict[tuple[str, str], dict[str, Any]] = {
    ("sd15", ""): {"steps": 30, "guidance": 7.0, "width": 512, "height": 512, "strength": 0.6},
    ("sdxl", ""): {"steps": 30, "guidance": 6.0, "width": 1024, "height": 1024, "strength": 0.6},
    ("flux1", "dev"): {"steps": 28, "guidance": 3.5, "width": 1024, "height": 1024, "strength": 0.8},
    ("flux1", "schnell"): {"steps": 4, "guidance": 0.0, "width": 1024, "height": 1024, "strength": 0.8},
    ("flux1_kontext", ""): {"steps": 28, "guidance": 2.5},
    ("flux1_fill", ""): {"steps": 40, "guidance": 30.0, "strength": 1.0},
    ("flux2_klein", ""): {"steps": 4, "guidance": 1.0, "width": 1024, "height": 1024, "strength": 0.8},
    ("flux2_klein", "base"): {"steps": 50, "guidance": 4.0, "width": 1024, "height": 1024, "strength": 0.8},
    ("z_image", "turbo"): {"steps": 9, "guidance": 0.0, "width": 1024, "height": 1024, "strength": 0.6},
    ("z_image", "base"): {"steps": 40, "guidance": 4.0, "width": 1024, "height": 1024, "strength": 0.6},
    ("qwen_image", ""): {"steps": 40, "guidance": 4.0, "width": 1328, "height": 1328, "strength": 0.6},
    ("qwen_image_edit", ""): {"steps": 40, "guidance": 4.0},
    ("qwen_image_21", ""): {"steps": 40, "guidance": 1.0, "width": 1024, "height": 1024},
    ("wan_t2v", "1.3b"): {"steps": 30, "guidance": 5.0, "width": 832, "height": 480, "fps": 16, "frames": 81, "strength": 0.7},
    ("wan_t2v", "14b"): {"steps": 40, "guidance": 5.0, "width": 832, "height": 480, "fps": 16, "frames": 81, "strength": 0.7},
    ("wan_ti2v", "5b"): {"steps": 40, "guidance": 5.0, "width": 1280, "height": 704, "fps": 24, "frames": 121},
    ("wan_i2v", "14b"): {"steps": 40, "guidance": 5.0, "width": 832, "height": 480, "fps": 16, "frames": 81},
    ("wan_a14b_t2v", ""): {"steps": 40, "guidance": 4.0, "guidance_2": 3.0, "width": 832, "height": 480, "fps": 16, "frames": 81},
    ("wan_a14b_i2v", ""): {"steps": 40, "guidance": 3.5, "width": 832, "height": 480, "fps": 16, "frames": 81},
}


def family_defaults(family: str, variant: str) -> dict[str, Any]:
    for key in ((family, variant), (family, variant.split("-")[-1] if variant else ""), (family, "")):
        if key in FAMILY_DEFAULTS:
            return dict(FAMILY_DEFAULTS[key])
    return {"steps": 30, "guidance": 4.0, "width": 1024, "height": 1024}


class ModelError(Exception):
    """Errors with an i18n key so the UI can explain what to do."""

    def __init__(self, key: str, message: str = "", **info):
        super().__init__(message or key)
        self.key = key
        self.info = info


@dataclass
class ModelSource:
    id: str
    name: str
    family: Family
    kind: str  # pipeline | checkpoint | denoiser
    repo: str | None = None
    path: str | None = None
    variant: str | None = None  # HF weight variant (fp16)
    model_variant: str = ""  # dev / schnell / turbo / 4b ...
    spec: ModelSpec | None = None
    local: LocalModel | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    pipeline_config: dict[str, Any] = field(default_factory=dict)

    @property
    def skip_weights(self) -> tuple[str, ...]:
        return self.family.denoisers[:1] if self.kind == "denoiser" else ()

    def tasks(self) -> tuple[str, ...]:
        if self.spec:
            return self.spec.supported_tasks()
        if self.local and self.local.variant == "inpaint":
            return ("inpaint",)
        return tuple(self.family.pipelines)


@dataclass
class LoadedModel:
    key: str
    source_id: str
    plan: Plan
    pipe: Any
    slots: list[int]
    loras: tuple = ()
    last_used: float = field(default_factory=time.time)


def _torch():
    import torch

    return torch


def _is_module_class(library: str, class_name: str) -> bool:
    torch = _torch()
    try:
        cls = getattr(importlib.import_module(library), class_name)
    except (ImportError, AttributeError):
        return False
    return isinstance(cls, type) and issubclass(cls, torch.nn.Module)


def read_model_index(folder: Path) -> dict:
    return json.loads((folder / "model_index.json").read_text(encoding="utf-8"))


def module_components(index: dict) -> dict[str, tuple[str, str]]:
    out = {}
    for name, value in index.items():
        if name.startswith("_") or not isinstance(value, list) or len(value) != 2 or value[0] is None:
            continue
        if _is_module_class(value[0], value[1]):
            out[name] = (value[0], value[1])
    return out


def folder_weight_bytes(folder: Path, bytes_per_param: int = 2) -> int:
    """Size of a component's weights at the compute dtype, from safetensors headers."""
    from .local import read_safetensors_header

    files = sorted(folder.glob("*.safetensors"))
    variant_free = [f for f in files if f.name.count(".") <= 2 or "-of-" in f.name]
    files = variant_free or files
    total = 0
    for f in files:
        try:
            header, _ = read_safetensors_header(str(f))
        except (OSError, ValueError):
            total += f.stat().st_size
            continue
        for info in header.values():
            n = 1
            for s in info.get("shape", []):
                n *= int(s)
            total += n * bytes_per_param
    if not files:
        total = sum(f.stat().st_size for f in folder.glob("*.bin"))
    return total


def _role(fam: Family, name: str) -> str:
    if name in fam.denoisers:
        return "denoiser"
    if name in fam.text_encoders:
        return "text_encoder"
    if name == "vae":
        return "vae"
    return "other"


def _pin_execution_device(pipe, device: str, plan: Plan) -> None:
    """Split / shard pipelines: every intermediate tensor starts on the main GPU.

    The device map also tells diffusers the parts are placed on purpose: otherwise loading a LoRA would take
    our hooks for CPU offloading, remove them and switch the whole pipeline to sequential offload."""
    torch = _torch()
    cls = pipe.__class__
    base = cls.__bases__[0] if getattr(cls, "_es_pinned", False) else cls
    pinned = type(
        base.__name__,
        (base,),
        {"_execution_device": property(lambda self: torch.device(self._es_device)), "_es_pinned": True},
    )
    pipe.__class__ = pinned
    pipe._es_device = device
    pipe.hf_device_map = {"": device, **{k: str(v) for k, v in plan.placement.items()}}


def _is_quantized(module) -> bool:
    return bool(getattr(module, "is_quantized", False) or getattr(module, "hf_quantizer", None) is not None)


def _is_gguf(module) -> bool:
    return type(getattr(module, "hf_quantizer", None)).__name__.startswith("GGUF")


class StreamLoadError(Exception):
    """The streaming loader could not build this model (unusual layout); the classic loader takes over."""


def _shard_units(model) -> list[tuple[str, int, str | None]]:
    """Top-level pieces of a model in registration order, with layer stacks (ModuleList) split into blocks:
    (name, bytes, stack name or None)."""
    torch = _torch()

    def nbytes(m) -> int:
        return sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))

    units = [(n, p.numel() * p.element_size(), None) for n, p in model.named_parameters(recurse=False)]
    for n, child in model.named_children():
        if isinstance(child, torch.nn.ModuleList) and len(child) > 1:
            units += [(f"{n}.{i}", nbytes(block), n) for i, block in enumerate(child)]
        else:
            units.append((n, nbytes(child), None))
    units += [(n, b.numel() * b.element_size(), None) for n, b in model.named_buffers(recurse=False)]
    return units


def shard_split(units: list[tuple[str, int, str | None]], room: list[int]) -> dict[str, int]:
    """Which of two GPUs (0 or 1) gets each piece: one contiguous run per GPU in registration order (the layout
    diffusers tests with accelerate's device maps), balanced so both fill the same share of their room.

    The hand-over happens inside the main layer stack (with any stacks right next to it, like FLUX's double and
    single blocks): smaller stacks such as Z-Image's refiners feed root code that mixes their output with tensors
    of the first GPU, so they must not be cut."""
    stacks: dict[str, int] = {}
    for _, size, stack in units:
        if stack:
            stacks[stack] = stacks.get(stack, 0) + size
    if stacks:
        main = max(stacks, key=stacks.get)
        lo = next(i for i, u in enumerate(units) if u[2] == main)
        hi = max(i for i, u in enumerate(units) if u[2] == main) + 1
        while lo > 0 and units[lo - 1][2]:
            lo -= 1
        while hi < len(units) and units[hi][2]:
            hi += 1
        cuts = range(lo, hi + 1)
    else:
        cuts = range(len(units) + 1)
    prefix = [0]
    for _, size, _ in units:
        prefix.append(prefix[-1] + size)
    total = prefix[-1]
    k = min(cuts, key=lambda k: max(prefix[k] / room[0], (total - prefix[k]) / room[1]))
    if prefix[k] > room[0] * 1.05 or total - prefix[k] > room[1] * 1.05:
        raise _torch().OutOfMemoryError(f"layers need {total / GB:.1f} GB, room is {sum(room) / GB:.1f} GB")
    return {name: 0 if i < k else 1 for i, (name, _, _) in enumerate(units)}


def _device_for_key(key: str, device_map: dict[str, str]) -> str:
    name = key
    while name and name not in device_map:
        name = name.rpartition(".")[0]
    return device_map.get(name, device_map.get("", "cpu"))


def _weight_files(folder: Path) -> list[Path]:
    for index in [folder / "diffusion_pytorch_model.safetensors.index.json", *sorted(folder.glob("*.safetensors.index*.json"))]:
        if index.is_file():
            names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
            return [folder / n for n in names]
    single = folder / "diffusion_pytorch_model.safetensors"
    files = [single] if single.is_file() else sorted(folder.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors weights in {folder}")
    return files


def _weight_stream(folder: Path, local: str | None, cls, model, config) -> Iterator[tuple[str, Any]]:
    """(name, tensor) pairs of a denoiser: one tensor at a time from a diffusers folder, or from a single file
    (ComfyUI / A1111 layout) converted to diffusers names the way `from_single_file` does it."""
    if local is None:
        from safetensors import safe_open

        for f in _weight_files(folder):
            with safe_open(str(f), framework="pt", device="cpu") as fh:
                for key in fh.keys():
                    yield key, fh.get_tensor(key)
        return
    from diffusers.loaders.single_file_model import (
        SINGLE_FILE_LOADABLE_CLASSES,
        _get_single_file_loadable_mapping_class,
        _should_convert_state_dict_to_diffusers,
    )
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint

    checkpoint = load_single_file_checkpoint(local, local_files_only=True)
    if _should_convert_state_dict_to_diffusers(model.state_dict(), checkpoint):
        convert = SINGLE_FILE_LOADABLE_CLASSES[_get_single_file_loadable_mapping_class(cls)]["checkpoint_mapping_fn"]
        checkpoint = convert(config=config, checkpoint=checkpoint)
    for key in list(checkpoint):
        yield key, checkpoint.pop(key)  # frees system RAM as the weights move to the GPUs


class ModelManager:
    def __init__(
        self,
        hw: Hardware,
        settings: SettingsStore,
        hub: HubClient,
        local: LocalLibrary,
        busy_slots: Callable[[], set[int]] = lambda: set(),
    ):
        self.hw = hw
        self.settings = settings
        self.hub = hub
        self.local = local
        self.busy_slots = busy_slots
        self.lock = threading.RLock()
        self.loaded: dict[int, LoadedModel] = {}
        self._oom_level: dict[str, int] = {}  # plan key -> how many times it ran out of memory
        self._simple_load: set[str] = set()  # models the streaming loader could not handle

    # ------------------------------------------------------------------ resolve
    def resolve(self, model_id: str) -> ModelSource:
        if model_id in CATALOG_BY_ID:
            spec = CATALOG_BY_ID[model_id]
            return ModelSource(
                id=spec.id,
                name=spec.name,
                family=spec.fam,
                kind="pipeline",
                repo=spec.repo,
                variant=spec.variant,
                spec=spec,
                defaults=dict(spec.defaults),
                pipeline_config=dict(spec.pipeline_config),
            )
        if model_id.startswith("hf:"):
            repo = model_id[3:]
            snap = self.hub.snapshot(repo)
            if snap is None or not (snap / "model_index.json").is_file():
                raise ModelError("model_missing", repo=repo)
            found = family_from_index(read_model_index(snap), repo)
            if found is None:
                raise ModelError("unsupported_model", repo=repo)
            fam_id, variant = found
            return ModelSource(
                id=model_id,
                name=repo.split("/")[-1],
                family=FAMILIES[fam_id],
                kind="pipeline",
                repo=repo,
                model_variant=variant,
                defaults=family_defaults(fam_id, variant),
            )
        if model_id.startswith("local:"):
            lm = self.local.get(model_id)
            if lm is None:
                raise ModelError("model_missing", path=model_id)
            if not lm.supported or not lm.family or lm.family not in FAMILIES:
                raise ModelError("unsupported_model", path=lm.path, note=lm.note)
            kind = "checkpoint" if lm.kind == "checkpoint" and lm.family in ("sd15", "sdxl") else "denoiser"
            pc = {"flow_shift": 3.0} if lm.family == "wan_t2v" else {}
            return ModelSource(
                id=lm.id,
                name=lm.name,
                family=FAMILIES[lm.family],
                kind=kind,
                repo=self.base_repo(lm),
                path=lm.path,
                model_variant=lm.variant,
                local=lm,
                defaults=family_defaults(lm.family, lm.variant),
                pipeline_config=pc,
            )
        raise ModelError("model_missing", model=model_id)

    def base_repo(self, lm: LocalModel) -> str | None:
        """Repo with the parts a single-file model lacks: the version its name says, else one already installed,
        else the default (then only its small parts are downloaded)."""
        from .local import base_candidates, version_base

        named = version_base(lm)
        if named:
            return named
        skip = FAMILIES[lm.family].denoisers[:1] if lm.family in FAMILIES else ()
        cands = base_candidates(lm)
        return next((r for r in cands if self.hub.is_installed(r, skip)), cands[0] if cands else None)

    # ------------------------------------------------------------------ install state
    def missing_downloads(self, src: ModelSource) -> list[dict]:
        """Hugging Face downloads needed before `src` can load."""
        if src.kind == "checkpoint" or not src.repo:
            return []
        if self.hub.is_installed(src.repo, src.skip_weights):
            return []
        return [{"repo": src.repo, "variant": src.variant, "skip_weights": list(src.skip_weights)}]

    # ------------------------------------------------------------------ planning
    _nf4_ok: bool | None = None

    def can_nf4(self) -> bool:
        """True when bitsandbytes 4-bit really runs on this GPU (a wheel without the right CUDA build imports fine)."""
        if self.hw.backend != "cuda":
            return False
        if ModelManager._nf4_ok is None:
            try:
                torch = _torch()
                bnb = importlib.import_module("bitsandbytes")
                layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").to("cuda:0")
                layer(torch.randn(1, 64, device="cuda:0", dtype=torch.bfloat16))
                del layer
                ModelManager._nf4_ok = True
            except Exception as e:
                log.warning("4-bit compression unavailable: %s", e)
                ModelManager._nf4_ok = False
        return ModelManager._nf4_ok

    def components(self, src: ModelSource) -> list[Component]:
        fam = src.family
        comps: list[Component] = []
        snap = self.hub.snapshot(src.repo) if src.repo else None
        if src.kind == "checkpoint":
            from .local import param_count, read_safetensors_header

            total = src.local.size if src.local else 7 * GB
            try:
                total = param_count(read_safetensors_header(src.path)[0]) * 2
            except Exception:
                pass
            unet = int(total * 0.75)
            # one file loaded as a whole pipeline: no per-part compression or spreading
            return [
                Component(fam.denoisers[0], unet, "denoiser", fixed=True, shardable=False),
                Component("text_encoder", int(total * 0.2), "text_encoder", fixed=True),
                Component("vae", int(total * 0.05), "vae", fixed=True),
            ]
        shardable = lambda name: name != "unet" and src.id not in self._simple_load  # noqa: E731
        if snap is not None and (snap / "model_index.json").is_file():
            for name in module_components(read_model_index(snap)):
                if src.kind == "denoiser" and name in fam.denoisers[:1]:
                    continue
                size = folder_weight_bytes(snap / name, 4 if (name == "vae" and fam.vae_fp32) else 2)
                comps.append(Component(name, size, _role(fam, name), shardable=shardable(name)))
        elif src.spec:
            comps = [Component(n, int(gb * GB), _role(fam, n), shardable=shardable(n)) for n, gb in src.spec.components_gb.items()]
        if src.kind == "denoiser" and src.local:
            from .local import param_count, read_safetensors_header

            size = src.local.size
            if src.local.fmt == "safetensors":
                try:
                    size = param_count(read_safetensors_header(src.path)[0]) * 2
                except Exception:
                    pass
            gguf = src.local.fmt == "gguf"  # already compressed: its file size is what it needs
            comps.append(Component(fam.denoisers[0], size, "denoiser", fixed=gguf, shardable=not gguf and shardable(fam.denoisers[0])))
        return comps

    def _gpu_vram(self, gpu_ids: list[int]) -> dict[int, int]:
        return {i: self.hw.usable_vram(i) for i in gpu_ids}

    def options(self, src: ModelSource, gpu_ids: list[int]) -> Options:
        """The best one-card and two-card plans for these GPUs, and which one a job should use."""
        s = self.settings.settings
        level = min(self._oom_level.get(self._override_key(src, gpu_ids), 0), len(OOM_HEADROOM) - 1)
        return plan_options(
            self.components(src),
            self._gpu_vram(gpu_ids),
            self.hw.ram_total(),
            int(src.family.activation_gb * GB * OOM_HEADROOM[level]),
            mode_pref=s.memory_mode,
            compression=s.compression,
            can_nf4=self.can_nf4(),
            backend=self.hw.backend,
            kind=src.family.kind,
        )

    def plan(self, src: ModelSource, gpu_ids: list[int]) -> Plan:
        """The plan for a job that got the GPUs `gpu_ids`."""
        if self._oom_level.get(self._override_key(src, gpu_ids), 0) >= len(OOM_HEADROOM):
            return sequential_plan(gpu_ids[0], self.components(src), self.can_nf4())
        return self.options(src, gpu_ids).for_gpus(len(gpu_ids))

    def _override_key(self, src: ModelSource, gpu_ids: list[int]) -> str:
        s = self.settings.settings
        return f"{src.id}|{','.join(map(str, gpu_ids))}|{s.memory_mode}|{s.compression}"

    def describe(self, src: ModelSource, gpu_ids: list[int]) -> dict:
        """Plan summary for the UI: what a job gets with these GPUs free, how many cards it needs or likes,
        and (for models that like two) what it does when only one card is free."""
        if self._oom_level.get(self._override_key(src, gpu_ids), 0) >= len(OOM_HEADROOM):
            return {**self.plan(src, gpu_ids).public(), "gpus": [1, 1]}
        opts = self.options(src, gpu_ids)
        out = {**opts.for_gpus(len(gpu_ids)).public(), "gpus": list(opts.gpu_range)}
        if opts.choice == "preferred":
            out["alone"] = opts.single.public()
        return out

    def gpu_range(self, model_id: str, enabled: list[int]) -> tuple[int, int]:
        """(fewest, most) graphics cards a job with this model should get."""
        if len(enabled) < 2:
            return (1, 1)
        try:
            src = self.resolve(model_id)
        except ModelError:
            return (1, 1)
        if self._oom_level.get(self._override_key(src, enabled), 0) >= len(OOM_HEADROOM):
            return (1, 1)
        return self.options(src, enabled).gpu_range

    # ------------------------------------------------------------------ cache
    def loaded_on(self, slot: int) -> LoadedModel | None:
        with self.lock:
            return self.loaded.get(slot)

    def loaded_source(self, slot: int) -> str | None:
        lm = self.loaded_on(slot)
        return lm.source_id if lm else None

    def _drop(self, lm: LoadedModel) -> None:
        with self.lock:
            for s in [s for s, v in self.loaded.items() if v is lm]:
                del self.loaded[s]
        pipe = lm.pipe
        lm.pipe = None
        try:
            if pipe is not None and hasattr(pipe, "remove_all_hooks"):
                pipe.remove_all_hooks()
        except Exception:
            pass
        del pipe
        self._collect()

    def _collect(self) -> None:
        gc.collect()
        if self.hw.backend == "cuda" and _torch().cuda.is_available():
            torch = _torch()
            for g in self.hw.gpus:
                with torch.cuda.device(g.index):
                    torch.cuda.empty_cache()
        elif self.hw.backend == "mps":
            try:
                _torch().mps.empty_cache()
            except Exception:
                pass

    def unload_slots(self, slots: list[int]) -> None:
        with self.lock:
            victims = {id(self.loaded[s]): self.loaded[s] for s in slots if s in self.loaded}
        for lm in victims.values():
            self._drop(lm)

    def unload_all(self, only_idle: bool = True) -> int:
        busy = self.busy_slots() if only_idle else set()
        with self.lock:
            victims = {id(lm): lm for s, lm in self.loaded.items() if not (set(lm.slots) & busy)}
        for lm in victims.values():
            self._drop(lm)
        return len(victims)

    def unload_model(self, source_id: str) -> None:
        busy = self.busy_slots()
        with self.lock:
            victims = {id(lm): lm for lm in self.loaded.values() if lm.source_id == source_id and not (set(lm.slots) & busy)}
        for lm in victims.values():
            self._drop(lm)

    def status(self) -> list[dict]:
        with self.lock:
            seen = {}
            for lm in self.loaded.values():
                seen.setdefault(id(lm), lm)
            return [
                {"model": lm.source_id, "slots": lm.slots, "plan": lm.plan.public(), "loras": [n for n, _ in lm.loras]}
                for lm in seen.values()
            ]

    # ------------------------------------------------------------------ loading
    def dtype(self, src: ModelSource):
        torch = _torch()
        if self.hw.backend == "cpu":
            return torch.float32
        if self.hw.backend == "mps":
            return torch.float16 if src.family.id in ("sd15", "sdxl") else torch.bfloat16
        return torch.bfloat16 if self.hw.supports_bf16(0) else torch.float16

    def acquire(
        self,
        src: ModelSource,
        task: str,
        slots: list[int],
        loras: list[tuple[str, float]] | None = None,
        status: Callable[[str], None] = lambda s: None,
    ):
        """Returns a pipeline for `task`, loading or re-using the cached one on `slots`."""
        if task not in src.tasks():
            raise ModelError("task_unsupported", task=task, model=src.name)
        plan = self.plan(src, slots)
        key = f"{src.id}|{plan.key}"
        with self.lock:
            lm = self.loaded.get(slots[0])
        if lm is not None and lm.key != key:
            self.unload_slots(list(set(slots) | set(lm.slots)))
            lm = None
        if lm is None:
            self.unload_slots(slots)
            self._make_room_in_ram(src, plan, slots)
            lm = self._load_with_fallback(src, task, plan, slots, status)
        pipe = self._task_pipeline(lm, src, task)
        self._apply_loras(lm, loras or [], status)
        lm.last_used = time.time()
        return pipe, lm.plan

    def _make_room_in_ram(self, src: ModelSource, plan: Plan, slots: list[int]) -> None:
        comps = self.components(src)
        if plan.mode in ("offload", "sequential", "cpu"):
            need = sum(size_of(c, plan.quant) for c in comps)
        else:  # text encoders kept on the CPU
            need = sum(c.bytes for c in comps if plan.placement.get(c.name) == CPU)
        if need and self.hw.ram_available() < need * 1.1:
            busy = self.busy_slots() - set(slots)
            with self.lock:
                idle = {id(lm): lm for s, lm in self.loaded.items() if not (set(lm.slots) & busy)}
            for lm in idle.values():
                self._drop(lm)

    def _load_with_fallback(self, src, task, plan: Plan, slots, status) -> LoadedModel:
        torch = _torch()
        attempt = plan
        while True:
            try:
                pipe = self._load(src, attempt, status)
            except torch.OutOfMemoryError:
                log.warning("Out of memory loading %s with %s", src.id, attempt.key)
                self._collect()
                attempt = self._safer_plan(src, slots)
                if attempt is None:
                    raise ModelError("oom")
                status("fallback:" + attempt.mode)
                continue
            except StreamLoadError as e:
                log.warning("Streaming load of %s failed (%s); loading it the classic way", src.id, e)
                self._simple_load.add(src.id)
                self._collect()
                attempt = self.plan(src, slots)
                status("fallback:" + attempt.mode)
                continue
            lm_slots = attempt.devices if attempt.mode in ("split", "shard") else slots[:1]
            lm = LoadedModel(f"{src.id}|{attempt.key}", src.id, attempt, pipe, lm_slots)
            with self.lock:
                for s in lm.slots:
                    self.loaded[s] = lm
            return lm

    def _safer_plan(self, src: ModelSource, slots: list[int]) -> Plan | None:
        """After running out of memory: plan again with more headroom. None when nothing safer is left."""
        key = self._override_key(src, slots)
        current = self.plan(src, slots).key
        while self._oom_level.get(key, 0) < len(OOM_HEADROOM):
            self._oom_level[key] = self._oom_level.get(key, 0) + 1
            nxt = self.plan(src, slots)
            if nxt.key != current:
                return nxt
        return None

    def report_oom(self, src: ModelSource, slots: list[int]) -> bool:
        """Called when generation runs out of memory: remember a safer plan. Returns True if one exists."""
        lm = self.loaded_on(slots[0])
        if lm:
            self._drop(lm)
        self._collect()
        return self._safer_plan(src, slots) is not None

    # -- component helpers
    def _quant_config(self, library: str, kind: str):
        torch = _torch()
        if kind != "nf4":
            return None
        if library == "transformers":
            from transformers import BitsAndBytesConfig
        else:
            from diffusers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def _device_of(self, plan: Plan, name: str) -> str | None:
        """Where a component is loaded directly (None = system RAM, moved later by offload hooks)."""
        if plan.mode in ("gpu", "split", "shard"):
            g = plan.placement.get(name, plan.devices[0]) if plan.mode != "gpu" else plan.devices[0]
            if g == SHARDED:
                return None  # several devices: the streaming loader places it
            return "cpu" if g == CPU else self.hw.device_for(g)
        if plan.quant.get(name) == "nf4":  # bitsandbytes quantizes while moving to the GPU
            return self.hw.device_for(plan.devices[0])
        return None

    def _load_component(self, folder: Path, name: str, library: str, class_name: str, dtype, quant: str | None, device: str | None):
        cls = getattr(importlib.import_module(library), class_name)
        kwargs: dict[str, Any] = {}
        kwargs["dtype"] = dtype
        qc = self._quant_config(library, quant or "")
        if qc is not None:
            kwargs["quantization_config"] = qc
        if device is not None and device != "cpu":
            kwargs["device_map"] = {"": device}
        return cls.from_pretrained(str(folder), subfolder=name, **kwargs)

    def _load(self, src: ModelSource, plan: Plan, status) -> Any:
        import diffusers

        torch = _torch()
        dtype = self.dtype(src)
        fam = src.family
        token = self.settings.hf_token()
        status("loading")

        if src.kind == "checkpoint":
            base_task = "inpaint" if src.local and src.local.variant == "inpaint" else "txt2img"
            cls = getattr(diffusers, fam.pipelines[base_task])
            pipe = cls.from_single_file(src.path, dtype=dtype, token=token, cache_dir=self.settings.hf_cache_dir())
        else:
            snap = self.hub.snapshot(src.repo)
            if snap is None:
                raise ModelError("model_missing", repo=src.repo)
            index = read_model_index(snap)
            comps = module_components(index)
            overrides: dict[str, Any] = {}
            for name in fam.denoisers:
                if name in comps:
                    den = self._load_denoiser(src, snap, name, comps[name][1], plan, dtype, status)
                    if den is not None:
                        overrides[name] = den
            for name, (lib, cls_name) in comps.items():
                if name in overrides:
                    continue
                comp_dtype = torch.float32 if (name == "vae" and fam.vae_fp32) else dtype
                quant = plan.quant.get(name) if plan.quant.get(name) == "nf4" else None
                device = self._device_of(plan, name)
                if quant or (device and device != "cpu") or comp_dtype != dtype:
                    status("loading:" + name)
                    overrides[name] = self._load_component(snap, name, lib, cls_name, comp_dtype, quant, device)
            status("loading")
            # The concrete class, not DiffusionPipeline: through the generic loader, parts that are optional in the
            # pipeline's __init__ (Wan's transformers) don't count as passed in and get loaded again from disk.
            pipe_cls = getattr(diffusers, str(index.get("_class_name", "")), None) or diffusers.DiffusionPipeline
            pipe = pipe_cls.from_pretrained(str(snap), dtype=dtype, **overrides)

        # fp8 storage for parts loaded the classic way: weights kept in 8 bits, each layer upcast just before it runs
        for name, q in plan.quant.items():
            module = getattr(pipe, name, None)
            if q == "fp8" and module is not None and not getattr(module, "_es_streamed", False):
                if hasattr(module, "enable_layerwise_casting"):
                    module.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)

        if src.pipeline_config.get("flow_shift") and hasattr(pipe, "scheduler"):
            sched = pipe.scheduler
            pipe.scheduler = type(sched).from_config(sched.config, flow_shift=src.pipeline_config["flow_shift"])

        self._place(pipe, plan)
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling") and (fam.kind == "video" or plan.mode == "shard"):
            pipe.vae.enable_tiling()  # bounded decode memory: videos, and image models packed tight over two GPUs
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._collect()  # hand back the scratch blocks of per-tensor casting
        return pipe

    def _load_denoiser(self, src: ModelSource, snap: Path, name: str, cls_name: str, plan: Plan, dtype, status):
        """Loads one denoiser, or returns None to let the pipeline load it from the folder as usual."""
        import diffusers

        torch = _torch()
        cls = getattr(diffusers, cls_name)
        local = src.path if (src.kind == "denoiser" and name == src.family.denoisers[0]) else None
        q = plan.quant.get(name)
        dev = self._device_of(plan, name)
        on_gpu = {"device": dev} if dev and dev != "cpu" else {}
        status("loading:" + name)
        if local and src.local and src.local.fmt == "gguf":
            gguf = diffusers.GGUFQuantizationConfig(compute_dtype=dtype)
            return cls.from_single_file(local, config=str(snap), subfolder=name, dtype=dtype, quantization_config=gguf, **on_gpu)
        if q == "nf4":
            if local:
                qc = self._quant_config("diffusers", "nf4")
                return cls.from_single_file(local, config=str(snap), subfolder=name, dtype=dtype, quantization_config=qc, **on_gpu)
            return self._load_component(snap, name, "diffusers", cls_name, dtype, "nf4", dev)
        if (q == "fp8" or local or plan.placement.get(name) == SHARDED) and src.id not in self._simple_load:
            try:
                return self._stream_denoiser(cls, snap, name, local, plan, dtype)
            except torch.OutOfMemoryError:
                raise
            except (ValueError, KeyError, RuntimeError, AttributeError, TypeError, ImportError, NotImplementedError, OSError) as e:
                log.exception("Streaming load failed for %s", src.id)
                raise StreamLoadError(f"{type(e).__name__}: {e}") from e
        if local:  # classic: whole file into system RAM, placed (and maybe cast to fp8) afterwards
            return cls.from_single_file(local, config=str(snap), subfolder=name, dtype=dtype)
        if q != "fp8" and dev and dev != "cpu":
            return self._load_component(snap, name, "diffusers", cls_name, dtype, None, dev)
        return None

    def _stream_denoiser(self, cls, snap: Path, name: str, local: str | None, plan: Plan, dtype):
        """Builds the denoiser empty (meta device) and streams the weights in one tensor at a time, each cast where it
        lands: fp8 storage never needs a full-precision copy in RAM, and shard plans spread the layers over GPUs."""
        import accelerate
        from accelerate.utils import set_module_tensor_to_device

        torch = _torch()
        config = cls.load_config(str(snap), subfolder=name)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)  # like from_pretrained: buffers made in __init__ get the compute dtype
        try:
            with accelerate.init_empty_weights():
                model = cls.from_config(config)
        finally:
            torch.set_default_dtype(previous)
        if plan.quant.get(name) == "fp8":
            model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)

        if plan.placement.get(name) == SHARDED:
            gpus = plan.devices[:2]
            where = shard_split(_shard_units(model), [plan.shard[g] for g in gpus])
            device_map = {unit: self.hw.device_for(gpus[i]) for unit, i in where.items()}
        else:
            device_map = {"": self._device_of(plan, name) or "cpu"}
        # casting is much faster on a GPU: weights headed for system RAM (offload) are cast there and brought back
        staging = self.hw.device_for(plan.devices[0]) if self.hw.backend == "cuda" else None
        expected = {k: v.dtype for k, v in model.state_dict().items()}
        keep = getattr(cls, "_keep_in_fp32_modules", None) or []  # matched on name parts, as diffusers does
        for k, dt in expected.items():
            if dt.is_floating_point and any(m in k.split(".") for m in keep):
                expected[k] = torch.float32
        missing = set(expected)
        for key, tensor in _weight_stream(snap / name, local, cls, model, config):
            if key not in expected:
                continue
            dev = _device_for_key(key, device_map)
            floating = tensor.is_floating_point()
            if floating and tensor.dtype != expected[key]:
                tensor = tensor.to(staging if (dev == "cpu" and staging) else dev).to(expected[key]).to(dev)
            else:
                tensor = tensor.to(dev)
            set_module_tensor_to_device(model, key, dev, value=tensor, dtype=expected[key] if floating else None, clear_cache=False)
            missing.discard(key)
        if missing:
            raise ValueError(f"{len(missing)} weights missing, e.g. {sorted(missing)[:3]}")
        if plan.placement.get(name) == SHARDED:
            with warnings.catch_warnings():  # top-level parameters are listed by name, which accelerate warns about
                warnings.simplefilter("ignore", UserWarning)
                accelerate.dispatch_model(
                    model, device_map=device_map, main_device=self.hw.device_for(plan.devices[0]), force_hooks=True
                )
            model._es_shard = {unit: gpus[i] for unit, i in where.items()}
        model.eval()
        model._es_streamed = True
        return model

    def _place(self, pipe, plan: Plan) -> None:
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module

        torch = _torch()
        main = self.hw.device_for(plan.devices[0]) if self.hw.backend != "cpu" else "cpu"
        if plan.mode in ("gpu", "cpu"):
            pipe.to(main)
        elif plan.mode == "offload":
            pipe.enable_model_cpu_offload(device=main)
        elif plan.mode == "sequential":
            if any(_is_gguf(m) for m in pipe.components.values()):
                pipe.enable_model_cpu_offload(device=main)  # GGUF can't be streamed layer by layer
            else:
                pipe.enable_sequential_cpu_offload(device=main)
        elif plan.mode in ("split", "shard"):
            for name, module in pipe.components.items():
                if not isinstance(module, torch.nn.Module):
                    continue
                g = plan.placement.get(name, plan.devices[0])
                if g == SHARDED:
                    continue  # spread over the GPUs while loading, with its own hooks
                dev = "cpu" if g == CPU else self.hw.device_for(g)
                if plan.quant.get(name) != "nf4" and not _is_quantized(module):
                    module.to(dev)
                if dev != main:
                    add_hook_to_module(module, AlignDevicesHook(execution_device=dev, io_same_device=True))
            _pin_execution_device(pipe, main, plan)

    def _task_pipeline(self, lm: LoadedModel, src: ModelSource, task: str):
        import diffusers

        want = src.family.pipelines[task]
        pipe = lm.pipe
        if type(pipe).__name__ == want:
            return pipe
        new = getattr(diffusers, want).from_pipe(pipe)
        if lm.plan.mode == "offload":
            new.enable_model_cpu_offload(device=self.hw.device_for(lm.plan.devices[0]))
        elif lm.plan.mode in ("split", "shard"):
            _pin_execution_device(new, self.hw.device_for(lm.plan.devices[0]), lm.plan)
        if hasattr(new, "set_progress_bar_config"):
            new.set_progress_bar_config(disable=True)
        lm.pipe = new
        return new

    def _apply_loras(self, lm: LoadedModel, loras: list[tuple[str, float]], status) -> None:
        pipe = lm.pipe
        wanted = tuple((("l" + hashlib.sha1(p.encode()).hexdigest()[:10]), p) for p, _w in loras)
        if tuple(n for n, _ in wanted) != tuple(n for n, _ in lm.loras):
            if lm.loras and hasattr(pipe, "unload_lora_weights"):
                pipe.unload_lora_weights()
            lm.loras = ()
            for name, path in wanted:
                status("loading_lora")
                try:
                    pipe.load_lora_weights(path, adapter_name=name)
                except Exception as e:
                    raise ModelError("lora_failed", str(e), path=path)
            lm.loras = wanted
        if wanted:
            pipe.set_adapters([n for n, _ in wanted], [float(w) for _p, w in loras])
