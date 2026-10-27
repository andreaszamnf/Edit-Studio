"""Loads diffusers pipelines onto the GPUs according to a memory plan and keeps them cached."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import SettingsStore
from ..hardware import GB, Hardware
from .catalog import CATALOG_BY_ID, FAMILIES, Family, ModelSpec, family_from_index
from .hub import HubClient
from .local import LocalLibrary, LocalModel
from .planner import CPU, Component, Plan, fallback, make_plan

log = logging.getLogger(__name__)

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


def _pin_execution_device(pipe, device: str) -> None:
    """Split pipelines: make every intermediate tensor start on the main GPU."""
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
        self._plan_overrides: dict[str, Plan] = {}

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
                repo=lm.base_repo,
                path=lm.path,
                model_variant=lm.variant,
                local=lm,
                defaults=family_defaults(lm.family, lm.variant),
                pipeline_config=pc,
            )
        raise ModelError("model_missing", model=model_id)

    # ------------------------------------------------------------------ install state
    def missing_downloads(self, src: ModelSource) -> list[dict]:
        """Hugging Face downloads needed before `src` can load."""
        if src.kind == "checkpoint" or not src.repo:
            return []
        if self.hub.is_installed(src.repo, src.skip_weights):
            return []
        return [{"repo": src.repo, "variant": src.variant, "skip_weights": list(src.skip_weights)}]

    # ------------------------------------------------------------------ planning
    def can_nf4(self) -> bool:
        if self.hw.backend != "cuda":
            return False
        try:
            importlib.import_module("bitsandbytes")
            return True
        except Exception:
            return False

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
            return [
                Component(fam.denoisers[0], unet, "denoiser"),
                Component("text_encoder", int(total * 0.2), "text_encoder"),
                Component("vae", int(total * 0.05), "vae"),
            ]
        if snap is not None and (snap / "model_index.json").is_file():
            for name in module_components(read_model_index(snap)):
                if src.kind == "denoiser" and name in fam.denoisers[:1]:
                    continue
                size = folder_weight_bytes(snap / name, 4 if (name == "vae" and fam.vae_fp32) else 2)
                comps.append(Component(name, size, _role(fam, name)))
        elif src.spec:
            comps = [Component(n, int(gb * GB), _role(fam, n)) for n, gb in src.spec.components_gb.items()]
        if src.kind == "denoiser" and src.local:
            from .local import param_count, read_safetensors_header

            size = src.local.size
            if src.local.fmt == "safetensors":
                try:
                    size = param_count(read_safetensors_header(src.path)[0]) * 2
                except Exception:
                    pass
            comps.append(Component(fam.denoisers[0], size, "denoiser"))
        return comps

    def _gpu_vram(self, gpu_ids: list[int]) -> dict[int, int]:
        return {i: self.hw.usable_vram(i) for i in gpu_ids}

    def plan(self, src: ModelSource, gpu_ids: list[int], *, gguf: bool = False) -> Plan:
        s = self.settings.settings
        comps = self.components(src)
        override_key = self._override_key(src, gpu_ids)
        if override_key in self._plan_overrides:
            return self._plan_overrides[override_key]
        act = int(src.family.activation_gb * GB)
        plan = make_plan(
            comps,
            self._gpu_vram(gpu_ids),
            self.hw.ram_total(),
            act,
            mode_pref=s.memory_mode,
            compression=s.compression,
            can_nf4=self.can_nf4(),
            backend=self.hw.backend,
        )
        if src.local and src.local.fmt == "gguf":  # already compressed; never re-quantize
            plan.quant.pop(src.family.denoisers[0], None)
        return plan

    def _override_key(self, src: ModelSource, gpu_ids: list[int]) -> str:
        s = self.settings.settings
        return f"{src.id}|{','.join(map(str, gpu_ids))}|{s.memory_mode}|{s.compression}"

    def gpus_needed(self, model_id: str, enabled: list[int]) -> int:
        try:
            src = self.resolve(model_id)
        except ModelError:
            return 1
        return self.plan(src, enabled).gpus_needed

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
        if self.hw.backend == "cuda":
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
        need = sum(c.bytes for c in self.components(src)) if plan.mode in ("offload", "sequential", "cpu") else 0
        if need and self.hw.ram_available() < need * 1.1:
            busy = self.busy_slots() - set(slots)
            with self.lock:
                idle = {id(lm): lm for s, lm in self.loaded.items() if not (set(lm.slots) & busy)}
            for lm in idle.values():
                self._drop(lm)

    def _load_with_fallback(self, src, task, plan: Plan, slots, status) -> LoadedModel:
        torch = _torch()
        comps = self.components(src)
        attempt = plan
        while True:
            try:
                pipe = self._load(src, attempt, status)
                lm_slots = attempt.devices if attempt.mode == "split" else slots[:1]
                lm = LoadedModel(f"{src.id}|{attempt.key}", src.id, attempt, pipe, lm_slots)
                with self.lock:
                    for s in lm.slots:
                        self.loaded[s] = lm
                if attempt is not plan:
                    self._plan_overrides[self._override_key(src, slots)] = attempt
                return lm
            except torch.OutOfMemoryError:
                log.warning("Out of memory loading %s with %s", src.id, attempt.key)
                self._collect()
                nxt = fallback(attempt, comps, self.can_nf4())
                if nxt is None:
                    raise ModelError("oom")
                status("fallback:" + nxt.mode)
                attempt = nxt

    def report_oom(self, src: ModelSource, slots: list[int]) -> bool:
        """Called when generation runs out of memory: remember a safer plan. Returns True if one exists."""
        lm = self.loaded_on(slots[0])
        current = lm.plan if lm else self.plan(src, slots)
        nxt = fallback(current, self.components(src), self.can_nf4())
        if lm:
            self._drop(lm)
        self._collect()
        if nxt is None:
            return False
        self._plan_overrides[self._override_key(src, slots)] = nxt
        return True

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
        """Where a component should be loaded directly (None = CPU, moved later by offload hooks)."""
        if plan.mode == "gpu":
            return self.hw.device_for(plan.devices[0])
        if plan.mode == "split":
            g = plan.placement.get(name, plan.devices[0])
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
            pipe = cls.from_single_file(src.path, dtype=dtype, token=token)
        else:
            snap = self.hub.snapshot(src.repo)
            if snap is None:
                raise ModelError("model_missing", repo=src.repo)
            index = read_model_index(snap)
            comps = module_components(index)
            overrides: dict[str, Any] = {}
            if src.kind == "denoiser":
                den = fam.denoisers[0]
                lib, cls_name = comps[den]
                den_cls = getattr(diffusers, cls_name)
                kwargs: dict[str, Any] = {"config": str(snap), "subfolder": den, "dtype": dtype}
                if src.local and src.local.fmt == "gguf":
                    kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(compute_dtype=dtype)
                elif plan.quant.get(den) == "nf4":
                    kwargs["quantization_config"] = self._quant_config("diffusers", "nf4")
                dev = self._device_of(plan, den)
                if dev and dev != "cpu":
                    kwargs["device_map"] = {"": dev}
                status("loading:" + den)
                overrides[den] = den_cls.from_single_file(src.path, **kwargs)
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
            pipe = diffusers.DiffusionPipeline.from_pretrained(str(snap), dtype=dtype, **overrides)

        # fp8 storage: weights kept in 8 bits, each layer upcast just before it runs
        for name, q in plan.quant.items():
            module = getattr(pipe, name, None)
            if q == "fp8" and module is not None and hasattr(module, "enable_layerwise_casting"):
                module.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)

        if src.pipeline_config.get("flow_shift") and hasattr(pipe, "scheduler"):
            sched = pipe.scheduler
            pipe.scheduler = type(sched).from_config(sched.config, flow_shift=src.pipeline_config["flow_shift"])

        self._place(pipe, plan)
        if fam.kind == "video" and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return pipe

    def _place(self, pipe, plan: Plan) -> None:
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module

        torch = _torch()
        main = self.hw.device_for(plan.devices[0]) if self.hw.backend != "cpu" else "cpu"
        if plan.mode in ("gpu", "cpu"):
            pipe.to(main)
        elif plan.mode == "offload":
            pipe.enable_model_cpu_offload(device=main)
        elif plan.mode == "sequential":
            pipe.enable_sequential_cpu_offload(device=main)
        elif plan.mode == "split":
            for name, module in pipe.components.items():
                if not isinstance(module, torch.nn.Module):
                    continue
                g = plan.placement.get(name, plan.devices[0])
                dev = "cpu" if g == CPU else self.hw.device_for(g)
                if plan.quant.get(name) != "nf4":
                    module.to(dev)
                if dev != main:
                    add_hook_to_module(module, AlignDevicesHook(execution_device=dev, io_same_device=True))
            _pin_execution_device(pipe, main)

    def _task_pipeline(self, lm: LoadedModel, src: ModelSource, task: str):
        import diffusers

        want = src.family.pipelines[task]
        pipe = lm.pipe
        if type(pipe).__name__ == want:
            return pipe
        new = getattr(diffusers, want).from_pipe(pipe)
        if lm.plan.mode == "offload":
            new.enable_model_cpu_offload(device=self.hw.device_for(lm.plan.devices[0]))
        elif lm.plan.mode == "split":
            _pin_execution_device(new, self.hw.device_for(lm.plan.devices[0]))
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
