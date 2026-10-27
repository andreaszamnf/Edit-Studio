"""Decides where each part of a model goes (which GPU, CPU offload) and whether to compress it.

Order of preference (first plan that fits wins):
  1. whole model on one GPU
  2. one GPU with CPU offload (parts take turns on the GPU)
  3. split over several GPUs (e.g. text encoder on GPU 1, transformer on GPU 0)
then the same again with the text encoders compressed, then with the big denoiser compressed
(fp8 storage when there is enough system RAM to load it, otherwise 4-bit NF4), and finally
sequential offload as the slow last resort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GB = 1024**3
SAFETY = int(0.6 * GB)  # cuDNN workspaces, fragmentation
NF4_RATIO = 0.27  # 4-bit weights + scales + the few layers kept in bf16
FP8_RATIO = 0.51
CPU = -1  # placement value for "stays on the CPU" (split mode text encoders)


@dataclass
class Component:
    name: str
    bytes: int
    role: str  # denoiser | text_encoder | vae | other


@dataclass
class Plan:
    mode: str  # gpu | offload | split | sequential | cpu
    devices: list[int]  # GPU indices used (first = main device)
    placement: dict[str, int] = field(default_factory=dict)  # component -> GPU index or CPU
    quant: dict[str, str] = field(default_factory=dict)  # component -> fp8 | nf4
    est_peak: dict[int, int] = field(default_factory=dict)

    @property
    def gpus_needed(self) -> int:
        return len(self.devices) if self.mode == "split" else 1

    @property
    def key(self) -> str:
        q = ",".join(f"{k}={v}" for k, v in sorted(self.quant.items()))
        p = ",".join(f"{k}@{v}" for k, v in sorted(self.placement.items()))
        return f"{self.mode}|{'-'.join(map(str, self.devices))}|{p}|{q}"

    def public(self) -> dict:
        return {
            "mode": self.mode,
            "devices": self.devices,
            "placement": self.placement,
            "quant": self.quant,
            "gpus_needed": self.gpus_needed,
            "peak_gb": {str(k): round(v / GB, 1) for k, v in self.est_peak.items()},
        }


def size_of(c: Component, quant: dict[str, str]) -> int:
    q = quant.get(c.name)
    if q == "nf4":
        return int(c.bytes * NF4_RATIO)
    if q == "fp8":
        return int(c.bytes * FP8_RATIO)
    return c.bytes


def _working(c: Component, act: int) -> int:
    """Extra memory a component needs while it runs."""
    return {"denoiser": act, "text_encoder": int(0.5 * GB), "vae": act // 2}.get(c.role, int(0.3 * GB))


def _try_single(comps, quant, vram, act) -> Plan | None:
    total = sum(size_of(c, quant) for c in comps)
    peak = total + max((_working(c, act) for c in comps), default=0)
    for gpu in sorted(vram, key=vram.get, reverse=True):
        if peak + SAFETY <= vram[gpu]:
            return Plan("gpu", [gpu], {c.name: gpu for c in comps}, dict(quant), {gpu: peak})
    return None


def _try_offload(comps, quant, vram, act, ram) -> Plan | None:
    total = sum(size_of(c, quant) for c in comps)
    if total > ram * 0.9:  # everything lives in system RAM between uses
        return None
    peak = max((size_of(c, quant) + _working(c, act) for c in comps), default=0)
    for gpu in sorted(vram, key=vram.get, reverse=True):
        if peak + SAFETY <= vram[gpu]:
            return Plan("offload", [gpu], {}, dict(quant), {gpu: peak})
    return None


def _try_split(comps, quant, vram, act, ram) -> Plan | None:
    if len(vram) < 2:
        return None
    quant = dict(quant)
    room = {g: v - SAFETY for g, v in vram.items()}
    placement: dict[str, int] = {}
    with_act: set[int] = set()
    denoisers = sorted((c for c in comps if c.role == "denoiser"), key=lambda c: -size_of(c, quant))
    others = sorted((c for c in comps if c.role != "denoiser"), key=lambda c: -size_of(c, quant))

    # Denoisers first, spread over GPUs (Wan 2.2 A14B: one expert per GPU).
    for c in denoisers:
        need = size_of(c, quant)
        for g in sorted(room, key=lambda g: (g in with_act, -room[g])):
            extra = 0 if g in with_act else act
            if need + extra <= room[g]:
                placement[c.name] = g
                room[g] -= need + extra
                with_act.add(g)
                break
        else:
            return None
    main = placement[denoisers[0].name] if denoisers else max(room, key=room.get)

    # Then encoders / VAE; the VAE prefers the main GPU where the latents live.
    for c in others:
        need = size_of(c, quant) + (_working(c, act) if c.role != "vae" else 0)
        cands = [main] if c.role == "vae" else []
        cands += sorted((g for g in room if g not in cands), key=lambda g: -room[g])
        for g in cands:
            if need <= room[g]:
                placement[c.name] = g
                room[g] -= need
                break
        else:
            # A text encoder runs once per prompt, so it may stay on the CPU (uncompressed).
            if c.role == "text_encoder" and c.bytes < ram * 0.5:
                placement[c.name] = CPU
                quant.pop(c.name, None)
                continue
            return None

    used = sorted({g for g in placement.values() if g != CPU}, key=lambda g: (g != main, g))
    if len(used) < 2:
        return None  # fits on one GPU; the single-GPU plans cover that
    return Plan("split", used, placement, quant, {g: vram[g] - SAFETY - room[g] for g in used})


def make_plan(
    comps: list[Component],
    vram: dict[int, int],
    ram: int,
    activation: int,
    *,
    mode_pref: str = "auto",
    compression: str = "auto",
    can_nf4: bool = False,
    backend: str = "cuda",
) -> Plan:
    if backend == "cpu" or not vram:
        return Plan("cpu", [0])
    if backend == "mps":  # unified memory: no offload, no bitsandbytes
        return Plan("gpu", [0], {c.name: 0 for c in comps})

    denoisers = [c for c in comps if c.role == "denoiser"]
    encoders = [c for c in comps if c.role == "text_encoder"]
    nf4_enc = {c.name: "nf4" for c in encoders} if can_nf4 else {}
    biggest_den = max((c.bytes for c in denoisers), default=0)
    fp8_ok = biggest_den and biggest_den < ram * 0.7  # fp8 storage loads bf16 weights into RAM first

    levels: list[dict[str, str]] = []
    if compression in ("auto", "none"):
        levels.append({})
    if compression == "auto":
        if nf4_enc:
            levels.append(nf4_enc)
        if fp8_ok:
            levels.append({**nf4_enc, **{c.name: "fp8" for c in denoisers}})
        if can_nf4:
            levels.append({c.name: "nf4" for c in encoders + denoisers})
    elif compression == "fp8" and fp8_ok:
        levels.append({c.name: "fp8" for c in denoisers})
    elif compression == "nf4" and can_nf4:
        levels.append({c.name: "nf4" for c in encoders + denoisers})
    if not levels:
        levels.append({})

    order = {
        "auto": ("gpu", "offload", "split"),
        "single": ("gpu", "offload"),
        "split": ("split", "gpu", "offload"),
        "offload": ("offload",),
        "lowvram": (),
    }.get(mode_pref, ("gpu", "offload", "split"))

    for quant in levels:
        for kind in order:
            if kind == "gpu":
                plan = _try_single(comps, quant, vram, activation)
            elif kind == "offload":
                plan = _try_offload(comps, quant, vram, activation, ram)
            else:
                plan = _try_split(comps, quant, vram, activation, ram)
            if plan:
                return plan
    best_gpu = max(vram, key=vram.get)
    return Plan("sequential", [best_gpu], {}, {} if mode_pref == "lowvram" else levels[-1])


def fallback(plan: Plan, comps: list[Component], can_nf4: bool) -> Plan | None:
    """Next safer plan after an out-of-memory error."""
    if plan.mode in ("gpu", "split"):
        return Plan("offload", plan.devices[:1], {}, {k: v for k, v in plan.quant.items()})
    if plan.mode == "offload":
        if can_nf4:
            stronger = dict(plan.quant)
            for c in comps:
                if c.role in ("denoiser", "text_encoder"):
                    stronger[c.name] = "nf4"
            if stronger != plan.quant:
                return Plan("offload", plan.devices, {}, stronger)
        return Plan("sequential", plan.devices, {}, dict(plan.quant))
    return None
