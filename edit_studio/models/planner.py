"""Decides where each part of a model goes (which GPU, CPU offload) and whether to compress it.

One graphics card (first plan that fits wins):
  1. whole model on the GPU
  2. CPU offload (the parts take turns on the GPU)
for each compression level in turn: none, text encoders 4-bit, denoiser stored in 8 bits (fp8),
everything 4-bit; sequential offload is the slow last resort.

Two graphics cards, same compression levels:
  1. split: whole parts on different GPUs (text encoder on one card, transformer on the other)
  2. shard: the denoiser's layers spread over both GPUs, for denoisers too big for one card

Which one a job gets (`choose`):
  * a model that needs more compression (or the slow mode) on one card than on two always gets both cards;
  * a mid-size image model gets both cards when both are free (no offload: faster), else one;
  * everything else uses one card, leaving the other free for a second job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GB = 1024**3
SAFETY = int(0.6 * GB)  # cuDNN workspaces, fragmentation
NF4_RATIO = 0.27  # 4-bit weights + scales + the few layers kept in bf16
FP8_RATIO = 0.51
CPU = -1  # placement value for "stays on the CPU" (text encoders that fit no GPU)
TE_ON_CPU_RAM = 0.55  # a text encoder may run on the CPU when it needs at most this share of system RAM
SHARDED = -2  # placement value for "layers spread over all plan devices"
QUALITY = {None: 0, "fp8": 1, "nf4": 2}


@dataclass
class Component:
    name: str
    bytes: int
    role: str  # denoiser | text_encoder | vae | other
    fixed: bool = False  # already compressed (GGUF files): never re-quantized
    shardable: bool = True  # layers can be spread over GPUs (diffusers transformers; not GGUF, not UNets)


@dataclass
class Plan:
    mode: str  # gpu | offload | split | shard | sequential | cpu
    devices: list[int]  # GPU indices used (first = main device: latents, VAE)
    placement: dict[str, int] = field(default_factory=dict)  # component -> GPU index, CPU or SHARDED
    quant: dict[str, str] = field(default_factory=dict)  # component -> fp8 | nf4
    est_peak: dict[int, int] = field(default_factory=dict)
    shard: dict[int, int] = field(default_factory=dict)  # shard mode: GPU -> room for denoiser layers (bytes)
    roles: dict[str, str] = field(default_factory=dict)  # component -> role, for the UI summary

    @property
    def gpus_needed(self) -> int:
        return len(self.devices) if self.mode in ("split", "shard") else 1

    @property
    def key(self) -> str:
        # `shard` room follows free VRAM, which drifts a little: it must not force a reload
        q = ",".join(f"{k}={v}" for k, v in sorted(self.quant.items()))
        p = ",".join(f"{k}@{v}" for k, v in sorted(self.placement.items()))
        return f"{self.mode}|{'-'.join(map(str, self.devices))}|{p}|{q}"

    def public(self) -> dict:
        def worst(role: str) -> str | None:
            qs = [self.quant.get(n) for n, r in self.roles.items() if r == role]
            return max(qs, key=lambda q: QUALITY[q], default=None)

        return {
            "mode": self.mode,
            "devices": self.devices,
            "placement": self.placement,
            "quant": self.quant,
            "main_quant": worst("denoiser"),  # the big model's compression
            "text_quant": worst("text_encoder"),
            "gpus_needed": self.gpus_needed,
            "cpu_parts": [k for k, v in self.placement.items() if v == CPU],
            "peak_gb": {str(k): round(v / GB, 1) for k, v in self.est_peak.items()},
        }


@dataclass
class Options:
    """The best one-card and two-card plans and which of them a job should use."""

    single: Plan
    multi: Plan | None
    choice: str  # single | preferred (2 cards when both free) | required (always 2 cards)

    @property
    def gpu_range(self) -> tuple[int, int]:
        return {"single": (1, 1), "preferred": (1, 2), "required": (2, 2)}[self.choice]

    def for_gpus(self, n: int) -> Plan:
        return self.multi if (n >= 2 and self.multi is not None and self.choice != "single") else self.single


def size_of(c: Component, quant: dict[str, str]) -> int:
    q = None if c.fixed else quant.get(c.name)
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

    # Then encoders / VAE. The VAE prefers the main GPU, where it decodes in the memory the denoiser
    # used for its activations; anywhere else it needs room of its own to decode.
    for c in others:
        cands = [main] if c.role == "vae" else []
        cands += sorted((g for g in room if g not in cands), key=lambda g: -room[g])
        for g in cands:
            need = size_of(c, quant) + (0 if (c.role == "vae" and g == main) else _working(c, act))
            if need <= room[g]:
                placement[c.name] = g
                room[g] -= need
                break
        else:
            # A text encoder runs once per prompt, so it may stay on the CPU (uncompressed).
            if c.role == "text_encoder" and c.bytes < ram * TE_ON_CPU_RAM:
                placement[c.name] = CPU
                quant.pop(c.name, None)
                continue
            return None

    used = sorted({g for g in placement.values() if g != CPU}, key=lambda g: (g != main, g))
    if len(used) < 2:
        return None  # fits on one GPU; the single-GPU plans cover that
    return Plan("split", used, placement, quant, {g: vram[g] - SAFETY - room[g] for g in used})


def _try_shard(comps, quant, vram, act, ram) -> Plan | None:
    """The denoiser's layers over the two GPUs with the most memory; each of them runs some layers."""
    dens = [c for c in comps if c.role == "denoiser"]
    if len(vram) < 2 or len(dens) != 1 or not dens[0].shardable or dens[0].fixed or quant.get(dens[0].name) == "nf4":
        return None
    den = dens[0]
    quant = dict(quant)
    main, other = sorted(vram, key=vram.get, reverse=True)[:2]
    room = {g: vram[g] - SAFETY - act for g in (main, other)}
    need = size_of(den, quant)
    margin = int(need * 0.03 + 0.3 * GB)  # layers come in whole blocks
    placement = {den.name: SHARDED}
    rest = sorted((c for c in comps if c is not den), key=lambda c: (c.role != "vae", -size_of(c, quant)))
    for c in rest:
        size = size_of(c, quant)
        if c.role == "vae":
            placement[c.name] = main  # decodes where the latents are, in the denoiser's activation memory
            room[main] -= size
            continue
        g = max(room, key=room.get)
        if size <= room[g] and sum(room.values()) - size >= need + margin:
            placement[c.name] = g
            room[g] -= size
        elif c.role == "text_encoder" and c.bytes < ram * TE_ON_CPU_RAM:
            placement[c.name] = CPU
            quant.pop(c.name, None)
        else:
            return None
    total = sum(room.values())
    if min(room.values()) <= int(0.5 * GB) or total < need + margin:
        return None
    peak = {g: vram[g] - SAFETY - room[g] + int(need * room[g] / total) for g in room}
    return Plan("shard", [main, other], placement, quant, peak, dict(room))


def _levels(comps, compression: str, can_nf4: bool) -> list[dict[str, str]]:
    dens = [c for c in comps if c.role == "denoiser" and not c.fixed]
    encs = [c for c in comps if c.role == "text_encoder"]
    nf4_enc = {c.name: "nf4" for c in encs} if can_nf4 else {}
    fp8_den = {c.name: "fp8" for c in dens}
    nf4_all = {c.name: "nf4" for c in encs + dens} if (can_nf4 and dens) else {}
    levels: list[dict[str, str]] = []
    if compression in ("auto", "none"):
        levels.append({})
    if compression == "auto":
        levels += [nf4_enc, fp8_den, {**nf4_enc, **fp8_den}, nf4_all]
    elif compression == "fp8":
        levels.append(fp8_den)
    elif compression == "nf4":
        levels.append(nf4_all)
    unique: list[dict[str, str]] = []
    for lv in levels:
        if (lv or not unique) and lv not in unique:
            unique.append(lv)
    return unique or [{}]


def _quality(plan: Plan, comps) -> tuple[int, int]:
    """(denoiser, text encoder) compression; lower is better. Parts on the CPU run uncompressed."""
    den = max((QUALITY[None if c.fixed else plan.quant.get(c.name)] for c in comps if c.role == "denoiser"), default=0)
    enc = max((QUALITY[plan.quant.get(c.name)] for c in comps if c.role == "text_encoder"), default=0)
    return den, enc


def _speed(plan: Plan) -> int:
    if plan.mode == "gpu":
        return 0
    if plan.mode in ("split", "shard"):
        return 3 if CPU in plan.placement.values() else 1  # a text encoder on the CPU is slower than offload
    return 2 if plan.mode == "offload" else 9


def choose(single: Plan, multi: Plan | None, comps, mode_pref: str, kind: str) -> str:
    if multi is None or mode_pref in ("single", "offload", "lowvram"):
        return "single"
    if mode_pref == "split" or single.mode == "sequential":
        return "required"
    qm, qs = _quality(multi, comps), _quality(single, comps)
    if qm < qs:
        return "required"  # better quality on two cards: worth waiting for both
    if qm == qs and _speed(multi) < _speed(single) and (kind == "image" or qs[0] > 0):
        # faster on two cards: worth it for short jobs and for big (compressed) models; when the other card
        # is busy one card is fine. Mid-size video models keep to one card: their minutes of denoising gain
        # nothing from skipping a few seconds of offload, and the other card stays free for a second job.
        return "preferred"
    return "single"


def plan_options(
    comps: list[Component],
    vram: dict[int, int],
    ram: int,
    activation: int,
    *,
    mode_pref: str = "auto",
    compression: str = "auto",
    can_nf4: bool = False,
    backend: str = "cuda",
    kind: str = "image",
) -> Options:
    if backend == "cpu" or not vram:
        return Options(Plan("cpu", [0]), None, "single")
    if backend == "mps":  # unified memory: no offload, no bitsandbytes
        return Options(Plan("gpu", [0], {c.name: 0 for c in comps}), None, "single")

    levels = _levels(comps, compression, can_nf4)
    single = None
    if mode_pref != "lowvram":
        kinds = ("offload",) if mode_pref == "offload" else ("gpu", "offload")
        for quant in levels:
            for k in kinds:
                single = _try_single(comps, quant, vram, activation) if k == "gpu" else _try_offload(comps, quant, vram, activation, ram)
                if single:
                    break
            if single:
                break
    if single is None:
        best_gpu = max(vram, key=vram.get)
        single = Plan("sequential", [best_gpu], {}, {} if mode_pref == "lowvram" else dict(levels[-1]))

    multi = None
    if len(vram) >= 2 and mode_pref in ("auto", "split"):
        for quant in levels:
            multi = _try_split(comps, quant, vram, activation, ram) or _try_shard(comps, quant, vram, activation, ram)
            if multi:
                break
    for p in (single, multi):
        if p is not None:
            p.roles = {c.name: c.role for c in comps if not c.fixed}
    return Options(single, multi, choose(single, multi, comps, mode_pref, kind))


def make_plan(comps: list[Component], vram: dict[int, int], ram: int, activation: int, **kw) -> Plan:
    """The plan for a job that got the GPUs in `vram`."""
    return plan_options(comps, vram, ram, activation, **kw).for_gpus(len(vram))


def sequential_plan(device: int, comps: list[Component], can_nf4: bool) -> Plan:
    """Last resort after repeated out-of-memory errors."""
    quant = {c.name: "nf4" for c in comps if c.role in ("denoiser", "text_encoder") and not c.fixed} if can_nf4 else {}
    return Plan("sequential", [device], {}, quant, roles={c.name: c.role for c in comps if not c.fixed})
