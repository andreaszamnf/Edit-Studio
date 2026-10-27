"""Detects GPUs / Apple Silicon / CPU and reports usable memory."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from functools import lru_cache

import psutil

log = logging.getLogger(__name__)

GB = 1024**3


@dataclass
class GpuInfo:
    index: int
    name: str
    total_bytes: int
    capability: tuple[int, int] = (0, 0)

    @property
    def device(self) -> str:
        return f"cuda:{self.index}"


@dataclass
class Hardware:
    backend: str  # cuda | mps | cpu
    gpus: list[GpuInfo] = field(default_factory=list)
    torch_version: str = ""
    cuda_version: str = ""
    error: str = ""

    # ---- memory -----------------------------------------------------------------
    def usable_vram(self, index: int) -> int:
        """Bytes this app could use on a GPU: free memory plus what our own cache holds."""
        if self.backend == "mps":
            try:
                import torch

                return int(torch.mps.recommended_max_memory())
            except Exception:
                return int(psutil.virtual_memory().total * 0.7)
        if self.backend != "cuda":
            return 0
        import torch

        try:
            free, _total = torch.cuda.mem_get_info(index)
            ours = torch.cuda.memory_reserved(index)
            return int(free + ours)
        except Exception:
            return self.gpus[index].total_bytes if index < len(self.gpus) else 0

    @staticmethod
    def ram_total() -> int:
        return int(psutil.virtual_memory().total)

    @staticmethod
    def ram_available() -> int:
        return int(psutil.virtual_memory().available)

    # ---- dtype / devices ------------------------------------------------------------
    def device_for(self, index: int) -> str:
        if self.backend == "cuda":
            return f"cuda:{index}"
        return self.backend

    def supports_bf16(self, index: int = 0) -> bool:
        if self.backend == "cuda":
            return self.gpus[index].capability >= (8, 0) if index < len(self.gpus) else False
        return self.backend == "mps"

    def slot_ids(self, enabled: list[int] | None) -> list[int]:
        """GPU indices the scheduler may use (one "slot" on CPU/MPS machines)."""
        if self.backend != "cuda":
            return [0]
        ids = [g.index for g in self.gpus]
        if enabled:
            ids = [i for i in ids if i in enabled] or ids
        return ids

    def summary(self) -> dict:
        vm = psutil.virtual_memory()
        return {
            "backend": self.backend,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_gb": round(g.total_bytes / GB, 1),
                    "usable_gb": round(self.usable_vram(g.index) / GB, 1),
                    "capability": f"{g.capability[0]}.{g.capability[1]}",
                }
                for g in self.gpus
            ],
            "ram_total_gb": round(vm.total / GB, 1),
            "ram_available_gb": round(vm.available / GB, 1),
            "torch": self.torch_version,
            "cuda": self.cuda_version,
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "error": self.error,
        }


@lru_cache(maxsize=1)
def detect() -> Hardware:
    try:
        import torch
    except Exception as e:  # torch missing or broken install
        log.error("PyTorch could not be imported: %s", e)
        return Hardware(backend="cpu", error=f"PyTorch: {e}")

    hw = Hardware(backend="cpu", torch_version=torch.__version__)
    try:
        if torch.cuda.is_available():
            hw.backend = "cuda"
            hw.cuda_version = torch.version.cuda or ""
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                hw.gpus.append(GpuInfo(i, p.name, int(p.total_memory), (p.major, p.minor)))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            hw.backend = "mps"
            hw.gpus.append(GpuInfo(0, "Apple Silicon (MPS)", int(psutil.virtual_memory().total)))
    except Exception as e:
        log.exception("GPU detection failed")
        hw.error = str(e)
    if hw.backend == "cpu" and not hw.error:
        hw.error = "no-gpu"
    return hw
