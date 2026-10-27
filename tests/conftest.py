import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from edit_studio.config import Paths  # noqa: E402
from edit_studio.hardware import GpuInfo, Hardware  # noqa: E402
from edit_studio.services import Services  # noqa: E402


@pytest.fixture
def services(tmp_path):
    svc = Services.create(Paths(tmp_path / "data"), hw=Hardware(backend="cpu"))
    yield svc
    svc.shutdown()


@pytest.fixture
def two_gpu_services(tmp_path, monkeypatch):
    """Services that believe there are two 16 GB GPUs and 64 GB of RAM (these tests never touch CUDA)."""
    try:
        import torch  # noqa: F401  (job threads import it; keep that out of timing-sensitive tests)
    except ImportError:
        pass
    hw = Hardware(backend="cuda", gpus=[GpuInfo(0, "Fake GPU 0", 16 * 1024**3), GpuInfo(1, "Fake GPU 1", 16 * 1024**3)])
    free = {0: int(14.8 * 1024**3), 1: int(15.6 * 1024**3)}  # GPU 0 also drives the display
    monkeypatch.setattr(Hardware, "usable_vram", lambda self, i: free[i])
    monkeypatch.setattr(Hardware, "ram_total", staticmethod(lambda: 64 * 1024**3))  # plans must not depend on the test machine
    svc = Services.create(Paths(tmp_path / "data"), hw=hw)
    yield svc
    svc.shutdown()


def wait(job, timeout=120):
    start = time.time()
    while not job.finished_state and time.time() - start < timeout:
        time.sleep(0.05)
    assert job.finished_state, f"job {job.task} still {job.status}"
    return job
