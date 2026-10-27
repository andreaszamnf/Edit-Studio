"""Job queue: runs tasks on GPU "slots" (one per graphics card), CPU workers and download workers."""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .services import Services

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


class TaskError(Exception):
    """A task failure with an i18n key for the UI."""

    def __init__(self, key: str, message: str = "", **info):
        super().__init__(message or key)
        self.key = key
        self.info = info


@dataclass
class TaskSpec:
    name: str
    func: Callable[[TaskContext, Any], list]
    resource: str  # gpu | cpu | net
    params_model: Any = None
    model_param: str | None = None


TASKS: dict[str, TaskSpec] = {}
MAX_JOBS = 300  # finished jobs beyond this are forgotten (their results stay in the library)


def task(name: str, resource: str = "gpu", params=None, model_param: str | None = None):
    def deco(fn):
        TASKS[name] = TaskSpec(name, fn, resource, params, model_param)
        return fn

    return deco


@dataclass
class Job:
    id: str
    task: str
    params: dict
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    stage: str = "queued"
    detail: str = ""
    step: int = 0
    steps: int = 0
    eta: float | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    slots: list[int] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    error: str = ""
    error_key: str = ""
    error_info: dict = field(default_factory=dict)
    depends_on: str | None = None
    gpu_range: tuple[int, int] | None = None  # (fewest, most) GPUs; None = not planned yet
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "id", "task", "params", "status", "progress", "stage", "detail", "step", "steps", "eta", "created",
                "started", "finished", "slots", "outputs", "error", "error_key", "error_info", "depends_on",
            )
        }

    @property
    def finished_state(self) -> bool:
        return self.status in ("done", "failed", "cancelled")


class TaskContext:
    def __init__(self, services: Services, manager: JobManager, job: Job, slots: list[int]):
        self.services = services
        self.manager = manager
        self.job = job
        self.slots = slots
        self._last_emit = 0.0
        self._t_steps: list[float] = []

    @property
    def device(self) -> str:
        hw = self.services.hw
        return hw.device_for(self.slots[0]) if self.slots and hw.backend != "cpu" else "cpu"

    def check(self) -> None:
        if self.job.cancel_event.is_set():
            raise Cancelled()

    def progress(self, value: float | None = None, stage: str | None = None, *, force: bool = False, **extra) -> None:
        j = self.job
        if value is not None:
            j.progress = max(0.0, min(1.0, float(value)))
        if stage is not None and stage != j.stage:
            j.stage = stage
            force = True
        for k, v in extra.items():
            setattr(j, k, v)
        now = time.time()
        if force or now - self._last_emit > 0.25:
            self._last_emit = now
            self.manager.emit(j)

    def status(self, stage: str) -> None:
        """Used by the model manager while loading (stage strings like 'loading:transformer')."""
        self.check()
        self.progress(stage=stage)

    def step_callback(self, total: int, start: float = 0.0, end: float = 1.0, stage: str = "generating",
                      final_stage: str | None = None):
        """A diffusers `callback_on_step_end` reporting progress / ETA and handling cancel.

        `final_stage` is shown after the last step (e.g. "decoding" while a video VAE runs).
        """
        self._t_steps = [time.time()]
        self.progress(start, stage, step=0, steps=total, eta=None)

        def cb(pipe, i, t, kwargs):
            self.check()
            n = getattr(pipe, "_num_timesteps", None) or total
            done = i + 1
            self._t_steps.append(time.time())
            per = (self._t_steps[-1] - self._t_steps[0]) / max(1, len(self._t_steps) - 1)
            label = final_stage if (final_stage and done >= n) else stage
            self.progress(start + (end - start) * done / n, label, step=done, steps=n, eta=per * (n - done))
            return kwargs

        return cb


class JobManager:
    def __init__(self, services: Services, slot_ids: list[int], cpu_workers: int = 2, net_workers: int = 2):
        self.services = services
        self.slot_ids = slot_ids
        self.slot_busy: dict[int, str | None] = {s: None for s in slot_ids}
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.cv = threading.Condition()
        self.cpu_workers = cpu_workers
        self.net_workers = net_workers
        self.running = {"cpu": 0, "net": 0}
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ public api
    def submit(self, task_name: str, params: dict, depends_on: str | None = None) -> Job:
        if task_name not in TASKS:
            raise KeyError(task_name)
        job = Job(id=uuid.uuid4().hex[:12], task=task_name, params=params, depends_on=depends_on)
        with self.cv:
            self.jobs[job.id] = job
            finished = [j.id for j in self.jobs.values() if j.finished_state]
            for old in finished[: max(0, len(self.jobs) - MAX_JOBS)]:
                del self.jobs[old]
            self.cv.notify_all()
        self.emit(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        with self.cv:
            return list(self.jobs.values())

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.finished_state:
            return False
        job.cancel_event.set()
        with self.cv:
            if job.status == "queued":
                job.status = "cancelled"
                job.finished = time.time()
            self.cv.notify_all()
        self.emit(job)
        return True

    def remove(self, job_id: str) -> bool:
        with self.cv:
            job = self.jobs.get(job_id)
            if job is None or not job.finished_state:
                return False
            del self.jobs[job_id]
        self.services.bus.publish("job_removed", {"id": job_id})
        return True

    def clear_finished(self) -> int:
        with self.cv:
            ids = [j.id for j in self.jobs.values() if j.finished_state]
            for i in ids:
                del self.jobs[i]
        for i in ids:
            self.services.bus.publish("job_removed", {"id": i})
        return len(ids)

    def busy_slots(self) -> set[int]:
        with self.cv:
            return {s for s, j in self.slot_busy.items() if j}

    def emit(self, job: Job) -> None:
        self.services.bus.publish("job", job.public())

    def stop(self) -> None:
        with self.cv:
            self._stop = True
            for j in self.jobs.values():
                j.cancel_event.set()
            self.cv.notify_all()

    # ------------------------------------------------------------------ scheduling
    def _loop(self) -> None:
        with self.cv:
            while not self._stop:
                try:
                    self._start_ready()
                except Exception:
                    log.exception("scheduler error")
                self.cv.wait(timeout=1.0)

    def _start_ready(self) -> None:
        free = [s for s in self.slot_ids if self.slot_busy[s] is None]
        gpu_blocked = False
        for job in list(self.jobs.values()):
            if job.status != "queued":
                continue
            if job.depends_on:
                dep = self.jobs.get(job.depends_on)
                if dep is not None and not dep.finished_state:
                    continue
                if dep is not None and dep.status != "done":
                    self._finish(job, "failed", key="dependency_failed", info={"dependency": dep.id})
                    continue
            spec = TASKS[job.task]
            if spec.resource == "gpu":
                if gpu_blocked:
                    continue
                fewest, most = self._gpu_range(job, spec)
                pinned = job.params.get("gpu")
                if isinstance(pinned, int) and pinned in self.slot_ids:  # the user picked a card: only that one
                    if pinned not in free:
                        continue
                    chosen = [pinned]
                elif len(free) < fewest:
                    if fewest > 1:
                        gpu_blocked = True  # a two-GPU job waits at the front; don't starve it
                    continue
                else:  # e.g. both cards when both are free, one when the other is busy
                    chosen = self._choose_slots(job, spec, free, min(most, len(free)))
                for s in chosen:
                    self.slot_busy[s] = job.id
                    free.remove(s)
                self._spawn(job, spec, chosen)
            else:
                limit = self.cpu_workers if spec.resource == "cpu" else self.net_workers
                if self.running[spec.resource] >= limit:
                    continue
                self.running[spec.resource] += 1
                self._spawn(job, spec, [])

    def _gpu_range(self, job: Job, spec: TaskSpec) -> tuple[int, int]:
        if job.gpu_range:
            return job.gpu_range
        fewest, most = 1, 1
        model_id = job.params.get(spec.model_param) if spec.model_param else None
        if model_id and len(self.slot_ids) > 1:
            try:
                fewest, most = self.services.models.gpu_range(model_id, self.slot_ids)
            except Exception:
                log.exception("planning failed for %s", model_id)
        n = len(self.slot_ids)
        job.gpu_range = (max(1, min(fewest, n)), max(1, min(most, n)))
        return job.gpu_range

    def _choose_slots(self, job: Job, spec: TaskSpec, free: list[int], need: int) -> list[int]:
        if need > 1:
            return sorted(free)[:need]
        model_id = job.params.get(spec.model_param) if spec.model_param else None
        models = self.services.models
        for s in free:  # the model is already loaded there
            if model_id and models.loaded_source(s) == model_id:
                return [s]
        empty = [s for s in free if models.loaded_on(s) is None]
        pool = empty or free
        hw = self.services.hw
        return [max(pool, key=lambda s: hw.usable_vram(s) if hw.backend == "cuda" else 0)]

    def _spawn(self, job: Job, spec: TaskSpec, slots: list[int]) -> None:
        job.status = "running"
        job.started = time.time()
        job.slots = slots
        job.stage = "starting"
        threading.Thread(target=self._run, args=(job, spec, slots), name=f"job-{job.id}", daemon=True).start()

    def _run(self, job: Job, spec: TaskSpec, slots: list[int]) -> None:
        from .models.manager import ModelError

        self.emit(job)
        ctx = TaskContext(self.services, self, job, slots)
        status, key, msg, info = "done", "", "", {}
        try:
            if spec.resource == "gpu" and slots and self.services.hw.backend == "cuda":
                _use_gpu(slots[0])  # implicit "cuda" tensors land on this job's GPU
            params = spec.params_model.model_validate(job.params) if spec.params_model else job.params
            outputs = spec.func(ctx, params) or []
            job.outputs = [getattr(o, "id", o) for o in outputs]
        except Cancelled:
            status = "cancelled"
        except (ModelError, TaskError) as e:
            status, key, msg, info = "failed", e.key, str(e), dict(e.info)
        except Exception as e:  # noqa: BLE001 - reported to the UI
            status, key, msg = "failed", _classify(e), f"{type(e).__name__}: {e}"
            log.error("Job %s (%s) failed:\n%s", job.id, job.task, traceback.format_exc())
        try:
            self._finish(job, status, key=key, message=msg, info=info)
            keep = self.services.settings.settings.keep_models_loaded
            if spec.resource == "gpu" and (not keep or (status == "failed" and key == "oom")):
                self.services.models.unload_slots(slots)  # before the slot is handed to another job
        finally:
            self._release(job, spec, slots)

    def _release(self, job: Job, spec: TaskSpec, slots: list[int]) -> None:
        with self.cv:
            for s in slots:
                if self.slot_busy.get(s) == job.id:
                    self.slot_busy[s] = None
            if spec.resource in self.running:
                self.running[spec.resource] = max(0, self.running[spec.resource] - 1)
            self.cv.notify_all()

    def _finish(self, job: Job, status: str, key: str = "", message: str = "", info: dict | None = None) -> None:
        job.status = status
        job.finished = time.time()
        job.eta = None
        if status == "done":
            job.progress = 1.0
        job.error_key = key
        job.error = message
        job.error_info = info or {}
        self.emit(job)
        if job.outputs:
            self.services.bus.publish("gallery", {"added": job.outputs})


def _use_gpu(index: int) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(index)


NETWORK_ERRORS = {
    "ConnectError", "ConnectionError", "ReadTimeout", "ConnectTimeout", "ProxyError", "RemoteProtocolError",
    "ReadError", "WriteError", "LocalEntryNotFoundError", "OfflineModeIsEnabled", "URLError", "TimeoutError",
}


def _classify(e: Exception) -> str:
    name = type(e).__name__
    text = str(e).lower()
    if name == "OutOfMemoryError" or "out of memory" in text:
        return "oom"
    if name in NETWORK_ERRORS:
        return "network"
    if name == "GatedRepoError" or "gated repo" in text or "restricted" in text and "access" in text:
        return "gated"
    if name in ("RepositoryNotFoundError", "EntryNotFoundError", "RevisionNotFoundError"):
        return "not_found"
    if "no space left" in text or getattr(e, "errno", None) == 28:
        return "disk_full"
    return "error"
