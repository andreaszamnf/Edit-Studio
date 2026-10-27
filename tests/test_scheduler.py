import threading
import time

from conftest import wait
from edit_studio.jobs import TaskError, task


def _register_fake_tasks():
    @task("test.gpu", "gpu", model_param="model")
    def fake_gpu(ctx, params):
        ctx.progress(0.1, "generating")
        until = time.time() + params.get("seconds", 0.4)
        while time.time() < until:
            ctx.check()
            time.sleep(0.02)
        params.setdefault("_seen", {})["slots"] = list(ctx.slots)
        return []

    @task("test.fail", "cpu")
    def fake_fail(ctx, params):
        raise TaskError("empty_mask")


_register_fake_tasks()


def test_two_gpus_run_jobs_in_parallel(two_gpu_services, monkeypatch):
    svc = two_gpu_services
    monkeypatch.setattr(svc.models, "gpus_needed", lambda model, enabled: 1)
    t0 = time.time()
    a = svc.jobs.submit("test.gpu", {"model": "m1", "seconds": 0.6})
    b = svc.jobs.submit("test.gpu", {"model": "m2", "seconds": 0.6})
    wait(a), wait(b)
    assert a.status == b.status == "done"
    assert sorted(a.slots + b.slots) == [0, 1]
    assert time.time() - t0 < 1.1  # ran side by side, not one after the other


def test_split_job_waits_for_both_gpus_and_blocks_others(two_gpu_services, monkeypatch):
    svc = two_gpu_services
    monkeypatch.setattr(svc.models, "gpus_needed", lambda model, enabled: 2 if model == "big" else 1)
    first = svc.jobs.submit("test.gpu", {"model": "small", "seconds": 0.5})
    time.sleep(0.15)
    big = svc.jobs.submit("test.gpu", {"model": "big", "seconds": 0.3})
    later = svc.jobs.submit("test.gpu", {"model": "small", "seconds": 0.1})
    time.sleep(0.15)
    # the free GPU must not be taken by `later` while the two-GPU job is waiting
    assert big.status == "queued" and later.status == "queued"
    wait(first), wait(big), wait(later)
    assert sorted(big.slots) == [0, 1]
    assert later.started >= big.started


def test_pinned_gpu_is_respected(two_gpu_services, monkeypatch):
    svc = two_gpu_services
    monkeypatch.setattr(svc.models, "gpus_needed", lambda model, enabled: 1)
    j = wait(svc.jobs.submit("test.gpu", {"model": "m", "gpu": 1, "seconds": 0.05}))
    assert j.slots == [1]


def test_cancel_running_and_queued(two_gpu_services, monkeypatch):
    svc = two_gpu_services
    monkeypatch.setattr(svc.models, "gpus_needed", lambda model, enabled: 2)
    running = svc.jobs.submit("test.gpu", {"model": "m", "seconds": 5})
    queued = svc.jobs.submit("test.gpu", {"model": "m", "seconds": 5})
    deadline = time.time() + 3
    while running.status != "running" and time.time() < deadline:
        time.sleep(0.02)
    assert svc.jobs.cancel(queued.id) and queued.status == "cancelled"
    assert svc.jobs.cancel(running.id)
    wait(running)
    assert running.status == "cancelled"


def test_task_errors_carry_ui_keys(services):
    j = wait(services.jobs.submit("test.fail", {}))
    assert j.status == "failed" and j.error_key == "empty_mask"
    assert services.jobs.remove(j.id) and services.jobs.get(j.id) is None


def test_events_are_published(services):
    seen = []
    lock = threading.Lock()
    orig = services.bus.publish

    def spy(kind, data):
        with lock:
            seen.append(kind)
        orig(kind, data)

    services.bus.publish = spy
    wait(services.jobs.submit("test.fail", {}))
    assert seen.count("job") >= 2
