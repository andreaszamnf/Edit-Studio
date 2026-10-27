"""Runs the real diffusers code paths (load, plan, from_pipe, callbacks, saving) with tiny random models."""

import pytest
from PIL import Image, ImageDraw

from conftest import wait

pytest.importorskip("torch")
pytest.importorskip("diffusers")

from tiny_models import put_in_hf_cache, tiny_sdxl, tiny_wan  # noqa: E402


@pytest.fixture(scope="module")
def hf_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("hf")
    put_in_hf_cache(tiny_sdxl(), cache, "test/tiny-sdxl")
    put_in_hf_cache(tiny_wan(), cache, "test/tiny-wan")
    return cache


@pytest.fixture
def svc(services, hf_cache, monkeypatch):
    import edit_studio.tasks.image as image_tasks

    monkeypatch.setitem(image_tasks.FAMILY_PIXELS, "sdxl", 128 * 128)  # tiny model: keep the work size small
    services.settings.update({"hf_cache_dir": str(hf_cache)})
    return services


def run(svc, name, params):
    job = wait(svc.jobs.submit(name, params), timeout=300)
    assert job.status == "done", (name, job.error_key, job.error)
    return job


def test_cached_pipelines_are_listed_and_resolved(svc):
    repos = {p["repo"] for p in svc.hub.cached_pipelines()}
    assert {"test/tiny-sdxl", "test/tiny-wan"} <= repos
    assert svc.models.resolve("hf:test/tiny-sdxl").family.id == "sdxl"
    assert svc.models.resolve("hf:test/tiny-wan").family.id == "wan_t2v"
    assert svc.models.missing_downloads(svc.models.resolve("hf:test/tiny-sdxl")) == []


def test_image_generate_variations_inpaint_outpaint(svc, tmp_path):
    job = run(svc, "image.generate", {"model": "hf:test/tiny-sdxl", "prompt": "a cat", "width": 64, "height": 64,
                                      "steps": 2, "count": 2, "seed": 7})
    assert len(job.outputs) == 2
    first = svc.storage.get(job.outputs[0])
    assert (first.width, first.height) == (64, 64) and first.params["seed"] == 7
    assert svc.storage.get(job.outputs[1]).params["seed"] == 8
    assert job.steps == 2 and job.progress == 1.0

    run(svc, "image.img2img", {"model": "hf:test/tiny-sdxl", "image": first.id, "prompt": "a dog", "steps": 3})

    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).rectangle((16, 16, 48, 48), fill=255)
    mask.save(tmp_path / "mask.png")
    mask_item = svc.storage.add_upload(tmp_path / "mask.png", "mask.png", hidden=True)
    inp = run(svc, "image.inpaint", {"model": "hf:test/tiny-sdxl", "image": first.id, "mask": mask_item.id,
                                      "prompt": "flowers", "steps": 3, "grow": 0, "feather": 0})
    out = Image.open(svc.storage.path_of(inp.outputs[0])).convert("RGB")
    orig = Image.open(svc.storage.path_of(first.id)).convert("RGB")
    assert out.getpixel((1, 1)) == orig.getpixel((1, 1))  # untouched corner is pasted back exactly

    ext = run(svc, "image.outpaint", {"model": "hf:test/tiny-sdxl", "image": first.id, "left": 32, "right": 32, "steps": 2})
    wide = svc.storage.get(ext.outputs[0])
    assert (wide.width, wide.height) == (128, 64)
    # the pipeline was reused between tasks (from_pipe), not reloaded
    assert len(svc.models.status()) == 1


def test_video_generate(svc):
    job = run(svc, "video.generate", {"model": "hf:test/tiny-wan", "prompt": "a robot", "width": 32, "height": 32,
                                      "frames": 5, "fps": 8, "steps": 2, "seed": 3})
    item = svc.storage.get(job.outputs[0])
    assert item.kind == "video" and (item.width, item.height) == (32, 32)
    assert abs(item.duration - 5 / 8) < 0.1
