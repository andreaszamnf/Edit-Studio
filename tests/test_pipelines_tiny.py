"""Runs the real diffusers code paths (load, plan, from_pipe, callbacks, saving) with tiny random models."""

import pytest
from PIL import Image, ImageDraw

from conftest import wait

pytest.importorskip("torch")
pytest.importorskip("diffusers")

from tiny_models import put_in_hf_cache, tiny_sdxl, tiny_wan, tiny_wan_lora  # noqa: E402


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


def test_split_placement_mechanics(svc, monkeypatch):
    """Split plans: hooks on the parts that live elsewhere, pinned execution device, survives from_pipe."""
    import torch

    from edit_studio.models.planner import Plan

    mm = svc.models
    src = mm.resolve("hf:test/tiny-sdxl")
    # pretend "GPU 1" is a different device: cpu:0 is a distinct string, so hooks get attached
    monkeypatch.setattr(svc.hw, "device_for", lambda i: "cpu" if i == 0 else "cpu:0")
    plan = Plan("split", [0, 1], {"unet": 0, "vae": 0, "text_encoder": 1, "text_encoder_2": 1})
    monkeypatch.setattr(mm, "plan", lambda s, slots, gguf=False: plan)
    pipe, used = mm.acquire(src, "txt2img", [0, 1])
    assert used.mode == "split"
    assert pipe._execution_device == torch.device("cpu")
    assert hasattr(pipe.text_encoder, "_hf_hook") and not hasattr(pipe.unet, "_hf_hook")
    assert pipe.hf_device_map  # tells diffusers the hooks are ours: LoRA loading must keep them
    image = pipe(prompt="a cat", num_inference_steps=2, width=64, height=64).images[0]
    assert image.size == (64, 64)
    pipe2, _ = mm.acquire(src, "img2img", [0, 1])
    assert type(pipe2).__name__ == "StableDiffusionXLImg2ImgPipeline"
    assert pipe2._execution_device == torch.device("cpu")
    out = pipe2(prompt="a dog", image=image, strength=0.5, num_inference_steps=2).images[0]
    assert out.size == (64, 64)
    assert {s["model"] for s in mm.status()} == {"hf:test/tiny-sdxl"}
    assert mm.status()[0]["slots"] == [0, 1]


def _frames(pipe, **kw):
    import torch

    g = torch.Generator("cpu").manual_seed(1)
    return pipe(prompt="a robot", num_frames=5, width=32, height=32, num_inference_steps=2, generator=g,
                output_type="pt", **kw).frames


def test_shard_streams_layers_over_two_gpus(svc, monkeypatch, tmp_path):
    """Shard plans: the denoiser is built empty and streamed in, its layers in one run per GPU, and the result
    matches the unsharded model exactly. Both "GPUs" are the CPU here; real cards can't be tested in CI."""
    import torch

    from edit_studio.models.planner import SHARDED, Plan

    mm = svc.models
    src = mm.resolve("hf:test/tiny-wan")
    monkeypatch.setattr(mm, "plan", lambda s, slots: Plan("cpu", [0]))
    pipe, _ = mm.acquire(src, "txt2vid", [0])
    reference = _frames(pipe)
    mm.unload_all(only_idle=False)

    monkeypatch.setattr(svc.hw, "device_for", lambda i: "cpu")
    plan = Plan("shard", [1, 0], {"transformer": SHARDED, "text_encoder": 0, "vae": 1}, {}, shard={0: 10**6, 1: 10**6})
    monkeypatch.setattr(mm, "plan", lambda s, slots: plan)
    pipe, used = mm.acquire(src, "txt2vid", [0, 1])
    tr = pipe.transformer
    assert used.mode == "shard" and getattr(tr, "_es_streamed", False)  # the pipeline kept the streamed model
    where = [tr._es_shard[name] for name in tr._es_shard]
    assert set(where) == {0, 1}
    assert where == [1] * where.count(1) + [0] * where.count(0)  # main GPU (1) first, one run per GPU
    assert hasattr(tr, "_hf_hook") and hasattr(tr.blocks[0], "_hf_hook")
    assert torch.equal(_frames(pipe), reference)
    assert {s["model"]: s["slots"] for s in mm.status()} == {"hf:test/tiny-wan": [1, 0]}

    # LoRAs load into the spread layers and keep our placement hooks (no switch to sequential offload)
    lora = tiny_wan_lora(tmp_path)
    pipe, _ = mm.acquire(src, "txt2vid", [0, 1], [(str(lora), 1.0)])
    assert hasattr(pipe.transformer, "_hf_hook") and not hasattr(pipe.vae, "_hf_hook")
    with_lora = _frames(pipe)
    assert torch.isfinite(with_lora).all() and not torch.equal(with_lora, reference)

    # other tasks reuse the loaded layers
    pipe2, _ = mm.acquire(src, "vid2vid", [0, 1])
    assert type(pipe2).__name__ == "WanVideoToVideoPipeline" and pipe2.transformer is tr
    assert pipe2._execution_device == torch.device("cpu") and pipe2.hf_device_map
    mm.unload_all(only_idle=False)

    # 8-bit storage on top: diffusers' casting hooks and accelerate's placement hooks on the same layers
    fp8 = Plan("shard", [1, 0], {"transformer": SHARDED, "text_encoder": 0, "vae": 1}, {"transformer": "fp8"},
               shard={0: 10**6, 1: 10**6})
    monkeypatch.setattr(mm, "plan", lambda s, slots: fp8)
    pipe, _ = mm.acquire(src, "txt2vid", [0, 1])
    assert pipe.transformer.blocks[1].attn1.to_q.weight.dtype == torch.float8_e4m3fn
    out = _frames(pipe)
    assert torch.isfinite(out).all() and (out - reference).abs().max() < 0.05


def test_fp8_storage_is_streamed_and_keeps_norms_precise(svc, monkeypatch):
    import torch

    from edit_studio.models.planner import Plan

    mm = svc.models
    src = mm.resolve("hf:test/tiny-wan")
    monkeypatch.setattr(mm, "plan", lambda s, slots: Plan("cpu", [0]))
    reference = _frames(mm.acquire(src, "txt2vid", [0])[0])
    mm.unload_all(only_idle=False)

    monkeypatch.setattr(mm, "plan", lambda s, slots: Plan("cpu", [0], {}, {"transformer": "fp8"}))
    pipe, _ = mm.acquire(src, "txt2vid", [0])
    tr = pipe.transformer
    assert getattr(tr, "_es_streamed", False)
    assert tr.blocks[0].attn1.to_q.weight.dtype == torch.float8_e4m3fn
    assert tr.blocks[0].norm2.weight.dtype == torch.float32  # skipped by layerwise casting
    assert tr.blocks[0].scale_shift_table.dtype == torch.float32  # _keep_in_fp32_modules, as in from_pretrained
    out = _frames(pipe)
    assert torch.isfinite(out).all() and (out - reference).abs().max() < 0.05


def test_single_file_denoiser_is_streamed(svc, tmp_path):
    """A transformer in one file (ComfyUI-style folders) loads tensor by tensor, with the rest from its base repo."""
    import torch
    from safetensors.torch import save_file

    from edit_studio.models.local import LocalModel

    reference = _frames(svc.models.acquire(svc.models.resolve("hf:test/tiny-wan"), "txt2vid", [0])[0])
    svc.models.unload_all(only_idle=False)

    snap = svc.hub.snapshot("test/tiny-wan")
    import diffusers

    weights = diffusers.WanTransformer3DModel.from_pretrained(str(snap), subfolder="transformer").state_dict()
    path = tmp_path / "wan_tiny_t2v.safetensors"
    save_file({k: v.contiguous() for k, v in weights.items()}, str(path))
    lm = LocalModel(id="local:tiny-wan-file", path=str(path), name="wan_tiny_t2v", kind="denoiser", family="wan_t2v",
                    variant="1.3b", fmt="safetensors", dtype="F32", size=path.stat().st_size, mtime=0.0,
                    source="test", supported=True, base_repo="test/tiny-wan")
    svc.local.models[lm.id] = lm
    src = svc.models.resolve(lm.id)
    assert src.kind == "denoiser" and svc.models.missing_downloads(src) == []
    pipe, _ = svc.models.acquire(src, "txt2vid", [0])
    assert getattr(pipe.transformer, "_es_streamed", False)
    out = _frames(pipe)
    assert torch.equal(out, reference)  # same weights, same result as the folder version


def test_streamed_weights_match_from_pretrained(svc):
    """Same dtypes (incl. _keep_in_fp32_modules) and values as diffusers' own loader, at bf16."""
    import diffusers
    import torch

    from edit_studio.models.planner import Plan

    snap = svc.hub.snapshot("test/tiny-wan")
    cls = diffusers.WanTransformer3DModel
    ours = svc.models._stream_denoiser(cls, snap, "transformer", None, Plan("cpu", [0]), torch.bfloat16)
    theirs = cls.from_pretrained(str(snap), subfolder="transformer", torch_dtype=torch.bfloat16)
    a, b = dict(ours.named_parameters()), dict(theirs.named_parameters())
    assert a.keys() == b.keys()
    assert {k: v.dtype for k, v in a.items()} == {k: v.dtype for k, v in b.items()}
    assert a["blocks.0.scale_shift_table"].dtype == torch.float32 and a["blocks.0.attn1.to_q.weight"].dtype == torch.bfloat16
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert {k: v.dtype for k, v in ours.named_buffers()} == {k: v.dtype for k, v in theirs.named_buffers()}
