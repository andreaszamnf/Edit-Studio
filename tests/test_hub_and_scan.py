import json

import numpy as np
from safetensors.numpy import save_file

from edit_studio.models.hub import HubClient, select_files, snapshot_looks_complete
from edit_studio.models.local import LocalLibrary, classify, inspect_file, roots_for


def test_select_files_prefers_safetensors_and_variant():
    siblings = [
        ("model_index.json", 1),
        ("sd_xl_base_1.0.safetensors", 7000),  # root single file: skipped
        ("unet/config.json", 1),
        ("unet/diffusion_pytorch_model.safetensors", 10000),
        ("unet/diffusion_pytorch_model.fp16.safetensors", 5000),
        ("unet/diffusion_pytorch_model.bin", 10000),
        ("text_encoder/model.fp16.safetensors", 100),
        ("text_encoder/model.safetensors", 200),
        ("tokenizer/vocab.json", 1),
        ("tokenizer/merges.txt", 1),
        ("vae/diffusion_pytorch_model.safetensors", 300),
        ("README.md", 1),
        ("images/sample.png", 1),
    ]
    picked = dict(select_files(siblings, ["unet", "text_encoder", "tokenizer", "vae"], "fp16"))
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in picked
    assert "unet/diffusion_pytorch_model.safetensors" not in picked
    assert "unet/diffusion_pytorch_model.bin" not in picked
    assert "text_encoder/model.fp16.safetensors" in picked
    assert "vae/diffusion_pytorch_model.safetensors" in picked  # no fp16 variant there: keep the default
    assert "tokenizer/merges.txt" in picked and "model_index.json" in picked
    assert "sd_xl_base_1.0.safetensors" not in picked and "README.md" not in picked


def test_select_files_skip_weights_keeps_configs():
    siblings = [("model_index.json", 1), ("transformer/config.json", 1),
                ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 5),
                ("transformer/diffusion_pytorch_model.safetensors.index.json", 1), ("vae/config.json", 1),
                ("vae/diffusion_pytorch_model.safetensors", 3)]
    picked = dict(select_files(siblings, ["transformer", "vae"], None, ("transformer",)))
    assert "transformer/config.json" in picked
    assert not any(k.startswith("transformer/diffusion") for k in picked)
    assert "vae/diffusion_pytorch_model.safetensors" in picked


def _snapshot(tmp_path, repo="org/model"):
    folder = tmp_path / ("models--" + repo.replace("/", "--"))
    snap = folder / "snapshots" / ("a" * 40)
    (snap / "unet").mkdir(parents=True)
    (snap / "tokenizer").mkdir()
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text("a" * 40)
    (snap / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "scheduler": [None, None],
    }))
    return snap


def test_install_state_from_cache_and_manifest(tmp_path):
    snap = _snapshot(tmp_path)
    hub = HubClient(lambda: None, lambda: str(tmp_path), tmp_path / "manifests")
    assert hub.snapshot("org/model") == snap
    assert not hub.is_installed("org/model")  # unet has no weights yet
    (snap / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    assert snapshot_looks_complete(snap)
    assert hub.is_installed("org/model")
    # a manifest from a configs-only download must not count as a full install
    hub.write_manifest("org/model", ["model_index.json"], 1, ("unet",))
    (snap / "unet" / "diffusion_pytorch_model.safetensors").unlink()
    assert hub.is_installed("org/model", ("unet",))
    assert not hub.is_installed("org/model")


def _fake(path, tensors):
    save_file({k: np.zeros(shape, dtype=np.float16) for k, shape in tensors.items()}, str(path))
    return path


def test_classify_common_files(tmp_path):
    sdxl = _fake(tmp_path / "juggernaut.safetensors", {
        "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
        "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias": (4,),
        "first_stage_model.decoder.conv_out.weight": (3, 1, 1, 1),
    })
    m = inspect_file(sdxl, "ComfyUI", "checkpoint")
    assert (m.kind, m.family, m.supported) == ("checkpoint", "sdxl", True)

    inpaint = _fake(tmp_path / "sd15-inpaint.safetensors", {
        "model.diffusion_model.input_blocks.0.0.weight": (320, 9, 3, 3),
        "model.diffusion_model.output_blocks.11.0.skip_connection.weight": (1,),
        "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight": (77, 8),
    })
    m = inspect_file(inpaint, "A1111", "checkpoint")
    assert (m.family, m.variant) == ("sd15", "inpaint")

    flux_dev = _fake(tmp_path / "flux1-dev.safetensors", {
        "double_blocks.0.img_attn.norm.key_norm.scale": (128,), "img_in.weight": (8, 64), "guidance_in.in_layer.weight": (4, 4),
    })
    m = inspect_file(flux_dev, "ComfyUI", "denoiser")
    assert (m.family, m.variant, m.base_repo) == ("flux1", "dev", "black-forest-labs/FLUX.1-dev")

    kontext = _fake(tmp_path / "flux1-kontext-dev.safetensors", {
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": (128,),
        "model.diffusion_model.img_in.weight": (8, 64), "model.diffusion_model.guidance_in.in_layer.weight": (4, 4),
    })
    assert inspect_file(kontext, "ComfyUI", "denoiser").family == "flux1_kontext"

    zimg = _fake(tmp_path / "z_image_turbo_bf16.safetensors", {"layers.0.adaLN_modulation.0.weight": (4, 4)})
    m = inspect_file(zimg, "ComfyUI", "denoiser")
    assert (m.family, m.variant, m.supported) == ("z_image", "turbo", True)

    qedit = _fake(tmp_path / "qwen_image_edit_2511_bf16.safetensors", {
        "txt_norm.weight": (8,), "img_in.weight": (8, 8), "transformer_blocks.0.img_mod.1.weight": (8, 8),
    })
    assert inspect_file(qedit, "ComfyUI", "denoiser").family == "qwen_image_edit"

    scaled = _fake(tmp_path / "qwen_image_fp8_scaled.safetensors", {
        "txt_norm.weight": (8,), "img_in.weight": (8, 8), "transformer_blocks.0.img_mod.1.weight": (8, 8),
        "transformer_blocks.0.img_mod.1.scale_weight": (1,),
    })
    m = inspect_file(scaled, "ComfyUI", "denoiser")
    assert not m.supported and m.note == "fp8_scaled"

    wan = _fake(tmp_path / "wan2.1_t2v_1.3B.safetensors", {"head.modulation": (1, 2, 1536), "patch_embedding.weight": (1536, 16, 1, 2, 2)})
    assert inspect_file(wan, "ComfyUI", "denoiser").family == "wan_t2v"

    lora = _fake(tmp_path / "style.safetensors", {"lora_unet_down_blocks_0_attn.lora_down.weight": (4, 8), "lora_te2_text.lora_up.weight": (8, 4)})
    m = inspect_file(lora, "ComfyUI", "lora")
    assert (m.kind, m.family) == ("lora", "sdxl")


def test_classify_unknown_returns_unsupported():
    kind, family, _variant, supported, _note = classify({"foo.bar": {"shape": [1]}}, "x.safetensors", None)
    assert family is None and not supported


def test_scan_comfyui_layout(tmp_path):
    root = tmp_path / "ComfyUI"
    (root / "models" / "checkpoints").mkdir(parents=True)
    (root / "models" / "loras").mkdir()
    (root / "models" / "upscale_models").mkdir()
    _fake(root / "models" / "checkpoints" / "realvis.safetensors", {
        "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
        "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias": (4,),
    })
    (root / "models" / "upscale_models" / "4x-UltraSharp.pth").write_bytes(b"not really")
    roots = roots_for([str(root)], auto=False)
    assert roots and roots[0].source == "ComfyUI"
    lib = LocalLibrary(tmp_path / "index.json")
    found = {m.name: m for m in lib.scan([str(root)], auto=False)}
    assert found["realvis"].family == "sdxl"
    assert found["4x-UltraSharp"].kind == "upscaler" and found["4x-UltraSharp"].supported
    # index is cached and reloaded
    again = LocalLibrary(tmp_path / "index.json")
    assert set(again.models) == set(lib.models)


def test_scan_stability_matrix_and_forge_layouts(tmp_path):
    sm = tmp_path / "StabilityMatrix" / "Data" / "Models"
    (sm / "StableDiffusion").mkdir(parents=True)
    (sm / "Lora").mkdir()
    forge = tmp_path / "webui_forge_cu121" / "webui" / "models"
    (forge / "Stable-diffusion").mkdir(parents=True)
    _fake(sm / "StableDiffusion" / "pony.safetensors", {
        "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
        "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias": (4,),
    })
    _fake(forge / "Stable-diffusion" / "dreamshaper.safetensors", {
        "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
        "model.diffusion_model.output_blocks.11.0.skip_connection.weight": (1,),
        "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight": (77, 8),
    })
    lib = LocalLibrary(tmp_path / "index.json")
    found = {m.name: m for m in lib.scan([str(sm), str(tmp_path / "webui_forge_cu121" / "webui")], auto=False)}
    assert found["pony"].source == "StabilityMatrix" and found["pony"].family == "sdxl"
    assert found["dreamshaper"].source == "Forge" and found["dreamshaper"].family == "sd15"


def test_cancel_stops_a_download_that_ignores_progress_exceptions(tmp_path, monkeypatch):
    """Xet transfers run in Rust and ignore exceptions from the progress callback: cancel must abort them."""
    import threading
    import time

    import edit_studio.models.hub as hub_mod
    from edit_studio.models.hub import DownloadCancelled

    aborted = threading.Event()

    def fake_snapshot_download(repo, **kw):
        bar = kw["tqdm_class"](total=100, name="huggingface_hub.snapshot_download")
        while not aborted.is_set():  # like Rust: swallows what the callback raises
            try:
                bar.update(1)
            except DownloadCancelled:
                pass
            time.sleep(0.01)
        raise RuntimeError("xet session aborted")

    monkeypatch.setattr(hub_mod, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub_mod, "_abort_transfers", aborted.set)
    hub = HubClient(lambda: None, lambda: str(tmp_path), tmp_path / "manifests")
    monkeypatch.setattr(hub, "remote_files", lambda repo, variant, skip=(): [("model_index.json", 100)])

    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.time()
    try:
        hub.download("org/model", None, lambda n, t: None, cancel)
        raise AssertionError("not cancelled")
    except DownloadCancelled:
        pass
    assert time.time() - t0 < 3 and aborted.is_set()


def test_download_resumes_when_another_download_was_cancelled(tmp_path, monkeypatch):
    import threading

    import edit_studio.models.hub as hub_mod

    hub = HubClient(lambda: None, lambda: str(tmp_path), tmp_path / "manifests")
    calls = []

    def fake_snapshot_download(repo, **kw):
        calls.append(repo)
        if len(calls) == 1:
            hub._aborts += 1  # someone cancelled their download meanwhile: Xet aborted ours too
            raise RuntimeError("xet session aborted")
        return str(tmp_path / "snap")

    monkeypatch.setattr(hub_mod, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "remote_files", lambda repo, variant, skip=(): [("model_index.json", 1)])
    assert hub.download("org/model", None, lambda n, t: None, threading.Event()) == tmp_path / "snap"
    assert len(calls) == 2


def test_hf_cache_in_a_model_folder_is_read(tmp_path):
    """A model folder that is another app's HF cache: pipelines count as installed, single files are scanned."""
    other = tmp_path / "zcore-img" / "models" / "hf" / "hub"
    other.mkdir(parents=True)
    snap = _snapshot(other, "org/model")
    (snap / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    lora_snap = other / "models--someone--cool-lora" / "snapshots" / ("b" * 40)
    lora_snap.mkdir(parents=True)
    _fake(lora_snap / "pytorch_lora_weights.safetensors", {"lora_unet_down_blocks_0_attn.lora_down.weight": (4, 8),
                                                           "lora_te2_text.lora_up.weight": (8, 4)})

    main_cache = tmp_path / "main-cache"
    hub = HubClient(lambda: None, lambda: str(main_cache), tmp_path / "manifests", lambda: [str(tmp_path / "zcore-img")])
    assert hub.snapshot("org/model") == snap and hub.is_installed("org/model")
    assert [p["repo"] for p in hub.cached_pipelines()] == ["org/model"]

    lib = LocalLibrary(tmp_path / "index.json")
    found = lib.scan([str(tmp_path / "zcore-img")], auto=False)
    assert [(m.name, m.kind, m.source) for m in found] == [("cool-lora · pytorch_lora_weights", "lora", "Hugging Face")]
