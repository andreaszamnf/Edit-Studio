import pytest

from edit_studio.models.catalog import CATALOG_BY_ID
from edit_studio.models.planner import CPU, GB, SHARDED, Component, Plan, make_plan, plan_options

TWO_GPUS = {0: int(14.8 * GB), 1: int(15.6 * GB)}  # two 16 GB cards; GPU 0 also drives the display


def comps(model_id):
    spec = CATALOG_BY_ID[model_id]
    fam = spec.fam
    out = []
    for name, gb in spec.components_gb.items():
        role = "denoiser" if name in fam.denoisers else "text_encoder" if name in fam.text_encoders else "vae" if name == "vae" else "other"
        out.append(Component(name, int(gb * GB), role, shardable=name != "unet"))
    return out, int(fam.activation_gb * GB), fam.kind


def options_for(model_id, vram=TWO_GPUS, ram=64, **kw):
    c, act, kind = comps(model_id)
    return plan_options(c, vram, int(ram * GB), act, can_nf4=True, kind=kind, **kw)


def test_small_model_fits_one_gpu_whole():
    o = options_for("sdxl-base")
    assert o.gpu_range == (1, 1)
    assert o.single.mode == "gpu" and o.single.quant == {} and o.single.gpus_needed == 1


def test_mid_image_models_use_both_cards_when_free_and_one_otherwise():
    for mid in ("z-image-turbo", "flux2-klein-4b"):
        o = options_for(mid)
        assert o.gpu_range == (1, 2), (mid, o.choice)
        both = o.for_gpus(2)
        assert both.mode == "split" and both.quant == {} and sorted(both.devices) == [0, 1]
        assert both.placement["transformer"] != both.placement["text_encoder"]
        assert both.placement["vae"] == both.placement["transformer"]  # latents stay on the main GPU
        alone = o.for_gpus(1)
        assert alone.mode == "offload" and alone.quant == {}
        assert alone.devices == [1]  # the card with more free memory


def test_mid_video_models_keep_to_one_card():
    for mid in ("wan2.2-ti2v-5b", "wan2.1-t2v-1.3b"):
        o = options_for(mid)
        assert o.gpu_range == (1, 1), mid
        assert o.single.mode == "offload" and o.single.quant == {}


@pytest.mark.parametrize("ram", [32, 64])
def test_big_models_spread_their_layers_over_both_cards_in_8_bit(ram):
    for big in ("qwen-image-edit-2511", "qwen-image-2512"):
        o = options_for(big, ram=ram)
        assert o.gpu_range == (2, 2), (big, ram)
        p = o.multi
        assert p.mode == "shard" and p.placement["transformer"] == SHARDED and p.gpus_needed == 2
        assert p.quant.get("transformer") == "fp8"  # on one card it would have to be 4-bit
        assert o.single.quant["transformer"] == "nf4"
        assert p.placement["text_encoder"] == CPU and "text_encoder" not in p.quant  # full precision, on the CPU
        assert p.placement["vae"] == p.devices[0]
        assert set(p.shard) == {0, 1} and sum(p.shard.values()) > 40.9 * GB * 0.51
        assert p.public()["main_quant"] == "fp8" and p.public()["cpu_parts"] == ["text_encoder"]


def test_big_model_with_little_ram_uses_4_bit_on_both_cards():
    o = options_for("qwen-image-edit-2511", ram=24)  # text encoder can't stay on the CPU
    assert o.gpu_range == (1, 2)
    p = o.for_gpus(2)
    assert p.mode == "split" and p.quant["transformer"] == "nf4"


def test_bf16_shard_when_only_the_text_encoder_needs_squeezing():
    o = options_for("qwen-image-2.1", ram=24)  # too little RAM to keep its text encoder on the CPU
    assert o.gpu_range == (2, 2)
    p = o.multi
    assert p.mode == "shard" and "transformer" not in p.quant  # bf16 layers over two cards
    assert p.quant["text_encoder"] == "nf4" and p.placement["text_encoder"] in (0, 1)
    assert o.single.quant["transformer"] == "fp8"


def test_wan_a14b_puts_one_expert_on_each_card_when_both_are_free():
    o = options_for("wan2.2-i2v-a14b")
    assert o.gpu_range == (1, 2)
    p = o.for_gpus(2)
    assert p.mode == "split"
    assert {p.placement["transformer"], p.placement["transformer_2"]} == {0, 1}


def test_memory_mode_preferences():
    assert options_for("z-image-turbo", mode_pref="split").gpu_range == (2, 2)
    single = options_for("qwen-image-edit-2511", mode_pref="single")
    assert single.gpu_range == (1, 1) and single.for_gpus(2).gpus_needed == 1
    assert options_for("qwen-image-edit-2511", mode_pref="lowvram").for_gpus(1).mode == "sequential"


def test_without_bitsandbytes_a_big_model_still_runs_on_two_cards():
    c, act, kind = comps("qwen-image-edit-2511")
    o = plan_options(c, TWO_GPUS, 32 * GB, act, can_nf4=False, kind=kind)
    assert o.single.mode == "sequential"
    assert o.gpu_range == (2, 2) and o.multi.mode == "shard"


def test_gguf_denoiser_is_never_recompressed_or_sharded():
    c = [Component("transformer", int(11 * GB), "denoiser", fixed=True, shardable=False),
         Component("text_encoder", int(16.6 * GB), "text_encoder"), Component("vae", int(0.3 * GB), "vae")]
    o = plan_options(c, TWO_GPUS, 64 * GB, int(3 * GB), can_nf4=True)
    for p in (o.single, o.multi):
        assert p is not None and "transformer" not in p.quant and p.mode != "shard"


def test_cpu_and_mps_backends():
    c, act, _ = comps("z-image-turbo")
    assert make_plan(c, {}, 32 * GB, act, backend="cpu").mode == "cpu"
    assert make_plan(c, {0: 64 * GB}, 64 * GB, act, backend="mps").mode == "gpu"


def test_plan_key_follows_placement_but_not_shard_room():
    assert Plan("gpu", [0], {"t": 0}).key != Plan("gpu", [1], {"t": 1}).key
    a = Plan("shard", [1, 0], {"t": SHARDED}, {"t": "fp8"}, shard={0: 10 * GB, 1: 11 * GB})
    b = Plan("shard", [1, 0], {"t": SHARDED}, {"t": "fp8"}, shard={0: int(10.2 * GB), 1: 11 * GB})
    assert a.key == b.key  # free VRAM drifts a little between jobs: no reload for that


def test_shard_split_is_one_run_per_card_and_balanced():
    torch = pytest.importorskip("torch")
    from edit_studio.models.manager import shard_split

    units = [("embed", 10, None)] + [(f"blocks.{i}", 100, "blocks") for i in range(10)] + [("out", 10, None)]
    where = shard_split(units, [600, 600])
    order = [where[n] for n, _, _ in units]
    assert order == sorted(order) and set(order) == {0, 1}  # contiguous: one hand-over per step
    first = sum(size for n, size, _ in units if where[n] == 0)
    assert 450 <= first <= 600
    lopsided = shard_split(units, [300, 900])
    assert sum(size for n, size, _ in units if lopsided[n] == 0) <= 315
    with pytest.raises(torch.OutOfMemoryError):
        shard_split(units, [300, 300])


def test_shard_split_never_cuts_small_stacks_that_feed_root_code():
    pytest.importorskip("torch")
    from edit_studio.models.manager import shard_split

    # Z-Image-like: two small refiner stacks, root code in between, then the main layers
    units = ([("pad", 1, None), ("x_embedder", 20, None)] + [(f"noise_refiner.{i}", 50, "noise_refiner") for i in range(2)]
             + [("t_embedder", 5, None)] + [(f"layers.{i}", 100, "layers") for i in range(30)])
    where = shard_split(units, [200, 4000])  # the first GPU is nearly full: it still keeps the refiners
    assert where["noise_refiner.1"] == 0 and where["t_embedder"] == 0
    assert where["layers.29"] == 1
    # FLUX-like: the double and single block stacks are back to back, so the cut may fall in either
    flux = [("embed", 10, None)] + [(f"double.{i}", 100, "double") for i in range(4)] + [(f"single.{i}", 40, "single") for i in range(6)]
    cut = shard_split(flux, [600, 100])
    assert cut["single.2"] == 0 and cut["single.5"] == 1
