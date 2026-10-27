from edit_studio.models.catalog import CATALOG_BY_ID
from edit_studio.models.planner import CPU, GB, Component, Plan, fallback, make_plan

TWO_GPUS = {0: int(14.8 * GB), 1: int(15.6 * GB)}  # GPU 0 also drives the display


def comps(model_id):
    spec = CATALOG_BY_ID[model_id]
    fam = spec.fam
    out = []
    for name, gb in spec.components_gb.items():
        role = "denoiser" if name in fam.denoisers else "text_encoder" if name in fam.text_encoders else "vae" if name == "vae" else "other"
        out.append(Component(name, int(gb * GB), role))
    return out, int(fam.activation_gb * GB)


def plan_for(model_id, vram=TWO_GPUS, ram=64, **kw):
    c, act = comps(model_id)
    return make_plan(c, vram, int(ram * GB), act, can_nf4=True, **kw)


def test_small_model_fits_one_gpu_whole():
    p = plan_for("sdxl-base")
    assert p.mode == "gpu" and p.quant == {} and p.gpus_needed == 1


def test_mid_models_use_one_gpu_with_offload_uncompressed():
    for mid in ("z-image-turbo", "flux2-klein-4b", "wan2.2-ti2v-5b", "wan2.1-t2v-1.3b"):
        p = plan_for(mid)
        assert p.mode == "offload", (mid, p)
        assert p.quant == {}, (mid, p)
        assert p.devices == [1]  # the GPU with more free memory


def test_split_preference_spreads_components():
    p = plan_for("z-image-turbo", mode_pref="split")
    assert p.mode == "split" and p.gpus_needed == 2
    assert p.placement["transformer"] != p.placement["text_encoder"]
    assert p.placement["vae"] == p.placement["transformer"]  # latents stay on the main GPU


def test_huge_edit_model_gets_compressed_but_runs_on_gpu():
    p = plan_for("qwen-image-edit-2511")
    assert p.mode in ("offload", "split")
    assert p.quant.get("transformer") == "nf4"


def test_wan_a14b_split_puts_one_expert_per_gpu():
    p = plan_for("wan2.2-i2v-a14b", mode_pref="split")
    assert p.mode == "split"
    assert {p.placement["transformer"], p.placement["transformer_2"]} == {0, 1}
    assert p.placement["text_encoder"] in (0, 1, CPU)


def test_without_bitsandbytes_falls_back_to_slow_but_valid_plan():
    c, act = comps("qwen-image-edit-2511")
    p = make_plan(c, TWO_GPUS, int(32 * GB), act, can_nf4=False)
    assert p.mode == "sequential"


def test_cpu_and_mps_backends():
    c, act = comps("z-image-turbo")
    assert make_plan(c, {}, 32 * GB, act, backend="cpu").mode == "cpu"
    assert make_plan(c, {0: 64 * GB}, 64 * GB, act, backend="mps").mode == "gpu"


def test_fallback_chain_after_oom():
    c, _ = comps("z-image-turbo")
    p = Plan("gpu", [0], {x.name: 0 for x in c})
    p2 = fallback(p, c, can_nf4=True)
    assert p2.mode == "offload"
    p3 = fallback(p2, c, can_nf4=True)
    assert p3.mode == "offload" and p3.quant["transformer"] == "nf4"
    p4 = fallback(p3, c, can_nf4=True)
    assert p4.mode == "sequential"
    assert fallback(p4, c, can_nf4=True) is None


def test_plan_key_changes_with_placement():
    a = Plan("gpu", [0], {"t": 0})
    b = Plan("gpu", [1], {"t": 1})
    assert a.key != b.key
