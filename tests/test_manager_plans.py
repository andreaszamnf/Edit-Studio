"""Model manager planning on a (fake) computer with two 16 GB graphics cards."""


def test_plan_summary_for_the_ui(two_gpu_services):
    mm = two_gpu_services.models
    big = mm.describe(mm.resolve("qwen-image-edit-2511"), [0, 1])
    assert big["gpus"] == [2, 2] and big["mode"] in ("shard", "split")
    mid = mm.describe(mm.resolve("z-image-turbo"), [0, 1])
    assert mid["gpus"] == [1, 2] and mid["mode"] == "split" and mid["alone"]["mode"] == "offload"
    small = mm.describe(mm.resolve("sdxl-base"), [0, 1])
    assert small["gpus"] == [1, 1] and small["mode"] == "gpu"
    assert mm.gpu_range("qwen-image-edit-2511", [0, 1]) == (2, 2)
    assert mm.gpu_range("qwen-image-edit-2511", [1]) == (1, 1)
    assert mm.gpu_range("no-such-model", [0, 1]) == (1, 1)


def test_out_of_memory_replans_with_more_headroom_until_sequential(two_gpu_services):
    mm = two_gpu_services.models
    src = mm.resolve("z-image-turbo")
    keys = [mm.plan(src, [0, 1]).key]
    while mm.report_oom(src, [0, 1]):
        keys.append(mm.plan(src, [0, 1]).key)
    assert len(keys) >= 3 and len(set(keys)) == len(keys)  # every step is a different, safer plan
    assert mm.plan(src, [0, 1]).mode == "sequential"
    assert mm.gpu_range("z-image-turbo", [0, 1]) == (1, 1)
    # other GPU sets keep their own (unharmed) plans
    assert mm.plan(src, [1]).mode == "offload"
