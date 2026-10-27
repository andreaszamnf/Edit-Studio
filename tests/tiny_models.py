"""Builds tiny random-weight diffusers pipelines (same architectures as SDXL / Wan) for CPU tests.

They are written into a fake Hugging Face cache so the app finds them exactly like downloaded models.
"""

from __future__ import annotations

from pathlib import Path


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / CLIP byte-level alphabet."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _byte_vocab() -> dict[str, int]:
    chars = list(_bytes_to_unicode().values())
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    for c in chars:
        vocab.setdefault(c + "</w>", len(vocab))
    return vocab


def clip_tokenizer():
    from transformers import CLIPTokenizer

    return CLIPTokenizer(vocab=_byte_vocab(), merges=[], model_max_length=77)


def t5_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<pad>", "</s>", "<unk>"] + [chr(c) for c in range(97, 123)] + ["a", "cat", "dog", "robot", "video"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")


def tiny_sdxl():
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4), use_linear_projection=True, addition_embed_type="text_time",
        addition_time_embed_dim=8, transformer_layers_per_block=(1, 2), projection_class_embeddings_input_dim=80,
        cross_attention_dim=64, norm_num_groups=1,
    )
    vae = AutoencoderKL(
        block_out_channels=[32, 64], in_channels=3, out_channels=3, down_block_types=["DownEncoderBlock2D"] * 2,
        up_block_types=["UpDecoderBlock2D"] * 2, latent_channels=4, sample_size=128,
    )
    cfg = CLIPTextConfig(bos_token_id=0, eos_token_id=1, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
                         num_attention_heads=4, num_hidden_layers=2, pad_token_id=1, vocab_size=600, hidden_act="gelu",
                         projection_dim=32)
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, steps_offset=1, beta_schedule="scaled_linear",
                                       timestep_spacing="leading")
    tok = clip_tokenizer()
    return StableDiffusionXLPipeline(
        vae=vae, text_encoder=CLIPTextModel(cfg), text_encoder_2=CLIPTextModelWithProjection(cfg),
        tokenizer=tok, tokenizer_2=tok, unet=unet, scheduler=scheduler,
    )


def tiny_wan():
    import torch
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
    from transformers import UMT5Config, UMT5EncoderModel

    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True])
    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
        text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32,
    )
    te = UMT5EncoderModel(UMT5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=37, num_layers=2, num_heads=4,
                                     relative_attention_num_buckets=8, decoder_start_token_id=0, pad_token_id=0))
    return WanPipeline(tokenizer=t5_tokenizer(), text_encoder=te, vae=vae, transformer=transformer,
                       scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0))


def put_in_hf_cache(pipe, cache: Path, repo: str) -> Path:
    """Saves a pipeline the way huggingface_hub lays out a downloaded snapshot."""
    folder = cache / ("models--" + repo.replace("/", "--"))
    commit = "0" * 40
    snap = folder / "snapshots" / commit
    pipe.save_pretrained(snap, safe_serialization=True)
    (folder / "refs").mkdir(parents=True, exist_ok=True)
    (folder / "refs" / "main").write_text(commit)
    (folder / "blobs").mkdir(exist_ok=True)
    return snap
