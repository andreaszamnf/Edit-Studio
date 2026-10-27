"""Built-in model catalog.

A *family* describes how a group of checkpoints is driven through diffusers (pipeline class per
task, which components are the heavy denoisers / text encoders, sensible defaults). A *model spec*
is one concrete downloadable checkpoint from Hugging Face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WAN_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


@dataclass(frozen=True)
class Family:
    id: str
    kind: str  # image | video
    pipelines: dict[str, str]  # task -> diffusers pipeline class
    denoisers: tuple[str, ...] = ("transformer",)
    text_encoders: tuple[str, ...] = ("text_encoder",)
    guidance_arg: str = "guidance_scale"  # or "true_cfg_scale" (Qwen)
    negative_prompt: bool = True
    lora: bool = True
    # rough activation memory (GB) for one image at ~1MP / one default clip
    activation_gb: float = 2.5
    multiple_of: int = 16  # width/height granularity
    vae_fp32: bool = False
    prompt_note: str = ""  # i18n key shown next to the prompt box


FAMILIES: dict[str, Family] = {
    f.id: f
    for f in [
        Family(
            "sd15",
            "image",
            {
                "txt2img": "StableDiffusionPipeline",
                "img2img": "StableDiffusionImg2ImgPipeline",
                "inpaint": "StableDiffusionInpaintPipeline",
            },
            denoisers=("unet",),
            activation_gb=1.5,
            multiple_of=8,
            prompt_note="english_prompts",
        ),
        Family(
            "sdxl",
            "image",
            {
                "txt2img": "StableDiffusionXLPipeline",
                "img2img": "StableDiffusionXLImg2ImgPipeline",
                "inpaint": "StableDiffusionXLInpaintPipeline",
            },
            denoisers=("unet",),
            text_encoders=("text_encoder", "text_encoder_2"),
            activation_gb=2.5,
            multiple_of=8,
            prompt_note="english_prompts",
        ),
        Family(
            "flux1",
            "image",
            {"txt2img": "FluxPipeline", "img2img": "FluxImg2ImgPipeline", "inpaint": "FluxInpaintPipeline"},
            text_encoders=("text_encoder", "text_encoder_2"),
            negative_prompt=False,
            activation_gb=2.5,
        ),
        Family(
            "flux1_kontext",
            "image",
            {"edit": "FluxKontextPipeline", "txt2img": "FluxKontextPipeline"},
            text_encoders=("text_encoder", "text_encoder_2"),
            negative_prompt=False,
            activation_gb=2.5,
        ),
        Family(
            "flux1_fill",
            "image",
            {"inpaint": "FluxFillPipeline"},
            text_encoders=("text_encoder", "text_encoder_2"),
            negative_prompt=False,
            activation_gb=2.5,
        ),
        Family(
            "flux2_klein",
            "image",
            {
                "txt2img": "Flux2KleinPipeline",
                "edit": "Flux2KleinPipeline",
                "img2img": "Flux2KleinPipeline",
                "inpaint": "Flux2KleinInpaintPipeline",
            },
            negative_prompt=False,
            activation_gb=2.5,
        ),
        Family(
            "z_image",
            "image",
            {
                "txt2img": "ZImagePipeline",
                "img2img": "ZImageImg2ImgPipeline",
                "inpaint": "ZImageInpaintPipeline",
            },
            activation_gb=2.5,
        ),
        Family(
            "qwen_image",
            "image",
            {
                "txt2img": "QwenImagePipeline",
                "img2img": "QwenImageImg2ImgPipeline",
                "inpaint": "QwenImageInpaintPipeline",
            },
            guidance_arg="true_cfg_scale",
            activation_gb=2.5,
        ),
        Family(
            "qwen_image_edit",
            "image",
            {"edit": "QwenImageEditPlusPipeline"},
            guidance_arg="true_cfg_scale",
            activation_gb=3.0,
        ),
        Family(
            "qwen_image_21",
            "image",
            {"txt2img": "QwenImage21Pipeline", "edit": "QwenImage21Pipeline"},
            guidance_arg="true_cfg_scale",
            activation_gb=2.5,
        ),
        Family(
            "wan_t2v",
            "video",
            {"txt2vid": "WanPipeline", "vid2vid": "WanVideoToVideoPipeline"},
            activation_gb=2.0,
            vae_fp32=True,
        ),
        Family(
            "wan_i2v",
            "video",
            {"img2vid": "WanImageToVideoPipeline"},
            activation_gb=3.5,
            vae_fp32=True,
        ),
        Family(
            "wan_ti2v",
            "video",
            {"txt2vid": "WanPipeline", "img2vid": "WanImageToVideoPipeline"},
            activation_gb=4.0,
            multiple_of=32,
            vae_fp32=True,
        ),
        Family(
            "wan_a14b_t2v",
            "video",
            {"txt2vid": "WanPipeline"},
            denoisers=("transformer", "transformer_2"),
            activation_gb=3.5,
            vae_fp32=True,
        ),
        Family(
            "wan_a14b_i2v",
            "video",
            {"img2vid": "WanImageToVideoPipeline"},
            denoisers=("transformer", "transformer_2"),
            activation_gb=3.5,
            vae_fp32=True,
        ),
    ]
}

def family_from_index(index: dict, name: str = "") -> tuple[str, str] | None:
    """(family, variant) for a diffusers pipeline folder, from its model_index.json."""
    cls = index.get("_class_name", "")
    lname = name.lower()
    has_expert = bool((index.get("transformer_2") or [None])[0])
    if cls == "WanPipeline":
        if has_expert:
            return "wan_a14b_t2v", ""
        if index.get("expand_timesteps"):
            return "wan_ti2v", "5b"
        return "wan_t2v", "1.3b" if "1.3b" in lname else "14b"
    if cls == "WanImageToVideoPipeline":
        if has_expert:
            return "wan_a14b_i2v", ""
        if index.get("expand_timesteps"):
            return "wan_ti2v", "5b"
        return "wan_i2v", "14b"
    fam = PIPELINE_CLASS_FAMILY.get(cls)
    if fam is None:
        return None
    variant = ""
    if fam.startswith("flux1"):
        variant = "schnell" if "schnell" in lname else "dev"
    elif fam == "flux2_klein":
        variant = "base" if "base" in lname else ""
    elif fam == "z_image":
        variant = "turbo" if "turbo" in lname else "base"
    return fam, variant


# diffusers pipeline class (model_index.json "_class_name") -> family
PIPELINE_CLASS_FAMILY = {
    "StableDiffusionPipeline": "sd15",
    "StableDiffusionXLPipeline": "sdxl",
    "FluxPipeline": "flux1",
    "FluxKontextPipeline": "flux1_kontext",
    "FluxFillPipeline": "flux1_fill",
    "Flux2KleinPipeline": "flux2_klein",
    "ZImagePipeline": "z_image",
    "QwenImagePipeline": "qwen_image",
    "QwenImageEditPlusPipeline": "qwen_image_edit",
    "QwenImageEditPipeline": "qwen_image_edit",
    "QwenImage21Pipeline": "qwen_image_21",
}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    repo: str
    license: str
    commercial: bool
    download_gb: float
    # bf16 size of the big components in GB, used for memory planning before download
    components_gb: dict[str, float]
    description: dict[str, str]
    defaults: dict[str, Any] = field(default_factory=dict)
    tasks: tuple[str, ...] = ()
    variant: str | None = None
    gated: bool = False
    recommended: bool = False
    pipeline_config: dict[str, Any] = field(default_factory=dict)  # e.g. scheduler flow shift
    tags: tuple[str, ...] = ()

    @property
    def fam(self) -> Family:
        return FAMILIES[self.family]

    @property
    def kind(self) -> str:
        return self.fam.kind

    def supported_tasks(self) -> tuple[str, ...]:
        return self.tasks or tuple(self.fam.pipelines)


CATALOG: list[ModelSpec] = [
    # ---------------------------------------------------------------- images
    ModelSpec(
        id="z-image-turbo",
        name="Z-Image Turbo",
        family="z_image",
        repo="Tongyi-MAI/Z-Image-Turbo",
        license="Apache 2.0",
        commercial=True,
        download_gb=20.5,
        components_gb={"transformer": 12.3, "text_encoder": 8.0, "vae": 0.2},
        description={
            "el": "Γρήγορο (8 βήματα), φωτορεαλιστικό, γράφει σωστά κείμενο. Η καλύτερη αρχή.",
            "en": "Fast (8 steps), photorealistic, renders text well. The best place to start.",
        },
        defaults={"steps": 9, "guidance": 0.0, "width": 1024, "height": 1024, "strength": 0.6},
        recommended=True,
        tags=("fast", "photo", "text"),
    ),
    ModelSpec(
        id="flux2-klein-4b",
        name="FLUX.2 [klein] 4B",
        family="flux2_klein",
        repo="black-forest-labs/FLUX.2-klein-4B",
        license="Apache 2.0",
        commercial=True,
        download_gb=16.0,
        components_gb={"transformer": 7.8, "text_encoder": 8.0, "vae": 0.2},
        description={
            "el": "Δημιουργία ΚΑΙ επεξεργασία με εικόνες αναφοράς σε ένα μοντέλο. 4 βήματα.",
            "en": "Generation AND reference-based editing in one model. 4 steps.",
        },
        defaults={"steps": 4, "guidance": 1.0, "width": 1024, "height": 1024, "strength": 0.8},
        recommended=True,
        tags=("fast", "edit"),
    ),
    ModelSpec(
        id="qwen-image-edit-2511",
        name="Qwen-Image-Edit 2511",
        family="qwen_image_edit",
        repo="Qwen/Qwen-Image-Edit-2511",
        license="Apache 2.0",
        commercial=True,
        download_gb=57.7,
        components_gb={"transformer": 40.9, "text_encoder": 16.6, "vae": 0.3},
        description={
            "el": "Κορυφαία επεξεργασία με οδηγίες («κάνε τον ουρανό ηλιοβασίλεμα»), κείμενο, συνδυασμός φωτογραφιών. Βαρύ: με 2 κάρτες τα στρώματά του μοιράζονται και στις δύο (8-bit)· σε μία κάρτα συμπιέζεται σε 4-bit.",
            "en": "Top instruction editing, text editing, multi-photo compositing. Heavy: with two cards its layers are spread over both (8-bit); on one card it is compressed to 4-bit.",
        },
        defaults={"steps": 40, "guidance": 4.0},
        recommended=True,
        tags=("edit", "heavy"),
    ),
    ModelSpec(
        id="qwen-image-2512",
        name="Qwen-Image 2512",
        family="qwen_image",
        repo="Qwen/Qwen-Image-2512",
        license="Apache 2.0",
        commercial=True,
        download_gb=57.7,
        components_gb={"transformer": 40.9, "text_encoder": 16.6, "vae": 0.3},
        description={
            "el": "Πολύ υψηλή ποιότητα και άψογο κείμενο μέσα στην εικόνα. Βαρύ: με 2 κάρτες τα στρώματά του μοιράζονται και στις δύο (8-bit).",
            "en": "Very high quality and excellent in-image text. Heavy: with two cards its layers are spread over both (8-bit).",
        },
        defaults={"steps": 40, "guidance": 4.0, "width": 1328, "height": 1328, "strength": 0.6},
        tags=("text", "heavy"),
    ),
    ModelSpec(
        id="qwen-image-2.1",
        name="Qwen-Image 2.1",
        family="qwen_image_21",
        repo="Qwen/Qwen-Image-2.1",
        license="Qwen Research (μη εμπορική)",
        commercial=False,
        download_gb=32.0,
        components_gb={"transformer": 14.0, "text_encoder": 17.5, "vae": 0.3},
        description={
            "el": "Νέο (2026): δημιουργία και επεξεργασία έως 10 εικόνων αναφοράς, διάφανο φόντο. Μόνο για προσωπική χρήση.",
            "en": "New (2026): generation and editing with up to 10 references, transparency. Personal use only.",
        },
        defaults={"steps": 40, "guidance": 1.0, "width": 1024, "height": 1024},
        tags=("edit", "new"),
    ),
    ModelSpec(
        id="sdxl-base",
        name="Stable Diffusion XL",
        family="sdxl",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        license="OpenRAIL++",
        commercial=True,
        download_gb=6.9,
        components_gb={"unet": 5.1, "text_encoder": 0.25, "text_encoder_2": 1.4, "vae": 0.2},
        description={
            "el": "Κλασικό, ελαφρύ, με τεράστια συλλογή από LoRA. Γράφε τις περιγραφές στα αγγλικά.",
            "en": "Classic and light with a huge LoRA ecosystem. Write prompts in English.",
        },
        defaults={"steps": 30, "guidance": 6.0, "width": 1024, "height": 1024, "strength": 0.6},
        variant="fp16",
        tags=("light", "lora"),
    ),
    ModelSpec(
        id="flux1-kontext-dev",
        name="FLUX.1 Kontext [dev]",
        family="flux1_kontext",
        repo="black-forest-labs/FLUX.1-Kontext-dev",
        license="FLUX.1 [dev] (μη εμπορική)",
        commercial=False,
        download_gb=34.0,
        components_gb={"transformer": 23.8, "text_encoder": 0.25, "text_encoder_2": 9.5, "vae": 0.2},
        description={
            "el": "Επεξεργασία με οδηγίες από τη Black Forest Labs. Χρειάζεται αποδοχή άδειας στο Hugging Face.",
            "en": "Instruction editing by Black Forest Labs. Requires accepting the license on Hugging Face.",
        },
        defaults={"steps": 28, "guidance": 2.5},
        gated=True,
        tags=("edit",),
    ),
    # ---------------------------------------------------------------- video
    ModelSpec(
        id="wan2.2-ti2v-5b",
        name="Wan 2.2 TI2V 5B",
        family="wan_ti2v",
        repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        license="Apache 2.0",
        commercial=True,
        download_gb=34.0,
        components_gb={"transformer": 10.0, "text_encoder": 11.4, "vae": 1.4},
        description={
            "el": "Βίντεο 720p/24fps από κείμενο ή από φωτογραφία. Χωράει σε μία 16GB κάρτα· η άλλη μένει ελεύθερη για εικόνες.",
            "en": "720p/24fps video from text or from a photo. Fits on one 16GB GPU, leaving the other free for images.",
        },
        defaults={"steps": 40, "guidance": 5.0, "width": 1280, "height": 704, "fps": 24, "frames": 121},
        recommended=True,
        tags=("video",),
    ),
    ModelSpec(
        id="wan2.1-t2v-1.3b",
        name="Wan 2.1 T2V 1.3B",
        family="wan_t2v",
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        license="Apache 2.0",
        commercial=True,
        download_gb=29.0,
        components_gb={"transformer": 2.8, "text_encoder": 11.4, "vae": 0.5},
        description={
            "el": "Ελαφρύ και γρήγορο βίντεο 480p, για δοκιμές. Κάνει και «restyle» σε υπάρχον βίντεο.",
            "en": "Light, fast 480p video for drafts. Also restyles existing videos.",
        },
        defaults={"steps": 30, "guidance": 5.0, "width": 832, "height": 480, "fps": 16, "frames": 81, "strength": 0.7},
        pipeline_config={"flow_shift": 3.0},
        tags=("video", "light"),
    ),
    ModelSpec(
        id="wan2.2-i2v-a14b",
        name="Wan 2.2 I2V A14B",
        family="wan_a14b_i2v",
        repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        license="Apache 2.0",
        commercial=True,
        download_gb=69.0,
        components_gb={"transformer": 28.6, "transformer_2": 28.6, "text_encoder": 11.4, "vae": 0.5},
        description={
            "el": "Κορυφαία κίνηση από φωτογραφία. Πολύ βαρύ: τρέχει συμπιεσμένο (4-bit)· όταν είναι ελεύθερες και οι δύο κάρτες, βάζει έναν από τους δύο «ειδικούς» του σε κάθε κάρτα.",
            "en": "Top-quality photo animation. Very heavy: runs compressed (4-bit); when both cards are free, each holds one of its two experts.",
        },
        defaults={"steps": 40, "guidance": 3.5, "width": 832, "height": 480, "fps": 16, "frames": 81},
        tags=("video", "heavy"),
    ),
    ModelSpec(
        id="wan2.2-t2v-a14b",
        name="Wan 2.2 T2V A14B",
        family="wan_a14b_t2v",
        repo="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        license="Apache 2.0",
        commercial=True,
        download_gb=69.0,
        components_gb={"transformer": 28.6, "transformer_2": 28.6, "text_encoder": 11.4, "vae": 0.5},
        description={
            "el": "Κορυφαία ποιότητα βίντεο από κείμενο. Πολύ βαρύ: τρέχει συμπιεσμένο (4-bit)· όταν είναι ελεύθερες και οι δύο κάρτες, βάζει έναν από τους δύο «ειδικούς» του σε κάθε κάρτα.",
            "en": "Top-quality text-to-video. Very heavy: runs compressed (4-bit); when both cards are free, each holds one of its two experts.",
        },
        defaults={"steps": 40, "guidance": 4.0, "guidance_2": 3.0, "width": 832, "height": 480, "fps": 16, "frames": 81},
        tags=("video", "heavy"),
    ),
]

CATALOG_BY_ID = {m.id: m for m in CATALOG}
CATALOG_BY_REPO = {m.repo.lower(): m for m in CATALOG}


# --------------------------------------------------------------------- tool models
@dataclass(frozen=True)
class ToolModel:
    id: str
    kind: str  # upscaler | background | eraser
    name: str
    url: str
    filename: str
    size_mb: int
    md5: str | None = None
    scale: int | None = None
    description: dict[str, str] = field(default_factory=dict)


TOOL_MODELS: list[ToolModel] = [
    ToolModel(
        "realesrgan-x4",
        "upscaler",
        "Real-ESRGAN x4 (φωτογραφίες)",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus.pth",
        64,
        scale=4,
    ),
    ToolModel(
        "realesrgan-x2",
        "upscaler",
        "Real-ESRGAN x2 (φωτογραφίες)",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x2plus.pth",
        64,
        scale=2,
    ),
    ToolModel(
        "realesrgan-anime-x4",
        "upscaler",
        "Real-ESRGAN Anime x4 (σχέδια)",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "RealESRGAN_x4plus_anime_6B.pth",
        17,
        scale=4,
    ),
    ToolModel(
        "realesr-general-x4",
        "upscaler",
        "Real-ESRGAN General x4 (γρήγορο, για βίντεο)",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "realesr-general-x4v3.pth",
        5,
        scale=4,
    ),
    ToolModel(
        "birefnet",
        "background",
        "BiRefNet (καλύτερη ποιότητα)",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
        "birefnet-general.onnx",
        973,
        md5="7a35a0141cbbc80de11d9c9a28f52697",
    ),
    ToolModel(
        "birefnet-lite",
        "background",
        "BiRefNet Lite (γρήγορο)",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "birefnet-general-lite.onnx",
        224,
        md5="4fab47adc4ff364be1713e97b7e66334",
    ),
    ToolModel(
        "isnet",
        "background",
        "ISNet (ελαφρύ)",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        "isnet-general-use.onnx",
        179,
        md5="fc16ebd8b0c10d971d3513d564d01e29",
    ),
    ToolModel(
        "lama",
        "eraser",
        "LaMa (γόμα αντικειμένων)",
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "big-lama.pt",
        196,
    ),
]

TOOLS_BY_ID = {t.id: t for t in TOOL_MODELS}
