"""
LTX-Video pipeline factory.

Supports both text-to-video (LTXPipeline) and image-to-video
(LTXImageToVideoPipeline) via the image_conditioning flag.

Model cached at .hf-cache/hub/models--Lightricks--LTX-Video.
Same CPU-offload and VAE-slicing guards as WanPipelineFactory.

Pipeline objects themselves are also cached at module level (see
`get_cached_pipeline`) so a 24/7 loop pays the weight-loading cost
once per process rather than on every clip.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

LTX_MODEL_ID = "Lightricks/LTX-Video"

# LTX-specific defaults — tuned for Lightricks/LTX-Video, NOT for Wan 2.1.
# The Wan defaults in generate_video() (Chinese negative prompt, fps=16,
# guidance=5.0) produce visibly worse output on LTX.
#
# These values match scripts/generate_ltx.py which was battle-tested against
# the Unity pipeline — keep them aligned.
LTX_DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "static, no movement, low quality, watermark, text, subtitle"
)
LTX_DEFAULT_GUIDANCE_SCALE = 3.0
LTX_DEFAULT_NUM_INFERENCE_STEPS = 50
LTX_DEFAULT_FPS = 25
LTX_DEFAULT_HEIGHT = 512
LTX_DEFAULT_WIDTH = 768
# LTX VAE temporal compression = 8, so num_frames must satisfy (n-1) % 8 == 0.
# Common values: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 97, 121, 161, 201
# At 25fps: 121 ≈ 4.84s, 161 ≈ 6.44s, 201 = 8s
LTX_DEFAULT_NUM_FRAMES = 121


class LTXPipelineFactory:
    """
    Creates LTX-Video pipeline instances.

    Args:
        model_id: HuggingFace model ID (default: Lightricks/LTX-Video)
        cache_dir: HuggingFace cache directory
        image_conditioning: If True, loads LTXImageToVideoPipeline (i2v);
                            if False, loads LTXPipeline (t2v)
        enable_model_cpu_offload: Offload model components to CPU between uses
        enable_vae_slicing: Process VAE batches one sample at a time (low VRAM)
        enable_vae_tiling: Process VAE spatially in tiles (required for ≥768px)
    """

    def __init__(
        self,
        model_id: str = LTX_MODEL_ID,
        cache_dir: Optional[str] = None,
        image_conditioning: bool = False,
        enable_model_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.image_conditioning = image_conditioning
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling

    def build(self):
        """Load and return the LTX pipeline (t2v or i2v based on image_conditioning)."""
        import torch
        if self.image_conditioning:
            from diffusers import LTXImageToVideoPipeline
            pipeline_cls = LTXImageToVideoPipeline
            logger.info("Loading LTXImageToVideoPipeline (i2v) [%s]", self.model_id)
        else:
            from diffusers import LTXPipeline
            pipeline_cls = LTXPipeline
            logger.info("Loading LTXPipeline (t2v) [%s]", self.model_id)

        # Pre-load tokenizer with use_fast=False to bypass the transformers >= 4.40
        # tiktoken conversion path that misroutes LTX-Video's SentencePiece spiece.model.
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, use_fast=False
        )
        pipe = pipeline_cls.from_pretrained(
            self.model_id,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
        )

        # VAE slicing — guard against API differences across diffusers versions
        if self.enable_vae_slicing:
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

        # VAE tiling — required for ≥768px output on 12GB VRAM
        if self.enable_vae_tiling:
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

        if self.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        return pipe


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------
# A single LTX pipeline occupies ~10GB of VRAM even with CPU offload. Rebuilding
# it per clip adds 10–60s of disk I/O each time. The cache holds at most one
# t2v and one i2v pipeline (keyed by model_id + image_conditioning) for the
# lifetime of the process.

_PIPELINE_CACHE: dict = {}


def get_cached_pipeline(
    model_id: str = LTX_MODEL_ID,
    cache_dir: Optional[str] = None,
    image_conditioning: bool = False,
    enable_model_cpu_offload: bool = True,
    enable_vae_slicing: bool = True,
):
    """
    Return a cached LTX pipeline, building it on first call.

    Keyed by (model_id, image_conditioning). A 24/7 generation loop pays the
    weight-loading cost once per process rather than on every clip.
    """
    key = (model_id, image_conditioning)
    pipe = _PIPELINE_CACHE.get(key)
    if pipe is None:
        pipe = LTXPipelineFactory(
            model_id=model_id,
            cache_dir=cache_dir,
            image_conditioning=image_conditioning,
            enable_model_cpu_offload=enable_model_cpu_offload,
            enable_vae_slicing=enable_vae_slicing,
        ).build()
        _PIPELINE_CACHE[key] = pipe
        logger.info("LTX pipeline cached: %s (i2v=%s)", model_id, image_conditioning)
    return pipe


def offload_pipeline_to_cpu() -> None:
    """Move cached pipeline components to CPU to free VRAM for another model.

    Unlike clear_pipeline_cache(), this keeps the pipeline objects alive in
    _PIPELINE_CACHE so the next call to get_cached_pipeline() skips the disk
    reload entirely — the model just moves CPU→GPU via offload hooks (~seconds).

    Use this between LTX inference and a VLM eval that needs the same VRAM.
    """
    import gc
    try:
        import torch
        for pipe in _PIPELINE_CACHE.values():
            # Transformer is the bulk of VRAM (~8 GB). Move it to CPU.
            if hasattr(pipe, "transformer") and pipe.transformer is not None:
                pipe.transformer.to("cpu")
            # VAE and text_encoder should already be CPU-offloaded, but be safe.
            if hasattr(pipe, "vae") and pipe.vae is not None:
                pipe.vae.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LTX pipeline offloaded to CPU (cache preserved)")
    except Exception as e:
        logger.warning("offload_pipeline_to_cpu failed: %s", e)


def clear_pipeline_cache() -> None:
    """Release cached pipelines and free VRAM. Call before switching models."""
    import gc
    _PIPELINE_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
