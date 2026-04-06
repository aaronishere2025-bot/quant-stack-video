"""
LTX-Video pipeline factory.

Supports both text-to-video (LTXPipeline) and image-to-video
(LTXImageToVideoPipeline) via the image_conditioning flag.

Model cached at .hf-cache/hub/models--Lightricks--LTX-Video.
Same CPU-offload and VAE-slicing guards as WanPipelineFactory.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

LTX_MODEL_ID = "Lightricks/LTX-Video"


class LTXPipelineFactory:
    """
    Creates LTX-Video pipeline instances.

    Args:
        model_id: HuggingFace model ID (default: Lightricks/LTX-Video)
        cache_dir: HuggingFace cache directory
        image_conditioning: If True, loads LTXImageToVideoPipeline (i2v);
                            if False, loads LTXPipeline (t2v)
        enable_model_cpu_offload: Offload model components to CPU between uses
        enable_vae_slicing: Process VAE in slices to reduce VRAM peak
    """

    def __init__(
        self,
        model_id: str = LTX_MODEL_ID,
        cache_dir: Optional[str] = None,
        image_conditioning: bool = False,
        enable_model_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.image_conditioning = image_conditioning
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing

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

        pipe = pipeline_cls.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
        )

        # VAE slicing — guard against API differences across diffusers versions
        if self.enable_vae_slicing:
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

        if self.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        return pipe
