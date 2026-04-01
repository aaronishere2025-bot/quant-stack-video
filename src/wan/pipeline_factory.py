"""
Wan 2.1 pipeline factory for quantization stacking.

Wan 2.1 (Wan Video) is Alibaba's open-source video generation model.
HuggingFace model IDs:
  - Wan-AI/Wan2.1-T2V-14B-Diffusers  (text-to-video, 14B params)
  - Wan-AI/Wan2.1-T2V-1.3B-Diffusers (text-to-video, 1.3B params — dev/testing)
  - Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (image-to-video)

The factory creates a fresh pipeline instance per quantization config,
which allows the engine to load/unload models across passes to manage VRAM.
"""

import logging
from typing import Optional

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from transformers import AutoTokenizer, T5EncoderModel

from ..quant.config import QuantConfig

logger = logging.getLogger(__name__)

# Default Wan 2.1 model IDs
WAN_14B_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
WAN_1_3B_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class WanPipelineFactory:
    """
    Creates WanPipeline instances with specified quantization config.

    Supports:
    - Full precision (bf16) — baseline reference
    - 4-bit quantization via BitsAndBytes (NF4 or FP4)
    - 8-bit quantization via BitsAndBytes
    - Mixed precision: text encoder in different precision than transformer

    Example:
        factory = WanPipelineFactory(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        pipe = factory(quant_config)
    """

    def __init__(
        self,
        model_id: str = WAN_1_3B_MODEL_ID,
        cache_dir: Optional[str] = None,
        text_encoder_precision: str = "bfloat16",
        vae_precision: str = "bfloat16",
        enable_model_cpu_offload: bool = True,
        enable_sequential_cpu_offload: bool = False,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.text_encoder_precision = text_encoder_precision
        self.vae_precision = vae_precision
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling

    def __call__(self, quant_config: QuantConfig):
        """Build and return a WanPipeline with the given quantization config."""
        return self._build_pipeline(quant_config)

    def _build_pipeline(self, qcfg: QuantConfig):
        """
        Build a WanPipeline with the transformer quantized to 4-bit.

        Architecture rationale:
        - The transformer (DiT) is the largest component (~14B params) → quantize this
        - The VAE and text encoder are smaller and kept in higher precision
          to preserve perceptual quality and text alignment
        """
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        text_dtype = dtype_map.get(self.text_encoder_precision, torch.bfloat16)
        vae_dtype = dtype_map.get(self.vae_precision, torch.bfloat16)
        transformer_dtype = dtype_map.get(qcfg.torch_dtype, torch.bfloat16)

        bnb_config = qcfg.to_bnb_config()

        logger.info(f"Loading Wan 2.1 pipeline [{self.model_id}] "
                    f"quant={qcfg.quant_type} "
                    f"bnb_4bit_quant_type={qcfg.bnb_4bit_quant_type} "
                    f"double_quant={qcfg.bnb_4bit_use_double_quant}")

        load_kwargs = dict(
            torch_dtype=transformer_dtype,
            cache_dir=self.cache_dir,
        )

        if bnb_config is not None:
            # Load transformer with quantization; other components in higher precision
            from diffusers import WanTransformer3DModel

            logger.info("Loading quantized transformer...")
            transformer = WanTransformer3DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                quantization_config=bnb_config,
                torch_dtype=transformer_dtype,
                cache_dir=self.cache_dir,
            )

            logger.info("Loading text encoder (bf16)...")
            text_encoder = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                torch_dtype=text_dtype,
                cache_dir=self.cache_dir,
            )

            logger.info("Loading VAE (bf16)...")
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id,
                subfolder="vae",
                torch_dtype=vae_dtype,
                cache_dir=self.cache_dir,
            )

            logger.info("Assembling pipeline...")
            pipe = WanPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                text_encoder=text_encoder,
                vae=vae,
                torch_dtype=transformer_dtype,
                cache_dir=self.cache_dir,
            )
        else:
            # Full precision (reference baseline)
            logger.info(f"Loading full-precision pipeline ({transformer_dtype})...")
            pipe = WanPipeline.from_pretrained(
                self.model_id,
                torch_dtype=transformer_dtype,
                cache_dir=self.cache_dir,
            )

        # Memory optimization (guard against API differences across diffusers versions)
        if self.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif self.enable_vae_slicing and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if self.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        elif self.enable_vae_tiling and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        # CPU offloading strategy
        if self.enable_sequential_cpu_offload:
            # Most VRAM-efficient, slowest
            pipe.enable_sequential_cpu_offload()
        elif self.enable_model_cpu_offload:
            # Moderate VRAM efficiency, faster than sequential
            pipe.enable_model_cpu_offload()
        else:
            # No offloading: fastest, requires most VRAM
            pipe.to("cuda")

        return pipe

    def build_reference(self):
        """Build a full-precision (bf16) reference pipeline for quality comparison."""
        ref_config = QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False)
        return self._build_pipeline(ref_config)
