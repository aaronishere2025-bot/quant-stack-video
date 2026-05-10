"""
Wan 2.1 pipeline factory for quantization stacking.

Wan 2.1 (Wan Video) is Alibaba's open-source video generation model.
HuggingFace model IDs:
  - Wan-AI/Wan2.1-T2V-14B-Diffusers  (text-to-video, 14B params)
  - Wan-AI/Wan2.1-T2V-1.3B-Diffusers (text-to-video, 1.3B params — dev/testing)
  - Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (image-to-video)

The factory creates a fresh pipeline instance per quantization config,
which allows the engine to load/unload models across passes to manage VRAM.

Engines:
  - "bnb" (default): BitsAndBytes 4-bit/8-bit quantization (NF4/FP4)
  - "gguf": GGUF Q4_0/Q3_K_M via diffusers GGUFQuantizationConfig +
             WanTransformer3DModel.from_single_file(). Requires a local
             .gguf file path passed as gguf_model_path.
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

# Known community GGUF repos for Wan 2.1 14B (Q4_0 and Q3_K_M)
GGUF_HF_REPO = "bartowski/Wan2.1-T2V-14B-GGUF"
GGUF_Q4_0_FILENAME = "Wan2.1-T2V-14B-Q4_0.gguf"
GGUF_Q3_KM_FILENAME = "Wan2.1-T2V-14B-Q3_K_M.gguf"


class WanPipelineFactory:
    """
    Creates WanPipeline instances with specified quantization config.

    Supports:
    - Full precision (bf16) — baseline reference
    - 4-bit quantization via BitsAndBytes (NF4 or FP4)  [engine="bnb"]
    - 8-bit quantization via BitsAndBytes               [engine="bnb"]
    - GGUF Q4_0/Q3_K_M via diffusers from_single_file  [engine="gguf"]

    Example (BnB):
        factory = WanPipelineFactory(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        pipe = factory(quant_config)

    Example (GGUF):
        factory = WanPipelineFactory(
            model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            engine="gguf",
            gguf_model_path="/path/to/wan2.1-Q4_0.gguf",
        )
        pipe = factory(None)  # quant_config unused for GGUF
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
        engine: str = "bnb",
        gguf_model_path: Optional[str] = None,
        gguf_compute_dtype: str = "bfloat16",
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.text_encoder_precision = text_encoder_precision
        self.vae_precision = vae_precision
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling
        self.engine = engine
        self.gguf_model_path = gguf_model_path
        self.gguf_compute_dtype = gguf_compute_dtype

        if engine == "gguf" and gguf_model_path is None:
            raise ValueError(
                "engine='gguf' requires gguf_model_path. "
                f"Download a GGUF file from {GGUF_HF_REPO} on HuggingFace."
            )

    def __call__(self, quant_config: Optional[QuantConfig]):
        """Build and return a WanPipeline with the given quantization config."""
        if self.engine == "gguf":
            return self._build_gguf_pipeline()
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

        return self._apply_memory_opts(pipe)

    def _build_gguf_pipeline(self):
        """
        Build a WanPipeline with the transformer loaded from a GGUF file.

        Uses diffusers' GGUFQuantizationConfig + WanTransformer3DModel.from_single_file().
        VAE and text encoder are loaded from the base HF model in bfloat16 — they must
        not be quantized (bfloat16 VAE = required for quality; quantized VAE = artifacts).

        VRAM profile (14B Q4_0): ~7–8 GB transformer + ~1 GB VAE + T5 on CPU ≈ 9 GB.
        """
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(self.gguf_compute_dtype, torch.bfloat16)
        text_dtype = dtype_map.get(self.text_encoder_precision, torch.bfloat16)
        vae_dtype = dtype_map.get(self.vae_precision, torch.bfloat16)

        gguf_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        logger.info(
            f"Loading GGUF transformer from {self.gguf_model_path} "
            f"compute_dtype={self.gguf_compute_dtype}"
        )
        transformer = WanTransformer3DModel.from_single_file(
            self.gguf_model_path,
            quantization_config=gguf_config,
            torch_dtype=compute_dtype,
            config=self.model_id,
        )

        logger.info("Loading text encoder (bf16, CPU)...")
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

        logger.info("Assembling GGUF pipeline...")
        pipe = WanPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
            torch_dtype=compute_dtype,
            cache_dir=self.cache_dir,
        )

        return self._apply_memory_opts(pipe)

    def _apply_memory_opts(self, pipe):
        """Apply VAE slicing, tiling, and CPU offload settings to a pipeline."""
        if self.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif self.enable_vae_slicing and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if self.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        elif self.enable_vae_tiling and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        if self.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        elif self.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        return pipe

    def build_reference(self):
        """Build a full-precision (bf16) reference pipeline for quality comparison."""
        ref_config = QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False)
        return self._build_pipeline(ref_config)
