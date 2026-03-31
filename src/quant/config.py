"""Configuration dataclasses for quantization stacking."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class QuantConfig:
    """Configuration for a single quantization pass."""

    # Quantization type: "4bit", "8bit", "none" (fp16/bf16)
    quant_type: str = "4bit"

    # BitsAndBytes config
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"  # compute dtype during forward pass
    bnb_4bit_quant_type: str = "nf4"          # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True     # nested quantization

    # Inference dtype for non-quant layers
    torch_dtype: str = "bfloat16"

    # Device placement
    device_map: str = "auto"

    # Layer-selective quantization: skip these module patterns
    llm_int8_skip_modules: Optional[List[str]] = None

    def to_bnb_config(self):
        """Convert to BitsAndBytesConfig."""
        import torch
        from transformers import BitsAndBytesConfig

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(self.bnb_4bit_compute_dtype, torch.bfloat16)

        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                llm_int8_skip_modules=self.llm_int8_skip_modules,
            )
        elif self.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=self.llm_int8_skip_modules,
            )
        return None


@dataclass
class StackConfig:
    """Configuration for multi-pass quantization stacking."""

    # Number of 4-bit passes to stack
    num_passes: int = 3

    # Quant config for each pass (if empty, uses defaults)
    pass_configs: List[QuantConfig] = field(default_factory=list)

    # Stacking strategy: "average", "weighted", "residual", "progressive"
    # - average: simple mean of latent outputs
    # - weighted: learned or fixed weights per pass
    # - residual: each pass refines the residual error of the previous
    # - progressive: iterative denoising where each pass refines the previous
    stacking_strategy: str = "progressive"

    # Weights for "weighted" strategy (must sum to 1.0)
    pass_weights: Optional[List[float]] = None

    # Guidance scale for refinement passes (progressive strategy)
    refinement_guidance_scale: float = 7.5

    # Noise level added between progressive passes (0.0 = no re-noise)
    inter_pass_noise_level: float = 0.05

    # Reference precision for quality comparison
    reference_dtype: str = "bfloat16"

    def __post_init__(self):
        if not self.pass_configs:
            # Default: vary quant type across passes for diversity
            quant_types = ["nf4", "fp4", "nf4"]
            double_quant = [True, False, True]
            self.pass_configs = [
                QuantConfig(
                    bnb_4bit_quant_type=quant_types[i % len(quant_types)],
                    bnb_4bit_use_double_quant=double_quant[i % len(double_quant)],
                )
                for i in range(self.num_passes)
            ]
        if self.pass_weights is None and self.stacking_strategy == "weighted":
            # Uniform weights by default
            self.pass_weights = [1.0 / self.num_passes] * self.num_passes
