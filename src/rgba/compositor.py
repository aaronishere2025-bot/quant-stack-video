"""
RGBA Alpha Compositor — Phase 2 of the layered infinite video pipeline.

Composites 3 RGBA video layers (background, midground, foreground) using
the Porter-Duff "over" operation.

Tensor convention: [B, 4, F, H, W], float32, values in [0, 1]
  - channels 0:3 = RGB
  - channel 3    = alpha matte
"""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class LayerSet:
    """Three RGBA video layers for compositing."""
    background: "torch.Tensor"   # [B, 4, F, H, W]
    midground: "torch.Tensor"    # [B, 4, F, H, W]
    foreground: "torch.Tensor"   # [B, 4, F, H, W]


def composite_over(top: "torch.Tensor", bottom: "torch.Tensor") -> "torch.Tensor":
    """
    Porter-Duff "over" compositing operation.

    Args:
        top:    [B, 4, F, H, W] RGBA tensor (front layer), values in [0, 1]
        bottom: [B, 4, F, H, W] RGBA tensor (back layer), values in [0, 1]

    Returns:
        [B, 4, F, H, W] composited RGBA tensor
    """
    import torch

    if top.shape != bottom.shape:
        raise ValueError(
            f"Layer shape mismatch: top={top.shape}, bottom={bottom.shape}. "
            "All layers must have identical [B, 4, F, H, W] shapes."
        )
    if top.shape[1] != 4:
        raise ValueError(f"Expected 4-channel RGBA tensors, got {top.shape[1]} channels")

    top_rgb = top[:, :3]    # [B, 3, F, H, W]
    top_a   = top[:, 3:4]   # [B, 1, F, H, W]
    bot_rgb = bottom[:, :3]
    bot_a   = bottom[:, 3:4]

    # Porter-Duff "over": out_a = a_top + a_bot * (1 - a_top)
    out_a = top_a + bot_a * (1.0 - top_a)

    # Premultiplied-alpha blend, guarded against divide-by-zero
    out_rgb = (top_rgb * top_a + bot_rgb * bot_a * (1.0 - top_a)) / (out_a + 1e-8)

    return torch.cat([out_rgb, out_a], dim=1)


def smooth_alpha(layer: "torch.Tensor", kernel_size: int = 3) -> "torch.Tensor":
    """
    Apply temporal smoothing to the alpha channel to reduce edge boiling.

    Boiling is caused by per-frame alpha jitter in regions near matte boundaries.
    A small temporal box filter suppresses this without blurring the RGB content.

    Args:
        layer:       [B, 4, F, H, W] RGBA tensor
        kernel_size: temporal window (must be odd, >= 1)

    Returns:
        [B, 4, F, H, W] with smoothed alpha channel
    """
    import torch
    import torch.nn.functional as F

    if kernel_size <= 1:
        return layer

    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    B, C, F_len, H, W = layer.shape
    rgb = layer[:, :3]  # [B, 3, F, H, W]
    alpha = layer[:, 3:4]  # [B, 1, F, H, W]

    # Reshape alpha for 1D temporal convolution: [B*H*W, 1, F]
    # Transpose to [B, H, W, 1, F] then reshape
    alpha_t = alpha.permute(0, 3, 4, 1, 2).reshape(B * H * W, 1, F_len)

    pad = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, device=layer.device, dtype=layer.dtype) / kernel_size
    smoothed = F.conv1d(alpha_t, weight, padding=pad)  # [B*H*W, 1, F]

    # Reshape back to [B, 1, F, H, W]
    smoothed = smoothed.reshape(B, H, W, 1, F_len).permute(0, 3, 4, 1, 2)

    return torch.cat([rgb, smoothed], dim=1)


class AlphaCompositor:
    """
    Composites three RGBA video layers into a single RGB output.

    Pipeline:
      1. Optionally smooth each layer's alpha to suppress edge boiling
      2. Composite background → midground → foreground (bottom to top)
      3. Return final RGB result (alpha channel dropped for video output)
    """

    def __init__(self, smooth_alpha_frames: bool = True, alpha_kernel_size: int = 3):
        """
        Args:
            smooth_alpha_frames: Apply temporal alpha smoothing before compositing
            alpha_kernel_size:   Temporal kernel size for alpha smoothing (must be odd)
        """
        self.smooth_alpha_frames = smooth_alpha_frames
        self.alpha_kernel_size = alpha_kernel_size

    def composite(self, layers: LayerSet) -> "torch.Tensor":
        """
        Composite three RGBA layers into a final RGB video tensor.

        Args:
            layers: LayerSet with background, midground, foreground tensors
                    each [B, 4, F, H, W], float32, values in [0, 1]

        Returns:
            [B, 3, F, H, W] RGB tensor, values in [0, 1]
        """
        bg = layers.background
        mg = layers.midground
        fg = layers.foreground

        if self.smooth_alpha_frames:
            bg = smooth_alpha(bg, self.alpha_kernel_size)
            mg = smooth_alpha(mg, self.alpha_kernel_size)
            fg = smooth_alpha(fg, self.alpha_kernel_size)
            logger.debug("Applied temporal alpha smoothing (kernel=%d)", self.alpha_kernel_size)

        # Composite bottom-to-top: bg first, then mg over it, then fg over that
        comp = composite_over(top=mg, bottom=bg)
        comp = composite_over(top=fg, bottom=comp)

        # Drop alpha — video output is RGB
        rgb = comp[:, :3]
        logger.debug("Composited 3 RGBA layers → RGB shape %s", list(rgb.shape))
        return rgb

    def composite_layers(self, layer_list: List["torch.Tensor"]) -> "torch.Tensor":
        """
        Composite an arbitrary ordered list of RGBA layers (bottom to top).

        Args:
            layer_list: List of [B, 4, F, H, W] tensors, index 0 = bottom

        Returns:
            [B, 3, F, H, W] RGB tensor
        """
        if len(layer_list) < 1:
            raise ValueError("Need at least one layer")

        base = layer_list[0]
        if self.smooth_alpha_frames:
            base = smooth_alpha(base, self.alpha_kernel_size)

        for layer in layer_list[1:]:
            if self.smooth_alpha_frames:
                layer = smooth_alpha(layer, self.alpha_kernel_size)
            base = composite_over(top=layer, bottom=base)

        return base[:, :3]
