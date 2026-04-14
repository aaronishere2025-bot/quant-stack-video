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
from pathlib import Path
from typing import List, Optional

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


# ---------------------------------------------------------------------------
# Video loading helpers (stand-in until Wan-Alpha ships in diffusers)
# ---------------------------------------------------------------------------

def load_rgb_from_video(video_path: str, max_frames: Optional[int] = None) -> "torch.Tensor":
    """
    Load an MP4 file into a float32 RGB tensor.

    Uses imageio with FFMPEG (always available) to read the video.

    Args:
        video_path:  Path to an .mp4 file.
        max_frames:  If set, truncate to this many frames.

    Returns:
        [1, 3, F, H, W] float32 tensor, values in [0, 1].

    Raises:
        FileNotFoundError: if video_path does not exist.
        RuntimeError:      if imageio cannot decode the file.
    """
    import torch
    import numpy as np
    import imageio

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path, format="FFMPEG")
    try:
        raw_frames: list = []
        for i, frame in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            raw_frames.append(frame)
    finally:
        reader.close()

    if not raw_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    frames = np.stack(raw_frames)  # [F, H, W, 3] uint8

    # [F, H, W, 3] uint8 → [1, 3, F, H, W] float32 in [0, 1]
    t = torch.from_numpy(frames.astype(np.float32) / 255.0)
    t = t.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, F, H, W]
    return t


def rgb_to_rgba_luminance(
    rgb: "torch.Tensor",
    layer_role: str = "midground",
    alpha_scale: float = 1.0,
) -> "torch.Tensor":
    """
    Add a luminance-derived alpha channel to an RGB video tensor.

    This is a stand-in compositing technique used when true RGBA generation
    (Wan-Alpha) is not yet available.  Different roles use different alpha
    strategies so the composite has visible depth:

      background  — fully opaque (alpha = 1.0)
      midground   — screen-matte: alpha = max(R, G, B)  (light areas visible)
      foreground  — luma-matte:   alpha = Y = 0.299R + 0.587G + 0.114B

    Args:
        rgb:         [B, 3, F, H, W] float32 tensor, values in [0, 1].
        layer_role:  "background", "midground", or "foreground".
        alpha_scale: Scalar multiplier applied to the computed alpha (clipped
                     to [0, 1]).  Use < 1.0 to make a layer semi-transparent.

    Returns:
        [B, 4, F, H, W] float32 RGBA tensor.
    """
    import torch

    if rgb.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB tensor, got shape {rgb.shape}")

    role = layer_role.lower()
    if role == "background":
        alpha = torch.ones_like(rgb[:, :1])
    elif role == "midground":
        # Screen matte: bright pixels (clouds, light sources) are more opaque
        alpha = rgb.max(dim=1, keepdim=True).values
    else:  # foreground
        # Luma matte: weighted luminance — good for subjects against dark BG
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        alpha = 0.299 * r + 0.587 * g + 0.114 * b

    if alpha_scale != 1.0:
        alpha = (alpha * alpha_scale).clamp(0.0, 1.0)

    return torch.cat([rgb, alpha], dim=1)  # [B, 4, F, H, W]


def load_rgba_from_video(
    video_path: str,
    layer_role: str = "midground",
    max_frames: Optional[int] = None,
    alpha_scale: float = 1.0,
) -> "torch.Tensor":
    """
    Load an MP4 and return an RGBA tensor with a luminance-derived alpha.

    Convenience wrapper around :func:`load_rgb_from_video` +
    :func:`rgb_to_rgba_luminance`.

    Args:
        video_path:  Path to an .mp4 file.
        layer_role:  "background", "midground", or "foreground".
        max_frames:  Truncate to this many frames if set.
        alpha_scale: Scale factor for the computed alpha.

    Returns:
        [1, 4, F, H, W] float32 RGBA tensor.
    """
    rgb = load_rgb_from_video(video_path, max_frames=max_frames)
    return rgb_to_rgba_luminance(rgb, layer_role=layer_role, alpha_scale=alpha_scale)


def save_rgb_tensor_as_mp4(
    rgb: "torch.Tensor",
    output_path: str,
    fps: int = 16,
    batch_idx: int = 0,
) -> str:
    """
    Encode an RGB video tensor to an MP4 file.

    Args:
        rgb:         [B, 3, F, H, W] float32 tensor, values in [0, 1].
        output_path: Destination .mp4 path (parent directory must exist).
        fps:         Output frame rate.
        batch_idx:   Which batch element to encode (default 0).

    Returns:
        output_path (unchanged).

    Raises:
        ValueError: if rgb does not have 3 channels.
    """
    import numpy as np
    import imageio

    if rgb.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB tensor, got shape {rgb.shape}")

    # [B, 3, F, H, W] → [F, H, W, 3] uint8
    frames_tensor = rgb[batch_idx]                          # [3, F, H, W]
    frames_tensor = frames_tensor.permute(1, 2, 3, 0)      # [F, H, W, 3]
    frames_uint8 = (frames_tensor.clamp(0.0, 1.0) * 255.0).byte().numpy()  # [F, H, W, 3]

    writer = imageio.get_writer(output_path, fps=fps, format="FFMPEG")
    try:
        for frame in frames_uint8:
            writer.append_data(frame)
    finally:
        writer.close()

    logger.debug("Saved composited RGB tensor to %s (%d frames @ %d fps)", output_path, len(frames_uint8), fps)
    return output_path
