"""
VACE Temporal Extension — Phase 3 of the layered infinite video pipeline.

Provides seamless segment-to-segment continuity for infinite video generation
by passing the last N latent frames from segment N as conditioning for segment N+1.

Critical constraints (do not change without benchmarking):
  - Shift = 1           (VACE-required scheduler parameter)
  - CFG = 2.0–3.0       (lower = consistency, higher = variation)
  - Overlap = 16 frames (frames 65–81 of an 81-frame segment)
  - Mask convention: known frames = black (0), unknown frames = white (1)
  - Padding color: #7F7F7F grey in pixel space → ~0.498 normalized

NEVER decode + re-encode latents for segment handoffs.
Color shift is caused by VAE reconstruction error accumulation.
Keep overlap in latent space at all times.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

# VACE grey padding value in [0, 1] pixel space  (#7F7F7F)
GREY_PIXEL_VALUE = 0x7F / 0xFF  # ≈ 0.498


@dataclass
class VACEConfig:
    """Configuration for VACE temporal extension."""
    shift: float = 1.0          # Scheduler shift — required by VACE, do not change
    cfg_scale: float = 2.5      # CFG guidance scale (2.0–3.0 range)
    overlap_frames: int = 16    # Frames from previous segment used as conditioning
    segment_frames: int = 81    # Total frames per segment (must be 4k+1)
    grey_pad_value: float = GREY_PIXEL_VALUE

    def __post_init__(self):
        if (self.segment_frames - 1) % 4 != 0:
            raise ValueError(
                f"segment_frames must be 4k+1 (e.g. 81, 49, 33). Got {self.segment_frames}"
            )
        if not 2.0 <= self.cfg_scale <= 3.0:
            logger.warning(
                "cfg_scale=%.1f is outside the validated 2.0–3.0 range for VACE. "
                "Quality may degrade.", self.cfg_scale
            )


@dataclass
class SegmentHandoff:
    """
    Latent-space state passed between consecutive video segments.

    Contains the last `overlap_frames` latents from the completed segment,
    used to condition the next segment's generation without decoding.
    """
    latents: "torch.Tensor"     # [B, C, overlap_frames, H_lat, W_lat] float32
    segment_idx: int            # Which segment these latents came from
    prompt: str                 # The prompt that generated this segment (for logging)

    @property
    def num_overlap_frames(self) -> int:
        return self.latents.shape[2]


def build_vace_mask(
    segment_frames: int,
    overlap_frames: int,
    device: "torch.device",
    dtype: "torch.dtype" = None,
) -> "torch.Tensor":
    """
    Build the VACE conditioning mask for a single segment.

    Mask convention:
      - known frames (overlap from previous segment): 0 (black)
      - unknown frames (to be generated):             1 (white)

    Args:
        segment_frames: Total frames in the segment (e.g. 81)
        overlap_frames: Frames provided from previous segment
        device:         Torch device for the output tensor
        dtype:          Tensor dtype (defaults to float32)

    Returns:
        [1, 1, segment_frames, 1, 1] mask tensor (broadcastable over spatial dims)
    """
    import torch

    if dtype is None:
        dtype = torch.float32

    mask = torch.ones(segment_frames, device=device, dtype=dtype)
    mask[:overlap_frames] = 0.0  # known frames → black

    # Expand to [1, 1, F, 1, 1] for broadcasting with [B, C, F, H, W] latents
    return mask.reshape(1, 1, segment_frames, 1, 1)


def pad_latents_with_grey(
    known_latents: "torch.Tensor",
    total_frames: int,
    vae_scale_factor: int = 8,
    grey_pixel: float = GREY_PIXEL_VALUE,
) -> "torch.Tensor":
    """
    Pad a partial latent tensor to full segment length using grey (#7F7F7F) pixels.

    The known latents occupy the first N frames; remaining frames are filled with
    the latent-space equivalent of grey padding.

    Note: Grey padding is applied in latent space by encoding a grey frame.
    Since we cannot call the VAE here without a model reference, callers should
    either pass pre-encoded grey latents or use `encode_grey_latent()` first.

    Args:
        known_latents:  [B, C, N, H_lat, W_lat] — the overlap latents
        total_frames:   Target number of frames after padding
        vae_scale_factor: Spatial downsampling factor of the VAE (default 8)
        grey_pixel:     Normalized grey pixel value to encode as padding

    Returns:
        [B, C, total_frames, H_lat, W_lat] with grey padding appended
    """
    import torch

    B, C, N, H_lat, W_lat = known_latents.shape
    unknown_frames = total_frames - N

    if unknown_frames < 0:
        raise ValueError(
            f"known_latents has {N} frames, which exceeds total_frames={total_frames}"
        )
    if unknown_frames == 0:
        return known_latents

    # Approximate latent-space grey: zero-mean assumption for a grey input
    # Exact encoding requires the VAE; callers with access to VAE should use
    # encode_grey_latent() instead. For mask-based generation, the padding
    # values don't affect output quality (they are masked out).
    grey_latents = torch.full(
        (B, C, unknown_frames, H_lat, W_lat),
        fill_value=0.0,  # VAE-encoded grey ≈ 0 for most video VAEs
        device=known_latents.device,
        dtype=known_latents.dtype,
    )

    return torch.cat([known_latents, grey_latents], dim=2)


class VACEExtension:
    """
    Manages latent-space state for VACE-conditioned long-form video generation.

    Usage:
        vace = VACEExtension(config)

        # For each segment:
        handoff = vace.prepare_next_segment(pipeline, prompt, prev_handoff)
        # handoff.latents contains overlap latents for the NEXT segment
    """

    def __init__(self, config: Optional[VACEConfig] = None):
        self.config = config or VACEConfig()
        self._handoff_history: List[SegmentHandoff] = []

    @property
    def has_prior_segment(self) -> bool:
        return len(self._handoff_history) > 0

    @property
    def last_handoff(self) -> Optional[SegmentHandoff]:
        return self._handoff_history[-1] if self._handoff_history else None

    def extract_overlap_latents(
        self,
        full_latents: "torch.Tensor",
        segment_idx: int,
        prompt: str,
    ) -> SegmentHandoff:
        """
        Extract the last `overlap_frames` latents from a completed segment.

        These latents are passed to the next segment's generation without
        decoding (to avoid color shift from VAE reconstruction error).

        Args:
            full_latents: [B, C, F, H_lat, W_lat] latents for the completed segment
            segment_idx:  Index of the completed segment
            prompt:       Prompt used for this segment

        Returns:
            SegmentHandoff containing the overlap latents
        """
        N = self.config.overlap_frames
        overlap = full_latents[:, :, -N:].clone()  # last N frames

        handoff = SegmentHandoff(
            latents=overlap,
            segment_idx=segment_idx,
            prompt=prompt,
        )
        self._handoff_history.append(handoff)
        logger.info(
            "Extracted %d overlap latents from segment %d (shape %s)",
            N, segment_idx, list(overlap.shape)
        )
        return handoff

    def build_conditioning(
        self,
        handoff: SegmentHandoff,
        target_frames: int,
        device: "torch.device",
    ) -> dict:
        """
        Build the VACE conditioning dict for the next segment's pipeline call.

        Returns a dict with:
          - "latents":    padded latent tensor  [B, C, target_frames, H_lat, W_lat]
          - "mask":       conditioning mask     [1, 1, target_frames, 1, 1]
          - "shift":      scheduler shift value
          - "cfg_scale":  guidance scale

        Args:
            handoff:       SegmentHandoff from the previous segment
            target_frames: Total frames for the new segment
            device:        Target device

        Returns:
            dict with conditioning tensors and generation params
        """
        overlap_latents = handoff.latents.to(device)
        padded = pad_latents_with_grey(
            overlap_latents,
            total_frames=target_frames,
        )
        mask = build_vace_mask(
            segment_frames=target_frames,
            overlap_frames=handoff.num_overlap_frames,
            device=device,
        )

        return {
            "latents": padded,
            "mask": mask,
            "shift": self.config.shift,
            "cfg_scale": self.config.cfg_scale,
        }

    def reset(self):
        """Reset handoff history (call on scene change to clear SVI error buffer too)."""
        self._handoff_history.clear()
        logger.info("VACE handoff history reset (scene change)")
