"""
SVI Error Recycling — Phase 4 of the layered infinite video pipeline.

Injects historical DiT prediction errors back into the flow matching denoiser
to counteract autoregressive drift over infinite segment generation.

Problem:
  Each segment conditions on the previous segment's output.
  Quantization error accumulates — after ~10 segments quality degrades visibly.

Solution (SVI-Shot variant):
  1. During segment N generation, capture DiT residuals (prediction errors)
  2. Maintain a running exponential moving average of these errors
  3. When generating segment N+1, inject the averaged error as a correction
     term into the flow matching noise prediction step
  4. This "recycles" the error signal, counteracting accumulation

Scene change handling:
  - Reset the error buffer when LLM director signals a scene change
  - Prevents cross-scene error contamination
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque

logger = logging.getLogger(__name__)


@dataclass
class SVIConfig:
    """Configuration for SVI error recycling."""
    buffer_size: int = 5             # How many segments of errors to keep
    ema_decay: float = 0.9           # EMA smoothing factor for error buffer
    injection_scale: float = 0.1     # Scale factor when injecting recycled errors
    enabled: bool = True             # Easy toggle for ablation studies
    reset_on_scene_change: bool = True  # Clear buffer on scene change signal

    def __post_init__(self):
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")
        if self.injection_scale < 0:
            raise ValueError(f"injection_scale must be >= 0, got {self.injection_scale}")


class SVIErrorBuffer:
    """
    Exponential moving average buffer of DiT prediction errors.

    Stores and combines per-segment error tensors using EMA so that
    recent errors have more influence than older ones.
    """

    def __init__(self, ema_decay: float, buffer_size: int):
        self.ema_decay = ema_decay
        self.buffer_size = buffer_size
        self._ema: Optional["torch.Tensor"] = None
        self._history: Deque["torch.Tensor"] = deque(maxlen=buffer_size)
        self._segment_count = 0

    @property
    def is_empty(self) -> bool:
        return self._ema is None

    @property
    def segment_count(self) -> int:
        return self._segment_count

    def update(self, error: "torch.Tensor") -> None:
        """
        Update the EMA with a new error tensor.

        Args:
            error: DiT prediction residual tensor from the completed segment.
                   Shape can be anything — will be stored and injected as-is.
        """
        self._history.append(error.detach().clone())

        if self._ema is None:
            self._ema = error.detach().clone()
        else:
            self._ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * error.detach()

        self._segment_count += 1
        logger.debug(
            "SVI buffer updated (segment=%d, ema_norm=%.4f)",
            self._segment_count,
            float(self._ema.norm()),
        )

    def get_correction(self) -> Optional["torch.Tensor"]:
        """
        Return the current EMA error for injection into the next segment.

        Returns None if buffer is empty (first segment, no prior error).
        """
        return self._ema.clone() if self._ema is not None else None

    def reset(self) -> None:
        """Clear all buffered errors (call on scene change)."""
        self._ema = None
        self._history.clear()
        logger.info("SVI error buffer reset")


class SVIRecycler:
    """
    SVI-Shot error recycler for single-scene continuous video generation.

    Integrates with the Wan 2.1 DiT inference loop to:
      1. Capture prediction residuals after each segment
      2. Inject the EMA-smoothed correction into the next segment's denoising

    Usage:
        recycler = SVIRecycler(config)

        # After generating segment N:
        recycler.record_segment_errors(dit_output, dit_target)

        # Before generating segment N+1:
        correction = recycler.get_injection_correction()
        # Pass correction to pipeline's custom_correction kwarg (if supported)

    Note: Direct DiT injection requires a custom pipeline wrapper. If using
    standard Diffusers pipelines, the correction can be applied as a latent
    offset on the initial noisy latents before each segment's denoising loop.
    """

    def __init__(self, config: Optional[SVIConfig] = None):
        self.config = config or SVIConfig()
        self._buffer = SVIErrorBuffer(
            ema_decay=self.config.ema_decay,
            buffer_size=self.config.buffer_size,
        )

    @property
    def segment_count(self) -> int:
        return self._buffer.segment_count

    @property
    def has_correction(self) -> bool:
        return not self._buffer.is_empty and self.config.enabled

    def record_segment_errors(
        self,
        predicted: "torch.Tensor",
        target: "torch.Tensor",
    ) -> None:
        """
        Capture DiT prediction error for a completed segment.

        Args:
            predicted: Model's velocity/noise prediction, any shape
            target:    Ground truth velocity (from flow matching target)
                       For quantized inference, this is typically the
                       full-precision or reference model prediction.
        """
        if not self.config.enabled:
            return

        error = (target - predicted).detach()
        self._buffer.update(error)

    def record_latent_delta(
        self,
        prev_latent: "torch.Tensor",
        curr_latent: "torch.Tensor",
    ) -> None:
        """
        Alternative: record error as the delta between consecutive segment latents.

        Useful when direct DiT access is unavailable. The latent-space delta
        approximates the accumulated error from quantization + autoregressive drift.

        Args:
            prev_latent: Final latent of segment N   [B, C, overlap, H_lat, W_lat]
            curr_latent: Initial latent of segment N+1 (matching overlap region)
        """
        if not self.config.enabled:
            return

        # Only use the overlap region for error estimation
        delta = (curr_latent - prev_latent).detach()
        self._buffer.update(delta)

    def get_injection_correction(
        self,
        target_shape: Optional[tuple] = None,
    ) -> Optional["torch.Tensor"]:
        """
        Get the correction tensor to inject into the next segment's generation.

        Args:
            target_shape: If provided, verify correction shape matches before returning.

        Returns:
            Scaled correction tensor, or None if no correction available.
        """
        if not self.has_correction:
            return None

        correction = self._buffer.get_correction()
        if correction is None:
            return None

        scaled = correction * self.config.injection_scale

        if target_shape is not None and correction.shape != target_shape:
            logger.warning(
                "SVI correction shape %s does not match target shape %s — skipping injection",
                list(correction.shape), list(target_shape)
            )
            return None

        logger.debug(
            "SVI injection correction: scale=%.3f, norm=%.4f",
            self.config.injection_scale, float(scaled.norm())
        )
        return scaled

    def apply_correction_to_latents(
        self,
        latents: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply the recycled error correction directly to initial noisy latents.

        This is the latent-offset approach for use with standard Diffusers pipelines
        that don't expose direct DiT hook points.

        Args:
            latents: Initial noisy latents for the next segment [B, C, F, H_lat, W_lat]

        Returns:
            Corrected latents (original if no correction available or shapes mismatch)
        """
        correction = self.get_injection_correction(target_shape=tuple(latents.shape))
        if correction is None:
            return latents

        corrected = latents + correction.to(latents.device, latents.dtype)
        logger.info(
            "Applied SVI correction to initial latents (segment %d → %d)",
            self.segment_count, self.segment_count + 1
        )
        return corrected

    def on_scene_change(self) -> None:
        """
        Signal a scene change — resets the error buffer.

        Call this when the LLM continuity director reports a scene transition.
        Prevents error contamination across scenes.
        """
        if self.config.reset_on_scene_change:
            self._buffer.reset()
            logger.info("SVIRecycler: scene change received, error buffer cleared")
