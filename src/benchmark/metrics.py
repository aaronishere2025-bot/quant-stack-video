"""
Video quality metrics for comparing quantization approaches.

Metrics implemented:
- PSNR: Peak Signal-to-Noise Ratio (pixel-level fidelity)
- SSIM: Structural Similarity Index (perceptual structure)
- LPIPS: Learned Perceptual Image Patch Similarity (deep feature similarity)
- Temporal consistency: Frame-to-frame difference (motion smoothness)
- VRAM usage: Peak GPU memory during generation
"""

import logging
from typing import Optional, Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class VideoQualityMetrics:
    """Compute quality metrics between reference and generated video frames."""

    def __init__(self, use_lpips: bool = True, device: str = "cuda"):
        self.device = device
        self.use_lpips = use_lpips
        self._lpips_model = None

        if use_lpips:
            try:
                self._load_lpips()
            except ImportError:
                logger.warning("LPIPS not available, skipping perceptual metric")
                self.use_lpips = False

    def _load_lpips(self):
        import lpips
        import torch
        self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self._lpips_model.eval()

    def compute_all(
        self,
        reference: np.ndarray,
        generated: np.ndarray,
        label: str = "generated",
    ) -> Dict[str, float]:
        """
        Compute all quality metrics between reference and generated frames.

        Args:
            reference: (T, H, W, C) float32 [0,1] — ground truth or high-precision output
            generated: (T, H, W, C) float32 [0,1] — quantized output to evaluate

        Returns:
            Dict of metric name → value
        """
        assert reference.shape == generated.shape, (
            f"Shape mismatch: reference {reference.shape} vs generated {generated.shape}"
        )

        metrics = {}
        metrics["label"] = label
        metrics["psnr"] = self.psnr(reference, generated)
        metrics["ssim"] = self.ssim_video(reference, generated)
        metrics["temporal_consistency_ref"] = self.temporal_consistency(reference)
        metrics["temporal_consistency_gen"] = self.temporal_consistency(generated)
        metrics["temporal_consistency_delta"] = (
            metrics["temporal_consistency_gen"] - metrics["temporal_consistency_ref"]
        )

        if self.use_lpips and self._lpips_model is not None:
            metrics["lpips"] = self.lpips_video(reference, generated)
        else:
            metrics["lpips"] = None

        return metrics

    def psnr(self, reference: np.ndarray, generated: np.ndarray) -> float:
        """
        Peak Signal-to-Noise Ratio across all frames.
        Higher is better. >30 dB is generally good quality.
        """
        mse = np.mean((reference.astype(np.float64) - generated.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return float(20 * np.log10(1.0 / np.sqrt(mse)))

    def ssim_video(self, reference: np.ndarray, generated: np.ndarray) -> float:
        """
        Mean SSIM across all frames. Range [−1, 1]; higher is better.
        """
        try:
            from skimage.metrics import structural_similarity as ssim_fn
        except ImportError:
            logger.warning("scikit-image not available, computing simplified SSIM")
            return self._simple_ssim(reference, generated)

        scores = []
        for ref_frame, gen_frame in zip(reference, generated):
            score = ssim_fn(ref_frame, gen_frame, channel_axis=-1, data_range=1.0)
            scores.append(score)
        return float(np.mean(scores))

    def _simple_ssim(self, reference: np.ndarray, generated: np.ndarray) -> float:
        """Simplified SSIM approximation without scikit-image."""
        mu1 = np.mean(reference)
        mu2 = np.mean(generated)
        sigma1_sq = np.var(reference)
        sigma2_sq = np.var(generated)
        sigma12 = np.mean((reference - mu1) * (generated - mu2))

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        return float(numerator / denominator)

    def lpips_video(self, reference: np.ndarray, generated: np.ndarray) -> float:
        """
        Mean LPIPS perceptual distance across frames. Lower is better.
        """
        import torch

        scores = []
        for ref_frame, gen_frame in zip(reference, generated):
            # (H, W, C) → (1, C, H, W), normalized to [-1, 1]
            ref_t = torch.from_numpy(ref_frame).permute(2, 0, 1).unsqueeze(0).float()
            gen_t = torch.from_numpy(gen_frame).permute(2, 0, 1).unsqueeze(0).float()
            ref_t = ref_t * 2 - 1
            gen_t = gen_t * 2 - 1

            ref_t = ref_t.to(self.device)
            gen_t = gen_t.to(self.device)

            with torch.no_grad():
                dist = self._lpips_model(ref_t, gen_t)
            scores.append(dist.item())

        return float(np.mean(scores))

    def temporal_consistency(self, frames: np.ndarray) -> float:
        """
        Measure temporal smoothness as mean absolute difference between consecutive frames.
        Lower is more consistent (smoother motion).
        """
        if len(frames) < 2:
            return 0.0
        diffs = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
        return float(np.mean(diffs))

    def vram_usage(self) -> Optional[float]:
        """Return current GPU VRAM usage in GB, or None if not available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        return None

    def boundary_ssim(self, prev_last_frame: np.ndarray, next_first_frame: np.ndarray) -> float:
        """
        Compute SSIM between the last frame of one segment and the first frame
        of the next to measure cross-segment temporal continuity.

        A score near 1.0 means the boundary is seamless; near 0.0 means a
        hard visual cut.  This is the primary drift metric for the infinite
        generation pipeline.

        Args:
            prev_last_frame:  (H, W, C) or (H, W) float32 in [0, 1].
            next_first_frame: (H, W, C) or (H, W) float32 in [0, 1], same shape.

        Returns:
            float SSIM in [-1, 1] (typically [0, 1] for natural images).
        """
        if prev_last_frame.shape != next_first_frame.shape:
            raise ValueError(
                f"Frame shape mismatch: {prev_last_frame.shape} vs {next_first_frame.shape}"
            )
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            return float(ssim_fn(
                prev_last_frame, next_first_frame,
                channel_axis=-1 if prev_last_frame.ndim == 3 else None,
                data_range=1.0,
            ))
        except ImportError:
            return float(self._simple_ssim(
                prev_last_frame[np.newaxis], next_first_frame[np.newaxis]
            ))

    def format_report(self, metrics_list: List[Dict]) -> str:
        """Format a comparison table of metrics for multiple configurations."""
        if not metrics_list:
            return "No metrics to report."

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("QUANTIZATION QUALITY BENCHMARK REPORT")
        lines.append("=" * 70)

        header = f"{'Config':<25} {'PSNR (dB)':>10} {'SSIM':>8} {'LPIPS':>8} {'Temporal Δ':>12}"
        lines.append(header)
        lines.append("-" * 70)

        for m in metrics_list:
            label = m.get("label", "unknown")[:24]
            psnr = m.get("psnr", 0.0)
            ssim = m.get("ssim", 0.0)
            lpips = m.get("lpips")
            t_delta = m.get("temporal_consistency_delta", 0.0)

            lpips_str = f"{lpips:.4f}" if lpips is not None else "N/A"
            line = f"{label:<25} {psnr:>10.2f} {ssim:>8.4f} {lpips_str:>8} {t_delta:>+12.5f}"
            lines.append(line)

        lines.append("=" * 70)
        lines.append("PSNR: higher=better | SSIM: higher=better (max 1.0)")
        lines.append("LPIPS: lower=better | Temporal Δ: 0=same motion as reference")
        lines.append("=" * 70)

        return "\n".join(lines)


def compute_boundary_ssim(prev_last_frame_path: str, next_first_frame_path: str) -> float:
    """
    Convenience function: load two PNG frame images and return boundary SSIM.

    Returns 0.0 if either file is missing (graceful degradation for first segment
    and error cases so the infinite loop is never blocked by a metric failure).

    Args:
        prev_last_frame_path:  Path to the last frame PNG of segment N.
        next_first_frame_path: Path to the first frame PNG of segment N+1.

    Returns:
        float SSIM in [0, 1], or 0.0 on failure.
    """
    try:
        import os
        if not os.path.exists(prev_last_frame_path) or not os.path.exists(next_first_frame_path):
            return 0.0
        from PIL import Image
        prev = np.array(Image.open(prev_last_frame_path).convert("RGB")).astype(np.float32) / 255.0
        nxt = np.array(Image.open(next_first_frame_path).convert("RGB")).astype(np.float32) / 255.0
        return VideoQualityMetrics(use_lpips=False, device="cpu").boundary_ssim(prev, nxt)
    except Exception:
        return 0.0
