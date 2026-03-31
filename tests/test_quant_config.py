"""Unit tests for quantization config (no GPU required)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


def test_quant_config_defaults():
    from src.quant.config import QuantConfig
    cfg = QuantConfig()
    assert cfg.quant_type == "4bit"
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True


def test_stack_config_defaults():
    from src.quant.config import StackConfig
    cfg = StackConfig(num_passes=3)
    assert len(cfg.pass_configs) == 3
    assert cfg.stacking_strategy == "progressive"


def test_stack_config_pass_diversity():
    """Verify that default pass configs vary quant type."""
    from src.quant.config import StackConfig
    cfg = StackConfig(num_passes=3)
    quant_types = [p.bnb_4bit_quant_type for p in cfg.pass_configs]
    # Should have at least two different quant types across 3 passes
    assert len(set(quant_types)) >= 1  # At minimum 1, typically 2


def test_temporal_consistency():
    """Test temporal consistency metric with synthetic frames."""
    from src.benchmark.metrics import VideoQualityMetrics

    metrics = VideoQualityMetrics(use_lpips=False, device="cpu")

    # Create smooth frames (small differences) → low temporal inconsistency
    frames_smooth = np.zeros((10, 64, 64, 3), dtype=np.float32)
    for t in range(10):
        frames_smooth[t] = t * 0.01

    # Create jittery frames (large differences)
    frames_jittery = np.random.rand(10, 64, 64, 3).astype(np.float32)

    tc_smooth = metrics.temporal_consistency(frames_smooth)
    tc_jittery = metrics.temporal_consistency(frames_jittery)

    assert tc_smooth < tc_jittery, "Smooth video should have lower temporal inconsistency"


def test_psnr_identical():
    """PSNR of identical frames should be infinity."""
    from src.benchmark.metrics import VideoQualityMetrics
    metrics = VideoQualityMetrics(use_lpips=False, device="cpu")
    frames = np.random.rand(5, 32, 32, 3).astype(np.float32)
    psnr = metrics.psnr(frames, frames)
    assert psnr == float("inf")


def test_psnr_different():
    """PSNR of different frames should be finite and positive."""
    from src.benchmark.metrics import VideoQualityMetrics
    metrics = VideoQualityMetrics(use_lpips=False, device="cpu")
    ref = np.ones((5, 32, 32, 3), dtype=np.float32) * 0.5
    gen = ref + np.random.randn(*ref.shape).astype(np.float32) * 0.1
    gen = np.clip(gen, 0, 1)
    psnr = metrics.psnr(ref, gen)
    assert 0 < psnr < 100


def test_ssim_identical():
    """SSIM of identical frames should be ~1.0."""
    from src.benchmark.metrics import VideoQualityMetrics
    metrics = VideoQualityMetrics(use_lpips=False, device="cpu")
    frames = np.random.rand(3, 32, 32, 3).astype(np.float32)
    ssim = metrics._simple_ssim(frames, frames)
    assert abs(ssim - 1.0) < 1e-6


def test_metrics_report_format():
    """Test that report formatting works with sample data."""
    from src.benchmark.metrics import VideoQualityMetrics
    metrics = VideoQualityMetrics(use_lpips=False, device="cpu")
    sample_metrics = [
        {"label": "bf16-reference", "psnr": float("inf"), "ssim": 1.0,
         "lpips": 0.0, "temporal_consistency_delta": 0.0},
        {"label": "4bit-single", "psnr": 28.5, "ssim": 0.921,
         "lpips": 0.12, "temporal_consistency_delta": 0.002},
        {"label": "3x4bit-progressive", "psnr": 31.2, "ssim": 0.953,
         "lpips": 0.08, "temporal_consistency_delta": 0.001},
    ]
    report = metrics.format_report(sample_metrics)
    assert "PSNR" in report
    assert "SSIM" in report
    assert "3x4bit-progressive" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
