#!/usr/bin/env python3
"""
Benchmark: single 4-bit pass vs 3-pass progressive stack.

Generates both with the same prompt/seed at full quality (480x832, 81 frames, 30 steps),
then computes PSNR, SSIM, LPIPS, and temporal consistency between them.
Also reports VRAM usage and timing for each.
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

PROMPT = "A golden retriever running through a sunlit meadow, cinematic lighting, slow motion"
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画面变形，畸形，毁容，"
    "形态畸形，手部畸形，腿部畸形，额外的手指，融合的手指，静止画面，动态不足，"
)
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 81
STEPS = 30
GUIDANCE = 5.0
SEED = 42
FPS = 16
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

OUT_DIR = Path("benchmark_outputs/1pass_vs_3pass")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_video(frames_np, path, fps=16):
    import imageio
    frames_uint8 = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    with imageio.get_writer(str(path), fps=fps, codec="libx264",
                            output_params=["-crf", "18"]) as w:
        for f in frames_uint8:
            w.append_data(f)
    logger.info(f"Saved: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


def run_single_pass():
    """Single 4-bit NF4 pass."""
    from src.wan.pipeline_factory import WanPipelineFactory
    from src.quant.config import QuantConfig

    logger.info("=" * 60)
    logger.info("SINGLE 4-BIT PASS")
    logger.info("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    factory = WanPipelineFactory(model_id=MODEL_ID)
    qcfg = QuantConfig(load_in_4bit=True, load_in_8bit=False)
    pipe = factory(qcfg)

    vram_after_load = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after load: {vram_after_load:.2f} GB")

    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED)

    t0 = time.time()
    output = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
        output_type="np",
    )
    elapsed = time.time() - t0

    frames = output.frames[0].astype(np.float32)
    vram_peak = torch.cuda.max_memory_allocated() / 1e9

    logger.info(f"Single pass done: {elapsed:.1f}s | VRAM peak: {vram_peak:.2f} GB")

    # Save
    save_video(frames, OUT_DIR / "single_4bit.mp4", fps=FPS)
    np.save(OUT_DIR / "single_4bit_frames.npy", frames)

    # Cleanup
    del pipe, output
    gc.collect()
    torch.cuda.empty_cache()

    return frames, elapsed, vram_peak


def run_3pass_progressive():
    """3-pass progressive stack."""
    from src.wan.pipeline_factory import WanPipelineFactory
    from src.quant.config import StackConfig
    from src.quant.engine import QuantStackEngine

    logger.info("=" * 60)
    logger.info("3-PASS PROGRESSIVE STACK")
    logger.info("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    factory = WanPipelineFactory(model_id=MODEL_ID)
    stack_cfg = StackConfig(num_passes=3, stacking_strategy="progressive")
    engine = QuantStackEngine(stack_cfg)

    t0 = time.time()
    result = engine.run_stacked(
        pipeline_factory=factory,
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        seed=SEED,
    )
    elapsed = time.time() - t0

    frames = result["frames"]
    vram_peak = torch.cuda.max_memory_allocated() / 1e9

    logger.info(f"3-pass done: {elapsed:.1f}s | VRAM peak: {vram_peak:.2f} GB")
    logger.info(f"Per-pass times: {result['pass_times']}")

    save_video(frames, OUT_DIR / "3pass_progressive.mp4", fps=FPS)
    np.save(OUT_DIR / "3pass_progressive_frames.npy", frames)

    del result
    gc.collect()
    torch.cuda.empty_cache()

    return frames, elapsed, vram_peak


def compute_metrics(single_frames, stacked_frames):
    """Compute quality metrics between single-pass and 3-pass."""
    from src.benchmark.metrics import VideoQualityMetrics

    logger.info("=" * 60)
    logger.info("COMPUTING METRICS")
    logger.info("=" * 60)

    metrics = VideoQualityMetrics(use_lpips=True)

    # Use single-pass as reference, measure how much stacking improves
    comparison = metrics.compute_all(single_frames, stacked_frames, label="3pass-vs-1pass")

    # Also compute self-metrics for temporal consistency
    single_temporal = metrics.temporal_consistency(single_frames)
    stacked_temporal = metrics.temporal_consistency(stacked_frames)

    return {
        **comparison,
        "single_temporal_consistency": single_temporal,
        "stacked_temporal_consistency": stacked_temporal,
    }


def main():
    logger.info("Benchmark: Single 4-bit vs 3-pass Progressive Stack")
    logger.info(f"Prompt: {PROMPT}")
    logger.info(f"Resolution: {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames, {STEPS} steps")
    logger.info("")

    # Run single pass
    single_frames, single_time, single_vram = run_single_pass()

    # Run 3-pass progressive
    stacked_frames, stacked_time, stacked_vram = run_3pass_progressive()

    # Compute metrics
    metrics = compute_metrics(single_frames, stacked_frames)

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<35} {'Single 4-bit':>15} {'3-Pass Stack':>15}")
    logger.info("-" * 70)
    logger.info(f"{'Generation time (s)':<35} {single_time:>15.1f} {stacked_time:>15.1f}")
    logger.info(f"{'VRAM peak (GB)':<35} {single_vram:>15.2f} {stacked_vram:>15.2f}")
    logger.info(f"{'Temporal consistency':<35} {metrics['single_temporal_consistency']:>15.5f} {metrics['stacked_temporal_consistency']:>15.5f}")
    logger.info("")
    logger.info("3-pass vs single-pass comparison:")
    logger.info(f"  PSNR:  {metrics['psnr']:.2f} dB  (higher = more similar)")
    logger.info(f"  SSIM:  {metrics['ssim']:.4f}  (1.0 = identical)")
    if metrics.get('lpips') is not None:
        logger.info(f"  LPIPS: {metrics['lpips']:.4f}  (0.0 = identical)")
    logger.info(f"  Temporal Δ: {metrics['temporal_consistency_delta']:+.5f}")
    logger.info("=" * 70)

    # Save report
    report = {
        "prompt": PROMPT,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "num_frames": NUM_FRAMES,
        "steps": STEPS,
        "single_pass": {"time_s": single_time, "vram_peak_gb": single_vram},
        "stacked_3pass": {"time_s": stacked_time, "vram_peak_gb": stacked_vram},
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()},
    }
    report_path = OUT_DIR / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
