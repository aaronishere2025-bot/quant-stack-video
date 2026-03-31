#!/usr/bin/env python3
"""
Long-video benchmark: layered quant stacking vs sequential single-pass.

Compares output quality and temporal continuity across segment boundaries
for videos up to 3 minutes, surfacing quality drift over time.

Usage:
    # Quick 30s test (default)
    python scripts/benchmark_long_video.py

    # 3-minute full benchmark
    python scripts/benchmark_long_video.py --duration 180

    # Custom prompt
    python scripts/benchmark_long_video.py --prompt "Ocean waves at sunrise" --duration 60

    # Skip stacking (faster, single-pass only)
    python scripts/benchmark_long_video.py --no-stack
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Long-video benchmark: layered quant vs sequential generation"
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Single prompt to use (overrides default prompt list)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Video duration in seconds (default: 30s; use 180 for 3 minutes)"
    )
    parser.add_argument(
        "--segment-frames", type=int, default=81,
        help="Frames per segment, must be 4k+1 (default: 81 = 5s @ 16fps)"
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HuggingFace model ID"
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output-dir", default="benchmark_outputs/long_video",
        help="Directory for output videos and report"
    )
    parser.add_argument(
        "--no-stack", action="store_true",
        help="Disable stacked configs (run sequential-4bit only)"
    )
    parser.add_argument(
        "--report", default="benchmark_outputs/long_video/report.json",
        help="Path to save JSON report"
    )
    args = parser.parse_args()

    # Validate num_frames constraint: must be 4k+1
    if (args.segment_frames - 1) % 4 != 0:
        logger.error(f"--segment-frames must be 4k+1 (e.g. 81, 49, 33). Got {args.segment_frames}")
        sys.exit(1)

    from src.benchmark.long_video_runner import LongVideoBenchmarkConfig, LongVideoBenchmarkRunner

    configs = [
        {"label": "bf16-baseline", "use_stacking": False, "quant_type": "none"},
        {"label": "sequential-4bit", "use_stacking": False, "quant_type": "4bit"},
    ]
    if not args.no_stack:
        configs += [
            {
                "label": "bf16-4pass-progressive",
                "use_stacking": True,
                "num_passes": 4,
                "strategy": "progressive",
                "quant_type": "none",
            },
            {
                "label": "layered-3x4bit-progressive",
                "use_stacking": True,
                "num_passes": 3,
                "strategy": "progressive",
                "quant_type": "4bit",
            },
            {
                "label": "layered-2x4bit-average",
                "use_stacking": True,
                "num_passes": 2,
                "strategy": "average",
                "quant_type": "4bit",
            },
        ]

    prompts = (
        [args.prompt] if args.prompt else [
            "A hiker walks through an ancient redwood forest, sunbeams filtering through the canopy, mist on the ground",
            "A coastal city at sunset, time-lapse of traffic and lights activating, waves rolling in",
        ]
    )

    cfg = LongVideoBenchmarkConfig(
        prompts=prompts,
        duration_seconds=args.duration,
        segment_frames=args.segment_frames,
        fps=args.fps,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        model_id=args.model,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        configs=configs,
    )

    logger.info(f"Long-video benchmark: {args.duration}s, {len(prompts)} prompt(s), {len(configs)} config(s)")

    import torch
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {total_vram:.1f} GB")
        assert total_vram >= 10.0, f"Insufficient VRAM: {total_vram:.1f} GB (need ≥10 GB)"
    else:
        logger.warning("CUDA not available — benchmark will be extremely slow")

    runner = LongVideoBenchmarkRunner(cfg)
    results = runner.run()
    runner.save_report(results, args.report)
    logger.info(f"\nDone. Report: {args.report}")


if __name__ == "__main__":
    main()
