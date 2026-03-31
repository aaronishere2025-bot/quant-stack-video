#!/usr/bin/env python3
"""
CLI for Wan 2.1 video generation with quantization stacking.

Examples:
    # Single-pass 4-bit
    python scripts/generate.py --prompt "A serene lake at sunrise" --quant 4bit

    # 3-pass progressive stack
    python scripts/generate.py --prompt "..." --stacked --passes 3 --strategy progressive

    # Long video (30 seconds)
    python scripts/generate.py --prompt "..." --long --duration 30
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


def main():
    parser = argparse.ArgumentParser(description="Wan 2.1 Video Generation with Quant Stacking")
    parser.add_argument("--prompt", required=True, help="Video generation prompt")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output", default="output.mp4", help="Output file path")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--cache-dir", default=None)

    # Resolution
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--fps", type=int, default=16)

    # Inference
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42)

    # Mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--stacked", action="store_true", help="Use multi-pass stacking")
    mode_group.add_argument("--long", action="store_true", help="Generate long video")
    mode_group.add_argument("--benchmark", action="store_true", help="Run benchmark comparison")

    # Single-pass quant type
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")

    # Stacking options
    parser.add_argument("--passes", type=int, default=3, help="Number of stack passes")
    parser.add_argument("--strategy", choices=["progressive", "average", "weighted", "residual"],
                        default="progressive")

    # Long video options
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    parser.add_argument("--segment-frames", type=int, default=81)
    parser.add_argument("--overlap-frames", type=int, default=8)

    args = parser.parse_args()

    neg_prompt = args.negative_prompt or (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画面变形，畸形，毁容，"
        "形态畸形，手部畸形，腿部畸形，额外的手指，融合的手指，静止画面，动态不足，"
    )

    if args.long:
        from src.wan.generate import generate_long_video
        print(f"Generating long video ({args.duration}s)...")
        result = generate_long_video(
            prompt=args.prompt,
            output_path=args.output,
            duration_seconds=args.duration,
            model_id=args.model,
            negative_prompt=neg_prompt,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            segment_frames=args.segment_frames,
            overlap_frames=args.overlap_frames,
            use_stacking=True,
            num_passes=args.passes,
            stacking_strategy=args.strategy,
            cache_dir=args.cache_dir,
        )
        print(f"\nDone! Saved to: {result['output_path']}")
        print(f"Segments: {result['num_segments']}, Frames: {result['total_frames']}, "
              f"Time: {result['total_time']:.1f}s")

    elif args.stacked:
        from src.wan.generate import generate_video_stacked
        print(f"Generating with {args.passes}-pass {args.strategy} stacking...")
        result = generate_video_stacked(
            prompt=args.prompt,
            output_path=args.output,
            model_id=args.model,
            negative_prompt=neg_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            num_passes=args.passes,
            stacking_strategy=args.strategy,
            cache_dir=args.cache_dir,
            fps=args.fps,
        )
        print(f"\nDone! Saved to: {result['output_path']}")
        print(f"Total time: {result['total_time']:.1f}s | Pass times: {result['pass_times']}")

    elif args.benchmark:
        from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig
        print("Running benchmark comparison...")
        cfg = BenchmarkConfig(
            prompts=[args.prompt],
            model_id=args.model,
            cache_dir=args.cache_dir,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            output_dir=str(Path(args.output).parent / "benchmark"),
        )
        runner = BenchmarkRunner(cfg)
        results = runner.run()
        report_path = str(Path(args.output).parent / "benchmark_report.json")
        runner.save_report(results, report_path)
        print(f"Report saved to: {report_path}")

    else:
        from src.wan.generate import generate_video
        print(f"Generating with single {args.quant} pass...")
        saved = generate_video(
            prompt=args.prompt,
            output_path=args.output,
            model_id=args.model,
            negative_prompt=neg_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            quant_type=args.quant,
            cache_dir=args.cache_dir,
            fps=args.fps,
        )
        print(f"\nDone! Saved to: {saved}")


if __name__ == "__main__":
    main()
