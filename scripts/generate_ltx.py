#!/usr/bin/env python3
"""
LTX-Video generation script for Unity pipeline.

Uses Lightricks/LTX-Video via diffusers.
Designed to be called from ltx-video-generator.ts via child_process.exec.

Usage:
    python scripts/generate_ltx.py \
        --prompt "Ancient warrior on horseback" \
        --output /path/to/output.mp4 \
        --duration 8 \
        --width 768 --height 512

For 3-minute videos: call this repeatedly per clip, then concatenate with ffmpeg.
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LTX_MODEL_ID = "Lightricks/LTX-Video"
LTX_CACHE_DIR = "/mnt/d/ai-workspace/.hf-cache"

# LTX frame constraints: frames must satisfy (frames - 1) % 8 == 0
# Common values: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 97, 121, 161, 201
# At 25fps: 25f=1s, 49f=2s, 97f=4s, 121f=~5s, 201f=8s
FPS = 25


def _valid_frame_count(seconds: float) -> int:
    """Round duration to nearest valid LTX frame count: (n-1) % 8 == 0."""
    target = int(seconds * FPS)
    # Find nearest valid value
    n = max(9, round((target - 1) / 8) * 8 + 1)
    return n


def generate_ltx_clip(
    prompt: str,
    output_path: str,
    duration_seconds: float = 5.0,
    width: int = 768,
    height: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    seed: int = 42,
) -> str:
    """
    Generate a single LTX-Video clip.

    Returns the output path on success, raises on failure.
    """
    from diffusers import LTXPipeline
    import imageio

    num_frames = _valid_frame_count(duration_seconds)
    logger.info(f"Generating {num_frames} frames ({num_frames/FPS:.1f}s) at {width}x{height}")
    logger.info(f"Prompt: {prompt[:100]}...")

    # Load pipeline — T5 on CPU to save VRAM, transformer on GPU
    pipe = LTXPipeline.from_pretrained(
        LTX_MODEL_ID,
        cache_dir=LTX_CACHE_DIR,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    logger.info("Running LTX inference...")
    result = pipe(
        prompt=prompt,
        negative_prompt=(
            "worst quality, inconsistent motion, blurry, jittery, distorted, "
            "static, no movement, low quality, watermark, text, subtitle"
        ),
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    frames = result.frames[0]  # list of PIL images
    logger.info(f"Generated {len(frames)} frames, saving to {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=FPS, quality=8, codec="libx264")

    # Free VRAM
    del pipe, result
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="LTX-Video clip generation")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--duration", type=float, default=5.0, help="Clip duration in seconds (default 5)")
    parser.add_argument("--width", type=int, default=768, help="Frame width (default 768)")
    parser.add_argument("--height", type=int, default=512, help="Frame height (default 512)")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (default 50)")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG guidance scale (default 3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        out = generate_ltx_clip(
            prompt=args.prompt,
            output_path=args.output,
            duration_seconds=args.duration,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
        print(f"SUCCESS:{out}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        print(f"ERROR:{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
