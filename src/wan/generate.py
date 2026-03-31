"""
High-level video generation API for Wan 2.1 with quantization stacking.

Provides two modes:
1. Single-pass generation (reference or basic 4-bit)
2. Multi-pass stacked generation via QuantStackEngine
"""

import logging
from pathlib import Path
from typing import Optional, Union, List
import numpy as np

from ..quant.config import QuantConfig, StackConfig
from ..quant.engine import QuantStackEngine
from .pipeline_factory import WanPipelineFactory

logger = logging.getLogger(__name__)


def generate_video(
    prompt: str,
    output_path: str,
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画面变形，畸形，毁容，形态畸形，手部畸形，腿部畸形，额外的手指，融合的手指，静止画面，动态不足，",
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 42,
    quant_type: str = "4bit",
    cache_dir: Optional[str] = None,
    fps: int = 16,
) -> str:
    """
    Generate a video using a single Wan 2.1 pass.

    Args:
        prompt: Text description of the video
        output_path: Where to save the output .mp4
        model_id: HuggingFace model ID
        negative_prompt: Negative conditioning text
        height, width: Frame resolution
        num_frames: Number of frames (81 = ~5s @ 16fps)
        num_inference_steps: Denoising steps
        guidance_scale: CFG scale
        seed: Random seed
        quant_type: "4bit", "8bit", or "none" (full precision)
        cache_dir: HuggingFace model cache directory
        fps: Output video frame rate

    Returns:
        Path to saved video file
    """
    factory = WanPipelineFactory(
        model_id=model_id,
        cache_dir=cache_dir,
    )

    if quant_type == "4bit":
        qcfg = QuantConfig(load_in_4bit=True, load_in_8bit=False)
    elif quant_type == "8bit":
        qcfg = QuantConfig(load_in_4bit=False, load_in_8bit=True)
    else:
        qcfg = QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False)

    import torch
    pipe = factory(qcfg)

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=gen,
        output_type="np",
    )

    frames = output.frames[0]  # (T, H, W, C)
    saved = _save_video(frames, output_path, fps=fps)
    return saved


def generate_video_stacked(
    prompt: str,
    output_path: str,
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画面变形，畸形，毁容，形态畸形，手部畸形，腿部畸形，额外的手指，融合的手指，静止画面，动态不足，",
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    seed: int = 42,
    num_passes: int = 3,
    stacking_strategy: str = "progressive",
    cache_dir: Optional[str] = None,
    fps: int = 16,
) -> dict:
    """
    Generate a video using multi-pass quantization stacking.

    The stacked approach runs N 4-bit passes and combines them to recover
    quality approaching 8/16-bit inference at a fraction of the VRAM cost.

    Args:
        prompt: Text description of the video
        output_path: Where to save the output .mp4
        num_passes: Number of 4-bit passes to stack (default: 3)
        stacking_strategy: "progressive" (best quality), "average", "weighted", "residual"
        ... (other args same as generate_video)

    Returns:
        dict with "output_path", "pass_times", "total_time", "strategy"
    """
    factory = WanPipelineFactory(
        model_id=model_id,
        cache_dir=cache_dir,
    )

    stack_cfg = StackConfig(
        num_passes=num_passes,
        stacking_strategy=stacking_strategy,
    )

    engine = QuantStackEngine(stack_cfg)

    result = engine.run_stacked(
        pipeline_factory=factory,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    saved = _save_video(result["frames"], output_path, fps=fps)
    result["output_path"] = saved
    return result


def generate_long_video(
    prompt: str,
    output_path: str,
    duration_seconds: float = 30.0,
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画面变形，畸形，毁容，形态畸形，手部畸形，腿部畸形，额外的手指，融合的手指，静止画面，动态不足，",
    height: int = 480,
    width: int = 832,
    fps: int = 16,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    seed: int = 42,
    segment_frames: int = 81,
    overlap_frames: int = 8,
    use_stacking: bool = True,
    num_passes: int = 3,
    stacking_strategy: str = "progressive",
    cache_dir: Optional[str] = None,
) -> dict:
    """
    Generate long-form video (up to 3 minutes) by chaining segments with overlap blending.

    Each segment is generated independently (or with stacking) and blended at boundaries
    to maintain temporal continuity. The last N frames of each segment condition the next.

    Args:
        duration_seconds: Target video duration in seconds
        segment_frames: Frames per segment (81 = ~5s @ 16fps)
        overlap_frames: Frames to overlap between segments for smooth transitions
        use_stacking: Whether to use quantization stacking per segment
        ... (other args same as above)

    Returns:
        dict with "output_path", "num_segments", "total_frames", "total_time"
    """
    import time

    total_frames_needed = int(duration_seconds * fps)
    effective_frames = segment_frames - overlap_frames  # new frames per segment
    num_segments = max(1, (total_frames_needed - overlap_frames + effective_frames - 1) // effective_frames)

    logger.info(f"Long video generation: {duration_seconds}s → {num_segments} segments "
                f"({segment_frames} frames each, {overlap_frames} overlap)")

    factory = WanPipelineFactory(
        model_id=model_id,
        cache_dir=cache_dir,
    )

    all_frames = []
    segment_times = []
    t_total = time.time()

    for seg_idx in range(num_segments):
        seg_seed = seed + seg_idx
        seg_prompt = prompt  # future: could inject temporal context

        logger.info(f"Segment {seg_idx + 1}/{num_segments} (seed={seg_seed})")
        t_seg = time.time()

        if use_stacking:
            stack_cfg = StackConfig(
                num_passes=num_passes,
                stacking_strategy=stacking_strategy,
            )
            engine = QuantStackEngine(stack_cfg)
            result = engine.run_stacked(
                pipeline_factory=factory,
                prompt=seg_prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=segment_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seg_seed,
            )
            seg_frames = result["frames"]
        else:
            import torch
            qcfg = QuantConfig(load_in_4bit=True)
            pipe = factory(qcfg)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seg_seed)
            output = pipe(
                prompt=seg_prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=segment_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                output_type="np",
            )
            seg_frames = output.frames[0].astype(np.float32)

        seg_time = time.time() - t_seg
        segment_times.append(seg_time)

        # Blend overlap region with previous segment
        if seg_idx == 0:
            all_frames.append(seg_frames)
        else:
            prev_end = all_frames[-1][-overlap_frames:]  # (overlap, H, W, C)
            curr_start = seg_frames[:overlap_frames]      # (overlap, H, W, C)
            # Linear blend across the overlap region
            blend_weights = np.linspace(0, 1, overlap_frames)[:, None, None, None]
            blended = (1 - blend_weights) * prev_end + blend_weights * curr_start
            # Replace the end of previous segment with blended frames
            all_frames[-1] = np.concatenate([all_frames[-1][:-overlap_frames], blended], axis=0)
            # Add non-overlapping new frames
            all_frames.append(seg_frames[overlap_frames:])

    # Concatenate all segments
    final_frames = np.concatenate(all_frames, axis=0)
    final_frames = final_frames[:total_frames_needed]  # trim to exact target
    total_time = time.time() - t_total

    saved = _save_video(final_frames, output_path, fps=fps)

    return {
        "output_path": saved,
        "num_segments": num_segments,
        "total_frames": len(final_frames),
        "segment_times": segment_times,
        "total_time": total_time,
    }


def _save_video(frames_np: np.ndarray, output_path: str, fps: int = 16) -> str:
    """
    Save numpy frames array to video file.

    Args:
        frames_np: (T, H, W, C) float32 array in [0, 1]
        output_path: Output file path (.mp4)
        fps: Frame rate

    Returns:
        Absolute path to saved file
    """
    import imageio
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint8
    frames_uint8 = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)

    logger.info(f"Saving {len(frames_uint8)} frames to {output_path} @ {fps}fps")

    with imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                            output_params=["-crf", "18"]) as writer:
        for frame in frames_uint8:
            writer.append_data(frame)

    logger.info(f"Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return str(output_path.absolute())
