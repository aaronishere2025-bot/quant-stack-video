#!/usr/bin/env python3
"""
Model benchmark: compare quantization strategies on Wan 2.1 1.3B.

Configurations tested:
  1. single 4-bit (NF4)
  2. single 8-bit (LLM.int8)
  3. stacked 2x4-bit (average strategy)
  4. stacked 4x4-bit (average strategy)

Results saved to benchmark_outputs/model_benchmark/ as JSON + MP4.

Usage:
    cd /mnt/d/ai-workspace/projects/quant-stack-video
    source .venv/bin/activate
    python scripts/model_benchmark.py [--quick] [--runs N]
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("model_benchmark")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    "A serene mountain lake at sunrise, mist rising from water, golden light, wide shot",
    "A busy city street at night, neon lights reflecting on wet pavement, people walking",
    "A time-lapse of clouds moving over a green forest, light shifting from dawn to dusk",
]

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画面变形，畸形，毁容，"
    "形态畸形，手部畸形，腿部畸形，额外的手指，融合的手指，静止画面，动态不足，"
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config_label: str
    prompt_idx: int
    run_idx: int
    generation_time_s: float
    vram_peak_gb: float
    vram_after_gb: float
    output_path: str
    error: Optional[str] = None
    pass_times: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def vram_peak_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def reset_vram():
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def free_pipeline(pipe):
    """Delete a pipeline and free GPU memory."""
    try:
        del pipe
    except Exception:
        pass
    reset_vram()
    time.sleep(1)


# ---------------------------------------------------------------------------
# Single-pass run
# ---------------------------------------------------------------------------

def run_single_pass(
    model_id: str,
    quant_type: str,
    prompt: str,
    prompt_idx: int,
    run_idx: int,
    output_dir: Path,
    steps: int,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
) -> RunResult:
    import torch
    from src.quant.config import QuantConfig
    from src.wan.pipeline_factory import WanPipelineFactory
    from src.wan.generate import _save_video

    label = f"{quant_type}-single"
    out_mp4 = str(output_dir / f"p{prompt_idx}_r{run_idx}_{label}.mp4")
    logger.info(f"  [{label}] prompt={prompt_idx} run={run_idx} starting...")

    if quant_type == "4bit":
        qcfg = QuantConfig(load_in_4bit=True, load_in_8bit=False)
    elif quant_type == "8bit":
        qcfg = QuantConfig(load_in_4bit=False, load_in_8bit=True, quant_type="8bit")
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")

    reset_vram()
    t0 = time.time()

    try:
        factory = WanPipelineFactory(
            model_id=model_id,
            enable_model_cpu_offload=True,
            enable_sequential_cpu_offload=False,
            enable_vae_slicing=True,
            enable_vae_tiling=False,
        )
        pipe = factory(qcfg)

        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        output = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
            generator=gen,
            output_type="np",
        )

        elapsed = time.time() - t0
        peak = vram_peak_gb()
        current = vram_gb()

        frames = output.frames[0].astype("float32")
        _save_video(frames, out_mp4, fps=fps)

        logger.info(f"  [{label}] done in {elapsed:.1f}s | peak VRAM: {peak:.2f} GB | current: {current:.2f} GB")

        result = RunResult(
            config_label=label,
            prompt_idx=prompt_idx,
            run_idx=run_idx,
            generation_time_s=elapsed,
            vram_peak_gb=peak,
            vram_after_gb=current,
            output_path=out_mp4,
        )

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  [{label}] FAILED after {elapsed:.1f}s: {repr(e)}", exc_info=True)
        result = RunResult(
            config_label=label,
            prompt_idx=prompt_idx,
            run_idx=run_idx,
            generation_time_s=elapsed,
            vram_peak_gb=vram_peak_gb(),
            vram_after_gb=vram_gb(),
            output_path=out_mp4,
            error=repr(e),
        )

    finally:
        try:
            free_pipeline(pipe)
        except Exception:
            reset_vram()

    return result


# ---------------------------------------------------------------------------
# Stacked run
# ---------------------------------------------------------------------------

def run_stacked(
    model_id: str,
    num_passes: int,
    strategy: str,
    prompt: str,
    prompt_idx: int,
    run_idx: int,
    output_dir: Path,
    steps: int,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
) -> RunResult:
    from src.quant.config import QuantConfig, StackConfig
    from src.quant.engine import QuantStackEngine
    from src.wan.pipeline_factory import WanPipelineFactory
    from src.wan.generate import _save_video

    label = f"{num_passes}x4bit-{strategy}"
    out_mp4 = str(output_dir / f"p{prompt_idx}_r{run_idx}_{label}.mp4")
    logger.info(f"  [{label}] prompt={prompt_idx} run={run_idx} starting...")

    reset_vram()
    t0 = time.time()

    try:
        factory = WanPipelineFactory(
            model_id=model_id,
            enable_model_cpu_offload=True,
            enable_sequential_cpu_offload=False,
            enable_vae_slicing=True,
            enable_vae_tiling=False,
        )

        stack_cfg = StackConfig(
            num_passes=num_passes,
            stacking_strategy=strategy,
        )
        engine = QuantStackEngine(stack_cfg)

        result_data = engine.run_stacked(
            pipeline_factory=factory,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
            seed=seed,
        )

        elapsed = time.time() - t0
        peak = vram_peak_gb()
        current = vram_gb()

        frames = result_data["frames"]
        pass_times = result_data.get("pass_times", [])
        _save_video(frames, out_mp4, fps=fps)

        logger.info(f"  [{label}] done in {elapsed:.1f}s | peak VRAM: {peak:.2f} GB | current: {current:.2f} GB")

        result = RunResult(
            config_label=label,
            prompt_idx=prompt_idx,
            run_idx=run_idx,
            generation_time_s=elapsed,
            vram_peak_gb=peak,
            vram_after_gb=current,
            output_path=out_mp4,
            pass_times=pass_times,
        )

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  [{label}] FAILED after {elapsed:.1f}s: {repr(e)}", exc_info=True)
        result = RunResult(
            config_label=label,
            prompt_idx=prompt_idx,
            run_idx=run_idx,
            generation_time_s=elapsed,
            vram_peak_gb=vram_peak_gb(),
            vram_after_gb=vram_gb(),
            output_path=out_mp4,
            error=repr(e),
        )

    finally:
        reset_vram()

    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: List[RunResult]):
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)

    # Group by config_label
    by_config: Dict[str, List[RunResult]] = {}
    for r in results:
        by_config.setdefault(r.config_label, []).append(r)

    logger.info(f"{'Config':<25} {'Runs':>5} {'Errors':>6} {'Avg Time':>10} {'Peak VRAM':>10}")
    logger.info("-" * 60)
    for label, runs in by_config.items():
        ok = [r for r in runs if r.error is None]
        errs = len(runs) - len(ok)
        avg_time = sum(r.generation_time_s for r in ok) / len(ok) if ok else 0.0
        avg_vram = sum(r.vram_peak_gb for r in ok) / len(ok) if ok else 0.0
        logger.info(f"  {label:<23} {len(runs):>5} {errs:>6} {avg_time:>9.1f}s {avg_vram:>9.2f}GB")

    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wan 2.1 1.3B model benchmark")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per config per prompt")
    parser.add_argument("--prompts", type=int, default=3, help="Number of prompts to test (1-3)")
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps (20 for benchmark speed)")
    parser.add_argument("--frames", type=int, default=49, help="Frames (49=~3s, 81=~5s)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="benchmark_outputs/model_benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 prompt, 20 steps, 49 frames")
    parser.add_argument("--skip-8bit", action="store_true", help="Skip 8-bit (slower on some setups)")
    parser.add_argument("--stacks-only", action="store_true", help="Only run stacked configs")
    args = parser.parse_args()

    if args.quick:
        args.prompts = 1
        args.steps = 20
        args.frames = 49

    # Check VRAM before starting
    try:
        import torch
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {total_gb:.1f} GB total, {free_gb:.1f} GB free")
        if total_gb < 8:
            logger.error("Need at least 8 GB VRAM for Wan 1.3B benchmarks. Aborting.")
            sys.exit(1)
    except Exception as e:
        logger.warning(f"Could not check VRAM: {e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS[:max(1, min(args.prompts, len(PROMPTS)))]
    runs = max(1, args.runs)

    logger.info(f"\nBenchmark config:")
    logger.info(f"  model:   {args.model}")
    logger.info(f"  prompts: {len(prompts)}")
    logger.info(f"  runs:    {runs} per config per prompt")
    logger.info(f"  steps:   {args.steps}")
    logger.info(f"  frames:  {args.frames}")
    logger.info(f"  output:  {output_dir}")

    all_results: List[RunResult] = []
    report_path = output_dir / "report.json"

    def save_incremental():
        """Save results so far — useful if the run crashes partway through."""
        data = [asdict(r) for r in all_results]
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"  Results saved to {report_path} ({len(all_results)} entries)")

    # Define configurations to run
    configs = []
    if not args.stacks_only:
        configs.append(("4bit", None, None))
        if not args.skip_8bit:
            configs.append(("8bit", None, None))
    configs.append(("stack", 2, "average"))
    configs.append(("stack", 4, "average"))

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {prompt_idx+1}/{len(prompts)}: {prompt[:70]}...")
        logger.info(f"{'='*60}")

        for run_idx in range(runs):
            for cfg in configs:
                kind, num_passes, strategy = cfg

                if kind in ("4bit", "8bit"):
                    result = run_single_pass(
                        model_id=args.model,
                        quant_type=kind,
                        prompt=prompt,
                        prompt_idx=prompt_idx,
                        run_idx=run_idx,
                        output_dir=output_dir,
                        steps=args.steps,
                        seed=args.seed + run_idx,
                        height=args.height,
                        width=args.width,
                        num_frames=args.frames,
                        fps=args.fps,
                    )
                else:
                    result = run_stacked(
                        model_id=args.model,
                        num_passes=num_passes,
                        strategy=strategy,
                        prompt=prompt,
                        prompt_idx=prompt_idx,
                        run_idx=run_idx,
                        output_dir=output_dir,
                        steps=args.steps,
                        seed=args.seed + run_idx,
                        height=args.height,
                        width=args.width,
                        num_frames=args.frames,
                        fps=args.fps,
                    )

                all_results.append(result)
                save_incremental()

    print_summary(all_results)

    # Final save
    save_incremental()
    logger.info(f"\nBenchmark complete. Results: {report_path}")
    logger.info(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
