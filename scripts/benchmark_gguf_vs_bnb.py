"""
Benchmark: GGUF Q4_0 vs BnB NF4 on Wan 2.1.

Runs 3 prompts × 2 backends × 81 frames and records:
  - Peak VRAM (GB)
  - Inference time (s/segment)
  - PSNR vs bf16 reference
  - SSIM vs bf16 reference

Usage:
    # With GGUF model already downloaded:
    python scripts/benchmark_gguf_vs_bnb.py \\
        --gguf-path /path/to/Wan2.1-T2V-14B-Q4_0.gguf

    # Download GGUF from HuggingFace first:
    huggingface-cli download bartowski/Wan2.1-T2V-14B-GGUF \\
        Wan2.1-T2V-14B-Q4_0.gguf --local-dir ./models/gguf/

    # Use 1.3B model for quick iteration (no GGUF available — BnB only):
    python scripts/benchmark_gguf_vs_bnb.py --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers

Results saved to docs/benchmark_gguf_vs_bnb.md and benchmark_outputs/gguf_vs_bnb_*.json.
"""

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PROMPTS = [
    "A serene mountain lake at sunrise, mist rising from the water, golden cinematic light",
    "A busy city street at night, neon lights reflecting on wet pavement, people walking",
    "A time-lapse of storm clouds over an open field, lightning illuminating the landscape",
]

MODEL_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
MODEL_1_3B = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class RunResult:
    backend: str
    prompt_idx: int
    prompt: str
    generation_time_s: float
    peak_vram_gb: Optional[float]
    psnr: Optional[float]
    ssim: Optional[float]
    error: Optional[str] = None


def _reset_vram():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb() -> Optional[float]:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return None


def _run_backend(backend: str, model_id: str, gguf_path: Optional[str], prompt: str, prompt_idx: int, args) -> RunResult:
    import numpy as np
    from src.wan.pipeline_factory import WanPipelineFactory
    from src.quant.config import QuantConfig
    from src.benchmark.metrics import VideoQualityMetrics

    _reset_vram()
    t0 = time.time()
    error = None
    frames = None

    try:
        if backend == "gguf":
            factory = WanPipelineFactory(
                model_id=model_id,
                engine="gguf",
                gguf_model_path=gguf_path,
                gguf_compute_dtype="bfloat16",
                enable_model_cpu_offload=True,
            )
            pipe = factory(None)
        elif backend == "bnb_nf4":
            factory = WanPipelineFactory(model_id=model_id, engine="bnb", enable_model_cpu_offload=True)
            pipe = factory(QuantConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))
        elif backend == "bf16":
            factory = WanPipelineFactory(model_id=model_id, engine="bnb", enable_model_cpu_offload=True)
            pipe = factory(QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False))
        else:
            raise ValueError(f"Unknown backend: {backend}")

        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        output = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            output_type="np",
        )
        frames = output.frames[0].astype("float32")

    except Exception as e:
        logger.error(f"[{backend}] prompt={prompt_idx} FAILED: {e}")
        error = str(e)

    elapsed = time.time() - t0
    peak_vram = _peak_vram_gb()

    _reset_vram()

    return RunResult(
        backend=backend,
        prompt_idx=prompt_idx,
        prompt=prompt,
        generation_time_s=elapsed,
        peak_vram_gb=peak_vram,
        psnr=None,
        ssim=None,
        error=error,
    ), frames


def run_benchmark(args):
    import numpy as np
    from src.benchmark.metrics import VideoQualityMetrics

    out_dir = Path("benchmark_outputs/gguf_vs_bnb")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_engine = VideoQualityMetrics(use_lpips=False)

    all_results = []
    reference_frames_by_prompt = {}

    # Determine which backends to run
    backends = ["bf16", "bnb_nf4"]
    if args.gguf_path:
        backends.append("gguf")
    else:
        logger.warning("--gguf-path not provided — skipping GGUF backend. "
                       "Download from: huggingface-cli download bartowski/Wan2.1-T2V-14B-GGUF "
                       "Wan2.1-T2V-14B-Q4_0.gguf --local-dir ./models/gguf/")

    prompts = BENCHMARK_PROMPTS if not args.single_prompt else [BENCHMARK_PROMPTS[0]]

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\n=== Prompt {prompt_idx + 1}/{len(prompts)} ===")
        logger.info(f"  {prompt[:80]}")

        for backend in backends:
            logger.info(f"  Running backend: {backend}")
            result, frames = _run_backend(backend, args.model_id, args.gguf_path, prompt, prompt_idx, args)

            if frames is not None:
                npy_path = out_dir / f"p{prompt_idx}_{backend}_frames.npy"
                np.save(npy_path, frames)

                if backend == "bf16":
                    reference_frames_by_prompt[prompt_idx] = frames
                elif prompt_idx in reference_frames_by_prompt:
                    ref = reference_frames_by_prompt[prompt_idx]
                    m = metrics_engine.compute_all(ref, frames, label=backend)
                    result.psnr = m["psnr"]
                    result.ssim = m["ssim"]

            all_results.append(result)
            logger.info(f"  {backend}: {result.generation_time_s:.1f}s | VRAM: {result.peak_vram_gb:.2f} GB | "
                        f"PSNR: {result.psnr:.2f if result.psnr else 'N/A'} | "
                        f"SSIM: {result.ssim:.4f if result.ssim else 'N/A'}")

    _write_report(all_results, out_dir, args)
    return all_results


def _write_report(results: list, out_dir: Path, args):
    # JSON
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"JSON results: {json_path}")

    # Markdown
    _write_markdown(results, args)


def _write_markdown(results: list, args):
    from collections import defaultdict

    doc_path = Path("docs/benchmark_gguf_vs_bnb.md")

    # Aggregate per backend
    by_backend = defaultdict(list)
    for r in results:
        by_backend[r.backend].append(r)

    lines = [
        "# Benchmark: GGUF Q4_0 vs BnB NF4 — Wan 2.1",
        "",
        f"**Model**: `{args.model_id}`  ",
        f"**Frames**: {args.num_frames} (81 = 5 s @ 16 fps)  ",
        f"**Resolution**: {args.width}×{args.height}  ",
        f"**Steps**: {args.num_inference_steps}  ",
        f"**Hardware**: RTX 4070 12 GB",
        "",
        "## Summary",
        "",
        "| Backend | Avg VRAM (GB) | Avg Time (s) | Avg PSNR (dB) | Avg SSIM |",
        "|---------|:-------------:|:------------:|:-------------:|:--------:|",
    ]

    for backend, rs in by_backend.items():
        valid = [r for r in rs if r.error is None]
        avg_vram = sum(r.peak_vram_gb for r in valid if r.peak_vram_gb) / max(len(valid), 1)
        avg_time = sum(r.generation_time_s for r in valid) / max(len(valid), 1)
        psnr_vals = [r.psnr for r in valid if r.psnr is not None]
        ssim_vals = [r.ssim for r in valid if r.ssim is not None]
        avg_psnr = sum(psnr_vals) / len(psnr_vals) if psnr_vals else None
        avg_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else None
        lines.append(
            f"| `{backend}` | {avg_vram:.2f} | {avg_time:.0f} | "
            f"{avg_psnr:.1f if avg_psnr else 'ref'} | "
            f"{avg_ssim:.4f if avg_ssim else 'ref'} |"
        )

    lines += [
        "",
        "## Per-Prompt Results",
        "",
    ]

    for r in results:
        status = f"FAILED: {r.error}" if r.error else (
            f"VRAM={r.peak_vram_gb:.2f} GB | {r.generation_time_s:.0f}s | "
            f"PSNR={r.psnr:.1f if r.psnr else 'ref'} dB | "
            f"SSIM={r.ssim:.4f if r.ssim else 'ref'}"
        )
        lines.append(f"- **P{r.prompt_idx} [{r.backend}]**: {status}")

    lines += [
        "",
        "## Interpretation",
        "",
        "- **PSNR > 30 dB** → perceptually close to bf16 reference",
        "- **SSIM > 0.95** → structural fidelity matches reference",
        "- **VRAM win** → backend fits in fewer GB, leaves more headroom for 3-layer RGBA stack",
        "",
        "## Recommendation",
        "",
        "_Fill in after running the benchmark._",
        "",
        "---",
        "_Generated by `scripts/benchmark_gguf_vs_bnb.py`_",
    ]

    doc_path.write_text("\n".join(lines))
    logger.info(f"Markdown report: {doc_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark GGUF Q4_0 vs BnB NF4 for Wan 2.1")
    p.add_argument("--gguf-path", default=None, help="Path to local .gguf file for Wan 2.1 14B")
    p.add_argument("--model-id", default=MODEL_14B, help="Base HF model ID for VAE/text encoder")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single-prompt", action="store_true", help="Only run first prompt (quick test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Safety check
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | {total_gb:.1f} GB VRAM")
        assert total_gb >= 10, f"Need at least 10 GB VRAM, have {total_gb:.1f} GB"
    else:
        logger.warning("No CUDA GPU — running on CPU (will be very slow)")

    run_benchmark(args)
