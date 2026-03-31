"""
Benchmark runner: side-by-side comparison of quantization configurations.

Runs multiple configurations (e.g., bf16 reference, 4-bit, 8-bit, 2-pass stack,
3-pass stack) on the same prompt and reports quality metrics.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

from ..quant.config import QuantConfig, StackConfig
from ..quant.engine import QuantStackEngine
from .metrics import VideoQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    # Prompts to test
    prompts: List[str] = field(default_factory=lambda: [
        "A serene mountain lake at sunrise, mist rising from the water, golden light",
        "A busy city street at night, neon lights reflecting on wet pavement, people walking",
        "A time-lapse of clouds moving over a forest, light changing from dawn to dusk",
    ])

    # Resolution and duration
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16

    # Inference
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 42

    # Model
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    cache_dir: Optional[str] = None

    # Output
    output_dir: str = "benchmark_outputs"

    # Which configurations to benchmark
    run_reference: bool = True  # bf16 baseline
    run_4bit_single: bool = True  # single 4-bit pass
    run_8bit_single: bool = True  # single 8-bit pass
    stack_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"num_passes": 2, "strategy": "average"},
        {"num_passes": 3, "strategy": "average"},
        {"num_passes": 2, "strategy": "progressive"},
        {"num_passes": 3, "strategy": "progressive"},
    ])


@dataclass
class BenchmarkResult:
    """Result for a single (config, prompt) combination."""
    config_label: str
    prompt: str
    prompt_idx: int
    output_path: str
    generation_time: float
    pass_times: List[float]
    metrics: Optional[Dict[str, Any]]
    vram_gb: Optional[float]
    error: Optional[str] = None


class BenchmarkRunner:
    """
    Orchestrates side-by-side benchmark comparisons.

    Usage:
        config = BenchmarkConfig(prompts=["..."], num_frames=49)
        runner = BenchmarkRunner(config)
        results = runner.run()
        runner.save_report(results, "benchmark_report.json")
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics_engine = VideoQualityMetrics(use_lpips=True)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[BenchmarkResult]:
        """Run all configured benchmarks and return results."""
        all_results = []

        for prompt_idx, prompt in enumerate(self.config.prompts):
            logger.info(f"\n{'='*60}")
            logger.info(f"Prompt {prompt_idx+1}/{len(self.config.prompts)}: {prompt[:60]}...")
            logger.info(f"{'='*60}")

            prompt_results = self._run_prompt(prompt, prompt_idx)
            all_results.extend(prompt_results)

        # Print summary report
        self._print_summary(all_results)
        return all_results

    def _run_prompt(self, prompt: str, prompt_idx: int) -> List[BenchmarkResult]:
        from ..wan.pipeline_factory import WanPipelineFactory

        factory = WanPipelineFactory(
            model_id=self.config.model_id,
            cache_dir=self.config.cache_dir,
        )

        results = []
        reference_frames = None

        common_kw = dict(
            prompt=prompt,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            seed=self.config.seed,
            output_type="np",
        )

        # --- Reference: bf16 ---
        if self.config.run_reference:
            result = self._run_single(
                factory=factory,
                qcfg=QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False),
                label="bf16-reference",
                prompt=prompt,
                prompt_idx=prompt_idx,
                common_kw=common_kw,
                reference_frames=None,  # is the reference
            )
            results.append(result)
            if result.error is None:
                reference_frames = np.load(result.output_path.replace(".mp4", "_frames.npy"))

        # --- Single-pass 4-bit ---
        if self.config.run_4bit_single:
            result = self._run_single(
                factory=factory,
                qcfg=QuantConfig(load_in_4bit=True),
                label="4bit-single-nf4",
                prompt=prompt,
                prompt_idx=prompt_idx,
                common_kw=common_kw,
                reference_frames=reference_frames,
            )
            results.append(result)

        # --- Single-pass 8-bit ---
        if self.config.run_8bit_single:
            result = self._run_single(
                factory=factory,
                qcfg=QuantConfig(load_in_4bit=False, load_in_8bit=True),
                label="8bit-single",
                prompt=prompt,
                prompt_idx=prompt_idx,
                common_kw=common_kw,
                reference_frames=reference_frames,
            )
            results.append(result)

        # --- Stacked configurations ---
        for stack_kw in self.config.stack_configs:
            num_passes = stack_kw["num_passes"]
            strategy = stack_kw["strategy"]
            label = f"{num_passes}x4bit-{strategy}"

            result = self._run_stacked(
                factory=factory,
                stack_cfg=StackConfig(num_passes=num_passes, stacking_strategy=strategy),
                label=label,
                prompt=prompt,
                prompt_idx=prompt_idx,
                common_kw=common_kw,
                reference_frames=reference_frames,
            )
            results.append(result)

        return results

    def _run_single(
        self,
        factory,
        qcfg: QuantConfig,
        label: str,
        prompt: str,
        prompt_idx: int,
        common_kw: dict,
        reference_frames,
    ) -> BenchmarkResult:
        import torch

        output_mp4 = str(self.output_dir / f"p{prompt_idx}_{label}.mp4")
        output_npy = output_mp4.replace(".mp4", "_frames.npy")

        logger.info(f"  Running: {label}")
        t0 = time.time()

        try:
            self._reset_vram_stats()
            pipe = factory(qcfg)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(self.config.seed)

            output = pipe(
                **{k: v for k, v in common_kw.items() if k != "seed"},
                generator=gen,
            )
            frames = output.frames[0].astype(np.float32)
            elapsed = time.time() - t0
            vram = self.metrics_engine.vram_usage()

            # Save frames as npy for metrics computation
            np.save(output_npy, frames)

            # Save video
            from ..wan.generate import _save_video
            _save_video(frames, output_mp4, fps=self.config.fps)

            # Compute metrics if reference available
            metrics = None
            if reference_frames is not None:
                metrics = self.metrics_engine.compute_all(reference_frames, frames, label=label)
            elif label == "bf16-reference":
                # Self-metrics (just temporal consistency)
                metrics = {
                    "label": label,
                    "psnr": float("inf"),
                    "ssim": 1.0,
                    "lpips": 0.0,
                    "temporal_consistency_ref": self.metrics_engine.temporal_consistency(frames),
                    "temporal_consistency_gen": self.metrics_engine.temporal_consistency(frames),
                    "temporal_consistency_delta": 0.0,
                }

            logger.info(f"  {label} done in {elapsed:.1f}s | VRAM: {vram:.2f} GB")
            return BenchmarkResult(
                config_label=label,
                prompt=prompt,
                prompt_idx=prompt_idx,
                output_path=output_mp4,
                generation_time=elapsed,
                pass_times=[elapsed],
                metrics=metrics,
                vram_gb=vram,
            )

        except Exception as e:
            logger.error(f"  {label} FAILED: {e}")
            return BenchmarkResult(
                config_label=label,
                prompt=prompt,
                prompt_idx=prompt_idx,
                output_path=output_mp4,
                generation_time=time.time() - t0,
                pass_times=[],
                metrics=None,
                vram_gb=None,
                error=str(e),
            )

    def _run_stacked(
        self,
        factory,
        stack_cfg: StackConfig,
        label: str,
        prompt: str,
        prompt_idx: int,
        common_kw: dict,
        reference_frames,
    ) -> BenchmarkResult:
        output_mp4 = str(self.output_dir / f"p{prompt_idx}_{label}.mp4")
        output_npy = output_mp4.replace(".mp4", "_frames.npy")

        logger.info(f"  Running stacked: {label}")
        t0 = time.time()

        try:
            self._reset_vram_stats()
            engine = QuantStackEngine(stack_cfg)

            result = engine.run_stacked(
                pipeline_factory=factory,
                prompt=common_kw["prompt"],
                height=common_kw["height"],
                width=common_kw["width"],
                num_frames=common_kw["num_frames"],
                num_inference_steps=common_kw["num_inference_steps"],
                guidance_scale=common_kw["guidance_scale"],
                seed=self.config.seed,
            )

            frames = result["frames"]
            elapsed = time.time() - t0
            vram = self.metrics_engine.vram_usage()

            np.save(output_npy, frames)

            from ..wan.generate import _save_video
            _save_video(frames, output_mp4, fps=self.config.fps)

            metrics = None
            if reference_frames is not None:
                metrics = self.metrics_engine.compute_all(reference_frames, frames, label=label)

            logger.info(f"  {label} done in {elapsed:.1f}s | VRAM: {vram:.2f} GB")
            return BenchmarkResult(
                config_label=label,
                prompt=prompt,
                prompt_idx=prompt_idx,
                output_path=output_mp4,
                generation_time=elapsed,
                pass_times=result.get("pass_times", []),
                metrics=metrics,
                vram_gb=vram,
            )

        except Exception as e:
            logger.error(f"  {label} FAILED: {e}")
            return BenchmarkResult(
                config_label=label,
                prompt=prompt,
                prompt_idx=prompt_idx,
                output_path=output_mp4,
                generation_time=time.time() - t0,
                pass_times=[],
                metrics=None,
                vram_gb=None,
                error=str(e),
            )

    def _reset_vram_stats(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _print_summary(self, results: List[BenchmarkResult]):
        successful = [r for r in results if r.error is None and r.metrics is not None]
        if not successful:
            logger.warning("No successful results with metrics to summarize")
            return

        metrics_list = [r.metrics for r in successful]
        timing_list = [(r.config_label, r.generation_time) for r in successful]

        report = self.metrics_engine.format_report(metrics_list)
        logger.info(report)

        logger.info("\nTiming summary:")
        for label, t in timing_list:
            logger.info(f"  {label:<30} {t:>8.1f}s")

    def save_report(self, results: List[BenchmarkResult], report_path: str):
        """Save benchmark results to JSON."""
        serializable = []
        for r in results:
            d = asdict(r)
            # Convert any non-serializable numpy values
            if d["metrics"]:
                for k, v in d["metrics"].items():
                    if isinstance(v, (np.float32, np.float64)):
                        d["metrics"][k] = float(v)
                    elif isinstance(v, float) and v == float("inf"):
                        d["metrics"][k] = "inf"
            serializable.append(d)

        with open(report_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Benchmark report saved to {report_path}")
