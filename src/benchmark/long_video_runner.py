"""
Long-video benchmark: layered quant stacking vs sequential single-pass generation.

Tests two approaches for generating videos up to 3 minutes:
  1. sequential-4bit: single-pass 4-bit quant, segments chained with overlap blending
  2. layered-3x4bit-progressive: 3-pass progressive stacked 4-bit per segment

Metrics tracked per segment:
  - PSNR / SSIM vs previous segment boundary (temporal fidelity across cuts)
  - Temporal consistency (frame-to-frame smoothness within segment)
  - Boundary score: mean pixel delta at the overlap join
  - Generation time per segment

Drift is tracked as metric degradation over the full video.
"""

import gc
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LongVideoBenchmarkConfig:
    """Configuration for the long-video benchmark."""

    prompts: List[str] = field(default_factory=lambda: [
        "A hiker walks through an ancient redwood forest, sunbeams filtering through the canopy, mist on the ground",
        "A coastal city at sunset, time-lapse of traffic and lights activating, waves rolling in",
    ])

    # Duration and segmentation
    duration_seconds: float = 30.0     # start with 30s; set to 180.0 for 3-min test
    segment_frames: int = 81           # 5s per segment @ 16fps (must be 4k+1)
    overlap_frames: int = 8            # pixel-space overlap for blending baseline
    fps: int = 16

    # Resolution
    height: int = 480
    width: int = 832

    # Inference
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 42

    # Model
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    cache_dir: Optional[str] = None

    # Output
    output_dir: str = "benchmark_outputs/long_video"

    # Which configs to run
    configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "label": "bf16-baseline",
            "use_stacking": False,
            "quant_type": "none",
        },
        {
            "label": "sequential-4bit",
            "use_stacking": False,
            "quant_type": "4bit",
        },
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
    ])


@dataclass
class SegmentMetrics:
    """Quality metrics for a single generated segment."""
    segment_idx: int
    generation_time: float
    temporal_consistency: float          # lower = smoother within segment
    boundary_score: Optional[float]      # pixel delta at join with prev segment (lower = better)
    psnr_vs_prev_end: Optional[float]    # PSNR at overlap zone vs prev segment end
    ssim_vs_prev_end: Optional[float]


@dataclass
class LongVideoResult:
    """Result for one config x prompt run."""
    config_label: str
    prompt: str
    prompt_idx: int
    duration_seconds: float
    num_segments: int
    total_time: float
    output_path: Optional[str]
    segment_metrics: List[SegmentMetrics]
    error: Optional[str] = None

    @property
    def mean_generation_time_per_segment(self) -> float:
        if not self.segment_metrics:
            return 0.0
        return float(np.mean([s.generation_time for s in self.segment_metrics]))

    @property
    def mean_temporal_consistency(self) -> float:
        if not self.segment_metrics:
            return 0.0
        return float(np.mean([s.temporal_consistency for s in self.segment_metrics]))

    @property
    def temporal_consistency_drift(self) -> float:
        """How much temporal consistency degrades from first to last segment."""
        scores = [s.temporal_consistency for s in self.segment_metrics]
        if len(scores) < 2:
            return 0.0
        # Higher drift means increasing jitter over time (worse)
        return float(scores[-1] - scores[0])

    @property
    def mean_boundary_score(self) -> Optional[float]:
        scores = [s.boundary_score for s in self.segment_metrics if s.boundary_score is not None]
        return float(np.mean(scores)) if scores else None


class LongVideoBenchmarkRunner:
    """
    Benchmarks layered quant stacking vs sequential generation for long-form video.

    Runs each configuration on the same prompt(s) and reports per-segment quality
    and continuity metrics to surface drift over time.

    Usage:
        config = LongVideoBenchmarkConfig(duration_seconds=30.0)
        runner = LongVideoBenchmarkRunner(config)
        results = runner.run()
        runner.save_report(results, "long_video_report.json")
    """

    def __init__(self, config: LongVideoBenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[LongVideoResult]:
        all_results = []

        for prompt_idx, prompt in enumerate(self.config.prompts):
            logger.info(f"\n{'='*65}")
            logger.info(f"Prompt {prompt_idx+1}/{len(self.config.prompts)}: {prompt[:60]}...")
            logger.info(f"Duration: {self.config.duration_seconds}s | Seed: {self.config.seed}")
            logger.info(f"{'='*65}")

            for cfg in self.config.configs:
                result = self._run_config(cfg, prompt, prompt_idx)
                all_results.append(result)

        self._print_summary(all_results)
        return all_results

    def _run_config(
        self,
        cfg: Dict[str, Any],
        prompt: str,
        prompt_idx: int,
    ) -> LongVideoResult:
        from ..quant.config import QuantConfig, StackConfig
        from ..quant.engine import QuantStackEngine
        from ..wan.pipeline_factory import WanPipelineFactory
        from ..wan.generate import _save_video
        from .metrics import VideoQualityMetrics

        label = cfg["label"]
        use_stacking = cfg.get("use_stacking", False)
        num_passes = cfg.get("num_passes", 3)
        strategy = cfg.get("strategy", "progressive")
        quant_type = cfg.get("quant_type", "4bit")

        logger.info(f"\n  Config: {label} | stacking={use_stacking}")

        fps = self.config.fps
        segment_frames = self.config.segment_frames
        overlap_frames = self.config.overlap_frames
        duration = self.config.duration_seconds
        seed = self.config.seed

        total_frames_needed = int(duration * fps)
        effective_frames = segment_frames - overlap_frames
        num_segments = max(1, (total_frames_needed - overlap_frames + effective_frames - 1) // effective_frames)

        factory = WanPipelineFactory(
            model_id=self.config.model_id,
            cache_dir=self.config.cache_dir,
        )
        metrics_engine = VideoQualityMetrics(use_lpips=False)

        all_frames = []
        segment_metrics_list = []
        t_total = time.time()

        for seg_idx in range(num_segments):
            seg_seed = seed + seg_idx
            logger.info(f"    Segment {seg_idx+1}/{num_segments} (seed={seg_seed})")
            t_seg = time.time()

            try:
                seg_frames = self._generate_segment(
                    factory=factory,
                    prompt=prompt,
                    seg_seed=seg_seed,
                    use_stacking=use_stacking,
                    num_passes=num_passes,
                    strategy=strategy,
                    quant_type=quant_type,
                )
            except Exception as exc:
                logger.error(f"    Segment {seg_idx+1} FAILED: {exc}")
                return LongVideoResult(
                    config_label=label,
                    prompt=prompt,
                    prompt_idx=prompt_idx,
                    duration_seconds=duration,
                    num_segments=num_segments,
                    total_time=time.time() - t_total,
                    output_path=None,
                    segment_metrics=segment_metrics_list,
                    error=str(exc),
                )

            seg_time = time.time() - t_seg

            # Temporal consistency within this segment
            tc = metrics_engine.temporal_consistency(seg_frames)

            # Boundary metrics vs previous segment end
            boundary_score = None
            psnr_vs_prev = None
            ssim_vs_prev = None

            if seg_idx > 0 and len(all_frames) > 0:
                prev_end = all_frames[-1][-overlap_frames:]   # (O, H, W, C)
                curr_start = seg_frames[:overlap_frames]       # (O, H, W, C)

                if prev_end.shape == curr_start.shape:
                    # Mean absolute pixel delta at the boundary
                    boundary_score = float(np.mean(np.abs(prev_end - curr_start)))
                    # PSNR and SSIM at the join zone
                    psnr_vs_prev = metrics_engine.psnr(prev_end, curr_start)
                    ssim_vs_prev = metrics_engine.ssim_video(prev_end, curr_start)

            segment_metrics_list.append(SegmentMetrics(
                segment_idx=seg_idx,
                generation_time=seg_time,
                temporal_consistency=tc,
                boundary_score=boundary_score,
                psnr_vs_prev_end=psnr_vs_prev,
                ssim_vs_prev_end=ssim_vs_prev,
            ))

            # Blend and accumulate frames
            if seg_idx == 0:
                all_frames.append(seg_frames)
            else:
                prev_end_frames = all_frames[-1][-overlap_frames:]
                curr_start_frames = seg_frames[:overlap_frames]
                blend_w = np.linspace(0, 1, overlap_frames)[:, None, None, None]
                blended = (1 - blend_w) * prev_end_frames + blend_w * curr_start_frames
                all_frames[-1] = np.concatenate(
                    [all_frames[-1][:-overlap_frames], blended], axis=0
                )
                all_frames.append(seg_frames[overlap_frames:])

            logger.info(
                f"    seg={seg_idx+1} time={seg_time:.1f}s "
                f"tc={tc:.4f}"
                + (f" boundary={boundary_score:.4f}" if boundary_score is not None else "")
                + (f" psnr@join={psnr_vs_prev:.1f}dB" if psnr_vs_prev is not None else "")
            )

            _free_memory()

        # Assemble final video
        final_frames = np.concatenate(all_frames, axis=0)[:total_frames_needed]
        total_time = time.time() - t_total

        output_path = str(self.output_dir / f"p{prompt_idx}_{label}.mp4")
        _save_video(final_frames, output_path, fps=fps)

        logger.info(
            f"  {label} complete: {total_time:.1f}s total, "
            f"{len(final_frames)} frames, {len(final_frames)/fps:.1f}s video"
        )

        return LongVideoResult(
            config_label=label,
            prompt=prompt,
            prompt_idx=prompt_idx,
            duration_seconds=duration,
            num_segments=num_segments,
            total_time=total_time,
            output_path=output_path,
            segment_metrics=segment_metrics_list,
        )

    def _generate_segment(
        self,
        factory,
        prompt: str,
        seg_seed: int,
        use_stacking: bool,
        num_passes: int,
        strategy: str,
        quant_type: str = "4bit",
    ) -> np.ndarray:
        """Generate one video segment; returns (T, H, W, C) float32 [0, 1]."""
        import torch
        from ..quant.config import QuantConfig, StackConfig
        from ..quant.engine import QuantStackEngine

        common_kw = dict(
            prompt=prompt,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.segment_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
        )

        if use_stacking:
            if quant_type == "none":
                # bf16 stacking: each pass runs at full precision
                pass_cfgs = [
                    QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False)
                    for _ in range(num_passes)
                ]
                stack_cfg = StackConfig(
                    num_passes=num_passes,
                    stacking_strategy=strategy,
                    pass_configs=pass_cfgs,
                )
            else:
                stack_cfg = StackConfig(num_passes=num_passes, stacking_strategy=strategy)
            engine = QuantStackEngine(stack_cfg)
            result = engine.run_stacked(
                pipeline_factory=factory,
                seed=seg_seed,
                **common_kw,
            )
            return result["frames"]
        else:
            if quant_type == "none":
                qcfg = QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False)
            elif quant_type == "8bit":
                qcfg = QuantConfig(quant_type="8bit", load_in_4bit=False, load_in_8bit=True)
            else:
                qcfg = QuantConfig(load_in_4bit=True)
            pipe = factory(qcfg)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seg_seed)
            output = pipe(
                **common_kw,
                generator=gen,
                output_type="np",
            )
            frames = output.frames[0].astype(np.float32)
            del pipe
            _free_memory()
            return frames

    def _print_summary(self, results: List[LongVideoResult]):
        successful = [r for r in results if r.error is None]
        if not successful:
            logger.warning("No successful results to summarize.")
            return

        logger.info("\n" + "=" * 75)
        logger.info("LONG VIDEO BENCHMARK SUMMARY")
        logger.info("=" * 75)
        header = (
            f"{'Config':<32} {'Seg/s':>6} {'TC mean':>8} {'TC drift':>9} "
            f"{'Boundary':>9} {'PSNR@join':>10}"
        )
        logger.info(header)
        logger.info("-" * 75)

        for r in successful:
            mean_seg_t = r.mean_generation_time_per_segment
            mean_tc = r.mean_temporal_consistency
            tc_drift = r.temporal_consistency_drift
            boundary = r.mean_boundary_score
            # Mean PSNR at all segment joins
            psnr_joins = [
                s.psnr_vs_prev_end for s in r.segment_metrics
                if s.psnr_vs_prev_end is not None
            ]
            mean_psnr_join = float(np.mean(psnr_joins)) if psnr_joins else float("nan")

            boundary_str = f"{boundary:.4f}" if boundary is not None else "N/A"
            psnr_str = f"{mean_psnr_join:.1f}" if not np.isnan(mean_psnr_join) else "N/A"

            logger.info(
                f"{r.config_label:<32} {mean_seg_t:>6.1f} {mean_tc:>8.4f} "
                f"{tc_drift:>+9.5f} {boundary_str:>9} {psnr_str:>10}"
            )

        logger.info("=" * 75)
        logger.info(
            "TC = temporal consistency (lower=smoother) | "
            "TC drift = last-first (positive=degrading) | "
            "Boundary = pixel delta at segment join (lower=better) | "
            "PSNR@join = quality at overlap boundary (higher=better)"
        )
        logger.info("=" * 75)

    def save_report(self, results: List[LongVideoResult], report_path: str):
        """Serialize results to JSON."""
        serializable = []
        for r in results:
            d = asdict(r)
            # Clean up non-serializable floats
            for sm in d.get("segment_metrics", []):
                for k, v in sm.items():
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        sm[k] = None
            serializable.append(d)

        with open(report_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Long video benchmark report saved to {report_path}")


def _free_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
