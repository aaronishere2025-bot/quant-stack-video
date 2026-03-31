"""
Quantization Stacking Engine

Core implementation of multi-pass 4-bit quantization for video generation.
The key insight: multiple 4-bit passes with different quant configurations,
when combined, can recover quality comparable to 8/16-bit inference.

Strategy overview:
- "average": N independent 4-bit passes → mean latents
- "weighted": N independent 4-bit passes → weighted sum of latents
- "residual": Pass 1 generates base latents; each subsequent pass
              adds a 4-bit residual correction term
- "progressive": Pass 1 generates latents; subsequent passes are conditioned
                 on (lightly re-noised) previous latents for iterative refinement
"""

import gc
import logging
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Callable

import numpy as np

from .config import QuantConfig, StackConfig

logger = logging.getLogger(__name__)


def _free_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class QuantStackEngine:
    """
    Multi-pass 4-bit quantization stacking engine.

    Usage:
        engine = QuantStackEngine(stack_config)
        # Pass a pipeline factory (called once per pass to avoid OOM)
        latents = engine.run_stacked(pipeline_factory, prompt, **generate_kwargs)
    """

    def __init__(self, stack_config: Optional[StackConfig] = None):
        self.config = stack_config or StackConfig()
        self._pass_latents: List[torch.Tensor] = []
        self._pass_times: List[float] = []

    def run_stacked(
        self,
        pipeline_factory: Callable[[QuantConfig], Any],
        prompt: str,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = 42,
        **extra_kwargs,
    ) -> Dict[str, Any]:
        """
        Run the full multi-pass stacking pipeline.

        Args:
            pipeline_factory: Callable(QuantConfig) -> diffusers pipeline
                              Called once per pass; pipeline is unloaded after each pass
            prompt: Text prompt
            negative_prompt: Negative text prompt
            height, width: Frame dimensions
            num_frames: Number of video frames
            num_inference_steps: Denoising steps per pass
            guidance_scale: CFG scale
            seed: Random seed for reproducibility

        Returns:
            dict with keys: "frames", "pass_latents", "pass_times", "strategy"
        """
        strategy = self.config.stacking_strategy
        logger.info(f"Starting {self.config.num_passes}-pass quant stack (strategy={strategy})")

        if strategy == "average":
            return self._run_average(
                pipeline_factory, prompt, negative_prompt,
                height, width, num_frames, num_inference_steps, guidance_scale, seed, **extra_kwargs
            )
        elif strategy == "weighted":
            return self._run_weighted(
                pipeline_factory, prompt, negative_prompt,
                height, width, num_frames, num_inference_steps, guidance_scale, seed, **extra_kwargs
            )
        elif strategy == "residual":
            return self._run_residual(
                pipeline_factory, prompt, negative_prompt,
                height, width, num_frames, num_inference_steps, guidance_scale, seed, **extra_kwargs
            )
        elif strategy == "progressive":
            return self._run_progressive(
                pipeline_factory, prompt, negative_prompt,
                height, width, num_frames, num_inference_steps, guidance_scale, seed, **extra_kwargs
            )
        else:
            raise ValueError(f"Unknown stacking strategy: {strategy}")

    def _make_generator(self, seed: Optional[int], device: str = "cuda"):
        import torch
        if seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return gen

    def _run_average(self, pipeline_factory, prompt, negative_prompt,
                     height, width, num_frames, num_inference_steps, guidance_scale, seed, **kw):
        """Run N independent passes, average their output tensors."""
        all_frames = []

        for i, qcfg in enumerate(self.config.pass_configs):
            logger.info(f"  Pass {i+1}/{self.config.num_passes}: independent 4-bit ({qcfg.bnb_4bit_quant_type})")
            t0 = time.time()

            with self._loaded_pipeline(pipeline_factory, qcfg) as pipe:
                device = pipe.device if hasattr(pipe, 'device') else "cuda"
                gen = self._make_generator(seed + i if seed is not None else None, str(device))
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
                    **kw,
                )
                frames_np = output.frames[0]  # (T, H, W, C) float32 [0,1]
                all_frames.append(frames_np.astype(np.float32))

            elapsed = time.time() - t0
            self._pass_times.append(elapsed)
            logger.info(f"  Pass {i+1} done in {elapsed:.1f}s")
            _free_memory()

        stacked = np.mean(all_frames, axis=0)
        stacked = np.clip(stacked, 0.0, 1.0)
        return self._make_result(stacked)

    def _run_weighted(self, pipeline_factory, prompt, negative_prompt,
                      height, width, num_frames, num_inference_steps, guidance_scale, seed, **kw):
        """Run N independent passes, combine with per-pass weights."""
        weights = self.config.pass_weights
        all_frames = []

        for i, qcfg in enumerate(self.config.pass_configs):
            logger.info(f"  Pass {i+1}/{self.config.num_passes}: weighted 4-bit (w={weights[i]:.2f})")
            t0 = time.time()

            with self._loaded_pipeline(pipeline_factory, qcfg) as pipe:
                device = pipe.device if hasattr(pipe, 'device') else "cuda"
                gen = self._make_generator(seed + i if seed is not None else None, str(device))
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
                    **kw,
                )
                frames_np = output.frames[0].astype(np.float32)
                all_frames.append(frames_np * weights[i])

            elapsed = time.time() - t0
            self._pass_times.append(elapsed)
            _free_memory()

        stacked = np.sum(all_frames, axis=0)
        stacked = np.clip(stacked, 0.0, 1.0)
        return self._make_result(stacked)

    def _run_residual(self, pipeline_factory, prompt, negative_prompt,
                      height, width, num_frames, num_inference_steps, guidance_scale, seed, **kw):
        """
        Residual stacking: pass 1 is the base; each subsequent pass models the
        residual between the reference (bf16 single pass) and the current best.
        Since we don't have a true reference, we approximate using higher-step
        inference on pass 1 and refine with shorter steps on subsequent passes.
        """
        all_frames = []

        for i, qcfg in enumerate(self.config.pass_configs):
            # First pass uses full steps; refinement passes use fewer steps
            steps = num_inference_steps if i == 0 else max(10, num_inference_steps // 3)
            logger.info(f"  Pass {i+1}/{self.config.num_passes}: residual 4-bit (steps={steps})")
            t0 = time.time()

            with self._loaded_pipeline(pipeline_factory, qcfg) as pipe:
                device = pipe.device if hasattr(pipe, 'device') else "cuda"
                gen = self._make_generator(seed if i == 0 else seed + i if seed is not None else None, str(device))
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                    output_type="np",
                    **kw,
                )
                frames_np = output.frames[0].astype(np.float32)
                all_frames.append(frames_np)

            elapsed = time.time() - t0
            self._pass_times.append(elapsed)
            _free_memory()

        # Residual combination: base + weighted corrections
        base = all_frames[0]
        result = base.copy()
        for j in range(1, len(all_frames)):
            residual = all_frames[j] - base
            weight = 1.0 / (j + 1)
            result = result + weight * residual

        result = np.clip(result, 0.0, 1.0)
        return self._make_result(result)

    def _run_progressive(self, pipeline_factory, prompt, negative_prompt,
                         height, width, num_frames, num_inference_steps, guidance_scale, seed, **kw):
        """
        Progressive refinement: each pass refines the previous result.
        Pass 1: full denoising from pure noise.
        Pass N (N>1): re-encode output of pass N-1 at low noise level,
                      then denoise the remaining steps.

        This is the most effective strategy for recovering quality:
        each 4-bit pass contributes a partial refinement, and the
        compounding effect approximates higher-precision inference.
        """
        prev_frames = None
        all_pass_frames = []

        for i, qcfg in enumerate(self.config.pass_configs):
            if i == 0:
                logger.info(f"  Pass 1/{self.config.num_passes}: base generation (4-bit {qcfg.bnb_4bit_quant_type})")
            else:
                logger.info(f"  Pass {i+1}/{self.config.num_passes}: progressive refinement "
                            f"(noise_lvl={self.config.inter_pass_noise_level:.2f})")

            t0 = time.time()

            with self._loaded_pipeline(pipeline_factory, qcfg) as pipe:
                device = pipe.device if hasattr(pipe, 'device') else "cuda"
                gen = self._make_generator(seed if i == 0 else seed + i if seed is not None else None, str(device))

                if i == 0 or prev_frames is None:
                    # First pass: standard text-to-video generation
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
                        **kw,
                    )
                else:
                    # Subsequent passes: use video2video-style refinement if available,
                    # otherwise fall back to standard generation with conditioned latents
                    output = self._progressive_refinement_pass(
                        pipe, prev_frames, prompt, negative_prompt,
                        height, width, num_frames, num_inference_steps,
                        guidance_scale, gen, **kw,
                    )

                frames_np = output.frames[0].astype(np.float32)
                all_pass_frames.append(frames_np)
                prev_frames = frames_np

            elapsed = time.time() - t0
            self._pass_times.append(elapsed)
            logger.info(f"  Pass {i+1} done in {elapsed:.1f}s")
            _free_memory()

        # Final result is the last pass (most refined)
        # Optionally blend last few passes for stability
        if len(all_pass_frames) >= 2:
            alpha = 0.8  # weight toward final pass
            result = alpha * all_pass_frames[-1] + (1 - alpha) * all_pass_frames[-2]
        else:
            result = all_pass_frames[-1]

        result = np.clip(result, 0.0, 1.0)
        return self._make_result(result, all_pass_frames=all_pass_frames)

    def _progressive_refinement_pass(
        self,
        pipe,
        prev_frames_np: np.ndarray,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator,
        **kw,
    ):
        """
        Attempt video-to-video refinement using the pipeline's VAE encoder.
        Falls back to standard generation if the pipeline doesn't support it.
        """
        noise_level = self.config.inter_pass_noise_level

        # Try to access VAE for latent-space refinement
        if hasattr(pipe, 'vae') and hasattr(pipe, 'scheduler'):
            try:
                return self._vae_encode_refine(
                    pipe, prev_frames_np, prompt, negative_prompt,
                    height, width, num_frames, num_inference_steps,
                    guidance_scale, generator, noise_level, **kw,
                )
            except Exception as e:
                logger.warning(f"VAE-based refinement failed ({e}), falling back to standard generation")

        # Fallback: standard generation (independent pass)
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np",
            **kw,
        )

    def _vae_encode_refine(
        self,
        pipe,
        prev_frames_np: np.ndarray,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator,
        noise_level: float,
        **kw,
    ):
        """
        Encode previous frames to latent space, add noise, denoise.
        This implements the img2img/vid2vid refinement loop in latent space.
        """
        import torch  # noqa: F811
        device = pipe.device
        dtype = pipe.vae.dtype if hasattr(pipe.vae, 'dtype') else torch.bfloat16

        # Convert numpy frames to tensor: (T, H, W, C) -> (1, C, T, H, W)
        frames_tensor = torch.from_numpy(prev_frames_np).to(device=device, dtype=dtype)
        frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
        frames_tensor = frames_tensor * 2.0 - 1.0  # normalize to [-1, 1]

        # Encode to latent space
        with torch.no_grad():
            # Different Wan VAE interfaces — try common patterns
            if hasattr(pipe.vae, 'encode'):
                vae_output = pipe.vae.encode(frames_tensor)
                if hasattr(vae_output, 'latent_dist'):
                    latents = vae_output.latent_dist.sample()
                elif hasattr(vae_output, 'latents'):
                    latents = vae_output.latents
                else:
                    latents = vae_output
                if hasattr(pipe.vae, 'config') and hasattr(pipe.vae.config, 'scaling_factor'):
                    latents = latents * pipe.vae.config.scaling_factor
            else:
                raise AttributeError("VAE doesn't have encode method")

        # Add a small amount of noise at the specified level
        # This allows the denoiser to refine rather than reproduce exactly
        scheduler = pipe.scheduler
        num_refine_steps = max(5, int(num_inference_steps * noise_level * 3))
        t_start = int((1.0 - noise_level) * scheduler.config.num_train_timesteps)
        noise = torch.randn_like(latents, generator=generator)
        t_tensor = torch.tensor([t_start], device=device)

        if hasattr(scheduler, 'add_noise'):
            noisy_latents = scheduler.add_noise(latents, noise, t_tensor)
        else:
            noisy_latents = latents + noise_level * noise

        # Run denoising with the noisy latents as starting point
        # Note: not all pipelines support latents kwarg — catch and fallback
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_refine_steps,
            guidance_scale=guidance_scale,
            latents=noisy_latents,
            output_type="np",
            **kw,
        )
        return output

    @contextmanager
    def _loaded_pipeline(self, pipeline_factory: Callable, qcfg: QuantConfig):
        """Context manager: load pipeline, yield it, then unload and free memory."""
        pipe = None
        try:
            pipe = pipeline_factory(qcfg)
            yield pipe
        finally:
            if pipe is not None:
                # Move to CPU first to free GPU VRAM before deleting
                try:
                    if hasattr(pipe, 'to'):
                        pipe.to('cpu')
                except Exception:
                    pass
                del pipe
            _free_memory()

    def _make_result(self, frames_np: np.ndarray, all_pass_frames=None) -> Dict[str, Any]:
        return {
            "frames": frames_np,           # (T, H, W, C) float32 [0,1]
            "pass_times": list(self._pass_times),
            "strategy": self.config.stacking_strategy,
            "num_passes": self.config.num_passes,
            "all_pass_frames": all_pass_frames,
            "total_time": sum(self._pass_times),
        }
