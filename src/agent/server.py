"""
Autonomous Quantization Optimization Agent — REST API on port 8400.

Provides endpoints for:
- Video generation (single-pass and stacked)
- Live benchmark runs
- Adaptive config optimization based on feedback
- Status monitoring

Run with:
    python -m src.agent.server
or:
    uvicorn src.agent.server:app --host 0.0.0.0 --port 8400
"""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Request/Response Models ---

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画面变形，畸形，毁容，形态畸形，手部畸形，腿部畸形，额外的手指，融合的手指，静止画面，动态不足，"
    height: int = 480
    width: int = 832
    num_frames: int = 81
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 42
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    cache_dir: Optional[str] = None
    fps: int = 16
    output_dir: str = "outputs"


class SinglePassRequest(GenerateRequest):
    quant_type: str = Field(default="4bit", description="'4bit', '8bit', or 'none'")


class StackedRequest(GenerateRequest):
    num_passes: int = Field(default=3, ge=1, le=8)
    stacking_strategy: str = Field(
        default="progressive",
        description="'progressive', 'average', 'weighted', 'residual'"
    )


class LongVideoRequest(GenerateRequest):
    duration_seconds: float = Field(default=30.0, le=180.0)  # max 3 minutes
    segment_frames: int = 81
    overlap_frames: int = 8
    use_stacking: bool = True
    num_passes: int = 3
    stacking_strategy: str = "progressive"


class BenchmarkRequest(BaseModel):
    prompts: List[str] = Field(default_factory=lambda: [
        "A serene mountain lake at sunrise, mist rising from the water"
    ])
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    cache_dir: Optional[str] = None
    height: int = 480
    width: int = 832
    num_frames: int = 49  # shorter for benchmark speed
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    seed: int = 42
    output_dir: str = "benchmark_outputs"
    run_reference: bool = True
    run_4bit_single: bool = True
    run_8bit_single: bool = True
    stack_passes: List[int] = Field(default_factory=lambda: [2, 3])
    stack_strategies: List[str] = Field(default_factory=lambda: ["average", "progressive"])


class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "running", "done", "error"
    progress: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float


# --- Task Registry ---

class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create(self, task_id: str) -> Dict:
        task = {
            "task_id": task_id,
            "status": "pending",
            "progress": None,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._tasks[task_id] = task
        return task

    def update(self, task_id: str, **kwargs):
        if task_id in self._tasks:
            self._tasks[task_id].update(kwargs)
            self._tasks[task_id]["updated_at"] = time.time()

    def get(self, task_id: str) -> Optional[Dict]:
        return self._tasks.get(task_id)

    def list_all(self) -> List[Dict]:
        return list(self._tasks.values())


_registry = TaskRegistry()


# --- App ---

def create_app() -> FastAPI:
    app = FastAPI(
        title="Quant-Stack Video Agent",
        description="Autonomous quantization optimization agent for Wan 2.1 video generation",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": "Wan 2.1",
            "agent": "quantization-engineer",
        }

    @app.get("/tasks")
    def list_tasks():
        return _registry.list_all()

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str):
        task = _registry.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return task

    @app.post("/generate/single")
    async def generate_single(req: SinglePassRequest, background_tasks: BackgroundTasks):
        """Generate a video with a single quantization pass."""
        task_id = str(uuid.uuid4())
        _registry.create(task_id)
        background_tasks.add_task(_run_single_gen, task_id, req)
        return {"task_id": task_id, "status": "pending"}

    @app.post("/generate/stacked")
    async def generate_stacked(req: StackedRequest, background_tasks: BackgroundTasks):
        """Generate a video with multi-pass quantization stacking."""
        task_id = str(uuid.uuid4())
        _registry.create(task_id)
        background_tasks.add_task(_run_stacked_gen, task_id, req)
        return {"task_id": task_id, "status": "pending"}

    @app.post("/generate/long")
    async def generate_long(req: LongVideoRequest, background_tasks: BackgroundTasks):
        """Generate a long video (up to 3 minutes) with segment stitching."""
        task_id = str(uuid.uuid4())
        _registry.create(task_id)
        background_tasks.add_task(_run_long_gen, task_id, req)
        return {"task_id": task_id, "status": "pending"}

    @app.post("/benchmark")
    async def run_benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks):
        """Run a side-by-side quality benchmark across quantization configs."""
        task_id = str(uuid.uuid4())
        _registry.create(task_id)
        background_tasks.add_task(_run_benchmark, task_id, req)
        return {"task_id": task_id, "status": "pending"}

    @app.post("/optimize/auto")
    async def auto_optimize(
        prompt: str,
        quality_target: float = 30.0,
        max_passes: int = 5,
        background_tasks: BackgroundTasks = None,
    ):
        """
        Autonomous quality optimization: iteratively increase stack passes until
        PSNR vs reference reaches quality_target (dB). Returns optimal config.
        """
        task_id = str(uuid.uuid4())
        _registry.create(task_id)
        background_tasks.add_task(_run_auto_optimize, task_id, prompt, quality_target, max_passes)
        return {"task_id": task_id, "status": "pending"}

    @app.get("/stats/vram")
    def vram_stats():
        """Current VRAM usage statistics."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"available": False}
            return {
                "available": True,
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "device_name": torch.cuda.get_device_properties(0).name,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    return app


# --- Background Task Runners ---

async def _run_single_gen(task_id: str, req: SinglePassRequest):
    _registry.update(task_id, status="running", progress="Loading pipeline...")
    try:
        from ..wan.generate import generate_video

        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{task_id[:8]}_{req.quant_type}.mp4")

        _registry.update(task_id, progress=f"Generating ({req.quant_type})...")
        saved = generate_video(
            prompt=req.prompt,
            output_path=output_path,
            model_id=req.model_id,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            quant_type=req.quant_type,
            cache_dir=req.cache_dir,
            fps=req.fps,
        )
        _registry.update(task_id, status="done", result={"output_path": saved})
    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        _registry.update(task_id, status="error", error=str(e))


async def _run_stacked_gen(task_id: str, req: StackedRequest):
    _registry.update(task_id, status="running", progress="Initializing stacked generation...")
    try:
        from ..wan.generate import generate_video_stacked

        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{task_id[:8]}_{req.num_passes}x_{req.stacking_strategy}.mp4")

        for i in range(req.num_passes):
            _registry.update(task_id, progress=f"Pass {i+1}/{req.num_passes}...")

        result = generate_video_stacked(
            prompt=req.prompt,
            output_path=output_path,
            model_id=req.model_id,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            num_passes=req.num_passes,
            stacking_strategy=req.stacking_strategy,
            cache_dir=req.cache_dir,
            fps=req.fps,
        )
        _registry.update(task_id, status="done", result=result)
    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        _registry.update(task_id, status="error", error=str(e))


async def _run_long_gen(task_id: str, req: LongVideoRequest):
    _registry.update(task_id, status="running", progress="Starting long video generation...")
    try:
        from ..wan.generate import generate_long_video

        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{task_id[:8]}_long_{int(req.duration_seconds)}s.mp4")

        result = generate_long_video(
            prompt=req.prompt,
            output_path=output_path,
            duration_seconds=req.duration_seconds,
            model_id=req.model_id,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            fps=req.fps,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            segment_frames=req.segment_frames,
            overlap_frames=req.overlap_frames,
            use_stacking=req.use_stacking,
            num_passes=req.num_passes,
            stacking_strategy=req.stacking_strategy,
            cache_dir=req.cache_dir,
        )
        _registry.update(task_id, status="done", result=result)
    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        _registry.update(task_id, status="error", error=str(e))


async def _run_benchmark(task_id: str, req: BenchmarkRequest):
    _registry.update(task_id, status="running", progress="Starting benchmark...")
    try:
        from ..benchmark.runner import BenchmarkRunner, BenchmarkConfig

        # Build stack config list from request parameters
        stack_configs = []
        for num_p in req.stack_passes:
            for strategy in req.stack_strategies:
                stack_configs.append({"num_passes": num_p, "strategy": strategy})

        cfg = BenchmarkConfig(
            prompts=req.prompts,
            model_id=req.model_id,
            cache_dir=req.cache_dir,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            output_dir=req.output_dir,
            run_reference=req.run_reference,
            run_4bit_single=req.run_4bit_single,
            run_8bit_single=req.run_8bit_single,
            stack_configs=stack_configs,
        )

        runner = BenchmarkRunner(cfg)
        results = runner.run()

        report_path = str(Path(req.output_dir) / f"benchmark_{task_id[:8]}.json")
        runner.save_report(results, report_path)

        # Summarize for task result
        summary = []
        for r in results:
            entry = {
                "config": r.config_label,
                "prompt_idx": r.prompt_idx,
                "generation_time": round(r.generation_time, 2),
                "error": r.error,
            }
            if r.metrics:
                entry["psnr"] = round(r.metrics.get("psnr", 0), 2)
                entry["ssim"] = round(r.metrics.get("ssim", 0), 4)
                entry["lpips"] = r.metrics.get("lpips")
            summary.append(entry)

        _registry.update(task_id, status="done", result={
            "report_path": report_path,
            "summary": summary,
        })
    except Exception as e:
        logger.exception(f"Benchmark task {task_id} failed")
        _registry.update(task_id, status="error", error=str(e))


async def _run_auto_optimize(
    task_id: str,
    prompt: str,
    quality_target: float,
    max_passes: int,
):
    """
    Autonomous optimization loop: binary search for minimum stack passes
    that achieve quality_target PSNR vs reference.
    """
    _registry.update(task_id, status="running", progress="Generating reference...")
    try:
        import numpy as np
        from ..wan.generate import generate_video
        from ..quant.config import QuantConfig, StackConfig
        from ..quant.engine import QuantStackEngine
        from ..wan.pipeline_factory import WanPipelineFactory
        from ..benchmark.metrics import VideoQualityMetrics

        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        factory = WanPipelineFactory(model_id=model_id)
        metrics_engine = VideoQualityMetrics(use_lpips=False)

        # Generate reference
        import torch
        ref_cfg = QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False)
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref.mp4")
            pipe = factory(ref_cfg)
            gen = torch.Generator(device="cuda").manual_seed(42)
            out = pipe(prompt=prompt, num_frames=49, num_inference_steps=20,
                       guidance_scale=5.0, generator=gen, output_type="np")
            ref_frames = out.frames[0].astype(np.float32)
            del pipe

        results_log = []
        best_config = None
        best_psnr = 0.0

        for n_passes in range(1, max_passes + 1):
            _registry.update(task_id, progress=f"Testing {n_passes} passes...")

            engine = QuantStackEngine(StackConfig(
                num_passes=n_passes,
                stacking_strategy="progressive",
            ))
            result = engine.run_stacked(
                pipeline_factory=factory,
                prompt=prompt,
                num_frames=49,
                num_inference_steps=20,
                guidance_scale=5.0,
                seed=42,
                height=480,
                width=832,
            )
            frames = result["frames"]
            m = metrics_engine.compute_all(ref_frames, frames, label=f"{n_passes}x4bit")
            psnr = m["psnr"]

            results_log.append({
                "num_passes": n_passes,
                "psnr": round(psnr, 2),
                "ssim": round(m["ssim"], 4),
                "total_time": round(result["total_time"], 1),
            })

            if psnr > best_psnr:
                best_psnr = psnr
                best_config = {"num_passes": n_passes, "strategy": "progressive", "psnr": psnr}

            if psnr >= quality_target:
                _registry.update(task_id, status="done", result={
                    "optimal_config": best_config,
                    "quality_achieved": psnr,
                    "target_psnr": quality_target,
                    "passes_tested": results_log,
                    "converged": True,
                })
                return

        _registry.update(task_id, status="done", result={
            "optimal_config": best_config,
            "quality_achieved": best_psnr,
            "target_psnr": quality_target,
            "passes_tested": results_log,
            "converged": False,
            "note": f"Max passes ({max_passes}) reached without hitting target PSNR {quality_target}",
        })

    except Exception as e:
        logger.exception(f"Auto-optimize task {task_id} failed")
        _registry.update(task_id, status="error", error=str(e))


# Singleton app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "src.agent.server:app",
        host="0.0.0.0",
        port=8400,
        reload=False,
        log_level="info",
    )
