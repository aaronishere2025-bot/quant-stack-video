"""
Quant-Stack Video API — Public HTTP interface on port 8400.

Added for external / public access (vs. the internal prototype):
  - API key auth   (X-API-Key header; keys in VIDEO_API_KEYS env var)
  - Rate limiting  (slowapi: 5/min for generation, 2/min for heavy ops, 60/min for reads)
  - Async job queue with single-GPU concurrency (one generation job runs at a time)
  - Queue position reported in task status

Run with:
    uvicorn src.agent.server:app --host 0.0.0.0 --port 8400

Auth configuration:
    export VIDEO_API_KEYS="key1,key2,key3"
    (If unset, the server runs in open mode — localhost-only is enforced by the OS/firewall)
    Localhost callers (e.g. Unity job worker at 127.0.0.1) always bypass the key check.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Billing — credit amounts
# ---------------------------------------------------------------------------

def _billing_enabled() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY"))


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_api_keys() -> set[str]:
    raw = os.environ.get("VIDEO_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(_API_KEY_HEADER),
) -> None:
    """FastAPI dependency — raises 401 when a key is required but missing/invalid."""
    keys = _load_api_keys()
    if not keys:
        return  # open mode: no keys configured, trust the firewall
    # Localhost callers (Unity job worker) bypass key check
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1"):
        return
    if not api_key or api_key not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def _get_caller_key(request: Request) -> Optional[str]:
    """Return the API key from the request, or None for localhost/open-mode callers."""
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1"):
        return None  # localhost bypass — no billing
    key = request.headers.get("X-API-Key", "").strip()
    return key if key else None


# ---------------------------------------------------------------------------
# Rate limiting  (slowapi)
# ---------------------------------------------------------------------------

def _rate_key(request: Request) -> str:
    """Use the API key as the rate-limit bucket; fall back to remote IP."""
    key = request.headers.get("X-API-Key")
    return key if key else get_remote_address(request)


limiter = Limiter(key_func=_rate_key)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create(self, task_id: str) -> Dict:
        task = {
            "task_id": task_id,
            "status": "queued",
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


# ---------------------------------------------------------------------------
# Job queue — single-GPU concurrency
# ---------------------------------------------------------------------------

_gpu_semaphore: asyncio.Semaphore
_job_queue: asyncio.Queue
_worker_task: asyncio.Task


async def _queue_worker() -> None:
    """Background worker: dequeues jobs and runs them one at a time."""
    while True:
        job_fn, task_id = await _job_queue.get()
        async with _gpu_semaphore:
            _registry.update(task_id, status="running", progress="Starting...")
            try:
                await job_fn()
            except Exception:
                logger.exception("Job %s raised unexpectedly in queue worker", task_id)
                _registry.update(task_id, status="error", error="Unexpected error in queue worker")
        _job_queue.task_done()


async def _enqueue(task_id: str, job_fn) -> None:
    """Add a coroutine function (zero-arg) to the GPU job queue."""
    pos = _job_queue.qsize()
    _registry.update(task_id, progress=f"Queue position {pos}" if pos > 0 else "Next in queue")
    await _job_queue.put((job_fn, task_id))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画面变形，畸形，毁容，"
        "形态畸形，手部畸形，腿部畸形，额外的手指，融合的手指，静止画面，动态不足，"
    )
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
        description="'progressive', 'average', 'weighted', 'residual'",
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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _gpu_semaphore, _job_queue, _worker_task
    _gpu_semaphore = asyncio.Semaphore(1)
    _job_queue = asyncio.Queue()
    _worker_task = asyncio.create_task(_queue_worker())
    logger.info("Job queue worker started")
    yield
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Quant-Stack Video API",
        description=(
            "Quantized video generation as a service. "
            "Authenticate with X-API-Key header. "
            "Generation jobs are queued and executed one at a time."
        ),
        version="2.0.0",
        lifespan=_lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # -----------------------------------------------------------------------
    # Read endpoints  (60/min)
    # -----------------------------------------------------------------------

    @app.get("/health")
    @limiter.limit("60/minute")
    def health(request: Request):
        return {
            "status": "ok",
            "model": "Wan 2.1",
            "agent": "quantization-engineer",
            "queue_depth": _job_queue.qsize(),
        }

    @app.get("/tasks")
    @limiter.limit("60/minute")
    def list_tasks(request: Request, _auth=Depends(require_api_key)):
        return _registry.list_all()

    @app.get("/tasks/{task_id}")
    @limiter.limit("60/minute")
    def get_task(request: Request, task_id: str, _auth=Depends(require_api_key)):
        task = _registry.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return task

    @app.get("/stats/vram")
    @limiter.limit("60/minute")
    def vram_stats(request: Request):
        try:
            import torch
            if not torch.cuda.is_available():
                return {"available": False}
            return {
                "available": True,
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3),
                "device_name": torch.cuda.get_device_properties(0).name,
                "queue_depth": _job_queue.qsize(),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    # -----------------------------------------------------------------------
    # Generation endpoints  (5/min — GPU-heavy)
    # -----------------------------------------------------------------------

    @app.post("/generate/single")
    @limiter.limit("5/minute")
    async def generate_single(
        request: Request,
        req: SinglePassRequest,
        _auth=Depends(require_api_key),
    ):
        """Generate a video with a single quantization pass."""
        task_id = str(uuid.uuid4())
        _registry.create(task_id)
        _register_task_key(task_id, _get_caller_key(request))

        async def job():
            await _run_single_gen(task_id, req)

        await _enqueue(task_id, job)
        return {"task_id": task_id, "status": "queued", "queue_position": _job_queue.qsize() - 1}

    @app.post("/generate/stacked")
    @limiter.limit("5/minute")
    async def generate_stacked(
        request: Request,
        req: StackedRequest,
        _auth=Depends(require_api_key),
    ):
        """Generate a video with multi-pass quantization stacking."""
        task_id = str(uuid.uuid4())
        _registry.create(task_id)
        _register_task_key(task_id, _get_caller_key(request))

        async def job():
            await _run_stacked_gen(task_id, req)

        await _enqueue(task_id, job)
        return {"task_id": task_id, "status": "queued", "queue_position": _job_queue.qsize() - 1}

    @app.post("/generate/long")
    @limiter.limit("5/minute")
    async def generate_long(
        request: Request,
        req: LongVideoRequest,
        _auth=Depends(require_api_key),
    ):
        """Generate a long video (up to 3 minutes) with segment stitching."""
        task_id = str(uuid.uuid4())
        _registry.create(task_id)
        _register_task_key(task_id, _get_caller_key(request))

        async def job():
            await _run_long_gen(task_id, req)

        await _enqueue(task_id, job)
        return {"task_id": task_id, "status": "queued", "queue_position": _job_queue.qsize() - 1}

    # -----------------------------------------------------------------------
    # Heavy ops endpoints  (2/min)
    # -----------------------------------------------------------------------

    @app.post("/benchmark")
    @limiter.limit("2/minute")
    async def run_benchmark(
        request: Request,
        req: BenchmarkRequest,
        _auth=Depends(require_api_key),
    ):
        """Run a side-by-side quality benchmark across quantization configs."""
        task_id = str(uuid.uuid4())
        _registry.create(task_id)

        async def job():
            await _run_benchmark(task_id, req)

        await _enqueue(task_id, job)
        return {"task_id": task_id, "status": "queued", "queue_position": _job_queue.qsize() - 1}

    @app.post("/optimize/auto")
    @limiter.limit("2/minute")
    async def auto_optimize(
        request: Request,
        prompt: str,
        quality_target: float = 30.0,
        max_passes: int = 5,
        _auth=Depends(require_api_key),
    ):
        """
        Autonomous quality optimization: iteratively increase stack passes until
        PSNR vs reference reaches quality_target (dB). Returns optimal config.
        """
        task_id = str(uuid.uuid4())
        _registry.create(task_id)

        async def job():
            await _run_auto_optimize(task_id, prompt, quality_target, max_passes)

        await _enqueue(task_id, job)
        return {"task_id": task_id, "status": "queued", "queue_position": _job_queue.qsize() - 1}

    # -----------------------------------------------------------------------
    # Billing endpoints
    # -----------------------------------------------------------------------

    @app.get("/billing/balance")
    @limiter.limit("60/minute")
    def billing_balance(request: Request, _auth=Depends(require_api_key)):
        """Return credit balance for the caller's API key."""
        from ..billing.store import get_balance, CENTS_PER_SECOND
        key = _get_caller_key(request)
        if key is None:
            return {"balance_cents": None, "note": "localhost callers are not billed"}
        balance = get_balance(key)
        return {
            "balance_cents": balance,
            "balance_dollars": round(balance / 100, 2),
            "seconds_remaining": round(balance / CENTS_PER_SECOND, 1),
        }

    @app.get("/billing/usage")
    @limiter.limit("60/minute")
    def billing_usage(request: Request, limit: int = 50, _auth=Depends(require_api_key)):
        """Return recent usage records for the caller's API key."""
        from ..billing.store import get_usage
        key = _get_caller_key(request)
        if key is None:
            return []
        records = get_usage(key, limit=min(limit, 200))
        return [
            {
                **r,
                "cost_dollars": round(r["cost_cents"] / 100, 4),
            }
            for r in records
        ]

    @app.post("/billing/checkout")
    @limiter.limit("10/minute")
    def billing_checkout(request: Request, package_id: str = "standard", _auth=Depends(require_api_key)):
        """
        Create a Stripe Checkout session to purchase credits.
        package_id: 'starter' ($5, 50s), 'standard' ($10, 100s), 'pro' ($25, 250s)
        Returns {"url": "https://checkout.stripe.com/..."}.
        """
        if not _billing_enabled():
            raise HTTPException(status_code=503, detail="Billing not configured (STRIPE_SECRET_KEY not set)")
        from ..billing.stripe_client import create_checkout_session, CREDIT_PACKAGES
        key = _get_caller_key(request)
        if key is None:
            raise HTTPException(status_code=400, detail="Cannot create checkout session for localhost callers")
        base_url = os.environ.get("BILLING_BASE_URL", f"http://{request.headers.get('host', 'localhost:8400')}")
        try:
            result = create_checkout_session(package_id, key, base_url=base_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {**result, "package": CREDIT_PACKAGES[package_id]}

    @app.post("/billing/webhook")
    async def billing_webhook(request: Request):
        """Stripe webhook — grants credits on successful payment. No auth required."""
        if not _billing_enabled():
            raise HTTPException(status_code=503, detail="Billing not configured")
        from ..billing.stripe_client import handle_webhook
        from ..billing.store import add_credits
        payload = await request.body()
        sig = request.headers.get("stripe-signature", "")
        try:
            action = handle_webhook(payload, sig)
        except Exception as e:
            logger.warning("Stripe webhook error: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        if action and action["action"] == "grant_credits":
            new_balance = add_credits(action["api_key"], action["credits_cents"])
            logger.info(
                "Granted %d cents to %s…  new balance=%d",
                action["credits_cents"], action["api_key"][:8], new_balance,
            )
        return {"received": True}

    @app.get("/billing/packages")
    @limiter.limit("60/minute")
    def billing_packages(request: Request):
        """List available credit packages."""
        from ..billing.stripe_client import CREDIT_PACKAGES
        return [
            {
                "id": pkg_id,
                "name": pkg["name"],
                "price_dollars": round(pkg["amount_cents"] / 100, 2),
                "video_seconds": pkg["credits_cents"] // 10,
                "description": pkg["description"],
            }
            for pkg_id, pkg in CREDIT_PACKAGES.items()
        ]

    @app.get("/dashboard", response_class=None)
    @limiter.limit("60/minute")
    def dashboard(request: Request, _auth=Depends(require_api_key)):
        """Simple HTML dashboard showing credit balance and usage history."""
        from fastapi.responses import HTMLResponse
        from ..billing.store import get_balance, get_usage, CENTS_PER_SECOND

        key = _get_caller_key(request)
        purchase_msg = request.query_params.get("purchase", "")

        if key:
            balance = get_balance(key)
            usage = get_usage(key, limit=20)
        else:
            balance = None
            usage = []

        balance_html = (
            f"<p><strong>Balance:</strong> ${balance/100:.2f} ({balance/CENTS_PER_SECOND:.1f}s of video)</p>"
            if balance is not None
            else "<p>Localhost caller — billing not applicable.</p>"
        )

        purchase_banner = ""
        if purchase_msg == "success":
            purchase_banner = '<div class="banner success">Payment successful! Credits added to your account.</div>'
        elif purchase_msg == "cancelled":
            purchase_banner = '<div class="banner warn">Purchase cancelled.</div>'

        packages_html = ""
        if _billing_enabled():
            packages_html = """
            <h2>Buy Credits</h2>
            <div class="packages">
              <form method="post" action="/billing/checkout?package_id=starter">
                <button type="submit">Starter — $5 (50 s)</button>
              </form>
              <form method="post" action="/billing/checkout?package_id=standard">
                <button type="submit">Standard — $10 (100 s)</button>
              </form>
              <form method="post" action="/billing/checkout?package_id=pro">
                <button type="submit">Pro — $25 (250 s)</button>
              </form>
            </div>"""

        rows_html = ""
        if usage:
            rows_html = "".join(
                f"<tr><td>{r['task_id'] or '—'}</td><td>{r['seconds']:.1f}s</td>"
                f"<td>${r['cost_cents']/100:.4f}</td></tr>"
                for r in usage
            )
        else:
            rows_html = "<tr><td colspan='3'>No usage yet.</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Quant-Stack Dashboard</title>
<style>
  body {{ font-family: monospace; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #0d0d0d; color: #e0e0e0; }}
  h1 {{ color: #7cf; }} h2 {{ color: #9af; margin-top: 2em; }}
  .banner {{ padding: 10px 16px; border-radius: 4px; margin: 1em 0; }}
  .success {{ background: #1a3a1a; color: #7f7; border: 1px solid #4a4; }}
  .warn {{ background: #3a2a0a; color: #fb7; border: 1px solid #a84; }}
  .packages {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 1em 0; }}
  .packages button {{ background: #1a2a3a; color: #7cf; border: 1px solid #37f; padding: 10px 20px; cursor: pointer; border-radius: 4px; font-size: 1em; }}
  .packages button:hover {{ background: #1e3a5a; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
  th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
  th {{ background: #1a1a2a; color: #9af; }}
</style>
</head><body>
<h1>Quant-Stack Video</h1>
{purchase_banner}
<h2>Credit Balance</h2>
{balance_html}
{packages_html}
<h2>Recent Usage</h2>
<table>
  <thead><tr><th>Task ID</th><th>Duration</th><th>Cost</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</body></html>"""
        return HTMLResponse(content=html)

    return app


# ---------------------------------------------------------------------------
# Billing helpers  (used inside job runners)
# ---------------------------------------------------------------------------

# task_id -> api_key map so background runners can bill after queuing
_task_api_keys: Dict[str, str] = {}


def _register_task_key(task_id: str, api_key: Optional[str]) -> None:
    if api_key:
        _task_api_keys[task_id] = api_key


def _deduct_for_task(task_id: str, num_frames: int, fps: int) -> None:
    """Deduct credits after a generation job completes."""
    if not _billing_enabled():
        return
    key = _task_api_keys.pop(task_id, None)
    if not key:
        return
    seconds = num_frames / max(fps, 1)
    from ..billing.store import deduct_credits
    result = deduct_credits(key, seconds, task_id=task_id)
    if not result["ok"]:
        logger.warning("Billing deduction failed for task %s: %s", task_id, result)
    else:
        logger.info(
            "Billed %.1fs ($%.4f) to key %s… task=%s",
            seconds, result["cost_cents"] / 100, key[:8], task_id,
        )


# ---------------------------------------------------------------------------
# Background job runners  (called by queue worker — GPU semaphore already held)
# ---------------------------------------------------------------------------

async def _run_single_gen(task_id: str, req: SinglePassRequest):
    _registry.update(task_id, progress="Loading pipeline...")
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
        _deduct_for_task(task_id, req.num_frames, req.fps)
    except Exception as e:
        logger.exception("Task %s failed", task_id)
        _registry.update(task_id, status="error", error=str(e))


async def _run_stacked_gen(task_id: str, req: StackedRequest):
    _registry.update(task_id, progress="Initializing stacked generation...")
    try:
        from ..wan.generate import generate_video_stacked

        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{task_id[:8]}_{req.num_passes}x_{req.stacking_strategy}.mp4")

        for i in range(req.num_passes):
            _registry.update(task_id, progress=f"Pass {i + 1}/{req.num_passes}...")

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
        _deduct_for_task(task_id, req.num_frames, req.fps)
    except Exception as e:
        logger.exception("Task %s failed", task_id)
        _registry.update(task_id, status="error", error=str(e))


async def _run_long_gen(task_id: str, req: LongVideoRequest):
    _registry.update(task_id, progress="Starting long video generation...")
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
        total_frames = int(req.duration_seconds * req.fps)
        _deduct_for_task(task_id, total_frames, req.fps)
    except Exception as e:
        logger.exception("Task %s failed", task_id)
        _registry.update(task_id, status="error", error=str(e))


async def _run_benchmark(task_id: str, req: BenchmarkRequest):
    _registry.update(task_id, progress="Starting benchmark...")
    try:
        from ..benchmark.runner import BenchmarkRunner, BenchmarkConfig

        stack_configs = [
            {"num_passes": np, "strategy": s}
            for np in req.stack_passes
            for s in req.stack_strategies
        ]

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
        logger.exception("Benchmark task %s failed", task_id)
        _registry.update(task_id, status="error", error=str(e))


async def _run_auto_optimize(
    task_id: str,
    prompt: str,
    quality_target: float,
    max_passes: int,
):
    """Iteratively increase stack passes until target PSNR is reached."""
    _registry.update(task_id, progress="Generating reference...")
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

        import torch
        import tempfile
        ref_cfg = QuantConfig(quant_type="none", load_in_4bit=False, load_in_8bit=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe = factory(ref_cfg)
            gen = torch.Generator(device="cuda").manual_seed(42)
            out = pipe(
                prompt=prompt,
                num_frames=49,
                num_inference_steps=20,
                guidance_scale=5.0,
                generator=gen,
                output_type="np",
            )
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
        logger.exception("Auto-optimize task %s failed", task_id)
        _registry.update(task_id, status="error", error=str(e))


# ---------------------------------------------------------------------------
# Singleton app instance
# ---------------------------------------------------------------------------

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
