# CLAUDE.md — Quant-Stack Video

Authoritative conventions for the quant-stack-video project. Read this every heartbeat.

## Project

Infinite layered video generation engine. Core thesis: 3 quantized RGBA layers + VACE temporal extension + SVI error recycling = infinite cinematic video on 12 GB VRAM (RTX 4070).

## Hardware Target

- **GPU**: RTX 4070, 12 GB VRAM
- All VRAM budgets assume 12 GB. The GPU is shared — check `nvidia-smi` before long runs.
- Environment variable for long runs: `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256` (note: `_CUDA_` not `_HIP_` — HIP is the AMD ROCm allocator; RTX 4070 is NVIDIA, so the `_HIP_` form is silently a no-op)

## Python Environment

Always activate the venv before running anything:
```bash
source .venv/bin/activate
```

Python 3.12+. Install deps: `pip install -r requirements.txt`.

## Architecture (5-Phase Pipeline)

### Phase 1 — 3× RGBA Generation
- Wan-Alpha generates 3 transparent video layers: background, midground, foreground
- Each layer: RGBA (4-channel) using Q4_0/Q3_K_M GGUF or BnB 4-bit quantization
- Quantize only the DiT transformer — VAE must stay float32/bfloat16, never quantize it

### Phase 2 — Alpha Compositing
```
Output = (Top_RGB × Top_Alpha) + (Bottom_RGB × (1 − Top_Alpha))
```
Compositing order: background → midground → foreground.

### Phase 3 — VACE Temporal Extension
- 16-frame bidirectional overlap: frames 65–81 of the previous segment
- Key params: **Shift=1**, **CFG=2.0–3.0**
- Pad unknown frames with `#7F7F7F` grey
- Mask convention: known=black, unknown=white
- Keep overlap data in latent space — never decode/re-encode for handoffs (causes color shift)

### Phase 4 — SVI Error Recycling
- Injects historical DiT errors back into flow matching
- Prevents autoregressive drift over infinite iterations
- SVI-Shot variant for single-scene continuous generation

### Phase 5 — LLM Continuity Agent + Bandit + EchoShot
- `LLMDirector` generates evolving prompts per segment via `src/llm/director.py`
- `WanPromptBandit` (Thompson sampling) selects prompt modifiers per-engine — state persists to `data/bandit/*.json`
- EchoShot: last frame of each segment saved as `_last_frame.png` and fed as `image_path` to `LTXImageToVideoPipeline` for i2v continuity
- Supports two engines: `wan` (default) and `ltx` — auto-detected from `model_id` in `/generate/infinite`

## Critical Constraints

| Rule | Detail |
|------|--------|
| VAE stays float32 | Never quantize the VAE — quality degrades severely |
| num_frames = 4k+1 | Must be 81 (5s @ 16fps), 49, 33, etc. |
| T5 on CPU | Text encoder must be CPU-offloaded to save VRAM |
| Tiled VAE decode | 10-frame temporal overlap — required to prevent OOM |
| GC after VAE decode | `gc.collect(); torch.cuda.empty_cache()` after every decode |
| Tensor shapes | RGBA: `[B, 4, F, H, W]` — RGB: `[B, 3, F, H, W]` — validate before stacking |

## Quantization Stacking

Multi-pass approach in `src/quant/`:
- **progressive** (default, best quality): each pass refines via VAE encode → add noise → denoise
- **average**: N independent passes, mean output
- **weighted**: N passes, weighted sum
- **residual**: base + residual corrections

Default config: 3 passes, NF4/FP4/NF4 variation, `inter_pass_noise_level=0.05`.

## Performance Optimizations

- **TeaCache** + **SageAttention2**: ~2× inference speedup
- `enable_model_cpu_offload()` for memory-constrained runs
- `enable_vae_slicing()` always on
- `enable_vae_tiling()` only for very high-res frames

## File Structure

```
src/
  quant/          # Quantization stacking engine
    config.py     # QuantConfig, StackConfig
    engine.py     # QuantStackEngine (4 strategies)
  wan/            # Wan 2.1 + LTX-Video integration
    __init__.py          # intentionally empty (lazy imports — avoids torch init on NTFS)
    pipeline_factory.py  # WanPipelineFactory (WAN 2.1)
    ltx_pipeline_factory.py  # LTXPipelineFactory (t2v + i2v EchoShot)
    generate.py          # generate_video (engine=wan|ltx, image_path for i2v)
  llm/            # LLM director + prompt bandit
    director.py          # LLMDirector — narrative state, next_segment(), record_segment_quality()
    prompt_bandit.py     # WanPromptBandit — Thompson sampling, per-engine arms, JSON persistence
  benchmark/      # Quality metrics and comparison runner
    metrics.py    # PSNR, SSIM, LPIPS, temporal consistency
    runner.py     # BenchmarkRunner
  agent/          # Autonomous optimization agent
    server.py          # FastAPI on :8400
    video_quality.py   # Clip evaluator — Gemini 2.5 Flash (full MP4 via File API); requires GEMINI_API_KEY in workspace .env
configs/
  default.yaml    # Default generation and stacking config
data/
  bandit/         # Bandit state JSON (wan-prompt-bandit.json, ltx-prompt-bandit.json)
scripts/
  generate.py     # CLI entry point
  start_agent.sh  # Start :8400 agent
docs/
  research-gemini-layered-infinite-video.md
tests/            # 26 tests, all no-GPU (monkeypatched)
  test_quant_config.py
  test_prompt_bandit.py
  test_ltx_pipeline_factory.py
  test_director_bandit.py
  test_generate_ltx.py
  test_infinite_chaining.py
  test_agent_server.py
  test_billing.py
  test_v2_components.py
```

## Running Things

```bash
# Single 4-bit pass
python scripts/generate.py --prompt "..." --quant 4bit

# 3-pass progressive stack
python scripts/generate.py --prompt "..." --stacked --passes 3 --strategy progressive

# Long video (30s)
python scripts/generate.py --prompt "..." --long --duration 30

# Side-by-side benchmark
python scripts/generate.py --prompt "..." --benchmark

# Start optimization agent
bash scripts/start_agent.sh
# OR:
uvicorn src.agent.server:app --host 0.0.0.0 --port 8400
```

## Agent API (:8400)

Unity queries this to decide local vs Kling generation.

> **Port note**: `:8400` is also claimed by `workspace-stackeroo-api` in the workspace `COMPOSE_FILE` chain. If you start both via `./workspace.sh up` while this agent is also running directly, the second binder will fail. If you don't run stackeroo locally, ignore.
>
> **Port mismatch with Unity caller**: `projects/unity-repo/server/services/local-video-service.ts` hardcodes `WAN_API_URL = http://localhost:8190` (not `:8400`). Either Unity is silently calling nothing, or there's a separate service on `:8190` not documented here. Worth investigating before relying on the local video pipeline from Unity.

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Status check (returns `engine: wan\|ltx`) |
| `POST /generate/single` | Single-pass generation |
| `POST /generate/stacked` | Multi-pass stacked |
| `POST /generate/long` | Long video (≤3 min) |
| `POST /generate/infinite` | Infinite video loop: bandit prompts + EchoShot frame chaining |
| `POST /benchmark` | Run quality comparison |
| `POST /optimize/auto` | Auto-optimize pass count |
| `GET /stats/vram` | Live VRAM stats |

**If you change the API contract, notify Unity Engineer (agent: 986a9984).**

### Infinite Gen Notes

`/generate/infinite` threads `prev_frame_path` through the segment loop for EchoShot.
Engine (`wan` vs `ltx`) is auto-detected from `model_id` (LTX if `"ltx"` or `"lightricks"` in name).
Bandit reward is recorded after each segment via `director.record_segment_quality()`.

## Benchmarking

Run after changing `src/quant/engine.py`, `src/wan/pipeline_factory.py`, or compositing logic:
```bash
python -m src.benchmark.runner  # or use /benchmark endpoint
```

Quality targets (vs bf16 reference):
- Single 4-bit: ~26–28 dB PSNR
- 3-pass progressive: ~30–32 dB PSNR (target: match 8-bit at ~31 dB)
- SSIM: 0.95+ for stacked passes

## Commit Style

```
feat: add progressive refinement to QuantStackEngine
fix: prevent VAE OOM on 81-frame generation
perf: add TeaCache for 2x speedup on Wan 2.1 14B

Co-Authored-By: Paperclip <noreply@paperclip.ing>
```

Never commit: benchmark output files (`*.mp4`, `*_frames.npy`), model weights, `.venv/`.

## Safety

- Never start a run that would exceed 12 GB VRAM without checking first:
  ```python
  assert torch.cuda.get_device_properties(0).total_memory / 1e9 >= required_gb
  ```
- Never leave GPU processes running unattended without timeout guards
- Cache dir can grow large — evict old segments regularly
