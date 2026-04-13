#!/usr/bin/env python3
"""
First end-to-end LTX run after optimization pass.

Demonstrates the full loop:
  1. Thompson sampling bandit picks enhancement arms
  2. LTX generates a clip using the cached pipeline
  3. LLaVA (via Ollama) scores the clip on 4 dimensions
  4. Bandit reward is updated with the real score

Env overrides (baked in here for this first run):
  LLM_EVAL_URL  = Ollama's OpenAI-compatible endpoint
  LLM_EVAL_MODEL = llava:13b (already pulled)

Usage:
    .venv/bin/python scripts/run_first_ltx.py "a majestic eagle soaring over a mountain valley at sunset"
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import time
from pathlib import Path

# Point the video evaluator at Ollama BEFORE importing video_quality
os.environ.setdefault("LLM_EVAL_URL", "http://localhost:11434/v1/chat/completions")
os.environ.setdefault("LLM_EVAL_MODEL", "llava:13b")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.prompt_bandit import prompt_bandit  # noqa: E402
from src.wan.generate import generate_video  # noqa: E402
from src.wan.ltx_pipeline_factory import clear_pipeline_cache  # noqa: E402
from src.agent.video_quality import evaluate  # noqa: E402


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else (
        "a majestic eagle soaring over a mountain valley at sunset"
    )

    print("=" * 78)
    print("LTX FIRST RUN — full bandit → generate → evaluate → reward loop")
    print("=" * 78)
    print(f"Base prompt: {base}")
    print()

    # --- Step 1: Bandit picks enhancement arms -----------------------------
    print("[1/4] Thompson sampling: picking enhancement arms...")
    enhanced, arm_ids = prompt_bandit.build_enhanced_prompt(base, engine="ltx")
    print(f"  Enhanced prompt: {enhanced}")
    print(f"  Arm IDs: {arm_ids}")
    print()

    # --- Step 2: LTX generation --------------------------------------------
    output_dir = ROOT / "outputs" / "ltx_first_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_path = str(output_dir / f"clip_{timestamp}.mp4")

    print("[2/4] LTX generation (this is the first call — pipeline loads now)...")
    t_gen_start = time.time()
    saved = generate_video(
        prompt=enhanced,
        output_path=output_path,
        engine="ltx",
        model_id="Lightricks/LTX-Video",
        # Use ext4 HF cache — NTFS via 9P is ~10-50x slower for large shard reads
        cache_dir="/home/aaron/.cache/huggingface",
        height=512,
        width=768,
        num_frames=121,  # ~4.84s @ 25fps
        num_inference_steps=50,
        guidance_scale=3.0,
        seed=timestamp % (2**31),
        fps=25,
    )
    gen_elapsed = time.time() - t_gen_start
    clip_size_mb = Path(saved).stat().st_size / 1024 / 1024
    print(f"  Generated in {gen_elapsed:.1f}s → {saved} ({clip_size_mb:.1f} MB)")
    print()

    # --- Step 3: Free pipeline, then VLM eval -------------------------------
    print("[3/4] Releasing LTX pipeline and evaluating with LLaVA...")
    clear_pipeline_cache()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    t_eval_start = time.time()
    result = asyncio.run(evaluate(prompt=enhanced, video_path=saved))
    eval_elapsed = time.time() - t_eval_start
    print(f"  Eval elapsed: {eval_elapsed:.1f}s")
    print()

    if not result.get("success"):
        print(f"  VLM eval FAILED: {result.get('error', 'unknown')}")
        return 1

    # --- Step 4: Print scoring and update bandit ---------------------------
    print("[4/4] Scoring breakdown:")
    print(f"  overall score       : {result['score']:.1f} / 10")
    print(f"  prompt match        : {result.get('prompt_match', 0):.1f} / 10")
    print(f"  motion quality      : {result.get('motion_quality', 0):.1f} / 10")
    print(f"  visual coherence    : {result.get('visual_coherence', 0):.1f} / 10")
    print(f"  composition         : {result.get('composition', 0):.1f} / 10")
    print(f"  best domain         : {result.get('best_domain', 'n/a')}  "
          f"(domain_score={result.get('domain_score', 0):.1f})")
    flags = result.get("flags", [])
    if flags:
        print(f"  flags               : {flags}")
    directives = result.get("next_prompt_directives", [])
    if directives:
        print(f"  next directives     : {directives}")
    feedback = result.get("feedback", "")
    if feedback:
        print(f"  feedback            : {feedback}")
    print()

    print("  Updating bandit with reward...")
    prompt_bandit.update_reward(arm_ids, float(result["score"]), engine="ltx")
    print("  Bandit persisted to data/bandit/ltx-prompt-bandit.json")
    print()

    print("=" * 78)
    print(f"DONE — gen {gen_elapsed:.0f}s, eval {eval_elapsed:.0f}s, "
          f"score {result['score']:.1f}/10")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
