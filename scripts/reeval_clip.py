#!/usr/bin/env python3
"""
Re-run the VLM eval on an already-generated clip using Gemini 2.5 Flash.

Uploads the full MP4 to the Gemini File API for scoring — no GPU required.
Requires GEMINI_API_KEY in the environment or the workspace .env file.

Usage:
    .venv/bin/python scripts/reeval_clip.py <path-to-clip.mp4> [base-prompt]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Load GEMINI_API_KEY from workspace .env BEFORE importing video_quality
# (video_quality reads _GEMINI_API_KEY at module import time)
if not os.environ.get("GEMINI_API_KEY"):
    _env_file = Path(__file__).parent.parent.parent.parent / ".env"  # workspace root
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            if _line.startswith("GEMINI_API_KEY="):
                os.environ["GEMINI_API_KEY"] = _line.split("=", 1)[1].strip().strip('"').strip("'")
                break

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.video_quality import evaluate  # noqa: E402
from src.llm.prompt_bandit import prompt_bandit  # noqa: E402


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: reeval_clip.py <clip.mp4> [enhanced-prompt]")
        return 2

    clip_path = sys.argv[1]
    default_prompt = (
        "a majestic eagle soaring over a mountain valley at sunset, "
        "wellness / meditation, calm therapeutic visual, "
        "flowing water surface, caustics, overcast soft diffused light, "
        "even tones, cinematic depth of field, bokeh, clear crisp air, high altitude"
    )
    prompt = sys.argv[2] if len(sys.argv) > 2 else default_prompt

    if not Path(clip_path).exists():
        print(f"Clip not found: {clip_path}")
        return 1

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Add it to your .env file or environment.")
        return 1

    print("=" * 78)
    print("CLIP RE-EVAL — Gemini 2.5 Flash (full video upload)")
    print("=" * 78)
    print(f"Clip:   {clip_path}")
    print(f"Size:   {Path(clip_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Prompt: {prompt[:100]}...")
    print()

    print("Uploading to Gemini File API and scoring...")
    result = await evaluate(prompt=prompt, video_path=clip_path)

    if not result.get("success"):
        print(f"\nFAILED: {result.get('error', 'unknown')}")
        print(f"Full result: {json.dumps(result, indent=2)}")
        return 1

    print()
    print("SCORING BREAKDOWN")
    print("-" * 78)
    print(f"  overall score       : {result['score']:.1f} / 10")
    print(f"  prompt match        : {result.get('prompt_match', 0):.1f} / 10")
    print(f"  motion quality      : {result.get('motion_quality', 0):.1f} / 10")
    print(f"  visual coherence    : {result.get('visual_coherence', 0):.1f} / 10")
    print(f"  composition         : {result.get('composition', 0):.1f} / 10")
    print(f"  best domain         : {result.get('best_domain', 'n/a')}  "
          f"(domain_score={result.get('domain_score', 0):.1f})")
    flags = result.get("flags", [])
    if flags:
        print(f"  flags               :")
        for f in flags:
            print(f"    - {f}")
    directives = result.get("next_prompt_directives", [])
    if directives:
        print(f"  next-prompt directives:")
        for d in directives:
            print(f"    - {d}")
    feedback = result.get("feedback", "")
    if feedback:
        print(f"  feedback            : {feedback}")
    print()

    # Update the bandit with the arm IDs from the original run
    arm_ids = {'domain': 9, 'subject': 6, 'lighting': 2, 'quality': 2, 'atmosphere': 6}
    print(f"Updating bandit — arm_ids={arm_ids}, score={result['score']}")
    prompt_bandit.update_reward(arm_ids, float(result["score"]), engine="ltx")
    print("Bandit persisted to data/bandit/ltx-prompt-bandit.json")
    print()
    print("=" * 78)
    print(f"DONE — score {result['score']:.1f}/10")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
