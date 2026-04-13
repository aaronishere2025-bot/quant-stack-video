#!/usr/bin/env python3
"""
Overnight Thompson-sampling loop on the Apollo 11 moon landing.

Pipeline per scene:
  1. Read runtime_config.json (lets external tuners override params per scene)
  2. Bandit picks arms (LTX-specific: domain/subject/lighting/quality/atmosphere)
  3. LTX generates a clip using the cached pipeline (~90s per scene after warmup)
  4. Gemini 2.5 Flash scores the full MP4 (motion/coherence are now REAL signals)
  5. Compute weighted overall score from dimensions (VLM overall field is ignored)
  6. Bandit reward updates
  7. In-process heuristic tuner adjusts runtime_config for next scene
  8. Fire-and-forget Paperclip heartbeat to Quantization Engineer agent
  9. Write latest_scene.json so the agent can read it on wake

Per iteration:
  - All 10 Apollo 11 scenes generated
  - Clips stitched into iteration.mp4 with ffmpeg concat
  - iteration.json written with full scene records
  - Best iteration tracked in-memory + logged

Logs:
  outputs/overnight_apollo11/
    runtime_config.json            (shared tunable state, read before every scene)
    latest_scene.json              (last scene result — heartbeat agent reads this)
    run.jsonl                      (append-only event stream)
    iter_NNNN/
      scene_NN_<name>.mp4
      scene_NN_<name>_last_frame.png
      iteration.mp4                (concatenated)
      iteration.json               (metadata)

Stop:
  kill <pid>   # SIGTERM — finishes current clip cleanly

Usage:
    .venv/bin/python -u scripts/overnight_apollo11.py
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import psutil

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

from src.llm.prompt_bandit import prompt_bandit  # noqa: E402
from src.wan.generate import generate_video  # noqa: E402
from src.wan.ltx_pipeline_factory import clear_pipeline_cache, offload_pipeline_to_cpu  # noqa: E402
from src.agent.video_quality import evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# Apollo 11 scene definitions — facts → events → base prompts
# ---------------------------------------------------------------------------
APOLLO11_SCENES: list[dict] = [
    {
        "id": 1,
        "name": "launch_ignition",
        "fact": "Saturn V lifts off from LC-39A at Kennedy Space Center, 9:32 AM EDT, July 16 1969",
        "base": (
            "a massive white Saturn V rocket launching from a concrete pad, "
            "huge orange flames and smoke billowing beneath, gantry tower retracting, "
            "slow vertical liftoff against a clear blue morning sky, Kennedy Space Center 1969"
        ),
    },
    {
        "id": 2,
        "name": "ascent_stage_sep",
        "fact": "S-IC first stage separates about 68 km altitude, 2 min 41 s after launch",
        "base": (
            "a rocket stage separating in the upper atmosphere, the huge booster falling away, "
            "white exhaust plume, curved horizon of the Earth visible below, deep blue sky "
            "transitioning to black space"
        ),
    },
    {
        "id": 3,
        "name": "earth_departure",
        "fact": "Trans-lunar injection burn of the S-IVB third stage, Earth orbit escape",
        "base": (
            "view of Earth from space, half illuminated by the sun, deep black space behind, "
            "white clouds swirling over blue oceans, continents visible, spacecraft departing in "
            "foreground with engine exhaust"
        ),
    },
    {
        "id": 4,
        "name": "zero_g_cabin",
        "fact": "Three astronauts in the Command Module during coast to Moon, 3 days",
        "base": (
            "three astronauts in white spacesuits floating weightlessly inside a cramped "
            "spacecraft cabin, analog control panels with hundreds of switches and gauges, "
            "small round porthole window, soft cabin lighting"
        ),
    },
    {
        "id": 5,
        "name": "lunar_approach",
        "fact": "Lunar orbit insertion, crater-pocked Moon fills the window",
        "base": (
            "the grey cratered surface of the Moon filling most of the frame, harsh sunlight "
            "casting long dark shadows from craters, the curved lunar horizon, deep black space above, "
            "spacecraft approaching in low orbit"
        ),
    },
    {
        "id": 6,
        "name": "powered_descent",
        "fact": "Lunar Module Eagle descends to the Sea of Tranquility, July 20 1969",
        "base": (
            "a gold foil and metal lunar lander descending toward the Moon's surface, "
            "exhaust plume kicking up fine grey dust, four landing legs extended, "
            "rocky crater field below, dramatic sidelighting from low sun"
        ),
    },
    {
        "id": 7,
        "name": "touchdown",
        "fact": "Eagle lands 20:17 UTC with less than 30 seconds of fuel remaining",
        "base": (
            "a lunar lander touching down on a flat grey plain, footpad compressing in loose dust, "
            "dust particles drifting in the low lunar gravity, long sharp shadow, black starless "
            "sky above the horizon"
        ),
    },
    {
        "id": 8,
        "name": "first_step",
        "fact": "Armstrong steps off the ladder 02:56:15 UTC July 21 1969",
        "base": (
            "a white-spacesuited astronaut descending a metal ladder from a lunar lander, "
            "bulky backpack life support unit, gold visor helmet reflecting the lander, "
            "bootprint pressed into fine grey lunar dust, stark shadows"
        ),
    },
    {
        "id": 9,
        "name": "flag_planting",
        "fact": "Armstrong and Aldrin plant the US flag approximately 6.4 meters from the LM",
        "base": (
            "two astronauts in white spacesuits standing on the Moon's surface planting a flag "
            "with a horizontal rod so it appears to wave, lunar lander in the background, "
            "long sharp shadows, harsh direct sunlight, pristine grey dust"
        ),
    },
    {
        "id": 10,
        "name": "splashdown",
        "fact": "Command Module Columbia splashes down in the Pacific Ocean July 24 1969",
        "base": (
            "a charred conical capsule with three orange and white parachutes descending toward "
            "the open blue Pacific Ocean, gentle waves below, sunny daylight, recovery helicopter "
            "hovering in the distance"
        ),
    },
]

# ---------------------------------------------------------------------------
# Defaults (overridable via runtime_config.json)
# ---------------------------------------------------------------------------

DEFAULT_LTX_PARAMS = {
    "height": 512,
    "width": 768,
    "num_frames": 121,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "fps": 25,
}
LTX_CACHE_DIR = "/home/aaron/.cache/huggingface"

ACCEPTABLE_SCORE = 7.0  # per-iteration average threshold

VLM_MODEL = "gemini-2.5-flash"  # cloud eval — no local VRAM impact

# Paperclip — Quantization Engineer agent
QUANT_ENGINEER_AGENT_ID = "5afc29a0-404c-4c92-b7b4-abb2e1f82c41"
PAPERCLIP_HEARTBEAT_TIMEOUT = 5  # seconds — fire and forget

# Memory watchdog — override with APOLLO_MEMORY_LIMIT_GB env var
MEMORY_LIMIT_GB = float(os.environ.get("APOLLO_MEMORY_LIMIT_GB", "8.0"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("/mnt/d/ai-workspace/videos/overnight_apollo11")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUTPUT_ROOT / "run.jsonl"
RUNTIME_CONFIG = OUTPUT_ROOT / "runtime_config.json"
LATEST_SCENE = OUTPUT_ROOT / "latest_scene.json"

_STOP_REQUESTED = False


# ---------------------------------------------------------------------------
# runtime_config.json — shared tunable state
# ---------------------------------------------------------------------------

def _default_runtime_config() -> dict:
    return {
        "version": 1,
        "scene_overrides": {},  # {scene_id: {"base_prompt": str, "num_inference_steps": int, ...}}
        "carry_forward_directives": [],  # global directives appended to every prompt
        "disabled_arms": {},  # {category: [arm_idx, ...]}  (TODO: honor in bandit)
        "scene_failure_counts": {},  # {scene_id: int} — tracked by tuner
    }


def load_runtime_config() -> dict:
    if RUNTIME_CONFIG.exists():
        try:
            return json.loads(RUNTIME_CONFIG.read_text())
        except Exception as e:
            print(f"[warn] runtime_config corrupt ({e}), using defaults")
    return _default_runtime_config()


def save_runtime_config(cfg: dict) -> None:
    RUNTIME_CONFIG.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

def _log_event(event: dict) -> None:
    event["ts"] = datetime.now().isoformat(timespec="seconds")
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _on_sigterm(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\n[STOP] Signal {signum} received — finishing current clip and exiting.")


signal.signal(signal.SIGTERM, _on_sigterm)
signal.signal(signal.SIGINT, _on_sigterm)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_rss_gb() -> float:
    """Return current process RSS in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _cleanup_iteration_gpu() -> None:
    """Offload pipeline to CPU, flush VRAM allocator, and run GC.

    Called at the end of every iteration to prevent RSS from compounding
    across iterations (the root cause of the 21 GB leak).
    """
    offload_pipeline_to_cpu()
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Per-scene generation + scoring
# ---------------------------------------------------------------------------


async def generate_and_score_scene(
    scene: dict,
    iteration_dir: Path,
    iteration_idx: int,
    prev_frame: str | None,
    cfg: dict,
) -> dict:
    """Generate one scene clip, score it with multi-frame VLM, update bandit, return record."""
    # Apply overrides in priority order: DEFAULT < _all < per-scene
    scene_id = str(scene["id"])
    all_override = cfg.get("scene_overrides", {}).get("_all", {})
    scene_override = cfg.get("scene_overrides", {}).get(scene_id, {})
    params = {**DEFAULT_LTX_PARAMS, **all_override, **scene_override}
    # Keep only recognized params to avoid passing junk to generate_video
    params = {k: params[k] for k in DEFAULT_LTX_PARAMS if k in params}
    base_prompt = scene_override.get("base_prompt", scene["base"])

    # Bandit picks arms on top of the base prompt
    enhanced, arm_ids = prompt_bandit.build_enhanced_prompt(base_prompt, engine="ltx")

    # Append any global carry-forward directives (from the tuner)
    directives = cfg.get("carry_forward_directives", [])
    if directives:
        enhanced = enhanced + ", " + ", ".join(directives)

    clip_path = str(iteration_dir / f"scene_{scene['id']:02d}_{scene['name']}.mp4")

    # LTX generation
    t0 = time.time()
    try:
        generate_video(
            prompt=enhanced,
            output_path=clip_path,
            engine="ltx",
            model_id="Lightricks/LTX-Video",
            cache_dir=LTX_CACHE_DIR,
            height=params["height"],
            width=params["width"],
            num_frames=params["num_frames"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            seed=int(time.time()) % (2**31),
            fps=params["fps"],
            image_path=prev_frame,  # EchoShot: last frame of previous scene
        )
    except Exception as e:
        return {
            "scene_id": scene["id"],
            "scene_name": scene["name"],
            "iteration": iteration_idx,
            "status": "gen_failed",
            "error": str(e),
            "arm_ids": arm_ids,
            "enhanced_prompt": enhanced,
            "params": params,
        }
    gen_elapsed = time.time() - t0

    # Offload LTX to CPU before loading Qwen2.5-VL (both need ~6-8 GB on a 12 GB card).
    # offload_pipeline_to_cpu() keeps the pipeline in _PIPELINE_CACHE so the next
    # scene moves CPU→GPU in seconds rather than reloading from disk.
    offload_pipeline_to_cpu()

    # VLM eval (4 frames, Qwen2.5-VL)
    t_eval = time.time()
    result = await evaluate(prompt=enhanced, video_path=clip_path)
    eval_elapsed = time.time() - t_eval

    if not result.get("success"):
        return {
            "scene_id": scene["id"],
            "scene_name": scene["name"],
            "iteration": iteration_idx,
            "status": "eval_failed",
            "error": result.get("error"),
            "arm_ids": arm_ids,
            "enhanced_prompt": enhanced,
            "clip_path": clip_path,
            "gen_elapsed": gen_elapsed,
            "eval_elapsed": eval_elapsed,
            "params": params,
        }

    # Compute our own overall from dimensions — VLM overall is unreliable
    dims = {
        "prompt_match": float(result.get("prompt_match", 5.0)),
        "motion_quality": float(result.get("motion_quality", 5.0)),
        "visual_coherence": float(result.get("visual_coherence", 5.0)),
        "composition": float(result.get("composition", 5.0)),
    }
    weights = {
        "prompt_match": 0.40,
        "visual_coherence": 0.25,
        "motion_quality": 0.20,
        "composition": 0.15,
    }
    computed_score = sum(dims[k] * weights[k] for k in dims)

    # Update bandit with our computed score
    prompt_bandit.update_reward(arm_ids, computed_score, engine="ltx")

    last_frame_path = clip_path.replace(".mp4", "_last_frame.png")
    return {
        "scene_id": scene["id"],
        "scene_name": scene["name"],
        "iteration": iteration_idx,
        "status": "ok",
        "arm_ids": arm_ids,
        "enhanced_prompt": enhanced,
        "clip_path": clip_path,
        "last_frame_path": last_frame_path if Path(last_frame_path).exists() else None,
        "gen_elapsed": gen_elapsed,
        "eval_elapsed": eval_elapsed,
        "dimensions": dims,
        "vlm_overall": float(result.get("score", 0)),
        "computed_score": computed_score,
        "best_domain": result.get("best_domain"),
        "flags": result.get("flags", []),
        "directives": result.get("next_prompt_directives", []),
        "feedback": result.get("feedback", ""),
        "params": params,
    }


# ---------------------------------------------------------------------------
# Heuristic tuner — runs in-process after every scene
# ---------------------------------------------------------------------------

# Rolling stats (per-process, reset on restart) — used by the tuner
_recent_scores: list[float] = []
_recent_motion: list[float] = []
_recent_coherence: list[float] = []
_MAX_RECENT = 10

TUNER_MAX_STEPS = 60              # ceiling when bumping num_inference_steps
TUNER_MIN_STEPS = 30              # floor
TUNER_MOTION_BUMP_THRESHOLD = 4.0  # if rolling motion avg below this, bump steps
TUNER_CARRY_MAX = 3                # max carry-forward directives


def tune_after_scene(rec: dict, cfg: dict) -> dict:
    """Apply heuristic rules to runtime_config based on the latest scene record.

    Returns the updated cfg (also persists to disk). Safe to call on any record
    status — only 'ok' records affect numeric tuning, but all statuses update
    failure counts.
    """
    global _recent_scores, _recent_motion, _recent_coherence

    scene_id = str(rec.get("scene_id", ""))
    if not scene_id:
        return cfg

    # Track failures
    failure_counts = cfg.setdefault("scene_failure_counts", {})
    if rec.get("status") != "ok":
        failure_counts[scene_id] = failure_counts.get(scene_id, 0) + 1
        save_runtime_config(cfg)
        return cfg

    # Reset failure counter on success
    if scene_id in failure_counts and failure_counts[scene_id] > 0:
        failure_counts[scene_id] = 0

    dims = rec.get("dimensions", {})
    score = rec.get("computed_score", 0.0)

    _recent_scores.append(score)
    _recent_motion.append(dims.get("motion_quality", 5.0))
    _recent_coherence.append(dims.get("visual_coherence", 5.0))
    for buf in (_recent_scores, _recent_motion, _recent_coherence):
        while len(buf) > _MAX_RECENT:
            buf.pop(0)

    changes = []

    # Rule 1: low prompt_match → add this scene's directives to carry-forward
    prompt_match = dims.get("prompt_match", 5.0)
    if prompt_match < 5.0 and rec.get("directives"):
        existing = cfg.setdefault("carry_forward_directives", [])
        for d in rec["directives"][:2]:
            if d and d not in existing:
                existing.append(d)
        # Keep the list bounded
        if len(existing) > TUNER_CARRY_MAX:
            cfg["carry_forward_directives"] = existing[-TUNER_CARRY_MAX:]
        changes.append(f"carry_forward += {rec['directives'][:2]}")

    # Rule 2: rolling motion avg < threshold → bump num_inference_steps globally
    if len(_recent_motion) >= 3:
        avg_motion = sum(_recent_motion) / len(_recent_motion)
        if avg_motion < TUNER_MOTION_BUMP_THRESHOLD:
            # Find the current global step count from any existing override or default
            current_steps = DEFAULT_LTX_PARAMS["num_inference_steps"]
            # We apply as a global default bump via an "all" pseudo-override
            all_override = cfg.setdefault("scene_overrides", {}).setdefault("_all", {})
            now_steps = all_override.get("num_inference_steps", current_steps)
            if now_steps < TUNER_MAX_STEPS:
                all_override["num_inference_steps"] = min(now_steps + 5, TUNER_MAX_STEPS)
                changes.append(f"steps {now_steps}→{all_override['num_inference_steps']} (motion avg {avg_motion:.1f})")

    # Rule 3: if rolling motion avg is HEALTHY (>=6), trickle steps back down
    if len(_recent_motion) >= 5:
        avg_motion = sum(_recent_motion) / len(_recent_motion)
        if avg_motion >= 6.0:
            all_override = cfg.get("scene_overrides", {}).get("_all", {})
            if all_override.get("num_inference_steps", 0) > DEFAULT_LTX_PARAMS["num_inference_steps"]:
                new_steps = max(all_override["num_inference_steps"] - 2, DEFAULT_LTX_PARAMS["num_inference_steps"])
                all_override["num_inference_steps"] = new_steps
                changes.append(f"steps trickling down → {new_steps} (motion avg {avg_motion:.1f})")

    # Rule 4: scene failed 3+ times → flag for base prompt rewrite
    # (we don't auto-rewrite — we add a note for the Paperclip agent/human to see)
    if failure_counts.get(scene_id, 0) >= 3:
        flags = cfg.setdefault("needs_rewrite", [])
        if scene_id not in flags:
            flags.append(scene_id)
            changes.append(f"scene {scene_id} flagged for rewrite")

    if changes:
        cfg.setdefault("tuner_history", []).append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "iteration": rec.get("iteration"),
            "scene_id": rec.get("scene_id"),
            "changes": changes,
        })
        # Keep tuner_history bounded
        if len(cfg["tuner_history"]) > 200:
            cfg["tuner_history"] = cfg["tuner_history"][-200:]

    save_runtime_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Ollama VLM lifecycle — explicit unload so LTX inference doesn't OOM
# ---------------------------------------------------------------------------

def unload_vlm() -> None:
    """No-op: Gemini eval is cloud-based, no local VRAM to reclaim."""
    pass


# ---------------------------------------------------------------------------
# Paperclip heartbeat trigger — fire and forget
# ---------------------------------------------------------------------------

def fire_heartbeat(scene_rec: dict) -> None:
    """Write latest_scene.json and trigger a Paperclip heartbeat on the
    Quantization Engineer agent. Never blocks the loop on failure."""
    try:
        LATEST_SCENE.write_text(json.dumps(scene_rec, indent=2))
    except Exception as e:
        print(f"  [heartbeat] write latest_scene failed: {e}")
        return

    try:
        # Fire and forget — detach from our stdout, short timeout, don't wait
        subprocess.Popen(
            ["npx", "paperclipai", "heartbeat", "run",
             "--agent-id", QUANT_ENGINEER_AGENT_ID,
             "--source", "automation",
             "--trigger", "callback",
             f"--timeout-ms", str(PAPERCLIP_HEARTBEAT_TIMEOUT * 1000)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach so SIGTERM to us doesn't kill it
        )
    except Exception as e:
        print(f"  [heartbeat] trigger failed: {e}")


# ---------------------------------------------------------------------------
# ffmpeg concat stitch
# ---------------------------------------------------------------------------

def stitch_iteration(iteration_dir: Path, scene_records: list[dict]) -> str | None:
    ok = [r for r in scene_records if r.get("status") == "ok"]
    if not ok:
        return None
    concat_list = iteration_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{Path(r['clip_path']).resolve()}'" for r in ok) + "\n"
    )
    out = str(iteration_dir / "iteration.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy", out],
            capture_output=True, timeout=60, check=True,
        )
        return out
    except Exception as e:
        print(f"  [stitch] ffmpeg failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main() -> int:
    global _STOP_REQUESTED
    # Support a quick smoke test with --max-scenes N --max-iterations N
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="Stop after N iterations (0 = forever)")
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="Only generate first N scenes per iteration (0 = all)")
    args = parser.parse_args()

    scenes_to_run = APOLLO11_SCENES if args.max_scenes == 0 else APOLLO11_SCENES[:args.max_scenes]

    print("=" * 78)
    print("APOLLO 11 OVERNIGHT LOOP")
    print(f"  pid:         {os.getpid()}")
    print(f"  output dir:  {OUTPUT_ROOT}")
    print(f"  run log:     {RUN_LOG}")
    print(f"  scenes:      {len(scenes_to_run)} {'(smoke test)' if args.max_scenes else ''}")
    print(f"  iterations:  {'forever' if args.max_iterations == 0 else args.max_iterations}")
    print(f"  vlm:         {VLM_MODEL} (full video, Gemini File API)")
    print(f"  stop:        kill {os.getpid()}")
    print("=" * 78)

    # Initialize runtime_config if absent
    if not RUNTIME_CONFIG.exists():
        save_runtime_config(_default_runtime_config())
        print(f"  [init] wrote default runtime_config.json")
    print()

    _log_event({
        "event": "run_start",
        "pid": os.getpid(),
        "scenes": [s["name"] for s in scenes_to_run],
        "vlm_model": VLM_MODEL,
        "ltx_defaults": DEFAULT_LTX_PARAMS,
        "args": vars(args),
    })

    iteration_idx = 0
    best_iteration = {"idx": None, "avg": -1}

    while not _STOP_REQUESTED:
        if args.max_iterations > 0 and iteration_idx >= args.max_iterations:
            print(f"\n[STOP] Reached max iterations ({args.max_iterations})")
            break

        iteration_idx += 1
        iter_dir = OUTPUT_ROOT / f"iter_{iteration_idx:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'━' * 78}")
        print(f"ITERATION {iteration_idx}  →  {iter_dir.name}")
        print(f"{'━' * 78}")

        iter_rss = get_rss_gb()
        print(f"  [mem] iteration start: {iter_rss:.2f} GB RSS  (limit {MEMORY_LIMIT_GB:.1f} GB)")
        _log_event({"event": "mem_iter_start", "iteration": iteration_idx, "rss_gb": round(iter_rss, 2)})

        iter_t0 = time.time()
        scene_records: list[dict] = []
        prev_frame: str | None = None

        for scene in scenes_to_run:
            if _STOP_REQUESTED:
                print("[STOP] breaking mid-iteration")
                break

            # Re-read runtime_config each scene so external tuners can take effect
            cfg = load_runtime_config()

            print(f"\n  [scene {scene['id']}/{len(scenes_to_run)}] {scene['name']}")
            print(f"    fact: {scene['fact']}")

            try:
                rec = await generate_and_score_scene(
                    scene, iter_dir, iteration_idx, prev_frame, cfg
                )
            except Exception as e:
                rec = {
                    "scene_id": scene["id"],
                    "scene_name": scene["name"],
                    "iteration": iteration_idx,
                    "status": "exception",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

            # Free the VLM from VRAM so the next LTX inference has room
            unload_vlm()

            scene_records.append(rec)
            _log_event({"event": "scene_done", **rec})

            if rec.get("status") == "ok":
                prev_frame = rec.get("last_frame_path")
                dims = rec["dimensions"]
                print(f"    ✓ gen={rec['gen_elapsed']:.0f}s "
                      f"eval={rec['eval_elapsed']:.0f}s "
                      f"score={rec['computed_score']:.2f} "
                      f"(pm={dims['prompt_match']:.0f}/"
                      f"mo={dims['motion_quality']:.0f}/"
                      f"co={dims['visual_coherence']:.0f}/"
                      f"cp={dims['composition']:.0f})")
                if rec.get("flags"):
                    print(f"      flags: {', '.join(rec['flags'][:3])}")
            else:
                print(f"    ✗ {rec.get('status')}: {str(rec.get('error', '?'))[:100]}")
                prev_frame = None

            # Heuristic tuner — runs after every scene
            cfg = tune_after_scene(rec, cfg)

            # Paperclip heartbeat — fire and forget
            fire_heartbeat(rec)

            gc.collect()

            # Memory watchdog — log RSS and abort if over limit
            rss = get_rss_gb()
            _log_event({"event": "mem_sample", "scene_id": scene["id"],
                        "iteration": iteration_idx, "rss_gb": round(rss, 2)})
            print(f"    [mem] {rss:.2f} GB RSS")
            if rss > MEMORY_LIMIT_GB:
                print(f"\n[WATCHDOG] RSS {rss:.2f} GB exceeds limit "
                      f"{MEMORY_LIMIT_GB:.1f} GB — stopping cleanly.")
                _log_event({"event": "memory_limit_exceeded",
                            "rss_gb": round(rss, 2), "limit_gb": MEMORY_LIMIT_GB})
                _STOP_REQUESTED = True
                break

        # GPU / model cleanup at iteration boundary — prevents RSS compounding
        _cleanup_iteration_gpu()
        post_rss = get_rss_gb()
        print(f"  [mem] iteration end (post-cleanup): {post_rss:.2f} GB RSS")
        _log_event({"event": "mem_iter_end", "iteration": iteration_idx,
                    "rss_gb": round(post_rss, 2)})

        # Stitch + summarize
        ok_recs = [r for r in scene_records if r.get("status") == "ok"]
        iteration_elapsed = time.time() - iter_t0
        avg_score = (sum(r["computed_score"] for r in ok_recs) / len(ok_recs)
                     if ok_recs else 0.0)
        final_video = stitch_iteration(iter_dir, scene_records) if ok_recs else None
        is_acceptable = avg_score >= ACCEPTABLE_SCORE
        is_best = avg_score > best_iteration["avg"]
        if is_best:
            best_iteration = {"idx": iteration_idx, "avg": avg_score}

        iter_meta = {
            "iteration": iteration_idx,
            "elapsed": iteration_elapsed,
            "scenes_ok": len(ok_recs),
            "scenes_total": len(scenes_to_run),
            "avg_score": avg_score,
            "acceptable": is_acceptable,
            "final_video": final_video,
            "scene_records": scene_records,
        }
        (iter_dir / "iteration.json").write_text(json.dumps(iter_meta, indent=2))
        _log_event({"event": "iteration_done",
                    **{k: v for k, v in iter_meta.items() if k != "scene_records"}})

        print()
        print(f"  iteration {iteration_idx}: avg={avg_score:.2f}/10 "
              f"ok={len(ok_recs)}/{len(scenes_to_run)} "
              f"time={iteration_elapsed/60:.1f}m"
              f"{' [ACCEPTABLE ✓]' if is_acceptable else ''}"
              f"{' [NEW BEST]' if is_best else ''}")
        if final_video:
            print(f"  → {final_video}")
        print(f"  best so far: iter {best_iteration['idx']} @ "
              f"{best_iteration['avg']:.2f}")

    # Clean exit
    print("\n[EXIT] clearing pipeline cache...")
    clear_pipeline_cache()
    _log_event({"event": "run_stop", "total_iterations": iteration_idx,
                "best_iteration": best_iteration})
    print(f"Done. {iteration_idx} iterations. Best: {best_iteration}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
