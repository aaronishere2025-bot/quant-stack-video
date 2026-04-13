"""LLM-based video quality evaluator.

Scores an AI-generated clip on quality dimensions and domain fit.

Primary backend: Gemini 2.5 Flash (GEMINI_API_KEY env var) — uploads the full
MP4 via the File API so Gemini watches the actual video, not just sampled frames.

Fallback: local Ollama VLM (LLM_EVAL_URL / LLM_EVAL_MODEL) — sends 4 evenly-
spaced frames as base64 JPEGs.

Usage:
    result = await evaluate(prompt="...", video_path="/path/to/clip.mp4")
    # or without a video file:
    result = await evaluate(prompt="...", video_description="dark moody scene")

Returns a dict with:
    score                   — overall 0-10 float
    prompt_match            — 0-10
    motion_quality          — 0-10
    visual_coherence        — 0-10
    composition             — 0-10
    best_domain             — one of DOMAINS
    domain_score            — how well it fits that domain (0-10)
    next_prompt_directives  — list of corrective phrases for next generation
    flags                   — list of specific visual issues
    feedback                — one-sentence summary
    success                 — bool
    error                   — present only on failure
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ── Backend selection ────────────────────────────────────────────────────────
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

_LLM_EVAL_URL = os.environ.get("LLM_EVAL_URL", "http://localhost:11434/v1/chat/completions")
_LLM_MODEL = os.environ.get("LLM_EVAL_MODEL", "qwen2.5vl:7b")

# Must match DOMAIN_ARMS in the prompt bandit
DOMAINS = [
    "music_video",
    "lofi_aesthetic",
    "trap_urban",
    "marketing_brand",
    "history_documentary",
    "science_educational",
    "nature_travel",
    "abstract_art",
    "cinematic_narrative",
    "wellness_meditation",
]

_SYSTEM = (
    "You are a video quality evaluator for an AI-powered YouTube channel. "
    "You evaluate short AI-generated video clips for visual quality and content domain fit. "
    "Return ONLY valid JSON — no markdown fences, no text outside the JSON object."
)

_EVAL_PROMPT = """Evaluate this AI-generated video clip.

Watch the full video carefully, paying attention to motion between frames,
subject consistency across time, and how well the content matches the prompt.

Prompt used to generate it:
{prompt}

Score on these dimensions (0-10 each):
- prompt_match: How closely does the content match the prompt text? Be strict.
- motion_quality: Is there visible, plausible movement throughout? 0 = static/frozen,
  10 = smooth natural motion. Jittery/morphing/teleporting subjects score LOW.
- visual_coherence: Do subjects, objects, and lighting persist consistently across
  the whole clip? 0 = morphing/flickering, 10 = fully stable.
- composition: Is the framing aesthetically intentional and well-balanced?

Also assess domain fit — which of these domains does this clip serve BEST:
  music_video | lofi_aesthetic | trap_urban | marketing_brand |
  history_documentary | science_educational | nature_travel |
  abstract_art | cinematic_narrative | wellness_meditation

For flags, list specific correctable visual problems you observe
(e.g. "subjects too dark", "background morphs", "motion is static",
"limbs duplicate", "foreground lacks detail").

For next_prompt_directives, write 1-3 SHORT corrective phrases to APPEND to
the next generation prompt. Keep them actionable and prompt-compatible.
Examples: "increase foreground brightness", "sharper subject edges",
"add more textural detail to surfaces"

Return exactly this JSON — no markdown, no trailing text:
{{
  "score": <0-10 number, one decimal>,
  "prompt_match": <0-10>,
  "motion_quality": <0-10>,
  "visual_coherence": <0-10>,
  "composition": <0-10>,
  "best_domain": "<one of the domain strings above>",
  "domain_score": <0-10>,
  "flags": ["issue 1", "issue 2"],
  "next_prompt_directives": ["directive 1", "directive 2"],
  "feedback": "<one sentence overall summary>"
}}"""


# ── Public entry point ───────────────────────────────────────────────────────

async def evaluate(
    prompt: str,
    video_path: str | None = None,
    video_description: str | None = None,
    n_frames: int = 4,  # only used by Ollama fallback
) -> dict:
    """Evaluate an AI-generated video clip.

    Uses Gemini 2.5 Flash if GEMINI_API_KEY is set (uploads full MP4).
    Falls back to local Ollama VLM (4 evenly-spaced frames) otherwise.
    """
    if not prompt:
        return {"success": False, "error": "prompt is required"}

    if not _GEMINI_API_KEY:
        return {"success": False, "error": "GEMINI_API_KEY not set — Gemini is the only supported evaluator"}
    return await _evaluate_gemini(prompt, video_path)


# ── Gemini backend ───────────────────────────────────────────────────────────

async def _evaluate_gemini(prompt: str, video_path: str | None) -> dict:
    """Upload the full MP4 to Gemini File API and score it."""
    try:
        import google.genai as genai
        from google.genai import types as genai_types
    except ImportError:
        return {"success": False, "error": "google-genai not installed — run: pip install google-genai"}

    client = genai.Client(api_key=_GEMINI_API_KEY)
    eval_text = _EVAL_PROMPT.format(prompt=prompt)

    try:
        if video_path and Path(video_path).exists():
            # Upload the video file — Gemini watches it natively
            logger.info("Uploading %s to Gemini File API...", video_path)
            video_file = await asyncio.to_thread(
                client.files.upload,
                file=video_path,
                config=genai_types.UploadFileConfig(mime_type="video/mp4"),
            )

            # Wait for processing (usually a few seconds for short clips)
            for _ in range(30):
                video_file = await asyncio.to_thread(client.files.get, name=video_file.name)
                if video_file.state.name == "ACTIVE":
                    break
                if video_file.state.name == "FAILED":
                    raise RuntimeError(f"Gemini file processing failed: {video_file.name}")
                await asyncio.sleep(2)
            else:
                raise RuntimeError("Gemini file processing timed out")

            contents = [video_file, eval_text]
        else:
            # No video — evaluate from prompt description only
            desc = f"(no video file available — evaluate from prompt description only)"
            contents = [eval_text + f"\n\nContext: {desc}"]

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=_GEMINI_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.1,
            ),
        )

        # Clean up the uploaded file
        if video_path and Path(video_path).exists():
            try:
                await asyncio.to_thread(client.files.delete, name=video_file.name)
            except Exception as cleanup_err:
                logger.debug("Gemini file cleanup failed (non-fatal): %s", repr(cleanup_err))

        # response.text raises ValueError when Gemini blocks or returns no text part
        try:
            raw = response.text
        except ValueError as e:
            return {"success": False, "error": f"Gemini returned no text (blocked/empty): {e}"}

        if not raw or not raw.strip():
            return {"success": False, "error": "Gemini returned an empty response"}

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        raw = raw.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Gemini JSON parse error: %s — raw response: %.500s", e, raw)
            return {"success": False, "error": f"JSON parse error: {e} — raw: {raw[:200]}"}
        return _normalize_result(parsed)

    except Exception as e:
        err_msg = repr(e) if not str(e) else str(e)
        logger.warning("Gemini evaluation error: %s", err_msg, exc_info=True)
        return {"success": False, "error": f"Gemini error: {err_msg}"}


# ── Ollama fallback ──────────────────────────────────────────────────────────

async def _evaluate_ollama(
    prompt: str,
    video_path: str | None,
    video_description: str | None,
    n_frames: int,
) -> dict:
    """Send n_frames evenly-spaced frames to a local Ollama VLM."""
    eval_text = _EVAL_PROMPT.format(prompt=prompt)
    messages = [{"role": "system", "content": _SYSTEM}]

    if video_path and Path(video_path).exists():
        frames_b64 = _extract_frame_sequence_b64(video_path, n_frames=n_frames)
        if frames_b64:
            content: list = [
                {"type": "text",
                 "text": (f"The following {len(frames_b64)} images are frames sampled "
                          f"evenly from a short AI-generated video clip, in chronological order "
                          f"(first → last):")},
            ]
            for b64 in frames_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            content.append({"type": "text", "text": eval_text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": eval_text})
    else:
        desc = video_description or "(no video file — evaluate from prompt description only)"
        messages.append({"role": "user", "content": eval_text + f"\n\nContext: {desc}"})

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                _LLM_EVAL_URL,
                json={"model": _LLM_MODEL, "messages": messages, "temperature": 0.1},
            )
            resp.raise_for_status()
            body = resp.json()

        raw = body["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())
        return _normalize_result(parsed)

    except httpx.HTTPError as e:
        logger.warning("Ollama request failed: %s", e)
        return {"success": False, "error": f"LLM request failed: {e}"}
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error from Ollama: %s", e)
        return {"success": False, "error": f"JSON parse error: {e}"}
    except Exception as e:
        logger.warning("Ollama evaluation error: %s", e)
        return {"success": False, "error": f"Evaluation error: {e}"}


# ── Shared helpers ───────────────────────────────────────────────────────────

def _normalize_result(parsed: dict) -> dict:
    best_domain = parsed.get("best_domain", "music_video")
    if best_domain not in DOMAINS:
        best_domain = "music_video"
    return {
        "success": True,
        **parsed,
        "score": float(parsed.get("score", 5.0)),
        "best_domain": best_domain,
        "domain_score": float(parsed.get("domain_score", parsed.get("score", 5.0))),
        "next_prompt_directives": parsed.get("next_prompt_directives", []),
        "flags": parsed.get("flags", []),
    }


def _get_video_duration(video_path: str) -> float:
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _extract_frame_sequence_b64(video_path: str, n_frames: int = 4) -> list[str]:
    """Extract n_frames evenly-spaced frames as base64 JPEGs (Ollama fallback)."""
    import subprocess
    import tempfile

    duration = _get_video_duration(video_path)
    if duration <= 0:
        single = _extract_first_frame_b64(video_path)
        return [single] if single else []

    if n_frames < 1:
        n_frames = 1
    end_safe = max(0.0, duration - 0.05)
    if n_frames == 1 or end_safe <= 0:
        timestamps = [0.0]
    else:
        timestamps = [end_safe * i / (n_frames - 1) for i in range(n_frames)]

    frames_b64: list[str] = []
    for ts in timestamps:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            result = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{ts:.3f}", "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", tmp_path],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0 and Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
                with open(tmp_path, "rb") as f:
                    frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))
        except Exception as e:
            logger.debug("frame extract failed at t=%.2f: %s", ts, e)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    return frames_b64


def _extract_first_frame_b64(video_path: str) -> str | None:
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", video_path, "-frames:v", "1", "-q:v", "2", tmp_path],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None
