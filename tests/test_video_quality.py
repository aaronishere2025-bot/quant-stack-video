"""
Unit tests for the VLM video quality evaluator.

Tests cover:
  - evaluate() routing: Gemini when GEMINI_API_KEY set, Ollama otherwise
  - _normalize_result: unknown domain clamped, score cast to float
  - _get_video_duration: subprocess success and failure paths
  - Missing prompt returns error dict (no API call)
  - Ollama HTTP error returns success=False
  - Gemini JSON parse error returns success=False

All external API calls are mocked — no GPU, no network required.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _normalize_result
# ---------------------------------------------------------------------------

class TestNormalizeResult:
    def _fn(self, parsed):
        from src.agent.video_quality import _normalize_result
        return _normalize_result(parsed)

    def test_success_true(self):
        r = self._fn({"score": 7.5, "best_domain": "music_video"})
        assert r["success"] is True

    def test_score_cast_to_float(self):
        r = self._fn({"score": "8", "best_domain": "lofi_aesthetic"})
        assert isinstance(r["score"], float)
        assert r["score"] == 8.0

    def test_unknown_domain_replaced_with_default(self):
        r = self._fn({"score": 5.0, "best_domain": "not_a_real_domain"})
        assert r["best_domain"] == "music_video"

    def test_known_domain_preserved(self):
        r = self._fn({"score": 5.0, "best_domain": "nature_travel"})
        assert r["best_domain"] == "nature_travel"

    def test_flags_default_empty(self):
        r = self._fn({"score": 5.0, "best_domain": "music_video"})
        assert r["flags"] == []

    def test_next_prompt_directives_default_empty(self):
        r = self._fn({"score": 5.0, "best_domain": "music_video"})
        assert r["next_prompt_directives"] == []

    def test_domain_score_falls_back_to_score(self):
        r = self._fn({"score": 6.5, "best_domain": "music_video"})
        assert r["domain_score"] == pytest.approx(6.5)


# ---------------------------------------------------------------------------
# evaluate() — routing and guard clauses
# ---------------------------------------------------------------------------

class TestEvaluateRouting:
    def test_empty_prompt_returns_error_without_api_call(self):
        from src.agent.video_quality import evaluate
        result = _run(evaluate(prompt="", video_path=None))
        assert result["success"] is False
        assert "prompt" in result["error"].lower()

    def test_no_gemini_key_routes_to_ollama(self):
        """Without GEMINI_API_KEY, evaluate() should call _evaluate_ollama."""
        from src.agent import video_quality

        async def fake_ollama(prompt, video_path, video_description, n_frames):
            return {"success": True, "score": 7.0, "best_domain": "music_video",
                    "domain_score": 7.0, "flags": [], "next_prompt_directives": []}

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with patch.object(video_quality, "_GEMINI_API_KEY", ""):
                with patch.object(video_quality, "_evaluate_ollama", side_effect=fake_ollama):
                    result = _run(video_quality.evaluate(prompt="A sunny beach"))
        assert result["success"] is True

    def test_gemini_key_routes_to_gemini(self):
        """With GEMINI_API_KEY, evaluate() should call _evaluate_gemini."""
        from src.agent import video_quality

        async def fake_gemini(prompt, video_path):
            return {"success": True, "score": 8.5, "best_domain": "nature_travel",
                    "domain_score": 8.5, "flags": [], "next_prompt_directives": []}

        with patch.object(video_quality, "_GEMINI_API_KEY", "fake-key"):
            with patch.object(video_quality, "_evaluate_gemini", side_effect=fake_gemini):
                result = _run(video_quality.evaluate(prompt="A mountain stream"))
        assert result["success"] is True
        assert result["score"] == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# _evaluate_ollama — HTTP success and failure paths
# ---------------------------------------------------------------------------

class TestEvaluateOllama:
    def _call(self, prompt="A test prompt", video_path=None, mock_response=None):
        """Call _evaluate_ollama with a mocked httpx response."""
        from src.agent import video_quality

        if mock_response is None:
            mock_response = {
                "choices": [{
                    "message": {"content": json.dumps({
                        "score": 7.0,
                        "prompt_match": 7,
                        "motion_quality": 6,
                        "visual_coherence": 7,
                        "composition": 7,
                        "best_domain": "cinematic_narrative",
                        "domain_score": 7.0,
                        "flags": [],
                        "next_prompt_directives": [],
                        "feedback": "Good overall quality.",
                    })}
                }]
            }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()

        import httpx
        mock_post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=mock_post))
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            return _run(video_quality._evaluate_ollama(prompt, video_path, None, 4))

    def test_success_response_parsed(self):
        result = self._call()
        assert result["success"] is True
        assert result["score"] == pytest.approx(7.0)
        assert result["best_domain"] == "cinematic_narrative"

    def test_http_error_returns_failure(self):
        from src.agent import video_quality
        import httpx

        async def raise_http_error(*args, **kwargs):
            raise httpx.HTTPError("Connection refused")

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=MagicMock(post=AsyncMock(side_effect=raise_http_error)))
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(video_quality._evaluate_ollama("test", None, None, 4))

        assert result["success"] is False
        assert "error" in result

    def test_json_parse_error_returns_failure(self):
        """Malformed JSON from Ollama should return success=False."""
        bad_response = {
            "choices": [{"message": {"content": "not valid json {{"}}]
        }
        result = self._call(mock_response=bad_response)
        assert result["success"] is False

    def test_markdown_fence_stripped(self):
        """Ollama sometimes wraps JSON in ```json ... ``` — strip it."""
        content = "```json\n" + json.dumps({
            "score": 6.0,
            "best_domain": "abstract_art",
            "domain_score": 6.0,
            "flags": [],
            "next_prompt_directives": [],
            "feedback": "ok",
        }) + "\n```"
        response = {"choices": [{"message": {"content": content}}]}
        result = self._call(mock_response=response)
        assert result["success"] is True
        assert result["score"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# _get_video_duration — subprocess helper
# ---------------------------------------------------------------------------

class TestGetVideoDuration:
    def test_success_returns_float(self):
        from src.agent.video_quality import _get_video_duration
        import subprocess

        mock_result = MagicMock()
        mock_result.stdout = "5.312000\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            dur = _get_video_duration("/fake/video.mp4")

        assert dur == pytest.approx(5.312)

    def test_failure_returns_zero(self):
        from src.agent.video_quality import _get_video_duration

        with patch("subprocess.run", side_effect=Exception("ffprobe not found")):
            dur = _get_video_duration("/fake/video.mp4")

        assert dur == 0.0

    def test_invalid_output_returns_zero(self):
        from src.agent.video_quality import _get_video_duration

        mock_result = MagicMock()
        mock_result.stdout = "N/A\n"

        with patch("subprocess.run", return_value=mock_result):
            dur = _get_video_duration("/fake/video.mp4")

        assert dur == 0.0


# ---------------------------------------------------------------------------
# DOMAINS constant
# ---------------------------------------------------------------------------

class TestDomains:
    def test_domains_not_empty(self):
        from src.agent.video_quality import DOMAINS
        assert len(DOMAINS) > 0

    def test_domains_all_strings(self):
        from src.agent.video_quality import DOMAINS
        assert all(isinstance(d, str) for d in DOMAINS)

    def test_music_video_in_domains(self):
        from src.agent.video_quality import DOMAINS
        assert "music_video" in DOMAINS
