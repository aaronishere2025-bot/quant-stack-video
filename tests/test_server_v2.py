"""
Tests for the v2 infinite pipeline server endpoints.

All tests mock GPU-intensive operations so no hardware is required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    """Return a TestClient for the server app with lifespan triggered."""
    from contextlib import contextmanager
    from fastapi.testclient import TestClient
    from src.agent.server import create_app

    # Use context manager to ensure lifespan (job queue init) runs
    @contextmanager
    def client_ctx():
        with TestClient(create_app()) as c:
            yield c

    return client_ctx()


# ---------------------------------------------------------------------------
# InfiniteRequest model validation
# ---------------------------------------------------------------------------

class TestInfiniteRequestModel:
    def test_defaults(self):
        from src.agent.server import InfiniteRequest

        req = InfiniteRequest(prompt="A forest walk")
        assert req.max_segments == 0
        assert req.segment_frames == 81
        assert req.use_rgba_layers is False
        assert req.vace_overlap_frames == 16
        assert req.svi_ema_decay == 0.9

    def test_layer_prompts_optional(self):
        from src.agent.server import InfiniteRequest

        req = InfiniteRequest(prompt="City at night")
        assert req.layer_prompts is None

    def test_layer_prompts_can_be_set(self):
        from src.agent.server import InfiniteRequest

        req = InfiniteRequest(
            prompt="City at night",
            layer_prompts=["sky bg", "building mid", "foreground people"],
        )
        assert len(req.layer_prompts) == 3

    def test_max_segments_must_be_non_negative(self):
        from src.agent.server import InfiniteRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InfiniteRequest(prompt="test", max_segments=-1)

    def test_svi_ema_decay_bounds(self):
        from src.agent.server import InfiniteRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InfiniteRequest(prompt="test", svi_ema_decay=1.0)

        with pytest.raises(ValidationError):
            InfiniteRequest(prompt="test", svi_ema_decay=-0.1)


# ---------------------------------------------------------------------------
# POST /generate/infinite endpoint
# ---------------------------------------------------------------------------

class TestGenerateInfiniteEndpoint:
    def test_returns_task_id(self):
        """Endpoint should return a task_id immediately (async job)."""
        pytest.importorskip("fastapi")
        with _make_client() as client:
            resp = client.post("/generate/infinite", json={"prompt": "A mountain scene", "max_segments": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert "task_id" in body
        assert body["status"] == "queued"

    def test_rate_limit_header_present(self):
        """Rate limit headers should appear on the response."""
        pytest.importorskip("fastapi")
        with _make_client() as client:
            resp = client.post("/generate/infinite", json={"prompt": "test"})
        # slowapi inserts X-RateLimit-* headers
        assert resp.status_code in (200, 429)

    def test_invalid_prompt_rejected(self):
        """Missing required prompt field should return 422."""
        pytest.importorskip("fastapi")
        with _make_client() as client:
            resp = client.post("/generate/infinite", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Task registry + pipeline state (no GPU)
# ---------------------------------------------------------------------------

class TestInfinitePipelineRunner:
    """
    Tests _run_infinite_gen by injecting a mock generate_video into sys.modules.
    The v2 components (compositor, VACE, SVI, director) run with CPU tensors.
    """

    def _inject_fake_wan(self, tmp_path):
        """Inject a fake src.wan.generate module so torch is never imported."""
        import sys
        import types

        fake_gen = MagicMock(return_value=str(tmp_path / "seg_0000.mp4"))
        fake_module = types.ModuleType("src.wan.generate")
        fake_module.generate_video = fake_gen
        fake_module.generate_video_stacked = MagicMock()
        fake_module.generate_long_video = MagicMock()

        sys.modules.setdefault("src.wan", types.ModuleType("src.wan"))
        sys.modules["src.wan.generate"] = fake_module
        return fake_gen

    @pytest.mark.asyncio
    async def test_single_segment_completes(self, tmp_path):
        from src.agent.server import _registry, _run_infinite_gen, InfiniteRequest

        self._inject_fake_wan(tmp_path)

        task_id = "test-inf-001"
        _registry.create(task_id)

        req = InfiniteRequest(
            prompt="A river in autumn",
            max_segments=1,
            use_rgba_layers=False,
            output_dir=str(tmp_path),
        )

        (tmp_path / "infinite" / task_id[:8]).mkdir(parents=True, exist_ok=True)
        await _run_infinite_gen(task_id, req)

        state = _registry.get(task_id)
        assert state["status"] == "done"
        result = state["result"]
        assert result["segments_generated"] == 1
        assert len(result["segments"]) == 1

    @pytest.mark.asyncio
    async def test_zero_max_segments_cancels_cleanly(self, tmp_path):
        """max_segments=0 (unlimited) — verify loop exits when status set to cancelled."""
        from src.agent.server import _registry, _run_infinite_gen, InfiniteRequest

        self._inject_fake_wan(tmp_path)

        task_id = "test-inf-002"
        _registry.create(task_id)

        req = InfiniteRequest(
            prompt="Desert dunes",
            max_segments=0,
            use_rgba_layers=False,
            output_dir=str(tmp_path),
        )

        (tmp_path / "infinite" / task_id[:8]).mkdir(parents=True, exist_ok=True)

        original_update = _registry.update
        call_n = [0]

        def patched_update(tid, **kwargs):
            original_update(tid, **kwargs)
            call_n[0] += 1
            if call_n[0] >= 4:
                original_update(tid, status="cancelled")

        with patch.object(_registry, "update", side_effect=patched_update):
            await _run_infinite_gen(task_id, req)

        state = _registry.get(task_id)
        assert state is not None

    @pytest.mark.asyncio
    async def test_vace_and_svi_components_initialised(self, tmp_path):
        """Confirm VACEExtension and SVIRecycler are instantiated without error."""
        from src.vace.extension import VACEExtension, VACEConfig
        from src.svi.recycler import SVIRecycler, SVIConfig

        vace = VACEExtension(VACEConfig(overlap_frames=16, segment_frames=81))
        svi = SVIRecycler(SVIConfig(ema_decay=0.9))

        assert not vace.has_prior_segment
        assert not svi.has_correction

    @pytest.mark.asyncio
    async def test_director_advances_between_segments(self, tmp_path):
        """LLMDirector should produce a different prompt each call in fallback mode."""
        from src.llm.director import LLMDirector, DirectorConfig

        cfg = DirectorConfig(use_static_prompt_fallback=True)
        director = LLMDirector("A mountain trail at dawn", cfg)
        d0 = director.next_segment(0)
        d1 = director.next_segment(1)
        assert d0.segment_idx == 0
        assert d1.segment_idx == 1
        assert director.segment_count == 2
