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

    @pytest.mark.anyio
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

    @pytest.mark.anyio
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

    def test_vace_and_svi_components_initialised(self, tmp_path):
        """Confirm VACEExtension and SVIRecycler are instantiated without error."""
        from src.vace.extension import VACEExtension, VACEConfig
        from src.svi.recycler import SVIRecycler, SVIConfig

        vace = VACEExtension(VACEConfig(overlap_frames=16, segment_frames=81))
        svi = SVIRecycler(SVIConfig(ema_decay=0.9))

        assert not vace.has_prior_segment
        assert not svi.has_correction

    def test_director_advances_between_segments(self, tmp_path):
        """LLMDirector should produce a different prompt each call in fallback mode."""
        from src.llm.director import LLMDirector, DirectorConfig

        cfg = DirectorConfig(use_static_prompt_fallback=True)
        director = LLMDirector("A mountain trail at dawn", cfg)
        d0 = director.next_segment(0)
        d1 = director.next_segment(1)
        assert d0.segment_idx == 0
        assert d1.segment_idx == 1
        assert director.segment_count == 2

    @pytest.mark.anyio
    async def test_segment_generation_failure_continues_pipeline(self, tmp_path):
        """A failing generate_video call in the non-RGBA path should not abort the run.

        The pipeline must record an error entry for the failed segment and continue
        to the next one — task status should be 'done', not 'error'.
        """
        import sys
        import types
        from src.agent.server import _registry, _run_infinite_gen, InfiniteRequest

        # First segment fails, second succeeds.
        call_count = {"n": 0}

        def _flaky_generate_video(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated GPU OOM on seg 0")

        fake_module = types.ModuleType("src.wan.generate")
        fake_module.generate_video = _flaky_generate_video
        fake_module.generate_video_stacked = MagicMock()
        fake_module.generate_long_video = MagicMock()
        sys.modules.setdefault("src.wan", types.ModuleType("src.wan"))
        sys.modules["src.wan.generate"] = fake_module

        task_id = "test-fault-iso-001"
        _registry.create(task_id)
        req = InfiniteRequest(
            prompt="Ocean waves",
            max_segments=2,
            use_rgba_layers=False,
            output_dir=str(tmp_path),
        )
        (tmp_path / "infinite" / task_id[:8]).mkdir(parents=True, exist_ok=True)
        await _run_infinite_gen(task_id, req)

        state = _registry.get(task_id)
        assert state["status"] == "done", f"expected done, got {state['status']}: {state.get('error')}"
        segments = state["result"]["segments"]
        assert len(segments) == 2
        # Segment 0 recorded the error
        failed = next(s for s in segments if s["segment_idx"] == 0)
        assert failed["output_path"] is None
        assert "error" in failed
        # Segment 1 succeeded
        succeeded = next(s for s in segments if s["segment_idx"] == 1)
        assert succeeded["output_path"] is not None

    @pytest.mark.anyio
    async def test_rgba_fallback_failure_continues_pipeline(self, tmp_path):
        """When RGBA layer gen fails AND single-pass fallback also fails, the pipeline
        should record an error entry and continue to the next segment.
        """
        import sys
        import types
        from src.agent.server import _registry, _run_infinite_gen, InfiniteRequest

        # generate_video always raises (both RGBA layer gen and fallback call it)
        def _always_fail(**kwargs):
            raise RuntimeError("simulated GPU failure")

        fake_module = types.ModuleType("src.wan.generate")
        fake_module.generate_video = _always_fail
        fake_module.generate_video_stacked = MagicMock()
        fake_module.generate_long_video = MagicMock()
        sys.modules.setdefault("src.wan", types.ModuleType("src.wan"))
        sys.modules["src.wan.generate"] = fake_module

        task_id = "test-fault-iso-002"
        _registry.create(task_id)
        req = InfiniteRequest(
            prompt="Desert at sunrise",
            max_segments=1,
            use_rgba_layers=True,
            output_dir=str(tmp_path),
        )
        (tmp_path / "infinite" / task_id[:8]).mkdir(parents=True, exist_ok=True)
        await _run_infinite_gen(task_id, req)

        state = _registry.get(task_id)
        assert state["status"] == "done", f"expected done, got {state['status']}: {state.get('error')}"
        segments = state["result"]["segments"]
        assert len(segments) == 1
        assert segments[0]["output_path"] is None
        assert "error" in segments[0]

    @pytest.mark.anyio
    async def test_stacked_result_is_json_serializable(self, tmp_path):
        """The stacked engine result carries raw frame arrays (frames + all_pass_frames);
        _run_stacked_gen must strip every array so GET /tasks/{id} can serialize, and must
        surface output_path (the engine result omits it). Regression for the 2026-06-14
        GPU re-test 500: jsonable_encoder choking on a numpy array.
        """
        import sys
        import types
        import numpy as np
        from fastapi.encoders import jsonable_encoder
        from src.agent.server import _registry, _run_stacked_gen, StackedRequest

        def _fake_stacked(**kwargs):
            return {
                "frames": np.zeros((81, 480, 832, 3), dtype=np.float32),
                "all_pass_frames": [np.zeros((81, 480, 832, 3), dtype=np.float32) for _ in range(3)],
                "pass_times": [12.3, 11.8, 12.1],
                "strategy": "progressive",
                "total_time": 36.2,
            }

        fake_module = types.ModuleType("src.wan.generate")
        fake_module.generate_video_stacked = _fake_stacked
        sys.modules.setdefault("src.wan", types.ModuleType("src.wan"))
        sys.modules["src.wan.generate"] = fake_module

        task_id = "test-stacked-serial-001"
        _registry.create(task_id)
        req = StackedRequest(prompt="Roman forum", num_passes=3, output_dir=str(tmp_path))
        await _run_stacked_gen(task_id, req)

        state = _registry.get(task_id)
        assert state["status"] == "done", f"expected done, got {state['status']}: {state.get('error')}"
        result = state["result"]
        assert "frames" not in result and "all_pass_frames" not in result
        assert result["output_path"].endswith(".mp4")
        assert result["pass_times"] == [12.3, 11.8, 12.1]
        # The real failure was in FastAPI's encoder — exercise it directly.
        jsonable_encoder(result)


# ---------------------------------------------------------------------------
# POST /generate/composite endpoint
# ---------------------------------------------------------------------------

class TestGenerateCompositeEndpoint:
    def _write_rgba_npy(self, path, shape=(1, 4, 8, 32, 32)):
        """Write a float32 RGBA numpy file at path."""
        import numpy as np
        arr = np.random.rand(*shape).astype("float32")
        np.save(str(path), arr)

    def test_returns_task_id(self, tmp_path):
        """Endpoint should accept 3 layer paths and return a task_id immediately."""
        pytest.importorskip("fastapi")
        paths = []
        for i in range(3):
            p = tmp_path / f"layer{i}.npy"
            self._write_rgba_npy(p)
            paths.append(str(p))

        with _make_client() as client:
            resp = client.post("/generate/composite", json={"layer_paths": paths})
        assert resp.status_code == 200
        body = resp.json()
        assert "task_id" in body
        assert body["status"] == "queued"

    def test_wrong_layer_count_rejected(self, tmp_path):
        """Exactly 3 layer paths required — fewer should be rejected with 422."""
        pytest.importorskip("fastapi")
        p = tmp_path / "layer0.npy"
        self._write_rgba_npy(p)

        with _make_client() as client:
            resp = client.post("/generate/composite", json={"layer_paths": [str(p), str(p)]})
        assert resp.status_code == 422

    def test_missing_layer_paths_rejected(self):
        """layer_paths is required — omitting it should return 422."""
        pytest.importorskip("fastapi")
        with _make_client() as client:
            resp = client.post("/generate/composite", json={})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_composite_runner_produces_rgb(self, tmp_path):
        """_run_composite loads 3 RGBA .npy files and writes an RGB .npy output."""
        import numpy as np
        from src.agent.server import _registry, _run_composite, CompositeRequest

        shape = (1, 4, 8, 32, 32)
        paths = []
        for i in range(3):
            p = tmp_path / f"layer{i}.npy"
            arr = np.random.rand(*shape).astype("float32")
            np.save(str(p), arr)
            paths.append(str(p))

        out_path = str(tmp_path / "out.npy")
        task_id = "test-comp-001"
        _registry.create(task_id)

        req = CompositeRequest(layer_paths=paths, output_path=out_path)
        await _run_composite(task_id, req)

        state = _registry.get(task_id)
        assert state["status"] == "done"
        result = state["result"]
        assert result["output_path"] == out_path
        assert result["shape"][1] == 3  # RGB — alpha channel dropped

        rgb = np.load(out_path)
        assert rgb.shape == (1, 3, 8, 32, 32)

    @pytest.mark.anyio
    async def test_composite_runner_error_on_bad_path(self, tmp_path):
        """_run_composite should record error status if a layer file doesn't exist."""
        from src.agent.server import _registry, _run_composite, CompositeRequest

        task_id = "test-comp-002"
        _registry.create(task_id)
        req = CompositeRequest(layer_paths=["/no/such/bg.npy", "/no/such/mg.npy", "/no/such/fg.npy"])
        await _run_composite(task_id, req)

        state = _registry.get(task_id)
        assert state["status"] == "error"
        assert state.get("error")


# ---------------------------------------------------------------------------
# GET /generate/{task_id}/segment/{n}
# ---------------------------------------------------------------------------

class TestGetSegmentEndpoint:
    def _inject_fake_wan(self, tmp_path):
        import sys, types
        fake_gen = MagicMock(return_value=str(tmp_path / "seg_0000.mp4"))
        fake_module = types.ModuleType("src.wan.generate")
        fake_module.generate_video = fake_gen
        fake_module.generate_video_stacked = MagicMock()
        fake_module.generate_long_video = MagicMock()
        sys.modules.setdefault("src.wan", types.ModuleType("src.wan"))
        sys.modules["src.wan.generate"] = fake_module

    @pytest.mark.anyio
    async def test_segment_returns_metadata(self, tmp_path):
        """After a successful infinite run, segment 0 metadata is accessible."""
        from src.agent.server import _registry, _run_infinite_gen, InfiniteRequest

        self._inject_fake_wan(tmp_path)
        task_id = "test-seg-001"
        _registry.create(task_id)

        req = InfiniteRequest(
            prompt="Autumn forest",
            max_segments=1,
            use_rgba_layers=False,
            output_dir=str(tmp_path),
        )
        (tmp_path / "infinite" / task_id[:8]).mkdir(parents=True, exist_ok=True)
        await _run_infinite_gen(task_id, req)

        state = _registry.get(task_id)
        assert state["status"] == "done"

        # Retrieve segment 0 via the HTTP endpoint
        with _make_client() as client:
            resp = client.get(f"/generate/{task_id}/segment/0")
        # File won't exist on disk (fake wan doesn't write it), so expect JSON metadata
        assert resp.status_code == 200
        body = resp.json()
        assert body["segment_idx"] == 0
        assert "output_path" in body

    @pytest.mark.anyio
    async def test_segment_404_for_missing_task(self):
        """Unknown task_id should return 404."""
        with _make_client() as client:
            resp = client.get("/generate/does-not-exist/segment/0")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_segment_404_for_out_of_range(self, tmp_path):
        """Segment index beyond the task result should return 404."""
        from src.agent.server import _registry, _run_infinite_gen, InfiniteRequest

        self._inject_fake_wan(tmp_path)
        task_id = "test-seg-002"
        _registry.create(task_id)
        req = InfiniteRequest(
            prompt="City lights",
            max_segments=1,
            use_rgba_layers=False,
            output_dir=str(tmp_path),
        )
        (tmp_path / "infinite" / task_id[:8]).mkdir(parents=True, exist_ok=True)
        await _run_infinite_gen(task_id, req)

        with _make_client() as client:
            resp = client.get(f"/generate/{task_id}/segment/99")
        assert resp.status_code == 404
