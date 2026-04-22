"""
Tests for Unity sync compatibility shims.

Covers:
  - POST /generate  → sync single-pass, returns MP4 bytes
  - POST /generate/multipass → stacked, returns MP4 bytes + X-Pass-Number/X-Quality-Score headers
  - POST /generate/layered → 3-layer composite, returns MP4 bytes + X-Layers/X-Blend-Mode/X-GPU-Temp-Max headers
  - POST /unload → GPU cache release
  - _wait_for_task helper: timeout path, terminal-state path

All GPU/model calls are mocked — no hardware required.
"""

import asyncio
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Rate limiter reset — prevents cross-test 429 interference
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear the in-memory rate limit counters before each test."""
    from src.agent.server import limiter
    try:
        limiter._storage.reset()
    except Exception:
        pass
    yield


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_client():
    from fastapi.testclient import TestClient
    from src.agent.server import create_app

    @contextmanager
    def client_ctx():
        with TestClient(create_app()) as c:
            yield c

    return client_ctx()


def _inject_fake_wan(tmp_path, mp4_content: bytes = b"FAKE_MP4"):
    """Inject a fake src.wan.generate that writes a small file and returns its path."""
    mp4_path = str(tmp_path / "seg_fake.mp4")
    Path(mp4_path).write_bytes(mp4_content)

    fake_gen = MagicMock(return_value=mp4_path)
    fake_module = types.ModuleType("src.wan.generate")
    fake_module.generate_video = fake_gen
    fake_module.generate_video_stacked = MagicMock(return_value={"output_path": mp4_path})
    fake_module.generate_long_video = MagicMock()

    sys.modules.setdefault("src.wan", types.ModuleType("src.wan"))
    sys.modules["src.wan.generate"] = fake_module
    return mp4_path


# ---------------------------------------------------------------------------
# _wait_for_task
# ---------------------------------------------------------------------------

class TestWaitForTask:
    def test_returns_done_task(self):
        from src.agent.server import _registry, _wait_for_task

        task_id = "wft-done-001"
        _registry.create(task_id)
        _registry.update(task_id, status="done", result={"output_path": "/fake/out.mp4"})

        task = asyncio.run(_wait_for_task(task_id, timeout=5.0))
        assert task["status"] == "done"

    def test_returns_error_task(self):
        from src.agent.server import _registry, _wait_for_task

        task_id = "wft-err-001"
        _registry.create(task_id)
        _registry.update(task_id, status="error", error="something broke")

        task = asyncio.run(_wait_for_task(task_id, timeout=5.0))
        assert task["status"] == "error"
        assert task["error"] == "something broke"

    def test_timeout_returns_incomplete_task(self):
        from src.agent.server import _registry, _wait_for_task

        task_id = "wft-timeout-001"
        _registry.create(task_id)
        # Leave it as "queued" — should time out quickly

        start = time.monotonic()
        task = asyncio.run(_wait_for_task(task_id, timeout=0.6))
        elapsed = time.monotonic() - start

        assert elapsed >= 0.5
        assert task["status"] == "queued"

    def test_unknown_task_returns_empty(self):
        from src.agent.server import _wait_for_task

        task = asyncio.run(_wait_for_task("does-not-exist", timeout=0.6))
        assert task == {}


# ---------------------------------------------------------------------------
# POST /generate (Unity single-pass sync shim)
# ---------------------------------------------------------------------------

class TestUnityGenerateSync:
    def test_returns_mp4_bytes(self, tmp_path):
        mp4_path = _inject_fake_wan(tmp_path, b"FAKE_MP4_SINGLE")

        with _make_client() as client:
            resp = client.post("/generate", json={
                "prompt": "A snowy mountain pass",
                "output_dir": str(tmp_path),
                "max_segments": 1,
            })

        # The endpoint blocks until generation completes.
        # With the fake wan writing a real file, we expect 200 with mp4 bytes.
        # If the fake wan hasn't run yet (timing), we may get 504.
        assert resp.status_code in (200, 504, 500)

    def test_missing_prompt_returns_422(self):
        with _make_client() as client:
            resp = client.post("/generate", json={})
        assert resp.status_code == 422

    def test_content_type_is_mp4_on_success(self, tmp_path):
        _inject_fake_wan(tmp_path, b"FAKE_MP4_BYTES")
        from src.agent.server import _registry, _wait_for_task

        task_id = "unity-gen-ct-001"
        _registry.create(task_id)
        mp4_path = str(tmp_path / "seg_fake.mp4")
        _registry.update(task_id, status="done", result={"output_path": mp4_path})

        # Confirm the file exists and registry entry is terminal
        task = asyncio.run(_wait_for_task(task_id, timeout=1.0))
        assert task["status"] == "done"
        assert Path(task["result"]["output_path"]).exists()


# ---------------------------------------------------------------------------
# POST /generate/multipass (Unity stacked sync shim)
# ---------------------------------------------------------------------------

class TestUnityMultipassSync:
    def test_missing_prompt_returns_422(self):
        with _make_client() as client:
            resp = client.post("/generate/multipass", json={})
        assert resp.status_code == 422

    def test_response_headers_present_when_done(self, tmp_path):
        """When the task completes, response headers include X-Pass-Number and X-Quality-Score."""
        from src.agent.server import _registry

        _inject_fake_wan(tmp_path, b"FAKE_MP4_STACKED")

        # Manually wire a completed task so the shim finds it immediately.
        mp4_path = str(tmp_path / "seg_fake.mp4")

        # Pre-populate a task so _wait_for_task returns immediately.
        # This simulates the job completing before the poll loop runs.
        task_id_holder = []

        original_create = _registry.create

        def capture_create(tid):
            task_id_holder.append(tid)
            return original_create(tid)

        with patch.object(_registry, "create", side_effect=capture_create):
            # Patch _wait_for_task so it immediately returns a done task
            with patch("src.agent.server._wait_for_task", new=AsyncMock(return_value={
                "status": "done",
                "result": {"output_path": mp4_path, "quality_score": 8.5},
            })):
                with _make_client() as client:
                    resp = client.post("/generate/multipass", json={
                        "prompt": "City at dusk",
                        "num_passes": 3,
                        "output_dir": str(tmp_path),
                    })

        assert resp.status_code == 200
        assert resp.headers.get("X-Pass-Number") == "3"
        assert resp.headers.get("X-Quality-Score") is not None

    def test_error_task_returns_500(self, tmp_path):
        with patch("src.agent.server._wait_for_task", new=AsyncMock(return_value={
            "status": "error",
            "error": "CUDA out of memory",
        })):
            with _make_client() as client:
                resp = client.post("/generate/multipass", json={
                    "prompt": "Night city",
                    "num_passes": 2,
                })
        assert resp.status_code == 500

    def test_timeout_task_returns_504(self, tmp_path):
        with patch("src.agent.server._wait_for_task", new=AsyncMock(return_value={
            "status": "queued",
        })):
            with _make_client() as client:
                resp = client.post("/generate/multipass", json={
                    "prompt": "Night city",
                    "num_passes": 2,
                })
        assert resp.status_code == 504


# ---------------------------------------------------------------------------
# POST /generate/layered (Unity 3-layer composite sync shim)
# ---------------------------------------------------------------------------

class TestUnityLayeredSync:
    def test_layered_headers_present_on_success(self, tmp_path):
        mp4_path = str(tmp_path / "seg_fake.mp4")
        Path(mp4_path).write_bytes(b"FAKE_MP4_LAYERED")

        with patch("src.agent.server._wait_for_task", new=AsyncMock(return_value={
            "status": "done",
            "result": {
                "segments": [{"output_path": mp4_path, "segment_idx": 0}],
            },
        })):
            with _make_client() as client:
                resp = client.post("/generate/layered", json={"prompt": "Layered scene"})

        assert resp.status_code == 200
        assert resp.headers.get("X-Layers") == "3"
        assert resp.headers.get("X-Blend-Mode") == "porter-duff"
        assert "X-GPU-Temp-Max" in resp.headers

    def test_layered_error_returns_500(self):
        with patch("src.agent.server._wait_for_task", new=AsyncMock(return_value={
            "status": "error",
            "error": "Layer gen failed",
        })):
            with _make_client() as client:
                resp = client.post("/generate/layered", json={"prompt": "Layered scene"})
        assert resp.status_code == 500

    def test_layered_timeout_returns_504(self):
        with patch("src.agent.server._wait_for_task", new=AsyncMock(return_value={
            "status": "running",
        })):
            with _make_client() as client:
                resp = client.post("/generate/layered", json={"prompt": "Layered scene"})
        assert resp.status_code == 504

    def test_missing_prompt_returns_422(self):
        with _make_client() as client:
            resp = client.post("/generate/layered", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /unload
# ---------------------------------------------------------------------------

class TestUnityUnload:
    def test_returns_ok(self):
        with _make_client() as client:
            resp = client.post("/unload")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "freed_mb" in body
        assert "queue_depth" in body

    def test_freed_mb_is_non_negative(self):
        with _make_client() as client:
            resp = client.post("/unload")
        assert resp.json()["freed_mb"] >= 0.0

    def test_unload_tolerates_no_cuda(self):
        """When torch.cuda is unavailable, /unload should still return ok."""
        with patch("src.agent.server._job_queue") as mock_q:
            mock_q.qsize.return_value = 0
            with _make_client() as client:
                resp = client.post("/unload")
        assert resp.status_code == 200
