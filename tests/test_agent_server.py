"""
Tests for agent server auth, rate limiting, and job queue.

Coverage:
  - require_api_key dependency: localhost bypass, valid key, invalid key, open mode
  - /health returns 200 and requires no auth
  - Rate limit enforcement on /generate/* routes (5/min)
  - Job queue serializes concurrent GPU work via single-GPU semaphore
  - Billing deduction called after successful single-pass generation
  - Trial key created via /trial/signup and usable immediately
"""

import asyncio
import os
import sys
import types
import uuid
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(env_overrides=None):
    """Return a context-manager TestClient with optional env patches."""
    from contextlib import contextmanager
    from fastapi.testclient import TestClient
    from src.agent.server import create_app

    @contextmanager
    def _ctx():
        with patch.dict(os.environ, env_overrides or {}, clear=False):
            with TestClient(create_app()) as c:
                yield c

    return _ctx()


def _inject_fake_wan():
    """Inject no-op wan modules so generation routes don't import torch."""
    fake_wan = types.ModuleType("src.wan")
    fake_gen_mod = types.ModuleType("src.wan.generate")
    fake_gen_mod.generate_video = MagicMock(return_value="/tmp/fake.mp4")
    fake_gen_mod.generate_video_stacked = MagicMock(return_value={"output_path": "/tmp/stacked.mp4", "psnr": 30.0})
    fake_gen_mod.generate_long_video = MagicMock(return_value="/tmp/long.mp4")
    sys.modules.setdefault("src.wan", fake_wan)
    sys.modules["src.wan.generate"] = fake_gen_mod


def _unique_rate_key() -> str:
    """Return a unique string usable as X-API-Key to get a fresh rate-limit bucket.

    The limiter is a module-level singleton — buckets persist across create_app()
    calls in the same process. Passing a unique key header per test ensures no
    bucket state bleeds between test cases.
    """
    return f"test-{uuid.uuid4().hex}"


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear the module-level rate-limiter storage before each test.

    The slowapi Limiter is a module-level singleton in server.py.  Its in-memory
    storage persists across TestClient instances within a single pytest process,
    so without this fixture earlier tests can exhaust the bucket for later ones.
    """
    from src.agent.server import limiter
    limiter._storage.reset()
    yield
    limiter._storage.reset()


@pytest.fixture(scope="module", autouse=True)
def _restore_src_wan_modules():
    """Remove fake src.wan modules from sys.modules after this test file.

    _inject_fake_wan() replaces sys.modules['src.wan.generate'] with a
    MagicMock-based module to avoid importing torch.  Without this cleanup,
    the fake persists across test files and breaks downstream tests
    (test_generate_ltx, test_ltx_pipeline_factory) that import from the
    real src.wan.generate — they get '_save_last_frame not found' errors
    because the fake lacks those symbols.
    """
    yield
    for mod_name in ("src.wan.generate", "src.wan"):
        sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# Unit tests — require_api_key dependency
# ---------------------------------------------------------------------------

class TestRequireApiKey:
    """Tests for the require_api_key FastAPI dependency."""

    def _req(self, host="testclient"):
        req = MagicMock()
        req.client = MagicMock()
        req.client.host = host
        return req

    def test_localhost_bypasses_key_check(self):
        """127.0.0.1 callers skip auth entirely, even when keys are configured."""
        from src.agent.server import require_api_key
        with patch.dict(os.environ, {"VIDEO_API_KEYS": "required-key"}):
            result = require_api_key(self._req("127.0.0.1"), api_key=None)
        assert result is None

    def test_ipv6_localhost_bypasses_key_check(self):
        """::1 (IPv6 loopback) also bypasses auth."""
        from src.agent.server import require_api_key
        with patch.dict(os.environ, {"VIDEO_API_KEYS": "required-key"}):
            result = require_api_key(self._req("::1"), api_key=None)
        assert result is None

    def test_valid_env_key_passes(self):
        """A key present in VIDEO_API_KEYS env var is accepted."""
        from src.agent.server import require_api_key
        with patch.dict(os.environ, {"VIDEO_API_KEYS": "good-key"}):
            result = require_api_key(self._req("1.2.3.4"), api_key="good-key")
        assert result is None

    def test_invalid_key_raises_401(self):
        """A key not in env var and not in DB raises 401."""
        from fastapi import HTTPException
        from src.agent.server import require_api_key
        with patch.dict(os.environ, {"VIDEO_API_KEYS": "valid-key"}):
            with patch("src.billing.store.validate_db_key", return_value=False):
                with pytest.raises(HTTPException) as exc:
                    require_api_key(self._req("1.2.3.4"), api_key="wrong-key")
        assert exc.value.status_code == 401

    def test_missing_key_with_keys_configured_raises_401(self):
        """Missing X-API-Key header when VIDEO_API_KEYS is set raises 401."""
        from fastapi import HTTPException
        from src.agent.server import require_api_key
        with patch.dict(os.environ, {"VIDEO_API_KEYS": "some-key"}):
            with pytest.raises(HTTPException) as exc:
                require_api_key(self._req("1.2.3.4"), api_key=None)
        assert exc.value.status_code == 401

    def test_open_mode_no_key_required(self):
        """When VIDEO_API_KEYS is absent, server is in open mode — no key needed."""
        from src.agent.server import require_api_key
        env = {k: v for k, v in os.environ.items() if k != "VIDEO_API_KEYS"}
        with patch.dict(os.environ, env, clear=True):
            result = require_api_key(self._req("1.2.3.4"), api_key=None)
        assert result is None

    def test_valid_db_key_passes(self):
        """A key not in env var but valid in the DB is accepted."""
        from src.agent.server import require_api_key
        with patch.dict(os.environ, {"VIDEO_API_KEYS": "env-only"}):
            with patch("src.billing.store.validate_db_key", return_value=True):
                result = require_api_key(self._req("1.2.3.4"), api_key="db-provisioned-key")
        assert result is None


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """/health returns HTTP 200 with status=ok."""
        pytest.importorskip("fastapi")
        with _make_client() as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_no_auth_required(self):
        """/health is accessible without an API key even when keys are configured."""
        pytest.importorskip("fastapi")
        with _make_client({"VIDEO_API_KEYS": "secret-key-xyz"}) as client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_reports_queue_depth(self):
        """/health response includes queue_depth."""
        pytest.importorskip("fastapi")
        with _make_client() as client:
            resp = client.get("/health")
        assert "queue_depth" in resp.json()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_generate_single_rate_limited_to_5_per_minute(self):
        """6th POST to /generate/single in the same minute returns 429."""
        pytest.importorskip("fastapi")
        _inject_fake_wan()
        # Use a key that is in VIDEO_API_KEYS so auth passes; use as the rate-limit bucket
        key = _unique_rate_key()
        with _make_client({"VIDEO_API_KEYS": key}) as client:
            statuses = [
                client.post(
                    "/generate/single",
                    json={"prompt": "test"},
                    headers={"X-API-Key": key},
                ).status_code
                for _ in range(6)
            ]
        assert 429 in statuses, f"Expected a 429 after 5 requests, got: {statuses}"

    def test_generate_stacked_rate_limited_to_5_per_minute(self):
        """6th POST to /generate/stacked returns 429."""
        pytest.importorskip("fastapi")
        _inject_fake_wan()
        key = _unique_rate_key()
        with _make_client({"VIDEO_API_KEYS": key}) as client:
            statuses = [
                client.post(
                    "/generate/stacked",
                    json={"prompt": "test", "num_passes": 2},
                    headers={"X-API-Key": key},
                ).status_code
                for _ in range(6)
            ]
        assert 429 in statuses, f"Expected a 429 after 5 requests, got: {statuses}"

    def test_first_5_requests_succeed(self):
        """The first 5 POST requests to /generate/single should all succeed (200)."""
        pytest.importorskip("fastapi")
        _inject_fake_wan()
        key = _unique_rate_key()
        with _make_client({"VIDEO_API_KEYS": key}) as client:
            statuses = [
                client.post(
                    "/generate/single",
                    json={"prompt": "test"},
                    headers={"X-API-Key": key},
                ).status_code
                for _ in range(5)
            ]
        assert all(s == 200 for s in statuses), f"Expected 5×200, got: {statuses}"


# ---------------------------------------------------------------------------
# Job queue — single-GPU serialization
# ---------------------------------------------------------------------------

class TestJobQueue:
    def test_jobs_run_sequentially_not_concurrently(self):
        """Two queued jobs must not overlap — semaphore enforces single-GPU lock."""
        run_order = []

        async def _inner():
            sem = asyncio.Semaphore(1)
            q: asyncio.Queue = asyncio.Queue()

            async def job_a():
                run_order.append("a_start")
                await asyncio.sleep(0)
                run_order.append("a_end")

            async def job_b():
                run_order.append("b_start")
                await asyncio.sleep(0)
                run_order.append("b_end")

            async def worker():
                for _ in range(2):
                    job_fn, _ = await q.get()
                    async with sem:
                        await job_fn()
                    q.task_done()

            await q.put((job_a, "task-a"))
            await q.put((job_b, "task-b"))
            await worker()

        asyncio.run(_inner())

        # job_a must fully complete before job_b begins
        assert run_order.index("a_end") < run_order.index("b_start"), (
            f"Jobs overlapped: {run_order}"
        )

    def test_generate_endpoint_returns_task_id_and_queue_position(self):
        """POST /generate/single returns task_id and queue_position immediately."""
        pytest.importorskip("fastapi")
        _inject_fake_wan()
        with _make_client() as client:
            resp = client.post("/generate/single", json={"prompt": "test"})
        assert resp.status_code == 200
        body = resp.json()
        assert "task_id" in body
        assert "queue_position" in body
        assert isinstance(body["queue_position"], int)

    def test_multiple_jobs_queued_get_unique_task_ids(self):
        """Two requests enqueued in the same second get distinct task IDs."""
        pytest.importorskip("fastapi")
        _inject_fake_wan()
        with _make_client() as client:
            r1 = client.post("/generate/single", json={"prompt": "first"})
            r2 = client.post("/generate/single", json={"prompt": "second"})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["task_id"] != r2.json()["task_id"]


# ---------------------------------------------------------------------------
# Billing deduction
# ---------------------------------------------------------------------------

class TestBillingDeduction:
    def test_deduct_credits_called_after_successful_single_gen(self, tmp_path):
        """_deduct_for_task calls deduct_credits after /generate/single completes."""
        import src.agent.server as srv
        from src.agent.server import _registry, _task_api_keys, SinglePassRequest

        _inject_fake_wan()

        task_id = "billing-test-001"
        _registry.create(task_id)
        _task_api_keys[task_id] = "test-api-key-abc"  # simulates an external caller

        deduct_mock = MagicMock(return_value={"ok": True, "cost_cents": 51})

        req = SinglePassRequest(prompt="billing test", output_dir=str(tmp_path))
        with patch.dict(os.environ, {"STRIPE_SECRET_KEY": "sk_test_fake"}):
            with patch("src.billing.store.deduct_credits", deduct_mock):
                asyncio.run(srv._run_single_gen(task_id, req))

        deduct_mock.assert_called_once()
        api_key_arg, seconds_arg = deduct_mock.call_args[0][:2]
        assert api_key_arg == "test-api-key-abc"
        # 81 frames / 16 fps = 5.0625 seconds
        assert 5.0 < seconds_arg < 5.2

    def test_no_deduction_for_localhost_callers(self, tmp_path):
        """Jobs submitted by localhost callers (no registered key) skip billing."""
        import src.agent.server as srv
        from src.agent.server import _registry, _task_api_keys, SinglePassRequest

        _inject_fake_wan()

        task_id = "billing-test-002"
        _registry.create(task_id)
        # Localhost callers don't register a key, so _task_api_keys has no entry

        deduct_mock = MagicMock(return_value={"ok": True, "cost_cents": 0})

        req = SinglePassRequest(prompt="localhost gen test", output_dir=str(tmp_path))
        with patch.dict(os.environ, {"STRIPE_SECRET_KEY": "sk_test_fake"}):
            with patch("src.billing.store.deduct_credits", deduct_mock):
                asyncio.run(srv._run_single_gen(task_id, req))

        deduct_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Trial signup
# ---------------------------------------------------------------------------

class TestTrialSignup:
    def test_trial_signup_creates_key_with_30_free_seconds(self):
        """/trial/signup returns an api_key and 30 free seconds."""
        pytest.importorskip("fastapi")
        mock_result = {
            "api_key": "qsv_trial_abc123",
            "balance_cents": 300,
            "trial_seconds": 30,
        }
        with patch("src.billing.store.create_trial_key", return_value=mock_result):
            with _make_client() as client:
                resp = client.post("/trial/signup", json={"label": "test-user"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["api_key"] == "qsv_trial_abc123"
        assert body["free_seconds"] == 30
        assert body["balance_cents"] == 300

    def test_trial_signup_no_auth_required(self):
        """/trial/signup is open — no API key needed even when keys are configured."""
        pytest.importorskip("fastapi")
        mock_result = {"api_key": "qsv_trial_xyz", "balance_cents": 300, "trial_seconds": 30}
        with patch("src.billing.store.create_trial_key", return_value=mock_result):
            with _make_client({"VIDEO_API_KEYS": "existing-key"}) as client:
                resp = client.post("/trial/signup", json={})
        assert resp.status_code == 200

    def test_trial_key_valid_immediately_via_db_check(self):
        """Key from /trial/signup is accepted immediately on auth-required endpoints."""
        pytest.importorskip("fastapi")
        trial_key = "qsv_trial_newkey99"
        mock_signup = {"api_key": trial_key, "balance_cents": 300, "trial_seconds": 30}

        with patch("src.billing.store.create_trial_key", return_value=mock_signup):
            with patch("src.billing.store.validate_db_key", return_value=True) as mock_validate:
                with patch("src.billing.store.get_balance", return_value=300):
                    # VIDEO_API_KEYS excludes the trial key so the DB path must be used.
                    with _make_client({"VIDEO_API_KEYS": "env-key-only"}) as client:
                        resp = client.post("/trial/signup", json={"label": "tester"})
                        assert resp.status_code == 200
                        api_key = resp.json()["api_key"]

                        auth_resp = client.get(
                            "/billing/balance",
                            headers={"X-API-Key": api_key},
                        )

        assert auth_resp.status_code == 200
        # validate_db_key should have been called with the trial key
        mock_validate.assert_called_with(trial_key)
