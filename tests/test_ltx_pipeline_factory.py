"""Unit tests for LTX pipeline factory (mocked — no GPU required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_default_image_conditioning_is_false():
    from src.wan.ltx_pipeline_factory import LTXPipelineFactory
    factory = LTXPipelineFactory()
    assert factory.image_conditioning is False


def test_i2v_flag_stored():
    from src.wan.ltx_pipeline_factory import LTXPipelineFactory
    factory = LTXPipelineFactory(image_conditioning=True)
    assert factory.image_conditioning is True


def test_default_model_id_is_lightricks():
    from src.wan.ltx_pipeline_factory import LTXPipelineFactory, LTX_MODEL_ID
    factory = LTXPipelineFactory()
    assert factory.model_id == LTX_MODEL_ID
    assert "Lightricks" in LTX_MODEL_ID


def test_vae_slicing_enabled_by_default():
    from src.wan.ltx_pipeline_factory import LTXPipelineFactory
    factory = LTXPipelineFactory()
    assert factory.enable_vae_slicing is True


def test_cpu_offload_enabled_by_default():
    from src.wan.ltx_pipeline_factory import LTXPipelineFactory
    factory = LTXPipelineFactory()
    assert factory.enable_model_cpu_offload is True


def test_vae_tiling_enabled_by_default():
    """VAE tiling must be on by default — required for ≥768px on 12GB VRAM."""
    from src.wan.ltx_pipeline_factory import LTXPipelineFactory
    factory = LTXPipelineFactory()
    assert factory.enable_vae_tiling is True


# ---------------------------------------------------------------------------
# Pipeline cache tests — verify the module-level cache prevents repeat builds
# ---------------------------------------------------------------------------


def test_cache_returns_same_pipeline_on_repeat_call(monkeypatch):
    """get_cached_pipeline must return the identical object on repeat calls."""
    from src.wan import ltx_pipeline_factory as mod

    mod.clear_pipeline_cache()

    build_calls = []

    class FakePipe:
        pass

    def fake_build(self):
        build_calls.append((self.model_id, self.image_conditioning))
        return FakePipe()

    monkeypatch.setattr(mod.LTXPipelineFactory, "build", fake_build)

    p1 = mod.get_cached_pipeline()
    p2 = mod.get_cached_pipeline()
    p3 = mod.get_cached_pipeline()

    assert p1 is p2 is p3  # Same identity — no rebuild
    assert len(build_calls) == 1  # build() only called once

    mod.clear_pipeline_cache()


def test_cache_separates_t2v_and_i2v(monkeypatch):
    """t2v and i2v pipelines must be cached under separate keys."""
    from src.wan import ltx_pipeline_factory as mod

    mod.clear_pipeline_cache()

    class FakePipe:
        def __init__(self, i2v):
            self.i2v = i2v

    def fake_build(self):
        return FakePipe(self.image_conditioning)

    monkeypatch.setattr(mod.LTXPipelineFactory, "build", fake_build)

    t2v = mod.get_cached_pipeline(image_conditioning=False)
    i2v = mod.get_cached_pipeline(image_conditioning=True)
    t2v_again = mod.get_cached_pipeline(image_conditioning=False)

    assert t2v is not i2v       # Different keys, different objects
    assert t2v is t2v_again     # Same key, same object
    assert t2v.i2v is False
    assert i2v.i2v is True

    mod.clear_pipeline_cache()


def test_clear_cache_forces_rebuild(monkeypatch):
    """clear_pipeline_cache() must force get_cached_pipeline() to rebuild."""
    from src.wan import ltx_pipeline_factory as mod

    mod.clear_pipeline_cache()

    build_count = [0]

    class FakePipe:
        pass

    def fake_build(self):
        build_count[0] += 1
        return FakePipe()

    monkeypatch.setattr(mod.LTXPipelineFactory, "build", fake_build)

    mod.get_cached_pipeline()
    mod.get_cached_pipeline()
    assert build_count[0] == 1

    mod.clear_pipeline_cache()

    mod.get_cached_pipeline()
    assert build_count[0] == 2  # Rebuilt after clear

    mod.clear_pipeline_cache()


def test_ltx_defaults_are_ltx_not_wan():
    """LTX-specific defaults should differ from Wan defaults."""
    from src.wan.ltx_pipeline_factory import (
        LTX_DEFAULT_NEGATIVE_PROMPT,
        LTX_DEFAULT_GUIDANCE_SCALE,
        LTX_DEFAULT_NUM_INFERENCE_STEPS,
        LTX_DEFAULT_FPS,
        LTX_DEFAULT_NUM_FRAMES,
        LTX_DEFAULT_WIDTH,
        LTX_DEFAULT_HEIGHT,
    )
    # English prompt (Wan default is Chinese)
    assert "worst quality" in LTX_DEFAULT_NEGATIVE_PROMPT.lower()
    assert "色调" not in LTX_DEFAULT_NEGATIVE_PROMPT
    # LTX has its own tuned params — match scripts/generate_ltx.py
    assert LTX_DEFAULT_GUIDANCE_SCALE == 3.0
    assert LTX_DEFAULT_NUM_INFERENCE_STEPS == 50
    assert LTX_DEFAULT_FPS == 25
    assert LTX_DEFAULT_WIDTH == 768
    assert LTX_DEFAULT_HEIGHT == 512
    # LTX VAE temporal compression = 8, so num_frames must be 8k+1
    assert (LTX_DEFAULT_NUM_FRAMES - 1) % 8 == 0


def test_wan_chinese_negative_prompt_is_swapped_for_ltx():
    """When generate_video(engine='ltx') inherits the Wan Chinese negative
    prompt, _generate_video_ltx should swap it for the LTX English default."""
    from src.wan.generate import _WAN_DEFAULT_NEGATIVE_PROMPT_SENTINEL
    from src.wan.ltx_pipeline_factory import LTX_DEFAULT_NEGATIVE_PROMPT

    wan_default_negative = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止"
    )
    assert _WAN_DEFAULT_NEGATIVE_PROMPT_SENTINEL in wan_default_negative
    assert _WAN_DEFAULT_NEGATIVE_PROMPT_SENTINEL not in LTX_DEFAULT_NEGATIVE_PROMPT
