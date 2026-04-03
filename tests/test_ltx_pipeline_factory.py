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
