"""
Unit tests for WanPipelineFactory — covers BnB and GGUF engine paths.
No GPU required: we test constructor logic and routing without building real pipelines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("diffusers", reason="diffusers not installed")


class TestWanPipelineFactoryBnb:
    def test_default_engine_is_bnb(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory()
        assert f.engine == "bnb"

    def test_gguf_path_not_required_for_bnb(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(engine="bnb")
        assert f.gguf_model_path is None

    def test_bnb_stores_params(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", engine="bnb")
        assert f.model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        assert f.engine == "bnb"

    def test_call_routes_to_bnb_pipeline(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        from src.quant.config import QuantConfig
        f = WanPipelineFactory(engine="bnb")
        with patch.object(f, "_build_pipeline", return_value=MagicMock()) as mock_build:
            cfg = QuantConfig()
            f(cfg)
            mock_build.assert_called_once_with(cfg)


class TestBnbTransformerCpuLoad:
    """Verify BnB transformer is loaded with device_map='cpu' to prevent OOM on 14B models."""

    def test_bnb_transformer_loads_to_cpu(self):
        """device_map='cpu' must be passed so shards quantize on CPU without double-buffering VRAM."""
        import diffusers
        from src.wan.pipeline_factory import WanPipelineFactory
        from src.quant.config import QuantConfig

        mock_bnb_config = MagicMock()
        mock_transformer = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.enable_model_cpu_offload = MagicMock()
        mock_pipe.enable_vae_slicing = MagicMock()

        mock_wan_transformer_cls = MagicMock()
        mock_wan_transformer_cls.from_pretrained.return_value = mock_transformer

        # WanTransformer3DModel is imported locally inside _build_pipeline, so patch
        # the attribute on the diffusers module object itself.
        with patch.object(diffusers, "WanTransformer3DModel", mock_wan_transformer_cls), \
             patch("src.wan.pipeline_factory.UMT5EncoderModel") as mock_te_cls, \
             patch("src.wan.pipeline_factory.AutoencoderKLWan") as mock_vae_cls, \
             patch("src.wan.pipeline_factory.WanPipeline") as mock_pipeline_cls:

            mock_te_cls.from_pretrained.return_value = MagicMock()
            mock_vae_cls.from_pretrained.return_value = MagicMock()
            mock_pipeline_cls.from_pretrained.return_value = mock_pipe

            factory = WanPipelineFactory(engine="bnb")
            cfg = QuantConfig()
            with patch.object(cfg, "to_bnb_config", return_value=mock_bnb_config):
                factory(cfg)

        call_kwargs = mock_wan_transformer_cls.from_pretrained.call_args
        assert call_kwargs is not None, "WanTransformer3DModel.from_pretrained was not called"
        kwargs = call_kwargs.kwargs
        assert kwargs.get("device_map") == "cpu", (
            f"Expected device_map='cpu' to prevent OOM during shard loading, got: {kwargs.get('device_map')!r}"
        )


class TestWanPipelineFactoryGGUF:
    def test_gguf_requires_path(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        with pytest.raises(ValueError, match="gguf_model_path"):
            WanPipelineFactory(engine="gguf")

    def test_gguf_stores_path(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(engine="gguf", gguf_model_path="/tmp/model.gguf")
        assert f.gguf_model_path == "/tmp/model.gguf"
        assert f.engine == "gguf"

    def test_gguf_default_compute_dtype(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(engine="gguf", gguf_model_path="/tmp/model.gguf")
        assert f.gguf_compute_dtype == "bfloat16"

    def test_gguf_custom_compute_dtype(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(engine="gguf", gguf_model_path="/tmp/model.gguf", gguf_compute_dtype="float16")
        assert f.gguf_compute_dtype == "float16"

    def test_call_routes_to_gguf_pipeline(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(engine="gguf", gguf_model_path="/tmp/model.gguf")
        with patch.object(f, "_build_gguf_pipeline", return_value=MagicMock()) as mock_build:
            f(None)
            mock_build.assert_called_once()

    def test_gguf_missing_file_raises_file_not_found(self, tmp_path):
        """_build_gguf_pipeline should raise FileNotFoundError if the .gguf file is missing."""
        from src.wan.pipeline_factory import WanPipelineFactory
        missing = str(tmp_path / "nonexistent_model.gguf")
        f = WanPipelineFactory(engine="gguf", gguf_model_path=missing)
        with pytest.raises(FileNotFoundError, match="not found"):
            f._build_gguf_pipeline()


class TestApplyMemoryOpts:
    def test_vae_slicing_applied(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(enable_vae_slicing=True)
        mock_pipe = MagicMock()
        mock_pipe.enable_vae_slicing = MagicMock()
        f._apply_memory_opts(mock_pipe)
        mock_pipe.enable_vae_slicing.assert_called_once()

    def test_cpu_offload_applied_by_default(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(enable_model_cpu_offload=True, enable_sequential_cpu_offload=False)
        mock_pipe = MagicMock()
        f._apply_memory_opts(mock_pipe)
        mock_pipe.enable_model_cpu_offload.assert_called_once()
        mock_pipe.enable_sequential_cpu_offload.assert_not_called()

    def test_sequential_offload_takes_priority(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(enable_model_cpu_offload=True, enable_sequential_cpu_offload=True)
        mock_pipe = MagicMock()
        f._apply_memory_opts(mock_pipe)
        mock_pipe.enable_sequential_cpu_offload.assert_called_once()
        mock_pipe.enable_model_cpu_offload.assert_not_called()

    def test_no_offload_calls_to_cuda(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory(enable_model_cpu_offload=False, enable_sequential_cpu_offload=False)
        mock_pipe = MagicMock()
        f._apply_memory_opts(mock_pipe)
        mock_pipe.to.assert_called_once_with("cuda")

    def test_returns_pipe(self):
        from src.wan.pipeline_factory import WanPipelineFactory
        f = WanPipelineFactory()
        mock_pipe = MagicMock()
        result = f._apply_memory_opts(mock_pipe)
        assert result is mock_pipe


class TestGGUFConstants:
    def test_gguf_hf_repo_defined(self):
        from src.wan.pipeline_factory import GGUF_HF_REPO, GGUF_Q4_0_FILENAME, GGUF_Q3_KM_FILENAME
        assert "city96" in GGUF_HF_REPO
        assert "Q4_0" in GGUF_Q4_0_FILENAME
        assert "Q3_K_M" in GGUF_Q3_KM_FILENAME
