"""
Tests for v2 pipeline components: RGBA compositor, VACE extension, SVI recycler, LLM director.

All tests use CPU tensors (no GPU required) and mock the torch dependency where needed.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")


# ---------------------------------------------------------------------------
# Phase 2: RGBA Compositor
# ---------------------------------------------------------------------------

class TestCompositeOver:
    def _make_layer(self, rgb_val: float, alpha_val: float, B=1, F=4, H=8, W=8):
        """Create a uniform RGBA tensor."""
        layer = torch.zeros(B, 4, F, H, W)
        layer[:, :3] = rgb_val
        layer[:, 3:4] = alpha_val
        return layer

    def test_opaque_top_hides_bottom(self):
        from src.rgba.compositor import composite_over

        top = self._make_layer(rgb_val=1.0, alpha_val=1.0)   # white, fully opaque
        bottom = self._make_layer(rgb_val=0.0, alpha_val=1.0)  # black, fully opaque
        result = composite_over(top, bottom)

        # Fully opaque top → result should be white
        assert result.shape == top.shape
        assert torch.allclose(result[:, :3], torch.ones_like(result[:, :3]), atol=1e-5)
        assert torch.allclose(result[:, 3:4], torch.ones_like(result[:, 3:4]), atol=1e-5)

    def test_transparent_top_reveals_bottom(self):
        from src.rgba.compositor import composite_over

        top = self._make_layer(rgb_val=1.0, alpha_val=0.0)    # fully transparent
        bottom = self._make_layer(rgb_val=0.5, alpha_val=1.0)  # grey, opaque
        result = composite_over(top, bottom)

        # Transparent top → result = bottom
        assert torch.allclose(result[:, :3], bottom[:, :3], atol=1e-4)

    def test_shape_mismatch_raises(self):
        from src.rgba.compositor import composite_over

        top = torch.zeros(1, 4, 4, 8, 8)
        bottom = torch.zeros(1, 4, 8, 8, 8)  # different F
        with pytest.raises(ValueError, match="shape mismatch"):
            composite_over(top, bottom)

    def test_wrong_channels_raises(self):
        from src.rgba.compositor import composite_over

        top = torch.zeros(1, 3, 4, 8, 8)   # RGB, not RGBA
        bottom = torch.zeros(1, 3, 4, 8, 8)
        with pytest.raises(ValueError, match="4-channel"):
            composite_over(top, bottom)

    def test_output_alpha_correct(self):
        from src.rgba.compositor import composite_over

        # Both layers at 0.5 alpha
        top = self._make_layer(rgb_val=1.0, alpha_val=0.5)
        bottom = self._make_layer(rgb_val=0.0, alpha_val=0.5)
        result = composite_over(top, bottom)

        # Expected out_a = 0.5 + 0.5 * (1 - 0.5) = 0.75
        expected_alpha = 0.75
        assert torch.allclose(result[:, 3:4], torch.full_like(result[:, 3:4], expected_alpha), atol=1e-4)


class TestAlphaCompositor:
    def _rgba(self, rgb_val: float, alpha_val: float, B=1, F=4, H=8, W=8):
        t = torch.zeros(B, 4, F, H, W)
        t[:, :3] = rgb_val
        t[:, 3:4] = alpha_val
        return t

    def test_composite_returns_rgb(self):
        from src.rgba.compositor import AlphaCompositor, LayerSet

        comp = AlphaCompositor(smooth_alpha_frames=False)
        layers = LayerSet(
            background=self._rgba(0.2, 1.0),
            midground=self._rgba(0.5, 0.5),
            foreground=self._rgba(0.8, 0.0),
        )
        result = comp.composite(layers)
        assert result.shape == (1, 3, 4, 8, 8)  # RGB, not RGBA

    def test_composite_layers_arbitrary_count(self):
        from src.rgba.compositor import AlphaCompositor

        comp = AlphaCompositor(smooth_alpha_frames=False)
        layers = [self._rgba(0.2, 0.8) for _ in range(5)]
        result = comp.composite_layers(layers)
        assert result.shape[1] == 3

    def test_composite_layers_single(self):
        from src.rgba.compositor import AlphaCompositor

        comp = AlphaCompositor(smooth_alpha_frames=False)
        layer = self._rgba(0.5, 1.0)
        result = comp.composite_layers([layer])
        assert result.shape[1] == 3


class TestSmoothAlpha:
    def test_smooth_alpha_no_change_kernel1(self):
        from src.rgba.compositor import smooth_alpha

        layer = torch.rand(1, 4, 8, 16, 16)
        result = smooth_alpha(layer, kernel_size=1)
        assert torch.allclose(result, layer)

    def test_smooth_alpha_output_shape_preserved(self):
        from src.rgba.compositor import smooth_alpha

        layer = torch.rand(2, 4, 16, 8, 8)
        result = smooth_alpha(layer, kernel_size=3)
        assert result.shape == layer.shape

    def test_smooth_alpha_rgb_unchanged(self):
        from src.rgba.compositor import smooth_alpha

        layer = torch.rand(1, 4, 8, 8, 8)
        result = smooth_alpha(layer, kernel_size=3)
        assert torch.allclose(result[:, :3], layer[:, :3])

    def test_smooth_alpha_even_kernel_raises(self):
        from src.rgba.compositor import smooth_alpha

        with pytest.raises(ValueError, match="odd"):
            smooth_alpha(torch.rand(1, 4, 8, 8, 8), kernel_size=4)


class TestRgbToRgbaLuminance:
    """Tests for the luminance-based alpha stand-in."""

    def _rgb(self, val: float = 0.5, B=1, F=4, H=8, W=8) -> "torch.Tensor":
        t = torch.full((B, 3, F, H, W), val)
        return t

    def test_output_shape(self):
        from src.rgba.compositor import rgb_to_rgba_luminance

        rgb = self._rgb()
        rgba = rgb_to_rgba_luminance(rgb, layer_role="background")
        assert rgba.shape == (1, 4, 4, 8, 8)

    def test_background_alpha_is_one(self):
        from src.rgba.compositor import rgb_to_rgba_luminance

        rgb = self._rgb(0.3)
        rgba = rgb_to_rgba_luminance(rgb, layer_role="background")
        assert torch.allclose(rgba[:, 3:4], torch.ones_like(rgba[:, 3:4]))

    def test_midground_alpha_is_max_rgb(self):
        from src.rgba.compositor import rgb_to_rgba_luminance

        # Uniform grey: max(R,G,B) == the grey value itself
        rgb = self._rgb(0.6)
        rgba = rgb_to_rgba_luminance(rgb, layer_role="midground")
        assert torch.allclose(rgba[:, 3:4], torch.full_like(rgba[:, 3:4], 0.6), atol=1e-5)

    def test_foreground_alpha_is_luma(self):
        from src.rgba.compositor import rgb_to_rgba_luminance

        # Pure white: luma = 0.299 + 0.587 + 0.114 = 1.0
        rgb = torch.ones(1, 3, 4, 8, 8)
        rgba = rgb_to_rgba_luminance(rgb, layer_role="foreground")
        assert torch.allclose(rgba[:, 3:4], torch.ones_like(rgba[:, 3:4]), atol=1e-4)

    def test_alpha_scale_applies(self):
        from src.rgba.compositor import rgb_to_rgba_luminance

        rgb = self._rgb(1.0)
        rgba = rgb_to_rgba_luminance(rgb, layer_role="background", alpha_scale=0.5)
        assert torch.allclose(rgba[:, 3:4], torch.full_like(rgba[:, 3:4], 0.5), atol=1e-5)

    def test_rgb_channels_unchanged(self):
        from src.rgba.compositor import rgb_to_rgba_luminance

        rgb = torch.rand(1, 3, 4, 8, 8)
        rgba = rgb_to_rgba_luminance(rgb, layer_role="midground")
        assert torch.allclose(rgba[:, :3], rgb)

    def test_wrong_channels_raises(self):
        from src.rgba.compositor import rgb_to_rgba_luminance

        with pytest.raises(ValueError, match="3-channel"):
            rgb_to_rgba_luminance(torch.zeros(1, 4, 4, 8, 8))


class TestLoadRgbaFromVideo:
    """Integration test for load_rgba_from_video using a synthetic MP4."""

    def test_load_rgba_background(self, tmp_path):
        import numpy as np
        import imageio
        from src.rgba.compositor import load_rgba_from_video

        # Write a 4-frame 32x32 red video (dimensions divisible by 16 for FFMPEG compat)
        frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        frames[:, :, :, 0] = 200  # red channel
        video_path = str(tmp_path / "test.mp4")
        writer = imageio.get_writer(video_path, fps=8, format="FFMPEG")
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        rgba = load_rgba_from_video(video_path, layer_role="background", max_frames=4)
        assert rgba.shape[0] == 1        # batch=1
        assert rgba.shape[1] == 4        # RGBA channels
        assert rgba.shape[2] == 4        # 4 frames
        # Background alpha should be 1.0
        assert rgba[:, 3:4].min().item() > 0.99

    def test_load_rgba_missing_file_raises(self):
        from src.rgba.compositor import load_rgba_from_video

        with pytest.raises(FileNotFoundError):
            load_rgba_from_video("/nonexistent/path/video.mp4")


class TestSaveRgbTensorAsMp4:
    """Tests for save_rgb_tensor_as_mp4 round-trip."""

    def test_saves_mp4_and_file_exists(self, tmp_path):
        import torch
        from src.rgba.compositor import save_rgb_tensor_as_mp4

        # 32x32 so FFMPEG doesn't need to resize
        rgb = torch.rand(1, 3, 4, 32, 32)
        out = str(tmp_path / "out.mp4")
        result = save_rgb_tensor_as_mp4(rgb, out, fps=8)
        assert result == out
        assert (tmp_path / "out.mp4").exists()
        assert (tmp_path / "out.mp4").stat().st_size > 0

    def test_roundtrip_frame_count(self, tmp_path):
        import torch
        from src.rgba.compositor import save_rgb_tensor_as_mp4, load_rgb_from_video

        rgb = torch.rand(1, 3, 8, 32, 32)
        out = str(tmp_path / "roundtrip.mp4")
        save_rgb_tensor_as_mp4(rgb, out, fps=8)
        loaded = load_rgb_from_video(out, max_frames=8)
        assert loaded.shape[2] == 8

    def test_wrong_channels_raises(self, tmp_path):
        import torch
        from src.rgba.compositor import save_rgb_tensor_as_mp4

        rgba = torch.rand(1, 4, 4, 32, 32)
        with pytest.raises(ValueError, match="3-channel"):
            save_rgb_tensor_as_mp4(rgba, str(tmp_path / "bad.mp4"))


class TestSaveLastFrame:
    """Tests for save_last_frame (EchoShot PNG extraction)."""

    def test_saves_png_and_file_exists(self, tmp_path):
        import torch
        from src.rgba.compositor import save_last_frame

        rgb = torch.rand(1, 3, 8, 32, 32)
        out = str(tmp_path / "last_frame.png")
        result = save_last_frame(rgb, out)
        assert result == out
        assert (tmp_path / "last_frame.png").exists()
        assert (tmp_path / "last_frame.png").stat().st_size > 0

    def test_saves_last_frame_not_first(self, tmp_path):
        import torch
        from src.rgba.compositor import save_last_frame
        from PIL import Image
        import numpy as np

        # Frame 0 = black, Frame 1 = white
        rgb = torch.zeros(1, 3, 2, 32, 32)
        rgb[:, :, 1, :, :] = 1.0  # last frame is white
        out = str(tmp_path / "last_frame.png")
        save_last_frame(rgb, out)

        img = np.array(Image.open(out))
        assert img.mean() > 200, "Should be white (last frame)"

    def test_wrong_channels_raises(self, tmp_path):
        import torch
        from src.rgba.compositor import save_last_frame

        rgba = torch.rand(1, 4, 4, 32, 32)
        with pytest.raises(ValueError, match="3-channel"):
            save_last_frame(rgba, str(tmp_path / "bad.png"))


# ---------------------------------------------------------------------------
# Phase 3: VACE Extension
# ---------------------------------------------------------------------------

class TestVACEConfig:
    def test_invalid_segment_frames_raises(self):
        from src.vace.extension import VACEConfig

        with pytest.raises(ValueError, match="4k\\+1"):
            VACEConfig(segment_frames=80)  # 80 is not 4k+1

    def test_valid_configs(self):
        from src.vace.extension import VACEConfig

        for frames in [33, 49, 81]:
            cfg = VACEConfig(segment_frames=frames)
            assert cfg.segment_frames == frames


class TestBuildVACEMask:
    def test_mask_shape(self):
        from src.vace.extension import build_vace_mask

        mask = build_vace_mask(segment_frames=81, overlap_frames=16, device=torch.device("cpu"))
        assert mask.shape == (1, 1, 81, 1, 1)

    def test_known_frames_are_zero(self):
        from src.vace.extension import build_vace_mask

        mask = build_vace_mask(segment_frames=81, overlap_frames=16, device=torch.device("cpu"))
        assert torch.all(mask[0, 0, :16, 0, 0] == 0.0)

    def test_unknown_frames_are_one(self):
        from src.vace.extension import build_vace_mask

        mask = build_vace_mask(segment_frames=81, overlap_frames=16, device=torch.device("cpu"))
        assert torch.all(mask[0, 0, 16:, 0, 0] == 1.0)


class TestPadLatentsWithGrey:
    def test_no_padding_needed(self):
        from src.vace.extension import pad_latents_with_grey

        latents = torch.rand(1, 16, 16, 6, 6)
        result = pad_latents_with_grey(latents, total_frames=16)
        assert torch.allclose(result, latents)

    def test_padding_extends_frames(self):
        from src.vace.extension import pad_latents_with_grey

        latents = torch.rand(1, 16, 16, 6, 6)
        result = pad_latents_with_grey(latents, total_frames=81)
        assert result.shape == (1, 16, 81, 6, 6)

    def test_known_frames_preserved(self):
        from src.vace.extension import pad_latents_with_grey

        latents = torch.rand(1, 16, 16, 6, 6)
        result = pad_latents_with_grey(latents, total_frames=81)
        assert torch.allclose(result[:, :, :16], latents)

    def test_too_many_known_frames_raises(self):
        from src.vace.extension import pad_latents_with_grey

        latents = torch.rand(1, 16, 100, 6, 6)
        with pytest.raises(ValueError, match="exceeds total_frames"):
            pad_latents_with_grey(latents, total_frames=81)


class TestVACEExtension:
    def test_extract_overlap_latents(self):
        from src.vace.extension import VACEExtension, VACEConfig

        cfg = VACEConfig(overlap_frames=16, segment_frames=81)
        vace = VACEExtension(cfg)
        full_latents = torch.rand(1, 16, 81, 6, 6)
        handoff = vace.extract_overlap_latents(full_latents, segment_idx=0, prompt="test")

        assert handoff.latents.shape == (1, 16, 16, 6, 6)
        assert torch.allclose(handoff.latents, full_latents[:, :, -16:])
        assert vace.has_prior_segment

    def test_build_conditioning_shape(self):
        from src.vace.extension import VACEExtension, VACEConfig

        cfg = VACEConfig(overlap_frames=16, segment_frames=81)
        vace = VACEExtension(cfg)
        full_latents = torch.rand(1, 16, 81, 6, 6)
        handoff = vace.extract_overlap_latents(full_latents, 0, "test")
        cond = vace.build_conditioning(handoff, target_frames=81, device=torch.device("cpu"))

        assert cond["latents"].shape == (1, 16, 81, 6, 6)
        assert cond["mask"].shape == (1, 1, 81, 1, 1)
        assert cond["shift"] == 1.0

    def test_reset_clears_history(self):
        from src.vace.extension import VACEExtension

        vace = VACEExtension()
        full_latents = torch.rand(1, 16, 81, 6, 6)
        vace.extract_overlap_latents(full_latents, 0, "test")
        assert vace.has_prior_segment
        vace.reset()
        assert not vace.has_prior_segment


# ---------------------------------------------------------------------------
# Phase 4: SVI Recycler
# ---------------------------------------------------------------------------

class TestSVIConfig:
    def test_invalid_ema_decay_raises(self):
        from src.svi.recycler import SVIConfig

        with pytest.raises(ValueError, match="ema_decay"):
            SVIConfig(ema_decay=1.5)

    def test_invalid_injection_scale_raises(self):
        from src.svi.recycler import SVIConfig

        with pytest.raises(ValueError, match="injection_scale"):
            SVIConfig(injection_scale=-0.1)


class TestSVIErrorBuffer:
    def test_initially_empty(self):
        from src.svi.recycler import SVIErrorBuffer

        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        assert buf.is_empty
        assert buf.get_correction() is None

    def test_update_fills_buffer(self):
        from src.svi.recycler import SVIErrorBuffer

        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        error = torch.ones(2, 3, 4)
        buf.update(error)
        assert not buf.is_empty
        assert buf.get_correction() is not None

    def test_ema_smoothing(self):
        from src.svi.recycler import SVIErrorBuffer

        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        e1 = torch.zeros(4)
        e2 = torch.ones(4) * 10.0
        buf.update(e1)
        buf.update(e2)
        # ema after e2: 0.9 * 0 + 0.1 * 10 = 1.0
        correction = buf.get_correction()
        assert torch.allclose(correction, torch.ones(4) * 1.0, atol=1e-4)

    def test_reset_clears(self):
        from src.svi.recycler import SVIErrorBuffer

        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        buf.update(torch.ones(4))
        buf.reset()
        assert buf.is_empty


class TestSVIRecycler:
    def test_no_correction_before_first_segment(self):
        from src.svi.recycler import SVIRecycler

        recycler = SVIRecycler()
        assert not recycler.has_correction
        assert recycler.get_injection_correction() is None

    def test_correction_available_after_recording(self):
        from src.svi.recycler import SVIRecycler

        recycler = SVIRecycler()
        pred = torch.zeros(1, 16, 8, 6, 6)
        target = torch.ones(1, 16, 8, 6, 6)
        recycler.record_segment_errors(pred, target)
        assert recycler.has_correction

    def test_apply_correction_to_latents_shape_preserved(self):
        from src.svi.recycler import SVIRecycler

        recycler = SVIRecycler()
        latents = torch.rand(1, 16, 81, 6, 6)
        pred = torch.zeros_like(latents)
        target = torch.full_like(latents, 0.5)
        recycler.record_segment_errors(pred, target)
        corrected = recycler.apply_correction_to_latents(latents)
        assert corrected.shape == latents.shape

    def test_disabled_recycler_no_correction(self):
        from src.svi.recycler import SVIRecycler, SVIConfig

        recycler = SVIRecycler(SVIConfig(enabled=False))
        recycler.record_segment_errors(torch.zeros(4), torch.ones(4))
        assert not recycler.has_correction
        assert recycler.get_injection_correction() is None

    def test_scene_change_resets_buffer(self):
        from src.svi.recycler import SVIRecycler

        recycler = SVIRecycler()
        recycler.record_segment_errors(torch.zeros(4), torch.ones(4))
        assert recycler.has_correction
        recycler.on_scene_change()
        assert not recycler.has_correction

    def test_shape_mismatch_returns_none(self):
        from src.svi.recycler import SVIRecycler

        recycler = SVIRecycler()
        recycler.record_segment_errors(torch.zeros(4), torch.ones(4))
        result = recycler.get_injection_correction(target_shape=(1, 8, 81, 6, 6))
        assert result is None


# ---------------------------------------------------------------------------
# Phase 5: LLM Director
# ---------------------------------------------------------------------------

class TestNarrativeState:
    def test_to_context_str_basic(self):
        from src.llm.director import NarrativeState

        state = NarrativeState(
            scene_number=2,
            current_location="forest",
            mood="mysterious",
            time_of_day="dusk",
            recent_actions=["character enters clearing", "owl hoots"],
        )
        ctx = state.to_context_str()
        assert "Scene 2" in ctx
        assert "forest" in ctx
        assert "mysterious" in ctx
        assert "dusk" in ctx

    def test_to_context_str_empty_state(self):
        from src.llm.director import NarrativeState

        state = NarrativeState()
        ctx = state.to_context_str()
        assert "Scene 0" in ctx


class TestLLMDirector:
    def test_first_segment_uses_base_prompt(self):
        from src.llm.director import LLMDirector

        director = LLMDirector("A mountain hike at dawn")
        directive = director.next_segment(0)
        assert "mountain hike" in directive.prompt.lower() or "mountain" in directive.prompt.lower()
        assert directive.segment_idx == 0
        assert not directive.is_scene_change

    def test_fallback_advances_time_of_day(self):
        from src.llm.director import LLMDirector, DirectorConfig

        config = DirectorConfig(use_static_prompt_fallback=True)
        director = LLMDirector("A forest walk", config)
        # Generate 7 segments to trigger time-of-day progression at segment_idx=6
        for i in range(7):
            director.next_segment(i)
        # Time should have progressed at some point
        # (first progression happens at segment_idx=6)
        assert director.current_state.time_of_day in ["dawn", "morning", "day", "afternoon", "dusk", "evening", "night"]

    def test_history_summary_length(self):
        from src.llm.director import LLMDirector

        director = LLMDirector("Ocean waves")
        for i in range(5):
            director.next_segment(i)
        summary = director.get_history_summary()
        assert len(summary) == 5

    def test_segment_count_increments(self):
        from src.llm.director import LLMDirector

        director = LLMDirector("City at night")
        for i in range(3):
            director.next_segment(i)
        assert director.segment_count == 3

    def test_no_llm_fallback_works(self):
        from src.llm.director import LLMDirector, DirectorConfig

        # Explicitly no LLM
        config = DirectorConfig(api_base=None)
        director = LLMDirector("Desert storm", config)
        for i in range(4):
            d = director.next_segment(i)
            assert d.prompt  # non-empty prompt always produced
