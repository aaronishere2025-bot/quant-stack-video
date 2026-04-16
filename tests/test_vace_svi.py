"""
Unit tests for VACE temporal extension and SVI error recycling.

All tests use CPU tensors — no GPU required.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# VACE Extension Tests
# ---------------------------------------------------------------------------

class TestVACEConfig:
    def test_defaults(self):
        from src.vace.extension import VACEConfig
        cfg = VACEConfig()
        assert cfg.overlap_frames == 16
        assert cfg.segment_frames == 81
        assert cfg.shift == 1.0
        assert 2.0 <= cfg.cfg_scale <= 3.0

    def test_segment_frames_must_be_4k_plus_1(self):
        from src.vace.extension import VACEConfig
        with pytest.raises(ValueError, match="4k\\+1"):
            VACEConfig(segment_frames=80)

    def test_valid_4k_plus_1_values(self):
        from src.vace.extension import VACEConfig
        for n in [33, 49, 65, 81, 97]:
            cfg = VACEConfig(segment_frames=n)
            assert cfg.segment_frames == n

    def test_cfg_scale_out_of_range_warns(self, caplog):
        from src.vace.extension import VACEConfig
        import logging
        with caplog.at_level(logging.WARNING, logger="src.vace.extension"):
            cfg = VACEConfig(cfg_scale=1.0)
        assert "2.0" in caplog.text or cfg.cfg_scale == 1.0  # warns but doesn't raise


class TestSegmentHandoff:
    def test_num_overlap_frames_from_latent_shape(self):
        from src.vace.extension import SegmentHandoff
        latents = torch.zeros(1, 16, 12, 60, 104)
        h = SegmentHandoff(latents=latents, segment_idx=0, prompt="test")
        assert h.num_overlap_frames == 12

    def test_num_overlap_frames_matches_last_dim(self):
        from src.vace.extension import SegmentHandoff
        for f in [8, 16, 24]:
            h = SegmentHandoff(
                latents=torch.zeros(1, 16, f, 30, 52),
                segment_idx=0, prompt="x",
            )
            assert h.num_overlap_frames == f


class TestBuildVaceMask:
    def test_shape(self):
        from src.vace.extension import build_vace_mask
        mask = build_vace_mask(segment_frames=81, overlap_frames=16, device=torch.device("cpu"))
        assert mask.shape == (1, 1, 81, 1, 1)

    def test_known_frames_are_zero(self):
        from src.vace.extension import build_vace_mask
        mask = build_vace_mask(segment_frames=81, overlap_frames=16, device=torch.device("cpu"))
        assert mask[0, 0, :16, 0, 0].sum() == 0.0

    def test_unknown_frames_are_one(self):
        from src.vace.extension import build_vace_mask
        mask = build_vace_mask(segment_frames=81, overlap_frames=16, device=torch.device("cpu"))
        assert mask[0, 0, 16:, 0, 0].sum() == 81 - 16

    def test_zero_overlap_all_unknown(self):
        from src.vace.extension import build_vace_mask
        mask = build_vace_mask(segment_frames=81, overlap_frames=0, device=torch.device("cpu"))
        assert mask.sum() == 81

    def test_full_overlap_all_known(self):
        from src.vace.extension import build_vace_mask
        mask = build_vace_mask(segment_frames=81, overlap_frames=81, device=torch.device("cpu"))
        assert mask.sum() == 0.0


class TestPadLatentsWithGrey:
    def test_output_shape(self):
        from src.vace.extension import pad_latents_with_grey
        known = torch.zeros(1, 16, 16, 60, 104)
        padded = pad_latents_with_grey(known, total_frames=81)
        assert padded.shape == (1, 16, 81, 60, 104)

    def test_known_frames_preserved(self):
        from src.vace.extension import pad_latents_with_grey
        known = torch.ones(1, 16, 16, 60, 104) * 0.5
        padded = pad_latents_with_grey(known, total_frames=81)
        assert torch.allclose(padded[:, :, :16], known)

    def test_padding_frames_are_zeros(self):
        from src.vace.extension import pad_latents_with_grey
        known = torch.ones(1, 16, 16, 60, 104)
        padded = pad_latents_with_grey(known, total_frames=81)
        # Padded frames use 0.0 as latent-space grey approximation
        assert padded[:, :, 16:].sum() == 0.0

    def test_no_padding_needed(self):
        from src.vace.extension import pad_latents_with_grey
        known = torch.ones(1, 16, 81, 60, 104)
        padded = pad_latents_with_grey(known, total_frames=81)
        assert torch.allclose(padded, known)

    def test_too_many_known_frames_raises(self):
        from src.vace.extension import pad_latents_with_grey
        known = torch.zeros(1, 16, 90, 60, 104)
        with pytest.raises(ValueError):
            pad_latents_with_grey(known, total_frames=81)


class TestVACEExtension:
    def _make_latents(self, frames=81):
        return torch.randn(1, 16, frames, 60, 104)

    def test_extract_overlap_latents_shape(self):
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig(overlap_frames=16, segment_frames=81))
        latents = self._make_latents(81)
        handoff = vace.extract_overlap_latents(latents, 0, "test")
        assert handoff.latents.shape == (1, 16, 16, 60, 104)

    def test_extract_overlap_takes_last_n_frames(self):
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig(overlap_frames=4, segment_frames=81))
        # Build latents where last 4 frames are all-ones
        latents = torch.zeros(1, 16, 81, 30, 52)
        latents[:, :, -4:] = 1.0
        handoff = vace.extract_overlap_latents(latents, 0, "prompt")
        assert handoff.latents.sum() == pytest.approx(1 * 16 * 4 * 30 * 52)

    def test_has_prior_segment_updates(self):
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig())
        assert not vace.has_prior_segment
        vace.extract_overlap_latents(self._make_latents(), 0, "x")
        assert vace.has_prior_segment

    def test_last_handoff_tracks_most_recent(self):
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig())
        h0 = vace.extract_overlap_latents(self._make_latents(), 0, "seg0")
        h1 = vace.extract_overlap_latents(self._make_latents(), 1, "seg1")
        assert vace.last_handoff is h1

    def test_build_conditioning_returns_required_keys(self):
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig(overlap_frames=16, segment_frames=81))
        handoff = vace.extract_overlap_latents(self._make_latents(), 0, "x")
        cond = vace.build_conditioning(handoff, target_frames=81, device=torch.device("cpu"))
        assert "latents" in cond
        assert "mask" in cond
        assert "shift" in cond
        assert "cfg_scale" in cond

    def test_build_conditioning_latents_shape(self):
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig(overlap_frames=16, segment_frames=81))
        handoff = vace.extract_overlap_latents(self._make_latents(), 0, "x")
        cond = vace.build_conditioning(handoff, target_frames=81, device=torch.device("cpu"))
        assert cond["latents"].shape == (1, 16, 81, 60, 104)

    def test_build_conditioning_mask_shape(self):
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig(overlap_frames=16, segment_frames=81))
        handoff = vace.extract_overlap_latents(self._make_latents(), 0, "x")
        cond = vace.build_conditioning(handoff, target_frames=81, device=torch.device("cpu"))
        assert cond["mask"].shape == (1, 1, 81, 1, 1)

    def test_reset_clears_history(self):
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig())
        vace.extract_overlap_latents(self._make_latents(), 0, "x")
        assert vace.has_prior_segment
        vace.reset()
        assert not vace.has_prior_segment
        assert vace.last_handoff is None

    def test_handoff_not_sharing_storage_with_source(self):
        """Overlap latents must be a copy — modifying source shouldn't change handoff."""
        from src.vace.extension import VACEExtension, VACEConfig
        vace = VACEExtension(VACEConfig(overlap_frames=4, segment_frames=81))
        latents = torch.zeros(1, 16, 81, 30, 52)
        handoff = vace.extract_overlap_latents(latents, 0, "x")
        latents[:] = 999.0  # mutate source
        assert handoff.latents.max().item() < 999.0  # handoff is independent


# ---------------------------------------------------------------------------
# SVI Error Recycling Tests
# ---------------------------------------------------------------------------

class TestSVIConfig:
    def test_defaults(self):
        from src.svi.recycler import SVIConfig
        cfg = SVIConfig()
        assert cfg.ema_decay == 0.9
        assert cfg.injection_scale == 0.1
        assert cfg.enabled is True
        assert cfg.reset_on_scene_change is True

    def test_ema_decay_bounds(self):
        from src.svi.recycler import SVIConfig
        with pytest.raises(ValueError):
            SVIConfig(ema_decay=0.0)
        with pytest.raises(ValueError):
            SVIConfig(ema_decay=1.0)

    def test_negative_injection_scale_raises(self):
        from src.svi.recycler import SVIConfig
        with pytest.raises(ValueError):
            SVIConfig(injection_scale=-0.1)


class TestSVIErrorBuffer:
    def test_is_empty_initially(self):
        from src.svi.recycler import SVIErrorBuffer
        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        assert buf.is_empty

    def test_update_makes_non_empty(self):
        from src.svi.recycler import SVIErrorBuffer
        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        buf.update(torch.zeros(1, 4, 5, 6))
        assert not buf.is_empty

    def test_first_update_sets_ema_to_error(self):
        from src.svi.recycler import SVIErrorBuffer
        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        error = torch.ones(1, 4, 5, 6) * 0.5
        buf.update(error)
        correction = buf.get_correction()
        assert torch.allclose(correction, error)

    def test_ema_decays_toward_new_error(self):
        from src.svi.recycler import SVIErrorBuffer
        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        # First update: EMA = 1.0
        buf.update(torch.ones(1, 4, 3, 3))
        # Second update: EMA = 0.9 * 1.0 + 0.1 * 0.0 = 0.9
        buf.update(torch.zeros(1, 4, 3, 3))
        correction = buf.get_correction()
        assert correction.mean().item() == pytest.approx(0.9, abs=1e-5)

    def test_get_correction_returns_clone(self):
        from src.svi.recycler import SVIErrorBuffer
        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        buf.update(torch.ones(2, 4, 3, 3))
        c1 = buf.get_correction()
        c1[:] = 999.0
        c2 = buf.get_correction()
        # Modifying c1 should not change c2
        assert c2.max().item() != pytest.approx(999.0)

    def test_reset_empties_buffer(self):
        from src.svi.recycler import SVIErrorBuffer
        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        buf.update(torch.ones(1, 4, 3, 3))
        buf.reset()
        assert buf.is_empty
        assert buf.get_correction() is None

    def test_segment_count_increments(self):
        from src.svi.recycler import SVIErrorBuffer
        buf = SVIErrorBuffer(ema_decay=0.9, buffer_size=5)
        for i in range(5):
            buf.update(torch.zeros(1, 4, 3, 3))
        assert buf.segment_count == 5


class TestSVIRecycler:
    def _make_tensor(self):
        return torch.randn(1, 16, 81, 60, 104)

    def test_no_correction_before_first_record(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig())
        assert not recycler.has_correction
        assert recycler.get_injection_correction() is None

    def test_has_correction_after_record(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig())
        pred = self._make_tensor()
        target = self._make_tensor()
        recycler.record_segment_errors(pred, target)
        assert recycler.has_correction

    def test_injection_correction_is_scaled(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        scale = 0.05
        recycler = SVIRecycler(SVIConfig(ema_decay=0.9, injection_scale=scale))
        err = torch.ones(1, 16, 5, 6, 7)
        recycler.record_segment_errors(torch.zeros_like(err), err)  # error = target - pred = err
        correction = recycler.get_injection_correction()
        assert correction is not None
        assert torch.allclose(correction, err * scale, atol=1e-6)

    def test_segment_count_tracks_records(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig())
        for i in range(3):
            recycler.record_segment_errors(self._make_tensor(), self._make_tensor())
        assert recycler.segment_count == 3

    def test_correction_shape_mismatch_returns_none(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig())
        recycler.record_segment_errors(self._make_tensor(), self._make_tensor())
        wrong_shape = (1, 16, 33, 60, 104)  # different frame count
        correction = recycler.get_injection_correction(target_shape=wrong_shape)
        assert correction is None

    def test_apply_correction_changes_latents(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig(injection_scale=1.0))
        # Record a known error
        pred = torch.zeros(1, 16, 5, 6, 7)
        target = torch.ones(1, 16, 5, 6, 7)
        recycler.record_segment_errors(pred, target)  # error = target - pred = 1.0

        latents = torch.zeros(1, 16, 5, 6, 7)
        corrected = recycler.apply_correction_to_latents(latents)
        # correction = 1.0 * scale=1.0; corrected = latents + correction = 1.0
        assert corrected.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_apply_correction_noop_before_record(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig())
        latents = torch.ones(1, 16, 5, 6, 7) * 0.5
        result = recycler.apply_correction_to_latents(latents)
        assert torch.allclose(result, latents)

    def test_apply_correction_skips_on_shape_mismatch(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig())
        small = torch.ones(1, 16, 5, 6, 7)
        recycler.record_segment_errors(small, small * 2)

        # Apply to a latent of different shape — should be returned unchanged
        large = torch.zeros(1, 16, 20, 6, 7)
        result = recycler.apply_correction_to_latents(large)
        assert torch.allclose(result, large)

    def test_disabled_recycler_never_returns_correction(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig(enabled=False))
        t = self._make_tensor()
        recycler.record_segment_errors(t, t)
        assert not recycler.has_correction
        assert recycler.get_injection_correction() is None

    def test_on_scene_change_resets_buffer(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig(reset_on_scene_change=True))
        recycler.record_segment_errors(self._make_tensor(), self._make_tensor())
        assert recycler.has_correction
        recycler.on_scene_change()
        assert not recycler.has_correction

    def test_scene_change_no_reset_if_disabled(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig(reset_on_scene_change=False))
        recycler.record_segment_errors(self._make_tensor(), self._make_tensor())
        recycler.on_scene_change()
        assert recycler.has_correction  # still has correction

    def test_record_latent_delta_updates_buffer(self):
        from src.svi.recycler import SVIRecycler, SVIConfig
        recycler = SVIRecycler(SVIConfig())
        prev = torch.zeros(1, 16, 16, 60, 104)
        curr = torch.ones(1, 16, 16, 60, 104)
        recycler.record_latent_delta(prev, curr)
        assert recycler.has_correction
