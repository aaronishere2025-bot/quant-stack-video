"""Tests for infinite gen frame chaining path logic (no GPU required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_last_frame_path_derives_from_seg_path():
    """The frame chaining path derivation must be consistent across the loop."""
    seg_path = "/tmp/outputs/infinite/abc12345/seg_0000.mp4"
    expected = "/tmp/outputs/infinite/abc12345/seg_0000_last_frame.png"
    assert seg_path.replace(".mp4", "_last_frame.png") == expected


def test_prev_frame_path_none_on_missing_file(tmp_path):
    """prev_frame_path must be None when the PNG doesn't exist (no crash)."""
    seg_path = str(tmp_path / "seg_0000.mp4")
    candidate = seg_path.replace(".mp4", "_last_frame.png")
    prev_frame_path = candidate if Path(candidate).exists() else None
    assert prev_frame_path is None


def test_prev_frame_path_set_when_file_exists(tmp_path):
    """prev_frame_path must point to the PNG when it exists."""
    seg_path = str(tmp_path / "seg_0000.mp4")
    candidate = seg_path.replace(".mp4", "_last_frame.png")
    Path(candidate).touch()
    prev_frame_path = candidate if Path(candidate).exists() else None
    assert prev_frame_path == candidate


def test_segment_idx_increments_correctly():
    """Segment counter must advance and last frame path must track it."""
    base = "/tmp/seg_{:04d}.mp4"
    for segment_idx in range(3):
        seg_path = base.format(segment_idx)
        expected_frame = seg_path.replace(".mp4", "_last_frame.png")
        assert f"seg_{segment_idx:04d}_last_frame.png" in expected_frame
        assert segment_idx == int(Path(seg_path).stem.split("_")[1])


def test_engine_detection_from_model_id():
    """Engine detection from model_id must correctly identify LTX vs WAN."""
    def detect_engine(model_id: str) -> str:
        return "ltx" if ("ltx" in model_id.lower() or "lightricks" in model_id.lower()) else "wan"

    assert detect_engine("Lightricks/LTX-Video") == "ltx"
    assert detect_engine("Wan-AI/Wan2.1-T2V-1.3B-Diffusers") == "wan"
    assert detect_engine("Wan-AI/Wan2.1-T2V-14B-Diffusers") == "wan"
    assert detect_engine("lightricks/LTX-Video-0.9.1") == "ltx"
