"""Tests for generate_video LTX support and last-frame extraction (no GPU required)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_last_frame_path_derivation():
    """Last frame path is always derivable from output_path."""
    output_path = "/tmp/seg_0000.mp4"
    expected = "/tmp/seg_0000_last_frame.png"
    assert output_path.replace(".mp4", "_last_frame.png") == expected


def test_save_last_frame_writes_png(tmp_path):
    from src.wan.generate import _save_last_frame
    frames = np.random.rand(49, 64, 64, 3).astype(np.float32)
    output_path = str(tmp_path / "seg_0000.mp4")
    Path(output_path).touch()
    last_frame_path = _save_last_frame(frames, output_path)
    assert Path(last_frame_path).exists()
    assert last_frame_path.endswith("_last_frame.png")


def test_save_last_frame_correct_dimensions(tmp_path):
    from src.wan.generate import _save_last_frame
    from PIL import Image
    frames = np.random.rand(49, 48, 80, 3).astype(np.float32)
    output_path = str(tmp_path / "seg_0000.mp4")
    Path(output_path).touch()
    last_frame_path = _save_last_frame(frames, output_path)
    img = Image.open(last_frame_path)
    assert img.size == (80, 48)  # PIL size is (width, height)


def test_save_last_frame_clips_float_values(tmp_path):
    from src.wan.generate import _save_last_frame
    from PIL import Image
    import numpy as np
    # Values outside [0,1] should be clipped, not wrap around
    frames = np.ones((10, 32, 32, 3), dtype=np.float32) * 1.5
    output_path = str(tmp_path / "seg_clip.mp4")
    Path(output_path).touch()
    last_frame_path = _save_last_frame(frames, output_path)
    img = Image.open(last_frame_path)
    pixels = np.array(img)
    assert pixels.max() == 255  # clipped to max, not 127 (which wrap-around would give)
