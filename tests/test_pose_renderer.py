"""Unit tests for pose-map rendering and landmark normalization."""

import numpy as np
import pytest

import config
from src.pose_extractor import LandmarkPoint
from src.pose_renderer import (
    load_preprocessing_config,
    normalize_and_clamp_point,
    render_pose_map,
    save_preprocessing_config,
)


def test_normalize_and_clamp_point():
    """Verify coordinate normalization and boundary clamping."""
    # Standard inside bounds
    px, py = normalize_and_clamp_point(0.5, 0.5, width=128, height=128)
    assert 0 <= px < 128
    assert 0 <= py < 128

    # Edge minimum
    px, py = normalize_and_clamp_point(0.0, 0.0, width=128, height=128)
    assert px == 0 and py == 0

    # Edge maximum
    px, py = normalize_and_clamp_point(1.0, 1.0, width=128, height=128)
    assert px == 127 and py == 127

    # Clamping out-of-frame values
    px_low, py_low = normalize_and_clamp_point(-0.3, -0.5, width=128, height=128)
    assert px_low == 0 and py_low == 0

    px_high, py_high = normalize_and_clamp_point(1.5, 2.0, width=128, height=128)
    assert px_high == 127 and py_high == 127


def test_render_pose_map_empty_or_none():
    """Test that absent poses produce a deterministic blank 128x128 canvas."""
    blank_none = render_pose_map(None)
    assert blank_none.shape == (128, 128, 1)
    assert blank_none.dtype == np.uint8
    assert np.all(blank_none == 0)

    blank_empty = render_pose_map([])
    assert blank_empty.shape == (128, 128, 1)
    assert np.all(blank_empty == 0)

    blank_float = render_pose_map(None, as_float=True)
    assert blank_float.shape == (128, 128, 1)
    assert blank_float.dtype == np.float32
    assert np.all(blank_float == 0.0)


def test_render_pose_map_with_visible_landmarks():
    """Test that visible landmarks and connecting bones are drawn on canvas."""
    landmarks = [
        LandmarkPoint(id=11, name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=0.9, presence=1.0),
        LandmarkPoint(id=12, name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=0.95, presence=1.0),
        LandmarkPoint(id=23, name="left_hip", x=0.42, y=0.6, z=0.0, visibility=0.88, presence=1.0),
        LandmarkPoint(id=24, name="right_hip", x=0.58, y=0.6, z=0.0, visibility=0.91, presence=1.0),
    ]

    pose_map = render_pose_map(landmarks)
    assert pose_map.shape == (128, 128, 1)
    assert pose_map.dtype == np.uint8

    # Must contain both joint pixels (intensity 255) and bone connection pixels (intensity 180)
    unique_vals = set(np.unique(pose_map))
    assert int(config.JOINT_INTENSITY) in unique_vals
    assert int(config.BONE_INTENSITY) in unique_vals


def test_render_pose_map_visibility_threshold():
    """Test that low-visibility landmarks (< 0.5) are omitted from rendering."""
    low_vis_lms = [
        LandmarkPoint(id=11, name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=0.2, presence=0.2),
        LandmarkPoint(id=12, name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=0.3, presence=0.3),
    ]

    pose_map = render_pose_map(low_vis_lms, min_visibility=0.5)
    assert pose_map.shape == (128, 128, 1)
    assert np.all(pose_map == 0), "Low visibility keypoints should not be rendered"


def test_save_and_load_preprocessing_config(tmp_path):
    """Test serialization and loading of deterministic preprocessing configuration."""
    cfg_file = tmp_path / "preprocessing_test.json"
    save_preprocessing_config(cfg_file)
    assert cfg_file.exists()

    loaded = load_preprocessing_config(cfg_file)
    assert loaded["canvas_height"] == 128
    assert loaded["canvas_width"] == 128
    assert loaded["canvas_channels"] == 1
    assert "skeletal_connections" in loaded
