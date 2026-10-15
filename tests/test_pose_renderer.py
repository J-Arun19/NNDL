import cv2
import numpy as np
import pytest

import config
from src.pose_extractor import LandmarkPoint
from src.pose_renderer import (
    draw_pose_overlay,
    load_preprocessing_config,
    normalize_and_clamp_point,
    render_pose_map,
    save_preprocessing_config,
)
from src.inference import transcode_to_browser_h264


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


def test_draw_pose_overlay_safe_and_unsafe():
    """Verify HUD bar, skeleton connections, joint circles, and biomechanical angles in overlay."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = [
        LandmarkPoint(id=11, name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=0.9, presence=1.0),
        LandmarkPoint(id=12, name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=0.95, presence=1.0),
        LandmarkPoint(id=23, name="left_hip", x=0.42, y=0.6, z=0.0, visibility=0.88, presence=1.0),
        LandmarkPoint(id=24, name="right_hip", x=0.58, y=0.6, z=0.0, visibility=0.91, presence=1.0),
    ]

    # Test SAFE posture overlay
    annotated_safe = draw_pose_overlay(
        frame_bgr=frame,
        landmarks=landmarks,
        status_label="SAFE POSTURE",
        probability=0.12,
        angles={"trunk_inclination": 24.5, "knee_angle_mean": 110.0},
        fps=30.0,
    )
    assert annotated_safe.shape == (480, 640, 3)
    assert annotated_safe.dtype == np.uint8
    # Top HUD banner should be non-zero
    assert np.any(annotated_safe[:74, :] > 0)
    # Bottom ribbon should be non-zero
    assert np.any(annotated_safe[-38:, :] > 0)

    # Test UNSAFE posture overlay
    annotated_unsafe = draw_pose_overlay(
        frame_bgr=frame,
        landmarks=landmarks,
        status_label="UNSAFE POSTURE",
        probability=0.88,
        angles={"trunk_inclination": 58.0, "knee_angle_mean": 150.0},
        fps=25.0,
    )
    assert annotated_unsafe.shape == (480, 640, 3)
    assert np.any(annotated_unsafe[:74, :] > 0)


def test_transcode_to_browser_h264(tmp_path):
    """Verify transcoding of OpenCV-created raw video to web-ready H.264 MP4."""
    raw_video = tmp_path / "raw_clip.mp4"
    h264_video = tmp_path / "h264_clip.mp4"

    vw = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for _ in range(5):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (80, 60), 20, (0, 255, 0), -1)
        vw.write(frame)
    vw.release()

    assert raw_video.exists() and raw_video.stat().st_size > 0

    success = transcode_to_browser_h264(raw_video, h264_video)
    assert h264_video.exists()
    assert h264_video.stat().st_size > 0
