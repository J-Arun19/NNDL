"""Unit tests for transparent ergonomic screening and biomechanical posture assessment."""

from pathlib import Path
import cv2
import numpy as np
import pytest

import config
from src.inference import (
    LiftingPostureClassifier,
    STATUS_SAFE_POSTURE,
    STATUS_UNSAFE_POSTURE,
    STATUS_INSUFFICIENT_VISIBILITY,
    MODE_RULE_BASED_ERGONOMICS,
    VideoAnalysisSummary,
)
from src.pose_extractor import LandmarkPoint, PoseExtractionResult


def _create_mock_landmarks(visibility: float = 0.95) -> list[LandmarkPoint]:
    """Helper to generate synthetic landmark points matching PoseExtractionResult schema."""
    landmarks = []
    for idx in range(33):
        landmarks.append(
            LandmarkPoint(
                id=idx,
                name=f"landmark_{idx}",
                x=0.5,
                y=0.5,
                z=0.0,
                visibility=visibility,
                presence=visibility,
            )
        )
    return landmarks


def test_classifier_defaults_to_transparent_rule_based_mode():
    """Verify classifier initializes in rule-based mode when CNN checkpoint is absent."""
    classifier = LiftingPostureClassifier(model_path=Path("non_existent_model.keras"))
    assert not classifier.is_model_trained
    assert classifier.current_assessment_method == MODE_RULE_BASED_ERGONOMICS
    classifier.close()


def test_ergonomic_rule_squat_posture_safe():
    """Verify that an upright trunk (<=45 deg) with flexed knees (<=125 deg) yields SAFE POSTURE."""
    classifier = LiftingPostureClassifier()
    
    mock_res = PoseExtractionResult(
        is_valid=True,
        status_message="Valid pose detected.",
        landmarks=_create_mock_landmarks(),
        raw_landmarks=None,
        critical_visibility_ratio=1.0,
        biomechanical_angles={
            "trunk_inclination": 28.0,
            "knee_angle_left": 98.0,
            "knee_angle_right": 98.0,
            "knee_angle_mean": 98.0,
        },
        diagnostic_notes=[],
    )
    
    status, reason = classifier.evaluate_frame_ergonomics(mock_res)
    assert status == STATUS_SAFE_POSTURE
    assert "squat" in reason.lower() or "upright" in reason.lower()
    classifier.close()


def test_ergonomic_rule_stoop_posture_unsafe():
    """Verify that a bent trunk (>45 deg) with extended knees (>=135 deg) yields UNSAFE POSTURE."""
    classifier = LiftingPostureClassifier()
    
    mock_res = PoseExtractionResult(
        is_valid=True,
        status_message="Valid pose detected.",
        landmarks=_create_mock_landmarks(),
        raw_landmarks=None,
        critical_visibility_ratio=1.0,
        biomechanical_angles={
            "trunk_inclination": 62.0,
            "knee_angle_left": 165.0,
            "knee_angle_right": 165.0,
            "knee_angle_mean": 165.0,
        },
        diagnostic_notes=[],
    )
    
    status, reason = classifier.evaluate_frame_ergonomics(mock_res)
    assert status == STATUS_UNSAFE_POSTURE
    assert "trunk" in reason.lower() or "bending" in reason.lower() or "shear" in reason.lower()
    classifier.close()


def test_ergonomic_rule_severe_trunk_flexion_unsafe():
    """Verify that severe forward trunk flexion (>50 deg) is classified UNSAFE regardless of knees."""
    classifier = LiftingPostureClassifier()
    
    mock_res = PoseExtractionResult(
        is_valid=True,
        status_message="Valid pose detected.",
        landmarks=_create_mock_landmarks(),
        raw_landmarks=None,
        critical_visibility_ratio=1.0,
        biomechanical_angles={
            "trunk_inclination": 58.0,
            "knee_angle_left": 110.0,
            "knee_angle_right": 110.0,
            "knee_angle_mean": 110.0,
        },
        diagnostic_notes=[],
    )
    
    status, reason = classifier.evaluate_frame_ergonomics(mock_res)
    assert status == STATUS_UNSAFE_POSTURE
    assert "flexion" in reason.lower() or "trunk" in reason.lower()
    classifier.close()


def test_insufficient_visibility_when_pose_invalid():
    """Verify invalid/occluded pose returns INSUFFICIENT VISIBILITY without guessing."""
    classifier = LiftingPostureClassifier()
    
    mock_res = PoseExtractionResult(
        is_valid=False,
        status_message="Critical keypoints (knees, ankles) occluded.",
        landmarks=[],
        raw_landmarks=None,
        critical_visibility_ratio=0.2,
        biomechanical_angles={
            "trunk_inclination": None,
            "knee_angle_mean": None,
        },
        diagnostic_notes=[],
    )
    
    status, reason = classifier.evaluate_frame_ergonomics(mock_res)
    assert status == STATUS_INSUFFICIENT_VISIBILITY
    assert "unreliable" in reason.lower() or "occluded" in reason.lower()
    classifier.close()


def test_temporal_smoothing_and_alert_debouncing():
    """Verify rolling majority smoothing and consecutive frame debouncing."""
    classifier = LiftingPostureClassifier(rolling_window_size=5, alert_debounce_threshold=3)
    
    classifier.status_buffer.clear()
    classifier.status_buffer.extend([STATUS_UNSAFE_POSTURE, STATUS_UNSAFE_POSTURE])
    classifier.consecutive_unsafe_frames = 2
    
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # Mock evaluate_frame_ergonomics to return UNSAFE
    classifier.evaluate_frame_ergonomics = lambda res: (STATUS_UNSAFE_POSTURE, "Simulated unsafe")
    classifier.extractor.extract_from_frame = lambda frame: PoseExtractionResult(
        is_valid=True,
        status_message="Valid",
        landmarks=_create_mock_landmarks(),
        raw_landmarks=None,
        critical_visibility_ratio=1.0,
        biomechanical_angles={"trunk_inclination": 55.0, "knee_angle_mean": 140.0},
        diagnostic_notes=[],
    )
    
    res1, _ = classifier.process_frame(blank, frame_index=1)
    assert res1.is_alert_triggered is True
    assert res1.smoothed_status == STATUS_UNSAFE_POSTURE
    
    classifier.close()


def test_process_recorded_video_on_synthetic_fixture(tmp_path):
    """Verify video analysis pipeline on synthetic video fixture returns summary and annotations."""
    from src.inference import process_recorded_video
    
    video_path = tmp_path / "test_clip.mp4"
    out_annotated = tmp_path / "test_annotated.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for i in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 120), 30, (200, 200, 200), -1)
        vw.write(frame)
    vw.release()
    
    classifier = LiftingPostureClassifier()
    df_preds, out_path, summary = process_recorded_video(
        video_path=video_path,
        output_annotated_video_path=out_annotated,
        classifier=classifier,
    )
    
    assert isinstance(summary, VideoAnalysisSummary)
    assert summary.total_frames == 10
    assert summary.overall_classification in (
        STATUS_SAFE_POSTURE,
        STATUS_UNSAFE_POSTURE,
        STATUS_INSUFFICIENT_VISIBILITY,
    )
    assert len(df_preds) == 10
    assert "status" in df_preds.columns
    assert "explanation" in df_preds.columns
    assert Path(out_path).exists()
    
    classifier.close()
