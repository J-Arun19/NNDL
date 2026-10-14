"""Pose extraction module using MediaPipe PoseLandmarker.

Extracts 33 body keypoints with coordinates and visibility scores, performs
landmark quality validation, and calculates explanatory biomechanical angles.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config

logger = logging.getLogger(__name__)


@dataclass
class LandmarkPoint:
    """Represents a single body landmark with normalized coordinates and visibility."""
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass
class PoseExtractionResult:
    """Encapsulates the result of pose extraction for a single frame."""
    is_valid: bool
    status_message: str
    landmarks: Optional[List[LandmarkPoint]]
    raw_landmarks: Optional[List[Dict[str, float]]]
    critical_visibility_ratio: float
    biomechanical_angles: Dict[str, Optional[float]]
    diagnostic_notes: List[str]


# Landmark name mapping for MediaPipe Pose (33 points)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]


def ensure_pose_landmarker_model(target_path: Optional[Path] = None) -> Path:
    """Ensure that the MediaPipe pose landmarker task file is available locally."""
    model_path = target_path or config.POSE_LANDMARKER_TASK_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists() or model_path.stat().st_size == 0:
        logger.info(f"Downloading MediaPipe pose landmarker model to {model_path}...")
        urllib.request.urlretrieve(config.POSE_LANDMARKER_URL, str(model_path))
        logger.info(f"Model downloaded successfully ({model_path.stat().st_size} bytes).")
    return model_path


def calculate_2d_angle(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float]
) -> float:
    """Calculate the 2D planar angle at vertex b between rays (b -> a) and (b -> c).

    Returns the angle in degrees in range [0.0, 180.0].
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_trunk_inclination(
    shoulder: Tuple[float, float],
    hip: Tuple[float, float]
) -> float:
    """Calculate forward trunk inclination from vertical (0 deg = fully upright).

    Image coordinate y increases downward, so vertical downward is dx=0, dy>0.
    """
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]  # Should be negative because shoulder is above hip
    # Angle of torso vector (hip -> shoulder) from the upright vertical vector (0, -1)
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0
    # upright vector is (0, -1)
    cos_angle = -dy / norm  # dot([dx, dy], [0, -1]) = -dy
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(math.degrees(math.acos(cos_angle)))


class PoseExtractor:
    """Extracts, validates, and analyzes human pose keypoints from video frames."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = ensure_pose_landmarker_model(model_path)
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def extract_from_frame(self, frame_bgr: np.ndarray) -> PoseExtractionResult:
        """Extract pose landmarks from a BGR image array.

        Validates presence, landmark visibility, and critical lifting joint quality.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return PoseExtractionResult(
                is_valid=False,
                status_message="Empty or invalid image frame provided.",
                landmarks=None,
                raw_landmarks=None,
                critical_visibility_ratio=0.0,
                biomechanical_angles={},
                diagnostic_notes=["Frame array is empty or None."]
            )

        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as exc:
            logger.error(f"MediaPipe detection failed: {exc}")
            return PoseExtractionResult(
                is_valid=False,
                status_message=f"Pose detection error: {exc}",
                landmarks=None,
                raw_landmarks=None,
                critical_visibility_ratio=0.0,
                biomechanical_angles={},
                diagnostic_notes=[str(exc)]
            )

        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return PoseExtractionResult(
                is_valid=False,
                status_message="No person detected in the frame.",
                landmarks=None,
                raw_landmarks=None,
                critical_visibility_ratio=0.0,
                biomechanical_angles={},
                diagnostic_notes=["MediaPipe detected 0 pose candidates."]
            )

        # Retrieve first detected person landmarks
        first_pose = detection_result.pose_landmarks[0]
        parsed_landmarks: List[LandmarkPoint] = []
        raw_list: List[Dict[str, float]] = []

        for idx, lm in enumerate(first_pose):
            name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"point_{idx}"
            visibility = float(getattr(lm, "visibility", 1.0) or 1.0)
            presence = float(getattr(lm, "presence", 1.0) or 1.0)
            parsed_landmarks.append(
                LandmarkPoint(
                    id=idx,
                    name=name,
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=visibility,
                    presence=presence
                )
            )
            raw_list.append({
                "id": idx,
                "name": name,
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": visibility,
            })

        # Quality Gate: Evaluate critical keypoints visibility
        critical_visible = 0
        total_critical = len(config.CRITICAL_LANDMARK_INDICES)
        diagnostic_notes: List[str] = []

        for c_idx in config.CRITICAL_LANDMARK_INDICES:
            if c_idx < len(parsed_landmarks):
                lm = parsed_landmarks[c_idx]
                if lm.visibility >= config.MIN_LANDMARK_VISIBILITY:
                    critical_visible += 1
                else:
                    diagnostic_notes.append(
                        f"Low visibility for {lm.name} ({lm.visibility:.2f} < {config.MIN_LANDMARK_VISIBILITY})"
                    )

        critical_ratio = critical_visible / total_critical if total_critical > 0 else 0.0
        is_quality_sufficient = critical_ratio >= config.MIN_CRITICAL_KEYPOINTS_RATIO

        if not is_quality_sufficient:
            status_msg = (
                f"Insufficient pose quality: only {critical_visible}/{total_critical} "
                f"critical lifting joints visible ({critical_ratio:.1%})."
            )
            return PoseExtractionResult(
                is_valid=False,
                status_message=status_msg,
                landmarks=parsed_landmarks,
                raw_landmarks=raw_list,
                critical_visibility_ratio=critical_ratio,
                biomechanical_angles={},
                diagnostic_notes=diagnostic_notes
            )

        # Compute explanatory biomechanical angles (Lateral or Bilateral average)
        angles = self._compute_biomechanical_angles(parsed_landmarks)

        return PoseExtractionResult(
            is_valid=True,
            status_message="Valid pose successfully extracted.",
            landmarks=parsed_landmarks,
            raw_landmarks=raw_list,
            critical_visibility_ratio=critical_ratio,
            biomechanical_angles=angles,
            diagnostic_notes=diagnostic_notes
        )

    def _compute_biomechanical_angles(
        self, landmarks: List[LandmarkPoint]
    ) -> Dict[str, Optional[float]]:
        """Compute kinematic angles for ergonomic inspection.

        These angles provide explanatory insight into knee flexion and trunk inclination,
        supplementing the trained CNN prediction.
        """
        lm_map = {lm.id: (lm.x, lm.y) for lm in landmarks}
        vis_map = {lm.id: lm.visibility for lm in landmarks}

        def is_vis(*ids: int) -> bool:
            return all(vis_map.get(i, 0.0) >= config.MIN_LANDMARK_VISIBILITY for i in ids)

        angles: Dict[str, Optional[float]] = {
            "knee_angle_left": None,
            "knee_angle_right": None,
            "knee_angle_mean": None,
            "trunk_inclination": None,
            "hip_angle_left": None,
            "hip_angle_right": None,
        }

        # Left Knee (Hip 23, Knee 25, Ankle 27)
        if is_vis(23, 25, 27):
            angles["knee_angle_left"] = round(calculate_2d_angle(lm_map[23], lm_map[25], lm_map[27]), 1)

        # Right Knee (Hip 24, Knee 26, Ankle 28)
        if is_vis(24, 26, 28):
            angles["knee_angle_right"] = round(calculate_2d_angle(lm_map[24], lm_map[26], lm_map[28]), 1)

        # Mean knee flexion
        k_angles = [a for a in [angles["knee_angle_left"], angles["knee_angle_right"]] if a is not None]
        if k_angles:
            angles["knee_angle_mean"] = round(float(np.mean(k_angles)), 1)

        # Trunk Inclination: Mid-shoulder to Mid-hip vector vs vertical
        if is_vis(11, 23) or is_vis(12, 24):
            # Use visible side or average midpoint
            sh_pts = [lm_map[i] for i in [11, 12] if vis_map.get(i, 0.0) >= config.MIN_LANDMARK_VISIBILITY]
            hp_pts = [lm_map[i] for i in [23, 24] if vis_map.get(i, 0.0) >= config.MIN_LANDMARK_VISIBILITY]
            if sh_pts and hp_pts:
                mid_sh = (float(np.mean([p[0] for p in sh_pts])), float(np.mean([p[1] for p in sh_pts])))
                mid_hp = (float(np.mean([p[0] for p in hp_pts])), float(np.mean([p[1] for p in hp_pts])))
                angles["trunk_inclination"] = round(calculate_trunk_inclination(mid_sh, mid_hp), 1)

        # Left Hip (Shoulder 11, Hip 23, Knee 25)
        if is_vis(11, 23, 25):
            angles["hip_angle_left"] = round(calculate_2d_angle(lm_map[11], lm_map[23], lm_map[25]), 1)

        # Right Hip (Shoulder 12, Hip 24, Knee 26)
        if is_vis(12, 24, 26):
            angles["hip_angle_right"] = round(calculate_2d_angle(lm_map[12], lm_map[24], lm_map[26]), 1)

        return angles

    def close(self) -> None:
        """Release underlying MediaPipe landmarker resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
