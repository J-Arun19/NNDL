"""Inference engine for real-time webcam streams and recorded videos.

Provides reliable SAFE/UNSAFE posture classification for both uploaded videos and live
webcam monitoring by coordinating pretrained MediaPipe Pose extraction, transparent
biomechanical ergonomic rules (NIOSH/OSHA), temporal rolling-window smoothing, alert
debouncing, and visual HUD overlays.

IMPORTANT SCIENTIFIC & ETHICAL DISCLOSURE:
Classification is based on documented ergonomic guidelines (torso inclination relative
to vertical, knee flexion angle, and landmark visibility quality) using Google's pretrained
MediaPipe Pose model for body keypoints. It serves as an occupational screening aid, NOT
as a certified clinical medical diagnosis.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

import config
from src.pose_extractor import PoseExtractionResult, PoseExtractor
from src.pose_renderer import draw_pose_overlay, render_pose_map

logger = logging.getLogger(__name__)

# Standardized Posture Classification Status Tokens
STATUS_SAFE_POSTURE = "SAFE POSTURE"
STATUS_UNSAFE_POSTURE = "UNSAFE POSTURE"
STATUS_INSUFFICIENT_VISIBILITY = "INSUFFICIENT VISIBILITY"

# Assessment Mode Tokens
MODE_RULE_BASED_ERGONOMICS = "Rule-Based Ergonomic Assessment"
MODE_CNN_MODEL = "Trained CNN"


@dataclass
class FrameInferenceResult:
    """Detailed posture classification result for an individual video frame."""

    frame_index: int
    timestamp_sec: float
    raw_status: str              # SAFE POSTURE, UNSAFE POSTURE, or INSUFFICIENT VISIBILITY
    smoothed_status: str         # Smoothed rolling-window status
    raw_probability: Optional[float]
    smoothed_probability: Optional[float]
    is_alert_triggered: bool     # True if debounced unsafe threshold is exceeded
    is_valid_pose: bool          # True if body landmarks meet quality gates
    pose_quality_message: str
    biomechanical_angles: Dict[str, Optional[float]]
    explanation: str             # Clear ergonomic reason for the classification
    assessment_method: str       # Transparent disclosure of method used
    inference_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAnalysisSummary:
    """Overall summary of posture analysis across a complete video clip."""

    total_frames: int
    valid_pose_frames: int
    safe_frames: int
    unsafe_frames: int
    insufficient_visibility_frames: int
    overall_classification: str  # SAFE POSTURE, UNSAFE POSTURE, or INSUFFICIENT VISIBILITY
    primary_reason: str          # Main ergonomic rationale for overall verdict
    mean_trunk_inclination_deg: Optional[float]
    max_trunk_inclination_deg: Optional[float]
    mean_knee_angle_deg: Optional[float]
    min_knee_angle_deg: Optional[float]
    valid_pose_pct: float
    safe_pct_of_valid: float
    unsafe_pct_of_valid: float
    duration_sec: float
    assessment_method: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiftingPostureClassifier:
    """Stateful, real-time classifier for manual lifting posture screening.

    Supports both pretrained pose-based ergonomic rules and optional custom-trained CNNs.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        rolling_window_size: int = config.DEFAULT_ROLLING_WINDOW,
        alert_debounce_threshold: int = config.DEFAULT_ALERT_DEBOUNCE,
        force_rule_based: bool = False,
    ):
        self.rolling_window_size = max(1, rolling_window_size)
        self.alert_debounce_threshold = max(1, alert_debounce_threshold)
        self.force_rule_based = force_rule_based

        # Initialize Pretrained MediaPipe Pose Extractor
        self.extractor = PoseExtractor()

        # Check for trained CNN model checkpoint
        self.model = None
        self.decision_threshold = config.DEFAULT_DECISION_THRESHOLD
        self.model_metadata: Dict[str, Any] = {}
        if not self.force_rule_based:
            self._load_model_weights(model_path, config_path)

        # Stateful buffers for temporal rolling-window smoothing
        self.status_buffer: Deque[str] = collections.deque(maxlen=self.rolling_window_size)
        self.prob_buffer: Deque[float] = collections.deque(maxlen=self.rolling_window_size)
        self.consecutive_unsafe_frames: int = 0

    def _load_model_weights(
        self, model_path: Optional[Path], config_path: Optional[Path]
    ) -> None:
        """Safely attempt to load Keras model weights if available."""
        m_path = model_path or config.BEST_MODEL_PATH
        c_path = config_path or config.MODEL_CONFIG_PATH

        if m_path.exists():
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(str(m_path))
                logger.info(f"Loaded trained CNN model from {m_path}")
            except Exception as exc:
                logger.warning(f"Could not load model from {m_path}: {exc}")
                self.model = None

        if c_path.exists():
            try:
                import json
                with open(c_path, "r", encoding="utf-8") as f:
                    self.model_metadata = json.load(f)
                self.decision_threshold = float(
                    self.model_metadata.get("decision_threshold", config.DEFAULT_DECISION_THRESHOLD)
                )
            except Exception as exc:
                logger.warning(f"Could not load config from {c_path}: {exc}")

    @property
    def is_model_trained(self) -> bool:
        """Check whether a trained CNN checkpoint is actively loaded."""
        return self.model is not None and not self.force_rule_based

    @property
    def current_assessment_method(self) -> str:
        """Return the active assessment method description."""
        if self.is_model_trained:
            return MODE_CNN_MODEL
        return MODE_RULE_BASED_ERGONOMICS

    def reset_state(self) -> None:
        """Reset temporal smoothing buffers when starting a new session or video."""
        self.status_buffer.clear()
        self.prob_buffer.clear()
        self.consecutive_unsafe_frames = 0

    def evaluate_frame_ergonomics(
        self,
        extraction_res: PoseExtractionResult,
    ) -> Tuple[str, str]:
        """Evaluate transparent ergonomic posture rules on visible body keypoints.

        Returns:
            Tuple of (status_token, explanation_string)
        """
        if not extraction_res.is_valid:
            return (
                STATUS_INSUFFICIENT_VISIBILITY,
                f"Body landmarks unreliable or occluded: {extraction_res.status_message}"
            )

        angles = extraction_res.biomechanical_angles
        trunk_inc = angles.get("trunk_inclination")
        knee_mean = angles.get("knee_angle_mean")

        # Fallback to single visible knee if mean is None
        if knee_mean is None:
            k_left = angles.get("knee_angle_left")
            k_right = angles.get("knee_angle_right")
            k_valid = [k for k in (k_left, k_right) if k is not None]
            if k_valid:
                knee_mean = float(np.mean(k_valid))

        # Check if critical angles could be calculated
        if trunk_inc is None or knee_mean is None:
            return (
                STATUS_INSUFFICIENT_VISIBILITY,
                "Critical lifting angles (trunk or knee) could not be calculated from visible joints."
            )

        # -------------------------------------------------------------
        # Ergonomic Rules (NIOSH / OSHA Manual Material Handling):
        # 1. Trunk Inclination: Angle of hip-to-shoulder vector from vertical.
        #    <= 45°: Upright trunk (safe lumbar alignment).
        #    > 45°: Severe forward torso flexion (compressive shear at L5/S1).
        # 2. Knee Flexion: Angle at knee vertex between hip and ankle.
        #    <= 125°: Knees flexed (squat lift, lower extremity drive).
        #    >= 135°: Knees extended (stoop lift, straight-leg bending).
        # -------------------------------------------------------------

        # Case 1: Clear Squat Lift (SAFE POSTURE)
        if trunk_inc <= config.MAX_SAFE_TRUNK_FLEXION_DEG and knee_mean <= 130.0:
            return (
                STATUS_SAFE_POSTURE,
                f"Safe squat mechanics: upright torso ({trunk_inc:.1f}° <= 45°) with flexed knees "
                f"({knee_mean:.1f}°); load absorbed by leg musculature rather than spine."
            )

        # Case 2: Classic Stoop Lift (UNSAFE POSTURE)
        if trunk_inc > config.MAX_SAFE_TRUNK_FLEXION_DEG and knee_mean >= 135.0:
            return (
                STATUS_UNSAFE_POSTURE,
                f"Excessive forward trunk bending ({trunk_inc:.1f}° > 45.0°) with extended knees "
                f"({knee_mean:.1f}° >= 135.0°); creates long lever arm and high lumbosacral shear at L5/S1."
            )

        # Case 3: Severe Forward Trunk Flexion (UNSAFE POSTURE)
        if trunk_inc > 50.0:
            return (
                STATUS_UNSAFE_POSTURE,
                f"Severe forward torso flexion ({trunk_inc:.1f}° > 45.0°); lumbar spine bears excessive "
                f"compressive load regardless of knee angle."
            )

        # Case 4: Straight-Leg Stiff Bending (UNSAFE POSTURE)
        if knee_mean > 145.0 and trunk_inc > 38.0:
            return (
                STATUS_UNSAFE_POSTURE,
                f"Legs nearly straight ({knee_mean:.1f}°); minimal knee bending shifts lifting effort directly "
                f"onto lower back."
            )

        # Case 5: Transitional / Intermediate Mechanics
        if trunk_inc > config.MAX_SAFE_TRUNK_FLEXION_DEG:
            return (
                STATUS_UNSAFE_POSTURE,
                f"Semi-stoop mechanics: forward torso flexion ({trunk_inc:.1f}°) exceeds 45° ergonomic threshold."
            )

        # Default within upright boundary
        return (
            STATUS_SAFE_POSTURE,
            f"Upright posture: trunk inclination ({trunk_inc:.1f}°) remains within 45° ergonomic boundary."
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
    ) -> Tuple[FrameInferenceResult, np.ndarray]:
        """Perform posture analysis on an individual BGR video frame.

        Returns:
            Tuple of (FrameInferenceResult, annotated_frame_bgr)
        """
        t_start = time.perf_counter()

        if frame_bgr is None or frame_bgr.size == 0:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            res = FrameInferenceResult(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                raw_status=STATUS_INSUFFICIENT_VISIBILITY,
                smoothed_status=STATUS_INSUFFICIENT_VISIBILITY,
                raw_probability=None,
                smoothed_probability=None,
                is_alert_triggered=False,
                is_valid_pose=False,
                pose_quality_message="Empty image frame provided.",
                biomechanical_angles={},
                explanation="No image data received.",
                assessment_method=self.current_assessment_method,
                inference_latency_ms=round(latency_ms, 2),
            )
            return res, frame_bgr

        # 1. Pose Extraction & Quality Verification
        extraction_res: PoseExtractionResult = self.extractor.extract_from_frame(frame_bgr)

        # 2. Posture Classification (CNN or Ergonomic Rules)
        raw_prob: Optional[float] = None
        raw_status: str
        explanation: str

        if not extraction_res.is_valid:
            raw_status = STATUS_INSUFFICIENT_VISIBILITY
            explanation = extraction_res.status_message
        elif self.is_model_trained:
            # CNN Model Evaluation
            pose_map = render_pose_map(extraction_res.landmarks, as_float=True)
            model_input = np.expand_dims(pose_map, axis=0)
            raw_prob = float(self.model(model_input, training=False)[0][0].numpy())
            raw_status = STATUS_UNSAFE_POSTURE if raw_prob >= self.decision_threshold else STATUS_SAFE_POSTURE
            explanation = (
                f"CNN model inference: unsafe probability {raw_prob * 100:.1f}% "
                f"(threshold: {self.decision_threshold * 100:.0f}%)."
            )
        else:
            # Transparent Ergonomic Biomechanical Evaluation
            raw_status, explanation = self.evaluate_frame_ergonomics(extraction_res)

        # 3. Temporal Smoothing (Rolling Window Majority Vote)
        self.status_buffer.append(raw_status)
        if raw_prob is not None:
            self.prob_buffer.append(raw_prob)

        # Determine smoothed status from rolling buffer
        status_counts = collections.Counter(self.status_buffer)
        smoothed_status = status_counts.most_common(1)[0][0]

        smoothed_prob: Optional[float] = None
        if self.prob_buffer:
            smoothed_prob = round(float(np.mean(self.prob_buffer)), 4)

        # 4. Alert Debouncing
        if smoothed_status == STATUS_UNSAFE_POSTURE:
            self.consecutive_unsafe_frames += 1
        else:
            self.consecutive_unsafe_frames = 0

        is_alert = self.consecutive_unsafe_frames >= self.alert_debounce_threshold

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        result = FrameInferenceResult(
            frame_index=frame_index,
            timestamp_sec=round(timestamp_sec, 2),
            raw_status=raw_status,
            smoothed_status=smoothed_status,
            raw_probability=round(raw_prob, 4) if raw_prob is not None else None,
            smoothed_probability=smoothed_prob,
            is_alert_triggered=is_alert,
            is_valid_pose=extraction_res.is_valid,
            pose_quality_message=extraction_res.status_message,
            biomechanical_angles=extraction_res.biomechanical_angles,
            explanation=explanation,
            assessment_method=self.current_assessment_method,
            inference_latency_ms=round(latency_ms, 2),
        )

        # 5. Render HUD Overlay
        annotated_frame = draw_pose_overlay(
            frame_bgr=frame_bgr,
            landmarks=extraction_res.landmarks,
            status_label=smoothed_status,
            probability=smoothed_prob,
            angles=extraction_res.biomechanical_angles,
            fps=(1000.0 / latency_ms) if latency_ms > 0 else 0.0,
            uncertain_reason=explanation if smoothed_status == STATUS_INSUFFICIENT_VISIBILITY else None,
            assessment_method=self.current_assessment_method,
            explanation=explanation if smoothed_status != STATUS_INSUFFICIENT_VISIBILITY else None,
        )

        return result, annotated_frame

    def close(self) -> None:
        """Release underlying pose extractor resources."""
        if hasattr(self, "extractor") and self.extractor is not None:
            self.extractor.close()


def process_recorded_video(
    video_path: Path,
    output_annotated_video_path: Optional[Path] = None,
    output_predictions_csv_path: Optional[Path] = None,
    classifier: Optional[LiftingPostureClassifier] = None,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Optional[Path], VideoAnalysisSummary]:
    """Process an uploaded or recorded video file and return predictions, video, and summary.

    Supports MP4, AVI, and MOV video containers.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    clf = classifier or LiftingPostureClassifier()
    clf.reset_state()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file with OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if max_frames and max_frames < total_frames:
        total_frames = max_frames

    out_writer = None
    if output_annotated_video_path:
        output_annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        # mp4v format for broad cross-platform compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            str(output_annotated_video_path), fourcc, fps, (width, height)
        )

    records: List[Dict[str, Any]] = []
    frame_idx = 0

    trunk_angles: List[float] = []
    knee_angles: List[float] = []
    safe_count = 0
    unsafe_count = 0
    insufficient_vis_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        timestamp = round(frame_idx / fps, 2) if fps > 0 else 0.0
        res, annotated = clf.process_frame(frame, frame_index=frame_idx, timestamp_sec=timestamp)

        if out_writer:
            out_writer.write(annotated)

        # Track statistics
        if res.smoothed_status == STATUS_SAFE_POSTURE:
            safe_count += 1
        elif res.smoothed_status == STATUS_UNSAFE_POSTURE:
            unsafe_count += 1
        else:
            insufficient_vis_count += 1

        trunk_val = res.biomechanical_angles.get("trunk_inclination")
        if trunk_val is not None:
            trunk_angles.append(trunk_val)

        knee_val = res.biomechanical_angles.get("knee_angle_mean")
        if knee_val is not None:
            knee_angles.append(knee_val)

        records.append({
            "frame_index": res.frame_index,
            "timestamp_sec": res.timestamp_sec,
            "status": res.smoothed_status,
            "raw_status": res.raw_status,
            "explanation": res.explanation,
            "is_valid_pose": res.is_valid_pose,
            "trunk_inclination_deg": trunk_val,
            "knee_angle_mean_deg": knee_val,
            "alert_triggered": res.is_alert_triggered,
            "probability": res.smoothed_probability,
            "latency_ms": res.inference_latency_ms,
        })

        frame_idx += 1
        if progress_callback and total_frames > 0:
            progress_callback(min(1.0, frame_idx / total_frames))

    cap.release()
    if out_writer:
        out_writer.release()

    df_preds = pd.DataFrame(records)
    if output_predictions_csv_path:
        output_predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_csv(output_predictions_csv_path, index=False)

    # Compute Overall Video Verdict
    total_processed = len(records)
    valid_poses = safe_count + unsafe_count
    valid_pose_pct = round((valid_poses / total_processed * 100.0), 1) if total_processed > 0 else 0.0
    safe_pct_of_valid = round((safe_count / valid_poses * 100.0), 1) if valid_poses > 0 else 0.0
    unsafe_pct_of_valid = round((unsafe_count / valid_poses * 100.0), 1) if valid_poses > 0 else 0.0
    dur_sec = round(total_processed / fps, 2) if fps > 0 else 0.0

    mean_trunk = round(float(np.mean(trunk_angles)), 1) if trunk_angles else None
    max_trunk = round(float(np.max(trunk_angles)), 1) if trunk_angles else None
    mean_knee = round(float(np.mean(knee_angles)), 1) if knee_angles else None
    min_knee = round(float(np.min(knee_angles)), 1) if knee_angles else None

    # Overall Classification Logic
    if valid_pose_pct < 25.0:
        overall_classification = STATUS_INSUFFICIENT_VISIBILITY
        primary_reason = (
            f"Insufficient landmark visibility ({valid_pose_pct}% valid frames). Lifter body joints were "
            f"frequently occluded or outside camera field of view."
        )
    elif unsafe_pct_of_valid >= 20.0 or (max_trunk is not None and max_trunk > 55.0):
        overall_classification = STATUS_UNSAFE_POSTURE
        reasons_list = []
        if max_trunk is not None and max_trunk > 45.0:
            reasons_list.append(f"peak trunk flexion reached {max_trunk:.1f}° (safe threshold <= 45.0°)")
        if mean_knee is not None and mean_knee >= 135.0:
            reasons_list.append(f"legs remained extended (mean knee angle {mean_knee:.1f}° >= 135.0°)")
        if not reasons_list:
            reasons_list.append(f"unsafe stoop posture was detected in {unsafe_pct_of_valid:.1f}% of active frames")
        primary_reason = (
            f"Unsafe posture detected: {', and '.join(reasons_list)}. Concentrates compressive shear on the L5/S1 discs."
        )
    else:
        overall_classification = STATUS_SAFE_POSTURE
        primary_reason = (
            f"Proper squat lifting mechanics maintained across {safe_pct_of_valid:.1f}% of valid frames. "
            f"Mean trunk inclination was {mean_trunk or 0.0:.1f}° (<= 45° threshold) with active knee flexion."
        )

    summary = VideoAnalysisSummary(
        total_frames=total_processed,
        valid_pose_frames=valid_poses,
        safe_frames=safe_count,
        unsafe_frames=unsafe_count,
        insufficient_visibility_frames=insufficient_vis_count,
        overall_classification=overall_classification,
        primary_reason=primary_reason,
        mean_trunk_inclination_deg=mean_trunk,
        max_trunk_inclination_deg=max_trunk,
        mean_knee_angle_deg=mean_knee,
        min_knee_angle_deg=min_knee,
        valid_pose_pct=valid_pose_pct,
        safe_pct_of_valid=safe_pct_of_valid,
        unsafe_pct_of_valid=unsafe_pct_of_valid,
        duration_sec=dur_sec,
        assessment_method=clf.current_assessment_method,
    )

    return df_preds, output_annotated_video_path, summary
