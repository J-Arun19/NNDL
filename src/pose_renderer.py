"""Pose-map renderer module.

Renders normalized body landmarks and skeletal topology onto a standardized
128x128 single-channel canvas for CNN training and inference. Also provides
real-time visual HUD overlays on original video frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import config
from src.pose_extractor import LandmarkPoint


def save_preprocessing_config(target_path: Optional[Path] = None) -> Path:
    """Save the deterministic preprocessing configuration to ensure training/inference parity."""
    path = target_path or config.PREPROCESSING_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "canvas_height": config.POSE_MAP_HEIGHT,
        "canvas_width": config.POSE_MAP_WIDTH,
        "canvas_channels": config.POSE_MAP_CHANNELS,
        "joint_radius": config.JOINT_RADIUS,
        "bone_thickness": config.BONE_THICKNESS,
        "joint_intensity": config.JOINT_INTENSITY,
        "bone_intensity": config.BONE_INTENSITY,
        "background_intensity": config.BACKGROUND_INTENSITY,
        "min_visibility": config.MIN_LANDMARK_VISIBILITY,
        "critical_keypoints_ratio": config.MIN_CRITICAL_KEYPOINTS_RATIO,
        "skeletal_connections": [list(c) for c in config.SKELETAL_CONNECTIONS],
        "coordinate_normalization": "scaled_to_canvas_with_boundary_clamping"
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def load_preprocessing_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the preprocessing configuration, or return defaults if not found."""
    p = path or config.PREPROCESSING_CONFIG_PATH
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "canvas_height": config.POSE_MAP_HEIGHT,
        "canvas_width": config.POSE_MAP_WIDTH,
        "canvas_channels": config.POSE_MAP_CHANNELS,
        "min_visibility": config.MIN_LANDMARK_VISIBILITY,
    }


def normalize_and_clamp_point(
    x: float, y: float, width: int, height: int
) -> Tuple[int, int]:
    """Convert normalized [0.0, 1.0] coordinates to canvas integer pixel indices, clamped to boundaries."""
    px = int(np.clip(round(x * (width - 1)), 0, width - 1))
    py = int(np.clip(round(y * (height - 1)), 0, height - 1))
    return px, py


def render_pose_map(
    landmarks: Optional[Union[List[LandmarkPoint], List[Dict[str, Any]]]],
    height: int = config.POSE_MAP_HEIGHT,
    width: int = config.POSE_MAP_WIDTH,
    min_visibility: float = config.MIN_LANDMARK_VISIBILITY,
    as_float: bool = False,
) -> np.ndarray:
    """Render a deterministic single-channel pose map containing skeletal bones and joint nodes.

    Args:
        landmarks: List of LandmarkPoint objects or landmark dicts, or None if pose absent.
        height: Canvas height (default 128).
        width: Canvas width (default 128).
        min_visibility: Minimum visibility score required to render joint/bone.
        as_float: If True, returns float32 in range [0.0, 1.0]; otherwise uint8 [0, 255].

    Returns:
        np.ndarray of shape (height, width, 1).
    """
    canvas = np.zeros((height, width), dtype=np.uint8)

    if not landmarks:
        out = canvas[:, :, np.newaxis]
        return out.astype(np.float32) / 255.0 if as_float else out

    # Standardize input format
    lm_dict: Dict[int, Tuple[float, float, float]] = {}
    for item in landmarks:
        if isinstance(item, LandmarkPoint):
            lm_dict[item.id] = (item.x, item.y, item.visibility)
        elif isinstance(item, dict):
            idx = int(item.get("id", 0))
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            vis = float(item.get("visibility", 1.0))
            lm_dict[idx] = (x, y, vis)

    # 1. Draw skeletal connections (bones) first
    for start_idx, end_idx in config.SKELETAL_CONNECTIONS:
        if start_idx in lm_dict and end_idx in lm_dict:
            x1, y1, v1 = lm_dict[start_idx]
            x2, y2, v2 = lm_dict[end_idx]
            if v1 >= min_visibility and v2 >= min_visibility:
                pt1 = normalize_and_clamp_point(x1, y1, width, height)
                pt2 = normalize_and_clamp_point(x2, y2, width, height)
                cv2.line(
                    canvas,
                    pt1,
                    pt2,
                    color=int(config.BONE_INTENSITY),
                    thickness=config.BONE_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

    # 2. Draw joint nodes on top
    for idx, (x, y, vis) in lm_dict.items():
        if vis >= min_visibility:
            pt = normalize_and_clamp_point(x, y, width, height)
            cv2.circle(
                canvas,
                pt,
                radius=config.JOINT_RADIUS,
                color=int(config.JOINT_INTENSITY),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    # Expand channel dimension to (H, W, 1)
    out = canvas[:, :, np.newaxis]
    if as_float:
        return (out.astype(np.float32) / 255.0)
    return out


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    landmarks: Optional[Union[List[LandmarkPoint], List[Dict[str, Any]]]],
    status_label: str,
    probability: Optional[float] = None,
    angles: Optional[Dict[str, Optional[float]]] = None,
    fps: Optional[float] = None,
    uncertain_reason: Optional[str] = None,
    assessment_method: Optional[str] = None,
    explanation: Optional[str] = None,
) -> np.ndarray:
    """Draw a rich, color-coded visual HUD and skeletal overlay on the raw video frame.

    Colors:
    - SAFE POSTURE: Bright Emerald Green (BGR: 40, 200, 40)
    - UNSAFE POSTURE: Bright Coral Red (BGR: 40, 40, 230)
    - INSUFFICIENT VISIBILITY / UNCERTAIN: Amber Yellow (BGR: 40, 180, 230)
    """
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    # Select color theme based on classification
    clean_lbl = status_label.upper()
    if "SAFE" in clean_lbl and "UNSAFE" not in clean_lbl:
        primary_color = (40, 200, 40)    # Emerald Green
        bg_banner = (20, 80, 20)
    elif "UNSAFE" in clean_lbl:
        primary_color = (40, 40, 230)    # Coral Red
        bg_banner = (20, 20, 100)
    else:
        primary_color = (40, 180, 230)   # Amber Yellow
        bg_banner = (20, 70, 90)

    # 1. Render Skeleton on the frame if landmarks exist
    if landmarks:
        lm_dict: Dict[int, Tuple[int, int, float]] = {}
        for item in landmarks:
            if isinstance(item, LandmarkPoint):
                px = int(np.clip(round(item.x * (w - 1)), 0, w - 1))
                py = int(np.clip(round(item.y * (h - 1)), 0, h - 1))
                lm_dict[item.id] = (px, py, item.visibility)
            elif isinstance(item, dict):
                idx = int(item.get("id", 0))
                px = int(np.clip(round(float(item.get("x", 0)) * (w - 1)), 0, w - 1))
                py = int(np.clip(round(float(item.get("y", 0)) * (h - 1)), 0, h - 1))
                vis = float(item.get("visibility", 1.0))
                lm_dict[idx] = (px, py, vis)

        # Bones
        for start_idx, end_idx in config.SKELETAL_CONNECTIONS:
            if start_idx in lm_dict and end_idx in lm_dict:
                p1, p2 = lm_dict[start_idx], lm_dict[end_idx]
                if p1[2] >= config.MIN_LANDMARK_VISIBILITY and p2[2] >= config.MIN_LANDMARK_VISIBILITY:
                    cv2.line(annotated, (p1[0], p1[1]), (p2[0], p2[1]), primary_color, 3, cv2.LINE_AA)

        # Joint points
        for idx, (px, py, vis) in lm_dict.items():
            if vis >= config.MIN_LANDMARK_VISIBILITY:
                cv2.circle(annotated, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), 6, primary_color, 2, cv2.LINE_AA)

    # 2. Top Header HUD Bar
    hud_h = 74
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), bg_banner, -1)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

    # Status Pill / Label
    status_text = f"POSTURE: {status_label}"
    cv2.putText(annotated, status_text, (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, primary_color, 2, cv2.LINE_AA)

    # Subtitle with Assessment Method
    method_str = assessment_method or "Method: Pretrained MediaPipe Pose + Biomechanical Rules"
    cv2.putText(annotated, method_str[:55], (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 225), 1, cv2.LINE_AA)

    # Right side HUD info (probability or reason, and FPS)
    if probability is not None:
        prob_percent = int(probability * 100)
        prob_text = f"Unsafe Prob: {prob_percent}% (CNN)"
        cv2.putText(annotated, prob_text, (w - 360, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

        # Mini probability bar
        bar_x = w - 360
        bar_y = 44
        bar_w = 200
        bar_h = 14
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * np.clip(probability, 0.0, 1.0))
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), primary_color, -1)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    elif explanation:
        cv2.putText(annotated, f"Note: {explanation[:48]}", (max(20, w - 480), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    elif uncertain_reason:
        cv2.putText(annotated, f"Note: {uncertain_reason[:48]}", (max(20, w - 480), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    # FPS counter in HUD
    if fps is not None:
        cv2.putText(annotated, f"{fps:.1f} FPS", (w - 95, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # 3. Bottom Biomechanical Angle Insights
    if angles:
        bottom_box_h = 38
        y_bottom = h - 12
        trunk = angles.get("trunk_inclination")
        knee = angles.get("knee_angle_mean")

        angle_strs = []
        if trunk is not None:
            angle_strs.append(f"Trunk Inclination: {trunk:.1f} deg")
        if knee is not None:
            angle_strs.append(f"Knee Flexion: {knee:.1f} deg")

        if angle_strs:
            # Translucent ribbon at bottom
            ribbon = annotated.copy()
            cv2.rectangle(ribbon, (0, h - bottom_box_h), (w, h), (15, 15, 20), -1)
            cv2.addWeighted(ribbon, 0.65, annotated, 0.35, 0, annotated)

            text_line = "Biomechanical Features: " + " | ".join(angle_strs)
            cv2.putText(annotated, text_line, (20, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 240, 255), 1, cv2.LINE_AA)

    return annotated
