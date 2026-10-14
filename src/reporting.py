"""Reporting module for posture screening sessions and video analysis.

Aggregates frame-by-frame predictions, calculates ergonomic risk scores,
and exports structured session summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class SessionReport:
    """Aggregated metrics and ergonomic risk rating for a video screening session."""
    total_frames: int
    duration_sec: float
    valid_pose_frames: int
    valid_pose_pct: float
    safe_frames: int
    safe_pct_of_valid: float
    unsafe_frames: int
    unsafe_pct_of_valid: float
    uncertain_frames: int
    uncertain_pct: float
    mean_latency_ms: float
    ergonomic_risk_level: str
    risk_summary_text: str
    mean_trunk_flexion_deg: Optional[float]
    mean_knee_angle_deg: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
            "valid_pose_frames": self.valid_pose_frames,
            "valid_pose_percentage": self.valid_pose_pct,
            "safe_frames": self.safe_frames,
            "safe_pct_of_valid": self.safe_pct_of_valid,
            "unsafe_frames": self.unsafe_frames,
            "unsafe_pct_of_valid": self.unsafe_pct_of_valid,
            "uncertain_frames": self.uncertain_frames,
            "uncertain_percentage": self.uncertain_pct,
            "mean_latency_ms": self.mean_latency_ms,
            "ergonomic_risk_level": self.ergonomic_risk_level,
            "risk_summary_text": self.risk_summary_text,
            "mean_trunk_flexion_deg": self.mean_trunk_flexion_deg,
            "mean_knee_angle_deg": self.mean_knee_angle_deg,
        }


def generate_session_report(df_predictions: pd.DataFrame) -> SessionReport:
    """Aggregate a session DataFrame into a formal ergonomic screening report."""
    if df_predictions.empty:
        return SessionReport(
            total_frames=0,
            duration_sec=0.0,
            valid_pose_frames=0,
            valid_pose_pct=0.0,
            safe_frames=0,
            safe_pct_of_valid=0.0,
            unsafe_frames=0,
            unsafe_pct_of_valid=0.0,
            uncertain_frames=0,
            uncertain_pct=0.0,
            mean_latency_ms=0.0,
            ergonomic_risk_level="INSUFFICIENT DATA",
            risk_summary_text="No frames were recorded in this session.",
            mean_trunk_flexion_deg=None,
            mean_knee_angle_deg=None,
        )

    total_frames = len(df_predictions)
    duration_sec = float(df_predictions["timestamp_sec"].max() - df_predictions["timestamp_sec"].min()) if "timestamp_sec" in df_predictions.columns else 0.0

    valid_mask = df_predictions["is_valid_pose"] == True
    valid_count = int(valid_mask.sum())
    valid_pct = round((valid_count / total_frames) * 100.0, 1)

    uncertain_count = total_frames - valid_count
    uncertain_pct = round((uncertain_count / total_frames) * 100.0, 1)

    df_valid = df_predictions[valid_mask]

    status_series = df_valid["status"].astype(str)
    safe_count = int((status_series.str.contains("SAFE", case=False, na=False) & ~status_series.str.contains("UNSAFE", case=False, na=False)).sum())
    unsafe_count = int(status_series.str.contains("UNSAFE", case=False, na=False).sum())

    if valid_count > 0:
        safe_pct = round((safe_count / valid_count) * 100.0, 1)
        unsafe_pct = round((unsafe_count / valid_count) * 100.0, 1)
    else:
        safe_pct = 0.0
        unsafe_pct = 0.0

    mean_latency = round(float(df_predictions["latency_ms"].mean()), 2) if "latency_ms" in df_predictions.columns else 0.0

    # Ergonomic risk categorization based on unsafe lifting duration
    if unsafe_pct < 15.0:
        risk_level = "LOW ERGONOMIC RISK"
        summary = "Predominantly safe squat lifting posture observed with proper knee flexion and upright trunk."
    elif unsafe_pct < 40.0:
        risk_level = "MODERATE ERGONOMIC RISK"
        summary = "Intermittent unsafe lifting mechanics detected. Operator bends forward at waist occasionally."
    else:
        risk_level = "HIGH ERGONOMIC RISK"
        summary = "Frequent or continuous stoop lifting detected. Excessive spinal shear forces indicated."

    mean_trunk = None
    if "trunk_inclination_deg" in df_valid.columns:
        valid_trunk = df_valid["trunk_inclination_deg"].dropna()
        if not valid_trunk.empty:
            mean_trunk = round(float(valid_trunk.mean()), 1)

    mean_knee = None
    if "knee_angle_mean_deg" in df_valid.columns:
        valid_knee = df_valid["knee_angle_mean_deg"].dropna()
        if not valid_knee.empty:
            mean_knee = round(float(valid_knee.mean()), 1)

    return SessionReport(
        total_frames=total_frames,
        duration_sec=round(duration_sec, 2),
        valid_pose_frames=valid_count,
        valid_pose_pct=valid_pct,
        safe_frames=safe_count,
        safe_pct_of_valid=safe_pct,
        unsafe_frames=unsafe_count,
        unsafe_pct_of_valid=unsafe_pct,
        uncertain_frames=uncertain_count,
        uncertain_pct=uncertain_pct,
        mean_latency_ms=mean_latency,
        ergonomic_risk_level=risk_level,
        risk_summary_text=summary,
        mean_trunk_flexion_deg=mean_trunk,
        mean_knee_angle_deg=mean_knee,
    )
