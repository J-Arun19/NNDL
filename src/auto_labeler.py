"""Semi-automated dataset labeling pipeline for SAFELIFT Environment 4 videos.

Extracts kinematic posture indicators across the annotated action interval using
MediaPipe Pose, evaluates documented ergonomic screening heuristics, and generates
provisional label proposals in data/automatic_label_candidates.csv.

IMPORTANT SCIENTIFIC AND ETHICAL DISCLOSURES:
1. Provisional Screening Only: Heuristic rules are based on biomechanical lifting
   guidelines (NIOSH / OSHA) and serve strictly as preliminary screening aids. They
   are NOT certified ergonomic evaluations or clinical medical diagnoses.
2. Participant Pool Limitation: The SAFELIFT Environment 4 subset comprises only 4
   participants (Subjects A, B, C, D). A 4-person dataset is too small to claim
   generalizable, participant-independent model performance. Individual anthropometric
   variations (limb proportions, personal movement styles) can easily confound
   cross-subject generalization.
3. Separation of Concerns: Proposed candidate labels are stored in a dedicated file
   (data/automatic_label_candidates.csv) and are NEVER automatically marked as
   manually reviewed or treated as ground truth without human verification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import config
from src.dataset_review import DatasetReviewManager, MatchedVideoMetadata
from src.pose_extractor import (
    LandmarkPoint,
    PoseExtractionResult,
    PoseExtractor,
    calculate_2d_angle,
    calculate_trunk_inclination,
)

logger = logging.getLogger(__name__)

# Proposed Label Tokens
PROP_LABEL_SAFE = "SAFE"
PROP_LABEL_UNSAFE = "UNSAFE"
PROP_LABEL_AMBIGUOUS = "AMBIGUOUS"

# Processing Status Tokens
STATUS_SUCCESS = "SUCCESS"
STATUS_LOW_COVERAGE = "LOW_COVERAGE"
STATUS_LOW_VISIBILITY = "LOW_VISIBILITY"
STATUS_HIGH_ASYMMETRY = "HIGH_ASYMMETRY"
STATUS_NO_POSE_DETECTED = "NO_POSE_DETECTED"
STATUS_VIDEO_NOT_FOUND = "VIDEO_NOT_FOUND"
STATUS_ERROR = "ERROR"

DISCLAIMER_TEXT = (
    "Provisional screening candidate based on 2D kinematic heuristics. "
    "Not a clinical diagnosis or certified ergonomic assessment."
)


@dataclass
class LabelingRulesConfig:
    """Configurable threshold parameters for ergonomic screening heuristics."""

    # Pose Quality & Coverage Gates
    min_coverage_pct: float = 35.0          # Min percentage of interval frames with valid pose
    min_valid_frames: int = 5               # Min absolute number of valid frames required
    min_critical_visibility: float = 0.55   # Min mean visibility of critical lifting joints

    # Squat / SAFE Lift Criteria (Legs bent, spine upright)
    squat_max_trunk_inclination: float = 45.0   # Torso max angle from vertical (degrees)
    squat_mean_trunk_inclination: float = 35.0  # Torso mean angle from vertical (degrees)
    squat_min_knee_angle: float = 120.0         # Deepest knee flexion (degrees: <90=deep, <120=moderate)
    squat_mean_knee_angle: float = 135.0        # Mean knee flexion across lift

    # Stoop / UNSAFE Lift Criteria (Legs straight, spine deeply flexed)
    stoop_min_trunk_inclination: float = 45.0   # Torso max angle from vertical (degrees)
    stoop_mean_trunk_inclination: float = 38.0  # Torso mean angle from vertical (degrees)
    stoop_min_knee_angle: float = 135.0         # Knees stay extended (degrees: >135 = nearly straight)
    stoop_mean_knee_angle: float = 145.0        # Mean knee extension across lift

    # Asymmetry Gate
    max_asymmetry_deg: float = 30.0             # Rotational asymmetry angle threshold (degrees)

    # Frame sampling stride (frames to skip between extractions for efficiency)
    sample_step_frames: int = 2                 # Sample every 2nd frame (~12-15 fps from 25-30 fps video)


@dataclass
class VideoBiomechanicalMetrics:
    """Measured kinematic and visibility indicators across an action interval."""

    video_filename: str
    total_frames_sampled: int
    valid_pose_frames: int
    pose_coverage_pct: float
    mean_critical_visibility: float
    trunk_inclination_mean: Optional[float]
    trunk_inclination_max: Optional[float]
    knee_angle_min: Optional[float]
    knee_angle_mean: Optional[float]
    angle_of_asymmetry_deg: float
    camera_view: str
    action_direction: str


@dataclass
class CandidateProposalResult:
    """Outcome of rule-based proposal evaluation for a video."""

    video_filename: str
    proposed_label: str
    processing_status: str
    rule_outcomes: str
    ambiguity_reason: str
    metrics: Optional[VideoBiomechanicalMetrics]
    disclaimer: str = DISCLAIMER_TEXT

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "video_filename": self.video_filename,
            "proposed_label": self.proposed_label,
            "processing_status": self.processing_status,
            "rule_outcomes": self.rule_outcomes,
            "ambiguity_reason": self.ambiguity_reason,
            "disclaimer": self.disclaimer,
        }
        if self.metrics:
            d.update({
                "total_frames_sampled": self.metrics.total_frames_sampled,
                "valid_pose_frames": self.metrics.valid_pose_frames,
                "pose_coverage_pct": round(self.metrics.pose_coverage_pct, 1),
                "mean_critical_visibility": round(self.metrics.mean_critical_visibility, 3),
                "trunk_inclination_mean": self.metrics.trunk_inclination_mean,
                "trunk_inclination_max": self.metrics.trunk_inclination_max,
                "knee_angle_min": self.metrics.knee_angle_min,
                "knee_angle_mean": self.metrics.knee_angle_mean,
                "angle_of_asymmetry_deg": self.metrics.angle_of_asymmetry_deg,
                "camera_view": self.metrics.camera_view,
                "action_direction": self.metrics.action_direction,
            })
        else:
            d.update({
                "total_frames_sampled": 0,
                "valid_pose_frames": 0,
                "pose_coverage_pct": 0.0,
                "mean_critical_visibility": 0.0,
                "trunk_inclination_mean": None,
                "trunk_inclination_max": None,
                "knee_angle_min": None,
                "knee_angle_mean": None,
                "angle_of_asymmetry_deg": 0.0,
                "camera_view": "unknown",
                "action_direction": "unknown",
            })
        return d


def evaluate_ergonomic_rules(
    metrics: VideoBiomechanicalMetrics,
    config_rules: Optional[LabelingRulesConfig] = None,
) -> Tuple[str, str, str, str]:
    """Apply transparent biomechanical rules to compute candidate label.

    Returns:
        (proposed_label, processing_status, rule_outcomes, ambiguity_reason)
    """
    rules = config_rules or LabelingRulesConfig()

    # Gate 1: Check coverage and minimum frame sample size
    if metrics.total_frames_sampled == 0:
        return (
            PROP_LABEL_AMBIGUOUS,
            STATUS_NO_POSE_DETECTED,
            "No frames could be extracted from video.",
            "Video stream was empty or unreadable.",
        )

    if metrics.valid_pose_frames < rules.min_valid_frames or metrics.pose_coverage_pct < rules.min_coverage_pct:
        reason = (
            f"Insufficient pose coverage: only {metrics.valid_pose_frames}/{metrics.total_frames_sampled} "
            f"valid frames ({metrics.pose_coverage_pct:.1f}% < {rules.min_coverage_pct}% required)."
        )
        return (
            PROP_LABEL_AMBIGUOUS,
            STATUS_LOW_COVERAGE,
            "Fails minimum pose coverage gate.",
            reason,
        )

    # Gate 2: Landmark visibility check
    if metrics.mean_critical_visibility < rules.min_critical_visibility:
        reason = (
            f"Low keypoint visibility: average critical landmark visibility was "
            f"{metrics.mean_critical_visibility:.2f} (< {rules.min_critical_visibility} required)."
        )
        return (
            PROP_LABEL_AMBIGUOUS,
            STATUS_LOW_VISIBILITY,
            "Fails critical joint visibility gate.",
            reason,
        )

    # Gate 3: Check measurements presence
    if (
        metrics.trunk_inclination_mean is None
        or metrics.trunk_inclination_max is None
        or metrics.knee_angle_min is None
        or metrics.knee_angle_mean is None
    ):
        return (
            PROP_LABEL_AMBIGUOUS,
            STATUS_LOW_VISIBILITY,
            "Incomplete kinematic angle set.",
            "Could not reliably calculate both trunk inclination and knee angle across the lift.",
        )

    t_mean = metrics.trunk_inclination_mean
    t_max = metrics.trunk_inclination_max
    k_min = metrics.knee_angle_min
    k_mean = metrics.knee_angle_mean

    # Gate 4: Extreme Asymmetry Check
    if metrics.angle_of_asymmetry_deg > rules.max_asymmetry_deg:
        reason = (
            f"High rotational asymmetry ({metrics.angle_of_asymmetry_deg:.1f}° > {rules.max_asymmetry_deg}°); "
            f"introduces 3D torsional spine loading not fully capturable in 2D."
        )
        return (
            PROP_LABEL_AMBIGUOUS,
            STATUS_HIGH_ASYMMETRY,
            "Excessive twisting/asymmetry during lift.",
            reason,
        )

    # Gate 5: Squat Lift Candidate (SAFE)
    # Knees flexed (k_min <= squat_min_knee_angle AND k_mean <= squat_mean_knee_angle)
    # AND Trunk upright (t_mean <= squat_mean_trunk_inclination AND t_max <= squat_max_trunk_inclination)
    is_squat_knees = (k_min <= rules.squat_min_knee_angle) and (k_mean <= rules.squat_mean_knee_angle)
    is_squat_trunk = (t_mean <= rules.squat_mean_trunk_inclination) and (t_max <= rules.squat_max_trunk_inclination)

    if is_squat_knees and is_squat_trunk:
        outcome = (
            f"Knees actively flexed (min={k_min:.1f}°, mean={k_mean:.1f}°) with upright trunk "
            f"(max={t_max:.1f}°, mean={t_mean:.1f}°)."
        )
        return (PROP_LABEL_SAFE, STATUS_SUCCESS, outcome, "")

    # Gate 6: Stoop Lift Candidate (UNSAFE)
    # Knees extended (k_min >= stoop_min_knee_angle)
    # AND Trunk deeply flexed (t_mean >= stoop_mean_trunk_inclination AND t_max >= stoop_min_trunk_inclination)
    is_stoop_knees = k_min >= rules.stoop_min_knee_angle
    is_stoop_trunk = (t_mean >= rules.stoop_mean_trunk_inclination) and (t_max >= rules.stoop_min_trunk_inclination)

    if is_stoop_knees and is_stoop_trunk:
        outcome = (
            f"Pronounced forward trunk flexion (max={t_max:.1f}°, mean={t_mean:.1f}°) with "
            f"extended knees (min={k_min:.1f}°, mean={k_mean:.1f}°)."
        )
        return (PROP_LABEL_UNSAFE, STATUS_SUCCESS, outcome, "")

    # Gate 7: Mixed Mechanics / Borderline / Conflicting Patterns (AMBIGUOUS)
    if is_squat_knees and not is_squat_trunk:
        reason = (
            f"Mixed mechanics: knees flexed (min={k_min:.1f}°) but trunk also bent forward "
            f"(max={t_max:.1f}° > {rules.squat_max_trunk_inclination}°)."
        )
        outcome = "Semi-squat lift with conflicting knee and trunk patterns."
        return (PROP_LABEL_AMBIGUOUS, STATUS_SUCCESS, outcome, reason)

    if is_stoop_knees and not is_stoop_trunk:
        reason = (
            f"Mixed mechanics: straight knees (min={k_min:.1f}°) but trunk relatively upright "
            f"(mean={t_mean:.1f}° < {rules.stoop_mean_trunk_inclination}°)."
        )
        outcome = "Uncharacteristic low-range extension without severe torso flexion."
        return (PROP_LABEL_AMBIGUOUS, STATUS_SUCCESS, outcome, reason)

    # General Borderline Case
    reason = (
        f"Borderline mechanics: knee angle (min={k_min:.1f}°, mean={k_mean:.1f}°) and "
        f"trunk inclination (mean={t_mean:.1f}°, max={t_max:.1f}°) fall into transitional intermediate range."
    )
    outcome = "Borderline intermediate posture; does not decisively satisfy squat or stoop screening criteria."
    return (PROP_LABEL_AMBIGUOUS, STATUS_SUCCESS, outcome, reason)


class AutoLabelPipeline:
    """Orchestrates video extraction, indicator measurement, and candidate proposal generation."""

    def __init__(
        self,
        candidates_csv_path: Optional[Path] = None,
        rules_config: Optional[LabelingRulesConfig] = None,
        pose_extractor: Optional[PoseExtractor] = None,
    ):
        self.candidates_csv_path = candidates_csv_path or config.AUTOMATIC_CANDIDATES_FILE
        self.rules_config = rules_config or LabelingRulesConfig()
        self._external_extractor = pose_extractor
        self._owned_extractor: Optional[PoseExtractor] = None

    def _get_extractor(self) -> PoseExtractor:
        if self._external_extractor is not None:
            return self._external_extractor
        if self._owned_extractor is None:
            self._owned_extractor = PoseExtractor()
        return self._owned_extractor

    def process_single_video(
        self,
        video_path: Path,
        video_meta: MatchedVideoMetadata,
    ) -> CandidateProposalResult:
        """Extract pose across the action interval and calculate biomechanical metrics."""
        v_name = video_meta.video_filename
        if not video_path.exists():
            return CandidateProposalResult(
                video_filename=v_name,
                proposed_label=PROP_LABEL_AMBIGUOUS,
                processing_status=STATUS_VIDEO_NOT_FOUND,
                rule_outcomes="Video file missing on filesystem.",
                ambiguity_reason=f"File not found: {video_path}",
                metrics=None,
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return CandidateProposalResult(
                video_filename=v_name,
                proposed_label=PROP_LABEL_AMBIGUOUS,
                processing_status=STATUS_ERROR,
                rule_outcomes="Failed to open video container with OpenCV.",
                ambiguity_reason=f"OpenCV cannot decode video file: {v_name}",
                metrics=None,
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = total_video_frames / fps if fps > 0 else 0.0

        # Determine interval frames
        start_sec = video_meta.action_start_sec if video_meta.action_start_sec is not None else 0.0
        end_sec = video_meta.action_end_sec if video_meta.action_end_sec is not None else duration_sec

        start_frame = max(0, int(start_sec * fps))
        end_frame = min(total_video_frames - 1, int(end_sec * fps)) if total_video_frames > 0 else int(end_sec * fps)

        if end_frame < start_frame:
            end_frame = start_frame + int(fps * 2.0)

        extractor = self._get_extractor()

        trunk_angles: List[float] = []
        knee_angles: List[float] = []
        crit_vis_scores: List[float] = []
        total_sampled = 0
        valid_frames = 0

        current_frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame_idx = start_frame

        step = max(1, self.rules_config.sample_step_frames)

        while current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            total_sampled += 1
            pose_res: PoseExtractionResult = extractor.extract_from_frame(frame)

            if pose_res.is_valid:
                valid_frames += 1
                crit_vis_scores.append(pose_res.critical_visibility_ratio)

                angles = pose_res.biomechanical_angles
                if angles.get("trunk_inclination") is not None:
                    trunk_angles.append(angles["trunk_inclination"])

                k_ang = angles.get("knee_angle_mean")
                if k_ang is None:
                    k_left = angles.get("knee_angle_left")
                    k_right = angles.get("knee_angle_right")
                    k_valid = [k for k in (k_left, k_right) if k is not None]
                    if k_valid:
                        k_ang = float(np.mean(k_valid))

                if k_ang is not None:
                    knee_angles.append(k_ang)

            # Advance by step
            if step > 1:
                current_frame_idx += step
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            else:
                current_frame_idx += 1

        cap.release()

        coverage_pct = (valid_frames / total_sampled * 100.0) if total_sampled > 0 else 0.0
        mean_vis = float(np.mean(crit_vis_scores)) if crit_vis_scores else 0.0

        t_mean = round(float(np.mean(trunk_angles)), 1) if trunk_angles else None
        t_max = round(float(np.max(trunk_angles)), 1) if trunk_angles else None
        k_min = round(float(np.min(knee_angles)), 1) if knee_angles else None
        k_mean = round(float(np.mean(knee_angles)), 1) if knee_angles else None

        asym_deg = float(video_meta.angle_of_asymmetry_deg or 0.0)

        metrics = VideoBiomechanicalMetrics(
            video_filename=v_name,
            total_frames_sampled=total_sampled,
            valid_pose_frames=valid_frames,
            pose_coverage_pct=coverage_pct,
            mean_critical_visibility=mean_vis,
            trunk_inclination_mean=t_mean,
            trunk_inclination_max=t_max,
            knee_angle_min=k_min,
            knee_angle_mean=k_mean,
            angle_of_asymmetry_deg=asym_deg,
            camera_view=video_meta.camera_view,
            action_direction=video_meta.action_direction,
        )

        p_label, p_status, p_outcomes, p_reason = evaluate_ergonomic_rules(
            metrics, self.rules_config
        )

        return CandidateProposalResult(
            video_filename=v_name,
            proposed_label=p_label,
            processing_status=p_status,
            rule_outcomes=p_outcomes,
            ambiguity_reason=p_reason,
            metrics=metrics,
        )

    def run_batch_processing(
        self,
        review_manager: Optional[DatasetReviewManager] = None,
        force_reprocess: bool = False,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> pd.DataFrame:
        """Batch-process all matched Environment 4 videos and persist proposals.

        Existing candidates are cached unless force_reprocess is True.
        Original annotations and manual-review records are left completely intact.
        """
        mgr = review_manager or DatasetReviewManager()
        matched_videos = mgr.matched_videos

        existing_df = self.load_candidates_dataframe()
        cached_records: Dict[str, Dict[str, Any]] = {}
        if not force_reprocess and not existing_df.empty:
            for _, r in existing_df.iterrows():
                v_name = r.get("video_filename")
                if v_name and r.get("processing_status") in (STATUS_SUCCESS, STATUS_LOW_COVERAGE, STATUS_LOW_VISIBILITY):
                    cached_records[v_name] = r.to_dict()

        total = len(matched_videos)
        results: List[Dict[str, Any]] = []

        logger.info(f"Starting batch auto-labeling across {total} Environment 4 videos...")

        for idx, meta in enumerate(matched_videos):
            v_name = meta.video_filename

            if progress_callback:
                progress_callback(idx + 1, total, v_name, "Processing...")

            if not force_reprocess and v_name in cached_records:
                results.append(cached_records[v_name])
                continue

            v_path = Path(meta.absolute_video_path)
            try:
                candidate = self.process_single_video(v_path, meta)
                d = candidate.to_dict()
                d["master_annotation_ref"] = meta.master_annotation_ref
                d["participant_id"] = meta.participant_id
                d["action_start_sec"] = meta.action_start_sec
                d["action_end_sec"] = meta.action_end_sec
                d["pipeline_timestamp"] = datetime.now().isoformat()
                results.append(d)
            except Exception as exc:
                logger.error(f"Error processing {v_name}: {exc}", exc_info=True)
                err_cand = CandidateProposalResult(
                    video_filename=v_name,
                    proposed_label=PROP_LABEL_AMBIGUOUS,
                    processing_status=STATUS_ERROR,
                    rule_outcomes=f"Exception during processing: {str(exc)}",
                    ambiguity_reason=str(exc),
                    metrics=None,
                )
                d = err_cand.to_dict()
                d["master_annotation_ref"] = meta.master_annotation_ref
                d["participant_id"] = meta.participant_id
                d["action_start_sec"] = meta.action_start_sec
                d["action_end_sec"] = meta.action_end_sec
                d["pipeline_timestamp"] = datetime.now().isoformat()
                results.append(d)

        df = pd.DataFrame(results)

        # Ensure consistent column ordering
        cols_order = [
            "video_filename", "master_annotation_ref", "participant_id", "camera_view",
            "action_direction", "action_start_sec", "action_end_sec",
            "proposed_label", "processing_status", "rule_outcomes", "ambiguity_reason",
            "total_frames_sampled", "valid_pose_frames", "pose_coverage_pct",
            "mean_critical_visibility", "trunk_inclination_mean", "trunk_inclination_max",
            "knee_angle_min", "knee_angle_mean", "angle_of_asymmetry_deg",
            "pipeline_timestamp", "disclaimer"
        ]
        present_cols = [c for c in cols_order if c in df.columns]
        extra_cols = [c for c in df.columns if c not in present_cols]
        df = df[present_cols + extra_cols]

        self.candidates_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.candidates_csv_path, index=False)
        logger.info(f"Persisted {len(df)} candidate proposals to {self.candidates_csv_path}")

        if progress_callback:
            progress_callback(total, total, "Complete", "Batch candidate labeling completed.")

        return df

    def load_candidates_dataframe(self) -> pd.DataFrame:
        """Load automatic_label_candidates.csv if present, or return empty DataFrame."""
        if not self.candidates_csv_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.candidates_csv_path, keep_default_na=False)
        except Exception as exc:
            logger.warning(f"Could not load candidates from {self.candidates_csv_path}: {exc}")
            return pd.DataFrame()

    def get_candidate_summary(self) -> Dict[str, Any]:
        """Compute aggregate summary of candidates, status, and human review requirements."""
        df = self.load_candidates_dataframe()
        if df.empty:
            return {
                "total": 0,
                "proposed_safe": 0,
                "proposed_unsafe": 0,
                "proposed_ambiguous": 0,
                "status_success": 0,
                "status_low_coverage": 0,
                "status_error": 0,
                "participant_distribution": {},
                "needs_human_review": 0,
            }

        total = len(df)
        safe_count = int((df["proposed_label"] == PROP_LABEL_SAFE).sum())
        unsafe_count = int((df["proposed_label"] == PROP_LABEL_UNSAFE).sum())
        ambiguous_count = int((df["proposed_label"] == PROP_LABEL_AMBIGUOUS).sum())

        status_success = int((df["processing_status"] == STATUS_SUCCESS).sum())
        status_low_cov = int((df["processing_status"].isin([STATUS_LOW_COVERAGE, STATUS_LOW_VISIBILITY])).sum())
        status_err = int((df["processing_status"].isin([STATUS_ERROR, STATUS_VIDEO_NOT_FOUND, STATUS_NO_POSE_DETECTED])).sum())

        p_dist: Dict[str, Dict[str, int]] = {}
        if "participant_id" in df.columns:
            for p in sorted(df["participant_id"].unique()):
                sub = df[df["participant_id"] == p]
                p_dist[str(p)] = {
                    "total": len(sub),
                    "safe": int((sub["proposed_label"] == PROP_LABEL_SAFE).sum()),
                    "unsafe": int((sub["proposed_label"] == PROP_LABEL_UNSAFE).sum()),
                    "ambiguous": int((sub["proposed_label"] == PROP_LABEL_AMBIGUOUS).sum()),
                }

        return {
            "total": total,
            "proposed_safe": safe_count,
            "proposed_unsafe": unsafe_count,
            "proposed_ambiguous": ambiguous_count,
            "status_success": status_success,
            "status_low_coverage": status_low_cov,
            "status_error": status_err,
            "participant_distribution": p_dist,
            # All proposals require human sign-off; ambiguous cases particularly require full manual inspection
            "needs_human_review": total,
        }

    def close(self) -> None:
        """Clean up owned pose extractor resources."""
        if self._owned_extractor is not None:
            self._owned_extractor.close()
            self._owned_extractor = None
