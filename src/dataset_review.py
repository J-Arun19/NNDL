"""Dataset review, metadata exploration, and manual labeling utility for SAFELIFT Environment 4.

Preserves all original videos and master annotations.json without modification.
Tracks review progress, manages manual annotations in data/manual_review_labels.csv,
and retains subject grouping and camera angles for leak-free model splitting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config

logger = logging.getLogger(__name__)

MANUAL_REVIEW_CSV_PATH = config.DATA_DIR / "manual_review_labels.csv"

# Label Constants
LABEL_SAFE = "SAFE"
LABEL_UNSAFE = "UNSAFE"
LABEL_AMBIGUOUS = "AMBIGUOUS"
STATUS_UNREVIEWED = "UNREVIEWED"
STATUS_REVIEWED = "REVIEWED"

VALID_REVIEW_LABELS = [LABEL_SAFE, LABEL_UNSAFE, LABEL_AMBIGUOUS]

LABELING_GUIDANCE_TEXT = r"""
### Ergonomic Labeling Criteria (Manual Material Handling)

> **Important Warning:** Ergonomic screening criteria are educational and occupational risk
> screening indicators based on biomechanical principles (NIOSH / OSHA), NOT certified medical diagnoses.

#### 1. SAFE (Squat Lift / Proper Mechanics)
- **Knee Flexion:** Marked knee flexion (knees bent $>80^\circ$).
- **Trunk Inclination:** Trunk kept relatively upright (torso angle $<45^\circ$ from vertical).
- **Center of Gravity:** Load remains close to the body without excessive horizontal reaching.
- **Spinal Mechanics:** Preserves lumbar lordosis; lower extremities (quadriceps, gluteals) drive the lift.

#### 2. UNSAFE (Stoop Lift / High Lumbosacral Shear)
- **Knee Flexion:** Extended knees (legs nearly straight or minimally flexed, $>140^\circ$).
- **Trunk Flexion:** Severe forward flexion of the lumbar spine and torso ($>45^\circ$ from vertical).
- **Extended Lever Arm:** Lifter bends predominantly at the hips and lower back, creating a large moment arm.
- **Spinal Shear:** High compressive and shear force concentrated on the **L5/S1** lumbosacral discs.

#### 3. AMBIGUOUS (Uncertain / Mixed Mechanics)
- **Severe Occlusion:** Critical joints (knees, hips, lower back) are obstructed from view.
- **Mixed / Hybrid Mechanics:** Intermediate semi-squat posture with borderline trunk flexion.
- **Transitional Movement:** Unclear start/end timing, stumbling, or non-standard lifting trajectory.
"""


@dataclass
class MatchedVideoMetadata:
    """Rich metadata bundle for an Environment 4 video trial."""
    video_filename: str
    relative_video_path: str
    absolute_video_path: str
    master_annotation_ref: str
    participant_id: str
    session_id: str
    action_direction: str
    camera_view: str
    repetition_index: int
    action_start_sec: Optional[float]
    action_end_sec: Optional[float]
    angle_of_asymmetry_deg: Optional[float]
    subjective_judgment_original: str
    box_width_cm: Optional[float]
    box_height_cm: Optional[float]
    box_depth_cm: Optional[float]
    subject_height_cm: Optional[float]
    subject_age: Optional[float]
    subject_gender: Optional[str]
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetReviewManager:
    """Manages read-only dataset exploration and manual labeling persistence."""

    def __init__(
        self,
        annotations_json_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        review_csv_path: Optional[Path] = None,
    ):
        self.annotations_json_path = annotations_json_path or (config.DATA_DIR / "annotations.json")
        self.data_dir = data_dir or config.DATA_DIR
        self.project_root = self.data_dir.parent
        self.raw_videos_dir = self.data_dir / "raw_videos"
        self.review_csv_path = review_csv_path or MANUAL_REVIEW_CSV_PATH

        self.matched_videos: List[MatchedVideoMetadata] = []
        self._load_matched_environment_4_videos()

    def _load_matched_environment_4_videos(self) -> None:
        """Scan raw_videos folders, match with annotations.json, and parse rich metadata."""
        if not self.annotations_json_path.exists():
            logger.warning(f"annotations.json not found at {self.annotations_json_path}")
            return

        with open(self.annotations_json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        dir_lateral = self.raw_videos_dir / "video_env_4_anonymized"
        dir_oblique = self.raw_videos_dir / "video_env_4_oblique_anonymized"

        files_lateral = {f.name.replace("_anonymized.mp4", ".mp4"): f for f in dir_lateral.glob("**/*.mp4")}
        files_oblique = {f.name.replace("_anonymized.mp4", ".mp4"): f for f in dir_oblique.glob("**/*.mp4")}

        all_disk_files: Dict[str, Tuple[Path, str]] = {}
        for base, path in files_lateral.items():
            all_disk_files[base] = (path, "lateral")
        for base, path in files_oblique.items():
            all_disk_files[base] = (path, "oblique")

        matched: List[MatchedVideoMetadata] = []

        for item in annotations:
            v_ref = item.get("video", "")
            if v_ref in all_disk_files:
                disk_path, view = all_disk_files[v_ref]
                # Parse participant ID from filename prefix (e.g., 'A.' -> 'A')
                p_id = v_ref.split(".")[0].strip() if "." in v_ref else "Unknown"

                # Parse action direction
                if "TERRA-CARRIOLA" in v_ref:
                    action_dir = "TERRA-CARRIOLA (Floor to Wheelbarrow - Lifting)"
                elif "CARRIOLA-TERRA" in v_ref:
                    action_dir = "CARRIOLA-TERRA (Wheelbarrow to Floor - Lowering)"
                else:
                    action_dir = "Other"

                # Parse trial repetition number
                rep_idx = 1
                for token in disk_path.stem.split():
                    clean_tok = token.replace("_anonymized", "").strip()
                    if clean_tok.isdigit():
                        rep_idx = int(clean_tok)
                        break

                act_start = item.get("action_start", {}) or {}
                act_end = item.get("action_end", {}) or {}

                start_sec = act_start.get("video_second")
                end_sec = act_end.get("video_second")
                asym_angle = act_start.get("angle_of_asymmetry_degree")

                try:
                    rel_v_path = str(disk_path.relative_to(self.project_root))
                except ValueError:
                    try:
                        rel_v_path = str(disk_path.relative_to(config.PROJECT_ROOT))
                    except ValueError:
                        rel_v_path = str(disk_path)

                meta = MatchedVideoMetadata(
                    video_filename=disk_path.name,
                    relative_video_path=rel_v_path,
                    absolute_video_path=str(disk_path.resolve()),
                    master_annotation_ref=v_ref,
                    participant_id=p_id,
                    session_id="env_4",
                    action_direction=action_dir,
                    camera_view=view,
                    repetition_index=rep_idx,
                    action_start_sec=float(start_sec) if start_sec is not None else None,
                    action_end_sec=float(end_sec) if end_sec is not None else None,
                    angle_of_asymmetry_deg=float(asym_angle) if asym_angle is not None else None,
                    subjective_judgment_original=str(item.get("subjective_judgment", "unknown")),
                    box_width_cm=item.get("box_width_cm"),
                    box_height_cm=item.get("box_height_cm"),
                    box_depth_cm=item.get("box_depth_cm"),
                    subject_height_cm=item.get("subject_height_cm"),
                    subject_age=item.get("subject_age"),
                    subject_gender=item.get("subject_gender"),
                    note=str(item.get("Note", "unknown")),
                )
                matched.append(meta)

        # Sort predictably by participant, view, action, repetition
        self.matched_videos = sorted(
            matched,
            key=lambda x: (x.participant_id, x.camera_view, x.action_direction, x.repetition_index)
        )

    def load_or_init_review_manifest(self) -> pd.DataFrame:
        """Load existing review CSV or initialize with all 49 videos as UNREVIEWED."""
        string_cols = {
            "video_filename": str,
            "relative_video_path": str,
            "master_annotation_ref": str,
            "participant_id": str,
            "session_id": str,
            "action_direction": str,
            "camera_view": str,
            "subjective_judgment_original": str,
            "subject_gender": str,
            "review_status": str,
            "reviewed_label": str,
            "reviewer_notes": str,
            "review_timestamp": str,
            "reviewer": str,
        }
        if self.review_csv_path.exists():
            df = pd.read_csv(self.review_csv_path, dtype=string_cols, keep_default_na=False)
            # Ensure all columns exist
            expected_cols = [
                "video_filename", "relative_video_path", "master_annotation_ref",
                "participant_id", "session_id", "action_direction", "camera_view",
                "repetition_index", "action_start_sec", "action_end_sec",
                "angle_of_asymmetry_deg", "subjective_judgment_original",
                "box_width_cm", "box_height_cm", "box_depth_cm",
                "subject_height_cm", "subject_age", "subject_gender",
                "review_status", "reviewed_label", "numeric_label",
                "reviewer_notes", "review_timestamp", "reviewer"
            ]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = ""
            df["numeric_label"] = pd.to_numeric(df["numeric_label"], errors="coerce").fillna(-1).astype(int)
            return df

        # Initialize template
        rows = []
        for m in self.matched_videos:
            rows.append({
                "video_filename": m.video_filename,
                "relative_video_path": m.relative_video_path,
                "master_annotation_ref": m.master_annotation_ref,
                "participant_id": m.participant_id,
                "session_id": m.session_id,
                "action_direction": m.action_direction,
                "camera_view": m.camera_view,
                "repetition_index": m.repetition_index,
                "action_start_sec": m.action_start_sec,
                "action_end_sec": m.action_end_sec,
                "angle_of_asymmetry_deg": m.angle_of_asymmetry_deg,
                "subjective_judgment_original": m.subjective_judgment_original,
                "box_width_cm": m.box_width_cm,
                "box_height_cm": m.box_height_cm,
                "box_depth_cm": m.box_depth_cm,
                "subject_height_cm": m.subject_height_cm,
                "subject_age": m.subject_age,
                "subject_gender": m.subject_gender,
                # Explicit unreviewed status
                "review_status": STATUS_UNREVIEWED,
                "reviewed_label": "",
                "numeric_label": -1,
                "reviewer_notes": "",
                "review_timestamp": "",
                "reviewer": "",
            })

        df = pd.DataFrame(rows)
        self.review_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.review_csv_path, index=False)
        return df

    def save_review_label(
        self,
        video_filename: str,
        label: str,
        reviewer_notes: str = "",
        reviewer: str = "Expert_Reviewer",
        action_start_override: Optional[float] = None,
        action_end_override: Optional[float] = None,
    ) -> bool:
        """Record manual review label (SAFE, UNSAFE, AMBIGUOUS) and update manifest.

        Preserves original 'subjective_judgment' without automatic conversion.
        """
        label_clean = label.strip().upper()
        if label_clean not in VALID_REVIEW_LABELS:
            raise ValueError(f"Invalid label '{label}'. Must be one of {VALID_REVIEW_LABELS}")

        df = self.load_or_init_review_manifest()
        mask = df["video_filename"] == video_filename

        if not mask.any():
            # Try matching by master_annotation_ref or base name
            mask = df["master_annotation_ref"] == video_filename
            if not mask.any():
                logger.error(f"Video {video_filename} not found in review manifest.")
                return False

        idx = df[mask].index[0]

        # Numeric mapping: SAFE -> 0, UNSAFE -> 1, AMBIGUOUS -> -1
        numeric_val = 0 if label_clean == LABEL_SAFE else (1 if label_clean == LABEL_UNSAFE else -1)

        df.at[idx, "review_status"] = STATUS_REVIEWED
        df.at[idx, "reviewed_label"] = label_clean
        df.at[idx, "numeric_label"] = numeric_val
        df.at[idx, "reviewer_notes"] = reviewer_notes.strip()
        df.at[idx, "review_timestamp"] = datetime.now().isoformat()
        df.at[idx, "reviewer"] = reviewer.strip()

        if action_start_override is not None:
            df.at[idx, "action_start_sec"] = action_start_override
        if action_end_override is not None:
            df.at[idx, "action_end_sec"] = action_end_override

        df.to_csv(self.review_csv_path, index=False)
        logger.info(f"Recorded manual review for {video_filename}: {label_clean}")
        return True

    def get_review_progress(self) -> Dict[str, Any]:
        """Calculate review completion metrics and label distribution."""
        df = self.load_or_init_review_manifest()
        total = len(df)
        if total == 0:
            return {
                "total": 0,
                "reviewed": 0,
                "unreviewed": 0,
                "percent_complete": 0.0,
                "safe_count": 0,
                "unsafe_count": 0,
                "ambiguous_count": 0,
                "participants_reviewed": {},
            }

        reviewed_mask = df["review_status"] == STATUS_REVIEWED
        reviewed_count = int(reviewed_mask.sum())
        unreviewed_count = total - reviewed_count

        safe_count = int((df["reviewed_label"] == LABEL_SAFE).sum())
        unsafe_count = int((df["reviewed_label"] == LABEL_UNSAFE).sum())
        ambiguous_count = int((df["reviewed_label"] == LABEL_AMBIGUOUS).sum())

        pct = round((reviewed_count / total) * 100.0, 1) if total > 0 else 0.0

        p_progress = {}
        for pid in sorted(df["participant_id"].unique()):
            sub = df[df["participant_id"] == pid]
            rev_sub = (sub["review_status"] == STATUS_REVIEWED).sum()
            p_progress[pid] = f"{rev_sub}/{len(sub)}"

        return {
            "total": total,
            "reviewed": reviewed_count,
            "unreviewed": unreviewed_count,
            "percent_complete": pct,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
            "ambiguous_count": ambiguous_count,
            "participants_progress": p_progress,
        }

    def get_video_metadata(self, video_filename: str) -> Optional[MatchedVideoMetadata]:
        """Retrieve rich metadata bundle for a specific video filename."""
        for m in self.matched_videos:
            if m.video_filename == video_filename or m.master_annotation_ref == video_filename:
                return m
        return None

    def export_reviewed_annotations_csv(
        self,
        target_path: Optional[Path] = None,
        exclude_ambiguous: bool = True,
    ) -> Tuple[pd.DataFrame, Path]:
        """Export reviewed labels into standard annotations.csv format for dataset loader.

        Schema: clip_path, participant_id, session_id, label, view
        """
        out_path = target_path or (config.DATA_DIR / "annotations.csv")
        df = self.load_or_init_review_manifest()

        # Only export entries that have been actively reviewed as SAFE (0) or UNSAFE (1)
        valid_mask = (df["review_status"] == STATUS_REVIEWED) & (df["numeric_label"].isin([0, 1]))
        if not exclude_ambiguous:
            valid_mask = df["review_status"] == STATUS_REVIEWED

        reviewed_df = df[valid_mask].copy()

        export_rows = []
        for _, row in reviewed_df.iterrows():
            export_rows.append({
                "clip_path": row["relative_video_path"],
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "label": int(row["numeric_label"]),
                "view": row["camera_view"],
            })

        out_df = pd.DataFrame(export_rows)
        out_df.to_csv(out_path, index=False)
        logger.info(f"Exported {len(out_df)} reviewed samples to {out_path}")
        return out_df, out_path
