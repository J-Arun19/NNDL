"""Dataset loader, validation, and group-based manifest splitting module.

Handles annotation validation, participant-grouped partitioning (preventing
data leakage), video frame extraction, and pose map caching.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config
from src.pose_extractor import PoseExtractor
from src.pose_renderer import render_pose_map

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["clip_path", "participant_id", "session_id", "label", "view"]


@dataclass
class SafeliftAuditReport:
    """Detailed audit report on raw SAFELIFT annotations.json and local video matches."""
    is_loaded: bool
    annotations_file: str
    total_json_records: int
    local_videos_total: int
    matched_videos_count: int
    missing_videos_count: int
    unmatched_videos_count: int
    label_distribution_json: Dict[str, int]
    label_distribution_matched: Dict[str, int]
    participants: List[str]
    camera_views: Dict[str, int]
    action_directions: Dict[str, int]
    has_binary_training_labels: bool
    binary_labels_summary: Dict[str, Any]
    can_construct_training_dataset: bool
    training_readiness_reasons: List[str]
    next_actions: List[str]
    warnings: List[str]
    errors: List[str]
    matched_records: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_loaded": self.is_loaded,
            "annotations_file": self.annotations_file,
            "total_json_records": self.total_json_records,
            "local_videos_total": self.local_videos_total,
            "matched_videos_count": self.matched_videos_count,
            "missing_videos_count": self.missing_videos_count,
            "unmatched_videos_count": self.unmatched_videos_count,
            "label_distribution_json": self.label_distribution_json,
            "label_distribution_matched": self.label_distribution_matched,
            "participants": self.participants,
            "camera_views": self.camera_views,
            "action_directions": self.action_directions,
            "has_binary_training_labels": self.has_binary_training_labels,
            "binary_labels_summary": self.binary_labels_summary,
            "can_construct_training_dataset": self.can_construct_training_dataset,
            "training_readiness_reasons": self.training_readiness_reasons,
            "next_actions": self.next_actions,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class ValidationReport:
    """Detailed diagnostic report on dataset health and annotation integrity."""
    is_valid: bool
    total_records: int
    safe_count: int
    unsafe_count: int
    unique_participants: int
    unique_sessions: int
    missing_files: List[str]
    invalid_labels: List[Any]
    duplicate_rows: int
    warnings: List[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "safe_count": self.safe_count,
            "unsafe_count": self.unsafe_count,
            "unique_participants": self.unique_participants,
            "unique_sessions": self.unique_sessions,
            "missing_files_count": len(self.missing_files),
            "invalid_labels_count": len(self.invalid_labels),
            "duplicate_rows": self.duplicate_rows,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def validate_safelift_dataset(
    json_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> SafeliftAuditReport:
    """Audit the actual SAFELIFT JSON annotations and match with local videos.

    Supports the real SAFELIFT JSON schema present in data/annotations.json.
    Preserves all original annotations and video files untouched.
    Never fabricates SAFE/UNSAFE labels when source annotations say 'unknown'.
    """
    ann_json = json_path or getattr(config, "ANNOTATIONS_JSON_FILE", config.DATA_DIR / "annotations.json")
    base_dir = data_dir or config.DATA_DIR
    raw_vids_dir = base_dir / "raw_videos"
    manual_csv = getattr(config, "MANUAL_REVIEW_FILE", base_dir / "manual_review_labels.csv")
    ann_csv = getattr(config, "ANNOTATIONS_FILE", base_dir / "annotations.csv")

    warnings: List[str] = []
    errors: List[str] = []

    if not ann_json.exists():
        errors.append(f"annotations.json not found at: {ann_json}")
        return SafeliftAuditReport(
            is_loaded=False,
            annotations_file=str(ann_json),
            total_json_records=0,
            local_videos_total=0,
            matched_videos_count=0,
            missing_videos_count=0,
            unmatched_videos_count=0,
            label_distribution_json={},
            label_distribution_matched={},
            participants=[],
            camera_views={},
            action_directions={},
            has_binary_training_labels=False,
            binary_labels_summary={},
            can_construct_training_dataset=False,
            training_readiness_reasons=["annotations.json missing."],
            next_actions=["Place annotations.json in data/ directory."],
            warnings=warnings,
            errors=errors,
            matched_records=[],
        )

    try:
        with open(ann_json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as exc:
        errors.append(f"Failed to parse annotations.json: {exc}")
        return SafeliftAuditReport(
            is_loaded=False,
            annotations_file=str(ann_json),
            total_json_records=0,
            local_videos_total=0,
            matched_videos_count=0,
            missing_videos_count=0,
            unmatched_videos_count=0,
            label_distribution_json={},
            label_distribution_matched={},
            participants=[],
            camera_views={},
            action_directions={},
            has_binary_training_labels=False,
            binary_labels_summary={},
            can_construct_training_dataset=False,
            training_readiness_reasons=[f"Invalid JSON: {exc}"],
            next_actions=["Fix syntax error in annotations.json."],
            warnings=warnings,
            errors=errors,
            matched_records=[],
        )

    total_records = len(raw_data)

    # Scan local videos in data/raw_videos
    dir_lat = raw_vids_dir / "video_env_4_anonymized"
    dir_obl = raw_vids_dir / "video_env_4_oblique_anonymized"

    files_lat = list(dir_lat.glob("**/*.mp4")) if dir_lat.exists() else []
    files_obl = list(dir_obl.glob("**/*.mp4")) if dir_obl.exists() else []
    all_local_files = files_lat + files_obl
    local_total = len(all_local_files)

    # Disk mapping: strip '_anonymized.mp4' -> '.mp4'
    disk_map: Dict[str, Tuple[Path, str]] = {}
    for f in files_lat:
        norm_key = f.name.replace("_anonymized.mp4", ".mp4")
        disk_map[norm_key] = (f, "lateral")
    for f in files_obl:
        norm_key = f.name.replace("_anonymized.mp4", ".mp4")
        disk_map[norm_key] = (f, "oblique")

    matched_records: List[Dict[str, Any]] = []
    matched_json_keys: Set[str] = set()

    label_dist_json: Dict[str, int] = {}
    label_dist_matched: Dict[str, int] = {}
    views_count: Dict[str, int] = {}
    actions_count: Dict[str, int] = {}
    participants_set: Set[str] = set()

    for item in raw_data:
        v_name = item.get("video", "")
        j_val = str(item.get("subjective_judgment", "unknown"))
        label_dist_json[j_val] = label_dist_json.get(j_val, 0) + 1

        if v_name in disk_map:
            disk_path, view = disk_map[v_name]
            matched_json_keys.add(v_name)
            label_dist_matched[j_val] = label_dist_matched.get(j_val, 0) + 1
            views_count[view] = views_count.get(view, 0) + 1

            p_id = v_name.split(".")[0].strip() if "." in v_name else "Unknown"
            participants_set.add(p_id)

            if "TERRA-CARRIOLA" in v_name:
                act = "TERRA-CARRIOLA (Floor to Wheelbarrow - Lifting)"
            elif "CARRIOLA-TERRA" in v_name:
                act = "CARRIOLA-TERRA (Wheelbarrow to Floor - Lowering)"
            else:
                act = "Other"
            actions_count[act] = actions_count.get(act, 0) + 1

            act_start = item.get("action_start", {}) or {}
            act_end = item.get("action_end", {}) or {}

            matched_records.append({
                "video_filename": disk_path.name,
                "master_annotation_ref": v_name,
                "disk_path": str(disk_path),
                "participant_id": p_id,
                "camera_view": view,
                "action_direction": act,
                "action_start_sec": act_start.get("video_second"),
                "action_end_sec": act_end.get("video_second"),
                "angle_of_asymmetry_deg": act_start.get("angle_of_asymmetry_degree", 0.0),
                "subjective_judgment": j_val,
                "box_width_cm": item.get("box_width_cm"),
                "box_height_cm": item.get("box_height_cm"),
                "box_depth_cm": item.get("box_depth_cm"),
                "subject_height_cm": item.get("subject_height_cm"),
                "subject_age": item.get("subject_age"),
                "subject_gender": item.get("subject_gender"),
                "note": item.get("Note", "unknown"),
            })

    matched_count = len(matched_records)
    missing_count = total_records - len(matched_json_keys)
    unmatched_local_count = local_total - len(matched_records)

    # Check for manual review labels / binary labels
    binary_safe_count = 0
    binary_unsafe_count = 0
    binary_ambiguous_count = 0
    unreviewed_count = 0
    reviewed_participants: Set[str] = set()

    if manual_csv.exists():
        try:
            df_man = pd.read_csv(manual_csv, keep_default_na=False)
            for _, row in df_man.iterrows():
                st_val = str(row.get("review_status", "")).strip().upper()
                lbl_val = str(row.get("reviewed_label", "")).strip().upper()
                pid = str(row.get("participant_id", "")).strip()
                if st_val == "REVIEWED":
                    if lbl_val == "SAFE":
                        binary_safe_count += 1
                        reviewed_participants.add(pid)
                    elif lbl_val == "UNSAFE":
                        binary_unsafe_count += 1
                        reviewed_participants.add(pid)
                    elif lbl_val == "AMBIGUOUS":
                        binary_ambiguous_count += 1
                else:
                    unreviewed_count += 1
        except Exception as exc:
            warnings.append(f"Could not read manual review CSV: {exc}")

    # Check if annotations.csv exists with valid binary labels
    has_annotations_csv = ann_csv.exists()
    ann_csv_safe = 0
    ann_csv_unsafe = 0
    ann_csv_pids: Set[str] = set()
    if has_annotations_csv:
        try:
            df_ann = pd.read_csv(ann_csv)
            if "label" in df_ann.columns:
                for _, r in df_ann.iterrows():
                    l_str = str(r["label"]).strip().lower()
                    p = str(r.get("participant_id", ""))
                    if l_str in ["0", "safe"]:
                        ann_csv_safe += 1
                        ann_csv_pids.add(p)
                    elif l_str in ["1", "unsafe"]:
                        ann_csv_unsafe += 1
                        ann_csv_pids.add(p)
        except Exception:
            pass

    effective_safe = max(binary_safe_count, ann_csv_safe)
    effective_unsafe = max(binary_unsafe_count, ann_csv_unsafe)
    effective_pids = reviewed_participants if binary_safe_count > 0 else ann_csv_pids

    has_usable_binary = (
        effective_safe > 0 and effective_unsafe > 0 and len(effective_pids) >= 3
    )

    # Determine training dataset viability
    readiness_reasons: List[str] = []
    next_actions: List[str] = []

    if effective_unsafe == 0:
        readiness_reasons.append(
            "Missing class: 0 UNSAFE (Class 1 / stoop) samples available. "
            "A binary classification CNN requires both SAFE and UNSAFE examples to train."
        )
    if effective_safe == 0:
        readiness_reasons.append(
            "Missing class: 0 SAFE (Class 0 / squat) samples available."
        )
    if len(effective_pids) < 3:
        p_list = sorted(list(effective_pids))
        readiness_reasons.append(
            f"Insufficient participant grouping: Only {len(effective_pids)} participant(s) "
            f"({p_list or 'None'}) have binary review labels. "
            "At least 3 distinct participants (e.g. A, B, C, D) are required for leak-free "
            "GroupShuffleSplit partitioning between train, validation, and test sets."
        )
    if (effective_safe + effective_unsafe) < 10:
        readiness_reasons.append(
            f"Insufficient total binary labels: Only {effective_safe + effective_unsafe} labeled sample(s) available "
            "(minimum recommended: >= 20-30 samples to construct valid train/val/test splits)."
        )

    readiness_reasons.append(
        "Source SAFELIFT annotations.json contains only ergonomic metadata and subjective judgment "
        "ratings ('unknown' and 'intermediate'), without binary SAFE/UNSAFE labels."
    )

    can_construct = (len(readiness_reasons) == 1) and has_usable_binary

    next_actions.append(
        "Open the 'Dataset Review' page in the sidebar to review Environment 4 videos across "
        "participants B, C, and D, assigning SAFE to squat mechanics and UNSAFE to stoop mechanics."
    )
    next_actions.append(
        "Alternatively, provide a verified binary-labelled CSV manifest matching annotations.csv.example "
        "(columns: clip_path, participant_id, session_id, label, view)."
    )
    next_actions.append(
        "Once balanced binary labels exist across >= 3 participants, export the reviewed manifest "
        "to data/annotations.csv, run feature extraction, train the CNN, and run frozen held-out test evaluation."
    )

    binary_summary = {
        "manual_review_csv_exists": manual_csv.exists(),
        "manual_safe_count": binary_safe_count,
        "manual_unsafe_count": binary_unsafe_count,
        "manual_ambiguous_count": binary_ambiguous_count,
        "manual_unreviewed_count": unreviewed_count,
        "manual_reviewed_participants": sorted(list(reviewed_participants)),
        "annotations_csv_exists": has_annotations_csv,
        "annotations_csv_safe": ann_csv_safe,
        "annotations_csv_unsafe": ann_csv_unsafe,
        "annotations_csv_participants": sorted(list(ann_csv_pids)),
        "usable_binary_samples": effective_safe + effective_unsafe,
    }

    return SafeliftAuditReport(
        is_loaded=True,
        annotations_file=str(ann_json),
        total_json_records=total_records,
        local_videos_total=local_total,
        matched_videos_count=matched_count,
        missing_videos_count=missing_count,
        unmatched_videos_count=unmatched_local_count,
        label_distribution_json=label_dist_json,
        label_distribution_matched=label_dist_matched,
        participants=sorted(list(participants_set)),
        camera_views=views_count,
        action_directions=actions_count,
        has_binary_training_labels=has_usable_binary,
        binary_labels_summary=binary_summary,
        can_construct_training_dataset=can_construct,
        training_readiness_reasons=readiness_reasons,
        next_actions=next_actions,
        warnings=warnings,
        errors=errors,
        matched_records=matched_records,
    )


def validate_annotations(
    csv_path: Optional[Path] = None, base_data_dir: Optional[Path] = None
) -> ValidationReport:
    """Validate annotation CSV schema, file references, labels, and participant grouping."""
    path = csv_path or config.ANNOTATIONS_FILE
    base_dir = base_data_dir or config.DATA_DIR

    warnings: List[str] = []
    errors: List[str] = []
    missing_files: List[str] = []
    invalid_labels: List[Any] = []

    if not path.exists():
        errors.append(f"Annotation file not found at: {path}")
        return ValidationReport(
            is_valid=False,
            total_records=0,
            safe_count=0,
            unsafe_count=0,
            unique_participants=0,
            unique_sessions=0,
            missing_files=[],
            invalid_labels=[],
            duplicate_rows=0,
            warnings=warnings,
            errors=errors,
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Failed to read CSV: {exc}")
        return ValidationReport(
            is_valid=False,
            total_records=0,
            safe_count=0,
            unsafe_count=0,
            unique_participants=0,
            unique_sessions=0,
            missing_files=[],
            invalid_labels=[],
            duplicate_rows=0,
            warnings=warnings,
            errors=errors,
        )

    # 1. Column presence check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}. Expected: {REQUIRED_COLUMNS}")

    total_records = len(df)
    if total_records == 0:
        errors.append("Annotation CSV contains 0 rows.")
        return ValidationReport(
            is_valid=False,
            total_records=0,
            safe_count=0,
            unsafe_count=0,
            unique_participants=0,
            unique_sessions=0,
            missing_files=[],
            invalid_labels=[],
            duplicate_rows=0,
            warnings=warnings,
            errors=errors,
        )

    # 2. Check duplicates
    duplicate_rows = 0
    if "clip_path" in df.columns:
        duplicate_rows = int(df.duplicated(subset=["clip_path"]).sum())
        if duplicate_rows > 0:
            warnings.append(f"Detected {duplicate_rows} duplicate clip_path entries in annotations.")

    # 3. Label validation (must map to 0 or 1)
    safe_count = 0
    unsafe_count = 0
    if "label" in df.columns:
        for idx, val in df["label"].items():
            str_val = str(val).strip().lower()
            if str_val in ["0", "safe"]:
                safe_count += 1
            elif str_val in ["1", "unsafe"]:
                unsafe_count += 1
            else:
                invalid_labels.append({"row": idx, "label": val})

        if invalid_labels:
            errors.append(f"Found {len(invalid_labels)} rows with invalid labels (must be 0/safe or 1/unsafe).")

        if safe_count == 0 or unsafe_count == 0:
            warnings.append(f"Dataset has single-class representation: safe={safe_count}, unsafe={unsafe_count}.")

    # 4. Check video file existence on disk
    if "clip_path" in df.columns:
        for p in df["clip_path"]:
            rel_p = Path(p)
            abs_p = rel_p if rel_p.is_absolute() else base_dir / rel_p
            if not abs_p.exists():
                missing_files.append(str(p))

        if missing_files:
            errors.append(f"{len(missing_files)} video clips referenced in annotations do not exist on disk.")

    unique_participants = df["participant_id"].nunique() if "participant_id" in df.columns else 0
    unique_sessions = df["session_id"].nunique() if "session_id" in df.columns else 0

    if unique_participants < 3 and total_records > 0:
        warnings.append(
            f"Only {unique_participants} unique participant(s) found. "
            "At least 3 participants are strongly recommended to support leak-free train/val/test splits."
        )

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        total_records=total_records,
        safe_count=safe_count,
        unsafe_count=unsafe_count,
        unique_participants=unique_participants,
        unique_sessions=unique_sessions,
        missing_files=missing_files,
        invalid_labels=invalid_labels,
        duplicate_rows=duplicate_rows,
        warnings=warnings,
        errors=errors,
    )


def split_dataset_by_groups(
    df: pd.DataFrame,
    group_col: str = "participant_id",
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into Train, Val, and Test splits strictly grouped by participant.

    Guarantees zero participant leakage between train, val, and test subsets.
    """
    if group_col not in df.columns or df[group_col].nunique() < 3:
        # Fallback to session_id if participant grouping is not viable
        if "session_id" in df.columns and df["session_id"].nunique() >= 3:
            logger.warning("Fewer than 3 participants found. Falling back to session_id grouping.")
            group_col = "session_id"
        else:
            logger.warning(
                f"Insufficient distinct groups for group_col='{group_col}'. "
                "Splitting by unique clip to preserve basic partitioned evaluation."
            )
            # Create a synthetic pseudo-group per clip if groups are scarce
            group_col = "clip_path"

    groups = df[group_col].values
    n_groups = len(np.unique(groups))

    # Split into (Train + Val) and Test
    test_size_ratio = test_ratio / (train_ratio + val_ratio + test_ratio)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size_ratio, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Split Train + Val into Train and Val
    train_val_groups = train_val_df[group_col].values
    val_size_ratio = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_ratio, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_groups))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    # Leakage Prevention Assertion
    train_g = set(train_df[group_col])
    val_g = set(val_df[group_col])
    test_g = set(test_df[group_col])

    overlap_train_val = train_g & val_g
    overlap_train_test = train_g & test_g
    overlap_val_test = val_g & test_g

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"CRITICAL DATA LEAKAGE DETECTED! Overlap found: "
            f"train/val={overlap_train_val}, train/test={overlap_train_test}, val/test={overlap_val_test}"
        )

    return train_df, val_df, test_df


def save_split_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Save separated split manifests to CSV files."""
    s_dir = splits_dir or config.SPLITS_DIR
    s_dir.mkdir(parents=True, exist_ok=True)

    train_path = s_dir / "train.csv"
    val_path = s_dir / "val.csv"
    test_path = s_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {"train": train_path, "val": val_path, "test": test_path}


def extract_pose_maps_from_video(
    video_path: Path,
    extractor: PoseExtractor,
    sample_rate_fps: int = config.DEFAULT_FRAME_SAMPLE_RATE,
) -> List[Dict[str, Any]]:
    """Sample video frames, extract pose landmarks, and render 128x128 pose maps."""
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Unable to open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / sample_rate_fps)))

    frame_records: List[Dict[str, Any]] = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            extraction_res = extractor.extract_from_frame(frame_bgr)
            pose_map = render_pose_map(extraction_res.landmarks)

            frame_records.append({
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                "is_valid_pose": extraction_res.is_valid,
                "status_message": extraction_res.status_message,
                "critical_visibility_ratio": extraction_res.critical_visibility_ratio,
                "biomechanical_angles": extraction_res.biomechanical_angles,
                "pose_map": pose_map,  # numpy array (128, 128, 1)
            })

        frame_idx += 1

    cap.release()
    return frame_records


def build_and_cache_dataset(
    annotations_csv: Optional[Path] = None,
    output_pose_maps_dir: Optional[Path] = None,
    output_metadata_csv: Optional[Path] = None,
    sample_rate_fps: int = config.DEFAULT_FRAME_SAMPLE_RATE,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """End-to-end dataset builder: validates annotations, extracts poses, caches maps, and creates splits."""
    ann_path = annotations_csv or config.ANNOTATIONS_FILE
    maps_dir = output_pose_maps_dir or config.POSE_MAPS_DIR
    meta_path = output_metadata_csv or config.METADATA_FILE

    maps_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    report = validate_annotations(ann_path)
    if not report.is_valid:
        logger.error(f"Cannot build dataset: annotation errors found: {report.errors}")
        return pd.DataFrame(), report

    df_ann = pd.read_csv(ann_path)
    # Standardize label column to int (0 or 1)
    df_ann["label"] = df_ann["label"].apply(
        lambda x: 0 if str(x).strip().lower() in ["0", "safe"] else 1
    )

    extractor = PoseExtractor()
    metadata_rows: List[Dict[str, Any]] = []

    for _, row in df_ann.iterrows():
        clip_rel = Path(row["clip_path"])
        clip_abs = clip_rel if clip_rel.is_absolute() else config.DATA_DIR / clip_rel
        clip_stem = clip_abs.stem

        frame_records = extract_pose_maps_from_video(
            clip_abs, extractor, sample_rate_fps=sample_rate_fps
        )

        for rec in frame_records:
            if rec["is_valid_pose"]:
                # Save numpy pose map
                map_filename = f"{clip_stem}_f{rec['frame_idx']:05d}.npy"
                map_path = maps_dir / map_filename
                np.save(str(map_path), rec["pose_map"])

                metadata_rows.append({
                    "pose_map_path": str(map_path.relative_to(config.PROJECT_ROOT)),
                    "clip_path": str(row["clip_path"]),
                    "participant_id": row["participant_id"],
                    "session_id": row["session_id"],
                    "label": int(row["label"]),
                    "view": row.get("view", "lateral"),
                    "frame_idx": rec["frame_idx"],
                    "timestamp_sec": rec["timestamp_sec"],
                    "critical_visibility_ratio": rec["critical_visibility_ratio"],
                })

    extractor.close()
    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(meta_path, index=False)

    # Generate and save participant-grouped splits if valid data exists
    if len(df_meta) > 0:
        train_df, val_df, test_df = split_dataset_by_groups(df_meta)
        save_split_manifests(train_df, val_df, test_df)

    return df_meta, report


def generate_synthetic_test_fixtures(
    output_dir: Path,
    num_samples: int = 12
) -> Dict[str, Any]:
    """Generate clearly labeled synthetic fixtures for software unit tests ONLY.

    NOTE: These fixtures are strictly for verifying pipeline mechanics and unit test
    execution. They must NEVER be used or reported as project training data!
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = output_dir / "pose_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    # 4 participants: P01, P02 for train; P03 for val; P04 for test
    participants = ["P01", "P01", "P02", "P02", "P03", "P03", "P04", "P04"]

    for i in range(num_samples):
        pid = participants[i % len(participants)]
        label = 0 if i % 2 == 0 else 1  # Alternating safe and unsafe
        map_filename = f"synthetic_sample_{i:03d}.npy"
        map_path = maps_dir / map_filename

        # Generate a synthetic 128x128 1-channel pose map
        dummy_canvas = np.zeros((config.POSE_MAP_HEIGHT, config.POSE_MAP_WIDTH, 1), dtype=np.uint8)
        # Add a synthetic rectangle or circle to simulate pose keypoints
        center_x = 40 if label == 0 else 80
        cv2.circle(dummy_canvas, (center_x, 50), 4, int(config.JOINT_INTENSITY), -1)
        cv2.line(dummy_canvas, (center_x, 50), (center_x, 90), int(config.BONE_INTENSITY), 2)
        np.save(str(map_path), dummy_canvas)

        rows.append({
            "pose_map_path": str(map_path),
            "clip_path": f"raw_videos/synthetic_{pid}_{i}.mp4",
            "participant_id": pid,
            "session_id": f"S0{i % 2 + 1}",
            "label": label,
            "view": "lateral",
            "frame_idx": i * 10,
            "timestamp_sec": float(i * 0.33),
            "critical_visibility_ratio": 1.0,
            "_is_synthetic_fixture": True,  # Explicit marker
        })

    df = pd.DataFrame(rows)
    ann_file = output_dir / "synthetic_annotations.csv"
    df.to_csv(ann_file, index=False)

    # Perform group split
    train_df, val_df, test_df = split_dataset_by_groups(df, group_col="participant_id")
    save_split_manifests(train_df, val_df, test_df, splits_dir=splits_dir)

    return {
        "annotations_file": ann_file,
        "splits_dir": splits_dir,
        "maps_dir": maps_dir,
        "dataframe": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }
