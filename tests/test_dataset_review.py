"""Unit tests for SAFELIFT dataset review utility using synthetic fixtures only.

All tests utilize isolated temporary directories and synthetic mock fixtures.
Never treats fixture labels as genuine data.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.dataset_review import (
    DatasetReviewManager,
    LABEL_SAFE,
    LABEL_UNSAFE,
    LABEL_AMBIGUOUS,
    STATUS_UNREVIEWED,
    STATUS_REVIEWED,
    LABELING_GUIDANCE_TEXT,
)


@pytest.fixture
def synthetic_review_environment(tmp_path: Path):
    """Create a completely isolated synthetic dataset environment for testing review workflows."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    lat_dir = data_dir / "raw_videos" / "video_env_4_anonymized"
    obl_dir = data_dir / "raw_videos" / "video_env_4_oblique_anonymized"
    lat_dir.mkdir(parents=True)
    obl_dir.mkdir(parents=True)

    # Create dummy video files
    v1_name = "A.TERRA-CARRIOLA LAT 1_anonymized.mp4"
    v2_name = "A.CARRIOLA-TERRA LAT 2_anonymized.mp4"
    v3_name = "B.TERRA-CARRIOLA 3quarti 1_anonymized.mp4"

    (lat_dir / v1_name).write_bytes(b"dummy_mp4_1")
    (lat_dir / v2_name).write_bytes(b"dummy_mp4_2")
    (obl_dir / v3_name).write_bytes(b"dummy_mp4_3")

    # Master annotations.json with original unedited subjective_judgment
    master_annotations = [
        {
            "video": "A.TERRA-CARRIOLA LAT 1.mp4",
            "action_start": {"video_second": 1.25, "angle_of_asymmetry_degree": 0.0},
            "action_end": {"video_second": 3.75},
            "subjective_judgment": "intermediate",
            "box_width_cm": 40.0,
            "box_height_cm": 30.0,
            "box_depth_cm": 25.0,
            "subject_height_cm": 178.0,
            "subject_age": 28.0,
            "subject_gender": "M",
            "Note": "Test Note 1"
        },
        {
            "video": "A.CARRIOLA-TERRA LAT 2.mp4",
            "action_start": {"video_second": 2.0, "angle_of_asymmetry_degree": 5.0},
            "action_end": {"video_second": 4.5},
            "subjective_judgment": "unknown",
            "box_width_cm": 40.0,
            "box_height_cm": 30.0,
            "box_depth_cm": 25.0,
            "subject_height_cm": 178.0,
            "subject_age": 28.0,
            "subject_gender": "M",
            "Note": "Test Note 2"
        },
        {
            "video": "B.TERRA-CARRIOLA 3quarti 1.mp4",
            "action_start": {"video_second": 1.0, "angle_of_asymmetry_degree": 12.0},
            "action_end": {"video_second": 3.2},
            "subjective_judgment": "unknown",
            "box_width_cm": 40.0,
            "box_height_cm": 30.0,
            "box_depth_cm": 25.0,
            "subject_height_cm": 165.0,
            "subject_age": 34.0,
            "subject_gender": "F",
            "Note": "Test Note 3"
        }
    ]

    annotations_file = data_dir / "annotations.json"
    with open(annotations_file, "w", encoding="utf-8") as f:
        json.dump(master_annotations, f)

    review_csv_path = data_dir / "manual_review_labels.csv"

    return {
        "data_dir": data_dir,
        "annotations_file": annotations_file,
        "review_csv_path": review_csv_path,
        "v1": v1_name,
        "v2": v2_name,
        "v3": v3_name,
    }


def test_manager_discovers_and_matches_videos(synthetic_review_environment):
    """Test that manager correctly associates disk videos with annotation metadata."""
    env = synthetic_review_environment
    manager = DatasetReviewManager(
        annotations_json_path=env["annotations_file"],
        data_dir=env["data_dir"],
        review_csv_path=env["review_csv_path"],
    )

    assert len(manager.matched_videos) == 3

    meta_v1 = manager.get_video_metadata(env["v1"])
    assert meta_v1 is not None
    assert meta_v1.participant_id == "A"
    assert meta_v1.camera_view == "lateral"
    assert meta_v1.action_start_sec == 1.25
    assert meta_v1.action_end_sec == 3.75
    assert meta_v1.angle_of_asymmetry_deg == 0.0
    assert meta_v1.subjective_judgment_original == "intermediate"

    meta_v3 = manager.get_video_metadata(env["v3"])
    assert meta_v3 is not None
    assert meta_v3.participant_id == "B"
    assert meta_v3.camera_view == "oblique"
    assert meta_v3.subjective_judgment_original == "unknown"


def test_manifest_initialization_marks_unreviewed(synthetic_review_environment):
    """Test manifest creation initializes all items as UNREVIEWED without modifying originals."""
    env = synthetic_review_environment
    manager = DatasetReviewManager(
        annotations_json_path=env["annotations_file"],
        data_dir=env["data_dir"],
        review_csv_path=env["review_csv_path"],
    )

    df = manager.load_or_init_review_manifest()
    assert len(df) == 3
    assert (df["review_status"] == STATUS_UNREVIEWED).all()
    assert (df["reviewed_label"] == "").all()
    assert (df["numeric_label"] == -1).all()

    # Verify original annotations file was NOT modified
    with open(env["annotations_file"], "r", encoding="utf-8") as f:
        orig = json.load(f)
    assert orig[0]["subjective_judgment"] == "intermediate"
    assert orig[1]["subjective_judgment"] == "unknown"


def test_save_review_label_recording(synthetic_review_environment):
    """Test recording SAFE, UNSAFE, and AMBIGUOUS labels with reviewer metadata."""
    env = synthetic_review_environment
    manager = DatasetReviewManager(
        annotations_json_path=env["annotations_file"],
        data_dir=env["data_dir"],
        review_csv_path=env["review_csv_path"],
    )

    # 1. Label video 1 as SAFE
    success1 = manager.save_review_label(
        video_filename=env["v1"],
        label=LABEL_SAFE,
        reviewer_notes="Clean squat lift, upright torso.",
        reviewer="Ergonomist_1",
    )
    assert success1 is True

    # 2. Label video 2 as UNSAFE
    success2 = manager.save_review_label(
        video_filename=env["v2"],
        label=LABEL_UNSAFE,
        reviewer_notes="High stoop lift, knees straight.",
        reviewer="Ergonomist_1",
    )
    assert success2 is True

    # 3. Label video 3 as AMBIGUOUS
    success3 = manager.save_review_label(
        video_filename=env["v3"],
        label=LABEL_AMBIGUOUS,
        reviewer_notes="Partial occlusion of knee angle.",
        reviewer="Ergonomist_2",
    )
    assert success3 is True

    # Verify persisted CSV
    df = pd.read_csv(env["review_csv_path"])
    v1_row = df[df["video_filename"] == env["v1"]].iloc[0]
    assert v1_row["review_status"] == STATUS_REVIEWED
    assert v1_row["reviewed_label"] == LABEL_SAFE
    assert v1_row["numeric_label"] == 0
    assert v1_row["reviewer"] == "Ergonomist_1"
    assert "Clean squat" in v1_row["reviewer_notes"]
    assert v1_row["review_timestamp"] != ""

    v2_row = df[df["video_filename"] == env["v2"]].iloc[0]
    assert v2_row["reviewed_label"] == LABEL_UNSAFE
    assert v2_row["numeric_label"] == 1

    v3_row = df[df["video_filename"] == env["v3"]].iloc[0]
    assert v3_row["reviewed_label"] == LABEL_AMBIGUOUS
    assert v3_row["numeric_label"] == -1


def test_save_review_label_rejects_invalid_label(synthetic_review_environment):
    """Test that invalid label strings are rejected with ValueError."""
    env = synthetic_review_environment
    manager = DatasetReviewManager(
        annotations_json_path=env["annotations_file"],
        data_dir=env["data_dir"],
        review_csv_path=env["review_csv_path"],
    )

    with pytest.raises(ValueError, match="Invalid label 'INVALID_LABEL'"):
        manager.save_review_label(
            video_filename=env["v1"],
            label="INVALID_LABEL",
        )


def test_get_review_progress_metrics(synthetic_review_environment):
    """Test review progress statistics tracking."""
    env = synthetic_review_environment
    manager = DatasetReviewManager(
        annotations_json_path=env["annotations_file"],
        data_dir=env["data_dir"],
        review_csv_path=env["review_csv_path"],
    )

    progress_before = manager.get_review_progress()
    assert progress_before["total"] == 3
    assert progress_before["reviewed"] == 0
    assert progress_before["unreviewed"] == 3
    assert progress_before["percent_complete"] == 0.0

    # Review 1 of 3
    manager.save_review_label(env["v1"], LABEL_SAFE)

    progress_mid = manager.get_review_progress()
    assert progress_mid["reviewed"] == 1
    assert progress_mid["unreviewed"] == 2
    assert progress_mid["safe_count"] == 1
    assert progress_mid["unsafe_count"] == 0
    assert progress_mid["percent_complete"] == 33.3
    assert progress_mid["participants_progress"]["A"] == "1/2"
    assert progress_mid["participants_progress"]["B"] == "0/1"


def test_export_reviewed_annotations_csv(synthetic_review_environment, tmp_path):
    """Test exporting verified binary labels into the model-compatible annotations schema."""
    env = synthetic_review_environment
    manager = DatasetReviewManager(
        annotations_json_path=env["annotations_file"],
        data_dir=env["data_dir"],
        review_csv_path=env["review_csv_path"],
    )

    manager.save_review_label(env["v1"], LABEL_SAFE)
    manager.save_review_label(env["v2"], LABEL_UNSAFE)
    manager.save_review_label(env["v3"], LABEL_AMBIGUOUS)

    export_path = tmp_path / "exported_annotations.csv"
    out_df, path = manager.export_reviewed_annotations_csv(
        target_path=export_path,
        exclude_ambiguous=True,
    )

    assert path == export_path
    # By default, exclude_ambiguous=True so only SAFE (0) and UNSAFE (1) are exported
    assert set(out_df.columns) == {"clip_path", "participant_id", "session_id", "label", "view"}
    assert set(out_df["label"]) == {0, 1}
    assert list(out_df["participant_id"].unique()) == ["A"]


def test_labeling_guidance_and_warning_contents():
    """Verify ergonomic guidance text contains required criteria and non-diagnostic disclaimer."""
    assert "Ergonomic Labeling Criteria" in LABELING_GUIDANCE_TEXT
    assert "NOT certified medical diagnoses" in LABELING_GUIDANCE_TEXT
    assert "SAFE (Squat Lift" in LABELING_GUIDANCE_TEXT
    assert "UNSAFE (Stoop Lift" in LABELING_GUIDANCE_TEXT
    assert "AMBIGUOUS" in LABELING_GUIDANCE_TEXT
    assert "L5/S1" in LABELING_GUIDANCE_TEXT
