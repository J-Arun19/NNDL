"""Unit tests for dataset manifest validation and group-based leakage prevention."""

from pathlib import Path
import pandas as pd
import pytest

import config
from src.dataset_loader import (
    generate_synthetic_test_fixtures,
    split_dataset_by_groups,
    validate_annotations,
)


def test_validate_annotations_missing_file(tmp_path):
    """Test validator behavior when CSV file does not exist."""
    missing_file = tmp_path / "non_existent.csv"
    report = validate_annotations(missing_file)
    assert not report.is_valid
    assert any("not found" in err.lower() for err in report.errors)


def test_validate_annotations_invalid_schema(tmp_path):
    """Test validator detects missing required columns."""
    bad_csv = tmp_path / "bad_schema.csv"
    df = pd.DataFrame({
        "video_path": ["clip1.mp4"],
        "user_id": ["P01"],
        # Missing label, session_id, view, clip_path
    })
    df.to_csv(bad_csv, index=False)

    report = validate_annotations(bad_csv)
    assert not report.is_valid
    assert any("missing required columns" in err.lower() for err in report.errors)


def test_validate_annotations_invalid_labels(tmp_path):
    """Test validator catches unknown or non-binary labels."""
    bad_labels_csv = tmp_path / "bad_labels.csv"
    df = pd.DataFrame({
        "clip_path": ["c1.mp4", "c2.mp4"],
        "participant_id": ["P01", "P02"],
        "session_id": ["S01", "S01"],
        "label": ["medium", "unknown"],  # Invalid
        "view": ["lateral", "lateral"]
    })
    df.to_csv(bad_labels_csv, index=False)

    report = validate_annotations(bad_labels_csv)
    assert not report.is_valid
    assert len(report.invalid_labels) == 2


def test_split_dataset_prevents_participant_leakage(tmp_path):
    """Verify GroupShuffleSplit completely prevents participant overlap between train, val, and test."""
    fixtures = generate_synthetic_test_fixtures(tmp_path / "fixtures", num_samples=24)
    train_df = fixtures["train_df"]
    val_df = fixtures["val_df"]
    test_df = fixtures["test_df"]

    train_pids = set(train_df["participant_id"].unique())
    val_pids = set(val_df["participant_id"].unique())
    test_pids = set(test_df["participant_id"].unique())

    # Strict Zero-Leakage Asserts
    assert len(train_pids & val_pids) == 0, f"Train-Val participant leakage: {train_pids & val_pids}"
    assert len(train_pids & test_pids) == 0, f"Train-Test participant leakage: {train_pids & test_pids}"
    assert len(val_pids & test_pids) == 0, f"Val-Test participant leakage: {val_pids & test_pids}"

    # Verify total samples match
    total_split_len = len(train_df) + len(val_df) + len(test_df)
    assert total_split_len == len(fixtures["dataframe"])


def test_synthetic_fixture_marker(tmp_path):
    """Ensure synthetic fixtures are unambiguously flagged as software tests only."""
    fixtures = generate_synthetic_test_fixtures(tmp_path / "synth", num_samples=8)
    df = fixtures["dataframe"]
    assert "_is_synthetic_fixture" in df.columns
    assert all(df["_is_synthetic_fixture"] == True)
