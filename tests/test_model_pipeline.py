"""Unit tests for CNN architecture, inference mechanics, and metric calculations."""

import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

import config
from src.dataset_loader import generate_synthetic_test_fixtures
from src.evaluate import evaluate_held_out_test_set, load_model_and_config
from src.inference import LiftingPostureClassifier
from src.pose_extractor import LandmarkPoint
from src.reporting import generate_session_report
from src.train import build_lifting_cnn, train_model


def test_cnn_architecture_shapes():
    """Verify CNN layer hierarchy, input shape, and output dimensionality."""
    model = build_lifting_cnn(input_shape=(128, 128, 1))

    # Verify input shape
    assert model.input_shape == (None, 128, 128, 1)
    # Verify output shape (single binary sigmoid probability)
    assert model.output_shape == (None, 1)

    # Forward pass on synthetic batch
    dummy_batch = np.zeros((2, 128, 128, 1), dtype=np.float32)
    preds = model(dummy_batch, training=False).numpy()
    assert preds.shape == (2, 1)
    # Sigmoid outputs must be strictly in [0.0, 1.0]
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_missing_model_handling(tmp_path):
    """Ensure missing model files are handled honestly without crashing."""
    non_existent_model = tmp_path / "missing.keras"
    model, cfg = load_model_and_config(model_path=non_existent_model)
    assert model is None
    assert cfg.get("status") in ("CNN NOT TRAINED", "MODEL NOT TRAINED")

    # Evaluator behavior on missing model
    eval_res = evaluate_held_out_test_set(model_path=non_existent_model)
    assert eval_res.get("status") in ("CNN NOT TRAINED", "MODEL NOT TRAINED")


def test_missing_pose_handling_outputs_uncertain():
    """Verify that absent or occluded poses are classified as UNCERTAIN, never defaulting to SAFE."""
    classifier = LiftingPostureClassifier()

    # Pass a blank black image (no human present)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res, annotated = classifier.process_frame(blank_frame, frame_index=0, timestamp_sec=0.0)

    assert not res.is_valid_pose
    assert res.raw_status in (config.STATUS_UNCERTAIN, "INSUFFICIENT VISIBILITY", "UNCERTAIN / POSE NOT DETECTED")
    assert res.smoothed_status in (config.STATUS_UNCERTAIN, "INSUFFICIENT VISIBILITY", "UNCERTAIN / POSE NOT DETECTED")
    assert res.raw_probability is None
    assert res.smoothed_probability is None
    assert annotated.shape == blank_frame.shape

    classifier.close()


def test_end_to_end_training_and_evaluation_on_synthetic_fixture(tmp_path):
    """Verify complete train -> evaluate pipeline using isolated synthetic test fixtures."""
    fixture_dir = tmp_path / "synthetic_run"
    fixtures = generate_synthetic_test_fixtures(fixture_dir, num_samples=16)

    train_df = fixtures["train_df"]
    val_df = fixtures["val_df"]
    test_df = fixtures["test_df"]

    model_path = fixture_dir / "test_model.keras"
    config_path = fixture_dir / "test_model_config.json"
    metrics_path = fixture_dir / "metrics.json"
    preds_path = fixture_dir / "test_predictions.csv"
    curves_path = fixture_dir / "training_curves.png"
    cm_path = fixture_dir / "confusion_matrix.png"
    rep_path = fixture_dir / "classification_report.txt"

    # Train CNN for 1 epoch on synthetic data
    model, summary = train_model(
        train_df=train_df,
        val_df=val_df,
        epochs=1,
        batch_size=4,
        seed=42,
        model_save_path=model_path,
        config_save_path=config_path,
        curves_plot_path=curves_path,
    )

    assert model_path.exists()
    assert config_path.exists()
    assert "decision_threshold" in summary

    # Run held-out evaluation on synthetic test set
    test_manifest = fixture_dir / "splits" / "test.csv"
    eval_results = evaluate_held_out_test_set(
        test_manifest_path=test_manifest,
        model_path=model_path,
        config_path=config_path,
        metrics_output_path=metrics_path,
        predictions_output_path=preds_path,
        report_output_path=rep_path,
        confusion_matrix_plot_path=cm_path,
    )

    assert eval_results["status"] == "EVALUATION COMPLETE"
    assert "accuracy" in eval_results
    assert "f1_unsafe" in eval_results
    assert "specificity" in eval_results
    assert "confusion_matrix" in eval_results
    assert metrics_path.exists()
    assert preds_path.exists()


def test_session_reporting_metrics():
    """Verify session report calculations, risk scoring, and percentages."""
    import pandas as pd

    df = pd.DataFrame({
        "frame_index": [0, 1, 2, 3],
        "timestamp_sec": [0.0, 0.1, 0.2, 0.3],
        "status": ["SAFE", "UNSAFE", "SAFE", "UNCERTAIN / POSE NOT DETECTED"],
        "is_valid_pose": [True, True, True, False],
        "latency_ms": [10.0, 12.0, 11.0, 5.0],
        "trunk_inclination_deg": [20.0, 55.0, 22.0, None],
        "knee_angle_mean_deg": [95.0, 150.0, 90.0, None],
    })

    rep = generate_session_report(df)
    assert rep.total_frames == 4
    assert rep.valid_pose_frames == 3
    assert rep.safe_frames == 2
    assert rep.unsafe_frames == 1
    assert rep.uncertain_frames == 1
    assert rep.safe_pct_of_valid == 66.7
    assert rep.unsafe_pct_of_valid == 33.3
    assert rep.ergonomic_risk_level == "MODERATE ERGONOMIC RISK"
