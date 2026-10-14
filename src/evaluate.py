"""Evaluation module for Unsafe Lifting Posture Detection CNN.

Evaluates the frozen CNN model on the held-out test set, calculates genuine
classification metrics (Accuracy, Precision, Recall, F1, Specificity, ROC-AUC),
plots confusion matrices, measures inference latency, and saves metrics files.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import tensorflow as tf
from tensorflow import keras

import config

logger = logging.getLogger(__name__)


def verify_cnn_checkpoint(
    model_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect and verify CNN model checkpoint, architecture, and weights.

    Returns structured verification dictionary with detailed diagnosis.
    """
    m_path = model_path or config.BEST_MODEL_PATH
    c_path = config_path or config.MODEL_CONFIG_PATH

    if not m_path.exists():
        return {
            "checkpoint_exists": False,
            "checkpoint_path": str(m_path),
            "is_valid": False,
            "status": "CNN NOT TRAINED",
            "message": f"Checkpoint not found at {m_path}.",
            "architecture": None,
            "weights_verified": False,
        }

    try:
        model = keras.models.load_model(str(m_path))
    except Exception as exc:
        logger.error(f"Failed to load Keras model from {m_path}: {exc}")
        return {
            "checkpoint_exists": True,
            "checkpoint_path": str(m_path),
            "is_valid": False,
            "status": "MODEL LOAD ERROR",
            "message": f"Failed to load Keras model from {m_path}: {exc}",
            "architecture": None,
            "weights_verified": False,
        }

    # Verify input shape: expected (None, 128, 128, 1)
    input_shape = getattr(model, "input_shape", None)
    expected_shape = (None, config.POSE_MAP_HEIGHT, config.POSE_MAP_WIDTH, config.POSE_MAP_CHANNELS)
    if input_shape != expected_shape and input_shape != (None, 128, 128, 1):
        return {
            "checkpoint_exists": True,
            "checkpoint_path": str(m_path),
            "is_valid": False,
            "status": "INVALID ARCHITECTURE",
            "message": f"Model input shape {input_shape} does not match expected {expected_shape}.",
            "architecture": {"input_shape": str(input_shape), "output_shape": str(getattr(model, "output_shape", None))},
            "weights_verified": False,
        }

    # Verify output shape: expected binary sigmoid probability (None, 1)
    output_shape = getattr(model, "output_shape", None)
    if output_shape != (None, 1):
        return {
            "checkpoint_exists": True,
            "checkpoint_path": str(m_path),
            "is_valid": False,
            "status": "INVALID ARCHITECTURE",
            "message": f"Model output shape {output_shape} does not match expected binary classification shape (None, 1).",
            "architecture": {"input_shape": str(input_shape), "output_shape": str(output_shape)},
            "weights_verified": False,
        }

    # Verify weights
    weights = model.weights
    if not weights or len(weights) == 0:
        return {
            "checkpoint_exists": True,
            "checkpoint_path": str(m_path),
            "is_valid": False,
            "status": "INVALID WEIGHTS",
            "message": "Model contains no trainable or initialized weights.",
            "architecture": {"input_shape": str(input_shape), "output_shape": str(output_shape)},
            "weights_verified": False,
        }

    # Check for NaN / Inf in weights
    for w in weights:
        w_arr = w.numpy() if hasattr(w, "numpy") else np.array(w)
        if np.isnan(w_arr).any() or np.isinf(w_arr).any():
            return {
                "checkpoint_exists": True,
                "checkpoint_path": str(m_path),
                "is_valid": False,
                "status": "INVALID WEIGHTS",
                "message": f"Weight tensor '{w.name}' contains NaN or Inf values.",
                "architecture": {"input_shape": str(input_shape), "output_shape": str(output_shape)},
                "weights_verified": False,
            }

    trainable_count = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total_count = int(sum(np.prod(w.shape) for w in model.weights))

    arch_info = {
        "input_shape": list(input_shape) if input_shape else [],
        "output_shape": list(output_shape) if output_shape else [],
        "layers_count": len(model.layers),
        "trainable_params": trainable_count,
        "total_params": total_count,
    }

    return {
        "checkpoint_exists": True,
        "checkpoint_path": str(m_path),
        "is_valid": True,
        "status": "VALID CHECKPOINT",
        "message": f"Valid trained CNN verified successfully ({total_count:,} parameters).",
        "architecture": arch_info,
        "weights_verified": True,
    }


def get_project_model_status(
    model_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Determine comprehensive status of the model, screening method, and held-out evaluation.

    Guarantees:
    - If no checkpoint exists: displays 'CNN NOT TRAINED'.
    - If checkpoint exists but un-evaluated: displays 'EVALUATION PENDING'.
    - If valid evaluation results exist: displays measured test metrics.
    - Screening method displayed separately: 'Rule-Based Ergonomic Assessment' or 'Trained CNN'.
    - Never claims CNN is trained or evaluated just because rule-based screening works.
    """
    m_path = model_path or config.BEST_MODEL_PATH
    c_path = config_path or config.MODEL_CONFIG_PATH
    met_path = metrics_path or (config.METRICS_DIR / "metrics.json")

    chk = verify_cnn_checkpoint(m_path, c_path)

    # Screening method is distinct from CNN status
    screening_method = "Trained CNN" if chk["is_valid"] else "Rule-Based Ergonomic Assessment"

    if not chk["checkpoint_exists"]:
        return {
            "status": "CNN NOT TRAINED",
            "model_status_label": "CNN NOT TRAINED",
            "is_valid_checkpoint": False,
            "checkpoint_exists": False,
            "evaluation_status": "CNN NOT TRAINED",
            "has_evaluation_results": False,
            "screening_method": screening_method,
            "metrics": None,
            "architecture": None,
            "reason": (
                "No CNN model checkpoint exists at models/best_lifting_cnn.keras. "
                "Real-time posture screening is powered by Rule-Based Ergonomic Assessment "
                "using MediaPipe Pose tracking and NIOSH/OSHA biomechanics."
            ),
        }

    if not chk["is_valid"]:
        return {
            "status": chk["status"],
            "model_status_label": chk["status"],
            "is_valid_checkpoint": False,
            "checkpoint_exists": True,
            "evaluation_status": "EVALUATION PENDING",
            "has_evaluation_results": False,
            "screening_method": "Rule-Based Ergonomic Assessment",
            "metrics": None,
            "architecture": chk.get("architecture"),
            "reason": chk["message"],
        }

    # Valid checkpoint exists; check held-out evaluation
    has_metrics = met_path.exists()
    metrics_data = None
    if has_metrics:
        try:
            with open(met_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            if not isinstance(metrics_data, dict) or metrics_data.get("status") != "EVALUATION COMPLETE" or metrics_data.get("test_sample_count", 0) <= 0:
                has_metrics = False
        except Exception:
            has_metrics = False

    if has_metrics and metrics_data:
        return {
            "status": "EVALUATED",
            "model_status_label": "EVALUATED",
            "is_valid_checkpoint": True,
            "checkpoint_exists": True,
            "evaluation_status": "EVALUATION COMPLETE",
            "has_evaluation_results": True,
            "screening_method": screening_method,
            "metrics": metrics_data,
            "architecture": chk.get("architecture"),
            "reason": (
                f"Valid trained CNN loaded and evaluated on held-out test data "
                f"({metrics_data.get('test_sample_count')} samples, "
                f"Accuracy: {metrics_data.get('accuracy', 0.0) * 100:.1f}%, "
                f"F1: {metrics_data.get('f1_unsafe', 0.0):.3f})."
            ),
        }
    else:
        return {
            "status": "EVALUATION PENDING",
            "model_status_label": "EVALUATION PENDING",
            "is_valid_checkpoint": True,
            "checkpoint_exists": True,
            "evaluation_status": "EVALUATION PENDING",
            "has_evaluation_results": False,
            "screening_method": screening_method,
            "metrics": None,
            "architecture": chk.get("architecture"),
            "reason": (
                "Valid trained CNN checkpoint exists, but has not yet been evaluated on "
                "held-out test data (outputs/metrics/metrics.json pending)."
            ),
        }


def load_model_and_config(
    model_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Tuple[Optional[keras.Model], Dict[str, Any]]:
    """Safely load trained Keras model and configuration metadata.

    Returns (None, status_dict) if the model does not exist or fails verification.
    """
    m_path = model_path or config.BEST_MODEL_PATH
    c_path = config_path or config.MODEL_CONFIG_PATH

    chk = verify_cnn_checkpoint(m_path, c_path)
    if not chk["checkpoint_exists"]:
        return None, {
            "status": "CNN NOT TRAINED",
            "message": f"Checkpoint not found at {m_path}. Real-time posture screening operates using Rule-Based Ergonomic Assessment."
        }

    if not chk["is_valid"]:
        return None, {
            "status": chk["status"],
            "message": chk["message"]
        }

    try:
        model = keras.models.load_model(str(m_path))
    except Exception as exc:
        logger.error(f"Failed to load Keras model: {exc}")
        return None, {
            "status": "MODEL LOAD ERROR",
            "message": f"Error loading model from {m_path}: {exc}"
        }

    model_config: Dict[str, Any] = {}
    if c_path.exists():
        with open(c_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
    else:
        model_config = {
            "decision_threshold": config.DEFAULT_DECISION_THRESHOLD,
            "class_mapping": {str(config.LABEL_SAFE): "SAFE", str(config.LABEL_UNSAFE): "UNSAFE"}
        }

    return model, model_config


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    output_path: Optional[Path] = None,
    class_names: Optional[List[str]] = None,
) -> Path:
    """Render and save an annotated high-resolution confusion matrix heatmap.

    Emphasizes False Negatives (unsafe lifts misclassified as safe).
    """
    path = output_path or (config.PLOTS_DIR / "confusion_matrix.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    names = class_names or ["SAFE (0)", "UNSAFE (1)"]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    # Format annotations: count and percentage
    total = np.sum(cm)
    cm_labels = []
    for i in range(cm.shape[0]):
        row_labels = []
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total * 100) if total > 0 else 0.0
            row_labels.append(f"{val}\n({pct:.1f}%)")
        cm_labels.append(row_labels)

    sns.heatmap(
        cm,
        annot=np.array(cm_labels),
        fmt="",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar=True,
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )

    ax.set_title("Held-Out Test Set Confusion Matrix\n(Highlighting Dangerous False Negatives)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class (Ground Truth)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()
    return path


def evaluate_held_out_test_set(
    test_manifest_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    metrics_output_path: Optional[Path] = None,
    predictions_output_path: Optional[Path] = None,
    report_output_path: Optional[Path] = None,
    confusion_matrix_plot_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute evaluation on the held-out test manifest without data leakage.

    Computes all standard NNDL metrics, saves metrics.json, classification_report.txt,
    predictions.csv, and confusion_matrix.png.
    """
    test_path = test_manifest_path or (config.SPLITS_DIR / "test.csv")
    metrics_file = metrics_output_path or (config.METRICS_DIR / "metrics.json")
    if metrics_output_path is not None:
        report_file = report_output_path or (metrics_output_path.parent / "classification_report.txt")
        cm_plot_file = confusion_matrix_plot_path or (metrics_output_path.parent / "confusion_matrix.png")
    else:
        report_file = report_output_path or (config.METRICS_DIR / "classification_report.txt")
        cm_plot_file = confusion_matrix_plot_path or (config.PLOTS_DIR / "confusion_matrix.png")
    pred_file = predictions_output_path or (config.PREDICTIONS_DIR / "test_predictions.csv")

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    cm_plot_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Check model availability
    model, model_cfg = load_model_and_config(model_path, config_path)
    if model is None:
        return {
            "status": "CNN NOT TRAINED",
            "message": model_cfg.get("message", "Model checkpoint does not exist.")
        }

    # 2. Check test split availability
    if not test_path.exists():
        return {
            "status": "TEST SET NOT PREPARED",
            "message": f"Test manifest not found at {test_path}. Generate splits via src/dataset_loader.py first."
        }

    test_df = pd.read_csv(test_path)
    if len(test_df) == 0:
        return {
            "status": "EMPTY TEST SET",
            "message": "Test manifest contains 0 samples."
        }

    threshold = float(model_cfg.get("decision_threshold", config.DEFAULT_DECISION_THRESHOLD))

    # 3. Load test pose maps and measure inference latency
    test_maps: List[np.ndarray] = []
    y_true: List[int] = []
    valid_indices: List[int] = []

    for idx, row in test_df.iterrows():
        p_rel = Path(row["pose_map_path"])
        p_abs = p_rel if p_rel.is_absolute() else config.PROJECT_ROOT / p_rel
        if p_abs.exists():
            arr = np.load(str(p_abs))
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            test_maps.append(arr)
            y_true.append(int(row["label"]))
            valid_indices.append(idx)

    if not test_maps:
        return {
            "status": "MISSING TEST DATA FILES",
            "message": "None of the pose map files referenced in test.csv exist on disk."
        }

    X_test = np.array(test_maps, dtype=np.float32)
    y_true_arr = np.array(y_true, dtype=int)

    # Benchmark inference latency
    latencies = []
    # Warmup
    _ = model.predict(X_test[:min(3, len(X_test))], verbose=0)
    for i in range(len(X_test)):
        sample = X_test[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict(sample, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency_ms = round(float(np.mean(latencies)), 2)
    std_latency_ms = round(float(np.std(latencies)), 2)

    # Predictions
    raw_probs = model.predict(X_test, verbose=0).flatten()
    y_pred_arr = (raw_probs >= threshold).astype(int)

    # 4. Compute Metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec_unsafe = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    rec_unsafe = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    f1_unsafe = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC if both classes present in ground truth
    has_both_classes = len(np.unique(y_true_arr)) > 1
    roc_auc = float(roc_auc_score(y_true_arr, raw_probs)) if has_both_classes else None

    # Text report
    cls_report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=["SAFE", "UNSAFE"],
        zero_division=0
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== HELD-OUT TEST CLASSIFICATION REPORT ===\n")
        f.write(f"Decision Threshold: {threshold:.2f}\n")
        f.write(f"Inference Latency: {mean_latency_ms} ms +/- {std_latency_ms} ms\n\n")
        f.write(cls_report)

    # Plot Confusion Matrix
    plot_and_save_confusion_matrix(cm, output_path=cm_plot_file)

    # Save Predictions CSV
    eval_df = test_df.iloc[valid_indices].copy().reset_index(drop=True)
    eval_df["predicted_probability"] = [round(float(p), 4) for p in raw_probs]
    eval_df["predicted_label"] = y_pred_arr
    eval_df["prediction_correct"] = (eval_df["label"] == eval_df["predicted_label"])
    eval_df.to_csv(pred_file, index=False)

    results = {
        "status": "EVALUATION COMPLETE",
        "decision_threshold": threshold,
        "test_sample_count": len(y_true_arr),
        "test_class_distribution": {
            "safe": int(np.sum(y_true_arr == 0)),
            "unsafe": int(np.sum(y_true_arr == 1))
        },
        "accuracy": round(acc, 4),
        "precision_unsafe": round(prec_unsafe, 4),
        "recall_unsafe": round(rec_unsafe, 4),
        "f1_unsafe": round(f1_unsafe, 4),
        "specificity": round(specificity, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A (single class)",
        "confusion_matrix": {
            "true_negatives_safe": tn,
            "false_positives_false_alarm": fp,
            "false_negatives_missed_unsafe": fn,
            "true_positives_unsafe": tp,
        },
        "critical_safety_note": (
            f"False Negatives ({fn} missed unsafe postures): Represents highest ergonomic risk. "
            f"Recall for unsafe postures is {rec_unsafe:.1%}."
        ),
        "mean_latency_ms": mean_latency_ms,
        "std_latency_ms": std_latency_ms,
        "validation_vs_test_comparison": {
            "validation_f1": model_cfg.get("validation_metrics_at_threshold", {}).get("f1"),
            "test_f1": round(f1_unsafe, 4)
        }
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Test set evaluation completed successfully.")
    return results


if __name__ == "__main__":
    res = evaluate_held_out_test_set()
    print(json.dumps(res, indent=2))
