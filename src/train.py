"""Convolutional Neural Network training module for Unsafe Lifting Posture Detection.

Builds a 2D CNN architecture, trains on 128x128 pose maps, prevents data leakage
using participant-grouped manifests, uses validation-only threshold tuning,
and outputs training diagnostic plots and model checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config

logger = logging.getLogger(__name__)


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set seeds for Python, NumPy, and TensorFlow to ensure deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lifting_cnn(
    input_shape: Tuple[int, int, int] = config.POSE_MAP_SHAPE,
    dropout_rate: float = config.DEFAULT_DROPOUT_RATE,
    learning_rate: float = config.DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Construct the 2D CNN for lifting posture classification.

    Architecture:
    - Input: (128, 128, 1) single-channel pose maps
    - 4 Conv blocks with Batch Normalization, ReLU, and MaxPooling
    - Global Average Pooling (reducing spatial dimensions and preventing overfitting)
    - Dense representation (64 units) with Dropout
    - Sigmoid binary classification output (0 = SAFE, 1 = UNSAFE)
    """
    inputs = layers.Input(shape=input_shape, name="pose_map_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="posture_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LiftingPostureCNN")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


class PoseMapDatasetGenerator(keras.utils.Sequence):
    """Custom Keras data generator that lazily loads 128x128 pose maps from disk."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        batch_size: int = config.DEFAULT_BATCH_SIZE,
        shuffle: bool = True,
        augment: bool = False,
        base_dir: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.df = dataframe.copy().reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.base_dir = base_dir or config.PROJECT_ROOT
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(math.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_rows = self.df.iloc[batch_idx]

        batch_x = []
        batch_y = []

        for _, row in batch_rows.iterrows():
            p_rel = Path(row["pose_map_path"])
            p_abs = p_rel if p_rel.is_absolute() else self.base_dir / p_rel

            if p_abs.exists():
                map_arr = np.load(str(p_abs))
                if map_arr.ndim == 2:
                    map_arr = map_arr[:, :, np.newaxis]
                # Normalize [0.0, 1.0]
                if map_arr.dtype == np.uint8:
                    map_arr = map_arr.astype(np.float32) / 255.0
            else:
                # Return zeros if file missing
                map_arr = np.zeros(config.POSE_MAP_SHAPE, dtype=np.float32)

            # Label-preserving horizontal flip augmentation
            if self.augment and np.random.rand() > 0.5:
                map_arr = np.fliplr(map_arr)

            batch_x.append(map_arr)
            batch_y.append(float(row["label"]))

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)


def optimize_classification_threshold(
    model: keras.Model,
    val_generator: PoseMapDatasetGenerator,
    val_labels: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Determine the optimal decision threshold exclusively on the validation set.

    Sweeps candidate thresholds from 0.1 to 0.9 to maximize F1-score on the unsafe class,
    preventing test-set contamination.
    """
    val_probs = model.predict(val_generator, verbose=0).flatten()
    val_y = np.array(val_labels).flatten()

    best_threshold = config.DEFAULT_DECISION_THRESHOLD
    best_f1 = -1.0
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # If only 1 class present in validation set, fallback gracefully
    if len(np.unique(val_y)) < 2:
        logger.warning("Validation set has only single class. Retaining default threshold 0.5.")
        return 0.5, best_metrics

    for thresh in np.arange(0.10, 0.92, 0.02):
        preds = (val_probs >= thresh).astype(int)
        # Prioritize F1 for UNSAFE class (1)
        prec = precision_score(val_y, preds, pos_label=1, zero_division=0)
        rec = recall_score(val_y, preds, pos_label=1, zero_division=0)
        f1 = f1_score(val_y, preds, pos_label=1, zero_division=0)

        # F1 objective, with tie-break favoring higher recall (safety-critical)
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-4 and rec > best_metrics["recall"]):
            best_f1 = f1
            best_threshold = round(float(thresh), 2)
            best_metrics = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    return best_threshold, best_metrics


def plot_and_save_training_curves(
    history: keras.callbacks.History,
    output_path: Optional[Path] = None
) -> Path:
    """Plot and save training and validation loss and accuracy trajectories."""
    path = output_path or (config.PLOTS_DIR / "training_curves.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Loss Curves
    ax1.plot(epochs_range, history.history["loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history.history:
        ax1.plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Binary Cross-Entropy Loss Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # 2. Accuracy Curves
    ax2.plot(epochs_range, history.history["accuracy"], label="Train Accuracy", color="#2ca02c", lw=2)
    if "val_accuracy" in history.history:
        ax2.plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy", color="#d62728", lw=2, linestyle="--")
    ax2.set_title("Classification Accuracy Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()
    return path


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    learning_rate: float = config.DEFAULT_LEARNING_RATE,
    dropout_rate: float = config.DEFAULT_DROPOUT_RATE,
    seed: int = config.RANDOM_SEED,
    model_save_path: Optional[Path] = None,
    config_save_path: Optional[Path] = None,
    curves_plot_path: Optional[Path] = None,
) -> Tuple[keras.Model, Dict[str, Any]]:
    """Execute complete CNN training workflow with early stopping, checkpointing, and threshold tuning."""
    set_seed(seed)
    save_path = model_save_path or config.BEST_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = config_save_path or config.MODEL_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Class weighting to address dataset imbalance
    y_train = train_df["label"].values.astype(int)
    unique_classes = np.unique(y_train)
    if len(unique_classes) > 1:
        weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    else:
        class_weight_dict = {0: 1.0, 1: 1.0}

    # Generators
    train_gen = PoseMapDatasetGenerator(train_df, batch_size=batch_size, shuffle=True, augment=True)
    val_gen = PoseMapDatasetGenerator(val_df, batch_size=batch_size, shuffle=False, augment=False)

    model = build_lifting_cnn(
        input_shape=config.POSE_MAP_SHAPE,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.DEFAULT_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save training curves plot
    if curves_plot_path is not None:
        plot_and_save_training_curves(history, output_path=curves_plot_path)
    elif model_save_path is not None:
        plot_and_save_training_curves(history, output_path=model_save_path.parent / "training_curves.png")
    else:
        plot_and_save_training_curves(history, output_path=config.PLOTS_DIR / "training_curves.png")

    # Threshold calibration on validation set only
    val_y = val_df["label"].values.astype(int)
    optimal_thresh, val_metrics = optimize_classification_threshold(model, val_gen, val_y)

    # Construct metadata
    training_summary = {
        "model_architecture": "2D-CNN-PoseMap",
        "input_shape": list(config.POSE_MAP_SHAPE),
        "decision_threshold": optimal_thresh,
        "class_mapping": {str(config.LABEL_SAFE): "SAFE", str(config.LABEL_UNSAFE): "UNSAFE"},
        "random_seed": seed,
        "epochs_trained": len(history.history["loss"]),
        "best_train_loss": round(float(min(history.history["loss"])), 4),
        "best_val_loss": round(float(min(history.history.get("val_loss", [0.0]))), 4),
        "best_val_accuracy": round(float(max(history.history.get("val_accuracy", [0.0]))), 4),
        "validation_metrics_at_threshold": val_metrics,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "training_timestamp": datetime.now().isoformat(),
    }

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    logger.info(f"Model and configuration saved successfully. Optimal threshold: {optimal_thresh}")
    return model, training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lifting Posture Detection CNN")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    # Load splits
    train_manifest = config.SPLITS_DIR / "train.csv"
    val_manifest = config.SPLITS_DIR / "val.csv"

    if not train_manifest.exists() or not val_manifest.exists():
        print("ERROR: Split manifests not found. Please prepare dataset first using src/dataset_loader.py")
        exit(1)

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    train_model(
        train_df,
        val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed
    )
