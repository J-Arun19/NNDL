"""Configuration module for Unsafe Lifting Posture Detection.

Defines all directory paths, hyperparameter presets, pose landmark definitions,
and evaluation thresholds across training, testing, and inference.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory hierarchy
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
PROCESSED_DIR = DATA_DIR / "processed"
POSE_MAPS_DIR = PROCESSED_DIR / "pose_maps"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Core file paths
ANNOTATIONS_FILE = DATA_DIR / "annotations.csv"
ANNOTATIONS_JSON_FILE = DATA_DIR / "annotations.json"
MANUAL_REVIEW_FILE = DATA_DIR / "manual_review_labels.csv"
AUTOMATIC_CANDIDATES_FILE = DATA_DIR / "automatic_label_candidates.csv"
METADATA_FILE = PROCESSED_DIR / "metadata.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_lifting_cnn.keras"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
PREPROCESSING_CONFIG_PATH = MODELS_DIR / "preprocessing_config.json"
POSE_LANDMARKER_TASK_PATH = MODELS_DIR / "pose_landmarker_lite.task"
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Pose Map Representation parameters
POSE_MAP_HEIGHT = 128
POSE_MAP_WIDTH = 128
POSE_MAP_CHANNELS = 1
POSE_MAP_SHAPE = (POSE_MAP_HEIGHT, POSE_MAP_WIDTH, POSE_MAP_CHANNELS)

# Keypoint rendering configuration
JOINT_RADIUS = 3
BONE_THICKNESS = 2
JOINT_INTENSITY = 255     # Pure white for joint nodes
BONE_INTENSITY = 180      # High grayscale for bone connections
BACKGROUND_INTENSITY = 0  # Black background

# Landmark quality and visibility constraints
MIN_LANDMARK_VISIBILITY = 0.5
MIN_CRITICAL_KEYPOINTS_RATIO = 0.7  # At least 70% of critical joints must be visible

# MediaPipe 33 Landmark indices of interest for ergonomic lifting:
# 11, 12: shoulders; 13, 14: elbows; 15, 16: wrists
# 23, 24: hips; 25, 26: knees; 27, 28: ankles; 29, 30: heels; 31, 32: foot index
CRITICAL_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Body skeletal connections for 2D pose map rendering
SKELETAL_CONNECTIONS = [
    # Torso
    (11, 12),  # shoulder - shoulder
    (11, 23),  # left shoulder - left hip
    (12, 24),  # right shoulder - right hip
    (23, 24),  # hip - hip
    # Left arm
    (11, 13),  # left shoulder - left elbow
    (13, 15),  # left elbow - left wrist
    # Right arm
    (12, 14),  # right shoulder - right elbow
    (14, 16),  # right elbow - right wrist
    # Left leg
    (23, 25),  # left hip - left knee
    (25, 27),  # left knee - left ankle
    (27, 29),  # left ankle - left heel
    (29, 31),  # left heel - left foot index
    (27, 31),  # left ankle - left foot index
    # Right leg
    (24, 26),  # right hip - right knee
    (26, 28),  # right knee - right ankle
    (28, 30),  # right ankle - right heel
    (30, 32),  # right heel - right foot index
    (28, 32),  # right ankle - right foot index
]

# Classification mapping
LABEL_SAFE = 0
LABEL_UNSAFE = 1
CLASS_NAMES = {LABEL_SAFE: "SAFE", LABEL_UNSAFE: "UNSAFE"}
STATUS_UNCERTAIN = "INSUFFICIENT VISIBILITY"

# Machine Learning & Training Configuration
RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 35
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_PATIENCE = 7
DEFAULT_DECISION_THRESHOLD = 0.5

# Validation & Test split ratios (Group-based by participant/session)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Real-time inference & Smoothing
DEFAULT_ROLLING_WINDOW = 5
DEFAULT_ALERT_DEBOUNCE = 3  # consecutive unsafe frames required to alert
DEFAULT_FRAME_SAMPLE_RATE = 5  # FPS to sample from recorded videos

# Biomechanical Angle Thresholds (Ergonomic references, purely explanatory)
# Based on NIOSH / OSHA manual lifting guidelines:
# Squat lift: Knees flexed (> 80 deg), Trunk relatively upright (trunk inclination < 45 deg)
# Stoop lift: Knees extended (> 140 deg), Trunk deeply flexed (trunk inclination > 45 deg)
MAX_SAFE_TRUNK_FLEXION_DEG = 45.0
MIN_SAFE_KNEE_FLEXION_DEG = 80.0
