# Unsafe Lifting Posture Detection Using CNN and Pose Estimation

An academically grounded, end-to-end Computer Vision and Deep Learning system designed to analyze manual lifting mechanics from recorded videos or live webcam streams. The system extracts body landmarks using MediaPipe, renders standardized 2D skeletal pose maps, and classifies posture into **SAFE**, **UNSAFE**, or **UNCERTAIN / POSE NOT DETECTED** using a custom 2D Convolutional Neural Network (CNN).

> **Academic & Educational Disclaimer:**
> This software is an educational prototype and research demonstration developed for posture screening and ergonomic awareness. It is **not** a certified medical diagnostic device or clinical biomechanical measurement instrument.

---

## 1. Ergonomic Foundation & Problem Statement

Work-related musculoskeletal disorders (WMSDs), particularly lower back pain, represent one of the leading causes of occupational disability worldwide. According to the **National Institute for Occupational Safety and Health (NIOSH)**, unsafe lifting mechanics induce severe compressive and shear forces on the lumbar spine (predominantly at the **L5/S1** lumbosacral joint).

Biomechanical posture paradigms:
- **SAFE Lifting (Squat Lift):**
  - Characterized by deep knee flexion (knees bent $>80^\circ$), an upright trunk angle (trunk flexion $<45^\circ$), and load held close to the center of gravity.
  - The quadriceps and gluteal muscles bear the load, preserving lumbar lordosis and significantly reducing disc compression.
- **UNSAFE Lifting (Stoop Lift):**
  - Characterized by extended knees (legs nearly straight) and deep forward trunk flexion ($>45^\circ$).
  - Creates an extended lever arm for the torso, placing extreme tensile stress on the erector spinae muscles and high shear force on the lumbar discs.

---

## 2. System Architecture

```text
[ Input: Video / Live Webcam ]
               │
               ▼
   [ Frame Extraction (OpenCV) ]
               │
               ▼
[ MediaPipe PoseLandmarker (33 Joints) ]
               │
               ├────────────────────────────────────────┐
               ▼                                        ▼
    [ Quality & Visibility Gate ]            [ Explanatory Biomechanical Angles ]
 (Check critical joint confidence > 0.5)     (Trunk Inclination, Knee Flexion)
               │                                        │
      ┌────────┴────────┐                              │
      │ Valid           │ Low Quality / Missing        │
      ▼                 ▼                              │
[ Pose-Map Renderer ] [ Output: UNCERTAIN ]            │
 (128×128 1-Channel)    (No invalid classification)    │
      │                                                │
      ▼                                                │
[ Trained 2D CNN ]                                     │
 (Conv2D -> Pool -> GAP -> Dense -> Sigmoid)          │
      │                                                │
      ▼                                                │
[ Validation-Calibrated Decision Threshold ]           │
      │                                                │
      ▼                                                │
[ Classification: SAFE / UNSAFE ] ◄────────────────────┘
               │
               ▼
[ Temporal Rolling Window Smoothing & Debounce ]
               │
               ▼
[ Real-Time Visual HUD Overlay & Session Report ]
```

---

## 3. Repository Directory Structure

```text
unsafe-lifting-detection/
├── app.py                     # Interactive Streamlit CV Dashboard
├── requirements.txt           # Verified Python package dependencies
├── README.md                  # Comprehensive documentation and guidelines
├── config.py                  # Centralized project configuration and paths
├── data/
│   ├── raw_videos/            # Directory for raw MP4/AVI/MOV video recordings
│   ├── annotations.csv        # Master dataset annotations manifest
│   ├── processed/
│   │   ├── pose_maps/         # Standardized 128x128 single-channel pose maps
│   │   └── metadata.csv       # Extracted pose-map frame metadata
│   └── splits/                # Group-separated manifests (train.csv, val.csv, test.csv)
├── src/
│   ├── pose_extractor.py      # MediaPipe Pose Landmarker wrapper and angle calculations
│   ├── pose_renderer.py       # Deterministic 128x128 pose map canvas rendering
│   ├── dataset_loader.py      # Annotation validation, group splitting, and map caching
│   ├── train.py               # CNN training with EarlyStopping and threshold calibration
│   ├── evaluate.py            # Held-out test evaluation, confusion matrix, and metrics
│   ├── inference.py           # Real-time frame and video inference with rolling smoothing
│   └── reporting.py           # Session metrics aggregation and CSV report exports
├── models/
│   ├── pose_landmarker_lite.task # MediaPipe model asset
│   ├── best_lifting_cnn.keras    # Best trained Keras CNN checkpoint (when trained)
│   ├── model_config.json         # Decision threshold, input shapes, and class labels
│   └── preprocessing_config.json # Fixed canvas and topological rendering specs
├── outputs/
│   ├── metrics/               # metrics.json and classification_report.txt
│   ├── plots/                 # confusion_matrix.png and training_curves.png
│   └── predictions/           # test_predictions.csv
├── notebooks/
│   └── train_lifting_cnn_colab.ipynb # Fully runnable Google Colab training notebook
└── tests/
    ├── test_pose_renderer.py  # Unit tests for 128x128 canvas, topology, and visibility
    ├── test_dataset_splits.py # Tests for schema validation and leakage prevention
    └── test_model_pipeline.py # Tests for CNN input/output shapes, metrics, and inference
```

---

## 4. Installation & Local Setup (Windows / Linux)

### Prerequisites
- Python 3.10 – 3.13
- Git
- Webcam (optional, for live inference)

### Setup Steps
```bash
# 1. Clone or navigate to the workspace
cd d:\Download\Ismail

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Preparation & Preventing Data Leakage

### Annotation CSV Schema (`data/annotations.csv`)
The dataset must adhere to the following schema:
```csv
clip_path,participant_id,session_id,label,view
raw_videos/p1_s1_safe.mp4,P01,S01,0,lateral
raw_videos/p1_s1_unsafe.mp4,P01,S01,1,lateral
raw_videos/p2_s1_safe.mp4,P02,S01,0,lateral
raw_videos/p2_s1_unsafe.mp4,P02,S01,1,lateral
raw_videos/p3_s1_safe.mp4,P03,S01,0,lateral
raw_videos/p3_s1_unsafe.mp4,P03,S01,1,lateral
```

### Critical Rules to Prevent Data Leakage
1. **Group-Based Partitioning:** Splits are strictly stratified across `participant_id`. No subject appearing in the training set may ever appear in the validation or test sets.
2. **Never Split Adjacent Frames:** Adjacent frames from the same video share high temporal autocorrelation. Splitting individual frames randomly leads to severe data leakage and artificially inflated metrics.
3. **Threshold Calibration:** The classification threshold is selected strictly on the **validation set** to optimize F1-score. The held-out test set is evaluated only once with the frozen model.

---

## 6. Training Pipeline (Google Colab & Local CLI)

### Option A: Google Colab (Recommended for GPU Training)
1. Open [`notebooks/train_lifting_cnn_colab.ipynb`](notebooks/train_lifting_cnn_colab.ipynb) in Google Colab.
2. Mount your Google Drive or upload your video dataset and `annotations.csv`.
3. Follow the guided cells to extract pose maps, split by participant, train the CNN, calibrate the decision threshold, and evaluate on held-out test subjects.
4. Download the generated `models/` and `outputs/` folders back to your local repository.

### Option B: Local Training via CLI
```bash
# 1. Process dataset and generate group-based splits
python -m src.dataset_loader --prepare

# 2. Train the CNN and calibrate threshold
python -m src.train --epochs 35 --batch-size 16

# 3. Evaluate frozen model on held-out test split
python -m src.evaluate
```

---

## 7. Launching the Streamlit Application

Launch the interactive dashboard:
```bash
streamlit run app.py
```

### Dashboard Modules
- **Overview & Architecture:** System pipeline overview, readiness indicators, and NIOSH ergonomic principles.
- **Upload Video:** Analyzes recorded MP4/AVI/MOV videos, visualizes skeleton overlay, charts posture over time, and exports downloadable annotated video.
- **Live Webcam:** Live posture screening at 25–30 FPS with temporal smoothing, probability gauge, and alert debouncing.
- **Model Evaluation:** Visualizes genuine held-out test metrics, confusion matrix, ROC-AUC, and training curves.
- **Dataset Management:** Inspects and validates annotation files, checks participant distribution, and flags potential leakage risks.
- **Reports:** Summarizes detection sessions and exports timestamped prediction logs.

---

## 8. Automated Software Tests

Execute the automated test suite:
```bash
pytest -v
```

All tests pass on clean environments without requiring pre-existing real datasets, using strictly isolated synthetic fixtures.
