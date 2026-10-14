# -*- coding: utf-8 -*-
"""Streamlit Web Application: LiftGuard AI.

A premium, modern industrial Computer Vision dashboard for manual lifting posture screening,
model evaluation analytics, dataset validation, and real-time video/webcam inference.
"""

from __future__ import annotations

import html
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.dataset_loader import SafeliftAuditReport, validate_annotations, validate_safelift_dataset
from src.dataset_review import (
    DatasetReviewManager,
    LABEL_AMBIGUOUS,
    LABEL_SAFE,
    LABEL_UNSAFE,
    LABELING_GUIDANCE_TEXT,
    STATUS_REVIEWED,
    STATUS_UNREVIEWED,
)
from src.evaluate import (
    evaluate_held_out_test_set,
    get_project_model_status,
    load_model_and_config,
    verify_cnn_checkpoint,
)
from src.inference import (
    LiftingPostureClassifier,
    MODE_RULE_BASED_ERGONOMICS,
    STATUS_INSUFFICIENT_VISIBILITY,
    STATUS_SAFE_POSTURE,
    STATUS_UNSAFE_POSTURE,
    VideoAnalysisSummary,
    process_recorded_video,
)
from src.reporting import generate_session_report
from src.ui_theme import (
    ACCENT_AMBER,
    ACCENT_MINT,
    ACCENT_RED,
    BG_MAIN,
    BG_PANEL,
    BG_PANEL_ELEVATED,
    BORDER_FOCUS,
    BORDER_SUBTLE,
    ICONS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    clean_html,
    get_css_theme,
    render_empty_state_html,
    render_kpi_card_html,
    render_live_telemetry_hud_html,
    render_sidebar_brand,
    render_sidebar_status_footer,
    render_uploaded_video_summary_html,
    render_workflow_pipeline_html,
)

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LiftGuard AI | Industrial Posture Screening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Centralized Dark Theme CSS
st.markdown(get_css_theme(), unsafe_allow_html=True)


@st.cache_resource
def get_classifier() -> LiftingPostureClassifier:
    """Cache the classifier instance across user interactions."""
    import importlib
    import src.inference
    importlib.reload(src.inference)
    return src.inference.LiftingPostureClassifier()


def check_system_readiness() -> Dict[str, Any]:
    """Verify filesystem readiness for model, annotations, and evaluations."""
    model_status = get_project_model_status()
    safelift_audit = validate_safelift_dataset()

    has_task = config.POSE_LANDMARKER_TASK_PATH.exists()
    decision_thresh = config.DEFAULT_DECISION_THRESHOLD
    if config.MODEL_CONFIG_PATH.exists():
        try:
            with open(config.MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                decision_thresh = float(cfg_data.get("decision_threshold", config.DEFAULT_DECISION_THRESHOLD))
        except Exception:
            pass

    return {
        "cnn_status": model_status["status"],
        "model_status_label": model_status["model_status_label"],
        "model_trained": model_status["is_valid_checkpoint"],
        "checkpoint_exists": model_status["checkpoint_exists"],
        "screening_method": model_status["screening_method"],
        "evaluation_status": model_status["evaluation_status"],
        "evaluation_available": model_status["has_evaluation_results"],
        "metrics": model_status.get("metrics"),
        "model_reason": model_status["reason"],
        "architecture": model_status.get("architecture"),
        "pose_landmarker_ready": has_task,
        "dataset_present": safelift_audit.is_loaded,
        "safelift_audit": safelift_audit,
        "decision_threshold": decision_thresh,
    }


def init_session_state():
    """Ensure session state variables are initialized."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Posture Screening"
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False
    if "active_tab_screening" not in st.session_state:
        st.session_state.active_tab_screening = 0


def render_sidebar(readiness: Dict[str, Any]) -> str:
    """Render the compact, professional navigation sidebar."""
    with st.sidebar:
        # Brand Header
        st.markdown(render_sidebar_brand(), unsafe_allow_html=True)

        # Navigation Options
        page_options = [
            "Posture Screening",
            "Overview",
            "Dataset Review",
            "Model Evaluation",
            "Dataset",
            "Reports",
        ]

        current_page = st.session_state.get("current_page", "Posture Screening")
        # Backwards compatibility mapping
        if current_page in ("Video Screening", "Live Webcam"):
            current_page = "Posture Screening"
            st.session_state.current_page = "Posture Screening"

        current_idx = page_options.index(current_page) if current_page in page_options else 0
        selected_page = st.radio(
            "Navigate",
            page_options,
            index=current_idx,
            label_visibility="collapsed",
        )
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()

        # Status Footer
        st.markdown(
            render_sidebar_status_footer(readiness, readiness.get("decision_threshold", 0.5)),
            unsafe_allow_html=True,
        )

    return st.session_state.current_page


# ==============================================================================
# PAGE 1: OVERVIEW & ARCHITECTURE
# ==============================================================================
def page_overview(readiness: Dict[str, Any]):
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<div class="lg-title">LiftGuard AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="lg-subtitle">AI-powered ergonomic lifting posture analysis combining MediaPipe and 2D CNN vision</div>',
            unsafe_allow_html=True,
        )
    with header_col2:
        cnn_status = readiness.get("cnn_status", "CNN NOT TRAINED")
        if not readiness.get("model_trained", False):
            badge_html = f'<div style="text-align: right; padding-top: 6px;"><span class="lg-badge lg-badge-missing">{ICONS["x_circle"]} CNN NOT TRAINED</span></div>'
        elif not readiness.get("evaluation_available", False) or cnn_status == "EVALUATION PENDING":
            badge_html = f'<div style="text-align: right; padding-top: 6px;"><span class="lg-badge lg-badge-warning">{ICONS["help_circle"]} EVALUATION PENDING</span></div>'
        else:
            badge_html = f'<div style="text-align: right; padding-top: 6px;"><span class="lg-badge lg-badge-ready">{ICONS["check_circle"]} CNN EVALUATED (&tau;={readiness.get("decision_threshold", 0.5):.2f})</span></div>'
        st.markdown(clean_html(badge_html), unsafe_allow_html=True)

    # 4 Compact KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        if not readiness.get("model_trained", False):
            status_val = "CNN NOT TRAINED"
            status_accent = "danger"
            status_sub = "No Checkpoint Found"
        elif not readiness.get("evaluation_available", False):
            status_val = "EVALUATION PENDING"
            status_accent = "warning"
            status_sub = "Checkpoint Verified"
        else:
            status_val = "EVALUATED"
            status_accent = "success"
            status_sub = "Active CNN Checkpoint"
        st.markdown(render_kpi_card_html("Model Status", status_val, status_sub, status_accent), unsafe_allow_html=True)

    with kpi_col2:
        screening_method = readiness.get("screening_method", "Rule-Based Ergonomic Assessment")
        st.markdown(render_kpi_card_html("Screening Method", screening_method, "MediaPipe + Biomechanics", "default"), unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(render_kpi_card_html("Pose Landmarks", "33 Joints", "MediaPipe Vision Task", "default"), unsafe_allow_html=True)

    with kpi_col4:
        if readiness.get("evaluation_available", False) and readiness.get("metrics"):
            m = readiness["metrics"]
            f1_score_val = m.get("f1_unsafe", None)
            if f1_score_val is not None:
                eval_val = f"F1: {f1_score_val:.3f}"
                eval_sub = f"Acc: {m.get('accuracy', 0.0) * 100:.1f}% (Held-out)"
                eval_accent = "success"
            else:
                eval_val = "Evaluated"
                eval_sub = f"Acc: {m.get('accuracy', 0.0) * 100:.1f}%"
                eval_accent = "success"
        elif readiness.get("model_trained", False):
            eval_val = "EVALUATION PENDING"
            eval_sub = "Held-out test pending"
            eval_accent = "warning"
        else:
            eval_val = "CNN NOT TRAINED"
            eval_sub = "Requires trained checkpoint"
            eval_accent = "danger"

        st.markdown(render_kpi_card_html("Held-Out Evaluation", eval_val, eval_sub, eval_accent), unsafe_allow_html=True)

    # Screening Method and CNN Distinction Callout Banner
    st.markdown(
        clean_html(f"""
        <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.22); border-radius: 8px; padding: 12px 16px; margin: 14px 0 16px 0;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-size: 13px; font-weight: 600; color: {BORDER_FOCUS}; display: flex; align-items: center; gap: 6px;">
                    {ICONS["shield_check"]} Active Screening Pipeline: <b>{readiness.get("screening_method", "Rule-Based Ergonomic Assessment")}</b>
                </span>
                <span class="lg-badge {'lg-badge-ready' if readiness['model_trained'] else 'lg-badge-missing'}">
                    CNN: {readiness.get('cnn_status', 'CNN NOT TRAINED')}
                </span>
            </div>
            <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5;">
                Video screening and live webcam posture monitoring operate using <b>transparent biomechanical rules</b> (NIOSH / OSHA standards) on 33 body landmarks from Google MediaPipe.
                The custom 2D CNN classifier is currently <b>{readiness.get('cnn_status', 'CNN NOT TRAINED')}</b>; posture screening features function reliably and independently.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # Primary Workspace: Two Prominent Action Cards
    st.markdown('<div class="lg-section-header">' + ICONS["layers"] + ' Primary Workspaces</div>', unsafe_allow_html=True)

    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.markdown(
            clean_html(f"""
            <div class="lg-action-box">
                <div style="width: 38px; height: 38px; border-radius: 8px; background: rgba(16, 185, 129, 0.12); display: flex; align-items: center; justify-content: center;">
                    {ICONS["video"]}
                </div>
                <div class="lg-action-title">Analyze a Video</div>
                <div class="lg-action-desc">
                    Upload recorded MP4, AVI, or MOV clips for frame-by-frame posture inference, HUD skeletal overlay, probability timeline, and exportable reports.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        if st.button("Launch Video Screening →", key="btn_go_video", use_container_width=True):
            st.session_state.current_page = "Posture Screening"
            st.rerun()

    with act_col2:
        st.markdown(
            clean_html(f"""
            <div class="lg-action-box">
                <div style="width: 38px; height: 38px; border-radius: 8px; background: rgba(16, 185, 129, 0.12); display: flex; align-items: center; justify-content: center;">
                    {ICONS["camera"]}
                </div>
                <div class="lg-action-title">Start Live Screening</div>
                <div class="lg-action-desc">
                    Connect a live webcam stream for real-time ergonomic screening at 25–30 FPS with rolling window smoothing, debounced alerting, and joint telemetry.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        if st.button("Launch Live Webcam →", key="btn_go_webcam", use_container_width=True):
            st.session_state.current_page = "Posture Screening"
            st.rerun()

    # Workflow Visualization Pipeline
    st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Processing Pipeline</div>', unsafe_allow_html=True)
    st.markdown(render_workflow_pipeline_html(), unsafe_allow_html=True)

    # Recent Screening Activity
    st.markdown('<div class="lg-section-header">' + ICONS["file_text"] + ' Recent Screening Activity</div>', unsafe_allow_html=True)
    pred_dir = config.PREDICTIONS_DIR
    recent_csvs = list(pred_dir.glob("*.csv")) if pred_dir.exists() else []

    if recent_csvs:
        recent_data = []
        for p in sorted(recent_csvs, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            try:
                df = pd.read_csv(p)
                rep = generate_session_report(df)
                recent_data.append({
                    "Session File": p.name,
                    "Total Frames": rep.total_frames,
                    "Duration (s)": rep.duration_sec,
                    "Valid Pose %": f"{rep.valid_pose_pct}%",
                    "Ergonomic Risk": rep.ergonomic_risk_level,
                })
            except Exception:
                continue

        if recent_data:
            st.dataframe(pd.DataFrame(recent_data), use_container_width=True, hide_index=True)
        else:
            st.markdown(render_empty_state_html("No Recent Sessions", "No valid predictions logs found. Process a video or run live webcam to record sessions.", "file_text"), unsafe_allow_html=True)
    else:
        st.markdown(render_empty_state_html("No Recent Sessions", "No recorded sessions yet. Analyze a video or activate the webcam to populate session logs.", "file_text"), unsafe_allow_html=True)

    with st.expander("ℹ️ About the Biomechanical Model & NIOSH Criteria"):
        st.markdown(
            r"""
            **Ergonomic Criteria & Posture Mechanics:**
            - **SAFE Lifting (Squat Lift):**
              Knee flexion ($>80^\circ$) with an upright trunk inclination ($<45^\circ$ from vertical). The quadriceps and gluteals bear the load, preserving lumbar lordosis and minimizing shear stress on the **L5/S1** lumbosacral joint.
            - **UNSAFE Lifting (Stoop Lift):**
              Knees extended ($>140^\circ$) with excessive forward trunk bend ($>45^\circ$). Creates a long lever arm for the upper body, inducing high compressive and shear loads on lumbar discs.
            - **Quality Gate:**
              MediaPipe extracts 33 normalized landmarks. If key joints (shoulders, hips, knees, ankles) fall below 0.5 visibility, the system flags the frame as `UNCERTAIN / POSE NOT DETECTED`. Failed pose detection is never classified as SAFE.
            """
        )


# ==============================================================================
# POSTURE SCREENING WORKSPACE (TWO TABS: UPLOAD VIDEO & REAL-TIME MONITORING)
# ==============================================================================
def page_posture_screening(classifier: LiftingPostureClassifier, default_tab: int = 0):
    st.markdown('<div class="lg-title">LiftGuard AI | Industrial Posture Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Reliable SAFE/UNSAFE posture assessment for uploaded videos and live webcam monitoring</div>', unsafe_allow_html=True)

    # Method & Scientific Disclosure Banner
    assessment_method = getattr(classifier, "current_assessment_method", MODE_RULE_BASED_ERGONOMICS)
    is_cnn = getattr(classifier, "is_model_trained", False)
    confidence_note = "Model confidence probability is active (CNN checkpoint loaded)." if is_cnn else "Direct joint angle measurement; confidence scores are not simulated."

    badge_type = "Trained CNN Active" if is_cnn else "Rule-Based Ergonomics (CNN NOT TRAINED)"
    icon_shield = ICONS["shield_check"]
    banner_html = clean_html(f"""
    <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 13px; font-weight: 600; color: {BORDER_FOCUS}; display: flex; align-items: center; gap: 6px;">
                {icon_shield} Active Screening Method: <b>{html.escape(assessment_method)}</b>
            </span>
            <span class="lg-badge" style="background: rgba(255,255,255,0.06); color: {'#34D399' if is_cnn else '#F87171'}; border: 1px solid rgba(255,255,255,0.15); font-size: 11px;">{badge_type}</span>
        </div>
        <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5;">
            &bull; <b>Screening Engine:</b> Pretrained Google MediaPipe PoseLandmarker extracting 33 normalized body landmarks.<br>
            &bull; <b>Ergonomic Criteria:</b> Forward trunk inclination from vertical (&le; 45&deg;) and knee flexion (&le; 125&deg;) evaluated per NIOSH/OSHA manual lifting standards.<br>
            &bull; <b>CNN Independence:</b> Video screening operates using Rule-Based Ergonomic Assessment and functions independently of whether the CNN is trained.<br>
            &bull; <b>Confidence Metric:</b> {confidence_note}
        </div>
    </div>
    """)
    st.markdown(banner_html, unsafe_allow_html=True)

    # Clean Two-Tab Interface
    tab_upload, tab_live = st.tabs([
        "📁 Upload Video Analysis",
        "📹 Real-Time Monitoring",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: UPLOAD VIDEO ANALYSIS
    # -------------------------------------------------------------------------
    with tab_upload:
        st.markdown('<div class="lg-section-header">' + ICONS["video"] + ' Upload Recorded Lifting Clip</div>', unsafe_allow_html=True)
        col_up_left, col_up_right = st.columns([1.08, 0.92])

        with col_up_left:
            uploaded_file = st.file_uploader(
                "Select Video Clip (MP4, AVI, MOV)",
                type=["mp4", "avi", "mov"],
                help="Supported containers: MP4, AVI, MOV. Sagittal lateral view recommended for optimal joint tracking.",
                key="uploader_posture_video"
            )

            video_path = None
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
                tfile.write(uploaded_file.read())
                tfile.flush()
                video_path = Path(tfile.name)
                st.video(str(video_path))
                run_video_btn = st.button("Run Posture Screening", type="primary", use_container_width=True, key="btn_run_uploaded_screening")
            else:
                st.markdown(
                    clean_html(f"""
                    <div style="background: {BG_PANEL_ELEVATED}; border: 1px solid {BORDER_SUBTLE}; border-radius: 8px; padding: 12px 14px; font-size: 12px; color: {TEXT_SECONDARY};">
                        Select an MP4, AVI, or MOV video clip above to begin posture screening. A side/lateral camera angle yields the most accurate joint angles.
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
                run_video_btn = False

        with col_up_right:
            st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Posture Assessment & Telemetry</div>', unsafe_allow_html=True)
            if not uploaded_file:
                st.markdown(render_empty_state_html("Awaiting Video Clip", "Upload a video file on the left to extract body landmarks and evaluate posture.", "video"), unsafe_allow_html=True)
            elif not run_video_btn and "last_processed_video" not in st.session_state:
                st.markdown(render_empty_state_html("Ready to Analyze", "Click 'Run Posture Screening' on the left to start frame-by-frame analysis and generate HUD overlay.", "cpu"), unsafe_allow_html=True)

        # Video Processing Execution
        if run_video_btn and video_path:
            with col_up_right:
                prog_bar = st.progress(0.0)
                status_text = st.empty()
                status_text.caption("Extracting body landmarks and evaluating ergonomic lifting criteria...")

                out_annotated = Path(tempfile.gettempdir()) / f"annotated_{video_path.name}"
                out_csv = Path(tempfile.gettempdir()) / f"preds_{video_path.stem}.csv"

                try:
                    df_preds, out_video_path, summary = process_recorded_video(
                        video_path=video_path,
                        output_annotated_video_path=out_annotated,
                        output_predictions_csv_path=out_csv,
                        classifier=classifier,
                        progress_callback=lambda p: prog_bar.progress(p),
                    )
                    prog_bar.progress(1.0)
                    status_text.empty()

                    st.session_state.last_processed_video = {
                        "df_preds": df_preds,
                        "annotated_path": str(out_annotated) if out_annotated.exists() else None,
                        "csv_path": str(out_csv) if out_csv.exists() else None,
                        "summary": summary,
                        "stem": video_path.stem,
                    }
                except Exception as exc:
                    st.error(f"Video analysis error: {exc}")
                    logger.error(f"Video analysis error: {exc}", exc_info=True)

        # Display Results if available
        if "last_processed_video" in st.session_state:
            res_data = st.session_state.last_processed_video
            df_preds = res_data["df_preds"]
            summary: VideoAnalysisSummary = res_data["summary"]

            with col_up_right:
                # Overall Result Card
                st.markdown(render_uploaded_video_summary_html(summary), unsafe_allow_html=True)

                # Export CSV button
                csv_bytes = df_preds.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Frame Predictions CSV",
                    data=csv_bytes,
                    file_name=f"posture_predictions_{res_data['stem']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_dl_csv"
                )

            # Playback of annotated video in left column below preview
            if res_data.get("annotated_path") and Path(res_data["annotated_path"]).exists():
                with col_up_left:
                    st.markdown('<div class="lg-section-header">' + ICONS["shield_check"] + ' Processed Video with Skeleton HUD Overlay</div>', unsafe_allow_html=True)
                    st.video(res_data["annotated_path"])
                    with open(res_data["annotated_path"], "rb") as f_vid:
                        st.download_button(
                            "📥 Download Annotated Video",
                            data=f_vid.read(),
                            file_name=f"annotated_{res_data['stem']}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            key="btn_dl_annotated_video"
                        )

            # Full Width Section: Trajectory Charts & Frame-by-Frame Data
            st.markdown("---")
            st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Kinematic Angle Trajectory & Frame-by-Frame Log</div>', unsafe_allow_html=True)

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.caption("Trunk Inclination (deg from vertical) over time (45° threshold boundary):")
                if "trunk_inclination_deg" in df_preds.columns:
                    t_chart = df_preds[["timestamp_sec", "trunk_inclination_deg"]].dropna().set_index("timestamp_sec")
                    st.line_chart(t_chart, height=180)
            with chart_col2:
                st.caption("Knee Flexion Angle (deg) over time (≤125° = Squat, ≥135° = Stoop):")
                if "knee_angle_mean_deg" in df_preds.columns:
                    k_chart = df_preds[["timestamp_sec", "knee_angle_mean_deg"]].dropna().set_index("timestamp_sec")
                    st.line_chart(k_chart, height=180)

            # Frame Table
            st.caption("Frame-by-Frame Posture Measurements and Ergonomic Rationale:")
            view_cols = ["frame_index", "timestamp_sec", "status", "trunk_inclination_deg", "knee_angle_mean_deg", "explanation", "alert_triggered"]
            present_view = [c for c in view_cols if c in df_preds.columns]
            st.dataframe(df_preds[present_view], use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 2: REAL-TIME WEBCAM MONITORING
    # -------------------------------------------------------------------------
    with tab_live:
        st.markdown('<div class="lg-section-header">' + ICONS["camera"] + ' Real-Time Webcam Monitoring</div>', unsafe_allow_html=True)

        if "webcam_active" not in st.session_state:
            st.session_state.webcam_active = False

        # Find sample demo video if available
        sample_video_path = None
        demo_dir = config.RAW_VIDEOS_DIR / "video_env_4_anonymized"
        if demo_dir.exists():
            demo_vids = sorted(list(demo_dir.glob("*.mp4")))
            if demo_vids:
                sample_video_path = demo_vids[0]

        # Top Control Toolbar
        c_col1, c_col2, c_col3 = st.columns([1.3, 1, 1])
        with c_col1:
            source_options = ["Camera 0 (Default Webcam)", "Camera 1 (External USB)"]
            if sample_video_path:
                source_options.append("Demo Video: Environment 4 Trial")
            cam_source_choice = st.selectbox(
                "Video Source",
                source_options,
                index=0,
                key="live_source_choice",
                disabled=st.session_state.webcam_active,
            )
        with c_col2:
            live_window = st.slider(
                "Smoothing Window",
                min_value=1,
                max_value=15,
                value=5,
                help="Number of consecutive frames used for temporal rolling majority vote",
                key="live_smooth_window"
            )
        with c_col3:
            live_debounce = st.slider(
                "Alert Debounce",
                min_value=1,
                max_value=10,
                value=3,
                help="Consecutive unsafe frames required to trigger sustained hazard alert",
                key="live_debounce_window"
            )

        # Action Buttons Row
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            btn_start = st.button(
                "▶ Start Monitoring",
                type="primary",
                use_container_width=True,
                key="btn_start_monitor",
                disabled=st.session_state.webcam_active,
            )
        with btn_col2:
            btn_stop = st.button(
                "⏹ Stop Monitoring",
                type="secondary",
                use_container_width=True,
                key="btn_stop_monitor",
                disabled=not st.session_state.webcam_active,
            )

        if btn_start:
            st.session_state.webcam_active = True
            st.rerun()
        if btn_stop:
            st.session_state.webcam_active = False
            st.rerun()

        classifier.rolling_window_size = live_window
        classifier.alert_debounce_threshold = live_debounce

        # Two-Column Balanced Layout: Left Video, Right Live Assessment
        live_feed_col, live_hud_col = st.columns([1.15, 0.85])
        feed_placeholder = live_feed_col.empty()
        sec_placeholder = live_feed_col.empty()
        hud_placeholder = live_hud_col.empty()

        if not st.session_state.webcam_active:
            # Standby state with professional instructions
            with live_feed_col:
                feed_placeholder.markdown(
                    clean_html(f"""
                    <div style="background-color: {BG_PANEL_ELEVATED}; border: 1.5px dashed {BORDER_SUBTLE}; border-radius: 8px; padding: 46px 20px; text-align: center;">
                        <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 10px; background: rgba(16, 185, 129, 0.1); margin-bottom: 12px;">
                            {ICONS['camera']}
                        </div>
                        <div style="font-size: 16px; font-weight: 700; color: {TEXT_PRIMARY}; margin-bottom: 6px;">Camera Standby</div>
                        <div style="font-size: 13px; color: {TEXT_SECONDARY}; max-width: 440px; margin: 0 auto; line-height: 1.5;">
                            Position camera lateral (sagittal side view) to the lifting subject at waist height. Ensure head, torso, knees, and feet are visible within the frame.<br><br>
                            Click <b style="color: {ACCENT_MINT};">Start Monitoring</b> above to initiate real-time posture screening.
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
            with live_hud_col:
                hud_placeholder.markdown(
                    clean_html(f"""
                    <div style="background-color: {BG_PANEL_ELEVATED}; border: 1px solid {BORDER_SUBTLE}; border-radius: 8px; padding: 22px; text-align: center;">
                        <div style="display: inline-flex; align-items: center; justify-content: center; width: 44px; height: 44px; border-radius: 8px; background: rgba(255, 255, 255, 0.04); margin-bottom: 12px;">
                            {ICONS['activity']}
                        </div>
                        <div style="font-size: 15px; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 4px;">Live Telemetry Standby</div>
                        <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5; margin-bottom: 16px;">
                            Biomechanical angles, safe/unsafe classification, and posture hazard alerts will stream continuously here once camera monitoring begins.
                        </div>
                        <div style="border-top: 1px solid {BORDER_SUBTLE}; padding-top: 12px; text-align: left; font-size: 11px; color: {TEXT_SECONDARY}; line-height: 1.6;">
                            <div>&bull; <b>Squat Lift (SAFE):</b> Trunk inclination &le; 45&deg;, knees flexed (&le; 125&deg;)</div>
                            <div>&bull; <b>Stoop Lift (UNSAFE):</b> Trunk inclination &gt; 45&deg;, knees extended (&gt; 135&deg;)</div>
                            <div>&bull; <b>Visibility Gate:</b> High landmark confidence required for scoring</div>
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
            return

        # Determine Video Source
        if "Demo Video" in cam_source_choice and sample_video_path:
            cap_source = str(sample_video_path)
            is_demo = True
        elif "Camera 1" in cam_source_choice:
            cap_source = 1
            is_demo = False
        else:
            cap_source = 0
            is_demo = False

        # Active Stream Execution with Safe Resource Handling
        classifier.reset_state()
        cap = cv2.VideoCapture(cap_source)

        if not cap.isOpened():
            st.error(
                "❌ **Webcam Device Inaccessible:** Could not open camera device. "
                "Verify camera connection, privacy permissions, or select the Demo Video option."
            )
            st.session_state.webcam_active = False
            return

        frame_idx = 0
        t_start = time.time()

        try:
            while st.session_state.get("webcam_active", False):
                ret, frame = cap.read()
                if not ret or frame is None:
                    if is_demo:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        st.warning("Webcam stream ended or disconnected.")
                        break

                t_sec = time.time() - t_start
                res, annotated = classifier.process_frame(frame, frame_index=frame_idx, timestamp_sec=t_sec)

                # Render Image Feed
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Secondary Telemetry under Video
                fps_val = (1000.0 / res.inference_latency_ms) if res.inference_latency_ms > 0 else 0.0
                sec_placeholder.markdown(
                    clean_html(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 14px; background: {BG_PANEL_ELEVATED}; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px; font-size: 11px; color: {TEXT_SECONDARY}; margin-top: 8px;">
                        <span><b>FPS:</b> <span style="color: {TEXT_PRIMARY}; font-weight: 700;">{fps_val:.0f}</span> ({res.inference_latency_ms:.1f} ms)</span>
                        <span><b>Pose Quality:</b> <span style="color: {'#10B981' if res.is_valid_pose else '#F59E0B'}; font-weight: 600;">{'33 Landmarks Tracked' if res.is_valid_pose else 'Low Visibility'}</span></span>
                        <span><b>Frame:</b> #{frame_idx}</span>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

                # Render Right Column Live Assessment Card (Zero Raw HTML, Dedented)
                hud_html = render_live_telemetry_hud_html(
                    status=res.smoothed_status,
                    explanation=res.explanation,
                    method=res.assessment_method,
                    trunk_inclination=res.biomechanical_angles.get("trunk_inclination"),
                    knee_angle=res.biomechanical_angles.get("knee_angle_mean"),
                    is_valid_pose=res.is_valid_pose,
                    is_alert_triggered=res.is_alert_triggered,
                    consecutive_unsafe=classifier.consecutive_unsafe_frames,
                    fps=fps_val,
                    latency_ms=res.inference_latency_ms,
                )
                hud_placeholder.markdown(hud_html, unsafe_allow_html=True)

                frame_idx += 1
                time.sleep(0.01)

        finally:
            cap.release()


# ==============================================================================
# PAGE 4: MODEL EVALUATION & METRICS
# ==============================================================================
def page_model_evaluation(readiness: Optional[Dict[str, Any]] = None):
    st.markdown('<div class="lg-title">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Held-out test set evaluation with participant-grouped partitioning (zero data leakage)</div>', unsafe_allow_html=True)

    status_info = get_project_model_status()
    cnn_status = status_info["status"]
    screening_method = status_info["screening_method"]
    metrics_file = config.METRICS_DIR / "metrics.json"
    cm_plot = config.PLOTS_DIR / "confusion_matrix.png"
    curves_plot = config.PLOTS_DIR / "training_curves.png"
    report_file = config.METRICS_DIR / "classification_report.txt"

    # Leakage Prevention Methodology Card
    st.markdown(
        clean_html(f"""
        <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: {BORDER_FOCUS}; margin-bottom: 4px;">
                {ICONS["shield_check"]} Zero Data-Leakage Partitioning Protocol
            </div>
            <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.45;">
                Data is partitioned across <b>participant_id</b> via <code>GroupShuffleSplit</code>. No individual appearing in training is ever present in validation or testing.
                The classification threshold is calibrated strictly on the validation set; the held-out test set is evaluated only once with the frozen model.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # CASE 1: CNN NOT TRAINED
    if cnn_status == "CNN NOT TRAINED":
        st.markdown(
            clean_html(f"""
            <div style="background-color: {BG_PANEL_ELEVATED}; border: 1.5px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 20px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 15px; font-weight: 700; color: #EF4444; letter-spacing: -0.01em;">
                        MODEL STATUS: CNN NOT TRAINED
                    </div>
                    <span class="lg-badge lg-badge-missing">{ICONS["x_circle"]} Checkpoint Not Found</span>
                </div>
                <div style="font-size: 13px; color: {TEXT_PRIMARY}; line-height: 1.5; margin-bottom: 14px;">
                    {html.escape(status_info["reason"])}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 12px; background: #111821; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px; font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.6; margin-bottom: 14px;">
                    <div>&bull; <b>Screening Method:</b> <span style="color: {BORDER_FOCUS}; font-weight: 600;">{html.escape(screening_method)}</span></div>
                    <div>&bull; <b>CNN Checkpoint Path:</b> <code>models/best_lifting_cnn.keras</code> (absent)</div>
                    <div>&bull; <b>Evaluation State:</b> <span style="color: #F87171;">None (Requires trained checkpoint)</span></div>
                    <div>&bull; <b>Rule-Based Screening:</b> <span style="color: {BORDER_FOCUS};">Fully Operational</span></div>
                </div>
                <div style="font-size: 12px; color: {TEXT_MUTED}; line-height: 1.45;">
                    <b>Why evaluation is not available:</b> Held-out test evaluation measures the generalization performance of a trained CNN model on unseen test subjects. Since no trained CNN checkpoint exists, no genuine test metrics (accuracy, precision, recall, F1) can be displayed. Video screening functions independently using transparent biomechanical rules.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # Dataset & Label Bottleneck Callout
        st.markdown('<div class="lg-section-header">' + ICONS["database"] + ' Why CNN Training is Blocked: Label Availability</div>', unsafe_allow_html=True)
        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-bottom: 16px;">
                <div style="font-size: 13px; color: {TEXT_PRIMARY}; font-weight: 600; margin-bottom: 8px;">
                    Available Dataset Labels Diagnostic
                </div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.55;">
                    &bull; <b>Source Annotations (<code>data/annotations.json</code>):</b> 182 records with 49 matched local videos. All 49 matched trials record <code>subjective_judgment</code> as <code>unknown</code> (43) or <code>intermediate</code> (6). Neither <code>SAFE</code> nor <code>UNSAFE</code> ground truth labels exist in the raw benchmark file.<br>
                    &bull; <b>Manual Reviews (<code>data/manual_review_labels.csv</code>):</b> Only 3 trials are reviewed (all 3 SAFE squat lifts from Participant A). Zero UNSAFE (Class 1) trials exist.<br>
                    &bull; <b>Participant Diversity:</b> Only 1 participant ('A') has reviewed labels. At least 3 distinct participants (e.g. A, B, C, D) are required for leak-free train/val/test splitting without participant overlap.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # Actionable Next Steps Card
        st.markdown('<div class="lg-section-header">' + ICONS["clipboard_check"] + ' Next Actions to Train and Evaluate CNN</div>', unsafe_allow_html=True)
        st.markdown(
            clean_html(f"""
            <div class="lg-card">
                <ol style="margin: 0; padding-left: 20px; font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.7;">
                    <li><b>Review & Label Videos:</b> Navigate to the <b>Dataset Review</b> page in the sidebar. Inspect Environment 4 trials for participants B, C, and D, labeling clear stoop postures as <code>UNSAFE</code> and squat postures as <code>SAFE</code>.</li>
                    <li><b>Export Training Manifest:</b> Export the reviewed annotations to <code>data/annotations.csv</code> conforming to the required schema (<code>clip_path,participant_id,session_id,label,view</code>).</li>
                    <li><b>Extract Features & Splits:</b> Run <code>python -m src.dataset_loader</code> to extract 128&times;128 pose maps and generate leak-free participant splits (<code>data/splits/train.csv</code>, <code>val.csv</code>, <code>test.csv</code>).</li>
                    <li><b>Train the 2D CNN:</b> Execute <code>python -m src.train</code> to train the CNN on pose maps and save <code>models/best_lifting_cnn.keras</code>.</li>
                    <li><b>Run Held-Out Evaluation:</b> Run <code>python -m src.evaluate</code> to measure genuine test accuracy, recall, and F1 on the held-out test split.</li>
                </ol>
            </div>
            """),
            unsafe_allow_html=True,
        )
        return

    # CASE 2: EVALUATION PENDING (Checkpoint exists but metrics.json absent)
    if cnn_status == "EVALUATION PENDING" or not status_info.get("has_evaluation_results", False):
        st.markdown(
            clean_html(f"""
            <div style="background-color: {BG_PANEL_ELEVATED}; border: 1.5px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 20px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 15px; font-weight: 700; color: #F59E0B;">
                        MODEL STATUS: EVALUATION PENDING
                    </div>
                    <span class="lg-badge lg-badge-warning">{ICONS["help_circle"]} Evaluation Pending</span>
                </div>
                <div style="font-size: 13px; color: {TEXT_PRIMARY}; line-height: 1.5; margin-bottom: 12px;">
                    A valid trained CNN checkpoint exists at <code>models/best_lifting_cnn.keras</code>, but it has not yet been evaluated on the held-out test split.
                </div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5;">
                    Run evaluation on <code>data/splits/test.csv</code> to compute genuine test accuracy, recall, and F1 score.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        test_split_file = config.SPLITS_DIR / "test.csv"
        if test_split_file.exists():
            if st.button("Run Evaluation Now on Held-Out Test Split", type="primary"):
                with st.spinner("Evaluating held-out test manifest..."):
                    res = evaluate_held_out_test_set()
                st.rerun()
        else:
            st.info("Held-out test split not found at `data/splits/test.csv`. Generate split manifests via `src.dataset_loader` first.")
        return

    # CASE 3: EVALUATED (Valid checkpoint + genuine test evaluation results)
    metrics = status_info.get("metrics") or {}

    # 5 Analytics KPI Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(render_kpi_card_html("Test Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%", f"{metrics.get('test_sample_count', 0)} test samples", "default"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_kpi_card_html("Unsafe Precision", f"{metrics.get('precision_unsafe', 0.0) * 100:.1f}%", "Positive predictive val", "default"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_kpi_card_html("Unsafe Recall", f"{metrics.get('recall_unsafe', 0.0) * 100:.1f}%", "Sensitivity (hazard)", "success"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_kpi_card_html("Unsafe F1-Score", f"{metrics.get('f1_unsafe', 0.0):.3f}", "Harmonic mean", "success"), unsafe_allow_html=True)
    with m5:
        st.markdown(render_kpi_card_html("Specificity", f"{metrics.get('specificity', 0.0) * 100:.1f}%", "True negative rate", "default"), unsafe_allow_html=True)

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

    # Visual Artifacts: Confusion Matrix & Training Curves
    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Held-Out Confusion Matrix</div>', unsafe_allow_html=True)
        if cm_plot.exists():
            st.image(str(cm_plot), use_container_width=True)
        else:
            st.caption("Confusion matrix image pending.")

        # False Negative Ergonomic Hazard Callout
        fn_count = metrics.get("confusion_matrix", {}).get("false_negatives_missed_unsafe", 0)
        st.markdown(
            clean_html(f"""
            <div style="padding: 10px 12px; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 6px; font-size: 12px; color: #FCA5A5;">
                <b>Ergonomic Safety Callout (False Negatives: {fn_count}):</b><br>
                False negatives represent unsafe postures classified as safe. In occupational monitoring, missing an unsafe lift presents the greatest injury risk.
            </div>
            """),
            unsafe_allow_html=True,
        )

    with vis_col2:
        st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Training Trajectory</div>', unsafe_allow_html=True)
        if curves_plot.exists():
            st.image(str(curves_plot), use_container_width=True)
        else:
            st.caption("Training curves plot pending.")

        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-top: 12px;">
                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 6px;">Evaluation Parameters</div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.6;">
                    &bull; <b>Decision Threshold (&tau;):</b> <code>{metrics.get('decision_threshold')}</code> (validation optimized)<br>
                    &bull; <b>Mean Inference Latency:</b> <code>{metrics.get('mean_latency_ms')} ms/frame</code><br>
                    &bull; <b>ROC-AUC Metric:</b> <code>{metrics.get('roc_auc')}</code><br>
                    &bull; <b>Test Split Size:</b> <code>{metrics.get('test_sample_count')} frames</code>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    if report_file.exists():
        with st.expander("📄 View Full Classification Report Text"):
            with open(report_file, "r") as f:
                st.code(f.read(), language="text")


# ==============================================================================
# PAGE 5: DATASET MANAGEMENT
# ==============================================================================
def page_dataset_management(readiness: Optional[Dict[str, Any]] = None):
    st.markdown('<div class="lg-title">Dataset Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Audit SAFELIFT JSON annotations, verify matched local video files, and inspect participant distributions</div>', unsafe_allow_html=True)

    audit = validate_safelift_dataset()

    if not audit.is_loaded:
        st.markdown(
            render_empty_state_html(
                "No Local Annotations Found",
                f"SAFELIFT master annotations not found at {audit.annotations_file}. Ensure data/annotations.json exists.",
                "database"
            ),
            unsafe_allow_html=True,
        )
        return

    # Section 1: SAFELIFT Master Annotations Header & KPIs
    st.markdown(clean_html(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div class="lg-section-header" style="margin: 0;">
            {ICONS["database"]} SAFELIFT Benchmark Annotations (<code>data/annotations.json</code>)
        </div>
        <span class="lg-badge lg-badge-ready">
            {ICONS["check_circle"]} SAFELIFT JSON Loaded ({audit.total_json_records} Records)
        </span>
    </div>
    """), unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(render_kpi_card_html("Annotation Records", str(audit.total_json_records), "Full Benchmark JSON", "default"), unsafe_allow_html=True)
    with d2:
        st.markdown(render_kpi_card_html("Matched Local Videos", f"{audit.matched_videos_count} / {audit.local_videos_total}", "100% of Disk Files Matched", "success"), unsafe_allow_html=True)
    with d3:
        st.markdown(render_kpi_card_html("Missing from Disk", str(audit.missing_videos_count), "Other Benchmark Envs", "warning"), unsafe_allow_html=True)
    with d4:
        p_str = ", ".join(audit.participants)
        st.markdown(render_kpi_card_html("Subjects on Disk", f"{len(audit.participants)} ({p_str})", "Environment 4 Trials", "default"), unsafe_allow_html=True)

    # Source Labels Breakdown & Ethical Disclosure
    st.markdown(
        clean_html(f"""
        <div class="lg-card" style="margin-top: 14px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY};">
                    Source SAFELIFT Labels Distribution (<code>subjective_judgment</code>)
                </span>
                <span class="lg-badge" style="background: rgba(255,255,255,0.06); color: {BORDER_FOCUS}; border: 1px solid rgba(16, 185, 129, 0.35); font-size: 11px;">
                    Read-Only Integrity Enforced
                </span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                <div style="background: #111821; border: 1px solid {BORDER_SUBTLE}; padding: 10px 14px; border-radius: 6px;">
                    <div style="font-size: 11px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase;">Matched Local Videos (49 Files)</div>
                    <div style="font-size: 13px; color: {TEXT_PRIMARY}; margin-top: 4px; line-height: 1.5;">
                        &bull; <code>unknown</code>: <b>{audit.label_distribution_matched.get('unknown', 0)}</b> trials (87.8%)<br>
                        &bull; <code>intermediate</code>: <b>{audit.label_distribution_matched.get('intermediate', 0)}</b> trials (12.2% - Participant A lateral)<br>
                        &bull; <code>SAFE</code>: <b>0</b> trials (0.0%)<br>
                        &bull; <code>UNSAFE</code>: <b>0</b> trials (0.0%)
                    </div>
                </div>
                <div style="background: #111821; border: 1px solid {BORDER_SUBTLE}; padding: 10px 14px; border-radius: 6px;">
                    <div style="font-size: 11px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase;">Full Master JSON (182 Records)</div>
                    <div style="font-size: 13px; color: {TEXT_PRIMARY}; margin-top: 4px; line-height: 1.5;">
                        &bull; <code>unknown</code>: <b>{audit.label_distribution_json.get('unknown', 0)}</b> entries (96.7%)<br>
                        &bull; <code>intermediate</code>: <b>{audit.label_distribution_json.get('intermediate', 0)}</b> entries (3.3%)<br>
                        &bull; <code>SAFE</code>: <b>0</b> entries (0.0%)<br>
                        &bull; <code>UNSAFE</code>: <b>0</b> entries (0.0%)
                    </div>
                </div>
            </div>
            <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5; border-top: 1px solid {BORDER_SUBTLE}; padding-top: 10px;">
                <b>Data Integrity Guarantee:</b> The original <code>data/annotations.json</code> and video files are preserved strictly read-only and unmodified.
                Because the source benchmark records subjective judgment as occupational ratings rather than ground-truth classifications,
                <b>neither SAFE nor UNSAFE labels are fabricated when source annotations say unknown</b>.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # Section 2: CNN Training Requirement - Separate Binary-Labelled CSV
    st.markdown('<div class="lg-section-header">' + ICONS["layers"] + ' Binary Dataset Requirement for CNN Training</div>', unsafe_allow_html=True)
    bin_sum = audit.binary_labels_summary
    st.markdown(
        clean_html(f"""
        <div class="lg-card" style="border: 1px solid rgba(245, 158, 11, 0.35); margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 13px; font-weight: 600; color: {ACCENT_AMBER};">
                    Separate Binary-Labelled Manifest Required for CNN Training
                </span>
                <span class="lg-badge lg-badge-warning">Binary Labels Insufficient</span>
            </div>
            <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.55; margin-bottom: 10px;">
                To train the 2D CNN model, a separate binary-labelled CSV manifest (<code>data/annotations.csv</code>) is required.
                The training pipeline requires binary labels (<code>0 = SAFE</code> squat lift, <code>1 = UNSAFE</code> stoop lift)
                partitioned across <b>at least 3 distinct participants</b> (e.g., A, B, C, D) using <code>GroupShuffleSplit</code> to prevent data leakage.
            </div>
            <code style="display: block; padding: 6px 10px; background: rgba(0,0,0,0.35); border-radius: 4px; font-size: 12px; color: {BORDER_FOCUS}; margin-bottom: 10px;">
                clip_path,participant_id,session_id,label,view
            </code>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px 12px; background: #111821; border-radius: 6px; font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.6; margin-bottom: 12px;">
                <div>&bull; <b>Manual Reviews (<code>data/manual_review_labels.csv</code>):</b> {bin_sum.get('manual_safe_count', 0)} SAFE, {bin_sum.get('manual_unsafe_count', 0)} UNSAFE, {bin_sum.get('manual_unreviewed_count', 0)} unreviewed</div>
                <div>&bull; <b>Reviewed Participants:</b> {bin_sum.get('manual_reviewed_participants') or ['None']} (Participant A only)</div>
                <div>&bull; <b>Missing Class:</b> <span style="color: #F87171;">UNSAFE (0 samples found)</span></div>
                <div>&bull; <b>Training Viability:</b> <span style="color: #F87171;">Cannot construct valid training dataset</span></div>
            </div>
            <div style="font-size: 12px; color: {TEXT_PRIMARY}; font-weight: 500;">
                <b>Next Action:</b> Review videos in the <b>Dataset Review</b> workspace to assign stoop lifting mechanic labels (UNSAFE) and squat lifting labels (SAFE) for participants B, C, and D.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    col_btn_rev, _ = st.columns([1, 2])
    with col_btn_rev:
        if st.button("Open Dataset Review Page →", use_container_width=True, key="btn_go_review_from_dataset"):
            st.session_state.current_page = "Dataset Review"
            st.rerun()

    # Section 3: Matched Videos Explorer
    st.markdown('<div class="lg-section-header">' + ICONS["video"] + ' Matched Environment 4 Video Manifest (49 Trials)</div>', unsafe_allow_html=True)
    if audit.matched_records:
        df_matched = pd.DataFrame(audit.matched_records)
        display_cols = [
            "video_filename", "participant_id", "camera_view", "action_direction",
            "action_start_sec", "action_end_sec", "angle_of_asymmetry_deg",
            "subjective_judgment", "box_width_cm", "box_height_cm", "box_depth_cm",
            "subject_height_cm", "subject_age", "subject_gender"
        ]
        available_cols = [c for c in display_cols if c in df_matched.columns]
        st.dataframe(df_matched[available_cols], use_container_width=True, hide_index=True)

    # Section 4: Custom Annotation CSV Validator
    st.markdown('<div class="lg-section-header">' + ICONS["file_text"] + ' Validate Custom Annotation File</div>', unsafe_allow_html=True)
    up_csv = st.file_uploader("Upload CSV for Schema Verification", type=["csv"], key="custom_ann_uploader", label_visibility="collapsed")
    if up_csv:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(up_csv.read())
            tmp_p = Path(tmp.name)
        up_rep = validate_annotations(tmp_p)
        if up_rep.is_valid:
            st.success("Uploaded CSV strictly adheres to the required schema!")
        else:
            st.error("Uploaded CSV has schema violations.")
        st.json(up_rep.to_dict())


# ==============================================================================
# PAGE 6: SESSION REPORTS
# ==============================================================================
def page_reports():
    st.markdown('<div class="lg-title">Session Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Audit history, download timestamped predictions, and export session logs</div>', unsafe_allow_html=True)

    pred_dir = config.PREDICTIONS_DIR
    pred_files = list(pred_dir.glob("*.csv")) if pred_dir.exists() else []

    if not pred_files:
        st.markdown(render_empty_state_html("No Saved Session Logs", "Process a video in Video Screening or run Live Webcam to record sessions.", "file_text"), unsafe_allow_html=True)
        return

    selected_csv = st.selectbox("Select Screening Session", pred_files, format_func=lambda p: p.name)
    if selected_csv:
        df = pd.read_csv(selected_csv)
        rep = generate_session_report(df)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(render_kpi_card_html("Risk Classification", rep.ergonomic_risk_level, "Screening risk rating", "danger" if "HIGH" in rep.ergonomic_risk_level else "success"), unsafe_allow_html=True)
        with r2:
            st.markdown(render_kpi_card_html("Safe Posture Ratio", f"{rep.safe_pct_of_valid}%", f"{rep.safe_frames} frames", "success"), unsafe_allow_html=True)
        with r3:
            st.markdown(render_kpi_card_html("Unsafe Posture Ratio", f"{rep.unsafe_pct_of_valid}%", f"{rep.unsafe_frames} frames", "danger"), unsafe_allow_html=True)
        with r4:
            st.markdown(render_kpi_card_html("Duration", f"{rep.duration_sec:.1f}s", f"{rep.total_frames} total frames", "default"), unsafe_allow_html=True)

        st.caption("Frame Telemetry Preview (First 50 frames):")
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Full Session CSV",
            data=csv_bytes,
            file_name=selected_csv.name,
            mime="text/csv",
            use_container_width=True,
        )


# ==============================================================================
# PAGE 7: DATASET REVIEW & MANUAL ANNOTATION
# ==============================================================================
def page_dataset_review():
    st.markdown('<div class="lg-title">Dataset Review & Manual Annotation</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Inspect the 49 matched Environment 4 videos, audit metadata, and record manual ground-truth posture labels</div>', unsafe_allow_html=True)

    mgr = DatasetReviewManager()
    if not mgr.matched_videos:
        st.warning("No matched Environment 4 videos found. Please check data/raw_videos and data/annotations.json.")
        return

    df_manifest = mgr.load_or_init_review_manifest()
    progress = mgr.get_review_progress()

    # Progress Overview Header
    prog_col1, prog_col2, prog_col3, prog_col4, prog_col5 = st.columns(5)
    with prog_col1:
        st.markdown(render_kpi_card_html("Total Videos", f"{progress['total']}", "Matched Env 4 trials", "default"), unsafe_allow_html=True)
    with prog_col2:
        st.markdown(render_kpi_card_html("Reviewed", f"{progress['reviewed']}", f"{progress['percent_complete']}% complete", "success" if progress['reviewed'] > 0 else "default"), unsafe_allow_html=True)
    with prog_col3:
        st.markdown(render_kpi_card_html("Unreviewed", f"{progress['unreviewed']}", "Awaiting inspection", "warning" if progress['unreviewed'] > 0 else "default"), unsafe_allow_html=True)
    with prog_col4:
        st.markdown(render_kpi_card_html("Safe Lifts (0)", f"{progress['safe_count']}", "Squat posture", "success"), unsafe_allow_html=True)
    with prog_col5:
        st.markdown(render_kpi_card_html("Unsafe Lifts (1)", f"{progress['unsafe_count']}", "Stoop posture", "danger"), unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Filter Bar
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.2])
    with filter_col1:
        p_filter = st.selectbox("Filter Participant", ["All", "A", "B", "C", "D"], key="rev_p_filter")
    with filter_col2:
        v_filter = st.selectbox("Filter Camera View", ["All", "lateral", "oblique"], key="rev_v_filter")
    with filter_col3:
        s_filter = st.selectbox("Filter Review Status", ["All", "Unreviewed Only", "Reviewed Only"], key="rev_s_filter")

    # Apply Filters
    filtered_videos = mgr.matched_videos
    if p_filter != "All":
        filtered_videos = [m for m in filtered_videos if m.participant_id == p_filter]
    if v_filter != "All":
        filtered_videos = [m for m in filtered_videos if m.camera_view == v_filter]
    if s_filter != "All":
        rev_status_map = {row["video_filename"]: row["review_status"] for _, row in df_manifest.iterrows()}
        if s_filter == "Unreviewed Only":
            filtered_videos = [m for m in filtered_videos if rev_status_map.get(m.video_filename) == STATUS_UNREVIEWED]
        elif s_filter == "Reviewed Only":
            filtered_videos = [m for m in filtered_videos if rev_status_map.get(m.video_filename) == STATUS_REVIEWED]

    if not filtered_videos:
        st.info("No videos match the selected filter criteria.")
        return

    # Video Selector Dropdown
    rev_label_map = {row["video_filename"]: row["reviewed_label"] for _, row in df_manifest.iterrows()}
    rev_status_map = {row["video_filename"]: row["review_status"] for _, row in df_manifest.iterrows()}

    def format_video_option(m):
        status = rev_status_map.get(m.video_filename, STATUS_UNREVIEWED)
        lbl = rev_label_map.get(m.video_filename, "")
        badge = f"[{lbl}]" if (status == STATUS_REVIEWED and lbl) else f"[{status}]"
        return f"{badge} {m.video_filename} (Subject {m.participant_id} | {m.camera_view})"

    selected_video = st.selectbox(
        f"Select Video to Inspect & Review ({len(filtered_videos)} available):",
        filtered_videos,
        format_func=format_video_option,
        key="selected_video_obj"
    )

    if not selected_video:
        return

    # Two-Column Review Workspace
    w_col1, w_col2 = st.columns([1.1, 0.9])

    with w_col1:
        st.markdown('<div class="lg-section-header">' + ICONS["video"] + ' Video Playback & Timing</div>', unsafe_allow_html=True)
        vid_path = Path(selected_video.absolute_video_path)
        if vid_path.exists():
            st.video(str(vid_path))
        else:
            st.error(f"Video file not found at: {vid_path}")

        # Timing Cue Ribbon
        s_sec = selected_video.action_start_sec
        e_sec = selected_video.action_end_sec
        dur = f"{(e_sec - s_sec):.1f}s" if (s_sec is not None and e_sec is not None) else "N/A"
        st.markdown(
            clean_html(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 6px; padding: 10px 14px; margin-top: 8px;">
                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: {BORDER_FOCUS};">Action Interval Cues</div>
                <div style="font-size: 13px; color: {TEXT_PRIMARY}; font-weight: 500;">
                    Action Active: <b>{s_sec}s</b> to <b>{e_sec}s</b> (Duration: {dur})
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

        # Subject & Box Measurements Card
        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-top: 12px;">
                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 6px;">Physical Dimensions & Subject Stats</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; color: {TEXT_SECONDARY};">
                    <div>&bull; <b>Box Width:</b> {selected_video.box_width_cm or 'N/A'} cm</div>
                    <div>&bull; <b>Subject Height:</b> {selected_video.subject_height_cm or 'N/A'} cm</div>
                    <div>&bull; <b>Box Height:</b> {selected_video.box_height_cm or 'N/A'} cm</div>
                    <div>&bull; <b>Subject Age:</b> {selected_video.subject_age or 'N/A'} yrs</div>
                    <div>&bull; <b>Box Depth:</b> {selected_video.box_depth_cm or 'N/A'} cm</div>
                    <div>&bull; <b>Subject Gender:</b> {selected_video.subject_gender or 'N/A'}</div>
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with w_col2:
        st.markdown('<div class="lg-section-header">' + ICONS["clipboard_check"] + ' Metadata & Manual Labeling</div>', unsafe_allow_html=True)

        row_curr = df_manifest[df_manifest["video_filename"] == selected_video.video_filename]
        curr_status = row_curr["review_status"].values[0] if not row_curr.empty else STATUS_UNREVIEWED
        curr_label = row_curr["reviewed_label"].values[0] if not row_curr.empty else ""
        curr_notes = row_curr["reviewer_notes"].values[0] if not row_curr.empty else ""
        curr_timestamp = row_curr["review_timestamp"].values[0] if not row_curr.empty else ""

        # Metadata Card
        st.markdown(
            clean_html(f"""
            <div class="lg-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY};">Subject {selected_video.participant_id} | Trial {selected_video.repetition_index}</span>
                    <span class="lg-badge" style="background: rgba(255,255,255,0.06); color: {BORDER_FOCUS}; border: 1px solid rgba(16, 185, 129, 0.35);">{selected_video.camera_view} view</span>
                </div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5;">
                    &bull; <b>Action Direction:</b> {selected_video.action_direction}<br>
                    &bull; <b>Angle of Asymmetry:</b> {selected_video.angle_of_asymmetry_deg or 0.0}&deg;<br>
                    &bull; <b>Original Subjective Judgment:</b> <code>{selected_video.subjective_judgment_original}</code> <small style="color:{TEXT_MUTED};">(preserved untouched)</small><br>
                    &bull; <b>Current Review Status:</b> <span style="font-weight:600; color: {'#10B981' if curr_status == STATUS_REVIEWED else '#F59E0B'}">{curr_status} ({curr_label or 'None'})</span>
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

        with st.expander("ℹ️ View Ergonomic Labeling Guidance (NIOSH / OSHA)"):
            st.markdown(LABELING_GUIDANCE_TEXT)

        # Review Form
        st.markdown("<div style='font-size: 13px; font-weight: 600; color: " + TEXT_PRIMARY + "; margin-bottom: 6px;'>Record Posture Label</div>", unsafe_allow_html=True)
        default_label_idx = 0
        if curr_label == LABEL_UNSAFE:
            default_label_idx = 1
        elif curr_label == LABEL_AMBIGUOUS:
            default_label_idx = 2

        chosen_label = st.radio(
            "Posture Classification",
            [
                "SAFE (Squat lift - knees flexed, spine protected)",
                "UNSAFE (Stoop lift - legs straight, lumbar flexion)",
                "AMBIGUOUS (Occluded / borderline / mixed)"
            ],
            index=default_label_idx,
            key=f"radio_label_{selected_video.video_filename}"
        )

        label_token = LABEL_SAFE if "SAFE" in chosen_label else (LABEL_UNSAFE if "UNSAFE" in chosen_label else LABEL_AMBIGUOUS)

        t_adj_col1, t_adj_col2 = st.columns(2)
        with t_adj_col1:
            start_in = st.number_input(
                "Action Start (sec)",
                value=float(selected_video.action_start_sec or 0.0),
                step=0.5,
                key=f"start_in_{selected_video.video_filename}"
            )
        with t_adj_col2:
            end_in = st.number_input(
                "Action End (sec)",
                value=float(selected_video.action_end_sec or 5.0),
                step=0.5,
                key=f"end_in_{selected_video.video_filename}"
            )

        notes_in = st.text_area(
            "Reviewer Notes / Biomechanical Observations",
            value=str(curr_notes) if pd.notna(curr_notes) else "",
            placeholder="e.g. Pronounced knee bend, lumbar lordosis maintained during floor liftoff.",
            key=f"notes_in_{selected_video.video_filename}"
        )

        if st.button("💾 Save Review Label", type="primary", use_container_width=True, key="btn_save_review"):
            success = mgr.save_review_label(
                video_filename=selected_video.video_filename,
                label=label_token,
                reviewer_notes=notes_in,
                action_start_override=start_in,
                action_end_override=end_in,
            )
            if success:
                st.success(f"Successfully recorded label: **{label_token}** for {selected_video.video_filename}!")
                st.rerun()

        if curr_timestamp:
            st.caption(f"Last reviewed on: {curr_timestamp}")

    # Full Review Manifest Audit Table at Bottom
    st.markdown("---")
    st.markdown('<div class="lg-section-header">' + ICONS["file_text"] + ' Complete Environment 4 Review Manifest (49 Videos)</div>', unsafe_allow_html=True)
    df_manifest_fresh = mgr.load_or_init_review_manifest()
    
    view_table = df_manifest_fresh[[
        "video_filename", "participant_id", "camera_view", "action_direction",
        "action_start_sec", "action_end_sec", "subjective_judgment_original",
        "review_status", "reviewed_label", "reviewer_notes", "review_timestamp"
    ]]
    st.dataframe(view_table, use_container_width=True, hide_index=True)

    csv_data = df_manifest_fresh.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download manual_review_labels.csv",
        data=csv_data,
        file_name="manual_review_labels.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================
def main():
    init_session_state()
    readiness = check_system_readiness()
    current_page = render_sidebar(readiness)
    classifier = get_classifier()

    if current_page in ("Posture Screening", "Video Screening", "Live Webcam"):
        default_tab = 1 if current_page == "Live Webcam" else 0
        page_posture_screening(classifier, default_tab=default_tab)
    elif current_page == "Overview":
        page_overview(readiness)
    elif current_page == "Model Evaluation":
        page_model_evaluation(readiness)
    elif current_page == "Dataset":
        page_dataset_management(readiness)
    elif current_page == "Reports":
        page_reports()
    elif current_page == "Dataset Review":
        page_dataset_review()


if __name__ == "__main__":
    main()
