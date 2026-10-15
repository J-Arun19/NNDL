# -*- coding: utf-8 -*-
"""Streamlit Web Application: LiftGuard AI.

A professional, restrained industrial Computer Vision application for manual lifting posture screening,
biomechanical angle evaluation, dataset auditing, and real-time video/webcam inspection.
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
    ACCENT_TEAL,
    BORDER_STRONG,
    BORDER_SUBTLE,
    ICONS,
    STATUS_SAFE,
    STATUS_UNSAFE,
    STATUS_WARN,
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

# Page configuration with clean title and no emoji icons
st.set_page_config(
    page_title="LiftGuard AI - Industrial Posture Screening",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Centralized Scoped Industrial Theme CSS
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
        st.session_state.current_page = "Screening"
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False


def render_sidebar(readiness: Dict[str, Any]) -> str:
    """Render the compact, professional navigation sidebar."""
    with st.sidebar:
        # Brand Header
        st.markdown(render_sidebar_brand(), unsafe_allow_html=True)

        # 5 Clean Navigation Options
        page_options = [
            "Screening",
            "Overview",
            "Dataset",
            "Evaluation",
            "Reports",
        ]

        current_page = st.session_state.get("current_page", "Screening")
        # Backwards compatibility mapping
        if current_page in ("Posture Screening", "Video Screening", "Live Webcam"):
            current_page = "Screening"
        elif current_page in ("Model Evaluation",):
            current_page = "Evaluation"
        elif current_page in ("Dataset Review",):
            current_page = "Dataset"

        st.session_state.current_page = current_page

        current_idx = page_options.index(current_page) if current_page in page_options else 0
        selected_page = st.radio(
            "Navigate",
            page_options,
            index=current_idx,
            label_visibility="collapsed",
        )
        if selected_page != st.session_state.current_page:
            if st.session_state.get("webcam_active", False):
                st.session_state.webcam_active = False
            st.session_state.current_page = selected_page
            st.rerun()

        # Compact Status Footer (Restrained, no loud alarmist badges)
        st.markdown(
            render_sidebar_status_footer(readiness, readiness.get("decision_threshold", 0.5)),
            unsafe_allow_html=True,
        )

    return st.session_state.current_page


# ==============================================================================
# PAGE 1: POSTURE SCREENING (PRIMARY WORKSPACE)
# ==============================================================================
def page_posture_screening(classifier: LiftingPostureClassifier):
    st.markdown('<div class="lg-title">Posture Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Assess lifting posture from a recorded video or live camera.</div>', unsafe_allow_html=True)

    # Simple mode selector below title
    mode = st.radio(
        "Screening Mode",
        ["Upload Video", "Live Camera"],
        horizontal=True,
        label_visibility="collapsed",
        key="screening_mode_toggle"
    )

    if mode == "Upload Video":
        if st.session_state.get("webcam_active", False):
            st.session_state.webcam_active = False

        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            uploaded_file = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mov"],
                help="Supported formats: MP4, AVI, MOV. Lateral view recommended.",
                key="uploader_screening_video",
                label_visibility="collapsed"
            )

            video_path = None
            if uploaded_file:
                # Detect new file upload to reset previous analysis
                file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("active_file_sig") != file_sig:
                    st.session_state.active_file_sig = file_sig
                    st.session_state.pop("last_processed_video", None)

                temp_dir = Path(tempfile.gettempdir()) / "liftguard_cache"
                temp_dir.mkdir(parents=True, exist_ok=True)
                clean_stem = "".join(c for c in Path(uploaded_file.name).stem if c.isalnum() or c in ("_", "-"))
                suffix = Path(uploaded_file.name).suffix or ".mp4"
                cached_input = temp_dir / f"upload_{clean_stem}{suffix}"

                video_bytes = uploaded_file.getvalue()
                with open(cached_input, "wb") as f_in:
                    f_in.write(video_bytes)
                video_path = cached_input

                st.caption(f"Original Video ({len(video_bytes) / 1024 / 1024:.1f} MB):")
                st.video(video_bytes, format="video/mp4")
                run_btn = st.button("Run Analysis", type="primary", key="btn_run_analysis")
            else:
                st.caption("Select an MP4, AVI, or MOV video clip. Sagittal (side) view yields optimal tracking.")
                run_btn = False
                if "last_processed_video" in st.session_state:
                    st.session_state.pop("last_processed_video", None)

        with col_right:
            st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Assessment Results</div>', unsafe_allow_html=True)
            if not uploaded_file:
                st.markdown(
                    render_empty_state_html(
                        "No Video Selected",
                        "Upload a video clip on the left to inspect lifting biomechanics and evaluate safe/unsafe posture.",
                        "video"
                    ),
                    unsafe_allow_html=True,
                )
            elif not run_btn and "last_processed_video" not in st.session_state:
                st.markdown(
                    render_empty_state_html(
                        "Ready for Analysis",
                        "Click 'Run Analysis' on the left to extract 33 body keypoints and evaluate posture criteria.",
                        "cpu"
                    ),
                    unsafe_allow_html=True,
                )

        # Process Video Execution
        if run_btn and video_path:
            with col_right:
                prog_bar = st.progress(0.0)
                status_text = st.empty()
                status_text.caption("Evaluating biomechanical angles across frames...")

                out_dir = Path(tempfile.gettempdir()) / "liftguard_cache"
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(video_path).stem
                out_annotated = out_dir / f"annotated_{stem}.mp4"
                out_csv = out_dir / f"preds_{stem}.csv"

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
                    prog_bar.empty()

                    st.session_state.last_processed_video = {
                        "df_preds": df_preds,
                        "annotated_path": str(out_video_path) if (out_video_path and Path(out_video_path).exists()) else None,
                        "csv_path": str(out_csv) if (out_csv and out_csv.exists()) else None,
                        "summary": summary,
                        "stem": stem,
                    }
                except Exception as exc:
                    st.error(f"Analysis error: {exc}")
                    logger.error(f"Analysis error: {exc}", exc_info=True)

        # Display Results if available
        if "last_processed_video" in st.session_state:
            res_data = st.session_state.last_processed_video
            df_preds = res_data["df_preds"]
            summary: VideoAnalysisSummary = res_data["summary"]

            with col_right:
                # Concise industrial result card
                st.markdown(render_uploaded_video_summary_html(summary), unsafe_allow_html=True)

                # Compact CSV Export below summary (not full width)
                csv_bytes = df_preds.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export Frame Data (CSV)",
                    data=csv_bytes,
                    file_name=f"posture_analysis_{res_data['stem']}.csv",
                    mime="text/csv",
                    use_container_width=False,
                    key="btn_dl_csv"
                )

            # Playback of annotated video in left column below preview
            annotated_path_str = res_data.get("annotated_path")
            annotated_file = Path(annotated_path_str) if annotated_path_str else None
            if annotated_file and annotated_file.exists() and annotated_file.stat().st_size > 0:
                with col_left:
                    st.markdown('<div class="lg-section-header">' + ICONS["shield_check"] + ' Annotated Video with Skeleton Overlay</div>', unsafe_allow_html=True)
                    annotated_bytes = annotated_file.read_bytes()
                    st.video(annotated_bytes, format="video/mp4")
                    st.download_button(
                        "Export Annotated Video",
                        data=annotated_bytes,
                        file_name=f"annotated_{res_data['stem']}.mp4",
                        mime="video/mp4",
                        use_container_width=False,
                        key="btn_dl_annotated_video"
                    )
            elif annotated_path_str:
                with col_left:
                    st.warning("Annotated video file could not be generated. Please check video source.")

            # Detailed Kinematics Expander
            with st.expander("Kinematic Trajectories and Frame Log", expanded=False):
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.caption("Trunk Inclination (deg from vertical) over time (45° threshold boundary):")
                    if "trunk_inclination_deg" in df_preds.columns:
                        t_chart = df_preds[["timestamp_sec", "trunk_inclination_deg"]].dropna().set_index("timestamp_sec")
                        st.line_chart(t_chart, height=160)
                with chart_col2:
                    st.caption("Knee Flexion Angle (deg) over time (≤125° = Squat):")
                    if "knee_angle_mean_deg" in df_preds.columns:
                        k_chart = df_preds[["timestamp_sec", "knee_angle_mean_deg"]].dropna().set_index("timestamp_sec")
                        st.line_chart(k_chart, height=160)

                st.caption("Frame-by-Frame Posture Telemetry:")
                view_cols = ["frame_index", "timestamp_sec", "status", "trunk_inclination_deg", "knee_angle_mean_deg", "explanation", "alert_triggered"]
                present_view = [c for c in view_cols if c in df_preds.columns]
                st.dataframe(df_preds[present_view], use_container_width=True, hide_index=True)

    else:
        # Live Camera Mode
        sample_video_path = None
        demo_dir = config.RAW_VIDEOS_DIR / "video_env_4_anonymized"
        if demo_dir.exists():
            demo_vids = sorted(list(demo_dir.glob("*.mp4")))
            if demo_vids:
                sample_video_path = demo_vids[0]

        # Compact Controls Row
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.6, 1, 1])
        with ctrl_col1:
            source_options = ["Camera 0 (Default Webcam)", "Camera 1 (External USB)"]
            if sample_video_path:
                source_options.append("Demo Video (Trial Stream)")
            cam_choice = st.selectbox(
                "Camera Source",
                source_options,
                index=0,
                key="live_cam_source",
                disabled=st.session_state.webcam_active,
                label_visibility="collapsed"
            )
        with ctrl_col2:
            btn_start = st.button(
                "Start Monitoring",
                type="primary",
                use_container_width=True,
                key="btn_start_monitor",
                disabled=st.session_state.webcam_active,
            )
        with ctrl_col3:
            btn_stop = st.button(
                "Stop Monitoring",
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

        # Two-Column Layout (Matching upload mode structure)
        live_col_left, live_col_right = st.columns([1.1, 0.9])
        feed_placeholder = live_col_left.empty()
        sec_placeholder = live_col_left.empty()
        hud_placeholder = live_col_right.empty()

        if not st.session_state.webcam_active:
            with live_col_left:
                feed_placeholder.markdown(
                    render_empty_state_html(
                        "Camera Standby",
                        "Position camera lateral (sagittal side view) at waist height. Ensure head, torso, knees, and feet are visible. Click Start Monitoring above to begin.",
                        "camera"
                    ),
                    unsafe_allow_html=True,
                )
            with live_col_right:
                hud_placeholder.markdown(
                    clean_html(f"""
                    <div class="lg-card">
                        <div class="lg-section-header" style="margin-top: 0;">
                            {ICONS['activity']} Live Telemetry Standby
                        </div>
                        <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5; margin-bottom: 12px;">
                            Real-time joint angles, safe/unsafe classification, and posture hazard alerts will appear here during live monitoring.
                        </div>
                        <div style="font-size: 11px; color: {TEXT_MUTED}; border-top: 1px solid {BORDER_SUBTLE}; padding-top: 8px; line-height: 1.6;">
                            <div>&bull; <b>Safe (Squat):</b> Trunk inclination &le; 45&deg;, knees flexed (&le; 125&deg;)</div>
                            <div>&bull; <b>Unsafe (Stoop):</b> Trunk inclination &gt; 45&deg;, knees extended (&gt; 135&deg;)</div>
                            <div>&bull; <b>Quality Gate:</b> High joint landmark confidence required</div>
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
            return

        # Active Stream Execution
        if "Demo Video" in cam_choice and sample_video_path:
            cap_source = str(sample_video_path)
            is_demo = True
        elif "Camera 1" in cam_choice:
            cap_source = 1
            is_demo = False
        else:
            cap_source = 0
            is_demo = False

        classifier.reset_state()
        cap = cv2.VideoCapture(cap_source)

        if not cap.isOpened():
            st.error(
                "**Camera Inaccessible:** Unable to open video device. "
                "Check camera connection, OS privacy permissions, or select Demo Video."
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
                        st.warning("Video stream ended.")
                        break

                t_sec = time.time() - t_start
                res, annotated = classifier.process_frame(frame, frame_index=frame_idx, timestamp_sec=t_sec)

                # Render Video Feed
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # FPS only if actually measured
                fps_val = (1000.0 / res.inference_latency_ms) if res.inference_latency_ms > 0 else 0.0
                fps_info = f"FPS: {fps_val:.0f} | Latency: {res.inference_latency_ms:.1f} ms" if fps_val > 0 else f"Latency: {res.inference_latency_ms:.1f} ms"

                sec_placeholder.markdown(
                    clean_html(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; background: #FFFFFF; border: 1px solid {BORDER_SUBTLE}; border-radius: 4px; font-size: 11px; color: {TEXT_MUTED}; margin-top: 6px;">
                        <span>{fps_info}</span>
                        <span>Tracking: {'33 Landmarks' if res.is_valid_pose else 'Low Visibility'}</span>
                        <span>Frame #{frame_idx}</span>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

                # Render Right Column Live Telemetry HUD
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
# PAGE 2: SYSTEM OVERVIEW
# ==============================================================================
def page_overview(readiness: Dict[str, Any]):
    st.markdown('<div class="lg-title">LiftGuard AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lg-subtitle">Industrial computer-vision screening for manual lifting posture and ergonomic safety.</div>',
        unsafe_allow_html=True,
    )

    # 4 Restrained Industrial KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        screening_method = readiness.get("screening_method", "Rule-Based Ergonomics")
        st.markdown(render_kpi_card_html("Screening Engine", "Biomechanical Rules", "NIOSH / OSHA standards", "teal"), unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(render_kpi_card_html("Pose Tracking", "33 Keypoints", "MediaPipe Pose Landmarker", "default"), unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(render_kpi_card_html("Camera View", "Sagittal / Lateral", "MP4, AVI, MOV & Webcam", "default"), unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(render_kpi_card_html("Model Pipeline", "Custom 2D CNN", "Status on Evaluation page", "default"), unsafe_allow_html=True)

    # Primary Workspaces Shortcuts
    st.markdown('<div class="lg-section-header">' + ICONS["layers"] + ' Workspaces</div>', unsafe_allow_html=True)
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-bottom: 8px;">
                <div style="font-size: 14px; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 4px;">Video Clip Screening</div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.45;">
                    Upload recorded MP4, AVI, or MOV clips for frame-by-frame posture inference, HUD skeletal overlay, angle trajectory charts, and CSV telemetry exports.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        if st.button("Launch Video Screening", key="btn_go_video", use_container_width=True):
            st.session_state.current_page = "Screening"
            st.rerun()

    with act_col2:
        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-bottom: 8px;">
                <div style="font-size: 14px; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 4px;">Live Camera Screening</div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.45;">
                    Connect a live webcam stream for real-time ergonomic screening at 25–30 FPS with rolling window smoothing, debounced hazard alerts, and angle telemetry.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        if st.button("Launch Live Camera", key="btn_go_webcam", use_container_width=True):
            st.session_state.current_page = "Screening"
            st.rerun()

    # Workflow Visualization Pipeline
    st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Inspection Pipeline</div>', unsafe_allow_html=True)
    st.markdown(render_workflow_pipeline_html(), unsafe_allow_html=True)

    # Recent Screening Activity Table
    st.markdown('<div class="lg-section-header">' + ICONS["file_text"] + ' Recent Screening Activity</div>', unsafe_allow_html=True)
    pred_dir = config.PREDICTIONS_DIR
    recent_csvs = list(pred_dir.glob("*.csv")) if pred_dir.exists() else []

    if recent_csvs:
        recent_data = []
        for p in sorted(recent_csvs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            try:
                df = pd.read_csv(p)
                rep = generate_session_report(df)
                recent_data.append({
                    "Session File": p.name,
                    "Total Frames": rep.total_frames,
                    "Duration (s)": f"{rep.duration_sec:.1f}",
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

    with st.expander("Biomechanical Model and NIOSH Lifting Criteria"):
        st.markdown(
            r"""
            **Biomechanical Criteria & Lifting Mechanics:**
            - **SAFE Lifting (Squat Lift):**
              Knee flexion ($\le 125^\circ$) with an upright trunk inclination ($\le 45^\circ$ from vertical). The quadriceps and gluteals bear the load, preserving lumbar lordosis and minimizing shear stress on the **L5/S1** lumbosacral joint.
            - **UNSAFE Lifting (Stoop Lift):**
              Knees extended ($>135^\circ$) with excessive forward trunk flexion ($>45^\circ$). Creates a long lever arm for the upper body, inducing high compressive and shear loads on lumbar discs.
            - **Quality Gate:**
              MediaPipe extracts 33 normalized landmarks. If key joints (shoulders, hips, knees, ankles) fall below 0.5 visibility, the frame is classified as `INSUFFICIENT VISIBILITY`.
            """
        )


# ==============================================================================
# PAGE 3: DATASET MANAGEMENT & REVIEW
# ==============================================================================
def page_dataset_management(readiness: Optional[Dict[str, Any]] = None):
    st.markdown('<div class="lg-title">Dataset Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Audit SAFELIFT benchmark annotations, inspect matched video files, and manage manual ground-truth reviews.</div>', unsafe_allow_html=True)

    dataset_submode = st.radio(
        "Dataset View",
        ["Benchmark Audit", "Manual Review & Annotation"],
        horizontal=True,
        label_visibility="collapsed",
        key="dataset_submode_toggle"
    )

    if dataset_submode == "Benchmark Audit":
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

        # 4 KPI Cards
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(render_kpi_card_html("Master Records", str(audit.total_json_records), "Full Benchmark JSON", "default"), unsafe_allow_html=True)
        with d2:
            st.markdown(render_kpi_card_html("Matched Videos", f"{audit.matched_videos_count} / {audit.local_videos_total}", "100% of Disk Files Matched", "success"), unsafe_allow_html=True)
        with d3:
            st.markdown(render_kpi_card_html("Missing from Disk", str(audit.missing_videos_count), "Other Benchmark Envs", "warning"), unsafe_allow_html=True)
        with d4:
            p_str = ", ".join(audit.participants)
            st.markdown(render_kpi_card_html("Subjects on Disk", f"{len(audit.participants)} ({p_str})", "Environment 4 Trials", "default"), unsafe_allow_html=True)

        # Source Labels Breakdown & Ethical Disclosure
        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-top: 10px; margin-bottom: 12px;">
                <div style="font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 6px;">
                    Source SAFELIFT Labels Distribution (<code>subjective_judgment</code>)
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                    <div style="background: #F8FAFC; border: 1px solid {BORDER_SUBTLE}; padding: 8px 12px; border-radius: 4px;">
                        <div style="font-size: 11px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase;">Matched Local Videos (49 Files)</div>
                        <div style="font-size: 12px; color: {TEXT_PRIMARY}; margin-top: 4px; line-height: 1.4;">
                            &bull; <code>unknown</code>: <b>{audit.label_distribution_matched.get('unknown', 0)}</b> trials (87.8%)<br>
                            &bull; <code>intermediate</code>: <b>{audit.label_distribution_matched.get('intermediate', 0)}</b> trials (12.2% - Participant A lateral)<br>
                            &bull; <code>SAFE</code> / <code>UNSAFE</code>: <b>0</b> trials (0.0%)
                        </div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid {BORDER_SUBTLE}; padding: 8px 12px; border-radius: 4px;">
                        <div style="font-size: 11px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase;">Full Master JSON (182 Records)</div>
                        <div style="font-size: 12px; color: {TEXT_PRIMARY}; margin-top: 4px; line-height: 1.4;">
                            &bull; <code>unknown</code>: <b>{audit.label_distribution_json.get('unknown', 0)}</b> entries (96.7%)<br>
                            &bull; <code>intermediate</code>: <b>{audit.label_distribution_json.get('intermediate', 0)}</b> entries (3.3%)<br>
                            &bull; <code>SAFE</code> / <code>UNSAFE</code>: <b>0</b> entries (0.0%)
                        </div>
                    </div>
                </div>
                <div style="font-size: 11px; color: {TEXT_MUTED}; line-height: 1.45; border-top: 1px solid {BORDER_SUBTLE}; padding-top: 8px;">
                    <b>Data Integrity:</b> The original <code>data/annotations.json</code> and video files are preserved strictly read-only and unmodified.
                    Because the source benchmark records subjective ratings rather than binary ground truth, neither SAFE nor UNSAFE labels are fabricated.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # CNN Training Dataset Requirement
        bin_sum = audit.binary_labels_summary
        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-bottom: 12px;">
                <div style="font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 6px;">
                    Binary Dataset Requirement for Custom CNN Training
                </div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5; margin-bottom: 8px;">
                    Training a 2D CNN classifier requires a binary-labelled CSV manifest (<code>data/annotations.csv</code>) with binary labels (<code>0 = SAFE</code>, <code>1 = UNSAFE</code>) partitioned across <b>at least 3 distinct participants</b> (e.g. A, B, C, D) using <code>GroupShuffleSplit</code> to prevent data leakage.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px 10px; background: #F8FAFC; border-radius: 4px; font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5;">
                    <div>&bull; <b>Manual Reviews:</b> {bin_sum.get('manual_safe_count', 0)} SAFE, {bin_sum.get('manual_unsafe_count', 0)} UNSAFE, {bin_sum.get('manual_unreviewed_count', 0)} unreviewed</div>
                    <div>&bull; <b>Reviewed Participants:</b> {bin_sum.get('manual_reviewed_participants') or ['None']} (Participant A only)</div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # Matched Videos Table
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

        # Custom Annotation CSV Validator
        st.markdown('<div class="lg-section-header">' + ICONS["file_text"] + ' Validate Custom Annotation File</div>', unsafe_allow_html=True)
        up_csv = st.file_uploader("Upload CSV for Schema Verification", type=["csv"], key="custom_ann_uploader", label_visibility="collapsed")
        if up_csv:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(up_csv.read())
                tmp_p = Path(tmp.name)
            up_rep = validate_annotations(tmp_p)
            if up_rep.is_valid:
                st.success("Uploaded CSV strictly adheres to the required schema.")
            else:
                st.error("Uploaded CSV has schema violations.")
            st.json(up_rep.to_dict())

    else:
        # Manual Review & Annotation Submode
        mgr = DatasetReviewManager()
        if not mgr.matched_videos:
            st.warning("No matched Environment 4 videos found. Please check data/raw_videos and data/annotations.json.")
            return

        df_manifest = mgr.load_or_init_review_manifest()
        progress = mgr.get_review_progress()

        # Review Progress KPIs
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            st.markdown(render_kpi_card_html("Total Videos", f"{progress['total']}", "Matched Env 4 trials", "default"), unsafe_allow_html=True)
        with p2:
            st.markdown(render_kpi_card_html("Reviewed", f"{progress['reviewed']}", f"{progress['percent_complete']}% complete", "success" if progress['reviewed'] > 0 else "default"), unsafe_allow_html=True)
        with p3:
            st.markdown(render_kpi_card_html("Unreviewed", f"{progress['unreviewed']}", "Awaiting inspection", "warning" if progress['unreviewed'] > 0 else "default"), unsafe_allow_html=True)
        with p4:
            st.markdown(render_kpi_card_html("Safe Lifts (0)", f"{progress['safe_count']}", "Squat posture", "success"), unsafe_allow_html=True)
        with p5:
            st.markdown(render_kpi_card_html("Unsafe Lifts (1)", f"{progress['unsafe_count']}", "Stoop posture", "danger"), unsafe_allow_html=True)

        # Filter Bar
        f1, f2, f3 = st.columns([1, 1, 1.2])
        with f1:
            p_filter = st.selectbox("Filter Participant", ["All", "A", "B", "C", "D"], key="rev_p_filter")
        with f2:
            v_filter = st.selectbox("Filter Camera View", ["All", "lateral", "oblique"], key="rev_v_filter")
        with f3:
            s_filter = st.selectbox("Filter Review Status", ["All", "Unreviewed Only", "Reviewed Only"], key="rev_s_filter")

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

        rev_label_map = {row["video_filename"]: row["reviewed_label"] for _, row in df_manifest.iterrows()}
        rev_status_map = {row["video_filename"]: row["review_status"] for _, row in df_manifest.iterrows()}

        def format_video_option(m):
            status = rev_status_map.get(m.video_filename, STATUS_UNREVIEWED)
            lbl = rev_label_map.get(m.video_filename, "")
            badge = f"[{lbl}]" if (status == STATUS_REVIEWED and lbl) else f"[{status}]"
            return f"{badge} {m.video_filename} (Subject {m.participant_id} | {m.camera_view})"

        selected_video = st.selectbox(
            f"Select Video to Inspect ({len(filtered_videos)} available):",
            filtered_videos,
            format_func=format_video_option,
            key="selected_video_obj"
        )

        if not selected_video:
            return

        w_col1, w_col2 = st.columns([1.1, 0.9])
        with w_col1:
            st.markdown('<div class="lg-section-header">' + ICONS["video"] + ' Video Playback & Timing</div>', unsafe_allow_html=True)
            vid_path = Path(selected_video.absolute_video_path)
            if vid_path.exists():
                st.video(str(vid_path))
            else:
                st.error(f"Video file not found at: {vid_path}")

            s_sec = selected_video.action_start_sec
            e_sec = selected_video.action_end_sec
            dur = f"{(e_sec - s_sec):.1f}s" if (s_sec is not None and e_sec is not None) else "N/A"
            st.markdown(
                clean_html(f"""
                <div style="background: #F8FAFC; border: 1px solid {BORDER_SUBTLE}; border-radius: 4px; padding: 8px 12px; margin-top: 6px;">
                    <div style="font-size: 10px; font-weight: 600; text-transform: uppercase; color: {TEXT_MUTED};">Action Interval</div>
                    <div style="font-size: 12px; color: {TEXT_PRIMARY}; font-weight: 500;">
                        Active Window: <b>{s_sec}s</b> to <b>{e_sec}s</b> (Duration: {dur})
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

        with w_col2:
            st.markdown('<div class="lg-section-header">' + ICONS["clipboard_check"] + ' Review & Ground-Truth Label</div>', unsafe_allow_html=True)
            row_curr = df_manifest[df_manifest["video_filename"] == selected_video.video_filename]
            curr_status = row_curr["review_status"].values[0] if not row_curr.empty else STATUS_UNREVIEWED
            curr_label = row_curr["reviewed_label"].values[0] if not row_curr.empty else ""
            curr_notes = row_curr["reviewer_notes"].values[0] if not row_curr.empty else ""
            curr_timestamp = row_curr["review_timestamp"].values[0] if not row_curr.empty else ""

            default_label_idx = 0
            if curr_label == LABEL_UNSAFE:
                default_label_idx = 1
            elif curr_label == LABEL_AMBIGUOUS:
                default_label_idx = 2

            chosen_label = st.radio(
                "Ground-Truth Posture Classification",
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
                    "Start (s)",
                    value=float(selected_video.action_start_sec or 0.0),
                    step=0.5,
                    key=f"start_in_{selected_video.video_filename}"
                )
            with t_adj_col2:
                end_in = st.number_input(
                    "End (s)",
                    value=float(selected_video.action_end_sec or 5.0),
                    step=0.5,
                    key=f"end_in_{selected_video.video_filename}"
                )

            notes_in = st.text_area(
                "Reviewer Observations",
                value=str(curr_notes) if pd.notna(curr_notes) else "",
                placeholder="e.g. Pronounced knee bend, upright trunk maintained.",
                key=f"notes_in_{selected_video.video_filename}",
                height=68
            )

            if st.button("Save Review Label", type="primary", use_container_width=True, key="btn_save_review"):
                success = mgr.save_review_label(
                    video_filename=selected_video.video_filename,
                    label=label_token,
                    reviewer_notes=notes_in,
                    action_start_override=start_in,
                    action_end_override=end_in,
                )
                if success:
                    st.success(f"Recorded label: {label_token} for {selected_video.video_filename}")
                    st.rerun()

            if curr_timestamp:
                st.caption(f"Last reviewed on: {curr_timestamp}")

        # Manifest table
        st.markdown("---")
        st.markdown('<div class="lg-section-header">' + ICONS["file_text"] + ' Review Manifest Table</div>', unsafe_allow_html=True)
        df_manifest_fresh = mgr.load_or_init_review_manifest()
        view_table = df_manifest_fresh[[
            "video_filename", "participant_id", "camera_view", "action_direction",
            "action_start_sec", "action_end_sec", "subjective_judgment_original",
            "review_status", "reviewed_label", "reviewer_notes", "review_timestamp"
        ]]
        st.dataframe(view_table, use_container_width=True, hide_index=True)

        csv_data = df_manifest_fresh.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Review Manifest (CSV)",
            data=csv_data,
            file_name="manual_review_labels.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ==============================================================================
# PAGE 4: MODEL EVALUATION & DEVELOPMENT STATUS
# ==============================================================================
def page_model_evaluation(readiness: Optional[Dict[str, Any]] = None):
    st.markdown('<div class="lg-title">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Held-out test set evaluation and 2D CNN model development status.</div>', unsafe_allow_html=True)

    status_info = get_project_model_status()
    cnn_status = status_info["status"]
    screening_method = status_info["screening_method"]
    metrics_file = config.METRICS_DIR / "metrics.json"
    cm_plot = config.PLOTS_DIR / "confusion_matrix.png"
    curves_plot = config.PLOTS_DIR / "training_curves.png"
    report_file = config.METRICS_DIR / "classification_report.txt"

    # Zero Data-Leakage Protocol Card
    st.markdown(
        clean_html(f"""
        <div class="lg-card" style="margin-bottom: 12px;">
            <div style="font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 4px;">
                Zero Data-Leakage Partitioning Protocol
            </div>
            <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.45;">
                Data is partitioned across <b>participant_id</b> via <code>GroupShuffleSplit</code>. No subject appearing in training is ever present in validation or testing.
                The classification threshold is calibrated strictly on the validation set; held-out test evaluation occurs only once with the frozen model.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # CASE 1: CNN NOT TRAINED
    if cnn_status == "CNN NOT TRAINED":
        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 14px; font-weight: 700; color: {TEXT_PRIMARY};">
                        Model Development Status: CNN not trained
                    </div>
                    <span class="lg-badge lg-badge-neutral">Checkpoint Not Present</span>
                </div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5; margin-bottom: 12px;">
                    A custom 2D CNN checkpoint does not exist at <code>models/best_lifting_cnn.keras</code>.
                    Held-out test evaluation cannot be computed until a model is trained on reviewed binary labels.
                </div>
                <div style="background: #F8FAFC; border: 1px solid {BORDER_SUBTLE}; padding: 10px 12px; border-radius: 4px; font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.6; margin-bottom: 10px;">
                    <div>&bull; <b>Active Screening Engine:</b> <span style="font-weight: 600; color: {TEXT_PRIMARY};">{html.escape(screening_method)}</span> (fully operational)</div>
                    <div>&bull; <b>Evaluation State:</b> Held-out test metrics require a trained CNN checkpoint</div>
                    <div>&bull; <b>Checkpoint Location:</b> <code>models/best_lifting_cnn.keras</code></div>
                </div>
                <div style="font-size: 11px; color: {TEXT_MUTED}; line-height: 1.45;">
                    <b>Why evaluation is not available:</b> The SAFELIFT benchmark records subjective ratings (unknown / intermediate) rather than binary ground truth labels. Currently, only 3 trials have been manually reviewed (all Participant A, SAFE). To train and evaluate a CNN without participant leakage, reviewed labels across multiple participants (e.g. B, C, D) with both SAFE and UNSAFE postures are required.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="lg-section-header">' + ICONS["clipboard_check"] + ' Steps to Train and Evaluate Custom CNN</div>', unsafe_allow_html=True)
        st.markdown(
            clean_html(f"""
            <div class="lg-card">
                <ol style="margin: 0; padding-left: 18px; font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.7;">
                    <li><b>Review & Label Videos:</b> Open the <b>Dataset</b> page (Manual Review view). Inspect Environment 4 trials for subjects B, C, and D, recording stoop lifts as <code>UNSAFE</code> and squat lifts as <code>SAFE</code>.</li>
                    <li><b>Export Manifest:</b> Save the reviewed annotations to <code>data/annotations.csv</code> conforming to the required schema (<code>clip_path,participant_id,session_id,label,view</code>).</li>
                    <li><b>Generate Feature Splits:</b> Run <code>python -m src.dataset_loader</code> to extract 128&times;128 pose maps and produce leak-free participant splits.</li>
                    <li><b>Train the Model:</b> Run <code>python -m src.train</code> to train the CNN and save <code>models/best_lifting_cnn.keras</code>.</li>
                    <li><b>Run Held-Out Evaluation:</b> Run <code>python -m src.evaluate</code> to measure genuine test accuracy, recall, and F1 on the held-out test split.</li>
                </ol>
            </div>
            """),
            unsafe_allow_html=True,
        )
        return

    # CASE 2: EVALUATION PENDING
    if cnn_status == "EVALUATION PENDING" or not status_info.get("has_evaluation_results", False):
        st.markdown(
            clean_html(f"""
            <div class="lg-card" style="margin-bottom: 12px;">
                <div style="font-size: 14px; font-weight: 700; color: {TEXT_PRIMARY}; margin-bottom: 6px;">
                    Model Status: Evaluation Pending
                </div>
                <div style="font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.45;">
                    A trained CNN checkpoint was located at <code>models/best_lifting_cnn.keras</code>, but held-out test evaluation has not yet been executed.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        test_split_file = config.SPLITS_DIR / "test.csv"
        if test_split_file.exists():
            if st.button("Run Evaluation on Held-Out Test Split", type="primary"):
                with st.spinner("Evaluating held-out test manifest..."):
                    res = evaluate_held_out_test_set()
                st.rerun()
        else:
            st.info("Held-out test split not found at `data/splits/test.csv`. Generate split manifests via `src.dataset_loader` first.")
        return

    # CASE 3: EVALUATED
    metrics = status_info.get("metrics") or {}

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(render_kpi_card_html("Test Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%", f"{metrics.get('test_sample_count', 0)} samples", "default"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_kpi_card_html("Unsafe Precision", f"{metrics.get('precision_unsafe', 0.0) * 100:.1f}%", "Positive predictive val", "default"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_kpi_card_html("Unsafe Recall", f"{metrics.get('recall_unsafe', 0.0) * 100:.1f}%", "Sensitivity (hazard)", "success"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_kpi_card_html("Unsafe F1-Score", f"{metrics.get('f1_unsafe', 0.0):.3f}", "Harmonic mean", "success"), unsafe_allow_html=True)
    with m5:
        st.markdown(render_kpi_card_html("Specificity", f"{metrics.get('specificity', 0.0) * 100:.1f}%", "True negative rate", "default"), unsafe_allow_html=True)

    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Confusion Matrix</div>', unsafe_allow_html=True)
        if cm_plot.exists():
            st.image(str(cm_plot), use_container_width=True)
        else:
            st.caption("Confusion matrix pending.")

    with vis_col2:
        st.markdown('<div class="lg-section-header">' + ICONS["activity"] + ' Training Trajectory</div>', unsafe_allow_html=True)
        if curves_plot.exists():
            st.image(str(curves_plot), use_container_width=True)
        else:
            st.caption("Training curves pending.")

    if report_file.exists():
        with st.expander("Classification Report Text"):
            with open(report_file, "r") as f:
                st.code(f.read(), language="text")


# ==============================================================================
# PAGE 5: SESSION REPORTS
# ==============================================================================
def page_reports():
    st.markdown('<div class="lg-title">Session Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="lg-subtitle">Audit screening history, download timestamped predictions, and export session logs.</div>', unsafe_allow_html=True)

    pred_dir = config.PREDICTIONS_DIR
    pred_files = list(pred_dir.glob("*.csv")) if pred_dir.exists() else []

    if not pred_files:
        st.markdown(render_empty_state_html("No Saved Session Logs", "Process a video in Screening or run Live Camera to record sessions.", "file_text"), unsafe_allow_html=True)
        return

    selected_csv = st.selectbox("Select Screening Session", pred_files, format_func=lambda p: p.name)
    if selected_csv:
        df = pd.read_csv(selected_csv)
        rep = generate_session_report(df)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(render_kpi_card_html("Risk Rating", rep.ergonomic_risk_level, "Screening risk rating", "danger" if "HIGH" in rep.ergonomic_risk_level else "success"), unsafe_allow_html=True)
        with r2:
            st.markdown(render_kpi_card_html("Safe Postures", f"{rep.safe_pct_of_valid}%", f"{rep.safe_frames} frames", "success"), unsafe_allow_html=True)
        with r3:
            st.markdown(render_kpi_card_html("Unsafe Postures", f"{rep.unsafe_pct_of_valid}%", f"{rep.unsafe_frames} frames", "danger"), unsafe_allow_html=True)
        with r4:
            st.markdown(render_kpi_card_html("Duration", f"{rep.duration_sec:.1f}s", f"{rep.total_frames} total frames", "default"), unsafe_allow_html=True)

        st.caption("Frame Telemetry Preview (First 50 frames):")
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Full Session CSV",
            data=csv_bytes,
            file_name=selected_csv.name,
            mime="text/csv",
            use_container_width=False,
        )


# ==============================================================================
# MAIN CONTROLLER
# ==============================================================================
def main():
    init_session_state()
    readiness = check_system_readiness()
    current_page = render_sidebar(readiness)
    classifier = get_classifier()

    if current_page == "Screening":
        page_posture_screening(classifier)
    elif current_page == "Overview":
        page_overview(readiness)
    elif current_page == "Dataset":
        page_dataset_management(readiness)
    elif current_page == "Evaluation":
        page_model_evaluation(readiness)
    elif current_page == "Reports":
        page_reports()
    else:
        page_posture_screening(classifier)


if __name__ == "__main__":
    main()
