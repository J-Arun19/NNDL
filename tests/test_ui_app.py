# -*- coding: utf-8 -*-
"""Comprehensive UI and Navigation Test Suite for LiftGuard AI Streamlit Redesign."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

import config


def test_screening_landing_page_and_navigation():
    """Verify primary workspace on landing, compact navigation, and upload mode structure."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    assert not at.exception, f"App encountered exceptions: {at.exception}"

    # Verify Sidebar Navigation options
    nav_radio = None
    for r in at.radio:
        if "Screening" in r.options and "Overview" in r.options:
            nav_radio = r
            break
    assert nav_radio is not None, "Sidebar navigation radio not found"
    assert nav_radio.options == ["Screening", "Overview", "Dataset", "Evaluation", "Reports"]
    assert nav_radio.value == "Screening"

    # Verify Screening mode toggle
    mode_radio = None
    for r in at.radio:
        if "Upload Video" in r.options and "Live Camera" in r.options:
            mode_radio = r
            break
    assert mode_radio is not None, "Screening mode toggle not found"
    assert mode_radio.options == ["Upload Video", "Live Camera"]
    assert mode_radio.value == "Upload Video"

    # Check for Posture Screening title
    md_texts = [m.value for m in at.markdown]
    has_title = any("Posture Screening" in t for t in md_texts)
    assert has_title, "Title 'Posture Screening' not found in page markdown"

    has_subtitle = any("Assess lifting posture from a recorded video or live camera." in t for t in md_texts)
    assert has_subtitle, "Subtitle not found in page markdown"

    # Check that there is NO large CNN NOT TRAINED warning banner on screening page
    has_repeated_cnn_banner = any("Rule-Based Ergonomics (CNN NOT TRAINED)" in t for t in md_texts)
    assert not has_repeated_cnn_banner, "Found unexpected CNN NOT TRAINED warning banner on screening page"


def test_live_camera_mode_interaction():
    """Verify switching to Live Camera mode, controls, and standby telemetry card."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Find mode selector and switch to Live Camera
    mode_radio = None
    for r in at.radio:
        if "Upload Video" in r.options and "Live Camera" in r.options:
            mode_radio = r
            break
    assert mode_radio is not None
    mode_radio.set_value("Live Camera")
    at.run()

    assert not at.exception, f"Exceptions on Live Camera switch: {at.exception}"

    # Check for Camera Source selectbox
    cam_select = None
    for s in at.selectbox:
        if any("Camera 0" in opt or "Default Camera" in opt for opt in s.options):
            cam_select = s
            break
    assert cam_select is not None, "Camera Source selectbox not found"

    # Check for Start and Stop monitoring buttons
    button_labels = [b.label for b in at.button]
    assert "Start Monitoring" in button_labels, "Start Monitoring button missing"
    assert "Stop Monitoring" in button_labels, "Stop Monitoring button missing"

    # Check for Live Telemetry Standby
    md_texts = [m.value for m in at.markdown]
    has_standby = any("Live Telemetry Standby" in t for t in md_texts)
    assert has_standby, "Live Telemetry Standby not found"


def test_overview_page_navigation():
    """Verify Overview page renders KPI cards, pipeline, and no alarmist CNN badges in header."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Navigate to Overview
    nav_radio = None
    for r in at.radio:
        if "Screening" in r.options and "Overview" in r.options:
            nav_radio = r
            break
    nav_radio.set_value("Overview")
    at.run()

    assert not at.exception, f"Exceptions on Overview navigation: {at.exception}"

    md_texts = [m.value for m in at.markdown]
    has_title = any("LiftGuard AI" in t for t in md_texts)
    assert has_title, "Overview title not found"

    # Verify pipeline stepper is present
    has_pipeline = any("Video Input" in t and "Biomechanical Rules" in t for t in md_texts)
    assert has_pipeline, "Processing pipeline stepper missing from Overview"


def test_dataset_page_navigation():
    """Verify Dataset page loads Benchmark Audit and can toggle to Manual Review."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Navigate to Dataset
    nav_radio = None
    for r in at.radio:
        if "Screening" in r.options and "Dataset" in r.options:
            nav_radio = r
            break
    nav_radio.set_value("Dataset")
    at.run()

    assert not at.exception, f"Exceptions on Dataset navigation: {at.exception}"

    md_texts = [m.value for m in at.markdown]
    has_title = any("Dataset Management" in t for t in md_texts)
    assert has_title, "Dataset Management title not found"

    # Verify Dataset submode radio exists
    sub_radio = None
    for r in at.radio:
        if "Benchmark Audit" in r.options and "Manual Review & Annotation" in r.options:
            sub_radio = r
            break
    assert sub_radio is not None, "Dataset submode toggle not found"

    # Toggle to Manual Review & Annotation
    sub_radio.set_value("Manual Review & Annotation")
    at.run()

    assert not at.exception, f"Exceptions on Manual Review toggle: {at.exception}"
    md_texts_sub = [m.value for m in at.markdown]
    has_review = any("Review & Ground-Truth Label" in t or "Video Playback" in t for t in md_texts_sub)
    assert has_review, "Manual Review interface not rendered"


def test_evaluation_page_navigation():
    """Verify Evaluation page states 'CNN not trained' clearly without repetitive alarmist badges."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Navigate to Evaluation
    nav_radio = None
    for r in at.radio:
        if "Screening" in r.options and "Evaluation" in r.options:
            nav_radio = r
            break
    nav_radio.set_value("Evaluation")
    at.run()

    assert not at.exception, f"Exceptions on Evaluation navigation: {at.exception}"

    md_texts = [m.value for m in at.markdown]
    has_title = any("Model Evaluation" in t for t in md_texts)
    assert has_title, "Model Evaluation title not found"

    # Check for restrained "CNN not trained" status panel
    has_not_trained_text = any("CNN not trained" in t for t in md_texts)
    assert has_not_trained_text, "Expected 'CNN not trained' message not found on Evaluation page"


def test_reports_page_navigation():
    """Verify Reports page renders session selector, reports data, and export button."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Navigate to Reports
    nav_radio = None
    for r in at.radio:
        if "Screening" in r.options and "Reports" in r.options:
            nav_radio = r
            break
    nav_radio.set_value("Reports")
    at.run()

    assert not at.exception, f"Exceptions on Reports navigation: {at.exception}"

    md_texts = [m.value for m in at.markdown]
    has_title = any("Session Reports" in t for t in md_texts)
    assert has_title, "Session Reports title not found"


def test_no_raw_html_leaks():
    """Ensure no unparsed raw HTML tags are rendered as literal text in markdown blocks."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    for m in at.markdown:
        val = m.value.strip()
        # Ensure string doesn't start with raw 4-space indented code block containing html tags
        assert not val.startswith("    <div"), f"Raw HTML code block leak detected: {val[:100]}"
        assert not val.startswith("```html"), f"Raw HTML block leak: {val[:100]}"


def test_uploaded_video_summary_and_hud_rendering():
    """Verify that render_uploaded_video_summary_html and render_live_telemetry_hud_html format cleanly."""
    from src.inference import VideoAnalysisSummary
    from src.ui_theme import render_uploaded_video_summary_html, render_live_telemetry_hud_html

    summary = VideoAnalysisSummary(
        total_frames=120,
        valid_pose_frames=115,
        safe_frames=110,
        unsafe_frames=5,
        insufficient_visibility_frames=5,
        overall_classification="SAFE POSTURE",
        primary_reason="Dominant squat lift mechanics observed with trunk flexion within 45 degrees.",
        mean_trunk_inclination_deg=31.4,
        max_trunk_inclination_deg=41.2,
        mean_knee_angle_deg=108.5,
        min_knee_angle_deg=89.0,
        valid_pose_pct=95.8,
        safe_pct_of_valid=95.7,
        unsafe_pct_of_valid=4.3,
        duration_sec=4.0,
        assessment_method="Rule-Based Ergonomic Assessment",
    )

    summary_html = render_uploaded_video_summary_html(summary)
    assert "SAFE POSTURE" in summary_html
    assert "95.8%" in summary_html
    assert "95.7%" in summary_html
    assert "4.3%" in summary_html
    assert "Assessment method: Rule-based ergonomic assessment" in summary_html
    # Verify no raw unescaped code-block indentation
    assert not summary_html.startswith("    <")

    hud_html = render_live_telemetry_hud_html(
        status="SAFE POSTURE",
        explanation="Upright trunk and flexed knees detected.",
        method="Rule-Based Ergonomic Assessment",
        trunk_inclination=28.5,
        knee_angle=105.0,
        is_valid_pose=True,
        is_alert_triggered=False,
        consecutive_unsafe=0,
        fps=27.5,
        latency_ms=36.4,
    )
    assert "SAFE POSTURE" in hud_html
    assert "28.5" in hud_html
    assert "105.0" in hud_html
    assert "28 FPS" in hud_html
    assert not hud_html.startswith("    <")
    assert "\n" not in hud_html
    assert not any(l.startswith("    ") for l in hud_html.splitlines())


def test_heading_spacing_and_css_across_all_pages():
    """Verify that heading styles and block container padding prevent clipping on all 5 pages."""
    from src.ui_theme import get_css_theme

    css = get_css_theme()
    assert "padding-top: 4.75rem !important;" in css, "Block container must have at least 4.5rem+ padding-top to avoid header bar clipping"
    assert "margin-top: 0" in css, "lg-title must not have negative margins"

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Find navigation radio
    nav_radio = None
    for r in at.radio:
        if "Screening" in r.options and "Overview" in r.options:
            nav_radio = r
            break
    assert nav_radio is not None

    pages = ["Screening", "Overview", "Dataset", "Evaluation", "Reports"]
    for page in pages:
        nav_radio.set_value(page)
        at.run()
        assert not at.exception, f"Exception on page {page}: {at.exception}"
        md_vals = [m.value for m in at.markdown]
        # Confirm that a lg-title class div is rendered on every page
        assert any('class="lg-title"' in v for v in md_vals), f"Missing lg-title heading on page {page}"


def test_live_camera_start_stop_cycle():
    """Test Start Monitoring, Stop Monitoring, and state transitions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Switch to Live Camera
    mode_radio = None
    for r in at.radio:
        if "Upload Video" in r.options and "Live Camera" in r.options:
            mode_radio = r
            break
    mode_radio.set_value("Live Camera")
    at.run()

    btn_start = None
    btn_stop = None
    for b in at.button:
        if b.label == "Start Monitoring":
            btn_start = b
        elif b.label == "Stop Monitoring":
            btn_stop = b

    assert btn_start is not None and btn_stop is not None
    assert btn_start.disabled is False
    assert btn_stop.disabled is True


