"""UI Theme and Visual Components for LiftGuard AI.

Provides the dark industrial AI design system, color tokens, clean SVG icons,
and reusable layout components for the Streamlit dashboard.
"""

from __future__ import annotations

import html
import textwrap
from typing import Any, Dict, Optional

# Color Palette Tokens - Professional Computer Vision Dashboard
BG_MAIN = "#0B0F14"           # Deep Charcoal Canvas
BG_PANEL = "#111821"          # Sidebar & Outer Surface
BG_PANEL_ELEVATED = "#151D28" # Card & Panel Surface
BG_PANEL_HOVER = "#1C2736"    # Interactive Elevated Surface
BORDER_SUBTLE = "#273444"     # Subtle Slate Border
BORDER_FOCUS = "#10B981"      # Primary Teal/Green Focus Accent
ACCENT_MINT = "#10B981"       # Primary Teal/Green (SAFE)
ACCENT_CYAN = "#06B6D4"       # Secondary Metric Cyan
ACCENT_AMBER = "#F59E0B"      # Restrained Warning (INSUFFICIENT VISIBILITY)
ACCENT_RED = "#EF4444"        # Ergonomic Hazard / Unsafe Alert (UNSAFE)
TEXT_PRIMARY = "#F8FAFC"      # Near-white High-contrast Headers & Text
TEXT_SECONDARY = "#94A3B8"    # Readable Slate-Gray Body
TEXT_MUTED = "#64748B"        # Subtle Captions / Metadata


def clean_html(raw_html: str) -> str:
    """Safely dedent and trim multiline HTML string.
    
    Prevents Streamlit / Python-Markdown from interpreting 4+ leading spaces
    as an indented code block (<pre><code>).
    """
    return textwrap.dedent(raw_html).strip()


# Crisp Industrial SVG Icons
ICONS = {
    "shield_check": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>""",
    "alert_triangle": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>""",
    "help_circle": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>""",
    "video": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>""",
    "camera": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>""",
    "activity": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>""",
    "database": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>""",
    "file_text": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>""",
    "cpu": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>""",
    "check_circle": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>""",
    "x_circle": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>""",
    "chevron_right": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>""",
    "layers": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>""",
    "clipboard_check": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect><path d="m9 14 2 2 4-4"></path></svg>""",
}


def get_css_theme() -> str:
    """Return the centralized CSS injection for Streamlit."""
    return f"""
    <style>
        /* Modern Inter Typography & Canvas Reset */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            letter-spacing: -0.01em;
        }}

        /* Dark Charcoal App Background */
        .stApp {{
            background-color: {BG_MAIN} !important;
            color: {TEXT_PRIMARY} !important;
        }}

        /* Header Bar Styling */
        header[data-testid="stHeader"] {{
            background-color: rgba(11, 15, 20, 0.92) !important;
            backdrop-filter: blur(8px);
            border-bottom: 1px solid {BORDER_SUBTLE} !important;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {BG_PANEL} !important;
            border-right: 1px solid {BORDER_SUBTLE} !important;
            min-width: 250px !important;
            max-width: 270px !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        /* Sidebar Radio Navigation */
        [data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
            display: none !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {{
            background-color: {BG_PANEL_ELEVATED} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            padding: 9px 12px !important;
            border-radius: 8px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            color: #CBD5E1 !important;
            transition: all 0.15s ease !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
            background-color: {BG_PANEL_HOVER} !important;
            border-color: #384B61 !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {{
            background-color: rgba(16, 185, 129, 0.15) !important;
            border-color: {ACCENT_MINT} !important;
            color: {ACCENT_MINT} !important;
            font-weight: 700 !important;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.12) !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}

        /* Button Styling - High Contrast, Prominent States */
        button[data-testid="baseButton-primary"],
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {{
            background-color: {ACCENT_MINT} !important;
            background-image: none !important;
            color: #0B0F14 !important;
            border: 1px solid {ACCENT_MINT} !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            padding: 8px 16px !important;
            letter-spacing: -0.01em !important;
            transition: all 0.15s ease !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
        }}
        button[data-testid="baseButton-primary"]:hover,
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {{
            background-color: #059669 !important;
            border-color: #059669 !important;
            color: #FFFFFF !important;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.35) !important;
        }}
        button[data-testid="baseButton-primary"]:active,
        .stButton > button[kind="primary"]:active {{
            background-color: #047857 !important;
            transform: translateY(1px);
        }}
        button[data-testid="baseButton-primary"]:disabled,
        .stButton > button[kind="primary"]:disabled {{
            background-color: {BG_PANEL_HOVER} !important;
            border-color: {BORDER_SUBTLE} !important;
            color: {TEXT_MUTED} !important;
            cursor: not-allowed !important;
            box-shadow: none !important;
        }}

        button[data-testid="baseButton-secondary"],
        .stButton > button[kind="secondary"],
        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button {{
            background-color: {BG_PANEL_ELEVATED} !important;
            background-image: none !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            color: {TEXT_PRIMARY} !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            padding: 8px 16px !important;
            transition: all 0.15s ease !important;
        }}
        button[data-testid="baseButton-secondary"]:hover,
        .stButton > button[kind="secondary"]:hover,
        .stButton > button:not([kind="primary"]):hover,
        .stDownloadButton > button:hover {{
            background-color: {BG_PANEL_HOVER} !important;
            border-color: {ACCENT_MINT} !important;
            color: {ACCENT_MINT} !important;
        }}
        button[data-testid="baseButton-secondary"]:active,
        .stButton > button[kind="secondary"]:active,
        .stDownloadButton > button:active {{
            background-color: #223145 !important;
            transform: translateY(1px);
        }}
        button[data-testid="baseButton-secondary"]:disabled,
        .stButton > button[kind="secondary"]:disabled,
        .stDownloadButton > button:disabled {{
            background-color: {BG_PANEL} !important;
            border-color: #1E293B !important;
            color: {TEXT_MUTED} !important;
            cursor: not-allowed !important;
        }}

        /* File Uploader - Deep Slate Theme, Visible Controls */
        div[data-testid="stFileUploader"] {{
            background: transparent !important;
        }}
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {{
            background-color: {BG_PANEL_ELEVATED} !important;
            border: 1.5px dashed {BORDER_SUBTLE} !important;
            border-radius: 8px !important;
            padding: 1.5rem !important;
            color: {TEXT_PRIMARY} !important;
            transition: border-color 0.15s ease, background-color 0.15s ease;
        }}
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {{
            border-color: {ACCENT_MINT} !important;
            background-color: {BG_PANEL_HOVER} !important;
        }}
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {{
            background-color: {BG_PANEL_HOVER} !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            padding: 6px 14px !important;
            transition: all 0.15s ease !important;
        }}
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:hover {{
            background-color: {ACCENT_MINT} !important;
            color: #0B0F14 !important;
            border-color: {ACCENT_MINT} !important;
        }}
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderInstructions"] {{
            color: {TEXT_PRIMARY} !important;
        }}
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderInstructions"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] label {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* Tabs Styling */
        div[data-baseweb="tab-list"] {{
            background-color: transparent !important;
            border-bottom: 1px solid {BORDER_SUBTLE} !important;
            gap: 8px;
            margin-bottom: 1.25rem;
        }}
        button[data-baseweb="tab"] {{
            background-color: transparent !important;
            color: {TEXT_SECONDARY} !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 10px 16px !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            transition: all 0.15s ease !important;
        }}
        button[data-baseweb="tab"]:hover {{
            color: {TEXT_PRIMARY} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {ACCENT_MINT} !important;
            border-bottom: 2px solid {ACCENT_MINT} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {ACCENT_MINT} !important;
        }}

        /* Inputs, Sliders & Selectboxes */
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-baseweb="select"] > div {{
            background-color: {BG_PANEL_ELEVATED} !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stTextInput"] input:focus {{
            border-color: {ACCENT_MINT} !important;
            box-shadow: 0 0 0 1px {ACCENT_MINT} !important;
        }}
        div[data-testid="stNumberInput"] button {{
            background-color: {BG_PANEL_HOVER} !important;
            color: #CBD5E1 !important;
            border: 1px solid {BORDER_SUBTLE} !important;
        }}
        div[data-testid="stNumberInput"] button:hover {{
            background-color: {BORDER_SUBTLE} !important;
            color: {ACCENT_MINT} !important;
        }}

        div[data-testid="stSlider"] div[role="slider"] {{
            background-color: {ACCENT_MINT} !important;
            border-color: {ACCENT_MINT} !important;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.4) !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] div {{
            color: {TEXT_PRIMARY} !important;
        }}
        div[data-baseweb="slider"] > div:nth-child(1) > div:nth-child(1) {{
            background-color: {ACCENT_MINT} !important;
        }}

        /* Cards & Metric Panels */
        .lg-card {{
            background-color: {BG_PANEL_ELEVATED};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 16px;
            transition: border-color 0.15s ease;
        }}
        .lg-card:hover {{
            border-color: #384B61;
        }}

        .lg-kpi-card {{
            background-color: {BG_PANEL_ELEVATED};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 8px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 96px;
        }}
        .lg-kpi-label {{
            font-size: 11px;
            font-weight: 500;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }}
        .lg-kpi-val {{
            font-size: 24px;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            line-height: 1.2;
            margin-bottom: 2px;
        }}
        .lg-kpi-sub {{
            font-size: 11px;
            color: {TEXT_SECONDARY};
            font-weight: 400;
        }}

        /* Badges & Status Pills */
        .lg-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .lg-badge-ready {{
            background-color: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }}
        .lg-badge-missing {{
            background-color: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }}
        .lg-badge-warning {{
            background-color: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }}

        /* Action Cards */
        .lg-action-box {{
            background-color: {BG_PANEL_ELEVATED};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 8px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 100%;
        }}
        .lg-action-box:hover {{
            border-color: {ACCENT_MINT};
        }}
        .lg-action-title {{
            font-size: 16px;
            font-weight: 600;
            color: {TEXT_PRIMARY};
            margin-top: 10px;
            margin-bottom: 6px;
        }}
        .lg-action-desc {{
            font-size: 13px;
            color: {TEXT_SECONDARY};
            line-height: 1.45;
            flex-grow: 1;
            margin-bottom: 16px;
        }}

        /* Workflow Pipeline */
        .lg-pipeline {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: {BG_PANEL_ELEVATED};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 8px;
            padding: 12px 18px;
            margin: 14px 0 24px 0;
            overflow-x: auto;
        }}
        .lg-pipe-step {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 500;
            color: {TEXT_PRIMARY};
            white-space: nowrap;
        }}
        .lg-pipe-num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: rgba(16, 185, 129, 0.18);
            color: {ACCENT_MINT};
            font-size: 11px;
            font-weight: 700;
        }}
        .lg-pipe-arrow {{
            color: {TEXT_MUTED};
            font-size: 14px;
            margin: 0 4px;
        }}

        /* Section Headings */
        .lg-title {{
            font-size: 24px;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            margin-bottom: 4px;
        }}
        .lg-subtitle {{
            font-size: 13px;
            color: {TEXT_SECONDARY};
            margin-bottom: 18px;
        }}
        .lg-section-header {{
            font-size: 15px;
            font-weight: 600;
            color: {TEXT_PRIMARY};
            margin: 14px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Clean Scrollbars */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {BG_MAIN};
        }}
        ::-webkit-scrollbar-thumb {{
            background: #1E293B;
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #334155;
        }}
    </style>
    """


def render_sidebar_brand() -> str:
    """Return HTML for the compact brand header in sidebar."""
    return clean_html(f"""
    <div style="padding: 4px 0 16px 0; border-bottom: 1px solid {BORDER_SUBTLE}; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; gap: 9px;">
            <div style="width: 28px; height: 28px; border-radius: 6px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); display: flex; align-items: center; justify-content: center;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            </div>
            <div>
                <div style="font-size: 15px; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: -0.02em; line-height: 1.1;">LiftGuard AI</div>
                <div style="font-size: 10px; font-weight: 500; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.06em;">Vision Inspection</div>
            </div>
        </div>
    </div>
    """)


def render_sidebar_status_footer(readiness: dict, threshold: float = 0.5) -> str:
    """Return HTML for the compact model-status indicator at sidebar footer."""
    cnn_status = readiness.get("cnn_status", "CNN NOT TRAINED")
    if not readiness.get("model_trained", False):
        badge_html = f"""<span class="lg-badge lg-badge-missing">{ICONS['x_circle']} CNN NOT TRAINED</span>"""
    elif not readiness.get("evaluation_available", False) or cnn_status == "EVALUATION PENDING":
        badge_html = f"""<span class="lg-badge lg-badge-warning">{ICONS['help_circle']} EVALUATION PENDING</span>"""
    else:
        badge_html = f"""<span class="lg-badge lg-badge-ready">{ICONS['check_circle']} CNN EVALUATED</span>"""

    screening_method = html.escape(str(readiness.get("screening_method", "Rule-Based Ergonomic Assessment")))

    return clean_html(f"""
    <div style="margin-top: 20px; padding: 12px; background: {BG_PANEL_ELEVATED}; border: 1px solid {BORDER_SUBTLE}; border-radius: 8px;">
        <div style="font-size: 10px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">Model Status</div>
        <div style="margin-bottom: 10px;">{badge_html}</div>
        <div style="font-size: 10px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">Screening Method</div>
        <div style="font-size: 11px; font-weight: 600; color: {ACCENT_MINT}; margin-bottom: 10px; line-height: 1.3;">
            {screening_method}
        </div>
        <div style="font-size: 10px; color: {TEXT_MUTED}; line-height: 1.35; border-top: 1px solid {BORDER_SUBTLE}; padding-top: 8px;">
            Screening is active via ergonomic rules. Custom CNN is separate.
        </div>
    </div>
    """)


def render_kpi_card_html(label: str, value: str, sublabel: str, accent: str = "default") -> str:
    """Generate a clean compact KPI card HTML."""
    val_color = TEXT_PRIMARY
    if accent == "success":
        val_color = ACCENT_MINT
    elif accent == "danger":
        val_color = ACCENT_RED
    elif accent == "warning":
        val_color = ACCENT_AMBER

    return clean_html(f"""
    <div class="lg-kpi-card">
        <div class="lg-kpi-label">{html.escape(label)}</div>
        <div class="lg-kpi-val" style="color: {val_color};">{html.escape(value)}</div>
        <div class="lg-kpi-sub">{html.escape(sublabel)}</div>
    </div>
    """)


def render_workflow_pipeline_html() -> str:
    """Generate the compact horizontal workflow pipeline stepper."""
    steps = [
        ("1", "Video Input"),
        ("2", "Pose Estimation"),
        ("3", "Quality Gate"),
        ("4", "128x128 Pose Map"),
        ("5", "2D CNN"),
        ("6", "Screening HUD")
    ]
    parts = []
    for idx, (num, name) in enumerate(steps):
        parts.append(f"""
        <div class="lg-pipe-step">
            <span class="lg-pipe-num">{num}</span>
            <span>{name}</span>
        </div>
        """)
        if idx < len(steps) - 1:
            parts.append(f"""<span class="lg-pipe-arrow">&rarr;</span>""")

    return clean_html(f"""<div class="lg-pipeline">{''.join(parts)}</div>""")


def render_empty_state_html(title: str, message: str, icon_key: str = "activity") -> str:
    """Generate an elegant empty state component."""
    icon_svg = ICONS.get(icon_key, ICONS["activity"])
    return clean_html(f"""
    <div style="background-color: {BG_PANEL_ELEVATED}; border: 1px dashed {BORDER_SUBTLE}; border-radius: 8px; padding: 32px 20px; text-align: center; margin: 8px 0;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 44px; height: 44px; border-radius: 8px; background: rgba(255,255,255,0.04); margin-bottom: 10px;">
            {icon_svg}
        </div>
        <div style="font-size: 14px; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 4px;">{html.escape(title)}</div>
        <div style="font-size: 12px; color: {TEXT_SECONDARY}; max-width: 440px; margin: 0 auto; line-height: 1.4;">{html.escape(message)}</div>
    </div>
    """)


def render_live_telemetry_hud_html(
    status: str,
    explanation: str,
    method: str,
    trunk_inclination: Optional[float] = None,
    knee_angle: Optional[float] = None,
    is_valid_pose: bool = True,
    is_alert_triggered: bool = False,
    consecutive_unsafe: int = 0,
    fps: float = 0.0,
    latency_ms: float = 0.0,
) -> str:
    """Generate the right-column live assessment card HTML with zero code-block indentation."""
    # Determine Status Palette
    if "SAFE" in status.upper() and "UNSAFE" not in status.upper():
        card_border = ACCENT_MINT
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_color = ACCENT_MINT
        badge_label = "SAFE POSTURE"
        status_icon = ICONS["check_circle"]
    elif "UNSAFE" in status.upper():
        card_border = ACCENT_RED
        badge_bg = "rgba(239, 68, 68, 0.15)"
        badge_color = ACCENT_RED
        badge_label = "UNSAFE POSTURE"
        status_icon = ICONS["x_circle"]
    else:
        card_border = ACCENT_AMBER
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_color = ACCENT_AMBER
        badge_label = "INSUFFICIENT VISIBILITY"
        status_icon = ICONS["help_circle"]

    # Debounced Unsafe Alert Callout
    alert_banner = ""
    if is_alert_triggered:
        alert_banner = f"""
        <div style="background: rgba(239, 68, 68, 0.15); border: 1.5px solid {ACCENT_RED}; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; color: {ACCENT_RED}; font-size: 12px; font-weight: 700; text-transform: uppercase;">
                {ICONS['alert_triangle']} PERSISTENT UNSAFE POSTURE ALERT
            </div>
            <div style="font-size: 12px; color: #FCA5A5; line-height: 1.35; margin-top: 4px;">
                Stoop lifting detected across <b>{consecutive_unsafe}+ consecutive frames</b>. Bend deeply at the knees and maintain an upright trunk to avoid excessive L5/S1 spinal torque.
            </div>
        </div>
        """

    # Biomechanical Angle Metrics (Only rendered when valid measurements exist)
    if is_valid_pose and (trunk_inclination is not None or knee_angle is not None):
        t_val = f"{trunk_inclination:.1f}&deg;" if trunk_inclination is not None else "N/A"
        k_val = f"{knee_angle:.1f}&deg;" if knee_angle is not None else "N/A"
        t_color = ACCENT_MINT if (trunk_inclination is not None and trunk_inclination <= 45.0) else (ACCENT_RED if trunk_inclination is not None else TEXT_PRIMARY)
        k_color = ACCENT_MINT if (knee_angle is not None and knee_angle <= 125.0) else (ACCENT_AMBER if knee_angle is not None else TEXT_PRIMARY)

        measurements_grid = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
            <div style="background: #111821; border: 1px solid {BORDER_SUBTLE}; padding: 10px 12px; border-radius: 6px;">
                <div style="font-size: 10px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.05em;">Trunk Inclination</div>
                <div style="font-size: 20px; font-weight: 700; color: {t_color}; margin: 2px 0;">{t_val}</div>
                <div style="font-size: 10px; color: {TEXT_MUTED};">Safe: &le; 45&deg; from vertical</div>
            </div>
            <div style="background: #111821; border: 1px solid {BORDER_SUBTLE}; padding: 10px 12px; border-radius: 6px;">
                <div style="font-size: 10px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.05em;">Knee Flexion</div>
                <div style="font-size: 20px; font-weight: 700; color: {k_color}; margin: 2px 0;">{k_val}</div>
                <div style="font-size: 10px; color: {TEXT_MUTED};">Squat: &le; 125&deg; flexion</div>
            </div>
        </div>
        """
    else:
        measurements_grid = f"""
        <div style="background: #111821; border: 1px dashed {BORDER_SUBTLE}; padding: 12px; border-radius: 6px; margin-bottom: 12px; text-align: center;">
            <div style="font-size: 12px; color: {ACCENT_AMBER}; font-weight: 500;">
                Awaiting valid full-body keypoints to compute biomechanical angles.
            </div>
            <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 2px;">
                Ensure shoulders, hips, knees, and ankles are clearly visible in camera frame.
            </div>
        </div>
        """

    # Tracking & Quality Indicators
    tracking_str = "Full Body Detected (33/33)" if is_valid_pose else "Low Visibility / Occluded"
    tracking_color = ACCENT_MINT if is_valid_pose else ACCENT_AMBER
    fps_str = f"{fps:.0f} FPS ({latency_ms:.1f} ms)" if fps > 0 else f"{latency_ms:.1f} ms"

    return clean_html(f"""
    <div style="background-color: {BG_PANEL_ELEVATED}; border: 1.5px solid {card_border}; border-radius: 8px; padding: 18px 20px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: {TEXT_SECONDARY}; letter-spacing: 0.06em;">LIVE POSTURE ASSESSMENT</span>
            <span class="lg-badge" style="background: {badge_bg}; color: {badge_color}; border: 1px solid {card_border}; font-weight: 700; font-size: 12px;">
                {status_icon} {badge_label}
            </span>
        </div>

        {alert_banner}

        <div style="font-size: 13px; color: {TEXT_PRIMARY}; line-height: 1.45; margin-bottom: 12px; padding: 10px 12px; background: #111821; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px;">
            <b style="color: {TEXT_SECONDARY};">Ergonomic Rationale:</b><br>
            {html.escape(explanation)}
        </div>

        {measurements_grid}

        <div style="border-top: 1px solid {BORDER_SUBTLE}; padding-top: 10px; font-size: 11px; color: {TEXT_SECONDARY}; line-height: 1.6;">
            <div>&bull; <b>Assessment Engine:</b> <span style="color: {TEXT_PRIMARY};">{html.escape(method)}</span></div>
            <div>&bull; <b>Pose Tracking Quality:</b> <span style="color: {tracking_color}; font-weight: 600;">{tracking_str}</span></div>
            <div>&bull; <b>Inference Latency:</b> <span style="color: {TEXT_PRIMARY};">{fps_str}</span></div>
        </div>
    </div>
    """)


def render_uploaded_video_summary_html(summary: Any) -> str:
    """Generate the overall result card for processed video analysis with zero code-block indentation."""
    verdict = summary.overall_classification
    if "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper():
        v_color = ACCENT_MINT
        v_bg = "rgba(16, 185, 129, 0.15)"
        v_border = ACCENT_MINT
        v_icon = ICONS["check_circle"]
    elif "UNSAFE" in verdict.upper():
        v_color = ACCENT_RED
        v_bg = "rgba(239, 68, 68, 0.15)"
        v_border = ACCENT_RED
        v_icon = ICONS["x_circle"]
    else:
        v_color = ACCENT_AMBER
        v_bg = "rgba(245, 158, 11, 0.15)"
        v_border = ACCENT_AMBER
        v_icon = ICONS["help_circle"]

    mean_t_str = f"{summary.mean_trunk_inclination_deg:.1f}&deg;" if summary.mean_trunk_inclination_deg is not None else "N/A"
    max_t_str = f"{summary.max_trunk_inclination_deg:.1f}&deg;" if summary.max_trunk_inclination_deg is not None else "N/A"
    mean_k_str = f"{summary.mean_knee_angle_deg:.1f}&deg;" if summary.mean_knee_angle_deg is not None else "N/A"
    min_k_str = f"{summary.min_knee_angle_deg:.1f}&deg;" if summary.min_knee_angle_deg is not None else "N/A"

    return clean_html(f"""
    <div style="background-color: {BG_PANEL_ELEVATED}; border: 1.5px solid {v_border}; border-radius: 8px; padding: 18px 20px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: {TEXT_SECONDARY}; letter-spacing: 0.06em;">SCREENING VERDICT</span>
            <span class="lg-badge" style="background-color: {v_bg}; color: {v_color}; border: 1px solid {v_border}; font-size: 12px; font-weight: 700;">
                {v_icon} {html.escape(verdict)}
            </span>
        </div>
        <div style="font-size: 13px; color: {TEXT_PRIMARY}; font-weight: 500; line-height: 1.45; margin-bottom: 12px; padding: 10px 12px; background: #111821; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px;">
            <b style="color: {TEXT_SECONDARY};">Primary Ergonomic Rationale:</b><br>
            {html.escape(summary.primary_reason)}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px;">
            <div style="background: #111821; border: 1px solid {BORDER_SUBTLE}; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="font-size: 10px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase;">Safe Poses</div>
                <div style="font-size: 18px; font-weight: 700; color: {ACCENT_MINT}; margin-top: 2px;">{summary.safe_pct_of_valid}%</div>
            </div>
            <div style="background: #111821; border: 1px solid {BORDER_SUBTLE}; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="font-size: 10px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase;">Unsafe Poses</div>
                <div style="font-size: 18px; font-weight: 700; color: {ACCENT_RED}; margin-top: 2px;">{summary.unsafe_pct_of_valid}%</div>
            </div>
            <div style="background: #111821; border: 1px solid {BORDER_SUBTLE}; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="font-size: 10px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase;">Valid Tracking</div>
                <div style="font-size: 18px; font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 2px;">{summary.valid_pose_pct}%</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; color: {TEXT_SECONDARY}; padding: 8px 12px; background: #111821; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px; line-height: 1.6;">
            <div>&bull; <b>Mean Trunk Flexion:</b> <span style="color: {TEXT_PRIMARY};">{mean_t_str}</span></div>
            <div>&bull; <b>Max Trunk Flexion:</b> <span style="color: {TEXT_PRIMARY};">{max_t_str}</span></div>
            <div>&bull; <b>Mean Knee Flexion:</b> <span style="color: {TEXT_PRIMARY};">{mean_k_str}</span></div>
            <div>&bull; <b>Min Knee Flexion:</b> <span style="color: {TEXT_PRIMARY};">{min_k_str}</span></div>
        </div>
    </div>
    """)

