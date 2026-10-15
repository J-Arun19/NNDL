"""UI Theme and Visual Components for LiftGuard AI.

Provides a polished, restrained industrial computer-vision design system,
neutral color palette with strict ergonomic safety indicators, and clean layout components.
"""

from __future__ import annotations

import html
import textwrap
from typing import Any, Dict, Optional

# Palette Tokens - Polished Industrial Inspection Aesthetic
BG_MAIN = "#F8FAFC"             # Crisp off-white canvas (Slate-50)
BG_PANEL = "#FFFFFF"            # White cards and panels
BG_PANEL_ALT = "#F1F5F9"        # Light neutral secondary surface (Slate-100)
BG_PANEL_HOVER = "#E2E8F0"      # Subtle hover surface (Slate-200)

BORDER_SUBTLE = "#E2E8F0"       # Thin neutral border (Slate-200)
BORDER_STRONG = "#CBD5E1"       # Defined neutral border (Slate-300)
BORDER_FOCUS = "#0E7490"        # Restrained deep teal focus accent (Teal-700)

# Functional Accent Tokens
ACCENT_TEAL = "#0E7490"         # Primary industrial teal for interactive elements
ACCENT_TEAL_LIGHT = "#F0FDFA"   # Subtle teal wash for active navigation
ACCENT_TEAL_BORDER = "#99F6E4"  # Subtle teal border

# Strict Ergonomic Safety Color Tokens
# GREEN: STRICTLY for SAFE POSTURE only
STATUS_SAFE = "#16A34A"         # Emerald-600
STATUS_SAFE_BG = "#F0FDF4"      # Emerald-50
STATUS_SAFE_BORDER = "#BBF7D0"  # Emerald-200

# RED: STRICTLY for UNSAFE POSTURE only
STATUS_UNSAFE = "#DC2626"       # Red-600
STATUS_UNSAFE_BG = "#FEF2F2"    # Red-50
STATUS_UNSAFE_BORDER = "#FECACA" # Red-200

# AMBER: STRICTLY for INSUFFICIENT VISIBILITY / WARNING
STATUS_WARN = "#D97706"         # Amber-600
STATUS_WARN_BG = "#FFFBEB"      # Amber-50
STATUS_WARN_BORDER = "#FDE68A"  # Amber-200

# Typography Tokens
TEXT_PRIMARY = "#0F172A"        # Deep charcoal headers & emphasis (Slate-900)
TEXT_SECONDARY = "#475569"      # Readable neutral slate body (Slate-600)
TEXT_MUTED = "#64748B"          # Subtle metadata / captions (Slate-500)

# Backward-compatibility alias tokens
ACCENT_MINT = STATUS_SAFE
ACCENT_RED = STATUS_UNSAFE
ACCENT_AMBER = STATUS_WARN
BG_PANEL_ELEVATED = BG_PANEL


def clean_html(raw_html: str) -> str:
    """Safely format and sanitize HTML for Streamlit markdown rendering.
    
    Strips leading and trailing whitespace from every line and drops blank lines,
    joining into a continuous HTML string. This completely eliminates the
    possibility of CommonMark/Markdown engines misinterpreting indented HTML
    tags (4+ spaces after blank lines) as <pre><code> blocks.
    """
    lines = [line.strip() for line in raw_html.strip().splitlines() if line.strip()]
    return "".join(lines)


# Crisp, Restrained Industrial SVG Icons (Neutral Slate / Functional Colors)
ICONS = {
    "shield": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0E7490" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>""",
    "shield_check": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>""",
    "alert_triangle": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>""",
    "help_circle": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>""",
    "video": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>""",
    "camera": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>""",
    "activity": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0E7490" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>""",
    "database": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>""",
    "file_text": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>""",
    "cpu": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>""",
    "check_circle": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>""",
    "x_circle": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>""",
    "layers": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>""",
    "clipboard_check": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect><path d="m9 14 2 2 4-4"></path></svg>""",
}


def get_css_theme() -> str:
    """Return the scoped, restrained CSS injection for Streamlit."""
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Base Typography and Surface Resets */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            letter-spacing: -0.005em;
            color: {TEXT_PRIMARY};
        }}

        /* Neutral Light App Canvas */
        .stApp {{
            background-color: {BG_MAIN} !important;
            color: {TEXT_PRIMARY} !important;
        }}

        /* Subtle Header Bar */
        header[data-testid="stHeader"] {{
            background-color: rgba(248, 250, 252, 0.94) !important;
            backdrop-filter: blur(6px);
            border-bottom: 1px solid {BORDER_SUBTLE} !important;
        }}

        /* Streamlit Main Content Block Padding - Comfortable spacing below fixed toolbar */
        .block-container {{
            padding-top: 4.75rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1280px !important;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {BG_PANEL} !important;
            border-right: 1px solid {BORDER_SUBTLE} !important;
            min-width: 220px !important;
            max-width: 250px !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding: 1.25rem 0.85rem !important;
        }}

        /* Compact Navigation Links in Sidebar */
        [data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
            display: none !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {{
            background-color: transparent !important;
            border: none !important;
            border-left: 3px solid transparent !important;
            border-radius: 4px !important;
            padding: 7px 11px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            color: {TEXT_SECONDARY} !important;
            transition: all 0.12s ease !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
            background-color: {BG_PANEL_ALT} !important;
            color: {TEXT_PRIMARY} !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {{
            background-color: {ACCENT_TEAL_LIGHT} !important;
            border-left: 3px solid {ACCENT_TEAL} !important;
            color: {ACCENT_TEAL} !important;
            font-weight: 600 !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}

        /* Horizontal Mode Selector (Segmented Bar) */
        div[data-testid="stRadio"]:has(input[value="Upload Video"]) div[role="radiogroup"],
        div[data-testid="stRadio"]:has(input[value="Benchmark Audit"]) div[role="radiogroup"] {{
            display: inline-flex !important;
            flex-direction: row !important;
            background-color: {BG_PANEL_ALT} !important;
            padding: 3px !important;
            border-radius: 6px !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            gap: 3px !important;
            margin-bottom: 0.75rem !important;
        }}
        div[data-testid="stRadio"]:has(input[value="Upload Video"]) div[role="radiogroup"] > label,
        div[data-testid="stRadio"]:has(input[value="Benchmark Audit"]) div[role="radiogroup"] > label {{
            background-color: transparent !important;
            border: none !important;
            padding: 6px 16px !important;
            border-radius: 4px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            color: {TEXT_SECONDARY} !important;
            cursor: pointer !important;
            margin: 0 !important;
            transition: all 0.12s ease !important;
            box-shadow: none !important;
        }}
        div[data-testid="stRadio"]:has(input[value="Upload Video"]) div[role="radiogroup"] > label:hover,
        div[data-testid="stRadio"]:has(input[value="Benchmark Audit"]) div[role="radiogroup"] > label:hover {{
            color: {TEXT_PRIMARY} !important;
        }}
        div[data-testid="stRadio"]:has(input[value="Upload Video"]) div[role="radiogroup"] > label:has(input:checked),
        div[data-testid="stRadio"]:has(input[value="Benchmark Audit"]) div[role="radiogroup"] > label:has(input:checked) {{
            background-color: {BG_PANEL} !important;
            color: {ACCENT_TEAL} !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        }}
        div[data-testid="stRadio"]:has(input[value="Upload Video"]) div[role="radiogroup"] > label > div:first-child,
        div[data-testid="stRadio"]:has(input[value="Benchmark Audit"]) div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}

        /* Restrained Industrial Buttons */
        button[data-testid="baseButton-primary"],
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {{
            background-color: {ACCENT_TEAL} !important;
            background-image: none !important;
            color: #FFFFFF !important;
            border: 1px solid {ACCENT_TEAL} !important;
            border-radius: 5px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            padding: 6px 14px !important;
            letter-spacing: -0.005em !important;
            transition: all 0.12s ease !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        }}
        button[data-testid="baseButton-primary"]:hover,
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {{
            background-color: #0C647D !important;
            border-color: #0C647D !important;
            color: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }}
        button[data-testid="baseButton-primary"]:active,
        .stButton > button[kind="primary"]:active {{
            background-color: #084D61 !important;
            transform: translateY(1px);
        }}
        button[data-testid="baseButton-primary"]:disabled,
        .stButton > button[kind="primary"]:disabled {{
            background-color: {BG_PANEL_ALT} !important;
            border-color: {BORDER_SUBTLE} !important;
            color: {TEXT_MUTED} !important;
            cursor: not-allowed !important;
            box-shadow: none !important;
        }}

        button[data-testid="baseButton-secondary"],
        .stButton > button[kind="secondary"],
        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button {{
            background-color: {BG_PANEL} !important;
            background-image: none !important;
            border: 1px solid {BORDER_STRONG} !important;
            color: {TEXT_PRIMARY} !important;
            border-radius: 5px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            padding: 6px 14px !important;
            transition: all 0.12s ease !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        }}
        button[data-testid="baseButton-secondary"]:hover,
        .stButton > button[kind="secondary"]:hover,
        .stButton > button:not([kind="primary"]):hover,
        .stDownloadButton > button:hover {{
            background-color: {BG_PANEL_ALT} !important;
            border-color: #94A3B8 !important;
            color: {TEXT_PRIMARY} !important;
        }}
        button[data-testid="baseButton-secondary"]:active,
        .stButton > button[kind="secondary"]:active,
        .stDownloadButton > button:active {{
            background-color: {BG_PANEL_HOVER} !important;
            transform: translateY(1px);
        }}

        /* Compact Native Tabs */
        div[data-baseweb="tab-list"] {{
            background-color: transparent !important;
            border-bottom: 1px solid {BORDER_SUBTLE} !important;
            gap: 12px;
            margin-bottom: 1rem;
        }}
        button[data-baseweb="tab"] {{
            background-color: transparent !important;
            color: {TEXT_SECONDARY} !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            padding: 8px 12px !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
        }}
        button[data-baseweb="tab"]:hover {{
            color: {TEXT_PRIMARY} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {ACCENT_TEAL} !important;
            font-weight: 600 !important;
            border-bottom: 2px solid {ACCENT_TEAL} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {ACCENT_TEAL} !important;
        }}

        /* Native File Uploader - Crisp Light Surface */
        div[data-testid="stFileUploader"] {{
            background: transparent !important;
        }}
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {{
            background-color: {BG_PANEL} !important;
            border: 1px dashed {BORDER_STRONG} !important;
            border-radius: 6px !important;
            padding: 1rem !important;
            color: {TEXT_PRIMARY} !important;
        }}
        div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {{
            border-color: {ACCENT_TEAL} !important;
            background-color: {ACCENT_TEAL_LIGHT} !important;
        }}

        /* Inputs, Sliders, and Selectboxes */
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-baseweb="select"] > div {{
            background-color: {BG_PANEL} !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid {BORDER_STRONG} !important;
            border-radius: 5px !important;
            font-size: 13px !important;
        }}
        div[data-testid="stSlider"] div[role="slider"] {{
            background-color: {ACCENT_TEAL} !important;
            border-color: {ACCENT_TEAL} !important;
            box-shadow: none !important;
        }}
        div[data-baseweb="slider"] > div:nth-child(1) > div:nth-child(1) {{
            background-color: {ACCENT_TEAL} !important;
        }}

        /* Industrial Panels and Cards */
        .lg-card {{
            background-color: {BG_PANEL};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 6px;
            padding: 14px 16px;
            margin-bottom: 14px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        }}
        .lg-kpi-card {{
            background-color: {BG_PANEL};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 6px;
            padding: 11px 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 84px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        }}
        .lg-kpi-label {{
            font-size: 11px;
            font-weight: 600;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 2px;
        }}
        .lg-kpi-val {{
            font-size: 20px;
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

        /* Subtle Functional Badges */
        .lg-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.01em;
            line-height: 1.4;
        }}
        .lg-badge-safe {{
            background-color: {STATUS_SAFE_BG};
            color: {STATUS_SAFE};
            border: 1px solid {STATUS_SAFE_BORDER};
        }}
        .lg-badge-unsafe {{
            background-color: {STATUS_UNSAFE_BG};
            color: {STATUS_UNSAFE};
            border: 1px solid {STATUS_UNSAFE_BORDER};
        }}
        .lg-badge-warn {{
            background-color: {STATUS_WARN_BG};
            color: {STATUS_WARN};
            border: 1px solid {STATUS_WARN_BORDER};
        }}
        .lg-badge-neutral {{
            background-color: {BG_PANEL_ALT};
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER_SUBTLE};
        }}

        /* Typography Hierarchy */
        .lg-title {{
            font-size: 20px;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            letter-spacing: -0.01em;
            margin-top: 0 !important;
            margin-bottom: 2px;
            line-height: 1.25;
        }}
        .lg-subtitle {{
            font-size: 13px;
            color: {TEXT_SECONDARY};
            margin-bottom: 14px;
            line-height: 1.4;
        }}
        .lg-section-header {{
            font-size: 13px;
            font-weight: 600;
            color: {TEXT_PRIMARY};
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin: 12px 0 8px 0;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* Clean Scrollbars */
        ::-webkit-scrollbar {{
            width: 5px;
            height: 5px;
        }}
        ::-webkit-scrollbar-track {{
            background: {BG_MAIN};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {BORDER_STRONG};
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #94A3B8;
        }}
    </style>
    """


def render_sidebar_brand() -> str:
    """Return HTML for the compact brand header in sidebar."""
    return clean_html(f"""
    <div style="padding: 2px 0 14px 0; border-bottom: 1px solid {BORDER_SUBTLE}; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 26px; height: 26px; border-radius: 5px; background: {ACCENT_TEAL_LIGHT}; border: 1px solid {ACCENT_TEAL_BORDER}; display: flex; align-items: center; justify-content: center;">
                {ICONS['shield']}
            </div>
            <div>
                <div style="font-size: 14px; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: -0.01em; line-height: 1.1;">LiftGuard AI</div>
                <div style="font-size: 10px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em;">Industrial Inspection</div>
            </div>
        </div>
    </div>
    """)


def render_sidebar_status_footer(readiness: dict, threshold: float = 0.5) -> str:
    """Return compact, restrained system info for sidebar footer without alarmist badges."""
    screening_method = html.escape(str(readiness.get("screening_method", "Rule-Based Ergonomic Assessment")))

    return clean_html(f"""
    <div style="margin-top: 24px; padding: 10px 12px; background: {BG_PANEL_ALT}; border: 1px solid {BORDER_SUBTLE}; border-radius: 5px; font-size: 11px;">
        <div style="font-size: 10px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px;">Screening Engine</div>
        <div style="font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 4px; line-height: 1.3;">{screening_method}</div>
        <div style="font-size: 10px; color: {TEXT_MUTED}; line-height: 1.4; border-top: 1px solid {BORDER_SUBTLE}; padding-top: 4px;">
            Landmarks: MediaPipe 33-joint<br>
            Criteria: NIOSH / OSHA biomechanics
        </div>
    </div>
    """)


def render_kpi_card_html(label: str, value: str, sublabel: str, accent: str = "default") -> str:
    """Generate a clean compact industrial KPI card HTML."""
    val_color = TEXT_PRIMARY
    if accent == "success":
        val_color = STATUS_SAFE
    elif accent == "danger":
        val_color = STATUS_UNSAFE
    elif accent == "warning":
        val_color = STATUS_WARN
    elif accent == "teal":
        val_color = ACCENT_TEAL

    return clean_html(f"""
    <div class="lg-kpi-card">
        <div class="lg-kpi-label">{html.escape(label)}</div>
        <div class="lg-kpi-val" style="color: {val_color};">{html.escape(value)}</div>
        <div class="lg-kpi-sub">{html.escape(sublabel)}</div>
    </div>
    """)


def render_workflow_pipeline_html() -> str:
    """Generate the compact horizontal workflow pipeline stepper in clean neutral styling."""
    steps = [
        ("1", "Video Input"),
        ("2", "Pose Estimation"),
        ("3", "Quality Gate"),
        ("4", "Biomechanical Rules"),
        ("5", "Screening Verdict")
    ]
    parts = []
    for idx, (num, name) in enumerate(steps):
        parts.append(f"""
        <div style="display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500; color: {TEXT_PRIMARY}; white-space: nowrap;">
            <span style="display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; border-radius: 50%; background: {ACCENT_TEAL_LIGHT}; color: {ACCENT_TEAL}; border: 1px solid {ACCENT_TEAL_BORDER}; font-size: 10px; font-weight: 700;">{num}</span>
            <span>{name}</span>
        </div>
        """)
        if idx < len(steps) - 1:
            parts.append(f"""<span style="color: {TEXT_MUTED}; font-size: 12px; margin: 0 2px;">&rarr;</span>""")

    return clean_html(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background-color: {BG_PANEL}; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px; padding: 10px 14px; margin: 8px 0 16px 0; overflow-x: auto;">
        {''.join(parts)}
    </div>
    """)


def render_empty_state_html(title: str, message: str, icon_key: str = "activity") -> str:
    """Generate an elegant, compact empty state component."""
    icon_svg = ICONS.get(icon_key, ICONS["activity"])
    return clean_html(f"""
    <div style="background-color: {BG_PANEL}; border: 1px dashed {BORDER_STRONG}; border-radius: 6px; padding: 24px 16px; text-align: center; margin: 6px 0;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 6px; background: {BG_PANEL_ALT}; margin-bottom: 8px;">
            {icon_svg}
        </div>
        <div style="font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 2px;">{html.escape(title)}</div>
        <div style="font-size: 12px; color: {TEXT_SECONDARY}; max-width: 420px; margin: 0 auto; line-height: 1.4;">{html.escape(message)}</div>
    </div>
    """)


def render_uploaded_video_summary_html(summary: Any) -> str:
    """Generate the concise, professional result card for processed video analysis."""
    verdict = summary.overall_classification
    if "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper():
        badge_class = "lg-badge-safe"
        badge_icon = ICONS["check_circle"]
        card_border = STATUS_SAFE_BORDER
        card_banner_bg = STATUS_SAFE_BG
        card_banner_color = STATUS_SAFE
        label_text = "SAFE POSTURE"
    elif "UNSAFE" in verdict.upper():
        badge_class = "lg-badge-unsafe"
        badge_icon = ICONS["x_circle"]
        card_border = STATUS_UNSAFE_BORDER
        card_banner_bg = STATUS_UNSAFE_BG
        card_banner_color = STATUS_UNSAFE
        label_text = "UNSAFE POSTURE"
    else:
        badge_class = "lg-badge-warn"
        badge_icon = ICONS["help_circle"]
        card_border = STATUS_WARN_BORDER
        card_banner_bg = STATUS_WARN_BG
        card_banner_color = STATUS_WARN
        label_text = "INSUFFICIENT VISIBILITY"

    mean_t_str = f"{summary.mean_trunk_inclination_deg:.1f}&deg;" if summary.mean_trunk_inclination_deg is not None else "N/A"
    max_t_str = f"{summary.max_trunk_inclination_deg:.1f}&deg;" if summary.max_trunk_inclination_deg is not None else "N/A"
    mean_k_str = f"{summary.mean_knee_angle_deg:.1f}&deg;" if summary.mean_knee_angle_deg is not None else "N/A"
    min_k_str = f"{summary.min_knee_angle_deg:.1f}&deg;" if summary.min_knee_angle_deg is not None else "N/A"

    parts = [
        f'<div style="background-color:{BG_PANEL};border:1px solid {card_border};border-radius:6px;overflow:hidden;margin-bottom:12px;box-shadow:0 1px 2px rgba(0,0,0,0.03);">',
        f'<div style="background:{card_banner_bg};padding:10px 14px;border-bottom:1px solid {card_border};display:flex;justify-content:space-between;align-items:center;">',
        f'<div style="display:flex;align-items:center;gap:7px;">{badge_icon}<span style="font-size:13px;font-weight:700;color:{card_banner_color};letter-spacing:0.02em;">{label_text}</span></div>',
        f'<span class="lg-badge {badge_class}">Overall Verdict</span>',
        f'</div>',
        f'<div style="padding:12px 14px;">',
        f'<div style="font-size:12px;color:{TEXT_PRIMARY};line-height:1.45;margin-bottom:12px;padding:8px 10px;background:{BG_PANEL_ALT};border-radius:4px;border-left:3px solid {card_banner_color};">{html.escape(summary.primary_reason)}</div>',
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">',
        f'<div style="background:{BG_PANEL_ALT};border:1px solid {BORDER_SUBTLE};padding:8px 10px;border-radius:4px;">',
        f'<div style="font-size:10px;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;">Trunk Inclination</div>',
        f'<div style="font-size:15px;font-weight:700;color:{TEXT_PRIMARY};margin:1px 0;">{mean_t_str} <span style="font-size:11px;font-weight:500;color:{TEXT_MUTED};">(Max: {max_t_str})</span></div>',
        f'<div style="font-size:10px;color:{TEXT_MUTED};">Safe limit: &le; 45&deg; from vertical</div>',
        f'</div>',
        f'<div style="background:{BG_PANEL_ALT};border:1px solid {BORDER_SUBTLE};padding:8px 10px;border-radius:4px;">',
        f'<div style="font-size:10px;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;">Knee Flexion</div>',
        f'<div style="font-size:15px;font-weight:700;color:{TEXT_PRIMARY};margin:1px 0;">{mean_k_str} <span style="font-size:11px;font-weight:500;color:{TEXT_MUTED};">(Min: {min_k_str})</span></div>',
        f'<div style="font-size:10px;color:{TEXT_MUTED};">Squat lift: &le; 125&deg; flexion</div>',
        f'</div>',
        f'</div>',
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:10px;text-align:center;">',
        f'<div style="background:{BG_PANEL_ALT};border:1px solid {BORDER_SUBTLE};padding:6px;border-radius:4px;"><div style="font-size:10px;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;">Valid Pose</div><div style="font-size:14px;font-weight:700;color:{TEXT_PRIMARY};">{summary.valid_pose_pct}%</div></div>',
        f'<div style="background:{BG_PANEL_ALT};border:1px solid {BORDER_SUBTLE};padding:6px;border-radius:4px;"><div style="font-size:10px;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;">Safe Frames</div><div style="font-size:14px;font-weight:700;color:{STATUS_SAFE};">{summary.safe_pct_of_valid}%</div></div>',
        f'<div style="background:{BG_PANEL_ALT};border:1px solid {BORDER_SUBTLE};padding:6px;border-radius:4px;"><div style="font-size:10px;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;">Unsafe Frames</div><div style="font-size:14px;font-weight:700;color:{STATUS_UNSAFE};">{summary.unsafe_pct_of_valid}%</div></div>',
        f'</div>',
        f'<div style="border-top:1px solid {BORDER_SUBTLE};padding-top:8px;font-size:11px;color:{TEXT_MUTED};">Assessment method: Rule-based ergonomic assessment</div>',
        f'</div>',
        f'</div>',
    ]
    return "".join(parts)


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
    if "SAFE" in status.upper() and "UNSAFE" not in status.upper():
        badge_class = "lg-badge-safe"
        badge_icon = ICONS["check_circle"]
        card_border = STATUS_SAFE_BORDER
        card_banner_bg = STATUS_SAFE_BG
        card_banner_color = STATUS_SAFE
        label_text = "SAFE POSTURE"
    elif "UNSAFE" in status.upper():
        badge_class = "lg-badge-unsafe"
        badge_icon = ICONS["x_circle"]
        card_border = STATUS_UNSAFE_BORDER
        card_banner_bg = STATUS_UNSAFE_BG
        card_banner_color = STATUS_UNSAFE
        label_text = "UNSAFE POSTURE"
    else:
        badge_class = "lg-badge-warn"
        badge_icon = ICONS["help_circle"]
        card_border = STATUS_WARN_BORDER
        card_banner_bg = STATUS_WARN_BG
        card_banner_color = STATUS_WARN
        label_text = "INSUFFICIENT VISIBILITY"

    alert_banner = ""
    if is_alert_triggered:
        alert_banner = (
            f'<div style="background:{STATUS_UNSAFE_BG};border:1px solid {STATUS_UNSAFE_BORDER};border-radius:4px;padding:7px 10px;margin-bottom:10px;">'
            f'<div style="display:flex;align-items:center;gap:5px;color:{STATUS_UNSAFE};font-size:11px;font-weight:700;">'
            f'{ICONS["alert_triangle"]} SUSTAINED STOOP POSTURE ALERT ({consecutive_unsafe}+ frames)'
            f'</div>'
            f'<div style="font-size:11px;color:{TEXT_SECONDARY};margin-top:2px;">'
            f'Excessive forward trunk flexion detected. Bend knees and lift with legs.'
            f'</div>'
            f'</div>'
        )

    vis_warning = ""
    if not is_valid_pose:
        vis_warning = (
            f'<div style="background:{STATUS_WARN_BG};border:1px solid {STATUS_WARN_BORDER};border-radius:4px;padding:7px 10px;margin-bottom:10px;">'
            f'<div style="display:flex;align-items:center;gap:5px;color:{STATUS_WARN};font-size:11px;font-weight:600;">'
            f'{ICONS["help_circle"]} Insufficient pose visibility'
            f'</div>'
            f'<div style="font-size:11px;color:{TEXT_SECONDARY};margin-top:2px;">'
            f'Ensure shoulders, hips, knees, and ankles are clearly visible in the camera frame.'
            f'</div>'
            f'</div>'
        )

    if is_valid_pose and (trunk_inclination is not None or knee_angle is not None):
        t_val = f"{trunk_inclination:.1f}&deg;" if trunk_inclination is not None else "N/A"
        k_val = f"{knee_angle:.1f}&deg;" if knee_angle is not None else "N/A"
        t_color = STATUS_SAFE if (trunk_inclination is not None and trunk_inclination <= 45.0) else (STATUS_UNSAFE if trunk_inclination is not None else TEXT_PRIMARY)
        k_color = STATUS_SAFE if (knee_angle is not None and knee_angle <= 125.0) else (STATUS_WARN if knee_angle is not None else TEXT_PRIMARY)

        measurements_grid = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">'
            f'<div style="background:{BG_PANEL_ALT};border:1px solid {BORDER_SUBTLE};padding:8px 10px;border-radius:4px;">'
            f'<div style="font-size:10px;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;">Trunk Inclination</div>'
            f'<div style="font-size:18px;font-weight:700;color:{t_color};margin:1px 0;">{t_val}</div>'
            f'<div style="font-size:10px;color:{TEXT_MUTED};">Safe: &le; 45&deg; from vertical</div>'
            f'</div>'
            f'<div style="background:{BG_PANEL_ALT};border:1px solid {BORDER_SUBTLE};padding:8px 10px;border-radius:4px;">'
            f'<div style="font-size:10px;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;">Knee Flexion</div>'
            f'<div style="font-size:18px;font-weight:700;color:{k_color};margin:1px 0;">{k_val}</div>'
            f'<div style="font-size:10px;color:{TEXT_MUTED};">Squat: &le; 125&deg; flexion</div>'
            f'</div>'
            f'</div>'
        )
    else:
        measurements_grid = (
            f'<div style="background:{BG_PANEL_ALT};border:1px dashed {BORDER_SUBTLE};padding:10px;border-radius:4px;margin-bottom:10px;text-align:center;">'
            f'<div style="font-size:11px;color:{TEXT_MUTED};">Awaiting full-body landmark visibility to compute angles.</div>'
            f'</div>'
        )

    perf_str = f"{fps:.0f} FPS ({latency_ms:.1f} ms)" if fps > 0 else f"{latency_ms:.1f} ms"
    tracking_str = "33/33 Tracked" if is_valid_pose else "Low visibility"

    parts = [
        f'<div style="background-color:{BG_PANEL};border:1px solid {card_border};border-radius:6px;overflow:hidden;margin-bottom:12px;box-shadow:0 1px 2px rgba(0,0,0,0.03);">',
        f'<div style="background:{card_banner_bg};padding:9px 13px;border-bottom:1px solid {card_border};display:flex;justify-content:space-between;align-items:center;">',
        f'<div style="display:flex;align-items:center;gap:6px;">{badge_icon}<span style="font-size:13px;font-weight:700;color:{card_banner_color};letter-spacing:0.02em;">{label_text}</span></div>',
        f'<span class="lg-badge {badge_class}">Live Status</span>',
        f'</div>',
        f'<div style="padding:12px 14px;">',
        alert_banner,
        vis_warning,
        f'<div style="font-size:12px;color:{TEXT_PRIMARY};line-height:1.45;margin-bottom:10px;padding:7px 10px;background:{BG_PANEL_ALT};border-radius:4px;">{html.escape(explanation)}</div>',
        measurements_grid,
        f'<div style="border-top:1px solid {BORDER_SUBTLE};padding-top:8px;font-size:11px;color:{TEXT_MUTED};line-height:1.5;">',
        f'<div>Assessment method: Rule-based ergonomic assessment</div>',
        f'<div>Latency: {perf_str} | Joints: {tracking_str}</div>',
        f'</div>',
        f'</div>',
        f'</div>',
    ]
    return "".join(p for p in parts if p)
