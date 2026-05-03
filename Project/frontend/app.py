import streamlit as st
from pathlib import Path
import sys

# Make backend importable
backend_path = Path(__file__).parent.parent / "backend" / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# App configuration
st.set_page_config(
    page_title="SafetyBoundary — Subtle Failure Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── GLOBAL STYLING ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 16px;
    border-radius: 10px;
    transition: all 0.3s ease;
    margin-bottom: 4px;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99, 102, 241, 0.15);
}

/* ── Cards / Containers ── */
[data-testid="stExpander"],
.stAlert, .element-container .stMarkdown {
    border-radius: 14px !important;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 18px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.18);
}
div[data-testid="stMetric"] label {
    color: #a5b4fc !important;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.8rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.45) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 24px;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.6);
}

/* ── Progress bars ── */
.stProgress > div > div {
    border-radius: 999px;
}

/* ── Text colors ── */
h1, h2, h3, h4 { color: #f1f5f9 !important; }
p, span, li, td, th { color: #cbd5e1 !important; }
.stMarkdown a { color: #818cf8 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 10px 20px;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
    border-color: rgba(99,102,241,0.4) !important;
}

/* ── Glass card helper ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 28px;
    margin: 10px 0;
}
.glass-card-accent {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.06));
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 18px;
    padding: 28px;
    margin: 10px 0;
}

/* ── Status badges ── */
.badge-safe {
    display: inline-block;
    background: linear-gradient(135deg, #059669, #10b981);
    color: white !important;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}
.badge-subtle {
    display: inline-block;
    background: linear-gradient(135deg, #d97706, #f59e0b);
    color: white !important;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}
.badge-obvious {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white !important;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}

/* ── Divider ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 32px 0;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: #6366f1 !important;
}

/* ── Selectbox / Slider ── */
.stSelectbox, .stSlider {
    color: #e0e0e0 !important;
}

/* Hide default streamlit hamburger & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ── Animated gradient text ── */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.gradient-text {
    background: linear-gradient(270deg, #6366f1, #8b5cf6, #a78bfa, #6366f1);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s ease infinite;
}

/* Pulse glow for status indicator */
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 4px rgba(16,185,129,0.4); }
    50% { box-shadow: 0 0 12px rgba(16,185,129,0.8); }
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 8px;
}
.status-dot.online {
    background: #10b981;
    animation: pulseGlow 2s ease-in-out infinite;
}
.status-dot.offline {
    background: #ef4444;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────── SIDEBAR ────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.8rem;">🛡️</div>
        <h2 style="margin:8px 0 2px 0; font-weight:700;" class="gradient-text">SafetyBoundary</h2>
        <p style="font-size:0.82rem; color:#94a3b8 !important; margin:0;">Subtle Failure Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API status check
    api_online = False
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            api_online = True
    except:
        pass

    if api_online:
        st.markdown("""
        <div style="display:flex;align-items:center;padding:8px 12px;background:rgba(16,185,129,0.08);border-radius:10px;border:1px solid rgba(16,185,129,0.2);">
            <span class="status-dot online"></span>
            <span style="color:#10b981 !important;font-weight:500;font-size:0.85rem;">API Connected</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;padding:8px 12px;background:rgba(239,68,68,0.08);border-radius:10px;border:1px solid rgba(239,68,68,0.2);">
            <span class="status-dot offline"></span>
            <span style="color:#ef4444 !important;font-weight:500;font-size:0.85rem;">API Offline</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    pages = st.radio(
        "NAVIGATION",
        ("🏠  Dashboard", "🔬  Analyze Image", "⚖️  Compare Thresholds", "📊  Evaluation"),
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model info
    st.markdown("""
    <div class="glass-card" style="padding:16px;">
        <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#64748b !important;margin-bottom:8px;">Model Info</p>
        <p style="font-size:0.85rem;margin:4px 0;"><strong style="color:#a5b4fc !important;">Backbone:</strong> YOLOv8-nano</p>
        <p style="font-size:0.85rem;margin:4px 0;"><strong style="color:#a5b4fc !important;">Classes:</strong> 3 (safe · subtle · obvious)</p>
        <p style="font-size:0.85rem;margin:4px 0;"><strong style="color:#a5b4fc !important;">Input:</strong> 320 × 320 px</p>
        <p style="font-size:0.85rem;margin:4px 0;"><strong style="color:#a5b4fc !important;">Training:</strong> 100 images · 5 epochs</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────── PAGES ────────────────────────
if pages == "🏠  Dashboard":
    from pages.safety_overview import render_dashboard
    render_dashboard()
elif pages == "🔬  Analyze Image":
    from pages.safety_overview import render_analyze
    render_analyze()
elif pages == "⚖️  Compare Thresholds":
    from pages.subtle_vs_obvious import app
    app()
else:
    from pages.evaluation import app
    app()