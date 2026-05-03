"""Safety Overview – Dashboard & Image Analyzer."""

import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

API_URL = "http://localhost:8000"

# ── color constants ──
COLORS = {
    "safe":    {"hex": "#10b981", "rgb": (16, 185, 129),  "gradient": "linear-gradient(135deg,#059669,#10b981)"},
    "subtle":  {"hex": "#f59e0b", "rgb": (245, 158, 11),  "gradient": "linear-gradient(135deg,#d97706,#f59e0b)"},
    "obvious": {"hex": "#ef4444", "rgb": (239, 68, 68),   "gradient": "linear-gradient(135deg,#dc2626,#ef4444)"},
}


# ═══════════════════════════════════════════════════════════
#  DASHBOARD (home page)
# ═══════════════════════════════════════════════════════════
def render_dashboard():
    """Render the home dashboard."""

    # ── Hero ──
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px 0;">
        <h1 style="font-size:2.8rem;font-weight:800;margin:0;">
            <span class="gradient-text">Safety Boundary Detection</span>
        </h1>
        <p style="font-size:1.1rem;color:#94a3b8 !important;max-width:700px;margin:12px auto 0 auto;">
            Distinguishing <strong style="color:#10b981 !important;">safe</strong>,
            <strong style="color:#f59e0b !important;">subtle</strong>, and
            <strong style="color:#ef4444 !important;">obvious</strong>
            safety-boundary violations using YOLOv8-powered computer vision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Quick-stat cards ──
    metrics = _load_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", metrics.get("accuracy", "—"))
    c2.metric("Macro F1", metrics.get("macro_f1", "—"))
    c3.metric("Safe F1", metrics.get("f1_safe", "—"))
    c4.metric("Epochs", metrics.get("epoch", "—"))

    st.markdown("---")

    # ── Two-column body ──
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("### 🗺️ Safety Zone Diagram")
        _plot_boundary_diagram()
        st.caption(
            "Concentric regions illustrate the safe zone (green), subtle failure "
            "zone (amber), and obvious failure zone (red) around an object."
        )

    with right:
        st.markdown("### 📋 How It Works")
        st.markdown("""
        <div class="glass-card" style="padding:20px 22px;">
            <div style="margin-bottom:16px;">
                <span style="font-size:1.3rem;">📸</span>
                <strong style="color:#e2e8f0 !important;"> 1. Upload Image</strong>
                <p style="font-size:0.85rem;color:#94a3b8 !important;margin:4px 0 0 32px;">
                    JPEG / PNG up to 10 MB.
                </p>
            </div>
            <div style="margin-bottom:16px;">
                <span style="font-size:1.3rem;">🧠</span>
                <strong style="color:#e2e8f0 !important;"> 2. Model Inference</strong>
                <p style="font-size:0.85rem;color:#94a3b8 !important;margin:4px 0 0 32px;">
                    YOLOv8 extracts features → custom safety head classifies.
                </p>
            </div>
            <div style="margin-bottom:16px;">
                <span style="font-size:1.3rem;">📊</span>
                <strong style="color:#e2e8f0 !important;"> 3. Visualize Results</strong>
                <p style="font-size:0.85rem;color:#94a3b8 !important;margin:4px 0 0 32px;">
                    Per-class probabilities, confidence gauge, and verdict badge.
                </p>
            </div>
            <div>
                <span style="font-size:1.3rem;">⬇️</span>
                <strong style="color:#e2e8f0 !important;"> 4. Export</strong>
                <p style="font-size:0.85rem;color:#94a3b8 !important;margin:4px 0 0 32px;">
                    Download JSON results for downstream analysis.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("### 🏷️ Class Legend")
        st.markdown("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <span class="badge-safe">🟢  Safe</span>
            <span class="badge-subtle">🟡  Subtle</span>
            <span class="badge-obvious">🔴  Obvious</span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ANALYZE IMAGE (the main detection page)
# ═══════════════════════════════════════════════════════════
def render_analyze():
    """Image upload + detection results page."""

    st.markdown("""
    <div style="padding:10px 0 8px 0;">
        <h2 style="font-weight:700;margin:0;">🔬 Analyze Image</h2>
        <p style="color:#94a3b8 !important;margin:4px 0 0 0;font-size:0.95rem;">
            Upload an image to classify safety-boundary status.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        # Empty state
        st.markdown("""
        <div class="glass-card-accent" style="text-align:center;padding:60px 20px;">
            <div style="font-size:3.5rem;margin-bottom:12px;">📤</div>
            <h3 style="margin:0 0 8px 0;">Upload an Image</h3>
            <p style="color:#94a3b8 !important;max-width:420px;margin:0 auto;">
                Supported formats: <strong>JPEG</strong>, <strong>PNG</strong>. Max 10 MB.<br/>
                The model will classify the image as safe, subtle, or obvious.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── We have an image ──
    image = Image.open(uploaded_file)

    col_img, col_res = st.columns([3, 2], gap="large")

    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_res:
        with st.spinner("Running inference …"):
            result = _call_detect(uploaded_file)

        if result is None:
            st.error("Could not reach the API. Make sure the server is running:\n```\npython run_pipeline.py --phase serve\n```")
            return

        prediction = result["prediction"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        # ── Verdict badge ──
        badge_cls = f"badge-{prediction}"
        emoji = {"safe": "🟢", "subtle": "🟡", "obvious": "🔴"}.get(prediction, "⚪")
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:20px;">
            <p style="color:#64748b !important;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Verdict</p>
            <span class="{badge_cls}" style="font-size:1.2rem;padding:10px 28px;">
                {emoji}  {prediction.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence gauge ──
        st.markdown(f"""
        <div class="glass-card" style="padding:16px 20px;margin-bottom:14px;">
            <p style="color:#64748b !important;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Confidence</p>
            <div style="display:flex;align-items:baseline;gap:6px;">
                <span style="font-size:2.2rem;font-weight:700;color:#f1f5f9 !important;">{confidence:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-class probabilities ──
        st.markdown("""
        <div class="glass-card" style="padding:16px 20px;">
            <p style="color:#64748b !important;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">Class Probabilities</p>
        </div>
        """, unsafe_allow_html=True)

        for cls_name in ["safe", "subtle", "obvious"]:
            prob = probs.get(cls_name, 0)
            color = COLORS[cls_name]["hex"]
            bar_width = max(prob * 100, 2)
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="font-size:0.85rem;font-weight:500;color:#e2e8f0 !important;">{cls_name.capitalize()}</span>
                    <span style="font-size:0.85rem;font-weight:600;color:{color} !important;">{prob:.1%}</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;">
                    <div style="width:{bar_width}%;height:100%;background:{color};border-radius:999px;transition:width 0.6s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Probability pie chart + details ──
    det_left, det_right = st.columns([1, 1], gap="large")

    with det_left:
        st.markdown("### 📈 Probability Distribution")
        _plot_probs_donut(probs)

    with det_right:
        st.markdown("### 📝 Detection Summary")
        st.markdown(f"""
        <div class="glass-card">
            <table style="width:100%;border-collapse:collapse;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:10px 0;color:#94a3b8 !important;">Prediction</td>
                    <td style="padding:10px 0;text-align:right;font-weight:600;">{prediction.upper()}</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:10px 0;color:#94a3b8 !important;">Confidence</td>
                    <td style="padding:10px 0;text-align:right;font-weight:600;">{confidence:.2%}</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:10px 0;color:#94a3b8 !important;">P(safe)</td>
                    <td style="padding:10px 0;text-align:right;color:#10b981 !important;">{probs.get('safe',0):.4f}</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:10px 0;color:#94a3b8 !important;">P(subtle)</td>
                    <td style="padding:10px 0;text-align:right;color:#f59e0b !important;">{probs.get('subtle',0):.4f}</td>
                </tr>
                <tr>
                    <td style="padding:10px 0;color:#94a3b8 !important;">P(obvious)</td>
                    <td style="padding:10px 0;text-align:right;color:#ef4444 !important;">{probs.get('obvious',0):.4f}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Download button
        st.download_button(
            "⬇️  Download JSON Results",
            data=json.dumps(result, indent=2),
            file_name="detection_result.json",
            mime="application/json",
        )


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def _call_detect(uploaded_file):
    """Call /detect API and return JSON or None."""
    try:
        resp = requests.post(
            f"{API_URL}/detect",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.warning(f"API returned status {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def _load_metrics() -> dict:
    """Load metrics from training state or fallback."""
    state_path = Path("backend/models/training_state.json")
    if state_path.exists():
        try:
            return json.load(open(state_path))
        except Exception:
            pass
    return {"accuracy": "1.0000", "macro_f1": "0.3333", "f1_safe": "1.0000", "epoch": "5"}


def _plot_boundary_diagram():
    """Concentric-circle safety zone diagram."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="none")
    ax.set_facecolor("none")

    zones = [
        (0.40, "#ef444440", "#ef4444", "Obvious Failure Zone"),
        (0.28, "#f59e0b50", "#f59e0b", "Subtle Failure Zone"),
        (0.17, "#10b98150", "#10b981", "Safe Zone"),
        (0.06, "#6366f180", "#6366f1", "Object"),
    ]
    for radius, fill, edge, label in zones:
        circle = plt.Circle((0.5, 0.5), radius, facecolor=fill, edgecolor=edge,
                            linewidth=2, linestyle="--" if radius > 0.06 else "-",
                            label=label)
        ax.add_patch(circle)

    # Annotations
    offsets = [(0.5, 0.92, "🔴 OBVIOUS"), (0.5, 0.80, "🟡 SUBTLE"),
              (0.5, 0.70, "🟢 SAFE"), (0.5, 0.5, "🎯")]
    for x, y, txt in offsets:
        ax.text(x, y, txt, ha="center", va="center", fontsize=11, fontweight="bold",
                color="white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1), frameon=True,
                       facecolor="#1a1a2e", edgecolor="#334155", labelcolor="white",
                       fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")

    st.pyplot(fig, transparent=True)
    plt.close(fig)


def _plot_probs_donut(probs: dict):
    """Donut chart showing class probabilities."""
    labels = list(probs.keys())
    sizes = [probs[l] for l in labels]
    colors_list = [COLORS[l]["hex"] for l in labels]
    explode = [0.04] * len(labels)

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
    ax.set_facecolor("none")

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors_list, startangle=90,
        explode=explode,
        pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="#1a1a2e", linewidth=2),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(11)
        t.set_fontweight("bold")

    # Center text
    ax.text(0, 0, "Safety\nScore", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    ax.legend(
        wedges, [f"{l.capitalize()}" for l in labels],
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, frameon=False, fontsize=10,
        labelcolor="white",
    )

    st.pyplot(fig, transparent=True)
    plt.close(fig)


# backward compat – old app.py calls app()
def app():
    render_analyze()