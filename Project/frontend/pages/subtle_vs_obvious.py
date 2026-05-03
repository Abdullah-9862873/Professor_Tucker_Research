"""Subtle vs Obvious — Threshold Comparison Page."""

import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

API_URL = "http://localhost:8000"

COLORS = {
    "safe":    "#10b981",
    "subtle":  "#f59e0b",
    "obvious": "#ef4444",
}


def app():
    st.markdown("""
    <div style="padding:10px 0 8px 0;">
        <h2 style="font-weight:700;margin:0;">⚖️ Compare Thresholds</h2>
        <p style="color:#94a3b8 !important;margin:4px 0 0 0;font-size:0.95rem;">
            Upload images with different threshold settings to see how classification boundaries shift.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Threshold controls ──
    st.markdown("### 🎚️ Threshold Configuration")
    t_left, t_right = st.columns(2, gap="medium")

    with t_left:
        st.markdown("""
        <div class="glass-card" style="padding:14px 18px;">
            <p style="color:#818cf8 !important;font-weight:600;margin:0 0 4px 0;">Scenario A</p>
        </div>
        """, unsafe_allow_html=True)
        threshold_a = st.slider("Subtle threshold (A)", 0.0, 1.0, 0.3, 0.05, key="th_a")
        uploaded_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"], key="img_a")

    with t_right:
        st.markdown("""
        <div class="glass-card" style="padding:14px 18px;">
            <p style="color:#a78bfa !important;font-weight:600;margin:0 0 4px 0;">Scenario B</p>
        </div>
        """, unsafe_allow_html=True)
        threshold_b = st.slider("Obvious threshold (B)", 0.0, 1.0, 0.7, 0.05, key="th_b")
        uploaded_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"], key="img_b")

    st.markdown("---")

    # ── Results ──
    result_a = _detect(uploaded_a) if uploaded_a else None
    result_b = _detect(uploaded_b) if uploaded_b else None

    res_left, res_right = st.columns(2, gap="large")

    with res_left:
        _render_result_card("A", threshold_a, uploaded_a, result_a, "#818cf8")

    with res_right:
        _render_result_card("B", threshold_b, uploaded_b, result_b, "#a78bfa")

    st.markdown("---")

    # ── Threshold visualisation ──
    st.markdown("### 📐 Threshold Visualisation")
    _plot_threshold_viz(threshold_a, threshold_b, result_a, result_b)

    # ── Explanation ──
    with st.expander("ℹ️  What do thresholds mean?"):
        st.markdown("""
        <div class="glass-card" style="padding:18px;">
            <ul style="padding-left:20px;">
                <li style="margin-bottom:8px;"><strong style="color:#10b981 !important;">Safe</strong> — model confidence that no boundary is violated (below Threshold A).</li>
                <li style="margin-bottom:8px;"><strong style="color:#f59e0b !important;">Subtle</strong> — confidence falls between Threshold A and Threshold B (a near-miss).</li>
                <li><strong style="color:#ef4444 !important;">Obvious</strong> — confidence exceeds Threshold B (clear violation).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ─── helpers ───────────────────────────────────────────────

def _detect(uploaded_file):
    """Call API."""
    try:
        resp = requests.post(
            f"{API_URL}/detect",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=30,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def _render_result_card(label, threshold, uploaded_file, result, accent):
    """Render a single scenario result card."""
    st.markdown(f"""
    <div class="glass-card-accent" style="border-color:{accent}30;padding:16px 20px;margin-bottom:10px;">
        <p style="font-weight:700;font-size:1.1rem;color:{accent} !important;margin:0;">Scenario {label}  ·  Threshold {threshold:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True, caption=f"Image {label}")

    if result:
        pred = result["prediction"]
        conf = result["confidence"]
        emoji = {"safe": "🟢", "subtle": "🟡", "obvious": "🔴"}.get(pred, "⚪")
        badge = f"badge-{pred}"

        st.markdown(f"""
        <div style="text-align:center;margin:12px 0;">
            <span class="{badge}" style="font-size:1rem;padding:8px 22px;">{emoji} {pred.upper()} — {conf:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

        # Mini probability bars
        for cls in ["safe", "subtle", "obvious"]:
            p = result["probabilities"].get(cls, 0)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="width:60px;font-size:0.8rem;color:#94a3b8 !important;">{cls.capitalize()}</span>
                <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:999px;height:6px;">
                    <div style="width:{p*100}%;height:100%;background:{COLORS[cls]};border-radius:999px;"></div>
                </div>
                <span style="font-size:0.8rem;color:{COLORS[cls]} !important;width:45px;text-align:right;">{p:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

    elif uploaded_file:
        st.warning("API did not return results.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:30px;color:#64748b !important;">
            <div style="font-size:2rem;margin-bottom:6px;">🖼️</div>
            <p>No image uploaded yet.</p>
        </div>
        """, unsafe_allow_html=True)


def _plot_threshold_viz(th_a, th_b, res_a, res_b):
    """Horizontal bar chart showing where thresholds sit on the 0–1 scale."""
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="none")
    ax.set_facecolor("none")

    # Background gradient zones
    ax.axvspan(0, th_a, color="#10b98130", label="Safe zone")
    ax.axvspan(th_a, th_b, color="#f59e0b30", label="Subtle zone")
    ax.axvspan(th_b, 1.0, color="#ef444430", label="Obvious zone")

    # Threshold lines
    ax.axvline(th_a, color="#f59e0b", linewidth=2, linestyle="--")
    ax.axvline(th_b, color="#ef4444", linewidth=2, linestyle="--")

    ax.text(th_a, 1.05, f"A={th_a:.2f}", ha="center", va="bottom", fontsize=10,
            color="#f59e0b", fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(th_b, 1.05, f"B={th_b:.2f}", ha="center", va="bottom", fontsize=10,
            color="#ef4444", fontweight="bold", transform=ax.get_xaxis_transform())

    # Plot result confidence markers
    y_pos = 0.5
    if res_a:
        conf_a = res_a["confidence"]
        ax.plot(conf_a, y_pos, "D", color="#818cf8", markersize=14, zorder=5)
        ax.annotate(f"A: {conf_a:.2f}", (conf_a, y_pos), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=9, color="#818cf8", fontweight="bold")
    if res_b:
        conf_b = res_b["confidence"]
        ax.plot(conf_b, y_pos, "D", color="#a78bfa", markersize=14, zorder=5)
        ax.annotate(f"B: {conf_b:.2f}", (conf_b, y_pos), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=9, color="#a78bfa", fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence Score", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.set_yticks([])

    legend = ax.legend(loc="upper right", frameon=True, facecolor="#1a1a2e",
                       edgecolor="#334155", labelcolor="white", fontsize=9)

    st.pyplot(fig, transparent=True)
    plt.close(fig)