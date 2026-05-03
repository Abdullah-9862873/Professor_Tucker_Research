import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import base64
import io

def plot_boundary_diagram():
    """Display a schematic diagram of safe, subtle, and obvious failure zones."""
    st.markdown("### Safety Boundary Diagram")
    st.markdown("""
    The graphic below shows three concentric safety zones around an object:
    - **Safe Zone (green)** – no violation  
    - **Subtle Failure Zone (yellow)** – near‑miss, model may misclassify  
    - **Obvious Failure Zone (red)** – clear boundary violation
    """)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    circle_obvious = plt.Circle((0.5, 0.5), 0.35, color='red', alpha=0.3, label='Obvious Failure')
    circle_subtle = plt.Circle((0.5, 0.5), 0.25, color='yellow', alpha=0.4, label='Subtle Failure')
    circle_safe = plt.Circle((0.5, 0.5), 0.15, color='green', alpha=0.5, label='Safe Zone')
    circle_center = plt.Circle((0.5, 0.5), 0.05, color='blue', alpha=0.8, label='Object')
    
    ax.add_patch(circle_obvious)
    ax.add_patch(circle_subtle)
    ax.add_patch(circle_safe)
    ax.add_patch(circle_center)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.set_title('Safety Boundary Classification Zones', fontsize=14, fontweight='bold')
    
    st.pyplot(fig)

def render_overview(image, result, thresholds):
    """Render the main detection overview with bounding boxes."""
    if image is None or result is None:
        return
    
    img_array = np.array(image)
    
    prediction = result.get('prediction', 'unknown')
    confidence = result.get('confidence', 0)
    probs = result.get('probabilities', {})
    
    color_map = {'safe': (0, 255, 0), 'subtle': (255, 255, 0), 'obvious': (255, 0, 0)}
    color = color_map.get(prediction, (128, 128, 128))
    
    cv2.putText(
        img_array,
        f"{prediction.upper()}: {confidence:.2%}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )
    
    y_offset = 60
    for cls, prob in probs.items():
        cv2.putText(
            img_array,
            f"{cls}: {prob:.2%}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        y_offset += 25
    
    st.image(img_array, caption=f"Detection: {prediction}", use_container_width=True)

def plot_failure_heatmaps(img, subtle_mask=None, obvious_mask=None):
    """Show side-by-side heatmaps overlayed on the image."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.image(img, caption="Original Image", use_column_width=True)
    with col2: 
        if subtle_mask is not None:
            st.image(subtle_mask, caption="Subtle Failure", use_column_width=True)
        else:
            st.info("No subtle mask available")
    with col3: 
        if obvious_mask is not None:
            st.image(obvious_mask, caption="Obvious Failure", use_column_width=True)
        else:
            st.info("No obvious mask available")

def plot_threshold_slider(threshold_a=0.3, threshold_b=0.7):
    """Demo of how thresholds affect classification."""
    st.markdown("### Threshold Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Threshold A:** {threshold_a}")
    with col2:
        st.markdown(f"**Threshold B:** {threshold_b}")
    
    st.markdown("""
    - **Subtle**: Predictions between threshold A and B
    - **Obvious**: Predictions above threshold B
    - **Safe**: Predictions below threshold A
    """)

def plot_metrics_table(metrics: dict):
    """Render a clean HTML table of performance numbers."""
    if not metrics:
        st.info("No metrics available")
        return
    
    html = """
    <style>
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
    }
    .metrics-table th {
        background-color: #0e639c;
        color: white;
        padding: 12px;
        text-align: left;
    }
    .metrics-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .metrics-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    </style>
    <table class="metrics-table">
    <tr><th>Metric</th><th>Value</th></tr>
    """
    
    for k, v in metrics.items():
        if isinstance(v, float):
            html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        else:
            html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

def plot_confusion_matrix(cm, class_names=None):
    """Display a confusion-matrix heatmap."""
    if cm is None:
        st.info("No confusion matrix available")
        return
    
    if class_names is None:
        class_names = ['Safe', 'Subtle', 'Obvious']
    
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def draw_boxes(img, boxes, colors):
    """Draw bounding boxes on image."""
    if isinstance(img, Image.Image):
        img = np.array(img).copy()
    else:
        img = img.copy()
    
    for box, color in zip(boxes, colors):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    return img

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Blend heatmap with image."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(heatmap, Image.Image):
        heatmap = np.array(heatmap)
    
    heatmap_colored = cv2.applyColorMap(
        cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    overlay = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    return overlay

def zoom_pan(img, zoom_level=1.0):
    """Simple zoom functionality for Streamlit."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom_level), int(w * zoom_level)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    if zoom_level > 1:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        resized = resized[start_h:start_h+h, start_w:start_w+w]
    elif zoom_level < 1:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        resized = result
    
    return resized

def image_to_base64(img):
    """Convert image to base64 for API transmission."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')