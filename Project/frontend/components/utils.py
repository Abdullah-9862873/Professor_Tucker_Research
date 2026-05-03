"""Utility functions for Streamlit frontend components."""

import streamlit as st
from PIL import Image
import torch
import numpy as np
import io
import base64


def load_image(file) -> Image.Image:
    """Load image from uploaded file."""
    if file is None:
        return None
    return Image.open(file).convert('RGB')


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to PyTorch tensor."""
    img_array = np.array(img)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert PyTorch tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.cpu().detach()
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def validate_image(file) -> bool:
    """Validate uploaded image file."""
    if file is None:
        return False
    
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if file.type not in allowed_types:
        return False
    
    max_size_mb = 10
    if file.size > max_size_mb * 1024 * 1024:
        return False
    
    return True


def session_get(key, default=None):
    """Get session state with default."""
    return st.session_state.get(key, default)


def session_set(key, value):
    """Set session state value."""
    st.session_state[key] = value


def session_setdefault(key, default):
    """Set session state default if not exists."""
    return st.session_state.setdefault(key, default)


def display_error(message: str):
    """Display error message."""
    st.error(f"❌ {message}")


def display_success(message: str):
    """Display success message."""
    st.success(f"✅ {message}")


def display_warning(message: str):
    """Display warning message."""
    st.warning(f"⚠️ {message}")


def display_info(message: str):
    """Display info message."""
    st.info(f"ℹ️ {message}")


def download_results(results: dict, filename: str = "results.json"):
    """Create download button for results."""
    import json
    
    json_str = json.dumps(results, indent=2)
    st.download_button(
        label="📥 Download Results",
        data=json_str,
        file_name=filename,
        mime="application/json"
    )


def format_probabilities(probs: dict) -> str:
    """Format probabilities for display."""
    lines = []
    for label, prob in probs.items():
        bar = "█" * int(prob * 20)
        lines.append(f"{label:10} {prob:.2%} {bar}")
    return "\n".join(lines)


def create_progress_bar(current: int, total: int) -> None:
    """Create a progress bar."""
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.caption(f"Progress: {current}/{total} ({progress:.1%})")


def metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """Create a metric card."""
    st.metric(label=title, value=value, delta=delta, help=help_text)


def export_csv(data, filename: str = "data.csv") -> None:
    """Create CSV download from data."""
    import pandas as pd
    
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        st.error("Unsupported data format for CSV export")
        return
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Export CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def legend_html() -> str:
    """Return HTML for legend."""
    return """
    <div style="padding: 10px; background: #f0f0f0; border-radius: 5px;">
        <span style="color: green;">● Safe</span> &nbsp;
        <span style="color: #cccc00;">● Subtle</span> &nbsp;
        <span style="color: red;">● Obvious</span>
    </div>
    """


def display_image(img, caption: str = None, use_container_width: bool = True):
    """Display image in Streamlit."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    st.image(img, caption=caption, use_container_width=use_container_width)


def resize_image(img: Image.Image, size: tuple) -> Image.Image:
    """Resize image to specified size."""
    return img.resize(size, Image.LANCZOS)


def get_image_dimensions(img: Image.Image) -> tuple:
    """Get image dimensions (width, height)."""
    return img.size


def aspect_ratio_preserving_resize(img: Image.Image, max_size: int) -> Image.Image:
    """Resize image while preserving aspect ratio."""
    w, h = img.size
    if w > h:
        new_w = max_size
        new_h = int(h * (max_size / w))
    else:
        new_h = max_size
        new_w = int(w * (max_size / h))
    
    return img.resize((new_w, new_h), Image.LANCZOS)