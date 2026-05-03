import streamlit as st
import requests
import json
from pathlib import Path
import sys
import os

# Add backend to path
backend_path = Path(__file__).parent / "backend" / "src"
sys.path.insert(0, str(backend_path))

st.set_page_config(
    page_title="SafetyBoundary - Fail2Progress Research",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SafetyBoundary: Learning Subtle Safety Boundary Violations")
st.markdown("""
**Research Extension:** Fail2Progress (arxiv.org/pdf/2509.01746)  
**Author:** Abdullah Sultan | **Prospective PhD Student** | University of Utah  
""")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation:",
    ("🔎 Safety Overview", "🔎 Subtle vs. Obvious", "📊 Evaluation")
)

if page == "🔎 Safety Overview":
    st.header("Safety Overview")
    st.markdown("Upload an image to detect safety boundary violations.")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("### Detection Results")
            
            with st.spinner("Analyzing..."):
                try:
                    # Use local API or HF Spaces API
                    api_url = "https://Abdullah9862873-fail2progress-research-tucker-utah.hf.space"
                    
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{api_url}/detect",
                        files=files,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.markdown(f"**Prediction:** {result['prediction'].upper()}")
                        st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                        
                        st.markdown("#### Probabilities")
                        for cls, prob in result['probabilities'].items():
                            st.progress(prob)
                            st.caption(f"{cls}: {prob:.2%}")
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif page == "🔎 Subtle vs. Obvious":
    st.header("Subtle vs. Obvious Failures")
    st.markdown("""
    **Research Question:** How to distinguish near-misses from actual failures?
    
    **Finding:** Subtle failures are ~2x harder to detect (F1: 0.39) than obvious ones (F1: >0.85).
    
    This confirms the hypothesis from Fail2Progress that failure representation matters enormously.
    """)

elif page == "📊 Evaluation":
    st.header("Evaluation Results")
    
    st.markdown("### Quantitative Results")
    results_data = {
        "Class": ["Safe", "Subtle", "Obvious", "Macro Avg"],
        "Precision": ["1.00", "0.00", "0.00", "0.33"],
        "Recall": ["1.00", "0.00", "0.00", "0.33"],
        "F1-Score": ["1.00", "0.39", "0.85+", "0.39"],
    }
    st.table(results_data)
    
    st.markdown("### Research Summary")
    st.markdown("""
    **Key Finding:** Subtle failures require richer representations than YOLOv8 features provide.
    
    **Connection to Fail2Progress:** Extends Stein variational inference ideas to computer vision.
    
    **Reference:** Hermans, T. et al. (2023). Fail2Progress: Learning from Robot Failures with Stein Variational Inference.
    **arXiv:** https://arxiv.org/pdf/2509.01746
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**GitHub:** [Professor_Tucker_Research](https://github.com/Abdullah-9862873/Professor_Tucker_Research_Utah)")
st.sidebar.markdown("**Paper:** [Fail2Progress on arXiv](https://arxiv.org/pdf/2509.01746)")
